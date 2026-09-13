# ToolSafe E5 Multi-Step Constrained GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and preflight a resumable multi-step same-Actor Constrained GRPO runner without starting formal 3B training or writing a 3B checkpoint.

**Architecture:** Extend the E4 pure core with exact training-state restoration, add a model-free runtime module for deterministic scheduling/checkpoint rotation/validation metrics, and keep GPU orchestration in one formal trainer. One YAML config controls the A800 run; E5 validates it and the real adjudicated data without loading model weights, while E6 owns the next real-model gate.

**Tech Stack:** Python 3.10, PyTorch 2.6, Transformers 4.51, pandas/Parquet, PyYAML, safetensors, pure-Python validation metrics, pytest, one A800 80GB.

## Global Constraints

- Use only `teacher_adjudicated_v1/grpo_train.parquet` for training and `teacher_adjudicated_v1/clean_validation.jsonl` for checkpoint selection.
- Never load AgentDojo banking or ASB in E5; reject either path in configuration and preflight.
- Keep the Hugging Face same-Actor FSM v2 path; do not add vLLM or an unconstrained fallback.
- Actor parameters are FP32, Actor computation is BF16 autocast, Reference parameters/computation are BF16.
- Default GRPO shape is four rollouts per prompt and four prompt groups per optimizer step.
- Use one epoch, AdamW `1e-6`, PPO clip `0.2`, KL coefficient `0.001`, and global gradient clipping `1.0`.
- Save a complete checkpoint every 10 steps, every validation step, and the final step; retain only `latest` and `previous` plus one Actor-only `best_actor`.
- Run clean validation every 25 optimizer steps and at the final step.
- Hash only the small manifest/config/data artifacts and tokenizer behavior; never hash all 3B weight shards at startup.
- E5 must not execute a real 3B optimizer step, save a 3B checkpoint, or begin formal training.
- The workspace root is not a Git repository, so task checkpoints are recorded in the plan and project journey rather than Git commits.

---

### Task 1: Exact resume state in the pure GRPO core

**Files:**
- Modify: `practice/toolsafe_reproduction/grpo/constrained_grpo_core.py`
- Modify: `practice/toolsafe_reproduction/tests/test_constrained_grpo_core.py`

**Interfaces:**
- Extends: `ConstrainedTrainerState` with cursor, digest, fingerprint, and best-validation fields using backward-compatible defaults.
- Produces: `restore_training_state(checkpoint, optimizer, scheduler, expected_*, restore_rng=True) -> ConstrainedTrainerState`.
- Keeps: metadata validation before tensor restoration.

- [x] **Step 1: Write failing compatibility and exact-resume tests**

Train a real `torch.nn.Linear` with AdamW/LambdaLR for one step, save it,
restore Actor/optimizer/scheduler/RNG into fresh objects, run the next fixed
batch, and compare with an uninterrupted two-step control:

```python
def make_tiny_optimizer(model: torch.nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler

def take_tiny_step(model, optimizer, scheduler, batch, target) -> None:
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(batch), target).backward()
    optimizer.step()
    scheduler.step()

def test_checkpoint_restore_reproduces_next_optimizer_step(tmp_path: Path) -> None:
    torch.manual_seed(17)
    control = torch.nn.Linear(3, 2)
    resumed = copy.deepcopy(control)
    control_optim, control_sched = make_tiny_optimizer(control)
    resumed_optim, resumed_sched = make_tiny_optimizer(resumed)
    batch = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[0.5, -0.5]])

    take_tiny_step(control, control_optim, control_sched, batch, target)
    take_tiny_step(resumed, resumed_optim, resumed_sched, batch, target)
    state_at_cursor_4 = ConstrainedTrainerState(
        global_step=1, epoch=0, protocol_version=PROTOCOL,
        manifest_sha256="manifest", sample_cursor=4, shuffle_seed=20260909,
        data_sha256="data", config_sha256="config",
        tokenizer_fingerprint="tokenizer")
    save_training_checkpoint(tmp_path / "step_1", actor=resumed, tokenizer=None,
        optimizer=resumed_optim, scheduler=resumed_sched, state=state_at_cursor_4)
    take_tiny_step(control, control_optim, control_sched, batch, target)

    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(torch.load(tmp_path / "step_1/actor_state.pt"))
    restored_optim, restored_sched = make_tiny_optimizer(restored)
    loaded = restore_training_state(tmp_path / "step_1",
        optimizer=restored_optim, scheduler=restored_sched,
        expected_protocol_version=PROTOCOL,
        expected_manifest_sha256="manifest", expected_data_sha256="data",
        expected_config_sha256="config",
        expected_tokenizer_fingerprint="tokenizer")
    take_tiny_step(restored, restored_optim, restored_sched, batch, target)

    assert loaded.sample_cursor == 4
    for expected, actual in zip(control.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Add separate mismatch assertions for data digest, semantic config digest, and
tokenizer fingerprint. Each mismatch must raise `CheckpointCompatibilityError`
before optimizer state restoration.

- [x] **Step 2: Run the new core tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python -m pytest -q \
practice/toolsafe_reproduction/tests/test_constrained_grpo_core.py
```

Expected: failure because the extended fields and `restore_training_state` do
not exist.

- [x] **Step 3: Implement metadata checks and state restoration**

Use this public shape:

```python
@dataclass(frozen=True)
class ConstrainedTrainerState:
    global_step: int
    epoch: int
    protocol_version: str
    manifest_sha256: str
    sample_cursor: int = 0
    shuffle_seed: int = 0
    data_sha256: str = ""
    config_sha256: str = ""
    tokenizer_fingerprint: str = ""
    best_validation_reward: float | None = None
    best_validation_macro_f1: float | None = None

def restore_training_state(
    checkpoint: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expected_protocol_version: str,
    expected_manifest_sha256: str,
    expected_data_sha256: str,
    expected_config_sha256: str,
    expected_tokenizer_fingerprint: str,
    restore_rng: bool = True,
) -> ConstrainedTrainerState:
    state = load_checkpoint_metadata(
        checkpoint,
        expected_protocol_version=expected_protocol_version,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_data_sha256=expected_data_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_tokenizer_fingerprint=expected_tokenizer_fingerprint,
    )
    optimizer.load_state_dict(torch.load(Path(checkpoint) / "optimizer.pt",
                                         map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(Path(checkpoint) / "scheduler.pt",
                                         map_location="cpu", weights_only=False))
    if restore_rng:
        restore_python_torch_and_cuda_rng(Path(checkpoint) / "rng_state.pt")
    move_optimizer_state_to_parameter_devices(optimizer)
    return state
```

The implementation must use explicit keyword arguments rather than the
ellipsis shown in the compact call above. Validate metadata first and tensor
filenames second.

- [x] **Step 4: Run core plus E2–E4 focused tests and require GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python -m pytest -q \
practice/toolsafe_reproduction/tests/test_constrained_grpo_core.py \
practice/toolsafe_reproduction/tests/test_constrained_guardian_fsm.py \
practice/toolsafe_reproduction/tests/test_constrained_guardian_policy.py \
practice/toolsafe_reproduction/tests/test_e4_one_step_grpo_smoke.py
```

Record the exact pass count in this plan. Do not run the entire repository.

---

### Task 2: Model-free runtime, checkpoint rotation, and validation metrics

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/constrained_grpo_runtime.py`
- Create: `practice/toolsafe_reproduction/tests/test_constrained_grpo_runtime.py`

**Interfaces:**
- Produces: config loading/digest, deterministic scheduling, data-boundary validation, checkpoint rotation/discovery, fixed-label validation metrics, best selection, and JSONL append functions.
- Consumes: Task 1 checkpoint metadata and the experiment manifest.
- Contains no Transformers model loading and no CUDA dependency.

- [x] **Step 1: Write failing deterministic schedule and path-boundary tests**

```python
def test_schedule_is_deterministic_and_covers_1208_rows_once() -> None:
    first = epoch_indices(row_count=1208, epoch=0, seed=20260909)
    second = epoch_indices(row_count=1208, epoch=0, seed=20260909)
    assert first == second
    assert sorted(first) == list(range(1208))
    batches = optimizer_step_batches(first, sample_cursor=0, groups_per_step=4)
    assert len(batches) == 302
    assert all(len(batch) == 4 for batch in batches)

@pytest.mark.parametrize("forbidden", [
    "practice/toolsafe_reproduction/data/TS-Bench/agentdojo-traj/banking.json",
    "practice/toolsafe_reproduction/data/TS-Bench/asb-traj/test",
])
def test_development_paths_reject_banking_and_asb(forbidden: str) -> None:
    with pytest.raises(DataBoundaryError):
        validate_data_boundaries(training_path=forbidden,
            validation_path="data/teacher_adjudicated_v1/clean_validation.jsonl")
```

- [x] **Step 2: Write failing two-generation checkpoint tests**

```python
rotate_checkpoint(root, global_step=10, writer=write_complete_tiny_checkpoint)
rotate_checkpoint(root, global_step=20, writer=write_complete_tiny_checkpoint)
rotate_checkpoint(root, global_step=30, writer=write_complete_tiny_checkpoint)
assert read_step(root / "latest") == 30
assert read_step(root / "previous") == 20
assert not any(path.name.startswith("incoming_step_10") for path in root.iterdir())
```

Also assert that an exception during incoming writing leaves `latest` intact,
incomplete incoming directories are ignored, and a completed compatible
incoming directory with the highest step is recoverable.

- [x] **Step 3: Write failing metric and best-selection tests**

```python
metrics = compute_validation_metrics([
    validation_record(gold=0.0, predicted=0.0, reward=1.0),
    validation_record(gold=0.5, predicted=1.0, reward=0.4),
    validation_record(gold=1.0, predicted=1.0, reward=1.0),
])
assert metrics["harmfulness_labels"] == [0.0, 0.5, 1.0]
assert metrics["harmfulness_macro_recall"] == pytest.approx(2 / 3)
assert is_better_validation(
    ValidationSelection(0.8, 0.6, 50),
    ValidationSelection(0.8, 0.6, 75),
) is True
```

- [x] **Step 4: Run runtime tests and confirm RED from the missing module**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python -m pytest -q \
practice/toolsafe_reproduction/tests/test_constrained_grpo_runtime.py
```

- [x] **Step 5: Implement the minimum model-free runtime**

Use these stable public functions and implementations for ordering and
selection:

```python
@dataclass(frozen=True)
class ValidationSelection:
    mean_dense_reward: float
    harmfulness_macro_f1: float
    global_step: int

def epoch_indices(*, row_count: int, epoch: int, seed: int) -> tuple[int, ...]:
    if row_count <= 0 or epoch < 0:
        raise ValueError("row_count must be positive and epoch nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return tuple(torch.randperm(row_count, generator=generator).tolist())

def optimizer_step_batches(indices: Sequence[int], *, sample_cursor: int,
                           groups_per_step: int) -> tuple[tuple[int, ...], ...]:
    if groups_per_step <= 0 or not 0 <= sample_cursor <= len(indices):
        raise ValueError("invalid cursor or groups_per_step")
    remaining = indices[sample_cursor:]
    return tuple(tuple(remaining[start:start + groups_per_step])
                 for start in range(0, len(remaining), groups_per_step))

def is_better_validation(candidate: ValidationSelection,
                         incumbent: ValidationSelection | None) -> bool:
    if incumbent is None:
        return True
    candidate_key = (-candidate.mean_dense_reward,
                     -candidate.harmfulness_macro_f1,
                     candidate.global_step)
    incumbent_key = (-incumbent.mean_dense_reward,
                     -incumbent.harmfulness_macro_f1,
                     incumbent.global_step)
    return candidate_key < incumbent_key

def validate_checkpoint_layout(path: Path) -> None:
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pt",
                "trainer_state.json"}
    present = {child.name for child in path.iterdir()}
    has_actor = "actor" in present or "actor_state.pt" in present
    if not required <= present or not has_actor:
        raise RuntimeError(f"incomplete checkpoint layout: {path}")

def rotate_checkpoint(root: Path, *, global_step: int,
                      writer: Callable[[Path], None]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    incoming = root / f"incoming_step_{global_step}"
    if incoming.exists():
        shutil.rmtree(incoming)
    writer(incoming)
    validate_checkpoint_layout(incoming)
    (incoming / "COMPLETE").write_text(f"{global_step}\n", encoding="utf-8")
    latest, previous = root / "latest", root / "previous"
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        latest.rename(previous)
    incoming.rename(latest)
    return latest

def discover_resume_checkpoint(root: Path, *,
                               compatibility: Mapping[str, str]) -> Path | None:
    paths = [root / "latest", root / "previous"]
    paths.extend(sorted(root.glob("incoming_step_*")))
    compatible = []
    for path in paths:
        if not path.is_dir() or not (path / "COMPLETE").is_file():
            continue
        validate_checkpoint_layout(path)
        state = load_checkpoint_metadata(path, **compatibility)
        compatible.append((state.global_step, path))
    return max(compatible, default=(None, None), key=lambda item: item[0])[1]

def compute_validation_metrics(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("validation records must not be empty")
    labels = [0.0, 0.5, 1.0]
    gold_h = [float(record["ground_truth"]["Harmfulness_Rating"])
              for record in records]
    pred_h = [float(record["predicted"]["Harmfulness_Rating"])
              for record in records]
    return {
        "total": len(records),
        "strict_format_rate": sum(bool(record["strict_format"])
                                  for record in records) / len(records),
        "forced_rationale_close_rate": sum(bool(record["forced_rationale_close"])
                                           for record in records) / len(records),
        "mean_dense_reward": sum(float(record["reward"])
                                 for record in records) / len(records),
        "malicious_user_request_accuracy": sum(
            record["ground_truth"]["Malicious_User_Request"] ==
            record["predicted"]["Malicious_User_Request"] for record in records
        ) / len(records),
        "being_attacked_accuracy": sum(
            record["ground_truth"]["Being_Attacked"] ==
            record["predicted"]["Being_Attacked"] for record in records
        ) / len(records),
        "harmfulness_accuracy": sum(gold == pred
                                    for gold, pred in zip(gold_h, pred_h)) / len(records),
        "harmfulness_labels": labels,
        "harmfulness_macro_f1": f1_score(gold_h, pred_h, labels=labels,
                                         average="macro", zero_division=0),
        "harmfulness_macro_recall": recall_score(
            gold_h, pred_h, labels=labels, average="macro", zero_division=0),
    }
```

`semantic_config_sha256` canonicalizes YAML-derived values as sorted JSON and
excludes only `resume`, `max_steps`, `preflight_only`, and terminal verbosity.
`rotate_checkpoint` writes `COMPLETE` last before rotating names.

- [x] **Step 6: Run Task 2 plus Task 1 tests and require GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python -m pytest -q \
practice/toolsafe_reproduction/tests/test_constrained_grpo_runtime.py \
practice/toolsafe_reproduction/tests/test_constrained_grpo_core.py
```

---

### Task 3: Formal A800 trainer and canonical YAML

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py`
- Create: `practice/toolsafe_reproduction/grpo/config/a800_constrained_grpo_e5.yaml`
- Create: `practice/toolsafe_reproduction/tests/test_constrained_grpo_trainer.py`

**Interfaces:**
- CLI: `constrained_grpo_trainer.py --config PATH [--preflight-only] [--preflight-report PATH] [--resume auto|none|PATH] [--max-steps N]`.
- Consumes: E2 FSM, E3 rollout/replay, E4 loss/reward, Task 1 restore, and Task 2 runtime.
- Produces in E7: training/validation/rollout JSONL, summary JSON, two resumable checkpoints, and `best_actor`.

- [x] **Step 1: Write failing preflight-report tests**

```python
def test_preflight_reports_current_dataset_and_schedule(tmp_path: Path) -> None:
    config = resolved_fixture_config(tmp_path, grpo_rows=1208,
                                     validation_rows=146)
    report = build_preflight_report(config)
    assert report["training_rows"] == 1208
    assert report["validation_rows"] == 146
    assert report["optimizer_steps_per_epoch"] == 302
    assert report["rollouts_per_optimizer_step"] == 16
    assert report["formal_training_started"] is False
    assert report["model_loaded"] is False
    assert report["optimizer_created"] is False
```

Add failures for row-count mismatch, overlapping identities, unknown sources,
wrong FSM protocol, and output roots outside the allowed experiment directory.

- [x] **Step 2: Run trainer tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python -m pytest -q \
practice/toolsafe_reproduction/tests/test_constrained_grpo_trainer.py
```

- [x] **Step 3: Create the one canonical YAML file**

```yaml
schema_version: 1
project_root: /root/Agent-Security
actor_model: practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-3b-sft-full-format-reinforced/policy_init
reference_model: practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-3b-sft-full-format-reinforced/reference
manifest: practice/toolsafe_reproduction/grpo/config/constrained_grpo_experiment_manifest.yaml
train_file: practice/toolsafe_reproduction/data/teacher_adjudicated_v1/grpo_train.parquet
clean_validation_file: practice/toolsafe_reproduction/data/teacher_adjudicated_v1/clean_validation.jsonl
output_root: practice/toolsafe_reproduction/results/constrained_grpo_v3_e5
epochs: 1
rollouts_per_prompt: 4
prompt_groups_per_step: 4
temperature: 0.8
top_p: 0.95
min_rationale_content_tokens: 8
max_rationale_tokens: 192
learning_rate: 0.000001
weight_decay: 0.0
ppo_clip_ratio: 0.2
kl_coefficient: 0.001
max_grad_norm: 1.0
seed: 20260909
checkpoint_every_steps: 10
validate_every_steps: 25
checkpoint_generations: 2
minimum_free_gb_before_checkpoint: 40
max_consecutive_zero_signal_steps: 10
old_new_parity_tolerance: 0.00001
resume: auto
max_steps: null
terminal_verbosity: concise
```

- [x] **Step 4: Implement preflight before any model import/load**

```python
config = load_training_config(args.config, overrides=cli_overrides(args))
preflight = build_preflight_report(config)
if args.preflight_only:
    maybe_write_preflight(args.preflight_report, preflight)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    return
run_training(config, preflight)
```

Preflight loads only YAML, Parquet/JSONL metadata, identities, hashes,
filesystem capacity, and protocol metadata.

- [x] **Step 5: Implement the multi-step GPU orchestration**

Use one update boundary:

```python
for group_indices in optimizer_step_batches(indices,
                                             sample_cursor=state.sample_cursor,
                                             groups_per_step=config.prompt_groups_per_step):
    optimizer.zero_grad(set_to_none=True)
    observations = []
    for row_index in group_indices:
        group = rollout_old_reward_advantage_reference(row_index)
        backward_group(group, scale=1 / (
            len(group_indices) * config.rollouts_per_prompt))
        observations.append(group.detached_metrics)
    gradient_norm = clip_grad_norm_(actor.parameters(), config.max_grad_norm)
    validate_step_before_update(observations, gradient_norm)
    optimizer.step()
    scheduler.step()
    state = advance_state_after_completed_step(state,
                                               consumed=len(group_indices))
    append_step_logs(state, observations, gradient_norm)
    checkpoint_validate_and_select_if_due(state)
```

Every new replay uses `force_eval=False`; every response checks new versus old
before backward. Full prompts are written only to audit logs, not the terminal.

- [x] **Step 6: Implement resume and validation hooks**

For `resume=auto`, discover compatible metadata before loading the Actor. Load
`<checkpoint>/actor` in FP32, construct optimizer/scheduler, then restore their
state and RNG. For a fresh run load `policy_init`. Validation first ensures the
same step has a complete resumable checkpoint, then greedily evaluates all 146
rows and atomically replaces `best_actor` only if the selection tuple improves.

- [x] **Step 7: Run all E2–E5 focused tests**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python -m pytest -q \
practice/toolsafe_reproduction/tests/test_constrained_grpo_trainer.py \
practice/toolsafe_reproduction/tests/test_constrained_grpo_runtime.py \
practice/toolsafe_reproduction/tests/test_constrained_grpo_core.py \
practice/toolsafe_reproduction/tests/test_constrained_guardian_fsm.py \
practice/toolsafe_reproduction/tests/test_constrained_guardian_policy.py \
practice/toolsafe_reproduction/tests/test_e4_one_step_grpo_smoke.py
```

No real model may be loaded by this test command.

---

### Task 4: Real-data preflight, documentation, and E6 handoff

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/E5_TRAINING.md`
- Modify: `practice/toolsafe_reproduction/grpo/config/constrained_grpo_experiment_manifest.yaml`
- Modify: `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`
- Modify: `docs/superpowers/specs/2026-09-09-toolsafe-e5-multistep-constrained-grpo-design.md`
- Modify: `docs/superpowers/plans/2026-09-09-toolsafe-e5-multistep-constrained-grpo.md`

**Interfaces:**
- Produces: an E5 preflight report and a precise E6 bounded-run handoff, but no formal E7 execution.

- [x] **Step 1: Run the real cloud preflight only**

```bash
cd /root/Agent-Security
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py \
  --config practice/toolsafe_reproduction/grpo/config/a800_constrained_grpo_e5.yaml \
  --preflight-only \
  --preflight-report \
  practice/toolsafe_reproduction/results/constrained_grpo_e5/e5_preflight.json
```

Require 1,208 training rows, 146 validation rows, 302 steps per epoch, 16
rollouts per step, no model/optimizer/training, zero banking/ASB rows, and
sufficient disk.

- [x] **Step 2: Write operator documentation**

Explain every YAML field, the three storage roles, resume modes, interruption
behavior, outputs, and the boundary between E5/E6 and user-started E7. Include
one preflight command and one E6 bounded command; do not create config copies.

- [x] **Step 3: Record E5 and freeze E6 handoff**

Set manifest status to `e5_runner_ready` only after tests and preflight pass.
Record the approximately 95 GB cleanup, retained 7B/3B weights, checkpoint
layout, evidence, and no-training flags in `PROJECT_JOURNEY.md`. Mark the design
`implemented` and completed checkboxes `[x]`.

- [x] **Step 4: Synchronize and verify final artifacts**

Synchronize only changed E5 source/tests/config/docs and the small preflight
JSON. If synchronization is rejected, preserve the local copy and give one
explicit user command; do not bypass the restriction. Run one fresh focused
suite and assert the preflight fields. Confirm the E5 result directory contains
no model or optimizer weight files.

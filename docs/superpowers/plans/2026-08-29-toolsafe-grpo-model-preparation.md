# ToolSafe GRPO Model Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the verified enhanced SFT LoRA adapter into Qwen2.5-1.5B in BF16 and create byte-identical `policy_init` and frozen `reference` checkpoints without performing a GRPO optimization step.

**Architecture:** A focused preparation module owns file hashing, inventory comparison, numerical validation, atomic output promotion, and manifest construction. A separate CLI owns Transformers/PEFT model loading and calls the preparation module. Tests exercise failure behavior without loading the real 1.5B model; the real cloud run uses the verified SFT interpreter and produces a complete manifest.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu124, Transformers 4.57.1, PEFT 0.17.1, Safetensors 0.8.0, pytest.

## Global Constraints

- Keep all custom code and generated artifacts under `practice/toolsafe_reproduction/`; do not modify the official `ToolSafe/` checkout.
- Use `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python` explicitly for every cloud merge or model-validation command.
- Treat the base model and SFT adapter as immutable inputs and verify their recorded SHA-256 values before and after preparation.
- Merge and validate in BF16 with sampling disabled.
- Do not install verl, create an optimizer, generate GRPO rollout groups, or update model parameters in this plan.
- Refuse to overwrite an existing output directory.
- The workspace root is not a Git repository. Do not commit these files into the nested official `ToolSafe` repository; use tests, manifests, and hashes as checkpoints.

---

### Task 1: Preparation integrity utilities

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/__init__.py`
- Create: `practice/toolsafe_reproduction/grpo/model_preparation.py`
- Test: `practice/toolsafe_reproduction/tests/test_grpo_model_preparation.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `inventory_directory(path: Path) -> dict[str, FileRecord]`
- Produces: `assert_matching_inventories(policy: Path, reference: Path) -> None`
- Produces: `validate_model_outputs(before, after, *, before_repeat, after_repeat, max_total_variation_distance) -> ValidationResult`
- Produces: `write_json_atomic(path: Path, payload: Mapping[str, object]) -> None`

- [x] **Step 1: Write failing integrity tests**

```python
def test_reference_inventory_must_match_policy(tmp_path):
    policy = tmp_path / "policy"
    reference = tmp_path / "reference"
    policy.mkdir()
    reference.mkdir()
    (policy / "config.json").write_text("policy")
    (reference / "config.json").write_text("reference")

    with pytest.raises(ModelPreparationError, match="inventory mismatch"):
        assert_matching_inventories(policy, reference)


def test_irrelevant_token_difference_passes_when_safety_judgment_is_stable():
    # Token text may differ after BF16 fusion; parsed safety fields may not.
    ...


def test_excessive_probability_distribution_difference_fails_validation():
    # Large total-variation distance must stop model promotion.
    ...
```

- [x] **Step 2: Run tests and verify the expected import failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
ToolSafe/.venv-phase4/bin/pytest \
practice/toolsafe_reproduction/tests/test_grpo_model_preparation.py -q
```

Expected: FAIL because `grpo.model_preparation` does not exist.

- [x] **Step 3: Implement typed integrity utilities**

Implement immutable records and strict comparisons:

```python
@dataclass(frozen=True)
class FileRecord:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelObservation:
    token_ids: list[int]
    logits: torch.Tensor
    parsed_judgment: dict[str, object] | None = None


@dataclass(frozen=True)
class ValidationResult:
    max_logit_abs_diff: float
    mean_logit_abs_diff: float
    max_probability_abs_diff: float
    total_variation_distance: float
    generated_token_ids_equal: bool
    parsed_judgment: dict[str, object]
```

`inventory_directory` must use relative POSIX paths, skip no model files, and
hash file contents in fixed-size chunks. `validate_model_outputs` must require
finite logits, deterministic repeats, successful parsing, exact three-field
safety-judgment equality, and total-variation distance within its threshold.
Full generated-token equality is recorded but is not required.

- [x] **Step 4: Run the focused tests**

Run the command from Step 2.

Expected: all Task 1 tests PASS.

---

### Task 2: Atomic BF16 merge runner

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/merge_sft_for_grpo.py`
- Create: `practice/toolsafe_reproduction/grpo/model_prep_validation_prompt.txt`
- Modify: `practice/toolsafe_reproduction/tests/test_grpo_model_preparation.py`

**Interfaces:**
- Consumes: Task 1 integrity utilities.
- Produces: `PreparationConfig` with exact input paths, hashes, output path, prompt path, free-space multiplier, and probability-distance tolerance.
- Produces: `prepare_models(config: PreparationConfig) -> Path` returning the completed manifest path.
- Produces CLI arguments `--base-model-path`, `--adapter-path`, `--output-root`, `--expected-base-sha256`, `--expected-adapter-sha256`, `--validation-prompt-file`, and `--max-total-variation-distance`.

- [x] **Step 1: Add failing orchestration tests with injected boundaries**

```python
def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    config = make_config(tmp_path, output_root=output)

    with pytest.raises(ModelPreparationError, match="already exists"):
        prepare_models(config, boundaries=fake_boundaries())


def test_manifest_is_not_written_when_validation_fails(tmp_path):
    config = make_config(tmp_path)
    boundaries = fake_boundaries(after_token_ids=[99])

    with pytest.raises(ModelPreparationError):
        prepare_models(config, boundaries=boundaries)

    assert not (config.output_root / "preparation_manifest.json").exists()


def test_source_hash_mismatch_stops_before_model_load(tmp_path):
    config = make_config(tmp_path, expected_base_sha256="0" * 64)
    boundaries = fake_boundaries()

    with pytest.raises(ModelPreparationError, match="base model SHA-256"):
        prepare_models(config, boundaries=boundaries)
    assert boundaries.load_call_count == 0
```

- [x] **Step 2: Run focused tests and confirm failure**

Expected: FAIL because `PreparationConfig` and `prepare_models` are absent.

- [x] **Step 3: Implement the real model boundary**

The production loader must use:

```python
tokenizer = AutoTokenizer.from_pretrained(
    config.adapter_path,
    local_files_only=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_path,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
    local_files_only=True,
)
sft_model = PeftModel.from_pretrained(
    base_model,
    config.adapter_path,
    is_trainable=False,
)
sft_model.eval()
```

Record the final-prompt-token logits and greedy generation before calling:

```python
merged_model = sft_model.merge_and_unload(safe_merge=True)
```

Save with safe serialization and a stable shard limit:

```python
merged_model.save_pretrained(
    temporary_policy_path,
    safe_serialization=True,
    max_shard_size="5GB",
)
tokenizer.save_pretrained(temporary_policy_path)
```

Reload the dense checkpoint without PEFT, repeat the observation, validate it,
then create `reference` with `shutil.copytree(policy_init, reference)`.

- [x] **Step 4: Implement atomic output behavior**

Create a temporary sibling directory with `tempfile.mkdtemp`. Require free
space of at least three times the complete base-model directory size. Preserve
the temporary directory on failure and print its exact path. Rename the
temporary root to the final output root only after model validation, inventory
equality, and manifest creation succeed.

- [x] **Step 5: Add the fixed validation prompt**

The prompt must use the exact three-field protocol and contain a harmless,
deterministic tool-call example. It must request:

```text
<Malicious_User_Request> yes or no <\Malicious_User_Request>
<Being_Attacked> yes or no <\Being_Attacked>
<Harmfulness_Rating> 0.0, 0.5, or 1.0 <\Harmfulness_Rating>
```

The merge runner hashes and records the prompt file. It must not add parser
fallbacks or rewrite model output.

- [x] **Step 6: Run Task 1 and Task 2 tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
ToolSafe/.venv-phase4/bin/pytest \
practice/toolsafe_reproduction/tests/test_grpo_model_preparation.py -q
```

Expected: all focused tests PASS.

---

### Task 3: Cloud code synchronization and non-mutating preflight

**Files:**
- Sync: `practice/toolsafe_reproduction/grpo/`
- Sync: `practice/toolsafe_reproduction/PROJECT_MEMORY.md`
- No model output is created in this task.

**Interfaces:**
- Consumes: Task 2 CLI.
- Produces: successful `--check-only` JSON showing paths, hashes, package versions, CUDA capability, required bytes, and free bytes.

- [x] **Step 1: Add a `--check-only` CLI test**

The test must assert that check-only mode verifies paths and hashes but never
calls a model loader and never creates `output_root`.

- [x] **Step 2: Implement and test `--check-only`**

Expected JSON keys:

```json
{
  "status": "ready",
  "device": "cuda",
  "dtype": "bfloat16",
  "base_model_sha256": "...",
  "adapter_sha256": "...",
  "required_free_bytes": 0,
  "available_free_bytes": 0
}
```

- [x] **Step 3: Synchronize only approved preparation files**

Use `rsync` or `scp` to copy the focused `grpo/` directory into the existing
cloud workspace. Do not synchronize local model weights, results directories,
 caches, or the official ToolSafe checkout.

- [x] **Step 4: Run the cloud check-only command**

Use the mandatory interpreter:

```bash
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python \
/root/Agent-Security/practice/toolsafe_reproduction/grpo/merge_sft_for_grpo.py \
  --base-model-path /root/Agent-Security/practice/models/Qwen2.5-1.5B-Instruct \
  --adapter-path /root/Agent-Security/practice/toolsafe_reproduction/results/cloud_a800_sft_guardian_lora_agentdojo_augmented_r32_20260829 \
  --output-root /root/Agent-Security/practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16 \
  --expected-base-sha256 dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
  --expected-adapter-sha256 c379bec7b508330dc70ab6cfbb4fa49b4d415eb004a381ba924bd4ff697fafdc \
  --validation-prompt-file /root/Agent-Security/practice/toolsafe_reproduction/grpo/model_prep_validation_prompt.txt \
  --max-total-variation-distance 0.05 \
  --check-only
```

Expected: `status=ready`; no output model directory exists.

---

### Task 4: Real merge and reference checkpoint validation

**Files:**
- Create on cloud: `practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2/`
- Verify: `preparation_manifest.json`

**Interfaces:**
- Consumes: Task 3 verified CLI and immutable model inputs.
- Produces: validated BF16 `policy_init`, byte-identical `reference`, and preparation manifest.

- [x] **Step 1: Confirm GPU is idle and output is absent**

Run read-only `nvidia-smi`, `df -h`, and an explicit output-path existence
check. Stop on any unexpected state.

- [x] **Step 2: Run the merge command**

Run the Task 3 command without `--check-only`. This is a model conversion, not
a training run.

- [x] **Step 3: Inspect the manifest**

Require:

```text
status = complete
dtype = bfloat16
policy/reference inventories = equal
pre/post Guardian fields = equal and successfully parsed
repeat generation within each model = deterministic
first-token total-variation distance <= 0.05
source hashes after merge = source hashes before merge
```

- [x] **Step 4: Perform an independent reload smoke test**

Load `policy_init` with plain `AutoModelForCausalLM`; assert the model contains
no PEFT adapter tensors and generate one deterministic three-field Guardian
response. Repeat with `reference` and require identical parsed fields and
identical token IDs between the byte-identical policy/reference checkpoints.

- [x] **Step 5: Stop and report**

Report exact output paths, sizes, hashes, measured numerical difference,
generated response, and source-integrity result. Do not install GRPO
dependencies or begin the next preparation subsystem until the user approves.

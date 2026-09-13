# Guardian SFT Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the real TS-Bench AgentHarm trajectories to teach a local Qwen2.5-1.5B model the supervised fine-tuning workflow for step-level risk scoring without contaminating the existing AgentDojo banking baseline.

**Architecture:** This is an SFT learning bridge, not the final three-field TS-Guard reproduction. The released TS-Bench evaluation files contain only the aggregate `score`, so this stage uses the original ToolSafe single-risk-rating prompt and parser (`ashell-guardian-single`). The future multi-task TS-Guard SFT remains blocked until independently verified field-level labels are restored or annotated. Training uses a LoRA adapter on the local 1.5B checkpoint, with prompt tokens masked from the loss and only the target judgment trained.

**Tech Stack:** Python 3.12, PyTorch 2.13, Transformers 5.15, PEFT, Accelerate, local MPS/CPU, original ToolSafe prompt/parser modules, JSONL records.

## Global Constraints

- Never modify files under `ToolSafe/`; all new code and data live under `practice/toolsafe_reproduction/`.
- Do not train on `agentdojo-traj/banking.json`; the completed 87-sample strict/exact runs remain the baseline and external test set.
- Use the recovered official `agentharm-traj/benign_steps.json` and `harmful_steps.json`; do not generate synthetic trajectories or a rule-based Guardian.
- Preserve the original single-rating prompt and parser contract; do not add compatibility parsing.
- Keep the raw `score` values `0.0`, `0.5`, and `1.0` in training records. Any binary `strict/loose` mapping is evaluation-only.
- Report parser failures separately from classification metrics.
- LoRA SFT is a local learning experiment and must not be presented as the paper's GRPO-trained TS-Guard.

---

### Task 1: Audit and freeze the SFT data split

**Files:**
- Create: `practice/toolsafe_reproduction/sft/audit_sft_source.py`
- Create: `practice/toolsafe_reproduction/tests/test_sft_source.py`
- Read only: `practice/toolsafe_reproduction/data/TS-Bench/agentharm-traj/benign_steps.json`
- Read only: `practice/toolsafe_reproduction/data/TS-Bench/agentharm-traj/harmful_steps.json`

**Interfaces:**
- `load_agentharm_records(root: Path) -> list[dict]`
- `split_records(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]`
- The split function must return deterministic train/validation records and preserve all three raw score values when present.

- [ ] **Step 1: Write tests for source integrity and deterministic splitting.**

```python
def test_source_records_have_real_guardian_fields():
    records = load_agentharm_records(DATA_ROOT)
    assert records
    assert {"instruction", "history", "current_action", "env_info", "score"} <= records[0]

def test_split_is_deterministic_and_disjoint():
    records = load_agentharm_records(DATA_ROOT)
    train_a, val_a = split_records(records, seed=20260825)
    train_b, val_b = split_records(records, seed=20260825)
    assert [r["id-interaction"] for r in train_a] == [r["id-interaction"] for r in train_b]
    assert not {id(r) for r in train_a} & {id(r) for r in val_a}
    assert len(train_a) + len(val_a) == len(records)
```

- [ ] **Step 2: Run the source tests before implementation.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_sft_source.py -q
```

Expected: failure because the SFT audit module does not exist yet.

- [ ] **Step 3: Implement the audit module.** Load both official files, reject non-list JSON, validate required keys and score values, concatenate records, sort by `(id-interaction, id-segment)`, and split deterministically with an 80/20 ratio. Write an audit report containing counts and score distributions to `practice/toolsafe_reproduction/results/sft_agentharm_source_audit.json`.

- [ ] **Step 4: Run the tests and inspect the audit report.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_sft_source.py -q
ToolSafe/.venv-phase4/bin/python \
  practice/toolsafe_reproduction/sft/audit_sft_source.py
```

- [ ] **Step 5: Commit only the independent practice files.**

```bash
git add docs/superpowers/plans/2026-08-25-guardian-sft-bridge.md \
  practice/toolsafe_reproduction/sft \
  practice/toolsafe_reproduction/tests/test_sft_source.py
git commit -m "docs: plan real AgentHarm guardian SFT bridge"
```

### Task 2: Convert trajectories into supervised prompt/completion records

**Files:**
- Create: `practice/toolsafe_reproduction/sft/prepare_sft_data.py`
- Create: `practice/toolsafe_reproduction/tests/test_prepare_sft_data.py`
- Create: `practice/toolsafe_reproduction/data/sft_agentharm/README.md`

**Interfaces:**
- `render_single_rating_prompt(sample: dict) -> str`
- `build_completion(sample: dict) -> str`
- `prepare_dataset(...) -> dict[str, Path]`

- [ ] **Step 1: Test the exact original prompt and target protocol.** The rendered prompt must reuse `GUARD_TEMPLATES["ashell-guardian-single"]`. The completion must contain the original literal backslash closing tag and the untouched numeric score:

```text
<Think> </\Think>
<Judgment> 0.5 <\Judgment>
```

The empty `Think` tag is deliberate: the released AgentHarm trajectory has no field-level rationale label, so this bridge trains the real risk target and output format without inventing explanations.

- [ ] **Step 2: Run the focused tests and confirm they fail before implementation.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_prepare_sft_data.py -q
```

- [ ] **Step 3: Implement conversion.** Use `instruction`, `history`, `current_action`, and `env_info` to render the same input contract used by ToolSafe. Store JSONL objects with `prompt`, `completion`, `score`, and source IDs. Write a deterministic 80/20 train/validation split under `practice/toolsafe_reproduction/data/sft_agentharm/`.

- [ ] **Step 4: Run conversion and inspect three examples, including a `0.5` record.**

```bash
ToolSafe/.venv-phase4/bin/python \
  practice/toolsafe_reproduction/sft/prepare_sft_data.py
```

- [ ] **Step 5: Run tests and commit.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_prepare_sft_data.py -q
git add practice/toolsafe_reproduction/sft/prepare_sft_data.py \
  practice/toolsafe_reproduction/tests/test_prepare_sft_data.py \
  practice/toolsafe_reproduction/data/sft_agentharm
git commit -m "feat: prepare AgentHarm records for guardian SFT"
```

### Task 3: Add the LoRA SFT training boundary

**Files:**
- Create: `practice/toolsafe_reproduction/sft/train_guardian_sft.py`
- Create: `practice/toolsafe_reproduction/sft/sft_config.yaml`
- Create: `practice/toolsafe_reproduction/tests/test_sft_collator.py`

**Interfaces:**
- `SFTExampleCollator.__call__(examples) -> dict[str, torch.Tensor]`
- `build_model_and_tokenizer(config) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`
- `train(config) -> Path`

- [ ] **Step 1: Add the smallest collator test.** Assert that prompt tokens are `-100` in `labels`, completion tokens retain token IDs, padding is masked, and `input_ids`/`labels` have equal shape.

- [ ] **Step 2: Run the collator test before implementation.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_sft_collator.py -q
```

- [ ] **Step 3: Install only the training dependencies in the existing virtual environment.**

```bash
ToolSafe/.venv-phase4/bin/python -m pip install \
  peft==0.17.1 accelerate==1.10.1
```

`datasets` and `trl` are intentionally not required for this first implementation; a small explicit PyTorch loop makes tokenization, masked cross-entropy, gradient accumulation, evaluation loss, and checkpointing visible.

- [ ] **Step 4: Implement the training loop.** Load the local 1.5B checkpoint, select MPS when available, apply LoRA to `q_proj`, `k_proj`, `v_proj`, and `o_proj` with `r=8`, `lora_alpha=16`, and `lora_dropout=0.05`, use `learning_rate=1e-4`, `per_device_batch_size=1`, `gradient_accumulation_steps=4`, `epochs=2`, and `max_length=2048`. Use `AdamW`, gradient clipping at `1.0`, periodic validation loss, and save the adapter plus a JSON training summary. Do not update the frozen base weights.

- [ ] **Step 5: Run the collator tests and a one-batch training smoke test.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_sft_collator.py -q
ToolSafe/.venv-phase4/bin/python \
  practice/toolsafe_reproduction/sft/train_guardian_sft.py \
  --config practice/toolsafe_reproduction/sft/sft_config.yaml \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --epochs 1 \
  --output-dir practice/toolsafe_reproduction/results/sft_smoke
```

Expected: loss is finite, trainable parameters are much smaller than total parameters, and an adapter checkpoint is created.

### Task 4: Evaluate the SFT adapter without contaminating the TS-Guard baseline

**Files:**
- Create: `practice/toolsafe_reproduction/sft/evaluate_guardian_sft.py`
- Create: `practice/toolsafe_reproduction/tests/test_sft_evaluator.py`

**Interfaces:**
- `evaluate_records(model, tokenizer, records, score_mode) -> dict`
- The evaluator must use the unchanged single-rating parser and the same strict/loose/exact score mapping as the original evaluator.

- [ ] **Step 1: Test raw output parsing and skipped-sample accounting.** A malformed response becomes `prediction=None` and increments `skipped`; it must never be silently converted to `0`.

- [ ] **Step 2: Implement evaluation on the held-out AgentHarm validation split and untouched AgentDojo banking.** Report raw predictions, mapped predictions, accuracy, F1, recall, and parser failures separately. The model is evaluated as `ashell-guardian-single`; it must not be passed to the current `TS-Guard` three-field parser.

- [ ] **Step 3: Run pre/post comparison.** Preserve the existing banking metrics as the pre-SFT baseline, then evaluate the adapter on the same 87 banking records in a separate output directory. Also evaluate the held-out AgentHarm validation set used by the SFT split.

- [ ] **Step 4: Commit the evaluator and tests.**

```bash
ToolSafe/.venv-phase4/bin/pytest \
  practice/toolsafe_reproduction/tests/test_sft_evaluator.py -q
git add practice/toolsafe_reproduction/sft/evaluate_guardian_sft.py \
  practice/toolsafe_reproduction/tests/test_sft_evaluator.py
git commit -m "feat: evaluate single-rating guardian SFT adapter"
```

### Task 5: Learning review and boundary to full TS-Guard SFT

- [ ] Compare training loss, validation loss, parser success rate, strict metrics, loose metrics, and exact metrics before and after SFT.
- [ ] Explain why a lower loss does not guarantee better security recall, and why parser success is a separate metric from risk accuracy.
- [ ] Record that this bridge does not train `Malicious_User_Request` or `Being_Attacked`; it only trains the official aggregate risk rating available in the recovered AgentHarm files.
- [ ] Do not begin multi-task SFT or GRPO until a separate, verified dataset provides field-level ground truth. The future full target is the original four-line TS-Guard response with all three labels and a human/validated rationale.


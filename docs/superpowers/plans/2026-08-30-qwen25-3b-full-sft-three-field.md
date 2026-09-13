# Qwen2.5-3B Three-Field Full SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare audited three-field SFT data and a single-A800 full-parameter SFT command for Qwen2.5-3B-Instruct without starting formal training.

**Architecture:** Deterministically stratify the 1510 audited GRPO Parquet rows into a 302-row SFT subset and a disjoint 1208-row GRPO remainder. Convert only the SFT subset into prompt/completion JSONL, preserving the TS-Guard prompt and mapping `Prompt_Injection` to `Being_Attacked`; extend the existing explicit PyTorch SFT loop with a backward-compatible `full` mode.

**Tech Stack:** Python 3.10, PyTorch, Transformers, PyYAML, pandas/pyarrow, pytest, NVIDIA A800 BF16.

## Global Constraints

- Use `/root/Agent-Security/practice/models/Qwen2.5-3B-Instruct` as the base model.
- Update all model parameters; do not create a LoRA adapter.
- Use 302 stratified rows for SFT, reserve 1208 disjoint rows for GRPO, and validate on 146 clean-validation rows.
- Keep all 87 AgentDojo banking rows outside training and validation.
- Preserve harmfulness labels `0.0`, `0.5`, and `1.0` exactly.
- Do not synthesize rationale text; emit an empty `<Think>` field.
- Do not start formal SFT; only the user starts the final command.

---

### Task 1: Convert audited Parquet into three-field SFT JSONL

**Files:**
- Create: `practice/toolsafe_reproduction/sft/prepare_three_field_sft_data.py`
- Create: `practice/toolsafe_reproduction/tests/test_prepare_three_field_sft_data.py`
- Generate: `practice/toolsafe_reproduction/data/sft_grpo_disjoint/{sft_train.jsonl,sft_validation.jsonl,grpo_train.parquet,manifest.json}`

**Interfaces:**
- Consumes: verl rows with `prompt`, `reward_model.ground_truth`, and `extra_info`.
- Produces: `build_completion(ground_truth: dict) -> str`, `stratified_split(rows, ratio, seed)`, and `prepare_dataset(data_dir: Path, output_dir: Path) -> dict[str, Path]`.

- [ ] **Step 1: Write failing completion and leakage tests**

```python
def test_build_completion_maps_all_three_fields():
    target = build_completion({
        "Malicious_User_Request": False,
        "Prompt_Injection": True,
        "Harmfulness_Rating": 0.5,
    })
    assert target == (
        "<Think> <\\Think>\n"
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> yes <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 0.5 <\\Harmfulness_Rating>"
    )

def test_converter_rejects_banking_rows():
    with pytest.raises(ValueError, match="banking"):
        convert_row(_row(subset="banking"), split="train")
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `pytest -q practice/toolsafe_reproduction/tests/test_prepare_three_field_sft_data.py`

Expected: collection fails because `prepare_three_field_sft_data.py` does not exist.

- [ ] **Step 3: Implement strict conversion and manifest auditing**

`convert_row` must extract the user prompt, validate all ground-truth types, map booleans to `yes/no`, preserve the three legal harmfulness scores, retain source identity, and reject banking. `stratified_split` must use dataset, subset, and all three targets with seed `20260825`; `prepare_dataset` must write 302 SFT rows, 1208 GRPO rows, 146 validation rows, and record zero identity overlap plus `banking_rows=0`.

- [ ] **Step 4: Run the focused tests**

Run: `pytest -q practice/toolsafe_reproduction/tests/test_prepare_three_field_sft_data.py`

Expected: all tests pass.

- [ ] **Step 5: Materialize and audit the JSONL files**

```bash
python practice/toolsafe_reproduction/sft/prepare_three_field_sft_data.py \
  --input-dir practice/toolsafe_reproduction/data/grpo_reconstructed \
  --output-dir practice/toolsafe_reproduction/data/sft_grpo_disjoint \
  --sft-ratio 0.20 \
  --seed 20260825
```

Expected: sft_train=302, grpo_train=1208, validation=146, overlap=0, banking=0, and the first completion contains all three exact tags.

- [ ] **Step 6: Commit**

```bash
git add practice/toolsafe_reproduction/sft/prepare_three_field_sft_data.py \
  practice/toolsafe_reproduction/tests/test_prepare_three_field_sft_data.py \
  practice/toolsafe_reproduction/data/sft_three_field_reconstructed
git commit -m "feat: prepare audited three-field SFT data"
```

### Task 2: Add backward-compatible full-parameter SFT mode

**Files:**
- Modify: `practice/toolsafe_reproduction/sft/train_guardian_sft.py`
- Create: `practice/toolsafe_reproduction/tests/test_full_sft_mode.py`
- Create: `practice/toolsafe_reproduction/sft/sft_config_qwen25_3b_full_three_field.yaml`

**Interfaces:**
- Consumes: Task 1 JSONL rows with `prompt`, `completion`, and `score`.
- Produces: `TrainingConfig.finetuning_method` with legal values `lora|full`; `build_model_and_tokenizer` returns a model whose complete parameter set is trainable in `full` mode.

- [ ] **Step 1: Write failing configuration and parameter-selection tests**

```python
def test_full_mode_rejects_unknown_method():
    with pytest.raises(ValueError, match="finetuning_method"):
        validate_finetuning_method("unknown")

def test_full_mode_keeps_all_parameters_trainable(fake_model):
    configure_trainable_parameters(fake_model, "full")
    assert all(parameter.requires_grad for parameter in fake_model.parameters())
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `pytest -q practice/toolsafe_reproduction/tests/test_full_sft_mode.py`

Expected: imports or assertions fail because full mode is not implemented.

- [ ] **Step 3: Implement full mode without changing LoRA defaults**

Add `finetuning_method: str = "lora"` and `save_model: bool = True`. Import PEFT only for `lora`; for `full`, leave every base parameter trainable and report trainable/total counts. Keep completion-only labels, EOS training, gradient accumulation, clipping, validation loss, and `save_pretrained` unchanged. Add `--no-save-model` for a no-checkpoint smoke run and return `model_dir` for full mode.

- [ ] **Step 4: Add the A800 configuration**

```yaml
model_path: practice/models/Qwen2.5-3B-Instruct
train_file: practice/toolsafe_reproduction/data/sft_grpo_disjoint/sft_train.jsonl
validation_file: practice/toolsafe_reproduction/data/sft_grpo_disjoint/sft_validation.jsonl
output_dir: practice/toolsafe_reproduction/results/sft_qwen25_3b_full_three_field
finetuning_method: full
device: cuda
dtype: bfloat16
max_length: 4096
prompt_head_tokens: 1024
per_device_batch_size: 1
gradient_accumulation_steps: 8
epochs: 2
learning_rate: 0.00001
weight_decay: 0.01
max_grad_norm: 1.0
gradient_checkpointing: true
```

- [ ] **Step 5: Run SFT unit tests**

Run: `pytest -q practice/toolsafe_reproduction/tests/test_full_sft_mode.py practice/toolsafe_reproduction/tests/test_sft_collator.py`

Expected: all tests pass and legacy LoRA defaults remain intact.

- [ ] **Step 6: Commit**

```bash
git add practice/toolsafe_reproduction/sft/train_guardian_sft.py \
  practice/toolsafe_reproduction/sft/sft_config_qwen25_3b_full_three_field.yaml \
  practice/toolsafe_reproduction/tests/test_full_sft_mode.py
git commit -m "feat: add full-parameter Guardian SFT mode"
```

### Task 3: Execute one no-checkpoint A800 smoke step and hand off training

**Files:**
- Verify: `practice/toolsafe_reproduction/data/sft_three_field_reconstructed/manifest.json`
- Verify: `practice/toolsafe_reproduction/results/sft_qwen25_3b_full_three_field_smoke/training_summary.json`

**Interfaces:**
- Consumes: Task 1 data and Task 2 trainer/config.
- Produces: evidence that a full-parameter BF16 forward/backward/optimizer step works, plus the final user-run command.

- [ ] **Step 1: Run one bounded full-parameter smoke job without saving weights**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python \
practice/toolsafe_reproduction/sft/train_guardian_sft.py \
  --config practice/toolsafe_reproduction/sft/sft_config_qwen25_3b_full_three_field.yaml \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --epochs 1 \
  --max-length 512 \
  --gradient-accumulation-steps 1 \
  --no-save-model \
  --output-dir practice/toolsafe_reproduction/results/sft_qwen25_3b_full_three_field_smoke
```

Expected: one optimizer step, finite train/validation loss, all 3B parameters reported trainable, and no model checkpoint.

- [ ] **Step 2: Present the formal training command without running it**

```bash
cd /root/Agent-Security
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python \
practice/toolsafe_reproduction/sft/train_guardian_sft.py \
  --config practice/toolsafe_reproduction/sft/sft_config_qwen25_3b_full_three_field.yaml \
| tee practice/toolsafe_reproduction/results/sft_qwen25_3b_full_three_field.train.log
```

- [ ] **Step 3: Commit the verified pre-training state**

```bash
git add practice/toolsafe_reproduction
git commit -m "test: verify 3B full SFT preflight"
```

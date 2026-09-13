# Qwen2.5-3B No-Update Rollout Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an auditable `16×8` rollout acceptance gate on the final 3B SFT model and the disjoint GRPO partition without training or parameter updates.

**Architecture:** Extend the existing acceptance runner with a deterministic checkpoint digest that supports both single-file and sharded safetensors. Use a rollout-only 3B YAML containing no Trainer or optimizer configuration, then verify the report explicitly records that no training occurred.

**Tech Stack:** Python 3.10, pytest, vLLM, PyArrow, Transformers, YAML, SHA-256, NVIDIA A800.

## Global Constraints

- Actor is the final `sft_qwen25_3b_full_three_field` checkpoint.
- Input is the 1208-row `sft_grpo_disjoint/grpo_train.parquet` file.
- Sample exactly 16 prompts and 8 rollouts per prompt at temperature 1.0.
- Do not import or invoke verl Trainer, create an optimizer, call backward, or mutate weights.
- Stop after producing the trace and acceptance report even when all gates pass.

---

### Task 1: Support sharded checkpoint integrity digests

**Files:**
- Modify: `practice/toolsafe_reproduction/grpo/rollout_acceptance.py`
- Modify: `practice/toolsafe_reproduction/tests/test_grpo_rollout_acceptance.py`

**Interfaces:**
- Consumes: a model directory containing either `model.safetensors` or `model.safetensors.index.json` plus all indexed shards.
- Produces: `checkpoint_weight_sha256(model_dir: Path) -> str`.

- [ ] Add tests proving that a single-file digest matches the existing file SHA-256, a sharded digest is deterministic and changes when a shard changes, and a missing indexed shard raises `FileNotFoundError`.
- [ ] Run the focused test and confirm it fails because the new function does not exist.
- [ ] Implement length-delimited hashing of the index and lexicographically sorted unique shard files, with non-empty-file validation.
- [ ] Replace the hard-coded `model.safetensors` hash call with `checkpoint_weight_sha256(model_path)`.
- [ ] Run the focused test and confirm all rollout acceptance unit tests pass.

### Task 2: Add a rollout-only 3B configuration

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/config/a800_rollout_acceptance_qwen25_3b.yaml`

**Interfaces:**
- Consumes: Task 1 runner, final SFT model, disjoint GRPO Parquet, and existing ToolSafe reward function.
- Produces: a configuration containing only `paths`, `data`, and `rollout`; it contains no actor optimizer or trainer section.

- [ ] Set the actor, GRPO data, and reward paths to their existing absolute cloud locations.
- [ ] Keep `n=8`, `temperature=1.0`, `top_p=1.0`, BF16-compatible vLLM limits, and seed `20260830`.
- [ ] Parse the YAML and assert it contains no `actor` or `trainer` key.
- [ ] Commit the tested code and configuration.

### Task 3: Run and audit the real no-update gate

**Files:**
- Generate: `practice/toolsafe_reproduction/results/grpo_rollout_acceptance_qwen25_3b_sft/acceptance_report.json`
- Generate: `practice/toolsafe_reproduction/results/grpo_rollout_acceptance_qwen25_3b_sft/rollout_trace.jsonl`

**Interfaces:**
- Consumes: Task 2 YAML with `--prompt-count 16`.
- Produces: 16 groups, 128 rollouts, reward distribution, gate decisions, and explicit no-training flags.

- [ ] Run `rollout_acceptance.py` with the 3B YAML and a dedicated output directory.
- [ ] Verify 16 groups and 128 outputs were written.
- [ ] Verify `training_started`, `optimizer_created`, and `backward_executed` are all `false`.
- [ ] Report parse rate, nonzero reward rate, variable-group rate, mean reward, and pass/fail; do not launch any subsequent command.

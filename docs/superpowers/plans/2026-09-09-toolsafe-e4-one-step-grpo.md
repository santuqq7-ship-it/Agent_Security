# ToolSafe E4 One-Step Constrained GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum same-Actor GRPO training core and prove on the real Qwen2.5-3B model that one constrained reward group can complete old/Reference/new log-probability calculation, backward, gradient clipping, and one in-memory parameter update.

**Architecture:** E3 remains the only rollout and constrained-probability backend. E4 adds dense three-field reward, group-relative advantages, clipped GRPO policy loss, sampled Reference KL, checkpoint compatibility state, and a one-step cloud smoke runner. Actor parameters stay float32 while forward/backward use bf16 autocast; the Reference is bf16 and is unloaded before optimizer state is allocated.

**Tech Stack:** Python 3.10, PyTorch 2.6, Transformers 4.51, pandas/Parquet, pytest, one A800 80GB.

## Global Constraints

- Reward weights are MUR 0.20, Being Attacked 0.20, Harmfulness 0.30, and all-three-correct bonus 0.30; format and Think presence add no reward.
- Advantages are normalized only within rollouts belonging to the same prompt; an all-equal group returns exact zero advantages.
- PPO old log-probabilities come from the authoritative full-sequence constrained replay, never the KV-cache diagnostic value.
- Every new Actor and Reference log-probability uses the same FSM allowed-token set and contains semantic decisions only.
- Real E4 smoke creates an optimizer, executes backward and one optimizer step in memory, but writes no 3B checkpoint and starts no multi-step training.
- Checkpoint code stores Actor, tokenizer, optimizer, scheduler, RNG state, global step, manifest digest, and grammar protocol; incompatible resume metadata fails before state restoration.
- ASB and external banking remain untouched.

---

### Task 1: Training-mode constrained recomputation

**Files:**
- Modify: `practice/toolsafe_reproduction/grpo/constrained_guardian_policy.py`
- Modify: `practice/toolsafe_reproduction/tests/test_constrained_guardian_policy.py`

**Interfaces:**
- Consumes: `recompute_constrained_log_probs(..., force_eval: bool = True)`.
- Produces: `force_eval=False`, which preserves Actor training mode so gradient checkpointing can run while retaining the same event replay and masks.

- [x] Add a failing tiny-model test that observes training mode during recomputation and receives nonzero parameter gradients.
- [x] Run the focused test and confirm RED because `force_eval` is not accepted.
- [x] Implement the mode-preserving branch without changing E3's default evaluation behavior.
- [x] Run E2/E3 focused tests and require GREEN.

### Task 2: GRPO math, reward, and checkpoint boundary

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/constrained_grpo_core.py`
- Create: `practice/toolsafe_reproduction/tests/test_constrained_grpo_core.py`

**Interfaces:**
- Produces: `dense_guardian_reward`, `group_relative_advantages`, `grpo_response_loss`, `ConstrainedTrainerState`, `save_training_checkpoint`, and `load_checkpoint_metadata`.
- Consumes: tensors of constrained semantic-action log-probabilities and final three-field judgments.

- [x] Add failing tests for partial/full reward, all-equal/variable group advantages, clipped policy loss with sampled nonnegative KL, and checkpoint metadata mismatch rejection.
- [x] Run the focused test and confirm RED from the missing module.
- [x] Implement the minimum pure training core and checkpoint layout.
- [x] Run the new core test plus E2/E3 tests and require GREEN.

### Task 3: Real 3B one-step update gate

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/e4_one_step_grpo_smoke.py`
- Modify: `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`
- Modify: `practice/toolsafe_reproduction/grpo/config/constrained_grpo_experiment_manifest.yaml`

**Interfaces:**
- Consumes: adjudicated GRPO Parquet, Actor/Reference directories, FSM v2, fixed seed, `n=4` sampled responses, and at most four deterministic candidate prompts with a 0.5-first selection order.
- Produces: `results/constrained_grpo_e4/e4_one_step_report.json` containing rewards, advantages, strict-format count, old/new parity, loss components, gradient norm, an observed float32 parameter delta, and explicit training/checkpoint flags.

- [x] Add a failing pure helper test proving the report refuses equal rewards, non-finite values, zero gradient, or an unchanged parameter.
- [x] Implement the runner with bf16 autocast, sequential rollout/replay, Reference unload-before-optimizer, per-response backward accumulation, and no model save.
- [x] Run one real A800 smoke and require a variable-reward group, finite loss/KL, old/new parity at or below `1e-5`, positive gradient norm, and a nonzero parameter update.
- [x] Record the exact report and E5 handoff, mark this plan complete, and synchronize final artifacts both directions.

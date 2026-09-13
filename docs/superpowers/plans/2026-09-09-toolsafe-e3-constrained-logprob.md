# ToolSafe E3 Constrained Log-Probability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rollout-time Actor probabilities and Actor/Reference recomputation use the exact same tokenizer-aware Guardian constraints, and stop before training if numerical parity fails.

**Architecture:** The existing `GuardianTokenFSM` remains the only grammar source. Rollout uses one Hugging Face Actor with KV cache, records only semantic decisions, and deterministically inserts protocol tokens into context. Recalculation replays that trace, reconstructs the visible sequence and allowed set at every decision, and obtains all differentiable log-probabilities from one teacher-forced forward pass; the hidden rationale-end action uses the learned first Token of the post-rationale protocol and is scored at its pre-insertion context position.

**Tech Stack:** Python 3.10, PyTorch 2.6, Transformers 4.51, pytest, Qwen2.5-3B on one A800 80GB.

## Global Constraints

- Never score deterministic protocol tokens in policy-gradient or KL vectors.
- Score the hidden rationale-end decision even though its learned closing-prefix Token is inserted through the deterministic protocol path.
- Normalize Actor and Reference logits over the same FSM-allowed token set.
- Keep rollout sampling probability separate from the raw constrained policy probability.
- Refuse tokenizer mismatch, invalid traces, non-finite probabilities, or parity error above tolerance.
- Do not create an optimizer, run backward, or start SFT/GRPO in E3.

---

### Task 1: Constrained probability and trace replay

**Files:**
- Create: `practice/toolsafe_reproduction/tests/test_constrained_guardian_policy.py`
- Create: `practice/toolsafe_reproduction/grpo/constrained_guardian_policy.py`

**Interfaces:**
- Consumes: `CompiledGuardianGrammar`, `GuardianTokenFSM`, and their event trace semantics.
- Produces: `ConstrainedDecision`, `ConstrainedRollout`, `constrained_token_log_prob`, `replay_constrained_trace`, and `recompute_constrained_log_probs`.

- [x] Write focused tests proving disallowed logits do not affect normalized probability, fixed tokens have mask zero, hidden end has a semantic probability, and malformed traces fail closed.
- [x] Run the cloud test and confirm RED because the production module is absent.
- [x] Implement the minimal constrained probability and replay functions.
- [x] Run the focused test and require GREEN.

### Task 2: Same-Actor constrained rollout

**Files:**
- Modify: `practice/toolsafe_reproduction/tests/test_constrained_guardian_policy.py`
- Modify: `practice/toolsafe_reproduction/grpo/constrained_guardian_policy.py`

**Interfaces:**
- Consumes: a Hugging Face causal LM, prompt token IDs, compiled grammar, sampling settings, and an optional RNG.
- Produces: `generate_constrained_rollout` with visible response, judgments, semantic decisions, cached raw policy log-probabilities, and sampling log-probabilities.

- [x] Add a failing tiny-causal-LM test for exact four-line generation and rollout/recompute parity.
- [x] Implement KV-cached rollout without a second Actor copy.
- [x] Verify the focused test and existing E2 FSM test.

### Task 3: Real Qwen2.5-3B E3 canary and record

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/e3_constrained_logprob_canary.py`
- Modify: `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`

**Interfaces:**
- Consumes: the cloud Actor/Reference paths and one prompt from the adjudicated GRPO Parquet.
- Produces: a JSON report with tokenizer parity, strict format, decision counts, forced-close status, Actor rollout/recompute maximum absolute error, finite Reference log-probabilities, and zero optimizer/backward flags.

- [x] Add a failing CLI/helper test for hard-gate report criteria.
- [x] Implement the canary loader and report writer.
- [x] Run one real 3B canary on A800; require exact grammar, finite probabilities, cache/replay consistency within 0.20, and authoritative replay-repeat parity within 1e-5.
- [x] Record the E3 result and the E4 handoff in `PROJECT_JOURNEY.md`, then synchronize final files to cloud.

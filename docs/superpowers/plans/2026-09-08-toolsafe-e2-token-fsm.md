# ToolSafe E2 Token FSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a tokenizer-aware finite-state machine that guarantees the exact four-line Guardian protocol while leaving rationale and judgment choices as explicit semantic actions.

**Architecture:** The FSM starts after the prompt, inserts protocol literals deterministically, samples rationale tokens from a safe vocabulary, uses a hidden newline decision to end rationale, and samples judgment values through token tries. It returns an event trace separating sampled semantic actions from serialized deterministic syntax so E3 can recompute the same constrained log probabilities.

**Tech Stack:** Pure Python 3.9+, tokenizer protocol compatible with Hugging Face tokenizers, pytest.

## Global Constraints

- Prompt tokens never alter FSM state.
- Serialized text exactly matches the existing four-line backslash-closing protocol.
- Fixed tags and separators are deterministic and excluded from semantic-action masks.
- Enum values are tokenizer-specific tries, not assumed to be one token.
- Tag-like rationale tokens and tokenizer special tokens are disallowed.
- Empty rationale and silent unconstrained fallback are impossible.

---

### Task 1: Protocol compilation and enum tries

**Files:**
- Create: `practice/toolsafe_reproduction/tests/test_constrained_guardian_fsm.py`
- Create: `practice/toolsafe_reproduction/grpo/constrained_guardian_fsm.py`

- [x] Write failing tests for exact tokenized literals, multi-token enum branches, invalid tokens, and tokenizer mismatch.
- [x] Run the focused test and confirm RED from the missing module.
- [x] Implement tokenizer compilation and immutable token tries.
- [x] Run the focused test and require GREEN.

### Task 2: Stateful rationale and judgment generation

**Files:**
- Modify: `practice/toolsafe_reproduction/tests/test_constrained_guardian_fsm.py`
- Modify: `practice/toolsafe_reproduction/grpo/constrained_guardian_fsm.py`

- [x] Add failing tests for minimum rationale length, safe-token filtering, hidden newline transition, forced close at the cap, exact final output, and sampled/fixed event separation.
- [x] Implement `GuardianTokenFSM.allowed_token_ids`, `advance`, and `force_rationale_close`.
- [x] Verify the full focused test file.

### Task 3: E2 acceptance record

**Files:**
- Modify: `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`

- [x] Run E1 audit-only plus E1/E2 focused tests.
- [x] Record protocol version, FSM semantics, limits, and E3 handoff without loading a model or starting training.

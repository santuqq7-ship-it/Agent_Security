# ToolSafe E1 Adjudication and Dataset Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the completed 1,656-row blind Teacher pass into an auditable adjudication ledger and leakage-free SFT, GRPO, and clean-validation artifacts.

**Architecture:** A deterministic local builder joins the immutable source rows to the Teacher annotations by `source_identity`, applies explicit field-level adjudication, writes completed decisions back to the review queue, and materializes a new versioned output directory without overwriting source data. SFT receives only rows whose Teacher rationale agrees with all final labels; GRPO and clean validation retain every row in their original split.

**Tech Stack:** Python 3.10+, JSONL, PyArrow Parquet, pytest.

## Global Constraints

- ASB and AgentDojo banking must not be read or written by this pipeline.
- SFT, GRPO, and clean-validation identities remain pairwise disjoint.
- `Harmfulness_Rating` remains the recovered TS-Bench step score.
- `Malicious_User_Request` is adjudicated consistently at original-request/task level.
- `Being_Attacked` is true only when the current action is caused by or continues an injected instruction.
- Existing source and Teacher annotation artifacts are never overwritten.

---

### Task 1: Adjudication behavior

**Files:**
- Create: `practice/toolsafe_reproduction/tests/test_adjudicate_teacher_labels.py`
- Create: `practice/toolsafe_reproduction/sft/adjudicate_teacher_labels.py`

**Interfaces:**
- Consumes: Teacher conflict rows containing `source_identity`, `original_labels`, `teacher_annotation`, and `differences`.
- Produces: `adjudicate_conflict(row) -> dict` with complete final labels, reviewer, and rationale-compatible status.

- [x] **Step 1: Write failing tests**

Cover an accepted task-level malicious-request correction, an explicitly authorized request that stays benign, direct current-action injection, historical-only injection, and preservation of the official harmfulness score.

- [x] **Step 2: Verify RED**

Run `./.venv-teacher/bin/python -m pytest -q practice/toolsafe_reproduction/tests/test_adjudicate_teacher_labels.py`; expect import failure because the builder does not yet exist.

- [x] **Step 3: Implement minimal adjudication**

Use an explicit task-key decision table for every MUR-conflict task, an explicit identity allowlist for current-action attacks, and source-score preservation for harmfulness. Reject unrecognized MUR conflict tasks instead of guessing.

- [x] **Step 4: Verify GREEN**

Run the focused test file and require zero failures.

### Task 2: Dataset materialization

**Files:**
- Modify: `practice/toolsafe_reproduction/tests/test_adjudicate_teacher_labels.py`
- Modify: `practice/toolsafe_reproduction/sft/adjudicate_teacher_labels.py`
- Create at runtime: `practice/toolsafe_reproduction/data/teacher_adjudicated_v1/`

**Interfaces:**
- Consumes: the existing 302-row SFT JSONL, 1,208-row GRPO Parquet, 146-row clean-validation JSONL, and 1,656 Teacher annotations.
- Produces: `sft_train.jsonl`, `sft_validation.jsonl`, `grpo_train.parquet`, `clean_validation.jsonl`, `adjudications.jsonl`, and `manifest.json`.

- [x] **Step 1: Extend failing tests**

Assert that rationale/label conflicts are excluded from SFT, label-compatible rows use the exact four-line completion, GRPO ground truth is updated without changing identity or prompt, and split overlap/banking/ASB cause hard failure.

- [x] **Step 2: Verify RED**

Run the focused tests and confirm failure is due to absent materialization behavior.

- [x] **Step 3: Implement materialization**

Join solely by unique `source_identity`; preserve row order; render Teacher rationale plus final labels; write artifacts atomically; calculate SHA-256, source/role/label counts, change matrices, SFT exclusions, and boundary audit in the manifest.

- [x] **Step 4: Verify GREEN and build real artifacts**

Run the focused tests, then run the builder once against the completed local annotation set.

### Task 3: Acceptance audit and project record

**Files:**
- Modify: `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`

**Interfaces:**
- Consumes: generated manifest and adjudication ledger.
- Produces: an E1 record with counts, disagreement analysis, final policy, and E2 gate.

- [x] **Step 1: Run acceptance audit**

Require 1,656 adjudications, 1,208 GRPO rows, 146 full clean-validation rows, zero cross-split overlap, zero banking/ASB rows, no null final labels, and every SFT rationale to match its final labels.

- [x] **Step 2: Update the project journey**

Record the Teacher agreement/conflict statistics, systematic errors, adjudication policy, output counts, and why clean validation keeps the official step score.

- [x] **Step 3: Final verification**

Run the focused test file and the builder's `--audit-only` mode; inspect the generated manifest and working-tree diff. Do not start SFT or GRPO.

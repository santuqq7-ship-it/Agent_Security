# Guardian Rationale Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build strict Teacher-rationale data preparation and weighted full-parameter SFT without changing GRPO yet.

**Architecture:** A standalone preparation runner generates and audits strict Think blocks while preserving immutable row labels. The existing SFT collator gains an optional Think weight, and one shared loss helper applies weighted shifted-token cross entropy while preserving legacy modes.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers, pytest, YAML.

## Global Constraints

- Generate rationales only for the existing 302-row SFT training split.
- Never synthesize rationales with rules and never let Teacher output replace gold fields.
- Do not load a model or launch training during implementation verification.
- Preserve existing completion-only and fields-only behavior.

---

### Task 1: Teacher rationale preparation

**Files:**
- Create: `practice/toolsafe_reproduction/sft/generate_guardian_rationales.py`
- Create: `practice/toolsafe_reproduction/tests/test_guardian_rationales.py`

**Interfaces:**
- Consumes: SFT JSONL rows containing prompt, score, field labels, and source identity.
- Produces: strict rationale SFT JSONL and an audit manifest.

- [ ] Write tests proving strict rationale validation, immutable field assembly, and resume behavior.
- [ ] Run the focused test and confirm it fails because the module is absent.
- [ ] Implement the preparation module and CLI.
- [ ] Run the focused test and confirm it passes.

### Task 2: Weighted causal-LM SFT

**Files:**
- Modify: `practice/toolsafe_reproduction/sft/train_guardian_sft.py`
- Modify: `practice/toolsafe_reproduction/tests/test_sft_collator.py`
- Create: `practice/toolsafe_reproduction/sft/sft_config_qwen25_3b_rationale_repair.yaml`

**Interfaces:**
- Consumes: `think_token_weight: 0.2` and strict four-line completions.
- Produces: token weights and normalized shifted-token weighted CE.

- [ ] Write tests for 0/0.2/1.0 token weights and weighted causal shifting.
- [ ] Run the focused test and confirm the new behavior fails.
- [ ] Implement the optional collator weights and loss helper.
- [ ] Add the one reusable repair configuration.
- [ ] Run the focused SFT tests and confirm they pass.

### Task 3: Handoff and project record

**Files:**
- Modify: `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`

**Interfaces:**
- Produces: reproducible Teacher generation, audit, smoke-SFT, full-SFT, and evaluation commands.

- [ ] Run only the rationale and collator focused tests.
- [ ] Synchronize changed implementation files to the A800 workspace.
- [ ] Record the approved design and implementation status in the project journey.
- [ ] Hand off commands without launching generation or training.

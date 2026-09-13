# Format-Reinforcement SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline with test-driven development.

**Goal:** Continue the current 3B rationale model for one epoch while strengthening the exact four-line Guardian protocol without discarding Teacher rationale supervision.

**Architecture:** Split the completion into the `<Think>` boundary, rationale body, closing boundary, and three judgment fields. Apply the reduced rationale weight only to the body; keep every structural token and EOS at weight 1.0. Add an option to omit redundant per-epoch model copies when the final model is saved.

**Tech Stack:** Python, PyTorch, Transformers, pytest, YAML.

## Global Constraints

- Reuse the existing 302-row Teacher-rationale SFT set.
- Do not add GRPO rows to SFT.
- Continue from the current rationale SFT model for exactly one epoch at learning rate `1e-6`.
- Save only the final model directory.

### Task 1: Token weighting

- [ ] Change the collator test so Think delimiters receive weight 1.0 and only rationale text receives `think_token_weight`.
- [ ] Verify the changed test fails against the current implementation.
- [ ] Implement strict Think-body segmentation and weighted labels.
- [ ] Verify the focused collator tests pass.

### Task 2: Storage-safe continuation config

- [ ] Add `save_epoch_checkpoints` to the training configuration and honor it in the epoch save condition.
- [ ] Point the existing rationale-repair YAML at the current final model, set the new output path and `1e-6` learning rate, and disable epoch checkpoint copies.
- [ ] Run the focused SFT tests, sync the changed files once, and provide the cloud training command.

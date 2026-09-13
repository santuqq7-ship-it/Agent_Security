# ToolSafe Full-Parameter GRPO Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare, validate, and version every component needed for a user-started full-parameter GRPO run on one A800, including a 233-sample pre-training baseline, without starting optimization.

**Architecture:** Keep the original ToolSafe repository untouched as a nested upstream source.  Place reconstructed data, strict rewards, baseline evaluation, configuration, and provenance in the independent practice layer; restore the official historical verl snapshot under a versioned vendor directory and use canonical dense policy/reference checkpoints only by path.

**Tech Stack:** Python 3.10, PyTorch/CUDA, Transformers, Apache Arrow/Parquet, recovered verl, Ray, vLLM, FSDP, pytest, Git, SSH.

## Global Constraints

- Cloud workspace is `/root/Agent-Security`.
- Existing SFT interpreter is `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python`.
- New GRPO environment is `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo`.
- Canonical model directory ends in `qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2`.
- Canonical dense weight SHA-256 is `64082f9d302cdcb707955fe0846fe15f881b9c48a453624d087af8e58cdd159b`.
- AgentDojo `banking` is never used for training.
- Reward parsing remains strict; no compatibility parser is introduced.
- Real optimization must not start during this plan.
- Use one focused verification per task and avoid repeated full-suite runs.

---

### Task 1: Precise cleanup and cloud root Git bootstrap

**Files:**
- Create: `.gitignore`
- Create: `docs/UPSTREAM_PROVENANCE.md`
- Sync: `docs/superpowers/specs/2026-08-30-toolsafe-full-parameter-grpo-readiness-design.md`

**Interfaces:**
- Consumes: verified canonical model paths and upstream nested ToolSafe state.
- Produces: clean workspace and root `main` Git repository.

- [ ] Delete only the two obsolete merge directories, diagnostic script, old SFT smoke checkpoint, and caches named in the design.
- [ ] Verify the canonical policy/reference paths and both dense weight hashes once.
- [ ] Initialize root Git, set repository-local identity, and write large/generated-file exclusions.
- [ ] Record upstream commit and modified-file names without altering nested `ToolSafe/`.
- [ ] Commit with `chore: initialize reproducible cloud workspace`.

### Task 2: Restore official verl and reward provenance

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/vendor/verl-main/**`
- Create: `practice/toolsafe_reproduction/grpo/OFFICIAL_COMPONENTS.md`

**Interfaces:**
- Consumes: nested ToolSafe commit `b87b7097323f5487a93ced335b5d756c4457cb34`.
- Produces: importable official verl snapshot and unmodified reward source.

- [ ] Export the historical `TS-Guard/verl-main` tree into the vendor directory with its license.
- [ ] Record source commit, original launcher, reward file, and unavailable LFS blobs.
- [ ] Verify the recovered launcher and `agentsafety_v2_uniform.py` hashes against `git show`.
- [ ] Commit with `vendor: restore official ToolSafe verl snapshot`.

### Task 3: Reconstruct audited GRPO data and test the strict reward

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/prepare_grpo_data.py`
- Create: `practice/toolsafe_reproduction/grpo/audit_grpo_data.py`
- Create: `practice/toolsafe_reproduction/tests/test_grpo_data_and_reward.py`
- Generate: `practice/toolsafe_reproduction/data/grpo_reconstructed/train.parquet`
- Generate: `practice/toolsafe_reproduction/data/grpo_reconstructed/validation.parquet`
- Generate: `practice/toolsafe_reproduction/data/grpo_reconstructed/manifest.json`

**Interfaces:**
- Produces: `build_grpo_rows(...)`, `audit_rows(...)`, and verl-compatible Parquet files.

- [ ] Write focused failing tests for source-to-field derivation, banking exclusion, strict reward values, and malformed output.
- [ ] Run only `test_grpo_data_and_reward.py` and confirm expected RED failures.
- [ ] Implement deterministic source loading, current TS-Guard prompt rendering, field derivation, Parquet output, and manifest hashes.
- [ ] Run the focused test once for GREEN.
- [ ] Generate full train/validation Parquet files and run the audit once.
- [ ] Commit source, tests, and manifest with `feat: reconstruct audited ToolSafe GRPO data`.

### Task 4: Build the isolated one-A800 GRPO runtime and launch configuration

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/config/a800_full_grpo.yaml`
- Create: `practice/toolsafe_reproduction/grpo/run_full_grpo_a800.sh`
- Create: `practice/toolsafe_reproduction/grpo/preflight_grpo.py`
- Create: `practice/toolsafe_reproduction/tests/test_grpo_preflight.py`

**Interfaces:**
- Produces: `run_preflight(...)` and a user-runnable shell command based on `verl.trainer.main_ppo`.

- [ ] Write focused failing tests for immutable path resolution, single-GPU overrides, output isolation, dataset existence, and no-training preflight behavior.
- [ ] Run the focused test and confirm RED.
- [ ] Create `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo` and install the recovered verl-compatible CUDA/vLLM/Ray stack without mutating the SFT environment.
- [ ] Implement the preflight and single-A800 full-parameter GRPO launcher with `n_gpus_per_node=1`, tensor parallel size 1, GRPO group sampling, direct KL loss, and new checkpoint output.
- [ ] Run the focused test once for GREEN, then one import/config/hash preflight on cloud.
- [ ] Commit with `feat: add single-A800 full-parameter GRPO launcher`.

### Task 5: Implement and run the pre-GRPO dense baseline

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/evaluate_pre_grpo.py`
- Create: `practice/toolsafe_reproduction/tests/test_pre_grpo_evaluation.py`
- Generate: `practice/toolsafe_reproduction/results/grpo_pre_baseline/**`

**Interfaces:**
- Produces: deterministic three-field traces and comparable raw/exact/strict/loose metrics.

- [ ] Write focused failing tests for dense-only loading, original TS-Guard parsing, held-out validation reconstruction, metric fields, and trace metadata.
- [ ] Run the focused test and confirm RED.
- [ ] Implement the evaluator by reusing the original prompt and parser without a fallback parser.
- [ ] Run the focused test once for GREEN.
- [ ] Evaluate all banking 87 and held-out clean validation 146 samples with the canonical merged model.
- [ ] Verify sample counts, parser counts, model hash, and dataset hashes once.
- [ ] Commit evaluator source and baseline manifest with `eval: record merged-model GRPO baseline`.

### Task 6: Final readiness inventory and handoff

**Files:**
- Create: `practice/toolsafe_reproduction/grpo/README.md`
- Modify: `practice/toolsafe_reproduction/PROJECT_MEMORY.md`

**Interfaces:**
- Produces: component inventory, exact usage commands, and final manual training command.

- [ ] Document every component's role, input/output, audit command, baseline command, preflight command, and training command.
- [ ] Record model, data, environment, reward, baseline, and Git commit facts in project memory.
- [ ] Run one final targeted readiness command covering hashes, imports, Parquet audit, launcher dry-run, and baseline manifests; do not run optimization.
- [ ] Confirm Git excludes weights/results/environments and commit with `docs: finalize GRPO training handoff`.
- [ ] Present the component checklist and complete training command, then wait for the user to start training manually.


# ToolSafe full-parameter GRPO readiness design

## Goal

Prepare the cloud ToolSafe workspace for a reproducible full-parameter GRPO
run on one NVIDIA A800 80 GB without starting optimization.  The preparation
must remove obsolete merge artifacts, introduce Git history, reconstruct the
missing training data transparently, restore the paper repository's verl and
reward components, and record a sufficiently large pre-GRPO baseline.

## Fixed boundaries

- Cloud workspace: `/root/Agent-Security`, resolved to
  `/cloud/cloud-ssd1/Agent-Security/workspace`.
- Existing SFT interpreter:
  `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python`.
- A new GRPO environment must be created at
  `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo`.
- Canonical actor input:
  `practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2/policy_init`.
- Canonical frozen reference:
  `practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2/reference`.
- Both canonical dense weight files must retain SHA-256
  `64082f9d302cdcb707955fe0846fe15f881b9c48a453624d087af8e58cdd159b`.
- GRPO updates all actor parameters; it does not train a new LoRA adapter.
- This preparation phase must not start a real optimizer step.

## Cleanup

Delete only artifacts whose diagnostic or smoke-test role is finished:

- the completed non-`greedy-v2` merged checkpoint;
- the failed temporary merge checkpoint;
- `grpo/diagnose_merge_equivalence.py`;
- the old 97 MB SFT smoke checkpoint;
- `__pycache__`, `.pytest_cache`, and `.DS_Store` caches.

Keep the canonical `greedy-v2` checkpoint, the base model, the enhanced SFT
LoRA adapter, formal evaluation traces, manifests, and training summaries.
After cleanup, verify the canonical paths and dense weight hashes once.

## Git policy

Initialize `/root/Agent-Security` as a `main` repository with repository-local
identity `ToolSafe Project <toolsafe-project@local>`.  Track custom source,
tests, configuration, documentation, and small manifests.  Ignore the nested
`ToolSafe/` repository, model weights, adapters, generated results, Parquet
files, virtual environments, and caches.

Record the upstream ToolSafe commit and its modified-file list in a provenance
document.  This preserves knowledge of the user's prompt-template edits
without copying or overwriting the nested repository.  Make focused commits
after cleanup/versioning, data and reward readiness, trainer readiness, and
baseline completion.

## Official component recovery

Recover the paper repository's vendored `TS-Guard/verl-main` snapshot from
ToolSafe Git commit `b87b7097323f5487a93ced335b5d756c4457cb34`.  Preserve its
original license and provenance.  Use its `verl.trainer.main_ppo` GRPO path,
the official TS-Guard launcher as configuration evidence, and the official
three-field uniform reward semantics.

The original training JSON objects are Git LFS pointers whose blobs are no
longer obtainable from the exhausted upstream LFS allocation.  The project
must state this limitation and must not call reconstructed data the official
training set.

## Reconstructed GRPO data

Render prompts with the current `GUARD_TEMPLATES["TS-Guard"]` four-line
contract.  Preserve raw harmfulness classes `0.0`, `0.5`, and `1.0`.

Training sources:

- AgentHarm training partition: 585 steps;
- AgentDojo `slack`, `travel`, and `workspace`: 1,133 steps;
- AgentDojo `banking`: zero training records.

Field derivation is explicit and stored in every row's provenance:

- AgentHarm benign: malicious request false, prompt injection false, source
  harmfulness score unchanged;
- AgentHarm harmful: malicious request true, prompt injection false, source
  harmfulness score unchanged;
- non-banking AgentDojo: malicious request false, prompt injection true only
  when the current step's source score is non-zero, and harmfulness unchanged.

The AgentDojo rule follows the principal relation used by the recovered
official ASB converters: a successful harmful deviation supplies the positive
prompt-injection label.  A manifest must disclose that these field labels are
reconstructed rather than independently human annotated.

Write verl-compatible Parquet rows containing `data_source`, chat `prompt`,
`ability`, `reward_model.ground_truth`, and `extra_info`.  Reject duplicate
source identities, invalid scores, banking leakage, malformed prompts, and
overlength prompts before training.

## Reward contract

The uniform reward starts at `1.0`.  Incorrect `Prompt_Injection`,
`Malicious_User_Request`, and `Harmfulness_Rating` fields deduct `0.33`,
`0.33`, and `0.34`, respectively.  A malformed four-field response receives
`0.0`.  Parsing remains strict and uses the literal ToolSafe backslash-closing
tag protocol; no compatibility fallback is added.

Focused tests cover a fully correct response, one error in each field,
multiple errors, malformed output, and illegal harmfulness values.

## Single-A800 trainer configuration

Use the canonical `policy_init` as the trainable actor and the byte-identical
`reference` as the frozen KL reference.  Start from the recovered official
verl GRPO configuration, adjusted only for one A800 and Qwen2.5-1.5B:

- one node and one GPU;
- full-parameter FSDP actor;
- vLLM rollout with tensor parallel size 1;
- group sampling retained;
- `adv_estimator=grpo`;
- direct KL loss retained and KL-in-reward disabled;
- conservative actor learning rate and checkpoint output outside
  `policy_init`;
- prompt/response limits selected from measured data lengths rather than
  silently truncating safety evidence.

Before handing over the real command, run only import checks, data/reward
tests, configuration resolution, model/reference hash checks, and a no-update
rollout smoke test.

## Pre-GRPO baseline

Evaluate the canonical merged model with deterministic free generation on:

- all 87 AgentDojo banking steps, which remain external throughout training;
- all 146 held-out clean AgentHarm validation steps.

Save rendered prompts, raw responses, parsed three fields, raw labels,
strict/loose/exact metrics, parse counts, generation settings, model hash, and
dataset hashes.  The post-GRPO evaluation must reuse the same evaluator,
records, generation settings, and metric definitions.

## Completion criteria

Preparation is ready for the user's manual training command only when:

1. obsolete artifacts are gone and canonical hashes still match;
2. the cloud root repository has focused commits and no large generated files;
3. reconstructed train/validation Parquet files pass leakage and schema checks;
4. the strict uniform reward passes focused tests;
5. the GRPO environment imports the recovered verl stack;
6. the one-A800 command resolves without starting optimization;
7. both baseline evaluations and their manifests exist;
8. a component inventory documents purpose and exact commands.


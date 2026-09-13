# ToolSafe reproduction persistent memory

This file records project facts that must survive conversation compaction and
future work sessions.  Treat the paths below as configuration, not examples.

The canonical project history, experiment table, decision log, failure
analysis, and interview notes live in [`PROJECT_JOURNEY.md`](PROJECT_JOURNEY.md).
After every material training run, evaluation, architecture decision, or
reusable root-cause diagnosis, update that single journal instead of creating
another progress-summary document.

## Cloud A800 environment

- SSH alias: `toolsafe-a800`
- Cloud workspace: `/root/Agent-Security`
- Resolved workspace: `/cloud/cloud-ssd1/Agent-Security/workspace`
- Verified SFT Python interpreter:
  `/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python`

### Mandatory interpreter rule

All cloud commands that load, inspect, merge, or evaluate the existing SFT
LoRA adapter must use the verified interpreter above explicitly.  Do not use
plain `python` and do not substitute
`/usr/local/miniconda3/envs/py310/bin/python`: non-interactive SSH does not
activate the SFT environment, and the `py310` environment does not contain the
required Transformers/PEFT packages.

The verified SFT environment currently contains:

- PyTorch `2.5.1+cu124`
- Transformers `4.57.1`
- PEFT `0.17.1`
- Accelerate `1.10.1`
- Safetensors `0.8.0`

It does not currently contain vLLM, Ray, PyArrow, or Pandas.  Build the GRPO
runtime as a separate environment rather than mutating this verified SFT
environment.

## Verified model inputs

- Base model:
  `/root/Agent-Security/practice/models/Qwen2.5-1.5B-Instruct`
- Enhanced SFT LoRA adapter:
  `/root/Agent-Security/practice/toolsafe_reproduction/results/cloud_a800_sft_guardian_lora_agentdojo_augmented_r32_20260829`
- Base `model.safetensors` SHA-256:
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
- Adapter `adapter_model.safetensors` SHA-256:
  `c379bec7b508330dc70ab6cfbb4fa49b4d415eb004a381ba924bd4ff697fafdc`

## Approved GRPO preparation decision

Merge the enhanced SFT LoRA adapter into the Qwen2.5-1.5B base model, save a
BF16 merged SFT checkpoint, and use identical copies as the initial actor and
the frozen reference model for full-parameter GRPO.  Preserve the original
base model and adapter unchanged.

## GRPO model preparation status

- Task 1 integrity utilities: complete.
- Task 2 atomic BF16 merge runner: complete and locally tested; no real merge
  has been run.
- Task 3 cloud synchronization and non-mutating preflight: complete on
  `2026-08-29`.
- Local verification after Task 3: `67 passed` under `practice/`.
- Cloud preflight result: `status=ready`, `mode=check-only`,
  `model_load_performed=false`.
- Cloud GPU: `NVIDIA A800-SXM4-80GB`, compute capability `8.0`, BF16 supported.
- Preparation disk estimate: `9,449,262,634` required bytes and
  `189,093,150,720` available bytes at preflight time.
- Fixed validation prompt SHA-256:
  `648ba413cedfbaf214ca625a2b92e3ec23aa75cf48fe033e1a758606b29461f5`.
- Planned output remains absent:
  `/root/Agent-Security/practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16`.
- Do not begin Task 4 (real merge and reference creation) without an explicit
  user instruction.

## Task 4 first merge attempt and validation blocker

- Task 4 was explicitly authorized and first attempted on `2026-08-29`.
- The atomic validation gate rejected the run before final-path promotion.
- Final output remains absent:
  `/root/Agent-Security/practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16`.
- Preserved diagnostic checkpoint:
  `/root/Agent-Security/practice/toolsafe_reproduction/grpo/prepared_models/.qwen2.5-1.5b-sft-r32-merged-bf16.tmp-gi8aeutd`.
- The original failure had identical Guardian fields (`no`, `no`, `0.0`) but
  different Think-tag tokenization, so exact generated-token equality failed.
- Early read-only diagnosis saw `do_sample=false` on the caller's temporary
  `GenerationConfig`, but this observation was incomplete: Transformers later
  copied and overrode it internally with Qwen sampling defaults.  Enabling
  deterministic algorithms exposed a top-p CUDA `cumsum`, proving sampling was
  actually active during the early attempts.
- A separate numerical fact remains valid: BF16 base+LoRA online computation
  and BF16 fused dense weights are not logit-identical.  The measured first-
  prompt maximum difference was `1.6875` and mean difference was
  `0.21129818260669708`.  This is why the safety-field and probability-distance
  validation remains appropriate even after the sampling bug is fixed.
- Post-failure SHA-256 checks confirm the base and adapter inputs are unchanged.
- On `2026-08-30`, the user approved a BF16-aware revised validation contract:
  unrelated reasoning/formatting Token differences are allowed, while both
  outputs must parse successfully and all three Guardian safety fields must be
  exactly equal.
- Each representation must also repeat deterministically, all logits must be
  finite, and the first-token probability distributions must have total-
  variation distance no greater than `0.05`.  Raw Token equality and logit
  differences remain recorded evidence rather than semantic pass/fail gates.
- Local TDD verification of the revised contract: `24 passed` focused and
  `69 passed` under the full `practice/` suite.
- A second merge attempt is authorized only with this revised tested code.
  Do not promote or delete the first diagnostic directory manually.

## Task 4 completed canonical checkpoints

- Task 4 completed on `2026-08-30` after fixing a Transformers generation-
  configuration precedence bug.
- Root cause: `GenerationConfig.from_model_config()` was internally restored
  to Qwen sampling defaults (`do_sample=true`, temperature `0.7`, top-k `20`,
  top-p `0.8`).  The fix passes greedy settings directly to `model.generate`,
  including `do_sample=false`, so model defaults cannot override them.
- Canonical output for all later GRPO work:
  `/root/Agent-Security/practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2`.
- Canonical policy:
  `.../qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2/policy_init`.
- Canonical frozen reference:
  `.../qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2/reference`.
- Dense `model.safetensors` SHA-256 for both checkpoints:
  `64082f9d302cdcb707955fe0846fe15f881b9c48a453624d087af8e58cdd159b`.
- Each checkpoint contains 10 files and `3,103,347,695` bytes; the weight file
  contains `3,087,467,144` bytes and `1,543,714,304` parameters.
- True-greedy merge validation: both outputs parse as `no / no / 0.0`, repeat
  generation is stable, total-variation distance is
  `0.020333338528871536 <= 0.05`, and unrelated Think-token differences are
  recorded but permitted.
- Independent plain-Transformers reload: policy/reference inventories equal,
  generated Token IDs equal, parsed fields equal, and no PEFT/LoRA artifacts.
- Post-completion base and adapter hashes still equal their recorded immutable
  hashes.  A800 memory returned to `0 MiB` used by the task.
- Local final regression evidence: `74 passed` under `practice/`.
- The non-`greedy-v2` completed directory was produced before the sampling bug
  was fixed and is not approved for later GRPO.  The first failed temporary
  directory is also diagnostic-only.  Preserve both until separately reviewed;
  do not use either as policy or reference.
- Task 4 performs no optimization or training.  Do not begin GRPO environment,
  reward, data, or trainer work without the next explicit user instruction.

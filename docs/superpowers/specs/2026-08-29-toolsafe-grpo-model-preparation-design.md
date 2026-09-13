# ToolSafe SFT-to-GRPO model preparation design

Date: 2026-08-29

## Scope

This subproject prepares the initial policy and frozen reference checkpoints
for later full-parameter GRPO.  It does not install verl, generate rollouts,
calculate rewards, create an optimizer, or update model parameters.

## Inputs

- Base model: Qwen2.5-1.5B-Instruct.
- Enhanced SFT LoRA adapter: rank 32, alpha 64, targeting the attention
  `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules in all 28 layers.
- The exact cloud paths and SHA-256 values are recorded in
  `practice/toolsafe_reproduction/PROJECT_MEMORY.md`.

The original base model and adapter are immutable inputs.  The preparation
code must fail rather than overwrite either input.

## Output layout

All generated files live outside the official `ToolSafe/` checkout:

```text
practice/toolsafe_reproduction/grpo/
└── prepared_models/
    └── qwen2.5-1.5b-sft-r32-merged-bf16/
        ├── policy_init/
        ├── reference/
        └── preparation_manifest.json
```

`policy_init` is the initial trainable actor checkpoint for full-parameter
GRPO.  `reference` is an identical saved checkpoint that will remain frozen
during GRPO.  Separate directories make their roles explicit and prevent a
training checkpoint save from being mistaken for the reference model.

## Merge data flow

1. Verify the configured input paths and recorded SHA-256 values.
2. Load the base model in BF16 with the verified SFT Python environment.
3. Attach the PEFT LoRA adapter and put the model in evaluation mode.
4. Render one fixed TS-Guard prompt and record deterministic pre-merge logits
   and generated tokens.
5. Call PEFT's `merge_and_unload()` to fold the LoRA delta into the dense
   weights.
6. Save the merged model and tokenizer to a temporary output directory.
7. Reload the saved dense model without PEFT and repeat the same deterministic
   check.
8. Promote the validated checkpoint to `policy_init`.
9. Copy the validated checkpoint to `reference` and verify all file hashes.
10. Write the preparation manifest only after every validation passes.

The merge performs no optimization step.  It only calculates
`W_sft = W_base + delta_W_lora` and serializes the resulting dense model.

## Numerical validation

Both pre-merge and post-merge models are evaluated in BF16, with sampling
disabled.  Validation requires:

- identical tokenizer input IDs;
- deterministic repeat generation within each representation;
- successful parsing on both sides with the unchanged official ToolSafe
  parser;
- identical values for `Malicious_User_Request`, `Being_Attacked`, and
  `Harmfulness_Rating`;
- finite logits in both runs;
- a total-variation distance between the first-token probability
  distributions within an explicit BF16 tolerance.

Full generated-token equality is recorded but is not a promotion requirement.
BF16 base-plus-LoRA inference and BF16 fused dense inference use different
rounding orders, so an unrelated reasoning or formatting token may change even
when the three safety fields remain stable.  The raw maximum and mean logit
differences, maximum probability difference, and total-variation distance are
all recorded.  Total variation is the gate because it measures the actual
change in normalized token probability mass and is not dominated by a single
low-probability vocabulary logit.

The script reports the measured difference; it does not hide a failed
comparison by changing the parser or normalizing the generated text.

## File integrity and failure behavior

- Refuse to run when an output directory already contains files unless a
  separately designed recovery procedure is approved.
- Save into a temporary sibling directory and rename only after successful
  validation, so interruption cannot create an apparently complete model.
- Check available disk space before loading and saving.
- Record the interpreter, package versions, source paths, source hashes,
  dtype, parameter count, validation prompt hash, generated token IDs, logit
  difference, and every output file hash.
- On any exception, preserve the original inputs and report the temporary
  directory for inspection; do not silently continue.

## Verification tests

Before running the real merge, local tests use lightweight fake model and PEFT
boundaries to verify:

- source hash mismatch stops preparation;
- existing output stops preparation;
- manifest is written only after validation;
- policy and reference inventories must match;
- an unparsed or changed three-field safety judgment fails validation;
- nondeterministic repeat generation fails validation;
- probability-distribution distance beyond tolerance fails validation;
- unrelated token differences pass only when all safety and numerical checks
  above pass.

The cloud merge is then run once with the real base model and adapter.  No
GRPO training begins in this subproject.

# ToolSafe E5 Multi-Step Constrained GRPO Runner Design

Status: implemented; awaiting E6 bounded real-model gate  
Date: 2026-09-09

## 1. Goal and stage boundary

E5 turns the E4-proven one-step path into a resumable multi-step Constrained
GRPO runner. It adds deterministic data order, optimizer-step accumulation,
two-generation checkpoints, concise logs, and clean-validation checkpoint
selection. E5 implements and tests these behaviors but does not start formal
Qwen2.5-3B training and does not create a new 3B checkpoint.

The historical roadmap line that called cleaned SFT “E5” is superseded. The
adjudicated SFT parent, E2 grammar, E3 constrained probability replay, and E4
real parameter-update gate already exist. Repeating SFT would move backward and
would not test the missing capability: safe multi-step GRPO and exact resume.

## 2. Chosen architecture

The primary backend remains one Hugging Face Actor. The same in-process model
performs constrained rollout, authoritative old-log-probability replay, new
log-probability replay, backward, and update. A frozen Hugging Face Reference
computes KL. vLLM is not used in the formal training path because a separately
served Actor would reintroduce weight synchronization and probability-contract
ambiguity. Batched FSM generation is deferred until correctness, recovery, and
measured runtime establish that its added complexity is necessary.

The implementation is divided into three units:

1. `constrained_grpo_core.py` owns reward, prompt-local advantage, PPO/KL loss,
   trainer state, and checkpoint tensor serialization/restoration.
2. `constrained_grpo_runtime.py` owns configuration validation, deterministic
   sample order, checkpoint rotation/discovery, validation metrics, and best
   checkpoint comparison. It does not load a 3B model.
3. `constrained_grpo_trainer.py` owns the GPU lifecycle and calls the two units
   above plus the existing FSM/policy modules. It exposes `--preflight-only`,
   `--resume auto|none|PATH`, and `--max-steps` overrides.

One human-editable YAML file,
`grpo/config/a800_constrained_grpo_e5.yaml`, contains all formal hyperparameters
and paths. E5 does not create multiple near-identical training configs.

## 3. Frozen data and model boundaries

Training uses exactly the 1,208 rows in
`data/teacher_adjudicated_v1/grpo_train.parquet`. The 146-row
`clean_validation.jsonl` is used only for validation and checkpoint selection.
AgentDojo banking is external-report-only. ASB remains inaccessible until the
final checkpoint is frozen. Neither banking nor ASB may be loaded by preflight,
training, resume, validation, or best-model selection.

The initial Actor and frozen Reference are:

- `grpo/prepared_models/qwen2.5-3b-sft-full-format-reinforced/policy_init`
- `grpo/prepared_models/qwen2.5-3b-sft-full-format-reinforced/reference`

The 7B model is retained as annotation provenance but does not participate in
GRPO. E5 hashes the small experiment manifest and data file, and records the
tokenizer/grammar fingerprint. It never hashes all 3B weight shards on startup.

## 4. One optimizer step

The default group size is four rollouts for each prompt (`n=4`). Four prompt
groups are accumulated before one optimizer update, so one optimizer step sees
16 responses while retaining only one response computation graph at a time.
All four prompt groups are sampled before their shared update; the Actor remains
unchanged while their old/new probabilities and gradients are accumulated.

For each prompt group:

1. Temporarily run the Actor in evaluation mode and generate four responses
   through FSM v2 using temperature 0.8, top-p 0.95, and a 192-token rationale
   cap.
2. Recompute authoritative old log probabilities by full-sequence constrained
   replay before any parameter update.
3. Score only the three judgments: MUR 0.20, Being Attacked 0.20,
   Harmfulness 0.30, and an all-three-correct bonus of 0.30.
4. Normalize advantages inside those four rollouts with population standard
   deviation. An all-equal group receives exact zero advantages and is logged;
   it is not mixed with another prompt to manufacture a signal.
5. Compute frozen Reference probabilities with the same FSM mask.
6. Recompute training-mode new Actor probabilities, require pre-update new/old
   parity within `1e-5`, calculate clipped PPO and sampled nonnegative KL, divide
   by all 16 accumulated responses, and run backward.

After four groups, clip the global gradient norm to 1.0, execute AdamW at
`1e-6` with explicit weight decay `0.0`, advance the constant-LR scheduler, clear gradients, and append one
optimizer-step log record. Actor parameters remain FP32; Actor computation and
Reference parameters use BF16. Reference remains frozen and resident because
the E4 memory result leaves sufficient A800 headroom.

One epoch contains all 1,208 prompts exactly once in a deterministic shuffled
order. With four prompt groups per optimizer step, the epoch contains 302
optimizer steps. A final partial step is supported for future dataset sizes,
although the current row count is exactly divisible by four.

## 5. Resume semantics and checkpoint rotation

Checkpoints are written only after a completed optimizer step. Trainer metadata
includes global step, epoch, sample cursor, shuffle seed, best validation score,
the consecutive zero-signal-step count,
protocol version, manifest digest, data digest, tokenizer/grammar fingerprint,
and the resolved training configuration digest. Operational controls that do
not change optimization semantics (`resume`, `max_steps`, `preflight_only`, and
terminal verbosity) are excluded from that digest, so extending a bounded run
does not make its own checkpoint incompatible. Tensor state includes the FP32
Actor, tokenizer, AdamW state, scheduler state, Python RNG, Torch CPU RNG, and
all CUDA RNG states. The frozen Reference is not copied; its immutable path and
fingerprint are recorded.

The checkpoint root contains at most:

- `latest`: the newest complete resumable checkpoint, approximately 36 GB;
- `previous`: the preceding complete resumable checkpoint, approximately 36 GB;
- `best_actor`: the best clean-validation Actor only, approximately 12 GB.

Saving uses an incoming directory and a `COMPLETE` marker. The runner first
writes and closes the incoming checkpoint while `latest` remains intact, then
removes the old `previous`, renames `latest` to `previous`, and atomically
renames incoming to `latest`. Resume scans `latest`, `previous`, and a completed
incoming directory, rejects incompatible or incomplete candidates, and chooses
the highest compatible global step. An interruption can lose the unfinished
optimizer step, but cannot silently resume a half-written state.

Default checkpoint frequency is every ten optimizer steps, at every validation
step, and at the final step. Validation is never allowed to produce a
best-Actor candidate that is newer than the latest resumable optimizer state.
This causes multiple write events in one epoch, not multiple retained copies.
Only two complete training generations remain on disk. The post-cleanup data
disk has approximately 140 GB free, enough for the estimated 84 GB steady state
and the temporary best-Actor replacement.

## 6. Clean validation and best-model selection

Every 25 optimizer steps and at the final step, the in-memory Actor performs
greedy FSM-constrained generation on all 146 clean-validation rows. Validation
does not create gradients or alter RNG used for subsequent training. It reports:

- strict-format rate and forced-rationale-close rate;
- mean dense three-field reward;
- accuracy for each judgment field;
- exact Harmfulness accuracy;
- fixed-label Harmfulness macro-F1 and macro-recall over `[0.0, 0.5, 1.0]`.

Checkpoint selection first maximizes mean dense reward, then fixed-label
Harmfulness macro-F1, then prefers the earlier global step. When a new best is
found, an incoming Actor-only directory is completely written before replacing
`best_actor`. Validation itself does not create another full optimizer
checkpoint.

## 7. Logs and observability

`train_metrics.jsonl` stores one concise record per optimizer step: loss terms,
KL, reward distribution, variable/all-equal group counts, response lengths,
old/new parity, gradient norm, learning rate, timing, and peak GPU memory.
`validation_metrics.jsonl` stores one record per validation run.
`rollouts.jsonl` stores source identity, judgments, reward, advantage, and the
response for all generated samples. Each record includes a run-session ID; if
an interruption discards steps newer than `latest`, replayed step numbers form
a new session instead of silently overwriting or masquerading as the abandoned
attempt. These append-only text files remain small relative to weights and make
safety decisions auditable.

The terminal prints one short line per prompt group and one summary line per
optimizer step. It does not dump full prompts on every iteration. Full content
remains available in `rollouts.jsonl`.

## 8. Failure policy

The runner fails closed before an optimizer update if any response violates the
FSM contract, any log probability/loss/gradient is non-finite, Actor/old replay
exceeds the configured parity tolerance, Reference or tokenizer fingerprints
change, or data/config/manifest compatibility fails. A whole all-equal group is
valid, but ten consecutive optimizer steps with zero variable-reward groups
abort the run to avoid spending GPU time without policy-gradient signal.

Before each checkpoint, the runner checks available space. An out-of-space
condition stops training while the existing `latest` remains untouched. A CUDA
or process failure likewise leaves the last completed checkpoint available.
Resume never silently falls back to an incompatible model, grammar, dataset, or
configuration.

## 9. E5 verification and non-goals

E5 uses focused tests only. Pure tests cover deterministic shuffle/cursor
recovery, fixed-label validation metrics, best-model ordering, forbidden data
paths, complete-marker discovery, and two-generation rotation. A tiny-model
test proves that uninterrupted training and save/resume training produce the
same next-step parameters, optimizer state, scheduler state, and cursor.
`--preflight-only` validates the real YAML, 1,208-row Parquet, 146-row clean
validation, path restrictions, digests, and 302-step schedule without loading a
3B model.

E5 does not run formal GRPO, save a 3B checkpoint, evaluate banking, inspect
ASB, implement batched FSM generation, or claim a final runtime. E6 will run a
bounded real-model throughput/resume gate and convert measured seconds per
prompt group into a reliable full-run estimate. E7 remains the user-started
formal training stage.

## 10. Implementation evidence

The model-free cloud preflight passed with 1,208 training rows, 146 clean
validation rows, 302 optimizer steps, 16 rollouts per step, zero identity
overlap, and zero banking/ASB rows. It reported
`formal_training_started=false`, `model_loaded=false`, and
`optimizer_created=false`. The final focused E2–E5 suite contains 41 passing
tests. The implementation and operator handoff are recorded in
`practice/toolsafe_reproduction/grpo/E5_TRAINING.md`; no formal 3B training or
3B checkpoint belongs to E5.

# ToolSafe Constrained GRPO v3 Design

Date: 2026-09-07  
Status: E0-E3 implemented; E4 remains gated  
Scope: Qwen2.5-3B Guardian, one A800 80GB, existing AgentHarm and AgentDojo data

## 1. Motivation

The previous SFT-to-GRPO experiments established three independent problems:

1. strict response formatting dominated rollout reward and discarded otherwise useful semantic judgments;
2. the in-process verl FSDP Actor to embedded vLLM execution boundary did not reproduce the exported model's behavior reliably;
3. dynamic macro averaging changed the reported denominator whenever a prediction class was absent.

Increasing SFT epochs, tuning temperature/top-p, increasing response length, or replacing the 3B model cannot correct these architectural problems. Version 3 therefore makes formatting an inference constraint, keeps one Actor as the source of truth, and freezes dataset and evaluation boundaries before further paid training.

## 2. Goals and non-goals

### Goals

- Guarantee the four-line Guardian response protocol during rollout.
- Let GRPO optimize safety reasoning and the three judgments instead of punctuation and closing tags.
- Use the same constrained probability distribution for rollout, Actor log-probability, and Reference KL.
- Eliminate Actor-to-vLLM weight-copy ambiguity from the primary training path.
- Keep SFT, GRPO, clean validation, external banking, and final ASB roles auditable.
- Produce checkpoints that can be resumed and evaluated by the existing Transformers runner.

### Non-goals

- Recover AgentAlign or the unavailable paper training artifacts.
- Use ASB for development, Teacher annotation, training, or checkpoint selection.
- Reproduce every paper number exactly.
- Add a learned reward model or online LLM judge in the first constrained-GRPO experiment.
- Start real SFT or GRPO automatically; the user starts paid training manually.

## 3. Chosen architecture

The primary backend is a single-GPU Hugging Face/PyTorch GRPO trainer. The current trainable Actor performs rollout generation and all old/new log-probability calculations. A separately loaded frozen Reference computes KL terms. There is no embedded vLLM Actor copy.

The implementation will expose a small rollout-backend interface so that a future vLLM implementation can be added only after weight and constrained-log-probability parity tests pass. vLLM is not the primary backend in this experiment.

## 4. Guardian token grammar

The emitted response remains:

```text
<Think> rationale <\Think>
<Malicious_User_Request> yes|no <\Malicious_User_Request>
<Being_Attacked> yes|no <\Being_Attacked>
<Harmfulness_Rating> 0.0|0.5|1.0 <\Harmfulness_Rating>
```

The decoder is a token-level finite-state machine:

1. fixed protocol literals are inserted deterministically and included in later context;
2. rationale tokens are sampled from a safe free-text set;
3. after the minimum rationale content, the tokenizer's first post-rationale protocol token is the learned transition to the first judgment; it must be absent from the safe free-text vocabulary, otherwise grammar compilation fails;
4. if the rationale reaches its configured cap, the decoder forces the transition and records it;
5. judgment values are sampled through tokenizer-specific tries over `yes/no` and `0.0/0.5/1.0`;
6. all closing literals are deterministic and the final state terminates generation.

The FSM operates only on the newly generated response. Tag-like text inside the untrusted prompt cannot change its state.

## 5. Constrained probability semantics

At every sampled semantic position, disallowed logits are set to negative infinity before `log_softmax`. The same allowed-token mask is applied when calculating:

- rollout/old policy log-probability;
- updated Actor log-probability;
- frozen Reference log-probability for KL.

Fixed syntax tokens are present in the sequence so later predictions condition on the exact protocol, but they are excluded from policy-gradient and KL masks. Consequently the model is trained on rationale and categorical choices, not on characters already guaranteed by code.

KV-cache decoding and full-sequence teacher forcing use different matrix shapes, so bf16 can introduce a small, measured numerical difference even with identical weights. The cache-time value is therefore a diagnostic only. The authoritative PPO old log-probability is computed by full-sequence constrained replay and must match an immediate second replay within the strict configured tolerance. Cache-vs-replay also has a wider hard bound that detects wrong weights, context, masks, or token positions; neither check may silently fall back to unconstrained training.

## 6. Reward

Grammar validity is no longer a reward gate or bonus because it is guaranteed by construction. The first reward version is:

- Malicious User Request correct: 0.20
- Being Attacked correct: 0.20
- Harmfulness Rating correct: 0.30
- all three correct: additional 0.30

The maximum is 1.0. Partial field correctness supplies dense signal. There is no constant Think-presence reward: a constant within a group cannot create useful relative advantage. Rationale tokens remain part of the sampled semantic sequence, so the outcome reward reinforces or suppresses their probability together with the final judgments.

Groups whose rewards are identical have zero relative advantage and correctly contribute no policy-gradient signal. The trainer reports them rather than manufacturing artificial differences.

## 7. Data policy

The machine-readable source of truth is:

`practice/toolsafe_reproduction/grpo/config/constrained_grpo_experiment_manifest.yaml`

Training candidates are AgentHarm and AgentDojo slack/travel/workspace. The 146-row clean validation set alone may select checkpoints. Banking is an 87-row external report set and may not select checkpoints. The 5,231 ASB rows remain untouched until the model and checkpoint are frozen for final evaluation.

An API Teacher may independently annotate training candidates and clean validation without seeing existing labels. Agreements may be accepted automatically; disagreements must be exported for human adjudication. Banking and ASB never enter this annotation pipeline.

## 8. Evaluation

Exact Harmfulness metrics always use the complete label space `[0.0, 0.5, 1.0]`, represented internally as `[0, 1, 2]`. A model that never predicts 0.5 therefore receives zero for that class rather than silently changing the macro denominator.

Constrained rollout reports additionally include:

- strict parse rate;
- constraint violation count;
- forced rationale close rate;
- reward mean and per-group standard deviation;
- variable-reward and all-equal-reward group rates;
- response length and clipping statistics.

## 9. Safety and failure behavior

- Tokenizer/model mismatch fails before generation.
- An empty enum trie or unreachable FSM state fails with the prompt identity and state name.
- A constraint violation stops the run before optimizer creation.
- A log-probability parity failure stops the run before backward.
- Checkpoints store Actor, optimizer, scheduler, random-number state, global step, and protocol version.
- Resume refuses checkpoints from a different manifest or grammar version.
- No code path may silently fall back to unconstrained decoding.

## 10. Staged execution

E0 freezes this design, the dataset manifest, and exact metric behavior. E1 adds blind API Teacher labeling and conflict review. E2 implements and tests the tokenizer-aware FSM. E3 adds constrained probability recomputation. E4 builds the same-Actor trainer and one-step smoke gate. E5 runs cleaned SFT. E6 runs the constrained rollout gate. E7 is user-started full GRPO. E8 evaluates clean validation, banking, and finally ASB. E9 updates the project record and interview materials.

Each stage ends with a small report and Git commit. Real training is never started as part of an implementation stage.

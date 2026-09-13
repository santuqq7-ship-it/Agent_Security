# Guardian Rationale Repair Design

## Goal

Use a larger local Teacher model to add evidence-based `Think` rationales to
only the 302 disjoint SFT training records, then repair the Qwen2.5-3B
fields-only Guardian with weighted full-parameter SFT.

## Data contract

- Input: `data/sft_grpo_disjoint/sft_train.jsonl` only.
- The Teacher sees the original Guardian prompt and immutable gold values for
  `Malicious_User_Request`, `Being_Attacked`, and `Harmfulness_Rating`.
- The Teacher returns exactly one strict `<Think> ... <\Think>` block.
- The preparation script validates a non-empty rationale and rejects extra
  judgment tags. It assembles the final four-line completion itself from the
  original gold fields, so Teacher output can never overwrite labels.
- Every accepted row is flushed immediately and keyed by `source_identity` so
  interrupted generation can resume without regenerating completed rows.
- Banking, held-out validation, and GRPO training rows are never sent to the
  Teacher.

## Training contract

The causal-LM input remains `chat_template(prompt) + completion`. Per-token
cross-entropy weights are:

- prompt and padding: `0.0`;
- `Think` block: `0.2`;
- all three judgment-field lines: `1.0`;
- EOS: `1.0`.

Weighted loss is calculated after the normal causal one-token shift and is
normalized by the sum of active token weights. Existing completion-only and
fields-only SFT modes remain unchanged.

The repair run starts from
`results/sft_qwen25_3b_full_three_field_fields_only`, uses one full-parameter
epoch at learning rate `2e-6`, and keeps the original non-Teacher validation
file for held-out field supervision. Model quality is accepted only through
free generation on held-out validation and external banking, including field
metrics, parser success, and non-empty Think coverage.

## Deferred work

GRPO v2 is not changed in this implementation. After SFT acceptance its reward
will be changed to 0.20/0.20/0.30 per field, 0.20 joint correctness, and 0.10
non-empty valid Think format.

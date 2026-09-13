# E1 API Teacher Blind Labeling

This stage independently annotates the existing SFT, GRPO, and clean-validation rows. It never changes the source datasets and never sends source identities, dataset/subset names, split metadata, or reconstructed labels to the API.

## Private API configuration

Edit this Git-ignored file on the cloud host:

`/root/Agent-Security/practice/toolsafe_reproduction/private/teacher_api.json`

Fill exactly three values:

```json
{
  "request_url": "FULL_OPENAI_COMPATIBLE_ENDPOINT",
  "api_key": "YOUR_API_KEY",
  "model": "MODEL_NAME"
}
```

`request_url` must be the complete request endpoint, not only a host. The script automatically recognizes an endpoint whose path ends in `/responses`; every other endpoint is treated as OpenAI-compatible Chat Completions. Authentication is sent as `Authorization: Bearer <api_key>`.

For Chat Completions, the program requests JSON mode. The initial output budget is 4096 tokens; if a reasoning model consumes that budget before emitting final JSON, later retries may use up to 8192 tokens. This limit is a ceiling, not a requirement to consume all those tokens.

## Inspect the real blind prompt without calling the API

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
/root/Agent-Security/practice/toolsafe_reproduction/sft/api_teacher_blind_label.py \
  --show-first-prompt
```

This loads and validates all selected input boundaries, prints the first request, and exits before reading the API configuration.

## Balanced real API smoke

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
/root/Agent-Security/practice/toolsafe_reproduction/sft/api_teacher_blind_label.py \
  --max-samples 10 \
  --balanced-smoke
```

The balancing uses the existing labels only to select diverse local records. Those labels and their combinations are never inserted into API messages. Version 2 also contains no concrete example answer such as `false/false/0.0`.

Inspect the five annotations before paying for the complete pass:

```bash
sed -n '1,10p' \
/root/Agent-Security/practice/toolsafe_reproduction/data/teacher_blind_v2/annotations.jsonl
```

## Complete pass

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
/root/Agent-Security/practice/toolsafe_reproduction/sft/api_teacher_blind_label.py
```

The same command is safe to rerun. Completed `source_identity` values are skipped, and only missing rows are sent to the API. A record that exhausts all retries is logged and processing continues with later rows.

Before sending new requests, resume also rechecks historical rejected responses with the current parser. If a prior response is now valid—for example, it contained all judgments but misspelled the rationale key—it is recovered locally and is not billed again.

## Offline audit

```bash
PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
/root/Agent-Security/practice/toolsafe_reproduction/sft/api_teacher_blind_label.py \
  --audit-only
```

This performs no inference and sends no API request. It validates existing annotations and rebuilds derived artifacts.

## Artifacts

All outputs are written under:

`/root/Agent-Security/practice/toolsafe_reproduction/data/teacher_blind_v2/`

- `annotations.jsonl`: immutable Teacher annotations and raw API responses.
- `agreements.jsonl`: all three Teacher fields agree with reconstructed fields.
- `conflicts.jsonl`: at least one field differs.
- `conflict_review_queue.jsonl`: conflict evidence plus human adjudication placeholders. Reruns preserve completed human entries.
- `rejected.jsonl`: exhausted HTTP or parsing failures; these identities remain pending.
- `report.json`: progress and disagreement counts.

E1 does not automatically replace labels. Conflict adjudication and construction of corrected SFT/GRPO files occur only after the complete report has been reviewed.

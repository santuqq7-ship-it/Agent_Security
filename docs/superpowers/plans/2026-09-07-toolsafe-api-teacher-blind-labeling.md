# E1 API Teacher Blind Labeling Implementation Plan

**Goal:** Build an auditable, resumable API pipeline that independently labels the existing SFT, GRPO, and clean-validation candidates without exposing their reconstructed labels to the Teacher.

**Architecture:** Normalize JSONL and Parquet rows into one private record schema, send only quoted Guardian evidence to an OpenAI-compatible API, parse one strict JSON judgment, and append immutable annotations. Derived agreement, conflict, review-queue, and report files are rebuilt locally without modifying source datasets.

**Runtime:** Python 3.10 standard library plus PyArrow for GRPO Parquet input. The existing `toolsafe-grpo` environment supplies PyArrow. No API SDK is required.

---

## Task 1: Freeze the blind-label contract

Create tests that require:

- no source identity, dataset name, subset name, or existing label in the API messages;
- the old Guardian output-contract suffix to be removed from quoted evidence;
- exact definitions for the three independent fields;
- robust extraction of a JSON object from plain or fenced API output;
- strict normalized types: two booleans, one of `0.0/0.5/1.0`, and a non-empty rationale.

## Task 2: Normalize source formats and enforce boundaries

Support these inputs only:

- 302-row SFT JSONL;
- 1208-row GRPO Parquet;
- 146-row clean-validation JSONL.

Reject AgentDojo banking, ASB, unknown datasets, missing identities, invalid labels, and duplicate source identities before any API request.

## Task 3: Implement the API boundary

Use a Git-ignored configuration containing exactly:

- `request_url`
- `api_key`
- `model`

Support both OpenAI-compatible Chat Completions and Responses response envelopes. Never print or persist the API key. Continue after record-level HTTP or parse failures, then retry only missing identities on the next invocation.

## Task 4: Produce auditable artifacts

Write:

- `annotations.jsonl`: immutable parsed annotation plus raw API response;
- `rejected.jsonl`: exhausted request/parse failures;
- `agreements.jsonl`: Teacher agrees with all reconstructed fields;
- `conflicts.jsonl`: at least one field differs;
- `conflict_review_queue.jsonl`: preserves human adjudication fields across reruns;
- `report.json`: counts by role, source, conflict field, and completion status.

Never rewrite SFT or GRPO training files in E1.

## Task 5: Verify and hand off

Run only the focused E1 unit tests and an audit-only synthetic smoke. Commit source, tests, prompt, documentation, and ignore rule. Leave the blank private API configuration untracked. The user then fills three values and runs a small real API smoke before approving full annotation.

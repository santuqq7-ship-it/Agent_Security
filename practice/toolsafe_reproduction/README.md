# ToolSafe Local Transformers Reproduction

This directory contains compatibility code and experiment records for the
local MPS/CPU reproduction. The original ToolSafe checkout remains at
`../../ToolSafe` and is not modified by this practice layer.

Persistent cloud paths and non-negotiable environment facts are recorded in
[`PROJECT_MEMORY.md`](PROJECT_MEMORY.md). Read that file before running cloud
model preparation, evaluation, or training commands.

## Task 1 audit result

- ToolSafe commit: `46358fa424a927a895c6c8322f99032c4eb5155e`
- Python: `3.12.13`
- PyTorch: `2.13.0`
- Transformers: `5.15.1`
- MPS was built into PyTorch but was unavailable during the audit process;
  local code must therefore support CPU fallback.
- `openai`, `pydantic`, `python-dotenv`, and `rich` are installed for the
  original module import boundary.
- vLLM is intentionally not installed on this Mac.
- `ToolSafe/TS-Guard/verl-main` is absent from the current checkout.
- Agent and baseline Guardian checkpoint:
  `/Users/qixinyao.1/Desktop/Agent-Security/practice/models/Qwen2.5-1.5B-Instruct`

## Why an overlay is needed

The original `ToolSafe/src/model/model.py` imports vLLM at module import time.
That prevents even the Transformers `analysis` path from importing on a Mac
without vLLM. Task 2 will copy only the required model/agent boundary files,
make vLLM imports lazy, and add a real Transformers branch to `Guardian`.
The original prompts, parser, evaluator, environment, and datasets remain the
source of truth.

## Task 2 adapter result

The independent overlay now provides:

- lazy vLLM imports, so local `analysis` mode imports without vLLM;
- real Transformers loading for both `Model` and `Guardian`, with MPS/CPU
  device selection and local-only checkpoint loading;
- one Guardian `_generate_text` boundary shared by judgment, tool safety, and
  alignment checks;
- the original `GUARD_TEMPLATES`, `guardian_paser_map`, retry counts, risk
  schema, `SecReAct_Agent`, and `runtime.run_function(...)` boundary;
- namespace fallback packages for the unchanged original agent, utility, and
  evaluator modules.

## TS-Bench recovery

The upstream `ToolSafe/TS-Bench` files are Git LFS pointers.  The repository's
LFS budget is currently exhausted, so the official objects are stored in the
independent overlay at `data/TS-Bench/`.  They were downloaded from GitHub's
public media endpoint and verified against all nine upstream SHA-256 digests.
This is the original trajectory benchmark, not a synthetic replacement.

To verify or re-download the files yourself:

```bash
ToolSafe/.venv-phase4/bin/python \
  practice/toolsafe_reproduction/scripts/fetch_tsbench.py
```

Use `--force` to download again.  The script checks byte size, SHA-256, and
JSON syntax before replacing any destination file.

## Phase 4.2: Guardian evaluation on official TS-Bench

The independent runner
`runners/run_tsbench_guardian.py` reuses the original ToolSafe Guardian
template, model parser, retry behavior, and score mapping.  It only adds a
local data root, a sample limit, and JSONL tracing so the model interaction is
visible.

Run one real AgentDojo banking record first:

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"

PYTHONDONTWRITEBYTECODE=1 \
ToolSafe/.venv-phase4/bin/python \
practice/toolsafe_reproduction/runners/run_tsbench_guardian.py \
  --dataset agentdojo \
  --subset banking \
  --max-samples 1 \
  --max-new-tokens 256 \
  --max-retries 3 \
  --score-mode strict \
  --show-prompt \
  --show-responses \
  --output-dir practice/toolsafe_reproduction/results/tsbench_guardian_banking_1
```

Parameter meanings:

- `--dataset` selects `agentdojo`, `agentharm`, or `asb`;
- `--subset` selects a file within that dataset, or `all`;
- `--max-samples` limits inference cost while preserving the real data and
  Guardian logic;
- `--max-new-tokens` bounds each Guardian response length;
- `--max-retries` preserves the original retry behavior when the output cannot
  be parsed as a TS-Guard judgment;
- `--score-mode strict` maps labels with the original ToolSafe convention
  (`0 -> 0`, `0.5 -> 1`, `1 -> 1`).
- `--show-prompt` prints the exact text passed to the Guardian before
  tokenization;
- `--show-responses` prints every raw model response, including responses from
  parser retries.

The command writes:

- `metrics.json`: summary metrics and the number of skipped unparseable
  predictions;
- `guardian_trace.jsonl`: one JSON object per trajectory, including the exact
  rendered prompt, raw Guardian response, parser result, and label.

An unparseable output is recorded as `prediction=null` and skipped by the
metric calculation.  It is not converted to safe or unsafe by a Python rule;
the current 1.5B checkpoint is an untrained baseline and may fail the original
TS-Guard format.  This distinction is important when interpreting the first
run.


Verification:

```bash
ToolSafe/.venv-phase4/bin/pytest practice -q
# 12 passed
```

A real local 1.5B Guardian generation completed successfully. Its output was
not in the original TS-Guard XML-like format, so the unchanged parser returned
only `{"reason": ...}` after the configured retry. This is an expected
untrained-baseline observation, not a Python rule-based risk decision.

## Verification commands

Run from the original repository when checking the source state:

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security/ToolSafe"
git rev-parse HEAD
git status --short
```

The expected tracked change is the user's existing `README.md` modification.
The compatibility overlay is outside this Git repository.

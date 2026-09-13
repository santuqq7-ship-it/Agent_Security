# AgentHarm Guardian SFT data

These JSONL files are generated from the released TS-Bench AgentHarm step
trajectories. The source records contain an aggregate `score`, so this first
LoRA exercise trains the official single-rating Guardian contract only.

- `train.jsonl`: 585 steps from 160 complete interactions.
- `validation.jsonl`: 146 steps from 37 complete interactions.
- `manifest.json`: split seed, source, target protocol, and external-test boundary.
- Raw labels remain exactly `0.0`, `0.5`, and `1.0`.
- The 87 AgentDojo banking records are not present here; they remain an
  untouched post-SFT external test set.

Each JSONL row contains:

```json
{
  "prompt": "the rendered ToolSafe single-rating prompt",
  "completion": "<Think> </\\Think>\n<Judgment> 0.5 <\\Judgment>",
  "score": 0.5,
  "source_file": "harmful_steps.json",
  "id-interaction": 0,
  "id-segment": 0
}
```

The empty `Think` body is intentional. The released files do not contain
validated rationale labels, so the SFT target does not fabricate one.

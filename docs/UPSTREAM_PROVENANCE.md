# Upstream and model provenance

## Official ToolSafe repository

- Nested path: `/root/Agent-Security/ToolSafe`
- Upstream commit: `46358fa424a927a895c6c8322f99032c4eb5155e`
- Historical commit used to recover the deleted TS-Guard/verl tree:
  `b87b7097323f5487a93ced335b5d756c4457cb34`
- The nested repository is intentionally excluded from the new root Git
  repository so its upstream history is not flattened or overwritten.

Before root-repository initialization, the nested working tree already had
user changes in:

- `README.md`
- `src/agent/agent_prompts.py`

The workspace cleanup also removed generated Python bytecode caches.  Some
bytecode files were incorrectly tracked by the upstream repository, so the
nested status reports those cache removals as deletions.  They are not source
changes and are not required at runtime.

## Canonical SFT-to-GRPO model

- Canonical directory:
  `practice/toolsafe_reproduction/grpo/prepared_models/qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2`
- Actor input: `policy_init/`
- Frozen reference: `reference/`
- Dense `model.safetensors` SHA-256 for both copies:
  `64082f9d302cdcb707955fe0846fe15f881b9c48a453624d087af8e58cdd159b`
- Parameters per checkpoint: `1,543,714,304`
- The model directories are excluded from Git; their immutable hashes and
  preparation manifest provide the reproducibility boundary.

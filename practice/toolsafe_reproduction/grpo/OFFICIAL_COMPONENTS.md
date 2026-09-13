# Recovered official GRPO components

## Source

The directory `vendor/verl-main/` is mechanically exported from the ToolSafe
Git object:

```text
b87b7097323f5487a93ced335b5d756c4457cb34:TS-Guard/verl-main
```

It contains 1,209 files and preserves the original verl implementation,
license, launcher, and TS-Guard reward modules.  No implementation edits were
made while recovering this snapshot.

## Paper-specific entry points

- Original launcher:
  `vendor/verl-main/examples/grpo_trainer/run_TSGuard_train.sh`
- Uniform three-field reward:
  `vendor/verl-main/verl/utils/reward_score/agentsafety_v2_uniform.py`
- verl training entry point:
  `vendor/verl-main/verl/trainer/main_ppo.py`

Recovered SHA-256 values:

```text
bfc5a8403b8b0c2a40e3786ae6e4dece6f0e20ce1d4a576ca103c02f81a775e0  run_TSGuard_train.sh
26316320561d01157d9b48d7489e6cd8ba202825e149937d9a156f07b94a10ad  agentsafety_v2_uniform.py
```

The original launcher documents the paper implementation choices:
`algorithm.adv_estimator=grpo`, vLLM rollout, group size 16, direct KL loss
with coefficient `0.001`, KL excluded from reward, four GPUs, Qwen2.5-7B, and
the custom uniform reward.  Our single-A800/Qwen2.5-1.5B launcher will retain
the algorithm and reward semantics while changing only resource-dependent
values.

## Unavailable official training objects

The historical `data/agentsafety/*.json` entries are Git LFS pointer objects.
The upstream repository has exhausted its LFS budget, so their referenced
blobs cannot be downloaded.  Recovery therefore disabled LFS smudge and kept
the pointer text as provenance.  The independently reconstructed Parquet data
used by this project is not represented as the paper's original training set.


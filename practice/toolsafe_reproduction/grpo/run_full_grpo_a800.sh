#!/usr/bin/env bash
set -euo pipefail

# Always use the isolated Python 3.10 GRPO runtime, never the SFT environment.
GRPO_PYTHON=/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python
PREFLIGHT=/root/Agent-Security/practice/toolsafe_reproduction/grpo/preflight_grpo.py
GRPO_CONFIG=/root/Agent-Security/practice/toolsafe_reproduction/grpo/config/a800_full_grpo.yaml
VERL_ROOT=/root/Agent-Security/practice/toolsafe_reproduction/grpo/vendor/verl-main

# Make the recovered official verl package importable and keep diagnostics
# complete if Hydra rejects a configuration key.
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# --launch is the explicit safety boundary: preflight first composes Hydra
# without training, then replaces itself with the real main_ppo process.
exec "${GRPO_PYTHON}" "${PREFLIGHT}" --config "${GRPO_CONFIG}" --launch


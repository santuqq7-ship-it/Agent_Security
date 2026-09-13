"""Tests for backward-compatible full-parameter SFT selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from train_guardian_sft import (  # noqa: E402
    TrainingConfig,
    configure_trainable_parameters,
    validate_finetuning_method,
)


def test_legacy_default_remains_lora() -> None:
    assert TrainingConfig.__dataclass_fields__["finetuning_method"].default == "lora"


def test_full_mode_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="finetuning_method"):
        validate_finetuning_method("unknown")


def test_full_mode_keeps_all_parameters_trainable() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    for parameter in model.parameters():
        parameter.requires_grad = False

    configure_trainable_parameters(model, "full")

    assert all(parameter.requires_grad for parameter in model.parameters())

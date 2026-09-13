"""Pure E4 acceptance-report tests; no model loading or training occurs here."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from e4_one_step_grpo_smoke import build_e4_gate_report  # noqa: E402


def _valid_observations() -> dict[str, object]:
    return {
        "rewards": [0.2, 1.0, 0.5, 1.0],
        "advantages": [-1.5, 0.9, -0.3, 0.9],
        "strict_format_count": 4,
        "rollout_count": 4,
        "old_new_max_abs_error": 0.0,
        "old_new_parity_tolerance": 1e-5,
        "policy_loss": -0.12,
        "kl_loss": 0.03,
        "total_loss": -0.11997,
        "gradient_norm": 2.0,
        "parameter_delta": 1e-7,
        "optimizer_created": True,
        "backward_executed": True,
        "optimizer_step_executed": True,
    }


def test_e4_report_accepts_only_a_real_variable_finite_parameter_update() -> None:
    accepted = build_e4_gate_report(**_valid_observations())
    assert accepted["passed"] is True

    failure_cases = (
        ("variable_reward_group", {"rewards": [1.0, 1.0, 1.0, 1.0]}),
        ("finite_math", {"total_loss": float("inf")}),
        ("positive_gradient", {"gradient_norm": 0.0}),
        ("parameter_updated", {"parameter_delta": 0.0}),
    )
    for criterion, mutation in failure_cases:
        observations = _valid_observations()
        observations.update(mutation)
        rejected = build_e4_gate_report(**observations)
        assert rejected["passed"] is False
        assert rejected["criteria"][criterion] is False


def test_e4_report_rejects_inconsistent_rollout_vector_lengths() -> None:
    observations = _valid_observations()
    observations["advantages"] = [-1.0, 1.0]
    report = build_e4_gate_report(**observations)

    assert report["passed"] is False
    assert report["criteria"]["rollout_accounting"] is False


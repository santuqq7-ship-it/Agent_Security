"""Focused E5 tests for scheduling, rotation, metrics, and data boundaries."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from constrained_grpo_core import (  # noqa: E402
    ConstrainedTrainerState,
    load_checkpoint_metadata,
    save_training_checkpoint,
)
from constrained_grpo_runtime import (  # noqa: E402
    DataBoundaryError,
    ValidationSelection,
    append_jsonl,
    compute_validation_metrics,
    discover_resume_checkpoint,
    epoch_indices,
    is_better_validation,
    load_training_config,
    optimizer_step_batches,
    rotate_checkpoint,
    semantic_config_sha256,
    sha256_file,
    validate_data_boundaries,
)


PROTOCOL = "toolsafe-guardian-token-fsm-v2"
COMPATIBILITY = {
    "expected_protocol_version": PROTOCOL,
    "expected_manifest_sha256": "manifest",
    "expected_data_sha256": "data",
    "expected_config_sha256": "config",
    "expected_tokenizer_fingerprint": "tokenizer",
}


def test_schedule_is_deterministic_and_covers_1208_rows_once() -> None:
    first = epoch_indices(row_count=1208, epoch=0, seed=20260909)
    second = epoch_indices(row_count=1208, epoch=0, seed=20260909)

    assert first == second
    assert sorted(first) == list(range(1208))
    batches = optimizer_step_batches(
        first, sample_cursor=0, groups_per_step=4
    )
    assert len(batches) == 302
    assert all(len(batch) == 4 for batch in batches)
    assert optimizer_step_batches(
        first, sample_cursor=1200, groups_per_step=4
    ) == batches[-2:]


@pytest.mark.parametrize(
    "forbidden",
    (
        "practice/toolsafe_reproduction/data/TS-Bench/agentdojo-traj/banking.json",
        "practice/toolsafe_reproduction/data/TS-Bench/asb-traj/test",
    ),
)
def test_development_paths_reject_banking_and_asb(forbidden: str) -> None:
    with pytest.raises(DataBoundaryError):
        validate_data_boundaries(
            training_path=forbidden,
            validation_path=(
                "practice/toolsafe_reproduction/data/teacher_adjudicated_v1/"
                "clean_validation.jsonl"
            ),
        )


def test_config_digest_excludes_only_operational_overrides() -> None:
    base = {
        "learning_rate": 1e-6,
        "rollouts_per_prompt": 4,
        "resume": "auto",
        "max_steps": None,
        "preflight_only": False,
        "terminal_verbosity": "concise",
    }
    operationally_changed = {
        **base,
        "resume": "none",
        "max_steps": 2,
        "preflight_only": True,
        "terminal_verbosity": "verbose",
    }
    semantically_changed = {**base, "learning_rate": 2e-6}

    assert semantic_config_sha256(base) == semantic_config_sha256(
        operationally_changed
    )
    assert semantic_config_sha256(base) != semantic_config_sha256(
        semantically_changed
    )


def test_yaml_loading_override_and_file_digest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: 1\nlearning_rate: 0.000001\nmax_steps: null\n",
        encoding="utf-8",
    )

    config = load_training_config(config_path, overrides={"max_steps": 2})

    assert config["schema_version"] == 1
    assert config["learning_rate"] == pytest.approx(1e-6)
    assert config["max_steps"] == 2
    assert sha256_file(config_path) == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()


def _tiny_checkpoint_writer(step: int):
    def write(path: Path) -> None:
        actor = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        save_training_checkpoint(
            path,
            actor=actor,
            tokenizer=None,
            optimizer=optimizer,
            scheduler=scheduler,
            state=ConstrainedTrainerState(
                global_step=step,
                epoch=0,
                protocol_version=PROTOCOL,
                manifest_sha256="manifest",
                sample_cursor=step * 4,
                data_sha256="data",
                config_sha256="config",
                tokenizer_fingerprint="tokenizer",
            ),
        )

    return write


def _checkpoint_step(path: Path) -> int:
    return load_checkpoint_metadata(path, **COMPATIBILITY).global_step


def test_rotation_keeps_only_latest_and_previous_complete_generations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    for step in (10, 20, 30):
        rotate_checkpoint(
            root, global_step=step, writer=_tiny_checkpoint_writer(step)
        )

    assert _checkpoint_step(root / "latest") == 30
    assert _checkpoint_step(root / "previous") == 20
    assert (root / "latest" / "COMPLETE").is_file()
    assert (root / "previous" / "COMPLETE").is_file()
    assert sorted(path.name for path in root.iterdir()) == ["latest", "previous"]


def test_failed_incoming_write_preserves_latest_and_is_not_resumable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    rotate_checkpoint(root, global_step=10, writer=_tiny_checkpoint_writer(10))

    def fail_after_partial_write(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "partial").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("interrupted write")

    with pytest.raises(RuntimeError, match="interrupted"):
        rotate_checkpoint(root, global_step=20, writer=fail_after_partial_write)

    assert _checkpoint_step(root / "latest") == 10
    assert discover_resume_checkpoint(
        root, compatibility=COMPATIBILITY
    ) == root / "latest"


def test_completed_incoming_generation_can_be_recovered(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    rotate_checkpoint(root, global_step=10, writer=_tiny_checkpoint_writer(10))
    incoming = root / "incoming_step_20"
    _tiny_checkpoint_writer(20)(incoming)
    (incoming / "COMPLETE").write_text("20\n", encoding="utf-8")

    assert discover_resume_checkpoint(
        root, compatibility=COMPATIBILITY
    ) == incoming


def _validation_record(
    gold: float,
    predicted: float,
    reward: float,
) -> dict[str, object]:
    return {
        "ground_truth": {
            "Malicious_User_Request": False,
            "Being_Attacked": False,
            "Harmfulness_Rating": gold,
        },
        "predicted": {
            "Malicious_User_Request": False,
            "Being_Attacked": False,
            "Harmfulness_Rating": predicted,
        },
        "reward": reward,
        "strict_format": True,
        "forced_rationale_close": False,
    }


def test_validation_metrics_fix_all_three_harmfulness_labels() -> None:
    metrics = compute_validation_metrics(
        (
            _validation_record(0.0, 0.0, 1.0),
            _validation_record(0.5, 1.0, 0.4),
            _validation_record(1.0, 1.0, 1.0),
        )
    )

    assert metrics["harmfulness_labels"] == [0.0, 0.5, 1.0]
    assert metrics["harmfulness_macro_recall"] == pytest.approx(2 / 3)
    assert metrics["mean_dense_reward"] == pytest.approx(0.8)
    assert metrics["strict_format_rate"] == 1.0


def test_best_selection_prefers_reward_then_macro_f1_then_earlier_step() -> None:
    incumbent = ValidationSelection(0.8, 0.6, 75)

    assert is_better_validation(ValidationSelection(0.81, 0.1, 100), incumbent)
    assert is_better_validation(ValidationSelection(0.8, 0.61, 100), incumbent)
    assert is_better_validation(ValidationSelection(0.8, 0.6, 50), incumbent)
    assert not is_better_validation(ValidationSelection(0.8, 0.6, 100), incumbent)


def test_append_jsonl_preserves_prior_audit_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"step": 1})
    append_jsonl(path, {"step": 2})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step": 1},
        {"step": 2},
    ]

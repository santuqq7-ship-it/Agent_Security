"""Focused E5 preflight tests for the formal constrained GRPO trainer."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from constrained_grpo_trainer import (  # noqa: E402
    PreflightError,
    _validation_record_for_step,
    build_preflight_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    project = tmp_path / "project"
    data = project / "practice/toolsafe_reproduction/data/teacher_adjudicated_v1"
    config_dir = project / "practice/toolsafe_reproduction/grpo/config"
    actor = project / "practice/models/actor"
    reference = project / "practice/models/reference"
    actor.mkdir(parents=True)
    reference.mkdir(parents=True)
    data.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    train_rows = []
    for index in range(4):
        train_rows.append(
            {
                "prompt": [{"role": "user", "content": f"train {index}"}],
                "reward_model": {
                    "ground_truth": {
                        "Malicious_User_Request": False,
                        "Prompt_Injection": False,
                        "Harmfulness_Rating": 0.0,
                    }
                },
                "extra_info": {
                    "source_identity": f"agentharm:benign:file:{index}:0:0",
                    "dataset": "agentharm",
                    "subset": "benign",
                },
            }
        )
    train_path = data / "grpo_train.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)

    validation_rows = [
        {
            "prompt": f"validation {index}",
            "source_identity": f"agentharm:benign:validation:{index}:0:0",
            "dataset": "agentharm",
            "subset": "benign",
            "malicious_user_request": False,
            "being_attacked": False,
            "score": 0.0,
        }
        for index in range(2)
    ]
    validation_path = data / "clean_validation.jsonl"
    _write_jsonl(validation_path, validation_rows)

    manifest = {
        "schema_version": 1,
        "data_boundaries": {
            "grpo_train": {
                "canonical_path": str(train_path.relative_to(project)),
                "logical_rows": 4,
                "allowed_sources": {"agentharm/benign": 4},
            },
            "clean_validation": {
                "canonical_path": str(validation_path.relative_to(project)),
                "logical_rows": 2,
                "allowed_sources": {"agentharm/benign": 2},
            },
            "external_banking": {"role": "external_report_only"},
            "final_asb": {"role": "final_untouched_test_only"},
        },
        "guardian_grammar": {
            "protocol_version": "toolsafe-guardian-token-fsm-v2"
        },
    }
    manifest_path = config_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    config: dict[str, object] = {
        "schema_version": 1,
        "project_root": str(project),
        "actor_model": str(actor.relative_to(project)),
        "reference_model": str(reference.relative_to(project)),
        "manifest": str(manifest_path.relative_to(project)),
        "train_file": str(train_path.relative_to(project)),
        "clean_validation_file": str(validation_path.relative_to(project)),
        "output_root": "practice/toolsafe_reproduction/results/constrained_grpo_v3_e5",
        "epochs": 1,
        "rollouts_per_prompt": 4,
        "prompt_groups_per_step": 4,
        "temperature": 0.8,
        "top_p": 0.95,
        "min_rationale_content_tokens": 8,
        "max_rationale_tokens": 192,
        "learning_rate": 1e-6,
        "weight_decay": 0.0,
        "ppo_clip_ratio": 0.2,
        "kl_coefficient": 0.001,
        "max_grad_norm": 1.0,
        "seed": 20260909,
        "checkpoint_every_steps": 10,
        "validate_every_steps": 25,
        "checkpoint_generations": 2,
        "minimum_free_gb_before_checkpoint": 0,
        "max_consecutive_zero_signal_steps": 10,
        "old_new_parity_tolerance": 1e-5,
        "resume": "auto",
        "max_steps": None,
        "terminal_verbosity": "concise",
    }
    return config, manifest_path, validation_path


def test_preflight_reports_dataset_schedule_and_no_training(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)

    report = build_preflight_report(config)

    assert report["passed"] is True
    assert report["training_rows"] == 4
    assert report["validation_rows"] == 2
    assert report["optimizer_steps_per_epoch"] == 1
    assert report["rollouts_per_optimizer_step"] == 16
    assert report["training_validation_identity_overlap"] == 0
    assert report["banking_rows"] == 0
    assert report["asb_rows"] == 0
    assert report["formal_training_started"] is False
    assert report["model_loaded"] is False
    assert report["optimizer_created"] is False


def test_preflight_rejects_identity_overlap(tmp_path: Path) -> None:
    config, _, validation_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
    rows[0]["source_identity"] = "agentharm:benign:file:0:0:0"
    _write_jsonl(validation_path, rows)

    with pytest.raises(PreflightError, match="overlap"):
        build_preflight_report(config)


def test_preflight_rejects_wrong_protocol(tmp_path: Path) -> None:
    config, manifest_path, _ = _fixture(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["guardian_grammar"]["protocol_version"] = "wrong"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(PreflightError, match="protocol"):
        build_preflight_report(config)


def test_preflight_rejects_unknown_source_counts(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    train_path = Path(str(config["project_root"])) / str(config["train_file"])
    frame = pd.read_parquet(train_path)
    changed = copy.deepcopy(frame.at[0, "extra_info"])
    changed["subset"] = "unknown"
    frame.at[0, "extra_info"] = changed
    frame.to_parquet(train_path, index=False)

    with pytest.raises(PreflightError, match="source counts"):
        build_preflight_report(config)


def test_preflight_rejects_output_outside_experiment_results(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    config["output_root"] = "outside-results"

    with pytest.raises(PreflightError, match="output_root"):
        build_preflight_report(config)


def test_preflight_rejects_negative_weight_decay(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    config["weight_decay"] = -0.01

    with pytest.raises(PreflightError, match="weight_decay"):
        build_preflight_report(config)


def test_preflight_supports_one_final_partial_optimizer_step(tmp_path: Path) -> None:
    config, manifest_path, _ = _fixture(tmp_path)
    train_path = Path(str(config["project_root"])) / str(config["train_file"])
    frame = pd.read_parquet(train_path)
    fifth = copy.deepcopy(frame.iloc[0].to_dict())
    fifth["extra_info"]["source_identity"] = "agentharm:benign:file:4:0:0"
    frame = pd.concat([frame, pd.DataFrame([fifth])], ignore_index=True)
    frame.to_parquet(train_path, index=False)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["data_boundaries"]["grpo_train"]["logical_rows"] = 5
    manifest["data_boundaries"]["grpo_train"]["allowed_sources"] = {
        "agentharm/benign": 5
    }
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    report = build_preflight_report(config)

    assert report["optimizer_steps_per_epoch"] == 2
    assert report["total_optimizer_steps"] == 2


def test_validation_record_lookup_ignores_other_steps_and_partial_rollouts(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "validation_metrics.jsonl"
    _write_jsonl(
        metrics,
        [
            {"global_step": 25, "mean_dense_reward": 0.4},
            {"global_step": 50, "mean_dense_reward": 0.6},
        ],
    )

    assert _validation_record_for_step(metrics, global_step=25) == {
        "global_step": 25,
        "mean_dense_reward": 0.4,
    }
    assert _validation_record_for_step(metrics, global_step=30) is None

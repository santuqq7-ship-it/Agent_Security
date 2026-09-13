"""Focused tests for E1 conflict adjudication and dataset rebuilding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from adjudicate_teacher_labels import (  # noqa: E402
    adjudicate_conflict,
    rebuild_datasets,
)


def _labels(malicious: bool, attacked: bool, score: float) -> dict:
    return {
        "Malicious_User_Request": malicious,
        "Being_Attacked": attacked,
        "Harmfulness_Rating": score,
    }


def _conflict(
    identity: str,
    *,
    original: dict,
    teacher: dict,
    differences: list[str],
) -> dict:
    return {
        "source_identity": identity,
        "dataset": identity.split(":")[0],
        "subset": identity.split(":")[1],
        "role": "sft_train",
        "original_labels": original,
        "teacher_annotation": {
            "rationale": "The evidence provides a concrete independent safety analysis.",
            **teacher,
        },
        "differences": differences,
        "status": "conflict",
    }


def test_adjudication_accepts_clear_malicious_request_but_respects_explicit_authorization() -> None:
    credential_exfiltration = _conflict(
        "agentharm:benign:benign_steps.json:116:2:74",
        original=_labels(False, False, 1.0),
        teacher=_labels(True, False, 1.0),
        differences=["Malicious_User_Request"],
    )
    authorized_test_accounts = _conflict(
        "agentharm:benign:benign_steps.json:72:2:113",
        original=_labels(False, False, 0.0),
        teacher=_labels(True, False, 1.0),
        differences=["Malicious_User_Request", "Harmfulness_Rating"],
    )

    assert adjudicate_conflict(credential_exfiltration)["final_labels"] == _labels(
        True, False, 1.0
    )
    assert adjudicate_conflict(authorized_test_accounts)["final_labels"] == _labels(
        False, False, 0.0
    )


def test_adjudication_marks_only_current_action_causal_injection() -> None:
    direct = _conflict(
        "agentdojo:travel:travel.json:0:1",
        original=_labels(False, False, 0.0),
        teacher=_labels(False, True, 0.5),
        differences=["Being_Attacked", "Harmfulness_Rating"],
    )
    historical_only = _conflict(
        "agentdojo:slack:slack.json:4:6",
        original=_labels(False, False, 0.0),
        teacher=_labels(False, True, 0.0),
        differences=["Being_Attacked"],
    )

    assert adjudicate_conflict(direct)["final_labels"] == _labels(False, True, 0.0)
    assert adjudicate_conflict(historical_only)["final_labels"] == _labels(
        False, False, 0.0
    )


def test_adjudication_preserves_official_harmfulness_and_rejects_unknown_mur_task() -> None:
    known = _conflict(
        "agentharm:harmful:harmful_steps.json:0:0:0",
        original=_labels(True, False, 0.5),
        teacher=_labels(True, False, 1.0),
        differences=["Harmfulness_Rating"],
    )
    unknown = _conflict(
        "agentharm:benign:benign_steps.json:999:0:0",
        original=_labels(False, False, 0.0),
        teacher=_labels(True, False, 0.0),
        differences=["Malicious_User_Request"],
    )

    assert adjudicate_conflict(known)["final_labels"]["Harmfulness_Rating"] == 0.5
    with pytest.raises(ValueError, match="unreviewed Malicious_User_Request task"):
        adjudicate_conflict(unknown)


def test_rebuild_filters_rationale_label_conflicts_and_updates_grpo(tmp_path: Path) -> None:
    agreement_identity = "agentharm:benign:benign_steps.json:5:6:36"
    conflict_identity = "agentharm:benign:benign_steps.json:116:2:74"
    task_agreement_identity = "agentharm:benign:benign_steps.json:116:1:144"
    grpo_identity = "agentharm:benign:benign_steps.json:116:3:147"
    validation_identity = "agentharm:harmful:harmful_steps.json:0:0:99"
    agreement_labels = _labels(False, False, 0.0)
    conflict_original = _labels(False, False, 1.0)
    conflict_teacher = _labels(True, False, 1.0)
    validation_labels = _labels(True, False, 0.5)
    sft_rows = [
        {
            "source_identity": agreement_identity,
            "prompt": "agreement prompt",
            "completion": "old",
            "score": 0.0,
            "malicious_user_request": False,
            "being_attacked": False,
            "dataset": "agentharm",
            "subset": "benign",
            "source_file": "benign_steps.json",
            "split": "sft_train",
        },
        {
            "source_identity": conflict_identity,
            "prompt": "conflict prompt",
            "completion": "old",
            "score": 1.0,
            "malicious_user_request": False,
            "being_attacked": False,
            "dataset": "agentharm",
            "subset": "benign",
            "source_file": "benign_steps.json",
            "split": "sft_train",
        },
        {
            "source_identity": task_agreement_identity,
            "prompt": "same request, earlier step",
            "completion": "old",
            "score": 0.0,
            "malicious_user_request": False,
            "being_attacked": False,
            "dataset": "agentharm",
            "subset": "benign",
            "source_file": "benign_steps.json",
            "split": "sft_train",
        },
    ]
    validation_rows = [
        {
            "source_identity": validation_identity,
            "prompt": "validation prompt",
            "completion": "old",
            "score": 0.5,
            "malicious_user_request": True,
            "being_attacked": False,
            "dataset": "agentharm",
            "subset": "harmful",
            "source_file": "harmful_steps.json",
            "split": "validation",
        }
    ]
    grpo_rows = [
        {
            "prompt": [{"role": "user", "content": "grpo prompt"}],
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "Malicious_User_Request": False,
                    "Prompt_Injection": False,
                    "Harmfulness_Rating": 1.0,
                },
            },
            "extra_info": {
                "source_identity": grpo_identity,
                "dataset": "agentharm",
                "subset": "benign",
            },
        }
    ]
    annotations = [
        {
            "source_identity": agreement_identity,
            "dataset": "agentharm",
            "subset": "benign",
            "role": "sft_train",
            "original_labels": agreement_labels,
            "teacher_annotation": {
                "rationale": "The current action is aligned and creates no meaningful safety risk.",
                **agreement_labels,
            },
            "differences": [],
            "status": "agreement",
        },
        _conflict(
            conflict_identity,
            original=conflict_original,
            teacher=conflict_teacher,
            differences=["Malicious_User_Request"],
        ),
        {
            "source_identity": task_agreement_identity,
            "dataset": "agentharm",
            "subset": "benign",
            "role": "sft_train",
            "original_labels": agreement_labels,
            "teacher_annotation": {
                "rationale": "The current read-only step is safe and the request appears benign.",
                **agreement_labels,
            },
            "differences": [],
            "status": "agreement",
        },
        {
            **_conflict(
                grpo_identity,
                original=conflict_original,
                teacher=conflict_teacher,
                differences=["Malicious_User_Request"],
            ),
            "role": "grpo_train",
        },
        {
            "source_identity": validation_identity,
            "dataset": "agentharm",
            "subset": "harmful",
            "role": "clean_validation",
            "original_labels": validation_labels,
            "teacher_annotation": {
                "rationale": "The request is malicious while this preparatory action has moderate risk.",
                **validation_labels,
            },
            "differences": [],
            "status": "agreement",
        },
    ]

    manifest = rebuild_datasets(
        sft_rows=sft_rows,
        grpo_rows=grpo_rows,
        validation_rows=validation_rows,
        annotations=annotations,
        output_dir=tmp_path,
    )

    rebuilt_sft = [
        json.loads(line) for line in (tmp_path / "sft_train.jsonl").read_text().splitlines()
    ]
    assert [row["source_identity"] for row in rebuilt_sft] == [
        agreement_identity,
        conflict_identity,
    ]
    assert rebuilt_sft[1]["malicious_user_request"] is True
    assert rebuilt_sft[1]["completion"].startswith("<Think> The evidence provides")
    assert manifest["sft"]["excluded_rationale_label_conflicts"] == 1

    rebuilt_grpo = pq.read_table(tmp_path / "grpo_train.parquet").to_pylist()
    assert rebuilt_grpo[0]["reward_model"]["ground_truth"] == {
        "Malicious_User_Request": True,
        "Prompt_Injection": False,
        "Harmfulness_Rating": 1.0,
    }
    assert rebuilt_grpo[0]["prompt"] == grpo_rows[0]["prompt"]
    assert manifest["boundary_audit"]["pairwise_identity_overlap"] == 0
    assert manifest["files"]["grpo_train"]["path"] == "grpo_train.parquet"


def test_rebuild_rejects_reserved_or_overlapping_rows(tmp_path: Path) -> None:
    identity = "agentdojo:banking:banking.json:1:0"
    row = {
        "source_identity": identity,
        "prompt": "prompt",
        "completion": "old",
        "score": 0.0,
        "malicious_user_request": False,
        "being_attacked": False,
        "dataset": "agentdojo",
        "subset": "banking",
        "source_file": "banking.json",
    }
    annotation = {
        "source_identity": identity,
        "dataset": "agentdojo",
        "subset": "banking",
        "role": "sft_train",
        "original_labels": _labels(False, False, 0.0),
        "teacher_annotation": {
            "rationale": "This row must never enter development data.",
            **_labels(False, False, 0.0),
        },
        "differences": [],
    }

    with pytest.raises(ValueError, match="reserved"):
        rebuild_datasets(
            sft_rows=[row],
            grpo_rows=[],
            validation_rows=[],
            annotations=[annotation],
            output_dir=tmp_path,
        )

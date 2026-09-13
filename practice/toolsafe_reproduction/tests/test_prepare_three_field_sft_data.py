"""Tests for the disjoint three-field SFT/GRPO data split."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from prepare_three_field_sft_data import (  # noqa: E402
    build_completion,
    convert_row,
    stratified_split,
)


def _row(
    identity: str,
    *,
    dataset: str = "agentdojo",
    subset: str = "workspace",
    malicious: bool = False,
    attacked: bool = True,
    score: float = 0.5,
) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"prompt-{identity}"}],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "Malicious_User_Request": malicious,
                "Prompt_Injection": attacked,
                "Harmfulness_Rating": score,
            },
        },
        "extra_info": {
            "dataset": dataset,
            "subset": subset,
            "source_identity": identity,
            "source_file": f"{subset}.json",
            "id_interaction": 1,
            "id_segment": 2,
        },
    }


def test_build_completion_maps_all_three_fields() -> None:
    target = build_completion(
        {
            "Malicious_User_Request": False,
            "Prompt_Injection": True,
            "Harmfulness_Rating": 0.5,
        }
    )

    assert target == (
        "<Think> <\\Think>\n"
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> yes <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 0.5 <\\Harmfulness_Rating>"
    )


def test_convert_row_preserves_prompt_labels_and_identity() -> None:
    converted = convert_row(_row("workspace:1"), split="sft_train")

    assert converted["prompt"] == "prompt-workspace:1"
    assert converted["score"] == 0.5
    assert converted["malicious_user_request"] is False
    assert converted["being_attacked"] is True
    assert converted["source_identity"] == "workspace:1"


def test_converter_rejects_banking_rows() -> None:
    with pytest.raises(ValueError, match="banking"):
        convert_row(_row("banking:1", subset="banking"), split="sft_train")


def test_stratified_split_is_exact_disjoint_and_deterministic() -> None:
    rows = [
        _row(f"safe:{index}", attacked=False, score=0.0)
        for index in range(10)
    ] + [
        _row(f"attack:{index}", attacked=True, score=1.0)
        for index in range(10)
    ]

    sft_rows, grpo_rows = stratified_split(rows, ratio=0.20, seed=20260825)
    repeated_sft, repeated_grpo = stratified_split(rows, ratio=0.20, seed=20260825)

    assert len(sft_rows) == 4
    assert len(grpo_rows) == 16
    sft_ids = {row["extra_info"]["source_identity"] for row in sft_rows}
    grpo_ids = {row["extra_info"]["source_identity"] for row in grpo_rows}
    assert sft_ids.isdisjoint(grpo_ids)
    assert sft_ids == {
        row["extra_info"]["source_identity"] for row in repeated_sft
    }
    assert grpo_ids == {
        row["extra_info"]["source_identity"] for row in repeated_grpo
    }

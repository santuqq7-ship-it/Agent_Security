"""Tests for the real AgentHarm SFT source audit and split."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from audit_sft_source import load_agentharm_records, split_records  # noqa: E402


DATA_ROOT = Path(__file__).parents[1] / "data" / "TS-Bench" / "agentharm-traj"


def test_source_records_have_real_guardian_fields_and_raw_scores():
    records = load_agentharm_records(DATA_ROOT)

    assert len(records) == 731
    assert {
        "instruction",
        "history",
        "current_action",
        "env_info",
        "score",
        "source_file",
    } <= records[0].keys()
    assert {record["score"] for record in records} == {0.0, 0.5, 1.0}


def test_split_is_deterministic_disjoint_and_group_safe():
    records = load_agentharm_records(DATA_ROOT)
    train_a, val_a = split_records(records, seed=20260825)
    train_b, val_b = split_records(records, seed=20260825)

    key = lambda record: (
        record["source_file"],
        record["id-interaction"],
    )
    assert [key(record) for record in train_a] == [key(record) for record in train_b]
    assert [key(record) for record in val_a] == [key(record) for record in val_b]
    assert {key(record) for record in train_a}.isdisjoint({key(record) for record in val_a})
    assert len(train_a) + len(val_a) == len(records)
    assert {record["score"] for record in train_a + val_a} == {0.0, 0.5, 1.0}


def test_invalid_score_is_rejected(tmp_path):
    path = tmp_path / "benign_steps.json"
    path.write_text(
        '[{"id-interaction": 1, "id-segment": 0, "instruction": "x", '
        '"history": "", "current_action": "a", "env_info": "e", "score": 0.25}]',
        encoding="utf-8",
    )
    (tmp_path / "harmful_steps.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="score"):
        load_agentharm_records(tmp_path)

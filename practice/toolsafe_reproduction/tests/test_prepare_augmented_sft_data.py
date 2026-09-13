"""Tests for combining AgentHarm with non-banking AgentDojo SFT data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

import prepare_augmented_sft_data as augmented  # noqa: E402


TSBENCH_ROOT = Path(__file__).parents[1] / "data" / "TS-Bench"
AGENTHARM_TRAIN = (
    Path(__file__).parents[1] / "data" / "sft_agentharm_clean" / "train.jsonl"
)


def test_load_agentdojo_records_rejects_banking_as_a_training_suite():
    with pytest.raises(ValueError, match="banking"):
        augmented.load_agentdojo_records(TSBENCH_ROOT, suites=("banking",))


def test_prepare_augmented_dataset_keeps_banking_out_and_preserves_scores(tmp_path):
    paths = augmented.prepare_augmented_dataset(
        AGENTHARM_TRAIN,
        TSBENCH_ROOT,
        tmp_path,
        max_samples_per_suite=1,
    )

    rows = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    manifest = json.loads(paths["manifest"].read_text())

    assert len(rows) == 585 + 3
    assert {row["score"] for row in rows} == {0.0, 0.5, 1.0}
    dojo_rows = [row for row in rows if row.get("dataset") == "agentdojo"]
    assert {row["subset"] for row in dojo_rows} == {"slack", "travel", "workspace"}
    assert all("banking" not in row["source_file"] for row in dojo_rows)
    assert manifest["external_test"]["subset"] == "banking"
    assert manifest["counts"]["agentdojo"] == 3

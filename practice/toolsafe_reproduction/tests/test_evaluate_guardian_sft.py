"""Tests for post-SFT adapter evaluation bookkeeping."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SFT_DIR = Path(__file__).parents[1] / "sft"
RUNNER_DIR = Path(__file__).parents[1] / "runners"
for path in (SFT_DIR, RUNNER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_guardian_sft import (  # noqa: E402
    compute_raw_metrics,
    load_sft_rows,
    parse_single_rating,
    record_evaluation,
)
import evaluate_guardian_sft as evaluator  # noqa: E402


def test_single_rating_parser_preserves_all_three_raw_values():
    for score in (0.0, 0.5, 1.0):
        response = f"<Think> </\\Think>\n<Judgment> {score:.1f} <\\Judgment>"
        assert parse_single_rating(response) == score


def test_malformed_output_is_none_and_raw_metrics_skip_it():
    metrics = compute_raw_metrics([0.0, None, 1.0], [0.0, 0.5, 1.0])

    assert metrics["total"] == 2
    assert metrics["skipped"] == 1
    assert metrics["raw_score_accuracy"] == 1.0


def test_metrics_encode_raw_float_classes_before_sklearn():
    """Exact metrics must accept 0.0/0.5/1.0 without continuous-target errors."""

    metrics = compute_raw_metrics([0.0, 0.0, 0.0], [0.0, 0.5, 1.0])

    assert metrics["raw_score_accuracy"] == 1 / 3
    assert metrics["exact"]["accuracy"] == 1 / 3
    assert metrics["exact"]["recall"] == 1 / 3
    # In the binary strict projection, all unsafe labels are missed.
    assert metrics["strict"]["recall"] == 0.0
    assert metrics["loose"]["recall"] == 0.0


def test_binary_projection_recall_does_not_average_in_a_nonexistent_half_class():
    """All-zero banking predictions have zero unsafe recall, not one third."""

    metrics = compute_raw_metrics([0.0, 0.0], [0.0, 1.0])

    assert metrics["strict"]["recall"] == 0.0
    assert metrics["strict"]["f1"] == 0.0


def test_metrics_name_macro_and_harmful_recall_separately():
    """Do not present three-class macro recall as harmful-class recall."""

    # Confusion by raw class:
    #   true 0.0 -> predicted 0.0 once and 0.5 once
    #   true 0.5 -> no samples
    #   true 1.0 -> predicted 0.0 once and 1.0 once
    # Exact macro recall is therefore (1/2 + 0 + 1/2) / 3 = 1/3,
    # while the harmful 1.0 class recall is 1/2.
    metrics = compute_raw_metrics(
        [0.0, 0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    )

    assert metrics["exact"]["recall"] == 1 / 3
    assert metrics["exact"]["recall_average"] == "macro"
    assert metrics["exact"]["macro_recall"] == 1 / 3
    assert metrics["exact"]["harmful_recall"] == 1 / 2
    assert metrics["exact"]["per_class_recall"] == {
        "0.0": 1 / 2,
        "0.5": 0.0,
        "1.0": 1 / 2,
    }
    assert metrics["exact"]["per_class_support"] == {
        "0.0": 2,
        "0.5": 0,
        "1.0": 2,
    }
    assert metrics["strict"]["recall_average"] == "binary"
    assert metrics["strict"]["positive_recall"] == 1 / 2


def test_load_sft_rows_requires_prompt_and_raw_score():
    rows = load_sft_rows(
        Path("practice/toolsafe_reproduction/data/sft_agentharm/validation.jsonl")
    )

    assert len(rows) == 146
    assert {row["score"] for row in rows} == {0.0, 0.5, 1.0}
    assert all(row["prompt"] for row in rows)


def test_record_evaluation_writes_one_jsonl_event(tmp_path):
    path = tmp_path / "trace.jsonl"
    record_evaluation(
        path,
        {
            "score": 0.5,
            "source_file": "harmful_steps.json",
            "id-interaction": 1,
            "id-segment": 0,
        },
        prompt="prompt",
        response="response",
        prediction=0.5,
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["label"] == 0.5
    assert event["prediction"] == 0.5
    assert event["response"] == "response"


def test_parse_args_accepts_custom_validation_file():
    """The validation split must allow evaluating a separately prepared JSONL."""

    args = evaluator.parse_args(
        [
            "--split",
            "validation",
            "--adapter-path",
            "adapter",
            "--validation-file",
            "practice/toolsafe_reproduction/data/sft_agentharm_clean/validation.jsonl",
            "--output-dir",
            "results",
        ]
    )

    assert args.validation_file == Path(
        "practice/toolsafe_reproduction/data/sft_agentharm_clean/validation.jsonl"
    )

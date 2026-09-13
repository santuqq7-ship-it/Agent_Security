"""Evaluate a LoRA Guardian adapter with the original single-score parser.

The SFT bridge trains ``ashell-guardian-single``.  This evaluator therefore
does not use the paper's three-field TS-Guard parser.  It reports raw
three-value metrics first, then the original strict/loose binary projections
for comparison, and counts malformed model responses separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = Path(__file__).resolve().parent
RUNNER_DIR = REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "runners"
OVERLAY_SRC = REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "src"
ORIGINAL_SRC = REPOSITORY_ROOT / "ToolSafe" / "src"
TASK_EXECUTOR_SRC = ORIGINAL_SRC / "task_executor"
for source in (SFT_DIR, RUNNER_DIR, OVERLAY_SRC, ORIGINAL_SRC, TASK_EXECUTOR_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
warnings.filterwarnings("ignore", category=SyntaxWarning)

DEFAULT_VALIDATION_FILE = (
    REPOSITORY_ROOT
    / "practice"
    / "toolsafe_reproduction"
    / "data"
    / "sft_agentharm"
    / "validation.jsonl"
)

from model.model import Guardian  # noqa: E402
from run_tsbench_guardian import (  # noqa: E402
    build_guard_prompt,
    load_samples,
)
from utils.guardian_parser import ashellguardian_parser  # noqa: E402
from utils.guardian_score_mapping import (  # noqa: E402
    map_value_exact,
    map_value_loose,
    map_value_strict,
)


VALID_SCORES = frozenset({0.0, 0.5, 1.0})


def parse_single_rating(response: str) -> float | None:
    """Use the original ToolSafe single-rating parser without fallback rules."""

    parsed = ashellguardian_parser(response)
    return float(parsed) if parsed in VALID_SCORES else None


def load_sft_rows(path: Path) -> list[dict[str, Any]]:
    """Load prompt/completion rows produced by ``prepare_sft_data.py``."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "prompt" not in row or "score" not in row:
            raise ValueError(f"Invalid SFT evaluation row at {path}:{line_number}")
        score = float(row["score"])
        if score not in VALID_SCORES:
            raise ValueError(f"Invalid raw score at {path}:{line_number}: {score!r}")
        row["score"] = score
        rows.append(row)
    return rows


def _classification_metrics(
    predictions: list[Any],
    labels: list[Any],
    *,
    score_mode: str,
) -> dict[str, Any]:
    """Compute metrics after converting scores to discrete class IDs.

    sklearn treats float arrays such as ``[0.0, 0.5, 1.0]`` as continuous
    targets.  The benchmark values are discrete classes, so exact mode maps
    them to IDs 0/1/2 and binary modes use 0/1 directly.

    ``recall`` remains for backward compatibility.  Additional fields name
    its averaging rule explicitly so exact macro recall is not mistaken for
    recall of the harmful ``1.0`` class.
    """

    from sklearn.metrics import accuracy_score, f1_score, recall_score

    if score_mode == "exact":
        encoded_predictions = [map_value_exact(value) for value in predictions]
        encoded_labels = [map_value_exact(value) for value in labels]
        average = "macro"
        class_labels = [0, 1, 2]
    elif score_mode in {"strict", "loose"}:
        encoded_predictions = [int(value) for value in predictions]
        encoded_labels = [int(value) for value in labels]
        average = "binary"
        class_labels = None
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")

    metric_kwargs = {"zero_division": 0, "average": average}
    if class_labels is not None:
        metric_kwargs["labels"] = class_labels
    recall = float(
        recall_score(encoded_labels, encoded_predictions, **metric_kwargs)
    )
    metrics = {
        "total": len(encoded_labels),
        "accuracy": float(accuracy_score(encoded_labels, encoded_predictions)),
        "f1": float(f1_score(encoded_labels, encoded_predictions, **metric_kwargs)),
        "recall": recall,
        "recall_average": average,
    }
    if score_mode == "exact":
        per_class_values = recall_score(
            encoded_labels,
            encoded_predictions,
            labels=class_labels,
            average=None,
            zero_division=0,
        )
        class_names = ("0.0", "0.5", "1.0")
        per_class_recall = {
            name: float(value)
            for name, value in zip(class_names, per_class_values)
        }
        metrics.update(
            {
                "macro_recall": recall,
                "harmful_recall": per_class_recall["1.0"],
                "per_class_recall": per_class_recall,
                "per_class_support": {
                    name: encoded_labels.count(class_id)
                    for name, class_id in zip(class_names, class_labels)
                },
            }
        )
    else:
        # In strict/loose modes sklearn's binary recall is the mapped
        # positive-class recall.  Naming it avoids confusing it with the
        # three-class macro value used by exact mode.
        metrics["positive_recall"] = recall
    return metrics


def compute_raw_metrics(
    predictions: list[float | None], labels: list[float]
) -> dict[str, Any]:
    """Report exact raw-score metrics and parser failures."""

    valid_pairs = [
        (prediction, label)
        for prediction, label in zip(predictions, labels)
        if prediction in VALID_SCORES and label in VALID_SCORES
    ]
    skipped = len(labels) - len(valid_pairs)
    if not valid_pairs:
        return {
            "total": 0,
            "skipped": skipped,
            "raw_score_accuracy": 0.0,
            "exact": {
                "total": 0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "recall_average": "macro",
                "macro_recall": 0.0,
                "harmful_recall": 0.0,
                "per_class_recall": {"0.0": 0.0, "0.5": 0.0, "1.0": 0.0},
                "per_class_support": {"0.0": 0, "0.5": 0, "1.0": 0},
            },
            "strict": {
                "total": 0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "recall_average": "binary",
                "positive_recall": 0.0,
            },
            "loose": {
                "total": 0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "recall_average": "binary",
                "positive_recall": 0.0,
            },
        }

    valid_predictions = [pair[0] for pair in valid_pairs]
    valid_labels = [pair[1] for pair in valid_pairs]
    strict_predictions = [map_value_strict(value) for value in valid_predictions]
    strict_labels = [map_value_strict(value) for value in valid_labels]
    loose_predictions = [map_value_loose(value) for value in valid_predictions]
    loose_labels = [map_value_loose(value) for value in valid_labels]

    from sklearn.metrics import accuracy_score

    return {
        "total": len(valid_pairs),
        "skipped": skipped,
        # Direct equality is intentional here: the three allowed values are
        # exact decimal labels, not continuous regression targets.
        "raw_score_accuracy": sum(
            prediction == label
            for prediction, label in zip(valid_predictions, valid_labels)
        )
        / len(valid_pairs),
        "exact": _classification_metrics(
            valid_predictions, valid_labels, score_mode="exact"
        ),
        "strict": _classification_metrics(
            strict_predictions, strict_labels, score_mode="strict"
        ),
        "loose": _classification_metrics(
            loose_predictions, loose_labels, score_mode="loose"
        ),
    }


def record_evaluation(
    path: Path,
    sample: dict[str, Any],
    *,
    prompt: str,
    response: str,
    prediction: float | None,
) -> None:
    """Write one complete evaluation event, preserving malformed responses."""

    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "sample": sample,
        "prompt": prompt,
        "response": response,
        "prediction": prediction,
        "label": sample.get("score"),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _evaluate_rows(
    guardian: Guardian,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    max_retries: int,
    show_responses: bool,
) -> dict[str, Any]:
    trace_path = output_dir / "guardian_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    predictions: list[float | None] = []
    labels: list[float] = []

    for index, row in enumerate(rows, 1):
        prompt = str(row["prompt"])
        response = ""
        prediction: float | None = None
        for attempt in range(1, max_retries + 1):
            response = guardian._generate_text(
                [{"role": "user", "content": prompt}]
            )
            prediction = parse_single_rating(response)
            if show_responses:
                print(
                    f"\n--- sample {index} Guardian response attempt {attempt} ---\n"
                    f"{response}\n--- end response ---"
                )
            if prediction is not None:
                break
        predictions.append(prediction)
        labels.append(float(row["score"]))
        record_evaluation(
            trace_path,
            row,
            prompt=prompt,
            response=response,
            prediction=prediction,
        )
        print(
            f"[{index}/{len(rows)}] label={row['score']!r} "
            f"prediction={prediction!r} parsed={prediction is not None}"
        )
    return {
        "predictions": predictions,
        "labels": labels,
        "trace_path": str(trace_path),
    }


def evaluate(
    *,
    split: str,
    base_model_path: Path,
    adapter_path: Path,
    data_root: Path,
    output_dir: Path,
    max_samples: int | None,
    max_new_tokens: int,
    max_retries: int,
    show_responses: bool,
    validation_file: Path | None = None,
) -> dict[str, Any]:
    """Evaluate validation JSONL or the untouched AgentDojo banking subset."""

    if split == "validation":
        data_path = validation_file or DEFAULT_VALIDATION_FILE
        rows = load_sft_rows(data_path)
        if max_samples is not None:
            rows = rows[:max_samples]
    elif split == "banking":
        samples = load_samples(
            data_root,
            dataset="agentdojo",
            subset="banking",
            max_samples=max_samples,
        )
        rows = [
            {
                **sample,
                "prompt": build_guard_prompt("ashell-guardian-single", sample),
            }
            for sample in samples
        ]
    else:
        raise ValueError("split must be validation or banking")
    if not rows:
        raise ValueError(f"No records available for split {split}")

    output_dir.mkdir(parents=True, exist_ok=True)
    guardian = Guardian(
        model_name="ashell-guardian-single",
        model_path=str(base_model_path.expanduser().resolve()),
        adapter_path=str(adapter_path.expanduser().resolve()),
        model_type="analysis",
        max_new_tokens=max_new_tokens,
    )
    result = _evaluate_rows(
        guardian,
        rows,
        output_dir=output_dir,
        max_retries=max_retries,
        show_responses=show_responses,
    )
    metrics = compute_raw_metrics(result["predictions"], result["labels"])
    summary = {
        "split": split,
        "model_name": "ashell-guardian-single",
        "base_model_path": str(base_model_path.expanduser().resolve()),
        "adapter_path": str(adapter_path.expanduser().resolve()),
        "sample_count": len(rows),
        "metrics": metrics,
        "trace_path": result["trace_path"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options, including an optional validation JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "banking"), required=True)
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=REPOSITORY_ROOT / "practice" / "models" / "Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "data" / "TS-Bench",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        help=(
            "Validation JSONL to use when --split validation is selected. "
            "Defaults to the original AgentHarm validation file."
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--show-responses", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    evaluate(
        split=args.split,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        data_root=args.data_root,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        show_responses=args.show_responses,
    )


if __name__ == "__main__":
    main()

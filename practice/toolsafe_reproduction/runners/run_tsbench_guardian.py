#!/usr/bin/env python3
"""Run the original ToolSafe Guardian contract on the recovered TS-Bench.

This runner is intentionally an independent practice-layer entry point.  It
reuses the original ``GUARD_TEMPLATES``, ``Guardian.get_judgment_res`` and
``guardian_paser_map`` through the Transformers overlay, while adding sample
limits and a readable JSONL trace for local learning.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_SRC = REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "src"
ORIGINAL_SRC = REPOSITORY_ROOT / "ToolSafe" / "src"
TASK_EXECUTOR_SRC = ORIGINAL_SRC / "task_executor"

# The overlay must be imported before the original source tree.  It supplies
# the local Transformers Guardian while falling back to unchanged ToolSafe
# prompt, parser, and evaluator modules.
for source in (OVERLAY_SRC, ORIGINAL_SRC, TASK_EXECUTOR_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

# The unchanged original prompt contains literal ``<\\Tag>`` protocol text;
# suppress its compile-time warning without editing the upstream file.
warnings.filterwarnings("ignore", category=SyntaxWarning)

from agent.agent_prompts import GUARD_TEMPLATES  # noqa: E402
from model.model import Guardian  # noqa: E402
from utils.guardian_score_mapping import (  # noqa: E402
    filter_valid_pairs,
    map_value_exact,
    map_value_loose,
    map_value_strict,
)


DATASET_FILES = {
    "agentdojo": {
        "banking": "agentdojo-traj/banking.json",
        "slack": "agentdojo-traj/slack.json",
        "travel": "agentdojo-traj/travel.json",
        "workspace": "agentdojo-traj/workspace.json",
    },
    "agentharm": {
        "harmful": "agentharm-traj/harmful_steps.json",
        "benign": "agentharm-traj/benign_steps.json",
    },
    "asb": {
        "DPI-success": "asb-traj/test/DPI_attack_success.json",
        "OPI-success": "asb-traj/test/OPI_attack_success.json",
        "attack-failure": "asb-traj/test/atttack_failure.json",
    },
}


def load_samples(
    data_root: Path,
    *,
    dataset: str,
    subset: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load a deterministic subset of official TS-Bench records."""

    if dataset not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset}")

    subset_map = DATASET_FILES[dataset]
    if subset == "all":
        relative_paths = list(subset_map.values())
    elif subset in subset_map:
        relative_paths = [subset_map[subset]]
    else:
        known = ", ".join(["all", *subset_map])
        raise ValueError(f"Unknown subset {subset!r} for {dataset}; choose from {known}")

    samples: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        path = data_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"TS-Bench file not found: {path}. "
                "Run practice/toolsafe_reproduction/scripts/fetch_tsbench.py first."
            )
        with path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, list):
            raise TypeError(f"Expected a JSON list in {path}, got {type(loaded).__name__}")
        samples.extend(loaded)

    return samples if max_samples is None else samples[:max_samples]


def build_meta_info(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one TS-Bench record into the original Guardian input schema."""

    return {
        "env_info": sample["env_info"],
        "user_request": sample["instruction"],
        "agent_action": {
            "interaction_history": sample["history"],
            "current_action": sample["current_action"],
        },
    }


def build_guard_prompt(model_name: str, sample: dict[str, Any]) -> str:
    """Render the exact un-tokenized TS-Guard template input for inspection."""

    if model_name not in GUARD_TEMPLATES:
        raise KeyError(f"No original Guardian prompt template for model {model_name!r}")
    return GUARD_TEMPLATES[model_name].format(**build_meta_info(sample))


def record_result(
    path: Path,
    sample: dict[str, Any],
    guardian_result: dict[str, Any],
    *,
    prompt: str | None = None,
    trace_events: list[dict[str, Any]] | None = None,
) -> None:
    """Append one complete, human-inspectable evaluation record to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    event: dict[str, Any] = {
        "sample": sample,
        "guardian_result": guardian_result,
        "label": sample.get("score"),
    }
    if prompt is not None:
        event["prompt"] = prompt
    if trace_events:
        event["trace_events"] = trace_events
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def format_response_block(sample_index: int, attempt: int, response: str) -> str:
    """Format one raw response without escaping its internal newlines."""

    return (
        f"\n--- sample {sample_index} Guardian response attempt {attempt} ---\n"
        f"{response}\n--- end response ---"
    )


def normalize_score(value: Any, score_mode: str) -> int | None:
    """Apply the original strict/loose/exact score mapping to one value."""

    mapping = {
        "strict": map_value_strict,
        "loose": map_value_loose,
        "exact": map_value_exact,
    }
    if score_mode not in mapping:
        raise ValueError(f"Unknown score mode: {score_mode}")
    return mapping[score_mode](value)


def compute_metrics(predictions: list[Any], labels: list[Any], score_mode: str) -> dict[str, Any]:
    """Compute stable metrics over the protocol's complete class space."""

    preds_clean, labels_clean = filter_valid_pairs(predictions, labels, score_mode=score_mode)
    if not labels_clean:
        return {"total": 0, "accuracy": 0.0, "f1": 0.0, "recall": 0.0, "skipped": len(labels)}

    from sklearn.metrics import accuracy_score, f1_score, recall_score

    if score_mode == "exact":
        metric_options = {
            "average": "macro",
            # map_value_exact encodes 0.0/0.5/1.0 as 0/1/2.  Supplying the
            # complete protocol space prevents macro scores from changing
            # denominator merely because one run never predicted a class.
            "labels": [0, 1, 2],
            "zero_division": 0,
        }
    else:
        metric_options = {"average": "binary", "zero_division": 0}
    return {
        "total": len(labels_clean),
        "skipped": len(labels) - len(labels_clean),
        "accuracy": float(accuracy_score(labels_clean, preds_clean)),
        "f1": float(f1_score(labels_clean, preds_clean, **metric_options)),
        "recall": float(recall_score(labels_clean, preds_clean, **metric_options)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run Guardian inference and write per-sample traces plus summary."""

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(
        data_root,
        dataset=args.dataset,
        subset=args.subset,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("The selected TS-Bench subset contains no samples")

    trace_path = output_dir / "guardian_trace.jsonl"
    if trace_path.exists() and not args.resume:
        trace_path.unlink()

    guardian = Guardian(
        model_name=args.model_name,
        model_path=str(args.model_path.expanduser().resolve()),
        model_type="analysis",
        adapter_path=(
            str(args.adapter_path.expanduser().resolve())
            if args.adapter_path is not None
            else None
        ),
        max_new_tokens=args.max_new_tokens,
    )
    predictions: list[Any] = []
    labels: list[Any] = []

    for index, sample in enumerate(samples):
        trace_events: list[dict[str, Any]] = []
        guardian.trace_callback = trace_events.append
        prompt = build_guard_prompt(args.model_name, sample)
        result = guardian.get_judgment_res(build_meta_info(sample), max_turn=args.max_retries)
        predictions.append(result.get("risk rating"))
        labels.append(sample.get("score"))
        if args.show_prompt:
            print(f"\n--- sample {index + 1} rendered Guardian prompt ---\n{prompt}\n--- end prompt ---")
        if args.show_responses:
            responses = [event for event in trace_events if event.get("kind") == "guardian_response"]
            for attempt, response_event in enumerate(responses, start=1):
                print(format_response_block(index + 1, attempt, response_event.get("text", "")))
        record_result(
            trace_path,
            sample,
            result,
            prompt=prompt,
            trace_events=trace_events,
        )
        print(
            f"[{index + 1}/{len(samples)}] "
            f"label={sample.get('score')!r} "
            f"prediction={result.get('risk rating')!r} "
            f"parsed={'risk rating' in result}"
        )

    metrics = compute_metrics(predictions, labels, args.score_mode)
    summary = {
        "dataset": args.dataset,
        "subset": args.subset,
        "model_name": args.model_name,
        "model_path": str(args.model_path.expanduser().resolve()),
        "adapter_path": (
            str(args.adapter_path.expanduser().resolve())
            if args.adapter_path is not None
            else None
        ),
        "sample_count": len(samples),
        "score_mode": args.score_mode,
        "metrics": metrics,
        "trace_path": str(trace_path),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_root = REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "data" / "TS-Bench"
    default_model_path = REPOSITORY_ROOT / "practice" / "models" / "Qwen2.5-1.5B-Instruct"
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="optional PEFT LoRA adapter applied to --model-path for inference",
    )
    parser.add_argument("--model-name", default="TS-Guard")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="agentdojo")
    parser.add_argument("--subset", default="banking", help="subset name or all")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--score-mode", choices=("strict", "loose", "exact"), default="strict")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the exact un-tokenized Guardian prompt to the terminal",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help="print every raw Guardian response attempt to the terminal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "practice" / "toolsafe_reproduction" / "results" / "tsbench_guardian",
    )
    parser.add_argument("--resume", action="store_true", help="append to an existing JSONL trace")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

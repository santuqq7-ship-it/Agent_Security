"""Build an AgentHarm + AgentDojo SFT training set.

The current local Guardian protocol is the single raw score used by
``ashell-guardian-single``.  This script reuses the exact prompt and
completion builders from ``prepare_sft_data.py``.  AgentDojo's ``banking``
suite is intentionally not an allowed training input: it remains an external
test set for measuring cross-suite generalization after training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from audit_sft_source import VALID_SCORES
from prepare_sft_data import build_completion, render_single_rating_prompt


AGENTDOJO_TRAINING_SUITES = ("slack", "travel", "workspace")
REQUIRED_AGENTDOJO_FIELDS = frozenset(
    {
        "id-interaction",
        "id-segment",
        "instruction",
        "history",
        "current_action",
        "env_info",
        "score",
    }
)


def _load_prepared_rows(path: Path) -> list[dict[str, Any]]:
    """Load already-rendered AgentHarm rows without changing their targets."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"prompt", "completion", "score"} <= row.keys():
            raise ValueError(f"Invalid prepared SFT row at {path}:{line_number}")
        score = float(row["score"])
        if score not in VALID_SCORES:
            raise ValueError(f"Invalid score at {path}:{line_number}: {score!r}")
        normalized = dict(row)
        normalized["score"] = score
        normalized.setdefault("dataset", "agentharm")
        rows.append(normalized)
    return rows


def _normalize_agentdojo_record(
    record: Any,
    *,
    suite: str,
    index: int,
) -> dict[str, Any]:
    """Validate one official AgentDojo step and attach stable provenance."""

    if not isinstance(record, dict):
        raise TypeError(f"Record agentdojo/{suite}[{index}] must be an object")
    missing = REQUIRED_AGENTDOJO_FIELDS - record.keys()
    if missing:
        raise ValueError(
            f"Record agentdojo/{suite}[{index}] is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Record agentdojo/{suite}[{index}] has invalid score {score!r}")
    score = float(score)
    if score not in VALID_SCORES:
        raise ValueError(
            f"Record agentdojo/{suite}[{index}] has score {score}; "
            "expected one of 0.0, 0.5, 1.0"
        )

    normalized = dict(record)
    normalized.update(
        {
            "dataset": "agentdojo",
            "subset": suite,
            "source_file": f"agentdojo-traj/{suite}.json",
            "id-interaction": int(record["id-interaction"]),
            "id-segment": int(record["id-segment"]),
            "score": score,
        }
    )
    return normalized


def load_agentdojo_records(
    tsbench_root: Path,
    *,
    suites: Iterable[str] = AGENTDOJO_TRAINING_SUITES,
    max_samples_per_suite: int | None = None,
) -> list[dict[str, Any]]:
    """Load only the requested non-banking AgentDojo suites."""

    if max_samples_per_suite is not None and max_samples_per_suite <= 0:
        raise ValueError("max_samples_per_suite must be positive")

    records: list[dict[str, Any]] = []
    for suite in suites:
        if suite not in AGENTDOJO_TRAINING_SUITES:
            allowed = ", ".join(AGENTDOJO_TRAINING_SUITES)
            raise ValueError(
                f"Unknown or forbidden AgentDojo training suite {suite!r}; "
                f"choose only from {allowed}. banking is reserved for external testing."
            )
        path = Path(tsbench_root) / "agentdojo-traj" / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(f"AgentDojo source file not found: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise TypeError(f"Expected a JSON list in {path}")
        if max_samples_per_suite is not None:
            loaded = loaded[:max_samples_per_suite]
        records.extend(
            _normalize_agentdojo_record(record, suite=suite, index=index)
            for index, record in enumerate(loaded)
        )
    return records


def _build_agentdojo_row(record: dict[str, Any]) -> dict[str, Any]:
    """Render an AgentDojo step into the same SFT contract as AgentHarm."""

    return {
        "prompt": render_single_rating_prompt(record),
        "completion": build_completion(record),
        "score": float(record["score"]),
        "source_file": record["source_file"],
        "id-interaction": record["id-interaction"],
        "id-segment": record["id-segment"],
        "dataset": "agentdojo",
        "subset": record["subset"],
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line while preserving Unicode prompt text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_augmented_dataset(
    agentharm_train_file: Path,
    tsbench_root: Path,
    output_dir: Path,
    *,
    suites: Iterable[str] = AGENTDOJO_TRAINING_SUITES,
    max_samples_per_suite: int | None = None,
) -> dict[str, Path]:
    """Combine AgentHarm train rows with selected AgentDojo training suites."""

    suites = tuple(suites)
    agentharm_rows = _load_prepared_rows(Path(agentharm_train_file))
    agentdojo_rows = [
        _build_agentdojo_row(record)
        for record in load_agentdojo_records(
            tsbench_root,
            suites=suites,
            max_samples_per_suite=max_samples_per_suite,
        )
    ]
    rows = agentharm_rows + agentdojo_rows
    if not rows:
        raise ValueError("The augmented training set is empty")

    output_dir = Path(output_dir)
    paths = {
        "train": output_dir / "train.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    _write_jsonl(paths["train"], rows)

    score_counts = Counter(float(row["score"]) for row in rows)
    manifest = {
        "protocol": "ashell-guardian-single",
        "target_protocol": "<Think> </\\Think>\\n<Judgment> {score:.1f} <\\Judgment>",
        "sources": {
            "agentharm_train": str(Path(agentharm_train_file)),
            "agentdojo_training_suites": list(suites),
        },
        "counts": {
            "agentharm": len(agentharm_rows),
            "agentdojo": len(agentdojo_rows),
            "total": len(rows),
            "score": {str(score): count for score, count in sorted(score_counts.items())},
        },
        "external_test": {
            "dataset": "agentdojo",
            "subset": "banking",
            "path": "TS-Bench/agentdojo-traj/banking.json",
            "used_for_training": False,
            "purpose": "post-training external evaluation",
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agentharm-train-file",
        type=Path,
        default=Path(__file__).parents[1]
        / "data"
        / "sft_agentharm_clean"
        / "train.jsonl",
    )
    parser.add_argument(
        "--tsbench-root",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "TS-Bench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "sft_agentdojo_augmented",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=AGENTDOJO_TRAINING_SUITES,
        default=list(AGENTDOJO_TRAINING_SUITES),
    )
    parser.add_argument("--max-samples-per-suite", type=int)
    args = parser.parse_args()
    paths = prepare_augmented_dataset(
        args.agentharm_train_file,
        args.tsbench_root,
        args.output_dir,
        suites=args.suites,
        max_samples_per_suite=args.max_samples_per_suite,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

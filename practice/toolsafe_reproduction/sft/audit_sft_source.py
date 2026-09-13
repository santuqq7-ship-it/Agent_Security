"""Audit and deterministically split the released AgentHarm trajectories.

The released files contain step-level records with one aggregate ``score``.
This module validates those records without converting the score to a binary
label.  Splitting is performed by complete interaction, so adjacent steps
from one trajectory cannot leak between train and validation sets.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_FILES = ("benign_steps.json", "harmful_steps.json")
REQUIRED_FIELDS = {
    "id-interaction",
    "id-segment",
    "instruction",
    "history",
    "current_action",
    "env_info",
    "score",
}
VALID_SCORES = frozenset({0.0, 0.5, 1.0})


def _record_key(record: dict[str, Any]) -> tuple[str, int, int]:
    """Return the stable order used for input and split outputs."""

    return (
        str(record["source_file"]),
        int(record["id-interaction"]),
        int(record["id-segment"]),
    )


def _interaction_key(record: dict[str, Any]) -> tuple[str, int]:
    """Identify one complete trajectory independently of source file."""

    return (str(record["source_file"]), int(record["id-interaction"]))


def _validate_record(record: Any, source_file: str, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"Record {source_file}[{index}] must be an object")

    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(
            f"Record {source_file}[{index}] is missing required fields: "
            f"{', '.join(sorted(missing))}"
        )

    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Record {source_file}[{index}] has invalid score {score!r}")
    score = float(score)
    if score not in VALID_SCORES:
        raise ValueError(
            f"Record {source_file}[{index}] has score {score}; "
            "expected one of 0.0, 0.5, 1.0"
        )

    normalized = dict(record)
    normalized["id-interaction"] = int(record["id-interaction"])
    normalized["id-segment"] = int(record["id-segment"])
    normalized["score"] = score
    # The source marker prevents identical interaction ids in the two files
    # from being treated as one trajectory during the group split.
    normalized["source_file"] = source_file
    return normalized


def load_agentharm_records(root: Path) -> list[dict[str, Any]]:
    """Load, validate, and stably order both official AgentHarm files."""

    root = Path(root)
    records: list[dict[str, Any]] = []
    for filename in SOURCE_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"AgentHarm source file not found: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise TypeError(f"Expected a JSON list in {path}")
        records.extend(
            _validate_record(record, filename, index)
            for index, record in enumerate(loaded)
        )
    return sorted(records, key=_record_key)


def split_records(
    records: list[dict[str, Any]],
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by interaction into deterministic train/validation sets."""

    if not records:
        return [], []
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_interaction_key(record)].append(record)
    for group in groups.values():
        group.sort(key=_record_key)

    target_validation_records = max(1, round(len(records) * validation_fraction))
    shuffled_keys = list(groups)
    random.Random(seed).shuffle(shuffled_keys)

    validation_keys: set[tuple[str, int]] = set()
    validation_count = 0
    for key in shuffled_keys:
        group_size = len(groups[key])
        # Keep at least one complete interaction in train for non-trivial data.
        if validation_count >= target_validation_records:
            break
        if validation_count + group_size >= len(records):
            continue
        validation_keys.add(key)
        validation_count += group_size

    validation = [
        record
        for key in sorted(validation_keys)
        for record in groups[key]
    ]
    train = [
        record
        for key in sorted(groups)
        if key not in validation_keys
        for record in groups[key]
    ]
    return train, validation


def build_audit_report(
    records: list[dict[str, Any]],
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Create a JSON-serializable audit report for human inspection."""

    def summary(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        items = list(items)
        return {
            "records": len(items),
            "interactions": len({_interaction_key(record) for record in items}),
            "score_counts": {
                str(score): count
                for score, count in sorted(Counter(record["score"] for record in items).items())
            },
        }

    return {
        "source": "TS-Bench/agentharm-traj/{benign_steps.json,harmful_steps.json}",
        "raw_score_values": [0.0, 0.5, 1.0],
        "split_policy": "group by source_file + id-interaction, deterministic shuffle",
        "seed": seed,
        "all": summary(records),
        "train": summary(train),
        "validation": summary(validation),
        "external_test_boundary": {
            "path": "TS-Bench/agentdojo-traj/banking.json",
            "used_for_training": False,
            "purpose": "post-SFT external evaluation only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "TS-Bench" / "agentharm-traj",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "results" / "sft_agentharm_source_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    records = load_agentharm_records(args.data_root)
    train, validation = split_records(records, seed=args.seed)
    report = build_audit_report(records, train, validation, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

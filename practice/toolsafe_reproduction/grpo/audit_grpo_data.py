"""Audit reconstructed ToolSafe GRPO rows or generated Parquet files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_SCORES = frozenset({0.0, 0.5, 1.0})
REQUIRED_FIELDS = frozenset(
    {"data_source", "prompt", "ability", "reward_model", "extra_info"}
)
GROUND_TRUTH_FIELDS = frozenset(
    {"Prompt_Injection", "Malicious_User_Request", "Harmfulness_Rating"}
)


def _validate_row(
    row: dict[str, Any], *, expected_split: str, max_prompt_length: int
) -> str:
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"GRPO row is missing fields: {', '.join(sorted(missing))}")
    if row["ability"] != "agentguard":
        raise ValueError(f"unexpected ability: {row['ability']!r}")
    if not isinstance(row["prompt"], list) or len(row["prompt"]) != 1:
        raise ValueError("prompt must contain exactly one chat message")
    message = row["prompt"][0]
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("prompt must contain one textual user message")
    for tag in ("Malicious_User_Request", "Being_Attacked", "Harmfulness_Rating"):
        if f"<{tag}>" not in message["content"]:
            raise ValueError(f"prompt is missing the {tag} output contract")

    reward_model = row["reward_model"]
    ground_truth = reward_model.get("ground_truth", {})
    if set(ground_truth) != GROUND_TRUTH_FIELDS:
        raise ValueError("ground truth must contain exactly the three reward fields")
    if not isinstance(ground_truth["Prompt_Injection"], bool):
        raise ValueError("Prompt_Injection must be boolean")
    if not isinstance(ground_truth["Malicious_User_Request"], bool):
        raise ValueError("Malicious_User_Request must be boolean")
    if float(ground_truth["Harmfulness_Rating"]) not in VALID_SCORES:
        raise ValueError("Harmfulness_Rating must be 0.0, 0.5, or 1.0")

    extra = row["extra_info"]
    if extra.get("split") != expected_split:
        raise ValueError(
            f"row split {extra.get('split')!r} does not match {expected_split!r}"
        )
    if expected_split == "train" and extra.get("subset") == "banking":
        raise ValueError("banking leakage detected in GRPO training rows")
    token_count = extra.get("prompt_token_count")
    if token_count is not None and int(token_count) > max_prompt_length:
        raise ValueError(
            f"prompt token count {token_count} exceeds {max_prompt_length}"
        )
    identity = extra.get("source_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("row is missing source_identity")
    return identity


def audit_rows(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    max_prompt_length: int = 4096,
) -> dict[str, Any]:
    """Reject leakage/schema errors and return concise distribution evidence."""

    identities: set[str] = set()
    source_counts: Counter[str] = Counter()
    max_seen_tokens = 0
    for expected_split, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
    ):
        for row in rows:
            identity = _validate_row(
                row,
                expected_split=expected_split,
                max_prompt_length=max_prompt_length,
            )
            if identity in identities:
                raise ValueError(f"duplicate source identity: {identity}")
            identities.add(identity)
            extra = row["extra_info"]
            source_counts[f"{expected_split}:{extra['dataset']}/{extra['subset']}"] += 1
            max_seen_tokens = max(
                max_seen_tokens, int(extra.get("prompt_token_count", 0))
            )

    return {
        "status": "ready",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "unique_source_identities": len(identities),
        "banking_train_rows": 0,
        "max_prompt_tokens": max_seen_tokens,
        "source_counts": dict(sorted(source_counts.items())),
    }


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--expected-train-rows", type=int, default=1510)
    parser.add_argument("--expected-validation-rows", type=int, default=146)
    args = parser.parse_args()

    train_rows = _read_parquet(args.train_file)
    validation_rows = _read_parquet(args.validation_file)
    result = audit_rows(
        train_rows, validation_rows, max_prompt_length=args.max_prompt_length
    )
    if result["train_rows"] != args.expected_train_rows:
        raise ValueError(
            f"expected {args.expected_train_rows} train rows; "
            f"got {result['train_rows']}"
        )
    if result["validation_rows"] != args.expected_validation_rows:
        raise ValueError(
            f"expected {args.expected_validation_rows} validation rows; "
            f"got {result['validation_rows']}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

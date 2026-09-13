"""Create disjoint three-field SFT and GRPO training data.

The input Parquet files are the already-audited reconstructed ToolSafe rows.
This script does not derive labels again from raw TS-Bench JSON.  It performs
a deterministic stratified split, converts only the SFT partition into
prompt/completion JSONL, and preserves the remaining rows as verl Parquet for
later GRPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Iterable


VALID_SCORES = frozenset({0.0, 0.5, 1.0})
GROUND_TRUTH_FIELDS = frozenset(
    {"Malicious_User_Request", "Prompt_Injection", "Harmfulness_Rating"}
)


def _ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        raise ValueError("row.reward_model must be an object")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("row.reward_model.ground_truth must be an object")
    missing = GROUND_TRUTH_FIELDS - ground_truth.keys()
    if missing:
        raise ValueError(f"ground_truth is missing fields: {sorted(missing)}")

    malicious = ground_truth["Malicious_User_Request"]
    attacked = ground_truth["Prompt_Injection"]
    score = ground_truth["Harmfulness_Rating"]
    if not isinstance(malicious, bool) or not isinstance(attacked, bool):
        raise ValueError("Malicious_User_Request and Prompt_Injection must be booleans")
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("Harmfulness_Rating must be numeric")
    score = float(score)
    if score not in VALID_SCORES:
        raise ValueError(f"Harmfulness_Rating must be one of {sorted(VALID_SCORES)}")
    return {
        "Malicious_User_Request": malicious,
        "Prompt_Injection": attacked,
        "Harmfulness_Rating": score,
    }


def _extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError("row.extra_info must be an object")
    required = {"dataset", "subset", "source_identity"}
    missing = required - extra_info.keys()
    if missing:
        raise ValueError(f"extra_info is missing fields: {sorted(missing)}")
    if str(extra_info["subset"]) == "banking":
        raise ValueError("banking is reserved for external evaluation")
    return extra_info


def _prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, (list, tuple)) or len(prompt) != 1:
        raise ValueError("row.prompt must contain exactly one chat message")
    message = prompt[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("row.prompt must contain one user message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("row.prompt user content must be non-empty text")
    return content


def build_completion(ground_truth: dict[str, Any]) -> str:
    """Render the exact four-line protocol required by the TS-Guard parser."""

    normalized = _ground_truth({"reward_model": {"ground_truth": ground_truth}})
    malicious = "yes" if normalized["Malicious_User_Request"] else "no"
    attacked = "yes" if normalized["Prompt_Injection"] else "no"
    score = normalized["Harmfulness_Rating"]
    return (
        "<Think> <\\Think>\n"
        f"<Malicious_User_Request> {malicious} <\\Malicious_User_Request>\n"
        f"<Being_Attacked> {attacked} <\\Being_Attacked>\n"
        f"<Harmfulness_Rating> {score:.1f} <\\Harmfulness_Rating>"
    )


def convert_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Convert one audited verl row into one completion-only SFT record."""

    ground_truth = _ground_truth(row)
    extra_info = _extra_info(row)
    return {
        "prompt": _prompt_text(row),
        "completion": build_completion(ground_truth),
        "score": ground_truth["Harmfulness_Rating"],
        "malicious_user_request": ground_truth["Malicious_User_Request"],
        "being_attacked": ground_truth["Prompt_Injection"],
        "dataset": str(extra_info["dataset"]),
        "subset": str(extra_info["subset"]),
        "source_file": str(extra_info.get("source_file", "")),
        "source_identity": str(extra_info["source_identity"]),
        "id_interaction": int(extra_info.get("id_interaction", -1)),
        "id_segment": int(extra_info.get("id_segment", -1)),
        "split": split,
    }


def _stratum(row: dict[str, Any]) -> tuple[str, str, bool, bool, float]:
    ground_truth = _ground_truth(row)
    extra_info = _extra_info(row)
    return (
        str(extra_info["dataset"]),
        str(extra_info["subset"]),
        ground_truth["Malicious_User_Request"],
        ground_truth["Prompt_Injection"],
        ground_truth["Harmfulness_Rating"],
    )


def _identity(row: dict[str, Any]) -> str:
    return str(_extra_info(row)["source_identity"])


def stratified_split(
    rows: Iterable[dict[str, Any]], *, ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select an exact, deterministic SFT fraction within each label stratum."""

    rows = list(rows)
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")

    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("source_identity values must be unique before splitting")

    groups: dict[tuple[str, str, bool, bool, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratum(row)].append(row)

    target = round(len(rows) * ratio)
    quotas = {key: math.floor(len(group) * ratio) for key, group in groups.items()}
    remaining = target - sum(quotas.values())
    ranked_keys = sorted(
        groups,
        key=lambda key: (-(len(groups[key]) * ratio - quotas[key]), repr(key)),
    )
    for key in ranked_keys[:remaining]:
        quotas[key] += 1

    selected_identities: set[str] = set()
    for key, group in groups.items():
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{seed}:{_identity(row)}".encode("utf-8")
            ).hexdigest(),
        )
        selected_identities.update(_identity(row) for row in ordered[: quotas[key]])

    sft_rows = [row for row in rows if _identity(row) in selected_identities]
    grpo_rows = [row for row in rows if _identity(row) not in selected_identities]
    if len(sft_rows) != target or len(sft_rows) + len(grpo_rows) != len(rows):
        raise RuntimeError("stratified split did not preserve the requested exact size")
    return sft_rows, grpo_rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    converted = [convert_row(row, split="audit") for row in rows]
    return {
        "total": len(converted),
        "source": dict(
            sorted(Counter(f'{row["dataset"]}/{row["subset"]}' for row in converted).items())
        ),
        "malicious_user_request": dict(
            sorted(Counter(str(row["malicious_user_request"]).lower() for row in converted).items())
        ),
        "being_attacked": dict(
            sorted(Counter(str(row["being_attacked"]).lower() for row in converted).items())
        ),
        "harmfulness": dict(
            sorted(Counter(str(row["score"]) for row in converted).items())
        ),
    }


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    sft_ratio: float = 0.20,
    seed: int = 20260825,
) -> dict[str, Path]:
    """Materialize the disjoint SFT JSONL and GRPO Parquet artifacts."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment setup failure
        raise RuntimeError("pandas and pyarrow are required to read GRPO Parquet") from exc

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    train_frame = pd.read_parquet(input_dir / "train.parquet")
    validation_frame = pd.read_parquet(input_dir / "validation.parquet")
    train_rows = train_frame.to_dict(orient="records")
    validation_rows = validation_frame.to_dict(orient="records")
    sft_rows, grpo_rows = stratified_split(train_rows, ratio=sft_ratio, seed=seed)

    sft_identities = {_identity(row) for row in sft_rows}
    grpo_identities = {_identity(row) for row in grpo_rows}
    overlap = sft_identities & grpo_identities
    if overlap:
        raise RuntimeError(f"SFT and GRPO source identities overlap: {sorted(overlap)[:3]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sft_train": output_dir / "sft_train.jsonl",
        "sft_validation": output_dir / "sft_validation.jsonl",
        "grpo_train": output_dir / "grpo_train.parquet",
        "manifest": output_dir / "manifest.json",
    }
    _write_jsonl(
        paths["sft_train"],
        (convert_row(row, split="sft_train") for row in sft_rows),
    )
    _write_jsonl(
        paths["sft_validation"],
        (convert_row(row, split="validation") for row in validation_rows),
    )
    grpo_mask = train_frame["extra_info"].map(
        lambda info: str(info["source_identity"]) in grpo_identities
    )
    train_frame.loc[grpo_mask].reset_index(drop=True).to_parquet(
        paths["grpo_train"], index=False
    )

    banking_rows = sum(
        1
        for row in train_rows + validation_rows
        if str(row["extra_info"]["subset"]) == "banking"
    )
    manifest = {
        "description": "Disjoint three-field SFT and GRPO split from audited reconstructed data",
        "seed": seed,
        "sft_ratio": sft_ratio,
        "stratification_fields": [
            "dataset",
            "subset",
            "Malicious_User_Request",
            "Prompt_Injection",
            "Harmfulness_Rating",
        ],
        "target_protocol": (
            "<Think> <\\Think>\\n"
            "<Malicious_User_Request> {yes|no} <\\Malicious_User_Request>\\n"
            "<Being_Attacked> {yes|no} <\\Being_Attacked>\\n"
            "<Harmfulness_Rating> {0.0|0.5|1.0} <\\Harmfulness_Rating>"
        ),
        "counts": {
            "candidate_train": len(train_rows),
            "sft_train": _counts(sft_rows),
            "grpo_train": _counts(grpo_rows),
            "validation": _counts(validation_rows),
        },
        "audit": {
            "sft_grpo_identity_overlap": len(overlap),
            "banking_rows": banking_rows,
            "sft_identity_count": len(sft_identities),
            "grpo_identity_count": len(grpo_identities),
        },
        "files": {},
    }
    for name, path in paths.items():
        if name != "manifest":
            manifest["files"][path.name] = {"sha256": _sha256(path)}
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repository_root
        / "practice"
        / "toolsafe_reproduction"
        / "data"
        / "grpo_reconstructed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root
        / "practice"
        / "toolsafe_reproduction"
        / "data"
        / "sft_grpo_disjoint",
    )
    parser.add_argument("--sft-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    paths = prepare_dataset(
        args.input_dir,
        args.output_dir,
        sft_ratio=args.sft_ratio,
        seed=args.seed,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

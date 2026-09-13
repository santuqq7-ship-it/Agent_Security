"""Model-free runtime services for resumable constrained GRPO training."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml

from constrained_grpo_core import load_checkpoint_metadata


class DataBoundaryError(ValueError):
    """A development path crosses a frozen evaluation boundary."""


@dataclass(frozen=True)
class ValidationSelection:
    mean_dense_reward: float
    harmfulness_macro_f1: float
    global_step: int


_OPERATIONAL_CONFIG_KEYS = frozenset(
    {"resume", "max_steps", "preflight_only", "terminal_verbosity"}
)
_CHECKPOINT_FILES = frozenset(
    {"optimizer.pt", "scheduler.pt", "rng_state.pt", "trainer_state.json"}
)


def load_training_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load schema-v1 YAML and apply explicit non-None operational overrides."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("training config schema_version must be 1")
    config = dict(payload)
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_config_sha256(config: Mapping[str, Any]) -> str:
    semantic = {
        str(key): value
        for key, value in config.items()
        if key not in _OPERATIONAL_CONFIG_KEYS
    }
    encoded = json.dumps(
        semantic,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_data_boundaries(
    *,
    training_path: str | Path,
    validation_path: str | Path,
) -> None:
    for role, raw_path in (
        ("training", training_path),
        ("validation", validation_path),
    ):
        normalized = "/" + str(raw_path).replace("\\", "/").lower().strip("/")
        if "/banking" in normalized or normalized.endswith("banking.json"):
            raise DataBoundaryError(f"{role} path enters external banking: {raw_path}")
        if "/asb" in normalized or "/asb-traj" in normalized:
            raise DataBoundaryError(f"{role} path enters frozen ASB: {raw_path}")


def epoch_indices(*, row_count: int, epoch: int, seed: int) -> tuple[int, ...]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(epoch))
    return tuple(
        int(index)
        for index in torch.randperm(row_count, generator=generator).tolist()
    )


def optimizer_step_batches(
    indices: Sequence[int],
    *,
    sample_cursor: int,
    groups_per_step: int,
) -> tuple[tuple[int, ...], ...]:
    if groups_per_step <= 0:
        raise ValueError("groups_per_step must be positive")
    if not 0 <= sample_cursor <= len(indices):
        raise ValueError("sample_cursor lies outside the epoch order")
    remaining = indices[sample_cursor:]
    return tuple(
        tuple(int(index) for index in remaining[start : start + groups_per_step])
        for start in range(0, len(remaining), groups_per_step)
    )


def _validate_checkpoint_layout(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"checkpoint directory is missing: {path}")
    present = {child.name for child in path.iterdir()}
    has_actor = "actor" in present or "actor_state.pt" in present
    missing = sorted(_CHECKPOINT_FILES - present)
    if missing or not has_actor:
        details = f"missing={missing}, has_actor={has_actor}"
        raise RuntimeError(f"incomplete checkpoint layout at {path}: {details}")


def rotate_checkpoint(
    root: str | Path,
    *,
    global_step: int,
    writer: Callable[[Path], None],
) -> Path:
    """Write incoming completely, then retain only latest and previous."""

    checkpoint_root = Path(root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    incoming = checkpoint_root / f"incoming_step_{int(global_step)}"
    if incoming.exists():
        shutil.rmtree(incoming)
    writer(incoming)
    _validate_checkpoint_layout(incoming)
    (incoming / "COMPLETE").write_text(
        f"{int(global_step)}\n", encoding="utf-8"
    )

    latest = checkpoint_root / "latest"
    previous = checkpoint_root / "previous"
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        latest.rename(previous)
    incoming.rename(latest)
    return latest


def discover_resume_checkpoint(
    root: str | Path,
    *,
    compatibility: Mapping[str, str],
) -> Path | None:
    checkpoint_root = Path(root)
    if not checkpoint_root.exists():
        return None
    candidates = [checkpoint_root / "latest", checkpoint_root / "previous"]
    candidates.extend(sorted(checkpoint_root.glob("incoming_step_*")))
    compatible: list[tuple[int, Path]] = []
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "COMPLETE").is_file():
            continue
        _validate_checkpoint_layout(candidate)
        state = load_checkpoint_metadata(candidate, **dict(compatibility))
        compatible.append((state.global_step, candidate))
    if not compatible:
        return None
    return max(compatible, key=lambda item: item[0])[1]


def compute_validation_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("validation records must not be empty")
    labels = [0.0, 0.5, 1.0]
    gold_h = [
        float(record["ground_truth"]["Harmfulness_Rating"])
        for record in records
    ]
    predicted_h = [
        float(record["predicted"]["Harmfulness_Rating"])
        for record in records
    ]
    count = len(records)
    per_label_f1: list[float] = []
    per_label_recall: list[float] = []
    for label in labels:
        true_positive = sum(
            gold == label and predicted == label
            for gold, predicted in zip(gold_h, predicted_h)
        )
        false_positive = sum(
            gold != label and predicted == label
            for gold, predicted in zip(gold_h, predicted_h)
        )
        false_negative = sum(
            gold == label and predicted != label
            for gold, predicted in zip(gold_h, predicted_h)
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        f1_denominator = precision + recall
        per_label_f1.append(
            2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
        )
        per_label_recall.append(recall)

    return {
        "total": count,
        "strict_format_rate": sum(
            bool(record["strict_format"]) for record in records
        )
        / count,
        "forced_rationale_close_rate": sum(
            bool(record["forced_rationale_close"]) for record in records
        )
        / count,
        "mean_dense_reward": sum(float(record["reward"]) for record in records)
        / count,
        "malicious_user_request_accuracy": sum(
            record["ground_truth"]["Malicious_User_Request"]
            == record["predicted"]["Malicious_User_Request"]
            for record in records
        )
        / count,
        "being_attacked_accuracy": sum(
            record["ground_truth"]["Being_Attacked"]
            == record["predicted"]["Being_Attacked"]
            for record in records
        )
        / count,
        "harmfulness_accuracy": sum(
            gold == predicted
            for gold, predicted in zip(gold_h, predicted_h)
        )
        / count,
        "harmfulness_labels": labels,
        "harmfulness_macro_f1": sum(per_label_f1) / len(labels),
        "harmfulness_macro_recall": sum(per_label_recall) / len(labels),
    }


def is_better_validation(
    candidate: ValidationSelection,
    incumbent: ValidationSelection | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        -float(candidate.mean_dense_reward),
        -float(candidate.harmfulness_macro_f1),
        int(candidate.global_step),
    )
    incumbent_key = (
        -float(incumbent.mean_dense_reward),
        -float(incumbent.harmfulness_macro_f1),
        int(incumbent.global_step),
    )
    return candidate_key < incumbent_key


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

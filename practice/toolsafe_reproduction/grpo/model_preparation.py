"""Integrity primitives for preparing GRPO policy and reference checkpoints.

This module intentionally contains no Transformers or PEFT model-loading
logic.  Keeping hashing and comparison code independent makes the safety
checks fast to test before any multi-gigabyte model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch


class ModelPreparationError(RuntimeError):
    """Raised when a checkpoint cannot be prepared without ambiguity."""


@dataclass(frozen=True)
class FileRecord:
    """Content identity for one file in a checkpoint directory."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelObservation:
    """Deterministic model behavior captured before or after LoRA merging."""

    token_ids: list[int]
    logits: torch.Tensor
    parsed_judgment: dict[str, object] | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Safety and BF16 distribution evidence for one validated merge."""

    max_logit_abs_diff: float
    mean_logit_abs_diff: float
    max_probability_abs_diff: float
    total_variation_distance: float
    generated_token_ids_equal: bool
    pre_merge_generated_token_ids: list[int]
    post_merge_generated_token_ids: list[int]
    parsed_judgment: dict[str, object]
    pre_merge_repeat_verified: bool
    post_merge_repeat_verified: bool


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a real file without loading it at once."""

    if not path.is_file():
        raise ModelPreparationError(f"file does not exist: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_directory(path: Path) -> dict[str, FileRecord]:
    """Hash every file under ``path`` using stable relative POSIX names."""

    if not path.is_dir():
        raise ModelPreparationError(f"checkpoint directory does not exist: {path}")

    inventory: dict[str, FileRecord] = {}
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        inventory[relative_path] = FileRecord(
            size_bytes=file_path.stat().st_size,
            sha256=sha256_file(file_path),
        )
    return inventory


def assert_matching_inventories(
    policy: Path,
    reference: Path,
) -> dict[str, FileRecord]:
    """Require policy and reference directories to contain identical bytes."""

    policy_inventory = inventory_directory(policy)
    reference_inventory = inventory_directory(reference)
    if policy_inventory != reference_inventory:
        policy_files = set(policy_inventory)
        reference_files = set(reference_inventory)
        missing_from_reference = sorted(policy_files - reference_files)
        extra_in_reference = sorted(reference_files - policy_files)
        changed = sorted(
            relative_path
            for relative_path in policy_files & reference_files
            if policy_inventory[relative_path] != reference_inventory[relative_path]
        )
        raise ModelPreparationError(
            "policy/reference inventory mismatch: "
            f"missing_from_reference={missing_from_reference}, "
            f"extra_in_reference={extra_in_reference}, changed={changed}"
        )
    return policy_inventory


def validate_model_outputs(
    before: ModelObservation,
    after: ModelObservation,
    *,
    before_repeat: ModelObservation,
    after_repeat: ModelObservation,
    max_total_variation_distance: float,
) -> ValidationResult:
    """Require stable safety fields while allowing unrelated BF16 token drift."""

    if (
        not math.isfinite(max_total_variation_distance)
        or not 0 <= max_total_variation_distance <= 1
    ):
        raise ValueError(
            "max_total_variation_distance must be a finite value between 0 and 1"
        )
    for label, original, repeated in (
        ("pre-merge", before, before_repeat),
        ("post-merge", after, after_repeat),
    ):
        if (
            original.token_ids != repeated.token_ids
            or original.parsed_judgment != repeated.parsed_judgment
        ):
            raise ModelPreparationError(
                f"{label} generation is not deterministic: "
                f"first_tokens={original.token_ids}, repeat_tokens={repeated.token_ids}, "
                f"first_judgment={original.parsed_judgment}, "
                f"repeat_judgment={repeated.parsed_judgment}"
            )

    required_fields = {
        "Malicious_User_Request",
        "Being_Attacked",
        "Harmfulness_Rating",
    }
    for label, observation in (("pre-merge", before), ("post-merge", after)):
        if observation.parsed_judgment is None:
            raise ModelPreparationError(
                f"{label} parsed judgment is required for safety validation"
            )
        actual_fields = set(observation.parsed_judgment)
        if actual_fields != required_fields:
            raise ModelPreparationError(
                f"{label} parsed judgment fields are incomplete or unexpected: "
                f"required={sorted(required_fields)}, actual={sorted(actual_fields)}"
            )
    if before.parsed_judgment != after.parsed_judgment:
        raise ModelPreparationError(
            "parsed judgment changed after merge: "
            f"before={before.parsed_judgment}, after={after.parsed_judgment}"
        )
    if before.logits.shape != after.logits.shape:
        raise ModelPreparationError(
            "logit shapes changed after merge: "
            f"before={tuple(before.logits.shape)}, after={tuple(after.logits.shape)}"
        )
    if before.logits.numel() == 0:
        raise ModelPreparationError("logits must not be empty")
    if not torch.isfinite(before.logits).all() or not torch.isfinite(after.logits).all():
        raise ModelPreparationError("all pre-merge and post-merge logits must be finite")

    absolute_difference = (before.logits.float() - after.logits.float()).abs()
    measured_maximum = float(absolute_difference.max().item())
    measured_mean = float(absolute_difference.mean().item())
    before_probabilities = torch.softmax(before.logits.float(), dim=-1)
    after_probabilities = torch.softmax(after.logits.float(), dim=-1)
    probability_difference = (before_probabilities - after_probabilities).abs()
    maximum_probability_difference = float(probability_difference.max().item())
    total_variation_distance = float(0.5 * probability_difference.sum().item())
    if total_variation_distance > max_total_variation_distance:
        raise ModelPreparationError(
            "token-distribution total-variation distance exceeds tolerance: "
            f"measured={total_variation_distance}, "
            f"allowed={max_total_variation_distance}"
        )

    return ValidationResult(
        max_logit_abs_diff=measured_maximum,
        mean_logit_abs_diff=measured_mean,
        max_probability_abs_diff=maximum_probability_difference,
        total_variation_distance=total_variation_distance,
        generated_token_ids_equal=before.token_ids == after.token_ids,
        pre_merge_generated_token_ids=list(before.token_ids),
        post_merge_generated_token_ids=list(after.token_ids),
        parsed_judgment=dict(before.parsed_judgment),
        pre_merge_repeat_verified=True,
        post_merge_repeat_verified=True,
    )


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write complete JSON and atomically replace the destination file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

#!/usr/bin/env python3
"""Resumable same-Actor constrained GRPO training for ToolSafe Guardian.

The module deliberately keeps preflight independent of Transformers and CUDA.
Model imports happen only inside :func:`run_training`, after all data and path
boundaries have been validated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch
import yaml

from constrained_grpo_core import (
    ConstrainedTrainerState,
    dense_guardian_reward,
    grpo_response_loss,
    group_relative_advantages,
    restore_training_state,
    save_training_checkpoint,
)
from constrained_grpo_runtime import (
    DataBoundaryError,
    ValidationSelection,
    append_jsonl,
    compute_validation_metrics,
    discover_resume_checkpoint,
    epoch_indices,
    is_better_validation,
    load_training_config,
    optimizer_step_batches,
    rotate_checkpoint,
    semantic_config_sha256,
    sha256_file,
    validate_data_boundaries,
)


PROTOCOL_VERSION = "toolsafe-guardian-token-fsm-v2"


class PreflightError(RuntimeError):
    """The run is unsafe or inconsistent before any model is loaded."""


class TrainingInvariantError(RuntimeError):
    """A fail-closed training invariant was violated."""


@dataclass(frozen=True)
class PromptGroupResult:
    source_identity: str
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    policy_loss: float
    kl_loss: float
    total_loss: float
    clip_fraction: float
    parity_max_abs_error: float
    variable_group: bool
    all_equal_group: bool
    forced_rationale_close_count: int
    response_lengths: tuple[int, ...]


def _resolve(project_root: Path, raw_path: object, *, name: str) -> Path:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        raise PreflightError(f"{name} must be a non-empty path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _require_file(path: Path, *, name: str) -> None:
    if not path.is_file():
        raise PreflightError(f"{name} file is missing: {path}")


def _require_directory(path: Path, *, name: str) -> None:
    if not path.is_dir():
        raise PreflightError(f"{name} directory is missing: {path}")


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise PreflightError(f"no existing parent directory for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def _as_mapping(value: object, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise PreflightError(f"{name} must be a mapping")


def _source_name(row: Mapping[str, Any]) -> str:
    extra = _as_mapping(row.get("extra_info"), name="GRPO extra_info")
    dataset = str(extra.get("dataset", "")).strip().lower()
    subset = str(extra.get("subset", "")).strip().lower()
    if not dataset or not subset:
        raise PreflightError("GRPO row lacks extra_info dataset/subset")
    return f"{dataset}/{subset}"


def _source_identity(row: Mapping[str, Any], *, role: str) -> str:
    if role == "training":
        extra = _as_mapping(row.get("extra_info"), name="GRPO extra_info")
        value = extra.get("source_identity")
    else:
        value = row.get("source_identity")
    identity = str(value or "").strip()
    if not identity:
        raise PreflightError(f"{role} row lacks source_identity")
    return identity


def _validation_source_name(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).strip().lower()
    subset = str(row.get("subset", "")).strip().lower()
    if not dataset or not subset:
        identity = str(row.get("source_identity", ""))
        parts = identity.split(":")
        if len(parts) >= 2:
            dataset, subset = parts[0].lower(), parts[1].lower()
    if not dataset or not subset:
        raise PreflightError("validation row lacks dataset/subset")
    return f"{dataset}/{subset}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise PreflightError(
                    f"JSONL row at {path}:{line_number} must be an object"
                )
            rows.append(value)
    return rows


def _expected_sources(section: Mapping[str, Any], *, role: str) -> Counter[str]:
    raw = section.get("allowed_sources")
    if not isinstance(raw, Mapping) or not raw:
        raise PreflightError(f"manifest {role} allowed_sources is missing")
    try:
        result = Counter({str(key): int(value) for key, value in raw.items()})
    except (TypeError, ValueError) as exc:
        raise PreflightError(
            f"manifest {role} source counts must be integers"
        ) from exc
    if any(value < 0 for value in result.values()):
        raise PreflightError(f"manifest {role} source counts must be nonnegative")
    return result


def _require_positive_number(config: Mapping[str, Any], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise PreflightError(f"{name} must be finite and positive")
    return numeric


def _validate_hyperparameters(config: Mapping[str, Any]) -> None:
    integer_names = (
        "epochs",
        "rollouts_per_prompt",
        "prompt_groups_per_step",
        "min_rationale_content_tokens",
        "max_rationale_tokens",
        "checkpoint_every_steps",
        "validate_every_steps",
        "checkpoint_generations",
        "max_consecutive_zero_signal_steps",
    )
    for name in integer_names:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PreflightError(f"{name} must be a positive integer")
    if config["checkpoint_generations"] != 2:
        raise PreflightError("checkpoint_generations must be exactly 2")
    if config["max_rationale_tokens"] < config["min_rationale_content_tokens"]:
        raise PreflightError("max_rationale_tokens is below the rationale minimum")
    for name in (
        "temperature",
        "learning_rate",
        "ppo_clip_ratio",
        "kl_coefficient",
        "max_grad_norm",
        "old_new_parity_tolerance",
    ):
        _require_positive_number(config, name)
    top_p = _require_positive_number(config, "top_p")
    if top_p > 1:
        raise PreflightError("top_p must be in (0, 1]")
    minimum_free = config.get("minimum_free_gb_before_checkpoint")
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, (int, float))
        or float(minimum_free) < 0
    ):
        raise PreflightError("minimum_free_gb_before_checkpoint must be nonnegative")
    weight_decay = config.get("weight_decay")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0
    ):
        raise PreflightError("weight_decay must be finite and nonnegative")


def build_preflight_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen experiment without importing or loading a model."""

    if config.get("schema_version") != 1:
        raise PreflightError("training config schema_version must be 1")
    project_root = _resolve(Path.cwd(), config.get("project_root"), name="project_root")
    _require_directory(project_root, name="project_root")
    actor_model = _resolve(project_root, config.get("actor_model"), name="actor_model")
    reference_model = _resolve(
        project_root, config.get("reference_model"), name="reference_model"
    )
    manifest_path = _resolve(project_root, config.get("manifest"), name="manifest")
    train_path = _resolve(project_root, config.get("train_file"), name="train_file")
    validation_path = _resolve(
        project_root,
        config.get("clean_validation_file"),
        name="clean_validation_file",
    )
    output_root = _resolve(project_root, config.get("output_root"), name="output_root")

    for path, name in (
        (actor_model, "actor_model"),
        (reference_model, "reference_model"),
    ):
        _require_directory(path, name=name)
    for path, name in (
        (manifest_path, "manifest"),
        (train_path, "train_file"),
        (validation_path, "clean_validation_file"),
    ):
        _require_file(path, name=name)

    allowed_results = (
        project_root / "practice/toolsafe_reproduction/results"
    ).resolve()
    if not _inside(output_root, allowed_results):
        raise PreflightError(
            f"output_root must remain inside {allowed_results}: {output_root}"
        )
    try:
        validate_data_boundaries(
            training_path=train_path, validation_path=validation_path
        )
    except DataBoundaryError as exc:
        raise PreflightError(str(exc)) from exc
    _validate_hyperparameters(config)

    manifest_raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = _as_mapping(manifest_raw, name="manifest")
    boundaries = _as_mapping(
        manifest.get("data_boundaries"), name="manifest data_boundaries"
    )
    train_manifest = _as_mapping(
        boundaries.get("grpo_train"), name="manifest grpo_train"
    )
    validation_manifest = _as_mapping(
        boundaries.get("clean_validation"), name="manifest clean_validation"
    )
    grammar_manifest = _as_mapping(
        manifest.get("guardian_grammar"), name="manifest guardian_grammar"
    )
    protocol = str(grammar_manifest.get("protocol_version", ""))
    if protocol != PROTOCOL_VERSION:
        raise PreflightError(
            f"guardian grammar protocol must be {PROTOCOL_VERSION}, got {protocol!r}"
        )

    expected_train_path = _resolve(
        project_root,
        train_manifest.get("canonical_path"),
        name="manifest grpo_train canonical_path",
    )
    expected_validation_path = _resolve(
        project_root,
        validation_manifest.get("canonical_path"),
        name="manifest clean_validation canonical_path",
    )
    if train_path != expected_train_path or validation_path != expected_validation_path:
        raise PreflightError("configured data paths differ from manifest canonical paths")

    import pandas as pd

    frame = pd.read_parquet(train_path)
    training_rows = frame.to_dict(orient="records")
    validation_rows = _read_jsonl(validation_path)
    expected_training_rows = int(train_manifest.get("logical_rows", -1))
    expected_validation_rows = int(validation_manifest.get("logical_rows", -1))
    if len(training_rows) != expected_training_rows:
        raise PreflightError(
            f"training row count mismatch: {len(training_rows)} != "
            f"{expected_training_rows}"
        )
    if len(validation_rows) != expected_validation_rows:
        raise PreflightError(
            f"validation row count mismatch: {len(validation_rows)} != "
            f"{expected_validation_rows}"
        )

    actual_training_sources = Counter(_source_name(row) for row in training_rows)
    actual_validation_sources = Counter(
        _validation_source_name(row) for row in validation_rows
    )
    expected_training_sources = _expected_sources(train_manifest, role="training")
    expected_validation_sources = _expected_sources(
        validation_manifest, role="validation"
    )
    if actual_training_sources != expected_training_sources:
        raise PreflightError(
            "training source counts differ from manifest: "
            f"actual={dict(actual_training_sources)}, "
            f"expected={dict(expected_training_sources)}"
        )
    if actual_validation_sources != expected_validation_sources:
        raise PreflightError(
            "validation source counts differ from manifest: "
            f"actual={dict(actual_validation_sources)}, "
            f"expected={dict(expected_validation_sources)}"
        )

    training_identities = {
        _source_identity(row, role="training") for row in training_rows
    }
    validation_identities = {
        _source_identity(row, role="validation") for row in validation_rows
    }
    if len(training_identities) != len(training_rows):
        raise PreflightError("training source_identity values are not unique")
    if len(validation_identities) != len(validation_rows):
        raise PreflightError("validation source_identity values are not unique")
    overlap = training_identities & validation_identities
    if overlap:
        raise PreflightError(
            f"training/validation source_identity overlap: {sorted(overlap)[:5]}"
        )

    groups_per_step = int(config["prompt_groups_per_step"])
    epochs = int(config["epochs"])
    steps_per_epoch = math.ceil(len(training_rows) / groups_per_step)
    disk_anchor = _nearest_existing_directory(output_root.parent)
    free_gb = shutil.disk_usage(disk_anchor).free / 1024**3
    minimum_free_gb = float(config["minimum_free_gb_before_checkpoint"])
    if free_gb < minimum_free_gb:
        raise PreflightError(
            f"insufficient disk: {free_gb:.2f} GiB free, "
            f"{minimum_free_gb:.2f} GiB required"
        )

    forbidden_sources = {
        name: count
        for name, count in (actual_training_sources + actual_validation_sources).items()
        if "banking" in name or name.startswith("asb/") or "/asb" in name
    }
    if forbidden_sources:
        raise PreflightError(f"forbidden development sources: {forbidden_sources}")

    return {
        "stage": "E5_model_free_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "project_root": str(project_root),
        "actor_model": str(actor_model),
        "reference_model": str(reference_model),
        "manifest": str(manifest_path),
        "train_file": str(train_path),
        "clean_validation_file": str(validation_path),
        "output_root": str(output_root),
        "protocol_version": protocol,
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": steps_per_epoch * epochs,
        "rollouts_per_prompt": int(config["rollouts_per_prompt"]),
        "prompt_groups_per_step": groups_per_step,
        "rollouts_per_optimizer_step": (
            int(config["rollouts_per_prompt"]) * groups_per_step
        ),
        "training_validation_identity_overlap": len(overlap),
        "training_source_counts": dict(sorted(actual_training_sources.items())),
        "validation_source_counts": dict(sorted(actual_validation_sources.items())),
        "banking_rows": 0,
        "asb_rows": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "train_file_sha256": sha256_file(train_path),
        "semantic_config_sha256": semantic_config_sha256(config),
        "free_disk_gb": free_gb,
        "minimum_free_gb_before_checkpoint": minimum_free_gb,
        "formal_training_started": False,
        "model_loaded": False,
        "optimizer_created": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validation_record_for_step(
    path: Path, *, global_step: int
) -> dict[str, Any] | None:
    """Return the last complete aggregate validation record for one step."""

    if not path.is_file():
        return None
    matched: dict[str, Any] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            record_step = int(record["global_step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TrainingInvariantError(
                f"invalid validation metric record at {path}:{line_number}"
            ) from exc
        if record_step == global_step:
            matched = dict(record)
    return matched


def _load_training_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]


def _messages_from_training_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise TrainingInvariantError("GRPO prompt is not a message sequence")
    messages = [dict(message) for message in raw]
    if not messages or any(
        not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise TrainingInvariantError("GRPO prompt messages require role/content")
    return messages


def _prompt_token_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    values = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if not isinstance(values, list) or not values:
        raise TrainingInvariantError("chat template produced no prompt tokens")
    return [int(value) for value in values]


def _training_ground_truth(row: Mapping[str, Any]) -> dict[str, Any]:
    reward_model = _as_mapping(row.get("reward_model"), name="reward_model")
    return _as_mapping(reward_model.get("ground_truth"), name="ground_truth")


def _validation_ground_truth(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "malicious_user_request",
        "being_attacked",
        "score",
    )
    if any(name not in row for name in required):
        raise TrainingInvariantError("clean validation row lacks a gold field")
    return {
        "Malicious_User_Request": bool(row["malicious_user_request"]),
        "Being_Attacked": bool(row["being_attacked"]),
        "Harmfulness_Rating": float(row["score"]),
    }


def _model_from_pretrained(path: Path, *, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    return model.to(torch.device("cuda"))


def _ensure_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise TrainingInvariantError(f"{name} contains a non-finite value")


def _checkpoint_compatibility(
    preflight: Mapping[str, Any], tokenizer_fingerprint: str
) -> dict[str, str]:
    return {
        "expected_protocol_version": str(preflight["protocol_version"]),
        "expected_manifest_sha256": str(preflight["manifest_sha256"]),
        "expected_data_sha256": str(preflight["train_file_sha256"]),
        "expected_config_sha256": str(preflight["semantic_config_sha256"]),
        "expected_tokenizer_fingerprint": tokenizer_fingerprint,
    }


def _resolve_resume_checkpoint(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    compatibility: Mapping[str, str],
) -> Path | None:
    output_root = Path(str(preflight["output_root"]))
    checkpoint_root = output_root / "checkpoints"
    resume = str(config.get("resume", "auto"))
    if resume == "auto":
        return discover_resume_checkpoint(
            checkpoint_root, compatibility=compatibility
        )
    if resume == "none":
        existing = [
            output_root / name
            for name in (
                "train_metrics.jsonl",
                "validation_metrics.jsonl",
                "rollouts.jsonl",
                "run_summary.json",
            )
            if (output_root / name).exists()
        ]
        if checkpoint_root.exists() or existing:
            raise TrainingInvariantError(
                "resume=none refuses an existing training output; choose auto or "
                "a new output_root"
            )
        return None
    requested = Path(resume).expanduser()
    if not requested.is_absolute():
        requested = Path(str(preflight["project_root"])) / requested
    requested = requested.resolve()
    allowed = (output_root / "checkpoints").resolve()
    if not _inside(requested, allowed):
        raise TrainingInvariantError(
            f"explicit resume checkpoint lies outside {allowed}: {requested}"
        )
    if not requested.is_dir():
        raise FileNotFoundError(requested)
    # Discovery performs both layout and metadata checks.  Use a temporary
    # one-candidate root only for auto; explicit paths are validated directly.
    from constrained_grpo_core import load_checkpoint_metadata

    load_checkpoint_metadata(requested, **dict(compatibility))
    if not (requested / "COMPLETE").is_file():
        raise TrainingInvariantError(f"resume checkpoint is incomplete: {requested}")
    return requested


def _checkpoint_free_space_guard(
    checkpoint_root: Path, *, minimum_free_gb: float
) -> None:
    anchor = _nearest_existing_directory(checkpoint_root.parent)
    free_gb = shutil.disk_usage(anchor).free / 1024**3
    if free_gb < minimum_free_gb:
        raise TrainingInvariantError(
            f"checkpoint refused: {free_gb:.2f} GiB free, "
            f"{minimum_free_gb:.2f} GiB required"
        )


def _save_checkpoint(
    *,
    output_root: Path,
    actor: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: ConstrainedTrainerState,
    minimum_free_gb: float,
) -> Path:
    checkpoint_root = output_root / "checkpoints"
    _checkpoint_free_space_guard(
        checkpoint_root, minimum_free_gb=minimum_free_gb
    )

    def writer(incoming: Path) -> None:
        save_training_checkpoint(
            incoming,
            actor=actor,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
        )

    return rotate_checkpoint(
        checkpoint_root, global_step=state.global_step, writer=writer
    )


def _best_selection(output_root: Path) -> ValidationSelection | None:
    final = output_root / "best_actor"
    previous = output_root / "best_actor_previous"
    if not final.is_dir() and (previous / "COMPLETE").is_file():
        previous.rename(final)
    elif final.is_dir() and previous.exists():
        shutil.rmtree(previous)
    path = final / "selection.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ValidationSelection(
        mean_dense_reward=float(payload["mean_dense_reward"]),
        harmfulness_macro_f1=float(payload["harmfulness_macro_f1"]),
        global_step=int(payload["global_step"]),
    )


def _replace_best_actor(
    *,
    output_root: Path,
    actor: torch.nn.Module,
    tokenizer: Any,
    selection: ValidationSelection,
    metrics: Mapping[str, Any],
) -> None:
    incoming = output_root / "best_actor_incoming"
    final = output_root / "best_actor"
    previous = output_root / "best_actor_previous"
    if incoming.exists():
        shutil.rmtree(incoming)
    actor.save_pretrained(incoming)
    tokenizer.save_pretrained(incoming)
    _write_json(
        incoming / "selection.json",
        {**asdict(selection), "metrics": dict(metrics)},
    )
    (incoming / "COMPLETE").write_text(
        f"{selection.global_step}\n", encoding="utf-8"
    )
    if previous.exists():
        shutil.rmtree(previous)
    if final.exists():
        final.rename(previous)
    incoming.rename(final)
    if previous.exists():
        shutil.rmtree(previous)


def _append_rollout_audit(
    path: Path,
    *,
    session_id: str,
    global_step: int,
    phase: str,
    source_identity: str,
    prompt: object,
    ground_truth: Mapping[str, Any],
    rollout: Any,
    reward: float,
    advantage: float | None,
) -> None:
    append_jsonl(
        path,
        {
            "session_id": session_id,
            "global_step": global_step,
            "phase": phase,
            "source_identity": source_identity,
            "prompt": prompt,
            "ground_truth": dict(ground_truth),
            "response": rollout.output_text,
            "predicted": dict(rollout.judgments),
            "reward": float(reward),
            "advantage": None if advantage is None else float(advantage),
            "strict_format": True,
            "forced_rationale_close": bool(rollout.forced_rationale_close),
            "decision_count": len(rollout.decisions),
            "response_token_count": len(rollout.response_token_ids),
        },
    )


def _train_prompt_group(
    *,
    actor: torch.nn.Module,
    reference: torch.nn.Module,
    actor_grammar: Any,
    reference_grammar: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    rollouts_per_prompt: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
    scale: float,
    clip_ratio: float,
    kl_coefficient: float,
    parity_tolerance: float,
    session_id: str,
    global_step: int,
    rollouts_path: Path,
) -> PromptGroupResult:
    from constrained_guardian_policy import (
        generate_constrained_rollout,
        recompute_constrained_log_probs,
    )
    from e3_constrained_logprob_canary import strict_response_matches_rollout

    messages = _messages_from_training_row(row)
    prompt_ids = _prompt_token_ids(tokenizer, messages)
    ground_truth = _training_ground_truth(row)
    extra = _as_mapping(row.get("extra_info"), name="extra_info")
    source_identity = str(extra["source_identity"])

    rollouts: list[Any] = []
    old_log_probs: list[torch.Tensor] = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for _ in range(rollouts_per_prompt):
            rollout = generate_constrained_rollout(
                actor,
                actor_grammar,
                prompt_token_ids=prompt_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            if not strict_response_matches_rollout(rollout):
                raise TrainingInvariantError(
                    f"FSM emitted a non-strict response for {source_identity}"
                )
            rollouts.append(rollout)
            old_log_probs.append(
                recompute_constrained_log_probs(
                    actor,
                    actor_grammar,
                    prompt_token_ids=prompt_ids,
                    decisions=rollout.decisions,
                ).detach().cpu()
            )

    rewards = tuple(
        dense_guardian_reward(rollout.judgments, ground_truth)
        for rollout in rollouts
    )
    advantages_tensor, stats = group_relative_advantages(
        torch.tensor(rewards, dtype=torch.float32),
        group_ids=(source_identity,) * len(rewards),
    )
    advantages = tuple(float(value) for value in advantages_tensor.tolist())

    reference_log_probs: list[torch.Tensor] = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for rollout in rollouts:
            reference_log_probs.append(
                recompute_constrained_log_probs(
                    reference,
                    reference_grammar,
                    prompt_token_ids=prompt_ids,
                    decisions=rollout.decisions,
                ).detach().cpu()
            )

    policy_loss = 0.0
    kl_loss = 0.0
    total_loss = 0.0
    clip_fraction = 0.0
    parity_errors: list[float] = []
    actor.train()
    for rollout, old_cpu, reference_cpu, advantage in zip(
        rollouts, old_log_probs, reference_log_probs, advantages
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            new_log_probs = recompute_constrained_log_probs(
                actor,
                actor_grammar,
                prompt_token_ids=prompt_ids,
                decisions=rollout.decisions,
                force_eval=False,
            )
            old_values = old_cpu.to(new_log_probs.device)
            reference_values = reference_cpu.to(new_log_probs.device)
            if not (
                new_log_probs.shape
                == old_values.shape
                == reference_values.shape
            ):
                raise TrainingInvariantError(
                    f"Actor/old/Reference decision counts differ for {source_identity}"
                )
            _ensure_finite_tensor(new_log_probs, name="new Actor log probabilities")
            _ensure_finite_tensor(old_values, name="old Actor log probabilities")
            _ensure_finite_tensor(
                reference_values, name="Reference log probabilities"
            )
            parity = float(
                (new_log_probs.detach() - old_values).abs().max().item()
            )
            parity_errors.append(parity)
            if parity > parity_tolerance:
                raise TrainingInvariantError(
                    f"old/new parity {parity:.8g} exceeds {parity_tolerance:.8g} "
                    f"for {source_identity}"
                )
            loss = grpo_response_loss(
                new_log_probs,
                old_log_probs=old_values,
                reference_log_probs=reference_values,
                advantage=advantage,
                clip_ratio=clip_ratio,
                kl_coefficient=kl_coefficient,
            )
            _ensure_finite_tensor(loss.total_loss, name="GRPO total loss")
            scaled_loss = loss.total_loss * scale
        scaled_loss.backward()
        policy_loss += float(loss.policy_loss.detach().item()) / len(rollouts)
        kl_loss += float(loss.kl_loss.detach().item()) / len(rollouts)
        total_loss += float(loss.total_loss.detach().item()) / len(rollouts)
        clip_fraction += float(loss.clip_fraction.detach().item()) / len(rollouts)
        del new_log_probs, old_values, reference_values, loss, scaled_loss

    for rollout, reward, advantage in zip(rollouts, rewards, advantages):
        _append_rollout_audit(
            rollouts_path,
            session_id=session_id,
            global_step=global_step,
            phase="train",
            source_identity=source_identity,
            prompt=messages,
            ground_truth=ground_truth,
            rollout=rollout,
            reward=reward,
            advantage=advantage,
        )
    return PromptGroupResult(
        source_identity=source_identity,
        rewards=rewards,
        advantages=advantages,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        total_loss=total_loss,
        clip_fraction=clip_fraction,
        parity_max_abs_error=max(parity_errors),
        variable_group=stats.variable_group_count == 1,
        all_equal_group=stats.all_equal_group_count == 1,
        forced_rationale_close_count=sum(
            int(rollout.forced_rationale_close) for rollout in rollouts
        ),
        response_lengths=tuple(
            len(rollout.response_token_ids) for rollout in rollouts
        ),
    )


def _run_validation(
    *,
    actor: torch.nn.Module,
    grammar: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    session_id: str,
    global_step: int,
    rollouts_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from constrained_guardian_policy import generate_constrained_rollout
    from e3_constrained_logprob_canary import strict_response_matches_rollout

    records: list[dict[str, Any]] = []
    actor.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for row in rows:
            prompt = str(row["prompt"])
            messages = [{"role": "user", "content": prompt}]
            prompt_ids = _prompt_token_ids(tokenizer, messages)
            rollout = generate_constrained_rollout(
                actor,
                grammar,
                prompt_token_ids=prompt_ids,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                generator=None,
            )
            strict = strict_response_matches_rollout(rollout)
            if not strict:
                raise TrainingInvariantError(
                    "FSM emitted a non-strict clean-validation response"
                )
            ground_truth = _validation_ground_truth(row)
            reward = dense_guardian_reward(rollout.judgments, ground_truth)
            source_identity = str(row["source_identity"])
            record = {
                "ground_truth": ground_truth,
                "predicted": dict(rollout.judgments),
                "strict_format": strict,
                "forced_rationale_close": rollout.forced_rationale_close,
                "reward": reward,
            }
            records.append(record)
            _append_rollout_audit(
                rollouts_path,
                session_id=session_id,
                global_step=global_step,
                phase="validation",
                source_identity=source_identity,
                prompt=prompt,
                ground_truth=ground_truth,
                rollout=rollout,
                reward=reward,
                advantage=None,
            )
    metrics = compute_validation_metrics(records)
    metrics["constraint_violation_count"] = 0
    return metrics, records


def _validate_and_select(
    *,
    actor: torch.nn.Module,
    grammar: Any,
    tokenizer: Any,
    validation_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    checkpoint: Path,
    state: ConstrainedTrainerState,
    session_id: str,
    validation_metrics_path: Path,
    rollouts_path: Path,
) -> tuple[ConstrainedTrainerState, dict[str, Any], bool]:
    """Complete or reconcile validation for an already-checkpointed step."""

    if not (checkpoint / "COMPLETE").is_file():
        raise TrainingInvariantError(
            "validation requires a complete resumable checkpoint"
        )
    checkpoint_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if int(checkpoint_state["global_step"]) != state.global_step:
        raise TrainingInvariantError(
            "validation checkpoint does not match the in-memory global step"
        )

    existing = _validation_record_for_step(
        validation_metrics_path, global_step=state.global_step
    )
    ran_validation = existing is None
    if existing is None:
        started = perf_counter()
        metrics, _ = _run_validation(
            actor=actor,
            grammar=grammar,
            tokenizer=tokenizer,
            rows=validation_rows,
            session_id=session_id,
            global_step=state.global_step,
            rollouts_path=rollouts_path,
        )
        existing = {
            "session_id": session_id,
            "global_step": state.global_step,
            "elapsed_seconds": perf_counter() - started,
            **metrics,
        }
        append_jsonl(validation_metrics_path, existing)
    metrics = dict(existing)
    candidate = ValidationSelection(
        mean_dense_reward=float(metrics["mean_dense_reward"]),
        harmfulness_macro_f1=float(metrics["harmfulness_macro_f1"]),
        global_step=state.global_step,
    )
    incumbent = _best_selection(output_root)
    if incumbent is not None and incumbent.global_step > state.global_step:
        raise TrainingInvariantError(
            "best_actor is newer than the resumable training checkpoint"
        )
    if is_better_validation(candidate, incumbent):
        _replace_best_actor(
            output_root=output_root,
            actor=actor,
            tokenizer=tokenizer,
            selection=candidate,
            metrics=metrics,
        )
        incumbent = candidate
    if incumbent is not None:
        state = replace(
            state,
            best_validation_reward=incumbent.mean_dense_reward,
            best_validation_macro_f1=incumbent.harmfulness_macro_f1,
        )
        _write_json(checkpoint / "trainer_state.json", asdict(state))
    return state, metrics, ran_validation


def _step_metrics(
    *,
    session_id: str,
    state: ConstrainedTrainerState,
    epoch: int,
    groups: Sequence[PromptGroupResult],
    gradient_norm: float,
    learning_rate: float,
    elapsed_seconds: float,
    peak_allocated_gb: float,
    peak_reserved_gb: float,
) -> dict[str, Any]:
    rewards = [reward for group in groups for reward in group.rewards]
    response_lengths = [
        length for group in groups for length in group.response_lengths
    ]
    return {
        "session_id": session_id,
        "global_step": state.global_step,
        "epoch": epoch,
        "sample_cursor": state.sample_cursor,
        "prompt_groups": len(groups),
        "rollouts": len(rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "minimum_reward": min(rewards),
        "maximum_reward": max(rewards),
        "reward_counts": dict(sorted(Counter(str(value) for value in rewards).items())),
        "variable_reward_groups": sum(group.variable_group for group in groups),
        "all_equal_reward_groups": sum(group.all_equal_group for group in groups),
        "variable_reward_group_rate": sum(
            group.variable_group for group in groups
        )
        / len(groups),
        "all_equal_reward_group_rate": sum(
            group.all_equal_group for group in groups
        )
        / len(groups),
        "strict_format_rate": 1.0,
        "constraint_violation_count": 0,
        "policy_loss": sum(group.policy_loss for group in groups) / len(groups),
        "kl_loss": sum(group.kl_loss for group in groups) / len(groups),
        "total_loss": sum(group.total_loss for group in groups) / len(groups),
        "clip_fraction": sum(group.clip_fraction for group in groups) / len(groups),
        "old_new_parity_max_abs_error": max(
            group.parity_max_abs_error for group in groups
        ),
        "gradient_norm_before_clipping": gradient_norm,
        "learning_rate": learning_rate,
        "forced_rationale_close_rate": sum(
            group.forced_rationale_close_count for group in groups
        )
        / len(rewards),
        "response_length_mean": sum(response_lengths) / len(response_lengths),
        "response_length_min": min(response_lengths),
        "response_length_max": max(response_lengths),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_allocated_gb": peak_allocated_gb,
        "peak_cuda_memory_reserved_gb": peak_reserved_gb,
    }


def run_training(config: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    """Run resumable same-Actor constrained GRPO after a passed preflight."""

    if not bool(preflight.get("passed")):
        raise PreflightError("formal training requires a passed preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("constrained GRPO training requires CUDA")

    from transformers import AutoTokenizer

    from constrained_guardian_fsm import compile_guardian_grammar
    from e3_constrained_logprob_canary import tokenizer_fingerprint

    actor_initial = Path(str(preflight["actor_model"]))
    reference_path = Path(str(preflight["reference_model"]))
    train_path = Path(str(preflight["train_file"]))
    validation_path = Path(str(preflight["clean_validation_file"]))
    output_root = Path(str(preflight["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        actor_initial, local_files_only=True
    )
    reference_tokenizer = AutoTokenizer.from_pretrained(
        reference_path, local_files_only=True
    )
    actor_grammar = compile_guardian_grammar(
        tokenizer,
        min_rationale_content_tokens=int(config["min_rationale_content_tokens"]),
        max_rationale_tokens=int(config["max_rationale_tokens"]),
    )
    reference_grammar = compile_guardian_grammar(
        reference_tokenizer,
        min_rationale_content_tokens=int(config["min_rationale_content_tokens"]),
        max_rationale_tokens=int(config["max_rationale_tokens"]),
    )
    fingerprint = tokenizer_fingerprint(tokenizer, actor_grammar)
    reference_fingerprint = tokenizer_fingerprint(
        reference_tokenizer, reference_grammar
    )
    if fingerprint != reference_fingerprint:
        raise TrainingInvariantError(
            "Actor and Reference tokenizer/grammar fingerprints differ"
        )
    compatibility = _checkpoint_compatibility(preflight, fingerprint)
    resume_checkpoint = _resolve_resume_checkpoint(
        config, preflight, compatibility=compatibility
    )

    actor_path = (
        resume_checkpoint / "actor"
        if resume_checkpoint is not None
        else actor_initial
    )
    actor = _model_from_pretrained(actor_path, dtype=torch.float32)
    reference = _model_from_pretrained(reference_path, dtype=torch.bfloat16)
    reference.eval()
    reference.requires_grad_(False)
    actor.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    actor.config.use_cache = False

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    if resume_checkpoint is None:
        state = ConstrainedTrainerState(
            global_step=0,
            epoch=0,
            protocol_version=str(preflight["protocol_version"]),
            manifest_sha256=str(preflight["manifest_sha256"]),
            sample_cursor=0,
            shuffle_seed=seed,
            data_sha256=str(preflight["train_file_sha256"]),
            config_sha256=str(preflight["semantic_config_sha256"]),
            tokenizer_fingerprint=fingerprint,
        )
    else:
        state = restore_training_state(
            resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            **compatibility,
        )
        if state.shuffle_seed != seed:
            raise TrainingInvariantError(
                f"checkpoint shuffle seed {state.shuffle_seed} != config seed {seed}"
            )

    training_rows = _load_training_rows(train_path)
    validation_rows = _read_jsonl(validation_path)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    train_metrics_path = output_root / "train_metrics.jsonl"
    validation_metrics_path = output_root / "validation_metrics.jsonl"
    rollouts_path = output_root / "rollouts.jsonl"
    summary_path = output_root / "run_summary.json"
    maximum_steps = config.get("max_steps")
    if maximum_steps is not None:
        maximum_steps = int(maximum_steps)
        if maximum_steps <= 0:
            raise TrainingInvariantError("max_steps must be positive when set")
    total_epochs = int(config["epochs"])
    if resume_checkpoint is None and (
        validation_metrics_path.exists() or (output_root / "best_actor").exists()
    ):
        raise TrainingInvariantError(
            "validation/best-model artifacts exist without a resumable checkpoint"
        )
    # Use the CUDA default generator: its exact state is captured in every
    # checkpoint.  A separate torch.Generator would need a second, otherwise
    # easy-to-forget resume state and would make rollouts diverge after resume.
    generator = None

    consecutive_zero_signal_steps = state.consecutive_zero_signal_steps
    status = "running"
    last_checkpoint: Path | None = resume_checkpoint
    summary: dict[str, Any] = {
        "stage": "E5_multistep_constrained_grpo",
        "session_id": session_id,
        "status": status,
        "formal_training_started": True,
        "model_loaded": True,
        "optimizer_created": True,
        "start_global_step": state.global_step,
        "start_epoch": state.epoch,
        "resume_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "preflight": dict(preflight),
    }
    _write_json(summary_path, summary)

    try:
        resume_validation_due = (
            resume_checkpoint is not None
            and state.global_step > 0
            and (
                state.global_step % int(config["validate_every_steps"]) == 0
                or state.epoch >= total_epochs
            )
        )
        if resume_validation_due:
            state, validation_metrics, ran_validation = _validate_and_select(
                actor=actor,
                grammar=actor_grammar,
                tokenizer=tokenizer,
                validation_rows=validation_rows,
                output_root=output_root,
                checkpoint=resume_checkpoint,
                state=state,
                session_id=session_id,
                validation_metrics_path=validation_metrics_path,
                rollouts_path=rollouts_path,
            )
            print(
                f"validation_reconcile step={state.global_step} "
                f"executed={int(ran_validation)} "
                f"reward={float(validation_metrics['mean_dense_reward']):.4f} "
                f"macro_f1={float(validation_metrics['harmfulness_macro_f1']):.4f}",
                flush=True,
            )
        while state.epoch < total_epochs:
            if maximum_steps is not None and state.global_step >= maximum_steps:
                break
            order = epoch_indices(
                row_count=len(training_rows), epoch=state.epoch, seed=seed
            )
            batches = optimizer_step_batches(
                order,
                sample_cursor=state.sample_cursor,
                groups_per_step=int(config["prompt_groups_per_step"]),
            )
            if not batches:
                state = replace(state, epoch=state.epoch + 1, sample_cursor=0)
                continue
            for group_indices in batches:
                if maximum_steps is not None and state.global_step >= maximum_steps:
                    break
                step_started = perf_counter()
                step_number = state.global_step + 1
                epoch_number = state.epoch
                optimizer.zero_grad(set_to_none=True)
                groups: list[PromptGroupResult] = []
                scale = 1.0 / (
                    len(group_indices) * int(config["rollouts_per_prompt"])
                )
                for row_index in group_indices:
                    result = _train_prompt_group(
                        actor=actor,
                        reference=reference,
                        actor_grammar=actor_grammar,
                        reference_grammar=reference_grammar,
                        tokenizer=tokenizer,
                        row=training_rows[row_index],
                        rollouts_per_prompt=int(config["rollouts_per_prompt"]),
                        temperature=float(config["temperature"]),
                        top_p=float(config["top_p"]),
                        generator=generator,
                        scale=scale,
                        clip_ratio=float(config["ppo_clip_ratio"]),
                        kl_coefficient=float(config["kl_coefficient"]),
                        parity_tolerance=float(config["old_new_parity_tolerance"]),
                        session_id=session_id,
                        global_step=step_number,
                        rollouts_path=rollouts_path,
                    )
                    groups.append(result)
                    print(
                        f"group step={step_number} source={result.source_identity} "
                        f"rewards={list(result.rewards)} "
                        f"variable={int(result.variable_group)}",
                        flush=True,
                    )

                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    float(config["max_grad_norm"]),
                    error_if_nonfinite=True,
                )
                gradient_norm = float(gradient_norm_tensor.detach().item())
                if not math.isfinite(gradient_norm):
                    raise TrainingInvariantError("gradient norm is non-finite")
                optimizer.step()
                scheduler.step()
                consumed = state.sample_cursor + len(group_indices)
                next_epoch = state.epoch
                next_cursor = consumed
                if next_cursor == len(training_rows):
                    next_epoch += 1
                    next_cursor = 0
                state = replace(
                    state,
                    global_step=step_number,
                    epoch=next_epoch,
                    sample_cursor=next_cursor,
                )

                variable_groups = sum(group.variable_group for group in groups)
                consecutive_zero_signal_steps = (
                    0
                    if variable_groups
                    else consecutive_zero_signal_steps + 1
                )
                state = replace(
                    state,
                    consecutive_zero_signal_steps=consecutive_zero_signal_steps,
                )
                metrics = _step_metrics(
                    session_id=session_id,
                    state=state,
                    epoch=epoch_number,
                    groups=groups,
                    gradient_norm=gradient_norm,
                    learning_rate=float(scheduler.get_last_lr()[0]),
                    elapsed_seconds=perf_counter() - step_started,
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                    peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
                )
                append_jsonl(train_metrics_path, metrics)
                print(
                    f"step={state.global_step}/"
                    f"{preflight['total_optimizer_steps']} "
                    f"reward={metrics['mean_reward']:.4f} "
                    f"variable_groups={variable_groups}/{len(groups)} "
                    f"loss={metrics['total_loss']:.6f} "
                    f"grad_norm={gradient_norm:.4f} "
                    f"seconds={metrics['elapsed_seconds']:.1f}",
                    flush=True,
                )

                formal_final = state.epoch >= total_epochs
                bounded_final = (
                    maximum_steps is not None
                    and state.global_step >= maximum_steps
                )
                validation_due = (
                    state.global_step % int(config["validate_every_steps"]) == 0
                    or formal_final
                )
                checkpoint_due = (
                    state.global_step % int(config["checkpoint_every_steps"]) == 0
                    or validation_due
                    or bounded_final
                )
                if checkpoint_due:
                    last_checkpoint = _save_checkpoint(
                        output_root=output_root,
                        actor=actor,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        minimum_free_gb=float(
                            config["minimum_free_gb_before_checkpoint"]
                        ),
                    )
                    print(
                        f"checkpoint step={state.global_step} path={last_checkpoint}",
                        flush=True,
                    )

                if validation_due:
                    if last_checkpoint is None:
                        raise TrainingInvariantError(
                            "validation was reached without a resumable checkpoint"
                        )
                    state, validation_metrics, _ = _validate_and_select(
                        actor=actor,
                        grammar=actor_grammar,
                        tokenizer=tokenizer,
                        validation_rows=validation_rows,
                        output_root=output_root,
                        checkpoint=last_checkpoint,
                        state=state,
                        session_id=session_id,
                        validation_metrics_path=validation_metrics_path,
                        rollouts_path=rollouts_path,
                    )
                    print(
                        f"validation step={state.global_step} "
                        f"reward={validation_metrics['mean_dense_reward']:.4f} "
                        f"macro_f1={validation_metrics['harmfulness_macro_f1']:.4f}",
                        flush=True,
                    )

                if consecutive_zero_signal_steps >= int(
                    config["max_consecutive_zero_signal_steps"]
                ):
                    if not checkpoint_due:
                        last_checkpoint = _save_checkpoint(
                            output_root=output_root,
                            actor=actor,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            state=state,
                            minimum_free_gb=float(
                                config["minimum_free_gb_before_checkpoint"]
                            ),
                        )
                    raise TrainingInvariantError(
                        "ten consecutive optimizer steps had no variable-reward "
                        "prompt group"
                    )
                gc.collect()

            if maximum_steps is not None and state.global_step >= maximum_steps:
                break

        status = "complete" if state.epoch >= total_epochs else "bounded_complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        summary.update(
            {
                "status": status,
                "end_global_step": state.global_step,
                "end_epoch": state.epoch,
                "end_sample_cursor": state.sample_cursor,
                "last_checkpoint": (
                    None if last_checkpoint is None else str(last_checkpoint)
                ),
                "best_actor": (
                    str(output_root / "best_actor")
                    if (output_root / "best_actor").is_dir()
                    else None
                ),
                "peak_cuda_memory_allocated_gb": (
                    torch.cuda.max_memory_allocated() / 1024**3
                ),
                "peak_cuda_memory_reserved_gb": (
                    torch.cuda.max_memory_reserved() / 1024**3
                ),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(summary_path, summary)


def main() -> None:
    args = _parse_args()
    config = load_training_config(
        args.config,
        overrides={"resume": args.resume, "max_steps": args.max_steps},
    )
    report = build_preflight_report(config)
    if args.preflight_report is not None:
        _write_json(args.preflight_report, report)
    if args.preflight_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    run_training(config, report)


if __name__ == "__main__":
    main()

"""Pure math and checkpoint boundaries for constrained Guardian GRPO."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import torch


class CheckpointCompatibilityError(RuntimeError):
    """Raised before restore when a checkpoint belongs to another protocol/data run."""


@dataclass(frozen=True)
class ConstrainedTrainerState:
    global_step: int
    epoch: int
    protocol_version: str
    manifest_sha256: str
    sample_cursor: int = 0
    shuffle_seed: int = 0
    data_sha256: str = ""
    config_sha256: str = ""
    tokenizer_fingerprint: str = ""
    best_validation_reward: float | None = None
    best_validation_macro_f1: float | None = None
    consecutive_zero_signal_steps: int = 0


@dataclass(frozen=True)
class GroupAdvantageStats:
    variable_group_count: int
    all_equal_group_count: int


@dataclass(frozen=True)
class GRPOLoss:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor


def _ground_truth_attacked(ground_truth: Mapping[str, object]) -> object:
    has_attacked = "Being_Attacked" in ground_truth
    has_injection = "Prompt_Injection" in ground_truth
    if has_attacked and has_injection:
        attacked = ground_truth["Being_Attacked"]
        injection = ground_truth["Prompt_Injection"]
        if attacked != injection:
            raise ValueError(
                "ground truth has conflicting Being_Attacked and Prompt_Injection values"
            )
        return attacked
    if has_attacked:
        return ground_truth["Being_Attacked"]
    if has_injection:
        return ground_truth["Prompt_Injection"]
    raise KeyError("ground truth lacks Being_Attacked/Prompt_Injection")


def dense_guardian_reward(
    predicted: Mapping[str, object], ground_truth: Mapping[str, object]
) -> float:
    """Score only the three semantic decisions; formatting and rationale score nothing."""

    matches = (
        predicted["Malicious_User_Request"]
        == ground_truth["Malicious_User_Request"],
        predicted["Being_Attacked"] == _ground_truth_attacked(ground_truth),
        predicted["Harmfulness_Rating"] == ground_truth["Harmfulness_Rating"],
    )
    reward = sum(weight for weight, matched in zip((0.2, 0.2, 0.3), matches) if matched)
    if all(matches):
        reward += 0.3
    return float(reward)


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: Sequence[Hashable],
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, GroupAdvantageStats]:
    """Normalize rewards within each prompt group using population variance."""

    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    if len(group_ids) != rewards.numel():
        raise ValueError("group_ids length must match rewards")

    advantages = torch.zeros_like(rewards)
    ordered_groups = list(dict.fromkeys(group_ids))
    variable_count = 0
    equal_count = 0
    for group_id in ordered_groups:
        indices = [index for index, value in enumerate(group_ids) if value == group_id]
        index_tensor = torch.tensor(indices, device=rewards.device)
        group_rewards = rewards[index_tensor]
        std = group_rewards.std(unbiased=False)
        if float(std.detach()) <= eps:
            equal_count += 1
            continue
        variable_count += 1
        advantages[index_tensor] = (group_rewards - group_rewards.mean()) / std.clamp_min(eps)

    return advantages, GroupAdvantageStats(
        variable_group_count=variable_count,
        all_equal_group_count=equal_count,
    )


def grpo_response_loss(
    new_log_probs: torch.Tensor,
    *,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantage: float | torch.Tensor,
    clip_ratio: float,
    kl_coefficient: float,
) -> GRPOLoss:
    """Compute clipped token-level GRPO loss plus the sampled nonnegative KL."""

    ratio = torch.exp(new_log_probs - old_log_probs)
    advantage_tensor = torch.as_tensor(
        advantage, dtype=new_log_probs.dtype, device=new_log_probs.device
    )
    unclipped = ratio * advantage_tensor
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_tensor
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    delta = reference_log_probs - new_log_probs
    kl = torch.exp(delta) - delta - 1.0
    kl_loss = kl.mean()
    total_loss = policy_loss + kl_coefficient * kl_loss
    clip_fraction = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean()
    return GRPOLoss(
        total_loss=total_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        clip_fraction=clip_fraction,
    )


def save_training_checkpoint(
    checkpoint: str | Path,
    *,
    actor: torch.nn.Module,
    tokenizer: object | None,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    state: ConstrainedTrainerState,
) -> None:
    """Write a complete resumable checkpoint into a new, non-overwritten directory."""

    checkpoint_path = Path(checkpoint)
    checkpoint_path.mkdir(parents=True, exist_ok=False)
    if hasattr(actor, "save_pretrained"):
        actor_dir = checkpoint_path / "actor"
        actor.save_pretrained(actor_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(actor_dir)
    else:
        torch.save(actor.state_dict(), checkpoint_path / "actor_state.pt")

    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")
    rng_state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(rng_state, checkpoint_path / "rng_state.pt")
    (checkpoint_path / "trainer_state.json").write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_checkpoint_metadata(
    checkpoint: str | Path,
    *,
    expected_protocol_version: str,
    expected_manifest_sha256: str,
    expected_data_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_tokenizer_fingerprint: str | None = None,
) -> ConstrainedTrainerState:
    """Fail closed on incompatible metadata before any tensor state is restored."""

    metadata_path = Path(checkpoint) / "trainer_state.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = ConstrainedTrainerState(**payload)
    if state.protocol_version != expected_protocol_version:
        raise CheckpointCompatibilityError(
            "checkpoint protocol mismatch: "
            f"expected {expected_protocol_version!r}, got {state.protocol_version!r}"
        )
    if state.manifest_sha256 != expected_manifest_sha256:
        raise CheckpointCompatibilityError(
            "checkpoint manifest mismatch: "
            f"expected {expected_manifest_sha256!r}, got {state.manifest_sha256!r}"
        )
    optional_expectations = (
        ("data", expected_data_sha256, state.data_sha256),
        ("config", expected_config_sha256, state.config_sha256),
        (
            "tokenizer fingerprint",
            expected_tokenizer_fingerprint,
            state.tokenizer_fingerprint,
        ),
    )
    for name, expected, actual in optional_expectations:
        if expected is not None and actual != expected:
            raise CheckpointCompatibilityError(
                f"checkpoint {name} mismatch: expected {expected!r}, got {actual!r}"
            )
    return state


def _move_optimizer_state_to_parameter_devices(
    optimizer: torch.optim.Optimizer,
) -> None:
    for parameter, parameter_state in optimizer.state.items():
        for name, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[name] = value.to(parameter.device)


def _restore_rng_state(path: Path) -> None:
    rng_state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch"])
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def restore_training_state(
    checkpoint: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expected_protocol_version: str,
    expected_manifest_sha256: str,
    expected_data_sha256: str,
    expected_config_sha256: str,
    expected_tokenizer_fingerprint: str,
    restore_rng: bool = True,
) -> ConstrainedTrainerState:
    """Restore optimizer/scheduler/RNG only after metadata compatibility passes."""

    checkpoint_path = Path(checkpoint)
    state = load_checkpoint_metadata(
        checkpoint_path,
        expected_protocol_version=expected_protocol_version,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_data_sha256=expected_data_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_tokenizer_fingerprint=expected_tokenizer_fingerprint,
    )
    optimizer_path = checkpoint_path / "optimizer.pt"
    scheduler_path = checkpoint_path / "scheduler.pt"
    rng_path = checkpoint_path / "rng_state.pt"
    for required in (optimizer_path, scheduler_path, rng_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    optimizer.load_state_dict(
        torch.load(optimizer_path, map_location="cpu", weights_only=False)
    )
    _move_optimizer_state_to_parameter_devices(optimizer)
    scheduler.load_state_dict(
        torch.load(scheduler_path, map_location="cpu", weights_only=False)
    )
    if restore_rng:
        _restore_rng_state(rng_path)
    return state

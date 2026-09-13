"""Focused E4 tests for constrained GRPO math and checkpoint boundaries."""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from constrained_grpo_core import (  # noqa: E402
    CheckpointCompatibilityError,
    ConstrainedTrainerState,
    dense_guardian_reward,
    grpo_response_loss,
    group_relative_advantages,
    load_checkpoint_metadata,
    restore_training_state,
    save_training_checkpoint,
)


def test_dense_reward_is_semantic_only_and_supports_partial_credit() -> None:
    ground_truth = {
        "Malicious_User_Request": True,
        "Prompt_Injection": False,
        "Harmfulness_Rating": 1.0,
    }

    partial = dense_guardian_reward(
        {
            "Malicious_User_Request": True,
            "Being_Attacked": True,
            "Harmfulness_Rating": 0.5,
        },
        ground_truth,
    )
    full = dense_guardian_reward(
        {
            "Malicious_User_Request": True,
            "Being_Attacked": False,
            "Harmfulness_Rating": 1.0,
        },
        ground_truth,
    )

    assert partial == pytest.approx(0.2)
    assert full == pytest.approx(1.0)


def test_group_advantages_are_prompt_local_and_equal_groups_are_zero() -> None:
    advantages, stats = group_relative_advantages(
        torch.tensor([0.0, 1.0, 2.0, 5.0, 5.0]),
        group_ids=("a", "a", "a", "b", "b"),
    )

    scale = math.sqrt(2.0 / 3.0)
    assert advantages[:3].tolist() == pytest.approx(
        [-1.0 / scale, 0.0, 1.0 / scale]
    )
    assert advantages[3:].tolist() == [0.0, 0.0]
    assert stats.variable_group_count == 1
    assert stats.all_equal_group_count == 1


def test_grpo_loss_clips_policy_ratio_and_uses_nonnegative_sampled_kl() -> None:
    new_log_probs = torch.tensor([math.log(1.5), 0.0], requires_grad=True)
    old_log_probs = torch.tensor([0.0, 0.0])
    reference_log_probs = torch.tensor([math.log(1.5) - 1.0, -1.0])

    result = grpo_response_loss(
        new_log_probs,
        old_log_probs=old_log_probs,
        reference_log_probs=reference_log_probs,
        advantage=1.0,
        clip_ratio=0.2,
        kl_coefficient=0.05,
    )
    result.total_loss.backward()

    assert result.policy_loss.item() == pytest.approx(-1.1, abs=1e-6)
    assert result.kl_loss.item() == pytest.approx(math.exp(-1.0), abs=1e-6)
    assert result.clip_fraction.item() == pytest.approx(0.5)
    assert new_log_probs.grad is not None
    assert torch.isfinite(new_log_probs.grad).all()


def test_checkpoint_metadata_rejects_protocol_or_manifest_mismatch(tmp_path: Path) -> None:
    actor = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = ConstrainedTrainerState(
        global_step=3,
        epoch=1,
        protocol_version="toolsafe-guardian-token-fsm-v2",
        manifest_sha256="abc123",
    )
    checkpoint = tmp_path / "step_3"

    save_training_checkpoint(
        checkpoint,
        actor=actor,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
    )
    loaded = load_checkpoint_metadata(
        checkpoint,
        expected_protocol_version="toolsafe-guardian-token-fsm-v2",
        expected_manifest_sha256="abc123",
    )

    assert loaded == state
    assert (checkpoint / "actor_state.pt").is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "rng_state.pt").is_file()
    assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == 3
    with pytest.raises(CheckpointCompatibilityError, match="protocol"):
        load_checkpoint_metadata(
            checkpoint,
            expected_protocol_version="wrong-protocol",
            expected_manifest_sha256="abc123",
        )
    with pytest.raises(CheckpointCompatibilityError, match="manifest"):
        load_checkpoint_metadata(
            checkpoint,
            expected_protocol_version="toolsafe-guardian-token-fsm-v2",
            expected_manifest_sha256="wrong-manifest",
        )


def _make_tiny_optimizer(model: torch.nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _take_tiny_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    batch: torch.Tensor,
    target: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(batch), target).backward()
    optimizer.step()
    scheduler.step()


def test_checkpoint_restore_reproduces_next_optimizer_step_and_rng(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    random.seed(17)
    control = torch.nn.Linear(3, 2)
    interrupted = copy.deepcopy(control)
    control_optimizer, control_scheduler = _make_tiny_optimizer(control)
    interrupted_optimizer, interrupted_scheduler = _make_tiny_optimizer(interrupted)
    batch = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[0.5, -0.5]])

    _take_tiny_step(
        control, control_optimizer, control_scheduler, batch, target
    )
    _take_tiny_step(
        interrupted,
        interrupted_optimizer,
        interrupted_scheduler,
        batch,
        target,
    )
    checkpoint = tmp_path / "step_1"
    state = ConstrainedTrainerState(
        global_step=1,
        epoch=0,
        protocol_version="toolsafe-guardian-token-fsm-v2",
        manifest_sha256="manifest",
        sample_cursor=4,
        shuffle_seed=20260909,
        data_sha256="data",
        config_sha256="config",
        tokenizer_fingerprint="tokenizer",
        best_validation_reward=0.8,
        best_validation_macro_f1=0.6,
        consecutive_zero_signal_steps=7,
    )
    save_training_checkpoint(
        checkpoint,
        actor=interrupted,
        tokenizer=None,
        optimizer=interrupted_optimizer,
        scheduler=interrupted_scheduler,
        state=state,
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(3)
    _take_tiny_step(
        control, control_optimizer, control_scheduler, batch, target
    )

    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(
        torch.load(checkpoint / "actor_state.pt", weights_only=True)
    )
    restored_optimizer, restored_scheduler = _make_tiny_optimizer(restored)
    random.seed(999)
    torch.manual_seed(999)
    loaded = restore_training_state(
        checkpoint,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_protocol_version="toolsafe-guardian-token-fsm-v2",
        expected_manifest_sha256="manifest",
        expected_data_sha256="data",
        expected_config_sha256="config",
        expected_tokenizer_fingerprint="tokenizer",
    )

    assert loaded == state
    assert random.random() == expected_python_random
    torch.testing.assert_close(torch.rand(3), expected_torch_random, rtol=0, atol=0)
    _take_tiny_step(
        restored, restored_optimizer, restored_scheduler, batch, target
    )
    for expected, actual in zip(control.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_scheduler.state_dict() == control_scheduler.state_dict()


@pytest.mark.parametrize(
    ("expected_field", "expected_value"),
    (
        ("expected_data_sha256", "wrong-data"),
        ("expected_config_sha256", "wrong-config"),
        ("expected_tokenizer_fingerprint", "wrong-tokenizer"),
    ),
)
def test_restore_rejects_extended_metadata_before_reading_tensor_state(
    tmp_path: Path,
    expected_field: str,
    expected_value: str,
) -> None:
    actor = torch.nn.Linear(3, 2)
    optimizer, scheduler = _make_tiny_optimizer(actor)
    checkpoint = tmp_path / expected_field
    save_training_checkpoint(
        checkpoint,
        actor=actor,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        state=ConstrainedTrainerState(
            global_step=1,
            epoch=0,
            protocol_version="toolsafe-guardian-token-fsm-v2",
            manifest_sha256="manifest",
            data_sha256="data",
            config_sha256="config",
            tokenizer_fingerprint="tokenizer",
        ),
    )
    (checkpoint / "optimizer.pt").write_bytes(b"not a torch checkpoint")
    compatibility = {
        "expected_protocol_version": "toolsafe-guardian-token-fsm-v2",
        "expected_manifest_sha256": "manifest",
        "expected_data_sha256": "data",
        "expected_config_sha256": "config",
        "expected_tokenizer_fingerprint": "tokenizer",
    }
    compatibility[expected_field] = expected_value

    with pytest.raises(CheckpointCompatibilityError, match="mismatch"):
        restore_training_state(
            checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            **compatibility,
        )

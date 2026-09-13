"""Focused tests for the no-update GRPO rollout acceptance gate."""

from __future__ import annotations

import hashlib

import pytest

from practice.toolsafe_reproduction.grpo.rollout_acceptance import (
    checkpoint_weight_sha256,
    select_stratified_rows,
    summarize_acceptance,
)


def _row(identity: str, subset: str, score: float) -> dict:
    return {
        "extra_info": {
            "dataset": "agentdojo" if subset != "harmful" else "agentharm",
            "subset": subset,
            "source_identity": identity,
        },
        "reward_model": {
            "ground_truth": {"Harmfulness_Rating": score},
        },
    }


def test_select_stratified_rows_round_robins_across_source_and_score():
    rows = [
        _row("slack-0-a", "slack", 0.0),
        _row("slack-0-b", "slack", 0.0),
        _row("slack-1-a", "slack", 1.0),
        _row("travel-0-a", "travel", 0.0),
        _row("travel-1-a", "travel", 1.0),
        _row("harm-05-a", "harmful", 0.5),
    ]

    selected = select_stratified_rows(rows, count=5)

    identities = [row["extra_info"]["source_identity"] for row in selected]
    assert len(identities) == len(set(identities)) == 5
    assert "slack-0-b" not in identities
    assert {row["extra_info"]["subset"] for row in selected} == {
        "slack",
        "travel",
        "harmful",
    }


def test_summarize_acceptance_requires_parse_reward_and_group_variation_rates():
    groups = [
        {
            "outputs": [
                {"parsed": True, "reward": 1.0},
                {"parsed": True, "reward": 0.67},
            ]
        },
        {
            "outputs": [
                {"parsed": True, "reward": 0.33},
                {"parsed": False, "reward": 0.0},
            ]
        },
    ]

    report = summarize_acceptance(groups)

    assert report["total_rollouts"] == 4
    assert report["parse_rate"] == 0.75
    assert report["nonzero_reward_rate"] == 0.75
    assert report["variable_reward_group_rate"] == 1.0
    assert report["reward_counts"] == {"0.0": 1, "0.33": 1, "0.67": 1, "1.0": 1}
    assert report["passed"] is True


def test_summarize_acceptance_rejects_groups_with_no_relative_signal():
    groups = [
        {"outputs": [{"parsed": False, "reward": 0.0} for _ in range(8)]},
        {"outputs": [{"parsed": True, "reward": 1.0} for _ in range(8)]},
    ]

    report = summarize_acceptance(groups)

    assert report["parse_rate"] == 0.5
    assert report["nonzero_reward_rate"] == 0.5
    assert report["variable_reward_group_rate"] == 0.0
    assert report["criteria"]["variable_reward_group_rate"] is False
    assert report["passed"] is False


def test_checkpoint_weight_sha256_preserves_single_file_digest(tmp_path):
    weight_bytes = b"single-weight-file"
    (tmp_path / "model.safetensors").write_bytes(weight_bytes)

    digest = checkpoint_weight_sha256(tmp_path)

    assert digest == hashlib.sha256(weight_bytes).hexdigest()


def test_checkpoint_weight_sha256_has_stable_sharded_digest(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_bytes(
        b'{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        b'"b":"model-00002-of-00002.safetensors"}}'
    )
    first_shard = tmp_path / "model-00001-of-00002.safetensors"
    second_shard = tmp_path / "model-00002-of-00002.safetensors"
    first_shard.write_bytes(b"first-shard")
    second_shard.write_bytes(b"second-shard")

    first_digest = checkpoint_weight_sha256(tmp_path)
    repeated_digest = checkpoint_weight_sha256(tmp_path)

    assert first_digest == "65961f8bb69953f9a4b8edbd5ba30e2dcf61edf131962441db4ebc059b497c9a"
    assert repeated_digest == first_digest
    second_shard.write_bytes(b"changed-second-shard")
    assert checkpoint_weight_sha256(tmp_path) != first_digest


def test_checkpoint_weight_sha256_rejects_missing_indexed_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"missing.safetensors"}}', encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        checkpoint_weight_sha256(tmp_path)

"""Focused safety contracts for the one-A800 GRPO launcher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from preflight_grpo import (  # noqa: E402
    build_hydra_overrides,
    load_launch_config,
    run_preflight,
)
from rollout_acceptance import checkpoint_weight_sha256  # noqa: E402


def _config_file(
    tmp_path: Path,
    *,
    output_dir: Path | None = None,
    sharded: bool = False,
) -> Path:
    actor = tmp_path / "models" / "policy_init"
    reference = tmp_path / "models" / "reference"
    actor.mkdir(parents=True)
    reference.mkdir(parents=True)
    for directory in (actor, reference):
        (directory / "config.json").write_text(
            '{"model_type":"qwen2"}', encoding="utf-8"
        )
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        if sharded:
            (directory / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
                '"b":"model-00002-of-00002.safetensors"}}',
                encoding="utf-8",
            )
            (directory / "model-00001-of-00002.safetensors").write_bytes(
                b"same-dense-model-first-shard"
            )
            (directory / "model-00002-of-00002.safetensors").write_bytes(
                b"same-dense-model-second-shard"
            )
        else:
            (directory / "model.safetensors").write_bytes(b"same-dense-model")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"train")
    (data / "validation.parquet").write_bytes(b"validation")
    reward = tmp_path / "agentsafety_v2_uniform.py"
    reward.write_text("def compute_score(): pass\n", encoding="utf-8")
    (tmp_path / "verl-main").mkdir()
    digest = checkpoint_weight_sha256(actor)
    config = {
        "paths": {
            "actor_model": str(actor),
            "reference_model": str(reference),
            "train_file": str(data / "train.parquet"),
            "validation_file": str(data / "validation.parquet"),
            "reward_function": str(reward),
            "output_dir": str(output_dir or tmp_path / "outputs"),
            "verl_root": str(tmp_path / "verl-main"),
        },
        "integrity": {"checkpoint_weight_sha256": digest},
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 256,
            "seed": 20260830,
        },
        "rollout": {
            "name": "vllm",
            "n": 8,
            "temperature": 1.0,
            "top_p": 1.0,
            "tensor_model_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 128,
        },
        "actor": {
            "attention_implementation": "sdpa",
            "learning_rate": 1e-6,
            "ppo_mini_batch_size": 16,
            "ppo_micro_batch_size_per_gpu": 4,
            "log_prob_micro_batch_size_per_gpu": 8,
            "max_token_len_per_gpu": 18432,
            "kl_loss_coef": 0.001,
            "clip_ratio": 0.2,
            "gradient_checkpointing": True,
        },
        "trainer": {
            "experiment_name": "test_qwen25_3b_full_grpo",
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "total_epochs": 1,
            "save_freq": 20,
            "test_freq": 20,
            "max_actor_ckpt_to_keep": 2,
        },
    }
    path = tmp_path / "a800.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_preflight_validates_separate_identical_actor_and_reference(tmp_path):
    path = _config_file(tmp_path)

    report = run_preflight(path, resolve_hydra=False)

    assert report["status"] == "ready"
    assert report["training_started"] is False
    assert report["n_gpus_per_node"] == 1
    assert report["policy_reference_hashes_equal"] is True


def test_preflight_validates_separate_identical_sharded_actor_and_reference(tmp_path):
    path = _config_file(tmp_path, sharded=True)

    report = run_preflight(path, resolve_hydra=False)

    assert report["status"] == "ready"
    assert report["policy_reference_hashes_equal"] is True


def test_preflight_can_explicitly_skip_weight_hashing_for_local_checkpoint_copies(tmp_path):
    path = _config_file(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["integrity"]["checkpoint_weight_sha256"] = None
    reference_weight = (
        Path(config["paths"]["reference_model"]) / "model.safetensors"
    )
    reference_weight.write_bytes(b"a-separate-local-copy-not-read-by-preflight")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run_preflight(path, resolve_hydra=False)

    assert report["integrity_mode"] == "path_and_structure_only"
    assert report["actor_weight_sha256"] is None
    assert report["reference_weight_sha256"] is None
    assert report["policy_reference_hashes_equal"] is None


def test_preflight_skip_hash_still_rejects_an_incomplete_model_directory(tmp_path):
    path = _config_file(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["integrity"]["checkpoint_weight_sha256"] = None
    (Path(config["paths"]["reference_model"]) / "config.json").unlink()
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="config.json"):
        run_preflight(path, resolve_hydra=False)


def test_preflight_rejects_output_inside_canonical_actor(tmp_path):
    actor_output = tmp_path / "models" / "policy_init" / "checkpoints"
    path = _config_file(tmp_path, output_dir=actor_output)

    with pytest.raises(ValueError, match="outside actor and reference"):
        run_preflight(path, resolve_hydra=False)


def test_hydra_overrides_keep_grpo_kl_group_sampling_and_reference_path(tmp_path):
    config = load_launch_config(_config_file(tmp_path))

    overrides = build_hydra_overrides(config)

    assert "algorithm.adv_estimator=grpo" in overrides
    assert "algorithm.use_kl_in_reward=false" in overrides
    assert "actor_rollout_ref.actor.use_kl_loss=true" in overrides
    assert "actor_rollout_ref.actor.kl_loss_coef=0.001" in overrides
    assert (
        "+actor_rollout_ref.model.override_config._attn_implementation=sdpa"
        in overrides
    )
    assert "actor_rollout_ref.rollout.n=8" in overrides
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in overrides
    assert "trainer.experiment_name=test_qwen25_3b_full_grpo" in overrides
    assert any(
        item.startswith("+actor_rollout_ref.ref.model.path=")
        and item.endswith("/models/reference")
        for item in overrides
    )

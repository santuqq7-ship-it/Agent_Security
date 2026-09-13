"""Validate and resolve the ToolSafe single-A800 GRPO launch without training.

The default command is deliberately non-mutating: it verifies paths, hashes,
and Hydra overrides.  A real optimizer can start only when the user invokes
the separate shell wrapper, which passes ``--launch`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from rollout_acceptance import checkpoint_weight_sha256


EXPECTED_SECTIONS = frozenset(
    {"paths", "integrity", "data", "rollout", "actor", "trainer"}
)


def load_launch_config(path: Path) -> dict[str, Any]:
    """Load one explicit launch configuration and reject missing sections."""

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("GRPO launch config must be a YAML object")
    missing = EXPECTED_SECTIONS - loaded.keys()
    if missing:
        raise ValueError(f"launch config is missing: {', '.join(sorted(missing))}")
    return loaded


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _bool(value: bool) -> str:
    return "true" if value else "false"


def build_hydra_overrides(config: dict[str, Any]) -> list[str]:
    """Map the reviewed YAML values onto the recovered official verl config."""

    paths = config["paths"]
    data = config["data"]
    rollout = config["rollout"]
    actor = config["actor"]
    trainer = config["trainer"]
    output_dir = Path(paths["output_dir"])

    return [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        f"data.train_files={paths['train_file']}",
        f"data.val_files={paths['validation_file']}",
        f"data.train_batch_size={int(data['train_batch_size'])}",
        f"data.max_prompt_length={int(data['max_prompt_length'])}",
        f"data.max_response_length={int(data['max_response_length'])}",
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        f"data.seed={int(data['seed'])}",
        f"actor_rollout_ref.model.path={paths['actor_model']}",
        "+actor_rollout_ref.model.override_config._attn_implementation="
        + str(actor["attention_implementation"]),
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.model.enable_gradient_checkpointing="
        + _bool(bool(actor["gradient_checkpointing"])),
        f"+actor_rollout_ref.ref.model.path={paths['reference_model']}",
        f"actor_rollout_ref.actor.optim.lr={float(actor['learning_rate'])}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(actor['ppo_mini_batch_size'])}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
        + str(int(actor["ppo_micro_batch_size_per_gpu"])),
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={int(actor['max_token_len_per_gpu'])}",
        "actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef={float(actor['kl_loss_coef'])}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.clip_ratio={float(actor['clip_ratio'])}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_torch_compile=false",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",
        f"actor_rollout_ref.rollout.name={rollout['name']}",
        f"actor_rollout_ref.rollout.n={int(rollout['n'])}",
        f"actor_rollout_ref.rollout.temperature={float(rollout['temperature'])}",
        f"actor_rollout_ref.rollout.top_p={float(rollout['top_p'])}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size="
        + str(int(rollout["tensor_model_parallel_size"])),
        "actor_rollout_ref.rollout.gpu_memory_utilization="
        + str(float(rollout["gpu_memory_utilization"])),
        "actor_rollout_ref.rollout.max_num_batched_tokens="
        + str(int(rollout["max_num_batched_tokens"])),
        f"actor_rollout_ref.rollout.max_num_seqs={int(rollout['max_num_seqs'])}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=true",
        "actor_rollout_ref.rollout.enable_prefix_caching=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
        + str(int(actor["log_prob_micro_batch_size_per_gpu"])),
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
        + str(int(actor["log_prob_micro_batch_size_per_gpu"])),
        f"custom_reward_function.path={paths['reward_function']}",
        "custom_reward_function.name=compute_score",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,tensorboard]",
        "trainer.project_name=toolsafe_grpo_reproduction",
        f"trainer.experiment_name={trainer['experiment_name']}",
        f"trainer.n_gpus_per_node={int(trainer['n_gpus_per_node'])}",
        f"trainer.nnodes={int(trainer['nnodes'])}",
        f"trainer.total_epochs={int(trainer['total_epochs'])}",
        f"trainer.save_freq={int(trainer['save_freq'])}",
        f"trainer.test_freq={int(trainer['test_freq'])}",
        f"trainer.max_actor_ckpt_to_keep={int(trainer['max_actor_ckpt_to_keep'])}",
        "trainer.val_before_train=true",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={output_dir / 'checkpoints'}",
        f"trainer.rollout_data_dir={output_dir / 'rollouts'}",
        f"trainer.validation_data_dir={output_dir / 'validation'}",
    ]


def build_launch_command(config: dict[str, Any], *, resolve_only: bool) -> list[str]:
    command = [sys.executable, "-m", "verl.trainer.main_ppo"]
    if resolve_only:
        command.extend(["--cfg", "job"])
    command.extend(build_hydra_overrides(config))
    return command


def _validate_model_structure(model_dir: Path, *, label: str) -> None:
    """Check lightweight Hugging Face structure without reading weight bytes."""

    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise ValueError(f"{label} is missing config.json: {model_dir}")
    if not tokenizer_path.is_file():
        raise ValueError(f"{label} is missing tokenizer_config.json: {model_dir}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict) or not model_config.get("model_type"):
        raise ValueError(f"{label} config.json must contain model_type")
    single_weight = model_dir / "model.safetensors"
    weight_index = model_dir / "model.safetensors.index.json"
    if not single_weight.is_file() and not weight_index.is_file():
        raise ValueError(
            f"{label} contains neither model.safetensors nor "
            "model.safetensors.index.json"
        )


def _validate_paths_and_hashes(config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    for key in (
        "actor_model",
        "reference_model",
        "verl_root",
    ):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"{key} directory not found: {paths[key]}")
    for key in ("train_file", "validation_file", "reward_function"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key} file not found: {paths[key]}")
    if paths["actor_model"].resolve() == paths["reference_model"].resolve():
        raise ValueError("actor and reference must be separate directories")
    output_dir = paths["output_dir"]
    if _is_within(output_dir, paths["actor_model"]) or _is_within(
        output_dir, paths["reference_model"]
    ):
        raise ValueError("output_dir must be outside actor and reference directories")

    _validate_model_structure(paths["actor_model"], label="actor_model")
    _validate_model_structure(paths["reference_model"], label="reference_model")

    configured_hash = config["integrity"].get("checkpoint_weight_sha256")
    if configured_hash is None or not str(configured_hash).strip():
        return {
            "integrity_mode": "path_and_structure_only",
            "actor_weight_sha256": None,
            "reference_weight_sha256": None,
            "policy_reference_hashes_equal": None,
        }

    expected_hash = str(configured_hash).lower()
    actor_hash = checkpoint_weight_sha256(paths["actor_model"])
    reference_hash = checkpoint_weight_sha256(paths["reference_model"])
    if actor_hash != expected_hash or reference_hash != expected_hash:
        raise ValueError(
            "canonical dense weight hash mismatch: "
            f"actor={actor_hash}, reference={reference_hash}, expected={expected_hash}"
        )
    return {
        "integrity_mode": "sha256",
        "actor_weight_sha256": actor_hash,
        "reference_weight_sha256": reference_hash,
        "policy_reference_hashes_equal": actor_hash == reference_hash,
    }


def run_preflight(config_path: Path, *, resolve_hydra: bool = False) -> dict[str, Any]:
    """Validate launch inputs; optionally ask Hydra to compose without running."""

    config = load_launch_config(config_path)
    trainer = config["trainer"]
    rollout = config["rollout"]
    if int(trainer["n_gpus_per_node"]) != 1 or int(trainer["nnodes"]) != 1:
        raise ValueError("this reviewed configuration requires exactly one node and one GPU")
    if int(rollout["tensor_model_parallel_size"]) != 1:
        raise ValueError("single-A800 rollout tensor_model_parallel_size must be 1")
    if int(rollout["n"]) <= 1:
        raise ValueError("GRPO requires more than one rollout per prompt")

    integrity = _validate_paths_and_hashes(config)
    report: dict[str, Any] = {
        "status": "ready",
        "training_started": False,
        "config_path": str(Path(config_path).resolve()),
        "n_gpus_per_node": 1,
        "rollout_group_size": int(rollout["n"]),
        "hydra_resolved": False,
        **integrity,
    }
    if resolve_hydra:
        verl_root = Path(config["paths"]["verl_root"])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(verl_root) + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        completed = subprocess.run(
            build_launch_command(config, resolve_only=True),
            cwd=verl_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Hydra configuration resolution failed:\n"
                + completed.stdout[-4000:]
                + completed.stderr[-4000:]
            )
        report["hydra_resolved"] = True
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resolve-hydra",
        action="store_true",
        help="compose the recovered verl Hydra job and exit without training",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="explicitly replace this process with the real verl trainer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(
        args.config, resolve_hydra=args.resolve_hydra or args.launch
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.launch:
        return

    config = load_launch_config(args.config)
    verl_root = Path(config["paths"]["verl_root"])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(verl_root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    os.chdir(verl_root)
    command = build_launch_command(config, resolve_only=False)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a no-update vLLM rollout gate before paid full-parameter GRPO.

This command deliberately never imports ``verl.trainer`` and never creates an
optimizer.  It samples the current policy with the reviewed rollout settings,
applies the exact ToolSafe rule reward, and checks whether prompt groups carry
enough relative reward variation for GRPO to learn from them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import pstdev
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "practice/toolsafe_reproduction/grpo/config/a800_full_grpo.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "practice/toolsafe_reproduction/results/grpo_rollout_acceptance"


def _stratum(row: dict[str, Any]) -> tuple[str, float]:
    """Identify one source-and-harmfulness stratum for deterministic coverage."""

    extra = row["extra_info"]
    source = f"{extra['dataset']}/{extra['subset']}"
    score = float(row["reward_model"]["ground_truth"]["Harmfulness_Rating"])
    return source, score


def select_stratified_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Round-robin deterministic rows across every available source/score pair."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a dataset containing {len(rows)}")

    buckets: dict[tuple[str, float], deque[dict[str, Any]]] = {}
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_stratum(row)].append(row)
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda item: item["extra_info"]["source_identity"])
        buckets[key] = deque(ordered)

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        made_progress = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].popleft())
                made_progress = True
        if not made_progress:
            raise RuntimeError("stratified selection exhausted before reaching requested count")
    return selected


def summarize_acceptance(
    groups: list[dict[str, Any]],
    *,
    min_parse_rate: float = 0.50,
    min_nonzero_reward_rate: float = 0.50,
    min_variable_group_rate: float = 0.50,
) -> dict[str, Any]:
    """Summarize rollout quality and apply the reviewed GRPO readiness gates."""

    outputs = [output for group in groups for output in group["outputs"]]
    if not outputs or not groups:
        raise ValueError("at least one rollout group is required")

    rewards = [float(output["reward"]) for output in outputs]
    parsed_count = sum(bool(output["parsed"]) for output in outputs)
    nonzero_count = sum(reward > 0.0 for reward in rewards)
    variable_count = sum(
        len({round(float(output["reward"]), 8) for output in group["outputs"]}) > 1
        for group in groups
    )
    parse_rate = parsed_count / len(outputs)
    nonzero_rate = nonzero_count / len(outputs)
    variable_rate = variable_count / len(groups)
    criteria = {
        "parse_rate": parse_rate >= min_parse_rate,
        "nonzero_reward_rate": nonzero_rate >= min_nonzero_reward_rate,
        "variable_reward_group_rate": variable_rate >= min_variable_group_rate,
    }
    reward_counts = Counter(str(round(value, 2)) for value in rewards)

    return {
        "total_groups": len(groups),
        "total_rollouts": len(outputs),
        "parsed_rollouts": parsed_count,
        "parse_rate": parse_rate,
        "nonzero_reward_rollouts": nonzero_count,
        "nonzero_reward_rate": nonzero_rate,
        "variable_reward_groups": variable_count,
        "variable_reward_group_rate": variable_rate,
        "mean_reward": sum(rewards) / len(rewards),
        "reward_counts": dict(sorted(reward_counts.items(), key=lambda item: float(item[0]))),
        "criteria": criteria,
        "thresholds": {
            "min_parse_rate": min_parse_rate,
            "min_nonzero_reward_rate": min_nonzero_reward_rate,
            "min_variable_reward_group_rate": min_variable_group_rate,
        },
        "passed": all(criteria.values()),
    }


def _load_reward_module(path: Path):
    spec = importlib.util.spec_from_file_location("toolsafe_rollout_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest_with_file(digest: Any, path: Path) -> None:
    """Hash one named file with length delimiters to avoid concatenation ambiguity."""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint weight file not found: {path.name}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"checkpoint weight file is empty: {path.name}")
    name = path.name.encode("utf-8")
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update(size.to_bytes(8, "big"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


def checkpoint_weight_sha256(model_dir: Path) -> str:
    """Return a reproducible digest for single-file or sharded safetensors."""

    model_dir = Path(model_dir)
    single_weight = model_dir / "model.safetensors"
    if single_weight.is_file():
        if single_weight.stat().st_size <= 0:
            raise ValueError("checkpoint weight file is empty: model.safetensors")
        return _sha256(single_weight)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "checkpoint contains neither model.safetensors nor "
            "model.safetensors.index.json"
        )
    if index_path.stat().st_size <= 0:
        raise ValueError("checkpoint weight index is empty")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint weight index must contain a non-empty weight_map")

    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and Path(name).name == name for name in shard_names):
        raise ValueError("checkpoint shard names must be direct relative file names")

    digest = hashlib.sha256()
    _update_digest_with_file(digest, index_path)
    for shard_name in shard_names:
        _update_digest_with_file(digest, model_dir / shard_name)
    return digest.hexdigest()


def run(config_path: Path, output_dir: Path, prompt_count: int) -> dict[str, Any]:
    """Sample rollout groups and save an auditable acceptance report."""

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("rollout acceptance requires a CUDA GPU")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    rollout = config["rollout"]
    data = config["data"]
    model_path = Path(paths["actor_model"])
    train_path = Path(paths["train_file"])
    reward_path = Path(paths["reward_function"])
    rows = pq.read_table(train_path).to_pylist()
    selected = select_stratified_rows(rows, count=prompt_count)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    rendered_prompts = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        for row in selected
    ]

    # vLLM is inference-only here: no optimizer, gradients, reference policy,
    # or Actor parameter mutation is constructed by this process.
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=int(rollout["tensor_model_parallel_size"]),
        gpu_memory_utilization=float(rollout["gpu_memory_utilization"]),
        max_model_len=int(data["max_prompt_length"]) + int(data["max_response_length"]),
        max_num_batched_tokens=int(rollout["max_num_batched_tokens"]),
        max_num_seqs=int(rollout["max_num_seqs"]),
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=int(rollout["n"]),
        temperature=float(rollout["temperature"]),
        top_p=float(rollout["top_p"]),
        max_tokens=int(data["max_response_length"]),
        seed=int(data["seed"]),
    )
    generated = engine.generate(rendered_prompts, sampling, use_tqdm=True)
    reward_module = _load_reward_module(reward_path)

    groups: list[dict[str, Any]] = []
    for index, (row, request_output) in enumerate(zip(selected, generated, strict=True), 1):
        ground_truth = dict(row["reward_model"]["ground_truth"])
        outputs: list[dict[str, Any]] = []
        for candidate in request_output.outputs:
            text = candidate.text.strip()
            parsed_result = reward_module.ashellguardian_parser_v2(text)
            reward_value = float(
                reward_module.compute_score(
                    row["data_source"], text, ground_truth, row.get("extra_info")
                )
            )
            outputs.append(
                {
                    "text": text,
                    "parsed": parsed_result != "error",
                    "parsed_fields": None if parsed_result == "error" else parsed_result,
                    "reward": reward_value,
                    "finish_reason": candidate.finish_reason,
                    "generated_token_count": len(candidate.token_ids),
                }
            )
        rewards = [float(output["reward"]) for output in outputs]
        group = {
            "group_index": index,
            "source_identity": row["extra_info"]["source_identity"],
            "source": f"{row['extra_info']['dataset']}/{row['extra_info']['subset']}",
            "ground_truth": ground_truth,
            "prompt": row["prompt"],
            "prompt_token_count": row["extra_info"]["prompt_token_count"],
            "reward_std": pstdev(rewards),
            "unique_rewards": sorted(set(rewards)),
            "outputs": outputs,
        }
        groups.append(group)
        print(
            f"[{index}/{len(selected)}] source={group['source']} "
            f"parsed={sum(output['parsed'] for output in outputs)}/{len(outputs)} "
            f"rewards={group['unique_rewards']} std={group['reward_std']:.4f}",
            flush=True,
        )

    summary = summarize_acceptance(groups)
    summary.update(
        {
            "training_started": False,
            "optimizer_created": False,
            "backward_executed": False,
            "model_path": str(model_path),
            "model_weight_sha256": checkpoint_weight_sha256(model_path),
            "train_file": str(train_path),
            "train_file_sha256": _sha256(train_path),
            "reward_function": str(reward_path),
            "sampling": {
                "prompt_count": prompt_count,
                "n": int(rollout["n"]),
                "temperature": float(rollout["temperature"]),
                "top_p": float(rollout["top_p"]),
                "max_tokens": int(data["max_response_length"]),
                "seed": int(data["seed"]),
            },
            "selected_sources": Counter(group["source"] for group in groups),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "rollout_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for group in groups:
            stream.write(json.dumps(group, ensure_ascii=False) + "\n")
    summary["trace_path"] = str(trace_path)
    report_path = output_dir / "acceptance_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-count", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.config.resolve(), args.output_dir.resolve(), args.prompt_count)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

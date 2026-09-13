#!/usr/bin/env python3
"""Run exactly one in-memory constrained GRPO update on the real 3B Actor.

This is an E4 acceptance gate, not formal training.  It samples one variable
reward group, evaluates old/Reference/new probabilities with the same Guardian
FSM, executes backward and one optimizer step, writes a small JSON report, and
never saves model weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from constrained_grpo_core import (
    dense_guardian_reward,
    grpo_response_loss,
    group_relative_advantages,
)
from constrained_guardian_fsm import compile_guardian_grammar
from constrained_guardian_policy import (
    ConstrainedRollout,
    generate_constrained_rollout,
    recompute_constrained_log_probs,
)
from e3_constrained_logprob_canary import (
    strict_response_matches_rollout,
    tokenizer_fingerprint,
)


def build_e4_gate_report(
    *,
    rewards: Sequence[float],
    advantages: Sequence[float],
    strict_format_count: int,
    rollout_count: int,
    old_new_max_abs_error: float,
    old_new_parity_tolerance: float,
    policy_loss: float,
    kl_loss: float,
    total_loss: float,
    gradient_norm: float,
    parameter_delta: float,
    optimizer_created: bool,
    backward_executed: bool,
    optimizer_step_executed: bool,
) -> dict[str, Any]:
    """Build a fail-closed report from observed one-step training quantities."""

    numeric_values = [
        *[float(value) for value in rewards],
        *[float(value) for value in advantages],
        float(old_new_max_abs_error),
        float(policy_loss),
        float(kl_loss),
        float(total_loss),
        float(gradient_norm),
        float(parameter_delta),
    ]
    rollout_accounting = (
        rollout_count > 0
        and len(rewards) == rollout_count
        and len(advantages) == rollout_count
        and 0 <= strict_format_count <= rollout_count
    )
    finite_math = rollout_accounting and all(
        math.isfinite(value) for value in numeric_values
    )
    criteria = {
        "rollout_accounting": rollout_accounting,
        "all_responses_strict_format": (
            rollout_accounting and strict_format_count == rollout_count
        ),
        "variable_reward_group": rollout_accounting and len(set(rewards)) > 1,
        "finite_math": finite_math,
        "nonnegative_sampled_kl": finite_math and float(kl_loss) >= -1e-8,
        "authoritative_old_new_parity": (
            finite_math
            and float(old_new_max_abs_error) <= old_new_parity_tolerance
        ),
        "positive_gradient": finite_math and float(gradient_norm) > 0.0,
        "parameter_updated": finite_math and float(parameter_delta) > 0.0,
        "optimizer_created": bool(optimizer_created),
        "backward_executed": bool(backward_executed),
        "optimizer_step_executed": bool(optimizer_step_executed),
    }
    return {
        "stage": "E4_one_step_constrained_grpo",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "rollout_count": int(rollout_count),
        "strict_format_count": int(strict_format_count),
        "rewards": [float(value) for value in rewards],
        "advantages": [float(value) for value in advantages],
        "old_new_max_abs_error": float(old_new_max_abs_error),
        "old_new_parity_tolerance": float(old_new_parity_tolerance),
        "policy_loss": float(policy_loss),
        "kl_loss": float(kl_loss),
        "total_loss": float(total_loss),
        "gradient_norm": float(gradient_norm),
        "parameter_delta": float(parameter_delta),
        "optimizer_created": bool(optimizer_created),
        "backward_executed": bool(backward_executed),
        "optimizer_step_executed": bool(optimizer_step_executed),
        "formal_training_started": False,
        "checkpoint_written": False,
    }


def _load_candidate_rows(train_file: Path, limit: int) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(train_file)
    buckets: dict[float, list[dict[str, Any]]] = {0.5: [], 1.0: [], 0.0: []}
    for index, row in frame.iterrows():
        reward_model = dict(row["reward_model"] or {})
        ground_truth = dict(reward_model.get("ground_truth") or {})
        rating = float(ground_truth["Harmfulness_Rating"])
        if rating not in buckets:
            continue
        raw_messages = row["prompt"]
        if hasattr(raw_messages, "tolist"):
            raw_messages = raw_messages.tolist()
        buckets[rating].append(
            {
                "index": int(index),
                "messages": [dict(message) for message in raw_messages],
                "ground_truth": ground_truth,
                "extra_info": dict(row["extra_info"] or {}),
            }
        )

    order: list[dict[str, Any]] = []
    positions = {rating: 0 for rating in buckets}
    rating_cycle = (0.5, 1.0, 0.0, 0.5)
    while len(order) < limit:
        added = False
        for rating in rating_cycle:
            position = positions[rating]
            if position < len(buckets[rating]):
                order.append(buckets[rating][position])
                positions[rating] += 1
                added = True
                if len(order) == limit:
                    break
        if not added:
            break
    if not order:
        raise ValueError("GRPO parquet contains no supported ground-truth rows")
    return order


def _load_model(path: Path, *, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    return model.to(torch.device("cuda"))


def _largest_gradient_position(parameter: torch.Tensor) -> tuple[int, float]:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("chosen update parameter has no gradient")
    flat = gradient.detach().view(-1)
    best_index = -1
    best_value = -1.0
    chunk_size = 1_000_000
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size]
        value, position = chunk.abs().max(dim=0)
        candidate = float(value.item())
        if candidate > best_value:
            best_value = candidate
            best_index = start + int(position.item())
    if best_index < 0 or not math.isfinite(best_value) or best_value <= 0.0:
        raise RuntimeError("chosen update parameter has no finite nonzero gradient")
    return best_index, best_value


def parse_args() -> argparse.Namespace:
    root = Path("/root/Agent-Security")
    prepared = root / (
        "practice/toolsafe_reproduction/grpo/prepared_models/"
        "qwen2.5-3b-sft-full-format-reinforced"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-model", type=Path, default=prepared / "policy_init")
    parser.add_argument("--reference-model", type=Path, default=prepared / "reference")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=root
        / "practice/toolsafe_reproduction/data/teacher_adjudicated_v1/grpo_train.parquet",
    )
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--max-candidate-prompts", type=int, default=4)
    parser.add_argument("--min-rationale-content-tokens", type=int, default=8)
    parser.add_argument("--max-rationale-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-coefficient", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--old-new-parity-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=root
        / "practice/toolsafe_reproduction/results/constrained_grpo_e4/"
        "e4_one_step_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E4 real-model smoke requires CUDA")
    if args.rollouts_per_prompt < 2 or args.max_candidate_prompts < 1:
        raise ValueError("E4 requires at least two rollouts and one candidate prompt")
    for path in (args.actor_model, args.reference_model, args.train_file):
        if not path.exists():
            raise FileNotFoundError(path)

    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        args.actor_model, local_files_only=True
    )
    reference_tokenizer = AutoTokenizer.from_pretrained(
        args.reference_model, local_files_only=True
    )
    grammar = compile_guardian_grammar(
        tokenizer,
        min_rationale_content_tokens=args.min_rationale_content_tokens,
        max_rationale_tokens=args.max_rationale_tokens,
    )
    reference_grammar = compile_guardian_grammar(
        reference_tokenizer,
        min_rationale_content_tokens=args.min_rationale_content_tokens,
        max_rationale_tokens=args.max_rationale_tokens,
    )
    fingerprint = tokenizer_fingerprint(tokenizer, grammar)
    if fingerprint != tokenizer_fingerprint(reference_tokenizer, reference_grammar):
        raise RuntimeError("Actor and Reference tokenizer/grammar fingerprints differ")

    candidates = _load_candidate_rows(args.train_file, args.max_candidate_prompts)
    actor = _load_model(args.actor_model, dtype=torch.float32)
    actor.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    selected: dict[str, Any] | None = None
    selected_rollouts: list[ConstrainedRollout] = []
    selected_old_log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    attempt_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        prompt_token_ids = tokenizer.apply_chat_template(
            candidate["messages"], tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.tolist()
        rollouts: list[ConstrainedRollout] = []
        old_log_probs: list[torch.Tensor] = []
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            for _ in range(args.rollouts_per_prompt):
                rollout = generate_constrained_rollout(
                    actor,
                    grammar,
                    prompt_token_ids=prompt_token_ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    generator=generator,
                )
                rollouts.append(rollout)
                old_log_probs.append(
                    recompute_constrained_log_probs(
                        actor,
                        grammar,
                        prompt_token_ids=prompt_token_ids,
                        decisions=rollout.decisions,
                    ).detach().cpu()
                )
        candidate_rewards = [
            dense_guardian_reward(rollout.judgments, candidate["ground_truth"])
            for rollout in rollouts
        ]
        attempt_summaries.append(
            {
                "sample_index": candidate["index"],
                "source_identity": candidate["extra_info"].get("source_identity"),
                "ground_truth": candidate["ground_truth"],
                "rewards": candidate_rewards,
                "judgments": [rollout.judgments for rollout in rollouts],
            }
        )
        print(
            f"candidate={candidate['index']} "
            f"source={candidate['extra_info'].get('source_identity')} "
            f"rewards={candidate_rewards}",
            flush=True,
        )
        if len(set(candidate_rewards)) > 1:
            selected = {**candidate, "prompt_token_ids": list(prompt_token_ids)}
            selected_rollouts = rollouts
            selected_old_log_probs = old_log_probs
            rewards = candidate_rewards
            break

    if selected is None:
        raise RuntimeError(
            "no variable-reward group found within the deterministic candidate limit"
        )

    advantages_tensor, advantage_stats = group_relative_advantages(
        torch.tensor(rewards, dtype=torch.float32),
        group_ids=("selected-prompt",) * len(rewards),
    )
    advantages = advantages_tensor.tolist()
    if advantage_stats.variable_group_count != 1:
        raise RuntimeError("selected group did not produce one variable advantage group")

    reference = _load_model(args.reference_model, dtype=torch.bfloat16)
    reference.eval()
    reference_log_probs: list[torch.Tensor] = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for rollout in selected_rollouts:
            reference_log_probs.append(
                recompute_constrained_log_probs(
                    reference,
                    reference_grammar,
                    prompt_token_ids=selected["prompt_token_ids"],
                    decisions=rollout.decisions,
                ).detach().cpu()
            )
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    actor.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    actor.config.use_cache = False
    actor.train()
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=args.learning_rate,
        foreach=False,
        fused=False,
    )
    optimizer_created = True
    optimizer.zero_grad(set_to_none=True)

    policy_loss = 0.0
    kl_loss = 0.0
    total_loss = 0.0
    parity_errors: list[float] = []
    for rollout, old_cpu, reference_cpu, advantage in zip(
        selected_rollouts,
        selected_old_log_probs,
        reference_log_probs,
        advantages,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            new_log_probs = recompute_constrained_log_probs(
                actor,
                grammar,
                prompt_token_ids=selected["prompt_token_ids"],
                decisions=rollout.decisions,
                force_eval=False,
            )
            old_log_probs = old_cpu.to(new_log_probs.device)
            reference_values = reference_cpu.to(new_log_probs.device)
            if not (
                new_log_probs.shape == old_log_probs.shape == reference_values.shape
            ):
                raise RuntimeError("Actor/old/Reference decision vectors differ")
            parity_errors.append(
                float((new_log_probs.detach() - old_log_probs).abs().max().item())
            )
            loss = grpo_response_loss(
                new_log_probs,
                old_log_probs=old_log_probs,
                reference_log_probs=reference_values,
                advantage=advantage,
                clip_ratio=args.clip_ratio,
                kl_coefficient=args.kl_coefficient,
            )
            scaled_loss = loss.total_loss / len(selected_rollouts)
        scaled_loss.backward()
        policy_loss += float(loss.policy_loss.detach().item()) / len(selected_rollouts)
        kl_loss += float(loss.kl_loss.detach().item()) / len(selected_rollouts)
        total_loss += float(loss.total_loss.detach().item()) / len(selected_rollouts)
        del new_log_probs, old_log_probs, reference_values, loss, scaled_loss

    backward_executed = True
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), args.max_grad_norm
    )
    gradient_norm = float(gradient_norm_tensor.detach().item())
    output_weight = actor.get_output_embeddings().weight
    parameter_index, selected_gradient = _largest_gradient_position(output_weight)
    flat_parameter = output_weight.detach().view(-1)
    parameter_before = float(flat_parameter[parameter_index].item())
    gc.collect()
    torch.cuda.empty_cache()
    optimizer.step()
    optimizer_step_executed = True
    parameter_after = float(flat_parameter[parameter_index].item())
    parameter_delta = abs(parameter_after - parameter_before)

    report = build_e4_gate_report(
        rewards=rewards,
        advantages=advantages,
        strict_format_count=sum(
            strict_response_matches_rollout(rollout)
            for rollout in selected_rollouts
        ),
        rollout_count=len(selected_rollouts),
        old_new_max_abs_error=max(parity_errors),
        old_new_parity_tolerance=args.old_new_parity_tolerance,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        total_loss=total_loss,
        gradient_norm=gradient_norm,
        parameter_delta=parameter_delta,
        optimizer_created=optimizer_created,
        backward_executed=backward_executed,
        optimizer_step_executed=optimizer_step_executed,
    )
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "actor_model": str(args.actor_model.resolve()),
            "reference_model": str(args.reference_model.resolve()),
            "train_file": str(args.train_file.resolve()),
            "tokenizer_fingerprint": fingerprint,
            "protocol_version": "toolsafe-guardian-token-fsm-v2",
            "selected_sample_index": selected["index"],
            "source_identity": selected["extra_info"].get("source_identity"),
            "ground_truth": selected["ground_truth"],
            "sampling": {
                "n": args.rollouts_per_prompt,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_rationale_tokens": args.max_rationale_tokens,
                "seed": args.seed,
            },
            "optimization": {
                "learning_rate": args.learning_rate,
                "clip_ratio": args.clip_ratio,
                "kl_coefficient": args.kl_coefficient,
                "max_grad_norm": args.max_grad_norm,
            },
            "candidate_attempts": attempt_summaries,
            "rollouts": [
                {
                    "response": rollout.output_text,
                    "judgments": rollout.judgments,
                    "reward": reward,
                    "advantage": advantage,
                    "decision_count": len(rollout.decisions),
                    "forced_rationale_close": rollout.forced_rationale_close,
                }
                for rollout, reward, advantage in zip(
                    selected_rollouts, rewards, advantages
                )
            ],
            "updated_parameter": {
                "name": "output_embeddings.weight",
                "flat_index": parameter_index,
                "selected_gradient_abs": selected_gradient,
                "before": parameter_before,
                "after": parameter_after,
            },
            "peak_cuda_memory_allocated_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
            "peak_cuda_memory_reserved_gb": (
                torch.cuda.max_memory_reserved() / 1024**3
            ),
        }
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

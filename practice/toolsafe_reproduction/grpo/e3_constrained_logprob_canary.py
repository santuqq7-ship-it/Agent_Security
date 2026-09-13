#!/usr/bin/env python3
"""E3 hard gate for same-Actor constrained log-probability parity.

This script loads one real adjudicated GRPO prompt, generates one response with
the Hugging Face Actor and Guardian FSM, recomputes the Actor probabilities in
one differentiable-style forward pass, then evaluates the same semantic trace
under the frozen Reference.  It never creates an optimizer or runs backward.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from constrained_guardian_fsm import CompiledGuardianGrammar, compile_guardian_grammar
from constrained_guardian_policy import (
    ConstrainedRollout,
    generate_constrained_rollout,
    recompute_constrained_log_probs,
)


STRICT_RESPONSE = re.compile(
    r"\A<Think> (?P<think>[^<>\r\n]+) <\\Think>\n"
    r"<Malicious_User_Request> (?P<malicious>yes|no) "
    r"<\\Malicious_User_Request>\n"
    r"<Being_Attacked> (?P<attacked>yes|no) <\\Being_Attacked>\n"
    r"<Harmfulness_Rating> (?P<harmfulness>0\.0|0\.5|1\.0) "
    r"<\\Harmfulness_Rating>\Z"
)


def _float_list(values: Iterable[float] | torch.Tensor) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().tolist()
    return [float(value) for value in values]


def build_e3_gate_report(
    *,
    tokenizer_match: bool,
    strict_format: bool,
    actor_rollout_log_probs: Sequence[float] | torch.Tensor,
    actor_recomputed_log_probs: Sequence[float] | torch.Tensor,
    actor_repeat_log_probs: Sequence[float] | torch.Tensor,
    reference_log_probs: Sequence[float] | torch.Tensor,
    cache_parity_tolerance: float,
    authoritative_parity_tolerance: float,
    decision_count: int,
    hidden_decision_count: int,
    forced_rationale_close: bool,
    decision_states: Sequence[str],
) -> dict[str, Any]:
    """Build the fail-closed E3 acceptance result from numeric observations."""

    rollout = _float_list(actor_rollout_log_probs)
    recomputed = _float_list(actor_recomputed_log_probs)
    repeated = _float_list(actor_repeat_log_probs)
    reference = _float_list(reference_log_probs)
    lengths_match = (
        decision_count > 0
        and len(rollout)
        == len(recomputed)
        == len(repeated)
        == len(reference)
        == decision_count
    )
    all_finite = lengths_match and all(
        math.isfinite(value)
        for value in rollout + recomputed + repeated + reference
    )
    parity_errors = (
        [abs(old - new) for old, new in zip(rollout, recomputed)]
        if lengths_match
        else []
    )
    parity_error = max(parity_errors) if parity_errors else math.inf
    authoritative_errors = (
        [abs(old - repeat) for old, repeat in zip(recomputed, repeated)]
        if lengths_match
        else []
    )
    authoritative_error = (
        max(authoritative_errors) if authoritative_errors else math.inf
    )
    max_error_index = (
        max(range(len(parity_errors)), key=parity_errors.__getitem__)
        if parity_errors
        else None
    )
    states_match = len(decision_states) == decision_count
    max_error_state = (
        str(decision_states[max_error_index])
        if states_match and max_error_index is not None
        else None
    )
    trace_accounting = (
        decision_count > 0
        and 0 <= hidden_decision_count <= decision_count
        and states_match
    )
    criteria = {
        "tokenizer_match": bool(tokenizer_match),
        "strict_format": bool(strict_format),
        "trace_accounting": trace_accounting,
        "finite_actor_and_reference_log_probs": all_finite,
        "actor_cache_recompute_consistency": bool(
            all_finite and parity_error <= cache_parity_tolerance
        ),
        "authoritative_old_logprob_repeat_parity": bool(
            all_finite and authoritative_error <= authoritative_parity_tolerance
        ),
    }
    sampled_kl_mean = (
        sum(old - ref for old, ref in zip(recomputed, reference)) / decision_count
        if all_finite
        else None
    )
    return {
        "stage": "E3_constrained_logprob",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "cache_parity_tolerance": float(cache_parity_tolerance),
        "authoritative_parity_tolerance": float(authoritative_parity_tolerance),
        "actor_parity_max_abs_error": parity_error,
        "actor_parity_max_error_index": max_error_index,
        "actor_parity_max_error_state": max_error_state,
        "authoritative_old_logprob_repeat_max_abs_error": authoritative_error,
        "decision_count": int(decision_count),
        "hidden_decision_count": int(hidden_decision_count),
        "forced_rationale_close": bool(forced_rationale_close),
        "sampled_actor_reference_log_ratio_mean": sampled_kl_mean,
        "training_started": False,
        "optimizer_created": False,
        "backward_executed": False,
    }


def tokenizer_fingerprint(
    tokenizer: Any,
    grammar: CompiledGuardianGrammar,
) -> str:
    """Hash tokenizer behavior plus every grammar-relevant token collection."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        tokenizer_payload: Any = backend.to_str()
    else:
        tokenizer_payload = sorted(
            (str(token), int(token_id))
            for token, token_id in tokenizer.get_vocab().items()
        )
    payload = {
        "tokenizer": tokenizer_payload,
        "special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
        "vocab_size": grammar.tokenizer_vocab_size,
        "start": grammar.start_tokens,
        "after_rationale": grammar.after_rationale_tokens,
        "after_malicious": grammar.after_malicious_tokens,
        "after_attacked": grammar.after_attacked_tokens,
        "after_harmfulness": grammar.after_harmfulness_tokens,
        "rationale_end": grammar.rationale_end_trie.choices,
        "enums": {
            name: trie.choices for name, trie in grammar.enum_tries.items()
        },
        "safe_rationale_ids": sorted(grammar.safe_rationale_token_ids),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_response_matches_rollout(rollout: ConstrainedRollout) -> bool:
    match = STRICT_RESPONSE.fullmatch(rollout.output_text)
    if match is None or not match.group("think").strip():
        return False
    return rollout.judgments == {
        "Malicious_User_Request": match.group("malicious") == "yes",
        "Being_Attacked": match.group("attacked") == "yes",
        "Harmfulness_Rating": float(match.group("harmfulness")),
    }


def _load_prompt(train_file: Path, sample_index: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(train_file)
    if not 0 <= sample_index < len(frame):
        raise IndexError(
            f"sample index {sample_index} outside parquet rows 0..{len(frame) - 1}"
        )
    row = frame.iloc[sample_index]
    raw_messages = row["prompt"]
    if hasattr(raw_messages, "tolist"):
        raw_messages = raw_messages.tolist()
    messages = [dict(message) for message in raw_messages]
    if not messages or any(
        set(message) < {"role", "content"} for message in messages
    ):
        raise ValueError("GRPO prompt must be a non-empty role/content message list")
    extra_info = dict(row.get("extra_info") or {})
    return messages, extra_info


def _load_model(model_path: Path) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.to(torch.device("cuda"))
    model.eval()
    return model


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
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--min-rationale-content-tokens", type=int, default=8)
    parser.add_argument("--max-rationale-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--cache-parity-tolerance", type=float, default=0.20)
    parser.add_argument(
        "--authoritative-parity-tolerance", type=float, default=1e-5
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=root
        / "practice/toolsafe_reproduction/results/constrained_grpo_e3/e3_canary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E3 real-model canary requires CUDA")
    for path in (args.actor_model, args.reference_model, args.train_file):
        if not path.exists():
            raise FileNotFoundError(path)

    from transformers import AutoTokenizer

    actor_tokenizer = AutoTokenizer.from_pretrained(
        args.actor_model, local_files_only=True
    )
    reference_tokenizer = AutoTokenizer.from_pretrained(
        args.reference_model, local_files_only=True
    )
    actor_grammar = compile_guardian_grammar(
        actor_tokenizer,
        min_rationale_content_tokens=args.min_rationale_content_tokens,
        max_rationale_tokens=args.max_rationale_tokens,
    )
    reference_grammar = compile_guardian_grammar(
        reference_tokenizer,
        min_rationale_content_tokens=args.min_rationale_content_tokens,
        max_rationale_tokens=args.max_rationale_tokens,
    )
    actor_fingerprint = tokenizer_fingerprint(actor_tokenizer, actor_grammar)
    reference_fingerprint = tokenizer_fingerprint(
        reference_tokenizer, reference_grammar
    )
    tokenizer_match = actor_fingerprint == reference_fingerprint
    if not tokenizer_match:
        raise RuntimeError("Actor and Reference tokenizer/grammar fingerprints differ")

    messages, extra_info = _load_prompt(args.train_file, args.sample_index)
    prompt_token_ids = actor_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(prompt_token_ids, torch.Tensor):
        prompt_token_ids = prompt_token_ids.tolist()

    generator = None
    if args.do_sample:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)

    actor = _load_model(args.actor_model)
    rollout = generate_constrained_rollout(
        actor,
        actor_grammar,
        prompt_token_ids=prompt_token_ids,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        generator=generator,
    )
    with torch.inference_mode():
        actor_recomputed = recompute_constrained_log_probs(
            actor,
            actor_grammar,
            prompt_token_ids=prompt_token_ids,
            decisions=rollout.decisions,
        )
        actor_repeated = recompute_constrained_log_probs(
            actor,
            actor_grammar,
            prompt_token_ids=prompt_token_ids,
            decisions=rollout.decisions,
        )
    actor_rollout = [
        decision.rollout_log_prob for decision in rollout.decisions
    ]
    del actor
    gc.collect()
    torch.cuda.empty_cache()

    reference = _load_model(args.reference_model)
    with torch.inference_mode():
        reference_log_probs = recompute_constrained_log_probs(
            reference,
            reference_grammar,
            prompt_token_ids=prompt_token_ids,
            decisions=rollout.decisions,
        )
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    report = build_e3_gate_report(
        tokenizer_match=tokenizer_match,
        strict_format=strict_response_matches_rollout(rollout),
        actor_rollout_log_probs=actor_rollout,
        actor_recomputed_log_probs=actor_recomputed,
        actor_repeat_log_probs=actor_repeated,
        reference_log_probs=reference_log_probs,
        cache_parity_tolerance=args.cache_parity_tolerance,
        authoritative_parity_tolerance=args.authoritative_parity_tolerance,
        decision_count=len(rollout.decisions),
        hidden_decision_count=(
            len(rollout.decisions) - sum(rollout.semantic_response_mask)
        ),
        forced_rationale_close=rollout.forced_rationale_close,
        decision_states=[
            decision.state_before for decision in rollout.decisions
        ],
    )
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "actor_model": str(args.actor_model.resolve()),
            "reference_model": str(args.reference_model.resolve()),
            "train_file": str(args.train_file.resolve()),
            "sample_index": args.sample_index,
            "source_identity": extra_info.get("source_identity"),
            "prompt_token_count": len(prompt_token_ids),
            "response_token_count": len(rollout.response_token_ids),
            "visible_semantic_token_count": sum(rollout.semantic_response_mask),
            "tokenizer_fingerprint": actor_fingerprint,
            "sampling": {
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            "judgments": rollout.judgments,
            "response": rollout.output_text,
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

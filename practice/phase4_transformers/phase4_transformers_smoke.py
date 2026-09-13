"""Run a local Transformers inference smoke test.

This script deliberately does not execute any tool. It only demonstrates how
an Agent model produces a candidate tool call and how per-token entropy can be
computed from the generation scores returned by Transformers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return one Shannon entropy value for each row of a logits tensor.

    The input shape is ``[batch, vocab_size]``. Logits are relative scores,
    so they are converted to log-probabilities before applying ``-sum(p log p)``.
    """

    # Top-k/top-p sampling can set removed candidates to ``-inf``. Their
    # probability is mathematically zero, but ``0 * -inf`` becomes NaN in
    # floating-point arithmetic, so mask those terms before summing.
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    finite_mask = torch.isfinite(log_probs)
    safe_log_probs = torch.where(finite_mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(finite_mask, log_probs.exp(), torch.zeros_like(log_probs))
    return -(probs * safe_log_probs).sum(dim=-1)


def resolve_device(requested: str = "auto", mps_available: bool | None = None) -> str:
    """Resolve a user-facing device name without assuming CUDA exists.

    ``auto`` prefers Apple MPS when it is available and otherwise falls back to
    CPU. The optional ``mps_available`` argument makes this decision easy to
    test without depending on the host machine.
    """

    if mps_available is None:
        mps_available = torch.backends.mps.is_available()

    if requested == "auto":
        return "mps" if mps_available else "cpu"
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but this PyTorch build cannot access MPS.")
    if requested not in {"cpu", "mps"}:
        raise ValueError(f"Unsupported device '{requested}'. Use 'auto', 'mps', or 'cpu'.")
    return requested


def parse_react_tool_call(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Extract a simple ReAct tool name and JSON argument object.

    The parser is intentionally small and only supports the smoke-test format.
    ToolSafe's production parser remains in ``src/utils/tool_parser.py``.
    """

    tool_match = re.search(r"Action:\s*([A-Za-z_][\w.-]*)", text)
    args_match = re.search(r"Action Input:\s*(\{.*?\})", text, flags=re.DOTALL)
    if tool_match is None or args_match is None:
        return None, None

    try:
        arguments = json.loads(args_match.group(1))
    except json.JSONDecodeError:
        return tool_match.group(1), None
    return tool_match.group(1), arguments


def build_messages() -> list[dict[str, str]]:
    """Build a harmless prompt plus one explicit ReAct demonstration.

    The assistant demonstration is intentionally part of the conversation
    history: it teaches the model the exact textual protocol that ToolSafe's
    parser consumes, instead of relying only on prose instructions.
    """

    system_prompt = (
        "You are a ReAct agent with exactly one read-only tool: "
        "get_order_status(order_id: string). For every tool request, output "
        "only these three lines and stop immediately after the JSON: "
        "Thought: <brief reason>\n"
        "Action: get_order_status\n"
        'Action Input: {"order_id": "A-100"}. '
        "Do not answer conversationally, invent tools, or add a final explanation."
    )
    example_user_prompt = "Example request: check the status of order A-100."
    example_assistant_response = (
        "Thought: I will check the order.\n"
        "Action: get_order_status\n"
        'Action Input: {"order_id": "A-100"}'
    )
    user_prompt = "Now check the status of order A-100 and emit the tool call."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example_user_prompt},
        {"role": "assistant", "content": example_assistant_response},
        {"role": "user", "content": user_prompt},
    ]


def _top_candidates(logits: torch.Tensor, tokenizer, limit: int = 8) -> list[dict[str, Any]]:
    """Return human-readable candidates from one generation step."""

    # Sampling filters represent removed tokens as -inf. If we call topk on
    # the resulting probability vector directly, all zero-probability tokens
    # tie and arbitrary vocabulary entries would appear in the report.
    finite_mask = torch.isfinite(logits)
    valid_count = int(finite_mask[0].sum().item())
    if valid_count == 0:
        return []

    safe_logits = logits.masked_fill(~finite_mask, float("-inf"))
    probabilities = torch.softmax(safe_logits, dim=-1)
    top_values, top_ids = torch.topk(
        probabilities,
        k=min(limit, valid_count),
        dim=-1,
    )
    candidates = []
    for probability, token_id in zip(top_values[0].tolist(), top_ids[0].tolist()):
        candidates.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([token_id]),
                "probability": float(probability),
            }
        )
    return candidates


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Load a local model, generate once, and return an inspectable report."""

    model_path = Path(config["model_path"]).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    requested_device = config.get("device", "auto")
    device = resolve_device(requested_device)
    messages = build_messages()

    # Keep model loading explicit: tokenizer maps text to IDs, while the model
    # maps those IDs to next-token scores.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16 if device == "mps" else torch.float32,
    )
    model.to(device)
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=int(config.get("max_new_tokens", 128)),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated_ids = outputs.sequences[0, prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ``outputs.scores`` contains one score vector per generated step. With
    # sampling enabled, these scores already include the model's configured
    # temperature/top-k/top-p filtering before a token is sampled.
    step_entropies = [compute_token_entropy(step_scores).item() for step_scores in outputs.scores]
    first_step = {}
    if outputs.scores:
        first_scores = outputs.scores[0].detach().float().cpu()
        first_step = {
            "entropy": float(compute_token_entropy(first_scores).item()),
            "top_candidates": _top_candidates(first_scores, tokenizer),
        }

    tool_name, tool_arguments = parse_react_tool_call(response)
    report = {
        "model_path": str(model_path),
        "requested_device": requested_device,
        "resolved_device": device,
        "mps_available": bool(torch.backends.mps.is_available()),
        "prompt_token_count": int(prompt_length),
        "generated_token_count": int(generated_ids.shape[-1]),
        "response": response,
        "parsed_tool_name": tool_name,
        "parsed_tool_arguments": tool_arguments,
        "step_entropy_count": len(step_entropies),
        "mean_token_entropy": float(sum(step_entropies) / len(step_entropies)) if step_entropies else 0.0,
        "first_generation_step": first_step,
        "note": "No tool was executed; this is an inference-only smoke test.",
    }
    return report


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the small YAML configuration used by this script."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    """Parse CLI arguments, run the experiment, and print JSON for inspection."""

    parser = argparse.ArgumentParser(description="ToolSafe local Transformers smoke test")
    parser.add_argument(
        "--config",
        # Keep the standalone practice runnable from any working directory.
        default=str(Path(__file__).with_name("phase4_transformers_smoke.yaml")),
        help="Path to the local smoke-test YAML configuration.",
    )
    args = parser.parse_args()
    report = run_smoke(load_config(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

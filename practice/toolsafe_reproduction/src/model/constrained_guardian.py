"""Official TS-Guard adapter backed by the project Token FSM.

The model chooses the rationale and three semantic judgments. Protocol labels,
spaces, newlines, and closing tags are emitted deterministically by the FSM.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.agent_prompts import GUARD_TEMPLATES
from grpo.constrained_guardian_fsm import compile_guardian_grammar
from grpo.constrained_guardian_policy import generate_constrained_rollout
from model.model import _dtype_for_device, _resolve_transformers_device


REQUIRED_JUDGMENTS = frozenset(
    {"Malicious_User_Request", "Being_Attacked", "Harmfulness_Rating"}
)


def aggregate_risk_rating(judgments: dict[str, Any]) -> float:
    """Reproduce the original TS-Guard parser's three-field aggregation."""

    if set(judgments) != REQUIRED_JUDGMENTS:
        raise ValueError(
            "Guardian judgments must contain exactly "
            f"{sorted(REQUIRED_JUDGMENTS)}"
        )
    malicious = judgments["Malicious_User_Request"]
    attacked = judgments["Being_Attacked"]
    harmfulness = judgments["Harmfulness_Rating"]
    if type(malicious) is not bool or type(attacked) is not bool:
        raise ValueError("Guardian boolean judgments must be bool values")
    if type(harmfulness) not in (int, float):
        raise ValueError("Harmfulness_Rating must be numeric")
    harmfulness = float(harmfulness)
    if harmfulness not in {0.0, 0.5, 1.0}:
        raise ValueError("Harmfulness_Rating must be 0.0, 0.5, or 1.0")

    score = 0.4 * int(attacked) + 0.4 * int(malicious)
    score += 0.1 if harmfulness == 0.5 else 0.0
    score += 0.2 if harmfulness == 1.0 else 0.0
    if score > 0.5:
        return 1.0
    if score == 0.5:
        return 0.5
    return 0.0


class ConstrainedGuardian:
    """Drop-in ``tool_safety_guardian`` using deterministic protocol decoding."""

    def __init__(
        self,
        *,
        model_name: str,
        model_path: str,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
        min_rationale_content_tokens: int = 8,
        max_rationale_tokens: int = 192,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.trace_callback = trace_callback

        if tokenizer is None or model is None:
            device = _resolve_transformers_device()
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    dtype=_dtype_for_device(device),
                )
                model.to(device)
        self.tokenizer = tokenizer
        self.llm = model.eval()
        self.grammar = compile_guardian_grammar(
            self.tokenizer,
            min_rationale_content_tokens=min_rationale_content_tokens,
            max_rationale_tokens=max_rationale_tokens,
        )

    def _emit_trace(self, event: dict[str, Any]) -> None:
        if self.trace_callback is not None:
            self.trace_callback({"source": "guardian", **event})

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "tool_safety_guardian":
            raise ValueError(
                "ConstrainedGuardian only supports tool_safety_guardian"
            )
        return self.tool_safety_guardian(**arguments)

    def tool_safety_guardian(
        self,
        *,
        user_request: str,
        interaction_history: Any,
        current_action: Any,
        current_action_description: str,
    ) -> dict[str, Any]:
        guard_input = GUARD_TEMPLATES[self.model_name].format(
            env_info=current_action_description,
            user_request=user_request,
            agent_action={
                "interaction_history": interaction_history,
                "current_action": current_action,
            },
        )
        messages = [{"role": "user", "content": guard_input}]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = self.tokenizer.encode(
            rendered_prompt,
            add_special_tokens=False,
        )
        self._emit_trace(
            {
                "kind": "guardian_request",
                "messages": deepcopy(messages),
                "rendered_prompt": rendered_prompt,
                "constrained": True,
            }
        )
        rollout = generate_constrained_rollout(
            self.llm,
            self.grammar,
            prompt_token_ids=prompt_token_ids,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        judgments = dict(rollout.judgments)
        risk_rating = aggregate_risk_rating(judgments)
        results = {
            "Malicious_User_Request": (
                "yes" if judgments["Malicious_User_Request"] else "no"
            ),
            "Being_Attacked": "yes" if judgments["Being_Attacked"] else "no",
            "Harmfulness_Rating": float(judgments["Harmfulness_Rating"]),
        }
        self._emit_trace(
            {
                "kind": "guardian_response",
                "text": rollout.output_text,
                "constrained": True,
                "forced_rationale_close": rollout.forced_rationale_close,
            }
        )
        self._emit_trace(
            {
                "kind": "guardian_decision",
                "risk_rating": risk_rating,
                "judgments": deepcopy(results),
            }
        )
        return {
            "risk rating": risk_rating,
            "results": results,
            "reason": rollout.output_text,
            "forced rationale close": rollout.forced_rationale_close,
        }

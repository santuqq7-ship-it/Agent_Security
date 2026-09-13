"""Shared constrained-policy semantics for ToolSafe Guardian rollout and GRPO.

The Guardian FSM owns the grammar.  This module records only decisions made by
the policy, replays those decisions against a fresh FSM, and normalizes logits
over exactly the tokens allowed at each decision.  Deterministic protocol text
is kept in the model context but never appears in the policy log-prob vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

try:
    from .constrained_guardian_fsm import (
        CompiledGuardianGrammar,
        ConstraintViolation,
        GuardianTokenFSM,
    )
except ImportError:  # Direct script/test imports place grpo/ itself on sys.path.
    from constrained_guardian_fsm import (
        CompiledGuardianGrammar,
        ConstraintViolation,
        GuardianTokenFSM,
    )


class InvalidConstrainedTrace(ValueError):
    """A recorded decision sequence cannot be reproduced by the grammar."""


class InvalidConstraintProbability(ValueError):
    """A constrained probability request is empty, invalid, or non-finite."""


@dataclass(frozen=True)
class ConstrainedDecision:
    """One sampled policy action, including hidden rationale termination."""

    state_before: str
    token_id: int
    rollout_log_prob: float
    sampling_log_prob: float
    hidden: bool
    serialized: bool


@dataclass(frozen=True)
class ConstrainedReplay:
    """Visible sequence and semantic positions reconstructed from decisions."""

    response_token_ids: tuple[int, ...]
    semantic_response_mask: tuple[int, ...]
    decision_context_lengths: tuple[int, ...]
    hidden_decision_count: int
    forced_rationale_close: bool
    judgments: dict[str, Any]


@dataclass(frozen=True)
class ConstrainedRollout:
    """One complete grammar-valid response sampled from an Actor."""

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    semantic_response_mask: tuple[int, ...]
    decisions: tuple[ConstrainedDecision, ...]
    output_text: str
    judgments: dict[str, Any]
    forced_rationale_close: bool


def constrained_token_log_prob(
    logits: torch.Tensor,
    *,
    allowed_token_ids: Iterable[int],
    selected_token_id: int,
) -> torch.Tensor:
    """Return log p(selected) after excluding every disallowed vocabulary item."""

    if logits.ndim != 1:
        raise InvalidConstraintProbability("logits must be a one-dimensional vector")
    allowed = tuple(int(token_id) for token_id in allowed_token_ids)
    if not allowed:
        raise InvalidConstraintProbability("allowed token set must not be empty")
    if len(set(allowed)) != len(allowed):
        raise InvalidConstraintProbability("allowed token set contains duplicates")
    selected_token_id = int(selected_token_id)
    if selected_token_id not in allowed:
        raise InvalidConstraintProbability(
            f"selected token {selected_token_id} is outside the allowed set"
        )
    if min(allowed) < 0 or max(allowed) >= logits.shape[0]:
        raise InvalidConstraintProbability("allowed token id is outside the vocabulary")

    allowed_index = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    allowed_logits = logits.index_select(0, allowed_index).float()
    if not bool(torch.isfinite(allowed_logits).all().item()):
        raise InvalidConstraintProbability("allowed logits contain a non-finite value")
    selected_position = allowed.index(selected_token_id)
    return allowed_logits[selected_position] - torch.logsumexp(allowed_logits, dim=0)


def _force_transition_if_needed(
    fsm: GuardianTokenFSM,
    semantic_mask: list[int],
) -> None:
    allowed = fsm.allowed_token_ids()
    if allowed or fsm.done:
        return
    if fsm.state != "rationale":
        raise InvalidConstrainedTrace(
            f"state {fsm.state!r} has no allowed token and cannot be forced"
        )
    try:
        event = fsm.force_rationale_close()
    except ConstraintViolation as exc:
        raise InvalidConstrainedTrace(str(exc)) from exc
    semantic_mask.extend(0 for _ in event.deterministic_token_ids)


def replay_constrained_trace(
    grammar: CompiledGuardianGrammar,
    *,
    prompt_length: int,
    decisions: Sequence[ConstrainedDecision],
) -> ConstrainedReplay:
    """Replay decisions and reconstruct visible context plus policy positions."""

    if prompt_length < 0:
        raise InvalidConstrainedTrace("prompt length must not be negative")
    fsm = GuardianTokenFSM(grammar)
    semantic_mask: list[int] = [0] * len(grammar.start_tokens)
    context_lengths: list[int] = []
    hidden_count = 0

    for index, decision in enumerate(decisions):
        _force_transition_if_needed(fsm, semantic_mask)
        if fsm.done:
            raise InvalidConstrainedTrace(
                f"decision {index} occurs after the grammar reached done"
            )
        if decision.state_before != fsm.state:
            raise InvalidConstrainedTrace(
                f"decision {index} state mismatch: recorded "
                f"{decision.state_before!r}, replayed {fsm.state!r}"
            )
        context_lengths.append(prompt_length + len(semantic_mask))
        try:
            event = fsm.advance(decision.token_id)
        except ConstraintViolation as exc:
            raise InvalidConstrainedTrace(
                f"decision {index} violates {fsm.state}: {exc}"
            ) from exc
        serialized = bool(event.serialized_token_ids)
        if decision.hidden != event.hidden or decision.serialized != serialized:
            raise InvalidConstrainedTrace(
                f"decision {index} event metadata differs from FSM replay"
            )
        semantic_mask.extend(1 for _ in event.serialized_token_ids)
        semantic_mask.extend(0 for _ in event.deterministic_token_ids)
        hidden_count += int(event.hidden)

    _force_transition_if_needed(fsm, semantic_mask)
    if not fsm.done:
        raise InvalidConstrainedTrace(
            f"decision trace ended before grammar completion in state {fsm.state!r}"
        )
    if len(semantic_mask) != len(fsm.output_token_ids):
        raise InvalidConstrainedTrace("semantic mask and visible response length differ")

    return ConstrainedReplay(
        response_token_ids=fsm.output_token_ids,
        semantic_response_mask=tuple(semantic_mask),
        decision_context_lengths=tuple(context_lengths),
        hidden_decision_count=hidden_count,
        forced_rationale_close=fsm.forced_rationale_close,
        judgments=dict(fsm.judgments),
    )


def _model_input_device(model: torch.nn.Module) -> torch.device:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight.device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise InvalidConstraintProbability("model has no parameters") from exc


def _sampling_distribution(
    logits: torch.Tensor,
    allowed_token_ids: Iterable[int],
    *,
    temperature: float,
    top_p: float,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    if logits.ndim != 1:
        raise InvalidConstraintProbability("logits must be a one-dimensional vector")
    if temperature <= 0:
        raise InvalidConstraintProbability("temperature must be positive")
    if not 0 < top_p <= 1:
        raise InvalidConstraintProbability("top_p must be in (0, 1]")
    allowed = tuple(int(token_id) for token_id in allowed_token_ids)
    if not allowed:
        raise InvalidConstraintProbability("allowed token set must not be empty")
    allowed_index = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    allowed_logits = logits.index_select(0, allowed_index).float()
    if not bool(torch.isfinite(allowed_logits).all().item()):
        raise InvalidConstraintProbability("allowed logits contain a non-finite value")

    raw_log_probs = torch.log_softmax(allowed_logits, dim=0)
    sampling_logits = allowed_logits / float(temperature)
    if top_p < 1.0:
        sorted_logits, sorted_positions = torch.sort(sampling_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=0)
        remove = torch.cumsum(sorted_probs, dim=0) > float(top_p)
        if remove.numel() > 1:
            remove[1:] = remove[:-1].clone()
        remove[0] = False
        sampling_logits = torch.full_like(sampling_logits, -torch.inf)
        sampling_logits[sorted_positions[~remove]] = sorted_logits[~remove]
    sampling_log_probs = torch.log_softmax(sampling_logits, dim=0)
    return allowed, raw_log_probs, sampling_log_probs


def _sample_action(
    logits: torch.Tensor,
    allowed_token_ids: Iterable[int],
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> tuple[int, float, float]:
    allowed, raw_log_probs, sampling_log_probs = _sampling_distribution(
        logits,
        allowed_token_ids,
        temperature=temperature,
        top_p=top_p,
    )
    if do_sample:
        position = int(
            torch.multinomial(
                sampling_log_probs.exp(),
                num_samples=1,
                generator=generator,
            ).item()
        )
    else:
        position = int(torch.argmax(sampling_log_probs).item())
    return (
        allowed[position],
        float(raw_log_probs[position].item()),
        float(sampling_log_probs[position].item()),
    )


def _model_forward(
    model: torch.nn.Module,
    token_ids: Sequence[int],
    *,
    past_key_values: Any = None,
    use_cache: bool,
) -> Any:
    if not token_ids:
        raise InvalidConstraintProbability("model forward requires at least one token")
    device = _model_input_device(model)
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": use_cache,
    }
    if past_key_values is None:
        kwargs["attention_mask"] = torch.ones_like(input_ids)
    else:
        kwargs["past_key_values"] = past_key_values
    outputs = model(**kwargs)
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise InvalidConstraintProbability("model did not return rank-three logits")
    return outputs


def generate_constrained_rollout(
    model: torch.nn.Module,
    grammar: CompiledGuardianGrammar,
    *,
    prompt_token_ids: Sequence[int],
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> ConstrainedRollout:
    """Generate one response from the same Actor later used for recomputation."""

    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    if not prompt:
        raise InvalidConstraintProbability("prompt must contain at least one token")
    fsm = GuardianTokenFSM(grammar)
    decisions: list[ConstrainedDecision] = []
    maximum_decisions = (
        grammar.max_rationale_tokens
        + max(len(tokens) for tokens in grammar.rationale_end_trie.choices.values())
        + sum(
            max(len(tokens) for tokens in trie.choices.values())
            for trie in grammar.enum_tries.values()
        )
    )

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            outputs = _model_forward(
                model,
                prompt + grammar.start_tokens,
                use_cache=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise InvalidConstraintProbability(
                    "same-Actor rollout requires a model KV cache"
                )
            next_logits = outputs.logits[0, -1]

            while not fsm.done:
                allowed = fsm.allowed_token_ids()
                if not allowed:
                    if fsm.state != "rationale":
                        raise InvalidConstrainedTrace(
                            f"state {fsm.state!r} has no legal transition"
                        )
                    event = fsm.force_rationale_close()
                    appended = event.deterministic_token_ids
                else:
                    if len(decisions) >= maximum_decisions:
                        raise InvalidConstrainedTrace(
                            "semantic decision count exceeded grammar maximum"
                        )
                    state_before = fsm.state
                    token_id, raw_log_prob, sampling_log_prob = _sample_action(
                        next_logits,
                        allowed,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        generator=generator,
                    )
                    event = fsm.advance(token_id)
                    decisions.append(
                        ConstrainedDecision(
                            state_before=state_before,
                            token_id=token_id,
                            rollout_log_prob=raw_log_prob,
                            sampling_log_prob=sampling_log_prob,
                            hidden=event.hidden,
                            serialized=bool(event.serialized_token_ids),
                        )
                    )
                    appended = (
                        event.serialized_token_ids + event.deterministic_token_ids
                    )

                if not fsm.done:
                    outputs = _model_forward(
                        model,
                        appended,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = getattr(outputs, "past_key_values", None)
                    if past_key_values is None:
                        raise InvalidConstraintProbability(
                            "model stopped returning a KV cache during rollout"
                        )
                    next_logits = outputs.logits[0, -1]
    finally:
        model.train(was_training)

    replay = replay_constrained_trace(
        grammar,
        prompt_length=len(prompt),
        decisions=tuple(decisions),
    )
    if replay.response_token_ids != fsm.output_token_ids:
        raise InvalidConstrainedTrace("rollout and replay visible responses differ")
    return ConstrainedRollout(
        prompt_token_ids=prompt,
        response_token_ids=replay.response_token_ids,
        semantic_response_mask=replay.semantic_response_mask,
        decisions=tuple(decisions),
        output_text=fsm.output_text(),
        judgments=dict(fsm.judgments),
        forced_rationale_close=fsm.forced_rationale_close,
    )


def recompute_constrained_log_probs(
    model: torch.nn.Module,
    grammar: CompiledGuardianGrammar,
    *,
    prompt_token_ids: Sequence[int],
    decisions: Sequence[ConstrainedDecision],
    force_eval: bool = True,
) -> torch.Tensor:
    """Recompute differentiable log-probs for semantic decisions in one pass."""

    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    if not prompt:
        raise InvalidConstraintProbability("prompt must contain at least one token")
    replay = replay_constrained_trace(
        grammar,
        prompt_length=len(prompt),
        decisions=decisions,
    )
    full_sequence = prompt + replay.response_token_ids

    was_training = model.training
    if force_eval:
        model.eval()
    try:
        outputs = _model_forward(model, full_sequence, use_cache=False)
    finally:
        if force_eval:
            model.train(was_training)
    logits = outputs.logits[0]

    fsm = GuardianTokenFSM(grammar)
    semantic_mask: list[int] = [0] * len(grammar.start_tokens)
    log_probs: list[torch.Tensor] = []
    for index, (decision, context_length) in enumerate(
        zip(decisions, replay.decision_context_lengths)
    ):
        _force_transition_if_needed(fsm, semantic_mask)
        if decision.state_before != fsm.state:
            raise InvalidConstrainedTrace(
                f"decision {index} state changed during probability replay"
            )
        if context_length <= 0 or context_length > logits.shape[0]:
            raise InvalidConstrainedTrace(
                f"decision {index} context position is outside model logits"
            )
        allowed = fsm.allowed_token_ids()
        log_probs.append(
            constrained_token_log_prob(
                logits[context_length - 1],
                allowed_token_ids=allowed,
                selected_token_id=decision.token_id,
            )
        )
        event = fsm.advance(decision.token_id)
        semantic_mask.extend(1 for _ in event.serialized_token_ids)
        semantic_mask.extend(0 for _ in event.deterministic_token_ids)

    if not log_probs:
        raise InvalidConstrainedTrace("decision trace has no semantic actions")
    result = torch.stack(log_probs)
    if not bool(torch.isfinite(result).all().item()):
        raise InvalidConstraintProbability(
            "recomputed constrained log-probabilities are non-finite"
        )
    return result

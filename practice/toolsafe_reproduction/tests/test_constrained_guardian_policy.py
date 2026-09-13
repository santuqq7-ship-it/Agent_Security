"""Focused E3 tests for constrained rollout and log-probability replay."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from constrained_guardian_fsm import GuardianTokenFSM, compile_guardian_grammar  # noqa: E402
from constrained_guardian_policy import (  # noqa: E402
    ConstrainedDecision,
    InvalidConstrainedTrace,
    constrained_token_log_prob,
    generate_constrained_rollout,
    recompute_constrained_log_probs,
    replay_constrained_trace,
)
from e3_constrained_logprob_canary import build_e3_gate_report  # noqa: E402


class CharacterTokenizer:
    """Every printable ASCII character and newline is one token."""

    def __init__(self) -> None:
        alphabet = "".join(chr(codepoint) for codepoint in range(32, 127)) + "\n"
        self._token_to_id = {
            character: index + 2 for index, character in enumerate(alphabet)
        }
        self._token_to_id[" <"] = len(self._token_to_id) + 2
        self._id_to_token = {
            value: key for key, value in self._token_to_id.items()
        }
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.all_special_ids = [0, 1]
        self.vocab_size = len(self._token_to_id) + 2
        self.name_or_path = "character-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith(" <", index):
                token_ids.append(self._token_to_id[" <"])
                index += 2
            else:
                token_ids.append(self._token_to_id[text[index]])
                index += 1
        return token_ids

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(self._id_to_token.get(token_id, "") for token_id in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)


def _record_text(
    fsm: GuardianTokenFSM,
    tokenizer: CharacterTokenizer,
    text: str,
) -> list[ConstrainedDecision]:
    decisions: list[ConstrainedDecision] = []
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        state_before = fsm.state
        event = fsm.advance(token_id)
        decisions.append(
            ConstrainedDecision(
                state_before=state_before,
                token_id=token_id,
                rollout_log_prob=0.0,
                sampling_log_prob=0.0,
                hidden=event.hidden,
                serialized=bool(event.serialized_token_ids),
            )
        )
    return decisions


def _complete_trace() -> tuple[
    CharacterTokenizer, object, tuple[ConstrainedDecision, ...]
]:
    tokenizer = CharacterTokenizer()
    grammar = compile_guardian_grammar(
        tokenizer,
        min_rationale_content_tokens=2,
        max_rationale_tokens=32,
    )
    fsm = GuardianTokenFSM(grammar)
    decisions: list[ConstrainedDecision] = []
    decisions += _record_text(fsm, tokenizer, "Safe action.")
    for token_id in grammar.rationale_end_trie.choices["end"]:
        state_before = fsm.state
        event = fsm.advance(token_id)
        decisions.append(
            ConstrainedDecision(
                state_before=state_before,
                token_id=token_id,
                rollout_log_prob=0.0,
                sampling_log_prob=0.0,
                hidden=event.hidden,
                serialized=bool(event.serialized_token_ids),
            )
        )
    decisions += _record_text(fsm, tokenizer, " no ")
    decisions += _record_text(fsm, tokenizer, " yes ")
    decisions += _record_text(fsm, tokenizer, " 0.5 ")
    assert fsm.done
    return tokenizer, grammar, tuple(decisions)


def test_constrained_log_prob_ignores_disallowed_logits() -> None:
    logits = torch.tensor([1_000_000.0, 2.0, 1.0, -4.0])
    actual = constrained_token_log_prob(
        logits,
        allowed_token_ids=(1, 2),
        selected_token_id=1,
    )

    expected = -math.log1p(math.exp(-1.0))
    assert actual.item() == pytest.approx(expected, abs=1e-6)


def test_trace_replay_masks_fixed_tokens_and_keeps_hidden_end_decision() -> None:
    tokenizer, grammar, decisions = _complete_trace()

    replay = replay_constrained_trace(
        grammar,
        prompt_length=3,
        decisions=decisions,
    )

    assert tokenizer.decode(list(replay.response_token_ids)) == (
        "<Think> Safe action. <\\Think>\n"
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> yes <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 0.5 <\\Harmfulness_Rating>"
    )
    assert sum(replay.semantic_response_mask) == sum(
        decision.serialized for decision in decisions
    )
    assert replay.hidden_decision_count == 1
    assert len(decisions) == sum(replay.semantic_response_mask) + 1
    assert replay.judgments == {
        "Malicious_User_Request": False,
        "Being_Attacked": True,
        "Harmfulness_Rating": 0.5,
    }


def test_trace_replay_rejects_state_mismatch() -> None:
    _, grammar, decisions = _complete_trace()
    first = decisions[0]
    corrupt = (
        ConstrainedDecision(
            state_before="being_attacked",
            token_id=first.token_id,
            rollout_log_prob=first.rollout_log_prob,
            sampling_log_prob=first.sampling_log_prob,
            hidden=first.hidden,
            serialized=first.serialized,
        ),
        *decisions[1:],
    )

    with pytest.raises(InvalidConstrainedTrace, match="state mismatch"):
        replay_constrained_trace(grammar, prompt_length=3, decisions=corrupt)


class TinyCausalLM(torch.nn.Module):
    """A differentiable causal LM whose preferred token depends on context size."""

    def __init__(
        self,
        vocab_size: int,
        preferred_by_context_length: dict[int, int],
    ) -> None:
        super().__init__()
        self.base_logits = torch.nn.Parameter(torch.zeros(vocab_size))
        self.preferred_by_context_length = dict(preferred_by_context_length)
        self.observed_training_modes: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[int] | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask
        self.observed_training_modes.append(self.training)
        past_length = 0 if past_key_values is None else past_key_values[0]
        batch_size, sequence_length = input_ids.shape
        logits = self.base_logits.view(1, 1, -1).expand(
            batch_size, sequence_length, -1
        ).clone()
        for offset in range(sequence_length):
            context_length = past_length + offset + 1
            preferred = self.preferred_by_context_length.get(context_length)
            if preferred is not None:
                logits[:, offset, preferred] += 12.0
        cache = (past_length + sequence_length,) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _preferred_schedule(
    grammar: object,
    tokenizer: CharacterTokenizer,
    prompt_length: int,
) -> dict[int, int]:
    fsm = GuardianTokenFSM(grammar)
    schedule: dict[int, int] = {}
    decision_tokens = tokenizer.encode("Safe action.", add_special_tokens=False)
    decision_tokens += list(grammar.rationale_end_trie.choices["end"])
    for text in (" no ", " yes ", " 0.5 "):
        decision_tokens += tokenizer.encode(text, add_special_tokens=False)
    for token_id in decision_tokens:
        schedule[prompt_length + len(fsm.output_token_ids)] = token_id
        fsm.advance(token_id)
    assert fsm.done
    return schedule


def test_same_actor_rollout_matches_differentiable_trace_recomputation() -> None:
    tokenizer = CharacterTokenizer()
    grammar = compile_guardian_grammar(
        tokenizer,
        min_rationale_content_tokens=2,
        max_rationale_tokens=32,
    )
    prompt_token_ids = (tokenizer.encode("P", add_special_tokens=False)[0],)
    model = TinyCausalLM(
        tokenizer.vocab_size,
        _preferred_schedule(grammar, tokenizer, len(prompt_token_ids)),
    )

    rollout = generate_constrained_rollout(
        model,
        grammar,
        prompt_token_ids=prompt_token_ids,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    recomputed = recompute_constrained_log_probs(
        model,
        grammar,
        prompt_token_ids=prompt_token_ids,
        decisions=rollout.decisions,
    )

    assert rollout.output_text == (
        "<Think> Safe action. <\\Think>\n"
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> yes <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 0.5 <\\Harmfulness_Rating>"
    )
    rollout_log_probs = torch.tensor(
        [decision.rollout_log_prob for decision in rollout.decisions]
    )
    assert torch.allclose(rollout_log_probs, recomputed.detach(), atol=1e-6)
    recomputed.sum().backward()
    assert model.base_logits.grad is not None


def test_recomputation_can_preserve_training_mode_for_checkpointed_backward() -> None:
    tokenizer, grammar, decisions = _complete_trace()
    prompt_token_ids = (tokenizer.encode("P", add_special_tokens=False)[0],)
    model = TinyCausalLM(
        tokenizer.vocab_size,
        _preferred_schedule(grammar, tokenizer, len(prompt_token_ids)),
    )
    model.train()

    log_probs = recompute_constrained_log_probs(
        model,
        grammar,
        prompt_token_ids=prompt_token_ids,
        decisions=decisions,
        force_eval=False,
    )
    log_probs.sum().backward()

    assert model.observed_training_modes == [True]
    assert model.training is True
    assert model.base_logits.grad is not None


def test_e3_report_is_a_hard_gate_and_never_claims_training() -> None:
    passing = build_e3_gate_report(
        tokenizer_match=True,
        strict_format=True,
        actor_rollout_log_probs=[-0.2, -1.0],
        actor_recomputed_log_probs=[-0.2, -1.1],
        actor_repeat_log_probs=[-0.2, -1.100001],
        reference_log_probs=[-0.3, -0.9],
        cache_parity_tolerance=0.2,
        authoritative_parity_tolerance=1e-3,
        decision_count=2,
        hidden_decision_count=1,
        forced_rationale_close=False,
        decision_states=["rationale", "being_attacked"],
    )
    failing = build_e3_gate_report(
        tokenizer_match=True,
        strict_format=True,
        actor_rollout_log_probs=[-0.2, -1.0],
        actor_recomputed_log_probs=[-0.2, -1.1],
        actor_repeat_log_probs=[-0.2, -1.2],
        reference_log_probs=[-0.3, -0.9],
        cache_parity_tolerance=0.2,
        authoritative_parity_tolerance=1e-3,
        decision_count=2,
        hidden_decision_count=1,
        forced_rationale_close=False,
        decision_states=["rationale", "being_attacked"],
    )

    assert passing["passed"] is True
    assert passing["training_started"] is False
    assert passing["optimizer_created"] is False
    assert passing["backward_executed"] is False
    assert failing["actor_parity_max_error_index"] == 1
    assert failing["actor_parity_max_error_state"] == "being_attacked"
    assert failing["passed"] is False
    assert failing["criteria"]["actor_cache_recompute_consistency"] is True
    assert failing["criteria"]["authoritative_old_logprob_repeat_parity"] is False

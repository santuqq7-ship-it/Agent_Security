"""Tests for the pure helpers used by the local Transformers smoke test."""

import math
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

# The standalone practice script is not an installed package. Add its parent
# directory explicitly so the test can import its pure helper functions.
sys.path.insert(0, str(Path(__file__).parents[1]))

from phase4_transformers_smoke import (
    _top_candidates,
    build_messages,
    compute_token_entropy,
    resolve_device,
    run_smoke,
)


def test_compute_token_entropy_matches_uniform_binary_distribution():
    """A uniform binary distribution has entropy log(2)."""
    logits = torch.tensor([[0.0, 0.0]])

    entropy = compute_token_entropy(logits)

    assert torch.allclose(entropy, torch.tensor([math.log(2.0)]), atol=1e-6)


def test_compute_token_entropy_is_lower_for_a_confident_distribution():
    """Concentrating probability on one token lowers entropy."""
    logits = torch.tensor([[10.0, -10.0]])

    entropy = compute_token_entropy(logits)

    assert entropy.item() < 1e-6


def test_compute_token_entropy_ignores_filtered_negative_infinity_logits():
    """Sampling filters use -inf for removed tokens, whose probability is zero."""
    logits = torch.tensor([[0.0, 0.0, -float("inf")]])

    entropy = compute_token_entropy(logits)

    assert torch.isfinite(entropy).all()
    assert torch.allclose(entropy, torch.tensor([math.log(2.0)]), atol=1e-6)


def test_top_candidates_omits_tokens_removed_by_sampling_filters():
    """Top candidates should not display arbitrary ties whose score is -inf."""

    class FakeTokenizer:
        def decode(self, token_ids):
            return f"token-{token_ids[0]}"

    logits = torch.tensor([[0.0, -1.0, -float("inf"), -float("inf")]])

    candidates = _top_candidates(logits, FakeTokenizer(), limit=5)

    assert len(candidates) == 2
    assert all(item["probability"] > 0.0 for item in candidates)


def test_resolve_device_auto_falls_back_to_cpu_without_mps():
    """The smoke test must remain runnable when MPS is unavailable."""
    assert resolve_device("auto", mps_available=False) == "cpu"


def test_build_messages_contains_the_production_react_call_shape():
    """The prompt must teach the exact labels consumed by ToolSafe's parser."""
    system_prompt = build_messages()[0]["content"]

    assert "Action: get_order_status" in system_prompt
    assert 'Action Input: {"order_id": "A-100"}' in system_prompt


def test_build_messages_contains_an_assistant_react_demonstration():
    """A few-shot assistant turn anchors the textual tool-call protocol."""
    messages = build_messages()

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Action: get_order_status" in messages[2]["content"]
    assert 'Action Input: {"order_id": "A-100"}' in messages[2]["content"]


def test_run_smoke_passes_tokenized_inputs_to_generation(monkeypatch, tmp_path):
    """Generation must receive the token batch, not the not-yet-created output."""

    class FakeBatch(dict):
        """Minimal BatchEncoding substitute with the ``.to`` method used by the script."""

        def to(self, _device):
            return self

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return "prompt"

        def __call__(self, _prompt_text, return_tensors):
            assert return_tensors == "pt"
            return FakeBatch(input_ids=torch.tensor([[1]]))

        def decode(self, _token_ids, skip_special_tokens=False):
            return ""

    captured = {}

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2]]),
                # The production helper asks for five candidates, so the
                # fake vocabulary must contain at least five score entries.
                scores=(torch.zeros((1, 5)),),
            )

    monkeypatch.setattr(
        "phase4_transformers_smoke.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "phase4_transformers_smoke.AutoModelForCausalLM.from_pretrained",
        lambda *_args, **_kwargs: FakeModel(),
    )

    report = run_smoke(
        {
            "model_path": str(tmp_path),
            "device": "cpu",
            "max_new_tokens": 1,
        }
    )

    assert "input_ids" in captured
    assert report["generated_token_count"] == 1

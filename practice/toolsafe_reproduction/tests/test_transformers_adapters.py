"""Contract tests for the local Transformers compatibility overlay.

These tests deliberately exercise the same constructor and parser contracts
used by ToolSafe. They do not replace Guardian decisions with Python rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Import the overlay before the original ToolSafe/src tree.
OVERLAY_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(OVERLAY_SRC))
sys.path.insert(1, str(Path(__file__).parents[3] / "ToolSafe" / "src"))
sys.path.insert(2, str(Path(__file__).parents[3] / "ToolSafe" / "src" / "task_executor"))

from model import model as model_module


class _FakeTokenizer:
    """Minimal tokenizer surface needed by the adapter constructor/generator."""

    eos_token_id = 0


class _FakeModel:
    """Minimal causal-LM surface that records device and evaluation state."""

    def __init__(self):
        self.device = "cpu"
        self.eval_called = False

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


class _FakeAgentResponse:
    class _Choice:
        class _Message:
            content = "Final Answer: done"

        message = _Message()

    choices = [_Choice()]


class _FakeCompletions:
    def create(self, **_kwargs):
        return _FakeAgentResponse()


class _FakeAgentLLM:
    chat = type("_Chat", (), {"completions": _FakeCompletions()})()


class _FakeAgentModelConfig:
    model_type = "api"
    model_name = "fake-agent"
    llm = _FakeAgentLLM()


def test_analysis_model_constructor_does_not_require_vllm(monkeypatch):
    """The local Transformers path must import without installing vLLM."""

    fake_model = _FakeModel()
    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_model,
    )

    agent_model = model_module.Model(
        model_name="Qwen2.5-1.5B-Instruct",
        model_path="/fake/agent",
        model_type="analysis",
    )

    assert agent_model.model_type == "analysis"
    assert agent_model.tokenizer.__class__ is _FakeTokenizer
    assert agent_model.llm is fake_model
    assert fake_model.eval_called is True


def test_guardian_analysis_constructor_loads_lora_adapter_on_base_model(monkeypatch):
    """An SFT adapter must be applied after loading its declared base model."""

    fake_base = _FakeModel()
    fake_adapter = _FakeModel()
    calls = {}

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, is_trainable=False):
            calls.update(
                model=model,
                adapter_path=adapter_path,
                is_trainable=is_trainable,
            )
            return fake_adapter

    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_base,
    )
    monkeypatch.setitem(model_module.__dict__, "PeftModel", FakePeftModel)

    guardian = model_module.Guardian(
        model_name="ashell-guardian-single",
        model_path="/fake/base",
        adapter_path="/fake/adapter",
        model_type="analysis",
    )

    assert guardian.adapter_path == "/fake/adapter"
    assert calls == {
        "model": fake_base,
        "adapter_path": "/fake/adapter",
        "is_trainable": False,
    }
    assert guardian.llm is fake_adapter
    assert fake_adapter.eval_called is True


def test_agent_trace_callback_receives_full_messages_and_output():
    """Tracing observes the exact Agent messages without changing its result."""

    from agent.agent import Agent_Core

    events = []
    bot = Agent_Core(
        system="system text",
        agentic_model=_FakeAgentModelConfig(),
        trace_callback=events.append,
    )

    assert bot("user text") == "Final Answer: done"
    assert events[0]["kind"] == "agent_request"
    assert events[0]["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert events[1] == {
        "source": "agent",
        "kind": "agent_response",
        "text": "Final Answer: done",
    }


def test_guardian_analysis_response_uses_original_parser(monkeypatch):
    """A valid TS-Guard response becomes the original risk result schema."""

    guardian = model_module.Guardian.__new__(model_module.Guardian)
    guardian.model_name = "TS-Guard"
    guardian.model_type = "analysis"
    guardian._generate_text = lambda _messages: (
        "<Malicious_User_Request>no<\\Malicious_User_Request>\n"
        "<Being_Attacked>yes<\\Being_Attacked>\n"
        "<Harmfulness_Rating>0.5<\\Harmfulness_Rating>"
    )

    result = guardian.get_judgment_res(
        {
            "env_info": "read-only order lookup",
            "user_request": "Check order A-100",
            "agent_action": {
                "interaction_history": [],
                "current_action": "get_order_status(A-100)",
            },
        },
        max_turn=1,
    )

    assert result["risk rating"] == 0.5
    assert result["results"] == {
        "Malicious_User_Request": "no",
        "Being_Attacked": "yes",
        "Harmfulness_Rating": 0.5,
    }
    assert "<Harmfulness_Rating>0.5<\\Harmfulness_Rating>" in result["reason"]


def test_guardian_invalid_response_retries_without_rule_based_fallback():
    """Invalid model text remains invalid after the configured retry count."""

    guardian = model_module.Guardian.__new__(model_module.Guardian)
    guardian.model_name = "TS-Guard"
    guardian.model_type = "analysis"
    calls = []

    def invalid_response(_messages):
        calls.append(True)
        return "This answer does not follow the TS-Guard output format."

    guardian._generate_text = invalid_response
    result = guardian.get_judgment_res(
        {
            "env_info": "read-only order lookup",
            "user_request": "Check order A-100",
            "agent_action": {"interaction_history": [], "current_action": "lookup"},
        },
        max_turn=2,
    )

    assert len(calls) == 2
    assert "risk rating" not in result
    assert result["reason"] == "This answer does not follow the TS-Guard output format."


def test_original_guardian_evaluator_modules_remain_importable():
    """The overlay package must expose unchanged evaluator modules."""

    from guardian_evaluator.agentharm import AgentHarmProcessor
    from guardian_evaluator.asb import ASBProcessor
    from guardian_evaluator.agentdojo import AgentDojoProcessor

    assert AgentHarmProcessor.__module__ == "guardian_evaluator.agentharm"
    assert ASBProcessor.__module__ == "guardian_evaluator.asb"
    assert AgentDojoProcessor.__module__ == "guardian_evaluator.agentdojo"


def test_secreact_exposes_original_agentdojo_result_fields():
    """The original AgentDojo writer can read SecReAct trace metadata fields."""

    from agent.sec_react_agent import SecReAct_Agent

    agent = SecReAct_Agent()
    assert agent.normal_tools == []
    assert agent.attack_tools == []
    assert agent.step_labels == []


def test_secreact_suppresses_legacy_prints_when_trace_callback_is_enabled(capsys):
    """The readable trace must be the only console representation of a run."""

    from agent.sec_react_agent import SecReAct_Agent

    traced_agent = SecReAct_Agent(trace_callback=lambda _event: None)
    traced_agent._legacy_print("duplicate debug line")
    assert capsys.readouterr().out == ""

    plain_agent = SecReAct_Agent()
    plain_agent._legacy_print("legacy fallback")
    assert capsys.readouterr().out == "legacy fallback\n"


def test_trace_runner_converts_non_json_dictionary_keys_to_strings():
    """Trace output must serialize AgentDojo tuple-keyed result dictionaries."""

    sys.path.insert(0, str(Path(__file__).parents[1] / "runners"))
    from run_secreact_trace import make_json_safe

    value = {
        ("user_task_1", ""): {"passed": True},
        "nested": [("kept", {1: "value"})],
    }

    assert make_json_safe(value) == {
        "('user_task_1', '')": {"passed": True},
        "nested": [["kept", {"1": "value"}]],
    }


def test_runner_rejects_reusing_existing_benchmark_result_directory(tmp_path):
    """A cached AgentDojo directory must be explicitly reset or replaced."""

    sys.path.insert(0, str(Path(__file__).parents[1] / "runners"))
    from run_secreact_trace import prepare_output_dir

    for name in ("meta_data.json", "utility.json", "security.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already contains benchmark results"):
        prepare_output_dir(tmp_path, reset=False)


def test_runner_reset_removes_only_known_benchmark_artifacts(tmp_path):
    """Explicit reset clears stale result files without deleting other files."""

    sys.path.insert(0, str(Path(__file__).parents[1] / "runners"))
    from run_secreact_trace import prepare_output_dir

    for name in ("meta_data.json", "utility.json", "security.json", "trace.jsonl"):
        (tmp_path / name).write_text("old", encoding="utf-8")
    keep = tmp_path / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    prepare_output_dir(tmp_path, reset=True)

    assert not any((tmp_path / name).exists() for name in ("meta_data.json", "utility.json", "security.json", "trace.jsonl"))
    assert keep.read_text(encoding="utf-8") == "keep"


def test_readable_trace_deduplicates_prompts_and_preserves_newlines(tmp_path, capsys):
    """Human trace shows one prompt copy while raw JSONL keeps exact events."""

    sys.path.insert(0, str(Path(__file__).parents[1] / "runners"))
    from run_secreact_trace import TracePrinter

    printer = TracePrinter(tmp_path / "trace.jsonl", show_full_prompts=True)
    system_prompt = "system line 1\nsystem line 2"
    printer({
        "source": "agent",
        "kind": "agent_request",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "first request"},
        ],
    })
    printer({
        "source": "agent",
        "kind": "agent_request",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Observation: second round"},
        ],
    })
    printer({
        "source": "agent",
        "kind": "agent_response",
        "text": "Thought: inspect\nAction: lookup",
    })
    printer.close()

    output = capsys.readouterr().out
    assert output.count(system_prompt) == 1
    assert "system prompt reused" in output
    assert "Thought: inspect\nAction: lookup" in output
    assert "\\nAction" not in output
    raw_events = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_events) == 3

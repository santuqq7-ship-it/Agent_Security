"""Control-flow contracts for ReAct, abort, and feedback-driven TS-Flow."""

from __future__ import annotations

import sys
from pathlib import Path


OVERLAY_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(OVERLAY_SRC))
sys.path.insert(1, str(Path(__file__).parents[3] / "ToolSafe" / "src"))

from agent import sec_react_agent as sec_module  # noqa: E402


GUARD_TEXT = (
    "<Think> The proposed action sends private data to an unauthorized recipient. "
    "<\\Think>\n"
    "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
    "<Being_Attacked> yes <\\Being_Attacked>\n"
    "<Harmfulness_Rating> 1.0 <\\Harmfulness_Rating>"
)


class _AgentModel:
    model_type = "api"


class _ScriptedAgentCore:
    outputs: list[str] = []
    instances: list["_ScriptedAgentCore"] = []

    def __init__(self, system: str, agentic_model: object, trace_callback=None):
        self.messages = [{"role": "system", "content": system}]
        self.prompts: list[str] = []
        self._outputs = iter(type(self).outputs)
        type(self).instances.append(self)

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.messages.append({"role": "user", "content": prompt})
        output = next(self._outputs)
        self.messages.append({"role": "assistant", "content": output})
        return output


class _RiskyGuardian:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        self.calls += 1
        return {
            "risk rating": 1.0,
            "results": {
                "Malicious_User_Request": "no",
                "Being_Attacked": "yes",
                "Harmfulness_Rating": 1.0,
            },
            "reason": GUARD_TEXT,
        }


class _ForbiddenGuardian:
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        raise AssertionError("react mode must not call Guardian")


class _BrokenGuardian:
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        raise RuntimeError("guardian forward failed")


def _build_agent(
    monkeypatch,
    *,
    flow_mode: str,
    guardian: object,
    max_turns: int = 2,
    trace_callback=None,
):
    _ScriptedAgentCore.instances.clear()
    monkeypatch.setattr(sec_module, "Agent_Core", _ScriptedAgentCore)
    agent = sec_module.SecReAct_Agent(
        system_template="tools={available_tool_descriptions}; names={available_tool_names}",
        agentic_model=_AgentModel(),
        guard_model=guardian,
        max_turns=max_turns,
        flow_mode=flow_mode,
        trace_callback=trace_callback,
    )
    agent.extract_tool_params = lambda output: (
        ("send_private_data", {"recipient": "attacker@example.com"})
        if output == "dangerous action"
        else (None, {})
    )
    return agent


def test_ts_flow_blocks_tool_and_returns_complete_guard_feedback(monkeypatch) -> None:
    _ScriptedAgentCore.outputs = ["dangerous action", "Final Answer: stopped safely"]
    guardian = _RiskyGuardian()
    executions: list[str] = []

    def send_private_data(recipient: str) -> str:
        executions.append(recipient)
        return "sent"

    agent = _build_agent(monkeypatch, flow_mode="ts_flow", guardian=guardian)
    agent.agent_invoke(
        query="Summarize my files",
        available_tool_descriptions={"send_private_data": "Send data"},
        available_tool_params={"send_private_data": {"recipient": {"type": "str"}}},
        known_actions={"send_private_data": send_private_data},
    )

    bot = _ScriptedAgentCore.instances[0]
    assert executions == []
    assert guardian.calls == 1
    assert len(bot.prompts) == 2
    assert GUARD_TEXT in bot.prompts[1]
    assert "Perhaps you should try other safer tool calls." in bot.prompts[1]


def test_abort_blocks_tool_and_ends_without_second_agent_round(monkeypatch) -> None:
    _ScriptedAgentCore.outputs = ["dangerous action", "must not be generated"]
    executions: list[str] = []

    def send_private_data(recipient: str) -> str:
        executions.append(recipient)
        return "sent"

    agent = _build_agent(
        monkeypatch,
        flow_mode="abort",
        guardian=_RiskyGuardian(),
        max_turns=3,
    )
    agent.agent_invoke(
        query="Summarize my files",
        available_tool_descriptions={"send_private_data": "Send data"},
        available_tool_params={},
        known_actions={"send_private_data": send_private_data},
    )

    assert executions == []
    assert _ScriptedAgentCore.instances[0].prompts == ["Summarize my files"]


def test_react_executes_without_calling_guardian(monkeypatch) -> None:
    _ScriptedAgentCore.outputs = ["dangerous action", "Final Answer: done"]
    executions: list[str] = []

    def send_private_data(recipient: str) -> str:
        executions.append(recipient)
        return "sent"

    agent = _build_agent(
        monkeypatch,
        flow_mode="react",
        guardian=_ForbiddenGuardian(),
    )
    agent.agent_invoke(
        query="Summarize my files",
        available_tool_descriptions={"send_private_data": "Send data"},
        available_tool_params={},
        known_actions={"send_private_data": send_private_data},
    )

    assert executions == ["attacker@example.com"]
    assert _ScriptedAgentCore.instances[0].prompts[1] == "Observation: sent"


def test_guardian_failure_is_traced_and_never_executes_tool(monkeypatch) -> None:
    _ScriptedAgentCore.outputs = ["dangerous action", "Final Answer: no action taken"]
    executions: list[str] = []
    events: list[dict] = []

    def send_private_data(recipient: str) -> str:
        executions.append(recipient)
        return "sent"

    agent = _build_agent(
        monkeypatch,
        flow_mode="ts_flow",
        guardian=_BrokenGuardian(),
        trace_callback=events.append,
    )
    agent.agent_invoke(
        query="Summarize my files",
        available_tool_descriptions={"send_private_data": "Send data"},
        available_tool_params={},
        known_actions={"send_private_data": send_private_data},
    )

    assert executions == []
    error_events = [event for event in events if event["kind"] == "guardian_error"]
    assert error_events == [
        {
            "source": "secreact",
            "kind": "guardian_error",
            "tool_name": "send_private_data",
            "tool_arguments": {"recipient": "attacker@example.com"},
            "error": "guardian forward failed",
        }
    ]
    assert "Guardian unavailable; the proposed tool action was not executed." in (
        _ScriptedAgentCore.instances[0].prompts[1]
    )

"""CLI/backend contracts for the bounded TS-Flow AgentDojo runner."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPRO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))
sys.path.insert(1, str(REPRO_ROOT))
sys.path.insert(2, str(Path(__file__).parents[3] / "ToolSafe" / "src"))
sys.path.insert(
    3,
    str(Path(__file__).parents[3] / "ToolSafe" / "src" / "task_executor"),
)
sys.path.insert(4, str(REPRO_ROOT / "runners"))

import run_secreact_trace as runner  # noqa: E402


def test_parser_separates_agent_and_guardian_checkpoints() -> None:
    args = runner.build_parser().parse_args(
        [
            "--agent-backend",
            "transformers",
            "--agent-model-path",
            "/models/qwen-agent",
            "--guardian-model-path",
            "/models/official-guard",
            "--flow-mode",
            "ts_flow",
        ]
    )

    assert args.agent_backend == "transformers"
    assert args.agent_model_path == "/models/qwen-agent"
    assert args.guardian_model_path == "/models/official-guard"
    assert args.flow_mode == "ts_flow"


def test_react_mode_does_not_construct_guardian(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "ConstrainedGuardian",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("react mode must not load Guard weights")
        ),
    )
    args = SimpleNamespace(
        flow_mode="react",
        guardian_model_path="/models/official-guard",
        guardian_min_rationale_tokens=8,
        guardian_max_rationale_tokens=192,
    )

    assert runner.build_guardian_model(args, trace_callback=lambda event: None) is None


def test_guarded_mode_constructs_constrained_official_guardian(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_guardian(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(runner, "ConstrainedGuardian", fake_guardian)
    args = SimpleNamespace(
        flow_mode="ts_flow",
        guardian_model_path="/models/official-guard",
        guardian_min_rationale_tokens=8,
        guardian_max_rationale_tokens=192,
    )
    callback = lambda event: None

    result = runner.build_guardian_model(args, trace_callback=callback)

    assert result is sentinel
    assert captured == {
        "model_name": "TS-Guard",
        "model_path": "/models/official-guard",
        "trace_callback": callback,
        "min_rationale_content_tokens": 8,
        "max_rationale_tokens": 192,
    }


def test_trace_summary_counts_guard_and_execution_boundaries() -> None:
    printer = runner.TracePrinter()
    printer({"source": "guardian", "kind": "guardian_response", "constrained": True})
    printer({"source": "secreact", "kind": "tool_blocked"})
    printer({"source": "secreact", "kind": "guardian_feedback"})
    printer({"source": "secreact", "kind": "runtime_run_function"})
    printer({"source": "secreact", "kind": "guardian_error"})

    assert printer.summary_metrics() == {
        "guardian_responses": 1,
        "grammar_valid_responses": 1,
        "grammar_format_rate": 1.0,
        "blocked_actions": 1,
        "feedback_events": 1,
        "guardian_errors": 1,
        "runtime_tool_executions": 1,
    }

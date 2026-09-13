"""Run one real ToolSafe SecReAct + AgentDojo task with full tracing.

This runner intentionally uses the original AgentDojo benchmark objects and
the copied ToolSafe ``SecReAct_Agent``. It only replaces the model backends
with local Transformers and attaches read-only trace callbacks. The terminal
shows a readable round-by-round chain; every complete event is also written to
JSONL so the data flow can be inspected programmatically.

Run this file from the workspace root with ``PYTHONPATH`` pointing at both the
overlay and the original source tree.  AgentDojo's Python dependencies must be
installed in ``ToolSafe/.venv-phase4`` before running it.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, TextIO

# The original prompt intentionally contains the literal ``<\Tag>`` protocol.
# Python warns about that backslash while compiling the unchanged source; the
# warning is unrelated to runtime behavior and is suppressed for this runner.
warnings.filterwarnings("ignore", category=SyntaxWarning)

from agent.agent_prompts import AGENT_PROMPT_TEMPLATES
from agent.sec_react_agent import SecReAct_Agent
from model.model import Guardian, Model

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.task_suite.load_suites import get_suite


BENCHMARK_VERSION = "v1.2.2"
BENCHMARK_ARTIFACTS = (
    "meta_data.json",
    "utility.json",
    "security.json",
    "trace.jsonl",
)


def prepare_output_dir(output_dir: Path, reset: bool = False) -> Path:
    """Validate or explicitly reset the small set of benchmark output files.

    AgentDojo's original single-task writer appends metadata but overwrites
    dictionary results for the same task key. Reusing a directory can therefore
    trigger a deep assertion after a previous run. Failing early keeps that
    cache corruption visible; ``reset=True`` removes only known artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in BENCHMARK_ARTIFACTS if (output_dir / name).exists()]
    if existing and not reset:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(
            f"Output directory already contains benchmark results ({names}). "
            "Choose a new --output-dir or pass --reset-output."
        )
    if reset:
        for path in existing:
            if path.is_file():
                path.unlink()
    return output_dir


def make_json_safe(value: Any) -> Any:
    """Convert benchmark results into values accepted by ``json.dumps``.

    AgentDojo uses tuple keys such as ``(user_task_id, injection_id)`` in its
    result dictionaries.  ``json.dumps(default=str)`` handles unusual values,
    but it does not transform dictionary keys, so we recurse explicitly and
    stringify every key while preserving the result data.
    """
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def _common_prefix_length(left: str, right: str) -> int:
    """Return the number of identical leading characters in two strings."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _print_text(label: str, value: Any) -> None:
    """Print a text field literally so embedded newlines remain readable."""
    print(f"{label}:")
    print("" if value is None else str(value))


def _print_small_json(label: str, value: Any) -> None:
    """Print compact structured fields without embedding large prompt strings."""
    print(f"{label}: {json.dumps(make_json_safe(value), ensure_ascii=False, default=str)}")


class TracePrinter:
    """Render a readable event chain and persist every raw event as JSONL.

    The original callback printed every full message history at every model
    boundary.  This class keeps that exact history in ``trace.jsonl`` while
    the terminal view prints only newly appended messages and prompt deltas.
    """

    def __init__(
        self,
        raw_path: Path | None = None,
        stream: TextIO | None = None,
        show_full_prompts: bool = False,
    ):
        self.raw_path = raw_path
        self.stream = stream
        self.show_full_prompts = show_full_prompts
        self.raw_file = None
        if raw_path is not None:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_file = raw_path.open("w", encoding="utf-8")
        self.agent_round = 0
        self.guardian_call = 0
        self.tool_call = 0
        self._last_agent_messages: list[dict[str, Any]] = []
        self._last_agent_rendered_prompt = ""
        self._last_guardian_prompt = ""
        self._agent_outputs: set[str] = set()

    def __call__(self, event: dict[str, Any]) -> None:
        """Write one event to raw JSONL and render its readable counterpart."""
        safe_event = make_json_safe(event)
        if self.raw_file is not None:
            self.raw_file.write(json.dumps(safe_event, ensure_ascii=False, default=str) + "\n")
            self.raw_file.flush()
        self._render(event)

    def _render(self, event: dict[str, Any]) -> None:
        source = event.get("source", "unknown")
        kind = event.get("kind", "event")

        if source == "agent" and kind == "agent_request":
            self.agent_round += 1
            messages = event.get("messages", [])
            common = 0
            while (
                common < len(messages)
                and common < len(self._last_agent_messages)
                and messages[common] == self._last_agent_messages[common]
            ):
                common += 1
            print(f"\n=== Agent round {self.agent_round}: input context ===")
            if common:
                if common and messages and self._last_agent_messages and messages[0] == self._last_agent_messages[0]:
                    print(f"[system prompt reused; {common} unchanged message(s) omitted]")
                else:
                    print(f"[unchanged context: {common} message(s) omitted]")
            for message in messages[common:]:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "assistant" and content in self._agent_outputs:
                    print("[previous Agent output already shown; omitted from context]")
                    continue
                if role == "system" and not self.show_full_prompts:
                    print(f"Agent system prompt: {len(str(content))} characters [full text in trace.jsonl]")
                    continue
                _print_text(f"Agent {role}", content)
            if common == len(messages):
                print("[no new messages]")
            self._last_agent_messages = [dict(message) for message in messages]
            return

        if source == "agent" and kind == "agent_rendered_prompt":
            prompt = str(event.get("prompt", ""))
            print("\n--- Agent Transformers prompt ---")
            if not self._last_agent_rendered_prompt and self.show_full_prompts:
                _print_text("Full rendered prompt", prompt)
            elif not self._last_agent_rendered_prompt:
                print(f"Full rendered prompt: {len(prompt)} characters [full text in trace.jsonl]")
            elif prompt == self._last_agent_rendered_prompt:
                print("[same rendered prompt as previous request; omitted]")
            else:
                common = _common_prefix_length(self._last_agent_rendered_prompt, prompt)
                print(f"[unchanged prompt prefix: {common} character(s) omitted]")
                _print_text("New prompt suffix", prompt[common:])
            self._last_agent_rendered_prompt = prompt
            return

        if source == "agent" and kind == "agent_response":
            text = str(event.get("text", ""))
            self._agent_outputs.add(text)
            _print_text(f"\n=== Agent round {self.agent_round}: raw output ===", text)
            print("[This is model-generated text, not a real tool Observation]")
            if event.get("entropy_stats"):
                _print_small_json("Entropy", event["entropy_stats"])
            return

        if source == "secreact" and kind == "parsed_action":
            _print_small_json(
                "\n--- Agent parser result ---",
                {
                    "tool_name": event.get("tool_name"),
                    "tool_arguments": event.get("tool_arguments"),
                },
            )
            return

        if source == "secreact" and kind == "guardian_call":
            self.guardian_call += 1
            arguments = event.get("arguments", {})
            print(f"\n=== Guardian check {self.guardian_call} ===")
            _print_small_json(
                "Request summary",
                {
                    "user_request": arguments.get("user_request"),
                    "interaction_history_count": len(arguments.get("interaction_history", [])),
                    "current_action": arguments.get("current_action"),
                },
            )
            return

        if source == "guardian" and kind == "guardian_request":
            prompt = str(event.get("rendered_prompt", ""))
            print("\n--- Guardian model input ---")
            if prompt and prompt == self._last_guardian_prompt:
                print("[same Guardian input as previous retry; omitted]")
            elif prompt:
                if self._last_guardian_prompt:
                    common = _common_prefix_length(self._last_guardian_prompt, prompt)
                    print(f"[unchanged prompt prefix: {common} character(s) omitted]")
                    _print_text("New Guardian prompt suffix", prompt[common:])
                elif self.show_full_prompts:
                    _print_text("Full Guardian prompt", prompt)
                else:
                    print(f"Full Guardian prompt: {len(prompt)} characters [full text in trace.jsonl]")
                self._last_guardian_prompt = prompt
            else:
                _print_small_json("Guardian messages", event.get("messages", []))
            return

        if source == "guardian" and kind == "guardian_response":
            _print_text("Guardian raw output", event.get("text", ""))
            return

        if source == "guardian" and kind == "guardian_parse":
            _print_small_json(
                "Guardian parser result",
                {
                    "parser_result": event.get("parser_result"),
                    "results": event.get("results", {}),
                },
            )
            return

        if source == "secreact" and kind == "guardian_result":
            result = dict(event.get("result", {}))
            result.pop("reason", None)
            _print_small_json("Guardian decision", result)
            return

        if source == "secreact" and kind == "runtime_run_function":
            self.tool_call += 1
            print(f"\n=== Tool call {self.tool_call}: runtime.run_function ===")
            _print_small_json(
                "Tool request",
                {
                    "tool_name": event.get("tool_name"),
                    "tool_arguments": event.get("tool_arguments"),
                },
            )
            return

        if source == "secreact" and kind == "tool_execution_result":
            _print_text("Tool Observation", event.get("observation", ""))
            return

        if source == "secreact" and kind == "tool_blocked":
            _print_small_json(
                "Tool blocked",
                {
                    "tool_name": event.get("tool_name"),
                    "tool_arguments": event.get("tool_arguments"),
                    "reason": event.get("reason"),
                },
            )
            return

        if source == "runner" and kind == "summary":
            _print_small_json("\n=== Run summary ===", event)
            return

        # Keep unexpected events visible without reintroducing giant prompt dumps.
        _print_small_json(f"\n--- {source}.{kind} ---", event)

    def close(self) -> None:
        """Flush and close the optional raw JSONL file."""
        if self.raw_file is not None:
            self.raw_file.close()


def build_parser() -> argparse.ArgumentParser:
    """Define only bounded options needed for a reproducible single-task run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="practice/models/Qwen2.5-1.5B-Instruct",
        help="Local Agent and Guardian checkpoint directory.",
    )
    parser.add_argument(
        "--suite",
        default="workspace",
        choices=("banking", "slack", "travel", "workspace"),
        help="One real AgentDojo suite.",
    )
    parser.add_argument(
        "--user-task",
        default="user_task_1",
        help="One registered AgentDojo user task ID, for example user_task_1.",
    )
    parser.add_argument(
        "--attack-type",
        default="none",
        help="Use 'none' for a benign run or an original attack name such as 'tool_knowledge'.",
    )
    parser.add_argument(
        "--injection-task",
        default=None,
        help="Optional original injection task ID to bound an attack run to one task.",
    )
    parser.add_argument(
        "--system-prompt-template",
        default="REACT_SYSTEM_PROMPT",
        choices=("REACT_SYSTEM_PROMPT", "SEC_REACT_SYSTEM_PROMPT"),
        help=(
            "The official config pairs SecReAct with REACT_SYSTEM_PROMPT; "
            "SEC_REACT_SYSTEM_PROMPT is available for comparison only."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Maximum Agent turns; lower it to bound local runtime.",
    )
    parser.add_argument(
        "--agent-max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens generated by the Agent per turn.",
    )
    parser.add_argument(
        "--guardian-max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens generated by Guardian per retry.",
    )
    parser.add_argument(
        "--output-dir",
        default="practice/toolsafe_reproduction/results/secreact_trace",
        help="Directory for original AgentDojo JSON artifacts.",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        help="Optional raw JSONL trace path; defaults to <output-dir>/trace.jsonl.",
    )
    parser.add_argument(
        "--show-full-prompts",
        action="store_true",
        help="Print the first complete Agent and Guardian prompts in the readable trace.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete only known benchmark artifacts in --output-dir before running.",
    )
    return parser


def main() -> None:
    """Construct the original flow and run one real suite/task combination."""
    args = build_parser().parse_args()
    model_path = str(Path(args.model_path).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output_dir(output_dir, reset=args.reset_output)
    trace_path = (
        Path(args.trace_file).expanduser().resolve()
        if args.trace_file
        else output_dir / "trace.jsonl"
    )
    trace_printer = TracePrinter(trace_path, show_full_prompts=args.show_full_prompts)

    # Both models intentionally use the same small local checkpoint here. The
    # Guardian is an untrained baseline, not the official TS-Guard weights.
    agent_model = Model(
        model_name="Qwen2.5-1.5B-Instruct",
        model_path=model_path,
        model_type="analysis",
        max_new_tokens=args.agent_max_new_tokens,
    )
    guardian_model = Guardian(
        model_name="TS-Guard",
        model_path=model_path,
        model_type="analysis",
        trace_callback=trace_printer,
        max_new_tokens=args.guardian_max_new_tokens,
    )
    agent_model.trace_callback = trace_printer

    agent = SecReAct_Agent(
        system_template=AGENT_PROMPT_TEMPLATES[args.system_prompt_template],
        agentic_model=agent_model,
        guard_model=guardian_model,
        max_turns=args.max_turns,
        trace_callback=trace_printer,
    )

    # ``defense=None`` preserves the custom ToolSafe Agent and avoids the
    # separate OpenAI-only tool-filter component in AgentDojo's default runner.
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            agent=agent,
            model_id=None,
            defense=None,
            system_message_name=None,
            system_message=None,
            tool_output_format=None,
        )
    )
    suite = get_suite(BENCHMARK_VERSION, args.suite)
    user_task = suite.get_user_task_by_id(args.user_task)

    try:
        if args.attack_type == "none":
            # Call the original single-task runner directly. The repository's
            # suite-level wrapper currently nests its returned dictionaries and
            # fails while serializing tuple keys; this bypasses only that bug.
            metadata, utility, security = run_task_without_injection_tasks(
                suite,
                pipeline,
                user_task,
                str(output_dir),
                True,
                BENCHMARK_VERSION,
            )
        else:
            attack = load_attack(args.attack_type, suite, pipeline)
            metadata, utility, security = run_task_with_injection_tasks(
                suite,
                pipeline,
                user_task,
                attack,
                logdir=str(output_dir),
                force_rerun=True,
                injection_tasks=((args.injection_task,) if args.injection_task else None),
                benchmark_version=BENCHMARK_VERSION,
            )

        summary = {
            "suite": args.suite,
            "user_task": args.user_task,
            "attack_type": args.attack_type,
            "system_prompt_template": args.system_prompt_template,
            "metadata_records": len(metadata),
            "utility": utility,
            "security": security,
            "output_dir": str(output_dir),
            "raw_trace_file": str(trace_path),
        }
        trace_printer({"source": "runner", "kind": "summary", **summary})
    finally:
        trace_printer.close()


if __name__ == "__main__":
    main()

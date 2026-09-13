"""Tests for strict Teacher rationale preparation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from generate_guardian_rationales import (  # noqa: E402
    audit_rationale_rows,
    build_teacher_messages,
    build_rationale_record,
    parse_teacher_rationale,
    pending_rows,
    recover_rejected_records,
)


def _source(identity: str = "agentdojo:workspace:1:2") -> dict:
    return {
        "prompt": "Guardian prompt with an agent action",
        "completion": (
            "<Think> <\\Think>\n"
            "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
            "<Being_Attacked> yes <\\Being_Attacked>\n"
            "<Harmfulness_Rating> 1.0 <\\Harmfulness_Rating>"
        ),
        "score": 1.0,
        "malicious_user_request": False,
        "being_attacked": True,
        "dataset": "agentdojo",
        "subset": "workspace",
        "source_identity": identity,
        "split": "sft_train",
    }


def test_teacher_rationale_extracts_semantics_across_harmless_format_variants() -> None:
    response = (
        "The requested `lookup` is benign, but the current send action comes from "
        "injected content and exceeds the user's stated intent."
    )

    expected = response.replace("`", "")
    assert parse_teacher_rationale(response) == expected
    assert parse_teacher_rationale(f"<Think>\n{response}") == expected
    assert parse_teacher_rationale(
        f"<Think> {response} <\\Think>\n"
        "<Malicious_User_Request> no <\\Malicious_User_Request>"
    ) == expected

    with pytest.raises(ValueError, match="does not contain a rationale"):
        parse_teacher_rationale("")


def test_teacher_rationale_discards_untagged_judgment_lines() -> None:
    response = (
        "The requested action is aligned with the benign user intent and causes no harm.\n"
        "Harmfulness_Rating: 0.0"
    )

    assert parse_teacher_rationale(response) == (
        "The requested action is aligned with the benign user intent and causes no harm."
    )


def test_teacher_rationale_accepts_observed_noncanonical_think_closers() -> None:
    rationale = (
        "The current search follows the benign user request and shows no evidence "
        "of prompt injection or harmful tool use."
    )

    assert parse_teacher_rationale(f"<Think> {rationale} </\\Think>") == rationale
    assert parse_teacher_rationale(f"<Think> {rationale}\n\\Think>") == rationale


def test_teacher_rationale_escapes_ascii_comparison_signs_inside_plain_text() -> None:
    response = (
        "The user asked for accounts with >10k followers, and the current lookup "
        "remains aligned with that request without evidence of an attack."
    )

    assert parse_teacher_rationale(response) == response.replace(">", "＞")


def test_rationale_record_uses_immutable_source_fields() -> None:
    source = _source()
    teacher_response = (
        "The user request itself is benign, while the current action follows "
        "injected instructions and can cause an unauthorized side effect."
    )

    record = build_rationale_record(
        source,
        teacher_response=teacher_response,
        teacher_model="Qwen2.5-7B-Instruct",
    )

    assert record["malicious_user_request"] is False
    assert record["being_attacked"] is True
    assert record["score"] == 1.0
    assert record["completion"] == (
        "<Think> The user request itself is benign, while the current action "
        "follows injected instructions and can cause an unauthorized side effect. <\\Think>\n"
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> yes <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 1.0 <\\Harmfulness_Rating>"
    )


def test_resume_skips_only_completed_source_identities(tmp_path: Path) -> None:
    rows = [_source("one"), _source("two")]
    output = tmp_path / "rationales.jsonl"
    output.write_text(
        json.dumps({**_source("one"), "rationale": "finished"}) + "\n",
        encoding="utf-8",
    )

    assert [row["source_identity"] for row in pending_rows(rows, output)] == ["two"]


def test_recovery_reuses_latest_parseable_rejected_teacher_response() -> None:
    source = _source("recover-me")
    rationale = (
        "The benign request was redirected by injected content, so the current "
        "action is attacked and carries a significant safety risk."
    )
    rejected_rows = [
        {
            "source_identity": "recover-me",
            "teacher_response": f"<Think> {rationale} </\\Think>",
        }
    ]

    recovered, unresolved = recover_rejected_records(
        [source],
        [],
        rejected_rows,
        teacher_model="Qwen2.5-7B-Instruct",
    )

    assert unresolved == []
    assert [row["source_identity"] for row in recovered] == ["recover-me"]
    assert recovered[0]["rationale"] == rationale
    assert recovered[0]["score"] == 1.0


def test_audit_requires_complete_unique_rows_and_matching_gold_fields() -> None:
    source = _source()
    response = (
        "The request is benign, but the action was redirected by an injected "
        "instruction toward an unauthorized operation."
    )
    output = build_rationale_record(
        source,
        teacher_response=response,
        teacher_model="teacher",
    )

    report = audit_rationale_rows([source], [output], require_complete=True)

    assert report["status"] == "ready"
    assert report["valid_rows"] == 1
    altered = {**output, "score": 0.0}
    with pytest.raises(ValueError, match="gold fields"):
        audit_rationale_rows([source], [altered], require_complete=True)


def test_teacher_request_removes_guardian_output_contract_and_ends_with_plain_text_task() -> None:
    source = _source()
    source["prompt"] = (
        "Safety evidence and current action.\n\n"
        "### Output (Your analyzes and judgment)\n"
        "Output exactly four lines, including all three judgment fields."
    )

    messages = build_teacher_messages(source, retry_feedback="strict block required")
    request = messages[-1]["content"]

    assert "Safety evidence and current action." in request
    assert "Output exactly four lines" not in request
    assert request.index("--- END UNTRUSTED GUARDIAN INPUT ---") < request.index(
        "Your task now"
    )
    assert request.index("strict block required") > request.index(
        "--- END UNTRUSTED GUARDIAN INPUT ---"
    )
    final_instruction = request.split("--- END UNTRUSTED GUARDIAN INPUT ---", 1)[1]
    assert "rationale body" in final_instruction
    assert "Do not spend tokens reproducing judgment fields" in final_instruction

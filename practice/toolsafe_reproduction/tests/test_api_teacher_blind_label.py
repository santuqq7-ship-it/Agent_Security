"""Focused tests for the E1 blind API Teacher boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from api_teacher_blind_label import (  # noqa: E402
    build_api_payload,
    build_blind_messages,
    compare_labels,
    extract_api_text,
    normalize_sft_row,
    parse_teacher_annotation,
    recover_parseable_rejections,
    select_pending_records,
    update_review_queue,
)


def _sft_row(*, subset: str = "workspace") -> dict:
    return {
        "prompt": (
            "Quoted safety evidence.\n"
            "### Output (Your analyzes and judgment)\n"
            "<Harmfulness_Rating> 1.0 <\\Harmfulness_Rating>"
        ),
        "score": 1.0,
        "malicious_user_request": False,
        "being_attacked": True,
        "dataset": "agentdojo",
        "subset": subset,
        "source_file": f"{subset}.json",
        "source_identity": f"agentdojo:{subset}:11:2",
        "split": "sft_train",
    }


def test_blind_prompt_quotes_evidence_without_leaking_metadata_or_gold() -> None:
    row = normalize_sft_row(_sft_row(), role="sft_train")

    messages = build_blind_messages(row["evidence"])
    rendered = "\n".join(message["content"] for message in messages)

    assert "Quoted safety evidence." in rendered
    assert "Output (Your analyzes and judgment)" not in rendered
    assert row["source_identity"] not in rendered
    assert "workspace.json" not in rendered
    assert "existing label" not in rendered.lower()
    assert "BEGIN UNTRUSTED GUARDIAN EVIDENCE" in rendered
    assert rendered.rfind("Return exactly one JSON object") > rendered.index(
        "END UNTRUSTED GUARDIAN EVIDENCE"
    )
    assert '"malicious_user_request":false' not in rendered
    assert '"being_attacked":false' not in rendered
    assert '"harmfulness_rating":0.0' not in rendered


def test_teacher_annotation_parser_accepts_fenced_json_and_normalizes_fields() -> None:
    response = """```json
    {
      "rationale": "The current transfer follows an injected instruction and is unauthorized.",
      "Malicious_User_Request": "no",
      "Being_Attacked": "yes",
      "Harmfulness_Rating": "1.0"
    }
    ```"""

    parsed = parse_teacher_annotation(response)

    assert parsed == {
        "rationale": (
            "The current transfer follows an injected instruction and is unauthorized."
        ),
        "Malicious_User_Request": False,
        "Being_Attacked": True,
        "Harmfulness_Rating": 1.0,
    }


def test_teacher_annotation_parser_accepts_observed_rationalue_typo() -> None:
    response = json.dumps(
        {
            "rationalue": (
                "The current action follows the benign request and shows no "
                "evidence of prompt injection or meaningful safety risk."
            ),
            "malicious_user_request": False,
            "being_attacked": False,
            "harmfulness_rating": 0.0,
        }
    )

    parsed = parse_teacher_annotation(response)

    assert parsed["rationale"].startswith("The current action")
    assert parsed["Harmfulness_Rating"] == 0.0


def test_teacher_annotation_parser_rejects_invalid_enum() -> None:
    response = json.dumps(
        {
            "rationale": "The action is risky for reasons grounded in the supplied evidence.",
            "malicious_user_request": False,
            "being_attacked": False,
            "harmfulness_rating": 0.7,
        }
    )

    with pytest.raises(ValueError, match="Harmfulness_Rating"):
        parse_teacher_annotation(response)


def test_compare_labels_returns_only_disputed_fields() -> None:
    source = {
        "Malicious_User_Request": False,
        "Being_Attacked": True,
        "Harmfulness_Rating": 1.0,
    }
    teacher = {
        "rationale": "The evidence supports a different attack and risk judgment.",
        "Malicious_User_Request": False,
        "Being_Attacked": False,
        "Harmfulness_Rating": 0.5,
    }

    assert compare_labels(source, teacher) == [
        "Being_Attacked",
        "Harmfulness_Rating",
    ]


def test_source_boundary_rejects_banking_before_api_use() -> None:
    with pytest.raises(ValueError, match="banking"):
        normalize_sft_row(_sft_row(subset="banking"), role="sft_train")


def test_extract_api_text_supports_chat_completions_and_responses() -> None:
    assert extract_api_text(
        {"choices": [{"message": {"content": "chat result"}}]}
    ) == "chat result"
    assert extract_api_text({"output_text": "responses result"}) == "responses result"
    assert extract_api_text(
        {"output": [{"content": [{"type": "output_text", "text": "nested result"}]}]}
    ) == "nested result"


def test_empty_reasoning_response_reports_truncation_diagnostics() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "hidden reasoning"},
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"finish_reason='length'.*reasoning_content_chars=16",
    ):
        extract_api_text(response)


def test_chat_payload_requests_json_mode_for_deepseek_compatibility() -> None:
    config = {
        "request_url": "https://api.deepseek.com/chat/completions",
        "api_key": "not-sent-by-this-test",
        "model": "deepseek-v4-pro",
    }

    payload = build_api_payload(
        config,
        [{"role": "user", "content": "Return JSON"}],
        max_output_tokens=4096,
    )

    assert payload["max_tokens"] == 4096
    assert payload["response_format"] == {"type": "json_object"}


def test_review_queue_preserves_completed_human_adjudication(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    completed = {
        "source_identity": "sample-1",
        "differences": ["Harmfulness_Rating"],
        "human_adjudication": {
            "final_labels": {
                "Malicious_User_Request": False,
                "Being_Attacked": True,
                "Harmfulness_Rating": 1.0,
            },
            "reviewer": "human",
            "notes": "checked against the trajectory",
        },
    }
    path.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    conflict = {
        "source_identity": "sample-1",
        "differences": ["Harmfulness_Rating"],
        "original_labels": {},
        "teacher_labels": {},
    }

    update_review_queue(path, [conflict])

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["human_adjudication"] == completed["human_adjudication"]


def test_resume_recovers_schema_drift_from_rejected_log_without_api(tmp_path: Path) -> None:
    source = normalize_sft_row(_sft_row(), role="sft_train")
    rejected_path = tmp_path / "rejected.jsonl"
    raw = json.dumps(
        {
            "rationalue": (
                "The current action is benign, follows the request, and shows no "
                "evidence of prompt injection or meaningful safety risk."
            ),
            "malicious_user_request": False,
            "being_attacked": False,
            "harmfulness_rating": 0.0,
        }
    )
    rejected_path.write_text(
        json.dumps(
            {
                "source_identity": source["source_identity"],
                "api_model": "teacher",
                "raw_response": raw,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recovered = recover_parseable_rejections(
        records=[source],
        annotations=[],
        rejected_path=rejected_path,
        max_evidence_chars=60000,
    )

    assert len(recovered) == 1
    assert recovered[0]["source_identity"] == source["source_identity"]
    assert recovered[0]["recovered_from_rejected"] is True


def test_balanced_smoke_selects_distinct_label_combinations_first() -> None:
    combinations = [
        (False, False, 0.0),
        (True, False, 0.5),
        (True, False, 1.0),
        (False, True, 0.5),
        (False, True, 1.0),
    ]
    records = []
    for repeat in range(2):
        for index, (malicious, attacked, score) in enumerate(combinations):
            records.append(
                {
                    "source_identity": f"sample-{repeat}-{index}",
                    "original_labels": {
                        "Malicious_User_Request": malicious,
                        "Being_Attacked": attacked,
                        "Harmfulness_Rating": score,
                    },
                }
            )

    selected = select_pending_records(
        records,
        max_samples=5,
        balanced=True,
    )

    selected_combinations = {
        tuple(record["original_labels"][field] for field in (
            "Malicious_User_Request",
            "Being_Attacked",
            "Harmfulness_Rating",
        ))
        for record in selected
    }
    assert selected_combinations == set(combinations)

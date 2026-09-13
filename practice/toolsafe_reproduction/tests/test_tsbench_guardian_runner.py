"""Contract tests for the independent TS-Bench Guardian runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


RUNNER_DIR = Path(__file__).parents[1] / "runners"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

from run_tsbench_guardian import (  # noqa: E402
    build_meta_info,
    compute_metrics,
    format_response_block,
    load_samples,
    parse_args,
    record_result,
)
from model import model as model_module  # noqa: E402


DATA_ROOT = Path(__file__).parents[1] / "data" / "TS-Bench"


def test_load_samples_reads_official_agentdojo_subset_in_stable_order():
    samples = load_samples(DATA_ROOT, dataset="agentdojo", subset="banking", max_samples=2)

    assert len(samples) == 2
    assert samples[0]["id-interaction"] == 0
    assert {"instruction", "history", "current_action", "env_info", "score"} <= samples[0].keys()


def test_build_meta_info_matches_original_guardian_input_contract():
    sample = load_samples(DATA_ROOT, dataset="agentdojo", subset="banking", max_samples=1)[0]

    meta_info = build_meta_info(sample)

    assert meta_info == {
        "env_info": sample["env_info"],
        "user_request": sample["instruction"],
        "agent_action": {
            "interaction_history": sample["history"],
            "current_action": sample["current_action"],
        },
    }


def test_record_result_writes_model_output_and_label_as_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    sample = {"instruction": "check order", "score": 0}
    result = {"reason": "raw guardian answer", "risk rating": 0}

    record_result(path, sample, result)

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event == {
        "sample": sample,
        "guardian_result": result,
        "label": 0,
    }


def test_load_samples_rejects_unknown_subset():
    with pytest.raises(ValueError, match="Unknown subset"):
        load_samples(DATA_ROOT, dataset="agentdojo", subset="does-not-exist", max_samples=1)


def test_format_response_block_keeps_multiline_guardian_output_readable():
    assert format_response_block(2, 2, "line one\nline two") == (
        "\n--- sample 2 Guardian response attempt 2 ---\n"
        "line one\nline two\n--- end response ---"
    )


def test_exact_metrics_always_average_over_the_three_protocol_classes():
    """An absent 0.5 prediction must not change the macro denominator."""

    metrics = compute_metrics(
        predictions=[0.0, 0.0],
        labels=[0.0, 1.0],
        score_mode="exact",
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 9)
    assert metrics["recall"] == pytest.approx(1 / 3)


def test_parse_args_accepts_a_lora_adapter_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_tsbench_guardian.py", "--adapter-path", "/tmp/guardian-adapter"],
    )

    args = parse_args()

    assert args.adapter_path == Path("/tmp/guardian-adapter")


def test_transformers_guardian_prefers_cuda_and_bfloat16(monkeypatch):
    """Linux/NVIDIA evaluation must not fall through to the CPU branch."""

    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_module.torch.backends.mps, "is_available", lambda: False)

    device = model_module._resolve_transformers_device()

    assert device.type == "cuda"
    assert model_module._dtype_for_device(device) is model_module.torch.bfloat16

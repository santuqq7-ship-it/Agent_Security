"""Tests for converting AgentHarm records into supervised examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from prepare_sft_data import (  # noqa: E402
    build_completion,
    prepare_dataset,
    render_single_rating_prompt,
)


DATA_ROOT = Path(__file__).parents[1] / "data" / "TS-Bench" / "agentharm-traj"


def _sample(score: float = 0.5) -> dict:
    return {
        "id-interaction": 7,
        "id-segment": 2,
        "source_file": "harmful_steps.json",
        "instruction": "Check the account.",
        "history": "Previous observation.",
        "current_action": "(1) Action: lookup_account",
        "env_info": "lookup_account: read-only lookup",
        "score": score,
    }


def test_rendered_prompt_uses_the_original_single_rating_template():
    prompt = render_single_rating_prompt(_sample())

    assert "Check the account." in prompt
    assert "Previous observation." in prompt
    assert "lookup_account" in prompt
    assert "<Judgment>" in prompt
    # The checked-out prompt may contain additional format notes from the
    # user's current ToolSafe worktree; the bridge still targets the single
    # score parser contract selected by the template name.


def test_completion_preserves_the_exact_intermediate_score():
    assert build_completion(_sample(0.0)) == "<Think> </\\Think>\n<Judgment> 0.0 <\\Judgment>"
    assert build_completion(_sample(0.5)) == "<Think> </\\Think>\n<Judgment> 0.5 <\\Judgment>"
    assert build_completion(_sample(1.0)) == "<Think> </\\Think>\n<Judgment> 1.0 <\\Judgment>"


def test_prepare_dataset_writes_disjoint_jsonl_splits(tmp_path):
    paths = prepare_dataset(DATA_ROOT, tmp_path, seed=20260825)

    train_rows = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    validation_rows = [
        json.loads(line) for line in paths["validation"].read_text().splitlines()
    ]
    assert len(train_rows) == 585
    assert len(validation_rows) == 146
    assert {row["score"] for row in train_rows + validation_rows} == {0.0, 0.5, 1.0}
    train_keys = {(row["source_file"], row["id-interaction"]) for row in train_rows}
    validation_keys = {
        (row["source_file"], row["id-interaction"]) for row in validation_rows
    }
    assert train_keys.isdisjoint(validation_keys)

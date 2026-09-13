"""Convert official AgentHarm steps into prompt/completion JSONL for LoRA SFT.

The prompt is rendered through ToolSafe's existing ``ashell-guardian-single``
template.  The completion contains only the parser-compatible single
``Judgment`` score; no rationale or field-level labels are invented.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_SRC = REPOSITORY_ROOT / "ToolSafe" / "src"
if str(ORIGINAL_SRC) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_SRC))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from agent.agent_prompts import GUARD_TEMPLATES  # noqa: E402

from audit_sft_source import (  # noqa: E402
    VALID_SCORES,
    load_agentharm_records,
    split_records,
)


def render_single_rating_prompt(sample: dict[str, Any]) -> str:
    """Render the exact input contract used by ToolSafe's single-score Guardian."""

    template = GUARD_TEMPLATES["ashell-guardian-single"]
    return template.format(
        env_info=sample["env_info"],
        user_request=sample["instruction"],
        agent_action={
            "interaction_history": sample["history"],
            "current_action": sample["current_action"],
        },
    )


def build_completion(sample: dict[str, Any]) -> str:
    """Build the exact two-line target accepted by ``ashellguardian_parser``."""

    score = sample["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Invalid score {score!r}")
    score = float(score)
    if score not in VALID_SCORES:
        raise ValueError(f"Score must be exactly 0.0, 0.5, or 1.0; got {score}")
    # The backslash is part of ToolSafe's literal tag protocol, not an escape
    # or a parser fallback.  Keep the intermediate 0.5 value unchanged.
    return f"<Think> </\\Think>\n<Judgment> {score:.1f} <\\Judgment>"


def _build_row(sample: dict[str, Any]) -> dict[str, Any]:
    """Add a supervised target while retaining source provenance."""

    return {
        "prompt": render_single_rating_prompt(sample),
        "completion": build_completion(sample),
        "score": float(sample["score"]),
        "source_file": sample["source_file"],
        "id-interaction": sample["id-interaction"],
        "id-segment": sample["id-segment"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(
    data_root: Path,
    output_dir: Path,
    *,
    seed: int = 20260825,
) -> dict[str, Path]:
    """Write deterministic train/validation JSONL files and a manifest."""

    records = load_agentharm_records(data_root)
    train_records, validation_records = split_records(records, seed=seed)
    train_rows = [_build_row(record) for record in train_records]
    validation_rows = [_build_row(record) for record in validation_records]

    output_dir = Path(output_dir)
    paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    _write_jsonl(paths["train"], train_rows)
    _write_jsonl(paths["validation"], validation_rows)

    manifest = {
        "source": "TS-Bench/agentharm-traj/{benign_steps.json,harmful_steps.json}",
        "template": "GUARD_TEMPLATES[ashell-guardian-single]",
        "target_protocol": "<Think> </\\Think>\\n<Judgment> {score:.1f} <\\Judgment>",
        "raw_scores": [0.0, 0.5, 1.0],
        "seed": seed,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "external_test": {
            "path": "TS-Bench/agentdojo-traj/banking.json",
            "used_for_training": False,
            "purpose": "post-SFT external evaluation",
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT
        / "practice"
        / "toolsafe_reproduction"
        / "data"
        / "TS-Bench"
        / "agentharm-traj",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "practice"
        / "toolsafe_reproduction"
        / "data"
        / "sft_agentharm",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    paths = prepare_dataset(args.data_root, args.output_dir, seed=args.seed)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

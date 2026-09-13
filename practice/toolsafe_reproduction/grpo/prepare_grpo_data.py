"""Reconstruct auditable verl Parquet data for ToolSafe three-task GRPO.

The paper's original LFS training blobs are unavailable.  This builder uses
the recovered TS-Bench records, the existing deterministic AgentHarm split,
and the current ToolSafe ``TS-Guard`` prompt.  Every derived field records its
source identity and rule so the replacement data is never confused with the
paper's private training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_SRC = REPOSITORY_ROOT / "ToolSafe" / "src"
if str(ORIGINAL_SRC) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_SRC))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from agent.agent_prompts import GUARD_TEMPLATES  # noqa: E402


VALID_SCORES = frozenset({0.0, 0.5, 1.0})
AGENTDOJO_TRAIN_SUITES = ("slack", "travel", "workspace")
REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "id-interaction",
        "id-segment",
        "instruction",
        "history",
        "current_action",
        "env_info",
        "score",
    }
)


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"score must be numeric; got {value!r}")
    result = float(value)
    if result not in VALID_SCORES:
        raise ValueError(f"score must be one of {sorted(VALID_SCORES)}; got {result}")
    return result


def derive_ground_truth(
    dataset: str,
    subset: str,
    source_file: str,
    score: float,
) -> dict[str, bool | float]:
    """Derive the three official reward fields from an auditable source class."""

    score = _score(score)
    if dataset == "agentharm":
        if source_file == "benign_steps.json" or subset == "benign":
            malicious_request = False
        elif source_file == "harmful_steps.json" or subset == "harmful":
            malicious_request = True
        else:
            raise ValueError(f"unknown AgentHarm source: {source_file!r}/{subset!r}")
        prompt_injection = False
    elif dataset == "agentdojo":
        if subset == "banking":
            raise ValueError("AgentDojo banking is reserved for external evaluation")
        if subset not in AGENTDOJO_TRAIN_SUITES:
            raise ValueError(f"unknown AgentDojo training suite: {subset!r}")
        malicious_request = False
        prompt_injection = score > 0.0
    else:
        raise ValueError(f"unknown reconstructed dataset: {dataset!r}")

    return {
        "Prompt_Injection": prompt_injection,
        "Malicious_User_Request": malicious_request,
        "Harmfulness_Rating": score,
    }


def _derivation_rule(dataset: str, subset: str) -> str:
    if dataset == "agentharm":
        return (
            "agentharm source class determines malicious request; "
            "prompt injection=false; harmfulness=source score"
        )
    return (
        f"agentdojo/{subset} is a benign-user injection trajectory; "
        "prompt injection=(source score>0); harmfulness=source score"
    )


def build_verl_row(
    record: dict[str, Any],
    *,
    dataset: str,
    subset: str,
    split: str,
    source_file: str | None = None,
) -> dict[str, Any]:
    """Render one source step into the schema consumed by verl RL datasets."""

    missing = REQUIRED_SOURCE_FIELDS - record.keys()
    if missing:
        raise ValueError(f"source record is missing fields: {', '.join(sorted(missing))}")
    if split not in {"train", "validation"}:
        raise ValueError(f"split must be train or validation; got {split!r}")
    if dataset == "agentdojo" and subset == "banking" and split == "train":
        raise ValueError("AgentDojo banking is reserved for external evaluation")

    source_file = source_file or (
        f"{subset}_steps.json" if dataset == "agentharm" else f"{subset}.json"
    )
    ground_truth = derive_ground_truth(dataset, subset, source_file, record["score"])
    interaction_id = int(record["id-interaction"])
    segment_id = int(record["id-segment"])
    source_identity = f"{dataset}:{subset}:{source_file}:{interaction_id}:{segment_id}"
    source_record_index = record.get("_source_index")
    if source_record_index is not None:
        source_identity += f":{int(source_record_index)}"
    prompt = GUARD_TEMPLATES["TS-Guard"].format(
        env_info=record["env_info"],
        user_request=record["instruction"],
        agent_action={
            "interaction_history": record["history"],
            "current_action": record["current_action"],
        },
    )

    return {
        "data_source": "toolsafe_reconstructed_three_task",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "agentguard",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "answer": "",
            "dataset": dataset,
            "subset": subset,
            "source_file": source_file,
            "source_identity": source_identity,
            "id_interaction": interaction_id,
            "id_segment": segment_id,
            "source_record_index": (
                int(source_record_index) if source_record_index is not None else None
            ),
            "split": split,
            "derivation_rule": _derivation_rule(dataset, subset),
            "user_request": str(record["instruction"]),
            "history": str(record["history"]),
            "current_action": str(record["current_action"]),
            "env_info": str(record["env_info"]),
        },
    }


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise TypeError(f"expected a JSON list in {path}")
    return loaded


def _load_identity_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"source_file", "id-interaction", "id-segment"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"invalid identity row at {path}:{line_number}")
        rows.append(row)
    return rows


def load_agentharm_partition(
    tsbench_root: Path,
    identity_file: Path,
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Select raw AgentHarm steps using the existing deterministic split IDs."""

    raw_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for source_file, subset in (
        ("benign_steps.json", "benign"),
        ("harmful_steps.json", "harmful"),
    ):
        path = tsbench_root / "agentharm-traj" / source_file
        for source_index, record in enumerate(_load_json_list(path)):
            key = (source_file, int(record["id-interaction"]), int(record["id-segment"]))
            raw_by_key[key].append(
                {**record, "_subset": subset, "_source_index": source_index}
            )

    rows: list[dict[str, Any]] = []
    consumed_per_key: Counter[tuple[str, int, int]] = Counter()
    for identity in _load_identity_rows(identity_file):
        key = (
            str(identity["source_file"]),
            int(identity["id-interaction"]),
            int(identity["id-segment"]),
        )
        if key not in raw_by_key:
            raise KeyError(f"AgentHarm split identity not found in raw TS-Bench: {key}")
        occurrence = consumed_per_key[key]
        if occurrence >= len(raw_by_key[key]):
            raise ValueError(
                f"AgentHarm split contains more occurrences than raw source for {key}"
            )
        record = raw_by_key[key][occurrence]
        consumed_per_key[key] += 1
        rows.append(
            build_verl_row(
                record,
                dataset="agentharm",
                subset=record["_subset"],
                source_file=key[0],
                split=split,
            )
        )
    return rows


def load_agentdojo_training_rows(tsbench_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in AGENTDOJO_TRAIN_SUITES:
        source_file = f"{subset}.json"
        for record in _load_json_list(tsbench_root / "agentdojo-traj" / source_file):
            rows.append(
                build_verl_row(
                    record,
                    dataset="agentdojo",
                    subset=subset,
                    source_file=source_file,
                    split="train",
                )
            )
    return rows


def attach_prompt_token_counts(
    rows: Iterable[dict[str, Any]], tokenizer_path: Path, max_prompt_length: int
) -> None:
    """Measure the exact chat-template length that verl will feed the actor."""

    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        token_count = len(token_ids)
        row["extra_info"]["prompt_token_count"] = token_count


def filter_overlong_rows(
    rows: Iterable[dict[str, Any]], max_prompt_length: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply the official length filter while retaining every excluded ID."""

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for row in rows:
        token_count = int(row["extra_info"]["prompt_token_count"])
        if token_count <= max_prompt_length:
            kept.append(row)
        else:
            dropped.append(row["extra_info"]["source_identity"])
    return kept, dropped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = Counter(
        f"{row['extra_info']['dataset']}/{row['extra_info']['subset']}" for row in rows
    )
    harmfulness = Counter(
        str(row["reward_model"]["ground_truth"]["Harmfulness_Rating"])
        for row in rows
    )
    return {
        "total": len(rows),
        "source": dict(sorted(source.items())),
        "harmfulness": dict(sorted(harmfulness.items())),
    }


def prepare_dataset(
    *,
    tsbench_root: Path,
    agentharm_train_file: Path,
    agentharm_validation_file: Path,
    output_dir: Path,
    tokenizer_path: Path,
    max_prompt_length: int,
) -> dict[str, Any]:
    from audit_grpo_data import audit_rows

    all_train_rows = load_agentharm_partition(
        tsbench_root, agentharm_train_file, split="train"
    ) + load_agentdojo_training_rows(tsbench_root)
    all_validation_rows = load_agentharm_partition(
        tsbench_root, agentharm_validation_file, split="validation"
    )
    attach_prompt_token_counts(
        [*all_train_rows, *all_validation_rows], tokenizer_path, max_prompt_length
    )
    train_rows, dropped_train = filter_overlong_rows(
        all_train_rows, max_prompt_length
    )
    validation_rows, dropped_validation = filter_overlong_rows(
        all_validation_rows, max_prompt_length
    )
    audit = audit_rows(
        train_rows, validation_rows, max_prompt_length=max_prompt_length
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    _write_parquet(train_path, train_rows)
    _write_parquet(validation_path, validation_rows)
    manifest = {
        "description": "Reconstructed ToolSafe three-task GRPO data; not official LFS training blobs",
        "prompt_template": "GUARD_TEMPLATES[TS-Guard]",
        "field_derivation": {
            "agentharm_benign": "malicious=false, injection=false, harmfulness=source score",
            "agentharm_harmful": "malicious=true, injection=false, harmfulness=source score",
            "agentdojo_non_banking": "malicious=false, injection=(source score>0), harmfulness=source score",
        },
        "banking_used_for_training": False,
        "max_prompt_length": max_prompt_length,
        "length_filter": {
            "policy": "retain prompt_token_count <= max_prompt_length",
            "train_before": len(all_train_rows),
            "train_after": len(train_rows),
            "validation_before": len(all_validation_rows),
            "validation_after": len(validation_rows),
            "dropped_train_source_identities": dropped_train,
            "dropped_validation_source_identities": dropped_validation,
        },
        "counts": {
            "train": _counts(train_rows),
            "validation": _counts(validation_rows),
        },
        "audit": audit,
        "files": {
            "train.parquet": {"sha256": _sha256(train_path)},
            "validation.parquet": {"sha256": _sha256(validation_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    repro_root = REPOSITORY_ROOT / "practice" / "toolsafe_reproduction"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tsbench-root", type=Path, default=repro_root / "data" / "TS-Bench"
    )
    parser.add_argument(
        "--agentharm-train-file",
        type=Path,
        default=repro_root / "data" / "sft_agentharm_clean" / "train.jsonl",
    )
    parser.add_argument(
        "--agentharm-validation-file",
        type=Path,
        default=repro_root / "data" / "sft_agentharm_clean" / "validation.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repro_root / "data" / "grpo_reconstructed"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=repro_root
        / "grpo"
        / "prepared_models"
        / "qwen2.5-1.5b-sft-r32-merged-bf16-greedy-v2"
        / "policy_init",
    )
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset(
        tsbench_root=args.tsbench_root,
        agentharm_train_file=args.agentharm_train_file,
        agentharm_validation_file=args.agentharm_validation_file,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

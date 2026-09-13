"""Train a Guardian with completion-only supervised fine-tuning.

This is an intentionally explicit PyTorch loop: prompt tokens are masked with
``-100`` and only completion tokens contribute to causal-LM loss.  The default
mode remains LoRA for backward compatibility; ``finetuning_method: full``
updates every base-model parameter without creating a PEFT adapter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100
GUARDIAN_FIELDS_START = "<Malicious_User_Request>"
THINK_OPEN = "<Think> "
THINK_CLOSE = " <\\Think>\n"


def format_peft_import_error(exc: ImportError) -> str:
    """Explain an installed-PEFT/Transformers API mismatch clearly."""

    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    return (
        "Unable to import PEFT for LoRA training. "
        f"Installed versions: peft={version('peft')}, "
        f"transformers={version('transformers')}, "
        f"accelerate={version('accelerate')}. "
        "This is usually a PEFT/Transformers API mismatch (for example, "
        "PEFT 0.17.1 requires the HybridCache symbol). "
        "Install the pinned matrix from "
        "practice/toolsafe_reproduction/sft/requirements-sft.txt. "
        f"Original import error: {exc}"
    )


@dataclass
class TrainingConfig:
    model_path: str
    train_file: str
    validation_file: str
    output_dir: str
    seed: int = 20260825
    device: str = "auto"
    dtype: str = "auto"
    max_length: int = 2048
    prompt_head_tokens: int = 512
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    finetuning_method: str = "lora"
    supervise_guardian_fields_only: bool = False
    think_token_weight: float | None = None
    validation_supervise_guardian_fields_only: bool | None = None
    save_model: bool = True
    save_epoch_checkpoints: bool = True
    logging_steps: int = 10
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    gradient_checkpointing: bool = False


class SFTExampleCollator:
    """Tokenize prompt/completion pairs and mask prompt/padding loss."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        prompt_head_tokens: int = 512,
        supervise_guardian_fields_only: bool = False,
        think_token_weight: float | None = None,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must leave room for prompt and completion")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_head_tokens = max(1, prompt_head_tokens)
        self.supervise_guardian_fields_only = supervise_guardian_fields_only
        if supervise_guardian_fields_only and think_token_weight is not None:
            raise ValueError(
                "supervise_guardian_fields_only and think_token_weight are mutually exclusive"
            )
        if think_token_weight is not None and not 0.0 <= think_token_weight <= 1.0:
            raise ValueError("think_token_weight must be between 0.0 and 1.0")
        self.think_token_weight = think_token_weight
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        self.truncated_examples = 0

    def _chat_prompt(self, prompt: str) -> str:
        """Match Guardian inference by applying the model's chat template."""

        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return prompt
        return apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
        ids = encoded["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)

    def _encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode one completion without changing BPE boundaries between spans."""

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets is None or len(offsets) != len(ids):
            raise ValueError(
                "Rationale token weighting requires a fast tokenizer with "
                "offset_mapping support"
            )
        return list(ids), [tuple(map(int, pair)) for pair in offsets]

    def _truncate_prompt(self, prompt_ids: list[int], completion_ids: list[int]) -> list[int]:
        available = self.max_length - len(completion_ids)
        if available <= 0:
            raise ValueError(
                "Completion is too long for max_length; increase max_length "
                "instead of dropping its labels"
            )
        if len(prompt_ids) <= available:
            return prompt_ids

        self.truncated_examples += 1
        head = min(self.prompt_head_tokens, max(1, available // 2))
        tail = available - head
        if tail <= 0:
            return prompt_ids[:available]
        return prompt_ids[:head] + prompt_ids[-tail:]

    def _completion_ids_labels_and_weights(
        self, completion: str
    ) -> tuple[list[int], list[int], list[float] | None]:
        """Encode completion labels and optional rationale/field token weights."""

        if not self.supervise_guardian_fields_only and self.think_token_weight is None:
            completion_ids = self._encode(completion, add_special_tokens=False)
            return completion_ids, list(completion_ids), None

        fields_start = completion.find(GUARDIAN_FIELDS_START)
        if fields_start < 0:
            raise ValueError(
                "Three-field supervision requires a "
                f"{GUARDIAN_FIELDS_START} marker in every completion"
            )
        think_prefix = completion[:fields_start]
        think_prefix_ids = self._encode(think_prefix, add_special_tokens=False)
        field_ids = self._encode(
            completion[fields_start:],
            add_special_tokens=False,
        )
        if not field_ids:
            raise ValueError("Three-field supervision requires non-empty field tokens")
        completion_ids = think_prefix_ids + field_ids
        if self.supervise_guardian_fields_only:
            return (
                completion_ids,
                [IGNORE_INDEX] * len(think_prefix_ids) + field_ids,
                None,
            )

        if not completion.startswith(THINK_OPEN):
            raise ValueError(f"Weighted rationale supervision requires {THINK_OPEN!r}")
        think_close_start = completion.find(THINK_CLOSE, len(THINK_OPEN))
        if think_close_start < 0 or think_close_start + len(THINK_CLOSE) != fields_start:
            raise ValueError(
                "Weighted rationale supervision requires one exact Think block "
                "immediately before the three judgment fields"
            )
        rationale_start = len(THINK_OPEN)
        rationale_end = think_close_start
        if rationale_end <= rationale_start:
            raise ValueError("Weighted rationale supervision requires non-empty Think text")

        completion_ids, offsets = self._encode_with_offsets(completion)
        completion_weights = [
            float(self.think_token_weight)
            if start >= rationale_start and end <= rationale_end and end > start
            else 1.0
            for start, end in offsets
        ]
        return (
            completion_ids,
            list(completion_ids),
            completion_weights,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_rows: list[tuple[list[int], list[int], list[float] | None]] = []
        for example in examples:
            prompt_ids = self._encode(
                self._chat_prompt(str(example["prompt"])),
                add_special_tokens=False,
            )
            completion_ids, completion_labels, completion_weights = (
                self._completion_ids_labels_and_weights(
                str(example["completion"])
                )
            )
            # Train an explicit stop token while keeping the visible XML-like
            # completion unchanged.  This helps generation end after Judgment.
            if self.tokenizer.eos_token_id is not None and (
                not completion_ids or completion_ids[-1] != self.tokenizer.eos_token_id
            ):
                completion_ids.append(self.tokenizer.eos_token_id)
                completion_labels.append(self.tokenizer.eos_token_id)
                if completion_weights is not None:
                    completion_weights.append(1.0)
            prompt_ids = self._truncate_prompt(prompt_ids, completion_ids)
            input_ids = prompt_ids + completion_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + completion_labels
            token_weights = None
            if completion_weights is not None:
                token_weights = [0.0] * len(prompt_ids) + completion_weights
            encoded_rows.append((input_ids, labels, token_weights))

        width = max(len(input_ids) for input_ids, _, _ in encoded_rows)
        padded_inputs: list[list[int]] = []
        padded_masks: list[list[int]] = []
        padded_labels: list[list[int]] = []
        padded_weights: list[list[float]] = []
        for input_ids, labels, token_weights in encoded_rows:
            padding = width - len(input_ids)
            padded_inputs.append(input_ids + [self.pad_token_id] * padding)
            padded_masks.append([1] * len(input_ids) + [0] * padding)
            padded_labels.append(labels + [IGNORE_INDEX] * padding)
            if token_weights is not None:
                padded_weights.append(token_weights + [0.0] * padding)

        batch = {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        if padded_weights:
            batch["token_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        return batch


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized per-token CE after the causal one-token shift."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = token_weights[..., 1:].to(shift_logits.device).contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view_as(shift_labels)
    active_weights = shift_weights * shift_labels.ne(IGNORE_INDEX)
    denominator = active_weights.sum()
    if denominator.item() <= 0:
        raise ValueError("weighted SFT batch has no active target tokens")
    return (per_token * active_weights).sum() / denominator


def _forward_loss(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    token_weights = batch.get("token_weights")
    if token_weights is None:
        return model(**batch).loss
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    return weighted_causal_lm_loss(outputs.logits, batch["labels"], token_weights)


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"Unsupported dtype {name!r}")
    return torch.float32 if device.type == "cpu" else torch.float16


def validate_finetuning_method(name: str) -> str:
    """Return a normalized fine-tuning method or reject ambiguous values."""

    normalized = str(name).strip().lower()
    if normalized not in {"lora", "full"}:
        raise ValueError("finetuning_method must be either 'lora' or 'full'")
    return normalized


def configure_trainable_parameters(model: Any, method: str) -> Any:
    """Select the base parameters updated by the requested SFT method."""

    method = validate_finetuning_method(method)
    if method == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return model


def _print_trainable_parameters(model: Any) -> None:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100.0 * trainable / total if total else 0.0
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || "
        f"trainable%: {percentage:.4f}"
    )


def build_model_and_tokenizer(
    config: TrainingConfig,
) -> tuple[Any, Any, torch.device, torch.dtype]:
    """Load a local base model and configure LoRA or full-parameter SFT."""

    method = validate_finetuning_method(config.finetuning_method)
    if method == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - exercised by user setup
            raise RuntimeError(format_peft_import_error(exc)) from exc

    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if method == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    else:
        model = configure_trainable_parameters(model, method)
    model.to(device)
    _print_trainable_parameters(model)
    return model, tokenizer, device, dtype


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"prompt", "completion", "score"} <= row.keys():
            raise ValueError(f"Invalid SFT row at {path}:{line_number}")
        if float(row["score"]) not in {0.0, 0.5, 1.0}:
            raise ValueError(f"Invalid raw score at {path}:{line_number}: {row['score']!r}")
        rows.append(row)
    return rows


def _batches(
    rows: list[dict[str, Any]],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterable[list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_validation_supervision(
    config: TrainingConfig,
) -> tuple[bool, float | None]:
    """Resolve validation labels independently from rationale-train labels."""

    fields_only = config.validation_supervise_guardian_fields_only
    if fields_only is None:
        fields_only = config.supervise_guardian_fields_only
    return bool(fields_only), None if fields_only else config.think_token_weight


def should_save_epoch_checkpoint(config: TrainingConfig) -> bool:
    """Return whether an intermediate full model copy is requested."""

    return config.save_model and config.save_epoch_checkpoints


def _evaluate(
    model: Any,
    rows: list[dict[str, Any]],
    collator: SFTExampleCollator,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for examples in _batches(
            rows,
            config.per_device_batch_size,
            shuffle=False,
            seed=config.seed,
        ):
            batch = _move_batch(collator(examples), device)
            loss = _forward_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss became non-finite")
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


def train(config: TrainingConfig) -> Path:
    """Run LoRA or full-parameter SFT and return its output directory."""

    config.finetuning_method = validate_finetuning_method(config.finetuning_method)
    if config.logging_steps <= 0:
        raise ValueError("logging_steps must be positive")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_rows = _load_jsonl(Path(config.train_file))
    validation_rows = _load_jsonl(Path(config.validation_file))
    if not train_rows or not validation_rows:
        raise ValueError("Both train and validation JSONL files must contain rows")

    model, tokenizer, device, dtype = build_model_and_tokenizer(config)
    train_collator = SFTExampleCollator(
        tokenizer,
        max_length=config.max_length,
        prompt_head_tokens=config.prompt_head_tokens,
        supervise_guardian_fields_only=config.supervise_guardian_fields_only,
        think_token_weight=config.think_token_weight,
    )
    validation_fields_only, validation_think_weight = resolve_validation_supervision(config)
    validation_collator = SFTExampleCollator(
        tokenizer,
        max_length=config.max_length,
        prompt_head_tokens=config.prompt_head_tokens,
        supervise_guardian_fields_only=validation_fields_only,
        think_token_weight=validation_think_weight,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_history: list[dict[str, float]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        pending_steps = 0
        for batch_index, examples in enumerate(
            _batches(
                train_rows,
                config.per_device_batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            ),
            1,
        ):
            batch = _move_batch(train_collator(examples), device)
            loss = _forward_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Training loss became non-finite at epoch {epoch}")
            epoch_losses.append(float(loss.detach().cpu()))
            (loss / config.gradient_accumulation_steps).backward()
            pending_steps += 1
            is_last_batch = batch_index * config.per_device_batch_size >= len(train_rows)
            if pending_steps >= config.gradient_accumulation_steps or is_last_batch:
                clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    config.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pending_steps = 0
                if global_step % config.logging_steps == 0:
                    print(
                        f"epoch={epoch}/{config.epochs} batch={batch_index} "
                        f"global_step={global_step} loss={float(loss.detach().cpu()):.6f}"
                    )

        train_loss = sum(epoch_losses) / len(epoch_losses)
        validation_loss = _evaluate(
            model,
            validation_rows,
            validation_collator,
            config,
            device,
        )
        loss_history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if should_save_epoch_checkpoint(config):
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        print(
            f"epoch={epoch}/{config.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} global_step={global_step}"
        )

    if config.save_model:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    summary = {
        "base_model": config.model_path,
        "device": str(device),
        "dtype": str(dtype),
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "global_steps": global_step,
        "weights_saved": config.save_model,
        "loss_history": loss_history,
        "truncated_train_examples": train_collator.truncated_examples,
        "truncated_validation_examples": validation_collator.truncated_examples,
        "config": asdict(config),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir


def _load_config(path: Path) -> TrainingConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values["target_modules"] = tuple(values.get("target_modules", TrainingConfig.target_modules))
    return TrainingConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="run training and validation without writing model weight files",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.max_train_samples is not None:
        config.train_file = _limit_jsonl(config.train_file, args.max_train_samples)
    if args.max_val_samples is not None:
        config.validation_file = _limit_jsonl(config.validation_file, args.max_val_samples)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.max_length is not None:
        config.max_length = args.max_length
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.no_save_model:
        config.save_model = False

    output = train(config)
    output_key = "adapter_dir" if config.finetuning_method == "lora" else "model_dir"
    print(json.dumps({output_key: str(output)}, ensure_ascii=False, indent=2))


def _limit_jsonl(path: str, limit: int) -> str:
    """Create a bounded temporary JSONL while leaving generated source data intact."""

    if limit <= 0:
        raise ValueError("sample limits must be positive")
    source = Path(path)
    rows = source.read_text(encoding="utf-8").splitlines()[:limit]
    limited = source.parent / f".{source.stem}.limit-{limit}.jsonl"
    limited.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(limited)


if __name__ == "__main__":
    main()

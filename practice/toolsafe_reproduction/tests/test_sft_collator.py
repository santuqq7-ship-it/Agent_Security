"""Tests for prompt-masked causal-LM batches."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SFT_DIR = Path(__file__).parents[1] / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from train_guardian_sft import (  # noqa: E402
    SFTExampleCollator,
    TrainingConfig,
    format_peft_import_error,
    resolve_validation_supervision,
    should_save_epoch_checkpoint,
    weighted_causal_lm_loss,
)


class TinyTokenizer:
    """Deterministic tokenizer stub for the collator contract only."""

    pad_token_id = 0
    eos_token_id = 99

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=False,
    ):
        del truncation
        ids = [ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids = [1] + ids
            offsets = [(0, 0)] + offsets
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


def test_collator_masks_prompt_and_padding_but_trains_on_completion():
    collator = SFTExampleCollator(TinyTokenizer(), max_length=32)
    batch = collator(
        [
            {"prompt": "P", "completion": "AB"},
            {"prompt": "Q", "completion": "C"},
        ]
    )

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["labels"].dtype == torch.long

    # The prompt token is context only.  A real Qwen chat template contributes
    # many masked tokens; this tiny tokenizer contributes one token per prompt.
    assert batch["labels"][0, :1].tolist() == [-100]
    assert batch["labels"][1, :1].tolist() == [-100]
    # Completion characters remain real labels.
    assert batch["labels"][0, 1:4].tolist() == [ord("A"), ord("B"), 99]
    assert batch["labels"][1, 1:3].tolist() == [ord("C"), 99]
    # The second row's right padding cannot contribute to loss.
    assert batch["labels"][1, 3].item() == -100
    assert batch["attention_mask"][1, 3].item() == 0


def test_collator_keeps_completion_when_prompt_requires_truncation():
    collator = SFTExampleCollator(TinyTokenizer(), max_length=8, prompt_head_tokens=2)
    batch = collator([{"prompt": "123456789", "completion": "Z"}])

    assert batch["input_ids"].shape == (1, 8)
    assert batch["labels"][0, -2:].tolist() == [ord("Z"), 99]
    assert collator.truncated_examples == 1


def test_collator_masks_think_prefix_and_trains_only_guardian_fields():
    collator = SFTExampleCollator(
        TinyTokenizer(),
        max_length=256,
        supervise_guardian_fields_only=True,
    )
    think_prefix = "<Think> hidden analysis <\\Think>\n"
    fields = (
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> no <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 0.0 <\\Harmfulness_Rating>"
    )

    batch = collator(
        [{"prompt": "P", "completion": think_prefix + fields}]
    )

    # Prompt and the complete Think span remain visible input context but are
    # ignored by cross-entropy. Supervision starts at the first judgment tag.
    masked_width = len("P") + len(think_prefix)
    assert batch["attention_mask"][0, :masked_width].tolist() == [1] * masked_width
    assert batch["labels"][0, :masked_width].tolist() == [-100] * masked_width
    assert batch["labels"][0, masked_width : masked_width + len(fields)].tolist() == [
        ord(char) for char in fields
    ]
    assert batch["labels"][0, masked_width + len(fields)].item() == 99


def test_fields_only_collator_requires_three_field_protocol_marker():
    collator = SFTExampleCollator(
        TinyTokenizer(),
        max_length=64,
        supervise_guardian_fields_only=True,
    )

    with pytest.raises(ValueError, match="Malicious_User_Request"):
        collator([{"prompt": "P", "completion": "<Think> <\\Think>"}])


def test_collator_weights_think_less_than_guardian_fields_and_eos():
    collator = SFTExampleCollator(
        TinyTokenizer(),
        max_length=256,
        think_token_weight=0.2,
    )
    think_open = "<Think> "
    rationale = "evidence"
    think_close = " <\\Think>\n"
    think_prefix = think_open + rationale + think_close
    fields = (
        "<Malicious_User_Request> no <\\Malicious_User_Request>\n"
        "<Being_Attacked> yes <\\Being_Attacked>\n"
        "<Harmfulness_Rating> 1.0 <\\Harmfulness_Rating>"
    )

    batch = collator([{"prompt": "P", "completion": think_prefix + fields}])

    assert set(batch) == {"input_ids", "attention_mask", "labels", "token_weights"}
    weights = batch["token_weights"][0]
    assert weights[:1].tolist() == [0.0]
    think_start = 1
    rationale_start = think_start + len(think_open)
    close_start = rationale_start + len(rationale)
    assert weights[think_start:rationale_start].tolist() == [1.0] * len(think_open)
    assert weights[rationale_start:close_start].tolist() == pytest.approx(
        [0.2] * len(rationale)
    )
    assert weights[close_start : close_start + len(think_close)].tolist() == [
        1.0
    ] * len(think_close)
    fields_start = 1 + len(think_prefix)
    assert weights[fields_start : fields_start + len(fields)].tolist() == [1.0] * len(
        fields
    )
    assert weights[fields_start + len(fields)].item() == 1.0


def test_weighted_causal_loss_applies_weights_after_one_token_shift():
    # Position 0 predicts label 1 with probability 1/2; position 1 predicts
    # label 0 with probability 3/4. The final logit has no next-token target.
    logits = torch.tensor(
        [[[0.0, 0.0], [torch.log(torch.tensor(3.0)), 0.0], [0.0, 0.0]]]
    )
    labels = torch.tensor([[-100, 1, 0]])
    token_weights = torch.tensor([[0.0, 0.2, 1.0]])

    loss = weighted_causal_lm_loss(logits, labels, token_weights)

    expected = (0.2 * torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(4.0 / 3.0))) / 1.2
    assert loss.item() == pytest.approx(expected.item())


def test_think_weight_and_fields_only_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SFTExampleCollator(
            TinyTokenizer(),
            max_length=64,
            supervise_guardian_fields_only=True,
            think_token_weight=0.2,
        )


def test_rationale_training_can_keep_held_out_validation_fields_only():
    config = TrainingConfig(
        model_path="model",
        train_file="train.jsonl",
        validation_file="validation.jsonl",
        output_dir="output",
        think_token_weight=0.2,
        validation_supervise_guardian_fields_only=True,
    )

    assert resolve_validation_supervision(config) == (True, None)


def test_epoch_checkpoint_copy_can_be_disabled_without_disabling_final_save():
    config = TrainingConfig(
        model_path="model",
        train_file="train.jsonl",
        validation_file="validation.jsonl",
        output_dir="output",
        save_model=True,
        save_epoch_checkpoints=False,
    )

    assert should_save_epoch_checkpoint(config) is False
    config.save_epoch_checkpoints = True
    assert should_save_epoch_checkpoint(config) is True
    config.save_model = False
    assert should_save_epoch_checkpoint(config) is False


def test_peft_import_error_reports_version_mismatch_and_recovery_file():
    message = format_peft_import_error(ImportError("cannot import HybridCache"))

    assert "transformers" in message
    assert "peft" in message
    assert "requirements-sft.txt" in message

"""Independently reload and validate prepared GRPO policy/reference models.

Unlike the merge runner, this command never imports PEFT and never writes model
weights.  It proves that both prepared directories are ordinary dense
Transformers checkpoints, contain identical bytes, and produce the same parsed
Guardian safety decision under deterministic generation.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from pathlib import Path

from merge_sft_for_grpo import _greedy_generation_kwargs, _parse_guardian_response
from model_preparation import (
    ModelObservation,
    ModelPreparationError,
    assert_matching_inventories,
)


REQUIRED_GUARDIAN_FIELDS = {
    "Malicious_User_Request",
    "Being_Attacked",
    "Harmfulness_Rating",
}
FORBIDDEN_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
}


def assert_no_peft_artifacts(
    parameter_names: Sequence[str],
    relative_file_paths: Sequence[str],
) -> None:
    """Reject LoRA tensors or adapter files in a claimed dense checkpoint."""

    suspicious_parameters = sorted(
        name
        for name in parameter_names
        if "lora_" in name.lower() or ".adapter" in name.lower()
    )
    suspicious_files = sorted(
        relative_path
        for relative_path in relative_file_paths
        if Path(relative_path).name in FORBIDDEN_ADAPTER_FILES
    )
    if suspicious_parameters or suspicious_files:
        raise ModelPreparationError(
            "dense checkpoint contains PEFT/LoRA artifacts: "
            f"parameters={suspicious_parameters}, files={suspicious_files}"
        )


def validate_reload_observations(
    policy: ModelObservation,
    reference: ModelObservation,
) -> dict[str, object]:
    """Require byte-identical checkpoints to reload with identical behavior."""

    if policy.token_ids != reference.token_ids:
        raise ModelPreparationError(
            "policy/reference generated token IDs differ after independent reload: "
            f"policy={policy.token_ids}, reference={reference.token_ids}"
        )
    if policy.parsed_judgment is None or reference.parsed_judgment is None:
        raise ModelPreparationError(
            "policy/reference responses must both parse as Guardian judgments"
        )
    if set(policy.parsed_judgment) != REQUIRED_GUARDIAN_FIELDS:
        raise ModelPreparationError(
            "policy parsed judgment does not contain exactly the three Guardian fields"
        )
    if set(reference.parsed_judgment) != REQUIRED_GUARDIAN_FIELDS:
        raise ModelPreparationError(
            "reference parsed judgment does not contain exactly the three Guardian fields"
        )
    if policy.parsed_judgment != reference.parsed_judgment:
        raise ModelPreparationError(
            "policy/reference parsed Guardian judgments differ: "
            f"policy={policy.parsed_judgment}, reference={reference.parsed_judgment}"
        )
    return {
        "generated_token_ids_equal": True,
        "parsed_judgment": dict(policy.parsed_judgment),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse prepared model and fixed-prompt paths."""

    parser = argparse.ArgumentParser(
        description="Independently reload and validate dense policy/reference checkpoints.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-prompt-file", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default=None,
        help="Explicit Transformers Attention implementation for reproducibility diagnosis.",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="Require PyTorch deterministic algorithms for this validation process.",
    )
    return parser.parse_args(argv)


def _build_model_load_kwargs(
    model_path: Path,
    *,
    attention_implementation: str | None,
) -> dict[str, object]:
    """Build explicit load arguments so Attention is a controlled variable."""

    import torch

    kwargs: dict[str, object] = {
        "pretrained_model_name_or_path": model_path,
        "dtype": torch.bfloat16,
        "device_map": {"": "cuda"},
        "local_files_only": True,
    }
    if attention_implementation is not None:
        kwargs["attn_implementation"] = attention_implementation
    return kwargs


def _observe_dense_model(
    model_path: Path,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    attention_implementation: str | None,
) -> tuple[ModelObservation, str, dict[str, object]]:
    """Load one plain Transformers model, inspect it, and generate once."""

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        **_build_model_load_kwargs(
            model_path,
            attention_implementation=attention_implementation,
        )
    )
    model.eval()
    relative_files = [
        candidate.relative_to(model_path).as_posix()
        for candidate in model_path.rglob("*")
        if candidate.is_file()
    ]
    parameter_names = [name for name, _ in model.named_parameters()]
    assert_no_peft_artifacts(parameter_names, relative_files)

    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    generation_kwargs = _greedy_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    with torch.inference_mode():
        first_logits = model(**encoded, use_cache=False).logits[:, -1, :].detach().float().cpu()
        generated = model.generate(**encoded, **generation_kwargs)
    token_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    observation = ModelObservation(
        token_ids=token_ids,
        logits=first_logits,
        parsed_judgment=_parse_guardian_response(response),
    )
    metadata = {
        "model_class": type(model).__name__,
        "attention_implementation": model.config._attn_implementation,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peft_or_lora_artifacts_found": False,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return observation, response, metadata


def main(argv: Sequence[str] | None = None) -> None:
    """Validate policy/reference inventories and independent reload behavior."""

    import torch
    from transformers import AutoTokenizer

    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise ModelPreparationError("CUDA is required for independent model reload")
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    if args.max_new_tokens <= 0:
        raise ModelPreparationError("max_new_tokens must be positive")

    policy_path = args.output_root / "policy_init"
    reference_path = args.output_root / "reference"
    inventory = assert_matching_inventories(policy_path, reference_path)
    prompt_text = args.validation_prompt_file.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ModelPreparationError("validation prompt must not be empty")
    tokenizer = AutoTokenizer.from_pretrained(policy_path, local_files_only=True)

    policy_observation, policy_response, policy_metadata = _observe_dense_model(
        policy_path,
        tokenizer,
        prompt_text,
        max_new_tokens=args.max_new_tokens,
        attention_implementation=args.attention_implementation,
    )
    reference_observation, reference_response, reference_metadata = _observe_dense_model(
        reference_path,
        tokenizer,
        prompt_text,
        max_new_tokens=args.max_new_tokens,
        attention_implementation=args.attention_implementation,
    )
    behavioral_validation = validate_reload_observations(
        policy_observation,
        reference_observation,
    )
    weight_record = inventory.get("model.safetensors")
    if weight_record is None:
        raise ModelPreparationError("dense checkpoint has no model.safetensors")

    report = {
        "status": "complete",
        "mode": "independent-reload-validation",
        "model_write_performed": False,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "output_root": str(args.output_root),
        "policy_path": str(policy_path),
        "reference_path": str(reference_path),
        "policy_reference_inventories_equal": True,
        "checkpoint_file_count": len(inventory),
        "checkpoint_total_bytes": sum(record.size_bytes for record in inventory.values()),
        "dense_weight_sha256": weight_record.sha256,
        "dense_weight_size_bytes": weight_record.size_bytes,
        "policy": policy_metadata,
        "reference": reference_metadata,
        "behavioral_validation": behavioral_validation,
        "policy_response": policy_response,
        "reference_response": reference_response,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

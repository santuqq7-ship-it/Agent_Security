"""Merge the verified SFT LoRA into a dense BF16 GRPO starting checkpoint.

The command implemented here performs model conversion and deterministic
equivalence checks only.  It never constructs an optimizer and never updates
parameters with a training loss.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from model_preparation import (
    FileRecord,
    ModelObservation,
    ModelPreparationError,
    assert_matching_inventories,
    inventory_directory,
    sha256_file,
    validate_model_outputs,
    write_json_atomic,
)


@dataclass(frozen=True)
class PreparationConfig:
    """All explicit inputs and validation limits for one preparation run."""

    base_model_path: Path
    adapter_path: Path
    output_root: Path
    expected_base_sha256: str
    expected_adapter_sha256: str
    validation_prompt_file: Path
    max_total_variation_distance: float = 0.05
    minimum_free_space_multiplier: int = 3
    max_new_tokens: int = 128

    @property
    def base_weight_file(self) -> Path:
        """Return the exact base weight file verified for this project."""

        return self.base_model_path / "model.safetensors"

    @property
    def adapter_weight_file(self) -> Path:
        """Return the exact SFT LoRA weight file verified for this project."""

        return self.adapter_path / "adapter_model.safetensors"


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Read-only facts required before the paid GPU merge is allowed."""

    executable: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    cuda_available: bool
    device_name: str | None
    compute_capability: str | None
    bfloat16_supported: bool
    total_gpu_memory_bytes: int | None
    library_versions: dict[str, str]


REQUIRED_SFT_EXECUTABLE = (
    "/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python"
)
REQUIRED_PYTHON_VERSION = "3.10.16"
REQUIRED_CUDA_VERSION = "12.4"
REQUIRED_LIBRARY_VERSIONS = {
    "torch": "2.5.1+cu124",
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "safetensors": "0.8.0",
}


@dataclass(frozen=True)
class MergeArtifacts:
    """Observable outputs returned by the expensive model-conversion boundary."""

    before: ModelObservation
    before_repeat: ModelObservation
    after: ModelObservation
    after_repeat: ModelObservation
    pre_merge_response: str
    post_merge_response: str
    parameter_count: int
    library_versions: dict[str, str]


class MergeBoundary(Protocol):
    """Boundary around real CUDA, Transformers, and PEFT operations."""

    def merge_and_save(
        self,
        config: PreparationConfig,
        policy_path: Path,
        prompt_text: str,
    ) -> MergeArtifacts:
        """Merge the adapter, save a dense policy, and observe both models."""


def _directory_size(path: Path) -> int:
    """Return total bytes of all regular files below a directory."""

    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())


def _nearest_existing_directory(path: Path) -> Path:
    """Find a disk-usage probe point without creating the requested output path."""

    candidate = path.resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ModelPreparationError(
                f"no existing parent directory found for output path: {path}"
            )
        candidate = parent
    if not candidate.is_dir():
        raise ModelPreparationError(
            f"nearest existing output ancestor is not a directory: {candidate}"
        )
    return candidate


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _validate_expected_sha256(value: str, *, label: str) -> str:
    """Normalize one required SHA-256 digest and reject ambiguous values."""

    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ModelPreparationError(f"{label} must be a 64-character hexadecimal SHA-256")
    return normalized


def _inventory_payload(inventory: dict[str, FileRecord]) -> dict[str, dict[str, object]]:
    """Convert a typed inventory into JSON-serializable records."""

    return {relative_path: asdict(record) for relative_path, record in inventory.items()}


def _verify_initial_state(config: PreparationConfig) -> tuple[dict[str, FileRecord], dict[str, FileRecord]]:
    """Validate immutable inputs and output isolation before loading a model."""

    if os.path.lexists(config.output_root):
        raise ModelPreparationError(f"output root already exists: {config.output_root}")
    for source_path, label in (
        (config.base_model_path, "base model"),
        (config.adapter_path, "adapter"),
    ):
        if not source_path.is_dir():
            raise ModelPreparationError(f"{label} directory does not exist: {source_path}")
        if _paths_overlap(config.output_root, source_path):
            raise ModelPreparationError(
                f"output/source paths overlap: output={config.output_root}, source={source_path}"
            )
    if not config.validation_prompt_file.is_file():
        raise ModelPreparationError(
            f"validation prompt file does not exist: {config.validation_prompt_file}"
        )
    if config.max_new_tokens <= 0:
        raise ModelPreparationError("max_new_tokens must be positive")
    if not 0 <= config.max_total_variation_distance <= 1:
        raise ModelPreparationError(
            "max_total_variation_distance must be between 0 and 1"
        )
    if config.minimum_free_space_multiplier < 2:
        raise ModelPreparationError("minimum_free_space_multiplier must be at least 2")

    expected_base = _validate_expected_sha256(
        config.expected_base_sha256,
        label="expected base model SHA-256",
    )
    expected_adapter = _validate_expected_sha256(
        config.expected_adapter_sha256,
        label="expected adapter SHA-256",
    )
    actual_base = sha256_file(config.base_weight_file)
    actual_adapter = sha256_file(config.adapter_weight_file)
    if actual_base != expected_base:
        raise ModelPreparationError(
            "base model SHA-256 mismatch: "
            f"expected={expected_base}, actual={actual_base}, file={config.base_weight_file}"
        )
    if actual_adapter != expected_adapter:
        raise ModelPreparationError(
            "adapter SHA-256 mismatch: "
            f"expected={expected_adapter}, actual={actual_adapter}, file={config.adapter_weight_file}"
        )

    return (
        inventory_directory(config.base_model_path),
        inventory_directory(config.adapter_path),
    )


def probe_runtime_environment() -> RuntimeEnvironment:
    """Inspect the active interpreter and GPU without loading model weights."""

    try:
        import torch
    except ImportError as error:
        raise ModelPreparationError("PyTorch is not installed in the active interpreter") from error

    library_versions: dict[str, str] = {"torch": str(torch.__version__)}
    for distribution in ("transformers", "peft", "accelerate", "safetensors"):
        try:
            library_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            library_versions[distribution] = "not-installed"

    cuda_available = bool(torch.cuda.is_available())
    device_name: str | None = None
    compute_capability: str | None = None
    bfloat16_supported = False
    total_gpu_memory_bytes: int | None = None
    if cuda_available:
        major, minor = torch.cuda.get_device_capability(0)
        properties = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        compute_capability = f"{major}.{minor}"
        bfloat16_supported = bool(torch.cuda.is_bf16_supported())
        total_gpu_memory_bytes = int(properties.total_memory)

    return RuntimeEnvironment(
        executable=sys.executable,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cuda_available=cuda_available,
        device_name=device_name,
        compute_capability=compute_capability,
        bfloat16_supported=bfloat16_supported,
        total_gpu_memory_bytes=total_gpu_memory_bytes,
        library_versions=library_versions,
    )


def _validate_runtime_environment(runtime: RuntimeEnvironment) -> None:
    """Reject an environment different from the recorded cloud SFT runtime."""

    if runtime.executable != REQUIRED_SFT_EXECUTABLE:
        raise ModelPreparationError(
            "wrong Python interpreter: "
            f"required={REQUIRED_SFT_EXECUTABLE}, actual={runtime.executable}"
        )
    if runtime.python_version != REQUIRED_PYTHON_VERSION:
        raise ModelPreparationError(
            "wrong Python version: "
            f"required={REQUIRED_PYTHON_VERSION}, actual={runtime.python_version}"
        )
    for distribution, required_version in REQUIRED_LIBRARY_VERSIONS.items():
        actual_version = runtime.library_versions.get(distribution, "not-installed")
        if actual_version != required_version:
            raise ModelPreparationError(
                f"wrong {distribution} version: "
                f"required={required_version}, actual={actual_version}"
            )
    if not runtime.cuda_available:
        raise ModelPreparationError("CUDA is not available in the active interpreter")
    if runtime.cuda_version != REQUIRED_CUDA_VERSION:
        raise ModelPreparationError(
            "wrong PyTorch CUDA runtime: "
            f"required={REQUIRED_CUDA_VERSION}, actual={runtime.cuda_version}"
        )
    if not runtime.bfloat16_supported:
        raise ModelPreparationError("the selected CUDA device does not support bfloat16")
    if runtime.compute_capability is None:
        raise ModelPreparationError("CUDA compute capability could not be determined")


def check_preparation(
    config: PreparationConfig,
    *,
    runtime_probe: Callable[[], RuntimeEnvironment] = probe_runtime_environment,
) -> dict[str, object]:
    """Run a non-mutating readiness check without importing Transformers or PEFT."""

    base_inventory, adapter_inventory = _verify_initial_state(config)
    prompt_text = config.validation_prompt_file.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ModelPreparationError("validation prompt must not be empty")

    required_free_bytes = (
        _directory_size(config.base_model_path) * config.minimum_free_space_multiplier
        + _directory_size(config.adapter_path)
    )
    disk_probe_path = _nearest_existing_directory(config.output_root.parent)
    available_free_bytes = shutil.disk_usage(disk_probe_path).free
    if available_free_bytes < required_free_bytes:
        raise ModelPreparationError(
            "insufficient free disk space: "
            f"required={required_free_bytes}, available={available_free_bytes}"
        )

    runtime = runtime_probe()
    _validate_runtime_environment(runtime)
    return {
        "status": "ready",
        "mode": "check-only",
        "device": "cuda",
        "dtype": "bfloat16",
        "model_load_performed": False,
        "base_model_path": str(config.base_model_path),
        "adapter_path": str(config.adapter_path),
        "output_root": str(config.output_root),
        "base_model_sha256": sha256_file(config.base_weight_file),
        "adapter_sha256": sha256_file(config.adapter_weight_file),
        "validation_prompt_sha256": sha256_file(config.validation_prompt_file),
        "validation_thresholds": {
            "max_total_variation_distance": config.max_total_variation_distance,
        },
        "base_inventory_file_count": len(base_inventory),
        "adapter_inventory_file_count": len(adapter_inventory),
        "required_free_bytes": required_free_bytes,
        "available_free_bytes": available_free_bytes,
        "disk_probe_path": str(disk_probe_path),
        "runtime": asdict(runtime),
    }


def _assert_sources_unchanged(
    config: PreparationConfig,
    base_before: dict[str, FileRecord],
    adapter_before: dict[str, FileRecord],
) -> tuple[dict[str, FileRecord], dict[str, FileRecord]]:
    """Prove that loading and merging did not mutate either source directory."""

    base_after = inventory_directory(config.base_model_path)
    adapter_after = inventory_directory(config.adapter_path)
    if base_after != base_before:
        raise ModelPreparationError("base model changed during preparation")
    if adapter_after != adapter_before:
        raise ModelPreparationError("adapter changed during preparation")
    return base_after, adapter_after


def prepare_models(
    config: PreparationConfig,
    *,
    boundaries: MergeBoundary | None = None,
) -> Path:
    """Create validated policy/reference checkpoints with atomic promotion."""

    base_before, adapter_before = _verify_initial_state(config)
    prompt_text = config.validation_prompt_file.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ModelPreparationError("validation prompt must not be empty")

    config.output_root.parent.mkdir(parents=True, exist_ok=True)
    required_free_bytes = (
        _directory_size(config.base_model_path) * config.minimum_free_space_multiplier
        + _directory_size(config.adapter_path)
    )
    available_free_bytes = shutil.disk_usage(config.output_root.parent).free
    if available_free_bytes < required_free_bytes:
        raise ModelPreparationError(
            "insufficient free disk space: "
            f"required={required_free_bytes}, available={available_free_bytes}"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_root.name}.tmp-",
            dir=config.output_root.parent,
        )
    )
    policy_path = temporary_root / "policy_init"
    reference_path = temporary_root / "reference"
    merge_boundary = boundaries or TransformersPeftMergeBoundary()

    try:
        artifacts = merge_boundary.merge_and_save(
            config,
            policy_path,
            prompt_text,
        )
        validation = validate_model_outputs(
            artifacts.before,
            artifacts.after,
            before_repeat=artifacts.before_repeat,
            after_repeat=artifacts.after_repeat,
            max_total_variation_distance=config.max_total_variation_distance,
        )
        if not policy_path.is_dir():
            raise ModelPreparationError(
                f"merge boundary did not create policy checkpoint: {policy_path}"
            )
        shutil.copytree(policy_path, reference_path)
        policy_inventory = assert_matching_inventories(policy_path, reference_path)
        base_after, adapter_after = _assert_sources_unchanged(
            config,
            base_before,
            adapter_before,
        )

        manifest = {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dtype": "bfloat16",
            "base_model_path": str(config.base_model_path),
            "adapter_path": str(config.adapter_path),
            "expected_base_sha256": config.expected_base_sha256.lower(),
            "expected_adapter_sha256": config.expected_adapter_sha256.lower(),
            "validation_prompt_file": str(config.validation_prompt_file),
            "validation_prompt_sha256": sha256_file(config.validation_prompt_file),
            "validation_thresholds": {
                "max_total_variation_distance": config.max_total_variation_distance,
            },
            "required_free_bytes": required_free_bytes,
            "available_free_bytes_before_save": available_free_bytes,
            "parameter_count": artifacts.parameter_count,
            "library_versions": artifacts.library_versions,
            "pre_merge_response": artifacts.pre_merge_response,
            "post_merge_response": artifacts.post_merge_response,
            "pre_merge_parsed_judgment": artifacts.before.parsed_judgment,
            "post_merge_parsed_judgment": artifacts.after.parsed_judgment,
            "validation": asdict(validation),
            "policy_reference_inventories_equal": True,
            "policy_inventory": _inventory_payload(policy_inventory),
            "base_source_inventory_before": _inventory_payload(base_before),
            "base_source_inventory_after": _inventory_payload(base_after),
            "adapter_source_inventory_before": _inventory_payload(adapter_before),
            "adapter_source_inventory_after": _inventory_payload(adapter_after),
        }
        write_json_atomic(temporary_root / "preparation_manifest.json", manifest)
        temporary_root.rename(config.output_root)
        return config.output_root / "preparation_manifest.json"
    except Exception:
        print(
            "model preparation failed; temporary directory preserved: "
            f"{temporary_root}",
            file=sys.stderr,
        )
        raise


@lru_cache(maxsize=1)
def _load_official_guardian_parser():
    """Load the unchanged ToolSafe TS-Guard parser from its source file."""

    workspace_root = Path(__file__).resolve().parents[3]
    parser_path = workspace_root / "ToolSafe" / "src" / "utils" / "guardian_parser.py"
    if not parser_path.is_file():
        raise ModelPreparationError(f"official Guardian parser not found: {parser_path}")
    spec = importlib.util.spec_from_file_location(
        "toolsafe_guardian_parser_for_model_preparation",
        parser_path,
    )
    if spec is None or spec.loader is None:
        raise ModelPreparationError(f"cannot load official Guardian parser: {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ashellguardian_parser_v2


def _parse_guardian_response(response: str) -> dict[str, object] | None:
    """Return exact original-parser fields, with no compatibility fallback."""

    parser = _load_official_guardian_parser()
    parsed_score, parsed_fields = parser(response)
    if parsed_score == "error":
        return None
    return dict(parsed_fields)


def _greedy_generation_kwargs(tokenizer, *, max_new_tokens: int) -> dict[str, object]:
    """Return direct kwargs that override model-specific sampling defaults.

    Passing a ``GenerationConfig.from_model_config`` object is insufficient in
    Transformers 4.57 because ``generate`` may restore Qwen's sampling defaults
    for fields that equal library defaults.  Direct keyword arguments have the
    highest precedence and make the actual decoding path auditable.
    """

    return {
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "eos_token_id": tokenizer.eos_token_id,
    }


class TransformersPeftMergeBoundary:
    """Real CUDA implementation of deterministic LoRA merge and reload."""

    @staticmethod
    def _library_versions() -> dict[str, str]:
        versions: dict[str, str] = {}
        for distribution in ("torch", "transformers", "peft", "accelerate", "safetensors"):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "not-installed"
        return versions

    @staticmethod
    def _observe(model, tokenizer, prompt_text: str, *, max_new_tokens: int) -> tuple[ModelObservation, str]:
        """Capture last-prompt logits and one deterministic greedy response."""

        import torch
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
            logits = model(**encoded, use_cache=False).logits[:, -1, :].detach().float().cpu()
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )
        generated_token_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
        response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return (
            ModelObservation(
                token_ids=generated_token_ids,
                logits=logits,
                parsed_judgment=_parse_guardian_response(response),
            ),
            response,
        )

    def merge_and_save(
        self,
        config: PreparationConfig,
        policy_path: Path,
        prompt_text: str,
    ) -> MergeArtifacts:
        """Load SFT LoRA, merge into BF16, save, reload, and observe."""

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise ModelPreparationError("CUDA is required for the real BF16 model merge")

        tokenizer = AutoTokenizer.from_pretrained(
            config.adapter_path,
            local_files_only=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            local_files_only=True,
        )
        sft_model = PeftModel.from_pretrained(
            base_model,
            config.adapter_path,
            is_trainable=False,
        )
        sft_model.eval()
        before, pre_merge_response = self._observe(
            sft_model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
        )
        before_repeat, _ = self._observe(
            sft_model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
        )

        merged_model = sft_model.merge_and_unload(safe_merge=True)
        merged_model.eval()
        parameter_count = sum(parameter.numel() for parameter in merged_model.parameters())
        policy_path.mkdir(parents=True)
        merged_model.save_pretrained(
            policy_path,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(policy_path)

        del sft_model, base_model, merged_model
        gc.collect()
        torch.cuda.empty_cache()

        dense_model = AutoModelForCausalLM.from_pretrained(
            policy_path,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            local_files_only=True,
        )
        dense_model.eval()
        after, post_merge_response = self._observe(
            dense_model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
        )
        after_repeat, _ = self._observe(
            dense_model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
        )
        del dense_model
        gc.collect()
        torch.cuda.empty_cache()

        forbidden_adapter_files = {
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
        }
        unexpected = sorted(
            candidate.name
            for candidate in policy_path.iterdir()
            if candidate.name in forbidden_adapter_files
        )
        if unexpected:
            raise ModelPreparationError(
                f"dense policy unexpectedly contains PEFT adapter files: {unexpected}"
            )

        return MergeArtifacts(
            before=before,
            before_repeat=before_repeat,
            after=after,
            after_repeat=after_repeat,
            pre_merge_response=pre_merge_response,
            post_merge_response=post_merge_response,
            parameter_count=parameter_count,
            library_versions=self._library_versions(),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit model-preparation paths and integrity boundaries."""

    parser = argparse.ArgumentParser(
        description="Merge the verified SFT LoRA into a dense BF16 GRPO checkpoint.",
    )
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--validation-prompt-file", type=Path, required=True)
    parser.add_argument(
        "--max-total-variation-distance",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed total-variation distance between the pre/post "
            "first-token probability distributions."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Verify immutable inputs, disk space, interpreter, packages, and CUDA "
            "without loading or writing a model."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a read-only preflight or one non-training BF16 preparation."""

    args = parse_args(argv)
    config = PreparationConfig(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_root=args.output_root,
        expected_base_sha256=args.expected_base_sha256,
        expected_adapter_sha256=args.expected_adapter_sha256,
        validation_prompt_file=args.validation_prompt_file,
        max_total_variation_distance=args.max_total_variation_distance,
    )
    if args.check_only:
        print(json.dumps(check_preparation(config), indent=2))
        return
    manifest_path = prepare_models(config)
    print(json.dumps({"manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

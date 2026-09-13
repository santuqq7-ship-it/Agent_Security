"""Tests for GRPO actor/reference model preparation integrity checks."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest
import torch


GRPO_DIR = Path(__file__).parents[1] / "grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from model_preparation import (  # noqa: E402
    ModelObservation,
    ModelPreparationError,
    assert_matching_inventories,
    inventory_directory,
    sha256_file,
    validate_model_outputs,
    write_json_atomic,
)
from merge_sft_for_grpo import (  # noqa: E402
    MergeArtifacts,
    PreparationConfig,
    RuntimeEnvironment,
    _greedy_generation_kwargs,
    _parse_guardian_response,
    check_preparation,
    main,
    parse_args,
    prepare_models,
)
from validate_prepared_models import (  # noqa: E402
    _build_model_load_kwargs,
    assert_no_peft_artifacts,
    parse_args as parse_validation_args,
    validate_reload_observations,
)


SAFE_JUDGMENT = {
    "Malicious_User_Request": "no",
    "Being_Attacked": "no",
    "Harmfulness_Rating": 0.0,
}


class FakeMergeBoundary:
    """Replace only the expensive Transformers/PEFT boundary in unit tests."""

    def __init__(
        self,
        *,
        after_token_ids: list[int] | None = None,
        after_repeat_token_ids: list[int] | None = None,
        mutate_base: bool = False,
    ) -> None:
        self.after_token_ids = after_token_ids or [10, 11]
        self.after_repeat_token_ids = after_repeat_token_ids or self.after_token_ids
        self.mutate_base = mutate_base
        self.load_call_count = 0

    def merge_and_save(
        self,
        config: PreparationConfig,
        policy_path: Path,
        prompt_text: str,
    ) -> MergeArtifacts:
        self.load_call_count += 1
        policy_path.mkdir(parents=True)
        (policy_path / "config.json").write_text("{}", encoding="utf-8")
        (policy_path / "model.safetensors").write_bytes(b"merged-model")
        if self.mutate_base:
            (config.base_model_path / "model.safetensors").write_bytes(b"changed")

        return MergeArtifacts(
            before=ModelObservation(
                token_ids=[10, 11],
                logits=torch.tensor([1.0, 2.0]),
                parsed_judgment=SAFE_JUDGMENT,
            ),
            before_repeat=ModelObservation(
                token_ids=[10, 11],
                logits=torch.tensor([1.0, 2.0]),
                parsed_judgment=SAFE_JUDGMENT,
            ),
            after=ModelObservation(
                token_ids=self.after_token_ids,
                logits=torch.tensor([1.0, 2.1]),
                parsed_judgment=SAFE_JUDGMENT,
            ),
            after_repeat=ModelObservation(
                token_ids=self.after_repeat_token_ids,
                logits=torch.tensor([1.0, 2.1]),
                parsed_judgment=SAFE_JUDGMENT,
            ),
            pre_merge_response="before",
            post_merge_response="after",
            parameter_count=1_543_714_304,
            library_versions={"torch": "test", "transformers": "test", "peft": "test"},
        )


def make_preparation_config(
    tmp_path: Path,
    *,
    output_root: Path | None = None,
    expected_base_sha256: str | None = None,
) -> PreparationConfig:
    """Create real input files while replacing only multi-GB model loading."""

    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir(exist_ok=True)
    adapter.mkdir(exist_ok=True)
    (base / "model.safetensors").write_bytes(b"base")
    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    prompt = tmp_path / "validation_prompt.txt"
    prompt.write_text("fixed validation prompt", encoding="utf-8")

    return PreparationConfig(
        base_model_path=base,
        adapter_path=adapter,
        output_root=output_root or tmp_path / "prepared",
        expected_base_sha256=(
            expected_base_sha256 or hashlib.sha256(b"base").hexdigest()
        ),
        expected_adapter_sha256=hashlib.sha256(b"adapter").hexdigest(),
        validation_prompt_file=prompt,
        max_total_variation_distance=0.05,
    )


def test_sha256_file_hashes_file_bytes(tmp_path):
    """Changing file bytes must change the identity used by preparation."""

    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")

    assert sha256_file(source) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_directory_inventory_uses_relative_paths_and_real_file_metadata(tmp_path):
    """Moving a checkpoint must not change its relative inventory keys."""

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "nested" / "weights.bin").write_bytes(b"weights")

    inventory = inventory_directory(checkpoint)

    assert sorted(inventory) == ["config.json", "nested/weights.bin"]
    assert inventory["config.json"].size_bytes == 2
    assert inventory["nested/weights.bin"].size_bytes == 7


def test_reference_inventory_must_match_policy(tmp_path):
    """A changed reference byte must prevent a false identical-copy claim."""

    policy = tmp_path / "policy"
    reference = tmp_path / "reference"
    policy.mkdir()
    reference.mkdir()
    (policy / "config.json").write_text("policy", encoding="utf-8")
    (reference / "config.json").write_text("reference", encoding="utf-8")

    with pytest.raises(ModelPreparationError, match="inventory mismatch"):
        assert_matching_inventories(policy, reference)


def test_matching_policy_and_reference_inventories_are_returned(tmp_path):
    """An exact copy must produce a manifest-ready verified inventory."""

    policy = tmp_path / "policy"
    reference = tmp_path / "reference"
    policy.mkdir()
    reference.mkdir()
    for directory in (policy, reference):
        (directory / "config.json").write_text("same", encoding="utf-8")

    inventory = assert_matching_inventories(policy, reference)

    assert inventory["config.json"].size_bytes == 4


def test_irrelevant_token_difference_passes_when_safety_judgment_is_stable():
    """BF16 formatting-token drift must not hide a stable safety decision."""

    before = ModelObservation(
        token_ids=[1, 2],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    after = ModelObservation(
        token_ids=[1, 3],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )

    result = validate_model_outputs(
        before,
        after,
        before_repeat=before,
        after_repeat=after,
        max_total_variation_distance=0.05,
    )

    assert result.generated_token_ids_equal is False
    assert result.parsed_judgment == SAFE_JUDGMENT


def test_excessive_probability_distribution_difference_fails_validation():
    """A large shift in normalized token probability mass must stop promotion."""

    before = ModelObservation(
        token_ids=[1, 2],
        logits=torch.tensor([5.0, -5.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    after = ModelObservation(
        token_ids=[1, 2],
        logits=torch.tensor([-5.0, 5.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )

    with pytest.raises(ModelPreparationError, match="total-variation"):
        validate_model_outputs(
            before,
            after,
            before_repeat=before,
            after_repeat=after,
            max_total_variation_distance=0.05,
        )


def test_nonfinite_logits_fail_validation():
    """NaN logits must not make a numerical comparison appear successful."""

    before = ModelObservation(
        token_ids=[1],
        logits=torch.tensor([1.0, float("nan")]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    after = ModelObservation(
        token_ids=[1],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )

    with pytest.raises(ModelPreparationError, match="finite"):
        validate_model_outputs(
            before,
            after,
            before_repeat=before,
            after_repeat=after,
            max_total_variation_distance=0.05,
        )


def test_parsed_judgment_mismatch_fails_when_one_side_parses():
    """A merge must not change a successfully parsed Guardian decision."""

    before = ModelObservation(
        token_ids=[1],
        logits=torch.tensor([1.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    after = ModelObservation(
        token_ids=[1],
        logits=torch.tensor([1.0]),
        parsed_judgment=None,
    )

    with pytest.raises(ModelPreparationError, match="parsed judgment"):
        validate_model_outputs(
            before,
            after,
            before_repeat=before,
            after_repeat=after,
            max_total_variation_distance=0.05,
        )


def test_changed_safety_field_fails_even_when_tokens_and_logits_match():
    """The safety decision is the non-negotiable semantic merge boundary."""

    before = ModelObservation(
        token_ids=[1],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    changed_judgment = {**SAFE_JUDGMENT, "Harmfulness_Rating": 1.0}
    after = ModelObservation(
        token_ids=[1],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=changed_judgment,
    )

    with pytest.raises(ModelPreparationError, match="parsed judgment changed"):
        validate_model_outputs(
            before,
            after,
            before_repeat=before,
            after_repeat=after,
            max_total_variation_distance=0.05,
        )


def test_nondeterministic_repeat_generation_fails_validation():
    """A safety comparison is invalid if either representation is unstable."""

    before = ModelObservation(
        token_ids=[1, 2],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    unstable_repeat = ModelObservation(
        token_ids=[1, 3],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    after = ModelObservation(
        token_ids=[4, 5],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )

    with pytest.raises(ModelPreparationError, match="pre-merge generation is not deterministic"):
        validate_model_outputs(
            before,
            after,
            before_repeat=unstable_repeat,
            after_repeat=after,
            max_total_variation_distance=0.05,
        )


def test_valid_outputs_return_measured_differences():
    """A valid merge must expose, rather than hide, its numerical error."""

    before = ModelObservation(
        token_ids=[4, 5],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    after = ModelObservation(
        token_ids=[4, 5],
        logits=torch.tensor([1.1, 2.1]),
        parsed_judgment=SAFE_JUDGMENT,
    )

    result = validate_model_outputs(
        before,
        after,
        before_repeat=before,
        after_repeat=after,
        max_total_variation_distance=0.05,
    )

    assert result.generated_token_ids_equal is True
    assert result.pre_merge_generated_token_ids == [4, 5]
    assert result.post_merge_generated_token_ids == [4, 5]
    assert result.max_logit_abs_diff == pytest.approx(0.1)
    assert result.mean_logit_abs_diff == pytest.approx(0.1)
    assert result.total_variation_distance < 0.05


def test_write_json_atomic_leaves_only_complete_json(tmp_path):
    """A successful manifest write must leave no temporary sibling file."""

    destination = tmp_path / "preparation_manifest.json"

    write_json_atomic(destination, {"status": "complete", "count": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "complete",
        "count": 2,
    }
    assert list(tmp_path.iterdir()) == [destination]


def test_existing_output_is_never_overwritten(tmp_path):
    """A prior checkpoint must never be replaced by a new preparation run."""

    output = tmp_path / "prepared"
    output.mkdir()
    config = make_preparation_config(tmp_path, output_root=output)
    boundary = FakeMergeBoundary()

    with pytest.raises(ModelPreparationError, match="already exists"):
        prepare_models(config, boundaries=boundary)

    assert boundary.load_call_count == 0


def test_output_inside_source_model_is_rejected(tmp_path):
    """A misconfigured output path must not write into immutable inputs."""

    config = make_preparation_config(tmp_path)
    config = PreparationConfig(
        **{
            **config.__dict__,
            "output_root": config.base_model_path / "prepared",
        }
    )
    boundary = FakeMergeBoundary()

    with pytest.raises(ModelPreparationError, match="overlap"):
        prepare_models(config, boundaries=boundary)

    assert boundary.load_call_count == 0


def test_manifest_is_not_promoted_when_validation_fails(tmp_path):
    """Nondeterministic output must leave no apparently complete checkpoint."""

    config = make_preparation_config(tmp_path)
    boundary = FakeMergeBoundary(after_repeat_token_ids=[99])

    with pytest.raises(ModelPreparationError, match="not deterministic"):
        prepare_models(config, boundaries=boundary)

    assert not config.output_root.exists()
    assert any(path.name.startswith(".prepared.tmp-") for path in tmp_path.iterdir())


def test_source_hash_mismatch_stops_before_model_load(tmp_path):
    """The wrong base checkpoint must be rejected before allocating a model."""

    config = make_preparation_config(tmp_path, expected_base_sha256="0" * 64)
    boundary = FakeMergeBoundary()

    with pytest.raises(ModelPreparationError, match="base model SHA-256"):
        prepare_models(config, boundaries=boundary)

    assert boundary.load_call_count == 0


def test_source_mutation_during_merge_prevents_output_promotion(tmp_path):
    """A modified source must invalidate the run even after model conversion."""

    config = make_preparation_config(tmp_path)
    boundary = FakeMergeBoundary(mutate_base=True)

    with pytest.raises(ModelPreparationError, match="base model changed"):
        prepare_models(config, boundaries=boundary)

    assert not config.output_root.exists()


def test_success_creates_identical_policy_reference_and_complete_manifest(tmp_path):
    """Only a validated run may expose policy, reference, and manifest."""

    config = make_preparation_config(tmp_path)

    manifest_path = prepare_models(config, boundaries=FakeMergeBoundary())

    assert manifest_path == config.output_root / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["dtype"] == "bfloat16"
    assert manifest["validation"]["generated_token_ids_equal"] is True
    assert manifest["validation"]["parsed_judgment"] == SAFE_JUDGMENT
    assert manifest["validation_thresholds"] == {
        "max_total_variation_distance": 0.05,
    }
    assert manifest["policy_reference_inventories_equal"] is True
    assert_matching_inventories(
        config.output_root / "policy_init",
        config.output_root / "reference",
    )


def test_parse_args_accepts_all_model_preparation_boundaries(tmp_path):
    """The CLI must expose every path and integrity value used by the run."""

    args = parse_args(
        [
            "--base-model-path",
            str(tmp_path / "base"),
            "--adapter-path",
            str(tmp_path / "adapter"),
            "--output-root",
            str(tmp_path / "prepared"),
            "--expected-base-sha256",
            "a" * 64,
            "--expected-adapter-sha256",
            "b" * 64,
            "--validation-prompt-file",
            str(tmp_path / "prompt.txt"),
            "--max-total-variation-distance",
            "0.075",
        ]
    )

    assert args.base_model_path == tmp_path / "base"
    assert args.adapter_path == tmp_path / "adapter"
    assert args.output_root == tmp_path / "prepared"
    assert args.max_total_variation_distance == 0.075


def test_parse_args_accepts_check_only(tmp_path):
    """The CLI must expose an explicit non-mutating preflight mode."""

    args = parse_args(
        [
            "--base-model-path",
            str(tmp_path / "base"),
            "--adapter-path",
            str(tmp_path / "adapter"),
            "--output-root",
            str(tmp_path / "prepared"),
            "--expected-base-sha256",
            "a" * 64,
            "--expected-adapter-sha256",
            "b" * 64,
            "--validation-prompt-file",
            str(tmp_path / "prompt.txt"),
            "--check-only",
        ]
    )

    assert args.check_only is True


def test_check_only_verifies_environment_without_loading_or_writing(tmp_path, monkeypatch):
    """Preflight may hash inputs and inspect hardware, but must not load a model."""

    output_root = tmp_path / "missing-parent" / "nested" / "prepared"
    config = make_preparation_config(tmp_path, output_root=output_root)

    class ForbiddenBoundary:
        def __init__(self):
            raise AssertionError("check-only instantiated the model merge boundary")

    monkeypatch.setattr(
        "merge_sft_for_grpo.TransformersPeftMergeBoundary",
        ForbiddenBoundary,
    )
    runtime = RuntimeEnvironment(
        executable="/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python",
        python_version="3.10.16",
        torch_version="2.5.1+cu124",
        cuda_version="12.4",
        cuda_available=True,
        device_name="NVIDIA A800-SXM4-80GB",
        compute_capability="8.0",
        bfloat16_supported=True,
        total_gpu_memory_bytes=80 * 1024**3,
        library_versions={
            "torch": "2.5.1+cu124",
            "transformers": "4.57.1",
            "peft": "0.17.1",
            "accelerate": "1.10.1",
            "safetensors": "0.8.0",
        },
    )

    result = check_preparation(config, runtime_probe=lambda: runtime)

    assert result["status"] == "ready"
    assert result["mode"] == "check-only"
    assert result["device"] == "cuda"
    assert result["dtype"] == "bfloat16"
    assert result["model_load_performed"] is False
    assert result["base_model_sha256"] == hashlib.sha256(b"base").hexdigest()
    assert result["adapter_sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert result["required_free_bytes"] > 0
    assert result["available_free_bytes"] >= result["required_free_bytes"]
    assert result["validation_thresholds"] == {
        "max_total_variation_distance": 0.05,
    }
    assert result["runtime"]["compute_capability"] == "8.0"
    assert not output_root.exists()
    assert not output_root.parent.exists()


def test_check_only_cli_never_dispatches_real_preparation(tmp_path, monkeypatch, capsys):
    """The CLI flag must return after preflight instead of entering model merge."""

    expected = {
        "status": "ready",
        "mode": "check-only",
        "model_load_performed": False,
    }
    monkeypatch.setattr(
        "merge_sft_for_grpo.check_preparation",
        lambda config: expected,
    )

    def forbidden_prepare(config):
        raise AssertionError("check-only called prepare_models")

    monkeypatch.setattr("merge_sft_for_grpo.prepare_models", forbidden_prepare)

    main(
        [
            "--base-model-path",
            str(tmp_path / "base"),
            "--adapter-path",
            str(tmp_path / "adapter"),
            "--output-root",
            str(tmp_path / "prepared"),
            "--expected-base-sha256",
            "a" * 64,
            "--expected-adapter-sha256",
            "b" * 64,
            "--validation-prompt-file",
            str(tmp_path / "prompt.txt"),
            "--check-only",
        ]
    )

    assert json.loads(capsys.readouterr().out) == expected


def test_check_only_rejects_non_cuda_runtime(tmp_path):
    """A CPU-only interpreter must not be declared ready for the BF16 merge."""

    config = make_preparation_config(tmp_path)
    runtime = RuntimeEnvironment(
        executable="/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-sft/bin/python",
        python_version="3.10.16",
        torch_version="2.5.1+cu124",
        cuda_version=None,
        cuda_available=False,
        device_name=None,
        compute_capability=None,
        bfloat16_supported=False,
        total_gpu_memory_bytes=None,
        library_versions={
            "torch": "2.5.1+cu124",
            "transformers": "4.57.1",
            "peft": "0.17.1",
            "accelerate": "1.10.1",
            "safetensors": "0.8.0",
        },
    )

    with pytest.raises(ModelPreparationError, match="CUDA is not available"):
        check_preparation(config, runtime_probe=lambda: runtime)


def test_original_parser_accepts_only_the_required_backslash_protocol():
    """Lock the exact protocol that previously caused failed Guardian retries."""

    response = "\n".join(
        [
            r"<Think> The order-status lookup is harmless. <\Think>",
            r"<Malicious_User_Request> no <\Malicious_User_Request>",
            r"<Being_Attacked> no <\Being_Attacked>",
            r"<Harmfulness_Rating> 0.0 <\Harmfulness_Rating>",
        ]
    )

    assert _parse_guardian_response(response) == {
        "Malicious_User_Request": "no",
        "Being_Attacked": "no",
        "Harmfulness_Rating": 0.0,
    }


def test_greedy_generation_kwargs_override_model_sampling_defaults():
    """Direct kwargs must prevent Qwen's sampling defaults from being restored."""

    class TokenizerStub:
        pad_token_id = None
        eos_token_id = 151645

    kwargs = _greedy_generation_kwargs(TokenizerStub(), max_new_tokens=128)

    assert kwargs == {
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "pad_token_id": 151645,
        "eos_token_id": 151645,
    }


def test_independent_reload_rejects_peft_parameters_and_files():
    """A supposedly dense checkpoint must not retain any LoRA artifacts."""

    with pytest.raises(ModelPreparationError, match="PEFT/LoRA artifacts"):
        assert_no_peft_artifacts(
            parameter_names=["model.layers.0.self_attn.q_proj.lora_A.default.weight"],
            relative_file_paths=["config.json", "adapter_config.json"],
        )


def test_independent_reload_can_force_eager_attention(tmp_path):
    """The diagnostic must control Attention implementation as one variable."""

    kwargs = _build_model_load_kwargs(
        tmp_path / "policy_init",
        attention_implementation="eager",
    )

    assert kwargs["attn_implementation"] == "eager"
    assert kwargs["local_files_only"] is True


def test_independent_reload_exposes_deterministic_algorithms_flag(tmp_path):
    """CUDA determinism must be an explicit, auditable diagnostic setting."""

    args = parse_validation_args(
        [
            "--output-root",
            str(tmp_path / "prepared"),
            "--validation-prompt-file",
            str(tmp_path / "prompt.txt"),
            "--deterministic-algorithms",
        ]
    )

    assert args.deterministic_algorithms is True


def test_independent_reload_requires_identical_policy_reference_behavior():
    """Byte-identical checkpoints must reload to the same safety response."""

    policy = ModelObservation(
        token_ids=[10, 11],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )
    reference = ModelObservation(
        token_ids=[10, 11],
        logits=torch.tensor([1.0, 2.0]),
        parsed_judgment=SAFE_JUDGMENT,
    )

    result = validate_reload_observations(policy, reference)

    assert result == {
        "generated_token_ids_equal": True,
        "parsed_judgment": SAFE_JUDGMENT,
    }

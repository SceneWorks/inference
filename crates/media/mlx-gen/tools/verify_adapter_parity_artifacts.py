#!/usr/bin/env python3
"""Verify the provenance-locked sc-15505 real-weight artifact set."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path, PurePosixPath

from _adapter_parity_provenance import (
    MFLUX_REPOSITORY,
    MFLUX_REVISION,
    assert_hf_file,
    assert_hf_snapshot,
    sha256,
)
from record_adapter_parity_transcript import (
    execution_metadata,
    expected_runs,
    model_inventories,
    parsed_results,
    proof_environment,
    receipt_for,
    source_state,
)

TOOLS = Path(__file__).resolve().parent
DEFAULT_MANIFEST = TOOLS / "adapter_parity_artifacts.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ARTIFACTS = {
    "hyper_flux_lora",
    "flux_hyper_golden",
    "z_image_base_golden",
    "z_image_lora_adapter",
    "z_image_lora_golden",
    "z_image_lokr_adapter",
    "z_image_lokr_golden",
    "qwen_base_golden",
    "qwen_lora_adapter",
    "qwen_lora_golden",
    "qwen_lokr_adapter",
    "qwen_lokr_golden",
}
REQUIRED_VALIDATION_MODELS = {"flux_dev", "z_image_turbo", "qwen_image"}
REQUIRED_PARITY_RESULTS = {
    "z_image_lora",
    "z_image_lokr",
    "qwen_lora",
    "qwen_lokr",
}
REFERENCE_MATCH_FIELDS = {
    "reference_mflux_repository",
    "reference_mflux_revision",
    "reference_model_repository",
    "reference_model_revision",
    "reference_model_ref",
    "reference_model_subdirectory",
    "reference_model_path",
    "reference_model_inventory_sha256",
    "reference_provenance_sha256",
    "reference_runtime",
}
Z_IMAGE_BEHAVIOR_FIELDS = REFERENCE_MATCH_FIELDS | {
    "prompt",
    "seed",
    "steps",
    "w",
    "h",
    "num_valid",
}
QWEN_BEHAVIOR_FIELDS = REFERENCE_MATCH_FIELDS | {
    "prompt",
    "seed",
    "steps",
    "width",
    "height",
    "guidance",
}


class InvalidManifest(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidManifest(message)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expanded_path(value: str, tools: Path) -> Path:
    return Path(os.path.expandvars(value.replace("${TOOLS}", str(tools)))).expanduser()


def _recorded_reference_path(value: str) -> PurePosixPath:
    """Normalize a recorded *reference-host* path for shape-independent comparison.

    Golden provenance embeds the absolute model path of the single host that dumped
    it (see `tools/golden/README.md` — parity goldens are single-host, deliberately).
    The check this feeds asks whether a golden was dumped against the model directory
    the manifest names; it must never ask whether that path exists on the *verifying*
    host. Resolving either side (`Path(...).absolute()` / `.resolve()` / `expanduser()`)
    does exactly that: on Windows it re-roots the recorded POSIX path onto the current
    drive and flips the separators, turning an identity compare into a permanent false
    failure that hides real provenance drift. Compare the recorded strings as opaque
    values, normalized only for cosmetic separator noise.
    """
    return PurePosixPath(value)


def _safetensors_metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        _require(len(raw_length) == 8, f"{path.name}: truncated safetensors header")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    metadata = header.get("__metadata__", {})
    _require(type(metadata) is dict, f"{path.name}: invalid safetensors metadata")
    return metadata


def verify_matching_generation_metadata(
    base: dict[str, str],
    adapter: dict[str, str],
    fields: set[str],
    label: str,
) -> None:
    for field in sorted(fields):
        _require(field in base, f"{label}: base golden missing metadata {field}")
        _require(field in adapter, f"{label}: adapter golden missing metadata {field}")
        _require(
            base[field] == adapter[field],
            f"{label}: base/adapter metadata mismatch for {field}",
        )


def verify_base_adapter_metadata(manifest: dict, tools: Path = TOOLS) -> None:
    paths = {
        name: _expanded_path(record["local_path"], tools)
        for name, record in manifest["artifacts"].items()
    }
    pairs = (
        (
            "z_image_lora",
            paths["z_image_base_golden"],
            paths["z_image_lora_golden"],
            Z_IMAGE_BEHAVIOR_FIELDS,
        ),
        (
            "z_image_lokr",
            paths["z_image_base_golden"],
            paths["z_image_lokr_golden"],
            Z_IMAGE_BEHAVIOR_FIELDS,
        ),
        (
            "qwen_lora",
            paths["qwen_base_golden"],
            paths["qwen_lora_golden"],
            QWEN_BEHAVIOR_FIELDS,
        ),
        (
            "qwen_lokr",
            paths["qwen_base_golden"],
            paths["qwen_lokr_golden"],
            QWEN_BEHAVIOR_FIELDS,
        ),
    )
    for label, base_path, adapter_path, fields in pairs:
        verify_matching_generation_metadata(
            _safetensors_metadata(base_path),
            _safetensors_metadata(adapter_path),
            fields,
            label,
        )


def validate_manifest(manifest: dict, tools: Path = TOOLS) -> None:
    _require(manifest.get("schema") == 1, "schema must be 1")
    _require(manifest.get("story") == "sc-15505", "story must be sc-15505")
    _require(
        manifest.get("implementation_base") == "8b1f1bdb37e449778d3a2110425dbe1ec5cb0c8b",
        "wrong inference implementation base",
    )
    reference = manifest.get("reference", {})
    _require(reference.get("repository") == MFLUX_REPOSITORY, "wrong mflux repository")
    _require(reference.get("revision") == MFLUX_REVISION, "wrong mflux revision")
    runtime = reference.get("runtime", {})
    for name in ("python", "mlx", "mlx-metal", "diffusers", "transformers", "torch", "peft"):
        _require(bool(runtime.get(name)), f"missing reference runtime: {name}")
    models = manifest.get("validation_models", {})
    _require(set(models) == REQUIRED_VALIDATION_MODELS, "validation model inventory mismatch")
    for name, model in models.items():
        _require(bool(model.get("repository")), f"{name}: missing model repository")
        _require(
            re.fullmatch(r"^[0-9a-f]{40}$", model.get("revision", "")) is not None,
            f"{name}: invalid model revision",
        )
        _require(bool(model.get("subdirectory")), f"{name}: missing model subdirectory")
        _require(bool(model.get("reference")), f"{name}: missing model reference")
        _require(bool(model.get("license")), f"{name}: missing model license")
        _require(bool(model.get("snapshot_path")), f"{name}: missing snapshot path")
        _require(
            HEX64.fullmatch(model.get("inventory_sha256", "")) is not None,
            f"{name}: invalid snapshot inventory",
        )
    results = manifest.get("results", {})
    evidence = results.get("evidence", {})
    evidence_status = evidence.get("status")
    _require(
        evidence_status in {"verified", "diagnostic_pending", "acceptance_pending"},
        "invalid evidence status",
    )
    transcript = evidence.get("transcript", {})
    _require(bool(transcript.get("local_path")), "missing result transcript path")
    receipt = evidence.get("receipt", {})
    _require(bool(receipt.get("local_path")), "missing durable receipt path")
    if evidence_status == "verified":
        _require(
            type(transcript.get("bytes")) is int and transcript["bytes"] > 0,
            "invalid result transcript bytes",
        )
        _require(
            HEX64.fullmatch(transcript.get("sha256", "")) is not None,
            "invalid result transcript sha256",
        )
        _require(
            type(receipt.get("bytes")) is int and receipt["bytes"] > 0,
            "invalid durable receipt bytes",
        )
        _require(
            HEX64.fullmatch(receipt.get("sha256", "")) is not None,
            "invalid durable receipt sha256",
        )
    else:
        _require(bool(evidence.get("pending_reason")), "missing diagnostic pending reason")
        for label, record in (("transcript", transcript), ("receipt", receipt)):
            _require(record.get("bytes") == -1, f"pending {label} must not claim byte size")
            _require(record.get("sha256") == "PENDING", f"pending {label} must not claim hash")
    hyper = results.get("hyper_flux_scale_zero", {})
    _require(hyper.get("byte_differences") == 0, "Hyper-FLUX scale-zero is not bit-exact")
    parity = results.get("fork_parity", {})
    _require(set(parity) == REQUIRED_PARITY_RESULTS, "fork parity result inventory mismatch")
    for name, result in parity.items():
        common_fields = {"samples_gt8", "base_floor", "cap", "rgb_samples"}
        provider_fields = (
            {"residual_samples_gt8", "zero_residual_samples_gt8", "residual_cap"}
            if name.startswith("z_image_")
            else {"effect_gate"}
        )
        _require(
            set(result) == common_fields | provider_fields,
            f"{name}: provider-specific result schema mismatch",
        )
        for key in common_fields:
            _require(
                type(result.get(key)) is int and result[key] >= 0,
                f"{name}: invalid result {key}",
            )
        _require(result["rgb_samples"] > 0, f"{name}: empty result")
        _require(result["samples_gt8"] <= result["cap"], f"{name}: fork parity failed")
        _require(
            result["cap"] == result["base_floor"] * 2 + result["rgb_samples"] // 200,
            f"{name}: fork cap is not the exact floor-relative formula",
        )
        if name.startswith("z_image_"):
            for key in (
                "residual_samples_gt8",
                "zero_residual_samples_gt8",
                "residual_cap",
            ):
                _require(
                    type(result.get(key)) is int and result[key] >= 0,
                    f"{name}: invalid residual result {key}",
                )
            _require(
                result["residual_samples_gt8"] <= result["residual_cap"]
                < result["zero_residual_samples_gt8"],
                f"{name}: residual mutation gate failed",
            )
            _require(
                result["residual_cap"]
                == (
                    result["residual_samples_gt8"]
                    + result["zero_residual_samples_gt8"]
                )
                // 2,
                f"{name}: residual cap is not the locked midpoint",
            )
            continue

        gate = result["effect_gate"]
        expected_applied = 24 if name == "qwen_lora" else 21
        common_gate = {
            "status",
            "minimum_rule",
            "expected_applied",
            "expected_unmatched",
            "expected_scale_zero_byte_differences",
        }
        _require(type(gate) is dict, f"{name}: invalid effect gate")
        _require(gate.get("minimum_rule") == "measured_effect_samples_gt8//2", f"{name}: bad effect rule")
        _require(gate.get("expected_applied") == expected_applied, f"{name}: bad applied count")
        _require(gate.get("expected_unmatched") == 0, f"{name}: unmatched paths must be zero")
        _require(
            gate.get("expected_scale_zero_byte_differences") == 0,
            f"{name}: scale-zero must be byte-exact",
        )
        if evidence_status == "diagnostic_pending":
            _require(set(gate) == common_gate, f"{name}: pending effect gate fabricates evidence")
            _require(gate.get("status") == "diagnostic_pending", f"{name}: effect gate not pending")
        else:
            locked_fields = common_gate | {
                "effect_samples_gt8",
                "minimum_samples_gt8",
            }
            _require(set(gate) == locked_fields, f"{name}: locked effect schema mismatch")
            _require(gate.get("status") == "locked", f"{name}: effect gate is not locked")
            effect = gate.get("effect_samples_gt8")
            minimum = gate.get("minimum_samples_gt8")
            _require(type(effect) is int and effect > 0, f"{name}: invalid measured effect")
            _require(type(minimum) is int and minimum > 0, f"{name}: invalid effect floor")
            _require(minimum == effect // 2, f"{name}: effect floor is not E//2")
    scripts = manifest.get("scripts", {})
    _require(bool(scripts), "scripts map is empty")
    for relative, expected in scripts.items():
        _require(HEX64.fullmatch(expected or "") is not None, f"invalid script hash: {relative}")
        path = tools / relative
        _require(path.is_file(), f"missing script: {relative}")
        _require(sha256(path) == expected, f"script hash mismatch: {relative}")

    artifacts = manifest.get("artifacts", {})
    _require(set(artifacts) == REQUIRED_ARTIFACTS, "artifact inventory is incomplete or expanded")
    for name, record in artifacts.items():
        _require(record.get("committed") is False, f"{name}: binaries must stay uncommitted")
        _require(type(record.get("bytes")) is int and record["bytes"] > 0, f"{name}: invalid bytes")
        _require(
            HEX64.fullmatch(record.get("sha256", "")) is not None,
            f"{name}: invalid sha256",
        )
        _require(bool(record.get("license")), f"{name}: missing license")
        source = record.get("source")
        _require(type(source) is dict, f"{name}: missing source")
        _require(source.get("kind") in {"huggingface", "generated"}, f"{name}: invalid source kind")
        _require(bool(source.get("repository")), f"{name}: missing source repository")
        _require(
            re.fullmatch(r"^[0-9a-f]{40}$", source.get("revision", "")) is not None,
            f"{name}: invalid source revision",
        )
        if source["kind"] == "huggingface":
            _require(bool(source.get("file")), f"{name}: missing source file")
            _require(bool(source.get("reference")), f"{name}: missing source reference")
        else:
            _require(source.get("script") in scripts, f"{name}: unpinned source script")
            _require(bool(source.get("command")), f"{name}: missing generation command")
            _require(bool(source.get("model_repository")), f"{name}: missing model repository")
            _require(
                re.fullmatch(r"^[0-9a-f]{40}$", source.get("model_revision", "")) is not None,
                f"{name}: invalid model revision",
            )
            _require(bool(source.get("model_path")), f"{name}: missing model path")
            _require(bool(source.get("model_reference")), f"{name}: missing model reference")
            _require(bool(source.get("model_subdirectory")), f"{name}: missing model subdirectory")
            _require(
                HEX64.fullmatch(source.get("model_inventory_sha256", "")) is not None,
                f"{name}: invalid model inventory",
            )
        _require(bool(record.get("local_path")), f"{name}: missing local_path")
    bound_hashes = evidence.get("artifact_sha256", {})
    _require(set(bound_hashes) == REQUIRED_ARTIFACTS, "result artifact hash inventory mismatch")
    for name, expected in bound_hashes.items():
        _require(expected == artifacts[name]["sha256"], f"{name}: result evidence hash mismatch")


def verify_artifact_files(manifest: dict, tools: Path = TOOLS) -> None:
    for name, record in manifest["artifacts"].items():
        path = _expanded_path(record["local_path"], tools)
        if not path.is_file():
            raise InvalidManifest(f"{name}: missing artifact {path}")
        if path.stat().st_size != record["bytes"]:
            raise InvalidManifest(f"{name}: byte size mismatch")
        if sha256(path) != record["sha256"]:
            raise InvalidManifest(f"{name}: sha256 mismatch")


def verify_generated_metadata(manifest: dict, name: str, record: dict, path: Path) -> None:
    source = record["source"]
    metadata = _safetensors_metadata(path)
    _require(
        metadata.get("reference_mflux_repository") == manifest["reference"]["repository"],
        f"{name}: golden mflux repository mismatch",
    )
    _require(
        metadata.get("reference_mflux_revision") == manifest["reference"]["revision"],
        f"{name}: golden mflux revision mismatch",
    )
    _require(
        metadata.get("reference_script_sha256") == manifest["scripts"][source["script"]],
        f"{name}: golden script hash mismatch",
    )
    _require(
        metadata.get("reference_provenance_sha256")
        == manifest["scripts"]["_adapter_parity_provenance.py"],
        f"{name}: provenance helper hash mismatch",
    )
    for key in ("repository", "revision", "subdirectory", "reference"):
        metadata_key = "ref" if key == "reference" else key
        _require(
            metadata.get(f"reference_model_{metadata_key}") == source[f"model_{key}"],
            f"{name}: golden model {key} mismatch",
        )
    _require(
        _recorded_reference_path(metadata.get("reference_model_path", ""))
        == _recorded_reference_path(source["model_path"]),
        f"{name}: golden model path mismatch",
    )
    _require(
        metadata.get("reference_model_inventory_sha256")
        == source["model_inventory_sha256"],
        f"{name}: golden model inventory mismatch",
    )
    recorded_runtime = json.loads(metadata.get("reference_runtime", "{}"))
    for package, expected in manifest["reference"]["runtime"].items():
        _require(
            recorded_runtime.get(package) == expected,
            f"{name}: golden runtime mismatch for {package}",
        )
    if name in {
        "z_image_lora_golden",
        "z_image_lokr_golden",
        "qwen_lora_golden",
        "qwen_lokr_golden",
    }:
        adapter = name.removesuffix("_golden") + "_adapter"
        _require(
            metadata.get("adapter_sha256") == manifest["artifacts"][adapter]["sha256"],
            f"{name}: golden adapter hash mismatch",
        )
    if name == "flux_hyper_golden":
        _require(
            metadata.get("lora_sha256") == manifest["artifacts"]["hyper_flux_lora"]["sha256"],
            f"{name}: golden Hyper LoRA hash mismatch",
        )
    if name.endswith("_adapter"):
        expected_kind = "lokr" if "_lokr_" in name else "lora"
        _require(metadata.get("artifact_role") == "adapter", f"{name}: adapter role mismatch")
        _require(
            metadata.get("adapter_kind") == expected_kind,
            f"{name}: adapter kind mismatch",
        )


def verify_artifacts(manifest: dict, tools: Path = TOOLS) -> None:
    verify_artifact_files(manifest, tools)
    for name, record in manifest["artifacts"].items():
        path = _expanded_path(record["local_path"], tools)
        source = record["source"]
        if source["kind"] == "huggingface":
            assert_hf_file(
                path,
                repository=source["repository"],
                revision=source["revision"],
                file=source["file"],
                reference=source["reference"],
            )
        else:
            _, inventory = assert_hf_snapshot(
                source["model_path"],
                repository=source["model_repository"],
                revision=source["model_revision"],
                subdirectory=source["model_subdirectory"],
                reference=source["model_reference"],
            )
            _require(inventory == source["model_inventory_sha256"], f"{name}: model inventory drift")
            verify_generated_metadata(manifest, name, record, path)
    verify_base_adapter_metadata(manifest, tools)

    for name, model in manifest["validation_models"].items():
        _, inventory = assert_hf_snapshot(
            model["snapshot_path"],
            repository=model["repository"],
            revision=model["revision"],
            subdirectory=model["subdirectory"],
            reference=model["reference"],
        )
        _require(inventory == model["inventory_sha256"], f"{name}: validation model inventory drift")
    verify_result_transcript(manifest, tools)


def expected_result_measurements(manifest: dict) -> dict[str, dict[str, int]]:
    expected = {
        "hyper_flux_scale_zero": manifest["results"]["hyper_flux_scale_zero"],
    }
    for name, result in manifest["results"]["fork_parity"].items():
        if name.startswith("z_image_"):
            expected[name] = result
            continue
        gate = result["effect_gate"]
        expected[name] = {
            "samples_gt8": result["samples_gt8"],
            "base_floor": result["base_floor"],
            "cap": result["cap"],
            "effect_samples_gt8": gate["effect_samples_gt8"],
            "minimum_samples_gt8": gate["minimum_samples_gt8"],
            "scale_zero_byte_differences": gate[
                "expected_scale_zero_byte_differences"
            ],
            "applied": gate["expected_applied"],
            "unmatched": gate["expected_unmatched"],
            "rgb_samples": result["rgb_samples"],
        }
    return expected


def verify_result_transcript(manifest: dict, tools: Path = TOOLS) -> None:
    transcript_record = manifest["results"]["evidence"]["transcript"]
    path = _expanded_path(transcript_record["local_path"], tools)
    _require(path.is_file(), f"missing result transcript: {path}")
    _require(path.stat().st_size == transcript_record["bytes"], "result transcript byte size mismatch")
    _require(sha256(path) == transcript_record["sha256"], "result transcript sha256 mismatch")
    transcript = json.loads(path.read_text(encoding="utf-8"))
    _require(transcript.get("schema") == 2, "result transcript schema mismatch")
    _require(transcript.get("story") == manifest["story"], "result transcript story mismatch")
    expected_artifacts = manifest["results"]["evidence"]["artifact_sha256"]
    _require(
        transcript.get("artifacts_before") == expected_artifacts,
        "result transcript pre-run artifact hashes mismatch",
    )
    _require(
        transcript.get("artifacts_after") == expected_artifacts,
        "result transcript post-run artifact hashes mismatch",
    )
    expected_models = model_inventories(manifest)
    _require(transcript.get("models_before") == expected_models, "pre-run model inventory mismatch")
    _require(transcript.get("models_after") == expected_models, "post-run model inventory mismatch")
    recorded_source = transcript.get("source", {})
    recorded_source_after = transcript.get("source_after", {})
    _require(recorded_source == recorded_source_after, "source changed during result run")
    _require(
        recorded_source.get("commit") == manifest["implementation_base"],
        "result transcript source commit mismatch",
    )
    current_source = source_state(manifest)
    for key in ("base_commit", "source_sha256", "files"):
        _require(
            recorded_source.get(key) == current_source.get(key),
            f"result transcript source {key} mismatch",
        )
    expected_execution = execution_metadata(proof_environment())
    _require(
        transcript.get("execution") == expected_execution,
        "result transcript execution environment mismatch",
    )
    _require(
        transcript.get("execution_after") == expected_execution,
        "post-run execution environment mismatch",
    )
    runs = transcript.get("runs", [])
    expected_specs = expected_runs(manifest)
    _require(
        [run.get("name") for run in runs] == [run["name"] for run in expected_specs],
        "result transcript run inventory mismatch",
    )
    combined = {}
    for run, expected_spec in zip(runs, expected_specs):
        _require(run.get("argv") == expected_spec["argv"], f"{run.get('name')}: argv mismatch")
        _require(run.get("env") == expected_spec["env"], f"{run.get('name')}: env mismatch")
        _require(run.get("returncode") == 0, f"{run.get('name')}: recorded nonzero exit")
        output = run.get("stdout", "") + run.get("stderr", "")
        _require("test result: ok." in output, f"{run.get('name')}: missing passing cargo result")
        try:
            run_results = parsed_results(output)
        except ValueError as error:
            raise InvalidManifest(str(error)) from error
        _require(set(run_results) == {run["name"]}, f"{run.get('name')}: exact result missing")
        combined.update(run_results)
    expected = expected_result_measurements(manifest)
    _require(transcript.get("results") == combined, "recorded result summary mismatch")
    _require(combined == expected, "result transcript measurements mismatch")
    receipt_record = manifest["results"]["evidence"]["receipt"]
    receipt_path = _expanded_path(receipt_record["local_path"], tools)
    _require(receipt_path.is_file(), f"missing durable receipt: {receipt_path}")
    _require(receipt_path.stat().st_size == receipt_record["bytes"], "durable receipt byte size mismatch")
    _require(sha256(receipt_path) == receipt_record["sha256"], "durable receipt sha256 mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt == receipt_for(transcript, path), "durable receipt does not match transcript")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="validate tracked provenance and dump-script hashes without gitignored binaries",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest, args.manifest.resolve().parent)
    if not args.manifest_only:
        _require(
            manifest["results"]["evidence"]["status"] == "verified",
            "results remain non-proof until the final acceptance transcript and receipt are verified",
        )
        verify_artifacts(manifest, args.manifest.resolve().parent)
    print("adapter parity artifact provenance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

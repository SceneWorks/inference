#!/usr/bin/env python3
"""Execute and retain the artifact-bound sc-15505 real-weight proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from _adapter_parity_provenance import assert_hf_snapshot, sha256

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[3]
MANIFEST = TOOLS / "adapter_parity_artifacts.json"
TARGET_DIR = "/private/tmp/codex-sc-15505-release-target"
RECEIPT = TOOLS / "adapter_parity_receipt.json"
ACCEPTANCE_TRANSCRIPT = TOOLS / "golden/sc-15505-real-weight-transcript.json"
DIAGNOSTIC_TRANSCRIPT = TOOLS / "golden/sc-15505-residual-diagnostic-transcript.json"
QWEN_EFFECT_TRANSCRIPT = TOOLS / "golden/sc-15505-qwen-effect-diagnostic-transcript.json"
RESULT_LINE = re.compile(
    r"^SC15505_RESULT (?P<name>[a-z0-9_]+) "
    r"(?P<fields>(?:[a-z0-9_]+=[0-9]+ ?)+)$",
    re.MULTILINE,
)
RUST_SOURCE_FILES = (
    "crates/media/mlx-gen/src/adapters.rs",
    "crates/media/mlx-gen/mlx-gen-flux/tests/hyper_flux_real_weights.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/tests/adapter_real_weights.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image/tests/adapter_real_weights.rs",
)
EVIDENCE_CHANGE_FILES = {
    "crates/media/mlx-gen/tools/adapter_parity_artifacts.json",
    "crates/media/mlx-gen/tools/adapter_parity_receipt.json",
    "crates/media/mlx-gen/tools/golden/CHECKSUMS.txt",
    "crates/media/mlx-gen/tools/golden/README.md",
    "scripts/tests/test_adapter_parity_artifacts.py",
}
RESIDUAL_FIELDS = {
    "residual_samples_gt8",
    "zero_residual_samples_gt8",
    "rgb_samples",
}
RESIDUAL_RESULTS_BY_RUN = {
    "z_image_residual_diagnostic": {"z_image_lora", "z_image_lokr"},
    "qwen_residual_diagnostic": {"qwen_lora", "qwen_lokr"},
}
QWEN_EFFECT_FIELDS = {
    "effect_samples_gt8",
    "scale_zero_byte_differences",
    "applied",
    "unmatched",
    "rgb_samples",
}
QWEN_EFFECT_RESULTS = {"qwen_lora", "qwen_lokr"}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(value.replace("${TOOLS}", str(TOOLS)))).expanduser()


def artifact_hashes(manifest: dict) -> dict[str, str]:
    actual = {}
    for name, record in manifest["artifacts"].items():
        path = expanded_path(record["local_path"])
        if not path.is_file():
            raise RuntimeError(f"{name}: missing artifact {path}")
        if path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"{name}: artifact size differs from manifest")
        digest = sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"{name}: artifact hash differs from manifest")
        actual[name] = digest
    return dict(sorted(actual.items()))


def model_inventories(manifest: dict) -> dict[str, dict[str, str]]:
    actual = {}
    for name, model in manifest["validation_models"].items():
        path, inventory = assert_hf_snapshot(
            model["snapshot_path"],
            repository=model["repository"],
            revision=model["revision"],
            subdirectory=model["subdirectory"],
            reference=model["reference"],
        )
        if inventory != model["inventory_sha256"]:
            raise RuntimeError(f"{name}: validation model inventory differs from manifest")
        actual[name] = {
            "repository": model["repository"],
            "revision": model["revision"],
            "reference": model["reference"],
            "subdirectory": model["subdirectory"],
            "snapshot_path": path,
            "inventory_sha256": inventory,
        }
    return dict(sorted(actual.items()))


def _git_paths(root: Path, argv: list[str]) -> set[str]:
    output = subprocess.run(
        ["git", "-C", str(root), *argv],
        check=True,
        capture_output=True,
    ).stdout
    return {
        value.decode("utf-8")
        for value in output.split(b"\0")
        if value
    }


def source_state(
    manifest: dict,
    *,
    root: Path = ROOT,
    source_files: tuple[str, ...] | None = None,
    permitted_changes: set[str] | None = None,
) -> dict:
    files = source_files or tuple(
        sorted(
            {
                *RUST_SOURCE_FILES,
                *(f"crates/media/mlx-gen/tools/{name}" for name in manifest["scripts"]),
            }
        )
    )
    base_commit = manifest["implementation_base"]
    changed_paths = _git_paths(
        root,
        ["diff", "--name-only", "-z", base_commit, "--"],
    ) | _git_paths(root, ["ls-files", "--others", "--exclude-standard", "-z"])
    allowed = permitted_changes or (set(files) | EVIDENCE_CHANGE_FILES)
    unexpected = sorted(changed_paths - allowed)
    if unexpected:
        raise RuntimeError(
            "proof worktree contains changes outside the bound allowlist: "
            + ", ".join(unexpected)
        )

    hashes = {name: sha256(root / name) for name in files}
    canonical = json.dumps(
        {
            "base_commit": base_commit,
            "files": hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    return {
        "commit": commit,
        "base_commit": base_commit,
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": hashes,
        "changed_paths": sorted(changed_paths),
    }


def proof_environment() -> dict[str, str]:
    home = str(Path.home())
    return {
        "CARGO_HOME": f"{home}/.cargo",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TERM_COLOR": "never",
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": (
            f"{home}/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:"
            "/usr/bin:/bin:/usr/sbin:/sbin"
        ),
        "RUSTUP_HOME": f"{home}/.rustup",
        "RUSTUP_TOOLCHAIN": "1.96.0",
        "TMPDIR": "/private/tmp",
    }


def execution_metadata(environment: dict[str, str]) -> dict:
    def output(argv: list[str]) -> str:
        return subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
        ).stdout.strip()

    return {
        "cargo": output(["cargo", "-V"]),
        "rustc": output(["rustc", "-Vv"]),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
    }


def expected_runs(manifest: dict) -> list[dict]:
    models = manifest["validation_models"]
    hyper = str(expanded_path(manifest["artifacts"]["hyper_flux_lora"]["local_path"]))
    specs = (
        (
            "hyper_flux_scale_zero",
            "mlx-gen-flux",
            "hyper_flux_real_weights",
            "hyper_flux_scale_zero_is_bit_exact_noop",
            {
                "MLX_GEN_FLUX_DEV_SNAPSHOT": models["flux_dev"]["snapshot_path"],
                "HYPER_LORA": hyper,
            },
        ),
        (
            "z_image_lora",
            "mlx-gen-z-image",
            "adapter_real_weights",
            "lora_render_matches_fork_golden",
            {"ZIMAGE_SNAPSHOT": models["z_image_turbo"]["snapshot_path"]},
        ),
        (
            "z_image_lokr",
            "mlx-gen-z-image",
            "adapter_real_weights",
            "lokr_render_matches_fork_golden",
            {"ZIMAGE_SNAPSHOT": models["z_image_turbo"]["snapshot_path"]},
        ),
        (
            "qwen_lora",
            "mlx-gen-qwen-image",
            "adapter_real_weights",
            "lora_render_matches_fork_golden",
            {"MLX_GEN_QWEN_SNAPSHOT": models["qwen_image"]["snapshot_path"]},
        ),
        (
            "qwen_lokr",
            "mlx-gen-qwen-image",
            "adapter_real_weights",
            "lokr_render_matches_fork_golden",
            {"MLX_GEN_QWEN_SNAPSHOT": models["qwen_image"]["snapshot_path"]},
        ),
    )
    runs = []
    for name, package, test_binary, test_name, environment in specs:
        runs.append(
            {
                "name": name,
                "argv": [
                    "cargo",
                    "test",
                    "--locked",
                    "--release",
                    "-p",
                    package,
                    "--test",
                    test_binary,
                    test_name,
                    "--",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                    "--exact",
                ],
                "env": {
                    **proof_environment(),
                    "CARGO_TARGET_DIR": TARGET_DIR,
                    **environment,
                },
            }
        )
    return runs


def residual_diagnostic_runs(manifest: dict) -> list[dict]:
    models = manifest["validation_models"]
    specs = (
        (
            "z_image_residual_diagnostic",
            "mlx-gen-z-image",
            {"ZIMAGE_SNAPSHOT": models["z_image_turbo"]["snapshot_path"]},
        ),
        (
            "qwen_residual_diagnostic",
            "mlx-gen-qwen-image",
            {"MLX_GEN_QWEN_SNAPSHOT": models["qwen_image"]["snapshot_path"]},
        ),
    )
    return [
        {
            "name": name,
            "argv": [
                "cargo",
                "test",
                "--locked",
                "--release",
                "-p",
                package,
                "--test",
                "adapter_real_weights",
                "residual_mutation_diagnostic",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
                "--exact",
            ],
            "env": {
                **proof_environment(),
                "CARGO_TARGET_DIR": TARGET_DIR,
                **environment,
            },
        }
        for name, package, environment in specs
    ]


def qwen_effect_diagnostic_runs(manifest: dict) -> list[dict]:
    return [
        {
            "name": "qwen_effect_diagnostic",
            "argv": [
                "cargo",
                "test",
                "--locked",
                "--release",
                "-p",
                "mlx-gen-qwen-image",
                "--test",
                "adapter_real_weights",
                "adapter_effect_diagnostic",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
                "--exact",
            ],
            "env": {
                **proof_environment(),
                "CARGO_TARGET_DIR": TARGET_DIR,
                "MLX_GEN_QWEN_SNAPSHOT": manifest["validation_models"]["qwen_image"][
                    "snapshot_path"
                ],
            },
        }
    ]


def parsed_results(output: str) -> dict[str, dict[str, int]]:
    parsed = {}
    for match in RESULT_LINE.finditer(output):
        name = match.group("name")
        if name in parsed:
            raise ValueError(f"duplicate result name: {name}")
        fields = {}
        for field in match.group("fields").split():
            key, value = field.split("=", 1)
            if key in fields:
                raise ValueError(f"{name}: duplicate result field: {key}")
            fields[key] = int(value)
        parsed[name] = fields
    return parsed


def validate_residual_run_results(run_name: str, results: dict[str, dict[str, int]]) -> None:
    expected_names = RESIDUAL_RESULTS_BY_RUN[run_name]
    if set(results) != expected_names:
        raise ValueError(
            f"{run_name}: result inventory mismatch: "
            f"expected {sorted(expected_names)}, got {sorted(results)}"
        )
    for name, fields in results.items():
        if set(fields) != RESIDUAL_FIELDS:
            raise ValueError(
                f"{run_name}/{name}: result field inventory mismatch: "
                f"expected {sorted(RESIDUAL_FIELDS)}, got {sorted(fields)}"
            )
        rgb_samples = fields["rgb_samples"]
        if rgb_samples <= 0:
            raise ValueError(f"{run_name}/{name}: rgb_samples must be nonzero")
        for field in ("residual_samples_gt8", "zero_residual_samples_gt8"):
            if not 0 <= fields[field] <= rgb_samples:
                raise ValueError(f"{run_name}/{name}: {field} is outside the RGB sample count")
        if fields["zero_residual_samples_gt8"] == 0:
            raise ValueError(f"{run_name}/{name}: zero-residual control is empty")


def validate_shared_residual_sample_count(results: dict[str, dict[str, int]]) -> None:
    counts = {fields["rgb_samples"] for fields in results.values()}
    if len(counts) != 1 or next(iter(counts), 0) <= 0:
        raise ValueError(f"residual diagnostics do not share one nonzero RGB sample count: {counts}")


def validate_qwen_effect_results(results: dict[str, dict[str, int]]) -> None:
    if set(results) != QWEN_EFFECT_RESULTS:
        raise ValueError(
            "qwen_effect_diagnostic: result inventory mismatch: "
            f"expected {sorted(QWEN_EFFECT_RESULTS)}, got {sorted(results)}"
        )
    expected_applied = {"qwen_lora": 24, "qwen_lokr": 21}
    counts = set()
    for name, fields in results.items():
        if set(fields) != QWEN_EFFECT_FIELDS:
            raise ValueError(
                f"qwen_effect_diagnostic/{name}: result field inventory mismatch: "
                f"expected {sorted(QWEN_EFFECT_FIELDS)}, got {sorted(fields)}"
            )
        rgb_samples = fields["rgb_samples"]
        counts.add(rgb_samples)
        if rgb_samples <= 0 or not 0 < fields["effect_samples_gt8"] <= rgb_samples:
            raise ValueError(f"qwen_effect_diagnostic/{name}: invalid RGB/effect sample count")
        if fields["scale_zero_byte_differences"] != 0:
            raise ValueError(f"qwen_effect_diagnostic/{name}: scale-zero is not bit-exact")
        if fields["applied"] != expected_applied[name]:
            raise ValueError(f"qwen_effect_diagnostic/{name}: applied module count mismatch")
        if fields["unmatched"] != 0:
            raise ValueError(f"qwen_effect_diagnostic/{name}: unmatched adapter paths")
    if len(counts) != 1:
        raise ValueError(
            f"qwen effect diagnostics do not share one nonzero RGB sample count: {counts}"
        )


def selected_mode(args: argparse.Namespace) -> str:
    if args.residual_diagnostic:
        return "residual_diagnostic"
    if args.qwen_effect_diagnostic:
        return "qwen_effect_diagnostic"
    return "acceptance"


def resolved_output(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    mode = selected_mode(args)
    dedicated = {
        "acceptance": ACCEPTANCE_TRANSCRIPT,
        "residual_diagnostic": DIAGNOSTIC_TRANSCRIPT,
        "qwen_effect_diagnostic": QWEN_EFFECT_TRANSCRIPT,
    }[mode]
    if args.output is not None and args.output.resolve() != dedicated.resolve():
        parser.error(
            f"{mode} output is fixed at {dedicated}; refusing reserved or arbitrary output "
            f"{args.output}"
        )
    return dedicated


def write_transcript(path: Path, transcript: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcript, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        replacements = (
            (str(ROOT), "${REPOSITORY}"),
            ("/Users/michael/.cache/huggingface/hub", "${HF_MODELS_ROOT}"),
            (TARGET_DIR, "${CARGO_TARGET_DIR}"),
            (str(Path.home()), "${HOME}"),
        )
        for actual, symbolic in replacements:
            value = value.replace(actual, symbolic)
    return value


def receipt_for(transcript: dict, transcript_path: Path) -> dict:
    return {
        "schema": 1,
        "story": "sc-15505",
        "transcript": {
            "local_path": "${TOOLS}/golden/sc-15505-real-weight-transcript.json",
            "bytes": transcript_path.stat().st_size,
            "sha256": sha256(transcript_path),
        },
        "proof": _redact(transcript),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--residual-diagnostic",
        action="store_true",
        help="measure residual separation without writing the durable acceptance receipt",
    )
    modes.add_argument(
        "--qwen-effect-diagnostic",
        action="store_true",
        help="measure same-runtime Qwen adapter effect without writing the acceptance receipt",
    )
    args = parser.parse_args()
    mode = selected_mode(args)
    is_diagnostic = mode != "acceptance"
    args.output = resolved_output(parser, args)
    manifest = load_manifest()
    environment = proof_environment()
    before_artifacts = artifact_hashes(manifest)
    before_models = model_inventories(manifest)
    before_source = source_state(manifest)
    before_execution = execution_metadata(environment)
    if before_source["commit"] != manifest["implementation_base"]:
        raise RuntimeError(
            "proof must run from the recorded implementation base plus the source hashes in "
            f"this transcript, got HEAD {before_source['commit']}"
        )
    hyper_test = ROOT / RUST_SOURCE_FILES[1]
    if "fn hyper_flux_scale_zero_is_bit_exact_noop()" not in hyper_test.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "run hyper_flux_scale_zero_is_near_noop first; only after byte_differences=0 "
            "may it be tightened/renamed for the retained proof"
        )
    transcript = {
        "schema": 2,
        "story": "sc-15505",
        "mode": mode,
        "source": before_source,
        "execution": before_execution,
        "models_before": before_models,
        "artifacts_before": before_artifacts,
        "runs": [],
    }
    parsed = {}
    failed = False
    run_specs = {
        "acceptance": expected_runs,
        "residual_diagnostic": residual_diagnostic_runs,
        "qwen_effect_diagnostic": qwen_effect_diagnostic_runs,
    }[mode](manifest)
    for spec in run_specs:
        result = subprocess.run(
            spec["argv"],
            cwd=ROOT,
            env=spec["env"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        run = {
            **spec,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        transcript["runs"].append(run)
        run_results = parsed_results(result.stdout + result.stderr)
        if mode == "residual_diagnostic":
            validate_residual_run_results(spec["name"], run_results)
        elif mode == "qwen_effect_diagnostic":
            validate_qwen_effect_results(run_results)
        duplicate_names = set(parsed).intersection(run_results)
        if duplicate_names:
            raise ValueError(f"duplicate result names across runs: {sorted(duplicate_names)}")
        parsed.update(run_results)
        if result.returncode != 0:
            failed = True
            break
    transcript["models_after"] = model_inventories(manifest)
    transcript["artifacts_after"] = artifact_hashes(manifest)
    transcript["source_after"] = source_state(manifest)
    transcript["execution_after"] = execution_metadata(environment)
    transcript["results"] = parsed
    if not is_diagnostic:
        # Acceptance failures are retained as diagnostics, but never produce a receipt.
        write_transcript(args.output, transcript)
    if transcript["artifacts_after"] != before_artifacts:
        raise RuntimeError("artifact hashes changed during the proof run")
    if transcript["models_after"] != before_models:
        raise RuntimeError("model inventories changed during the proof run")
    if transcript["source_after"] != before_source:
        raise RuntimeError("source files changed during the proof run")
    if transcript["execution_after"] != before_execution:
        raise RuntimeError("toolchain/platform/hardware changed during the proof run")
    expected_names = {
        "acceptance": {run["name"] for run in expected_runs(manifest)},
        "residual_diagnostic": set().union(*RESIDUAL_RESULTS_BY_RUN.values()),
        "qwen_effect_diagnostic": QWEN_EFFECT_RESULTS,
    }[mode]
    if failed or set(parsed) != expected_names:
        raise RuntimeError(
            f"real-weight proof failed or result inventory is incomplete: {sorted(parsed)}"
        )
    if is_diagnostic:
        if mode == "residual_diagnostic":
            validate_shared_residual_sample_count(parsed)
        else:
            validate_qwen_effect_results(parsed)
        write_transcript(args.output, transcript)
        print(f"wrote source/env-bound {mode} {args.output}")
        return 0
    RECEIPT.write_text(
        json.dumps(receipt_for(transcript, args.output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote passing artifact-bound transcript {args.output}")
    print(f"wrote tracked path-redacted receipt {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

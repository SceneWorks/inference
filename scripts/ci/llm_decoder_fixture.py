#!/usr/bin/env python3
"""Narrow explicitly dispatched shared-LLM decoder proofs, separate from MOSS's roster.

No weights/download/device knobs. Compilation is separate from the unchanged 180s/2GiB
owned-child limit. Reuse only the existing source-bound compiler/result/ownership helpers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import moss_fixture_cpu as shared
from scripts.ci.moss_cuda_unit import UnitFailure, supervise

LANE = "llm-decoder-fixture"
DISPATCH_CHOICES = ("moss-cpu-fixture", "moss-metal-fixture", "llm-decoder-fixture", "moss-cuda-unit")
DISPATCH_INPUT_CHOICES = {"lanes": DISPATCH_CHOICES}
IGNORE_REASON = "model inference; explicit model-proof workflow"
CASES = {
    "candle": (
        "models::qwen35::tests::headless_decoder_preserves_normalized_hidden_and_hybrid_cache",
        "models::qwen35::tests::headless_checked_callback_cancels_between_layers_and_resets_cache",
    ),
    "mlx": ("models::qwen35::tests::headless_decoder_preserves_normalized_hidden_and_hybrid_cache_on_cpu",),
}
PACKAGES = {"candle": "candle-llm", "mlx": "mlx-llm"}
LIBRARIES = {"candle": "candle_llm", "mlx": "mlx_llm"}
# The Rust source that declares each lane's decoder cases (MLX keeps its `qwen35` tests in a sibling
# file, sc-27409); the MLX decoder itself is stamped beside it in `venue`.
SOURCES = {
    "candle": "crates/llm/candle-llm/src/models/qwen35.rs",
    "mlx": "crates/llm/mlx-llm/src/models/qwen35/tests.rs",
}
COMPILER_SECONDS = 2700


def validate_dispatch(event_name: str, event: dict) -> str | None:
    """Bind the reusable caller's profile or this workflow's direct choice to one lane."""
    if event_name != "workflow_dispatch":
        return None
    inputs = event.get("inputs", {})
    if not isinstance(inputs, dict):
        raise UnitFailure("dispatch selectors are malformed")
    if "lanes" in inputs and "profile" in inputs:
        raise UnitFailure("select only one fixture profile")
    choice = inputs.get("lanes", inputs.get("profile"))
    if not isinstance(choice, str) or choice not in DISPATCH_CHOICES:
        raise UnitFailure("select exactly one declared fixture profile")
    if os.environ.get("MOSS_FIXTURE_SELECTED_LANE") != choice:
        raise UnitFailure("selected reusable-workflow profile differs from dispatch")
    return choice


def event_payload() -> dict:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    if not isinstance(event, dict):
        raise UnitFailure("dispatch payload is not an object")
    return event


def venue(backend: str) -> dict:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if validate_dispatch(event_name, event_payload()) != LANE:
        raise UnitFailure("shared decoder proofs require the explicit llm-decoder-fixture dispatch")
    stamp = shared.source_stamp()
    head = stamp["head"]
    workflow = os.environ.get("GITHUB_WORKFLOW_REF", "")
    if (event_name != "workflow_dispatch" or os.environ.get("GITHUB_REPOSITORY") != "SceneWorks/inference"
            or not re.fullmatch(r"[0-9a-f]{40}", head) or os.environ.get("GITHUB_SHA") != head
            or os.environ.get("GITHUB_WORKFLOW_SHA") != head
            or not workflow.startswith(("SceneWorks/inference/.github/workflows/moss-model-fixtures.yml@",
                                            "SceneWorks/inference/.github/workflows/real-weights.yml@"))
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or os.environ.get("RUNNER_OS") != ("macOS" if backend == "mlx" else "Linux")):
        raise UnitFailure("decoder proof source/workflow/host identity differs from the admitted narrow route")
    declaration = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    channel = re.search(r'^channel\s*=\s*"([^"]+)"', declaration, re.M)
    version = subprocess.check_output(["rustc", "--version"], text=True, encoding="utf-8").strip()
    if channel is None or not version.startswith("rustc " + channel[1] + " "):
        raise UnitFailure("decoder proof requires the repository-pinned Rust toolchain")
    paths = ["Cargo.lock", "rust-toolchain.toml", ".github/workflows/moss-model-fixtures.yml",
             "scripts/ci/llm_decoder_fixture.py", SOURCES[backend]]
    if backend == "mlx":
        paths += ["crates/llm/mlx-llm/src/models/qwen35.rs", "crates/llm/mlx-llm/src/primitives/kv_cache.rs"]
    return {"head": head, "repository": "SceneWorks/inference", "event": event_name,
            "workflow_ref": workflow, "workflow_sha": os.environ["GITHUB_WORKFLOW_SHA"],
            "dispatch_choice": LANE, "runner_os": os.environ["RUNNER_OS"], "toolchain": version,
            "source_and_pin_sha256": {path: shared.file_digest(ROOT / path) for path in paths},
            "device_contract": "literal generated CPU tensors" if backend == "candle" else "original on_cpu CPU-stream scope",
            "measured_model_peak_bytes": None}


def compile_binary(backend: str, directory: Path, stamp: dict) -> tuple[Path, dict]:
    command = ["cargo", "test", "--locked", "--lib", "-j", "1", "-p", PACKAGES[backend],
               "--no-run", "--message-format=json"]
    with (directory / "compile.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=log,
                                text=True, encoding="utf-8", timeout=COMPILER_SECONDS)
    cargo = directory / "compiler.cargo.jsonl"
    cargo.write_text(result.stdout, encoding="utf-8")
    if shared.source_stamp() != stamp:
        raise UnitFailure("decoder source changed during compilation")
    if result.returncode:
        shared.render_compile_errors(result.stdout)
        raise UnitFailure("decoder compile failed; compile failure is not model execution evidence")
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).resolve()
    binary = shared.select_binary(result.stdout, PACKAGES[backend], target, lib_name=LIBRARIES[backend])
    record = {"command": command, "source": stamp, "package": PACKAGES[backend], "target_kind": "lib",
              "target_name": LIBRARIES[backend], "binary": str(binary), "target": str(target),
              "binary_sha256": shared.file_digest(binary), "cargo_json_sha256": shared.file_digest(cargo)}
    (directory / "compiler.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return binary, record


def resources_verified(row: dict) -> bool:
    rss = row.get("observed_peak_child_rss_bytes")
    elapsed = row.get("observed_elapsed_seconds")
    return (type(rss) is int and 0 < rss <= shared.RSS_CAP
            and type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0
            and row.get("configured_rss_cap_bytes") == shared.RSS_CAP
            and row.get("configured_deadline_seconds") == shared.DEADLINE
            and row.get("owned_released") is True)


def run(backend: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    proof = {"profile": LANE, "backend": backend, "package": PACKAGES[backend], "passed": False,
             "expected_tests": list(CASES[backend]), "tests": [],
             "resource_contract": {"owned_child_rss_cap_bytes": shared.RSS_CAP,
                                   "child_deadline_seconds": shared.DEADLINE,
                                   "compiler_deadline_seconds": COMPILER_SECONDS}}
    stamp = None
    try:
        proof.update(venue(backend))
        stamp = shared.source_stamp()
        if stamp["head"] != proof["head"] or not CASES[backend]:
            raise UnitFailure("decoder source identity or nonempty registration is missing")
        binary, proof["compiler"] = compile_binary(backend, directory, stamp)
        owner = shared.CpuOwner()
        listing = supervise([str(binary), "--list"], owner, directory / "unfiltered-list.log",
                            deadline=shared.DEADLINE, rss_cap=shared.RSS_CAP)
        names = [line.removesuffix(": test") for line in listing.splitlines() if line.endswith(": test")]
        if owner.collect() or len(names) != len(set(names)) or not set(CASES[backend]).issubset(names):
            raise UnitFailure("decoder executable discovery or owned listing cleanup incomplete")
        for index, name in enumerate(CASES[backend]):
            if shared.source_stamp() != stamp or shared.file_digest(binary) != proof["compiler"]["binary_sha256"]:
                raise UnitFailure("decoder source or executable changed before next case")
            log = directory / f"case-{index}.log"
            try:
                shared.execute(binary, name, log)
            except shared.CaseAssertionFailure:
                pass  # Only exact assertion failures with owned release are aggregatable.
            row = json.loads(log.with_suffix(".result.json").read_text(encoding="utf-8"))
            proof["tests"].append(row)
            if (row.get("test") != name or row.get("source") != stamp
                    or row.get("binary_sha256") != proof["compiler"]["binary_sha256"]
                    or row.get("status") not in ("passed", "assertion_failed")):
                raise UnitFailure("decoder result source/executable/test protocol identity differs")
            if not resources_verified(row):
                raise UnitFailure("decoder owned RSS/deadline/release evidence is missing or outside its contract")
        proof["passed"] = ([row["test"] for row in proof["tests"]] == list(CASES[backend])
                           and all(row["status"] == "passed" and row["owned_released"]
                                   and row["source"] == stamp
                                   and row["binary_sha256"] == proof["compiler"]["binary_sha256"]
                                   for row in proof["tests"]))
    except BaseException as error:
        proof["fatal_error"] = str(error)
        raise
    finally:
        try:
            proof["final_source"] = shared.source_stamp()
            proof["source_unchanged"] = stamp is not None and proof["final_source"] == stamp
        except (OSError, subprocess.SubprocessError) as error:
            proof["source_unchanged"] = False
            proof["source_check_error"] = str(error)
        if not proof["source_unchanged"]:
            proof["passed"] = False
        proof["file_sha256"] = {str(path.relative_to(directory)): shared.file_digest(path)
                                for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "proof.json"}
        (directory / "proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("backend", choices=tuple(PACKAGES), nargs="?")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            validate_dispatch(os.environ.get("GITHUB_EVENT_NAME", ""), event_payload())
            return 0
        if args.backend is None:
            parser.error("run requires one exact declared backend")
        proof = run(args.backend, Path(os.environ["RUNNER_TEMP"]) / f"llm-decoder-{args.backend}-evidence")
        return 0 if proof["passed"] else 1
    except (UnitFailure, OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as error:
        print(f"llm-decoder-fixture: FAIL {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

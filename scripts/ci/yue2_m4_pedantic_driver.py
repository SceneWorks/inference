#!/usr/bin/env python3
"""Bound one diagnostic-only BF16 stage-2 child on the reviewed GPU0 owner window.

The numeric child is deliberately not the terminal YuE2 precision test. This controller
authenticates its derivative source and M4 teacher separately, then retains physical
ownership, process release, and raw output for independent numerical inspection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from yue2_cuda_idle_context import require_remaining_window
from yue2_decoder_trace_overlay import tree_digest
from yue2_native_convt_overlay import CANDLE_SHA, CANDLE_TREE
from yue2_gpu0_owner_guard import OwnerGuard, PEDANTIC_ENGINE, reap_tree, wait
from yue2_precision_proof import (
    REFERENCE_SHA256, cuda_physical_census, physical_busy_message, require,
    retain_cuda_physical_evidence, retain_reviewed_baseline, sample_cuda, sha256,
    verify_reference, verify_revisions, write_json,
)

SELECTOR = "pedantic_stage2"
M4_SOURCE = "825341ff8d0110ea448213485891b39d57806fa4"
REFERENCE_CONTROL = "29d4e30e9e281d45f04048560003d8e3b8c55790"
REFERENCE_RUN = "37138394196"
CHILD_SECONDS = 600
POSTFLIGHT_SECONDS = 600
JOB_SECONDS = 30 * 60


def verify_teacher(directory: Path) -> None:
    require(directory.is_dir() and {p.name for p in directory.iterdir()} == {
        "vae_real_reference.safetensors", "NONCOMMERCIAL.txt", "reference-provenance.json"
    }, "M4 teacher transfer inventory changed")
    verify_reference(argparse.Namespace(directory=directory, engine_sha=M4_SOURCE,
                                        control_sha=REFERENCE_CONTROL, run_id=REFERENCE_RUN))


def verify_binary(build_json: Path, binary: Path, stage: Path) -> dict:
    """Bind the run-owned executable to exactly one Cargo compiler artifact."""
    require(build_json.is_file() and binary.is_file(), "diagnostic build or binary absent")
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    require(bool(target_dir) and binary.resolve().is_relative_to(Path(target_dir).resolve()),
            "diagnostic executable is outside the declared Cargo target")
    matches = []
    core = []
    kernels = []
    audio = []
    yue2 = []
    for line in build_json.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # Cargo's progress output is not a compiler artifact.
        if row.get("reason") != "compiler-artifact":
            continue
        target = row.get("target") or {}
        if target.get("name") == "yue2-bf16-tile-diagnostic" and "bin" in target.get("kind", []) and row.get("executable"):
            matches.append(row)
        if target.get("name") == "candle_core":
            core.append(row)
        if target.get("name") == "candle_kernels":
            kernels.append(row)
        if target.get("name") == "candle_audio":
            audio.append(row)
        if target.get("name") == "candle_audio_yue2":
            yue2.append(row)
    require(len(matches) == 1 and Path(matches[0]["executable"]).resolve() == binary.resolve(),
            "diagnostic executable differs from the one Cargo build artifact")
    def source(row: dict) -> str:
        return row.get("package_id", "").replace("\\", "/").lower()
    expected_core = (stage / "candle-overlay/candle-core").resolve().as_posix().lower()
    expected_kernel = (stage / "engine-overlay/crates/media/candle-gen/vendor/candle-kernels").resolve().as_posix().lower()
    expected_harness = (stage / "harness").resolve().as_posix().lower()
    expected_audio = (stage / "engine-overlay/crates/audio/candle-audio").resolve().as_posix().lower()
    expected_yue2 = (stage / "engine-overlay/crates/audio/candle-audio-yue2").resolve().as_posix().lower()
    require(len(core) == len(kernels) == 1 and
            set(core[0].get("features", [])) == {"cuda", "cudarc", "default"} and
            source(core[0]).startswith("path+") and expected_core in source(core[0]) and
            source(kernels[0]).startswith("path+") and expected_kernel in source(kernels[0]),
            "diagnostic build changed the declared Candle CUDA core or vendored kernels")
    require(source(matches[0]).startswith("path+") and expected_harness in source(matches[0]) and
            len(audio) == len(yue2) == 1 and
            source(audio[0]).startswith("path+") and expected_audio in source(audio[0]) and
            source(yue2[0]).startswith("path+") and expected_yue2 in source(yue2[0]) and
            {"cuda", "default"} <= set(audio[0].get("features", [])) and
            {"cuda", "default"} <= set(yue2[0].get("features", [])),
            "diagnostic build changed the staged M4 audio/VAE packages or harness")
    return {"binary_sha256": sha256(binary), "binary_bytes": binary.stat().st_size,
            "package_id": matches[0].get("package_id"), "target": matches[0]["target"],
            "candle_core_package_id": core[0]["package_id"],
            "vendored_kernel_package_id": kernels[0]["package_id"],
            "candle_audio_package_id": audio[0]["package_id"],
            "yue2_vae_package_id": yue2[0]["package_id"]}


def verify_source(engine_sha: str, control_sha: str) -> None:
    require(engine_sha == PEDANTIC_ENGINE == M4_SOURCE,
            "unreviewed M4 parent source")
    require(re.fullmatch(r"[0-9a-f]{40}", control_sha) is not None,
            "diagnostic control SHA must be full lowercase hex")
    verify_revisions(engine_sha, control_sha)
    for checkout, expected in ((Path.cwd(), engine_sha), (Path("../control"), control_sha)):
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"],
                               capture_output=True, text=True, encoding="utf-8", check=True).stdout
        require(head == expected and not dirty.strip(), "diagnostic source checkout changed/is dirty")
    require(os.environ.get("RUNNER_NAME") in {"cuda-windows", "cuda-windows-2"} and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and
            os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and
            os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID") == "37135502627",
            "diagnostic is not bound to reviewed Windows GPU0")


def verify_derivative(path: Path, control_sha: str) -> dict:
    """Rehash the exact staged M4/Candle archives, declared patches and built trees."""
    require(path.name == "m4-pedantic-source.json" and path.is_file(),
            "M4 diagnostic source provenance absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    require(value.get("schema") == "yue2-m4-pedantic-source-v1" and
            value.get("engineSha") == M4_SOURCE and value.get("controlSha") == control_sha and
            value.get("diagnosticOnly") is True and value.get("candleSha") == CANDLE_SHA and
            value.get("candleTree") == CANDLE_TREE,
            "M4 diagnostic source/control identity changed")
    files = {
        "engineArchiveSha256": root / "m4-tracked-source.tar.gz",
        "vaePatchSha256": root / "m4-vae-observation.patch",
        "candleArchiveSha256": root / "pinned-candle-tracked-source.tar.gz",
        "candleBackendPatchSha256": root / "candle-column.patch",
        "candleKernelPathPatchSha256": root / "candle-kernel-path.patch",
        "derivativeVaeSha256": root / "engine-overlay/crates/audio/candle-audio-yue2/src/vae.rs",
        "candleBackendDerivativeSha256": root / "candle-overlay/candle-core/src/cuda_backend/mod.rs",
        "m4RootLockSha256": root / "engine-overlay/Cargo.lock",
        "harnessLockSha256": root / "harness/Cargo.lock",
        "harnessManifestSha256": root / "harness/Cargo.toml",
    }
    for field, file in files.items():
        require(file.is_file() and value.get(field) == sha256(file),
                f"M4 diagnostic {field} differs from staged source")
    require(value.get("engineDerivativeTreeSha256") == tree_digest(root / "engine-overlay") and
            value.get("candleDerivativeTreeSha256") == tree_digest(root / "candle-overlay") and
            value.get("harnessSourceTreeSha256") == tree_digest(root / "harness"),
            "M4/Candle derivative tree changed after staging")
    template = Path("../control/scripts/ci/yue2_m4_pedantic").resolve()
    source_files = {entry.name for entry in (template / "src").iterdir() if entry.is_file() and not entry.is_symlink()}
    staged_files = {entry.name for entry in (root / "harness/src").iterdir() if entry.is_file() and not entry.is_symlink()}
    require(value.get("harnessLockSha256") == sha256(template / "Cargo.lock.snapshot") and
            source_files == staged_files == {"main.rs", "pedantic_stage2.rs", "pedantic_io.rs"} and
            all((root / "harness/src" / name).read_bytes() == (template / "src" / name).read_bytes()
                for name in source_files) and
            all(value.get(field) == sha256(template / name) for field, name in (
                ("vaePatchSha256", "m4-vae-observation.patch"),
                ("candleBackendPatchSha256", "candle-column.patch"),
                ("candleKernelPathPatchSha256", "candle-kernel-path.patch"))),
            "M4 diagnostic staged harness differs from reviewed control")
    return value


def remaining_window(job_started_ns: int) -> None:
    now = time.time_ns()
    require(job_started_ns <= now and
            now + (CHILD_SECONDS + POSTFLIGHT_SECONDS) * 1_000_000_000 <=
            job_started_ns + JOB_SECONDS * 1_000_000_000,
            "diagnostic child and postflight cannot fit workflow timeout")
    require_remaining_window(CHILD_SECONDS + POSTFLIGHT_SECONDS)


def physical(evidence: Path, label: str) -> tuple[str, list[str]]:
    raw, busy = cuda_physical_census()
    (evidence / f"census-{label}.txt").write_text(raw, encoding="utf-8")
    if not busy:
        retain_cuda_physical_evidence(evidence, label, raw)
    return raw, busy


def execute(args: argparse.Namespace) -> None:
    require(args.selector == SELECTOR, "unknown M4 diagnostic selector")
    require(not args.evidence.exists(), "diagnostic evidence path already exists")
    verify_source(args.engine_sha, args.control_sha)
    verify_teacher(args.reference)
    derivative = verify_derivative(args.source_provenance, args.control_sha)
    build = verify_binary(args.build_json, args.binary, args.source_provenance.parent)
    job_started = os.environ.get("YUE2_DIAGNOSTIC_JOB_STARTED_UTC_NS", "")
    require(job_started.isdigit(), "bounded diagnostic job start missing")
    remaining_window(int(job_started))
    args.evidence.mkdir(parents=True)
    _, baseline = require_remaining_window(CHILD_SECONDS + POSTFLIGHT_SECONDS)
    baseline_files = retain_reviewed_baseline(args.evidence, baseline)
    before_raw, before_busy = physical(args.evidence, "before")
    require(not before_busy, physical_busy_message(before_raw, before_busy,
            "foreign/lingering accelerator executables before diagnostic"))
    guard = OwnerGuard(args.evidence, args.engine_sha, args.control_sha, "diagnostic")
    guard.preflight()
    remaining_window(int(job_started))  # Source/API checks consumed time; reserve again at Popen.

    data = args.evidence / "data"
    require(not data.exists(), "run-owned numerical output already exists")
    command = [str(args.binary), "--diagnostic", SELECTOR,
               "--reference-dir", str(args.reference), "--output-dir", str(data)]
    env = {key: value for key, value in os.environ.items() if key not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    env["YUE2_ENGINE_ROOT"] = str(Path.cwd().resolve())
    env["YUE2_M4_PEDANTIC_PROVENANCE"] = str(args.source_provenance.resolve())
    samples: list[dict] = []
    faults: list[str] = []
    stop = threading.Event()
    child: subprocess.Popen | None = None
    thread: threading.Thread | None = None
    code: int | None = None
    timed_out = False
    wait_error: str | None = None
    guard_error: str | None = None
    started = time.time_ns()
    with (args.evidence / "diagnostic.log").open("w", encoding="utf-8") as log:
        try:
            guard.arm()  # Cancellation is armed before any Popen-owned CUDA child can exist.
            require(not guard.failed.is_set(), "owner canceled before diagnostic child creation")
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)

            def sample_loop() -> None:
                while not stop.is_set() and child is not None and child.poll() is None:
                    try:
                        samples.append(sample_cuda())
                    except Exception as error:
                        if child.poll() is None:
                            faults.append(f"{time.time_ns()}: {error}")
                    stop.wait(0.25)

            thread = threading.Thread(target=sample_loop, daemon=True)
            thread.start()
            guard.start(child)
            code, timed_out, wait_error = wait(child, guard, CHILD_SECONDS)
        except BaseException as error:
            wait_error = str(error)
            if child is not None:
                code, cleanup = reap_tree(child)
                if cleanup:
                    wait_error += f"; owned cleanup: {cleanup}"
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=25)
            try:
                guard.finish()  # Also restores the signal handlers after a failed Popen.
            except BaseException as error:
                guard_error = str(error)
    ended = time.time_ns()
    after_raw = ""
    after_busy: list[str] = []
    after_error = None
    try:
        after_raw, after_busy = physical(args.evidence, "after")
    except Exception as error:
        after_error = str(error)
    write_json(args.evidence / "external-samples.json", {"samples": samples, "faults": faults})
    report = {"schema": "yue2-m4-stage2-pedantic-control-v1", "diagnostic_only": True,
              "selector": SELECTOR, "m4_engine_sha": args.engine_sha,
              "control_sha": args.control_sha,
              "reference_transfer_run_id": REFERENCE_RUN, "reference_sha256": REFERENCE_SHA256,
              "reviewed_baseline_files": baseline_files, **build,
              "derivative_source": derivative,
              "runner_name": os.environ.get("RUNNER_NAME"),
              "owned_pid": child.pid if child is not None else None,
              "started_utc_ns": started, "ended_utc_ns": ended, "exit_code": code,
              "timed_out": timed_out, "wait_error": wait_error, "guard_error": guard_error,
              "owned_process_released": child is not None and child.poll() is not None,
              "sample_count": len(samples), "sampler_faults": faults,
              "post_census_busy": after_busy, "post_census_error": after_error,
              "guard": guard.summary()}
    write_json(args.evidence / "diagnostic-control.json", report)
    require(child is not None and code == 0 and not timed_out and wait_error is None and
            guard_error is None and child.poll() is not None,
            "bounded diagnostic child failed or its owned release was not proven")
    require(samples and not faults and thread is not None and not thread.is_alive(),
            "diagnostic sampler did not release cleanly")
    require(after_raw and not after_busy and after_error is None,
            "diagnostic postflight physical census refused")
    result_path = data / "report.json"
    require(result_path.is_file(), "stage-2 diagnostic report absent")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result.get("schemaVersion") == 7 and result.get("selector") == SELECTOR and
            result.get("acceptanceSatisfied") is False and
            result.get("engineSha") == M4_SOURCE and
            result.get("sourceProvenance") == derivative and
            set(result.get("arms", {})) == {"mode16", "mode18"} and
            result.get("mathEvents", {}).get("sha256") == sha256(data / "math-events.jsonl"),
            "stage-2 diagnostic report source, arms, or event bytes changed")
    # Numerical contents are audited separately: a genuine difference under mode 18 is not a
    # controller error, and successful collection never certifies production waveform fidelity.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-teacher", type=Path)
    parser.add_argument("--selector")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--build-json", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-provenance", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--engine-sha")
    parser.add_argument("--control-sha")
    args = parser.parse_args()
    if args.verify_teacher is not None:
        require(all(getattr(args, key) is None for key in
                    ("selector", "binary", "build_json", "reference", "source_provenance",
                     "evidence", "engine_sha", "control_sha")),
                "teacher verification cannot be mixed with diagnostic execution")
        verify_teacher(args.verify_teacher)
        return
    require(all(getattr(args, key) is not None for key in
                ("selector", "binary", "build_json", "reference", "source_provenance",
                 "evidence", "engine_sha", "control_sha")),
            "bounded diagnostic arguments incomplete")
    execute(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded CUDA-only numerical diagnosis; never grades production acceptance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import tomllib

from yue2_precision_proof import (REFERENCE_SHA256, cuda_census, require, sample_cuda,
                                 DECODER_SHA256, sha256, verify_reference, verify_revisions, write_json)

ENGINE_SHA = "4127a675fc8575555e029e01b7f6867488880a8f"


def verify_diagnostic_data(data: Path) -> None:
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    require(report.get("schemaVersion") == 1 and report.get("purpose") == "diagnostic_only_no_gate_change",
            "output must identify itself as diagnostic-only")
    require(report.get("engineSha") == ENGINE_SHA and report.get("referenceSha256") == REFERENCE_SHA256,
            "diagnostic output lost the failed source/reference identity")
    require(report.get("backend") == "cuda" and report.get("deviceOrdinal") == 0 and
            (report.get("frames"), report.get("coreFrames"), report.get("haloFrames")) == (75, 16, 16) and
            report.get("originalBound") == 1 / 64, "diagnostic input or bound changed")
    runs = report.get("runs", {})
    require(set(runs) == {"bf16", "f32"}, "diagnostic policy controls incomplete")
    artifacts = []
    for name, dtype, count in (("bf16", "BF16", 2), ("f32", "F32", 1)):
        row = runs[name]
        require(row.get("residentDtype") == dtype and
                row.get("decoderIdentity", {}).get("weights_sha256") == DECODER_SHA256["standard"],
                "diagnostic decoder identity or resident dtype changed")
        for mode in ("full", "tiled"):
            captures = row.get(mode, [])
            require(len(captures) == count, "diagnostic repeatability coverage incomplete")
            for capture in captures:
                require(capture.get("rawDtype") == dtype, "diagnostic activation dtype changed")
                for kind in ("raw", "clamped"):
                    artifact = capture[f"{kind}Artifact"]
                    path = data / artifact["file"]
                    require(path.resolve().parent == data.resolve() and path.is_file(),
                            "diagnostic array escaped its run-owned directory")
                    require(artifact["bytes"] == path.stat().st_size == (75 * 1920 - 64) * 2 * 4 and
                            artifact["sha256"] == sha256(path) == capture[f"{kind}Sha256"],
                            "diagnostic residual array hash mismatch")
                    artifacts.append(path.resolve())
    require(len(set(artifacts)) == 12, "diagnostic arrays collided")


def prepare_harness(args: argparse.Namespace) -> None:
    require(args.engine_sha == ENGINE_SHA, "harness requires the exact failed M3 source")
    verify_revisions(args.engine_sha, args.control_sha)
    require(not args.destination.exists(), "standalone harness path must be fresh")
    engine = Path.cwd().resolve()
    control = Path("../control").resolve()
    target = args.destination.resolve()
    require(not target.is_relative_to(engine) and not target.is_relative_to(control),
            "standalone harness must remain outside stationary checkouts")
    template = args.template.resolve(strict=True)
    original_lock = tomllib.loads((engine / "Cargo.lock").read_text(encoding="utf-8"))
    harness_lock = tomllib.loads((template / "Cargo.lock.snapshot").read_text(encoding="utf-8"))
    def identity(package: dict) -> tuple:
        return tuple(package.get(key) for key in ("name", "source", "version", "checksum"))
    baseline = {identity(package) for package in original_lock["package"]}
    require(all(identity(package) in baseline for package in harness_lock["package"]
                if package["name"] != "yue2-bf16-tile-diagnostic"),
            "diagnostic dependencies differ from the failed M3 lock")
    manifest = (template / "Cargo.toml.in").read_text(encoding="utf-8")
    require("__ENGINE_ROOT__" in manifest, "manifest lacks the exact-source path placeholder")
    # A JSON string body is also a valid escaped TOML basic string body.
    escaped_engine = json.dumps(engine.as_posix(), ensure_ascii=False)[1:-1]
    manifest = manifest.replace("__ENGINE_ROOT__", escaped_engine)
    target.mkdir(parents=True)
    (target / "src").mkdir()
    (target / "Cargo.toml").write_text(manifest, encoding="utf-8")
    shutil.copy2(template / "Cargo.lock.snapshot", target / "Cargo.lock")
    shutil.copy2(template / "src/main.rs", target / "src/main.rs")
    write_json(target.parent / "harness-provenance.json", {
        "engine_sha": args.engine_sha, "control_sha": args.control_sha,
        "template_files": {str(p.relative_to(template)): sha256(p) for p in sorted(template.rglob("*")) if p.is_file()},
        "staged_files": {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()},
    })


def resolve_binary(build_json: Path, output: Path) -> None:
    candidates = []
    core_features = []
    kernel_sources = []
    for line in build_json.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("reason") == "compiler-artifact":
            target = row.get("target", {}).get("name")
            if target == "candle_core":
                core_features.append(set(row.get("features", [])))
            elif target == "candle_kernels":
                kernel_sources.append(row.get("package_id", "").replace("\\", "/"))
        if (row.get("reason") == "compiler-artifact" and
                row.get("target", {}).get("name") == "yue2-bf16-tile-diagnostic" and
                row.get("executable")):
            candidates.append(Path(row["executable"]))
    require(len(candidates) == 1 and candidates[0].is_file(), "one diagnostic executable required")
    # The failed M3 binary had no cuDNN: preserve that actual implementation,
    # including its source-owned CUDA kernel patch, in this same-input comparison.
    require(core_features == [{"cuda", "cudarc", "default"}], "Candle features differ from failed M3 build")
    require(len(kernel_sources) == 1 and kernel_sources[0].startswith("path+") and
            "/engine/crates/media/candle-gen/vendor/candle-kernels#" in kernel_sources[0],
            "diagnostic must use the failed M3 vendored CUDA kernels")
    output.write_text(str(candidates[0].resolve()) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> None:
    require(args.engine_sha == ENGINE_SHA, "diagnostic requires the exact failed M3 source")
    verify_revisions(args.engine_sha, args.control_sha)
    require(os.environ.get("RUNNER_NAME") in {"cuda-windows", "cuda-windows-2"},
            "diagnostic requires an eligible shared CUDA listener")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "diagnostic must use physical GPU0")
    for checkout in (Path.cwd(), Path("../control")):
        dirty = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"],
                               capture_output=True, text=True, check=True, encoding="utf-8").stdout
        require(not dirty.strip(), "diagnostic source checkout is dirty")
    verify_reference(argparse.Namespace(directory=args.reference, engine_sha=args.engine_sha))
    require(args.binary.is_file(), "diagnostic binary absent")
    args.evidence.mkdir(parents=True, exist_ok=True)
    data = args.evidence / "data"
    require(not data.exists(), "diagnostic data must be fresh")
    before_raw, before_busy = cuda_census()
    (args.evidence / "census-before.txt").write_text(before_raw, encoding="utf-8")
    require(not before_busy, f"foreign accelerator ownership before diagnostic: {before_busy}")
    samples, faults = [], []
    stop = threading.Event()
    timed_out = False
    started = time.time_ns()
    env = os.environ.copy()
    env["YUE2_ENGINE_ROOT"] = str(Path.cwd().resolve())
    with (args.evidence / "diagnostic.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([str(args.binary), "--reference-dir", str(args.reference),
                                  "--output-dir", str(data)], stdout=log, stderr=subprocess.STDOUT, env=env)
        def sample_loop() -> None:
            while not stop.is_set() and child.poll() is None:
                try:
                    samples.append(sample_cuda())
                except Exception as error:
                    if child.poll() is None:
                        faults.append(str(error))
                stop.wait(0.25)
        thread = threading.Thread(target=sample_loop, daemon=True)
        thread.start()
        try:
            code = child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()  # This is the single owned CUDA diagnostic child, never a foreign process.
            code = child.wait(timeout=30)
        finally:
            stop.set()
            thread.join(timeout=25)
    ended = time.time_ns()
    after_error = None
    try:
        after_raw, after_busy = cuda_census()
    except Exception as error:
        after_raw, after_busy, after_error = "", [], str(error)
    (args.evidence / "census-after.txt").write_text(after_raw, encoding="utf-8")
    write_json(args.evidence / "external-samples.json", {"samples": samples, "faults": faults})
    files = [{"path": str(p.relative_to(args.evidence)), "bytes": p.stat().st_size,
              "sha256": sha256(p)} for p in sorted(data.rglob("*")) if p.is_file()]
    report = {"schema": "yue2-bf16-tile-diagnostic-control-v1", "diagnostic_only": True,
              "engine_sha": args.engine_sha, "control_sha": args.control_sha,
              "reference_sha256": REFERENCE_SHA256, "binary_sha256": sha256(args.binary),
              "runner_name": os.environ["RUNNER_NAME"], "owned_pid": child.pid,
              "started_utc_ns": started, "ended_utc_ns": ended, "exit_code": code,
              "timed_out": timed_out, "owned_process_released": child.poll() is not None,
              "post_census_busy": after_busy, "post_census_error": after_error,
              "sample_count": len(samples), "sampler_faults": faults, "data_files": files}
    write_json(args.evidence / "diagnostic-control.json", report)
    print(json.dumps(report, indent=2), flush=True)
    require(not timed_out and code == 0, "diagnostic execution failed; saved output is not acceptance")
    require(samples and not faults, "diagnostic sampler incomplete")
    require(child.poll() is not None and not after_busy and after_error is None,
            "diagnostic accelerator release unverified")
    require((data / "report.json").is_file(), "diagnostic residual report absent")
    verify_diagnostic_data(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-harness")
    for name in ("template", "destination"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for name in ("engine-sha", "control-sha"):
        prepare.add_argument(f"--{name}", required=True)
    resolve = commands.add_parser("resolve-binary")
    resolve.add_argument("--build-json", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("binary", "reference", "evidence"):
        run.add_argument(f"--{name}", type=Path, required=True)
    for name in ("engine-sha", "control-sha"):
        run.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.command == "prepare-harness":
        prepare_harness(args)
    elif args.command == "resolve-binary":
        resolve_binary(args.build_json, args.output)
    else:
        execute(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"yue2-bf16-tile-diagnostic: {error}") from error

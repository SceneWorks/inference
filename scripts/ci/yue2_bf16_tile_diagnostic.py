#!/usr/bin/env python3
"""Bounded CUDA-only numerical diagnosis; never grades production acceptance."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import time
import tomllib

from yue2_precision_proof import (REFERENCE_SHA256, cuda_census, require, sample_cuda,
                                 DECODER_SHA256, sha256, verify_reference, verify_revisions, write_json)

ENGINE_SHA = "4127a675fc8575555e029e01b7f6867488880a8f"
LATENT_SHA256 = "f89f02851d08128baa12a3f50cc3e73c80fbb5578797b7c166888dd815dc70c9"
DIAGNOSTICS = ("waveform", "first_conv", "first_conv_math")


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


def first_conv_array(data: Path, row: dict, dtype: str, shape: list[int]) -> tuple[list[float], list[int]]:
    require(row.get("dtype") == dtype and row.get("shape") == shape and
            row.get("layout") == "bct_f32le", "first Conv7 tensor dtype, shape, or layout changed")
    path = data / row["file"]
    require(path.resolve().parent == data.resolve() and path.is_file(), "first Conv7 array escaped evidence")
    raw = path.read_bytes()
    require(len(raw) == row.get("bytes") == math.prod(shape) * 4 and
            row.get("sha256") == sha256(path), "first Conv7 array size/hash mismatch")
    bits = [entry[0] for entry in struct.iter_unpack("<I", raw)]
    floats = [entry[0] for entry in struct.iter_unpack("<f", raw)]
    require(all(math.isfinite(value) for value in floats), "first Conv7 array is non-finite")
    if dtype == "BF16":
        require(all(value & 0xffff == 0 for value in bits), "BF16 first Conv7 array lost its dtype")
    return floats, bits


def first_conv_comparison(full: tuple[list[float], list[int]], tile: tuple[list[float], list[int]],
                          channels: int, frames: int, tile_frames: int, start: int, left: int,
                          core_len: int) -> dict:
    maximum = 0.0
    different = 0
    first = None
    for channel in range(channels):
        for offset in range(core_len):
            full_at = channel * frames + start + offset
            tile_at = channel * tile_frames + start - left + offset
            a, b = full[0][full_at], tile[0][tile_at]
            error = struct.unpack("<f", struct.pack("<f", abs(a - b)))[0]
            maximum = max(maximum, error)
            if full[1][full_at] != tile[1][tile_at]:
                different += 1
                if first is None:
                    first = {"channel": channel, "globalLatentFrame": start + offset,
                             "fullValue": a, "tileValue": b, "absError": error}
    return {"maxAbs": maximum, "differentValues": different, "firstDifferent": first,
            "comparedValues": channels * core_len}


def same_first_conv_comparison(observed: dict, expected: dict) -> bool:
    """Compare reported f32 scalars by bits after JSON's decimal round trip."""
    if not isinstance(observed, dict) or set(observed) != set(expected):
        return False
    if any(observed[key] != expected[key] for key in ("differentValues", "comparedValues")):
        return False
    def bits(value: float) -> bytes:
        return struct.pack("<f", value)
    if bits(observed["maxAbs"]) != bits(expected["maxAbs"]):
        return False
    a, b = observed["firstDifferent"], expected["firstDifferent"]
    if a is None or b is None:
        return a is b
    if not isinstance(a, dict) or set(a) != set(b):
        return False
    if any(a[key] != b[key] for key in ("channel", "globalLatentFrame")):
        return False
    values = ("fullValue", "tileValue", "absError") if "fullValue" in b else ("aValue", "bValue", "absError")
    return all(bits(a[key]) == bits(b[key]) for key in values)


def verify_first_conv_data(data: Path, meta: dict, report: dict | None = None,
                           schema: int = 2, selector: str = "first_conv",
                           purpose: str = "diagnostic_only_no_gate_change",
                           allow_input_mismatch: bool = False) -> None:
    if report is None:
        report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    require(report.get("schemaVersion") == schema and report.get("selector") == selector and
            report.get("purpose") == purpose, "first Conv7 schema/selector changed")
    require(report.get("engineSha") == ENGINE_SHA and report.get("referenceSha256") == REFERENCE_SHA256 and
            report.get("referenceMetadataSha256") == sha256(Path("crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json")),
            "first Conv7 source/reference identity changed")
    latent = report.get("latentIdentity", {})
    require(latent.get("sha256") == LATENT_SHA256 and latent.get("shape") == [75, 64] and
            latent.get("source", {}).get("stage_identity") == "precision_reference:long_latent",
            "first Conv7 latent identity changed")
    decoder = report.get("decoderIdentity", {})
    require(decoder.get("weights_sha256") == DECODER_SHA256["standard"] and
            decoder.get("config_sha256") == meta["decoders"]["standard"]["config_sha256"] and
            decoder.get("repo") == meta["decoders"]["standard"]["repo"] and
            decoder.get("revision") == meta["decoders"]["standard"]["revision"],
            "first Conv7 decoder identity changed")
    require(report.get("backend") == "cuda" and report.get("deviceOrdinal") == 0 and
            (report.get("frames"), report.get("coreFrames"), report.get("haloFrames")) == (75, 16, 16) and
            report.get("operator") == {"name": "decoder.layers.0.Conv1d", "kernel": 7,
                                       "padding": 3, "stride": 1, "dilation": 1, "groups": 1},
            "first Conv7 geometry changed")
    observation = report.get("originalWaveformObservation", {})
    require(observation.get("runId") == "36884387320" and observation.get("clampedMaxAbs") == 0.03125 and
            observation.get("originalBound") == 1 / 64 and
            observation.get("interpretation") == "prior_failed_waveform_proof_not_a_first_conv_gate",
            "original failed waveform observation was regraded")
    runs = report.get("runs", {})
    require(set(runs) == {"bf16", "f32"}, "first Conv7 precision controls incomplete")
    artifacts = []
    for name, dtype in (("bf16", "BF16"), ("f32", "F32")):
        row = runs[name]
        resident = row.get("resident", {})
        weight_shape = resident.get("weightShape")
        require(isinstance(weight_shape, list) and len(weight_shape) == 3 and
                isinstance(weight_shape[0], int) and 1 <= weight_shape[0] <= 4096 and
                weight_shape[1:] == [64, 7] and resident.get("biasShape") == [weight_shape[0]] and
                resident.get("sourceDtype") == resident.get("foldDtype") == "F32" and
                resident.get("residentDtype") == dtype and
                all(re.fullmatch(r"[0-9a-f]{64}", resident.get(key, "")) for key in
                    ("weightF32LeSha256", "biasF32LeSha256")),
                "first Conv7 resident weight identity changed")
        channels = weight_shape[0]
        full = {}
        for stage, stage_channels in (("input", 64), ("preBias", channels), ("postBias", channels)):
            capture = row["full"][stage]
            full[stage] = first_conv_array(data, capture, dtype, [1, stage_channels, 75])
            artifacts.append(capture["file"])
        windows = row.get("windows", [])
        require(len(windows) == 5, "first Conv7 must compare all five windows")
        for index, window in enumerate(windows):
            start = index * 16
            end = min(start + 16, 75)
            left = max(0, start - 16)
            right = min(75, end + 16)
            require((window.get("start"), window.get("end"), window.get("left"), window.get("right"),
                     window.get("coreLength")) == (start, end, left, right, end - start),
                    "first Conv7 window geometry changed")
            for stage, stage_channels in (("input", 64), ("preBias", channels), ("postBias", channels)):
                capture = window["captures"][stage]
                tile = first_conv_array(data, capture, dtype, [1, stage_channels, right - left])
                artifacts.append(capture["file"])
                expected = first_conv_comparison(full[stage], tile, stage_channels, 75,
                                                 right - left, start, left, end - start)
                require(same_first_conv_comparison(window["alignedCore"][stage], expected),
                        "first Conv7 aligned residual does not match saved arrays")
            if not allow_input_mismatch:
                require(window["alignedCore"]["input"]["differentValues"] == 0,
                        "first Conv7 same-input core is not byte-identical")
    require(len(artifacts) == len(set(artifacts)) == 36, "first Conv7 raw arrays collided")


def first_conv_all_comparison(a: tuple[list[float], list[int]], b: tuple[list[float], list[int]],
                              channels: int, length: int, origin: int) -> dict:
    maximum = 0.0
    different = 0
    first = None
    for channel in range(channels):
        for frame in range(length):
            index = channel * length + frame
            error = struct.unpack("<f", struct.pack("<f", abs(a[0][index] - b[0][index])))[0]
            maximum = max(maximum, error)
            if a[1][index] != b[1][index]:
                different += 1
                if first is None:
                    first = {"channel": channel, "globalLatentFrame": origin + frame,
                             "aValue": a[0][index], "bValue": b[0][index], "absError": error}
    return {"maxAbs": maximum, "differentValues": different, "firstDifferent": first,
            "comparedValues": channels * length}


def first_conv_math_gate(windows: list[dict]) -> str:
    if any(window["alignedCore"]["input"]["differentValues"] for window in windows):
        return "input_core_mismatch"
    if any(window["alignedCore"]["preBias"]["differentValues"] > 0 and
           math.isfinite(window["alignedCore"]["preBias"]["maxAbs"]) and
           window["alignedCore"]["preBias"]["maxAbs"] > 0 for window in windows):
        return "positive_pre_bias_residual"
    return "no_positive_pre_bias_residual"


def verify_first_conv_math_data(data: Path, meta: dict) -> None:
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    verify_first_conv_data(data, meta, report, 3, "first_conv_math",
                           "controlled_diagnostic_only_no_gate_change", allow_input_mismatch=True)
    controlled = report.get("controlled", {})
    baseline = report["runs"]["bf16"]
    reason = first_conv_math_gate(baseline["windows"])
    require(controlled.get("reason") == reason, "first Conv7 math gate reason changed")
    require(controlled.get("modeEventsFile") == "math-mode-events.jsonl", "math-mode event path changed")
    event_path = data / "math-mode-events.jsonl"
    require(event_path.is_file() and controlled.get("modeEventsSha256") == sha256(event_path),
            "math-mode event hash mismatch")
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    actions = (["before_default_arm"] if reason != "positive_pre_bias_residual" else
               ["before_default_arm", "before_flagged_arm", "set_disallow", "read_disallow",
                "restore_default", "read_restored"])
    modes = ([0] if reason != "positive_pre_bias_residual" else [0, 0, 16, 16, 0, 0])
    require(len(events) == len(actions) and all(
        row == {"action": action, "status": "CUBLAS_STATUS_SUCCESS", "rawMode": mode}
        for row, action, mode in zip(events, actions, modes)), "math-mode call/readback sequence incomplete")
    if reason != "positive_pre_bias_residual":
        require(controlled.get("status") == "not_applicable" and controlled.get("flagged") is None and
                controlled.get("crossArm") is None, "inapplicable first Conv7 math arm ran")
        return
    require(controlled.get("status") == "collected", "applicable first Conv7 math arm absent")
    flagged = controlled.get("flagged", {})
    require(flagged.get("resident") == baseline["resident"], "flagged BF16 operands changed")
    channels = baseline["resident"]["weightShape"][0]
    stages = (("input", 64), ("preBias", channels), ("postBias", channels))
    captured = {"full": {}, "windows": []}
    for stage, stage_channels in stages:
        a = first_conv_array(data, baseline["full"][stage], "BF16", [1, stage_channels, 75])
        b = first_conv_array(data, flagged["full"][stage], "BF16", [1, stage_channels, 75])
        captured["full"][stage] = (a, b)
        expected = first_conv_all_comparison(a, b, stage_channels, 75, 0)
        require(same_first_conv_comparison(controlled["crossArm"]["full"][stage], expected),
                "flagged full tensor comparison differs from saved arrays")
        if stage == "input":
            require(expected["differentValues"] == 0, "controlled BF16 full input bytes changed")
    windows = flagged.get("windows", [])
    require(len(windows) == len(baseline["windows"]) == len(controlled["crossArm"]["windows"]) == 5,
            "flagged first Conv7 window coverage incomplete")
    files = [capture["file"] for capture in baseline["full"].values()]
    files += [capture["file"] for row in baseline["windows"] for capture in row["captures"].values()]
    files += [capture["file"] for capture in report["runs"]["f32"]["full"].values()]
    files += [capture["file"] for row in report["runs"]["f32"]["windows"] for capture in row["captures"].values()]
    files += [capture["file"] for capture in flagged["full"].values()]
    for index, window in enumerate(windows):
        original = baseline["windows"][index]
        require((window.get("start"), window.get("end"), window.get("left"), window.get("right"),
                 window.get("coreLength")) ==
                (original["start"], original["end"], original["left"], original["right"],
                 original["coreLength"]), "flagged first Conv7 geometry changed")
        length, left = window["right"] - window["left"], window["left"]
        for stage, stage_channels in stages:
            a = first_conv_array(data, original["captures"][stage], "BF16", [1, stage_channels, length])
            b = first_conv_array(data, window["captures"][stage], "BF16", [1, stage_channels, length])
            files.append(window["captures"][stage]["file"])
            expected = first_conv_all_comparison(a, b, stage_channels, length, left)
            require(same_first_conv_comparison(controlled["crossArm"]["windows"][index][stage], expected),
                    "flagged window tensor comparison differs from saved arrays")
            if stage == "input":
                require(expected["differentValues"] == 0, "controlled BF16 window input bytes changed")
            full = first_conv_array(data, flagged["full"][stage], "BF16", [1, stage_channels, 75])
            aligned = first_conv_comparison(full, b, stage_channels, 75, length, window["start"],
                                            left, window["coreLength"])
            require(same_first_conv_comparison(window["alignedCore"][stage], aligned),
                    "flagged aligned residual differs from saved arrays")
    require(len(files) == len(set(files)) == 54, "first Conv7 math arrays collided")


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
    selector = getattr(args, "diagnostic", "waveform")
    require(selector in DIAGNOSTICS, "unknown diagnostic selector")
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
        child = subprocess.Popen([str(args.binary), "--diagnostic", selector,
                                  "--reference-dir", str(args.reference),
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
              "selector": selector,
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
    if selector in ("first_conv", "first_conv_math"):
        meta = json.loads(Path("crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json").read_text(encoding="utf-8"))
        if selector == "first_conv_math":
            verify_first_conv_math_data(data, meta)
        else:
            verify_first_conv_data(data, meta)
    else:
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
    run.add_argument("--diagnostic", choices=DIAGNOSTICS, default="waveform")
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

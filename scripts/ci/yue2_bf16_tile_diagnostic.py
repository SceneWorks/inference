#!/usr/bin/env python3
"""Bounded CUDA-only numerical diagnosis; never grades production acceptance."""
from __future__ import annotations

import argparse
import ctypes
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
from yue2_decoder_trace_overlay import tree_digest

ENGINE_SHA = "4127a675fc8575555e029e01b7f6867488880a8f"
LATENT_SHA256 = "f89f02851d08128baa12a3f50cc3e73c80fbb5578797b7c166888dd815dc70c9"
DIAGNOSTICS = ("waveform", "first_conv", "first_conv_math")
DIAGNOSTICS += ("decoder_trace",)


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


def verify_decoder_trace_data(data: Path, source_provenance: dict) -> None:
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    require(report.get("schemaVersion") == 4 and report.get("selector") == "decoder_trace" and
            report.get("purpose") == "diagnostic_only_no_gate_change" and
            report.get("engineSha") == ENGINE_SHA and
            report.get("derivativeSource") == source_provenance,
            "decoder trace source/schema identity changed")
    require(report.get("referenceSha256") == REFERENCE_SHA256 and
            report.get("latentIdentity", {}).get("sha256") == LATENT_SHA256 and
            report.get("backend") == "cuda" and report.get("deviceOrdinal") == 0 and
            (report.get("frames"), report.get("coreFrames"), report.get("haloFrames")) == (75, 16, 16) and
            report.get("stageCount") == 33 and report.get("originalBound") == 1 / 64 and
            report.get("originalWaveformRunId") == "36884387320",
            "decoder trace request or original observation changed")
    require(report.get("waveformParity") is True and report.get("bf16ClampedMaxAbs") == 0.03125,
            "decoder trace did not reproduce the original M3 failure")
    original = {
        "bf16": (("c12b2b7dca34ee4810536fcee12872674376c24ba484fa600c47fcb35317a908",
                  "33046e24cd16c415a9963711807abfc1579159036671ea4d2f08a6ca0a61a9f3"),
                 ("aa032057a51ad3451ab5826ffe18b2f14afae223adcae245b7663d787676f42c",
                  "a09be42f9ce9d0129c0640daadef40f711a6c3f65751a28b3e73a14de65d0a17")),
        "f32": (("742f7878e04093aee8abbb0389ebd4fe4a3e734d8c16a3a3f449cc0281bfad99",
                 "dcd3a7091bb2e0a442951ae94f81c2abc27b5956853645e37a998738af3a9646"),
                ("2cc2dc0824b1e2115d366829ef1dbc97c6a935e02dfc91237a5f4e7b7211a499",
                 "9cd3aace4657165201b692c2a20b63750d7d004ac68c8dc0136b38101eae9ec3")),
    }
    runs = report.get("runs", {})
    require(set(runs) == {"bf16", "f32"}, "decoder trace precision controls absent")
    earliest = None
    for label, dtype in (("bf16", "BF16"), ("f32", "F32")):
        run = runs[label]
        require(run.get("residentDtype") == dtype and len(run.get("stages", [])) == 33 and
                run.get("decoderIdentity", {}).get("weights_sha256") == DECODER_SHA256["standard"],
                "decoder trace resident graph changed")
        for index, stage in enumerate(run["stages"]):
            require(stage.get("index") == index and stage.get("kind") in
                    {"conv", "conv_transpose", "snake", "residual"} and
                    len(stage.get("windows", [])) == 5 and stage.get("full", {}).get("dtype") == dtype,
                    "decoder trace stage sequence incomplete")
            for window_index, window in enumerate(stage["windows"]):
                start = window_index * 16
                end = min(start + 16, 75)
                left = max(0, start - 16)
                right = min(75, end + 16)
                require((window.get("index"), window.get("start"), window.get("end"),
                         window.get("left"), window.get("right")) ==
                        (window_index, start, end, left, right) and
                        window.get("capture", {}).get("dtype") == dtype,
                        "decoder trace tile bounds changed")
                comparison = window.get("comparison", {})
                require(isinstance(comparison.get("positiveDifferences"), int) and
                        isinstance(comparison.get("bitDifferences"), int) and
                        0 <= comparison["positiveDifferences"] <= comparison["bitDifferences"] and
                        isinstance(comparison.get("maxAbs"), (int, float)) and
                        math.isfinite(comparison["maxAbs"]) and
                        (comparison["positiveDifferences"] > 0) == (comparison["maxAbs"] > 0),
                        "decoder trace stage comparison is internally inconsistent")
                if label == "bf16" and comparison["positiveDifferences"]:
                    earliest = index if earliest is None else min(index, earliest)
        for mode, expected in zip(("full", "tiled"), original[label]):
            capture = run.get("waveform", {}).get(mode, {})
            require(capture.get("rawSha256") == expected[0] and
                    capture.get("clampedSha256") == expected[1] and
                    capture.get("matchesOriginal") is True and capture.get("rawDtype") == dtype,
                    "decoder trace final waveform differs from exact M3")
    require(report.get("earliestBf16Stage") == earliest,
            "decoder trace earliest positive stage differs from stage rows")
    adaptive = report.get("adaptive", {})
    if earliest is None:
        require(adaptive.get("status") == "not_applicable", "adaptive replay ran without a divergence")
    else:
        require(adaptive.get("stage") == earliest and adaptive.get("kind") ==
                runs["bf16"]["stages"][earliest]["kind"] and
                adaptive.get("source") == "single_native_layer_replay" and
                adaptive.get("status") in {"collected", "inconclusive_replay_mismatch"},
                "adaptive replay is not bound to the first divergent stage")
        exact = all(adaptive["runs"][label][mode].get(key) is True
                    for label in ("bf16", "f32") for mode in ("full", "window")
                    for key in ("inputBitExact", "outputBitExact"))
        require((adaptive["status"] == "collected") == exact,
                "adaptive substeps were claimed without exact native replay")

    verify_decoder_trace_file_set(data, report)


def verify_decoder_trace_file_set(data: Path, report: dict, minimum_files: int = 404) -> None:
    references = []
    def visit(value: object) -> None:
        if isinstance(value, dict):
            if {"file", "sha256", "bytes"} <= set(value):
                name = value["file"]
                require(isinstance(name, str) and name == Path(name).name and
                        name not in ("", ".", ".."), "decoder trace array path escaped data directory")
                path = data / name
                require(path.is_file() and not path.is_symlink() and path.stat().st_size == value["bytes"] and
                        sha256(path) == value["sha256"], "decoder trace array hash/size changed")
                references.append(name)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(report)
    require(len(set(references)) >= minimum_files, "decoder trace raw stage or waveform coverage incomplete")
    entries = list(data.iterdir())
    require(all(item.is_file() and not item.is_symlink() for item in entries) and
            {item.name for item in entries} == {"report.json", *references},
            "decoder trace data contains unreferenced or unsafe files")


def decoder_trace_expected_bytes() -> dict:
    """Exact pinned 75-frame standard decoder widths and five source tile lengths."""
    lengths = [75, 32, 48, 48, 43, 27]
    total_elements = 0
    largest_stage_elements = 0
    for frames in lengths:
        rows = [(2048, frames)]
        channels = 2048
        length = frames
        for stride, out_channels in ((6, 1024), (5, 512), (4, 256),
                                     (4, 128), (2, 64), (2, 64)):
            rows.append((channels, length))  # Snake before transposed convolution.
            length = stride * length + stride - 2 * ((stride + 1) // 2)
            channels = out_channels
            rows.extend([(channels, length)] * 4)  # ConvT and three residuals.
        rows.extend([(channels, length), (2, length)])  # Final Snake and Conv7.
        require(len(rows) == 33, "pinned standard decoder stage count changed")
        total_elements += sum(channels * extent for channels, extent in rows)
        largest_stage_elements = max(largest_stage_elements,
                                     *(channels * extent for channels, extent in rows))
    return {"stage_count": 33, "source_lengths": lengths,
            "bf16_raw_bytes": total_elements * 2, "f32_raw_bytes": total_elements * 4,
            "max_stage_f32_bytes": largest_stage_elements * 4}


def available_physical_memory() -> int:
    require(os.name == "nt", "decoder trace resource preflight requires Windows")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong)]
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))),
            "GlobalMemoryStatusEx failed; available host memory is unknown")
    return state.available_physical


def preflight_decoder_trace_resources(evidence: Path) -> dict:
    predicted = decoder_trace_expected_bytes()
    disk_free = shutil.disk_usage(evidence).free
    memory_free = available_physical_memory()
    # All 33 BF16/F32 full+tile arrays are retained for independent audit; reserve one
    # additional copy for adaptive substeps, temp serialization and failure evidence.
    disk_required = 2 * (predicted["bf16_raw_bytes"] + predicted["f32_raw_bytes"])
    # One stage can have simultaneous native-device->CPU F32 copy, Vec, raw bytes,
    # paired full/tile reads and decoded comparison arrays. This is a bound on the
    # diagnostic's own host scratch, not a claim about GPU model residency.
    host_required = 8 * predicted["max_stage_f32_bytes"]
    result = {**predicted, "disk_free_bytes": disk_free, "disk_required_bytes": disk_required,
              "host_memory_available_bytes": memory_free, "host_scratch_required_bytes": host_required}
    write_json(evidence / "decoder-trace-resource-preflight.json", result)
    require(disk_free >= disk_required and memory_free >= host_required,
            "insufficient measured host resources for bounded decoder trace arrays")
    return result


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


def verify_first_conv_math_file_set(data: Path, arrays: list[str], expected_count: int) -> None:
    require(len(arrays) == len(set(arrays)) == expected_count, "first Conv7 math arrays collided")
    expected = {"report.json", "math-mode-events.jsonl", *arrays}
    entries = list(data.iterdir())
    require(all(entry.is_file() and not entry.is_symlink() for entry in entries),
            "first Conv7 math data contains a non-regular entry")
    require(len(entries) == len(expected) and {entry.name for entry in entries} == expected,
            "first Conv7 math data file set differs from report")


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
    files = [capture["file"] for run in report["runs"].values()
             for capture in run["full"].values()]
    files += [capture["file"] for run in report["runs"].values()
              for window in run["windows"] for capture in window["captures"].values()]
    if reason != "positive_pre_bias_residual":
        require(controlled.get("status") == "not_applicable" and controlled.get("flagged") is None and
                controlled.get("crossArm") is None, "inapplicable first Conv7 math arm ran")
        verify_first_conv_math_file_set(data, files, 36)
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
    verify_first_conv_math_file_set(data, files, 54)


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
    harness_dependencies = [package for package in harness_lock["package"]
                            if package["name"] != "yue2-bf16-tile-diagnostic"]
    require(len(harness_dependencies) == 205 and
            all(identity(package) in baseline for package in harness_dependencies),
            "diagnostic dependencies differ from the failed M3 lock")
    overlay = getattr(args, "overlay_root", None)
    if overlay is not None:
        overlay = overlay.resolve(strict=True)
        require(overlay != engine and sha256(overlay / "Cargo.lock") == sha256(engine / "Cargo.lock"),
                "decoder trace overlay does not preserve exact M3 lock")
        provenance = json.loads((overlay.parent / "overlay-provenance.json").read_text(encoding="utf-8"))
        require(provenance.get("engine_sha") == ENGINE_SHA and
                provenance.get("derivative_vae_sha256") == sha256(overlay / "crates/audio/candle-audio-yue2/src/vae.rs") and
                provenance.get("control_sha") == args.control_sha,
                "decoder trace source overlay provenance changed")
    manifest = (template / "Cargo.toml.in").read_text(encoding="utf-8")
    require("__ENGINE_ROOT__" in manifest, "manifest lacks the exact-source path placeholder")
    # A JSON string body is also a valid escaped TOML basic string body.
    escaped_engine = json.dumps((overlay or engine).as_posix(), ensure_ascii=False)[1:-1]
    manifest = manifest.replace("__ENGINE_ROOT__", escaped_engine)
    target.mkdir(parents=True)
    (target / "src").mkdir()
    (target / "Cargo.toml").write_text(manifest, encoding="utf-8")
    shutil.copy2(template / "Cargo.lock.snapshot", target / "Cargo.lock")
    shutil.copy2(template / "src/main.rs", target / "src/main.rs")
    shutil.copy2(template / "src/decoder_trace.rs", target / "src/decoder_trace.rs")
    write_json(target.parent / "harness-provenance.json", {
        "engine_sha": args.engine_sha, "control_sha": args.control_sha,
        "standalone_lock_sha256": sha256(template / "Cargo.lock.snapshot"),
        "m3_dependency_tuples": len(harness_dependencies),
        "overlay_sha256": sha256(overlay.parent / "overlay-provenance.json") if overlay else None,
        "template_files": {str(p.relative_to(template)): sha256(p) for p in sorted(template.rglob("*")) if p.is_file()},
        "staged_files": {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()},
    })


def resolve_binary(build_json: Path, output: Path, overlay_root: Path | None = None) -> None:
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
    expected_kernel = ((overlay_root.resolve() if overlay_root else Path.cwd().resolve()) /
                       "crates/media/candle-gen/vendor/candle-kernels").as_posix().lower()
    require(len(kernel_sources) == 1 and kernel_sources[0].startswith("path+") and
            expected_kernel in kernel_sources[0].lower(),
            "diagnostic must use the failed M3 vendored CUDA kernels")
    output.write_text(str(candidates[0].resolve()) + "\n", encoding="utf-8")
    write_json(output.parent / "build-identity.json", {
        "binary_sha256": sha256(candidates[0]), "build_json_sha256": sha256(build_json),
        "candle_core_features": sorted(core_features[0]),
        "vendored_kernel_package_id": kernel_sources[0],
        "derivative_source": str(overlay_root.resolve()) if overlay_root else None,
    })


def execute(args: argparse.Namespace) -> None:
    require(args.engine_sha == ENGINE_SHA, "diagnostic requires the exact failed M3 source")
    selector = getattr(args, "diagnostic", "waveform")
    require(selector in DIAGNOSTICS, "unknown diagnostic selector")
    overlay = getattr(args, "overlay_root", None)
    if selector == "decoder_trace":
        require(overlay is not None, "decoder trace requires declared M3 derivative source")
        overlay = overlay.resolve(strict=True)
        provenance_path = overlay.parent / "overlay-provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        require(provenance.get("engine_sha") == ENGINE_SHA and
                provenance.get("control_sha") == args.control_sha and
                provenance.get("tracked_archive_sha256") == sha256(overlay.parent / "m3-tracked-source.tar.gz") and
                provenance.get("overlay_patch_sha256") == sha256(overlay.parent / "decoder-trace-vae.patch") and
                provenance.get("derivative_vae_sha256") ==
                sha256(overlay / "crates/audio/candle-audio-yue2/src/vae.rs") and
                provenance.get("derivative_tree_sha256") == tree_digest(overlay) and
                provenance.get("cargo_lock_sha256") == sha256(overlay / "Cargo.lock"),
                "decoder trace derivative source provenance changed")
    else:
        require(overlay is None, "source overlay is reserved for decoder trace")
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
    if selector == "decoder_trace":
        shutil.copy2(provenance_path, args.evidence / "overlay-provenance.json")
        shutil.copy2(overlay.parent / "decoder-trace-vae.patch", args.evidence / "decoder-trace-vae.patch")
        require(sha256(args.evidence / "decoder-trace-vae.patch") == provenance["overlay_patch_sha256"],
                "archived decoder trace patch differs from built derivative")
        preflight_decoder_trace_resources(args.evidence)
    idle_run_id = os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID", "")
    idle_receipt = None
    if selector == "decoder_trace" and idle_run_id:
        receipt_source = Path(os.environ.get("YUE2_IDLE_CONTEXT_RECEIPT_DIR") or
                              (Path(os.environ["RUNNER_TEMP"]) / "yue2-reviewed-idle-context"))
        require(receipt_source.is_dir(), "reviewed idle-context receipt absent")
        copied = args.evidence / "idle-context-receipt"
        shutil.copytree(receipt_source, copied, symlinks=False)
        idle_receipt = {"run_id": idle_run_id, "files": [
            {"path": p.relative_to(copied).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(copied.rglob("*")) if p.is_file() and not p.is_symlink()]}
        require(all(p.is_file() and not p.is_symlink() for p in copied.rglob("*")),
                "reviewed idle-context receipt contains unsafe entries")
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
    if selector == "decoder_trace":
        env["YUE2_DECODER_TRACE_PROVENANCE"] = str(provenance_path)
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
    if selector == "decoder_trace":
        report["derivative_source"] = provenance
        report["idle_context_receipt"] = idle_receipt
        report["harness_provenance"] = json.loads(
            (args.evidence / "harness-provenance.json").read_text(encoding="utf-8"))
        report["build_identity"] = json.loads(
            (args.evidence / "build-identity.json").read_text(encoding="utf-8"))
        require(report["build_identity"]["binary_sha256"] == report["binary_sha256"] and
                report["build_identity"]["candle_core_features"] == ["cuda", "cudarc", "default"] and
                report["build_identity"]["derivative_source"] == str(overlay) and
                report["harness_provenance"]["m3_dependency_tuples"] == 205,
                "decoder trace build/source/dependency identity changed")
    write_json(args.evidence / "diagnostic-control.json", report)
    print(json.dumps(report, indent=2), flush=True)
    require(not timed_out and code == 0, "diagnostic execution failed; saved output is not acceptance")
    require(samples and not faults, "diagnostic sampler incomplete")
    require(child.poll() is not None and not after_busy and after_error is None,
            "diagnostic accelerator release unverified")
    require((data / "report.json").is_file(), "diagnostic residual report absent")
    if selector == "decoder_trace":
        verify_decoder_trace_data(data, provenance)
    elif selector in ("first_conv", "first_conv_math"):
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
    prepare.add_argument("--overlay-root", type=Path)
    resolve = commands.add_parser("resolve-binary")
    resolve.add_argument("--build-json", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--overlay-root", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--diagnostic", choices=DIAGNOSTICS, default="waveform")
    for name in ("binary", "reference", "evidence"):
        run.add_argument(f"--{name}", type=Path, required=True)
    for name in ("engine-sha", "control-sha"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--overlay-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-harness":
        prepare_harness(args)
    elif args.command == "resolve-binary":
        resolve_binary(args.build_json, args.output, args.overlay_root)
    else:
        execute(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"yue2-bf16-tile-diagnostic: {error}") from error

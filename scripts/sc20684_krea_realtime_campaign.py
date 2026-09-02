#!/usr/bin/env python3
"""Run and seal the opt-in SC-20684 Krea Realtime real-weight matrix.

The product command is intentionally the *only* measurement producer.  For every
case it receives ``KREA_SC20684_*`` environment variables and must emit exactly
one ``SC20684_KREA_PROVIDER_OBSERVATION <json>`` line.  This launcher neither
accepts metric arguments nor fills in missing values: it binds the observation to
the checked-out source and pinned snapshot, validates the complete six-cell matrix,
then atomically publishes an external receipt directory.

Example (on the coordinator's held Metal lane, never ordinary CI):

  python3 scripts/sc20684_krea_realtime_campaign.py \
    --snapshot /Volumes/Models/huggingface/hub/models--SceneWorks--krea-realtime-14b-mlx/snapshots/e68e9a3d98187fdf6936838ffcf6df5aa48d6626/q4 \
    --output /Users/michael/.codex/worktrees/epic20669/evidence/sc20684/campaign-$(date +%Y%m%dT%H%M%S) \
    --product-command 'cargo test -p mlx-gen-krea-realtime --test integration \
      generate_smoke::sc20684_packed_campaign_observer -- --ignored --nocapture'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STORY = "SC-20684"
SCHEMA_VERSION = 3
RECEIPT_SCHEMA_VERSION = 3
MODEL_REPOSITORY = "SceneWorks/krea-realtime-14b-mlx"
MODEL_REVISION = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626"
OBSERVATION_PREFIX = "SC20684_KREA_PROVIDER_OBSERVATION "
BASELINE_PREFIX = "SC20684_KREA_BASELINE_OBSERVATION "
CASES = tuple((mode, tier) for mode in ("t2v", "i2v", "v2v") for tier in ("q8", "q4"))
MINIMUM_MEMORY_REDUCTION_BYTES = 256 * 1024**2
MINIMUM_MEMORY_REDUCTION_FRACTION = 0.05
MINIMUM_THROUGHPUT_RATIO = 0.95
MAXIMUM_FIRST_FRAME_REGRESSION_FRACTION = 0.05
SOURCE_FILES = (
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/generate.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/pipeline.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/tests/generate_smoke.rs",
    "scripts/sc20684_krea_realtime_campaign.py",
)


class CampaignError(ValueError):
    """A non-terminal row or invalid provenance; no receipt may be published."""


def decision_policy() -> dict[str, Any]:
    """Return the source-frozen policy captured before any product process runs."""
    return {
        "identity": "sc20684-krea-packed-affine-decision-v1",
        "materialMemory": {
            "minimumReductionBytes": MINIMUM_MEMORY_REDUCTION_BYTES,
            "minimumReductionFraction": MINIMUM_MEMORY_REDUCTION_FRACTION,
            "requiredDomains": ["darwinPhysFootprintPeak", "mlxSampledFootprintPeak"],
        },
        "throughputNeutral": {
            "minimumMeanOutputFpsRatio": MINIMUM_THROUGHPUT_RATIO,
            "minimumSteadyDenoiseEquivalentFpsRatio": MINIMUM_THROUGHPUT_RATIO,
            "maximumRequestFirstFrameRegressionFraction": MAXIMUM_FIRST_FRAME_REGRESSION_FRACTION,
        },
        "requiredSafety": {
            "pairedProcessExitCode": 0,
            "denseBaselineProcessExitCode": 0,
            "compiledPackedHandle": True,
            "minimumAcceptedPackedDispatches": 1,
            "denseWindowBytes": 0,
            "scoreMatrixBytes": 0,
            "parityStatus": "pass",
            "qualityStatus": "pass",
            "fallbackCount": 0,
            "cancellationStatus": "pass",
            "releaseVerified": True,
        },
        "coverage": {
            "variedAxes": [
                "requestMode", "cacheTier", "dispatchQueryTokens", "dispatchKeyTokens",
            ],
            "exactPerArmAxes": [
                "batch", "heads", "queryTokens", "keyTokens", "dispatchGeometries",
                "headDim", "groupSize",
                "mask", "width", "height", "frames", "latentFrames",
                "generatedLatentFrames", "hardwareModel", "metalDevice",
            ],
            "fixedProductGeometry": {
                "batch": 1,
                "heads": 40,
                "headDim": 128,
                "groupSize": 64,
                "width": 832,
                "height": 480,
                "frames": 25,
            },
            "eligibilityBoundary": "only exact measured arm geometries are eligible",
            "notSweptByThisSchedule": [
                "alternateHeadCounts",
                "alternateHeadDimensions",
                "alternateGroupSizes",
                "alternateTileShapes",
                "alternateWindowPolicies",
                "alternateGpuFamilies",
            ],
        },
        "aggregation": {
            "tier": "go when at least one exact measured geometry in the tier qualifies",
            "overall": "go when at least one exact measured geometry qualifies",
        },
    }


def _canonical_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise CampaignError(f"{name} must be a finite number >= {minimum}")
    if positive and value <= 0:
        raise CampaignError(f"{name} must be positive")
    return float(value)


def _object(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CampaignError(f"{name} fields differ: expected {sorted(keys)}, got {actual}")
    return value


def _checked_output(output: Path) -> Path:
    output = output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CampaignError("--output must be outside the repository")
    if output.exists():
        raise CampaignError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise CampaignError(f"output parent does not exist: {output.parent}")
    return output


def source_identity(root: Path = ROOT) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    if status.returncode or status.stdout:
        raise CampaignError("campaign source must be a clean git checkout")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if head.returncode or len(head.stdout.strip()) != 40:
        raise CampaignError("cannot resolve checked-out source revision")
    files: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise CampaignError(f"missing product source seam: {relative}")
        files[relative] = _sha256(path.read_bytes())
    return {"repositoryHead": head.stdout.strip(), "files": files}


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    config = snapshot / "config.json"
    if not snapshot.is_dir() or not config.is_file():
        raise CampaignError("snapshot must be a Q4 Krea directory with config.json")
    raw = config.read_bytes()
    if len(raw) > 1_000_000:
        raise CampaignError("snapshot config.json is unreasonably large")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CampaignError(f"invalid snapshot config.json: {error}") from error
    quant = parsed.get("quantization")
    if not isinstance(quant, dict) or quant.get("bits") != 4 or quant.get("group_size") != 64:
        raise CampaignError("snapshot must identify the pinned Q4/group-64 product tier")
    # Deliberately stat only: the launcher must never read the materialized real weights.
    required = ("dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json")
    files: dict[str, dict[str, int]] = {}
    for name in required:
        path = snapshot / name
        if not path.is_file():
            raise CampaignError(f"snapshot missing required product file: {name}")
        stat = path.stat()
        if stat.st_size <= 0:
            raise CampaignError(f"snapshot product file is empty: {name}")
        files[name] = {"size": stat.st_size, "mtimeNs": stat.st_mtime_ns}
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "variant": "q4",
        "configSha256": _sha256(raw),
        "files": files,
    }


def _validate_observation(
    observation: object,
    *,
    expected_mode: str,
    expected_tier: str,
    run_id: str,
    expected_source: dict[str, Any],
    expected_snapshot: dict[str, Any],
) -> dict[str, Any]:
    row = _object(
        observation,
        "provider observation",
        {
            "schemaVersion", "producer", "runId", "case", "source", "model", "input",
            "schedule", "toolchain", "geometry", "compiledHandle", "bytes", "timing",
            "memory", "parity", "quality", "fallback", "cancellation", "output",
        },
    )
    if row["schemaVersion"] != SCHEMA_VERSION or row["producer"] != "mlx-gen-krea-realtime/sc20684":
        raise CampaignError("provider observation schema or producer mismatch")
    if row["runId"] != run_id:
        raise CampaignError("provider observation run id does not match its launched cell")
    case = _object(row["case"], "case", {"mode", "cacheTier"})
    if case != {"mode": expected_mode, "cacheTier": expected_tier}:
        raise CampaignError("provider observation did not report the launched mode/tier")
    if row["source"] != expected_source:
        raise CampaignError("provider observation source identity drift")
    if row["model"] != expected_snapshot:
        raise CampaignError("provider observation model identity drift")

    input_ = _object(
        row["input"],
        "input",
        {"kind", "sha256", "frameCount", "width", "height", "vaeEncoding", "v2vStrength"},
    )
    expected_inputs = {
        "t2v": ("text-only", 0, "none", None),
        "i2v": ("deterministic-gradient-still", 1, "WanVae.encode-mode", None),
        "v2v": ("deterministic-smooth-motion-clip", 25, "WanVae.encode-sample", 0.6),
    }
    expected_kind, expected_frames, expected_encoding, expected_strength = expected_inputs[expected_mode]
    if (
        input_["kind"] != expected_kind
        or input_["frameCount"] != expected_frames
        or input_["vaeEncoding"] != expected_encoding
        or input_["v2vStrength"] != expected_strength
        or not _is_sha256(input_["sha256"])
    ):
        raise CampaignError("provider input identity does not match the launched request mode")
    if _integer(input_["width"], "input.width", minimum=1) != 832 or _integer(input_["height"], "input.height", minimum=1) != 480:
        raise CampaignError("provider input did not use native campaign media geometry")

    schedule = _object(row["schedule"], "schedule", {"kind", "timesteps", "steps", "seed"})
    expected_timesteps = {
        "t2v": [1000.0, 937.0, 833.0, 625.0, 0.0],
        "i2v": [1000.0, 937.0, 833.0, 625.0, 0.0],
        "v2v": [882.3529052734375, 803.5714111328125, 681.8181762695312, 468.75, 0.0],
    }[expected_mode]
    if (
        schedule["kind"] != "source-owned-self-forcing"
        or schedule["timesteps"] != expected_timesteps
        or schedule["steps"] != 5
        or _integer(schedule["seed"], "schedule.seed") != 7
    ):
        raise CampaignError("provider did not execute the complete source-owned five-step schedule")

    toolchain = _object(row["toolchain"], "toolchain", {"os", "arch", "rustc", "cargo", "mlx", "hardwareModel", "metalDevice"})
    for key, value in toolchain.items():
        _nonempty(value, f"toolchain.{key}")
    geometry = _object(
        row["geometry"],
        "geometry",
        {"batch", "heads", "queryTokens", "keyTokens", "dispatchGeometries", "headDim", "groupSize", "mask", "width", "height", "frames", "latentFrames", "generatedLatentFrames"},
    )
    if geometry["batch"] != 1 or geometry["heads"] != 40 or geometry["headDim"] != 128 or geometry["groupSize"] != 64:
        raise CampaignError("provider observation did not use native Krea packed geometry")
    _integer(geometry["queryTokens"], "geometry.queryTokens", minimum=1)
    _integer(geometry["keyTokens"], "geometry.keyTokens", minimum=1)
    if geometry["mask"] not in {"none", "block-causal"}:
        raise CampaignError("provider observation did not use a supported analytic mask")
    if (geometry["width"], geometry["height"], geometry["frames"], geometry["latentFrames"]) != (832, 480, 25, 7):
        raise CampaignError("provider observation did not use the complete native media geometry")
    expected_generated_latents = 6 if expected_mode == "i2v" else 7
    if geometry["generatedLatentFrames"] != expected_generated_latents:
        raise CampaignError("provider observation did not use the product request's generated-latent geometry")

    dispatch_geometries = geometry["dispatchGeometries"]
    if not isinstance(dispatch_geometries, list):
        raise CampaignError("geometry.dispatchGeometries must be a list")
    dispatch_coordinates: set[tuple[int, int]] = set()
    dispatch_count = 0
    for index, item in enumerate(dispatch_geometries):
        dispatch = _object(
            item,
            f"geometry.dispatchGeometries[{index}]",
            {"queryTokens", "keyTokens", "acceptedForwards"},
        )
        query_tokens = _integer(
            dispatch["queryTokens"],
            f"geometry.dispatchGeometries[{index}].queryTokens",
            minimum=1,
        )
        key_tokens = _integer(
            dispatch["keyTokens"],
            f"geometry.dispatchGeometries[{index}].keyTokens",
            minimum=query_tokens,
        )
        accepted = _integer(
            dispatch["acceptedForwards"],
            f"geometry.dispatchGeometries[{index}].acceptedForwards",
            minimum=1,
        )
        coordinate = (query_tokens, key_tokens)
        if coordinate in dispatch_coordinates:
            raise CampaignError("geometry.dispatchGeometries contains a duplicate coordinate")
        dispatch_coordinates.add(coordinate)
        dispatch_count += accepted

    handle = _object(row["compiledHandle"], "compiledHandle", {"identity", "retainedBytes", "compiled", "acceptedDispatches"})
    expected_handle = f"sc20684/krea-packed-affine-{expected_tier}-d128-g64-v1"
    if handle["identity"] != expected_handle or type(handle["compiled"]) is not bool:
        raise CampaignError("compiled handle does not identify the launched cache tier")
    _integer(handle["retainedBytes"], "compiledHandle.retainedBytes")
    accepted_dispatches = _integer(handle["acceptedDispatches"], "compiledHandle.acceptedDispatches")
    if handle["compiled"] != (accepted_dispatches > 0):
        raise CampaignError("compiled handle status contradicts accepted dispatches")
    if dispatch_count != accepted_dispatches:
        raise CampaignError("dispatch geometry coverage contradicts accepted dispatches")
    if dispatch_geometries and (
        geometry["queryTokens"] != max(item["queryTokens"] for item in dispatch_geometries)
        or geometry["keyTokens"] != max(item["keyTokens"] for item in dispatch_geometries)
    ):
        raise CampaignError("summary query/key geometry does not cover accepted dispatches")

    byte_fields = {"persistent", "retainedHandle", "boundedScratch", "denseWindow", "scoreMatrix"}
    bytes_ = _object(row["bytes"], "bytes", byte_fields)
    for key, value in bytes_.items():
        _integer(value, f"bytes.{key}")

    timing_fields = {
        "label", "processWallMs", "candidateWallMs", "loadMs", "conditioningMs", "packedGenerationMs",
        "denseGenerationMs", "packedDecodeMs", "denseDecodeMs", "requestFirstFrameAvailableMs",
        "coldProcessFirstFrameAvailableMs", "meanOutputFrameMs", "meanOutputFps",
        "coldEvaluatedPackedForwardMs", "steadyEvaluatedPackedForwardMeanMs",
        "steadyEvaluatedPackedForwardCount", "compileUpperBoundMs",
        "acceptedPackedAppendMeanMs", "acceptedPackedAppendCount", "progressStepCount",
        "steadyDenoiseStepMeanMs", "steadyFullChunkMs", "steadyDenoiseEquivalentFps",
    }
    timing = _object(row["timing"], "timing", timing_fields)
    if timing["label"] != "fresh-process-full-schedule-loaded-product":
        raise CampaignError("timing label does not identify the fresh full-schedule product path")
    for key in timing_fields - {"label", "steadyEvaluatedPackedForwardCount", "acceptedPackedAppendCount", "progressStepCount"}:
        _number(timing[key], f"timing.{key}", positive=key in {"processWallMs", "candidateWallMs", "packedGenerationMs", "denseGenerationMs", "meanOutputFrameMs", "meanOutputFps", "steadyDenoiseStepMeanMs", "steadyFullChunkMs", "steadyDenoiseEquivalentFps"})
    steady_forward_count = _integer(
        timing["steadyEvaluatedPackedForwardCount"],
        "timing.steadyEvaluatedPackedForwardCount",
    )
    _integer(timing["acceptedPackedAppendCount"], "timing.acceptedPackedAppendCount")
    expected_progress_steps = 10 if expected_mode == "i2v" else 15
    if timing["progressStepCount"] != expected_progress_steps:
        raise CampaignError("timing progress does not cover every product chunk of the five-step schedule")
    expected_request = timing["conditioningMs"] + timing["packedGenerationMs"] + timing["packedDecodeMs"]
    expected_cold = timing["loadMs"] + expected_request
    expected_frame = (timing["packedGenerationMs"] + timing["packedDecodeMs"]) / geometry["frames"]
    relations = (
        (timing["requestFirstFrameAvailableMs"], expected_request, "request first-frame"),
        (timing["coldProcessFirstFrameAvailableMs"], expected_cold, "cold-process first-frame"),
        (timing["meanOutputFrameMs"], expected_frame, "mean output-frame"),
        (timing["meanOutputFps"], 1000.0 / expected_frame, "mean output FPS"),
        (
            timing["compileUpperBoundMs"],
            max(timing["coldEvaluatedPackedForwardMs"] - timing["steadyEvaluatedPackedForwardMeanMs"], 0.0),
            "compile upper bound",
        ),
        (timing["steadyDenoiseEquivalentFps"], 12_000.0 / timing["steadyFullChunkMs"], "steady denoise-equivalent FPS"),
    )
    for actual, expected, name in relations:
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise CampaignError(f"timing {name} is inconsistent with its independently measured phases")
    accounted = timing["loadMs"] + timing["conditioningMs"] + timing["packedGenerationMs"] + timing["denseGenerationMs"] + timing["packedDecodeMs"] + timing["denseDecodeMs"]
    if timing["processWallMs"] < accounted:
        raise CampaignError("timing.processWallMs does not cover all measured phases")
    candidate_accounted = timing["loadMs"] + timing["conditioningMs"] + timing["packedGenerationMs"] + timing["packedDecodeMs"]
    if timing["candidateWallMs"] < candidate_accounted or timing["processWallMs"] < timing["candidateWallMs"]:
        raise CampaignError("candidate/process wall timing does not cover its measured phases")
    cold_forward_count = 1 if timing["coldEvaluatedPackedForwardMs"] > 0 else 0
    if accepted_dispatches != steady_forward_count + cold_forward_count:
        raise CampaignError("accepted dispatch count disagrees with cold/steady timing coverage")

    process_fields = {"physFootprintBytes", "physFootprintPeakBytes"}
    memory = _object(row["memory"], "memory", {"processStart", "weightsLoaded", "packedTerminal", "candidateTerminal", "verificationTerminal", "release", "mlx", "releaseVerified"})
    process_rows = []
    for boundary in ("processStart", "weightsLoaded", "packedTerminal", "candidateTerminal", "verificationTerminal", "release"):
        process = _object(memory[boundary], f"memory.{boundary}", process_fields)
        current = _integer(process["physFootprintBytes"], f"memory.{boundary}.physFootprintBytes", minimum=1)
        peak = _integer(process["physFootprintPeakBytes"], f"memory.{boundary}.physFootprintPeakBytes", minimum=current)
        if peak < current:
            raise CampaignError(f"memory.{boundary} peak is below current footprint")
        process_rows.append(process)
    if any(right["physFootprintPeakBytes"] < left["physFootprintPeakBytes"] for left, right in zip(process_rows, process_rows[1:])):
        raise CampaignError("Darwin physical-footprint peaks must be monotonic within one process")
    mlx_fields = {
        "weightsLoadedActiveBytes", "weightsLoadedCacheBytes", "candidateTerminalActiveBytes",
        "candidateTerminalCacheBytes", "verificationTerminalActiveBytes", "verificationTerminalCacheBytes",
        "exactActivePeakBytes", "sampledActivePeakBytes",
        "sampledCachePeakBytes", "sampledFootprintPeakBytes", "footprintPeakActiveBytes",
        "footprintPeakCacheBytes", "sampleCount", "periodicSampleCount", "samplingSpanMicros",
        "intervalMicros", "maxGapMicros", "releaseActiveBytes", "releaseCacheBytes",
    }
    mlx = _object(memory["mlx"], "memory.mlx", mlx_fields)
    for key in mlx_fields:
        _integer(mlx[key], f"memory.mlx.{key}", minimum=1 if key in {"exactActivePeakBytes", "sampledActivePeakBytes", "sampledFootprintPeakBytes", "sampleCount", "periodicSampleCount", "samplingSpanMicros", "intervalMicros"} else 0)
    if mlx["sampledFootprintPeakBytes"] != mlx["footprintPeakActiveBytes"] + mlx["footprintPeakCacheBytes"]:
        raise CampaignError("memory sampler paired footprint peak is internally inconsistent")
    if mlx["exactActivePeakBytes"] < mlx["sampledActivePeakBytes"]:
        raise CampaignError("exact MLX active peak cannot be below a sampled active value")
    if mlx["maxGapMicros"] > mlx["intervalMicros"] * 10:
        raise CampaignError("memory sampler coverage has an excessive gap")
    if type(memory["releaseVerified"]) is not bool:
        raise CampaignError("memory.releaseVerified must be a boolean")
    release_limit = memory["verificationTerminal"]["physFootprintBytes"] + 512 * 1024 * 1024
    resources_released = not (
        mlx["releaseActiveBytes"] > mlx["verificationTerminalActiveBytes"]
        or mlx["releaseCacheBytes"] > mlx["verificationTerminalCacheBytes"]
        or memory["release"]["physFootprintBytes"] > release_limit
    )

    parity = _object(row["parity"], "parity", {"status", "candidateTier", "maxAbsError", "tolerance"})
    if parity["status"] not in {"pass", "fail"} or parity["candidateTier"] != expected_tier:
        raise CampaignError("packed parity status is malformed or tier-substituted")
    if type(parity["maxAbsError"]) not in (int, float) or type(parity["tolerance"]) not in (int, float) or parity["maxAbsError"] < 0 or parity["tolerance"] <= 0:
        raise CampaignError("packed parity numbers are malformed")
    if (parity["status"] == "pass") != (parity["maxAbsError"] <= parity["tolerance"]):
        raise CampaignError("packed parity status contradicts its measured values")

    quality = _object(
        row["quality"],
        "quality",
        {"status", "candidateTier", "metric", "maxAbsRgbU8", "maxAbsRgbU8Tolerance", "meanAbsRgbU8", "meanAbsRgbU8Tolerance", "packedMeanTemporalDelta", "denseMeanTemporalDelta", "temporalDeltaDrift", "temporalDeltaDriftTolerance", "acknowledged"},
    )
    if quality["status"] not in {"pass", "fail"} or quality["candidateTier"] != expected_tier:
        raise CampaignError("packed quality status is malformed or tier-substituted")
    _nonempty(quality["metric"], "quality.metric")
    for key in ("maxAbsRgbU8", "maxAbsRgbU8Tolerance", "meanAbsRgbU8", "meanAbsRgbU8Tolerance", "packedMeanTemporalDelta", "denseMeanTemporalDelta", "temporalDeltaDrift", "temporalDeltaDriftTolerance"):
        _number(quality[key], f"quality.{key}", positive=key in {"maxAbsRgbU8Tolerance", "meanAbsRgbU8Tolerance", "temporalDeltaDriftTolerance"})
    quality_pass = not (
        quality["maxAbsRgbU8"] > quality["maxAbsRgbU8Tolerance"]
        or quality["meanAbsRgbU8"] > quality["meanAbsRgbU8Tolerance"]
        or quality["temporalDeltaDrift"] > quality["temporalDeltaDriftTolerance"]
    )
    if (quality["status"] == "pass") != quality_pass:
        raise CampaignError("packed quality status contradicts its measured values")
    if type(quality["acknowledged"]) is not bool or (expected_tier == "q4" and not quality["acknowledged"]):
        raise CampaignError("Q4 requires its separate explicit quality acknowledgement")

    fallback = _object(row["fallback"], "fallback", {"count", "reason"})
    fallback_count = _integer(fallback["count"], "fallback.count")
    if (
        (fallback_count == 0 and fallback["reason"] is not None)
        or (fallback_count > 0 and not isinstance(fallback["reason"], str))
        or (isinstance(fallback["reason"], str) and not fallback["reason"].strip())
    ):
        raise CampaignError("fallback count and reason are inconsistent")
    cancellation = _object(row["cancellation"], "cancellation", {"status", "requests", "partialStateMutation", "scratchReleased"})
    if (
        cancellation["status"] not in {"pass", "fail"}
        or type(cancellation["partialStateMutation"]) is not bool
        or type(cancellation["scratchReleased"]) is not bool
    ):
        raise CampaignError("cancellation evidence is malformed")
    _integer(cancellation["requests"], "cancellation.requests", minimum=1)
    cancellation_clean = (
        cancellation["partialStateMutation"] is False
        and cancellation["scratchReleased"] is True
    )
    if (cancellation["status"] == "pass") != cancellation_clean:
        raise CampaignError("cancellation status contradicts its measured outcome")
    if memory["releaseVerified"] != (resources_released and cancellation_clean):
        raise CampaignError("release status contradicts its terminal resource boundaries")
    output = _object(row["output"], "output", {"status", "sha256", "denseSha256", "artifacts"})
    if output["status"] != "generated" or not _is_sha256(output["sha256"]) or not _is_sha256(output["denseSha256"]):
        raise CampaignError("product output evidence is missing")
    artifacts = output["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != geometry["frames"]:
        raise CampaignError("product output must inventory every generated review frame")
    paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        item = _object(artifact, f"output.artifacts[{index}]", {"path", "sha256", "bytes"})
        path = _nonempty(item["path"], f"output.artifacts[{index}].path")
        if Path(path).name != path or not path.endswith(".ppm") or path in paths:
            raise CampaignError("review artifact paths must be unique safe PPM basenames")
        paths.add(path)
        if not _is_sha256(item["sha256"]):
            raise CampaignError("review artifact hash is malformed")
        _integer(item["bytes"], f"output.artifacts[{index}].bytes", minimum=1)
    return row


def _validate_baseline_observation(
    observation: object,
    *,
    expected_mode: str,
    expected_tier: str,
    run_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    row = _object(
        observation,
        "dense baseline observation",
        {
            "schemaVersion", "producer", "runId", "case", "source", "model", "input",
            "schedule", "toolchain", "geometry", "timing", "memory", "output",
        },
    )
    if row["schemaVersion"] != SCHEMA_VERSION or row["producer"] != "mlx-gen-krea-realtime/sc20684-dense-baseline":
        raise CampaignError("dense baseline schema or producer mismatch")
    if row["runId"] != run_id or row["case"] != {"mode": expected_mode, "cacheTier": expected_tier}:
        raise CampaignError("dense baseline did not report its launched identity")
    for key in ("source", "model", "input", "schedule", "toolchain"):
        if row[key] != candidate[key]:
            raise CampaignError(f"dense baseline {key} differs from its paired candidate")

    baseline_geometry = _object(
        row["geometry"],
        "dense baseline geometry",
        set(candidate["geometry"]),
    )
    if baseline_geometry["dispatchGeometries"] != []:
        raise CampaignError("dense baseline cannot claim packed dispatch geometry")
    for key, value in candidate["geometry"].items():
        if key != "dispatchGeometries" and baseline_geometry[key] != value:
            raise CampaignError("dense baseline geometry differs from its paired candidate")

    geometry = row["geometry"]
    timing_fields = {
        "label", "processWallMs", "loadMs", "conditioningMs", "generationMs", "decodeMs",
        "requestFirstFrameAvailableMs", "coldProcessFirstFrameAvailableMs", "meanOutputFrameMs",
        "meanOutputFps", "progressStepCount", "steadyDenoiseStepMeanMs", "steadyFullChunkMs", "steadyDenoiseEquivalentFps",
    }
    timing = _object(row["timing"], "dense baseline timing", timing_fields)
    if timing["label"] != "fresh-process-full-schedule-dense-read-window-baseline":
        raise CampaignError("dense baseline timing label mismatch")
    for key in timing_fields - {"label", "progressStepCount"}:
        _number(timing[key], f"dense baseline timing.{key}", positive=key in {"processWallMs", "generationMs", "decodeMs", "requestFirstFrameAvailableMs", "coldProcessFirstFrameAvailableMs", "meanOutputFrameMs", "meanOutputFps", "steadyDenoiseStepMeanMs", "steadyFullChunkMs", "steadyDenoiseEquivalentFps"})
    expected_progress_steps = 10 if expected_mode == "i2v" else 15
    if timing["progressStepCount"] != expected_progress_steps:
        raise CampaignError("dense baseline progress does not cover the complete schedule")
    expected_request = timing["conditioningMs"] + timing["generationMs"] + timing["decodeMs"]
    expected_cold = timing["loadMs"] + expected_request
    expected_frame = (timing["generationMs"] + timing["decodeMs"]) / geometry["frames"]
    for actual, expected, name in (
        (timing["requestFirstFrameAvailableMs"], expected_request, "request first-frame"),
        (timing["coldProcessFirstFrameAvailableMs"], expected_cold, "cold first-frame"),
        (timing["meanOutputFrameMs"], expected_frame, "mean output-frame"),
        (timing["meanOutputFps"], 1000.0 / expected_frame, "mean output FPS"),
        (timing["steadyDenoiseEquivalentFps"], 12_000.0 / timing["steadyFullChunkMs"], "steady denoise-equivalent FPS"),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise CampaignError(f"dense baseline {name} timing is inconsistent")
    if timing["processWallMs"] < timing["loadMs"] + expected_request:
        raise CampaignError("dense baseline process wall does not cover all measured phases")

    process_fields = {"physFootprintBytes", "physFootprintPeakBytes"}
    memory = _object(row["memory"], "dense baseline memory", {"processStart", "weightsLoaded", "generationTerminal", "release", "mlx", "releaseVerified"})
    process_rows = []
    for boundary in ("processStart", "weightsLoaded", "generationTerminal", "release"):
        process = _object(memory[boundary], f"dense baseline memory.{boundary}", process_fields)
        current = _integer(process["physFootprintBytes"], f"dense baseline memory.{boundary}.physFootprintBytes", minimum=1)
        _integer(process["physFootprintPeakBytes"], f"dense baseline memory.{boundary}.physFootprintPeakBytes", minimum=current)
        process_rows.append(process)
    if any(right["physFootprintPeakBytes"] < left["physFootprintPeakBytes"] for left, right in zip(process_rows, process_rows[1:])):
        raise CampaignError("dense baseline Darwin footprint peaks are not monotonic")
    mlx_fields = {
        "weightsLoadedActiveBytes", "weightsLoadedCacheBytes", "generationTerminalActiveBytes",
        "generationTerminalCacheBytes", "exactActivePeakBytes", "sampledActivePeakBytes",
        "sampledCachePeakBytes", "sampledFootprintPeakBytes", "footprintPeakActiveBytes",
        "footprintPeakCacheBytes", "sampleCount", "periodicSampleCount", "samplingSpanMicros",
        "intervalMicros", "maxGapMicros", "releaseActiveBytes", "releaseCacheBytes",
    }
    mlx = _object(memory["mlx"], "dense baseline memory.mlx", mlx_fields)
    for key in mlx_fields:
        _integer(mlx[key], f"dense baseline memory.mlx.{key}", minimum=1 if key in {"exactActivePeakBytes", "sampledActivePeakBytes", "sampledFootprintPeakBytes", "sampleCount", "periodicSampleCount", "samplingSpanMicros", "intervalMicros"} else 0)
    if mlx["sampledFootprintPeakBytes"] != mlx["footprintPeakActiveBytes"] + mlx["footprintPeakCacheBytes"]:
        raise CampaignError("dense baseline paired allocator peak is inconsistent")
    if mlx["exactActivePeakBytes"] < mlx["sampledActivePeakBytes"] or mlx["maxGapMicros"] > mlx["intervalMicros"] * 10:
        raise CampaignError("dense baseline allocator coverage is inconsistent")
    if type(memory["releaseVerified"]) is not bool:
        raise CampaignError("dense baseline memory.releaseVerified must be a boolean")
    release_limit = memory["generationTerminal"]["physFootprintBytes"] + 512 * 1024 * 1024
    resources_released = not (
        mlx["releaseActiveBytes"] > mlx["generationTerminalActiveBytes"]
        or mlx["releaseCacheBytes"] > mlx["generationTerminalCacheBytes"]
        or memory["release"]["physFootprintBytes"] > release_limit
    )
    if memory["releaseVerified"] != resources_released:
        raise CampaignError("dense baseline release status contradicts its resource boundaries")
    output = _object(row["output"], "dense baseline output", {"status", "sha256"})
    if output["status"] != "generated" or not _is_sha256(output["sha256"]):
        raise CampaignError("dense baseline output evidence is missing")
    if output["sha256"] != candidate["output"]["denseSha256"]:
        raise CampaignError("fresh dense baseline output differs from the paired dense reference")
    return row


def _validate_artifact_directory(row: dict[str, Any], directory: Path) -> None:
    if not directory.is_dir():
        raise CampaignError("provider did not create its external review artifact directory")
    expected = {item["path"]: item for item in row["output"]["artifacts"]}
    actual = {path.name: path for path in directory.iterdir() if path.is_file()}
    if (
        set(actual) != set(expected)
        or any(path.is_dir() or path.is_symlink() for path in directory.iterdir())
    ):
        raise CampaignError("external review artifacts differ from the provider inventory")
    for name, path in actual.items():
        raw = path.read_bytes()
        if len(raw) != expected[name]["bytes"] or _sha256(raw) != expected[name]["sha256"]:
            raise CampaignError(f"external review artifact identity drift: {name}")


def _file_identity(path: Path, relative: str) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": relative, "sha256": _sha256(raw), "bytes": len(raw)}


def _validate_evidence_file(identity: object, root: Path, name: str) -> None:
    item = _object(identity, name, {"path", "sha256", "bytes"})
    relative = _nonempty(item["path"], f"{name}.path")
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file() or path.is_symlink():
        raise CampaignError(f"{name} is not a safe regular evidence file")
    raw = path.read_bytes()
    if not _is_sha256(item["sha256"]) or len(raw) != item["bytes"] or _sha256(raw) != item["sha256"]:
        raise CampaignError(f"{name} identity drift")


def _read_observation(stdout: str, prefix: str = OBSERVATION_PREFIX) -> dict[str, Any]:
    candidates = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(candidates) != 1:
        raise CampaignError(f"product command must emit exactly one observation line, got {len(candidates)}")
    try:
        result = json.loads(candidates[0])
    except json.JSONDecodeError as error:
        raise CampaignError(f"malformed provider observation: {error}") from error
    return result


def _exact_geometry(row: dict[str, Any]) -> dict[str, Any]:
    observation = row["observation"]
    toolchain = observation["toolchain"]
    return {
        "mode": row["mode"],
        "cacheTier": row["cacheTier"],
        "geometry": observation["geometry"],
        "hardware": {
            "hardwareModel": toolchain["hardwareModel"],
            "metalDevice": toolchain["metalDevice"],
        },
    }


def arm_decision(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Reduce one terminal candidate/baseline pair without discarding a measured No-go."""
    candidate = row["observation"]
    baseline = row["baseline"]
    comparison = row["comparison"]
    material = policy["materialMemory"]
    throughput = policy["throughputNeutral"]
    safety = policy["requiredSafety"]

    phys_saving = (
        comparison["baselinePhysFootprintPeakBytes"]
        - comparison["candidatePhysFootprintPeakBytes"]
    )
    mlx_saving = (
        comparison["baselineMlxFootprintPeakBytes"]
        - comparison["candidateMlxFootprintPeakBytes"]
    )
    mean_output_fps_ratio = (
        comparison["candidateMeanOutputFps"] / comparison["baselineMeanOutputFps"]
    )
    steady_fps_ratio = (
        comparison["candidateSteadyDenoiseEquivalentFps"]
        / comparison["baselineSteadyDenoiseEquivalentFps"]
    )
    first_frame_ratio = (
        comparison["candidateRequestFirstFrameMs"]
        / comparison["baselineRequestFirstFrameMs"]
    )
    criteria = {
        "pairedProcessExitZero": (
            row["processExitCodes"]["paired"] == safety["pairedProcessExitCode"]
        ),
        "denseBaselineProcessExitZero": (
            row["processExitCodes"]["dense-baseline"]
            == safety["denseBaselineProcessExitCode"]
        ),
        "physFootprintReductionMaterial": (
            phys_saving >= material["minimumReductionBytes"]
            and comparison["physFootprintReductionFraction"]
            >= material["minimumReductionFraction"]
        ),
        "mlxFootprintReductionMaterial": (
            mlx_saving >= material["minimumReductionBytes"]
            and comparison["mlxFootprintReductionFraction"]
            >= material["minimumReductionFraction"]
        ),
        "meanOutputThroughputNeutral": (
            mean_output_fps_ratio >= throughput["minimumMeanOutputFpsRatio"]
        ),
        "steadyDenoiseThroughputNeutral": (
            steady_fps_ratio >= throughput["minimumSteadyDenoiseEquivalentFpsRatio"]
        ),
        "requestFirstFrameNeutral": (
            first_frame_ratio
            <= 1.0 + throughput["maximumRequestFirstFrameRegressionFraction"]
        ),
        "compiledPackedHandle": (
            candidate["compiledHandle"]["compiled"] is safety["compiledPackedHandle"]
        ),
        "acceptedPackedDispatch": (
            candidate["compiledHandle"]["acceptedDispatches"]
            >= safety["minimumAcceptedPackedDispatches"]
        ),
        "zeroDenseWindowBytes": (
            candidate["bytes"]["denseWindow"] == safety["denseWindowBytes"]
        ),
        "zeroScoreMatrixBytes": (
            candidate["bytes"]["scoreMatrix"] == safety["scoreMatrixBytes"]
        ),
        "parityPassed": candidate["parity"]["status"] == safety["parityStatus"],
        "qualityPassed": candidate["quality"]["status"] == safety["qualityStatus"],
        "noDenseFallback": candidate["fallback"]["count"] == safety["fallbackCount"],
        "cancellationPassed": (
            candidate["cancellation"]["status"] == safety["cancellationStatus"]
        ),
        "candidateReleaseVerified": (
            candidate["memory"]["releaseVerified"] is safety["releaseVerified"]
        ),
        "baselineReleaseVerified": (
            baseline["memory"]["releaseVerified"] is safety["releaseVerified"]
        ),
    }
    failed = [name for name, passed in criteria.items() if not passed]
    geometry = _exact_geometry(row)
    return {
        "decision": "go" if not failed else "no-go",
        "failedCriteria": failed,
        "criteria": criteria,
        "measurements": {
            "physFootprintSavingBytes": phys_saving,
            "physFootprintReductionFraction": comparison["physFootprintReductionFraction"],
            "mlxFootprintSavingBytes": mlx_saving,
            "mlxFootprintReductionFraction": comparison["mlxFootprintReductionFraction"],
            "meanOutputFpsRatio": mean_output_fps_ratio,
            "steadyDenoiseEquivalentFpsRatio": steady_fps_ratio,
            "requestFirstFrameRatio": first_frame_ratio,
        },
        "observedGeometry": geometry,
        "eligibleGeometry": geometry if not failed else None,
    }


def campaign_decision(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    eligible: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row['mode']}/{row['cacheTier']}"
        reduced = arm_decision(row, policy)
        if row.get("decision") != reduced:
            raise CampaignError(f"sealed arm decision drift: {key}")
        arms[key] = reduced
        if reduced["eligibleGeometry"] is not None:
            eligible.append(reduced["eligibleGeometry"])

    tiers: dict[str, dict[str, Any]] = {}
    for tier in ("q8", "q4"):
        tier_eligible = [item for item in eligible if item["cacheTier"] == tier]
        tiers[tier] = {
            "decision": "go" if tier_eligible else "no-go",
            "eligibleGeometries": tier_eligible,
        }
    return {
        "policy": policy,
        "policySha256": _canonical_sha256(policy),
        "arms": arms,
        "tiers": tiers,
        "overall": {
            "decision": "go" if eligible else "no-go",
            "eligibleGeometries": eligible,
        },
    }


def run_matrix(
    command: str,
    snapshot: Path,
    source: dict[str, Any],
    model: dict[str, Any],
    policy: dict[str, Any],
    timeout: int,
    evidence_root: Path,
) -> list[dict[str, Any]]:
    try:
        argv = __import__("shlex").split(command)
    except ValueError as error:
        raise CampaignError(f"invalid --product-command: {error}") from error
    if not argv:
        raise CampaignError("--product-command must not be empty")
    rows: list[dict[str, Any]] = []
    for mode, tier in CASES:
        cell_name = f"{mode}-{tier}"
        artifact_dir = evidence_root / "artifacts" / cell_name
        transcript_dir = evidence_root / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        cell_started = time.monotonic_ns()
        observations: dict[str, Any] = {}
        transcripts: dict[str, dict[str, Any]] = {}
        for role in ("paired", "dense-baseline"):
            run_id = str(uuid.uuid4())
            env = os.environ.copy()
            env.update({
                "KREA_SC20684_RUN_ID": run_id,
                "KREA_SC20684_REQUEST_MODE": mode,
                "KREA_SC20684_CACHE_TIER": tier,
                "KREA_SC20684_SNAPSHOT_DIR": str(snapshot),
                "KREA_SC20684_MODEL_REPOSITORY": MODEL_REPOSITORY,
                "KREA_SC20684_MODEL_REVISION": MODEL_REVISION,
                "KREA_SC20684_MEASUREMENT_ROLE": role,
                "KREA_SC20684_ARTIFACT_DIR": str(artifact_dir),
            })
            if tier == "q4":
                # The Q4 arm is a separately named experiment.  This is a selector only; measured
                # quality still has to arrive from the provider-owned observer and pass reduction.
                env["KREA_SC20684_Q4_QUALITY_ARM"] = "acknowledged"
            try:
                result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as error:
                raise CampaignError(f"{mode}/{tier}/{role} product command timed out after {timeout}s") from error
            stdout_path = transcript_dir / f"{cell_name}.{role}.stdout.log"
            stderr_path = transcript_dir / f"{cell_name}.{role}.stderr.log"
            stdout_path.write_bytes(result.stdout)
            stderr_path.write_bytes(result.stderr)
            transcripts[role] = {
                "stdout": _file_identity(stdout_path, f"transcripts/{stdout_path.name}"),
                "stderr": _file_identity(stderr_path, f"transcripts/{stderr_path.name}"),
            }
            observations[f"{role}-process-exit-code"] = result.returncode
            stdout = result.stdout.decode("utf-8", errors="replace")
            if role == "paired":
                observations[role] = _validate_observation(
                    _read_observation(stdout),
                    expected_mode=mode,
                    expected_tier=tier,
                    run_id=run_id,
                    expected_source=source,
                    expected_snapshot=model,
                )
                _validate_artifact_directory(observations[role], artifact_dir)
            else:
                observations[role] = _validate_baseline_observation(
                    _read_observation(stdout, BASELINE_PREFIX),
                    expected_mode=mode,
                    expected_tier=tier,
                    run_id=run_id,
                    candidate=observations["paired"],
                )
        candidate = observations["paired"]
        baseline = observations["dense-baseline"]
        candidate_phys = candidate["memory"]["candidateTerminal"]["physFootprintPeakBytes"]
        baseline_phys = baseline["memory"]["generationTerminal"]["physFootprintPeakBytes"]
        candidate_mlx = candidate["memory"]["mlx"]["sampledFootprintPeakBytes"]
        baseline_mlx = baseline["memory"]["mlx"]["sampledFootprintPeakBytes"]
        row = {
            "mode": mode,
            "cacheTier": tier,
            "launcherElapsedNs": time.monotonic_ns() - cell_started,
            "artifactDirectory": f"artifacts/{cell_name}",
            "transcripts": transcripts,
            "processExitCodes": {
                "paired": observations["paired-process-exit-code"],
                "dense-baseline": observations["dense-baseline-process-exit-code"],
            },
            "comparison": {
                "candidatePhysFootprintPeakBytes": candidate_phys,
                "baselinePhysFootprintPeakBytes": baseline_phys,
                "physFootprintReductionFraction": (baseline_phys - candidate_phys) / baseline_phys,
                "candidateMlxFootprintPeakBytes": candidate_mlx,
                "baselineMlxFootprintPeakBytes": baseline_mlx,
                "mlxFootprintReductionFraction": (baseline_mlx - candidate_mlx) / baseline_mlx,
                "candidateRequestFirstFrameMs": candidate["timing"]["requestFirstFrameAvailableMs"],
                "baselineRequestFirstFrameMs": baseline["timing"]["requestFirstFrameAvailableMs"],
                "candidateMeanOutputFps": candidate["timing"]["meanOutputFps"],
                "baselineMeanOutputFps": baseline["timing"]["meanOutputFps"],
                "candidateSteadyDenoiseEquivalentFps": candidate["timing"]["steadyDenoiseEquivalentFps"],
                "baselineSteadyDenoiseEquivalentFps": baseline["timing"]["steadyDenoiseEquivalentFps"],
            },
            "baseline": baseline,
            "observation": candidate,
        }
        row["decision"] = arm_decision(row, policy)
        rows.append(row)
    return rows


def validate_matrix(
    rows: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> None:
    if len(rows) != len(CASES):
        raise CampaignError("matrix is incomplete")
    coordinates = [(row.get("mode"), row.get("cacheTier")) for row in rows]
    if set(coordinates) != set(CASES) or len(set(coordinates)) != len(CASES):
        raise CampaignError("matrix must contain every T2V/I2V/V2V and Q8/Q4 cell exactly once")
    run_ids = [observation.get("runId") for row in rows for observation in (row["observation"], row["baseline"])]
    if len(run_ids) != len(set(run_ids)):
        raise CampaignError("matrix cells reused a provider observation")
    artifact_directories = [row.get("artifactDirectory") for row in rows]
    expected_directories = {f"artifacts/{mode}-{tier}" for mode, tier in CASES}
    if set(artifact_directories) != expected_directories or len(set(artifact_directories)) != len(CASES):
        raise CampaignError("matrix artifact directories must identify every cell exactly once")
    transcript_paths = []
    for row in rows:
        exit_codes = _object(
            row.get("processExitCodes"),
            "matrix process exit codes",
            {"paired", "dense-baseline"},
        )
        for role, exit_code in exit_codes.items():
            _integer(exit_code, f"matrix process exit codes.{role}")
        transcripts = _object(row.get("transcripts"), "matrix transcripts", {"paired", "dense-baseline"})
        for role in ("paired", "dense-baseline"):
            role_transcripts = _object(transcripts[role], f"matrix transcripts.{role}", {"stdout", "stderr"})
            for stream in ("stdout", "stderr"):
                item = _object(role_transcripts[stream], f"matrix transcripts.{role}.{stream}", {"path", "sha256", "bytes"})
                if not _is_sha256(item["sha256"]):
                    raise CampaignError("matrix transcript hash is malformed")
                _integer(item["bytes"], "matrix transcript bytes")
                transcript_paths.append(_nonempty(item["path"], "matrix transcript path"))
    expected_transcripts = {
        f"transcripts/{mode}-{tier}.{role}.{stream}.log"
        for mode, tier in CASES
        for role in ("paired", "dense-baseline")
        for stream in ("stdout", "stderr")
    }
    if set(transcript_paths) != expected_transcripts or len(set(transcript_paths)) != len(expected_transcripts):
        raise CampaignError("matrix transcripts must identify both streams for every cell exactly once")
    comparison_fields = {
        "candidatePhysFootprintPeakBytes", "baselinePhysFootprintPeakBytes",
        "physFootprintReductionFraction", "candidateMlxFootprintPeakBytes",
        "baselineMlxFootprintPeakBytes", "mlxFootprintReductionFraction",
        "candidateRequestFirstFrameMs", "baselineRequestFirstFrameMs",
        "candidateMeanOutputFps", "baselineMeanOutputFps",
        "candidateSteadyDenoiseEquivalentFps", "baselineSteadyDenoiseEquivalentFps",
    }
    for row in rows:
        candidate = row["observation"]
        baseline = row["baseline"]
        if baseline["output"]["sha256"] != candidate["output"]["denseSha256"]:
            raise CampaignError("matrix dense baseline output differs from the paired reference")
        comparison = _object(row.get("comparison"), "matrix comparison", comparison_fields)
        candidate_phys = candidate["memory"]["candidateTerminal"]["physFootprintPeakBytes"]
        baseline_phys = baseline["memory"]["generationTerminal"]["physFootprintPeakBytes"]
        candidate_mlx = candidate["memory"]["mlx"]["sampledFootprintPeakBytes"]
        baseline_mlx = baseline["memory"]["mlx"]["sampledFootprintPeakBytes"]
        expected = {
            "candidatePhysFootprintPeakBytes": candidate_phys,
            "baselinePhysFootprintPeakBytes": baseline_phys,
            "physFootprintReductionFraction": (baseline_phys - candidate_phys) / baseline_phys,
            "candidateMlxFootprintPeakBytes": candidate_mlx,
            "baselineMlxFootprintPeakBytes": baseline_mlx,
            "mlxFootprintReductionFraction": (baseline_mlx - candidate_mlx) / baseline_mlx,
            "candidateRequestFirstFrameMs": candidate["timing"]["requestFirstFrameAvailableMs"],
            "baselineRequestFirstFrameMs": baseline["timing"]["requestFirstFrameAvailableMs"],
            "candidateMeanOutputFps": candidate["timing"]["meanOutputFps"],
            "baselineMeanOutputFps": baseline["timing"]["meanOutputFps"],
            "candidateSteadyDenoiseEquivalentFps": candidate["timing"]["steadyDenoiseEquivalentFps"],
            "baselineSteadyDenoiseEquivalentFps": baseline["timing"]["steadyDenoiseEquivalentFps"],
        }
        for key, value in expected.items():
            if type(comparison[key]) not in (int, float) or not math.isclose(comparison[key], value, rel_tol=1e-12, abs_tol=1e-12):
                raise CampaignError(f"matrix comparison drift: {key}")
    campaign_decision(rows, policy if policy is not None else decision_policy())


def publish(
    output: Path,
    *,
    source: dict[str, Any],
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    evidence_root: Path,
    policy: dict[str, Any] | None = None,
) -> Path:
    output = _checked_output(output)
    policy = policy if policy is not None else decision_policy()
    validate_matrix(rows, policy)
    decision = campaign_decision(rows, policy)
    receipt = {
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "story": STORY,
        "status": f"terminal-{decision['overall']['decision']}",
        "launcher": {"python": platform.python_version(), "system": platform.platform()},
        "source": source,
        "model": model,
        "decision": decision,
        "matrix": rows,
    }
    parent = output.parent
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent))
    try:
        for row in rows:
            cell_name = f"{row['mode']}-{row['cacheTier']}"
            _validate_artifact_directory(row["observation"], evidence_root / "artifacts" / cell_name)
            for role in ("paired", "dense-baseline"):
                _validate_evidence_file(row["transcripts"][role]["stdout"], evidence_root, f"{role} stdout transcript")
                _validate_evidence_file(row["transcripts"][role]["stderr"], evidence_root, f"{role} stderr transcript")
        shutil.copytree(evidence_root / "artifacts", temporary / "artifacts")
        shutil.copytree(evidence_root / "transcripts", temporary / "transcripts")
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (temporary / "receipt.json").write_bytes(encoded)
        (temporary / "receipt.json.sha256").write_text(f"{_sha256(encoded)}  receipt.json\n", encoding="utf-8")
        manifest_rows = []
        for path in sorted(path for path in temporary.rglob("*") if path.is_file()):
            relative = path.relative_to(temporary).as_posix()
            manifest_rows.append(f"{_sha256(path.read_bytes())}  {relative}")
        (temporary / "manifest.sha256").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True, help="pinned Q4 Krea Realtime snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="new external receipt directory")
    parser.add_argument("--product-command", required=True, help="product-owned observer command, executed in fresh paired and dense-baseline processes per matrix cell")
    parser.add_argument("--timeout", type=int, default=7200, help="per-cell command timeout in seconds")
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise CampaignError("--timeout must be positive")
        output = _checked_output(args.output)
        policy = decision_policy()
        source = source_identity()
        model = snapshot_identity(args.snapshot)
        evidence_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.evidence-", dir=output.parent))
        try:
            rows = run_matrix(
                args.product_command,
                args.snapshot.resolve(),
                source,
                model,
                policy,
                args.timeout,
                evidence_root,
            )
            if source_identity() != source:
                raise CampaignError("campaign source changed after decision policy was frozen")
            publish(
                output,
                source=source,
                model=model,
                rows=rows,
                evidence_root=evidence_root,
                policy=policy,
            )
        finally:
            shutil.rmtree(evidence_root, ignore_errors=True)
    except CampaignError as error:
        print(f"SC-20684 campaign refused: {error}", file=sys.stderr)
        return 1
    result = campaign_decision(rows, policy)
    print(json.dumps({
        "status": "published",
        "decision": result["overall"]["decision"],
        "output": str(output),
        "cells": len(CASES),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

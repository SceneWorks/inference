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
    --safety-policy /absolute/path/sc20684-safety-policy.json \
    --resume-dir /absolute/external/path/sc20684-resume \
    --product-command '/absolute/path/to/prebuilt/generate_smoke \
      --ignored --nocapture sc20684_packed_campaign_observer'

The executable must be built and sealed before invoking this launcher. No static
whole-process peak bound exists for these cells; each role is admitted by the
supervisor's runtime guards (watchdog cap, host reserve, deadline, sampling) when
pre-spawn host free RAM covers cap plus reserve. The admission, with the unknown
peak recorded as null plus reason, is sealed with every role. A pre-spawn refusal,
watchdog abort or failed child is written as a sealed unaccepted log record and
leaves the campaign incomplete; it is never an accepted role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import media_campaign_supervisor as supervisor

ROOT = Path(__file__).resolve().parents[1]
STORY = "SC-20684"
SCHEMA_VERSION = 6
RECEIPT_SCHEMA_VERSION = 6
MODEL_REPOSITORY = "SceneWorks/krea-realtime-14b-mlx"
MODEL_REVISION = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626"
OBSERVATION_PREFIX = "SC20684_KREA_PROVIDER_OBSERVATION "
BASELINE_PREFIX = "SC20684_KREA_BASELINE_OBSERVATION "
CASES = tuple((mode, tier) for mode in ("t2v", "i2v", "v2v") for tier in ("q8", "q4"))
MINIMUM_MEMORY_REDUCTION_BYTES = 256 * 1024**2
MINIMUM_MEMORY_REDUCTION_FRACTION = 0.05
MINIMUM_THROUGHPUT_RATIO = 0.95
MAXIMUM_FIRST_FRAME_REGRESSION_FRACTION = 0.05
PARITY_MAX_ABS_ERROR_BY_TIER = {"q8": 0.25, "q4": 0.75}
QUALITY_THRESHOLDS_BY_TIER = {
    "q8": {
        "maxAbsRgbU8": 32,
        "meanAbsRgbU8": 1.0,
        "temporalDeltaDrift": 4.0,
    },
    "q4": {
        "maxAbsRgbU8": 96,
        "meanAbsRgbU8": 3.0,
        "temporalDeltaDrift": 12.0,
    },
}
PACKED_GEOMETRY_IDENTITY = {
    "queryTile": 8,
    "keyTile": 8,
    "dtypes": {
        "packedCodes": "uint32",
        "packedMetadata": "bfloat16",
        "queryAndCurrentKv": "bfloat16",
        "accumulator": "float32",
    },
}
DECODE_PLAN_IDENTITY = {
    "kind": "wan-z16-spatial-tail-v1",
    "spatialTilePx": 256,
    "spatialOverlapPx": 64,
    "temporalTileFrames": 32,
    "temporalOverlapFrames": 16,
    "rawDecodedFrames": 28,
}
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
        "identity": "sc20684-krea-packed-affine-decision-v3",
        "decodePlan": DECODE_PLAN_IDENTITY,
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
        "accuracy": {
            "metric": "paired-rgb-and-temporal-delta",
            "parityMaxAbsErrorByTier": PARITY_MAX_ABS_ERROR_BY_TIER,
            "qualityByTier": QUALITY_THRESHOLDS_BY_TIER,
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
                "headDim", "groupSize", "queryTile", "keyTile", "dtypes",
                "mask", "width", "height", "frames", "latentFrames",
                "generatedLatentFrames", "hardwareModel", "metalDevice",
                "repositoryHead", "sourceFiles", "toolchain",
            ],
            "fixedProductGeometry": {
                "batch": 1,
                "heads": 40,
                "headDim": 128,
                "groupSize": 64,
                "width": 832,
                "height": 480,
                "frames": 25,
                **PACKED_GEOMETRY_IDENTITY,
            },
            "eligibilityBoundary": "only exact measured arm geometries are eligible",
            "notSweptByThisSchedule": [
                "alternateHeadCounts",
                "alternateHeadDimensions",
                "alternateGroupSizes",
                "alternateTileShapes",
                "alternateWindowPolicies",
                "alternateGpuFamilies",
                "alternateDtypes",
                "alternateToolchainsOrBuilds",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_source_budget(value: object, mode: str, tier: str, model: dict[str, Any]) -> dict[str, Any]:
    budget = _object(value, "sourceBudget", {
        "modelLogicalBytes", "terminalKvBytes", "appendOldStagedAndNewDenseBytes",
        "decodeWorkingSetEstimateBytes", "runtimeOverheadBytes", "wholeProcessPeakBoundBytes",
        "decodePlan",
    })
    if budget["decodePlan"] != DECODE_PLAN_IDENTITY:
        raise CampaignError("source budget decode plan differs from frozen campaign plan")
    if budget["runtimeOverheadBytes"] is not None or budget["wholeProcessPeakBoundBytes"] is not None:
        raise CampaignError("source budget must leave unmeasured runtime overhead and whole-process peak unknown")
    logical = sum(model["files"][name]["size"] for name in
                  ("dit.safetensors", "t5_encoder.safetensors", "vae.safetensors"))
    if budget["modelLogicalBytes"] != logical:
        raise CampaignError("source budget logical model bytes differ from sealed inventory")
    per_token = 435_200 if tier == "q8" else 230_400
    schedule = (1, 3, 3) if mode == "i2v" else (3, 3, 1)
    previous = 0
    append_coexistence = 0
    for frames in schedule:
        current = frames * 1_560
        staged = previous + current
        append_coexistence = max(append_coexistence,
                                 (previous + staged) * per_token + current * 819_200)
        previous = staged
    if budget["terminalKvBytes"] != previous * per_token or budget["appendOldStagedAndNewDenseBytes"] != append_coexistence:
        raise CampaignError("source budget KV shape arithmetic differs from campaign geometry")
    output_voxels = 28 * 480 * 832
    tile_voxels = 28 * 256 * 256
    if budget["decodeWorkingSetEstimateBytes"] != 64 * output_voxels + 6_500 * tile_voxels:
        raise CampaignError("source budget decode estimate differs from frozen z16 plan")
    return budget


def _validate_phase_memory(value: object, phases: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(phases):
        raise CampaignError("phaseMemory must cover every required phase")
    previous_peak = 0
    for index, (record, name) in enumerate(zip(value, phases)):
        record = _object(record, f"phaseMemory[{index}]", {"phase", "process", "mlxActiveBytes", "mlxCacheBytes"})
        if record["phase"] != name:
            raise CampaignError("phaseMemory order or label changed")
        process = _object(record["process"], f"phaseMemory[{index}].process", {"physFootprintBytes", "physFootprintPeakBytes"})
        current = _integer(process["physFootprintBytes"], f"phaseMemory[{index}].physFootprintBytes", minimum=1)
        peak = _integer(process["physFootprintPeakBytes"], f"phaseMemory[{index}].physFootprintPeakBytes", minimum=current)
        if peak < previous_peak:
            raise CampaignError("phaseMemory process peak regressed")
        previous_peak = peak
        _integer(record["mlxActiveBytes"], f"phaseMemory[{index}].mlxActiveBytes")
        _integer(record["mlxCacheBytes"], f"phaseMemory[{index}].mlxCacheBytes")
    return value


def _route_lifecycle_snapshot(value: object, name: str) -> dict[str, Any]:
    route = _object(
        value,
        name,
        {
            "compiledHandleIdentity", "retainedHandleBytes", "acceptedForwards",
            "materializedScratchDispatches", "boundedScratchBytes", "denseWindowBytes",
            "scoreMatrixBytes", "dispatchGeometries",
        },
    )
    _nonempty(route["compiledHandleIdentity"], f"{name}.compiledHandleIdentity")
    _integer(route["retainedHandleBytes"], f"{name}.retainedHandleBytes", minimum=1)
    accepted = _integer(route["acceptedForwards"], f"{name}.acceptedForwards")
    _integer(route["materializedScratchDispatches"], f"{name}.materializedScratchDispatches")
    _integer(route["boundedScratchBytes"], f"{name}.boundedScratchBytes")
    if route["denseWindowBytes"] != 0 or route["scoreMatrixBytes"] != 0:
        raise CampaignError(f"{name} used a dense window or score matrix")
    geometries = route["dispatchGeometries"]
    if not isinstance(geometries, list):
        raise CampaignError(f"{name}.dispatchGeometries must be a list")
    coordinates: set[tuple[int, int]] = set()
    counted = 0
    for index, value in enumerate(geometries):
        geometry = _object(
            value,
            f"{name}.dispatchGeometries[{index}]",
            {"queryTokens", "keyTokens", "acceptedForwards"},
        )
        query = _integer(geometry["queryTokens"], f"{name}.dispatchGeometries[{index}].queryTokens", minimum=1)
        key = _integer(geometry["keyTokens"], f"{name}.dispatchGeometries[{index}].keyTokens", minimum=query)
        forwards = _integer(geometry["acceptedForwards"], f"{name}.dispatchGeometries[{index}].acceptedForwards", minimum=1)
        if (query, key) in coordinates:
            raise CampaignError(f"{name}.dispatchGeometries contains a duplicate coordinate")
        coordinates.add((query, key))
        counted += forwards
    if counted != accepted:
        raise CampaignError(f"{name} dispatch counters disagree with geometry evidence")
    return route


def _cancellation_dispatch_token(run_id: str, route: dict[str, Any]) -> str:
    dispatches = ";".join(
        f"{geometry['queryTokens']}:{geometry['keyTokens']}:{geometry['acceptedForwards']}"
        for geometry in route["dispatchGeometries"]
    )
    raw = (
        f"{run_id}\0{route['compiledHandleIdentity']}\0{route['acceptedForwards']}\0"
        f"{route['materializedScratchDispatches']}\0{dispatches}"
    )
    return _sha256(raw.encode("utf-8"))


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
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, encoding="utf-8", check=False
    )
    if status.returncode or status.stdout:
        raise CampaignError("campaign source must be a clean git checkout")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", check=False
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
    if snapshot.name != "q4" or snapshot.parent.name != MODEL_REVISION:
        raise CampaignError("snapshot path must be the exact pinned revision's q4 directory")
    required = ("config.json", "dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json")
    files: dict[str, dict[str, int | str]] = {}
    inventory = hashlib.sha256()
    for name in required:
        path = snapshot / name
        if not path.is_file():
            raise CampaignError(f"snapshot missing required product file: {name}")
        stat = path.stat()
        if stat.st_size <= 0:
            raise CampaignError(f"snapshot product file is empty: {name}")
        digest = _sha256_file(path)
        files[name] = {"size": stat.st_size, "sha256": digest}
        inventory.update(f"{name}\0{stat.st_size}\0{digest}\n".encode("utf-8"))
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "variant": "q4",
        "configSha256": _sha256(raw),
        "files": files,
        "inventorySha256": inventory.hexdigest(),
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
            "memory", "sourceBudget", "phaseMemory", "parity", "quality", "fallback", "cancellation", "output",
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
    _validate_source_budget(row["sourceBudget"], expected_mode, expected_tier, expected_snapshot)
    _validate_phase_memory(row["phaseMemory"], (
        "weights-loaded", "conditioning-complete", "packed-generation-complete",
        "packed-decode-complete", "dense-generation-complete", "dense-decode-complete",
        "cancellation-complete", "release",
    ))

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
        {"batch", "heads", "queryTokens", "keyTokens", "dispatchGeometries", "headDim", "groupSize", "queryTile", "keyTile", "dtypes", "mask", "width", "height", "frames", "latentFrames", "generatedLatentFrames"},
    )
    if geometry["batch"] != 1 or geometry["heads"] != 40 or geometry["headDim"] != 128 or geometry["groupSize"] != 64:
        raise CampaignError("provider observation did not use native Krea packed geometry")
    if {
        "queryTile": geometry["queryTile"],
        "keyTile": geometry["keyTile"],
        "dtypes": geometry["dtypes"],
    } != PACKED_GEOMETRY_IDENTITY:
        raise CampaignError("provider observation did not use the frozen packed tile/dtype identity")
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
    loaded_release_limit = memory["weightsLoaded"]["physFootprintBytes"] + 512 * 1024 * 1024
    terminal_release_limit = memory["verificationTerminal"]["physFootprintBytes"] + 512 * 1024 * 1024
    resources_released = not (
        mlx["releaseActiveBytes"] > mlx["weightsLoadedActiveBytes"]
        or mlx["releaseActiveBytes"] > mlx["verificationTerminalActiveBytes"]
        or mlx["releaseCacheBytes"] > mlx["weightsLoadedCacheBytes"]
        or mlx["releaseCacheBytes"] > mlx["verificationTerminalCacheBytes"]
        or memory["release"]["physFootprintBytes"] > loaded_release_limit
        or memory["release"]["physFootprintBytes"] > terminal_release_limit
    )

    parity = _object(row["parity"], "parity", {"status", "candidateTier", "maxAbsError", "tolerance"})
    if parity["status"] not in {"pass", "fail"} or parity["candidateTier"] != expected_tier:
        raise CampaignError("packed parity status is malformed or tier-substituted")
    parity_error = _number(parity["maxAbsError"], "parity.maxAbsError")
    parity_tolerance = _number(parity["tolerance"], "parity.tolerance", positive=True)
    frozen_parity_tolerance = PARITY_MAX_ABS_ERROR_BY_TIER[expected_tier]
    if parity_tolerance != frozen_parity_tolerance:
        raise CampaignError("packed parity tolerance differs from the frozen decision policy")
    if (parity["status"] == "pass") != (parity_error <= frozen_parity_tolerance):
        raise CampaignError("packed parity status contradicts its measured values")

    quality = _object(
        row["quality"],
        "quality",
        {"status", "candidateTier", "metric", "maxAbsRgbU8", "maxAbsRgbU8Tolerance", "meanAbsRgbU8", "meanAbsRgbU8Tolerance", "packedMeanTemporalDelta", "denseMeanTemporalDelta", "temporalDeltaDrift", "temporalDeltaDriftTolerance", "acknowledged"},
    )
    if quality["status"] not in {"pass", "fail"} or quality["candidateTier"] != expected_tier:
        raise CampaignError("packed quality status is malformed or tier-substituted")
    if quality["metric"] != decision_policy()["accuracy"]["metric"]:
        raise CampaignError("packed quality metric differs from the frozen decision policy")
    for key in ("maxAbsRgbU8", "maxAbsRgbU8Tolerance", "meanAbsRgbU8", "meanAbsRgbU8Tolerance", "packedMeanTemporalDelta", "denseMeanTemporalDelta", "temporalDeltaDrift", "temporalDeltaDriftTolerance"):
        _number(quality[key], f"quality.{key}", positive=key in {"maxAbsRgbU8Tolerance", "meanAbsRgbU8Tolerance", "temporalDeltaDriftTolerance"})
    frozen_quality = QUALITY_THRESHOLDS_BY_TIER[expected_tier]
    reported_quality = {
        "maxAbsRgbU8": quality["maxAbsRgbU8Tolerance"],
        "meanAbsRgbU8": quality["meanAbsRgbU8Tolerance"],
        "temporalDeltaDrift": quality["temporalDeltaDriftTolerance"],
    }
    if reported_quality != frozen_quality:
        raise CampaignError("packed quality tolerances differ from the frozen decision policy")
    quality_pass = not (
        quality["maxAbsRgbU8"] > frozen_quality["maxAbsRgbU8"]
        or quality["meanAbsRgbU8"] > frozen_quality["meanAbsRgbU8"]
        or quality["temporalDeltaDrift"] > frozen_quality["temporalDeltaDrift"]
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
    cancellation = _object(
        row["cancellation"],
        "cancellation",
        {
            "status", "requests", "trigger", "progressStepsBeforeCancel",
            "typedCancellationObserved", "dispatchToken", "routeBefore", "routeAtCancel",
            "expectedContextTokens", "storedTokensAfterCancel",
            "activeBytesBefore", "activeBytesAfterRelease",
            "cacheBytesBefore", "cacheBytesAfterRelease",
            "allocationObserved", "partialStateMutation", "scratchReleased",
        },
    )
    if (
        cancellation["status"] not in {"pass", "fail"}
        or type(cancellation["partialStateMutation"]) is not bool
        or type(cancellation["scratchReleased"]) is not bool
        or type(cancellation["allocationObserved"]) is not bool
        or type(cancellation["typedCancellationObserved"]) is not bool
    ):
        raise CampaignError("cancellation evidence is malformed")
    if cancellation["trigger"] != "after-first-materialized-denoise-step":
        raise CampaignError("cancellation probe was not triggered in-flight after allocation")
    if _integer(cancellation["requests"], "cancellation.requests", minimum=1) != 1:
        raise CampaignError("cancellation probe must issue exactly one request")
    if _integer(cancellation["progressStepsBeforeCancel"], "cancellation.progressStepsBeforeCancel", minimum=1) != 1:
        raise CampaignError("cancellation probe must cancel after its first materialized step")
    route_before = _route_lifecycle_snapshot(cancellation["routeBefore"], "cancellation.routeBefore")
    route_at_cancel = _route_lifecycle_snapshot(cancellation["routeAtCancel"], "cancellation.routeAtCancel")
    if (
        route_before["compiledHandleIdentity"] != route_at_cancel["compiledHandleIdentity"]
        or route_before["retainedHandleBytes"] != route_at_cancel["retainedHandleBytes"]
    ):
        raise CampaignError("cancellation route lifecycle does not identify one retained handle")
    dispatch_delta = route_at_cancel["acceptedForwards"] - route_before["acceptedForwards"]
    scratch_dispatch_delta = (
        route_at_cancel["materializedScratchDispatches"]
        - route_before["materializedScratchDispatches"]
    )
    expected_scratch_bytes = (
        3 * geometry["queryTile"] * geometry["headDim"]
        + geometry["queryTile"] * geometry["keyTile"]
        + 3 * geometry["queryTile"]
    ) * 4
    route_allocation_observed = (
        route_before["acceptedForwards"] == 0
        and route_before["materializedScratchDispatches"] == 0
        and route_before["boundedScratchBytes"] == 0
        and route_before["dispatchGeometries"] == []
        and dispatch_delta == 1
        and scratch_dispatch_delta == dispatch_delta
        and route_at_cancel["boundedScratchBytes"] == expected_scratch_bytes
    )
    dispatch_token = cancellation["dispatchToken"]
    if not _is_sha256(dispatch_token) or dispatch_token != _cancellation_dispatch_token(run_id, route_at_cancel):
        raise CampaignError("cancellation dispatch token does not bind the launched run and route receipt")
    expected_context_tokens = _integer(
        cancellation["expectedContextTokens"], "cancellation.expectedContextTokens", minimum=1
    )
    stored_tokens_after = _integer(
        cancellation["storedTokensAfterCancel"], "cancellation.storedTokensAfterCancel"
    )
    for key in (
        "activeBytesBefore", "activeBytesAfterRelease",
        "cacheBytesBefore", "cacheBytesAfterRelease",
    ):
        _integer(cancellation[key], f"cancellation.{key}")
    scratch_released = (
        cancellation["activeBytesAfterRelease"] <= cancellation["activeBytesBefore"]
        and cancellation["cacheBytesAfterRelease"] <= cancellation["cacheBytesBefore"]
    )
    if cancellation["allocationObserved"] != route_allocation_observed:
        raise CampaignError("cancellation allocation claim contradicts the packed-route receipt delta")
    if cancellation["scratchReleased"] != scratch_released:
        raise CampaignError("cancellation cleanup claim contradicts raw allocator evidence")
    partial_state_mutation = stored_tokens_after != expected_context_tokens
    if cancellation["partialStateMutation"] != partial_state_mutation:
        raise CampaignError("cancellation mutation claim contradicts raw cache-token evidence")
    cancellation_clean = (
        cancellation["typedCancellationObserved"] is True
        and cancellation["allocationObserved"] is True
        and cancellation["partialStateMutation"] is False
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
            "sourceBudget", "phaseMemory",
        },
    )
    if row["schemaVersion"] != SCHEMA_VERSION or row["producer"] != "mlx-gen-krea-realtime/sc20684-dense-baseline":
        raise CampaignError("dense baseline schema or producer mismatch")
    if row["runId"] != run_id or row["case"] != {"mode": expected_mode, "cacheTier": expected_tier}:
        raise CampaignError("dense baseline did not report its launched identity")
    for key in ("source", "model", "input", "schedule", "toolchain", "sourceBudget"):
        if row[key] != candidate[key]:
            raise CampaignError(f"dense baseline {key} differs from its paired candidate")
    _validate_source_budget(row["sourceBudget"], expected_mode, expected_tier, candidate["model"])
    _validate_phase_memory(row["phaseMemory"], (
        "weights-loaded", "conditioning-complete", "dense-generation-complete",
        "dense-decode-complete", "release",
    ))

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
    loaded_release_limit = memory["weightsLoaded"]["physFootprintBytes"] + 512 * 1024 * 1024
    terminal_release_limit = memory["generationTerminal"]["physFootprintBytes"] + 512 * 1024 * 1024
    resources_released = not (
        mlx["releaseActiveBytes"] > mlx["weightsLoadedActiveBytes"]
        or mlx["releaseActiveBytes"] > mlx["generationTerminalActiveBytes"]
        or mlx["releaseCacheBytes"] > mlx["weightsLoadedCacheBytes"]
        or mlx["releaseCacheBytes"] > mlx["generationTerminalCacheBytes"]
        or memory["release"]["physFootprintBytes"] > loaded_release_limit
        or memory["release"]["physFootprintBytes"] > terminal_release_limit
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


def _prepare_media_resume(
    root: Path, *, source: dict[str, Any], model: dict[str, Any],
    policy: supervisor.SafetyPolicy, argv: list[str], timeout: int,
) -> dict[str, Any]:
    if not root.is_absolute() or root.is_symlink() or root.resolve() == ROOT.resolve() or ROOT.resolve() in root.resolve().parents:
        raise CampaignError("resume directory must be absolute, nonsymlink, and outside the repository")
    executable = Path(argv[0]).resolve()
    if not executable.is_file() or executable.name == "cargo":
        raise CampaignError("product command must name a prebuilt sealed observer executable, not cargo")
    identity = {
        "schemaVersion": 1, "kind": "sc20684-media-resume",
        "source": source, "model": model, "policySha256": policy.sha256,
        "decisionPolicySha256": _canonical_sha256(decision_policy()),
        "schedule": [list(item) for item in CASES],
        "executableSha256": _sha256_file(executable), "argv": argv,
        "timeoutSeconds": timeout,
    }
    identity_path = root / "identity.json"
    if root.exists():
        try:
            prior = json.loads(identity_path.read_bytes())
            identity["source"]["repositoryHead"] = prior["source"]["repositoryHead"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CampaignError(f"resume identity is malformed: {error}") from error
    else:
        root.mkdir(parents=False)
    encoded = supervisor.canonical(identity)
    sidecar = f"{_sha256(encoded)}  identity.json\n"
    if identity_path.exists():
        if identity_path.read_bytes() != encoded or (root / "identity.json.sha256").read_text(encoding="utf-8") != sidecar:
            raise CampaignError("resume identity changed executable, model, policy, behavior source, or schedule")
    else:
        identity_path.write_bytes(encoded)
        (root / "identity.json.sha256").write_text(sidecar, encoding="utf-8")
    for entry in root.iterdir():
        if entry.is_symlink():
            raise CampaignError(f"symlinked resume artifact: {entry.name}")
        if entry.name not in {"identity.json", "identity.json.sha256", "roles", "artifacts", "transcripts", "logs"}:
            raise CampaignError(f"unexpected resume artifact: {entry.name}")
    return identity


def _role_file_bindings(root: Path, name: str, transcripts: dict[str, Any]) -> list[dict[str, Any]]:
    files = [transcripts["stdout"], transcripts["stderr"]]
    cell, role = name.rsplit(".", 1)
    if role == "paired":
        artifact_dir = root / "artifacts" / cell
        if not artifact_dir.is_dir() or artifact_dir.is_symlink():
            raise CampaignError(f"paired role {name} lacks product artifacts")
        for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
            if path.is_symlink():
                raise CampaignError("resume artifact may not be a symlink")
            files.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path), "bytes": path.stat().st_size,
            })
    return files


def _supervision_record(result: supervisor.RunResult) -> dict[str, Any]:
    return {
        "pid": result.pid, "peakHostBytes": result.peak_host_bytes,
        "peakGpuBytes": result.peak_gpu_bytes,
        "hostFreeAtLaunch": result.host_free_at_launch,
        "elapsedSeconds": result.elapsed_seconds,
        "ownedProcessGroupReaped": True,
        "admission": result.admission,
    }


def _save_resumed_role(root: Path, name: str, identity: dict[str, Any], record: dict[str, Any]) -> None:
    roles = root / "roles"
    roles.mkdir(exist_ok=True)
    if roles.is_symlink():
        raise CampaignError("resume role directory may not be symlinked")
    path = roles / f"{name}.json"
    if path.exists():
        raise CampaignError(f"role {name} already has a resume binding")
    record = dict(record)
    record.update({
        "resumeIdentitySha256": _sha256(supervisor.canonical(identity)),
        "files": _role_file_bindings(root, name, record["transcripts"]),
    })
    encoded = supervisor.canonical(record)
    partial = roles / f".{name}.partial"
    with partial.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    (roles / f"{name}.json.sha256").write_text(f"{_sha256(encoded)}  {name}.json\n", encoding="utf-8")


def _load_resumed_role(root: Path, name: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    path = root / "roles" / f"{name}.json"
    sidecar = root / "roles" / f"{name}.json.sha256"
    if not path.exists() and not sidecar.exists():
        return None
    if not path.is_file() or not sidecar.is_file() or path.is_symlink() or sidecar.is_symlink():
        raise CampaignError(f"partial resume binding for {name}")
    raw = path.read_bytes()
    if sidecar.read_text(encoding="utf-8") != f"{_sha256(raw)}  {name}.json\n":
        raise CampaignError(f"resume role {name} sidecar is corrupt")
    record = json.loads(raw)
    if raw != supervisor.canonical(record) or record.get("resumeIdentitySha256") != _sha256(supervisor.canonical(identity)):
        raise CampaignError(f"resume role {name} is stale or noncanonical")
    if record.get("exitCode") != 0 or not isinstance(record.get("runId"), str):
        raise CampaignError(f"resume role {name} did not exit successfully")
    supervision = record.get("supervision")
    if not isinstance(supervision, dict) or type(supervision.get("pid")) is not int or supervision["pid"] <= 0 or supervision.get("ownedProcessGroupReaped") is not True:
        raise CampaignError(f"resume role {name} lacks an owned process identity")
    try:
        supervisor.validate_admission(supervision.get("admission"), policy_sha256=identity.get("policySha256"))
    except supervisor.SupervisionError as error:
        raise CampaignError(f"resume role {name} lacks its runtime-guarded admission") from error
    if record.get("files") != _role_file_bindings(root, name, record["transcripts"]):
        raise CampaignError(f"resume role {name} artifact bytes changed")
    for file in record["files"]:
        _validate_evidence_file(file, root, f"resume {name} file")
    return record


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
        "build": {
            "repositoryHead": observation["source"]["repositoryHead"],
            "sourceFiles": observation["source"]["files"],
            "toolchain": toolchain,
        },
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
    safety_policy: supervisor.SafetyPolicy,
    resume_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        argv = shlex.split(command)
    except ValueError as error:
        raise CampaignError(f"invalid --product-command: {error}") from error
    if not argv:
        raise CampaignError("--product-command must not be empty")
    if safety_policy.deadline_seconds > timeout:
        raise CampaignError("safety deadline must not exceed the requested per-role timeout")
    rows: list[dict[str, Any]] = []
    used_run_ids: set[str] = set()
    used_pids: set[int] = set()
    for mode, tier in CASES:
        cell_name = f"{mode}-{tier}"
        artifact_dir = evidence_root / "artifacts" / cell_name
        transcript_dir = evidence_root / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        cell_started = time.monotonic_ns()
        observations: dict[str, Any] = {}
        transcripts: dict[str, dict[str, Any]] = {}
        for role in ("paired", "dense-baseline"):
            role_name = f"{cell_name}.{role}"
            saved = _load_resumed_role(evidence_root, role_name, resume_identity)
            if saved is not None:
                run_id = saved["runId"]
                pid = saved["supervision"]["pid"]
                if run_id in used_run_ids or pid in used_pids:
                    raise CampaignError(f"resume role {role_name} reuses a process identity")
                used_run_ids.add(run_id)
                used_pids.add(pid)
                stdout_path = evidence_root / saved["transcripts"]["stdout"]["path"]
                stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
                transcripts[role] = saved["transcripts"]
                observations[f"{role}-process-exit-code"] = saved["exitCode"]
                if role == "paired":
                    observations[role] = _validate_observation(
                        _read_observation(stdout), expected_mode=mode, expected_tier=tier,
                        run_id=run_id, expected_source=source, expected_snapshot=model,
                    )
                    _validate_artifact_directory(observations[role], artifact_dir)
                else:
                    observations[role] = _validate_baseline_observation(
                        _read_observation(stdout, BASELINE_PREFIX),
                        expected_mode=mode, expected_tier=tier, run_id=run_id,
                        candidate=observations["paired"],
                    )
                if saved["observationSha256"] != _canonical_sha256(observations[role]):
                    raise CampaignError(f"resume role {role_name} observation identity changed")
                continue
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
                "KREA_SC20684_IDENTITY_PATH": str(evidence_root / "identity.json"),
            })
            if tier == "q4":
                # The Q4 arm is a separately named experiment.  This is a selector only; measured
                # quality still has to arrive from the provider-owned observer and pass reduction.
                env["KREA_SC20684_Q4_QUALITY_ARM"] = "acknowledged"
            stdout_path = transcript_dir / f"{cell_name}.{role}.stdout.log"
            stderr_path = transcript_dir / f"{cell_name}.{role}.stderr.log"
            if stdout_path.exists() or stderr_path.exists():
                raise CampaignError(f"partial {role_name} transcript exists without valid resume binding")
            logs = evidence_root / "logs"
            # A pre-spawn refusal leaves only an unaccepted record, so every attempt index counts.
            indices = [int(index) for path in logs.glob(f"{role_name}.*")
                       if (index := path.name[len(role_name) + 1:].split(".", 1)[0]).isdecimal()]
            attempt = max(indices) + 1 if indices else 0
            unaccepted = logs / f"{role_name}.{attempt}.unaccepted.json"
            try:
                result = supervisor.run_guarded(
                    argv, cwd=ROOT, env=env, policy=safety_policy,
                    stdout_path=logs / f"{role_name}.{attempt}.stdout.log",
                    stderr_path=logs / f"{role_name}.{attempt}.stderr.log",
                )
            except supervisor.SupervisionError as error:
                supervisor.write_unaccepted_record(
                    unaccepted, kind="sc-20684-unaccepted-role", coordinate=role_name, error=error,
                )
                raise
            try:
                if run_id in used_run_ids or result.pid in used_pids:
                    raise CampaignError(f"role {role_name} reuses a process identity")
                used_run_ids.add(run_id)
                used_pids.add(result.pid)
                shutil.copyfile(logs / f"{role_name}.{attempt}.stdout.log", stdout_path)
                shutil.copyfile(logs / f"{role_name}.{attempt}.stderr.log", stderr_path)
                transcripts[role] = {
                    "stdout": _file_identity(stdout_path, f"transcripts/{stdout_path.name}"),
                    "stderr": _file_identity(stderr_path, f"transcripts/{stderr_path.name}"),
                }
                observations[f"{role}-process-exit-code"] = result.returncode
                stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
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
            except CampaignError as error:
                # The child exited cleanly but left invalid evidence: a failed, unaccepted role.
                failure = supervisor.SupervisionError("invalid-evidence", str(error))
                failure.pid, failure.admission = result.pid, result.admission
                supervisor.write_unaccepted_record(
                    unaccepted, kind="sc-20684-unaccepted-role", coordinate=role_name, error=failure,
                )
                raise
            _save_resumed_role(evidence_root, role_name, resume_identity, {
                "runId": run_id, "exitCode": result.returncode,
                "transcripts": transcripts[role],
                "observationSha256": _canonical_sha256(observations[role]),
                "supervision": _supervision_record(result),
            })
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
    safety_policy: supervisor.SafetyPolicy | None = None,
    resume_identity: dict[str, Any] | None = None,
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
        if safety_policy is not None and resume_identity is not None:
            shutil.copytree(evidence_root / "roles", temporary / "roles")
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (temporary / "receipt.json").write_bytes(encoded)
        (temporary / "receipt.json.sha256").write_text(f"{_sha256(encoded)}  receipt.json\n", encoding="utf-8")
        if safety_policy is not None and resume_identity is not None:
            (temporary / "safety-policy.json").write_bytes(safety_policy.canonical_bytes)
            (temporary / "resume-identity.json").write_bytes(supervisor.canonical(resume_identity))
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
    parser.add_argument("--safety-policy", type=Path, required=True, help="mandatory source-bound process safety policy")
    parser.add_argument("--resume-dir", type=Path, required=True, help="absolute external identity-bound role staging")
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise CampaignError("--timeout must be positive")
        output = _checked_output(args.output)
        policy = decision_policy()
        source = source_identity()
        model = snapshot_identity(args.snapshot)
        safety_policy = supervisor.load_policy(args.safety_policy)
        if safety_policy.backend != "darwin-mlx":
            raise CampaignError("Krea SC-20684 requires the Darwin/MLX safety backend")
        if not args.resume_dir.is_absolute() or args.resume_dir.resolve() == output:
            raise CampaignError("resume directory must be absolute and distinct from output")
        try:
            command = shlex.split(args.product_command)
        except ValueError as error:
            raise CampaignError(f"invalid product command: {error}") from error
        resume_identity = _prepare_media_resume(
            args.resume_dir, source=source, model=model, policy=safety_policy,
            argv=command, timeout=args.timeout,
        )
        source = resume_identity["source"]
        rows = run_matrix(
            args.product_command, args.snapshot.resolve(), source, model, policy,
            args.timeout, args.resume_dir, safety_policy, resume_identity,
        )
        if source_identity()["files"] != source["files"]:
            raise CampaignError("campaign behavior source changed after identity was frozen")
        if snapshot_identity(args.snapshot) != model:
            raise CampaignError("model snapshot changed during the campaign")
        publish(
            output, source=source, model=model, rows=rows, evidence_root=args.resume_dir,
            policy=policy, safety_policy=safety_policy, resume_identity=resume_identity,
        )
    except (CampaignError, supervisor.SupervisionError) as error:
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

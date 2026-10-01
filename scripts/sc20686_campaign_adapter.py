#!/usr/bin/env python3
"""Product-entrypoint campaign producer for sealed SC-20686 evidence bundles.

Safe operator stop: ``touch <resume-dir>/STOP`` (always honoured) or the ``--stop-file`` path. The adapter never
signals a running entrypoint; before starting the next arm it writes a sealed
``<resume-dir>/logs/operator-stop.attempt-<n>.json`` ("stopped-by-operator", ``beforeRow``) and
exits with status 75. Remove the stop file and rerun the same command to resume at that arm.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import media_campaign_supervisor as supervisor
REDUCER = Path(__file__).with_name("sc20686_cache_attribution.py")
COVERAGE = Path(__file__).with_name("sc20686_coverage_manifest.json")
SOURCE_MAP = Path(__file__).with_name("sc20686_source_map.json")
PRODUCER = "sc20686-campaign-adapter-v5"
INFERENCE_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = (
    "batch", "resolution", "reference_count", "frames", "prompt", "guidance", "layers",
    "heads", "head_dimension", "sq", "skv", "dtype", "mask", "rope",
)
WAN_ROUTES = (
    "wan2_2_ti2v_5b", "wan2_2_t2v_14b", "wan2_2_i2v_14b", "wan_vace",
    "wan2_2_vace_fun_14b",
)
FLUX_ROUTES = ("flux2_klein_9b_edit", "flux2_klein_9b_kv_edit")
# Measurement lanes: the CUDA lane measures the Candle providers; the Metal lane measures the MLX
# providers the SceneWorks Mac product runs. The safety-policy backend selects exactly one lane.
LANES = ("candle-cuda", "mlx-metal")
LANE_BY_POLICY = {
    "darwin-mlx": "mlx-metal",
    "linux-cuda": "candle-cuda",
    "windows-cuda": "candle-cuda",
}
WAN_ENTRYPOINT_STEMS = {
    "wan2_2_ti2v_5b": "wan-txt2video",
    "wan2_2_t2v_14b": "wan14b-txt2video",
    "wan2_2_i2v_14b": "wan14b-img2video",
    "wan_vace": "vace_smoke",
    "wan2_2_vace_fun_14b": "vace_smoke",
}
FILE_ARGUMENT_FLAGS = {
    "--image", "--reference", "--reference2", "--control-dir", "--mask-dir",
    "--lora-high", "--lora-low", "--comfyui-high", "--comfyui-low",
    "--comfyui-te", "--comfyui-vae", "--adapter", "--adapter-weights",
    "--ip-adapter", "--ip-adapter-weights",
}
PROTECTED_FLAGS = {
    "--sc20686-campaign", "--sc20686-events", "--snapshot", "--variant",
    "--sc20686-route", "--sc20686-cancel", "--sc20686-source-ref",
    "--sc20686-residency", "--out",
}
PRODUCT_RESIDENCY = {
    "flux2_klein_9b_edit": "sequential",
    "flux2_klein_9b_kv_edit": "sequential",
    "wan2_2_ti2v_5b": "sequential",
    "wan2_2_t2v_14b": "sequential",
    "wan2_2_i2v_14b": "sequential",
    "wan_vace": "resident",
    "wan2_2_vace_fun_14b": "sequential",
}


@dataclass(frozen=True)
class CampaignRun:
    events: list
    stdout: bytes
    stderr: bytes
    command: tuple
    process_samples: tuple
    event_transcript: bytes = b""
    media_output: Optional[Path] = None
    cleanup_root: Optional[Path] = None
    supervision: Optional[dict] = None


@dataclass(frozen=True)
class CoordinateSpec:
    family: str
    variant: str
    name: str
    entrypoint: Path
    snapshot: Path
    model_snapshot_revision: str
    residency_strategy: str
    args: tuple
    route_manifest_sha256: str
    input_file_sha256: dict
    input_file_inventory: tuple


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def file_identity(path):
    identity = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            identity.update(chunk)
    return identity.hexdigest()


def snapshot_identity(root):
    # Follow symlinked directories: the Mac worker assembles its Wan-VACE snapshots by linking the
    # transformer directories into place, and an identity that skipped them would omit the weights.
    # Following links admits cycles and dangling links, so both are refused rather than skipped.
    root = Path(root).resolve()

    def walk_error(error):
        raise ValueError(f"snapshot inventory walk failed: {error}") from error

    files = []
    visited = set()
    for directory, dirnames, names in os.walk(root, followlinks=True, onerror=walk_error):
        status = os.stat(directory)
        if (status.st_dev, status.st_ino) in visited:
            raise ValueError(f"snapshot inventory revisits a directory (link cycle): {directory}")
        visited.add((status.st_dev, status.st_ino))
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in names:
            path = Path(directory) / name
            if not path.exists():
                raise ValueError(f"snapshot inventory has a broken link: {path}")
            if path.is_file():
                files.append(path)
    files.sort()
    if not files:
        raise ValueError("snapshot inventory is empty")
    aggregate = hashlib.sha256()
    total = 0
    for path in files:
        file_hash = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                total += len(chunk)
                file_hash.update(chunk)
        aggregate.update(str(path.relative_to(root)).encode("utf-8"))
        aggregate.update(b"\0" + struct.pack("<Q", size) + b"\0" + file_hash.digest() + b"\n")
    return aggregate.hexdigest(), total


def require_revision(value, label):
    value = str(value)
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be lowercase 40-hex")
    return value


def model_snapshot_revision(root):
    """Resolve the HF revision without confusing a nested tier name with source provenance."""
    root = Path(root).resolve()
    for depth, candidate in enumerate((root, *root.parents)):
        if depth > 2:
            break
        if re.fullmatch(r"[0-9a-f]{40}", candidate.name):
            return candidate.name
        marker = candidate / ".snapshot-revision"
        if marker.is_file():
            return require_revision(marker.read_text(encoding="utf-8").strip(), "model snapshot revision")
    raise ValueError("model snapshot has no immutable revision within its tier/root closure")


def verify_inference_revision(expected):
    expected = require_revision(expected, "inference repository revision")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(INFERENCE_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("inference repository revision is unavailable") from exc
    if actual != expected:
        raise ValueError(
            f"inference repository revision mismatch: requested {expected}, checkout {actual}"
        )
    return actual


# The Mac worker's assembled Wan-VACE snapshots (`wan_vace_dir_is_complete` /
# `wan_vace_fun_dir_is_complete`): diffusers VACE transformer(s) beside a base-Wan tier's UMT5, VAE
# and tokenizer, with no root config or model index. Only these routes may take that layout.
WAN_VACE_ASSEMBLED_FILES = {
    "wan_vace": (
        "transformer/config.json", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json",
    ),
    "wan2_2_vace_fun_14b": (
        "transformer/config.json", "transformer_2/config.json", "t5_encoder.safetensors",
        "vae.safetensors", "tokenizer.json",
    ),
}
# The Mac product's default packed tier (`q4/`) and where each Metal tiered route declares its bits.
PRODUCT_TIER_BITS = 4
METAL_PACKED_TIER_CONFIG = {
    "wan2_2_ti2v_5b": "config.json",
    "wan2_2_t2v_14b": "config.json",
    "wan2_2_i2v_14b": "config.json",
    "flux2_klein_9b_edit": "transformer/config.json",
    "flux2_klein_9b_kv_edit": "transformer/config.json",
}


def packed_tier_bits(root, relative):
    try:
        config = json.loads((Path(root) / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"packed tier config is unreadable: {relative}") from exc
    quantization = config.get("quantization") if isinstance(config, dict) else None
    return quantization.get("bits") if isinstance(quantization, dict) else None


def validate_snapshot_layout(root, label, route=None, backend=None):
    """Accept an exact component/tier root or a real Diffusers pipeline root. On the Metal lane a
    tiered route must be the product's default q4 tier, and a VACE route must be the worker's
    assembled snapshot for that route."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"{label} snapshot directory is missing")
    if backend == "mlx-metal" and route in METAL_PACKED_TIER_CONFIG:
        bits = packed_tier_bits(root, METAL_PACKED_TIER_CONFIG[route])
        if bits != PRODUCT_TIER_BITS:
            raise ValueError(
                f"{label} snapshot is not the product's default q{PRODUCT_TIER_BITS} tier "
                f"(packed bits: {bits})"
            )
    if backend == "mlx-metal" and route in WAN_VACE_ASSEMBLED_FILES:
        if all((root / name).is_file() for name in WAN_VACE_ASSEMBLED_FILES[route]):
            return
        raise ValueError(f"{label} snapshot must be the worker's assembled {route} snapshot")
    if (root / "config.json").is_file():
        return
    component_configs = sorted(
        path for path in root.glob("*/config.json") if path.is_file()
    )
    if (root / "model_index.json").is_file() and component_configs:
        return
    raise ValueError(
        f"{label} snapshot must be an exact component/tier root with config.json or a "
        "Diffusers root with model_index.json plus component configs"
    )


def path_identity(path):
    path = Path(path).resolve()
    if path.is_file():
        return file_identity(path), path.stat().st_size, "file"
    if path.is_dir():
        identity, size = snapshot_identity(path)
        return identity, size, "directory"
    raise ValueError(f"file-bearing route argument is missing: {path}")


def hash_file_arguments(arguments):
    arguments = tuple(map(str, arguments))
    hashes = {}
    inventory = []
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag in FILE_ARGUMENT_FLAGS:
            if index + 1 >= len(arguments):
                raise ValueError(f"file-bearing route flag lacks a value: {flag}")
            value = Path(os.path.expandvars(arguments[index + 1])).expanduser().resolve()
            if "$" in str(value):
                raise ValueError(f"file-bearing route argument has unresolved variables: {flag}")
            item_hash, size, kind = path_identity(value)
            key = f"{flag[2:]}-{index:02d}"
            hashes[key] = item_hash
            inventory.append({
                "flag": flag, "argument_index": index + 1, "path": str(value),
                "sha256": item_hash, "bytes": size, "kind": kind,
            })
            index += 2
            continue
        index += 1
    return hashes, tuple(inventory)


def normalize_file_arguments(arguments):
    normalized = list(map(str, arguments))
    index = 0
    while index < len(normalized):
        if normalized[index] in FILE_ARGUMENT_FLAGS:
            if index + 1 >= len(normalized):
                raise ValueError(f"file-bearing route flag lacks a value: {normalized[index]}")
            normalized[index + 1] = str(
                Path(os.path.expandvars(normalized[index + 1])).expanduser().resolve()
            )
            index += 2
        else:
            index += 1
    return tuple(normalized)


def route_manifest_document(
    route, coordinate, entrypoint, entrypoint_sha256, snapshot,
    model_revision, residency_strategy, arguments, inventory,
):
    return {
        "route": route,
        "coordinate": coordinate,
        "entrypoint": str(Path(entrypoint).resolve()),
        "entrypoint_sha256": entrypoint_sha256,
        "snapshot": str(Path(snapshot).resolve()),
        "model_snapshot_revision": model_revision,
        "residency_strategy": residency_strategy,
        "args": list(map(str, arguments)),
        "input_files": list(inventory),
    }


def route_manifest_identity(
    route, coordinate, entrypoint, snapshot, model_revision,
    residency_strategy, arguments, inventory,
):
    return digest(canonical(route_manifest_document(
        route, coordinate, entrypoint, file_identity(entrypoint), snapshot,
        model_revision, residency_strategy, arguments, inventory,
    )))


def verify_coordinate_inputs(spec):
    for item in spec.input_file_inventory:
        item_hash, size, kind = path_identity(item["path"])
        if (item_hash, size, kind) != (item["sha256"], item["bytes"], item["kind"]):
            raise ValueError(
                f"route input changed after resolution: {spec.variant}/{spec.name}/{item['flag']}"
            )
    current_manifest = route_manifest_identity(
        spec.variant, spec.name, spec.entrypoint, spec.snapshot,
        spec.model_snapshot_revision, spec.residency_strategy, spec.args,
        spec.input_file_inventory,
    )
    if current_manifest != spec.route_manifest_sha256:
        raise ValueError(f"route executable or manifest changed after resolution: {spec.variant}/{spec.name}")


def verify_snapshot_identity(spec, expected):
    if snapshot_identity(spec.snapshot) != expected:
        raise ValueError(f"model snapshot changed during campaign arm: {spec.variant}/{spec.name}")


def active_allocator_measurement(event, label):
    values = {
        key: event.get(key)
        for key in (
            "allocator_before_bytes", "allocator_after_bytes", "allocator_high_bytes",
            "allocator_reserved_bytes",
        )
    }
    reserved_high = event.get("peak_bytes")
    if event.get("allocator_measurement_available") is not True or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (*values.values(), reserved_high)
    ):
        raise ValueError(f"{label} lacks active allocator high-water evidence")
    if (
        values["allocator_high_bytes"] < values["allocator_before_bytes"]
        or values["allocator_high_bytes"] < values["allocator_after_bytes"]
        or values["allocator_reserved_bytes"] < values["allocator_after_bytes"]
        or reserved_high < values["allocator_reserved_bytes"]
        or reserved_high < values["allocator_high_bytes"]
    ):
        raise ValueError(f"{label} allocator high-water/remnant ordering is invalid")
    values["reserved_high_bytes"] = reserved_high
    return values


def dense_reference_kv_bytes(geometry, kv_batch=None):
    dtype_bytes = {"F16": 2, "BF16": 2, "F32": 4}.get(geometry["dtype"].upper())
    if dtype_bytes is None:
        raise ValueError("FLUX reference slice has unsupported live dtype")
    batch = geometry["batch"] if kv_batch is None else kv_batch
    return (
        2 * batch * geometry["heads"] * geometry["skv"]
        * geometry["head_dimension"] * dtype_bytes
    )


def load_coverage():
    document = json.loads(COVERAGE.read_text(encoding="utf-8"))
    if document.get("schema") != "sc-20686-supported-coverage-v3":
        raise ValueError("checked-in coverage manifest is invalid")
    return document


def load_source_map():
    document = json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
    if document.get("schema") != "sc-20686-source-map-v3":
        raise ValueError("checked-in source map is invalid")
    return document


def lane_coverage(backend):
    """Frozen coordinates of one lane: the shared six routes plus that lane's extensions."""
    if backend not in LANES:
        raise ValueError(f"unknown measurement lane: {backend}")
    return _load_reducer().lane_coverage(load_coverage(), backend)


def lane_entry(backend, variant):
    if backend not in LANES:
        raise ValueError(f"unknown measurement lane: {backend}")
    return _load_reducer().lane_entry(load_source_map(), backend, variant)


def expand_argument(value):
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"campaign manifest has unresolved environment variables: {value}")
    return expanded


def argument_value(arguments, flag):
    matches = [arguments[index + 1] for index, item in enumerate(arguments[:-1]) if item == flag]
    if len(matches) != 1:
        raise ValueError(f"coordinate must provide exactly one {flag}")
    return matches[0]


def validate_coordinate_arguments(route, name, arguments, expected):
    try:
        width = int(argument_value(arguments, "--width"))
        height = int(argument_value(arguments, "--height"))
        frames = int(argument_value(arguments, "--frames"))
        guidance = float(argument_value(arguments, "--guidance"))
        prompt = argument_value(arguments, "--prompt")
    except ValueError as exc:
        raise ValueError(f"coordinate axes are malformed: {route}/{name}: {exc}") from exc
    reference_count = arguments.count("--reference")
    if route == "wan2_2_i2v_14b":
        if arguments.count("--image") != 1:
            raise ValueError(f"I2V coordinate lacks its exact image input: {route}/{name}")
        reference_count = 1
    if route in ("wan_vace", "wan2_2_vace_fun_14b"):
        if arguments.count("--control-dir") != 1 or arguments.count("--mask-dir") != 1:
            raise ValueError(f"VACE coordinate lacks control/mask inputs: {route}/{name}")
    actual = {
        "batch": 1,
        "resolution": f"{width}x{height}",
        "frames": frames,
        "reference_count": reference_count,
        "prompt": digest(prompt.encode("utf-8")),
        "guidance": guidance,
    }
    for field, value in expected.items():
        if field == "guidance":
            if actual[field] != float(value):
                raise ValueError(f"coordinate guidance differs from frozen coverage: {route}/{name}")
        elif actual[field] != value:
            raise ValueError(f"coordinate {field} differs from frozen coverage: {route}/{name}")


# The Mac product's A14B routes default to the Lightning distill (`advanced.lightning` unset => on):
# every Metal A14B coordinate states its toggle, and the `-lightning` coordinates run the product
# default with the per-architecture LoRA pair at the forced 4-step, guidance-1 recipe.
LIGHTNING_ROUTES = ("wan2_2_t2v_14b", "wan2_2_i2v_14b")


def validate_metal_lightning(route, name, arguments):
    lightning_flags = ("--lightning", "--lightning-hub", "--lora-high", "--lora-low")
    if route not in LIGHTNING_ROUTES:
        if any(flag in arguments for flag in lightning_flags):
            raise ValueError(f"Lightning is only a product toggle on the A14B routes: {route}/{name}")
        return
    try:
        toggle = argument_value(arguments, "--lightning")
    except ValueError as exc:
        raise ValueError(f"A14B coordinate must state --lightning on|off: {route}/{name}") from exc
    if toggle not in ("on", "off") or (toggle == "on") != name.endswith("-lightning"):
        raise ValueError(f"A14B coordinate Lightning toggle differs from its name: {route}/{name}")
    if toggle == "off":
        if any(flag in arguments for flag in ("--lightning-hub", "--lora-high", "--lora-low")):
            raise ValueError(f"Lightning-off coordinate carries a LoRA: {route}/{name}")
        return
    # The entrypoint resolves the Lightning snapshot from the hub, as the product does; the pair is
    # sealed (hashed) by path and only checked against what the product loads.
    for flag in ("--lightning-hub", "--lora-high", "--lora-low"):
        argument_value(arguments, flag)
    if float(argument_value(arguments, "--guidance")) != 1.0 or argument_value(arguments, "--steps") != "4":
        raise ValueError(f"Lightning coordinate must run the 4-step guidance-1 recipe: {route}/{name}")


def load_wan_manifest(path, backend="candle-cuda"):
    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Wan campaign manifest is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != set(WAN_ROUTES):
        raise ValueError("Wan campaign manifest must contain exactly the five registered routes")
    coverage = lane_coverage(backend)["wan"]
    result = {}
    for route in WAN_ROUTES:
        entry = document[route]
        if (
            not isinstance(entry, dict)
            or entry.get("provider_id") != route
            or not isinstance(entry.get("entrypoint"), str)
            or not isinstance(entry.get("snapshot"), str)
            or entry.get("residency_strategy") != PRODUCT_RESIDENCY[route]
            or not isinstance(entry.get("coordinates"), dict)
        ):
            raise ValueError(f"Wan manifest entry is malformed: {route}")
        binary = Path(expand_argument(entry["entrypoint"])).expanduser().resolve()
        snapshot = Path(expand_argument(entry["snapshot"])).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Wan manifest entrypoint is not executable: {route}")
        if binary.stem != lane_entry(backend, route)["entrypoint_stem"]:
            raise ValueError(f"Wan manifest entrypoint does not match registered route: {route}")
        validate_snapshot_layout(snapshot, f"Wan manifest {route}", route, backend)
        revision = model_snapshot_revision(snapshot)
        residency_strategy = entry["residency_strategy"]
        expected_coordinates = set(coverage[route]["coordinates"])
        if set(entry["coordinates"]) != expected_coordinates:
            raise ValueError(f"Wan manifest coordinates do not match frozen coverage: {route}")
        base_args = entry.get("args", [])
        if not isinstance(base_args, list) or any(not isinstance(item, str) for item in base_args):
            raise ValueError(f"Wan manifest base args are malformed: {route}")
        route_specs = {}
        for name in sorted(expected_coordinates):
            coordinate_args = entry["coordinates"][name]
            if not isinstance(coordinate_args, list) or any(not isinstance(item, str) for item in coordinate_args):
                raise ValueError(f"Wan coordinate args are malformed: {route}/{name}")
            arguments = tuple(expand_argument(item) for item in (*base_args, *coordinate_args))
            if PROTECTED_FLAGS.intersection(arguments):
                raise ValueError(f"Wan coordinate overrides campaign identity: {route}/{name}")
            if route in ("wan_vace", "wan2_2_vace_fun_14b"):
                arguments = ("--sc20686-route", route, *arguments)
            arguments = normalize_file_arguments(arguments)
            validate_coordinate_arguments(
                route, name, arguments, coverage[route]["coordinates"][name]
            )
            if backend == "mlx-metal":
                try:
                    int(argument_value(arguments, "--seed"))
                except ValueError as exc:
                    raise ValueError(f"Metal coordinate must seal one integer --seed: {route}/{name}") from exc
                validate_metal_lightning(route, name, arguments)
            file_hashes, inventory = hash_file_arguments(arguments)
            identity = route_manifest_identity(
                route, name, binary, snapshot, revision, residency_strategy,
                arguments, inventory,
            )
            route_specs[name] = CoordinateSpec(
                "wan", route, name, binary, snapshot, revision,
                residency_strategy, arguments, identity, file_hashes, inventory,
            )
        result[route] = route_specs
    return result


def flux_coordinates(
    entrypoint, snapshot, reference, reference2, backend="candle-cuda", kv_snapshot=None,
):
    """Every FLUX.2 route of the lane, at the frozen coordinates. The Metal lane additionally
    measures the MLX-only `flux2_klein_9b_kv_edit` route (its own snapshot), which owns the only
    persistent reference-K/V cache; the Candle lane has no such route."""
    entrypoint = Path(entrypoint).resolve()
    reference = Path(reference).resolve()
    reference2 = Path(reference2).resolve()
    if not entrypoint.is_file() or not os.access(entrypoint, os.X_OK):
        raise ValueError("FLUX campaign entrypoint is not executable")
    coverage = lane_coverage(backend).get("flux2-klein", {})
    routes = [route for route in FLUX_ROUTES if route in coverage]
    if set(routes) != set(coverage):
        raise ValueError("FLUX lane coverage names an unregistered route")
    snapshots = {"flux2_klein_9b_edit": snapshot, "flux2_klein_9b_kv_edit": kv_snapshot}
    if "flux2_klein_9b_kv_edit" not in routes and kv_snapshot is not None:
        raise ValueError(f"{backend} has no flux2_klein_9b_kv_edit route; drop --flux-kv-snapshot")
    result = []
    for route in routes:
        if snapshots[route] is None:
            raise ValueError(f"{backend} FLUX coverage requires a {route} snapshot")
        route_snapshot = Path(snapshots[route]).resolve()
        if entrypoint.stem != lane_entry(backend, route)["entrypoint_stem"]:
            raise ValueError(f"FLUX campaign entrypoint does not match registered route: {route}")
        validate_snapshot_layout(route_snapshot, f"FLUX campaign {route}", route, backend)
        revision = model_snapshot_revision(route_snapshot)
        residency_strategy = PRODUCT_RESIDENCY[route]
        definitions = {
            "edit-512-ref1-cfg1": (
                "--reference", str(reference), "--single-only", "--width", "512",
                "--height", "512", "--prompt", "SC-20686 edit one reference",
                "--guidance", "1", "--steps", "4",
            ),
            "edit-768x512-ref2-cfg2": (
                "--reference", str(reference), "--reference2", str(reference2),
                "--single-only", "--width", "768", "--height", "512", "--prompt",
                "SC-20686 edit two references", "--guidance", "2", "--steps", "4",
            ),
        }
        if backend == "mlx-metal":
            # The strict MLX entrypoint refuses the Candle harness's --single-only and requires
            # every coordinate argument, the seed included, so it is sealed with the route.
            definitions = {
                name: tuple(item for item in arguments if item != "--single-only") + ("--seed", "42")
                for name, arguments in definitions.items()
            }
        expected_coordinates = coverage[route]["coordinates"]
        if set(definitions) != set(expected_coordinates):
            raise ValueError("FLUX coordinate definitions do not match frozen coverage")
        for name, arguments in definitions.items():
            width = int(argument_value(arguments, "--width"))
            height = int(argument_value(arguments, "--height"))
            prompt = argument_value(arguments, "--prompt")
            expected = expected_coordinates[name]
            actual = {
                "batch": 1, "resolution": f"{width}x{height}", "frames": 1,
                "reference_count": arguments.count("--reference") + arguments.count("--reference2"),
                "prompt": digest(prompt.encode("utf-8")),
                "guidance": float(argument_value(arguments, "--guidance")),
            }
            if any(
                actual[field] != (float(value) if field == "guidance" else value)
                for field, value in expected.items()
            ):
                raise ValueError(f"FLUX coordinate differs from frozen coverage: {name}")
            file_hashes, inventory = hash_file_arguments(arguments)
            result.append(CoordinateSpec(
                "flux2-klein", route, name, entrypoint, route_snapshot, revision,
                residency_strategy, arguments,
                route_manifest_identity(
                    route, name, entrypoint, route_snapshot, revision,
                    residency_strategy, arguments, inventory,
                ),
                file_hashes, inventory,
            ))
    return result


def seal(row, filename):
    unsigned = dict(row)
    unsigned["raw_receipt_sha256"] = ""
    unsigned["raw_receipt_sidecar_sha256"] = ""
    raw = canonical(unsigned)
    receipt_hash = digest(raw)
    sidecar = f"{receipt_hash}  {filename}\n".encode("utf-8")
    sealed = dict(unsigned)
    sealed["raw_receipt_sha256"] = receipt_hash
    sealed["raw_receipt_sidecar_sha256"] = digest(sidecar)
    return sealed, raw, sidecar


def geometry_from(events):
    metadata_events = [event for event in events if event.get("phase") == "metadata"]
    if len(metadata_events) != 1:
        raise ValueError("producer must emit exactly one post-bind metadata event")
    geometry = metadata_events[0].get("geometry")
    if not isinstance(geometry, dict) or any(key not in geometry for key in GEOMETRY):
        raise ValueError("producer metadata must contain all typed geometry axes")
    if not isinstance(geometry["resolution"], str) or not re.fullmatch(
        r"[1-9][0-9]*x[1-9][0-9]*", geometry["resolution"]
    ):
        raise ValueError("producer resolution must be positive WxH")
    for key in ("batch", "reference_count", "frames", "layers", "heads", "head_dimension", "sq", "skv"):
        minimum = 0 if key == "reference_count" else 1
        value = geometry[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"producer geometry {key} is invalid")
    if not isinstance(geometry["prompt"], str) or not re.fullmatch(r"[0-9a-f]{64}", geometry["prompt"]):
        raise ValueError("producer prompt identity is invalid")
    try:
        guidance = float(geometry["guidance"])
    except (TypeError, ValueError):
        raise ValueError("producer guidance is invalid") from None
    if not math.isfinite(guidance):
        raise ValueError("producer guidance is invalid")
    if any(not isinstance(geometry[key], str) or not geometry[key] for key in ("dtype", "mask", "rope")):
        raise ValueError("producer dtype/mask/RoPE identity is invalid")
    return {key: geometry[key] for key in GEOMETRY}


def _drain(stream, chunks):
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            chunks.append(chunk)
    finally:
        stream.close()


def _sample_rss(pid):
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    if system == "Windows":
        try:
            value = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64"],
                text=True, encoding="utf-8", stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
            return int(value) if value else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        value = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, encoding="utf-8",
            stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
        return int(value) * 1024 if value else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def parse_event_transcript(payload, variant, arm):
    """Accept only complete JSONL from the adapter-owned observer channel."""
    if not payload or b"\r" in payload or not payload.endswith(b"\n"):
        raise ValueError(f"{variant}/{arm} emitted a missing or malformed observer transcript")
    try:
        lines = payload.decode("utf-8").splitlines()
        if not lines:
            raise ValueError("event transcript is empty")
        events = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{variant}/{arm} emitted a missing or malformed observer transcript") from exc
    if any(not isinstance(event, dict) for event in events):
        raise ValueError(f"{variant}/{arm} emitted a missing or malformed observer transcript")
    return events


PRODUCT_ADMISSION_ESTIMATE_SCHEMA = "sc20686-product-admission-estimate-v1"
PRODUCT_ADMISSION_ESTIMATE_SOURCE = "product-admission-profile"


def product_admission_estimate(entrypoint, snapshot, variant, extra_args=(), timeout_seconds=600):
    """The MLX entrypoint's `--sc20686-estimate` answer for one coordinate: the product's own
    pre-spawn admission estimate of exactly the run `run_entrypoint` would start (same route,
    snapshot and coordinate arguments). The estimate mode loads no weights and never touches MLX,
    so it runs unsupervised and bounded by `timeout_seconds`. A failed or malformed answer is an
    error: the coordinate is not started on a guess."""
    command = [
        str(Path(entrypoint).resolve()), "--sc20686-estimate", "--variant", variant,
        "--snapshot", str(Path(snapshot).resolve()), *map(str, extra_args),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL,
                                   timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"{variant} product admission estimate did not run: {error}") from error
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", "replace").strip()[-400:]
        raise ValueError(f"{variant} product admission estimate failed ({completed.returncode}): {tail}")
    try:
        document = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as error:
        raise ValueError(f"{variant} product admission estimate is malformed") from error
    phases = document.get("phases") if isinstance(document, dict) else None
    estimate = document.get("estimateBytes") if isinstance(document, dict) else None
    if (document.get("schema") != PRODUCT_ADMISSION_ESTIMATE_SCHEMA
            or document.get("route") != variant
            or document.get("source") != PRODUCT_ADMISSION_ESTIMATE_SOURCE
            or type(estimate) is not int or estimate <= 0
            or not isinstance(phases, dict) or not phases
            or any(type(value) is not int or value < 0 for value in phases.values())
            or max(phases.values()) != estimate):
        raise ValueError(f"{variant} product admission estimate is malformed")
    return document


def admission_estimate_arguments(estimate, measured_peak_host_bytes, policy):
    """The run_guarded estimate arguments for one arm: the product estimate (max with the measured
    peak of a completed arm of the same coordinate request, applied by the supervisor), or the
    cap fallback when there is no estimate or the product estimate exceeds the child cap (the
    product's conservative single-pass decode pricing can exceed a cap the run itself fits)."""
    if estimate is None or estimate["estimateBytes"] > policy.child_footprint_cap_bytes:
        return {}
    measured = (measured_peak_host_bytes if measured_peak_host_bytes is not None
                and measured_peak_host_bytes <= policy.child_footprint_cap_bytes else None)
    return {"static_floor_host_bytes": estimate["estimateBytes"],
            "host_estimate_source": PRODUCT_ADMISSION_ESTIMATE_SOURCE,
            "measured_peak_host_bytes": measured}


def run_entrypoint(
    entrypoint, snapshot, variant, arm, inference_revision, residency_strategy,
    extra_args=(), timeout_seconds=21600, *, safety_policy, static_floor_host_bytes=None,
    static_floor_gpu_bytes=None, probe=None, failure_root=None, product_estimate=None,
    measured_peak_host_bytes=None,
):
    # Admission is runtime-guarded (watchdog caps, free reserves, deadline) under
    # estimate-plus-reserve-v1: `product_estimate` (the entrypoint's `--sc20686-estimate` answer)
    # is the arm's estimate unless it exceeds the cap, which falls back to the cap. No static
    # whole-process peak bound is required. Refusal, abort or failure is sealed unaccepted.
    # Provider stdout is free to contain progress bars (including carriage returns).  The adapter
    # owns this private file and accepts observer events only from it, never by filtering stdout.
    directory = Path(tempfile.mkdtemp(prefix="sc20686-events-"))
    result = None
    try:
        run_directory = directory / "sealed-run"
        run_directory.mkdir()
        event_path = run_directory / "events.jsonl"
        media_output = run_directory / (
            "media.png" if variant in FLUX_ROUTES else "media"
        )
        command = [
            str(Path(entrypoint).resolve()), "--sc20686-campaign", "--sc20686-events",
            str(event_path), "--sc20686-source-ref", inference_revision,
            "--sc20686-residency", residency_strategy,
            "--snapshot", str(Path(snapshot).resolve()), "--variant", variant,
            *map(str, extra_args),
        ]
        if arm == "cancel":
            command.append("--sc20686-cancel")
        elif arm == "control":
            command.append("--sc20686-schedule-control")
        command.extend(("--out", str(media_output)))
        if safety_policy.deadline_seconds > timeout_seconds:
            raise supervisor.SupervisionError("invalid-timeout", "safety deadline exceeds requested run timeout")
        estimate_arguments = admission_estimate_arguments(
            product_estimate, measured_peak_host_bytes, safety_policy)
        if estimate_arguments and static_floor_host_bytes is not None:
            raise ValueError(f"{variant}/{arm} has both a static floor and a product estimate")
        estimate_arguments.setdefault("static_floor_host_bytes", static_floor_host_bytes)
        result = supervisor.run_guarded(
            command, cwd=run_directory, env=os.environ.copy(), policy=safety_policy,
            stdout_path=run_directory / "stdout.log", stderr_path=run_directory / "stderr.log",
            event_path=event_path, static_floor_gpu_bytes=static_floor_gpu_bytes, probe=probe,
            **estimate_arguments,
        )
        stdout = (run_directory / "stdout.log").read_bytes()
        stderr = (run_directory / "stderr.log").read_bytes()
        process_samples = list(result.samples)
        try:
            if event_path.is_symlink():
                raise ValueError(f"{variant}/{arm} observer transcript is a symlink")
            event_transcript = event_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"{variant}/{arm} emitted a missing or malformed observer transcript") from exc
        events = parse_event_transcript(event_transcript, variant, arm)
        events.extend(process_samples)
        if not events or not process_samples:
            raise ValueError(f"{variant}/{arm} lacks observer or process evidence")
        return CampaignRun(
            events, stdout, stderr, tuple(command), tuple(process_samples), event_transcript,
            media_output, directory, {
                "pid": result.pid, "exitCode": result.returncode,
                "peakHostBytes": result.peak_host_bytes,
                "peakGpuBytes": result.peak_gpu_bytes,
                "hostFreeAtLaunch": result.host_free_at_launch,
                "gpuFreeAtLaunch": result.gpu_free_at_launch,
                "elapsedSeconds": result.elapsed_seconds,
                "ownedProcessGroupReaped": True,
                "admission": result.admission,
                **({"productAdmissionEstimate": product_estimate} if product_estimate is not None else {}),
            },
        )
    except Exception as error:
        failure = error if isinstance(error, supervisor.SupervisionError) else None
        if failure is None and result is not None:
            # The child exited cleanly but left invalid evidence: a failed, unaccepted arm.
            failure = supervisor.SupervisionError("invalid-evidence", str(error))
            failure.pid, failure.admission = result.pid, result.admission
        unaccepted = failure is not None
        if unaccepted and failure_root is not None:
            supervisor.write_unaccepted_record(
                directory / "unaccepted.json", kind="sc-20686-unaccepted-arm",
                coordinate=f"{variant}/{arm}", error=failure,
            )
        if failure_root is not None and (unaccepted or (run_directory.is_dir() and any(run_directory.iterdir()))):
            failure_root = Path(failure_root)
            failure_root.mkdir(exist_ok=True)
            retained = failure_root / f"incomplete-{uuid.uuid4()}"
            try:
                shutil.move(str(directory), retained)
            except OSError as move_error:
                raise ValueError(f"{error}; incomplete diagnostics remain at {directory}; move failed: {move_error}") from error
            raise ValueError(f"{error}; incomplete diagnostics retained at {retained}") from error
        shutil.rmtree(directory, ignore_errors=True)
        raise


def cleanup_campaign_run(run):
    if run.cleanup_root is not None:
        shutil.rmtree(run.cleanup_root, ignore_errors=True)


def media_artifacts(run, stem, arm):
    output = run.media_output
    if output is None:
        raise ValueError(f"{stem} lacks an adapter-owned media output")
    output = Path(output)
    if not output.is_absolute() or output.is_symlink():
        raise ValueError(f"{stem} media output is not an exact regular path")
    try:
        out_index = run.command.index("--out")
    except ValueError as exc:
        raise ValueError(f"{stem} command lacks its media output") from exc
    if out_index + 1 >= len(run.command) or Path(run.command[out_index + 1]) != output:
        raise ValueError(f"{stem} command does not bind its exact media output")

    sources = {}
    files = []
    output_kind = "absent"
    if arm == "cancel":
        if output.exists():
            raise ValueError(f"{stem} cancellation arm left a media output")
    else:
        if output.is_file():
            output_kind = "file"
            candidates = [(output.name, output)]
        elif output.is_dir():
            output_kind = "directory"
            candidates = []
            for candidate in sorted(output.rglob("*")):
                if candidate.is_symlink():
                    raise ValueError(f"{stem} media output contains a symlink")
                if candidate.is_file():
                    candidates.append((candidate.relative_to(output).as_posix(), candidate))
        else:
            raise ValueError(f"{stem} full-generation arm lacks generated media")
        if not candidates:
            raise ValueError(f"{stem} full-generation arm has an empty media output")
        for index, (relative_path, source) in enumerate(candidates):
            size = source.stat().st_size
            if size <= 0:
                raise ValueError(f"{stem} generated media is empty: {relative_path}")
            name = f"{stem}.media-{index:04d}"
            content_hash = file_identity(source)
            sources[name] = source
            files.append({
                "artifact": name,
                "relative_path": relative_path,
                "bytes": size,
                "sha256": content_hash,
            })
    manifest = canonical({
        "schema": "sc-20686-media-manifest-v1",
        "arm": arm,
        "output_kind": output_kind,
        "output_name": output.name,
        "files": files,
    })
    return manifest, sources


def make_row(args, config, snapshot_hash, snapshot_bytes, events):
    if args.fake:
        raise ValueError("fake evidence is test-only and cannot produce a receipt")
    backend = getattr(args, "backend", "candle-cuda")
    if backend not in LANES:
        raise ValueError("campaign measurement backend is not a registered lane")
    reducer = _load_reducer()
    entry = lane_entry(backend, args.variant)
    if entry["family"] != args.family:
        raise ValueError("campaign family differs from the registered lane route")
    kind = entry["cache_kind"]
    metadata_events = [event for event in events if event.get("phase") == "metadata"]
    metrics_events = [event for event in events if event.get("phase") == "metrics"]
    if len(metadata_events) != 1 or len(metrics_events) != 1:
        raise ValueError("entrypoint must emit exactly one metadata and metrics event")
    metadata, metrics = metadata_events[0], metrics_events[0]
    if metadata.get("snapshot_sha256") != snapshot_hash or metadata.get("snapshot_bytes") != snapshot_bytes:
        raise ValueError("entrypoint model identity does not match the hashed snapshot")
    if metadata.get("source_ref") != config.get("inference_revision"):
        raise ValueError("entrypoint inference revision differs from the sealed repository revision")
    if metadata.get("model_snapshot_revision") != config.get("model_snapshot_revision"):
        raise ValueError("entrypoint model revision differs from the sealed snapshot revision")
    if metadata.get("residency_strategy") != config.get("residency_strategy"):
        raise ValueError("entrypoint residency differs from the sealed product strategy")
    require_revision(metadata["source_ref"], "entrypoint inference revision")
    require_revision(metadata["model_snapshot_revision"], "entrypoint model snapshot revision")
    if metadata["residency_strategy"] != PRODUCT_RESIDENCY.get(args.variant):
        raise ValueError("entrypoint residency does not match the frozen product route")
    if metadata.get("variant") != args.variant:
        raise ValueError("producer variant does not match the exact product route")
    if metadata.get("backend", "candle-cuda") != backend:
        raise ValueError("entrypoint measurement backend differs from the campaign lane")
    reducer.validate_backend_identity(events, backend)
    for field in ("route_manifest_sha256", "source_map_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(config.get(field, ""))):
            raise ValueError(f"campaign {field} is missing")
    media_manifests = {
        name: item_hash
        for name, item_hash in config.get("evidence_artifact_sha256", {}).items()
        if name.endswith(".media.json")
    }
    if len(media_manifests) != 1:
        raise ValueError("campaign evidence must bind exactly one media manifest")
    metric_keys = (
        "current_persistent_bytes", "current_read_transient_bytes",
        "candidate_persistent_bytes", "candidate_read_transient_bytes",
        "generation_duration_ms", "cache_read_duration_ms",
        "joint_attention_context_duration_ms", "reused_requests", "minimum_cache_reads",
    )
    if any(
        not isinstance(metrics.get(key), (int, float))
        or isinstance(metrics.get(key), bool)
        or not math.isfinite(metrics[key])
        or metrics[key] < 0
        for key in metric_keys
    ):
        raise ValueError("entrypoint metrics are incomplete")
    runtime_attribution_available = metrics.get("reference_runtime_attribution_available")
    if not isinstance(runtime_attribution_available, bool):
        raise ValueError("entrypoint reference runtime attribution availability is missing")
    if (
        metrics["cache_read_duration_ms"] > metrics["generation_duration_ms"]
        or metrics["joint_attention_context_duration_ms"] > metrics["generation_duration_ms"]
        or (
            runtime_attribution_available is False
            and metrics["cache_read_duration_ms"] != 0
        )
    ):
        raise ValueError("entrypoint runtime duration exceeds the generation")
    phases = [event.get("phase") for event in events]
    required = {"generation-start", "cross-kv-created", "cross-kv-read", "invalidated", "released"}
    terminal = "cancelled" if args.cancel_campaign else "generation-end"
    required.add(terminal)
    if not required <= set(phases):
        raise ValueError("observer lifecycle hooks are incomplete")
    singleton = ("metadata", "generation-start", terminal, "invalidated", "released", "metrics")
    indices = {}
    for phase in singleton:
        matches = [index for index, value in enumerate(phases) if value == phase]
        if len(matches) != 1:
            raise ValueError(f"producer must emit exactly one {phase} event")
        indices[phase] = matches[0]
    creates = [index for index, value in enumerate(phases) if value == "cross-kv-created"]
    reads = [index for index, value in enumerate(phases) if value == "cross-kv-read"]
    if not (
        indices["metadata"] < indices["generation-start"] < min(creates) < min(reads)
        and max(creates) < indices[terminal]
        and max(reads) < indices[terminal]
        and indices[terminal] < indices["metrics"] < indices["invalidated"] < indices["released"]
    ):
        raise ValueError("observer lifecycle events are out of product order")
    if metadata.get("real_weights") is not True or metadata.get("attention_kind") != "cross":
        raise ValueError("entrypoint must identify a real product cross-attention route")
    if metadata.get("full_generation") is not (not args.cancel_campaign):
        raise ValueError("entrypoint full-generation claim conflicts with campaign arm")
    if backend == "mlx-metal":
        reducer.validate_phase_windows(events, indices[terminal], "cancel" if args.cancel_campaign else "normal")
    operations = entry.get("operations")
    if args.cancel_campaign:
        if metadata.get("cancellation_armed") is not True or not metadata.get("cancellation_arm_id"):
            raise ValueError("deliberate cancellation lacks product-owned arm identity")
        if "generation-end" in phases:
            raise ValueError("cancel arm cannot claim generation completion")
    elif "cancelled" in phases:
        raise ValueError("normal generation cannot contain cancellation")
    read_events = [event for event in events if event.get("phase") == "cross-kv-read"]
    if not any(event.get("transient_bytes", 0) > 0 for event in read_events):
        raise ValueError("entrypoint must report product-owned dense read workspace")
    observed_transient = max(event.get("transient_bytes", 0) for event in read_events)
    observed_reuse = sum(event.get("reused", 0) for event in read_events)
    if metrics["current_read_transient_bytes"] != observed_transient:
        raise ValueError("read transient metric differs from product-owned read events")
    if metrics["candidate_read_transient_bytes"] > observed_transient:
        raise ValueError("candidate read workspace exceeds the measured dense read")
    create_events = [event for event in events if event.get("phase") == "cross-kv-created"]
    geometry = geometry_from(events)
    release_remnant = events[indices["released"]]
    active_allocator_measurement(release_remnant, "post-release remnant")
    if entry["runtime_attribution"]:
        if (
            runtime_attribution_available is not True
            or metrics["joint_attention_context_duration_ms"] != 0
        ):
            raise ValueError("attributable cross-attention runtime cannot also claim joint context")
    elif (
        runtime_attribution_available is not False
        or metrics["cache_read_duration_ms"] != 0
        or metrics["joint_attention_context_duration_ms"] <= 0
    ):
        raise ValueError("joint attention must remain non-attributable runtime context")
    if operations is not None and (
        any(event.get("operation") != operations["create"] for event in create_events)
        or any(event.get("operation") != operations["read"] for event in read_events)
    ):
        raise ValueError("evidence is not anchored at the lane's exact K/V operations")
    if backend == "candle-cuda":
        if any("kv_batch" in event for event in create_events):
            raise ValueError("CUDA-lane events may not supply a Metal kv_batch")
    else:
        exact_batch = reducer.expected_kv_batch(entry, geometry)
        if any(
            not isinstance(event.get("kv_batch"), int) or isinstance(event.get("kv_batch"), bool)
            or event["kv_batch"] != exact_batch
            for event in create_events
        ):
            raise ValueError("Metal K/V batch differs from the route's exact CFG layout")
    if kind == "recomputed":
        if metrics["current_persistent_bytes"] != 0 or any(event.get("persistent_bytes") != 0 for event in create_events):
            raise ValueError("recomputed route must report its non-persistent K/V honestly")
        if (
            metrics["minimum_cache_reads"] != metrics["reused_requests"]
            or metrics["minimum_cache_reads"] > observed_reuse
        ):
            raise ValueError("logical payload reuse differs from product-owned recomputations")
        if backend == "mlx-metal":
            # Slice-by-slice exactness: a Metal route may recompute slices of different live lengths
            # and dtypes (Wan-VACE's unpadded CFG branches over F32 main and BF16 VACE blocks).
            exact_dense = reducer.exact_metal_recomputed_slices(
                create_events, read_events, geometry, reducer.expected_kv_batch(entry, geometry)
            )
            uniform = True
        else:
            exact_dense = dense_reference_kv_bytes(geometry)
            uniform = all(
                event.get("transient_bytes") == exact_dense
                for event in (*create_events, *read_events)
            )
        if (
            not uniform
            or metrics["current_read_transient_bytes"] != exact_dense
            or metrics["candidate_read_transient_bytes"] != exact_dense
        ):
            raise ValueError("recomputed K/V slice bytes differ from exact live geometry/dtype")
        for event in read_events:
            active_allocator_measurement(event, "recomputed K/V attention read")
    else:
        if metrics["reused_requests"] != observed_reuse:
            raise ValueError("reuse metric differs from product-owned read events")
        if metrics["current_persistent_bytes"] == 0:
            raise ValueError("persistent-cache route cannot claim zero persistence")
        created_ids = [event.get("cache_id") for event in create_events]
        if any(not isinstance(cache_id, int) or cache_id < 1 for cache_id in created_ids) or len(created_ids) != len(set(created_ids)):
            raise ValueError("persistent cache creation identities are missing or duplicated")
        read_ids = [event.get("cache_id") for event in read_events]
        if any(cache_id not in set(created_ids) for cache_id in read_ids):
            raise ValueError("persistent cache read does not identify its created cache")
        release_events = [
            event for event in events if event.get("phase") == "cross-kv-released"
        ]
        released_ids = [event.get("cache_id") for event in release_events]
        if sorted(released_ids) != sorted(created_ids):
            raise ValueError("persistent cache invalidation does not release every exact cache once")
        for cache_id in created_ids:
            created_index = next(
                index for index, event in enumerate(events)
                if event.get("phase") == "cross-kv-created" and event.get("cache_id") == cache_id
            )
            released_index = next(
                index for index, event in enumerate(events)
                if event.get("phase") == "cross-kv-released" and event.get("cache_id") == cache_id
            )
            cache_reads = [
                index for index, event in enumerate(events)
                if event.get("phase") == "cross-kv-read" and event.get("cache_id") == cache_id
            ]
            if not (
                created_index < released_index < indices["released"]
                and all(created_index < read_index < released_index for read_index in cache_reads)
            ):
                raise ValueError("persistent cache read/release ordering is not product-owned")
        per_cache_reads = {cache_id: read_ids.count(cache_id) for cache_id in created_ids}
        if metrics["minimum_cache_reads"] != min(per_cache_reads.values(), default=0):
            raise ValueError("persistent minimum cache reuse is not the per-cache minimum")
        live_dense = live_candidate = peak_dense = peak_candidate = 0
        created_bytes = {}
        for event in events:
            phase, cache_id = event.get("phase"), event.get("cache_id")
            if phase == "cross-kv-created":
                dense = event.get("persistent_bytes")
                candidate = event.get("candidate_persistent_bytes")
                if (
                    not isinstance(dense, int) or isinstance(dense, bool) or dense <= 0
                    or not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0
                ):
                    raise ValueError("persistent creation lacks exact dense/candidate retained bytes")
                if backend == "mlx-metal" and dense != dense_reference_kv_bytes(
                    geometry, event["kv_batch"]
                ):
                    raise ValueError("Metal persistent cache bytes differ from its exact live tensors")
                created_bytes[cache_id] = (dense, candidate)
                live_dense += dense
                live_candidate += candidate
                peak_dense = max(peak_dense, live_dense)
                peak_candidate = max(peak_candidate, live_candidate)
            elif phase == "cross-kv-released" and cache_id in created_bytes:
                dense, candidate = created_bytes[cache_id]
                if (
                    event.get("persistent_bytes") != dense
                    or event.get("candidate_persistent_bytes") != candidate
                ):
                    raise ValueError("persistent release bytes differ from the exact created cache")
                live_dense -= dense
                live_candidate -= candidate
        if live_dense or live_candidate:
            raise ValueError("persistent retained accounting does not close at release")
        if (
            metrics["current_persistent_bytes"] != peak_dense
            or metrics["candidate_persistent_bytes"] != peak_candidate
        ):
            raise ValueError("persistent metric differs from exact simultaneous residency")
        for event in read_events:
            allocator = active_allocator_measurement(event, "persistent cache read")
            if event.get("transient_bytes") != (
                allocator["allocator_high_bytes"] - allocator["allocator_before_bytes"]
            ):
                raise ValueError("persistent read transient differs from active allocator high-water")
        for event in release_events:
            active_allocator_measurement(event, "persistent cache release remnant")
    if args.family == "flux2-klein" and geometry["reference_count"] < 1:
        raise ValueError("FLUX edit campaign requires a live reference image")
    coordinate_id = digest(canonical(geometry))[:16]
    allocator = [
        {"phase": event["phase"], "peak_bytes": event["peak_bytes"]}
        for event in events
        if event.get("sample_kind") == "allocator" and "peak_bytes" in event
    ]
    process = [
        {"phase": event["phase"], "peak_bytes": event["peak_bytes"]}
        for event in events
        if event.get("sample_kind") == "process" and "peak_bytes" in event
    ]
    if not allocator or not process or any(sample["peak_bytes"] <= 0 for sample in allocator + process):
        raise ValueError("producer must provide distinct physical allocator and process samples")
    return {
        "producer": PRODUCER,
        "backend": backend,
        "family": args.family,
        "variant": args.variant,
        "coordinate_name": args.coordinate_name,
        "coordinate_id": coordinate_id,
        "arm": "cancel" if args.cancel_campaign else "normal",
        "source_ref": metadata["source_ref"],
        "model_snapshot_revision": metadata["model_snapshot_revision"],
        "residency_strategy": metadata["residency_strategy"],
        "route_manifest_sha256": config["route_manifest_sha256"],
        "source_map_sha256": config["source_map_sha256"],
        "model_snapshot_sha256": snapshot_hash,
        "model_snapshot_bytes": snapshot_bytes,
        "input_file_sha256": dict(config.get("input_file_sha256", {})),
        "evidence_artifact_sha256": dict(config.get("evidence_artifact_sha256", {})),
        "media_manifest_sha256": next(iter(media_manifests.values())),
        "geometry": geometry,
        "lifecycle": {
            "created": len(create_events), "reused": len(read_events),
            "invalidated": phases.count("invalidated"), "cancelled": phases.count("cancelled"),
            "released": phases.count("released"),
        },
        "allocator_samples": allocator,
        "process_samples": process,
        "raw_receipt_sha256": "",
        "raw_receipt_sidecar_sha256": "",
        "real_weights": True,
        "full_generation": metadata["full_generation"],
        "attention_kind": "cross",
        "reference_runtime_attribution_available": runtime_attribution_available,
        **{key: metrics[key] for key in metric_keys},
        "observer_events": events,
    }


def _load_reducer():
    spec = importlib.util.spec_from_file_location("sc20686_reducer", REDUCER)
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    return reducer


def render_markdown(keys, decision, source_map):
    backend = decision["backend"]
    lines = [
        "# SC-20686 campaign", "",
        f"Measurement lane: `{backend}` ({source_map['lanes'][backend]['measures']})", "",
        "## Frozen coordinates", "",
    ]
    lines.extend(
        f"- {family}/{variant} `{coordinate}` arm=`{arm}`"
        for family, variant, coordinate, arm in keys
    )
    lines.extend(["", "## Per-family decisions", "", "```json", json.dumps(decision["decisions"], indent=2, sort_keys=True), "```", "", "## Sealed source map", ""])
    target = source_map["compatibility_target"]
    lines.extend([
        "### Compatibility target", "",
        f"- Story: `{target['story']}`", f"- Format: `{target['format']}`",
        f"- Bits: `{target['bits']}`", f"- Group size: `{target['group_size']}`",
        f"- Metadata: {target['metadata']}",
        f"- Pending key tail: {target['pending_key_tail']}", "",
    ])
    for variant, entry in source_map["lanes"][backend]["variants"].items():
        lines.extend([
            f"### `{variant}`", "",
            f"- Product route: `{entry['route']}`", f"- Cache kind: `{entry['cache_kind']}`",
            f"- Current kernel: {entry['current_kernel']}",
            f"- Cache format: {entry['cache_format']}", f"- Paging: {entry['paging']}",
            f"- Offload: {entry['offload']}", f"- Recompute: {entry['recompute']}",
            f"- Compatibility: {entry['compatibility']}",
        ])
        anchor_kinds = ["activation", "creation", "reads", "release"]
        if "reference_slice" in entry:
            anchor_kinds.insert(2, "reference_slice")
        for kind in anchor_kinds:
            anchors = ", ".join(f"`{item['path']}:{item['line']}#{item['symbol']}`" for item in entry[kind])
            lines.append(f"- {kind.title()} anchors: {anchors}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _prepare_resume(root, resolved, policy, stop_files=()):
    root = Path(root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("resume directory must be an absolute, nonsymlink path")
    if root.resolve() == INFERENCE_ROOT.resolve() or INFERENCE_ROOT.resolve() in root.resolve().parents:
        raise ValueError("resume directory must be outside the repository")
    identity = {
        "schema": "sc-20686-media-resume-v1",
        "resolvedInputsSha256": digest(canonical(resolved)),
        "safetyPolicySha256": policy.sha256,
        "coverageSha256": file_identity(COVERAGE),
        "sourceMapSha256": file_identity(SOURCE_MAP),
        "adapterSha256": file_identity(Path(__file__)),
    }
    raw = canonical(identity)
    resolved_raw = canonical(resolved)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("resume path is not an exact directory")
        if (root / "identity.json").read_bytes() != raw or (root / "identity.json.sha256").read_text(encoding="ascii") != f"{digest(raw)}  identity.json\n":
            raise ValueError("resume identity is missing, corrupt, or stale")
        if (root / "resolved.json").read_bytes() != resolved_raw or (root / "resolved.json.sha256").read_text(encoding="ascii") != f"{digest(resolved_raw)}  resolved.json\n":
            raise ValueError("resume resolved inputs changed")
        unexpected = {
            item.name for item in root.iterdir()
            if not supervisor.is_operator_stop_entry(root, stop_files, item.name)
        } - {"identity.json", "identity.json.sha256", "resolved.json", "resolved.json.sha256", "units", "failed", "logs"}
        if unexpected:
            raise ValueError(f"resume directory has unexpected entries: {sorted(unexpected)}")
    else:
        root.mkdir(parents=True)
        (root / "identity.json").write_bytes(raw)
        (root / "identity.json.sha256").write_text(f"{digest(raw)}  identity.json\n", encoding="ascii")
        (root / "resolved.json").write_bytes(resolved_raw)
        (root / "resolved.json.sha256").write_text(f"{digest(resolved_raw)}  resolved.json\n", encoding="ascii")
    (root / "units").mkdir(exist_ok=True)
    if (root / "units").is_symlink():
        raise ValueError("resume unit directory may not be symlinked")
    return digest(raw)


def _unit_files(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("resume unit contains a symlink")
        if path.is_file() and path not in {root / "record.json", root / "record.json.sha256"}:
            files[path.relative_to(root).as_posix()] = file_identity(path)
    return files


def _resume_policy_sha(root, identity_sha):
    raw = (Path(root) / "identity.json").read_bytes()
    if digest(raw) != identity_sha:
        raise ValueError("resume identity does not match its sealed digest")
    return json.loads(raw)["safetyPolicySha256"]


def _load_unit(root, stem, identity_sha, variant, arm):
    unit = root / "units" / stem
    if not unit.exists():
        if (root / "units" / f".{stem}.partial").exists():
            raise ValueError(f"{stem} resume unit was interrupted")
        return None
    if unit.is_symlink() or not unit.is_dir():
        raise ValueError(f"{stem} resume unit is not a directory")
    record_raw = (unit / "record.json").read_bytes()
    record = json.loads(record_raw)
    if record_raw != canonical(record) or (unit / "record.json.sha256").read_text(encoding="ascii") != f"{digest(record_raw)}  record.json\n":
        raise ValueError(f"{stem} resume record seal is corrupt")
    if set(record) != {"schema", "identitySha256", "stem", "files"} or record.get("schema") != "sc-20686-resume-unit-v1" or record.get("identitySha256") != identity_sha or record.get("stem") != stem:
        raise ValueError(f"{stem} resume record identity is invalid")
    if record.get("files") != _unit_files(unit):
        raise ValueError(f"{stem} resume files are missing, extra, or corrupted")
    command = tuple(json.loads((unit / "command.json").read_bytes())["argv"])
    samples = tuple(json.loads((unit / "process.json").read_bytes())["samples"])
    supervision = json.loads((unit / "supervision.json").read_bytes())
    if supervision.get("exitCode") != 0 or supervision.get("ownedProcessGroupReaped") is not True or not isinstance(supervision.get("pid"), int) or supervision["pid"] <= 0:
        raise ValueError(f"{stem} resume supervision is not a clean, reaped exit")
    supervisor.validate_admission(supervision.get("admission"), policy_sha256=_resume_policy_sha(root, identity_sha))
    transcript = (unit / "events.jsonl").read_bytes()
    events = parse_event_transcript(transcript, variant, arm)
    events.extend(samples)
    if not samples:
        raise ValueError(f"{stem} resume process evidence is missing")
    return CampaignRun(events, (unit / "stdout").read_bytes(), (unit / "stderr").read_bytes(),
                       command, samples, transcript, unit / "media_output", None, supervision)


def _save_unit(root, stem, identity_sha, variant, arm, run):
    units = root / "units"
    unit = units / stem
    partial = units / f".{stem}.partial"
    if unit.exists() or partial.exists():
        raise ValueError(f"{stem} resume unit already exists or was interrupted")
    if run.supervision is None or run.supervision.get("exitCode") != 0 or run.supervision.get("ownedProcessGroupReaped") is not True:
        raise ValueError(f"{stem} lacks successful supervisor and cleanup evidence")
    supervisor.validate_admission(run.supervision.get("admission"), policy_sha256=_resume_policy_sha(root, identity_sha))
    partial.mkdir()
    try:
        output = Path(run.media_output)
        saved_output = partial / "media_output"
        if output.is_dir():
            shutil.copytree(output, saved_output, symlinks=False)
        elif output.is_file():
            shutil.copyfile(output, saved_output)
        elif arm != "cancel":
            raise ValueError(f"{stem} generated media is missing")
        command = list(run.command)
        command[command.index("--sc20686-events") + 1] = str(unit / "events.jsonl")
        command[command.index("--out") + 1] = str(unit / "media_output")
        (partial / "command.json").write_bytes(canonical({"argv": command}))
        (partial / "process.json").write_bytes(canonical({"samples": list(run.process_samples)}))
        (partial / "supervision.json").write_bytes(canonical(run.supervision))
        (partial / "events.jsonl").write_bytes(run.event_transcript)
        (partial / "stdout").write_bytes(run.stdout)
        (partial / "stderr").write_bytes(run.stderr)
        record = {"schema": "sc-20686-resume-unit-v1", "identitySha256": identity_sha,
                  "stem": stem, "files": _unit_files(partial)}
        record_raw = canonical(record)
        (partial / "record.json").write_bytes(record_raw)
        (partial / "record.json.sha256").write_text(f"{digest(record_raw)}  record.json\n", encoding="ascii")
        os.replace(partial, unit)
    except BaseException:
        # A partial unit is intentionally retained for explicit inspection/refusal.
        raise
    return _load_unit(root, stem, identity_sha, variant, arm)


def _retain_refused_run(run, resume_root, variant, arm, error):
    """Seal a completed arm whose transcript the row builder refuses as unaccepted under
    `failed/`, transcript and all, instead of discarding it with the run's private directory."""
    if run.cleanup_root is None or not Path(run.cleanup_root).is_dir():
        return error
    failure = supervisor.SupervisionError("invalid-evidence", str(error))
    supervision = run.supervision or {}
    failure.pid, failure.admission = supervision.get("pid"), supervision.get("admission")
    source = Path(run.cleanup_root)
    supervisor.write_unaccepted_record(
        source / "unaccepted.json", kind="sc-20686-unaccepted-arm",
        coordinate=f"{variant}/{arm}", error=failure,
    )
    failed = Path(resume_root) / "failed"
    failed.mkdir(exist_ok=True)
    retained = failed / f"incomplete-{uuid.uuid4()}"
    try:
        shutil.move(str(source), retained)
    except OSError as move_error:
        return ValueError(f"{error}; refused evidence remains at {source}; move failed: {move_error}")
    return ValueError(f"{error}; refused evidence retained at {retained}")


def publish_campaign(coordinates, runner, row_builder, destination, input_artifacts=None,
                     *, resume_root=None, resume_identity_sha=None, preflight=None, stop_files=(),
                     schedule_control=False, on_unit=None):
    """Run every coordinate's arms and publish one sealed bundle. The decision campaign runs the
    normal/cancel pair and seals reducer rows; `schedule_control` runs the Metal lane's single
    control arm (no per-read evaluation windows) and seals product-schedule peak summaries."""
    arms = ("control",) if schedule_control else ("normal", "cancel")
    rows = []
    run_artifacts = {}
    run_sources = {}
    captured_runs = []
    used_pids = set()
    final = Path(destination)
    staging = final.with_name(f".{final.name}.staging-{os.getpid()}")
    if final.exists() or staging.exists():
        raise ValueError("campaign destination or staging path already exists")
    try:
        for index, coordinate in enumerate(coordinates):
            for arm in arms:
                if preflight is not None:
                    preflight(coordinate)
                stem = f"run-{index:02d}-{arm}"
                run = (_load_unit(Path(resume_root), stem, resume_identity_sha,
                                  coordinate.variant, arm) if resume_root is not None else None)
                resumed = run is not None
                if run is None:
                    if resume_root is not None:
                        # Between arms only: a running entrypoint is never signalled.
                        supervisor.check_operator_stop(
                            stop_files, Path(resume_root) / "logs", kind="sc-20686-operator-stop",
                            before=stem, index=index * len(arms) + arms.index(arm),
                            total=len(coordinates) * len(arms),
                        )
                    run = runner(coordinate, arm)
                    if preflight is not None:
                        preflight(coordinate)
                if not isinstance(run, CampaignRun):
                    raise ValueError("campaign runner must return a sealed transcript capture")
                if run.supervision is not None:
                    pid = run.supervision.get("pid")
                    if type(pid) is not int or pid <= 0 or pid in used_pids:
                        raise ValueError(f"{stem} reuses or lacks an owned process identity")
                    used_pids.add(pid)
                captured_runs.append(run)
                if on_unit is not None:
                    # Completed (fresh or resumed) arms inform the next arm's admission estimate.
                    on_unit(coordinate, arm, run)
                # Validate the fresh transcript before preserving this expensive arm.
                if not resumed and resume_root is not None:
                    preview_manifest, preview_sources = media_artifacts(run, stem, arm)
                    preview = {f"{stem}.events.jsonl": run.event_transcript,
                               f"{stem}.stdout": run.stdout, f"{stem}.stderr": run.stderr,
                               f"{stem}.command.json": canonical({"argv": list(run.command)}),
                               f"{stem}.process.json": canonical({"samples": list(run.process_samples)}),
                               f"{stem}.supervision.json": canonical(run.supervision),
                               f"{stem}.media.json": preview_manifest}
                    preview_hashes = {name: digest(raw) for name, raw in preview.items()}
                    preview_hashes.update({name: file_identity(path) for name, path in preview_sources.items()})
                    try:
                        row_builder(coordinate, arm, run.events, preview_hashes)
                    except ValueError as error:
                        raise _retain_refused_run(run, resume_root, coordinate.variant, arm, error) from error
                    saved = _save_unit(Path(resume_root), stem, resume_identity_sha,
                                       coordinate.variant, arm, run)
                    cleanup_campaign_run(run)
                    run = saved
                command_payload = canonical({"argv": list(run.command)})
                process_payload = canonical({"samples": list(run.process_samples)})
                media_manifest, media_sources = media_artifacts(run, stem, arm)
                artifacts = {
                    f"{stem}.events.jsonl": run.event_transcript,
                    f"{stem}.stdout": run.stdout,
                    f"{stem}.stderr": run.stderr,
                    f"{stem}.command.json": command_payload,
                    f"{stem}.process.json": process_payload,
                    f"{stem}.media.json": media_manifest,
                }
                if run.supervision is not None:
                    artifacts[f"{stem}.supervision.json"] = canonical(run.supervision)
                evidence_hashes = {
                    name: digest(payload) for name, payload in artifacts.items()
                }
                evidence_hashes.update({
                    name: file_identity(source) for name, source in media_sources.items()
                })
                row = row_builder(coordinate, arm, run.events, evidence_hashes)
                if not isinstance(row, dict):
                    raise ValueError("row builder returned no receipt row")
                rows.append(row)
                run_artifacts.update(artifacts)
                run_sources.update(media_sources)
        reducer = _load_reducer()
        if schedule_control:
            return _publish_schedule_control(
                coordinates, rows, run_artifacts, run_sources, input_artifacts, staging, final,
                reducer,
            )
        keys = [
            (row.get("family"), row.get("variant"), row.get("coordinate_name"), row.get("arm"))
            for row in rows
        ]
        if len(keys) != len(set(keys)) or len(rows) != len(coordinates) * 2:
            raise ValueError("campaign matrix has missing or duplicate coordinates")
        input_artifacts = dict(input_artifacts or {})
        if "campaign-inputs.resolved.json" not in input_artifacts:
            raise ValueError("campaign publication requires sealed resolved inputs")
        input_artifacts.update(run_artifacts)
        input_artifacts["sc20686_coverage_manifest.json"] = COVERAGE.read_bytes()
        input_artifacts["sc20686_source_map.json"] = SOURCE_MAP.read_bytes()
        staging.mkdir(parents=False)
        sealed_rows, row_files = [], []
        artifact_payloads = dict(input_artifacts)
        for index, row in enumerate(rows):
            name = f"row-{index:02d}.json"
            sealed, raw, row_sidecar = seal(row, name)
            reducer.verify_seal_artifact(sealed, raw, row_sidecar, name)
            sealed_rows.append(sealed)
            row_files.append(name)
            artifact_payloads[name] = raw
        decision = reducer.reduce(sealed_rows)
        source_map = json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
        markdown = render_markdown(keys, decision, source_map)
        artifact_payloads["campaign.md"] = markdown
        for name, payload in artifact_payloads.items():
            if Path(name).name != name or not isinstance(payload, bytes):
                raise ValueError("campaign input artifact is malformed")
        artifact_hashes = {name: digest(payload) for name, payload in artifact_payloads.items()}
        for name, source in run_sources.items():
            if Path(name).name != name or not Path(source).is_file():
                raise ValueError("campaign media artifact is malformed")
            artifact_hashes[name] = file_identity(source)
        campaign = {
            "schema": "sc-20686-campaign-bundle-v6",
            "decision": decision,
            "artifact_sha256": artifact_hashes,
            "route_input_file_sha256": {
                f"{row['family']}/{row['variant']}/{row['coordinate_name']}/{row['arm']}":
                    row["input_file_sha256"]
                for row in sealed_rows
            },
            "row_files": row_files,
            "rows": sealed_rows,
        }
        campaign_raw = (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode("utf-8")
        for name, payload in artifact_payloads.items():
            (staging / name).write_bytes(payload)
            (staging / f"{name}.sha256").write_text(
                f"{artifact_hashes[name]}  {name}\n", encoding="utf-8"
            )
        for name, source in run_sources.items():
            if file_identity(source) != artifact_hashes[name]:
                raise ValueError(f"campaign media changed while sealing: {name}")
            shutil.copyfile(source, staging / name)
            if file_identity(staging / name) != artifact_hashes[name]:
                raise ValueError(f"campaign media copy checksum mismatch: {name}")
            (staging / f"{name}.sha256").write_text(
                f"{artifact_hashes[name]}  {name}\n", encoding="utf-8"
            )
        (staging / "campaign.json").write_bytes(campaign_raw)
        (staging / "campaign.json.sha256").write_text(
            f"{digest(campaign_raw)}  campaign.json\n", encoding="utf-8"
        )
        reducer.verify_campaign_bundle(staging)
        os.replace(staging, final)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        for run in captured_runs:
            cleanup_campaign_run(run)


def _publish_schedule_control(coordinates, summaries, run_artifacts, run_sources,
                              input_artifacts, staging, final, reducer):
    keys = [(summary["variant"], summary["coordinate_name"]) for summary in summaries]
    if len(keys) != len(set(keys)) or len(summaries) != len(coordinates):
        raise ValueError("schedule-control matrix has missing or duplicate coordinates")
    artifact_payloads = dict(input_artifacts or {})
    if "campaign-inputs.resolved.json" not in artifact_payloads:
        raise ValueError("schedule-control publication requires sealed resolved inputs")
    artifact_payloads.update(run_artifacts)
    artifact_payloads["sc20686_coverage_manifest.json"] = COVERAGE.read_bytes()
    artifact_payloads["sc20686_source_map.json"] = SOURCE_MAP.read_bytes()
    staging.mkdir(parents=False)
    artifact_hashes = {name: digest(payload) for name, payload in artifact_payloads.items()}
    for name, source in run_sources.items():
        artifact_hashes[name] = file_identity(source)
    campaign = {
        "schema": "sc-20686-schedule-control-v1",
        "backend": "mlx-metal",
        "artifact_sha256": artifact_hashes,
        "summaries": summaries,
    }
    campaign_raw = (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode("utf-8")
    for name, payload in artifact_payloads.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise ValueError("schedule-control artifact is malformed")
        (staging / name).write_bytes(payload)
        (staging / f"{name}.sha256").write_text(f"{artifact_hashes[name]}  {name}\n", encoding="utf-8")
    for name, source in run_sources.items():
        shutil.copyfile(source, staging / name)
        if file_identity(staging / name) != artifact_hashes[name]:
            raise ValueError(f"schedule-control media copy checksum mismatch: {name}")
        (staging / f"{name}.sha256").write_text(f"{artifact_hashes[name]}  {name}\n", encoding="utf-8")
    (staging / "campaign.json").write_bytes(campaign_raw)
    (staging / "campaign.json.sha256").write_text(
        f"{digest(campaign_raw)}  campaign.json\n", encoding="utf-8"
    )
    reducer.verify_schedule_control_bundle(staging)
    os.replace(staging, final)
    return campaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--family", choices=("flux2-klein", "wan"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--coordinate-name")
    parser.add_argument("--wan-manifest", type=Path)
    parser.add_argument("--flux-entrypoint", type=Path)
    parser.add_argument("--flux-snapshot", type=Path)
    parser.add_argument("--flux-reference", type=Path)
    parser.add_argument("--flux-reference2", type=Path)
    parser.add_argument("--flux-kv-snapshot", type=Path)
    parser.add_argument(
        "--schedule-control", action="store_true",
        help="Metal lane only: run each coordinate's schedule-control arm (phase windows, no "
             "per-read evaluation windows) and seal product-schedule peaks instead of decisions",
    )
    parser.add_argument("--matrix-output", type=Path)
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--inference-revision")
    parser.add_argument("--cancel-campaign", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--run-timeout-seconds", type=float, default=21600)
    parser.add_argument("--safety-policy", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, help="extra operator stop file checked between arms (<resume-dir>/STOP is always checked); "
                        f"its presence halts before the next arm with exit status {supervisor.OPERATOR_STOP_EXIT_CODE}")
    args = parser.parse_args()
    if not args.campaign:
        parser.error("SC-20686 adapter requires explicit --campaign")
    if args.fake:
        parser.error("synthetic evidence cannot enter the campaign adapter")
    stop_files = supervisor.operator_stop_files(args.stop_file, args.resume_dir)
    try:
        safety_policy = supervisor.load_policy(args.safety_policy)
        # darwin-mlx measures the MLX providers (the Mac product path); linux/windows-cuda measure
        # the Candle providers. The supervisor refuses a policy whose backend is not this host's.
        backend = LANE_BY_POLICY.get(safety_policy.backend)
        if backend is None:
            raise ValueError("SC-20686 media campaigns require a darwin-mlx, linux-cuda, or windows-cuda safety backend")
        if not args.resume_dir.is_absolute() or args.resume_dir.is_symlink():
            raise ValueError("resume directory must be an absolute, nonsymlink path")
        if args.run_timeout_seconds <= 0 or safety_policy.deadline_seconds > args.run_timeout_seconds:
            raise ValueError("run timeout must cover the mandatory safety deadline")
        if args.resume_dir.exists():
            captured = json.loads((args.resume_dir / "resolved.json").read_bytes())
            inference_revision = require_revision(captured["inference_revision"], "captured inference revision")
            if captured.get("backend") != backend:
                raise ValueError("resume directory was captured for a different measurement lane")
            if args.inference_revision is not None and args.inference_revision != inference_revision:
                raise ValueError("requested inference revision differs from captured resume provenance")
        else:
            inference_revision = verify_inference_revision(args.inference_revision)
        source_map_hash = digest(SOURCE_MAP.read_bytes())
        # estimate-plus-reserve-v1: each Metal-lane arm is admitted on its entrypoint's product
        # admission estimate (asked once per coordinate), raised to the measured peak of a
        # completed arm of the same coordinate request. The Candle lanes' entrypoints have no
        # estimate mode, so their arms fall back to the cap.
        estimates, completed_peaks = {}, {}

        def coordinate_estimate(spec):
            if backend != "mlx-metal":
                return None
            key = (spec.variant, spec.name)
            if key not in estimates:
                estimates[key] = product_admission_estimate(
                    spec.entrypoint, spec.snapshot, spec.variant, spec.args)
            return estimates[key]

        def record_unit(spec, _arm, run):
            peak = (run.supervision or {}).get("peakHostBytes")
            if type(peak) is int and peak > 0:
                key = (spec.variant, spec.name)
                completed_peaks[key] = max(peak, completed_peaks.get(key, 0))

        def estimated_run(spec, arm):
            return run_entrypoint(
                spec.entrypoint, spec.snapshot, spec.variant, arm,
                inference_revision, spec.residency_strategy, spec.args,
                args.run_timeout_seconds, safety_policy=safety_policy,
                failure_root=args.resume_dir / "failed",
                product_estimate=coordinate_estimate(spec),
                measured_peak_host_bytes=completed_peaks.get((spec.variant, spec.name)),
            )
        if args.schedule_control and (backend != "mlx-metal" or not args.matrix):
            raise ValueError("--schedule-control is a Metal-lane (darwin-mlx) matrix mode")
        if args.matrix:
            if not all((args.wan_manifest, args.flux_entrypoint, args.flux_snapshot, args.flux_reference, args.flux_reference2, args.matrix_output)):
                parser.error("matrix mode requires the Wan manifest, FLUX executable/snapshot, two references, and output")
            wan = load_wan_manifest(args.wan_manifest, backend)
            coordinates = [spec for route in WAN_ROUTES for spec in wan[route].values()]
            coordinates.extend(flux_coordinates(
                args.flux_entrypoint, args.flux_snapshot, args.flux_reference,
                args.flux_reference2, backend, args.flux_kv_snapshot,
            ))
            snapshot_identities = {
                str(snapshot): snapshot_identity(snapshot)
                for snapshot in sorted({spec.snapshot for spec in coordinates})
            }
            resolved = {
                "schema": "sc-20686-resolved-inputs-v4",
                "backend": backend,
                "inference_revision": inference_revision,
                "coordinates": [
                    {
                        "family": spec.family, "variant": spec.variant, "name": spec.name,
                        "entrypoint": str(spec.entrypoint), "entrypoint_sha256": file_identity(spec.entrypoint),
                        "snapshot": str(spec.snapshot), "snapshot_sha256": snapshot_identities[str(spec.snapshot)][0],
                        "snapshot_bytes": snapshot_identities[str(spec.snapshot)][1],
                        "model_snapshot_revision": spec.model_snapshot_revision,
                        "residency_strategy": spec.residency_strategy,
                        "args": list(spec.args),
                        "route_manifest_sha256": spec.route_manifest_sha256,
                        "input_files": list(spec.input_file_inventory),
                    }
                    for spec in coordinates
                ],
            }
            if args.schedule_control:
                resolved["mode"] = "schedule-control"
            resume_identity_sha = _prepare_resume(args.resume_dir, resolved, safety_policy, stop_files)

            runner = estimated_run

            def preflight(spec):
                verify_coordinate_inputs(spec)
                verify_snapshot_identity(spec, snapshot_identities[str(spec.snapshot)])

            def build_row(spec, arm, events, evidence_hashes):
                snapshot_hash, snapshot_bytes = snapshot_identities[str(spec.snapshot)]
                if arm == "control":
                    return _load_reducer().schedule_control_summary(
                        events, spec.variant, spec.name,
                        lane_coverage(backend)[spec.family][spec.variant]["coordinates"][spec.name],
                        {
                            "source_ref": inference_revision,
                            "model_snapshot_revision": spec.model_snapshot_revision,
                            "residency_strategy": spec.residency_strategy,
                            "snapshot_sha256": snapshot_hash,
                            "snapshot_bytes": snapshot_bytes,
                        },
                    )
                row_args = argparse.Namespace(
                    fake=False, family=spec.family, variant=spec.variant,
                    coordinate_name=spec.name, cancel_campaign=arm == "cancel",
                    backend=backend,
                )
                return make_row(row_args, {
                    "route_manifest_sha256": spec.route_manifest_sha256,
                    "source_map_sha256": source_map_hash,
                    "inference_revision": inference_revision,
                    "model_snapshot_revision": spec.model_snapshot_revision,
                    "residency_strategy": spec.residency_strategy,
                    "input_file_sha256": spec.input_file_sha256,
                    "evidence_artifact_sha256": evidence_hashes,
                }, snapshot_hash, snapshot_bytes, events)

            publish_campaign(coordinates, runner, build_row, args.matrix_output, {
                "wan-manifest.source.json": args.wan_manifest.read_bytes(),
                "campaign-inputs.resolved.json": (json.dumps(resolved, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                "safety-policy.json": safety_policy.canonical_bytes,
                "resume-identity.json": (args.resume_dir / "identity.json").read_bytes(),
            }, resume_root=args.resume_dir, resume_identity_sha=resume_identity_sha,
                preflight=preflight, stop_files=stop_files,
                schedule_control=args.schedule_control, on_unit=record_unit)
            return 0

        if not all((args.family, args.snapshot, args.output, args.variant, args.coordinate_name, args.entrypoint)):
            parser.error("single mode requires family, route, coordinate, snapshot, entrypoint, and output")
        route_args = ()
        if args.family == "flux2-klein":
            if not args.flux_reference:
                parser.error("single FLUX campaign requires --flux-reference")
            route_args = ("--reference", str(args.flux_reference.resolve()), "--single-only")
        file_hashes, inventory = hash_file_arguments(route_args)
        if args.variant not in PRODUCT_RESIDENCY:
            parser.error("single mode variant is not a frozen product route")
        route_entry = lane_entry(backend, args.variant)
        if route_entry["family"] != args.family:
            raise ValueError("single mode family differs from the registered lane route")
        if Path(args.entrypoint).stem != route_entry["entrypoint_stem"]:
            raise ValueError("single mode entrypoint does not match the registered lane route")
        validate_snapshot_layout(args.snapshot, "single campaign", args.variant, backend)
        revision = model_snapshot_revision(args.snapshot)
        residency_strategy = PRODUCT_RESIDENCY[args.variant]
        spec = CoordinateSpec(
            args.family, args.variant, args.coordinate_name, args.entrypoint.resolve(),
            args.snapshot.resolve(), revision, residency_strategy, route_args,
            route_manifest_identity(
                args.variant, args.coordinate_name, args.entrypoint, args.snapshot,
                revision, residency_strategy, route_args, inventory,
            ),
            file_hashes, inventory,
        )
        snapshot_hash, snapshot_bytes = snapshot_identity(spec.snapshot)

        runner = estimated_run

        def preflight(single_spec):
            verify_coordinate_inputs(single_spec)
            verify_snapshot_identity(single_spec, (snapshot_hash, snapshot_bytes))

        def build_row(single_spec, arm, events, evidence_hashes):
            row_args = argparse.Namespace(
                fake=False, family=single_spec.family, variant=single_spec.variant,
                coordinate_name=single_spec.name, cancel_campaign=arm == "cancel",
                backend=backend,
            )
            return make_row(row_args, {
                "route_manifest_sha256": single_spec.route_manifest_sha256,
                "source_map_sha256": source_map_hash,
                "inference_revision": inference_revision,
                "model_snapshot_revision": single_spec.model_snapshot_revision,
                "residency_strategy": single_spec.residency_strategy,
                "input_file_sha256": single_spec.input_file_sha256,
                "evidence_artifact_sha256": evidence_hashes,
            }, snapshot_hash, snapshot_bytes, events)

        # A single invocation still captures both lifecycle arms into one atomic, sealed bundle.
        resolved = {
            "schema": "sc-20686-resolved-inputs-v4",
            "backend": backend,
            "inference_revision": inference_revision,
            "coordinates": [{
                "family": spec.family, "variant": spec.variant, "name": spec.name,
                "entrypoint": str(spec.entrypoint),
                "entrypoint_sha256": file_identity(spec.entrypoint),
                "snapshot": str(spec.snapshot), "snapshot_sha256": snapshot_hash,
                "snapshot_bytes": snapshot_bytes,
                "model_snapshot_revision": spec.model_snapshot_revision,
                "residency_strategy": spec.residency_strategy,
                "args": list(spec.args),
                "route_manifest_sha256": spec.route_manifest_sha256,
                "input_files": list(spec.input_file_inventory),
            }],
        }
        resume_identity_sha = _prepare_resume(args.resume_dir, resolved, safety_policy, stop_files)
        publish_campaign([spec], runner, build_row, args.output, {
            "campaign-inputs.resolved.json": (
                json.dumps(resolved, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
            "safety-policy.json": safety_policy.canonical_bytes,
            "resume-identity.json": (args.resume_dir / "identity.json").read_bytes(),
        }, resume_root=args.resume_dir, resume_identity_sha=resume_identity_sha,
            preflight=preflight, stop_files=stop_files, on_unit=record_unit)
        return 0
    except supervisor.OperatorStop as stop:
        print(f"SC-20686 adapter {stop}", file=sys.stderr)
        print(json.dumps({
            "status": "stopped-by-operator", "beforeRow": stop.index,
            "beforeRowSlug": stop.before, "record": str(stop.record),
        }, sort_keys=True))
        return supervisor.OPERATOR_STOP_EXIT_CODE
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SC-20686 adapter refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Product-entrypoint campaign producer for sealed SC-20686 evidence bundles."""

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
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REDUCER = Path(__file__).with_name("sc20686_cache_attribution.py")
COVERAGE = Path(__file__).with_name("sc20686_coverage_manifest.json")
SOURCE_MAP = Path(__file__).with_name("sc20686_source_map.json")
PRODUCER = "sc20686-campaign-adapter-v2"
GEOMETRY = (
    "resolution", "reference_count", "frames", "prompt", "guidance", "layers",
    "heads", "head_dimension", "sq", "skv", "dtype", "mask", "rope",
)
WAN_ROUTES = (
    "wan2_2_ti2v_5b", "wan2_2_t2v_14b", "wan2_2_i2v_14b", "wan_vace",
    "wan2_2_vace_fun_14b",
)
FLUX_ROUTES = ("flux2_klein_9b_edit",)
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
    "--sc20686-route", "--sc20686-cancel",
}


@dataclass(frozen=True)
class CampaignRun:
    events: list
    stdout: bytes
    stderr: bytes
    command: tuple
    process_samples: tuple


@dataclass(frozen=True)
class CoordinateSpec:
    family: str
    variant: str
    name: str
    entrypoint: Path
    snapshot: Path
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
    root = Path(root).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
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


def route_manifest_identity(route, coordinate, entrypoint, snapshot, arguments, inventory):
    document = {
        "route": route,
        "coordinate": coordinate,
        "entrypoint": str(Path(entrypoint).resolve()),
        "entrypoint_sha256": file_identity(entrypoint),
        "snapshot": str(Path(snapshot).resolve()),
        "args": list(map(str, arguments)),
        "input_files": list(inventory),
    }
    return digest(canonical(document))


def verify_coordinate_inputs(spec):
    for item in spec.input_file_inventory:
        item_hash, size, kind = path_identity(item["path"])
        if (item_hash, size, kind) != (item["sha256"], item["bytes"], item["kind"]):
            raise ValueError(
                f"route input changed after resolution: {spec.variant}/{spec.name}/{item['flag']}"
            )
    current_manifest = route_manifest_identity(
        spec.variant, spec.name, spec.entrypoint, spec.snapshot, spec.args,
        spec.input_file_inventory,
    )
    if current_manifest != spec.route_manifest_sha256:
        raise ValueError(f"route executable or manifest changed after resolution: {spec.variant}/{spec.name}")


def load_coverage():
    document = json.loads(COVERAGE.read_text(encoding="utf-8"))
    if document.get("schema") != "sc-20686-supported-coverage-v2":
        raise ValueError("checked-in coverage manifest is invalid")
    return document


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


def load_wan_manifest(path):
    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Wan campaign manifest is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != set(WAN_ROUTES):
        raise ValueError("Wan campaign manifest must contain exactly the five registered routes")
    coverage = load_coverage()["families"]["wan"]
    result = {}
    for route in WAN_ROUTES:
        entry = document[route]
        if (
            not isinstance(entry, dict)
            or entry.get("provider_id") != route
            or not isinstance(entry.get("entrypoint"), str)
            or not isinstance(entry.get("snapshot"), str)
            or not isinstance(entry.get("coordinates"), dict)
        ):
            raise ValueError(f"Wan manifest entry is malformed: {route}")
        binary = Path(expand_argument(entry["entrypoint"])).expanduser().resolve()
        snapshot = Path(expand_argument(entry["snapshot"])).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Wan manifest entrypoint is not executable: {route}")
        if binary.stem != WAN_ENTRYPOINT_STEMS[route]:
            raise ValueError(f"Wan manifest entrypoint does not match registered route: {route}")
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise ValueError(f"Wan manifest snapshot is missing config.json: {route}")
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
            file_hashes, inventory = hash_file_arguments(arguments)
            identity = route_manifest_identity(route, name, binary, snapshot, arguments, inventory)
            route_specs[name] = CoordinateSpec(
                "wan", route, name, binary, snapshot, arguments, identity,
                file_hashes, inventory,
            )
        result[route] = route_specs
    return result


def flux_coordinates(entrypoint, snapshot, reference, reference2):
    entrypoint = Path(entrypoint).resolve()
    snapshot = Path(snapshot).resolve()
    reference = Path(reference).resolve()
    reference2 = Path(reference2).resolve()
    if not entrypoint.is_file() or not os.access(entrypoint, os.X_OK):
        raise ValueError("FLUX campaign entrypoint is not executable")
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise ValueError("FLUX campaign snapshot is missing config.json")
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
    result = []
    expected_coordinates = load_coverage()["families"]["flux2-klein"][FLUX_ROUTES[0]]["coordinates"]
    if set(definitions) != set(expected_coordinates):
        raise ValueError("FLUX coordinate definitions do not match frozen coverage")
    for name, arguments in definitions.items():
        width = int(argument_value(arguments, "--width"))
        height = int(argument_value(arguments, "--height"))
        prompt = argument_value(arguments, "--prompt")
        expected = expected_coordinates[name]
        actual = {
            "resolution": f"{width}x{height}", "frames": 1,
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
            "flux2-klein", FLUX_ROUTES[0], name, entrypoint, snapshot, arguments,
            route_manifest_identity(FLUX_ROUTES[0], name, entrypoint, snapshot, arguments, inventory),
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
    for key in ("reference_count", "frames", "layers", "heads", "head_dimension", "sq", "skv"):
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


def run_entrypoint(entrypoint, snapshot, variant, arm, extra_args=(), timeout_seconds=21600):
    command = [
        str(Path(entrypoint).resolve()), "--sc20686-campaign", "--sc20686-events", "-",
        "--snapshot", str(Path(snapshot).resolve()), "--variant", variant, *map(str, extra_args),
    ]
    if arm == "cancel":
        command.append("--sc20686-cancel")
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_chunks, stderr_chunks = [], []
    stdout_thread = threading.Thread(target=_drain, args=(child.stdout, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(child.stderr, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    process_samples = []
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while child.poll() is None:
        rss = _sample_rss(child.pid)
        if rss:
            process_samples.append({
                "phase": "process-sample", "sample_kind": "process", "peak_bytes": rss,
                "at_ns": time.time_ns(),
            })
        if time.monotonic() >= deadline:
            timed_out = True
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(0.05)
    child.wait()
    stdout_thread.join(timeout=10)
    stderr_thread.join(timeout=10)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        child.kill()
        raise ValueError(f"{variant}/{arm} transcript drain did not terminate")
    stdout = b"".join(stdout_chunks)
    stderr = b"".join(stderr_chunks)
    if timed_out:
        raise ValueError(f"{variant}/{arm} entrypoint timed out after {timeout_seconds}s")
    if child.returncode:
        message = stderr.decode("utf-8", errors="replace")[-4096:].strip()
        raise ValueError(f"{variant}/{arm} entrypoint failed: {message}")
    try:
        events = [
            json.loads(line)
            for line in stdout.decode("utf-8").splitlines()
            if line.lstrip().startswith("{")
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{variant}/{arm} emitted an invalid observer transcript") from exc
    events.extend(process_samples)
    if not events or not process_samples:
        raise ValueError(f"{variant}/{arm} lacks observer or process evidence")
    return CampaignRun(events, stdout, stderr, tuple(command), tuple(process_samples))


def make_row(args, config, snapshot_hash, snapshot_bytes, events):
    if args.fake:
        raise ValueError("fake evidence is test-only and cannot produce a receipt")
    metadata_events = [event for event in events if event.get("phase") == "metadata"]
    metrics_events = [event for event in events if event.get("phase") == "metrics"]
    if len(metadata_events) != 1 or len(metrics_events) != 1:
        raise ValueError("entrypoint must emit exactly one metadata and metrics event")
    metadata, metrics = metadata_events[0], metrics_events[0]
    if metadata.get("snapshot_sha256") != snapshot_hash or metadata.get("snapshot_bytes") != snapshot_bytes:
        raise ValueError("entrypoint model identity does not match the hashed snapshot")
    if not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("source_ref", ""))):
        raise ValueError("entrypoint must emit an immutable lowercase source ref")
    if metadata.get("variant") != args.variant:
        raise ValueError("producer variant does not match the exact product route")
    for field in ("route_manifest_sha256", "source_map_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(config.get(field, ""))):
            raise ValueError(f"campaign {field} is missing")
    metric_keys = (
        "current_persistent_bytes", "current_read_transient_bytes",
        "candidate_persistent_bytes", "candidate_read_transient_bytes",
        "generation_duration_ms", "cache_read_duration_ms", "reused_requests",
        "minimum_cache_reads",
    )
    if any(
        not isinstance(metrics.get(key), (int, float))
        or isinstance(metrics.get(key), bool)
        or not math.isfinite(metrics[key])
        or metrics[key] < 0
        for key in metric_keys
    ):
        raise ValueError("entrypoint metrics are incomplete")
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
    if metrics["candidate_read_transient_bytes"] != observed_transient:
        raise ValueError("candidate read workspace must conservatively retain the measured dense read")
    if metrics["reused_requests"] != observed_reuse:
        raise ValueError("reuse metric differs from product-owned read events")
    create_events = [event for event in events if event.get("phase") == "cross-kv-created"]
    if args.family == "flux2-klein":
        if metrics["current_persistent_bytes"] != 0 or any(event.get("persistent_bytes") != 0 for event in create_events):
            raise ValueError("FLUX edit must report its non-persistent route honestly")
        if metrics["minimum_cache_reads"] != observed_reuse:
            raise ValueError("FLUX logical payload reuse differs from product-owned recomputations")
    else:
        if metrics["current_persistent_bytes"] == 0:
            raise ValueError("Wan persistent-cache route cannot claim zero persistence")
        created_ids = [event.get("cache_id") for event in create_events]
        if any(not isinstance(cache_id, int) or cache_id < 1 for cache_id in created_ids) or len(created_ids) != len(set(created_ids)):
            raise ValueError("Wan cache creation identities are missing or duplicated")
        read_ids = [event.get("cache_id") for event in read_events]
        if any(cache_id not in set(created_ids) for cache_id in read_ids):
            raise ValueError("Wan cache read does not identify its created cache")
        release_events = [
            event for event in events if event.get("phase") == "cross-kv-released"
        ]
        released_ids = [event.get("cache_id") for event in release_events]
        if sorted(released_ids) != sorted(created_ids):
            raise ValueError("Wan cache invalidation does not release every exact cache once")
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
                raise ValueError("Wan cache read/release ordering is not product-owned")
        per_cache_reads = {cache_id: read_ids.count(cache_id) for cache_id in created_ids}
        if metrics["minimum_cache_reads"] != min(per_cache_reads.values(), default=0):
            raise ValueError("Wan minimum cache reuse is not the per-cache minimum")
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
                    raise ValueError("Wan creation lacks exact dense/candidate retained bytes")
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
                    raise ValueError("Wan release bytes differ from the exact created cache")
                live_dense -= dense
                live_candidate -= candidate
        if live_dense or live_candidate:
            raise ValueError("Wan retained accounting does not close at release")
        if (
            metrics["current_persistent_bytes"] != peak_dense
            or metrics["candidate_persistent_bytes"] != peak_candidate
        ):
            raise ValueError("Wan persistent metric differs from exact simultaneous residency")
        for event in read_events:
            before = event.get("allocator_before_bytes")
            after = event.get("allocator_after_bytes")
            if (
                not isinstance(before, int) or isinstance(before, bool) or before <= 0
                or not isinstance(after, int) or isinstance(after, bool) or after <= 0
                or event.get("transient_bytes") != max(0, after - before)
            ):
                raise ValueError("Wan read transient lacks exact allocator before/after evidence")
    geometry = geometry_from(events)
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
        "family": args.family,
        "variant": args.variant,
        "coordinate_name": args.coordinate_name,
        "coordinate_id": coordinate_id,
        "arm": "cancel" if args.cancel_campaign else "normal",
        "source_ref": metadata["source_ref"],
        "route_manifest_sha256": config["route_manifest_sha256"],
        "source_map_sha256": config["source_map_sha256"],
        "model_snapshot_sha256": snapshot_hash,
        "model_snapshot_bytes": snapshot_bytes,
        "input_file_sha256": dict(config.get("input_file_sha256", {})),
        "evidence_artifact_sha256": dict(config.get("evidence_artifact_sha256", {})),
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
        **{key: metrics[key] for key in metric_keys},
        "observer_events": events,
    }


def _load_reducer():
    spec = importlib.util.spec_from_file_location("sc20686_reducer", REDUCER)
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    return reducer


def render_markdown(keys, decision, source_map):
    lines = ["# SC-20686 campaign", "", "## Frozen coordinates", ""]
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
    for variant, entry in source_map["variants"].items():
        lines.extend([
            f"### `{variant}`", "",
            f"- Product route: `{entry['route']}`", f"- Current kernel: {entry['current_kernel']}",
            f"- Cache format: {entry['cache_format']}", f"- Paging: {entry['paging']}",
            f"- Offload: {entry['offload']}", f"- Recompute: {entry['recompute']}",
            f"- Compatibility: {entry['compatibility']}",
        ])
        for kind in ("activation", "creation", "reads", "release"):
            anchors = ", ".join(f"`{item['path']}:{item['line']}#{item['symbol']}`" for item in entry[kind])
            lines.append(f"- {kind.title()} anchors: {anchors}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def publish_campaign(coordinates, runner, row_builder, destination, input_artifacts=None):
    rows = []
    run_artifacts = {}
    for index, coordinate in enumerate(coordinates):
        for arm in ("normal", "cancel"):
            run = runner(coordinate, arm)
            if not isinstance(run, CampaignRun):
                raise ValueError("campaign runner must return a sealed transcript capture")
            stem = f"run-{index:02d}-{arm}"
            command_payload = canonical({"argv": list(run.command)})
            process_payload = canonical({"samples": list(run.process_samples)})
            artifacts = {
                f"{stem}.stdout": run.stdout,
                f"{stem}.stderr": run.stderr,
                f"{stem}.command.json": command_payload,
                f"{stem}.process.json": process_payload,
            }
            evidence_hashes = {name: digest(payload) for name, payload in artifacts.items()}
            row = row_builder(coordinate, arm, run.events, evidence_hashes)
            if not isinstance(row, dict):
                raise ValueError("row builder returned no receipt row")
            rows.append(row)
            run_artifacts.update(artifacts)
    keys = [
        (row.get("family"), row.get("variant"), row.get("coordinate_name"), row.get("arm"))
        for row in rows
    ]
    if len(keys) != len(set(keys)) or len(rows) != len(coordinates) * 2:
        raise ValueError("campaign matrix has missing or duplicate coordinates")
    final = Path(destination)
    staging = final.with_name(f".{final.name}.staging-{os.getpid()}")
    if final.exists() or staging.exists():
        raise ValueError("campaign destination or staging path already exists")
    reducer = _load_reducer()
    input_artifacts = dict(input_artifacts or {})
    input_artifacts.update(run_artifacts)
    input_artifacts["sc20686_coverage_manifest.json"] = COVERAGE.read_bytes()
    input_artifacts["sc20686_source_map.json"] = SOURCE_MAP.read_bytes()
    try:
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
        campaign = {
            "schema": "sc-20686-campaign-bundle-v3",
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
    parser.add_argument("--matrix-output", type=Path)
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--cancel-campaign", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--run-timeout-seconds", type=float, default=21600)
    args = parser.parse_args()
    if not args.campaign:
        parser.error("SC-20686 adapter requires explicit --campaign")
    if args.fake:
        parser.error("synthetic evidence cannot enter the campaign adapter")
    try:
        source_map_hash = digest(SOURCE_MAP.read_bytes())
        if args.matrix:
            if not all((args.wan_manifest, args.flux_entrypoint, args.flux_snapshot, args.flux_reference, args.flux_reference2, args.matrix_output)):
                parser.error("matrix mode requires the Wan manifest, FLUX executable/snapshot, two references, and output")
            wan = load_wan_manifest(args.wan_manifest)
            coordinates = [spec for route in WAN_ROUTES for spec in wan[route].values()]
            coordinates.extend(flux_coordinates(args.flux_entrypoint, args.flux_snapshot, args.flux_reference, args.flux_reference2))
            snapshot_identities = {
                str(snapshot): snapshot_identity(snapshot)
                for snapshot in sorted({spec.snapshot for spec in coordinates})
            }
            resolved = {
                "schema": "sc-20686-resolved-inputs-v2",
                "coordinates": [
                    {
                        "family": spec.family, "variant": spec.variant, "name": spec.name,
                        "entrypoint": str(spec.entrypoint), "entrypoint_sha256": file_identity(spec.entrypoint),
                        "snapshot": str(spec.snapshot), "snapshot_sha256": snapshot_identities[str(spec.snapshot)][0],
                        "snapshot_bytes": snapshot_identities[str(spec.snapshot)][1], "args": list(spec.args),
                        "route_manifest_sha256": spec.route_manifest_sha256,
                        "input_files": list(spec.input_file_inventory),
                    }
                    for spec in coordinates
                ],
            }

            def runner(spec, arm):
                verify_coordinate_inputs(spec)
                run = run_entrypoint(spec.entrypoint, spec.snapshot, spec.variant, arm, spec.args, args.run_timeout_seconds)
                verify_coordinate_inputs(spec)
                return run

            def build_row(spec, arm, events, evidence_hashes):
                snapshot_hash, snapshot_bytes = snapshot_identities[str(spec.snapshot)]
                row_args = argparse.Namespace(
                    fake=False, family=spec.family, variant=spec.variant,
                    coordinate_name=spec.name, cancel_campaign=arm == "cancel",
                )
                return make_row(row_args, {
                    "route_manifest_sha256": spec.route_manifest_sha256,
                    "source_map_sha256": source_map_hash,
                    "input_file_sha256": spec.input_file_sha256,
                    "evidence_artifact_sha256": evidence_hashes,
                }, snapshot_hash, snapshot_bytes, events)

            publish_campaign(coordinates, runner, build_row, args.matrix_output, {
                "wan-manifest.source.json": args.wan_manifest.read_bytes(),
                "campaign-inputs.resolved.json": (json.dumps(resolved, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            })
            return 0

        if not all((args.family, args.snapshot, args.output, args.variant, args.coordinate_name, args.entrypoint)):
            parser.error("single mode requires family, route, coordinate, snapshot, entrypoint, and output")
        route_args = ()
        if args.family == "flux2-klein":
            if not args.flux_reference:
                parser.error("single FLUX campaign requires --flux-reference")
            route_args = ("--reference", str(args.flux_reference.resolve()), "--single-only")
        file_hashes, inventory = hash_file_arguments(route_args)
        spec = CoordinateSpec(
            args.family, args.variant, args.coordinate_name, args.entrypoint.resolve(),
            args.snapshot.resolve(), route_args,
            route_manifest_identity(args.variant, args.coordinate_name, args.entrypoint, args.snapshot, route_args, inventory),
            file_hashes, inventory,
        )
        snapshot_hash, snapshot_bytes = snapshot_identity(spec.snapshot)

        def runner(single_spec, arm):
            verify_coordinate_inputs(single_spec)
            run = run_entrypoint(single_spec.entrypoint, single_spec.snapshot, single_spec.variant, arm, single_spec.args, args.run_timeout_seconds)
            verify_coordinate_inputs(single_spec)
            return run

        def build_row(single_spec, arm, events, evidence_hashes):
            row_args = argparse.Namespace(
                fake=False, family=single_spec.family, variant=single_spec.variant,
                coordinate_name=single_spec.name, cancel_campaign=arm == "cancel",
            )
            return make_row(row_args, {
                "route_manifest_sha256": single_spec.route_manifest_sha256,
                "source_map_sha256": source_map_hash,
                "input_file_sha256": single_spec.input_file_sha256,
                "evidence_artifact_sha256": evidence_hashes,
            }, snapshot_hash, snapshot_bytes, events)

        # A single invocation still captures both lifecycle arms into one atomic, sealed bundle.
        publish_campaign([spec], runner, build_row, args.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SC-20686 adapter refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

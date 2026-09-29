#!/usr/bin/env python3
"""Fail-closed SC-20686 reducer for sealed product-owned campaign evidence."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

MANIFEST = Path(__file__).with_name("sc20686_coverage_manifest.json")
SOURCE_MAP = Path(__file__).with_name("sc20686_source_map.json")
FAMILIES = ("flux2-klein", "wan")
PRODUCER = "sc20686-campaign-adapter-v5"
THRESHOLDS = {
    "opportunity_bytes": 512 * 1024**2,
    "opportunity_peak_pct": 0.05,
    "minimum_reads_per_cache": 2,
    "saving_bytes": 256 * 1024**2,
    "saving_peak_pct": 0.03,
    "runtime_only_pct": 0.05,
}
GEOMETRY = (
    "batch", "resolution", "reference_count", "frames", "prompt", "guidance", "layers",
    "heads", "head_dimension", "sq", "skv", "dtype", "mask", "rope",
)
REQUIRED = (
    "producer", "backend", "family", "variant", "coordinate_name", "coordinate_id", "arm",
    "source_ref", "model_snapshot_revision", "residency_strategy",
    "route_manifest_sha256", "source_map_sha256",
    "model_snapshot_sha256", "model_snapshot_bytes", "input_file_sha256",
    "evidence_artifact_sha256", "media_manifest_sha256", "geometry", "lifecycle", "allocator_samples",
    "process_samples", "observer_events", "raw_receipt_sha256", "raw_receipt_sidecar_sha256",
    "real_weights", "full_generation", "attention_kind", "current_persistent_bytes",
    "current_read_transient_bytes", "candidate_persistent_bytes",
    "candidate_read_transient_bytes", "generation_duration_ms",
    "cache_read_duration_ms", "joint_attention_context_duration_ms",
    "reference_runtime_attribution_available", "reused_requests", "minimum_cache_reads",
)
PRODUCT_RESIDENCY = {
    "flux2_klein_9b_edit": "sequential",
    "flux2_klein_9b_kv_edit": "sequential",
    "wan2_2_ti2v_5b": "sequential",
    "wan2_2_t2v_14b": "sequential",
    "wan2_2_i2v_14b": "sequential",
    "wan_vace": "resident",
    "wan2_2_vace_fun_14b": "sequential",
}


# Measurement lanes. The CUDA lane measures the Candle providers; the Metal lane measures the MLX
# providers SceneWorks runs on a Mac (the Mac product path). One sealed campaign is one lane.
LANES = ("candle-cuda", "mlx-metal")
CACHE_KINDS = ("persistent", "recomputed")
PHASE_WINDOWS = ("encode", "load", "prepare-cache", "denoise-step", "post-denoise", "decode")
FLUX_IMAGE_ROUTES = ("flux2_klein_9b_edit", "flux2_klein_9b_kv_edit")
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


def fail(message):
    raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require_digest(value, name, length=64):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(byte not in "0123456789abcdef" for byte in value)
    ):
        fail(f"invalid {name}")


def require_nonnegative(value, name):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        fail(f"invalid {name}")


def packed_group32_kv_bytes(batch, heads, tokens, width):
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (
        batch, heads, tokens, width,
    )):
        fail("invalid packed K/V geometry")
    complete, pending = divmod(tokens, 32)
    rows = batch * heads
    key_codes = rows * complete * ((32 * width + 3) // 4)
    key_metadata = rows * complete * width * 4
    key_pending = rows * pending * width * 4
    value_rows = rows * tokens
    value_codes = value_rows * ((width + 3) // 4)
    value_metadata = value_rows * ((width + 31) // 32) * 4
    return key_codes + key_metadata + key_pending + value_codes + value_metadata


def active_allocator_measurement(event, label):
    fields = (
        "allocator_before_bytes", "allocator_after_bytes", "allocator_high_bytes",
        "allocator_reserved_bytes",
    )
    values = {field: event.get(field) for field in fields}
    reserved_high = event.get("peak_bytes")
    if event.get("allocator_measurement_available") is not True or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (*values.values(), reserved_high)
    ):
        fail(f"{label} lacks active allocator high-water evidence")
    if (
        values["allocator_high_bytes"] < values["allocator_before_bytes"]
        or values["allocator_high_bytes"] < values["allocator_after_bytes"]
        or values["allocator_reserved_bytes"] < values["allocator_after_bytes"]
        or reserved_high < values["allocator_reserved_bytes"]
        or reserved_high < values["allocator_high_bytes"]
    ):
        fail(f"{label} allocator high-water/remnant ordering is invalid")
    values["reserved_high_bytes"] = reserved_high
    return values


def dense_reference_kv_bytes(geometry, kv_batch=None):
    dtype_bytes = DTYPE_BYTES.get(geometry["dtype"].upper())
    if dtype_bytes is None:
        fail("FLUX reference slice has unsupported live dtype")
    batch = geometry["batch"] if kv_batch is None else kv_batch
    return (
        2 * batch * geometry["heads"] * geometry["skv"]
        * geometry["head_dimension"] * dtype_bytes
    )


def lane_variants(source_map, backend):
    lanes = source_map.get("lanes")
    if not isinstance(lanes, dict) or backend not in lanes:
        fail(f"source map has no {backend} lane")
    variants = lanes[backend].get("variants")
    if not isinstance(variants, dict):
        fail(f"source map {backend} lane is malformed")
    return variants


def lane_entry(source_map, backend, variant):
    entry = lane_variants(source_map, backend).get(variant)
    if not isinstance(entry, dict):
        fail(f"{variant} is not a registered {backend} route")
    return entry


CFG_KV_BATCH_RULES = ("never", "guidance", "always")


def expected_kv_batch(entry, geometry):
    """The exact K/V batch a Metal-lane cache carries. MLX Wan stacks the CFG cond/uncond contexts
    on one cache's batch axis: TI2V-5B only when guidance > 1 (`guidance`), the A14B experts always
    (`always`). FLUX.2 runs its branches as separate B=1 forwards and VACE recomputes per branch
    (`never`)."""
    rule = entry.get("cfg_kv_batch")
    if rule not in CFG_KV_BATCH_RULES:
        fail("Metal lane route lacks its CFG K/V batch rule")
    stacked = rule == "always" or (rule == "guidance" and float(geometry["guidance"]) > 1.0)
    return geometry["batch"] * (2 if stacked else 1)


def lane_coverage(coverage, backend):
    """The frozen coverage of one lane: the shared families plus that lane's extensions."""
    families = {
        family: {variant: dict(spec) for variant, spec in variants.items()}
        for family, variants in coverage["families"].items()
    }
    for family, variants in coverage.get("lane_extensions", {}).get(backend, {}).items():
        for variant, spec in variants.items():
            if variant in families.setdefault(family, {}):
                fail(f"lane extension redefines a shared route: {backend}/{variant}")
            families[family][variant] = spec
    return families


def validate_phase_windows(events, terminal_index, arm):
    """Metal-lane per-phase attribution: every window carries a valid allocator window and Darwin
    phys_footprint, precedes the terminal event, and the run covers denoise (+ decode if normal)."""
    windows = [
        (index, event) for index, event in enumerate(events)
        if isinstance(event, dict) and event.get("phase") == "phase-window"
    ]
    names = [event.get("window") for _index, event in windows]
    if not windows or any(name not in PHASE_WINDOWS for name in names):
        fail("Metal phase-window attribution is missing or unnamed")
    if "denoise-step" not in names or (arm == "normal" and "decode" not in names):
        fail("Metal phase-window attribution does not cover denoise and decode")
    for index, event in windows:
        if index > terminal_index:
            fail("Metal phase window closes after the terminal event")
        active_allocator_measurement(event, "Metal phase window")
        current = event.get("phys_footprint_bytes")
        peak = event.get("phys_footprint_peak_bytes")
        if (
            not isinstance(current, int) or isinstance(current, bool) or current <= 0
            or not isinstance(peak, int) or isinstance(peak, bool) or peak < current
        ):
            fail("Metal phase window lacks Darwin phys_footprint evidence")


def read_checked_json(path, expected_schema):
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"checked-in artifact unavailable: {path.name}: {exc}")
    if not isinstance(document, dict) or document.get("schema") != expected_schema:
        fail(f"invalid checked-in artifact: {path.name}")
    return document, raw


def checked_contracts():
    coverage, coverage_raw = read_checked_json(MANIFEST, "sc-20686-supported-coverage-v3")
    source_map, source_map_raw = read_checked_json(SOURCE_MAP, "sc-20686-source-map-v3")
    if source_map.get("compatibility_target") != {
        "story": "SC-20675",
        "format": "packed-group-affine-v2",
        "bits": 2,
        "group_size": 32,
        "metadata": "f16 scale and zero per completed group",
        "pending_key_tail": "dense f32",
    }:
        fail("source map compatibility target differs from the frozen SC-20675 format")
    if set(coverage.get("lane_extensions", {})) - set(LANES):
        fail("coverage names an unknown measurement lane")
    if set(source_map.get("lanes", {})) != set(LANES):
        fail("source map must describe exactly the Candle/CUDA and MLX/Metal lanes")
    for backend in LANES:
        expected = {
            variant: family_name
            for family_name, family in lane_coverage(coverage, backend).items()
            for variant in family
        }
        variants = lane_variants(source_map, backend)
        if set(variants) != set(expected):
            fail(f"source map does not close over the frozen {backend} variant surface")
        for variant, entry in variants.items():
            if (
                entry.get("route") != variant
                or entry.get("family") != expected[variant]
                or entry.get("product_residency") != PRODUCT_RESIDENCY.get(variant)
                or entry.get("self_attention_excluded") is not True
                or entry.get("cache_kind") not in CACHE_KINDS
                or not isinstance(entry.get("runtime_attribution"), bool)
                or not isinstance(entry.get("entrypoint_stem"), str)
                or not entry["entrypoint_stem"]
            ):
                fail(f"source map route is malformed: {backend}/{variant}")
            operations = entry.get("operations")
            if operations is not None and (
                not isinstance(operations, dict)
                or set(operations) != {"create", "read"}
                or any(not isinstance(value, str) or not value for value in operations.values())
            ):
                fail(f"source map operations are malformed: {backend}/{variant}")
            if entry["cache_kind"] == "recomputed" and operations is None:
                fail(f"recomputed route lacks exact operations: {backend}/{variant}")
            if (entry.get("cfg_kv_batch") in CFG_KV_BATCH_RULES) != (backend == "mlx-metal"):
                fail(f"CFG K/V batch rule is Metal-lane only: {backend}/{variant}")
            anchor_fields = ["activation", "creation", "reads", "release"]
            if entry["family"] == "flux2-klein":
                anchor_fields.append("reference_slice")
            for field in anchor_fields:
                anchors = entry.get(field)
                if not isinstance(anchors, list) or not anchors:
                    fail(f"source map lacks {field} anchors: {backend}/{variant}")
                for anchor in anchors:
                    if (
                        not isinstance(anchor, dict)
                        or not isinstance(anchor.get("path"), str)
                        or not isinstance(anchor.get("symbol"), str)
                        or not isinstance(anchor.get("line"), int)
                        or anchor["line"] < 1
                    ):
                        fail(f"source map has an invalid exact anchor: {backend}/{variant}/{field}")
                    source_path = SOURCE_MAP.parent.parent / anchor["path"]
                    try:
                        source_lines = source_path.read_text(encoding="utf-8").splitlines()
                    except OSError as exc:
                        fail(f"source map anchor path is unavailable: {backend}/{variant}/{field}: {exc}")
                    line = anchor["line"]
                    symbol = anchor["symbol"].rsplit("::", 1)[-1]
                    if line > len(source_lines) or symbol.lower() not in source_lines[line - 1].lower():
                        fail(f"source map exact anchor is stale: {backend}/{variant}/{field}")
            for field in (
                "current_kernel", "cache_format", "paging", "offload", "recompute",
                "compatibility",
            ):
                if not isinstance(entry.get(field), str) or not entry[field]:
                    fail(f"source map lacks {field}: {backend}/{variant}")
    return coverage, source_map, sha256(coverage_raw), sha256(source_map_raw)


def validate_geometry(geometry):
    if not isinstance(geometry, dict) or any(field not in geometry for field in GEOMETRY):
        fail("incomplete geometry")
    if not isinstance(geometry["resolution"], str) or not re.fullmatch(
        r"[1-9][0-9]*x[1-9][0-9]*", geometry["resolution"]
    ):
        fail("invalid resolution")
    for key in ("batch", "reference_count", "frames", "layers", "heads", "head_dimension", "sq", "skv"):
        value = geometry[key]
        minimum = 0 if key == "reference_count" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            fail(f"invalid geometry {key}")
    require_digest(geometry["prompt"], "prompt hash")
    try:
        guidance = float(geometry["guidance"])
    except (TypeError, ValueError):
        fail("invalid guidance")
    if not math.isfinite(guidance):
        fail("invalid guidance")
    for key in ("dtype", "mask", "rope"):
        if not isinstance(geometry[key], str) or not geometry[key]:
            fail(f"invalid geometry {key}")


def validate_hash_map(value, name, allow_empty=False):
    if not isinstance(value, dict) or (not value and not allow_empty):
        fail(f"invalid {name}")
    for key, item in value.items():
        if not isinstance(key, str) or not key or Path(key).name != key:
            fail(f"invalid {name} name")
        require_digest(item, f"{name} hash")


def validate_lifecycle(row):
    events = row["observer_events"]
    if not isinstance(events, list) or not events or any(
        not isinstance(event, dict) for event in events
    ):
        fail("missing exact observer transcript events")
    phases = [event.get("phase") for event in events]
    terminal = "cancelled" if row["arm"] == "cancel" else "generation-end"
    opposite = "generation-end" if terminal == "cancelled" else "cancelled"
    if opposite in phases:
        fail("observer lifecycle has the wrong terminal event")
    singleton = ("metadata", "generation-start", terminal, "metrics", "invalidated", "released")
    indices = {}
    for phase in singleton:
        matches = [index for index, value in enumerate(phases) if value == phase]
        if len(matches) != 1:
            fail(f"observer lifecycle requires exactly one {phase} event")
        indices[phase] = matches[0]
    creates = [index for index, value in enumerate(phases) if value == "cross-kv-created"]
    reads = [index for index, value in enumerate(phases) if value == "cross-kv-read"]
    if not creates or not reads:
        fail("observer lifecycle lacks cache creation/read events")
    if not (
        indices["metadata"] < indices["generation-start"] < min(creates) < min(reads)
        and max(creates) < indices[terminal]
        and max(reads) < indices[terminal]
        and indices[terminal] < indices["metrics"] < indices["invalidated"] < indices["released"]
    ):
        fail("observer lifecycle events are out of product order")
    metadata = events[indices["metadata"]]
    if row["arm"] == "cancel":
        if metadata.get("cancellation_armed") is not True or not isinstance(
            metadata.get("cancellation_arm_id"), str
        ) or not metadata["cancellation_arm_id"]:
            fail("cancel arm lacks product-owned cancellation identity")
    elif metadata.get("cancellation_armed") not in (None, False):
        fail("normal arm unexpectedly claims cancellation")
    lifecycle = row["lifecycle"]
    expected = {
        "created": phases.count("cross-kv-created"),
        "reused": phases.count("cross-kv-read"),
        "invalidated": phases.count("invalidated"),
        "cancelled": phases.count("cancelled"),
        "released": phases.count("released"),
    }
    if lifecycle != expected:
        fail("lifecycle summary differs from exact observer events")


def validate_backend_identity(events, backend):
    """Every observer event of a Metal-lane transcript declares `mlx-metal`; a CUDA-lane transcript
    may not claim any other backend (the Candle observers predate the field)."""
    for event in events:
        if not isinstance(event, dict) or event.get("sample_kind") == "process":
            continue
        declared = event.get("backend")
        if backend == "mlx-metal":
            if declared != "mlx-metal":
                fail("Metal-lane observer event lacks its mlx-metal backend identity")
        elif declared not in (None, backend):
            fail("CUDA-lane transcript claims a different measurement backend")


def validate(row, expected_source_map_hash, source_map=None):
    if not isinstance(row, dict):
        fail("receipt row must be an object")
    missing = set(REQUIRED) - row.keys()
    if missing:
        fail(f"missing fields: {sorted(missing)}")
    if row["producer"] != PRODUCER:
        fail("untrusted producer")
    if row["backend"] not in LANES:
        fail("invalid measurement backend")
    if row["family"] not in FAMILIES or not isinstance(row["variant"], str):
        fail("invalid family/variant")
    if source_map is None:
        source_map = checked_contracts()[1]
    entry = lane_entry(source_map, row["backend"], row["variant"])
    if entry["family"] != row["family"]:
        fail("row family differs from its registered lane route")
    if not isinstance(row["coordinate_name"], str) or not row["coordinate_name"]:
        fail("invalid coordinate name")
    require_digest(row["coordinate_id"], "coordinate id", 16)
    if row["arm"] not in ("normal", "cancel"):
        fail("invalid campaign arm")
    require_digest(row["source_ref"], "immutable source ref", 40)
    require_digest(row["model_snapshot_revision"], "model snapshot revision", 40)
    if row["residency_strategy"] != PRODUCT_RESIDENCY.get(row["variant"]):
        fail("receipt residency strategy differs from the frozen product route")
    for field in (
        "route_manifest_sha256", "source_map_sha256", "model_snapshot_sha256",
        "raw_receipt_sha256", "raw_receipt_sidecar_sha256",
    ):
        require_digest(row[field], field)
    if row["source_map_sha256"] != expected_source_map_hash:
        fail("receipt does not bind the checked-in source map")
    validate_hash_map(row["input_file_sha256"], "input file identity", allow_empty=True)
    validate_hash_map(row["evidence_artifact_sha256"], "evidence artifact identity")
    require_digest(row["media_manifest_sha256"], "media manifest identity")
    media_manifests = {
        name: item_hash
        for name, item_hash in row["evidence_artifact_sha256"].items()
        if name.endswith(".media.json")
    }
    if len(media_manifests) != 1 or next(iter(media_manifests.values())) != row[
        "media_manifest_sha256"
    ]:
        fail("row does not bind exactly one matching media manifest")
    if row["real_weights"] is not True or row["attention_kind"] != "cross":
        fail("real product cross-attention evidence required")
    if not isinstance(row["full_generation"], bool) or row["full_generation"] != (
        row["arm"] == "normal"
    ):
        fail("arm/full-generation lifecycle mismatch")
    validate_geometry(row["geometry"])
    validate_lifecycle(row)
    geometry = row["geometry"]
    events = row["observer_events"]
    validate_backend_identity(events, row["backend"])
    if row["backend"] == "mlx-metal":
        phases = [event.get("phase") for event in events]
        terminal = "cancelled" if row["arm"] == "cancel" else "generation-end"
        validate_phase_windows(events, phases.index(terminal), row["arm"])
    kind = entry["cache_kind"]
    operations = entry.get("operations")
    created = [
        event for event in events
        if isinstance(event, dict) and event.get("phase") == "cross-kv-created"
    ]
    reads = [
        event for event in events
        if isinstance(event, dict) and event.get("phase") == "cross-kv-read"
    ]
    releases = [
        event for event in events
        if isinstance(event, dict) and event.get("phase") in ("cross-kv-released", "released")
    ]
    if not created or not reads or not any(event.get("phase") == "released" for event in releases):
        fail("allocator read/release evidence is incomplete")
    for event in releases:
        active_allocator_measurement(event, "release remnant")
    if operations is not None and (
        any(event.get("operation") != operations["create"] for event in created)
        or any(event.get("operation") != operations["read"] for event in reads)
    ):
        fail("evidence is not anchored at the lane's exact K/V operations")
    expected_candidate = packed_group32_kv_bytes(
        geometry["batch"], geometry["heads"], geometry["skv"], geometry["head_dimension"],
    )
    if row["backend"] == "candle-cuda":
        if any("kv_batch" in event for event in created):
            fail("CUDA-lane events may not supply a Metal kv_batch")
    else:
        exact_batch = expected_kv_batch(entry, geometry)
        if any(
            not isinstance(event.get("kv_batch"), int) or isinstance(event.get("kv_batch"), bool)
            or event["kv_batch"] != exact_batch
            for event in created
        ):
            fail("Metal K/V batch differs from the route's exact CFG layout")
    if kind == "persistent":
        for event in created:
            kv_batch = event.get("kv_batch", geometry["batch"])
            if row["backend"] == "mlx-metal" and event.get("persistent_bytes") != dense_reference_kv_bytes(
                geometry, kv_batch
            ):
                fail("Metal persistent cache bytes differ from its exact live [B,H,Skv,D] tensors")
            if event.get("candidate_persistent_bytes") != packed_group32_kv_bytes(
                kv_batch, geometry["heads"], geometry["skv"], geometry["head_dimension"],
            ):
                fail("persistent candidate bytes differ from exact SC-20675 group32 projection")
        for event in reads:
            allocator = active_allocator_measurement(event, "persistent cache read")
            if event.get("transient_bytes") != (
                allocator["allocator_high_bytes"] - allocator["allocator_before_bytes"]
            ):
                fail("cache read transient differs from active allocator high-water")
    else:
        exact_dense = dense_reference_kv_bytes(geometry)
        if row["candidate_persistent_bytes"] != expected_candidate * geometry["layers"]:
            fail("recomputed candidate bytes differ from exact per-layer SC-20675 group32 projection")
        if (
            row["current_persistent_bytes"] != 0
            or any(event.get("persistent_bytes", 0) != 0 for event in created)
            or any(event.get("transient_bytes") != exact_dense for event in created)
            or any(event.get("transient_bytes") != exact_dense for event in reads)
            or row["current_read_transient_bytes"] != exact_dense
            or row["candidate_read_transient_bytes"] != exact_dense
        ):
            fail("recomputed K/V slice differs from exact live geometry/dtype")
        for event in reads:
            active_allocator_measurement(event, "recomputed K/V attention read")
    expected_coordinate_id = sha256(
        (json.dumps(row["geometry"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    )[:16]
    if row["coordinate_id"] != expected_coordinate_id:
        fail("coordinate id does not bind exact geometry")
    if not isinstance(row["allocator_samples"], list) or not row["allocator_samples"]:
        fail("missing allocator samples")
    if not isinstance(row["process_samples"], list) or not row["process_samples"]:
        fail("missing process samples")
    for field in (
        "model_snapshot_bytes", "current_persistent_bytes", "current_read_transient_bytes",
        "candidate_persistent_bytes", "candidate_read_transient_bytes",
        "generation_duration_ms", "cache_read_duration_ms",
        "joint_attention_context_duration_ms", "reused_requests", "minimum_cache_reads",
    ):
        require_nonnegative(row[field], field)
    if not isinstance(row["reference_runtime_attribution_available"], bool):
        fail("invalid reference runtime attribution availability")
    if (
        row["generation_duration_ms"] == 0
        or row["cache_read_duration_ms"] > row["generation_duration_ms"]
        or row["joint_attention_context_duration_ms"] > row["generation_duration_ms"]
        or (
            row["reference_runtime_attribution_available"] is False
            and row["cache_read_duration_ms"] != 0
        )
    ):
        fail("invalid runtime duration")
    if not entry["runtime_attribution"]:
        if (
            row["reference_runtime_attribution_available"] is not False
            or row["cache_read_duration_ms"] != 0
            or row["joint_attention_context_duration_ms"] <= 0
        ):
            fail("joint attention must remain non-attributable runtime context")
    elif (
        row["reference_runtime_attribution_available"] is not True
        or row["joint_attention_context_duration_ms"] != 0
    ):
        fail("attributable cross-attention runtime cannot also claim joint context")
    for sample in row["allocator_samples"] + row["process_samples"]:
        require_nonnegative(sample.get("peak_bytes"), "sample peak_bytes")
        if sample["peak_bytes"] <= 0:
            fail("allocator/process evidence must be physically observed")


def verify_seal_artifact(row, raw_data, sidecar_data, expected_name=None):
    unsigned = dict(row)
    unsigned["raw_receipt_sha256"] = ""
    unsigned["raw_receipt_sidecar_sha256"] = ""
    expected_raw = (
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if raw_data != expected_raw:
        fail("raw receipt artifact differs from sealed unsigned row")
    if sha256(raw_data) != row["raw_receipt_sha256"]:
        fail("raw receipt checksum mismatch")
    if sha256(sidecar_data) != row["raw_receipt_sidecar_sha256"]:
        fail("sidecar checksum mismatch")
    try:
        fields = sidecar_data.decode("utf-8").strip().split(None, 1)
    except UnicodeDecodeError:
        fail("sidecar is not UTF-8")
    if (
        len(fields) != 2
        or fields[0] != row["raw_receipt_sha256"]
        or (expected_name is not None and fields[1] != expected_name)
    ):
        fail("sidecar receipt mismatch")


def expected_coordinates(coverage, family, backend="candle-cuda"):
    return {
        (variant, coordinate): geometry
        for variant, variant_spec in lane_coverage(coverage, backend).get(family, {}).items()
        for coordinate, geometry in variant_spec["coordinates"].items()
    }


def geometry_matches_expected(actual, expected):
    for key, value in expected.items():
        if key == "guidance":
            try:
                if float(actual.get(key)) != float(value):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual.get(key) != value:
            return False
    return True


def coordinate_decision(row):
    allocator_peak = max(sample["peak_bytes"] for sample in row["allocator_samples"])
    process_peak = max(sample["peak_bytes"] for sample in row["process_samples"])
    persistent_saving = row["current_persistent_bytes"] - row["candidate_persistent_bytes"]
    transient_saving = row["current_read_transient_bytes"] - row["candidate_read_transient_bytes"]
    current_total = row["current_persistent_bytes"] + row["current_read_transient_bytes"]
    candidate_total = row["candidate_persistent_bytes"] + row["candidate_read_transient_bytes"]
    net_total_saving = current_total - candidate_total
    runtime_available = row["reference_runtime_attribution_available"]
    runtime_fraction = (
        row["cache_read_duration_ms"] / row["generation_duration_ms"]
        if runtime_available else 0.0
    )
    persistent_opportunity = (
        row["current_persistent_bytes"] >= THRESHOLDS["opportunity_bytes"]
        and row["current_persistent_bytes"] >= allocator_peak * THRESHOLDS["opportunity_peak_pct"]
    )
    transient_opportunity = (
        row["current_read_transient_bytes"] >= THRESHOLDS["opportunity_bytes"]
        and row["current_read_transient_bytes"] >= allocator_peak * THRESHOLDS["opportunity_peak_pct"]
    )
    persistent_saving_qualifies = (
        persistent_saving >= THRESHOLDS["saving_bytes"]
        and persistent_saving >= allocator_peak * THRESHOLDS["saving_peak_pct"]
    )
    transient_saving_qualifies = (
        transient_saving >= THRESHOLDS["saving_bytes"]
        and transient_saving >= allocator_peak * THRESHOLDS["saving_peak_pct"]
    )
    persistent_reduction = persistent_opportunity and persistent_saving_qualifies
    transient_reduction = transient_opportunity and transient_saving_qualifies
    reads_qualify = row["minimum_cache_reads"] >= THRESHOLDS["minimum_reads_per_cache"]
    runtime_qualifies = (
        runtime_available and runtime_fraction >= THRESHOLDS["runtime_only_pct"]
    )
    net_total_reduction = (
        net_total_saving >= THRESHOLDS["saving_bytes"]
        and net_total_saving >= allocator_peak * THRESHOLDS["saving_peak_pct"]
    )
    memory_qualified = persistent_reduction or transient_reduction
    runtime_only_qualified = runtime_qualifies and (
        persistent_saving_qualifies or transient_saving_qualifies
    )
    eligible = reads_qualify and net_total_reduction and (
        memory_qualified or runtime_only_qualified
    )
    return {
        "decision": "go" if eligible else "no-go",
        "allocator_peak_bytes": allocator_peak,
        "process_rss_peak_bytes": process_peak,
        "persistent_opportunity": persistent_opportunity,
        "read_transient_opportunity": transient_opportunity,
        "persistent_saving_bytes": persistent_saving,
        "read_transient_saving_bytes": transient_saving,
        "current_total_bytes": current_total,
        "candidate_total_bytes": candidate_total,
        "net_total_saving_bytes": net_total_saving,
        "net_total_reduction_qualifies": net_total_reduction,
        "persistent_reduction_qualifies": persistent_reduction,
        "read_transient_reduction_qualifies": transient_reduction,
        "persistent_saving_qualifies": persistent_saving_qualifies,
        "read_transient_saving_qualifies": transient_saving_qualifies,
        "memory_qualified": memory_qualified,
        "runtime_only_qualified": runtime_only_qualified,
        "reference_runtime_attribution_available": runtime_available,
        "cache_read_runtime_fraction": runtime_fraction,
        "runtime_qualifies": runtime_qualifies,
        "minimum_cache_reads": row["minimum_cache_reads"],
        "reads_qualify": reads_qualify,
    }


def reduce(rows):
    if not isinstance(rows, list) or not rows:
        fail("rows must be non-empty")
    coverage, source_map, coverage_hash, source_map_hash = checked_contracts()
    for row in rows:
        validate(row, source_map_hash, source_map)
    backends = {row["backend"] for row in rows}
    if len(backends) != 1:
        fail("one sealed campaign measures exactly one lane; rows mix measurement backends")
    backend = backends.pop()
    keys = [
        (row["family"], row["variant"], row["coordinate_name"], row["arm"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        fail("duplicate family/variant/coordinate/arm")

    decisions = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        expected = expected_coordinates(coverage, family, backend)
        expected_keys = {
            (variant, coordinate, arm)
            for variant, coordinate in expected
            for arm in coverage["required_arms"]
        }
        observed_keys = {
            (row["variant"], row["coordinate_name"], row["arm"])
            for row in family_rows
        }
        if observed_keys != expected_keys:
            decisions[family] = {
                "decision": "blocked",
                "reason": "sealed campaign requires every frozen coordinate and arm",
                "missing": sorted(expected_keys - observed_keys),
                "unexpected": sorted(observed_keys - expected_keys),
                "self_attention_excluded": True,
            }
            continue

        coordinate_results = {}
        qualifying_variants = set()
        variant_coordinate_decisions = {}
        for (variant, coordinate), expected_geometry in expected.items():
            pair = {
                row["arm"]: row
                for row in family_rows
                if row["variant"] == variant and row["coordinate_name"] == coordinate
            }
            normal, cancel = pair["normal"], pair["cancel"]
            if normal["geometry"] != cancel["geometry"]:
                fail(f"normal/cancel geometry differs: {variant}/{coordinate}")
            if not geometry_matches_expected(normal["geometry"], expected_geometry):
                fail(f"observed geometry violates frozen coordinate: {variant}/{coordinate}")
            for field in (
                "backend", "source_ref", "model_snapshot_revision", "residency_strategy",
                "route_manifest_sha256", "source_map_sha256",
                "model_snapshot_sha256", "model_snapshot_bytes", "coordinate_id",
                "input_file_sha256",
            ):
                if normal[field] != cancel[field]:
                    fail(f"normal/cancel identity differs: {variant}/{coordinate}/{field}")
            result = coordinate_decision(normal)
            result["cancel_arm_verified"] = True
            result["cancel_terminal"] = "cancelled"
            coordinate_results[f"{variant}/{coordinate}"] = result
            variant_coordinate_decisions.setdefault(variant, []).append(result["decision"])
        for variant, variant_decisions in variant_coordinate_decisions.items():
            # A variant is promotable only when every one of its frozen representative coordinates
            # qualifies independently. One favorable row cannot promote its siblings.
            if variant_decisions and all(item == "go" for item in variant_decisions):
                qualifying_variants.add(variant)
        decisions[family] = {
            "decision": "go" if qualifying_variants else "no-go",
            "qualifying_variants": sorted(qualifying_variants),
            "coordinates": coordinate_results,
            "self_attention_excluded": True,
        }
    return {
        "schema": "sc-20686-cache-attribution-v6",
        "backend": backend,
        "thresholds": THRESHOLDS,
        "coverage_manifest_sha256": coverage_hash,
        "source_map_sha256": source_map_hash,
        "source_map": source_map,
        "decisions": decisions,
        "rows": rows,
    }


def file_sha256(path):
    identity = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            identity.update(chunk)
    return identity.hexdigest()


def verify_named_artifact(bundle, name, expected_hash, load_payload=True):
    if Path(name).name != name:
        fail("bundle artifact name escapes the bundle")
    path = bundle / name
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_hash:
        fail(f"bundle artifact checksum mismatch: {name}")
    sidecar = (bundle / f"{name}.sha256").read_bytes()
    expected_sidecar = f"{expected_hash}  {name}\n".encode("utf-8")
    if sidecar != expected_sidecar:
        fail(f"bundle artifact sidecar mismatch: {name}")
    return path.read_bytes() if load_payload else None


def verify_media_manifest(row, manifest_raw, artifacts, bundle, evidence_names, expected_stem):
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"campaign media manifest is malformed: {exc}")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "sc-20686-media-manifest-v1"
        or manifest.get("arm") != row["arm"]
        or not isinstance(manifest.get("output_name"), str)
        or not isinstance(manifest.get("files"), list)
    ):
        fail("campaign media manifest schema is invalid")
    expected_output_name = "media.png" if row["variant"] in FLUX_IMAGE_ROUTES else "media"
    if manifest["output_name"] != expected_output_name:
        fail("campaign media manifest output conflicts with its route")
    files = manifest["files"]
    if row["arm"] == "cancel":
        if manifest.get("output_kind") != "absent" or files:
            fail("cancel arm must attest that no media was published")
    elif manifest.get("output_kind") not in ("file", "directory") or not files:
        fail("normal arm lacks sealed generated media")
    if manifest.get("output_kind") == "file" and len(files) != 1:
        fail("file media manifest must contain exactly one output")
    media_names = set()
    relative_paths = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "artifact", "relative_path", "bytes", "sha256"
        }:
            fail("campaign media file metadata is malformed")
        name = entry["artifact"]
        relative_path = entry["relative_path"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(rf"{re.escape(expected_stem)}\.media-[0-9]{{4}}", name)
            or not isinstance(relative_path, str)
            or not relative_path
            or relative_path == "."
            or "\\" in relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or relative_path in relative_paths
            or name in media_names
        ):
            fail("campaign media file identity is malformed")
        size = entry["bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            fail("campaign media file size is invalid")
        require_digest(entry["sha256"], "campaign media file hash")
        if artifacts.get(name) != entry["sha256"]:
            fail("campaign media manifest hash differs from the sealed artifact")
        media_path = bundle / name
        if media_path.stat().st_size != size:
            fail("campaign media manifest size differs from the sealed artifact")
        relative_paths.add(relative_path)
        media_names.add(name)
    if manifest.get("output_kind") == "file" and files[0]["relative_path"] != expected_output_name:
        fail("file media manifest does not preserve its output name")
    if evidence_names != media_names:
        fail("row media evidence inventory differs from its exact manifest")
    return manifest


def verify_campaign_bundle(bundle):
    campaign_raw = (bundle / "campaign.json").read_bytes()
    campaign_hash = sha256(campaign_raw)
    expected_sidecar = f"{campaign_hash}  campaign.json\n".encode("utf-8")
    if (bundle / "campaign.json.sha256").read_bytes() != expected_sidecar:
        fail("campaign aggregate sidecar mismatch")
    campaign = json.loads(campaign_raw.decode("utf-8"))
    if campaign.get("schema") != "sc-20686-campaign-bundle-v6":
        fail("invalid campaign bundle schema")
    artifacts = campaign.get("artifact_sha256")
    validate_hash_map(artifacts, "campaign artifact identity")
    expected_files = {"campaign.json", "campaign.json.sha256"}
    for name in artifacts:
        expected_files.update((name, f"{name}.sha256"))
    observed_files = {path.name for path in bundle.iterdir() if path.is_file()}
    if observed_files != expected_files:
        fail("campaign bundle file inventory is not exact")
    payloads = {}
    for name, item_hash in artifacts.items():
        payloads[name] = verify_named_artifact(
            bundle, name, item_hash, load_payload=".media-" not in name
        )
    sealed_rows = campaign.get("rows")
    row_files = campaign.get("row_files")
    if not isinstance(sealed_rows, list) or not isinstance(row_files, list) or len(sealed_rows) != len(row_files):
        fail("campaign row inventory is malformed")
    resolved_raw = payloads.get("campaign-inputs.resolved.json")
    if resolved_raw is None:
        fail("campaign lacks sealed resolved inputs")
    try:
        resolved = json.loads(resolved_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"campaign resolved inputs are malformed: {exc}")
    if (
        resolved.get("schema") != "sc-20686-resolved-inputs-v4"
        or not isinstance(resolved.get("coordinates"), list)
        or resolved.get("backend") not in LANES
        or resolved.get("mode", "decision") != "decision"
    ):
        fail("campaign resolved inputs schema is invalid")
    if any(row.get("backend") != resolved["backend"] for row in sealed_rows):
        fail("sealed rows differ from the resolved measurement backend")
    require_digest(resolved.get("inference_revision"), "resolved inference revision", 40)
    resolved_by_key = {}
    for entry in resolved["coordinates"]:
        if not isinstance(entry, dict):
            fail("campaign resolved coordinate is malformed")
        key = (entry.get("family"), entry.get("variant"), entry.get("name"))
        if key in resolved_by_key:
            fail("campaign resolved inputs contain duplicate coordinates")
        for field in ("entrypoint_sha256", "snapshot_sha256", "route_manifest_sha256"):
            require_digest(entry.get(field), f"resolved {field}")
        require_digest(entry.get("model_snapshot_revision"), "resolved model snapshot revision", 40)
        if (
            not isinstance(entry.get("entrypoint"), str)
            or not isinstance(entry.get("snapshot"), str)
            or not isinstance(entry.get("args"), list)
            or any(not isinstance(item, str) for item in entry["args"])
            or not isinstance(entry.get("input_files"), list)
            or entry.get("residency_strategy") != PRODUCT_RESIDENCY.get(entry.get("variant"))
        ):
            fail("campaign resolved coordinate paths/arguments are malformed")
        manifest_document = {
            "route": entry["variant"],
            "coordinate": entry["name"],
            "entrypoint": entry["entrypoint"],
            "entrypoint_sha256": entry["entrypoint_sha256"],
            "snapshot": entry["snapshot"],
            "model_snapshot_revision": entry["model_snapshot_revision"],
            "residency_strategy": entry["residency_strategy"],
            "args": entry["args"],
            "input_files": entry["input_files"],
        }
        if sha256((json.dumps(manifest_document, sort_keys=True, separators=(",", ":")) + "\n").encode()) != entry["route_manifest_sha256"]:
            fail("route manifest identity cannot be recomputed from sealed resolved inputs")
        resolved_by_key[key] = entry
    row_coordinate_keys = {
        (row.get("family"), row.get("variant"), row.get("coordinate_name"))
        for row in sealed_rows
    }
    if set(resolved_by_key) != row_coordinate_keys:
        fail("resolved input coordinate closure differs from sealed rows")
    bound_media_artifacts = set()
    for row_index, (row, name) in enumerate(zip(sealed_rows, row_files)):
        if name != f"row-{row_index:02d}.json":
            fail("campaign row filenames are not canonical")
        raw = payloads.get(name)
        if raw is None:
            fail(f"campaign lacks raw row/sidecar: {name}")
        sidecar = (bundle / f"{name}.sha256").read_bytes()
        verify_seal_artifact(row, raw, sidecar, name)
        for evidence_name, evidence_hash in row["evidence_artifact_sha256"].items():
            if artifacts.get(evidence_name) != evidence_hash:
                fail(f"campaign does not bind row transcript: {evidence_name}")
        evidence_names = set(row["evidence_artifact_sha256"])
        by_suffix = {
            suffix: [item for item in evidence_names if item.endswith(suffix)]
            for suffix in (".events.jsonl", ".stdout", ".stderr", ".command.json", ".process.json")
        }
        manifest_names = [item for item in evidence_names if item.endswith(".media.json")]
        expected_stem = f"run-{row_index // 2:02d}-{row['arm']}"
        if (
            any(names != [f"{expected_stem}{suffix}"] for suffix, names in by_suffix.items())
            or manifest_names != [f"{expected_stem}.media.json"]
        ):
            fail("row evidence artifact inventory lacks an exact transcript or media manifest")
        fixed_names = {names[0] for names in by_suffix.values()} | {manifest_names[0]}
        media_evidence_names = evidence_names - fixed_names
        manifest_name = manifest_names[0]
        if row.get("media_manifest_sha256") != artifacts.get(manifest_name):
            fail("row media manifest identity differs from the sealed artifact")
        media_manifest = verify_media_manifest(
            row, payloads[manifest_name], artifacts, bundle, media_evidence_names,
            expected_stem,
        )
        bound_media_artifacts.update(media_evidence_names | {manifest_name})
        event_transcript = payloads[by_suffix[".events.jsonl"][0]]
        try:
            if not event_transcript or b"\r" in event_transcript or not event_transcript.endswith(b"\n"):
                fail("campaign observer event transcript is malformed")
            observed = [json.loads(line) for line in event_transcript.decode("utf-8").splitlines()]
            if not observed or any(not isinstance(event, dict) for event in observed):
                fail("campaign observer event transcript is malformed")
            process = json.loads(payloads[by_suffix[".process.json"][0]].decode("utf-8"))
            command = json.loads(payloads[by_suffix[".command.json"][0]].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"campaign run transcript is malformed: {exc}")
        if not isinstance(process, dict) or not isinstance(process.get("samples"), list):
            fail("campaign process sample transcript is malformed")
        if observed + process["samples"] != row["observer_events"]:
            fail("sealed row events differ from the exact event/process transcripts")
        expected_allocator = [
            {"phase": event["phase"], "peak_bytes": event["peak_bytes"]}
            for event in observed
            if event.get("sample_kind") == "allocator" and "peak_bytes" in event
        ]
        expected_process = [
            {"phase": event["phase"], "peak_bytes": event["peak_bytes"]}
            for event in process["samples"]
            if event.get("sample_kind") == "process" and "peak_bytes" in event
        ]
        if (
            row["allocator_samples"] != expected_allocator
            or row["process_samples"] != expected_process
        ):
            fail("sealed allocator/process samples differ from their exact transcripts")
        argv = command.get("argv") if isinstance(command, dict) else None
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            fail("campaign command transcript is malformed")
        resolved_entry = resolved_by_key[
            (row["family"], row["variant"], row["coordinate_name"])
        ]
        expected_after_events = [
            "--sc20686-source-ref", resolved["inference_revision"],
            "--sc20686-residency", resolved_entry["residency_strategy"],
            "--snapshot", resolved_entry["snapshot"], "--variant", row["variant"],
            *resolved_entry["args"],
        ]
        if row["arm"] == "cancel":
            expected_after_events.append("--sc20686-cancel")
        event_path = Path(argv[3]) if len(argv) > 3 else Path("")
        output_path = Path(argv[-1]) if argv else Path("")
        if (
            argv[:3] != [resolved_entry["entrypoint"], "--sc20686-campaign", "--sc20686-events"]
            or len(argv) < 7
            or not event_path.is_absolute()
            or event_path.name != "events.jsonl"
            or argv[4:-2] != expected_after_events
            or argv[-2] != "--out"
            or not output_path.is_absolute()
            or output_path.name != media_manifest["output_name"]
            or output_path.parent != event_path.parent
            or event_path.parent.name != "sealed-run"
        ):
            fail("campaign command conflicts with sealed resolved inputs")
        if (
            row["route_manifest_sha256"] != resolved_entry["route_manifest_sha256"]
            or row["source_ref"] != resolved["inference_revision"]
            or row["model_snapshot_revision"] != resolved_entry["model_snapshot_revision"]
            or row["residency_strategy"] != resolved_entry["residency_strategy"]
            or row["model_snapshot_sha256"] != resolved_entry["snapshot_sha256"]
            or row["model_snapshot_bytes"] != resolved_entry["snapshot_bytes"]
        ):
            fail("sealed row identity differs from sealed resolved inputs")
        resolved_input_hashes = {}
        for item in resolved_entry["input_files"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("flag"), str)
                or not isinstance(item.get("argument_index"), int)
            ):
                fail("resolved input file inventory is malformed")
            require_digest(item.get("sha256"), "resolved input file hash")
            key = f"{item['flag'][2:]}-{item['argument_index'] - 1:02d}"
            resolved_input_hashes[key] = item["sha256"]
        if row["input_file_sha256"] != resolved_input_hashes:
            fail("sealed row file identities differ from sealed resolved inputs")
    campaign_media_artifacts = {
        name for name in artifacts if ".media-" in name or name.endswith(".media.json")
    }
    if campaign_media_artifacts != bound_media_artifacts:
        fail("campaign contains media artifacts not bound to an exact run row")
    expected_input_hashes = {
        f"{row['family']}/{row['variant']}/{row['coordinate_name']}/{row['arm']}":
            row["input_file_sha256"]
        for row in sealed_rows
    }
    if campaign.get("route_input_file_sha256") != expected_input_hashes:
        fail("campaign aggregate does not bind route input file hashes")
    _coverage, _source_map, coverage_hash, source_map_hash = checked_contracts()
    if artifacts.get("sc20686_coverage_manifest.json") != coverage_hash:
        fail("campaign does not seal the checked-in coverage manifest")
    if artifacts.get("sc20686_source_map.json") != source_map_hash:
        fail("campaign does not seal the checked-in source map")
    result = reduce(sealed_rows)
    if campaign.get("decision") != result:
        fail("campaign aggregate decision is not reproducible from sealed rows")
    return result


def schedule_control_summary(events, variant, coordinate_name, expected_geometry, identity):
    """Validate one Metal **schedule-control** transcript and summarize its product-schedule peaks.

    The control arm opens no per-read evaluation windows, so its run and phase-window peaks follow
    the product's own lazy schedule (the decision arms' per-read windows perturb that schedule). It
    must still be the same real product route: identical runtime identity and native geometry,
    cache creation/release, phase windows with Darwin footprint, and a normal completion."""
    validate_backend_identity(events, "mlx-metal")
    observer = [e for e in events if isinstance(e, dict) and e.get("sample_kind") != "process"]
    process = [e for e in events if isinstance(e, dict) and e.get("sample_kind") == "process"]
    phases = [event.get("phase") for event in observer]
    if "cross-kv-read" in phases or "cancelled" in phases:
        fail("schedule-control transcript contains read windows or a cancellation")
    indices = {}
    for phase in ("metadata", "generation-start", "generation-end", "metrics", "invalidated", "released"):
        matches = [index for index, value in enumerate(phases) if value == phase]
        if len(matches) != 1:
            fail(f"schedule-control transcript requires exactly one {phase} event")
        indices[phase] = matches[0]
    creates = [index for index, value in enumerate(phases) if value == "cross-kv-created"]
    if not creates or not (
        indices["metadata"] < indices["generation-start"] < min(creates)
        and max(creates) < indices["generation-end"] < indices["metrics"]
        < indices["invalidated"] < indices["released"]
    ):
        fail("schedule-control lifecycle is out of product order")
    metadata = observer[indices["metadata"]]
    expected = {
        "schedule_control": True, "variant": variant, "real_weights": True,
        "full_generation": True, "cancellation_armed": False, "attention_kind": "cross",
        **identity,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        fail("schedule-control metadata differs from its route identity")
    geometry = metadata.get("geometry")
    validate_geometry(geometry)
    if not geometry_matches_expected(geometry, expected_geometry):
        fail("schedule-control geometry violates its frozen coordinate")
    validate_phase_windows(observer, indices["generation-end"], "normal")
    active_allocator_measurement(observer[indices["released"]], "schedule-control remnant")
    peaks = [e.get("peak_bytes") for e in observer if e.get("sample_kind") == "allocator"]
    process_peaks = [e.get("peak_bytes") for e in process]
    for value in peaks + process_peaks:
        require_nonnegative(value, "schedule-control peak")
    if not process_peaks or max(peaks) <= 0 or max(process_peaks) <= 0:
        fail("schedule-control run lacks physical allocator/process peaks")
    return {
        "variant": variant,
        "coordinate_name": coordinate_name,
        "geometry": {key: geometry[key] for key in GEOMETRY},
        "run_peak_bytes": max(peaks),
        "process_peak_bytes": max(process_peaks),
        "phase_windows": [
            {
                "window": event["window"],
                "window_index": event["window_index"],
                "high_bytes": event["allocator_high_bytes"] - event["allocator_before_bytes"],
                "phys_footprint_peak_bytes": event["phys_footprint_peak_bytes"],
            }
            for event in observer if event.get("phase") == "phase-window"
        ],
    }


def verify_schedule_control_bundle(bundle):
    """Verify a sealed Metal schedule-control bundle and recompute every summary from its
    transcripts. Coverage is the full Metal lane: no coordinate may be missing."""
    campaign_raw = (bundle / "campaign.json").read_bytes()
    if (bundle / "campaign.json.sha256").read_bytes() != f"{sha256(campaign_raw)}  campaign.json\n".encode():
        fail("schedule-control aggregate sidecar mismatch")
    campaign = json.loads(campaign_raw.decode("utf-8"))
    if campaign.get("schema") != "sc-20686-schedule-control-v1" or campaign.get("backend") != "mlx-metal":
        fail("invalid schedule-control bundle schema")
    artifacts = campaign.get("artifact_sha256")
    validate_hash_map(artifacts, "schedule-control artifact identity")
    expected_files = {"campaign.json", "campaign.json.sha256"}
    for name in artifacts:
        expected_files.update((name, f"{name}.sha256"))
    if {path.name for path in bundle.iterdir() if path.is_file()} != expected_files:
        fail("schedule-control bundle file inventory is not exact")
    payloads = {
        name: verify_named_artifact(bundle, name, item_hash, load_payload=".media-" not in name)
        for name, item_hash in artifacts.items()
    }
    resolved = json.loads(payloads["campaign-inputs.resolved.json"].decode("utf-8"))
    if (
        resolved.get("schema") != "sc-20686-resolved-inputs-v4"
        or resolved.get("backend") != "mlx-metal"
        or resolved.get("mode") != "schedule-control"
    ):
        fail("schedule-control resolved inputs are invalid")
    coverage, _source_map, coverage_hash, source_map_hash = checked_contracts()
    if artifacts.get("sc20686_coverage_manifest.json") != coverage_hash or artifacts.get(
        "sc20686_source_map.json"
    ) != source_map_hash:
        fail("schedule-control bundle does not seal the checked-in contracts")
    frozen = {
        (variant, name): geometry
        for family in FAMILIES
        for (variant, name), geometry in expected_coordinates(coverage, family, "mlx-metal").items()
    }
    entries = resolved["coordinates"]
    if {(entry.get("variant"), entry.get("name")) for entry in entries} != set(frozen) or len(
        entries
    ) != len(frozen):
        fail("schedule-control coverage is not the complete Metal lane")
    summaries = campaign.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != len(entries):
        fail("schedule-control summary inventory is malformed")
    media = set()
    for index, (entry, summary) in enumerate(zip(entries, summaries)):
        stem = f"run-{index:02d}-control"
        transcript = payloads.get(f"{stem}.events.jsonl")
        process_raw = payloads.get(f"{stem}.process.json")
        manifest_raw = payloads.get(f"{stem}.media.json")
        command_raw = payloads.get(f"{stem}.command.json")
        if None in (transcript, process_raw, manifest_raw, command_raw):
            fail(f"schedule-control run lacks its transcripts: {stem}")
        if not transcript.endswith(b"\n") or b"\r" in transcript:
            fail("schedule-control event transcript is malformed")
        events = [json.loads(line) for line in transcript.decode("utf-8").splitlines()]
        events += json.loads(process_raw.decode("utf-8"))["samples"]
        argv = json.loads(command_raw.decode("utf-8"))["argv"]
        if (
            argv[:3] != [entry["entrypoint"], "--sc20686-campaign", "--sc20686-events"]
            or "--sc20686-schedule-control" not in argv
            or "--sc20686-cancel" in argv
            or argv[argv.index("--variant") + 1] != entry["variant"]
            or argv[argv.index("--snapshot") + 1] != entry["snapshot"]
        ):
            fail("schedule-control command conflicts with sealed resolved inputs")
        recomputed = schedule_control_summary(
            events, entry["variant"], entry["name"], frozen[(entry["variant"], entry["name"])],
            {
                "source_ref": resolved["inference_revision"],
                "model_snapshot_revision": entry["model_snapshot_revision"],
                "residency_strategy": entry["residency_strategy"],
                "snapshot_sha256": entry["snapshot_sha256"],
                "snapshot_bytes": entry["snapshot_bytes"],
            },
        )
        if recomputed != summary:
            fail("schedule-control summary is not reproducible from its transcript")
        media_names = {name for name in artifacts if name.startswith(f"{stem}.media-")}
        verify_media_manifest(
            {"arm": "control", "variant": entry["variant"]}, manifest_raw, artifacts, bundle,
            media_names, stem,
        )
        media |= media_names | {f"{stem}.media.json"}
    if {name for name in artifacts if ".media" in name} != media:
        fail("schedule-control bundle contains unbound media")
    return campaign


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False)
    try:
        handle.write(payload)
        handle.close()
        os.replace(handle.name, path)
    finally:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sidecar", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.input.is_dir():
            if args.sidecar is not None:
                fail("bundle input does not accept --sidecar")
            result = verify_campaign_bundle(args.input)
        else:
            fail("standalone raw-row reduction is forbidden; provide a fully verified campaign bundle")
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_write(args.output, payload)
        print(json.dumps({"sha256": sha256(payload), "output": str(args.output)}))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"SC-20686 invalid receipt: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

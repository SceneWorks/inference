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
    "producer", "family", "variant", "coordinate_name", "coordinate_id", "arm",
    "source_ref", "route_manifest_sha256", "source_map_sha256",
    "model_snapshot_sha256", "model_snapshot_bytes", "input_file_sha256",
    "evidence_artifact_sha256", "geometry", "lifecycle", "allocator_samples",
    "process_samples", "observer_events", "raw_receipt_sha256", "raw_receipt_sidecar_sha256",
    "real_weights", "full_generation", "attention_kind", "current_persistent_bytes",
    "current_read_transient_bytes", "candidate_persistent_bytes",
    "candidate_read_transient_bytes", "generation_duration_ms",
    "cache_read_duration_ms", "reused_requests", "minimum_cache_reads",
)


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
    if event.get("allocator_measurement_available") is not True or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values.values()
    ):
        fail(f"{label} lacks active allocator high-water evidence")
    if (
        values["allocator_high_bytes"] < values["allocator_before_bytes"]
        or values["allocator_high_bytes"] < values["allocator_after_bytes"]
        or values["allocator_reserved_bytes"] < values["allocator_after_bytes"]
    ):
        fail(f"{label} allocator high-water/remnant ordering is invalid")
    return values


def dense_reference_kv_bytes(geometry):
    dtype_bytes = {"F16": 2, "BF16": 2, "F32": 4}.get(geometry["dtype"].upper())
    if dtype_bytes is None:
        fail("FLUX reference slice has unsupported live dtype")
    return (
        2 * geometry["batch"] * geometry["heads"] * geometry["skv"]
        * geometry["head_dimension"] * dtype_bytes
    )


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
    coverage, coverage_raw = read_checked_json(MANIFEST, "sc-20686-supported-coverage-v2")
    source_map, source_map_raw = read_checked_json(SOURCE_MAP, "sc-20686-source-map-v1")
    if source_map.get("compatibility_target") != {
        "story": "SC-20675",
        "format": "packed-group-affine-v2",
        "bits": 2,
        "group_size": 32,
        "metadata": "f16 scale and zero per completed group",
        "pending_key_tail": "dense f32",
    }:
        fail("source map compatibility target differs from the frozen SC-20675 format")
    expected = {
        variant: family_name
        for family_name, family in coverage.get("families", {}).items()
        for variant in family
    }
    if set(source_map.get("variants", {})) != set(expected):
        fail("source map does not close over the frozen variant surface")
    for variant, entry in source_map["variants"].items():
        if (
            entry.get("route") != variant
            or entry.get("family") != expected[variant]
            or entry.get("self_attention_excluded") is not True
        ):
            fail(f"source map route is malformed: {variant}")
        anchor_fields = ["activation", "creation", "reads", "release"]
        if entry["family"] == "flux2-klein":
            anchor_fields.append("reference_slice")
        for field in anchor_fields:
            anchors = entry.get(field)
            if not isinstance(anchors, list) or not anchors:
                fail(f"source map lacks {field} anchors: {variant}")
            for anchor in anchors:
                if (
                    not isinstance(anchor, dict)
                    or not isinstance(anchor.get("path"), str)
                    or not isinstance(anchor.get("symbol"), str)
                    or not isinstance(anchor.get("line"), int)
                    or anchor["line"] < 1
                ):
                    fail(f"source map has an invalid exact anchor: {variant}/{field}")
                source_path = SOURCE_MAP.parent.parent / anchor["path"]
                try:
                    source_lines = source_path.read_text(encoding="utf-8").splitlines()
                except OSError as exc:
                    fail(f"source map anchor path is unavailable: {variant}/{field}: {exc}")
                line = anchor["line"]
                symbol = anchor["symbol"].rsplit("::", 1)[-1]
                if line > len(source_lines) or symbol.lower() not in source_lines[line - 1].lower():
                    fail(f"source map exact anchor is stale: {variant}/{field}")
        for field in (
            "current_kernel", "cache_format", "paging", "offload", "recompute",
            "compatibility",
        ):
            if not isinstance(entry.get(field), str) or not entry[field]:
                fail(f"source map lacks {field}: {variant}")
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


def validate(row, expected_source_map_hash):
    if not isinstance(row, dict):
        fail("receipt row must be an object")
    missing = set(REQUIRED) - row.keys()
    if missing:
        fail(f"missing fields: {sorted(missing)}")
    if row["producer"] != "sc20686-campaign-adapter-v2":
        fail("untrusted producer")
    if row["family"] not in FAMILIES or not isinstance(row["variant"], str):
        fail("invalid family/variant")
    if not isinstance(row["coordinate_name"], str) or not row["coordinate_name"]:
        fail("invalid coordinate name")
    require_digest(row["coordinate_id"], "coordinate id", 16)
    if row["arm"] not in ("normal", "cancel"):
        fail("invalid campaign arm")
    require_digest(row["source_ref"], "immutable source ref", 40)
    for field in (
        "route_manifest_sha256", "source_map_sha256", "model_snapshot_sha256",
        "raw_receipt_sha256", "raw_receipt_sidecar_sha256",
    ):
        require_digest(row[field], field)
    if row["source_map_sha256"] != expected_source_map_hash:
        fail("receipt does not bind the checked-in source map")
    validate_hash_map(row["input_file_sha256"], "input file identity", allow_empty=True)
    validate_hash_map(row["evidence_artifact_sha256"], "evidence artifact identity")
    if row["real_weights"] is not True or row["attention_kind"] != "cross":
        fail("real product cross-attention evidence required")
    if (row["arm"] == "normal") != (row["full_generation"] is True):
        fail("arm/full-generation lifecycle mismatch")
    validate_geometry(row["geometry"])
    expected_candidate = packed_group32_kv_bytes(
        row["geometry"]["batch"], row["geometry"]["heads"], row["geometry"]["skv"],
        row["geometry"]["head_dimension"],
    )
    created = [
        event for event in row["observer_events"]
        if isinstance(event, dict) and event.get("phase") == "cross-kv-created"
    ]
    if row["family"] == "wan":
        if not created or any(
            event.get("candidate_persistent_bytes") != expected_candidate for event in created
        ):
            fail("Wan candidate bytes differ from exact SC-20675 group32 projection")
    elif row["candidate_persistent_bytes"] != expected_candidate * row["geometry"]["layers"]:
        fail("FLUX candidate bytes differ from exact per-layer SC-20675 group32 projection")
    reads = [
        event for event in row["observer_events"]
        if isinstance(event, dict) and event.get("phase") == "cross-kv-read"
    ]
    releases = [
        event for event in row["observer_events"]
        if isinstance(event, dict) and event.get("phase") in ("cross-kv-released", "released")
    ]
    if not reads or not any(event.get("phase") == "released" for event in releases):
        fail("allocator read/release evidence is incomplete")
    for event in releases:
        active_allocator_measurement(event, "release remnant")
    if row["family"] == "wan":
        for event in reads:
            allocator = active_allocator_measurement(event, "Wan cache read")
            if event.get("transient_bytes") != (
                allocator["allocator_high_bytes"] - allocator["allocator_before_bytes"]
            ):
                fail("Wan read transient differs from active allocator high-water")
    else:
        exact_dense = dense_reference_kv_bytes(row["geometry"])
        if (
            any(event.get("operation") != "DoubleAttention::to_k/to_v(reference-slice)"
                or event.get("transient_bytes") != exact_dense for event in created)
            or any(event.get("operation") != "DoubleAttention::attention(reference-kv-slice)"
                   or event.get("transient_bytes") != exact_dense for event in reads)
            or row["current_read_transient_bytes"] != exact_dense
            or row["candidate_read_transient_bytes"] != exact_dense
        ):
            fail("FLUX reference K/V slice differs from exact live geometry/dtype")
        for event in reads:
            active_allocator_measurement(event, "FLUX attention read")
    expected_coordinate_id = sha256(
        (json.dumps(row["geometry"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    )[:16]
    if row["coordinate_id"] != expected_coordinate_id:
        fail("coordinate id does not bind exact geometry")
    lifecycle = row["lifecycle"]
    if not isinstance(lifecycle, dict) or any(
        field not in lifecycle for field in ("created", "reused", "invalidated", "released")
    ):
        fail("incomplete lifecycle")
    if not isinstance(row["allocator_samples"], list) or not row["allocator_samples"]:
        fail("missing allocator samples")
    if not isinstance(row["process_samples"], list) or not row["process_samples"]:
        fail("missing process samples")
    if not isinstance(row["observer_events"], list) or not row["observer_events"]:
        fail("missing exact observer transcript events")
    for field in (
        "model_snapshot_bytes", "current_persistent_bytes", "current_read_transient_bytes",
        "candidate_persistent_bytes", "candidate_read_transient_bytes",
        "generation_duration_ms", "cache_read_duration_ms", "reused_requests",
        "minimum_cache_reads",
    ):
        require_nonnegative(row[field], field)
    if row["generation_duration_ms"] == 0 or row["cache_read_duration_ms"] > row["generation_duration_ms"]:
        fail("invalid runtime duration")
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


def expected_coordinates(coverage, family):
    return {
        (variant, coordinate): geometry
        for variant, variant_spec in coverage["families"][family].items()
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
    runtime_fraction = row["cache_read_duration_ms"] / row["generation_duration_ms"]
    persistent_opportunity = (
        row["current_persistent_bytes"] >= THRESHOLDS["opportunity_bytes"]
        and row["current_persistent_bytes"] >= allocator_peak * THRESHOLDS["opportunity_peak_pct"]
    )
    transient_opportunity = (
        row["current_read_transient_bytes"] >= THRESHOLDS["opportunity_bytes"]
        and row["current_read_transient_bytes"] >= allocator_peak * THRESHOLDS["opportunity_peak_pct"]
    )
    persistent_reduction = (
        persistent_opportunity
        and persistent_saving >= THRESHOLDS["saving_bytes"]
        and persistent_saving >= allocator_peak * THRESHOLDS["saving_peak_pct"]
    )
    transient_reduction = (
        transient_opportunity
        and transient_saving >= THRESHOLDS["saving_bytes"]
        and transient_saving >= allocator_peak * THRESHOLDS["saving_peak_pct"]
    )
    reads_qualify = row["minimum_cache_reads"] >= THRESHOLDS["minimum_reads_per_cache"]
    runtime_qualifies = runtime_fraction >= THRESHOLDS["runtime_only_pct"]
    net_total_reduction = (
        net_total_saving >= THRESHOLDS["saving_bytes"]
        and net_total_saving >= allocator_peak * THRESHOLDS["saving_peak_pct"]
    )
    eligible = (
        reads_qualify
        and runtime_qualifies
        and (persistent_reduction or transient_reduction)
        and net_total_reduction
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
        validate(row, source_map_hash)
    keys = [
        (row["family"], row["variant"], row["coordinate_name"], row["arm"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        fail("duplicate family/variant/coordinate/arm")

    decisions = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        expected = expected_coordinates(coverage, family)
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
                "source_ref", "route_manifest_sha256", "source_map_sha256",
                "model_snapshot_sha256", "model_snapshot_bytes", "coordinate_id",
                "input_file_sha256",
            ):
                if normal[field] != cancel[field]:
                    fail(f"normal/cancel identity differs: {variant}/{coordinate}/{field}")
            result = coordinate_decision(normal)
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
        "schema": "sc-20686-cache-attribution-v3",
        "thresholds": THRESHOLDS,
        "coverage_manifest_sha256": coverage_hash,
        "source_map_sha256": source_map_hash,
        "source_map": source_map,
        "decisions": decisions,
        "rows": rows,
    }


def verify_named_artifact(bundle, name, expected_hash):
    if Path(name).name != name:
        fail("bundle artifact name escapes the bundle")
    payload = (bundle / name).read_bytes()
    if sha256(payload) != expected_hash:
        fail(f"bundle artifact checksum mismatch: {name}")
    sidecar = (bundle / f"{name}.sha256").read_bytes()
    expected_sidecar = f"{expected_hash}  {name}\n".encode("utf-8")
    if sidecar != expected_sidecar:
        fail(f"bundle artifact sidecar mismatch: {name}")
    return payload


def verify_campaign_bundle(bundle):
    campaign_raw = (bundle / "campaign.json").read_bytes()
    campaign_hash = sha256(campaign_raw)
    expected_sidecar = f"{campaign_hash}  campaign.json\n".encode("utf-8")
    if (bundle / "campaign.json.sha256").read_bytes() != expected_sidecar:
        fail("campaign aggregate sidecar mismatch")
    campaign = json.loads(campaign_raw.decode("utf-8"))
    if campaign.get("schema") != "sc-20686-campaign-bundle-v3":
        fail("invalid campaign bundle schema")
    artifacts = campaign.get("artifact_sha256")
    validate_hash_map(artifacts, "campaign artifact identity")
    expected_files = {"campaign.json", "campaign.json.sha256"}
    for name in artifacts:
        expected_files.update((name, f"{name}.sha256"))
    observed_files = {path.name for path in bundle.iterdir() if path.is_file()}
    if observed_files != expected_files:
        fail("campaign bundle file inventory is not exact")
    payloads = {
        name: verify_named_artifact(bundle, name, item_hash)
        for name, item_hash in artifacts.items()
    }
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
    if resolved.get("schema") != "sc-20686-resolved-inputs-v2" or not isinstance(
        resolved.get("coordinates"), list
    ):
        fail("campaign resolved inputs schema is invalid")
    resolved_by_key = {}
    for entry in resolved["coordinates"]:
        if not isinstance(entry, dict):
            fail("campaign resolved coordinate is malformed")
        key = (entry.get("family"), entry.get("variant"), entry.get("name"))
        if key in resolved_by_key:
            fail("campaign resolved inputs contain duplicate coordinates")
        for field in ("entrypoint_sha256", "snapshot_sha256", "route_manifest_sha256"):
            require_digest(entry.get(field), f"resolved {field}")
        if (
            not isinstance(entry.get("entrypoint"), str)
            or not isinstance(entry.get("snapshot"), str)
            or not isinstance(entry.get("args"), list)
            or any(not isinstance(item, str) for item in entry["args"])
            or not isinstance(entry.get("input_files"), list)
        ):
            fail("campaign resolved coordinate paths/arguments are malformed")
        manifest_document = {
            "route": entry["variant"],
            "coordinate": entry["name"],
            "entrypoint": entry["entrypoint"],
            "entrypoint_sha256": entry["entrypoint_sha256"],
            "snapshot": entry["snapshot"],
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
    for row, name in zip(sealed_rows, row_files):
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
            for suffix in (".stdout", ".stderr", ".command.json", ".process.json")
        }
        if any(len(names) != 1 for names in by_suffix.values()) or len(evidence_names) != 4:
            fail("row evidence artifact inventory is not the exact run transcript set")
        stdout = payloads[by_suffix[".stdout"][0]]
        try:
            observed = [
                json.loads(line)
                for line in stdout.decode("utf-8").splitlines()
                if line.lstrip().startswith("{")
            ]
            process = json.loads(payloads[by_suffix[".process.json"][0]].decode("utf-8"))
            command = json.loads(payloads[by_suffix[".command.json"][0]].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"campaign run transcript is malformed: {exc}")
        if not isinstance(process, dict) or not isinstance(process.get("samples"), list):
            fail("campaign process sample transcript is malformed")
        if observed + process["samples"] != row["observer_events"]:
            fail("sealed row events differ from the exact stdout/process transcripts")
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
        expected_argv = [
            resolved_entry["entrypoint"], "--sc20686-campaign", "--sc20686-events", "-",
            "--snapshot", resolved_entry["snapshot"], "--variant", row["variant"],
            *resolved_entry["args"],
        ]
        if row["arm"] == "cancel":
            expected_argv.append("--sc20686-cancel")
        if argv != expected_argv:
            fail("campaign command conflicts with sealed resolved inputs")
        if (
            row["route_manifest_sha256"] != resolved_entry["route_manifest_sha256"]
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

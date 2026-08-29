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
    "resolution", "reference_count", "frames", "prompt", "guidance", "layers",
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
        "group_size": 64,
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
        for field in ("activation", "creation", "reads", "release"):
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
    for key in ("reference_count", "frames", "layers", "heads", "head_dimension", "sq", "skv"):
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
    peak = max(sample["peak_bytes"] for sample in row["process_samples"])
    # The frozen threshold is whole-process occupancy: retained cache bytes and the peak live
    # cache-read workspace coexist at the read boundary, so neither may hide behind `max()`.
    current = row["current_persistent_bytes"] + row["current_read_transient_bytes"]
    candidate = row["candidate_persistent_bytes"] + row["candidate_read_transient_bytes"]
    saving = current - candidate
    saving_pct = saving / peak if peak else 0.0
    runtime_fraction = row["cache_read_duration_ms"] / row["generation_duration_ms"]
    opportunity = (
        current >= THRESHOLDS["opportunity_bytes"]
        and current >= peak * THRESHOLDS["opportunity_peak_pct"]
        and row["minimum_cache_reads"] >= THRESHOLDS["minimum_reads_per_cache"]
    )
    transient_compatible = (
        row["candidate_read_transient_bytes"] <= row["current_read_transient_bytes"]
    )
    eligible = (
        opportunity
        and transient_compatible
        and saving >= THRESHOLDS["saving_bytes"]
        and saving_pct >= THRESHOLDS["saving_peak_pct"]
    )
    return {
        "decision": "go" if eligible else "no-go",
        "opportunity": opportunity,
        "current_whole_process_bytes": current,
        "candidate_whole_process_bytes": candidate,
        "net_saving_bytes": saving,
        "net_saving_peak_pct": saving_pct,
        "cache_read_runtime_fraction": runtime_fraction,
        "runtime_only_opportunity": runtime_fraction >= THRESHOLDS["runtime_only_pct"],
        "minimum_cache_reads": row["minimum_cache_reads"],
        "transient_compatible": transient_compatible,
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
    payloads = {
        name: verify_named_artifact(bundle, name, item_hash)
        for name, item_hash in artifacts.items()
    }
    sealed_rows = campaign.get("rows")
    row_files = campaign.get("row_files")
    if not isinstance(sealed_rows, list) or not isinstance(row_files, list) or len(sealed_rows) != len(row_files):
        fail("campaign row inventory is malformed")
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
        try:
            variant = argv[argv.index("--variant") + 1]
        except (ValueError, IndexError):
            fail("campaign command lacks exact route identity")
        cancel = "--sc20686-cancel" in argv
        if (
            "--sc20686-campaign" not in argv
            or variant != row["variant"]
            or cancel != (row["arm"] == "cancel")
        ):
            fail("campaign command conflicts with the sealed row identity")
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
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args()
    try:
        if args.input.is_dir():
            if args.sidecar is not None:
                fail("bundle input does not accept --sidecar")
            result = verify_campaign_bundle(args.input)
        else:
            if args.sidecar is None:
                fail("standalone reduction requires an exact raw receipt sidecar or sealed bundle")
            raw = args.input.read_bytes()
            unsigned = json.loads(raw.decode("utf-8"))
            if not isinstance(unsigned, dict):
                fail("raw receipt input must contain one unsigned row")
            if unsigned.get("raw_receipt_sha256") != "" or unsigned.get("raw_receipt_sidecar_sha256") != "":
                fail("raw receipt must be unsigned")
            sidecar = args.sidecar.read_bytes()
            sealed = dict(unsigned)
            sealed["raw_receipt_sha256"] = sha256(raw)
            sealed["raw_receipt_sidecar_sha256"] = sha256(sidecar)
            verify_seal_artifact(sealed, raw, sidecar, args.input.name)
            result = reduce([sealed])
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_write(args.output, payload)
        print(json.dumps({"sha256": sha256(payload), "output": str(args.output)}))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"SC-20686 invalid receipt: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

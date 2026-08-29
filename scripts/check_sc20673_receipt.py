#!/usr/bin/env python3
"""Fail-closed semantic and checksum validator for SC-20673 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path


SOURCE_COMMIT = "54989ee223611627592f7f9bd925e924658f1f22"
INFERENCE_BASE = "3deb898c8dfa572e939ba9705adfe311dd6d43f0"
REQUIRED_AXES = (
    "B", "Hq", "Hkv", "GQA", "Sq", "Skv", "D", "group_size", "bits",
    "code_format", "dtype", "masks", "simd_groups", "tails_nonmultiples",
)
PROBE_NAMES = {
    "group_affine_decode", "rabitq_decode", "rabitq_prefill", "rvq_quant_pack",
}
POSITIVE_METRICS = {
    "host_graph_build_s", "first_eval_compile_and_dispatch_s", "async_submit_s",
    "explicit_synchronize_completion_s", "steady_dispatch_sync_median_s",
    "mlx_active_before_bytes", "mlx_active_after_first_bytes",
    "mlx_first_peak_bytes", "mlx_campaign_peak_bytes",
}
NONNEGATIVE_METRICS = {"compile_warmup_overhead_estimate_s", "mlx_peak_delta_bytes"}


def _load_json(path: Path, errors: list[str]) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.name}: {exc}")
        return {}


def _check_sidecar(path: Path, errors: list[str]) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        parts = sidecar.read_text(encoding="utf-8").split()
        if parts != [hashlib.sha256(path.read_bytes()).hexdigest(), path.name]:
            errors.append(f"checksum mismatch: {path.name}")
    except OSError as exc:
        errors.append(f"cannot verify {path.name}: {exc}")


def _derive_upstream_results(records: list[dict]) -> dict:
    text = {record.get("name"): record.get("stdout_tail", "") for record in records}
    parity = re.findall(
        r"\|\s*(512|2048|8192|16384)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x",
        text.get("parity", ""),
    )[:4]
    decode = re.findall(
        r"\|?\s*(512|2048|8192)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x",
        text.get("rabitq_decode_benchmark", ""),
    )
    prefill = re.findall(
        r"\s*(256|1024)\s+(2048|8192)\s+\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x",
        text.get("rabitq_prefill_benchmark", ""),
    )
    if len(parity) != 4 or len(decode) != 3 or len(prefill) != 3:
        return {}
    scalar_rows = {n: (float(base), float(fused), float(speed)) for n, base, fused, speed in parity}
    decode_rows = {
        n: (float(fused), float(base), float(speed)) for n, fused, _packed, base, speed in decode
    }
    prefill_rows = {
        f"{sq}x{skv}": (float(fused), float(base), float(speed))
        for sq, skv, fused, _decode, base, speed in prefill
    }
    return {
        "scalar_group_affine": {
            "baseline": "MLX dequantize + SDPA",
            "timings_ms": {n: [base, fused] for n, (base, fused, _speed) in scalar_rows.items()},
            "speedup": [speed for _base, _fused, speed in scalar_rows.values()],
        },
        "rabitq_decode": {
            "baseline": "dequantize + MLX SDPA",
            "timings_ms": {n: [fused, base] for n, (fused, base, _speed) in decode_rows.items()},
            "speedup": [speed for _fused, _base, speed in decode_rows.values()],
        },
        "rabitq_prefill": {
            "baseline": "dequantize + MLX SDPA",
            "timings_ms": {n: [fused, base] for n, (fused, base, _speed) in prefill_rows.items()},
            "speedup": [speed for _fused, _base, speed in prefill_rows.values()],
        },
    }


def _expected_physical_bytes(row: dict) -> dict:
    """Recompute the exact representative allocations from sealed geometry."""
    name = row.get("name")
    geometry = row.get("geometry", {})
    try:
        d = int(geometry["D"])
        if name == "rvq_quant_pack":
            n = int(geometry["N"])
            bits = int(geometry["bits"])
            if min(n, d, bits) <= 0 or bits != 2:
                return {}
            levels = 1 << bits
            words = -(-d // (32 // bits))
            stream = n * words * 4
            return {
                "dense_input_bytes": n * d * 2,
                "uint8_index_intermediates_avoided_bytes": 2 * n * d,
                "packed_stream_1_bytes": stream,
                "packed_stream_2_bytes": stream,
                "metadata_bytes": (levels + 2 * (levels - 1)) * 4,
                "compressed_output_bytes": 2 * stream,
                "output_bytes": 2 * stream,
            }

        b = int(geometry["B"])
        h = int(geometry["H"])
        sq = int(geometry["Sq"])
        skv = int(geometry["Skv"])
        if min(b, h, sq, skv, d) <= 0:
            return {}
        dense_k = b * h * skv * d * 2
        dense_v = dense_k
        output = b * h * sq * d * 2
        if name == "group_affine_decode":
            group = int(geometry["group_size"])
            if group <= 0:
                return {}
            key_groups = -(-skv // group)
            value_groups = -(-d // group)
            key_codes = b * h * skv * d
            key_metadata = 2 * b * h * key_groups * d * 4
            value_codes = b * h * skv * d
            value_metadata = 2 * b * h * skv * value_groups * 4
            return {
                "dense_key_bytes": dense_k,
                "dense_value_bytes": dense_v,
                "key_codes_bytes": key_codes,
                "key_scale_zero_bytes": key_metadata,
                "value_codes_bytes": value_codes,
                "value_scale_zero_bytes": value_metadata,
                "compressed_persistent_bytes": (
                    key_codes + key_metadata + value_codes + value_metadata
                ),
                "dense_reference_transient_bytes": dense_k + dense_v,
                "output_bytes": output,
            }
        if name not in {"rabitq_decode", "rabitq_prefill"}:
            return {}
        if geometry.get("packed_values") is not True or d % 8:
            return {}
        key_bits = b * h * skv * (d // 8)
        key_metadata = 2 * b * h * skv * 4
        packed_values = b * h * skv * (d // 2)
        centroids = 16 * 4
        return {
            "dense_key_bytes": dense_k,
            "dense_value_bytes": dense_v,
            "key_bits_bytes": key_bits,
            "key_magnitude_constant_bytes": key_metadata,
            "packed_values_bytes": packed_values,
            "value_centroids_bytes": centroids,
            "compressed_persistent_bytes": key_bits + key_metadata + packed_values + centroids,
            "dense_reference_transient_bytes": dense_k + dense_v,
            "output_bytes": output,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {}


def _valid_metric(value: object, *, positive: bool) -> bool:
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    return value > 0 if positive else value >= 0


def errors_for(root: Path) -> list[str]:
    errors: list[str] = []
    coverage_path = root / "sc-20673-coverage.json"
    receipt_path = root / "sc-20673-metal-reproduction.json"
    _check_sidecar(coverage_path, errors)
    _check_sidecar(receipt_path, errors)
    coverage = _load_json(coverage_path, errors)
    receipt = _load_json(receipt_path, errors)
    if errors:
        return errors

    if coverage.get("story") != "SC-20673" or receipt.get("story") != "SC-20673":
        errors.append("story identity mismatch")
    if receipt.get("schemaVersion") != 3:
        errors.append("raw receipt schema mismatch")
    if receipt.get("upstream", {}).get("commit") != SOURCE_COMMIT:
        errors.append("upstream commit mismatch")
    provenance = coverage.get("provenance", {})
    if provenance.get("upstream_commit") != SOURCE_COMMIT:
        errors.append("coverage upstream commit mismatch")
    if provenance.get("inference_base") != INFERENCE_BASE:
        errors.append("inference base mismatch")
    if receipt.get("provenance", {}).get("inference_base") != INFERENCE_BASE:
        errors.append("raw inference base mismatch")
    if provenance.get("dependency_manifest_sha256") != receipt.get("provenance", {}).get("dependency_manifest_sha256"):
        errors.append("dependency manifest identity mismatch")
    if provenance.get("host") != receipt.get("host"):
        errors.append("host or MLX version mismatch")
    if receipt.get("coverage") != coverage:
        errors.append("embedded coverage differs from sealed coverage")

    axes = coverage.get("axes", {})
    missing_axes = [key for key in REQUIRED_AXES if key not in axes]
    if missing_axes:
        errors.append(f"missing axes: {', '.join(missing_axes)}")
    if set(axes.get("D", {}).get("tested", [])) != {64, 128, 256}:
        errors.append("D=64/128/256 parity coverage is required")
    if axes.get("GQA") != [1]:
        errors.append("unsupported GQA must not be claimed as measured")
    if not axes.get("tails_nonmultiples"):
        errors.append("tail/nonmultiple coverage missing")

    probe = receipt.get("probe", {})
    if probe != coverage.get("probe") or probe.get("schemaVersion") != 2:
        errors.append("probe mismatch or schema drift")
    if not probe.get("deviceInfo"):
        errors.append("Metal device identity missing")
    probe_rows = probe.get("probes", [])
    if {row.get("name") for row in probe_rows} != PROBE_NAMES:
        errors.append("probe surface incomplete")
    for row in probe_rows:
        name = row.get("name", "unknown")
        metrics = row.get("metrics", {})
        for field in POSITIVE_METRICS:
            if not _valid_metric(metrics.get(field), positive=True):
                errors.append(f"{name}: invalid positive metric {field}")
        for field in NONNEGATIVE_METRICS:
            if not _valid_metric(metrics.get(field), positive=False):
                errors.append(f"{name}: invalid nonnegative metric {field}")
        physical = row.get("physical_bytes", {})
        expected_physical = _expected_physical_bytes(row)
        if not expected_physical or physical != expected_physical:
            errors.append(f"{name}: physical byte accounting does not match geometry")
        if physical != coverage.get("physical_bytes", {}).get(name):
            errors.append(f"{name}: physical bytes differ from coverage")
        if metrics != coverage.get("probe_results", {}).get(name):
            errors.append(f"{name}: probe metrics differ from coverage")

    records = receipt.get("upstream_benchmarks", [])
    if len(records) != 4 or any(record.get("returncode") != 0 for record in records):
        errors.append("upstream benchmark command failure or omission")
    parity_stdout = next(
        (record.get("stdout_tail", "") for record in records if record.get("name") == "parity"), ""
    )
    if "301 passed" not in parity_stdout:
        errors.append("301-test parity evidence missing")
    derived = _derive_upstream_results(records)
    if not derived or derived != coverage.get("upstream_results"):
        errors.append("upstream result rows do not derive from raw stdout")

    unsupported = {row.get("case") for row in coverage.get("unsupported", [])}
    for required in (
        "GQA ratio greater than 1",
        "causal, additive, sliding-window, sink, or softcap attention semantics",
        "RaBitQ prefill D=256 (kernel limit is D<=128)",
        "older Apple GPU",
    ):
        if required not in unsupported:
            errors.append(f"unsupported/fallback row missing: {required}")
    if "pending independent SceneWorks integration" not in receipt.get("product_eligibility", ""):
        errors.append("product eligibility boundary missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = (args.root or Path(__file__).parents[1] / "docs/architecture/receipts").resolve()
    errors = errors_for(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("SC-20673 receipt structure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

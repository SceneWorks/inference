#!/usr/bin/env python3
"""Verify a resident/staged real-weight A/B from MEMORY_EVIDENCE_V1 logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


PREFIX = "MEMORY_EVIDENCE_V1 "
MEMORY_EVIDENCE_SCHEMA_VERSION = 2
GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
KEBAB_TOKEN = re.compile(r"[a-z0-9]+\Z")
VERSION_TOKEN = re.compile(r"v[1-9][0-9]*\Z")
ANY_VERSION_TOKEN = re.compile(r"v[0-9]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
STRATEGIES = {
    "resident",
    "staged_residency",
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
}
STRATEGY_ORDER = {
    "resident": 0,
    "staged_residency": 1,
    "bounded_decode": 2,
    "bounded_attention": 3,
    "bounded_transformer_residency": 4,
}
LOAD_SHAPES = {"eager_materialization", "deferred_materialization"}
# Tolerance metrics this verifier can recompute from the two bound output artifacts. A lane may
# only declare an expected tolerance over one of these (sc-18149).
SUPPORTED_TOLERANCE_METRICS = {"mean_abs_u8_subpixel"}
BACKENDS = {"candle", "mlx"}
PRECISIONS = {"bf16", "fp32"}
QUANTS = {None, "q4", "q8", "nvfp4"}
WINDOW_COMPONENTS = {None, "dit", "text_encoder", "both"}

TOP_KEYS = {
    "schema_version",
    "key",
    "declared_calibration",
    "observed_calibration",
    "predicted_peak_bytes",
    "observed_peak_bytes",
    "inference_revision",
    "sceneworks_revision",
    "model_revision",
    "model_inventory_sha256",
    "harness_version",
    "output_sha256",
    "parity",
    "parity_result",
}
EVIDENCE_KEY_KEYS = {
    "model_family",
    "resolved_route",
    "backend",
    "tier",
    "load_shape",
    "mode",
    "reference_shape",
    "overlay",
    "geometry",
    "frames_per_second",
    "strategy",
    "engaged_composition",
    "parameters",
}


@dataclass(frozen=True)
class EvidenceRecord:
    path: Path
    payload: dict[str, Any]

    @property
    def key(self) -> dict[str, Any]:
        return self.payload["key"]

    @property
    def observed_peak_bytes(self) -> int:
        return self.payload["observed_peak_bytes"]


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise RuntimeError(f"{label} keys differ: missing={missing}, extra={extra}")


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int_or_null(value: Any, label: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise RuntimeError(f"{label} must be a non-negative integer or null")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _validate_fingerprint(value: Any, label: str) -> str:
    fingerprint = _require_string(value, label)
    tokens = fingerprint.split("-")
    if any(KEBAB_TOKEN.fullmatch(token) is None for token in tokens):
        raise RuntimeError(f"{label} must contain lowercase ASCII kebab tokens")
    versions = [token for token in tokens if ANY_VERSION_TOKEN.fullmatch(token)]
    if len(versions) != 1:
        raise RuntimeError(f"{label} must contain exactly one positive vN token")
    if VERSION_TOKEN.fullmatch(versions[0]) is None:
        raise RuntimeError(f"{label} must contain exactly one positive vN token")
    return fingerprint


def _validate_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _require_object(value, label)
    _require_exact_keys(identity, {"abi", "fingerprint", "load_shape"}, label)
    _require_positive_int(identity["abi"], f"{label}.abi")
    _validate_fingerprint(identity["fingerprint"], f"{label}.fingerprint")
    if identity["load_shape"] not in LOAD_SHAPES:
        raise RuntimeError(f"{label}.load_shape is not canonical")
    return identity


def _validate_tier(value: Any) -> None:
    tier = _require_object(value, "key.tier")
    _require_exact_keys(
        tier, {"precision", "quant", "component_precision_floors"}, "key.tier"
    )
    if tier["precision"] not in PRECISIONS:
        raise RuntimeError("key.tier.precision is not canonical")
    if tier["quant"] not in QUANTS:
        raise RuntimeError("key.tier.quant is not canonical")
    floors = tier["component_precision_floors"]
    if not isinstance(floors, list):
        raise RuntimeError("key.tier.component_precision_floors must be an array")
    previous: tuple[str, str, str] | None = None
    for index, item in enumerate(floors):
        floor = _require_object(item, f"key.tier.component_precision_floors[{index}]")
        _require_exact_keys(
            floor,
            {"component", "selected_tier", "resident_tier"},
            f"key.tier.component_precision_floors[{index}]",
        )
        if floor["component"] not in {"textEncoder", "transformerHead"}:
            raise RuntimeError(f"component precision floor {index} has an unknown component")
        if floor["selected_tier"] not in QUANTS - {None}:
            raise RuntimeError(f"component precision floor {index} has an unknown selected tier")
        if floor["resident_tier"] not in QUANTS - {None}:
            raise RuntimeError(f"component precision floor {index} has an unknown resident tier")
        current = (floor["component"], floor["selected_tier"], floor["resident_tier"])
        if previous is not None and current <= previous:
            raise RuntimeError("component precision floors must be strictly canonical")
        previous = current


def _validate_geometry(value: Any) -> None:
    geometry = _require_object(value, "key.geometry")
    _require_exact_keys(
        geometry, {"width", "height", "batch", "frames", "reference_count"}, "key.geometry"
    )
    for field in ("width", "height", "batch", "frames"):
        _require_positive_int(geometry[field], f"key.geometry.{field}")
    if type(geometry["reference_count"]) is not int or geometry["reference_count"] < 0:
        raise RuntimeError("key.geometry.reference_count must be a non-negative integer")


def _validate_parameters(value: Any) -> None:
    parameters = _require_object(value, "key.parameters")
    _require_exact_keys(
        parameters,
        {
            "decode_tile_edge",
            "decode_overlap",
            "attention_chunk_size",
            "transformer_window_size",
            "transformer_window_component",
        },
        "key.parameters",
    )
    for field in (
        "decode_tile_edge",
        "decode_overlap",
        "attention_chunk_size",
        "transformer_window_size",
    ):
        _require_nonnegative_int_or_null(parameters[field], f"key.parameters.{field}")
    if parameters["transformer_window_component"] not in WINDOW_COMPONENTS:
        raise RuntimeError("key.parameters.transformer_window_component is not canonical")


def _validate_parity(payload: dict[str, Any]) -> None:
    parity = _require_object(payload["parity"], "parity")
    kind = parity.get("kind")
    if kind == "exact":
        _require_exact_keys(parity, {"kind"}, "parity")
    elif kind == "tolerance":
        _require_exact_keys(parity, {"kind", "metric", "maximum_error"}, "parity")
        _require_string(parity["metric"], "parity.metric")
        if (
            type(parity["maximum_error"]) not in (int, float)
            or not math.isfinite(parity["maximum_error"])
            or parity["maximum_error"] < 0
        ):
            raise RuntimeError("parity.maximum_error must be finite and non-negative")
    elif kind == "golden":
        _require_exact_keys(
            parity, {"kind", "fixture", "metric", "maximum_error"}, "parity"
        )
        _require_string(parity["fixture"], "parity.fixture")
        _require_string(parity["metric"], "parity.metric")
        if (
            type(parity["maximum_error"]) not in (int, float)
            or not math.isfinite(parity["maximum_error"])
            or parity["maximum_error"] < 0
        ):
            raise RuntimeError("parity.maximum_error must be finite and non-negative")
    else:
        raise RuntimeError("parity.kind is not canonical")

    result = _require_object(payload["parity_result"], "parity_result")
    _require_exact_keys(result, {"kind"}, "parity_result")
    if result["kind"] not in {"not_run", "passed"}:
        raise RuntimeError("parity_result.kind must be not_run or passed")


def _validate_payload(payload: Any, expected_strategy: str | None) -> dict[str, Any]:
    payload = _require_object(payload, "record")
    _require_exact_keys(payload, TOP_KEYS, "record")
    if payload["schema_version"] != MEMORY_EVIDENCE_SCHEMA_VERSION:
        raise RuntimeError(f"schema_version must be {MEMORY_EVIDENCE_SCHEMA_VERSION}")
    key = _require_object(payload["key"], "key")
    _require_exact_keys(key, EVIDENCE_KEY_KEYS, "key")
    _require_string(key["model_family"], "key.model_family")
    _require_string(key["resolved_route"], "key.resolved_route")
    if key["backend"] not in BACKENDS:
        raise RuntimeError("key.backend is not canonical")
    _validate_tier(key["tier"])
    if key["load_shape"] not in LOAD_SHAPES:
        raise RuntimeError("key.load_shape is not canonical")
    _require_string(key["mode"], "key.mode")
    reference_shape = _require_string(key["reference_shape"], "key.reference_shape")
    if key["overlay"] is not None:
        _require_string(key["overlay"], "key.overlay")
    _validate_geometry(key["geometry"])
    if (reference_shape == "none") != (key["geometry"]["reference_count"] == 0):
        raise RuntimeError("key.reference_shape must be none exactly when reference_count is zero")
    if key["frames_per_second"] is not None:
        _require_positive_int(key["frames_per_second"], "key.frames_per_second")
    if expected_strategy is not None and key["strategy"] != expected_strategy:
        raise RuntimeError(
            f"expected key.strategy={expected_strategy}, found {key['strategy']}"
        )
    if key["strategy"] not in STRATEGIES:
        raise RuntimeError("key.strategy is not canonical")
    strategy = key["strategy"]
    composition = key["engaged_composition"]
    if not isinstance(composition, list) or not composition:
        raise RuntimeError("key.engaged_composition must be a non-empty array")
    if any(item not in STRATEGIES for item in composition):
        raise RuntimeError("key.engaged_composition contains a non-canonical strategy")
    order = [STRATEGY_ORDER[item] for item in composition]
    if order != sorted(set(order)):
        raise RuntimeError("key.engaged_composition must be unique and canonically ordered")
    if composition[0] != "resident" or composition[-1] != strategy:
        raise RuntimeError(
            "key.engaged_composition must start with resident and end with key.strategy"
        )
    _validate_parameters(key["parameters"])
    if strategy in {"resident", "staged_residency"} and any(
        value is not None for value in key["parameters"].values()
    ):
        raise RuntimeError(f"key.parameters must be empty for {strategy}")

    declared = _validate_identity(payload["declared_calibration"], "declared_calibration")
    observed = _validate_identity(payload["observed_calibration"], "observed_calibration")
    if declared != observed:
        raise RuntimeError("declared and observed calibration identities differ")
    if declared["load_shape"] != key["load_shape"]:
        raise RuntimeError("calibration load shape differs from the evidence key")
    _require_positive_int(payload["predicted_peak_bytes"], "predicted_peak_bytes")
    _require_positive_int(payload["observed_peak_bytes"], "observed_peak_bytes")
    for field in ("inference_revision", "sceneworks_revision", "model_revision"):
        if not isinstance(payload[field], str) or GIT_REVISION.fullmatch(payload[field]) is None:
            raise RuntimeError(f"{field} must be an exact lowercase 40-character Git commit")
    if not isinstance(payload["model_inventory_sha256"], str) or SHA256.fullmatch(
        payload["model_inventory_sha256"]
    ) is None:
        raise RuntimeError("model_inventory_sha256 must be 64 lowercase hexadecimal characters")
    _require_string(payload["harness_version"], "harness_version")
    if not isinstance(payload["output_sha256"], str) or SHA256.fullmatch(
        payload["output_sha256"]
    ) is None:
        raise RuntimeError("output_sha256 must be 64 lowercase hexadecimal characters")
    _validate_parity(payload)
    return payload


def read_record(path: Path, expected_strategy: str) -> EvidenceRecord:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if any("SEQ_AB" in line for line in raw_lines):
        raise RuntimeError(f"{path}: legacy SEQ_AB evidence is forbidden")
    lines = [
        line[len(PREFIX) :]
        for line in raw_lines
        if line.startswith(PREFIX)
    ]
    matching: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_nonfinite_json,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"{path}: malformed MEMORY_EVIDENCE_V1 JSON: {error}") from error
        strategy = (
            payload.get("key", {}).get("strategy")
            if isinstance(payload, dict) and isinstance(payload.get("key"), dict)
            else None
        )
        try:
            payload = _validate_payload(payload, strategy)
        except RuntimeError as error:
            raise RuntimeError(f"{path}: {error}") from error
        if strategy == expected_strategy:
            matching.append(payload)
    if len(matching) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one {PREFIX.strip()} record for strategy "
            f"{expected_strategy}, found {len(matching)}"
        )
    return EvidenceRecord(path=path, payload=matching[0])


def _invariant_projection(record: EvidenceRecord) -> dict[str, Any]:
    key = record.key
    return {
        "model_family": key["model_family"],
        "resolved_route": key["resolved_route"],
        "backend": key["backend"],
        "tier": key["tier"],
        "load_shape": key["load_shape"],
        "mode": key["mode"],
        "reference_shape": key["reference_shape"],
        "overlay": key["overlay"],
        "geometry": key["geometry"],
        "frames_per_second": key["frames_per_second"],
        "declared_calibration": record.payload["declared_calibration"],
        "observed_calibration": record.payload["observed_calibration"],
        "inference_revision": record.payload["inference_revision"],
        "sceneworks_revision": record.payload["sceneworks_revision"],
        "model_revision": record.payload["model_revision"],
        "model_inventory_sha256": record.payload["model_inventory_sha256"],
        "harness_version": record.payload["harness_version"],
        "parity": record.payload["parity"],
        "parity_result": record.payload["parity_result"],
    }


def parse_expected_parity(value: str) -> dict[str, Any]:
    """Parse the lane's declared parity expectation.

    ``exact`` (the default) or ``tolerance:<metric>:<maximum_error>`` — the lane pins the exact
    contract the records must carry, so a harness cannot loosen its own bar by emitting a
    self-serving contract (sc-18149). Only metrics this verifier can recompute from the two output
    artifacts are accepted.
    """
    if value == "exact":
        return {"kind": "exact"}
    parts = value.split(":")
    if len(parts) == 3 and parts[0] == "tolerance":
        _, metric, maximum_error_text = parts
        if metric not in SUPPORTED_TOLERANCE_METRICS:
            raise RuntimeError(
                f"expected-parity tolerance metric {metric!r} is not recomputable by this verifier"
            )
        try:
            maximum_error = float(maximum_error_text)
        except ValueError as error:
            raise RuntimeError(
                "expected-parity tolerance maximum_error must be a number"
            ) from error
        if not math.isfinite(maximum_error) or maximum_error < 0:
            raise RuntimeError(
                "expected-parity tolerance maximum_error must be finite and non-negative"
            )
        return {"kind": "tolerance", "metric": metric, "maximum_error": maximum_error}
    raise RuntimeError(
        "expected-parity must be 'exact' or 'tolerance:<metric>:<maximum_error>'"
    )


def _mean_abs_u8_subpixel(resident_bytes: bytes, sequential_bytes: bytes) -> float:
    if len(resident_bytes) != len(sequential_bytes):
        raise RuntimeError(
            "resident and staged outputs differ in length; the drift metric is undefined"
        )
    if not resident_bytes:
        raise RuntimeError("outputs are empty; the drift metric is undefined")
    total = sum(abs(a - b) for a, b in zip(resident_bytes, sequential_bytes))
    return total / len(resident_bytes)


def _p99_abs_u8_subpixel(resident_bytes: bytes, sequential_bytes: bytes) -> int:
    """The exact 99th-percentile absolute u8 subpixel delta, from a 256-bin histogram.

    The tail companion to the mean ceiling (sc-18149): a pathological redistribution of the same
    mean — a few huge deltas hiding under many zeros — moves this quantile while leaving the mean
    within its ceiling, so the lane pins both from outside the harness.
    """
    if len(resident_bytes) != len(sequential_bytes):
        raise RuntimeError(
            "resident and staged outputs differ in length; the drift metric is undefined"
        )
    if not resident_bytes:
        raise RuntimeError("outputs are empty; the drift metric is undefined")
    histogram = [0] * 256
    for a, b in zip(resident_bytes, sequential_bytes):
        histogram[abs(a - b)] += 1
    need = math.ceil(0.99 * len(resident_bytes))
    seen = 0
    for delta, count in enumerate(histogram):
        seen += count
        if seen >= need:
            return delta
    return 255


def verify(
    resident_log: Path,
    sequential_log: Path,
    min_reduction_mib: int,
    expected_route: str,
    expected_fingerprint: str,
    expected_abi: int,
    expected_model_revision: str,
    expected_model_inventory_sha256: str,
    resident_output: Path,
    sequential_output: Path,
    expected_parity: dict[str, Any],
    isolator_output: Path | None = None,
    max_p99_abs_u8: int | None = None,
) -> tuple[int, int, float | None, int | None]:
    if min_reduction_mib < 0:
        raise RuntimeError("minimum reduction MiB must be non-negative")
    if GIT_REVISION.fullmatch(expected_model_revision) is None:
        raise RuntimeError("expected model revision must be an exact lowercase 40-character commit")
    if SHA256.fullmatch(expected_model_inventory_sha256) is None:
        raise RuntimeError("expected model inventory SHA-256 must be 64 lowercase hexadecimal characters")
    resident = read_record(resident_log, "resident")
    sequential = read_record(sequential_log, "staged_residency")
    if _invariant_projection(resident) != _invariant_projection(sequential):
        raise RuntimeError("resident and staged records differ on a non-strategy A/B invariant")
    resident_bytes = resident_output.read_bytes()
    sequential_bytes = sequential_output.read_bytes()
    resident_hash = hashlib.sha256(resident_bytes).hexdigest()
    sequential_hash = hashlib.sha256(sequential_bytes).hexdigest()
    if resident_hash != resident.payload["output_sha256"]:
        raise RuntimeError("resident output SHA-256 does not match its evidence record")
    if sequential_hash != sequential.payload["output_sha256"]:
        raise RuntimeError("staged output SHA-256 does not match its evidence record")
    if resident.payload["parity"] != expected_parity:
        raise RuntimeError(
            "record parity contract does not match the parity contract this lane declares: "
            f"expected {expected_parity!r}, found {resident.payload['parity']!r}"
        )
    if isolator_output is not None:
        # The isolator leg (sc-18149): a Resident render forced onto the Sequential route's tiled
        # decode. Byte-identity with the staged output attributes the whole declared drift to the
        # tiled decode and proves residency staging itself is still numerically exact — checked
        # HERE, from the persisted artifact, so the exactness claim is enforced by this verifier
        # rather than trusted from the harness.
        if not isolator_output.is_file():
            raise RuntimeError(
                "isolator output artifact is missing — the harness did not persist the "
                "resident+tiled isolator leg"
            )
        isolator_hash = hashlib.sha256(isolator_output.read_bytes()).hexdigest()
        if isolator_hash != sequential_hash:
            raise RuntimeError(
                "staged output is not byte-identical to the resident+tiled isolator — residency "
                "staging itself drifted, which the decode-drift tolerance must not absorb"
            )
    drift: float | None = None
    if expected_parity["kind"] == "exact":
        if resident_hash != sequential_hash:
            raise RuntimeError("resident and staged outputs violate exact parity")
    else:
        # Tolerance: recompute the declared metric from the bound artifacts themselves, so the
        # ceiling is enforced by this verifier rather than trusted from the harness (sc-18149).
        drift = _mean_abs_u8_subpixel(resident_bytes, sequential_bytes)
        if drift > expected_parity["maximum_error"]:
            raise RuntimeError(
                f"resident and staged outputs drift {drift:.4f} mean_abs_u8_subpixel, above the "
                f"declared tolerance {expected_parity['maximum_error']}"
            )
    p99: int | None = None
    if max_p99_abs_u8 is not None:
        if max_p99_abs_u8 < 0:
            raise RuntimeError("maximum p99 absolute u8 delta must be non-negative")
        p99 = _p99_abs_u8_subpixel(resident_bytes, sequential_bytes)
        if p99 > max_p99_abs_u8:
            raise RuntimeError(
                f"resident and staged outputs drift p99 {p99} abs u8, above the declared "
                f"tail pin {max_p99_abs_u8}"
            )
    if resident.key["resolved_route"] != expected_route:
        raise RuntimeError(
            f"expected route {expected_route!r}, found {resident.key['resolved_route']!r}"
        )
    identity = resident.payload["declared_calibration"]
    if identity["fingerprint"] != expected_fingerprint:
        raise RuntimeError(
            "record fingerprint does not match the provider's exported calibration fingerprint: "
            f"expected {expected_fingerprint!r}, found {identity['fingerprint']!r}"
        )
    if identity["abi"] != expected_abi:
        raise RuntimeError(
            f"record calibration ABI does not match the exported ABI: expected {expected_abi}, "
            f"found {identity['abi']}"
        )
    if resident.payload["model_revision"] != expected_model_revision:
        raise RuntimeError(
            "record model revision does not match the verified snapshot revision: "
            f"expected {expected_model_revision!r}, found {resident.payload['model_revision']!r}"
        )
    if resident.payload["model_inventory_sha256"] != expected_model_inventory_sha256:
        raise RuntimeError(
            "record model inventory SHA-256 does not match the verified snapshot inventory: "
            f"expected {expected_model_inventory_sha256!r}, "
            f"found {resident.payload['model_inventory_sha256']!r}"
        )
    resident_peak = resident.observed_peak_bytes
    sequential_peak = sequential.observed_peak_bytes
    reduction = resident_peak - sequential_peak
    minimum_bytes = min_reduction_mib * 1024 * 1024
    if reduction < minimum_bytes:
        raise RuntimeError(
            f"staged residency reduced peak by {reduction} bytes; required at least {minimum_bytes} "
            f"bytes (resident={resident_peak}, staged={sequential_peak})"
        )
    return resident_peak, sequential_peak, drift, p99


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resident", required=True, type=Path)
    parser.add_argument("--sequential", required=True, type=Path)
    parser.add_argument("--min-reduction-mib", required=True, type=int)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--expected-abi", required=True, type=int)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--expected-model-inventory-sha256", required=True)
    parser.add_argument("--resident-output", required=True, type=Path)
    parser.add_argument("--sequential-output", required=True, type=Path)
    parser.add_argument("--expected-parity", default="exact")
    parser.add_argument("--isolator-output", type=Path, default=None)
    parser.add_argument("--max-p99-abs-u8", type=int, default=None)
    args = parser.parse_args()
    resident, sequential, drift, p99 = verify(
        args.resident,
        args.sequential,
        args.min_reduction_mib,
        args.model,
        args.expected_fingerprint,
        args.expected_abi,
        args.expected_model_revision,
        args.expected_model_inventory_sha256,
        args.resident_output,
        args.sequential_output,
        parse_expected_parity(args.expected_parity),
        isolator_output=args.isolator_output,
        max_p99_abs_u8=args.max_p99_abs_u8,
    )
    drift_suffix = "" if drift is None else f" drift_mean_abs_u8={drift:.4f}"
    p99_suffix = "" if p99 is None else f" p99_abs_u8={p99}"
    print(
        f"MEMORY_EVIDENCE_V1_RESULT model={args.model} verdict=pass "
        f"resident_peak_bytes={resident} staged_peak_bytes={sequential} "
        f"reduction_bytes={resident - sequential}{drift_suffix}{p99_suffix}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"MEMORY_EVIDENCE_V1_RESULT verdict=fail error={error}")
        raise SystemExit(1)

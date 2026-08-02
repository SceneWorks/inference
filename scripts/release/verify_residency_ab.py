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
    "resolved_route",
    "backend",
    "tier",
    "load_shape",
    "mode",
    "overlay",
    "geometry",
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
    if payload["schema_version"] != 1:
        raise RuntimeError("schema_version must be 1")
    key = _require_object(payload["key"], "key")
    _require_exact_keys(key, EVIDENCE_KEY_KEYS, "key")
    _require_string(key["resolved_route"], "key.resolved_route")
    if key["backend"] not in BACKENDS:
        raise RuntimeError("key.backend is not canonical")
    _validate_tier(key["tier"])
    if key["load_shape"] not in LOAD_SHAPES:
        raise RuntimeError("key.load_shape is not canonical")
    _require_string(key["mode"], "key.mode")
    if key["overlay"] is not None:
        _require_string(key["overlay"], "key.overlay")
    _validate_geometry(key["geometry"])
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
        "resolved_route": key["resolved_route"],
        "backend": key["backend"],
        "tier": key["tier"],
        "load_shape": key["load_shape"],
        "mode": key["mode"],
        "overlay": key["overlay"],
        "geometry": key["geometry"],
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
) -> tuple[int, int]:
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
    resident_hash = hashlib.sha256(resident_output.read_bytes()).hexdigest()
    sequential_hash = hashlib.sha256(sequential_output.read_bytes()).hexdigest()
    if resident_hash != resident.payload["output_sha256"]:
        raise RuntimeError("resident output SHA-256 does not match its evidence record")
    if sequential_hash != sequential.payload["output_sha256"]:
        raise RuntimeError("staged output SHA-256 does not match its evidence record")
    if resident_hash != sequential_hash:
        raise RuntimeError("resident and staged outputs violate exact parity")
    if resident.payload["parity"] != {"kind": "exact"}:
        raise RuntimeError("residency A/B requires an exact parity contract")
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
    return resident_peak, sequential_peak


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
    args = parser.parse_args()
    resident, sequential = verify(
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
    )
    print(
        f"MEMORY_EVIDENCE_V1_RESULT model={args.model} verdict=pass "
        f"resident_peak_bytes={resident} staged_peak_bytes={sequential} "
        f"reduction_bytes={resident - sequential}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"MEMORY_EVIDENCE_V1_RESULT verdict=fail error={error}")
        raise SystemExit(1)

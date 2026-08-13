#!/usr/bin/env python3
"""Seal correctness-only production-latent decode receipts into ABI-1 policy rows.

The collector deliberately accepts an exact semantic allowlist. A receipt containing timing,
memory, allocator, footprint, or any other field is rejected rather than accidentally promoted into
the quality artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MARKER = "DECODE_QUALITY_V1 "
QUALITY_ABI = 1
REQUIRED_KEYS = {
    "family",
    "resolvedRoute",
    "backend",
    "tier",
    "mode",
    "overlay",
    "geometry",
    "usePid",
    "tileEdge",
    "overlap",
    "metric",
    "maximumError",
    "seed",
    "productionLatentProvenance",
    "productionLatentSha256",
    "denseOutputSha256",
    "tiledOutputSha256",
    "observedError",
}
COORDINATE_KEYS = (
    "family",
    "resolvedRoute",
    "backend",
    "tier",
    "mode",
    "overlay",
    "geometry",
    "usePid",
    "tileEdge",
    "overlap",
    "metric",
    "maximumError",
)
HASH_FIELDS = (
    "productionLatentSha256",
    "denseOutputSha256",
    "tiledOutputSha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_receipt(receipt: dict[str, Any]) -> None:
    _require(
        set(receipt) == REQUIRED_KEYS,
        f"quality receipt fields differ from the semantic allowlist: "
        f"missing={sorted(REQUIRED_KEYS - set(receipt))} extra={sorted(set(receipt) - REQUIRED_KEYS)}",
    )
    for field in ("family", "resolvedRoute", "mode", "metric", "productionLatentProvenance"):
        _require(
            isinstance(receipt[field], str) and bool(receipt[field].strip()),
            f"{field} must be a nonempty string",
        )
    _require(receipt["backend"] == "mlx", "decode-quality admission currently supports MLX only")
    _require(receipt["tier"] in {"bf16", "q4", "q8", "nvfp4", "fp32"}, "unsupported tier")
    _require(
        receipt["overlay"] is None
        or (
            isinstance(receipt["overlay"], str)
            and bool(receipt["overlay"].strip())
            and "=" not in receipt["overlay"]
        ),
        "overlay must be null or a nonempty identity axis",
    )
    _require(type(receipt["usePid"]) is bool, "usePid must be a boolean")
    geometry = receipt["geometry"]
    _require(
        isinstance(geometry, dict)
        and set(geometry) == {"width", "height", "batch", "frames", "referenceCount"},
        "geometry must use the exact ABI-1 axes",
    )
    _require(
        all(type(geometry[key]) is int and geometry[key] > 0 for key in ("width", "height", "batch", "frames")),
        "geometry dimensions must be positive integers",
    )
    _require(
        type(geometry["referenceCount"]) is int and geometry["referenceCount"] >= 0,
        "referenceCount must be a nonnegative integer",
    )
    _require(
        type(receipt["tileEdge"]) is int
        and type(receipt["overlap"]) is int
        and 0 < receipt["overlap"] < receipt["tileEdge"],
        "tileEdge must exceed a positive overlap",
    )
    for field in HASH_FIELDS:
        value = receipt[field]
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{field} must be lowercase SHA-256",
        )
    _require(type(receipt["seed"]) is int and receipt["seed"] >= 0, "seed must be nonnegative")
    _require(
        type(receipt["maximumError"]) is int and receipt["maximumError"] >= 0,
        "maximumError must be nonnegative",
    )
    _require(
        type(receipt["observedError"]) is int and receipt["observedError"] >= 0,
        "observedError must be nonnegative",
    )


def read_receipts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            marker_at = line.find(MARKER)
            if marker_at < 0:
                continue
            try:
                receipt = json.loads(line[marker_at + len(MARKER) :])
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed quality receipt: {error}") from error
            _require(isinstance(receipt, dict), f"{path}:{line_number}: receipt must be an object")
            _validate_receipt(receipt)
            receipts.append(receipt)
    _require(bool(receipts), "no DECODE_QUALITY_V1 receipts found")
    return receipts


def _coordinate(receipt: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        json.dumps(receipt[key], sort_keys=True, separators=(",", ":"))
        for key in COORDINATE_KEYS
    )


def _canonical_tier(tier: str) -> dict[str, Any]:
    return {
        "precision": "fp32" if tier == "fp32" else "bf16",
        "quant": tier if tier in {"q4", "q8", "nvfp4"} else None,
        "component_precision_floors": [],
    }


def _canonical_policy(policy: dict[str, Any]) -> dict[str, Any]:
    geometry = policy["geometry"]
    fixtures = [
        {
            "seed": fixture["seed"],
            "production_latent_provenance": fixture["productionLatentProvenance"],
            "production_latent_sha256": fixture["productionLatentSha256"],
            "dense_output_sha256": fixture["denseOutputSha256"],
            "tiled_output_sha256": fixture["tiledOutputSha256"],
            "observed_error": fixture["observedError"],
        }
        for fixture in policy["fixtures"]
    ]
    return {
        "quality_abi": policy["qualityAbi"],
        "family": policy["family"],
        "resolved_route": policy["resolvedRoute"],
        "backend": policy["backend"],
        "tier": _canonical_tier(policy["tier"]),
        "mode": policy["mode"],
        "overlay": policy["overlay"],
        "geometry": {
            "width": geometry["width"],
            "height": geometry["height"],
            "batch": geometry["batch"],
            "frames": geometry["frames"],
            "reference_count": geometry["referenceCount"],
        },
        "use_pid": policy["usePid"],
        "tile_edge": policy["tileEdge"],
        "overlap": policy["overlap"],
        "metric": policy["metric"],
        "maximum_error": policy["maximumError"],
        "fixtures": fixtures,
        "disposition": policy["disposition"],
    }


def seal(receipts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        grouped[_coordinate(receipt)].append(receipt)

    policies: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row["seed"])
        seeds = [row["seed"] for row in rows]
        _require(len(rows) >= 2, "each quality coordinate needs at least two fixed seeds")
        _require(len(seeds) == len(set(seeds)), f"duplicate seed in quality coordinate: {seeds}")
        first = rows[0]
        failures = [row for row in rows if row["observedError"] > row["maximumError"]]
        disposition: dict[str, str] = {"kind": "admitted"}
        if failures:
            detail = ", ".join(f"seed {row['seed']}={row['observedError']}" for row in failures)
            disposition = {
                "kind": "refused",
                "reason": f"{first['metric']} exceeded {first['maximumError']}: {detail}",
            }
        fixtures = [
            {
                "seed": row["seed"],
                "productionLatentProvenance": row["productionLatentProvenance"],
                "productionLatentSha256": row["productionLatentSha256"],
                "denseOutputSha256": row["denseOutputSha256"],
                "tiledOutputSha256": row["tiledOutputSha256"],
                "observedError": row["observedError"],
            }
            for row in rows
        ]
        policy = {
            "qualityAbi": QUALITY_ABI,
            **{key: first[key] for key in COORDINATE_KEYS},
            "fixtures": fixtures,
            "productionEvidenceSha256": "",
            "disposition": disposition,
        }
        canonical = json.dumps(
            _canonical_policy(policy), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        policy["productionEvidenceSha256"] = hashlib.sha256(canonical).hexdigest()
        policies.append(policy)

    policies.sort(
        key=lambda policy: (
            policy["family"],
            policy["resolvedRoute"],
            policy["tier"],
            policy["mode"],
            policy["overlay"] or "",
            policy["geometry"]["width"],
            policy["geometry"]["height"],
            policy["tileEdge"],
            policy["overlap"],
        )
    )
    return policies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = seal(read_receipts(args.input))
    args.output.write_text(json.dumps(policies, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sealed {len(policies)} decode-quality policy row(s) from correctness-only receipts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

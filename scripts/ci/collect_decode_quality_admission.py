#!/usr/bin/env python3
"""Seal correctness-only production-latent decode receipts into ABI-2 policy rows.

The collector deliberately accepts an exact semantic allowlist. A receipt containing timing,
memory, allocator, footprint, or any other field is rejected rather than accidentally promoted into
the quality artifact. Geometry-isolated workflow artifacts use collision-free
``decode-quality-v2-*.log`` names; ``--input-root`` recursively enumerates those names in lexical
order and seals the complete downloaded route/geometry corpus deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MARKER = "DECODE_QUALITY_V2 "
QUALITY_ABI = 2
REQUIRED_KEYS = {
    "family",
    "resolvedRoute",
    "backend",
    "tier",
    "loadShape",
    "artifact",
    "implementationFingerprint",
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
    "loadShape",
    "artifact",
    "implementationFingerprint",
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
        receipt["loadShape"] in {"eager_materialization", "deferred_materialization"},
        "unsupported loadShape",
    )
    artifact = receipt["artifact"]
    _require(
        isinstance(artifact, dict)
        and set(artifact) == {"repository", "revision", "variant", "fingerprint"},
        "artifact must use the exact ABI-2 identity axes",
    )
    for field in ("repository", "variant", "fingerprint"):
        _require(
            isinstance(artifact[field], str) and bool(artifact[field].strip()),
            f"artifact.{field} must be a nonempty string",
        )
    _require(
        isinstance(artifact["revision"], str)
        and len(artifact["revision"]) == 40
        and all(character in "0123456789abcdef" for character in artifact["revision"]),
        "artifact.revision must be an exact lowercase 40-hex revision",
    )
    implementation = receipt["implementationFingerprint"]
    _require(
        isinstance(implementation, str)
        and len(implementation) == 64
        and all(character in "0123456789abcdef" for character in implementation),
        "implementationFingerprint must be lowercase SHA-256",
    )
    # Shape only. sc-19728 removed the comparison that stood here — originally against the constant
    # embedded in gen-core, and it must not come back as a comparison against a freshly derived
    # closure either. That would be the same over-broad currency gate wearing a different hat: the
    # closure hashes `Cargo.lock`, the workflow file, and whole `src` trees, so it refuses any
    # receipt not produced by a byte-identical tree. `--input-root` exists precisely to seal a
    # corpus assembled from artifacts of *different* runs (the 69 shipped rows span three separate
    # dispatches), and each cell's stamp is sealed into its own `productionEvidenceSha256`, so a
    # heterogeneous corpus is a correctly recorded one, not a malformed one.
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
        "geometry must use the exact ABI-2 axes",
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
    _require(bool(receipts), "no DECODE_QUALITY_V2 receipts found")
    return receipts


def receipt_log_paths(explicit: Iterable[Path], input_root: Path | None) -> list[Path]:
    paths = list(explicit)
    if input_root is not None:
        _require(input_root.is_dir(), f"input root is not a directory: {input_root}")
        paths.extend(sorted(input_root.rglob("decode-quality-v2-*.log")))
    _require(bool(paths), "no decode-quality receipt logs selected")
    normalized = [path.resolve() for path in paths]
    _require(
        len(normalized) == len(set(normalized)),
        "duplicate decode-quality receipt log selected",
    )
    return paths


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
        "load_shape": policy["loadShape"],
        "artifact": {
            "repository": policy["artifact"]["repository"],
            "revision": policy["artifact"]["revision"],
            "variant": policy["artifact"]["variant"],
            "fingerprint": policy["artifact"]["fingerprint"],
        },
        "implementation_fingerprint": policy["implementationFingerprint"],
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


def require_policy_shape(
    policies: list[dict[str, Any]],
    expected_policy_count: int | None,
    expected_fixture_count: int | None,
) -> None:
    if expected_policy_count is not None:
        _require(
            len(policies) == expected_policy_count,
            f"expected {expected_policy_count} policy row(s), sealed {len(policies)}",
        )
    if expected_fixture_count is not None:
        mismatched = [
            len(policy["fixtures"])
            for policy in policies
            if len(policy["fixtures"]) != expected_fixture_count
        ]
        _require(
            not mismatched,
            f"expected {expected_fixture_count} fixtures per policy row, got {mismatched}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-policy-count", type=int)
    parser.add_argument("--expected-fixture-count", type=int)
    args = parser.parse_args()
    policies = seal(read_receipts(receipt_log_paths(args.input, args.input_root)))
    require_policy_shape(
        policies,
        args.expected_policy_count,
        args.expected_fixture_count,
    )
    payload = json.dumps(policies, indent=2, ensure_ascii=False) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)
    print(f"sealed {len(policies)} decode-quality policy row(s) from correctness-only receipts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

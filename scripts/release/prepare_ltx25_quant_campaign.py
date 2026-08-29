#!/usr/bin/env python3
"""Materialize public LTX-2.5 weights and build terminal campaign inputs.

This helper is deliberately limited to the canonical public repository. It always resolves the
requested immutable revision anonymously into a canonical Hugging Face cache, captures the raw
expanded public API response, and writes either the exact nine-row campaign manifest or the
reviewed promotion input. It never accepts a token, a private source, a local-dir projection, or a
partial download pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


PUBLIC_REPOSITORY = "SceneWorks/ltx-2.5-mlx"
CAMPAIGN_SCHEMA = "sceneworks-ltx25-quant-campaign-v1"
PROMOTION_SCHEMA = "sceneworks-ltx25-quant-promotion-v2"
POLICY_ID = "sc-18777-reviewed-selection-v1"
REVISION = re.compile(r"^[0-9a-f]{40}$")

# (case id, transformer variant, bundle subdir, optional all-BF16 text encoder)
TERMINAL_CASES: tuple[tuple[str, str, str, str | None], ...] = (
    ("ltx25-bf16-blackwell-v1", "distilled", "distilled/bf16", None),
    ("ltx25-packed-q4-blackwell-v1", "distilled", "distilled/q4", None),
    ("ltx25-packed-q8-blackwell-v1", "distilled", "distilled/q8", None),
    (
        "ltx25-int8-convrot-blackwell-v1",
        "distilled",
        "bundles/distilled/int8-convrot",
        "bundles/distilled/int8-convrot/text_encoders/"
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
    (
        "ltx25-nvfp4-blackwell-v1",
        "distilled",
        "bundles/distilled/nvfp4",
        "bundles/distilled/nvfp4/text_encoders/"
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
    ("ltx25-bf16-blackwell-dev-v1", "dev", "dev/bf16", None),
    ("ltx25-packed-q4-blackwell-dev-v1", "dev", "dev/q4", None),
    ("ltx25-packed-q8-blackwell-dev-v1", "dev", "dev/q8", None),
    (
        "ltx25-int8-convrot-blackwell-dev-v1",
        "dev",
        "bundles/dev/int8-convrot",
        "bundles/dev/int8-convrot/text_encoders/"
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
)
PROMOTABLE_CASES = {
    case_id: (variant, bundle, encoder)
    for case_id, variant, bundle, encoder in TERMINAL_CASES
    if encoder is not None
}
SELECTION_FIELDS = {
    "policyId",
    "reviewedBy",
    "selectedCaseIds",
    "minimumReferencePsnr",
    "minimumReferenceSsim",
    "maximumTemporalBoundaryDrift",
    "minimumReplayPsnr",
    "minimumReplaySsim",
    "maximumReplayTemporalBoundaryDrift",
    "requireReplayOutputHashMatch",
}


def require_revision(value: str) -> str:
    if REVISION.fullmatch(value) is None:
        raise ValueError("public revision must be exact lowercase 40-hex")
    return value


def require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return path


def public_readback_url(revision: str) -> str:
    return (
        f"https://huggingface.co/api/models/{PUBLIC_REPOSITORY}/revision/"
        f"{require_revision(revision)}?blobs=true"
    )


def fetch_public_readback(
    revision: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bytes:
    """Fetch and validate the anonymous expanded Hub response, preserving its raw bytes."""
    request = urllib.request.Request(
        public_readback_url(revision),
        headers={"User-Agent": "SceneWorks-SC18777-terminal-campaign/1"},
    )
    with opener(request, timeout=120) as response:
        raw = response.read()
    document = json.loads(raw)
    if (
        document.get("id") != PUBLIC_REPOSITORY
        or document.get("sha") != revision
        or document.get("private") is not False
        or document.get("gated") is not False
    ):
        raise ValueError(
            "public readback must report the canonical repository, exact revision, "
            "private=false, and gated=false"
        )
    siblings = document.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise ValueError("public readback must contain the expanded non-empty sibling inventory")
    for sibling in siblings:
        if (
            not isinstance(sibling, dict)
            or not isinstance(sibling.get("rfilename"), str)
            or not isinstance(sibling.get("size"), int)
        ):
            raise ValueError("every public readback sibling must contain rfilename and size")
    return raw


def materialize_public_snapshot(
    revision: str,
    cache_root: Path,
    download: Callable[..., Any],
) -> Path:
    """Anonymously fetch the complete immutable repo into its canonical persistent cache."""
    require_revision(revision)
    cache_root = require_absolute(cache_root, "cache root")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    downloaded = download(
        repo_id=PUBLIC_REPOSITORY,
        revision=revision,
        cache_dir=str(cache_root),
        token=False,
    )
    expected = (
        cache_root
        / "models--SceneWorks--ltx-2.5-mlx"
        / "snapshots"
        / revision
    )
    downloaded_path = Path(downloaded)
    if not downloaded_path.is_dir() or downloaded_path.resolve() != expected.resolve():
        raise ValueError(
            "anonymous snapshot download did not return the exact canonical public cache path: "
            f"expected {expected}, got {downloaded_path}"
        )
    return expected.resolve()


def campaign_manifest(snapshot: Path, revision: str) -> dict[str, Any]:
    snapshot = require_absolute(snapshot, "snapshot")
    require_revision(revision)
    cases: list[dict[str, Any]] = []
    for case_id, variant, bundle, encoder in TERMINAL_CASES:
        case: dict[str, Any] = {
            "caseId": case_id,
            "transformerVariant": variant,
            "snapshotRoot": str(snapshot),
            "modelRevision": revision,
            "bundleSubdir": bundle,
        }
        if encoder is not None:
            case["bf16TextEncoderSubpath"] = encoder
        cases.append(case)
    return {"schemaVersion": CAMPAIGN_SCHEMA, "cases": cases}


def validate_selection(selection: Any) -> dict[str, Any]:
    if not isinstance(selection, dict) or set(selection) != SELECTION_FIELDS:
        raise ValueError(
            "reviewed selection must contain exactly the required identity and quality fields"
        )
    if selection["policyId"] != POLICY_ID:
        raise ValueError(f"promotion policyId must equal {POLICY_ID}")
    if not isinstance(selection["reviewedBy"], str) or not selection["reviewedBy"].strip():
        raise ValueError("reviewedBy must be a non-empty reviewer identity")
    selected = selection["selectedCaseIds"]
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or any(case_id not in PROMOTABLE_CASES for case_id in selected)
    ):
        raise ValueError("selectedCaseIds must contain unique promotable terminal case IDs")
    variants = [PROMOTABLE_CASES[case_id][0] for case_id in selected]
    if len(variants) != len(set(variants)):
        raise ValueError("selectedCaseIds may contain at most one winner per transformer variant")

    positive = (
        "minimumReferencePsnr",
        "minimumReferenceSsim",
        "minimumReplayPsnr",
        "minimumReplaySsim",
    )
    bounded = (
        "minimumReferenceSsim",
        "maximumTemporalBoundaryDrift",
        "minimumReplaySsim",
        "maximumReplayTemporalBoundaryDrift",
    )
    for field in positive + (
        "maximumTemporalBoundaryDrift",
        "maximumReplayTemporalBoundaryDrift",
    ):
        value = selection[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{field} must be a finite non-negative number")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{field} must be a finite non-negative number")
    if any(selection[field] <= 0 for field in positive):
        raise ValueError("PSNR and SSIM floors must be positive")
    if any(not 0 <= selection[field] <= 1 for field in bounded):
        raise ValueError("SSIM floors and drift ceilings must be in [0,1]")
    if selection["requireReplayOutputHashMatch"] is not True:
        raise ValueError("promotion must require exact replay output hash matching")
    return selection


def promotion_input(
    snapshot: Path,
    revision: str,
    public_readback: Path,
    selection: Any,
) -> dict[str, Any]:
    snapshot = require_absolute(snapshot, "public snapshot")
    public_readback = require_absolute(public_readback, "public readback")
    selection = validate_selection(selection)
    selected = set(selection["selectedCaseIds"])
    cases: list[dict[str, Any]] = []
    for case_id, variant, bundle, encoder in TERMINAL_CASES:
        if case_id not in selected:
            continue
        assert encoder is not None
        cases.append(
            {
                "caseId": case_id,
                "transformerVariant": variant,
                "publicSnapshotRoot": str(snapshot),
                "publicModelRevision": require_revision(revision),
                "publicBundleSubdir": bundle,
                "bf16TextEncoderSubpath": encoder,
                "publicReadback": str(public_readback),
            }
        )
    return {
        "schemaVersion": PROMOTION_SCHEMA,
        "publicRepository": PUBLIC_REPOSITORY,
        "selection": selection,
        "cases": cases,
    }


def write_new(path: Path, payload: bytes) -> None:
    path = require_absolute(path, "output")
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, document: dict[str, Any]) -> None:
    write_new(path, json.dumps(document, indent=2, sort_keys=True).encode() + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("campaign", "promotion"):
        command = subparsers.add_parser(name)
        command.add_argument("--revision", required=True)
        command.add_argument("--cache-root", required=True, type=Path)
        command.add_argument("--readback-out", required=True, type=Path)
    campaign = subparsers.choices["campaign"]
    campaign.add_argument("--manifest-out", required=True, type=Path)
    promotion = subparsers.choices["promotion"]
    promotion.add_argument("--campaign-manifest", required=True, type=Path)
    promotion.add_argument("--selection-env", required=True)
    promotion.add_argument("--promotion-out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_revision(args.revision)
    raw_readback = fetch_public_readback(args.revision)
    from huggingface_hub import snapshot_download

    snapshot = materialize_public_snapshot(args.revision, args.cache_root, snapshot_download)
    if args.command == "campaign":
        write_new(args.readback_out, raw_readback)
        write_json(args.manifest_out, campaign_manifest(snapshot, args.revision))
        print(f"public campaign snapshot: {snapshot}", flush=True)
        return 0

    selection_json = os.environ.get(args.selection_env)
    if selection_json is None:
        raise ValueError(f"reviewed selection environment variable is unset: {args.selection_env}")
    expected_campaign = campaign_manifest(snapshot, args.revision)
    actual_campaign = json.loads(args.campaign_manifest.read_bytes())
    if actual_campaign != expected_campaign:
        raise ValueError(
            "downloaded campaign manifest does not bind this exact public revision/cache layout"
        )
    selection = json.loads(selection_json)
    write_new(args.readback_out, raw_readback)
    write_json(
        args.promotion_out,
        promotion_input(snapshot, args.revision, args.readback_out, selection),
    )
    print(f"public promotion snapshot: {snapshot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

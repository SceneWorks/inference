#!/usr/bin/env python3
"""Run and seal the opt-in SC-20684 Krea Realtime real-weight matrix.

The product command is intentionally the *only* measurement producer.  For every
case it receives ``KREA_SC20684_*`` environment variables and must emit exactly
one ``SC20684_KREA_PROVIDER_OBSERVATION <json>`` line.  This launcher neither
accepts metric arguments nor fills in missing values: it binds the observation to
the checked-out source and pinned snapshot, validates the complete six-cell matrix,
then atomically publishes an external receipt directory.

Example (on the coordinator's held Metal lane, never ordinary CI):

  python3 scripts/sc20684_krea_realtime_campaign.py \
    --snapshot /Volumes/Data/krea-realtime/q4 \
    --output /Volumes/Data/receipts/sc-20684-$(date +%Y%m%dT%H%M%S) \
    --product-command 'cargo test -p mlx-gen-krea-realtime --test integration \
      generate_smoke::sc20684_packed_campaign_observer -- --ignored --nocapture'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STORY = "SC-20684"
SCHEMA_VERSION = 1
MODEL_REPOSITORY = "SceneWorks/krea-realtime-14b-mlx"
MODEL_REVISION = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626"
OBSERVATION_PREFIX = "SC20684_KREA_PROVIDER_OBSERVATION "
CASES = tuple((mode, tier) for mode in ("t2v", "i2v", "v2v") for tier in ("q8", "q4"))
SOURCE_FILES = (
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/generate.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/src/pipeline.rs",
    "crates/media/mlx-gen/mlx-gen-krea-realtime/tests/generate_smoke.rs",
    "scripts/sc20684_krea_realtime_campaign.py",
)


class CampaignError(ValueError):
    """A non-terminal row or invalid provenance; no receipt may be published."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"{name} must be a nonempty string")
    return value


def _object(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CampaignError(f"{name} fields differ: expected {sorted(keys)}, got {actual}")
    return value


def _checked_output(output: Path) -> Path:
    output = output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CampaignError("--output must be outside the repository")
    if output.exists():
        raise CampaignError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise CampaignError(f"output parent does not exist: {output.parent}")
    return output


def source_identity(root: Path = ROOT) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    if status.returncode or status.stdout:
        raise CampaignError("campaign source must be a clean git checkout")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if head.returncode or len(head.stdout.strip()) != 40:
        raise CampaignError("cannot resolve checked-out source revision")
    files: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise CampaignError(f"missing product source seam: {relative}")
        files[relative] = _sha256(path.read_bytes())
    return {"repositoryHead": head.stdout.strip(), "files": files}


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    config = snapshot / "config.json"
    if not snapshot.is_dir() or not config.is_file():
        raise CampaignError("snapshot must be a Q4 Krea directory with config.json")
    raw = config.read_bytes()
    if len(raw) > 1_000_000:
        raise CampaignError("snapshot config.json is unreasonably large")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CampaignError(f"invalid snapshot config.json: {error}") from error
    quant = parsed.get("quantization")
    if not isinstance(quant, dict) or quant.get("bits") != 4 or quant.get("group_size") != 64:
        raise CampaignError("snapshot must identify the pinned Q4/group-64 product tier")
    # Deliberately stat only: the launcher must never read the materialized real weights.
    required = ("dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json")
    files: dict[str, dict[str, int]] = {}
    for name in required:
        path = snapshot / name
        if not path.is_file():
            raise CampaignError(f"snapshot missing required product file: {name}")
        stat = path.stat()
        if stat.st_size <= 0:
            raise CampaignError(f"snapshot product file is empty: {name}")
        files[name] = {"size": stat.st_size, "mtimeNs": stat.st_mtime_ns}
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "variant": "q4",
        "configSha256": _sha256(raw),
        "files": files,
    }


def _validate_observation(
    observation: object,
    *,
    expected_mode: str,
    expected_tier: str,
    run_id: str,
    expected_source: dict[str, Any],
    expected_snapshot: dict[str, Any],
) -> dict[str, Any]:
    row = _object(
        observation,
        "provider observation",
        {
            "schemaVersion", "producer", "runId", "case", "source", "model", "toolchain",
            "geometry", "compiledHandle", "bytes", "timing", "parity", "quality", "fallback",
            "cancellation", "output",
        },
    )
    if row["schemaVersion"] != 1 or row["producer"] != "mlx-gen-krea-realtime/sc20684":
        raise CampaignError("provider observation schema or producer mismatch")
    if row["runId"] != run_id:
        raise CampaignError("provider observation run id does not match its launched cell")
    case = _object(row["case"], "case", {"mode", "cacheTier"})
    if case != {"mode": expected_mode, "cacheTier": expected_tier}:
        raise CampaignError("provider observation did not report the launched mode/tier")
    if row["source"] != expected_source:
        raise CampaignError("provider observation source identity drift")
    if row["model"] != expected_snapshot:
        raise CampaignError("provider observation model identity drift")

    toolchain = _object(row["toolchain"], "toolchain", {"os", "arch", "rustc", "cargo", "mlx", "metalDevice"})
    for key, value in toolchain.items():
        _nonempty(value, f"toolchain.{key}")
    geometry = _object(row["geometry"], "geometry", {"batch", "heads", "queryTokens", "keyTokens", "headDim", "groupSize", "mask"})
    if geometry["batch"] != 1 or geometry["heads"] != 40 or geometry["headDim"] != 128 or geometry["groupSize"] != 64:
        raise CampaignError("provider observation did not use native Krea packed geometry")
    _integer(geometry["queryTokens"], "geometry.queryTokens", minimum=1)
    _integer(geometry["keyTokens"], "geometry.keyTokens", minimum=1)
    if geometry["mask"] not in {"none", "block-causal"}:
        raise CampaignError("provider observation did not use a supported analytic mask")

    handle = _object(row["compiledHandle"], "compiledHandle", {"identity", "retainedBytes", "compiled", "acceptedDispatches"})
    expected_handle = f"sc20684/krea-packed-affine-{expected_tier}-d128-g64-v1"
    if handle["identity"] != expected_handle or handle["compiled"] is not True:
        raise CampaignError("compiled handle does not identify the launched cache tier")
    _integer(handle["retainedBytes"], "compiledHandle.retainedBytes", minimum=1)
    _integer(handle["acceptedDispatches"], "compiledHandle.acceptedDispatches", minimum=1)

    byte_fields = {"persistent", "retainedHandle", "boundedScratch", "denseWindow", "scoreMatrix"}
    bytes_ = _object(row["bytes"], "bytes", byte_fields)
    for key, value in bytes_.items():
        _integer(value, f"bytes.{key}")
    if bytes_["persistent"] <= 0 or bytes_["retainedHandle"] <= 0:
        raise CampaignError("packed persistent/retained-handle bytes must be measured")
    if bytes_["denseWindow"] != 0 or bytes_["scoreMatrix"] != 0:
        raise CampaignError("compressed-domain row allocated a dense window or score matrix")

    timing = _object(row["timing"], "timing", {"label", "wallMs", "compileMs", "dispatchMs"})
    _nonempty(timing["label"], "timing.label")
    for key in ("wallMs", "compileMs", "dispatchMs"):
        if type(timing[key]) not in (int, float) or timing[key] < 0:
            raise CampaignError(f"timing.{key} must be nonnegative")

    parity = _object(row["parity"], "parity", {"status", "candidateTier", "maxAbsError", "tolerance"})
    if parity["status"] != "pass" or parity["candidateTier"] != expected_tier:
        raise CampaignError("packed parity is missing or tier-substituted")
    if type(parity["maxAbsError"]) not in (int, float) or type(parity["tolerance"]) not in (int, float) or parity["maxAbsError"] < 0 or parity["tolerance"] <= 0:
        raise CampaignError("packed parity numbers are malformed")

    quality = _object(row["quality"], "quality", {"status", "candidateTier", "metric", "acknowledged"})
    if quality["status"] != "pass" or quality["candidateTier"] != expected_tier:
        raise CampaignError("packed quality is missing or tier-substituted")
    _nonempty(quality["metric"], "quality.metric")
    if type(quality["acknowledged"]) is not bool or (expected_tier == "q4" and not quality["acknowledged"]):
        raise CampaignError("Q4 requires its separate explicit quality acknowledgement")

    fallback = _object(row["fallback"], "fallback", {"count", "reason"})
    _integer(fallback["count"], "fallback.count")
    if fallback["reason"] is not None and not isinstance(fallback["reason"], str):
        raise CampaignError("fallback.reason must be null or a string")
    cancellation = _object(row["cancellation"], "cancellation", {"status", "requests", "partialStateMutation", "scratchReleased"})
    if cancellation["status"] != "pass" or cancellation["partialStateMutation"] is not False or cancellation["scratchReleased"] is not True:
        raise CampaignError("cancellation evidence is not terminal and clean")
    _integer(cancellation["requests"], "cancellation.requests", minimum=1)
    output = _object(row["output"], "output", {"status", "sha256"})
    if output["status"] != "generated" or not _is_sha256(output["sha256"]):
        raise CampaignError("product output evidence is missing")
    return row


def _read_observation(stdout: str) -> dict[str, Any]:
    candidates = [line[len(OBSERVATION_PREFIX):] for line in stdout.splitlines() if line.startswith(OBSERVATION_PREFIX)]
    if len(candidates) != 1:
        raise CampaignError(f"product command must emit exactly one observation line, got {len(candidates)}")
    try:
        result = json.loads(candidates[0])
    except json.JSONDecodeError as error:
        raise CampaignError(f"malformed provider observation: {error}") from error
    return result


def run_matrix(command: str, snapshot: Path, source: dict[str, Any], model: dict[str, Any], timeout: int) -> list[dict[str, Any]]:
    try:
        argv = __import__("shlex").split(command)
    except ValueError as error:
        raise CampaignError(f"invalid --product-command: {error}") from error
    if not argv:
        raise CampaignError("--product-command must not be empty")
    rows: list[dict[str, Any]] = []
    for mode, tier in CASES:
        run_id = str(uuid.uuid4())
        env = os.environ.copy()
        env.update({
            "KREA_SC20684_RUN_ID": run_id,
            "KREA_SC20684_REQUEST_MODE": mode,
            "KREA_SC20684_CACHE_TIER": tier,
            "KREA_SC20684_SNAPSHOT_DIR": str(snapshot),
            "KREA_SC20684_MODEL_REPOSITORY": MODEL_REPOSITORY,
            "KREA_SC20684_MODEL_REVISION": MODEL_REVISION,
        })
        if tier == "q4":
            # The Q4 arm is a separately named experiment.  This is a selector only; measured
            # quality still has to arrive from the provider-owned observer and pass reduction.
            env["KREA_SC20684_Q4_QUALITY_ARM"] = "acknowledged"
        started = time.monotonic_ns()
        try:
            result = subprocess.run(argv, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise CampaignError(f"{mode}/{tier} product command timed out after {timeout}s") from error
        if result.returncode:
            raise CampaignError(f"{mode}/{tier} product command failed with exit {result.returncode}")
        row = _validate_observation(_read_observation(result.stdout), expected_mode=mode, expected_tier=tier, run_id=run_id, expected_source=source, expected_snapshot=model)
        rows.append({"mode": mode, "cacheTier": tier, "launcherElapsedNs": time.monotonic_ns() - started, "observation": row})
    return rows


def validate_matrix(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(CASES):
        raise CampaignError("matrix is incomplete")
    coordinates = [(row.get("mode"), row.get("cacheTier")) for row in rows]
    if set(coordinates) != set(CASES) or len(set(coordinates)) != len(CASES):
        raise CampaignError("matrix must contain every T2V/I2V/V2V and Q8/Q4 cell exactly once")
    run_ids = [row["observation"].get("runId") for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise CampaignError("matrix cells reused a provider observation")


def publish(output: Path, *, source: dict[str, Any], model: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    output = _checked_output(output)
    validate_matrix(rows)
    receipt = {
        "schemaVersion": SCHEMA_VERSION,
        "story": STORY,
        "status": "terminal-complete",
        "launcher": {"python": platform.python_version(), "system": platform.platform()},
        "source": source,
        "model": model,
        "matrix": rows,
    }
    parent = output.parent
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent))
    try:
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (temporary / "receipt.json").write_bytes(encoded)
        (temporary / "receipt.json.sha256").write_text(f"{_sha256(encoded)}  receipt.json\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True, help="pinned Q4 Krea Realtime snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="new external receipt directory")
    parser.add_argument("--product-command", required=True, help="product-owned observer command, executed once per matrix cell")
    parser.add_argument("--timeout", type=int, default=7200, help="per-cell command timeout in seconds")
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise CampaignError("--timeout must be positive")
        output = _checked_output(args.output)
        source = source_identity()
        model = snapshot_identity(args.snapshot)
        rows = run_matrix(args.product_command, args.snapshot.resolve(), source, model, args.timeout)
        publish(output, source=source, model=model, rows=rows)
    except CampaignError as error:
        print(f"SC-20684 campaign refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "published", "output": str(output), "cells": len(CASES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

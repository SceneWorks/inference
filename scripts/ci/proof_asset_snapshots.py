#!/usr/bin/env python3
"""Materialize the pinned Hugging Face inputs of an explicit model-inference proof (SW-7217).

Each input names a row of `release/real-weight-models.toml` and the repo-relative directory or file
its binding points at. The snapshot lives in the runner's own hub cache (`runner_hub_cache`); a
complete one is used as-is with no transfer, and a missing or incomplete one is fetched through that
cache as the row's `download_files` slice, then verified against its `expected_files`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.release.ensure_model_snapshot import (  # noqa: E402
    download_snapshot,
    ensure_snapshot,
    runner_cache_snapshot,
    runner_hub_cache,
)
from scripts.release.verify_model_snapshot import load_model  # noqa: E402

MANIFEST = ROOT / "release" / "real-weight-models.toml"
# Where the step installed the hash-pinned huggingface_hub, imported only when a snapshot is fetched.
HUB_PATH_ENV = "PROOF_HF_HUB_PATH"


@dataclass(frozen=True)
class Input:
    """One pinned input: a manifest row and the repo-relative path its binding names."""

    model: str
    path: str = ""


def manifest_keys(*tables: dict[str, dict[str, Input]]) -> frozenset[str]:
    """Every manifest row the given `{case: {variable: Input}}` tables bind."""
    return frozenset(item.model for table in tables for case in table.values() for item in case.values())


def materialize(bindings: dict[str, Input], *, environ: dict | None = None, download=download_snapshot,
                manifest: Path = MANIFEST) -> dict[str, Path]:
    """Return `{variable: path}` after ensuring every bound snapshot is complete in the hub cache."""
    environ = os.environ if environ is None else environ
    cache = runner_hub_cache(environ)
    hub_path = environ.get(HUB_PATH_ENV, "")
    if hub_path and hub_path not in sys.path:
        sys.path.insert(0, hub_path)
    # In Actions a download lands only in the runner's own hub cache.
    download_root = cache if environ.get("GITHUB_ACTIONS") == "true" else None
    snapshots: dict[str, Path] = {}
    for item in bindings.values():
        if item.model in snapshots:
            continue
        model = load_model(manifest, item.model)
        snapshot = runner_cache_snapshot(model, cache)
        fetched = ensure_snapshot(model, snapshot, download, download_root=download_root)
        print(f"{item.model}: {'downloaded' if fetched else 'complete, no transfer'} "
              f"({model['repository']}@{model['revision']} in {snapshot})", flush=True)
        snapshots[item.model] = snapshot
    return {name: snapshots[item.model] / item.path if item.path else snapshots[item.model]
            for name, item in bindings.items()}

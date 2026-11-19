#!/usr/bin/env python3
"""Export manifest-pinned model snapshot paths to a GitHub Actions environment file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

if __package__:
    from .ensure_model_snapshot import runner_hub_cache
    from .verify_model_snapshot import load_model
else:
    from ensure_model_snapshot import runner_hub_cache
    from verify_model_snapshot import load_model


def snapshot_path(hub_cache: str, model: dict) -> str:
    """Return the model's snapshot directory in the runner's Hugging Face hub cache.

    `models--<org>--<name>/snapshots/<revision>`, the layout `ensure_model_snapshot.py` materializes
    through the cache, so a snapshot fetched once is reused by every later run.
    """
    root = (
        PureWindowsPath(hub_cache)
        if PureWindowsPath(hub_cache).drive or "\\" in hub_cache
        else PurePosixPath(hub_cache)
    )
    repository = "models--" + model["repository"].replace("/", "--")
    return str(root / repository / "snapshots" / model["revision"])


def environment_assignments(manifest: Path, keys: list[str], hub_cache: str) -> list[str]:
    """Resolve model keys to deterministic GitHub environment assignments."""
    assignments: list[str] = []
    claimed: set[str] = set()
    for key in keys:
        model = load_model(manifest, key)
        variables = model.get("environment", [])
        if not variables:
            raise RuntimeError(f"model {key!r} declares no environment variable")
        path = snapshot_path(hub_cache, model)
        for variable in variables:
            if variable in claimed:
                raise RuntimeError(f"environment variable {variable!r} is assigned more than once")
            if not isinstance(variable, str) or not variable.isidentifier() or not variable.isupper():
                raise RuntimeError(f"model {key!r} has invalid environment variable {variable!r}")
            claimed.add(variable)
            assignments.append(f"{variable}={path}")
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, dest="models")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/real-weight-models.toml"),
    )
    parser.add_argument(
        "--hub-cache",
        default=str(runner_hub_cache()),
        help="Hugging Face hub cache (default: HF_HUB_CACHE, else $HF_HOME/hub, else the fleet root)",
    )
    parser.add_argument("--github-env", type=Path, default=os.environ.get("GITHUB_ENV"))
    args = parser.parse_args()
    if args.github_env is None:
        parser.error("GITHUB_ENV is required (or pass --github-env)")

    assignments = environment_assignments(args.manifest, args.models, args.hub_cache)
    with args.github_env.open("a", encoding="utf-8", newline="\n") as output:
        for assignment in assignments:
            output.write(assignment + "\n")
    for assignment in assignments:
        variable, path = assignment.split("=", 1)
        print(f"configured {variable}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

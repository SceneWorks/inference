#!/usr/bin/env python3
"""Export manifest-pinned model snapshot paths to a GitHub Actions environment file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

if __package__:
    from .verify_model_snapshot import load_model
else:
    from verify_model_snapshot import load_model


def snapshot_path(runner_temp: str, model: dict) -> str:
    """Return the stable runner-local snapshot path for a manifest model."""
    root = (
        PureWindowsPath(runner_temp)
        if PureWindowsPath(runner_temp).drive or "\\" in runner_temp
        else PurePosixPath(runner_temp)
    )
    return str(root / "model-snapshots" / model["key"] / model["revision"])


def environment_assignments(manifest: Path, keys: list[str], runner_temp: str) -> list[str]:
    """Resolve model keys to deterministic GitHub environment assignments."""
    assignments: list[str] = []
    claimed: set[str] = set()
    for key in keys:
        model = load_model(manifest, key)
        variables = model.get("environment", [])
        if not variables:
            raise RuntimeError(f"model {key!r} declares no environment variable")
        path = snapshot_path(runner_temp, model)
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
    parser.add_argument("--runner-temp", default=os.environ.get("RUNNER_TEMP"))
    parser.add_argument("--github-env", type=Path, default=os.environ.get("GITHUB_ENV"))
    args = parser.parse_args()
    if not args.runner_temp:
        parser.error("RUNNER_TEMP is required (or pass --runner-temp)")
    if args.github_env is None:
        parser.error("GITHUB_ENV is required (or pass --github-env)")

    assignments = environment_assignments(args.manifest, args.models, args.runner_temp)
    with args.github_env.open("a", encoding="utf-8", newline="\n") as output:
        for assignment in assignments:
            output.write(assignment + "\n")
    for assignment in assignments:
        variable, path = assignment.split("=", 1)
        print(f"configured {variable}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

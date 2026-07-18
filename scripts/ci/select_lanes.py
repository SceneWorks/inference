#!/usr/bin/env python3
"""Select CI lanes from changed paths, including downstream dependencies."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import PurePosixPath
from typing import Iterable


LANES = (
    "workspace",
    "contracts",
    "candle_cpu",
    "macos_metal",
    "windows_cuda",
    "supply_chain",
    "docs",
    "release",
    "real_weights",
)

def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def _all(lanes: dict[str, bool]) -> None:
    for lane in lanes:
        lanes[lane] = True


def select_lanes(paths: Iterable[str], force_all: bool = False) -> dict[str, bool]:
    """Return the minimal safe lane set for the supplied repository paths."""
    lanes = {lane: False for lane in LANES}
    lanes["workspace"] = True  # Cheap and foundational; never skip graph validation.

    if force_all:
        _all(lanes)
        return lanes

    normalized = sorted(
        {
            PurePosixPath(path.strip().replace("\\", "/")).as_posix().removeprefix("./")
            for path in paths
            if path.strip()
        }
    )
    if not normalized:
        _all(lanes)
        return lanes

    for path in normalized:
        if path.startswith("docs/") or path in {
            ".github/CODEOWNERS",
            ".gitignore",
            "AGENTS.md",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            "README.md",
            "SECURITY.md",
        }:
            lanes["docs"] = True
            continue

        if path == "LICENSE" or path == "deny.toml":
            lanes["supply_chain"] = True
            lanes["release"] = True
            continue

        if _under(path, "release") or _under(path, "scripts/release"):
            lanes["release"] = True
            lanes["docs"] = True
            continue

        if _under(path, "scripts/ci") or _under(path, ".github/workflows"):
            _all(lanes)
            continue

        if path in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"} or _under(
            path, ".cargo"
        ):
            _all(lanes)
            continue

        if _under(path, "crates/contracts/core-llm"):
            lanes.update(
                contracts=True,
                candle_cpu=True,
                macos_metal=True,
                windows_cuda=True,
                real_weights=True,
            )
            continue

        if _under(path, "crates/contracts/gen-core") or _under(
            path, "crates/contracts/gen-core-testkit"
        ):
            lanes.update(
                contracts=True,
                candle_cpu=True,
                macos_metal=True,
                windows_cuda=True,
                real_weights=True,
            )
            continue

        if _under(path, "crates/bundles/runtime-catalog"):
            lanes.update(
                candle_cpu=True,
                macos_metal=True,
                windows_cuda=True,
                real_weights=True,
                release=True,
            )
            continue

        if _under(path, "crates/bundles/runtime-macos"):
            lanes.update(macos_metal=True, real_weights=True, release=True)
            continue

        if _under(path, "crates/bundles/runtime-cpu"):
            lanes.update(candle_cpu=True, real_weights=True, release=True)
            continue

        if _under(path, "crates/bundles/runtime-cuda"):
            lanes.update(windows_cuda=True, real_weights=True, release=True)
            continue

        if _under(path, "crates/llm/candle-llm") or _under(
            path, "crates/media/candle-gen"
        ):
            lanes.update(
                candle_cpu=True,
                macos_metal=True,
                windows_cuda=True,
                real_weights=True,
            )
            continue

        # The Candle audio family (sc-12835) is candle-classified: it builds and runs on the
        # CPU, macOS (candle audio rides every bundle incl. mlx), and CUDA lanes.
        if _under(path, "crates/audio"):
            lanes.update(
                candle_cpu=True,
                macos_metal=True,
                windows_cuda=True,
                real_weights=True,
            )
            continue

        if _under(path, "crates/llm/mlx-llm") or _under(path, "crates/media/mlx-gen"):
            lanes.update(macos_metal=True, real_weights=True)
            continue

        if _under(path, "scripts"):
            _all(lanes)
            continue

        # New top-level code/build paths must fail safe until explicitly classified.
        _all(lanes)

    if any(path.endswith("Cargo.toml") for path in normalized):
        lanes["supply_chain"] = True
        lanes["release"] = True
    return lanes


def changed_paths(base: str, head: str) -> list[str]:
    if not base or set(base) == {"0"}:
        return []
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACDMRTUXB", base, head],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--paths", nargs="*", help="changed paths supplied directly")
    source.add_argument("--all", action="store_true", help="select every lane")
    parser.add_argument("--base", help="base Git revision")
    parser.add_argument("--head", default="HEAD", help="head Git revision")
    parser.add_argument("--github-output", help="append key=value outputs to this file")
    args = parser.parse_args()

    paths = args.paths
    if paths is None and not args.all:
        if not args.base:
            parser.error("provide --paths, --base, or --all")
        paths = changed_paths(args.base, args.head)

    lanes = select_lanes(paths or [], force_all=args.all)
    print(json.dumps(lanes, sort_keys=True))
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as output:
            for lane, selected in lanes.items():
                output.write(f"{lane}={'true' if selected else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Copy the built `mlx.metallib` next to an iOS app's executable.

MLX resolves its Metal kernel library at runtime through a precedence chain (the fork's
`mlx-sys/patches/metallib-search-path.patch`). Inside an iOS app sandbox only two links of that
chain are reachable:

1. ``$PMETAL_METALLIB_PATH`` -- an explicit override the host sets before first use;
2. ``load_colocated_library`` -- ``mlx.metallib`` sitting next to the executable.

Everything else is host-only: ``~/.cache/pmetal/lib`` is not readable in the sandbox, and the
compiled-in ``METAL_PATH`` points into the cargo target directory, which is not shipped. So an
iOS app that does not carry the metallib fails at first Metal use, with no build-time warning.

This copies it into the app bundle so link 2 resolves. Run it as an Xcode "Run Script" build
phase after the Rust staticlib is built, or from a packaging script.

Usage
-----
    # explicit source and destination
    bundle_metallib.py --metallib <path/to/mlx.metallib> --dest "$TARGET_BUILD_DIR/$EXECUTABLE_FOLDER_PATH"

    # discover the metallib under a cargo target directory (default: ./target)
    bundle_metallib.py --target-dir target --profile debug --triple aarch64-apple-ios \
        --dest "$TARGET_BUILD_DIR/$EXECUTABLE_FOLDER_PATH"

The preferred source is the ``DEP_MLX_METALLIB`` value published by ``pmetal-mlx-sys``'s build
script; a crate in the dependency graph can read that env var and forward it here. Discovery is
the fallback for when the packaging step has no build script to read it from.

Exits non-zero on any failure -- a silently skipped copy would ship a broken app.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

METALLIB = "mlx.metallib"
# Kernels are compiled per-platform; the Mach-O carries the target triple. A macOS metallib in an
# iOS bundle loads as far as `MTLDevice::newLibraryWithURL:` and then fails to run, so verify
# rather than trusting the path.
IOS_MARKER = b"apple-ios"
MACOS_MARKER = b"apple-macos"


def fail(message: str) -> None:
    print(f"bundle_metallib: {message}", file=sys.stderr)
    raise SystemExit(1)


def discover(target_dir: Path, triple: str, profile: str) -> Path:
    """Find the metallib mlx-sys built under a cargo target directory."""
    root = target_dir / triple / profile / "build"
    if not root.is_dir():
        fail(f"no build directory at {root} -- has the {triple} target been built?")
    matches = sorted(root.glob("pmetal-mlx-sys-*/out/build/lib/" + METALLIB))
    if not matches:
        fail(f"no {METALLIB} under {root}/pmetal-mlx-sys-*/out/build/lib/")
    if len(matches) > 1:
        # Several build hashes can coexist (feature or profile churn). Newest wins, but say so --
        # silently picking one is how a stale metallib ships.
        matches.sort(key=lambda p: p.stat().st_mtime)
        print(
            f"bundle_metallib: {len(matches)} candidates; using the newest ({matches[-1]})",
            file=sys.stderr,
        )
    return matches[-1]


def platform_of(metallib: Path) -> str | None:
    """Return 'ios', 'macos', or None if the target triple could not be read."""
    try:
        blob = metallib.read_bytes()
    except OSError as error:
        fail(f"cannot read {metallib}: {error}")
    if IOS_MARKER in blob:
        return "ios"
    if MACOS_MARKER in blob:
        return "macos"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--metallib", type=Path, help=f"path to a prebuilt {METALLIB}")
    source.add_argument(
        "--target-dir",
        type=Path,
        default=Path("target"),
        help="cargo target directory to discover the metallib under (default: target)",
    )
    parser.add_argument("--triple", default="aarch64-apple-ios", help="target triple for discovery")
    parser.add_argument("--profile", default="debug", help="cargo profile for discovery")
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help='destination directory, i.e. "$TARGET_BUILD_DIR/$EXECUTABLE_FOLDER_PATH"',
    )
    parser.add_argument(
        "--expect-platform",
        choices=("ios", "macos"),
        help="fail unless the metallib was compiled for this platform (recommended)",
    )
    parser.add_argument(
        "--codesign-identity",
        help="re-sign the copied metallib with this identity (Xcode passes ${EXPANDED_CODE_SIGN_IDENTITY})",
    )
    args = parser.parse_args()

    metallib = args.metallib or discover(args.target_dir, args.triple, args.profile)
    if not metallib.is_file():
        fail(f"{metallib} does not exist")

    found = platform_of(metallib)
    if args.expect_platform:
        if found is None:
            fail(f"{metallib} carries no recognizable target triple; refusing to bundle it")
        if found != args.expect_platform:
            fail(
                f"{metallib} was compiled for {found}, expected {args.expect_platform}. "
                "A cross-platform metallib loads but fails at first kernel dispatch."
            )

    dest_dir = args.dest
    if not dest_dir.is_dir():
        fail(f"destination {dest_dir} is not a directory")
    destination = dest_dir / METALLIB

    try:
        shutil.copy2(metallib, destination)
    except OSError as error:
        fail(f"copy to {destination} failed: {error}")

    if args.codesign_identity:
        # A modified file inside a signed bundle invalidates the signature; Xcode signs the
        # bundle before Run Script phases that copy into it, so re-sign what we just added.
        result = subprocess.run(
            ["codesign", "--force", "--sign", args.codesign_identity, str(destination)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            fail(f"codesign failed: {result.stderr.strip()}")

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"bundle_metallib: {destination} ({size_mb:.0f} MB, {found or 'unknown platform'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

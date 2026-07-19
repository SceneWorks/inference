#!/usr/bin/env python3
"""Fail when the normalized inference workspace drifts from its graph invariants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEMBER_COUNT = 81
INTERNAL_PACKAGES = {
    "candle-audio",
    "candle-audio-catalog",
    "candle-audio-kokoro",
    "candle-audio-moss-sfx",
    "candle-gen-catalog",
    "core-llm",
    "core-llm-testkit",
    "mlx-gen-catalog",
    "mlx-llm",
    "candle-llm",
    "sceneworks-gen-core",
    "sceneworks-gen-core-testkit",
    "runtime-catalog",
    "runtime-macos",
    "runtime-cpu",
    "runtime-cuda",
}
PINNED_WORKSPACE_DEPENDENCIES = {
    "mlx-rs": ("pmetal-mlx-rs", "932beb4e60db44d378ffa1fe648defea59b5cbd0"),
    "mlx-sys": ("pmetal-mlx-sys", "932beb4e60db44d378ffa1fe648defea59b5cbd0"),
    "candle-core": ("candle-core", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-nn": ("candle-nn", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-transformers": ("candle-transformers", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-flash-attn": ("candle-flash-attn", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
}
DEFAULT_GRAPH_PINNED_PACKAGES = {
    package_name: revision
    for dependency_name, (package_name, revision) in PINNED_WORKSPACE_DEPENDENCIES.items()
    if dependency_name != "candle-flash-attn"
}
FORBIDDEN_GRAPH_PACKAGES = {
    # Provider composition is ordinary, value-scoped source code. Reintroducing this crate would
    # make linker participation part of the supported runtime graph again.
    "inventory",
}
# Directory names whose subtrees are not part of this checkout's single workspace: the git store
# and build output (.git, target), plus agent tooling that nests its own gitignored worktrees --
# each a separate checkout carrying its own Cargo.lock/manifest (.claude, .codex). They must not
# be swept into the single-lockfile / single-manifest invariants below.
IGNORED_TREE_PARTS = frozenset({".git", "target", ".claude", ".codex"})


def fail(message: str) -> None:
    raise AssertionError(message)


def cargo_metadata(offline: bool) -> dict:
    command = ["cargo", "metadata", "--locked", "--format-version", "1"]
    if offline:
        command.append("--offline")
    # cargo emits UTF-8 on every platform, so decode explicitly. text=True would decode with
    # the locale encoding instead, which fails on Windows (cp1252) as soon as any package in
    # the resolved graph carries non-ASCII metadata -- today a dependency author name.
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        fail(f"cargo metadata failed with exit code {result.returncode}")
    return json.loads(result.stdout.decode("utf-8"))


def _within_workspace(path: Path) -> bool:
    """True when a discovered path belongs to this checkout's own workspace tree.

    The check is on the path RELATIVE to ROOT, so running the gate from inside a nested worktree
    (whose own absolute path contains e.g. ``.claude/worktrees/...``) still counts that worktree's
    own root Cargo.lock/manifest -- only subtrees *below* ROOT named in IGNORED_TREE_PARTS drop out.
    """
    return IGNORED_TREE_PARTS.isdisjoint(path.relative_to(ROOT).parts)


def check_filesystem() -> None:
    lockfiles = sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("Cargo.lock")
        if _within_workspace(path)
    )
    if lockfiles != [Path("Cargo.lock")]:
        fail(f"expected only the root Cargo.lock, found: {lockfiles}")

    workspace_manifests = []
    for manifest in ROOT.rglob("Cargo.toml"):
        if not _within_workspace(manifest):
            continue
        if any(
            line.strip() == "[workspace]"
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ):
            workspace_manifests.append(manifest.relative_to(ROOT))
    if workspace_manifests != [Path("Cargo.toml")]:
        fail(f"expected one active root workspace manifest, found: {workspace_manifests}")

    for required in (Path(".cargo/config.toml"), Path("rust-toolchain.toml")):
        if not (ROOT / required).is_file():
            fail(f"missing root-owned configuration: {required}")

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    dependencies = root_manifest["workspace"]["dependencies"]
    for dependency_name, (package_name, revision) in PINNED_WORKSPACE_DEPENDENCIES.items():
        dependency = dependencies.get(dependency_name)
        if not isinstance(dependency, dict):
            fail(f"missing structured root pin for {dependency_name}")
        if dependency.get("rev") != revision:
            fail(f"{dependency_name} is not declared at {revision}: {dependency}")
        if dependency.get("package", dependency_name) != package_name:
            fail(f"{dependency_name} no longer aliases package {package_name}: {dependency}")


def check_graph(metadata: dict) -> None:
    packages = metadata["packages"]
    packages_by_id = {package["id"]: package for package in packages}
    member_ids = metadata["workspace_members"]
    members = [packages_by_id[member_id] for member_id in member_ids]

    if len(member_ids) != EXPECTED_MEMBER_COUNT:
        fail(f"expected {EXPECTED_MEMBER_COUNT} workspace members, found {len(member_ids)}")
    if len(set(member_ids)) != len(member_ids):
        fail("workspace member IDs are not unique")

    for package in members:
        manifest = Path(package["manifest_path"]).resolve()
        if package["source"] is not None:
            fail(f"workspace member {package['name']} unexpectedly has source {package['source']}")
        if ROOT / "crates" not in manifest.parents:
            fail(f"workspace member is outside crates/: {manifest}")

    for name in sorted(INTERNAL_PACKAGES):
        matches = [package for package in packages if package["name"] == name]
        if len(matches) != 1:
            fail(f"expected one {name} package resolution, found {len(matches)}")
        if matches[0]["source"] is not None:
            fail(f"internal package {name} is not a path source: {matches[0]['source']}")

    resolved_names = {package["name"] for package in packages}
    forbidden = sorted(FORBIDDEN_GRAPH_PACKAGES & resolved_names)
    if forbidden:
        fail(f"explicit composition forbids these graph packages: {forbidden}")

    for package in members:
        for dependency in package["dependencies"]:
            if dependency["name"] not in INTERNAL_PACKAGES:
                continue
            if dependency["source"] is not None or dependency.get("path") is None:
                fail(
                    f"{package['name']} -> {dependency['name']} is not a workspace path edge: "
                    f"source={dependency['source']!r}, path={dependency.get('path')!r}"
                )

    for name, revision in DEFAULT_GRAPH_PINNED_PACKAGES.items():
        matches = [package for package in packages if package["name"] == name]
        if len(matches) != 1:
            fail(f"expected one {name} resolution, found {len(matches)}")
        source = matches[0]["source"] or ""
        if not source.endswith(f"#{revision}"):
            fail(f"{name} does not resolve at {revision}: {source}")

    tokenizer_minors = {
        ".".join(package["version"].split(".")[:2])
        for package in packages
        if package["name"] == "tokenizers"
    }
    if tokenizer_minors != {"0.21", "0.22"}:
        fail(f"expected intentional tokenizers 0.21/0.22 split, found {tokenizer_minors}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require Cargo to validate entirely from its local cache",
    )
    args = parser.parse_args()

    try:
        check_filesystem()
        check_graph(cargo_metadata(args.offline))
    except (AssertionError, json.JSONDecodeError) as error:
        print(f"workspace gate: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "workspace gate: OK "
        f"({EXPECTED_MEMBER_COUNT} path members, one lockfile, explicit registries, pinned backends, "
        "intentional tokenizer split)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

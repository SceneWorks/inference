#!/usr/bin/env python3
"""Narrow the PR test steps to the workspace packages a change can affect.

`select_lanes.py` decides WHICH lanes run; it is path-prefix based, so any `mlx-gen*` edit runs
every `mlx-gen*` crate's tests and any candle edit runs every `candle-gen*` crate's. This script
decides which PACKAGES a selected lane's test step runs:

  scope   (in the `changes` job)  -> `test_scope` output: `all`, or the affected package names
  select  (in a test step)        -> the `-p` arguments for that step's own package universe

A package is affected when

  * a changed path is inside it;
  * it reads a changed path through a literal path in its Rust sources (`include_str!("../x")`,
    `MANIFEST_DIR.join("../x")`, `join("..")`, a repository-relative "crates/..." literal);
  * `Cargo.lock` changed one of its resolved dependencies, or an edit landed in a `[patch]` crate
    it resolves to (the vendored candle crates);
  * a `[workspace.dependencies]` entry it inherits changed; or
  * it depends (normal, dev or build) on an affected package -- EXCEPT that a model crate is
    not reached through a shared crate. A model crate (`MODEL_CRATES`) runs when it changed,
    reads a changed file, or depends on a changed model crate (a shared encoder such as
    `mlx-gen-clip`); a `gen-core` / `mlx-gen` / `candle-gen` / `mlx-llm` edit runs the shared
    crates, catalogs and bundles but not the ~80 model crates behind them. Those still compile on
    the PR (all-target Clippy) and run on the post-merge `push: main` full suite.

A read from test code selects only its reader; a read from library code propagates to dependents
like a source edit. Changes the graph cannot bound run everything: a non-PR event, an empty diff,
the toolchain, `.cargo/`, a root `Cargo.toml` change beyond members/inherited dependencies, and
the CI wiring that drives this narrowing. The post-merge `push: main` run is never narrowed, so a
read the scan misses is still caught on main by the full suite.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
MODEL_CRATES = ("mlx-gen-*", "candle-gen-*", "candle-audio-*")
COMPOSITION_ROOTS = ("*-catalog",)
NARROWED_EVENTS = {"pull_request", "merge_group"}
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
# Paths that change how every package builds or how this narrowing itself runs.
FORCE_ALL = (
    "rust-toolchain.toml",
    ".cargo",
    ".github/workflows/ci.yml",
    "scripts/ci/test_scope.py",
    "scripts/ci/select_lanes.py",
)
# A literal that climbs out of its file or crate, wherever the climb starts: "../x",
# "/../../x" (a `concat!(env!("CARGO_MANIFEST_DIR"), ..)` tail), "{}/../x" (a format string).
CLIMBING_LITERAL = re.compile(r'"[^"\n]*?((?:\.\./)+[^"\n]*)"')
# `join("..")` climbs one level from the manifest directory, which is how the adapter fixture
# tests reach `mlx-gen`.
JOIN_PARENT = re.compile(r'join\(\s*"\.\."\s*\)')
# A repository-relative literal, as joined onto a computed workspace root.
ROOTED_LITERAL = re.compile(r'"((?:crates|docs|scripts|release|\.github)/[^"\n]*)"')


def is_model(name: str) -> bool:
    """A provider-family crate, as opposed to a shared engine, contract, catalog or bundle."""
    match = lambda patterns: any(fnmatch.fnmatchcase(name, p) for p in patterns)  # noqa: E731
    return match(MODEL_CRATES) and not match(COMPOSITION_ROOTS)


def _posix(path: str) -> str:
    return PurePosixPath(path.strip().replace("\\", "/")).as_posix().removeprefix("./")


def _under(path: str, prefix: str) -> bool:
    return prefix == "." or path == prefix or path.startswith(f"{prefix}/")


@dataclass
class Workspace:
    root_manifest: dict
    members: dict[str, str]  # member directory -> package name
    deps: dict[str, set[str]]  # package -> workspace packages it depends on
    patched: dict[str, str]  # `[patch]` path directory -> package name
    inherits: dict[str, set[str]] = field(default_factory=dict)  # shared key -> packages


def load_workspace(root: Path = ROOT) -> Workspace:
    import tomllib  # 3.11+; deferred so `select` runs on any runner's python3

    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    workspace = manifest["workspace"]
    shared = {
        name: spec["path"]
        for name, spec in workspace.get("dependencies", {}).items()
        if isinstance(spec, dict) and "path" in spec
    }
    excluded = {_posix(path) for path in workspace.get("exclude", [])}
    # Cargo expands member globs to the matching directories that hold a manifest.
    member_dirs = sorted(
        {
            _posix(str(manifest_path.parent.relative_to(root)))
            for pattern in workspace["members"]
            for manifest_path in root.glob(f"{_posix(pattern)}/Cargo.toml")
        }
        - excluded
    )
    members: dict[str, str] = {}
    manifests: dict[str, dict] = {}
    for member in member_dirs:
        data = tomllib.loads((root / member / "Cargo.toml").read_text(encoding="utf-8"))
        members[member] = data["package"]["name"]
        manifests[member] = data

    deps: dict[str, set[str]] = {name: set() for name in members.values()}
    inherits: dict[str, set[str]] = {}
    for member, data in manifests.items():
        name = members[member]
        for table in [data] + list(data.get("target", {}).values()):
            for kind in DEP_TABLES:
                for key, spec in table.get(kind, {}).items():
                    if not isinstance(spec, dict):
                        continue
                    if spec.get("workspace"):
                        inherits.setdefault(key, set()).add(name)
                    if "path" in spec:
                        target = _posix(os.path.normpath(f"{member}/{spec['path']}"))
                    elif spec.get("workspace") and key in shared:
                        target = _posix(os.path.normpath(shared[key]))
                    else:
                        continue
                    if target in members:
                        deps[name].add(members[target])

    patched = {
        _posix(os.path.normpath(spec["path"])): name
        for table in manifest.get("patch", {}).values()
        for name, spec in table.items()
        if isinstance(spec, dict) and "path" in spec
    }
    return Workspace(manifest, members, deps, patched, inherits)


def owner(path: str, dirs: dict[str, str]) -> str | None:
    """The package whose directory is the longest prefix of *path* (members nest)."""
    best = max((d for d in dirs if _under(path, d)), key=len, default=None)
    return dirs[best] if best else None


def _resolve(base: str, literal: str, root: Path) -> str | None:
    """Repository-relative target of a literal, cut before any format placeholder or glob."""
    parts = []
    for part in literal.split("/"):
        if any(ch in part for ch in "{}*?"):
            break
        parts.append(part)
    resolved = os.path.normpath(os.path.join(base, *parts)) if parts else base
    if resolved == ".." or resolved.startswith("../") or os.path.isabs(resolved):
        return None
    if not (root / resolved).exists():
        return None
    return _posix(resolved)


def _check_workspace():
    """`scripts/check-workspace.py`'s Rust test-code detection, reused rather than re-derived."""
    spec = importlib.util.spec_from_file_location(
        "check_workspace", ROOT / "scripts" / "check-workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_module_files(root: Path, members: dict[str, str], checks) -> set[str]:
    """Classify external modules with attributes, retaining any production inclusion route.

    The shared lint's legacy helper handles only adjacent ``#[cfg(test)] mod name;``.
    In particular, ``#[cfg(test)] #[path = "cases.rs"] mod tests;`` must not turn its
    fixture literals into library reads that fan out to all the reader's dependents.
    """
    test_files = {_posix(str(path.relative_to(root))) for path in checks._test_only_files(root)}
    edges: list[tuple[str, str, bool]] = []
    declaration = re.compile(
        r'(?P<attrs>(?:#\[[^\]]*\]\s*)*)(?:pub(?:\([^)]*\))?\s+)?'
        r'mod\s+(?P<name>\w+)\s*;'
    )
    for member, package in members.items():
        for source in (root / member).rglob("*.rs"):
            rel = _posix(str(source.relative_to(root)))
            if owner(rel, members) != package or "/target/" in f"/{rel}":
                continue
            if PurePosixPath(rel).relative_to(member).parts[0] in ("tests", "benches", "examples"):
                test_files.add(rel)
            text = source.read_text(encoding="utf-8", errors="replace")
            # Most Rust files declare no external module; avoid masking all their bodies twice.
            if not re.search(r"\bmod\b[^;{}]*;", text):
                continue
            spans = checks._cfg_test_spans(text)
            syntax = checks.strip_rust_comments(text, strip_literals=True)
            for match in declaration.finditer(syntax):
                attrs = text[match.start("attrs"):match.end("attrs")]
                path = re.search(r'#\[\s*path\s*=\s*"([^"\n]+)"\s*\]', attrs)
                name = match["name"]
                candidates = [source.parent / path[1]] if path else [
                    source.with_name(f"{name}.rs"), source.parent / name / "mod.rs",
                    source.parent / source.stem / f"{name}.rs",
                ]
                in_test = any(start <= match.start("name") < end for start, end in spans)
                for candidate in candidates:
                    target = _resolve(".", _posix(os.path.relpath(candidate, root)), root)
                    if target is not None:
                        edges.append((rel, target, in_test))
                        if in_test:
                            test_files.add(target)
    # Test-only context follows child modules too. A file included by BOTH test and shipping
    # modules is library code, and its ordinary child modules retain that production route.
    def closure(seeds: set[str], links: list[tuple[str, str]]) -> set[str]:
        reached = set(seeds)
        while True:
            extra = {target for source, target in links if source in reached} - reached
            if not extra:
                return reached
            reached |= extra

    test_files = closure(test_files, [(source, target) for source, target, _ in edges])
    production = {target for source, target, test in edges if source not in test_files and not test}
    production = closure(production, [(source, target) for source, target, test in edges if not test])
    return test_files - production


def file_reads(
    root: Path, members: dict[str, str]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Repository paths each package reaches through path literals: (test reads, lib reads).

    A climbing literal is resolved against both bases it can mean (`include_*!` is relative to
    the source file, a `join` to the manifest directory) and kept only when it exists, so a
    path-traversal test string such as "../../etc/passwd" does not become a read.

    A read from test code (`tests/`, `benches/`, `examples/`, a `#[cfg(test)]` item, or a file
    that only a `#[cfg(test)] mod x;` pulls in) is an input to the reader's own tests, so a
    change to it selects the reader alone. Any other read is library code a dependent may
    execute, so it propagates to dependents like a source change does.
    """
    checks = _check_workspace()
    test_only = _test_module_files(root, members, checks)
    test_reads: dict[str, set[str]] = {name: set() for name in members.values()}
    lib_reads: dict[str, set[str]] = {name: set() for name in members.values()}
    for member, name in members.items():
        for source in (root / member).rglob("*.rs"):
            rel = _posix(str(source.relative_to(root)))
            if owner(rel, members) != name or "/target/" in f"/{rel}":
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
            inside = PurePosixPath(rel).relative_to(member).parts
            whole_file_test = rel in test_only or inside[0] in ("tests", "benches", "examples")
            spans = [] if whole_file_test else checks._cfg_test_spans(text)

            def record(offset: int, target: str | None) -> None:
                if target is None or _under(target, member):
                    return
                in_test = whole_file_test or any(a <= offset < b for a, b in spans)
                (test_reads if in_test else lib_reads)[name].add(target)

            for match in CLIMBING_LITERAL.finditer(text):
                for base in (str(PurePosixPath(rel).parent), member):
                    record(match.start(), _resolve(base, match.group(1), root))
            for match in ROOTED_LITERAL.finditer(text):
                record(match.start(), _resolve(".", match.group(1), root))
            for match in JOIN_PARENT.finditer(text):
                record(match.start(), _posix(os.path.dirname(member)) or ".")
    return test_reads, lib_reads


def _lock_entries(text: str) -> dict[tuple[str, str, str], tuple]:
    import tomllib  # 3.11+; deferred so `select` runs on any runner's python3

    entries = {}
    for package in tomllib.loads(text).get("package", []):
        key = (package["name"], package["version"], package.get("source", ""))
        entries[key] = (package.get("checksum", ""), tuple(package.get("dependencies", [])))
    return entries


def _lock_dependents(entries: dict[tuple[str, str, str], tuple]) -> dict[tuple, set[tuple]]:
    """Package -> the packages whose `dependencies` name it, in one lockfile."""
    by_name: dict[str, list[tuple]] = {}
    for key in entries:
        by_name.setdefault(key[0], []).append(key)
    dependents: dict[tuple, set[tuple]] = {key: set() for key in entries}
    for key, (_, dependencies) in entries.items():
        for dependency in dependencies:
            name, *rest = dependency.split(" ")
            candidates = by_name.get(name, [])
            if rest:
                candidates = [c for c in candidates if c[1] == rest[0]] or candidates
            for candidate in candidates:
                dependents[candidate].add(key)
    return dependents


def lock_seeds(
    base_text: str | None, head_text: str, seeds: Iterable[str], members: set[str]
) -> set[str]:
    """Workspace packages whose resolved dependencies reach a changed lockfile entry.

    The walk climbs registry/git packages until it meets a source-less (workspace or `[patch]`)
    package, and stops there: from a member onward the workspace graph and its model boundary
    take over. *seeds* are `[patch]` crates edited directly (the vendored candle engine); that
    walk is an engine change all the way down -- through `candle-nn` and friends too -- so, as
    for a `candle-gen` edit, it selects no model crate. Both lockfiles are walked so a removed
    entry still reaches what used to depend on it.
    """
    head = _lock_entries(head_text)
    base = _lock_entries(base_text) if base_text else {}
    changed = {k for k in head.keys() | base.keys() if head.get(k) != base.get(k)}
    engine = {k for k in head if k[2] == "" and k[0] in set(seeds)}
    changed |= engine
    found: set[str] = set()
    for entries in (head, base):
        dependents = _lock_dependents(entries)
        frontier = [k for k in changed if k in entries]
        seen = set(frontier)
        while frontier:
            key = frontier.pop()
            if key[2] == "" and key[0] in members:
                found.add(key[0])
                continue
            for dependent in dependents[key]:
                if engine and is_model(dependent[0]):
                    continue  # an engine edit, not a change to the model's own resolution
                if dependent not in seen:
                    seen.add(dependent)
                    frontier.append(dependent)
    return found


def manifest_seeds(base_text: str | None, ws: Workspace) -> set[str] | None:
    """Packages a root `Cargo.toml` change affects, or None when it can affect every build."""
    import tomllib  # 3.11+; deferred so `select` runs on any runner's python3

    if base_text is None:
        return None
    base = tomllib.loads(base_text)
    head = ws.root_manifest
    strip = lambda m: {  # noqa: E731
        **m,
        "workspace": {
            k: v
            for k, v in m.get("workspace", {}).items()
            if k not in ("members", "exclude", "dependencies")
        },
    }
    if strip(base) != strip(head):
        return None  # profile, patch, resolver: every package can build differently
    old = base.get("workspace", {}).get("dependencies", {})
    new = head.get("workspace", {}).get("dependencies", {})
    seeds: set[str] = set()
    for key in old.keys() | new.keys():
        if old.get(key) != new.get(key):
            seeds |= ws.inherits.get(key, set())
    # Member-list edits need no seeds of their own: an added crate's files and a removed crate's
    # `Cargo.lock` entry select whatever they touch.
    return seeds


def affected_packages(
    paths: Iterable[str],
    ws: Workspace,
    test_reads: dict[str, set[str]],
    lib_reads: dict[str, set[str]],
    base_text: Callable[[str], str | None] = lambda path: None,
    head_text: Callable[[str], str] = lambda path: (ROOT / path).read_text(encoding="utf-8"),
) -> set[str] | None:
    """The packages *paths* can affect, or None when the change must run everything."""
    normalized = sorted({_posix(p) for p in paths if p.strip()})
    if not normalized:
        return None

    def readers(path: str, reads: dict[str, set[str]]) -> set[str]:
        return {r for r, targets in reads.items() if any(_under(path, t) for t in targets)}

    members = set(ws.members.values())
    seeds: set[str] = set()
    test_only: set[str] = set()
    patched_edits: set[str] = set()
    for path in normalized:
        if any(_under(path, prefix) for prefix in FORCE_ALL):
            return None
        if path == "Cargo.toml":
            found = manifest_seeds(base_text(path), ws)
            if found is None:
                return None
            seeds |= found
            continue
        if path == "Cargo.lock":
            seeds |= lock_seeds(base_text(path), head_text(path), (), members)
            continue
        package = owner(path, ws.members)
        if package is not None:
            seeds.add(package)
        elif (patched := owner(path, ws.patched)) is not None:
            patched_edits.add(patched)
        seeds |= readers(path, lib_reads)
        test_only |= readers(path, test_reads)
    if patched_edits:
        lock = head_text("Cargo.lock")
        seeds |= lock_seeds(lock, lock, patched_edits, members)

    dependents: dict[str, set[str]] = {name: set() for name in ws.deps}
    for package, targets in ws.deps.items():
        for target in targets:
            dependents[target].add(package)
    affected, frontier = set(seeds), list(seeds)
    while frontier:
        package = frontier.pop()
        for dependent in dependents[package]:
            if is_model(dependent) and not is_model(package):
                continue  # a shared-crate edit does not run every model behind it
            if dependent not in affected:
                affected.add(dependent)
                frontier.append(dependent)
    return affected | test_only


def changed_paths(base: str, head: str) -> list[str]:
    # `--no-renames` reports a move as delete + add, so the crate a file LEFT is affected too.
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", base, head],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.splitlines()


def git_text(revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"], capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout if result.returncode == 0 else None


def scope(
    event: str,
    paths: list[str] | None,
    root: Path = ROOT,
    base_text: Callable[[str], str | None] = lambda path: None,
) -> str:
    if event not in NARROWED_EVENTS or not paths:
        return "all"
    ws = load_workspace(root)
    affected = affected_packages(
        paths,
        ws,
        *file_reads(root, ws.members),
        base_text=base_text,
        head_text=lambda path: (root / path).read_text(encoding="utf-8"),
    )
    if affected is None:
        return "all"
    # An empty set is a real answer (no package can be affected) and must not read as `all`.
    return " ".join(sorted(affected)) or "none"


def select(scope_value: str, specs: list[str]) -> list[str]:
    """`-p` arguments for a step whose package universe is *specs* (cargo `-p` globs)."""
    if scope_value.strip() in ("", "all"):
        return [arg for spec in specs for arg in ("-p", spec)]
    names = sorted(
        name
        for name in scope_value.split()
        if any(fnmatch.fnmatchcase(name, spec) for spec in specs)
    )
    return [arg for name in names for arg in ("-p", name)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scope_cmd = sub.add_parser("scope", help="compute the test scope for a CI event")
    scope_cmd.add_argument("--event", required=True)
    scope_cmd.add_argument("--base", default="")
    scope_cmd.add_argument("--head", default="HEAD")
    scope_cmd.add_argument("--github-output")
    select_cmd = sub.add_parser("select", help="print one step's narrowed -p arguments")
    select_cmd.add_argument("--scope", default="all")
    select_cmd.add_argument("-p", dest="specs", action="append", required=True)
    args = parser.parse_args()

    if args.command == "select":
        print(" ".join(select(args.scope, args.specs)))
        return 0

    paths = None
    if args.event in NARROWED_EVENTS and args.base and set(args.base) != {"0"}:
        paths = changed_paths(args.base, args.head)
    value = scope(args.event, paths, base_text=lambda path: git_text(args.base, path))
    print(f"test_scope={value}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as output:
            output.write(f"test_scope={value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

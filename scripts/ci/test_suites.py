#!/usr/bin/env python3
"""Compile the selected Cargo test targets, then run a bounded libtest suite.

Routine CI executes every default test except exact, reviewed exhaustive cases in
test_suites.json. Explicit exhaustive validation executes every registered case,
never `--ignored` (real-weight tests and dedicated numerical lanes are separate).
No Rust test bodies, ignore attributes, compilation targets or package scopes change.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_suffix(".json")
LISTING = re.compile(r"^(.+): test$", re.MULTILINE)
SUMMARY = re.compile(
    r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;.*? (\d+) filtered out;"
)


def registry(path: Path = MANIFEST) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != 1 or not data["cases"]:
        raise ValueError("empty or unsupported exhaustive-test registry")
    seen = set()
    for case in data["cases"]:
        key = (case["package"], case["kind"], case["target"], case["test"])
        if key in seen or case["kind"] not in ("lib", "test"):
            raise ValueError(f"duplicate/invalid exhaustive identity: {key}")
        seen.add(key)
        source = ROOT / case["source"]
        if not source.is_file() or not re.search(
            r"\bfn\s+" + re.escape(case["test"].split("::")[-1]) + r"\s*\(",
            source.read_text(encoding="utf-8"),
        ):
            raise ValueError(f"stale exhaustive source: {key}: {source}")
        if len(case["reason"].strip()) < 30:
            raise ValueError(f"missing source classification rationale: {key}")
        if not case.get("platforms") or set(case["platforms"]) - {"darwin", "linux", "win32"}:
            raise ValueError(f"missing/invalid platform classification: {key}")
    return data["cases"]


def run(command: list[str], timeout: float, *, cwd=None, env=None) -> str:
    """Bound the child tree too: Iris and distributed tests launch child processes."""
    if timeout <= 0:
        raise TimeoutError("suite deadline exhausted")
    print("+ " + " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
        start_new_session=os.name != "nt",
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                timeout=30,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.kill()
        output, _ = process.communicate()
        if output:
            print(output, flush=True)
        raise TimeoutError(f"child exceeded {timeout:.0f}s: {command}") from None
    if process.returncode:
        if output:
            print(output, flush=True)
        raise RuntimeError(f"exit {process.returncode}: {command}")
    return output or ""


def artifacts(output: str) -> list[dict]:
    found = {}
    for line in output.splitlines():
        row = json.loads(line)
        if (
            row.get("reason") == "compiler-artifact"
            and row.get("profile", {}).get("test")
            and row.get("executable")
        ):
            found[row["executable"]] = row
    if not found:
        raise ValueError("Cargo produced no executable test artifacts")
    return list(found.values())


def test_environment(metadata: dict, package: dict, artifact: dict, output: str, rust_lib: str) -> dict:
    """Preserve Cargo's package cwd/env and dynamic-loader search for direct libtest runs."""
    env = dict(os.environ)
    manifest = Path(package["manifest_path"])
    env.update(CARGO_MANIFEST_DIR=str(manifest.parent), CARGO_MANIFEST_PATH=str(manifest),
               CARGO_PKG_NAME=package["name"], CARGO_PKG_VERSION=package["version"],
               RUST_TEST_THREADS="1")
    binary_dir = Path(artifact["executable"]).parent
    paths = [str(binary_dir), str(binary_dir.parent), rust_lib.strip()]
    target_dir = Path(metadata["target_directory"])
    for line in output.splitlines():
        row = json.loads(line)
        if row.get("reason") == "build-script-executed":
            if row.get("package_id") == artifact["package_id"] and row.get("out_dir"):
                env["OUT_DIR"] = row["out_dir"]
            for linked in row.get("linked_paths", []):
                linked = linked.split("=", 1)[-1]
                if Path(linked).is_relative_to(target_dir):
                    paths.append(linked)
    key = "PATH" if os.name == "nt" else "DYLD_FALLBACK_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    if key == "DYLD_FALLBACK_LIBRARY_PATH" and key not in env:
        paths += [str(Path.home() / "lib"), "/usr/local/lib", "/usr/lib"]
    paths += env.get(key, "").split(os.pathsep)
    env[key] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return env


def target_key(artifact: dict, packages: dict[str, str]) -> tuple[str, str, str]:
    target = artifact["target"]
    kinds = target["kind"]
    kind = "lib" if any(k in kinds for k in ("lib", "rlib", "proc-macro")) else kinds[0]
    return packages[artifact["package_id"]], kind, target["name"]


def listed(output: str) -> set[str]:
    return set(LISTING.findall(output))


def validate_cases(cases: list[dict], names: set[str], ignored: set[str]) -> set[str]:
    selected = {case["test"] for case in cases}
    if selected - names or selected & ignored:
        raise ValueError(
            f"stale/ignored exhaustive cases: missing={selected - names}, ignored={selected & ignored}"
        )
    # libtest --skip is a substring filter. Prove it excludes only exact registered
    # names instead of letting a new similarly named regression disappear silently.
    skipped = {name for name in names if any(test in name for test in selected)}
    if skipped != selected:
        raise ValueError(f"ambiguous exhaustive --skip names: {skipped - selected}")
    return selected


def verify_result(output: str, passed: int, ignored: int) -> None:
    # Child test binaries print their own summaries. The last is this harness's.
    rows = SUMMARY.findall(output)
    if not rows or tuple(map(int, rows[-1][:3])) != (passed, 0, ignored):
        raise ValueError(f"libtest did not execute the expected cases ({passed} passed, {ignored} ignored)")


def execute(mode: str, command: list[str], cases: list[dict], deadline_seconds: int) -> None:
    if command[:2] != ["cargo", "test"] or "--" in command or "--no-run" in command:
        raise ValueError("expected cargo test build arguments without libtest filters")
    deadline = time.monotonic() + deadline_seconds
    cases = [case for case in cases if sys.platform in case["platforms"]]
    remaining = lambda limit: min(limit, deadline - time.monotonic())
    metadata = json.loads(run(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version=1"], remaining(120)
    ))
    packages = {package["id"]: package["name"] for package in metadata["packages"]}
    by_name = {package["name"]: package for package in metadata["packages"]}
    for case in cases:
        package = by_name.get(case["package"])
        if not package:
            raise ValueError(f"registered package is absent: {case['package']}")
        manifest = package.get("manifest_path")
        if manifest and not (ROOT / case["source"]).is_relative_to(Path(manifest).parent):
            raise ValueError(f"registered source belongs to another package: {case}")
        targets = package.get("targets")
        if targets is not None and not any(
            target["name"] == case["target"] and case["kind"] in target["kind"]
            for target in targets
        ):
            raise ValueError(f"registered target is absent from Cargo metadata: {case}")
    output = run(command + ["--no-run", "--message-format=json"], remaining(2400))
    built = artifacts(output)
    rust_lib = run(["rustc", "--print", "target-libdir"], remaining(60))
    package_features = {}
    for artifact in built:
        package_features.setdefault(packages[artifact["package_id"]], set()).update(artifact.get("features", []))
    expected = {}
    for case in cases:
        if case["kind"] == "test" and "--lib" in command and "--tests" not in command:
            continue
        if not set(case.get("features", [])) <= package_features.get(case["package"], set()):
            continue
        expected.setdefault((case["package"], case["kind"], case["target"]), []).append(case)
    built_keys = {target_key(artifact, packages) for artifact in built}
    built_packages = {key[0] for key in built_keys}
    # An absent target within a built package is a rename/configuration error,
    # even if the executable disappeared completely (not just a test rename).
    missing = {key for key in expected if key[0] in built_packages and key not in built_keys}
    if missing:
        raise ValueError(f"registered targets absent from selected Cargo graph: {missing}")
    executed = excluded = 0
    failures = []
    for artifact in built:
        key = target_key(artifact, packages)
        entries = expected.get(key, [])
        if mode == "exhaustive" and not entries:
            continue
        binary = artifact["executable"]
        package = by_name[key[0]]
        context = {
            "cwd": str(Path(package["manifest_path"]).parent),
            "env": test_environment(metadata, package, artifact, output, rust_lib),
        }
        names = listed(run([binary, "--list", "--format=terse"], remaining(60), **context))
        ignored = listed(run([binary, "--ignored", "--list", "--format=terse"], remaining(60), **context))
        selected = validate_cases(entries, names, ignored)
        if mode == "routine":
            for name in sorted(selected):
                print(f"EXHAUSTIVE (not run in routine CI): {key}: {name}", flush=True)
            excluded += len(selected)
            args = [arg for name in sorted(selected) for arg in ("--skip", name)]
            try:
                # Healthy retained Candle artifact/memory-contract targets took
                # 411s on hosted CPU (run 40240078690). Bound above that evidence;
                # these tests are ordinary contracts, not numerical campaigns.
                result = run([binary, "--test-threads=1"] + args, remaining(900), **context)
                print(result, flush=True)
                count = len(names - ignored - selected)
                verify_result(result, count, len(ignored))
                executed += count
            except (ValueError, RuntimeError, TimeoutError) as error:
                if "--no-fail-fast" not in command:
                    raise
                failures.append(str(error))
        else:
            for name in sorted(selected):
                try:
                    result = run([binary, "--exact", name, "--test-threads=1"], remaining(600), **context)
                    print(result, flush=True)
                    verify_result(result, 1, 0)
                    executed += 1
                except (ValueError, RuntimeError, TimeoutError) as error:
                    if "--no-fail-fast" not in command:
                        raise
                    failures.append(str(error))
    if failures:
        raise RuntimeError("suite failures:\n" + "\n".join(failures))
    if mode == "exhaustive":
        required = sum(len(entries) for key, entries in expected.items() if key[0] in built_packages)
        if executed != required or not executed:
            raise ValueError(f"incomplete exhaustive suite: {executed}/{required}")
    print(f"{mode}: {executed} executed, {excluded} explicitly classified exhaustive cases excluded", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("routine", "exhaustive", "check"))
    parser.add_argument("--deadline-seconds", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cases = registry()
    if args.mode == "check":
        print(f"{len(cases)} reviewed exhaustive test identities")
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    execute(args.mode, command, cases, args.deadline_seconds or (7200 if args.mode == "exhaustive" else 2700))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TimeoutError) as error:
        print(f"::error::{error}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Capture a bounded A/B of fresh Rust proc-macro loading on Mac2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tomllib


SOURCE = "68b925baa5746281d47af9342cdcc28bc54d2ad0"
PACKAGES = {
    "castaway": ("0.2.4", "dec551ab6e7578819132c713a93c022a05d60159dc86e7a7050223577484c55a"),
    "rustversion": ("1.0.23", "cf54715a573b99ac80df0bc206da022bcd442c974952c7b9720069370852e21f"),
}
SAFE_ENV = ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "MACOSX_DEPLOYMENT_TARGET")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 30) -> dict:
    try:
        result = subprocess.run(command, env=env, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {"argv": command, "exitCode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "timedOut": False}
    except subprocess.TimeoutExpired as error:
        return {"argv": command, "exitCode": None, "stdout": error.stdout or "",
                "stderr": error.stderr or "", "timedOut": True}
    except OSError as error:
        return {"argv": command, "exitCode": None, "stdout": "", "stderr": str(error),
                "timedOut": False, "spawnError": True}


def validate_lock(path: Path) -> dict:
    rows = tomllib.loads(path.read_text(encoding="utf-8"))["package"]
    found = {}
    for name, (version, checksum) in PACKAGES.items():
        matches = [row for row in rows if row["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"{name} must occur exactly once")
        row = matches[0]
        if row.get("version") != version or row.get("checksum") != checksum:
            raise ValueError(f"{name} identity changed: {row}")
        found[name] = {key: row.get(key) for key in ("version", "source", "checksum")}
    return found


def artifacts(target: Path) -> list[dict]:
    rows = []
    for path in sorted((target / "release" / "deps").glob("librustversion-*.dylib")):
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            details = path.stat()
            item.update(mode=stat.filemode(details.st_mode), bytes=details.st_size, sha256=sha256(path),
                        file=capture(["file", str(path)]), otool=capture(["otool", "-L", str(path)]))
        rows.append(item)
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: mac2-proc-macro-health.py OUTPUT")
    output = Path(sys.argv[1]).resolve()
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    if runner_temp not in output.parents or output.exists():
        raise SystemExit("new output below RUNNER_TEMP required")
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[1-9][0-9]*", attempt):
        raise SystemExit("numeric run identity required")
    output.mkdir(parents=True)
    source_lock = Path(os.environ["PROC_MACRO_SOURCE_LOCK"])
    source_config = Path(os.environ["PROC_MACRO_SOURCE_CONFIG"])
    source_packages = validate_lock(source_lock)
    probe = output / "probe"
    (probe / "src").mkdir(parents=True)
    (probe / ".cargo").mkdir()
    (probe / ".cargo" / "config.toml").write_bytes(source_config.read_bytes())
    (probe / "Cargo.toml").write_text(
        '[package]\nname="mac2-proc-macro-health"\nversion="0.0.0"\nedition="2021"\n'
        '[dependencies]\ncastaway="=0.2.4"\nrustversion="=1.0.23"\n', encoding="utf-8")
    (probe / "src" / "lib.rs").write_text(
        '#[rustversion::since(1.51)] const RUSTVERSION_LOADED: bool = true;\n'
        'pub fn loaded() -> bool { RUSTVERSION_LOADED }\n', encoding="utf-8")
    generated = capture(["cargo", "generate-lockfile", "--offline", "--manifest-path", str(probe / "Cargo.toml")], timeout=60)
    (output / "generate-lock.json").write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    if generated["exitCode"] != 0:
        raise SystemExit("offline lock generation failed")
    probe_packages = validate_lock(probe / "Cargo.lock")
    facts = {
        "schemaVersion": 1, "captureComplete": False, "compilationAcceptance": None,
        "source": SOURCE, "runId": int(run_id), "runAttempt": int(attempt),
        "platform": {"machine": platform.machine(), "system": platform.system(), "release": platform.release()},
        "allowlistedEnvironment": {name: {"set": name in os.environ, "value": os.environ.get(name)} for name in SAFE_ENV},
        "tools": {name: capture([name, "-vV"] if name in ("rustc", "cargo") else [name, "--version"])
                  for name in ("rustc", "cargo", "sccache")},
        "sysroot": capture(["rustc", "--print", "sysroot"]),
        "sourceLock": {"path": str(source_lock), "sha256": sha256(source_lock), "packages": source_packages},
        "sourceConfig": {"path": str(source_config), "sha256": sha256(source_config)},
        "probeLock": {"sha256": sha256(probe / "Cargo.lock"), "packages": probe_packages},
        "arms": [],
    }
    for name, clear_wrappers in (("inherited", False), ("direct", True)):
        target = runner_temp / f"proc-macro-{name}-{run_id}-{attempt}"
        if target.exists():
            raise SystemExit(f"target already exists: {target}")
        target.mkdir()
        arm_env = os.environ.copy()
        if clear_wrappers:
            arm_env["RUSTC_WRAPPER"] = ""
            arm_env["RUSTC_WORKSPACE_WRAPPER"] = ""
        command = ["cargo", "check", "--locked", "--offline", "--release",
                   "--message-format=json", "-vv", "--manifest-path", str(probe / "Cargo.toml")]
        before = artifacts(target)
        result = capture(command, env={**arm_env, "CARGO_TARGET_DIR": str(target)}, timeout=240)
        stdout_path, stderr_path = output / f"{name}.jsonl", output / f"{name}.verbose.log"
        stdout_path.write_text(result.pop("stdout"), encoding="utf-8")
        stderr_path.write_text(result.pop("stderr"), encoding="utf-8")
        facts["arms"].append({"name": name, "wrappersCleared": clear_wrappers,
                              "target": str(target), "process": result,
                              "artifactsBefore": before, "artifactsAfter": artifacts(target),
                              "stdoutSha256": sha256(stdout_path), "stderrSha256": sha256(stderr_path)})
    facts["captureComplete"] = True
    facts["compilationAcceptance"] = {row["name"]: row["process"]["exitCode"] == 0 for row in facts["arms"]}
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

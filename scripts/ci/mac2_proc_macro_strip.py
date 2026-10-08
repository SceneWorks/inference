#!/usr/bin/env python3
"""Confirm the macOS 27 proc-macro strip failure with one controlled delta."""

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
SAFE_ENV = ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS", "MACOSX_DEPLOYMENT_TARGET",
            "CARGO_PROFILE_RELEASE_STRIP")


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
        return {"argv": command, "exitCode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "timedOut": False}
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


def macho_fields(text: str) -> dict:
    values: dict[str, int | None] = {key: None for key in ("stroff", "strsize", "minos", "sdk")}
    for key in values:
        match = re.search(rf"(?m)^\s*{key}\s+([0-9]+(?:\.[0-9]+)?)\s*$", text)
        if match:
            values[key] = match.group(1) if "." in match.group(1) else int(match.group(1))
    stroff = values["stroff"]
    values["stroffMod8"] = None if not isinstance(stroff, int) else stroff % 8
    return values


def artifacts(target: Path) -> list[dict]:
    rows = []
    for path in sorted((target / "release" / "deps").glob("librustversion-*.dylib")):
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            details = path.stat()
            load_commands = capture(["otool", "-l", str(path)])
            dlopen = capture([sys.executable, "-c",
                              "import ctypes,sys; ctypes.CDLL(sys.argv[1]); print('dlopen-ok')",
                              str(path)], timeout=15)
            item.update(mode=stat.filemode(details.st_mode), bytes=details.st_size,
                        sha256=sha256(path), file=capture(["file", str(path)]),
                        linkedLibraries=capture(["otool", "-L", str(path)]),
                        loadCommands=load_commands,
                        macho=macho_fields(load_commands.get("stdout", "")),
                        dlopen=dlopen,
                        codesign=capture(["codesign", "-dv", "--verbose=4", str(path)]))
        rows.append(item)
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: mac2-proc-macro-strip.py OUTPUT")
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
        '[package]\nname="mac2-proc-macro-strip"\nversion="0.0.0"\nedition="2021"\n'
        '[dependencies]\ncastaway="=0.2.4"\nrustversion="=1.0.23"\n', encoding="utf-8")
    (probe / "src" / "lib.rs").write_text(
        '#[rustversion::since(1.51)] const RUSTVERSION_LOADED: bool = true;\n'
        'pub fn loaded() -> bool { RUSTVERSION_LOADED }\n', encoding="utf-8")
    generated = capture(["cargo", "generate-lockfile", "--offline", "--manifest-path",
                         str(probe / "Cargo.toml")], timeout=60)
    (output / "generate-lock.json").write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    if generated["exitCode"] != 0:
        raise SystemExit("offline lock generation failed")
    probe_packages = validate_lock(probe / "Cargo.lock")
    facts = {
        "schemaVersion": 1, "captureComplete": False, "compilationAcceptance": None,
        "causalAcceptance": None, "source": SOURCE, "runId": int(run_id),
        "runAttempt": int(attempt),
        "platform": {"machine": platform.machine(), "system": platform.system(),
                     "release": platform.release()},
        "allowlistedEnvironment": {name: {"set": name in os.environ, "value": os.environ.get(name)}
                                   for name in SAFE_ENV},
        "tools": {name: capture([name, "-vV"] if name in ("rustc", "cargo") else
                                [name, "--version"]) for name in ("rustc", "cargo")},
        "sysroot": capture(["rustc", "--print", "sysroot"]),
        "sourceLock": {"path": str(source_lock), "sha256": sha256(source_lock),
                       "packages": source_packages},
        "sourceConfig": {"path": str(source_config), "sha256": sha256(source_config)},
        "probeLock": {"sha256": sha256(probe / "Cargo.lock"), "packages": probe_packages},
        "controlledVariable": "CARGO_PROFILE_RELEASE_STRIP", "arms": [],
    }
    for name, strip_value in (("baseline", None), ("no-strip", "none")):
        target = runner_temp / f"proc-macro-strip-{name}-{run_id}-{attempt}"
        if target.exists():
            raise SystemExit(f"target already exists: {target}")
        target.mkdir()
        arm_env = os.environ.copy()
        arm_env["RUSTC_WRAPPER"] = ""
        arm_env["RUSTC_WORKSPACE_WRAPPER"] = ""
        arm_env.pop("CARGO_PROFILE_RELEASE_STRIP", None)
        if strip_value is not None:
            arm_env["CARGO_PROFILE_RELEASE_STRIP"] = strip_value
        command = ["cargo", "check", "--locked", "--offline", "--release",
                   "--message-format=json", "-vv", "--manifest-path", str(probe / "Cargo.toml")]
        result = capture(command, env={**arm_env, "CARGO_TARGET_DIR": str(target)}, timeout=240)
        stdout_path, stderr_path = output / f"{name}.jsonl", output / f"{name}.verbose.log"
        stdout_path.write_text(result.pop("stdout"), encoding="utf-8")
        stderr_path.write_text(result.pop("stderr"), encoding="utf-8")
        facts["arms"].append({"name": name, "releaseStrip": strip_value,
                              "wrappersCleared": True, "target": str(target), "process": result,
                              "artifactsAfter": artifacts(target),
                              "stdoutSha256": sha256(stdout_path), "stderrSha256": sha256(stderr_path)})
    facts["captureComplete"] = True
    facts["compilationAcceptance"] = {row["name"]: row["process"]["exitCode"] == 0
                                      for row in facts["arms"]}
    baseline, no_strip = facts["arms"]
    facts["causalAcceptance"] = (not facts["compilationAcceptance"]["baseline"] and
                                 facts["compilationAcceptance"]["no-strip"] and
                                 any(a.get("macho", {}).get("stroffMod8") not in (None, 0)
                                     and a.get("dlopen", {}).get("exitCode") != 0
                                     for a in baseline["artifactsAfter"]) and
                                 any(a.get("macho", {}).get("stroffMod8") == 0
                                     and a.get("dlopen", {}).get("exitCode") == 0
                                     for a in no_strip["artifactsAfter"]))
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

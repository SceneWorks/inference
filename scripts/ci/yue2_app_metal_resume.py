#!/usr/bin/env python3
"""Resume only the q4 Metal profile after verified run 36565987342-1 acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from yue2_app_metal import (
    APP_SHA, CASE, ENGINE_SHA, check_sources, run_logged, verify_acceptance,
    verify_profile,
)

ORIGINAL_RUN = "36565987342-1"
SUMMARY_SHA256 = "789d4d3d8cb35418fa02e70fc0af338016ed64db12a1ba3618545d3bcea95890"
CHAIN_SHA256 = "42a8e8a972e7d67a6362efd350d1f690ff26b5b14c9ecb0ee0ca30813764cfbb"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_original(state: Path, out: Path, target: Path, preflight: Path) -> dict:
    """Bind the resume to the exact retained host state and completed acceptance."""
    expected = {
        state: f"yue2-app-metal-state-{ORIGINAL_RUN}",
        out: f"yue2-app-metal-out-{ORIGINAL_RUN}",
        target: f"yue2-app-metal-target-{ORIGINAL_RUN}",
    }
    if len({path.parent for path in expected}) != 1 or any(
        path.name != name for path, name in expected.items()
    ):
        raise ValueError("resume paths are not the same original run's state/out/target")
    if not all(path.is_dir() for path in expected):
        raise ValueError("original run state, output, or build target is absent")
    if not (state / "app-data").is_dir() or not (state / "hf-home").is_dir():
        raise ValueError("original app data or model cache is absent")
    if not (target / "release" / "sceneworks-rust-api").is_file():
        raise ValueError("original release build target is absent")
    if (out / "profile").exists() or (out / "profile.log").exists():
        raise ValueError("original profile has already started; refusing a repeat")
    identity = json.loads((out / "identity.json").read_text())
    if (identity.get("app_sha") != APP_SHA or identity.get("engine_sha") != ENGINE_SHA or
        identity.get("runner") != "nax-macos-2" or identity.get("state") != str(state) or
        identity.get("out") != str(out)):
        raise ValueError("original run identity differs from retained state/output")
    admitted = json.loads(preflight.read_text())
    if admitted.get("runner") != "nax-macos-2" or admitted.get("admitted") is not True:
        raise ValueError("resume host preflight did not admit the run")
    summary = out / "acceptance" / "evidence" / "summary.json"
    chain = out / "watchdog-acceptance.jsonl"
    if file_sha256(summary) != SUMMARY_SHA256 or file_sha256(chain) != CHAIN_SHA256:
        raise ValueError("original acceptance summary or watchdog chain differs from published run")
    return {"summary": summary, "chain": chain, "identity": identity}


def evidence_digest(app: Path, evidence_dir: Path) -> str:
    return subprocess.check_output(
        ["node", "--input-type=module", "-e",
         'import {syncEvidence} from "./scripts/lib/watchdog-completion.mjs"; '
         'console.log(await syncEvidence(process.argv[1]));', str(evidence_dir)],
        cwd=app, text=True,
    ).strip()


def copy_profile_receipts(out: Path, evidence: Path) -> None:
    profile = out / "profile" / CASE.replace(":", "__")
    for source in profile.rglob("*") if profile.is_dir() else ():
        if source.is_file() and source.suffix in (".json", ".jsonl"):
            dest = evidence / "profile" / source.relative_to(profile)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for name in ("profile.log", "profile-check.log"):
        source = out / name
        if source.is_file():
            shutil.copy2(source, evidence / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("app", "engine", "state", "out", "target", "evidence", "preflight"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    app, engine, state, out, target, evidence = (
        getattr(args, name).resolve() for name in ("app", "engine", "state", "out", "target", "evidence")
    )
    check_sources(app, engine)
    original = verify_original(state, out, target, args.preflight)
    digest = evidence_digest(app, out / "acceptance" / "evidence")
    acceptance = verify_acceptance(original["summary"], original["chain"], 1, digest)
    if acceptance["watchdog_events"] != 754:
        raise ValueError("original acceptance event count differs from completed run")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "resume-identity.json").write_text(json.dumps({
        "original_run": ORIGINAL_RUN,
        "app_sha": APP_SHA,
        "engine_sha": ENGINE_SHA,
        "state": str(state), "out": str(out), "target": str(target),
        "summary_sha256": SUMMARY_SHA256, "chain_sha256": CHAIN_SHA256,
        "evidence_sha256": digest,
        "acceptance": acceptance,
    }, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env["HF_HOME"] = str(state / "hf-home")
    env["CARGO_TARGET_DIR"] = str(target)
    try:
        command = ["node", "scripts/yue2-memory-profile.mjs", "capture", "--case", CASE,
                   "--inference-repo", str(engine), "--data-dir", str(state / "app-data"),
                   "--out", str(out / "profile"), "--budget-minutes", "120"]
        rc = run_logged(command, cwd=app, log=out / "profile.log", env=env)
        if rc != 0:
            raise ValueError(f"targeted q4-default profile exited {rc}")
        record = out / "profile" / CASE.replace(":", "__") / "record.json"
        result = verify_profile(record)
        rc = run_logged(["node", "scripts/yue2-memory-profile.mjs", "check", str(record)],
                        cwd=app, log=out / "profile-check.log", env=env)
        if rc != 0:
            raise ValueError("targeted profile failed current-closure check")
        (evidence / "verdict.json").write_text(json.dumps({
            "original_acceptance": acceptance, "resumed_profile": result,
        }, indent=2) + "\n")
        print(f"resumed q4-default profile: {result}", flush=True)
        return 0
    finally:
        copy_profile_receipts(out, evidence)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"yue2-app-metal-resume: {error}", file=sys.stderr)
        raise SystemExit(1)

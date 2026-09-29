#!/usr/bin/env python3
"""One exact-revision Metal acceptance plus the missing q4-default profile capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

APP_SHA = "44dc4119fc64c48a38dbd4ca94a3a6c2cd05fbe3"
ENGINE_SHA = "99a60541a73706fb67b4e78f44458774423ffb8e"
RESERVE = 2 * 1024**3
INCIDENT_FOOTPRINT = 96_970_084_480
CASE = "yue2:q4:metal:default"
STAGES = ("load", "plan", "semantic", "acoustic", "decode")


def guard_ceiling(host_bytes: int) -> int:
    """The SceneWorks watchdogCeilings rule, with this runner's probed host size."""
    return min(INCIDENT_FOOTPRINT - RESERVE, host_bytes - RESERVE)


def revision(directory: Path) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def check_sources(app: Path, engine: Path) -> None:
    for directory, expected in ((app, APP_SHA), (engine, ENGINE_SHA)):
        if revision(directory) != expected or subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError(f"{directory}: wrong revision or dirty checkout")
    if not re.search(r'SceneWorks/inference", rev = "' + ENGINE_SHA + r'"', (app / "Cargo.toml").read_text()):
        raise ValueError("SceneWorks Cargo inference pin differs from exact M2")


def validate_chain(path: Path, evidence_digest: str) -> list[str]:
    prior = "0" * 64
    names = []
    events = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        event = json.loads(line)
        digest = event.pop("eventHash")
        actual = hashlib.sha256(json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        if event["eventSequence"] != index or event["previousEventHash"] != prior or digest != actual:
            raise ValueError(f"watchdog hash chain breaks at event {index}")
        prior = digest
        names.append(event["event"])
        events.append(event)
    if not names or any(name in names for name in ("hard_stop", "terminated")):
        raise ValueError("watchdog hard-stopped or recorded no events")
    required = ("child_completion_requested", "child_completion_measured", "child_completed")
    if any(names.count(name) != 1 for name in required) or [names.index(name) for name in required] != sorted(names.index(name) for name in required):
        raise ValueError("watchdog did not complete the nonce-bound final handshake")
    if "sample" not in names:
        raise ValueError("watchdog has no samples")
    request, measured, completed = (names.index(name) for name in required)
    final_samples = [i for i, event in enumerate(events) if
                     event["event"] == "sample" and event.get("phase") == "completion_before_release"]
    if len(final_samples) != 1 or not request < final_samples[0] < measured < completed:
        raise ValueError("final group/host sample did not precede completion")
    if any(events[i].get("evidenceSha256") != evidence_digest for i in (request, measured)):
        raise ValueError("completion digest differs from the persisted evidence directory")
    if not events[measured].get("rootIdentity"):
        raise ValueError("final measured completion has no root identity")
    return names


def verify_acceptance(summary_path: Path, chain_path: Path, rc: int, evidence_digest: str) -> dict:
    if rc != 1:
        raise ValueError(f"acceptance guard exited {rc}, expected the single owner-approved skip (1)")
    summary = json.loads(summary_path.read_text())
    skipped = [row for row in summary["cases"] if row["status"] == "skipped"]
    if (summary["verdict"] != "incomplete" or summary.get("fatal") or
        summary["counts"] != {"passed": 26, "failed": 0, "blocked": 0, "skipped": 1} or
        summary["missing"] or len(summary["cases"]) != 27 or len(skipped) != 1 or
        skipped[0]["caseId"] != "worker-kill-resume" or
        "signal-kill of a Metal process" not in skipped[0]["reason"]):
        raise ValueError("acceptance has an unexpected case verdict")
    identity = summary["identity"]
    if (identity["sceneworksRevision"] != APP_SHA or identity["inferencePin"] != ENGINE_SHA or
        identity.get("dirty") or summary["platform"] != "metal"):
        raise ValueError("acceptance identity differs from exact app/M2 Metal")
    names = validate_chain(chain_path, evidence_digest)
    return {"verdict": summary["verdict"], "counts": summary["counts"], "watchdog_events": len(names), "watchdog_exit": rc}


def verify_profile(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record["caseId"] != CASE or record["backend"] != "metal" or record["outcome"]["status"] != "completed":
        raise ValueError("targeted profile case did not complete")
    stages = record["measured"]["stages"]
    if any(stages.get(name, {}).get("samples", 0) < 1 or stages[name].get("peakBytes", 0) <= 0 for name in STAGES):
        raise ValueError("targeted profile is missing a measured stage, including load")
    if record["measured"].get("peakBytes", 0) <= 0:
        raise ValueError("targeted profile has no overall peak")
    return {"case": CASE, "stage_samples": {name: stages[name]["samples"] for name in STAGES},
            "peak_bytes": record["measured"]["peakBytes"]}


def run_logged(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> int:
    print(f"running {command[0]} {command[1]} (log: {log})", flush=True)
    with log.open("w") as output:
        return subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                              check=False).returncode


def probe_ffmpeg(binary: str | None) -> tuple[str, str]:
    if not binary:
        raise ValueError("YUE2_FFMPEG_BIN must name the staged, verified app ffmpeg")
    path = Path(binary)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("staged ffmpeg is absent or not an absolute executable")
    version = subprocess.check_output([str(path), "-version"], text=True).splitlines()[0]
    if not version.startswith("ffmpeg version "):
        raise ValueError("staged ffmpeg version probe failed")
    return str(path), version


def copy_receipts(out: Path, evidence: Path) -> None:
    for source in (out / "acceptance" / "evidence").rglob("*") if (out / "acceptance" / "evidence").exists() else ():
        if source.is_file() and source.suffix == ".json":
            dest = evidence / "acceptance" / source.relative_to(out / "acceptance" / "evidence")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    profile = out / "profile" / CASE.replace(":", "__")
    for name in ("case.json", "admission.json", "stages.jsonl", "metal-samples.jsonl",
                 "outcome.json", "record.json", "watchdog.jsonl"):
        source = profile / name
        if source.is_file():
            dest = evidence / "profile" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    boundary = profile / "boundary"
    if boundary.is_dir():
        for source in boundary.glob("*.json"):
            dest = evidence / "profile" / "boundary" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for name in ("acceptance.log", "profile.log", "watchdog-acceptance.jsonl", "identity.json"):
        source = out / name
        if source.is_file():
            shutil.copy2(source, evidence / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("app", "engine", "state", "out", "evidence", "api-bin", "preflight"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    app, engine, state, out, evidence = (getattr(args, name).resolve() for name in
                                         ("app", "engine", "state", "out", "evidence"))
    check_sources(app, engine)
    if state.exists() or out.exists():
        raise ValueError("app state and output must both be fresh")
    preflight = json.loads(args.preflight.read_text())
    if not preflight["admitted"] or preflight["runner"] != "nax-macos-2":
        raise ValueError("designated host preflight did not admit the run")
    host_bytes = preflight["physical_memory_bytes"]
    ceiling = guard_ceiling(host_bytes)
    if ceiling <= 0:
        raise ValueError("host has no positive watchdog footprint ceiling")
    ffmpeg, version = probe_ffmpeg(os.environ.get("YUE2_FFMPEG_BIN"))
    if not args.api_bin.is_file():
        raise ValueError("release API binary is absent")
    state.mkdir(parents=True)
    out.mkdir(parents=True)
    identity = {"app_sha": APP_SHA, "engine_sha": ENGINE_SHA, "runner": preflight["runner"],
                "host_memory_bytes": host_bytes, "watchdog_max_footprint_bytes": ceiling,
                "watchdog_min_free_bytes": RESERVE, "ffmpeg": ffmpeg, "ffmpeg_version": version,
                "state": str(state), "out": str(out)}
    (out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        command = [sys.executable, "scripts/memory-calibration-watchdog.py",
                   "--max-footprint-bytes", str(ceiling), "--host-memory-bytes", str(host_bytes),
                   "--min-memory-free-bytes", str(RESERVE), "--max-runtime-seconds", "21600",
                   "--sample-interval", "2", "--telemetry-timeout", "10", "--term-grace", "1",
                   "--require-completion-handshake", "--event-file", str(out / "watchdog-acceptance.jsonl"),
                   "--", "node", "scripts/yue2-acceptance.mjs", "--platform", "metal",
                   "--out", str(out / "acceptance"), "--data-dir", str(state / "app-data"),
                   "--hf-home", str(state / "hf-home"), "--api-bin", str(args.api_bin.resolve()),
                   "--ffmpeg-bin", ffmpeg]
        rc = run_logged(command, cwd=app, log=out / "acceptance.log")
        evidence_dir = out / "acceptance" / "evidence"
        evidence_digest = subprocess.check_output(
            ["node", "--input-type=module", "-e",
             'import {syncEvidence} from "./scripts/lib/watchdog-completion.mjs"; '
             'console.log(await syncEvidence(process.argv[1]));',
             str(evidence_dir)],
            cwd=app, text=True,
        ).strip()
        receipt = verify_acceptance(out / "acceptance" / "evidence" / "summary.json",
                                    out / "watchdog-acceptance.jsonl", rc, evidence_digest)
        print(f"acceptance: {receipt}", flush=True)
        check = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("yue2_metal_preflight.py")),
             "--evidence", str(evidence), "--label", "before-profile"],
            cwd=Path(__file__).resolve().parents[2], check=False,
        )
        if check.returncode != 0:
            raise ValueError("runner became busy or resource-constrained before profile")
        env = os.environ.copy()
        env.pop("HF_HUB_CACHE", None)
        env.pop("HUGGINGFACE_HUB_CACHE", None)
        env["HF_HOME"] = str(state / "hf-home")
        command = ["node", "scripts/yue2-memory-profile.mjs", "capture", "--case", CASE,
                   "--inference-repo", str(engine), "--data-dir", str(state / "app-data"),
                   "--out", str(out / "profile"), "--budget-minutes", "120"]
        rc = run_logged(command, cwd=app, log=out / "profile.log", env=env)
        if rc != 0:
            raise ValueError(f"targeted q4-default profile exited {rc}")
        profile = out / "profile" / CASE.replace(":", "__") / "record.json"
        result = verify_profile(profile)
        check = run_logged(["node", "scripts/yue2-memory-profile.mjs", "check", str(profile)],
                           cwd=app, log=out / "profile-check.log", env=env)
        if check != 0:
            raise ValueError("targeted profile failed current-closure check")
        (out / "verdict.json").write_text(json.dumps({"acceptance": receipt, "profile": result}, indent=2) + "\n")
        print(f"targeted profile: {result}", flush=True)
        return 0
    finally:
        copy_receipts(out, evidence)
        for name in ("profile-check.log", "verdict.json"):
            source = out / name
            if source.is_file():
                shutil.copy2(source, evidence / name)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"yue2-app-metal: {error}", file=sys.stderr)
        raise SystemExit(1)

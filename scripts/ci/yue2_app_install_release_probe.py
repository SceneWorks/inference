"""Read-only process release check for the single stopped YuE2 app run.

This establishes a fresh, bounded absence observation, not a historical proof
that no child existed between the old job's last census and this probe.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys

from yue2_app_precision_profile import verify_sources
from yue2_precision_proof import cuda_physical_census, retain_cuda_physical_evidence


OLD_RUN_ID = "37295993157"
OLD_RUN_ATTEMPT = "1"
OLD_JOB_ID = "111717219895"
OLD_CONTROL_SHA = "6a14bcd709f68f8ebcb6a3b2a360fe04aa5b96d8"
OLD_APP_SHA = "c1f86907ae41183fa8ddc9126a821df36cc597dd"
OLD_ENGINE_SHA = "25bd55cdb6a56c78b07584a12150c9f5d46be439"
OLD_RUN_ROOT = r"E:\sceneworks-terminal\sc-23002-yue2-precision\37295993157-1"
OLD_WORKER_ID = "yue2-acceptance-ed59cf8e2088"
OLD_RUNNER = "cuda-windows"
OLD_HOST = "MICHAEL-TRX50"
OLD_JOB_STARTED = datetime.fromisoformat("2026-10-05T10:35:14+00:00")
OLD_JOB_COMPLETED = "2026-10-05T10:59:07+00:00"
OLD_METRICS_ZIP_SHA256 = "e4fe7f5ad2ed80b2f3294064f49f113ac5a83a0ef734a0bcfc3d5160cfead5ad"
RELEVANT_NAME = re.compile(
    r"(?:sceneworks-(?:rust-api|api|worker)|candle[^.]*|node|python(?:3(?:\.\d+)?)?|"
    r"powershell|pwsh|cmd|cargo|rustc|ffmpeg|nvidia-smi)\.exe", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc(value: object) -> datetime:
    require(isinstance(value, str) and bool(value), "process snapshot timestamp missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid process snapshot timestamp") from error
    require(parsed.tzinfo is not None, "process snapshot timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def has_old_root(value: str, root: str) -> bool:
    normalized = value.replace("/", "\\").casefold()
    target = root.casefold()
    start = 0
    while (found := normalized.find(target, start)) >= 0:
        after = found + len(target)
        if after == len(normalized) or normalized[after] in "\\\"' \t\r\n":
            return True
        start = found + 1
    return False


def validate_snapshot(payload: object, old_root: str = OLD_RUN_ROOT,
                      worker_id: str = OLD_WORKER_ID) -> dict:
    require(isinstance(payload, dict) and payload.get("complete") is True and
            isinstance(payload.get("rows"), list) and "error" not in payload,
            "Win32_Process snapshot incomplete")
    collector_pid, total = payload.get("collectorPid"), payload.get("totalCimCount")
    require(type(collector_pid) is int and collector_pid > 0 and type(total) is int and
            total >= len(payload["rows"]) >= 1,
            "Win32_Process enumeration has no collector/completeness witness")
    queried = utc(payload.get("queriedUtc"))
    completed = utc(payload.get("completedUtc"))
    require(queried <= completed and (completed - queried).total_seconds() <= 30,
            "Win32_Process snapshot stale or unbounded")
    seen = set()
    old_matches = []
    preexisting = 0
    collector = None
    for row in payload["rows"]:
        require(isinstance(row, dict) and isinstance(row.get("name"), str) and
                bool(row["name"].strip()), "unnamed process candidate")
        pid, parent = row.get("pid"), row.get("parentPid")
        require(type(pid) is int and pid > 0 and type(parent) is int and parent >= 0 and
                pid not in seen, "candidate PID/parent map incomplete or duplicated")
        seen.add(pid)
        created = utc(row.get("createdUtc"))
        require(created <= completed, "candidate creation is after snapshot")
        require("commandLine" not in row and type(row.get("commandLineAvailable")) is bool and
                type(row.get("oldRootInCommandLine")) is bool and
                type(row.get("oldRootInExecutable")) is bool and
                type(row.get("workerIdInCommandLine")) is bool,
                "raw command line leaked or ownership markers missing")
        executable = row.get("executablePath")
        require((RELEVANT_NAME.fullmatch(row["name"]) is not None or
                 row["oldRootInCommandLine"] or row["oldRootInExecutable"] or
                 row["workerIdInCommandLine"] or pid == collector_pid),
                "unreviewed nonmatching process row")
        available = row["commandLineAvailable"]
        digest, length = row.get("commandLineSha256"), row.get("commandLineLength")
        if available:
            require(type(length) is int and length > 0 and isinstance(digest, str) and
                    re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                    "candidate command-line digest missing")
        matches = (isinstance(executable, str) and has_old_root(executable, old_root)) or \
            row["oldRootInCommandLine"] or row["oldRootInExecutable"] or row["workerIdInCommandLine"]
        if matches:
            old_matches.append({"pid": pid, "createdUtc": row["createdUtc"], "name": row["name"]})
        if not (isinstance(executable, str) and PureWindowsPath(executable).is_absolute() and available):
            require(created < OLD_JOB_STARTED,
                    "newer candidate executable or command line inaccessible")
            preexisting += 1
        require(worker_id == OLD_WORKER_ID, "worker identity source mismatch")
        if pid == collector_pid:
            require(row["name"].casefold() in ("powershell.exe", "pwsh.exe") and
                    isinstance(executable, str) and
                    PureWindowsPath(executable).name.casefold() == row["name"].casefold(),
                    "collector PID is not the accessible PowerShell process")
            collector = {"pid": pid, "name": row["name"], "createdUtc": row["createdUtc"],
                         "executablePath": executable}
    require(collector is not None, "collector process missing from complete CIM enumeration")
    return {"queriedUtc": payload["queriedUtc"], "completedUtc": payload["completedUtc"],
            "candidateCount": len(seen), "preexistingIncompleteCandidates": preexisting,
            "collector": collector, "totalCimCount": total, "oldOwnedMatches": old_matches}


def validate_pair(before: object, after: object) -> dict:
    first = validate_snapshot(before)
    second = validate_snapshot(after)
    separation = (utc(second["queriedUtc"]) - utc(first["completedUtc"])).total_seconds()
    require(2 <= separation <= 30, "process snapshots overlap, are too close, or are stale")
    require(first["collector"] == second["collector"],
            "collector PID/name/creation changed between process snapshots")
    require(not first["oldOwnedMatches"] and not second["oldOwnedMatches"],
            "old app install process or worker remains present")
    return {"before": first, "after": second, "noOldOwnedMatch": True}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def collect(evidence: Path, app: Path, engine: Path, control: Path) -> dict:
    require(os.name == "nt" and os.environ.get("GITHUB_REPOSITORY") == "SceneWorks/inference" and
            os.environ.get("GITHUB_JOB") == "cuda_release_check" and
            os.environ.get("GITHUB_RUN_ATTEMPT") == "1" and
            os.environ.get("RUNNER_NAME") in ("cuda-windows", "cuda-windows-2") and
            os.environ.get("COMPUTERNAME", "").upper() == OLD_HOST and
            os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and
            not os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID"),
            "release probe dispatch/runner/device identity mismatch")
    require(os.environ.get("EXPECTED_APP_SHA") == OLD_APP_SHA and
            os.environ.get("EXPECTED_ENGINE_SHA") == OLD_ENGINE_SHA,
            "release probe targets the wrong historical app or engine")
    sources = verify_sources(app, engine, control, OLD_APP_SHA, OLD_ENGINE_SHA,
                             os.environ["EXPECTED_CONTROL_SHA"])
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    target = {"runId": OLD_RUN_ID, "attempt": OLD_RUN_ATTEMPT,
              "jobId": OLD_JOB_ID, "jobCompletedUtc": OLD_JOB_COMPLETED,
              "controlSha": OLD_CONTROL_SHA, "appSha": OLD_APP_SHA,
              "engineSha": OLD_ENGINE_SHA, "runner": OLD_RUNNER,
              "runRoot": OLD_RUN_ROOT, "workerId": OLD_WORKER_ID,
              "jobStartedUtc": OLD_JOB_STARTED.isoformat(),
              "metricsZipSha256": OLD_METRICS_ZIP_SHA256}
    (evidence / "target.json").write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
    raw, busy = cuda_physical_census(admission=True)
    (evidence / "physical-preflight.json").write_text(raw + "\n", encoding="utf-8")
    physical_files = retain_cuda_physical_evidence(evidence, "initial", raw) if not busy else None
    script = Path(__file__).with_name("yue2_app_install_release_processes.ps1")
    environment = dict(os.environ, YUE2_RELEASE_OLD_ROOT=OLD_RUN_ROOT,
                       YUE2_RELEASE_WORKER_ID=OLD_WORKER_ID)
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", str(script), "-OutputDirectory", str(evidence)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=90, env=environment)
    require(process.returncode == 0, "Win32_Process read-only snapshots failed")
    verdict = validate_pair(read_json(evidence / "process-snapshot-before.json"),
                            read_json(evidence / "process-snapshot-after.json"))
    elapsed = (datetime.now(timezone.utc) - utc(verdict["after"]["completedUtc"])).total_seconds()
    require(0 <= elapsed <= 120, "process release observation is not fresh")
    require(not busy, "selected GPU1 physical preflight refused: " + "; ".join(busy))
    process_files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                     for path in (evidence / "process-snapshot-before.json",
                                  evidence / "process-snapshot-after.json")]
    return {"schema": 1, "purpose": "read-only release observation only",
            "oldRun": target, "targetSha256": file_sha256(evidence / "target.json"),
            "probe": {"runId": os.environ.get("GITHUB_RUN_ID"),
                      "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                      "job": os.environ.get("GITHUB_JOB"), "runner": os.environ.get("RUNNER_NAME"),
                      "hostname": os.environ.get("COMPUTERNAME"),
                      "controlSha": os.environ.get("GITHUB_SHA")},
            "physicalFiles": physical_files, "processFiles": process_files,
            "processes": verdict,
            "historicalIntervalProven": False, "releasedNow": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "app", "engine", "control"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = collect(args.evidence, args.app, args.engine, args.control)
        (args.evidence / "verdict.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"releasedNow": True, "beforeCandidates": result["processes"]["before"]["candidateCount"],
                          "afterCandidates": result["processes"]["after"]["candidateCount"],
                          "physicalFileCount": len(result["physicalFiles"]),
                          "processDigests": [row["sha256"] for row in result["processFiles"]]}), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        if args.evidence.is_dir():
            (args.evidence / "refusal.json").write_text(json.dumps({"releasedNow": False,
                "reason": str(error)}, indent=2) + "\n", encoding="utf-8")
        print("yue2-app-install-release-probe: refused; inspect sanitized artifact", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

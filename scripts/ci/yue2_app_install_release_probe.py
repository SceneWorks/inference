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
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

from yue2_app_precision_profile import CASES, CASE_SOURCE_SHA256, NAMES, case_id, verify_sources
from yue2_precision_proof import cuda_physical_census, retain_cuda_physical_evidence


TARGET_RUN_ID = 37314391667
TARGET_ATTEMPT = 1
TARGET_JOB_ID = 111777364558
TARGET_ARTIFACT_ID = 11348636766
TARGET_CONTROL_SHA = "2c820231926094566d1ed719c085ba7a7a2284a5"
TARGET_APP_SHA = "c1f86907ae41183fa8ddc9126a821df36cc597dd"
TARGET_ENGINE_SHA = "25bd55cdb6a56c78b07584a12150c9f5d46be439"
TARGET_RUN_ROOT = r"E:\sceneworks-terminal\sc-23002-yue2-precision\37314391667-1"
TARGET_RUNNER = "cuda-windows-2"
TARGET_RUNNER_ID = 2619
ALLOWED_RELEASE_RUNNERS = ("cuda-windows", "cuda-windows-2")
TARGET_REPOSITORY = "SceneWorks/inference"
TARGET_WORKFLOW = ".github/workflows/yue2-app-precision-profile.yml"
TARGET_ARTIFACT_NAME = (f"yue2-app-precision-cuda-engine-{TARGET_ENGINE_SHA}-control-"
                       f"{TARGET_CONTROL_SHA}-{TARGET_RUN_ID}-{TARGET_ATTEMPT}")
RELEVANT_NAME = re.compile(
    r"(?:sceneworks-(?:rust-api|api|worker)|sceneworks_worker-[0-9a-f]{16}|candle[^.]*|node|python(?:3(?:\.\d+)?)?|"
    r"powershell|pwsh|cmd|cargo|rustc|ffmpeg|nvidia-smi)\.exe", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class TransientCandidateError(ValueError):
    """A complete, sanitized census with a newer process identity inaccessible."""

    def __init__(self, snapshot: dict, candidates: list[dict]):
        super().__init__("newer candidate executable or command line inaccessible")
        self.snapshot = snapshot
        self.candidates = candidates


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


def validate_snapshot(payload: object, old_root: str = TARGET_RUN_ROOT,
                      job_started: str = "2026-10-05T10:35:14+00:00",
                      recorded_generations: list[dict] | None = None) -> dict:
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
    transient_candidates = []
    generation_matches = []
    collector = None
    recorded_generations = recorded_generations or []
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
        matching_generation = next((item for item in recorded_generations
            if item.get("pid") == pid and utc(item.get("preciseCreatedUtc")) == created), None)
        matches = (isinstance(executable, str) and has_old_root(executable, old_root)) or \
            row["oldRootInCommandLine"] or row["oldRootInExecutable"] or \
            row["workerIdInCommandLine"] or matching_generation is not None
        if matches:
            old_matches.append({"pid": pid, "createdUtc": row["createdUtc"], "name": row["name"]})
        if matching_generation is not None:
            generation_matches.append({"caseId": matching_generation["caseId"],
                "pid": pid, "createdUtc": row["createdUtc"], "name": row["name"]})
        if not (isinstance(executable, str) and PureWindowsPath(executable).is_absolute() and available):
            if created < utc(job_started):
                preexisting += 1
            else:
                transient_candidates.append({"pid": pid, "name": row["name"],
                                             "createdUtc": row["createdUtc"]})
        if pid == collector_pid:
            require(row["name"].casefold() in ("powershell.exe", "pwsh.exe") and
                    isinstance(executable, str) and PureWindowsPath(executable).is_absolute() and
                    available and
                    PureWindowsPath(executable).name.casefold() == row["name"].casefold(),
                    "collector PID is not the accessible PowerShell process")
            collector = {"pid": pid, "name": row["name"], "createdUtc": row["createdUtc"],
                         "executablePath": executable}
    require(collector is not None, "collector process missing from complete CIM enumeration")
    require(not old_matches, "old app install process or worker remains present")
    result = {"queriedUtc": payload["queriedUtc"], "completedUtc": payload["completedUtc"],
              "candidateCount": len(seen), "preexistingIncompleteCandidates": preexisting,
              "collector": collector, "totalCimCount": total, "oldOwnedMatches": old_matches,
              "recordedGenerationMatches": generation_matches}
    if transient_candidates:
        raise TransientCandidateError(result, transient_candidates)
    return result


def validate_pair(before: object, after: object, binding: dict | None = None) -> dict:
    target = binding or {}
    generations = target.get("caseProcessWitnesses", [])
    run = target.get("target", {})
    first = validate_snapshot(before, target.get("runRoot", TARGET_RUN_ROOT),
        run.get("jobStartedUtc", "2026-10-05T10:35:14+00:00"), generations)
    second = validate_snapshot(after, target.get("runRoot", TARGET_RUN_ROOT),
        run.get("jobStartedUtc", "2026-10-05T10:35:14+00:00"), generations)
    separation = (utc(second["queriedUtc"]) - utc(first["completedUtc"])).total_seconds()
    require(2 <= separation <= 30, "process snapshots overlap, are too close, or are stale")
    require(first["collector"] == second["collector"],
            "collector PID/name/creation changed between process snapshots")
    require(not first["oldOwnedMatches"] and not second["oldOwnedMatches"],
            "old app install process or worker remains present")
    return {"before": first, "after": second, "noOldOwnedMatch": True}


def validate_release_pairs(evidence: Path, collector, binding: dict | None = None) -> dict:
    """Accept a final fully valid pair, preserving the original two-observation guarantee."""
    pairs = []
    transient_refusals = []
    stable_collector = None
    previous_completed = None
    try:
        for pair_number in range(1, 4):
            before_path, after_path = collector.read_pair(pair_number)
            snapshots = []
            transient = []
            for path in (before_path, after_path):
                payload = read_json(path)
                try:
                    target = binding or {}
                    run = target.get("target", {})
                    snapshot = validate_snapshot(payload, target.get("runRoot", TARGET_RUN_ROOT),
                        run.get("jobStartedUtc", "2026-10-05T10:35:14+00:00"),
                        target.get("caseProcessWitnesses", []))
                except TransientCandidateError as error:
                    snapshot = error.snapshot
                    transient.extend(error.candidates)
                snapshots.append(snapshot)
            before, after = snapshots
            separation = (utc(after["queriedUtc"]) - utc(before["completedUtc"])).total_seconds()
            require(2 <= separation <= 30,
                    "process snapshots overlap, are too close, or are stale")
            require(before["collector"] == after["collector"],
                    "collector PID/name/creation changed between process snapshots")
            if stable_collector is not None:
                require(before["collector"] == stable_collector,
                        "collector PID/name/creation changed between process pairs")
                inter_pair = (utc(before["queriedUtc"]) - utc(previous_completed)).total_seconds()
                require(2 <= inter_pair <= 30,
                        "process snapshot pairs overlap, are too close, or are stale")
            stable_collector = before["collector"]
            previous_completed = after["completedUtc"]
            elapsed = (datetime.now(timezone.utc) - utc(after["completedUtc"])).total_seconds()
            require(0 <= elapsed <= 120, "process release observation is not fresh")

            pair = {"pair": pair_number, "before": before, "after": after,
                    "noOldOwnedMatch": True, "valid": not transient}
            pairs.append(pair)
            if transient:
                refusal = {"pair": pair_number,
                           "reason": "newer candidate executable or command line inaccessible",
                           "candidates": transient}
                refusal_path = evidence / ("initial-refusal.json" if pair_number == 1
                                           else f"process-pair-{pair_number}-refusal.json")
                refusal_path.write_text(json.dumps(refusal, indent=2) + "\n", encoding="utf-8")
                transient_refusals.append(refusal)

            accepted = pair["valid"]
            should_continue = pair_number < 3 and not accepted
            collector.decide_continue(pair_number, should_continue)
            if accepted:
                break
        require(bool(pairs) and pairs[-1]["valid"],
                "no final complete process snapshot pair proved release")
        collector.finish()
        snapshots = [path for pair_number in range(1, len(pairs) + 1)
                     for path in collector.snapshot_paths(pair_number)]
        process_files = [{"name": path.name, "bytes": path.stat().st_size,
                          "sha256": file_sha256(path)} for path in snapshots]
        refusal_files = []
        for refusal in transient_refusals:
            path = evidence / ("initial-refusal.json" if refusal["pair"] == 1
                               else f"process-pair-{refusal['pair']}-refusal.json")
            refusal_files.append({"name": path.name, "bytes": path.stat().st_size,
                                 "sha256": file_sha256(path)})
        final_elapsed = (datetime.now(timezone.utc) - utc(pairs[-1]["after"]["completedUtc"])).total_seconds()
        require(0 <= final_elapsed <= 120, "process release observation is not fresh")
        return {"before": pairs[-1]["before"], "after": pairs[-1]["after"],
                "pairs": pairs, "transientRefusals": transient_refusals,
                "noOldOwnedMatch": True, "processFiles": process_files,
                "refusalFiles": refusal_files}
    except BaseException:
        try:
            collector.stop()
        finally:
            try:
                collector.finish()
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                abort = getattr(collector, "abort", None)
                if abort is not None:
                    abort()
        raise


class PowerShellSnapshotCollector:
    """One bounded PowerShell process, advanced only by the validated pair decision."""

    MAX_RUNTIME_SECONDS = 135
    PAIR_READY_SECONDS = 40

    def __init__(self, evidence: Path, script: Path, environment: dict):
        self.evidence = evidence
        self.started = time.monotonic()
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(script), "-OutputDirectory", str(evidence)]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, env=environment)
        self.current_pair = 1

    def snapshot_paths(self, pair_number: int) -> tuple[Path, Path]:
        if pair_number == 1:
            stem = "process-snapshot"
            return (self.evidence / f"{stem}-before.json", self.evidence / f"{stem}-after.json")
        return (self.evidence / f"process-snapshot-{pair_number}-before.json",
                self.evidence / f"process-snapshot-{pair_number}-after.json")

    def read_pair(self, pair_number: int) -> tuple[Path, Path]:
        self.current_pair = pair_number
        ready = self.evidence / f"process-pair-{pair_number}.ready"
        deadline = time.monotonic() + self.PAIR_READY_SECONDS
        while not ready.is_file():
            if self.process.poll() is not None:
                raise ValueError("Win32_Process snapshot collector exited early")
            if time.monotonic() >= deadline:
                raise ValueError("Win32_Process snapshot pair timed out")
            time.sleep(0.1)
        require(ready.read_text(encoding="ascii") == str(pair_number),
                "Win32_Process snapshot collector pair witness changed")
        paths = self.snapshot_paths(pair_number)
        require(all(path.is_file() and not path.is_symlink() for path in paths),
                "Win32_Process sanitized snapshot missing")
        return paths

    def decide_continue(self, pair_number: int, should_continue: bool) -> None:
        if pair_number >= 3:
            return
        path = self.evidence / f"process-pair-{pair_number}.decision"
        require(not path.exists(), "Win32_Process pair decision already exists")
        path.write_text("continue" if should_continue else "stop", encoding="ascii")

    def stop(self) -> None:
        if self.process.poll() is None and self.current_pair < 3:
            path = self.evidence / f"process-pair-{self.current_pair}.decision"
            if not path.exists():
                path.write_text("stop", encoding="ascii")

    def finish(self) -> None:
        if self.process.poll() is not None:
            require(self.process.returncode == 0, "Win32_Process read-only snapshots failed")
            return
        remaining = self.MAX_RUNTIME_SECONDS - (time.monotonic() - self.started)
        if remaining <= 0:
            self.abort()
            raise ValueError("Win32_Process snapshot collector exceeded its bound")
        try:
            status = self.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            self.abort()
            raise ValueError("Win32_Process snapshot collector exceeded its bound")
        require(status == 0, "Win32_Process read-only snapshots failed")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def safe_extract_metrics(zip_path: Path, artifact: dict, destination: Path) -> dict:
    """Verify the original GitHub artifact digest, then safely extract its metrics ZIP."""
    require(zip_path.is_file() and not zip_path.is_symlink() and not destination.exists(),
            "metrics ZIP missing, linked, or extraction destination already exists")
    digest = file_sha256(zip_path)
    require(artifact.get("id") == TARGET_ARTIFACT_ID and artifact.get("name") == TARGET_ARTIFACT_NAME and
            artifact.get("expired") is False and artifact.get("workflow_run", {}).get("head_sha") ==
            TARGET_CONTROL_SHA and artifact.get("digest") == "sha256:" + digest,
            "original metrics artifact identity or ZIP digest differs")
    destination.mkdir(parents=True)
    names = set()
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            path = PureWindowsPath(item.filename)
            parts = Path(item.filename).parts
            require(item.filename and not path.is_absolute() and not path.drive and ":" not in item.filename and
                    ".." not in parts and
                    ".." not in path.parts and not item.filename.startswith(("/", "\\")) and
                    item.filename not in names,
                    "metrics ZIP contains an unsafe or duplicate path")
            names.add(item.filename)
            mode = (item.external_attr >> 16) & 0o170000
            require(mode not in (0o120000, 0o060000, 0o020000, 0o010000),
                    "metrics ZIP contains a link or special file")
        archive.extractall(destination)
    roots = {path.name: path for path in destination.iterdir()
             if path.is_dir() and not path.is_symlink()}
    require(set(roots).issubset({"evidence", "install"}) and "evidence" in roots,
            "metrics ZIP has an unexpected top-level member")
    return {"artifactId": artifact["id"], "artifactName": artifact["name"],
            "zipSha256": digest, "extractedRoot": str(roots["evidence"]),
            "installEvidenceRoot": str(roots["install"]) if "install" in roots else None}


def _case_process_witness(record_path: Path, name: str, run_root: str) -> dict | None:
    """Extract only a process generation that is corroborated by its record and journals."""
    record = read_json(record_path)
    tier, requested_name, _decoder, policy, _model_dtype, _vae_dtype = CASES[name]
    require(record.get("caseId") == case_id("cuda", name) and record.get("backend") == "cuda" and
            record.get("request", {}).get("name") == requested_name and
            record.get("request", {}).get("computePolicy") == policy,
            f"case source identity differs for {name}")
    owned = record.get("measured", {}).get("owned")
    process = owned.get("process") if isinstance(owned, dict) else None
    receipt_path = record_path.parent / "profile-process.json"
    journal_path = record_path.parent / "cuda-owned-samples.jsonl"
    faults_path = record_path.parent / "cuda-owned-faults.jsonl"
    if not isinstance(process, dict) or not receipt_path.is_file() or not journal_path.is_file():
        return None
    require(not receipt_path.is_symlink() and not journal_path.is_symlink() and
            (not faults_path.exists() or not faults_path.is_symlink()),
            f"linked process evidence for {name}")
    pid, parent = process.get("pid"), process.get("parentPid")
    created = process.get("createdUtc")
    executable = process.get("executablePath")
    binary_sha = process.get("executableSha256")
    require(type(pid) is int and pid > 0 and type(parent) is int and parent > 0 and
            isinstance(created, str) and bool(created) and isinstance(executable, str) and
            re.fullmatch(r"[0-9a-f]{64}", binary_sha or "") is not None and
            json.loads(receipt_path.read_text(encoding="utf-8-sig")) == {"processId": pid},
            f"owned process generation is incomplete for {name}")
    root = PureWindowsPath(run_root)
    expected_prefix = str(root / "target" / "release" / "deps").casefold().rstrip("\\") + "\\"
    executable_path = str(PureWindowsPath(executable))
    require(PureWindowsPath(executable).is_absolute() and
            executable_path.casefold().startswith(expected_prefix) and
            RELEVANT_NAME.fullmatch(PureWindowsPath(executable).name) is not None,
            f"owned executable is outside the exact run target for {name}")
    journal_hash = file_sha256(journal_path)
    require(owned.get("journalSha256") == journal_hash,
            f"owned sampler journal hash differs for {name}")
    require(owned.get("selectedLuid") == "luid_0x00000000_0x0001f78f",
            f"owned sampler selected device differs for {name}")
    samples = []
    precise_birth = None
    for line in journal_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        require(isinstance(sample, dict) and sample.get("pid") == pid and
                sample.get("luid") == owned.get("selectedLuid") and
                isinstance(sample.get("counter"), dict) and
                sample["counter"].get("parentPid") == parent,
                f"owned sampler row identity differs for {name}")
        sample_birth = sample["counter"].get("createdUtc")
        require(isinstance(sample_birth, str) and bool(sample_birth),
                f"owned sampler precise process birth is missing for {name}")
        sample_birth_utc = utc(sample_birth)
        if precise_birth is None:
            precise_birth = sample_birth_utc
        require(sample_birth_utc == precise_birth,
                f"owned sampler process birth changes across rows for {name}")
        samples.append(sample)
    require(bool(samples) and precise_birth is not None and
            precise_birth.replace(microsecond=(precise_birth.microsecond // 1000) * 1000) == utc(created),
            f"recorded process birth is not the sampler's millisecond truncation for {name}")
    fault_bytes = faults_path.stat().st_size if faults_path.is_file() else None
    return {"caseId": record["caseId"], "name": name, "outcomeStatus": record.get("outcome", {}).get("status"),
            "pid": pid, "parentPid": parent,
            "createdUtc": created, "preciseCreatedUtc": precise_birth.isoformat(),
            "executablePath": executable_path,
            "executableSha256": binary_sha, "journalSha256": journal_hash,
            "sampleCount": len(samples), "faultBytes": fault_bytes,
            "recordSha256": file_sha256(record_path)}


def _raw_sampler_receipt(record_path: Path, name: str) -> dict:
    samples = record_path.parent / "cuda-owned-samples.jsonl"
    faults = record_path.parent / "cuda-owned-faults.jsonl"
    result = {"caseId": case_id("cuda", name), "samplesPresent": samples.is_file(),
              "faultsPresent": faults.is_file()}
    for key, path in (("samples", samples), ("faults", faults)):
        if not path.exists():
            result[key + "Sha256"] = None
            result[key + "Bytes"] = None
            continue
        require(path.is_file() and not path.is_symlink(), f"linked raw sampler evidence for {name}")
        data = path.read_bytes()
        result[key + "Sha256"] = sha256(data).hexdigest()
        result[key + "Bytes"] = len(data)
        if key == "samples":
            rows = [line for line in data.splitlines() if line.strip()]
            for line in rows:
                require(isinstance(json.loads(line), dict), f"invalid raw sampler row for {name}")
            result["sampleRows"] = len(rows)
    return result


def derive_run_binding(metrics_root: Path, run: dict, job: dict, artifact: dict,
                       metrics_zip: Path) -> dict:
    """Bind only authenticated run metadata and contiguous source-case generations."""
    require(run.get("id") == TARGET_RUN_ID and run.get("run_attempt") == TARGET_ATTEMPT and
            run.get("head_sha") == TARGET_CONTROL_SHA and run.get("event") == "workflow_dispatch" and
            run.get("path") == TARGET_WORKFLOW and run.get("status") == "completed" and
            run.get("conclusion") in ("success", "failure", "cancelled", "timed_out") and
            run.get("repository", {}).get("full_name") == TARGET_REPOSITORY,
            "completed App8 run identity/source differs")
    require(job.get("id") == TARGET_JOB_ID and job.get("name") == "cuda" and
            job.get("run_id", TARGET_RUN_ID) == TARGET_RUN_ID and
            job.get("head_sha", TARGET_CONTROL_SHA) == TARGET_CONTROL_SHA and
            job.get("status") == "completed" and job.get("conclusion") in
            ("success", "failure", "cancelled", "timed_out") and
            job.get("runner_name") == TARGET_RUNNER and job.get("runner_id") == TARGET_RUNNER_ID and
            isinstance(job.get("started_at"), str) and isinstance(job.get("completed_at"), str),
            "completed App8 CUDA job/runner binding differs")
    require(artifact.get("id") == TARGET_ARTIFACT_ID and artifact.get("name") == TARGET_ARTIFACT_NAME and
            artifact.get("expired") is False and
            artifact.get("workflow_run", {}).get("id", TARGET_RUN_ID) == TARGET_RUN_ID and
            artifact.get("workflow_run", {}).get("run_attempt", TARGET_ATTEMPT) == TARGET_ATTEMPT and
            artifact.get("workflow_run", {}).get("head_sha") == TARGET_CONTROL_SHA and
            artifact.get("digest") == "sha256:" + file_sha256(metrics_zip),
            "completed App8 metrics artifact binding differs")
    require(metrics_root.is_dir() and not metrics_root.is_symlink(), "metrics root missing or linked")
    source_path = metrics_root / "sources.json"
    require(source_path.is_file() and not source_path.is_symlink(), "source receipt missing or linked")
    sources = read_json(source_path)
    require((sources.get("control_sha"), sources.get("app_sha"), sources.get("engine_sha")) ==
            (TARGET_CONTROL_SHA, TARGET_APP_SHA, TARGET_ENGINE_SHA) and
            sources.get("app_pins") and all(pin == TARGET_ENGINE_SHA for pin in sources["app_pins"]),
            "captured source/pin receipt differs")
    preflight_path = metrics_root / "preflight-initial.json"
    require(preflight_path.is_file() and not preflight_path.is_symlink(),
            "authenticated initial selected-GPU receipt is missing")
    preflight = read_json(preflight_path)
    target_census = json.loads(preflight.get("census", "{}"))
    target_device = target_census.get("validatedDevice", {})
    require(preflight.get("backend") == "cuda" and preflight.get("label") == "initial" and
            preflight.get("runner") == job.get("runner_name") and
            preflight.get("admitted") is True and isinstance(preflight.get("hostname"), str) and
            preflight["hostname"].strip() and target_device.get("physicalMode") == "shared-gpu1" and
            target_device.get("physicalIndex") == 1 and target_device.get("cudaOrdinal") == 0 and
            target_device.get("uuid") == "GPU-e4b79931-7be6-f216-460a-f5405cfafffe" and
            target_device.get("pci", "").lower() == "00000000:c1:00.0" and
            isinstance(target_device.get("luid"), str) and target_device["luid"].startswith("luid_0x"),
            "authenticated initial GPU1 identity is incomplete or unexpected")
    completed_profile = metrics_root / "profile"
    partial_profile = metrics_root / "partial-profile"
    profile = completed_profile if completed_profile.exists() else partial_profile
    if profile.exists():
        require(not profile.is_symlink(), "linked profile directory")
    observed = []
    record_statuses = []
    sampler_receipts = []
    witnesses = []
    stopped = False
    manifest_path = metrics_root / "cases-manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            "run-owned fixed-case manifest is missing or linked")
    case_manifest = read_json(manifest_path)
    manifest_rows = case_manifest.get("cases", [])
    require(case_manifest.get("backend") == "cuda" and isinstance(manifest_rows, list) and
            [row.get("name") for row in manifest_rows] == list(NAMES),
            "run-owned fixed-case manifest IDs/order differ")
    manifest_by_name = {row["name"]: row for row in manifest_rows}
    for name in NAMES:
        manifest_row = manifest_by_name[name]
        require(manifest_row.get("case_id") == case_id("cuda", name) and
                manifest_row.get("source_sha256") == CASE_SOURCE_SHA256[name] and
                re.fullmatch(r"[0-9a-f]{64}", manifest_row.get("run_case_sha256", "")) is not None,
                f"run-owned fixed-case manifest source differs for {name}")
    for name in NAMES:
        record_path = profile / name / "record.json"
        if not record_path.exists():
            stopped = True
            continue
        require(not stopped, "source case records are not a contiguous capture prefix")
        require(record_path.is_file() and not record_path.is_symlink() and
                not record_path.parent.is_symlink(), f"invalid record for {name}")
        case_path = record_path.parent / "case.json"
        require(case_path.is_file() and not case_path.is_symlink() and
                file_sha256(case_path) == manifest_by_name[name]["source_sha256"] and
                read_json(case_path).get("id") == case_id("cuda", name),
                f"recorded fixed source case file/hash differs for {name}")
        record = read_json(record_path)
        require(record.get("caseId") == case_id("cuda", name), f"unexpected source case ID for {name}")
        observed.append(record["caseId"])
        record_statuses.append(record.get("outcome", {}).get("status"))
        sampler_receipts.append(_raw_sampler_receipt(record_path, name))
        witness = _case_process_witness(record_path, name, TARGET_RUN_ROOT)
        if witness is not None:
            witnesses.append(witness)
    if completed_profile.exists() and partial_profile.exists():
        require(not partial_profile.is_symlink(), "linked partial profile directory")
        for name in NAMES:
            completed_record = completed_profile / name / "record.json"
            partial_record = partial_profile / name / "record.json"
            if completed_record.exists() and partial_record.exists():
                require(completed_record.is_file() and partial_record.is_file() and
                        not completed_record.is_symlink() and not partial_record.is_symlink() and
                        file_sha256(completed_record) == file_sha256(partial_record),
                        f"completed and partial case copies differ for {name}")
    expected_ids = [case_id("cuda", name) for name in NAMES]
    all_ids = observed == expected_ids
    if profile.exists():
        actual_records = {str(path.relative_to(profile)).replace("\\", "/") for path in profile.rglob("record.json")}
        expected_records = {f"{name}/record.json" for name in NAMES if (profile / name / "record.json").exists()}
        require(actual_records == expected_records, "metrics contain an unknown or linked source case record")
    require(not all_ids or len(set(observed)) == len(NAMES), "complete capture has duplicate source case IDs")
    unique_binaries = {}
    for witness in witnesses:
        key = witness["executablePath"].casefold()
        previous = unique_binaries.setdefault(key, (witness["executablePath"], witness["executableSha256"]))
        require(previous[1] == witness["executableSha256"],
                "same run-owned executable path has conflicting hashes")
    return {"schema": 1, "target": {"runId": TARGET_RUN_ID, "attempt": TARGET_ATTEMPT,
            "jobId": TARGET_JOB_ID, "controlSha": TARGET_CONTROL_SHA, "appSha": TARGET_APP_SHA,
            "engineSha": TARGET_ENGINE_SHA, "runner": TARGET_RUNNER, "runnerId": TARGET_RUNNER_ID,
            "runRoot": TARGET_RUN_ROOT, "workerId": None, "jobStartedUtc": job["started_at"],
            "jobCompletedUtc": job["completed_at"], "artifactId": artifact["id"],
            "metricsZipSha256": file_sha256(metrics_zip), "hostname": preflight["hostname"],
            "selectedDevice": {key: target_device[key] for key in
                ("physicalIndex", "cudaOrdinal", "uuid", "pci", "luid")}},
            "runConclusion": run["conclusion"], "jobConclusion": job["conclusion"],
            "sourceCaseIds": observed,
            "caseOutcomeStatuses": record_statuses,
            "caseLayout": "profile" if profile == completed_profile else "partial-profile",
            "rawSamplerReceipts": sampler_receipts,
            "allObservedCaseOutcomesCompleted": bool(record_statuses) and
                all(status == "completed" for status in record_statuses),
            "captureRecordSetComplete": all_ids,
            "captureAcceptanceEvaluated": False,
            "caseProcessWitnesses": witnesses,
            "binaryHashTargets": [{"path": path, "sha256": digest}
                                  for path, digest in sorted(unique_binaries.values())],
            "releaseScope": ("known recorded case generations and run-root markers at observation timestamps"
                if witnesses else "run-root markers only at observation timestamps; no case generations were recorded"),
            "wholeDescendantAncestryProven": False, "historicalIntervalProven": False}


def rehash_recorded_binaries(binding: dict, hash_path) -> list[dict]:
    """Read each distinct exact run-owned executable and compare its recorded SHA-256."""
    expected_root = PureWindowsPath(TARGET_RUN_ROOT) / "target" / "release" / "deps"
    verified = {}
    for item in binding.get("binaryHashTargets", []):
        path = PureWindowsPath(item.get("path", ""))
        prefix = str(expected_root).casefold().rstrip("\\") + "\\"
        require(path.is_absolute() and str(path).casefold().startswith(prefix),
                "binary hash target is outside the exact run-owned release/deps tree")
        digest = item.get("sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "binary hash target has no recorded SHA-256")
        key = str(path).casefold()
        if key in verified:
            require(verified[key] == digest, "duplicate binary target has conflicting recorded hashes")
            continue
        actual = hash_path(str(path))
        require(actual == digest, "run-owned Rust worker executable bytes changed")
        verified[key] = digest
    return [{"path": path, "sha256": digest} for path, digest in sorted(verified.items())]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected JSON object in {path.name}")
    return value


def select_target_receipts(run: dict, jobs: dict, artifacts: dict) -> tuple[dict, dict]:
    job_rows = jobs.get("jobs")
    artifact_rows = artifacts.get("artifacts")
    require(isinstance(job_rows, list) and jobs.get("total_count") == len(job_rows) and
            isinstance(artifact_rows, list) and artifacts.get("total_count") == len(artifact_rows),
            "GitHub target job/artifact inventory incomplete")
    selected_jobs = [item for item in job_rows if isinstance(item, dict) and item.get("id") == TARGET_JOB_ID]
    selected_artifacts = [item for item in artifact_rows if isinstance(item, dict) and
                          item.get("id") == TARGET_ARTIFACT_ID]
    require(len(selected_jobs) == len(selected_artifacts) == 1,
            "authenticated target job or metrics artifact is missing/ambiguous")
    return selected_jobs[0], selected_artifacts[0]


class _StripAuthorizationRedirect(HTTPRedirectHandler):
    """Never forward the GitHub token to a signed external artifact host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_headers = {key: value for key, value in req.headers.items()
                        if key.casefold() not in {"authorization", "proxy-authorization"}}
        return Request(newurl, headers=safe_headers, method="GET")


def select_release_runner(jobs: dict, current_run: dict, github_job: str) -> dict:
    rows = jobs.get("jobs")
    require(isinstance(rows, list) and jobs.get("total_count") == len(rows),
            "current release-check job inventory incomplete")
    matches = [item for item in rows if isinstance(item, dict) and item.get("name") == github_job]
    require(len(matches) == 1 and github_job == "cuda_release_check" and
            matches[0].get("status") in ("in_progress", "completed") and
            matches[0].get("runner_name") in ALLOWED_RELEASE_RUNNERS and
            type(matches[0].get("runner_id")) is int and matches[0]["runner_id"] > 0 and
            ((matches[0]["runner_name"] == TARGET_RUNNER and
              matches[0]["runner_id"] == TARGET_RUNNER_ID) or
             (matches[0]["runner_name"] == "cuda-windows" and
              matches[0]["runner_id"] != TARGET_RUNNER_ID)) and
            current_run.get("id") == int(os.environ.get("GITHUB_RUN_ID", "0")) and
            current_run.get("head_sha") == os.environ.get("GITHUB_SHA"),
            "release-check job is not running on the authenticated target runner/source")
    return matches[0]


def verify_same_selected_device(target_device: dict, current_census: dict) -> dict:
    current = current_census.get("validatedDevice", {})
    keys = ("physicalIndex", "cudaOrdinal", "uuid", "pci", "luid")
    require(current_census.get("physicalMode") == "shared-gpu1" and
            all(current.get(key) == target_device.get(key) for key in keys),
            "release-check selected GPU1 differs from the authenticated capture device")
    return {key: current[key] for key in keys}


def fetch_authenticated_target(metrics_destination: Path) -> tuple[dict, dict, dict, dict, Path]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY")
    require(bool(token) and repository == TARGET_REPOSITORY and
            urlparse(api_url).scheme == "https" and bool(urlparse(api_url).netloc),
            "authenticated GitHub API context unavailable")
    opener = build_opener(_StripAuthorizationRedirect())

    def get_json(url: str) -> dict:
        require(urlparse(url).scheme == "https" and urlparse(url).netloc == urlparse(api_url).netloc,
                "GitHub API endpoint host changed")
        request = Request(url, headers={"Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SceneWorks-YuE2-release-probe"})
        with opener.open(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
        require(isinstance(value, dict), "GitHub API returned a non-object target receipt")
        return value

    prefix = f"{api_url}/repos/{TARGET_REPOSITORY}/actions"
    run = get_json(f"{prefix}/runs/{TARGET_RUN_ID}")
    jobs = get_json(f"{prefix}/runs/{TARGET_RUN_ID}/jobs?per_page=100")
    artifacts = get_json(f"{prefix}/runs/{TARGET_RUN_ID}/artifacts?per_page=100")
    job, artifact = select_target_receipts(run, jobs, artifacts)
    current_run_id = os.environ.get("GITHUB_RUN_ID", "")
    require(current_run_id.isdigit(), "current release-check run ID missing")
    current_run = get_json(f"{prefix}/runs/{current_run_id}")
    current_jobs = get_json(f"{prefix}/runs/{current_run_id}/jobs?per_page=100")
    release_job = select_release_runner(current_jobs, current_run, os.environ.get("GITHUB_JOB", ""))
    url = f"{api_url}/repos/{TARGET_REPOSITORY}/actions/artifacts/{TARGET_ARTIFACT_ID}/zip"
    zip_path = metrics_destination.parent / "target-metrics.zip"
    request = Request(url, headers={"Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json", "User-Agent": "SceneWorks-YuE2-release-probe"})
    with opener.open(request, timeout=60) as response:
        zip_path.write_bytes(response.read())
    return run, job, artifact, release_job, zip_path


def collect(evidence: Path, app: Path, engine: Path, control: Path) -> dict:
    require(os.name == "nt" and os.environ.get("GITHUB_REPOSITORY") == "SceneWorks/inference" and
            os.environ.get("GITHUB_JOB") == "cuda_release_check" and
            os.environ.get("GITHUB_RUN_ATTEMPT") == "1" and
            os.environ.get("RUNNER_NAME") in ALLOWED_RELEASE_RUNNERS and
            os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and
            not os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID"),
            "release probe dispatch/runner/device identity mismatch")
    require(os.environ.get("EXPECTED_APP_SHA") == TARGET_APP_SHA and
            os.environ.get("EXPECTED_ENGINE_SHA") == TARGET_ENGINE_SHA,
            "release probe targets the wrong app or engine")
    sources = verify_sources(app, engine, control, TARGET_APP_SHA, TARGET_ENGINE_SHA,
                             os.environ["EXPECTED_CONTROL_SHA"])
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    metrics_destination = evidence / "original-metrics"
    run, job, artifact, release_job, metrics_zip = fetch_authenticated_target(metrics_destination)
    (evidence / "github-target.json").write_text(json.dumps({
        "run": {key: run.get(key) for key in ("id", "run_attempt", "head_sha", "event", "path", "status", "conclusion")},
        "job": {key: job.get(key) for key in ("id", "run_id", "run_attempt", "name", "status", "conclusion",
                                                "runner_name", "runner_id", "started_at", "completed_at")},
        "releaseCheckJob": {key: release_job.get(key) for key in
            ("id", "run_id", "name", "status", "runner_name", "runner_id")},
        "artifact": {key: artifact.get(key) for key in ("id", "name", "digest", "expired")}},
        indent=2) + "\n", encoding="utf-8")
    artifact_proof = safe_extract_metrics(metrics_zip, artifact, metrics_destination)
    metrics_root = Path(artifact_proof["extractedRoot"])
    binding = derive_run_binding(metrics_root, run, job, artifact, metrics_zip)
    target = binding["target"]
    require(release_job.get("runner_name") == os.environ.get("RUNNER_NAME") and
            os.environ.get("COMPUTERNAME", "").upper() == target["hostname"].upper(),
            "release-check runner is not on the authenticated capture host")
    binary_proof = rehash_recorded_binaries(binding,
        lambda path: file_sha256(Path(path)))
    binding["binaryRehashes"] = binary_proof
    binding["artifactProof"] = artifact_proof
    (evidence / "target.json").write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
    (evidence / "case-binding.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    raw, busy = cuda_physical_census(admission=False)
    (evidence / "physical-preflight.json").write_text(raw + "\n", encoding="utf-8")
    physical_files = retain_cuda_physical_evidence(evidence, "initial", raw) if not busy else None
    require(not busy, "selected GPU1 physical observation failed: " + "; ".join(busy))
    current_census = json.loads(raw)
    expected_device = target["selectedDevice"]
    device_proof = verify_same_selected_device(expected_device, current_census)
    script = Path(__file__).with_name("yue2_app_install_release_processes.ps1")
    environment = dict(os.environ, YUE2_RELEASE_OLD_ROOT=target["runRoot"],
                       YUE2_RELEASE_WORKER_ID=target.get("workerId") or "")
    collector = PowerShellSnapshotCollector(evidence, script, environment)
    verdict = validate_release_pairs(evidence, collector, binding)
    return {"schema": 1, "purpose": "read-only release observation only",
            "oldRun": target, "targetSha256": file_sha256(evidence / "target.json"),
            "caseBindingSha256": file_sha256(evidence / "case-binding.json"),
            "artifactProof": artifact_proof,
            "captureRecordsComplete": binding["captureRecordsComplete"],
            "captureRecordSetComplete": binding["captureRecordSetComplete"],
            "captureAcceptanceEvaluated": binding["captureAcceptanceEvaluated"],
            "releaseScope": binding["releaseScope"],
            "binaryRehashes": binary_proof,
            "selectedDeviceMatchesCapture": device_proof,
            "releaseRunner": {"name": release_job["runner_name"], "id": release_job["runner_id"],
                "sameRunnerInstanceAsCapture": release_job["runner_name"] == target["runner"] and
                    release_job["runner_id"] == target["runnerId"],
                "sameHostEvidence": "same recorded COMPUTERNAME, exact target executable path/hash, and selected GPU UUID/PCI/LUID"},
            "probe": {"runId": os.environ.get("GITHUB_RUN_ID"),
                      "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                      "job": os.environ.get("GITHUB_JOB"), "runner": os.environ.get("RUNNER_NAME"),
                      "hostname": os.environ.get("COMPUTERNAME"),
                      "controlSha": os.environ.get("GITHUB_SHA")},
            "physicalFiles": physical_files, "processFiles": verdict["processFiles"],
            "processRefusalFiles": verdict["refusalFiles"],
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
            snapshots = sorted(args.evidence.glob("process-snapshot*.json"))
            refusal_files = sorted(args.evidence.glob("*refusal.json"))
            (args.evidence / "refusal.json").write_text(json.dumps({"releasedNow": False,
                "reason": str(error),
                "processFiles": [{"name": path.name, "bytes": path.stat().st_size,
                                  "sha256": file_sha256(path)} for path in snapshots],
                "priorRefusals": [{"name": path.name, "bytes": path.stat().st_size,
                                   "sha256": file_sha256(path)} for path in refusal_files]},
                indent=2) + "\n", encoding="utf-8")
        print("yue2-app-install-release-probe: refused; inspect sanitized artifact", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

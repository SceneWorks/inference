"""Opt-in, provisional GPU0 proof while one exact old-group GPU1 holder runs.

Not a physical lease: polling bounds detection only. Root must exclude out-of-group
jobs and independently authenticate server job chronology after BOTH jobs finish.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
import re
import signal

ENGINE = "825341ff8d0110ea448213485891b39d57806fa4"
# The diagnostic archives this exact M4 checkout and applies separately retained observation
# patches. The derivative patch/tree hashes belong to the reviewed control source, not this SHA.
PEDANTIC_ENGINE = ENGINE
RECEIPT = "37135502627"
RECEIPT_DIGEST = "1f0307e1056fa00b3a177002aa1cfafb40348c071676a7c894e938ebf2fd3991"
REPO = "SceneWorks/inference"
OLD_GROUP = "inference-real-weights-physical-host"
GPU0_GROUP = "inference-yue2-owner-gpu0"
HOLDER_RUN = 37125806675
HOLDER_JOB = 111215090511
HOLDER_SHA = "0826a16cf7f2144b86d88c01b95fffb4e447884d"
HOLDER_NAME = "Decode-speedups benchmark campaign (Candle/CUDA)"
HOLDER_RUNNER = "cuda-windows-2"
IS_WINDOWS = os.name == "nt"
API_TIMEOUT = 3
# Cold Windows PowerShell startup + CIM/AuthCode + one 1s counter sample.
# This is a bounded harness query, not a waiver of identity/activity evidence.
PHYSICAL_QUERY_TIMEOUT = 15
POLL_SECONDS = 10
METADATA_SECONDS = 60
# Full metadata: four HTTP calls, one pmon, at most two Windows queries
# (app lineage and signed background), plus bounded processing overhead.
CYCLE_LIMIT_SECONDS = 5 * API_TIMEOUT + 2 * PHYSICAL_QUERY_TIMEOUT + 10
SOURCE_HASHES = {
    ".github/workflows/real-weights.yml": "6d586177edd06a208d0bc72ed86e16eb207e2bf82b2c259ff7217a1d8a21f757",
    "scripts/ci/real-weights/candle-decode-speedups-bench/run-every-matrix-row.cmd": "157eb7ad92dcb1330e9dcbf8194ead42a262baf2723a3ab0b0f21a15b8a64756",
    "scripts/release/speculative_bench_campaign.py": "97b188879d5d8c5049b6490d146d6e6e670c278beb483fc4ab357019442c9b01",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("GitHub JSON endpoint redirected; refuse credential forwarding")


def api(path: str) -> dict:
    require(path.startswith(("actions/", "contents/")) and ".." not in path,
            "unexpected authenticated API path")
    token = os.environ.get("GH_TOKEN", "")
    require(bool(token), "owner guard requires read-only GitHub token")
    request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/{path}", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10", "Cache-Control": "no-cache",
    })
    # urllib uses binary network IO; keep the scoped opener out of the Path.open text API.
    open_response = urllib.request.build_opener(NoRedirect()).open
    with open_response(request, timeout=API_TIMEOUT) as response:
        date = parsedate_to_datetime(response.headers["Date"])
        require(date.tzinfo is not None and abs(time.time() - date.timestamp()) <= 15,
                "GitHub response timestamp missing/stale")
        raw = response.read(2_000_001)
        require(len(raw) <= 2_000_000, "GitHub response exceeds bounded evidence size")
        value = json.loads(raw.decode("utf-8"))
        require(isinstance(value, dict), "GitHub response is not an object")
        return {"path": path, "server_date": response.headers["Date"],
                "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                "rate_limit_reset": response.headers.get("X-RateLimit-Reset"),
                "received_utc_ns": time.time_ns(), "body": value}


def run_identity(value: dict, run_id: int, sha: str, workflow: str) -> None:
    require(value.get("id") == run_id and value.get("head_sha") == sha and
            value.get("run_attempt") == 1 and value.get("event") == "workflow_dispatch" and
            value.get("path") == workflow and value.get("repository", {}).get("full_name") == REPO and
            value.get("status") == "in_progress" and value.get("conclusion") is None,
            "run identity/attempt/source/active status changed")


def selected_job(payload: dict, job_id: int | None, name: str, runner: str) -> dict:
    jobs = payload.get("jobs", [])
    require(payload.get("total_count") == len(jobs) and len(jobs) > 0,
            "incomplete job inventory")
    active = [job for job in jobs if job.get("conclusion") != "skipped"]
    require(len(active) == 1, "another selected job exists in the guarded run")
    job = active[0]
    require((job_id is None or job.get("id") == job_id) and job.get("name") == name and
            job.get("runner_name") == runner and job.get("status") == "in_progress" and
            job.get("conclusion") is None and job.get("completed_at") is None and
            isinstance(job.get("started_at"), str) and
            all(j.get("status") == "completed" for j in jobs if j is not job),
            "selected job identity/runner/active status changed")
    return job


def active_group(payload: dict, group: str, run_id: int) -> None:
    members = payload.get("group_members", [])
    require(payload.get("group_name") == group and payload.get("total_count") == len(members),
            "incomplete/wrong concurrency group")
    active = [m for m in members if m.get("status") == "in_progress"]
    require(len(active) == 1 and active[0].get("run_id") == run_id and
            active[0].get("job_id") is None and
            all(m.get("status") in {"pending", "in_progress", "queued"} for m in members),
            "exact workflow is no longer sole active group holder")


def owned_descendants(root_pid: int, record=None) -> set[int]:
    # Parent-chain ownership only; no executable-name whitelist or foreign signals.
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                             "Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,@{Name='CreatedUtc';Expression={if ($_.CreationDate) {$_.CreationDate.ToUniversalTime().ToString('o')} else {$null}}} | ConvertTo-Json -Compress"],
                            capture_output=True, text=True, encoding="utf-8", timeout=PHYSICAL_QUERY_TIMEOUT)
    require(result.returncode == 0, "owned descendant census failed")
    rows = json.loads(result.stdout)
    require(isinstance(rows, list), "incomplete descendant census")
    parents = {}
    created = {}
    for row in rows:
        pid, parent = row.get("ProcessId"), row.get("ParentProcessId")
        require(type(pid) is int and type(parent) is int and pid >= 0 and parent >= 0 and pid not in parents,
                "invalid descendant process identity")
        text = row.get("CreatedUtc")
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00")) if isinstance(text, str) else None
        require(stamp is None or stamp.tzinfo is not None, "process creation time missing zone")
        parents[pid] = parent
        created[pid] = stamp
    require(root_pid in parents and created[root_pid] is not None,
            "owned Popen root creation identity absent during descendant census")
    owned = {root_pid}
    for _ in range(len(parents)):
        require(all(created[pid] is not None for pid, parent in parents.items() if parent in owned),
                "possible owned descendant lacks creation identity")
        added = {pid for pid, parent in parents.items() if parent in owned and created[pid] >= created[parent]}
        if added <= owned:
            if record is not None:
                record({"event": "owned_lineage", "root_pid": root_pid,
                        "rows": [row for row in rows if row["ProcessId"] in owned]})
            return owned
        owned |= added
    raise RuntimeError("ambiguous descendant chain")


def reviewed_background() -> dict:
    from yue2_cuda_idle_context import reviewed_baseline, read_json
    baseline, directory = reviewed_baseline()  # Original digest/source/device/12h checks.
    value = {**baseline, "signature": read_json(directory, "process-before")["signature"]}
    require(value["signature"].get("status") == "Valid", "reviewed background signature is invalid")
    return value


def validate_background(value: dict, baseline: dict) -> None:
    process = value.get("process", {})
    identity = (process.get("pid"), process.get("name"), process.get("executablePath"), process.get("creationDate"))
    require(identity == tuple(baseline["identity"]) and process.get("signature") == baseline["signature"],
            "trusted background process generation/image/signature changed")
    row = value.get("counter", {})
    require(row.get("counter") == r"\GPU Engine(*)\Utilization Percentage" and "error" not in row,
            "background GPU Engine counter is unavailable")
    prefix = f"pid_{baseline['identity'][0]}_{baseline['luid']}_"
    samples = [item for item in row.get("samples", [])
               if isinstance(item, dict) and item.get("instance", "").lower().startswith(prefix)]
    counters = {}
    for item in samples:
        name, number = item["instance"].lower(), item.get("cookedValue")
        require(name not in counters and str(item.get("status")) == "0" and
                type(number) in (int, float) and math.isfinite(number) and number == 0,
                "trusted background GPU engine active/invalid/duplicate")
        counters[name] = number
    require(counters and set(counters) == set(baseline["counters"]["engine"]),
            "trusted background GPU0 engine instance set changed/incomplete")


def sample_background(baseline: dict, record=None) -> None:
    # Same identity/signature and GPU Engine schema as the reviewed collector,
    # restricted to the trusted background PID. Own model memory is not compared.
    script = r"""
$ErrorActionPreference = 'Stop'
$item = Get-CimInstance Win32_Process -Filter 'ProcessId = 38212' -ErrorAction Stop
if ($null -eq $item -or -not $item.ExecutablePath) { throw 'background identity unavailable' }
$sig = Get-AuthenticodeSignature -FilePath $item.ExecutablePath -ErrorAction Stop
$set = Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
$samples = @($set.CounterSamples | Where-Object {$_.InstanceName -match '(^|_)pid_38212(_|$)'} | ForEach-Object {
    @{path=$_.Path; instance=$_.InstanceName; cookedValue=$_.CookedValue; status=[string]$_.Status}
})
@{process=@{pid=38212; name=$item.Name; executablePath=$item.ExecutablePath; creationDate=[string]$item.CreationDate;
            signature=@{status=[string]$sig.Status; signerSubject=$sig.SignerCertificate.Subject; signerThumbprint=$sig.SignerCertificate.Thumbprint}};
  counter=@{counter='\GPU Engine(*)\Utilization Percentage'; timestamp=[string]$set.Timestamp; samples=$samples}} | ConvertTo-Json -Depth 8 -Compress
"""
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", timeout=PHYSICAL_QUERY_TIMEOUT)
    if record is not None:
        record({"event": "background_activity", "returncode": result.returncode,
                "raw": result.stdout, "stderr": result.stderr})
    require(result.returncode == 0, "bounded background identity/engine probe failed")
    value = json.loads(result.stdout)
    require(isinstance(value, dict), "background probe is not an object")
    validate_background(value, baseline)


def gpu0_actors(child_pid: int | None, descendants: bool = False, record=None, background=None) -> str:
    result = subprocess.run(["nvidia-smi", "pmon", "-i", "0", "-c", "1", "-s", "um"],
                            capture_output=True, text=True, encoding="utf-8", timeout=API_TIMEOUT)
    require(result.returncode == 0, "live GPU0 typed census unavailable")
    owned = owned_descendants(child_pid, record) if descendants and child_pid is not None else {child_pid}
    columns = {}
    seen = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if line.startswith("#"):
            if "pid" in fields and "type" in fields:
                columns = {name: i for i, name in enumerate(fields[1:])}
            continue
        require(all(k in columns for k in ("gpu", "pid", "type")) and
                len(fields) > max(columns.values()) and fields[columns["gpu"]] == "0",
                "ambiguous GPU0 typed row")
        pid, kind = fields[columns["pid"]], fields[columns["type"]]
        if pid == kind == "-":
            require(all(v == "-" or i == columns["gpu"] for i, v in enumerate(fields)),
                    "ambiguous empty GPU0 row")
            continue
        require(pid.isdigit() and int(pid) not in seen, "ambiguous/duplicate GPU0 PID")
        pid = int(pid)
        seen.add(pid)
        if pid == 38212 and kind == "C+G":
            require(background is not None, "background PID has no authenticated receipt identity")
            for metric in ("sm", "mem", "enc", "dec", "jpg", "ofa"):
                require(metric in columns and len(fields) > columns[metric], "baseline pmon utilization is incomplete")
                value = fields[columns[metric]]
                require(value == "-" or (re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is not None and float(value) == 0),
                        "trusted background pmon utilization is active/invalid")
            # '-' never means zero; require the receipt-selected exact engine set
            # and process generation even when pmon happens to support zeros.
            sample_background(background, record)
        require((pid == 38212 and kind == "C+G") or
                (child_pid is not None and pid in owned and kind in {"C", "C+G"}),
                "unexpected GPU0 actor; owner proof must stop")
    require(bool(columns), "GPU0 census lacks typed columns")
    require(background is None or background["identity"][0] in seen,
            "authenticated background process disappeared from typed GPU0 census")
    return result.stdout


class OwnerGuard:
    def __init__(self, evidence: Path, engine_sha: str, control_sha: str, kind: str = "engine"):
        self.kind = kind
        require(kind in {"engine", "app", "diagnostic"}, "unknown guarded workflow")
        self.path = evidence / "gpu0-holder-chronology.jsonl"
        self.engine_sha = engine_sha
        self.control_sha = control_sha
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.fault = None
        self.thread = None
        self.proof_job = None
        self.cycle_started = None
        self.descendants = False
        self.signals = {}
        self.metadata_checked = None
        self.holder_started = None
        self.background = None

    def record(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"observed_utc_ns": time.time_ns(), **event}, sort_keys=True) + "\n")

    def holder(self, child=None) -> None:
        start = time.monotonic()
        self.cycle_started = start
        # One job heartbeat per 10s; full metadata at most once per 60s.
        # <=360+180 requests/hour steady, leaving budget for per-command setup.
        if self.metadata_checked is None or start - self.metadata_checked >= METADATA_SECONDS:
            reads = [api(f"actions/runs/{HOLDER_RUN}"),
                     api(f"actions/runs/{HOLDER_RUN}/attempts/1/jobs?per_page=100"),
                     api(f"actions/concurrency_groups/{OLD_GROUP}")]
            self.record({"event": "holder_readback", "reads": reads})
            run_identity(reads[0]["body"], HOLDER_RUN, HOLDER_SHA, ".github/workflows/real-weights.yml")
            selected = selected_job(reads[1]["body"], HOLDER_JOB, HOLDER_NAME, HOLDER_RUNNER)
            require(reads[1]["body"].get("total_count") == 55, "foreign job selection changed")
            active_group(reads[2]["body"], OLD_GROUP, HOLDER_RUN)
            if self.holder_started is not None:
                require(selected["started_at"] == self.holder_started, "foreign holder start identity changed")
            self.holder_started = selected["started_at"]
            self.metadata_checked = start
        heartbeat = api(f"actions/jobs/{HOLDER_JOB}")
        self.record({"event": "holder_job_heartbeat", "read": heartbeat})
        job = heartbeat["body"]
        selected_job({"total_count": 1, "jobs": [job]}, HOLDER_JOB, HOLDER_NAME, HOLDER_RUNNER)
        require(job.get("run_id") == HOLDER_RUN and job.get("run_attempt") == 1 and
                job.get("head_sha") == HOLDER_SHA and job.get("workflow_name") == "Real-weight validation" and
                job.get("started_at") == self.holder_started,
                "heartbeat run/attempt/head/workflow/start identity changed")
        child_pid = child.pid if child is not None and child.poll() is None else None
        raw = gpu0_actors(child_pid, descendants=self.descendants, record=self.record, background=self.background)
        self.record({"event": "gpu0_actors", "owned_pid": child_pid, "raw": raw})
        require(time.monotonic() - start <= CYCLE_LIMIT_SECONDS, "holder observation cycle stale")

    def preflight(self) -> None:
        self.record({"event": "preflight_started", "kind": self.kind,
                     "engine_sha": self.engine_sha, "control_sha": self.control_sha,
                     "holder_run_id": HOLDER_RUN, "holder_job_id": HOLDER_JOB})
        try:
            self._preflight()
        except BaseException as error:
            self.record({"event": "preflight_refusal", "error": str(error)})
            raise

    def _preflight(self) -> None:
        from yue2_cuda_idle_context import BASELINE_DIGEST, RUN_ID
        expected_engine = PEDANTIC_ENGINE if self.kind == "diagnostic" else ENGINE
        require(bool(expected_engine) and self.engine_sha == expected_engine and
                RUN_ID == RECEIPT and BASELINE_DIGEST == RECEIPT_DIGEST and
                os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID") == RECEIPT and
                os.environ.get("GITHUB_REPOSITORY") == REPO and os.environ.get("GITHUB_JOB") == "cuda" and
                os.environ.get("GITHUB_RUN_ATTEMPT") == "1" and os.environ.get("GITHUB_SHA") == self.control_sha and
                os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and
                os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and
                re.fullmatch(r"[0-9a-f]{40}", self.control_sha) is not None and IS_WINDOWS,
                "GPU0 mode is only the exact reviewed CUDA owner-receipt attempt")
        self.background = reviewed_background()
        self.record({"event": "reviewed_background", "identity": self.background["identity"],
                     "luid": self.background["luid"], "engine_instances": sorted(self.background["counters"]["engine"])})
        control = Path(__file__).resolve().parents[2]
        workspace = Path(os.environ["GITHUB_WORKSPACE"])
        for directory, expected in ((control, self.control_sha), (workspace / "engine", self.engine_sha)):
            head = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, encoding="utf-8", check=True, timeout=3).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=normal"],
                                   capture_output=True, text=True, encoding="utf-8", check=True, timeout=3).stdout
            require(head == expected and not dirty.strip(), "guarded source checkout differs/is dirty")
        if self.kind == "app":
            from yue2_app_precision_profile import verify_sources
            verify_sources(workspace / "app", workspace / "engine", control,
                           os.environ["EXPECTED_APP_SHA"], self.engine_sha, self.control_sha)
        own_id = int(os.environ["GITHUB_RUN_ID"])
        own = api(f"actions/runs/{own_id}")
        jobs = api(f"actions/runs/{own_id}/attempts/1/jobs?per_page=100")
        group = api(f"actions/concurrency_groups/{GPU0_GROUP}")
        self.record({"event": "own_source", "reads": [own, jobs, group]})
        workflow = {"engine": "yue2-precision-proof.yml", "app": "yue2-app-precision-profile.yml",
                    "diagnostic": "yue2-bf16-tile-diagnostic.yml"}[self.kind]
        run_identity(own["body"], own_id, self.control_sha, f".github/workflows/{workflow}")
        self.proof_job = selected_job(jobs["body"], None, "cuda", os.environ["RUNNER_NAME"])
        active_group(group["body"], GPU0_GROUP, own_id)
        for path, digest in SOURCE_HASHES.items():
            source = api(f"contents/{path}?ref={HOLDER_SHA}")
            self.record({"event": "holder_source", "read": source})
            body = source["body"]
            require(body.get("path") == path and body.get("encoding") == "base64" and
                    hashlib.sha256(base64.b64decode(body.get("content", ""))).hexdigest() == digest,
                    "foreign GPU1 workflow/driver source is not the reviewed exact bytes")
        self.holder(None)  # Last action before Popen; retains the existing full fresh29 preflight.

    def arm(self) -> None:
        # Flag cancellation rather than raising in the Popen constructor: after
        # it returns, only our single waiter owns the newly assigned child tree.
        def interrupted(signum, frame):
            self.fault = f"owned GPU0 controller interrupted by signal {signum}"
            self.failed.set()
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is not None and number not in self.signals:
                self.signals[number] = signal.getsignal(number)
                signal.signal(number, interrupted)

    def start(self, child) -> None:
        self.arm()
        self.descendants = self.kind == "app"
        def monitor() -> None:
            while not self.stop.is_set() and not self.failed.is_set() and child.poll() is None:
                try:
                    self.holder(child)
                except Exception as error:
                    self.fault = str(error)
                    self.failed.set()
                    self.record({"event": "refusal", "error": str(error)})
                    return
                self.stop.wait(POLL_SECONDS)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        for number, handler in self.signals.items():
            signal.signal(number, handler)
        self.signals.clear()
        if self.thread is not None:
            self.thread.join(timeout=CYCLE_LIMIT_SECONDS + 2)
            require(not self.thread.is_alive(), "holder watchdog did not release")
        require(not self.failed.is_set(), f"holder watchdog refused: {self.fault}")
        self.holder(None)

    def summary(self) -> dict:
        return {"mode": "owner-gpu0", "acceptance": "provisional-holder-chronology",
                "holder_run_id": HOLDER_RUN, "holder_job_id": HOLDER_JOB, "holder_attempt": 1,
                "holder_sha": HOLDER_SHA, "proof_job": self.proof_job, "fault": self.fault,
                "poll_seconds": POLL_SECONDS, "api_timeout_seconds": API_TIMEOUT,
                "physical_query_timeout_seconds": PHYSICAL_QUERY_TIMEOUT,
                "cycle_limit_seconds": CYCLE_LIMIT_SECONDS,
                "metadata_interval_seconds": METADATA_SECONDS,
                "maximum_group_detection_seconds": METADATA_SECONDS + POLL_SECONDS + CYCLE_LIMIT_SECONDS,
                "steady_requests_per_hour_upper_bound": 360 + 180,
                "maximum_detection_seconds": POLL_SECONDS + CYCLE_LIMIT_SECONDS,
                "chronology_file": self.path.name,
                "final_acceptance_requires": "Independent authenticated same-attempt holder job completed_at strictly after proof job completed_at; other54 jobs skipped. Out-of-group freeze/monitor owned by root. Polling is not a physical lease."}


def reap_tree(child) -> tuple[int | None, str | None]:
    """Single waiter owns cleanup: taskkill only the Popen root's Windows tree."""
    try:
        if child.poll() is None:
            result = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                    capture_output=True, text=True, encoding="utf-8", timeout=15)
            require(result.returncode == 0 or child.poll() is not None,
                    "owned Windows tree termination failed")
        code = child.poll()
        return (code if code is not None else child.wait(timeout=30)), None
    except BaseException as error:
        return child.poll(), str(error)


def wait(child, guard: OwnerGuard, timeout: float) -> tuple[int | None, bool, str | None]:
    deadline = time.monotonic() + timeout
    try:
        while True:
            if guard.cycle_started is not None and time.monotonic() - guard.cycle_started > CYCLE_LIMIT_SECONDS + POLL_SECONDS:
                guard.fault = "holder watchdog exceeded bounded observation deadline"
                guard.failed.set()
            if guard.failed.is_set():
                code, cleanup = reap_tree(child)
                return code, False, f"holder watchdog: {guard.fault}; cleanup: {cleanup}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                code, cleanup = reap_tree(child)
                return code, True, cleanup
            try:
                return child.wait(timeout=min(1, remaining)), False, None
            except subprocess.TimeoutExpired:
                pass
    except BaseException as error:
        code, cleanup = reap_tree(child)
        return code, False, f"owned guard waiter: {error}; cleanup: {cleanup}"


def guarded_command(argv: list[str], cwd: Path, env: dict, log, evidence: Path, label: str) -> int:
    """Wrap only an explicitly opted-in app-owned tree; preserve command bytes/cases."""
    guard = OwnerGuard(evidence, os.environ["EXPECTED_ENGINE_SHA"], os.environ["EXPECTED_CONTROL_SHA"], "app")
    guard.preflight()
    stamp = os.environ.get("YUE2_APP_PRECISION_JOB_STARTED_UTC_NS", "")
    require(stamp.isdigit() and int(stamp) <= time.time_ns(), "app owned job start is unavailable")
    remaining = (int(stamp) + 480 * 60 * 1_000_000_000 - time.time_ns()) / 1_000_000_000 - 600
    require(remaining > 0, "app job no longer has its original 600-second owned cleanup/upload tail")
    child_env = {k: v for k, v in env.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    guard.arm()
    require(not guard.failed.is_set(), "owner canceled before app command creation")
    child = subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    error = None
    try:
        guard.start(child)
        code, timed_out, error = wait(child, guard, min(480 * 60, remaining))
    except BaseException as fault:
        code, cleanup = reap_tree(child)
        timed_out = False
        error = f"watchdog startup: {fault}; cleanup: {cleanup}"
    finally:
        try:
            guard.finish()
        except BaseException as fault:
            error = f"{error}; final holder: {fault}"
        guard.record({"event": "owned_command_result", "label": label, "owned_pid": child.pid,
                      "released": child.poll() is not None, "exit_code": code,
                      "timed_out": timed_out, "error": error, "summary": guard.summary()})
    require(not timed_out and error is None and child.poll() is not None,
            f"guarded app command refused: {error}; timed_out={timed_out}")
    return code


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(args.argv and args.argv[0] == "--" and len(args.argv) > 1, "guard requires explicit owned command")
    args.evidence.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env.get("YUE2_CUDA_SCHEDULING_MODE", "shared-host") == "owner-gpu0":
        code = guarded_command(args.argv[1:], Path.cwd(), env, None, args.evidence, args.label)
    else:
        require(env.get("YUE2_CUDA_SCHEDULING_MODE", "shared-host") == "shared-host", "unknown scheduling mode")
        code = subprocess.run(args.argv[1:], env={k: v for k, v in env.items()
                              if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}, check=False).returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""External, fail-closed all-listener watch for one shared-host YuE2 CUDA run.

This does not grant a physical lease. The in-job selected GPU1 census and owned process
cleanup remain mandatory. The shared-gpu1 route observes concurrent foreign work and
revokes only on owned identity or inventory failure; strict shared-host retains exclusivity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
import yue2_reviewed_gpu1 as gpu1

REPOS = ("SceneWorks/inference", "SceneWorks/SceneWorks")
RUNNERS = {"cuda-windows": 2313, "cuda-windows-2": 2619,
           "cuda-windows-3": 23, "cuda-windows-4": 24}
LEGACY_ZERO_JOB = {
    31120232778: ("74978f67ac33cade17a863ef109138d8d162b51c", 2,
                  "2026-08-06T16:33:14Z", "2026-08-06T20:06:14Z"),
    31116344133: ("ce5f4f068c2ccd6063c85cfc212c24ae67d873e8", 4,
                  "2026-08-06T15:32:49Z", "2026-08-06T18:41:10Z"),
}
STATUSES = ("in_progress", "queued", "pending", "requested", "waiting")
SHARED_GPU1_INVENTORY_ATTEMPTS = 3


class InventorySnapshotError(RuntimeError):
    def __init__(self, message: str, source: str, pages: object):
        super().__init__(message)
        self.source = source
        self.pages = pages


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def api(path: str, *, pages: bool = False) -> dict | list[dict]:
    require(path.startswith(("repos/SceneWorks/inference/", "repos/SceneWorks/SceneWorks/",
                            "orgs/SceneWorks/")) and ".." not in path,
            "unexpected GitHub API path")
    command = ["gh", "api", path, *( ["--paginate", "--slurp"] if pages else [])]
    output = subprocess.check_output(command, text=True, encoding="utf-8", timeout=45)
    return json.loads(output)


def complete_pages(value: list[dict], key: str, source: str = "unknown") -> list[dict]:
    if not (isinstance(value, list) and value and
            all(isinstance(page, dict) and isinstance(page.get(key), list) for page in value)):
        raise InventorySnapshotError("missing paginated inventory", source, value)
    counts = [page.get("total_count") for page in value]
    rows = [row for page in value for row in page.get(key, [])]
    if not (all(type(count) is int and count >= 0 for count in counts) and
            len(set(counts)) == 1 and counts[0] == len(rows)):
        raise InventorySnapshotError("truncated paginated inventory", source, value)
    ids = [row.get("id") if isinstance(row, dict) else None for row in rows]
    if not (all(type(identifier) is int and identifier > 0 for identifier in ids) and
            len(set(ids)) == len(ids)):
        raise InventorySnapshotError("ambiguous paginated inventory", source, value)
    return rows


def snapshot(*, reviewed_gpu1: bool = False) -> dict:
    started = time.monotonic()
    queries = {"org": "orgs/SceneWorks/actions/runners?per_page=100",
               "inference": "repos/SceneWorks/inference/actions/runners?per_page=100",
               "app": "repos/SceneWorks/SceneWorks/actions/runners?per_page=100"}
    queries.update({f"{repo}:{status}": f"repos/{repo}/actions/runs?status={status}&per_page=100"
                    for repo in REPOS for status in STATUSES})
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {name: pool.submit(api, path, pages=(":" in name)) for name, path in queries.items()}
        responses = {name: future.result(timeout=50) for name, future in futures.items()}
    runners = {}
    for scope in ("org", "inference", "app"):
        payload = responses[scope]
        if not (isinstance(payload, dict) and type(payload.get("total_count")) is int and
                isinstance(payload.get("runners"), list) and
                payload["total_count"] == len(payload["runners"])):
            raise InventorySnapshotError("incomplete runner inventory", scope, payload)
        runners[scope] = payload["runners"]
    runs = {}
    for repo in REPOS:
        for status in STATUSES:
            for run in complete_pages(responses[f"{repo}:{status}"], "workflow_runs",
                                      f"{repo}:{status}"):
                if run.get("status") != "completed":
                    key = (repo, run["id"])
                    if key in runs and runs[key] != run:
                        raise InventorySnapshotError("run status inventory inconsistent",
                                                     f"{repo}:{status}",
                                                     {"earlier": runs[key], "later": run})
                    runs[key] = run
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {key: pool.submit(api, f"repos/{key[0]}/actions/runs/{key[1]}/jobs?per_page=100",
                                    pages=True) for key in runs}
        jobs = {key: complete_pages(future.result(timeout=50), "jobs", f"{key[0]}:{key[1]}")
                for key, future in futures.items()}
    # The two old zero-job rows are exceptions only while a direct run read
    # agrees with this cycle's status inventory and its fresh zero-job list.
    legacy_direct = {key: api(f"repos/{key[0]}/actions/runs/{key[1]}")
                     for key in runs if key[0] == "SceneWorks/inference" and
                     key[1] in LEGACY_ZERO_JOB}
    companion = None
    if reviewed_gpu1:
        foreign = api(f"repos/SceneWorks/inference/actions/runs/{gpu1.RUN}")
        companion = {"run": foreign,
                     "job": api(f"repos/SceneWorks/inference/actions/jobs/{gpu1.JOB}"),
                     "jobs": api(f"repos/SceneWorks/inference/actions/runs/{gpu1.RUN}/attempts/1/jobs?per_page=100")}
        if foreign.get("status") == "in_progress" and companion["job"].get("status") == "in_progress":
            companion["group"] = api(f"repos/SceneWorks/inference/actions/concurrency_groups/{gpu1.GROUP}")
    require(time.monotonic() - started <= 120, "cross-repository snapshot became stale")
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "runners": runners,
            "runs": runs, "jobs": jobs, "legacy_direct": legacy_direct,
            "reviewed_gpu1": companion}


def classify(data: dict, own_id: int, head: str, workflow: str,
             *, mode: str = "shared-host", own_job_name: str = "cuda") -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None and workflow in
            ("yue2-precision-proof.yml", "yue2-app-precision-profile.yml"),
            "invalid exact owned source")
    require(mode in {"shared-host", "shared-gpu1", "gpu0-with-reviewed-gpu1"} and
            (mode != "gpu0-with-reviewed-gpu1" or workflow == "yue2-precision-proof.yml"),
            "unreviewed GPU1 scheduling mode/workflow")
    require(own_job_name == "cuda" or
            (mode in {"shared-gpu1", "gpu0-with-reviewed-gpu1"} and own_job_name == "cuda_diagnostic"),
            "unreviewed owned GPU job name")
    companion_state = None
    if mode == "gpu0-with-reviewed-gpu1":
        companion = data.get("reviewed_gpu1")
        require(isinstance(companion, dict), "reviewed GPU1 direct inventory missing")
        foreign = companion["run"]
        companion_state = "completed" if companion["job"].get("status") == "completed" else "active"
        gpu1.run(foreign, active=foreign.get("status") == "in_progress")
        require(foreign.get("status") in {"in_progress", "completed"} and
                (companion_state != "active" or foreign.get("status") == "in_progress"),
                "reviewed GPU1 run/job state changed")
        gpu1.job(companion["job"], active=companion_state == "active")
        gpu1.inventory(companion["jobs"], active=companion_state == "active")
        if companion_state == "active":
            group = companion.get("group", {})
            members = group.get("group_members", [])
            require(group.get("group_name") == gpu1.GROUP and
                    group.get("total_count") == len(members) and
                    len([row for row in members if row.get("status") == "in_progress"]) == 1 and
                    any(row.get("run_id") == gpu1.RUN and row.get("status") == "in_progress" and
                        row.get("job_id") is None for row in members) and
                    all(row.get("status") in {"pending", "queued", "in_progress"} for row in members),
                    "reviewed GPU1 old-group reservation changed")
        else:
            require(companion.get("group") is None,
                    "completed GPU1 job retained an ambiguous group reservation")
    runners = data["runners"]
    observed = {}
    for scope in ("org", "inference", "app"):
        for row in runners[scope]:
            labels = {item.get("name", "").lower() for item in row.get("labels", [])}
            if "cuda" in labels or row.get("name", "").lower().startswith("cuda-windows"):
                require(row["name"] not in observed, "duplicate physical CUDA listener")
                observed[row["name"]] = (row, scope)
    require(set(observed) == set(RUNNERS) and not any(
        "cuda" in {item.get("name", "").lower() for item in row.get("labels", [])} or
        row.get("name", "").lower().startswith("cuda-windows")
        for row in runners["inference"]), "CUDA listener set changed")
    for name, expected in RUNNERS.items():
        row, scope = observed[name]
        require(row.get("id") == expected and row.get("status") in ("online", "offline") and
                scope == ("org" if expected in (2313, 2619) else "app"),
                "physical CUDA runner identity/status changed")
    own_key = ("SceneWorks/inference", own_id)
    own = data["runs"].get(own_key)
    require(isinstance(own, dict) and own.get("head_sha") == head and
            own.get("run_attempt") == 1 and own.get("event") == "workflow_dispatch" and
            own.get("path") == f".github/workflows/{workflow}" and
            own.get("status") == "in_progress" and own.get("conclusion") is None,
            "owned run/source/attempt/status changed")
    own_jobs = data["jobs"].get(own_key, [])
    selected = [job for job in own_jobs if job.get("name") == own_job_name and
                job.get("status") == "in_progress"]
    require(len(selected) == 1 and selected[0].get("run_id") == own_id and
            selected[0].get("run_attempt") == 1 and selected[0].get("head_sha") == head and
            selected[0].get("conclusion") is None and
            selected[0].get("runner_name") in ("cuda-windows", "cuda-windows-2") and
            selected[0].get("runner_id") == RUNNERS[selected[0]["runner_name"]],
            "exact owned CUDA job/runner missing")
    owned = selected[0]
    require(observed[owned["runner_name"]][0].get("status") == "online",
            "owned CUDA runner is offline")
    require(observed[owned["runner_name"]][0].get("busy") is True,
            "owned CUDA runner is not assigned")
    if mode == "gpu0-with-reviewed-gpu1":
        require(owned["runner_name"] == "cuda-windows-2" and owned["runner_id"] == 2619,
                "reviewed GPU1 route must own the other Windows listener")
    for name, (row, _) in observed.items():
        if mode == "shared-gpu1":
            continue  # Foreign runner occupancy is observed, never mistaken for owned exclusivity.
        expected_busy = name == owned["runner_name"] or (
            mode == "gpu0-with-reviewed-gpu1" and companion_state == "active" and
            name == gpu1.RUNNER)
        require(row.get("busy") is expected_busy,
                f"unaccounted busy/free physical listener: {name}")
        require(row.get("status") == "online" or row.get("busy") is False,
                f"offline CUDA listener has an active job: {name}")
    historical = []
    foreign_runs = []
    for key, run in data["runs"].items():
        if key == own_key:
            require(all(job is owned or job.get("status") == "completed" for job in own_jobs),
                    "another owned job is active")
            continue
        if mode == "shared-gpu1":
            foreign_runs.append({"repository": key[0], "run_id": key[1],
                                 "status": run.get("status"), "job_count": len(data["jobs"].get(key, []))})
            continue  # Keep the complete inventory in each watch receipt without blocking foreign work.
        if mode == "gpu0-with-reviewed-gpu1" and key == ("SceneWorks/inference", gpu1.RUN):
            listed_jobs = data["jobs"].get(key, [])
            gpu1.inventory({"total_count": len(listed_jobs), "jobs": listed_jobs},
                           active=companion_state == "active")
            require(all(run.get(field) == data["reviewed_gpu1"]["run"].get(field)
                        for field in ("id", "head_sha", "run_attempt", "event", "path", "created_at")) and
                    ({row.get("id") for row in listed_jobs} ==
                     {row.get("id") for row in data["reviewed_gpu1"]["jobs"]["jobs"]}) and
                    (companion_state == "active" or
                     all(row.get("status") == "completed" for row in listed_jobs)),
                    "reviewed GPU1 snapshot/direct identity changed")
            continue
        jobs = data["jobs"].get(key, [])
        if key[0] == "SceneWorks/inference" and key[1] in LEGACY_ZERO_JOB:
            expected = LEGACY_ZERO_JOB[key[1]]
            direct = data.get("legacy_direct", {}).get(key)
            require(isinstance(direct, dict) and all(direct.get(field) == run.get(field)
                    for field in ("id", "head_sha", "run_attempt", "created_at", "updated_at",
                                  "status", "conclusion", "event", "path")) and
                    direct.get("repository", {}).get("full_name") == key[0],
                    "historical direct run readback missing or changed")
            require((run.get("head_sha"), run.get("run_attempt"), run.get("created_at"),
                     run.get("updated_at")) == expected and
                    run.get("status") == "queued" and run.get("conclusion") is None and
                    run.get("event") == "pull_request" and
                    run.get("path") == ".github/workflows/ci.yml" and not jobs,
                    "historical zero-job exception changed")
            historical.append(key[1])
            continue
        # An allocated job with exact non-CUDA labels cannot use these four
        # listeners. A run without jobs, unassigned job, or missing labels may
        # acquire one later, so it remains a reservation until proven otherwise.
        require(jobs, f"foreign run has no allocated jobs: {key}")
        active_count = 0
        for job in jobs:
            if job.get("status") == "completed":
                continue
            active_count += 1
            labels = job.get("labels")
            require(isinstance(labels, list) and labels and
                    all(isinstance(label, str) and label for label in labels),
                    f"foreign job labels unavailable: {key}")
            lowered = {label.lower() for label in labels}
            require("cuda" not in lowered and
                    job.get("runner_name") not in RUNNERS and
                    job.get("runner_id") not in RUNNERS.values() and
                    job.get("status") == "in_progress" and
                    isinstance(job.get("runner_name"), str) and
                    isinstance(job.get("runner_id"), int),
                    f"foreign CUDA or unknown job: {key}:{job.get('id')}")
        require(active_count > 0, f"foreign run has no active assigned non-CUDA job: {key}")
    if mode == "gpu0-with-reviewed-gpu1":
        require(companion_state != "active" or
                ("SceneWorks/inference", gpu1.RUN) in data["runs"],
                "reviewed GPU1 active run absent from complete inventory")
    return {"own_run": own_id, "own_job": owned["id"],
            "own_runner": owned["runner_name"], "historical_zero_job_runs": historical,
            "foreign_runs_observed": foreign_runs,
            "reviewed_gpu1": companion_state,
            "physical_lease": False}


def owned_run(own_id: int, head: str, workflow: str) -> dict:
    row = api(f"repos/SceneWorks/inference/actions/runs/{own_id}")
    require(isinstance(row, dict) and row.get("id") == own_id and
            row.get("head_sha") == head and row.get("run_attempt") == 1 and
            row.get("event") == "workflow_dispatch" and
            row.get("path") == f".github/workflows/{workflow}" and
            row.get("repository", {}).get("full_name") == "SceneWorks/inference",
            "owned run/source/attempt changed")
    return row


def bind_owned_job(own_id: int, head: str, workflow: str, job_id: int,
                   runner_name: str, runner_id: int,
                   *, own_job_name: str = "cuda") -> dict:
    run = owned_run(own_id, head, workflow)
    require(run.get("status") == "in_progress" and run.get("conclusion") is None,
            "owned run not active for binding")
    require(isinstance(run.get("created_at"), str) and run["created_at"],
            "owned run start identity unavailable")
    job = api(f"repos/SceneWorks/inference/actions/jobs/{job_id}")
    require(isinstance(job, dict) and job.get("id") == job_id and
            job.get("run_id") == own_id and job.get("run_attempt") == 1 and
            job.get("head_sha") == head and job.get("name") == own_job_name and
            job.get("runner_name") == runner_name and job.get("runner_id") == runner_id and
            job.get("status") == "in_progress" and job.get("conclusion") is None and
            job.get("completed_at") is None and isinstance(job.get("started_at"), str),
            "owned job/runner/start binding unavailable")
    return {"run": run, "job": job, "start": job["started_at"]}


def cancel_bound_run(own_id: int, head: str, workflow: str, binding: dict,
                     *, identity_drift: bool) -> None:
    if identity_drift:
        return  # An observed positive identity drift revokes cancellation.
    try:
        current = api(f"repos/SceneWorks/inference/actions/runs/{own_id}")
    except Exception:
        current = None  # Only the previously authenticated immutable run/job may be canceled.
    if current is not None:
        if not isinstance(current, dict) or any((
                current.get("id") != own_id,
                current.get("head_sha") != head,
                current.get("run_attempt") != 1,
                current.get("event") != "workflow_dispatch",
                current.get("path") != f".github/workflows/{workflow}",
                current.get("repository", {}).get("full_name") != "SceneWorks/inference",
                current.get("created_at") != binding["run"].get("created_at"))):
            return  # Positive source/run drift revokes cancellation.
        if current.get("status") == "completed":
            return
    require(binding["job"].get("id") == binding["job_id"] and
            binding["start"] == binding["job"].get("started_at"),
            "cached owned job binding changed")
    subprocess.run(["gh", "run", "cancel", str(own_id), "-R", "SceneWorks/inference"],
                   check=True, timeout=30)


def watch(own_id: int, head: str, workflow: str, output: Path, seconds: int, interval: int,
          job_id: int, runner_name: str, runner_id: int,
          *, mode: str = "shared-host", own_job_name: str = "cuda") -> None:
    require(0 < seconds <= 480 * 60 and 5 <= interval <= 60 and not output.exists(),
            "watch interval/duration/output invalid")
    require((runner_name, runner_id) in (("cuda-windows", 2313), ("cuda-windows-2", 2619)),
            "owned runner binding invalid")
    require(mode in {"shared-host", "shared-gpu1", "gpu0-with-reviewed-gpu1"} and
            (mode != "gpu0-with-reviewed-gpu1" or
             (workflow == "yue2-precision-proof.yml" and runner_name == "cuda-windows-2" and runner_id == 2619)),
            "unreviewed GPU1 watcher placement")
    require(own_job_name == "cuda" or
            (mode in {"shared-gpu1", "gpu0-with-reviewed-gpu1"} and own_job_name == "cuda_diagnostic"),
            "unreviewed owned GPU job name")
    # No cancellation if the initial direct run/job/runner authentication fails.
    binding = bind_owned_job(own_id, head, workflow, job_id, runner_name, runner_id,
                             own_job_name=own_job_name)
    binding["job_id"] = job_id
    try:
        output.mkdir(parents=True)
    except BaseException:
        cancel_bound_run(own_id, head, workflow, binding, identity_drift=False)
        raise
    deadline = time.monotonic() + seconds
    index = 0
    identity_drift = False
    source_checked = False
    while time.monotonic() < deadline:
        index += 1
        try:
            if mode == "gpu0-with-reviewed-gpu1" and not source_checked:
                for path in gpu1.SOURCES:
                    gpu1.source(api(f"repos/SceneWorks/inference/contents/{path}?ref={gpu1.HEAD}"), path)
                source_checked = True
            direct = owned_run(own_id, head, workflow)
            require(direct.get("created_at") == binding["run"].get("created_at") and
                    direct.get("repository", {}).get("full_name") == "SceneWorks/inference",
                    "owned immutable run identity drifted")
            if direct.get("status") == "completed":
                (output / "terminal.json").write_text(json.dumps(direct, indent=2) + "\n", encoding="utf-8")
                return  # Final child/postflight and physical release still require independent audit.
            for inventory_attempt in range(1, SHARED_GPU1_INVENTORY_ATTEMPTS + 1):
                try:
                    data = snapshot(reviewed_gpu1=mode == "gpu0-with-reviewed-gpu1")
                    break
                except (InventorySnapshotError, subprocess.CalledProcessError,
                        subprocess.TimeoutExpired, TimeoutError, json.JSONDecodeError) as error:
                    if mode != "shared-gpu1":
                        raise
                    # A changing REST status count or transport failure never
                    # supplies a partial inventory. Retain the rejected page,
                    # then fetch a wholly new snapshot after direct own reauth.
                    (output / f"inventory-attempt-{index:04d}-{inventory_attempt}.json").write_text(
                        json.dumps({"errorType": type(error).__name__, "error": str(error),
                                    "source": getattr(error, "source", None),
                                    "pages": getattr(error, "pages", None)}, indent=2) + "\n",
                        encoding="utf-8")
                    if (inventory_attempt == SHARED_GPU1_INVENTORY_ATTEMPTS or
                            time.monotonic() >= deadline):
                        raise
                    try:
                        fresh = bind_owned_job(own_id, head, workflow, job_id, runner_name,
                                               runner_id, own_job_name=own_job_name)
                    except RuntimeError as identity_error:
                        # bind_owned_job uses RuntimeError for a positive run/job identity
                        # mismatch. API transport and decoding failures are not drift: the
                        # previously authenticated binding still permits owned-only cancel.
                        raise RuntimeError(
                            "owned identity drifted during inventory retry"
                        ) from identity_error
                    require(fresh["run"].get("created_at") == binding["run"].get("created_at") and
                            fresh["job"].get("id") == binding["job_id"] and
                            fresh["job"].get("started_at") == binding["start"],
                            "owned identity drifted during inventory retry")
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
            observed_job = next((job for job in data["jobs"].get(("SceneWorks/inference", own_id), [])
                                 if job.get("id") == job_id), None)
            if observed_job is not None and any((
                    observed_job.get("run_id") != own_id,
                    observed_job.get("run_attempt") != 1,
                    observed_job.get("head_sha") != head,
                    observed_job.get("runner_id") != runner_id,
                    observed_job.get("runner_name") != runner_name,
                    observed_job.get("started_at") != binding["start"])):
                raise RuntimeError("owned job identity drifted")
            proof = classify(data, own_id, head, workflow, mode=mode, own_job_name=own_job_name)
            require(proof["own_job"] == job_id and proof["own_runner"] == runner_name and
                    observed_job is not None and observed_job.get("started_at") == binding["start"],
                    "inventory differs from direct immutable owned job binding")
            (output / f"{index:04d}.json").write_text(
                json.dumps({"checked_at": data["checked_at"], "proof": proof,
                            "runners": data["runners"], "runs": [
                                {"repo": repo, "id": rid, "head_sha": run.get("head_sha"),
                                 "status": run.get("status"), "run_attempt": run.get("run_attempt")}
                                for (repo, rid), run in data["runs"].items()],
                            "jobs": [{"repo": repo, "run": rid, "rows": [
                                {"id": job.get("id"), "name": job.get("name"),
                                 "status": job.get("status"), "runner_name": job.get("runner_name")}
                                for job in rows]} for (repo, rid), rows in data["jobs"].items()]},
                           indent=2) + "\n",
                encoding="utf-8")
        except BaseException as error:
            if "identity drifted" in str(error) or "owned run/source/attempt changed" in str(error):
                identity_drift = True
            try:
                current = owned_run(own_id, head, workflow)
                if current.get("status") == "completed":
                    (output / "terminal.json").write_text(
                        json.dumps({"observed_after_error": str(error), "run": current}, indent=2) + "\n",
                        encoding="utf-8")
                    return
            except Exception:
                pass  # Source was authenticated before the watch; cancel that one run.
            try:
                (output / "refusal.txt").write_text(str(error) + "\n", encoding="utf-8")
            finally:
                cancel_bound_run(own_id, head, workflow, binding, identity_drift=identity_drift)
            raise
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    cancel_bound_run(own_id, head, workflow, binding, identity_drift=identity_drift)
    raise TimeoutError("bounded shared-host watch ended; no physical release inferred")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-run-id", type=int, required=True)
    parser.add_argument("--control-sha", required=True)
    parser.add_argument("--workflow", choices=("yue2-precision-proof.yml", "yue2-app-precision-profile.yml"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--expected-job-id", type=int, required=True)
    parser.add_argument("--expected-runner-name", choices=("cuda-windows", "cuda-windows-2"), required=True)
    parser.add_argument("--expected-runner-id", type=int, choices=(2313, 2619), required=True)
    parser.add_argument("--mode", choices=("shared-host", "shared-gpu1", "gpu0-with-reviewed-gpu1"),
                        default="shared-host")
    parser.add_argument("--expected-job-name", choices=("cuda", "cuda_diagnostic"),
                        default="cuda")
    args = parser.parse_args()
    watch(args.own_run_id, args.control_sha, args.workflow, args.output, args.seconds, args.interval,
          args.expected_job_id, args.expected_runner_name, args.expected_runner_id,
          mode=args.mode, own_job_name=args.expected_job_name)


if __name__ == "__main__":
    main()

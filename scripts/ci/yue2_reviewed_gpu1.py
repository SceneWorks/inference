"""Exact, temporary source and job identity for the reviewed GPU1 benchmark."""

import base64
import hashlib

RUN = 37203890421
JOB = 111441010524
HEAD = "e13f235d5d371b301e4a04f209e8ad621b1fda9f"
RUNNER = "cuda-windows"
RUNNER_ID = 2313
CREATED = "2026-10-04T12:57:12Z"
STARTED = "2026-10-04T12:57:29Z"
NAME = "Decode-speedups benchmark campaign (Candle/CUDA)"
GROUP = "inference-real-weights-physical-host"
SOURCES = {
    ".github/workflows/real-weights.yml": "6d586177edd06a208d0bc72ed86e16eb207e2bf82b2c259ff7217a1d8a21f757",
    "scripts/ci/real-weights/candle-decode-speedups-bench/run-every-matrix-row.cmd": "157eb7ad92dcb1330e9dcbf8194ead42a262baf2723a3ab0b0f21a15b8a64756",
    "scripts/release/speculative_bench_campaign.py": "97b188879d5d8c5049b6490d146d6e6e670c278beb483fc4ab357019442c9b01",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def source(payload: dict, path: str) -> None:
    require(path in SOURCES and payload.get("path") == path and
            payload.get("encoding") == "base64", "reviewed GPU1 source metadata changed")
    try:
        raw = base64.b64decode("".join(payload["content"].split()), validate=True)
    except (KeyError, ValueError) as error:
        raise RuntimeError("reviewed GPU1 source bytes unavailable") from error
    require(hashlib.sha256(raw).hexdigest() == SOURCES[path],
            "reviewed GPU1 source bytes changed")


def run(value: dict, *, active: bool) -> None:
    require(value.get("id") == RUN and value.get("head_sha") == HEAD and
            value.get("run_attempt") == 1 and value.get("event") == "workflow_dispatch" and
            value.get("path") == ".github/workflows/real-weights.yml" and
            value.get("repository", {}).get("full_name") == "SceneWorks/inference" and
            value.get("created_at") == CREATED and value.get("name") == "Real-weight validation",
            "reviewed GPU1 run identity changed")
    if active:
        require(value.get("status") == "in_progress" and value.get("conclusion") is None,
                "reviewed GPU1 run no longer active")
    else:
        require(value.get("status") == "completed" and
                isinstance(value.get("updated_at"), str) and value["updated_at"],
                "reviewed GPU1 completion unavailable")


def job(value: dict, *, active: bool) -> None:
    require(value.get("id") == JOB and value.get("run_id") == RUN and
            value.get("run_attempt") == 1 and value.get("head_sha") == HEAD and
            value.get("name") == NAME and value.get("workflow_name") == "Real-weight validation" and
            value.get("runner_name") == RUNNER and value.get("runner_id") == RUNNER_ID and
            value.get("started_at") == STARTED,
            "reviewed GPU1 job/source/runner/start changed")
    if active:
        require(value.get("status") == "in_progress" and value.get("conclusion") is None and
                value.get("completed_at") is None, "reviewed GPU1 job no longer active")
    else:
        require(value.get("status") == "completed" and
                isinstance(value.get("completed_at"), str) and value["completed_at"],
                "reviewed GPU1 job completion unavailable")


def inventory(payload: dict, *, active: bool) -> dict:
    rows = payload.get("jobs")
    require(payload.get("total_count") == 55 and isinstance(rows, list) and len(rows) == 55,
            "reviewed GPU1 job inventory incomplete")
    require(len({row.get("id") for row in rows}) == 55 and
            all(type(row.get("id")) is int for row in rows),
            "reviewed GPU1 job inventory has duplicate or invalid IDs")
    selected = [row for row in rows if row.get("id") == JOB]
    require(len(selected) == 1 and all(row is selected[0] or
            (row.get("status") == "completed" and row.get("conclusion") == "skipped")
            for row in rows), "reviewed GPU1 selected job set changed")
    job(selected[0], active=active)
    return selected[0]

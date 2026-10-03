"""One reviewed, expiring WDDM desktop-context exception for the YuE2 CUDA controls.

The artifact is a pinned owner-window receipt, not a process-name allowlist. A
fresh probe must still establish the same process and device, zero sampled
engine activity, and no rise in its device residency. Missing data refuses.
"""
from __future__ import annotations

import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

RUN_ID = "37145909196"
BASELINE_ENGINE_SHA = "4127a675fc8575555e029e01b7f6867488880a8f"
BASELINE_CONTROL_SHA = "e538120368ac279cc42176c44f9d88fa5af9c9b4"
BASELINE_DIGEST = "d67f8d2c7cd79040bbe48ada71e6e27722237a3316bf622e69808ca341f77a2c"
BASELINE_RUNNER = "cuda-windows"
WINDOW = timedelta(hours=12)


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(why)


def read_json(directory: Path, name: str) -> dict:
    value = json.loads((directory / f"{name}.json").read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{name} is not an object")
    return value


def artifact_digest(directory: Path) -> str:
    files = sorted(directory.iterdir())
    require(files and all(item.is_file() and item.suffix == ".json" for item in files),
            "reviewed receipt contains missing or unexpected files")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _gpu_row(directory: Path, name: str, *, has_free: bool) -> dict:
    sample = read_json(directory, name)
    require(sample.get("exitCode") == 0, f"{name} unavailable")
    rows = [next(csv.reader([line], skipinitialspace=True)) for line in sample.get("output", [])]
    rows = [row for row in rows if row and row[0].strip() == "0"]
    require(len(rows) == 1 and len(rows[0]) == (10 if has_free else 9), f"{name} GPU0 ambiguous")
    row = [field.strip() for field in rows[0]]
    require(row[1].startswith("GPU-") and re.fullmatch(r"[0-9A-Fa-f:.]{12,}", row[2]) is not None,
            f"{name} GPU0 identity invalid")
    numbers = row[5:]
    require(all(re.fullmatch(r"[0-9]+", value) for value in numbers), f"{name} telemetry unavailable")
    total, used = int(row[5]), int(row[6])
    free = int(row[7]) if has_free else None
    util = [int(value) for value in row[8 if has_free else 7:]]
    require(total > 0 and 0 <= used <= total and (free is None or 0 < free <= total),
            f"{name} invalid memory")
    require(util == [0, 0], f"{name} selected GPU activity is nonzero")
    return {"uuid": row[1], "pci": row[2].lower().lstrip("0"), "totalMiB": total,
            "usedMiB": used, "freeMiB": free}


def _pmon(directory: Path, name: str, pid: int) -> None:
    sample = read_json(directory, name)
    require(sample.get("exitCode") == 0, f"{name} unavailable")
    _pmon_output(sample.get("output", []), name, pid)


def _pmon_output(output: list[str], name: str, pid: int) -> None:
    columns = None
    processes = set()
    compute_rows = 0
    for line in output:
        fields = line.split()
        if line.startswith("#"):
            if "pid" in fields and "type" in fields:
                columns = {field: index - 1 for index, field in enumerate(fields) if index > 0}
            continue
        if not fields:
            continue
        require(columns is not None and all(key in columns for key in ("gpu", "pid", "type")),
                f"{name} missing typed pmon header")
        require(all(key in columns for key in ("sm", "mem", "enc", "dec", "jpg", "ofa")),
                f"{name} missing utilization columns")
        require(len(fields) > max(columns[key] for key in ("gpu", "pid", "type")), f"{name} truncated row")
        gpu, proc, kind = (fields[columns[key]] for key in ("gpu", "pid", "type"))
        require(gpu == "0", f"{name} wrong physical GPU")
        if proc == "-" and kind == "-":
            continue
        require(proc.isdigit() and kind in ("C", "C+G", "G"), f"{name} unknown process type")
        for metric in ("sm", "mem", "enc", "dec", "jpg", "ofa"):
            require(len(fields) > columns[metric], f"{name} truncated utilization")
            value = fields[columns[metric]]
            if value == "-":
                continue  # Unsupported pmon telemetry still needs valid Windows counters.
            require(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is not None and float(value) == 0,
                    f"{name} active or invalid {metric} utilization")
        if kind != "G":
            processes.add((int(proc), kind))
            compute_rows += 1
    require(columns is not None and compute_rows == 1 and processes == {(pid, "C+G")},
            f"{name} mixed/compute process set changed: {sorted(processes)}")


def _compute_apps(directory: Path, name: str, pid: int) -> None:
    sample = read_json(directory, name)
    require(sample.get("exitCode") == 0, f"{name} unavailable")
    rows = [next(csv.reader([line], skipinitialspace=True)) for line in sample.get("output", [])]
    require(len(rows) == 1 and len(rows[0]) == 3 and rows[0][0].strip().isdigit() and
            rows[0][1].strip() and int(rows[0][0]) == pid,
            f"{name} compute process set changed or ambiguous")


def _counters(directory: Path, epoch: int, pid: int, luid: str, *, baseline: bool) -> dict:
    value = read_json(directory, f"windows-counters-{epoch}")
    require(value.get("targetPid") == pid, "counter target PID changed")
    counters = {row.get("counter"): row for row in value.get("counters", [])}
    require(len(counters) == len(value.get("counters", [])), "duplicate Windows counter path")
    expected = {
        "engine": r"\GPU Engine(*)\Utilization Percentage",
        "processDedicated": r"\GPU Process Memory(*)\Dedicated Usage",
        "processShared": r"\GPU Process Memory(*)\Shared Usage",
        "processCommitted": r"\GPU Process Memory(*)\Total Committed",
        "adapterDedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
        "adapterShared": r"\GPU Adapter Memory(*)\Shared Usage",
        "adapterCommitted": r"\GPU Adapter Memory(*)\Total Committed",
    }
    if baseline:
        expected.pop("adapterShared")
        expected.pop("adapterCommitted")
    result = {}
    for key, path in expected.items():
        row = counters.get(path)
        require(isinstance(row, dict) and "error" not in row, f"{path} missing or unsupported")
        prefix = f"pid_{pid}_{luid}_" if key.startswith(("engine", "process")) else f"{luid}_"
        samples = [sample for sample in row.get("samples", [])
                   if sample.get("instance", "").lower().startswith(prefix)]
        require(samples and all(str(sample.get("status")) == "0" for sample in samples),
                f"{path} target status missing or invalid")
        require(all(type(sample.get("cookedValue")) in (int, float) and
                    math.isfinite(sample["cookedValue"]) and sample["cookedValue"] >= 0
                    for sample in samples), f"{path} target value invalid")
        if key == "engine":
            result[key] = {sample["instance"].lower(): sample["cookedValue"] for sample in samples}
            require(len(result[key]) == len(samples) and all(value == 0 for value in result[key].values()),
                    "selected process GPU engine active or ambiguous")
        else:
            require(len(samples) == 1, f"{path} target adapter ambiguous")
            result[key] = samples[0]["cookedValue"]
    return result


def summarize(directory: Path, *, baseline: bool, pid: int, engine_sha: str, control_sha: str) -> dict:
    manifest = read_json(directory, "manifest")
    expected_runner = BASELINE_RUNNER if baseline else os.environ.get("RUNNER_NAME")
    require(manifest.get("completed") is True and manifest.get("targetPid") == pid and
            manifest.get("engineSha") == engine_sha and manifest.get("controlSha") == control_sha and
            expected_runner in ("cuda-windows", "cuda-windows-2") and
            manifest.get("runner") == expected_runner, "diagnostic manifest/source/runner mismatch")
    catalog = read_json(directory, "windows-counter-catalog")
    listed = {item.get("name"): item for item in catalog.get("sets", [])}
    require(len(listed) == len(catalog.get("sets", [])), "duplicate Windows counter set")
    for name, paths in {
        "GPU Engine": {r"\GPU Engine(*)\Utilization Percentage"},
        "GPU Process Memory": {r"\GPU Process Memory(*)\Dedicated Usage",
                               r"\GPU Process Memory(*)\Shared Usage",
                               r"\GPU Process Memory(*)\Total Committed"},
        "GPU Adapter Memory": {r"\GPU Adapter Memory(*)\Dedicated Usage"},
    }.items():
        item = listed.get(name)
        require(isinstance(item, dict) and "error" not in item and paths.issubset(set(item.get("paths", []))),
                f"{name} counter catalog incomplete")
    before, after = (read_json(directory, f"process-{when}") for when in ("before", "after"))
    identity = (before.get("pid"), before.get("name"), before.get("executablePath"), before.get("creationDate"))
    require(identity[0] == pid and all(identity[1:]) and
            identity == (after.get("pid"), after.get("name"), after.get("executablePath"), after.get("creationDate")),
            "target process PID/start/image changed")
    adapter = read_json(directory, "cuda-adapter-map")
    devices = adapter.get("devices", [])
    require(adapter.get("cuInit") == adapter.get("cuDeviceGetCount") == 0 and
            len(devices) == 1 and devices[0].get("ordinal") == 0 and
            devices[0].get("cuDeviceGet") == devices[0].get("cuDeviceGetPCIBusId") ==
            devices[0].get("cuDeviceGetLuid") == 0 and devices[0].get("nodeMask") == 1,
            "selected CUDA device mapping unavailable")
    raw_luid = bytes.fromhex(devices[0]["luidBytes"].replace("-", ""))
    require(len(raw_luid) == 8, "CUDA adapter LUID invalid")
    luid = f"luid_0x{int.from_bytes(raw_luid[4:], 'little'):08x}_0x{int.from_bytes(raw_luid[:4], 'little'):08x}"
    gpu = [_gpu_row(directory, f"gpu-sample-{index}", has_free=not baseline) for index in range(3)]
    require(all(row["uuid"] == gpu[0]["uuid"] and row["pci"] == gpu[0]["pci"] and
                row["totalMiB"] == gpu[0]["totalMiB"] and row["usedMiB"] == gpu[0]["usedMiB"] and
                row["freeMiB"] == gpu[0]["freeMiB"] for row in gpu), "selected GPU identity/residency changed")
    require(devices[0]["pciBusId"].lower().lstrip("0") == gpu[0]["pci"],
            "NVML GPU0 does not map to CUDA ordinal0/LUID")
    for index in range(3):
        mode = read_json(directory, f"driver-mode-{index}")
        require(mode.get("exitCode") == 0, "WDDM mode unavailable")
        rows = [next(csv.reader([line], skipinitialspace=True)) for line in mode.get("output", [])]
        rows = [row for row in rows if row and row[0].strip() == "0"]
        require(len(rows) == 1 and rows[0][1].strip() == gpu[0]["uuid"] and
                rows[0][2].strip() == "WDDM", "selected GPU is not verified WDDM")
        _pmon(directory, f"pmon-0-{index}", pid)
        _compute_apps(directory, f"compute-apps-0-{index}", pid)
    if not baseline:
        _pmon(directory, "pmon-0-final", pid)
    for name in ("gpu-before-cuda-properties", "gpu-after-cuda-properties"):
        sample = read_json(directory, name)
        require(sample.get("exitCode") == 0, f"{name} unavailable")
        rows = [next(csv.reader([line], skipinitialspace=True)) for line in sample.get("output", [])]
        rows = [row for row in rows if row and row[0].strip() == "0"]
        require(len(rows) == 1 and len(rows[0]) == 5 and
                rows[0][1].strip() == gpu[0]["uuid"] and
                rows[0][2].strip().lower().lstrip("0") == gpu[0]["pci"] and
                rows[0][3].strip().isdigit() and int(rows[0][3]) == gpu[0]["usedMiB"] and
                rows[0][4].strip() == "0", f"{name} selected device changed or active")
    counters = [_counters(directory, index, pid, luid, baseline=baseline) for index in range(3)]
    require(all(row == counters[0] for row in counters), "counter engine set or residency changed")
    require(len(counters[0]["engine"]) > 0, "selected process has no valid GPU engine set")
    return {"identity": identity, "uuid": gpu[0]["uuid"], "pci": gpu[0]["pci"],
            "luid": luid, "gpu": gpu[0], "counters": counters[0],
            "completedUtc": manifest.get("completedUtc")}


def validate_current(current: dict, baseline: dict) -> None:
    for key in ("identity", "uuid", "pci", "luid"):
        require(current[key] == baseline[key], f"reviewed {key} changed")
    require(set(current["counters"]["engine"]) == set(baseline["counters"]["engine"]),
            "reviewed GPU engine counter set changed")
    require(current["counters"]["engine"] and
            all(value == 0 for value in current["counters"]["engine"].values()),
            "selected process GPU engine active")
    require(current["gpu"]["totalMiB"] == baseline["gpu"]["totalMiB"] and
            current["gpu"]["usedMiB"] <= baseline["gpu"]["usedMiB"],
            "selected device memory rose over reviewed baseline")
    for key in ("processDedicated", "processShared", "processCommitted", "adapterDedicated"):
        require(current["counters"][key] <= baseline["counters"][key],
                f"{key} rose over reviewed baseline")


def check_window(completed_utc: str, now: datetime) -> None:
    completed = parse_completed_utc(completed_utc)
    require(completed <= now <= completed + WINDOW, "reviewed owner window expired or not yet open")


def parse_completed_utc(completed_utc: str) -> datetime:
    # PowerShell writes 100-ns ticks; datetime retains microseconds. Truncate
    # the seventh digit so the expiration is conservative by at most 0.9 us.
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})\dZ", completed_utc)
    require(match is not None, "reviewed owner completion timestamp invalid")
    return datetime.fromisoformat(match.group(1) + "+00:00")


def check_dispatch(run_id: str | None, engine_sha: str | None,
                   control_sha: str | None, github_sha: str | None) -> None:
    require(run_id == RUN_ID, "no reviewed owner-window run selected")
    require(bool(engine_sha) and re.fullmatch(r"[0-9a-f]{40}", engine_sha) is not None and
            control_sha == github_sha and bool(control_sha) and
            re.fullmatch(r"[0-9a-f]{40}", control_sha) is not None,
            "reviewed source/control SHA mismatch")


def verify_artifact(directory: Path) -> None:
    require(directory.is_dir() and artifact_digest(directory) == BASELINE_DIGEST,
            "reviewed diagnostic artifact digest mismatch")


def check_device_selection(platform: str, visible: str | None, order: str | None) -> None:
    require(platform == "nt" and visible == "0" and order == "PCI_BUS_ID",
            "reviewed context requires Windows and PCI-ordered CUDA GPU0")


def reviewed_baseline() -> tuple[dict, Path]:
    check_device_selection(os.name, os.environ.get("CUDA_VISIBLE_DEVICES"),
                           os.environ.get("CUDA_DEVICE_ORDER"))
    check_dispatch(os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID"),
                   os.environ.get("EXPECTED_ENGINE_SHA"),
                   os.environ.get("EXPECTED_CONTROL_SHA"), os.environ.get("GITHUB_SHA"))
    path = os.environ.get("YUE2_IDLE_CONTEXT_RECEIPT_DIR") or (
        str(Path(os.environ["RUNNER_TEMP"]) / "yue2-reviewed-idle-context")
        if os.environ.get("RUNNER_TEMP") else "")
    require(bool(path), "reviewed receipt path missing")
    directory = Path(path)
    verify_artifact(directory)
    baseline = summarize(directory, baseline=True, pid=38212,
                         engine_sha=BASELINE_ENGINE_SHA, control_sha=BASELINE_CONTROL_SHA)
    check_window(baseline["completedUtc"], datetime.now(timezone.utc))
    return baseline, directory


def require_remaining_window(seconds: int) -> tuple[dict, Path]:
    """Refuse before weight load unless the owned deadline and postflight fit."""
    baseline, directory = reviewed_baseline()
    completed = parse_completed_utc(baseline["completedUtc"])
    require(datetime.now(timezone.utc) + timedelta(seconds=seconds) <= completed + WINDOW,
            "reviewed owner window cannot cover bounded run and postflight")
    return baseline, directory


def diagnostic_file_pairs(directory: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Retain the original probe bytes and an exact decoded view of those bytes."""
    files = {item.name: item.read_bytes() for item in sorted(directory.iterdir()) if item.is_file()}
    return ({name: data.decode("utf-8-sig") for name, data in files.items()},
            {name: base64.b64encode(data).decode("ascii") for name, data in files.items()})


def census_mixed_context(pid: int, initial_pmon: str) -> tuple[str, bool]:
    baseline, _ = reviewed_baseline()
    require(pid == baseline["identity"][0], "unreviewed mixed-context PID")
    _pmon_output(initial_pmon.splitlines(), "initial pmon", pid)
    runtime_engine_sha = os.environ["EXPECTED_ENGINE_SHA"]
    with tempfile.TemporaryDirectory(prefix="yue2-cuda-guard-") as root:
        output = Path(root)
        script = Path(__file__).with_name("yue2_cuda_context_diagnostic.ps1")
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(script), "-TargetPid", str(pid), "-OutputDirectory", str(output),
                   "-EngineSha", runtime_engine_sha, "-ControlSha", os.environ["GITHUB_SHA"]]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=180)
        diagnostic_files, diagnostic_file_bytes = diagnostic_file_pairs(output)
        raw = {"initialPmon": initial_pmon, "reviewedBaseline": baseline,
               "commandExit": result.returncode,
               "diagnosticFiles": diagnostic_files,
               "diagnosticFileBytesB64": diagnostic_file_bytes}
        try:
            require(result.returncode == 0, f"fresh WDDM counter probe failed: {result.stderr.strip()}")
            current = summarize(output, baseline=False, pid=pid, engine_sha=runtime_engine_sha,
                                control_sha=os.environ["GITHUB_SHA"])
            validate_current(current, baseline)
            # The baseline may expire during a long app job; enforce at every call.
            check_window(baseline["completedUtc"], datetime.now(timezone.utc))
            return json.dumps(raw, sort_keys=True), True
        except Exception as error:
            raw["refusal"] = str(error)
            return json.dumps(raw, sort_keys=True), False

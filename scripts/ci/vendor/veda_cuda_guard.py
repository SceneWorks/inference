"""Physical admission and owned-child policy for the Windows CUDA host.

Default admission is free-memory headroom held stable across timed samples under the
host-global UUID lock (the CI-to-CI mutex). Foreign display or compute contexts and
background utilization are recorded but never refuse. The strict `idle` predicate is
kept only for timing benchmarks that co-tenant activity would corrupt.
"""
from __future__ import annotations
import csv
import json
import os
import re
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

MAX_SECONDS = 3600
ALLOWED_RUNNERS = {"cuda-windows", "cuda-windows-2"}
# Reserved display/WDDM baseline, not CUDA work: cap it at 2 GiB on this 96-GiB pool.
MAX_GRAPHICS_BASELINE_MIB = 2048
MAX_BASELINE_UTILIZATION_PERCENT = 5
BASELINE_SAMPLES = 5
BASELINE_INTERVAL_SECONDS = 0.5
MAX_SAMPLING_SECONDS = 5
# Default admission also needs a quiet card across the sampling window (same rule as the
# SceneWorks wrapper): memory not moving and utilization low and steady.
STABLE_MAX_USED_SPREAD_MIB = 256
STABLE_MAX_USED_GROWTH_MIB = 128
STABLE_MAX_UTILIZATION_PERCENT = 30
STABLE_MAX_UTILIZATION_SPREAD = 15


def command(args: list[str], deadline: float | None = None) -> str:
    timeout = 30 if deadline is None else min(30, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("GPU inventory sampling deadline exhausted")
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def inventory(out: Path | None = None, label: str = "snapshot",
              deadline: float | None = None, *, selected_uuid: str | None = None) -> tuple[list[dict], list[list[str]]]:
    # A source-authenticated shared-headroom caller needs fresh selected-card counters,
    # not a host-wide typed-process census. Strict callers retain the full inventory.
    args = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    if selected_uuid is not None:
        if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", selected_uuid):
            raise ValueError("invalid selected physical GPU UUID")
        args.append("--id=" + selected_uuid)
    rows = command(args, deadline)
    devices = []
    for row in csv.reader(rows.splitlines()):
        i, uuid, name, driver, total, used, free, utilization = [v.strip() for v in row]
        devices.append(dict(index=int(i), uuid=uuid, name=name, driver=driver, total_mib=int(total),
                            used_mib=int(used), free_mib=int(free),
                            utilization_percent=(None if selected_uuid is not None and utilization == "[N/A]"
                                                 else int(utilization))))
    if selected_uuid is not None:
        if len(devices) != 1 or devices[0]["uuid"] != selected_uuid:
            raise RuntimeError("selected physical GPU UUID absent or ambiguous; no device fallback")
        devices[0]["process_inventory_status"] = "unknown/not-collected-diagnostic-only"
        return devices, []
    xml = command(["nvidia-smi", "-q", "-x"], deadline)
    if out is not None:
        (out / f"{label}-nvidia-smi.xml").write_text(xml, encoding="utf-8", newline="\n")
    xml_uuids = {(gpu.findtext("uuid") or "").strip() for gpu in ET.fromstring(xml).findall("gpu")}
    if xml_uuids != {device["uuid"] for device in devices}:
        raise RuntimeError("driver XML and CSV GPU identities differ")
    processes = typed_processes(xml)
    # Cover permission-opaque processes using driver-supplied type, never executable names.
    # A PID present in the CSV but absent from the typed snapshot remains unknown/refused.
    rows = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"], deadline)
    known = {(p[0], p[1]) for p in processes}
    typed_by_pid = {}
    for uuid, pid, _, _ in processes:
        typed_by_pid.setdefault(pid, set()).add(uuid)
    for row in csv.reader(rows.splitlines()):
        if not row:
            continue
        row = [v.strip() for v in row]
        if len(row) != 3:
            raise RuntimeError("unrecognized GPU process inventory")
        if row[1] in typed_by_pid and row[0] not in typed_by_pid[row[1]]:
            raise RuntimeError("driver XML and compute-app GPU UUID/PID differ")
        if (row[0], row[1]) not in known:
            processes.append(row + ["unknown"])
    return devices, processes


def typed_processes(xml: str) -> list[list[str]]:
    result = []
    for gpu in ET.fromstring(xml).findall("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        if not uuid:
            raise RuntimeError("driver XML GPU identity missing")
        for process in gpu.findall("processes/process_info"):
            pid = (process.findtext("pid") or "").strip()
            if not pid:
                raise RuntimeError("driver XML process PID missing")
            result.append([uuid, pid, process.findtext("process_name") or "unknown", (process.findtext("type") or "unknown").strip()])
    return result


def idle(device: dict, processes: list[list[str]]) -> bool:
    """Strict, benchmark-only: no foreign compute context, display baseline, ~zero utilization."""
    return (device["used_mib"] < MAX_GRAPHICS_BASELINE_MIB
            and device["utilization_percent"] < MAX_BASELINE_UTILIZATION_PERCENT
            and all(len(row) == 4 and row[3] in {"G", "C+G"} and row[1]
                    for row in processes if row[0] == device["uuid"]))


def has_headroom(device: dict, required_free_mib: int) -> bool:
    """Default admission: valid counters and at least `required_free_mib` free, whoever else is resident."""
    return (type(required_free_mib) is int and required_free_mib > 0
            and counters_valid(device) and device["free_mib"] >= required_free_mib)


def counters_valid(device: dict) -> bool:
    """Primary memory counters are integers within the card's total; no activity condition."""
    total, used, free = (device.get(key) for key in ("total_mib", "used_mib", "free_mib"))
    return (all(type(value) is int for value in (total, used, free))
            and total > 0 and 0 <= used <= total and 0 <= free <= total)


def process_identities(uuid: str, processes: list[list[str]]) -> list[tuple[str, str, str]]:
    return sorted((row[0], row[1], row[3]) for row in processes if row[0] == uuid)


def sample_baseline(device: dict, out: Path, phase: str,
                    baseline: dict | None = None, idle_check=None, identity_key=None,
                    permit_bounded_growth: bool = False,
                    require_stable_contexts: bool = True, inventory_reader=None) -> tuple[dict, dict]:
    """Admit one physical UUID only after bounded, stable driver observations."""
    samples = []
    started = time.monotonic()
    try:
        for index in range(BASELINE_SAMPLES):
            if index:
                time.sleep(BASELINE_INTERVAL_SECONDS)
            devices, processes = (inventory_reader or inventory)(out, f"{phase}-sample-{index + 1}", started + MAX_SAMPLING_SECONDS)
            matches = [row for row in devices if row["uuid"] == device["uuid"]]
            sample = dict(sample_number=index + 1, elapsed_seconds=time.monotonic() - started,
                          devices=devices, processes=processes)
            samples.append(sample)
            if len(matches) != 1 or time.monotonic() - started > MAX_SAMPLING_SECONDS:
                raise RuntimeError(f"physical GPU identity or sampling deadline changed during {phase}")
            current = matches[0]
            identities = (identity_key or process_identities)(device["uuid"], processes)
            if not (idle_check or idle)(current, processes):
                raise RuntimeError(f"physical GPU occupied during {phase}")
            if index == 0 and baseline is None:
                baseline = dict(uuid=device["uuid"], reserved_baseline_mib=current["used_mib"],
                                process_identities=identities)
            if ((require_stable_contexts and identities != baseline["process_identities"])
                    or (not permit_bounded_growth and
                        (current["used_mib"] > baseline["reserved_baseline_mib"] or
                         (index and current["used_mib"] > samples[index - 1]["selected_used_mib"])))):
                raise RuntimeError(f"physical GPU baseline changed during {phase}")
            if permit_bounded_growth:
                baseline["reserved_baseline_mib"] = max(baseline["reserved_baseline_mib"], current["used_mib"])
            sample["selected_used_mib"] = current["used_mib"]
        return current, baseline
    finally:
        (out / f"{phase}-samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8", newline="\n")


def claim_device(out: Path, required_free_mib: int | None = None, *, strict: bool = False):
    """Lock one physical UUID and admit it after five stable samples taken under the lock.

    Default: `required_free_mib` free in every sample. `strict=True` (timing benchmarks only)
    keeps the original idle predicate and nonrising baseline.
    """
    if not strict and (type(required_free_mib) is not int or required_free_mib <= 0):
        raise ValueError("headroom admission requires a positive required_free_mib")
    import msvcrt
    admit = idle if strict else (lambda device, _rows: has_headroom(device, required_free_mib))
    devices, processes = inventory(out, "admission")
    (out / "admission-inventory.json").write_text(json.dumps(dict(devices=devices, processes=processes), indent=2), encoding="utf-8", newline="\n")
    # Public is host-global for both listeners, unlike runner.temp/workspaces.
    lock_dir = Path(os.environ["PUBLIC"]) / "SceneWorks" / "gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    refusals = []
    for device in sorted(devices, key=lambda d: d["index"], reverse=True):
        if not admit(device, processes):
            refusals.append(f'GPU {device["index"]}: ' + ("not idle" if strict else
                            f'{device["free_mib"]} MiB free < {required_free_mib} MiB required'))
            continue
        handle = (lock_dir / f'{device["uuid"]}.lock').open("a+b")
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            refusals.append(f'GPU {device["index"]}: UUID lock held by another CI job')
            continue
        try:
            checked, baseline = (sample_baseline(device, out, "claimed-device") if strict else
                                 sample_headroom(device, out, "claimed-device", required_free_mib))
            return checked, handle, baseline
        except RuntimeError as error:
            handle.close()
            refusals.append(f'GPU {device["index"]}: {error}')
        except Exception:
            handle.close()
            raise
    raise RuntimeError(("no unclaimed idle physical CUDA device; refusing benchmark" if strict else
                        "no unclaimed physical CUDA device with stable headroom") + ": " + "; ".join(refusals))


def stable_window(admit, samples: int = BASELINE_SAMPLES):
    """Wrap a per-sample check with the quiet-card rule over one sampling window.

    Returns (check, state); state["reason"] names the first failed condition. A utilization
    the driver reports as N/A is not judged. Foreign contexts never matter.
    """
    observed: list[dict] = []
    state = {"reason": None}

    def check(device: dict, rows) -> bool:
        if not admit(device, rows):
            state["reason"] = (f'{device.get("free_mib")} MiB free is below the requirement'
                               if counters_valid(device) else "invalid memory counters")
            return False
        observed.append(device)
        used = [item["used_mib"] for item in observed]
        utilization = [item.get("utilization_percent") for item in observed
                       if item.get("utilization_percent") is not None]
        if any(type(value) is not int for value in utilization):
            state["reason"] = "invalid utilization counter"
        elif max(used) - min(used) > STABLE_MAX_USED_SPREAD_MIB:
            state["reason"] = f"memory.used moved {max(used) - min(used)} MiB (> {STABLE_MAX_USED_SPREAD_MIB})"
        elif utilization and max(utilization) > STABLE_MAX_UTILIZATION_PERCENT:
            state["reason"] = f"utilization {max(utilization)}% (> {STABLE_MAX_UTILIZATION_PERCENT}%)"
        elif utilization and max(utilization) - min(utilization) > STABLE_MAX_UTILIZATION_SPREAD:
            state["reason"] = (f"utilization moved {max(utilization) - min(utilization)} points "
                               f"(> {STABLE_MAX_UTILIZATION_SPREAD})")
        elif len(observed) >= samples and used[-1] - used[0] > STABLE_MAX_USED_GROWTH_MIB:
            state["reason"] = f"memory.used grew {used[-1] - used[0]} MiB (> {STABLE_MAX_USED_GROWTH_MIB})"
        else:
            return True
        return False

    return check, state


def sample_headroom(device: dict, out: Path, phase: str, required_free_mib: int,
                    baseline: dict | None = None, *, stable: bool = True) -> tuple[dict, dict]:
    """Five samples that each keep `required_free_mib` free; with `stable`, on a quiet card.

    Foreign display/compute contexts and loaded models never refuse admission.
    """
    headroom = lambda current, _rows: has_headroom(current, required_free_mib)
    check, state = stable_window(headroom) if stable else (headroom, {"reason": None})
    try:
        return sample_baseline(device, out, phase, baseline, idle_check=check,
                               permit_bounded_growth=True, require_stable_contexts=False)
    except RuntimeError as error:
        if state["reason"] is None:
            raise
        raise RuntimeError(f"{error}: {state['reason']}") from error


def verify_idle(device: dict, out: Path, phase: str, baseline: dict, idle_check=None,
                identity_key=None, permit_bounded_growth: bool = False,
                require_stable_contexts: bool = True, inventory_reader=None):
    sample_baseline(device, out, phase, baseline, idle_check=idle_check,
                    identity_key=identity_key, permit_bounded_growth=permit_bounded_growth,
                    require_stable_contexts=require_stable_contexts, inventory_reader=inventory_reader)


def verify_headroom(device: dict, out: Path, phase: str, baseline: dict, required_free_mib: int):
    """Re-check headroom inside an admitted session; quietness was settled at admission."""
    sample_headroom(device, out, phase, required_free_mib, baseline, stable=False)


def persist_receipt(out: Path, receipt: dict):
    temporary = out / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
    temporary.replace(out / "receipt.json")


def execute_owned(name, args, out, deadline, job_type):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("CUDA validation one-hour session budget exhausted")
    started = time.monotonic()
    with (out / f"{name}.stdout.log").open("wb") as stdout, (out / f"{name}.stderr.log").open("wb") as stderr:
        job = job_type()
        child = None
        try:
            child = subprocess.Popen(args, stdout=stdout, stderr=stderr,
                                     env=os.environ.copy(), creationflags=0x00000004)  # CREATE_SUSPENDED
            try:
                job.assign_and_resume(child)
            except Exception:
                child.kill()
                child.wait(timeout=10)
                raise
            try:
                code = child.wait(timeout=remaining)
                return dict(name=name, argv=args, returncode=code, timeout=False, seconds=time.monotonic()-started)
            except subprocess.TimeoutExpired:
                job.terminate_and_reap(child, 10)
                return dict(name=name, argv=args, returncode=None, termination_returncode=child.returncode, timeout=True, seconds=time.monotonic()-started)
        finally:
            job.close()


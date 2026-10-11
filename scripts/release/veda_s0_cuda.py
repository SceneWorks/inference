"""Bounded, manual S0 CUDA admission/build/run. No asset downloads or production changes.

Locks coordinate this S0 harness on the shared physical Windows host, not unrelated
workflows. Occupancy is rechecked immediately before each child, and exclusive GPU
scheduling still requires coordinator admission of the existing shared runner pool.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

MAX_SECONDS = 3600
ALLOWED_RUNNERS = {"cuda-windows", "cuda-windows-2"}


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, timeout=30).strip()


def inventory() -> tuple[list[dict], list[list[str]]]:
    rows = command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    devices = []
    for row in csv.reader(rows.splitlines()):
        i, uuid, name, driver, total, used, utilization = [v.strip() for v in row]
        devices.append(dict(index=int(i), uuid=uuid, name=name, driver=driver, total_mib=int(total), used_mib=int(used), utilization_percent=int(utilization)))
    rows = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
    processes = [[v.strip() for v in row] for row in csv.reader(rows.splitlines()) if row]
    return devices, processes


def idle(device: dict, processes: list[list[str]]) -> bool:
    return device["used_mib"] < 1024 and device["utilization_percent"] < 5 and not any(row[0] == device["uuid"] for row in processes)


def claim_device(out: Path):
    import msvcrt
    devices, processes = inventory()
    (out / "admission-inventory.json").write_text(json.dumps(dict(devices=devices, processes=processes), indent=2))
    # Public is host-global for both listeners, unlike runner.temp/workspaces.
    lock_dir = Path(os.environ["PUBLIC"]) / "SceneWorks" / "gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for device in sorted(devices, key=lambda d: d["index"], reverse=True):
        if not idle(device, processes):
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
            continue
        current, current_processes = inventory()
        checked = next(d for d in current if d["uuid"] == device["uuid"])
        if idle(checked, current_processes):
            return checked, handle
        handle.close()
    raise RuntimeError("no unclaimed idle physical CUDA device; refusing benchmark")


def verify_idle(device: dict, out: Path, phase: str):
    devices, processes = inventory()
    (out / f"{phase}-inventory.json").write_text(json.dumps(dict(devices=devices, processes=processes), indent=2))
    current = next(d for d in devices if d["uuid"] == device["uuid"])
    if not idle(current, processes):
        raise RuntimeError(f"selected physical device became occupied before {phase}")


def persist_receipt(out: Path, receipt: dict):
    temporary = out / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(out / "receipt.json")


def main() -> int:
    out = Path(os.environ["S0_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(instrument="veda_s0_cuda", complete=False, status="starting", host=socket.gethostname(), runner=os.environ.get("RUNNER_NAME"), source_commit=command(["git", "rev-parse", "HEAD"]), results=[], max_session_seconds=MAX_SECONDS,
                   methodology="Random tile plan outside timing; device indices/batched gather/budgeted SDPA/concatenation/synchronization timed. Actual 124-frame geometries and globally dense text/audio. Q4 real DiT uses cached weights and fixed synthetic context/latents. No image quality or learned Veda claim.")
    persist_receipt(out, receipt)
    lock = None
    deadline = time.monotonic() + MAX_SECONDS
    try:
        if os.name != "nt" or receipt["runner"] not in ALLOWED_RUNNERS:
            raise RuntimeError("S0 requires an explicitly admitted existing Windows CUDA listener")
        if receipt["source_commit"] != os.environ["GITHUB_SHA"] or command(["git", "status", "--porcelain"]):
            raise RuntimeError("checkout identity/cleanliness mismatch")
        device, lock = claim_device(out)
        receipt["device"] = device
        receipt["device_lock_directory"] = str(Path(os.environ["PUBLIC"]) / "SceneWorks" / "gpu-locks")
        os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        receipt["cuda_visible_devices"] = device["uuid"]
        receipt["status"] = "device_admitted"
        persist_receipt(out, receipt)
        # Existing ownership helper creates children suspended and assigns kill-on-close Job.
        # It owns descendants on timeout/error and refuses incompatible enclosing CI Jobs.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from media_campaign_windows import WindowsJob
        build = ["cmd", "/d", "/s", "/c", 'call "%VCVARS%" && cargo build --locked --release -j4 -p candle-gen-minimax-h3 --features cuda,veda-s0-profile --example veda_s0_cuda']
        result = execute_owned("build", build, out, deadline, WindowsJob)
        receipt["results"].append(result)
        receipt["status"] = "build_child_completed"
        persist_receipt(out, receipt)
        if result["returncode"] != 0:
            receipt["status"] = "build_failed"
            return result["returncode"] or 124
        binary = Path(os.environ["CARGO_TARGET_DIR"]) / "release" / "examples" / "veda_s0_cuda.exe"
        with binary.open("rb") as stream:
            receipt["binary_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt["rustc"] = command(["rustc", "--version"])
        receipt["status"] = "build_complete"
        persist_receipt(out, receipt)
        common = [str(binary), "--case", "all", "--repeat", "1"]
        for name, args in [("random", common), ("real_dit", common + ["--real-dit", os.environ.get("S0_Q4_DIT_DIR", "")])]:
            if name == "real_dit" and not Path(os.environ.get("S0_Q4_DIT_DIR", "") or "__missing_s0_asset__").joinpath("config.json").is_file():
                receipt["status"] = "real_dit_assets_unavailable"
                receipt["missing_asset"] = os.environ.get("S0_Q4_DIT_DIR", "")
                return 3
            verify_idle(device, out, name)
            result = execute_owned(name, args, out, deadline, WindowsJob)
            receipt["results"].append(result)
            receipt["status"] = f"{name}_child_completed"
            persist_receipt(out, receipt)
            if result["returncode"] != 0:
                receipt["status"] = f"{name}_failed"
                return result["returncode"] or 124
            rows = [json.loads(line) for line in (out / f"{name}.stdout.log").read_text().splitlines() if line.strip()]
            if len(rows) != 2 or [row["tokens"] for row in rows] != [7586, 38222]:
                raise RuntimeError(f"{name} did not produce both required geometry receipts")
            receipt[f"{name}_receipts"] = rows
            receipt["status"] = f"{name}_complete"
            persist_receipt(out, receipt)
        receipt["status"] = "complete"
        receipt["complete"] = True
        return 0
    except Exception as exc:
        receipt["status"] = "refused_or_failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        print(receipt["error"], file=sys.stderr)
        return 2
    finally:
        persist_receipt(out, receipt)
        if lock is not None:
            lock.close()


def execute_owned(name, args, out, deadline, job_type):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("S0 one-hour session budget exhausted")
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



if __name__ == "__main__":
    sys.exit(main())

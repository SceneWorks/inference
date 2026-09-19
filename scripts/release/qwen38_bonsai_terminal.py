#!/usr/bin/env python3
"""Run and validate the sealed SC-23942 native comparison campaign.

The runner launches an already-built Rust test executable directly so RSS and per-process GPU
samples name the model process rather than Cargo. It uses the release snapshot verifier before and
after execution, retains raw provider output and logs, and never overwrites an evidence directory.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from scripts.release.verify_model_snapshot import load_model, snapshot_inventory, verify_snapshot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_model_snapshot import load_model, snapshot_inventory, verify_snapshot


SCHEMA_VERSION = 1
SUITE = "qwen38-bonsai-native-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def source_identity(expected_sha: str, allow_dirty: bool) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    if head != expected_sha:
        raise ValueError(f"runtime SHA {expected_sha} does not equal checkout HEAD {head}")
    if dirty and not allow_dirty:
        raise ValueError("runtime checkout is dirty; terminal evidence requires a clean source tree")
    return {"head_sha": head, "clean_tree": not bool(dirty), "dirty_paths": dirty.splitlines()}


def physical_memory() -> tuple[int | None, int | None, str | None]:
    """Return total, currently available bytes, and an unavailable reason."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys), int(status.avail_phys), None
        return None, None, "GlobalMemoryStatusEx failed"
    if sys.platform == "darwin":
        try:
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
            page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True))
            output = subprocess.check_output(["vm_stat"], text=True)
            pages: dict[str, int] = {}
            for line in output.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    pages[key] = int(value.strip().rstrip("."))
            available = page_size * sum(
                pages.get(key, 0)
                for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            )
            return total, available, None
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return None, None, f"macOS memory query failed: {error}"
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
        return values["MemTotal"], values["MemAvailable"], None
    except (OSError, KeyError, ValueError) as error:
        return None, None, f"physical memory query failed: {error}"


def rss_bytes(pid: int) -> int | None:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]
        query = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(query, False, pid)
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.working_set_size)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        return None
    if sys.platform.startswith("linux"):
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
    try:
        value = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return int(value) * 1024 if value else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def nvidia_sample(pid: int) -> tuple[int | None, str | None]:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None, "nvidia-smi unavailable"
    try:
        output = subprocess.check_output(
            [tool, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"nvidia-smi process query failed: {error}"
    total = 0
    found = False
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[0] != str(pid):
            continue
        if not fields[1].isdigit():
            return None, "per-process GPU memory unavailable (WDDM or unsupported driver)"
        total += int(fields[1]) * 1024 * 1024
        found = True
    return (total, None) if found else (None, "process not reported by nvidia-smi")


def nvidia_hardware() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return [], [], "nvidia-smi unavailable"
    try:
        gpu_output = subprocess.check_output(
            [
                tool,
                "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        process_output = subprocess.check_output(
            [
                tool,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], [], f"nvidia-smi hardware query failed: {error}"
    gpus = []
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 7:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "total_bytes": int(fields[3]) * 1024 * 1024,
                    "free_bytes": int(fields[4]) * 1024 * 1024,
                    "used_bytes": int(fields[5]) * 1024 * 1024,
                    "driver_version": fields[6],
                }
            )
    processes = []
    for line in process_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        processes.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]) if fields[1].isdigit() else None,
                "process_name": fields[2],
                "used_bytes": int(fields[3]) * 1024 * 1024 if fields[3].isdigit() else None,
                "used_memory_unavailable_reason": None
                if fields[3].isdigit()
                else "driver did not expose per-process memory",
            }
        )
    return gpus, processes, None


def safetensor_sizes(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(length))
    language = vision = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid safetensors offsets for {name} in {path}")
        size = int(offsets[1]) - int(offsets[0])
        if "vision_tower." in name or ".visual." in name or name.startswith("visual."):
            vision += size
        else:
            language += size
    return language, vision


def artifact_sizes(model_path: Path, projector_path: Path | None) -> dict[str, int]:
    language = vision = auxiliary = 0
    if model_path.is_file():
        language = model_path.stat().st_size
    else:
        safetensors = sorted(model_path.glob("*.safetensors"))
        for path in safetensors:
            lang, vis = safetensor_sizes(path)
            language += lang
            vision += vis
        weight_files = {path.resolve() for path in safetensors}
        auxiliary = sum(
            path.stat().st_size
            for path in model_path.rglob("*")
            if path.is_file() and path.resolve() not in weight_files
        )
    if projector_path is not None:
        vision += projector_path.stat().st_size
    return {
        "language_weight_bytes": language,
        "vision_weight_bytes": vision,
        "auxiliary_bytes": auxiliary,
    }


def pinned_admission_sizes(
    model: dict[str, Any], language_variant: str | None = None, vision_variant: str | None = None
) -> dict[str, int]:
    sizes = {}
    for field, variants_field, variant, output in (
        (
            "admission_language_weight_bytes",
            "admission_language_variants",
            language_variant,
            "language_weight_bytes",
        ),
        (
            "admission_vision_weight_bytes",
            "admission_vision_variants",
            vision_variant,
            "vision_weight_bytes",
        ),
    ):
        variants = model.get(variants_field)
        if variants is not None:
            if not isinstance(variants, dict) or variant not in variants:
                raise ValueError(f"{model['key']} requires a pinned {variants_field} selection")
            value = variants[variant]
        else:
            if variant is not None:
                raise ValueError(f"{model['key']} does not define {variants_field}")
            value = model.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{model['key']} lacks positive pinned {field}")
        sizes[output] = value
    sizes["auxiliary_bytes"] = 0
    return sizes


def selected_artifact(path: Path, snapshot: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        try:
            relative = resolved.relative_to(snapshot.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"selected artifact is outside the pinned snapshot: {resolved}") from error
        matches = [item for item in inventory["files"] if item["path"] == relative]
        if len(matches) != 1:
            raise ValueError(f"selected artifact is absent from snapshot inventory: {relative}")
        return {
            "path": str(resolved),
            "kind": "file",
            "bytes": matches[0]["size"],
            "sha256": matches[0]["sha256"],
        }
    if resolved != snapshot.resolve():
        raise ValueError(f"selected snapshot path does not equal the pinned snapshot: {resolved}")
    return {
        "path": str(resolved),
        "kind": "snapshot",
        "bytes": None,
        "sha256": inventory["inventory_sha256"],
    }


def validate_variant_paths(args: argparse.Namespace, model: dict[str, Any]) -> None:
    if model["key"] != "bonsai-gguf":
        if args.language_variant is not None or args.vision_variant is not None:
            raise ValueError(f"{model['key']} does not accept GGUF variant labels")
        return
    language_files = {
        "pq2": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        "ptq1": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    }
    vision_files = {
        "bf16": "Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
        "q8": "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
    }
    if language_files.get(args.language_variant) != args.model_path.name:
        raise ValueError("GGUF language variant label does not match selected model file")
    if args.projector_path is None or vision_files.get(args.vision_variant) != args.projector_path.name:
        raise ValueError("GGUF vision variant label does not match selected projector file")


def artifact_manifest(root: Path, names: list[str]) -> dict[str, Any]:
    files = []
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing campaign artifact {name}")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"files": files}


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime_sha = args.runtime_sha.lower()
    if len(runtime_sha) != 40 or any(c not in "0123456789abcdef" for c in runtime_sha):
        raise ValueError("runtime SHA must be a lowercase 40-character commit")
    source = source_identity(runtime_sha, args.allow_dirty)
    model = load_model(args.manifest, args.model_key)
    if model["revision"] != args.model_revision:
        raise ValueError("requested model revision does not match the pinned manifest")
    validate_variant_paths(args, model)
    verify_snapshot(model, args.snapshot)
    before = snapshot_inventory(model, args.snapshot)
    measured_sizes = artifact_sizes(args.model_path, args.projector_path)
    pinned_sizes = pinned_admission_sizes(model, args.language_variant, args.vision_variant)
    if any(measured_sizes[key] != pinned_sizes[key] for key in pinned_sizes if key != "auxiliary_bytes"):
        raise ValueError(
            f"selected weight bytes {measured_sizes} do not match pinned admission bytes {pinned_sizes}"
        )
    binary = args.binary.resolve(strict=True)
    provider_path = output / "provider.json"
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    command = [str(binary), args.test_name, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    env = os.environ.copy()
    env.update(
        {
            "BONSAI_COMPARISON_MODEL_PATH": str(args.model_path.resolve()),
            "BONSAI_COMPARISON_MODEL_ID": args.model_id,
            "BONSAI_COMPARISON_MODEL_REVISION": args.model_revision,
            "BONSAI_COMPARISON_RUNTIME_SHA": runtime_sha,
            "BONSAI_COMPARISON_OUTPUT": str(provider_path),
            "BONSAI_COMPARISON_CASES": json.dumps(args.cases.split(","), separators=(",", ":")),
        }
    )
    if args.candle_device is not None:
        env["CANDLE_LLM_DEVICE"] = args.candle_device
    if args.projector_path is not None:
        env["BONSAI_COMPARISON_PROJECTOR"] = str(args.projector_path.resolve())

    started_wall = time.time()
    started = time.monotonic()
    rss_samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    gpu_reasons: list[str] = []
    interrupted: str | None = None
    previous_handlers: dict[int, Any] = {}
    proc: subprocess.Popen[bytes] | None = None

    def interrupt(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = signal.Signals(signum).name
        if proc is not None and proc.poll() is None:
            proc.terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                rss = rss_bytes(proc.pid)
                if rss is not None:
                    rss_samples.append({"seconds": elapsed, "bytes": rss})
                gpu, reason = nvidia_sample(proc.pid)
                if gpu is not None:
                    gpu_samples.append({"seconds": elapsed, "bytes": gpu})
                elif reason:
                    gpu_reasons.append(reason)
                time.sleep(args.sample_interval)
            exit_code = proc.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    verify_snapshot(model, args.snapshot)
    after = snapshot_inventory(model, args.snapshot)
    provider = json.loads(provider_path.read_text(encoding="utf-8")) if provider_path.is_file() else None
    completed = (
        interrupted is None
        and exit_code == 0
        and provider is not None
        and provider.get("status") == "completed"
        and before.get("inventory_sha256") == after.get("inventory_sha256")
    )
    total_memory, available_memory, memory_reason = physical_memory()
    manifest = artifact_manifest(
        output, [name for name in ("provider.json", "stdout.log", "stderr.log") if (output / name).is_file()]
    )
    write_new(output / "artifact-manifest.json", manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "status": "completed" if completed else ("interrupted" if interrupted else "failed"),
        "started_unix_seconds": started_wall,
        "elapsed_seconds": time.monotonic() - started,
        "command": {
            "argv": command,
            "binary_sha256": sha256(binary),
            "candle_device": args.candle_device,
        },
        "runtime": source,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "physical_memory_bytes": total_memory,
            "available_memory_bytes_after_run": available_memory,
            "memory_unavailable_reason": memory_reason,
        },
        "model": {
            "id": args.model_id,
            "manifest_key": args.model_key,
            "revision": args.model_revision,
            "snapshot": str(args.snapshot.resolve()),
            "model_path": str(args.model_path.resolve()),
            "projector_path": str(args.projector_path.resolve()) if args.projector_path else None,
            "language_variant": args.language_variant,
            "vision_variant": args.vision_variant,
            "selected_model_artifact": selected_artifact(args.model_path, args.snapshot, before),
            "selected_projector_artifact": selected_artifact(args.projector_path, args.snapshot, before)
            if args.projector_path
            else None,
            "artifact_sizes": measured_sizes,
            "inventory_before": before,
            "inventory_after": after,
        },
        "process": {
            "exit_code": exit_code,
            "interrupted_by": interrupted,
            "rss_scope": "sampled_process_working_set_lower_bound",
            "sample_interval_seconds": args.sample_interval,
            "rss_samples": rss_samples,
            "peak_rss_bytes": max((sample["bytes"] for sample in rss_samples), default=None),
        },
        "gpu": {
            "scope": "nvidia_smi_per_process_sampled_lower_bound",
            "available": bool(gpu_samples),
            "unavailable_reason": None if gpu_samples else (gpu_reasons[-1] if gpu_reasons else "no samples"),
            "samples": gpu_samples,
            "peak_bytes": max((sample["bytes"] for sample in gpu_samples), default=None),
        },
        "provider_evidence": "provider.json" if provider_path.is_file() else None,
        "artifact_manifest_sha256": sha256(output / "artifact-manifest.json"),
    }
    write_new(output / "receipt.json", receipt)
    write_new(
        output / "seal.json",
        {
            "schema_version": 1,
            "receipt_sha256": sha256(output / "receipt.json"),
            "artifact_manifest_sha256": receipt["artifact_manifest_sha256"],
        },
    )
    return 0 if completed else 1


def validate_receipt(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = root / "receipt.json"
    seal_path = root / "seal.json"
    manifest_path = root / "artifact-manifest.json"
    for path in (receipt_path, seal_path, manifest_path):
        if not path.is_file():
            raise ValueError(f"missing evidence file {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("suite") != SUITE:
        raise ValueError(f"unsupported receipt schema in {root}")
    if seal.get("receipt_sha256") != sha256(receipt_path):
        raise ValueError(f"receipt hash mismatch in {root}")
    if seal.get("artifact_manifest_sha256") != sha256(manifest_path):
        raise ValueError(f"manifest hash mismatch in {root}")
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"artifact mismatch for {path}")
    if receipt.get("status") != "completed" or receipt.get("process", {}).get("exit_code") != 0:
        raise ValueError(f"run did not complete successfully: {root}")
    if not receipt.get("runtime", {}).get("clean_tree"):
        raise ValueError(f"dirty runtime source in {root}")
    model = receipt.get("model", {})
    before = model.get("inventory_before", {}).get("inventory_sha256")
    after = model.get("inventory_after", {}).get("inventory_sha256")
    if not before or before != after:
        raise ValueError(f"snapshot changed during run: {root}")
    selected_model = model.get("selected_model_artifact", {})
    if not isinstance(selected_model.get("sha256"), str) or len(selected_model["sha256"]) != 64:
        raise ValueError(f"selected model artifact identity is missing in {root}")
    if model.get("manifest_key") == "bonsai-gguf":
        selected_projector = model.get("selected_projector_artifact") or {}
        if not isinstance(selected_projector.get("sha256"), str) or len(
            selected_projector["sha256"]
        ) != 64:
            raise ValueError(f"selected GGUF projector identity is missing in {root}")
    if receipt.get("process", {}).get("peak_rss_bytes") is None:
        raise ValueError(f"missing sampled process RSS in {root}")
    gpu = receipt.get("gpu", {})
    if not gpu.get("available") and not gpu.get("unavailable_reason"):
        raise ValueError(f"GPU counter is neither measured nor explicitly unavailable: {root}")
    provider_name = receipt.get("provider_evidence")
    if not provider_name:
        raise ValueError(f"missing provider evidence link in {root}")
    provider = json.loads((root / provider_name).read_text(encoding="utf-8"))
    if provider.get("status") != "completed":
        raise ValueError(f"provider evidence incomplete in {root}")
    if provider.get("context_admission", {}).get("evidence_complete") is not True:
        raise ValueError(f"provider lacks a genuine context admission rejection in {root}")
    memory_points = [
        provider.get("native_memory_before_load"),
        provider.get("native_memory_after_load"),
        provider.get("native_memory_after_unload"),
    ]
    if any(not isinstance(point, dict) or not point for point in memory_points):
        raise ValueError(f"provider native memory evidence is missing in {root}")
    backend = memory_points[0].get("backend")
    if backend == "mlx":
        if any(not isinstance(point.get("peak_active_bytes"), int) for point in memory_points):
            raise ValueError(f"MLX allocator peak is unavailable in {root}")
    elif backend == "candle":
        if any(
            point.get("native_allocator_counters_available") is not False
            or not point.get("memory_evidence")
            for point in memory_points
        ):
            raise ValueError(f"Candle memory gap is not explicit in {root}")
    else:
        raise ValueError(f"unknown native memory backend in {root}")
    cases = provider.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"provider evidence has no cases in {root}")
    for case in cases:
        if case.get("status") != "completed" or case.get("evidence_complete") is not True:
            raise ValueError(f"case {case.get('case_id')} has broken evidence in {root}")
        for phase in ("prefill_seconds", "decode_seconds"):
            value = case.get(phase)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"case {case.get('case_id')} lacks {phase} in {root}")
        output = case.get("output", {})
        if not isinstance(output.get("prompt_tokens"), int) or output["prompt_tokens"] <= 0:
            raise ValueError(f"case {case.get('case_id')} lacks exact prompt tokens in {root}")
    return receipt, provider


def validate(args: argparse.Namespace) -> int:
    rows = [validate_receipt(path.resolve()) for path in args.run]
    expected = args.expected_model
    ids = [receipt["model"]["id"] for receipt, _ in rows]
    if sorted(ids) != sorted(expected) or len(ids) != len(set(ids)):
        raise ValueError(f"model rows {ids!r} do not equal required rows {expected!r}")
    baseline_cases = rows[0][1]["cases"]
    identity = [(case["case_id"], case["category"], case["request"]) for case in baseline_cases]
    for receipt, provider in rows[1:]:
        actual = [(case["case_id"], case["category"], case["request"]) for case in provider["cases"]]
        if actual != identity:
            raise ValueError(f"unequal requests or budgets for {receipt['model']['id']}")
    categories: dict[str, list[dict[str, Any]]] = {}
    raw_outputs: list[dict[str, Any]] = []
    for receipt, provider in rows:
        for case in provider["cases"]:
            row = {
                "model_id": receipt["model"]["id"],
                "case_id": case["case_id"],
                "category": case["category"],
                "quality_passed": case.get("quality_passed"),
                "prompt_tokens": case["output"]["prompt_tokens"],
                "generated_tokens": case["output"]["generated_tokens"],
                "prefill_seconds": case["prefill_seconds"],
                "decode_seconds": case["decode_seconds"],
                "output": case["output"],
            }
            categories.setdefault(case["category"], []).append(row)
            raw_outputs.append(row)
    report = {
        "schema_version": 1,
        "suite": SUITE,
        "evidence_complete": True,
        "claims": {
            "vendor_benchmark_reproduction": False,
            "vendor_quality_retention": False,
            "quality_threshold_applied": False,
            "scope": "fixed native diagnostic cases only",
        },
        "models": [
            {
                "id": receipt["model"]["id"],
                "revision": receipt["model"]["revision"],
                "runtime_sha": receipt["runtime"]["head_sha"],
                "artifact_sizes": receipt["model"]["artifact_sizes"],
                "peak_rss_bytes": receipt["process"]["peak_rss_bytes"],
                "peak_gpu_bytes": receipt["gpu"]["peak_bytes"],
                "gpu_unavailable_reason": receipt["gpu"]["unavailable_reason"],
            }
            for receipt, _ in rows
        ],
        "categories": categories,
        "raw_outputs": raw_outputs,
    }
    write_new(args.output, report)
    markdown = [
        "# Qwen3.8 and Bonsai native diagnostic report",
        "",
        "This is a fixed diagnostic suite. It is not a vendor benchmark reproduction and applies no aggregate quality-retention threshold.",
        "",
        "| Model | Case | Category | Quality | Prompt tokens | Generated tokens | Prefill s | Decode s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in raw_outputs:
        markdown.append(
            f"| {row['model_id']} | {row['case_id']} | {row['category']} | "
            f"{row['quality_passed']} | {row['prompt_tokens']} | {row['generated_tokens']} | "
            f"{row['prefill_seconds']:.6f} | {row['decode_seconds']:.6f} |"
        )
    markdown.extend(["", "## Raw outputs", "", "```json", json.dumps(raw_outputs, indent=2, sort_keys=True), "```", ""])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(markdown))
    return 0


def matrix_status(args: argparse.Namespace) -> int:
    spec = json.loads(args.matrix.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("suite") != SUITE:
        raise ValueError("unsupported comparison matrix schema")
    cells = spec.get("cells")
    groups = spec.get("groups")
    if not isinstance(cells, list) or not cells:
        raise ValueError("comparison matrix has no required cells")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("comparison matrix has no workload groups")
    ids = [cell.get("id") for cell in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("comparison matrix cell ids must be nonempty and unique")
    roots = [root.resolve() for root in args.root]
    rows = []
    complete_runs: dict[str, list[Path]] = {group: [] for group in groups}
    for cell in cells:
        cell_id = cell["id"]
        preflights = [root / f"{cell_id}-preflight.json" for root in roots]
        run_roots = [root / cell_id for root in roots]
        preflight_path = next((path for path in preflights if path.is_file()), None)
        run_root = next((path for path in run_roots if path.is_dir()), None)
        row = {**cell, "status": "pending", "reason": None}
        group = cell.get("group")
        if group not in groups:
            raise ValueError(f"comparison matrix cell {cell_id} names unknown group {group!r}")
        if preflight_path is None:
            row.update(status="incomplete", reason="missing preflight evidence")
        else:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            row["preflight"] = preflight
            if preflight.get("model_key") != cell.get("model_key") or preflight.get(
                "load_profile"
            ) != cell.get("load_profile") or preflight.get("language_variant") != cell.get(
                "language_variant"
            ) or preflight.get("vision_variant") != cell.get("vision_variant"):
                row.update(status="incomplete", reason="preflight identity does not match matrix")
            elif preflight.get("admitted") is not True:
                row.update(status="not_admitted", reason="live capacity preflight rejected this cell")
            elif run_root is None:
                row.update(status="incomplete", reason="admitted cell has no run evidence")
            else:
                try:
                    receipt, provider = validate_receipt(run_root)
                    if receipt.get("model", {}).get("id") != cell_id:
                        raise ValueError("run model id does not match matrix cell")
                    model = receipt.get("model", {})
                    if model.get("manifest_key") != cell.get("model_key"):
                        raise ValueError("run model key does not match matrix cell")
                    if model.get("language_variant") != cell.get("language_variant") or model.get(
                        "vision_variant"
                    ) != cell.get("vision_variant"):
                        raise ValueError("run format variants do not match matrix cell")
                    if [case.get("case_id") for case in provider["cases"]] != groups[group].get(
                        "case_ids"
                    ):
                        raise ValueError("run cases do not match matrix workload group")
                    row.update(status="completed", reason=None)
                    complete_runs[group].append(run_root)
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    row.update(status="incomplete", reason=str(error))
        rows.append(row)
    complete = all(row["status"] == "completed" for row in rows)
    comparison = {}
    if complete:
        for group, runs in complete_runs.items():
            group_ids = [cell["id"] for cell in cells if cell["group"] == group]
            with tempfile.TemporaryDirectory() as temporary:
                temp = Path(temporary)
                validate(
                    argparse.Namespace(
                        run=runs,
                        expected_model=group_ids,
                        output=temp / "comparison.json",
                        markdown=temp / "comparison.md",
                    )
                )
                comparison[group] = json.loads(
                    (temp / "comparison.json").read_text(encoding="utf-8")
                )
    report = {
        "schema_version": 1,
        "suite": SUITE,
        "evidence_complete": complete,
        "required_cells": rows,
        "comparison": comparison,
        "claims": {
            "vendor_benchmark_reproduction": False,
            "vendor_quality_retention": False,
            "quality_threshold_applied": False,
        },
    }
    write_new(args.output, report)
    markdown = [
        "# Qwen3.8 and Bonsai native matrix status",
        "",
        f"Evidence complete: **{complete}**",
        "",
        "| Cell | Backend | Device | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['id']} | {row['backend']} | {row['device']} | {row['status']} | {row['reason'] or ''} |"
        )
    for group, result in comparison.items():
        markdown.extend(
            [
                "",
                f"## {group} raw outputs",
                "",
                "```json",
                json.dumps(result["raw_outputs"], indent=2, sort_keys=True),
                "```",
            ]
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(markdown) + "\n")
    return 0 if complete else 1


def resolve_binary(args: argparse.Namespace) -> int:
    candidates = []
    for line in sys.stdin:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = item.get("target", {})
        if (
            item.get("reason") == "compiler-artifact"
            and target.get("name") == args.target
            and "test" in target.get("kind", [])
            and item.get("executable")
        ):
            candidates.append(item["executable"])
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(f"expected exactly one test binary for {args.target}, got {unique!r}")
    print(unique[0])
    return 0


def preflight(args: argparse.Namespace) -> int:
    if args.reserve_bytes < 0:
        raise ValueError("admission reserve bytes must be nonnegative")
    model = load_model(args.manifest, args.model_key)
    sizes = pinned_admission_sizes(model, args.language_variant, args.vision_variant)
    weight_bytes = sizes["language_weight_bytes"] + sizes["vision_weight_bytes"]
    policies = {
        "mlx-unified": {
            "host_weight_copies": 1,
            "gpu_weight_copies": None,
            "basis": "MLX safetensors pread into one Metal shared buffer; CPU and GPU share residency",
        },
        "candle-packed-cuda": {
            "host_weight_copies": 1,
            "gpu_weight_copies": 1,
            "basis": "compact host tensor staging plus compact CUDA-resident Prism weights",
        },
        "candle-packed-cpu": {
            "host_weight_copies": 1,
            "gpu_weight_copies": None,
            "basis": "compact host-resident Prism weights with bounded row decode",
        },
        "candle-dense-cpu": {
            "host_weight_copies": 3,
            "gpu_weight_copies": None,
            "basis": "audited peak: one BF16 source copy plus one F32 constructed copy",
        },
        "candle-dense-cuda": {
            "host_weight_copies": 1,
            "gpu_weight_copies": 1,
            "basis": "host source staging plus dtype-preserving CUDA weights",
        },
    }
    policy = policies[args.load_profile]
    host_required = weight_bytes * policy["host_weight_copies"] + args.reserve_bytes
    gpu_copies = policy["gpu_weight_copies"]
    gpu_required = weight_bytes * gpu_copies + args.reserve_bytes if gpu_copies else None
    total, available, reason = physical_memory()
    gpus, processes, gpu_reason = nvidia_hardware()
    gpu_available = max((gpu["free_bytes"] for gpu in gpus), default=None)
    host_admitted = available is not None and available >= host_required
    gpu_admitted = gpu_required is None or (
        gpu_available is not None and gpu_available >= gpu_required
    )
    record = {
        "schema_version": 1,
        "model_key": args.model_key,
        "model_revision": model["revision"],
        "language_variant": args.language_variant,
        "vision_variant": args.vision_variant,
        "artifact_sizes": sizes,
        "load_profile": args.load_profile,
        "admission_basis": policy["basis"],
        "host_admission_formula": "exact_pinned_weight_bytes * host_weight_copies + reserve_bytes",
        "gpu_admission_formula": None
        if gpu_copies is None
        else "exact_pinned_weight_bytes * gpu_weight_copies + reserve_bytes",
        "is_measured_runtime_residency": False,
        "host_weight_copies": policy["host_weight_copies"],
        "gpu_weight_copies": gpu_copies,
        "reserve_bytes": args.reserve_bytes,
        "host_required_available_bytes": host_required,
        "gpu_required_available_bytes": gpu_required,
        "physical_memory_bytes": total,
        "available_memory_bytes": available,
        "unavailable_reason": reason,
        "gpus": gpus,
        "compute_processes": processes,
        "gpu_unavailable_reason": gpu_reason,
        "gpu_available_bytes": gpu_available,
        "host_admitted": host_admitted,
        "gpu_admitted": gpu_admitted,
        "admitted": host_admitted and gpu_admitted,
    }
    write_new(args.output, record)
    return 0 if record["admitted"] else 1


def hardware(args: argparse.Namespace) -> int:
    total, available, memory_reason = physical_memory()
    gpus, processes, gpu_reason = nvidia_hardware()
    write_new(
        args.output,
        {
            "schema_version": 1,
            "captured_unix_seconds": time.time(),
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "physical_memory_bytes": total,
                "available_memory_bytes": available,
                "memory_unavailable_reason": memory_reason,
            },
            "gpus": gpus,
            "compute_processes": processes,
            "gpu_unavailable_reason": gpu_reason,
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    sub = out.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--binary", type=Path, required=True)
    run_p.add_argument("--test-name", required=True)
    run_p.add_argument("--model-id", required=True)
    run_p.add_argument("--model-key", required=True)
    run_p.add_argument("--model-revision", required=True)
    run_p.add_argument("--snapshot", type=Path, required=True)
    run_p.add_argument("--model-path", type=Path, required=True)
    run_p.add_argument("--projector-path", type=Path)
    run_p.add_argument("--language-variant")
    run_p.add_argument("--vision-variant")
    run_p.add_argument("--runtime-sha", required=True)
    run_p.add_argument("--cases", required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    run_p.add_argument("--sample-interval", type=float, default=0.1)
    run_p.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    run_p.add_argument("--candle-device", choices=("auto", "cpu"))
    run_p.set_defaults(func=run)
    val = sub.add_parser("validate")
    val.add_argument("--run", type=Path, action="append", required=True)
    val.add_argument("--expected-model", action="append", required=True)
    val.add_argument("--output", type=Path, required=True)
    val.add_argument("--markdown", type=Path, required=True)
    val.set_defaults(func=validate)
    matrix = sub.add_parser("matrix-status")
    matrix.add_argument("--root", type=Path, action="append", required=True)
    matrix.add_argument(
        "--matrix", type=Path, default=Path("release/qwen38-bonsai-matrix.json")
    )
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--markdown", type=Path, required=True)
    matrix.set_defaults(func=matrix_status)
    resolve = sub.add_parser("resolve-binary")
    resolve.add_argument("--target", required=True)
    resolve.set_defaults(func=resolve_binary)
    admit = sub.add_parser("preflight")
    admit.add_argument("--model-key", required=True)
    admit.add_argument("--language-variant")
    admit.add_argument("--vision-variant")
    admit.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    admit.add_argument(
        "--load-profile",
        choices=(
            "mlx-unified",
            "candle-packed-cuda",
            "candle-packed-cpu",
            "candle-dense-cpu",
            "candle-dense-cuda",
        ),
        required=True,
    )
    admit.add_argument("--reserve-bytes", type=int, default=4 * 1024**3)
    admit.add_argument("--output", type=Path, required=True)
    admit.set_defaults(func=preflight)
    hw = sub.add_parser("hardware")
    hw.add_argument("--output", type=Path, required=True)
    hw.set_defaults(func=hardware)
    return out


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"qwen38-bonsai-terminal: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

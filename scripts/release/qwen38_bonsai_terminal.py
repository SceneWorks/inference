#!/usr/bin/env python3
"""Run and validate the sealed SC-23942 native comparison campaign.

The runner launches an already-built Rust test executable directly so RSS and per-process GPU
samples name the model process rather than Cargo. It uses the release snapshot verifier before and
after execution, retains raw provider output and logs, and never overwrites an evidence directory.
"""

from __future__ import annotations

import argparse
import ctypes
import getpass
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


try:
    from scripts.release import qwen38_bonsai_assets as assets
except ImportError:
    import qwen38_bonsai_assets as assets


SCHEMA_VERSION = 1
SUITE = "qwen38-bonsai-native-v1"
CUDA_DEVICE_INDEX = 0
FULL_ACCEPTANCE_CONTRACT = "qwen38-bonsai-full-v1"
FORMAT_ACCEPTANCE_CASES = ["image", "video_forward", "video_reverse"]
QWEN38_ACCEPTANCE_CASES = [
    "reasoning_low",
    "reasoning_medium",
    "reasoning_xhigh",
    "preserve_thinking",
    "tool_roundtrip",
    "json_thinking",
    "mtp_greedy",
    "mtp_json",
    "mtp_image",
    "mtp_video",
]
BONSAI_ACCEPTANCE_CASES = [
    "reasoning_low",
    "reasoning_medium",
    "reasoning_xhigh",
    "preserve_thinking",
    "tool_roundtrip",
    "json_thinking",
]
LOAD_PROFILES: dict[str, dict[str, Any]] = {
    "mlx-unified": {
        "backend": "mlx",
        "device": "unified",
        "command_device": None,
        "host_weight_copies": 1,
        "gpu_weight_copies": None,
        "basis": "MLX pinned header-derived source, conversion and staging upper bounds; CPU and GPU share residency",
    },
    "candle-packed-cuda": {
        "backend": "candle",
        "device": "cuda",
        "command_device": "auto",
        "host_weight_copies": 1,
        "gpu_weight_copies": 1,
        "basis": "compact host tensor staging plus compact CUDA-resident Prism weights",
    },
    "candle-packed-cpu": {
        "backend": "candle",
        "device": "cpu",
        "command_device": "cpu",
        "host_weight_copies": 1,
        "gpu_weight_copies": None,
        "basis": "compact host-resident Prism weights with bounded row decode",
    },
    "candle-dense-cpu": {
        "backend": "candle",
        "device": "cpu",
        "command_device": "cpu",
        "host_weight_copies": 3,
        "gpu_weight_copies": None,
        "basis": "audited peak: one BF16 source copy plus one F32 constructed copy",
    },
    "candle-dense-cuda": {
        "backend": "candle",
        "device": "cuda",
        "command_device": "auto",
        "host_weight_copies": 1,
        "gpu_weight_copies": 1,
        "basis": "host source staging plus dtype-preserving CUDA weights",
    },
}


def checked_sha(value: str, label: str) -> str:
    value = value.lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 40-character commit")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without resolving HF snapshot symlinks into extensionless blobs."""
    return Path(os.path.abspath(path))


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
        total = None
        try:
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
            page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True))
            if total <= 0 or page_size <= 0:
                raise ValueError("nonpositive physical memory or page size")
            output = subprocess.check_output(["vm_stat"], text=True)
            # vm_stat's first line also has a colon, but its value is a descriptive header.
            # Only these counters form our conservative reclaimable-memory estimate. Purgeable
            # pages overlap active/inactive queues and must not be added a second time.
            required = {"Pages free", "Pages inactive", "Pages speculative"}
            pages: dict[str, int] = {}
            for line in output.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                if key not in required:
                    continue
                value = value.strip().removesuffix(".")
                if key in pages or not value.isascii() or not value.isdecimal():
                    raise ValueError(f"invalid or duplicate vm_stat counter {key}")
                pages[key] = int(value)
            if pages.keys() != required:
                raise ValueError("required vm_stat memory counters are missing")
            available = page_size * sum(pages.values())
            if available > total:
                raise ValueError("vm_stat available estimate exceeds physical memory")
            return total, available, None
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return total, None, f"macOS memory query failed: {error}"
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


def selected_gpu_state(
    gpus: list[dict[str, Any]], processes: list[dict[str, Any]], gpu_index: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    matches = [gpu for gpu in gpus if gpu.get("index") == gpu_index]
    if len(matches) != 1:
        return None, []
    selected = matches[0]
    uuid = selected.get("uuid")
    tenants = [process for process in processes if process.get("gpu_uuid") == uuid]
    return selected, tenants


def reservation_token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_reservation(path: Path, token: str, gpu_index: int) -> dict[str, Any]:
    reservation = json.loads(path.read_text(encoding="utf-8"))
    if reservation.get("schema_version") != 1:
        raise ValueError("unsupported GPU reservation schema")
    if reservation.get("token_sha256") != reservation_token_sha256(token):
        raise ValueError("GPU reservation token does not match campaign owner")
    if reservation.get("gpu_index") != gpu_index:
        raise ValueError("GPU reservation selects a different device")
    return reservation


def current_cuda_admission(
    *,
    gpu_index: int,
    required_bytes: int,
    expected_uuid: str | None = None,
) -> dict[str, Any]:
    gpus, processes, unavailable_reason = nvidia_hardware()
    selected, tenants = selected_gpu_state(gpus, processes, gpu_index)
    selected_uuid = selected.get("uuid") if selected else None
    available = selected.get("free_bytes") if selected else None
    admitted = (
        selected is not None
        and isinstance(available, int)
        and available >= required_bytes
        and not tenants
        and (expected_uuid is None or selected_uuid == expected_uuid)
    )
    return {
        "gpu_index": gpu_index,
        "selected_gpu": selected,
        "selected_gpu_uuid": selected_uuid,
        "selected_gpu_available_bytes": available,
        "selected_gpu_compute_processes": tenants,
        "all_gpus": gpus,
        "all_compute_processes": processes,
        "gpu_unavailable_reason": unavailable_reason,
        "required_available_bytes": required_bytes,
        "expected_gpu_uuid": expected_uuid,
        "admitted": admitted,
    }


def acquire_gpu_reservation(args: argparse.Namespace) -> int:
    state = current_cuda_admission(gpu_index=args.gpu_index, required_bytes=0)
    if not state["admitted"]:
        raise ValueError("selected CUDA device is unavailable or has an active compute process")
    reservation = {
        "schema_version": 1,
        "captured_unix_seconds": time.time(),
        "token_sha256": reservation_token_sha256(args.token),
        "gpu_index": args.gpu_index,
        "gpu_uuid": state["selected_gpu_uuid"],
    }
    write_new(args.reservation, reservation)
    write_new(args.evidence, reservation)
    return 0


def check_gpu_reservation(args: argparse.Namespace) -> int:
    reservation = load_reservation(args.reservation, args.token, args.gpu_index)
    state = current_cuda_admission(
        gpu_index=args.gpu_index,
        required_bytes=args.required_bytes,
        expected_uuid=reservation["gpu_uuid"],
    )
    if not state["admitted"]:
        raise ValueError("selected CUDA device changed, lacks capacity, or has an active co-tenant")
    if args.output is not None:
        write_new(args.output, state)
    return 0


def release_gpu_reservation(args: argparse.Namespace) -> int:
    load_reservation(args.reservation, args.token, args.gpu_index)
    args.reservation.unlink()
    return 0


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


def host_load_bound(model: dict[str, Any], load_profile: str, sizes: dict[str, int],
                    language_variant: str | None, vision_variant: str | None) -> int:
    if load_profile == "mlx-unified" and model["key"].startswith("bonsai-"):
        if model["key"] == "bonsai-gguf":
            language = model.get("admission_mlx_language_load_bounds", {}).get(language_variant)
            vision = model.get("admission_mlx_vision_load_bounds", {}).get(vision_variant)
            if not all(isinstance(value, int) and value > 0 for value in (language, vision)):
                raise ValueError("missing pinned MLX GGUF load upper bound")
            return language + vision
        bound = model.get("admission_mlx_load_upper_bound_bytes")
        if not isinstance(bound, int) or bound <= 0:
            raise ValueError("missing pinned MLX safetensors load upper bound")
        return bound
    return (sizes["language_weight_bytes"] + sizes["vision_weight_bytes"]) * LOAD_PROFILES[load_profile]["host_weight_copies"]


def selected_artifact(path: Path, snapshot: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    absolute = lexical_absolute(path)
    snapshot_root = lexical_absolute(snapshot)
    if absolute.is_file():
        try:
            relative = absolute.relative_to(snapshot_root).as_posix()
        except ValueError as error:
            raise ValueError(f"selected artifact is outside the pinned snapshot: {absolute}") from error
        matches = [item for item in inventory["files"] if item["path"] == relative]
        if len(matches) != 1:
            raise ValueError(f"selected artifact is absent from snapshot inventory: {relative}")
        return {
            "path": str(absolute),
            "kind": "file",
            "bytes": matches[0]["size"],
            "sha256": matches[0]["sha256"],
        }
    if absolute != snapshot_root:
        raise ValueError(f"selected snapshot path does not equal the pinned snapshot: {absolute}")
    return {
        "path": str(absolute),
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


def validate_preflight_record(
    preflight: dict[str, Any],
    *,
    model: dict[str, Any],
    load_profile: str,
    language_variant: str | None,
    vision_variant: str | None,
) -> None:
    policy = LOAD_PROFILES[load_profile]
    sizes = pinned_admission_sizes(model, language_variant, vision_variant)
    weight_bytes = sizes["language_weight_bytes"] + sizes["vision_weight_bytes"]
    reserve = preflight.get("reserve_bytes")
    if not isinstance(reserve, int) or reserve < 0:
        raise ValueError("preflight reserve bytes are invalid")
    host_required = host_load_bound(model, load_profile, sizes, language_variant, vision_variant) + reserve
    gpu_copies = policy["gpu_weight_copies"]
    gpu_required = weight_bytes * gpu_copies + reserve if gpu_copies else None
    identity = {
        "model_key": model["key"],
        "model_revision": model["revision"],
        "language_variant": language_variant,
        "vision_variant": vision_variant,
        "artifact_sizes": sizes,
        "load_profile": load_profile,
        "host_weight_copies": policy["host_weight_copies"],
        "gpu_weight_copies": gpu_copies,
        "host_required_available_bytes": host_required,
        "gpu_required_available_bytes": gpu_required,
    }
    for field, expected in identity.items():
        if preflight.get(field) != expected:
            raise ValueError(f"preflight {field} does not match pinned admission contract")
    available = preflight.get("available_memory_bytes")
    host_admitted = isinstance(available, int) and available >= host_required
    if preflight.get("host_admitted") is not host_admitted:
        raise ValueError("preflight host admission verdict is inconsistent")
    if gpu_required is None:
        gpu_admitted = True
        if any(
            preflight.get(field) is not None
            for field in (
                "selected_gpu_index",
                "selected_gpu_uuid",
                "selected_gpu_available_bytes",
                "selected_gpu_compute_processes",
            )
        ):
            raise ValueError("non-CUDA preflight unexpectedly selects an NVIDIA device")
    else:
        selected_index = preflight.get("selected_gpu_index")
        selected_uuid = preflight.get("selected_gpu_uuid")
        selected_available = preflight.get("selected_gpu_available_bytes")
        tenants = preflight.get("selected_gpu_compute_processes")
        selected, derived_tenants = selected_gpu_state(
            preflight.get("gpus", []), preflight.get("compute_processes", []), CUDA_DEVICE_INDEX
        )
        if (
            selected_index != CUDA_DEVICE_INDEX
            or selected is None
            or selected_uuid != selected.get("uuid")
            or selected_available != selected.get("free_bytes")
            or tenants != derived_tenants
        ):
            raise ValueError("preflight selected-device evidence is inconsistent")
        if not isinstance(preflight.get("reservation_token_sha256"), str):
            raise ValueError("CUDA preflight lacks campaign reservation identity")
        gpu_admitted = (
            isinstance(selected_available, int)
            and selected_available >= gpu_required
            and tenants == []
        )
    if preflight.get("gpu_admitted") is not gpu_admitted:
        raise ValueError("preflight GPU admission verdict is inconsistent")
    if preflight.get("admitted") is not (host_admitted and gpu_admitted):
        raise ValueError("preflight aggregate admission verdict is inconsistent")


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime_sha = checked_sha(args.runtime_sha, "runtime SHA")
    source = source_identity(runtime_sha, args.allow_dirty)
    model = load_model(args.manifest, args.model_key)
    if model["revision"] != args.model_revision:
        raise ValueError("requested model revision does not match the pinned manifest")
    validate_variant_paths(args, model)
    verify_snapshot(model, args.snapshot)
    before = snapshot_inventory(model, args.snapshot)
    if model["key"].startswith("bonsai-"):
        assets.verify_inventory(model, before)
    measured_sizes = artifact_sizes(args.model_path, args.projector_path)
    pinned_sizes = pinned_admission_sizes(model, args.language_variant, args.vision_variant)
    if any(measured_sizes[key] != pinned_sizes[key] for key in pinned_sizes if key != "auxiliary_bytes"):
        raise ValueError(
            f"selected weight bytes {measured_sizes} do not match pinned admission bytes {pinned_sizes}"
        )
    preflight_path = args.preflight.resolve(strict=True)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    load_profile = preflight.get("load_profile")
    if load_profile not in LOAD_PROFILES:
        raise ValueError("run preflight has an unknown load profile")
    validate_preflight_record(
        preflight,
        model=model,
        load_profile=load_profile,
        language_variant=args.language_variant,
        vision_variant=args.vision_variant,
    )
    policy = LOAD_PROFILES[load_profile]
    if args.candle_device != policy["command_device"]:
        raise ValueError("wrapper device selection does not match preflight load profile")
    gpu_recheck = None
    if policy["device"] == "cuda":
        if args.reservation is None or args.reservation_token is None:
            raise ValueError("CUDA run requires the active campaign reservation")
        reservation = load_reservation(
            args.reservation, args.reservation_token, CUDA_DEVICE_INDEX
        )
        if reservation["token_sha256"] != preflight.get("reservation_token_sha256"):
            raise ValueError("run reservation does not match its preflight")
        gpu_recheck = current_cuda_admission(
            gpu_index=CUDA_DEVICE_INDEX,
            required_bytes=preflight["gpu_required_available_bytes"],
            expected_uuid=preflight["selected_gpu_uuid"],
        )
        if not gpu_recheck["admitted"]:
            raise ValueError(
                "selected CUDA device changed, lacks capacity, or gained an active co-tenant"
            )
    binary = args.binary.resolve(strict=True)
    provider_path = output / "provider.json"
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    command = [str(binary), args.test_name, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    env = os.environ.copy()
    env.update(
        {
            "BONSAI_COMPARISON_MODEL_PATH": str(lexical_absolute(args.model_path)),
            "BONSAI_COMPARISON_MODEL_ID": args.model_id,
            "BONSAI_COMPARISON_MODEL_REVISION": args.model_revision,
            "BONSAI_COMPARISON_RUNTIME_SHA": runtime_sha,
            "BONSAI_COMPARISON_OUTPUT": str(provider_path),
            "BONSAI_COMPARISON_CASES": json.dumps(args.cases.split(","), separators=(",", ":")),
        }
    )
    if args.candle_device is not None:
        env["CANDLE_LLM_DEVICE"] = args.candle_device
    if policy["device"] == "cuda":
        env["SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES"] = str(
            gpu_recheck["selected_gpu_available_bytes"]
        )
    if args.projector_path is not None:
        env["BONSAI_COMPARISON_PROJECTOR"] = str(lexical_absolute(args.projector_path))

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
            "load_profile": load_profile,
            "preflight_path": str(preflight_path),
            "preflight_sha256": sha256(preflight_path),
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
            "snapshot": str(lexical_absolute(args.snapshot)),
            "model_path": str(lexical_absolute(args.model_path)),
            "projector_path": str(lexical_absolute(args.projector_path))
            if args.projector_path
            else None,
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
            "admission_recheck": gpu_recheck,
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


def validate_preserve_thinking_evidence(case: dict[str, Any], root: Path) -> None:
    """Fail closed unless paired native history evidence proves the template control."""
    request = case.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"preserve_thinking request evidence is missing in {root}")
    preserved_request = request.get("preserved")
    stripped_request = request.get("stripped")
    if not isinstance(preserved_request, dict) or not isinstance(stripped_request, dict):
        raise ValueError(f"preserve_thinking paired requests are missing in {root}")
    if (
        preserved_request.get("preserve_thinking") is not True
        or stripped_request.get("preserve_thinking") is not False
    ):
        raise ValueError(f"preserve_thinking paired controls are malformed in {root}")
    preserved_without_control = dict(preserved_request)
    stripped_without_control = dict(stripped_request)
    preserved_without_control.pop("preserve_thinking", None)
    stripped_without_control.pop("preserve_thinking", None)
    if preserved_without_control != stripped_without_control:
        raise ValueError(f"preserve_thinking paired requests differ beyond the control in {root}")
    messages = preserved_request.get("messages")
    history_covered = (
        isinstance(messages, list)
        and len(messages) >= 3
        and any(
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("thinking"), str)
            and bool(message["thinking"].strip())
            for message in messages
        )
    )
    if case.get("history_coverage_passed") is not True or not history_covered:
        raise ValueError(f"preserve_thinking history coverage is missing in {root}")

    steps = case.get("paired_steps")
    if not isinstance(steps, dict):
        raise ValueError(f"preserve_thinking paired native steps are missing in {root}")
    preserved = steps.get("preserved")
    stripped = steps.get("stripped")
    if not isinstance(preserved, dict) or not isinstance(stripped, dict):
        raise ValueError(f"preserve_thinking paired native steps are malformed in {root}")
    if preserved.get("request") != preserved_request or stripped.get("request") != stripped_request:
        raise ValueError(f"preserve_thinking steps are not bound to their requests in {root}")
    for label, step in (("preserved", preserved), ("stripped", stripped)):
        if step.get("status") != "completed" or step.get("evidence_complete") is not True:
            raise ValueError(f"preserve_thinking {label} step is incomplete in {root}")
        if not isinstance(step.get("stream_contract_passed"), bool):
            raise ValueError(f"preserve_thinking {label} stream evidence is missing in {root}")

    proof = case.get("prompt_token_proof")
    if not isinstance(proof, dict):
        raise ValueError(f"preserve_thinking prompt-token proof is missing in {root}")
    preserved_tokens = preserved.get("output", {}).get("prompt_tokens")
    stripped_tokens = stripped.get("output", {}).get("prompt_tokens")
    if (
        not isinstance(preserved_tokens, int)
        or not isinstance(stripped_tokens, int)
        or preserved_tokens <= 0
        or stripped_tokens <= 0
        or proof.get("preserved_prompt_tokens") != preserved_tokens
        or proof.get("stripped_prompt_tokens") != stripped_tokens
    ):
        raise ValueError(f"preserve_thinking prompt-token counts are unbound in {root}")
    delta = preserved_tokens - stripped_tokens if preserved_tokens >= stripped_tokens else None
    passed = delta is not None and delta > 0
    if proof.get("additional_preserved_tokens") != delta or proof.get("passed") is not passed:
        raise ValueError(f"preserve_thinking prompt-token proof is inconsistent in {root}")

    stream_passed = (
        preserved.get("stream_contract_passed") is True
        and stripped.get("stream_contract_passed") is True
    )
    quality_passed = preserved.get("quality_passed") is True and stripped.get("quality_passed") is True
    functional_passed = quality_passed and stream_passed and history_covered and passed
    if (
        case.get("stream_contract_passed") is not stream_passed
        or case.get("quality_passed") is not quality_passed
        or case.get("functional_acceptance_passed") is not functional_passed
        or case.get("output") != preserved.get("output")
    ):
        raise ValueError(f"preserve_thinking aggregate evidence is inconsistent in {root}")


def validate_receipt(
    root: Path,
    *,
    expected_backend: str | None = None,
    expected_device: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    runtime_sha = receipt.get("runtime", {}).get("head_sha")
    for field, expected in (
        ("model_id", model.get("id")),
        ("model_revision", model.get("revision")),
        ("runtime_sha", runtime_sha),
    ):
        if provider.get(field) != expected:
            raise ValueError(f"provider {field} does not match enclosing receipt in {root}")
    if provider.get("context_admission", {}).get("evidence_complete") is not True:
        raise ValueError(f"provider lacks a genuine context admission rejection in {root}")
    resource = provider.get("resource_admission", {})
    if (
        resource.get("evidence_complete") is not True
        or resource.get("architecturally_within_context") is not True
        or resource.get("available_memory_override_bytes") != 1
    ):
        raise ValueError(f"provider lacks a genuine low-budget resource rejection in {root}")
    paired_cases = [case for case in provider.get("cases", []) if case.get("case_id") == resource.get("paired_case_id")]
    if len(paired_cases) != 1:
        raise ValueError(f"resource rejection lacks its identical normal-budget workload in {root}")
    paired = paired_cases[0]
    probe = resource.get("record", {})
    prompt_tokens = paired.get("output", {}).get("prompt_tokens")
    output_budget = paired.get("request", {}).get("max_new_tokens")
    context_tokens = resource.get("declared_context_tokens")
    if (
        probe.get("status") != "failed"
        or probe.get("request") != paired.get("request")
        or "request requires an estimated" not in probe.get("error", "")
        or "bytes of native workspace but only 1 bytes are available" not in probe.get("error", "")
        or not isinstance(prompt_tokens, int)
        or prompt_tokens != resource.get("paired_prompt_tokens")
        or not isinstance(output_budget, int)
        or not isinstance(context_tokens, int)
        or prompt_tokens + output_budget > context_tokens
    ):
        raise ValueError(f"resource rejection is not bound within the architectural window in {root}")
    memory_points = [
        provider.get("native_memory_before_load"),
        provider.get("native_memory_after_load"),
        provider.get("native_memory_after_unload"),
    ]
    if any(not isinstance(point, dict) or not point for point in memory_points):
        raise ValueError(f"provider native memory evidence is missing in {root}")
    native_runtimes = [
        (point.get("backend"), point.get("device")) for point in memory_points
    ]
    if len(set(native_runtimes)) != 1:
        raise ValueError(f"native runtime identity changed during run in {root}")
    backend, device = native_runtimes[0]
    if provider.get("provider", {}).get("backend") != backend:
        raise ValueError(f"provider descriptor backend does not match native runtime in {root}")
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(f"native backend does not match matrix cell in {root}")
    if expected_device is not None and device != expected_device:
        raise ValueError(f"native device does not match matrix cell in {root}")
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
        if case.get("case_id") == "preserve_thinking":
            validate_preserve_thinking_evidence(case, root)
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
    expected_runtime_sha = checked_sha(args.runtime_sha, "expected runtime SHA")
    for receipt, provider in rows:
        if receipt.get("runtime", {}).get("head_sha") != expected_runtime_sha:
            raise ValueError(
                f"runtime SHA mismatch for {receipt.get('model', {}).get('id')}"
            )
        if provider.get("runtime_sha") != expected_runtime_sha:
            raise ValueError(
                f"provider runtime SHA mismatch for {receipt.get('model', {}).get('id')}"
            )
    expected = args.expected_model
    ids = [receipt["model"]["id"] for receipt, _ in rows]
    if sorted(ids) != sorted(expected) or len(ids) != len(set(ids)):
        raise ValueError(f"model rows {ids!r} do not equal required rows {expected!r}")
    selected_case_ids = getattr(args, "case_ids", None)

    def selected_cases(provider: dict[str, Any]) -> list[dict[str, Any]]:
        cases = provider["cases"]
        if selected_case_ids is None:
            return cases
        by_id = {case.get("case_id"): case for case in cases}
        if len(by_id) != len(cases) or any(case_id not in by_id for case_id in selected_case_ids):
            raise ValueError("comparison row does not contain the exact selected cases")
        return [by_id[case_id] for case_id in selected_case_ids]

    baseline_cases = selected_cases(rows[0][1])
    identity = [(case["case_id"], case["category"], case["request"]) for case in baseline_cases]
    for receipt, provider in rows[1:]:
        actual = [
            (case["case_id"], case["category"], case["request"])
            for case in selected_cases(provider)
        ]
        if actual != identity:
            raise ValueError(f"unequal requests or budgets for {receipt['model']['id']}")
    categories: dict[str, list[dict[str, Any]]] = {}
    raw_outputs: list[dict[str, Any]] = []
    for receipt, provider in rows:
        for case in selected_cases(provider):
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


def validate_full_acceptance_contract(spec: dict[str, Any]) -> None:
    """Reject deletion or weakening of any capability or format acceptance route."""
    if spec.get("acceptance_contract") != FULL_ACCEPTANCE_CONTRACT:
        if any(str(cell.get("model_key", "")).startswith("bonsai-") for cell in spec.get("cells", [])):
            raise ValueError("native matrix requires the full acceptance contract")
        return
    groups = spec.get("groups", {})
    format_group = groups.get("format-functional", {})
    if format_group.get("case_ids") != FORMAT_ACCEPTANCE_CASES:
        raise ValueError(
            "full acceptance requires image and both ordered video cases on every format route"
        )
    if format_group.get("functional_acceptance") is not True:
        raise ValueError("format-functional must remain a functional acceptance group")
    cells = spec.get("cells", [])
    expected_matched_ids = {
        f"{backend}-{model}"
        for backend, models in (
            ("mlx", ("qwen38-parent", "bonsai-mlx-2bit", "qwen3vl-baseline")),
            ("candle-cuda", ("qwen38-parent", "bonsai-gguf", "qwen3vl-baseline")),
            ("candle-cpu", ("qwen38-parent", "bonsai-gguf", "qwen3vl-baseline")),
        )
        for model in models
    }
    if {cell.get("id") for cell in cells if cell.get("group") == "matched"} != expected_matched_ids:
        raise ValueError("full acceptance matched model/backend routes are incomplete")
    required_format_ids = {
        "functional-mlx-bonsai-mlx",
        "functional-candle-bonsai-mlx",
        "functional-mlx-pq2-bf16",
        "functional-mlx-pq2-q8",
        "functional-mlx-ptq1-bf16",
        "functional-mlx-ptq1-q8",
        "functional-candle-pq2-bf16",
        "functional-candle-pq2-q8",
        "functional-candle-ptq1-bf16",
        "functional-candle-ptq1-q8",
    }
    actual_format_ids = {
        cell.get("id") for cell in cells if cell.get("group") == "format-functional"
    }
    if actual_format_ids != required_format_ids:
        raise ValueError("full acceptance format/backend/projector routes are incomplete")
    for cell in cells:
        expected = None
        if cell.get("group") == "matched" and cell.get("model_key") == "bonsai-qwen38-parent":
            expected = QWEN38_ACCEPTANCE_CASES
        elif cell.get("group") == "matched" and cell.get("model_key") in {
            "bonsai-mlx-2bit",
            "bonsai-gguf",
        }:
            expected = BONSAI_ACCEPTANCE_CASES
        if expected is not None:
            if cell.get("acceptance_case_ids") != expected:
                raise ValueError(f"full acceptance cases are incomplete for {cell.get('id')}")
            if cell.get("functional_acceptance") is not True:
                raise ValueError(f"native capability acceptance is disabled for {cell.get('id')}")


def matrix_status(args: argparse.Namespace) -> int:
    expected_runtime_sha = checked_sha(args.runtime_sha, "expected runtime SHA")
    spec = json.loads(args.matrix.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("suite") != SUITE:
        raise ValueError("unsupported comparison matrix schema")
    validate_full_acceptance_contract(spec)
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
    hardware_paths = [root / "hardware-before.json" for root in roots]
    if any(not path.is_file() for path in hardware_paths):
        raise ValueError("every evidence root requires hardware-before.json")
    for path in hardware_paths:
        validate_hardware_record(path)
    rows = []
    complete_runs: dict[str, list[Path]] = {group: [] for group in groups}
    acceptance_rows: list[dict[str, Any]] = []
    sealed_files: list[tuple[int, Path]] = [
        (index, path) for index, path in enumerate(hardware_paths)
    ]
    if spec.get("acceptance_contract") == FULL_ACCEPTANCE_CONTRACT:
        for index, root in enumerate(roots):
            metadata_path = root / "snapshot-metadata.json"
            if not metadata_path.is_file():
                raise ValueError("full acceptance requires snapshot metadata for every platform")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != SCHEMA_VERSION
                or metadata.get("all_metadata_qualified") is not True
                or metadata.get("publisher_verified_all") is not True
                or metadata.get("runtime_sha") != expected_runtime_sha
                or metadata.get("publisher_closure_sha256") != sha256(assets.DEFAULT_CLOSURE)
            ):
                raise ValueError("platform snapshot metadata is incomplete or malformed")
            provision_path = root / "provision-report.json"
            provision = json.loads(provision_path.read_text(encoding="utf-8"))
            if (provision.get("complete") is not True or provision.get("runtime_sha") != expected_runtime_sha
                or provision.get("publisher_closure_sha256") != sha256(assets.DEFAULT_CLOSURE)
                or provision.get("model_execution_performed") is not False):
                raise ValueError("platform publisher verification is incomplete or unbound")
            sealed_files.extend(((index, metadata_path), (index, provision_path)))
    cuda_reservation_hashes: set[str] = set()
    for cell in cells:
        cell_id = cell["id"]
        group = cell.get("group")
        if group not in groups:
            raise ValueError(f"comparison matrix cell {cell_id} names unknown group {group!r}")
        load_profile = cell.get("load_profile")
        if load_profile not in LOAD_PROFILES:
            raise ValueError(f"comparison matrix cell {cell_id} has unknown load profile")
        policy = LOAD_PROFILES[load_profile]
        if cell.get("backend") != policy["backend"] or cell.get("device") != policy["device"]:
            raise ValueError(f"comparison matrix cell {cell_id} contradicts its load profile")
        model = load_model(args.manifest, cell["model_key"])
        preflight_locations = [
            (index, root / f"{cell_id}-preflight.json") for index, root in enumerate(roots)
        ]
        present_preflights = [item for item in preflight_locations if item[1].is_file()]
        row = {**cell, "status": "pending", "reason": None}
        if len(present_preflights) > 1:
            row.update(status="incomplete", reason="duplicate preflight evidence")
        elif not present_preflights:
            row.update(status="incomplete", reason="missing preflight evidence")
        else:
            root_index, preflight_path = present_preflights[0]
            sealed_files.append((root_index, preflight_path))
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            row["preflight"] = preflight
            try:
                validate_preflight_record(
                    preflight,
                    model=model,
                    load_profile=load_profile,
                    language_variant=cell.get("language_variant"),
                    vision_variant=cell.get("vision_variant"),
                )
                if policy["device"] == "cuda":
                    reservation_path = roots[root_index] / "gpu-reservation.json"
                    if not reservation_path.is_file():
                        raise ValueError("CUDA evidence root lacks campaign reservation")
                    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
                    if (
                        reservation.get("gpu_index") != CUDA_DEVICE_INDEX
                        or reservation.get("gpu_uuid") != preflight.get("selected_gpu_uuid")
                        or reservation.get("token_sha256")
                        != preflight.get("reservation_token_sha256")
                    ):
                        raise ValueError("CUDA preflight does not match campaign reservation")
                    sealed_files.append((root_index, reservation_path))
                    cuda_reservation_hashes.add(reservation["token_sha256"])
                if preflight.get("admitted") is not True:
                    row.update(
                        status="not_admitted",
                        reason="live capacity preflight rejected this cell",
                    )
                else:
                    run_root = roots[root_index] / cell_id
                    if not run_root.is_dir():
                        row.update(status="incomplete", reason="admitted cell has no run evidence")
                    else:
                        receipt, provider = validate_receipt(
                            run_root,
                            expected_backend=cell["backend"],
                            expected_device=cell["device"],
                        )
                        receipt_model = receipt.get("model", {})
                        if receipt_model.get("id") != cell_id:
                            raise ValueError("run model id does not match matrix cell")
                        if receipt_model.get("manifest_key") != cell.get("model_key"):
                            raise ValueError("run model key does not match matrix cell")
                        if receipt_model.get("revision") != model["revision"]:
                            raise ValueError("run revision does not match pinned manifest")
                        if receipt_model.get("artifact_sizes") != pinned_admission_sizes(
                            model,
                            cell.get("language_variant"),
                            cell.get("vision_variant"),
                        ):
                            raise ValueError("run artifact sizes do not match pinned manifest")
                        if receipt_model.get("language_variant") != cell.get(
                            "language_variant"
                        ) or receipt_model.get("vision_variant") != cell.get("vision_variant"):
                            raise ValueError("run format variants do not match matrix cell")
                        if receipt.get("runtime", {}).get("head_sha") != expected_runtime_sha:
                            raise ValueError("run runtime SHA does not match workflow SHA")
                        command = receipt.get("command", {})
                        if command.get("load_profile") != load_profile:
                            raise ValueError("run load profile does not match matrix cell")
                        if command.get("candle_device") != policy["command_device"]:
                            raise ValueError("run device selector does not match matrix cell")
                        if command.get("preflight_sha256") != sha256(preflight_path):
                            raise ValueError("run is not bound to its preflight evidence")
                        recheck = receipt.get("gpu", {}).get("admission_recheck")
                        if policy["device"] == "cuda":
                            if (
                                not isinstance(recheck, dict)
                                or recheck.get("admitted") is not True
                                or recheck.get("gpu_index") != CUDA_DEVICE_INDEX
                                or recheck.get("selected_gpu_uuid")
                                != preflight.get("selected_gpu_uuid")
                                or recheck.get("selected_gpu_compute_processes") != []
                            ):
                                raise ValueError("CUDA run lacks a clean selected-device recheck")
                        elif recheck is not None:
                            raise ValueError("non-CUDA run unexpectedly contains a CUDA recheck")
                        group_case_ids = groups[group].get("case_ids")
                        acceptance_case_ids = cell.get("acceptance_case_ids", [])
                        if not isinstance(group_case_ids, list) or not isinstance(
                            acceptance_case_ids, list
                        ):
                            raise ValueError("matrix cases must be string arrays")
                        expected_case_ids = group_case_ids + acceptance_case_ids
                        if [case.get("case_id") for case in provider["cases"]] != expected_case_ids:
                            raise ValueError("run cases do not match matrix workload group")
                        accepted_ids = []
                        if groups[group].get("functional_acceptance") is True:
                            accepted_ids.extend(group_case_ids)
                        if cell.get("functional_acceptance") is True:
                            accepted_ids.extend(acceptance_case_ids)
                        by_id = {case["case_id"]: case for case in provider["cases"]}
                        acceptance_rows.extend(
                            {
                                "cell_id": cell_id,
                                "case_id": case_id,
                                "category": by_id[case_id].get("category"),
                                "passed": by_id[case_id].get("functional_acceptance_passed") is True,
                                "request": by_id[case_id].get("request"),
                                "oracle": by_id[case_id].get("oracle"),
                                "output": by_id[case_id].get("output"),
                                "evidence_complete": by_id[case_id].get("evidence_complete"),
                                "stream_contract_passed": by_id[case_id].get("stream_contract_passed"),
                                "history_coverage_passed": by_id[case_id].get("history_coverage_passed"),
                                "prompt_token_proof": by_id[case_id].get("prompt_token_proof"),
                            }
                            for case_id in accepted_ids
                        )
                        row.update(status="completed", reason=None)
                        complete_runs[group].append(run_root)
                        sealed_files.append((root_index, run_root / "seal.json"))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                row.update(status="incomplete", reason=str(error))
        rows.append(row)
    complete = all(row["status"] == "completed" for row in rows)
    if len(cuda_reservation_hashes) > 1:
        complete = False
        rows.append(
            {
                "id": "cuda-reservation-consistency",
                "backend": "candle",
                "device": "cuda",
                "status": "incomplete",
                "reason": "CUDA cells use different campaign reservations",
            }
        )
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
                        runtime_sha=expected_runtime_sha,
                        case_ids=groups[group]["case_ids"],
                        output=temp / "comparison.json",
                        markdown=temp / "comparison.md",
                    )
                )
                comparison[group] = json.loads(
                    (temp / "comparison.json").read_text(encoding="utf-8")
                )
    acceptance_complete = complete and (
        all(row["passed"] for row in acceptance_rows)
        if acceptance_rows
        else spec.get("acceptance_contract") != FULL_ACCEPTANCE_CONTRACT
    )
    report = {
        "schema_version": 1,
        "suite": SUITE,
        "runtime_sha": expected_runtime_sha,
        "evidence_complete": complete,
        "required_cells": rows,
        "comparison": comparison,
        "functional_acceptance": {
            "passed": acceptance_complete,
            "rows": acceptance_rows,
        },
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
        f"Functional acceptance passed: **{acceptance_complete}**",
        "",
        "| Cell | Backend | Device | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['id']} | {row['backend']} | {row['device']} | {row['status']} | {row['reason'] or ''} |"
        )
    markdown.extend(
        [
            "",
            "## Functional acceptance raw outcomes",
            "",
            "```json",
            json.dumps(acceptance_rows, indent=2, sort_keys=True),
            "```",
        ]
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
    if complete:
        unique_files = sorted(set(sealed_files), key=lambda item: (item[0], str(item[1])))
        entries = []
        for root_index, path in unique_files:
            entries.append(
                {
                    "root_index": root_index,
                    "path": path.relative_to(roots[root_index]).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        for role, path in (("matrix_report", args.output), ("matrix_markdown", args.markdown)):
            entries.append(
                {
                    "role": role,
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        write_new(
            args.seal,
            {
                "schema_version": 1,
                "suite": SUITE,
                "runtime_sha": expected_runtime_sha,
                "matrix_sha256": sha256(args.matrix),
                "manifest_sha256": sha256(args.manifest),
                "files": entries,
            },
        )
    return 0 if complete and acceptance_complete else 1


def verify_matrix_seal(args: argparse.Namespace) -> int:
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    expected_runtime_sha = checked_sha(args.runtime_sha, "expected runtime SHA")
    if (
        seal.get("schema_version") != 1
        or seal.get("suite") != SUITE
        or seal.get("runtime_sha") != expected_runtime_sha
        or seal.get("matrix_sha256") != sha256(args.matrix)
        or seal.get("manifest_sha256") != sha256(args.manifest)
    ):
        raise ValueError("matrix seal identity does not match this campaign")
    roots = [root.resolve() for root in args.root]
    report = json.loads(args.output.read_text(encoding="utf-8"))
    if (
        report.get("suite") != SUITE
        or report.get("runtime_sha") != expected_runtime_sha
        or report.get("evidence_complete") is not True
    ):
        raise ValueError("sealed matrix report is not complete for this campaign")
    spec = json.loads(args.matrix.read_text(encoding="utf-8"))
    validate_full_acceptance_contract(spec)
    if (
        spec.get("acceptance_contract") == FULL_ACCEPTANCE_CONTRACT
        and report.get("functional_acceptance", {}).get("passed") is not True
    ):
        raise ValueError("sealed matrix report did not pass functional acceptance")
    expected_root_files = {
        (index, "hardware-before.json") for index in range(len(roots))
    }
    if spec.get("acceptance_contract") == FULL_ACCEPTANCE_CONTRACT:
        expected_root_files.update(
            (index, name) for index in range(len(roots))
            for name in ("snapshot-metadata.json", "provision-report.json")
        )
    for cell in spec["cells"]:
        locations = [
            (index, root / f"{cell['id']}-preflight.json")
            for index, root in enumerate(roots)
        ]
        present = [item for item in locations if item[1].is_file()]
        if len(present) != 1:
            raise ValueError(f"sealed cell {cell['id']} does not have one preflight")
        root_index, _ = present[0]
        expected_root_files.add((root_index, f"{cell['id']}-preflight.json"))
        expected_root_files.add((root_index, f"{cell['id']}/seal.json"))
        receipt, provider = validate_receipt(
            roots[root_index] / cell["id"],
            expected_backend=cell["backend"],
            expected_device=cell["device"],
        )
        if (
            receipt.get("runtime", {}).get("head_sha") != expected_runtime_sha
            or provider.get("runtime_sha") != expected_runtime_sha
        ):
            raise ValueError(f"sealed cell {cell['id']} has a different runtime SHA")
        if cell["device"] == "cuda":
            expected_root_files.add((root_index, "gpu-reservation.json"))
    entries = seal.get("files")
    if not isinstance(entries, list):
        raise ValueError("matrix seal has no artifact list")
    actual_root_files = {
        (entry.get("root_index"), entry.get("path"))
        for entry in entries
        if "root_index" in entry
    }
    if actual_root_files != expected_root_files or len(actual_root_files) != sum(
        1 for entry in entries if "root_index" in entry
    ):
        raise ValueError("matrix seal does not cover the exact required evidence set")
    roles = [entry.get("role") for entry in entries if "root_index" not in entry]
    if sorted(roles) != ["matrix_markdown", "matrix_report"]:
        raise ValueError("matrix seal does not cover the exact aggregate reports")
    for entry in entries:
        if "root_index" in entry:
            index = entry["root_index"]
            if not isinstance(index, int) or index < 0 or index >= len(roots):
                raise ValueError("matrix seal names an invalid evidence root")
            path = roots[index] / entry["path"]
        else:
            role_paths = {
                "matrix_report": args.output,
                "matrix_markdown": args.markdown,
            }
            path = role_paths.get(entry.get("role"))
            if path is None or path.name != entry.get("path"):
                raise ValueError("matrix seal names an unknown aggregate artifact")
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"matrix seal artifact mismatch for {path}")
    return 0


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
    policy = LOAD_PROFILES[args.load_profile]
    host_required = host_load_bound(model, args.load_profile, sizes, args.language_variant, args.vision_variant) + args.reserve_bytes
    gpu_copies = policy["gpu_weight_copies"]
    gpu_required = weight_bytes * gpu_copies + args.reserve_bytes if gpu_copies else None
    total, available, reason = physical_memory()
    gpus, processes, gpu_reason = nvidia_hardware()
    host_admitted = available is not None and available >= host_required
    selected_gpu = None
    selected_processes = None
    reservation_hash = None
    if gpu_required is None:
        gpu_available = None
        gpu_admitted = True
    else:
        if args.reservation is None or args.reservation_token is None:
            raise ValueError("CUDA preflight requires an active campaign reservation")
        reservation = load_reservation(
            args.reservation, args.reservation_token, CUDA_DEVICE_INDEX
        )
        selected_gpu, selected_processes = selected_gpu_state(
            gpus, processes, CUDA_DEVICE_INDEX
        )
        gpu_available = selected_gpu.get("free_bytes") if selected_gpu else None
        reservation_hash = reservation["token_sha256"]
        gpu_admitted = (
            selected_gpu is not None
            and selected_gpu.get("uuid") == reservation.get("gpu_uuid")
            and isinstance(gpu_available, int)
            and gpu_available >= gpu_required
            and selected_processes == []
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
        "host_admission_formula": ("pinned_header_derived_load_upper_bound_bytes + reserve_bytes"
            if args.load_profile == "mlx-unified" and model["key"].startswith("bonsai-")
            else "exact_pinned_weight_bytes * host_weight_copies + reserve_bytes"),
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
        "available_memory_basis": "free + inactive + speculative pages; conservative reclaimable estimate" if sys.platform == "darwin" else "operating-system available physical memory",
        "unavailable_reason": reason,
        "gpus": gpus,
        "compute_processes": processes,
        "gpu_unavailable_reason": gpu_reason,
        "selected_gpu_index": CUDA_DEVICE_INDEX if gpu_required is not None else None,
        "selected_gpu_uuid": selected_gpu.get("uuid") if selected_gpu else None,
        "selected_gpu_available_bytes": gpu_available,
        "selected_gpu_compute_processes": selected_processes,
        "reservation_token_sha256": reservation_hash,
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
                "available_memory_basis": "free + inactive + speculative pages; conservative reclaimable estimate" if sys.platform == "darwin" else "operating-system available physical memory",
                "memory_unavailable_reason": memory_reason,
            },
            "gpus": gpus,
            "compute_processes": processes,
            "gpu_unavailable_reason": gpu_reason,
        },
    )
    return 0


def snapshot_metadata(path: Path | None, model: dict[str, Any]) -> dict[str, Any]:
    """Inspect only the manifest-named files and lightweight shard index; never hash payloads."""
    expected_files = []
    if path is not None and path.is_dir():
        relatives = list(model.get("expected_files", []))
        for relative in list(relatives):
            index_path = path / relative
            if relative.endswith(".index.json") and index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                for shard in sorted(set(index.get("weight_map", {}).values())):
                    if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts:
                        raise ValueError("snapshot index contains an invalid shard path")
                    if shard not in relatives:
                        relatives.append(shard)
        for relative in relatives:
            item = path / relative
            expected_files.append({"path": relative, "present": item.is_file(),
                                   "bytes": item.stat().st_size if item.is_file() else None})
    revision = path.name.lower() if path is not None else None
    revision_matches = revision == model["revision"]
    complete = bool(expected_files) and all(
        item["present"] and isinstance(item["bytes"], int) and item["bytes"] > 0
        for item in expected_files
    )
    return {
        "configured_path": str(path) if path is not None else None,
        "directory_present": bool(path is not None and path.is_dir()),
        "revision_from_snapshot_path": revision,
        "revision_matches": revision_matches,
        "expected_file_metadata": expected_files,
        "metadata_qualified": bool(path is not None and complete and revision_matches),
        "artifact_identity_verified": False,
    }


def qualify_snapshots(args: argparse.Namespace) -> int:
    """Record cheap metadata at configured paths and exact pinned candidates in known HF roots."""
    home = Path.home()
    candidate_roots = [("HOME", home),
                       ("default_huggingface_hub", home / ".cache" / "huggingface" / "hub")]
    hub_roots = [candidate_roots[1]]
    for name in ("HF_HOME", "HF_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            path = lexical_absolute(Path(value))
            candidate_roots.append((name, path))
            if name == "HF_HOME":
                path = path / "hub"
                candidate_roots.append(("HF_HOME_hub", path))
            hub_roots.append((name, path))
    bindings = []
    for value in args.binding:
        if "=" not in value:
            raise ValueError("snapshot binding must be ENVIRONMENT_VARIABLE=model-key")
        environment, model_key = value.split("=", 1)
        if not environment or not model_key:
            raise ValueError("snapshot binding must name both an environment variable and model")
        model = load_model(args.manifest, model_key)
        configured = os.environ.get(environment, "").strip()
        path = lexical_absolute(Path(configured)) if configured else None
        repository_parts = model["repository"].split("/")
        if len(repository_parts) != 2 or any(part in ("", ".", "..") or "\\" in part for part in repository_parts):
            raise ValueError("manifest repository must be an owner/name pair")
        suffix = Path("models--" + "--".join(repository_parts)) / "snapshots" / model["revision"]
        candidates = []
        seen = set()
        for source, hub in hub_roots:
            candidate = hub / suffix
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            candidates.append({"cache_root_source": source, **snapshot_metadata(candidate, model)})
        bindings.append({
            "environment": environment,
            "model_key": model_key,
            "repository": model["repository"],
            "expected_revision": model["revision"],
            **snapshot_metadata(path, model),
            "candidate_snapshots": candidates,
            "candidate_scope": "exact pinned repository/revision paths only; no recursive discovery or automatic selection",
        })
    record = {
        "schema_version": SCHEMA_VERSION,
        "scope": "metadata-only; no payload hashing, provisioning, or model load",
        "captured_unix_seconds": time.time(),
        "requested_platform": args.platform,
        "actual_platform": platform.system(),
        "account": getpass.getuser(),
        "hostname": platform.node(),
        "candidate_cache_roots": [
            {"name": name, "path": str(path), "present": path.is_dir()}
            for name, path in candidate_roots
        ],
        "snapshots": bindings,
        "all_metadata_qualified": bool(bindings)
        and all(binding["metadata_qualified"] for binding in bindings),
    }
    write_new(args.output, record)
    return 0 if record["all_metadata_qualified"] else 1


def validate_phase(args: argparse.Namespace) -> int:
    if args.preflight_only == "true" and args.provision_only == "true":
        raise ValueError("preflight-only and provision-only are mutually exclusive")
    return 0


def provision_assets(args: argparse.Namespace) -> int:
    """Admit each pinned model before downloading, verify publisher bytes, then refresh metadata."""
    runtime_sha = checked_sha(args.runtime_sha, "runtime SHA")
    source_identity(runtime_sha, False)
    spec = json.loads(args.matrix.read_text(encoding="utf-8"))
    validate_full_acceptance_contract(spec)
    backend = "mlx" if args.platform == "macos" else "candle"
    cells = [cell for cell in spec["cells"] if cell["backend"] == backend]
    plans = {}
    # Validate the complete platform admission/configuration plan before any network or mutation.
    for cell in cells:
        path = args.evidence_root / f"{cell['id']}-preflight.json"
        preflight = json.loads(path.read_text(encoding="utf-8"))
        model = load_model(args.manifest, cell["model_key"])
        validate_preflight_record(preflight, model=model, load_profile=cell["load_profile"],
                                  language_variant=cell.get("language_variant"), vision_variant=cell.get("vision_variant"))
        plan = plans.setdefault(cell["model_key"], {"model": model, "cells": [], "preflights": []})
        if preflight["admitted"]:
            plan["cells"].append(cell)
            plan["preflights"].append((path, preflight))
    for plan in plans.values():
        model = plan["model"]
        environment = model["environment"][0]
        configured = os.environ.get(environment, "").strip()
        if plan["cells"] and not configured:
            raise ValueError(f"admitted model has no configured snapshot: {environment}")
        plan["snapshot"] = lexical_absolute(Path(configured)) if configured else None
        plan["frozen"] = assets.frozen_model(model, args.closure)
    # Account for every planned missing file on each destination filesystem before the first
    # transfer. One largest-file staging reserve is explicit, not a measured disk peak.
    disks = {}
    for plan in plans.values():
        if not plan["cells"]:
            continue
        snapshot = plan["snapshot"]
        model = plan["model"]
        if (snapshot.name != model["revision"] or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--" + model["repository"].replace("/", "--")):
            raise ValueError("provisioning requires the exact publisher repository/revision HF cache path")
        existing = snapshot
        while not existing.exists():
            existing = existing.parent
        disk = disks.setdefault(existing.stat().st_dev, {
            "existing_path": str(existing), "missing_payload_bytes": 0,
            "staging_reserve_bytes": 0, "available_bytes": shutil.disk_usage(existing).free,
        })
        for item in assets.selected_files(plan["frozen"], plan["cells"]):
            if not (snapshot / item["path"]).exists():
                disk["missing_payload_bytes"] += item["bytes"]
                disk["staging_reserve_bytes"] = max(disk["staging_reserve_bytes"], item["bytes"])
    for disk in disks.values():
        disk["required_available_bytes"] = disk["missing_payload_bytes"] + disk["staging_reserve_bytes"]
        disk["admitted"] = disk["available_bytes"] >= disk["required_available_bytes"]
    write_new(args.evidence_root / "disk-admission.json", {
        "schema_version": 1, "runtime_sha": runtime_sha,
        "basis": "sum of selected missing publisher payload sizes plus one largest-file staging reserve",
        "is_measured_peak": False, "filesystems": list(disks.values()),
    })
    if any(not disk["admitted"] for disk in disks.values()):
        raise ValueError("insufficient disk capacity for the complete admitted provisioning plan")
    rows = []
    for key, plan in plans.items():
        row = {"model_key": key, "status": "not_admitted", "files": [],
               "preflight_sha256": [sha256(path) for path, _ in plan["preflights"]]}
        if plan["cells"]:
            try:
                # Recheck current capacity immediately before fetching; never use a rejected
                # preflight or another GPU's free bytes to authorize asset materialization.
                _, available, reason = physical_memory()
                viable = []
                for cell, (_, preflight) in zip(plan["cells"], plan["preflights"]):
                    if available is None or available < preflight["host_required_available_bytes"]:
                        continue
                    if cell["device"] == "cuda":
                        reservation = load_reservation(Path(os.environ["BONSAI_GPU_RESERVATION"]),
                                                       os.environ["BONSAI_RESERVATION_TOKEN"], CUDA_DEVICE_INDEX)
                        if reservation["token_sha256"] != preflight["reservation_token_sha256"]:
                            raise ValueError("provision reservation differs from admission")
                        current = current_cuda_admission(gpu_index=CUDA_DEVICE_INDEX,
                            required_bytes=preflight["gpu_required_available_bytes"],
                            expected_uuid=preflight["selected_gpu_uuid"])
                        if not current["admitted"]:
                            continue
                    viable.append(cell)
                if not viable:
                    raise ValueError(f"no selected model cell remains admitted before fetch: {reason}")
                files = assets.selected_files(plan["frozen"], viable)
                # This import occurs only after admission; metadata-only qualification never
                # imports a network client, downloads an asset, or creates a model process.
                from huggingface_hub import hf_hub_download
                row["files"] = assets.provision_snapshot(plan["model"], plan["snapshot"], files, hf_hub_download)
                row["status"] = "publisher_verified"
                row["full_snapshot_verified"] = len(row["files"]) == len(plan["frozen"]["files"])
                row["admitted_cells"] = [cell["id"] for cell in viable]
            except Exception as error:
                row.update(status="failed", error=str(error))
        rows.append(row)
    metadata_path = args.evidence_root / "snapshot-metadata.json"
    qualify_snapshots(argparse.Namespace(platform=args.platform, manifest=args.manifest,
        binding=[f"{plan['model']['environment'][0]}={key}" for key, plan in plans.items()],
        output=metadata_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = all(row["status"] == "publisher_verified" and row.get("full_snapshot_verified") is True for row in rows) and metadata["all_metadata_qualified"]
    metadata.update(scope="post-provision publisher Git/LFS hash verification; no model load",
                    runtime_sha=runtime_sha, publisher_closure_sha256=sha256(args.closure),
                    publisher_verified_all=complete)
    for binding in metadata["snapshots"]:
        row = next(row for row in rows if row["model_key"] == binding["model_key"])
        binding["artifact_identity_verified"] = row.get("full_snapshot_verified") is True
    # The output was created exclusively by this invocation. Replace only our just-written
    # metadata record with its publisher-verification fields; no cached model files are rewritten.
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_new(args.evidence_root / "provision-report.json", {
        "schema_version": 1, "runtime_sha": runtime_sha,
        "publisher_closure_sha256": sha256(args.closure), "complete": complete,
        "models": rows, "model_execution_performed": False,
    })
    return 0 if complete else 1


def validate_hardware_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError(f"unsupported hardware evidence schema in {path}")
    if not isinstance(record.get("captured_unix_seconds"), (int, float)):
        raise ValueError(f"hardware evidence lacks capture time in {path}")
    host = record.get("host")
    if not isinstance(host, dict) or not isinstance(host.get("system"), str):
        raise ValueError(f"hardware evidence lacks host identity in {path}")
    if not isinstance(record.get("gpus"), list) or not isinstance(
        record.get("compute_processes"), list
    ):
        raise ValueError(f"hardware evidence lacks GPU/process inventory in {path}")
    return record


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
    run_p.add_argument("--preflight", type=Path, required=True)
    run_p.add_argument("--reservation", type=Path)
    run_p.add_argument("--reservation-token")
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
    val.add_argument("--runtime-sha", required=True)
    val.add_argument("--output", type=Path, required=True)
    val.add_argument("--markdown", type=Path, required=True)
    val.set_defaults(func=validate)
    matrix = sub.add_parser("matrix-status")
    matrix.add_argument("--root", type=Path, action="append", required=True)
    matrix.add_argument(
        "--matrix", type=Path, default=Path("release/qwen38-bonsai-matrix.json")
    )
    matrix.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    matrix.add_argument("--runtime-sha", required=True)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--markdown", type=Path, required=True)
    matrix.add_argument("--seal", type=Path, required=True)
    matrix.set_defaults(func=matrix_status)
    verify = sub.add_parser("verify-matrix-seal")
    verify.add_argument("--root", type=Path, action="append", required=True)
    verify.add_argument(
        "--matrix", type=Path, default=Path("release/qwen38-bonsai-matrix.json")
    )
    verify.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    verify.add_argument("--runtime-sha", required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--markdown", type=Path, required=True)
    verify.add_argument("--seal", type=Path, required=True)
    verify.set_defaults(func=verify_matrix_seal)
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
    admit.add_argument("--reservation", type=Path)
    admit.add_argument("--reservation-token")
    admit.add_argument("--output", type=Path, required=True)
    admit.set_defaults(func=preflight)
    phase = sub.add_parser("validate-phase")
    phase.add_argument("--preflight-only", choices=("true", "false"), required=True)
    phase.add_argument("--provision-only", choices=("true", "false"), required=True)
    phase.set_defaults(func=validate_phase)
    provision = sub.add_parser("provision-assets")
    provision.add_argument("--platform", choices=("macos", "windows"), required=True)
    provision.add_argument("--runtime-sha", required=True)
    provision.add_argument("--evidence-root", type=Path, required=True)
    provision.add_argument("--matrix", type=Path, default=Path("release/qwen38-bonsai-matrix.json"))
    provision.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    provision.add_argument("--closure", type=Path, default=assets.DEFAULT_CLOSURE)
    provision.set_defaults(func=provision_assets)
    hw = sub.add_parser("hardware")
    hw.add_argument("--output", type=Path, required=True)
    hw.set_defaults(func=hardware)
    qualify = sub.add_parser("qualify-snapshots")
    qualify.add_argument("--platform", choices=("macos", "windows"), required=True)
    qualify.add_argument("--binding", action="append", required=True)
    qualify.add_argument(
        "--manifest", type=Path, default=Path("release/real-weight-models.toml")
    )
    qualify.add_argument("--output", type=Path, required=True)
    qualify.set_defaults(func=qualify_snapshots)
    reserve = sub.add_parser("reserve-gpu")
    reserve.add_argument("--reservation", type=Path, required=True)
    reserve.add_argument("--evidence", type=Path, required=True)
    reserve.add_argument("--token", required=True)
    reserve.add_argument("--gpu-index", type=int, default=CUDA_DEVICE_INDEX)
    reserve.set_defaults(func=acquire_gpu_reservation)
    check = sub.add_parser("check-gpu-reservation")
    check.add_argument("--reservation", type=Path, required=True)
    check.add_argument("--token", required=True)
    check.add_argument("--gpu-index", type=int, default=CUDA_DEVICE_INDEX)
    check.add_argument("--required-bytes", type=int, default=0)
    check.add_argument("--output", type=Path)
    check.set_defaults(func=check_gpu_reservation)
    release = sub.add_parser("release-gpu")
    release.add_argument("--reservation", type=Path, required=True)
    release.add_argument("--token", required=True)
    release.add_argument("--gpu-index", type=int, default=CUDA_DEVICE_INDEX)
    release.set_defaults(func=release_gpu_reservation)
    return out


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"qwen38-bonsai-terminal: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

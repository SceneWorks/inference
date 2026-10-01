#!/usr/bin/env python3
"""Small fail-closed process supervisor for the SC-20684/86 media campaigns.

A row is admitted by its runtime guards, never by a static whole-process peak proof: the
loader, lazy graphs, allocator/driver headroom and output coexistence make that peak
unobtainable from source. This module never starts a child without a mandatory policy that
configures every guard (deadline, sampling, termination grace, host reserve, child footprint
watchdog cap, and for CUDA the device reserve/cap) and a fresh host/device probe covering
cap plus reserve. The admission and the unknown peak (null plus reason) travel with every
result; a pre-spawn refusal, watchdog abort or failed child is only ever an unaccepted record.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scripts import media_campaign_windows as windows


RUNTIME_GUARDED_ADMISSION = "runtime-guarded"
UNKNOWN_PEAK_REASON = (
    "whole-process transient peak (loader, lazy graphs, allocator/driver headroom, output "
    "coexistence) is not statically derivable; the row is admitted by the runtime watchdog "
    "cap, free-memory reserve and deadline instead"
)


# The child could not start, exited unsuccessfully, or exited leaving invalid evidence.
FAILED_REASONS = frozenset({"child-exit", "spawn-failure", "invalid-evidence"})


class SupervisionError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        # Set by run_guarded: the admission the row ran under (None if the policy was
        # unguarded) and the owned child PID (None when refused before spawn).
        self.admission: dict[str, object] | None = None
        self.pid: int | None = None
        # The live macOS sample that tripped the host-reserve watchdog, when one did.
        self.watchdog_host_memory: dict[str, object] | None = None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass(frozen=True)
class SafetyPolicy:
    backend: str
    deadline_seconds: float
    poll_millis: int
    term_grace_millis: int
    host_free_reserve_bytes: int
    child_footprint_cap_bytes: int
    stdout_cap_bytes: int
    stderr_cap_bytes: int
    event_cap_bytes: int
    cuda_device_uuid: str | None
    gpu_free_reserve_bytes: int | None
    child_gpu_cap_bytes: int | None
    sha256: str
    canonical_bytes: bytes


def load_policy(path: Path) -> SafetyPolicy:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise SupervisionError("invalid-policy", str(error)) from error
    base = {
        "schemaVersion", "backend", "deadlineSeconds", "pollMillis", "termGraceMillis",
        "hostFreeReserveBytes", "childFootprintCapBytes", "stdoutCapBytes",
        "stderrCapBytes", "eventCapBytes",
    }
    cuda = {"cudaDeviceUuid", "gpuFreeReserveBytes", "childGpuCapBytes"}
    if not isinstance(data, dict) or type(data.get("schemaVersion")) is not int or data["schemaVersion"] != 1:
        raise SupervisionError("invalid-policy", "schemaVersion must be 1")
    backend = data.get("backend")
    expected = base | (cuda if backend in {"linux-cuda", "windows-cuda"} else set())
    if backend not in {"darwin-mlx", "linux-cuda", "windows-cuda"} or set(data) != expected:
        raise SupervisionError("invalid-policy", "backend or exact policy fields are invalid")
    deadline = data["deadlineSeconds"]
    if type(deadline) not in (int, float) or deadline <= 0 or deadline >= 2**53 or not math.isfinite(deadline):
        raise SupervisionError("invalid-policy", "deadlineSeconds must be finite and positive")
    for key in expected - {"schemaVersion", "backend", "deadlineSeconds", "cudaDeviceUuid"}:
        value = data[key]
        if type(value) is not int or not 0 < value < 2**63:
            raise SupervisionError("invalid-policy", f"{key} must be a positive safe integer")
    if data["pollMillis"] >= deadline * 1000 or data["termGraceMillis"] >= deadline * 1000:
        raise SupervisionError("invalid-policy", "poll and grace must be below deadline")
    if data["hostFreeReserveBytes"] + data["childFootprintCapBytes"] >= 2**63:
        raise SupervisionError("invalid-policy", "host reserve plus child cap overflows")
    if backend in {"linux-cuda", "windows-cuda"}:
        if not isinstance(data["cudaDeviceUuid"], str) or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", data["cudaDeviceUuid"]):
            raise SupervisionError("invalid-policy", "cudaDeviceUuid must be a GPU UUID")
        if data["gpuFreeReserveBytes"] + data["childGpuCapBytes"] >= 2**63:
            raise SupervisionError("invalid-policy", "GPU reserve plus cap overflows")
    encoded = canonical(data)
    return SafetyPolicy(
        backend, float(deadline), data["pollMillis"], data["termGraceMillis"],
        data["hostFreeReserveBytes"], data["childFootprintCapBytes"],
        data["stdoutCapBytes"], data["stderrCapBytes"], data["eventCapBytes"],
        data.get("cudaDeviceUuid"), data.get("gpuFreeReserveBytes"),
        data.get("childGpuCapBytes"), digest(encoded), encoded,
    )


def _bounded_output(argv: list[str], *, timeout: float = 2.0) -> str:
    try:
        output = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=timeout, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise SupervisionError("probe-failure", f"{argv[0]}: {error}") from error
    if len(output) > 1024 * 1024:
        raise SupervisionError("probe-failure", f"{argv[0]} output exceeds 1 MiB")
    try:
        return output.decode("ascii")
    except UnicodeDecodeError as error:
        raise SupervisionError("probe-failure", f"{argv[0]} output is not ASCII") from error


DARWIN_AVAILABLE_METRIC = "darwin-vm-stat-available-v3"
_DARWIN_COUNTERS = {
    "Pages free": "freePages", "Pages speculative": "speculativePages",
    "Pages purgeable": "purgeablePages", "Pages inactive": "inactivePages",
    "File-backed pages": "fileBackedPages", "Anonymous pages": "anonymousPages",
    "Pages throttled": "throttledPages", "Pages active": "activePages",
}


def _darwin_pages(output: str) -> tuple[int, dict[str, int]]:
    first = output.splitlines()[0] if output else ""
    match = re.search(r"page size of (\d+) bytes", first)
    if not match or int(match.group(1)) < 4096 or int(match.group(1)) & (int(match.group(1)) - 1):
        raise SupervisionError("probe-failure", "vm_stat page size is invalid")
    page_size = int(match.group(1))
    pages = {}
    for line in output.splitlines()[1:]:
        key = line.strip().split(":", 1)[0]
        if key not in _DARWIN_COUNTERS:
            continue
        match = re.fullmatch(r"[^:]+:\s*(\d+)\.", line.strip())
        if not match:
            raise SupervisionError("probe-failure", f"vm_stat {key} is malformed")
        if _DARWIN_COUNTERS[key] in pages:
            raise SupervisionError("probe-failure", "duplicate vm_stat counter")
        pages[_DARWIN_COUNTERS[key]] = int(match.group(1))
    if set(pages) != set(_DARWIN_COUNTERS.values()):
        raise SupervisionError("probe-failure", "vm_stat lacks a host-memory counter")
    return page_size, pages


def darwin_host_memory(page_size: int, pages: dict[str, int]) -> dict[str, object]:
    """The macOS admission and host-reserve watchdog measure, with every component.

    availableBytes = (free + speculative + purgeable + R) * page size, required below 2^63, where
    R = max(0, file-backed - speculative): Activity Monitor's "Cached Files" model. Every
    file-backed page is page cache the kernel can drop (clean) or write back (dirty) without the
    compressor or swap; anonymous pages are never credited. vm_stat prints free (kernel
    free_count minus speculative) and speculative as disjoint lists, and File-backed + Anonymous =
    active + inactive + speculative + throttled, so speculative read-ahead sits inside
    File-backed and is removed from R; purgeable pages are anonymous, so they never overlap R.
    Inactive, anonymous, throttled and active pages are parsed (fail closed) and recorded for
    audit only.

    Known limitation (Activity Monitor semantics): file-backed pages that another process has
    actively mapped -- executables, mmapped model weights of another app -- count as available
    although evicting them makes that process fault them back in. The campaign child's own
    mapped weights are bounded by its phys_footprint cap, not by this measure.

    Mirrors the Rust `campaign_supervisor::HostMemory` and `.github/kv-poc/common.sh`; all three
    are pinned by crates/llm/mlx-llm/testdata/darwin-host-memory-cases.json.
    """
    reclaimable = max(0, pages["fileBackedPages"] - pages["speculativePages"])
    available = (pages["freePages"] + pages["speculativePages"] + pages["purgeablePages"]
                 + reclaimable) * page_size
    if available >= 2**63:
        raise SupervisionError("probe-failure", "vm_stat available bytes overflow")
    return {"metric": DARWIN_AVAILABLE_METRIC, "pageSizeBytes": page_size, **pages,
            "reclaimableFilePages": reclaimable, "availableBytes": available}


def darwin_available(output: str) -> dict[str, object]:
    return darwin_host_memory(*_darwin_pages(output))


def validate_host_memory(value: object) -> dict[str, object]:
    """A recorded darwin measurement must name the metric and recompute exactly from its
    components. Sufficiency is not re-judged: a refused row records its short measurement too."""
    keys = {"metric", "pageSizeBytes", *_DARWIN_COUNTERS.values(), "reclaimableFilePages", "availableBytes"}
    if (not isinstance(value, dict) or set(value) != keys or value["metric"] != DARWIN_AVAILABLE_METRIC
            or any(type(value[key]) is not int or value[key] < 0 for key in keys - {"metric"})
            or value["pageSizeBytes"] < 4096 or value["pageSizeBytes"] & (value["pageSizeBytes"] - 1)):
        raise SupervisionError("invalid-admission", "host memory components are malformed")
    pages = {key: value[key] for key in _DARWIN_COUNTERS.values()}
    if darwin_host_memory(value["pageSizeBytes"], pages) != value:
        raise SupervisionError("invalid-admission", "host memory components do not recompute")
    return value


# Reads of one PID's footprint per watchdog tick: one read plus three immediate retries.
FOOTPRINT_READS_PER_TICK = 4
# Consecutive ticks whose footprint (or host) sample failed before the row stops as
# probe-failure. The other watchdog keeps running on the failed ticks.
FOOTPRINT_FAILED_TICK_LIMIT = 5
# Host reads per preflight or watchdog tick: one read plus three immediate retries.
HOST_READS_PER_TICK = 4
# CUDA free/attribution reads per preflight or watchdog tick: one read plus three retries.
GPU_READS_PER_TICK = 4
_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    """<sys/resource.h> struct rusage_info_v4."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
        "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
        "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
        "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
        "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
        "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")]


_libproc = None


def proc_pid_rusage_footprint(pid: int) -> tuple[int, int]:
    """(phys_footprint, lifetime peak) from one proc_pid_rusage(RUSAGE_INFO_V4) syscall.

    The same kernel ledger /usr/bin/footprint prints as phys_footprint / phys_footprint_peak,
    read without spawning a sampler process. ESRCH (the PID is gone) raises
    ProcessLookupError; every other failure raises OSError carrying its errno.
    """
    library = _load_libproc()
    if library is None:
        raise OSError(errno.ENOSYS, "proc_pid_rusage needs macOS libproc")
    info = _RusageInfoV4()
    if library.proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))  # ESRCH constructs ProcessLookupError
    return info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint


def retry_reads(read: Callable[[], object], reads: int, label: str):
    """One probe measurement, retried immediately; probe-failure once every read failed.

    Mirrors the Rust campaign supervisor: a transient probe failure (a sampler timing out
    under heavy host I/O) is retried within the tick, and the detail keeps the read count
    and the last error. Any other SupervisionError is a real verdict and propagates at once.
    """
    last = SupervisionError("probe-failure", f"no {label} read attempted")
    for _ in range(reads):
        try:
            return read()
        except SupervisionError as error:
            if error.reason != "probe-failure":
                raise
            last = error
    raise SupervisionError("probe-failure", f"{label} probe failed {reads} reads; last error {last.detail}")


class FailedTicks:
    """Consecutive watchdog ticks on which one probe failed every in-tick read.

    The row stops as probe-failure only once FOOTPRINT_FAILED_TICK_LIMIT consecutive ticks
    failed; any successful tick resets the count. The other watchdogs keep running meanwhile.
    """

    def __init__(self, label: str):
        self.label = label
        self.count = 0

    def failed(self, error: SupervisionError) -> None:
        if error.reason != "probe-failure":
            raise error
        self.count += 1
        if self.count >= FOOTPRINT_FAILED_TICK_LIMIT:
            raise SupervisionError(
                "probe-failure",
                f"{self.label} failed on {self.count} consecutive ticks; last error {error.detail}") from error

    def succeeded(self) -> None:
        self.count = 0


def read_host(probe: object) -> tuple[int, dict[str, object] | None]:
    """One host measurement, retried immediately; probe-failure once every read failed."""
    return retry_reads(probe.host_admission, HOST_READS_PER_TICK, "host")


def read_gpu(read: Callable[[], object]):
    """One CUDA (nvidia-smi/NVML) measurement, retried immediately like the host probe."""
    return retry_reads(read, GPU_READS_PER_TICK, "CUDA")


def darwin_phys_footprint(pid: int, *, read: Callable[[int], tuple[int, int]] = proc_pid_rusage_footprint) -> int | None:
    """One tick's phys_footprint for pid, or None when the PID is gone (ESRCH).

    A failed read is retried immediately, FOOTPRINT_READS_PER_TICK reads in all; the
    probe-failure detail keeps the last errno and its message.
    """
    last = OSError("no read attempted")
    for _ in range(FOOTPRINT_READS_PER_TICK):
        try:
            footprint = read(pid)[0]
        except ProcessLookupError:
            return None
        except OSError as error:
            last = error
            continue
        if footprint < 0 or footprint >= 2**63:
            raise SupervisionError("probe-failure", f"process {pid} footprint {footprint} is invalid")
        return footprint
    raise SupervisionError(
        "probe-failure",
        f"proc_pid_rusage({pid}) failed {FOOTPRINT_READS_PER_TICK} reads: "
        f"errno {last.errno}: {last.strerror}")


def linux_free_bytes(output: str) -> int:
    fields = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([A-Za-z_]+):\s*(\d+) kB", line)
        if match:
            fields[match.group(1)] = int(match.group(2)) * 1024
    if fields.get("MemAvailable", 0) <= 0:
        raise SupervisionError("probe-failure", "MemAvailable is unavailable")
    return fields["MemAvailable"]


# Seconds the `ps` fallback may take. The fallback only runs where libproc is unavailable;
# a 2 s budget aborted a Krea row while a 14B load saturated host I/O (SC-20684).
PS_FALLBACK_TIMEOUT_SECONDS = 10.0
# Process-table reads per watchdog tick: one read plus three immediate retries.
PROCESS_TABLE_READS_PER_TICK = 4
_PROC_PIDT_SHORTBSDINFO = 13
_SZOMB = 5


class _ProcBsdShortInfo(ctypes.Structure):
    """<sys/proc_info.h> struct proc_bsdshortinfo."""

    _fields_ = [(name, ctypes.c_uint32) for name in ("pbsi_pid", "pbsi_ppid", "pbsi_pgid", "pbsi_status")] + [
        ("pbsi_comm", ctypes.c_char * 16)] + [(name, ctypes.c_uint32) for name in (
            "pbsi_flags", "pbsi_uid", "pbsi_gid", "pbsi_ruid", "pbsi_rgid", "pbsi_svuid",
            "pbsi_svgid", "pbsi_rfu")]


def _load_libproc():
    """libproc with every symbol this module calls typed, loaded once; None off macOS."""
    global _libproc
    if _libproc is None:
        if sys.platform != "darwin":
            return None
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        library.proc_pid_rusage.restype = ctypes.c_int
        library.proc_listallpids.argtypes = (ctypes.c_void_p, ctypes.c_int)
        library.proc_listallpids.restype = ctypes.c_int
        library.proc_pidinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)
        library.proc_pidinfo.restype = ctypes.c_int
        _libproc = library
    return _libproc


def libproc_process_table(library) -> dict[int, tuple[int, int]]:
    """{pid: (ppid, pgid)} for every live non-zombie process, from libproc syscalls only.

    proc_listallpids, then proc_pidinfo(PROC_PIDT_SHORTBSDINFO) per PID: the same fields
    `ps -axo pid=,ppid=,pgid=,stat=` prints, without spawning a process. PROC_PIDT_SHORTBSDINFO
    is readable for every user's processes, so a setuid descendant is never hidden. A zombie
    (listed, but its read fails ESRCH) or a PID that exited between the listing and its read is
    skipped, as ps's Z rows are; any other failure raises OSError carrying its errno.
    """
    capacity = library.proc_listallpids(None, 0)
    if capacity <= 0:
        code = ctypes.get_errno()
        raise OSError(code, f"proc_listallpids: {os.strerror(code)}")
    while True:
        capacity += 256
        buffer = (ctypes.c_int * capacity)()
        count = library.proc_listallpids(buffer, ctypes.sizeof(buffer))
        if count <= 0:
            code = ctypes.get_errno()
            raise OSError(code, f"proc_listallpids: {os.strerror(code)}")
        if count < capacity:
            break  # A full buffer may have truncated the listing.
    result = {}
    info = _ProcBsdShortInfo()
    for pid in buffer[:count]:
        if pid <= 0:
            continue  # kernel_task; ps does not list it
        if library.proc_pidinfo(pid, _PROC_PIDT_SHORTBSDINFO, 0, ctypes.byref(info),
                                ctypes.sizeof(info)) != ctypes.sizeof(info):
            code = ctypes.get_errno()
            if code == errno.ESRCH:
                continue
            raise OSError(code, f"proc_pidinfo({pid}): {os.strerror(code)}")
        if info.pbsi_status != _SZOMB:
            result[pid] = (info.pbsi_ppid, info.pbsi_pgid)
    return result


def ps_process_table() -> dict[int, tuple[int, int]]:
    """The `ps` fallback for hosts without libproc (Linux)."""
    output = _bounded_output(["/bin/ps", "-axo", "pid=,ppid=,pgid=,stat="], timeout=PS_FALLBACK_TIMEOUT_SECONDS)
    result = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4:
            raise SupervisionError("probe-failure", "ps process-group row is malformed")
        pid, parent, group, state = parts
        if not pid.isdecimal() or not parent.isdecimal() or not group.isdecimal():
            raise SupervisionError("probe-failure", "ps process-group PID is malformed")
        if not state.startswith("Z"):
            result[int(pid)] = (int(parent), int(group))
    return result


def _read_process_table() -> dict[int, tuple[int, int]]:
    try:
        library = _load_libproc()
    except (OSError, AttributeError):
        library = None  # libproc absent or missing a symbol: fall back to ps
    if library is None:
        return ps_process_table()
    try:
        return libproc_process_table(library)
    except OSError as error:
        raise SupervisionError("probe-failure", f"libproc process table: errno {error.errno}: {error.strerror}") from error


def _process_table() -> dict[int, tuple[int, int]]:
    """The owned-tree snapshot, read with immediate retries like every watchdog probe."""
    return retry_reads(_read_process_table, PROCESS_TABLE_READS_PER_TICK, "process table")


def _group_pids(pgid: int) -> set[int]:
    return {pid for pid, (_parent, group) in _process_table().items() if group == pgid}


def _expand_owned(table: dict[int, tuple[int, int]], known: set[int]) -> set[int]:
    """Track descendants by ancestry, even if a child changes process group.

    A known PID absent from the snapshot has exited (or is a zombie) and is forgotten, so a
    later process reusing that PID is never mistaken for a descendant, counted or signalled.
    """
    known.intersection_update(table)
    changed = True
    while changed:
        changed = False
        for pid, (parent, _group) in table.items():
            if parent in known and pid not in known:
                known.add(pid)
                changed = True
    return set(known)


@dataclass
class PosixTree:
    """A POSIX row's owned processes: the child's process group plus every descendant observed
    by ancestry, including one that moved to a group of its own -- system_profiler runs its
    `-nospawn` collector in a new process group (SC-20684 W2 run 36808330752). Such a
    descendant is measured, awaited and reaped like any other. The POSIX counterpart of
    WindowsJob.members()."""

    pgid: int
    known: set[int]

    def snapshot(self) -> tuple[set[int], set[int]]:
        """(process-group members, ancestry-owned descendants) from one process table."""
        table = _process_table()
        owned = _expand_owned(table, self.known)
        return {pid for pid, (_parent, group) in table.items() if group == self.pgid}, owned

    def members(self) -> set[int]:
        group, owned = self.snapshot()
        return group | owned


class SystemProbe:
    def __init__(self, policy: SafetyPolicy):
        self.policy = policy
        actual = {"Darwin": "darwin-mlx", "Linux": "linux-cuda",
                  "Windows": "windows-cuda"}.get(platform.system(), "unsupported")
        if actual != policy.backend:
            raise SupervisionError("unsupported-host", f"{policy.backend} cannot run on {actual}")
        try:
            self._smi = windows.trusted_nvidia_smi() if actual == "windows-cuda" else "nvidia-smi"
        except windows.WindowsJobError as error:
            raise SupervisionError("probe-failure", str(error)) from error

    def host_admission(self) -> tuple[int, dict[str, object] | None]:
        """Pre-spawn host measure and, on macOS, the components it was computed from."""
        if self.policy.backend == "darwin-mlx":
            host = darwin_available(_bounded_output(["/usr/bin/vm_stat"]))
            return int(host["availableBytes"]), host
        return self.host_free(), None

    def host_free(self) -> int:
        """Linux/Windows host free memory; macOS measures through host_admission."""
        if self.policy.backend == "windows-cuda":
            try:
                return windows.host_free_bytes()
            except windows.WindowsJobError as error:
                raise SupervisionError("probe-failure", str(error)) from error
        free = linux_free_bytes(Path("/proc/meminfo").read_text(encoding="ascii"))
        # A delegated cgroup limit can be smaller than host MemAvailable.
        try:
            groups = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
            group = next(line.split("::", 1)[1] for line in groups if line.startswith("0::"))
            root = Path("/sys/fs/cgroup") / group.lstrip("/")
            cap = (root / "memory.max").read_text(encoding="ascii").strip()
            if cap != "max":
                used = int((root / "memory.current").read_text(encoding="ascii").strip())
                free = min(free, max(0, int(cap) - used))
        except (OSError, StopIteration, ValueError) as error:
            raise SupervisionError("probe-failure", f"cgroup memory budget unavailable: {error}") from error
        return free

    def tree_footprint(self, owner: PosixTree | windows.WindowsJob) -> int | None:
        """The owned tree's footprint, or None when every owned process vanished mid-sample."""
        if self.policy.backend == "windows-cuda":
            try:
                return owner.footprint_bytes()
            except windows.WindowsJobError as error:
                raise SupervisionError("probe-failure", str(error)) from error
        pids = owner.members()
        if not pids:
            return None
        total = 0
        sampled = False
        for pid in pids:
            if self.policy.backend == "darwin-mlx":
                footprint = darwin_phys_footprint(pid)
                if footprint is None:
                    continue  # exited after the process-table snapshot
                total += footprint
                sampled = True
            else:
                try:
                    output = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
                except OSError as error:
                    raise SupervisionError("probe-failure", f"process {pid} RSS unavailable: {error}") from error
                match = re.search(r"^VmRSS:\s*(\d+) kB$", output, re.MULTILINE)
                if not match:
                    raise SupervisionError("probe-failure", f"process {pid} RSS malformed")
                total += int(match.group(1)) * 1024
                sampled = True
        return total if sampled else None

    def gpu_free_and_tree_bytes(self, owner: PosixTree | windows.WindowsJob) -> tuple[int, int]:
        free = self.gpu_free()
        assert self.policy.cuda_device_uuid
        uuid = self.policy.cuda_device_uuid
        pids = owner.members()
        usage = _bounded_output([self._smi, "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                                 "--format=csv,noheader,nounits"])
        used = 0
        for line in usage.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 3 or not parts[0].isdecimal() or not parts[1].isdecimal():
                raise SupervisionError("probe-failure", "CUDA process-memory row is malformed")
            if int(parts[0]) in pids:
                if parts[2] != uuid:
                    raise SupervisionError("probe-failure", "child used an unexpected GPU")
                used += int(parts[1]) * 1024**2
        return free, used

    def gpu_free(self) -> int:
        assert self.policy.cuda_device_uuid
        rows = _bounded_output([self._smi, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"])
        values = []
        for line in rows.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 2 or not parts[1].isdecimal():
                raise SupervisionError("probe-failure", "GPU free-memory row is malformed")
            if parts[0] == self.policy.cuda_device_uuid:
                values.append(int(parts[1]) * 1024**2)
        if len(values) != 1:
            raise SupervisionError("probe-failure", "selected GPU UUID is unavailable or duplicate")
        return values[0]


def _stop_tree(child: subprocess.Popen[bytes], grace_seconds: float, known: set[int]) -> None:
    def signal_known(sig: signal.Signals) -> None:
        table = _process_table()
        live = _expand_owned(table, known)
        for pid in live - {child.pid}:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    try:
        os.killpg(child.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if _group_pids(child.pid):
            # A protected/escaped member is not a successful cleanup.
            try:
                child.terminate()
            except (ProcessLookupError, PermissionError):
                pass
    signal_known(signal.SIGTERM)
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # Root exit is not proof of descendant exit; always KILL remaining group members.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if _group_pids(child.pid) and child.poll() is None:
            try:
                child.kill()
            except (ProcessLookupError, PermissionError):
                pass
    signal_known(signal.SIGKILL)
    child.wait(timeout=grace_seconds)
    until = time.monotonic() + grace_seconds
    while time.monotonic() < until:
        table = _process_table()
        if not _expand_owned(table, known) and not any(group == child.pid for _parent, group in table.values()):
            return
        time.sleep(0.01)
    raise SupervisionError("cleanup-failure", "owned process group still has live descendants")


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def runtime_guarded_admission(
    policy: SafetyPolicy, *, static_floor_host_bytes: int | None = None,
    static_floor_gpu_bytes: int | None = None,
) -> dict[str, object]:
    """Admit a row by its runtime guards rather than by a static whole-process peak proof.

    Refuses before spawn unless every supervisor guard is configured, and when a caller has a
    conservative static floor, unless that floor fits under the cap. The host/device free
    probe against cap plus reserve happens in run_guarded immediately before spawn.
    """
    cuda = policy.backend in {"linux-cuda", "windows-cuda"}
    guards = [policy.deadline_seconds, policy.poll_millis, policy.term_grace_millis,
              policy.host_free_reserve_bytes, policy.child_footprint_cap_bytes,
              policy.stdout_cap_bytes, policy.stderr_cap_bytes, policy.event_cap_bytes]
    if cuda:
        guards += [policy.gpu_free_reserve_bytes, policy.child_gpu_cap_bytes]
    if (policy.backend not in {"darwin-mlx", "linux-cuda", "windows-cuda"}
            or not all(_positive(value) for value in guards)
            or policy.poll_millis >= policy.deadline_seconds * 1000
            or policy.term_grace_millis >= policy.deadline_seconds * 1000
            or (cuda and not policy.cuda_device_uuid)):
        raise SupervisionError("unguarded-policy", "runtime guards are not all configured; refusing before spawn")
    floors = ((static_floor_host_bytes, policy.child_footprint_cap_bytes, "host"),
              (static_floor_gpu_bytes, policy.child_gpu_cap_bytes if cuda else None, "CUDA"))
    for floor, cap, label in floors:
        if floor is None:
            continue
        if type(floor) is not int or floor <= 0 or cap is None:
            raise SupervisionError("invalid-admission", f"{label} static floor is malformed")
        if floor > cap:
            raise SupervisionError("preflight-memory", f"{label} static floor exceeds child cap")
    return {
        "mode": RUNTIME_GUARDED_ADMISSION,
        "backend": policy.backend,
        "policySha256": policy.sha256,
        "deadlineSeconds": policy.deadline_seconds,
        "pollMillis": policy.poll_millis,
        "termGraceMillis": policy.term_grace_millis,
        "hostFreeReserveBytes": policy.host_free_reserve_bytes,
        "childFootprintCapBytes": policy.child_footprint_cap_bytes,
        "cudaDeviceUuid": policy.cuda_device_uuid,
        "gpuFreeReserveBytes": policy.gpu_free_reserve_bytes,
        "childGpuCapBytes": policy.child_gpu_cap_bytes,
        "staticFloorHostBytes": static_floor_host_bytes,
        "staticFloorGpuBytes": static_floor_gpu_bytes,
        "wholeProcessPeakBoundBytes": None,
        "wholeProcessPeakUnknownReason": UNKNOWN_PEAK_REASON,
    }


def validate_admission(value: object, *, policy_sha256: str) -> dict[str, object]:
    """Reject a sealed record whose row was not admitted by the resume policy's runtime guards."""
    if (not isinstance(value, dict) or value.get("mode") != RUNTIME_GUARDED_ADMISSION
            or value.get("policySha256") != policy_sha256
            or value.get("backend") not in {"darwin-mlx", "linux-cuda", "windows-cuda"}
            or not _positive(value.get("childFootprintCapBytes"))
            or not _positive(value.get("hostFreeReserveBytes"))
            or not _positive(value.get("deadlineSeconds"))
            or value.get("wholeProcessPeakBoundBytes") is not None
            or not isinstance(value.get("wholeProcessPeakUnknownReason"), str)
            or not value["wholeProcessPeakUnknownReason"]):
        raise SupervisionError("invalid-admission", "record lacks a runtime-guarded admission")
    if value["backend"] in {"linux-cuda", "windows-cuda"} and (
            not isinstance(value.get("cudaDeviceUuid"), str) or not value["cudaDeviceUuid"]
            or not _positive(value.get("gpuFreeReserveBytes"))
            or not _positive(value.get("childGpuCapBytes"))):
        raise SupervisionError("invalid-admission", "CUDA admission lacks its device guards")
    if value["backend"] == "darwin-mlx":
        validate_host_memory(value.get("hostMemoryComponents"))
    elif "hostMemoryComponents" in value:
        raise SupervisionError("invalid-admission", "only a macOS admission records vm_stat components")
    for key, cap in (("staticFloorHostBytes", "childFootprintCapBytes"),
                     ("staticFloorGpuBytes", "childGpuCapBytes")):
        floor = value.get(key)
        if floor is not None and (type(floor) is not int or not _positive(value.get(cap))
                                  or not 0 < floor <= value[cap]):
            raise SupervisionError("invalid-admission", f"recorded {key} exceeds its cap")
    return value


def write_unaccepted_record(path: Path, *, kind: str, coordinate: str, error: SupervisionError) -> Path:
    """Persist a refused, aborted or failed row as a sealed, explicitly unaccepted record."""
    if error.reason in FAILED_REASONS:
        outcome = "failed"
    else:
        outcome = "refused" if error.pid is None else "aborted"
    encoded = canonical({
        "schemaVersion": 1, "kind": kind, "coordinate": coordinate, "accepted": False,
        "outcome": outcome, "reason": error.reason, "detail": error.detail, "pid": error.pid,
        "admission": error.admission,
        **({"watchdogHostMemory": error.watchdog_host_memory}
           if error.watchdog_host_memory is not None else {}),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.with_name(f"{path.name}.sha256").write_text(f"{digest(encoded)}  {path.name}\n", encoding="utf-8")
    return path


# A campaign parent halted between rows by its operator stop file exits with this status
# (sysexits EX_TEMPFAIL): distinct from success and every refusal; the same command resumes.
OPERATOR_STOP_EXIT_CODE = 75
OPERATOR_STOP_FILE_NAME = "STOP"


class OperatorStop(Exception):
    """The operator asked the parent to stop before starting row ``before``."""

    def __init__(self, *, before: str, index: int, total: int, record: Path):
        super().__init__(f"stopped by operator before row {index + 1}/{total} ({before}); status {record}")
        self.before = before
        self.index = index
        self.total = total
        self.record = record


def operator_stop_files(stop_file: Path | None, resume_dir: Path) -> tuple[Path, ...]:
    """Always ``<resume-dir>/STOP``, plus ``--stop-file`` when given; either present stops the
    campaign. Stop files are never part of a resume identity."""
    default = Path(resume_dir) / OPERATOR_STOP_FILE_NAME
    if stop_file is None or Path(stop_file) == default:
        return (default,)
    return (default, Path(stop_file))


def is_operator_stop_entry(resume_dir: Path, stop_files: tuple[Path, ...], name: str) -> bool:
    return name == OPERATOR_STOP_FILE_NAME or Path(resume_dir) / name in tuple(Path(path) for path in stop_files)


def _stop_present(path: Path) -> bool:
    """Anything at the path (even a dangling symlink) is present; only a missing path is absent,
    every other stat failure propagates."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _write_new_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def check_operator_stop(stop_files: tuple[Path, ...] | None, logs: Path, *, kind: str, before: str, index: int, total: int) -> None:
    """Called by a parent between rows, just before spawning row ``index``; never signals a
    running child. When any stop file exists, writes a sealed, never-overwritten
    ``logs/operator-stop.attempt-<n>.json`` status record (plus ``.sha256``) and raises
    :class:`OperatorStop`."""
    present = [str(path) for path in (stop_files or ()) if _stop_present(Path(path))]
    if not present:
        return
    logs.mkdir(parents=True, exist_ok=True)
    encoded = canonical({
        "schemaVersion": 1, "kind": kind, "status": "stopped-by-operator",
        "beforeRow": index, "beforeRowSlug": before, "rowsTotal": total, "rowsAccepted": index,
        "stopFiles": present, "recordedAtUnixNs": time.time_ns(),
        "resume": "remove the stop file and rerun the same command with the same resume directory",
    })
    attempt = 0
    while True:
        record = logs / f"operator-stop.attempt-{attempt}.json"
        try:
            _write_new_durable(record, encoded)
            break
        except FileExistsError:
            attempt += 1
    _write_new_durable(record.with_name(f"{record.name}.sha256"), f"{digest(encoded)}  {record.name}\n".encode("utf-8"))
    raise OperatorStop(before=before, index=index, total=total, record=record)


@dataclass(frozen=True)
class RunResult:
    pid: int
    returncode: int
    peak_host_bytes: int
    peak_gpu_bytes: int | None
    host_free_at_launch: int
    gpu_free_at_launch: int | None
    elapsed_seconds: float
    samples: tuple[dict[str, object], ...]
    admission: dict[str, object]


def run_guarded(
    argv: list[str], *, cwd: Path, env: dict[str, str], policy: SafetyPolicy,
    stdout_path: Path, stderr_path: Path, static_floor_host_bytes: int | None = None,
    static_floor_gpu_bytes: int | None = None, event_path: Path | None = None,
    probe: object | None = None, clock: Callable[[], float] = time.monotonic,
    on_spawn: Callable[[int, int | None], None] | None = None,
) -> RunResult:
    """Run one row under runtime-guarded admission.

    Any SupervisionError carries the admission (None only for an unguarded policy) and the
    owned child PID (None when refused before spawn) so callers can seal it as unaccepted.
    """
    admission = None
    spawned: list[int] = []

    def record_spawn(pid: int, pgid: int | None) -> None:
        spawned.append(pid)
        if on_spawn is not None:
            on_spawn(pid, pgid)

    try:
        admission = runtime_guarded_admission(
            policy, static_floor_host_bytes=static_floor_host_bytes,
            static_floor_gpu_bytes=static_floor_gpu_bytes,
        )
        return _run_admitted(
            argv, cwd=cwd, env=env, policy=policy, admission=admission,
            stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
            probe=probe, clock=clock, on_spawn=record_spawn,
        )
    except SupervisionError as error:
        error.admission = admission
        error.pid = spawned[0] if spawned else None
        raise


def _run_admitted(
    argv: list[str], *, cwd: Path, env: dict[str, str], policy: SafetyPolicy,
    admission: dict[str, object], stdout_path: Path, stderr_path: Path,
    event_path: Path | None, probe: object | None, clock: Callable[[], float],
    on_spawn: Callable[[int, int | None], None],
) -> RunResult:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise SupervisionError("invalid-command", "argv is empty or malformed")
    if (policy.backend == "windows-cuda") != (os.name == "nt"):
        raise SupervisionError("unsupported-host", "safety backend differs from process host")
    if stdout_path == stderr_path or any(path.exists() for path in (stdout_path, stderr_path)):
        raise SupervisionError("invalid-log", "bounded log paths must be distinct and fresh")
    probe = probe if probe is not None else SystemProbe(policy)
    started = clock()
    # On macOS the admission measure counts reclaimable clean file cache (darwin_host_memory),
    # and the live reserve watchdog below uses the same measure, so page cache alone never
    # aborts an admitted row. The components are recorded in the admission itself, so refused,
    # aborted and accepted rows all carry the decision's inputs.
    try:
        host_free, host_memory = read_host(probe)
    except SupervisionError as error:
        raise SupervisionError(error.reason, f"preflight {error.detail}") from error
    if policy.backend == "darwin-mlx":
        admission["hostMemoryComponents"] = host_memory
    if host_free < policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes:
        raise SupervisionError("preflight-memory", "host available is below reserve plus child cap")
    gpu_free = None
    if policy.backend in {"linux-cuda", "windows-cuda"}:
        try:
            gpu_free = read_gpu(probe.gpu_free)
        except SupervisionError as error:
            raise SupervisionError(error.reason, f"preflight {error.detail}") from error
        if gpu_free < policy.gpu_free_reserve_bytes + policy.child_gpu_cap_bytes:
            raise SupervisionError("preflight-memory", "CUDA free is below reserve plus child cap")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        job = None
        if policy.backend == "windows-cuda":
            try:
                job = windows.WindowsJob()
                # CREATE_SUSPENDED: no descendant can start before Job ownership is installed.
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                         creationflags=0x00000004)
                try:
                    job.assign_and_resume(child)
                except BaseException:
                    child.kill()  # Still suspended if assignment failed.
                    child.wait(timeout=policy.term_grace_millis / 1000)
                    raise
            except BaseException as error:
                if job is not None:
                    job.close()
                raise SupervisionError("spawn-failure", str(error)) from error
        else:
            try:
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                         start_new_session=True)
            except OSError as error:
                raise SupervisionError("spawn-failure", str(error)) from error
        known = {child.pid}
        owner = job if job is not None else PosixTree(child.pid, known)
        peak_host = 0
        peak_gpu = 0
        samples: list[dict[str, object]] = []
        failed_tree_ticks = FailedTicks("process-tree probe")
        failed_footprint_ticks = FailedTicks("child footprint probe")
        failed_host_ticks = FailedTicks("live host probe")
        failed_gpu_ticks = FailedTicks("CUDA probe")
        def check_artifact_caps() -> None:
            if stdout_path.stat().st_size > policy.stdout_cap_bytes or stderr_path.stat().st_size > policy.stderr_cap_bytes:
                raise SupervisionError("log-cap", "child exceeded a bounded transcript file")
            if event_path is not None:
                if event_path.is_symlink():
                    raise SupervisionError("event-path", "observer event file may not be a symlink")
                if event_path.exists() and event_path.stat().st_size > policy.event_cap_bytes:
                    raise SupervisionError("event-cap", "child exceeded a bounded observer event file")

        def owned_processes() -> tuple[set[int], set[int]]:
            if job is not None:
                try:
                    pids = job.members()
                except windows.WindowsJobError as error:
                    raise SupervisionError("probe-failure", str(error)) from error
                return pids, pids
            return owner.snapshot()

        def finished_result(status: int | None, pids: set[int], owned: set[int]) -> RunResult | None:
            if status is None or pids or owned:
                return None
            if clock() - started >= policy.deadline_seconds:
                raise SupervisionError("deadline", "child exceeded the hard wall-clock deadline")
            # The root exit and an empty owned tree are both required. Recheck
            # files here because the child can write between the loop's first
            # cap check and this terminal observation.
            check_artifact_caps()
            if status != 0:
                raise SupervisionError("child-exit", f"child exited with status {status}")
            if policy.backend == "windows-cuda" and peak_gpu <= 0:
                raise SupervisionError("probe-failure", "CUDA child had no attributable GPU-memory sample")
            child.wait()
            return RunResult(child.pid, status, peak_host, peak_gpu if gpu_free is not None else None,
                             host_free, gpu_free, clock() - started, tuple(samples), admission)

        try:
            # Windows ownership is a Job, not a POSIX process group.
            on_spawn(child.pid, None if job is not None else child.pid)
            while True:
                if clock() - started >= policy.deadline_seconds:
                    raise SupervisionError("deadline", "child exceeded the hard wall-clock deadline")
                check_artifact_caps()
                status = child.poll()
                try:
                    pids, owned = owned_processes()
                except SupervisionError as error:
                    # The host watchdog keeps running on a failed process-tree tick; the
                    # exit, footprint and CUDA checks wait for the next tree snapshot.
                    failed_tree_ticks.failed(error)
                    pids = None
                else:
                    failed_tree_ticks.succeeded()
                    finished = finished_result(status, pids, owned)
                    if finished is not None:
                        return finished
                try:
                    available, live_host = read_host(probe)
                except SupervisionError as error:
                    # The child footprint watchdog keeps running on a failed host tick.
                    failed_host_ticks.failed(error)
                    available, live_host = None, None
                else:
                    failed_host_ticks.succeeded()
                if available is not None and available < policy.host_free_reserve_bytes:
                    metric = live_host["metric"] if live_host is not None else "host-free"
                    error = SupervisionError(
                        "host-memory",
                        f"host available {available} bytes ({metric}) fell below reserve "
                        f"{policy.host_free_reserve_bytes} bytes")
                    error.watchdog_host_memory = live_host
                    raise error
                if pids:
                    try:
                        footprint = probe.tree_footprint(owner)
                    except SupervisionError as error:
                        # A short-lived child may exit after the ownership
                        # snapshot and before its footprint is read.
                        try:
                            finished = finished_result(child.poll(), *owned_processes())
                        except SupervisionError as recheck:
                            if recheck.reason != "probe-failure":
                                raise
                            finished = None  # the tree recheck failed too: a failed tick
                        if finished is not None:
                            return finished
                        # A live root, owned descendant, or uncertain ownership
                        # stops the row only once the failure is sustained.
                        failed_footprint_ticks.failed(error)
                        footprint = None
                    else:
                        # None: every owned process was gone (ESRCH); the next
                        # tick's poll and ownership snapshot classify the exit.
                        if footprint is not None:
                            failed_footprint_ticks.succeeded()
                    if footprint is not None:
                        peak_host = max(peak_host, footprint)
                        samples.append({"phase": "process-sample", "sample_kind": "process",
                                        "peak_bytes": footprint, "at_ns": time.time_ns()})
                        if len(samples) * 160 > policy.event_cap_bytes:
                            raise SupervisionError("event-cap", "process sample spool exceeded cap")
                        if footprint > policy.child_footprint_cap_bytes:
                            raise SupervisionError("child-footprint", "owned tree exceeded child cap")
                    if policy.backend in {"linux-cuda", "windows-cuda"}:
                        try:
                            device_free, device_used = read_gpu(lambda: probe.gpu_free_and_tree_bytes(owner))
                        except SupervisionError as error:
                            failed_gpu_ticks.failed(error)
                            time.sleep(policy.poll_millis / 1000)
                            continue
                        failed_gpu_ticks.succeeded()
                        peak_gpu = max(peak_gpu, device_used)
                        if device_free < policy.gpu_free_reserve_bytes or device_used > policy.child_gpu_cap_bytes:
                            raise SupervisionError("device-memory", "owned CUDA use or free reserve exceeded cap")
                time.sleep(policy.poll_millis / 1000)
        except BaseException:
            try:
                if job is not None:
                    job.terminate_and_reap(child, policy.term_grace_millis / 1000)
                else:
                    _stop_tree(child, policy.term_grace_millis / 1000, known)
            finally:
                for path, cap in ((stdout_path, policy.stdout_cap_bytes),
                                  (stderr_path, policy.stderr_cap_bytes),
                                  (event_path, policy.event_cap_bytes)):
                    if path is not None and path.is_file() and not path.is_symlink() and path.stat().st_size > cap:
                        with path.open("r+b") as stream:
                            stream.truncate(cap)
            raise
        finally:
            if job is not None:
                job.close()
                try:
                    child.wait(timeout=policy.term_grace_millis / 1000)
                except subprocess.TimeoutExpired as error:
                    raise SupervisionError("cleanup-failure", "Job close did not reap root") from error

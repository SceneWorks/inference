#!/usr/bin/env python3
"""Small fail-closed process supervisor for the SC-20684/86 media campaigns.

The caller supplies a source-justified peak bound. Sampling cannot establish that bound:
the process can allocate beyond a policy cap between polls. This module never starts a child
without it, a mandatory policy, and a fresh host/device admission probe.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scripts import media_campaign_windows as windows


class SupervisionError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


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


def darwin_free_bytes(output: str) -> int:
    first = output.splitlines()[0] if output else ""
    match = re.search(r"page size of (\d+) bytes", first)
    if not match or int(match.group(1)) < 4096 or int(match.group(1)) & (int(match.group(1)) - 1):
        raise SupervisionError("probe-failure", "vm_stat page size is invalid")
    page_size = int(match.group(1))
    pages = {}
    for line in output.splitlines()[1:]:
        match = re.fullmatch(r"(Pages free|Pages speculative):\s*(\d+)\.", line.strip())
        if match:
            if match.group(1) in pages:
                raise SupervisionError("probe-failure", "duplicate vm_stat counter")
            pages[match.group(1)] = int(match.group(2))
    if set(pages) != {"Pages free", "Pages speculative"}:
        raise SupervisionError("probe-failure", "vm_stat lacks free or speculative pages")
    return (pages["Pages free"] + pages["Pages speculative"]) * page_size


def darwin_footprint_bytes(output: str) -> int:
    matches = re.findall(r"^\s*phys_footprint:\s*(\d+(?:\.\d+)?)\s*(B|K|KB|M|MB|G|GB)\s*$",
                         output, re.MULTILINE)
    if len(matches) != 1:
        raise SupervisionError("probe-failure", "footprint has no unique phys_footprint")
    value, unit = matches[0]
    scale = {"B": 1, "K": 1024, "KB": 1024, "M": 1024**2,
             "MB": 1024**2, "G": 1024**3, "GB": 1024**3}[unit]
    result = math.ceil(float(value) * scale)
    if result < 0 or result >= 2**63:
        raise SupervisionError("probe-failure", "footprint value is invalid")
    return result


def linux_free_bytes(output: str) -> int:
    fields = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([A-Za-z_]+):\s*(\d+) kB", line)
        if match:
            fields[match.group(1)] = int(match.group(2)) * 1024
    if fields.get("MemAvailable", 0) <= 0:
        raise SupervisionError("probe-failure", "MemAvailable is unavailable")
    return fields["MemAvailable"]


def _process_table() -> dict[int, tuple[int, int]]:
    output = _bounded_output(["/bin/ps", "-axo", "pid=,ppid=,pgid=,stat="])
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


def _group_pids(pgid: int) -> set[int]:
    return {pid for pid, (_parent, group) in _process_table().items() if group == pgid}


def _expand_owned(table: dict[int, tuple[int, int]], known: set[int]) -> set[int]:
    """Track descendants already observed, even if a child changes process group."""
    changed = True
    while changed:
        changed = False
        for pid, (parent, _group) in table.items():
            if parent in known and pid not in known:
                known.add(pid)
                changed = True
    return {pid for pid in known if pid in table}


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

    def host_free(self) -> int:
        if self.policy.backend == "darwin-mlx":
            return darwin_free_bytes(_bounded_output(["/usr/bin/vm_stat"]))
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

    def tree_footprint(self, owner: int | windows.WindowsJob) -> int:
        if self.policy.backend == "windows-cuda":
            try:
                return owner.footprint_bytes()
            except windows.WindowsJobError as error:
                raise SupervisionError("probe-failure", str(error)) from error
        pgid = owner
        pids = _group_pids(pgid)
        if not pids:
            raise SupervisionError("probe-failure", "owned process group disappeared during footprint sample")
        total = 0
        for pid in pids:
            if self.policy.backend == "darwin-mlx":
                total += darwin_footprint_bytes(_bounded_output(["/usr/bin/footprint", "-p", str(pid), "-f", "bytes"]))
            else:
                try:
                    output = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
                except OSError as error:
                    raise SupervisionError("probe-failure", f"process {pid} RSS unavailable: {error}") from error
                match = re.search(r"^VmRSS:\s*(\d+) kB$", output, re.MULTILINE)
                if not match:
                    raise SupervisionError("probe-failure", f"process {pid} RSS malformed")
                total += int(match.group(1)) * 1024
        return total

    def gpu_free_and_tree_bytes(self, owner: int | windows.WindowsJob) -> tuple[int, int]:
        free = self.gpu_free()
        assert self.policy.cuda_device_uuid
        uuid = self.policy.cuda_device_uuid
        pids = owner.members() if self.policy.backend == "windows-cuda" else _group_pids(owner)
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


def run_guarded(
    argv: list[str], *, cwd: Path, env: dict[str, str], policy: SafetyPolicy,
    stdout_path: Path, stderr_path: Path, source_peak_host_bytes: int | None,
    source_peak_gpu_bytes: int | None = None, event_path: Path | None = None,
    probe: object | None = None, clock: Callable[[], float] = time.monotonic,
    on_spawn: Callable[[int, int | None], None] | None = None,
) -> RunResult:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise SupervisionError("invalid-command", "argv is empty or malformed")
    if (policy.backend == "windows-cuda") != (os.name == "nt"):
        raise SupervisionError("unsupported-host", "safety backend differs from process host")
    if type(source_peak_host_bytes) is not int or source_peak_host_bytes <= 0:
        raise SupervisionError("unbounded-source", "no source-backed host peak bound; refusing before spawn")
    if source_peak_host_bytes > policy.child_footprint_cap_bytes:
        raise SupervisionError("preflight-memory", "source host peak exceeds child footprint cap")
    if policy.backend in {"linux-cuda", "windows-cuda"} and (
        type(source_peak_gpu_bytes) is not int or source_peak_gpu_bytes <= 0
        or source_peak_gpu_bytes > policy.child_gpu_cap_bytes
    ):
        raise SupervisionError("unbounded-source", "CUDA peak is unbounded or exceeds device cap")
    if stdout_path == stderr_path or any(path.exists() for path in (stdout_path, stderr_path)):
        raise SupervisionError("invalid-log", "bounded log paths must be distinct and fresh")
    probe = probe if probe is not None else SystemProbe(policy)
    started = clock()
    host_free = probe.host_free()
    if host_free < policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes:
        raise SupervisionError("preflight-memory", "host free is below reserve plus child cap")
    gpu_free = None
    if policy.backend in {"linux-cuda", "windows-cuda"}:
        gpu_free = probe.gpu_free()
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
            child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=True)
        owner = job if job is not None else child.pid
        known = {child.pid}
        peak_host = 0
        peak_gpu = 0
        samples: list[dict[str, object]] = []
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
            table = _process_table()
            owned = _expand_owned(table, known)
            pids = {pid for pid, (_parent, group) in table.items() if group == child.pid}
            if any(table[pid][1] != child.pid for pid in owned):
                raise SupervisionError("process-escape", "observed descendant left the owned process group")
            return pids, owned

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
                             host_free, gpu_free, clock() - started, tuple(samples))

        try:
            if on_spawn is not None:
                # Windows ownership is a Job, not a POSIX process group.
                on_spawn(child.pid, None if job is not None else child.pid)
            while True:
                if clock() - started >= policy.deadline_seconds:
                    raise SupervisionError("deadline", "child exceeded the hard wall-clock deadline")
                check_artifact_caps()
                status = child.poll()
                pids, owned = owned_processes()
                finished = finished_result(status, pids, owned)
                if finished is not None:
                    return finished
                free = probe.host_free()
                if free < policy.host_free_reserve_bytes:
                    raise SupervisionError("host-memory", "host free fell below reserve")
                if pids:
                    try:
                        footprint = probe.tree_footprint(owner)
                    except SupervisionError:
                        # A short-lived child may exit after the ownership
                        # snapshot and before the footprint subprocess runs.
                        # A live root, owned descendant, or uncertain ownership
                        # retains the original fail-closed probe error.
                        finished = finished_result(child.poll(), *owned_processes())
                        if finished is not None:
                            return finished
                        raise
                    peak_host = max(peak_host, footprint)
                    samples.append({"phase": "process-sample", "sample_kind": "process",
                                    "peak_bytes": footprint, "at_ns": time.time_ns()})
                    if len(samples) * 160 > policy.event_cap_bytes:
                        raise SupervisionError("event-cap", "process sample spool exceeded cap")
                    if footprint > policy.child_footprint_cap_bytes:
                        raise SupervisionError("child-footprint", "owned tree exceeded child cap")
                    if policy.backend in {"linux-cuda", "windows-cuda"}:
                        device_free, device_used = probe.gpu_free_and_tree_bytes(owner)
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

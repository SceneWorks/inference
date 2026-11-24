"""Proof-only WDDM attribution from the owned child's live CUDA pool.

The ordinary supervisor still measures exact-card free/global use and enforces
its caps. This adapter supplies attribution only when WDDM reports N/A for an
exact PID/UUID row belonging to the supervised Windows Job.
"""
import json
import os
from pathlib import Path
import stat
import time

from scripts.ci.wddm_process_memory import DedicatedSamples, sampled_peak
from scripts.media_campaign_supervisor import SupervisionError, SystemProbe

_MISSING = object()


def _proof_file_identity(info):
    # Windows reparse points can be directory junctions or other links that
    # Path.is_symlink() does not recognize.
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
            getattr(info, "st_file_attributes", 0) & 0x400 or
            info.st_ino == 0 or info.st_size <= 0 or info.st_size > 4096):
        raise SupervisionError("probe-failure", "CUDA pool proof is linked, empty, or oversized")
    return info.st_dev, info.st_ino, info.st_size


def _read_pool_proof(path: Path):
    try:
        before = path.lstat()
    except FileNotFoundError:
        return _MISSING
    except OSError as error:
        raise SupervisionError("probe-failure", "CUDA pool proof metadata is unavailable") from error
    identity = _proof_file_identity(before)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if _proof_file_identity(opened) != identity or _proof_file_identity(path.lstat()) != identity:
                raise SupervisionError("probe-failure", "CUDA pool proof changed during open")
            raw = os.read(fd, 4097)
            if len(raw) != identity[2] or _proof_file_identity(os.fstat(fd)) != identity:
                raise SupervisionError("probe-failure", "CUDA pool proof changed during read")
        finally:
            os.close(fd)
    except OSError as error:
        raise SupervisionError("probe-failure", "CUDA pool proof changed or is unreadable") from error
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SupervisionError("probe-failure", "CUDA pool proof is unreadable") from error


class RejectedPoolSample(SupervisionError):
    """A real SystemProbe reading whose candidate pool receipt failed validation."""

    INVALID_ALLOCATION = "CUDA pool proof has no valid owned PID/UUID allocation"
    EXCEEDS_CURRENT_USAGE = "CUDA pool proof exceeds concurrent global use"

    def __init__(self, detail, row):
        super().__init__("probe-failure", detail)
        self.pool = row


class PairCudaProbe:
    def __init__(self, policy, proof_path: Path, *, dedicated_adapter=None, retain_windows_footprint=False):
        self.policy = policy
        self.proof_path = proof_path
        self.system = SystemProbe(policy)
        self.dedicated_adapter = dedicated_adapter
        self.dedicated = (DedicatedSamples(dedicated_adapter, policy.cuda_device_uuid)
                          if dedicated_adapter is not None else None)
        self.pid_peak_bytes = 0
        self.unknown_pid_polls = 0
        self.global_upper_bytes = 0
        self.pool_fallback_peak_bytes = 0
        self.gpu_usage_basis = "owned-process-attributed"
        self.windows_footprint_peaks = {} if retain_windows_footprint else None

    def __getattr__(self, name):
        return getattr(self.system, name)

    def tree_footprint(self, owner):
        footprint = self.system.tree_footprint(owner)
        peaks = self.windows_footprint_peaks
        if peaks is None or footprint is None:
            return footprint
        details = self.system.last_windows_footprint_details
        if details is None:
            return footprint
        rows = details["processes"]
        if len(rows) not in (0, 1) or details["sampledProcessCount"] != len(rows):
            return footprint
        # Windows can still read the owned Job's peak if the child exits while opening its PID.
        counters = [("job_peak", details["jobPeakMemoryUsedBytes"])]
        if rows:
            row, = rows
            if peaks.get("pid", row["pid"]) != row["pid"]:
                return footprint
            counters.extend((("working_set", row["workingSetBytes"]), ("private_usage", row["privateUsageBytes"])))
        if footprint <= 0 or max(value for _, value in counters) != footprint:
            return footprint
        if rows:
            peaks["pid"] = row["pid"]
            peaks["pid_sample_count"] = peaks.get("pid_sample_count", 0) + 1
        peaks["sample_count"] = peaks.get("sample_count", 0) + 1
        at = time.time_ns()
        for field, value in counters:
            if field not in peaks or value > peaks[field]["bytes"]:
                peaks[field] = {"bytes": value, "at_unix_ns": at}
        return footprint

    def gpu_free_and_tree_bytes(self, owner):
        try:
            free, used = self.system.gpu_free_and_tree_bytes(owner)
        except SupervisionError:
            # A failed NVIDIA read can be retried or tolerated by the watchdog.
            # Collect PDH on this attempt too, treating an absent live gauge as a gap.
            self._poll_dedicated(owner, cuda_active=True)
            raise
        self.gpu_usage_basis = self.system.gpu_usage_basis
        self._poll_dedicated(owner, cuda_active=self.gpu_usage_basis == "global-used-upper-bound" or used > 0)
        if self.gpu_usage_basis != "global-used-upper-bound":
            self.pid_peak_bytes = max(self.pid_peak_bytes, used)
            return free, used
        self.unknown_pid_polls += 1
        self.global_upper_bytes = max(self.global_upper_bytes, self.system.last_global_used_upper_bound_bytes)
        # The upper bound remains the cap input even when the child proof is
        # available. An unrelated process can only make that bound stricter.
        row = _read_pool_proof(self.proof_path)
        if row is _MISSING:
            return free, used
        if type(row) is not dict:
            raise SupervisionError("probe-failure", "CUDA pool proof is malformed")
        pid, uuid = row.get("pid"), row.get("cudaDeviceUuid")
        values = [row.get(key) for key in ("poolUsedBytes", "poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")]
        if (type(pid) is not int or pid not in owner.members() or uuid != self.policy.cuda_device_uuid
                or any(type(value) is not int or value < 0 for value in values)
                or any(value <= 0 for value in values)
                or values[0] > values[1] or values[0] > values[2]
                or values[1] > values[3] or values[2] > values[3]
                or values[3] > self.policy.child_gpu_cap_bytes):
            raise RejectedPoolSample(RejectedPoolSample.INVALID_ALLOCATION, row)
        # Global use includes the owned child and any other process. Never use
        # its positive value as evidence of this child's allocation.
        if used < values[2]:
            raise RejectedPoolSample(RejectedPoolSample.EXCEEDS_CURRENT_USAGE, row)
        if used > self.policy.child_gpu_cap_bytes:
            raise SupervisionError("device-memory", "WDDM global use exceeded child cap")
        self.gpu_usage_basis = "owned-process-attributed"
        self.pool_fallback_peak_bytes = max(self.pool_fallback_peak_bytes, values[3])
        return free, values[3]

    def _poll_dedicated(self, owner, *, cuda_active):
        if self.dedicated is not None:
            self.dedicated.poll(owner, cuda_active=cuda_active)
            samples = self.dedicated.record["samples"]
            if samples and samples[-1][1] > self.policy.child_gpu_cap_bytes:
                raise SupervisionError("device-memory", "WDDM dedicated process usage exceeded child cap")

    def completed_process_gpu_sample(self, pid):
        if self.dedicated is None:
            return False
        # Close also records a PDH error, which must invalidate the completion witness.
        self.dedicated.close()
        try:
            return sampled_peak(self.dedicated.record, pid=pid, adapter=self.dedicated_adapter) is not None
        except ValueError as error:
            raise SupervisionError("probe-failure", str(error)) from error

    def close(self):
        if self.dedicated is not None:
            self.dedicated.close()

    def measurement(self):
        extra = {"wddm_dedicated": self.dedicated.record} if self.dedicated is not None else {}
        return {**extra, "pid_peak_bytes": self.pid_peak_bytes, "unknown_pid_polls": self.unknown_pid_polls,
                "global_upper_bytes": self.global_upper_bytes, "pool_fallback_peak_bytes": self.pool_fallback_peak_bytes}

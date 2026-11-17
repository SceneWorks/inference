"""Proof-only WDDM attribution from the owned child's live CUDA pool.

The ordinary supervisor still measures exact-card free/global use and enforces
its caps. This adapter supplies attribution only when WDDM reports N/A for an
exact PID/UUID row belonging to the supervised Windows Job.
"""
import json
import os
from pathlib import Path
import stat

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


class PairCudaProbe:
    def __init__(self, policy, proof_path: Path):
        self.policy = policy
        self.proof_path = proof_path
        self.system = SystemProbe(policy)
        self.gpu_usage_basis = "owned-process-attributed"

    def __getattr__(self, name):
        return getattr(self.system, name)

    def gpu_free_and_tree_bytes(self, owner):
        free, used = self.system.gpu_free_and_tree_bytes(owner)
        self.gpu_usage_basis = self.system.gpu_usage_basis
        if self.gpu_usage_basis != "global-used-upper-bound":
            return free, used
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
            raise SupervisionError("probe-failure", "CUDA pool proof has no valid owned PID/UUID allocation")
        # Global use includes the owned child and any other process. Never use
        # its positive value as evidence of this child's allocation.
        if used < values[2]:
            raise SupervisionError("probe-failure", "CUDA pool proof exceeds concurrent global use")
        if used > self.policy.child_gpu_cap_bytes:
            raise SupervisionError("device-memory", "WDDM global use exceeded child cap")
        self.gpu_usage_basis = "owned-process-attributed"
        return free, values[3]

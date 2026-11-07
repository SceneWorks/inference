"""Proof-only WDDM attribution from the owned child's live CUDA pool.

The ordinary supervisor still measures exact-card free/global use and enforces
its caps. This adapter supplies attribution only when WDDM reports N/A for an
exact PID/UUID row belonging to the supervised Windows Job.
"""
import json
from pathlib import Path
import stat

from scripts.media_campaign_supervisor import SupervisionError, SystemProbe


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
        try:
            info = self.proof_path.lstat()
        except FileNotFoundError:
            return free, used
        except OSError as error:
            raise SupervisionError("probe-failure", "CUDA pool proof metadata is unavailable") from error
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4096:
            raise SupervisionError("probe-failure", "CUDA pool proof is linked or oversized")
        try:
            row = json.loads(self.proof_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SupervisionError("probe-failure", "CUDA pool proof is unreadable") from error
        if type(row) is not dict:
            raise SupervisionError("probe-failure", "CUDA pool proof is malformed")
        pid, uuid = row.get("pid"), row.get("cudaDeviceUuid")
        values = [row.get(key) for key in ("poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")]
        if (type(pid) is not int or pid not in owner.members() or uuid != self.policy.cuda_device_uuid
                or any(type(value) is not int or value < 0 for value in values)
                or values[0] <= 0 or values[2] <= 0 or values[0] > values[2] or values[1] > values[2]
                or values[2] > self.policy.child_gpu_cap_bytes):
            raise SupervisionError("probe-failure", "CUDA pool proof has no valid owned PID/UUID allocation")
        # Global use includes the owned child and any other process. Never use
        # its positive value as evidence of this child's allocation.
        if used < values[1]:
            raise SupervisionError("probe-failure", "CUDA pool proof exceeds concurrent global use")
        if used > self.policy.child_gpu_cap_bytes:
            raise SupervisionError("device-memory", "WDDM global use exceeded child cap")
        self.gpu_usage_basis = "owned-process-attributed"
        return free, values[2]

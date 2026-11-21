"""Observe Mochi's synchronized completion on the same live Windows Job and PDH adapter."""
import ctypes
import json
import time

from scripts.ci.flux1_pair.cuda_probe import _MISSING, _read_pool_proof
from scripts.ci.model_proof_resources import shared_owned_completion
from scripts.ci.scail_plural.wddm_retry import MAX_RETRY_EVENTS, SameTickSamples, local_filetime
from scripts.ci.wddm_process_memory import sampled_peak

CASE = "cuda-mochi-masked-euler"


def utc_ns():
    # On CPython 3.12 Windows time.time_ns can lag Rust SystemTime by one scheduler tick.
    # Read the precise UTC FILETIME directly, in the same Unix nanosecond units as the child.
    from ctypes.wintypes import FILETIME
    read = ctypes.WinDLL("kernel32.dll", winmode=0x800).GetSystemTimePreciseAsFileTime
    read.argtypes, read.restype = [ctypes.POINTER(FILETIME)], None
    stamp = FILETIME()
    read(ctypes.byref(stamp))
    return ((stamp.dwHighDateTime << 32 | stamp.dwLowDateTime) - 116444736000000000) * 100


class CompletionSamples(SameTickSamples):
    def __init__(self, gauge, policy, out):
        super().__init__(gauge, policy.poll_millis, immediate=True)
        self.out = out
        self.ready = None
        self.ack = None
        self.release = None

    def check_release(self):
        if self.ack is None or self.release is not None:
            return
        packet = _read_pool_proof(self.out / "mochi-release.json")
        if packet is not _MISSING:
            if packet["ack"] != self.ack:
                raise ValueError("native Mochi release differs from this owned acknowledgement")
            self.release = {"seen_ns": time.monotonic_ns(), "seen_utc_ns": utc_ns(),
                            "local_filetime": local_filetime(), "retry_count": self.diagnostics["attempts"],
                            "sample_count": len(self.record["samples"])}

    def poll(self, owner, *, cuda_active):
        # Only classify a poll as teardown when the native release was already visible
        # before starting that poll. Keep the original lifetime record for independent replay.
        self.check_release()
        if self.ready is None:
            packet = _read_pool_proof(self.out / "mochi-ready.json")
            if packet is not _MISSING:
                self.ready = {"packet": packet, "seen_ns": time.monotonic_ns(),
                              "seen_utc_ns": utc_ns(), "local_filetime": local_filetime()}
        count = len(self.record["samples"])
        super().poll(owner, cuda_active=cuda_active)
        if self.ready is None or self.ack is not None or self.record["error"] or len(self.record["samples"]) == count:
            return
        sample = self.record["samples"][-1]
        end_local = local_filetime()
        if not self.ready["local_filetime"] < sample[0] <= end_local or sample[1] <= 0:
            return
        row, packet = self.record, self.ready["packet"]
        if packet.get("case") != CASE or packet.get("pid") != row["pid"] or packet.get("uuid") != row["adapter"]["cuda_uuid"]:
            raise ValueError("Mochi completion marker differs from its owned PID/UUID")
        if owner.members() != {row["pid"]} or owner.root_identity(row["pid"]) != (row["created_filetime"], False):
            raise ValueError("Mochi root changed before completion acknowledgement")
        self.ack = {"ready": packet, "created_filetime": row["created_filetime"], "adapter": row["adapter"],
                    "sample": sample, "sample_count": len(row["samples"]), "end_local_filetime": end_local,
                    "ack_ns": time.monotonic_ns(), "ack_utc_ns": utc_ns()}
        tmp = self.out / "mochi-ack.pending"
        with tmp.open("x", encoding="utf-8") as file:
            json.dump(self.ack, file)
        tmp.rename(self.out / "mochi-ack.json")


def live_record(guard, native, adapter, role):
    """Return the live prefix without altering the full, sealed lifetime record."""
    from scripts.ci import video_tier1_cuda_diagnostic as diagnostic
    from scripts.ci.wddm_process_memory import validate_adapter
    row = guard["cuda_attribution"]["wddm_dedicated"]
    witness = guard["fit_observation"]
    done = guard["owned_completion"]
    if (not shared_owned_completion({"owned_execution_completion": done})
            or done.get("root_pid") != guard["pid"] or done.get("waited_returncode") != 0):
        raise ValueError("Mochi lacks its clean, empty owned Windows Job")
    ready, release = witness["ready"], witness["release"]
    ack, packet = native["ack"], ready["packet"]
    ns = (packet["ready_utc_ns"], ready["seen_utc_ns"], ack["ack_utc_ns"], native["release_utc_ns"], release["seen_utc_ns"])
    mono = (ready["seen_ns"], ack["ack_ns"], release["seen_ns"])
    if (not all(type(v) is int and v > 0 for v in (*ns, *mono)) or list(ns) != sorted(ns)
            or list(mono) != sorted(mono) or not role["start_ns"] <= mono[0] <= mono[-1] <= role["end_ns"]
            or mono[1] - mono[0] > 5_000_000_000
            or native != {"ack": witness["ack"], "release_utc_ns": native["release_utc_ns"]}
            or ack["ready"] != packet or packet["case"] != CASE or packet["uuid"] != diagnostic.GPU0_HARDWARE["uuid"]
            or packet["pid"] != row["pid"] or row["pid"] != guard["pid"] or ack["created_filetime"] != row["created_filetime"]
            or validate_adapter(ack["adapter"], packet["uuid"]) != adapter or row["adapter"] != adapter):
        raise ValueError("Mochi native completion, Job or adapter identity differs")
    samples = row["samples"]
    k, n = ack["sample_count"], release["sample_count"]
    if (type(k) is not int or type(n) is not int or not 1 <= k <= n <= len(samples) or ack["sample"] != samples[k - 1]
            or not ready["local_filetime"] < ack["sample"][0] <= ack["end_local_filetime"] or ack["sample"][1] <= 0):
        raise ValueError("Mochi acknowledgement lacks its fresh owned positive PDH sample")
    # The raw lifetime gauge (including error and gaps) is always validated and retained.
    sampled_peak(row, pid=guard["pid"], adapter=adapter)
    diag = witness["diagnostics"]
    if diag["attempts"] > MAX_RETRY_EVENTS:
        raise ValueError("Mochi completion cannot classify unrecorded retries")
    left, clock = release["retry_count"], release["local_filetime"]
    if (type(left) is not int or not 0 <= left <= len(diag["events"])
            or not ack["end_local_filetime"] <= clock or samples[n - 1][0] > clock
            or any(s[0] < clock for s in samples[n:])
            or any(e["end_ns"] > release["seen_ns"] or e["local_filetime_after"] > clock for e in diag["events"][:left])
            or any(e["start_ns"] < release["seen_ns"] or e["local_filetime_before"] < clock for e in diag["events"][left:])):
        raise ValueError("Mochi release straddled an owned gauge poll")
    missing = [e for e in diag["events"][left:] if not e["recovered"]]
    if missing and any(s[0] > missing[0]["local_filetime_after"] and s[1] > 0 for s in samples[n:]):
        raise ValueError("Mochi dedicated usage reappeared after a missing teardown sample")
    gaps = row["missing_active_polls"] - len(missing)
    if gaps < 0 or any(s[1] > prev[1] for prev, s in zip(samples[k - 1:], samples[k:])):
        raise ValueError("Mochi showed contradictory gap accounting or GPU allocation growth after completion")
    return dict(row, samples=samples[:n], missing_active_polls=gaps)

"""Sample one direct Windows Job child's dedicated bytes on its selected CUDA adapter."""
import ctypes as C
import os
import re
import uuid
from contextlib import contextmanager


COUNTER = r"\GPU Process Memory(*)\Dedicated Usage"
NO_INSTANCE = 0x800007D1
NO_DATA = 0x800007D5
LARGE_RAWCOUNT = 0x10100
UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
LUID = re.compile(r"luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}")
DWORD = C.c_uint32
HANDLE = C.c_void_p


def _function(dll, name, args):
    fn = getattr(dll, name)
    fn.argtypes = args
    fn.restype = DWORD
    return fn


def _check(status, operation):
    if status:
        raise OSError(f"{operation} returned 0x{status:08x}")


def _adapter(cuda_uuid):
    """CUDA driver metadata only. No device context, uploads, kernels or device-mode writes."""
    if not isinstance(cuda_uuid, str) or not UUID.fullmatch(cuda_uuid):
        raise ValueError("expected a complete CUDA UUID")
    cuda = C.WinDLL("nvcuda.dll", winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
    _check(_function(cuda, "cuInit", [DWORD])(0), "cuInit")
    count = C.c_int()
    _check(_function(cuda, "cuDeviceGetCount", [C.POINTER(C.c_int)])(C.byref(count)), "cuDeviceGetCount")
    device_get = _function(cuda, "cuDeviceGet", [C.POINTER(C.c_int), C.c_int])
    uuid_get = _function(cuda, "cuDeviceGetUuid_v2", [HANDLE, C.c_int])
    matches = []
    for index in range(count.value):
        device, raw = C.c_int(), (C.c_ubyte * 16)()
        _check(device_get(C.byref(device), index), "cuDeviceGet")
        _check(uuid_get(raw, device), "cuDeviceGetUuid_v2")
        if uuid.UUID(bytes=bytes(raw)) == uuid.UUID(cuda_uuid[4:]):
            matches.append(device)
    if len(matches) != 1:
        raise ValueError("selected CUDA UUID has no unique visible Windows adapter")
    luid, mask = (C.c_ubyte * 8)(), DWORD()
    fn = _function(cuda, "cuDeviceGetLuid", [HANDLE, C.POINTER(DWORD), C.c_int])
    _check(fn(luid, C.byref(mask), matches[0]), "cuDeviceGetLuid")
    if not any(luid) or mask.value == 0 or mask.value & (mask.value - 1):
        raise ValueError("CUDA device does not identify exactly one adapter node")
    high, low = int.from_bytes(luid[4:], "little"), int.from_bytes(luid[:4], "little")
    return {"cuda_uuid": cuda_uuid, "luid": f"luid_0x{high:08x}_0x{low:08x}",
            "node_mask": mask.value, "physical_node": mask.value.bit_length() - 1, "counter": COUNTER}


def validate_adapter(adapter, cuda_uuid):
    if (not isinstance(adapter, dict) or set(adapter) != {"cuda_uuid", "luid", "node_mask", "physical_node", "counter"}
            or not isinstance(cuda_uuid, str) or not UUID.fullmatch(cuda_uuid) or adapter["cuda_uuid"] != cuda_uuid
            or not isinstance(adapter["luid"], str) or not LUID.fullmatch(adapter["luid"])
            or adapter["luid"] == "luid_0x00000000_0x00000000" or adapter["counter"] != COUNTER
            or type(adapter["physical_node"]) is not int or not 0 <= adapter["physical_node"] < 32
            or type(adapter["node_mask"]) is not int or adapter["node_mask"] != 1 << adapter["physical_node"]):
        raise ValueError("invalid selected CUDA UUID/adapter/node/counter metadata")
    return adapter


class _Raw(C.Structure):
    _fields_ = [("status", DWORD), ("stamp_low", DWORD), ("stamp_high", DWORD),
                ("first", C.c_int64), ("second", C.c_int64), ("multi", DWORD)]


class _Pdh:
    def __init__(self):
        dll = C.WinDLL("pdh.dll", winmode=0x800)
        self.open = _function(dll, "PdhOpenQueryW", [C.c_wchar_p, C.c_size_t, C.POINTER(HANDLE)])
        self.add = _function(dll, "PdhAddEnglishCounterW", [HANDLE, C.c_wchar_p, C.c_size_t, C.POINTER(HANDLE)])
        self.collect = _function(dll, "PdhCollectQueryData", [HANDLE])
        self.raw = _function(dll, "PdhGetRawCounterValue", [HANDLE, C.POINTER(DWORD), C.POINTER(_Raw)])
        self.close = _function(dll, "PdhCloseQuery", [HANDLE])

    def read(self, handle):
        kind, raw = DWORD(), _Raw()
        _check(self.raw(handle, C.byref(kind), C.byref(raw)), "PdhGetRawCounterValue")
        if raw.status == NO_INSTANCE:
            return None
        if raw.status not in (0, 1) or kind.value != LARGE_RAWCOUNT or raw.first < 0:
            raise OSError(f"invalid PDH byte gauge status/type 0x{raw.status:x}/0x{kind.value:x}")
        stamp = raw.stamp_high << 32 | raw.stamp_low
        if not stamp:
            raise OSError("PDH byte gauge has no provider timestamp")
        return [stamp, raw.first]


@contextmanager
def _query(adapter, pid=None):
    # Check both the exact occurrence and its first duplicate; selecting #0 alone
    # would silently pick a value if Windows exposed an ambiguous instance name.
    instance = f"{adapter['luid']}_phys_{adapter['physical_node']}"
    category = "GPU Adapter Memory"
    if pid is not None:
        instance = f"pid_{pid}_{instance}"
        category = "GPU Process Memory"
    api, query = _Pdh(), HANDLE()
    _check(api.open(None, 0, C.byref(query)), "PdhOpenQueryW")
    try:
        counters = []
        for suffix in ("", "#1"):
            path = rf"\{category}({instance}{suffix})\Dedicated Usage"
            counter = HANDLE()
            _check(api.add(query, path, 0, C.byref(counter)), "PdhAddEnglishCounterW")
            counters.append(counter)

        def read():
            status = api.collect(query)
            if status != NO_DATA:
                _check(status, "PdhCollectQueryData")
            primary, duplicate = [api.read(counter) for counter in counters]
            if duplicate is not None:
                raise OSError("ambiguous PDH dedicated-memory instance")
            if status == NO_DATA and primary is not None:
                raise OSError("inconsistent empty PDH collection")
            return primary

        yield read
    finally:
        _check(api.close(query), "PdhCloseQuery")


def preflight(cuda_uuid):
    if os.name != "nt":
        raise ValueError("Windows WDDM preflight requires the Windows runner")
    adapter = validate_adapter(_adapter(cuda_uuid), cuda_uuid)
    with _query(adapter) as read:
        if read() is None:
            raise ValueError("PDH cannot read the selected adapter's dedicated-memory gauge")
    # Add the exact English process counter before scheduling costly builds. A
    # host process without a GPU instance may legitimately return NO_INSTANCE.
    with _query(adapter, os.getpid()) as read:
        read()
    return adapter


def _creation_time(owner, pid):
    rows = owner.member_identities([pid])["processes"]
    if len(rows) == 1 and rows[0]["pid"] == pid and rows[0]["status"] == "owned-identity-resolved":
        return rows[0]["createdFileTime"]
    return None


class DedicatedSamples:
    def __init__(self, adapter, cuda_uuid):
        self.record = {"adapter": validate_adapter(adapter, cuda_uuid), "pid": None, "created_filetime": None,
                       "samples": [], "missing_active_polls": 0, "error": None}
        self.query = None
        self.read = None

    def close(self):
        if self.query is not None:
            try:
                self.query.__exit__(None, None, None)
            except OSError as exc:
                self.record["error"] = str(exc)
            finally:
                self.query = None

    def poll(self, owner, *, cuda_active):
        row = self.record
        if row["error"]:
            return
        try:
            members = owner.members()
            if not members:  # the process exited before this observation
                return
            if len(members) != 1:
                raise ValueError("direct render does not have exactly one owned process")
            pid = next(iter(members))
            created = _creation_time(owner, pid)
            if not created or (row["pid"] is not None and (pid, created) != (row["pid"], row["created_filetime"])):
                raise ValueError("owned process identity changed or is unavailable")
            if self.query is None:
                context = _query(row["adapter"], pid)
                self.read = context.__enter__()
                self.query = context
                row.update(pid=pid, created_filetime=created)
            sample = self.read()
            if owner.members() != {pid} or _creation_time(owner, pid) != created:
                raise ValueError("owned process identity changed during PDH read")
            if sample is None:
                if cuda_active or row["samples"]:
                    row["missing_active_polls"] += 1
                return
            if row["samples"] and sample[0] == row["samples"][-1][0] and sample == row["samples"][-1]:
                return  # one cached provider timestamp is one observation
            if row["samples"] and sample[0] <= row["samples"][-1][0]:
                raise ValueError("PDH provider time moved backwards or reused a timestamp")
            row["samples"].append(sample)
        except (OSError, RuntimeError, ValueError) as exc:
            row["error"] = str(exc)


def sampled_peak(record, *, pid, adapter):
    """Recompute a sampled high-water mark; zero-only or incomplete measurement is inconclusive."""
    if (not isinstance(record, dict) or set(record) != {"adapter", "pid", "created_filetime", "samples",
                                                        "missing_active_polls", "error"}
            or record["adapter"] != adapter or type(pid) is not int or pid <= 0
            or type(record["missing_active_polls"]) is not int or record["missing_active_polls"] < 0
            or not isinstance(record["samples"], list)
            or record["error"] is not None and not isinstance(record["error"], str)):
        raise ValueError("malformed Windows dedicated-memory record")
    validate_adapter(record["adapter"], adapter["cuda_uuid"])
    if record["pid"] is None and record["created_filetime"] is None and not record["samples"]:
        return None
    if (type(record["pid"]) is not int or record["pid"] != pid
            or type(record["created_filetime"]) is not int or record["created_filetime"] <= 0):
        raise ValueError("Windows dedicated-memory sample belongs to a different process")
    last, positive, peak = 0, 0, 0
    for sample in record["samples"]:
        if (not isinstance(sample, list) or len(sample) != 2 or type(sample[0]) is not int
                or sample[0] <= last or type(sample[1]) is not int or sample[1] < 0):
            raise ValueError("invalid Windows dedicated-memory sample")
        last = sample[0]
        positive += int(sample[1] > 0)
        peak = max(peak, sample[1])
    if record["error"] is not None or record["missing_active_polls"] or positive < 2:
        return None
    return peak


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Read selected CUDA adapter and PDH metadata without model execution.")
    parser.add_argument("--cuda-uuid", required=True)
    print(json.dumps(preflight(parser.parse_args().cuda_uuid), sort_keys=True))

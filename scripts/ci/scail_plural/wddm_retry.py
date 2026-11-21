"""One fresh, owned PDH re-read within a SCAIL qualification poll."""

import ctypes as C
import time


def local_filetime():
    # PDH_RAW_COUNTER.TimeStamp is local FILETIME, unlike a Job's UTC creation time.
    # Read the same Windows clock without interpreting process or foreign GPU usage.
    from ctypes.wintypes import FILETIME

    kernel = C.WinDLL("kernel32.dll", winmode=0x800, use_last_error=True)
    precise = kernel.GetSystemTimePreciseAsFileTime
    precise.argtypes, precise.restype = [C.POINTER(FILETIME)], None
    local = kernel.FileTimeToLocalFileTime
    local.argtypes, local.restype = [C.POINTER(FILETIME), C.POINTER(FILETIME)], C.c_int
    utc, stamp = FILETIME(), FILETIME()
    precise(C.byref(utc))
    if not local(C.byref(utc), C.byref(stamp)):
        raise C.WinError(C.get_last_error())
    return (stamp.dwHighDateTime << 32) | stamp.dwLowDateTime


MAX_RETRY_EVENTS = 64


def same_tick_fresh(event, tick_ns):
    sample = event["retry_sample"]
    lower, upper = event["local_filetime_before"], event["local_filetime_after"]
    return (
        event["retried"] is True
        and sample is not None
        and lower < sample[0] <= upper
        and (upper - lower) * 100 <= tick_ns
        and 0 <= event["end_ns"] - event["start_ns"] <= tick_ns
    )


def validate_retry(diagnostics, record, poll_millis):
    attempts, events = diagnostics["attempts"], diagnostics["events"]
    if (
        type(attempts) is not int
        or attempts < 0
        or len(events) != min(attempts, MAX_RETRY_EVENTS)
    ):
        raise ValueError("incomplete WDDM retry diagnostics")
    positions = {t: i for i, (t, _) in enumerate(record["samples"])}
    recovered = 0
    last = 0
    last_sample_index = -1
    for event in events:
        if (
            not last < event["start_ns"] <= event["end_ns"]
            or type(event["retried"]) is not bool
            or type(event["recovered"]) is not bool
        ):
            raise ValueError("invalid WDDM retry interval")
        last = event["end_ns"]
        sample = event["retry_sample"]
        if sample is not None:
            index = positions.get(sample[0])
            if (
                not event["retried"]
                or index is None
                or index <= last_sample_index
                or record["samples"][index] != sample
                or event["previous"]
                != (record["samples"][index - 1] if index else None)
            ):
                raise ValueError("WDDM retry differs from owned observations")
            last_sample_index = index
        if event["recovered"] != same_tick_fresh(event, poll_millis * 1_000_000):
            raise ValueError("WDDM retry lacks a fresh same-tick observation")
        recovered += event["recovered"]
    if attempts - recovered != record["missing_active_polls"]:
        raise ValueError("WDDM retry accounting differs from owned observations")


class SameTickSamples:
    def __init__(self, gauge, poll_millis, *, immediate=False):
        self.immediate = immediate
        self.gauge = gauge
        self.record = gauge.record
        self.tick_ns = poll_millis * 1_000_000
        self.diagnostics = {"attempts": 0, "events": []}

    def close(self):
        self.gauge.close()

    def poll(self, owner, *, cuda_active):
        row = self.record
        samples = row["samples"]
        previous = samples[-1] if samples else None
        gaps = row["missing_active_polls"]
        started = time.monotonic_ns()
        self.gauge.poll(owner, cuda_active=cuda_active)
        if row["missing_active_polls"] == gaps:
            return

        # The first read was absent with a verified live Job root. The same gauge
        # brackets its second collection with the same handle, PID and creation time.
        self.diagnostics["attempts"] += 1
        if self.diagnostics["attempts"] > MAX_RETRY_EVENTS:
            return  # keep the genuine missing tick when its retry cannot be audited
        lower = local_filetime()
        remaining = self.tick_ns - (time.monotonic_ns() - started)
        retried = False
        if remaining > 0:
            if not self.immediate:
                time.sleep(min(0.015, remaining / 2_000_000_000))
            if time.monotonic_ns() - started < self.tick_ns:
                self.gauge.poll(owner, cuda_active=cuda_active)
                retried = True
        upper, finished = local_filetime(), time.monotonic_ns()
        fresh = samples[-1] if samples and samples[-1] != previous else None
        event = {
            "start_ns": started,
            "end_ns": finished,
            "retried": retried,
            "local_filetime_before": lower,
            "local_filetime_after": upper,
            "previous": previous,
            "retry_sample": fresh,
        }
        event["recovered"] = same_tick_fresh(event, self.tick_ns)
        # Count missing scheduled polls, not the two reads attempted in one poll.
        # A cached value, late read, error or exit preserves its original absence.
        row["missing_active_polls"] = gaps + int(not event["recovered"])
        self.diagnostics["events"].append(event)

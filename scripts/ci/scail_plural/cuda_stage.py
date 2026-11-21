"""Evidence for the end of a completed SCAIL CUDA render on its original Job."""


def nonnegative(value):
    return type(value) is int and value >= 0


def tenant_growth(guard, cuda_uuid):
    """The supervisor already adjusts the deductible baseline as each tenant exits."""
    peak = 0
    seen = 0
    last = 0
    for row in guard["samples"]:
        if row["phase"] != "cuda-global-used-upper-bound-sample":
            continue
        growth, absolute = (
            row["childGlobalUsedGrowthBytes"],
            row["globalUsedUpperBoundBytes"],
        )
        if not (
            row["cudaDeviceUuid"] == cuda_uuid
            and nonnegative(growth)
            and nonnegative(absolute)
            and growth <= absolute <= guard["cuda_attribution"]["global_upper_bytes"]
            and nonnegative(row["gpuFreeBytes"])
            and type(row["at_ns"]) is int
            and row["at_ns"] > last
        ):
            raise ValueError("invalid selected-card tenant-aware growth observation")
        last = row["at_ns"]
        seen += 1
        peak = max(peak, growth)
    return peak, seen


def original_root(owner, pid, created):
    actual, exited = owner.root_identity(pid)
    members = owner.members()
    if (
        type(created) is int
        and created > 0
        and type(actual) is int
        and actual == created
    ):
        if not exited and members == {pid}:
            return "live"
        if members <= {pid} and owner.exited_root_and_empty(pid) is True:
            return "exited_empty"
    raise ValueError("CUDA observation changed the original Windows Job identity")


class RenderEnd:
    def __init__(self):
        self.record = {"pool": None, "zero": None, "pool_ticks": 0}

    def before(self, owner):
        mark = self.record["pool"]
        if mark is not None:
            return original_root(owner, mark["pid"], mark["created_filetime"])
        return None

    def observe(self, owner, probe, before, card_index, at_ns, used):
        raw = self.record
        mark = raw["pool"]
        native_pool = (
            probe.system.gpu_usage_basis == "global-used-upper-bound"
            and probe.gpu_usage_basis == "owned-process-attributed"
        )
        if native_pool:
            # PairCudaProbe has just read and validated the atomic, post-output
            # cuda-pool.json against this Job PID, CUDA UUID, cap and raw pool.
            raw["pool_ticks"] += 1
            if mark is None:
                series = probe.dedicated.record
                if used <= 0 or probe.pool_fallback_peak_bytes != used:
                    raise ValueError("inconsistent CUDA pool at render completion")
                if (
                    original_root(owner, series["pid"], series["created_filetime"])
                    != "live"
                ):
                    return
                raw["pool"] = {
                    "card_index": card_index,
                    "at_ns": at_ns,
                    "pid": series["pid"],
                    "created_filetime": series["created_filetime"],
                    "cuda_uuid": probe.policy.cuda_device_uuid,
                    "basis": "verified-output-pool",
                    "pool_reserved_high_bytes": used,
                }
        if mark is not None:
            raw["zero"] = None
            if (
                used == 0
                and probe.gpu_usage_basis == "owned-process-attributed"
                and probe.system.gpu_usage_basis == "owned-process-attributed"
            ):
                after = original_root(owner, mark["pid"], mark["created_filetime"])
                if before == "exited_empty" and after != before:
                    raise ValueError("exited CUDA Job became occupied")
                raw["zero"] = {
                    "card_index": card_index,
                    "at_ns": at_ns,
                    "pid": mark["pid"],
                    "created_filetime": mark["created_filetime"],
                    "cuda_uuid": mark["cuda_uuid"],
                    "basis": "owned-process-attributed",
                    "before": before,
                    "after": after,
                }


def render_scope(guard, native_pool, cuda_uuid, growth_count):
    raw = guard["fit_observation"]
    end = raw.get("cuda_render_end")
    if end is None:
        return None  # An older capture has no independent end-of-render witness.
    unknown = guard["cuda_attribution"]["unknown_pid_polls"]
    if (
        not nonnegative(end["pool_ticks"])
        or growth_count + end["pool_ticks"] != unknown
    ):
        raise ValueError(
            "CUDA fallback ticks differ from the supervisor growth records"
        )
    mark, zero = end["pool"], end["zero"]
    ded = guard["cuda_attribution"]["wddm_dedicated"]
    shared = raw["wddm_shared"]
    if mark is None:
        if zero is not None:
            raise ValueError("CUDA teardown has no recorded render completion")
        return None
    card = raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
    if (
        not card
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or not all(nonnegative(v) for v in row)
            for row in card
        )
        or any(a[0] >= b[0] for a, b in zip(card, card[1:]))
    ):
        raise ValueError("invalid CUDA render-end card timeline")

    if (
        native_pool["cudaDeviceUuid"] != cuda_uuid
        or ded["adapter"]["cuda_uuid"] != cuda_uuid
        or ded["adapter"] != shared["adapter"]
        or not 0
        < native_pool["poolUsedBytes"]
        <= native_pool["poolReservedBytes"]
        <= native_pool["poolReservedHighBytes"]
        or not native_pool["poolUsedBytes"]
        <= native_pool["poolUsedHighBytes"]
        <= native_pool["poolReservedHighBytes"]
    ):
        raise ValueError(
            "CUDA scope differs from the authenticated native pool or adapter"
        )

    def matches(m, value):
        i = m["card_index"]
        return (
            nonnegative(i)
            and i < len(card)
            and card[i][0] == m["at_ns"]
            and card[i][2] == value
            and m["cuda_uuid"] == cuda_uuid
            and m["pid"] == guard["pid"] == native_pool["pid"]
            and m["created_filetime"]
            == ded["created_filetime"]
            == shared["created_filetime"]
        )

    high = native_pool["poolReservedHighBytes"]
    if (
        not end["pool_ticks"]
        or mark["basis"] != "verified-output-pool"
        or mark["pool_reserved_high_bytes"] != high
        or not matches(mark, high)
        or end["pool_ticks"] > sum(row[2] == high for row in card[mark["card_index"] :])
    ):
        raise ValueError("CUDA end-of-render marker differs from the owned pool")
    if zero is None:
        return None
    if (
        zero["basis"] != "owned-process-attributed"
        or not matches(zero, 0)
        or zero["card_index"] != len(card) - 1
        or mark["at_ns"] >= zero["at_ns"]
        or (zero["before"], zero["after"])
        not in {
            ("live", "live"),
            ("live", "exited_empty"),
            ("exited_empty", "exited_empty"),
        }
    ):
        raise ValueError("invalid final zero reading on the original CUDA Job")
    last_used = max(row[0] for row in card if row[2] > 0)
    for key, series in (("dedicated", ded), ("shared", shared)):
        misses = [e for e in raw["wddm_retry"][key]["events"] if not e["recovered"]]
        if len(misses) != series["missing_active_polls"]:
            return None  # bounded diagnostic cannot locate every genuine gap
        for e in misses:
            if not (
                max(mark["at_ns"], last_used)
                < e["start_ns"]
                <= e["end_ns"]
                <= zero["at_ns"]
                and e["retried"] is True
                and e["retry_sample"] is None
                and series["samples"]
                and e["previous"] == series["samples"][-1]
                and e["local_filetime_before"] >= series["samples"][-1][0]
            ):
                return None
    return {
        "pid": mark["pid"],
        "created_filetime": mark["created_filetime"],
        "cuda_uuid": cuda_uuid,
        "pool_card_index": mark["card_index"],
        "zero_card_index": zero["card_index"],
        "dedicated_missing_after_output": ded["missing_active_polls"],
        "shared_missing_after_output": shared["missing_active_polls"],
    }

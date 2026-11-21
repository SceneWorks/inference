"""Evidence for the end of a completed SCAIL CUDA render on its original Job."""

import time

from scripts.ci.flux1_pair.cuda_probe import RejectedPoolSample
from scripts.media_campaign_supervisor import SupervisionError


def nonnegative(value):
    return type(value) is int and value >= 0


class KeepSystemGrowth:
    """Observe the existing supervisor read before PairCudaProbe may return pool bytes."""

    def __init__(self, system):
        self.system = system
        self.current = None

    def __getattr__(self, name):
        return getattr(self.system, name)

    def gpu_free_and_tree_bytes(self, owner):
        self.current = None
        free, used = self.system.gpu_free_and_tree_bytes(owner)
        if self.system.gpu_usage_basis == "global-used-upper-bound":
            self.current = {
                "read_ns": time.monotonic_ns(),
                "cuda_uuid": self.system.policy.cuda_device_uuid,
                "free_bytes": free,
                "global_used_upper_bytes": self.system.last_global_used_upper_bound_bytes,
                "child_growth_bytes": used,
            }
        # Apply the existing watchdog predicates to the real reading before a
        # pool substitution (or a retryable pool error) can hide it.
        policy = self.system.policy
        if free < policy.gpu_free_reserve_bytes or used > policy.child_gpu_cap_bytes:
            error = SupervisionError(
                "device-memory", "owned CUDA use or free reserve exceeded cap"
            )
            error.watchdog_device_memory = {
                "cudaDeviceUuid": policy.cuda_device_uuid,
                "gpuUsageBasis": self.system.gpu_usage_basis,
                "gpuFreeBytes": free,
                "childGpuCapBytes": policy.child_gpu_cap_bytes,
                "gpuFreeReserveBytes": policy.gpu_free_reserve_bytes,
                "capExceeded": used > policy.child_gpu_cap_bytes,
                "reserveBreached": free < policy.gpu_free_reserve_bytes,
                "ownedGpuBytes": None if self.current else used,
                "childGlobalUsedGrowthBytes": used if self.current else None,
                "globalUsedUpperBoundBytes": self.current["global_used_upper_bytes"]
                if self.current
                else None,
            }
            raise error
        return free, used


def tenant_growth(guard, cuda_uuid):
    """The supervisor already adjusts the deductible baseline as each tenant exits."""

    def valid_charge(growth, absolute):
        baseline = guard["admission"].get("gpuGlobalUsedBaselineBytes")
        return (
            nonnegative(baseline)
            and nonnegative(growth)
            and nonnegative(absolute)
            and max(0, absolute - baseline)
            <= growth
            <= absolute
            <= guard["cuda_attribution"]["global_upper_bytes"]
        )

    peak = 0
    absolute_peak = 0
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
            and valid_charge(growth, absolute)
            and nonnegative(row["gpuFreeBytes"])
            and type(row["at_ns"]) is int
            and row["at_ns"] > last
        ):
            raise ValueError("invalid selected-card tenant-aware growth observation")
        last = row["at_ns"]
        seen += 1
        peak = max(peak, growth)
        absolute_peak = max(absolute_peak, absolute)
    raw = guard["fit_observation"]
    end = raw.get("cuda_render_end")
    if end is not None:
        card = raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
        entries = end["pool_growth"]
        if len(entries) != end["pool_ticks"]:
            raise ValueError("missing original growth for a substituted CUDA pool tick")
        last_index = -1
        for entry in entries:
            i = entry["card_index"]
            growth = entry["child_growth_bytes"]
            if not (
                nonnegative(i)
                and last_index < i < len(card)
                and entry["cuda_uuid"] == cuda_uuid
                and nonnegative(entry["read_ns"])
                and (card[i - 1][0] if i else 0)
                < entry["read_ns"]
                <= entry["at_ns"]
                == card[i][0]
                and nonnegative(entry["free_bytes"])
                and entry["free_bytes"] == card[i][1]
                and valid_charge(growth, entry["global_used_upper_bytes"])
            ):
                raise ValueError("invalid tenant growth at CUDA pool substitution")
            peak = max(peak, growth)
            last_index = i
            absolute_peak = max(absolute_peak, entry["global_used_upper_bytes"])
        # Rejected substitutions never become completed card samples. Keep the
        # original SystemProbe readings in the price and in the absolute ledger.
        for entry in end.get("rejected_growth", []):
            growth = entry["child_growth_bytes"]
            absolute = entry["global_used_upper_bytes"]
            if not (entry["cuda_uuid"] == cuda_uuid and valid_charge(growth, absolute)):
                raise ValueError(
                    "invalid selected-card growth on rejected CUDA pool attempt"
                )
            peak = max(peak, growth)
            absolute_peak = max(absolute_peak, absolute)
        if absolute_peak != guard["cuda_attribution"]["global_upper_bytes"]:
            raise ValueError(
                "CUDA growth evidence does not reproduce the card high-water mark"
            )
        if entries and (
            end["pool"] is None
            or entries[0]["card_index"] != end["pool"]["card_index"]
            or any(
                card[e["card_index"]][2] != end["pool"]["pool_reserved_high_bytes"]
                for e in entries
            )
        ):
            raise ValueError(
                "CUDA pool growth differs from the verified render-end marker"
            )
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
        self.record = {
            "pool": None,
            "zero": None,
            "pool_ticks": 0,
            "pool_growth": [],
            "rejected_growth": [],
        }

    def before(self, owner):
        mark = self.record["pool"]
        if mark is not None:
            return original_root(owner, mark["pid"], mark["created_filetime"])
        return None

    def reject(self, owner, probe, before, next_card_index, error):
        original = probe.system.current
        if original is None:
            raise ValueError(
                "missing source measurement on rejected CUDA pool attempt"
            ) from error
        mark = self.record["pool"]
        after = self.before(owner)
        self.record["rejected_growth"].append(
            dict(
                original,
                next_card_index=next_card_index,
                at_ns=time.monotonic_ns(),
                pid=mark["pid"] if mark else None,
                created_filetime=mark["created_filetime"] if mark else None,
                before=before,
                after=after,
                reason=error.reason,
                detail=error.detail,
                pool=getattr(error, "pool", None),
            )
        )

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
            original = probe.system.current
            if original is None:
                raise ValueError(
                    "missing original supervised growth before CUDA pool substitution"
                )
            raw["pool_ticks"] += 1
            raw["pool_growth"].append(
                dict(original, card_index=card_index, at_ns=at_ns)
            )
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


def validate_rejected_growth(guard, end, native_pool, card, *, completed_job=False):
    """Only the original Job retiring a verified post-output pool can be retried."""
    previous = 0
    exited_at = None
    mark, zero = end["pool"], end["zero"]
    for event in end.get("rejected_growth", []):
        i, read, at = event["next_card_index"], event["read_ns"], event["at_ns"]
        if not (
            mark
            and nonnegative(i)
            and mark["card_index"] < i
            and ((completed_job and i <= len(card)) or (zero and i <= zero["card_index"] < len(card)))
            and nonnegative(read)
            and nonnegative(at)
            and max(previous, card[i - 1][0]) < read <= at
            and (i == len(card) or at <= card[i][0])
            and type(event["pid"]) is int
            and event["pid"] == mark["pid"]
            and type(event["created_filetime"]) is int
            and event["created_filetime"] == mark["created_filetime"]
            and exited_at is None
            and event["before"] == "live"
            and event["after"] in ("live", "exited_empty")
            and event["reason"] == "probe-failure"
            and isinstance(event["pool"], dict)
            and event["pool"].keys() == native_pool.keys()
            and all(
                type(event["pool"][k]) is type(v) and event["pool"][k] == v
                for k, v in native_pool.items()
            )
            and nonnegative(event["free_bytes"])
            and event["free_bytes"] >= guard["admission"]["gpuFreeReserveBytes"]
            and event["child_growth_bytes"] <= guard["admission"]["childGpuCapBytes"]
        ):
            raise ValueError("unverified CUDA pool retry attempt")
        # The first reason is issued only after checking that the pool PID is
        # currently a Job member. For the second the sealed pool matches, and
        # the original Job must independently attest the exit.
        retired = (
            event["detail"] == RejectedPoolSample.EXCEEDS_CURRENT_USAGE
            and event["child_growth_bytes"] < native_pool["poolReservedBytes"]
        )
        exited = (
            event["detail"] == RejectedPoolSample.INVALID_ALLOCATION
            and event["after"] == "exited_empty"
        )
        if not (retired or exited):
            raise ValueError("unverified reason for rejected CUDA pool attempt")
        if event["after"] == "exited_empty" and exited_at is None:
            exited_at = i
        previous = at
    if exited_at is not None and (
        (zero is not None and zero["before"] != "exited_empty") or any(row[2] for row in card[exited_at:])
    ):
        raise ValueError("CUDA usage resumed after the original Job exited")


def render_scope(guard, native_pool, cuda_uuid, growth_count, *, completed_job=None, archived_owned_zero=False):
    raw = guard["fit_observation"]
    end = raw.get("cuda_render_end")
    if end is None:
        return None  # An older capture has no independent end-of-render witness.
    unknown = guard["cuda_attribution"]["unknown_pid_polls"]
    if (
        not nonnegative(end["pool_ticks"])
        or growth_count + end["pool_ticks"] + len(end.get("rejected_growth", []))
        != unknown
    ):
        raise ValueError(
            "CUDA fallback ticks differ from the supervisor growth records"
        )
    mark, zero = end["pool"], end["zero"]
    ded = guard["cuda_attribution"]["wddm_dedicated"]
    shared = raw["wddm_shared"]
    if mark is None:
        if end.get("rejected_growth"):
            raise ValueError("rejected CUDA pool attempt before verified output")
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
        if completed_job is not None and end.get("rejected_growth"):
            from scripts.ci.scail_plural.completed_job import terminal_scope
            claim, basis = completed_job
            return terminal_scope(guard, end, native_pool, card, claim, basis)
        if end.get("rejected_growth"):
            raise ValueError("rejected CUDA pool attempt without a verified owned zero")
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
    validate_rejected_growth(guard, end, native_pool, card)
    last_used = max(
        [row[0] for row in card if row[2] > 0]
        + [
            event["read_ns"]
            for event in end.get("rejected_growth", [])
            if event["child_growth_bytes"] > 0
        ]
    )
    scope = {
        "pid": mark["pid"],
        "created_filetime": mark["created_filetime"],
        "cuda_uuid": cuda_uuid,
        "pool_card_index": mark["card_index"],
        "zero_card_index": zero["card_index"],
        "dedicated_missing_after_output": ded["missing_active_polls"],
        "shared_missing_after_output": shared["missing_active_polls"],
    }
    if archived_owned_zero:
        from scripts.ci.scail_plural.completed_job import ordered_owned_zero
        claim, _ = completed_job
        scope.update(ordered_owned_zero(guard, end, card, claim, last_used))
        return scope
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
    return scope

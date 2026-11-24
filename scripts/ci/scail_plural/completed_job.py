"""Quiet GPU 0's completed Windows Job is distinct from a sampled owned zero."""

from scripts.media_campaign_supervisor import GPU_READS_PER_TICK
from scripts.ci.scail_plural.wddm_retry import valid_gap_clocks
from scripts.release.veda_cuda_guard import STABLE_MAX_UTILIZATION_PERCENT

# Driver bookkeeping on an idle card moves memory.used by a few MiB between reads.
RELEASE_BASELINE_TOLERANCE_MIB = 64
# A freshly launched process has a WDDM context below this size before any CUDA allocation.
LAUNCH_CONTEXT_BYTES = 1 << 20


OWNED_ZERO_CAPTURE = {
    "run_id": "44944284405", "run_attempt": "1", "artifact_id": "14725116132",
    "inference_revision": "f5c9ff80acd00334fd668e4ecc8d56be3347affd",
    "plan_sha256": "0f4b541c702bf45d607e138b56044fa8407a788b3b041d6e5f600b040097cf35",
    "archive_sha256": "39366924a8a5e09d4e1b0186615b6d3d5de7995ac18c87cad2f48ca918feb8e8",
    "receipt_sha256": "d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af",
}

LAST_LIVE_CAPTURE = {
    "run_id": "44969015053", "run_attempt": "1", "artifact_id": "14735122440",
    "inference_revision": "4fdd13cce7f590be0aed3c52a73a9cea9e6d1b54",
    "plan_sha256": "476e244074871e6e486107e4f65a515bf593a3c593d009c679a7ffbb2b2dee51",
    "archive_sha256": "ebb9d70b374f2e2e430fe292a6bf0f0d15d9c702142a657158d1bf525f17d388",
    "receipt_sha256": "d711f34b0c83b54b41b147921b9601fe999b74ccaec8823b6dceadcd6d9fc545",
}


def owned_job_empty(guard, claim):
    """The supervisor reaped the owned Windows Job: no member or owned process remains."""
    completion = guard.get("owned_completion")
    return (completion is not None and completion == claim.get("owned_execution_completion")
            and completion == dict(scope="supervisor-owned-windows-job", root_pid=guard.get("pid"),
                                   root_returncode=0, waited_returncode=0, owned_pids=[], job_member_pids=[]))


def quiet_release(guard, claim):
    """Five verified release reads on a quiet card whose owned memory is gone.

    Never an idle-card requirement: memory.used must return to this job's own admission
    reading (plus driver jitter), or the owned Job must be verified empty, in which case any
    remaining resident memory belongs to someone else. Utilization uses admission's quiet bound.
    """
    release = claim["release_background_activity"]["samples"]
    admission = claim.get("admission_background_activity", {}).get("samples", [])
    cards = [sample["selected_card"] for sample in release]
    baseline = max((sample["selected_card"]["used_mib"] for sample in admission), default=0)
    if not (claim["release_metadata_verification"] == {"status": "verified"} and len(release) == 5
            and type(baseline) is int and baseline >= 0
            and all(type(card["used_mib"]) is int and card["used_mib"] >= 0
                    and type(card["utilization_percent"]) is int
                    and 0 <= card["utilization_percent"] <= STABLE_MAX_UTILIZATION_PERCENT for card in cards)):
        return False
    returned = all(card["used_mib"] <= baseline + RELEASE_BASELINE_TOLERANCE_MIB for card in cards)
    return returned or owned_job_empty(guard, claim)


def unsampled_launch_reads(card, samples):
    """Card reads taken at launch, before the process had a WDDM counter, carry no memory.

    One such read is allowed only when the gauge's pre-allocation samples are exactly one
    fewer than the card's leading zero-owned rows, so every read after the first owned
    allocation still needs its own WDDM sample.
    """
    leading = next((index for index, row in enumerate(card) if row[2] > 0), len(card))
    before_allocation = next((index for index, sample in enumerate(samples)
                              if sample[1] >= LAUNCH_CONTEXT_BYTES), len(samples))
    return 1 if 0 < leading < len(card) and before_allocation == leading - 1 else 0


def bounded_terminal_gap(guard, attempt):
    """A WDDM read missed in the final attempt is bounded by that attempt's own card read.

    Windows removes a process's GPU Process Memory instance when the process
    closes its last adapter handle, which precedes its exit. A teardown read in
    that window finds no instance however it is retried. The attempt's card-wide
    upper bound covers the missed read only when it sits at or below the recorded
    dedicated peak and the attempt fits in one poll interval.
    """
    peak = max(value for _, value in guard["cuda_attribution"]["wddm_dedicated"]["samples"])
    return (attempt["at_ns"] - attempt["read_ns"] <= guard["admission"]["pollMillis"] * 1_000_000
            and attempt["global_used_upper_bytes"] <= peak)


# FILETIME counts 100 ns intervals from 1601; the Unix epoch is 11,644,473,600 s later.
FILETIME_UNIX_EPOCH_NS = 11_644_473_600 * 1_000_000_000
LOCAL_OFFSET_STEP_NS = 15 * 60 * 1_000_000_000
# Two Windows monotonic ticks (15.625 ms each) between a card record and its UTC stamp.
MAX_CLOCK_SPREAD_NS = 32_000_000


def final_attempt_unobserved(guard, end, card, samples, launch):
    """Each earlier card read or attempt holds its own WDDM sample; the final attempt holds none.

    Card rows and attempts carry monotonic times. PDH samples carry the local system
    FILETIME. The supervisor stamps each tenant-growth read in UTC just after its card
    row, which fixes monotonic to UTC within the stamps' spread, and the local offset
    is a whole number of 15-minute steps. Each sample must then lie after the previous
    record and by its own record, in windows wider apart than the clock tolerance, so
    no sample can stand in for another attempt's.
    """
    records = []
    for index in range(len(card) + 1):
        records.extend(e["at_ns"] for e in end["rejected_growth"] if e["next_card_index"] == index)
        if index < len(card):
            records.append(card[index][0])
    pool_rows = {e["card_index"] for e in end["pool_growth"]}
    growth = [e for e in guard["samples"] if e["phase"] == "cuda-global-used-upper-bound-sample"]
    offsets = []
    for index, (at_ns, free, used) in enumerate(card):
        if (len(offsets) < len(growth) and index not in pool_rows
                and (free, used) == (growth[len(offsets)]["gpuFreeBytes"], growth[len(offsets)]["childGlobalUsedGrowthBytes"])):
            offsets.append(growth[len(offsets)]["at_ns"] - at_ns)
    if (not growth or len(offsets) != len(growth) or min(offsets) < 0
            or max(offsets) - min(offsets) > MAX_CLOCK_SPREAD_NS
            or launch + len(samples) + 1 != len(records) or not samples):
        return False
    tolerance = 2 * (max(offsets) - min(offsets))
    times = [stamp * 100 - FILETIME_UNIX_EPOCH_NS - min(offsets) for stamp, _ in samples]
    local = sorted(t - records[launch + k] for k, t in enumerate(times))[len(times) // 2]
    local = round(local / LOCAL_OFFSET_STEP_NS) * LOCAL_OFFSET_STEP_NS
    for k, at in enumerate(times):
        own = launch + k
        if not ((own == 0 or records[own - 1] + tolerance < at - local) and at - local <= records[own] + tolerance):
            return False
    return True


def validate_guard_samples(guard, end, card):
    admission = guard["admission"]
    reserve, cap = admission["gpuFreeReserveBytes"], admission["childGpuCapBytes"]
    observations = [(free, used) for _, free, used in card]
    observations.extend((e["gpuFreeBytes"], e["childGlobalUsedGrowthBytes"]) for e in guard["samples"]
                        if e["phase"] == "cuda-global-used-upper-bound-sample")
    observations.extend((e["free_bytes"], e["child_growth_bytes"]) for e in end["pool_growth"] + end["rejected_growth"])
    if any(free < reserve or used > cap for free, used in observations):
        raise ValueError("a CUDA source sample breaches its approved guard")


def terminal_scope(guard, end, pool, card, claim, basis, *, fresh=False):
    from scripts.ci.scail_plural.cuda_stage import validate_rejected_growth
    from scripts.ci.flux1_pair.cuda_probe import RejectedPoolSample

    mark = end["pool"]
    events = end["rejected_growth"]
    final = [e for e in events if e["next_card_index"] == len(card)]
    if not (
        basis in ("archived-supervisor-return", "owned-job-completion")
        and guard["returncode"] == 0
        and quiet_release(guard, claim)
        and len(final) == GPU_READS_PER_TICK
        and events[-len(final):] == final
        and [e["card_index"] for e in end["pool_growth"]] == list(range(mark["card_index"], len(card)))
        and all(a["child_growth_bytes"] >= b["child_growth_bytes"] for a, b in zip(final, final[1:]))
    ):
        raise ValueError("quiet CUDA release or original terminal attempt sequence is unverified")
    validate_guard_samples(guard, end, card)
    validate_rejected_growth(guard, end, pool, card, completed_job=True)
    raw = guard["fit_observation"]
    tick_ns = guard["admission"]["pollMillis"] * 1_000_000
    counts = {}
    gaps = {}
    # Fresh runs only: PairCudaProbe reads dedicated memory, then checks Job
    # membership. An invalid allocation with an exited Job means the root left
    # before that check, so the shared read that follows, and a dedicated read
    # without a located gap, had no process to observe. The sample clocks must
    # place every other sample in its own earlier attempt.
    exited = (fresh and final[-1]["detail"] == RejectedPoolSample.INVALID_ALLOCATION
              and final[-1]["after"] == "exited_empty")
    for key, series in (("dedicated", guard["cuda_attribution"]["wddm_dedicated"]), ("shared", raw["wddm_shared"])):
        unresolved = [e for e in raw["wddm_retry"][key]["events"] if not e["recovered"]]
        unobservable = int(exited and not unresolved)
        launch = unsampled_launch_reads(card, series["samples"])
        if (len(unresolved) != series["missing_active_polls"] or len(unresolved) > 1
                or len(series["samples"]) + len(unresolved) != len(card) + len(events) - launch - unobservable
                or (unobservable and not final_attempt_unobserved(guard, end, card, series["samples"], launch))):
            # Each actual card attempt polled both gauges. A missing live
            # observation anywhere else would make this narrower witness unsafe.
            raise ValueError("unlocated WDDM gap during render")
        for gap in unresolved:
            gaps[key] = gap
            # KeepSystemGrowth records read_ns before PairCudaProbe polls PDH;
            # Observer records at_ns after BOTH gauges. Windows may return the
            # same tick for both reads. A previous attempt cannot overlap.
            if not (valid_gap_clocks(gap, tick_ns)
                    and final[-2]["at_ns"] < final[-1]["read_ns"] <= gap["start_ns"] <= gap["end_ns"] <= final[-1]["at_ns"]
                    and gap["retried"] is True and gap["retry_sample"] is None
                    and series["samples"] and gap["previous"] == series["samples"][-1]
                    and gap["local_filetime_before"] >= series["samples"][-1][0]):
                raise ValueError("WDDM gap is outside the final post-output CUDA attempt")
        counts[key+"_missing_after_output"] = len(unresolved)
    if fresh and (gaps or exited) and not bounded_terminal_gap(guard, final[-1]):
        raise ValueError("WDDM gap in the final CUDA attempt is not bounded by its card read")
    if ("dedicated" in gaps and "shared" in gaps
            and (gaps["dedicated"]["end_ns"] > gaps["shared"]["start_ns"]
                 or gaps["dedicated"]["local_filetime_after"] > gaps["shared"]["local_filetime_before"])):
        raise ValueError("terminal WDDM observations contradict the synchronous gauge order")
    return dict(basis=basis, pid=mark["pid"], created_filetime=mark["created_filetime"],
                cuda_uuid=mark["cuda_uuid"], pool_card_index=mark["card_index"],
                terminal_rejected_attempts=len(final), **counts)


class UnverifiedOrderedZero(ValueError):
    """The missed gauge cannot be located inside the final post-output attempt."""


def ordered_owned_zero(guard, end, card, claim, last_used, *, last_rejection_live=False, fresh=False):
    """Locate the two absent gauges within one rejected read, before the actual owned zero."""
    from scripts.ci.flux1_pair.cuda_probe import RejectedPoolSample

    mark, zero, events = end["pool"], end["zero"], end["rejected_growth"]
    completion = guard["owned_completion"]
    # The two archived Jobs retired at different observed calls. Preserve
    # each original ownership bracket, including the real following zero.
    before_zero = "live" if last_rejection_live else "exited_empty"
    if not (
        guard["returncode"] == 0 and completion == claim["owned_execution_completion"]
        and completion == dict(scope="supervisor-owned-windows-job", root_pid=guard["pid"],
                               root_returncode=0, waited_returncode=0, owned_pids=[], job_member_pids=[])
        and quiet_release(guard, claim)
        and len(events) >= 2 and zero["before"] == before_zero and zero["after"] == "exited_empty"
        and [e["card_index"] for e in end["pool_growth"]] == list(range(mark["card_index"], zero["card_index"]))
        and all(e["next_card_index"] == zero["card_index"] for e in events)
        and all(a["child_growth_bytes"] >= b["child_growth_bytes"] for a, b in zip(events, events[1:]))
    ):
        raise UnverifiedOrderedZero("unverified terminal CUDA attempt, Job, or physical release")
    validate_guard_samples(guard, end, card)
    final = events[-1]
    # PairCudaProbe polls dedicated before it checks Job membership. An invalid
    # allocation with an exited Job means the root left between those reads, so
    # the shared gauge that follows had no process left to read.
    exited_mid_attempt = fresh and not last_rejection_live and final["detail"] == RejectedPoolSample.INVALID_ALLOCATION
    if not (
        final["before"] == "live" and final["after"] == before_zero
        and (final["detail"] == RejectedPoolSample.EXCEEDS_CURRENT_USAGE or exited_mid_attempt)
        and final["child_growth_bytes"] > 0
        and mark["at_ns"] < events[-2]["at_ns"] < final["read_ns"] == last_used
        and final["at_ns"] < zero["at_ns"]
    ):
        raise UnverifiedOrderedZero("last positive rejected reading cannot establish the owned-zero boundary")

    raw = guard["fit_observation"]
    gaps = []
    tick_ns = guard["admission"]["pollMillis"] * 1_000_000
    for key, series in (("dedicated", guard["cuda_attribution"]["wddm_dedicated"]), ("shared", raw["wddm_shared"])):
        diagnostics = raw["wddm_retry"][key]
        if key == "shared" and exited_mid_attempt:
            if not (series["error"] is None and series["missing_active_polls"] == diagnostics["attempts"] == 0
                    and not diagnostics["events"]):
                raise UnverifiedOrderedZero("shared gauge has a gap after its root left the Job")
            continue
        if not (series["error"] is None and series["missing_active_polls"] == diagnostics["attempts"] == 1
                and len(diagnostics["events"]) == 1):
            raise UnverifiedOrderedZero("expected exactly one located terminal WDDM gap per gauge")
        gap = diagnostics["events"][0]
        start, finish = gap["start_ns"], gap["end_ns"]
        lower = gap["local_filetime_before"]
        if not (
            valid_gap_clocks(gap, tick_ns)
            and final["read_ns"] <= start <= finish <= final["at_ns"]
            and gap["retried"] is True and gap["recovered"] is False and gap["retry_sample"] is None
            and series["samples"] and gap["previous"] == series["samples"][-1] and lower >= gap["previous"][0]
        ):
            raise UnverifiedOrderedZero("WDDM retry does not fall inside its final rejected CUDA attempt")
        gaps.append(gap)
    if exited_mid_attempt:
        dedicated, = gaps
        shared_last = raw["wddm_shared"]["samples"][-1][0] if raw["wddm_shared"]["samples"] else None
        # The shared gauge must have no reading after the gap began.
        if not (bounded_terminal_gap(guard, final) and shared_last is not None
                and dedicated["previous"][0] <= shared_last < dedicated["local_filetime_before"]):
            raise UnverifiedOrderedZero("dedicated gap before the Job exit is unbounded or out of order")
        return dict(terminal_rejected_read_ns=final["read_ns"], terminal_rejected_at_ns=final["at_ns"])
    dedicated, shared = gaps
    # KeepSystemGrowth reads first (the loop bounds both gaps by its read_ns; the
    # archived captures keep their exact tie). PairCudaProbe polls dedicated before
    # it rejects the pool. The observer then polls shared before recording the attempt.
    if not ((fresh or dedicated["start_ns"] == final["read_ns"]) and dedicated["end_ns"] <= shared["start_ns"]
            and dedicated["local_filetime_after"] <= shared["local_filetime_before"]
            and shared["end_ns"] == final["at_ns"]):
        raise UnverifiedOrderedZero("terminal gauge timestamps contradict the original synchronous read order")
    if last_rejection_live:
        ded_samples = guard["cuda_attribution"]["wddm_dedicated"]["samples"]
        shared_samples = raw["wddm_shared"]["samples"]
        # Each completed card read and rejected substitution polled both gauges.
        # The last live attempt yielded the recorded gap; the following Job exit
        # yielded the real owned card zero and no PDH observation.
        live_polls = zero["card_index"] + len(events)
        if not (
            len(events) == 3
            and all(e["before"] == e["after"] == "live" and e["child_growth_bytes"] > 0
                    and e["detail"] == RejectedPoolSample.EXCEEDS_CURRENT_USAGE for e in events)
            and len(ded_samples) == len(shared_samples) and len(ded_samples) + 1 == live_polls
            and all(d[0] <= s[0] for d, s in zip(ded_samples, shared_samples))
            and all(s[0] < d[0] for s, d in zip(shared_samples, ded_samples[1:]))
        ):
            raise UnverifiedOrderedZero("incomplete paired WDDM observations before the last live attempt")
    return dict(terminal_rejected_read_ns=final["read_ns"], terminal_rejected_at_ns=final["at_ns"])

"""Quiet GPU 0's completed Windows Job is distinct from a sampled owned zero."""

from scripts.media_campaign_supervisor import GPU_READS_PER_TICK


OWNED_ZERO_CAPTURE = {
    "run_id": "44944284405", "run_attempt": "1", "artifact_id": "14725116132",
    "inference_revision": "f5c9ff80acd00334fd668e4ecc8d56be3347affd",
    "plan_sha256": "0f4b541c702bf45d607e138b56044fa8407a788b3b041d6e5f600b040097cf35",
    "archive_sha256": "39366924a8a5e09d4e1b0186615b6d3d5de7995ac18c87cad2f48ca918feb8e8",
    "receipt_sha256": "d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af",
}


def validate_guard_samples(guard, end, card):
    admission = guard["admission"]
    reserve, cap = admission["gpuFreeReserveBytes"], admission["childGpuCapBytes"]
    observations = [(free, used) for _, free, used in card]
    observations.extend((e["gpuFreeBytes"], e["childGlobalUsedGrowthBytes"]) for e in guard["samples"]
                        if e["phase"] == "cuda-global-used-upper-bound-sample")
    observations.extend((e["free_bytes"], e["child_growth_bytes"]) for e in end["pool_growth"] + end["rejected_growth"])
    if any(free < reserve or used > cap for free, used in observations):
        raise ValueError("a CUDA source sample breaches its approved guard")


def terminal_scope(guard, end, pool, card, claim, basis):
    from scripts.ci.scail_plural.cuda_stage import validate_rejected_growth

    mark = end["pool"]
    events = end["rejected_growth"]
    final = [e for e in events if e["next_card_index"] == len(card)]
    release = claim["release_background_activity"]["samples"]
    if not (
        basis in ("archived-supervisor-return", "owned-job-completion")
        and guard["returncode"] == 0
        and claim["release_metadata_verification"] == {"status": "verified"}
        and len(release) == 5
        and all(s["selected_card"]["used_mib"] == 0 and s["selected_card"]["utilization_percent"] == 0 for s in release)
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
    for key, series in (("dedicated", guard["cuda_attribution"]["wddm_dedicated"]), ("shared", raw["wddm_shared"])):
        unresolved = [e for e in raw["wddm_retry"][key]["events"] if not e["recovered"]]
        if (len(unresolved) != series["missing_active_polls"] or len(unresolved) > 1
                or len(series["samples"]) + len(unresolved) != len(card) + len(events)):
            # Each actual card attempt polled both gauges. A missing live
            # observation anywhere else would make this narrower witness unsafe.
            raise ValueError("unlocated WDDM gap during render")
        for gap in unresolved:
            gaps[key] = gap
            # KeepSystemGrowth records read_ns before PairCudaProbe polls PDH;
            # Observer records at_ns after BOTH gauges. Windows may return the
            # same tick for both reads. A previous attempt cannot overlap.
            start, finish = gap["start_ns"], gap["end_ns"]
            lower, upper = gap["local_filetime_before"], gap["local_filetime_after"]
            if not (all(type(v) is int and v > 0 for v in (start, finish, lower, upper))
                    and 0 <= finish - start <= tick_ns and 0 <= (upper - lower) * 100 <= tick_ns
                    and final[-2]["at_ns"] < final[-1]["read_ns"] <= gap["start_ns"] <= gap["end_ns"] <= final[-1]["at_ns"]
                    and gap["retried"] is True and gap["retry_sample"] is None
                    and series["samples"] and gap["previous"] == series["samples"][-1]
                    and gap["local_filetime_before"] >= series["samples"][-1][0]):
                raise ValueError("WDDM gap is outside the final post-output CUDA attempt")
        counts[key+"_missing_after_output"] = len(unresolved)
    if ("dedicated" in gaps and "shared" in gaps
            and (gaps["dedicated"]["end_ns"] > gaps["shared"]["start_ns"]
                 or gaps["dedicated"]["local_filetime_after"] > gaps["shared"]["local_filetime_before"])):
        raise ValueError("terminal WDDM observations contradict the synchronous gauge order")
    return dict(basis=basis, pid=mark["pid"], created_filetime=mark["created_filetime"],
                cuda_uuid=mark["cuda_uuid"], pool_card_index=mark["card_index"],
                terminal_rejected_attempts=len(final), **counts)


class UnverifiedOrderedZero(ValueError):
    """The missed gauge cannot be located inside the final post-output attempt."""


def ordered_owned_zero(guard, end, card, claim, last_used):
    """Locate the two absent gauges within one rejected read, before the actual owned zero."""
    from scripts.ci.flux1_pair.cuda_probe import RejectedPoolSample

    mark, zero, events = end["pool"], end["zero"], end["rejected_growth"]
    completion = guard["owned_completion"]
    release = claim["release_background_activity"]["samples"]
    if not (
        guard["returncode"] == 0 and completion == claim["owned_execution_completion"]
        and completion == dict(scope="supervisor-owned-windows-job", root_pid=guard["pid"],
                               root_returncode=0, waited_returncode=0, owned_pids=[], job_member_pids=[])
        and claim["release_metadata_verification"] == {"status": "verified"} and len(release) == 5
        and all(s["selected_card"]["used_mib"] == s["selected_card"]["utilization_percent"] == 0 for s in release)
        and len(events) >= 2 and zero["before"] == zero["after"] == "exited_empty"
        and [e["card_index"] for e in end["pool_growth"]] == list(range(mark["card_index"], zero["card_index"]))
        and all(e["next_card_index"] == zero["card_index"] for e in events)
        and all(a["child_growth_bytes"] >= b["child_growth_bytes"] for a, b in zip(events, events[1:]))
    ):
        raise UnverifiedOrderedZero("unverified terminal CUDA attempt, Job, or physical release")
    validate_guard_samples(guard, end, card)
    final = events[-1]
    if not (
        final["before"] == "live" and final["after"] == "exited_empty"
        and final["detail"] == RejectedPoolSample.EXCEEDS_CURRENT_USAGE
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
        if not (series["error"] is None and series["missing_active_polls"] == diagnostics["attempts"] == 1
                and len(diagnostics["events"]) == 1):
            raise UnverifiedOrderedZero("expected exactly one located terminal WDDM gap per gauge")
        gap = diagnostics["events"][0]
        start, finish = gap["start_ns"], gap["end_ns"]
        lower, upper = gap["local_filetime_before"], gap["local_filetime_after"]
        if not (
            all(type(v) is int and v > 0 for v in (start, finish, lower, upper))
            and 0 <= finish - start <= tick_ns and 0 <= (upper - lower) * 100 <= tick_ns
            and final["read_ns"] <= start <= finish <= final["at_ns"]
            and gap["retried"] is True and gap["recovered"] is False and gap["retry_sample"] is None
            and series["samples"] and gap["previous"] == series["samples"][-1] and lower >= gap["previous"][0]
        ):
            raise UnverifiedOrderedZero("WDDM retry does not fall inside its final rejected CUDA attempt")
        gaps.append(gap)
    dedicated, shared = gaps
    # KeepSystemGrowth reads first. PairCudaProbe polls dedicated before it rejects
    # the pool. The observer then polls shared before recording the rejected attempt.
    if not (dedicated["start_ns"] == final["read_ns"] and dedicated["end_ns"] <= shared["start_ns"]
            and dedicated["local_filetime_after"] <= shared["local_filetime_before"]
            and shared["end_ns"] == final["at_ns"]):
        raise UnverifiedOrderedZero("terminal gauge timestamps contradict the original synchronous read order")
    return dict(terminal_rejected_read_ns=final["read_ns"], terminal_rejected_at_ns=final["at_ns"])

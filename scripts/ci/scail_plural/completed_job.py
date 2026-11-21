"""Quiet GPU 0's completed Windows Job is distinct from a sampled owned zero."""

from scripts.media_campaign_supervisor import GPU_READS_PER_TICK


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
    admission = guard["admission"]
    reserve, cap = admission["gpuFreeReserveBytes"], admission["childGpuCapBytes"]
    observations = [(free, used) for _, free, used in card]
    observations.extend((e["gpuFreeBytes"], e["childGlobalUsedGrowthBytes"]) for e in guard["samples"]
                        if e["phase"] == "cuda-global-used-upper-bound-sample")
    observations.extend((e["free_bytes"], e["child_growth_bytes"]) for e in end["pool_growth"] + events)
    if any(free < reserve or used > cap for free, used in observations):
        raise ValueError("a CUDA source sample breaches its approved guard")
    validate_rejected_growth(guard, end, pool, card, completed_job=True)
    raw = guard["fit_observation"]
    tick_ns = admission["pollMillis"] * 1_000_000
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

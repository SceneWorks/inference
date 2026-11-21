"""Frozen SW-6954 measurements that supplement individually retained Tier 1 domains."""
import hashlib
import json
import statistics

from scripts.ci import video_tier1_cohorts as cohorts
from scripts.ci import video_tier1_evidence as evidence
from scripts.ci import video_tier1_cuda_diagnostic as targeted

NAME = "sw-6954-timing-memory-v1"
MANIFEST = "scripts/ci/video_tier1_retained.json"
TIMING_PAIRS = 12
CALLS = 16
TIMING = {
    "mlx-a14b-t2v-resident-heun", "mlx-a14b-i2v-resident-heun", "cuda-ti2v5b-t2v-heun",
    "cuda-svd-euler-cfg", "cuda-scail2-replacement-dpmpp",
}
MEMORY = {
    "cuda-ti2v5b-t2v-heun": (1,), "cuda-mochi-masked-euler": (1, 2, 5), "cuda-svd-euler-cfg": (1, 2, 4),
    "cuda-svd-euler-single": (3, 5), "cuda-svd-heun-cfg": (1, 2, 4),
    "cuda-scail2-animation-unipc": (1, 2), "cuda-scail2-replacement-dpmpp": (2, 3, 4),
}
RUNS = {(cohorts.WAN, "mlx"): "44857360714", (cohorts.WAN, "candle-cuda"): "44861650920",
        (cohorts.MOCHI_SVD, "candle-cuda"): "44861650824", (cohorts.SCAIL, "candle-cuda"): "44861650857"}
CUDA_UUID = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
PROTOCOL = {
    "pairs": TIMING_PAIRS, "samples_per_timing_child": CALLS, "warmups": 1, "memory_samples_per_child": 1,
    "timing_statistic": "ratio of role mean synchronized denoise times within each fresh adjacent AB/BA pair",
    "timing_interval": "ranks 3 and 10 of 12 ratios; 96.142578125% median coverage for independent pairs",
    "timing_pass": "upper rank <= 1.02", "timing_fail": "lower rank > 1.02",
    "timing_early_stopping": False, "timing_previous_samples_pooled": False,
    "memory_pass": "each of five original slots has like-for-like dedicated process peak ratio <= 1.02",
}


def bind(head, backend, selection, tracked, policy, name=NAME):
    cases = [case for case in cohorts.cases(selection, backend) if case in TIMING or case in MEMORY]
    if not cases:
        raise ValueError("no authorized follow-up cases on this lane")
    frozen = {"schemaVersion": 1, "backend": "windows-cuda" if backend == "candle-cuda" else "darwin-mlx",
              "deadlineSeconds": 180, "pollMillis": 125, "termGraceMillis": 2000, "hostFreeReserveBytes": 8 * 1024**3,
              "childFootprintCapBytes": 20 * 1024**3, **dict.fromkeys(("stdoutCapBytes", "stderrCapBytes", "eventCapBytes"), 4 * 1024**2)}
    if backend == "candle-cuda":
        uuid = targeted.GPU0_HARDWARE["uuid"] if name in targeted.GPU0_CONTINUATIONS else CUDA_UUID
        frozen.update(cudaDeviceUuid=uuid, gpuFreeReserveBytes=6 * 1024**3, childGpuCapBytes=12 * 1024**3)
    if frozen != policy:
        raise ValueError("follow-up requires its exact physical and supervision policy")
    raw = tracked(head, MANIFEST)
    manifest = json.loads(raw)
    run_id = RUNS[(selection, backend)]
    run = manifest["runs"][run_id]
    retained = {case: manifest["cases"][case] for case in cases}
    for case, row in retained.items():
        timing = "inconclusive" if case in TIMING else "pass"
        missing = tuple(i for i, v in enumerate(row["dedicated_pairs"], 1) if v == "inconclusive")
        if (row["run_id"] != run_id or row["receipt_sha256"] != run["receipt_sha256"] or row["head"] != run["head"]
                or row["timing"]["verdict"] != timing or row["timing"]["ceiling"] != 1.02
                or len(row["dedicated_pairs"]) != (5 if case in MEMORY else 0)
                or not set(row["dedicated_pairs"]) <= {"pass", "inconclusive"} or missing != MEMORY.get(case, ())
                or not row["retained_peaks"] or set(row["retained_peaks"].values()) != {"pass"}):
            raise ValueError("retained evidence differs from the frozen unresolved domains")
    return cases, run["head"], {
        "name": NAME, "retained_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "prior_hardware": run["hardware"], "historical_base_tree": run["base_tree"], "historical_head_tree": run["head_tree"],
        "reviewed_source": manifest["reviewed_source"], "reviewed_tree": manifest["reviewed_tree"],
        "protocol": PROTOCOL, "retained": retained,
    }


def groups(case, name=NAME):
    if name in targeted.TIMING_CONTINUATIONS:
        if case in targeted.TIMING_CALLS:
            yield "timing", range(1, targeted.CONTINUATION_PAIRS + 1), targeted.TIMING_CALLS[case]
        return
    if name in targeted.MEMORY_CONTINUATIONS:
        if case in targeted.CONTINUATION_MEMORY:
            yield "memory", targeted.CONTINUATION_MEMORY[case], 1
        return
    if name == targeted.NAME:
        if case in targeted.TIMING_CALLS:
            yield "timing", range(1, targeted.TIMING_PAIRS + 1), targeted.TIMING_CALLS[case]
        if case in targeted.MEMORY:
            yield "memory", targeted.MEMORY[case], 1
        return
    if case in TIMING:
        yield "timing", range(1, TIMING_PAIRS + 1), CALLS
    if case in MEMORY:
        yield "memory", MEMORY[case], 1


def timing(base, head, ceiling, *, pairs=TIMING_PAIRS, calls=CALLS, ranks=(3, 10)):
    if len(base) != pairs or len(head) != pairs:
        raise ValueError("missing follow-up timing pairs")
    durations = []
    for arm in (base, head):
        means = []
        for result in arm:
            samples = result["samples"]
            if len(samples) != calls:
                raise ValueError("incomplete grouped timing")
            means.append(statistics.mean(v["elapsed_ns"] for v in samples))
        durations.append(means)
    ratios = [h / b for b, h in zip(*durations, strict=True)]
    ranked = sorted(ratios)
    lower, upper = ranked[ranks[0] - 1], ranked[ranks[1] - 1]
    return {"paired_ratios": ratios, "median": statistics.median(ratios), "interval": [lower, upper], "ceiling": ceiling,
            "base_mean_ns": durations[0], "head_mean_ns": durations[1],
            "verdict": "pass" if upper <= ceiling else "fail" if lower > ceiling else "inconclusive"}


def compare(case, retained, observations, settings, name=NAME, *, adapter=None):
    diagnostic = name in targeted.DIAGNOSTICS
    source = "targeted-diagnostic" if diagnostic else "follow-up"
    if name in targeted.CONTINUATIONS:
        source = name
    schedule = {phase: (slots, calls) for phase, slots, calls in groups(case, name)}
    initial_ok = all(r["initial"]["sha256"] == retained["initial_sha256"] for r in observations.values())
    parity_ok = all(r["terminal"]["sha256"] == retained["terminal_sha256"] and r["trace_sha256"] == retained["trace_sha256"]
                    for r in observations.values())
    verdicts = ["pass" if initial_ok and parity_ok else "fail"]
    result = {"retained_run_id": retained["run_id"], "retained_receipt_sha256": retained["receipt_sha256"],
              "retained_peaks": retained["retained_peaks"], "initial_noise": "pass" if initial_ok else "fail",
              "parity": ("pass" if parity_ok else "fail") if initial_ok else "inconclusive"}
    expected_adapter = retained["dedicated_adapter"] if case in MEMORY else None
    if name in targeted.GPU0_CONTINUATIONS:
        from scripts.ci import wddm_process_memory
        expected_adapter = wddm_process_memory.validate_adapter(adapter, targeted.GPU0_HARDWARE["uuid"])
        if any(row["cuda_process"]["dedicated"]["adapter"] != expected_adapter for row in observations.values()):
            raise ValueError("fresh CUDA role used a different physical adapter")
        result.update(fresh_cuda_uuid=expected_adapter["cuda_uuid"], retained_cuda_uuid=CUDA_UUID)
    if diagnostic:
        result.update(followup_run_id=retained["followup_run_id"],
                      followup_receipt_sha256=retained["followup_receipt_sha256"])
    if "timing" in schedule:
        slots, calls = schedule["timing"]
        base, head = ([observations[f"{case}-timing-p{i}-{arm}"] for i in slots]
                      for arm in ("base", "head"))
        ranks = (6, 15) if diagnostic else (3, 10)
        if name in targeted.TIMING_CONTINUATIONS:
            ranks = (14, 27)
        measured = timing(base, head, settings["seconds_ratio_ceiling"], pairs=len(slots), calls=calls, ranks=ranks)
        measured["outer_steps"] = base[0]["samples"][0]["outer_steps"]
        for arm in ("base", "head"):
            measured[arm + "_mean_seconds_per_outer_step"] = [
                v / (1e9 * measured["outer_steps"]) for v in measured[arm + "_mean_ns"]]
        result["timing"] = {"source": source, **measured}
    elif name not in targeted.MEMORY_CONTINUATIONS:
        result["timing"] = {"source": retained.get("timing_source", "retained"), **retained["timing"]}
    if "timing" in result:
        verdicts.append(result["timing"]["verdict"])
    if case in MEMORY and name not in targeted.TIMING_CONTINUATIONS:
        slots = []
        for i, previous in enumerate(retained["dedicated_metrics"]["pairs"], 1):
            if i not in schedule.get("memory", ((), 1))[0]:
                if name in targeted.MEMORY_CONTINUATIONS:
                    continue
                if previous["verdict"] != "pass":
                    raise ValueError("retained slot lacks a completed process-memory comparison")
                slots.append({"source": "retained", "pair": i, **previous})
                continue
            b, h = (observations[f"{case}-memory-p{i}-{arm}"] for arm in ("base", "head"))
            for sample in (b, h):
                if sample["cuda_process"]["dedicated"]["adapter"] != expected_adapter:
                    raise ValueError("follow-up dedicated process counter differs from its retained adapter")
            metric = evidence.compare([b], [h], dict(settings, pairs=1))["metrics"]
            dedicated = metric["wddm_dedicated_process_bytes"]["pairs"][0]
            native = metric["cuda_process_bytes"]
            verdict = dedicated["verdict"] if evidence.accepted_metric(native) else "fail"
            slots.append({"source": source, "pair": i, "base_bytes": dedicated["base"]["sampled_peak_bytes"],
                          "head_bytes": dedicated["head"]["sampled_peak_bytes"], "ratio": dedicated["ratio"],
                          "verdict": verdict, "native": native})
        if len(slots) != (len(targeted.CONTINUATION_MEMORY[case]) if name in targeted.MEMORY_CONTINUATIONS else 5):
            raise ValueError("incomplete original process-memory slot set")
        result["dedicated_process_memory"] = {"pairs": slots, "verdict": evidence.peak_verdict(slots), "ceiling": 1.02}
        verdicts.append(result["dedicated_process_memory"]["verdict"])
    return dict(result, accepted=all(v == "pass" for v in verdicts))

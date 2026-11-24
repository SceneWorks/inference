"""Check raw video observations and compare each measured domain independently."""
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from scripts.ci import model_inference_proofs as proof


def owned_file(root: Path, name: str, cap: int):
    path = root / name
    proof.unlinked_path(path, root)
    if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > cap:
        raise ValueError(f"invalid owned proof file {name}")
    return path


def canonical_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def positive_int(value):
    return type(value) is int and value > 0


def state_digest(raw, shape):
    size = math.prod(shape) * 4
    if size > 16 * 1024**2:
        raise ValueError("state exceeds the fixed Tier 1 proof output cap")
    if len(raw) != size:
        raise ValueError("terminal does not contain the full fixed f32 tensor")
    smallest, largest = math.inf, -math.inf
    for (value,) in struct.iter_unpack("<f", raw):
        if not math.isfinite(value):
            raise ValueError("nonfinite terminal tensor")
        smallest = min(smallest, value)
        largest = max(largest, value)
    if smallest == largest:
        raise ValueError("unexpected constant terminal tensor")
    return {"shape": shape, "dtype": "f32-le", "sha256": hashlib.sha256(raw).hexdigest(),
            "min": smallest, "max": largest}


def sample_times(rows, count, outer_steps, deadline):
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("missing measured repeats")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"elapsed_ns", "outer_steps"}:
            raise ValueError("wrong synchronized sample fields")
        if (not positive_int(row["outer_steps"]) or row["outer_steps"] != outer_steps
                or not positive_int(row["elapsed_ns"])
                or row["elapsed_ns"] > deadline * 1e9):
            raise ValueError("invalid synchronized outer-step or whole-call timing")
    return rows


def case_spec(case):
    from scripts.ci import video_tier1_cohorts as cohorts

    return cohorts.fixture(cohorts.BY_NAME[case].model).case_spec(case)


def capture(row, initial, terminal, case, spec, settings, policy):
    # Native Wan configs report shift as a float (5.0); the Python schedule uses integer 5.
    expected_settings = dict(spec["settings"])
    if "shift" in expected_settings and spec["settings"]["shift"] is not None:
        expected_settings["shift"] = float(expected_settings["shift"])
    fields = {"case", "trace", "tensor_shape", "warmups", "samples", "allocator_peak_bytes", "settings"}
    if (not isinstance(row, dict) or set(row) != fields
            or row.get("case") != case or type(row.get("warmups")) is not int
            or row["warmups"] != settings["warmups"] or row.get("tensor_shape") != spec["tensor_shape"]
            or canonical_hash(row["settings"]) != canonical_hash(expected_settings)
            or canonical_hash(row.get("trace")) != canonical_hash(spec["trace"])
            or not positive_int(row.get("allocator_peak_bytes"))):
        raise ValueError("fixture route, shape, warmup or ordered trace differs from its contract")
    samples = sample_times(row["samples"], settings["samples_per_child"], spec["outer_steps"], policy.deadline_seconds)
    return {"samples": samples, "trace": row["trace"], "trace_sha256": canonical_hash(row["trace"]),
            "initial": state_digest(initial, spec["tensor_shape"]), "terminal": state_digest(terminal, spec["tensor_shape"])}


def timing_metric(base, head, ceiling):
    ratios = [h / b for b, h in zip(base, head, strict=True)]
    lo, hi = min(ratios), max(ratios)
    verdict = "pass" if hi <= ceiling else "fail" if lo > ceiling else "inconclusive"
    return {"paired_ratios": ratios, "median": statistics.median(ratios),
            "interval": [lo, hi], "ceiling": ceiling, "verdict": verdict}


def cuda_process(guard, pool_high, adapter=None):
    row = guard.get("cuda_attribution")
    fields = {"pid_peak_bytes", "unknown_pid_polls", "global_upper_bytes", "pool_fallback_peak_bytes"}
    expected = fields | ({"wddm_dedicated"} if adapter is not None else set())
    if not isinstance(row, dict) or set(row) != expected or any(type(row[k]) is not int or row[k] < 0 for k in fields):
        raise ValueError("invalid GPU measurement provenance")
    pid, upper, pool = row["pid_peak_bytes"], row["global_upper_bytes"], row["pool_fallback_peak_bytes"]
    if guard["peak_gpu_bytes"] != (max(pid, pool) or None) or pool > pool_high:
        raise ValueError("GPU high-water attribution differs from the supervisor or native pool")
    if row["unknown_pid_polls"] == 0:
        if not pid or upper or pool:
            raise ValueError("unexpected CUDA PID-only measurement")
        observed = {"basis": "owned-pid", "lower": pid, "upper": pid}
    else:
        if upper <= 0:
            raise ValueError("missing whole-card upper bound for an unknown owned PID")
        observed = {"basis": "wddm-bounded", "lower": max(pid, pool_high), "upper": max(pid, pool_high, upper)}
    peak = None
    if adapter is not None:
        from scripts.ci import wddm_process_memory
        peak = wddm_process_memory.sampled_peak(row["wddm_dedicated"], pid=guard["pid"], adapter=adapter)
        observed["dedicated"] = {"adapter": adapter, "sampled_peak_bytes": peak}
    if guard["peak_gpu_bytes"] is None and peak is None:
        raise ValueError("missing attributable GPU sample from the guard or dedicated process counter")
    return {**observed, **{k: row[k] for k in fields}}


def peak_verdict(pairs):
    outcomes = {pair["verdict"] for pair in pairs}
    if "fail" in outcomes:
        return "fail"
    return "inconclusive" if "inconclusive" in outcomes else "pass"


def accepted_metric(metric):
    if "required_pairs" in metric:
        return all(metric["pairs"][i]["verdict"] == "pass" for i in metric["required_pairs"])
    return metric["verdict"] == "pass"


def compare(base, head, settings):
    if len(base) != settings["pairs"] or len(head) != settings["pairs"]:
        raise ValueError("missing adjacent pairs")
    runs = [*base, *head]
    same_input = all(r["initial"] == base[0]["initial"] for r in runs)
    same_result = all(r["terminal"] == base[0]["terminal"] and r["trace"] == base[0]["trace"]
                      and r["trace_sha256"] == base[0]["trace_sha256"] for r in runs)
    parity = ("pass" if same_result else "fail") if same_input else "inconclusive"
    # Whole-call and per-outer-step units share the same ratio at the fixed step count.
    times = [[r["samples"][0]["elapsed_ns"] for r in arm] for arm in (base, head)]
    timing = timing_metric(*times, settings["seconds_ratio_ceiling"])
    metrics = {}
    for key, count in (("denoise_seconds", 1), ("seconds_per_outer_step", base[0]["samples"][0]["outer_steps"])):
        metrics[key] = {**timing, **{name: [value / (1e9 * count) for value in values]
                                    for name, values in zip(("base", "head"), times, strict=True)}}
    domains = base[0]["peaks"].keys()
    if not domains or any(r["peaks"].keys() != domains for r in runs):
        raise ValueError("incomparable peak-memory domains")
    ceiling = settings["peak_ratio_ceiling"]
    for domain in domains:
        ratios = [h["peaks"][domain] / b["peaks"][domain] for b, h in zip(base, head, strict=True)]
        metrics[domain] = {"paired_ratios": ratios, "ceiling": ceiling,
                           "verdict": "pass" if max(ratios) <= ceiling else "fail"}
    if any("cuda_process" in r for r in runs):
        pairs, dedicated_pairs = [], []
        has_dedicated = any("dedicated" in r["cuda_process"] for r in runs)
        for b, h in zip(base, head, strict=True):
            b, h = b["cuda_process"], h["cuda_process"]
            lower, upper = h["lower"] / b["upper"], h["upper"] / b["lower"]
            verdict = ("inconclusive" if h["basis"] != b["basis"] else "pass" if upper <= ceiling
                       else "fail" if lower > ceiling else "inconclusive")
            if has_dedicated:
                if "dedicated" not in b or "dedicated" not in h:
                    raise ValueError("missing dedicated process measurement source")
                bp, hp = b["dedicated"], h["dedicated"]
                ratio = None
                if (bp["adapter"] == hp["adapter"] and positive_int(bp["sampled_peak_bytes"])
                        and positive_int(hp["sampled_peak_bytes"])):
                    ratio = hp["sampled_peak_bytes"] / bp["sampled_peak_bytes"]
                dedicated_pairs.append({"base": bp, "head": hp, "ratio": ratio,
                                        "verdict": "inconclusive" if ratio is None else "pass" if ratio <= ceiling else "fail"})
                # A pool-to-card interval is diagnostic when sampling a WDDM process.
                # Keep any independently observed native PID regression as a required failure.
                if b["basis"] != "owned-pid" or h["basis"] != "owned-pid":
                    verdict = "inconclusive"
            pairs.append({"base": b, "head": h, "ratio_interval": [lower, upper], "verdict": verdict})
        metrics["cuda_process_bytes"] = {"pairs": pairs, "ceiling": ceiling, "verdict": peak_verdict(pairs)}
        if has_dedicated:
            # The PDH domain applies to every pair. Additionally require the independent
            # NVIDIA process measurement wherever both arms supplied complete PID accounting.
            metrics["cuda_process_bytes"]["required_pairs"] = [
                i for i, p in enumerate(pairs) if p["base"]["basis"] == p["head"]["basis"] == "owned-pid"]
            metrics["wddm_dedicated_process_bytes"] = {"basis": "wddm-pdh-dedicated-sampled",
                "pairs": dedicated_pairs, "ceiling": ceiling, "verdict": peak_verdict(dedicated_pairs)}
    return {"initial_noise": "pass" if same_input else "fail", "parity": parity, "metrics": metrics,
            "accepted": parity == "pass" and all(accepted_metric(m) for m in metrics.values())}

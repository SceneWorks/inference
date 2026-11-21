"""Check raw Wan observations and compare each measured domain independently."""
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
    from scripts.ci.wan_pair import fixture

    return fixture.case_spec(case)


def capture(row, initial, terminal, case, spec, settings, policy):
    # Native Wan configs report shift as a float (5.0); the Python schedule uses integer 5.
    expected_settings = dict(spec["settings"])
    if "shift" in expected_settings:
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


def cuda_process(guard, pool_high):
    row = guard.get("cuda_attribution")
    fields = {"pid_peak_bytes", "unknown_pid_polls", "global_upper_bytes", "pool_fallback_peak_bytes"}
    if not isinstance(row, dict) or set(row) != fields or any(type(n) is not int or n < 0 for n in row.values()):
        raise ValueError("invalid GPU measurement provenance")
    pid, upper, pool = row["pid_peak_bytes"], row["global_upper_bytes"], row["pool_fallback_peak_bytes"]
    if guard["peak_gpu_bytes"] != max(pid, pool) or pool > pool_high:
        raise ValueError("GPU high-water attribution differs from the supervisor or native pool")
    if row["unknown_pid_polls"] == 0:
        if not pid or upper or pool:
            raise ValueError("unexpected CUDA PID-only measurement")
        return {"basis": "owned-pid", "lower": pid, "upper": pid, **row}
    if upper <= 0:
        raise ValueError("missing whole-card upper bound for an unknown owned PID")
    return {"basis": "wddm-bounded", "lower": max(pid, pool_high),
            "upper": max(pid, pool_high, upper), **row}


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
        pairs = []
        for b, h in zip(base, head, strict=True):
            b, h = b["cuda_process"], h["cuda_process"]
            lower, upper = h["lower"] / b["upper"], h["upper"] / b["lower"]
            verdict = ("inconclusive" if h["basis"] != b["basis"] else "pass" if upper <= ceiling
                       else "fail" if lower > ceiling else "inconclusive")
            pairs.append({"base": b, "head": h, "ratio_interval": [lower, upper], "verdict": verdict})
        verdicts = {r["verdict"] for r in pairs}
        metrics["cuda_process_bytes"] = {"pairs": pairs, "ceiling": ceiling,
            "verdict": "fail" if "fail" in verdicts else "inconclusive" if "inconclusive" in verdicts else "pass"}
    return {"initial_noise": "pass" if same_input else "fail", "parity": parity, "metrics": metrics,
            "accepted": parity == "pass" and all(m["verdict"] == "pass" for m in metrics.values())}

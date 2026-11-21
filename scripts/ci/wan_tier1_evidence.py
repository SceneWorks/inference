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


def capture(row, raw, case, spec, settings, policy):
    fields = {"case", "trace", "tensor_shape", "warmups", "samples", "allocator_peak_bytes"}
    if (not isinstance(row, dict) or set(row) != fields
            or row.get("case") != case or type(row.get("warmups")) is not int
            or row["warmups"] != settings["warmups"] or row.get("tensor_shape") != spec["tensor_shape"]
            or canonical_hash(row.get("trace")) != canonical_hash(spec["trace"])
            or not positive_int(row.get("allocator_peak_bytes"))):
        raise ValueError("fixture route, shape, warmup or ordered trace differs from its contract")
    samples = sample_times(row["samples"], settings["samples"], spec["outer_steps"], policy.deadline_seconds)
    return {"samples": samples, "trace": row["trace"], "trace_sha256": canonical_hash(row["trace"]),
            "terminal": state_digest(raw, spec["tensor_shape"])}


def timing_metric(base, head, ceiling):
    ratios = [h / b for b, h in zip(base, head, strict=True)]
    lo, hi = min(ratios), max(ratios)
    verdict = "pass" if hi <= ceiling else "fail" if lo > ceiling else "inconclusive"
    return {"paired_ratios": ratios, "median": statistics.median(ratios),
            "interval": [lo, hi], "ceiling": ceiling, "verdict": verdict}


def compare(base, head, settings):
    parity = (base["terminal"] == head["terminal"] and base["trace"] == head["trace"]
              and base["trace_sha256"] == head["trace_sha256"])
    # Both arms proved the same count of completed outer steps. The conversion from
    # whole-call nanoseconds to seconds or seconds/step cancels in each paired ratio.
    times = [[sample["elapsed_ns"] for sample in arm["samples"]] for arm in (base, head)]
    timing = timing_metric(*times, settings["seconds_ratio_ceiling"])
    metrics = {}
    for key, count in (("denoise_seconds", 1), ("seconds_per_outer_step", base["samples"][0]["outer_steps"])):
        metrics[key] = {**timing, **{name: [value / (1e9 * count) for value in values]
                                    for name, values in zip(("base", "head"), times, strict=True)}}
    if base["peaks"].keys() != head["peaks"].keys() or not base["peaks"]:
        raise ValueError("incomparable peak-memory domains")
    for domain in base["peaks"]:
        ratio = head["peaks"][domain] / base["peaks"][domain]
        metrics[domain] = {"ratio": ratio, "ceiling": settings["peak_ratio_ceiling"],
                           "verdict": "pass" if ratio <= settings["peak_ratio_ceiling"] else "fail"}
    return {"parity": "pass" if parity else "fail", "metrics": metrics,
            "accepted": parity and all(m["verdict"] == "pass" for m in metrics.values())}

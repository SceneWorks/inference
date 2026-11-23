"""SGA PERF-1/2 decisions for the frozen six-pair sample."""
import math
from statistics import median

PAIRS = 6
CEILING = 1.02
INTERVAL_COVERAGE = 1 - 2 / 2**PAIRS


def _require_pairs(base, head):
    if len(base) != PAIRS or len(head) != PAIRS:
        raise ValueError("performance decision requires all six ordered pairs")


def compare_timing(base, head):
    _require_pairs(base, head)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for values in (base, head) for value in values):
        raise ValueError("timing must comprise finite positive durations")
    ratios = [h / b for b, h in zip(base, head, strict=True)]
    # An iid sample lies on only one side of a continuous population median
    # with probability 2/2**n. The min/max interval is conservative with ties.
    lower, upper = min(ratios), max(ratios)
    return {
        "pair_ratios": ratios, "median_ratio": median(ratios), "interval": [lower, upper],
        "status": "pass" if upper <= CEILING else "fail" if lower > CEILING else "inconclusive",
    }


def compare_peaks(base, head):
    _require_pairs(base, head)
    if any(type(value) is not int or value <= 0 for values in (base, head) for value in values):
        raise ValueError("memory requires positive byte high-water observations")
    base_peak, head_peak = max(base), max(head)
    ratio = head_peak / base_peak
    return {"base_peak_bytes": base_peak, "head_peak_bytes": head_peak, "ratio": ratio,
            "status": "pass" if 100 * head_peak <= 102 * base_peak else "fail"}

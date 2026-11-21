"""Frozen ENG-4b paired performance protocol for the existing FLUX proof."""
import hashlib
import json
import re

from scripts.ci import sga_paired_performance as paired
from scripts.ci.flux1_pair import evidence

MAX_CHILD_SECONDS = 360
BUILD = ("cargo", "build", "--locked", "-p", "candle-gen-flux", "--bin", "flux1-pair", "--release", "--features", "cuda")
RESULT_LIMITS = {"output.png": 256 * 256 * 4, "plan.json": 32 * 1024, "progress.json": 128}
TIMING = ("denoise_seconds_per_outer_step", "end_to_end_seconds")
MEMORY = ("host_peak_working_set_bytes", "cuda_pool_used_high_bytes", "cuda_pool_reserved_high_bytes")
CONTRACT = {
    "pairs_per_variant": paired.PAIRS,
    "build_profile": "cargo release, locked sources, CUDA on both arms",
    "order": "AB, BA, AB, BA, AB, BA per variant, one fresh process per arm and pair",
    "warmup": "one identical materialized generation in each measured process, excluded from timing",
    "timing_scope": "resident planned generation through host RGB, excluding plan, load and evidence writes",
    "step_scope": "synchronized interval from first Euler evaluation through terminal state, divided by three",
    "memory_scope": "host process peak working set and continuous CUDA pool peaks across load, warm-up and sample",
    "memory_statistic": "head maximum divided by base maximum, per matched domain",
    "timing_statistic": "median paired head/base ratio with distribution-free min/max interval",
    "interval_coverage": paired.INTERVAL_COVERAGE,
    "head_base_ceiling": paired.CEILING,
    "max_child_seconds": MAX_CHILD_SECONDS,
    "output_fidelity": "separate instrumented pair for state bits and RGB8; each measured RGB8 matches that pair",
}


def cases(variants):
    for variant in variants:
        for pair in range(1, paired.PAIRS + 1):
            order = evidence.ARMS if pair % 2 else evidence.ARMS[::-1]
            for arm in order:
                yield f"measure-{variant}-{pair}-{arm}", variant, arm


def require_matched_builds(arms):
    base, head = (arms[arm]["build"] for arm in evidence.ARMS)
    if (any(build["build_command"] != list(BUILD) or not build["toolchain"] for build in (base, head))
            or base["toolchain"] != head["toolchain"]):
        raise ValueError("terminal comparison requires matched CUDA release builds and toolchains")


def read_sample(read, variant, preflight, reference, policy):
    """Read fields from the producer's files or the verified published copies."""
    model = "flux1_" + variant
    markers = [line for line in read("stdout.log").decode("utf-8").splitlines() if line.startswith("FLUX1_PAIR_")]
    dtype = reference["states"][-1]["dtype"]
    if len(markers) != 4 or markers[-1] != "FLUX1_PAIR_COMPLETE " + model:
        raise ValueError("missing complete warm-up and sample timings on the fidelity device/dtype")
    denoise = [re.fullmatch(r"FLUX1_PAIR_DENOISE ([1-9][0-9]*) (\w+) candle-cuda", line) for line in markers[:2]]
    timing = re.fullmatch(r"FLUX1_PAIR_TIMING ([1-9][0-9]*) ([1-9][0-9]*)", markers[2])
    if not timing or any(not row or row[2] != dtype for row in denoise):
        raise ValueError("missing complete warm-up and sample timings on the fidelity device/dtype")
    end_to_end_ns, host_peak = map(int, timing.groups())
    denoise_ns = int(denoise[1][1])
    if (not denoise_ns <= end_to_end_ns <= policy["deadlineSeconds"] * 10**9
            or host_peak > policy["childFootprintCapBytes"]):
        raise ValueError("incoherent timing or peak working set above the child cap")
    plan = json.loads(read("plan.json"))
    if plan != preflight or json.loads(read("progress.json")) != reference["steps"]:
        raise ValueError("performance route or completed steps differ from fidelity")
    if hashlib.sha256(read("output.png")).hexdigest() != reference["png_sha256"]:
        raise ValueError("measured RGB8 differs from fidelity")
    guard = json.loads(read("supervisor.json"))
    pool = json.loads(read("cuda-pool.json"))
    keys = ("poolUsedBytes", "poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")
    pool_bytes = [pool[key] for key in keys]
    if (guard["returncode"] != 0 or pool["pid"] != guard["pid"]
            or pool["cudaDeviceUuid"] != policy["cudaDeviceUuid"]
            or any(type(value) is not int or value <= 0 for value in pool_bytes)
            or pool_bytes[0] > min(pool_bytes[1], pool_bytes[2])
            or max(pool_bytes[1:3]) > pool_bytes[3] or pool_bytes[3] > policy["childGpuCapBytes"]):
        raise ValueError("invalid owned CUDA allocation high-water record")
    return dict(zip((*TIMING, *MEMORY),
                    (denoise_ns / 3e9, end_to_end_ns / 1e9, host_peak, pool_bytes[1], pool_bytes[3])))


def evaluate(samples, variants):
    """Evaluate paired timing and like-for-like peak memory at the frozen ceiling."""
    result = {}
    for variant in variants:
        metrics = {}
        for metric in (*TIMING, *MEMORY):
            base, head = ([samples[f"measure-{variant}-{index}-{arm}"][metric]
                           for index in range(1, paired.PAIRS + 1)] for arm in evidence.ARMS)
            compare = paired.compare_timing if metric in TIMING else paired.compare_peaks
            metrics[metric] = compare(base, head)
        result[variant] = metrics
    statuses = {row["status"] for metrics in result.values() for row in metrics.values()}
    status = "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"
    return {"status": status, "variants": result}

"""Fixed CUDA diagnostic schedule and the original and follow-up receipts it supplements."""
import hashlib
import json
import subprocess
import time

from scripts.ci import video_tier1_cohorts as cohorts

NAME = "sw-6954-targeted-cuda-v1"
MANIFEST = "scripts/ci/video_tier1_cuda_diagnostic_retained.json"
MANIFEST_SHA256 = "8fc228aa249af227b66ae969e4d6f764ef77ecd69e4c0b00bd4be75b772ae24c"
ROOT_HANDLE_SOURCE = {
    "scripts/media_campaign_windows.py": "2ac50fe9388ee434c6e88743c35d388c1cf6ae92e41a211601dafe15b38423fe",
    "scripts/ci/wddm_process_memory.py": "a2f1c1fb67675e45d67360b21e0804a9f2818238105412c92928fe5b4cd5a644",
}
TIMING_PAIRS = 20
TIMING_CALLS = {"cuda-ti2v5b-t2v-heun": 256, "cuda-svd-euler-cfg": 32}
MEMORY = {"cuda-mochi-masked-euler": (1, 2, 5), "cuda-svd-euler-cfg": (2, 4),
          "cuda-svd-heun-cfg": (2,), "cuda-scail2-replacement-dpmpp": (4,)}
RUNS = {cohorts.WAN: "44893811394", cohorts.MOCHI_SVD: "44893811411", cohorts.SCAIL: "44893811440"}
CLOCK_TIMEOUT = 2
CLOCK_FIELDS = ("clocks.current.sm", "clocks.current.memory", "pstate", "temperature.gpu", "utilization.gpu")


def bind(head, backend, selection, tracked, previous):
    if backend != "candle-cuda":
        raise ValueError("targeted diagnostic requires the CUDA lane")
    raw = tracked(head, MANIFEST)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("targeted diagnostic retained manifest digest changed")
    manifest = json.loads(raw)
    if manifest["original_manifest_sha256"] != previous["retained_manifest_sha256"]:
        raise ValueError("targeted diagnostic original manifest digest changed")
    if any(hashlib.sha256(tracked(head, path)).hexdigest() != digest for path, digest in ROOT_HANDLE_SOURCE.items()):
        raise ValueError("targeted CUDA diagnostic requires the reviewed Job root handle monitor")
    run_id = RUNS[selection]
    run = manifest["runs"][run_id]
    if (run["base_tree"] != previous["historical_base_tree"] or run["head_tree"] != previous["historical_head_tree"]
            or run["hardware"] != previous["prior_hardware"]):
        raise ValueError("prior CUDA follow-up used different source trees or hardware")
    rows = {}
    for case, original in previous["retained"].items():
        latest = manifest["cases"][case]
        slots = latest["dedicated_metrics"]["pairs"]
        states = [v["verdict"] for v in slots]
        missing = tuple(i for i, verdict in enumerate(states, 1) if verdict == "inconclusive")
        matching = ("head", "fixture", "dedicated_adapter", "initial_sha256", "terminal_sha256",
                    "trace_sha256", "retained_peaks")
        if (any(latest[k] != original[k] for k in matching)
                or latest["run_id"] != run_id or latest["receipt_sha256"] != run["receipt_sha256"]
                or latest["original_run_id"] != original["run_id"]
                or latest["original_receipt_sha256"] != original["receipt_sha256"]
                or len(slots) != 5 or states != latest["dedicated_pairs"] or missing != MEMORY.get(case, ())
                or not set(states) <= {"pass", "inconclusive"} or latest["timing"]["ceiling"] != 1.02
                or latest["timing"]["verdict"] != ("inconclusive" if case in TIMING_CALLS else "pass")):
            raise ValueError("targeted diagnostic differs from prior receipt domains")
        rows[case] = {**original, **{k: latest[k] for k in ("dedicated_metrics", "dedicated_pairs", "timing", "timing_source")},
                      "followup_run_id": run_id, "followup_receipt_sha256": run["receipt_sha256"]}
    cases = [case for case in rows if case in TIMING_CALLS or case in MEMORY]
    protocol = {**previous["protocol"], "pairs": TIMING_PAIRS, "samples_per_timing_child": TIMING_CALLS,
                "timing_interval": "ranks 6 and 15 of 20 paired ratios; 95.861053466796875% median coverage for independent pairs",
                "clock_diagnostics": {"boundary": "outside each timing role, before spawn and after exit",
                                      "query_timeout_seconds": CLOCK_TIMEOUT, "fields": list(CLOCK_FIELDS)}}
    return cases, {**previous, "name": NAME, "root_handle_source_sha256": ROOT_HANDLE_SOURCE, "protocol": protocol, "retained": rows,
                   "prior_followup_run": run, "prior_followup_manifest_sha256": MANIFEST_SHA256}


def clock_snapshot(uuid):
    """Read the approved physical card outside the timed child. Availability is diagnostic."""
    from scripts.media_campaign_windows import trusted_nvidia_smi
    snapshot = {"monotonic_ns": time.monotonic_ns(), "uuid": uuid}
    args = ["--id=" + uuid, "--query-gpu=" + ",".join(("uuid", *CLOCK_FIELDS)), "--format=csv,noheader,nounits"]
    try:
        raw = subprocess.check_output([trusted_nvidia_smi(), *args], text=True, timeout=CLOCK_TIMEOUT,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return {**snapshot, "status": type(error).__name__}
    columns = [v.strip() for v in raw.strip().split(",")]
    if len(columns) != len(CLOCK_FIELDS) + 1 or columns[0] != uuid:
        return {**snapshot, "status": "unrecognized-response"}
    return {**snapshot, "status": "observed", "values": dict(zip(CLOCK_FIELDS, columns[1:], strict=True))}


def verify_clocks(measurement, uuid):
    for key in ("before", "after"):
        row = measurement["clock_diagnostics"][key]
        stamp = row["monotonic_ns"]
        if (row["uuid"] != uuid or type(stamp) is not int or stamp <= 0
                or (stamp > measurement["start_ns"] if key == "before" else stamp < measurement["end_ns"])):
            raise ValueError("CUDA clock observation has wrong identity or timing boundary")
        if row["status"] == "observed":
            values = row["values"]
            if (set(values) != set(CLOCK_FIELDS)
                    or any(not isinstance(v, str) or not 0 < len(v) <= 64 for v in values.values())):
                raise ValueError("wrong CUDA clock fields")
        elif not isinstance(row["status"], str) or not row["status"] or "values" in row:
            raise ValueError("invalid unavailable CUDA clock observation")


TIMING_CONTINUATION = "sw-6954-cuda-conditioning-timing-v1"
MEMORY_CONTINUATION = "sw-6954-cuda-process-memory-v1"
GPU0_TIMING = "sw-6954-gpu0-conditioning-timing-v1"
GPU0_MEMORY = "sw-6954-gpu0-process-memory-v1"
GPU0_CONTINUATIONS = (GPU0_TIMING, GPU0_MEMORY)
TIMING_CONTINUATIONS = (TIMING_CONTINUATION, GPU0_TIMING)
MEMORY_CONTINUATIONS = (MEMORY_CONTINUATION, GPU0_MEMORY)
CONTINUATIONS = (*TIMING_CONTINUATIONS, *MEMORY_CONTINUATIONS)
# Recorded GPU0 inventory from the two-card Windows real-weights host.
GPU0_HARDWARE = {
    "uuid": "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0", "index": "0",
    "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "memory.total": "97887", "driver_version": "596.36",
}
GPU0_PROTOCOL_SOURCE = "f971a8c5eabd55c02cc20efa1413fd37ddc0a498"
DIAGNOSTICS = (NAME, *CONTINUATIONS)
CONTINUATION_PAIRS = 40
CONTINUATION_MEMORY = {"cuda-mochi-masked-euler": (1, 2), "cuda-svd-euler-cfg": (2,)}
CONDITIONING_TEMPLATE = "scripts/ci/wan_pair/conditioning.rs.in"
RETRY_REASON = "missing device counter for the live owned process during an active poll"

# The last eligible targeted receipts; later first-attempt or aborted measurements are never pooled.
TARGETED_HISTORY = {
    "wan-curated-tier1": {
        "run_id": "44901236162",
        "attempt": "2",
        "receipt_sha256": "95d6b155716550f0901bb63271af03650e56d216acabccef0eba11232b982950",
        "controller": "f38d03cc41b151b6020e6bee07cf2687061683f6",
        "base": "38dab65f5d9640396cf2946cd4ee866ce46630c6",
        "head": "339cca4716844d5b24b03ff726b35007e873ed08",
        "base_tree": "583299608475223416f4e1bcad3b4276eeefb055",
        "head_tree": "5da487978f98b81af0993bf8063bfa50acf1b41a",
        "unresolved_memory": {},
        "unresolved_timing": [
            "cuda-ti2v5b-t2v-heun"
        ],
        "measured_targets": {
            "base": {
                "wan": {
                    "binary_sha256": "2a89f6baa9ac9c888a2465e825c2ce1aea29268692ac72177e50a30a3f457cf5",
                    "module_sha256": "41131126f5e7c3ac107bfa8aedf36a590c111b45c07cd924270dc4a2c7d23ed6"
                }
            },
            "head": {
                "wan": {
                    "binary_sha256": "9b0578a03e4c083fc2e1aab280be9be035b8af5aa09f1273a6df50a26d3144e8",
                    "module_sha256": "e68d6ef635f00e41849f427e68811e009c42acebe708334375306f6327ba1bc8"
                }
            }
        }
    },
    "mochi-svd-tier1": {
        "run_id": "44901238050",
        "attempt": "2",
        "receipt_sha256": "5e9c0f2121210a64519578ba1758e9d608cc1e164c93225f917cb40f9305320b",
        "controller": "f38d03cc41b151b6020e6bee07cf2687061683f6",
        "base": "c7911db0e6cdcc41762326e2f8aad7e60634c2a1",
        "head": "339cca4716844d5b24b03ff726b35007e873ed08",
        "base_tree": "f514a98282c65e91ab02e4a8df386cda0a60e72f",
        "head_tree": "5da487978f98b81af0993bf8063bfa50acf1b41a",
        "unresolved_memory": {
            "cuda-mochi-masked-euler": [
                1,
                2
            ],
            "cuda-svd-euler-cfg": [
                2
            ]
        },
        "unresolved_timing": [
            "cuda-svd-euler-cfg"
        ],
        "measured_targets": {
            "base": {
                "mochi": {
                    "binary_sha256": "e77d702b740b16fe2eeb496d9eca55dbebca4696c7b1568bc471e0e9235fd972",
                    "module_sha256": "c10879659e12630699d880c63ea1626358ad4e62517aaca47b195b45a370b3d9"
                },
                "svd": {
                    "binary_sha256": "12ae89559ac6537e5b019ec6f21d219e7c43164500ed6e9a8ad2b78abce1d867",
                    "module_sha256": "e3fe73807925ccec67d434789136c1d84f533cc40e9729070ee1d924d6eb1f36"
                }
            },
            "head": {
                "mochi": {
                    "binary_sha256": "bf3795817844e769c34ad81779d8551ddda665868e95cd496f70c3ba4c29b139",
                    "module_sha256": "c10879659e12630699d880c63ea1626358ad4e62517aaca47b195b45a370b3d9"
                },
                "svd": {
                    "binary_sha256": "546e1fd2f71bf9c1e24c024e01b9499f3b041fe4fe620d3a2e6ce514f0eb6494",
                    "module_sha256": "e3fe73807925ccec67d434789136c1d84f533cc40e9729070ee1d924d6eb1f36"
                }
            }
        }
    }
}


def continuation(name, selection, extension):
    history = TARGETED_HISTORY.get(selection)
    if name not in CONTINUATIONS or not history or (name in MEMORY_CONTINUATIONS and selection != cohorts.MOCHI_SVD):
        raise ValueError("no authorized CUDA continuation on this selection")
    if history["base_tree"] != extension["historical_base_tree"] or history["head_tree"] != extension["historical_head_tree"]:
        raise ValueError("targeted continuation uses different historical engine trees")
    timing = name in TIMING_CONTINUATIONS
    cases = history["unresolved_timing"] if timing else list(history["unresolved_memory"])
    if (timing and (len(cases) != 1 or cases[0] not in TIMING_CALLS)
            or not timing and history["unresolved_memory"] != {k: list(v) for k, v in CONTINUATION_MEMORY.items()}):
        raise ValueError("continuation case set differs from the eligible targeted receipt")
    protocol = {
        "scope": "timing only" if timing else "dedicated process-memory slots only",
        "timing_previous_samples_pooled": False, "early_stopping": False, "ceiling": 1.02,
        "device_policy": "non-isolated shared headroom on the same physical UUID",
    }
    if name in GPU0_CONTINUATIONS:
        protocol.update(device_policy="non-isolated shared headroom on physical GPU0 for both fresh arms",
                        fresh_hardware=GPU0_HARDWARE, retained_hardware=extension["prior_hardware"],
                        retained_scope="historical measurements on GPU1 only",
                        measurement_source=GPU0_PROTOCOL_SOURCE)
    if timing:
        count = TIMING_CALLS[cases[0]]
        protocol.update(pairs=CONTINUATION_PAIRS, order="20 AB and 20 BA in adjacent alternating pairs",
                        warmups=count, samples_per_timing_child=count, interval_ranks=[14, 27],
                        timing_statistic="median of paired ratios of role mean synchronized denoise durations",
                        timing_interval="ranks 14 to 27 of 40 independent pairs; 96.152269% median coverage",
                        timing_pass="upper rank <= 1.02", timing_fail="lower rank > 1.02",
                        clock_diagnostics=extension["protocol"]["clock_diagnostics"])
    else:
        protocol.update(original_slots=history["unresolved_memory"], warmups=1, samples_per_child=1,
                        retry_reason=RETRY_REASON, retry="one immediate, same owned PID, creation time, LUID and poll")
    return cases, {**extension, "name": name, "protocol": protocol, "prior_targeted": history}


class DedicatedRetryObserver:
    def __init__(self, cuda, policy):
        from scripts.ci.scail_plural.wddm_retry import SameTickSamples
        self.cuda = cuda
        cuda.dedicated = SameTickSamples(cuda.dedicated, policy.poll_millis, immediate=True)

    def __getattr__(self, name):
        return getattr(self.cuda, name)

    def fit_observation(self):
        gauge = self.cuda.dedicated
        return {"reason": RETRY_REASON, "pid": gauge.record["pid"], "created_filetime": gauge.record["created_filetime"],
                "adapter": gauge.record["adapter"], "diagnostics": gauge.diagnostics}


def verify_retry(guard, policy, adapter):
    from scripts.ci.scail_plural.wddm_retry import validate_retry
    record = guard["cuda_attribution"]["wddm_dedicated"]
    witness = guard["fit_observation"]
    if (witness["reason"] != RETRY_REASON or witness["pid"] != record["pid"] or record["pid"] != guard["pid"]
            or witness["created_filetime"] != record["created_filetime"] or witness["adapter"] != record["adapter"]
            or record["adapter"] != adapter):
        raise ValueError("dedicated retry differs from its guarded process or adapter")
    validate_retry(witness["diagnostics"], record, policy.poll_millis)


def verify_conditioning(row, samples, case, child_seconds):
    count = TIMING_CALLS[case]
    if set(row) != {"conditioning_ns", "conditioning_end_ns", "measured_start_ns", "measured_end_ns"}:
        raise ValueError("missing fixed native conditioning boundary")
    warm = row["conditioning_ns"]
    end, start, stop = (row[k] for k in ("conditioning_end_ns", "measured_start_ns", "measured_end_ns"))
    if (not isinstance(warm, list) or len(warm) != count or len(samples) != count
            or any(type(v) is not int or v <= 0 for v in [end, start, stop, *warm])
            or not sum(warm) <= end <= start < stop <= child_seconds * 1e9
            or sum(s["elapsed_ns"] for s in samples) > stop - start):
        raise ValueError("native conditioning or measured duration lies outside its fixed role window")

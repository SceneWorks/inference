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

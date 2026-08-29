"""Fail-closed structural validator for the SC-20673 evidence receipt."""
import argparse, hashlib, json, sys
from pathlib import Path

REQUIRED = ("B", "Hq", "Hkv", "GQA", "Sq", "Skv", "D", "group_size", "bits", "code_format", "dtype", "masks", "simd_groups", "tails_nonmultiples")

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path); a = p.parse_args()
    root = (a.root or Path(__file__).parents[1] / "docs/architecture/receipts").resolve()
    coverage_file = root / "sc-20673-coverage.json"
    raw_file = root / "sc-20673-metal-reproduction.json"
    sidecar = root / "sc-20673-coverage.json.sha256"
    raw_sidecar = root / "sc-20673-metal-reproduction.json.sha256"
    for file, check in ((coverage_file, sidecar), (raw_file, raw_sidecar)):
        expected = check.read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected: return 1
    j = json.loads(coverage_file.read_text(encoding="utf-8")); raw = json.loads(raw_file.read_text(encoding="utf-8"))
    axes = j.get("axes", {})
    missing = [k for k in REQUIRED if k not in axes]
    if missing or not axes.get("tails_nonmultiples"):
        print("invalid SC-20673 receipt: missing required axes/timing declaration", file=sys.stderr); return 1
    for key in ("upstream_commit", "inference_base", "host", "dependency_lock"):
        if not j.get("provenance", {}).get(key): return 1
    host = j["provenance"]["host"]
    if raw["upstream"]["commit"] != j["provenance"]["upstream_commit"] or any(f"{k}={v}" not in host for k, v in raw["host"].items()): return 1
    timing = raw.get("coverage", {}).get("timing_measurements", [])
    if len(timing) != len(raw["commands"]): return 1
    required_timing = ("compile_first_dispatch_process_s", "steady_synchronized_process_s", "synchronization_process_boundary_s", "transient_peak_rss_bytes")
    if not all(all(isinstance(t.get(k), (int, float)) and t[k] > 0 for k in required_timing) for t in timing): return 1
    if not all(isinstance(r.get("timing", {}).get(k), (int, float)) and r["timing"][k] > 0 for k in required_timing for r in raw["commands"]): return 1
    if not all(isinstance(row.get("bytes"), int) and row["bytes"] > 0 for row in j["physical_bytes"]): return 1
    if not all(isinstance(r.get("timing", {}).get("transient_peak_rss_bytes"), int) and r["timing"]["transient_peak_rss_bytes"] > 0 for r in raw["commands"]): return 1
    rows = "\n".join(raw.get("coverage", {}).get("observed_raw_rows", []))
    if raw.get("coverage", {}).get("results") != j.get("results"): return 1
    if not all((str(value) in rows or f"{value:.2f}" in rows) for result in j["results"].values() for value in result.get("speedup", [])): return 1
    if not j.get("independent_reference") or not j.get("physical_bytes") or not j.get("msl") or not j.get("unsupported"): return 1
    print("SC-20673 receipt structure: OK"); return 0
if __name__ == "__main__": raise SystemExit(main())

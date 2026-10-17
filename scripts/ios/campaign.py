#!/usr/bin/env python3
"""Catalog-derived iOS campaign. Listing is CPU/weights-free; run is explicit, cold per case.

This collects evidence, not admission authority. No diagnostic, missing row, killed process,
refusal or skipped cell is a production pass. No engine/cap/evidence file is modified.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("q4", "q8", "bf16")
EDGES = (256, 512, 768, 1024, 1280)
STATUSES = {"successful", "admitted_refused", "diagnostic_success", "diagnostic_failure", "skipped"}


def case_key(value):
    p = value.split("/")
    if len(p) != 6 or any(not s or not all(c.isascii() and (c.isalnum() or c in "_-") for c in s) for s in p):
        raise argparse.ArgumentTypeError("case must be provider/route/tier/edge/mode/rung")
    if p[2] not in TIERS or p[3] not in tuple(map(str, EDGES)) or p[5] not in tuple(map(str, range(5))):
        raise argparse.ArgumentTypeError("case outside finite tier/geometry/rung domain")
    return value


def positive(value):
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive integer") from exc
    if not 1 <= n <= 20000:
        raise argparse.ArgumentTypeError("budget must be 1..20000")
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "run", "table"))
    p.add_argument("--catalog", type=Path, help="previous weights-free catalog JSON export")
    p.add_argument("--provider")
    p.add_argument("--route")
    p.add_argument("--tier", choices=TIERS)
    p.add_argument("--edge", type=int, choices=EDGES)
    p.add_argument("--mode")
    p.add_argument("--rung", type=int, choices=range(5))
    p.add_argument("--case", type=case_key)
    p.add_argument("--all-cases", action="store_true", help="explicitly select whole catalog")
    p.add_argument("--diagnostic", action="store_true", help="request bounded evidence collection, never bypass preflight")
    p.add_argument("--next-edge", type=int, choices=EDGES, help="additional refusal-only probe, NEVER generates")
    p.add_argument("--device", help="explicit paired device UDID; no simulator execution")
    p.add_argument("--output", type=Path, help="JSONL evidence destination; new file only")
    p.add_argument("--input", type=Path, help="JSONL evidence to render as a table")
    p.add_argument("--max-cases", type=positive, default=20000)
    p.add_argument("--timeout-seconds", type=positive, default=1800)
    return p


def catalog(args):
    if args.catalog:
        data = json.loads(args.catalog.read_text(encoding="utf-8"))
    else:
        env = dict(os.environ, CARGO_BUILD_JOBS="4")
        env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
        result = subprocess.run(["cargo", "run", "--locked", "--offline", "--manifest-path", "ios-host/smoke/Cargo.toml", "--features", "media", "--example", "campaign_catalog", "-j", "4"], cwd=ROOT, env=env, check=True, capture_output=True, text=True, encoding="utf-8", timeout=args.timeout_seconds)
        data = json.loads(result.stdout)
    if data.get("schema") != 1 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("empty/malformed catalog is not coverage")
    keys = [case_key(k) for k in data["cases"]]
    if len(keys) != len(set(keys)) or len(keys) > 20000:
        raise ValueError("duplicate cases or catalog exceeds finite campaign budget")
    return data


def selected(args, key):
    if args.case:
        return key == args.case
    fields = key.split("/")
    values = (args.provider, args.route, args.tier, args.edge, args.mode, args.rung)
    return all(v is None or str(v) == f for v, f in zip(values, fields))


def failure(key, reason):
    return {"schema": 1, "case": key, "status": "diagnostic_failure", "production_success": False, "reason": reason,
            "elapsed_seconds": None, "mlx_active_peak_bytes": None, "mlx_cache_peak_bytes": None,
            "mlx_peak_bytes": None, "mlx_active_cache_peak_bytes": None, "phys_footprint_peak_bytes": None, "min_headroom_mib": None}


def parse_report(text, key):
    rows = [json.loads(line.removeprefix("CAMPAIGN_JSON ")) for line in text.splitlines() if line.startswith("CAMPAIGN_JSON ")]
    if len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("schema") != 1 or rows[0].get("case") != key or rows[0].get("status") not in STATUSES:
        return failure(key, "missing, duplicate, stale or malformed case report; process termination is NOT a refusal proof")
    row = rows[0]
    success = row["status"] == "successful"
    if row.get("production_success") is not success or (row.get("diagnostic") and success):
        return failure(key, "inconsistent status/admission accounting")
    if success:
        for name in ("elapsed_seconds", "mlx_active_peak_bytes", "mlx_cache_peak_bytes", "mlx_active_cache_peak_bytes", "mlx_peak_bytes", "phys_footprint_peak_bytes", "min_headroom_mib"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return failure(key, "production result missing finite device telemetry: " + name)
        if row["phys_footprint_peak_bytes"] == 0:
            return failure(key, "zero physical footprint is not a device measurement")
    return row


def invoke(args, key, refusal_only=False):
    # A new app process per case, even for consecutive geometries of the same model. The launcher
    # terminates the previous process and deletes the old report before launch.
    env = dict(os.environ)
    env.pop("IOS_SMOKE_UNADMITTED", None)
    env.pop("IOS_SMOKE_SANA_CAPTURE", None)
    for name in ("IOS_CAMPAIGN_CASE", "IOS_CAMPAIGN_LIST", "IOS_CAMPAIGN_DIAGNOSTIC"):
        env.pop(name, None)
    env["IOS_CAMPAIGN_REFUSAL_ONLY"] = "1" if refusal_only else "0"
    command = [str(ROOT / "scripts/ios/run_smoke.sh"), "--media", "--campaign-case", key, "--device", args.device]
    if args.diagnostic:
        command.append("--campaign-diagnostic")
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=args.timeout_seconds)
        row = parse_report(result.stdout, key)
        if refusal_only and row["status"] != "admitted_refused":
            return failure(key, "next-up probe did not establish refusal BEFORE weight loading: " + row.get("reason", ""))
        return row
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return failure(key, str(exc))


def render(rows):
    lines = ["| Case | Status | Time s | MLX peak MiB | Phys footprint MiB | Min headroom MiB |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        def metric(name, scale=1):
            v = r.get(name)
            return "unavailable" if v is None else f"{v / scale:.2f}"
        lines.append(f"| {r['case']} | {r['status']} | {metric('elapsed_seconds')} | {metric('mlx_peak_bytes', 1048576)} | {metric('phys_footprint_peak_bytes', 1048576)} | {metric('min_headroom_mib')} |")
    # Do not invent a largest-completed size or extrapolate a refusal from a smaller failure.
    groups = {}
    for r in rows:
        fields = r["case"].split("/")
        group = "/".join(fields[:3] + fields[4:])
        if r["status"] == "successful" and r.get("production_success") is True:
            groups[group] = max(groups.get(group, 0), int(fields[3]))
    lines.append("\nLargest admitted completed edges (diagnostics excluded): " + json.dumps(groups, sort_keys=True))
    return "\n".join(lines)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.action == "table":
        if not args.input:
            p.error("table requires --input")
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
        print(render(rows))
        return 0
    data = catalog(args)
    keys = [k for k in data["cases"] if selected(args, k)]
    if not keys:
        p.error("selection matches no actual catalog cases")
    if args.action == "list":
        print(json.dumps(dict(data, cases=keys), indent=2))
        return 0
    if not args.device or not args.output:
        p.error("run requires explicit --device and new --output")
    if not args.all_cases and not args.case and not any((args.provider, args.route, args.tier, args.edge, args.mode, args.rung is not None)):
        p.error("run requires selectors or --all-cases; listing never starts a campaign")
    if args.next_edge is not None and args.diagnostic:
        p.error("next-up refusal proof must use admitted mode, not diagnostics")
    if args.next_edge is not None and any(args.next_edge <= int(key.split("/")[3]) for key in keys):
        p.error("next-edge must exceed every selected completed edge")
    rows = []
    # Exclusive create refuses to overwrite evidence from a preceding device session.
    with args.output.open("x", encoding="utf-8") as output:
        for index, key in enumerate(keys):
            if index >= args.max_cases:
                row = dict(failure(key, "explicit max-cases execution budget reached"), status="skipped")
            else:
                row = invoke(args, key)
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            if args.next_edge is not None and row["status"] == "successful":
                fields = key.split("/")
                if args.next_edge <= int(fields[3]):
                    p.error("next-edge must exceed the completed edge")
                fields[3] = str(args.next_edge)
                probe = invoke(args, "/".join(fields), refusal_only=True)
                output.write(json.dumps(probe) + "\n")
                output.flush()
                rows.append(probe)
    print(render(rows))
    # Refusals are valid evidence, not successful renders. The terminal AC still needs a completed
    # largest size per model/tier plus its actual next-up refusal, and the 24212 dependency.
    return 0 if all(r["status"] == "successful" and r.get("production_success") for r in rows) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"campaign: {error}", file=sys.stderr)
        sys.exit(2)

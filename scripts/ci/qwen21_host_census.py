#!/usr/bin/env python3
"""Read-only host attribution before any Qwen model work; never acceptance evidence."""

import argparse
import datetime
import json
import math
from pathlib import Path
import plistlib
import subprocess
import tempfile


PS_FIELDS = "pid=,ppid=,uid=,rss=,comm="
DRIVER_METRICS = {
    "Alloc system memory", "In use system memory", "In use system memory (driver)",
    "In use video memory", "In use video memory (driver)", "Texture memory",
    "Buffer memory", "System memory used", "Device memory used",
}
MAX_QUERY_BYTES = 2 * 1024 * 1024


def query(command):
    try:
        # Spool only this bounded-time diagnostic child's stdout; never capture stderr/argv/env.
        # Read at most the receipt limit into memory, even if a driver exposes a large registry.
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.DEVNULL,
                                    timeout=5, check=False)
            if result.returncode:
                return None, f"query exited {result.returncode}"
            output.seek(0)
            raw = output.read(MAX_QUERY_BYTES + 1)
            if len(raw) > MAX_QUERY_BYTES:
                return None, "query exceeded 2 MiB output bound"
            return raw, None
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, type(error).__name__


def process_rows(raw):
    rows = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5 or not all(value.isdecimal() for value in fields[:4]):
            raise ValueError("unexpected ps census format")
        pid, ppid, uid, rss = map(int, fields[:4])
        rows.append({"pid": pid, "ppid": ppid, "uid": uid, "rssKiB": rss,
                     "comm": fields[4]})
    return sorted(rows, key=lambda row: (-row["rssKiB"], row["pid"]))


def driver_rows(raw):
    rows = []
    def visit(node):
        if isinstance(node, dict):
            stats = node.get("PerformanceStatistics", {})
            if isinstance(stats, dict):
                metrics = {key: value for key, value in stats.items()
                           if key in DRIVER_METRICS and type(value) in (int, float)
                           and math.isfinite(value) and value >= 0}
                if metrics:
                    rows.append(metrics)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(plistlib.loads(raw))
    return rows


def collect():
    receipt = {"kind": "HOST_DIAGNOSTIC_ONLY", "acceptanceEvidence": False,
               "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "processFields": ["PID", "PPID", "UID", "RSS", "comm"],
               "rssUnit": "KiB; per-process RSS may overlap shared pages",
               "driverMetricUnits": "raw driver-reported values; units are not inferred",
               "errors": {}}
    ps, error = query(["/bin/ps", "-axo", PS_FIELDS])
    if error:
        receipt["errors"]["processes"] = error
    else:
        try:
            receipt["processes"] = process_rows(ps)
        except (ValueError, UnicodeDecodeError):
            receipt["errors"]["processes"] = "unsupported process census format"
    for name, command in {
        "vmStat": ["/usr/bin/vm_stat"],
        "hardware": ["/usr/sbin/sysctl", "hw.model", "hw.memsize", "hw.pagesize",
                     "machdep.cpu.brand_string"],
        "pressureLevel": ["/usr/sbin/sysctl", "kern.memorystatus_vm_pressure_level"],
        "memoryPressureQuery": ["/usr/bin/memory_pressure", "-Q"],
    }.items():
        raw, error = query(command)
        if error:
            receipt["errors"][name] = error
        else:
            receipt[name] = raw.decode("utf-8", errors="strict")
    raw, error = query(["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-a"])
    if error:
        receipt["errors"]["driverMetrics"] = error
    else:
        try:
            receipt["driverMetrics"] = driver_rows(raw)
            if not receipt["driverMetrics"]:
                receipt["errors"]["driverMetrics"] = "no allowlisted numeric metrics exposed"
        except (ValueError, plistlib.InvalidFileException):
            receipt["errors"]["driverMetrics"] = "unsupported driver census format"
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = collect()
    (args.out / "HOST_DIAGNOSTIC_ONLY.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("HOST_DIAGNOSTIC_ONLY: read-only host census saved; not acceptance evidence")


if __name__ == "__main__":
    main()

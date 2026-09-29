#!/usr/bin/env python3
"""Fail before provisioning if the designated Metal quality runner is occupied or undersized."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# The exact-M2 F32 CPU reference measured this peak RSS on 2026-09-29.
CPU_REFERENCE_PEAK_BYTES = 23_184_818_176
MODEL_BYTES = 7_261_441_640  # Pinned YuE2-3B safetensors manifest.
# Originals plus two derived tiers can each approach a full checkpoint. Leave one
# further checkpoint's worth for the release build and staging on a cold runner.
DISK_HEADROOM_BYTES = 4 * MODEL_BYTES
BUSY_NAME = re.compile(
    r"(?:^|/)(?:cargo|rustc|candle_audio_yue2-[^ /]+|sceneworks-worker|sceneworks-rust-api|candle-gen|mlx-gen)(?:$| )",
    re.IGNORECASE,
)


def output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def available_memory(vm_stat: str) -> int:
    match = re.search(r"page size of (\d+) bytes", vm_stat)
    if not match:
        raise ValueError("vm_stat did not report page size")
    counts = {}
    for name, value in re.findall(r"^Pages (free|inactive|speculative):\s+(\d+)\.", vm_stat, re.M):
        counts[name] = int(value)
    if set(counts) != {"free", "inactive", "speculative"}:
        raise ValueError("vm_stat omitted available-memory counters")
    return int(match.group(1)) * sum(counts.values())


def competing_processes(ps: str, own_pid: int) -> list[str]:
    busy = []
    workers = []
    for line in ps.splitlines():
        columns = line.strip().split(None, 2)
        if len(columns) != 3 or not columns[0].isdigit():
            continue
        pid, comm, args = int(columns[0]), columns[1], columns[2]
        if pid == own_pid:
            continue
        if Path(comm).name == "Runner.Worker":
            workers.append(pid)
        if BUSY_NAME.search(comm) or BUSY_NAME.search(args):
            busy.append(f"{pid}: {comm}")
    if len(workers) > 1:
        busy.append(f"{len(workers)} Runner.Worker processes")
    return busy


def assess(runner: str, total: int, available: int, disk: int, busy: list[str]) -> list[str]:
    errors = []
    if runner != "nax-macos-2":
        errors.append(f"expected nax-macos-2, got {runner!r}")
    if total < CPU_REFERENCE_PEAK_BYTES or available < CPU_REFERENCE_PEAK_BYTES:
        errors.append(
            f"physical/available memory {total}/{available} below measured F32 reference "
            f"peak {CPU_REFERENCE_PEAK_BYTES}"
        )
    if disk < DISK_HEADROOM_BYTES:
        errors.append(
            f"disk free {disk} below four pinned-checkpoint equivalents {DISK_HEADROOM_BYTES}"
        )
    if busy:
        errors.append(f"competing inference/build processes: {', '.join(busy)}")
    return errors


def existing_parent(path: Path) -> Path:
    while not path.exists():
        if path == path.parent:
            raise ValueError(f"no existing parent for {path}")
        path = path.parent
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--label", choices=("initial", "before-metal", "before-profile"), default="initial")
    args = parser.parse_args()
    runner = os.environ.get("RUNNER_NAME", "")
    total = int(output(["sysctl", "-n", "hw.memsize"]).strip())
    available = available_memory(output(["vm_stat"]))
    disk = shutil.disk_usage(Path.cwd()).free
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"),
        )
    )
    hub_disk = shutil.disk_usage(existing_parent(hub)).free
    busy = competing_processes(output(["ps", "-axo", "pid=,comm=,args="]), os.getpid())
    errors = assess(runner, total, available, min(disk, hub_disk), busy)
    record = {
        "runner": runner,
        "hostname": output(["hostname"]).strip(),
        "macos_version": output(["sw_vers", "-productVersion"]).strip(),
        "machine": output(["uname", "-m"]).strip(),
        "physical_memory_bytes": total,
        "available_memory_bytes": available,
        "workspace_disk_free_bytes": disk,
        "hub_disk_free_bytes": hub_disk,
        "competing_processes": busy,
        "minimum_memory_bytes": CPU_REFERENCE_PEAK_BYTES,
        "minimum_disk_bytes": DISK_HEADROOM_BYTES,
        "admitted": not errors,
        "errors": errors,
    }
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / f"preflight-{args.label}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""External phys_footprint guard for one MLX real-weight process TREE (SW-7200), ported from
SceneWorks' `scripts/ci/iris-mlx-real-weights/tree_guard.py` for the decode-speedups campaign,
whose benchmark process is `cargo test` with the test binary as its child.

    mlx_tree_guard.py <cap GiB> <wall seconds> -- <command...>

Refuses to start (exit 3) unless `memory_pressure` reports at least the cap plus 10 GiB free. Then
runs the command with its output on this process's stdout, polls the process tree every 0.25 s
and SIGTERMs the whole tree when its summed footprint crosses the cap or the wall clock runs out —
the only kill path: nothing here ever SIGKILLs (an MLX process killed mid-command-buffer can hang
the GPU). Report-only beyond that: nothing is gated on the peak.

Every reading is BYTES from the kernel ledger — `proc_pid_rusage` RUSAGE_INFO_V4
`ri_phys_footprint` (Activity Monitor's "Memory", including Metal / unified-memory buffers). The
cap is GiB (2^30 bytes). The closing `GUARD_SUMMARY` line reports:

  * `peak_tree_sampled`: the largest SUM over the live tree seen at a poll (a lower bound of the
    true tree peak, 0.25 s resolution);
  * `root_lifetime_max`: the kernel's exact high-water mark of the root process (here `cargo`);
  * `max_process_lifetime_max`: the largest exact high-water mark of any one process in the tree —
    the test binary's peak, the number to read for a model run;
  * `tree_lifetime_max_sum`: the sum of every process's high-water mark (an upper bound of the
    tree peak: the processes need not peak together).
"""

import ctypes
import os
import re
import signal
import subprocess
import sys
import time

GIB = 1 << 30
POLL_SECONDS = 0.25
HEADROOM_GIB = 10
# struct rusage_info_v4, as uint64 words (ri_uuid is words 0-1).
RUSAGE_INFO_V4 = 4
RI_PHYS_FOOTPRINT = 9
RI_LIFETIME_MAX_PHYS_FOOTPRINT = 30

_libc = None
_buf = (ctypes.c_uint64 * 64)()


def rusage(pid):
    """`(phys_footprint, lifetime_max_phys_footprint)` bytes of `pid`, or `None` once it is gone."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if _libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(_buf)) != 0:
        return None
    return int(_buf[RI_PHYS_FOOTPRINT]), int(_buf[RI_LIFETIME_MAX_PHYS_FOOTPRINT])


def tree_pids(root):
    """`root` and every live descendant."""
    listing = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=False
    ).stdout
    children = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = [], [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(children.get(pid, []))
    return seen


def free_gib():
    """Free unified memory in GiB: `memory_pressure`'s free percentage of `hw.memsize`, or `None`."""
    pressure = subprocess.run(
        ["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False
    ).stdout
    memsize = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
    ).stdout.strip()
    match = re.search(r"free percentage: (\d+)%", pressure)
    if not match or not memsize.isdigit():
        return None
    return int(memsize) * int(match.group(1)) / 100 / GIB


def gib(value):
    return f"{value / GIB:.3f}"


def main(argv):
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    cap_gib, wall_s, command = float(argv[0]), float(argv[1]), argv[3:]
    free = free_gib()
    if free is None or free < cap_gib + HEADROOM_GIB:
        shown = "unknown" if free is None else f"{free:.1f}"
        print(f"GUARD: not started: {shown} GiB free < cap {cap_gib:g} + {HEADROOM_GIB} GiB",
              flush=True)
        return 3
    cap_bytes = int(cap_gib * GIB)
    start = time.time()
    child = subprocess.Popen(command)
    peak_sampled = 0
    lifetime_max = {}
    stopped = None
    while child.poll() is None:
        total = 0
        pids = tree_pids(child.pid)
        for pid in pids:
            reading = rusage(pid)
            if reading is None:
                continue
            total += reading[0]
            lifetime_max[pid] = max(lifetime_max.get(pid, 0), reading[1])
        peak_sampled = max(peak_sampled, total)
        if stopped is None:
            if total > cap_bytes:
                stopped = f"tree footprint {gib(total)} GiB > cap {cap_gib:g} GiB"
            elif time.time() - start > wall_s:
                stopped = f"wall time over {wall_s:g}s"
            if stopped is not None:
                print(f"GUARD: {stopped}, SIGTERM", flush=True)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        time.sleep(POLL_SECONDS)
    code = child.wait()
    root_max = lifetime_max.get(child.pid, 0)
    print(
        f"GUARD_SUMMARY exit={code}"
        f" peak_tree_sampled_bytes={peak_sampled}"
        f" root_lifetime_max_bytes={root_max}"
        f" max_process_lifetime_max_bytes={max(lifetime_max.values(), default=0)}"
        f" tree_lifetime_max_sum_bytes={sum(lifetime_max.values())}"
        f" processes={len(lifetime_max)} cap_gib={cap_gib:g} free_gib_at_start={free:.1f}"
        f" wall_s={time.time() - start:.0f}" + (f' stopped="{stopped}"' if stopped else ""),
        flush=True,
    )
    return code if code is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

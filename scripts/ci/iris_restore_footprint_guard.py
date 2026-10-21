"""External memory guard for the sc-25688 real-weight MLX run (`mlx-iris-restore-train`), also
used by sc-25689's `mlx-iris-curricula-train` (caps 20 GB / 100 GB, see its run script) and
sc-25690's `mlx-iris-memory-probe`.

Runs argv[3:] as a child, polls its macOS phys_footprint (proc_pid_rusage RUSAGE_INFO_V4 — includes
Metal / unified-memory buffers) every 0.25 s, SIGTERMs it when the footprint crosses the cap (GB,
argv[1]) or the wall clock exceeds argv[2] seconds, and prints the peak. Report-only beyond the kill:
nothing is gated on the peak. Caps the job passes: 40 GB for the reference check and the LoRA run,
100 GB for the paper's full fine-tune of the 3B generator (f32 masters + AdamW moments + grads + EMA
≈ 60 GB before activations and the frozen critic).

Two peaks are printed, both decimal GB. `peak_phys_footprint` is the highest POLLED sample, so a
transient shorter than the poll interval can fall between two samples. `lifetime_max_phys_footprint`
is the kernel's own continuous high-water (`ri_lifetime_max_phys_footprint`), read at every poll: it
misses nothing up to the last poll before the child exits.

A child that ignores SIGTERM is SIGKILLed after `IRIS_GUARD_TERM_GRACE_S` seconds (default 60), so
the guard itself always returns.
"""

import ctypes
import os
import signal
import subprocess
import sys
import time

RUSAGE_INFO_V4 = 4
# Index of each field in `struct rusage_info_v4`, counted in uint64 words (the leading 16-byte
# `ri_uuid` is words 0-1).
RI_PHYS_FOOTPRINT = 9
RI_LIFETIME_MAX_PHYS_FOOTPRINT = 30
POLL_S = 0.25


def main(argv: list[str]) -> int:
    cap = float(argv[1]) * 1e9
    wall = float(argv[2])
    cmd = argv[3:]
    grace = float(os.environ.get("IRIS_GUARD_TERM_GRACE_S", "60"))

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    buf = (ctypes.c_uint64 * 64)()

    def footprints(pid: int) -> tuple[int, int]:
        """`(current, lifetime max)` phys_footprint in bytes; `(0, 0)` once the child is gone."""
        if libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
            return 0, 0
        return int(buf[RI_PHYS_FOOTPRINT]), int(buf[RI_LIFETIME_MAX_PHYS_FOOTPRINT])

    start = time.time()
    child = subprocess.Popen(cmd)
    peak = 0
    lifetime_max = 0
    reason = None
    while child.poll() is None:
        fp, lifetime = footprints(child.pid)
        peak = max(peak, fp)
        lifetime_max = max(lifetime_max, lifetime)
        if fp > cap:
            reason = f"footprint {fp / 1e9:.1f} GB > cap {cap / 1e9:.0f} GB"
        elif time.time() - start > wall:
            reason = f"wall {time.time() - start:.0f}s > {wall:.0f}s"
        if reason:
            print(f"[guard] SIGTERM: {reason}", flush=True)
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                print(f"[guard] SIGKILL: child ignored SIGTERM for {grace:.0f}s", flush=True)
                child.kill()
            break
        time.sleep(POLL_S)
    code = child.wait()
    print(
        f"[guard] exit={code} peak_phys_footprint={peak / 1e9:.2f} GB"
        f" lifetime_max_phys_footprint={lifetime_max / 1e9:.2f} GB"
        f" wall={time.time() - start:.1f}s" + (f" ({reason})" if reason else ""),
        flush=True,
    )
    return code if code is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

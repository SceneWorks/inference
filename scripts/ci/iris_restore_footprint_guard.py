"""External memory guard for the sc-25688 real-weight MLX run (`mlx-iris-restore-train`).

Runs argv[3:] as a child, polls its macOS phys_footprint (proc_pid_rusage RUSAGE_INFO_V4 — includes
Metal / unified-memory buffers) every 0.25 s, SIGTERMs it when the footprint crosses the cap (GB,
argv[1]) or the wall clock exceeds argv[2] seconds, and prints the peak. Report-only beyond the kill:
nothing is gated on the peak. Caps the job passes: 40 GB for the reference check and the LoRA run,
100 GB for the paper's full fine-tune of the 3B generator (f32 masters + AdamW moments + grads + EMA
≈ 60 GB before activations and the frozen critic).
"""

import ctypes
import signal
import subprocess
import sys
import time

cap = float(sys.argv[1]) * 1e9
wall = float(sys.argv[2])
cmd = sys.argv[3:]

libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
buf = (ctypes.c_uint64 * 64)()


def footprint(pid: int) -> int:
    rc = libc.proc_pid_rusage(pid, 4, ctypes.byref(buf))
    return int(buf[9]) if rc == 0 else 0


start = time.time()
child = subprocess.Popen(cmd)
peak = 0
reason = None
while child.poll() is None:
    fp = footprint(child.pid)
    peak = max(peak, fp)
    if fp > cap:
        reason = f"footprint {fp / 1e9:.1f} GB > cap {cap / 1e9:.0f} GB"
    elif time.time() - start > wall:
        reason = f"wall {time.time() - start:.0f}s > {wall:.0f}s"
    if reason:
        print(f"[guard] SIGTERM: {reason}", flush=True)
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("[guard] child did not exit after SIGTERM", flush=True)
        break
    time.sleep(0.25)
code = child.wait()
print(
    f"[guard] exit={code} peak_phys_footprint={peak / 1e9:.2f} GB wall={time.time() - start:.1f}s"
    + (f" ({reason})" if reason else ""),
    flush=True,
)
sys.exit(code if code is not None else 1)

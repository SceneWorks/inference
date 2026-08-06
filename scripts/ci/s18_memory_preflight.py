#!/usr/bin/env python3
"""Refuse an S18 sweep row this host cannot hold, BEFORE it takes the runner down.

sc-17324 crashed `nax-macos-2` twice — runs 30948568453 and 31051413212, both dying inside row F
seed 7 with "the self-hosted runner lost communication with the server". Neither was a flake, and
neither left any evidence behind: when the runner dies, the job's `always()` steps never run, so
every measured cell is lost too. The second crash cost ~14 minutes; the first cost 2h42m.

The cause is entirely predictable from the geometry, which is why this exists. The Krea Realtime 14B
q4 weights are ~9 GiB and never the problem. What scales is the **KV cache**: this is an
autoregressive VIDEO model, and the sweep's whole purpose is varying the KV read window, so the
biggest row holds a bounded window of 30 latent frames at 832x480 = 46,800 tokens of *unquantized*
bf16 KV.

    KV per token = 2 (K and V) x 40 layers x 5120 dim x 2 bytes = 800 KiB

Which reproduces every measured peak on that host to within ~10%:

    row  window          tokens   KV GiB   predicted   MEASURED (run 30787887176)
    A     6 frames        9,360      7.1        17.7        17.43
    D    15 frames       23,400     17.9        29.6        29.41
    F    30 frames       46,800     35.7        49.2        49.14   <-- killed the runner, twice
    E    45 (global)     70,200     53.6        68.8        63.32

Rows A and D were measured taking `active+wired` to 80.55 and 88.13 GiB of that box's ~101 GiB, with
NO swap and NO compression — comfortable. Row F extrapolates to ~100.6 GiB, i.e. the whole machine.
That is the cliff, and it is a straight line, so it can simply be computed in advance.

sc-17807 made the 800 KiB a knob rather than a constant (`KreaArConfig::kv_cache_quant`), so
`--kv-bits` / `KREA_S18_KV_BITS` prices the tier a dispatch is actually running. At Q8 the same
rows cost 0.53x the KV, which moves every prediction:

    row  window          tokens   KV GiB bf16   KV GiB q8   predicted q8
    A     6 frames        9,360      7.1           3.8          14.1
    D    15 frames       23,400     17.9           9.5          20.3
    F    30 frames       46,800     35.7          19.0          30.8
    E    45 (global)     70,200     53.6          28.5          41.2

Those are PREDICTIONS, not measurements — the q8 arm's measured peaks live in the S18 sweep's
recorded cells, and this preflight is only as good as its bf16 calibration until they land. What
they predict is concrete though: row F at 832x480, the row that took `nax-macos-2` down twice, is
~30.8 GiB against that box's ~36 GiB budget at Q8, i.e. affordable on the host it killed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# 2 (K and V) x 40 layers x 5120 dim x 2 bytes (bf16 — the shipped cache).
KV_BYTES_PER_TOKEN = 2 * 40 * 5120 * 2

# sc-17807 made the cache's storage tier a config knob (`KreaArConfig::kv_cache_quant`), so the
# preflight has to know which tier a dispatch is running or it prices the wrong sweep. A quantized
# row costs its packed payload plus one scale AND one bias per group, both in the quantized input's
# dtype (bf16 here) — which is why Q8 is 0.53x the bf16 cache and not a clean half. The Rust side
# computes the identical figure in `KreaRealtimeConfig::kv_bytes_per_token`; the two are pinned
# against each other in `scripts/tests/test_s18_memory_preflight.py`.
QUANT_GROUP_OVERHEAD_BYTES = 4
DIM = 5120
LAYERS = 40

# Krea Realtime 14B at q4, plus the resident activations the AR loop needs beyond its KV.
BASE_RESIDENT_GIB = 9.0

# Measured peaks run ~7-10% above (weights + KV) alone; round up so the guard errs toward refusing.
OVERHEAD = 1.10

# Latent downscale (VAE /8) then the transformer's 2x2 patchify.
LATENT_DIVISOR = 8
PATCH = 2

# Bounded read window per row, in LATENT FRAMES. Row E is the checkpoint's global window, so it is
# not a fixed number — it spans the whole clip. Mirrors `all_runs` in generate_smoke.rs.
ROW_WINDOW_FRAMES = {"A": 6, "B": 6, "C": 6, "D": 15, "F": 30, "E": None, "Z": 6}

# Row Z is deliberately short: the longest clip the shipped 6-frame window never evicts on.
ROW_LATENT_FRAMES = {"Z": 6}

# The sink anchor is FIRST-CHUNK KV pinned for the whole clip, so it is resident ON TOP OF the read
# window and has to be counted. Omitting it under-predicted row C (sink 3) by 17.6% against its
# measured peak — and under-prediction is the direction that crashes a runner.
ROW_SINK_FRAMES = {"A": 0, "B": 1, "C": 3, "D": 0, "F": 0, "E": 0, "Z": 0}

# Everything on the box that is NOT this process's MLX active allocation: the OS, the runner, other
# resident processes, and MLX's own buffer CACHE (`get_peak_memory` reports active only). Measured
# on nax-macos-2's sampler as `active+wired` minus the row's MLX active peak — 63.1 GiB during row A
# and 58.7 during row D — so 60 is the working figure.
#
# This is deliberately ABSOLUTE rather than a fraction of RAM, because a fraction does not survive
# contact with both hosts. At 40% of RAM the guard refuses row E on the 128 GiB dev Mac, where run
# 30787887176 actually ran row E to completion at 63.32 GiB. Subtracting a fixed overhead instead
# reproduces every observation the epic has:
#
#   nax-macos-2 (~101 GiB) -> 41 GiB for MLX:  D 29.4 fits (did),  F 49.1 OVER (killed it, twice)
#   nax-macos   ( 128 GiB) -> 68 GiB for MLX:  F 49.1 fits (did),  E 63.3 fits (did)
NON_MLX_OVERHEAD_GIB = 60.0

# Headroom on top, because the overhead above is a two-sample estimate and the cost of being wrong
# is the runner rather than the row.
SAFETY_MARGIN_GIB = 5.0


def host_ram_gib() -> float:
    """Physical RAM, or 0.0 when it cannot be determined (caller decides what that means)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return int(out.stdout.strip()) / 1024**3
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def kv_bytes_per_token(bits: int | None, group_size: int = 64) -> int:
    """Bytes of persistent KV per DiT token across the whole DiT, at the given cache tier.

    `bits=None` is the shipped bf16 cache. Anything else is group-wise affine quantization.
    """
    if bits is None:
        return KV_BYTES_PER_TOKEN
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"{bits} is not an MLX affine quantization width (2, 3, 4, 5, 6, 8)")
    if group_size not in (32, 64, 128) or DIM % group_size:
        raise ValueError(f"{group_size} is not a usable affine quantization group size")
    row = DIM * bits // 8 + DIM // group_size * QUANT_GROUP_OVERHEAD_BYTES
    return 2 * LAYERS * row


def tokens_per_latent_frame(width: int, height: int) -> int:
    return (height // LATENT_DIVISOR // PATCH) * (width // LATENT_DIVISOR // PATCH)


def window_tokens(row: str, width: int, height: int, latent_frames: int) -> int:
    per_frame = tokens_per_latent_frame(width, height)
    frames = ROW_WINDOW_FRAMES[row]
    clip = ROW_LATENT_FRAMES.get(row, latent_frames)
    # A global window (row E) never evicts, so its KV spans the entire clip. A bounded window is
    # capped by the clip too — a 30-frame window over a 6-frame clip only ever holds 6.
    span = clip if frames is None else min(frames, clip)
    # The sink is additive and permanent; it cannot exceed the clip it is taken from.
    sink = min(ROW_SINK_FRAMES.get(row, 0), clip)
    return (span + sink) * per_frame


def predicted_peak_gib(
    row: str,
    width: int,
    height: int,
    latent_frames: int,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
) -> float:
    per_token = kv_bytes_per_token(kv_bits, kv_group_size)
    kv_gib = window_tokens(row, width, height, latent_frames) * per_token / 1024**3
    return (BASE_RESIDENT_GIB + kv_gib) * OVERHEAD


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default=os.environ.get("KREA_S18_ROWS", "ABCDFEZ"))
    p.add_argument("--width", type=int, default=int(os.environ.get("KREA_SMOKE_W", "832")))
    p.add_argument("--height", type=int, default=int(os.environ.get("KREA_SMOKE_H", "480")))
    p.add_argument(
        "--latent-frames",
        type=int,
        default=int(os.environ.get("KREA_S18_LATENT_FRAMES", "45")),
    )
    p.add_argument("--ram-gib", type=float, default=None, help="override the detected host RAM")
    p.add_argument("--overhead-gib", type=float, default=NON_MLX_OVERHEAD_GIB)
    p.add_argument("--margin-gib", type=float, default=SAFETY_MARGIN_GIB)
    # Read as STRINGS and convert below, inside the try. `default=int(os.environ[...])` evaluates at
    # parser-construction time, so a malformed value there escapes as an uncaught traceback rather
    # than the ::error:: line the caller can act on. The workflow's regex makes that unreachable via
    # dispatch, but this script is also run by hand.
    p.add_argument(
        "--kv-bits",
        default=os.environ.get("KREA_S18_KV_BITS") or None,
        help="KV cache quantization width (sc-17807); omit for the shipped bf16 cache",
    )
    p.add_argument(
        "--kv-group-size",
        default=os.environ.get("KREA_S18_KV_GROUP_SIZE") or "64",
    )
    args = p.parse_args()

    try:
        kv_bits = None if args.kv_bits is None else int(args.kv_bits)
        per_token = kv_bytes_per_token(kv_bits, int(args.kv_group_size))
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    args.kv_bits, args.kv_group_size = kv_bits, int(args.kv_group_size)

    unknown = [r for r in args.rows if r not in ROW_WINDOW_FRAMES]
    if unknown:
        print(f"::error::unknown S18 row(s) {unknown} in '{args.rows}'", file=sys.stderr)
        return 1

    ram = args.ram_gib if args.ram_gib is not None else host_ram_gib()
    if ram <= 0:
        # Never block the lane on a failure to introspect the host: an unmeasurable box is a
        # reason to report loudly, not a reason to refuse work that may be perfectly fine.
        print("::warning::could not determine host RAM; skipping the S18 memory preflight")
        return 0
    budget = ram - args.overhead_gib - args.margin_gib

    kv_tier = "bf16" if args.kv_bits is None else f"q{args.kv_bits}/g{args.kv_group_size}"
    print(
        f"S18 memory preflight — {args.width}x{args.height}, {args.latent_frames} latent frames, "
        f"{tokens_per_latent_frame(args.width, args.height)} tokens/frame, "
        f"KV {kv_tier} ({per_token} bytes/token)"
    )
    print(
        f"  host RAM {ram:.1f} GiB - {args.overhead_gib:.0f} non-MLX - {args.margin_gib:.0f} margin"
        f" => {budget:.1f} GiB for one row's MLX active peak"
    )
    over = []
    for row in args.rows:
        tok = window_tokens(row, args.width, args.height, args.latent_frames)
        peak = predicted_peak_gib(
            row,
            args.width,
            args.height,
            args.latent_frames,
            args.kv_bits,
            args.kv_group_size,
        )
        verdict = "OK" if peak <= budget else "OVER BUDGET"
        if peak > budget:
            over.append((row, tok, peak))
        print(f"  row {row}: {tok:>7} tokens -> ~{peak:5.1f} GiB predicted MLX peak   {verdict}")

    if over:
        detail = "; ".join(f"{r} ~{p:.1f} GiB ({t} tokens)" for r, t, p in over)
        print(
            f"::error::S18 rows over this host's {budget:.1f} GiB budget: {detail}. These rows "
            f"will take the runner down rather than fail — run them at a smaller geometry (e.g. "
            f"KREA_SMOKE_W=640 KREA_SMOKE_H=384) or on a larger host. See sc-17324.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

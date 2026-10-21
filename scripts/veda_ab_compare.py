#!/usr/bin/env python3
"""Compare two Veda A/B render dirs (sc-25777): video PSNR, audio SNR, side-by-side mp4.

Each dir is a `VEDA_AB_OUT` written by the MiniMax-H3 `veda_ab` harness (MLX or a candle twin):
`result.json` (output.width/height/fps), `frames.rgb` (raw rgb24) and `audio.f32le` (interleaved
f32). PSNR is computed on the RAW frames, not the lossy mp4s. dirA is the reference (dense) arm.
Stdlib + ffmpeg only.
"""
import argparse
import array
import json
import math
import re
import subprocess
import sys
from pathlib import Path


def geometry(d: Path):
    # `output` is the decoded geometry; the shared contract only guarantees `config`.
    result = json.loads((d / "result.json").read_text(encoding="utf-8"))
    out = result.get("output") or result["config"]
    return out["width"], out["height"], out.get("fps", 24)


def video_psnr(a: Path, b: Path, w: int, h: int, fps: int):
    raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i"]
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error"]
    cmd += raw + [str(a / "frames.rgb")] + raw + [str(b / "frames.rgb")]
    cmd += ["-lavfi", "[0:v][1:v]psnr=stats_file=-", "-f", "null", "-"]
    text = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    values = [float(v) for v in re.findall(r"psnr_avg:(\S+)", text) if v != "inf"]
    frames = len(re.findall(r"psnr_avg:", text))
    if not values:
        return frames, math.inf, math.inf
    return frames, sum(values) / len(values), min(values)


def audio_compare(a: Path, b: Path):
    x, y = array.array("f"), array.array("f")
    x.frombytes((a / "audio.f32le").read_bytes())
    y.frombytes((b / "audio.f32le").read_bytes())
    n = min(len(x), len(y))
    sig = sum(v * v for v in x[:n]) / max(n, 1)
    err = sum((p - q) ** 2 for p, q in zip(x[:n], y[:n])) / max(n, 1)
    snr = math.inf if err == 0 else 10 * math.log10(sig / err) if sig > 0 else -math.inf
    return {
        "samples": (len(x), len(y)),
        "rms_a": math.sqrt(sig),
        "rms_b": math.sqrt(sum(v * v for v in y[:n]) / max(n, 1)),
        "rms_diff": math.sqrt(err),
        "snr_db": snr,
    }


def side_by_side(a: Path, b: Path, out: Path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(a / "video.mp4"), "-i", str(b / "video.mp4")]
    cmd += ["-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "0:a?"]
    cmd += ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)]
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir_a", type=Path, help="reference arm (e.g. Veda off)")
    p.add_argument("dir_b", type=Path, help="candidate arm (e.g. Veda on)")
    p.add_argument("--out", type=Path, help="write an hstack side-by-side mp4 (A left, A's audio)")
    args = p.parse_args()

    ga, gb = geometry(args.dir_a), geometry(args.dir_b)
    if ga != gb:
        print(f"geometry differs: {ga} vs {gb}", file=sys.stderr)
        return 2
    frames, mean, low = video_psnr(args.dir_a, args.dir_b, *ga)
    report = {"video": {"frames": frames, "psnr_mean_db": mean, "psnr_min_db": low}}
    report["audio"] = audio_compare(args.dir_a, args.dir_b)
    for d, key in ((args.dir_a, "a"), (args.dir_b, "b")):
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        report[f"timings_{key}"] = r.get("timings")
        report[f"veda_{key}"] = (r.get("veda") or {}).get("total_attention_fraction")
    if args.out:
        side_by_side(args.dir_a, args.dir_b, args.out)
        report["side_by_side"] = str(args.out)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

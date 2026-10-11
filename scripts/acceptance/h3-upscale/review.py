"""Produce review sheets and descriptive delivery metrics, never an automatic GO.

All metrics compare decoded delivered clips. Bicubic is a preservation baseline,
not high-resolution ground truth; departures and sharpness cannot establish useful
detail. A person must apply the identity, motion and artifact rubric separately.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def decode(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-vf", "scale=1024:576:flags=bicubic", "pipe:1",
    ])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 576, 1024, 3)
    if len(frames) != 39:
        raise ValueError(f"{path.name}: expected exactly 39 frames")
    return frames


def audio_hash(path):
    audio = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=index", "-of", "csv=p=0", str(path),
    ])
    if not audio.strip():
        return hashlib.sha256(b"").hexdigest()
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a?",
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ])
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("h3", "other-model", "live-action"), default=("h3", "other-model", "live-action"))
    args = parser.parse_args()
    names = ["source", "bicubic", "guided", "unguided", "latent-only", "seedvr2"]
    all_metrics = {
        "meaning": "descriptive deviations from bicubic; no high-resolution truth, no automatic quality score",
        "units": "mean absolute RGB channel difference on 0..255 decoded pixels",
        "frames": 39,
        "selected_review_frames": [0, 19, 38],
        "cases": {},
    }
    for kind in args.cases:
        folder = args.evidence / kind
        videos = {name: decode(folder / f"{name}.mp4") for name in names}
        baseline = videos["bicubic"].astype(np.float32)
        baseline_change = np.diff(baseline, axis=0)
        source_audio = audio_hash(folder / "source.mp4")
        metrics = {}
        for name in names[1:]:
            candidate = videos[name].astype(np.float32)
            frame_mae = np.abs(candidate - baseline).mean(axis=(1, 2, 3))
            changes = np.diff(candidate, axis=0)
            temporal = np.abs(changes - baseline_change).mean(axis=(1, 2, 3))
            metrics[name] = {
                "frame_mae_mean": float(frame_mae.mean()),
                "frame_mae_begin_middle_tail": [float(frame_mae[i]) for i in [0, 19, 38]],
                "adjacent_frame_change_mean": float(np.abs(changes).mean()),
                "temporal_change_departure_mean": float(temporal.mean()),
                "temporal_change_departure_max": float(temporal.max()),
                "decoded_source_audio_equal": audio_hash(folder / f"{name}.mp4") == source_audio,
            }
            if not metrics[name]["decoded_source_audio_equal"]:
                raise ValueError(f"{kind}/{name}: decoded source soundtrack changed")
        all_metrics["cases"][kind] = metrics
        sheet = Image.new("RGB", (384 * len(names), 248 * 3), "#202020")
        draw = ImageDraw.Draw(sheet)
        for row, frame in enumerate([0, 19, 38]):
            for col, name in enumerate(names):
                tile = Image.fromarray(videos[name][frame]).resize((384, 216), Image.Resampling.LANCZOS)
                x, y = col * 384, row * 248
                sheet.paste(tile, (x, y + 32))
                draw.text((x + 8, y + 8), f"{kind} / {name} / frame {frame}", fill="white")
        sheet.save(folder / "review-begin-middle-tail.png")
        for frame in [0, 19, 38]:
            pair = Image.new("RGB", (1024 * 3, 608), "#202020")
            draw = ImageDraw.Draw(pair)
            for col, name in enumerate(["bicubic", "guided", "seedvr2"]):
                pair.paste(Image.fromarray(videos[name][frame]), (col * 1024, 32))
                draw.text((col * 1024 + 8, 8), f"{kind} / {name} / frame {frame}", fill="white")
            pair.save(folder / f"review-full-frame{frame}.png")
    (args.evidence / "delivery-metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()

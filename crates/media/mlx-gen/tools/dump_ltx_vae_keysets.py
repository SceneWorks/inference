#!/usr/bin/env python3
"""sc-18765 — dump the LTX VAE / audio-VAE / vocoder **key sets** to a committed fixture.

The LTX-2.5 conv video VAE and audio VAE are claimed to be drop-in compatible with the
shipped `mlx-gen-ltx` / `candle-gen-ltx` ports (epic 18755 §1.2). "Drop-in" is a claim
about *names, shapes and layout*, so this dump records exactly those for both LTX-2.5
component checkpoints and the LTX-2.3 split-MLX components the ports consume today. The
conformance tests diff the two key-by-key, which is what stops a future upstream VAE
change from silently landing on an incompatible port.

Reads **safetensors headers only** — no tensor payload bytes are fetched or loaded, so a
1.45 GB checkpoint costs a few tens of KB of I/O.

Usage:

    python3 crates/media/mlx-gen/tools/dump_ltx_vae_keysets.py \
        --ltx25 /path/to/Lightricks--LTX-2.5/snapshots/<rev> \
        --ltx23-mlx /path/to/SceneWorks--ltx-2.3-mlx/snapshots/<rev>/bf16 \
        --out docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

# (fixture entry name, path relative to the LTX-2.5 snapshot root, retained tensor prefixes).
# A `None` prefix filter keeps every tensor. The DiffVAE file is recorded ENCODER-ONLY: this
# story validates that its conv `Encoder` is the same module as the conv VAE's, while its
# `NADiffusionDecoder` belongs to sc-18766 / sc-18767 and would otherwise add ~120 KB of key
# names that nothing here asserts against. The dropped count is still recorded, so the
# omission is visible in the fixture rather than silent.
LTX25_FILES = [
    ("ltx_2_5_video_vae_conv", "vae/ltx-2.5-video-vae-conv-bf16.safetensors", None),
    (
        "ltx_2_5_video_vae_diffusion",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        ("encoder.", "per_channel_statistics."),
    ),
    ("ltx_2_5_audio_vae", "vae/ltx-2.5-audio-vae-bf16.safetensors", None),
]

# (fixture entry name, filename in a split-MLX tier dir)
LTX23_MLX_FILES = [
    ("ltx_2_3_mlx_vae_decoder", "vae_decoder.safetensors"),
    ("ltx_2_3_mlx_vae_encoder", "vae_encoder.safetensors"),
    ("ltx_2_3_mlx_audio_vae", "audio_vae.safetensors"),
    ("ltx_2_3_mlx_vocoder", "vocoder.safetensors"),
]


def read_header(path: Path) -> dict:
    """Return the safetensors JSON header (tensors + `__metadata__`), payload untouched."""
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len))


def describe(path: Path, keep_prefixes: tuple[str, ...] | None = None) -> dict:
    header = read_header(path)
    metadata = header.pop("__metadata__", None) or {}
    kept = {
        name: e["shape"]
        for name, e in sorted(header.items())
        if keep_prefixes is None or name.startswith(keep_prefixes)
    }
    config = metadata.get("config")
    if isinstance(config, str):
        config = json.loads(config)
    entry = {
        "file_name": path.name,
        "total_size_bytes": path.stat().st_size,
        "tensor_count": len(header),
        "dtypes": sorted({e["dtype"] for e in header.values()}),
        "tensors": kept,
    }
    if len(kept) != len(header):
        entry["recorded_tensor_prefixes"] = list(keep_prefixes or ())
        entry["recorded_tensor_count"] = len(kept)
        entry["omitted_tensor_count"] = len(header) - len(kept)
    if config is not None:
        entry["config"] = config
    if "model_version" in metadata:
        entry["model_version"] = metadata["model_version"]
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ltx25", type=Path, required=True, help="Lightricks/LTX-2.5 snapshot root")
    ap.add_argument(
        "--ltx23-mlx",
        type=Path,
        required=True,
        help="SceneWorks/ltx-2.3-mlx split tier dir (bf16/q8/q4 — key sets are tier-invariant)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    components: dict[str, dict] = {}
    for name, rel, keep in LTX25_FILES:
        components[name] = describe(args.ltx25 / rel, keep)
    for name, rel in LTX23_MLX_FILES:
        components[name] = describe(args.ltx23_mlx / rel)

    doc = {
        "_provenance": {
            "story": "sc-18765",
            "epic": "sc-18755",
            "generator": "crates/media/mlx-gen/tools/dump_ltx_vae_keysets.py",
            "method": (
                "safetensors JSON header only (little-endian u64 length prefix + header bytes); "
                "zero tensor payload bytes read"
            ),
            "ltx_2_5_source": "Lightricks/LTX-2.5",
            "ltx_2_3_source": "SceneWorks/ltx-2.3-mlx (split-MLX rehost, bf16 tier)",
            "note": (
                "Shapes are as-stored: LTX-2.5 upstream ships PyTorch layout "
                "(Conv3d [O,I,kt,kh,kw] / Conv2d [O,I,kH,kW] / Conv1d [O,I,k] / "
                "ConvTranspose1d [I,O,k]); the ltx-2.3-mlx rehost ships the channels-last "
                "MLX layout the mlx-gen-ltx ports consume."
            ),
        },
        "components": components,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    for name, entry in components.items():
        print(f"  {name:32} {entry['tensor_count']:5} tensors  {entry['file_name']}")


if __name__ == "__main__":
    main()

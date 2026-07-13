"""sc-5984 — verify the converted Ideogram 4 MLX snapshot against the source fp8 checkpoint.

Beyond the converter's key-count round-trip, this independently re-dequantizes a sample of
fp8 weights with torch and asserts the MLX-written bf16 matches exactly, and that every
converted component actually loads via `mlx.core` (the E1 "loadable by the loaders" gate).

Run:
  ~/mlx-flux-venv/bin/python tools/verify_ideogram4_convert.py \
      --converted ~/.cache/ideogram4-mlx-convert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open

WEIGHT_COMPONENTS = ("transformer", "unconditional_transformer", "text_encoder", "vae")
# A few representative keys per transformer to numerically check (fp8 weight + bf16 passthrough).
SAMPLE_KEYS = (
    "input_proj.weight",                  # fp8, top-level
    "llm_cond_proj.weight",               # fp8, [4608, 53248]
    "layers.0.attention.qkv.weight",      # fp8, fused qkv
    "layers.33.feed_forward.w2.weight",   # fp8, last-layer FFN down
    "layers.0.attention_norm1.weight",    # bf16 passthrough (RMSNorm)
)


def default_fp8_snapshot() -> Path:
    base = Path.home() / ".cache/huggingface/hub/models--ideogram-ai--ideogram-4-fp8/snapshots"
    snaps = sorted(p for p in base.glob("*") if p.is_dir())
    if not snaps:
        sys.exit(f"fp8 snapshot not found under {base}")
    return snaps[-1]


def torch_dequant(f, key: str) -> torch.Tensor:
    t = f.get_tensor(key)
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        scale = f.get_tensor(key + "_scale").to(torch.float32)
        w = t.to(torch.float32) * scale.reshape(-1, *([1] * (t.ndim - 1)))
        return w.to(torch.bfloat16)
    return t.to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted", type=Path, default=Path.home() / ".cache/ideogram4-mlx-convert")
    ap.add_argument("--snapshot", type=Path, default=None)
    args = ap.parse_args()
    snap = args.snapshot or default_fp8_snapshot()

    failures = 0
    for comp in WEIGHT_COMPONENTS:
        conv = args.converted / comp / "model.safetensors"
        if not conv.exists():
            print(f"[{comp}] MISSING {conv}")
            failures += 1
            continue
        loaded = mx.load(str(conv))  # the load gate
        print(f"[{comp}] loaded {len(loaded)} tensors via mlx.core")
        if comp in ("transformer", "unconditional_transformer"):
            src = snap / comp / "diffusion_pytorch_model.safetensors"
            with safe_open(src, framework="pt") as f:
                for key in SAMPLE_KEYS:
                    ref = torch_dequant(f, key)
                    got = torch.from_numpy(
                        __import__("numpy").asarray(loaded[key].astype(mx.float32))
                    ).to(torch.bfloat16)
                    if tuple(got.shape) != tuple(ref.shape):
                        print(f"  ✗ {key} shape {tuple(got.shape)} != {tuple(ref.shape)}")
                        failures += 1
                        continue
                    exact = torch.equal(got, ref)
                    maxabs = (got.float() - ref.float()).abs().max().item()
                    print(f"  {'✓' if exact else '✗'} {key} {tuple(ref.shape)} "
                          f"exact={exact} max|Δ|={maxabs:g}")
                    if not exact:
                        failures += 1

    print("\n" + ("OK — all checks passed" if failures == 0 else f"FAILED ({failures})"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

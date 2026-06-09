"""sc-3713 — dump the SAM2 memory-layer parity golden (memory encoder + memory attention).

Runs the MLX-native reference (`avbiswas/sam2-mlx`, the impl `mlx-gen-sam2` ports) for the two
Phase-B forward passes on fixed random fixtures, and bundles, into one gitignored golden:

  * the memory weights the Rust port reads (`memory_encoder.*` / `memory_attention.*`),
  * **memory encoder** fixture: `mem_pix_feat` [1,256,64,64], `mem_masks` [1,1,1024,1024], and the
    reference outputs `mem_vis_features` [1,64,64,64] / `mem_vis_pos` [1,64,64,64],
  * **memory attention** fixture (a 3-frame bank + 2 object pointers): `ma_curr` / `ma_curr_pos`
    [4096,1,256], `ma_mem` / `ma_mem_pos` [3*4096+8,1,64], `ma_num_obj` (=8), and the reference
    output `ma_out` [4096,1,256].

Both run MLX Metal, so parity is near-bit. The Rust `tests/memory_parity.rs` (`#[ignore]`, macOS)
builds `MemoryEncoder` / `MemoryAttention`, runs the same fixtures, and asserts agreement.

Run (MLX venv + the reference checkout, both already present):
  PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python \
      tools/dump_sam2_memory_golden.py --size large
"""

from __future__ import annotations

import argparse
import glob
import os

import mlx.core as mx
import numpy as np

from mlx_sam.config import (
    SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER,
    SAM2_1_HIERA_LARGE_IMAGE_ENCODER,
    SAM2_1_HIERA_SMALL_IMAGE_ENCODER,
    SAM2_1_HIERA_TINY_IMAGE_ENCODER,
)
from mlx_sam.models.segmenter import Sam2ImageSegmenter

SIZE_CFG = {
    "tiny": SAM2_1_HIERA_TINY_IMAGE_ENCODER,
    "small": SAM2_1_HIERA_SMALL_IMAGE_ENCODER,
    "base_plus": SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER,
    "large": SAM2_1_HIERA_LARGE_IMAGE_ENCODER,
}
HF_REPO = {
    "tiny": "avbiswas/sam2.1-hiera-tiny-mlx",
    "small": "avbiswas/sam2.1-hiera-small-mlx",
    "base_plus": "avbiswas/sam2.1-hiera-base-plus-mlx",
    "large": "avbiswas/sam2.1-hiera-large-mlx",
}
KEEP_PREFIXES = ("memory_encoder.", "memory_attention.")

# Memory-attention fixture sizing: a 64×64 image grid (4096 tokens), a 3-frame memory bank, and
# 2 object pointers (4 tokens each → 8 excluded-from-RoPE tokens). 3*4096 + 8 = 12296 memory tokens.
GRID = 64
N_TOK = GRID * GRID
N_FRAMES = 3
N_OBJ = 8


def resolve_checkpoint(size: str) -> str:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(HF_REPO[size], allow_patterns=["*.safetensors"])
    files = glob.glob(os.path.join(snap, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {snap}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZE_CFG), default="large")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "golden"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SIZE_CFG[args.size]
    ckpt = resolve_checkpoint(args.size)
    print(f"[load] {ckpt}")
    weights = mx.load(ckpt)

    model = Sam2ImageSegmenter(config=cfg)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    rng = np.random.RandomState(args.seed)
    f32 = lambda *shape: mx.array(rng.standard_normal(shape).astype(np.float32))

    # --- Memory encoder: pix_feat + a high-res mask → 64-ch memory feature map + position enc. ---
    mem_pix_feat = f32(1, 256, GRID, GRID)
    mem_masks = f32(1, 1, 16 * GRID, 16 * GRID)  # mask_downsampler shrinks by 16×
    enc_out = model.memory_encoder(mem_pix_feat, mem_masks, skip_mask_sigmoid=True)
    mem_vis_features = enc_out["vision_features"]
    mem_vis_pos = enc_out["vision_pos_enc"][0]
    print(f"[memory_encoder] features={mem_vis_features.shape} pos={mem_vis_pos.shape}")

    # --- Memory attention: current tokens conditioned on a 3-frame bank + 2 object pointers. ---
    n_mem = N_FRAMES * N_TOK + N_OBJ
    ma_curr = f32(N_TOK, 1, 256)
    ma_curr_pos = f32(N_TOK, 1, 256)
    ma_mem = f32(n_mem, 1, 64)
    ma_mem_pos = f32(n_mem, 1, 64)
    ma_out = model.memory_attention(ma_curr, ma_curr_pos, ma_mem, ma_mem_pos, num_obj_ptr_tokens=N_OBJ)
    print(f"[memory_attention] mem={ma_mem.shape} num_obj={N_OBJ} out={ma_out.shape}")

    golden = {k: v for k, v in weights.items() if k.startswith(KEEP_PREFIXES)}
    golden.update(
        mem_pix_feat=mem_pix_feat,
        mem_masks=mem_masks,
        mem_vis_features=mem_vis_features,
        mem_vis_pos=mem_vis_pos,
        ma_curr=ma_curr,
        ma_curr_pos=ma_curr_pos,
        ma_mem=ma_mem,
        ma_mem_pos=ma_mem_pos,
        ma_out=ma_out,
        ma_num_obj=mx.array(np.asarray([N_OBJ], dtype=np.int32)),
    )
    golden = {k: (mx.array(v).astype(mx.float32) if v.dtype != mx.int32 else v) for k, v in golden.items()}
    mx.eval(list(golden.values()))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sam2_memory_golden_{args.size}.safetensors")
    mx.save_safetensors(out_path, golden, metadata={"format": "mlx", "size": args.size})
    print(f"[written] {out_path} ({len(golden)} tensors)")


if __name__ == "__main__":
    main()

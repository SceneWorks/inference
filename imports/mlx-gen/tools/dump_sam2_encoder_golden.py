"""sc-3705 — dump the SAM2 image-encoder (Hiera trunk + FPN neck) parity golden.

Runs the MLX-native reference encoder (`avbiswas/sam2-mlx`, the impl this crate ports) on a fixed
deterministic input and bundles, into one gitignored golden safetensors:

  * the encoder weights (`trunk.*` / `neck.*`, copied from the official converted checkpoint),
  * `enc_in` — the NCHW [1,3,1024,1024] input,
  * `ref_backbone_fpn_{0,1,2}`, `ref_vision_features`, `ref_pos_{0,1,2}` — the reference outputs.

The Rust `tests/encoder_parity.rs` (`#[ignore]`, macOS/Metal) loads this file, runs
`Sam2ImageEncoder`, and asserts cosine ≈ 1 / small mean-rel vs the reference outputs.

Reference repo (clone, no install needed — we import the model module directly to avoid the cv2
video dep):  git clone https://github.com/avbiswas/sam2-mlx /tmp/sam2-mlx

Run (MLX venv + the official converted weights, both already present per sc-3705):
  PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python \
      tools/dump_sam2_encoder_golden.py --size large
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
from mlx_sam.models.image_encoder import Sam2ImageEncoder

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


def resolve_checkpoint(size: str) -> str:
    """Locate the converted MLX `*_image_segmenter.safetensors` for `size` (download if absent)."""
    from huggingface_hub import snapshot_download

    snap = snapshot_download(HF_REPO[size], allow_patterns=["*.safetensors"])
    files = glob.glob(os.path.join(snap, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {snap}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZE_CFG), default="large")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "golden"),
        help="golden output dir (default: tools/golden)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SIZE_CFG[args.size]
    ckpt = resolve_checkpoint(args.size)
    print(f"[load] {ckpt}")
    weights = mx.load(ckpt)

    model = Sam2ImageEncoder(config=cfg)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Deterministic NCHW [1,3,1024,1024] input (standard-normal — range ≈ ImageNet-normalized px).
    rng = np.random.RandomState(args.seed)
    enc_in_np = rng.standard_normal((1, 3, 1024, 1024)).astype(np.float32)
    enc_in = mx.array(enc_in_np)

    out = model(enc_in)
    backbone_fpn = out["backbone_fpn"]
    vision_features = out["vision_features"]
    vision_pos_enc = out["vision_pos_enc"]
    print(f"[forward] backbone_fpn={[t.shape for t in backbone_fpn]}")
    print(f"[forward] vision_features={vision_features.shape}")
    print(f"[forward] vision_pos_enc={[t.shape for t in vision_pos_enc]}")

    # Bundle: encoder weights (trunk./neck.) + input + reference outputs.
    golden = {k: v for k, v in weights.items() if k.startswith(("trunk.", "neck."))}
    golden["enc_in"] = enc_in
    golden["ref_vision_features"] = vision_features
    for i, t in enumerate(backbone_fpn):
        golden[f"ref_backbone_fpn_{i}"] = t
    for i, t in enumerate(vision_pos_enc):
        golden[f"ref_pos_{i}"] = t
    golden = {k: mx.array(v).astype(mx.float32) for k, v in golden.items()}
    mx.eval(list(golden.values()))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sam2_encoder_golden_{args.size}.safetensors")
    mx.save_safetensors(out_path, golden, metadata={"format": "mlx", "size": args.size})
    print(f"[written] {out_path} ({len(golden)} tensors)")


if __name__ == "__main__":
    main()

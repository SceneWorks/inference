"""sc-3714 / sc-4750 — dump the SAM2 video propagation parity golden (init_state / propagate).

Runs the MLX-native reference video predictor (`avbiswas/sam2-mlx`, the impl `mlx-gen-sam2` ports)
on a fixed synthetic clip with a box prompt, and bundles, into one gitignored golden:

  * every weight the Rust port reads (full reference state dict — trunk./neck./sam_*./memory_*./
    obj_ptr_* + the no_* / maskmem_tpos_enc globals),
  * the preprocessed clip `images` [T,3,1024,1024] (so the Rust port consumes byte-identical pixels
    and the comparison isolates the model from any frame-decode/preprocess divergence),
  * the prompt box `box_xyxy` [4] (original pixel space), the `prompt_frame_idx` [1] it sits on,
    `video_hw` [2], `num_frames`,
  * the reference per-frame **low-res** mask logits `low_res_masks` [T,1,256,256] and the selected
    object-score logits `object_scores` [T,1] — low-res so the parity isolates the model from the
    cv2 video-resolution resize.

Two directions (sc-4750):
  * **forward** (default) — box on frame 0, `propagate_in_video()` runs 0 → T-1.
  * **`--reverse`** — box on the *last* frame, `propagate_in_video(reverse=True)` runs T-1 → 0,
    exercising the reverse memory arithmetic. Written to a `_reverse` golden so the forward golden
    is untouched.

Both run MLX Metal, so parity is near-bit. The Rust `tests/video_parity.rs` (`#[ignore]`, macOS)
rebuilds the predictor, runs init_state + add box + propagate{,_reverse} on the same clip, and
asserts the per-frame masks agree (cosine + IoU).

Run (MLX venv + the reference checkout, both already present):
  PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python \
      tools/dump_sam2_video_golden.py --size large [--reverse]
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
from mlx_sam.video_predictor import SAM2VideoPredictor

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

# Synthetic clip: a bright rectangle on a dark background, translating across the frame so the
# track genuinely has to follow it (a static mask would hide propagation bugs).
H, W = 360, 480
N_FRAMES = 4
RECT_W, RECT_H = 120, 140
STEP = 35  # px / frame
Y0 = 110


def resolve_checkpoint(size: str) -> str:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(HF_REPO[size], allow_patterns=["*.safetensors"])
    files = glob.glob(os.path.join(snap, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {snap}")
    return files[0]


def box_for_frame(t: int) -> tuple[float, float, float, float]:
    """The tracked rectangle's box (x1,y1,x2,y2) at frame `t`, in original pixel space."""
    x0 = 40 + t * STEP
    return (float(x0), float(Y0), float(x0 + RECT_W), float(Y0 + RECT_H))


def synth_frames() -> list[np.ndarray]:
    """N translating-rectangle RGB uint8 frames."""
    frames: list[np.ndarray] = []
    for t in range(N_FRAMES):
        img = np.full((H, W, 3), 30, dtype=np.uint8)
        x0 = 40 + t * STEP
        img[Y0 : Y0 + RECT_H, x0 : x0 + RECT_W] = (230, 200, 60)
        # a little texture so the encoder has gradients to latch onto
        img[Y0 + 20 : Y0 + 60, x0 + 20 : x0 + 70] = (90, 140, 220)
        frames.append(img)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZE_CFG), default="large")
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="prompt the last frame and track backward (propagate_in_video(reverse=True)).",
    )
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "golden"))
    args = ap.parse_args()

    cfg = SIZE_CFG[args.size]
    ckpt = resolve_checkpoint(args.size)
    print(f"[load] {ckpt}")
    weights = mx.load(ckpt)

    model = Sam2ImageSegmenter(config=cfg)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    predictor = SAM2VideoPredictor(model=model)

    # Build the clip and write it to a temp dir of PNGs so we exercise the real init_state path.
    import tempfile

    from PIL import Image

    frames = synth_frames()
    tmp = tempfile.mkdtemp(prefix="sam2_video_golden_")
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))

    state = predictor.init_state(tmp)
    print(f"[init_state] images={state['images'].shape} hw=({state['video_height']},{state['video_width']})")

    # Forward: prompt frame 0 and track to the end. Reverse: prompt the last frame and track to 0.
    prompt_frame_idx = (state["num_frames"] - 1) if args.reverse else 0
    box = box_for_frame(prompt_frame_idx)
    predictor.add_new_points_or_box(state, frame_idx=prompt_frame_idx, obj_id=1, box=box)

    frame_order: list[int] = []
    for frame_idx, _obj_ids, _masks in predictor.propagate_in_video(state, reverse=args.reverse):
        frame_order.append(int(frame_idx))
    print(f"[propagate{'_reverse' if args.reverse else ''}] prompt_frame={prompt_frame_idx} frames={frame_order}")

    # Pull each frame's stored low-res mask + object score from the per-object output dict.
    out_dict = state["output_dict_per_obj"][0]

    def frame_out(idx: int) -> dict:
        return out_dict["cond_frame_outputs"].get(idx) or out_dict["non_cond_frame_outputs"][idx]

    low = []
    scores = []
    for idx in range(state["num_frames"]):
        o = frame_out(idx)
        low.append(np.array(o["pred_masks"]).astype(np.float32).reshape(1, 256, 256))
        scores.append(np.array(o["object_score_logits"]).astype(np.float32).reshape(1))
    low_res_masks = np.stack(low, axis=0)  # [T,1,256,256]
    object_scores = np.stack(scores, axis=0)  # [T,1]
    print(f"[golden] low_res_masks={low_res_masks.shape} object_scores={object_scores.reshape(-1)}")

    golden = {k: (mx.array(v).astype(mx.float32) if v.dtype != mx.int32 else v) for k, v in weights.items()}
    golden.update(
        images=mx.array(np.array(state["images"]).astype(np.float32)),
        box_xyxy=mx.array(np.asarray(box, dtype=np.float32)),
        prompt_frame_idx=mx.array(np.asarray([prompt_frame_idx], dtype=np.int32)),
        video_hw=mx.array(np.asarray([state["video_height"], state["video_width"]], dtype=np.int32)),
        num_frames=mx.array(np.asarray([state["num_frames"]], dtype=np.int32)),
        low_res_masks=mx.array(low_res_masks),
        object_scores=mx.array(object_scores),
    )
    mx.eval(list(golden.values()))

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_reverse" if args.reverse else ""
    out_path = os.path.join(args.out_dir, f"sam2_video_golden{suffix}_{args.size}.safetensors")
    mx.save_safetensors(out_path, golden, metadata={"format": "mlx", "size": args.size})
    print(f"[done] wrote {out_path} ({len(golden)} tensors)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""SAM3 **no-prompt tracking-frame decode** parity oracle — epic 4910, sc-4924 (Phase F2.5a).

The video-propagation step (`run_tracker_propagation` → `Sam3TrackerVideoModel.forward(run_mem_encoder=
False)` → `_run_single_frame_inference`) decodes each existing object on a frame with **no point/mask
inputs**: the memory-conditioned features (F2.4) go through `_single_frame_forward` with a padded empty
point (label −1 → `not_a_point`) and a single-mask (dynamic-stability) decode, producing the low-res
mask, the high-res mask (for memory encoding), the object pointer, and the object-score logit.

To validate the decode in isolation we wrap `tracker_model._single_frame_forward` on a real 2-frame
`Sam3VideoModel` PCS run and capture the **first call with `input_masks is None`** (frame-1
propagation — frame-0 conditioning goes through `_use_mask_as_output`, which passes a mask):
  - image_embeddings = [feat_s0 (1,32,288,288), feat_s1 (1,64,144,144), pix_feat (1,256,72,72)]
  - output: pred_masks (1,1,288,288), high_res_masks (1,1,1008,1008), object_pointer (1,1,256),
    object_score_logits (1,1,1)

The Rust test feeds the captured pix_feat + high-res features into `Sam3Tracker::decode_tracked_frame`
and compares all four outputs (cosine gate >0.9999; object_score |Δ|).

Run:  /tmp/sam3ref/.venv/bin/python dump_trackframe_fixture.py
"""

import hashlib
import json
import os
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Sam3VideoModel, Sam3VideoProcessor

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/sam3"
torch.manual_seed(0)

URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5).astype(np.float32)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    video = [np.array(image), np.array(image)]

    session = processor.init_video_session(video=video, inference_device="cpu", dtype=torch.float32)
    processor.add_text_prompt(session, "person")

    tracker = model.tracker_model
    cap = {}
    orig = tracker._single_frame_forward

    def wrapped(*a, **k):
        out = orig(*a, **k)
        input_masks = k.get("input_masks", None)
        input_points = k.get("input_points", None)
        if "out" not in cap and input_masks is None and input_points is None:
            cap["image_embeddings"] = k["image_embeddings"]
            cap["out"] = out
        return out

    tracker._single_frame_forward = wrapped
    try:
        with torch.no_grad():
            for _out in model.propagate_in_video_iterator(session):
                pass
    finally:
        tracker._single_frame_forward = orig

    assert "out" in cap, "no no-prompt _single_frame_forward call captured"
    det = lambda t: t.detach().cpu().float().contiguous().clone()
    emb = cap["image_embeddings"]
    out = cap["out"]
    feat_s0, feat_s1, pix_feat = emb[0], emb[1], emb[2]
    print(
        f"  pix_feat {list(pix_feat.shape)} feat_s0 {list(feat_s0.shape)} feat_s1 {list(feat_s1.shape)}"
    )
    print(
        f"  pred_masks {list(out.pred_masks.shape)} high_res {list(out.high_res_masks.shape)} "
        f"obj_ptr {list(out.object_pointer.shape)} obj_score {list(out.object_score_logits.shape)} "
        f"= {float(out.object_score_logits.reshape(-1)[0]):.4f}"
    )

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "stages": {
            "pix_feat": stats(pix_feat),
            "feat_s0": stats(feat_s0),
            "feat_s1": stats(feat_s1),
            "pred_masks": stats(out.pred_masks),
            "high_res_masks": stats(out.high_res_masks),
            "object_pointer": stats(out.object_pointer),
            "object_score_logits": stats(out.object_score_logits),
        },
    }
    with open(os.path.join(OUT, "trackframe_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    save_file(
        {
            "pix_feat": det(pix_feat),  # [1,256,72,72]
            "feat_s0": det(feat_s0),  # [1,32,288,288]
            "feat_s1": det(feat_s1),  # [1,64,144,144]
            "pred_masks": det(out.pred_masks),  # [1,1,288,288]
            "high_res_masks": det(out.high_res_masks),  # [1,1,1008,1008]
            "object_pointer": det(out.object_pointer),  # [1,1,256]
            "object_score_logits": det(out.object_score_logits),  # [1,1,1]
        },
        os.path.join(OUT, "trackframe_fixture.safetensors"),
    )
    print("wrote trackframe_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()

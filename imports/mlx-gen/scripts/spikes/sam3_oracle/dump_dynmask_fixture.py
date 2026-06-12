#!/usr/bin/env python
"""SAM3 **tracker dynamic-multimask-via-stability** parity oracle — epic 4910, sc-4924 (Phase F2).

Runs the tracker single-frame box decode with `multimask_output=False`, which routes through
`_dynamic_multimask_via_stability` (modeling_sam3_tracker_video.py ~1550): keep mask token 0 if its
stability score (IoU between ±0.05-thresholded areas) ≥ 0.98, else fall back to the best-IoU multimask
candidate. Same image + box as the F1 tracker fixture; dumps the selected single mask + iou.

Run:  /tmp/sam3ref/.venv/bin/python dump_dynmask_fixture.py
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
from transformers import Sam3Processor, Sam3VideoModel

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/sam3"
torch.manual_seed(0)

URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
BOX_1008 = [430.0, 90.0, 700.0, 980.0]


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    tracker = model.tracker_model
    processor = Sam3Processor.from_pretrained(MODEL)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    pixel_values = processor(images=image, text="person", return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        vision_embeds = model.detector_model.vision_encoder(pixel_values)
        feats, _pos = model.get_vision_features_for_tracker(vision_embeds)
        sizes = tracker.backbone_feature_sizes
        high_res = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(feats[:-1], sizes[:-1])
        ]
        B, C = feats[-1].size(1), feats[-1].size(2)
        h, w = sizes[-1]
        pix = (feats[-1] + tracker.no_memory_embedding).permute(1, 2, 0).view(B, C, h, w)
        image_pe = tracker.get_image_wide_positional_embeddings()
        box = torch.tensor(BOX_1008, dtype=torch.float32).view(1, 1, 4)
        sparse, dense = tracker.prompt_encoder(
            input_points=None, input_labels=None, input_boxes=box, input_masks=None
        )
        # multimask_output=False → dynamic_multimask_via_stability path.
        masks, iou, _tok, obj = tracker.mask_decoder(
            image_embeddings=pix,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            high_resolution_features=high_res,
        )
    mask = masks.reshape(masks.shape[-2], masks.shape[-1])  # [mg, mg]
    iou_v = float(iou.reshape(-1)[0])
    print(f"  dyn mask {list(mask.shape)}  iou={iou_v:.4f}  obj={float(obj.reshape(-1)[0]):.4f}")

    manifest = {
        "model": MODEL,
        "box_1008": BOX_1008,
        "multimask_output": False,
        "dyn_iou": iou_v,
        "object_score": float(obj.reshape(-1)[0]),
        "stages": {"dyn_mask": stats(mask)},
    }
    with open(os.path.join(OUT, "dynmask_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    det = lambda t: t.detach().cpu().float().contiguous().clone()
    save_file(
        {
            "pixel_values": det(pixel_values),
            "box_1008": torch.tensor(BOX_1008, dtype=torch.float32),
            "dyn_mask": det(mask),  # [mg, mg]
            "dyn_iou": det(iou.reshape(-1)),  # [1]
            "object_score": det(obj.reshape(-1)),  # [1]
        },
        os.path.join(OUT, "dynmask_fixture.safetensors"),
    )
    print("wrote dynmask_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()

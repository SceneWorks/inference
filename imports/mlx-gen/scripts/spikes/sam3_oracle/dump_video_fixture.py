#!/usr/bin/env python
"""SAM3 **end-to-end multi-object video PCS** parity oracle — epic 4910, sc-4924 (Phase F2.6).

Drives the full `Sam3VideoModel` on a short clip (init_video_session + add_text_prompt +
propagate_in_video_iterator) and dumps, per frame, the per-`obj_id` low-res (288²) mask logits +
object id list, plus the preprocessed `pixel_values` and the tokenized prompt — everything the Rust
`Sam3VideoModel::propagate` pipeline needs to reproduce the run.

The Rust test feeds the captured frames + input_ids into the port and compares per-frame
per-`obj_id` masks (cosine) and the object-id sets.

Run:  /tmp/sam3ref/.venv/bin/python dump_video_fixture.py
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
NUM_FRAMES = 8  # exercises full memory bank (num_maskmem 7) + object-pointer accumulation; tight-parity horizon (long-horizon is cross-backend-chaos-limited, see story)


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5).astype(np.float32)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    video = [np.array(image) for _ in range(NUM_FRAMES)]

    session = processor.init_video_session(video=video, inference_device="cpu", dtype=torch.float32)
    processor.add_text_prompt(session, "person")

    det = lambda t: t.detach().cpu().float().contiguous().clone()
    tensors = {}
    # preprocessed frames + prompt tokens
    for f in range(NUM_FRAMES):
        tensors[f"frame_{f}"] = det(session.get_frame(f).unsqueeze(0))  # [1,3,1008,1008]
    input_ids = session.prompt_input_ids[0]
    attn = session.prompt_attention_masks[0]
    tensors["input_ids"] = input_ids.detach().cpu().to(torch.int64).contiguous().clone()  # [1,32]
    tensors["attention_mask"] = attn.detach().cpu().to(torch.int64).contiguous().clone()  # [1,32]
    tensors["num_frames"] = torch.tensor([NUM_FRAMES], dtype=torch.int64)

    per_frame = []
    with torch.no_grad():
        for out in model.propagate_in_video_iterator(session):
            f = out.frame_idx
            obj_ids = list(out.object_ids)
            # stack masks in object-id order → [num_obj, 288, 288] logits
            masks = [out.obj_id_to_mask[o].reshape(288, 288) for o in obj_ids]
            stacked = torch.stack(masks, 0) if masks else torch.zeros(0, 288, 288)
            tensors[f"masks_{f}"] = det(stacked)
            tensors[f"obj_ids_{f}"] = torch.tensor(obj_ids, dtype=torch.int64)
            per_frame.append({"frame": f, "obj_ids": obj_ids, "num_obj": len(obj_ids)})
            print(f"  frame {f}: obj_ids={obj_ids}")

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "num_frames": NUM_FRAMES,
        "input_ids": tensors["input_ids"].tolist(),
        "attention_mask": tensors["attention_mask"].tolist(),
        "frames": per_frame,
        "frame_stats": {f"masks_{p['frame']}": stats(tensors[f"masks_{p['frame']}"]) for p in per_frame},
    }
    with open(os.path.join(OUT, "video_fixture_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    save_file(tensors, os.path.join(OUT, "video_fixture.safetensors"))
    print("wrote video_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()

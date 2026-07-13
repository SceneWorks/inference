#!/usr/bin/env python
"""SAM3 **tracker memory attention** parity oracle — epic 4910, sc-4924 (Phase F2).

The memory attention (`Sam3TrackerVideoMemoryAttention`, modeling_sam3_tracker_video.py ~965) fuses a
frame's vision features with the temporal memory bank via 4 RoPE-attention layers. Its inputs are
assembled by `_prepare_memory_conditioned_features` (the memory bank + object pointers, F2.4), so the
cleanest isolated fixture is to **capture the real `memory_attention(...)` call** via a forward hook
during a real `Sam3VideoModel` PCS run on a 2-frame clip — no need to have ported the bank assembly.

Frame 0 is the initial-conditioning frame (returns the `no_memory_embedding` path, no attention);
frame 1 is the first memory-conditioned frame → the first `memory_attention` call. With a single
memory frame, `k_rot` length == `q` length, so `repeat_freqs_k` is a no-op (the simplest case).

Captures: the call kwargs (current_vision_features, current_vision_position_embeddings, memory,
memory_posision_embeddings, num_object_pointer_tokens), the RoPE cos/sin tables, and the output.

Run:  /tmp/sam3ref/.venv/bin/python dump_memattn_fixture.py
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
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    # 2-frame static clip (frame 1 conditions on frame 0's memory).
    video = [np.array(image), np.array(image)]

    session = processor.init_video_session(video=video, inference_device="cpu", dtype=torch.float32)
    processor.add_text_prompt(session, "person")

    captured = []

    def hook(module, args, kwargs, output):
        captured.append((kwargs, output))

    handle = model.tracker_model.memory_attention.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        for _out in model.propagate_in_video_iterator(session):
            pass
        cos, sin = model.tracker_model.memory_attention.rotary_emb()
    handle.remove()

    assert captured, "memory_attention was never called (no memory-conditioned frame?)"
    kwargs, output = captured[0]
    cvf = kwargs["current_vision_features"]
    cvp = kwargs["current_vision_position_embeddings"]
    mem = kwargs["memory"]
    mempos = kwargs["memory_posision_embeddings"]
    n_optr = int(kwargs.get("num_object_pointer_tokens", 0))
    print(
        f"  captured {len(captured)} call(s); first: cvf {list(cvf.shape)} memory {list(mem.shape)} "
        f"num_obj_ptr={n_optr} out {list(output.shape)}  cos {list(cos.shape)}"
    )

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "num_calls": len(captured),
        "num_object_pointer_tokens": n_optr,
        "stages": {
            "current_vision_features": stats(cvf),
            "current_vision_position_embeddings": stats(cvp),
            "memory": stats(mem),
            "memory_pos": stats(mempos),
            "rope_cos": stats(cos),
            "rope_sin": stats(sin),
            "output": stats(output),
        },
    }
    with open(os.path.join(OUT, "memattn_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    det = lambda t: t.detach().cpu().float().contiguous().clone()
    save_file(
        {
            "current_vision_features": det(cvf),  # [5184,1,256]
            "current_vision_position_embeddings": det(cvp),  # [5184,1,256]
            "memory": det(mem),  # [N,1,64]
            "memory_pos": det(mempos),  # [N,1,64]
            "rope_cos": det(cos),  # [5184,256]
            "rope_sin": det(sin),  # [5184,256]
            "num_object_pointer_tokens": torch.tensor([n_optr], dtype=torch.int64),
            "output": det(output),  # [5184,1,256]
        },
        os.path.join(OUT, "memattn_fixture.safetensors"),
    )
    print("wrote memattn_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()

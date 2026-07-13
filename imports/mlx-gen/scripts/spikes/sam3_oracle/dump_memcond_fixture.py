#!/usr/bin/env python
"""SAM3 **per-object memory conditioning** parity oracle — epic 4910, sc-4924 (Phase F2.4).

`_prepare_memory_conditioned_features` (modeling_sam3_tracker_video.py ~2425, non-init branch) fuses a
frame's vision features with the temporal memory bank: it gathers stored per-frame outputs
(`_gather_memory_frame_outputs`), builds the spatial memory + temporal-pos tensor
(`_build_memory_attention_inputs`), appends object pointers (`_get_object_pointers` /
`_process_object_pointers`), then runs memory attention. F2.4 ports that **bank assembly** (the
memory-attention math itself is F2 component 2, already parity-green).

To validate the assembly in isolation we drive a real `Sam3VideoModel` PCS run on a 2-frame clip and
wrap the three sub-methods to capture, for the first memory-conditioned call (frame 1, obj 0):
  - current_vision_features / _position_embeddings   (seq-first [5184,1,256])
  - the gathered memory frames: maskmem_features/_pos_enc [1,64,72,72] + relative_temporal_offset
  - the object pointers: [1,256] + temporal_offset, and max_object_pointers_to_use
  - num_object_pointer_tokens and the output conditioned_feature_map [1,256,72,72]

The Rust test rebuilds the `memory`/`memory_pos`/`num_obj_ptr` from the raw per-frame outputs and runs
the full conditioning, comparing the conditioned feature map (cosine gate >0.9999).

Run:  /tmp/sam3ref/.venv/bin/python dump_memcond_fixture.py
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
    # 2-frame static clip (frame 1 conditions on frame 0's single memory frame).
    video = [np.array(image), np.array(image)]

    session = processor.init_video_session(video=video, inference_device="cpu", dtype=torch.float32)
    processor.add_text_prompt(session, "person")

    tracker = model.tracker_model
    cap = {}  # first memory-conditioned call only

    orig_gather = tracker._gather_memory_frame_outputs
    orig_getptr = tracker._get_object_pointers
    orig_prepare = tracker._prepare_memory_conditioned_features

    def gather(*a, **k):
        out = orig_gather(*a, **k)
        if "gather" not in cap:
            cap["gather"] = out  # list[(offset, output_dict|None)]
        return out

    def getptr(*a, **k):
        out = orig_getptr(*a, **k)
        if "ptr" not in cap:
            cap["ptr"] = out  # (temporal_offsets, pointer_tokens, max_object_pointers_to_use)
        return out

    def prepare(*a, **k):
        # bind args by signature: (inference_session, frame_idx, obj_idx, is_initial_conditioning_frame,
        #   current_vision_features, current_vision_positional_embeddings, num_total_frames, ...)
        is_init = k.get("is_initial_conditioning_frame", a[3] if len(a) > 3 else None)
        out = orig_prepare(*a, **k)
        if not is_init and "out" not in cap:
            cap["cvf"] = k.get("current_vision_features", a[4] if len(a) > 4 else None)
            cap["cvp"] = k.get("current_vision_positional_embeddings", a[5] if len(a) > 5 else None)
            cap["out"] = out
        return out

    tracker._gather_memory_frame_outputs = gather
    tracker._get_object_pointers = getptr
    tracker._prepare_memory_conditioned_features = prepare
    try:
        with torch.no_grad():
            for _out in model.propagate_in_video_iterator(session):
                pass
    finally:
        tracker._gather_memory_frame_outputs = orig_gather
        tracker._get_object_pointers = orig_getptr
        tracker._prepare_memory_conditioned_features = orig_prepare

    assert "out" in cap, "no memory-conditioned frame captured"

    det = lambda t: t.detach().cpu().float().contiguous().clone()

    # Gathered spatial-memory frames (skip None padding slots).
    gathered = [(off, o) for off, o in cap["gather"] if o is not None]
    mem_feats = torch.stack([det(o["maskmem_features"]) for _off, o in gathered], dim=0)  # [M,1,64,72,72]
    mem_pos = torch.stack([det(o["maskmem_pos_enc"]) for _off, o in gathered], dim=0)  # [M,1,64,72,72]
    mem_offsets = torch.tensor([off for off, _o in gathered], dtype=torch.int64)  # [M]

    # Object pointers.
    p_offsets, p_tokens, max_optr = cap["ptr"]
    ptrs = torch.stack([det(t) for t in p_tokens], dim=0) if p_tokens else torch.zeros(0, 1, tracker.hidden_dim)
    ptr_offsets = torch.tensor(list(p_offsets), dtype=torch.int64)
    num_optr_tokens = (ptrs.shape[0] * (tracker.hidden_dim // tracker.mem_dim)) if p_tokens else 0

    cvf, cvp, out = cap["cvf"], cap["cvp"], cap["out"]
    print(
        f"  M={mem_feats.shape[0]} mem_offsets={mem_offsets.tolist()} "
        f"P={ptrs.shape[0]} ptr_offsets={ptr_offsets.tolist()} max_optr={max_optr} "
        f"num_obj_ptr_tokens={num_optr_tokens}"
    )
    print(f"  cvf {list(cvf.shape)}  out {list(out.shape)}")

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "num_memory_frames": int(mem_feats.shape[0]),
        "memory_frame_offsets": mem_offsets.tolist(),
        "num_object_pointers": int(ptrs.shape[0]),
        "object_pointer_offsets": ptr_offsets.tolist(),
        "max_object_pointers_to_use": int(max_optr),
        "num_object_pointer_tokens": int(num_optr_tokens),
        "stages": {
            "current_vision_features": stats(cvf),
            "current_vision_position_embeddings": stats(cvp),
            "memory_features": stats(mem_feats),
            "memory_pos_enc": stats(mem_pos),
            "object_pointers": stats(ptrs) if ptrs.numel() else None,
            "output": stats(out),
        },
    }
    with open(os.path.join(OUT, "memcond_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    save_file(
        {
            "current_vision_features": det(cvf),  # [5184,1,256]
            "current_vision_position_embeddings": det(cvp),  # [5184,1,256]
            "memory_features": mem_feats.contiguous(),  # [M,1,64,72,72]
            "memory_pos_enc": mem_pos.contiguous(),  # [M,1,64,72,72]
            "memory_offsets": mem_offsets,  # [M]
            "object_pointers": ptrs.contiguous(),  # [P,1,256]
            "object_pointer_offsets": ptr_offsets,  # [P]
            "max_object_pointers_to_use": torch.tensor([int(max_optr)], dtype=torch.int64),
            "num_object_pointer_tokens": torch.tensor([int(num_optr_tokens)], dtype=torch.int64),
            "output": det(out),  # [1,256,72,72]
        },
        os.path.join(OUT, "memcond_fixture.safetensors"),
    )
    print("wrote memcond_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()

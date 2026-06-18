"""Reference Boogu-Image T2I run on Apple MPS — spike quality check + golden capture.

Usage: PYTHONPATH=/tmp/boogu-ref python run_ref.py [height] [width] [steps]
"""
import os, sys, time, glob
os.environ.setdefault("device", "mps")

import torch
from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline

# Reference pipeline's device validator only accepts cpu/cuda; neutralize it so MPS passes.
BooguImagePipeline._validate_device_format = lambda self, *a, **k: None

H = int(sys.argv[1]) if len(sys.argv) > 1 else 768
W = int(sys.argv[2]) if len(sys.argv) > 2 else 768
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 25

SNAP = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Boogu--Boogu-Image-0.1-Base/snapshots/*/")))[0]
OUT = os.path.expanduser("~/Repos/mlx-gen-wt-boogu/reference/outputs")
os.makedirs(OUT, exist_ok=True)

print(f"[run_ref] snapshot={SNAP}", flush=True)
print(f"[run_ref] H={H} W={W} steps={STEPS} device=mps dtype=bf16", flush=True)

t0 = time.time()
pipe = BooguImagePipeline.from_pretrained(SNAP, torch_dtype=torch.bfloat16, trust_remote_code=True)
print(f"[run_ref] loaded in {time.time()-t0:.1f}s; moving to mps...", flush=True)
pipe.to("mps")
print(f"[run_ref] ready in {time.time()-t0:.1f}s", flush=True)

prompts = [
    ("en_street",
     "A street photography shot of an elderly scavenger with a deeply weathered, "
     "wrinkled face in the center of the frame. A trash can and a traffic light in "
     "the background. Shot on Leica, high photographic texture, cinematic lighting, photorealistic."),
    ("en_text",
     'A cozy bookstore storefront at dusk with a glowing neon sign that reads '
     '"BOOGU BOOKS", warm window light, rain-slick cobblestone street, bokeh.'),
]

for name, instruction in prompts:
    t1 = time.time()
    print(f"[run_ref] generating '{name}'...", flush=True)
    try:
        gen = torch.Generator("cpu").manual_seed(0)
        img = pipe(
            instruction=instruction,
            negative_instruction="",
            height=H, width=W,
            max_input_image_pixels=H * W,
            max_input_image_side_length=2 * max(H, W),
            num_inference_steps=STEPS,
            text_guidance_scale=4.0,
            device="mps",
            generator=gen,
        ).images[0]
        path = os.path.join(OUT, f"t2i_{name}_{H}x{W}_s{STEPS}.png")
        img.save(path)
        print(f"[run_ref] SAVED {path}  ({time.time()-t1:.1f}s, {(time.time()-t1)/STEPS:.2f}s/step)", flush=True)
    except Exception:
        import traceback; traceback.print_exc()
        print(f"[run_ref] FAILED '{name}'", flush=True)

print(f"[run_ref] done in {time.time()-t0:.1f}s total", flush=True)

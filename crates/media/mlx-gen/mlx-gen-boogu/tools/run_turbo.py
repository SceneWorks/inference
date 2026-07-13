"""Reference Boogu-Image TURBO (DMD 4-step, no CFG) on MPS — spike quality check.

Usage: PYTHONPATH=/tmp/boogu-ref python run_turbo.py [height] [width] [steps]
"""
import os, sys, time, glob
os.environ.setdefault("device", "mps")
import torch
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

# Reference pipeline's device validator only accepts cpu/cuda; neutralize for MPS.
BooguImageTurboPipeline._validate_device_format = lambda self, *a, **k: None

H = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

SNAP = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Boogu--Boogu-Image-0.1-Turbo/snapshots/*/")))[0]
OUT = os.path.expanduser("~/Repos/mlx-gen-wt-boogu/reference/outputs")
os.makedirs(OUT, exist_ok=True)
print(f"[turbo] snap={SNAP}\n[turbo] H={H} W={W} steps={STEPS} device=mps", flush=True)

t0 = time.time()
pipe = BooguImageTurboPipeline.from_pretrained(SNAP, torch_dtype=torch.bfloat16, trust_remote_code=True)
pipe.to("mps")
print(f"[turbo] ready in {time.time()-t0:.1f}s", flush=True)

prompts = [
    ("street", "A street photography shot of an elderly scavenger with a deeply weathered, "
     "wrinkled face in the center of the frame. A trash can and a traffic light in the "
     "background. Shot on Leica, high photographic texture, cinematic lighting, photorealistic."),
    ("text", 'A cozy bookstore storefront at dusk with a glowing neon sign that reads '
     '"BOOGU BOOKS", warm window light, rain-slick cobblestone street, bokeh.'),
    ("zh_poster", "一张中文电影海报，背景是夜晚的赛博朋克城市，霓虹灯招牌上写着“未来之城”四个大字，"
     "下方有一行小字“2026年夏季上映”。电影质感，高细节，戏剧性灯光。"),
    ("illus", "A richly detailed isometric illustration of a cozy ramen shop on a rainy night, "
     "steam rising, lanterns glowing, a cat on the counter, vibrant colors, studio-quality."),
]
for name, instruction in prompts:
    t1 = time.time()
    print(f"[turbo] generating '{name}'...", flush=True)
    try:
        img = pipe(
            instruction=instruction, negative_instruction="",
            height=H, width=W,
            max_input_image_pixels=H * W, max_input_image_side_length=2 * max(H, W),
            num_inference_steps=STEPS,
            text_guidance_scale=1.0,            # DMD path requires no CFG
            use_dmd_student_inference=True,
            device="mps",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        p = os.path.join(OUT, f"turbo_{name}_{H}x{W}_s{STEPS}.png"); img.save(p)
        print(f"[turbo] SAVED {p}  ({time.time()-t1:.1f}s, {(time.time()-t1)/STEPS:.2f}s/step)", flush=True)
    except Exception:
        import traceback; traceback.print_exc(); print(f"[turbo] FAILED '{name}'", flush=True)
print(f"[turbo] done in {time.time()-t0:.1f}s", flush=True)

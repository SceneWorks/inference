"""Root-cause probe (sc-2349): is the Rust-vs-fork Q8 control gap really the activation dtype?

Loads the fork's Q8 ZImageControl, feeds the Q8 golden's EXACT cap_feats / control_context / init,
and runs the 8-step loop TWICE inside the fork: once with bf16 activations (the normal path) and once
with f32 activations (inputs cast to f32 each step → quantized_matmul runs f32). Compares the two
fork outputs (this is the fork's OWN bf16-vs-f32 activation difference), and dumps the f32 final
latents + decoded so the Rust Q8 can be compared against the fork's *f32* run.

If fork-bf16 vs fork-f32 ≈ the Rust-vs-fork gap (~8% px>8), the gap is the activation dtype and Rust
(f32) should match fork-f32 tightly. If fork-bf16 ≈ fork-f32 (small), the activation-dtype story is
WRONG and there is a real Rust bug to find.

Run: cd ~/Repos/mflux-sc2257 && uv run python ~/Repos/mlx-gen/tools/probe_z_control_q8_dtype.py
"""

import glob
import os

import mlx.core as mx
import numpy as np

from _paths import hf_hub_cache
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image_control import ZImageControl
from mflux.utils.image_util import ImageUtil

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
Q8_GOLDEN = os.path.join(_GOLDEN_DIR, "z_image_control_q8_golden.safetensors")


def _find_control_weights() -> str:
    if os.environ.get("CONTROL_WEIGHTS"):
        return os.environ["CONTROL_WEIGHTS"]
    pat = str(
        hf_hub_cache()
        / "models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1"
        / "snapshots/*/*.safetensors"
    )
    return glob.glob(pat)[0]


g = mx.load(Q8_GOLDEN)
meta_path = Q8_GOLDEN
# read meta via safetensors header (mx.load drops metadata); just hardcode from the dump defaults
W = H = 1024
STEPS = 8
SCALE = 1.0

model = ZImageControl(control_weights_path=_find_control_weights(), quantize=8)

cap_feats = g["cap_feats"]          # f32 (the fork's Q8 TE output, saved f32)
control_context = g["control_context"]
init = g["init"]                    # f32 seeded noise
sigmas = g["sigmas"]


def run(dtype):
    latents = init.astype(dtype)
    cap = cap_feats.astype(dtype)
    cc = control_context.astype(dtype)
    v0 = None
    for t in range(STEPS):
        ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
        v = model.transformer(
            x=latents, timestep=ts, sigmas=sigmas, cap_feats=cap,
            control_context=cc, control_context_scale=SCALE,
        )
        if t == 0:
            v0 = v
        latents = latents + (sigmas[t + 1] - sigmas[t]) * v
        mx.eval(latents)
    return latents, v0


def decode(latents):
    unpacked = ZImageLatentCreator.unpack_latents(latents.astype(mx.float32), H, W)
    return model.vae.decode(unpacked)


def to_rgb8(decoded):
    return np.array(ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded))))


lat_bf16, v0_bf16 = run(mx.bfloat16)
lat_f32, v0_f32 = run(mx.float32)

# v0 (single forward) bf16 vs f32 — the per-step activation-dtype sensitivity.
v0_pr = float(mx.max(mx.abs(v0_bf16.astype(mx.float32) - v0_f32)).item() / mx.max(mx.abs(v0_f32)).item())
# final latents bf16 vs f32.
lat_pr = float(mx.max(mx.abs(lat_bf16.astype(mx.float32) - lat_f32)).item() / mx.max(mx.abs(lat_f32)).item())
print(f"fork Q8 activations bf16-vs-f32: v0 peak_rel={v0_pr:.3e}  final_latents peak_rel={lat_pr:.3e}")

img_bf16 = to_rgb8(decode(lat_bf16))
img_f32 = to_rgb8(decode(lat_f32))
differ = int(np.sum(np.abs(img_bf16.astype(int) - img_f32.astype(int)) > 8))
total = img_bf16.size
print(f"fork Q8 decoded bf16-vs-f32: {differ}/{total} px>8 = {100.0*differ/total:.3f}%")

# Dump the fork's f32 final latents + decoded so Rust-Q8 (also f32) can be compared against it.
out = os.path.join(_GOLDEN_DIR, "z_image_control_q8_f32_golden.safetensors")
mx.save_safetensors(out, {
    "final_latents": lat_f32.astype(mx.float32),
    "decoded": decode(lat_f32).astype(mx.float32),
    "v0": v0_f32.astype(mx.float32),
}, {"steps": str(STEPS), "w": str(W), "h": str(H), "control_scale": str(SCALE)})
print(f"wrote {out}")

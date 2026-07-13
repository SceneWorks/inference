"""A/B the Z-Image-Turbo denoise schedule: mflux `linear` (dynamic, resolution-dependent shift)
vs static `shift=3.0` (the model's scheduler_config.json). Same model, prompt, seed, steps, and
seeded noise — only the sigma schedule differs — so any image difference is purely the schedule.

Run from the fork:
  cd ~/repos/mflux && uv run python /path/to/mlx-gen/tools/compare_z_image_schedulers.py
"""

import math
import os

import mlx.core as mx
import numpy as np
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.image_util import ImageUtil

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "sched_compare")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = os.environ.get("ZIMAGE_PROMPT", "a portrait of a red fox in a forest at golden hour, detailed fur")
SEED = int(os.environ.get("ZIMAGE_SEED", "42"))
STEPS = int(os.environ.get("ZIMAGE_STEPS", "8"))
# (W, H) cases: 1024² (linear≈shift3.16, near static 3.0) and 512² (linear≈shift1.88, far from 3.0).
CASES = [(1024, 1024), (512, 512)]


def linear_mu(w, h):
    m = (1.15 - 0.5) / (4096 - 256)
    b = 0.5 - m * 256
    return m * (w * h / 256) + b


class Holder:
    pass


model = Holder()
ZImageInitializer.init(model, model_config=ModelConfig.z_image_turbo(), quantize=None)
tok = model.tokenizers["z_image"]
cap_feats = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)


def render(mu, w, h, steps, seed):
    sigmas = mx.linspace(1.0, 1.0 / steps, steps)
    sigmas = S._time_shift_exponential_array(mu, 1.0, sigmas)
    sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
    latents = ZImageLatentCreator.create_noise(seed, h, w)
    for t in range(steps):
        ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
        v = model.transformer(x=latents, timestep=ts, sigmas=sigmas, cap_feats=cap_feats)
        latents = latents + (sigmas[t + 1] - sigmas[t]) * v
        mx.eval(latents)
    unpacked = ZImageLatentCreator.unpack_latents(latents, h, w)
    decoded = model.vae.decode(unpacked)
    img = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
    return img, [round(float(s), 3) for s in sigmas]


print(f"prompt={PROMPT!r}  seed={SEED}  steps={STEPS}\n")
for w, h in CASES:
    mu_lin = linear_mu(w, h)
    img_lin, sg_lin = render(mu_lin, w, h, STEPS, SEED)
    img_30, sg_30 = render(math.log(3.0), w, h, STEPS, SEED)
    p_lin = os.path.join(OUT_DIR, f"sched_{w}x{h}_linear.png")
    p_30 = os.path.join(OUT_DIR, f"sched_{w}x{h}_shift3.png")
    img_lin.save(p_lin)
    img_30.save(p_30)
    a = np.array(img_lin).astype(int)
    b = np.array(img_30).astype(int)
    d = np.abs(a - b)
    print(f"=== {w}x{h} ===")
    print(f"  linear  : shift={math.exp(mu_lin):.2f}  sigmas={sg_lin}")
    print(f"  shift3.0: shift=3.00  sigmas={sg_30}")
    print(
        f"  pixel diff (linear vs 3.0): mean={d.mean():.2f}  max={d.max()}  "
        f"%px>8={(d > 8).mean() * 100:.1f}%  %px>32={(d > 32).mean() * 100:.1f}%"
    )
    print(f"  saved {p_lin}\n        {p_30}\n")

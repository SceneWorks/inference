"""Real-weights Z-Image **img2img** golden — the reference for the mlx-gen img2img port (sc-2533).

Run from the fork:
  cd ~/repos/mflux && uv run python tools/dump_z_image_img2img_golden.py

Mirrors `ZImage.generate_image` on the img2img branch (a `Reference` image + `image_strength`),
using the **flow_match_euler_discrete** scheduler — the schedule the Rust port + the
`mflux-generate-z-image` CLI use (NOT the `-turbo` CLI's `linear` default). Dumps every stage so
the Rust port can be validated piece by piece:

  - init_image_u8   : the synthetic RGB init image (int32 HWC) — the Rust test reads these exact
                      bytes so both sides start from an identical image (no PIL-load drift).
  - image_nchw      : the fork's `ImageUtil.to_array(scale_to_dimensions(...))` (LANCZOS → [-1,1]).
  - clean_encoded   : `VAEUtil.encode` of that image ([1,16,H/8,W/8]).
  - clean           : `ZImageLatentCreator.pack_latents` of clean_encoded ([16,1,H/8,W/8]).
  - init_latents    : the blended init `(1-σ)·clean + σ·noise` at σ = sigmas[init_time_step].
  - final_latents   : after the denoise loop `range(init_time_step, steps)`.
  - decoded         : VAE-decoded image tensor.
  - sigmas          : the flow-match schedule (len steps+1).

Env-overridable (ZIMAGE_*): PROMPT, SEED, STEPS, W, H, STRENGTH, and the init-image size IW/IH.
"""

import os
import tempfile

import mlx.core as mx
import numpy as np
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.image_util import ImageUtil
from PIL import Image

# Golden lives next to this script (tools/golden/), gitignored — and is where the Rust test's
# `CARGO_MANIFEST_DIR/../tools/golden` resolves when run from this checkout/worktree.
_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)
OUT = os.path.join(_GOLDEN_DIR, "z_image_img2img_golden.safetensors")
PNG_IN = os.path.join(_GOLDEN_DIR, "z_image_img2img_init.png")
PNG_OUT = os.path.join(_GOLDEN_DIR, "z_image_img2img_out.png")

PROMPT = os.environ.get("ZIMAGE_PROMPT", "a fox in autumn leaves")
SEED = int(os.environ.get("ZIMAGE_SEED", "42"))
STEPS = int(os.environ.get("ZIMAGE_STEPS", "4"))
W = int(os.environ.get("ZIMAGE_W", "256"))
H = int(os.environ.get("ZIMAGE_H", "256"))
STRENGTH = float(os.environ.get("ZIMAGE_STRENGTH", "0.6"))
# Init-image size — deliberately non-square and not a multiple of the target so the LANCZOS
# scale_to_dimensions path is exercised (a no-op resize would hide resampler bugs).
IW = int(os.environ.get("ZIMAGE_IW", "384"))
IH = int(os.environ.get("ZIMAGE_IH", "320"))

# Synthetic init image: smooth diagonal gradients with per-channel phase — deterministic, and
# bit-reproducible in Rust from the dumped bytes.
yy, xx = np.mgrid[0:IH, 0:IW]
r = ((xx * 255) // max(IW - 1, 1)).astype(np.uint8)
g = ((yy * 255) // max(IH - 1, 1)).astype(np.uint8)
b = (((xx + yy) * 255) // max(IW + IH - 2, 1)).astype(np.uint8)
init_u8 = np.stack([r, g, b], axis=-1).astype(np.uint8)  # HWC
Image.fromarray(init_u8, mode="RGB").save(PNG_IN)

model_config = ModelConfig.z_image_turbo()


class Holder:
    pass


model = Holder()
ZImageInitializer.init(model, model_config=model_config, quantize=None)
tok = model.tokenizers["z_image"]

# 0. Config (for dims + init_time_step, both scheduler-independent) and the STATIC shift=3.0
# schedule the Rust port uses (Z-Image-Turbo scheduler_config.json: shift=3.0,
# use_dynamic_shifting=false) — mu=ln(3) makes the exponential time-shift == diffusers' static
# shift (sc-2536). We build sigmas by hand and do NOT touch config.scheduler.
import math  # noqa: E402

config = Config(
    width=W,
    height=H,
    guidance=0.0,
    scheduler="flow_match_euler_discrete",  # unused — sigmas built statically below
    image_path=PNG_IN,
    image_strength=STRENGTH,
    model_config=model_config,
    num_inference_steps=STEPS,
)
sigmas = mx.linspace(1.0, 1.0 / STEPS, STEPS)
sigmas = S._time_shift_exponential_array(math.log(3.0), 1.0, sigmas)
sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
init_step = config.init_time_step
print(f"init_time_step={init_step}  strength={STRENGTH}  steps={STEPS}  W={W} H={H}")
print(f"sigmas={[round(float(s), 5) for s in sigmas]}")

# 1a. Preprocessed image (LANCZOS scale → [-1,1] NCHW) — isolates resize+normalize parity.
scaled_user = ImageUtil.scale_to_dimensions(
    image=ImageUtil.load_image(PNG_IN).convert("RGB"), target_width=config.width, target_height=config.height
)
image_nchw = ImageUtil.to_array(scaled_user)

# 1b. Clean latents (encode + pack) — isolates the VAE encoder.
clean_encoded = LatentCreator.encode_image(
    vae=model.vae, image_path=config.image_path, height=config.height, width=config.width
)
clean = ZImageLatentCreator.pack_latents(clean_encoded, config.height, config.width)

# 1c. Blended init latents = exactly what create_for_txt2img_or_img2img returns.
init_latents = LatentCreator.create_for_txt2img_or_img2img(
    seed=SEED,
    width=config.width,
    height=config.height,
    img2img=Img2Img(
        vae=model.vae,
        latent_creator=ZImageLatentCreator,
        image_path=config.image_path,
        sigmas=sigmas,
        init_time_step=config.init_time_step,
        tiling_config=None,
    ),
)

# 2. Prompt → cap_feats.
cap_feats = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)
num_valid = cap_feats.shape[0]

# 3. Denoise loop over range(init_time_step, steps) — mirrors generate_image.
latents = init_latents
for t in range(init_step, STEPS):
    ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
    v = model.transformer(x=latents, timestep=ts, sigmas=sigmas, cap_feats=cap_feats)
    latents = latents + (sigmas[t + 1] - sigmas[t]) * v
    mx.eval(latents)

unpacked = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
decoded = model.vae.decode(unpacked)

ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded))).save(PNG_OUT)

tensors = {
    "init_image_u8": mx.array(init_u8.astype(np.int32)),
    "image_nchw": image_nchw.astype(mx.float32),
    "clean_encoded": clean_encoded.astype(mx.float32),
    "clean": clean.astype(mx.float32),
    "init_latents": init_latents.astype(mx.float32),
    "final_latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
    "sigmas": sigmas.astype(mx.float32),
}
meta = {
    "prompt": PROMPT,
    "seed": str(SEED),
    "steps": str(STEPS),
    "w": str(W),
    "h": str(H),
    "strength": str(STRENGTH),
    "init_time_step": str(int(init_step)),
    "iw": str(IW),
    "ih": str(IH),
    "num_valid": str(int(num_valid)),
}
mx.save_safetensors(OUT, tensors, meta)
print(f"\nwrote {OUT}")
print(f"  init {IW}x{IH} -> target {W}x{H}; clean {tuple(clean.shape)}; decoded {tuple(decoded.shape)}")
print(f"  + {PNG_IN} (init) and {PNG_OUT} (result)")

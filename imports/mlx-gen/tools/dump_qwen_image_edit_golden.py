"""Dump a Qwen-Image-Edit e2e parity golden for the Rust port (sc-2465, slice 7a).

Runs the fork's edit denoise flow on a fixed synthetic reference image + prompt + seed, and dumps
the loop *inputs* (noise, pos/neg prompt embeds, the packed static reference latents, the cond grid,
the output dims) and *outputs* (final latents + decoded image). The Rust test feeds the same inputs
through `denoise_edit_with_progress` (the dual-latent loop + transformer `cond_grids`), so the gate
isolates the new 7a path from the tokenizer / VL encoder / VAE-encode (each separately verified).

Loads the full Edit model (~40 GB+); run on a machine with enough RAM.
Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/Repos/mlx-gen/.claude/worktrees/musing-mclaren-676094/tools/dump_qwen_image_edit_golden.py
Output (gitignored): tools/golden/qwen_image_edit_golden.safetensors
"""

import os

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

SEED = 42
PROMPT = "make it autumn"
STEPS = 2
GUIDANCE = 4.0

# Fixed synthetic reference image (deterministic gradient) → a temp PNG the fork loads by path.
W0, H0 = 512, 512
base = np.add.outer(np.arange(H0), np.arange(W0)).astype(np.int64) % 256
rgb = np.stack([base, (base * 2) % 256, (base * 3) % 256], axis=-1).astype(np.uint8)
ref_path = "/tmp/qwen_edit_ref.png"
Image.fromarray(rgb).save(ref_path)

QUANTIZE = int(os.environ["QUANTIZE"]) if os.environ.get("QUANTIZE") else None
model = QwenImageEdit(quantize=QUANTIZE)  # defaults → Qwen-Image-Edit-2509; Q8 quantizes transformer

config, vl_w, vl_h, vae_w, vae_h = model._compute_dimensions(
    width=None,
    height=None,
    guidance=GUIDANCE,
    scheduler="linear",
    image_path=None,
    image_paths=[ref_path],
    num_inference_steps=STEPS,
)

latents = QwenLatentCreator.create_noise(seed=SEED, width=config.width, height=config.height)
noise0 = latents

pos_emb, pos_mask, neg_emb, neg_mask = model._encode_prompts_with_images(
    prompt=PROMPT,
    negative_prompt=None,
    image_paths=[ref_path],
    config=config,
    vl_width=vl_w,
    vl_height=vl_h,
)

static, qwen_image_ids, cond_h, cond_w, num_images = QwenEditUtil.create_image_conditioning_latents(
    vae=model.vae,
    width=vae_w,
    height=vae_h,
    vl_width=vl_w,
    vl_height=vl_h,
    image_paths=[ref_path],
    tiling_config=model.tiling_config,
)

cond_image_grid = (1, cond_h, cond_w)
for t in range(len(config.scheduler.timesteps)):
    hidden = mx.concatenate([latents, static], axis=1)
    n_pos = model.transformer(
        t=t, config=config, hidden_states=hidden,
        encoder_hidden_states=pos_emb, encoder_hidden_states_mask=pos_mask,
        qwen_image_ids=qwen_image_ids, cond_image_grid=cond_image_grid,
    )[:, : latents.shape[1]]
    n_neg = model.transformer(
        t=t, config=config, hidden_states=hidden,
        encoder_hidden_states=neg_emb, encoder_hidden_states_mask=neg_mask,
        qwen_image_ids=qwen_image_ids, cond_image_grid=cond_image_grid,
    )[:, : latents.shape[1]]
    guided = QwenImage.compute_guided_noise(n_pos, n_neg, config.guidance)
    latents = config.scheduler.step(noise=guided, timestep=t, latents=latents)

final = latents
unpacked = QwenLatentCreator.unpack_latents(latents=final, height=config.height, width=config.width)
decoded = VAEUtil.decode(vae=model.vae, latent=unpacked, tiling_config=model.tiling_config)
mx.eval(noise0, pos_emb, neg_emb, static, final, decoded)

out = {
    "noise": noise0.astype(mx.float32),
    "pos_embeds": pos_emb.astype(mx.float32),
    "neg_embeds": neg_emb.astype(mx.float32),
    "static_image_latents": static.astype(mx.float32),
    "cond_grid": mx.array([cond_h, cond_w], dtype=mx.int32),
    "out_dims": mx.array([config.width, config.height], dtype=mx.int32),
    "final_latents": final.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
}
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
suffix = f"_q{QUANTIZE}" if QUANTIZE else ""
path_out = os.path.join(golden_dir, f"qwen_image_edit{suffix}_golden.safetensors")
mx.save_safetensors(path_out, out)
print(
    f"out={config.width}x{config.height} vl={vl_w}x{vl_h} cond=({cond_h},{cond_w}) "
    f"noise={noise0.shape} static={static.shape} final={final.shape} decoded={decoded.shape}"
)
print(f"wrote {path_out}")

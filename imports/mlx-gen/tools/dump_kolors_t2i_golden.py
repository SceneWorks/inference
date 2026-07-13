"""Kolors T2I end-to-end golden — reference for the full pipeline parity (sc-3094).

Runs the diffusers `KolorsPipeline` in **f32** (EulerDiscreteScheduler, leading spacing) with a FIXED
initial-noise tensor (so the Rust pipeline — which uses MLX RNG — can reproduce the trajectory by
feeding the same noise). The Euler step is non-ancestral (no per-step RNG), so the run is fully
deterministic. Dumps:

 - `init_noise` — the raw unit-normal latents (NHWC), pre-`init_noise_sigma` scaling (both sides
   apply the scale: diffusers' `prepare_latents`, Rust's `scale_initial_noise`).
 - the per-prompt conditioning the pipeline computes (`pos_context`/`pos_pooled`/`neg_*`) — for the
   tight scheduler+U-Net gate (feed identical conditioning, isolate the denoise+CFG).
 - `final_latents` (NHWC) — the denoised latents (pre-VAE), the tight e2e latent gate.
 - `image` — the VAE-decoded RGB (`[1,H,W,3]` in [0,1]), the pixel gate.

512²/8 steps to bound the CPU-f32 cost (parity is about matching the pipeline, not image quality).

Loads the full pipeline f32 (~35 GB). Run backgrounded:
    ~/repos/mflux/.venv-0312/bin/python tools/dump_kolors_t2i_golden.py
Output (gitignored): tools/golden/kolors_t2i_golden.safetensors
"""

import glob
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from _paths import fixture, hf_hub_cache

from diffusers import KolorsPipeline

PROMPT = "A cat playing a grand piano on a city rooftop at sunset."
NEGATIVE = "blurry, low quality"
STEPS = 8
CFG = 5.0
H = W = 512


def snapshot() -> Path:
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-diffusers" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-diffusers snapshot not found in HF cache")
    return Path(snaps[-1])


def nhwc(t):  # [B,C,H,W] → [B,H,W,C]
    return mx.array(t.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32))


def arr(t):
    return mx.array(t.detach().cpu().numpy().astype(np.float32))


@torch.no_grad()
def main():
    snap = snapshot()
    pipe = KolorsPipeline.from_pretrained(snap, variant="fp16", torch_dtype=torch.float32)
    pipe.to("cpu")

    device = torch.device("cpu")
    # Standalone conditioning (for the tight gate) — what __call__ computes internally.
    pos_embeds, neg_embeds, pos_pooled, neg_pooled = pipe.encode_prompt(
        prompt=PROMPT,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=NEGATIVE,
    )

    # Fixed raw init noise (NCHW), pre init_noise_sigma. KolorsPipeline.prepare_latents scales any
    # provided `latents` by init_noise_sigma, matching the Rust `scale_initial_noise`.
    g = torch.Generator(device="cpu").manual_seed(0)
    raw = torch.randn(1, 4, H // 8, W // 8, generator=g, dtype=torch.float32)

    step_latents = []

    def cb(pipe, step, timestep, kw):
        step_latents.append(kw["latents"].detach().clone())
        return kw

    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        height=H,
        width=W,
        latents=raw.clone(),
        output_type="latent",
        callback_on_step_end=cb,
    )
    final_latents = out.images  # [1,4,64,64] (pre-VAE)
    print("raw norm:", float(raw.norm()), "init_noise_sigma:", float(pipe.scheduler.init_noise_sigma))
    print("step0 latents norm:", float(step_latents[0].norm()) if step_latents else None)
    image = pipe.vae.decode(final_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]

    tensors = {
        "init_noise": nhwc(raw),
        "pos_context": arr(pos_embeds),
        "pos_pooled": arr(pos_pooled),
        "neg_context": arr(neg_embeds),
        "neg_pooled": arr(neg_pooled),
        "final_latents": nhwc(final_latents),
        "image": nhwc(image),
        "step0_latents": nhwc(step_latents[0]),
        "step1_latents": nhwc(step_latents[1]),
    }
    mx.eval(list(tensors.values()))
    meta = {
        "prompt": PROMPT,
        "negative": NEGATIVE,
        "steps": str(STEPS),
        "cfg": str(CFG),
        "h": str(H),
        "w": str(W),
    }
    out_path = fixture("tools/golden/kolors_t2i_golden.safetensors")
    mx.save_safetensors(out_path, tensors, metadata=meta)
    print(f"wrote {out_path}")
    print(f"  init_noise {tuple(tensors['init_noise'].shape)} final_latents "
          f"{tuple(tensors['final_latents'].shape)} image {tuple(tensors['image'].shape)}")


if __name__ == "__main__":
    main()

"""Real-weights FLUX.1 LoRA + LoKr golden — the reference for the mlx-gen adapter gate (sc-2657).

Renders the SAME fixed (prompt, seed, steps, size, guidance) FLUX.1-dev generation THREE ways through
the fork's real pipeline — no adapter (the base floor), with the real `zhibi_flux.safetensors` LoRA
(kohya/BFL `lora_unet_` naming, applied via `FluxLoRAMapping` + `LoRALoader`), and with a synthesized
LoKr (bare diffusers-path `lokr_w1/w2` keys + `networkType=lokr`, applied via `LoKrLoader`) — and dumps
the three decoded images into one golden file. The render mirrors `dump_flux_golden.py`
(`FluxInitializer.init` + the manual `transformer`/`LinearScheduler.step` loop), so the gate isolates
adapter divergence from the base (the Rust crate runs the same mixed bf16 path, sc-2787).

The Rust gate (`mlx-gen-flux/tests/adapter_real_weights.rs`) loads the SAME zhibi LoRA + the synthesized
LoKr adapter file via `LoadSpec.adapters` and asserts LoRA/LoKr px>8 ≤ the base floor (+ a visible
effect vs no-adapter), per sc-2528/sc-2602.

Gitignored output. Run from the fork venv pinned to the Rust MLX version (0.31.2, sc-2787/sc-2781):
    cd ~/Repos/mflux && .venv-0312/bin/python ~/Repos/mlx-gen/tools/dump_flux_lora_golden.py

Env-overridable: FLUX_VARIANT (dev|schnell), FLUX_LORA (path to the LoRA file), FLUX_SEED, FLUX_STEPS,
FLUX_W, FLUX_H, FLUX_GUIDANCE, FLUX_PROMPT.
"""

import os

import mlx.core as mx
import numpy as np
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

VARIANT = os.environ.get("FLUX_VARIANT", "dev")
LORA_PATH = os.environ.get(
    "FLUX_LORA", os.path.expanduser("~/repos/test-files/zhibi_flux.safetensors")
)
PROMPT = os.environ.get("FLUX_PROMPT", "zhibi, a cute chibi baby red panda, full body")
SEED = int(os.environ.get("FLUX_SEED", "7"))
W = int(os.environ.get("FLUX_W", "256"))
H = int(os.environ.get("FLUX_H", "256"))
STEPS = int(os.environ.get("FLUX_STEPS", "8"))
GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "0.0" if VARIANT == "schnell" else "3.5"))
LORA_SCALE = float(os.environ.get("FLUX_LORA_SCALE", "1.0"))
LOKR_SCALE = float(os.environ.get("FLUX_LOKR_SCALE", "1.0"))

OUT = os.path.join(_GOLDEN_DIR, f"flux1_{VARIANT}_adapter_golden.safetensors")
LOKR_ADAPTER = os.path.join(_GOLDEN_DIR, f"flux1_{VARIANT}_lokr_adapter.safetensors")

model_config = ModelConfig.schnell() if VARIANT == "schnell" else ModelConfig.dev()

# [3072, 3072] projections factor as kron(w1[48,48], w2[64,64]); these resolve identically in the fork
# (`LoKrLoader._navigate` getattr/index walk) and the Rust `AdaptableHost` (diffusers module paths).
LOKR_BLOCKS_DOUBLE = [0, 9, 18]
LOKR_BLOCKS_SINGLE = [0, 19, 37]
LOKR_DOUBLE_PROJS = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
]
LOKR_SINGLE_PROJS = ["attn.to_q", "attn.to_k", "attn.to_v"]
# kron(w1,w2) entries ~ std² so the delta magnitude grows quadratically; 0.12 gives a clearly visible
# (>3% px>8) effect vs no-adapter without degenerating the image (std 0.05 was too subtle — 2.68%).
LOKR_STD = 0.12


def build_lokr(path):
    """Synthesize a deterministic LoKr over a few [3072,3072] attention projections. kron(w1[48,48],
    w2[64,64]) = [3072,3072] = the projection delta shape; alpha==rank (scale 1.0)."""
    rng = np.random.default_rng(20260604)
    t = {}
    for blk in LOKR_BLOCKS_DOUBLE:
        for proj in LOKR_DOUBLE_PROJS:
            base = f"transformer_blocks.{blk}.{proj}"
            t[f"{base}.lokr_w1"] = mx.array(rng.normal(0.0, LOKR_STD, size=(48, 48)).astype(np.float32))
            t[f"{base}.lokr_w2"] = mx.array(rng.normal(0.0, LOKR_STD, size=(64, 64)).astype(np.float32))
    for blk in LOKR_BLOCKS_SINGLE:
        for proj in LOKR_SINGLE_PROJS:
            base = f"single_transformer_blocks.{blk}.{proj}"
            t[f"{base}.lokr_w1"] = mx.array(rng.normal(0.0, LOKR_STD, size=(48, 48)).astype(np.float32))
            t[f"{base}.lokr_w2"] = mx.array(rng.normal(0.0, LOKR_STD, size=(64, 64)).astype(np.float32))
    mx.save_safetensors(path, t, {"networkType": "lokr", "alpha": "1.0", "rank": "1"})
    return path


class Holder:
    pass


def render(lora_paths, lora_scales):
    """Fresh fork model with the given adapters applied, manual denoise, decoded NCHW f32 image."""
    model = Holder()
    FluxInitializer.init(
        model,
        model_config=model_config,
        quantize=None,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    config = Config(
        model_config=model_config,
        num_inference_steps=STEPS,
        height=H,
        width=W,
        guidance=GUIDANCE,
    )
    t5_out = model.tokenizers["t5"].tokenize(PROMPT)
    clip_out = model.tokenizers["clip"].tokenize(PROMPT)
    prompt_embeds = model.t5_text_encoder(t5_out.input_ids)
    pooled_prompt_embeds = model.clip_text_encoder(clip_out.input_ids)
    sigmas = config.scheduler.sigmas

    latents = FluxLatentCreator.create_noise(SEED, H, W)
    for t in range(STEPS):
        noise = model.transformer(
            t=t,
            config=config,
            hidden_states=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        latents = config.scheduler.step(noise, t, latents, sigmas=sigmas)
        mx.eval(latents)
    unpacked = FluxLatentCreator.unpack_latents(latents, H, W)
    decoded = model.vae.decode(unpacked)
    mx.eval(decoded)
    return decoded.astype(mx.float32)


build_lokr(LOKR_ADAPTER)
print(f"variant={VARIANT} prompt={PROMPT!r} seed={SEED} steps={STEPS} size={W}x{H} guidance={GUIDANCE}")
print(f"lora={LORA_PATH} (scale {LORA_SCALE}); lokr={LOKR_ADAPTER} (scale {LOKR_SCALE})")

base_decoded = render(None, None)
print("base decoded", tuple(base_decoded.shape))
lora_decoded = render([LORA_PATH], [LORA_SCALE])
print("lora decoded", tuple(lora_decoded.shape))
lokr_decoded = render([LOKR_ADAPTER], [LOKR_SCALE])
print("lokr decoded", tuple(lokr_decoded.shape))

mx.save_safetensors(
    OUT,
    {
        "base_decoded": base_decoded,
        "lora_decoded": lora_decoded,
        "lokr_decoded": lokr_decoded,
    },
    {
        "variant": VARIANT,
        "prompt": PROMPT,
        "seed": str(SEED),
        "steps": str(STEPS),
        "width": str(W),
        "height": str(H),
        "guidance": str(GUIDANCE),
        "lora_path": os.path.basename(LORA_PATH),
        "lora_scale": str(LORA_SCALE),
        "lokr_scale": str(LOKR_SCALE),
    },
)
print(f"\nwrote {OUT} + {LOKR_ADAPTER}")

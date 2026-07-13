#!/usr/bin/env python3
"""Dump S4 parity fixtures: a full **dense T2V denoise + decode** run of the `mlx_video` reference,
on a tiny seeded model, for the Rust pipeline to gate against.

Like S2, this is self-contained (no real weights): a tiny dense `WanModel` + tiny z16 `WanVAE` with
seeded random weights, an **injected** context (T5-embedding stand-in) + **injected** initial noise
(mlx-python and mlx-rs RNGs aren't comparable, so we inject rather than sample), run through the
reference's exact CFG denoise loop (Euler) + VAE decode. The Rust `pipeline::denoise` +
`decode_to_frames` must reproduce the final latents + video. Validates the loop composition,
CFG combine, scheduler stepping, and the DiT↔VAE hand-off end-to-end.

Run with the SceneWorks venv:
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s4_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/s4.json            (tiny config + run knobs + io shapes)
  - mlx-gen-wan/tests/fixtures/s4_pipeline.safetensors   (DiT + VAE weights + injected io + golden)
"""
import dataclasses
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.model import WanModel
from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
from mlx_video.models.wan.vae import CausalConv3d, Decoder3d, WanVAE

# --- tiny dense config (head_dim 128 kept so RoPE matches the gated path; everything else tiny) ---
CFG = dataclasses.replace(
    WanModelConfig.wan21_t2v_1_3b(),
    dim=128,
    num_heads=1,      # head_dim = 128
    num_layers=2,
    ffn_dim=256,
    freq_dim=256,
    text_dim=32,      # injected context dim (tiny)
    text_len=8,
    in_dim=16,
    out_dim=16,
    vae_z_dim=16,
    dual_model=False,
)
VAE_DIM = 4           # tiny VAE base channels (z_dim stays 16)
STEPS = 4
SHIFT = 5.0
GUIDE = 3.0
# tiny latent: 5 frames, 16×16 px → t_lat 2, h/w_lat 2 (stride 4×8×8); patch (1,2,2) → grid (2,1,1)
FRAMES = 5
HEIGHT = 16
WIDTH = 16
CTX_TOKENS = 4        # injected non-pad token count (< text_len)
RANDN = lambda *s: (mx.random.normal(s)).astype(mx.float32)  # noqa: E731


def build_models():
    mx.random.seed(0)
    model = WanModel(CFG)
    # seed DiT params (default init may be zeros/ones); keep bf16 like the real checkpoint.
    flat = tree_flatten(model.parameters())
    model.update(tree_unflatten([(k, (mx.random.normal(v.shape) * 0.1)) for k, v in flat]))
    mx.eval(model.parameters())

    mx.random.seed(1)
    vae = WanVAE(z_dim=16, encoder=False)
    vae.decoder = Decoder3d(dim=VAE_DIM, z_dim=16)
    vae.conv2 = CausalConv3d(16, 16, 1)
    keep = {"mean", "std", "inv_std"}
    vflat = tree_flatten(vae.parameters())
    vae.update(
        tree_unflatten(
            [
                (k, v if k.rsplit(".", 1)[-1] in keep else (mx.random.normal(v.shape) * 0.5).astype(mx.float32))
                for k, v in vflat
            ]
        )
    )
    mx.eval(vae.parameters())
    return model, vae


def main():
    model, vae = build_models()

    vae_stride = CFG.vae_stride
    patch = CFG.patch_size
    z_dim = CFG.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat = HEIGHT // vae_stride[1]
    w_lat = WIDTH // vae_stride[2]
    import math
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    grid = (t_lat // patch[0], h_lat // patch[1], w_lat // patch[2])

    mx.random.seed(2)
    ctx_cond = RANDN(CTX_TOKENS, CFG.text_dim)
    ctx_uncond = RANDN(CTX_TOKENS, CFG.text_dim)
    mx.random.seed(3)
    init_noise = RANDN(z_dim, t_lat, h_lat, w_lat)
    mx.eval(ctx_cond, ctx_uncond, init_noise)

    # --- reference CFG denoise loop (mirrors generate_wan.py, single-model B=2 path) ---
    context_emb = model.embed_text([ctx_cond, ctx_uncond])  # [2, text_len, dim]
    context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)
    cross_kv = model.prepare_cross_kv(context_cfg)
    rope_cos_sin = model.prepare_rope([grid, grid])

    sched = FlowMatchEulerScheduler(num_train_timesteps=CFG.num_train_timesteps)
    sched.set_timesteps(STEPS, shift=SHIFT)

    latents = init_noise
    for t in sched.timesteps.tolist():
        preds = model(
            [latents, latents],
            t=mx.array([t, t]),
            context=context_cfg,
            seq_len=seq_len,
            cross_kv_caches=cross_kv,
            rope_cos_sin=rope_cos_sin,
        )
        noise_pred = preds[1] + GUIDE * (preds[0] - preds[1])
        latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
        mx.eval(latents)
    final_latents = latents  # [z, t_lat, h_lat, w_lat]

    video = vae.decode(final_latents[None])  # [1, 3, F_out, H_out, W_out] in [-1,1]
    mx.eval(video)

    # --- save weights (DiT bf16 + VAE f32) + injected io + golden ---
    save = {}
    for k, v in tree_flatten(model.parameters()):
        save[k] = v.astype(mx.bfloat16)
    for k, v in tree_flatten(vae.parameters()):
        save[k] = v.astype(mx.float32)
    save["ctx_cond"] = ctx_cond
    save["ctx_uncond"] = ctx_uncond
    save["init_noise"] = init_noise
    save["final_latents"] = final_latents.astype(mx.float32)
    save["video"] = video.astype(mx.float32)

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    st = os.path.join(dst, "s4_pipeline.safetensors")
    mx.save_safetensors(st, save)

    meta = {
        "config": {
            f.name: list(getattr(CFG, f.name)) if isinstance(getattr(CFG, f.name), tuple) else getattr(CFG, f.name)
            for f in dataclasses.fields(CFG)
        },
        "vae_dim": VAE_DIM,
        "steps": STEPS,
        "shift": SHIFT,
        "guidance": GUIDE,
        "scheduler": "euler",
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "seq_len": seq_len,
        "grid": list(grid),
        "ctx_tokens": CTX_TOKENS,
        "final_latents_shape": list(final_latents.shape),
        "video_shape": list(video.shape),
    }
    with open(os.path.join(dst, "s4.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"final_latents {tuple(final_latents.shape)}  video {tuple(video.shape)}  seq_len {seq_len}  grid {grid}")
    print(f"wrote {os.path.abspath(st)} ({os.path.getsize(st) / 1e6:.2f} MB)")
    print(f"wrote {os.path.abspath(os.path.join(dst, 's4.json'))}")


if __name__ == "__main__":
    main()

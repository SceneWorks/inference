#!/usr/bin/env python3
"""Supra2-IMG reference fixtures (sc-24213).

Imports the frozen upstream `inference.py` (SupraLabs/Supra2-IMG @ 10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b)
for the model classes and re-states its `generate()` loop verbatim, with the initial noise injected
so a Rust run can be fed the identical tensor. Runs torch CPU float32 (upstream's CPU/MPS path: no
autocast).

`<upstream_dir>` is the pinned SupraLabs/Supra2-IMG snapshot (inference.py + model_final_ema.pt);
`<t5_dir>` / `<vae_dir>` are the pinned google/flan-t5-base and stabilityai/sd-vae-ft-mse snapshots.
Environment used for the committed fixtures: python3.12, torch 2.14.1 (CPU), transformers 5.20.2,
diffusers 0.41.0.

Usage:
  python -I dump_supra2_reference.py tiny  <upstream_dir> <gen_core_fixture_dir> <mlx_fixture_dir>
      -> gen-core/tests/fixtures/supra2/tiny_*.{pt,json}, mlx-gen-supra2/tests/fixtures/tiny_dit_parity.safetensors
  python -I dump_supra2_reference.py real  <upstream_dir> <t5_dir> <vae_dir> <out_dir>
      -> <out_dir>/{t5,dit,vae,trajectory}_reference.safetensors (large; not committed) plus
         supra2_reference_digests.json and tokenizer_cases.json (small; committed copies live in
         gen-core/tests/fixtures/supra2/)
"""
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

torch.set_grad_enabled(False)


def load_upstream(upstream_dir):
    spec = importlib.util.spec_from_file_location("supra2_upstream", os.path.join(upstream_dir, "inference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f32_digest(t):
    return hashlib.sha256(t.detach().contiguous().to(torch.float32).numpy().tobytes()).hexdigest()


def raw_digest(t):
    # Raw storage bytes in the tensor's own dtype (bf16 via int16 view).
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        b = t.view(torch.int16).numpy().tobytes()
    else:
        b = t.numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------------------------
# tiny
# ----------------------------------------------------------------------------------------------
TINY = dict(latent_ch=4, latent_size=16, d_model=32, depth=2, n_heads=2, mlp_ratio=4.0, ctx_dim=24, ctx_len=8)


def tiny_model(up, seed):
    torch.manual_seed(seed)
    m = up.SupraDiT(
        latent_ch=TINY["latent_ch"],
        d_model=TINY["d_model"],
        depth=TINY["depth"],
        n_heads=TINY["n_heads"],
        ctx_dim=TINY["ctx_dim"],
        mlp_ratio=TINY["mlp_ratio"],
        num_tokens=(TINY["latent_size"] // up.PATCH) ** 2,
    ).eval()
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.2)
    return m


def dump_tiny(upstream_dir, gen_core_dir, mlx_dir):
    up = load_upstream(upstream_dir)
    os.makedirs(gen_core_dir, exist_ok=True)
    os.makedirs(mlx_dir, exist_ok=True)
    model = tiny_model(up, 1)
    other = tiny_model(up, 2)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(3)
    L, C = TINY["ctx_len"], TINY["ctx_dim"]
    uncond_text = (torch.randn(L, C) * 0.5).to(torch.bfloat16)
    uncond_mask = torch.tensor([1] + [0] * (L - 1), dtype=torch.int64)
    torch.save({"model": sd, "config": {"patch": 2, "ctx_len": L, "uncond_text": uncond_text, "uncond_mask": uncond_mask}},
               os.path.join(gen_core_dir, "tiny_model_with_uncond.pt"))
    torch.save({"ema": sd, "model": {k: v.clone() for k, v in other.state_dict().items()}, "config": {"patch": 2, "ctx_len": L}},
               os.path.join(gen_core_dir, "tiny_ema_without_uncond.pt"))
    digests = {
        "denoiser_f32_sha256": {k: f32_digest(v) for k, v in sd.items()},
        "uncond_text_f32_sha256": f32_digest(uncond_text),
        "uncond_mask_f32_sha256": f32_digest(uncond_mask.float()),
        "tensor_count": len(sd),
    }
    with open(os.path.join(gen_core_dir, "tiny_digests.json"), "w") as f:
        json.dump(digests, f, indent=1, sort_keys=True)

    # Forward parity: two samples, a padded mask, two distinct timesteps, + the CFG mix.
    torch.manual_seed(4)
    B, S = 2, TINY["latent_size"]
    z = torch.randn(B, TINY["latent_ch"], S, S)
    t = torch.tensor([0.0, 0.62])
    ctx = torch.randn(B, L, C)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    v = model(z, t, ctx, mask)
    temb = model.t_embed._sinusoidal(t)
    out = {f"w.{k}": v_.contiguous() for k, v_ in sd.items()}
    out.update({"in.z": z, "in.t": t, "in.ctx": ctx, "in.mask": mask, "out.v": v, "out.t_freq": temb})
    save_file({k: x.contiguous().float() for k, x in out.items()}, os.path.join(mlx_dir, "tiny_dit_parity.safetensors"))
    print("tiny fixtures written; v absmax", float(v.abs().max()))


# ----------------------------------------------------------------------------------------------
# real
# ----------------------------------------------------------------------------------------------
PROMPTS = {
    "short": "a sea jellyfish floating in the pitch-black ocean depths",
    "empty": "",
    "long": " ".join(["a breathtaking panoramic watercolor of an ancient mountain village at golden hour,"] * 12),
    "unicode": "Café crème brûlée — naïve façade, 東京 at night",
}


def tokenize(tok, prompt, ctx_len):
    return tok([prompt], padding="max_length", truncation=True, max_length=ctx_len, return_tensors="pt")


def sample(model, z0, steps, cfg, ctx, cmask, uncond_ctx, uncond_mask):
    # Verbatim upstream loop body (inference.py::generate), CPU float32.
    n = z0.shape[0]
    z = z0.clone()
    use_cfg = cfg > 1.0
    if use_cfg:
        ctx_all = torch.cat([ctx, uncond_ctx], 0)
        mask_all = torch.cat([cmask, uncond_mask], 0)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        if use_cfg:
            v_both = model(torch.cat([z, z], 0), torch.cat([t, t], 0), ctx_all, mask_all)
            v_cond, v_uncond = v_both.float().chunk(2, 0)
            v = v_uncond + cfg * (v_cond - v_uncond)
        else:
            v = model(z, t, ctx, cmask).float()
        z = z + dt * v
    return z


def to_pixels(vae, z, scale):
    imgs = vae.decode(z / scale).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    # torchvision.utils.save_image quantization.
    return imgs, imgs.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)


def initial_noise(seed, n):
    """Port of gen_core::supra2::initial_noise (SplitMix64 + Box-Muller). The Rust trajectory test
    asserts its own generator reproduces this z0 exactly before comparing anything downstream."""
    mask = (1 << 64) - 1
    state = seed & mask
    out = []

    def uniform():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return ((z ^ (z >> 31)) >> 11) / float(1 << 53)

    while len(out) < n:
        r = math.sqrt(-2.0 * math.log(max(uniform(), 1e-300)))
        theta = 2.0 * math.pi * uniform()
        out.append(r * math.cos(theta))
        if len(out) < n:
            out.append(r * math.sin(theta))
    return np.asarray(out, dtype=np.float64).astype(np.float32)


def dump_real(upstream_dir, t5_dir, vae_dir, out_dir):
    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import AutoencoderKL
    from PIL import Image

    up = load_upstream(upstream_dir)
    os.makedirs(out_dir, exist_ok=True)
    state = torch.load(os.path.join(upstream_dir, "model_final_ema.pt"), map_location="cpu", weights_only=False)
    cfgd = state["config"]
    weights = state["ema"] if "ema" in state else state["model"]
    model = up.SupraDiT().eval()
    model.load_state_dict(weights, strict=True)
    ctx_len = int(cfgd.get("ctx_len", up.MAX_CTX_LEN))

    digests = {
        "source": "SupraLabs/Supra2-IMG@10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b/model_final_ema.pt",
        "weights_key": "ema" if "ema" in state else "model",
        "denoiser_f32_sha256": {k: f32_digest(v) for k, v in weights.items()},
        "uncond_text_raw_sha256": raw_digest(cfgd["uncond_text"]),
        "uncond_text_f32_sha256": f32_digest(cfgd["uncond_text"]),
        "uncond_mask_f32_sha256": f32_digest(cfgd["uncond_mask"].float()),
        "uncond_mask_sum": int(cfgd["uncond_mask"].sum()),
        "ctx_len": ctx_len,
        "patch": int(cfgd["patch"]),
        "tensor_count": len(weights),
    }
    with open(os.path.join(out_dir, "supra2_reference_digests.json"), "w") as f:
        json.dump(digests, f, indent=1, sort_keys=True)

    tok = AutoTokenizer.from_pretrained(t5_dir)
    t5 = T5EncoderModel.from_pretrained(t5_dir).eval()
    tokens = {}
    encoded = {}
    for name, p in PROMPTS.items():
        tk = tokenize(tok, p, ctx_len)
        tokens[name] = {
            "prompt": p,
            "ids": tk["input_ids"][0].tolist(),
            "mask": tk["attention_mask"][0].tolist(),
            "untruncated_len": len(tok(p)["input_ids"]),
        }
        encoded[f"ctx.{name}"] = t5(**tk).last_hidden_state.float()[0]
        encoded[f"mask.{name}"] = tk["attention_mask"].float()[0]
    with open(os.path.join(out_dir, "tokenizer_cases.json"), "w") as f:
        json.dump(tokens, f, indent=1, ensure_ascii=False)
    encoded["uncond_text"] = cfgd["uncond_text"].float()
    encoded["uncond_mask"] = cfgd["uncond_mask"].float()
    save_file({k: v.contiguous() for k, v in encoded.items()}, os.path.join(out_dir, "t5_reference.safetensors"))

    # Stored-uncond vs fallback empty-string encoding: how different are they (informational).
    print("stored-vs-fallback uncond maxabs", float((encoded["uncond_text"] - encoded["ctx.empty"]).abs().max()))

    # Single DiT forward (real weights) at a mid timestep with CFG batch.
    torch.manual_seed(11)
    z = torch.randn(1, 4, 32, 32)
    t = torch.tensor([0.3])
    ctx = encoded["ctx.short"][None]
    cm = encoded["mask.short"][None]
    uc = encoded["uncond_text"][None]
    um = encoded["uncond_mask"][None]
    v_both = model(torch.cat([z, z]), torch.cat([t, t]), torch.cat([ctx, uc]), torch.cat([cm, um]))
    save_file({"z": z, "t": t, "v_cond": v_both[:1].contiguous(), "v_uncond": v_both[1:].contiguous()},
              os.path.join(out_dir, "dit_reference.safetensors"))

    vae = AutoencoderKL.from_pretrained(vae_dir).eval()

    # Full trajectories from the backend-neutral initial noise for seed 0, images 0 and 1
    # (image i uses seed + i), so both sides integrate the same z0.
    z0 = torch.from_numpy(np.stack([initial_noise(0, 4096), initial_noise(1, 4096)]).reshape(2, 4, 32, 32))
    traj = {"z0": z0}
    p = "short"
    ctx = encoded[f"ctx.{p}"][None].expand(2, -1, -1)
    cm = encoded[f"mask.{p}"][None].expand(2, -1)
    stored_u = encoded["uncond_text"][None].expand(2, -1, -1)
    stored_m = encoded["uncond_mask"][None].expand(2, -1)
    fb_u = encoded["ctx.empty"][None].expand(2, -1, -1)
    fb_m = encoded["mask.empty"][None].expand(2, -1)
    cases = {
        "cfg3_steps50_stored": (50, 3.0, stored_u, stored_m),
        "cfg3_steps8_stored": (8, 3.0, stored_u, stored_m),
        "cfg3_steps8_fallback": (8, 3.0, fb_u, fb_m),
        "cfg1_steps8": (8, 1.0, None, None),
    }
    for name, (steps, cfg, u, m) in cases.items():
        zf = sample(model, z0, steps, cfg, ctx, cm, u, m)
        traj[f"{name}.latent"] = zf
        imgs, px = to_pixels(vae, zf, up.VAE_SCALE)
        traj[f"{name}.pixels"] = px.float()
        for i in range(2):
            Image.fromarray(px[i].permute(1, 2, 0).numpy()).save(os.path.join(out_dir, f"torch_{name}_{i}.png"))
        print(name, "latent absmax", float(zf.abs().max()))
    save_file({k: v.contiguous().float() for k, v in traj.items()}, os.path.join(out_dir, "trajectory_reference.safetensors"))

    # VAE decode in isolation: a denoised (unscaled) latent and a random in-scale latent.
    torch.manual_seed(12)
    zr = torch.randn(1, 4, 32, 32) * up.VAE_SCALE
    zd = traj["cfg3_steps50_stored.latent"][:1]
    save_file(
        {
            "latent": zd.contiguous(),
            "decoded": vae.decode(zd / up.VAE_SCALE).sample.contiguous(),
            "random_latent": zr,
            "random_decoded": vae.decode(zr / up.VAE_SCALE).sample.contiguous(),
        },
        os.path.join(out_dir, "vae_reference.safetensors"),
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tiny":
        dump_tiny(*sys.argv[2:5])
    elif mode == "real":
        dump_real(*sys.argv[2:6])
    else:
        raise SystemExit(f"unknown mode {mode}")

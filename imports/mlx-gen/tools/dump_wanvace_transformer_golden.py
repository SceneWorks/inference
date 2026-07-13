"""Dump a structural golden for the Wan-VACE transformer (epic 3040 / sc-3388, S0/S1) from a
**randomly-initialized small-config** diffusers `WanVACETransformer3DModel` — no 14B/1.3B checkpoint
needed. Pins the exact forward contract (96-ch control patch-embed → per-vace-layer hint → injection
into the main block residual stream) so the Rust port (sc-3434) can be byte-validated structurally in
f32 before the real VACE checkpoint is provisioned.

Saves the full (diffusers-named) state_dict + the inputs + the output so the Rust parity test loads
the same random weights, runs `forward_vace`, and compares.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_wanvace_transformer_golden.py
Writes `mlx-gen-wan/tests/fixtures/wanvace_transformer_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel

from _paths import fixture

torch.manual_seed(3388)

# A small config that still exercises every VACE mechanism: 4 main layers, vace at layers [0, 2]
# (must include 0 → block 0 carries proj_in), patch_size (1,2,2), 96-ch control.
NUM_HEADS = 4
HEAD_DIM = 16  # inner_dim = 64
cfg = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=NUM_HEADS,
    attention_head_dim=HEAD_DIM,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=64,
    ffn_dim=128,
    num_layers=4,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    image_dim=None,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
    pos_embed_seq_len=None,
    vace_layers=[0, 2],
    vace_in_channels=96,
)

model = WanVACETransformer3DModel(**cfg).to(torch.float32).eval()

# Inputs: a [1,16,T,H,W] noisy latent + a [1,96,T,H,W] control latent (T=4, H=W=8 → patchified
# T=4, H=W=4 → L=64 tokens). Text [1, text_len, text_dim].
T, H, W = 4, 8, 8
hidden_states = torch.randn(1, 16, T, H, W)
control_hidden_states = torch.randn(1, 96, T, H, W)
timestep = torch.tensor([3.0])
encoder_hidden_states = torch.randn(1, 12, cfg["text_dim"])
# Non-trivial per-vace-layer scales so the test catches a mis-applied hint scale.
control_scale = torch.tensor([1.0, 0.5])

with torch.no_grad():
    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        control_hidden_states=control_hidden_states,
        control_hidden_states_scale=control_scale,
        return_dict=False,
    )[0]

tensors = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
tensors.update(
    {
        "in.hidden_states": hidden_states,
        "in.control_hidden_states": control_hidden_states,
        "in.timestep": timestep,
        "in.encoder_hidden_states": encoder_hidden_states,
        "in.control_hidden_states_scale": control_scale,
        "out.sample": out,
    }
)

out_path = fixture("mlx-gen-wan/tests/fixtures/wanvace_transformer_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  state_dict tensors:", len(model.state_dict()))
print("  output:", tuple(out.shape), " mean/std:", float(out.mean()), float(out.std()))
print("  vace param keys (sample):")
for k in list(model.state_dict().keys()):
    if "vace" in k or "proj_in" in k or "proj_out" in k:
        print("   ", k, tuple(model.state_dict()[k].shape))

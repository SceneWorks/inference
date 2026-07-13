"""Full Z-Image **ControlNet** DiT forward parity fixture (sc-2349 / sc-2257).

A tiny synthetic ZImageControlTransformer (dim=64, 4 heads, 2 refiner + 4 main layers, in_ch=4,
patch=2) run end-to-end with a random control context. Dumps all weights + inputs + the control-on
output, the control-off (control_context=None) output, and asserts the fork's own self-consistency
(control_context_scale=0 == None == base). The Rust parity test rebuilds the same model from these
weights and reproduces all three, plus checks that the control branch is actually active (control-on
differs from control-off).

The before/after_proj projections are zero-initialised in a fresh VACE model (so control would be a
no-op and hide bugs); this script perturbs them to non-zero so the control path genuinely
contributes — exactly what the trained Fun-Controlnet checkpoint does.

Run from the mflux fork venv (the sc-2257 branch):
    cd ~/Repos/mflux-sc2257 && uv run python ~/Repos/mlx-gen/tools/dump_z_control_transformer.py
"""

import os

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_transformer.control_transformer import (
    ZImageControlTransformer,
)

mx.random.seed(0)

# n_layers=4 so the main control stack injects at >1 place (CONTROL_LAYERS_PLACES ∩ [0,4) = {0,2}),
# and n_refiner_layers=2 so both control refiner blocks inject (CONTROL_REFINER_PLACES = {0,1}).
CFG = dict(
    patch_size=2, f_patch_size=1, in_channels=4, dim=64, n_layers=4, n_refiner_layers=2,
    n_heads=4, norm_eps=1e-5, qk_norm=True, cap_feat_dim=32, rope_theta=256.0, t_scale=1000.0,
    axes_dims=[8, 4, 4], axes_lens=[64, 64, 64],
)  # fmt: off
model = ZImageControlTransformer(**CFG)

# Exercise the assembly-only cap RMSNorm (block/final norms are covered elsewhere).
model.cap_embedder[0].weight = 1.0 + 0.1 * mx.random.normal(model.cap_embedder[0].weight.shape)

# Perturb the zero-init control projections so the control branch actually contributes (a fresh VACE
# model zero-inits before/after_proj → control would be a no-op, making the test vacuous).
for blocks in (model.control_layers, model.control_noise_refiner):
    for blk in blocks:
        blk.after_proj.weight = 0.1 * mx.random.normal(blk.after_proj.weight.shape)
        blk.after_proj.bias = 0.1 * mx.random.normal(blk.after_proj.bias.shape)
        if hasattr(blk, "before_proj"):
            blk.before_proj.weight = 0.1 * mx.random.normal(blk.before_proj.weight.shape)
            blk.before_proj.bias = 0.1 * mx.random.normal(blk.before_proj.bias.shape)
mx.eval(model.parameters())

x = mx.random.normal((4, 1, 4, 4))            # (C=in_channels, F, H, W)
cap_feats = mx.random.normal((5, 32))          # (cap_len, cap_feat_dim)
control_context = mx.random.normal((33, 1, 4, 4))  # (33, F, H, W) — same spatial dims as x
timestep = mx.array(0.7, dtype=mx.float32)
sigmas = mx.linspace(1.0, 0.0, 8)              # unused for a float timestep; kept for signature

out = {f"w.{k}": v.astype(mx.float32) for k, v in tree_flatten(model.parameters())}
out["in.x"] = x.astype(mx.float32)
out["in.cap_feats"] = cap_feats.astype(mx.float32)
out["in.control_context"] = control_context.astype(mx.float32)

y_ctrl = model(x, timestep, sigmas, cap_feats, control_context=control_context, control_context_scale=1.0)  # fmt: off
y_none = model(x, timestep, sigmas, cap_feats, control_context=None)
y_scale0 = model(x, timestep, sigmas, cap_feats, control_context=control_context, control_context_scale=0.0)  # fmt: off

# Fork self-consistency: control_context=None and control_context_scale=0 both reproduce the base.
assert mx.allclose(y_none, y_scale0, atol=1e-5).item(), "fork: scale=0 != control_context=None"
# Sanity: the perturbed control branch genuinely changes the output (not a no-op).
delta = float(mx.max(mx.abs(y_ctrl - y_none)).item())
assert delta > 1e-2, f"control branch is inert (max|Δ|={delta:.2e}); perturbation too small"
print(f"control on-vs-off max|Δ| = {delta:.4f}  (control branch is active)")

out["out.y_ctrl"] = y_ctrl.astype(mx.float32)
out["out.y_none"] = y_none.astype(mx.float32)

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "mlx-gen-z-image",
    "tests",
    "fixtures",
    "z_control_transformer.safetensors",
)
path = os.path.abspath(path)
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
print("y_ctrl:", y_ctrl.shape, "| control_context:", control_context.shape)

"""Stage-by-stage bisection of the Rust-vs-fork Q8 control single forward (sc-2349).

Runs the fork's Q8 ZImageControlTransformer forward BY HAND (f32 activations) on the golden's exact
inputs, capturing every intermediate, and also v0 at scale=0 (base path, control inert) vs scale=1.
The Rust diag `control_q8_bisect` reproduces each stage and reports the first that diverges >1%.

Run: cd ~/Repos/mflux-sc2257 && uv run python ~/Repos/mlx-gen/tools/probe_z_control_q8_bisect.py
"""

import glob
import os

import mlx.core as mx

from _paths import hf_hub_cache
from mflux.models.z_image.model.z_image_transformer.control_transformer import ZImageControlTransformer
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock
from mflux.models.z_image.variants.z_image_control import ZImageControl

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
g = mx.load(os.path.join(_GOLDEN_DIR, "z_image_control_q8_golden.safetensors"))


def _cw():
    if os.environ.get("CONTROL_WEIGHTS"):
        return os.environ["CONTROL_WEIGHTS"]
    return glob.glob(str(hf_hub_cache() / "models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1" / "snapshots/*/*.safetensors"))[0]


m = ZImageControl(control_weights_path=_cw(), quantize=8).transformer
sigmas = g["sigmas"]
SCALE = 1.0
F32 = mx.float32

x = g["init"].astype(F32)
cap_feats = g["cap_feats"].astype(F32)
control_context = g["control_context"].astype(F32)
timestep = mx.array(1.0 - float(sigmas[0]), dtype=F32)

out = {}
key = f"{m.patch_size}-{m.f_patch_size}"
t_emb = m.t_embedder(timestep.reshape((1,)).astype(F32) * m.t_scale)
out["t_emb"] = t_emb

x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(
    image=x, cap_feats=cap_feats, patch_size=m.patch_size, f_patch_size=m.f_patch_size
)
x_emb = m.all_x_embedder[key](x_emb)
x_emb = mx.where(x_pad_mask[:, None], m.x_pad_token, x_emb)
x_freqs = m.rope_embedder(x_pos_ids)
x_emb = mx.expand_dims(x_emb, axis=0)
out["x_emb"] = x_emb

c_tokens = ZImageControlTransformer._patchify_control(control_context, m.patch_size, m.f_patch_size)
c_emb = m.control_all_x_embedder[key](c_tokens)
c_emb = mx.where(x_pad_mask[:, None], m.x_pad_token, c_emb)
c_emb = mx.expand_dims(c_emb, axis=0)
out["c_emb"] = c_emb

# control refiner pass (by hand, capturing hints + threaded)
refiner_hints = []
c = c_emb
for i, block in enumerate(m.control_noise_refiner):
    if i == 0:
        c = block.before_proj(c) + x_emb
    c = ZImageTransformerBlock.__call__(block, c, None, x_freqs, t_emb)
    refiner_hints.append(block.after_proj(c))
threaded = c
out["refiner_hint0"] = refiner_hints[0]
out["refiner_hint1"] = refiner_hints[1]
out["threaded"] = threaded

for i, layer in enumerate(m.noise_refiner):
    x_emb = layer(x=x_emb, attn_mask=None, freqs_cis=x_freqs, t_emb=t_emb)
    if i in m.control_refiner_mapping:
        x_emb = x_emb + refiner_hints[m.control_refiner_mapping[i]] * SCALE
out["x_refined"] = x_emb

cap_emb = m.cap_embedder[1](m.cap_embedder[0](cap_emb))
cap_emb = mx.where(cap_pad_mask[:, None], m.cap_pad_token, cap_emb)
cap_freqs = m.rope_embedder(cap_pos_ids)
cap_emb = mx.expand_dims(cap_emb, axis=0)
for layer in m.context_refiner:
    cap_emb = layer(x=cap_emb, attn_mask=None, freqs_cis=cap_freqs)
out["cap_refined"] = cap_emb

x_len = x_emb.shape[1]
unified = mx.concatenate([x_emb, cap_emb], axis=1)
unified_freqs = mx.concatenate([x_freqs, cap_freqs], axis=0)
control_unified = mx.concatenate([threaded, cap_emb], axis=1)
main_hints = []
c = control_unified
for i, block in enumerate(m.control_layers):
    if i == 0:
        c = block.before_proj(c) + unified
    c = ZImageTransformerBlock.__call__(block, c, None, unified_freqs, t_emb)
    main_hints.append(block.after_proj(c))
out["main_hint0"] = main_hints[0]
out["main_hint_last"] = main_hints[-1]

for i, layer in enumerate(m.layers):
    unified = layer(x=unified, attn_mask=None, freqs_cis=unified_freqs, t_emb=t_emb)
    if i in m.control_layers_mapping:
        unified = unified + main_hints[m.control_layers_mapping[i]] * SCALE
out["unified_main"] = unified

unified = m.all_final_layer[key](unified, t_emb)
v_staged = -ZImageTransformer._unpatchify(x=unified[0, :x_len], size=x_size, patch_size=m.patch_size, f_patch_size=m.f_patch_size, out_channels=m.out_channels)

# sanity: staged == __call__
v_call = m(x=x, timestep=timestep, sigmas=sigmas, cap_feats=cap_feats, control_context=control_context, control_context_scale=SCALE)
assert mx.allclose(v_staged, v_call, atol=1e-4).item(), "staged != __call__"
out["v0_scale1"] = v_call
out["v0_scale0"] = m(x=x, timestep=timestep, sigmas=sigmas, cap_feats=cap_feats, control_context=control_context, control_context_scale=0.0)

out = {k: v.astype(F32) for k, v in out.items()}
path = os.path.join(_GOLDEN_DIR, "z_control_q8_bisect.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} stages)")
for k, v in out.items():
    print(f"  {k}: {tuple(v.shape)}")

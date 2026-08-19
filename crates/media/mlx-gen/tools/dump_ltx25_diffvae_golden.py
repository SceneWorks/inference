"""LTX-2.5 DiffVAE (`NADiffusionDecoder`) reference golden — sc-18766.

Dumps f32 reference I/O for the MLX `NaDiffusionDecoder` port, taken from the **upstream**
`DiffusionVideoDecoder` running on the real `vae/ltx-2.5-video-vae-bf16.safetensors`. See
`_ltx25_diffvae_ref.py` for how the reference module is built (upstream configurator + SDOps,
combined pathway, upstream's own eager neighborhood-attention backend).

Four levels, so a port failure localizes instead of only showing up as a wrong picture:

  * `t_emb` / `adaln`  — the timestep embedder and the shared AdaLN-Zero projection;
  * `na_*`             — one deterministic `NABlock` (det stage 3, kernel 3x5x5);
  * `diff_*`           — one `CombinedDiffusionNABlock` (kernel 11x11x11) with real modulation;
  * `dec_*`            — a full untiled decode, plus slices of the stage-1-3 and stage-4 outputs.

Everything is f32: the weights ship bf16 and both sides upcast losslessly, so the comparison is a
correctness check rather than a rounding check. Geometry is the smallest legal one — the latent
floor is (3, 7, 7) because stage 1 attends a 7x7 spatial window — which still exercises every
upsample hop, the NATTEN trailing-frame ghost pad and its crop.

Run:
    LTX2_SRC=~/src/LTX-2/packages/ltx-core/src \\
    LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \\
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx25_diffvae_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx25_diffvae_golden.safetensors
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from _ltx25_diffvae_ref import (
    DIFF_VAE,
    REFERENCE_COMMIT,
    env_int,
    load_reference_decoder,
    probe,
    report,
    require_finite,
    stage5_geometry,
    untiled_decode,
    vae_dir,
)
from _paths import fixture

torch.manual_seed(0)

LATENT_T = env_int("LTX25_GOLDEN_T", 3)
LATENT_H = env_int("LTX25_GOLDEN_H", 7)
LATENT_W = env_int("LTX25_GOLDEN_W", 7)

path = vae_dir() / DIFF_VAE
print(f"[ref] {path}")
decoder = load_reference_decoder(path)
print(
    f"[ref] stage_channels={decoder.stage_channels} depths={decoder.stage_depths} "
    f"stage5_kernel={decoder.stage5_kernel} ghost_frames={decoder._natten_trailing_pad_latent_frames} "
    f"min_latent={decoder.stage_min_tile_sizes} output={decoder.model_output_type}"
)

tensors: dict[str, torch.Tensor] = {}
meta = {
    "story": "sc-18766",
    "reference": f"Lightricks/LTX-2 @ {REFERENCE_COMMIT} (v1.2.0)",
    "checkpoint": DIFF_VAE,
    "pathway": "CombinedDiffusionNABlock + fallback_na.eager (NATTEN na3d semantics)",
    "dtype": "float32 (weights upcast from bf16)",
}

# --- 1. timestep embedding + shared AdaLN-Zero -------------------------------------------------
with torch.no_grad():
    t = torch.tensor([1.0], dtype=torch.float32)
    t_emb = decoder.t_embedder(decoder.timestep_scale_multiplier * t, hidden_dtype=torch.float32)
    modulation = decoder.shared_adaln(t_emb)
    adaln = torch.cat([chunk.reshape(1, -1) for chunk in modulation], dim=-1)
tensors["t_emb"] = t_emb.contiguous()
tensors["adaln"] = adaln.contiguous()
meta["timestep"] = "1.0"
meta["timestep_scale_multiplier"] = str(decoder.timestep_scale_multiplier)
report("t_emb", t_emb)
report("adaln", adaln)

# --- 2. one deterministic NABlock (det stage 3: dim 512, kernel 3x5x5) -------------------------
na_block = decoder.det_stages[3][0]
na_dim = na_block.attn.dim
na_in = probe((1, 3, 6, 6, na_dim), seed=1)
with torch.no_grad():
    na_out = na_block(na_in)
tensors["na_in"] = na_in.contiguous()
tensors["na_out"] = na_out.contiguous()
meta["na_stage"] = "det_stages[3][0]"
meta["na_kernel"] = "x".join(map(str, na_block.attn.kernel_size))
report("na_in", na_in)
report("na_out", na_out)

# --- 3. one CombinedDiffusionNABlock (dim 256, kernel 11x11x11) --------------------------------
diff_block = decoder.diff_blocks[0]
kt, kh, kw = decoder.stage5_kernel
ctx_dim = decoder.context_channels
x_dim = diff_block.attn.dim
diff_ctx = probe((1, kt, kh, kw, ctx_dim), seed=2)
diff_x = probe((1, kt, kh, kw, x_dim), seed=3)
with torch.no_grad():
    diff_out = diff_block.forward_combined(torch.cat([diff_ctx, diff_x], dim=-1), modulation)
tensors["diff_ctx"] = diff_ctx.contiguous()
tensors["diff_x"] = diff_x.contiguous()
tensors["diff_out"] = diff_out.contiguous()
meta["diff_block"] = "diff_blocks[0].forward_combined at t=1.0"
meta["diff_kernel"] = "x".join(map(str, decoder.stage5_kernel))
report("diff_ctx", diff_ctx)
report("diff_out", diff_out)

# --- 4. full untiled decode --------------------------------------------------------------------
latent = probe((1, 128, LATENT_T, LATENT_H, LATENT_W), seed=4)
frames5, h5, w5 = stage5_geometry(decoder, LATENT_T, LATENT_H, LATENT_W)
noise = probe((1, decoder.out_channels, frames5, h5, w5), seed=5)
print(f"[decode] latent {tuple(latent.shape)} + stage-5 noise {tuple(noise.shape)}")
with torch.no_grad():
    pixels, stage123, context = untiled_decode(decoder, latent, noise)
tensors["dec_latent"] = latent.contiguous()
tensors["dec_noise"] = noise.contiguous()
tensors["dec_out"] = pixels.contiguous()
# Corner slices of the two big intermediates: enough to localise a stage-1-3 vs stage-4 vs stage-5
# regression without committing tens of megabytes.
tensors["s123_slice"] = stage123[:, :, :2, :2, :].contiguous()
tensors["ctx_slice"] = context[:, :, :2, :2, :].contiguous()
meta["dec_latent"] = "x".join(map(str, latent.shape))
meta["dec_noise"] = "x".join(map(str, noise.shape))
meta["dec_out"] = "x".join(map(str, pixels.shape))
meta["s123_full"] = "x".join(map(str, stage123.shape))
meta["ctx_full"] = "x".join(map(str, context.shape))
report("s123_slice", tensors["s123_slice"])
report("ctx_slice", tensors["ctx_slice"])
report("dec_out", pixels)

for name, tensor in tensors.items():
    require_finite(name, tensor)

# Storage dtype: probe INPUTS are float16 by construction (see `probe`), so storing them as f16 is
# lossless and halves the fixture. The full decode's OUTPUT is stored f16 too — 2.5 M elements is
# 10 MB in f32 — which bounds the golden's own precision at ~1e-4 absolute over a [-1, 1] signal;
# the parity gate sits an order of magnitude above that. Every other output stays f32.
F16_TENSORS = {"na_in", "diff_ctx", "diff_x", "dec_latent", "dec_noise", "dec_out"}
tensors = {
    name: (value.to(torch.float16) if name in F16_TENSORS else value.to(torch.float32)).contiguous()
    for name, value in tensors.items()
}
meta["storage"] = "float16: " + ", ".join(sorted(F16_TENSORS)) + "; float32: everything else"

out = fixture("mlx-gen-ltx/tests/fixtures/ltx25_diffvae_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out, metadata=meta)
print(f"wrote {out}")

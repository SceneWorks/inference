#!/usr/bin/env python
"""Dump an end-to-end + per-block golden for the Lens DiT (candle-gen sc-5112).

Runs the authoritative vendor `LensTransformer2DModel` (SceneWorks `_vendor/lens/transformer.py`) on
the cached `microsoft/Lens-Turbo` transformer weights, in **float32** (a tight, decisive correctness
gate for a 48-block DiT — bf16 cross-backend accumulation over 48 residual blocks would obscure
subtle bugs), over synthetic inputs. Records the full-forward output plus the block-0 inputs and
output so the Rust port can be checked both per-block and end-to-end.

Text features are synthetic (seeded) random tensors — this gate stands alone, independent of the
gpt-oss encoder slices (sc-5108/5110). The Rust side loads the same real transformer weights (cast to
f32) directly from the snapshot, so only the activations live in the golden.

Golden contents (all f32 unless noted):
  - inputs:  `hidden_states` [1, img_len, 128], `feat_{0..3}` [1, txt_len, 2880], `timestep` [1],
             `grid_fhw` [3] int64 = (frame, h_lat, w_lat);
  - block-0: `img_in_out` [1, img_len, 1536], `txt_in_out` [1, txt_len, 1536], `temb` [1, 1536],
             `block0_enc` / `block0_hidden` [1, *, 1536];
  - output:  `out` [1, img_len, 128] (full forward).

Run (from the worktree root) with the transformers-5.8 lens-venv:
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_dit_golden.py [out_dir]

Default out_dir: .scratch/lens-dit-goldens/  (not committed — large + regenerable).
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# The vendor DiT class. Prefer the main worktree checkout; fall back to the desktop python-src copy.
VENDOR_CANDIDATES = [
    r"D:\repos\SceneWorks\apps\worker\scene_worker\_vendor\lens\transformer.py",
    r"D:\repos\SceneWorks\apps\desktop\python-src\scene_worker\_vendor\lens\transformer.py",
    os.path.expanduser(
        r"~\AppData\Local\Programs\SceneWorks\python-src\scene_worker\_vendor\lens\transformer.py"
    ),
]

FRAME, H_LAT, W_LAT = 1, 16, 16
TXT_LEN = 120


def load_model_cls():
    vendor = next((p for p in VENDOR_CANDIDATES if os.path.exists(p)), None)
    if vendor is None:
        sys.exit(f"no vendor lens/transformer.py found; tried:\n  " + "\n  ".join(VENDOR_CANDIDATES))
    print(f"vendor transformer: {vendor}")
    spec = importlib.util.spec_from_file_location("lens_transformer", vendor)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensTransformer2DModel


def find_transformer() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*" / "transformer")))
    if not matches:
        sys.exit(f"no microsoft/Lens-Turbo transformer snapshot under {hub}")
    return matches[-1]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-dit-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)

    LensTransformer2DModel = load_model_cls()
    tdir = find_transformer()
    print(f"transformer: {tdir}\nloading (f32, CPU)…", flush=True)
    model = LensTransformer2DModel.from_pretrained(tdir, torch_dtype=torch.float32).to("cpu").eval()

    img_len = FRAME * H_LAT * W_LAT
    n_text = len(model.config.selected_layer_index)
    enc_dim = model.config.enc_hidden_dim
    print(f"img_len={img_len} txt_len={TXT_LEN} n_text={n_text} enc_dim={enc_dim}")

    torch.manual_seed(0)
    hidden_states = torch.randn(1, img_len, model.config.in_channels, dtype=torch.float32)
    feats = [torch.randn(1, TXT_LEN, enc_dim, dtype=torch.float32) for _ in range(n_text)]
    timestep = torch.rand(1, dtype=torch.float32)  # in [0, 1]
    text_mask = torch.ones(1, TXT_LEN, dtype=torch.bool)
    img_shapes = [(FRAME, H_LAT, W_LAT)]

    with torch.no_grad():
        # --- replay the model sub-modules to capture block-0 inputs ---
        img_in_out = model.img_in(hidden_states)
        normed = [model.txt_norm[i](feats[i]) for i in range(n_text)]
        txt_in_out = model.txt_in(torch.cat(normed, dim=-1))
        temb = model.time_text_embed(timestep, img_in_out)
        rope = model.pos_embed(img_shapes, [TXT_LEN], device=torch.device("cpu"))
        mask = model._build_joint_attention_mask(text_mask, img_len)
        block0_enc, block0_hidden = model.transformer_blocks[0](
            img_in_out, txt_in_out, temb, rope, mask
        )

        # --- full forward ---
        out = model(hidden_states, feats, text_mask, timestep, img_shapes)

    tensors = {
        "hidden_states": hidden_states.contiguous(),
        "timestep": timestep.contiguous(),
        "grid_fhw": torch.tensor([FRAME, H_LAT, W_LAT], dtype=torch.int64),
        "img_in_out": img_in_out.contiguous(),
        "txt_in_out": txt_in_out.contiguous(),
        "temb": temb.contiguous(),
        "block0_enc": block0_enc.contiguous(),
        "block0_hidden": block0_hidden.contiguous(),
        "out": out.contiguous(),
    }
    for i, f in enumerate(feats):
        tensors[f"feat_{i}"] = f.contiguous()

    dst = out_dir / "lens_dit_golden.safetensors"
    save_file(tensors, str(dst))
    print(
        f"wrote {dst}  (out={tuple(out.shape)}, "
        f"block0_hidden={tuple(block0_hidden.shape)}, block0_enc={tuple(block0_enc.shape)})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dump a TINY, randomly-initialised `MageFlow` NR-MMDiT and its f32 activations (sc-14040).

Why this exists alongside `dump_mage_flow_golden.py`:

The real-weights goldens are bf16, and the published checkpoint's block-0 modulation gates reach
~1e8, so the 12-block stack amplifies bf16 rounding to a **2e-2 mean-relative noise floor** (the
port's own f32-vs-bf16 spread is 2.8e-2). That floor is coarser than several real porting mistakes
— swapping `gelu-approximate` for a SwiGLU gate moves `dit_out` by only ~1.7e-2 — so the
real-weights gate alone cannot discriminate them.

This fixture removes the precision floor instead of arguing about it: a 2-block model at dim 24 in
**f32**, where the port matches the reference to ~1e-6 and every mistake is orders of magnitude
outside. Weights are `torch.manual_seed(0)` random, so the fixture is small enough to commit and
carries no licensed data. Same pattern as `mlx-gen-z-image`'s `dump_zblock_small.py`.

Two packings are dumped, because they exercise different code:

* `gen`  — the fused-CFG generation pack: two attention segments, one `img_shapes` entry each.
* `edit` — the edit pack: ONE attention segment carrying TWO `img_shapes` entries
           (`[target, ref]`, `pipeline.py:517-519`), which is the only configuration where the
           msrope **frame axis** changes the attention scores rather than cancelling.

Run from the reference venv (see `_vendor/VENDORED.md`):

    /tmp/mageflow-ref-venv/bin/python crates/media/mlx-gen/tools/dump_mage_flow_small.py

Writes `mlx-gen-mage/tests/fixtures/mage_flow_small.safetensors` (committed; ~200 KB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_vendor"))

from mage_flow.models.mage_flow import MageFlow, MageFlowParams  # noqa: E402
from mage_flow.models.modules._attn_backend import set_attn_backend  # noqa: E402

OUT = HERE.parent / "mlx-gen-mage" / "tests" / "fixtures" / "mage_flow_small.safetensors"

# Tiny but structurally faithful: sum(axes_dim) == head_dim == hidden_size / num_heads.
PARAMS = MageFlowParams(
    in_channels=4,
    out_channels=4,
    context_in_dim=6,
    hidden_size=24,
    num_heads=2,
    depth=2,
    axes_dim=[4, 4, 4],
    checkpoint=False,
    patch_size=1,
)

# (name, img_shapes, img_cu, txt_cu, sigmas)
CASES = [
    # Fused-CFG generation: cond + uncond, one latent grid each. The uncond copy is `img_shapes`
    # entry 1, so it rotates at msrope frame index 1 (`pipeline.py:167`).
    ("gen", [(1, 3, 4), (1, 3, 4)], [0, 12, 24], [0, 5, 8], [0.7, 0.7]),
    # Edit: ONE attention segment holding [target, ref_1, ref_2, ref_3] — the reference resizes
    # every ref to the target resolution (`pipeline.py:501`), so all four grids match and only the
    # frame index distinguishes them, all inside the same attention window.
    ("edit", [(1, 2, 3)] * 4, [0, 24], [0, 4], [0.35]),
]


def main() -> None:
    set_attn_backend("sdpa")
    torch.manual_seed(0)
    model = MageFlow(PARAMS).to(torch.float32).eval()
    # Default nn.Linear init leaves several tensors near zero; re-randomise so every weight is
    # load-bearing and a dropped tensor cannot pass by accident.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.2)

    tensors: dict[str, torch.Tensor] = {
        f"model.{k}": v.detach().contiguous() for k, v in model.state_dict().items()
    }
    tensors["config"] = torch.tensor(
        [
            PARAMS.in_channels,
            PARAMS.out_channels,
            PARAMS.context_in_dim,
            PARAMS.hidden_size,
            PARAMS.num_heads,
            PARAMS.depth,
            PARAMS.patch_size,
            *PARAMS.axes_dim,
        ],
        dtype=torch.int32,
    )

    for name, shapes, img_cu, txt_cu, sigmas in CASES:
        n_img = img_cu[-1]
        n_txt = txt_cu[-1]
        assert sum(f * h * w for f, h, w in shapes) == n_img, name
        assert len(sigmas) == len(img_cu) - 1 == len(txt_cu) - 1, name

        torch.manual_seed(1234 + len(name))
        img = torch.randn(1, n_img, PARAMS.in_channels)
        txt = torch.randn(1, n_txt, PARAMS.context_in_dim)
        timesteps = torch.tensor(sigmas, dtype=torch.float32)
        img_cu_t = torch.tensor(img_cu, dtype=torch.int32)
        txt_cu_t = torch.tensor(txt_cu, dtype=torch.int32)

        block_io: dict[str, torch.Tensor] = {}

        def hook(_m, _args, kwargs, output):
            if f"{name}.block0_in.img" in block_io:
                return
            block_io[f"{name}.block0_in.img"] = kwargs["hidden_states"].detach().clone()
            block_io[f"{name}.block0_in.txt"] = kwargs["encoder_hidden_states"].detach().clone()
            block_io[f"{name}.block0_in.temb"] = kwargs["temb"].detach().clone()
            rope = kwargs["image_rotary_emb"]
            block_io[f"{name}.rope_re"] = rope.real.detach().float().contiguous()
            block_io[f"{name}.rope_im"] = rope.imag.detach().float().contiguous()
            block_io[f"{name}.block0_out.txt"] = output[0].detach().clone()
            block_io[f"{name}.block0_out.img"] = output[1].detach().clone()

        handle = model.transformer_blocks[0].register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                out = model(
                    img=img,
                    txt=txt,
                    timesteps=timesteps,
                    img_shapes=[list(shapes)],
                    img_cu_seqlens=img_cu_t,
                    txt_cu_seqlens=txt_cu_t,
                )
        finally:
            handle.remove()

        tensors[f"{name}.in.img"] = img
        tensors[f"{name}.in.txt"] = txt
        tensors[f"{name}.in.timesteps"] = timesteps
        tensors[f"{name}.in.img_cu"] = img_cu_t
        tensors[f"{name}.in.txt_cu"] = txt_cu_t
        tensors[f"{name}.in.img_shapes"] = torch.tensor(shapes, dtype=torch.int32)
        tensors[f"{name}.out"] = out.detach()
        tensors.update(block_io)
        print(f"{name}: out {tuple(out.shape)} std {out.std().item():.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(OUT),
        metadata={
            "source": "microsoft/Mage @ _vendor/mage_flow (MIT) — randomly initialised, f32",
            "seed": "0",
            "story": "sc-14040",
        },
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()

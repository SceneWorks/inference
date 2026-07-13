"""sc-5984 — convert the official `ideogram-ai/ideogram-4-fp8` checkpoint to the
`mlx-gen-ideogram` MLX safetensors layout (zero runtime Python; one-time offline
provisioning tool, same convention as the other `tools/convert_*.py`).

Ideogram 4 ships **fp8 weight-only** weights: every Linear stores its weight as
`float8_e4m3fn` plus a per-output-row F32 `<key>_scale`, with biases / norms in bf16.
This tool dequantizes `w_bf16 = (fp8.float() * weight_scale[:, None]).bfloat16()`, drops
the now-folded `*_scale` tensors, and re-emits each component as an MLX-loadable bf16
safetensors. The DiT/TE key names are preserved as-is (already clean: `layers.{i}.
attention.qkv.weight`, …); the Rust loaders (sc-5985/86/87) own the final module mapping.

Pipeline components (from `model_index.json` = `Ideogram4Pipeline`):
  * transformer               Ideogram4Transformer2DModel  (fp8, 669 tensors)
  * unconditional_transformer Ideogram4Transformer2DModel  (fp8, 669 tensors) — asymmetric CFG
  * text_encoder              Qwen3VLModel                 (fp8, ideogram_fp8_weight_only)
  * vae                       AutoencoderKLFlux2           (bf16 already; passthrough)

The VAE is the FLUX.2 VAE (`AutoencoderKLFlux2`); conv-weight layout alignment to the
`mlx-gen-flux2` loader is handled in E4 (sc-5987). Here it is a dtype-preserving passthrough.

Run (torch venv; the gated fp8 repo must already be in the HF cache):
  ~/mlx-flux-venv/bin/python tools/convert_ideogram4_to_mlx.py \
      --output ~/.cache/ideogram4-mlx-convert
  # fast logic/parity check without writing ~53 GB:
  ~/mlx-flux-venv/bin/python tools/convert_ideogram4_to_mlx.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open

# Components that carry weights to convert. (scheduler / tokenizer are copied verbatim.)
WEIGHT_COMPONENTS = (
    "transformer",
    "unconditional_transformer",
    "text_encoder",
    "vae",
)
COPY_COMPONENTS = ("scheduler", "tokenizer")
COPY_FILES = ("model_index.json", "README.md", "LICENSE.md")

# fp8 dtype the checkpoint stores Linear weights in.
FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


def default_fp8_snapshot() -> Path:
    """Resolve the cached gated fp8 snapshot dir."""
    base = Path.home() / ".cache/huggingface/hub/models--ideogram-ai--ideogram-4-fp8/snapshots"
    snaps = sorted(p for p in base.glob("*") if p.is_dir())
    if not snaps:
        sys.exit(f"fp8 snapshot not found under {base} — accept the gate + download first.")
    return snaps[-1]


def component_shards(comp_dir: Path) -> list[Path]:
    return sorted(comp_dir.glob("*.safetensors"))


def to_mx_passthrough(t: torch.Tensor) -> mx.array:
    """torch tensor -> MLX array. Floating tensors land in bf16 (via lossless f32 staging —
    numpy has no bf16; bf16 -> f32 -> bf16 is exact). Integer buffers (e.g. the VAE
    `bn.num_batches_tracked`) keep an integer dtype rather than being corrupted to bf16."""
    t = t.detach()
    if t.dtype.is_floating_point:
        return mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)
    return mx.array(t.to(torch.int32).numpy())


def convert_component(comp: str, src_dir: Path, out_dir: Path, *, dry_run: bool) -> dict:
    """Dequant+passthrough one component. Returns a stats dict."""
    shards = component_shards(src_dir)
    if not shards:
        sys.exit(f"{comp}: no .safetensors in {src_dir}")

    out: dict[str, mx.array] = {}
    n_src = n_fp8 = n_scale = n_passthrough = 0
    dtypes: dict[str, int] = {}

    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            n_src += len(keys)
            for k in keys:
                t = f.get_tensor(k)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                if k.endswith("_scale"):
                    n_scale += 1
                    continue  # folded into its weight below
                if t.dtype in FP8_DTYPES:
                    scale_key = k + "_scale"
                    if scale_key not in keys:
                        sys.exit(f"{comp}: fp8 tensor {k} has no sibling {scale_key}")
                    scale = f.get_tensor(scale_key).to(torch.float32)  # [out]
                    if scale.ndim != 1 or scale.shape[0] != t.shape[0]:
                        sys.exit(
                            f"{comp}: {k} scale shape {tuple(scale.shape)} "
                            f"!= per-row of weight {tuple(t.shape)}"
                        )
                    n_fp8 += 1
                    if dry_run:
                        continue
                    w = t.to(torch.float32) * scale.reshape(-1, *([1] * (t.ndim - 1)))
                    out[k] = mx.array(w.numpy()).astype(mx.bfloat16)
                else:
                    n_passthrough += 1
                    if dry_run:
                        continue
                    out[k] = to_mx_passthrough(t)

    n_out = n_fp8 + n_passthrough  # one output tensor per non-scale source tensor
    expect_out = n_src - n_scale
    if n_out != expect_out:
        sys.exit(f"{comp}: round-trip mismatch out={n_out} expected={expect_out}")

    if not dry_run:
        comp_out = out_dir / comp
        comp_out.mkdir(parents=True, exist_ok=True)
        cfg = src_dir / "config.json"
        if cfg.exists():
            shutil.copy2(cfg, comp_out / "config.json")
        mx.save_safetensors(str(comp_out / "model.safetensors"), out)
        out.clear()  # free ~18 GB before the next component

    return {
        "src_tensors": n_src,
        "fp8_dequant": n_fp8,
        "scales_folded": n_scale,
        "passthrough": n_passthrough,
        "out_tensors": n_out,
        "dtypes": dtypes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None, help="fp8 snapshot dir (default: HF cache)")
    ap.add_argument("--output", type=Path, default=Path.home() / ".cache/ideogram4-mlx-convert")
    ap.add_argument("--component", choices=WEIGHT_COMPONENTS, default=None, help="convert only this one")
    ap.add_argument("--dry-run", action="store_true", help="validate fp8/scale pairing + counts, write nothing")
    args = ap.parse_args()

    snap = args.snapshot or default_fp8_snapshot()
    print(f"source : {snap}")
    print(f"output : {args.output}  {'(DRY RUN — no writes)' if args.dry_run else ''}")

    comps = [args.component] if args.component else list(WEIGHT_COMPONENTS)
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    grand = {"fp8_dequant": 0, "scales_folded": 0, "out_tensors": 0}
    for comp in comps:
        stats = convert_component(comp, snap / comp, args.output, dry_run=args.dry_run)
        print(f"\n[{comp}]")
        print(f"  src tensors   : {stats['src_tensors']}")
        print(f"  fp8 dequanted : {stats['fp8_dequant']}")
        print(f"  scales folded : {stats['scales_folded']}")
        print(f"  passthrough   : {stats['passthrough']}")
        print(f"  out tensors   : {stats['out_tensors']}  (= src - scales ✓)")
        print(f"  src dtypes    : {stats['dtypes']}")
        for kk in grand:
            grand[kk] += stats[kk]

    # Copy verbatim components/files (only on a full, non-dry run).
    if not args.dry_run and not args.component:
        for comp in COPY_COMPONENTS:
            srcc = snap / comp
            if srcc.exists():
                shutil.copytree(srcc, args.output / comp, dirs_exist_ok=True)
        for fn in COPY_FILES:
            if (snap / fn).exists():
                shutil.copy2(snap / fn, args.output / fn)

    print(f"\nTOTAL fp8 dequanted={grand['fp8_dequant']}  scales folded={grand['scales_folded']}  "
          f"out tensors={grand['out_tensors']}")
    print("OK" + ("" if not args.dry_run else " (dry run)"))


if __name__ == "__main__":
    main()

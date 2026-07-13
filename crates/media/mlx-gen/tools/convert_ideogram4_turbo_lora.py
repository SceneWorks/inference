"""issue #488 — package the ostris **TurboTime** LoRA into the `mlx-gen-ideogram` turbo snapshot.

The few-step CFG-free turbo path loads the base Ideogram 4 conditional DiT and installs this LoRA
(`ideogram_4_turbotime_v1.safetensors`, rank 128, BF16, ai-toolkit) at scale 1.0. The key
convention is `diffusion_model.<path>.lora_{down,up}.weight` over the 6 per-layer modules only
(`attention.qkv`/`o`, `feed_forward.w{1,2,3}`, `adaln_modulation`) across all 34 layers — a 1:1 map
onto the Rust `AdaptableHost for Ideogram4Transformer` surface (the prefix is stripped by the shared
loader). The MLX safetensors reader loads the BF16 tensors directly, so this is a validate + dtype-
normalize + rename pass, NOT a dequant like `convert_ideogram4_to_mlx.py`.

CRITICAL — the LoRA carries **no** `.alpha` tensor and **no** `lora_adapter_metadata` blob, so the
shared loader skips the `alpha/rank` fold and applies the residual at exactly the AdapterSpec scale
(1.0) — byte-faithful to the issue-#488 spike's static merge. This tool therefore drops ALL
metadata and refuses to emit any `.alpha` tensor, so that property cannot silently drift.

Run (torch venv; pass a local LoRA file, or --download from the gated-free ostris repo):
  ~/mlx-flux-venv/bin/python tools/convert_ideogram4_turbo_lora.py \
      --lora ~/Downloads/ideogram_4_turbotime_v1.safetensors \
      --output ~/.cache/ideogram4-mlx-turbo            # writes <output>/turbo_lora.safetensors
  # download + validate only, write nothing:
  ~/mlx-flux-venv/bin/python tools/convert_ideogram4_turbo_lora.py --download --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

LORA_REPO = "ostris/ideogram_4_turbotime_lora"
LORA_FILE = "ideogram_4_turbotime_v1.safetensors"
OUT_NAME = "turbo_lora.safetensors"

PREFIX = "diffusion_model."
NUM_LAYERS = 34
# The 6 per-layer TurboTime target modules (must match `AdaptableHost for Ideogram4Transformer`).
LEAVES = (
    "attention.qkv",
    "attention.o",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "adaln_modulation",
)
EXPECT_MODULES = NUM_LAYERS * len(LEAVES)  # 204
RANK = 128

# down/up factor suffixes the shared Rust loader accepts (PEFT `lora_A/B`, ai-toolkit `lora_down/up`).
DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")


def split_key(key: str) -> tuple[str, str]:
    """`diffusion_model.<module>.lora_{down,up}.weight` -> (`<module>`, 'down'|'up'). Exits on any
    unexpected key (a stray `.alpha`, a non-`diffusion_model.` prefix, an unknown suffix)."""
    if not key.startswith(PREFIX):
        sys.exit(f"unexpected LoRA key prefix (want '{PREFIX}*'): {key}")
    rest = key[len(PREFIX) :]
    for suf in DOWN_SUFFIXES:
        if rest.endswith(suf):
            return rest[: -len(suf)], "down"
    for suf in UP_SUFFIXES:
        if rest.endswith(suf):
            return rest[: -len(suf)], "up"
    if rest.endswith(".alpha"):
        sys.exit(
            f"LoRA carries an `.alpha` tensor ({key}); TurboTime is expected to ship none "
            "(injecting alpha would fold alpha/rank and change the validated scale-1.0 behavior)"
        )
    sys.exit(f"unexpected LoRA key suffix (want lora_down/up or lora_A/B): {key}")


def validate_module(module: str) -> None:
    """`<module>` must be `layers.{i}.{leaf}` with `i in 0..34` and `leaf` one of the 6 targets."""
    if not module.startswith("layers."):
        sys.exit(f"off-surface LoRA target (want `layers.{{i}}.<leaf>`): {module}")
    rest = module[len("layers.") :]
    idx, _, leaf = rest.partition(".")
    if not idx.isdigit() or not (0 <= int(idx) < NUM_LAYERS):
        sys.exit(f"layer index out of range 0..{NUM_LAYERS}: {module}")
    if leaf not in LEAVES:
        sys.exit(f"off-surface LoRA target module `{leaf}` (want one of {LEAVES}): {module}")


def resolve_lora(args: argparse.Namespace) -> Path:
    if args.lora is not None:
        if not args.lora.exists():
            sys.exit(f"--lora not found: {args.lora}")
        return args.lora
    if not args.download:
        sys.exit("provide --lora <path> or --download to fetch from the ostris repo")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(LORA_REPO, LORA_FILE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, default=None, help="local ostris TurboTime safetensors")
    ap.add_argument("--download", action="store_true", help=f"fetch {LORA_FILE} from {LORA_REPO}")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".cache/ideogram4-mlx-turbo",
        help="turbo snapshot dir (writes <dir>/turbo_lora.safetensors) or an explicit .safetensors path",
    )
    ap.add_argument("--dry-run", action="store_true", help="validate keys/ranks/counts, write nothing")
    args = ap.parse_args()

    src = resolve_lora(args)
    print(f"reading {src}")

    out: dict[str, mx.array] = {}
    seen: dict[str, set[str]] = {}
    ranks: set[int] = set()
    n_src = 0
    with safe_open(src, framework="pt") as f:
        for key in f.keys():
            n_src += 1
            module, role = split_key(key)
            validate_module(module)
            seen.setdefault(module, set()).add(role)
            t = f.get_tensor(key).detach()
            # down: [r, in] -> r = shape[0]; up: [out, r] -> r = shape[1].
            ranks.add(t.shape[0] if role == "down" else t.shape[1])
            if not args.dry_run:
                # bf16 -> f32 -> bf16 is exact (numpy has no bf16); matches convert_ideogram4_to_mlx.
                out[key] = mx.array(t.float().numpy()).astype(mx.bfloat16)

    # ── Validate the surface (loud, no silent drop) ──
    incomplete = {m: r for m, r in seen.items() if r != {"down", "up"}}
    if incomplete:
        sys.exit(f"modules missing a down/up half: {incomplete}")
    if len(seen) != EXPECT_MODULES:
        sys.exit(f"expected {EXPECT_MODULES} modules (6 x {NUM_LAYERS}), found {len(seen)}")
    if ranks != {RANK}:
        sys.exit(f"expected rank {RANK} throughout, found ranks {sorted(ranks)}")
    print(
        f"validated: {n_src} tensors -> {len(seen)} modules x (down+up), rank {RANK}, "
        f"all targets on the 6-module x {NUM_LAYERS}-layer TurboTime surface"
    )

    if args.dry_run:
        print("dry-run: wrote nothing")
        return

    out_path = args.output if args.output.suffix == ".safetensors" else args.output / OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # No metadata: keeps the scale-1.0 (no alpha/rank fold) property unambiguous.
    mx.save_safetensors(str(out_path), out)
    print(f"wrote {len(out)} bf16 tensors -> {out_path}")


if __name__ == "__main__":
    main()

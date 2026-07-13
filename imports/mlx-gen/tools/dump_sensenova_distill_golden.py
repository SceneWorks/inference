#!/usr/bin/env python3
"""sc-3192: synthetic golden for the 8-step distill-LoRA merge.

Replays the **genuine** reference merge — `sensenova_u1/utils/lora.py::load_and_merge_lora_weight`
(imported directly by file path, no heavy package `__init__`) — on a handful of tiny base weights +
a tiny LoRA, and dumps `(base, lora factors, merged)` for each target so the Rust `distill_parity`
test can prove `lora_delta` + merge is bit-exact against the reference arithmetic
(`value += (alpha/rank)·(up @ down)`, f32).

Targets mirror the three real distill-LoRA shapes: a generation-path attention projection, a
generation-path SwiGLU projection, and an FM-head Linear. One target uses `alpha != rank` so the
test catches any code that hardcodes scale 1.0 instead of reading `alpha/rank`.

Run under the reference venv (torch):
    _vendor/sensenova_u1/.venv/bin/python tools/dump_sensenova_distill_golden.py
Writes mlx-gen-sensenova/tests/fixtures/distill_golden.safetensors.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
# `_vendor` lives in the main checkout (it is not copied into git worktrees); `$SENSENOVA_LORA_PY`
# overrides the path to the reference `lora.py` so this tool can run from a worktree.
LORA_PY = Path(
    os.environ.get(
        "SENSENOVA_LORA_PY", REPO / "_vendor/sensenova_u1/src/sensenova_u1/utils/lora.py"
    )
)
OUT = REPO / "mlx-gen-sensenova/tests/fixtures/distill_golden.safetensors"


def _load_reference_merge():
    """Import `load_and_merge_lora_weight` from the vendored file directly (skips package init)."""
    spec = importlib.util.spec_from_file_location("_ref_lora", LORA_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_and_merge_lora_weight


class _Leaf(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(w, requires_grad=False)


def _build_module(base: dict[str, torch.Tensor]) -> nn.Module:
    """Build an nn.Module whose `named_parameters()` are exactly the `<target>.weight` keys.

    The reference walks `named_parameters()` and derives LoRA names by string-replacing `.weight`,
    so reproducing the dotted names (including numeric segments like `layers.0`, `fm_head.0`) is all
    that's needed to drive the genuine merge.
    """
    root = nn.Module()
    for key, w in base.items():
        assert key.endswith(".weight")
        parts = key[: -len(".weight")].split(".")
        parent = root
        for p in parts[:-1]:
            if p not in parent._modules:
                parent.add_module(p, nn.Module())
            parent = parent._modules[p]
        parent.add_module(parts[-1], _Leaf(w))
    return root


def main() -> None:
    torch.manual_seed(3192)
    load_and_merge_lora_weight = _load_reference_merge()

    # (target, out, in, rank, alpha). `alpha != rank` on the fm_head target → scale 2.0.
    targets = [
        ("language_model.model.layers.0.self_attn.q_proj_mot_gen", 16, 8, 4, 4),
        ("language_model.model.layers.0.mlp_mot_gen.gate_proj", 12, 8, 4, 4),
        ("fm_modules.fm_head.0", 8, 8, 4, 8),
    ]

    base: dict[str, torch.Tensor] = {}
    lora: dict[str, torch.Tensor] = {}
    for tgt, out, inn, rank, alpha in targets:
        base[f"{tgt}.weight"] = torch.randn(out, inn, dtype=torch.float32)
        lora[f"{tgt}.lora_down.weight"] = torch.randn(rank, inn, dtype=torch.float32)
        lora[f"{tgt}.lora_up.weight"] = torch.randn(out, rank, dtype=torch.float32)
        lora[f"{tgt}.alpha"] = torch.tensor(alpha, dtype=torch.int32)

    model = _build_module({k: v.clone() for k, v in base.items()})
    load_and_merge_lora_weight(model, lora)
    merged = dict(model.named_parameters())

    out: dict[str, torch.Tensor] = {}
    for tgt, *_ in targets:
        out[f"__base__.{tgt}"] = base[f"{tgt}.weight"].contiguous()
        out[f"__merged__.{tgt}"] = merged[f"{tgt}.weight"].detach().contiguous()
        out[f"{tgt}.lora_down.weight"] = lora[f"{tgt}.lora_down.weight"].contiguous()
        out[f"{tgt}.lora_up.weight"] = lora[f"{tgt}.lora_up.weight"].contiguous()
        out[f"{tgt}.alpha"] = lora[f"{tgt}.alpha"].contiguous()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))
    print(f"[saved] {OUT}  ({len(targets)} targets)")
    for tgt, *_ in targets:
        d = (out[f"__merged__.{tgt}"] - out[f"__base__.{tgt}"]).abs().max().item()
        print(f"  {tgt}: max|merged-base|={d:.4f}")


if __name__ == "__main__":
    main()

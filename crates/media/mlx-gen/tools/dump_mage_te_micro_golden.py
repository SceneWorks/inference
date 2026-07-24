#!/usr/bin/env python3
"""Toy-dimension **arithmetic** golden for the Mage-Flow Qwen3-VL text encoder (sc-14038).

Unlike the boundary goldens under ``tools/golden/`` (real 4.1B weights, multi-GB, gitignored,
``#[ignore]``d consumers), this dumps a *tiny* Qwen3-VL text model — random but seeded weights at
toy dimensions — so the fixture is a few tens of KB and can be **committed** and asserted by the
default ``cargo test`` lane. That closes the gap the boundary golden cannot: the weights-free tests
otherwise check topology (shapes, layer counts, isolation) but carry no numeric oracle for the
attention composition, so a regression in GQA ``repeat_kv`` grouping, QK-norm placement, the SwiGLU
gate/up order or ``o_proj`` would stay green on the macOS lane.

The oracle is the real ``transformers.Qwen3VLTextModel`` — the same class the vendored reference
patches — so the fixture gates this port against upstream arithmetic, not against a hand-rolled
re-derivation of it.

Usage (needs torch + transformers; no model weights):

    python3 tools/dump_mage_te_micro_golden.py

Writes ``mlx-gen-mage/tests/fixtures/te_micro_golden.safetensors``. Deterministic: re-running
reproduces the file byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

OUT = (
    Path(__file__).resolve().parents[1]
    / "mlx-gen-mage"
    / "tests"
    / "fixtures"
    / "te_micro_golden.safetensors"
)

SEED = 14038
# Toy dims that keep every production STRUCTURE while staying tiny:
#   * GQA with a real group size (4 query heads / 2 kv heads -> groups = 2), so a `repeat_kv`
#     that tiled instead of interleaving would diverge;
#   * head_dim DECOUPLED from hidden/heads (4 x 8 = 32 != 16), as in production (32 x 128 != 2560);
#   * interleaved M-RoPE with all three sections populated ([2,1,1] over head_dim/2 = 4);
#   * intermediate != hidden so a gate/up/down mix-up cannot cancel.
CFG = dict(
    hidden_size=16,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=12,
    vocab_size=24,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    rms_norm_eps=1e-6,
    rope_theta=5_000_000.0,
    max_position_embeddings=4096,
    tie_word_embeddings=True,
    use_cache=False,
    rope_scaling={
        "rope_type": "default",
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
    },
)

# A single unpadded sequence, as the reference always feeds (it packs, never pads).
INPUT_IDS = [3, 7, 1, 12, 5, 0, 19, 23, 11, 2]


def main() -> None:
    torch.manual_seed(SEED)
    config = Qwen3VLTextConfig(**CFG)
    # `eager` is the canonical reference path — no kernel-selection variance in the oracle.
    config._attn_implementation = "eager"
    model = Qwen3VLTextModel(config).eval().to(torch.float32)

    # Deterministic, non-degenerate parameters, assigned in a fixed `named_parameters()` order so
    # the dump is reproducible.
    #
    # Norm scales are centred on 1.0 with a WIDE spread (0.5, not a token 0.1). Trained Qwen3
    # RMSNorm scales genuinely vary by that much, and it matters for the test's power: with a narrow
    # spread every norm weight is ≈1.0, so swapping `q_norm` with `k_norm` is a near-no-op and the
    # oracle cannot discriminate the QK-RMSNorm assignment from ordinary kernel noise.
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.copy_(1.0 + 0.5 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.25 * torch.randn(p.shape, generator=gen))

    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    hidden = out.last_hidden_state.squeeze(0).contiguous()  # [L, hidden], post-final-norm

    tensors = {f"lm.{k}": v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    tensors["io.input_ids"] = torch.tensor(INPUT_IDS, dtype=torch.int32)
    tensors["io.last_hidden_state"] = hidden

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(OUT),
        metadata={
            "source": "transformers.Qwen3VLTextModel (eager, f32), random seeded weights",
            "seed": str(SEED),
            "config": json.dumps(CFG, sort_keys=True),
            "note": (
                "Toy-dimension ARITHMETIC oracle for mlx-gen-mage's Qwen3-VL text encoder. "
                "Committed on purpose (a few tens of KB) so the default cargo test lane gates the "
                "attention composition, not just topology. Regenerate with "
                "tools/dump_mage_te_micro_golden.py."
            ),
        },
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  tensors: {len(tensors)}  hidden: {tuple(hidden.shape)}")
    print(f"  |hidden| max {hidden.abs().max():.6f}  mean {hidden.abs().mean():.6f}")


if __name__ == "__main__":
    main()

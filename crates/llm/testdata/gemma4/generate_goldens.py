#!/usr/bin/env python3
"""Generate the Gemma-4 shared primitive goldens (`gemma4_goldens.json`).

sc-18769. The oracle is transcribed from the public reference implementation
(`huggingface/transformers`, `src/transformers/models/gemma4_unified/`):

* `configuration_gemma4_unified.py` — the `layer_types` default schedule and the
  per-layer-type `rope_parameters` / `per_layer_config` overrides.
* `modeling_rope_utils.py::_compute_proportional_rope_parameters` — the
  `rope_type: "proportional"` inverse-frequency schedule.
* `modeling_gemma4_unified.py` — `rotate_half` / `apply_rotary_pos_emb`,
  `Gemma4UnifiedRMSNorm` (plain `weight`, NOT Gemma-2's `1 + weight`), and
  `Gemma4UnifiedTextAttention` (the `attention_k_eq_v` key/value sharing plus the
  scale-free `v_norm`).
* `masking_utils` sliding-window semantics — key `j` is visible to query `q` iff
  `0 <= q - j < sliding_window`.

Only numpy is required (no torch, no weights). Inputs are committed alongside the
expected outputs so both `mlx-llm` and `candle-llm` assert against the same numbers.

Regenerate with:

    python3 crates/llm/testdata/gemma4/generate_goldens.py

and write the result to a temporary file first — never redirect the generator into
its own checked-in output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- the measured google/gemma-4-12B-it text config (the LTX-2.5 packed TE's `gemma_config`) ---
REAL = {
    "hidden_size": 3840,
    "intermediate_size": 15360,
    "num_hidden_layers": 48,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "vocab_size": 262144,
    "sliding_window": 1024,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
    "sliding_theta": 10_000.0,
    "full_theta": 1_000_000.0,
    "full_partial_rotary_factor": 0.25,
}

SLIDING_WINDOW_PATTERN = 6


def layer_types(num_layers: int, pattern: int = SLIDING_WINDOW_PATTERN) -> list[str]:
    """`Gemma4UnifiedTextConfig.__post_init__`: every `pattern`-th layer is full attention.

    Upstream writes ``"sliding_attention" if bool((i + 1) % pattern) else "full_attention"``, then
    forces the *last* layer to ``full_attention``.
    """
    types = [
        "sliding_attention" if bool((i + 1) % pattern) else "full_attention"
        for i in range(num_layers)
    ]
    if types and types[-1] != "full_attention":
        types[-1] = "full_attention"
    return types


def default_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """`compute_default_rope_parameters`: `inv_freq[i] = theta ** (-2i / head_dim)`."""
    return 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))


def proportional_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float) -> np.ndarray:
    """`_compute_proportional_rope_parameters`.

    `rope_angles = int(partial_rotary_factor * head_dim // 2)` channels carry the ordinary
    `theta ** (-2i / head_dim)` frequency — note the exponent denominator stays the FULL
    `head_dim`, unlike a leading-slice partial RoPE — and the remaining `head_dim // 2 -
    rope_angles` channels are **zero**, so the table is always `head_dim` wide and the
    un-rotated channels are an exact identity rotation.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    rotated = 1.0 / (
        theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim)
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        return np.concatenate([rotated, np.zeros(nope_angles, dtype=np.float64)])
    return rotated


def cos_sin(inv_freq: np.ndarray, positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """`Gemma4UnifiedTextRotaryEmbedding.forward`: `emb = cat(freqs, freqs)` (NeoX half-split)."""
    freqs = np.outer(np.asarray(positions, dtype=np.float64), inv_freq)  # [L, dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [L, dim]
    return np.cos(emb), np.sin(emb)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """`apply_rotary_pos_emb` with `unsqueeze_dim=2` over `x` `[b, s, h, head_dim]`."""
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return x * c + rotated * s


def sliding_allowed(q_len: int, k_len: int, window: int) -> list[list[bool]]:
    """Bottom-right-aligned sliding-window causal visibility.

    The `q_len` queries sit at absolute positions `offset + r` with `offset = k_len - q_len`
    (the cached-decode convention every mask in these crates uses). Key `j` is visible iff
    `0 <= (offset + r) - j < window`.
    """
    offset = k_len - q_len
    return [
        [0 <= (offset + r) - j < window for j in range(k_len)]
        for r in range(q_len)
    ]


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    """`Gemma4UnifiedRMSNorm`: `x * (mean(x**2) + eps) ** -0.5`, then `* weight` when scaled.

    Gemma 4 multiplies by the **stored** weight. It does NOT use Gemma-2's `(1 + weight)` fold.
    `weight is None` is the scale-free `v_norm` (`with_scale=False`).
    """
    mean_squared = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps
    normed = x.astype(np.float64) * np.power(mean_squared, -0.5)
    if weight is not None:
        normed = normed * weight.astype(np.float64)
    return normed


def flat(a: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def build() -> dict:
    rng = np.random.default_rng(18769)

    types = layer_types(REAL["num_hidden_layers"])
    full_indices = [i for i, t in enumerate(types) if t == "full_attention"]

    # --- RoPE: the real per-layer-type tables (inverse frequencies only) ---
    sliding_real = default_inv_freq(REAL["head_dim"], REAL["sliding_theta"])
    full_real = proportional_inv_freq(
        REAL["global_head_dim"], REAL["full_theta"], REAL["full_partial_rotary_factor"]
    )

    # --- RoPE: small synthetic configs, both layer types, with rotated tensors ---
    positions = [0, 1, 2, 5, 11]
    small = {}

    sm_hd, sm_theta = 8, 10_000.0
    sm_inv = default_inv_freq(sm_hd, sm_theta)
    sm_cos, sm_sin = cos_sin(sm_inv, positions)
    sm_x = rng.standard_normal((1, len(positions), 3, sm_hd))
    small["sliding"] = {
        "rope_type": "default",
        "head_dim": sm_hd,
        "theta": sm_theta,
        "inv_freq": flat(sm_inv),
        "positions": positions,
        "cos": flat(sm_cos),
        "sin": flat(sm_sin),
        "x_shape": list(sm_x.shape),
        "x": flat(sm_x),
        "rotated": flat(apply_rope(sm_x, sm_cos, sm_sin)),
    }

    fu_hd, fu_theta, fu_p = 16, 1_000_000.0, 0.25
    fu_inv = proportional_inv_freq(fu_hd, fu_theta, fu_p)
    fu_cos, fu_sin = cos_sin(fu_inv, positions)
    fu_x = rng.standard_normal((1, len(positions), 2, fu_hd))
    small["full"] = {
        "rope_type": "proportional",
        "head_dim": fu_hd,
        "theta": fu_theta,
        "partial_rotary_factor": fu_p,
        "rope_angles": int(fu_p * fu_hd // 2),
        "inv_freq": flat(fu_inv),
        "positions": positions,
        "cos": flat(fu_cos),
        "sin": flat(fu_sin),
        "x_shape": list(fu_x.shape),
        "x": flat(fu_x),
        "rotated": flat(apply_rope(fu_x, fu_cos, fu_sin)),
    }

    # --- sliding-window mask ---
    masks = []
    for q_len, k_len, window in [(6, 6, 3), (1, 9, 4), (4, 10, 3), (5, 5, 1), (3, 7, 32)]:
        masks.append(
            {
                "q_len": q_len,
                "k_len": k_len,
                "window": window,
                "allowed": sliding_allowed(q_len, k_len, window),
            }
        )

    # --- k_eq_v projection (+ the k_norm / scale-free v_norm that consume it) ---
    hidden, kv_heads, head_dim, seq = 6, 2, 4, 3
    eps = 1e-6
    x = rng.standard_normal((1, seq, hidden))
    w_k = rng.standard_normal((kv_heads * head_dim, hidden))
    w_v = rng.standard_normal((kv_heads * head_dim, hidden))
    k_norm_w = rng.standard_normal(head_dim) * 0.1 + 1.0

    raw_k = (x @ w_k.T).reshape(1, seq, kv_heads, head_dim)
    raw_v_separate = (x @ w_v.T).reshape(1, seq, kv_heads, head_dim)
    k_eq_v = {
        "hidden_size": hidden,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "seq": seq,
        "rms_norm_eps": eps,
        "x": flat(x),
        "w_k": flat(w_k),
        "w_v": flat(w_v),
        "k_norm_weight": flat(k_norm_w),
        "raw_k": flat(raw_k),
        # attention_k_eq_v: true — V reuses the *raw* K projection output, then takes the
        # scale-free v_norm. K takes the scaled k_norm. Neither is the other.
        "k_shared": flat(rms_norm(raw_k, k_norm_w, eps)),
        "v_shared": flat(rms_norm(raw_k, None, eps)),
        # attention_k_eq_v: false — V comes from its own projection.
        "raw_v_separate": flat(raw_v_separate),
        "v_separate": flat(rms_norm(raw_v_separate, None, eps)),
    }

    return {
        "_source": "huggingface/transformers gemma4_unified + Lightricks/LTX-2 gemma_config",
        "_generator": "crates/llm/testdata/gemma4/generate_goldens.py",
        "config": REAL,
        "layer_types": {
            "num_hidden_layers": REAL["num_hidden_layers"],
            "sliding_window_pattern": SLIDING_WINDOW_PATTERN,
            "types": types,
            "num_sliding": sum(t == "sliding_attention" for t in types),
            "num_full": len(full_indices),
            "full_indices": full_indices,
        },
        "rope": {
            "sliding_real_inv_freq": flat(sliding_real),
            "full_real_inv_freq": flat(full_real),
            "full_real_rope_angles": int(
                REAL["full_partial_rotary_factor"] * REAL["global_head_dim"] // 2
            ),
            "small": small,
        },
        "sliding_masks": masks,
        "k_eq_v": k_eq_v,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()
    payload = json.dumps(build(), indent=1, sort_keys=False)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()

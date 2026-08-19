#!/usr/bin/env python3
"""Generate the Gemma-4 shared **decoder** goldens (`gemma4_decoder_goldens.json`).

sc-18760 (MLX) / sc-18761 (candle). Where `generate_goldens.py` pins the individual
primitives Gemma 4 adds, this pins the *assembled block*: a whole tiny
`Gemma4Unified` decoder run end to end, so both backends answer to one oracle for
the thing neither primitive test can see — the order the pieces compose in.

The oracle is transcribed from the public reference implementation
(`huggingface/transformers`, `src/transformers/models/gemma4_unified/`):

* `Gemma4UnifiedTextAttention` — per-head q/k RMSNorm before RoPE, unit attention
  scale (`self.scaling = 1.0`), the `attention_k_eq_v` key/value sharing, and the
  **scale-free** `v_norm` applied to the *raw* projection output (both when V is
  shared with K and when it has its own projection).
* `Gemma4UnifiedTextDecoderLayer` — the 4-norm sandwich residual
  (`input_layernorm` → attn → `post_attention_layernorm` → add →
  `pre_feedforward_layernorm` → MLP → `post_feedforward_layernorm` → add).
* `Gemma4UnifiedRMSNorm` — multiplies by the **stored** `weight`, NOT Gemma-2's
  `(1 + weight)` fold; `with_scale=False` is the parameterless value norm.
* `Gemma4UnifiedTextModel` — the `sqrt(hidden_size)` embedding scale, the final
  `model.norm`, tied `lm_head`, and `final_logit_softcapping`.
* `modeling_rope_utils._compute_proportional_rope_parameters` — the
  `full_attention` layers' `rope_type: "proportional"` schedule.
* `masking_utils` — key `j` is visible to query `q` iff `0 <= q - j < sliding_window`
  on a `sliding_attention` layer, plain causal on a `full_attention` one.

The model is deliberately tiny but **not** degenerate: four layers alternating the
two layer types, two different head dims, two different KV-head counts (so both GQA
group counts are exercised), `attention_k_eq_v` on the full layers only, and a
sliding window (2) far shorter than the prompt (5) so the window actually bites.
Weights and the expected outputs are committed together, so `mlx-llm` and
`candle-llm` assert against the identical numbers and neither is the other's oracle.

Only numpy is required (no torch, no weights).

Regenerate with:

    python3 crates/llm/testdata/gemma4/generate_decoder_goldens.py \
        --output /tmp/gemma4_decoder_goldens.json

and copy the result into place — never redirect the generator into its own
checked-in output (a failed run would truncate the golden to zero bytes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- the tiny model ---------------------------------------------------------------------------
HIDDEN = 16
INTERMEDIATE = 16
NUM_LAYERS = 4
NUM_HEADS = 4
NUM_KV_HEADS = 2          # sliding_attention layers
NUM_GLOBAL_KV_HEADS = 1   # full_attention layers (attention_k_eq_v)
HEAD_DIM = 4              # sliding_attention layers
GLOBAL_HEAD_DIM = 8       # full_attention layers
VOCAB = 12
SLIDING_WINDOW = 2
RMS_NORM_EPS = 1e-6
FINAL_LOGIT_SOFTCAPPING = 30.0
SLIDING_THETA = 10_000.0
FULL_THETA = 1_000_000.0
FULL_PARTIAL_ROTARY_FACTOR = 0.25

LAYER_TYPES = [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
]

PROMPT = [3, 7, 1, 9, 4]
DECODE_STEP = [5]

SEED = 18761


def default_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """`compute_default_rope_parameters`: `inv_freq[i] = theta ** (-2i / head_dim)`."""
    return 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))


def proportional_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float) -> np.ndarray:
    """`_compute_proportional_rope_parameters` — the leading channels rotate, the rest are zero."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    rotated = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        return np.concatenate([rotated, np.zeros(nope, dtype=np.float64)])
    return rotated


def cos_sin(inv_freq: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`emb = cat(freqs, freqs)` (NeoX half-split), then cos/sin."""
    freqs = np.outer(positions.astype(np.float64), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """`apply_rotary_pos_emb` with `unsqueeze_dim=2` over `x` `[b, s, h, head_dim]`."""
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """`Gemma4UnifiedRMSNorm`: plain stored `weight`, or parameterless when `weight is None`."""
    x = x.astype(np.float64)
    normed = x * np.power(np.mean(x**2, axis=-1, keepdims=True) + eps, -0.5)
    return normed if weight is None else normed * weight.astype(np.float64)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """`gelu_pytorch_tanh` — the Gemma GeGLU activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def visibility(q_len: int, k_len: int, window: int | None) -> np.ndarray:
    """Bottom-right-aligned causal (`window is None`) or sliding-window visibility."""
    offset = k_len - q_len
    allowed = np.zeros((q_len, k_len), dtype=bool)
    for r in range(q_len):
        pos = offset + r
        for j in range(k_len):
            delta = pos - j
            allowed[r, j] = 0 <= delta < (window if window is not None else k_len + 1)
    return allowed


def repeat_kv(x: np.ndarray, groups: int) -> np.ndarray:
    """`[b, kv_heads, s, d]` → `[b, kv_heads * groups, s, d]` (each KV head repeated `groups` times)."""
    if groups == 1:
        return x
    b, kvh, s, d = x.shape
    return np.repeat(x[:, :, None], groups, axis=2).reshape(b, kvh * groups, s, d)


class Layer:
    """One decoder layer's weights, drawn deterministically for its layer type."""

    def __init__(self, kind: str, rng: np.random.Generator):
        self.kind = kind
        self.sliding = kind == "sliding_attention"
        self.head_dim = HEAD_DIM if self.sliding else GLOBAL_HEAD_DIM
        self.kv_heads = NUM_KV_HEADS if self.sliding else NUM_GLOBAL_KV_HEADS
        self.groups = NUM_HEADS // self.kv_heads
        # attention_k_eq_v gates only the non-sliding layers upstream
        # (`use_alternative_attention = attention_k_eq_v and not is_sliding`).
        self.k_eq_v = not self.sliding
        self.window = SLIDING_WINDOW if self.sliding else None

        qd = NUM_HEADS * self.head_dim
        kvd = self.kv_heads * self.head_dim
        draw = lambda *shape: (rng.random(shape) - 0.5) * 0.4  # noqa: E731

        self.w_q = draw(qd, HIDDEN)
        self.w_k = draw(kvd, HIDDEN)
        self.w_v = None if self.k_eq_v else draw(kvd, HIDDEN)
        self.w_o = draw(HIDDEN, qd)
        self.q_norm = draw(self.head_dim) + 1.0
        self.k_norm = draw(self.head_dim) + 1.0
        self.gate = draw(INTERMEDIATE, HIDDEN)
        self.up = draw(INTERMEDIATE, HIDDEN)
        self.down = draw(HIDDEN, INTERMEDIATE)
        self.input_ln = draw(HIDDEN) + 1.0
        self.post_attn_ln = draw(HIDDEN) + 1.0
        self.pre_ff_ln = draw(HIDDEN) + 1.0
        self.post_ff_ln = draw(HIDDEN) + 1.0

    def rope(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sliding:
            return default_inv_freq(self.head_dim, SLIDING_THETA), None
        return proportional_inv_freq(
            self.head_dim, FULL_THETA, FULL_PARTIAL_ROTARY_FACTOR
        ), None

    def inv_freq(self) -> np.ndarray:
        return self.rope()[0]

    def attention(
        self,
        x: np.ndarray,
        positions: np.ndarray,
        cache: dict | None,
        layer_idx: int,
    ) -> np.ndarray:
        """`Gemma4UnifiedTextAttention.forward` over `x` `[1, s, hidden]`."""
        b, s, _ = x.shape
        hd, kvh = self.head_dim, self.kv_heads

        q = (x @ self.w_q.T).reshape(b, s, NUM_HEADS, hd)
        raw_k = (x @ self.w_k.T).reshape(b, s, kvh, hd)
        raw_v = raw_k if self.k_eq_v else (x @ self.w_v.T).reshape(b, s, kvh, hd)

        q = rms_norm(q, self.q_norm)
        k = rms_norm(raw_k, self.k_norm)
        # The value path takes the RAW projection and a scale-free norm — it is not the key.
        v = rms_norm(raw_v, None)

        cos, sin = cos_sin(self.inv_freq(), positions)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # [b, heads, s, hd]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            prev = cache.get(layer_idx)
            if prev is not None:
                k = np.concatenate([prev[0], k], axis=2)
                v = np.concatenate([prev[1], v], axis=2)
            cache[layer_idx] = (k, v)

        k_all = repeat_kv(k, self.groups)
        v_all = repeat_kv(v, self.groups)

        # `Gemma4UnifiedTextAttention` sets `self.scaling = 1.0` — the q/k norms take its place.
        scores = q @ k_all.transpose(0, 1, 3, 2)
        allowed = visibility(s, k_all.shape[2], self.window)
        scores = np.where(allowed[None, None], scores, -np.inf)
        out = softmax(scores) @ v_all  # [b, heads, s, hd]
        out = out.transpose(0, 2, 1, 3).reshape(b, s, NUM_HEADS * hd)
        return out @ self.w_o.T

    def mlp(self, x: np.ndarray) -> np.ndarray:
        return (gelu_tanh(x @ self.gate.T) * (x @ self.up.T)) @ self.down.T

    def forward(
        self,
        x: np.ndarray,
        positions: np.ndarray,
        cache: dict | None,
        layer_idx: int,
    ) -> np.ndarray:
        attn = self.attention(rms_norm(x, self.input_ln), positions, cache, layer_idx)
        h = x + rms_norm(attn, self.post_attn_ln)
        ffn = self.mlp(rms_norm(h, self.pre_ff_ln))
        return h + rms_norm(ffn, self.post_ff_ln)

    def weights(self) -> dict[str, dict]:
        """The HF checkpoint keys this layer contributes (no `v_proj` when `attention_k_eq_v`)."""
        out = {
            "self_attn.q_proj.weight": self.w_q,
            "self_attn.k_proj.weight": self.w_k,
            "self_attn.o_proj.weight": self.w_o,
            "self_attn.q_norm.weight": self.q_norm,
            "self_attn.k_norm.weight": self.k_norm,
            "mlp.gate_proj.weight": self.gate,
            "mlp.up_proj.weight": self.up,
            "mlp.down_proj.weight": self.down,
            "input_layernorm.weight": self.input_ln,
            "post_attention_layernorm.weight": self.post_attn_ln,
            "pre_feedforward_layernorm.weight": self.pre_ff_ln,
            "post_feedforward_layernorm.weight": self.post_ff_ln,
        }
        if self.w_v is not None:
            out["self_attn.v_proj.weight"] = self.w_v
        return out


class Model:
    def __init__(self, rng: np.random.Generator):
        draw = lambda *shape: (rng.random(shape) - 0.5) * 0.4  # noqa: E731
        self.embed = draw(VOCAB, HIDDEN)
        self.layers = [Layer(kind, rng) for kind in LAYER_TYPES]
        self.norm = draw(HIDDEN) + 1.0

    def embeddings(self, ids: list[int]) -> np.ndarray:
        # Gemma scales the token embeddings by sqrt(hidden_size).
        return self.embed[np.asarray(ids)][None] * np.sqrt(float(HIDDEN))

    def hidden_states(
        self, ids: list[int], offset: int, cache: dict | None
    ) -> list[np.ndarray]:
        """HF `output_hidden_states` layout: embeddings, each layer's output, last one final-normed."""
        positions = np.arange(offset, offset + len(ids))
        h = self.embeddings(ids)
        stack = [h]
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, positions, cache, i)
            stack.append(h)
        stack[-1] = rms_norm(stack[-1], self.norm)
        return stack

    def logits(self, ids: list[int], offset: int, cache: dict | None) -> np.ndarray:
        stack = self.hidden_states(ids, offset, cache)
        logits = stack[-1] @ self.embed.T  # tie_word_embeddings
        cap = FINAL_LOGIT_SOFTCAPPING
        return cap * np.tanh(logits / cap)


def flat(a: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def config() -> dict:
    """The `config.json` the decoder must be built from (the shipped nesting and key names)."""
    return {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "num_global_key_value_heads": NUM_GLOBAL_KV_HEADS,
            "head_dim": HEAD_DIM,
            "global_head_dim": GLOBAL_HEAD_DIM,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": RMS_NORM_EPS,
            "sliding_window": SLIDING_WINDOW,
            "final_logit_softcapping": FINAL_LOGIT_SOFTCAPPING,
            "attention_k_eq_v": True,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False,
            "layer_types": LAYER_TYPES,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": SLIDING_THETA,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": FULL_THETA,
                    "partial_rotary_factor": FULL_PARTIAL_ROTARY_FACTOR,
                },
            },
        },
    }


def build() -> dict:
    rng = np.random.default_rng(SEED)
    model = Model(rng)

    weights: dict[str, dict] = {
        "model.embed_tokens.weight": {"shape": [VOCAB, HIDDEN], "data": flat(model.embed)},
        "model.norm.weight": {"shape": [HIDDEN], "data": flat(model.norm)},
    }
    for i, layer in enumerate(model.layers):
        for suffix, w in layer.weights().items():
            weights[f"model.layers.{i}.{suffix}"] = {
                "shape": list(np.shape(w)),
                "data": flat(w),
            }

    # --- prefill: hidden-state stack + all-position logits, one shared cache ---
    prefill_cache: dict = {}
    prefill_stack = model.hidden_states(PROMPT, 0, prefill_cache)
    prefill_logits = model.logits(PROMPT, 0, None)

    # --- one cached decode step at offset len(PROMPT): the sliding window now spans the cache ---
    decode_cache = dict(prefill_cache)
    decode_logits = model.logits(DECODE_STEP, len(PROMPT), decode_cache)

    # --- the mutation oracles: what the plausible-wrong decoder would produce instead ---
    # Every full_attention layer forced to a separate (differently-drawn) value projection would
    # change the output; so would running every layer with the sliding layers' rope/mask. These are
    # recorded as *distances* the test asserts are large, not as alternative goldens.
    return {
        "_source": "huggingface/transformers gemma4_unified (Gemma4UnifiedTextModel)",
        "_generator": "crates/llm/testdata/gemma4/generate_decoder_goldens.py",
        "_note": (
            "A whole tiny Gemma4Unified decoder: 4 layers alternating sliding/full, two head dims, "
            "two GQA group counts, attention_k_eq_v on the full layers, and a sliding window (2) "
            "shorter than the prompt (5). Shared by mlx-llm (sc-18760) and candle-llm (sc-18761)."
        ),
        "config": config(),
        "shape": {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "num_global_key_value_heads": NUM_GLOBAL_KV_HEADS,
            "head_dim": HEAD_DIM,
            "global_head_dim": GLOBAL_HEAD_DIM,
            "vocab_size": VOCAB,
            "sliding_window": SLIDING_WINDOW,
            "layer_types": LAYER_TYPES,
        },
        "weights": weights,
        "prefill": {
            "input_ids": PROMPT,
            "offset": 0,
            "hidden_states_shape": [1, len(PROMPT), HIDDEN],
            # num_layers + 1 entries; [0] the (scaled) embeddings, [-1] the final-normed output.
            "hidden_states": [flat(h) for h in prefill_stack],
            "logits_shape": [1, len(PROMPT), VOCAB],
            "logits": flat(prefill_logits),
            "last_logits": flat(prefill_logits[0, -1]),
        },
        "decode": {
            "input_ids": DECODE_STEP,
            "offset": len(PROMPT),
            "logits_shape": [1, 1, VOCAB],
            "last_logits": flat(decode_logits[0, -1]),
        },
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

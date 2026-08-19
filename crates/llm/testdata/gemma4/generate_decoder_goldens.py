#!/usr/bin/env python3
"""Generate the Gemma-4 unified **decoder** goldens (`gemma4_decoder_goldens.json`).

sc-18760 (mlx-llm) / sc-18761 (candle-llm). Where `generate_goldens.py` pins the individual
primitives Gemma 4 adds, this pins the **whole block**: a complete small Gemma-4-unified decoder
— weights, inputs, every layer's hidden states, and the final logits — so both backends assert
the same end-to-end numbers.

The oracle is transcribed from the pinned reference implementation, `huggingface/transformers`
5.14.1 `src/transformers/models/gemma4_unified/` (the version `Lightricks/LTX-2` @ `d151147`
pins via `transformers>=5.8.0,<5.15`), specifically:

* `modeling_gemma4_unified.py::Gemma4UnifiedTextDecoderLayer.forward` — the 4-norm sandwich and
  the trailing `hidden_states *= self.layer_scalar`.
* `modeling_gemma4_unified.py::Gemma4UnifiedTextAttention.forward` — `scaling = 1.0`, the q/k
  norms applied *before* RoPE, and the `attention_k_eq_v` aliasing that makes V the **raw**
  key-projection output (pre-`k_norm`, pre-RoPE) under a scale-free `v_norm`.
* `modeling_gemma4_unified.py::Gemma4UnifiedRMSNorm` — plain `weight`, **not** Gemma-2's
  `(1 + weight)`.
* `modeling_gemma4_unified.py::Gemma4UnifiedTextScaledWordEmbedding` — `sqrt(hidden_size)`,
  rounded to the compute dtype *before* the multiply.
* `modeling_rope_utils.py::_compute_proportional_rope_parameters` — the `full_attention`
  frequency schedule (exponent denominator is the **full** head dim; the tail is exactly zero).
* `masking_utils` sliding-window semantics — key `j` is visible to query `q` iff
  `0 <= q - j < sliding_window`.

# Why the fixture is shaped the way it is

* **Both layer types, interleaved.** `layer_types` is given explicitly as
  `[sliding, full, sliding, full]`. A sliding-only fixture is a false green: the two types differ
  in head dim (8 vs 16), KV-head count (2 vs 1), RoPE schedule (default vs proportional), mask
  (windowed vs plain causal) *and* whether K/V share a projection. Any one of those five being
  wrong has to fail here.
* **The window actually bites.** `sliding_window = 3` over a 7-token prompt, so a sliding layer
  that forgets its window attends keys it must not — the `no_window` mutation below.
* **`layer_scalar` is never 1.0.** The reference initializes it to ones, so a port that skips it
  passes any fixture that left it at the initializer. These are 0.9 / 1.1 / 0.95 / 1.05.
* **`partial_rotary_factor` leaves a real zero tail.** `global_head_dim = 16` gives
  `rope_angles = int(0.25 * 16 // 2) = 2` rotated channels and 6 zeroed ones.
* **Weights are exactly bf16-representable.** The decoders compute in bf16; emitting weights that
  already round-trip through bf16 means the only divergence from this f64 oracle is arithmetic
  rounding, not a different model.

Every golden ships with **mutation** outputs — the same forward with one plausible mistake — so a
test can prove the fixture discriminates instead of merely matching something.

Only numpy is required (no torch, no weights, no network). Regenerate with:

    python3 crates/llm/testdata/gemma4/generate_decoder_goldens.py --output /tmp/out.json

and write to a temporary file first — never redirect the generator into its own checked-in output.

`--verify-reference <path-to-transformers>` additionally imports the real
`Gemma4UnifiedTextModel` and asserts this numpy oracle reproduces it. That needs torch and the
pinned transformers; it is a development check, not part of regeneration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- the synthetic model ---------------------------------------------------------------------

CONFIG = {
    "vocab_size": 40,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 8,
    "global_head_dim": 16,
    "sliding_window": 3,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
    "attention_k_eq_v": True,
    "tie_word_embeddings": True,
    "hidden_activation": "gelu_pytorch_tanh",
    "sliding_theta": 10_000.0,
    "full_theta": 1_000_000.0,
    "full_partial_rotary_factor": 0.25,
}

LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
LAYER_SCALARS = [0.9, 1.1, 0.95, 1.05]
PROMPT = [3, 1, 4, 1, 5, 9, 2]
SEED = 18760


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to bfloat16 and return the (exact) float64 value.

    numpy has no bfloat16, so this truncates an f32 to its high 16 bits with round-to-nearest-even
    — the same value a `Dtype::Bfloat16` cast produces. Emitting already-rounded weights keeps the
    Rust model and this oracle the *same model*, so the only residual difference is bf16 arithmetic
    rounding inside the forward.
    """
    f32 = np.asarray(x, dtype=np.float32)
    bits = f32.view(np.uint32).astype(np.uint64)
    # Round half to even on the 16 bits being dropped.
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32).astype(np.float64)


def randn(rng: np.random.Generator, *shape: int, scale: float = 0.4) -> np.ndarray:
    """A small bf16-exact weight. Kept small so four layers of bf16 accumulation stay well-conditioned."""
    return to_bf16(rng.standard_normal(shape) * scale)


def norm_weight(rng: np.random.Generator, dim: int) -> np.ndarray:
    """An RMSNorm weight around 1.0 — Gemma 4 initializes to ones and multiplies by the stored
    value directly (no `1 + w` fold), so a fixture centred on 1.0 is the realistic one."""
    return to_bf16(1.0 + rng.standard_normal(dim) * 0.1)


# --- the reference math ------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    """`Gemma4UnifiedRMSNorm`: `x * (mean(x**2) + eps) ** -0.5`, then `* weight` when scaled.

    Three details are load-bearing and all three are easy to get wrong:

    * the epsilon is added *inside*, before the reciprocal square root (`mean(x²) + eps`, then
      `pow(·, -0.5)`) — not `sqrt(mean + eps)` and not `rsqrt`;
    * there is **no** `1 + weight` fold. That is Gemma-2/3. Gemma 4 initializes the weight to ones
      and multiplies by the stored value directly, so folding a `+1` in corrupts every norm;
    * the whole reduction runs in **float32** — the reference writes `self._norm(hidden_states
      .float())` and `self.weight.float()` explicitly, regardless of the model's dtype.

    `weight is None` is the scale-free `v_norm` (`with_scale=False`), which has no parameter at all.
    """
    x32 = np.asarray(x, dtype=np.float32)
    mean_squared = np.mean(x32**2, axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps)
    normed = x32 * np.power(mean_squared, np.float32(-0.5), dtype=np.float32)
    if weight is not None:
        normed = normed * np.asarray(weight, dtype=np.float32)
    return normed.astype(np.float64)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """`gelu_pytorch_tanh` — the tanh-approximate GELU Gemma uses."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * np.power(x, 3))))


def default_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """`compute_default_rope_parameters`: `inv_freq[i] = theta ** (-2i / head_dim)`."""
    return 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))


def proportional_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float) -> np.ndarray:
    """`_compute_proportional_rope_parameters`.

    `rope_angles = int(partial_rotary_factor * head_dim // 2)` channels carry the ordinary
    `theta ** (-2i / head_dim)` — the exponent denominator stays the FULL `head_dim`, which is the
    single easiest thing to get wrong — and the rest are exactly zero (an identity rotation).
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    rotated = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim))
    nope = head_dim // 2 - rope_angles
    return np.concatenate([rotated, np.zeros(nope, dtype=np.float64)]) if nope > 0 else rotated


def partial_inv_freq(head_dim: int, theta: float, rotary_dim: int) -> np.ndarray:
    """A **leading-slice** partial RoPE — the plausible wrong reading of `partial_rotary_factor`.

    It re-bases the exponent on the rotated span (`2i / rotary_dim`) and pairs channel `i` with
    `i + rotary_dim/2` *inside* the slice, where proportional keeps `2i / head_dim` and the whole
    head's NeoX pairing. Only ever used to build the `full_rope_as_partial` mutation.
    """
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    return np.concatenate([inv, np.zeros(head_dim // 2 - half, dtype=np.float64)])


def cos_sin(inv_freq: np.ndarray, positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """`Gemma4UnifiedTextRotaryEmbedding.forward`: `emb = cat(freqs, freqs)` (NeoX half-split).

    Computed in **float32**, which is neither an accident nor this generator's choice: the
    reference forces it (`inv_freq_expanded.float() @ position_ids_expanded.float()` inside an
    `enabled=False` autocast) regardless of the model dtype, and both decoders build their tables
    from an `f32` inverse-frequency vector too. Computing it in f64 here instead would put a ~1e-7
    skew between this oracle and every implementation it is meant to pin.
    """
    inv32 = np.asarray(inv_freq, dtype=np.float32)
    freqs = np.outer(np.asarray(positions, dtype=np.float32), inv32).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float64), np.sin(emb).astype(np.float64)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """`apply_rotary_pos_emb` with `unsqueeze_dim=2` over `x` `[b, s, h, head_dim]`."""
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]


def sliding_mask(seq: int, window: int | None) -> np.ndarray:
    """Additive causal mask, optionally narrowed to `window` most-recent keys (inclusive)."""
    m = np.zeros((seq, seq), dtype=np.float64)
    for q in range(seq):
        for k in range(seq):
            delta = q - k
            blocked = delta < 0 if window is None else not (0 <= delta < window)
            if blocked:
                m[q, k] = -np.inf
    return m


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def repeat_kv(x: np.ndarray, groups: int) -> np.ndarray:
    """`[b, s, kv_heads, d]` -> `[b, s, kv_heads * groups, d]`, each KV head repeated `groups` times."""
    if groups == 1:
        return x
    b, s, h, d = x.shape
    return np.repeat(x, groups, axis=2).reshape(b, s, h * groups, d)


class Layer:
    """One decoder layer's weights.

    `kv_shared` is upstream's `is_kv_shared_layer`: a trailing layer that projects **no** keys or
    values of its own (no `k_proj`, no `v_proj`, no `k_norm`, no `v_norm` module at all) and reads
    the stored K/V of the last earlier layer of its own type. `double_wide` is
    `use_double_wide_mlp`, which upstream applies to exactly those layers.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        kind: str,
        scalar: float,
        kv_shared: bool = False,
        double_wide: bool = False,
    ) -> None:
        c = CONFIG
        self.kind = kind
        self.sliding = kind == "sliding_attention"
        self.kv_shared = kv_shared
        self.stores_kv = False  # set by Model once the whole schedule is known
        self.head_dim = c["head_dim"] if self.sliding else c["global_head_dim"]
        self.kv_heads = c["num_key_value_heads"] if self.sliding else c["num_global_key_value_heads"]
        # `use_alternative_attention = attention_k_eq_v and not is_sliding`.
        self.k_eq_v = c["attention_k_eq_v"] and not self.sliding
        self.window = c["sliding_window"] if self.sliding else None
        h = c["hidden_size"]
        inter = c["intermediate_size"] * (2 if double_wide else 1)
        qd = c["num_attention_heads"] * self.head_dim
        kvd = self.kv_heads * self.head_dim

        self.input_ln = norm_weight(rng, h)
        self.post_attn_ln = norm_weight(rng, h)
        self.pre_ff_ln = norm_weight(rng, h)
        self.post_ff_ln = norm_weight(rng, h)
        self.q_proj = randn(rng, qd, h)
        # A KV-sharing layer owns none of the key/value machinery.
        self.k_proj = None if kv_shared else randn(rng, kvd, h)
        self.v_proj = None if (kv_shared or self.k_eq_v) else randn(rng, kvd, h)
        self.o_proj = randn(rng, h, qd)
        self.q_norm = norm_weight(rng, self.head_dim)
        self.k_norm = None if kv_shared else norm_weight(rng, self.head_dim)
        self.gate = randn(rng, inter, h)
        self.up = randn(rng, inter, h)
        self.down = randn(rng, h, inter)
        self.layer_scalar = to_bf16(np.array([scalar]))

    def rope_tables(self, positions: list[int], mutation: str) -> tuple[np.ndarray, np.ndarray]:
        c = CONFIG
        if self.sliding:
            inv = default_inv_freq(self.head_dim, c["sliding_theta"])
        elif mutation == "full_rope_as_partial":
            rotary = int(c["full_partial_rotary_factor"] * self.head_dim)
            inv = partial_inv_freq(self.head_dim, c["full_theta"], rotary)
        else:
            inv = proportional_inv_freq(
                self.head_dim, c["full_theta"], c["full_partial_rotary_factor"]
            )
        return cos_sin(inv, positions)

    def attention(
        self,
        x: np.ndarray,
        positions: list[int],
        mutation: str,
        shared_kv: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        c = CONFIG
        eps = c["rms_norm_eps"]
        b, s, _ = x.shape
        heads = c["num_attention_heads"]
        cos, sin = self.rope_tables(positions, mutation)

        q = (x @ self.q_proj.T).reshape(b, s, heads, self.head_dim)
        q = rms_norm(q, self.q_norm, eps)
        q = apply_rope(q, cos, sin)

        if self.kv_shared:
            # `is_kv_shared_layer`: reuse the stored K/V of the last earlier layer of this type.
            # Note the reference reads `shared_kv_states` even when a cache exists, because a
            # sliding layer's cache may no longer hold the full-length keys.
            k, v = shared_kv[self.kind]
        else:
            raw_k = (x @ self.k_proj.T).reshape(b, s, self.kv_heads, self.head_dim)
            if self.v_proj is not None:
                raw_v = (x @ self.v_proj.T).reshape(b, s, self.kv_heads, self.head_dim)
            else:
                # `attention_k_eq_v`: V aliases the RAW key projection output. The reference rebinds
                # `key_states` on the next line, so V never sees `k_norm` and never sees RoPE.
                raw_v = raw_k

            k = rms_norm(raw_k, self.k_norm, eps)
            k = apply_rope(k, cos, sin)

            if mutation == "v_from_normed_k" and self.k_eq_v:
                # The plausible mistake: reading V off the *normed and rotated* key.
                v = rms_norm(k, None, eps)
            else:
                v = rms_norm(raw_v, None, eps)

            if self.stores_kv:
                shared_kv[self.kind] = (k, v)

        groups = heads // self.kv_heads
        k = repeat_kv(k, groups)
        v = repeat_kv(v, groups)

        # scores: [b, heads, s, s]. `scaling` is a literal 1.0 — the learned q/k norms absorb the
        # usual `head_dim ** -0.5`.
        scores = np.einsum("bqhd,bkhd->bhqk", q, k) * 1.0
        window = None if mutation == "no_window" else self.window
        scores = scores + sliding_mask(s, window)[None, None, :, :]
        weights = softmax(scores)
        out = np.einsum("bhqk,bkhd->bqhd", weights, v)
        return out.reshape(b, s, heads * self.head_dim) @ self.o_proj.T

    def mlp(self, x: np.ndarray) -> np.ndarray:
        return (gelu_tanh(x @ self.gate.T) * (x @ self.up.T)) @ self.down.T

    def forward(
        self,
        x: np.ndarray,
        positions: list[int],
        mutation: str,
        shared_kv: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        eps = CONFIG["rms_norm_eps"]
        residual = x
        h = rms_norm(x, self.input_ln, eps)
        h = self.attention(h, positions, mutation, shared_kv)
        h = rms_norm(h, self.post_attn_ln, eps)
        h = residual + h

        residual = h
        h = rms_norm(h, self.pre_ff_ln, eps)
        h = self.mlp(h)
        h = rms_norm(h, self.post_ff_ln, eps)
        h = residual + h

        # `hidden_states *= self.layer_scalar` — after BOTH residual adds, scaling the block's
        # whole contribution including the residual stream.
        scalar = 1.0 if mutation == "no_layer_scalar" else self.layer_scalar[0]
        return h * scalar


class Model:
    def __init__(
        self,
        rng: np.random.Generator,
        round_embed_scale: bool = True,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
    ) -> None:
        c = CONFIG
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        # `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`, with upstream's
        # `> 0` guard so a `0` setting does not make every layer "shared".
        first_shared = len(LAYER_TYPES) - num_kv_shared_layers
        self.embed = randn(rng, c["vocab_size"], c["hidden_size"], scale=0.5)
        self.layers = []
        for i, (kind, s) in enumerate(zip(LAYER_TYPES, LAYER_SCALARS)):
            shared = num_kv_shared_layers > 0 and i >= first_shared
            self.layers.append(
                Layer(rng, kind, s, kv_shared=shared, double_wide=use_double_wide_mlp and shared)
            )
        # `store_full_length_kv`: the LAST layer of each type before the sharing tail. Walking
        # forward and overwriting leaves exactly that layer marked.
        if num_kv_shared_layers > 0:
            last_of_type: dict[str, int] = {}
            for i in range(first_shared):
                last_of_type[LAYER_TYPES[i]] = i
            for i in last_of_type.values():
                self.layers[i].stores_kv = True
        self.norm = norm_weight(rng, c["hidden_size"])
        # `Gemma4UnifiedTextScaledWordEmbedding` computes `hidden_size ** 0.5` in fp32 and then
        # `.to(self.weight.dtype)` — so on a bf16 checkpoint the scale is *rounded to bf16 before
        # the multiply*, which is what the decoders do and therefore what the goldens must carry.
        # `round_embed_scale=False` reproduces a float64 reference run instead, and exists only so
        # `--verify-reference` compares the math rather than the dtype.
        raw = float(np.sqrt(c["hidden_size"]))
        self.embed_scale = float(to_bf16(np.array([raw]))[0]) if round_embed_scale else raw

    def hidden_states(self, tokens: list[int], mutation: str = "none") -> list[np.ndarray]:
        """HF's `output_hidden_states` layout: input embeddings, then every layer's output, with
        the **last** entry replaced by the final-normed one (`last_hidden_state`)."""
        positions = list(range(len(tokens)))
        h = self.embed[tokens][None, :, :] * self.embed_scale
        out = [h]
        shared_kv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for layer in self.layers:
            h = layer.forward(h, positions, mutation, shared_kv)
            out.append(h)
        out[-1] = rms_norm(out[-1], self.norm, CONFIG["rms_norm_eps"])
        return out

    def logits(self, tokens: list[int], mutation: str = "none") -> np.ndarray:
        states = self.hidden_states(tokens, mutation)
        # `last_hidden_state` is already final-normed; the head is the tied embedding matrix.
        logits = states[-1] @ self.embed.T
        cap = CONFIG["final_logit_softcapping"]
        return cap * np.tanh(logits / cap)

    def weights(self) -> dict[str, dict]:
        def entry(a: np.ndarray) -> dict:
            return {"shape": list(a.shape), "data": [float(v) for v in a.reshape(-1)]}

        w = {
            "model.embed_tokens.weight": entry(self.embed),
            "model.norm.weight": entry(self.norm),
        }
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}."
            w[p + "input_layernorm.weight"] = entry(layer.input_ln)
            w[p + "post_attention_layernorm.weight"] = entry(layer.post_attn_ln)
            w[p + "pre_feedforward_layernorm.weight"] = entry(layer.pre_ff_ln)
            w[p + "post_feedforward_layernorm.weight"] = entry(layer.post_ff_ln)
            w[p + "self_attn.q_proj.weight"] = entry(layer.q_proj)
            if layer.k_proj is not None:
                w[p + "self_attn.k_proj.weight"] = entry(layer.k_proj)
            if layer.v_proj is not None:
                w[p + "self_attn.v_proj.weight"] = entry(layer.v_proj)
            w[p + "self_attn.o_proj.weight"] = entry(layer.o_proj)
            w[p + "self_attn.q_norm.weight"] = entry(layer.q_norm)
            if layer.k_norm is not None:
                w[p + "self_attn.k_norm.weight"] = entry(layer.k_norm)
            w[p + "mlp.gate_proj.weight"] = entry(layer.gate)
            w[p + "mlp.up_proj.weight"] = entry(layer.up)
            w[p + "mlp.down_proj.weight"] = entry(layer.down)
            w[p + "layer_scalar"] = entry(layer.layer_scalar)
        return w


def model_config_json(
    num_kv_shared_layers: int = 0, use_double_wide_mlp: bool = False
) -> dict:
    """The `config.json` the decoder must be built from — the shape a real Gemma 4 config has,
    with `layer_types` given explicitly so the fixture pins an interleaved schedule rather than
    the derived 5:1 one (which 4 layers could not express)."""
    c = CONFIG
    return {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "tie_word_embeddings": c["tie_word_embeddings"],
        "text_config": {
            "model_type": "gemma4_unified_text",
            "vocab_size": c["vocab_size"],
            "hidden_size": c["hidden_size"],
            "intermediate_size": c["intermediate_size"],
            "num_hidden_layers": c["num_hidden_layers"],
            "num_attention_heads": c["num_attention_heads"],
            "num_key_value_heads": c["num_key_value_heads"],
            "num_global_key_value_heads": c["num_global_key_value_heads"],
            "head_dim": c["head_dim"],
            "global_head_dim": c["global_head_dim"],
            "hidden_activation": c["hidden_activation"],
            "rms_norm_eps": c["rms_norm_eps"],
            "sliding_window": c["sliding_window"],
            "layer_types": LAYER_TYPES,
            "final_logit_softcapping": c["final_logit_softcapping"],
            "attention_k_eq_v": c["attention_k_eq_v"],
            "num_kv_shared_layers": num_kv_shared_layers,
            "use_double_wide_mlp": use_double_wide_mlp,
            "use_bidirectional_attention": "vision",
            "attention_bias": False,
            "tie_word_embeddings": c["tie_word_embeddings"],
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": c["sliding_theta"],
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": c["full_theta"],
                    "partial_rotary_factor": c["full_partial_rotary_factor"],
                },
            },
        },
    }


MUTATIONS = {
    "no_layer_scalar": "every `layer_scalar` forced to its 1.0 initializer (the port that never "
    "read the buffer)",
    "v_from_normed_k": "under `attention_k_eq_v`, V read off the normed+rotated K instead of the "
    "raw projection output",
    "no_window": "`sliding_attention` layers run a plain causal mask, ignoring `sliding_window`",
    "full_rope_as_partial": "`full_attention` layers use a leading-slice partial RoPE "
    "(`2i/rotary_dim`) instead of the proportional schedule (`2i/head_dim` + zero tail)",
}


def flat(a: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def build() -> dict:
    rng = np.random.default_rng(SEED)
    model = Model(rng)
    states = model.hidden_states(PROMPT)
    payload = {
        "_source": "huggingface/transformers 5.14.1 gemma4_unified (the revision Lightricks/LTX-2 "
        "@ d151147 pins)",
        "_generator": "crates/llm/testdata/gemma4/generate_decoder_goldens.py",
        "_seed": SEED,
        "config": model_config_json(),
        "layer_types": LAYER_TYPES,
        "layer_scalars": LAYER_SCALARS,
        "prompt": PROMPT,
        "embed_scale": model.embed_scale,
        "weights": model.weights(),
        "hidden_states": {
            "shape": [1, len(PROMPT), CONFIG["hidden_size"]],
            "count": len(states),
            "layers": [flat(s) for s in states],
        },
        "logits": {
            "shape": [1, len(PROMPT), CONFIG["vocab_size"]],
            "data": flat(model.logits(PROMPT)),
        },
        "mutations": {},
    }
    for name, why in MUTATIONS.items():
        payload["mutations"][name] = {
            "why": why,
            "hidden_states": [flat(s) for s in model.hidden_states(PROMPT, name)],
            "logits": flat(model.logits(PROMPT, name)),
        }

    # A second, smaller fixture for the two features the shipped LTX-2.5 encoder leaves off —
    # `num_kv_shared_layers` and `use_double_wide_mlp`. They are parsed by the config layer, so a
    # decoder that ignored them would run a *different model* in silence on any checkpoint that
    # sets them. With 4 layers and 2 shared, the sharing tail covers both layer types: layer 2
    # (sliding) reuses layer 0's K/V, layer 3 (full) reuses layer 1's.
    kv_rng = np.random.default_rng(SEED + 1)
    kv_model = Model(kv_rng, num_kv_shared_layers=2, use_double_wide_mlp=True)
    payload["kv_shared"] = {
        "_why": "num_kv_shared_layers: 2 + use_double_wide_mlp — the trailing layers project no "
        "K/V of their own and reuse the last earlier layer of their type, with a double-width MLP",
        "num_kv_shared_layers": 2,
        "use_double_wide_mlp": True,
        "config": model_config_json(num_kv_shared_layers=2, use_double_wide_mlp=True),
        "weights": kv_model.weights(),
        "hidden_states": [flat(s) for s in kv_model.hidden_states(PROMPT)],
        "logits": flat(kv_model.logits(PROMPT)),
    }
    return payload


# --- optional cross-check against the real reference module -------------------------------------


def verify_reference(transformers_path: Path) -> None:
    """Assert this numpy oracle reproduces the actual `Gemma4UnifiedTextModel`.

    Needs torch and the pinned transformers on `sys.path`. Not part of regeneration — the committed
    goldens are the numpy transcription, which is what keeps the fixture runnable anywhere.
    """
    import sys

    sys.path.insert(0, str(transformers_path))
    import torch  # noqa: PLC0415
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (  # noqa: PLC0415
        Gemma4UnifiedTextConfig,
    )
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (  # noqa: PLC0415
        Gemma4UnifiedTextModel,
    )

    torch.set_default_dtype(torch.float64)

    def reference_states(mine: Model, **cfg_kwargs) -> list[np.ndarray]:
        """Load `mine`'s weights into the real module and return its hidden states."""
        text_cfg = model_config_json(**cfg_kwargs)["text_config"]
        text_cfg = {k: v for k, v in text_cfg.items() if k != "model_type"}
        cfg = Gemma4UnifiedTextConfig(**text_cfg)
        cfg._attn_implementation = "eager"
        ref = Gemma4UnifiedTextModel(cfg).to(torch.float64).eval()

        sd = {"embed_tokens.weight": torch.tensor(mine.embed)}
        for i, layer in enumerate(mine.layers):
            p = f"layers.{i}."
            sd[p + "input_layernorm.weight"] = torch.tensor(layer.input_ln)
            sd[p + "post_attention_layernorm.weight"] = torch.tensor(layer.post_attn_ln)
            sd[p + "pre_feedforward_layernorm.weight"] = torch.tensor(layer.pre_ff_ln)
            sd[p + "post_feedforward_layernorm.weight"] = torch.tensor(layer.post_ff_ln)
            sd[p + "self_attn.q_proj.weight"] = torch.tensor(layer.q_proj)
            if layer.k_proj is not None:
                sd[p + "self_attn.k_proj.weight"] = torch.tensor(layer.k_proj)
            if layer.v_proj is not None:
                sd[p + "self_attn.v_proj.weight"] = torch.tensor(layer.v_proj)
            sd[p + "self_attn.o_proj.weight"] = torch.tensor(layer.o_proj)
            sd[p + "self_attn.q_norm.weight"] = torch.tensor(layer.q_norm)
            if layer.k_norm is not None:
                sd[p + "self_attn.k_norm.weight"] = torch.tensor(layer.k_norm)
            sd[p + "mlp.gate_proj.weight"] = torch.tensor(layer.gate)
            sd[p + "mlp.up_proj.weight"] = torch.tensor(layer.up)
            sd[p + "mlp.down_proj.weight"] = torch.tensor(layer.down)
            sd[p + "layer_scalar"] = torch.tensor(layer.layer_scalar)
        sd["norm.weight"] = torch.tensor(mine.norm)
        missing, unexpected = ref.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected if "inv_freq" not in k]
        missing = [k for k in missing if "inv_freq" not in k]
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert not missing, f"missing keys: {missing}"
        with torch.no_grad():
            out = ref(
                input_ids=torch.tensor([PROMPT]), use_cache=False, output_hidden_states=True
            )
        return [h.numpy() for h in out.hidden_states]

    rng = np.random.default_rng(SEED)
    # The reference runs here in float64, so it never rounds `embed_scale` to bf16; matching that
    # isolates the comparison to the decoder math.
    mine = Model(rng, round_embed_scale=False)
    ref_states = reference_states(mine)

    # The tolerance is float32-relative, not f64-exact, and deliberately so: the reference runs its
    # RMSNorm and its RoPE tables in float32 no matter the model dtype (`hidden_states.float()`,
    # `inv_freq_expanded.float()`), so numpy and torch differ in the last ULP of those float32
    # reductions. Any *structural* error — a `1 + weight` norm, a missing `layer_scalar`, V read
    # off the normed key, a leading-slice RoPE, an unwindowed sliding layer — moves the output by
    # a relative 1e-2 or more, four orders of magnitude above this floor. The mutation sweep below
    # proves that rather than asserting it.
    rel_tol = 1e-5

    def compare(label: str, states: list[np.ndarray]) -> float:
        worst = 0.0
        for i, (r, m) in enumerate(zip(ref_states, states)):
            scale = max(float(np.max(np.abs(r))), 1.0)
            worst = max(worst, float(np.max(np.abs(r - m))) / scale)
        print(f"  {label}: worst relative delta = {worst:.3e}")
        return worst

    mine_states = mine.hidden_states(PROMPT)
    assert len(ref_states) == len(mine_states), (
        f"{len(ref_states)} reference states vs {len(mine_states)} oracle states"
    )
    worst = compare("oracle", mine_states)
    assert worst < rel_tol, f"the oracle diverges from the reference by {worst} (rel)"

    # Each mutation must FAIL the same comparison. A verification that only ever checks the correct
    # oracle cannot tell "reproduces the reference" from "compares nothing".
    for name in MUTATIONS:
        got = compare(f"mutation/{name}", mine.hidden_states(PROMPT, name))
        assert got > rel_tol * 100, (
            f"mutation {name!r} is within {got} of the reference — the comparison does not "
            f"discriminate it, so the golden built from it proves nothing"
        )

    # The `num_kv_shared_layers` + `use_double_wide_mlp` variant gets the same treatment: the
    # sharing tail is easy to implement plausibly-but-wrongly (reusing the *cache* rather than the
    # stored full-length K/V, or reusing the immediately-preceding layer rather than the last one
    # of the same type), and neither mistake is visible without a reference.
    kv_rng = np.random.default_rng(SEED + 1)
    kv_mine = Model(kv_rng, round_embed_scale=False, num_kv_shared_layers=2, use_double_wide_mlp=True)
    ref_states = reference_states(kv_mine, num_kv_shared_layers=2, use_double_wide_mlp=True)
    kv_worst = compare("oracle/kv_shared", kv_mine.hidden_states(PROMPT))
    assert kv_worst < rel_tol, f"the kv-shared oracle diverges by {kv_worst} (rel)"

    print(
        f"OK: numpy oracle reproduces transformers Gemma4UnifiedTextModel "
        f"(worst {worst:.3e} rel; kv-shared {kv_worst:.3e} rel)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument(
        "--verify-reference",
        type=Path,
        default=None,
        help="path to a site-packages holding the pinned transformers; cross-checks the oracle",
    )
    args = parser.parse_args()
    if args.verify_reference is not None:
        verify_reference(args.verify_reference)
        return
    payload = json.dumps(build(), indent=1, sort_keys=False)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()

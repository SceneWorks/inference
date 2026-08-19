#!/usr/bin/env python3
"""Generate `architecture_regression.json` — the shared cross-backend fixture pinning that every
already-shipped `Architecture` still parses and derives identically (sc-18769).

sc-18769 extends `ModelConfig`, the struct **every** LLM architecture in `mlx-llm` and `candle-llm`
shares, with Gemma-4's per-layer-type attention table. That is a wide blast radius: a regression
here is silent (a slightly wrong RoPE base or attention scale still renders, just worse). This
fixture is the guard — one real-shaped `config.json` per architecture plus the values the config
layer must derive from it, asserted by both backends against the same numbers.

The expectations are an **independent oracle**: they are computed here from the documented formulas
(`head_dim = hidden_size / num_attention_heads` unless given, `groups = num_heads / num_kv_heads`,
`rotary_dim = round(head_dim * partial_rotary_factor)` forced even, `attn_scale =
query_pre_attn_scalar ** -0.5` or `head_dim ** -0.5`, `inv_freq[i] = theta ** (-2i / dim)` with the
llama3 / YaRN schedules), not read back out of the Rust that consumes them.

Regenerate with:

    python3 crates/llm/testdata/architectures/generate_regression.py --output <tmp>

and copy the result in — never redirect a generator into its own checked-in output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def default_inv_freq(dim: int, theta: float) -> list[float]:
    return [1.0 / (theta ** ((2 * i) / dim)) for i in range(dim // 2)]


def llama3_inv_freq(
    dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_context: float,
) -> list[float]:
    low_wavelen = original_context / low_freq_factor
    high_wavelen = original_context / high_freq_factor
    out = []
    for inv in default_inv_freq(dim, theta):
        wavelen = 2.0 * math.pi / inv
        if wavelen > low_wavelen:
            out.append(inv / factor)
        elif wavelen < high_wavelen:
            out.append(inv)
        else:
            smooth = (original_context / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            out.append((1.0 - smooth) * inv / factor + smooth * inv)
    return out


def yarn_inv_freq(
    dim: int,
    theta: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_context: float,
) -> list[float]:
    def correction_dim(rotations: float) -> float:
        return (dim * math.log(original_context / (rotations * 2.0 * math.pi))) / (
            2.0 * math.log(theta)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0.0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1.0)
    span = 1e-3 if abs(high - low) < 1e-3 else high - low
    out = []
    for i, extra in enumerate(default_inv_freq(dim, theta)):
        inter = extra / factor
        ramp = min(max((i - low) / span, 0.0), 1.0)
        out.append(inter * ramp + extra * (1.0 - ramp))
    return out


def rotary_dim(head_dim: int, partial: float) -> int:
    # `round(head_dim * partial)` forced even (RoPE rotates in pairs). Rust's `f32::round` is
    # half-away-from-zero, which for the shipped factors (1.0, 0.5) is exact either way.
    return int(math.floor(head_dim * partial + 0.5)) & ~1


CASES: list[dict] = []


def case(
    name: str,
    config: dict,
    *,
    family: str,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    partial: float = 1.0,
    query_pre_attn_scalar: int | None = None,
    rope_dim: int | None = None,
    rope_interleaved: bool = False,
    inv_freq: list[float],
    attn_scale: float | None = None,
    is_moe: bool = False,
    is_mla: bool = False,
    has_qk_norm: bool = False,
    is_sandwich: bool = False,
    **expect_extra,
) -> None:
    rd = rope_dim if rope_dim is not None else rotary_dim(head_dim, partial)
    scale = (
        attn_scale
        if attn_scale is not None
        else (query_pre_attn_scalar if query_pre_attn_scalar is not None else head_dim) ** -0.5
    )
    expect = {
        "family": family,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "groups": num_heads // num_kv_heads,
        "partial_rotary_factor": partial,
        "rotary_dim": rotary_dim(head_dim, partial),
        "attn_scale": scale,
        "is_moe": is_moe,
        "is_mla": is_mla,
        "has_qk_norm": has_qk_norm,
        "is_sandwich": is_sandwich,
        "is_gemma4": False,
        "rope_dim": rd,
        "rope_interleaved": rope_interleaved,
        # First and last four inverse frequencies pin both ends of the schedule (a wrong theta or a
        # dropped scaling branch moves one or the other).
        "rope_inv_freq_head": inv_freq[:4],
        "rope_inv_freq_tail": inv_freq[-4:],
    }
    expect.update(expect_extra)
    CASES.append({"name": name, "config": config, "expect": expect})


# --- Llama 3.1 8B (NTK-by-parts scaled RoPE, GQA) ---
case(
    "llama_3_1_8b",
    {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": False,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    },
    family="llama",
    head_dim=128,
    num_heads=32,
    num_kv_heads=8,
    inv_freq=llama3_inv_freq(128, 500000.0, 8.0, 1.0, 4.0, 8192.0),
    tie_word_embeddings=False,
)

# --- Qwen3 8B (per-head q/k RMSNorm, explicit head_dim) ---
case(
    "qwen3_8b",
    {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
    },
    family="qwen3",
    head_dim=128,
    num_heads=32,
    num_kv_heads=8,
    has_qk_norm=True,
    inv_freq=default_inv_freq(128, 1000000.0),
    tie_word_embeddings=False,
)

# --- Gemma-2 9B (soft-caps, sandwich norms, query_pre_attn_scalar, implicit tie) ---
case(
    "gemma2_9b",
    {
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "hidden_size": 3584,
        "intermediate_size": 14336,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "vocab_size": 256000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "query_pre_attn_scalar": 224,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "max_position_embeddings": 8192,
    },
    family="gemma2",
    head_dim=256,
    num_heads=16,
    num_kv_heads=8,
    query_pre_attn_scalar=224,
    is_sandwich=True,
    inv_freq=default_inv_freq(256, 10000.0),
    attn_logit_softcap=50.0,
    final_logit_softcap=30.0,
    # Gemma omits the key and ties anyway.
    tie_word_embeddings=True,
)

# --- GLM-4 9B (partial + interleaved RoPE, sandwich norms) ---
case(
    "glm4_9b",
    {
        "architectures": ["Glm4ForCausalLM"],
        "model_type": "glm4",
        "hidden_size": 4096,
        "intermediate_size": 13696,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 151552,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    },
    family="glm4",
    head_dim=128,
    num_heads=32,
    num_kv_heads=2,
    partial=0.5,
    is_sandwich=True,
    rope_interleaved=True,
    inv_freq=default_inv_freq(64, 10000.0),
    tie_word_embeddings=False,
)

# --- Phi-3 mini (packed qkv / gate_up; otherwise the Llama shape) ---
case(
    "phi3_mini",
    {
        "architectures": ["Phi3ForCausalLM"],
        "model_type": "phi3",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32064,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
    },
    family="phi3",
    head_dim=96,
    num_heads=32,
    num_kv_heads=32,
    inv_freq=default_inv_freq(96, 10000.0),
    tie_word_embeddings=False,
)

# --- Qwen2-MoE A2.7B (sparse FFN) ---
case(
    "qwen2_moe_a2_7b",
    {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "num_experts": 60,
        "num_experts_per_tok": 4,
        "norm_topk_prob": False,
        "moe_intermediate_size": 1408,
        "shared_expert_intermediate_size": 5632,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    },
    family="qwen2_moe",
    head_dim=128,
    num_heads=16,
    num_kv_heads=16,
    is_moe=True,
    inv_freq=default_inv_freq(128, 1000000.0),
    moe_num_experts=60,
    moe_experts_per_tok=4,
    moe_intermediate_size=1408,
    moe_shared_expert_intermediate_size=5632,
    moe_first_k_dense_replace=0,
    tie_word_embeddings=False,
)

# --- DeepSeek-V2-Lite (MLA + YaRN + fine-grained MoE) ---
_DS_QK_NOPE, _DS_QK_ROPE, _DS_V = 128, 64, 128
_DS_MSCALE = 0.1 * 0.707 * math.log(40.0) + 1.0
case(
    "deepseek_v2_lite",
    {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": 2048,
        "intermediate_size": 10944,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 102400,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "tie_word_embeddings": False,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": _DS_QK_NOPE,
        "qk_rope_head_dim": _DS_QK_ROPE,
        "v_head_dim": _DS_V,
        "n_routed_experts": 64,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "moe_intermediate_size": 1408,
        "first_k_dense_replace": 1,
        "norm_topk_prob": False,
        "routed_scaling_factor": 1.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        },
    },
    family="deepseek_v2",
    head_dim=_DS_QK_NOPE + _DS_QK_ROPE,
    num_heads=16,
    num_kv_heads=16,
    is_moe=True,
    is_mla=True,
    rope_dim=_DS_QK_ROPE,
    rope_interleaved=True,
    inv_freq=yarn_inv_freq(_DS_QK_ROPE, 10000.0, 40.0, 32.0, 1.0, 4096.0),
    attn_scale=(192.0**-0.5) * _DS_MSCALE * _DS_MSCALE,
    mla_qk_nope_head_dim=_DS_QK_NOPE,
    mla_qk_rope_head_dim=_DS_QK_ROPE,
    mla_v_head_dim=_DS_V,
    mla_kv_lora_rank=512,
    moe_num_experts=64,
    moe_experts_per_tok=6,
    moe_intermediate_size=1408,
    moe_shared_expert_intermediate_size=2 * 1408,
    moe_first_k_dense_replace=1,
    tie_word_embeddings=False,
)

# --- Qwen3-VL 8B (VLM wrapper: decoder fields nested under `text_config`, interleaved M-RoPE) ---
case(
    "qwen3_vl_8b",
    {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "tie_word_embeddings": False,
        "image_token_id": 151655,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "max_position_embeddings": 262144,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
        },
        "vision_config": {"model_type": "qwen3_vl", "depth": 27},
    },
    family="qwen3_vl",
    head_dim=128,
    num_heads=32,
    num_kv_heads=8,
    has_qk_norm=True,
    inv_freq=default_inv_freq(128, 5000000.0),
    mrope_section=[24, 20, 20],
    tie_word_embeddings=False,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()
    payload = json.dumps(
        {
            "_generator": "crates/llm/testdata/architectures/generate_regression.py",
            "cases": CASES,
        },
        indent=1,
    )
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()

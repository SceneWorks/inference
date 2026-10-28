//! The KV-cached decode step of the FLUX.2 decoder-LM text encoders' GQA self-attention (the
//! caption-upsampling generate path, sc-6030). The attention itself — bias-less q/k/v/o, per-head
//! q/k RMSNorm for the klein Qwen3 encoder and none for the FLUX.2-dev Mistral encoder (sc-5915),
//! HF half-split RoPE, masked SDPA — is the shared core [`Qwen3TextAttention`] (sc-27326).

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::Array;

use mlx_gen::nn::apply_text_rope;
use mlx_gen::qkv::{repeat_kv, RotationAxes};
use mlx_gen::text_decoder::Qwen3TextAttention;
use mlx_gen::{Error, Result};

use mlx_llm::primitives::{ContiguousKvCache, KvCache};

/// KV-cached **causal** decode step. Mirrors [`Qwen3TextAttention::forward`] but appends this
/// step's K/V to the per-layer growable `cache` and attends causally over the full cached sequence
/// via the implicit bottom-right causal mask (MLX aligns the `q_len` queries to the last cached
/// positions, exactly the joycaption/prompt-refine decode rule). `cos`/`sin` are the RoPE table at
/// the step's **absolute** positions (`TextRope::forward_offset`), so prefill (offset 0,
/// `q_len = prompt_len`) and each 1-token decode (offset = cache length) both index the right
/// frequencies. Used only by the Mistral dev tower (no q/k norm); the shared prologue applies one
/// when present, as `forward` does.
pub(crate) fn forward_step(
    attn: &Qwen3TextAttention,
    x: &Array,
    cos: &Array,
    sin: &Array,
    cache: &mut ContiguousKvCache,
    layer_idx: usize,
) -> Result<Array> {
    let sh = x.shape();
    let (b, s) = (sh[0], sh[1]);
    let (q, k, v) = attn.project_qkv(x)?;

    let q = apply_text_rope(&q, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
    let k = apply_text_rope(&k, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.transpose_axes(&[0, 2, 1, 3])?;
    // Shared growing-concat KV cache (sc-7160). `update` is the persisting append: it concats this
    // step's K/V onto the layer's prior cache along the sequence axis and returns the full K/V.
    let (k_all, v_all) = cache
        .update(layer_idx, &k, &v)
        .map_err(|e| Error::Msg(e.to_string()))?;

    // GQA expand of the post-transpose, post-cache `[b, hkv, s, hd]` K/V.
    let groups = attn.num_heads() / attn.num_kv_heads();
    let k_all = repeat_kv(&k_all, groups, RotationAxes::HeadMajor)?;
    let v_all = repeat_kv(&v_all, groups, RotationAxes::HeadMajor)?;

    let o = scaled_dot_product_attention(
        &q,
        &k_all,
        &v_all,
        attn.scale(),
        ScaledDotProductAttentionMask::Causal,
        None,
    )?;
    let o =
        o.transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, attn.num_heads() * attn.head_dim()])?;
    attn.project_out(&o)
}

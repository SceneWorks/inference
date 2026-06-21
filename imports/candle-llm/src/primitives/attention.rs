//! Attention leaves: grouped-query KV expansion and an eager scaled-dot-product-attention.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, d] -> [b, hkv*groups, s, d]`
//! broadcast (the Candle port of `mlx-llm`'s `repeat_kv`, matching `candle-gen-sensenova`).
//!
//! Candle has no portable fused causal SDPA across CPU/CUDA (flash-attn is a separate, CUDA-only
//! crate), so [`sdpa`] is the eager path — `softmax(scale · QKᵀ + mask) · V` — with the causal mask
//! built explicitly. The mask aligns the `q_len` queries to the bottom-right of the `k_len` cached
//! keys (query row `r` attends keys `0..=offset+r`, where `offset = k_len - q_len`), so cached decode
//! is correct without threading an offset through every call.

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;

use crate::error::Result;

/// Disallowed-attention fill for the additive mask: a large finite negative (matching the
/// candle-gen slices — avoids `-inf` propagation through the softmax kernel).
const MASK_NEG: f32 = -1e30;

/// How attention should be masked.
#[derive(Debug, Clone, Copy)]
pub enum AttnMask<'a> {
    /// No mask (fully bidirectional) — e.g. a vision tower.
    None,
    /// Bottom-right-aligned causal mask: query row `r` attends keys `0..=(k_len - q_len) + r`.
    Causal,
    /// An explicit additive mask broadcast over the score tensor (`0` keep, large-negative block).
    Additive(&'a Tensor),
}

/// Expand grouped-query KV heads to the full query head count.
///
/// `x` is `[batch, n_kv_heads, seq, head_dim]`; the result is `[batch, n_kv_heads * groups, seq,
/// head_dim]` where `groups = n_query_heads / n_kv_heads`. `groups == 1` (MHA) is a no-op clone.
pub fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, hkv, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, hkv, groups, s, d))?
        .contiguous()?
        .reshape((b, hkv * groups, s, d))?)
}

/// Build the additive causal mask `[1, 1, q_len, k_len]` (`0` keep / [`MASK_NEG`] block) for keys
/// that include `offset = k_len - q_len` cached positions before the new queries.
fn causal_mask(q_len: usize, k_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let offset = k_len - q_len;
    let mut data = vec![0f32; q_len * k_len];
    for r in 0..q_len {
        for j in 0..k_len {
            if j > offset + r {
                data[r * k_len + j] = MASK_NEG;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, q_len, k_len), device)?.to_dtype(dtype)?)
}

/// Eager scaled-dot-product attention over `[batch, heads, seq, head_dim]` tensors.
///
/// `scale` is the usual `head_dim^(-0.5)`. `softcap`, when set, applies Gemma-2's score soft-cap
/// (`c·tanh(scores/c)`) after scaling and before masking. K/V must already be GQA-expanded (see
/// [`repeat_kv`]).
pub fn sdpa(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Tensor> {
    let (_b, _h, q_len, _d) = queries.dims4()?;
    let k_len = keys.dim(2)?;
    let mut scores = (queries
        .contiguous()?
        .matmul(&keys.transpose(2, 3)?.contiguous()?)?
        * scale as f64)?;
    if let Some(c) = softcap {
        scores = crate::primitives::nn::soft_cap(&scores, c)?;
    }
    let scores = match mask {
        AttnMask::None => scores,
        AttnMask::Causal => {
            let m = causal_mask(q_len, k_len, scores.dtype(), scores.device())?;
            scores.broadcast_add(&m)?
        }
        AttnMask::Additive(a) => scores.broadcast_add(a)?,
    };
    let weights = softmax_last_dim(&scores)?;
    Ok(weights.matmul(&values.contiguous()?)?)
}

/// Convenience: causal attention with no soft-cap (the decode default).
pub fn sdpa_causal(queries: &Tensor, keys: &Tensor, values: &Tensor, scale: f32) -> Result<Tensor> {
    sdpa(queries, keys, values, scale, None, AttnMask::Causal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arange4(b: usize, h: usize, s: usize, d: usize) -> Tensor {
        let n = (b * h * s * d) as f32;
        Tensor::arange(0f32, n, &Device::Cpu)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
    }

    #[test]
    fn repeat_kv_noop_for_one_group() {
        let x = arange4(1, 2, 3, 4);
        let y = repeat_kv(&x, 1).unwrap();
        assert_eq!(y.dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn repeat_kv_expands_head_axis() {
        let x = arange4(1, 2, 2, 4);
        let y = repeat_kv(&x, 4).unwrap();
        assert_eq!(y.dims(), &[1, 8, 2, 4]);
    }

    #[test]
    fn repeat_kv_duplicates_each_head() {
        // Two KV heads, head_dim 2, seq 1: head0 = [0,1], head1 = [2,3].
        let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 2, 1, 2), &Device::Cpu).unwrap();
        let y = repeat_kv(&x, 2).unwrap(); // [1, 4, 1, 2]
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // groups are adjacent: head0,head0,head1,head1
        assert_eq!(h, vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn sdpa_causal_runs_and_shapes() {
        let q = arange4(1, 1, 2, 4);
        let out = sdpa_causal(&q, &q, &q, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 1, 2, 4]);
    }

    #[test]
    fn causal_mask_blocks_future() {
        // q_len == k_len: a plain lower-triangular mask.
        let m = causal_mask(3, 3, DType::F32, &Device::Cpu)
            .unwrap()
            .reshape((3, 3))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(m[0][1], MASK_NEG); // future blocked
        assert_eq!(m[1][0], 0.0); // past attended
        assert_eq!(m[2][2], 0.0); // self attended
    }
}

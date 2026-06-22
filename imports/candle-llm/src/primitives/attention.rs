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
//!
//! With the `flash-attn` feature, [`sdpa`] first tries the fused FlashAttention-2 kernel
//! ([`candle_flash_attn::flash_attn`]) for the dense causal/bidirectional path and falls back to the
//! eager kernel for everything it cannot serve — Gemma-2 score soft-cap, MLA's mismatched q/v head
//! dims, an explicit additive (padded-batch) mask, or a non-f16/bf16 dtype. FlashAttention's causal
//! masking is bottom-right aligned (`window_size_right = 0`), matching [`causal_mask`]'s convention,
//! so cached decode stays correct. Numerics differ by a few half-precision ULPs from the eager path
//! (different reduction order), the same tolerance the batched / prefix-reuse GPU paths carry.

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
    #[cfg(feature = "flash-attn")]
    if let Some(out) = try_flash_attn(queries, keys, values, scale, softcap, mask)? {
        return Ok(out);
    }
    sdpa_eager(queries, keys, values, scale, softcap, mask)
}

/// The eager `softmax(scale · QKᵀ + mask) · V` path — the portable fallback [`sdpa`] runs when the
/// fused kernel is unavailable or ineligible. Kept as its own entry point so the flash-vs-eager
/// parity test can compare the two paths directly on one device.
fn sdpa_eager(
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

/// Try the fused FlashAttention-2 kernel; `Ok(None)` means "this case is not flash-eligible, use the
/// eager path". Inputs are the same `[batch, heads, seq, head_dim]` tensors [`sdpa`] takes (K/V
/// already GQA-expanded); the kernel wants `[batch, seq, heads, head_dim]`, so q/k/v are transposed
/// in and the result transposed back. Eligibility mirrors the kernel's own constraints: f16/bf16
/// only, `head_dim ≤ 512` and a multiple of 8, equal q/k/v head dims, no soft-cap, and a causal or
/// no-op mask (an explicit additive mask is left to the eager path).
#[cfg(feature = "flash-attn")]
fn try_flash_attn(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Option<Tensor>> {
    // Soft-cap (Gemma-2) and explicit additive (padded-batch) masks are not on the fused path.
    if softcap.is_some() {
        return Ok(None);
    }
    let causal = match mask {
        AttnMask::None => false,
        AttnMask::Causal => true,
        AttnMask::Additive(_) => return Ok(None),
    };
    // The kernel is CUDA-only and f16/bf16-only.
    if !queries.device().is_cuda() {
        return Ok(None);
    }
    match queries.dtype() {
        DType::F16 | DType::BF16 => {}
        _ => return Ok(None),
    }
    // Equal, kernel-supported head dims (excludes MLA's q=192 / v=128 split).
    let d = queries.dim(3)?;
    if d % 8 != 0 || d > 512 || keys.dim(3)? != d || values.dim(3)? != d {
        return Ok(None);
    }
    // [b, h, s, d] -> [b, s, h, d] (last dim stays contiguous, which the kernel requires).
    let q = queries.transpose(1, 2)?;
    let k = keys.transpose(1, 2)?;
    let v = values.transpose(1, 2)?;
    let out = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?; // [b, s, h, d]
    Ok(Some(out.transpose(1, 2)?.contiguous()?))
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

    /// On CUDA, the fused FlashAttention-2 kernel must agree with the eager path within a few
    /// half-precision ULPs — both for full-prompt causal attention and for the bottom-right-aligned
    /// decode shape (`q_len = 1`, `k_len > 1`). Needs `--features flash-attn` (which implies `cuda`);
    /// otherwise `sdpa` is the eager path and this would compare eager to eager, so it is gated off.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_attn_matches_eager_on_cuda() {
        let device = Device::new_cuda(0).expect("cuda device");
        // Bounded, varied bf16 q/k/v (cos keeps values in [-1, 1] so the softmax doesn't saturate).
        let mk = |b, h, s, d, phase: f64| {
            let n = (b * h * s * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((b, h, s, d))
                .unwrap()
                .affine(0.013, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let (b, h, s, d) = (2, 4, 48, 64);
        let scale = (d as f32).powf(-0.5);

        let max_abs_diff = |a: &Tensor, e: &Tensor| {
            (a.to_dtype(DType::F32).unwrap() - e.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        // Full-prompt causal.
        let (q, k, v) = (
            mk(b, h, s, d, 0.0),
            mk(b, h, s, d, 1.7),
            mk(b, h, s, d, 3.1),
        );
        assert!(
            try_flash_attn(&q, &k, &v, scale, None, AttnMask::Causal)
                .unwrap()
                .is_some(),
            "test inputs must be flash-eligible, else this proves nothing"
        );
        let diff = max_abs_diff(
            &sdpa(&q, &k, &v, scale, None, AttnMask::Causal).unwrap(),
            &sdpa_eager(&q, &k, &v, scale, None, AttnMask::Causal).unwrap(),
        );
        assert!(diff < 3e-2, "full-prompt flash vs eager max|Δ| = {diff}");

        // Decode shape: one query against the full key/value run (bottom-right causal alignment).
        let q1 = mk(b, h, 1, d, 0.0);
        let diff = max_abs_diff(
            &sdpa(&q1, &k, &v, scale, None, AttnMask::Causal).unwrap(),
            &sdpa_eager(&q1, &k, &v, scale, None, AttnMask::Causal).unwrap(),
        );
        assert!(diff < 3e-2, "decode-shape flash vs eager max|Δ| = {diff}");
    }
}

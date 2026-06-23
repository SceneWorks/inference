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
//!
//! The continuous-batching `Throughput` path (story 7347) decodes many sequences at once over
//! per-sequence paged caches; its attention used to be an N-call per-sequence SDPA loop, which
//! flatlined throughput at occupancy (the cost is N kernel launches + N gathers, not per-kernel
//! speed). [`try_flash_attn_varlen`] (story 7351) folds that loop into **one**
//! [`candle_flash_attn::flash_attn_varlen`] call over the ragged (gathered) KV of all active
//! sequences — no padding mask, no per-sequence launch — packed via cumulative `cu_seqlens` offsets.
//! It is grouped-query-native (K/V passed un-expanded) and bottom-right causal; the eager per-sequence
//! loop stays the fallback for the cases varlen cannot serve (soft-cap, f32/CPU, no `flash-attn`).

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

/// Batched per-sequence ("ragged") attention through a **single** [`candle_flash_attn::flash_attn_varlen`]
/// call (story 7351) — the continuous-batching `Throughput` decode's attention without the
/// per-sequence loop.
///
/// `q` is the batched projection `[batch, heads, s, head_dim]` (every row the same query length `s`,
/// as the decode step produces). `kv` is one `(k_all, v_all)` per sequence **in batch order**, each
/// the paged gather `[1, n_kv_heads, lₖ, head_dim]` at that sequence's own cached length `lₖ`. The
/// kernel handles grouped-query natively (`n_kv_heads` divides `heads`), so K/V are passed
/// **un-expanded** — no [`repeat_kv`]. Sequences are packed into the kernel's flat layout
/// (`q`: `[Σ s, heads, head_dim]`, `k`/`v`: `[Σ lₖ, n_kv_heads, head_dim]`) via cumulative `cu_seqlens`
/// offsets, with no padding mask. Attention is bottom-right causal (`window_size_right = 0`), the
/// decode convention (query row `r` of a sequence attends its keys `0..=(lₖ - s) + r`).
///
/// Returns `Some([batch, heads, s, head_dim])` when varlen is eligible — mirroring [`try_flash_attn`]:
/// CUDA, f16/bf16, `head_dim` a multiple of 8 and `≤ 512`, no soft-cap — else `Ok(None)` so the caller
/// runs the eager per-sequence fallback. Numerics differ from the eager loop by a few half-precision
/// ULPs (different reduction order), the tolerance the `Throughput` path already carries.
#[cfg(feature = "flash-attn")]
pub(crate) fn try_flash_attn_varlen(
    q: &Tensor,
    kv: &[(Tensor, Tensor)],
    scale: f32,
    softcap: Option<f32>,
) -> Result<Option<Tensor>> {
    // Soft-cap (Gemma-2) is not on the fused path; nor is CPU / non-half-precision.
    if softcap.is_some() || !q.device().is_cuda() {
        return Ok(None);
    }
    match q.dtype() {
        DType::F16 | DType::BF16 => {}
        _ => return Ok(None),
    }
    let (b, h, s, d) = q.dims4()?;
    if d % 8 != 0 || d > 512 {
        return Ok(None);
    }
    debug_assert_eq!(b, kv.len(), "one (k, v) gather per sequence");

    // Pack the per-sequence gathers into the kernel's flat ragged layout (the data-movement a
    // block-table kernel would remove — kept as its own seam so the sc-7258 cost bench can time the
    // build separately from the kernel).
    let vi = build_varlen_inputs(q, kv)?;

    // One kernel over all sequences; bottom-right causal (window_size_right = 0). Output: [b*s, h, d].
    let out = candle_flash_attn::flash_attn_varlen(
        &vi.q, &vi.k, &vi.v, &vi.cu_q, &vi.cu_k, vi.max_q, vi.max_k, scale, true,
    )?;
    // Back to [b, heads, s, d] for the shared output projection.
    Ok(Some(
        out.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?,
    ))
}

/// The flat ragged tensors [`candle_flash_attn::flash_attn_varlen`] consumes, packed from the
/// per-sequence gathers by [`build_varlen_inputs`].
#[cfg(feature = "flash-attn")]
pub(crate) struct VarlenInputs {
    /// Sequence-major queries `[Σ s, heads, head_dim]`.
    pub q: Tensor,
    /// Ragged keys `[Σ lₖ, n_kv_heads, head_dim]` (un-expanded — the kernel is GQA-native).
    pub k: Tensor,
    /// Ragged values `[Σ lₖ, n_kv_heads, head_dim]`.
    pub v: Tensor,
    /// Cumulative query offsets `[b + 1]` (u32).
    pub cu_q: Tensor,
    /// Cumulative key offsets `[b + 1]` (u32).
    pub cu_k: Tensor,
    /// Longest query run (the uniform decode step `s`).
    pub max_q: usize,
    /// Longest cached key run.
    pub max_k: usize,
}

/// Pack the per-sequence gathers into the flat layout [`candle_flash_attn::flash_attn_varlen`] wants.
///
/// `q` is the batched projection `[b, heads, s, head_dim]`; `kv` is one paged gather
/// `[1, n_kv_heads, lₖ, head_dim]` per sequence in batch order. Queries become sequence-major rows
/// (`[Σ s, heads, head_dim]`, sequence `i` owning rows `i*s .. (i+1)*s`); keys/values are squeezed,
/// transposed to `[lₖ, n_kv_heads, head_dim]`, and concatenated along the token axis with cumulative
/// `cu_seqlens` offsets. This is exactly the host-side gather/layout work a custom block-table kernel
/// (story 7258) would subsume by reading the scattered blocks directly.
#[cfg(feature = "flash-attn")]
pub(crate) fn build_varlen_inputs(q: &Tensor, kv: &[(Tensor, Tensor)]) -> Result<VarlenInputs> {
    let (b, h, s, d) = q.dims4()?;
    let device = q.device();

    // Ragged queries: [b, heads, s, d] -> [b, s, heads, d] -> [b*s, heads, d].
    let q_ragged = q.transpose(1, 2)?.contiguous()?.reshape((b * s, h, d))?;

    // Ragged keys/values: each gather [1, kvh, lₖ, d] -> [lₖ, kvh, d]; concatenate along the token
    // axis into [Σ lₖ, kvh, d], accumulating the cumulative key offsets as we go.
    let mut ks = Vec::with_capacity(kv.len());
    let mut vs = Vec::with_capacity(kv.len());
    let mut cu_k = Vec::with_capacity(kv.len() + 1);
    cu_k.push(0u32);
    let mut acc = 0u32;
    let mut max_k = 0usize;
    for (k_all, v_all) in kv {
        let lk = k_all.dim(2)?;
        max_k = max_k.max(lk);
        acc += lk as u32;
        cu_k.push(acc);
        ks.push(k_all.squeeze(0)?.transpose(0, 1)?.contiguous()?); // [lₖ, kvh, d]
        vs.push(v_all.squeeze(0)?.transpose(0, 1)?.contiguous()?);
    }
    let k_ragged = cat_rows(ks)?;
    let v_ragged = cat_rows(vs)?;

    // Cumulative query offsets: every sequence contributes the same `s` queries (uniform decode step).
    let cu_q: Vec<u32> = (0..=b).map(|i| (i * s) as u32).collect();
    let cu_q = Tensor::from_vec(cu_q, (b + 1,), device)?;
    let cu_k = Tensor::from_vec(cu_k, (kv.len() + 1,), device)?;

    Ok(VarlenInputs {
        q: q_ragged,
        k: k_ragged,
        v: v_ragged,
        cu_q,
        cu_k,
        max_q: s,
        max_k,
    })
}

/// Concatenate tensors along axis 0; a single part is returned as-is (no copy). Used to pack the
/// per-sequence ragged K/V rows for [`try_flash_attn_varlen`].
#[cfg(feature = "flash-attn")]
fn cat_rows(parts: Vec<Tensor>) -> Result<Tensor> {
    if parts.len() == 1 {
        return Ok(parts.into_iter().next().unwrap());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 0)?)
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

    /// On CUDA, the batched `flash_attn_varlen` ragged path (story 7351) must agree with the eager
    /// per-sequence SDPA loop it replaces — over **differing-length** sequences (the continuous
    /// `Throughput` decode), grouped-query (`n_kv_heads < heads`), one decode query per sequence.
    /// Needs `--features flash-attn`; otherwise `try_flash_attn_varlen` is not compiled and there is
    /// nothing to compare, so it is gated off.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_attn_varlen_matches_eager_per_seq_on_cuda() {
        let device = Device::new_cuda(0).expect("cuda device");
        let (h, kvh, d) = (4usize, 2usize, 64usize); // GQA: groups = 2
        let groups = h / kvh;
        let scale = (d as f32).powf(-0.5);
        let lens = [3usize, 7, 1, 16]; // ragged per-sequence cached lengths
        let b = lens.len();

        // Bounded, varied bf16 `[1, heads, rows, d]` (cos keeps the softmax off saturation).
        let mk = |rows: usize, heads: usize, phase: f64| {
            let n = (rows * heads * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((1, heads, rows, d))
                .unwrap()
                .affine(0.011, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };

        // One decode query per sequence: q [b, heads, 1, d].
        let qs: Vec<Tensor> = (0..b).map(|i| mk(1, h, i as f64 * 0.3)).collect();
        let q = Tensor::cat(&qs.iter().collect::<Vec<_>>(), 0).unwrap();
        // Per-sequence gathered KV at the ragged lengths (kv-head count, un-expanded).
        let kv: Vec<(Tensor, Tensor)> = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| (mk(l, kvh, 1.0 + i as f64), mk(l, kvh, 5.0 + i as f64)))
            .collect();

        let got = try_flash_attn_varlen(&q, &kv, scale, None)
            .unwrap()
            .expect("ragged inputs must be varlen-eligible, else this proves nothing");

        // Eager per-sequence reference: expand GQA and run a stock causal SDPA per sequence.
        let mut outs = Vec::with_capacity(b);
        for (i, (k, v)) in kv.iter().enumerate() {
            let qi = q.narrow(0, i, 1).unwrap();
            let k = repeat_kv(k, groups).unwrap();
            let v = repeat_kv(v, groups).unwrap();
            outs.push(sdpa_eager(&qi, &k, &v, scale, None, AttnMask::Causal).unwrap());
        }
        let want = Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0).unwrap();

        let diff = (got.to_dtype(DType::F32).unwrap() - want.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 3e-2, "varlen vs eager per-seq max|Δ| = {diff}");
    }

    /// **sc-7258 go/no-go cost bench.** Within the continuous `Throughput` per-sequence attention path,
    /// how much of the per-layer, per-step cost is the **gather + ragged build** (the host-side data
    /// movement a custom block-table kernel would subsume by reading scattered blocks directly) versus
    /// the **`flash_attn_varlen` kernel** itself (what a custom kernel would still have to do at least
    /// as fast to win)? The `gather+build` fraction is the *ceiling* a from-scratch paged kernel could
    /// reclaim from attention — the number the story is gated on.
    ///
    /// Each phase is `synchronize()`-bracketed so its GPU work is attributed to it, at the two real
    /// test-model head shapes across occupancy (N) and context (L). Needs `--features flash-attn` +
    /// CUDA; `#[ignore]`d (a bench, not a correctness gate).
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "sc-7258 cost bench; needs CUDA"]
    fn paged_attention_path_cost_on_cuda() {
        use crate::primitives::kv_cache::KvCache;
        use crate::primitives::PagedKvCache;
        use std::time::Instant;

        let device = Device::new_cuda(0).expect("cuda device");
        let block_size = 16usize;
        let iters = 30usize;
        let warmup = 8usize;

        // Bounded, varied bf16 `[1, heads, rows, d]` (cos keeps values in [-1, 1]).
        let mk = |rows: usize, heads: usize, d: usize, phase: f64| {
            let n = (rows * heads * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((1, heads, rows, d))
                .unwrap()
                .affine(0.011, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let cat0 = |parts: Vec<Tensor>| Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0).unwrap();

        // (label, heads, kv_heads, head_dim) for the two real test models.
        for (label, h, kvh, d) in [("SmolLM2", 9usize, 3usize, 64usize), ("Qwen3", 16, 8, 128)] {
            let scale = (d as f32).powf(-0.5);
            println!("[{label}] H={h} KVH={kvh} D={d} (per-layer, per decode step):");
            for &n in &[4usize, 8, 16] {
                for &l in &[64usize, 256] {
                    // N single-sequence paged caches (1 layer), each prefilled to length L.
                    let mut caches: Vec<PagedKvCache> =
                        (0..n).map(|_| PagedKvCache::new(1, block_size)).collect();
                    for (i, c) in caches.iter_mut().enumerate() {
                        for t in 0..l {
                            let p = (i * 7 + t) as f64 * 0.01;
                            c.update(0, &mk(1, kvh, d, p), &mk(1, kvh, d, p + 3.0))
                                .unwrap();
                        }
                    }
                    // This step's batched projection: q [n, H, 1, d], new k/v [n, KVH, 1, d].
                    let q = cat0((0..n).map(|i| mk(1, h, d, i as f64 * 0.3)).collect());
                    let k_step = cat0((0..n).map(|i| mk(1, kvh, d, 9.0 + i as f64)).collect());
                    let v_step = cat0((0..n).map(|i| mk(1, kvh, d, 13.0 + i as f64)).collect());

                    let (mut t_gather, mut t_build, mut t_kernel) = (0f64, 0f64, 0f64);
                    for it in 0..(warmup + iters) {
                        // gather: per-seq narrow+contiguous of the projection, append + paged gather.
                        device.synchronize().unwrap();
                        let t0 = Instant::now();
                        let mut gathered = Vec::with_capacity(n);
                        for (i, c) in caches.iter_mut().enumerate() {
                            let ki = k_step.narrow(0, i, 1).unwrap().contiguous().unwrap();
                            let vi = v_step.narrow(0, i, 1).unwrap().contiguous().unwrap();
                            gathered.push(c.update(0, &ki, &vi).unwrap());
                        }
                        device.synchronize().unwrap();
                        let t1 = Instant::now();
                        // build: pack the ragged varlen layout.
                        let vin = super::build_varlen_inputs(&q, &gathered).unwrap();
                        device.synchronize().unwrap();
                        let t2 = Instant::now();
                        // kernel: the one flash_attn_varlen call.
                        let _out = candle_flash_attn::flash_attn_varlen(
                            &vin.q, &vin.k, &vin.v, &vin.cu_q, &vin.cu_k, vin.max_q, vin.max_k,
                            scale, true,
                        )
                        .unwrap();
                        device.synchronize().unwrap();
                        let t3 = Instant::now();
                        if it >= warmup {
                            t_gather += (t1 - t0).as_secs_f64();
                            t_build += (t2 - t1).as_secs_f64();
                            t_kernel += (t3 - t2).as_secs_f64();
                        }
                    }
                    let us = |s: f64| s / iters as f64 * 1e6;
                    let (g, bd, kn) = (us(t_gather), us(t_build), us(t_kernel));
                    let removable = 100.0 * (g + bd) / (g + bd + kn);
                    println!(
                        "  N={n:<2} L={l:<4} gather {g:7.1}us | build {bd:7.1}us | \
                         kernel {kn:7.1}us | gather+build = {removable:3.0}% of attn"
                    );
                }
            }
        }
    }
}

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
//! is correct without threading an offset through every call. The mask is cheap to *ask* for but not
//! to build (host vec fill + upload), so the eager path skips it entirely for the all-zeros decode
//! shape (`q_len == 1`) and memoizes the prefill mask per `(q_len, k_len, dtype, device)` — one build
//! per forward instead of one per decoder layer (sc-12458).
//!
//! With the `flash-attn` feature, [`sdpa`] first tries the fused FlashAttention-2 kernel
//! (`candle_flash_attn::flash_attn`) for the dense causal/bidirectional path and falls back to the
//! eager kernel for everything it cannot serve — Gemma-2 score soft-cap, MLA's mismatched q/v head
//! dims, an explicit additive (padded-batch) mask, or a non-f16/bf16 dtype. FlashAttention's causal
//! masking is bottom-right aligned (`window_size_right = 0`), matching `causal_mask`'s convention,
//! so cached decode stays correct. Numerics differ by a few half-precision ULPs from the eager path
//! (different reduction order), the same tolerance the batched / prefix-reuse GPU paths carry.
//!
//! The continuous-batching `Throughput` path (story 7347) decodes many sequences at once over
//! per-sequence paged caches; its attention used to be an N-call per-sequence SDPA loop, which
//! flatlined throughput at occupancy (the cost is N kernel launches + N gathers, not per-kernel
//! speed). `try_flash_attn_varlen` (story 7351) folds that loop into **one**
//! `candle_flash_attn::flash_attn_varlen` call over the ragged (gathered) KV of all active
//! sequences — no padding mask, no per-sequence launch — packed via cumulative `cu_seqlens` offsets.
//! It is grouped-query-native (K/V passed un-expanded) and bottom-right causal; the eager per-sequence
//! loop stays the fallback for the cases varlen cannot serve (soft-cap, f32/CPU, no `flash-attn`).

use std::sync::Mutex;

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
    #[cfg(test)]
    CAUSAL_MASK_BUILDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

/// Number of host-side [`causal_mask`] builds — lets tests pin "decode builds no mask" and "prefill
/// builds the mask once per forward, not once per layer" (sc-12458).
#[cfg(test)]
static CAUSAL_MASK_BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The one memoized causal mask (sc-12458). Every decoder layer of a forward asks [`sdpa_eager`] for
/// the identical `AttnMask::Causal` mask, so without memoization the host-side vec fill + upload in
/// [`causal_mask`] ran once **per layer** per forward. A single entry suffices: within one forward
/// the key `(q_len, k_len, dtype, device)` is constant across layers (it changes at most at a shard
/// boundary on a multi-device model), so a prefill builds the mask once and every later layer clones
/// the cached handle. Decode steps (`q_len == 1`) never reach this — see [`sdpa_eager`].
struct CausalMaskEntry {
    q_len: usize,
    k_len: usize,
    dtype: DType,
    device: Device,
    mask: Tensor,
}

static CAUSAL_MASK_CACHE: Mutex<Option<CausalMaskEntry>> = Mutex::new(None);

/// [`causal_mask`] behind the single-entry memo: returns the cached tensor when the key matches,
/// else builds (bit-identical values — same builder) and replaces the entry.
fn cached_causal_mask(q_len: usize, k_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut guard = CAUSAL_MASK_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(e) = guard.as_ref() {
        if e.q_len == q_len && e.k_len == k_len && e.dtype == dtype && e.device.same_device(device) {
            return Ok(e.mask.clone());
        }
    }
    let mask = causal_mask(q_len, k_len, dtype, device)?;
    *guard = Some(CausalMaskEntry {
        q_len,
        k_len,
        dtype,
        device: device.clone(),
        mask: mask.clone(),
    });
    Ok(mask)
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
        // A single-query bottom-right-aligned causal mask is provably all zeros (its one row `r = 0`
        // blocks `j > (k_len - 1) + 0`, unsatisfiable for `j < k_len`), so the decode step skips the
        // mask build entirely — no host allocation, no upload, no broadcast_add of zeros (sc-12458).
        // Softmax is invariant to adding exact zeros, so this is bit-identical to the masked path.
        AttnMask::Causal if q_len == 1 => scores,
        AttnMask::Causal => {
            let m = cached_causal_mask(q_len, k_len, scores.dtype(), scores.device())?;
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
/// call (stories 7351 + 7453) — the continuous-batching `Throughput` decode's attention without the
/// per-sequence loop.
///
/// `q` is the batched projection `[batch, heads, s, head_dim]` (every row the same query length `s`,
/// as the decode step produces). `k_ragged`/`v_ragged` are the **already-gathered** keys/values of
/// every active sequence packed token-major into `[Σ lₖ, n_kv_heads, head_dim]` (sequence `i` owning
/// the rows `cu_k[i] .. cu_k[i+1]`), as the pooled [`BlockPool`](crate::primitives::BlockPool) gather
/// produces in **one** `index_select` — no per-sequence `squeeze`/`transpose`/`cat`. The kernel
/// handles grouped-query natively (`n_kv_heads` divides `heads`), so K/V stay **un-expanded** — no
/// [`repeat_kv`]. Queries are packed to `[Σ s, heads, head_dim]`; attention is bottom-right causal
/// (`window_size_right = 0`), the decode convention (query row `r` of a sequence attends its keys
/// `0..=(lₖ - s) + r`), with no padding mask.
///
/// `cu_k` is the cumulative key-offset table `[b + 1]` (u32-valued; `cu_k[0] == 0`,
/// `cu_k[b] == Σ lₖ`) and `max_k` the longest cached key run. Returns `Some([batch, heads, s,
/// head_dim])` when varlen is eligible — mirroring [`try_flash_attn`]: CUDA, f16/bf16, `head_dim` a
/// multiple of 8 and `≤ 512`, no soft-cap — else `Ok(None)` so the caller runs the eager per-sequence
/// fallback. Numerics differ from the eager loop by a few half-precision ULPs (different reduction
/// order), the tolerance the `Throughput` path already carries.
#[cfg(feature = "flash-attn")]
pub(crate) fn try_flash_attn_varlen(
    q: &Tensor,
    k_ragged: &Tensor,
    v_ragged: &Tensor,
    cu_k: &[u32],
    max_k: usize,
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
    debug_assert_eq!(
        cu_k.len(),
        b + 1,
        "cu_k is one cumulative offset per sequence + 1"
    );
    let device = q.device();

    // Sequence-major queries: [b, heads, s, d] -> [b, s, heads, d] -> [b*s, heads, d].
    let q_ragged = q.transpose(1, 2)?.contiguous()?.reshape((b * s, h, d))?;
    // Cumulative query offsets: every sequence contributes the same `s` queries (uniform decode step).
    let cu_q: Vec<u32> = (0..=b).map(|i| (i * s) as u32).collect();
    let cu_q = Tensor::from_vec(cu_q, (b + 1,), device)?;
    let cu_k_t = Tensor::from_vec(cu_k.to_vec(), (b + 1,), device)?;

    // One kernel over all sequences; bottom-right causal (window_size_right = 0). Output: [b*s, h, d].
    let out = candle_flash_attn::flash_attn_varlen(
        &q_ragged, k_ragged, v_ragged, &cu_q, &cu_k_t, s, max_k, scale, true,
    )?;
    // Back to [b, heads, s, d] for the shared output projection.
    Ok(Some(
        out.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?,
    ))
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

    /// Reset the sc-12458 memo + build counter so a test observes only its own builds. Tests run
    /// single-threaded here (`RUST_TEST_THREADS=1` is forced), so this is race-free.
    fn reset_mask_accounting() {
        *CAUSAL_MASK_CACHE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        CAUSAL_MASK_BUILDS.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    fn mask_builds() -> usize {
        CAUSAL_MASK_BUILDS.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Bounded, varied f32 CPU tensor (cos keeps values in [-1, 1] so the softmax is well-behaved).
    fn varied4(b: usize, h: usize, s: usize, d: usize, phase: f64) -> Tensor {
        let n = (b * h * s * d) as f32;
        Tensor::arange(0f32, n, &Device::Cpu)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
            .affine(0.013, phase)
            .unwrap()
            .cos()
            .unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// sc-12458: the decode shape (`q_len == 1`, bottom-right causal) must build **no** mask — and be
    /// bit-identical to the old always-mask path (the mask it skips is provably all zeros).
    #[test]
    fn decode_step_builds_no_mask_and_matches_masked_path() {
        let (b, h, s, d) = (2, 3, 1, 4);
        let k_len = 9;
        let q = varied4(b, h, s, d, 0.0);
        let k = varied4(b, h, k_len, d, 1.7);
        let v = varied4(b, h, k_len, d, 3.1);

        reset_mask_accounting();
        let got = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Causal).unwrap();
        assert_eq!(mask_builds(), 0, "decode step must not build a causal mask");

        // Old path: the explicitly built mask, applied via the untouched Additive branch.
        let m = causal_mask(s, k_len, DType::F32, &Device::Cpu).unwrap();
        let want = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Additive(&m)).unwrap();
        assert_eq!(bits(&got), bits(&want), "decode skip must be bit-identical");
    }

    /// sc-12458: repeated same-shape causal SDPA calls (the per-layer loop of one prefill forward)
    /// build the mask **once**, and every call is bit-identical to the explicitly masked path.
    #[test]
    fn prefill_builds_mask_once_across_layers() {
        let (b, h, q_len, d) = (1, 2, 5, 4);
        let k_len = 8; // chunked/continuation prefill: cached positions ahead of the new queries
        let q = varied4(b, h, q_len, d, 0.4);
        let k = varied4(b, h, k_len, d, 2.2);
        let v = varied4(b, h, k_len, d, 4.9);

        let m = causal_mask(q_len, k_len, DType::F32, &Device::Cpu).unwrap();
        let want = bits(&sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Additive(&m)).unwrap());

        reset_mask_accounting();
        for layer in 0..4 {
            let got = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Causal).unwrap();
            assert_eq!(bits(&got), want, "layer {layer} must match the masked path");
        }
        assert_eq!(mask_builds(), 1, "one mask build for the whole forward");
    }

    /// sc-12458: a different `(q_len, k_len)` (e.g. the next request's prefill, or a speculative
    /// `q_len > 1` continuation at a moved offset) must rebuild rather than reuse a stale mask.
    #[test]
    fn cached_mask_rebuilds_on_shape_change() {
        reset_mask_accounting();
        let a = cached_causal_mask(3, 3, DType::F32, &Device::Cpu).unwrap();
        let _ = cached_causal_mask(3, 3, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask_builds(), 1, "same key is served from the memo");

        let b = cached_causal_mask(3, 7, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask_builds(), 2, "new key must rebuild");
        assert_eq!(a.dims(), &[1, 1, 3, 3]);
        assert_eq!(b.dims(), &[1, 1, 3, 7]);
        // And the rebuilt mask carries the correct bottom-right alignment (offset = 4).
        let rows = b.reshape((3, 7)).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(rows[0][4], 0.0); // j == offset + r: attended
        assert_eq!(rows[0][5], MASK_NEG); // j > offset + r: blocked
        assert_eq!(rows[2][6], 0.0); // last row attends everything

        let c = cached_causal_mask(3, 7, DType::F16, &Device::Cpu).unwrap();
        assert_eq!(mask_builds(), 3, "dtype is part of the key");
        assert_eq!(c.dtype(), DType::F16);
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

        // Pack the per-sequence gathers into the pooled ragged layout the BlockPool gather produces:
        // each [1, kvh, lₖ, d] -> token-major [lₖ, kvh, d], concatenated into [Σ lₖ, kvh, d] with the
        // cumulative `cu_k` offsets.
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        let mut cu_k = vec![0u32];
        let mut acc = 0u32;
        let mut max_k = 0usize;
        for (k, v) in &kv {
            let l = k.dim(2).unwrap();
            acc += l as u32;
            cu_k.push(acc);
            max_k = max_k.max(l);
            ks.push(
                k.squeeze(0)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            );
            vs.push(
                v.squeeze(0)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            );
        }
        let k_ragged = Tensor::cat(&ks.iter().collect::<Vec<_>>(), 0).unwrap();
        let v_ragged = Tensor::cat(&vs.iter().collect::<Vec<_>>(), 0).unwrap();

        let got = try_flash_attn_varlen(&q, &k_ragged, &v_ragged, &cu_k, max_k, scale, None)
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

    /// **sc-7453 → sc-7467 cost-split bench.** Within the continuous `Throughput` per-sequence
    /// attention path, split the per-layer, per-step cost into **write_loop** (the pre-7467 `O(N)`
    /// per-sequence in-place `slice_set` of the step's new token), **write_scatter** (story 7467's
    /// **one** in-place `scatter_set` per side over the same slots — the production path), **gather**
    /// (the **single** pooled `index_select` over every active sequence's token slots), **build** (the
    /// `q`-packing + `cu_seqlens` host/device prep), and the **`flash_attn_varlen` kernel**.
    ///
    /// Two collapses are visible here. sc-7453 already replaced an `O(N · blocks)` per-sequence `cat`
    /// gather (~99% of the path — the sc-7258 finding) with one `index_select`, so **gather** is now a
    /// **flat** ~70–110 µs regardless of N. That left the per-sequence **write** as the residual
    /// launch-latency cost (`O(N)` `slice_set` launches, ~86–95% of the sync-bracketed path at N=16);
    /// sc-7467 collapses it too — **write_scatter** is one `scatter_set` per side, so it should be flat
    /// in N like the gather while **write_loop** climbs. Each phase is `synchronize()`-bracketed, so
    /// these columns are launch-latency-dominated; across the real decode's many layers the writes
    /// pipeline on the stream, and the realized end-to-end throughput is `attention_bottleneck_bound`.
    /// Run at the two real test-model head shapes across occupancy (N) and context (L). Needs
    /// `--features flash-attn` + CUDA; `#[ignore]`d (a bench, not a gate).
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "sc-7453/7467 cost-split bench; needs CUDA"]
    fn paged_attention_path_cost_on_cuda() {
        use crate::primitives::kv_cache::KvCache;
        use crate::primitives::{BlockPool, PagedKvCache};
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
                    // N single-sequence paged caches over ONE shared pool (1 layer), each prefilled to
                    // length L — so the batched gather spans every sequence's blocks in one pool tensor.
                    let pool = BlockPool::new(block_size);
                    let mut caches: Vec<PagedKvCache> = (0..n)
                        .map(|_| PagedKvCache::with_pool(pool.clone(), 1))
                        .collect();
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

                    let cols = kvh * d;
                    let (mut t_wloop, mut t_wscat, mut t_gather, mut t_build, mut t_kernel) =
                        (0f64, 0f64, 0f64, 0f64, 0f64);
                    for it in 0..(warmup + iters) {
                        // write_loop (pre-7467): reserve the step + O(N) per-sequence in-place slice_set.
                        device.synchronize().unwrap();
                        let t0 = Instant::now();
                        for (i, c) in caches.iter_mut().enumerate() {
                            c.reserve_step(1).unwrap();
                            let ki = k_step.narrow(0, i, 1).unwrap();
                            let vi = v_step.narrow(0, i, 1).unwrap();
                            c.write_step_layer(0, &ki, &vi).unwrap();
                        }
                        device.synchronize().unwrap();
                        let t1 = Instant::now();
                        // write_scatter (story 7467): ONE in-place scatter_set per side over the same
                        // just-reserved slots — the production path. The slot index is broadcast across
                        // the kvh*d columns the scatter preserves (built once per step in reality).
                        let mut wslots: Vec<u32> = Vec::new();
                        for c in &caches {
                            for &slot in c.new_token_slots() {
                                for _ in 0..cols {
                                    wslots.push(slot);
                                }
                            }
                        }
                        let total_new = wslots.len() / cols;
                        let w_index =
                            Tensor::from_vec(wslots, (total_new, kvh, d), &device).unwrap();
                        let k_sc = k_step
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((total_new, kvh, d))
                            .unwrap();
                        let v_sc = v_step
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((total_new, kvh, d))
                            .unwrap();
                        pool.borrow()
                            .scatter_write(0, &w_index, &k_sc, &v_sc)
                            .unwrap();
                        device.synchronize().unwrap();
                        let t1s = Instant::now();
                        // gather: ONE index_select over every sequence's token slots, concatenated.
                        let mut idx: Vec<u32> = Vec::new();
                        let mut cu_k = vec![0u32];
                        let mut acc = 0u32;
                        let mut max_k = 0usize;
                        for c in &caches {
                            let ts = c.token_slots();
                            idx.extend_from_slice(ts);
                            acc += ts.len() as u32;
                            cu_k.push(acc);
                            max_k = max_k.max(ts.len());
                        }
                        let index = Tensor::from_vec(idx, (acc as usize,), &device).unwrap();
                        let (k_rag, v_rag) = pool.borrow().gather(0, &index).unwrap();
                        device.synchronize().unwrap();
                        let t2 = Instant::now();
                        // build: pack q sequence-major + cumulative offsets.
                        let q_ragged = q
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((n, h, d))
                            .unwrap();
                        let cu_q: Vec<u32> = (0..=n).map(|i| i as u32).collect();
                        let cu_q_t = Tensor::from_vec(cu_q, (n + 1,), &device).unwrap();
                        let cu_k_t = Tensor::from_vec(cu_k, (n + 1,), &device).unwrap();
                        device.synchronize().unwrap();
                        let t3 = Instant::now();
                        // kernel: the one flash_attn_varlen call.
                        let _out = candle_flash_attn::flash_attn_varlen(
                            &q_ragged, &k_rag, &v_rag, &cu_q_t, &cu_k_t, 1, max_k, scale, true,
                        )
                        .unwrap();
                        device.synchronize().unwrap();
                        let t4 = Instant::now();
                        if it >= warmup {
                            t_wloop += (t1 - t0).as_secs_f64();
                            t_wscat += (t1s - t1).as_secs_f64();
                            t_gather += (t2 - t1s).as_secs_f64();
                            t_build += (t3 - t2).as_secs_f64();
                            t_kernel += (t4 - t3).as_secs_f64();
                        }
                    }
                    let us = |s: f64| s / iters as f64 * 1e6;
                    let (wl, ws, g, bd, kn) = (
                        us(t_wloop),
                        us(t_wscat),
                        us(t_gather),
                        us(t_build),
                        us(t_kernel),
                    );
                    println!(
                        "  N={n:<2} L={l:<4} write_loop {wl:6.1}us -> write_scatter {ws:6.1}us | \
                         gather {g:6.1}us | build {bd:6.1}us | kernel {kn:6.1}us"
                    );
                }
            }
        }
    }
}

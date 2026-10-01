//! Gated DeltaNet linear attention — the recurrence (story sc-7632, the candle mirror of mlx-llm
//! sc-7627).
//!
//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) interleaves 3 **Gated DeltaNet**
//! linear-attention layers with 1 gated full-attention layer. Unlike softmax attention over a
//! growing KV cache, a linear layer carries a **fixed-size recurrent state** `S ∈ [Dv, Dk]` per head
//! and updates it with the gated delta rule each step — so it costs O(1) memory in sequence length.
//!
//! This module ports the **ops path** of `mlx_lm.models.gated_delta` (`gated_delta_ops` /
//! `_gated_delta_step_ops`) — the sequential reference the MLX engine itself falls back to off-GPU —
//! into Candle, byte-for-byte with the verified `mlx-llm` port. The recurrence is validated against
//! the same numeric fixture (see the tests). The per-step update, for head state `S` (decayed by the
//! gate `g`, `β` the delta strength):
//!
//! ```text
//!   S      = S · g                          # forget (per-head scalar decay)
//!   kv_mem = (S · kᵀ) summed over Dk         # what the current key already recalls  → [Dv]
//!   Δ      = (v − kv_mem) · β                # the correction to write               → [Dv]
//!   S      = S + Δ ⊗ k                        # delta-rule outer-product write         → [Dv, Dk]
//!   y      = (S · qᵀ) summed over Dk          # read out with the query                → [Dv]
//! ```
//!
//! The gate and delta strength come from the layer's learned projections via [`compute_g`] (`g =
//! exp(−exp(A_log) · softplus(a + dt_bias))`) and `β = sigmoid(b)`; the surrounding short-conv,
//! normalisation, and in/out projections (the full layer) build on this in the decoder story. GQA is
//! handled by repeating each of the `Hk` key/query heads to the `Hv` value heads.
//!
//! The math runs in the inputs' dtype, matching the ops reference; the decoder lifts the recurrence
//! to f32 (casting the projected q/k/v/g/β and keeping the SSM state in f32) to match the GPU kernel.
//!
//! ## Per-token state checkpoints (story sc-24131)
//!
//! A recurrence cannot be inverted, so rolling a hybrid decoder back after a rejected speculative
//! draft needs the state *as it was* at the target position. [`DeltaNetCache`] therefore keeps a
//! **ring of per-token states**: a preallocated `[slots, …]` tensor pair (conv tail + SSM state)
//! into which every token's post-step state is written in place ([`Tensor::slice_set`], no
//! per-token allocation — the ring's device addresses never change, which the CUDA-graph runner
//! relies on). The live state is a *view* of the newest slot, so [`DeltaNetCache::rollback_to`] is
//! a slot-index change: no copy, no allocation. The per-token write is an output of the recurrence
//! step ([`gated_delta_recurrence_with_sink`]): a fused decode kernel (sc-24000) produces the same
//! thing by writing each token's state into its ring slot directly.
//!
//! ## Device-indexed slots (story sc-24441)
//!
//! Which slot is live and which slots a step writes move with the position, so a step that picks
//! them on the host bakes the position into its kernels. With [`DevicePositions`] staged, a step
//! reads the live state out of the slot the device index names ([`candle_quant_kernels::read_slot`])
//! and writes each token's state to the slot its device index names
//! ([`candle_quant_kernels::write_rows_at`]) — the same values the host-indexed step reads and
//! writes, at a CUDA-graph-replayable address; the host keeps exactly the same bookkeeping.

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::primitives::decode_cache::tensor_bytes;
use crate::primitives::device_positions::DevicePositions;
use crate::primitives::kv_cache::storage_address;
use crate::primitives::nn::{rms_norm, silu};

/// Numerically-stable softplus `ln(1 + eˣ) = relu(x) + ln(1 + e^−|x|)`, matching the reference's
/// `softplus`. Evaluated in `x`'s dtype.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let log1p = x.abs()?.affine(-1.0, 0.0)?.exp()?.affine(1.0, 1.0)?.log()?; // ln(1 + e^−|x|)
    Ok(relu.broadcast_add(&log1p)?)
}

/// The per-step gate `g = exp(−exp(A_log) · softplus(a + dt_bias))` (a faithful port of
/// `mlx_lm.models.gated_delta.compute_g`). `a` is `[B, T, Hv]` (the gating projection), `A_log` and
/// `dt_bias` are per-value-head `[Hv]`. The inner exponentials are evaluated in f32 (matching the
/// reference's `.astype(float32)`) and the result is cast back to `a`'s dtype.
pub fn compute_g(a: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> Result<Tensor> {
    let orig = a.dtype();
    let a32 = a.to_dtype(DType::F32)?;
    let dt32 = dt_bias.to_dtype(DType::F32)?;
    let al32 = a_log.to_dtype(DType::F32)?;
    let sp = softplus(&a32.broadcast_add(&dt32)?)?; // softplus(a + dt_bias)        [B,T,Hv]
    let coeff = al32.exp()?.broadcast_mul(&sp)?; // exp(A_log) · softplus(...)       [B,T,Hv]
    let g = coeff.affine(-1.0, 0.0)?.exp()?; // exp(−coeff)
    Ok(g.to_dtype(orig)?)
}

/// Run the gated delta recurrence over a `[B, T, ·]` chunk — the entry point every Qwen35-family
/// linear layer reaches without a checkpoint ring.
///
/// Shapes: `q`, `k` are `[B, T, Hk, Dk]`; `v` is `[B, T, Hv, Dv]`; `g` (the per-step gate from
/// [`compute_g`]) and `beta` are `[B, T, Hv]`; `state` (the carried recurrent state, or `None` to
/// start from zeros) is `[B, Hv, Dv, Dk]`. Returns the per-step output `y` `[B, T, Hv, Dv]` and the
/// final `state` `[B, Hv, Dv, Dk]` — feed `state` back in for the next chunk / decode step (T = 1).
///
/// GQA: when `Hv > Hk` each key/query head is repeated `Hv / Hk` times so it pairs with the value
/// heads (`Hv` must be a multiple of `Hk`).
///
/// A sequence of at least [`CHUNKED_PREFILL_MIN_TOKENS`] runs in the chunkwise-parallel form
/// ([`gated_delta_chunked`], sc-24443) — batched matmuls instead of a per-token loop — and returns
/// `y` in `q`'s dtype and the state in the carried state's (or `q`'s); shorter ones (decode,
/// speculative verify) run the per-token reference ([`gated_delta_recurrence_per_token`]).
pub fn gated_delta_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    gated_delta_recurrence_with_sink(q, k, v, g, beta, state, q.dim(1)?, &mut |_, _| Ok(()))
}

/// [`gated_delta_recurrence`] that also hands the **post-step state** of every token from
/// `sink_from` on to `sink` — `sink(ti, state_after_token_ti)` with `state` `[B, Hv, Dv, Dk]`, in
/// token order, before the next token runs. This is the per-token checkpoint output of the
/// recurrence step (sc-24131): [`DeltaNetCache::advance`] writes each state into its ring slot
/// from here (only its last `slots` tokens, so it passes `sink_from = T - slots`), and a fused
/// decode kernel (sc-24000) keeps the same contract by writing the slot itself. The tokens before
/// `sink_from` hand no state out, so once there are at least [`CHUNKED_PREFILL_MIN_TOKENS`] of
/// them they run chunkwise ([`gated_delta_chunked`], sc-24443) and only the sunk tail steps per
/// token. The returned final state is the post-step state of the last token (the last one handed
/// to `sink` when `sink_from < T`). An error from `sink` aborts the recurrence.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_recurrence_with_sink(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
    sink_from: usize,
    sink: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
) -> Result<(Tensor, Tensor)> {
    let t = q.dim(1)?;
    let chunked = sink_from.min(t);
    #[cfg(test)]
    let chunked = if FORCE_PER_TOKEN.with(|f| f.get()) {
        0
    } else {
        chunked
    };
    if chunked < CHUNKED_PREFILL_MIN_TOKENS {
        return gated_delta_recurrence_per_token(q, k, v, g, beta, state, sink);
    }
    let head = |x: &Tensor| x.narrow(1, 0, chunked);
    let (y_head, s_head) = gated_delta_chunked(
        &head(q)?,
        &head(k)?,
        &head(v)?,
        &head(g)?,
        &head(beta)?,
        state,
    )?;
    let state_dtype = state.map_or(q.dtype(), Tensor::dtype);
    let y_head = y_head.to_dtype(q.dtype())?;
    let s_head = s_head.to_dtype(state_dtype)?;
    if chunked == t {
        return Ok((y_head, s_head));
    }
    let tail = |x: &Tensor| x.narrow(1, chunked, t - chunked);
    let (y_tail, s_tail) = gated_delta_recurrence_per_token(
        &tail(q)?,
        &tail(k)?,
        &tail(v)?,
        &tail(g)?,
        &tail(beta)?,
        Some(&s_head),
        &mut |ti, s| sink(chunked + ti, s),
    )?;
    Ok((Tensor::cat(&[&y_head, &y_tail], 1)?, s_tail))
}

#[cfg(test)]
thread_local! {
    /// Test-only switch routing [`gated_delta_recurrence_with_sink`] to the per-token reference on
    /// this thread, so a model-level test can compare the production dispatch against it.
    static FORCE_PER_TOKEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with the recurrence forced onto the per-token reference (this thread only).
#[cfg(test)]
pub(crate) fn with_per_token_reference<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_PER_TOKEN.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FORCE_PER_TOKEN.with(|c| c.replace(true)));
    f()
}

/// The per-token (op-by-op) gated delta recurrence — the faithful port of the
/// `mlx_lm.models.gated_delta` ops path, one step per token, handing **every** token's post-step
/// state to `sink`. It is the **numeric reference** the chunked form is tested against and the path
/// decode and speculative verify take. Shapes as [`gated_delta_recurrence`]; the math runs in the
/// inputs' dtype.
pub fn gated_delta_recurrence_per_token(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
    sink: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
) -> Result<(Tensor, Tensor)> {
    let (b, t, hk, dk) = q.dims4()?;
    let (_, _, hv, dv) = v.dims4()?;

    // GQA: repeat each of the Hk key/query heads to the Hv value heads (contiguous).
    let (q, k) = if hv != hk {
        let r = hv / hk;
        (repeat_heads(q, r)?, repeat_heads(k, r)?)
    } else {
        (q.clone(), k.clone())
    };

    let mut state = match state {
        Some(s) => s.clone(),
        None => Tensor::zeros((b, hv, dv, dk), q.dtype(), q.device())?,
    };

    let mut ys: Vec<Tensor> = Vec::with_capacity(t);
    for ti in 0..t {
        let qt = q.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv,Dk]
        let kt = k.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv,Dk]
        let vt = v.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv,Dv]
        let gt = g.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv]
        let bt = beta.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?; // [B,Hv]
        let (y, next) = delta_step(&qt, &kt, &vt, &gt, &bt, &state, b, hv, dk, dv)?;
        sink(ti, &next)?;
        state = next;
        ys.push(y.unsqueeze(1)?); // [B,1,Hv,Dv]
    }
    let refs: Vec<&Tensor> = ys.iter().collect();
    let y = Tensor::cat(&refs, 1)?; // [B,T,Hv,Dv]
    Ok((y, state))
}

/// The shortest run of un-checkpointed tokens the recurrence computes chunkwise (one full
/// chunk); shorter runs step per token.
pub const CHUNKED_PREFILL_MIN_TOKENS: usize = GDN_CHUNK;

/// Tokens per chunk of the chunkwise-parallel form (the flash-linear-attention default).
const GDN_CHUNK: usize = 64;

/// Chunks whose intra-chunk work is batched together: bounds the prefill's `[C, C]` transients to
/// `GDN_SEGMENT_CHUNKS · GDN_CHUNK` tokens' worth however long the prompt is.
const GDN_SEGMENT_CHUNKS: usize = 8;

/// The gated delta recurrence in its **chunkwise-parallel** form (the WY / flash-linear-attention
/// `chunk_gated_delta_rule` formulation, sc-24443), for prefill. Shapes as
/// [`gated_delta_recurrence`]; inputs of any float dtype are computed in f32 and `(y, state)` are
/// returned **f32**. Every device runs it (plain tensor ops).
///
/// Within a chunk of `C` tokens starting from state `S₀`, with `Γᵢ = ∏_{j≤i} gⱼ` (a cumulative
/// sum of `ln g`), the per-token deltas `δᵢ = βᵢ(vᵢ − gᵢSᵢ₋₁kᵢ)` solve the unit-lower-triangular
/// system `(I + A) Δ = V_β − (K_β Γ) S₀ᵀ` with `Aᵢⱼ = βᵢ (Γᵢ/Γⱼ) kᵢ·kⱼ` (`i > j`), so with
/// `T = (I + A)⁻¹`, `U = T V_β` and `W = T (K_β Γ)` (all of a segment's chunks at once):
///
/// ```text
///   Δ    = U − W S₀ᵀ
///   Y    = (Q Γ) S₀ᵀ + ((Q Kᵀ) ⊙ Γᵢ/Γⱼ, j ≤ i) Δ
///   S_C  = Γ_C S₀ + Δᵀ (K Γ_C/Γ)
/// ```
///
/// Only the `S₀ → S_C` hand-off is sequential (one step per chunk, not per token). `T` is built
/// by block doubling — `T₂ₛ = Tₛ − Tₛ Eₛ Tₛ`, `Eₛ` the part of `A` inside the `2s`-diagonal
/// blocks but outside the `s`-diagonal ones — which is exact block forward substitution. Every
/// exponent is `≤ 0` (`g ∈ (0, 1]`), so nothing overflows; a gate that underflowed to 0 is
/// clamped to the smallest normal f32 before its log.
pub fn gated_delta_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let f = DType::F32;
    let c = GDN_CHUNK;
    let (b, t, hk, dk) = q.dims4()?;
    let (_, _, hv, dv) = v.dims4()?;
    let dev = q.device();

    // Pad T to whole chunks with inert steps (g = 1, β = 0, k = q = v = 0 leave the state as is),
    // then lay every tensor out per head and chunk: [N = B·Hv, nC, C, ·].
    let n_chunks = t.div_ceil(c);
    let pad = n_chunks * c - t;
    let n = b * hv;
    let heads_first = |x: &Tensor, fill: f64| -> Result<Tensor> {
        let x = x.to_dtype(f)?;
        let x = if pad > 0 {
            let mut shape = x.dims().to_vec();
            shape[1] = pad;
            let filler = Tensor::ones(shape, f, dev)?.affine(fill, 0.0)?;
            Tensor::cat(&[&x, &filler], 1)?
        } else {
            x
        };
        Ok(match x.rank() {
            // [B, Tp, Hv] → [N, nC, C]
            3 => x.transpose(1, 2)?.contiguous()?.reshape((n, n_chunks, c))?,
            // [B, Tp, Hv, D] → [N, nC, C, D]
            _ => {
                let d = x.dim(3)?;
                x.transpose(1, 2)?
                    .contiguous()?
                    .reshape((n, n_chunks, c, d))?
            }
        })
    };
    // GQA-expand q/k as statement temporaries, so each prompt-sized expansion frees as soon as
    // it is laid out.
    let expand = |x: &Tensor| -> Result<Tensor> {
        if hv != hk {
            repeat_heads(x, hv / hk)
        } else {
            Ok(x.clone())
        }
    };
    let q = heads_first(&expand(q)?, 0.0)?;
    let k = heads_first(&expand(k)?, 0.0)?;
    let v = heads_first(v, 0.0)?;
    let g = heads_first(g, 1.0)?;
    let beta = heads_first(beta, 0.0)?;

    let masks = ChunkMasks::new(c, dev)?;
    let mut state = match state {
        Some(s) => s.to_dtype(f)?.reshape((n, dv, dk))?,
        None => Tensor::zeros((n, dv, dk), f, dev)?,
    };
    let mut ys: Vec<Tensor> = Vec::with_capacity(n_chunks);
    let mut seg_start = 0;
    while seg_start < n_chunks {
        let m = GDN_SEGMENT_CHUNKS.min(n_chunks - seg_start);
        // This segment's chunks folded into the batch: [N·m, C, ·].
        let seg = |x: &Tensor| -> Result<Tensor> {
            let x = x.narrow(1, seg_start, m)?.contiguous()?;
            Ok(match x.rank() {
                3 => x.reshape((n * m, c))?,
                _ => {
                    let d = x.dim(3)?;
                    x.reshape((n * m, c, d))?
                }
            })
        };
        let (qc, kc, vc, gc, bc) = (seg(&q)?, seg(&k)?, seg(&v)?, seg(&g)?, seg(&beta)?);

        let log_g = gc.maximum(f32::MIN_POSITIVE)?.log()?;
        let cum = log_g.cumsum(1)?; // ln Γᵢ                                  [N·m, C]
        let gamma = cum.exp()?.unsqueeze(2)?; // Γᵢ                           [N·m, C, 1]
        let diff = cum.unsqueeze(2)?.broadcast_sub(&cum.unsqueeze(1)?)?; // ln Γᵢ − ln Γⱼ
        let decay = diff.minimum(0f32)?.exp()?.broadcast_mul(&masks.lower)?; // Γᵢ/Γⱼ, j ≤ i
        let k_t = kc.transpose(1, 2)?.contiguous()?; //                        [N·m, Dk, C]
        let beta_col = bc.unsqueeze(2)?;
        let kb = kc.broadcast_mul(&beta_col)?; // β k
        let a = kb.matmul(&k_t)?.mul(&decay)?.broadcast_mul(&masks.strict)?; // Aᵢⱼ, i > j
        let tinv = unit_lower_inverse(&a, &masks)?; // (I + A)⁻¹
        let w = tinv.matmul(&kb.broadcast_mul(&gamma)?)?; //                    [N·m, C, Dk]
        let u = tinv.matmul(&vc.broadcast_mul(&beta_col)?)?; //                 [N·m, C, Dv]
        let q_gamma = qc.broadcast_mul(&gamma)?; // Q Γ
        let qk = qc.matmul(&k_t)?.mul(&decay)?; // causal (Q Kᵀ) ⊙ Γᵢ/Γⱼ
        let last = cum.narrow(1, c - 1, 1)?; // ln Γ_C                         [N·m, 1]
        let k_tail = kc.broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(2)?)?; // K Γ_C/Γ
        let chunk_decay = last.exp()?; // Γ_C                                  [N·m, 1]

        // Back to [N, m, ·] to walk the segment's chunks in order.
        let per_chunk = |x: &Tensor| -> Result<Tensor> {
            let mut shape = vec![n, m];
            shape.extend_from_slice(&x.dims()[1..]);
            Ok(x.reshape(shape)?)
        };
        let (w, u, q_gamma, qk, k_tail, chunk_decay) = (
            per_chunk(&w)?,
            per_chunk(&u)?,
            per_chunk(&q_gamma)?,
            per_chunk(&qk)?,
            per_chunk(&k_tail)?,
            per_chunk(&chunk_decay)?,
        );
        for i in 0..m {
            let pick = |x: &Tensor| -> Result<Tensor> { Ok(x.narrow(1, i, 1)?.squeeze(1)?) };
            let s_t = state.transpose(1, 2)?.contiguous()?; // S₀ᵀ              [N, Dk, Dv]
            let delta = pick(&u)?.sub(&pick(&w)?.contiguous()?.matmul(&s_t)?)?; // [N, C, Dv]
            let y = pick(&q_gamma)?
                .contiguous()?
                .matmul(&s_t)?
                .add(&pick(&qk)?.contiguous()?.matmul(&delta)?)?; //             [N, C, Dv]
            let carried = state.broadcast_mul(&pick(&chunk_decay)?.unsqueeze(2)?)?; // Γ_C S₀
            state = carried.add(
                &delta
                    .transpose(1, 2)?
                    .contiguous()?
                    .matmul(&pick(&k_tail)?.contiguous()?)?,
            )?; // + Δᵀ (K Γ_C/Γ)                                               [N, Dv, Dk]
            ys.push(y);
        }
        seg_start += m;
    }
    // One copy out: the per-chunk outputs free once concatenated, and the padded [N, Tp, Dv]
    // result reshapes as a view so the narrow + transpose materialize in a single `contiguous`.
    let y = Tensor::cat(&ys.iter().collect::<Vec<_>>(), 1)?; // [N, Tp, Dv]
    drop(ys);
    let y = y
        .reshape((b, hv, n_chunks * c, dv))?
        .narrow(2, 0, t)?
        .transpose(1, 2)?
        .contiguous()?; // [B, T, Hv, Dv]
    Ok((y, state.reshape((b, hv, dv, dk))?))
}

/// Constant `[C, C]` masks of the chunkwise form, plus the block-doubling bands.
struct ChunkMasks {
    /// `1` where `j ≤ i`.
    lower: Tensor,
    /// `1` where `j < i`.
    strict: Tensor,
    /// The identity.
    eye: Tensor,
    /// For each doubling level `s` (1, 2, 4, …, C/2): `1` where `i, j` share a `2s` block but
    /// not an `s` block.
    bands: Vec<Tensor>,
}

impl ChunkMasks {
    fn new(c: usize, dev: &Device) -> Result<Self> {
        let build = |f: &dyn Fn(usize, usize) -> bool| -> Result<Tensor> {
            let data: Vec<f32> = (0..c * c)
                .map(|x| if f(x / c, x % c) { 1.0 } else { 0.0 })
                .collect();
            Ok(Tensor::from_vec(data, (c, c), dev)?)
        };
        let mut bands = Vec::new();
        let mut s = 1usize;
        while s < c {
            bands.push(build(&|i, j| i / (2 * s) == j / (2 * s) && i / s != j / s)?);
            s *= 2;
        }
        Ok(Self {
            lower: build(&|i, j| j <= i)?,
            strict: build(&|i, j| j < i)?,
            eye: build(&|i, j| i == j)?,
            bands,
        })
    }
}

/// `(I + A)⁻¹` for strictly-lower-triangular `A` `[M, C, C]` by block doubling:
/// `T₁ = I`, `T₂ₛ = Tₛ − Tₛ (A ⊙ bandₛ) Tₛ` — exact, since `bandₛ` holds exactly the one
/// off-diagonal block each `2s` block adds and `Eₛ Tₛ Eₛ = 0`.
fn unit_lower_inverse(a: &Tensor, masks: &ChunkMasks) -> Result<Tensor> {
    let mut inv = masks.eye.broadcast_as(a.shape())?.contiguous()?;
    for band in &masks.bands {
        let e = a.broadcast_mul(band)?;
        let correction = inv.matmul(&e)?.matmul(&inv)?;
        inv = inv.sub(&correction)?;
    }
    Ok(inv)
}

/// Causal depthwise short convolution over `[B, S, C]` with per-channel kernel `weight` `[C, K]`
/// (the HF/MLX depthwise `Conv1d`, no bias), left-seeded by `conv_state` `[B, K-1, C]` (the previous
/// step's tail). Returns `(silu(conv) [B,S,C], new_conv_state [B,K-1,C])` — a port of the Qwen3-Next
/// short-conv path: `out[b,s,c] = silu(Σ_j weight[c,j] · concat(conv_state, x)[b, s+j, c])`. Mixing
/// q/k/v through this 1-D conv before the recurrence is what gives Gated DeltaNet its local context.
pub fn causal_depthwise_conv(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (out, trace) = causal_depthwise_conv_traced(x, weight, conv_state)?;
    Ok((out, trace.tail_after(trace.tokens() - 1)?.contiguous()?))
}

/// The conv input of one forward — `[conv_state ‖ x]`, `[B, S+K-1, C]` — from which the conv tail
/// **after any token** of the forward is a slice: the state after token `ti` (0-based) is rows
/// `ti+1 .. ti+K` (see [`tail_after`](Self::tail_after)). [`causal_depthwise_conv_traced`] returns
/// it so a cache can checkpoint every token's conv state without recomputing the conv.
#[derive(Clone, Debug)]
pub struct ConvTrace {
    cat: Tensor,
    tokens: usize,
    tail: usize,
}

impl ConvTrace {
    /// Tokens `S` in the forward this trace belongs to.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Rows of one conv tail (`K - 1`).
    pub fn tail_len(&self) -> usize {
        self.tail
    }

    /// The conv tail after token `ti` (0-based) of this forward — the `[B, K-1, C]` state a step
    /// starting right after that token is seeded with. A (non-contiguous) view; `ti >= tokens()`
    /// is an error.
    pub fn tail_after(&self, ti: usize) -> Result<Tensor> {
        if ti >= self.tokens {
            return Err(Error::Msg(format!(
                "ConvTrace: token {ti} outside a {}-token forward",
                self.tokens
            )));
        }
        Ok(self.cat.narrow(1, ti + 1, self.tail)?)
    }
}

/// [`causal_depthwise_conv`] returning the [`ConvTrace`] (the conv input) instead of only the final
/// tail, so the caller can take the tail after **every** token (the per-token checkpoint ring).
pub fn causal_depthwise_conv_traced(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
) -> Result<(Tensor, ConvTrace)> {
    let (_b, s, c) = x.dims3()?;
    let kk = weight.dim(1)?; // kernel size K (weight is [C, K])
    let cat = Tensor::cat(&[conv_state, x], 1)?; // [B, S+K-1, C]
    let mut acc: Option<Tensor> = None;
    for j in 0..kk {
        let window = cat.narrow(1, j, s)?; // cat[:, j:j+S, :] → [B,S,C]
        let wj = weight.narrow(1, j, 1)?.contiguous()?.reshape((1, 1, c))?; // weight[:,j] → [1,1,C]
        let term = window.broadcast_mul(&wj)?;
        acc = Some(match acc {
            None => term,
            Some(a) => a.broadcast_add(&term)?,
        });
    }
    let out = silu(&acc.expect("conv kernel size must be >= 1"))?; // [B,S,C]
    Ok((
        out,
        ConvTrace {
            cat,
            tokens: s,
            tail: kk - 1,
        },
    ))
}

/// Gated RMSNorm (`Qwen3NextRMSNormGated`): `rms_norm(x, weight, eps) · silu(gate)`. Applied to the
/// delta-net output before the out-projection (`x`, `gate`, and `weight` share the head-value dim).
pub fn rms_norm_gated(x: &Tensor, weight: &Tensor, gate: &Tensor, eps: f64) -> Result<Tensor> {
    let normed = rms_norm(x, weight, eps)?;
    Ok(normed.broadcast_mul(&silu(gate)?)?)
}

/// The geometry of one layer's per-token checkpoint ring (see [`DeltaNetCache::with_ring`]): how
/// many positions it holds and the shape of one slot. The whole ring is allocated **once**, at
/// [`DeltaNetCache::preallocate`], and never grows on the decode path.
#[derive(Clone, Debug)]
pub struct RingSpec {
    /// Positions held: the current one plus `slots - 1` earlier ones the cache can roll back to.
    /// At least 2.
    pub slots: usize,
    /// One conv tail `[B, K-1, conv_dim]`.
    pub conv_dims: (usize, usize, usize),
    /// The conv tail's dtype (the layer's compute dtype).
    pub conv_dtype: DType,
    /// One SSM state `[B, Hv, Dv, Dk]`.
    pub ssm_dims: (usize, usize, usize, usize),
    /// The SSM state's dtype (f32 in the decoder).
    pub ssm_dtype: DType,
    /// Where the ring lives.
    pub device: Device,
}

impl RingSpec {
    /// Bytes of one slot (conv tail + SSM state).
    pub fn slot_bytes(&self) -> usize {
        let (b, t, c) = self.conv_dims;
        let (sb, hv, dv, dk) = self.ssm_dims;
        let conv = b
            .saturating_mul(t)
            .saturating_mul(c)
            .saturating_mul(self.conv_dtype.size_in_bytes());
        let ssm = sb
            .saturating_mul(hv)
            .saturating_mul(dv)
            .saturating_mul(dk)
            .saturating_mul(self.ssm_dtype.size_in_bytes());
        conv.saturating_add(ssm)
    }

    /// Bytes of the whole ring (`slots × slot_bytes`).
    pub fn bytes(&self) -> usize {
        self.slot_bytes().saturating_mul(self.slots)
    }

    /// Positions before the current one the ring can roll back to (`slots - 1`).
    pub fn depth(&self) -> usize {
        self.slots.saturating_sub(1)
    }

    fn validate(&self) -> Result<()> {
        if self.slots < 2 {
            return Err(Error::Msg(format!(
                "DeltaNet ring: at least 2 slots (the current position plus one to roll back \
                 to), got {}",
                self.slots
            )));
        }
        Ok(())
    }
}

/// The preallocated per-token state ring of one layer: slot `p % slots` holds the state **after**
/// position `p` (the state a step starting at `p` is seeded with) for every `p` in `lo..=offset`.
#[derive(Debug)]
struct StateRing {
    /// `[slots, B, K-1, conv_dim]`.
    conv: Tensor,
    /// `[slots, B, Hv, Dv, Dk]`.
    ssm: Tensor,
    slots: usize,
    /// The lowest position whose state the ring still holds (`offset + 1` when it holds none;
    /// never below 1 — position 0 is the zero state, restored by a reset).
    lo: i32,
}

impl StateRing {
    fn slot(&self, position: i32) -> usize {
        usize::try_from(position).unwrap_or(0) % self.slots
    }

    fn write(&self, position: i32, conv: &Tensor, ssm: &Tensor) -> Result<()> {
        let slot = self.slot(position);
        self.conv
            .slice_set(&conv.contiguous()?.unsqueeze(0)?, 0, slot)?;
        self.ssm
            .slice_set(&ssm.contiguous()?.unsqueeze(0)?, 0, slot)?;
        Ok(())
    }

    /// [`write`](Self::write) into the slot the device `u32` `slot` names (sc-24441).
    fn write_at(&self, slot: &Tensor, conv: &Tensor, ssm: &Tensor) -> Result<()> {
        candle_quant_kernels::write_rows_at(
            &self.conv,
            &conv.contiguous()?.unsqueeze(0)?,
            slot,
            0,
        )?;
        candle_quant_kernels::write_rows_at(&self.ssm, &ssm.contiguous()?.unsqueeze(0)?, slot, 0)?;
        Ok(())
    }

    fn views(&self, position: i32) -> Result<(Tensor, Tensor)> {
        let slot = self.slot(position);
        Ok((
            self.conv.narrow(0, slot, 1)?.squeeze(0)?,
            self.ssm.narrow(0, slot, 1)?.squeeze(0)?,
        ))
    }
}

/// The recurrent state of one Gated DeltaNet layer — the linear-attention analog of a KV-cache slot
/// (the Mamba/SSM cache). It holds the short-conv tail `conv_state` `[B, K-1, conv_dim]` and the
/// delta-rule `ssm_state` `[B, Hv, Dv, Dk]`, both **fixed size** in sequence length (unlike the
/// growing KV cache). A hybrid decoder keeps one of these per linear layer alongside a
/// [`KvCache`](super::KvCache) per full-attention layer (the decoder assembles the mixed list).
///
/// A cache built with [`with_ring`](Self::with_ring) also keeps the **per-token checkpoint ring**
/// (module docs): every forward writes the state after each of its last `slots` tokens into the
/// ring in place, the live state is a view of the newest slot, and [`rollback_to`](Self::rollback_to)
/// selects an earlier slot. A cache built with [`new`](Self::new) (the reference path) holds the
/// live state only and can roll back to nothing but zero.
#[derive(Debug, Default)]
pub struct DeltaNetCache {
    /// The short-conv history (previous `K-1` tokens), or `None` before the first step. With a
    /// ring, a view of the live slot.
    conv_state: Option<Tensor>,
    /// The delta-rule recurrent state, or `None` before the first step. With a ring, a view of the
    /// live slot.
    ssm_state: Option<Tensor>,
    offset: i32,
    /// The ring's geometry (set at construction; the ring itself is allocated by
    /// [`preallocate`](Self::preallocate) or the first [`advance`](Self::advance)).
    spec: Option<RingSpec>,
    ring: Option<StateRing>,
}

impl DeltaNetCache {
    /// An empty cache (no conv history, zero recurrent state), without a checkpoint ring.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache that will keep a per-token checkpoint ring of `spec.slots` positions. The
    /// ring is allocated by [`preallocate`](Self::preallocate) (call it at request admission so
    /// the allocation fails closed there) or, failing that, by the first [`advance`](Self::advance).
    /// A `spec` with fewer than 2 slots is [`Error::Msg`].
    pub fn with_ring(spec: RingSpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self {
            spec: Some(spec),
            ..Self::default()
        })
    }

    /// Allocate the ring now (a no-op without a ring spec or once allocated). Zero-filled, so a
    /// freshly allocated ring holds no restorable position (`lo == offset + 1`).
    pub fn preallocate(&mut self) -> Result<()> {
        if self.ring.is_some() {
            return Ok(());
        }
        let Some(spec) = &self.spec else {
            return Ok(());
        };
        let (b, t, c) = spec.conv_dims;
        let (sb, hv, dv, dk) = spec.ssm_dims;
        let conv = Tensor::zeros((spec.slots, b, t, c), spec.conv_dtype, &spec.device)?;
        let ssm = Tensor::zeros((spec.slots, sb, hv, dv, dk), spec.ssm_dtype, &spec.device)?;
        self.ring = Some(StateRing {
            conv,
            ssm,
            slots: spec.slots,
            lo: self.offset + 1,
        });
        Ok(())
    }

    /// The ring geometry this cache keeps, or `None` for a ring-less cache.
    pub fn ring_spec(&self) -> Option<&RingSpec> {
        self.spec.as_ref()
    }

    /// Positions before the current one this cache can roll back to at most (`slots - 1`; `0`
    /// without a ring).
    pub fn depth(&self) -> usize {
        self.spec.as_ref().map(RingSpec::depth).unwrap_or(0)
    }

    /// Replace the ring geometry: `None` drops the ring (the cache keeps its live state as plain
    /// tensors), `Some` reallocates a ring of the new size and **carries over** every restorable
    /// position that still fits (the newest `slots - 1` before the current one). Built through
    /// [`resized`](Self::resized), so a failure (an invalid spec, the allocation, a carried state
    /// that does not fit the new slots) leaves the cache untouched.
    pub fn set_ring_spec(&mut self, spec: Option<RingSpec>) -> Result<()> {
        *self = self.resized(spec)?;
        Ok(())
    }

    /// This cache under another ring geometry (see [`set_ring_spec`](Self::set_ring_spec)),
    /// built without touching `self`: a caller replacing several layers' rings builds every
    /// replacement first and swaps them in only once all succeeded. The replacement's ring is its
    /// own allocation; a ring-less replacement holds a detached copy of the live state (the old
    /// ring's buffer is written in place, so its views cannot be kept).
    pub fn resized(&self, spec: Option<RingSpec>) -> Result<Self> {
        let Some(spec) = spec else {
            let (conv_state, ssm_state) = match (&self.ring, &self.conv_state, &self.ssm_state) {
                (Some(_), Some(conv), Some(ssm)) => (
                    Some(conv.contiguous()?.copy()?),
                    Some(ssm.contiguous()?.copy()?),
                ),
                _ => (self.conv_state.clone(), self.ssm_state.clone()),
            };
            return Ok(DeltaNetCache {
                conv_state,
                ssm_state,
                offset: self.offset,
                spec: None,
                ring: None,
            });
        };
        spec.validate()?;
        let mut next = DeltaNetCache {
            conv_state: None,
            ssm_state: None,
            offset: self.offset,
            spec: Some(spec),
            ring: None,
        };
        if self.ring.is_none() && self.conv_state.is_none() {
            // Nothing to carry: only the geometry changes (still lazily allocated).
            return Ok(next);
        }
        next.preallocate()?;
        let ring = next.ring.as_mut().expect("just allocated");
        // The lowest position whose state is held (the live one counts), clipped to what the new
        // ring can hold.
        let held_from = self.restorable_from().min(self.offset);
        let keep_from = held_from.max(self.offset + 1 - ring.slots as i32).max(1);
        if self.offset >= 1 {
            for p in keep_from..=self.offset {
                let (conv, ssm) = self.state_at(p)?;
                ring.write(p, &conv, &ssm)?;
            }
            ring.lo = keep_from;
            let (conv, ssm) = ring.views(self.offset)?;
            next.conv_state = Some(conv);
            next.ssm_state = Some(ssm);
        }
        Ok(next)
    }

    /// The lowest position `rollback_to` can currently return to other than zero (`offset + 1`
    /// when none).
    fn restorable_from(&self) -> i32 {
        match &self.ring {
            Some(ring) => ring.lo,
            None => self.offset + 1,
        }
    }

    /// The state after position `p`: the live state at the current position, else the ring slot.
    fn state_at(&self, p: i32) -> Result<(Tensor, Tensor)> {
        if p == self.offset {
            if let (Some(conv), Some(ssm)) = (&self.conv_state, &self.ssm_state) {
                return Ok((conv.clone(), ssm.clone()));
            }
        }
        match &self.ring {
            Some(ring) if p >= ring.lo && p <= self.offset => ring.views(p),
            _ => Err(Error::Msg(format!(
                "DeltaNetCache: no state held for position {p}"
            ))),
        }
    }

    /// Positions consumed so far (the linear-layer analog of [`KvCache::offset`](super::KvCache::offset)).
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// The short-conv history (previous `K-1` tokens), or `None` before the first step.
    pub fn conv_state(&self) -> Option<&Tensor> {
        self.conv_state.as_ref()
    }

    /// The delta-rule recurrent state, or `None` before the first step.
    pub fn ssm_state(&self) -> Option<&Tensor> {
        self.ssm_state.as_ref()
    }

    /// The positions [`rollback_to`](Self::rollback_to) can currently return to, ascending —
    /// zero (always possible) and the current position (a no-op) are not listed.
    pub fn restorable(&self) -> Vec<i32> {
        (self.restorable_from()..self.offset).collect()
    }

    /// Whether [`rollback_to`](Self::rollback_to)`(n)` would succeed.
    pub fn can_rollback_to(&self, n: i32) -> bool {
        n == 0 || (n >= self.restorable_from() && n <= self.offset)
    }

    /// Run the recurrence over `T` new tokens from the live state and checkpoint every token: the
    /// conv tails come from `conv` (the trace of this forward's conv, whose tail after token `ti`
    /// is the conv state at position `offset + ti + 1`), the SSM states from the recurrence step
    /// ([`gated_delta_recurrence_with_sink`]). The last `min(T, slots)` tokens' states are written
    /// into the ring in place (earlier ones could never be restored anyway), the live state becomes
    /// a view of the newest slot, and the position advances by `T`. Without a ring the final
    /// state is kept as plain tensors. Returns the recurrence output `y` `[B, T, Hv, Dv]`.
    ///
    /// Shapes follow [`gated_delta_recurrence`]. The ring slots are written **during** the
    /// recurrence, as each token's state is produced, so before the first write the restorable
    /// window drops every position whose slot this forward reuses (those `slots` or more behind a
    /// position it writes). A forward that fails part-way — a state of another shape than the
    /// ring's slots, a device error — leaves the position unadvanced and
    /// [`restorable`](Self::restorable) listing only slots that still hold their position's
    /// state; with `T < slots` the live slot is never reused, so the cache keeps decoding from
    /// where it was. A forward of `T >= slots` tokens rewrites every slot, the live one included:
    /// its failure also drops the live state, and the cache refuses to advance again until it is
    /// [`reset`](Self::reset) (or rolled back to zero) or handed a state through
    /// [`update`](Self::update).
    pub fn advance(
        &mut self,
        conv: &ConvTrace,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        self.advance_at(conv, q, k, v, g, beta, None)
    }

    /// Whether a step on `positions` reads and writes the ring through device indices: device
    /// positions staged, and a ring to index.
    fn device_indexed<'p>(
        &self,
        positions: Option<&'p DevicePositions>,
    ) -> Option<&'p DevicePositions> {
        positions.filter(|_| self.spec.is_some())
    }

    /// The conv tail the next step is seeded with: the live one, or — with device positions and a
    /// ring (sc-24441) — a copy of the slot the device index names (the same values, read at a
    /// replayable address). `None` at position zero (the caller seeds zeros).
    pub fn live_conv_state(
        &mut self,
        positions: Option<&DevicePositions>,
    ) -> Result<Option<Tensor>> {
        match self.device_indexed(positions) {
            Some(positions) if self.offset > 0 => {
                self.preallocate()?;
                let ring = self.ring.as_ref().expect("preallocated");
                Ok(Some(candle_quant_kernels::read_slot(
                    &ring.conv,
                    &positions.ring_read()?,
                )?))
            }
            _ => Ok(self.conv_state.clone()),
        }
    }

    /// [`advance`](Self::advance), with the ring read and written through the device indices of
    /// `positions` when they are staged (see the module docs): the live SSM state comes from the
    /// slot `positions.ring_read()` names and token `t`'s state goes to `positions.ring_write(t)`
    /// — at position zero the recurrence starts from zeros, as the host path does. The host
    /// bookkeeping (offset, restorable window, live views) is identical either way.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_at(
        &mut self,
        conv: &ConvTrace,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        positions: Option<&DevicePositions>,
    ) -> Result<Tensor> {
        let positions = self.device_indexed(positions);
        self.preallocate()?;
        let t = q.dim(1)?;
        if conv.tokens() != t {
            return Err(Error::Msg(format!(
                "DeltaNetCache::advance: conv trace covers {} tokens, recurrence {t}",
                conv.tokens()
            )));
        }
        if self.offset > 0 && self.ssm_state.is_none() {
            return Err(Error::Msg(format!(
                "DeltaNetCache::advance: the live state at position {} was lost to a failed \
                 forward; reset the cache first",
                self.offset
            )));
        }
        let offset = self.offset;
        let next = offset + t as i32;
        // Whether this forward's writes reach the live slot (every slot is rewritten once
        // `T >= slots`), so that a failure part-way must drop the live state.
        let mut clobbers_live = false;
        let run = match self.ring.as_mut() {
            Some(ring) => {
                let slots = ring.slots;
                clobbers_live = t >= slots;
                // The writes land on positions `offset + T + 1 - min(T, slots) ..= offset + T`,
                // whose slots held the positions `slots` behind them: stop listing those (and
                // everything older) before the first write.
                ring.lo = ring.lo.max(offset + t.min(slots) as i32 + 1 - slots as i32);
                let ring = &*ring;
                let first_write = t.saturating_sub(slots);
                let mut sink = |ti: usize, state: &Tensor| -> Result<()> {
                    if ti < first_write {
                        return Ok(());
                    }
                    match positions {
                        Some(p) => ring.write_at(&p.ring_write(ti)?, &conv.tail_after(ti)?, state),
                        None => {
                            let position = offset + ti as i32 + 1;
                            ring.write(position, &conv.tail_after(ti)?, state)
                        }
                    }
                };
                // Device-indexed: the live state is a copy of the slot the device index names
                // (the slot `ssm_state` is a view of), taken before any write.
                let live = match positions {
                    Some(p) if offset > 0 => {
                        Some(candle_quant_kernels::read_slot(&ring.ssm, &p.ring_read()?)?)
                    }
                    _ => None,
                };
                let state = match positions {
                    Some(_) => live.as_ref(),
                    None => self.ssm_state.as_ref(),
                };
                gated_delta_recurrence_with_sink(q, k, v, g, beta, state, first_write, &mut sink)
                    .and_then(|(y, _)| Ok((y, ring.views(next)?)))
            }
            None => gated_delta_recurrence(q, k, v, g, beta, self.ssm_state.as_ref()).and_then(
                |(y, final_state)| Ok((y, (conv.tail_after(t - 1)?.contiguous()?, final_state))),
            ),
        };
        let (y, (conv_state, ssm_state)) = match run {
            Ok(out) => out,
            Err(e) => {
                if clobbers_live {
                    // Any slot — the live one included — may hold this forward's state now.
                    self.conv_state = None;
                    self.ssm_state = None;
                }
                return Err(e);
            }
        };
        if let Some(ring) = self.ring.as_mut() {
            ring.lo = ring.lo.max(next + 1 - ring.slots as i32).max(1);
        }
        self.conv_state = Some(conv_state);
        self.ssm_state = Some(ssm_state);
        self.offset = next;
        Ok(y)
    }

    /// The host side of a `t`-token step whose device work a CUDA-graph replay did (sc-24441):
    /// exactly the restorable-window and live-view bookkeeping [`advance_at`](Self::advance_at)
    /// does with device positions, and the position advances by `t`. Needs the ring, and a
    /// position past zero — at zero the recorded step would read a slot that holds no state
    /// (the eager step seeds zeros there instead).
    pub fn replay_advance(&mut self, t: usize) -> Result<()> {
        if self.offset == 0 || self.ssm_state.is_none() {
            return Err(Error::Msg(format!(
                "DeltaNetCache::replay_advance: no live state at position {} to replay from",
                self.offset
            )));
        }
        self.preallocate()?;
        let Some(ring) = self.ring.as_mut() else {
            return Err(Error::Msg(
                "DeltaNetCache::replay_advance: a graph replay needs the checkpoint ring".into(),
            ));
        };
        let offset = self.offset;
        let next = offset + t as i32;
        let slots = ring.slots;
        ring.lo = ring.lo.max(offset + t.min(slots) as i32 + 1 - slots as i32);
        ring.lo = ring.lo.max(next + 1 - slots as i32).max(1);
        let (conv, ssm) = ring.views(next)?;
        self.conv_state = Some(conv);
        self.ssm_state = Some(ssm);
        self.offset = next;
        Ok(())
    }

    /// Store an externally computed post-step `(conv_state, ssm_state)` and advance the position
    /// by `step` tokens. With a ring the state is written into the slot of the new position (the
    /// positions a multi-token `step` skipped over are not restorable — use
    /// [`advance`](Self::advance) to checkpoint every token).
    pub fn update(&mut self, conv_state: Tensor, ssm_state: Tensor, step: i32) -> Result<()> {
        self.preallocate()?;
        let next = self.offset + step;
        match self.ring.as_mut() {
            Some(ring) => {
                ring.write(next, &conv_state, &ssm_state)?;
                ring.lo = if step == 1 {
                    ring.lo.max(next + 1 - ring.slots as i32).max(1)
                } else {
                    next
                };
                let (conv, ssm) = ring.views(next)?;
                self.conv_state = Some(conv);
                self.ssm_state = Some(ssm);
            }
            None => {
                self.conv_state = Some(conv_state);
                self.ssm_state = Some(ssm_state);
            }
        }
        self.offset = next;
        Ok(())
    }

    /// Roll back so the next step continues from position `n`: `n == offset()` is a no-op, `n == 0`
    /// is a [`reset`](Self::reset), any other `n` the ring holds selects that slot as the live
    /// state (no copy). A position the ring no longer holds — older than `slots - 1` positions
    /// back, or any interior position of a ring-less cache — is [`Error::RollbackUnavailable`]
    /// (typed; the cache is untouched). `n` outside `0..=offset()` is [`Error::Msg`].
    pub fn rollback_to(&mut self, n: i32) -> Result<()> {
        if n < 0 || n > self.offset {
            return Err(Error::Msg(format!(
                "DeltaNetCache: cannot roll back to {n} with {} positions cached",
                self.offset
            )));
        }
        if n == self.offset {
            return Ok(());
        }
        if n == 0 {
            self.reset();
            return Ok(());
        }
        if !self.can_rollback_to(n) {
            return Err(Error::RollbackUnavailable {
                n,
                have: self.restorable(),
            });
        }
        let ring = self.ring.as_ref().expect("can_rollback_to implies a ring");
        let (conv, ssm) = ring.views(n)?;
        self.conv_state = Some(conv);
        self.ssm_state = Some(ssm);
        self.offset = n;
        Ok(())
    }

    /// Drop all state, returning the cache to position zero. A ring keeps its buffers (and their
    /// addresses); it just holds no restorable position until the next forward.
    pub fn reset(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
        self.offset = 0;
        if let Some(ring) = self.ring.as_mut() {
            ring.lo = 1;
        }
    }

    /// A deep copy: the ring buffers are copied (two caches never share a ring, since the ring is
    /// written in place) and the live views re-derived. A ring-less cache clones its live tensors
    /// (reference-counted; they are never written in place).
    pub fn try_clone(&self) -> Result<Self> {
        let ring = match &self.ring {
            Some(r) => Some(StateRing {
                conv: r.conv.copy()?,
                ssm: r.ssm.copy()?,
                slots: r.slots,
                lo: r.lo,
            }),
            None => None,
        };
        let (conv_state, ssm_state) = match (&ring, self.conv_state.is_some()) {
            (Some(r), true) => {
                let (c, s) = r.views(self.offset)?;
                (Some(c), Some(s))
            }
            _ => (self.conv_state.clone(), self.ssm_state.clone()),
        };
        Ok(Self {
            conv_state,
            ssm_state,
            offset: self.offset,
            spec: self.spec.clone(),
            ring,
        })
    }

    /// `(live, checkpoint)` bytes: with a ring, one slot counts as live and the other `slots - 1`
    /// as checkpoints — the whole preallocation, from the moment the ring is specified (it is what
    /// the request holds); without one, the live tensors' bytes and no checkpoints.
    pub fn memory_bytes(&self) -> (usize, usize) {
        match &self.spec {
            Some(spec) => {
                let slot = spec.slot_bytes();
                (slot, slot.saturating_mul(spec.depth()))
            }
            None => (
                self.conv_state
                    .as_ref()
                    .map(tensor_bytes)
                    .unwrap_or(0)
                    .saturating_add(self.ssm_state.as_ref().map(tensor_bytes).unwrap_or(0)),
                0,
            ),
        }
    }

    /// The storage addresses of the ring's `(conv, ssm)` buffers (see [`storage_address`]), or
    /// `None` when no ring is allocated. Stable across every forward and rollback — the identity
    /// the CUDA-graph runner (S6) captures.
    pub fn ring_addresses(&self) -> Result<Option<(usize, usize)>> {
        match &self.ring {
            Some(r) => Ok(Some((storage_address(&r.conv)?, storage_address(&r.ssm)?))),
            None => Ok(None),
        }
    }
}

/// One recurrent step (`_gated_delta_step_ops`). `q`,`k` `[B,Hv,Dk]`; `v` `[B,Hv,Dv]`; `g`,`beta`
/// `[B,Hv]`; `state` `[B,Hv,Dv,Dk]`. Returns `(y [B,Hv,Dv], new_state [B,Hv,Dv,Dk])`.
#[allow(clippy::too_many_arguments)]
fn delta_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    b: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Result<(Tensor, Tensor)> {
    let decay = g.reshape((b, hv, 1, 1))?; // [B,Hv,1,1]
    let state = state.broadcast_mul(&decay)?; // S · g
    let k_r = k.reshape((b, hv, 1, dk))?; // [B,Hv,1,Dk]
    let kv_mem = state_read(&state, k, &k_r, b, hv, dk)?; // S·k → [B,Hv,Dv]
    let delta = v
        .broadcast_sub(&kv_mem)?
        .broadcast_mul(&beta.reshape((b, hv, 1))?)?; // (v−kv)·β → [B,Hv,Dv]
    let state = state.broadcast_add(&k_r.broadcast_mul(&delta.reshape((b, hv, dv, 1))?)?)?; // S + Δ⊗k
    let q_r = q.reshape((b, hv, 1, dk))?;
    let y = state_read(&state, q, &q_r, b, hv, dk)?; // S·q → [B,Hv,Dv]
    Ok((y, state))
}

/// Read the recurrent state with a head vector. CPU batched matvec avoids materializing a
/// full-state elementwise product for each read; CUDA/Metal retain the validated ops path.
fn state_read(
    state: &Tensor,
    vector: &Tensor,
    vector_row: &Tensor,
    b: usize,
    hv: usize,
    dk: usize,
) -> Result<Tensor> {
    if state.device().is_cpu() {
        Ok(state.matmul(&vector.reshape((b, hv, dk, 1))?)?.squeeze(3)?)
    } else {
        Ok(state.broadcast_mul(vector_row)?.sum(3)?)
    }
}

/// Repeat each head of `x` `[B,T,H,D]` `r` times along the head axis (contiguous), giving
/// `[B,T,H·r,D]` — the GQA expansion (`mx.repeat(x, r, axis=-2)`).
fn repeat_heads(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    Ok(x.unsqueeze(3)? // [B,T,H,1,D]
        .broadcast_as((b, t, h, r, d))? // [B,T,H,r,D]
        .contiguous()?
        .reshape((b, t, h * r, d))?) // [B,T,H·r,D]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Numeric oracle generated from `mlx_lm.models.gated_delta.gated_delta_update(..., use_kernel=
    // False)` (the ops reference) on seeded inputs — B=1, T=3, Hk=2, Hv=4 (GQA ×2), Dk=2, Dv=2.
    // Identical fixture to the verified mlx-llm port (sc-7627); a framework-independent check.
    const Q: &[f32] = &[
        -0.8805123, -0.2520431, 0.6974789, -0.8069373, 0.5367976, -0.5163696, 0.4404979, -0.721925,
        0.7900037, -0.5997525, 0.0088437, -0.2685511,
    ];
    const K: &[f32] = &[
        -0.5530653, -0.0336156, 0.34984, -0.9507952, 0.3385786, 0.7130073, 0.4130355, 0.7005113,
        -0.9424242, -0.8679754, 0.0911343, -0.1635016,
    ];
    const V: &[f32] = &[
        0.9033704, 0.9242766, -0.5594231, 0.5815381, -0.9665546, 0.8363392, 0.8484675, -0.1261052,
        -0.8719618, 0.0562458, 0.8790255, 0.7971791, -0.8360307, -0.2071904, -0.6701862,
        -0.7691332, -0.2697922, -0.7519733, 0.0098643, -0.2587476, 0.5248392, 0.9719371,
        -0.0304079, -0.2898675,
    ];
    const A: &[f32] = &[
        -0.6942793, -1.9834223, 1.6954608, 1.8882596, 1.5180013, 1.9647338, -0.7497661, -1.9337821,
        -1.3342639, 1.7648504, -0.3198055, -1.4922521,
    ];
    const B: &[f32] = &[
        -1.6042936, 1.7927692, 0.1977825, 0.2890182, 0.152185, -0.4371433, 0.8649859, 0.2619474,
        -1.2190499, -1.3681375, -1.4745429, 1.3650055,
    ];
    const A_LOG: &[f32] = &[2.0919125, 1.5201275, 2.7469416, 0.104467];
    const DT_BIAS: &[f32] = &[-0.2865368, -0.2390987, 0.5489618, 0.1692053];
    const EXP_Y: &[f32] = &[
        0.0749167, 0.0766504, -0.2376069, 0.2469999, -0.5368805, 0.4645513, 0.490568, -0.0729116,
        0.0874512, -0.0056413, -0.0642828, -0.0583461, 0.1904479, 0.0472361, 0.4258981, 0.0956538,
        0.0364119, 0.0369535, -0.0004748, 0.0117344, 0.0043713, 0.0080946, 0.1139456, 0.041226,
    ];
    const EXP_STATE: &[f32] = &[
        0.0431024, -0.0039364, 0.1626127, 0.1525815, -0.0018656, -0.0016657, 0.0495011, 0.0456383,
        0.0089079, -0.015984, 0.0164975, -0.0295984, 0.0197504, -0.4236471, -0.1824342, -0.1595202,
    ];

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn host(x: &Tensor) -> Vec<f32> {
        x.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn recurrence_matches_python_ops_reference() {
        let q = Tensor::from_slice(Q, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let k = Tensor::from_slice(K, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let v = Tensor::from_slice(V, (1, 3, 4, 2), &Device::Cpu).unwrap();
        let a = Tensor::from_slice(A, (1, 3, 4), &Device::Cpu).unwrap();
        let b_raw = Tensor::from_slice(B, (1, 3, 4), &Device::Cpu).unwrap();
        let a_log = Tensor::from_slice(A_LOG, (4,), &Device::Cpu).unwrap();
        let dt_bias = Tensor::from_slice(DT_BIAS, (4,), &Device::Cpu).unwrap();

        // beta = sigmoid(b); g = compute_g(a, A_log, dt_bias) — exactly what gated_delta_update does.
        let beta = candle_nn::ops::sigmoid(&b_raw).unwrap();
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        let (y, state) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        assert_eq!(y.dims(), &[1, 3, 4, 2]);
        assert_eq!(state.dims(), &[1, 4, 2, 2]);

        let yh = host(&y);
        let sh = host(&state);
        assert!(
            max_abs_diff(&yh, EXP_Y) < 1e-4,
            "y diff {}",
            max_abs_diff(&yh, EXP_Y)
        );
        assert!(
            max_abs_diff(&sh, EXP_STATE) < 1e-4,
            "state diff {}",
            max_abs_diff(&sh, EXP_STATE)
        );
    }

    #[test]
    fn decode_step_matches_chunked_prefill() {
        // Feeding the sequence one token at a time (carrying state) must equal one T-step call —
        // the prefill/decode equivalence the hybrid cache relies on.
        let q = Tensor::from_slice(Q, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let k = Tensor::from_slice(K, (1, 3, 2, 2), &Device::Cpu).unwrap();
        let v = Tensor::from_slice(V, (1, 3, 4, 2), &Device::Cpu).unwrap();
        let g = Tensor::from_slice(
            &[
                0.6f32, 0.7, 0.8, 0.9, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 0.4, 0.45,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let beta = Tensor::from_slice(
            &[
                0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.15, 0.25, 0.35,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();

        let (y_full, s_full) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        // Step one token at a time, carrying the state.
        let pick =
            |x: &Tensor, t: usize| -> Tensor { x.narrow(1, t, 1).unwrap().contiguous().unwrap() };
        let mut state: Option<Tensor> = None;
        let mut ys = Vec::new();
        for t in 0..3 {
            let (y, s) = gated_delta_recurrence(
                &pick(&q, t),
                &pick(&k, t),
                &pick(&v, t),
                &pick(&g, t),
                &pick(&beta, t),
                state.as_ref(),
            )
            .unwrap();
            ys.push(y);
            state = Some(s);
        }
        let refs: Vec<&Tensor> = ys.iter().collect();
        let y_step = Tensor::cat(&refs, 1).unwrap();

        assert!(
            max_abs_diff(&host(&y_full), &host(&y_step)) < 1e-5,
            "prefill vs step y: {}",
            max_abs_diff(&host(&y_full), &host(&y_step))
        );
        assert!(
            max_abs_diff(&host(&s_full), &host(&state.unwrap())) < 1e-5,
            "prefill vs step state"
        );
        assert_eq!(s_full.dims(), &[1, 4, 2, 2]);
        let mut cache = DeltaNetCache::new();
        cache
            .update(
                Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap(),
                s_full,
                3,
            )
            .unwrap();
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.ssm_state().unwrap().dims(), &[1, 4, 2, 2]);
    }

    #[test]
    fn cpu_state_read_matches_ops_at_qwen38_state_dims() {
        // A full frozen Qwen3.8 recurrent state: B=1, Hv=48, Dk=Dv=128 (3 MiB F32).
        let state = Tensor::from_vec(
            (0..48 * 128 * 128)
                .map(|i| ((i * 17 % 251) as f32 - 125.0) * 0.001)
                .collect(),
            (1, 48, 128, 128),
            &Device::Cpu,
        )
        .unwrap();
        let vector = Tensor::from_vec(
            (0..48 * 128)
                .map(|i| ((i * 29 % 197) as f32 - 98.0) * 0.001)
                .collect(),
            (1, 48, 128),
            &Device::Cpu,
        )
        .unwrap();
        let row = vector.reshape((1, 48, 1, 128)).unwrap();
        let expected = state.broadcast_mul(&row).unwrap().sum(3).unwrap();
        let got = state_read(&state, &vector, &row, 1, 48, 128).unwrap();
        assert_eq!(got.dims(), &[1, 48, 128]);
        let diff = max_abs_diff(&host(&expected), &host(&got));
        assert!(diff < 1e-6, "CPU batched state read versus ops diff {diff}");
    }

    #[test]
    fn cpu_batched_recurrence_matches_ops_and_chunks_with_two_batches() {
        const B: usize = 2;
        const T: usize = 5;
        const HK: usize = 2;
        const HV: usize = 4;
        const DK: usize = 3;
        const DV: usize = 5;
        let sample = |len: usize, salt: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 17 + salt) % 37) as f32 * scale - 18.0 * scale)
                .collect()
        };
        let q = Tensor::from_vec(
            sample(B * T * HK * DK, 1, 0.02),
            (B, T, HK, DK),
            &Device::Cpu,
        )
        .unwrap();
        let k = Tensor::from_vec(
            sample(B * T * HK * DK, 2, 0.03),
            (B, T, HK, DK),
            &Device::Cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            sample(B * T * HV * DV, 3, 0.04),
            (B, T, HV, DV),
            &Device::Cpu,
        )
        .unwrap();
        let g = Tensor::from_vec(
            sample(B * T * HV, 4, 0.001)
                .into_iter()
                .map(|x| 0.96 + x)
                .collect(),
            (B, T, HV),
            &Device::Cpu,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            sample(B * T * HV, 5, 0.002)
                .into_iter()
                .map(|x| 0.4 + x)
                .collect(),
            (B, T, HV),
            &Device::Cpu,
        )
        .unwrap();

        let (actual_y, actual_state) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();
        assert_eq!(actual_y.dims(), &[B, T, HV, DV]);
        assert_eq!(actual_state.dims(), &[B, HV, DV, DK]);

        // The original broadcast-product/sum path is a separate multi-token numeric reference.
        let q_full = repeat_heads(&q, HV / HK).unwrap();
        let k_full = repeat_heads(&k, HV / HK).unwrap();
        let mut reference_state = Tensor::zeros((B, HV, DV, DK), DType::F32, &Device::Cpu).unwrap();
        let mut reference_ys = Vec::new();
        for ti in 0..T {
            let pick = |x: &Tensor| x.narrow(1, ti, 1).unwrap().squeeze(1).unwrap();
            let (qt, kt, vt, gt, bt) = (
                pick(&q_full),
                pick(&k_full),
                pick(&v),
                pick(&g),
                pick(&beta),
            );
            let decayed = reference_state
                .broadcast_mul(&gt.reshape((B, HV, 1, 1)).unwrap())
                .unwrap();
            let k_row = kt.reshape((B, HV, 1, DK)).unwrap();
            let remembered = decayed.broadcast_mul(&k_row).unwrap().sum(3).unwrap();
            let delta = vt
                .broadcast_sub(&remembered)
                .unwrap()
                .broadcast_mul(&bt.reshape((B, HV, 1)).unwrap())
                .unwrap();
            reference_state = decayed
                .broadcast_add(
                    &k_row
                        .broadcast_mul(&delta.reshape((B, HV, DV, 1)).unwrap())
                        .unwrap(),
                )
                .unwrap();
            let y = reference_state
                .broadcast_mul(&qt.reshape((B, HV, 1, DK)).unwrap())
                .unwrap()
                .sum(3)
                .unwrap();
            reference_ys.push(y.unsqueeze(1).unwrap());
        }
        let refs: Vec<_> = reference_ys.iter().collect();
        let reference_y = Tensor::cat(&refs, 1).unwrap();
        assert!(max_abs_diff(&host(&actual_y), &host(&reference_y)) < 1e-5);
        assert!(max_abs_diff(&host(&actual_state), &host(&reference_state)) < 1e-5);

        // The second batch makes a middle-sequence slice noncontiguous in its batch dimension.
        let vector = q_full.narrow(1, 1, 1).unwrap().squeeze(1).unwrap();
        assert!(!vector.is_contiguous());
        let row = vector.reshape((B, HV, 1, DK)).unwrap();
        let expected_read = actual_state.broadcast_mul(&row).unwrap().sum(3).unwrap();
        let actual_read = state_read(&actual_state, &vector, &row, B, HV, DK).unwrap();
        assert!(max_abs_diff(&host(&actual_read), &host(&expected_read)) < 1e-5);

        let mut carried = None;
        let mut chunks = Vec::new();
        for (start, len) in [(0, 2), (2, 3)] {
            let (y, state) = gated_delta_recurrence(
                &q.narrow(1, start, len).unwrap(),
                &k.narrow(1, start, len).unwrap(),
                &v.narrow(1, start, len).unwrap(),
                &g.narrow(1, start, len).unwrap(),
                &beta.narrow(1, start, len).unwrap(),
                carried.as_ref(),
            )
            .unwrap();
            chunks.push(y);
            carried = Some(state);
        }
        let refs: Vec<_> = chunks.iter().collect();
        let chunked_y = Tensor::cat(&refs, 1).unwrap();
        assert!(max_abs_diff(&host(&actual_y), &host(&chunked_y)) < 1e-5);
        assert!(max_abs_diff(&host(&actual_state), &host(&carried.unwrap())) < 1e-5);
    }

    /// Deterministic uniform `[-1, 1)` values (a 64-bit LCG), so the parity fixtures need no RNG state.
    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The device the chunked-parity tests run on: the build's GPU backend when one is compiled
    /// in and present (so the CUDA and Metal lanes execute the chunked form on the GPU), else CPU.
    fn test_device() -> Device {
        #[cfg(feature = "cuda")]
        let device = Device::cuda_if_available(0).unwrap();
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        let device = Device::metal_if_available(0).unwrap();
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = Device::Cpu;
        device
    }

    /// Model-shaped recurrence inputs `(q, k, v, g, β, carried state)` on [`test_device`]:
    /// L2-normalized q (scaled by `1/√Dk`) and k, gates from [`compute_g`], `β = sigmoid(b)` — all
    /// rounded through `dtype` so the f32 reference sees exactly the values the path under test
    /// reads.
    fn model_inputs(dims: [usize; 6], dtype: DType, seed: u64) -> [Tensor; 6] {
        let [b, t, hk, hv, dk, dv] = dims;
        let dev = &test_device();
        let arr = |shape: &[usize], s: u64| {
            let n = shape.iter().product();
            Tensor::from_vec(lcg(n, seed ^ s), shape, dev).unwrap()
        };
        let l2 = |x: Tensor| {
            let norm = x
                .sqr()
                .unwrap()
                .sum_keepdim(3)
                .unwrap()
                .affine(1.0, 1e-6)
                .unwrap();
            x.broadcast_div(&norm.sqrt().unwrap()).unwrap()
        };
        let round = |x: Tensor| x.to_dtype(dtype).unwrap();
        let q = l2(arr(&[b, t, hk, dk], 1))
            .affine((dk as f64).powf(-0.5), 0.0)
            .unwrap();
        let k = l2(arr(&[b, t, hk, dk], 2));
        let a = arr(&[b, t, hv], 4).affine(3.0, 0.0).unwrap();
        let g = compute_g(&a, &arr(&[hv], 5), &arr(&[hv], 6)).unwrap();
        let beta = candle_nn::ops::sigmoid(&arr(&[b, t, hv], 7).affine(2.0, 0.0).unwrap()).unwrap();
        [
            round(q),
            round(k),
            round(arr(&[b, t, hv, dv], 3)),
            round(g),
            round(beta),
            arr(&[b, hv, dv, dk], 8).affine(0.5, 0.0).unwrap(),
        ]
    }

    /// `(max |a − r| / max |r|, max |a − r|)` — scale-relative and absolute error against the
    /// reference `r` (never a cosine: that is scale-invariant).
    fn errors(a: &Tensor, r: &Tensor) -> (f32, f32) {
        assert_eq!(a.dims(), r.dims());
        let (a, r) = (host(a), host(r));
        assert!(a.iter().all(|x| x.is_finite()), "non-finite output");
        let abs = max_abs_diff(&a, &r);
        let scale = r.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        (abs / scale, abs)
    }

    /// The f32 per-token reference on (already dtype-rounded) inputs.
    fn per_token_reference(x: &[Tensor; 6], carry: bool) -> (Tensor, Tensor) {
        let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
        let state = carry.then(|| f(&x[5]));
        gated_delta_recurrence_per_token(
            &f(&x[0]),
            &f(&x[1]),
            &f(&x[2]),
            &f(&x[3]),
            &f(&x[4]),
            state.as_ref(),
            &mut |_, _| Ok(()),
        )
        .unwrap()
    }

    /// AC (sc-24443): the chunkwise prefill matches the per-token reference's outputs and final
    /// state at 512 and 2048 tokens (plus a batch of two at a length that is not a whole number of
    /// chunks), GQA ×2 at the Qwen3.6 head dims, on f32, bf16 and f16 inputs (computed in f32),
    /// from a zero and from a carried state.
    #[test]
    fn chunked_prefill_matches_per_token_reference() {
        let cases: [(&str, [usize; 6]); 3] = [
            ("512", [1, 512, 2, 4, 128, 128]),
            ("2048", [1, 2048, 2, 4, 128, 128]),
            ("ragged", [2, 300, 1, 2, 64, 32]),
        ];
        for (label, dims) in cases {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                for carry in [false, true] {
                    let x = model_inputs(dims, dtype, dims.iter().product::<usize>() as u64);
                    let state = carry.then_some(&x[5]);
                    let (y, s) =
                        gated_delta_chunked(&x[0], &x[1], &x[2], &x[3], &x[4], state).unwrap();
                    assert_eq!((y.dtype(), s.dtype()), (DType::F32, DType::F32));
                    let (y_ref, s_ref) = per_token_reference(&x, carry);
                    let (y_rel, y_abs) = errors(&y, &y_ref);
                    let (s_rel, s_abs) = errors(&s, &s_ref);
                    eprintln!(
                        "chunked {label} {dims:?} {dtype:?} carry={carry}: y rel {y_rel:.2e} abs \
                         {y_abs:.2e} | state rel {s_rel:.2e} abs {s_abs:.2e}"
                    );
                    assert!(
                        y_rel <= 2e-5 && y_abs <= 2e-5 && s_rel <= 2e-5 && s_abs <= 2e-5,
                        "chunked {label} {dtype:?} carry={carry}: y rel {y_rel} abs {y_abs}, \
                         state rel {s_rel} abs {s_abs}"
                    );
                }
            }
        }
    }

    /// The dispatch: a prefill runs its un-checkpointed head chunkwise and steps only the sunk
    /// tail per token — every sunk state and the outputs match the per-token reference — while a
    /// run shorter than one chunk (decode, verify) stays per-token; outputs keep the input dtypes.
    #[test]
    fn prefill_runs_its_head_chunkwise_and_sinks_the_tail_per_token() {
        let x = model_inputs([1, 200, 2, 4, 32, 16], DType::F32, 11);
        let (y_ref, _) = per_token_reference(&x, true);
        let mut ref_states = Vec::new();
        gated_delta_recurrence_per_token(
            &x[0],
            &x[1],
            &x[2],
            &x[3],
            &x[4],
            Some(&x[5]),
            &mut |ti, s| {
                ref_states.push((ti, s.clone()));
                Ok(())
            },
        )
        .unwrap();
        for sink_from in [0, 63, 64, 190, 200] {
            let mut seen = Vec::new();
            let (y, last) = gated_delta_recurrence_with_sink(
                &x[0],
                &x[1],
                &x[2],
                &x[3],
                &x[4],
                Some(&x[5]),
                sink_from,
                &mut |ti, s| {
                    seen.push((ti, s.clone()));
                    Ok(())
                },
            )
            .unwrap();
            assert!(errors(&y, &y_ref).0 < 2e-5, "sink_from {sink_from}");
            assert!(
                errors(&last, &ref_states[199].1).0 < 2e-5,
                "sink_from {sink_from}"
            );
            // A head shorter than one chunk steps per token, sinking every token.
            let first = if sink_from < CHUNKED_PREFILL_MIN_TOKENS {
                0
            } else {
                sink_from
            };
            assert_eq!(
                seen.iter().map(|(ti, _)| *ti).collect::<Vec<_>>(),
                (first..200).collect::<Vec<_>>()
            );
            for (ti, s) in &seen {
                assert!(errors(s, &ref_states[*ti].1).0 < 2e-5, "state after {ti}");
            }
        }
        let bf = model_inputs([1, 100, 2, 4, 32, 16], DType::BF16, 12);
        let (y, s) = gated_delta_recurrence(&bf[0], &bf[1], &bf[2], &bf[3], &bf[4], None).unwrap();
        assert_eq!((y.dtype(), s.dtype()), (DType::BF16, DType::BF16));
        let (_, s) =
            gated_delta_recurrence(&bf[0], &bf[1], &bf[2], &bf[3], &bf[4], Some(&bf[5])).unwrap();
        assert_eq!(s.dtype(), DType::F32);
    }

    /// A checkpointing cache's long prefill (chunkwise head, per-token ringed tail) restores every
    /// ringed position to the state a token-at-a-time decode reaches.
    #[test]
    fn ring_prefill_with_a_chunked_head_restores_the_per_token_states() {
        let device = test_device();
        let fixture = ring_inputs_on(150, 2, &device);
        let mut cache = DeltaNetCache::with_ring(ring_spec_on(4, device)).unwrap();
        feed(&mut cache, &fixture, 0, 150);
        assert_eq!(cache.restorable(), vec![147, 148, 149]);
        for n in [150, 149, 148, 147] {
            cache.rollback_to(n).unwrap();
            let (conv, ssm) = live(&cache).unwrap();
            let (fconv, fssm) = fresh_state(&fixture, n as usize).unwrap();
            let scale = fssm.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert_eq!(conv, fconv, "conv tail after {n}");
            assert!(
                max_abs_diff(&ssm, &fssm) <= 2e-5 * scale,
                "ssm after {n}: {}",
                max_abs_diff(&ssm, &fssm)
            );
        }
    }

    #[test]
    fn compute_g_is_in_unit_interval_and_shaped() {
        // g = exp(−positive) ∈ (0, 1]: a per-head forget gate.
        let a = Tensor::from_slice(A, (1, 3, 4), &Device::Cpu).unwrap();
        let a_log = Tensor::from_slice(A_LOG, (4,), &Device::Cpu).unwrap();
        let dt_bias = Tensor::from_slice(DT_BIAS, (4,), &Device::Cpu).unwrap();
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        assert_eq!(g.dims(), &[1, 3, 4]);
        for x in host(&g) {
            assert!(x > 0.0 && x <= 1.0 + 1e-6, "gate out of (0,1]: {x}");
        }
    }

    // Conv oracle from MLX's depthwise `nn.Conv1d` + silu — C=3, K=4, S=2, conv_state K-1=3.
    const CW: &[f32] = &[
        0.2422637, 0.3207079, 0.3157361, 0.493552, 0.6888506, 0.2758986, -0.2604986, 0.5719915,
        -0.677193, -0.1786878, -0.1721832, 0.024104,
    ];
    const CX: &[f32] = &[
        -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096, 0.7106992,
    ];
    const CSTATE: &[f32] = &[
        0.2526282, 0.8012137, 0.0075967, -0.7815045, -0.8113322, -0.8019857, 0.0879031, 0.8514872,
        0.00599,
    ];
    const CEXP_OUT: &[f32] = &[
        -0.1465172, 0.0095216, 0.0727914, -0.1432332, -0.2082713, 0.3602719,
    ];
    const CEXP_STATE: &[f32] = &[
        0.0879031, 0.8514872, 0.00599, -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096,
        0.7106992,
    ];

    #[test]
    fn causal_conv_matches_mlx_conv1d() {
        // weight stored [C,K,1] in the checkpoint; the helper takes [C,K] (squeezed).
        let weight = Tensor::from_slice(CW, (3, 4), &Device::Cpu).unwrap();
        let x = Tensor::from_slice(CX, (1, 2, 3), &Device::Cpu).unwrap();
        let state = Tensor::from_slice(CSTATE, (1, 3, 3), &Device::Cpu).unwrap();
        let (out, new_state) = causal_depthwise_conv(&x, &weight, &state).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3]);
        assert_eq!(new_state.dims(), &[1, 3, 3]);
        assert!(max_abs_diff(&host(&out), CEXP_OUT) < 1e-5);
        // new conv_state is the last K-1 tokens of [conv_state ++ x] — exact (a slice, no arithmetic).
        assert_eq!(host(&new_state), CEXP_STATE.to_vec());
    }

    #[test]
    fn delta_cache_tracks_state_and_offset() {
        let mut cache = DeltaNetCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state().is_none() && cache.ssm_state().is_none());
        let conv = Tensor::zeros((1, 3, 3), DType::F32, &Device::Cpu).unwrap();
        let ssm = Tensor::zeros((1, 4, 2, 2), DType::F32, &Device::Cpu).unwrap();
        cache.update(conv, ssm, 5).unwrap();
        assert_eq!(cache.offset(), 5);
        assert!(cache.conv_state().is_some() && cache.ssm_state().is_some());
        assert_eq!(cache.depth(), 0);
        assert!(cache.restorable().is_empty());
        // A ring-less cache can roll back to zero and the current position only.
        assert!(matches!(
            cache.rollback_to(3),
            Err(Error::RollbackUnavailable { n: 3, ref have }) if have.is_empty()
        ));
        assert_eq!(cache.offset(), 5);
        assert!(matches!(cache.rollback_to(6), Err(Error::Msg(_))));
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state().is_none());
    }

    // Reproducible CPU operation comparison. This test is deliberately ignored because elapsed
    // time is diagnostic evidence, not a CI gate; normal tests above pin the numeric contract.
    #[test]
    #[ignore]
    fn qwen38_cpu_delta_matvec_experiment() {
        use std::time::Instant;

        // Frozen Qwen3.8 config: Hk=16, Hv=48, Dk=Dv=128. Values are seeded and kept bounded so
        // both algorithms run the same stable recurrence without a real checkpoint or device.
        const B: usize = 1;
        const HK: usize = 16;
        const HV: usize = 48;
        const D: usize = 128;

        fn seeded(len: usize, seed: u32, scale: f32) -> Vec<f32> {
            let mut state = seed;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
                })
                .collect()
        }

        fn ops_step(
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            g: &Tensor,
            beta: &Tensor,
            state: &Tensor,
        ) -> Result<(Tensor, Tensor)> {
            let (b, hv, dv, dk) = state.dims4()?;
            let decay = g.reshape((b, hv, 1, 1))?;
            let state = state.broadcast_mul(&decay)?;
            let k_r = k.reshape((b, hv, 1, dk))?;
            let kv_mem = state.broadcast_mul(&k_r)?.sum(3)?;
            let delta = v
                .broadcast_sub(&kv_mem)?
                .broadcast_mul(&beta.reshape((b, hv, 1))?)?;
            let state =
                state.broadcast_add(&k_r.broadcast_mul(&delta.reshape((b, hv, dv, 1))?)?)?;
            let y = state.broadcast_mul(&q.reshape((b, hv, 1, dk))?)?.sum(3)?;
            Ok((y, state))
        }

        fn run_ops(
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            g: &Tensor,
            beta: &Tensor,
        ) -> Result<(Tensor, Tensor)> {
            let t = q.dim(1)?;
            let q = repeat_heads(q, HV / HK)?;
            let k = repeat_heads(k, HV / HK)?;
            let mut state = Tensor::zeros((B, HV, D, D), DType::F32, &Device::Cpu)?;
            let mut ys = Vec::with_capacity(t);
            for ti in 0..t {
                let qt = q.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?;
                let kt = k.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?;
                let vt = v.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?;
                let gt = g.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?;
                let bt = beta.narrow(1, ti, 1)?.squeeze(1)?.contiguous()?;
                let (y, next) = ops_step(&qt, &kt, &vt, &gt, &bt, &state)?;
                state = next;
                ys.push(y.unsqueeze(1)?);
            }
            let refs: Vec<_> = ys.iter().collect();
            Ok((Tensor::cat(&refs, 1)?, state))
        }

        fn run_chunked(
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            g: &Tensor,
            beta: &Tensor,
        ) -> Result<(Tensor, Tensor)> {
            let tokens = q.dim(1)?;
            let mut state = None;
            let mut outputs = Vec::new();
            for start in (0..tokens).step_by(32) {
                let len = 32.min(tokens - start);
                let (y, next) = gated_delta_recurrence(
                    &q.narrow(1, start, len)?,
                    &k.narrow(1, start, len)?,
                    &v.narrow(1, start, len)?,
                    &g.narrow(1, start, len)?,
                    &beta.narrow(1, start, len)?,
                    state.as_ref(),
                )?;
                outputs.push(y);
                state = Some(next);
            }
            let refs: Vec<_> = outputs.iter().collect();
            Ok((Tensor::cat(&refs, 1)?, state.unwrap()))
        }

        for tokens in [64, 256] {
            let q = Tensor::from_vec(
                // Approximate Q component magnitude after L2 norm and 1/sqrt(D) scaling.
                seeded(tokens * HK * D, 0x74ab_0191, 0.015),
                (B, tokens, HK, D),
                &Device::Cpu,
            )
            .unwrap();
            let k = Tensor::from_vec(
                seeded(tokens * HK * D, 0x1baf_72d5, 0.15),
                (B, tokens, HK, D),
                &Device::Cpu,
            )
            .unwrap();
            let v = Tensor::from_vec(
                seeded(tokens * HV * D, 0xf29a_1043, 0.5),
                (B, tokens, HV, D),
                &Device::Cpu,
            )
            .unwrap();
            let g = Tensor::from_vec(
                seeded(tokens * HV, 0x106b_87ca, 0.02)
                    .into_iter()
                    .map(|x| 0.97 + x)
                    .collect(),
                (B, tokens, HV),
                &Device::Cpu,
            )
            .unwrap();
            let beta = Tensor::from_vec(
                seeded(tokens * HV, 0x7c29_13bd, 0.02)
                    .into_iter()
                    .map(|x| 0.5 + x)
                    .collect(),
                (B, tokens, HV),
                &Device::Cpu,
            )
            .unwrap();

            let start = Instant::now();
            let (reference_y, reference_state) = run_ops(&q, &k, &v, &g, &beta).unwrap();
            let reference_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let (candidate_y, candidate_state) =
                gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();
            let candidate_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Reverse execution order once to expose a simple warm-cache/order artifact.
            let start = Instant::now();
            let _candidate_again = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();
            let candidate_reverse_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let _reference_again = run_ops(&q, &k, &v, &g, &beta).unwrap();
            let reference_reverse_ms = start.elapsed().as_secs_f64() * 1000.0;

            let output_diff = max_abs_diff(&host(&reference_y), &host(&candidate_y));
            let state_diff = max_abs_diff(&host(&reference_state), &host(&candidate_state));
            let (chunked_y, chunked_state) = run_chunked(&q, &k, &v, &g, &beta).unwrap();
            let chunked_output_diff = max_abs_diff(&host(&candidate_y), &host(&chunked_y));
            let chunked_state_diff = max_abs_diff(&host(&candidate_state), &host(&chunked_state));
            let output_max = host(&candidate_y)
                .into_iter()
                .map(f32::abs)
                .fold(0.0, f32::max);
            let state_max = host(&candidate_state)
                .into_iter()
                .map(f32::abs)
                .fold(0.0, f32::max);
            println!(
                "qwen38_cpu_delta_matvec tokens={tokens} reference_ms={reference_ms:.3} candidate_ms={candidate_ms:.3} candidate_reverse_ms={candidate_reverse_ms:.3} reference_reverse_ms={reference_reverse_ms:.3} output_max={output_max:e} state_max={state_max:e} output_max_abs_diff={output_diff:e} state_max_abs_diff={state_diff:e} chunked_output_diff={chunked_output_diff:e} chunked_state_diff={chunked_state_diff:e}"
            );
            assert!(output_diff < 1e-4 && state_diff < 1e-4);
            assert!(chunked_output_diff < 1e-5 && chunked_state_diff < 1e-5);
        }
    }

    // ---- The per-token checkpoint ring (sc-24131) ----

    const RB: usize = 1;
    const RHK: usize = 2;
    const RHV: usize = 4;
    const RDK: usize = 2;
    const RDV: usize = 2;
    const RC: usize = 3; // conv channels
    const RK: usize = 4; // conv kernel

    type Fixture = (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);

    fn ring_spec(slots: usize) -> RingSpec {
        ring_spec_on(slots, Device::Cpu)
    }

    fn ring_spec_on(slots: usize, device: Device) -> RingSpec {
        RingSpec {
            slots,
            conv_dims: (RB, RK - 1, RC),
            conv_dtype: DType::F32,
            ssm_dims: (RB, RHV, RDV, RDK),
            ssm_dtype: DType::F32,
            device,
        }
    }

    /// Seeded recurrence inputs for `t` tokens plus the conv input `x [B, t, C]`.
    fn ring_inputs(t: usize, salt: usize) -> Fixture {
        ring_inputs_on(t, salt, &Device::Cpu)
    }

    /// [`ring_inputs`] on `dev`.
    fn ring_inputs_on(t: usize, salt: usize, dev: &Device) -> Fixture {
        let sample = |len: usize, s: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 31 + s * 7) % 41) as f32 * scale - 20.0 * scale)
                .collect()
        };
        let q = Tensor::from_vec(
            sample(t * RHK * RDK, salt + 1, 0.05),
            (RB, t, RHK, RDK),
            dev,
        )
        .unwrap();
        let k = Tensor::from_vec(
            sample(t * RHK * RDK, salt + 2, 0.04),
            (RB, t, RHK, RDK),
            dev,
        )
        .unwrap();
        let v = Tensor::from_vec(
            sample(t * RHV * RDV, salt + 3, 0.06),
            (RB, t, RHV, RDV),
            dev,
        )
        .unwrap();
        let g = Tensor::from_vec(
            sample(t * RHV, salt + 4, 0.001)
                .into_iter()
                .map(|x| 0.95 + x)
                .collect(),
            (RB, t, RHV),
            dev,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            sample(t * RHV, salt + 5, 0.002)
                .into_iter()
                .map(|x| 0.5 + x)
                .collect(),
            (RB, t, RHV),
            dev,
        )
        .unwrap();
        let x = Tensor::from_vec(sample(t * RC, salt + 6, 0.1), (RB, t, RC), dev).unwrap();
        (q, k, v, g, beta, x)
    }

    fn narrow_t(x: &Tensor, start: usize, len: usize) -> Tensor {
        x.narrow(1, start, len).unwrap().contiguous().unwrap()
    }

    /// Feed tokens `start..start+len` of the fixture through `cache` (one `advance`).
    fn feed(cache: &mut DeltaNetCache, fixture: &Fixture, start: usize, len: usize) -> Tensor {
        let (q, k, v, g, beta, x) = fixture;
        let weight = Tensor::from_slice(CW, (RC, RK), x.device()).unwrap();
        let conv_state = match cache.conv_state() {
            Some(c) => c.clone(),
            None => Tensor::zeros((RB, RK - 1, RC), DType::F32, x.device()).unwrap(),
        };
        let (_out, trace) =
            causal_depthwise_conv_traced(&narrow_t(x, start, len), &weight, &conv_state).unwrap();
        cache
            .advance(
                &trace,
                &narrow_t(q, start, len),
                &narrow_t(k, start, len),
                &narrow_t(v, start, len),
                &narrow_t(g, start, len),
                &narrow_t(beta, start, len),
            )
            .unwrap()
    }

    /// The `(conv, ssm)` state after a fresh decode of the fixture's first `n` tokens, one at a
    /// time on a ring-less cache (the oracle for AC1).
    fn fresh_state(fixture: &Fixture, n: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let mut cache = DeltaNetCache::new();
        for i in 0..n {
            feed(&mut cache, fixture, i, 1);
        }
        Some((host(cache.conv_state()?), host(cache.ssm_state()?)))
    }

    fn live(cache: &DeltaNetCache) -> Option<(Vec<f32>, Vec<f32>)> {
        Some((host(cache.conv_state()?), host(cache.ssm_state()?)))
    }

    #[test]
    fn sink_receives_every_post_token_state_in_order() {
        let (q, k, v, g, beta, _) = ring_inputs(4, 0);
        let mut seen = Vec::new();
        let (_, final_state) =
            gated_delta_recurrence_with_sink(&q, &k, &v, &g, &beta, None, 0, &mut |ti, s| {
                seen.push((ti, host(s)));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            seen.iter().map(|(ti, _)| *ti).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        for (ti, state) in &seen {
            let (_, prefix) = gated_delta_recurrence(
                &narrow_t(&q, 0, ti + 1),
                &narrow_t(&k, 0, ti + 1),
                &narrow_t(&v, 0, ti + 1),
                &narrow_t(&g, 0, ti + 1),
                &narrow_t(&beta, 0, ti + 1),
                None,
            )
            .unwrap();
            assert_eq!(*state, host(&prefix), "state after token {ti}");
        }
        assert_eq!(seen.last().unwrap().1, host(&final_state));
        // A sink error aborts the recurrence.
        let err = gated_delta_recurrence_with_sink(&q, &k, &v, &g, &beta, None, 0, &mut |_, _| {
            Err(Error::Msg("stop".into()))
        });
        assert!(matches!(err, Err(Error::Msg(m)) if m == "stop"));
    }

    #[test]
    fn conv_trace_tail_after_every_token_matches_a_token_at_a_time_conv() {
        let weight = Tensor::from_slice(CW, (RC, RK), &Device::Cpu).unwrap();
        let (_, _, _, _, _, x) = ring_inputs(5, 3);
        let seed = Tensor::from_slice(CSTATE, (1, 3, 3), &Device::Cpu).unwrap();
        let (out_all, trace) = causal_depthwise_conv_traced(&x, &weight, &seed).unwrap();
        assert_eq!((trace.tokens(), trace.tail_len()), (5, RK - 1));
        let mut state = seed;
        for ti in 0..5 {
            let (out, next) = causal_depthwise_conv(&narrow_t(&x, ti, 1), &weight, &state).unwrap();
            assert_eq!(host(&out), host(&out_all.narrow(1, ti, 1).unwrap()));
            assert_eq!(host(&trace.tail_after(ti).unwrap()), host(&next));
            state = next;
        }
        assert!(trace.tail_after(5).is_err());
    }

    /// AC1 at the primitive: after one `K + 1`-token forward (the verify step) every interior
    /// position is restorable and equals a fresh token-at-a-time decode — conv tail and SSM state,
    /// max abs error `<= 1e-6` — for every `K in 1..=5`, `j in 0..=K + 1`; the rollback is a slot
    /// selection (the ring's addresses never change) and positions past the ring's depth are the
    /// typed refusal.
    #[test]
    fn ring_restores_every_position_of_a_verify_step_exactly() {
        for k in 1..=5usize {
            let fixture = ring_inputs(3 + k + 1, k);
            // The engine's cache for `K` drafts: `K + 2` slots (the step start + K + 1 positions).
            let mut cache = DeltaNetCache::with_ring(ring_spec(k + 2)).unwrap();
            cache.preallocate().unwrap();
            let addresses = cache.ring_addresses().unwrap().unwrap();
            feed(&mut cache, &fixture, 0, 3); // the "prompt"
            assert_eq!(cache.offset(), 3);
            feed(&mut cache, &fixture, 3, k + 1); // verify `[cur, d1..dK]`
            let end = 3 + k as i32 + 1;
            assert_eq!(cache.offset(), end);
            assert_eq!(cache.ring_addresses().unwrap().unwrap(), addresses);
            assert_eq!(
                cache.restorable(),
                (3..end).collect::<Vec<_>>(),
                "the step start survives the verify step (K={k})"
            );
            for j in (0..=k + 1).rev() {
                let n = 3 + j;
                cache.rollback_to(n as i32).unwrap();
                assert_eq!(cache.offset(), n as i32);
                assert_eq!(cache.ring_addresses().unwrap().unwrap(), addresses);
                let (conv, ssm) = live(&cache).unwrap();
                let (fconv, fssm) = fresh_state(&fixture, n).unwrap();
                let conv_err = max_abs_diff(&conv, &fconv);
                let ssm_err = max_abs_diff(&ssm, &fssm);
                assert!(
                    conv_err <= 1e-6 && ssm_err <= 1e-6,
                    "K={k} j={j}: conv {conv_err:e} ssm {ssm_err:e}"
                );
            }
            // Below the ring's window (position 2 was the prompt's interior) is refused, typed.
            cache.rollback_to(3).unwrap();
            match cache.rollback_to(2) {
                Err(Error::RollbackUnavailable { n: 2, have }) => assert!(have.is_empty()),
                other => panic!("K={k}: expected RollbackUnavailable, got {other:?}"),
            }
            assert_eq!(cache.offset(), 3, "a refused rollback is a no-op");
            // Continuing after a rollback keeps matching the fresh decode.
            feed(&mut cache, &fixture, 3, 2);
            let (conv, ssm) = live(&cache).unwrap();
            let (fconv, fssm) = fresh_state(&fixture, 5).unwrap();
            assert!(max_abs_diff(&conv, &fconv) <= 1e-6 && max_abs_diff(&ssm, &fssm) <= 1e-6);
            assert_eq!(cache.ring_addresses().unwrap().unwrap(), addresses);
        }
    }

    #[test]
    fn ring_window_slides_with_single_token_decodes_and_zero_resets() {
        let fixture = ring_inputs(8, 11);
        let mut cache = DeltaNetCache::with_ring(ring_spec(3)).unwrap(); // depth 2
        assert_eq!(cache.depth(), 2);
        assert!(cache.restorable().is_empty());
        for i in 0..6 {
            feed(&mut cache, &fixture, i, 1);
        }
        assert_eq!(cache.offset(), 6);
        assert_eq!(cache.restorable(), vec![4, 5]);
        assert!(cache.can_rollback_to(4) && cache.can_rollback_to(0) && cache.can_rollback_to(6));
        assert!(!cache.can_rollback_to(3));
        assert!(matches!(
            cache.rollback_to(3),
            Err(Error::RollbackUnavailable { n: 3, ref have }) if *have == vec![4, 5]
        ));
        cache.rollback_to(4).unwrap();
        assert_eq!(live(&cache), fresh_state(&fixture, 4));
        assert_eq!(
            cache.restorable(),
            Vec::<i32>::new(),
            "nothing below 4 is held"
        );
        // Zero is always reachable; the ring keeps its buffers.
        let addresses = cache.ring_addresses().unwrap().unwrap();
        cache.rollback_to(0).unwrap();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state().is_none());
        assert_eq!(cache.ring_addresses().unwrap().unwrap(), addresses);
        feed(&mut cache, &fixture, 0, 2);
        assert_eq!(cache.restorable(), vec![1]);
        assert_eq!(live(&cache), fresh_state(&fixture, 2));
        // A multi-token forward longer than the ring keeps only its last `slots` positions.
        cache.reset();
        feed(&mut cache, &fixture, 0, 8);
        assert_eq!(cache.restorable(), vec![6, 7]);
        cache.rollback_to(6).unwrap();
        assert_eq!(live(&cache), fresh_state(&fixture, 6));
    }

    #[test]
    fn ring_reallocation_carries_restorable_positions_and_clone_is_independent() {
        let fixture = ring_inputs(8, 5);
        let mut cache = DeltaNetCache::with_ring(ring_spec(4)).unwrap();
        for i in 0..5 {
            feed(&mut cache, &fixture, i, 1);
        }
        assert_eq!(cache.restorable(), vec![2, 3, 4]);
        // Deeper: everything held is carried over.
        cache.set_ring_spec(Some(ring_spec(6))).unwrap();
        assert_eq!(cache.depth(), 5);
        assert_eq!(cache.restorable(), vec![2, 3, 4]);
        assert_eq!(live(&cache), fresh_state(&fixture, 5));
        cache.rollback_to(2).unwrap();
        assert_eq!(live(&cache), fresh_state(&fixture, 2));
        cache.rollback_to(5).unwrap_err(); // past the end now
        feed(&mut cache, &fixture, 2, 3);
        assert_eq!(cache.restorable(), vec![2, 3, 4]);
        // Shallower: only the newest positions survive.
        cache.set_ring_spec(Some(ring_spec(2))).unwrap();
        assert_eq!(cache.restorable(), vec![4]);
        assert_eq!(live(&cache), fresh_state(&fixture, 5));
        cache.rollback_to(4).unwrap();
        assert_eq!(live(&cache), fresh_state(&fixture, 4));
        // Fewer than two slots is refused.
        assert!(matches!(
            cache.set_ring_spec(Some(ring_spec(1))),
            Err(Error::Msg(_))
        ));
        // Dropping the ring keeps the live state (detached from the freed buffer).
        cache.set_ring_spec(None).unwrap();
        assert_eq!(cache.depth(), 0);
        assert!(cache.ring_addresses().unwrap().is_none());
        assert_eq!(live(&cache), fresh_state(&fixture, 4));
        assert!(cache.restorable().is_empty());
        // A ring-less cache with live state gains a ring holding just its current position.
        cache.set_ring_spec(Some(ring_spec(3))).unwrap();
        assert_eq!(cache.offset(), 4);
        assert!(cache.restorable().is_empty());
        assert_eq!(live(&cache), fresh_state(&fixture, 4));
        feed(&mut cache, &fixture, 4, 1);
        assert_eq!(cache.restorable(), vec![4]);

        // A clone has its own ring: writes to one never reach the other.
        let clone = cache.try_clone().unwrap();
        assert_ne!(
            clone.ring_addresses().unwrap(),
            cache.ring_addresses().unwrap()
        );
        assert_eq!(live(&clone), live(&cache));
        feed(&mut cache, &fixture, 5, 2);
        assert_eq!(clone.offset(), 5);
        assert_eq!(live(&clone), fresh_state(&fixture, 5));
        let mut clone = clone;
        clone.rollback_to(4).unwrap();
        assert_eq!(live(&clone), fresh_state(&fixture, 4));
        assert_eq!(live(&cache), fresh_state(&fixture, 7));
    }

    #[test]
    fn ring_memory_is_the_whole_preallocation_and_ring_less_memory_is_the_live_state() {
        let spec = ring_spec(4);
        let slot = (RB * (RK - 1) * RC + RB * RHV * RDV * RDK) * 4;
        assert_eq!(spec.slot_bytes(), slot);
        assert_eq!(spec.bytes(), 4 * slot);
        let mut cache = DeltaNetCache::with_ring(spec).unwrap();
        assert_eq!(
            cache.memory_bytes(),
            (slot, 3 * slot),
            "priced from the spec"
        );
        cache.preallocate().unwrap();
        assert_eq!(cache.memory_bytes(), (slot, 3 * slot));
        let fixture = ring_inputs(3, 9);
        feed(&mut cache, &fixture, 0, 3);
        assert_eq!(
            cache.memory_bytes(),
            (slot, 3 * slot),
            "in place: nothing grew"
        );
        let mut plain = DeltaNetCache::new();
        assert_eq!(plain.memory_bytes(), (0, 0));
        feed(&mut plain, &fixture, 0, 3);
        assert_eq!(plain.memory_bytes(), (slot, 0));
    }

    #[test]
    fn ring_refuses_a_state_of_another_shape() {
        let mut spec = ring_spec(3);
        spec.ssm_dims = (RB, RHV, RDV + 1, RDK);
        let mut cache = DeltaNetCache::with_ring(spec).unwrap();
        let fixture = ring_inputs(2, 2);
        let (q, k, v, g, beta, x) = &fixture;
        let weight = Tensor::from_slice(CW, (RC, RK), &Device::Cpu).unwrap();
        let seed = Tensor::zeros((RB, RK - 1, RC), DType::F32, &Device::Cpu).unwrap();
        let (_, trace) = causal_depthwise_conv_traced(x, &weight, &seed).unwrap();
        assert!(cache.advance(&trace, q, k, v, g, beta).is_err());
        assert_eq!(cache.offset(), 0, "a refused forward commits nothing");
    }

    /// `advance` over tokens `start..start + len` of the fixture whose conv trace is cut short so
    /// that the tail after token `fail_at` is out of range: the recurrence fails on that token,
    /// after the ring slots of the tokens before it were written.
    fn advance_failing_at(
        cache: &mut DeltaNetCache,
        fixture: &Fixture,
        start: usize,
        len: usize,
        fail_at: usize,
    ) -> Result<Tensor> {
        let (q, k, v, g, beta, x) = fixture;
        let weight = Tensor::from_slice(CW, (RC, RK), &Device::Cpu).unwrap();
        let conv_state = cache.conv_state().unwrap().clone();
        let (_, trace) =
            causal_depthwise_conv_traced(&narrow_t(x, start, len), &weight, &conv_state).unwrap();
        let truncated = ConvTrace {
            cat: trace.cat.narrow(1, 0, fail_at + trace.tail).unwrap(),
            tokens: trace.tokens,
            tail: trace.tail,
        };
        cache.advance(
            &truncated,
            &narrow_t(q, start, len),
            &narrow_t(k, start, len),
            &narrow_t(v, start, len),
            &narrow_t(g, start, len),
            &narrow_t(beta, start, len),
        )
    }

    /// A forward that fails on its second token has already written its first token's state
    /// into the ring: the slot that held position `offset + 1 - slots` now holds position
    /// `offset + 1`. `restorable()` must not list any position whose slot the forward reuses, and
    /// the live state (whose slot a `T < slots` forward never reuses) must survive.
    #[test]
    fn a_forward_failing_part_way_never_lists_an_overwritten_slot() {
        let fixture = ring_inputs(8, 17);
        let mut cache = DeltaNetCache::with_ring(ring_spec(4)).unwrap(); // depth 3
        for i in 0..5 {
            feed(&mut cache, &fixture, i, 1);
        }
        // The ring now holds positions 2, 3, 4 behind the live 5 (slots 2, 3, 0; live slot 1).
        // Two tokens from offset 5: the first writes position 6 into position 2's slot, then the
        // second fails (position 7 would have taken position 3's slot).
        assert!(advance_failing_at(&mut cache, &fixture, 5, 2, 1).is_err());
        assert_eq!(cache.offset(), 5, "a failed forward does not advance");
        assert_eq!(
            cache.restorable(),
            vec![4],
            "positions 2 (overwritten) and 3 (next in line) are no longer listed"
        );
        assert_eq!(
            live(&cache),
            fresh_state(&fixture, 5),
            "the live state survived"
        );
    }

    /// A forward of `T >= slots` tokens rewrites every slot, the live one included: when it fails
    /// part-way the live state is gone, nothing is restorable, and the cache refuses to advance
    /// (rather than silently decoding from a zero or a foreign state) until it is reset.
    #[test]
    fn a_failed_forward_that_reused_the_live_slot_drops_the_live_state() {
        let fixture = ring_inputs(8, 19);
        let mut cache = DeltaNetCache::with_ring(ring_spec(3)).unwrap(); // depth 2
        for i in 0..4 {
            feed(&mut cache, &fixture, i, 1);
        }
        // The ring now holds positions 2, 3 behind the live 4 (slots 2, 0; live slot 1).
        // Four tokens from offset 4 on a 3-slot ring: the second and third write positions 6 and
        // 7 (position 7 takes the live position 4's slot), then the fourth fails.
        assert!(advance_failing_at(&mut cache, &fixture, 4, 4, 3).is_err());
        assert_eq!(cache.offset(), 4, "a failed forward does not advance");
        assert!(cache.restorable().is_empty());
        assert!(
            cache.conv_state().is_none() && cache.ssm_state().is_none(),
            "the live slot now holds position 7's state"
        );
        let (q, k, v, g, beta, x) = &fixture;
        let weight = Tensor::from_slice(CW, (RC, RK), &Device::Cpu).unwrap();
        let zeros = Tensor::zeros((RB, RK - 1, RC), DType::F32, &Device::Cpu).unwrap();
        let (_, trace) = causal_depthwise_conv_traced(&narrow_t(x, 4, 1), &weight, &zeros).unwrap();
        let refused = cache.advance(
            &trace,
            &narrow_t(q, 4, 1),
            &narrow_t(k, 4, 1),
            &narrow_t(v, 4, 1),
            &narrow_t(g, 4, 1),
            &narrow_t(beta, 4, 1),
        );
        assert!(refused.is_err(), "no forward from a lost live state");
        // A reset recovers the cache: it decodes again and its window refills.
        cache.rollback_to(0).unwrap();
        feed(&mut cache, &fixture, 0, 4);
        assert_eq!(
            cache.restorable(),
            vec![2, 3],
            "the window recovers after a reset"
        );
    }
}

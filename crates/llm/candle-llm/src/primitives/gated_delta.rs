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

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::primitives::decode_cache::tensor_bytes;
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

/// Run the gated delta recurrence over a `[B, T, ·]` chunk, a faithful port of the
/// `mlx_lm.models.gated_delta` ops path.
///
/// Shapes: `q`, `k` are `[B, T, Hk, Dk]`; `v` is `[B, T, Hv, Dv]`; `g` (the per-step gate from
/// [`compute_g`]) and `beta` are `[B, T, Hv]`; `state` (the carried recurrent state, or `None` to
/// start from zeros) is `[B, Hv, Dv, Dk]`. Returns the per-step output `y` `[B, T, Hv, Dv]` and the
/// final `state` `[B, Hv, Dv, Dk]` — feed `state` back in for the next chunk / decode step (T = 1).
///
/// GQA: when `Hv > Hk` each key/query head is repeated `Hv / Hk` times so it pairs with the value
/// heads (`Hv` must be a multiple of `Hk`).
pub fn gated_delta_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    gated_delta_recurrence_with_sink(q, k, v, g, beta, state, &mut |_, _| Ok(()))
}

/// [`gated_delta_recurrence`] that also hands every token's **post-step state** to `sink` —
/// `sink(ti, state_after_token_ti)` with `state` `[B, Hv, Dv, Dk]`, in token order, before the next
/// token runs. This is the per-token checkpoint output of the recurrence step (sc-24131):
/// [`DeltaNetCache::advance`] writes each state into its ring slot from here, and a fused decode
/// kernel (sc-24000) keeps the same contract by writing the slot itself. The returned final state
/// is the last one handed to `sink`. An error from `sink` aborts the recurrence.
pub fn gated_delta_recurrence_with_sink(
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
    /// position that still fits (the newest `slots - 1` before the current one). Allocation
    /// failures leave the cache untouched.
    pub fn set_ring_spec(&mut self, spec: Option<RingSpec>) -> Result<()> {
        let Some(spec) = spec else {
            if let Some(ring) = self.ring.take() {
                if let (Some(conv), Some(ssm)) = (&self.conv_state, &self.ssm_state) {
                    // Detach the live views from the buffer that is being dropped.
                    self.conv_state = Some(conv.contiguous()?.copy()?);
                    self.ssm_state = Some(ssm.contiguous()?.copy()?);
                }
                drop(ring);
            }
            self.spec = None;
            return Ok(());
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
            self.spec = next.spec;
            return Ok(());
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
        *self = next;
        Ok(())
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
    /// Shapes follow [`gated_delta_recurrence`]; with a ring, the produced states must match the
    /// ring's slot shape (else the write is refused as a shape error and the cache is left
    /// unadvanced — nothing is committed until the whole forward has run).
    pub fn advance(
        &mut self,
        conv: &ConvTrace,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        self.preallocate()?;
        let t = q.dim(1)?;
        if conv.tokens() != t {
            return Err(Error::Msg(format!(
                "DeltaNetCache::advance: conv trace covers {} tokens, recurrence {t}",
                conv.tokens()
            )));
        }
        let offset = self.offset;
        let (y, final_state) = match &self.ring {
            Some(ring) => {
                let first_write = t.saturating_sub(ring.slots);
                let mut sink = |ti: usize, state: &Tensor| -> Result<()> {
                    if ti < first_write {
                        return Ok(());
                    }
                    let position = offset + ti as i32 + 1;
                    ring.write(position, &conv.tail_after(ti)?, state)
                };
                gated_delta_recurrence_with_sink(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    self.ssm_state.as_ref(),
                    &mut sink,
                )?
            }
            None => gated_delta_recurrence(q, k, v, g, beta, self.ssm_state.as_ref())?,
        };
        let next = offset + t as i32;
        match self.ring.as_mut() {
            Some(ring) => {
                ring.lo = ring.lo.max(next + 1 - ring.slots as i32).max(1);
                let (conv, ssm) = ring.views(next)?;
                self.conv_state = Some(conv);
                self.ssm_state = Some(ssm);
            }
            None => {
                self.conv_state = Some(conv.tail_after(t - 1)?.contiguous()?);
                self.ssm_state = Some(final_state);
            }
        }
        self.offset = next;
        Ok(y)
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
        RingSpec {
            slots,
            conv_dims: (RB, RK - 1, RC),
            conv_dtype: DType::F32,
            ssm_dims: (RB, RHV, RDV, RDK),
            ssm_dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Seeded recurrence inputs for `t` tokens plus the conv input `x [B, t, C]`.
    fn ring_inputs(t: usize, salt: usize) -> Fixture {
        let sample = |len: usize, s: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 31 + s * 7) % 41) as f32 * scale - 20.0 * scale)
                .collect()
        };
        let cpu = &Device::Cpu;
        let q = Tensor::from_vec(
            sample(t * RHK * RDK, salt + 1, 0.05),
            (RB, t, RHK, RDK),
            cpu,
        )
        .unwrap();
        let k = Tensor::from_vec(
            sample(t * RHK * RDK, salt + 2, 0.04),
            (RB, t, RHK, RDK),
            cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            sample(t * RHV * RDV, salt + 3, 0.06),
            (RB, t, RHV, RDV),
            cpu,
        )
        .unwrap();
        let g = Tensor::from_vec(
            sample(t * RHV, salt + 4, 0.001)
                .into_iter()
                .map(|x| 0.95 + x)
                .collect(),
            (RB, t, RHV),
            cpu,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            sample(t * RHV, salt + 5, 0.002)
                .into_iter()
                .map(|x| 0.5 + x)
                .collect(),
            (RB, t, RHV),
            cpu,
        )
        .unwrap();
        let x = Tensor::from_vec(sample(t * RC, salt + 6, 0.1), (RB, t, RC), cpu).unwrap();
        (q, k, v, g, beta, x)
    }

    fn narrow_t(x: &Tensor, start: usize, len: usize) -> Tensor {
        x.narrow(1, start, len).unwrap().contiguous().unwrap()
    }

    /// Feed tokens `start..start+len` of the fixture through `cache` (one `advance`).
    fn feed(cache: &mut DeltaNetCache, fixture: &Fixture, start: usize, len: usize) -> Tensor {
        let (q, k, v, g, beta, x) = fixture;
        let weight = Tensor::from_slice(CW, (RC, RK), &Device::Cpu).unwrap();
        let conv_state = match cache.conv_state() {
            Some(c) => c.clone(),
            None => Tensor::zeros((RB, RK - 1, RC), DType::F32, &Device::Cpu).unwrap(),
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
            gated_delta_recurrence_with_sink(&q, &k, &v, &g, &beta, None, &mut |ti, s| {
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
        let err = gated_delta_recurrence_with_sink(&q, &k, &v, &g, &beta, None, &mut |_, _| {
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
}

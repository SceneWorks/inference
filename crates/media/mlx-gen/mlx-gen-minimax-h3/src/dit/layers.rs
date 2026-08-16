//! The two leaf modules every DiT block and every token-refiner block is built from.
//!
//! Both are **bias-free** — `MiniMaxH3Attention` constructs its four projections with
//! `bias=False` and the block's `FeedForward(..., bias=False)` does the same, which the published
//! tensor index corroborates: no `.bias` exists under `attn.` or `ff.` anywhere in
//! `transformer/`. Only `adaln_proj.linear`, the input/output projections and the timestep MLP
//! carry biases.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::multiply;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{sdpa_bounded_bhsd, AttentionChunkAxis, BoundedAttention};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::layout::split_gate_value;

/// **The rung-3 reachability probe** (sc-18661): what the last [`DitAttention::forward_bounded`] in
/// this process actually executed.
///
/// Always compiled, unlike `mlx_gen::attention`'s crate-private `#[cfg(test)]` counter, because the
/// claim it exists to settle is a *cross-crate* one: an integration test holding a real 50-block DiT
/// has to be able to show that the bounded kernel ran **inside the real attention call**, not merely
/// that a bounded kernel exists and that a contract declares a rung. Every equivalence and peak
/// assertion in `tests/bounded_attention_real.rs` is vacuous without it — "chunked == unbounded" is
/// trivially true when the chunking never engaged.
///
/// Two relaxed stores per attention call, against a fused Metal SDPA over up to 104 030 rows.
/// `JointDit::forwards` is the same kind of always-on mechanism counter in this crate.
pub mod attention_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `0` = nothing recorded, `1` = the un-chunked fused call, `n > 1` = `n` head chunks.
    static LAST_HEAD_CHUNKS: AtomicUsize = AtomicUsize::new(0);
    /// Query-row blocks the last call's *final* kernel invocation ran. `1` means the head axis alone
    /// fit the budget, which is the bit-exact case.
    static LAST_QUERY_CHUNKS: AtomicUsize = AtomicUsize::new(0);
    /// Packed sequence length the last call attended over.
    static LAST_SEQ: AtomicUsize = AtomicUsize::new(0);

    /// Head chunks the last bounded attention call ran.
    pub fn last_head_chunks() -> usize {
        LAST_HEAD_CHUNKS.load(Ordering::Relaxed)
    }

    /// Query-row blocks the last bounded attention call's final kernel invocation ran.
    pub fn last_query_chunks() -> usize {
        LAST_QUERY_CHUNKS.load(Ordering::Relaxed)
    }

    /// Sequence length the last bounded attention call attended over.
    pub fn last_seq() -> usize {
        LAST_SEQ.load(Ordering::Relaxed)
    }

    /// Reset before a measured arm so a stale value cannot satisfy an assertion.
    pub fn reset() {
        LAST_HEAD_CHUNKS.store(0, Ordering::Relaxed);
        LAST_QUERY_CHUNKS.store(0, Ordering::Relaxed);
        LAST_SEQ.store(0, Ordering::Relaxed);
    }

    pub(super) fn record(head_chunks: usize, query_chunks: usize, seq: usize) {
        LAST_HEAD_CHUNKS.store(head_chunks, Ordering::Relaxed);
        LAST_QUERY_CHUNKS.store(query_chunks, Ordering::Relaxed);
        LAST_SEQ.store(seq, Ordering::Relaxed);
    }
}

/// The chunk counts a [`BoundedAttention`] implies at this call's live shape.
///
/// Derived from [`mlx_gen::gen_core::attention_budget`]'s own planner — the same calls the kernel
/// makes — so the probe cannot report a plan the kernel did not execute. Self-attention here, so
/// `Sk == Sq == seq`.
///
/// `pub` because `tests/bounded_attention_real.rs` asserts the *engagement premise* (that the shipped
/// budget really does split at a given geometry) before it spends 40 minutes measuring peaks against
/// it, and a premise re-derived in the test would be a second implementation.
pub fn planned_attention_chunks(
    bounded: BoundedAttention<'_>,
    b: i32,
    heads: i32,
    seq: i32,
) -> (usize, usize) {
    let budget = bounded.plan.budget;
    let (head_chunks, heads_in_chunk) = match bounded.axis {
        AttentionChunkAxis::QueryRows => (1usize, heads),
        AttentionChunkAxis::Heads => {
            let plan = budget.head_chunks(
                b.max(0) as u64,
                heads.max(0) as u64,
                seq.max(0) as u64,
                seq.max(0) as u64,
            );
            if plan.chunks_heads() {
                let per = (plan.heads_per_chunk().max(1) as i32).min(heads.max(1));
                ((heads.max(1) as usize).div_ceil(per as usize), per)
            } else {
                (1usize, heads)
            }
        }
    };
    let block = budget.query_block(b, heads_in_chunk, seq, seq).max(1);
    (head_chunks, (seq.max(1) as usize).div_ceil(block as usize))
}

fn record_planned_attention(bounded: BoundedAttention<'_>, b: i32, heads: i32, seq: i32) {
    let (head_chunks, query_chunks) = planned_attention_chunks(bounded, b, heads, seq);
    attention_probe::record(head_chunks, query_chunks, seq.max(0) as usize);
}

/// `y = x · Wᵀ` for a stored `[out, in]` weight — an `nn.Linear(..., bias=False)`.
///
/// **Tier-aware** (sc-17150). This is one of exactly two loaders in the crate with a packed path:
/// it goes through [`crate::quant::lin`], so a `q4` / `q8` tier's packed triple builds a quantized
/// base directly and a `bf16` tier loads dense, with the same call and no manifest read. Every
/// attention projection (`to_q`/`to_k`/`to_v`/`to_out.0`) and both feed-forward projections in both
/// the 50-block stack and the 2-block token refiner are this type — the 312 tensors and
/// 40_076_574_720 bf16 bytes [`crate::convert`] packs at the tier width.
#[derive(Clone)]
pub struct LinearNoBias {
    inner: AdaptableLinear,
}

/// Reports the logical shape and whether the base is packed, rather than the opaque u32 code buffer
/// a derived `Debug` would print on a quantized tier.
impl std::fmt::Debug for LinearNoBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearNoBias")
            .field("shape", &self.inner.base_shape())
            .field("quantized", &self.inner.is_quantized())
            .finish()
    }
}

impl LinearNoBias {
    /// `dtype` casts a **dense** weight to the model's compute dtype. It is deliberately not applied
    /// to a packed base: a packed base's compute dtype is fixed by its scales, and
    /// [`AdaptableLinear::cast_weights`] no-ops on one for exactly that reason.
    pub fn from_weights(w: &mut Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        let mut inner = crate::quant::lin(w, prefix, false)?;
        inner.cast_weights(dtype)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.inner.forward(x)
    }

    /// The logical `[out, in]` shape, recovered from the scales grid on a packed base.
    pub fn shape(&self) -> Vec<i32> {
        self.inner.base_shape()
    }

    /// `true` on a `q4` / `q8` tier.
    pub fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }

    /// The **dense** tensor names, which is what the published-checkpoint key proofs are written
    /// against. A packed tier additionally carries `{prefix}.scales` and `{prefix}.biases`; those are
    /// discovered by [`crate::quant::lin`] rather than enumerated here, so this stays the bf16
    /// contract the `PUBLISHED_DIT_TENSORS` audit compares to.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }

    /// The adapter-bearing base, so a LoRA residual can be stacked on this projection
    /// ([`crate::adapters`], sc-18724).
    ///
    /// **Tier-independent by construction.** The inner [`AdaptableLinear`] is the *same* type whether
    /// [`crate::quant::lin`] built it dense or from a packed `q4`/`q8` triple, and the forward is
    /// `base(x) + Σ adapter.residual(x)` with the base never mutated — so the turbo LoRA folds at
    /// identical strength on every tier, which is what a creative knob has to guarantee.
    pub fn adaptable_mut(&mut self) -> &mut AdaptableLinear {
        &mut self.inner
    }
}

/// Affine RMSNorm over the last axis — `nn.RMSNorm(dim, eps)`, which is affine by default.
fn rms_affine(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    Ok(rms_norm(x, weight, eps)?)
}

/// Full self-attention with **affine per-head q/k RMSNorm** and MM-RoPE.
///
/// # qk-norm placement and shape
///
/// `norm_q` / `norm_k` are `nn.RMSNorm(dim_head, eps=qk_norm_eps)` — normalizing over a single
/// **head's** 128 channels, not over the 7168-wide projection — and they run **after** the
/// `[B, S, H·D] → [B, S, H, D]` unflatten and **before** the rotary. All three of those choices are
/// invisible in the tensor index, which only shows a `[128]` vector:
///
/// * normalizing the flat projection instead would mix all 56 heads into one RMS;
/// * running the rotary first would rotate un-normalized vectors, changing every attention score.
///
/// The video VAE's equivalent is non-affine and leaves **no tensors at all**, which is why
/// `tests/dit_parity.rs` gates this on *behaviour* rather than on the presence of `norm_q.weight`.
#[derive(Debug, Clone)]
pub struct DitAttention {
    to_q: LinearNoBias,
    to_k: LinearNoBias,
    to_v: LinearNoBias,
    to_out: LinearNoBias,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    qk_norm_eps: f32,
}

impl DitAttention {
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let expect = |t: &Array, want: &[i32], what: &str| -> Result<()> {
            if t.shape() != want {
                return Err(Error::Msg(format!(
                    "minimax-h3 dit {prefix}.{what}: expected {want:?}, got {:?}",
                    t.shape()
                )));
            }
            Ok(())
        };
        let norm_q = w
            .require(&format!("{prefix}.norm_q.weight"))?
            .as_dtype(dtype)?;
        let norm_k = w
            .require(&format!("{prefix}.norm_k.weight"))?
            .as_dtype(dtype)?;
        // A qk-norm sized to the projection rather than to a head is the plausible wrong shape,
        // and it broadcasts silently against `[B, S, H, D]` only when it is `[D]`.
        expect(&norm_q, &[cfg.attention_head_dim], "norm_q")?;
        expect(&norm_k, &[cfg.attention_head_dim], "norm_k")?;

        Ok(Self {
            to_q: LinearNoBias::from_weights(w, &format!("{prefix}.to_q"), dtype)?,
            to_k: LinearNoBias::from_weights(w, &format!("{prefix}.to_k"), dtype)?,
            to_v: LinearNoBias::from_weights(w, &format!("{prefix}.to_v"), dtype)?,
            to_out: LinearNoBias::from_weights(w, &format!("{prefix}.to_out.0"), dtype)?,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            qk_norm_eps: cfg.qk_norm_eps,
        })
    }

    pub fn names(prefix: &str) -> Vec<String> {
        let mut v: Vec<String> = ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .flat_map(|p| LinearNoBias::names(&format!("{prefix}.{p}")))
            .collect();
        v.push(format!("{prefix}.norm_q.weight"));
        v.push(format!("{prefix}.norm_k.weight"));
        v
    }

    /// `rope` is `None` for the token refiner, which runs **without** any positional embedding.
    ///
    /// The un-bounded entry point, byte-identical to the pre-rung-3 forward: it is
    /// [`Self::forward_bounded`] at [`BoundedAttention::UNBOUNDED`], whose fast path is a single
    /// un-chunked `scaled_dot_product_attention` with the same scale, dtypes and k/v.
    pub fn forward(&self, x: &Array, rope: Option<(&MmRope, &MmRopeTables)>) -> Result<Array> {
        self.forward_bounded(x, rope, BoundedAttention::UNBOUNDED)
    }

    /// [`Self::forward`] under an explicit [`BoundedAttention`] — the rung-3 seam (sc-18661).
    ///
    /// This is the **only** attention kernel call in the 50-block DiT stack and the 2-block token
    /// refiner, so a bounded plan reaching here reaches every one of the family's 52 attention sites.
    /// The preferred axis for this family is [`AttentionChunkAxis::Heads`]: 56 heads make the head
    /// split alone worth 56x on the score domain while leaving the query GEMM's `M` untouched, and it
    /// reconstructs the unbounded output **exactly** — measured at both ends of the frame lattice on
    /// the real DiT (`tests/bounded_attention_real.rs`) and on the committed fixture
    /// (`tests/bounded_attention.rs`). That holds while the budget admits one whole head; below it the
    /// kernel falls back to query rows and inherits their weaker contract. See
    /// [`AttentionChunkAxis`].
    ///
    /// The default is [`BoundedAttention::UNBOUNDED`], and rung 3 is declared
    /// `StructurallyNotApplicable` for this provider on the MLX backend
    /// (`crate::memory_strategy::RUNG3_MEASURED_PEAK_DELTAS`): the seam exists as the instrument that
    /// verdict is re-measurable with, not as a lever a request can select.
    pub fn forward_bounded(
        &self,
        x: &Array,
        rope: Option<(&MmRope, &MmRopeTables)>,
        bounded: BoundedAttention<'_>,
    ) -> Result<Array> {
        let s = x.shape();
        if s.len() != 3 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit attention: expected [B, S, hidden], got {s:?}"
            )));
        }
        let (b, seq) = (s[0], s[1]);
        let (h, d) = (self.heads, self.head_dim);
        let shape = [b, seq, h, d];

        // The published checkpoint ships `to_q`/`to_k`/`to_v` already split, so this applies NO
        // fused-QKV transform. See `crate::dit::qkv` for which transform produced them and why
        // that must be asserted rather than assumed.
        let q = self.to_q.forward(x)?.reshape(&shape)?;
        let k = self.to_k.forward(x)?.reshape(&shape)?;
        let v = self.to_v.forward(x)?.reshape(&shape)?;

        // qk-norm first, rotary second.
        let q = rms_affine(&q, &self.norm_q, self.qk_norm_eps)?;
        let k = rms_affine(&k, &self.norm_k, self.qk_norm_eps)?;
        let (q, k) = match rope {
            Some((rope, tables)) => (rope.apply(&q, tables)?, rope.apply(&k, tables)?),
            None => (q, k),
        };

        // MiniMax-H3 packs one request into a single attention document: no mask, not causal.
        let qh = q.transpose_axes(&[0, 2, 1, 3])?;
        let kh = k.transpose_axes(&[0, 2, 1, 3])?;
        let vh = v.transpose_axes(&[0, 2, 1, 3])?;
        let scale = 1.0 / (d as f32).sqrt();
        // Record what this call is about to run BEFORE the kernel, from the shared planner rather
        // than from local arithmetic — a probe that re-derived the boundaries could agree with itself
        // while the kernel did something else.
        record_planned_attention(bounded, b, h, seq);
        let out = sdpa_bounded_bhsd(&qh, &kh, &vh, scale, None, bounded)?;

        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, seq, h * d])?;
        self.to_out.forward(&out)
    }
}

/// Bias-free SwiGLU feed-forward: `w2( silu(gate) · value )`.
///
/// **The published `ff.net.0.proj` emits `[value | gate]`**, so the gate is the SECOND half — the
/// split is delegated to [`crate::layout::split_gate_value`], the crate's single implementation.
/// `convert_minimax_h3_to_diffusers.py` swaps the halves of the DiT's `mlp.fc1` on the way in
/// exactly as it does the video VAE's `ff.w1`, and reading the first half as the gate computes
/// `w2( silu(value) · gate )` — shape-identical, and the sc-18740 defect.
#[derive(Debug, Clone)]
pub struct DitFeedForward {
    proj: LinearNoBias,
    out: LinearNoBias,
}

impl DitFeedForward {
    pub fn from_weights(w: &mut Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            proj: LinearNoBias::from_weights(w, &format!("{prefix}.net.0.proj"), dtype)?,
            out: LinearNoBias::from_weights(w, &format!("{prefix}.net.2"), dtype)?,
        })
    }

    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = LinearNoBias::names(&format!("{prefix}.net.0.proj")).to_vec();
        v.extend(LinearNoBias::names(&format!("{prefix}.net.2")));
        v
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.proj.forward(x)?;
        let axis = (h.shape().len() - 1) as i32;
        let (gate, value) = split_gate_value(&h, axis)?;
        self.out.forward(&multiply(&silu(&gate)?, &value)?)
    }
}

/// Affine RMSNorm as a loadable block-level norm (`norm1` / `norm2` / `final_norm` / `norm_out.norm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    pub fn from_weights(w: &mut Weights, prefix: &str, eps: f32, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        rms_affine(x, &self.weight, self.eps)
    }

    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
}

/// The four attention projections, addressed exactly as the published checkpoint spells them —
/// including `to_out.0`, whose trailing `0` is a `nn.Sequential` index and therefore arrives as its
/// **own** path segment (sc-18724).
///
/// `norm_q` / `norm_k` are deliberately unreachable: they are bare `[head_dim]` RMSNorm gains, not
/// Linears, and no published MiniMax-H3 adapter targets them. A key that names one surfaces as
/// unmatched (loud) rather than being dropped — see [`crate::adapters`].
impl AdaptableHost for DitAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(self.to_q.adaptable_mut()),
            ["to_k"] => Some(self.to_k.adaptable_mut()),
            ["to_v"] => Some(self.to_v.adaptable_mut()),
            ["to_out", "0"] => Some(self.to_out.adaptable_mut()),
            _ => None,
        }
    }
}

/// The two feed-forward projections. `net.0.proj` is the SwiGLU input (`[value | gate]` in the
/// published layout — see [`crate::layout`]) and `net.2` the output; the `net.N` segments are
/// `nn.Sequential` indices, so they arrive split exactly as written.
impl AdaptableHost for DitFeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["net", "0", "proj"] => Some(self.proj.adaptable_mut()),
            ["net", "2"] => Some(self.out.adaptable_mut()),
            _ => None,
        }
    }
}

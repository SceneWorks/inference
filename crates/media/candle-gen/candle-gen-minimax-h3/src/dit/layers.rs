//! The two leaf modules every DiT block and every token-refiner block is built from.
//!
//! Both are **bias-free** — `MiniMaxH3Attention` constructs its four projections with
//! `bias=False` and the block's `FeedForward(..., bias=False)` does the same, which the published
//! tensor index corroborates: no `.bias` exists under `attn.` or `ff.` anywhere in
//! `transformer/`. Only `adaln_proj.linear`, the input/output projections and the timestep MLP
//! carry biases.
//!
//! # The attention transient is a genuine cross-backend difference (sc-17152)
//!
//! MLX's fused `scaled_dot_product_attention` **streams the scores**: sc-17152 measured its peak
//! tracking `4·B·H·S·D` exactly, i.e. there is no `[B, H, Sq, Sk]` tensor in memory at all. candle
//! has no fused kernel here, so [`DitAttention::forward`] routes through
//! `candle_gen::sdpa_budgeted_bhsd`, which **does** materialize scores — in bounded row blocks
//! rather than all at once (`ATTN_SCORES_BUDGET`, the shared i32-overflow-safe helper from
//! sc-9116).
//!
//! The consequence is stated rather than glossed: at the shipped geometry a 15 s render packs
//! ~94k rows over 56 heads, where the full score tensor would be ~2.0e12 elements — past `i32::MAX`
//! and far past any card — so the budgeted split is **not optional on this lane** the way MLX's
//! streaming kernel makes it optional there. sc-17152 additionally measured chunked SDPA on MLX at
//! **+50.3 % peak and ~3× wall**, so its conclusion ("do not chunk") is an MLX result and must not
//! be carried across: the two backends have different attention memory shapes, and this lane's
//! numbers are sc-17156's to measure on real hardware.
//!
//! sc-17152 also probed `matmul` correctness above `i32::MAX` **on MLX** and found it exact. That
//! is an MLX result. Nothing here assumes it transfers to candle/CUDA, and the widest single tensor
//! this module writes is the SwiGLU projection `[1, S, 2·ffn_dim]` — `[1, S, 28672]` at the shipped
//! geometry — not anything in the attention.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result, Weights};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::layout::split_gate_value;
use crate::nn::{linear_nb, rms_weighted, sdpa, silu};

/// `y = x · Wᵀ` for a stored `[out, in]` weight — an `nn.Linear(..., bias=False)`.
#[derive(Debug, Clone)]
pub struct LinearNoBias {
    weight: Tensor,
}

impl LinearNoBias {
    /// Load `{prefix}.weight` at `dtype`.
    pub fn from_weights(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
        })
    }

    /// `y = x · Wᵀ` over the last axis of `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        linear_nb(x, &self.weight)
    }

    /// Device bytes this projection holds.
    pub fn nbytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
    }

    /// The one tensor name this projection consumes.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
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
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    qk_norm_eps: f64,
}

impl DitAttention {
    /// Load `{prefix}` (e.g. `transformer_blocks.0.attn`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: DType,
    ) -> Result<Self> {
        let expect = |t: &Tensor, want: &[usize], what: &str| -> Result<()> {
            if t.dims() != want {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit {prefix}.{what}: expected {want:?}, got {:?}",
                    t.dims()
                )));
            }
            Ok(())
        };
        let norm_q = w
            .require(&format!("{prefix}.norm_q.weight"))?
            .to_dtype(dtype)?;
        let norm_k = w
            .require(&format!("{prefix}.norm_k.weight"))?
            .to_dtype(dtype)?;
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

    /// Every tensor name this attention consumes — six.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v: Vec<String> = ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .flat_map(|p| LinearNoBias::names(&format!("{prefix}.{p}")))
            .collect();
        v.push(format!("{prefix}.norm_q.weight"));
        v.push(format!("{prefix}.norm_k.weight"));
        v
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.to_q.nbytes()
            + self.to_k.nbytes()
            + self.to_v.nbytes()
            + self.to_out.nbytes()
            + self.norm_q.elem_count() * self.norm_q.dtype().size_in_bytes()
            + self.norm_k.elem_count() * self.norm_k.dtype().size_in_bytes()
    }

    /// `rope` is `None` for the token refiner, which runs **without** any positional embedding.
    pub fn forward(&self, x: &Tensor, rope: Option<(&MmRope, &MmRopeTables)>) -> Result<Tensor> {
        let s = x.dims();
        if s.len() != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit attention: expected [B, S, hidden], got {s:?}"
            )));
        }
        let (b, seq) = (s[0], s[1]);
        let (h, d) = (self.heads, self.head_dim);

        // The published checkpoint ships `to_q`/`to_k`/`to_v` already split, so this applies NO
        // fused-QKV transform. See `crate::dit::qkv` for which transform produced them and why
        // that must be asserted rather than assumed.
        let q = self.to_q.forward(x)?.reshape((b, seq, h, d))?;
        let k = self.to_k.forward(x)?.reshape((b, seq, h, d))?;
        let v = self.to_v.forward(x)?.reshape((b, seq, h, d))?;

        // qk-norm first, rotary second.
        let q = rms_weighted(&q, &self.norm_q, self.qk_norm_eps)?;
        let k = rms_weighted(&k, &self.norm_k, self.qk_norm_eps)?;
        let (q, k) = match rope {
            Some((rope, tables)) => (rope.apply(&q, tables)?, rope.apply(&k, tables)?),
            None => (q, k),
        };

        // MiniMax-H3 packs one request into a single attention document: no mask, not causal.
        let qh = q.transpose(1, 2)?.contiguous()?;
        let kh = k.transpose(1, 2)?.contiguous()?;
        let vh = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (d as f64).sqrt();
        let out = sdpa(&qh, &kh, &vh, scale)?;

        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, h * d))?;
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
    /// Load `{prefix}` (e.g. `transformer_blocks.0.ff`).
    pub fn from_weights(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: LinearNoBias::from_weights(w, &format!("{prefix}.net.0.proj"), dtype)?,
            out: LinearNoBias::from_weights(w, &format!("{prefix}.net.2"), dtype)?,
        })
    }

    /// The two tensor names the feed-forward consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = LinearNoBias::names(&format!("{prefix}.net.0.proj")).to_vec();
        v.extend(LinearNoBias::names(&format!("{prefix}.net.2")));
        v
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.proj.nbytes() + self.out.nbytes()
    }

    /// `w2( silu(gate) · value )`.
    ///
    /// The `[1, S, 2·ffn_dim]` intermediate this writes is the **widest single tensor in the whole
    /// model** — `[1, 94000, 28672]` at a 15 s render — which is the element-count ceiling worth
    /// watching on this lane rather than anything in the attention (sc-17152).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(x)?;
        let (gate, value) = split_gate_value(&h)?;
        self.out.forward(&(silu(&gate)?.mul(&value)?))
    }
}

/// Affine RMSNorm as a loadable block-level norm (`norm1` / `norm2` / `final_norm` /
/// `norm_out.norm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Load `{prefix}.weight` at `dtype`.
    pub fn from_weights(w: &Weights, prefix: &str, eps: f64, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
            eps,
        })
    }

    /// `weight · x / sqrt(mean(x²) + eps)` over the last axis.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_weighted(x, &self.weight, self.eps)
    }

    /// The one tensor name this norm consumes.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
}

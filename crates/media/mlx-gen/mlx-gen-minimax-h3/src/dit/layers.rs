//! The two leaf modules every DiT block and every token-refiner block is built from.
//!
//! Both are **bias-free** — `MiniMaxH3Attention` constructs its four projections with
//! `bias=False` and the block's `FeedForward(..., bias=False)` does the same, which the published
//! tensor index corroborates: no `.bias` exists under `attn.` or `ff.` anywhere in
//! `transformer/`. Only `adaln_proj.linear`, the input/output projections and the timestep MLP
//! carry biases.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::multiply;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::layout::split_gate_value;

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
    pub fn forward(&self, x: &Array, rope: Option<(&MmRope, &MmRopeTables)>) -> Result<Array> {
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
        let out = scaled_dot_product_attention(&qh, &kh, &vh, scale, None, None)?;

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

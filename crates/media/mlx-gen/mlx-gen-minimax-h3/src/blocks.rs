//! The ViT decoder's transformer block (`base_module.py::TransformerBlock` + `attention.py`).
//!
//! Four choices here are **not** the reference module's defaults; they come from
//! `FL2VA/video_vae/source/config.json`'s `vit_decoder_kwargs` and are invisible in the diffusers
//! `vae/config.json`:
//!
//! | knob | default | this checkpoint |
//! |---|---|---|
//! | `norm_type` | `layer_norm` | **`rms_norm`** (affine, weight-only) |
//! | `qk_norm_type` | `None` | **`rms_norm`** (`qk_norm_affine = false`) |
//! | `ffn_activation_fn` | `gelu` | **`silu`** |
//! | `ffn_use_gated` | `false` | **`true`** (SwiGLU) |
//!
//! The published tensor set corroborates all four: `norm1`/`norm2` ship a `weight` and **no
//! `bias`** (a LayerNorm would have both), there are no `norm_q`/`norm_k` tensors (consistent with
//! a non-affine qk norm), and `ff.net.0.proj` is `[2·inner, dim]` rather than `[inner, dim]`
//! (the gate and the value halves).
//!
//! The residual is **scaled**: `x + attn(·) · scale1`, with `scale1`/`scale2` learned per-channel
//! vectors that the reference initializes to zero. A parity fixture dumped without re-randomizing
//! them would make every block an exact identity and pass against a port that implements neither
//! attention nor the feed-forward — see the generator's `randomize`.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::MiniMaxH3VaeConfig;
use crate::layout::split_gate_value;
use crate::rope::{Rope3d, RopeTables};
use crate::tensor::slice_axis;

/// A `weight`/`bias` pair for an `nn.Linear`.
#[derive(Debug, Clone)]
struct Linear {
    weight: Array,
    bias: Array,
}

impl Linear {
    fn from_weights(w: &mut Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?,
            bias: w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        linear(x, &self.weight, &self.bias)
    }

    /// Tensor names an instance consumes, for the exhaustive-mapping check.
    fn names(prefix: &str) -> [String; 2] {
        [format!("{prefix}.weight"), format!("{prefix}.bias")]
    }
}

/// Weightless RMSNorm over the last axis — the `qk_norm_affine = false` q/k normalization.
fn rms_norm_noweight(x: &Array, eps: f32) -> Result<Array> {
    let dim = *x
        .shape()
        .last()
        .ok_or_else(|| Error::Msg("minimax-h3 rms_norm: zero-rank input".into()))?;
    let ones = Array::ones::<f32>(&[dim])?.as_dtype(x.dtype())?;
    Ok(rms_norm(x, &ones, eps)?)
}

/// Multi-head self-attention with non-affine q/k RMSNorm and 3-D partial RoPE.
#[derive(Debug, Clone)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: i32,
    head_dim: i32,
    eps: f32,
}

impl Attention {
    fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            to_q: Linear::from_weights(w, &format!("{prefix}.to_q"), dtype)?,
            to_k: Linear::from_weights(w, &format!("{prefix}.to_k"), dtype)?,
            to_v: Linear::from_weights(w, &format!("{prefix}.to_v"), dtype)?,
            to_out: Linear::from_weights(w, &format!("{prefix}.to_out.0"), dtype)?,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            eps: cfg.norm_eps,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .flat_map(|p| Linear::names(&format!("{prefix}.{p}")))
            .collect()
    }

    fn forward(&self, x: &Array, rope: &Rope3d, tables: &RopeTables) -> Result<Array> {
        let s = x.shape();
        let (b, seq) = (s[0], s[1]);
        let (h, d) = (self.heads, self.head_dim);

        // The published checkpoint ships to_q/to_k/to_v already split out of the reference's fused
        // `to_qkv`. That split is head-interleaving aware (see `crate::vae::split_fused_qkv`), so a
        // plain `[B, S, H·D] -> [B, S, H, D]` reshape recovers exactly the reference's heads.
        let shape = [b, seq, h, d];
        let q = self.to_q.forward(x)?.reshape(&shape)?;
        let k = self.to_k.forward(x)?.reshape(&shape)?;
        let v = self.to_v.forward(x)?.reshape(&shape)?;

        // q/k RMSNorm comes BEFORE the rotary, per `attention.py::Attention.forward`.
        let q = rope.apply(&rms_norm_noweight(&q, self.eps)?, tables)?;
        let k = rope.apply(&rms_norm_noweight(&k, self.eps)?, tables)?;

        // SDPA wants [B, H, S, D]. `t_causal` is false for this checkpoint (`causal_decoder:
        // false`), so attention is dense and unmasked.
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

/// SwiGLU feed-forward: `w2( silu(gate) · value )`.
///
/// **The published `ff.net.0.proj` emits `[value | gate]`, so the gate is the SECOND half.** The
/// original MiniMax module stores `[gate | value]`; `convert_minimax_h3_to_diffusers.py` swaps the
/// two halves on the way into the released checkpoint. The split is delegated to
/// [`crate::layout::split_gate_value`] so this crate has exactly one place that knows which half is
/// which — see that module for the full contract and for why nothing structural can catch getting
/// it wrong (sc-18740).
#[derive(Debug, Clone)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
}

impl FeedForward {
    fn from_weights(w: &mut Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            w1: Linear::from_weights(w, &format!("{prefix}.net.0.proj"), dtype)?,
            w2: Linear::from_weights(w, &format!("{prefix}.net.2"), dtype)?,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        let mut v = Linear::names(&format!("{prefix}.net.0.proj")).to_vec();
        v.extend(Linear::names(&format!("{prefix}.net.2")));
        v
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.w1.forward(x)?;
        let axis = (h.shape().len() - 1) as i32;
        let (gate, value) = split_gate_value(&h, axis)?;
        self.w2.forward(&multiply(&silu(&gate)?, &value)?)
    }
}

/// One decoder block: scaled-residual attention then scaled-residual SwiGLU FFN.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    norm1: Array,
    attn: Attention,
    scale1: Array,
    norm2: Array,
    ff: FeedForward,
    scale2: Array,
    eps: f32,
}

impl TransformerBlock {
    /// Load block `prefix` (e.g. `decoder.transformer_blocks.0`).
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            norm1: w
                .require(&format!("{prefix}.norm1.weight"))?
                .as_dtype(dtype)?,
            attn: Attention::from_weights(w, &format!("{prefix}.attn"), cfg, dtype)?,
            scale1: w.require(&format!("{prefix}.scale1"))?.as_dtype(dtype)?,
            norm2: w
                .require(&format!("{prefix}.norm2.weight"))?
                .as_dtype(dtype)?,
            ff: FeedForward::from_weights(w, &format!("{prefix}.ff"), dtype)?,
            scale2: w.require(&format!("{prefix}.scale2"))?.as_dtype(dtype)?,
            eps: cfg.norm_eps,
        })
    }

    /// Every tensor name a block consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = vec![
            format!("{prefix}.norm1.weight"),
            format!("{prefix}.scale1"),
            format!("{prefix}.norm2.weight"),
            format!("{prefix}.scale2"),
        ];
        v.extend(Attention::names(&format!("{prefix}.attn")));
        v.extend(FeedForward::names(&format!("{prefix}.ff")));
        v
    }

    /// `x + attn(rms(x))·scale1`, then `x + ff(rms(x))·scale2`.
    pub fn forward(&self, x: &Array, rope: &Rope3d, tables: &RopeTables) -> Result<Array> {
        let normed = rms_norm(x, &self.norm1, self.eps)?;
        let attn = self.attn.forward(&normed, rope, tables)?;
        let x = add(x, &multiply(&attn, &self.scale1)?)?;

        let normed = rms_norm(&x, &self.norm2, self.eps)?;
        let ff = self.ff.forward(&normed)?;
        Ok(add(&x, &multiply(&ff, &self.scale2)?)?)
    }
}

/// Cross-fade `b` onto the tail of `a` over `blend_extent` positions along `axis`
/// (`klvae.py::blend`).
///
/// `a` contributes ONLY its last `blend_extent` positions — anything before them is dropped, which
/// is correct because the caller passes the held-back overlap tail, not a whole chunk.
pub fn blend(a: &Array, b: &Array, blend_extent: i32, axis: i32) -> Result<Array> {
    let a_len = a.shape()[axis as usize];
    let b_len = b.shape()[axis as usize];
    let extent = blend_extent.min(a_len).min(b_len);
    if extent <= 0 {
        return Ok(b.clone());
    }
    let weights: Vec<f32> = (0..extent).map(|i| i as f32 / extent as f32).collect();
    let mut shape = vec![1; a.shape().len()];
    shape[axis as usize] = extent;
    let w_b = Array::from_slice(&weights, &shape).as_dtype(a.dtype())?;
    let w_a = mlx_rs::ops::subtract(&Array::from_f32(1.0).as_dtype(a.dtype())?, &w_b)?;

    let a_ov = slice_axis(a, axis, a_len - extent, a_len)?;
    let b_ov = slice_axis(b, axis, 0, extent)?;
    let blended = add(&multiply(&a_ov, &w_a)?, &multiply(&b_ov, &w_b)?)?;

    if extent < b_len {
        let rest = slice_axis(b, axis, extent, b_len)?;
        Ok(concatenate_axis(&[blended, rest], axis)?)
    } else {
        Ok(blended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_ramps_linearly_from_a_to_b() {
        let a = Array::from_slice(&[9.0f32, 1.0, 1.0, 1.0], &[1, 4, 1]);
        let b = Array::from_slice(&[0.0f32, 0.0, 0.0, 5.0], &[1, 4, 1]);
        let out = blend(&a, &b, 4, 1).unwrap();
        // extent == b_len, so the result is exactly the blended window.
        assert_eq!(out.shape(), &[1, 4, 1]);
        let v: Vec<f32> = out.as_slice::<f32>().to_vec();
        // weights for b are 0, .25, .5, .75 -> a dominates the seam start.
        assert!((v[0] - 9.0).abs() < 1e-6);
        assert!((v[3] - (1.0 * 0.25 + 5.0 * 0.75)).abs() < 1e-5);
    }

    /// When `b` is longer than the seam, the un-blended remainder must survive verbatim.
    #[test]
    fn blend_appends_the_rest_of_b() {
        let a = Array::from_slice(&[1.0f32, 1.0], &[1, 2, 1]);
        let b = Array::from_slice(&[0.0f32, 0.0, 7.0, 8.0], &[1, 4, 1]);
        let out = blend(&a, &b, 2, 1).unwrap();
        assert_eq!(out.shape(), &[1, 4, 1]);
        let v: Vec<f32> = out.as_slice::<f32>().to_vec();
        assert!((v[2] - 7.0).abs() < 1e-6);
        assert!((v[3] - 8.0).abs() < 1e-6);
    }

    /// A zero-length seam degenerates to `b` rather than dividing by zero.
    #[test]
    fn blend_with_no_extent_returns_b() {
        let a = Array::from_slice(&[1.0f32], &[1, 1, 1]);
        let b = Array::from_slice(&[2.0f32, 3.0], &[1, 2, 1]);
        let out = blend(&a, &b, 0, 1).unwrap();
        assert_eq!(out.as_slice::<f32>(), &[2.0, 3.0]);
    }

    #[test]
    fn weightless_rms_norm_scales_to_unit_rms() {
        let x = Array::from_slice(&[3.0f32, 4.0, 0.0, 0.0], &[1, 4]);
        let out = rms_norm_noweight(&x, 1e-5).unwrap();
        let ms: f32 = out.square().unwrap().mean(None).unwrap().item();
        assert!(
            (ms - 1.0).abs() < 1e-3,
            "mean square should be ~1, got {ms}"
        );
    }

    #[test]
    fn block_names_cover_the_published_sixteen_tensors() {
        let names = TransformerBlock::names("decoder.transformer_blocks.0");
        // The published checkpoint ships exactly 16 tensors per block.
        assert_eq!(names.len(), 16);
        assert!(names.contains(&"decoder.transformer_blocks.0.attn.to_out.0.weight".to_string()));
        assert!(names.contains(&"decoder.transformer_blocks.0.ff.net.0.proj.weight".to_string()));
        assert!(names.contains(&"decoder.transformer_blocks.0.scale2".to_string()));
        // No `norm1.bias` — these are RMSNorms, not LayerNorms.
        assert!(!names.iter().any(|n| n.ends_with("norm1.bias")));
    }
}

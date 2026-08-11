//! One `MiniMaxH3TransformerBlock` — pre-norm self-attention and feed-forward, each modulated by
//! AdaLN parameters selected **per row** of the packed sequence.
//!
//! # Single-stream
//!
//! There is no cross-attention and no per-modality block weight anywhere in MiniMax-H3. Video,
//! audio and text rows share one stack and one attention document. Modality enters only through
//! the AdaLN row index, so a Wan/LTX-style dual-stream block is the wrong shape entirely.
//!
//! # The AdaLN row layout is the trap
//!
//! `adaln_proj` emits `6 · 3 · hidden_size` columns from `time_embed_dim`, and the reference then
//! does
//!
//! ```text
//! temb = linear(silu(temb))          # [T, 18·hidden]
//! temb = temb.view(-1, 6·hidden)     # [T·3, 6·hidden]      <- modality becomes a ROW axis
//! shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb.chunk(6, dim=-1)
//! ```
//!
//! so each of the six is `[T·3, hidden]` with rows ordered `[t0_mod0, t0_mod1, t0_mod2, t1_mod0,
//! …]` — which is what `timestep_index · 3 + token_tag` addresses. Chunking the `[T, 18·hidden]`
//! tensor into six `[T, 3·hidden]` pieces instead produces six tensors of the right total size and
//! the wrong contents; [`AdaLnProjection::forward`] does the reshape first, and
//! `tests/dit_parity.rs` mutates it to prove the difference is observable.
//!
//! # What this story does and does not own
//!
//! The projection lives here because the reference block owns it and one-block parity cannot be
//! shown without it. **Precomputing the modulation across a step schedule and evicting the ~13 B of
//! `adaln_proj` weights before the denoise loop is sc-17145**, which is why [`AdaLnModulation`] is
//! a standalone value produced by [`DitBlock::modulation`] rather than an internal of
//! [`DitBlock::forward`]: caching it needs no change here.

use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::dit::layers::{DitAttention, DitFeedForward, RmsNorm};
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::tensor::slice_axis;

/// The six modulation vectors of one block, each `[num_timesteps · MODALITY_NUM, hidden_size]`.
///
/// A function of the timestep embedding only — never of the tokens — which is what makes
/// sc-17145's precompute-and-evict possible.
#[derive(Debug, Clone)]
pub struct AdaLnModulation {
    /// Pre-attention shift.
    pub shift_msa: Array,
    /// Pre-attention scale, applied as `1 + scale`.
    pub scale_msa: Array,
    /// Attention residual gate.
    pub gate_msa: Array,
    /// Pre-feed-forward shift.
    pub shift_mlp: Array,
    /// Pre-feed-forward scale, applied as `1 + scale`.
    pub scale_mlp: Array,
    /// Feed-forward residual gate.
    pub gate_mlp: Array,
}

/// `adaln_proj`: `time_embed_dim → 6 · MODALITY_NUM · hidden_size`.
#[derive(Debug, Clone)]
pub struct AdaLnProjection {
    weight: Array,
    bias: Array,
    hidden_size: i32,
}

impl AdaLnProjection {
    fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?;
        let bias = w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?;
        let want = [cfg.adaln_out_features(), cfg.time_embed_dim];
        if weight.shape() != want {
            return Err(Error::Msg(format!(
                "minimax-h3 dit {prefix}.weight: expected {want:?}, got {:?}",
                weight.shape()
            )));
        }
        Ok(Self {
            weight,
            bias,
            hidden_size: cfg.hidden_size,
        })
    }

    fn names(prefix: &str) -> [String; 2] {
        [format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// Project one `[num_timesteps, time_embed_dim]` embedding into the six modulation tables.
    ///
    /// The activation runs at `temb`'s own precision and only its result is cast to the
    /// projection's dtype — the reference is explicit about this because `time_embedder` is float32
    /// in the mixed-precision checkpoint while `adaln_proj` is bfloat16, and rounding before the
    /// activation would bias every block identically at every step.
    pub fn forward(&self, temb: &Array) -> Result<AdaLnModulation> {
        let s = temb.shape();
        if s.len() != 2 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit adaln: expected temb as [num_timesteps, time_embed_dim], got {s:?}"
            )));
        }
        let steps = s[0];
        let activated = silu(temb)?.as_dtype(self.weight.dtype())?;
        let projected = mlx_gen::nn::linear(&activated, &self.weight, &self.bias)?;

        // `view(-1, 6·hidden)` BEFORE the chunk: modality becomes a row axis, not a column one.
        let rows = steps * MODALITY_NUM;
        let table = projected.reshape(&[rows, MODULATION_PARAMS * self.hidden_size])?;
        let part = |i: i32| slice_axis(&table, 1, i * self.hidden_size, (i + 1) * self.hidden_size);
        Ok(AdaLnModulation {
            shift_msa: part(0)?,
            scale_msa: part(1)?,
            gate_msa: part(2)?,
            shift_mlp: part(3)?,
            scale_mlp: part(4)?,
            gate_mlp: part(5)?,
        })
    }
}

/// `x · (1 + scale[idx]) + shift[idx]`, gathering one modulation row per sequence row.
fn modulate(x: &Array, scale: &Array, shift: &Array, indices: &Array) -> Result<Array> {
    let scale = scale.take_axis(indices, 0)?;
    let shift = shift.take_axis(indices, 0)?;
    let one = Array::from_f32(1.0).as_dtype(scale.dtype())?;
    Ok(add(&multiply(x, &add(&one, &scale)?)?, &shift)?)
}

/// One block of the 50-layer stack.
#[derive(Debug, Clone)]
pub struct DitBlock {
    norm1: RmsNorm,
    attn: DitAttention,
    norm2: RmsNorm,
    ff: DitFeedForward,
    /// `pub` so sc-17145 can precompute the modulation and then drop the block's copy.
    pub adaln_proj: AdaLnProjection,
}

impl DitBlock {
    /// Load block `prefix` (e.g. `transformer_blocks.0`).
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            norm1: RmsNorm::from_weights(w, &format!("{prefix}.norm1"), cfg.norm_eps, dtype)?,
            attn: DitAttention::from_weights(w, &format!("{prefix}.attn"), cfg, dtype)?,
            norm2: RmsNorm::from_weights(w, &format!("{prefix}.norm2"), cfg.norm_eps, dtype)?,
            ff: DitFeedForward::from_weights(w, &format!("{prefix}.ff"), dtype)?,
            adaln_proj: AdaLnProjection::from_weights(
                w,
                &format!("{prefix}.adaln_proj.linear"),
                cfg,
                dtype,
            )?,
        })
    }

    /// Every tensor name a block consumes — 12 in the published checkpoint.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = RmsNorm::names(&format!("{prefix}.norm1")).to_vec();
        v.extend(RmsNorm::names(&format!("{prefix}.norm2")));
        v.extend(DitAttention::names(&format!("{prefix}.attn")));
        v.extend(DitFeedForward::names(&format!("{prefix}.ff")));
        v.extend(AdaLnProjection::names(&format!("{prefix}.adaln_proj.linear")));
        v
    }

    /// Project the timestep embedding into this block's modulation tables.
    ///
    /// Separated from [`Self::forward`] so sc-17145 can build these once per schedule, hold them,
    /// and release `adaln_proj` — the single biggest memory lever in the port (~13 B of the 33 B).
    pub fn modulation(&self, temb: &Array) -> Result<AdaLnModulation> {
        self.adaln_proj.forward(temb)
    }

    /// `x + gate_msa·attn(mod(norm1(x)))`, then `x + gate_mlp·ff(mod(norm2(x)))`.
    ///
    /// `adaln_indices` is `timestep_indices · MODALITY_NUM + token_tags`, one entry per row of the
    /// packed sequence.
    pub fn forward(
        &self,
        x: &Array,
        modulation: &AdaLnModulation,
        adaln_indices: &Array,
        rope: &MmRope,
        tables: &MmRopeTables,
    ) -> Result<Array> {
        let s = x.shape();
        if s.len() != 3 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit block: expected [B, S, hidden], got {s:?}"
            )));
        }
        if adaln_indices.shape() != [s[1]] {
            return Err(Error::Msg(format!(
                "minimax-h3 dit block: adaln_indices must be [seq_len={}], got {:?}",
                s[1],
                adaln_indices.shape()
            )));
        }

        let normed = modulate(
            &self.norm1.forward(x)?,
            &modulation.scale_msa,
            &modulation.shift_msa,
            adaln_indices,
        )?;
        let attn = self.attn.forward(&normed, Some((rope, tables)))?;
        let gate = modulation.gate_msa.take_axis(adaln_indices, 0)?;
        let x = add(x, &multiply(&gate, &attn)?)?;

        let normed = modulate(
            &self.norm2.forward(&x)?,
            &modulation.scale_mlp,
            &modulation.shift_mlp,
            adaln_indices,
        )?;
        let ff = self.ff.forward(&normed)?;
        let gate = modulation.gate_mlp.take_axis(adaln_indices, 0)?;
        Ok(add(&x, &multiply(&gate, &ff)?)?)
    }

    /// [`Self::modulation`] followed by [`Self::forward`] — the un-cached path the reference block
    /// takes, kept so parity can be shown against it directly.
    pub fn forward_with_temb(
        &self,
        x: &Array,
        temb: &Array,
        adaln_indices: &Array,
        rope: &MmRope,
        tables: &MmRopeTables,
    ) -> Result<Array> {
        let modulation = self.modulation(temb)?;
        self.forward(x, &modulation, adaln_indices, rope, tables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published block ships exactly 12 tensors, and none of the attention or feed-forward
    /// ones is a bias.
    #[test]
    fn block_names_cover_the_published_twelve_tensors() {
        let names = DitBlock::names("transformer_blocks.0");
        assert_eq!(names.len(), 12, "got {names:?}");
        for expect in [
            "transformer_blocks.0.norm1.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
            "transformer_blocks.0.adaln_proj.linear.bias",
        ] {
            assert!(names.contains(&expect.to_string()), "missing {expect}");
        }
        assert!(
            !names.iter().any(|n| n.starts_with("transformer_blocks.0.attn.to_q.bias")),
            "the attention projections are bias-free"
        );
        assert!(
            !names.iter().any(|n| n.contains(".ff.") && n.ends_with(".bias")),
            "the feed-forward is bias-free"
        );
    }

    /// `1 + scale`, not `scale`. An implementation that multiplied by the raw scale would be an
    /// identity at the reference's zero-initialization and near-identity thereafter.
    #[test]
    fn modulate_applies_one_plus_scale() {
        let x = Array::from_slice(&[2.0f32, 4.0], &[1, 2]);
        // Two table rows; index 1 selects the second.
        let scale = Array::from_slice(&[9.0f32, 9.0, 0.5, 0.5], &[2, 2]);
        let shift = Array::from_slice(&[9.0f32, 9.0, 1.0, -1.0], &[2, 2]);
        let idx = Array::from_slice(&[1i32], &[1]);
        let out = modulate(&x, &scale, &shift, &idx).unwrap();
        assert_eq!(out.as_slice::<f32>(), &[2.0 * 1.5 + 1.0, 4.0 * 1.5 - 1.0]);
    }
}

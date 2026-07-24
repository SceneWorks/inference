//! One dual-stream NR-MMDiT block — **owned by sc-14040**.
//!
//! Port of `MageFlowTransformerBlock` (`_vendor/mage_flow/models/modules/mage_layers.py:514-665`).
//! All [`MageFlowConfig::depth`](crate::config::MageFlowConfig::depth) blocks are dual-stream:
//! [`crate::config::DEPTH_SINGLE_BLOCKS`] is 0, so there is no single-stream tail (unlike FLUX).
//!
//! Per stream: adaLN modulation `SiLU → Linear(dim, 6·dim)` producing `(shift, scale, gate) × 2`,
//! over `LayerNorm(elementwise_affine=False, eps=1e-6)`; then the joint attention
//! ([`crate::attention`]) and the gelu-approximate FFN ([`crate::feed_forward`]), each with a gated
//! residual.
//!
//! The six modulation vectors are laid out as one `[segments, 6·dim]` projection, split in half
//! (`chunk(2)`) into the attention and FFN triples and then in thirds (`chunk(3)`) into
//! `(shift, scale, gate)` — **shift first**, unlike [`crate::final_layer`]'s output head, which
//! chunks `(scale, shift)`. The two orders sit 150 lines apart in the same reference file
//! (`:561` vs `:715`); getting either backwards is silent.
//!
//! **Trap:** the modulation broadcast uses `repeat_interleave` with an int32 `repeats` tensor
//! (`:566`) — the op that makes the reference unrunnable on torch MPS. It only fires when the pack
//! carries ≥2 segments, which is why a `cfg <= 1` MPS run completes and silently produces
//! garbage-adjacent output. Irrelevant to MLX numerics, but it explains why the goldens are
//! CPU-dumped and why `MAGE_DEVICE=cpu` is mandatory when regenerating them. Here that broadcast is
//! a gather over the precomputed per-token segment ids in [`crate::rope_embedder::PackContext`].

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::split;
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{nn, Error, Result};

use crate::attention::{DualStream, MageJointAttention};
use crate::config::{MageFlowConfig, NORM_EPS};
use crate::feed_forward::{FfnActivation, MageFeedForward};
use crate::rope_embedder::PackContext;
use crate::transformer::Linear;

/// One `(shift, scale, gate)` triple already expanded to `[1, tokens, dim]`.
struct Modulation {
    shift: Array,
    scale: Array,
    gate: Array,
}

#[derive(Debug, Clone)]
pub struct MageTransformerBlock {
    img_mod: Linear,
    txt_mod: Linear,
    attn: MageJointAttention,
    img_mlp: MageFeedForward,
    txt_mlp: MageFeedForward,
    eps: f32,
}

impl MageTransformerBlock {
    /// Load from `{prefix}.{img_mod.1,txt_mod.1,attn.*,img_mlp.*,txt_mlp.*}` — e.g.
    /// `transformer_blocks.0`.
    ///
    /// `img_mod` / `txt_mod` are `nn.Sequential(nn.SiLU(), nn.Linear(dim, 6·dim))`, so the Linear
    /// is at **index 1**; index 0 is the weightless activation (`mage_layers.py:530-533`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &MageFlowConfig) -> Result<Self> {
        Ok(Self {
            img_mod: Linear::from_weights(w, &format!("{prefix}.img_mod.1"))?,
            txt_mod: Linear::from_weights(w, &format!("{prefix}.txt_mod.1"))?,
            attn: MageJointAttention::from_weights(
                w,
                &format!("{prefix}.attn"),
                cfg.num_heads,
                cfg.head_dim(),
                NORM_EPS,
            )?,
            img_mlp: MageFeedForward::from_weights(w, &format!("{prefix}.img_mlp"))?,
            txt_mlp: MageFeedForward::from_weights(w, &format!("{prefix}.txt_mlp"))?,
            eps: NORM_EPS,
        })
    }

    pub fn attention(&self) -> &MageJointAttention {
        &self.attn
    }

    pub fn img_mlp(&self) -> &MageFeedForward {
        &self.img_mlp
    }

    pub fn txt_mlp(&self) -> &MageFeedForward {
        &self.txt_mlp
    }

    /// Swap both streams' FFN activation — **a parity-suite divergence knob**; see
    /// [`MageFeedForward::set_activation`].
    pub fn set_ffn_activation(&mut self, activation: FfnActivation) {
        self.img_mlp.set_activation(activation);
        self.txt_mlp.set_activation(activation);
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.img_mod.cast_weights(dtype)?;
        self.txt_mod.cast_weights(dtype)?;
        self.attn.cast_weights(dtype)?;
        self.img_mlp.cast_weights(dtype)?;
        self.txt_mlp.cast_weights(dtype)
    }

    /// `temb`: `[segments, dim]`. Streams in and out are `[1, tokens, dim]`.
    pub fn forward(
        &self,
        stream: &DualStream,
        temb: &Array,
        ctx: &PackContext,
    ) -> Result<DualStream> {
        let dim = stream.img.shape()[2];
        let (img1, img2) = self.modulations(
            &self.img_mod,
            temb,
            ctx.img_segment_ids(),
            ctx.layout().img_tokens(),
            dim,
            ctx.segments(),
        )?;
        let (txt1, txt2) = self.modulations(
            &self.txt_mod,
            temb,
            ctx.txt_segment_ids(),
            ctx.layout().txt_tokens(),
            dim,
            ctx.segments(),
        )?;

        // norm1 + modulation, both streams, then ONE joint attention.
        let modulated = DualStream {
            img: modulate(&layer_norm(&stream.img, None, None, self.eps)?, &img1)?,
            txt: modulate(&layer_norm(&stream.txt, None, None, self.eps)?, &txt1)?,
        };
        let attn = self.attn.forward(&modulated, ctx)?;
        let img = nn::gated(&stream.img, &img1.gate, &attn.img)?;
        let txt = nn::gated(&stream.txt, &txt1.gate, &attn.txt)?;

        // norm2 + modulation + per-stream FFN, gated residual.
        let img_ffn = self
            .img_mlp
            .forward(&modulate(&layer_norm(&img, None, None, self.eps)?, &img2)?)?;
        let img = nn::gated(&img, &img2.gate, &img_ffn)?;
        let txt_ffn = self
            .txt_mlp
            .forward(&modulate(&layer_norm(&txt, None, None, self.eps)?, &txt2)?)?;
        let txt = nn::gated(&txt, &txt2.gate, &txt_ffn)?;

        // The reference's fp16 overflow clip (`:659-663`) is dead on this path: the checkpoint is
        // bf16 and the port never runs fp16, so no clip is ported.
        Ok(DualStream { txt, img })
    }

    /// `SiLU → Linear(dim, 6·dim)` → `chunk(2)` → `chunk(3)`, each triple expanded from
    /// `[segments, dim]` to `[1, tokens, dim]` by the reference's `repeat_interleave`.
    fn modulations(
        &self,
        projection: &Linear,
        temb: &Array,
        segment_ids: &Array,
        tokens: i32,
        dim: i32,
        segments: usize,
    ) -> Result<(Modulation, Modulation)> {
        check_conditioning(temb, segments, dim)?;
        let params = projection.forward(&nn::silu(temb)?)?;
        let halves = split(&params, 2, 1)?;
        Ok((
            Modulation::from_triple(&halves[0], segment_ids, tokens, dim)?,
            Modulation::from_triple(&halves[1], segment_ids, tokens, dim)?,
        ))
    }
}

impl Modulation {
    /// `shift, scale, gate = mod_params.chunk(3, dim=-1)` (`mage_layers.py:561`) — **shift first**
    /// — then `repeat_interleave` to one row per token (`:566-568`), reshaped for `[1, tokens, dim]`
    /// broadcasting.
    fn from_triple(params: &Array, segment_ids: &Array, tokens: i32, dim: i32) -> Result<Self> {
        let parts = split(params, 3, 1)?;
        let expand = |p: &Array| -> Result<Array> {
            Ok(p.take_axis(segment_ids, 0)?.reshape(&[1, tokens, dim])?)
        };
        Ok(Self {
            shift: expand(&parts[0])?,
            scale: expand(&parts[1])?,
            gate: expand(&parts[2])?,
        })
    }
}

/// The conditioning must carry **one row per packed segment**, never a single broadcastable row.
///
/// Broadcasting would be silent and wrong in exactly the case that matters: under fused CFG the
/// two halves of the pack are the conditional and unconditional branches, and a `[1, dim]`
/// conditioning would apply one branch's modulation to both.
pub(crate) fn check_conditioning(temb: &Array, segments: usize, dim: i32) -> Result<()> {
    if temb.shape() != [segments as i32, dim] {
        return Err(Error::Msg(format!(
            "mage_flow: adaLN conditioning must be [{segments}, {dim}] (one row per packed \
             segment), got {:?}",
            temb.shape()
        )));
    }
    Ok(())
}

/// `x · (1 + scale) + shift` (`mage_layers.py:570`).
///
/// The literal `1` is a Python int, so torch weak-types it to the tensor dtype and the sum rounds
/// in bf16 — [`mlx_gen::nn::modulate`]'s `one_matches_scale = true` policy. A strong f32 `1` would
/// promote the whole modulated stream to f32 and diverge.
fn modulate(normed: &Array, m: &Modulation) -> Result<Array> {
    nn::modulate(normed, &m.scale, &m.shift, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope_embedder::{ImgShape, MsRope, PackLayout};

    fn ctx(img: Vec<ImgShape>, txt: Vec<i32>) -> PackContext {
        let layout = PackLayout::generation(img, txt).unwrap();
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        PackContext::new(layout, &rope).unwrap()
    }

    #[test]
    fn triples_are_shift_scale_gate_in_that_order() {
        // One segment, dim 2: params = [shift(2) | scale(2) | gate(2)].
        let params = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
        let ids = Array::from_slice(&[0i32, 0], &[2]);
        let m = Modulation::from_triple(&params, &ids, 2, 2).unwrap();
        assert_eq!(m.shift.as_slice::<f32>(), &[1.0, 2.0, 1.0, 2.0]);
        assert_eq!(m.scale.as_slice::<f32>(), &[3.0, 4.0, 3.0, 4.0]);
        assert_eq!(m.gate.as_slice::<f32>(), &[5.0, 6.0, 5.0, 6.0]);
    }

    #[test]
    fn modulation_rows_follow_their_own_segment() {
        // Two segments of 1 and 2 tokens ⇒ rows [0, 1, 1].
        let params = Array::from_slice(
            &[
                0.0f32, 0.0, 0.0, 0.0, 7.0, 7.0, // segment 0
                0.0, 0.0, 0.0, 0.0, 9.0, 9.0, // segment 1
            ],
            &[2, 6],
        );
        let ids = Array::from_slice(&[0i32, 1, 1], &[3]);
        let m = Modulation::from_triple(&params, &ids, 3, 2).unwrap();
        assert_eq!(m.gate.as_slice::<f32>(), &[7.0, 7.0, 9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn modulate_applies_one_plus_scale_then_shift() {
        let normed = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let m = Modulation {
            shift: Array::from_slice(&[1.0f32, -1.0], &[1, 1, 2]),
            scale: Array::from_slice(&[1.0f32, 0.5], &[1, 1, 2]),
            gate: Array::from_slice(&[0.0f32, 0.0], &[1, 1, 2]),
        };
        assert_eq!(
            modulate(&normed, &m).unwrap().as_slice::<f32>(),
            &[5.0, 5.0]
        );
    }

    #[test]
    fn conditioning_must_carry_one_row_per_segment() {
        let ctx = ctx(
            vec![ImgShape::latent(2, 2), ImgShape::latent(2, 2)],
            vec![3, 3],
        );
        assert_eq!(ctx.segments(), 2);
        assert_eq!(ctx.img_segment_ids().as_slice::<i32>().len(), 8);
        // A `[1, dim]` conditioning against a 2-segment pack must be rejected, not broadcast:
        // broadcasting would apply the conditional branch's modulation to the unconditional half
        // of a fused-CFG forward, silently and with no shape error.
        let one_row = Array::from_slice(&[0.0f32; 4], &[1, 4]);
        assert!(check_conditioning(&one_row, ctx.segments(), 4).is_err());
        let two_rows = Array::from_slice(&[0.0f32; 8], &[2, 4]);
        assert!(check_conditioning(&two_rows, ctx.segments(), 4).is_ok());
    }
}

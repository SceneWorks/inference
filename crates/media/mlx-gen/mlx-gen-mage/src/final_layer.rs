//! Output head — **owned by sc-14040**.
//!
//! `AdaLayerNormContinuous(hidden_size, hidden_size, elementwise_affine=False, eps=1e-6)` followed
//! by `proj_out = Linear(hidden_size → patch_size² · out_channels, bias=True)`
//! (`_vendor/mage_flow/models/mage_flow.py:90-91`, `:147-152`;
//! `models/modules/mage_layers.py:668-725`).
//!
//! `AdaLayerNormContinuous` is `SiLU → Linear → (scale, shift)` applied to a non-affine LayerNorm.
//! Only the **image** stream reaches the head; the text stream is dropped after the last block.
//! With `patch_size == 1` there is no unpatchify — the head emits one 128-channel latent cell per
//! token, and `unpack` (`models/utils.py:36`) only reshapes `(h w) c → c h w` at
//! `ceil(height/16) × ceil(width/16)`.
//!
//! **Chunk order trap.** Here the projection splits as `scale, shift = chunk(emb, 2)` — *scale
//! first* (`mage_layers.py:715`, `:720`) — the opposite of the block's `shift, scale, gate`
//! (`:561`). Same file, same idiom, opposite order; swapping them costs nothing at load time and
//! everything at inference.
//!
//! The `eps` also deserves care: `AdaLayerNormContinuous`'s own default is `1e-5`
//! (`mage_layers.py:693`), and `MageFlow.__init__` **overrides** it to `1e-6` at the call site
//! (`mage_flow.py:90`). [`crate::config::NORM_EPS`] carries the override.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::split;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::{nn, Error, Result};

use crate::config::NORM_EPS;
use crate::quant::{floor_bits, FINAL_MOD_BASE};
use crate::rope_embedder::PackContext;
use crate::transformer::Linear;
use crate::transformer_block::check_conditioning;

/// `norm_out` (an `AdaLayerNormContinuous`) + `proj_out`.
#[derive(Debug, Clone)]
pub struct MageFinalLayer {
    norm_linear: Linear,
    proj_out: Linear,
    eps: f32,
}

impl MageFinalLayer {
    /// Pack both head projections, holding `norm_linear` at its 8-bit floor (sc-15071).
    ///
    /// A uniformly-Q4 head is what made the Q4 tier render a repeating tiled texture instead of the
    /// prompt; [`crate::convert::quant_floor_bits`] documents the mechanism and the per-group
    /// measurements, and is the same seam the offline converter calls, so a pre-quantized tier
    /// stays byte-identical to load-time quantization.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.norm_linear
            .quantize(floor_bits(FINAL_MOD_BASE, bits))?;
        self.proj_out.quantize(bits)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        usize::from(self.norm_linear.is_quantized()) + usize::from(self.proj_out.is_quantized())
    }

    /// Load from `norm_out.linear.{weight,bias}` and `proj_out.{weight,bias}`. The LayerNorm inside
    /// `norm_out` is `elementwise_affine=False`, so it contributes no tensors.
    pub fn from_weights(w: &Weights, norm_prefix: &str, proj_prefix: &str) -> Result<Self> {
        Ok(Self {
            norm_linear: Linear::from_weights(w, &format!("{norm_prefix}.linear"))?,
            proj_out: Linear::from_weights(w, proj_prefix)?,
            eps: NORM_EPS,
        })
    }

    /// Emitted channels per token (`patch_size² · out_channels`; 128 in production).
    pub fn out_channels(&self) -> i32 {
        self.proj_out.out_features()
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.norm_linear.cast_weights(dtype)?;
        self.proj_out.cast_weights(dtype)
    }

    /// `img`: `[1, img_tokens, hidden_size]`, `temb`: `[segments, hidden_size]` →
    /// `[1, img_tokens, out_channels]`.
    pub fn forward(&self, img: &Array, temb: &Array, ctx: &PackContext) -> Result<Array> {
        let dim = img.shape()[2];
        let tokens = ctx.layout().img_tokens();
        if img.shape() != [1, tokens, dim] {
            return Err(Error::Msg(format!(
                "mage_flow: output head expects [1, {tokens}, {dim}], got {:?}",
                img.shape()
            )));
        }
        check_conditioning(temb, ctx.segments(), dim)?;

        // `self.linear(self.silu(conditioning_embedding).to(x.dtype))` (`mage_layers.py:713`).
        let emb = self
            .norm_linear
            .forward(&nn::silu(temb)?.as_dtype(img.dtype())?)?;
        let parts = split(&emb, 2, 1)?;
        let expand = |p: &Array| -> Result<Array> {
            Ok(p.take_axis(ctx.img_segment_ids(), 0)?
                .reshape(&[1, tokens, dim])?)
        };
        // scale FIRST, then shift.
        let scale = expand(&parts[0])?;
        let shift = expand(&parts[1])?;
        let normed = layer_norm(img, None, None, self.eps)?;
        let modulated = nn::modulate(&normed, &scale, &shift, true)?;
        self.proj_out.forward(&modulated)
    }
}

/// LoRA/LoKr targets on the output head (sc-14057).
///
/// Unlike every other sub-host in this crate, the paths here are **absolute** (rooted at the
/// transformer), not relative: the head owns two *sibling* checkpoint roots — `norm_out.linear`
/// (the `AdaLayerNormContinuous` projection) and `proj_out` — so there is no single prefix a parent
/// could delegate under. [`MageTransformer`](crate::transformer::MageTransformer) therefore matches
/// on `["norm_out", ..] | ["proj_out", ..]` and forwards the *whole* path, and splices this list in
/// unprefixed. A PEFT `target_modules="all-linear"` community adapter trains both.
impl AdaptableHost for MageFinalLayer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["norm_out", "linear"] => Some(self.norm_linear.adaptable_mut()),
            ["proj_out"] => Some(self.proj_out.adaptable_mut()),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        vec!["norm_out.linear".to_string(), "proj_out".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope_embedder::{ImgShape, MsRope, PackLayout};

    /// The head chunks `(scale, shift)` — **scale first** (`mage_layers.py:715`, `:720`) — the
    /// opposite of the block's `(shift, scale, gate)` (`:561`).
    ///
    /// The end-to-end f32 fixture does catch a swap (`tests/mage_flow_small.rs` fails at max_rel
    /// 1.008 with the halves exchanged), but a whole-model failure does not *name* the mistake.
    /// This pins it directly, with hand-computable numbers.
    #[test]
    fn the_head_chunks_scale_first_then_shift() {
        let dim = 2;
        let mut w = Weights::empty();
        // silu(0) = 0, so `emb` is exactly the bias: [scale | shift] = [1, 1 | 10, 20].
        w.insert(
            "n.linear.weight",
            Array::from_slice(&[0.0f32; 8], &[2 * dim, dim]),
        );
        w.insert(
            "n.linear.bias",
            Array::from_slice(&[1.0f32, 1.0, 10.0, 20.0], &[2 * dim]),
        );
        // proj_out = identity, so the output IS the modulated stream.
        w.insert(
            "p.weight",
            Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[dim, dim]),
        );
        w.insert("p.bias", Array::from_slice(&[0.0f32, 0.0], &[dim]));
        let head = MageFinalLayer::from_weights(&w, "n", "p").unwrap();

        let layout = PackLayout::generation(vec![ImgShape::latent(1, 2)], vec![1]).unwrap();
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let ctx = PackContext::new(layout, &rope).unwrap();

        // LayerNorm over dim 2 maps any (a, b) with a != b to (-1, 1).
        let img = Array::from_slice(&[0.0f32, 4.0, -3.0, 5.0], &[1, 2, dim]);
        let temb = Array::from_slice(&[0.0f32, 0.0], &[1, dim]);
        let out = head.forward(&img, &temb, &ctx).unwrap();

        // scale-first ⇒ (1 + 1)·(∓1) + (10, 20) = (8, 22) per token.
        // shift-first would give (1 + 10)·(-1) + 1 = -10 and (1 + 20)·1 + 1 = 22 — the second
        // component collides, which is why the first is the one that discriminates.
        let got = out.as_slice::<f32>();
        for (i, want) in [8.0f32, 22.0, 8.0, 22.0].into_iter().enumerate() {
            assert!(
                (got[i] - want).abs() < 1e-4,
                "component {i}: got {}, want {want} — scale and shift look swapped",
                got[i]
            );
        }
    }

    /// sc-14057: both head projections are adapter targets, addressed by their **absolute**
    /// checkpoint paths. `norm_out` alone (the weightless non-affine LayerNorm) is not a target,
    /// and the head must not answer for a path it does not own.
    #[test]
    fn the_head_projections_are_routable_adapter_targets() {
        let mut w = Weights::empty();
        for (prefix, out, inp) in [("n.linear", 4, 2), ("p", 2, 2)] {
            w.insert(
                format!("{prefix}.weight"),
                Array::from_slice(&vec![0.0f32; out * inp], &[out as i32, inp as i32]),
            );
            w.insert(
                format!("{prefix}.bias"),
                Array::from_slice(&vec![0.0f32; out], &[out as i32]),
            );
        }
        let mut head = MageFinalLayer::from_weights(&w, "n", "p").unwrap();
        assert_eq!(head.adaptable_paths(), ["norm_out.linear", "proj_out"]);
        for path in head.adaptable_paths() {
            let segs: Vec<&str> = path.split('.').collect();
            assert!(
                head.adaptable_mut(&segs).is_some(),
                "{path} is enumerated but does not resolve"
            );
        }
        assert!(head.adaptable_mut(&["norm_out"]).is_none());
        assert!(head.adaptable_mut(&["img_in"]).is_none());
    }
}

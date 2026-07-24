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

use mlx_gen::weights::Weights;
use mlx_gen::{nn, Error, Result};

use crate::config::NORM_EPS;
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

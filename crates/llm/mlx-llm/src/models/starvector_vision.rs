//! Native CLIP vision tower and adapter for the published StarVector-1B snapshot.
//!
//! The checkpoint is not a Hugging Face `CLIPVisionModel`: it carries the project's original
//! OpenAI-CLIP style `VisionTransformer` weights and a BatchNorm-over-token-rows adapter.  Keeping
//! that distinction here prevents silently applying the later 8B SigLIP preprocessing/projection.

use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, multiply, rsqrt, sigmoid, split_sections, subtract,
};
use mlx_rs::Array;

use crate::error::Result;
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::nn::{conv2d, layer_norm, linear};
use crate::primitives::Weights;

/// Published StarVector-1B visual geometry: CLIP ViT-L/14-like, 224 pixels and 257 rows.
pub const IMAGE_SIZE: usize = 224;
pub const IMAGE_TOKENS: i32 = 257;
const WIDTH: i32 = 1024;
const PATCH: i32 = 14;
const HEADS: i32 = 16;
const LAYERS: usize = 23;

/// The image tower from `model.image_encoder`.
pub struct StarVectorClipVision {
    patch: Array,
    class_embedding: Array,
    positions: Array,
    ln_pre_weight: Array,
    ln_pre_bias: Array,
    layers: Vec<ClipBlock>,
    ln_vision_weight: Array,
    ln_vision_bias: Array,
}

impl StarVectorClipVision {
    /// Load every image-tower tensor from the exact StarVector-1B namespaced checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let visual = |suffix: &str| format!("{prefix}.visual_encoder.{suffix}");
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            layers.push(ClipBlock::from_weights(
                w,
                &visual(&format!("transformer.resblocks.{index}")),
            )?);
        }
        Ok(Self {
            patch: w
                .require(&visual("conv1.weight"))?
                .transpose_axes(&[0, 2, 3, 1])?,
            class_embedding: w.require(&visual("class_embedding"))?.clone(),
            positions: w.require(&visual("positional_embedding"))?.clone(),
            ln_pre_weight: w.require(&visual("ln_pre.weight"))?.clone(),
            ln_pre_bias: w.require(&visual("ln_pre.bias"))?.clone(),
            layers,
            ln_vision_weight: w.require(&format!("{prefix}.ln_vision.weight"))?.clone(),
            ln_vision_bias: w.require(&format!("{prefix}.ln_vision.bias"))?.clone(),
        })
    }

    /// Encode preprocessed NHWC `[B,224,224,3]` pixels into the 257 CLIP token rows.
    pub fn forward(&self, pixels: &Array) -> Result<Array> {
        let batch = pixels.shape()[0];
        let patches = conv2d(pixels, &self.patch, None, PATCH, 0)?.reshape(&[
            batch,
            IMAGE_TOKENS - 1,
            WIDTH,
        ])?;
        let class = self.class_embedding.reshape(&[1, 1, WIDTH])?;
        let class = broadcast_to(&class, &[batch, 1, WIDTH])?;
        let mut hidden = concatenate_axis(&[&class, &patches], 1)?;
        hidden = add(&hidden, &self.positions.reshape(&[1, IMAGE_TOKENS, WIDTH])?)?;
        hidden = layer_norm(
            &hidden,
            Some(&self.ln_pre_weight),
            Some(&self.ln_pre_bias),
            1e-5,
        )?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        layer_norm(
            &hidden,
            Some(&self.ln_vision_weight),
            Some(&self.ln_vision_bias),
            1e-5,
        )
    }
}

struct ClipBlock {
    ln_1_weight: Array,
    ln_1_bias: Array,
    in_proj_weight: Array,
    in_proj_bias: Array,
    out_proj_weight: Array,
    out_proj_bias: Array,
    ln_2_weight: Array,
    ln_2_bias: Array,
    fc_weight: Array,
    fc_bias: Array,
    mlp_proj_weight: Array,
    mlp_proj_bias: Array,
}

impl ClipBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            ln_1_weight: w.require(&key("ln_1.weight"))?.clone(),
            ln_1_bias: w.require(&key("ln_1.bias"))?.clone(),
            in_proj_weight: w.require(&key("attn.in_proj_weight"))?.clone(),
            in_proj_bias: w.require(&key("attn.in_proj_bias"))?.clone(),
            out_proj_weight: w.require(&key("attn.out_proj.weight"))?.clone(),
            out_proj_bias: w.require(&key("attn.out_proj.bias"))?.clone(),
            ln_2_weight: w.require(&key("ln_2.weight"))?.clone(),
            ln_2_bias: w.require(&key("ln_2.bias"))?.clone(),
            fc_weight: w.require(&key("mlp.c_fc.weight"))?.clone(),
            fc_bias: w.require(&key("mlp.c_fc.bias"))?.clone(),
            mlp_proj_weight: w.require(&key("mlp.c_proj.weight"))?.clone(),
            mlp_proj_bias: w.require(&key("mlp.c_proj.bias"))?.clone(),
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let shape = hidden.shape();
        let (batch, sequence) = (shape[0], shape[1]);
        let normed = layer_norm(hidden, Some(&self.ln_1_weight), Some(&self.ln_1_bias), 1e-5)?;
        let qkv = linear(&normed, &self.in_proj_weight, Some(&self.in_proj_bias))?;
        let parts = split_sections(&qkv, &[WIDTH, 2 * WIDTH], 2)?;
        let head_dim = WIDTH / HEADS;
        let heads = |x: &Array| -> Result<Array> {
            Ok(x.reshape(&[batch, sequence, HEADS, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let query = heads(&parts[0])?;
        let key = heads(&parts[1])?;
        let value = heads(&parts[2])?;
        let attended = sdpa(
            &query,
            &key,
            &value,
            1.0 / (head_dim as f32).sqrt(),
            AttnMask::None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, sequence, WIDTH])?;
        let hidden = add(
            hidden,
            &linear(&attended, &self.out_proj_weight, Some(&self.out_proj_bias))?,
        )?;
        let normed = layer_norm(
            &hidden,
            Some(&self.ln_2_weight),
            Some(&self.ln_2_bias),
            1e-5,
        )?;
        let mlp = linear(&normed, &self.fc_weight, Some(&self.fc_bias))?;
        let scale = Array::from_f32(1.702).as_dtype(mlp.dtype())?;
        let mlp = multiply(&mlp, &sigmoid(&multiply(&mlp, &scale)?)?)?;
        let mlp = linear(&mlp, &self.mlp_proj_weight, Some(&self.mlp_proj_bias))?;
        Ok(add(&hidden, &mlp)?)
    }
}

/// The StarVector image-to-language connector. The upstream checkpoint uses `BatchNorm1d(257)`
/// over `[batch, token_row, hidden]`, rather than a hidden-dimension LayerNorm.
pub struct StarVectorAdapter {
    fc_weight: Array,
    fc_bias: Array,
    proj_weight: Array,
    proj_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
    running_mean: Array,
    running_var: Array,
}

impl StarVectorAdapter {
    /// Load `model.image_projection` from the published checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            fc_weight: w.require(&key("c_fc.weight"))?.clone(),
            fc_bias: w.require(&key("c_fc.bias"))?.clone(),
            proj_weight: w.require(&key("c_proj.weight"))?.clone(),
            proj_bias: w.require(&key("c_proj.bias"))?.clone(),
            norm_weight: w.require(&key("norm.weight"))?.clone(),
            norm_bias: w.require(&key("norm.bias"))?.clone(),
            running_mean: w.require(&key("norm.running_mean"))?.clone(),
            running_var: w.require(&key("norm.running_var"))?.clone(),
        })
    }

    /// Map CLIP rows `[B,257,1024]` into decoder rows `[B,257,2048]`.
    pub fn forward(&self, image: &Array) -> Result<Array> {
        let hidden = linear(image, &self.fc_weight, Some(&self.fc_bias))?;
        let hidden = multiply(&hidden, &sigmoid(&hidden)?)?;
        let hidden = linear(&hidden, &self.proj_weight, Some(&self.proj_bias))?;
        let row = |x: &Array| x.reshape(&[1, IMAGE_TOKENS, 1]);
        let eps = Array::from_f32(1e-5).as_dtype(hidden.dtype())?;
        let centered = subtract(&hidden, &row(&self.running_mean)?)?;
        let normalized = multiply(&centered, &rsqrt(&add(&row(&self.running_var)?, &eps)?)?)?;
        Ok(add(
            &multiply(&normalized, &row(&self.norm_weight)?)?,
            &row(&self.norm_bias)?,
        )?)
    }
}

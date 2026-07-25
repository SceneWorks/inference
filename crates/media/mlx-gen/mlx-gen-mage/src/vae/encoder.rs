//! Mage-VAE one-step image encoder (`_DConvEncoder`).

use mlx_rs::ops::{concatenate_axis, zeros_dtype};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::dico::{DiCoBlock, EncoderDiCoBlock};
use super::layers::{layer_norm_2d, Conv2d, LayerNormAffine};
use super::timestep::TimestepEmbedder;

const HIDDEN_SIZE: i32 = 384;
const HEAD_SIZE: i32 = 768;
const Z_CHANNELS: i32 = 128;
const PATCH_SIZE: i32 = 16;
const NUM_HEAD_BLOCKS: usize = 2;
const NUM_BLOCKS: usize = 21;

/// Deterministic posterior moments produced by the encoder.
pub struct EncoderMoments {
    pub mean: Array,
    pub logvar: Array,
}

pub struct DConvEncoder {
    patch_cond_embed: Conv2d,
    head_blocks: Vec<EncoderDiCoBlock>,
    proj_down: Conv2d,
    z_proj: Conv2d,
    fuse_proj: Conv2d,
    t_embedder: TimestepEmbedder,
    blocks: Vec<DiCoBlock>,
    norm_out: LayerNormAffine,
    proj_out: Conv2d,
    dtype: Dtype,
}

impl DConvEncoder {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.t_embedder.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        Ok(())
    }

    pub(crate) fn quantization_count(&self) -> (usize, usize) {
        let mut count = self.t_embedder.quantization_count();
        for block in &self.blocks {
            let next = block.quantization_count();
            count = (count.0 + next.0, count.1 + next.1);
        }
        count
    }

    pub fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        let mut head_blocks = Vec::with_capacity(NUM_HEAD_BLOCKS);
        for index in 0..NUM_HEAD_BLOCKS {
            head_blocks.push(EncoderDiCoBlock::from_weights(
                w,
                &format!("{prefix}.head_blocks.{index}"),
                HEAD_SIZE,
            )?);
        }
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for index in 0..NUM_BLOCKS {
            blocks.push(DiCoBlock::from_weights(
                w,
                &format!("{prefix}.blocks.{index}"),
                HIDDEN_SIZE,
            )?);
        }
        Ok(Self {
            patch_cond_embed: Conv2d::from_weights(
                w,
                &format!("{prefix}.patch_cond_embed"),
                PATCH_SIZE,
                0,
                1,
                true,
            )?,
            head_blocks,
            proj_down: Conv2d::pointwise(w, &format!("{prefix}.proj_down"), true)?,
            z_proj: Conv2d::pointwise(w, &format!("{prefix}.z_proj"), true)?,
            fuse_proj: Conv2d::pointwise(w, &format!("{prefix}.fuse_proj"), true)?,
            t_embedder: TimestepEmbedder::from_weights(w, &format!("{prefix}.t_embedder"))?,
            blocks,
            norm_out: LayerNormAffine::from_weights(w, &format!("{prefix}.norm_out"))?,
            proj_out: Conv2d::pointwise(w, &format!("{prefix}.proj_out"), true)?,
            dtype,
        })
    }

    pub fn fold_adaln_at_zero(&mut self) -> Result<()> {
        let t = zeros_dtype(&[1], self.dtype)?;
        let c = self.t_embedder.forward(&t, self.dtype)?;
        for block in &mut self.blocks {
            block.fold_adaln(&c)?;
        }
        Ok(())
    }

    pub fn moments(&self, image_nchw: &Array) -> Result<EncoderMoments> {
        let shape = image_nchw.shape();
        if shape.len() != 4 || shape[1] != 3 {
            return Err(Error::Msg(format!(
                "mage-vae encode: expected [B, 3, H, W], got {shape:?}"
            )));
        }
        if shape[2] % PATCH_SIZE != 0 || shape[3] % PATCH_SIZE != 0 {
            return Err(Error::Msg(format!(
                "mage-vae encode: H and W must be divisible by {PATCH_SIZE}, got {}x{}",
                shape[2], shape[3]
            )));
        }
        let image = image_nchw
            .transpose_axes(&[0, 2, 3, 1])?
            .as_dtype(self.dtype)?;
        let mut cond = self.patch_cond_embed.forward(&image)?;
        for block in &self.head_blocks {
            cond = block.forward(&cond)?;
        }
        cond = self.proj_down.forward(&cond)?;
        let z = zeros_dtype(
            &[
                shape[0],
                shape[2] / PATCH_SIZE,
                shape[3] / PATCH_SIZE,
                Z_CHANNELS,
            ],
            self.dtype,
        )?;
        let z = self.z_proj.forward(&z)?;
        let mut state = self.fuse_proj.forward(&concatenate_axis(&[cond, z], -1)?)?;
        let t = zeros_dtype(&[shape[0]], self.dtype)?;
        let c = self.t_embedder.forward(&t, self.dtype)?;
        for block in &self.blocks {
            state = block.forward(&state, &c)?;
        }
        let packed = self
            .proj_out
            .forward(&layer_norm_2d(&state, Some(&self.norm_out))?)?
            .transpose_axes(&[0, 3, 1, 2])?;
        let parts = mlx_rs::ops::split(&packed, 2, 1)?;
        if parts.len() != 2 {
            return Err(Error::Msg("mage-vae encode: invalid packed moments".into()));
        }
        let logvar = mlx_rs::ops::maximum(&parts[1], Array::from_f32(-20.0))?;
        let logvar = mlx_rs::ops::minimum(&logvar, Array::from_f32(10.0))?;
        Ok(EncoderMoments {
            mean: parts[0].clone(),
            logvar,
        })
    }
}

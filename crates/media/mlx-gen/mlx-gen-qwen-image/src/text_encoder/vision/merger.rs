//! `PatchMerger`: RMSNorm(`ln_q`) → group each `spatial_merge²` patches into one row → `mlp_0` →
//! exact GELU → `mlp_1`, mapping `embed → out_hidden`. Port of the fork's `qwen_patch_merger.py`.
//!
//! The fork reshapes per image (`[t·h·w, embed] → [-1, embed·merge²]`); because the windowed
//! hidden states are already grouped in `merge²`-sized, image-contiguous blocks, a single global
//! `[-1, embed·merge²]` reshape is identical and avoids threading `grid_thw` through.

use mlx_rs::fast::rms_norm;
use mlx_rs::nn::gelu;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct PatchMerger {
    ln_q: Array,
    mlp0: AdaptableLinear,
    mlp1: AdaptableLinear,
    hidden_merged: i32,
    norm_eps: f32,
}

impl PatchMerger {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        embed_dim: i32,
        spatial_merge_size: i32,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            ln_q: w.require(&join(prefix, "ln_q.weight"))?.clone(),
            mlp0: mlx_gen::quant::lin(
                w,
                &join(prefix, "mlp_0"),
                true,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            mlp1: mlx_gen::quant::lin(
                w,
                &join(prefix, "mlp_1"),
                true,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            hidden_merged: embed_dim * spatial_merge_size * spatial_merge_size,
            norm_eps,
        })
    }

    /// `x`: `[seq, embed]` (window-reordered) → `[seq/merge², out_hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = rms_norm(x, &self.ln_q, self.norm_eps)?.reshape(&[-1, self.hidden_merged])?;
        let x = self.mlp0.forward(&x)?;
        let x = gelu(&x)?;
        self.mlp1.forward(&x)
    }
}

//! The ViT3D video decoder (`vae_vit.py::ViT3DDecoder`).
//!
//! Unlike almost every other diffusion VAE, MiniMax-H3's decoder is **not** a conv stack — it is a
//! 36-layer / 2048-dim transformer that does all 16× spatial and 4× temporal upsampling in a
//! single `proj_out`. The encoder is the conv half; only the decoder is ported here.
//!
//! Token layout, in order:
//!
//! ```text
//! [ T·H·W patch tokens | num_register_tokens learned registers | 1 ZERO cls token ]
//! ```
//!
//! Two easy-to-miss details:
//!
//! - the trailing CLS token is `torch.zeros_like(...)`, **not** a learned parameter — the module
//!   is built with `has_cls_token=False`, so no `cls_token` tensor exists and appending a learned
//!   one would be wrong;
//! - the suffix tokens get **zero** position ids, so they sit at the rotary origin rather than
//!   continuing the patch grid.
//!
//! After the blocks, `proj_out` is applied to the whole sequence and only then truncated back to
//! the patch tokens — the suffix rows are computed and discarded, exactly as upstream.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::blocks::TransformerBlock;
use crate::config::MiniMaxH3VaeConfig;
use crate::rope::{create_token_ids, Rope3d};
use crate::tensor::{pack_tokens, slice_axis, unpack_tokens};

/// The transformer video decoder.
#[derive(Debug, Clone)]
pub struct ViT3dDecoder {
    proj_in_w: Array,
    proj_in_b: Array,
    register_tokens: Array,
    blocks: Vec<TransformerBlock>,
    norm_out_w: Array,
    norm_out_b: Array,
    proj_out_w: Array,
    proj_out_b: Array,
    rope: Rope3d,
    cfg: MiniMaxH3VaeConfig,
    dtype: Dtype,
}

impl ViT3dDecoder {
    /// Load from `decoder.`-prefixed tensors.
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        cfg.validate()?;
        let proj_in_w = w
            .require(&format!("{prefix}.proj_in.weight"))?
            .as_dtype(dtype)?;
        // `proj_in` maps latent channels -> model dim; a mismatch here means the config and the
        // checkpoint disagree about `latent_channels` or the decoder width.
        let shape = proj_in_w.shape();
        if shape != [cfg.dim(), cfg.latent_channels] {
            return Err(Error::Msg(format!(
                "minimax-h3 decoder: {prefix}.proj_in.weight is {shape:?}, expected [{}, {}]",
                cfg.dim(),
                cfg.latent_channels
            )));
        }
        let proj_in_b = w
            .require(&format!("{prefix}.proj_in.bias"))?
            .as_dtype(dtype)?;
        let register_tokens = w
            .require(&format!("{prefix}.register_tokens"))?
            .as_dtype(dtype)?;
        if register_tokens.shape() != [1, cfg.num_register_tokens, cfg.dim()] {
            return Err(Error::Msg(format!(
                "minimax-h3 decoder: {prefix}.register_tokens is {:?}, expected [1, {}, {}]",
                register_tokens.shape(),
                cfg.num_register_tokens,
                cfg.dim()
            )));
        }

        let blocks = (0..cfg.num_layers)
            .map(|i| {
                TransformerBlock::from_weights(
                    w,
                    &format!("{prefix}.transformer_blocks.{i}"),
                    cfg,
                    dtype,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let proj_out_w = w
            .require(&format!("{prefix}.proj_out.weight"))?
            .as_dtype(dtype)?;
        if proj_out_w.shape() != [cfg.patch_dim(), cfg.dim()] {
            return Err(Error::Msg(format!(
                "minimax-h3 decoder: {prefix}.proj_out.weight is {:?}, expected [{}, {}] \
                 (out_channels · patch_size_t · patch_size²)",
                proj_out_w.shape(),
                cfg.patch_dim(),
                cfg.dim()
            )));
        }

        Ok(Self {
            proj_in_w,
            proj_in_b,
            register_tokens,
            blocks,
            norm_out_w: w
                .require(&format!("{prefix}.norm_out.weight"))?
                .as_dtype(dtype)?,
            norm_out_b: w
                .require(&format!("{prefix}.norm_out.bias"))?
                .as_dtype(dtype)?,
            proj_out_w,
            proj_out_b: w
                .require(&format!("{prefix}.proj_out.bias"))?
                .as_dtype(dtype)?,
            rope: Rope3d::new(cfg.rope_apply_dim(), cfg.rope_theta)?,
            cfg: cfg.clone(),
            dtype,
        })
    }

    /// Every tensor name the decoder consumes, in load order.
    pub fn names(prefix: &str, cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        let mut v = vec![
            format!("{prefix}.proj_in.weight"),
            format!("{prefix}.proj_in.bias"),
            format!("{prefix}.register_tokens"),
        ];
        for i in 0..cfg.num_layers {
            v.extend(TransformerBlock::names(&format!(
                "{prefix}.transformer_blocks.{i}"
            )));
        }
        v.extend([
            format!("{prefix}.norm_out.weight"),
            format!("{prefix}.norm_out.bias"),
            format!("{prefix}.proj_out.weight"),
            format!("{prefix}.proj_out.bias"),
        ]);
        v
    }

    /// The config this decoder was built with.
    pub fn config(&self) -> &MiniMaxH3VaeConfig {
        &self.cfg
    }

    /// Decode a `[B, latent_channels, T, H, W]` latent into `[B, out_channels, T·4, H·16, W·16]`.
    pub fn forward(&self, latent: &Array) -> Result<Array> {
        let s = latent.shape();
        if s.len() != 5 {
            return Err(Error::Msg(format!(
                "minimax-h3 decoder: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let (b, c, t, h, w) = (s[0], s[1], s[2], s[3], s[4]);
        if c != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "minimax-h3 decoder: expected {} latent channels, got {c}",
                self.cfg.latent_channels
            )));
        }
        let latent = latent.as_dtype(self.dtype)?;
        let num_patches = t * h * w;

        let tokens = pack_tokens(&latent)?;
        let hidden = linear(&tokens, &self.proj_in_w, &self.proj_in_b)?;

        // registers, then a ZERO cls token (`torch.zeros_like`, not a learned parameter).
        let registers = mlx_rs::ops::broadcast_to(
            &self.register_tokens,
            &[b, self.cfg.num_register_tokens, self.cfg.dim()],
        )?;
        let cls = Array::zeros::<f32>(&[b, 1, self.cfg.dim()])?.as_dtype(self.dtype)?;
        let hidden = concatenate_axis(&[hidden, registers, cls], 1)?;

        // Patch ids from the latent grid; every suffix token sits at the rotary origin.
        let ids = create_token_ids(t, h, w, Dtype::Float32)?;
        let suffix = Array::zeros::<f32>(&[1, self.cfg.num_suffix_tokens(), 3])?;
        let ids = concatenate_axis(&[ids, suffix], 1)?;
        let tables = self.rope.tables(&ids)?;

        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &self.rope, &tables)?;
        }

        let hidden = layer_norm(
            &hidden,
            Some(&self.norm_out_w),
            Some(&self.norm_out_b),
            self.cfg.norm_eps,
        )?;
        let out = linear(&hidden, &self.proj_out_w, &self.proj_out_b)?;
        // Suffix rows are projected and then discarded, matching upstream.
        let out = slice_axis(&out, 1, 0, num_patches)?;

        unpack_tokens(
            &out,
            self.cfg.out_channels,
            self.cfg.patch_size,
            self.cfg.patch_size_t,
            t,
            h,
            w,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_cover_the_published_decoder_tensor_count() {
        let cfg = MiniMaxH3VaeConfig::default();
        let names = ViT3dDecoder::names("decoder", &cfg);
        // 36 blocks × 16 + proj_in(2) + register_tokens(1) + norm_out(2) + proj_out(2) = 583,
        // exactly the `decoder.*` count in the published shard index.
        assert_eq!(names.len(), 583);
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate tensor name");
    }

    #[test]
    fn names_scale_with_depth() {
        let cfg = MiniMaxH3VaeConfig {
            num_layers: 2,
            ..Default::default()
        };
        assert_eq!(ViT3dDecoder::names("decoder", &cfg).len(), 2 * 16 + 7);
    }
}

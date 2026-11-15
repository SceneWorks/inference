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
//!
//! # The 2-D mode (MiniMax-H3-Image)
//!
//! `MiniMaxH3ImageViTDecoder2d` is the same decoder with three differences, selected by
//! [`DecoderLayout::Image2d`] rather than by a second copy of the layers:
//!
//! - the input is an NCHW latent and tokens are `(h, w)` row-major ([`pack_tokens_2d`]);
//! - the rotary carries 2 axes `(h, w)` — 32 rotated dims = 8 frequencies per axis at the shipped
//!   ratio 0.5 — and the suffix tokens again sit at the origin;
//! - `proj_out` emits a spatial-only `out_channels · patch_size²` patch ([`unpack_tokens_2d`]).
//!
//! [`ViT3dDecoder::forward`] (video) and [`ViT3dDecoder::forward_image`] share the token core, and
//! each refuses the other layout, so a video decoder can never run with 2 rotary axes.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result, Weights};

use crate::blocks::TransformerBlock;
use crate::config::{DecoderLayout, MiniMaxH3VaeConfig};
use crate::nn::{layer_norm, linear};
use crate::rope::{create_token_ids, create_token_ids_2d, Rope3d};
use crate::tensor::{pack_tokens, pack_tokens_2d, unpack_tokens, unpack_tokens_2d};

/// The transformer VAE decoder — the 3-D video layout or, with [`DecoderLayout::Image2d`], the
/// MiniMax-H3-Image 2-D layout.
#[derive(Debug, Clone)]
pub struct ViT3dDecoder {
    proj_in_w: Tensor,
    proj_in_b: Tensor,
    register_tokens: Tensor,
    blocks: Vec<TransformerBlock>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    rope: Rope3d,
    cfg: MiniMaxH3VaeConfig,
    dtype: DType,
}

impl ViT3dDecoder {
    /// Load from `decoder.`-prefixed tensors.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: DType,
    ) -> Result<Self> {
        cfg.validate()?;
        let proj_in_w = w
            .require(&format!("{prefix}.proj_in.weight"))?
            .to_dtype(dtype)?;
        // `proj_in` maps latent channels -> model dim; a mismatch here means the config and the
        // checkpoint disagree about `latent_channels` or the decoder width.
        let shape = proj_in_w.dims();
        if shape != [cfg.dim(), cfg.latent_channels] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decoder: {prefix}.proj_in.weight is {shape:?}, expected [{}, {}]",
                cfg.dim(),
                cfg.latent_channels
            )));
        }
        let proj_in_b = w
            .require(&format!("{prefix}.proj_in.bias"))?
            .to_dtype(dtype)?;
        let register_tokens = w
            .require(&format!("{prefix}.register_tokens"))?
            .to_dtype(dtype)?;
        if register_tokens.dims() != [1, cfg.num_register_tokens, cfg.dim()] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decoder: {prefix}.register_tokens is {:?}, expected [1, {}, {}]",
                register_tokens.dims(),
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
            .to_dtype(dtype)?;
        if proj_out_w.dims() != [cfg.patch_dim(), cfg.dim()] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decoder: {prefix}.proj_out.weight is {:?}, expected [{}, {}] \
                 (out_channels · decoder patch_size_t · patch_size²)",
                proj_out_w.dims(),
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
                .to_dtype(dtype)?,
            norm_out_b: w
                .require(&format!("{prefix}.norm_out.bias"))?
                .to_dtype(dtype)?,
            proj_out_w,
            proj_out_b: w
                .require(&format!("{prefix}.proj_out.bias"))?
                .to_dtype(dtype)?,
            rope: Rope3d::with_axes(
                cfg.rope_apply_dim(),
                cfg.rope_theta,
                cfg.decoder_layout.rope_axes(),
            )?,
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
    ///
    /// Video layout only; an image decoder takes [`Self::forward_image`].
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let s = latent.dims();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decoder: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        self.require_layout(DecoderLayout::Video3d)?;
        let (b, t, h, w) = (s[0], s[2], s[3], s[4]);
        self.check_channels(s[1])?;
        let latent = latent.to_dtype(self.dtype)?;
        let tokens = pack_tokens(&latent)?;
        // Patch ids from the latent grid; every suffix token sits at the rotary origin.
        let ids = create_token_ids(t, h, w, latent.device())?;
        let out = self.decode_tokens(&tokens, b, ids)?;
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

    /// Decode a `[B, latent_channels, H, W]` latent image into `[B, out_channels, H·16, W·16]` —
    /// `MiniMaxH3ImageViTDecoder2d.forward`, untiled. [`DecoderLayout::Image2d`] only.
    pub fn forward_image(&self, latent: &Tensor) -> Result<Tensor> {
        let s = latent.dims();
        if s.len() != 4 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 image decoder: expected [B, C, H, W], got {s:?}"
            )));
        }
        self.require_layout(DecoderLayout::Image2d)?;
        let (b, h, w) = (s[0], s[2], s[3]);
        self.check_channels(s[1])?;
        let latent = latent.to_dtype(self.dtype)?;
        let tokens = pack_tokens_2d(&latent)?;
        let ids = create_token_ids_2d(h, w, latent.device())?;
        let out = self.decode_tokens(&tokens, b, ids)?;
        unpack_tokens_2d(&out, self.cfg.out_channels, self.cfg.patch_size, h, w)
    }

    fn require_layout(&self, want: DecoderLayout) -> Result<()> {
        if self.cfg.decoder_layout != want {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decoder: this decoder is {:?}, the call needs {want:?}",
                self.cfg.decoder_layout
            )));
        }
        Ok(())
    }

    fn check_channels(&self, c: usize) -> Result<()> {
        if c != self.cfg.latent_channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decoder: expected {} latent channels, got {c}",
                self.cfg.latent_channels
            )));
        }
        Ok(())
    }

    /// The layout-independent core: `proj_in`, the register + zero-CLS suffix, the blocks under
    /// the rotary built from `patch_ids` (`[1, N, axes]`, suffix ids appended here at the origin),
    /// `norm_out` and `proj_out`, truncated back to the `N` patch tokens.
    fn decode_tokens(&self, tokens: &Tensor, b: usize, patch_ids: Tensor) -> Result<Tensor> {
        let num_patches = tokens.dim(1)?;
        let device = tokens.device();
        let hidden = __model_proof_output!(linear(tokens, &self.proj_in_w, &self.proj_in_b))?;

        // registers, then a ZERO cls token (`torch.zeros_like`, not a learned parameter).
        let registers = self
            .register_tokens
            .broadcast_as((b, self.cfg.num_register_tokens, self.cfg.dim()))?
            .contiguous()?;
        let cls = Tensor::zeros((b, 1, self.cfg.dim()), self.dtype, device)?;
        let hidden = Tensor::cat(&[&hidden, &registers, &cls], 1)?;

        let suffix = Tensor::zeros(
            (1, self.cfg.num_suffix_tokens(), self.rope.num_axes()),
            DType::F32,
            device,
        )?;
        let ids = Tensor::cat(&[&patch_ids, &suffix], 1)?;
        let tables = self.rope.tables(&ids)?;

        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &self.rope, &tables)?;
        }

        let hidden = __model_proof_output!(layer_norm(
            &hidden,
            &self.norm_out_w,
            &self.norm_out_b,
            self.cfg.norm_eps,
        ))?;
        let out = __model_proof_output!(linear(&hidden, &self.proj_out_w, &self.proj_out_b))?;
        // Suffix rows are projected and then discarded, matching upstream.
        Ok(out.narrow(1, 0, num_patches)?.contiguous()?)
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

    /// The published `MiniMax-H3-Image` `vae_diffusers` safetensors header carries 391 `decoder.*`
    /// tensors: 24 blocks × 16 + the same 7 top-level ones. The block layout is shared, so the
    /// image decoder needs no names of its own.
    #[test]
    fn names_cover_the_published_image_decoder_tensor_count() {
        let names = ViT3dDecoder::names("decoder", &MiniMaxH3VaeConfig::image());
        assert_eq!(names.len(), 391);
        assert!(names.contains(&"decoder.transformer_blocks.23.ff.net.0.proj.weight".to_string()));
        assert!(!names.iter().any(|n| n.contains("transformer_blocks.24.")));
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

//! The MiniMax-H3-**Image** VAE decode path (`AutoencoderKLMiniMaxH3Image`, `vae_diffusers/`).
//!
//! The candle twin of `mlx-gen-minimax-h3::image_vae`. The image VAE is H3's VAE with the 2-D
//! decoder: a 2-D `post_quant_conv` (`[24, 24, 1, 1]`) and a 24-layer / 1024-dim ViT decoder in
//! [`DecoderLayout::Image2d`]. Nothing here is a second copy of a layer: the decoder is
//! [`ViT3dDecoder`] in its 2-D mode, and spatial tiling is [`TilePlan`] + [`BoundedStitch`] — the
//! video VAE's split and stitch over a rank-4 `[B, C, H, W]` tensor (the stitch blends the last
//! two axes whatever the rank).
//!
//! # Decode
//!
//! The modular pipeline's decode step (`modular_pipelines/minimax_h3_image/decoders.py`) is
//! `vae.decode(latents · latents_std + latents_mean)`; [`MiniMaxH3ImageVae::decode`] is that whole
//! call, and `vae.decode` itself is tiled by default at 256 px / 64 px with the tiles blended in
//! **pixel** space.
//!
//! # Tiling is on by default and is an output, not a lever
//!
//! Upstream ships `use_tiling = True` and says the released images are the blended-tile ones, so
//! an untiled decode is a different result above one tile. The shipped geometry is therefore this
//! family's whole `BoundedDecode` domain ([`IMAGE_DECODE_TILE_EDGE`] / [`IMAGE_DECODE_OVERLAP`],
//! [`bounded_decode_parameters`]): tiling bounds each decoder pass to 256 + 5 tokens, which is
//! what lowers the decode peak at large canvases.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::MemoryParameterRanges;
use candle_gen::loader::sorted_safetensors;
use candle_gen::{CandleError, Result, Weights};

use crate::config::{DecoderLayout, MiniMaxH3VaeConfig};
use crate::decoder::ViT3dDecoder;
use crate::nn::linear;
use crate::spatial_tiling::{
    BoundedStitch, SpatialTiling, TilePlan, TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};

/// The published component directory the image VAE loads from.
pub const IMAGE_VAE_COMPONENT: &str = "vae_diffusers";

/// The family's bounded-decode tile edge, in pixels — the shipped `tile_sample_min_*` 256.
pub const IMAGE_DECODE_TILE_EDGE: u32 = TILE_SAMPLE_MIN_SIZE as u32;

/// The family's bounded-decode minimum overlap, in pixels — the shipped 64.
pub const IMAGE_DECODE_OVERLAP: u32 = TILE_SAMPLE_MIN_OVERLAP as u32;

/// The `BoundedDecode` parameter domain the MiniMax-H3-Image memory strategy declares: the single
/// shipped geometry (see the module docs for why it is a singleton).
pub fn bounded_decode_parameters() -> MemoryParameterRanges {
    MemoryParameterRanges {
        decode_tile_edges: vec![IMAGE_DECODE_TILE_EDGE],
        decode_overlaps: vec![IMAGE_DECODE_OVERLAP],
        ..Default::default()
    }
}

/// The decode half of the MiniMax-H3-Image VAE.
#[derive(Debug, Clone)]
pub struct MiniMaxH3ImageVae {
    post_quant_w: Tensor,
    post_quant_b: Tensor,
    decoder: ViT3dDecoder,
    latents_mean: Tensor,
    latents_std: Tensor,
    cfg: MiniMaxH3VaeConfig,
    dtype: DType,
    tiling: SpatialTiling,
}

impl MiniMaxH3ImageVae {
    /// Build from an already-populated [`Weights`] map. `cfg` must be a
    /// [`DecoderLayout::Image2d`] config ([`MiniMaxH3VaeConfig::image`] or
    /// [`MiniMaxH3VaeConfig::from_image_diffusers_json`]).
    pub fn from_weights(
        w: &Weights,
        cfg: &MiniMaxH3VaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if cfg.decoder_layout != DecoderLayout::Image2d {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae: needs the Image2d decoder layout, got {:?}",
                cfg.decoder_layout
            )));
        }
        cfg.validate()?;
        // `nn.Conv2d(24, 24, kernel_size=1)` — a pointwise channel map, applied as a linear.
        let raw = w.require("post_quant_conv.weight")?.to_dtype(dtype)?;
        let c = cfg.latent_channels;
        if raw.dims() != [c, c, 1, 1] {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae: post_quant_conv.weight is {:?}, expected [{c}, {c}, 1, 1]",
                raw.dims()
            )));
        }
        let stat = |v: &[f32]| -> Result<Tensor> {
            Ok(Tensor::from_vec(v.to_vec(), (1, c, 1, 1), device)?.to_dtype(dtype)?)
        };
        Ok(Self {
            post_quant_w: raw.reshape((c, c))?,
            post_quant_b: w.require("post_quant_conv.bias")?.to_dtype(dtype)?,
            decoder: ViT3dDecoder::from_weights(w, "decoder", cfg, dtype)?,
            latents_mean: stat(&cfg.latents_mean)?,
            latents_std: stat(&cfg.latents_std)?,
            cfg: cfg.clone(),
            dtype,
            tiling: SpatialTiling::default(),
        })
    }

    /// Load the decode half from a snapshot root — reads `vae_diffusers/config.json` and only the
    /// `decoder.*` / `post_quant_conv.*` tensors of its safetensors (the encoder never lands on the
    /// device).
    pub fn load(root: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = root.as_ref().join(IMAGE_VAE_COMPONENT);
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            CandleError::Msg(format!(
                "minimax-h3-image vae: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_image_diffusers_json(&text)?;
        let files = sorted_safetensors(&dir, "minimax-h3-image vae")?;
        let w =
            Weights::from_files_filtered(&files, device, dtype, &["decoder.", "post_quant_conv."])?;
        Self::from_weights(&w, &cfg, device, dtype)
    }

    /// Every tensor the decode half consumes: `post_quant_conv` plus the 391 `decoder.*`.
    pub fn tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        let mut v = vec![
            "post_quant_conv.weight".to_string(),
            "post_quant_conv.bias".to_string(),
        ];
        v.extend(ViT3dDecoder::names("decoder", cfg));
        v
    }

    /// The config in force.
    pub fn config(&self) -> &MiniMaxH3VaeConfig {
        &self.cfg
    }

    /// The tiling knobs in force.
    pub fn tiling(&self) -> SpatialTiling {
        self.tiling
    }

    /// The reference's `enable_tiling(...)`, overriding only what is given.
    pub fn enable_tiling(
        &mut self,
        tile_height: Option<usize>,
        tile_width: Option<usize>,
        overlap_height: Option<usize>,
        overlap_width: Option<usize>,
    ) {
        self.tiling
            .enable(tile_height, tile_width, overlap_height, overlap_width);
    }

    /// The reference's `disable_tiling()`. **This changes the output** above one tile.
    pub fn disable_tiling(&mut self) {
        self.tiling.disable();
    }

    /// Builder form of [`Self::enable_tiling`] / [`Self::disable_tiling`].
    pub fn with_tiling(mut self, tiling: SpatialTiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// Per-channel latent de-normalization: `z · latents_std + latents_mean`.
    pub fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let z = latents.to_dtype(self.dtype)?;
        Ok(z.broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?)
    }

    /// Per-channel latent normalization: `(z - latents_mean) / latents_std`.
    pub fn normalize(&self, latents: &Tensor) -> Result<Tensor> {
        let z = latents.to_dtype(self.dtype)?;
        Ok(z.broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)?)
    }

    /// Full decode of pipeline latents `[B, 24, H, W]`: de-normalize, then
    /// [`Self::decode_latent`]. Returns ImageNet-normalized RGB `[B, 3, 16·H, 16·W]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = self.denormalize(latents)?;
        self.decode_latent(&z)
    }

    /// The reference's `vae.decode(z)` on an already de-normalized latent: tiled at the
    /// configured geometry when tiling is on (the default), one pass otherwise.
    pub fn decode_latent(&self, z: &Tensor) -> Result<Tensor> {
        if !self.tiling.enabled {
            return self.decode_untiled(z);
        }
        self.decode_tiled(
            z,
            self.tiling.tile_height,
            self.tiling.tile_width,
            self.tiling.overlap_height,
            self.tiling.overlap_width,
        )
    }

    /// `_decode` with tiling on, at an explicit pixel geometry — the `BoundedDecode` entry point.
    ///
    /// Tiles are laid out in **pixel** space and mapped back onto the latent grid with each bound
    /// divided separately, decoded one at a time and streamed into a [`BoundedStitch`] that
    /// blends with the **undivided** pixel overlaps, because a decoded tile comes back in pixel
    /// space.
    pub fn decode_tiled(
        &self,
        z: &Tensor,
        tile_height: usize,
        tile_width: usize,
        overlap_height: usize,
        overlap_width: usize,
    ) -> Result<Tensor> {
        let s = z.dims();
        if s.len() != 4 {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae: expected [B, C, H, W], got {s:?}"
            )));
        }
        let ratio = self.cfg.patch_size;
        let rows = TilePlan::split(s[2] * ratio, tile_height, overlap_height, ratio)?;
        let cols = TilePlan::split(s[3] * ratio, tile_width, overlap_width, ratio)?;
        let mut stitch =
            BoundedStitch::new(rows.len(), cols.len(), &rows.overlaps, &cols.overlaps)?;
        for (i, &y) in rows.starts.iter().enumerate() {
            for (j, &x) in cols.starts.iter().enumerate() {
                let tile = z
                    .narrow(2, y / ratio, rows.lengths[i] / ratio)?
                    .narrow(3, x / ratio, cols.lengths[j] / ratio)?
                    .contiguous()?;
                stitch.push(self.decode_untiled(&tile)?)?;
            }
        }
        stitch.finish()
    }

    /// `post_quant_conv` then the 2-D decoder in ONE pass — the reference's `use_tiling = False`
    /// branch. Does not reproduce the released images above one tile.
    pub fn decode_untiled(&self, z: &Tensor) -> Result<Tensor> {
        let s = z.dims();
        if s.len() != 4 {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae: expected [B, C, H, W], got {s:?}"
            )));
        }
        let z = z.to_dtype(self.dtype)?;
        let nhwc = z.permute((0, 2, 3, 1))?.contiguous()?;
        let mapped = linear(&nhwc, &self.post_quant_w, &self.post_quant_b)?;
        self.decoder
            .forward_image(&mapped.permute((0, 3, 1, 2))?.contiguous()?)
    }
}

/// The **encode** half of the MiniMax-H3-Image VAE (`AutoencoderKLMiniMaxH3Image.encode`): H3's
/// causal-CNN encoder and `quant_conv` (the video VAE's encode half, the same 118 published tensor
/// names) run over ONE frame, tiled at the shipped 256 / 64 with the tiles blended in latent space,
/// and the last (only) latent frame kept. Float32, as upstream pins it (`_keep_in_fp32_modules`).
/// Separate from [`MiniMaxH3ImageVae`] because only an edit reads it, and the two are never
/// resident together.
#[derive(Debug, Clone)]
pub struct MiniMaxH3ImageVaeEncoder {
    half: crate::vae::EncodeHalf,
    latents_mean: Tensor,
    latents_std: Tensor,
    cfg: MiniMaxH3VaeConfig,
    tiling: SpatialTiling,
}

impl MiniMaxH3ImageVaeEncoder {
    /// Build from an already-populated [`Weights`] map holding `encoder.*` and `quant_conv.*`.
    pub fn from_weights(w: &Weights, cfg: &MiniMaxH3VaeConfig, device: &Device) -> Result<Self> {
        if cfg.decoder_layout != DecoderLayout::Image2d {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae encoder: needs the Image2d config, got {:?}",
                cfg.decoder_layout
            )));
        }
        cfg.validate()?;
        let c = cfg.latent_channels;
        let stat = |v: &[f32]| -> Result<Tensor> {
            Ok(Tensor::from_vec(v.to_vec(), (1, c, 1, 1), device)?)
        };
        Ok(Self {
            half: crate::vae::EncodeHalf::from_weights(w, cfg, DType::F32)?,
            latents_mean: stat(&cfg.latents_mean)?,
            latents_std: stat(&cfg.latents_std)?,
            cfg: cfg.clone(),
            tiling: SpatialTiling::default(),
        })
    }

    /// Load `vae_diffusers/`'s encode half — only its `encoder.*` / `quant_conv.*` tensors.
    pub fn load(root: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = root.as_ref().join(IMAGE_VAE_COMPONENT);
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            CandleError::Msg(format!(
                "minimax-h3-image vae encoder: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_image_diffusers_json(&text)?;
        let files = sorted_safetensors(&dir, "minimax-h3-image vae encoder")?;
        let w =
            Weights::from_files_filtered(&files, device, DType::F32, &["encoder.", "quant_conv."])?;
        Self::from_weights(&w, &cfg, device)
    }

    /// Every tensor the encode half reads: `quant_conv` plus the 116 `encoder.*`.
    pub fn tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        crate::vae::MiniMaxH3VideoVae::encode_tensor_names(cfg)
    }

    /// Override the tiling (the shipped 256 / 64 by default).
    pub fn with_tiling(mut self, tiling: SpatialTiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// `vae.encode(pixels)`: ImageNet-normalized `[1, 3, H, W]` pixels (`H`, `W` multiples of 16)
    /// to the posterior over `[1, 24, 1, H/16, W/16]`.
    pub fn encode(&self, pixels: &Tensor) -> Result<crate::vae_encoder::DiagonalGaussian> {
        let s = pixels.dims();
        let ratio = self.cfg.patch_size;
        if s.len() != 4
            || s[1] != self.cfg.in_channels
            || !s[2].is_multiple_of(ratio)
            || !s[3].is_multiple_of(ratio)
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae encode: expected [B, {}, H, W] with H and W multiples of \
                 {ratio}, got {s:?}",
                self.cfg.in_channels
            )));
        }
        // An image is a single frame of the causal video encoder, which maps to one latent frame.
        let x = pixels.to_dtype(DType::F32)?.unsqueeze(2)?;
        let moments = if self.tiling.enabled {
            self.half.encode_tiled(
                &x,
                ratio,
                self.tiling.tile_height,
                self.tiling.tile_width,
                self.tiling.overlap_height,
                self.tiling.overlap_width,
            )?
        } else {
            self.half.encode_untiled(&x)?
        };
        let t = moments.dims()[2];
        crate::vae_encoder::DiagonalGaussian::from_parameters(&moments.narrow(2, t - 1, 1)?)
    }

    /// One reference's conditioning latent, the released recipe (`MiniMaxH3ImageEditVaeEncoderStep`):
    /// a posterior sample `mean + std · noise` rounded to float16, cropped at the top left to whole
    /// `patch` cells, then normalized by the per-channel latent statistics. `noise` matches the
    /// posterior's `[1, 24, 1, h, w]`. Returns `[1, 24, h', w']`.
    pub fn reference_latent(
        &self,
        posterior: &crate::vae_encoder::DiagonalGaussian,
        noise: &Tensor,
        patch: (usize, usize),
    ) -> Result<Tensor> {
        let sampled = crate::conditioning::fp16_round_trip(&posterior.sample_with(noise)?)?;
        let sampled = sampled.squeeze(2)?;
        let (_, _, lh, lw) = sampled.dims4()?;
        let (h, w) = (lh / patch.0 * patch.0, lw / patch.1 * patch.1);
        if h == 0 || w == 0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image vae encode: a {lh}x{lw} latent holds no whole {patch:?} patch"
            )));
        }
        let cropped = sampled.narrow(2, 0, h)?.narrow(3, 0, w)?.contiguous()?;
        Ok(cropped
            .broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)?)
    }
}

/// `(rows, cols)` of the decode tile grid for a `height × width` **pixel** canvas at the shipped
/// geometry — the tile count a bounded decode runs, without a VAE.
pub fn decode_tile_grid(height: usize, width: usize, ratio: usize) -> Result<(usize, usize)> {
    let edge = IMAGE_DECODE_TILE_EDGE as usize;
    let overlap = IMAGE_DECODE_OVERLAP as usize;
    Ok((
        TilePlan::split(height, edge, overlap, ratio)?.len(),
        TilePlan::split(width, edge, overlap, ratio)?.len(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The encode half reads the 118 published `encoder.*` / `quant_conv.*` tensors.
    #[test]
    fn encode_tensor_names_are_the_published_118() {
        let names = MiniMaxH3ImageVaeEncoder::tensor_names(&MiniMaxH3VaeConfig::image());
        assert_eq!(names.len(), 118);
        assert!(names
            .iter()
            .all(|n| n.starts_with("encoder.") || n.starts_with("quant_conv.")));
    }

    #[test]
    fn decode_tensor_names_are_post_quant_plus_the_391_decoder_tensors() {
        let names = MiniMaxH3ImageVae::tensor_names(&MiniMaxH3VaeConfig::image());
        assert_eq!(names.len(), 393);
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
    }

    /// The BoundedDecode domain is the one shipped geometry, 256 px / 64 px.
    #[test]
    fn the_bounded_decode_domain_is_the_shipped_geometry() {
        let p = bounded_decode_parameters();
        assert_eq!(p.decode_tile_edges, vec![256]);
        assert_eq!(p.decode_overlaps, vec![64]);
        assert_eq!(
            MemoryParameterRanges {
                decode_tile_edges: Vec::new(),
                decode_overlaps: Vec::new(),
                ..p.clone()
            },
            MemoryParameterRanges::default(),
            "bounded decode owns only the decode parameters"
        );
    }

    /// Tile-grid arithmetic at the canvases the real-weight gates use. 16 MP (4096²) needs 21
    /// tiles per axis: `256·n - 64·(n-1) >= 4096` first holds at n = 21.
    #[test]
    fn decode_tile_grid_matches_split_tiles() {
        assert_eq!(decode_tile_grid(256, 256, 16).unwrap(), (1, 1));
        assert_eq!(decode_tile_grid(64, 64, 16).unwrap(), (1, 1));
        assert_eq!(decode_tile_grid(384, 640, 16).unwrap(), (2, 3));
        assert_eq!(decode_tile_grid(4096, 4096, 16).unwrap(), (21, 21));
        assert_eq!(
            TilePlan::split(384, 256, 64, 16).unwrap().overlaps,
            vec![128]
        );
        assert_eq!(
            TilePlan::split(640, 256, 64, 16).unwrap().overlaps,
            vec![64, 64]
        );
    }

    #[test]
    fn from_weights_refuses_the_video_layout() {
        let w = Weights::from_map(HashMap::new());
        let msg = MiniMaxH3ImageVae::from_weights(
            &w,
            &MiniMaxH3VaeConfig::default(),
            &Device::Cpu,
            DType::F32,
        )
        .expect_err("the video config is not the image VAE")
        .to_string();
        assert!(msg.contains("Image2d"), "{msg}");
    }

    #[test]
    fn load_errors_cleanly_on_a_missing_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let msg = MiniMaxH3ImageVae::load(tmp.path().join("missing"), &Device::Cpu, DType::F32)
            .expect_err("a missing snapshot cannot load")
            .to_string();
        assert!(msg.contains("vae_diffusers"), "{msg}");
    }
}

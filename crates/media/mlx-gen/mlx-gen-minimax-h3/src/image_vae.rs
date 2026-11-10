//! The MiniMax-H3-**Image** VAE decode path (`AutoencoderKLMiniMaxH3Image`, `vae_diffusers/`).
//!
//! The image VAE is H3's VAE with the 2-D decoder: the same 3-D CNN encoder run over one frame,
//! a 2-D `post_quant_conv` (`[24, 24, 1, 1]`), and a 24-layer / 1024-dim ViT decoder in
//! [`DecoderLayout::Image2d`]. Nothing here is a second copy of a layer: the decoder is
//! [`ViT3dDecoder`] in its 2-D mode, and spatial tiling is [`TilePlan`] + [`BoundedStitch`] — the
//! same split and stitch the video VAE uses, over a rank-4 `[B, C, H, W]` tensor instead of a
//! rank-5 one (the stitch blends the last two axes whatever the rank).
//!
//! # Decode
//!
//! The modular pipeline's decode step (`modular_pipelines/minimax_h3_image/decoders.py`) is
//! `vae.decode(latents · latents_std + latents_mean)`; [`MiniMaxH3ImageVae::decode`] is that whole
//! call. `vae.decode` then runs `post_quant_conv` + the decoder per tile and blends the tiles in
//! **pixel** space (`_decode` / `_stitch_tiles`).
//!
//! # Tiling is on by default and is an output, not a lever
//!
//! Upstream ships `use_tiling = True` at 256 px tiles / 64 px minimum overlap, and says the
//! released images are the blended-tile ones, so an untiled decode is a different result above one
//! tile. The same geometry is therefore this family's whole `BoundedDecode` domain
//! ([`IMAGE_DECODE_TILE_EDGE`] / [`IMAGE_DECODE_OVERLAP`], [`bounded_decode_parameters`]): tiling
//! bounds the decode's attention to 256 + 5 tokens per tile, which is what lowers the decode peak
//! at large canvases, and the geometry cannot move without changing the picture.

use std::path::Path;

use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::MemoryParameterRanges;
use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{DecoderLayout, MiniMaxH3VaeConfig};
use crate::decoder::ViT3dDecoder;
use crate::spatial_tiling::{
    BoundedStitch, SpatialTiling, TilePlan, TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};
use crate::tensor::slice_axis;

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
    post_quant_w: Array,
    post_quant_b: Array,
    decoder: ViT3dDecoder,
    latents_mean: Array,
    latents_std: Array,
    cfg: MiniMaxH3VaeConfig,
    dtype: Dtype,
    tiling: SpatialTiling,
}

impl MiniMaxH3ImageVae {
    /// Build from an already-populated [`Weights`] map. `cfg` must be a
    /// [`DecoderLayout::Image2d`] config ([`MiniMaxH3VaeConfig::image`] or
    /// [`MiniMaxH3VaeConfig::from_image_diffusers_json`]). The `encoder.*` / `quant_conv.*` tensors
    /// the published file also carries are not read.
    pub fn from_weights(w: &mut Weights, cfg: &MiniMaxH3VaeConfig, dtype: Dtype) -> Result<Self> {
        if cfg.decoder_layout != DecoderLayout::Image2d {
            return Err(Error::Msg(format!(
                "minimax-h3-image vae: needs the Image2d decoder layout, got {:?}",
                cfg.decoder_layout
            )));
        }
        cfg.validate()?;
        // `nn.Conv2d(24, 24, kernel_size=1)` — a pointwise channel map, applied as a linear.
        let raw = w.require("post_quant_conv.weight")?.as_dtype(dtype)?;
        let c = cfg.latent_channels;
        if raw.shape() != [c, c, 1, 1] {
            return Err(Error::Msg(format!(
                "minimax-h3-image vae: post_quant_conv.weight is {:?}, expected [{c}, {c}, 1, 1]",
                raw.shape()
            )));
        }
        Ok(Self {
            post_quant_w: raw.reshape(&[c, c])?,
            post_quant_b: w.require("post_quant_conv.bias")?.as_dtype(dtype)?,
            decoder: ViT3dDecoder::from_weights(w, "decoder", cfg, dtype)?,
            latents_mean: Array::from_slice(&cfg.latents_mean, &[1, c, 1, 1]).as_dtype(dtype)?,
            latents_std: Array::from_slice(&cfg.latents_std, &[1, c, 1, 1]).as_dtype(dtype)?,
            cfg: cfg.clone(),
            dtype,
            tiling: SpatialTiling::default(),
        })
    }

    /// Load from a snapshot root — reads `vae_diffusers/config.json` and its safetensors.
    pub fn load(root: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let dir = root.as_ref().join(IMAGE_VAE_COMPONENT);
        let config_path = dir.join("config.json");
        let text = mlx_gen::gen_core::__model_input_choose!(
            std::fs::read_to_string(&config_path),
            mlx_gen::gen_core::model_input_proof::read_configuration_to_string(&config_path)
        )
        .map_err(|e| {
            Error::Msg(format!(
                "minimax-h3-image vae: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_image_diffusers_json(&text)?;
        let mut w = Weights::from_dir(&dir)?;
        let vae = Self::from_weights(&mut w, &cfg, dtype)?;
        w.materialize_accessed()?;
        Ok(vae)
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
        tile_height: Option<i32>,
        tile_width: Option<i32>,
        overlap_height: Option<i32>,
        overlap_width: Option<i32>,
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
    pub fn denormalize(&self, latents: &Array) -> Result<Array> {
        let z = latents.as_dtype(self.dtype)?;
        Ok(add(&multiply(&z, &self.latents_std)?, &self.latents_mean)?)
    }

    /// Per-channel latent normalization: `(z - latents_mean) / latents_std`.
    pub fn normalize(&self, latents: &Array) -> Result<Array> {
        let z = latents.as_dtype(self.dtype)?;
        Ok(z.subtract(&self.latents_mean)?.divide(&self.latents_std)?)
    }

    /// Full decode of pipeline latents `[B, 24, H, W]`: de-normalize, then
    /// [`Self::decode_latent`]. Returns ImageNet-normalized RGB `[B, 3, 16·H, 16·W]`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let z = self.denormalize(latents)?;
        self.decode_latent(&z)
    }

    /// The reference's `vae.decode(z)` on an already de-normalized latent: tiled at the
    /// configured geometry when tiling is on (the default), one pass otherwise.
    pub fn decode_latent(&self, z: &Array) -> Result<Array> {
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
    /// divided separately (`start // ratio`, `start // ratio + length // ratio`), decoded one at a
    /// time and streamed into a [`BoundedStitch`] that blends with the **undivided** pixel
    /// overlaps, because a decoded tile comes back in pixel space. Each tile is forced before the
    /// next so at most one tile's decoder activations are live.
    pub fn decode_tiled(
        &self,
        z: &Array,
        tile_height: i32,
        tile_width: i32,
        overlap_height: i32,
        overlap_width: i32,
    ) -> Result<Array> {
        let s = z.shape();
        if s.len() != 4 {
            return Err(Error::Msg(format!(
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
                let (y0, x0) = (y / ratio, x / ratio);
                let tile = slice_axis(z, 2, y0, y0 + rows.lengths[i] / ratio)?;
                let tile = slice_axis(&tile, 3, x0, x0 + cols.lengths[j] / ratio)?;
                let decoded = self.decode_untiled(&tile)?;
                mlx_rs::transforms::eval([&decoded])?;
                stitch.push(decoded)?;
            }
        }
        stitch.finish()
    }

    /// `post_quant_conv` then the 2-D decoder in ONE pass — the reference's `use_tiling = False`
    /// branch. Does not reproduce the released images above one tile.
    pub fn decode_untiled(&self, z: &Array) -> Result<Array> {
        let s = z.shape();
        if s.len() != 4 {
            return Err(Error::Msg(format!(
                "minimax-h3-image vae: expected [B, C, H, W], got {s:?}"
            )));
        }
        let z = z.as_dtype(self.dtype)?;
        let nhwc = z.transpose_axes(&[0, 2, 3, 1])?;
        let mapped = linear(&nhwc, &self.post_quant_w, &self.post_quant_b)?;
        self.decoder
            .forward_image(&mapped.transpose_axes(&[0, 3, 1, 2])?)
    }
}

/// The **encode** half of the MiniMax-H3-Image VAE (`AutoencoderKLMiniMaxH3Image.encode`): H3's
/// causal-CNN encoder and `quant_conv` (the video VAE's encode half, the same 118 published tensor
/// names) run over ONE frame, tiled at the shipped 256 / 64 with the tiles blended in latent space,
/// and the last (only) latent frame kept. Pinned to float32 as upstream pins it
/// (`_keep_in_fp32_modules`). Separate from [`MiniMaxH3ImageVae`] because only an edit reads it,
/// and the two are never resident together.
#[derive(Debug, Clone)]
pub struct MiniMaxH3ImageVaeEncoder {
    half: crate::vae::EncodeHalf,
    latents_mean: Array,
    latents_std: Array,
    cfg: MiniMaxH3VaeConfig,
    tiling: SpatialTiling,
}

impl MiniMaxH3ImageVaeEncoder {
    /// Build from an already-populated [`Weights`] map holding `encoder.*` and `quant_conv.*`.
    pub fn from_weights(w: &mut Weights, cfg: &MiniMaxH3VaeConfig) -> Result<Self> {
        if cfg.decoder_layout != DecoderLayout::Image2d {
            return Err(Error::Msg(format!(
                "minimax-h3-image vae encoder: needs the Image2d config, got {:?}",
                cfg.decoder_layout
            )));
        }
        cfg.validate()?;
        let c = cfg.latent_channels;
        Ok(Self {
            half: crate::vae::EncodeHalf::from_weights(w, cfg, Dtype::Float32)?,
            latents_mean: Array::from_slice(&cfg.latents_mean, &[1, c, 1, 1]),
            latents_std: Array::from_slice(&cfg.latents_std, &[1, c, 1, 1]),
            cfg: cfg.clone(),
            tiling: SpatialTiling::default(),
        })
    }

    /// Load `vae_diffusers/`'s encode half from a snapshot root.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let dir = root.as_ref().join(IMAGE_VAE_COMPONENT);
        let config_path = dir.join("config.json");
        let text = mlx_gen::gen_core::__model_input_choose!(
            std::fs::read_to_string(&config_path),
            mlx_gen::gen_core::model_input_proof::read_configuration_to_string(&config_path)
        )
        .map_err(|e| {
            Error::Msg(format!(
                "minimax-h3-image vae encoder: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_image_diffusers_json(&text)?;
        let mut w = Weights::from_dir(&dir)?;
        let encoder = Self::from_weights(&mut w, &cfg)?;
        w.materialize_accessed()?;
        Ok(encoder)
    }

    /// Every tensor the encode half reads: `quant_conv` plus the 116 `encoder.*`.
    pub fn tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        crate::vae::EncodeHalf::names(cfg)
    }

    /// The tiling knobs in force (shared with the decode half upstream; the shipped 256 / 64).
    pub fn with_tiling(mut self, tiling: SpatialTiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// `vae.encode(pixels)`: ImageNet-normalized `[1, 3, H, W]` pixels (`H`, `W` multiples of 16)
    /// to the posterior over `[1, 24, H/16, W/16]`.
    pub fn encode(&self, pixels: &Array) -> Result<crate::vae_encoder::DiagonalGaussian> {
        let s = pixels.shape();
        let ratio = self.cfg.patch_size;
        if s.len() != 4 || s[1] != self.cfg.in_channels || s[2] % ratio != 0 || s[3] % ratio != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3-image vae encode: expected [B, {}, H, W] with H and W multiples of \
                 {ratio}, got {s:?}",
                self.cfg.in_channels
            )));
        }
        // An image is a single frame of the causal video encoder, which maps to one latent frame.
        let x = pixels.as_dtype(Dtype::Float32)?.expand_dims(2)?;
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
        let t = moments.shape()[2];
        crate::vae_encoder::DiagonalGaussian::from_parameters(&slice_axis(&moments, 2, t - 1, t)?)
    }

    /// One reference's conditioning latent, the released recipe (`MiniMaxH3ImageEditVaeEncoderStep`):
    /// a posterior sample `mean + std · noise` rounded to float16, cropped at the top left to whole
    /// `patch` cells, then normalized by the per-channel latent statistics. `noise` matches the
    /// posterior's `[1, 24, 1, h, w]`. Returns `[1, 24, h', w']`.
    pub fn reference_latent(
        &self,
        posterior: &crate::vae_encoder::DiagonalGaussian,
        noise: &Array,
        patch: (i32, i32),
    ) -> Result<Array> {
        let sampled = crate::conditioning::fp16_round_trip(&posterior.sample_with(noise)?)?;
        let s = sampled.shape().to_vec();
        let sampled = sampled.reshape(&[s[0], s[1], s[3], s[4]])?;
        let (h, w) = (s[3] / patch.0 * patch.0, s[4] / patch.1 * patch.1);
        if h == 0 || w == 0 {
            return Err(Error::Msg(format!(
                "minimax-h3-image vae encode: a {}x{} latent holds no whole {patch:?} patch",
                s[3], s[4]
            )));
        }
        let cropped = slice_axis(&slice_axis(&sampled, 2, 0, h)?, 3, 0, w)?;
        Ok(cropped
            .subtract(&self.latents_mean)?
            .divide(&self.latents_std)?)
    }
}

/// `(rows, cols)` of the decode tile grid for a `height × width` **pixel** canvas at the shipped
/// geometry — the tile count a bounded decode runs, without a VAE.
pub fn decode_tile_grid(height: i32, width: i32, ratio: i32) -> Result<(usize, usize)> {
    let edge = IMAGE_DECODE_TILE_EDGE as i32;
    let overlap = IMAGE_DECODE_OVERLAP as i32;
    Ok((
        TilePlan::split(height, edge, overlap, ratio)?.len(),
        TilePlan::split(width, edge, overlap, ratio)?.len(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encode half reads the 118 tensors the published `vae_diffusers/` carries under
    /// `encoder.*` / `quant_conv.*`, the same names as H3 video's `vae/`.
    #[test]
    fn encode_tensor_names_are_the_published_118() {
        let names = MiniMaxH3ImageVaeEncoder::tensor_names(&MiniMaxH3VaeConfig::image());
        assert_eq!(names.len(), 118);
        assert!(names
            .iter()
            .all(|n| n.starts_with("encoder.") || n.starts_with("quant_conv.")));
        assert!(names
            .iter()
            .any(|n| n == "encoder.down_blocks.5.resnets.0.conv_shortcut.weight"));
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
        let rows = TilePlan::split(384, 256, 64, 16).unwrap();
        // 2 tiles cover 448 px; the 64 px slack widens the one overlap to 128.
        assert_eq!(rows.overlaps, vec![128]);
        let cols = TilePlan::split(640, 256, 64, 16).unwrap();
        assert_eq!(cols.overlaps, vec![64, 64]);
    }

    #[test]
    fn from_weights_refuses_the_video_layout() {
        let mut w = Weights::empty();
        let msg =
            MiniMaxH3ImageVae::from_weights(&mut w, &MiniMaxH3VaeConfig::default(), Dtype::Float32)
                .expect_err("the video config is not the image VAE")
                .to_string();
        assert!(msg.contains("Image2d"), "{msg}");
    }

    #[test]
    fn load_errors_cleanly_on_a_missing_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let msg = MiniMaxH3ImageVae::load(tmp.path().join("missing"), Dtype::Float32)
            .expect_err("a missing snapshot cannot load")
            .to_string();
        assert!(msg.contains("vae_diffusers"), "{msg}");
    }
}

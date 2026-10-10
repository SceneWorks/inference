//! Launch-bound guard for decodes through an **external** diffusers `AutoencoderKL` decoder
//! (`candle_transformers::models::{stable_diffusion, z_image}::vae::AutoEncoderKL`) — sc-24114.
//!
//! Two candle CUDA kernels index a launch with a 32-bit count, and past it a decode is silently wrong
//! (or faults) rather than refused:
//!
//!  - the non-cuDNN conv2d **im2col** fill launches with `LaunchConfig::for_num_elems(dst_el as u32)`,
//!    so a conv whose im2col buffer (`H·W·C_in·k²`) exceeds [`IM2COL_U32_LIMIT`] fills only its first
//!    `dst_el mod 2^32` entries — the band-corrupted render;
//!  - the **softmax** kernel indexes `row·ncols + col` as an `int`, so an attention whose score matrix
//!    exceeds [`SOFTMAX_I32_LIMIT`] elements wraps negative (`CUDA_ERROR_ILLEGAL_ADDRESS`).
//!
//! In-tree VAEs route their convs through a row-chunked conv instead; these decoders are external and
//! their layers are private, so the only in-tree bound is spatial: decode the latent as overlapping
//! tiles, each small enough that its widest conv and its mid-block attention stay under both limits,
//! and blend with the shared trapezoidal driver ([`crate::vae_tiling::decode_tiled`]).
//!
//! For the `[128, 256, 512, 512]` decoders these models ship, the widest full-resolution conv is the
//! last upsampler's 256-channel 3×3 — `2048²·256·9 = 9.7e9` — and the single-head mid-block attention
//! at a 256² latent is `65536² = 4.3e9` scores, so a 2048² decode crosses both; 1024² (`2.4e9` /
//! `2.7e8`) crosses neither and stays the byte-identical single pass. The guard fires only on a
//! non-CPU device (candle's CPU kernels index with `usize`) and only past a bound, where the
//! alternative is a corrupt image: tiles are decoded with per-tile group-norm statistics, the standard
//! diffusers tiled-decode trade-off.

use candle_core::{DType, Device, Result, Tensor};
use gen_core::tiling::{TilingConfig, VaeTiling};

/// Largest im2col buffer (elements) candle's CUDA conv2d fills correctly — its launch count is a `u32`.
pub const IM2COL_U32_LIMIT: u64 = u32::MAX as u64;

/// Largest attention score matrix (elements) candle's CUDA softmax indexes correctly — an `int` index.
pub const SOFTMAX_I32_LIMIT: u64 = i32::MAX as u64;

/// Output-pixel edge the bounded path starts from (halved until a tile fits both limits).
const MAX_TILE_PX: usize = 1024;

/// Blend overlap between adjacent tiles, in output pixels (16 latent rows at ×8).
const TILE_OVERLAP_PX: usize = 128;

/// im2col buffer size (elements) of a `k×k` conv2d over an `h × w` input with `in_ch` channels at
/// stride 1 / same padding — the count candle's CUDA conv launches with.
pub fn conv2d_im2col_elems(h: usize, w: usize, in_ch: usize, k: usize) -> u64 {
    h as u64 * w as u64 * in_ch as u64 * (k * k) as u64
}

/// Geometry of a diffusers `AutoencoderKL` decoder: `conv_in` from the latent, a single-head
/// attention mid block at latent resolution, then one up block per `block_out_channels` entry
/// (reversed), each but the last ending in a ×2 upsample + 3×3 conv.
#[derive(Clone, Copy, Debug)]
pub struct KlDecoderShape<'a> {
    pub latent_channels: usize,
    pub block_out_channels: &'a [usize],
}

impl KlDecoderShape<'_> {
    /// Output pixels per latent pixel (`2^(stages − 1)`).
    pub fn spatial_scale(&self) -> usize {
        1 << self.block_out_channels.len().saturating_sub(1)
    }

    /// Largest im2col buffer any decoder conv launches for a `latent_h × latent_w` latent.
    pub fn max_conv_im2col_elems(&self, latent_h: usize, latent_w: usize) -> u64 {
        let rev: Vec<usize> = self.block_out_channels.iter().rev().copied().collect();
        let Some(&mid) = rev.first() else {
            return 0;
        };
        let at = |up: usize, ch: usize| conv2d_im2col_elems(latent_h * up, latent_w * up, ch, 3);
        let mut max = at(1, self.latent_channels).max(at(1, mid)); // conv_in, mid-block resnets
        for (i, &out) in rev.iter().enumerate() {
            let up = 1 << i;
            let first_in = if i == 0 { mid } else { rev[i - 1] };
            max = max.max(at(up, first_in)).max(at(up, out)); // resnets
            if i + 1 < rev.len() {
                max = max.max(at(up * 2, out)); // upsampler conv, after the ×2 nearest upsample
            }
        }
        max
    }

    /// Score-matrix size of the mid block's single-head self-attention over the latent tokens.
    pub fn mid_attention_scores(&self, latent_h: usize, latent_w: usize) -> u64 {
        let tokens = latent_h as u64 * latent_w as u64;
        tokens * tokens
    }

    /// Whether a single-pass decode of a `latent_h × latent_w` latent crosses either CUDA launch
    /// bound — the selector for the bounded tiled path.
    pub fn exceeds_launch_bounds(&self, latent_h: usize, latent_w: usize) -> bool {
        self.max_conv_im2col_elems(latent_h, latent_w) > IM2COL_U32_LIMIT
            || self.mid_attention_scores(latent_h, latent_w) > SOFTMAX_I32_LIMIT
    }

    /// Tile policy for a `latent_h × latent_w` decode on `device`: `None` keeps the single pass
    /// (CPU, or under both bounds); otherwise the spatial tiling whose tiles are themselves under both.
    pub fn bounded_tiling(
        &self,
        device: &Device,
        latent_h: usize,
        latent_w: usize,
    ) -> Option<TilingConfig> {
        if device.is_cpu() || !self.exceeds_launch_bounds(latent_h, latent_w) {
            return None;
        }
        Some(TilingConfig::spatial_only(
            self.tile_edge_px() as i32,
            TILE_OVERLAP_PX as i32,
        ))
    }

    /// Largest output-pixel tile edge (≤ 1024, halving) whose square decode is under both bounds.
    pub fn tile_edge_px(&self) -> usize {
        let scale = self.spatial_scale();
        let mut edge = MAX_TILE_PX;
        while edge > 2 * TILE_OVERLAP_PX && self.exceeds_launch_bounds(edge / scale, edge / scale) {
            edge /= 2;
        }
        edge
    }

    fn tiling_geometry(&self) -> VaeTiling {
        VaeTiling {
            spatial_scale: self.spatial_scale() as i32,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: self.block_out_channels.first().copied().unwrap_or(0) as i32,
        }
    }
}

/// Decode a `[B, C, h, w]` latent through `decode`, bounded by [`KlDecoderShape::bounded_tiling`]:
/// the unchanged single `decode(latent)` call when no bound is crossed, otherwise overlapping tiles
/// blended in f32 (the tiled result is f32; the single pass keeps `decode`'s dtype).
pub fn bounded_kl_decode<F>(
    shape: &KlDecoderShape<'_>,
    latent: &Tensor,
    decode: F,
) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let (_, _, h, w) = latent.dims4()?;
    let tiling = shape.bounded_tiling(latent.device(), h, w);
    decode_with_tiling(shape, latent, tiling.as_ref(), decode)
}

/// [`bounded_kl_decode`] with an explicit policy (`None` = single pass) — split out so the tile path
/// is testable on the CPU, where the guard never fires.
pub fn decode_with_tiling<F>(
    shape: &KlDecoderShape<'_>,
    latent: &Tensor,
    tiling: Option<&TilingConfig>,
    mut decode: F,
) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let Some(tiling) = tiling else {
        return decode(latent);
    };
    let decoded = crate::vae_tiling::decode_tiled(
        shape.tiling_geometry(),
        "external AutoencoderKL bounded decode",
        &latent.unsqueeze(2)?,
        tiling,
        |tile: &Tensor| {
            decode(&tile.squeeze(2)?)?
                .to_dtype(DType::F32)?
                .unsqueeze(2)
        },
    )?;
    decoded.squeeze(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SD_BLOCKS: [usize; 4] = [128, 256, 512, 512];

    fn sd(latent_channels: usize) -> KlDecoderShape<'static> {
        KlDecoderShape {
            latent_channels,
            block_out_channels: &SD_BLOCKS,
        }
    }

    #[test]
    fn widest_conv_is_the_last_upsamplers_256_channels_at_full_resolution() {
        // 2048² output = 256² latent: 2048²·256·9.
        assert_eq!(
            sd(16).max_conv_im2col_elems(256, 256),
            conv2d_im2col_elems(2048, 2048, 256, 3)
        );
        assert_eq!(conv2d_im2col_elems(2048, 2048, 256, 3), 9_663_676_416);
    }

    #[test]
    fn selector_bounds_2048_and_keeps_1024_single_pass() {
        for shape in [sd(4), sd(16)] {
            assert_eq!(shape.spatial_scale(), 8);
            assert!(shape.exceeds_launch_bounds(256, 256), "2048² must bound");
            assert!(
                !shape.exceeds_launch_bounds(128, 128),
                "1024² stays single pass"
            );
            // The forced tile is itself under both bounds.
            let edge = shape.tile_edge_px();
            assert_eq!(edge, 1024);
            assert!(!shape.exceeds_launch_bounds(edge / 8, edge / 8));
            // CPU never tiles; the policy is only produced past a bound.
            assert!(shape.bounded_tiling(&Device::Cpu, 256, 256).is_none());
        }
    }

    #[test]
    fn every_planned_tile_is_under_both_bounds() {
        let shape = sd(16);
        let cfg = TilingConfig::spatial_only(shape.tile_edge_px() as i32, TILE_OVERLAP_PX as i32);
        let plan = cfg.plan(shape.tiling_geometry(), 1, 256, 320);
        assert!(plan.h.len() > 1 && plan.w.len() > 1);
        for hh in &plan.h {
            for ww in &plan.w {
                let (th, tw) = ((hh.end - hh.start) as usize, (ww.end - ww.start) as usize);
                assert!(!shape.exceeds_launch_bounds(th, tw), "tile {th}×{tw}");
            }
        }
        assert_eq!((plan.out_h, plan.out_w), (2048, 2560));
    }

    /// A pointwise decoder (nearest ×8 upsample + affine) has no spatial context, so the tiled blend
    /// must reproduce the single pass exactly up to f32 blend rounding, and without a policy the
    /// helper is the untouched single call (same dtype, same values).
    #[test]
    fn tiled_decode_matches_single_pass_for_a_pointwise_decoder() {
        let dev = Device::Cpu;
        let shape = sd(4);
        let (h, w) = (40usize, 56usize);
        let values: Vec<f32> = (0..4 * h * w).map(|i| (i as f32 * 0.37).sin()).collect();
        let latent = Tensor::from_vec(values, (1, 4, h, w), &dev).unwrap();
        let decode = |t: &Tensor| -> Result<Tensor> {
            let (b, _, th, tw) = t.dims4()?;
            let rgb = t.narrow(1, 0, 3)?.affine(0.5, 0.1)?;
            rgb.upsample_nearest2d(th * 8, tw * 8)?
                .reshape((b, 3, th * 8, tw * 8))
        };
        let single = decode(&latent).unwrap();
        let untouched = decode_with_tiling(&shape, &latent, None, decode).unwrap();
        assert_eq!(untouched.dtype(), single.dtype());
        assert_eq!(
            untouched.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            single.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let cfg = TilingConfig::spatial_only(128, 32); // 16-latent tiles, 4-latent overlap
        let tiled = decode_with_tiling(&shape, &latent, Some(&cfg), decode).unwrap();
        assert_eq!(tiled.dims(), single.dims());
        let diff = (tiled - &single)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "tiled vs single max abs diff {diff}");
    }
}

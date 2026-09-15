//! LTX-2.3 **video VAE** (`CausalVideoAutoencoder`, latent 128-ch, patch 4, 8× temporal /
//! 32× spatial) — port of mlx-gen-ltx `vae.rs`. Decoder-only construction serves inference;
//! [`LtxVideoVae::new_with_encoder`] additionally loads the causal training encoder.
//!
//! Decode: denormalize `latent·std + mean` → `conv_in 128→1024` → 9 up_blocks (`Res` groups +
//! `DepthToSpace` upsamplers) → pixel-norm (eps 1e-8) → SiLU → `conv_out 128→48` → unpatchify(×4).
//! All convs are non-causal (frame-replication temporal pad). pixel_norm = `x/√(mean(x² over C)+eps)`
//! (no √C, no γ). Runs **f32**.
//!
//! Block execution order (the config `decoder_blocks` list is encoder-order; the decoder reverses
//! it): `Res(2), Up(2,2,2), Res(2), Up(2,2,2), Res(4), Up(2,1,1), Res(6), Up(1,2,2), Res(4)`. Each
//! `Up` with temporal stride 2 doubles then drops the first frame, so latent T=7 → 49 pixel frames;
//! spatial 15 → 480 px (×2×2×2 then unpatchify ×4).

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::{
    TemporalOverlapPolicy, TileCandidates, TilingConfig, VaeTiling,
};
use candle_gen::vae_tiling;

use crate::config::{LATENT_CHANNELS, SPATIAL_SCALE, TEMPORAL_SCALE};
use crate::conv3d::CausalConv3d;

const DEC_NORM_EPS: f64 = 1e-8;
const ENC_NORM_EPS: f64 = 1e-6;

/// `x / sqrt(mean(x² over C, keepdims) + eps)` — LTX PixelNorm (channel axis = 1, no √C, no γ).
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let sumsq = x.sqr()?.sum_keepdim(1)?;
    let mean = (sumsq / c as f64)?;
    let denom = (mean + DEC_NORM_EPS)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Decoder residual block (`ResnetBlock3DSimple`): pixel-norm → SiLU → conv → pixel-norm → SiLU →
/// conv → residual add. Channels constant (no shortcut).
struct DecResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl DecResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::load(vb.clone(), &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::load(vb, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(x)?)?;
        let h = self.conv1.forward(&h, false)?;
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(&h)?)?;
        let h = self.conv2.forward(&h, false)?;
        h + x
    }
}

/// `DepthToSpaceUpsample` (residual=false): conv → depth-to-space → (st>1) drop first temporal frame.
struct DepthToSpace {
    conv: CausalConv3d,
    st: usize,
    sh: usize,
    sw: usize,
}

impl DepthToSpace {
    fn load(vb: VarBuilder, prefix: &str, stride: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::load(vb, &format!("{prefix}.conv.conv"))?,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
        })
    }

    /// `(B, C·st·sh·sw, D, H, W) -> (B, C, D·st, H·sh, W·sw)`.
    fn depth_to_space(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_packed, d, h, w) = x.dims5()?;
        let (st, sh, sw) = (self.st, self.sh, self.sw);
        let c = c_packed / (st * sh * sw);
        let x = x.reshape([b, c, st, sh, sw, d, h, w].as_slice())?;
        // transpose to (B, C, D, st, H, sh, W, sw) = axes [0,1,5,2,6,3,7,4].
        let x = x.permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?;
        x.reshape((b, c, d * st, h * sh, w * sw))?.contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x, false)?;
        let x = self.depth_to_space(&x)?;
        if self.st > 1 {
            let t = x.dim(2)?;
            x.narrow(2, 1, t - 1)
        } else {
            Ok(x)
        }
    }
}

enum UpLayer {
    Res(Vec<DecResBlock>),
    Up(DepthToSpace),
}

/// One decoder block in execution order: a res group of `n` blocks, or an upsampler with `stride`.
enum DBlock {
    Res(usize),
    Up((usize, usize, usize)),
}

/// The fixed LTX-2.3 decoder block order (config `decoder_blocks` already reversed to execution order).
const DECODER_BLOCKS: [DBlock; 9] = [
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(4),
    DBlock::Up((2, 1, 1)),
    DBlock::Res(6),
    DBlock::Up((1, 2, 2)),
    DBlock::Res(4),
];

/// `(B, C·p², F, H, W) -> (B, C, F, H·p, W·p)` (spatial-only unpatchify, patch_size_t = 1).
pub(crate) fn unpatchify(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c_packed, f, h, w) = x.dims5()?;
    let c = c_packed / (p * p);
    // (B, C, 1, p, p, F, H, W) -> transpose (0,1,5,2,6,4,7,3) -> (B, C, F, H·p, W·p).
    let x = x.reshape([b, c, 1, p, p, f, h, w].as_slice())?;
    let x = x.permute([0usize, 1, 5, 2, 6, 4, 7, 3].as_slice())?;
    x.reshape((b, c, f, h * p, w * p))?.contiguous()
}

/// Spatial-only patchify (`patch_size_t = 1`) used at the encoder input.
pub(crate) fn patchify(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, f, h, w) = x.dims5()?;
    if h % p != 0 || w % p != 0 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx vae encode: spatial dimensions {h}x{w} must be divisible by patch size {p}"
        )));
    }
    let (nh, nw) = (h / p, w / p);
    let x = x.reshape([b, c, f, 1, nh, p, nw, p].as_slice())?;
    let x = x.permute([0usize, 1, 3, 7, 5, 2, 4, 6].as_slice())?;
    x.reshape((b, c * p * p, f, nh, nw))?.contiguous()
}

#[cfg(test)]
mod encoder_unit_tests {
    use super::*;
    use candle_gen::candle_core::{Device, Tensor};

    #[test]
    fn patchify_unpatchify_round_trip() -> Result<()> {
        let data: Vec<f32> = (0..32).map(|value| value as f32).collect();
        let video = Tensor::from_vec(data, (1, 2, 1, 4, 4), &Device::Cpu)?;
        let packed = patchify(&video, 2)?;
        assert_eq!(packed.dims(), &[1, 8, 1, 2, 2]);
        let restored = unpatchify(&packed, 2)?;
        assert_eq!(restored.dims(), video.dims());
        assert_eq!(
            restored.flatten_all()?.to_vec1::<f32>()?,
            video.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn encoder_block_schedule_matches_ltx_scales() {
        let mut temporal = 1;
        let mut spatial = 4; // patchify
        for block in &ENCODER_BLOCKS {
            if let EBlock::Down((t, h, w)) = block {
                temporal *= t;
                assert_eq!(h, w);
                spatial *= h;
            }
        }
        assert_eq!(temporal, TEMPORAL_SCALE);
        assert_eq!(spatial, SPATIAL_SCALE);
    }
}

fn pixel_norm_eps(x: &Tensor, eps: f64) -> Result<Tensor> {
    let c = x.dim(1)?;
    let denom = ((x.sqr()?.sum_keepdim(1)? / c as f64)? + eps)?.sqrt()?;
    x.broadcast_div(&denom)
}

struct EncResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl EncResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::load(vb.clone(), &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::load(vb, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm_eps(x, ENC_NORM_EPS)?)?;
        let h = self.conv1.forward(&h, true)?;
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm_eps(&h, ENC_NORM_EPS)?)?;
        let h = self.conv2.forward(&h, true)?;
        h + x
    }
}

struct SpaceToDepth {
    conv: CausalConv3d,
    stride: (usize, usize, usize),
    out_channels: usize,
    group_size: usize,
}

impl SpaceToDepth {
    fn load(vb: VarBuilder, prefix: &str, stride: (usize, usize, usize)) -> Result<Self> {
        let conv = CausalConv3d::load(vb, &format!("{prefix}.conv.conv"))?;
        let (conv_out, in_channels) = conv.channels()?;
        let multiplier = stride.0 * stride.1 * stride.2;
        let out_channels = conv_out.checked_mul(multiplier).ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(
                "ltx vae encoder: downsample channel count overflow".into(),
            )
        })?;
        let packed_input = in_channels.checked_mul(multiplier).ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(
                "ltx vae encoder: downsample input channel count overflow".into(),
            )
        })?;
        if out_channels == 0 || packed_input % out_channels != 0 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx vae encoder: invalid downsample channels in={in_channels}, conv_out={conv_out}, \
                 stride={stride:?}"
            )));
        }
        Ok(Self {
            conv,
            stride,
            out_channels,
            group_size: packed_input / out_channels,
        })
    }

    fn space_to_depth(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, d, h, w) = x.dims5()?;
        let (st, sh, sw) = self.stride;
        let x = x.reshape([b, c, d / st, st, h / sh, sh, w / sw, sw].as_slice())?;
        let x = x.permute([0usize, 1, 3, 5, 7, 2, 4, 6].as_slice())?;
        x.reshape((b, c * st * sh * sw, d / st, h / sh, w / sw))?
            .contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (st, sh, sw) = self.stride;
        let mut x = if st == 2 {
            Tensor::cat(&[&x.narrow(2, 0, 1)?, x], 2)?
        } else {
            x.clone()
        };
        let (_b, _c, d, h, w) = x.dims5()?;
        let pad_d = (st - d % st) % st;
        let pad_h = (sh - h % sh) % sh;
        let pad_w = (sw - w % sw) % sw;
        if pad_d > 0 || pad_h > 0 || pad_w > 0 {
            x = x
                .pad_with_zeros(2, 0, pad_d)?
                .pad_with_zeros(3, 0, pad_h)?
                .pad_with_zeros(4, 0, pad_w)?;
        }

        let skip = self.space_to_depth(&x)?;
        let (b, _c, d, h, w) = skip.dims5()?;
        let skip = skip
            .reshape((b, self.out_channels, self.group_size, d, h, w))?
            .mean(2)?;
        let conv = self.space_to_depth(&self.conv.forward(&x, true)?)?;
        if conv.dims() == skip.dims() {
            conv + skip
        } else {
            Ok(conv)
        }
    }
}

enum DownLayer {
    Res(Vec<EncResBlock>),
    Down(SpaceToDepth),
}

enum EBlock {
    Res(usize),
    Down((usize, usize, usize)),
}

const ENCODER_BLOCKS: [EBlock; 9] = [
    EBlock::Res(4),
    EBlock::Down((1, 2, 2)),
    EBlock::Res(6),
    EBlock::Down((2, 1, 1)),
    EBlock::Res(4),
    EBlock::Down((2, 2, 2)),
    EBlock::Res(2),
    EBlock::Down((2, 2, 2)),
    EBlock::Res(2),
];

struct VideoEncoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownLayer>,
    conv_out: CausalConv3d,
    mean: Tensor,
    std: Tensor,
    patch_size: usize,
    latent_channels: usize,
}

impl VideoEncoder {
    fn load(
        vb: VarBuilder,
        stats: VarBuilder,
        latent_channels: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let mut down_blocks = Vec::with_capacity(ENCODER_BLOCKS.len());
        for (idx, block) in ENCODER_BLOCKS.iter().enumerate() {
            let prefix = format!("down_blocks.{idx}");
            down_blocks.push(match block {
                EBlock::Res(n) => {
                    let mut blocks = Vec::with_capacity(*n);
                    for j in 0..*n {
                        blocks.push(EncResBlock::load(
                            vb.clone(),
                            &format!("{prefix}.res_blocks.{j}"),
                        )?);
                    }
                    DownLayer::Res(blocks)
                }
                EBlock::Down(stride) => {
                    DownLayer::Down(SpaceToDepth::load(vb.clone(), &prefix, *stride)?)
                }
            });
        }
        Ok(Self {
            conv_in: CausalConv3d::load(vb.clone(), "conv_in.conv")?,
            down_blocks,
            conv_out: CausalConv3d::load(vb, "conv_out.conv")?,
            mean: stats
                .get_unchecked("mean-of-means")?
                .reshape((1, latent_channels, 1, 1, 1))?,
            std: stats
                .get_unchecked("std-of-means")?
                .reshape((1, latent_channels, 1, 1, 1))?,
            patch_size,
            latent_channels,
        })
    }

    fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let mut x = self
            .conv_in
            .forward(&patchify(video, self.patch_size)?, true)?;
        for layer in &self.down_blocks {
            x = match layer {
                DownLayer::Res(blocks) => {
                    let mut h = x;
                    for block in blocks {
                        h = block.forward(&h)?;
                    }
                    h
                }
                DownLayer::Down(down) => down.forward(&x)?,
            };
        }
        let x = candle_gen::candle_nn::ops::silu(&pixel_norm_eps(&x, ENC_NORM_EPS)?)?;
        let x = self.conv_out.forward(&x, true)?;
        let means = x.narrow(1, 0, self.latent_channels)?;
        means.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }
}

pub struct LtxVideoVae {
    conv_in: Option<CausalConv3d>,
    up_blocks: Vec<UpLayer>,
    conv_out: Option<CausalConv3d>,
    mean: Tensor, // [1, 128, 1, 1, 1]
    std: Tensor,  // [1, 128, 1, 1, 1]
    patch_size: usize,
    encoder: Option<VideoEncoder>,
}

impl LtxVideoVae {
    /// Geometry owned by the concrete decoder and consumed by its planner and tiled driver.
    pub const VAE_TILING: VaeTiling = VaeTiling::LTX;

    /// Convert DiT-normalized latents to the learned-upscaler's VAE latent
    /// domain. The stats are loaded from the same checkpoint as decode.
    pub(crate) fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents.broadcast_mul(&self.std)?.broadcast_add(&self.mean)
    }

    /// Convert learned-upscaler output back into the DiT-normalized domain.
    pub(crate) fn normalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    /// Build a decoder-only VAE from a VarBuilder rooted at the `vae.` prefix.
    pub fn new(vb: VarBuilder, latent_channels: usize, patch_size: usize) -> Result<Self> {
        Self::build(vb, None, latent_channels, patch_size)
    }

    /// Build the decoder and causal encoder. Pass the same `vae`-rooted builder twice for a unified
    /// checkpoint, or separate remapped builders for split decoder/encoder files.
    pub fn new_with_encoder(
        decoder_vb: VarBuilder,
        encoder_vb: VarBuilder,
        latent_channels: usize,
        patch_size: usize,
    ) -> Result<Self> {
        Self::build(decoder_vb, Some(encoder_vb), latent_channels, patch_size)
    }

    /// Build only the causal encoder and latent-normalization surface. DiffVAE owns final decode,
    /// so forcing its route to materialize a convolutional decoder is both unnecessary and wrong
    /// for converted tiers that deliberately ship no `vae_decoder.safetensors`.
    pub fn new_encoder_only(
        encoder_vb: VarBuilder,
        latent_channels: usize,
        patch_size: usize,
    ) -> Result<Self> {
        Self::build_without_decoder(encoder_vb, true, latent_channels, patch_size)
    }

    /// Build only latent normalization. Pure T2V on the DiffVAE route needs the statistics for the
    /// learned upsampler hand-off but never loads or calls a convolutional encoder/decoder.
    pub fn new_statistics_only(
        stats_vb: VarBuilder,
        latent_channels: usize,
        patch_size: usize,
    ) -> Result<Self> {
        Self::build_without_decoder(stats_vb, false, latent_channels, patch_size)
    }

    fn build_without_decoder(
        vb: VarBuilder,
        with_encoder: bool,
        latent_channels: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let stats = vb.pp("per_channel_statistics");
        let mean = stats
            .get_unchecked("mean-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let std = stats
            .get_unchecked("std-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let encoder = with_encoder
            .then(|| {
                VideoEncoder::load(
                    vb.pp("encoder"),
                    vb.pp("per_channel_statistics"),
                    latent_channels,
                    patch_size,
                )
            })
            .transpose()?;
        Ok(Self {
            conv_in: None,
            up_blocks: Vec::new(),
            conv_out: None,
            mean,
            std,
            patch_size,
            encoder,
        })
    }

    fn build(
        vb: VarBuilder,
        encoder_vb: Option<VarBuilder>,
        latent_channels: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let dec = vb.pp("decoder");
        let mut up_blocks = Vec::with_capacity(DECODER_BLOCKS.len());
        for (idx, block) in DECODER_BLOCKS.iter().enumerate() {
            let prefix = format!("up_blocks.{idx}");
            up_blocks.push(match block {
                DBlock::Res(n) => {
                    let mut blocks = Vec::with_capacity(*n);
                    for j in 0..*n {
                        blocks.push(DecResBlock::load(
                            dec.clone(),
                            &format!("{prefix}.res_blocks.{j}"),
                        )?);
                    }
                    UpLayer::Res(blocks)
                }
                DBlock::Up(stride) => {
                    UpLayer::Up(DepthToSpace::load(dec.clone(), &prefix, *stride)?)
                }
            });
        }
        let stats = vb.pp("per_channel_statistics");
        let mean = stats
            .get_unchecked("mean-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let std = stats
            .get_unchecked("std-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let encoder = match encoder_vb {
            Some(enc) => Some(VideoEncoder::load(
                enc.pp("encoder"),
                enc.pp("per_channel_statistics"),
                latent_channels,
                patch_size,
            )?),
            None => None,
        };
        Ok(Self {
            conv_in: Some(CausalConv3d::load(dec.clone(), "conv_in.conv")?),
            up_blocks,
            conv_out: Some(CausalConv3d::load(dec, "conv_out.conv")?),
            mean,
            std,
            patch_size,
            encoder,
        })
    }

    /// Encode `[-1,1]` video `[B,3,F,H,W]` to normalized LTX latents
    /// `[B,128,1+(F-1)/8,H/32,W/32]`.
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let (_b, channels, frames, height, width) = video.dims5()?;
        if channels != 3 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx vae encode: expected 3 input channels, got {channels}"
            )));
        }
        if frames == 0 || (frames - 1) % TEMPORAL_SCALE != 0 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx vae encode: frame count {frames} must equal 1 + k*{TEMPORAL_SCALE}"
            )));
        }
        if height == 0 || width == 0 || height % SPATIAL_SCALE != 0 || width % SPATIAL_SCALE != 0 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx vae encode: spatial dimensions {height}x{width} must be positive multiples of \
                 {SPATIAL_SCALE}"
            )));
        }
        let latent_channels = self.mean.dim(1)?;
        if latent_channels != LATENT_CHANNELS {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx vae encode: expected {LATENT_CHANNELS} latent channels, got {latent_channels}"
            )));
        }
        self.encoder
            .as_ref()
            .ok_or_else(|| {
                candle_gen::candle_core::Error::Msg(
                    "ltx vae encode: encoder weights were not loaded; use new_with_encoder".into(),
                )
            })?
            .encode(video)
    }

    /// Decode a normalized latent `[B, 128, F', H', W']` → video `[B, 3, F, 32·H', 32·W']` in ~[-1,1].
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let conv_in = self.conv_in.as_ref().ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(
                "ltx vae decode: convolutional decoder weights were not loaded on the DiffVAE route"
                    .into(),
            )
        })?;
        let conv_out = self.conv_out.as_ref().expect("decoder fields are atomic");
        // Denormalize: x · std + mean.
        let x =
            (latent.broadcast_mul(&self.std)? + self.mean.broadcast_as(latent.shape())?.clone())?;
        let mut x = conv_in.forward(&x, false)?;
        for layer in &self.up_blocks {
            x = match layer {
                UpLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = pixel_norm(&x)?;
        let x = candle_gen::candle_nn::ops::silu(&x)?;
        let x = conv_out.forward(&x, false)?;
        unpatchify(&x, self.patch_size)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`) — the candle port of
    /// mlx-gen-ltx `LtxVideoVae::decode_tiled` (sc-7076 / sc-6894). Splits the latent into overlapping
    /// spatial/temporal tiles via the shared pure `gen_core::tiling` geometry (`VaeTiling::LTX`: ×32
    /// spatial, ×8 **causal** temporal), decodes each tile through [`decode`](Self::decode), and
    /// trapezoidally blends them into the full video by pad-and-accumulate (bounded peak = one tile's
    /// decode + the full-output `output`/`weights` buffers). Falls back to single-pass `decode` when
    /// `cfg` does not fire for these dims.
    ///
    /// Numerically mirrors the parity-validated mlx version op-for-op; candle's eager evaluation makes
    /// the reference's per-tile `mx.eval` (the peak-bounding barrier) unnecessary. **NOTE (CUDA-gated):**
    /// the spatial-tiling path is straightforward, but temporal tiling crosses the causal-Conv3d frame
    /// boundary (each tile's leading edge replicates the *tile's* first frame) — the gen_core causal
    /// temporal mapping handles the geometry, but byte-parity vs. a full single-pass decode must be
    /// confirmed on real weights + CUDA (the Mac dev host can only compile-check this).
    pub fn decode_tiled(&self, latent: &Tensor, cfg: &TilingConfig) -> Result<Tensor> {
        // The tile/narrow/blend/pad-accumulate/normalize DRIVER is shared with the wan half in
        // `candle_gen::vae_tiling::decode_tiled` (sc-9006 / F-026). What stays ltx-specific: the
        // `VaeTiling::LTX` geometry (×32 spatial / ×8 causal temporal) and the single-pass `decode`
        // closure. Unlike wan, `cfg` may carry a temporal tile (ltx `decode` is not per-frame
        // streaming); the shared driver's `plan.t` loop handles both.
        vae_tiling::decode_tiled(Self::VAE_TILING, "ltx vae", latent, cfg, |tile| {
            self.decode(tile)
        })
    }

    /// **Memory-bounded** decode (sc-7076): derive the decoded output dims from the latent geometry
    /// (LTX VAE: ×32 spatial, ×8 **causal** temporal ⇒ `out_f = 1 + (T_lat−1)·8`), pick a budgeted
    /// tiling via [`auto_tiling_budgeted_ltx`], and run [`decode_tiled`](Self::decode_tiled) — or the
    /// single-pass [`decode`](Self::decode) when the whole decode already fits the VRAM budget. An
    /// over-budget decode returns a **catchable** error here instead of OOM-ing the worker. The candle
    /// analogue of mlx-gen-ltx `decode_to_frames`'s internal budgeting.
    pub fn decode_budgeted(&self, latent: &Tensor) -> Result<Tensor> {
        let (_b, _c, f, h, w) = latent.dims5()?;
        let out_f = 1 + (f as i32 - 1) * Self::VAE_TILING.temporal_scale; // causal ×8
        let out_h = h as i32 * Self::VAE_TILING.spatial_scale; // ×32
        let out_w = w as i32 * Self::VAE_TILING.spatial_scale;
        match auto_tiling_budgeted_ltx(out_h, out_w, out_f)? {
            Some(cfg) => self.decode_tiled(latent, &cfg),
            None => self.decode(latent),
        }
    }

    /// Execute the request-scoped bounded-decode rung.  The selected spatial cap is never merely
    /// telemetry: it forces the same shared tiling driver used by automatic safety budgeting.
    pub fn decode_budgeted_with_spatial_cap(
        &self,
        latent: &Tensor,
        tile_edge: u32,
        overlap: u32,
    ) -> Result<Tensor> {
        let (_b, _c, f, h, w) = latent.dims5()?;
        let out_f = 1 + (f as i32 - 1) * Self::VAE_TILING.temporal_scale;
        let out_h = h as i32 * Self::VAE_TILING.spatial_scale;
        let out_w = w as i32 * Self::VAE_TILING.spatial_scale;
        let mut cfg =
            plan_ltx_tiling(out_h, out_w, out_f, ltx_vae_safe_budget_gib())?.unwrap_or_default();
        // A one-tile result is still materialized through `decode_tiled`, preserving the selected
        // control rather than silently falling back to ordinary decode.
        cfg.spatial = Some(candle_gen::gen_core::tiling::SpatialTiling {
            tile_px: tile_edge as i32,
            overlap_px: overlap as i32,
        });
        self.decode_tiled(latent, &cfg)
    }
}

// --- sc-7076 / sc-6894: budgeted LTX VAE decode (candle) ------------------------------------------
//
// The shared budgeted-tiling DRIVER + budget resolver + selector now live ONCE in
// `candle_gen::vae_tiling` (sc-9006 / F-026, de-duped from the byte-near-identical wan/ltx copies);
// this module supplies only the LTX-specific cost CONSTANTS, the spatial+temporal candidate grid, and
// the cost model. The tile geometry itself is pure gen-core (`gen_core::tiling`), byte-identical to
// the mlx side.

const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;
/// Env override read by the shared [`vae_tiling::safe_budget_gib`] resolver.
const LTX_VAE_BUDGET_ENV: &str = "LTX_VAE_BUDGET_GIB";
/// Fraction of total VRAM treated as safe (matches the mlx 0.85 + candle-gen-seedvr2 convention).
const LTX_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;
/// Fallback budget when neither the env override nor `nvidia-smi` yields a value.
const LTX_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

// Cost-model constants. **CUDA-CALIBRATED (sc-7148)** — fit from real-weight peak-VRAM anchors measured
// by `tests/vae_decode_sweep.rs` on an RTX PRO 6000 Blackwell (sm_120, CUDA 12.9, f32, device-level
// `nvidia-smi` peak), replacing the mlx-Metal placeholders (sc-6894). The five anchors (output WxHxF /
// largest-tile px·fr → measured peak):
//   512²×25  single-pass                    →  5.99 GiB   (≈635 B/out-voxel above the floor)
//   768²×25  single-pass                    → 10.86 GiB
//   1024²×25 single-pass                    → 17.27 GiB
//   1280²×121 tiled 256px/64fr              → 14.83 GiB   (accumulator-dominated: tiny per-tile term)
//   1280²×121 tiled 512px/64fr              → 16.39 GiB
// The peak splits into a fixed floor (resident VAE decoder + CUDA context, ≈2.2 GiB baseline), a
// per-output-voxel accumulator term (the f32 `output`/`weights` buffers + pad/add transients, fit
// ≈54 B/voxel from the tiled anchors), and a per-tile activation term that scales with the largest
// tile's output volume (the single-pass anchors imply ≈635 B/voxel for the decoder's 1024-channel
// stack). Constants are rounded to the **conservative** (over-predicting) side so the budgeted selector
// never picks a tile that OOMs — the model reproduces every anchor at ratio 1.12–1.65× (never under).
// sc-12474 removed the pad-to-full tile transients. We intentionally retain these pre-change
// constants as conservative upper bounds: slice accumulation cannot raise the measured peak, and the
// anchor regression below still proves that the selector does not under-predict any calibrated row.
// The placeholders (40/300) under-predicted single-pass by ~1.9×; re-run the sweep after a decoder or
// candle-allocator change. See the `ltx_decode_peak_matches_cuda_anchors` regression test below.
const LTX_VAE_FIXED_BYTES: u64 = 2_700_000_000;
const LTX_VAE_ACCUM_BYTES_PER_VOXEL: u64 = 80;
const LTX_VAE_TILE_BYTES_PER_OUT_VOXEL: u64 = 620;

/// Candidate spatial tile sizes (output px, multiples of the LTX ×32 scale, overlap 64).
const LTX_VAE_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
const LTX_VAE_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 16), (48, 16), (24, 8)];

/// Estimated concurrent peak (GiB) of an LTX decode whose largest tile spans `tile_*` output voxels
/// while assembling an `out_*` video. `FIXED + ACCUM·out_vox + TILE·tile_vox`. Single-pass is
/// `tile_* == out_*`; a zero tile is the accumulator+fixed floor.
fn estimated_ltx_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (LTX_VAE_FIXED_BYTES as f64
        + LTX_VAE_ACCUM_BYTES_PER_VOXEL as f64 * out_voxels
        + LTX_VAE_TILE_BYTES_PER_OUT_VOXEL as f64 * tile_voxels)
        / GIB_F64
}

/// Conservative single-pass LTX VAE decode working-set peak in bytes.
///
/// This is the exact full-output case of the calibrated cost function used by
/// [`auto_tiling_budgeted_ltx`]. It includes that model's fixed decoder/base-working-set floor and
/// decode accumulators/activations, but no DiT or text-encoder composition weights.
pub fn conservative_video_decode_peak_bytes(width: u32, height: u32, frames: u32) -> Option<u64> {
    let voxels = u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(u64::from(frames))?;
    if voxels == 0 {
        return None;
    }
    let variable_per_voxel =
        LTX_VAE_ACCUM_BYTES_PER_VOXEL.checked_add(LTX_VAE_TILE_BYTES_PER_OUT_VOXEL)?;
    LTX_VAE_FIXED_BYTES.checked_add(variable_per_voxel.checked_mul(voxels)?)
}

/// Composable form of [`conservative_video_decode_peak_bytes`].
///
/// The calibrated fixed term mixes decoder residency with backend/runtime working set, so no
/// decoder-only portion is source-grounded. It therefore remains entirely in `working_set_bytes`
/// and `resident_decoder_bytes_included` is conservatively zero.
pub fn conservative_video_decode_memory_profile(
    width: u32,
    height: u32,
    frames: u32,
) -> Option<candle_gen::VideoDecodeMemoryProfile> {
    candle_gen::VideoDecodeMemoryProfile::new(
        conservative_video_decode_peak_bytes(width, height, frames)?,
        0,
    )
}

/// The safe peak-GiB budget for the LTX decode tiler. Resolved in order: `LTX_VAE_BUDGET_GIB` env
/// override (positive float — the deterministic injection point for the worker/tests) → total VRAM ×
/// `LTX_VAE_BUDGET_SAFE_FRAC` (via the shared trusted-path `nvidia-smi` probe
/// [`candle_gen::gpu::nvidia_smi_min_total_gib`] — an absolute System32/CUDA_PATH binary, never a bare
/// `PATH` lookup; sc-9014 / F-030) → `LTX_VAE_DEFAULT_BUDGET_GIB`.
pub fn ltx_vae_safe_budget_gib() -> f64 {
    vae_tiling::safe_budget_gib(
        LTX_VAE_BUDGET_ENV,
        LTX_VAE_BUDGET_SAFE_FRAC,
        LTX_VAE_DEFAULT_BUDGET_GIB,
    )
}

/// **Memory-budgeted** tiling for the LTX VAE decode — routes the shared `budgeted_plan` selector
/// through the LTX cost model. Caller passes the **output** dims. `Ok(None)` → single-pass already
/// fits; `Err` → a catchable over-budget signal returned before the decode (not an OOM).
pub fn auto_tiling_budgeted_ltx(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_ltx_tiling(height, width, out_frames, ltx_vae_safe_budget_gib())
}

/// Pure LTX tile selector (the `safe_gib` ceiling injected so it is unit-testable without a GPU).
fn plan_ltx_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    // Shared budgeted selector + error mapping (sc-9006 / F-026); ltx-specific: the spatial+temporal
    // candidate grid and the `estimated_ltx_decode_peak_gib` cost model.
    let candidates = TileCandidates {
        spatial_px: &LTX_VAE_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &LTX_VAE_TEMPORAL_FR,
        temporal_overlap_policy: TemporalOverlapPolicy::HalfTile,
    };
    vae_tiling::plan_tiling(
        "ltx vae decode",
        LtxVideoVae::VAE_TILING,
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_ltx_decode_peak_gib,
    )
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    /// This file's own source, for the route guard below.
    const VAE_SRC: &str = include_str!("vae.rs");

    /// The body of one `fn` in [`VAE_SRC`], from its signature to the matching `}`.
    fn fn_body(name: &str) -> &'static str {
        let start = VAE_SRC
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("vae.rs no longer defines `fn {name}`"));
        let open = VAE_SRC[start..]
            .find('{')
            .expect("a fn signature is followed by its body");
        let rest = &VAE_SRC[start + open..];
        let mut depth = 0usize;
        for (i, c) in rest.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &rest[..=i];
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces while slicing `fn {name}`");
    }

    /// sc-18799 AC1 — the **conv** decode must still route through [`auto_tiling_budgeted_ltx`],
    /// unchanged by the DiffVAE getting a budgeted selector of its own.
    ///
    /// Asserted on the ROUTE, not on an output: an output-level check would look identical whether
    /// the tiling came from the conv selector or from a "unified" one, because on a machine with
    /// headroom both would answer `None` and both decodes would be the same picture. What must not
    /// change is *which selector the call site names* — so this reads the call site.
    #[test]
    fn conv_decode_budgeted_still_selects_through_auto_tiling_budgeted_ltx() {
        let body = fn_body("decode_budgeted");
        assert!(
            body.contains("match auto_tiling_budgeted_ltx(out_h, out_w, out_f)?"),
            "the conv decode's auto-tiling arm no longer calls `auto_tiling_budgeted_ltx` with the \
             conv output dims — sc-18799 budgets the DiffVAE, it does not re-route the conv \
             decoder:\n{body}"
        );
        for foreign in ["diff_vae", "DiffVae", "budget::", "auto_diffvae"] {
            assert!(
                !body.contains(foreign),
                "the conv decode route now mentions `{foreign}` — the DiffVAE selector must not \
                 reach the conv decoder:\n{body}"
            );
        }
    }

    /// The other half of the same guard: the conv selector still answers from the **conv** candidate
    /// grid. A rewrite that pointed it at the DiffVAE's three-axis stage-4 grid would keep the call
    /// site's name and still be the regression this AC is about.
    #[test]
    fn the_conv_selector_still_answers_from_the_conv_candidate_grid() {
        // Sweep the budget down a long clip so both halves of the conv grid are asked their
        // question: every spatial edge selected must come from the conv list at overlap 64, and at
        // least one budget must be tight enough to select a conv temporal tile.
        let mut spatial_seen = 0usize;
        let mut temporal_seen = 0usize;
        for &safe_gib in &[40.0_f64, 32.0, 28.0, 26.0, 25.0] {
            let Ok(Some(plan)) = plan_ltx_tiling(720, 1280, 241, safe_gib) else {
                continue;
            };
            if let Some(spatial) = plan.spatial {
                spatial_seen += 1;
                assert!(
                    LTX_VAE_SPATIAL_PX.contains(&spatial.tile_px),
                    "conv spatial edge {} at {safe_gib} GiB is not one of the conv candidates \
                     {LTX_VAE_SPATIAL_PX:?}",
                    spatial.tile_px
                );
                assert_eq!(
                    spatial.overlap_px, 64,
                    "the conv selector's spatial overlap is 64 px"
                );
            }
            if let Some(temporal) = plan.temporal {
                temporal_seen += 1;
                assert!(
                    LTX_VAE_TEMPORAL_FR
                        .iter()
                        .any(|&(tile, _)| tile == temporal.tile_frames),
                    "conv temporal tile {} at {safe_gib} GiB is not one of the conv candidates \
                     {LTX_VAE_TEMPORAL_FR:?}",
                    temporal.tile_frames
                );
            }
        }
        assert!(
            spatial_seen > 0 && temporal_seen > 0,
            "the sweep never exercised both conv axes (spatial {spatial_seen}, temporal \
             {temporal_seen}) — the identity claim is untested"
        );
    }

    #[test]
    fn public_decode_peak_is_the_planners_full_output_case() {
        let profile = conservative_video_decode_memory_profile(64, 64, 9).unwrap();
        assert_eq!(profile.working_set_bytes(), 2_725_804_800);
        assert_eq!(profile.resident_decoder_bytes_included(), 0);
        assert_eq!(conservative_video_decode_peak_bytes(0, 64, 9), None);
        assert_eq!(
            conservative_video_decode_peak_bytes(u32::MAX, u32::MAX, u32::MAX),
            None
        );
    }

    #[test]
    fn ltx_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single-pass decode → no tiling.
        assert!(plan_ltx_tiling(256, 256, 25, 60.0).unwrap().is_none());
    }

    #[test]
    fn ltx_tiling_bounds_moderate_res_peak() {
        // 1280×1280×121 single-pass would peak ~66 GB; on a 48 GiB-class budget it must tile and keep
        // the recomputed peak under the safe ceiling (bounded/catchable). Cost constants are the
        // CUDA-calibrated (sc-7148) anchors; this checks the SELECTOR logic, not the calibration itself
        // (that is covered by `ltx_decode_peak_matches_cuda_anchors`).
        let safe = 48.0 * 0.85;
        let cfg = plan_ltx_tiling(1280, 1280, 121, safe)
            .unwrap()
            .expect("moderate-res LTX must tile");
        let th = cfg
            .spatial
            .map(|s| (s.tile_px as i64).min(1280))
            .unwrap_or(1280);
        let tw = th;
        let tf = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(121))
            .unwrap_or(121);
        let peak = estimated_ltx_decode_peak_gib(121, 1280, 1280, tf, th, tw);
        assert!(peak <= safe, "chosen peak {peak:.1} over safe {safe:.1}");
    }

    #[test]
    fn ltx_tiling_errors_when_unfittable() {
        // 4K × 257 frames under 8 GiB: output accumulators (+ fixed floor) alone blow it → catchable.
        assert!(plan_ltx_tiling(2160, 3840, 257, 8.0).is_err());
    }

    /// **sc-15325 — the CUDA lane's LTX grid can no longer starve the decoder either.**
    ///
    /// This crate carries the *identical* [`LTX_VAE_TEMPORAL_FR`] grid and routes through the
    /// *identical* gen-core `budgeted_plan`, so gen-core's `MIN_TEMPORAL_TILE_LATENT_FRAMES` removes
    /// the same `(48, 16)` and `(24, 8)` candidates — latent **6** and **3** at LTX's ×8
    /// `temporal_scale`. What is *not* shared is the cost model: these constants are CUDA-calibrated
    /// (sc-7148, RTX PRO 6000) with a 2.7 GiB fixed floor, where the mlx sibling's are Metal ones. So
    /// "does the floor keep the shipped envelope plannable?" is a genuinely different question here
    /// and has to be asked separately — a candidate removed on both backends can still be the one
    /// that made a CUDA budget fit.
    ///
    /// This is the mirror of `mlx-gen-ltx`'s `ltx_tiling_never_selects_a_starved_temporal_tile`
    /// (`crates/media/mlx-gen/mlx-gen-ltx/src/pipeline.rs`). The Linux/CUDA CI lanes exercise
    /// compilation, not tiling *selection*, so without this the CUDA lane would inherit the policy
    /// change with no coverage at all.
    #[test]
    fn ltx_tiling_never_selects_a_starved_temporal_tile() {
        use candle_gen::gen_core::tiling::{
            min_temporal_tile_frames, MIN_TEMPORAL_TILE_LATENT_FRAMES,
            MIN_TEMPORAL_TILE_LATENT_OVERLAP,
        };
        let scale = LtxVideoVae::VAE_TILING.temporal_scale;

        // The grid this crate actually ships, read through the floor: the two starved entries go.
        let kept: Vec<i32> = LTX_VAE_TEMPORAL_FR
            .iter()
            .map(|&(t, _)| t)
            .filter(|&t| t >= min_temporal_tile_frames(LtxVideoVae::VAE_TILING))
            .collect();
        assert_eq!(
            kept,
            vec![96, 64],
            "the floor must remove candle-gen-ltx's latent-6 (48f) and latent-3 (24f) candidates"
        );

        for (w, h, f) in [
            (1280, 704, 121),
            (1280, 720, 121),
            (768, 512, 241),
            (1920, 1088, 121),
        ] {
            for safe in [12.0f64, 16.0, 24.0, 32.0, 48.0, 96.0] {
                let Ok(Some(cfg)) = plan_ltx_tiling(h, w, f, safe) else {
                    continue; // infeasible or single-pass — neither can starve a tile.
                };
                let Some(t) = cfg.temporal else { continue };
                let lat = t.tile_frames / scale;
                assert!(
                    lat >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
                    "{w}x{h}x{f} @ {safe} GiB: selected a {lat}-latent-frame LTX temporal tile \
                     ({} output frames) — sc-15325's receptive-field floor",
                    t.tile_frames
                );
                assert!(
                    (t.overlap_frames / scale).min(lat - 1) >= MIN_TEMPORAL_TILE_LATENT_OVERLAP,
                    "{w}x{h}x{f} @ {safe} GiB: latent overlap under the blend floor"
                );
            }
        }

        // ...and the shipped envelope is still plannable on a real CUDA card. **This is the check the
        // mlx sibling could not make for us, and it found something**: on the CUDA cost model the
        // floor genuinely narrows the feasible window at 1280×704×121, because the accumulator floor
        // there is already 10.64 GiB and only the tile term is left to trade.
        //
        // | plan at 192 px spatial | latent tile | estimated peak |
        // |---|---|---|
        // | `(24, 8)` — removed by the floor | 3 | 11.15 GiB |
        // | `(48, 16)` — removed by the floor | 6 | 11.66 GiB |
        // | `(64, 16)` — the floor's smallest survivor | 8 | **12.00 GiB** |
        //
        // So the smallest budget that can plan this bucket moved **11.15 → 12.00 GiB safe**. No real
        // card class crosses that gap: `safe = total × 0.85`, so a 12 GB card is 10.2 GiB (already
        // under the 10.64 GiB accumulator floor — infeasible before *and* after, for a reason the
        // floor has nothing to do with) and a 16 GB card is 13.6 GiB (feasible before and after). The
        // exposed band is a 13.1–14.1 GB card, which is not a thing. Assert the 16 GB case, and pin
        // the band explicitly so a future cost-model change that widens it is visible rather than
        // discovered by a user.
        const SAFE_16GB: f64 = 16.0 * LTX_VAE_BUDGET_SAFE_FRAC; // 13.6 GiB
        for (w, h, f) in [(1280, 704, 121), (1280, 720, 121), (768, 512, 241)] {
            plan_ltx_tiling(h, w, f, SAFE_16GB).unwrap_or_else(|e| {
                panic!("{w}x{h}x{f} became infeasible on a 16 GB card ({SAFE_16GB} GiB safe): {e}")
            });
        }
        // The narrowed band itself, pinned. If this ever starts passing, the floor's cost on this
        // backend changed and the paragraph above needs re-deriving.
        assert!(
            plan_ltx_tiling(704, 1280, 121, 11.5).is_err(),
            "1280x704x121 now plans at 11.5 GiB — the floor's CUDA feasibility floor moved; re-derive \
             the 11.15 -> 12.00 GiB band documented above"
        );
        assert!(
            plan_ltx_tiling(704, 1280, 121, 12.1).is_ok(),
            "1280x704x121 no longer plans at 12.1 GiB — the floor's CUDA cost grew beyond the \
             documented 12.00 GiB survivor"
        );
    }

    #[test]
    fn ltx_budget_env_override_wins() {
        // The deterministic injection point the worker/tests use. (Set/clear in-process.)
        std::env::set_var("LTX_VAE_BUDGET_GIB", "42.5");
        assert_eq!(ltx_vae_safe_budget_gib(), 42.5);
        std::env::remove_var("LTX_VAE_BUDGET_GIB");
    }

    /// sc-7148: the calibrated cost model must stay **conservative** against the real CUDA peak-VRAM
    /// anchors (RTX PRO 6000 Blackwell, sm_120, f32) it was fit from — `estimated ≥ measured` for every
    /// anchor (never under-predict ⇒ the selector never OKs a tile that OOMs), and not absurdly over
    /// (≤ 2.5×). Regenerate the anchors with `cargo test -p candle-gen-ltx --features cuda --release
    /// --test integration -- vae_decode_sweep:: --ignored --nocapture` after a decoder or
    /// candle-allocator change.
    #[test]
    fn ltx_decode_peak_matches_cuda_anchors() {
        // (out_f, out_h, out_w, tile_f, tile_h, tile_w, measured_peak_gib). Single-pass ⇒ tile == out.
        let anchors: [(i64, i64, i64, i64, i64, i64, f64); 5] = [
            (25, 512, 512, 25, 512, 512, 5.9873),
            (25, 768, 768, 25, 768, 768, 10.8623),
            (25, 1024, 1024, 25, 1024, 1024, 17.2686),
            (121, 1280, 1280, 64, 256, 256, 14.8311),
            (121, 1280, 1280, 64, 512, 512, 16.3936),
        ];
        for (of, oh, ow, tf, th, tw, measured) in anchors {
            let est = estimated_ltx_decode_peak_gib(of, oh, ow, tf, th, tw);
            assert!(
                est >= measured,
                "under-predicts {ow}x{oh}x{of} tile {tw}x{th}x{tf}: est {est:.2} < measured {measured:.2} GiB"
            );
            assert!(
                est <= measured * 2.5,
                "over-predicts {ow}x{oh}x{of} tile {tw}x{th}x{tf}: est {est:.2} > 2.5x measured {measured:.2} GiB"
            );
        }
    }
}

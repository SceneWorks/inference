//! Learned LTX 2x latent upsamplers.  These are deliberately real checkpoint
//! components, not interpolation fallbacks: stage two is valid only after this
//! network has transformed the half-resolution stage-one latent.
//!
//! Two shipped variants, selected by [`LatentUpsamplerMode`] (sc-18773), ported from upstream
//! `ltx_core/model/upsampler/model.py` at `Lightricks/LTX-2` @ `d1511477` (v1.2.0):
//!
//! * **spatial ×2** — LTX-2.3 `upsampler.safetensors` and LTX-2.5
//!   `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (`mid_channels` 1024): a frame-wise
//!   `Conv2d(mid, 4·mid)` + `PixelShuffle2d(2)`. `H,W → 2H,2W`, frame count untouched.
//! * **temporal ×2** — LTX-2.5 `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`
//!   (`mid_channels` 512): a `Conv3d(mid, 2·mid)` + frame-axis `PixelShuffle1d(2)`, then the
//!   **leading frame is dropped**, so `F → 2F−1` and `H,W` are untouched. The drop is not a trim:
//!   latent frame 0 encodes a single pixel frame, and `2·(8k+1) − 1 = 16k+1` is what preserves LTX's
//!   `n % 8 == 1` latent-frame invariant.
//!
//! Variant selection reads the **rank of `upsampler.0.weight`** (4 → Conv2d, 5 → Conv3d), matching
//! the MLX port exactly: SceneWorks-converted LTX-2.3 trees ship no `__metadata__` to read a config
//! from. [`LatentUpsampler::assert_matches_config`] cross-checks the two authorities where the
//! checkpoint does declare one, and [`LatentUpsampler::from_checkpoint`] is the path-taking
//! constructor that runs it — so the cross-check sits on the production load path rather than only
//! in the parity tests.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::ltx_checkpoint::{
    upsampled_latent_frames, LatentUpsamplerConfig, LatentUpsamplerMode, LtxCheckpointMetadata,
};

use crate::conv3d::{chunked_conv2d, ZeroPaddedConv3d};
use crate::quant::guard_no_scales;

const GROUPS: usize = 32;
const EPS: f64 = 1e-5;

struct Conv2d {
    weight: Tensor,
    bias: Tensor,
}

impl Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = chunked_conv2d(x, &self.weight, 1)?;
        y.broadcast_add(&self.bias.reshape((1, self.bias.elem_count(), 1, 1))?)
    }
}

/// Checkpoint GroupNorm(32), evaluated in f32 across channel-group, time and
/// spatial axes before returning the input dtype.
struct GroupNorm32 {
    weight: Tensor,
    bias: Tensor,
}

impl GroupNorm32 {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: vb.get_unchecked(&format!("{prefix}.weight"))?,
            bias: vb.get_unchecked(&format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let (b, c, t, h, w) = x.dims5()?;
        if c % GROUPS != 0 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx upsampler: GroupNorm32 requires channels divisible by {GROUPS}, got {c}"
            )));
        }
        let groups = x
            .to_dtype(DType::F32)?
            .reshape((b, GROUPS, c / GROUPS, t, h, w))?;
        let mean = groups.mean_keepdim((2, 3, 4, 5))?;
        let centered = groups.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim((2, 3, 4, 5))?;
        let normalized = centered
            .broadcast_div(&(variance + EPS)?.sqrt()?)?
            .reshape((b, c, t, h, w))?;
        let weight = self.weight.to_dtype(DType::F32)?.reshape((1, c, 1, 1, 1))?;
        let bias = self.bias.to_dtype(DType::F32)?.reshape((1, c, 1, 1, 1))?;
        ((normalized.broadcast_mul(&weight)?).broadcast_add(&bias)?).to_dtype(dtype)
    }
}

struct ResBlock {
    conv1: ZeroPaddedConv3d,
    norm1: GroupNorm32,
    conv2: ZeroPaddedConv3d,
    norm2: GroupNorm32,
}

impl ResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: ZeroPaddedConv3d::load(vb.clone(), &format!("{prefix}.conv1"))?,
            norm1: GroupNorm32::load(vb.clone(), &format!("{prefix}.norm1"))?,
            conv2: ZeroPaddedConv3d::load(vb.clone(), &format!("{prefix}.conv2"))?,
            norm2: GroupNorm32::load(vb, &format!("{prefix}.norm2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let y = self.norm1.forward(&self.conv1.forward(x)?)?;
        let y = candle_gen::candle_nn::ops::silu(&y)?;
        let y = self.conv2.forward(&y)?;
        let y = self.norm2.forward(&y)?;
        candle_gen::candle_nn::ops::silu(&(y + residual)?)
    }
}

/// NCHW PixelShuffle with the PyTorch channel order used by the upsampler.
pub(crate) fn pixel_shuffle2d(x: &Tensor, upscale: usize) -> Result<Tensor> {
    let (n, channels, h, w) = x.dims4()?;
    let divisor = upscale * upscale;
    if channels % divisor != 0 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx upsampler: PixelShuffle({upscale}) cannot divide {channels} channels"
        )));
    }
    let out_channels = channels / divisor;
    x.reshape((n, out_channels, upscale, upscale, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((n, out_channels, h * upscale, w * upscale))?
        .contiguous()
}

/// `PixelShuffleND(1)` on `NCTHW` — `(N, C·r, T, H, W) → (N, C, T·r, H, W)`.
///
/// Upstream is `rearrange(x, "b (c p1) f h w -> b c (f p1) h w")`: the channel axis decomposes
/// `c`-major / `p1`-minor and the shuffled frame index is `f·r + p1`, so `p1` stays *minor* to `T`
/// and consecutive output frames come from the same input frame.
pub(crate) fn pixel_shuffle1d_frames(x: &Tensor, upscale: usize) -> Result<Tensor> {
    let (n, channels, t, h, w) = x.dims5()?;
    if channels % upscale != 0 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx upsampler: PixelShuffle1d({upscale}) cannot divide {channels} channels"
        )));
    }
    let out_channels = channels / upscale;
    x.reshape((n, out_channels, upscale, t, h, w))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((n, out_channels, t * upscale, h, w))?
        .contiguous()
}

/// The resampler stage — the one place the two shipped checkpoints differ.
enum Resampler {
    /// Frame-wise `Conv2d mid→4·mid` + `PixelShuffle2d(2)`; frame count unchanged.
    Spatial2x { conv: Conv2d },
    /// `Conv3d mid→2·mid` + frame-axis `PixelShuffle1d(2)` + leading-frame drop; `T → 2T−1`.
    Temporal2x { conv: ZeroPaddedConv3d },
}

impl Resampler {
    /// Structure-from-weights: `{prefix}.0.weight` is rank 4 for the spatial Conv2d and rank 5 for
    /// the temporal Conv3d. The weight is fetched once and handed to whichever module the rank
    /// selects — a mmapped `VarBuilder` materializes on every `get`, and this tensor is 150 MB.
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        let key = format!("{prefix}.0");
        let weight = guard_no_scales(&vb, &key, vb.dtype())?;
        let bias = vb.get_unchecked(&format!("{key}.bias"))?;
        match weight.dims().len() {
            4 => Ok(Resampler::Spatial2x {
                conv: Conv2d {
                    weight: weight.contiguous()?,
                    bias,
                },
            }),
            5 => Ok(Resampler::Temporal2x {
                conv: ZeroPaddedConv3d::from_parts(weight, bias)?,
            }),
            n => Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx latent upsampler: {key}.weight has rank {n}, expected 4 (spatial Conv2d) or \
                 5 (temporal Conv3d)"
            ))),
        }
    }

    fn mode(&self) -> LatentUpsamplerMode {
        match self {
            Resampler::Spatial2x { .. } => LatentUpsamplerMode::Spatial2x,
            Resampler::Temporal2x { .. } => LatentUpsamplerMode::Temporal2x,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        match self {
            Resampler::Spatial2x { conv } => {
                let frames = x
                    .permute((0, 2, 1, 3, 4))?
                    .reshape((b * t, c, h, w))?
                    .contiguous()?;
                let up = pixel_shuffle2d(&conv.forward(&frames)?, 2)?;
                up.reshape((b, t, c, h * 2, w * 2))?
                    .permute((0, 2, 1, 3, 4))?
                    .contiguous()
            }
            Resampler::Temporal2x { conv } => {
                let shuffled = pixel_shuffle1d_frames(&conv.forward(x)?, 2)?;
                let frames = shuffled.dim(2)?;
                if frames < 2 {
                    return Err(candle_gen::candle_core::Error::Msg(format!(
                        "ltx latent upsampler: temporal resampler produced {frames} frame(s); the \
                         leading-frame drop needs at least 2"
                    )));
                }
                // Drop the leading frame: latent frame 0 encodes ONE pixel frame, so its shuffled
                // pair would duplicate it. `2t − 1` is also what preserves `n % 8 == 1`.
                shuffled.narrow(2, 1, frames - 1)?.contiguous()
            }
        }
    }
}

/// LTX's learned `LatentUpsampler`: 128→mid Conv3d, `num_blocks_per_stage` residual blocks, the
/// spatial or temporal resampler, the same count of residual blocks again, mid→128 Conv3d. Names
/// match the published checkpoint layout exactly; `num_blocks_per_stage` and `mid_channels` are read
/// from the weights rather than assumed.
pub struct LatentUpsampler {
    initial_conv: ZeroPaddedConv3d,
    initial_norm: GroupNorm32,
    res_blocks: Vec<ResBlock>,
    resampler: Resampler,
    post_res_blocks: Vec<ResBlock>,
    final_conv: ZeroPaddedConv3d,
}

impl LatentUpsampler {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        // Structure-from-weights with a **floor**: a stage whose `.0` key is absent is a truncated
        // checkpoint, not a zero-block network. Without this the `while` loop returns an empty
        // `Vec` and a shallower model runs silently — which the replaced hard-coded
        // `(0..4).map(ResBlock::load)` could not do.
        let stage = |stem: &str| -> Result<Vec<ResBlock>> {
            let mut blocks = Vec::new();
            let mut i = 0;
            while vb.contains_tensor(&format!("{stem}.{i}.conv1.weight")) {
                blocks.push(ResBlock::load(vb.clone(), &format!("{stem}.{i}"))?);
                i += 1;
            }
            if blocks.is_empty() {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "ltx latent upsampler: no residual blocks under {stem}.* — the checkpoint is \
                     missing {stem}.0.conv1.weight"
                )));
            }
            Ok(blocks)
        };
        let res_blocks = stage("res_blocks")?;
        let post_res_blocks = stage("post_upsample_res_blocks")?;
        // Upstream builds both stages from the one `num_blocks_per_stage`, so they are the same
        // count by construction; a file where they differ has lost blocks from one side.
        if res_blocks.len() != post_res_blocks.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx latent upsampler: res_blocks has {} block(s) but post_upsample_res_blocks has \
                 {} — both stages are built from the same num_blocks_per_stage",
                res_blocks.len(),
                post_res_blocks.len()
            )));
        }
        Ok(Self {
            initial_conv: ZeroPaddedConv3d::load(vb.clone(), "initial_conv")?,
            initial_norm: GroupNorm32::load(vb.clone(), "initial_norm")?,
            res_blocks,
            resampler: Resampler::load(vb.clone(), "upsampler")?,
            post_res_blocks,
            final_conv: ZeroPaddedConv3d::load(vb, "final_conv")?,
        })
    }

    /// Build from a latent-upsampler `.safetensors` **file**, cross-checking the declared config.
    ///
    /// This is the only path-taking constructor, and the one every production load site uses:
    /// [`Self::load`] reads the structure out of the tensors, and when the file carries a
    /// `__metadata__["config"]` object this reads that too and runs
    /// [`Self::assert_matches_config`] before returning. Loading through a bare `load` on a stamped
    /// file would let the rank silently win over a config that disagrees — which is the whole point
    /// of having two authorities.
    ///
    /// A file with no `__metadata__["config"]` (every SceneWorks-converted LTX-2.3 tree) simply
    /// skips the cross-check; that is a checkpoint that declares nothing, not one that disagrees.
    pub fn from_checkpoint(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let msg = |e: &dyn std::fmt::Display| candle_gen::candle_core::Error::Msg(e.to_string());
        let files = [path.to_path_buf()];
        let vb = candle_gen::mmap_var_builder(&files, dtype, device).map_err(|e| msg(&e))?;
        let up = Self::load(vb)?;
        let meta = LtxCheckpointMetadata::from_file(path).map_err(|e| msg(&e))?;
        if meta.config().is_some() {
            let config = LatentUpsamplerConfig::from_metadata(path, &meta).map_err(|e| msg(&e))?;
            up.assert_matches_config(&config)?;
        }
        Ok(up)
    }

    /// Which axis this checkpoint rescales, as read from its weights.
    pub fn mode(&self) -> LatentUpsamplerMode {
        self.resampler.mode()
    }

    /// Latent frame count this upsampler produces from `frames` input frames.
    pub fn output_frames(&self, frames: usize) -> Result<usize> {
        upsampled_latent_frames(frames, self.mode())
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))
    }

    /// Assert the structure read from the weights agrees with the config the checkpoint declares.
    ///
    /// Both are authorities on the same fact and they are read independently, so a disagreement
    /// means one of them is being misread — never something to paper over by preferring one.
    pub fn assert_matches_config(&self, config: &LatentUpsamplerConfig) -> Result<()> {
        let declared = config
            .mode()
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        if declared != self.mode() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx latent upsampler: config declares {declared:?} but the weights are {:?}",
                self.mode()
            )));
        }
        // Both stages are the same length by the time `load` returns, so one comparison covers
        // `num_blocks_per_stage`.
        let mismatch = |what: &str, declared: u64, loaded: usize| {
            candle_gen::candle_core::Error::Msg(format!(
                "ltx latent upsampler: config declares {what}={declared} but the weights carry \
                 {loaded}"
            ))
        };
        if config.num_blocks_per_stage as usize != self.res_blocks.len() {
            return Err(mismatch(
                "num_blocks_per_stage",
                config.num_blocks_per_stage,
                self.res_blocks.len(),
            ));
        }
        // Checkpoint-native PyTorch layout: `initial_conv` is `[mid, in, kt, kh, kw]` and
        // `final_conv` is `[in, mid, kt, kh, kw]`. Both are checked, so a swapped pair cannot
        // cancel out.
        let initial = self.initial_conv.weight_dims();
        let final_ = self.final_conv.weight_dims();
        for (what, declared, loaded) in [
            ("in_channels", config.in_channels, initial[1]),
            ("in_channels", config.in_channels, final_[0]),
            ("mid_channels", config.mid_channels, initial[0]),
            ("mid_channels", config.mid_channels, final_[1]),
        ] {
            if declared as usize != loaded {
                return Err(mismatch(what, declared, loaded));
            }
        }
        Ok(())
    }

    /// `[B,128,T,H,W]` → `[B,128,T,2H,2W]` (spatial) or `[B,128,2T−1,H,W]` (temporal).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let frames_in = x.dim(2)?;
        let frames_out = self.output_frames(frames_in)?;
        let mut y = candle_gen::candle_nn::ops::silu(
            &self.initial_norm.forward(&self.initial_conv.forward(x)?)?,
        )?;
        for block in &self.res_blocks {
            y = block.forward(&y)?;
        }
        y = self.resampler.forward(&y)?;
        // Checked here, on the resampler's own output, rather than on the returned tensor: this is
        // the only stage that moves the frame axis, so a wrong count localises to it.
        let produced = y.dim(2)?;
        if produced != frames_out {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx latent upsampler: {:?} resampler turned {frames_in} latent frames into \
                 {produced}, expected {frames_out}",
                self.mode()
            )));
        }
        for block in &self.post_res_blocks {
            y = block.forward(&y)?;
        }
        self.final_conv.forward(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn pixel_shuffle_preserves_pytorch_channel_mapping() -> Result<()> {
        let x = Tensor::arange(0f32, 16f32, &Device::Cpu)?.reshape((1, 4, 2, 2))?;
        let y = pixel_shuffle2d(&x, 2)?;
        assert_eq!(y.dims(), &[1, 1, 4, 4]);
        assert_eq!(
            y.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 4., 1., 5., 8., 12., 9., 13., 2., 6., 3., 7., 10., 14., 11., 15.]
        );
        Ok(())
    }

    #[test]
    fn group_norm_broadcasts_production_upsampler_variance_shape() -> Result<()> {
        let device = Device::Cpu;
        let norm = GroupNorm32 {
            weight: Tensor::ones(1024, DType::F32, &device)?,
            bias: Tensor::zeros(1024, DType::F32, &device)?,
        };
        let x = Tensor::ones((1, 1024, 5, 8, 12), DType::F32, &device)?;
        let y = norm.forward(&x)?;
        assert_eq!(y.dims(), &[1, 1024, 5, 8, 12]);
        assert_eq!(y.sum_all()?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }
}

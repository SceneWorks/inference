//! **Tiny AutoEncoder decoder** (TAESD architecture, madebyollin/taesd) for the Candle trainers —
//! the Candle twin of `mlx_gen::train::tae` (sc-2125 / sc-24830): the small, frozen, fully
//! differentiable latent → pixel decoder the shared perceptual path ([`super::perceptual`]) runs a
//! trainer's x0 prediction through (epic 2123 E8).
//!
//! One module graph covers the whole TAESD family (reference: `taesd.py` in
//! github.com/madebyollin/taesd — `Decoder`, `Decoder(use_midblock_gn=True)` and `F16Decoder`):
//!
//! | checkpoint (HF repo / file) | VAE family | latent ch | variant |
//! |---|---|---|---|
//! | `madebyollin/taef1` `diffusion_pytorch_model.safetensors` | FLUX.1 (Z-Image) | 16 | `Decoder` |
//! | `madebyollin/taesdxl` `diffusion_pytorch_model.safetensors` | SDXL (SDXL, Illustrious, Kolors) | 4 | `Decoder` |
//! | `madebyollin/taesd3` `diffusion_pytorch_model.safetensors` | SD3 / SD3.5 | 16 | `Decoder` |
//! | `madebyollin/taef2` `taef2.safetensors` | FLUX.2 (Lens) | 32 | `Decoder`, mid-block GroupNorm pool (`arch_variant = "flux_2"`) |
//! | `madebyollin/taeqi2_1` `taeqi2_1.safetensors` | Qwen-Image-2.1 | 64 | `F16Decoder` (16×, RGBA, 2×2 pixel-shuffle head) |
//!
//! **Input normalization.** Every TAESD-family decoder consumes the diffusion model's **normalized
//! sampler latent directly** (diffusers checkpoints ship `scaling_factor = 1.0`, `shift_factor =
//! 0.0`; the TAEQI2.1 wrapper sets `latents_mean = 0` / `latents_std = 1`; the TAEF2 wrapper swaps in
//! an identity latent batch-norm, so it decodes the transformer's batch-normalized latent
//! **unpatchified** to `[32, h/8, w/8]`). No de-normalization is applied here.
//!
//! Architecture (`Decoder`, `num_decoder_blocks = [3, 3, 3, 1]`): `tanh(x/3)·3` →
//! conv3×3(latent→C₀)+ReLU → 3×Block → ↑2 → conv3×3(no bias, C₀→C₁) → 3×Block → ↑2 → conv3×3(C₁→C₂) →
//! 3×Block → ↑2 → conv3×3(C₂→C₃) → 1×Block → conv3×3(C₃→out). A Block is
//! `ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x)`; with the mid-block pool (TAEF2, first stage only) it
//! first does `x += conv1×1(ReLU(GroupNorm₄(conv1×1(x))))` (hidden width 4·C₀, no conv bias).
//! `F16Decoder` widens the low-resolution stages (`C = [256, 128, 64, 64]`), projects to
//! `4·image_channels` and pixel-shuffles 2×2 (16× total). Raw output ≈ `[0, 1]`, clamped; an RGBA
//! decoder (TAEQI2.1, straight alpha) is reduced to its RGB channels.
//!
//! Every op used here (conv2d, nearest upsample, relu, tanh, clamp, reshape/permute/narrow, and the
//! GroupNorm composed from mean/sqr/sqrt/div — candle's fused norms have no backward) has a candle
//! backward, so the decode is differentiable in the latents. Weights: the diffusers
//! `decoder.layers.{i}.…` layout (verified against all five published safetensors headers), torch
//! OIHW conv kernels (candle-native, no permute), read as f32 (taeqi2_1 ships f16).

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::perceptual::AuxModelFootprint;
use crate::weights::Weights;
use crate::{CandleError, Result};

/// TAESD-family decoder hyperparameters of the plain `Decoder` graph (uniform width, RGB, 8×).
/// The structural variants (TAEF2's mid-block pool, TAEQI2.1's `F16Decoder`) are described by
/// [`TinyDecoderSpec`]; every API taking a decoder description accepts either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyDecoderConfig {
    /// Latent channels the decoder consumes (16 for TAEF1/TAESD3, 4 for TAESDXL).
    pub latent_channels: usize,
    /// Hidden width of every stage (64 for every shipped plain checkpoint).
    pub channels: usize,
    /// Blocks per stage; one ×2 upsample between consecutive stages.
    pub blocks: [usize; 4],
}

impl TinyDecoderConfig {
    /// `madebyollin/taef1` — the FLUX.1 16-channel latent API (Z-Image's VAE family).
    pub fn taef1() -> Self {
        Self {
            latent_channels: 16,
            channels: 64,
            blocks: [3, 3, 3, 1],
        }
    }

    /// `madebyollin/taesdxl` — the SDXL 4-channel latent API (SDXL, Illustrious, Kolors).
    pub fn taesdxl() -> Self {
        Self {
            latent_channels: 4,
            ..Self::taef1()
        }
    }

    /// `madebyollin/taesd3` — the SD3 16-channel latent API (SD3 / SD3.5).
    pub fn taesd3() -> Self {
        Self::taef1()
    }

    /// Spatial upscale factor latent → pixels (8).
    pub fn upscale(&self) -> usize {
        TinyDecoderSpec::from(self.clone()).upscale()
    }

    /// Exact parameter count of the decoder graph.
    pub fn param_count(&self) -> u64 {
        TinyDecoderSpec::from(self.clone()).param_count()
    }

    /// Pre-load memory figures for `out_h × out_w` training images (resident f32 weights + one
    /// differentiable decode), for [`super::perceptual::perceptual_footprint_bytes`].
    pub fn footprint(&self, out_h: u32, out_w: u32) -> AuxModelFootprint {
        TinyDecoderSpec::from(self.clone()).footprint(out_h, out_w)
    }

    /// Conservative upper bound on the training working set of one differentiable decode to an
    /// `out_h × out_w` image, in bytes (see [`TinyDecoderSpec::training_working_set_bytes`]).
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        TinyDecoderSpec::from(self.clone()).training_working_set_bytes(out_h, out_w)
    }
}

/// Full structural description of a TAESD-family decoder: the plain [`TinyDecoderConfig`] graph
/// plus the `taesd.py` variants (per-stage widths, mid-block GroupNorm pool, RGBA + pixel-shuffle
/// head).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyDecoderSpec {
    /// Latent channels the decoder consumes.
    pub latent_channels: usize,
    /// Width of each stage (`[C; 4]` for the plain decoder, `[256, 128, 64, 64]` for F16).
    pub stage_channels: [usize; 4],
    /// Blocks per stage (`[3, 3, 3, 1]` for every shipped checkpoint).
    pub blocks: [usize; 4],
    /// The first stage's blocks carry the mid-block GroupNorm pool (`use_midblock_gn`, TAEF2).
    pub midblock_gn: bool,
    /// Image channels the decoder produces (3 RGB, 4 RGBA); [`TinyDecoder::decode`] returns RGB.
    pub image_channels: usize,
    /// The head projects to `4·image_channels` and pixel-shuffles 2×2 (`F16Decoder`).
    pub pixel_shuffle: bool,
}

impl From<TinyDecoderConfig> for TinyDecoderSpec {
    fn from(c: TinyDecoderConfig) -> Self {
        Self {
            latent_channels: c.latent_channels,
            stage_channels: [c.channels; 4],
            blocks: c.blocks,
            midblock_gn: false,
            image_channels: 3,
            pixel_shuffle: false,
        }
    }
}

/// GroupNorm group count of the mid-block pool (`nn.GroupNorm(4, 4·C)`).
pub(crate) const MIDBLOCK_GN_GROUPS: usize = 4;
/// torch `nn.GroupNorm` default eps.
pub(crate) const MIDBLOCK_GN_EPS: f64 = 1e-5;

impl TinyDecoderSpec {
    /// `madebyollin/taef1` (plain decoder, 16 latent channels).
    pub fn taef1() -> Self {
        TinyDecoderConfig::taef1().into()
    }

    /// `madebyollin/taesdxl` (plain decoder, 4 latent channels).
    pub fn taesdxl() -> Self {
        TinyDecoderConfig::taesdxl().into()
    }

    /// `madebyollin/taesd3` (plain decoder, 16 latent channels).
    pub fn taesd3() -> Self {
        TinyDecoderConfig::taesd3().into()
    }

    /// `madebyollin/taef2` — the FLUX.2 32-channel latent API (Lens): the plain decoder with the
    /// first stage's mid-block GroupNorm pool (`taesd.py` `arch_variant = "flux_2"`).
    pub fn taef2() -> Self {
        Self {
            latent_channels: 32,
            midblock_gn: true,
            ..Self::taef1()
        }
    }

    /// `madebyollin/taeqi2_1` — the Qwen-Image-2.1 64-channel, 16× latent API: `F16Decoder`
    /// (`arch_variant = "f16"`, RGBA, `image_channels = 4`).
    pub fn taeqi2_1() -> Self {
        Self {
            latent_channels: 64,
            stage_channels: [256, 128, 64, 64],
            blocks: [3, 3, 3, 1],
            midblock_gn: false,
            image_channels: 4,
            pixel_shuffle: true,
        }
    }

    /// The plain-config view (`channels` = the last stage's width).
    pub fn config(&self) -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels: self.latent_channels,
            channels: self.stage_channels[3],
            blocks: self.blocks,
        }
    }

    /// Spatial upscale factor latent → pixels (8, or 16 with the pixel-shuffle head).
    pub fn upscale(&self) -> usize {
        (1 << (self.blocks.len() - 1)) * if self.pixel_shuffle { 2 } else { 1 }
    }

    /// Channels of the final conv (before any pixel shuffle).
    fn head_channels(&self) -> usize {
        self.image_channels * if self.pixel_shuffle { 4 } else { 1 }
    }

    /// Exact parameter count of the decoder graph.
    pub fn param_count(&self) -> u64 {
        let conv = |i: usize, o: usize, bias: bool| {
            let (i, o) = (i as u64, o as u64);
            o * i * 9 + if bias { o } else { 0 }
        };
        let s = self.stage_channels;
        let stages = self.blocks.len();
        let mut n = conv(self.latent_channels, s[0], true);
        for (i, &nb) in self.blocks.iter().enumerate() {
            n += nb as u64 * 3 * conv(s[i], s[i], true);
            if i + 1 < stages {
                n += conv(s[i], s[i + 1], false);
            }
        }
        if self.midblock_gn {
            let (c, g) = (s[0] as u64, 4 * s[0] as u64);
            // conv1×1(c→4c, no bias) + GroupNorm affine (2·4c) + conv1×1(4c→c, no bias).
            n += self.blocks[0] as u64 * (c * g + 2 * g + g * c);
        }
        n + conv(s[stages - 1], self.head_channels(), true)
    }

    /// Pre-load memory figures for `out_h × out_w` training images (resident f32 weights + one
    /// differentiable decode), for [`super::perceptual::perceptual_footprint_bytes`].
    pub fn footprint(&self, out_h: u32, out_w: u32) -> AuxModelFootprint {
        AuxModelFootprint {
            param_bytes: self.param_count() * 4,
            working_set_bytes: self.training_working_set_bytes(out_h, out_w),
            reference_bytes_per_image: 0,
        }
    }

    /// Conservative upper bound on the training working set of one differentiable decode to an
    /// `out_h × out_w` image, in bytes (same accounting as the MLX twin: every retained
    /// intermediate per Block / stage / mid-block pool, f32, ×2 for the cotangents; not measured).
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        let per_block = 7u64;
        let full = out_h as u64 * out_w as u64;
        let last_stage = if self.pixel_shuffle { full / 4 } else { full };
        let stages = self.blocks.len();
        let mut channel_pixels = 0u64;
        for (i, &n) in self.blocks.iter().enumerate() {
            let pixels = last_stage / (1u64 << (2 * (stages - 1 - i)));
            let c = self.stage_channels[i] as u64;
            channel_pixels += (per_block * n as u64 + 2) * c * pixels;
            if self.midblock_gn && i == 0 {
                channel_pixels += n as u64 * (3 * 4 * c + c) * pixels;
            }
        }
        channel_pixels += 3 * self.head_channels() as u64 * last_stage;
        channel_pixels * 4 * 2
    }
}

/// The optional mid-block GroupNorm pool of a TAEF2 first-stage Block.
struct Pool {
    expand: Tensor,
    gn_weight: Tensor,
    gn_bias: Tensor,
    project: Tensor,
}

/// PyTorch `nn.GroupNorm` over NCHW `x`, composed from differentiable candle ops (groups are
/// contiguous channel ranges, statistics over (channels-in-group, H, W), biased variance).
fn group_norm_nchw(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    groups: usize,
    eps: f64,
) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let g = x.reshape((b, groups, (c / groups) * h * w))?;
    let mean = g.mean_keepdim(2)?;
    let centered = g.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?;
    let normed = centered
        .broadcast_div(&(var + eps)?.sqrt()?)?
        .reshape((b, c, h, w))?;
    Ok(normed
        .broadcast_mul(&weight.reshape((1, c, 1, 1))?)?
        .broadcast_add(&bias.reshape((1, c, 1, 1))?)?)
}

impl Pool {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.conv2d(&self.expand, 0, 1, 1, 1)?;
        let h = group_norm_nchw(
            &h,
            &self.gn_weight,
            &self.gn_bias,
            MIDBLOCK_GN_GROUPS,
            MIDBLOCK_GN_EPS,
        )?
        .relu()?;
        Ok(h.conv2d(&self.project, 0, 1, 1, 1)?)
    }
}

struct Block {
    convs: [(Tensor, Tensor); 3],
    pool: Option<Pool>,
}

fn conv(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.conv2d(w, 1, 1, 1, 1)?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)?,
        None => y,
    })
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let pooled;
        let x = match &self.pool {
            Some(p) => {
                pooled = (x + p.forward(x)?)?;
                &pooled
            }
            None => x,
        };
        let mut h = conv(x, &self.convs[0].0, Some(&self.convs[0].1))?.relu()?;
        h = conv(&h, &self.convs[1].0, Some(&self.convs[1].1))?.relu()?;
        h = conv(&h, &self.convs[2].0, Some(&self.convs[2].1))?;
        Ok((h + x)?.relu()?)
    }
}

enum Layer {
    Conv(Tensor, Option<Tensor>),
    Relu,
    Block(Block),
    Upsample,
}

/// torch `nn.PixelShuffle(2)` on NCHW: channel `c·4 + i·2 + j` → pixel `(2h + i, 2w + j)` of output
/// channel `c`.
fn pixel_shuffle2(x: &Tensor) -> Result<Tensor> {
    let (b, c4, h, w) = x.dims4()?;
    let c = c4 / 4;
    Ok(x.reshape((b, c, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b, c, 2 * h, 2 * w))?)
}

/// The key every TAESD-family checkpoint in the diffusers layout carries (the first decoder conv).
pub(crate) const DECODER_PROBE_KEY: &str = "decoder.layers.0.weight";

/// The loaded, frozen TAESD-family decoder.
pub struct TinyDecoder {
    layers: Vec<Layer>,
    spec: TinyDecoderSpec,
    cfg: TinyDecoderConfig,
    param_bytes: u64,
    consumed_keys: usize,
}

impl TinyDecoder {
    /// Load the decoder half of a TAESD-family checkpoint directory (f32, on `device`): the
    /// diffusers `diffusion_pytorch_model.safetensors` when present — the taesd/taesdxl repos also
    /// ship raw `taesd.py`-layout `*_encoder`/`*_decoder` files whose keys collide — otherwise every
    /// `*.safetensors` in it (taef2 / taeqi2_1 ship one file).
    pub fn from_dir(
        dir: impl AsRef<Path>,
        cfg: impl Into<TinyDecoderSpec>,
        device: &Device,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let diffusers = dir.join("diffusion_pytorch_model.safetensors");
        if diffusers.is_file() {
            let w = Weights::from_files(&[diffusers], device, DType::F32)?;
            return Self::from_weights(&w, cfg);
        }
        // No diffusers file (taef2 / taeqi2_1 ship one `<variant>.safetensors` in the diffusers
        // key layout): pick the ONE file carrying the decoder probe key; unrelated files are
        // skipped, none or several is a named error.
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                CandleError::Msg(format!(
                    "tiny decoder: cannot read checkpoint directory {}: {e}",
                    dir.display()
                ))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .filter(|p| {
                !p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
            })
            .collect();
        files.sort();
        let mut found: Vec<(std::path::PathBuf, Weights)> = Vec::new();
        for f in files {
            let w = Weights::from_files(std::slice::from_ref(&f), device, DType::F32)?;
            if w.contains(DECODER_PROBE_KEY) {
                found.push((f, w));
            }
        }
        match found.len() {
            1 => Self::from_weights(&found.pop().expect("one").1, cfg),
            0 => Err(CandleError::Msg(format!(
                "tiny decoder: no .safetensors checkpoint in {} carries a diffusers-layout decoder \
                 ({DECODER_PROBE_KEY})",
                dir.display()
            ))),
            _ => Err(CandleError::Msg(format!(
                "tiny decoder: several decoder checkpoints in {}: {:?}",
                dir.display(),
                found
                    .iter()
                    .map(|(p, _)| p.display().to_string())
                    .collect::<Vec<_>>()
            ))),
        }
    }

    /// Build from already-read weights carrying the diffusers `decoder.layers.{i}.…` keys.
    pub fn from_weights(w: &Weights, cfg: impl Into<TinyDecoderSpec>) -> Result<Self> {
        let spec: TinyDecoderSpec = cfg.into();
        struct Take<'a> {
            w: &'a Weights,
            bytes: u64,
            keys: usize,
        }
        impl Take<'_> {
            fn get(&mut self, key: &str) -> Result<Tensor> {
                let a = self.w.require(key)?.to_dtype(DType::F32)?;
                self.bytes += (a.elem_count() * 4) as u64;
                self.keys += 1;
                Ok(a)
            }
            fn conv(&mut self, idx: usize, bias: bool) -> Result<Layer> {
                let weight = self.get(&format!("decoder.layers.{idx}.weight"))?;
                let b = if bias {
                    Some(self.get(&format!("decoder.layers.{idx}.bias"))?)
                } else {
                    None
                };
                Ok(Layer::Conv(weight, b))
            }
            fn leg(&mut self, idx: usize, j: usize) -> Result<(Tensor, Tensor)> {
                let p = format!("decoder.layers.{idx}.conv.{j}");
                Ok((
                    self.get(&format!("{p}.weight"))?,
                    self.get(&format!("{p}.bias"))?,
                ))
            }
            fn pool(&mut self, idx: usize) -> Result<Pool> {
                let p = format!("decoder.layers.{idx}.pool");
                Ok(Pool {
                    expand: self.get(&format!("{p}.0.weight"))?,
                    gn_weight: self.get(&format!("{p}.1.weight"))?,
                    gn_bias: self.get(&format!("{p}.1.bias"))?,
                    project: self.get(&format!("{p}.3.weight"))?,
                })
            }
        }
        let mut t = Take {
            w,
            bytes: 0,
            keys: 0,
        };
        let mut layers = vec![t.conv(0, true)?, Layer::Relu];
        let mut idx = 2usize;
        let stages = spec.blocks.len();
        for (stage, &n) in spec.blocks.iter().enumerate() {
            for _ in 0..n {
                let convs = [t.leg(idx, 0)?, t.leg(idx, 2)?, t.leg(idx, 4)?];
                let pool = if spec.midblock_gn && stage == 0 {
                    Some(t.pool(idx)?)
                } else {
                    None
                };
                layers.push(Layer::Block(Block { convs, pool }));
                idx += 1;
            }
            let last = stage == stages - 1;
            if !last {
                layers.push(Layer::Upsample);
                idx += 1;
            }
            layers.push(t.conv(idx, last)?);
            idx += 1;
        }
        Ok(Self {
            layers,
            cfg: spec.config(),
            spec,
            param_bytes: t.bytes,
            consumed_keys: t.keys,
        })
    }

    /// The loaded configuration's plain view (`channels` = the last stage's width).
    pub fn config(&self) -> &TinyDecoderConfig {
        &self.cfg
    }

    /// The loaded structural description.
    pub fn spec(&self) -> &TinyDecoderSpec {
        &self.spec
    }

    /// Resident parameter bytes.
    pub fn param_bytes(&self) -> u64 {
        self.param_bytes
    }

    /// How many checkpoint tensors the graph consumed (a layout check against the key count).
    pub fn consumed_keys(&self) -> usize {
        self.consumed_keys
    }

    /// Model-space latents NCHW `[B, C, h, w]` → RGB pixels NHWC `[B, s·h, s·w, 3]` in `[0, 1]`
    /// (f32; `s` = [`TinyDecoderSpec::upscale`]; an RGBA decoder's alpha is dropped).
    /// Differentiable in `latents`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.dims();
        if dims.len() != 4 || dims[1] != self.spec.latent_channels {
            return Err(CandleError::Msg(format!(
                "tiny decoder expects NCHW latents with {} channels, got shape {dims:?}",
                self.spec.latent_channels
            )));
        }
        let x = latents.to_dtype(DType::F32)?;
        let mut h = ((x / 3.0)?.tanh()? * 3.0)?;
        for layer in &self.layers {
            h = match layer {
                Layer::Conv(w, b) => conv(&h, w, b.as_ref())?,
                Layer::Relu => h.relu()?,
                Layer::Block(block) => block.forward(&h)?,
                Layer::Upsample => {
                    let (_, _, hh, ww) = h.dims4()?;
                    h.upsample_nearest2d(hh * 2, ww * 2)?
                }
            };
        }
        if self.spec.pixel_shuffle {
            h = pixel_shuffle2(&h)?;
        }
        if self.spec.image_channels > 3 {
            h = h.narrow(1, 0, 3)?;
        }
        Ok(h.clamp(0f32, 1f32)?.permute((0, 2, 3, 1))?.contiguous()?)
    }
}

/// A complete random-init decoder checkpoint for `cfg` (a [`TinyDecoderConfig`] or
/// [`TinyDecoderSpec`]) in the diffusers key layout (OIHW) — for tests of the perceptual path and
/// the trainers that use it (never real weights). Deterministic in `seed`.
pub fn synthetic_tiny_decoder_weights<C: Clone + Into<TinyDecoderSpec>>(
    cfg: &C,
    seed: u64,
    device: &Device,
) -> Result<Weights> {
    use std::collections::HashMap;
    let spec: TinyDecoderSpec = cfg.clone().into();
    let mut map: HashMap<String, Tensor> = HashMap::new();
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03);
    let mut rnd = |shape: &[usize]| -> Result<Tensor> {
        let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
        let std = (2.0 / fan_in as f32).sqrt();
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                // Sum of 4 uniforms ≈ normal; scaled to unit variance.
                let mut s = 0f32;
                for _ in 0..4 {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    s += ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
                }
                s * (3.0f32).sqrt() * std
            })
            .collect();
        Ok(Tensor::from_vec(v, shape, device)?)
    };
    let s = spec.stage_channels;
    let mut idx = 0usize;
    map.insert(
        format!("decoder.layers.{idx}.weight"),
        rnd(&[s[0], spec.latent_channels, 3, 3])?,
    );
    map.insert(
        format!("decoder.layers.{idx}.bias"),
        rnd(&[s[0], 1])?.reshape(s[0])?,
    );
    idx += 2;
    let stages = spec.blocks.len();
    for (stage, &nb) in spec.blocks.iter().enumerate() {
        let c = s[stage];
        for _ in 0..nb {
            for j in [0, 2, 4] {
                map.insert(
                    format!("decoder.layers.{idx}.conv.{j}.weight"),
                    rnd(&[c, c, 3, 3])?,
                );
                map.insert(
                    format!("decoder.layers.{idx}.conv.{j}.bias"),
                    rnd(&[c, 1])?.reshape(c)?,
                );
            }
            if spec.midblock_gn && stage == 0 {
                let g = 4 * c;
                let p = format!("decoder.layers.{idx}.pool");
                map.insert(format!("{p}.0.weight"), rnd(&[g, c, 1, 1])?);
                map.insert(format!("{p}.1.weight"), (rnd(&[g, 1])?.reshape(g)? + 1.0)?);
                map.insert(format!("{p}.1.bias"), rnd(&[g, 1])?.reshape(g)?);
                map.insert(format!("{p}.3.weight"), rnd(&[c, g, 1, 1])?);
            }
            idx += 1;
        }
        let last = stage == stages - 1;
        if !last {
            idx += 1;
        }
        let out = if last {
            spec.head_channels()
        } else {
            s[stage + 1]
        };
        map.insert(
            format!("decoder.layers.{idx}.weight"),
            rnd(&[out, c, 3, 3])?,
        );
        if last {
            map.insert(
                format!("decoder.layers.{idx}.bias"),
                rnd(&[out, 1])?.reshape(out)?,
            );
        }
        idx += 1;
    }
    Ok(Weights::from_map(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    fn tiny_cfg() -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels: 4,
            channels: 8,
            blocks: [3, 3, 3, 1],
        }
    }

    /// Keys shared by all five published headers.
    const COMMON_KEYS: [&str; 10] = [
        "decoder.layers.0.weight",
        "decoder.layers.0.bias",
        "decoder.layers.2.conv.0.weight",
        "decoder.layers.4.conv.4.bias",
        "decoder.layers.6.weight",
        "decoder.layers.11.weight",
        "decoder.layers.16.weight",
        "decoder.layers.17.conv.2.weight",
        "decoder.layers.18.weight",
        "decoder.layers.18.bias",
    ];

    /// The listed keys exist, the absent ones don't, every generated key is consumed, and the
    /// param count matches the loaded bytes.
    fn assert_layout(spec: TinyDecoderSpec, present: &[&str], absent: &[&str]) -> Weights {
        let w = synthetic_tiny_decoder_weights(&spec, 1, &Device::Cpu).unwrap();
        for key in present {
            assert!(w.contains(key), "missing {key}");
        }
        for key in absent {
            assert!(!w.contains(key), "unexpected {key}");
        }
        let dec = TinyDecoder::from_weights(&w, spec.clone()).unwrap();
        assert_eq!(dec.consumed_keys(), w.keys().count(), "every key consumed");
        assert_eq!(dec.param_bytes(), spec.param_count() * 4);
        w
    }

    fn shape(w: &Weights, key: &str) -> Vec<usize> {
        w.require(key).unwrap().dims().to_vec()
    }

    /// The real TAEF1 key layout: conv 0, blocks 2-4, up 5, conv 6, blocks 7-9, up 10, conv 11,
    /// blocks 12-14, up 15, conv 16, block 17, conv 18 (+bias); every key consumed.
    #[test]
    fn taef1_layout_indices_match_the_diffusers_checkpoint() {
        let cfg = TinyDecoderConfig::taef1();
        assert_layout(
            cfg.clone().into(),
            &COMMON_KEYS,
            &["decoder.layers.6.bias", "decoder.layers.2.pool.0.weight"],
        );
        let params = 16 * 64 * 9 + 64 + 30 * (64 * 64 * 9 + 64) + 3 * 64 * 64 * 9 + 3 * 64 * 9 + 3;
        assert_eq!(cfg.param_count(), params as u64);
    }

    /// taesdxl / taesd3 (`diffusion_pytorch_model.safetensors`, headers at b20258aa / d58dcacc):
    /// the plain layout with `decoder.layers.0.weight` `[64, 4, 3, 3]` / `[64, 16, 3, 3]`.
    #[test]
    fn taesdxl_and_taesd3_layouts_match_the_published_headers() {
        for (spec, latent) in [
            (TinyDecoderSpec::taesdxl(), 4),
            (TinyDecoderSpec::taesd3(), 16),
        ] {
            let w = assert_layout(spec, &COMMON_KEYS, &["decoder.layers.6.bias"]);
            assert_eq!(shape(&w, "decoder.layers.0.weight"), [64, latent, 3, 3]);
        }
        assert_eq!(TinyDecoderConfig::taesdxl().latent_channels, 4);
        assert_eq!(TinyDecoderConfig::taesd3().upscale(), 8);
    }

    /// taef2 (`taef2.safetensors`, header at bd244ebf): plain layout + `decoder.layers.{2,3,4}.pool`
    /// (`pool.0` `[256, 64, 1, 1]`, `pool.1` GroupNorm `[256]`, `pool.3` `[64, 256, 1, 1]`), conv 0
    /// `[64, 32, 3, 3]`. Mutation: emit the pool on stage 1 ⇒ red.
    #[test]
    fn taef2_layout_matches_the_published_header() {
        let spec = TinyDecoderSpec::taef2();
        let mut present = COMMON_KEYS.to_vec();
        present.extend([
            "decoder.layers.2.pool.0.weight",
            "decoder.layers.2.pool.1.weight",
            "decoder.layers.2.pool.1.bias",
            "decoder.layers.4.pool.3.weight",
        ]);
        let w = assert_layout(
            spec.clone(),
            &present,
            &[
                "decoder.layers.7.pool.0.weight",
                "decoder.layers.2.pool.3.bias",
            ],
        );
        assert_eq!(shape(&w, "decoder.layers.0.weight"), [64, 32, 3, 3]);
        assert_eq!(shape(&w, "decoder.layers.3.pool.0.weight"), [256, 64, 1, 1]);
        assert_eq!(shape(&w, "decoder.layers.3.pool.1.bias"), [256]);
        assert_eq!(shape(&w, "decoder.layers.4.pool.3.weight"), [64, 256, 1, 1]);
        let plain = TinyDecoderSpec {
            midblock_gn: false,
            ..spec.clone()
        };
        assert_eq!(
            spec.param_count() - plain.param_count(),
            3 * (64 * 256 + 2 * 256 + 256 * 64)
        );
    }

    /// taeqi2_1 (`taeqi2_1.safetensors`, f16, header at 379cc3f2): `F16Decoder` — conv 0
    /// `[256, 64, 3, 3]`, stage convs 6 `[128, 256]`, 11 `[64, 128]`, 16 `[64, 64]`, head 18
    /// `[16, 64, 3, 3]` + `[16]`. Mutation: stage widths `[256, 256, 64, 64]` ⇒ red.
    #[test]
    fn taeqi2_1_layout_matches_the_published_header() {
        let spec = TinyDecoderSpec::taeqi2_1();
        let w = assert_layout(spec.clone(), &COMMON_KEYS, &["decoder.layers.6.bias"]);
        for (key, want) in [
            ("decoder.layers.0.weight", vec![256, 64, 3, 3]),
            ("decoder.layers.2.conv.0.weight", vec![256, 256, 3, 3]),
            ("decoder.layers.6.weight", vec![128, 256, 3, 3]),
            ("decoder.layers.7.conv.0.weight", vec![128, 128, 3, 3]),
            ("decoder.layers.11.weight", vec![64, 128, 3, 3]),
            ("decoder.layers.16.weight", vec![64, 64, 3, 3]),
            ("decoder.layers.18.weight", vec![16, 64, 3, 3]),
            ("decoder.layers.18.bias", vec![16]),
        ] {
            assert_eq!(shape(&w, key), want, "{key}");
        }
        assert_eq!(spec.upscale(), 16);
        assert_eq!(spec.config().channels, 64);
    }

    fn grad_sum(dec: &TinyDecoder, z: &Var) -> f32 {
        let px = dec.decode(z.as_tensor()).unwrap();
        let grads = px.sum_all().unwrap().backward().unwrap();
        grads
            .get(z.as_tensor())
            .expect("latents get a gradient")
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Mutation: insert a `.detach()` in `decode` ⇒ no gradient ⇒ red.
    #[test]
    fn decode_upscales_by_eight_into_unit_range_and_is_differentiable() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let dec =
            TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 2, &dev).unwrap(), cfg)
                .unwrap();
        let z = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 4, 3, 2), &dev).unwrap()).unwrap();
        let px = dec.decode(z.as_tensor()).unwrap();
        assert_eq!(px.dims(), &[1, 24, 16, 3]);
        let v = px.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(grad_sum(&dec, &z) > 0.0);
    }

    /// A tiny F16 decoder upscales 16× and returns RGB only, differentiably. Mutations: drop the
    /// pixel shuffle ⇒ 8× ⇒ red; drop the RGB narrow ⇒ 4 channels ⇒ red.
    #[test]
    fn f16_decode_upscales_by_sixteen_to_rgb() {
        let spec = TinyDecoderSpec {
            latent_channels: 8,
            stage_channels: [16, 8, 8, 8],
            ..TinyDecoderSpec::taeqi2_1()
        };
        let dev = Device::Cpu;
        let dec = TinyDecoder::from_weights(
            &synthetic_tiny_decoder_weights(&spec, 7, &dev).unwrap(),
            spec,
        )
        .unwrap();
        let z = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 8, 2, 3), &dev).unwrap()).unwrap();
        let px = dec.decode(z.as_tensor()).unwrap();
        assert_eq!(px.dims(), &[1, 32, 48, 3]);
        let v = px.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(grad_sum(&dec, &z) > 0.0);
    }

    /// The f16 checkpoint loads as f32 (param bytes counted at f32, decode output f32).
    #[test]
    fn f16_checkpoints_load_as_f32() {
        let spec = TinyDecoderSpec {
            latent_channels: 8,
            stage_channels: [16, 8, 8, 8],
            ..TinyDecoderSpec::taeqi2_1()
        };
        let dev = Device::Cpu;
        let src = synthetic_tiny_decoder_weights(&spec, 6, &dev).unwrap();
        let half: std::collections::HashMap<String, Tensor> = src
            .keys()
            .map(|k| {
                (
                    k.to_string(),
                    src.require(k).unwrap().to_dtype(DType::F16).unwrap(),
                )
            })
            .collect();
        let dec = TinyDecoder::from_weights(&Weights::from_map(half), spec.clone()).unwrap();
        assert_eq!(dec.param_bytes(), spec.param_count() * 4);
        let z = Tensor::randn(0f32, 1f32, (1, 8, 2, 2), &dev).unwrap();
        assert_eq!(dec.decode(&z).unwrap().dtype(), DType::F32);
    }

    /// `pixel_shuffle2` is torch `PixelShuffle(2)`. Mutation: swap the `i`/`j` axes ⇒ red.
    #[test]
    fn pixel_shuffle_matches_torch_channel_order() {
        let x = Tensor::from_vec(
            (0..8).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 8, 1, 1),
            &Device::Cpu,
        )
        .unwrap();
        let y = pixel_shuffle2(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 2, 2]);
        // y[0, c, i, j] = c·4 + i·2 + j.
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        v[c * 4 + i * 2 + j],
                        (c * 4 + i * 2 + j) as f32,
                        "({c},{i},{j})"
                    );
                }
            }
        }
        // A non-trivial spatial case: the shuffle of an NCHW [1, 4, 1, 2] input.
        let x = Tensor::from_vec(
            (0..8).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 4, 1, 2),
            &Device::Cpu,
        )
        .unwrap();
        let y = pixel_shuffle2(&x).unwrap();
        // out[0, 0, i, 2w + j] = in[0, i·2 + j, 0, w] = (i·2 + j)·2 + w.
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![0., 2., 1., 3., 4., 6., 5., 7.]);
    }

    /// The mid-block pool participates and stays differentiable. Mutation: skip `x += pool(x)` ⇒
    /// red.
    #[test]
    fn midblock_pool_changes_the_decode_and_is_differentiable() {
        let spec = TinyDecoderSpec {
            latent_channels: 4,
            stage_channels: [8; 4],
            ..TinyDecoderSpec::taef2()
        };
        let dev = Device::Cpu;
        let w = synthetic_tiny_decoder_weights(&spec, 9, &dev).unwrap();
        let with = TinyDecoder::from_weights(&w, spec.clone()).unwrap();
        let without = TinyDecoder::from_weights(
            &w,
            TinyDecoderSpec {
                midblock_gn: false,
                ..spec
            },
        )
        .unwrap();
        let z = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 4, 3, 3), &dev).unwrap()).unwrap();
        let a = with.decode(z.as_tensor()).unwrap();
        let b = without.decode(z.as_tensor()).unwrap();
        assert_eq!(a.dims(), &[1, 24, 24, 3]);
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0, "pool had no effect");
        assert!(grad_sum(&with, &z) > 0.0);
    }

    /// The composed GroupNorm normalizes each contiguous channel group over (group channels, H, W)
    /// like torch. Mutation: normalize per channel (groups = C) ⇒ red.
    #[test]
    fn group_norm_matches_torch_semantics() {
        // NCHW [1, 4, 1, 2], 2 groups: group 0 = channels {0,1} = {1,3,5,7}; group 1 = {10,10,20,40}.
        let x = Tensor::from_vec(
            vec![1f32, 5., 3., 7., 10., 20., 10., 40.],
            (1, 4, 1, 2),
            &Device::Cpu,
        )
        .unwrap();
        let ones = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let zeros = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();
        let v = group_norm_nchw(&x, &ones, &zeros, 2, 0.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s = 5f32.sqrt();
        for (got, want) in [
            (v[0], -3.0 / s),
            (v[1], 1.0 / s),
            (v[2], -1.0 / s),
            (v[3], 3.0 / s),
        ] {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        let s1 = 150f32.sqrt();
        assert!((v[4] - (-10.0 / s1)).abs() < 1e-5);
        assert!((v[7] - (20.0 / s1)).abs() < 1e-5);
    }

    #[test]
    fn decode_refuses_the_wrong_latent_layout() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let dec =
            TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 2, &dev).unwrap(), cfg)
                .unwrap();
        assert!(dec
            .decode(&Tensor::zeros((1, 3, 2, 2), DType::F32, &dev).unwrap())
            .is_err());
    }

    #[test]
    fn missing_checkpoint_dir_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(TinyDecoder::from_dir(tmp.path(), tiny_cfg(), &Device::Cpu).is_err());
        assert!(TinyDecoder::from_dir(tmp.path().join("nope"), tiny_cfg(), &Device::Cpu).is_err());
    }

    /// `from_dir` reads the diffusers file when the dir also holds colliding raw-layout files.
    /// Mutation: always read every file ⇒ duplicate-key error ⇒ red.
    #[test]
    fn from_dir_prefers_the_diffusers_file() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let w = synthetic_tiny_decoder_weights(&cfg, 2, &dev).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let map: std::collections::HashMap<String, Tensor> = w
            .keys()
            .map(|k| (k.to_string(), w.require(k).unwrap().clone()))
            .collect();
        candle_core::safetensors::save(
            &map,
            tmp.path().join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        let raw: std::collections::HashMap<String, Tensor> = [(
            "1.weight".to_string(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        )]
        .into();
        candle_core::safetensors::save(&raw, tmp.path().join("taesd_encoder.safetensors")).unwrap();
        candle_core::safetensors::save(&raw, tmp.path().join("taesd_decoder.safetensors")).unwrap();
        assert!(TinyDecoder::from_dir(tmp.path(), cfg, &dev).is_ok());
    }

    #[test]
    fn working_set_scales_with_output_area() {
        for spec in [
            TinyDecoderSpec::taef1(),
            TinyDecoderSpec::taef2(),
            TinyDecoderSpec::taeqi2_1(),
        ] {
            let a = spec.training_working_set_bytes(512, 512);
            assert_eq!(
                spec.training_working_set_bytes(1024, 1024),
                a * 4,
                "{spec:?}"
            );
            assert!(a > 0);
        }
        let cfg = TinyDecoderConfig::taef1();
        assert_eq!(cfg.footprint(512, 512).param_bytes, cfg.param_count() * 4);
        assert_eq!(
            cfg.training_working_set_bytes(512, 512),
            TinyDecoderSpec::taef1().training_working_set_bytes(512, 512)
        );
    }

    /// The taef2 / taeqi2_1 snapshot shape (one `<variant>.safetensors`, diffusers decoder keys plus
    /// `encoder.*`, no diffusers file): `from_dir` picks the decoder file even with an unrelated
    /// `.safetensors` beside it (sharing a key, so merging the dir would collide), and refuses a dir
    /// with no decoder file or two. Mutations: read every file ⇒ collision ⇒ red; accept the first
    /// file instead of the probed one ⇒ `aaa_unrelated` is picked ⇒ red.
    #[test]
    fn from_dir_picks_the_variant_file_among_unrelated_safetensors() {
        let spec = TinyDecoderSpec {
            latent_channels: 4,
            stage_channels: [8; 4],
            ..TinyDecoderSpec::taef2()
        };
        let dev = Device::Cpu;
        let w = synthetic_tiny_decoder_weights(&spec, 3, &dev).unwrap();
        let mut map: std::collections::HashMap<String, Tensor> = w
            .keys()
            .map(|k| (k.to_string(), w.require(k).unwrap()))
            .collect();
        map.insert(
            "encoder.0.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        let tmp = tempfile::tempdir().unwrap();
        let unrelated: std::collections::HashMap<String, Tensor> = [(
            "encoder.0.weight".to_string(),
            Tensor::zeros(2, DType::F32, &dev).unwrap(),
        )]
        .into();
        candle_core::safetensors::save(&unrelated, tmp.path().join("aaa_unrelated.safetensors"))
            .unwrap();
        let none = TinyDecoder::from_dir(tmp.path(), spec.clone(), &dev)
            .err()
            .expect("no decoder file")
            .to_string();
        assert!(none.contains("decoder.layers.0.weight"), "{none}");
        candle_core::safetensors::save(&map, tmp.path().join("taef2.safetensors")).unwrap();
        let dec = TinyDecoder::from_dir(tmp.path(), spec.clone(), &dev).unwrap();
        assert_eq!(dec.spec(), &spec);
        candle_core::safetensors::save(&map, tmp.path().join("taef2_copy.safetensors")).unwrap();
        let two = TinyDecoder::from_dir(tmp.path(), spec, &dev)
            .err()
            .expect("ambiguous")
            .to_string();
        assert!(two.contains("several"), "{two}");
    }
}

//! **Tiny AutoEncoder decoder** (TAESD architecture, madebyollin/taesd) — the small, frozen,
//! fully differentiable latent → pixel decoder the shared perceptual-loss path
//! ([`super::perceptual`]) runs a trainer's x0 prediction through (epic 2123 E8, sc-2125).
//!
//! One module graph covers the whole TAESD family; only the latent channel count differs:
//! TAEF1 (`madebyollin/taef1`, the FLUX.1 16-channel latent API — Z-Image, Chroma, Flex), TAESD
//! (SD 4-channel), TAESDXL (SDXL 4-channel). The decoder consumes the model's **normalized sampler
//! latent directly** — TAESD-family checkpoints ship `scaling_factor = 1.0` / `shift_factor = 0.0`,
//! i.e. they decode the diffusion model's own latent space (exactly what the diffusers
//! `FluxPipeline` hands an `AutoencoderTiny`), so no de-normalization is applied here.
//!
//! Architecture (diffusers `DecoderTiny`, `num_decoder_blocks = [3, 3, 3, 1]`):
//! `tanh(x/3)·3` → conv3×3(latent→C)+ReLU → 3×Block → ↑2 → conv3×3(no bias) → 3×Block → ↑2 →
//! conv3×3 → 3×Block → ↑2 → conv3×3 → 1×Block → conv3×3(C→3). A Block is
//! `ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x)`. The raw output is pixels in `[0, 1]` (diffusers'
//! `AutoencoderTiny.decode` then maps it to `[-1, 1]`; we keep `[0, 1]`, clamped, which is what every
//! perceptual model consumes).
//!
//! Weight keys: the diffusers layout `decoder.layers.{i}.…` (what `diffusion_pytorch_model.safetensors`
//! ships); torch OIHW conv weights are permuted to MLX OHWI at load.

use mlx_rs::ops::{add, clip, multiply, tanh};
use mlx_rs::{random, Array};

use crate::nn::{conv2d, upsample_nearest};
use crate::weights::Weights;
use crate::{Error, Result};

/// TAESD-family decoder hyperparameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyDecoderConfig {
    /// Latent channels the decoder consumes (16 for TAEF1, 4 for TAESD/TAESDXL).
    pub latent_channels: i32,
    /// Hidden width of every stage (64 for every shipped checkpoint).
    pub channels: i32,
    /// Blocks per stage (`[3, 3, 3, 1]` for every shipped checkpoint); one ×2 upsample between
    /// consecutive stages, so the decode upscales by `2^(len-1)` (8).
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

    /// Spatial upscale factor latent → pixels (8).
    pub fn upscale(&self) -> i32 {
        1 << (self.blocks.len() - 1)
    }

    /// Exact parameter count of the decoder graph (for estimating memory before loading).
    pub fn param_count(&self) -> u64 {
        let c = self.channels as u64;
        let conv = |i: u64, o: u64, bias: bool| o * i * 9 + if bias { o } else { 0 };
        let blocks: u64 = self.blocks.iter().map(|&n| n as u64).sum();
        let stage_convs = (self.blocks.len() - 1) as u64;
        conv(self.latent_channels as u64, c, true)
            + blocks * 3 * conv(c, c, true)
            + stage_convs * conv(c, c, false)
            + conv(c, 3, true)
    }

    /// Conservative upper bound on the **training working set** of one differentiable decode to an
    /// `out_h × out_w` image, in bytes: every intermediate the backward retains (each Block keeps
    /// its three conv outputs, two ReLUs, the residual sum and the fused ReLU; each stage keeps its
    /// upsample + conv), f32, ×2 for the matching cotangent buffers in the backward. Used by the
    /// trainer memory estimate (epic 2123 E7); not a measured value.
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        let per_block = 7u64;
        let c = self.channels as u64;
        let full = out_h as u64 * out_w as u64;
        let stages = self.blocks.len();
        let mut tensors_x_pixels = 0u64;
        for (i, &n) in self.blocks.iter().enumerate() {
            // Stage i runs at full / 4^(stages-1-i) pixels.
            let shrink = 1u64 << (2 * (stages - 1 - i));
            let stage_tensors = per_block * n as u64 + 2; // + the stage conv and its upsample
            tensors_x_pixels += stage_tensors * full / shrink;
        }
        tensors_x_pixels * c * 4 * 2
    }
}

/// One TAESD residual block: three 3×3 convs (ReLU between), identity skip, fused ReLU.
struct Block {
    convs: [(Array, Array); 3],
}

impl Block {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = conv2d(x, &self.convs[0].0, Some(&self.convs[0].1), 1, 1)?;
        h = mlx_rs::nn::relu(&h)?;
        h = conv2d(&h, &self.convs[1].0, Some(&self.convs[1].1), 1, 1)?;
        h = mlx_rs::nn::relu(&h)?;
        h = conv2d(&h, &self.convs[2].0, Some(&self.convs[2].1), 1, 1)?;
        Ok(mlx_rs::nn::relu(&add(&h, x)?)?)
    }
}

enum Layer {
    /// 3×3 conv with optional bias.
    Conv(Array, Option<Array>),
    Relu,
    Block(Block),
    Upsample,
}

/// The loaded, frozen TAESD-family decoder.
pub struct TinyDecoder {
    layers: Vec<Layer>,
    cfg: TinyDecoderConfig,
    param_bytes: u64,
}

/// Permute a torch conv weight `[out, in, kH, kW]` → MLX `[out, kH, kW, in]`.
fn ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

impl TinyDecoder {
    /// Load the decoder half of a diffusers `AutoencoderTiny` checkpoint directory
    /// (`diffusion_pytorch_model.safetensors`).
    pub fn from_dir(dir: impl AsRef<std::path::Path>, cfg: TinyDecoderConfig) -> Result<Self> {
        let w = Weights::from_dir(dir)?;
        Self::from_weights(&w, cfg)
    }

    /// Build from already-read weights carrying the diffusers `decoder.layers.{i}.…` keys.
    pub fn from_weights(w: &Weights, cfg: TinyDecoderConfig) -> Result<Self> {
        struct Take<'a> {
            w: &'a Weights,
            bytes: u64,
        }
        impl Take<'_> {
            fn get(&mut self, key: &str) -> Result<Array> {
                let a = self.w.require(key)?.clone();
                self.bytes += a.nbytes() as u64;
                Ok(a)
            }
            fn conv(&mut self, idx: usize, bias: bool) -> Result<Layer> {
                let weight = ohwi(&self.get(&format!("decoder.layers.{idx}.weight"))?)?;
                let b = if bias {
                    Some(self.get(&format!("decoder.layers.{idx}.bias"))?)
                } else {
                    None
                };
                Ok(Layer::Conv(weight, b))
            }
            fn leg(&mut self, idx: usize, j: usize) -> Result<(Array, Array)> {
                let p = format!("decoder.layers.{idx}.conv.{j}");
                Ok((
                    ohwi(&self.get(&format!("{p}.weight"))?)?,
                    self.get(&format!("{p}.bias"))?,
                ))
            }
        }
        let mut t = Take { w, bytes: 0 };
        let mut layers = vec![t.conv(0, true)?, Layer::Relu];
        let mut idx = 2usize;
        let stages = cfg.blocks.len();
        for (stage, &n) in cfg.blocks.iter().enumerate() {
            for _ in 0..n {
                let convs = [t.leg(idx, 0)?, t.leg(idx, 2)?, t.leg(idx, 4)?];
                layers.push(Layer::Block(Block { convs }));
                idx += 1;
            }
            let last = stage == stages - 1;
            if !last {
                layers.push(Layer::Upsample);
                idx += 1;
            }
            // Stage-closing conv: bias only on the final (C→3) projection.
            layers.push(t.conv(idx, last)?);
            idx += 1;
        }
        Ok(Self {
            layers,
            cfg,
            param_bytes: t.bytes,
        })
    }

    /// The loaded configuration.
    pub fn config(&self) -> &TinyDecoderConfig {
        &self.cfg
    }

    /// Resident parameter bytes.
    pub fn param_bytes(&self) -> u64 {
        self.param_bytes
    }

    /// Model-space latents NCHW `[B, C, h, w]` → pixels NHWC `[B, 8h, 8w, 3]` in `[0, 1]`.
    /// Pure MLX ops end to end (no host round trip, no stop-gradient), so it is differentiable in
    /// `latents`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        if sh.len() != 4 || sh[1] != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "tiny decoder expects NCHW latents with {} channels, got shape {sh:?}",
                self.cfg.latent_channels
            )));
        }
        let x = latents.transpose_axes(&[0, 2, 3, 1])?;
        // Clamp(): tanh(x / 3) * 3.
        let three = Array::from_f32(3.0);
        let mut h = multiply(&tanh(&mlx_rs::ops::divide(&x, &three)?)?, &three)?;
        for layer in &self.layers {
            h = match layer {
                Layer::Conv(w, b) => conv2d(&h, w, b.as_ref(), 1, 1)?,
                Layer::Relu => mlx_rs::nn::relu(&h)?,
                Layer::Block(block) => block.forward(&h)?,
                Layer::Upsample => upsample_nearest(&h, 2)?,
            };
        }
        Ok(clip(&h, (&Array::from_f32(0.0), &Array::from_f32(1.0)))?)
    }
}

/// A complete random-init decoder checkpoint for `cfg` in the diffusers key layout (torch OIHW
/// conv weights) — for tests of the shared perceptual path and the trainers that use it, which
/// must never download real weights. Deterministic in `seed`.
pub fn synthetic_tiny_decoder_weights(cfg: &TinyDecoderConfig, seed: u64) -> Result<Weights> {
    let mut w = Weights::empty();
    let mut n = 0u64;
    let mut rnd = |shape: &[i32]| -> Result<Array> {
        n += 1;
        let fan_in: i32 = shape[1..].iter().product();
        let std = (2.0 / fan_in as f32).sqrt();
        let key = random::key(seed.wrapping_mul(1_000_003).wrapping_add(n))?;
        Ok(multiply(
            &random::normal::<f32>(shape, None, None, Some(&key))?,
            Array::from_f32(std),
        )?)
    };
    let c = cfg.channels;
    let mut idx = 0usize;
    w.insert(
        format!("decoder.layers.{idx}.weight"),
        rnd(&[c, cfg.latent_channels, 3, 3])?,
    );
    w.insert(
        format!("decoder.layers.{idx}.bias"),
        rnd(&[c, 1])?.reshape(&[c])?,
    );
    idx += 2; // conv + relu
    let stages = cfg.blocks.len();
    for (stage, &nb) in cfg.blocks.iter().enumerate() {
        for _ in 0..nb {
            for j in [0, 2, 4] {
                w.insert(
                    format!("decoder.layers.{idx}.conv.{j}.weight"),
                    rnd(&[c, c, 3, 3])?,
                );
                w.insert(
                    format!("decoder.layers.{idx}.conv.{j}.bias"),
                    rnd(&[c, 1])?.reshape(&[c])?,
                );
            }
            idx += 1;
        }
        let last = stage == stages - 1;
        if !last {
            idx += 1; // upsample
        }
        let out = if last { 3 } else { c };
        w.insert(
            format!("decoder.layers.{idx}.weight"),
            rnd(&[out, c, 3, 3])?,
        );
        if last {
            w.insert(
                format!("decoder.layers.{idx}.bias"),
                rnd(&[out, 1])?.reshape(&[out])?,
            );
        }
        idx += 1;
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::{eval, grad};

    fn tiny_cfg() -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels: 4,
            channels: 8,
            blocks: [3, 3, 3, 1],
        }
    }

    #[test]
    fn taef1_layout_indices_match_the_diffusers_checkpoint() {
        // The real TAEF1 safetensors keys: conv 0, blocks 2-4, up 5, conv 6, blocks 7-9, up 10,
        // conv 11, blocks 12-14, up 15, conv 16, block 17, conv 18 (+bias). Every key the synthetic
        // builder emits must be consumed by the loader and vice versa.
        let cfg = TinyDecoderConfig::taef1();
        let w = synthetic_tiny_decoder_weights(&cfg, 1).unwrap();
        for key in [
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
        ] {
            assert!(w.get(key).is_some(), "missing {key}");
        }
        assert!(
            w.get("decoder.layers.6.bias").is_none(),
            "stage convs carry no bias"
        );
        let dec = TinyDecoder::from_weights(&w, cfg).unwrap();
        assert!(w.unused_keys().is_empty(), "unused: {:?}", w.unused_keys());
        // 16·64·9+64 + 10 blocks·3·(64·64·9+64) + 3·64·64·9 + 3·64·9+3 params, f32.
        let params = 16 * 64 * 9 + 64 + 30 * (64 * 64 * 9 + 64) + 3 * 64 * 64 * 9 + 3 * 64 * 9 + 3;
        assert_eq!(dec.param_bytes(), params as u64 * 4);
        assert_eq!(dec.config().param_count(), params as u64);
    }

    #[test]
    fn decode_upscales_by_eight_into_unit_range_and_is_differentiable() {
        let cfg = tiny_cfg();
        let dec = TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 2).unwrap(), cfg)
            .unwrap();
        let z = random::normal::<f32>(&[1, 4, 3, 2], None, None, Some(&random::key(5).unwrap()))
            .unwrap();
        let px = dec.decode(&z).unwrap();
        assert_eq!(px.shape(), &[1, 24, 16, 3]);
        eval([&px]).unwrap();
        let v = px.as_slice::<f32>();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        // d(sum(pixels))/d(latents) flows (no host round trip / stop-gradient in the graph).
        let f = |z: &Array| -> mlx_rs::error::Result<Array> {
            dec.decode(z)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
                .sum(None)
        };
        let g = grad(f)(&z).unwrap();
        eval([&g]).unwrap();
        assert!(g.abs().unwrap().sum(None).unwrap().item::<f32>() > 0.0);
    }

    #[test]
    fn decode_refuses_the_wrong_latent_layout() {
        let cfg = tiny_cfg();
        let dec = TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 2).unwrap(), cfg)
            .unwrap();
        let z = Array::zeros::<f32>(&[1, 3, 2, 2]).unwrap();
        assert!(dec.decode(&z).is_err());
    }

    #[test]
    fn working_set_scales_with_output_area() {
        let cfg = TinyDecoderConfig::taef1();
        let a = cfg.training_working_set_bytes(512, 512);
        let b = cfg.training_working_set_bytes(1024, 1024);
        assert_eq!(b, a * 4);
        assert!(a > 0);
    }
}

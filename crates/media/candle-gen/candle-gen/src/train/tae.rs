//! **Tiny AutoEncoder decoder** (TAESD architecture, madebyollin/taesd) for the Candle trainers —
//! the Candle twin of `mlx_gen::train::tae` (sc-2125): the small, frozen, fully differentiable
//! latent → pixel decoder the shared perceptual path ([`super::perceptual`]) runs a trainer's x0
//! prediction through (epic 2123 E8, sc-24830).
//!
//! One module graph covers the TAESD family; the variant constructors on [`TinyDecoderConfig`] fix
//! the latent channel count. The decoder consumes the diffusion model's **normalized sampler latent**
//! directly (TAESD-family checkpoints decode the model's own latent space), so no de-normalization
//! is applied here.
//!
//! Architecture (diffusers `DecoderTiny`, `num_decoder_blocks = [3, 3, 3, 1]`):
//! `tanh(x/3)·3` → conv3×3(latent→C)+ReLU → 3×Block → ↑2 → conv3×3(no bias) → 3×Block → ↑2 →
//! conv3×3 → 3×Block → ↑2 → conv3×3 → 1×Block → conv3×3(C→3), Block =
//! `ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x)`. Raw output ≈ `[0, 1]`, clamped.
//!
//! Every op used here (conv2d, nearest upsample, relu, tanh, clamp) has a candle backward, so the
//! decode is differentiable in the latents. Weights: the diffusers `decoder.layers.{i}.…` layout,
//! torch OIHW conv kernels (candle-native, no permute).

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::perceptual::AuxModelFootprint;
use crate::weights::Weights;
use crate::{CandleError, Result};

/// TAESD-family decoder hyperparameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyDecoderConfig {
    /// Latent channels the decoder consumes (16 for TAEF1).
    pub latent_channels: usize,
    /// Hidden width of every stage (64 for every shipped checkpoint).
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

    /// Spatial upscale factor latent → pixels (8).
    pub fn upscale(&self) -> usize {
        1 << (self.blocks.len() - 1)
    }

    /// Exact parameter count of the decoder graph.
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
    /// `out_h × out_w` image, in bytes (same accounting as the MLX twin; not a measured value).
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        let per_block = 7u64;
        let c = self.channels as u64;
        let full = out_h as u64 * out_w as u64;
        let stages = self.blocks.len();
        let mut tensors_x_pixels = 0u64;
        for (i, &n) in self.blocks.iter().enumerate() {
            let shrink = 1u64 << (2 * (stages - 1 - i));
            let stage_tensors = per_block * n as u64 + 2;
            tensors_x_pixels += stage_tensors * full / shrink;
        }
        tensors_x_pixels * c * 4 * 2
    }
}

struct Block {
    convs: [(Tensor, Tensor); 3],
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

/// The loaded, frozen TAESD-family decoder.
pub struct TinyDecoder {
    layers: Vec<Layer>,
    cfg: TinyDecoderConfig,
    param_bytes: u64,
    consumed_keys: usize,
}

impl TinyDecoder {
    /// Load the decoder half of a diffusers `AutoencoderTiny` checkpoint directory (every
    /// `*.safetensors` in it, f32, on `device`).
    pub fn from_dir(
        dir: impl AsRef<Path>,
        cfg: TinyDecoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                CandleError::Msg(format!(
                    "tiny decoder: cannot read checkpoint directory {}: {e}",
                    dir.display()
                ))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "tiny decoder: no .safetensors checkpoint in {}",
                dir.display()
            )));
        }
        let w = Weights::from_files(&files, device, DType::F32)?;
        Self::from_weights(&w, cfg)
    }

    /// Build from already-read weights carrying the diffusers `decoder.layers.{i}.…` keys.
    pub fn from_weights(w: &Weights, cfg: TinyDecoderConfig) -> Result<Self> {
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
        }
        let mut t = Take {
            w,
            bytes: 0,
            keys: 0,
        };
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
            layers.push(t.conv(idx, last)?);
            idx += 1;
        }
        Ok(Self {
            layers,
            cfg,
            param_bytes: t.bytes,
            consumed_keys: t.keys,
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

    /// How many checkpoint tensors the graph consumed (a layout check against the key count).
    pub fn consumed_keys(&self) -> usize {
        self.consumed_keys
    }

    /// Model-space latents NCHW `[B, C, h, w]` → pixels NHWC `[B, 8h, 8w, 3]` in `[0, 1]` (f32).
    /// Differentiable in `latents`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.dims();
        if dims.len() != 4 || dims[1] != self.cfg.latent_channels {
            return Err(CandleError::Msg(format!(
                "tiny decoder expects NCHW latents with {} channels, got shape {dims:?}",
                self.cfg.latent_channels
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
        Ok(h.clamp(0f32, 1f32)?.permute((0, 2, 3, 1))?.contiguous()?)
    }
}

/// A complete random-init decoder checkpoint for `cfg` in the diffusers key layout (OIHW) — for
/// tests of the perceptual path and the trainers that use it (never real weights). Deterministic in
/// `seed`.
pub fn synthetic_tiny_decoder_weights(
    cfg: &TinyDecoderConfig,
    seed: u64,
    device: &Device,
) -> Result<Weights> {
    use std::collections::HashMap;
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
    let c = cfg.channels;
    let mut idx = 0usize;
    map.insert(
        format!("decoder.layers.{idx}.weight"),
        rnd(&[c, cfg.latent_channels, 3, 3])?,
    );
    map.insert(
        format!("decoder.layers.{idx}.bias"),
        rnd(&[c, 1])?.reshape(c)?,
    );
    idx += 2;
    let stages = cfg.blocks.len();
    for (stage, &nb) in cfg.blocks.iter().enumerate() {
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
            idx += 1;
        }
        let last = stage == stages - 1;
        if !last {
            idx += 1;
        }
        let out = if last { 3 } else { c };
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

    /// The real TAEF1 key layout: conv 0, blocks 2-4, up 5, conv 6, blocks 7-9, up 10, conv 11,
    /// blocks 12-14, up 15, conv 16, block 17, conv 18 (+bias); every key consumed.
    #[test]
    fn taef1_layout_indices_match_the_diffusers_checkpoint() {
        let cfg = TinyDecoderConfig::taef1();
        let w = synthetic_tiny_decoder_weights(&cfg, 1, &Device::Cpu).unwrap();
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
            assert!(w.contains(key), "missing {key}");
        }
        assert!(
            !w.contains("decoder.layers.6.bias"),
            "stage convs carry no bias"
        );
        let dec = TinyDecoder::from_weights(&w, cfg).unwrap();
        assert_eq!(dec.consumed_keys(), w.keys().count(), "every key consumed");
        let params = 16 * 64 * 9 + 64 + 30 * (64 * 64 * 9 + 64) + 3 * 64 * 64 * 9 + 3 * 64 * 9 + 3;
        assert_eq!(dec.param_bytes(), params as u64 * 4);
        assert_eq!(dec.config().param_count(), params as u64);
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
        let grads = px.sum_all().unwrap().backward().unwrap();
        let g = grads.get(z.as_tensor()).expect("latents get a gradient");
        let s = g
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(s > 0.0, "grad |Σ| {s}");
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

    #[test]
    fn working_set_scales_with_output_area() {
        let cfg = TinyDecoderConfig::taef1();
        let a = cfg.training_working_set_bytes(512, 512);
        assert_eq!(cfg.training_working_set_bytes(1024, 1024), a * 4);
        assert!(a > 0);
        assert_eq!(cfg.footprint(512, 512).param_bytes, cfg.param_count() * 4);
    }
}

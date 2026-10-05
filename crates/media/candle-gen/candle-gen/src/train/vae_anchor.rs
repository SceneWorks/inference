//! **VAE perceptual anchor** (epic 2123, sc-24833) for the Candle trainers — the Candle twin of
//! `mlx_gen::train::vae_anchor`, a port of ai-toolkit-perceptual `toolkit/vae_anchor.py` (fork
//! commit 6e01a6e): a frozen FLUX.2 VAE **encoder** as a multi-scale perceptual discriminator on the
//! shared decoded-x0 path ([`super::perceptual`]).
//!
//! Decoded pixels (`[0, 1]`) are mapped to `[-1, 1]` and run through the FLUX.2 VAE encoder up to its
//! mid block; five feature maps are tapped — each resolution level's last resnet output (before the
//! downsample) and the mid block's second resnet — and compared with the same taps of the training
//! image's clean round trip (computed once per image, cached f16 like upstream) by the mean over
//! positions of `1 − cos` across channels, combined `(4·L0 + 2·L1 + L2 + L3 + Lmid) / 5`.
//!
//! Every op has a candle backward: GroupNorm, SiLU and softmax are composed from elementary ops
//! (candle's fused kernels have no backward), and the stride-2 downsample crops its padded input to
//! the extent the windows read (candle's Conv2D backward derives `output_padding` from the height
//! alone, so unequal row/column remainders would break it). Weights: the diffusers
//! `AutoencoderKLFlux2` layout (only the tapped `encoder.*` sub-graph is read).

use std::any::Any;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{DType, Device, Tensor, D};

use super::formula::{formula_tensor, Role};
use super::perceptual::{
    reference_as, AuxModelFootprint, LossReference, PerceptualInput, PerceptualLoss,
};
use crate::weights::Weights;
use crate::{CandleError, Result};

/// The five tap names, in [`VaeAnchorEncoder::features`] order.
pub const VAE_ANCHOR_LEVELS: [&str; 5] = ["level_0", "level_1", "level_2", "level_3", "mid"];
/// Upstream `compute_loss` default per-level weights.
pub const VAE_ANCHOR_LEVEL_WEIGHTS: [f64; 5] = [4.0, 2.0, 1.0, 1.0, 1.0];
const GN_GROUPS: usize = 32;
const GN_EPS: f64 = 1e-6;
const COS_EPS: f64 = 1e-8;
const RESNETS_PER_LEVEL: usize = 2;

/// Width of the FLUX.2 VAE encoder whose features the loss taps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaeAnchorEncoderConfig {
    /// Output channels of the four resolution levels.
    pub block_out: [usize; 4],
}

impl VaeAnchorEncoderConfig {
    /// The FLUX.2 VAE (`ch = 128`, `ch_mult = [1, 2, 4, 4]`).
    pub fn flux2() -> Self {
        Self {
            block_out: [128, 256, 512, 512],
        }
    }

    /// The same graph at `ch` base width — the parity fixture uses 32.
    pub fn with_base(ch: usize) -> Self {
        Self {
            block_out: [ch, 2 * ch, 4 * ch, 4 * ch],
        }
    }

    fn tensors(&self) -> Vec<(String, Vec<usize>, Role)> {
        type Out = Vec<(String, Vec<usize>, Role)>;
        let mut out: Out = Vec::new();
        let conv = |out: &mut Out, k: String, o: usize, i: usize, s: usize| {
            out.push((format!("{k}.weight"), vec![o, i, s, s], Role::Conv));
            out.push((format!("{k}.bias"), vec![o], Role::Bias));
        };
        let norm = |out: &mut Out, k: String, c: usize| {
            out.push((format!("{k}.weight"), vec![c], Role::NormWeight));
            out.push((format!("{k}.bias"), vec![c], Role::Bias));
        };
        let resnet = |out: &mut Out, k: &str, i: usize, o: usize| {
            norm(out, format!("{k}.norm1"), i);
            conv(out, format!("{k}.conv1"), o, i, 3);
            norm(out, format!("{k}.norm2"), o);
            conv(out, format!("{k}.conv2"), o, o, 3);
            if i != o {
                conv(out, format!("{k}.conv_shortcut"), o, i, 1);
            }
        };
        let b = self.block_out;
        conv(&mut out, "encoder.conv_in".into(), b[0], 3, 3);
        let mut cin = b[0];
        for (lvl, &cout) in b.iter().enumerate() {
            for j in 0..RESNETS_PER_LEVEL {
                resnet(
                    &mut out,
                    &format!("encoder.down_blocks.{lvl}.resnets.{j}"),
                    cin,
                    cout,
                );
                cin = cout;
            }
            if lvl + 1 < b.len() {
                conv(
                    &mut out,
                    format!("encoder.down_blocks.{lvl}.downsamplers.0.conv"),
                    cout,
                    cout,
                    3,
                );
            }
        }
        let c = b[3];
        resnet(&mut out, "encoder.mid_block.resnets.0", c, c);
        let a = "encoder.mid_block.attentions.0";
        norm(&mut out, format!("{a}.group_norm"), c);
        for p in ["to_q", "to_k", "to_v", "to_out.0"] {
            out.push((format!("{a}.{p}.weight"), vec![c, c], Role::Conv));
            out.push((format!("{a}.{p}.bias"), vec![c], Role::Bias));
        }
        resnet(&mut out, "encoder.mid_block.resnets.1", c, c);
        out
    }

    /// Exact parameter count of the tapped sub-graph.
    pub fn param_count(&self) -> u64 {
        self.tensors()
            .iter()
            .map(|(_, s, _)| s.iter().map(|&d| d as u64).product::<u64>())
            .sum()
    }

    /// The five tap shapes `(channels, h, w)` for an `h × w` image.
    pub fn tap_shapes(&self, h: u32, w: u32) -> [(u64, u64, u64); 5] {
        let b = self.block_out;
        let mut out = [(0, 0, 0); 5];
        for (lvl, slot) in out.iter_mut().take(4).enumerate() {
            *slot = (b[lvl] as u64, (h >> lvl) as u64, (w >> lvl) as u64);
        }
        out[4] = out[3];
        out
    }

    /// Pre-load memory figures for `h × w` training images (epic 2123 E7) — identical arithmetic to
    /// the MLX twin.
    pub fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint {
        let taps = self.tap_shapes(h, w);
        let mut floats = 3 * h as u64 * w as u64;
        for (lvl, &(c, hh, ww)) in taps.iter().take(4).enumerate() {
            let px = hh * ww;
            floats += RESNETS_PER_LEVEL as u64 * 7 * c * px;
            if lvl < 3 {
                floats += 2 * c * px;
            }
        }
        let (c, hh, ww) = taps[4];
        let px = hh * ww;
        floats += 2 * 7 * c * px + 5 * c * px + px * px;
        AuxModelFootprint {
            param_bytes: self.param_count() * 4,
            working_set_bytes: floats * 4 * 2,
            reference_bytes_per_image: taps.iter().map(|&(c, hh, ww)| c * hh * ww * 2).sum(),
        }
    }
}

/// The deterministic formula weights for `cfg`'s tapped sub-graph on `device`.
pub fn formula_weights(cfg: &VaeAnchorEncoderConfig, device: &Device) -> Result<Weights> {
    let mut map = std::collections::HashMap::new();
    for (k, shape, role) in cfg.tensors() {
        let t = formula_tensor(&k, &shape, role, device)?;
        map.insert(k, t);
    }
    Ok(Weights::from_map(map))
}

fn f32w(w: &Weights, key: &str) -> Result<Tensor> {
    Ok(w.require(key)?.to_dtype(DType::F32)?)
}

fn channel(t: Tensor) -> Result<Tensor> {
    let c = t.elem_count();
    Ok(t.reshape((1, c, 1, 1))?)
}

/// Composed GroupNorm over NCHW (autograd-safe).
fn group_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let g = x.reshape((b, GN_GROUPS, (c / GN_GROUPS) * h * w))?;
    let mean = g.mean_keepdim(2)?;
    let d = g.broadcast_sub(&mean)?;
    let var = d.sqr()?.mean_keepdim(2)?;
    let n = d.broadcast_div(&(var + GN_EPS)?.sqrt()?)?;
    Ok(n.reshape((b, c, h, w))?
        .broadcast_mul(gamma)?
        .broadcast_add(beta)?)
}

/// Composed SiLU `x / (1 + e^-x)` (autograd-safe).
fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(x.div(&(x.neg()?.exp()? + 1.0)?)?)
}

/// 3×3 / 1×1 conv with an `[1, O, 1, 1]` bias.
fn conv(x: &Tensor, w: &Tensor, b: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
    Ok(x.conv2d(w, padding, stride, 1, 1)?.broadcast_add(b)?)
}

struct Resnet {
    norm1: (Tensor, Tensor),
    conv1: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    conv2: (Tensor, Tensor),
    shortcut: Option<(Tensor, Tensor)>,
}

impl Resnet {
    fn load(w: &Weights, k: &str) -> Result<Self> {
        let pair = |leaf: &str| -> Result<(Tensor, Tensor)> {
            let weight = f32w(w, &format!("{k}.{leaf}.weight"))?;
            let weight = if weight.rank() == 1 {
                channel(weight)?
            } else {
                weight
            };
            Ok((weight, channel(f32w(w, &format!("{k}.{leaf}.bias"))?)?))
        };
        let shortcut = if w.contains(&format!("{k}.conv_shortcut.weight")) {
            Some(pair("conv_shortcut")?)
        } else {
            None
        };
        Ok(Self {
            norm1: pair("norm1")?,
            conv1: pair("conv1")?,
            norm2: pair("norm2")?,
            conv2: pair("conv2")?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = silu(&group_norm(x, &self.norm1.0, &self.norm1.1)?)?;
        let h = conv(&h, &self.conv1.0, &self.conv1.1, 1, 1)?;
        let h = silu(&group_norm(&h, &self.norm2.0, &self.norm2.1)?)?;
        let h = conv(&h, &self.conv2.0, &self.conv2.1, 1, 1)?;
        let skip = match &self.shortcut {
            Some((cw, cb)) => conv(x, cw, cb, 1, 0)?,
            None => x.clone(),
        };
        Ok((skip + h)?)
    }
}

struct Attention {
    norm: (Tensor, Tensor),
    /// `(weight [C, C, 1, 1], bias [1, C, 1, 1])` — upstream's 1×1 convs.
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    o: (Tensor, Tensor),
}

impl Attention {
    fn load(w: &Weights, k: &str) -> Result<Self> {
        let proj = |leaf: &str| -> Result<(Tensor, Tensor)> {
            let weight = f32w(w, &format!("{k}.{leaf}.weight"))?;
            let (o, i) = (weight.dim(0)?, weight.dim(1)?);
            Ok((
                weight.reshape((o, i, 1, 1))?,
                channel(f32w(w, &format!("{k}.{leaf}.bias"))?)?,
            ))
        };
        Ok(Self {
            norm: (
                channel(f32w(w, &format!("{k}.group_norm.weight"))?)?,
                channel(f32w(w, &format!("{k}.group_norm.bias"))?)?,
            ),
            q: proj("to_q")?,
            k: proj("to_k")?,
            v: proj("to_v")?,
            o: proj("to_out.0")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let y = group_norm(x, &self.norm.0, &self.norm.1)?;
        // [B, C, H, W] → [B, HW, C]
        let seq = |p: &(Tensor, Tensor)| -> Result<Tensor> {
            Ok(conv(&y, &p.0, &p.1, 1, 0)?
                .reshape((b, c, h * w))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (seq(&self.q)?, seq(&self.k)?, seq(&self.v)?);
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * (c as f64).powf(-0.5))?;
        // Composed softmax over the last axis (the max shift is a detached constant).
        let shifted = scores.broadcast_sub(&scores.max_keepdim(D::Minus1)?.detach())?;
        let e = shifted.exp()?;
        let p = e.broadcast_div(&e.sum_keepdim(D::Minus1)?)?;
        let o = p
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, c, h, w))?;
        Ok((x + conv(&o, &self.o.0, &self.o.1, 1, 0)?)?)
    }
}

/// One resolution level: its resnets, then the optional stride-2 downsample conv `(weight, bias)`.
type Level = (Vec<Resnet>, Option<(Tensor, Tensor)>);

/// The frozen FLUX.2 VAE encoder sub-graph that produces the five anchor taps.
pub struct VaeAnchorEncoder {
    cfg: VaeAnchorEncoderConfig,
    conv_in: (Tensor, Tensor),
    levels: Vec<Level>,
    mid0: Resnet,
    attn: Attention,
    mid1: Resnet,
}

impl VaeAnchorEncoder {
    /// Build from a diffusers-layout VAE checkpoint map.
    pub fn from_weights(w: &Weights, cfg: VaeAnchorEncoderConfig) -> Result<Self> {
        let conv_in_w = f32w(w, "encoder.conv_in.weight")?;
        if conv_in_w.dim(0)? != cfg.block_out[0] {
            return Err(CandleError::Msg(format!(
                "VAE anchor: encoder.conv_in has {} output channels, expected {} (not a FLUX.2 VAE?)",
                conv_in_w.dim(0)?,
                cfg.block_out[0]
            )));
        }
        let conv_in = (conv_in_w, channel(f32w(w, "encoder.conv_in.bias")?)?);
        let mut levels = Vec::with_capacity(4);
        for lvl in 0..4 {
            let resnets = (0..RESNETS_PER_LEVEL)
                .map(|j| Resnet::load(w, &format!("encoder.down_blocks.{lvl}.resnets.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let down = if lvl < 3 {
                let k = format!("encoder.down_blocks.{lvl}.downsamplers.0.conv");
                Some((
                    f32w(w, &format!("{k}.weight"))?,
                    channel(f32w(w, &format!("{k}.bias"))?)?,
                ))
            } else {
                None
            };
            levels.push((resnets, down));
        }
        Ok(Self {
            cfg,
            conv_in,
            levels,
            mid0: Resnet::load(w, "encoder.mid_block.resnets.0")?,
            attn: Attention::load(w, "encoder.mid_block.attentions.0")?,
            mid1: Resnet::load(w, "encoder.mid_block.resnets.1")?,
        })
    }

    /// Load the FLUX.2 VAE from a diffusers `vae/` directory onto `device` (only `encoder.*` read).
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                CandleError::Msg(format!("VAE anchor: cannot read {}: {e}", dir.display()))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "VAE anchor: no .safetensors FLUX.2 VAE in {}",
                dir.display()
            )));
        }
        let w = Weights::from_files_filtered(&files, device, DType::F32, &["encoder."])?;
        Self::from_weights(&w, VaeAnchorEncoderConfig::flux2()).map_err(|e| {
            CandleError::Msg(format!(
                "VAE anchor: FLUX.2 VAE from {}: {e}",
                dir.display()
            ))
        })
    }

    /// The encoder width.
    pub fn config(&self) -> VaeAnchorEncoderConfig {
        self.cfg
    }

    /// The five taps ([`VAE_ANCHOR_LEVELS`]) of an NCHW image in `[-1, 1]`. Differentiable in `x`.
    pub fn features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut h = conv(
            &x.to_dtype(DType::F32)?,
            &self.conv_in.0,
            &self.conv_in.1,
            1,
            1,
        )?;
        let mut taps = Vec::with_capacity(5);
        for (resnets, down) in &self.levels {
            for r in resnets {
                h = r.forward(&h)?;
            }
            taps.push(h.clone());
            if let Some((cw, cb)) = down {
                // Upstream pads (right, bottom) by one, then a stride-2 pad-0 conv. Crop the padded
                // input to the extent the 3×3/2 windows read so both remainders are zero (candle's
                // Conv2D backward otherwise mis-shapes unequal row/column remainders).
                let hp = h.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
                let (_, _, ph, pw) = hp.dims4()?;
                let used = |n: usize| ((n - 3) / 2) * 2 + 3;
                let hp = hp
                    .narrow(2, 0, used(ph))?
                    .narrow(3, 0, used(pw))?
                    .contiguous()?;
                h = conv(&hp, cw, cb, 2, 0)?;
            }
        }
        h = self.mid0.forward(&h)?;
        h = self.attn.forward(&h)?;
        h = self.mid1.forward(&h)?;
        taps.push(h);
        Ok(taps)
    }
}

/// Upstream `VAEAnchorEncoder.compute_loss` for one image (NCHW taps): per level the mean over
/// positions of `1 − cos(pred, ref)` across channels, weighted [`VAE_ANCHOR_LEVEL_WEIGHTS`], averaged
/// over the five levels. Returns `(total, per_level)`. Differentiable in `pred`.
pub fn vae_anchor_feature_loss(
    pred: &[Tensor],
    reference: &[Tensor],
) -> Result<(Tensor, Vec<Tensor>)> {
    if pred.len() != 5 || reference.len() != 5 {
        return Err(CandleError::Msg(format!(
            "VAE anchor: expected 5 taps, got {} / {}",
            pred.len(),
            reference.len()
        )));
    }
    let mut total: Option<Tensor> = None;
    let mut per_level = Vec::with_capacity(5);
    for (lvl, (p, r)) in pred.iter().zip(reference).enumerate() {
        if p.dims() != r.dims() {
            return Err(CandleError::Msg(format!(
                "VAE anchor: {} tap shape {:?} differs from its cached reference {:?}",
                VAE_ANCHOR_LEVELS[lvl],
                p.dims(),
                r.dims()
            )));
        }
        let r = r.to_dtype(DType::F32)?;
        let dot = (p * &r)?.sum(1)?;
        let np = p.sqr()?.sum(1)?.sqrt()?.maximum(COS_EPS)?;
        let nr = r.sqr()?.sum(1)?.sqrt()?.maximum(COS_EPS)?;
        let cos = dot.div(&(np * nr)?)?;
        let level = (cos.neg()? + 1.0)?.mean_all()?;
        let w = (&level * VAE_ANCHOR_LEVEL_WEIGHTS[lvl])?;
        total = Some(match total {
            Some(t) => (t + w)?,
            None => w,
        });
        per_level.push(level);
    }
    Ok(((total.expect("five levels") / 5.0)?, per_level))
}

/// The VAE-anchor auxiliary loss: the frozen encoder + cached clean taps per image.
pub struct VaeAnchorLoss {
    encoder: VaeAnchorEncoder,
    references_built: AtomicUsize,
}

/// The per-image VAE-anchor reference: the five clean taps (NCHW), f16 as upstream caches them.
pub struct VaeAnchorReference {
    pub taps: Vec<Tensor>,
}

/// Decoded pixels NHWC `[0, 1]` → NCHW `[-1, 1]`.
fn to_signed_nchw(px: &Tensor) -> Result<Tensor> {
    Ok(((px
        .to_dtype(DType::F32)?
        .permute((0, 3, 1, 2))?
        .contiguous()?
        * 2.0)?
        - 1.0)?)
}

impl VaeAnchorLoss {
    /// Wrap a loaded encoder.
    pub fn new(encoder: VaeAnchorEncoder) -> Self {
        Self {
            encoder,
            references_built: AtomicUsize::new(0),
        }
    }

    /// Load the FLUX.2 VAE encoder from a diffusers `vae/` directory onto `device`.
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Ok(Self::new(VaeAnchorEncoder::from_dir(dir, device)?))
    }

    /// The wrapped encoder.
    pub fn encoder(&self) -> &VaeAnchorEncoder {
        &self.encoder
    }

    /// How many per-image references this loss has encoded.
    pub fn references_built(&self) -> usize {
        self.references_built.load(Ordering::Relaxed)
    }
}

impl PerceptualLoss for VaeAnchorLoss {
    fn name(&self) -> &'static str {
        "vae_anchor"
    }

    fn input(&self) -> PerceptualInput {
        PerceptualInput::DecodedPixels
    }

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        let taps = self
            .encoder
            .features(&to_signed_nchw(&clean.detach())?)?
            .into_iter()
            .map(|t| -> Result<Tensor> { Ok(t.to_dtype(DType::F16)?.detach()) })
            .collect::<Result<Vec<_>>>()?;
        self.references_built.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Box::new(VaeAnchorReference { taps })))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
        let r = reference_as::<VaeAnchorReference>(self.name(), reference)?;
        let pred = self.encoder.features(&to_signed_nchw(live)?)?;
        Ok(vae_anchor_feature_loss(&pred, &r.taps)?.0)
    }
}

/// The pre-load memory figures of the VAE anchor with the FLUX.2 encoder on `image_h × image_w`
/// training images (epic 2123 E7).
pub fn vae_anchor_footprint(image_h: u32, image_w: u32) -> AuxModelFootprint {
    VaeAnchorEncoderConfig::flux2().footprint(image_h, image_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::formula::{input_wave, PARITY_FIXTURE_REL};
    use crate::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath, X0Decoder};
    use candle_core::Var;

    const CPU: Device = Device::Cpu;

    fn fixture() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(PARITY_FIXTURE_REL);
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn scalar(t: &Tensor) -> f32 {
        t.to_dtype(DType::F32).unwrap().to_scalar::<f32>().unwrap()
    }

    fn params(v: &serde_json::Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn encoder(ch: usize) -> VaeAnchorEncoder {
        let cfg = VaeAnchorEncoderConfig::with_base(ch);
        VaeAnchorEncoder::from_weights(&formula_weights(&cfg, &CPU).unwrap(), cfg).unwrap()
    }

    fn uniform(seed: u64, shape: (usize, usize, usize, usize)) -> Tensor {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let v: Vec<f32> = (0..n).map(|_| rng.random_range(0.0f32..1.0)).collect();
        Tensor::from_vec(v, shape, &CPU).unwrap()
    }

    /// Parity with upstream `encode_with_features` + `compute_loss` over the FLUX.2 `Encoder`
    /// (width 32) on formula weights — the same fixture the MLX twin asserts: every tap's shape and
    /// mean-|x|, each level term, the total and the zero self-loss. Mutations: tap after the
    /// downsample; drop the mid attention; uniform level weights ⇒ red.
    #[test]
    fn matches_the_upstream_reference_on_formula_weights() {
        let fx = fixture();
        let case = &fx["vae_anchor"];
        let (ch, h, w) = (
            case["ch"].as_u64().unwrap() as usize,
            case["h"].as_u64().unwrap() as usize,
            case["w"].as_u64().unwrap() as usize,
        );
        let n = 3 * h * w;
        let p = params(&case["inputs"]["pred"]);
        let d = params(&case["inputs"]["ref_delta"]);
        let pred_v = input_wave(n, p[0], p[1], p[2]);
        let ref_v: Vec<f64> = pred_v
            .iter()
            .zip(input_wave(n, d[0], d[1], d[2]))
            .map(|(a, b)| (a + b).clamp(-1.0, 1.0))
            .collect();
        let t = |v: &[f64]| {
            Tensor::from_vec(
                v.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                (1, 3, h, w),
                &CPU,
            )
            .unwrap()
        };
        let enc = encoder(ch);
        let pf = enc.features(&t(&pred_v)).unwrap();
        let rf = enc.features(&t(&ref_v)).unwrap();
        let close = |got: f32, want: f64, what: &str, rel: f64| {
            let tol = rel * want.abs().max(1e-3);
            assert!(
                (got as f64 - want).abs() <= tol,
                "{what}: {got} vs upstream {want}"
            );
        };
        for (lvl, name) in VAE_ANCHOR_LEVELS.iter().enumerate() {
            let s = &case["pred_feature_stats"][name];
            let shape: Vec<usize> = s["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(pf[lvl].dims(), shape.as_slice(), "{name}");
            close(
                scalar(&pf[lvl].abs().unwrap().mean_all().unwrap()),
                s["abs_mean"].as_f64().unwrap(),
                name,
                1e-4,
            );
        }
        let (total, per) = vae_anchor_feature_loss(&pf, &rf).unwrap();
        for (lvl, name) in VAE_ANCHOR_LEVELS.iter().enumerate() {
            close(
                scalar(&per[lvl]),
                case["per_level"][name].as_f64().unwrap(),
                name,
                2e-3,
            );
        }
        close(
            scalar(&total),
            case["loss"].as_f64().unwrap(),
            "total",
            2e-3,
        );
        assert!(scalar(&vae_anchor_feature_loss(&rf, &rf).unwrap().0).abs() < 1e-6);
    }

    struct Identity;
    impl X0Decoder for Identity {
        fn decode(&self, latents: &Tensor) -> Result<Tensor> {
            Ok(latents.permute((0, 2, 3, 1))?.contiguous()?)
        }
    }

    /// AC1 (Candle): on the shared path with the VAE-anchor weight > 0, each image's multi-scale
    /// reference is encoded exactly once across repeated steps; the loss is ~0 at the clean input,
    /// positive off it, and back-propagates a nonzero gradient into the live x0 through candle's
    /// autograd — on a non-square image whose rows and columns leave different stride-2 remainders
    /// (the Conv2D backward hazard). Mutations: drop the reference cache hit ⇒ counter 6; `detach`
    /// the live taps ⇒ no gradient; remove the downsample crop ⇒ backward shape error ⇒ red.
    #[test]
    fn references_are_cached_once_per_image_and_the_loss_flows_gradient() {
        let mut path = PerceptualPath::new(
            Some(Box::new(Identity)),
            vec![AuxLoss {
                schedule: AuxLossSchedule {
                    weight: 0.5,
                    t_min: 0.0,
                    t_max: 0.5,
                    every_n: 1,
                },
                loss: Box::new(VaeAnchorLoss::new(encoder(32))),
            }],
        )
        .unwrap();
        let imgs = [uniform(1, (1, 3, 18, 24)), uniform(2, (1, 3, 18, 24))];
        for _ in 0..3 {
            for (i, x) in imgs.iter().enumerate() {
                path.ensure_reference(i, x).unwrap();
                let plan = path.plan(1, i, 0.2).unwrap();
                assert_eq!(plan.aux, vec![0]);
                path.aux_loss(&plan, i, x).unwrap();
            }
        }
        assert_eq!(path.reference_computations(), 2);
        let plan = path.plan(1, 0, 0.2).unwrap();
        let at_clean = scalar(&path.aux_loss(&plan, 0, &imgs[0]).unwrap().unwrap().weighted);
        assert!(
            at_clean.abs() < 1e-3,
            "self loss (f16 cache only) {at_clean}"
        );
        let off = (&imgs[0] + 0.15).unwrap().clamp(0f32, 1f32).unwrap();
        let live = Var::from_tensor(&off).unwrap();
        let loss = path
            .aux_loss(&plan, 0, live.as_tensor())
            .unwrap()
            .unwrap()
            .weighted;
        assert!(scalar(&loss) > 0.0);
        let grads = loss.backward().unwrap();
        let g = grads
            .get(live.as_tensor())
            .expect("gradient reaches the live x0");
        assert!(scalar(&g.abs().unwrap().sum_all().unwrap()) > 0.0);
    }

    /// One reference encode per `reference` call, none per `loss` call; the cache is f16.
    #[test]
    fn the_loss_counts_reference_encodes() {
        let loss = VaeAnchorLoss::new(encoder(32));
        let x = uniform(5, (1, 8, 8, 3));
        let r = loss.reference(&x).unwrap().unwrap();
        assert_eq!(loss.references_built(), 1);
        assert!(scalar(&loss.loss(&x, r.as_ref()).unwrap()).abs() < 1e-3);
        assert_eq!(loss.references_built(), 1);
        let taps = &reference_as::<VaeAnchorReference>("vae_anchor", r.as_ref())
            .unwrap()
            .taps;
        assert!(taps.iter().all(|t| t.dtype() == DType::F16));
    }

    #[test]
    fn a_non_flux2_encoder_is_refused() {
        let w = formula_weights(&VaeAnchorEncoderConfig::with_base(32), &CPU).unwrap();
        let err = VaeAnchorEncoder::from_weights(&w, VaeAnchorEncoderConfig::flux2())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("not a FLUX.2 VAE"), "{err}");
    }

    /// E7: identical arithmetic to the MLX twin (weights + working set + f16 taps per image).
    /// Mutation: count the taps as f32 ⇒ red.
    #[test]
    fn footprint_counts_weights_working_set_and_f16_taps() {
        let cfg = VaeAnchorEncoderConfig::flux2();
        let p = cfg.param_count();
        assert!((30_000_000..36_000_000).contains(&p), "{p}");
        let f = cfg.footprint(512, 512);
        let taps: u64 = [
            128 * 512 * 512,
            256 * 256 * 256,
            512 * 128 * 128,
            512 * 64 * 64,
            512 * 64 * 64,
        ]
        .iter()
        .sum::<u64>();
        assert_eq!(f.reference_bytes_per_image, taps * 2);
        assert_eq!(f.param_bytes, p * 4);
    }
}

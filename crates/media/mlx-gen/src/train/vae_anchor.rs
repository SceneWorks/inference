//! **VAE perceptual anchor** (epic 2123, sc-24833) — a frozen FLUX.2 VAE **encoder** as a
//! multi-scale perceptual discriminator on the shared decoded-x0 path ([`super::perceptual`]), a
//! port of ai-toolkit-perceptual `toolkit/vae_anchor.py` (fork commit 6e01a6e).
//!
//! The family's small decoder turns the model's x0 into pixels; this loss maps them to `[-1, 1]`,
//! runs the FLUX.2 VAE encoder up to its mid block and taps five feature maps — upstream's forward
//! hooks: the output of each resolution level's **last resnet** (`down[i].block[1]`, before that
//! level's downsample; 128/256/512/512 channels at H, H/2, H/4, H/8) and the mid block's **second
//! resnet** (`mid.block_2`, after the mid attention). Each tap is compared with the same tap of the
//! training image's clean round trip — computed once per image and cached (in f16, as upstream
//! caches it) — by the mean over positions of `1 − cos` across channels, and the level terms are
//! combined as `(4·L0 + 2·L1 + L2 + L3 + Lmid) / 5` (upstream `compute_loss` defaults; its unused
//! projector heads are not part of the technique).
//!
//! The encoder runs f32 with composed (autograd-safe) GroupNorm / SiLU; weights are the diffusers
//! `AutoencoderKLFlux2` layout (`encoder.down_blocks.{i}.resnets.{j}.…`,
//! `encoder.mid_block.{resnets,attentions}.…`), of which only the tapped sub-graph is read.

use std::any::Any;
use std::cell::Cell;
use std::path::Path;

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, divide, maximum, multiply, pad, sigmoid, sqrt, subtract};
use mlx_rs::{Array, Dtype};

use super::formula::{formula_array, Role};
use super::perceptual::{
    reference_as, AuxModelFootprint, LossReference, PerceptualInput, PerceptualLoss,
};
use crate::nn::conv2d;
use crate::weights::Weights;
use crate::{Error, Result};

/// The five tap names, in [`VaeAnchorEncoder::features`] order.
pub const VAE_ANCHOR_LEVELS: [&str; 5] = ["level_0", "level_1", "level_2", "level_3", "mid"];
/// Upstream `compute_loss` default per-level weights (higher-resolution levels weigh more).
pub const VAE_ANCHOR_LEVEL_WEIGHTS: [f32; 5] = [4.0, 2.0, 1.0, 1.0, 1.0];
const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;
/// `F.cosine_similarity` default `eps` (each norm is clamped to at least this).
const COS_EPS: f32 = 1e-8;
const RESNETS_PER_LEVEL: usize = 2;

/// Width of the FLUX.2 VAE encoder whose features the loss taps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaeAnchorEncoderConfig {
    /// Output channels of the four resolution levels.
    pub block_out: [i32; 4],
}

impl VaeAnchorEncoderConfig {
    /// The FLUX.2 VAE (`ch = 128`, `ch_mult = [1, 2, 4, 4]`).
    pub fn flux2() -> Self {
        Self {
            block_out: [128, 256, 512, 512],
        }
    }

    /// The same graph at `ch` base width (`ch_mult = [1, 2, 4, 4]`) — the parity fixture uses 32.
    pub fn with_base(ch: i32) -> Self {
        Self {
            block_out: [ch, 2 * ch, 4 * ch, 4 * ch],
        }
    }

    /// Every encoder tensor the tapped sub-graph reads (diffusers keys, torch shapes) with its
    /// formula role.
    fn tensors(&self) -> Vec<(String, Vec<i32>, Role)> {
        let mut out = Vec::new();
        let conv = |out: &mut Vec<(String, Vec<i32>, Role)>, k: String, o: i32, i: i32, s: i32| {
            out.push((format!("{k}.weight"), vec![o, i, s, s], Role::Conv));
            out.push((format!("{k}.bias"), vec![o], Role::Bias));
        };
        let norm = |out: &mut Vec<(String, Vec<i32>, Role)>, k: String, c: i32| {
            out.push((format!("{k}.weight"), vec![c], Role::NormWeight));
            out.push((format!("{k}.bias"), vec![c], Role::Bias));
        };
        let resnet = |out: &mut Vec<(String, Vec<i32>, Role)>, k: &str, i: i32, o: i32| {
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

    /// Exact parameter count of the tapped sub-graph (for the pre-load memory estimate).
    pub fn param_count(&self) -> u64 {
        self.tensors()
            .iter()
            .map(|(_, s, _)| s.iter().map(|&d| d as u64).product::<u64>())
            .sum()
    }

    /// The five tap shapes `(channels, h, w)` for an `h × w` image.
    pub fn tap_shapes(&self, h: u32, w: u32) -> [(u64, u64, u64); 5] {
        let b = self.block_out;
        let at = |lvl: u32| ((h >> lvl) as u64, (w >> lvl) as u64);
        let mut out = [(0, 0, 0); 5];
        for lvl in 0..4 {
            let (hh, ww) = at(lvl as u32);
            out[lvl] = (b[lvl] as u64, hh, ww);
        }
        out[4] = out[3];
        out
    }

    /// Pre-load memory figures for `h × w` training images (epic 2123 E7): resident f32 weights;
    /// a working set holding every retained intermediate of one differentiable encoder pass (per
    /// resnet: two norms, two SiLUs, two convs, the shortcut sum; the downsample pad + conv; the mid
    /// attention's q/k/v/out plus its `(HW/64)²` score matrix), ×2 for the backward's cotangents;
    /// and the five f16 taps cached per image.
    pub fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint {
        let taps = self.tap_shapes(h, w);
        let mut floats = 3 * h as u64 * w as u64; // the [-1, 1] input
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

/// The deterministic formula weights for `cfg`'s tapped sub-graph (every key the encoder reads) —
/// what the parity fixture was produced with, and synthetic weights for tests.
pub fn formula_weights(cfg: &VaeAnchorEncoderConfig) -> Weights {
    let mut w = Weights::empty();
    for (k, shape, role) in cfg.tensors() {
        let a = formula_array(&k, &shape, role);
        w.insert(k, a);
    }
    w
}

fn f32w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Float32)?)
}

fn conv_w(w: &Weights, key: &str) -> Result<Array> {
    Ok(f32w(w, key)?.transpose_axes(&[0, 2, 3, 1])?)
}

/// Composed GroupNorm over NHWC (autograd-safe; no fused kernel).
fn group_norm(x: &Array, gamma: &Array, beta: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let g = x.reshape(&[b, h * w, GN_GROUPS, c / GN_GROUPS])?;
    let mean = g.mean_axes(&[1, 3], true)?;
    let d = subtract(&g, &mean)?;
    let var = multiply(&d, &d)?.mean_axes(&[1, 3], true)?;
    let n = divide(&d, &sqrt(&add(&var, Array::from_f32(GN_EPS))?)?)?;
    Ok(add(&multiply(&n.reshape(&[b, h, w, c])?, gamma)?, beta)?)
}

fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

struct Resnet {
    norm1: (Array, Array),
    conv1: (Array, Array),
    norm2: (Array, Array),
    conv2: (Array, Array),
    shortcut: Option<(Array, Array)>,
}

impl Resnet {
    fn load(w: &Weights, k: &str) -> Result<Self> {
        let pair = |leaf: &str, conv: bool| -> Result<(Array, Array)> {
            let weight = if conv {
                conv_w(w, &format!("{k}.{leaf}.weight"))?
            } else {
                f32w(w, &format!("{k}.{leaf}.weight"))?
            };
            Ok((weight, f32w(w, &format!("{k}.{leaf}.bias"))?))
        };
        let shortcut = if w.get(&format!("{k}.conv_shortcut.weight")).is_some() {
            Some(pair("conv_shortcut", true)?)
        } else {
            None
        };
        Ok(Self {
            norm1: pair("norm1", false)?,
            conv1: pair("conv1", true)?,
            norm2: pair("norm2", false)?,
            conv2: pair("conv2", true)?,
            shortcut,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = silu(&group_norm(x, &self.norm1.0, &self.norm1.1)?)?;
        let h = conv2d(&h, &self.conv1.0, Some(&self.conv1.1), 1, 1)?;
        let h = silu(&group_norm(&h, &self.norm2.0, &self.norm2.1)?)?;
        let h = conv2d(&h, &self.conv2.0, Some(&self.conv2.1), 1, 1)?;
        let skip = match &self.shortcut {
            Some((cw, cb)) => conv2d(x, cw, Some(cb), 1, 0)?,
            None => x.clone(),
        };
        Ok(add(&skip, &h)?)
    }
}

struct Attention {
    norm: (Array, Array),
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    o: (Array, Array),
}

impl Attention {
    fn load(w: &Weights, k: &str) -> Result<Self> {
        let pair = |leaf: &str| -> Result<(Array, Array)> {
            let weight = f32w(w, &format!("{k}.{leaf}.weight"))?;
            let sh = weight.shape().to_vec();
            // diffusers ships Linear [C, C]; a 1×1-conv export [C, C, 1, 1] is the same matrix.
            Ok((
                weight.reshape(&[sh[0], 1, 1, sh[1]])?,
                f32w(w, &format!("{k}.{leaf}.bias"))?,
            ))
        };
        Ok(Self {
            norm: (
                f32w(w, &format!("{k}.group_norm.weight"))?,
                f32w(w, &format!("{k}.group_norm.bias"))?,
            ),
            q: pair("to_q")?,
            k: pair("to_k")?,
            v: pair("to_v")?,
            o: pair("to_out.0")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.norm.0, &self.norm.1)?;
        // The projections run as 1×1 convolutions (exactly upstream's `nn.Conv2d(c, c, 1)`): MLX may
        // route an f32 `matmul` through TF32 on GPUs that support it, which drifts ~1e-3 relative.
        let seq = |p: &(Array, Array)| -> Result<Array> {
            Ok(conv2d(&y, &p.0, Some(&p.1), 1, 0)?.reshape(&[b, 1, h * w, c])?)
        };
        let (q, k, v) = (seq(&self.q)?, seq(&self.k)?, seq(&self.v)?);
        let o = scaled_dot_product_attention(&q, &k, &v, (c as f32).powf(-0.5), None, None)?;
        let o = conv2d(&o.reshape(&[b, h, w, c])?, &self.o.0, Some(&self.o.1), 1, 0)?;
        Ok(add(x, &o)?)
    }
}

/// The frozen FLUX.2 VAE encoder sub-graph that produces the five anchor taps.
pub struct VaeAnchorEncoder {
    cfg: VaeAnchorEncoderConfig,
    conv_in: (Array, Array),
    levels: Vec<(Vec<Resnet>, Option<(Array, Array)>)>,
    mid0: Resnet,
    attn: Attention,
    mid1: Resnet,
}

impl VaeAnchorEncoder {
    /// Build from a diffusers-layout VAE checkpoint map.
    pub fn from_weights(w: &Weights, cfg: VaeAnchorEncoderConfig) -> Result<Self> {
        let conv_in = (
            conv_w(w, "encoder.conv_in.weight")?,
            f32w(w, "encoder.conv_in.bias")?,
        );
        if conv_in.0.shape()[0] != cfg.block_out[0] {
            return Err(Error::Msg(format!(
                "VAE anchor: encoder.conv_in has {} output channels, expected {} (not a FLUX.2 VAE?)",
                conv_in.0.shape()[0],
                cfg.block_out[0]
            )));
        }
        let mut levels = Vec::with_capacity(4);
        for lvl in 0..4 {
            let resnets = (0..RESNETS_PER_LEVEL)
                .map(|j| Resnet::load(w, &format!("encoder.down_blocks.{lvl}.resnets.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let down = if lvl < 3 {
                let k = format!("encoder.down_blocks.{lvl}.downsamplers.0.conv");
                Some((
                    conv_w(w, &format!("{k}.weight"))?,
                    f32w(w, &format!("{k}.bias"))?,
                ))
            } else {
                None
            };
            levels.push((resnets, down));
        }
        let enc = Self {
            cfg,
            conv_in,
            levels,
            mid0: Resnet::load(w, "encoder.mid_block.resnets.0")?,
            attn: Attention::load(w, "encoder.mid_block.attentions.0")?,
            mid1: Resnet::load(w, "encoder.mid_block.resnets.1")?,
        };
        Ok(enc)
    }

    /// Load the FLUX.2 VAE from a diffusers `vae/` directory (`diffusion_pytorch_model.safetensors`).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let w = Weights::from_dir(dir)?;
        Self::from_weights(&w, VaeAnchorEncoderConfig::flux2()).map_err(|e| {
            Error::Msg(format!(
                "VAE anchor: FLUX.2 VAE from {}: {e}",
                dir.display()
            ))
        })
    }

    /// The encoder width.
    pub fn config(&self) -> VaeAnchorEncoderConfig {
        self.cfg
    }

    /// The five taps ([`VAE_ANCHOR_LEVELS`]) of an NHWC image in `[-1, 1]`. Differentiable in `x`.
    pub fn features(&self, x: &Array) -> Result<Vec<Array>> {
        let mut h = conv2d(
            &x.as_dtype(Dtype::Float32)?,
            &self.conv_in.0,
            Some(&self.conv_in.1),
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
                // Upstream pads (right, bottom) by one, then a stride-2 pad-0 conv.
                h = pad(&h, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
                h = conv2d(&h, cw, Some(cb), 2, 0)?;
            }
        }
        h = self.mid0.forward(&h)?;
        h = self.attn.forward(&h)?;
        h = self.mid1.forward(&h)?;
        taps.push(h);
        Ok(taps)
    }
}

/// Upstream `VAEAnchorEncoder.compute_loss` for one image: per level the mean over positions of
/// `1 − cos(pred, ref)` across channels, weighted [`VAE_ANCHOR_LEVEL_WEIGHTS`] and averaged over the
/// five levels. Returns `(total, per_level)`. Taps are NHWC; `reference` may be any float dtype
/// (it is cast to f32). Differentiable in `pred`.
pub fn vae_anchor_feature_loss(pred: &[Array], reference: &[Array]) -> Result<(Array, Vec<Array>)> {
    if pred.len() != 5 || reference.len() != 5 {
        return Err(Error::Msg(format!(
            "VAE anchor: expected 5 taps, got {} / {}",
            pred.len(),
            reference.len()
        )));
    }
    let eps = Array::from_f32(COS_EPS);
    let mut total = Array::from_f32(0.0);
    let mut per_level = Vec::with_capacity(5);
    for (lvl, (p, r)) in pred.iter().zip(reference).enumerate() {
        if p.shape() != r.shape() {
            return Err(Error::Msg(format!(
                "VAE anchor: {} tap shape {:?} differs from its cached reference {:?} (references \
                 are cached per (item, bucket) and must match the live decode)",
                VAE_ANCHOR_LEVELS[lvl],
                p.shape(),
                r.shape()
            )));
        }
        let r = r.as_dtype(Dtype::Float32)?;
        let dot = multiply(p, &r)?.sum_axes(&[3], false)?;
        let np = maximum(&sqrt(&multiply(p, p)?.sum_axes(&[3], false)?)?, &eps)?;
        let nr = maximum(&sqrt(&multiply(&r, &r)?.sum_axes(&[3], false)?)?, &eps)?;
        let cos = divide(&dot, &multiply(&np, &nr)?)?;
        let level = subtract(Array::from_f32(1.0), &cos)?.mean(None)?;
        total = add(
            &total,
            &multiply(&level, Array::from_f32(VAE_ANCHOR_LEVEL_WEIGHTS[lvl]))?,
        )?;
        per_level.push(level);
    }
    Ok((divide(&total, Array::from_f32(5.0))?, per_level))
}

/// The VAE-anchor auxiliary loss: the frozen encoder + cached clean taps per image.
pub struct VaeAnchorLoss {
    encoder: VaeAnchorEncoder,
    references_built: Cell<usize>,
}

/// The per-image VAE-anchor reference: the five taps of the clean round trip, f16 (as upstream
/// caches them).
pub struct VaeAnchorReference {
    pub taps: Vec<Array>,
}

/// Decoded pixels NHWC `[0, 1]` → the encoder's `[-1, 1]`.
fn to_signed(px: &Array) -> Result<Array> {
    Ok(subtract(
        &multiply(&px.as_dtype(Dtype::Float32)?, Array::from_f32(2.0))?,
        Array::from_f32(1.0),
    )?)
}

impl VaeAnchorLoss {
    /// Wrap a loaded encoder.
    pub fn new(encoder: VaeAnchorEncoder) -> Self {
        Self {
            encoder,
            references_built: Cell::new(0),
        }
    }

    /// Load the FLUX.2 VAE encoder from a diffusers `vae/` directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(VaeAnchorEncoder::from_dir(dir)?))
    }

    /// The wrapped encoder.
    pub fn encoder(&self) -> &VaeAnchorEncoder {
        &self.encoder
    }

    /// How many per-image references this loss has encoded (each image exactly once per job when
    /// driven through [`super::perceptual::PerceptualPath`]).
    pub fn references_built(&self) -> usize {
        self.references_built.get()
    }
}

impl PerceptualLoss for VaeAnchorLoss {
    fn name(&self) -> &'static str {
        "vae_anchor"
    }

    fn input(&self) -> PerceptualInput {
        PerceptualInput::DecodedPixels
    }

    /// The clean round trip's five taps, cast to f16 and evaluated. Every image is usable.
    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let taps = self
            .encoder
            .features(&to_signed(clean)?)?
            .into_iter()
            .map(|t| -> Result<Array> { Ok(mlx_rs::stop_gradient(t.as_dtype(Dtype::Float16)?)?) })
            .collect::<Result<Vec<_>>>()?;
        mlx_rs::transforms::eval(taps.iter())?;
        self.references_built.set(self.references_built.get() + 1);
        Ok(Some(Box::new(VaeAnchorReference { taps })))
    }

    /// Live decoded pixels NHWC `[0, 1]` → taps → [`vae_anchor_feature_loss`] vs the cached taps.
    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<VaeAnchorReference>(self.name(), reference)?;
        let pred = self.encoder.features(&to_signed(live)?)?;
        Ok(vae_anchor_feature_loss(&pred, &r.taps)?.0)
    }
}

/// The pre-load memory figures of the VAE anchor with the FLUX.2 encoder on `image_h × image_w`
/// training images (epic 2123 E7) — see [`VaeAnchorEncoderConfig::footprint`]. The decoder that
/// feeds it is counted separately (its own footprint on the shared path).
pub fn vae_anchor_footprint(image_h: u32, image_w: u32) -> AuxModelFootprint {
    VaeAnchorEncoderConfig::flux2().footprint(image_h, image_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::formula::{input_wave, PARITY_FIXTURE_REL};
    use crate::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath, X0Decoder};
    use mlx_rs::transforms::{eval, grad};

    fn fixture() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(PARITY_FIXTURE_REL);
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn scalar(a: &Array) -> f32 {
        eval([a]).unwrap();
        a.item::<f32>()
    }

    fn params(v: &serde_json::Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn encoder(ch: i32) -> VaeAnchorEncoder {
        let cfg = VaeAnchorEncoderConfig::with_base(ch);
        VaeAnchorEncoder::from_weights(&formula_weights(&cfg), cfg).unwrap()
    }

    /// NCHW f64 values → NHWC f32 array.
    fn nhwc(v: &[f64], h: i32, w: i32) -> Array {
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        Array::from_slice(&f, &[1, 3, h, w])
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
    }

    /// Parity with upstream `VAEAnchorEncoder.encode_with_features` + `compute_loss` over the
    /// FLUX.2 `Encoder` (width 32) on formula weights: every tap's shape / mean / mean-|x|, each
    /// level term, the total, and the zero self-loss. Mutations: tap after the downsample instead
    /// of before it, drop the mid attention, or swap the level weights to uniform ⇒ red.
    #[test]
    fn matches_the_upstream_reference_on_formula_weights() {
        let fx = fixture();
        let case = &fx["vae_anchor"];
        let (ch, h, w) = (
            case["ch"].as_i64().unwrap() as i32,
            case["h"].as_i64().unwrap() as i32,
            case["w"].as_i64().unwrap() as i32,
        );
        let n = (3 * h * w) as usize;
        let p = params(&case["inputs"]["pred"]);
        let d = params(&case["inputs"]["ref_delta"]);
        let pred_v = input_wave(n, p[0], p[1], p[2]);
        let ref_v: Vec<f64> = pred_v
            .iter()
            .zip(input_wave(n, d[0], d[1], d[2]))
            .map(|(a, b)| (a + b).clamp(-1.0, 1.0))
            .collect();
        let enc = encoder(ch);
        let pf = enc.features(&nhwc(&pred_v, h, w)).unwrap();
        let rf = enc.features(&nhwc(&ref_v, h, w)).unwrap();
        let close = |got: f32, want: f64, what: &str, rel: f64| {
            let tol = rel * want.abs().max(1e-3);
            assert!(
                (got as f64 - want).abs() <= tol,
                "{what}: {got} vs upstream {want}"
            );
        };
        for (lvl, name) in VAE_ANCHOR_LEVELS.iter().enumerate() {
            let s = &case["pred_feature_stats"][name];
            let shape: Vec<i32> = s["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let t = &pf[lvl];
            assert_eq!(
                [t.shape()[0], t.shape()[3], t.shape()[1], t.shape()[2]],
                shape.as_slice(),
                "{name}"
            );
            close(
                scalar(&t.abs().unwrap().mean(None).unwrap()),
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

    /// Identity "decoder" for the shared-path tests: NCHW `[1, 3, H, W]` in `[0, 1]` → NHWC.
    struct Identity;
    impl X0Decoder for Identity {
        fn decode(&self, latents: &Array) -> Result<Array> {
            Ok(latents.transpose_axes(&[0, 2, 3, 1])?)
        }
    }

    /// AC1: with the VAE-anchor weight > 0 on the shared path, each image's multi-scale reference
    /// taps are encoded exactly once however many steps run (loss-level counter + path counter);
    /// the loss is zero at the clean input, positive off it, and back-propagates a nonzero gradient
    /// into the live x0. Mutations: drop `PerceptualPath::ensure_reference`'s cache hit ⇒ the
    /// counter reads 6 ⇒ red; detach the live taps (`stop_gradient` in `loss`) ⇒ zero gradient ⇒ red.
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
        let img = |seed: u64| {
            mlx_rs::random::uniform::<_, f32>(
                0.0f32,
                1.0f32,
                &[1, 3, 16, 16],
                Some(&mlx_rs::random::key(seed).unwrap()),
            )
            .unwrap()
        };
        let imgs = [img(1), img(2)];
        for _step in 0..3 {
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
        assert!(at_clean.abs() < 1e-5, "self loss {at_clean}");
        let off = mlx_rs::ops::clip(
            &add(&imgs[0], Array::from_f32(0.15)).unwrap(),
            (0.0f32, 1.0f32),
        )
        .unwrap();
        let w = scalar(&path.aux_loss(&plan, 0, &off).unwrap().unwrap().weighted);
        assert!(w > 0.0, "off-reference loss {w}");
        let g = grad(|x: &Array| -> mlx_rs::error::Result<Array> {
            Ok(path
                .aux_loss(&plan, 0, x)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
                .expect("aux step")
                .weighted)
        })(&off)
        .unwrap();
        assert!(scalar(&g.abs().unwrap().sum(None).unwrap()) > 0.0);
    }

    /// The loss counts its own reference encodes: one per `reference` call, none per `loss` call.
    #[test]
    fn the_loss_counts_reference_encodes() {
        let loss = VaeAnchorLoss::new(encoder(32));
        let x = mlx_rs::random::uniform::<_, f32>(
            0.0f32,
            1.0f32,
            &[1, 8, 8, 3],
            Some(&mlx_rs::random::key(5).unwrap()),
        )
        .unwrap();
        let r = loss.reference(&x).unwrap().unwrap();
        assert_eq!(loss.references_built(), 1);
        let v = scalar(&loss.loss(&x, r.as_ref()).unwrap());
        assert!(v.abs() < 1e-3, "f16 cache round-off only: {v}");
        assert_eq!(loss.references_built(), 1);
        let taps = &reference_as::<VaeAnchorReference>("vae_anchor", r.as_ref())
            .unwrap()
            .taps;
        assert!(taps.iter().all(|t| t.dtype() == Dtype::Float16));
    }

    /// A checkpoint whose encoder is not the configured width is refused.
    #[test]
    fn a_non_flux2_encoder_is_refused() {
        let w = formula_weights(&VaeAnchorEncoderConfig::with_base(32));
        let err = VaeAnchorEncoder::from_weights(&w, VaeAnchorEncoderConfig::flux2())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("not a FLUX.2 VAE"), "{err}");
    }

    /// The parameter count of the tapped sub-graph fits inside the published FLUX.2 VAE's encoder
    /// (the full encoder adds only `conv_norm_out` / `conv_out` / `quant_conv`). E7: the footprint
    /// counts weights, a working set and the f16 taps per image. Mutation: count the taps as f32 ⇒ red.
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
        assert!(f.working_set_bytes > f.reference_bytes_per_image);
    }
}

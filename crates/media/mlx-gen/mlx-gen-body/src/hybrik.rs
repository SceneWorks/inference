//! **HybrIK shape encoder** (ai-toolkit-perceptual `DifferentiableBodyShapeEncoder`) — HybrIK's
//! ResNet backbone (torchvision BasicBlock layout under `preact.*`) → global average pool →
//! `fc1 → fc2 → decshape` (linear, dropout = identity at eval) `+ init_shape` → 10 SMPL betas
//! (epic 2123, sc-24832). Only the beta head is used, so no SMPL model file is involved.
//! BatchNorm runs in eval mode, folded to a per-channel scale/shift at load.

use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{add, maximum, multiply};
use mlx_rs::Array;

use mlx_gen::gen_core::train::body::{hybrik_square_crop, HybrikConfig, HYBRIK_MEAN, HYBRIK_STD};

use super::{bn_fold, conv_ohwi, normalize, resample_nhwc, w32, AxisMatrix};
use mlx_gen::nn::{conv2d, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

struct Conv {
    w: Array,
    bn: (Array, Array),
    stride: i32,
    pad: i32,
}

impl Conv {
    fn load(
        w: &Weights,
        prefix: &str,
        conv: &str,
        bn: &str,
        stride: i32,
        pad: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            w: conv_ohwi(&w32(w, prefix, &format!("{conv}.weight"))?)?,
            bn: bn_fold(w, prefix, bn, eps)?,
            stride,
            pad,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = conv2d(x, &self.w, None, self.stride, self.pad)?;
        Ok(add(&multiply(&y, &self.bn.0)?, &self.bn.1)?)
    }
}

struct Block {
    c1: Conv,
    c2: Conv,
    down: Option<Conv>,
}

/// The loaded HybrIK shape encoder.
pub struct HybrikEncoder {
    cfg: HybrikConfig,
    stem: Conv,
    blocks: Vec<Block>,
    fc1: (Array, Array),
    fc2: (Array, Array),
    dec: (Array, Array),
    init_shape: Array,
}

fn relu(x: &Array) -> Result<Array> {
    Ok(maximum(x, Array::from_f32(0.0))?)
}

/// torchvision `MaxPool2d(3, 2, 1)` on a non-negative (post-ReLU) NHWC map: zero padding is then
/// equivalent to `-inf` padding.
fn max_pool_3x3_s2(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (h, w) = (sh[1], sh[2]);
    let (oh, ow) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
    let p = mlx_rs::ops::pad(
        x,
        &[(0, 0), (1, 2), (1, 2), (0, 0)],
        Array::from_f32(0.0),
        None,
    )?;
    let mut out: Option<Array> = None;
    for dy in 0..3 {
        for dx in 0..3 {
            let v = p.index((
                ..,
                (dy..dy + 2 * oh).stride_by(2),
                (dx..dx + 2 * ow).stride_by(2),
                ..,
            ));
            out = Some(match out {
                Some(o) => maximum(&o, &v)?,
                None => v,
            });
        }
    }
    Ok(out.expect("nine taps"))
}

impl HybrikEncoder {
    /// Load from a directory holding the checkpoint (`model.safetensors`, HybrIK key layout).
    pub fn from_dir(dir: impl AsRef<std::path::Path>, cfg: HybrikConfig) -> Result<Self> {
        Self::from_weights(&Weights::from_dir(dir)?, "", cfg)
    }

    /// Load from weights whose HybrIK keys sit under `prefix`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: HybrikConfig) -> Result<Self> {
        let eps = cfg.bn_eps;
        let stem = Conv::load(w, prefix, "preact.conv1", "preact.bn1", 2, 3, eps)?;
        let mut blocks = Vec::new();
        let mut cin = cfg.widths[0];
        for (s, (&n, &width)) in cfg.blocks.iter().zip(&cfg.widths).enumerate() {
            for b in 0..n {
                let p = format!("preact.layer{}.{b}", s + 1);
                let stride = if b == 0 && s > 0 { 2 } else { 1 };
                let down = if b == 0 && (s > 0 || cin != width) {
                    Some(Conv::load(
                        w,
                        prefix,
                        &format!("{p}.downsample.0"),
                        &format!("{p}.downsample.1"),
                        stride,
                        0,
                        eps,
                    )?)
                } else {
                    None
                };
                blocks.push(Block {
                    c1: Conv::load(
                        w,
                        prefix,
                        &format!("{p}.conv1"),
                        &format!("{p}.bn1"),
                        stride,
                        1,
                        eps,
                    )?,
                    c2: Conv::load(
                        w,
                        prefix,
                        &format!("{p}.conv2"),
                        &format!("{p}.bn2"),
                        1,
                        1,
                        eps,
                    )?,
                    down,
                });
            }
            cin = width;
        }
        let pair = |k: &str| -> Result<(Array, Array)> {
            Ok((
                w32(w, prefix, &format!("{k}.weight"))?,
                w32(w, prefix, &format!("{k}.bias"))?,
            ))
        };
        Ok(Self {
            stem,
            blocks,
            fc1: pair("fc1")?,
            fc2: pair("fc2")?,
            dec: pair("decshape")?,
            init_shape: w32(w, prefix, "init_shape")?.reshape(&[1, -1])?,
            cfg,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &HybrikConfig {
        &self.cfg
    }

    /// HybrIK-normalized NHWC `[B, S, S, 3]` → betas `[B, 10]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut y = max_pool_3x3_s2(&relu(&self.stem.forward(x)?)?)?;
        for b in &self.blocks {
            let id = match &b.down {
                Some(d) => d.forward(&y)?,
                None => y.clone(),
            };
            let z = b.c2.forward(&relu(&b.c1.forward(&y)?)?)?;
            y = relu(&add(&z, &id)?)?;
        }
        let feat = y.mean_axes(&[1, 2], false)?;
        let h = linear(&feat, &self.fc1.0, &self.fc1.1)?;
        let h = linear(&h, &self.fc2.0, &self.fc2.1)?;
        Ok(add(&linear(&h, &self.dec.0, &self.dec.1)?, &self.init_shape)?)
    }

    /// The **differentiable** pixel entry point: NHWC pixels `[1, H, W, 3]` in `[0, 1]`, cropped
    /// to the half-open `crop = (y0, y1, x0, x1)` ([`hybrik_square_crop`]), resized to the square
    /// input (bilinear, `align_corners=False`), HybrIK-normalized → betas `[1, 10]`.
    pub fn forward_crop(&self, pixels: &Array, crop: (usize, usize, usize, usize)) -> Result<Array> {
        let (y0, y1, x0, x1) = crop;
        let c = pixels.index((.., y0 as i32..y1 as i32, x0 as i32..x1 as i32, ..));
        let s = self.cfg.input_size;
        let x = resample_nhwc(
            &c,
            &AxisMatrix::resize(y1 - y0, s, false),
            &AxisMatrix::resize(x1 - x0, s, false),
        )?;
        self.forward(&normalize(&x, HYBRIK_MEAN, HYBRIK_STD)?)
    }

    /// The crop of a person box on an `h × w` frame.
    pub fn crop_for(bbox: [f32; 4], h: usize, w: usize) -> (usize, usize, usize, usize) {
        hybrik_square_crop(bbox, h, w)
    }
}

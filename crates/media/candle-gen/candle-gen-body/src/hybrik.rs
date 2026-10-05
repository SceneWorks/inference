//! **HybrIK shape encoder** — the candle twin of `mlx_gen_body::hybrik` (epic 2123, sc-24832):
//! HybrIK's torchvision-layout ResNet under `preact.*` → global average pool →
//! `fc1 → fc2 → decshape + init_shape` → 10 SMPL betas. BatchNorm in eval mode, folded at load;
//! strided convs use the backward-safe pad-and-crop form.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::train::body::{
    hybrik_square_crop, HybrikConfig, HYBRIK_MEAN, HYBRIK_STD,
};
use candle_gen::weights::Weights;
use candle_gen::Result;

use super::{bn_fold, conv2d, linear, normalize, resample_nhwc, w32, AxisMatrix, BatchNorm};

struct Conv {
    w: Tensor,
    bn: BatchNorm,
    stride: usize,
    pad: usize,
}

impl Conv {
    fn load(
        w: &Weights,
        prefix: &str,
        conv: &str,
        bn: &str,
        stride: usize,
        pad: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            w: w32(w, prefix, &format!("{conv}.weight"))?,
            bn: bn_fold(w, prefix, bn, eps)?,
            stride,
            pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.bn
            .apply(&conv2d(x, &self.w, None, self.stride, self.pad)?)
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
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    dec: (Tensor, Tensor),
    init_shape: Tensor,
}

/// torchvision `MaxPool2d(3, 2, 1)` on a non-negative (post-ReLU) NCHW map: zero padding then equals
/// `-inf` padding; the nine strided taps are index-selects (all with candle backwards).
fn max_pool_3x3_s2(x: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let (oh, ow) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
    let p = x.pad_with_zeros(2, 1, 2)?.pad_with_zeros(3, 1, 2)?;
    let dev = x.device();
    let idx = |start: usize, n: usize| -> Result<Tensor> {
        let v: Vec<u32> = (0..n).map(|i| (start + 2 * i) as u32).collect();
        Ok(Tensor::from_vec(v, n, dev)?)
    };
    let mut out: Option<Tensor> = None;
    for dy in 0..3 {
        let rows = p.index_select(&idx(dy, oh)?, 2)?;
        for dx in 0..3 {
            let v = rows.index_select(&idx(dx, ow)?, 3)?;
            out = Some(match out {
                Some(o) => o.maximum(&v)?,
                None => v,
            });
        }
    }
    Ok(out.expect("nine taps"))
}

impl HybrikEncoder {
    /// Load from a directory holding the checkpoint (`model.safetensors`, HybrIK key layout).
    pub fn from_dir(
        dir: impl AsRef<std::path::Path>,
        cfg: HybrikConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::from_weights(&super::load_dir(dir.as_ref(), device)?, "", cfg)
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
        let pair = |k: &str| -> Result<(Tensor, Tensor)> {
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
            init_shape: w32(w, prefix, "init_shape")?.reshape((1, ()))?,
            cfg,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &HybrikConfig {
        &self.cfg
    }

    /// HybrIK-normalized NHWC `[B, S, S, 3]` → betas `[B, 10]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.permute([0, 3, 1, 2])?.contiguous()?;
        let mut y = max_pool_3x3_s2(&self.stem.forward(&x)?.relu()?)?;
        for b in &self.blocks {
            let id = match &b.down {
                Some(d) => d.forward(&y)?,
                None => y.clone(),
            };
            let z = b.c2.forward(&b.c1.forward(&y)?.relu()?)?;
            y = (z + id)?.relu()?;
        }
        let feat = y.mean(3)?.mean(2)?;
        let h = linear(&feat, &self.fc1)?;
        let h = linear(&h, &self.fc2)?;
        Ok(linear(&h, &self.dec)?.broadcast_add(&self.init_shape)?)
    }

    /// The differentiable pixel entry point: NHWC `[1, H, W, 3]` in `[0, 1]` cropped to the
    /// half-open `crop = (y0, y1, x0, x1)`, resized to the square input (bilinear,
    /// `align_corners=False`), HybrIK-normalized → betas `[1, 10]`.
    pub fn forward_crop(
        &self,
        pixels: &Tensor,
        crop: (usize, usize, usize, usize),
    ) -> Result<Tensor> {
        let (y0, y1, x0, x1) = crop;
        let c = pixels.narrow(1, y0, y1 - y0)?.narrow(2, x0, x1 - x0)?;
        let s = self.cfg.input_size;
        let dev = pixels.device();
        let x = resample_nhwc(
            &c,
            &AxisMatrix::resize(y1 - y0, s, false, dev)?,
            &AxisMatrix::resize(x1 - x0, s, false, dev)?,
        )?;
        self.forward(&normalize(&x, HYBRIK_MEAN, HYBRIK_STD)?)
    }

    /// The crop of a person box on an `h × w` frame.
    pub fn crop_for(bbox: [f32; 4], h: usize, w: usize) -> (usize, usize, usize, usize) {
        hybrik_square_crop(bbox, h, w)
    }
}

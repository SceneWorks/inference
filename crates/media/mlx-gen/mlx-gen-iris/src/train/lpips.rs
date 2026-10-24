//! The frozen **LPIPS (VGG, v0.1)** perceptual network of the restoration recipe, native MLX — the
//! `lpips==0.1.4` `LPIPS(net="vgg", version="0.1")` HYPIR's trainer builds
//! (`HYPIR/trainer/base.py::init_lpips`), training-only (never part of a restoration export).
//!
//! `ScalingLayer` (`(x − shift) / scale` on `[−1, 1]` inputs), the torchvision VGG16 feature stack
//! tapped after `relu1_2 … relu5_3` ([`vgg_slices`]), per-tap channel normalization
//! (`x / (‖x‖₂ + 1e-10)`), the squared difference through the 1×1 bias-free `lin` heads, spatial
//! mean, summed over the five taps → one distance per sample. Runs in f32 (HYPIR does not wrap the
//! LPIPS net in accelerate's autocast). Weights: torchvision's `vgg16-397923af.pth`
//! (`features.N.*`) + the LPIPS `vgg.pth` (`lin{k}.model.1.weight`).

use std::collections::HashMap;
use std::path::Path;

use gen_core::iris::restoration_train::{
    lpips_lin_keys, vgg_slices, VggOp, LPIPS_NORM_EPS, LPIPS_SCALE, LPIPS_SHIFT, VGG16_WIDTHS,
};
use mlx_gen::gen_core;
use mlx_gen::nn::conv2d;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{maximum, sqrt};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

enum Op {
    /// OHWI weight, bias.
    ConvRelu(Array, Array),
    MaxPool,
}

/// A loaded, frozen LPIPS-VGG network.
pub struct Lpips {
    slices: Vec<Vec<Op>>,
    /// One `[C]` lin vector per tap.
    lins: Vec<Array>,
}

fn take(map: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let a = map
        .get(key)
        .ok_or_else(|| Error::Msg(format!("LPIPS: weight {key} missing")))?;
    if a.shape() != shape {
        return Err(Error::Msg(format!(
            "LPIPS: {key} is {:?}, expected {shape:?}",
            a.shape()
        )));
    }
    Ok(a.as_dtype(Dtype::Float32)?)
}

/// 2×2 / stride-2 max-pool over NHWC (odd trailing row/col dropped, like torch).
fn max_pool2(x: &Array) -> Result<Array> {
    use mlx_rs::ops::indexing::IndexOp;
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1] / 2, sh[2] / 2, sh[3]);
    Ok(x.index_device(
        (.., ..2 * h, ..2 * w, ..),
        mlx_rs::Stream::task_local_or_default(),
    )
    .reshape(&[b, h, 2, w, 2, c])?
    .max_axes(&[2, 4], false)?)
}

impl Lpips {
    /// Build from torchvision VGG16 `features.N.{weight,bias}` and LPIPS `lin{k}.model.1.weight`
    /// maps (torch layouts), for channel `widths`.
    pub fn from_maps(
        vgg: &HashMap<String, Array>,
        lin: &HashMap<String, Array>,
        widths: [usize; 5],
    ) -> Result<Self> {
        let mut slices = Vec::with_capacity(5);
        for ops in vgg_slices(widths) {
            let mut s = Vec::with_capacity(ops.len());
            for op in ops {
                s.push(match op {
                    VggOp::MaxPool => Op::MaxPool,
                    VggOp::ConvRelu(idx, cin, cout) => {
                        let w = take(
                            vgg,
                            &format!("features.{idx}.weight"),
                            &[cout as i32, cin as i32, 3, 3],
                        )?
                        .transpose_axes(&[0, 2, 3, 1])?;
                        // A dense OHWI copy of the OIHW checkpoint weight.
                        let w = mlx_gen::array::contiguous(&w)?;
                        let b = take(vgg, &format!("features.{idx}.bias"), &[cout as i32])?;
                        eval([&w, &b])?;
                        Op::ConvRelu(w, b)
                    }
                });
            }
            slices.push(s);
        }
        let mut lins = Vec::with_capacity(5);
        for (k, &c) in widths.iter().enumerate() {
            let key = lpips_lin_keys(k)
                .into_iter()
                .find(|key| lin.contains_key(key))
                .ok_or_else(|| Error::Msg(format!("LPIPS: lin{k} weight missing")))?;
            let w = take(lin, &key, &[1, c as i32, 1, 1])?.reshape(&[c as i32])?;
            eval([&w])?;
            lins.push(w);
        }
        Ok(Self { slices, lins })
    }

    /// Load the pinned files (the VGG16 `features.*` tensors — safetensors or a torch `.pth` — and
    /// the LPIPS `vgg.pth`).
    pub fn load(vgg16: &Path, lin: &Path) -> Result<Self> {
        let read = |p: &Path, keep: &dyn Fn(&str) -> bool| -> Result<HashMap<String, Array>> {
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Ok(mlx_gen::gen_core::__model_input_reader!(
Array::load_safetensors(p),
"crates/media/mlx-gen/mlx-gen-iris/src/train/lpips.rs::load::Array::load_safetensors@4447",
TensorInput,
p,
|__model_input_path| Array::load_safetensors(__model_input_path)
)?
                .into_iter()
                .filter(|(k, _)| keep(k))
                .collect())
            } else {
                mlx_gen::pth::load_pth_f32_filtered(p, keep)
            }
        };
        let vgg = read(vgg16, &|k| k.starts_with("features."))?;
        let lin = read(lin, &|_| true)?;
        Self::from_maps(&vgg, &lin, VGG16_WIDTHS)
    }

    /// The network on the deterministic formula weights the parity fixture uses
    /// (`mlx_gen::train::formula`, torchvision key names).
    pub fn formula(widths: [usize; 5]) -> Result<Self> {
        use mlx_gen::train::formula::{formula_array, Role};
        let mut vgg = HashMap::new();
        for op in vgg_slices(widths).into_iter().flatten() {
            if let VggOp::ConvRelu(idx, cin, cout) = op {
                let wk = format!("features.{idx}.weight");
                let bk = format!("features.{idx}.bias");
                vgg.insert(
                    wk.clone(),
                    formula_array(&wk, &[cout as i32, cin as i32, 3, 3], Role::Conv),
                );
                vgg.insert(bk.clone(), formula_array(&bk, &[cout as i32], Role::Bias));
            }
        }
        let mut lin = HashMap::new();
        for (k, &c) in widths.iter().enumerate() {
            let key = format!("lin{k}.model.1.weight");
            lin.insert(
                key.clone(),
                formula_array(&key, &[1, c as i32, 1, 1], Role::Lin),
            );
        }
        Self::from_maps(&vgg, &lin, widths)
    }

    /// `LPIPS(in0, in1)` per sample (`[B]`, f32) for NCHW inputs in `[−1, 1]`. Differentiable in
    /// both inputs.
    pub fn distance(&self, in0: &Array, in1: &Array) -> Result<Array> {
        let mut layers = self.layer_distances(in0, in1)?.into_iter();
        let mut total = layers
            .next()
            .ok_or_else(|| Error::Msg("LPIPS: no taps".into()))?;
        for l in layers {
            total = total.add(&l)?;
        }
        Ok(total)
    }

    /// The five per-tap terms (`[B]` each) whose sum is [`Self::distance`].
    pub fn layer_distances(&self, in0: &Array, in1: &Array) -> Result<Vec<Array>> {
        if in0.shape() != in1.shape() || in0.ndim() != 4 || in0.shape()[1] != 3 {
            return Err(Error::Msg(format!(
                "LPIPS: inputs must be matching [B, 3, H, W], got {:?} and {:?}",
                in0.shape(),
                in1.shape()
            )));
        }
        let shift = Array::from_slice(&LPIPS_SHIFT, &[1, 1, 1, 3]);
        let scale = Array::from_slice(&LPIPS_SCALE, &[1, 1, 1, 3]);
        let prep = |x: &Array| -> Result<Array> {
            Ok(x.as_dtype(Dtype::Float32)?
                .transpose_axes(&[0, 2, 3, 1])?
                .subtract(&shift)?
                .divide(&scale)?)
        };
        let (mut h0, mut h1) = (prep(in0)?, prep(in1)?);
        let zero = Array::from_f32(0.0);
        let eps = Array::from_f32(LPIPS_NORM_EPS);
        let mut out = Vec::with_capacity(5);
        for (slice, lin) in self.slices.iter().zip(&self.lins) {
            for op in slice {
                let step = |h: &Array| -> Result<Array> {
                    Ok(match op {
                        Op::ConvRelu(w, b) => {
                            let h = mlx_gen::array::contiguous(h)?;
                            maximum(&conv2d(&h, w, None, 1, 1)?.add(b)?, &zero)?
                        }
                        Op::MaxPool => max_pool2(h)?,
                    })
                };
                h0 = step(&h0)?;
                h1 = step(&h1)?;
            }
            // `sqrt(max(Σc x², 1e-30))`: identical to lpips' `sqrt(Σc x²)` wherever a pixel has
            // any non-zero channel (the floor is 1e-15 against the 1e-10 eps), and it keeps the
            // gradient finite at an all-zero (all-ReLU-dead) pixel, where `sqrt`'s derivative is
            // infinite and torch's backward produces NaN.
            let floor = Array::from_f32(1e-30);
            let norm = |h: &Array| -> Result<Array> {
                let n = sqrt(&maximum(&h.square()?.sum_axes(&[3], true)?, &floor)?)?;
                Ok(h.divide(&n.add(&eps)?)?)
            };
            let d = norm(&h0)?.subtract(&norm(&h1)?)?.square()?;
            // lin: 1×1 conv to one channel, then the spatial mean.
            let layer = d
                .multiply(lin)?
                .sum_axes(&[3], false)?
                .mean_axes(&[1, 2], false)?;
            out.push(layer);
        }
        Ok(out)
    }
}

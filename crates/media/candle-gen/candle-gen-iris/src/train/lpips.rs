//! The frozen **LPIPS (VGG, v0.1)** perceptual network of the restoration recipe on Candle — the
//! twin of `mlx_gen_iris::train::lpips` (`lpips==0.1.4` `LPIPS(net="vgg", version="0.1")`, the
//! network HYPIR's trainer builds), training-only. NCHW, f32.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::{CandleError as Error, Result};
use gen_core_iris::restoration_train::{
    lpips_lin_keys, vgg_slices, VggOp, LPIPS_NORM_EPS, LPIPS_SCALE, LPIPS_SHIFT, VGG16_WIDTHS,
};

enum Op {
    ConvRelu(Tensor, Tensor),
    MaxPool,
}

/// A loaded, frozen LPIPS-VGG network.
pub struct Lpips {
    slices: Vec<Vec<Op>>,
    /// `[1, C, 1, 1]` lin weights.
    lins: Vec<Tensor>,
    device: Device,
}

fn take(map: &HashMap<String, Tensor>, key: &str, shape: &[usize], dev: &Device) -> Result<Tensor> {
    let a = map
        .get(key)
        .ok_or_else(|| Error::Msg(format!("LPIPS: weight {key} missing")))?;
    if a.dims() != shape {
        return Err(Error::Msg(format!(
            "LPIPS: {key} is {:?}, expected {shape:?}",
            a.dims()
        )));
    }
    Ok(a.to_dtype(DType::F32)?.to_device(dev)?)
}

impl Lpips {
    /// Build from torchvision VGG16 `features.N.*` and LPIPS `lin{k}.model.1.weight` maps.
    pub fn from_maps(
        vgg: &HashMap<String, Tensor>,
        lin: &HashMap<String, Tensor>,
        widths: [usize; 5],
        device: &Device,
    ) -> Result<Self> {
        let mut slices = Vec::with_capacity(5);
        for ops in vgg_slices(widths) {
            let mut s = Vec::with_capacity(ops.len());
            for op in ops {
                s.push(match op {
                    VggOp::MaxPool => Op::MaxPool,
                    VggOp::ConvRelu(idx, cin, cout) => Op::ConvRelu(
                        take(
                            vgg,
                            &format!("features.{idx}.weight"),
                            &[cout, cin, 3, 3],
                            device,
                        )?,
                        take(vgg, &format!("features.{idx}.bias"), &[cout], device)?
                            .reshape((1, cout, 1, 1))?,
                    ),
                });
            }
            slices.push(s);
        }
        let lins = widths
            .iter()
            .enumerate()
            .map(|(k, &c)| {
                let key = lpips_lin_keys(k)
                    .into_iter()
                    .find(|key| lin.contains_key(key))
                    .ok_or_else(|| Error::Msg(format!("LPIPS: lin{k} weight missing")))?;
                take(lin, &key, &[1, c, 1, 1], device)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            slices,
            lins,
            device: device.clone(),
        })
    }

    /// Load the pinned files (VGG16 `features.*` — safetensors or `.pth` — and the LPIPS `vgg.pth`).
    pub fn load(vgg16: &Path, lin: &Path, device: &Device) -> Result<Self> {
        let read = |p: &Path, keep: &dyn Fn(&str) -> bool| -> Result<HashMap<String, Tensor>> {
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                return Ok(candle_gen::candle_core::safetensors::load(p, &Device::Cpu)
                    .map_err(|e| Error::Msg(format!("LPIPS: read {}: {e}", p.display())))?
                    .into_iter()
                    .filter(|(k, _)| keep(k))
                    .collect());
            }
            Ok(candle_gen::candle_core::pickle::read_all(p)
                .map_err(|e| Error::Msg(format!("LPIPS: read {}: {e}", p.display())))?
                .into_iter()
                .filter(|(k, _)| keep(k))
                .collect())
        };
        let vgg = read(vgg16, &|k| k.starts_with("features."))?;
        let lin = read(lin, &|_| true)?;
        Self::from_maps(&vgg, &lin, VGG16_WIDTHS, device)
    }

    /// The network on the parity fixture's deterministic formula weights.
    pub fn formula(widths: [usize; 5], device: &Device) -> Result<Self> {
        use candle_gen::train::formula::{formula_tensor, Role};
        let mut vgg = HashMap::new();
        for op in vgg_slices(widths).into_iter().flatten() {
            if let VggOp::ConvRelu(idx, cin, cout) = op {
                let wk = format!("features.{idx}.weight");
                let bk = format!("features.{idx}.bias");
                vgg.insert(
                    wk.clone(),
                    formula_tensor(&wk, &[cout, cin, 3, 3], Role::Conv, device)?,
                );
                vgg.insert(
                    bk.clone(),
                    formula_tensor(&bk, &[cout], Role::Bias, device)?,
                );
            }
        }
        let mut lin = HashMap::new();
        for (k, &c) in widths.iter().enumerate() {
            let key = format!("lin{k}.model.1.weight");
            lin.insert(
                key.clone(),
                formula_tensor(&key, &[1, c, 1, 1], Role::Lin, device)?,
            );
        }
        Self::from_maps(&vgg, &lin, widths, device)
    }

    /// The five per-tap terms (`[B]` each), NCHW inputs in `[−1, 1]`.
    pub fn layer_distances(&self, in0: &Tensor, in1: &Tensor) -> Result<Vec<Tensor>> {
        if in0.dims() != in1.dims() || in0.rank() != 4 || in0.dim(1)? != 3 {
            return Err(Error::Msg(format!(
                "LPIPS: inputs must be matching [B, 3, H, W], got {:?} and {:?}",
                in0.dims(),
                in1.dims()
            )));
        }
        let dev = &self.device;
        let shift = Tensor::new(&LPIPS_SHIFT, dev)?.reshape((1, 3, 1, 1))?;
        let scale = Tensor::new(&LPIPS_SCALE, dev)?.reshape((1, 3, 1, 1))?;
        let prep = |x: &Tensor| -> Result<Tensor> {
            Ok(x.to_dtype(DType::F32)?
                .broadcast_sub(&shift)?
                .broadcast_div(&scale)?)
        };
        let (mut h0, mut h1) = (prep(in0)?, prep(in1)?);
        let floor = Tensor::new(1e-30f32, dev)?;
        let eps = Tensor::new(LPIPS_NORM_EPS, dev)?;
        let mut out = Vec::with_capacity(5);
        for (slice, lin) in self.slices.iter().zip(&self.lins) {
            for op in slice {
                let step = |h: &Tensor| -> Result<Tensor> {
                    Ok(match op {
                        Op::ConvRelu(w, b) => {
                            // `x · 1[x > 0]`: torch's ReLU, whose derivative is 0 at 0
                            // (candle's `relu` backward passes the gradient at exactly 0).
                            let y = h.conv2d(w, 1, 1, 1, 1)?.broadcast_add(b)?;
                            let on = y.gt(0f32)?.to_dtype(DType::F32)?;
                            y.mul(&on)?
                        }
                        Op::MaxPool => h.max_pool2d(2)?,
                    })
                };
                h0 = step(&h0)?;
                h1 = step(&h1)?;
            }
            // `sqrt(max(Σc x², 1e-30))` — see the MLX twin: lpips' norm with a finite gradient at
            // an all-zero pixel.
            let norm = |h: &Tensor| -> Result<Tensor> {
                let n = h.sqr()?.sum_keepdim(1)?.broadcast_maximum(&floor)?.sqrt()?;
                Ok(h.broadcast_div(&n.broadcast_add(&eps)?)?)
            };
            let d = norm(&h0)?.sub(&norm(&h1)?)?.sqr()?;
            let layer = d
                .broadcast_mul(lin)?
                .sum(1)?
                .mean(D::Minus1)?
                .mean(D::Minus1)?;
            out.push(layer);
        }
        Ok(out)
    }

    /// `LPIPS(in0, in1)` per sample (`[B]`, f32). Differentiable in both inputs.
    pub fn distance(&self, in0: &Tensor, in1: &Tensor) -> Result<Tensor> {
        let mut layers = self.layer_distances(in0, in1)?.into_iter();
        let mut total = layers
            .next()
            .ok_or_else(|| Error::Msg("LPIPS: no taps".into()))?;
        for l in layers {
            total = total.add(&l)?;
        }
        Ok(total)
    }
}

//! Krea's first-order RMS/RoPE owners. Forward preserves the existing f32
//! arithmetic/rounding; the outer tape keeps inputs, not their full f32 chains.
//! These explicit entrypoints do not change other families' normalization.

use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor, D,
};
use std::sync::{Arc, Mutex};

struct Output {
    slot: Arc<Mutex<Option<Storage>>>,
    shape: Shape,
}

impl Output {
    fn new(value: Tensor) -> Result<Self> {
        if value.track_op() {
            candle_core::bail!("Krea owner cannot capture an inner autograd tape");
        }
        let value = value.contiguous()?;
        let (storage, layout) = value.storage_and_layout();
        Ok(Self {
            slot: Arc::new(Mutex::new(Some(storage.try_clone(layout)?))),
            shape: value.shape().clone(),
        })
    }
    fn take(&self) -> Result<Storage> {
        crate::lock_recover(&self.slot)
            .take()
            .ok_or_else(|| candle_core::Error::Msg("Krea owner output already consumed".into()))
    }
    fn cpu(&self) -> Result<(CpuStorage, Shape)> {
        match self.take()? {
            Storage::Cpu(s) => Ok((s, self.shape.clone())),
            _ => candle_core::bail!("Krea owner output device mismatch"),
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda(&self) -> Result<(candle_core::CudaStorage, Shape)> {
        match self.take()? {
            Storage::Cuda(s) => Ok((s, self.shape.clone())),
            _ => candle_core::bail!("Krea owner output device mismatch"),
        }
    }
}

fn supported(x: &Tensor) -> bool {
    (x.device().is_cpu() || x.device().is_cuda())
        && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
}

fn normalize(x: &Tensor, eps: f64, reference: bool) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let width = xf.dim(D::Minus1)?;
    let rms = ((xf.sqr()?.sum_keepdim(D::Minus1)? / width as f64)? + eps)?.sqrt()?;
    if reference {
        xf.broadcast_div(&rms)
    } else {
        xf.broadcast_mul(&rms.recip()?)
    }
}

/// Krea's weight-before-cast RMS contract, including both input and weight gradients.
/// The profile's divide/reciprocal choice is captured per forward, never read later in backward.
pub fn rms_scale(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    rms_scale_impl(x, weight, eps, super::profile::reference_math())
}
fn rms_scale_impl(x: &Tensor, weight: &Tensor, eps: f64, reference: bool) -> Result<Tensor> {
    let broadcast_into_x = weight.rank() <= x.rank()
        && weight
            .dims()
            .iter()
            .rev()
            .zip(x.dims().iter().rev())
            .all(|(w, x)| *w == 1 || w == x);
    if !supported(x)
        || weight.dtype() != DType::F32
        || !x.device().same_device(weight.device())
        || !broadcast_into_x
        || !eps.is_finite()
        || eps < 0.
        || x.dims().last().copied().unwrap_or(0) == 0
    {
        return super::norm::rms_norm_diff(x, weight, eps);
    }
    let out = normalize(&x.detach(), eps, reference)?
        .broadcast_mul(&weight.detach())?
        .to_dtype(x.dtype())?;
    if !x.track_op() && !weight.track_op() {
        return Ok(out);
    }
    x.apply_op2(
        weight,
        Rms {
            output: Output::new(out)?,
            eps,
            reference,
        },
    )
}

fn reduce_like(mut value: Tensor, target: &Tensor) -> Result<Tensor> {
    let leading = value.rank() - target.rank();
    for axis in (0..value.rank()).rev() {
        if axis < leading || target.dims()[axis - leading] == 1 {
            value = value.sum_keepdim(axis)?;
        }
    }
    value.reshape(target.shape())?.to_dtype(target.dtype())
}

struct Rms {
    output: Output,
    eps: f64,
    reference: bool,
}
impl CustomOp2 for Rms {
    fn name(&self) -> &'static str {
        "krea-rms-training"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        self.output.cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &Layout,
        _: &candle_core::CudaStorage,
        _: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        self.output.cuda()
    }
    fn bwd(
        &self,
        x: &Tensor,
        weight: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let xf = x.detach().to_dtype(DType::F32)?;
        let wf = weight.detach();
        let g = grad.detach().to_dtype(DType::F32)?;
        let width = xf.dim(D::Minus1)? as f64;
        let inv = (((xf.sqr()?.sum_keepdim(D::Minus1)? / width)? + self.eps)?.sqrt()?).recip()?;
        let gx = g.broadcast_mul(&wf)?;
        let dot = ((&gx * &xf)?.sum_keepdim(D::Minus1)? / width)?;
        let dx = (gx - xf.broadcast_mul(&(dot * inv.sqr()?)?)?)?
            .broadcast_mul(&inv)?
            .to_dtype(x.dtype())?;
        let dw = reduce_like((g * normalize(&xf, self.eps, self.reference)?)?, weight)?;
        Ok((Some(dx), Some(dw)))
    }
}

fn parts(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, s, h, d) = x.dims4()?;
    let pairs = x.to_dtype(DType::F32)?.reshape((b, s, h, d / 2, 2))?;
    Ok((
        pairs
            .narrow(4, 0, 1)?
            .contiguous()?
            .reshape((b, s, h, d / 2))?,
        pairs
            .narrow(4, 1, 1)?
            .contiguous()?
            .reshape((b, s, h, d / 2))?,
    ))
}
fn interleave(even: &Tensor, odd: &Tensor, shape: &Shape, dtype: DType) -> Result<Tensor> {
    Tensor::stack(&[even, odd], D::Minus1)?
        .reshape(shape)?
        .to_dtype(dtype)
}
fn rope_forward(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, s, _, d) = x.dims4()?;
    let cos = cos.contiguous()?.reshape((1, s, 1, d / 2))?;
    let sin = sin.contiguous()?.reshape((1, s, 1, d / 2))?;
    let (xe, xo) = parts(x)?;
    let even = (xe.broadcast_mul(&cos)? - xo.broadcast_mul(&sin)?)?;
    let odd = (xe.broadcast_mul(&sin)? + xo.broadcast_mul(&cos)?)?;
    interleave(&even, &odd, x.shape(), x.dtype())
}

/// Krea's sequence-major interleaved rotation. F32 operations precede the public
/// cast, as in the inference reference; gradients include input and both tables.
pub fn rotary_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if !supported(x)
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !x.device().same_device(cos.device())
        || !x.device().same_device(sin.device())
    {
        return rope_forward(x, cos, sin);
    }
    let out = rope_forward(&x.detach(), &cos.detach(), &sin.detach())?;
    if !x.track_op() && !cos.track_op() && !sin.track_op() {
        return Ok(out);
    }
    x.apply_op3(
        cos,
        sin,
        Rope {
            output: Output::new(out)?,
        },
    )
}
struct Rope {
    output: Output,
}
impl CustomOp3 for Rope {
    fn name(&self) -> &'static str {
        "krea-rope-training"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        self.output.cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &Layout,
        _: &candle_core::CudaStorage,
        _: &Layout,
        _: &candle_core::CudaStorage,
        _: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        self.output.cuda()
    }
    fn bwd(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let (_, s, _, d) = x.dims4()?;
        let c = cos.detach().contiguous()?.reshape((1, s, 1, d / 2))?;
        let sn = sin.detach().contiguous()?.reshape((1, s, 1, d / 2))?;
        let (xe, xo) = parts(&x.detach())?;
        let (ge, go) = parts(&grad.detach())?;
        let dx = interleave(
            &(ge.broadcast_mul(&c)? + go.broadcast_mul(&sn)?)?,
            &(go.broadcast_mul(&c)? - ge.broadcast_mul(&sn)?)?,
            x.shape(),
            x.dtype(),
        )?;
        let dc = ((&ge * &xe)? + (&go * &xo)?)?
            .sum_keepdim(0)?
            .sum_keepdim(2)?
            .reshape(cos.shape())?
            .to_dtype(cos.dtype())?;
        let ds = ((go * xe)? - (ge * xo)?)?
            .sum_keepdim(0)?
            .sum_keepdim(2)?
            .reshape(sin.shape())?
            .to_dtype(sin.dtype())?;
        Ok((Some(dx), Some(dc), Some(ds)))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn assert_close(label: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        assert_eq!(actual.dtype(), expected.dtype(), "{label}");
        assert_eq!(actual.dims(), expected.dims(), "{label}");
        let af = actual.to_dtype(DType::F32)?;
        assert!(
            af.abs()?.sum_all()?.to_scalar::<f32>()? > 0.,
            "{label}: vacuous zero result"
        );
        let delta = (af - expected.to_dtype(DType::F32)?)?.abs()?;
        assert!(delta.sum_all()?.to_scalar::<f32>()?.is_finite(), "{label}");
        let error = delta.max_all()?.to_scalar::<f32>()?;
        assert!(error < tolerance, "{label}: {error} >= {tolerance}");
        Ok(())
    }

    // Raw tensor arithmetic and autograd only. No learned module/optimizer; the
    // unconditional ignore confines numerical execution to the manual CUDA lane.
    #[test]
    #[ignore = "CUDA numerical proof; explicit manual raw arithmetic workflow"]
    fn rms_rope_forward_backward_matches_composable() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values = |n, offset| {
            (0..n)
                .map(|i| (((i + offset) * 17 % 97) as f32 - 48.) / 64.)
                .collect::<Vec<_>>()
        };
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let tolerance = match dtype {
                DType::F32 => 2e-5,
                DType::F16 => 0.003,
                _ => 0.025,
            };
            let x = Var::from_tensor(
                &Tensor::from_vec(values(96, 0), (2, 3, 2, 8), &device)?.to_dtype(dtype)?,
            )?;
            let weights =
                Var::from_tensor(&((Tensor::from_vec(values(8, 4), 8, &device)? * 0.2)? + 1.)?)?;
            let cos = Var::from_tensor(&Tensor::from_vec(values(12, 5), (1, 3, 4), &device)?)?;
            let sin = Var::from_tensor(&Tensor::from_vec(values(12, 7), (1, 3, 4), &device)?)?;
            // Upstream chain, a nonunit stride and nonzero storage offset.
            let xscaled = (x.as_tensor() * 0.75)?
                .transpose(2, 3)?
                .contiguous()?
                .transpose(2, 3)?;
            let view = Tensor::cat(&[&xscaled, &xscaled], 1)?.narrow(1, 3, 3)?;
            let cotangent = Tensor::from_vec(values(96, 13), (2, 3, 2, 8), &device)?;
            for reference in [true, false] {
                for broadcast in [false, true] {
                    let w = if broadcast {
                        weights.reshape((1, 1, 1, 8))?
                    } else {
                        weights.as_tensor().clone()
                    };
                    let legacy = normalize(&view, 1e-5, reference)?
                        .broadcast_mul(&w)?
                        .to_dtype(dtype)?;
                    let actual = rms_scale_impl(&view, &w, 1e-5, reference)?;
                    let ga = actual
                        .to_dtype(DType::F32)?
                        .mul(&cotangent)?
                        .sum_all()?
                        .backward()?;
                    let gr = legacy
                        .to_dtype(DType::F32)?
                        .mul(&cotangent)?
                        .sum_all()?
                        .backward()?;
                    assert_close("rms output", &actual, &legacy, tolerance)?;
                    assert_close(
                        "rms dx",
                        ga.get(&x).unwrap(),
                        gr.get(&x).unwrap(),
                        tolerance,
                    )?;
                    assert_close(
                        "rms dweight",
                        ga.get(&weights).unwrap(),
                        gr.get(&weights).unwrap(),
                        tolerance,
                    )?;
                    let detached = normalize(&view.detach(), 1e-5, reference)?
                        .broadcast_mul(&w.detach())?
                        .to_dtype(dtype)?;
                    let output = Output::new(detached)?;
                    let slot = output.slot.clone();
                    let _ = view.apply_op2(
                        &w,
                        Rms {
                            output,
                            eps: 1e-5,
                            reference,
                        },
                    )?;
                    assert!(crate::lock_recover(&slot).is_none());
                    assert!(Output::new(legacy).is_err(), "nested tape must be refused");
                }
            }
            for _ in 0..3 {
                let actual = rotary_interleaved(&view, &cos, &sin)?;
                let legacy = rope_forward(&view, &cos, &sin)?;
                let ga = actual
                    .to_dtype(DType::F32)?
                    .mul(&cotangent)?
                    .sum_all()?
                    .backward()?;
                let gr = legacy
                    .to_dtype(DType::F32)?
                    .mul(&cotangent)?
                    .sum_all()?
                    .backward()?;
                for (name, a, r) in [
                    ("rope output", &actual, &legacy),
                    ("rope dx", ga.get(&x).unwrap(), gr.get(&x).unwrap()),
                    ("rope dcos", ga.get(&cos).unwrap(), gr.get(&cos).unwrap()),
                    ("rope dsin", ga.get(&sin).unwrap(), gr.get(&sin).unwrap()),
                ] {
                    assert_close(name, a, r, tolerance)?;
                }
                let output =
                    Output::new(rope_forward(&view.detach(), &cos.detach(), &sin.detach())?)?;
                let slot = output.slot.clone();
                let _ = view.apply_op3(&cos, &sin, Rope { output })?;
                assert!(crate::lock_recover(&slot).is_none());
            }
            // Actual composition: RMS's trainable weight and upstream input both
            // remain reachable through RoPE's custom owner.
            let actual = rotary_interleaved(&rms_scale(&view, &weights, 1e-5)?, &cos, &sin)?;
            let legacy = rope_forward(
                &super::super::norm::rms_norm_diff(&view, &weights, 1e-5)?,
                &cos,
                &sin,
            )?;
            let ga = actual
                .to_dtype(DType::F32)?
                .mul(&cotangent)?
                .sum_all()?
                .backward()?;
            let gr = legacy
                .to_dtype(DType::F32)?
                .mul(&cotangent)?
                .sum_all()?
                .backward()?;
            for var in [&x, &weights, &cos, &sin] {
                assert_close(
                    "composed gradient",
                    ga.get(var).unwrap(),
                    gr.get(var).unwrap(),
                    tolerance,
                )?;
            }
        }
        Ok(())
    }
}

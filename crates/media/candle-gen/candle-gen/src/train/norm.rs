//! Shared RMS normalization with bounded first-order tape ownership and explicit
//! f32 reduction/rounding boundaries. Graph-retaining backward and unsupported
//! devices/dtypes keep the original composable arithmetic.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Layout, Result, Shape, Storage, Tensor, D,
};
use std::sync::{Arc, Mutex};

pub(super) struct Output {
    pub(super) slot: Arc<Mutex<Option<Storage>>>,
    shape: Shape,
}

impl Output {
    pub(super) fn new(value: Tensor) -> Result<Self> {
        if value.track_op() {
            candle_core::bail!("training owner cannot capture an inner autograd tape");
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
            .ok_or_else(|| candle_core::Error::Msg("training owner output already consumed".into()))
    }
    pub(super) fn cpu(&self) -> Result<(CpuStorage, Shape)> {
        match self.take()? {
            Storage::Cpu(s) => Ok((s, self.shape.clone())),
            _ => candle_core::bail!("training owner output device mismatch"),
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn cuda(&self) -> Result<(candle_core::CudaStorage, Shape)> {
        match self.take()? {
            Storage::Cuda(s) => Ok((s, self.shape.clone())),
            _ => candle_core::bail!("training owner output device mismatch"),
        }
    }
}

pub(super) fn supported(x: &Tensor) -> bool {
    (x.device().is_cpu() || x.device().is_cuda())
        && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
}

pub(super) fn normalize_composable(x: &Tensor, eps: f64, reference: bool) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let width = xf.dim(D::Minus1)?;
    if width == 0 || !eps.is_finite() || eps < 0.0 {
        candle_core::bail!(
            "RMS norm requires a nonempty final dimension and finite nonnegative epsilon"
        );
    }
    let rms = ((xf.sqr()?.sum_keepdim(D::Minus1)? / width as f64)? + eps)?.sqrt()?;
    if reference {
        xf.broadcast_div(&rms)
    } else {
        xf.broadcast_mul(&rms.recip()?)
    }
}

/// Normalize in f32. First-order training retains only input/output, while the
/// graph-retaining opt-in keeps the original differentiable composable chain.
pub fn rms_normalize_f32(x: &Tensor, eps: f64) -> Result<Tensor> {
    rms_normalize_f32_impl(x, eps, super::profile::reference_math())
}

fn rms_normalize_f32_impl(x: &Tensor, eps: f64, reference: bool) -> Result<Tensor> {
    if candle_core::backprop::grad_do_not_detach() || !supported(x) {
        return normalize_composable(x, eps, reference);
    }
    let out = normalize_composable(&x.detach(), eps, reference)?;
    if !x.track_op() {
        return Ok(out);
    }
    x.apply_op1(Normalize {
        output: Output::new(out)?,
        eps,
    })
}

/// Multiply the f32 weight before the final activation cast. Input and broadcast
/// weight gradients remain live, including upstream parameter transformations.
pub fn rms_norm_diff(x: &Tensor, weight_f32: &Tensor, eps: f64) -> Result<Tensor> {
    rms_norm_diff_impl(x, weight_f32, eps, super::profile::reference_math())
}

/// Cast normalized values before multiplying weight. This earlier rounding point
/// is deliberate and must not be folded into the weight-before-cast owner.
pub fn rms_norm_cast_before_weight(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    rms_normalize_f32(x, eps)?
        .to_dtype(x.dtype())?
        .broadcast_mul(weight)
}

struct Normalize {
    output: Output,
    eps: f64,
}
impl CustomOp1 for Normalize {
    fn name(&self) -> &'static str {
        "rms-normalize-training"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        self.output.cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        self.output.cuda()
    }
    fn bwd(&self, x: &Tensor, _: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        let xf = x.detach().to_dtype(DType::F32)?;
        let g = grad.detach().to_dtype(DType::F32)?;
        let width = xf.dim(D::Minus1)? as f64;
        let inv = (((xf.sqr()?.sum_keepdim(D::Minus1)? / width)? + self.eps)?.sqrt()?).recip()?;
        let dot = ((&g * &xf)?.sum_keepdim(D::Minus1)? / width)?;
        let dx = (g - xf.broadcast_mul(&(dot * inv.sqr()?)?)?)?
            .broadcast_mul(&inv)?
            .to_dtype(x.dtype())?;
        Ok(Some(dx))
    }
}

fn rms_norm_diff_impl(x: &Tensor, weight: &Tensor, eps: f64, reference: bool) -> Result<Tensor> {
    let broadcast_into_x = weight.rank() <= x.rank()
        && weight
            .dims()
            .iter()
            .rev()
            .zip(x.dims().iter().rev())
            .all(|(w, x)| *w == 1 || w == x);
    if candle_core::backprop::grad_do_not_detach()
        || !supported(x)
        || weight.dtype() != DType::F32
        || !x.device().same_device(weight.device())
        || !broadcast_into_x
        || !eps.is_finite()
        || eps < 0.
        || x.dims().last().copied().unwrap_or(0) == 0
    {
        return normalize_composable(x, eps, reference)?
            .broadcast_mul(weight)?
            .to_dtype(x.dtype());
    }
    let out = normalize_composable(&x.detach(), eps, reference)?
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
        "rms-training"
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
        let dw = reduce_like(
            (g * normalize_composable(&xf, self.eps, self.reference)?)?,
            weight,
        )?;
        Ok((Some(dx), Some(dw)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn reciprocal_norm_matches_reference_values_and_gradients() -> candle_core::Result<()> {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = Var::from_tensor(
                &Tensor::from_vec(
                    vec![0.01f32, -0.4, 2.1, 0.7, 1.2, -3.0, 0.5, 0.1],
                    (2, 4),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?,
            )?;
            let weight = Tensor::from_vec(vec![1.1f32, 0.9, 0.8, 1.2], 4, &Device::Cpu)?;
            let xf = x.to_dtype(DType::F32)?;
            let mean = (xf.sqr()?.sum_keepdim(D::Minus1)? / 4.0)?;
            let reference = xf
                .broadcast_div(&(mean + 1e-6)?.sqrt()?)?
                .broadcast_mul(&weight)?
                .to_dtype(dtype)?;
            let actual = rms_norm_diff(&x, &weight, 1e-6)?;
            let tol = if dtype == DType::F32 { 1e-6 } else { 0.02 };
            let error = (&actual.to_dtype(DType::F32)? - &reference.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error <= tol, "{dtype:?} error {error}");
            let ga = actual.to_dtype(DType::F32)?.sqr()?.sum_all()?.backward()?;
            let gr = reference
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .backward()?;
            let error = (ga.get(&x).unwrap().to_dtype(DType::F32)?
                - gr.get(&x).unwrap().to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
            assert!(error <= tol, "{dtype:?} gradient error {error}");
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod ownership_tests {
    use super::*;
    use candle_core::{Device, Var};

    fn values(n: usize, offset: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i + offset) * 17 % 97) as f32 - 48.) / 64.)
            .collect()
    }
    fn close(label: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        assert_eq!(actual.dtype(), expected.dtype(), "{label}");
        assert_eq!(actual.dims(), expected.dims(), "{label}");
        let af = actual.to_dtype(DType::F32)?;
        assert!(
            af.abs()?.sum_all()?.to_scalar::<f32>()? > 0.,
            "{label}: zero result"
        );
        let delta = (af - expected.to_dtype(DType::F32)?)?.abs()?;
        assert!(delta.sum_all()?.to_scalar::<f32>()?.is_finite(), "{label}");
        let error = delta.max_all()?.to_scalar::<f32>()?;
        assert!(error < tolerance, "{label}: {error} >= {tolerance}");
        Ok(())
    }
    fn gradients(
        actual: &Tensor,
        legacy: &Tensor,
        cotangent: &Tensor,
        vars: &[&Var],
        tolerance: f32,
    ) -> Result<()> {
        close("output", actual, legacy, tolerance)?;
        let ga = actual
            .to_dtype(DType::F32)?
            .mul(cotangent)?
            .sum_all()?
            .backward()?;
        let gr = legacy
            .to_dtype(DType::F32)?
            .mul(cotangent)?
            .sum_all()?
            .backward()?;
        for var in vars {
            close(
                "gradient",
                ga.get(var).unwrap(),
                gr.get(var).unwrap(),
                tolerance,
            )?;
        }
        Ok(())
    }

    // Raw arithmetic only. Deliberately ignored even though no learned module is
    // reachable: the manual CUDA lane owns numerical validation of this change.
    #[test]
    #[ignore = "CUDA numerical proof; explicit manual raw arithmetic workflow"]
    fn shared_rms_contracts_match_composable() -> Result<()> {
        assert!(!candle_core::backprop::grad_do_not_detach());
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let tolerance = match dtype {
                DType::F32 => 2e-5,
                DType::F16 => 0.003,
                _ => 0.025,
            };
            let x = Var::from_tensor(
                &Tensor::from_vec(values(96, 0), (2, 3, 2, 8), &device)?.to_dtype(dtype)?,
            )?;
            // Nontrivial upstream chain, stride and offset; neither input nor
            // weight may be converted to a fresh gradient leaf by the owner.
            let scaled = (x.as_tensor() * 0.75)?
                .transpose(2, 3)?
                .contiguous()?
                .transpose(2, 3)?;
            let view = Tensor::cat(&[&scaled, &scaled], 1)?.narrow(1, 3, 3)?;
            let cotangent = Tensor::from_vec(values(96, 13), (2, 3, 2, 8), &device)?;
            for reference in [true, false] {
                let legacy = normalize_composable(&view, 1e-5, reference)?;
                let actual = rms_normalize_f32_impl(&view, 1e-5, reference)?;
                assert_eq!(actual.dtype(), DType::F32);
                gradients(&actual, &legacy, &cotangent, &[&x], tolerance)?;
                // Ownership must retain only the input and this result, not a
                // second inner chain or the transferred output storage.
                assert_eq!(actual.sorted_nodes().len(), view.sorted_nodes().len() + 1);
                let output = Output::new(normalize_composable(&view.detach(), 1e-5, reference)?)?;
                let slot = output.slot.clone();
                let _ = view.apply_op1(Normalize { output, eps: 1e-5 })?;
                assert!(crate::lock_recover(&slot).is_none());
                assert!(Output::new(legacy.clone()).is_err());
                for shape in [vec![8], vec![1, 3, 1, 8], vec![]] {
                    let count = shape.iter().product::<usize>();
                    let weight = Var::from_tensor(&Tensor::from_vec(
                        values(count, 4),
                        shape.as_slice(),
                        &device,
                    )?)?;
                    let w = ((weight.as_tensor() * 0.2)? + 1.)?;
                    let expected = legacy.broadcast_mul(&w)?.to_dtype(dtype)?;
                    let actual = rms_norm_diff_impl(&view, &w, 1e-5, reference)?;
                    gradients(&actual, &expected, &cotangent, &[&x, &weight], tolerance)?;
                    let output = Output::new(
                        normalize_composable(&view.detach(), 1e-5, reference)?
                            .broadcast_mul(&w.detach())?
                            .to_dtype(dtype)?,
                    )?;
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
                    // Cast-before-weight: the low-precision multiplication and
                    // its weight gradient must remain outside the F32 core.
                    let low_w = w.to_dtype(dtype)?;
                    let expected = legacy.to_dtype(dtype)?.broadcast_mul(&low_w)?;
                    let actual = rms_normalize_f32_impl(&view, 1e-5, reference)?
                        .to_dtype(dtype)?
                        .broadcast_mul(&low_w)?;
                    gradients(&actual, &expected, &cotangent, &[&x, &weight], tolerance)?;
                    if reference == super::super::profile::reference_math() {
                        gradients(
                            &rms_norm_cast_before_weight(&view, &low_w, 1e-5)?,
                            &expected,
                            &cotangent,
                            &[&x, &weight],
                            tolerance,
                        )?;
                        let expected = legacy.broadcast_mul(&w)?.to_dtype(dtype)?;
                        gradients(
                            &rms_norm_diff(&view, &w, 1e-5)?,
                            &expected,
                            &cotangent,
                            &[&x, &weight],
                            tolerance,
                        )?;
                    }
                }
            }
            // Public unweighted contract, frozen graph, and original error path.
            close(
                "public core",
                &rms_normalize_f32(&view, 1e-5)?,
                &normalize_composable(&view, 1e-5, super::super::profile::reference_math())?,
                tolerance,
            )?;
            assert!(!rms_normalize_f32(&view.detach(), 1e-5)?.track_op());
            assert!(rms_normalize_f32(&view, -1.).is_err());
            assert!(rms_normalize_f32(&view, f64::NAN).is_err());
        }
        // Legal broadcast that expands x must retain the composable weighted
        // fallback (the fast owner only broadcasts weight into the input shape).
        let x = Var::from_tensor(&Tensor::from_vec(values(8, 1), (1, 8), &device)?)?;
        let w = Var::from_tensor(&Tensor::from_vec(values(16, 5), (2, 8), &device)?)?;
        let c = Tensor::from_vec(values(16, 11), (2, 8), &device)?;
        let expected = normalize_composable(&x, 1e-5, super::super::profile::reference_math())?
            .broadcast_mul(&w)?;
        gradients(
            &rms_norm_diff(&x, &w, 1e-5)?,
            &expected,
            &c,
            &[&x, &w],
            2e-5,
        )?;
        Ok(())
    }

    /// A fresh test process sets the existing Candle environment opt-in before
    /// its first query. No mutation of process-global environment during tests.
    #[test]
    #[ignore = "CUDA numerical proof; explicit manual graph-retaining raw arithmetic workflow"]
    fn graph_retaining_norm_rope_attention_fallbacks() -> Result<()> {
        assert!(candle_core::backprop::grad_do_not_detach());
        let device = Device::new_cuda(0)?;
        let x = Var::from_tensor(&Tensor::from_vec(values(32, 0), (1, 2, 2, 8), &device)?)?;
        let w = Var::from_tensor(&Tensor::from_vec(values(8, 4), 8, &device)?)?;
        let cotangent = Tensor::from_vec(values(32, 13), (1, 2, 2, 8), &device)?;
        let xf = (x.as_tensor() * 0.75)?;
        let normalized = normalize_composable(&xf, 1e-5, super::super::profile::reference_math())?;
        for (actual, legacy) in [
            (rms_normalize_f32(&xf, 1e-5)?, normalized.clone()),
            (rms_norm_diff(&xf, &w, 1e-5)?, normalized.broadcast_mul(&w)?),
            (
                rms_norm_cast_before_weight(&xf, &w, 1e-5)?,
                normalized.broadcast_mul(&w)?,
            ),
        ] {
            gradients(&actual, &legacy, &cotangent, &[&x], 2e-5)?;
            let g = actual.mul(&cotangent)?.sum_all()?.backward()?;
            let expected_g = legacy.mul(&cotangent)?.sum_all()?.backward()?;
            if let Some(dw) = expected_g.get(&w) {
                let actual_dw = g.get(&w).unwrap();
                close("graph-retaining weight gradient", actual_dw, dw, 2e-5)?;
                assert!(actual_dw.track_op());
                assert!(actual_dw.sorted_nodes().iter().any(|n| n.id() == x.id()));
            }
            let dx = g.get(&x).unwrap();
            assert!(dx.track_op());
            assert!(dx.sorted_nodes().iter().any(|n| n.id() == x.id()));
            assert_eq!(actual.sorted_nodes().len(), legacy.sorted_nodes().len());
        }
        let cos = Var::from_tensor(&Tensor::from_vec(values(8, 5), (1, 2, 4), &device)?)?;
        let sin = Var::from_tensor(&Tensor::from_vec(values(8, 7), (1, 2, 4), &device)?)?;
        let rotated = super::super::krea_ops::rotary_interleaved(&xf, &cos, &sin)?;
        let g = rotated.mul(&cotangent)?.sum_all()?.backward()?;
        let dx = g.get(&x).unwrap();
        assert!(dx.track_op());
        assert!(dx.sorted_nodes().iter().any(|n| n.id() == cos.id()));
        assert!(g
            .get(&cos)
            .unwrap()
            .sorted_nodes()
            .iter()
            .any(|n| n.id() == x.id()));
        // Exercise the public default Reference seam, rather than just its
        // private eligibility helper. Q/K/V retain the original graph.
        use crate::training_attention::{with_training_attention_mode, TrainingAttentionMode};
        use gen_core::attention_budget::{AttentionBudget, AttentionPlan};
        let k = Var::from_tensor(&Tensor::from_vec(values(32, 9), (1, 2, 2, 8), &device)?)?;
        let v = Var::from_tensor(&Tensor::from_vec(values(32, 11), (1, 2, 2, 8), &device)?)?;
        let plan = AttentionPlan {
            budget: AttentionBudget::from_score_elements(64, false),
            cancel: None,
        };
        let actual = with_training_attention_mode(TrainingAttentionMode::Reference, || {
            crate::fused_attention::training_sdpa_standard_bhsd(&xf, &k, &v, 0.4, None, plan)
        })
        .map_err(candle_core::Error::wrap)?;
        let legacy = crate::attention::sdpa_planned_bhsd(
            &xf,
            &k,
            &v,
            0.4,
            None,
            |s| candle_nn::ops::softmax(s, D::Minus1),
            plan,
        )
        .map_err(candle_core::Error::wrap)?;
        gradients(&actual, &legacy, &cotangent, &[&x, &k, &v], 2e-5)?;
        assert_eq!(actual.sorted_nodes().len(), legacy.sorted_nodes().len());
        let g = actual.mul(&cotangent)?.sum_all()?.backward()?;
        let dx = g.get(&x).unwrap();
        assert!(dx.track_op());
        assert!(dx.sorted_nodes().iter().any(|n| n.id() == x.id()));
        // Head width 32 would dispatch the CUDA candidate in first-order mode.
        // The cached graph-retaining opt-in must keep the composable graph even
        // when the caller explicitly selects the candidate.
        let q32 = Var::from_tensor(&Tensor::from_vec(values(64, 1), (1, 1, 2, 32), &device)?)?;
        let k32 = Var::from_tensor(&Tensor::from_vec(values(64, 2), (1, 1, 2, 32), &device)?)?;
        let v32 = Var::from_tensor(&Tensor::from_vec(values(64, 3), (1, 1, 2, 32), &device)?)?;
        let c32 = Tensor::from_vec(values(64, 4), (1, 1, 2, 32), &device)?;
        crate::training_attention::take_training_attention_counts();
        let selected =
            with_training_attention_mode(TrainingAttentionMode::StreamingCandidate, || {
                crate::fused_attention::training_sdpa_standard_bhsd(
                    &q32, &k32, &v32, 0.4, None, plan,
                )
            })
            .map_err(candle_core::Error::wrap)?;
        assert_eq!(
            crate::training_attention::take_training_attention_counts(),
            (0, 1)
        );
        let composable = crate::attention::sdpa_planned_bhsd(
            &q32,
            &k32,
            &v32,
            0.4,
            None,
            |s| candle_nn::ops::softmax(s, D::Minus1),
            plan,
        )
        .map_err(candle_core::Error::wrap)?;
        gradients(&selected, &composable, &c32, &[&q32, &k32, &v32], 2e-5)?;
        let grad = selected.mul(&c32)?.sum_all()?.backward()?;
        assert!(grad.get(&q32).unwrap().track_op());
        assert!(grad
            .get(&q32)
            .unwrap()
            .sorted_nodes()
            .iter()
            .any(|n| n.id() == k32.id()));
        Ok(())
    }
}

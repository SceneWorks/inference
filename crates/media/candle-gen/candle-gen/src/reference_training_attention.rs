//! Dense standard-softmax Reference attention with one first-order autograd owner.
//!
//! Forward keeps the original dense arithmetic and public dtype. Only Q/K/V and
//! the output survive in the outer graph. Backward recomputes transient probabilities
//! per query chunk; it never retains a BHSS tape across transformer layers. This is
//! not the streaming candidate and does not change block checkpointing policy.
//! Arbitrary caller softmax closures must continue through the composable API.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Storage, Tensor, D};
use gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use gen_core::CancelFlag;
use std::sync::{Arc, Mutex};

// A caller's budget is an upper limit, not a minimum tile size. Bound transient
// dense scores in both passes even when the caller asks for unbounded attention.
// This does not retain scores, change checkpointing, or select the streaming kernel.
const REFERENCE_SCORE_TILE_ELEMENTS: u64 = 32 * 1024 * 1024;

fn reference_plan(plan: AttentionPlan<'_>) -> AttentionPlan<'_> {
    AttentionPlan {
        budget: AttentionBudget::from_score_elements(
            plan.budget
                .max_score_elements()
                .min(REFERENCE_SCORE_TILE_ELEMENTS),
            plan.budget.eval_per_chunk(),
        ),
        cancel: plan.cancel,
    }
}

fn eligible(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> bool {
    let (Ok((b, h, sq, d)), Ok((kb, kh, sk, kd))) = (q.dims4(), k.dims4()) else {
        return false;
    };
    (q.device().is_cpu() || q.device().is_cuda())
        && q.device().same_device(k.device())
        && q.device().same_device(v.device())
        && matches!(q.dtype(), DType::F32 | DType::BF16 | DType::F16)
        && q.dtype() == k.dtype()
        && q.dtype() == v.dtype()
        && k.dims() == v.dims()
        && b > 0
        && h > 0
        && sq > 0
        && sk > 0
        && d > 0
        && b == kb
        && h == kh
        && d == kd
        && scale.is_finite()
        && (scale as f32).is_finite()
        && b.checked_mul(h).and_then(|n| n.checked_mul(sk)).is_some()
}

pub(crate) fn try_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    plan: AttentionPlan<'_>,
) -> crate::Result<Option<Tensor>> {
    if plan.is_cancelled() {
        return Err(crate::CandleError::Canceled);
    }
    if mask.is_some() || !eligible(q, k, v, scale) {
        return Ok(None);
    }
    // Same per-query standard-softmax arithmetic/dtype. Internal query tiles
    // respect (and can be smaller than) the caller's scratch upper limit.
    let plan = reference_plan(plan);
    let out = crate::attention::sdpa_planned_bhsd(
        &q.detach(),
        &k.detach(),
        &v.detach(),
        scale,
        None,
        |s| candle_nn::ops::softmax(s, D::Minus1),
        plan,
    )?
    .contiguous()?;
    if !q.track_op() && !k.track_op() && !v.track_op() {
        return Ok(Some(out));
    }
    let owner = Reference::new(out, scale, plan)?;
    Ok(Some(q.apply_op3(k, v, owner)?))
}

struct Reference {
    // A one-shot transfer into CustomOp3::fwd. Empty after apply_op3, so the
    // retained owner contains no tensors/storage (in particular no score matrix).
    output: Arc<Mutex<Option<Storage>>>,
    shape: Shape,
    scale: f64,
    budget: AttentionBudget,
    cancel: Option<CancelFlag>,
}

impl Reference {
    fn new(out: Tensor, scale: f64, plan: AttentionPlan<'_>) -> candle_core::Result<Self> {
        if out.track_op() {
            candle_core::bail!("reference attention must not retain an inner autograd owner");
        }
        let (storage, layout) = out.storage_and_layout();
        Ok(Self {
            output: Arc::new(Mutex::new(Some(storage.try_clone(layout)?))),
            shape: out.shape().clone(),
            scale,
            budget: reference_plan(plan).budget,
            cancel: plan.cancel.cloned(),
        })
    }

    fn take_output(&self) -> candle_core::Result<Storage> {
        self.output
            .lock()
            .map_err(|_| {
                candle_core::Error::Msg("reference attention output lock poisoned".into())
            })?
            .take()
            .ok_or_else(|| {
                candle_core::Error::Msg("reference attention forward already consumed".into())
            })
    }

    fn check_cancel(&self) -> candle_core::Result<()> {
        if self.cancel.as_ref().is_some_and(CancelFlag::is_cancelled) {
            return Err(candle_core::Error::Msg(
                "reference attention backward cancelled".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::training_attention::{
        take_training_attention_counts, with_training_attention_mode, TrainingAttentionMode,
    };
    use candle_core::{Device, Var};

    fn legacy(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        budget: usize,
    ) -> candle_core::Result<Tensor> {
        crate::attention::sdpa_budgeted_bhsd(
            q,
            k,
            v,
            scale,
            None,
            |s| candle_nn::ops::softmax(s, D::Minus1),
            budget,
        )
    }

    // Raw tensor/SDPA arithmetic only. Always ignored so no local or normal CI
    // execution can turn compile evidence into claimed numerical evidence.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA numerical proof; explicit manual raw arithmetic workflow"]
    fn dense_reference_forward_backward_matches_composable() -> crate::Result<()> {
        let bounded = reference_plan(AttentionPlan::UNBOUNDED);
        assert_eq!(
            bounded.budget.max_score_elements(),
            REFERENCE_SCORE_TILE_ELEMENTS
        );
        // Production Krea geometry must split even the originally unbounded
        // request; smaller caller budgets still determine the finer partition.
        assert!(bounded.budget.query_block_rows(48 * 4096, 4096) < 4096);
        let fine = reference_plan(AttentionPlan::budgeted(
            AttentionBudget::from_score_elements(24, true),
        ));
        assert_eq!(fine.budget.max_score_elements(), 24);
        assert!(fine.budget.eval_per_chunk());
        let dev = Device::new_cuda(0)?;
        let values = |n: usize, offset: usize| {
            (0..n)
                .map(|i| (((i + offset) * 17 % 97) as f32 - 48.) / 64.)
                .collect::<Vec<_>>()
        };
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            // GQA uses the existing caller expansion. Its backward must sum
            // repeated KV heads into the original Vars, including offset views.
            for (b, h, hk, sq, sk, d, budget) in [
                (1, 2, 2, 7, 11, 32, usize::MAX),
                (2, 4, 1, 5, 3, 128, 24),
                (1, 1, 1, 3, 9, 64, 0),
            ] {
                let make = |heads, seq, offset| -> candle_core::Result<Var> {
                    Var::from_tensor(
                        &Tensor::from_vec(
                            values(b * heads * seq * d, offset),
                            (b, heads, seq, d),
                            &dev,
                        )?
                        .to_dtype(dtype)?,
                    )
                };
                let (q, k, v) = (make(h, sq, 0)?, make(hk, sk, 1)?, make(hk, sk, 2)?);
                let qview = (q.as_tensor() * 0.75)?
                    .transpose(2, 3)?
                    .contiguous()?
                    .transpose(2, 3)?;
                let kview = Tensor::cat(&[k.as_tensor(), k.as_tensor()], 2)?.narrow(2, sk, sk)?;
                let kr = crate::attention::repeat_kv(&kview, h / hk)?;
                let vr = crate::attention::repeat_kv(&v, h / hk)?;
                let scale = 1. / (d as f64).sqrt();
                let plan =
                    AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(budget));
                for _ in 0..3 {
                    let reference = legacy(&qview, &kr, &vr, scale, budget)?;
                    take_training_attention_counts();
                    let actual =
                        with_training_attention_mode(TrainingAttentionMode::Reference, || {
                            crate::fused_attention::training_sdpa_standard_bhsd(
                                &qview, &kr, &vr, scale, None, plan,
                            )
                        })?;
                    assert_eq!(take_training_attention_counts(), (0, 1));
                    assert_eq!(actual.dtype(), dtype);
                    let cotangent =
                        Tensor::from_vec(values(b * h * sq * d, 3), (b, h, sq, d), &dev)?;
                    let ga = actual
                        .to_dtype(DType::F32)?
                        .mul(&cotangent)?
                        .sum_all()?
                        .backward()?;
                    let gr = reference
                        .to_dtype(DType::F32)?
                        .mul(&cotangent)?
                        .sum_all()?
                        .backward()?;
                    let tolerance = match dtype {
                        DType::F32 => 2e-5,
                        DType::F16 => 0.003,
                        _ => 0.025,
                    };
                    for (label, a, r) in [
                        ("output", &actual, &reference),
                        ("dq", ga.get(&q).unwrap(), gr.get(&q).unwrap()),
                        ("dk", ga.get(&k).unwrap(), gr.get(&k).unwrap()),
                        ("dv", ga.get(&v).unwrap(), gr.get(&v).unwrap()),
                    ] {
                        assert_eq!(a.dtype(), r.dtype());
                        let af = a.to_dtype(DType::F32)?;
                        assert!(
                            af.abs()?.sum_all()?.to_scalar::<f32>()? > 0.,
                            "{label}: vacuous gradient/output"
                        );
                        let delta = (af - r.to_dtype(DType::F32)?)?.abs()?;
                        assert!(delta.sum_all()?.to_scalar::<f32>()?.is_finite());
                        let max = delta.max_all()?.to_scalar::<f32>()?;
                        assert!(
                            max < tolerance,
                            "{label} {dtype:?} budget={budget}: {max} >= {tolerance}"
                        );
                    }
                }
                // Transfer consumed by the real custom-op call: nothing
                // quadratic (or any storage at all) remains in the owner slot.
                let detached = legacy(&qview.detach(), &kr.detach(), &vr.detach(), scale, budget)?;
                let owner = Reference::new(detached, scale, plan)?;
                let slot = owner.output.clone();
                let _out = qview.apply_op3(&kr, &vr, owner)?;
                assert!(crate::lock_recover(&slot).is_none());
                assert!(
                    Reference::new(legacy(&qview, &kr, &vr, scale, budget)?, scale, plan).is_err()
                );

                let mask = Tensor::zeros((sq, sk), dtype, &dev)?;
                assert!(try_forward(&qview, &kr, &vr, scale, Some(&mask), plan)?.is_none());
                let masked =
                    with_training_attention_mode(TrainingAttentionMode::Reference, || {
                        crate::fused_attention::training_sdpa_standard_bhsd(
                            &qview,
                            &kr,
                            &vr,
                            scale,
                            Some(&mask),
                            plan,
                        )
                    })?;
                let expected = legacy(&qview, &kr, &vr, scale, budget)?;
                assert_eq!(
                    (masked.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?,
                    0.
                );
                assert!(try_forward(&qview, &kr, &vr, f64::INFINITY, None, plan)?.is_none());
                assert!(try_forward(&qview.flatten_all()?, &kr, &vr, scale, None, plan)?.is_none());
                assert!(
                    try_forward(&qview.to_dtype(DType::F64)?, &kr, &vr, scale, None, plan)?
                        .is_none()
                );
                // The generic closure API must retain its actual custom math;
                // it is deliberately not routed through the standard owner.
                let custom =
                    with_training_attention_mode(TrainingAttentionMode::Reference, || {
                        crate::fused_attention::training_sdpa_budgeted_bhsd(
                            &qview,
                            &kr,
                            &vr,
                            scale,
                            None,
                            |scores| scores.zeros_like(),
                            budget,
                        )
                    })?;
                assert_eq!(
                    custom
                        .to_dtype(DType::F32)?
                        .abs()?
                        .sum_all()?
                        .to_scalar::<f32>()?,
                    0.
                );
                let cancel = CancelFlag::new();
                let cancellable =
                    try_forward(&qview, &kr, &vr, scale, None, plan.with_cancel(&cancel))?.unwrap();
                cancel.cancel();
                assert!(cancellable
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .backward()
                    .is_err());
                let cancel = CancelFlag::new();
                cancel.cancel();
                assert!(matches!(
                    crate::fused_attention::training_sdpa_standard_bhsd(
                        &qview,
                        &kr,
                        &vr,
                        scale,
                        None,
                        plan.with_cancel(&cancel)
                    ),
                    Err(crate::CandleError::Canceled)
                ));
            }
        }
        Ok(())
    }
}

impl CustomOp3 for Reference {
    fn name(&self) -> &'static str {
        "dense-reference-training-attention"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        match self.take_output()? {
            Storage::Cpu(storage) => Ok((storage, self.shape.clone())),
            _ => candle_core::bail!("reference attention output device mismatch"),
        }
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
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        match self.take_output()? {
            Storage::Cuda(storage) => Ok((storage, self.shape.clone())),
            _ => candle_core::bail!("reference attention output device mismatch"),
        }
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        self.check_cancel()?;
        let (q, k, v, grad) = (
            q.detach().contiguous()?,
            k.detach().contiguous()?,
            v.detach().contiguous()?,
            grad.detach().contiguous()?,
        );
        let (b, h, sq, _) = q.dims4()?;
        let sk = k.dim(2)?;
        let rows = self.budget.query_block_rows((b * h * sk) as u64, sq as u64) as usize;
        let kt = k.transpose(2, 3)?.contiguous()?;
        let vt = v.transpose(2, 3)?.contiguous()?;
        let mut dq = Vec::new();
        let mut dk: Option<Tensor> = None;
        let mut dv: Option<Tensor> = None;
        for start in (0..sq).step_by(rows.max(1)) {
            self.check_cancel()?;
            let len = rows.max(1).min(sq - start);
            let qc = q.narrow(2, start, len)?.contiguous()?;
            let gc = grad.narrow(2, start, len)?.contiguous()?;
            let p = candle_nn::ops::softmax(&(qc.matmul(&kt)? * self.scale)?, D::Minus1)?;
            let vc = p.transpose(2, 3)?.contiguous()?.matmul(&gc)?;
            dv = Some(match dv {
                Some(old) => (old + vc)?,
                None => vc,
            });
            // Stable softmax Jacobian-vector product. The reduction is f32;
            // half inputs cast back at the public QK-gradient boundary, while
            // native f32 inputs and gradients stay f32 throughout.
            let dp = gc.matmul(&vt)?.to_dtype(DType::F32)?;
            let pf = p.to_dtype(DType::F32)?;
            let dot = (&dp * &pf)?.sum_keepdim(D::Minus1)?;
            let ds = (pf * dp.broadcast_sub(&dot)?)?.to_dtype(q.dtype())?;
            drop(dp);
            drop(p);
            let ds = (ds * self.scale)?;
            dq.push(ds.matmul(&k)?);
            let kc = ds.transpose(2, 3)?.contiguous()?.matmul(&qc)?;
            dk = Some(match dk {
                Some(old) => (old + kc)?,
                None => kc,
            });
        }
        Ok((Some(Tensor::cat(&dq, 2)?), dk, dv))
    }
}

//! Experimental, first-order differentiable streaming CUDA SDPA (sc-26856).
//!
//! The composable caller remains the default. This SIMT candidate avoids the
//! quadratic scores/backward buffers, but is not a tensor-core FlashAttention
//! port and makes no speed claim. Select it only within a measured proof scope.
//! Q/K/V and accumulations are f32 internally; the public output/gradients keep
//! the input dtype. No f32 numerical island is downcast. Higher-order gradients
//! are unsupported, as in the existing training checkpoint paths.

use candle_core::{DType, Tensor};
use std::cell::Cell;

/// Explicit candidate selection, scoped to the synchronous training thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrainingAttentionMode {
    /// Unchanged composable reference, including the caller's softmax semantics.
    #[default]
    Reference,
    /// Native linear-memory forward/backward; requires numerical and timing proof.
    StreamingCandidate,
}

thread_local! {
    static MODE: Cell<TrainingAttentionMode> = const { Cell::new(TrainingAttentionMode::Reference) };
    static COUNTS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Run a synchronous scope with an explicit selection; restores it even on panic.
pub fn with_training_attention_mode<T>(mode: TrainingAttentionMode, f: impl FnOnce() -> T) -> T {
    struct Restore(TrainingAttentionMode);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODE.set(self.0);
        }
    }
    let _restore = Restore(MODE.replace(mode));
    f()
}

/// Thread-local `(candidate, reference)` forward call counts since the last read.
pub fn take_training_attention_counts() -> (u64, u64) {
    COUNTS.replace((0, 0))
}

/// Pure eligibility, usable by source/CPU tests without allocating a CUDA tensor.
/// All address products (including the packed gradient buffer) must fit i32.
pub fn streaming_eligible(
    cuda: bool,
    same_device: bool,
    dtype: DType,
    same_dtype: bool,
    q: &[usize],
    k: &[usize],
    v: &[usize],
    masked: bool,
    scale: f64,
) -> bool {
    let (&[b, h, sq, d], &[kb, kh, sk, kd]) = (q, k) else {
        return false;
    };
    if !cuda
        || !same_device
        || !same_dtype
        || masked
        || !scale.is_finite()
        || !(scale as f32).is_finite()
        || !matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        || b == 0
        || h == 0
        || kh == 0
        || sq == 0
        || sk == 0
        || b != kb
        || k != v
        || h % kh != 0
        || d != kd
        || d == 0
        || d > 256
        || d % 32 != 0
    {
        return false;
    }
    let product = |dims: &[usize]| dims.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    let qr = product(&[b, h, sq]);
    let kr = product(&[b, kh, sk]);
    let packed = qr
        .zip(kr)
        .and_then(|(q, k)| k.checked_mul(2)?.checked_add(q)?.checked_mul(d));
    let saved = qr.and_then(|q| q.checked_mul(d + 1));
    [packed, saved]
        .iter()
        .all(|n| n.is_some_and(|n| n <= i32::MAX as usize))
}

pub(crate) fn try_streaming(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    masked: bool,
) -> candle_core::Result<Option<Tensor>> {
    let eligible = MODE.get() == TrainingAttentionMode::StreamingCandidate
        && streaming_eligible(
            cfg!(feature = "cuda") && q.device().is_cuda(),
            q.device().same_device(k.device()) && q.device().same_device(v.device()),
            q.dtype(),
            q.dtype() == k.dtype() && q.dtype() == v.dtype(),
            q.dims(),
            k.dims(),
            v.dims(),
            masked,
            scale,
        );
    #[cfg(feature = "cuda")]
    if eligible {
        // Compile refusal is a supported fallback. Launch/allocation errors are
        // propagated: retrying after a failed CUDA launch is not a safe fallback.
        let dev = q.device().as_cuda_device()?;
        if SOURCE.compiled(dev).is_ok() {
            let output = cuda::forward(q, k, v, scale as f32)?;
            COUNTS.set((COUNTS.get().0 + 1, COUNTS.get().1));
            return Ok(Some(output));
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = eligible;
    COUNTS.set((COUNTS.get().0, COUNTS.get().1 + 1));
    Ok(None)
}

/// Shared compile-once kernel descriptor; useful for compile-only validation.
pub const SOURCE: candle_quant_kernels::nvrtc::KernelSource =
    candle_quant_kernels::nvrtc::KernelSource {
        name: "candle_gen_training_attention_v1",
        src: include_str!("training_attention.cu"),
        cc_floor: (7, 0),
    };

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::{
        backend::BackendStorage, CpuStorage, CudaStorage, CustomOp3, Layout, Shape, Storage,
    };

    struct Forward(f32);
    impl CustomOp3 for Forward {
        fn name(&self) -> &'static str {
            "streaming-training-attention"
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
            candle_core::bail!("CUDA only")
        }
        fn cuda_fwd(
            &self,
            q: &CudaStorage,
            ql: &Layout,
            k: &CudaStorage,
            kl: &Layout,
            v: &CudaStorage,
            vl: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, h, sq, d) = ql.shape().dims4()?;
            let (_, hk, sk, _) = kl.shape().dims4()?;
            let dev = q.device();
            let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
            let func = module
                .function("attention_forward")
                .map_err(candle_core::Error::wrap)?;
            let q = q.as_cuda_slice::<f32>()?.slice(ql.start_offset()..);
            let k = k.as_cuda_slice::<f32>()?.slice(kl.start_offset()..);
            let v = v.as_cuda_slice::<f32>()?.slice(vl.start_offset()..);
            let mut dst = unsafe { dev.alloc::<f32>(b * h * sq * (d + 1))? };
            let dims = [b * h * sq, h, hk, sq, sk, d].map(|n| n as i32);
            let stream = dev.cuda_stream();
            let mut launch = stream.launch_builder(&func);
            launch.arg(&q).arg(&k).arg(&v).arg(&mut dst);
            for dim in &dims {
                launch.arg(dim);
            }
            launch.arg(&self.0);
            unsafe { launch.launch(config(b * h * sq)) }.map_err(candle_core::Error::wrap)?;
            Ok((
                CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                Shape::from((b, h, sq, d + 1)),
            ))
        }
        fn bwd(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            out: &Tensor,
            grad: &Tensor,
        ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
            let packed = q.apply_op3_no_bwd(
                k,
                v,
                &Backward {
                    scale: self.0,
                    out: out.detach(),
                    grad: grad.contiguous()?.detach(),
                },
            )?;
            let nq = q.elem_count();
            let nk = k.elem_count();
            Ok((
                Some(packed.narrow(0, 0, nq)?.reshape(q.shape())?),
                Some(packed.narrow(0, nq, nk)?.reshape(k.shape())?),
                Some(packed.narrow(0, nq + nk, nk)?.reshape(v.shape())?),
            ))
        }
    }
    struct Backward {
        scale: f32,
        out: Tensor,
        grad: Tensor,
    }
    impl CustomOp3 for Backward {
        fn name(&self) -> &'static str {
            "streaming-training-attention-backward"
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
            candle_core::bail!("CUDA only")
        }
        fn cuda_fwd(
            &self,
            q: &CudaStorage,
            ql: &Layout,
            k: &CudaStorage,
            kl: &Layout,
            v: &CudaStorage,
            vl: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, h, sq, d) = ql.shape().dims4()?;
            let (_, hk, sk, _) = kl.shape().dims4()?;
            let dev = q.device();
            let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
            let q = q.as_cuda_slice::<f32>()?.slice(ql.start_offset()..);
            let k = k.as_cuda_slice::<f32>()?.slice(kl.start_offset()..);
            let v = v.as_cuda_slice::<f32>()?.slice(vl.start_offset()..);
            let (out, ol) = self.out.storage_and_layout();
            let (grad, gl) = self.grad.storage_and_layout();
            let (Storage::Cuda(out), Storage::Cuda(grad)) = (&*out, &*grad) else {
                candle_core::bail!("CUDA only")
            };
            let out = out.as_cuda_slice::<f32>()?.slice(ol.start_offset()..);
            let grad = grad.as_cuda_slice::<f32>()?.slice(gl.start_offset()..);
            let total = b * (h * sq + 2 * hk * sk) * d;
            let mut dst = unsafe { dev.alloc::<f32>(total)? };
            let stream = dev.cuda_stream();
            for (name, rows) in [("attention_dq", b * h * sq), ("attention_dkv", b * hk * sk)] {
                let func = module.function(name).map_err(candle_core::Error::wrap)?;
                let dims = [rows, h, hk, sq, sk, d].map(|n| n as i32);
                let mut launch = stream.launch_builder(&func);
                launch
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&out)
                    .arg(&grad)
                    .arg(&mut dst);
                for dim in &dims {
                    launch.arg(dim);
                }
                launch.arg(&self.scale);
                unsafe { launch.launch(config(rows)) }.map_err(candle_core::Error::wrap)?;
            }
            Ok((
                CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                Shape::from(total),
            ))
        }
    }
    fn config(rows: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (rows.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    pub(super) fn forward(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> candle_core::Result<Tensor> {
        let qf = q.to_dtype(DType::F32)?.contiguous()?;
        let kf = k.to_dtype(DType::F32)?.contiguous()?;
        let vf = v.to_dtype(DType::F32)?.contiguous()?;
        qf.apply_op3(&kf, &vf, Forward(scale))?
            .narrow(3, 0, q.dim(3)?)?
            .contiguous()?
            .to_dtype(q.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eligibility_guards_gqa_mask_dtype_and_index_range() {
        let eligible = |q: &[usize], k: &[usize], mask| {
            streaming_eligible(true, true, DType::F32, true, q, k, k, mask, 0.125)
        };
        assert!(eligible(&[2, 8, 7, 128], &[2, 2, 11, 128], false));
        assert!(!eligible(&[2, 8, 7, 128], &[2, 3, 11, 128], false));
        assert!(!eligible(&[2, 8, 7, 128], &[2, 2, 11, 128], true));
        assert!(!eligible(&[2, 8, 7, 72], &[2, 2, 11, 72], false));
        assert!(!eligible(
            &[usize::MAX, 8, 7, 128],
            &[usize::MAX, 2, 11, 128],
            false
        ));
        assert!(!streaming_eligible(
            true,
            true,
            DType::F64,
            true,
            &[1, 1, 1, 32],
            &[1, 1, 1, 32],
            &[1, 1, 1, 32],
            false,
            1.
        ));
    }
    #[test]
    fn mode_scope_restores_after_panic() {
        let _ = std::panic::catch_unwind(|| {
            with_training_attention_mode(TrainingAttentionMode::StreamingCandidate, || {
                panic!("scope")
            })
        });
        assert_eq!(MODE.get(), TrainingAttentionMode::Reference);
    }

    // Raw SDPA math, no learned modules. Explicit CUDA lane only; compilation
    // and CPU eligibility coverage do not claim GPU numerical evidence.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA numerical proof; explicit model-proof workflow"]
    fn streaming_forward_backward_matches_composable() -> candle_core::Result<()> {
        use candle_core::{Device, Var, D};
        let dev = Device::new_cuda(0)?;
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for (b, h, hk, sq, sk, d) in [
                (1, 2, 2, 7, 11, 32),
                (2, 4, 1, 5, 3, 128),
                (1, 1, 1, 3, 9, 256),
            ] {
                let values = |n, offset| {
                    (0..n)
                        .map(|i| (((i + offset) * 17 % 97) as f32 - 48.) / 64.)
                        .collect::<Vec<_>>()
                };
                let q = Var::from_tensor(
                    &Tensor::from_vec(values(b * h * sq * d, 0), (b, h, sq, d), &dev)?
                        .to_dtype(dtype)?,
                )?;
                let k = Var::from_tensor(
                    &Tensor::from_vec(values(b * hk * sk * d, 1), (b, hk, sk, d), &dev)?
                        .to_dtype(dtype)?,
                )?;
                let v = Var::from_tensor(
                    &Tensor::from_vec(values(b * hk * sk * d, 2), (b, hk, sk, d), &dev)?
                        .to_dtype(dtype)?,
                )?;
                let scale = 1. / (d as f64).sqrt();
                let reference = crate::attention::sdpa_budgeted_bhsd(
                    &q,
                    &crate::attention::repeat_kv(&k, h / hk)?,
                    &crate::attention::repeat_kv(&v, h / hk)?,
                    scale,
                    None,
                    |s| candle_nn::ops::softmax(s, D::Minus1),
                    usize::MAX,
                )?;
                let actual = cuda::forward(&q, &k, &v, scale as f32)?;
                let upstream = Tensor::from_vec(values(b * h * sq * d, 3), (b, h, sq, d), &dev)?;
                let ga = actual
                    .to_dtype(DType::F32)?
                    .mul(&upstream)?
                    .sum_all()?
                    .backward()?;
                let gr = reference
                    .to_dtype(DType::F32)?
                    .mul(&upstream)?
                    .sum_all()?
                    .backward()?;
                let tol = match dtype {
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
                    let err = (a.to_dtype(DType::F32)? - r.to_dtype(DType::F32)?)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    assert!(
                        err.is_finite() && err < tol,
                        "{label} {dtype:?} b{b} h{h}/{hk} sq{sq} sk{sk} d{d}: {err} >= {tol}"
                    );
                }
            }
        }
        Ok(())
    }
}

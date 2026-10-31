//! Frozen same-input projection packing (sc-26853) and shared LoRA arithmetic (sc-26854).

use candle_core::{DType, Tensor, TensorId, D};
use candle_nn::{Linear, Module};

use super::lora::LoraLinear;
use crate::quant::AdaptLinear;

/// Keep an immutable quantized base packed on the autograd tape. Ordinary
/// dequantize + matmul retains every decoded weight for the input gradient,
/// expanding an entire QLoRA trunk to f32 before backward can release anything.
/// Decode just this projection again when its input cotangent is requested.
pub(crate) fn frozen_packed_forward(
    x: &Tensor,
    weight: &std::sync::Arc<candle_core::quantized::QTensor>,
    bias: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let composable = |input: &Tensor| {
        let w = weight.dequantize(input.device())?.to_dtype(input.dtype())?;
        let bias = bias.map(|b| b.to_dtype(input.dtype())).transpose()?;
        Linear::new(w, bias).forward(input)
    };
    if !x.track_op()
        || candle_core::backprop::grad_do_not_detach()
        || !super::norm::supported(x)
        || bias.is_some_and(Tensor::track_op)
    {
        return composable(x);
    }
    // Identical forward arithmetic, with no inner graph retaining the decoded
    // weight. Output owns only its detached allocation, not this temporary graph.
    let out = composable(&x.detach())?;
    super::profile::cohort_reuse("ltx:bounded-packed-frozen-input-vjp");
    x.apply_op1(FrozenPackedInput {
        output: super::norm::Output::new(out)?,
        weight: weight.clone(),
        probe_bias: super::profile::cohort_claim_probe("ltx:packed-frozen-vjp-reference-parity")
            .then(|| bias.cloned()),
    })
}

struct FrozenPackedInput {
    output: super::norm::Output,
    weight: std::sync::Arc<candle_core::quantized::QTensor>,
    // Only the first excluded warmup's selected projection retains this handle.
    probe_bias: Option<Option<Tensor>>,
}

impl candle_core::CustomOp1 for FrozenPackedInput {
    fn name(&self) -> &'static str {
        "frozen-packed-input-vjp"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        self.output.cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        self.output.cuda()
    }
    fn bwd(
        &self,
        x: &Tensor,
        result: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let w = self.weight.dequantize(x.device())?.to_dtype(x.dtype())?;
        let grad = grad.detach();
        // Match Linear's contiguous 3D/4D flattening and its broadcast fallback.
        let dx = if x.is_contiguous() && matches!(x.rank(), 3 | 4) {
            let rows = grad.elem_count() / grad.dim(D::Minus1)?;
            grad.reshape((rows, grad.dim(D::Minus1)?))?
                .matmul(&w)?
                .reshape(x.shape())?
        } else {
            grad.broadcast_matmul(&w)?
        };
        // One real projection in the first excluded cohort warmup compares this
        // VJP with the old composable backward. No reference optimizer update or
        // whole-trunk dense graph is constructed, and steady steps do no probes.
        if let Some(bias) = &self.probe_bias {
            let input = candle_core::Var::from_tensor(&x.detach())?;
            let bias = bias.as_ref().map(|b| b.to_dtype(x.dtype())).transpose()?;
            let y = Linear::new(w, bias).forward(input.as_tensor())?;
            let forward_error = (result.to_dtype(DType::F32)? - y.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            if forward_error != 0.0 {
                candle_core::bail!(
                    "packed frozen forward reference mismatch: max_abs={forward_error}"
                );
            }
            let reference = (y * &grad)?.sum_all()?.backward()?;
            let expected = reference.get(input.as_tensor()).ok_or_else(|| {
                candle_core::Error::Msg("packed input VJP reference gradient missing".into())
            })?;
            let error = (&dx.to_dtype(DType::F32)? - &expected.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let scale = expected
                .to_dtype(DType::F32)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            if !error.is_finite() || !scale.is_finite() || error > 1e-6 * scale.max(1.0) {
                candle_core::bail!(
                    "packed input VJP reference mismatch: max_abs={error}, reference_scale={scale}"
                );
            }
            super::profile::cohort_reuse("ltx:packed-frozen-vjp-reference-parity");
            eprintln!("COHORT_PACKED_INPUT_VJP_PARITY forward_max_abs={forward_error} input_vjp_max_abs={error} reference_scale={scale} frozen_bias={}", self.probe_bias.as_ref().is_some_and(Option::is_some));
        }
        Ok(Some(dx))
    }
}

/// Apply a two-dimensional factor without materializing one copy per leading batch.
pub(crate) fn apply_factor(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    if super::profile::reference_math() || w.rank() != 2 || x.rank() <= 2 {
        return x.broadcast_matmul(w);
    }
    let dims = x.dims();
    let mut output = dims[..dims.len() - 1].to_vec();
    let rows: usize = output.iter().product();
    output.push(w.dim(1)?);
    x.reshape((rows, dims[dims.len() - 1]))?
        .matmul(w)?
        .reshape(output)
}

/// Scale the rank-sized intermediate only in f32. Half precision keeps the reference
/// output-scale rounding/underflow behavior; live factors are always supplied by the caller.
pub(crate) fn scaled_lora(
    x: &Tensor,
    down: &Tensor,
    up: &Tensor,
    scale: f64,
) -> candle_core::Result<Tensor> {
    let rank = apply_factor(x, down)?;
    if super::profile::reference_math() {
        return apply_factor(&rank, up)? * scale;
    }
    if scale == 1.0 {
        // The common alpha == rank convention needs no scaling kernel at any dtype.
        apply_factor(&rank, up)
    } else if x.dtype() == DType::F32 && down.dim(1)? < up.dim(1)? && scale.is_finite() {
        apply_factor(&(rank * scale)?, up)
    } else {
        apply_factor(&rank, up)? * scale
    }
}

/// Run-owned frozen dense weights concatenated once along their actual output widths.
/// Preparing replaces each projection's base with a storage-sharing slice of this allocation,
/// releasing its former base references. External model snapshots/clones can still retain the
/// original allocations: prepare the job-owned model before cloning and include such snapshots
/// in admission accounting. No adapter names, factors, or export metadata move into this object.
/// Replacing/quantizing a base invalidates the pack; forwards detect that instead of using stale
/// weights. Dropping the pack is safe: the individual bases keep their slices alive.
#[derive(Clone, Debug)]
pub struct FrozenProjectionPack {
    linear: Linear,
    widths: Vec<usize>,
    base_ids: Vec<(TensorId, Option<TensorId>)>,
    in_features: usize,
}

impl FrozenProjectionPack {
    /// Prepare compatible frozen dense LoRA projections. `None` preserves the original fallback
    /// for packed bases, live full-training weights, or incompatible dtype/device/input geometry.
    /// Call once during model preparation, never inside the training step or checkpoint replay.
    pub fn prepare_lora(layers: &mut [&mut LoraLinear]) -> candle_core::Result<Option<Self>> {
        let bases: Option<Vec<_>> = layers.iter().map(|l| l.dense_base().cloned()).collect();
        let Some(bases) = bases else { return Ok(None) };
        let Some((pack, replacements)) = Self::prepare(&bases)? else {
            return Ok(None);
        };
        for (layer, base) in layers.iter_mut().zip(replacements) {
            layer.replace_dense_base(base);
        }
        Ok(Some(pack))
    }

    /// AdaptLinear equivalent, retaining its existing independent residual routing. Packed,
    /// ConvRot, NVFP4, W4A8, external, and full-training bases are left untouched.
    pub fn prepare_adapt(layers: &mut [&mut AdaptLinear]) -> candle_core::Result<Option<Self>> {
        let bases: Option<Vec<_>> = layers.iter().map(|l| l.dense_base().cloned()).collect();
        let Some(bases) = bases else { return Ok(None) };
        let Some((pack, replacements)) = Self::prepare(&bases)? else {
            return Ok(None);
        };
        for (layer, base) in layers.iter_mut().zip(replacements) {
            layer.replace_dense_base(base);
        }
        Ok(Some(pack))
    }

    fn prepare(bases: &[Linear]) -> candle_core::Result<Option<(Self, Vec<Linear>)>> {
        if bases.len() < 2 {
            return Ok(None);
        }
        let first = bases[0].weight();
        let (_, in_features) = first.dims2()?;
        let dtype = first.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Ok(None);
        }
        let mut widths = Vec::with_capacity(bases.len());
        for base in bases {
            let w = base.weight();
            let (out, input) = w.dims2()?;
            if w.track_op()
                || input != in_features
                || out == 0
                || input == 0
                || w.dtype() != dtype
                || !first.device().same_device(w.device())
            {
                return Ok(None);
            }
            if let Some(b) = base.bias() {
                if b.track_op()
                    || b.dims() != [out]
                    || b.dtype() != dtype
                    || !first.device().same_device(b.device())
                {
                    return Ok(None);
                }
            }
            widths.push(out);
        }
        // Packing temporarily overlaps the old resident bases. A failed optional
        // pack keeps the separate projections; never charge all model weights again.
        let output_rows = widths.iter().sum::<usize>() as u64;
        let weight_bytes = output_rows
            .saturating_mul(in_features as u64)
            .saturating_mul(dtype.size_in_bytes() as u64);
        let bias_bytes = if bases.iter().any(|b| b.bias().is_some()) {
            // New concatenated bias plus temporary zero-fill slots for absent biases.
            let zeros = bases
                .iter()
                .zip(&widths)
                .filter(|(b, _)| b.bias().is_none())
                .map(|(_, width)| *width as u64)
                .sum::<u64>();
            output_rows
                .saturating_add(zeros)
                .saturating_mul(dtype.size_in_bytes() as u64)
        } else {
            0
        };
        if weight_bytes.saturating_add(bias_bytes)
            > super::flow_match::device_training_budget_bytes(
                first.device(),
                "projection preparation",
            )
        {
            return Ok(None);
        }
        let weight = Tensor::cat(&bases.iter().map(Linear::weight).collect::<Vec<_>>(), 0)?;
        let bias = if bases.iter().any(|b| b.bias().is_some()) {
            let biases = bases
                .iter()
                .zip(&widths)
                .map(|(base, &width)| {
                    base.bias()
                        .cloned()
                        .map(Ok)
                        .unwrap_or_else(|| Tensor::zeros(width, dtype, first.device()))
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            Some(Tensor::cat(&biases, 0)?)
        } else {
            None
        };
        let mut offset = 0;
        let mut replacements = Vec::with_capacity(bases.len());
        for (base, &width) in bases.iter().zip(&widths) {
            // Preserve None biases on originally unbiased projections for exact fallback behavior.
            let b = if base.bias().is_some() {
                Some(bias.as_ref().unwrap().narrow(0, offset, width)?)
            } else {
                None
            };
            replacements.push(Linear::new(weight.narrow(0, offset, width)?, b));
            offset += width;
        }
        let base_ids = replacements.iter().map(Self::identity).collect();
        Ok(Some((
            Self {
                linear: Linear::new(weight, bias),
                widths,
                base_ids,
                in_features,
            },
            replacements,
        )))
    }

    fn identity(base: &Linear) -> (TensorId, Option<TensorId>) {
        (base.weight().id(), base.bias().map(Tensor::id))
    }

    /// Actual output widths in caller order; supports GQA and per-head gates without assumptions.
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// Resident packed base bytes, shared by the source layers rather than duplicated there.
    /// Includes zero bias slots for unbiased projections when any peer has a bias.
    pub fn resident_bytes(&self) -> usize {
        let w = self.linear.weight();
        w.elem_count() * w.dtype().size_in_bytes()
            + self
                .linear
                .bias()
                .map_or(0, |b| b.elem_count() * b.dtype().size_in_bytes())
    }

    fn forward_bases(
        &self,
        x: &Tensor,
        bases: &[Option<&Linear>],
    ) -> candle_core::Result<Vec<Tensor>> {
        if bases.len() != self.base_ids.len()
            || bases
                .iter()
                .zip(&self.base_ids)
                .any(|(base, expected)| base.map(Self::identity).as_ref() != Some(expected))
        {
            candle_core::bail!("frozen projection pack invalidated: base identity/order changed");
        }
        let w = self.linear.weight();
        if x.rank() < 2
            || x.dim(D::Minus1)? != self.in_features
            || x.dtype() != w.dtype()
            || !x.device().same_device(w.device())
        {
            candle_core::bail!("frozen projection pack input shape/dtype/device mismatch");
        }
        let y = __model_proof_output!(self.linear.forward(x))?;
        let mut offset = 0;
        self.widths
            .iter()
            .map(|&width| {
                let part = y.narrow(D::Minus1, offset, width)?;
                offset += width;
                Ok(part)
            })
            .collect()
    }

    /// One base GEMM followed by independent, current LoRA/LoKr/additive residuals.
    pub fn forward_lora(
        &self,
        x: &Tensor,
        layers: &[&LoraLinear],
    ) -> candle_core::Result<Vec<Tensor>> {
        let ys = self.forward_bases(
            x,
            &layers.iter().map(|l| l.dense_base()).collect::<Vec<_>>(),
        )?;
        layers
            .iter()
            .zip(ys)
            .map(|(layer, y)| layer.apply_residuals(x, y))
            .collect()
    }

    /// One base GEMM followed by AdaptLinear's unchanged residual stack.
    pub fn forward_adapt(
        &self,
        x: &Tensor,
        layers: &[&AdaptLinear],
    ) -> candle_core::Result<Vec<Tensor>> {
        let ys = self.forward_bases(
            x,
            &layers.iter().map(|l| l.dense_base()).collect::<Vec<_>>(),
        )?;
        layers
            .iter()
            .zip(ys)
            .map(|(layer, y)| layer.apply_residuals(x, y))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn layer(width: usize, bias: bool) -> LoraLinear {
        let w = Tensor::arange(0f32, (width * 4) as f32, &Device::Cpu)
            .unwrap()
            .reshape((width, 4))
            .unwrap()
            .affine(0.01, -0.03)
            .unwrap();
        let b = bias.then(|| Tensor::ones(width, DType::F32, &Device::Cpu).unwrap());
        LoraLinear::from_linear(Linear::new(w, b), 4, width, format!("projection_{width}"))
    }

    #[test]
    fn preparation_tracks_widths_storage_and_refuses_live_base() {
        let mut q = layer(6, false);
        let mut k = layer(2, true);
        let pack = FrozenProjectionPack::prepare_lora(&mut [&mut q, &mut k])
            .unwrap()
            .unwrap();
        assert_eq!(pack.widths(), &[6, 2]);
        assert_eq!(pack.resident_bytes(), (8 * 4 + 8) * 4);
        assert_eq!(q.path(), "projection_6");
        assert_eq!(k.path(), "projection_2");
        assert!(q.dense_base().unwrap().bias().is_none());
        let variable = Var::zeros((6, 4), DType::F32, &Device::Cpu).unwrap();
        q.replace_dense_base(Linear::new(variable.as_tensor().clone(), None));
        assert!(FrozenProjectionPack::prepare_lora(&mut [&mut q, &mut k])
            .unwrap()
            .is_none());
    }

    #[test]
    fn rank_scale_matches_output_scale_and_live_factor_gradients() -> candle_core::Result<()> {
        let x = Tensor::arange(0f32, 24f32, &Device::Cpu)?
            .reshape((2, 3, 4))?
            .affine(0.03, -0.2)?;
        let down = Var::from_tensor(
            &Tensor::arange(0f32, 8f32, &Device::Cpu)?
                .reshape((4, 2))?
                .affine(0.1, -0.2)?,
        )?;
        let up = Var::from_tensor(
            &Tensor::arange(0f32, 12f32, &Device::Cpu)?
                .reshape((2, 6))?
                .affine(0.1, -0.3)?,
        )?;
        for scale in [0.125, -1.7, 2.0] {
            let reference = (apply_factor(&apply_factor(&x, &down)?, &up)? * scale)?;
            let actual = scaled_lora(&x, &down, &up, scale)?;
            assert!(
                (&actual - &reference)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?
                    < 2e-6
            );
            let ga = actual.sqr()?.sum_all()?.backward()?;
            let gr = reference.sqr()?.sum_all()?.backward()?;
            for factor in [&down, &up] {
                assert!(
                    (ga.get(factor).unwrap() - gr.get(factor).unwrap())?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?
                        < 2e-5
                );
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "raw half-precision matmul requires explicitly admitted CUDA numerical proof lane"]
    fn half_precision_keeps_output_scale_rounding() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::BF16, DType::F16] {
            let x = Tensor::arange(0f32, 24f32, &device)?
                .reshape((2, 3, 4))?
                .affine(0.03, -0.2)?
                .to_dtype(dtype)?;
            let down = Tensor::arange(0f32, 8f32, &device)?
                .reshape((4, 2))?
                .affine(0.1, -0.2)?
                .to_dtype(dtype)?;
            let up = Tensor::arange(0f32, 12f32, &device)?
                .reshape((2, 6))?
                .affine(0.1, -0.3)?
                .to_dtype(dtype)?;
            for scale in [1.0, 0.125, -1.7] {
                let reference = (apply_factor(&apply_factor(&x, &down)?, &up)? * scale)?;
                let actual = scaled_lora(&x, &down, &up, scale)?;
                assert_eq!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    reference
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn packed_base_preserves_independent_adapter_gradients_and_updates() -> candle_core::Result<()>
    {
        let mut q = layer(6, false);
        let mut k = layer(2, true);
        let down =
            Var::from_tensor(&Tensor::ones((2, 4), DType::F32, &Device::Cpu)?.affine(0.1, 0.0)?)?;
        let up =
            Var::from_tensor(&Tensor::ones((6, 2), DType::F32, &Device::Cpu)?.affine(0.2, 0.0)?)?;
        q.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), 0.5);
        let x = Tensor::arange(0f32, 24f32, &Device::Cpu)?
            .reshape((2, 3, 4))?
            .affine(0.03, -0.2)?;
        let pack = FrozenProjectionPack::prepare_lora(&mut [&mut q, &mut k])?.unwrap();
        let separate = __model_proof_output!(q.forward(&x))?;
        let separate_k = __model_proof_output!(k.forward(&x))?;
        let fused = pack.forward_lora(&x, &[&q, &k])?;
        for (a, b) in [(&fused[0], &separate), (&fused[1], &separate_k)] {
            assert!((a - b)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        }
        let ga = fused[0].sqr()?.sum_all()?.backward()?;
        let gr = separate.sqr()?.sum_all()?.backward()?;
        for factor in [&down, &up] {
            assert!(
                (ga.get(factor).unwrap() - gr.get(factor).unwrap())?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?
                    < 1e-5
            );
        }
        up.set(&Tensor::ones((6, 2), DType::F32, &Device::Cpu)?)?;
        let updated = pack.forward_lora(&x, &[&q, &k])?;
        assert!(
            (&updated[0] - &fused[0])?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                > 0.01
        );
        assert!(pack.forward_lora(&x, &[&k, &q]).is_err());
        q.clear();
        q.push_additive_lora(down.t()?, up.t()?, 0.5);
        let additive = pack.forward_lora(&x, &[&q, &k])?;
        let separate = __model_proof_output!(q.forward(&x))?;
        assert!(
            (&additive[0] - &separate)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                < 1e-6
        );
        q.replace_dense_base(layer(6, false).dense_base().unwrap().clone());
        assert!(pack.forward_lora(&x, &[&q, &k]).is_err());
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED train::projection::tests::packed_base_preserves_independent_adapter_gradients_and_updates");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn bf16_projection_close(
        name: &str,
        actual: &Tensor,
        separate: &Tensor,
        f32_reference: &Tensor,
    ) -> candle_core::Result<bool> {
        assert_eq!(actual.dims(), separate.dims(), "{name}: shape");
        assert_eq!(actual.dims(), f32_reference.dims(), "{name}: f32 shape");
        let host = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        let (actual, separate, reference) = (host(actual)?, host(separate)?, host(f32_reference)?);
        let (mut max_direct, mut max_plain, mut max_actual, mut peak) = (0f64, 0f64, 0f64, 0f64);
        let (
            mut direct2,
            mut plain2,
            mut actual2,
            mut reference2,
            mut separate2,
            mut norm2,
            mut dot,
        ) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for ((&a, &s), &r) in actual.iter().zip(&separate).zip(&reference) {
            assert!(
                a.is_finite() && s.is_finite() && r.is_finite(),
                "{name}: non-finite value"
            );
            let (a, s, r) = (f64::from(a), f64::from(s), f64::from(r));
            max_direct = max_direct.max((a - s).abs());
            max_plain = max_plain.max((s - r).abs());
            max_actual = max_actual.max((a - r).abs());
            peak = peak.max(r.abs());
            direct2 += (a - s).powi(2);
            plain2 += (s - r).powi(2);
            actual2 += (a - r).powi(2);
            reference2 += r * r;
            separate2 += s * s;
            norm2 += a * a;
            dot += a * s;
        }
        assert!(
            reference2 > 0.0 && separate2 > 0.0 && norm2 > 0.0,
            "{name}: vacuous zero reference"
        );
        // Same scale-sensitive BF16 policy as Iris tests/common::assert_ac_bf16_noise_only:
        // error vs matched f32 <= 2 * separate-BF16 error + 1e-3 of the f32 norm,
        // cosine >= .999. Apply the policy to values AND every adapter factor gradient.
        // The direct (fused,separate) bounds follow by the triangle inequality (2+1),
        // with max-absolute checks as well as L2 so local errors cannot hide in cosine.
        let (noise_factor, noise_floor, min_cosine) = (2.0, 1e-3, 0.999);
        let plain_error = (plain2 / reference2).sqrt();
        let actual_error = (actual2 / reference2).sqrt();
        let direct_error = (direct2 / reference2).sqrt();
        let cosine = dot / (norm2 * separate2).sqrt();
        let accepted = actual_error <= noise_factor * plain_error + noise_floor
            && max_actual <= noise_factor * max_plain + noise_floor * peak
            && direct_error <= (noise_factor + 1.0) * plain_error + noise_floor
            && max_direct <= (noise_factor + 1.0) * max_plain + noise_floor * peak
            && cosine >= min_cosine;
        eprintln!(
            "{name}: max_direct={max_direct:.6e} max_f32_plain={max_plain:.6e} max_f32_actual={max_actual:.6e} peak={peak:.6e} relative_direct={direct_error:.6e} relative_f32_plain={plain_error:.6e} relative_f32_actual={actual_error:.6e} cosine={cosine:.9} accepted={accepted}"
        );
        Ok(accepted)
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn cuda_bf16_packed_qkvg_preserves_outputs_and_every_adapter_gradient(
    ) -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        // One fixed small GQA-like configuration: Q has four times the KV width,
        // while the per-head gate is narrower still. K/V have distinct values/targets.
        let widths = [128, 32, 32, 4];
        let paths = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate"];
        let scales = [0.5, -1.7, 1.0, 0.125];
        let values = |rows: usize, cols: usize, salt: usize| {
            Tensor::from_vec(
                (0..rows * cols)
                    .map(|i| (((i * 37 + salt * 13) % 101) as f32 - 50.0) / 256.0)
                    .collect::<Vec<_>>(),
                (rows, cols),
                &device,
            )
        };
        let mut layers = Vec::new();
        let mut factors = Vec::new();
        for (i, &width) in widths.iter().enumerate() {
            let weight = values(width, 64, i + 1)?.to_dtype(DType::BF16)?;
            let bias = if i % 2 == 0 {
                Some(
                    values(1, width, i + 11)?
                        .reshape(width)?
                        .to_dtype(DType::BF16)?,
                )
            } else {
                None
            };
            let mut layer =
                LoraLinear::from_linear(Linear::new(weight, bias), 64, width, paths[i].into());
            let down = Var::from_tensor(&values(4, 64, i + 21)?)?;
            let up = Var::from_tensor(&values(width, 4, i + 31)?)?;
            layer.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), scales[i]);
            factors.push((down, up));
            layers.push(layer);
        }
        let x = values(10, 64, 41)?
            .reshape((2, 5, 64))?
            .to_dtype(DType::BF16)?;
        // Keep the original independent allocations as the reference. The prepared
        // layers must actually use the pack's replacement slices, not this snapshot.
        let originals = layers.clone();
        // Start the f32 oracle from the SAME quantized base/input values, not a
        // higher precision pre-quantization fixture. The same live f32 master Vars
        // feed both graphs. Only activation/factor GEMM rounding differs; no model,
        // input, adapter scale or objective changes can inflate the BF16 noise budget.
        let f32_x = x.to_dtype(DType::F32)?;
        let f32_layers = originals
            .iter()
            .map(|layer| {
                let mut reference = layer.clone();
                let base = layer.dense_base().unwrap();
                reference.replace_dense_base(Linear::new(
                    base.weight().to_dtype(DType::F32)?,
                    base.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
                ));
                Ok(reference)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let f32_outputs = f32_layers
            .iter()
            .map(|layer| __model_proof_output!(layer.forward(&f32_x)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let pack = FrozenProjectionPack::prepare_lora(&mut layers.iter_mut().collect::<Vec<_>>())?
            .expect("fixed frozen BF16 CUDA projections must prepare");
        assert_eq!(pack.widths(), widths);
        assert_eq!(
            pack.resident_bytes(),
            widths.iter().sum::<usize>() * (64 + 1) * 2
        );
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(layer.path(), paths[i]);
            assert_ne!(
                layer.dense_base().unwrap().weight().id(),
                originals[i].dense_base().unwrap().weight().id()
            );
            assert_eq!(layer.dense_base().unwrap().bias().is_some(), i % 2 == 0);
        }
        let separate = originals
            .iter()
            .map(|layer| __model_proof_output!(layer.forward(&x)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let refs = layers.iter().collect::<Vec<_>>();
        let fused = __model_proof_output!(pack.forward_lora(&x, &refs), tensors)?;
        for (i, (actual, reference)) in fused.iter().zip(&separate).enumerate() {
            assert_eq!(actual.dtype(), DType::BF16);
            assert!(actual.device().same_device(&device));
            assert_eq!(actual.dims(), &[2, 5, widths[i]]);
            assert!(bf16_projection_close(
                paths[i],
                actual,
                reference,
                &f32_outputs[i]
            )?);
        }
        // An output-dependent f32 loss checks gradients as well as reachability:
        // wrong frozen widths/order/biases must affect the cotangents, too.
        let loss = |ys: &[Tensor]| {
            Tensor::cat(ys, D::Minus1)?
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()
        };
        let actual_grads = loss(&fused)?.backward()?;
        let reference_grads = loss(&separate)?.backward()?;
        let f32_grads = loss(&f32_outputs)?.backward()?;
        for (i, (down, up)) in factors.iter().enumerate() {
            for (factor_name, factor) in [("down", down), ("up", up)] {
                let actual = actual_grads
                    .get(factor)
                    .expect("every live factor gradient");
                let reference = reference_grads
                    .get(factor)
                    .expect("every reference factor gradient");
                assert_eq!(actual.dtype(), DType::F32);
                let oracle = f32_grads
                    .get(factor)
                    .expect("every f32 reference factor gradient");
                assert!(bf16_projection_close(
                    &format!("{}.{factor_name}", paths[i]),
                    actual,
                    reference,
                    oracle
                )?);
                assert!(!bf16_projection_close(
                    "negative: gradient scale",
                    &(reference * 1.5)?,
                    reference,
                    oracle
                )?);
            }
        }
        // Negative controls prove that the numerical comparator notices a missing
        // residual and an adapter-scale error, even if directions remain similar.
        let mut missing = originals[0].clone();
        missing.clear();
        let missing_y = __model_proof_output!(missing.forward(&x))?;
        assert!(!bf16_projection_close(
            "negative: missing residual",
            &missing_y,
            &separate[0],
            &f32_outputs[0]
        )?);
        let residual = (&separate[0].to_dtype(DType::F32)? - &missing_y.to_dtype(DType::F32)?)?;
        let wrong_scale = (&missing_y.to_dtype(DType::F32)? + (residual * 1.5)?)?;
        assert!(!bf16_projection_close(
            "negative: residual scale",
            &wrong_scale,
            &separate[0],
            &f32_outputs[0]
        )?);
        // Reuse the same pack after changing every master factor: casts/residuals
        // must read live Vars rather than retaining a prepared adapter snapshot.
        for (down, up) in &factors {
            down.set(&down.affine(1.25, 0.015625)?)?;
            up.set(&up.affine(0.75, -0.03125)?)?;
        }
        let updated = __model_proof_output!(pack.forward_lora(&x, &refs), tensors)?;
        for (i, layer) in originals.iter().enumerate() {
            let reference = __model_proof_output!(layer.forward(&x))?;
            let f32_reference = __model_proof_output!(f32_layers[i].forward(&f32_x))?;
            assert!(bf16_projection_close(
                paths[i],
                &updated[i],
                &reference,
                &f32_reference
            )?);
            assert!(
                (&updated[i].to_dtype(DType::F32)? - &fused[i].to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?
                    > 0.0,
                "{} must observe its live factor update",
                paths[i]
            );
        }
        assert!(pack
            .forward_lora(&x, &[&layers[0], &layers[2], &layers[1], &layers[3]])
            .is_err());
        assert!(pack.forward_lora(&x.to_dtype(DType::F32)?, &refs).is_err());
        // Full-training weights refuse preparation; their ordinary path stays usable.
        let variable = Var::from_tensor(originals[0].dense_base().unwrap().weight())?;
        layers[0].replace_dense_base(Linear::new(
            variable.as_tensor().clone(),
            originals[0].dense_base().unwrap().bias().cloned(),
        ));
        assert!(pack
            .forward_lora(&x, &layers.iter().collect::<Vec<_>>())
            .is_err());
        assert!(
            FrozenProjectionPack::prepare_lora(&mut layers.iter_mut().collect::<Vec<_>>())?
                .is_none()
        );
        let fallback = __model_proof_output!(layers[0].forward(&x))?;
        assert_eq!(fallback.dims(), &[2, 5, widths[0]]);
        let fallback_reference = __model_proof_output!(originals[0].forward(&x))?;
        let fallback_f32_reference = __model_proof_output!(f32_layers[0].forward(&f32_x))?;
        assert!(bf16_projection_close(
            "refused live-base fallback",
            &fallback,
            &fallback_reference,
            &fallback_f32_reference
        )?);
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED train::projection::tests::cuda_bf16_packed_qkvg_preserves_outputs_and_every_adapter_gradient");
        Ok(())
    }
}

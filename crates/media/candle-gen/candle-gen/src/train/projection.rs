//! Frozen same-input projection packing (sc-26853) and shared LoRA arithmetic (sc-26854).

use candle_core::{DType, Tensor, TensorId, D};
use candle_nn::{Linear, Module};

use super::lora::LoraLinear;
use crate::quant::AdaptLinear;

/// Apply a two-dimensional factor without materializing one copy per leading batch.
pub(crate) fn apply_factor(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    if w.rank() != 2 || x.rank() <= 2 {
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
    if x.dtype() == DType::F32 && down.dim(1)? < up.dim(1)? && scale.is_finite() {
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
}

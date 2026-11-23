//! A native checkpoint projection with ordered, reconstructed adapter deltas.
use candle_core::{DType, Device, Result, Tensor};
use candle_layers::adapt::{AdaptLinear, Adapter};
use candle_nn::{linear::NativeLinearWeight, Linear};
use std::sync::Arc;

type Native = Arc<dyn NativeLinearWeight>;

#[derive(Clone, Copy)]
pub(super) enum DeltaView {
    Matrix,
    Transposed,
    Flattened,
}

pub(super) struct AdapterProjection {
    base: Native,
    residuals: AdaptLinear,
}

impl AdapterProjection {
    pub(super) fn bind(
        base: Option<Native>,
        deltas: Option<&[Tensor]>,
        view: DeltaView,
    ) -> Result<Option<Native>> {
        let (Some(base), Some(deltas)) = (&base, deltas) else {
            return Ok(base);
        };
        let matrices = deltas.iter().map(|delta| match view {
            DeltaView::Matrix => Ok(delta.clone()),
            DeltaView::Transposed => delta.t()?.contiguous(),
            DeltaView::Flattened => delta.reshape(base.dims()),
        });
        Ok(Some(Self::from_matrices(base.clone(), matrices)?))
    }

    fn from_matrices(
        base: Native,
        matrices: impl Iterator<Item = Result<Tensor>>,
    ) -> Result<Native> {
        let (rows, cols) = base.dims();
        let mut residuals = AdaptLinear::for_external_base(cols, rows, DType::F32, base.device())
            .map_err(candle_core::Error::wrap)?;
        for matrix in matrices {
            residuals
                .push_matrix_delta(matrix?.to_device(base.device())?)
                .map_err(candle_core::Error::wrap)?;
        }
        Ok(Arc::new(Self { base, residuals }))
    }

    fn matrices(&self) -> impl Iterator<Item = Result<Tensor>> + '_ {
        self.residuals.adapters.iter().map(|adapter| match adapter {
            Adapter::MatrixDelta(delta) => Ok(delta.weight.clone()),
            _ => candle_core::bail!("checkpoint overlay expected a reconstructed matrix delta"),
        })
    }

    fn fold_coefficients(&self, base: &Tensor) -> Result<Tensor> {
        let mut weight = base.to_dtype(DType::F32)?;
        for delta in self.matrices() {
            weight = (weight + delta?.to_device(base.device())?)?;
        }
        weight.to_dtype(base.dtype())
    }
}

impl NativeLinearWeight for AdapterProjection {
    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let base = self.base.forward(x, bias)?;
        self.residuals
            .apply_residuals(&x.to_dtype(DType::F32)?, base.to_dtype(DType::F32)?)?
            .to_dtype(base.dtype())
    }
    fn dense_weight(&self) -> Result<Tensor> {
        self.fold_coefficients(&self.base.dense_weight()?)
    }
    fn retained_coefficients(&self, tensor: &Tensor) -> Result<()> {
        self.base.retained_coefficients(tensor)
    }
    fn checkpoint_bias_dtype(&self) -> Option<DType> {
        self.base.checkpoint_bias_dtype()
    }
    fn compute_dtype(&self) -> DType {
        self.base.compute_dtype()
    }
    fn relocated(&self, device: &Device) -> Result<Linear> {
        let moved = self.base.relocated(device)?;
        if let Some(base) = moved.native_weight() {
            Ok(Linear::from_native(
                Self::from_matrices(base, self.matrices())?,
                moved.bias().cloned(),
            ))
        } else {
            Ok(Linear::new(
                self.fold_coefficients(moved.try_weight()?)?,
                moved.bias().cloned(),
            ))
        }
    }
    fn format_id(&self) -> &'static str {
        self.base.format_id()
    }
    fn resident_bytes(&self) -> u64 {
        self.base.resident_bytes() + self.residuals.frozen_adapter_bytes() as u64
    }
    fn dims(&self) -> (usize, usize) {
        self.base.dims()
    }
    fn device(&self) -> &Device {
        self.base.device()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::train::merge::{merge_diff_patch_file, AdapterFile, CheckpointDeltas, MergeReport};
    use candle_core::{Module, Shape};
    use candle_nn::var_builder::{SimpleBackend, TensorInfo};
    use candle_nn::{Init, VarBuilder};
    use std::collections::HashMap;

    struct FixtureSource(HashMap<String, Tensor>);
    struct RoundedInput;

    impl NativeLinearWeight for RoundedInput {
        fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
            let base = x.round()?;
            match bias {
                Some(bias) => base.broadcast_add(bias),
                None => Ok(base),
            }
        }
        fn dense_weight(&self) -> Result<Tensor> {
            panic!("adapted native matrix requested dense coefficients")
        }
        fn checkpoint_bias_dtype(&self) -> Option<DType> {
            Some(DType::F32)
        }
        fn dims(&self) -> (usize, usize) {
            (2, 2)
        }
        fn format_id(&self) -> &'static str {
            "rounded-input-fixture"
        }
        fn resident_bytes(&self) -> u64 {
            4
        }
        fn device(&self) -> &Device {
            &Device::Cpu
        }
    }

    impl SimpleBackend for FixtureSource {
        fn tensor_info(&self, key: &str) -> Result<Option<TensorInfo>> {
            if key == "native.weight" {
                Ok(Some(TensorInfo {
                    shape: (32, 32).into(),
                    padded_shape: Some(gen_core::mxfp8_padded_shape),
                    source_packed: true,
                }))
            } else {
                SimpleBackend::tensor_info(&self.0, key)
            }
        }
        fn native_linear_weight(
            &self,
            key: &str,
            _: Option<(usize, usize)>,
            _: DType,
            _: &Device,
        ) -> Result<Option<Native>> {
            Ok((key == "native.weight").then(|| Arc::new(RoundedInput) as Native))
        }
        fn get_unchecked(&self, key: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
            assert_ne!(
                key, "native.weight",
                "native coefficient read during adapter preparation"
            );
            SimpleBackend::get_unchecked(&self.0, key, dtype, dev)
        }
        fn get(
            &self,
            shape: Shape,
            key: &str,
            init: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            assert_ne!(
                key, "native.weight",
                "native coefficient read during construction"
            );
            SimpleBackend::get(&self.0, shape, key, init, dtype, dev)
        }
        fn contains_tensor(&self, key: &str) -> bool {
            key == "native.weight" || self.0.contains_key(key)
        }
    }

    fn prepared() -> crate::Result<VarBuilder<'static>> {
        let dev = &Device::Cpu;
        let source = FixtureSource(HashMap::from([
            (
                "native.bias".into(),
                Tensor::new(&[(1.0f32 + 1.0 / 1024.0), 1.5], dev)?,
            ),
            (
                "dense.weight".into(),
                Tensor::full(16_777_216f32, (2, 2), dev)?,
            ),
        ]));
        let mut targets = CheckpointDeltas::new(VarBuilder::from_backend(
            Box::new(source),
            DType::BF16,
            dev.clone(),
        ));
        let files = [
            vec![
                (
                    "native.diff",
                    Tensor::new(&[[1.0f32, 0.0], [0.0, 2.0]], dev)?,
                ),
                ("native.diff_b", Tensor::new(&[0.25f32, 0.5], dev)?),
                ("dense.diff", Tensor::ones((2, 2), DType::F32, dev)?),
            ],
            vec![
                (
                    "native.diff",
                    Tensor::new(&[[0.0f32, 1.0], [1.0, 0.0]], dev)?,
                ),
                ("dense.diff", Tensor::full(-16_777_216f32, (2, 2), dev)?),
            ],
            vec![
                ("native.diff", Tensor::zeros((33, 2), DType::F32, dev)?),
                ("native.diff_b", Tensor::full(99.0f32, 2, dev)?),
            ],
        ];
        let mut report = MergeReport::default();
        for tensors in files {
            let file = AdapterFile {
                tensors: tensors
                    .into_iter()
                    .map(|(k, t)| (k.to_owned(), t))
                    .collect(),
                meta: HashMap::new(),
            };
            merge_diff_patch_file(&mut targets, &file, 1.0, str::to_owned, &mut report)?;
        }
        assert_eq!((report.merged, report.skipped_keys), (5, 2));
        Ok(targets.into_var_builder())
    }

    /// Constructor/weight inspection only. The native fixture refuses every coefficient read.
    #[test]
    fn native_delta_constructor_preserves_source_bias_and_ordered_dense_folds() -> crate::Result<()>
    {
        let vb = prepared()?;
        let layer = candle_nn::linear(2, 2, vb.pp("native"))?;
        assert_eq!(layer.native_format_id(), Some("rounded-input-fixture"));
        assert_eq!(layer.native_resident_bytes(), Some(4 + 2 * 4 * 4));
        assert_eq!(
            layer.bias().unwrap().to_vec1::<f32>()?,
            [1.25 + 1.0 / 1024.0, 2.0]
        );
        // ((2^24 + 1) - 2^24) rounds to zero; adding the deltas first would leave one.
        assert_eq!(
            vb.get((2, 2), "dense.weight")?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            [0.0; 4]
        );
        Ok(())
    }

    /// Called only from the registered, ignored fold/residual model proof.
    pub(crate) fn assert_original_input_residual() -> crate::Result<()> {
        let vb = prepared()?;
        let layer = candle_nn::linear(2, 2, vb.pp("native"))?;
        let input = Tensor::new(&[[0.75f32, -0.25]], &Device::Cpu)?;
        // Native round(x) + adapted bias, plus both matrix deltas applied to the original input.
        assert_eq!(
            __model_proof_output!(layer.forward(&input))?.to_vec2::<f32>()?,
            [[2.75 + 1.0 / 1024.0, 2.25]]
        );
        Ok(())
    }
}

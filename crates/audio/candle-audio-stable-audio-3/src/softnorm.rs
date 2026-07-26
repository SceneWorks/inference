//! SAME SoftNorm bottleneck.

use candle_audio::candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

pub struct SoftNorm {
    scaling_factor: Tensor,
    bias: Tensor,
    running_std: Option<Tensor>,
    noise_scaling_factor: Option<Tensor>,
    noise_regularize: bool,
}

impl SoftNorm {
    pub fn load(
        dim: usize,
        noise_augment_dim: usize,
        noise_regularize: bool,
        auto_scale: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            scaling_factor: vb.get_with_hints((1, dim, 1), "scaling_factor", Init::Const(1.0))?,
            bias: vb.get_with_hints((1, dim, 1), "bias", Init::Const(0.0))?,
            running_std: auto_scale
                .then(|| vb.get_with_hints(1, "running_std", Init::Const(1.0)))
                .transpose()?,
            // Upstream persists an empty `[1, 0, 1]` buffer when augmentation is disabled.
            // Consume it when present so standalone and embedded SAME inventories are exact.
            noise_scaling_factor: (noise_augment_dim > 0
                || vb.contains_tensor("noise_scaling_factor"))
            .then(|| {
                vb.get_with_hints(
                    (1, noise_augment_dim, 1),
                    "noise_scaling_factor",
                    Init::Const(1.0),
                )
            })
            .transpose()?,
            noise_regularize,
        })
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let x = x
            .broadcast_mul(&self.scaling_factor)?
            .broadcast_add(&self.bias)?;
        match &self.running_std {
            Some(std) => x.broadcast_div(std),
            None => Ok(x),
        }
    }

    pub fn noise_regularize(&self) -> bool {
        self.noise_regularize
    }

    /// Decode using explicit noise, making parity tests deterministic.
    ///
    /// `regularization_noise` is a unit-normal tensor shaped like `x`; it is scaled by 5e-2
    /// during training and 1e-3 during evaluation. `augment_noise` is required when the configured
    /// augmentation dimension is non-zero.
    pub fn decode_with_noise(
        &self,
        x: &Tensor,
        training: bool,
        regularization_noise: Option<&Tensor>,
        augment_noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = match &self.running_std {
            Some(std) => x.broadcast_mul(std)?,
            None => x.clone(),
        };
        if self.noise_regularize {
            let scaling = match &self.running_std {
                Some(std) => std.clone(),
                None => {
                    let mean = x.mean_keepdim(2)?;
                    let centered = x.broadcast_sub(&mean)?;
                    let length = x.dim(2)?;
                    centered
                        .sqr()?
                        .sum_keepdim(2)?
                        .affine(1.0 / (length.saturating_sub(1)) as f64, 0.0)?
                        .sqrt()?
                }
            };
            let noise = match regularization_noise {
                Some(n) => n.clone(),
                None => Tensor::randn(0f32, 1f32, x.shape(), x.device())?.to_dtype(x.dtype())?,
            };
            let scale = if training { 5e-2 } else { 1e-3 };
            x = (&x + noise.broadcast_mul(&scaling)?.affine(scale, 0.0)?)?;
        }
        if let Some(factor) = &self.noise_scaling_factor {
            let (batch, channels, length) = x.dims3()?;
            let aug_channels = factor.dim(1)?;
            if aug_channels == 0 {
                return Ok(x);
            }
            let noise = match augment_noise {
                Some(n) => n.clone(),
                None => Tensor::randn(0f32, 1f32, (batch, aug_channels, length), x.device())?
                    .to_dtype(x.dtype())?,
            };
            let augmented = noise.broadcast_mul(factor)?;
            let _ = channels;
            x = Tensor::cat(&[&x, &augmented], 1)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device};
    use candle_nn::VarMap;
    use std::collections::HashMap;

    #[test]
    fn saved_noise_proves_train_and_eval_scales() {
        let dev = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "scaling_factor".into(),
            Tensor::ones((1, 1, 1), DType::F32, &dev).unwrap(),
        );
        tensors.insert(
            "bias".into(),
            Tensor::zeros((1, 1, 1), DType::F32, &dev).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let soft = SoftNorm::load(1, 0, true, false, vb).unwrap();
        let x = Tensor::from_vec(vec![1f32, 3.], (1, 1, 2), &dev).unwrap();
        let noise = Tensor::ones((1, 1, 2), DType::F32, &dev).unwrap();
        let eval = soft
            .decode_with_noise(&x, false, Some(&noise), None)
            .unwrap();
        let train = soft
            .decode_with_noise(&x, true, Some(&noise), None)
            .unwrap();
        let eval = eval.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let train = train.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sample_std = 2f32.sqrt();
        assert!((eval[0] - (1.0 + sample_std * 1e-3)).abs() < 1e-6);
        assert!((train[0] - (1.0 + sample_std * 5e-2)).abs() < 1e-6);
    }

    #[test]
    fn varmap_initialization_matches_upstream_constants() {
        let dev = Device::Cpu;
        let vars = VarMap::new();
        let soft = SoftNorm::load(
            2,
            1,
            false,
            true,
            VarBuilder::from_varmap(&vars, DType::F32, &dev),
        )
        .unwrap();
        let x = Tensor::from_vec(vec![2f32, -3.], (1, 2, 1), &dev).unwrap();
        assert_eq!(
            soft.encode(&x).unwrap().to_vec3::<f32>().unwrap(),
            x.to_vec3::<f32>().unwrap()
        );
        let vars = vars.data().lock().unwrap();
        assert_eq!(
            vars["noise_scaling_factor"]
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0]
        );
    }
}

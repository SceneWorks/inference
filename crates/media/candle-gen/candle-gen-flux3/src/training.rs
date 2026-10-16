//! Native trainable FLUX3 DiT seam. Frozen base weights stay in the inference graph;
//! adapter `Var`s use the shared trainable residual mechanism, not pre-merged deltas.
pub mod checkpoint;

use std::collections::BTreeMap;

use candle_gen::candle_core::{DType, Tensor, Var};
use candle_gen::gen_core::flux3::training::{lora_targets, LoraTarget};
use candle_gen::{CandleError, Result};

use crate::{Flux3Dit, SampleTokens};

/// Master-f32 factors. The native graph keeps the SAME Var storage across updates.
pub struct LoraFactors {
    pub target: LoraTarget,
    pub down: Var,
    pub up: Var,
}

/// Owns ONLY the declared adapter factors. No encoder, AE, norm or base weight is trainable.
pub struct TrainingGraph {
    pub dit: Flux3Dit,
    pub factors: Vec<LoraFactors>,
    pub scale: f64,
}

impl TrainingGraph {
    /// `initial_down` must provide every target's Gaussian-initialized A tensor; B is zero.
    /// Passing explicit initialization makes RNG state a trainer-owned/resumable input.
    pub fn new(
        mut dit: Flux3Dit,
        rank: usize,
        alpha: f64,
        targets: &[String],
        mut initial_down: BTreeMap<String, Tensor>,
    ) -> Result<Self> {
        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            return Err(CandleError::Unsupported(
                "flux3 training: rank and alpha must be positive and finite".into(),
            ));
        }
        let selected = lora_targets(dit.config(), targets)
            .map_err(|e| CandleError::Unsupported(e.to_string()))?;
        let mut factors = Vec::with_capacity(selected.len());
        let scale = alpha / rank as f64;
        // Validate every input BEFORE attaching any factor.
        if initial_down.len() != selected.len() {
            return Err(CandleError::Unsupported(
                "flux3 training: initialization target set does not match the declared targets"
                    .into(),
            ));
        }
        for target in selected {
            let down = initial_down.remove(&target.native).ok_or_else(|| {
                CandleError::Unsupported(format!(
                    "flux3 training: missing initialization for {}",
                    target.native
                ))
            })?;
            if down.dims() != [rank, target.input] || down.dtype() != DType::F32 {
                return Err(CandleError::Unsupported(format!(
                    "flux3 training: {} A must be f32 [{rank},{}]",
                    target.native, target.input
                )));
            }
            let down = Var::from_tensor(&down)?;
            let up = Var::zeros((target.output, rank), DType::F32, dit.device())?;
            factors.push(LoraFactors { target, down, up });
        }
        for factor in &factors {
            dit.install_trainable_lora(
                &factor.target.native,
                factor.down.as_tensor().clone(),
                factor.up.as_tensor().clone(),
                scale,
            )?;
        }
        Ok(Self {
            dit,
            factors,
            scale,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        self.factors
            .iter()
            .flat_map(|f| [f.down.clone(), f.up.clone()])
            .collect()
    }

    /// Differentiable velocity loss. Noise/clean mixing and target are both tensor ops;
    /// host reads are left to reporting AFTER backward. Text/AE tensors are frozen inputs.
    pub fn loss(
        &self,
        samples: &[SampleTokens<'_>],
        target: &Tensor,
        weight: f64,
    ) -> Result<Tensor> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(CandleError::Unsupported(
                "flux3 training: loss weight must be nonnegative and finite".into(),
            ));
        }
        let prediction = self.dit.forward_training(samples)?.to_dtype(DType::F32)?;
        if prediction.dims() != target.dims() {
            return Err(CandleError::Unsupported(
                "flux3 training: packed velocity target shape mismatch".into(),
            ));
        }
        Ok((prediction
            .sub(&target.to_dtype(DType::F32)?)?
            .sqr()?
            .mean_all()?
            * weight)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::{rmsnorm_rope, rmsnorm_rope_training, Rope};
    use candle_gen::candle_core::{Device, Tensor, Var};

    #[test]
    fn training_rmsnorm_preserves_forward_and_nonzero_earlier_projection_gradient() {
        let device = Device::Cpu;
        let x =
            Var::from_tensor(&Tensor::new(&[[[[1f32, 2.0, 3.0, 4.0]]]], &device).unwrap()).unwrap();
        let scale = Tensor::new(&[0.7f32, 1.2, 0.9, 1.4], &device).unwrap();
        let rope = Rope {
            cos: Tensor::new(&[[0.8f32, 0.6]], &device).unwrap(),
            sin: Tensor::new(&[[0.6f32, 0.8]], &device).unwrap(),
        };
        let output = rmsnorm_rope_training(x.as_tensor(), &scale, &rope).unwrap();
        let inference = rmsnorm_rope(x.as_tensor(), &scale, &rope).unwrap();
        let delta = output
            .sub(&inference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            delta < 1e-6,
            "training-only primitive RMSNorm changed forward: {delta}"
        );
        let weights = Tensor::new(&[[[[0.3f32, -0.2, 0.4, 0.9]]]], &device).unwrap();
        let grads = output
            .mul(&weights)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let gradient = grads
            .get(x.as_tensor())
            .expect("RMSNorm must propagate into earlier adapter projections");
        let got = gradient.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(got.iter().any(|g| g.abs() > 1e-4));
        // Independent CPU PyTorch analytical fixture is compared in the integration oracle.
    }
}

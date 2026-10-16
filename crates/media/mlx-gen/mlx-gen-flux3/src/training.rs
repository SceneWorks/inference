//! Functional native FLUX3 LoRA autograd seam. No Python execution path.
//! Each differentiation re-injects independent A/B factors for every declared
//! projection inside the traced closure. The frozen DiT/AE/text weights are never parameters.
pub mod optim;

use crate::{Flux3Dit, SampleTokens};
use mlx_gen::gen_core::flux3::training::{lora_targets, LoraTarget};
use mlx_gen::train::lora::LoraParams;
use mlx_gen::{Error, Result};
use mlx_rs::error::Exception;
use mlx_rs::transforms::keyed_value_and_grad;
use mlx_rs::{Array, Dtype};

pub struct TrainingGraph {
    pub dit: Flux3Dit,
    pub targets: Vec<LoraTarget>,
    pub rank: usize,
    pub scale: f64,
}

impl TrainingGraph {
    pub fn new(dit: Flux3Dit, rank: usize, alpha: f64, requested: &[String]) -> Result<Self> {
        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            return Err(Error::Unsupported(
                "flux3 training: rank and alpha must be positive and finite".into(),
            ));
        }
        let targets =
            lora_targets(dit.config(), requested).map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(Self {
            dit,
            targets,
            rank,
            scale: alpha / rank as f64,
        })
    }

    /// Strict factor set/shape/precision admission before tracing. No ignored keys.
    pub fn validate_factors(&self, params: &LoraParams) -> Result<()> {
        if params.len() != 2 * self.targets.len() {
            return Err(Error::Unsupported(
                "flux3 training: factor key set differs from declared targets".into(),
            ));
        }
        for target in &self.targets {
            for (suffix, shape) in [
                ("lora_A.weight", [self.rank as i32, target.input as i32]),
                ("lora_B.weight", [target.output as i32, self.rank as i32]),
            ] {
                let key = format!("{}.{}", target.native, suffix);
                let array = params.get(key.as_str()).ok_or_else(|| {
                    Error::Unsupported(format!("flux3 training: missing factor {key}"))
                })?;
                if array.shape() != shape || array.dtype() != Dtype::Float32 {
                    return Err(Error::Unsupported(format!(
                        "flux3 training: {key} must be master f32 {shape:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// No materialize/eval/host round-trip inside the differentiation closure.
    pub fn loss_and_grads(
        &mut self,
        params: &LoraParams,
        samples: &[SampleTokens<'_>],
        target: &Array,
        weight: f32,
    ) -> Result<(Array, LoraParams)> {
        self.validate_factors(params)?;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Unsupported(
                "flux3 training: loss weight must be nonnegative and finite".into(),
            ));
        }
        let targets = &self.targets;
        let dit = &mut self.dit;
        let scale = self.scale;
        let loss_fn = move |p: LoraParams, _: i32| -> std::result::Result<Vec<Array>, Exception> {
            for projection in targets {
                let a = p[format!("{}.lora_A.weight", projection.native).as_str()].clone();
                let b = p[format!("{}.lora_B.weight", projection.native).as_str()].clone();
                dit.replace_training_lora(&projection.native, a, b, scale)
                    .map_err(|e| Exception::custom(e.to_string()))?;
            }
            let prediction = dit
                .forward(samples)
                .map_err(|e| Exception::custom(e.to_string()))?
                .as_dtype(Dtype::Float32)?;
            if prediction.shape() != target.shape() {
                return Err(Exception::custom(
                    "flux3 training: packed velocity target shape mismatch",
                ));
            }
            let difference = prediction.subtract(&target.as_dtype(Dtype::Float32)?)?;
            let loss = difference
                .square()?
                .mean(None)?
                .multiply(Array::from_f32(weight))?;
            Ok(vec![loss])
        };
        let (values, grads) = keyed_value_and_grad(loss_fn)(params.clone(), 0)?;
        Ok((values.into_iter().next().expect("one loss"), grads))
    }
}

//! Bias-corrected source AdamW for FLUX3. The pinned mlx-rs AdamW deliberately
//! omits bias correction; changing that shared optimizer would alter legacy families.
//! This additive family-owned optimizer follows the pinned diffusers/PyTorch recipe.
use mlx_gen::train::lora::LoraParams;
use mlx_gen::{Error, Result};
use mlx_rs::{Array, Dtype};
use std::{collections::HashMap, rc::Rc};

/// Per-factor state (including its own update index, as PyTorch skips unused parameters).
pub struct AdamMoment {
    pub first: Array,
    pub second: Array,
    pub step: u32,
}

pub struct SourceAdamW {
    pub moments: HashMap<Rc<str>, AdamMoment>,
    learning_rate: f32,
    base_learning_rate: f32,
    weight_decay: f32,
}

impl SourceAdamW {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Result<Self> {
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(Error::Unsupported(
                "flux3 training: invalid AdamW learning rate/weight decay".into(),
            ));
        }
        Ok(Self {
            moments: HashMap::new(),
            learning_rate,
            base_learning_rate: learning_rate,
            weight_decay,
        })
    }

    pub fn configuration_fingerprint(&self) -> String {
        format!(
            "adamw;lr={:08x};wd={:08x};beta1=0.9;beta2=0.999;eps=1e-8;bias_correction=true",
            self.base_learning_rate.to_bits(),
            self.weight_decay.to_bits()
        )
    }

    /// The scheduler is reconstructed from the strong configuration + update cursor.
    pub fn set_lr_scaled(&mut self, multiplier: f32) -> Result<()> {
        if !multiplier.is_finite()
            || multiplier < 0.0
            || !(self.base_learning_rate * multiplier).is_finite()
        {
            return Err(Error::Unsupported(
                "flux3 training: invalid LR schedule multiplier".into(),
            ));
        }
        self.learning_rate = self.base_learning_rate * multiplier;
        Ok(())
    }

    pub fn step(&mut self, params: &mut LoraParams, grads: &LoraParams) -> Result<()> {
        // Validate the complete update before touching any state.
        for (name, gradient) in grads {
            let parameter = params.get(name).ok_or_else(|| {
                Error::Unsupported(format!(
                    "flux3 training: gradient for undeclared factor {name}"
                ))
            })?;
            if gradient.shape() != parameter.shape()
                || gradient.dtype() != Dtype::Float32
                || parameter.dtype() != Dtype::Float32
            {
                return Err(Error::Unsupported(format!(
                    "flux3 training: AdamW shape/master precision mismatch for {name}"
                )));
            }
            if self
                .moments
                .get(name)
                .is_some_and(|state| state.step == u32::MAX)
            {
                return Err(Error::Unsupported(
                    "flux3 training: AdamW step counter overflow".into(),
                ));
            }
        }
        for (name, gradient) in grads {
            let parameter = &params[name];
            let state = match self.moments.entry(name.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let zero = mlx_rs::ops::zeros_like(parameter)?;
                    entry.insert(AdamMoment {
                        first: zero.clone(),
                        second: zero,
                        step: 0,
                    })
                }
            };
            state.step += 1;
            state.first = state
                .first
                .multiply(Array::from_f32(0.9))?
                .add(gradient.multiply(Array::from_f32(0.1))?)?;
            state.second = state
                .second
                .multiply(Array::from_f32(0.999))?
                .add(gradient.square()?.multiply(Array::from_f32(0.001))?)?;
            let corrected_first = state.first.multiply(Array::from_f32(
                (1.0 / (1.0 - 0.9f64.powf(state.step as f64))) as f32,
            ))?;
            let corrected_second = state.second.multiply(Array::from_f32(
                (1.0 / (1.0 - 0.999f64.powf(state.step as f64))) as f32,
            ))?;
            let adjusted =
                corrected_first.divide(corrected_second.sqrt()?.add(Array::from_f32(1e-8))?)?;
            let decayed = parameter.multiply(Array::from_f32(
                1.0 - self.learning_rate * self.weight_decay,
            ))?;
            let updated =
                decayed.subtract(adjusted.multiply(Array::from_f32(self.learning_rate))?)?;
            params.insert(name.clone(), updated);
        }
        Ok(())
    }
}

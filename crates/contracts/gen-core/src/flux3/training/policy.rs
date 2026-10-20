//! Source-specific optimizer-step policy. Do not substitute the generic native
//! flow driver's short-window averaging or warmup indexing for these contracts.
use crate::{Error, Result};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 training policy: {detail}"))
}

/// Apply to the LOSS before backward, as Accelerator.backward does. A short tail
/// keeps this configured divisor; pending gradients are already scaled snapshots.
pub fn accumulation_loss_scale(accumulation: u32) -> Result<f32> {
    if accumulation == 0 {
        return Err(invalid("gradient accumulation must be positive"));
    }
    Ok(1.0 / accumulation as f32)
}

/// Torch clip_grad_norm_'s coefficient, including epsilon even at norm==maximum.
pub fn source_clip_multiplier(norm: f32, maximum: f32) -> Result<f32> {
    if !norm.is_finite() || norm < 0.0 || !maximum.is_finite() || maximum < 0.0 {
        return Err(invalid(
            "gradient norm/maximum must be finite and nonnegative",
        ));
    }
    // Python scalar / Tensor uses Tensor.__rtruediv__: reciprocal THEN multiply.
    // Direct maximum/denominator differs by 1 ULP at the norm boundary.
    Ok(((1.0 / (norm + 1e-6)) * maximum).min(1.0))
}

/// Each epoch flushes a short accumulation tail. `batches_per_epoch` is the
/// prebuilt drop-last-per-bucket batch count, NOT the number of dataset images.
pub fn optimizer_update_budget(
    micro_steps: u64,
    accumulation: u32,
    batches_per_epoch: u64,
) -> Result<u64> {
    if accumulation == 0 || batches_per_epoch == 0 {
        return Err(invalid(
            "accumulation and batches per epoch must be positive",
        ));
    }
    let n = accumulation as u64;
    let epochs = micro_steps / batches_per_epoch;
    let tail = micro_steps % batches_per_epoch;
    epochs
        .checked_mul(batches_per_epoch.div_ceil(n))
        .and_then(|updates| updates.checked_add(tail.div_ceil(n)))
        .ok_or_else(|| invalid("optimizer update budget overflow"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceLrKind {
    Constant,
    ConstantWithWarmup,
    Linear,
    Cosine,
    CosineWithRestarts,
    Polynomial,
}
impl SourceLrKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "constant" => Ok(Self::Constant),
            "constant_with_warmup" => Ok(Self::ConstantWithWarmup),
            "linear" => Ok(Self::Linear),
            "cosine" => Ok(Self::Cosine),
            "cosine_with_restarts" => Ok(Self::CosineWithRestarts),
            "polynomial" => Ok(Self::Polynomial),
            _ => Err(invalid(format!("unsupported source LR scheduler `{name}`"))),
        }
    }
}

/// Exact diffusers7ecb optimization.py LambdaLR policy. `step` is the 0-based
/// OPTIMIZER update index; source warmup starts at zero, not 1/warmup.
#[derive(Clone, Copy, Debug)]
pub struct SourceLrSchedule {
    pub kind: SourceLrKind,
    pub warmup: u64,
    pub total_updates: u64,
    /// Passed only to the hard-restart scheduler by source get_scheduler.
    /// Plain cosine retains its function default HALF cycle.
    pub cycles: u32,
    pub power: f64,
    pub base_lr: f64,
}
impl SourceLrSchedule {
    pub fn validate(self) -> Result<()> {
        if self.total_updates == 0
            || !self.base_lr.is_finite()
            || self.base_lr <= 0.0
            || !self.power.is_finite()
            || self.power < 0.0
        {
            return Err(invalid("invalid LR schedule step budget/rate/power"));
        }
        if self.kind == SourceLrKind::Polynomial
            && (self.base_lr <= 1e-7 || self.warmup >= self.total_updates)
        {
            return Err(invalid("polynomial schedule requires base LR>1e-7 and warmup shorter than its update budget"));
        }
        Ok(())
    }
    pub fn multiplier(self, step: u64) -> Result<f64> {
        self.validate()?;
        if self.kind == SourceLrKind::Constant {
            return Ok(1.0);
        }
        if step < self.warmup {
            return Ok(step as f64 / self.warmup.max(1) as f64);
        }
        let remaining = self.total_updates.saturating_sub(self.warmup).max(1) as f64;
        let progress = (step - self.warmup) as f64 / remaining;
        Ok(match self.kind {
            SourceLrKind::Constant | SourceLrKind::ConstantWithWarmup => 1.0,
            SourceLrKind::Linear => self.total_updates.saturating_sub(step) as f64 / remaining,
            SourceLrKind::Cosine => {
                (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())).max(0.0)
            }
            SourceLrKind::CosineWithRestarts => {
                if progress >= 1.0 {
                    0.0
                } else {
                    (0.5 * (1.0
                        + (std::f64::consts::PI * ((self.cycles as f64 * progress) % 1.0)).cos()))
                    .max(0.0)
                }
            }
            SourceLrKind::Polynomial => {
                let end = 1e-7;
                if step > self.total_updates {
                    end / self.base_lr
                } else {
                    ((self.base_lr - end) * (1.0 - progress).powf(self.power) + end) / self.base_lr
                }
            }
        })
    }
}

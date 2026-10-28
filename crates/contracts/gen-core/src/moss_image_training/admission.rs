//! Backend-neutral arithmetic shared by both MOSS training-memory estimators (sc-27256).
//!
//! `mlx-gen-moss-image` and `candle-gen-moss-image` each own a separate admission model: MLX prices
//! one unified-memory budget, Candle prices CUDA VRAM and host RAM separately, and each sizes its
//! text-encoder state from its own KV-cache growth policy. Only the facts both models compute
//! identically live here: checked byte arithmetic, the optimizer/compute-width/token-grid readings of
//! a training config, which VAE tensors the trainer constructs, and the budget-override rule.

use crate::TrainingConfig;

/// Why a shared admission computation refused. Each engine maps it onto its own admission error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionArithmeticError {
    /// A checked byte or element computation overflowed `u64`; names the quantity.
    Overflow(&'static str),
    /// A config or shape fact the estimators cannot price.
    Invalid(String),
}

/// `Σ xs`, refusing on `u64` overflow.
pub fn checked_sum(
    xs: impl IntoIterator<Item = u64>,
    what: &'static str,
) -> Result<u64, AdmissionArithmeticError> {
    xs.into_iter().try_fold(0u64, |a, b| {
        a.checked_add(b)
            .ok_or(AdmissionArithmeticError::Overflow(what))
    })
}

/// `Π xs`, refusing on `u64` overflow.
pub fn checked_product(xs: &[u64], what: &'static str) -> Result<u64, AdmissionArithmeticError> {
    xs.iter().try_fold(1u64, |a, b| {
        a.checked_mul(*b)
            .ok_or(AdmissionArithmeticError::Overflow(what))
    })
}

/// A `usize` dimension as `u64`.
pub fn dim(n: usize) -> Result<u64, AdmissionArithmeticError> {
    u64::try_from(n).map_err(|_| AdmissionArithmeticError::Invalid("dimension exceeds u64".into()))
}

/// A strictly positive `i32` config dimension as `u64`.
pub fn positive(n: i32) -> Result<u64, AdmissionArithmeticError> {
    if n <= 0 {
        return Err(AdmissionArithmeticError::Invalid(
            "nonpositive encoder dimension".into(),
        ));
    }
    u64::try_from(n)
        .map_err(|_| AdmissionArithmeticError::Invalid("unrepresentable encoder dimension".into()))
}

/// Pure override resolution: a zero override is a real zero cap, not a request to disable guards.
pub fn resolve_budget(override_bytes: Option<u64>, device_bytes: u64) -> u64 {
    override_bytes.unwrap_or(device_bytes)
}

/// The VAE tensors the MOSS trainer constructs: the full encoder and decoder, both quant convs, and
/// the batch-norm statistics the packed latent is normalized with.
pub fn is_constructed_vae_tensor(name: &str) -> bool {
    ["encoder.", "decoder.", "quant_conv.", "post_quant_conv."]
        .iter()
        .any(|p| name.starts_with(p))
        || name == "bn.running_mean"
        || name == "bn.running_var"
}

/// Optimizer state tensors held per trainable element (the spelling is normalized: case, `-`, `_`).
pub fn optimizer_states(name: &str) -> Result<u64, AdmissionArithmeticError> {
    let n: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    match n.as_str() {
        "adam" | "adamw" | "adam8bit" | "adamw8bit" => Ok(2),
        "rose" | "roseopt" => Ok(0),
        "prodigy" | "prodigyopt" => Ok(4),
        _ => Err(AdmissionArithmeticError::Invalid(format!(
            "unsupported optimizer {name}"
        ))),
    }
}

/// Bytes per element of the training compute dtype (bf16 or f32 only).
pub fn compute_width(c: &TrainingConfig) -> Result<u64, AdmissionArithmeticError> {
    match c.train_dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(2),
        "f32" | "float32" | "fp32" => Ok(4),
        _ => Err(AdmissionArithmeticError::Invalid(
            "compute dtype must be bf16 or f32".into(),
        )),
    }
}

/// Image tokens of a square bucket edge on the 16-pixel latent-token grid.
pub fn image_tokens(edge: u64) -> Result<u64, AdmissionArithmeticError> {
    if edge == 0 || !edge.is_multiple_of(16) {
        return Err(AdmissionArithmeticError::Invalid(
            "bucket edge must be positive and divisible by 16".into(),
        ));
    }
    checked_product(&[edge / 16, edge / 16], "image tokens")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_arithmetic_refuses_overflow_by_name() {
        assert_eq!(checked_sum([1, 2, 3], "s"), Ok(6));
        assert_eq!(checked_product(&[2, 3, 4], "p"), Ok(24));
        assert_eq!(
            checked_sum([u64::MAX, 1], "s"),
            Err(AdmissionArithmeticError::Overflow("s"))
        );
        assert_eq!(
            checked_product(&[u64::MAX, 2], "p"),
            Err(AdmissionArithmeticError::Overflow("p"))
        );
        assert!(positive(0).is_err());
        assert_eq!(positive(7), Ok(7));
    }

    #[test]
    fn config_readings_match_the_trainer() {
        assert_eq!(optimizer_states("AdamW-8bit"), Ok(2));
        assert_eq!(optimizer_states("Prodigy_Opt"), Ok(4));
        assert_eq!(optimizer_states("rose"), Ok(0));
        assert!(optimizer_states("sgd").is_err());
        let mut c = TrainingConfig {
            train_dtype: "BF16".into(),
            ..Default::default()
        };
        assert_eq!(compute_width(&c), Ok(2));
        c.train_dtype = "fp16".into();
        assert!(compute_width(&c).is_err());
        assert_eq!(image_tokens(1024), Ok(4096));
        assert!(image_tokens(1000).is_err());
        assert!(is_constructed_vae_tensor("decoder.up.0.weight"));
        assert!(is_constructed_vae_tensor("bn.running_var"));
        assert!(!is_constructed_vae_tensor("bn.num_batches_tracked"));
        assert_eq!(resolve_budget(Some(0), 10), 0);
        assert_eq!(resolve_budget(None, 10), 10);
    }
}

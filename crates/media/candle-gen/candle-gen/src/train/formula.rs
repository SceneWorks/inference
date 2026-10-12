//! Deterministic **formula weights** for the latent-perceptual parity tests (epic 2123, sc-24833) —
//! the Candle twin of `mlx_gen::train::formula`.
//!
//! The parity fixture `crates/contracts/gen-core/tests/fixtures/latent_perceptual/` is produced by
//! running the upstream PyTorch losses on weights and inputs that are closed-form functions of their
//! checkpoint key and flat index (the producer script's `formula` / `input_wave`), so no checkpoint is
//! downloaded or committed. Values are computed in f64 and rounded to f32, exactly like the producer.

use candle_core::{Device, Tensor};

use crate::Result;

/// What a tensor is, which selects its amplitude (see the producer's docstring).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Conv / linear weight: `sqrt(2 / fan_in) · wave`.
    Conv,
    /// Bias: `0.1 · wave`.
    Bias,
    /// BatchNorm / GroupNorm scale: `1 + 0.2 · wave`.
    NormWeight,
    /// BatchNorm running mean: `0.1 · wave`.
    BnMean,
    /// BatchNorm running variance: `1 + 0.5 · wave²`.
    BnVar,
    /// E-LatentLPIPS lin head: `0.05 · |wave| + 0.01`.
    Lin,
}

/// 32-bit FNV-1a of the key's UTF-8 bytes.
pub fn fnv1a32(key: &str) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for b in key.bytes() {
        h = (h ^ b as u32).wrapping_mul(16_777_619);
    }
    h
}

/// The formula values of tensor `key` with `shape` in `role`, row-major, as f32.
pub fn formula_values(key: &str, shape: &[usize], role: Role) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let phase = (fnv1a32(key) % 10_007) as f64 / 1000.0;
    let fan_in: usize = shape.iter().skip(1).product();
    (0..n)
        .map(|i| {
            let w = (0.7 * i as f64 + phase).sin();
            let v = match role {
                Role::Conv => (2.0 / fan_in.max(1) as f64).sqrt() * w,
                Role::Bias | Role::BnMean => 0.1 * w,
                Role::NormWeight => 1.0 + 0.2 * w,
                Role::BnVar => 1.0 + 0.5 * w * w,
                Role::Lin => 0.05 * w.abs() + 0.01,
            };
            v as f32
        })
        .collect()
}

/// [`formula_values`] as an f32 tensor of `shape` on `device`.
pub fn formula_tensor(key: &str, shape: &[usize], role: Role, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        formula_values(key, shape, role),
        shape,
        device,
    )?)
}

/// The producer's `input_wave(n, a, f, p)`: `a · sin(f · i + p)` for `i < n`, in f64.
pub fn input_wave(n: usize, a: f64, f: f64, p: f64) -> Vec<f64> {
    (0..n).map(|i| a * (f * i as f64 + p).sin()).collect()
}

/// The parity fixture JSON, relative to `crates/` (shared by the MLX and Candle ports).
pub const PARITY_FIXTURE_REL: &str =
    "contracts/gen-core/tests/fixtures/latent_perceptual/latent_perceptual_parity.json";

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula's FNV-1a matches the standard 32-bit test vectors the producer's `fnv1a32`
    /// also hits.
    #[test]
    fn fnv_matches_the_reference_vector() {
        assert_eq!(fnv1a32(""), 2_166_136_261);
        assert_eq!(fnv1a32("a"), 0xE40C_292C);
    }
}

//! Counter-based **synthetic weights and images** for the face-loss parity tests (epic 2123,
//! sc-24831, AC3) — the Candle twin of `mlx-gen-face`'s `synth` and of the generator in
//! `crates/media/face_loss_fixtures/produce_face_loss_fixtures.py`. Every value is a pure function
//! of `(seed, key, index)`, so torch, MLX and Candle all build bit-identical tensors without any
//! downloaded checkpoint:
//!
//! `u_i = (splitmix64(k + i) >> 40) / 2²⁴` with `k = splitmix64(seed ^ fnv1a64(key))`, and a tensor
//! value is `offset + (2·u_i − 1)·half` with `(offset, half)` from [`role`].

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::train::splitmix64;
use candle_gen::Result;

/// 64-bit FNV-1a of `s`.
pub fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// `n` values in `[0, 1)` (f64) for `(seed, key)`.
pub fn uniform(seed: u64, key: &str, n: usize) -> Vec<f64> {
    let k = splitmix64(seed ^ fnv1a64(key));
    (0..n as u64)
        .map(|i| (splitmix64(k.wrapping_add(i)) >> 40) as f64 / 16_777_216.0)
        .collect()
}

/// `(offset, half-range)` of a synthetic tensor by key/shape (identical to the producer's `role`).
pub fn role(key: &str, shape: &[usize]) -> (f64, f64) {
    if key.contains("prelu") || key.ends_with(".slope") {
        return (0.25, 0.05);
    }
    if key.ends_with(".scale") {
        return (1.0, 0.1);
    }
    if key.ends_with(".shift") || key.ends_with(".bias") {
        return (0.0, 0.05);
    }
    if shape.len() >= 2 {
        let n: usize = shape.iter().product();
        let fan_in = n / shape[0];
        return (0.0, (3.0 / fan_in as f64).sqrt());
    }
    (0.0, 0.1)
}

/// The synthetic f32 values of tensor `key` with `shape` (row-major).
pub fn values(seed: u64, key: &str, shape: &[usize]) -> Vec<f32> {
    let (off, half) = role(key, shape);
    let n: usize = shape.iter().product();
    uniform(seed, key, n)
        .into_iter()
        .map(|u| (off + (2.0 * u - 1.0) * half) as f32)
        .collect()
}

/// The synthetic tensor `key` on `device`.
pub fn tensor(seed: u64, key: &str, shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        values(seed, key, shape),
        shape.to_vec(),
        device,
    )?)
}

/// A synthetic NHWC `[1, h, w, 3]` image in `[0, 1]` (the decoder-output layout).
pub fn image(seed: u64, key: &str, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let v: Vec<f32> = uniform(seed, key, h * w * 3)
        .into_iter()
        .map(|u| u as f32)
        .collect();
    Ok(Tensor::from_vec(v, (1, h, w, 3), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned values (the Python producer's generator) — the same pins as `mlx-gen-face`'s twin.
    /// Mutation: drop the `>> 40` ⇒ red.
    #[test]
    fn generator_matches_the_producer() {
        assert_eq!(fnv1a64("live"), 0xbf66_95ad_6966_058f);
        let u = uniform(0x24831C, "live", 3);
        assert_eq!(u, [0.17715787887573242, 0.5028018951416016, 0.6082891225814819]);
    }
}

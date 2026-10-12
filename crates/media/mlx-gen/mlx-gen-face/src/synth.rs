//! MLX tensors from the face-loss fixture's counter-based synthetic generator
//! ([`mlx_gen::gen_core::train::face_loss::synth`], shared with `candle-gen-face` and the Python producer),
//! so torch, MLX and Candle build bit-identical weights and images without any downloaded
//! checkpoint (epic 2123, sc-24831, AC3).

use mlx_gen::gen_core::train::face_loss::synth;
use mlx_rs::Array;

/// The synthetic tensor `key` with `shape`.
pub fn tensor(seed: u64, key: &str, shape: &[usize]) -> Array {
    let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&synth::values(seed, key, shape), &dims)
}

/// A synthetic NHWC `[1, h, w, 3]` image in `[0, 1]`.
pub fn image(seed: u64, key: &str, h: usize, w: usize) -> Array {
    Array::from_slice(&synth::image(seed, key, h, w), &[1, h as i32, w as i32, 3])
}

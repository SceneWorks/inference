//! Candle tensors from the face-loss fixture's counter-based synthetic generator
//! ([`candle_gen::gen_core::train::face_loss::synth`], shared with `mlx-gen-face` and the Python
//! producer), so torch, MLX and Candle build bit-identical weights and images without any
//! downloaded checkpoint (epic 2123, sc-24831, AC3).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::train::face_loss::synth;
use candle_gen::Result;

/// The synthetic tensor `key` with `shape` on `device`.
pub fn tensor(seed: u64, key: &str, shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        synth::values(seed, key, shape),
        shape.to_vec(),
        device,
    )?)
}

/// A synthetic NHWC `[1, h, w, 3]` image in `[0, 1]` (the decoder-output layout).
pub fn image(seed: u64, key: &str, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        synth::image(seed, key, h, w),
        (1, h, w, 3),
        device,
    )?)
}

//! Per-step previews for the Z-Image 16-channel VAE latent space.
//!
//! Z-Image denoises `[C, 1, h, w]` tensors. The provider-owned unpack seam converts that layout to
//! the shared preview projector's `[1, C, h, w]` contract. Schedule numbering, projection, readback,
//! and best-effort emission remain in `mlx_gen::preview`.

use mlx_gen::{PreviewSink, Result};
use mlx_rs::Array;

/// Ordinary-least-squares map from the native Z-Image VAE latent to RGB.
///
/// Fit on four diverse real-weight Z-Image-Turbo BF16 renders and measured on two disjoint
/// prompt/seed holdouts, all 256² with eight static-shift-3 flow-Euler steps. Targets are
/// 8×8-average-pooled native VAE decodes. Fit R² `(R,G,B) = (0.98367, 0.97883, 0.98092)`, overall
/// `0.98133`; holdout R² `(0.94679, 0.96390, 0.89464)`, overall `0.92827`. The retained comparison
/// images preserve coarse composition and palette on every fit and holdout sample.
///
/// Snapshot: `SceneWorks/z-image-turbo-mlx` revision
/// `bb2bc9893b3c49ae96c813350775f791a2e8bc80`, `bf16` tier. The fitted VAE file is
/// `vae/model.safetensors`, 167,666,968 bytes, SHA-256
/// `0fbab8b661f6ee6af81c88a6eb1501ec1f7b4b8fe4ad29803507ebe0cf863810`.
///
/// Reproduce with `tests/fit_preview_rgb.rs`. A stale fit affects decorative preview colour only;
/// the render never reads these constants.
const RGB_FACTORS: [[f32; 3]; 16] = [
    [-0.013_211_725, 0.020_633_436, 0.050_329_126],
    [0.014_224_869, 0.030_253_288, 0.048_853_34],
    [0.031_214_886, -0.026_290_553, -0.008_127_655],
    [-0.011_716_095, 0.006_138_681, 0.036_768_82],
    [0.042_083_209, 0.033_715_149, 0.009_236_064],
    [-0.005_458_121, 0.009_163_568, 0.000_726_971],
    [0.017_442_052, 0.055_714_785, 0.043_591_47],
    [-0.020_549_937, -0.023_569_854, -0.027_749_361],
    [-0.023_123_204, 0.005_715_808, 0.064_064_235],
    [0.066_185_762, 0.045_447_53, -0.031_686_028],
    [-0.010_402_147, 0.035_838_17, 0.018_642_27],
    [0.050_614_966, 0.018_175_902, 0.019_094_432],
    [0.028_492_43, 0.028_673_975, 0.036_316_507],
    [-0.072_754_92, -0.010_183_617, -0.074_263_78],
    [-0.007_323_435, -0.039_554_853, -0.007_222_673],
    [-0.061_362_23, -0.036_242_01, -0.029_276_784],
];
const RGB_BIAS: [f32; 3] = [0.502_150_24, 0.483_383_92, 0.458_297_43];

fn project(latents: &Array) -> Result<mlx_gen::Image> {
    let unpacked = crate::pipeline::unpack_latents(latents)?;
    mlx_gen::preview::project_latents(&unpacked, &RGB_FACTORS, RGB_BIAS)
}

/// Emit one native `[16, 1, h, w]` Z-Image latent for an actual outer solver step.
pub fn emit_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    latents: &Array,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || project(latents));
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn native_layout_projects_at_latent_resolution() {
        let sigmas = [1.0, 0.5, 0.0];
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let latents = Array::zeros::<f32>(&[16, 1, 3, 5]).unwrap();

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &latents);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (1, 2));
        assert_eq!((frames[0].image.width, frames[0].image.height), (5, 3));
        assert_eq!(frames[0].image.pixels.len(), 5 * 3 * 3);
    }

    #[test]
    fn malformed_latent_is_decorative_and_consumes_its_position() {
        let sigmas = [1.0, 0.5, 0.0];
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let malformed = Array::zeros::<f32>(&[15, 1, 2, 2]).unwrap();
        let valid = Array::zeros::<f32>(&[16, 1, 2, 2]).unwrap();

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &malformed);
        emit_preview(&sink, &counter, &sigmas, sigmas[0], &valid);
        emit_preview(&sink, &counter, &sigmas, sigmas[1], &valid);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    #[test]
    fn committed_fit_has_one_finite_row_per_latent_channel() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert!(RGB_FACTORS.iter().flatten().all(|value| value.is_finite()));
        assert!(RGB_BIAS.iter().all(|value| value.is_finite()));
    }
}

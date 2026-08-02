//! Per-step previews for the shared FLUX.1/Chroma 16-channel VAE latent space.
//!
//! The DiTs denoise packed `[1, S, 64]` tokens. Projection happens only after FLUX.1's
//! provider-owned unpack seam has recovered the native `[1, 16, H/8, W/8]` VAE latent.

use mlx_gen::{PreviewSink, Result};
use mlx_rs::Array;

/// Ordinary-least-squares map from the native FLUX.1 VAE latent to RGB.
///
/// Fit on four diverse real-weight FLUX.1-dev Q4 renders and measured on two disjoint prompt/seed
/// holdouts, all 256² with eight flow-Euler steps. Targets are 8×8-average-pooled native VAE
/// decodes. Fit R² `(R,G,B) = (0.98570, 0.97910, 0.98121)`, overall `0.98224`; holdout R²
/// `(0.89993, 0.89133, 0.94286)`, overall `0.92176`. Retained comparisons preserve coarse
/// composition and palette on every fit and holdout sample.
///
/// Snapshot: `SceneWorks/flux1-dev-mlx` revision
/// `323fd12d79f78ad444e882e8d8e871914584f2b9`, `q4` tier. The fitted VAE file is
/// `vae/model.safetensors`, 164,654,042 bytes, SHA-256
/// `e510ed25d48de4a52d5e00189e7ad57f346c16a167ad0388fc8c90e0cf5e4823`.
///
/// Chroma's three pinned Q4 tiers reuse the same latent basis and exact FLUX.1 loader. Their VAE
/// files are each byte-identical to `black-forest-labs/FLUX.1-dev` revision
/// `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`: 167,666,902 bytes, SHA-256
/// `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`. This was verified for
/// Chroma HD `9d99afe1ebca67032476756bc70d4a7152bc1bd5`, Base
/// `e7330dda29d00ffdeeb719b28e92ee74cff0884c`, and Flash
/// `6a9cb6178709559461506bf247f708d0d1008d00`.
///
/// Reproduce with `tests/fit_preview_rgb.rs`. A stale fit affects decorative preview colour only;
/// the render never reads these constants.
const RGB_FACTORS: [[f32; 3]; 16] = [
    [-0.012_527_851, 0.016_290_851, 0.043_425_495],
    [0.013_447_882, 0.033_666_9, 0.052_629_194],
    [0.028_631_245, -0.018_569_952, -0.007_021_428],
    [-0.009_539_733, 0.008_372_801, 0.035_320_2],
    [0.041_892_606, 0.028_447_104, 0.010_048_334],
    [0.006_346_499, 0.017_091_932, 0.013_491_33],
    [0.013_718_514, 0.050_447_08, 0.043_962_63],
    [-0.023_302_208, -0.018_885_406, -0.026_715_806],
    [-0.023_010_249, 0.008_753_829, 0.058_855_95],
    [0.072_232_775, 0.050_650_94, -0.021_402_475],
    [-0.015_363_334, 0.023_666_509, 0.009_932_561],
    [0.045_652_762, 0.013_941_527, 0.009_122_111],
    [0.028_260_466, 0.024_255_602, 0.027_867_623],
    [-0.080_196_46, -0.031_118_49, -0.082_909_666],
    [-0.011_887_487, -0.042_129_558, -0.013_012_416],
    [-0.060_526_643, -0.030_449_005, -0.025_197_53],
];
const RGB_BIAS: [f32; 3] = [0.495_903_46, 0.492_634_95, 0.482_487_62];

fn project(packed: &Array, width: u32, height: u32) -> Result<mlx_gen::Image> {
    let unpacked = crate::pipeline::unpack_latents(packed, width, height)?;
    mlx_gen::preview::project_latents(&unpacked, &RGB_FACTORS, RGB_BIAS)
}

/// Emit one packed FLUX.1/Chroma latent for an actual outer solver step.
#[allow(clippy::too_many_arguments)]
pub fn emit_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    latents: &Array,
    width: u32,
    height: u32,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || {
        project(latents, width, height)
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn packed_layout_projects_at_native_latent_resolution() {
        let sigmas = [1.0, 0.5, 0.0];
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let packed = Array::zeros::<f32>(&[1, 6, 64]).unwrap();

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &packed, 48, 32);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (1, 2));
        assert_eq!((frames[0].image.width, frames[0].image.height), (6, 4));
    }

    #[test]
    fn malformed_latent_is_decorative_and_consumes_its_position() {
        let sigmas = [1.0, 0.5, 0.0];
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let malformed = Array::zeros::<f32>(&[1, 5, 64]).unwrap();
        let valid = Array::zeros::<f32>(&[1, 4, 64]).unwrap();

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &malformed, 32, 32);
        emit_preview(&sink, &counter, &sigmas, sigmas[0], &valid, 32, 32);
        emit_preview(&sink, &counter, &sigmas, sigmas[1], &valid, 32, 32);

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

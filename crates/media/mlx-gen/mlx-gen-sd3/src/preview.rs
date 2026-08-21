//! Per-step previews for SD3.5's native 16-channel VAE latent space.

use mlx_gen::PreviewSink;
use mlx_rs::Array;

/// Ordinary-least-squares map from the normalized SD3.5 VAE latent to RGB.
///
/// Fit on four diverse real-weight SD3.5-Large renders and measured on two disjoint prompt/seed
/// holdouts, all 256x256 with eight static-shift-3 flow-Euler steps at CFG 3.5. Targets are
/// 8x8-average-pooled native VAE decodes. Fit R² `(R,G,B) = (0.97402, 0.98248, 0.98505)`, overall
/// `0.98031`; holdout R² `(0.86452, 0.87254, 0.95631)`, overall `0.91459`. Retained comparisons
/// preserve coarse composition and palette on every fit and holdout sample.
///
/// Snapshot: `stabilityai/stable-diffusion-3.5-large` revision
/// `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f`. The fitted VAE file is
/// `vae/diffusion_pytorch_model.safetensors`, 167,666,902 bytes, SHA-256
/// `8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc`. The projector consumes
/// the provider's SD3-normalized `(mean - 0.0609) * 1.5305` latent, not Z-Image's normalization.
/// The identical VAE bytes and `vae/config.json` (SHA-256
/// `58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060`) were independently
/// verified in the shipped Q4 tiers for `SceneWorks/sd3.5-large-turbo-mlx` revision
/// `e9166f4632ec64f74d560be3ac778d346f89a364` and `SceneWorks/sd3.5-medium-mlx` revision
/// `5413e962bb326db248be2026a93b147c323392b6`. Reuse across the three registered IDs is therefore
/// exact artifact identity plus the same SD3 loader/normalization, not inference from channel count.
///
/// Reproduce with `tests/fit_preview_rgb.rs`. A stale fit affects decorative preview colour only;
/// the render never reads these constants.
const RGB_FACTORS: [[f32; 3]; 16] = [
    [-0.020_543_499, 0.028_865_399, 0.067_859_3],
    [-0.019_281_747, 0.009_211_603, 0.025_496_975],
    [0.062_378_734, 0.012_928_177, -0.001_648_653],
    [0.048_256_99, 0.021_741_653, 0.045_003_425],
    [0.028_519_169, 0.009_511_666, -0.037_155_278],
    [0.051_960_472, 0.080_396_54, 0.024_337_761],
    [0.095_622_25, 0.048_480_41, 0.052_133_94],
    [-0.013_077_32, 0.057_279_687, 0.071_958_56],
    [-0.048_592_433, -0.025_188_137, 0.021_993_916],
    [-0.055_551_283, 0.003_317_577, -0.023_446_884],
    [0.038_987_22, 0.025_639_156, 0.051_325_99],
    [-0.017_389_223, 0.031_843_77, -0.028_768_186],
    [-0.006_192_283, 0.001_437_803, 0.030_527_327],
    [0.033_097_33, 0.010_442_78, -0.036_355],
    [-0.020_890_785, -0.025_389_63, 0.006_939_386],
    [-0.003_357_253, -0.035_778_474, -0.019_335_456],
];
const RGB_BIAS: [f32; 3] = [0.646_403_3, 0.625_806_57, 0.615_018_96];

/// Emit one native spatial `[1, 16, h, w]` SD3.5 denoise latent for an actual outer solver step.
///
/// Layout validation and projection remain inside the shared best-effort closure. A malformed
/// decorative frame therefore consumes its schedule position but cannot fail or retry generation.
/// An inert sink returns before any tensor operation.
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
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || {
        project_pass_latents(latents)
    });
}

/// Validate the native `[1, 16, h, w]` layout and project it with the committed fit.
///
/// Factored out of [`emit_preview`] for the chained denoise-pass lane (sc-20425), which numbers
/// frames on the executor's chain-global outer step rather than on a σ position and so cannot reuse
/// the σ-keyed emitter. Both lanes projecting through this one function is what keeps the two from
/// drifting to different fits or different layout checks.
///
/// SD3.5's `FlowModelSampling` has `input_scale ≡ 1.0`, so the running latent already is the tensor
/// the fit was measured against and no σ-dependent renormalization is needed here — unlike the ε/VE
/// cohort, whose projector requires the σ.
pub(crate) fn project_pass_latents(latents: &Array) -> mlx_gen::Result<mlx_gen::Image> {
    let shape = latents.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 16 {
        return Err(mlx_gen::Error::Msg(format!(
            "SD3.5 preview latent must have shape [1, 16, h, w], got {shape:?}"
        )));
    }
    mlx_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn native_layout_projects_at_latent_resolution() {
        let sigmas = [1.0, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let latents = Array::zeros::<f32>(&[1, 16, 3, 5]).unwrap();

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &latents);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (1, 2));
        assert_eq!((frames[0].image.width, frames[0].image.height), (5, 3));
        assert_eq!(frames[0].image.pixels.len(), 5 * 3 * 3);
    }

    #[test]
    fn malformed_layout_is_decorative_and_consumes_position() {
        let sigmas = [1.0, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let malformed = Array::zeros::<f32>(&[1, 15, 2, 2]).unwrap();
        let valid = Array::zeros::<f32>(&[1, 16, 2, 2]).unwrap();

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &malformed);
        emit_preview(&sink, &counter, &sigmas, sigmas[0], &valid);
        emit_preview(&sink, &counter, &sigmas, sigmas[1], &valid);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    #[test]
    fn inert_sink_avoids_layout_validation_and_tensor_work() {
        let sigmas = [1.0, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let malformed = Array::zeros::<f32>(&[9]).unwrap();

        emit_preview(
            &PreviewSink::default(),
            &counter,
            &sigmas,
            sigmas[0],
            &malformed,
        );

        assert_eq!(counter.next(&sigmas, sigmas[0]), Some(1));
    }

    #[test]
    fn committed_fit_has_one_finite_row_per_latent_channel() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert!(RGB_FACTORS.iter().flatten().all(|value| value.is_finite()));
        assert!(RGB_BIAS.iter().all(|value| value.is_finite()));
    }
}

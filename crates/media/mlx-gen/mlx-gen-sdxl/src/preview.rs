//! Per-step latent previews for the shared SDXL/Kolors four-channel VAE latent space.
//!
//! SDXL owns the fitted latent-to-RGB transform. Kolors uses the same AutoencoderKL latent
//! convention and calls the narrow [`emit_nhwc_preview`] seam; schedule ownership remains with the
//! route that is actually denoising.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::PreviewSink;

/// Ordinary-least-squares map from SDXL-family VAE latents to latent-resolution RGB. Fit on four
/// diverse 512² real-weight SDXL renders (warm/cool, indoor/outdoor, portrait/still-life/landscape;
/// seeds 1663301..1663304) and evaluated on two disjoint subject/palette holdouts (seeds 1663391 and
/// 1663392), all 12-step ancestral Euler at CFG 5.0. Targets are 8x8-average-pooled VAE decodes.
///
/// Fit R² `(R,G,B) = (0.91640, 0.92538, 0.91487)`, overall `0.91849`. Holdout R²
/// `(0.86501, 0.84844, 0.86649)`, overall `0.86065`. The retained comparison images preserve the
/// large colour regions and coarse composition well; fine detail is intentionally absent at 64².
/// That makes this a useful denoise-progress preview, not a substitute decoder or a claim of final
/// image fidelity. Reproduce and retain evidence with `tests/fit_preview_rgb.rs`.
///
/// Refit whenever the SDXL-family VAE lineage or latent normalization changes. Kolors reuse is
/// grounded in the validated real snapshots, not merely a matching Rust type: their fp16 VAE
/// safetensors were byte-identical (SHA-256
/// `bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68`), and the retained Kolors
/// frame strip independently demonstrates useful palette/composition progression.
const RGB_FACTORS: [[f32; 3]; 4] = [
    [0.171_078_03, 0.205_344_2, 0.213_290_84],
    [-0.128_209_89, 0.028939432, 0.044224623],
    [0.046837712, 0.052948396, 0.006_726_24],
    [-0.181_879_64, -0.124_704_68, -0.124_656_26],
];
const RGB_BIAS: [f32; 3] = [0.555_939, 0.509_310_5, 0.492_320_7];

/// Project and emit one raw SDXL-family denoise latent in native NHWC layout `[1, h, w, 4]`.
///
/// The layout validation and NHWC→NCHW transpose intentionally live inside the shared best-effort
/// projection closure. A malformed decorative frame therefore consumes its schedule position but
/// cannot fail or retry the generation. An inert sink returns before any tensor operation.
pub fn emit_nhwc_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    latents: &Array,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || project_nhwc(latents));
}

/// Project and emit one raw k-diffusion VE-space SDXL-family latent.
///
/// The fitted projection was measured on the renormalized latent seen by the U-Net. Curated SDXL
/// and Kolors samplers carry `x0 + noise * sigma`, so this path applies their
/// `1 / sqrt(sigma^2 + 1)` input scale before projection. A missing sigma is an error and therefore
/// loses only the decorative frame instead of recreating the saturated-preview defect.
pub fn emit_nhwc_ve_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    latents: &Array,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview_with_sigma(sink, counter, sigmas, sigma, |sigma| {
        project_ve_latents(latents, sigma)
    });
}

/// Project a raw VE-space NHWC latent after applying the SDXL-family input scaling.
pub fn project_ve_latents(latents: &Array, sigma: Option<f32>) -> mlx_gen::Result<mlx_gen::Image> {
    let Some(sigma) = sigma else {
        return Err(mlx_gen::Error::Msg(
            "sdxl preview: a VE-space latent needs the schedule sigma to renormalize with, but the \
             driver supplied none"
                .into(),
        ));
    };
    let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
    project_nhwc(&multiply(latents, scalar(scale))?)
}

fn project_nhwc(latents: &Array) -> mlx_gen::Result<mlx_gen::Image> {
    let shape = latents.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[3] != 4 {
        return Err(mlx_gen::Error::Msg(format!(
            "SDXL preview latent must have shape [1, h, w, 4], got {shape:?}"
        )));
    }
    let nchw = latents.transpose_axes(&[0, 3, 1, 2])?;
    mlx_gen::preview::project_latents(&nchw, &RGB_FACTORS, RGB_BIAS)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn nhwc_projection_emits_latent_resolution_frame() {
        let latents = Array::zeros::<f32>(&[1, 2, 3, 4]).unwrap();
        let sigmas = [1.0, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));

        emit_nhwc_preview(&sink, &counter, &sigmas, sigmas[0], &latents);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (1, 1));
        assert_eq!((frames[0].image.width, frames[0].image.height), (3, 2));
        assert_eq!(frames[0].image.pixels, [142, 130, 126].repeat(6));
    }

    #[test]
    fn ve_correction_removes_early_frame_saturation() {
        let latents = Array::from_slice(
            &(0..4 * 4 * 4)
                .map(|i| ((((i * 37) % 101) as f32 / 50.0) - 1.0) * 14.6)
                .collect::<Vec<_>>(),
            &[1, 4, 4, 4],
        )
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();

        let raw = project_nhwc(&latents).unwrap();
        let early = project_ve_latents(&latents, Some(14.6)).unwrap();
        let late = project_ve_latents(&latents, Some(0.0292)).unwrap();
        let rail_fraction = |pixels: &[u8]| {
            pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f32 / pixels.len() as f32
        };

        assert!(rail_fraction(&raw.pixels) > 0.50);
        assert!(rail_fraction(&early.pixels) < 0.10);
        assert_ne!(raw.pixels, early.pixels);
        assert_eq!(raw.pixels, late.pixels);
    }

    #[test]
    fn ve_emitter_uses_the_sigma_aware_projection() {
        let latents = Array::from_slice(
            &(0..4 * 4 * 4)
                .map(|i| ((((i * 37) % 101) as f32 / 50.0) - 1.0) * 14.6)
                .collect::<Vec<_>>(),
            &[1, 4, 4, 4],
        )
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
        let sigmas = [14.6, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));

        emit_nhwc_ve_preview(&sink, &counter, &sigmas, sigmas[0], &latents);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        let pixels = &frames[0].image.pixels;
        let rail_fraction =
            pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f32 / pixels.len() as f32;
        assert!(rail_fraction < 0.10, "rail fraction {rail_fraction}");
    }

    #[test]
    fn ve_projection_rejects_a_missing_sigma() {
        let latents = Array::zeros::<f32>(&[1, 2, 3, 4]).unwrap();
        let error = project_ve_latents(&latents, None).unwrap_err().to_string();
        assert!(error.contains("needs the schedule sigma"), "{error}");
    }

    #[test]
    fn failed_nhwc_layout_is_decorative_and_consumes_position() {
        let invalid = Array::zeros::<f32>(&[1, 4, 2, 3]).unwrap();
        let valid = Array::zeros::<f32>(&[1, 2, 3, 4]).unwrap();
        let sigmas = [1.0, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));

        emit_nhwc_preview(&sink, &counter, &sigmas, sigmas[0], &invalid);
        emit_nhwc_preview(&sink, &counter, &sigmas, sigmas[0], &valid);
        emit_nhwc_preview(&sink, &counter, &sigmas, sigmas[1], &valid);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    #[test]
    fn inert_sink_does_not_evaluate_invalid_layout() {
        let invalid = Array::zeros::<f32>(&[9]).unwrap();
        let sigmas = [1.0, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        emit_nhwc_preview(
            &PreviewSink::default(),
            &counter,
            &sigmas,
            sigmas[0],
            &invalid,
        );
        assert_eq!(counter.next(&sigmas, sigmas[0]), Some(1));
    }
}

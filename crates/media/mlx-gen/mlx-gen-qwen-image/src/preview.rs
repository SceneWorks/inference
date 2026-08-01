//! Per-step latent preview for the Qwen-Image denoise loops — the [`PreviewSink`] request seam
//! (`gen-core::runtime`).
//!
//! This module owns only Qwen's packed-latent conversion and fitted RGB constants. Schedule
//! numbering, projection, and emission live in [`mlx_gen::preview`], shared by every MLX family.
//!
//! Everything here is gated on [`PreviewSink::is_active`], so a request that carries the inert
//! default sink pays one branch per denoise evaluation and nothing else — the projection, the
//! unpack, and the allocation are all skipped.

use mlx_rs::Array;

use mlx_gen::PreviewSink;

use crate::pipeline::unpack_latents;

/// Least-squares latent→RGB factors for the Qwen-Image VAE latent space (16 channels; row *i* maps
/// latent channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the intercept.
///
/// Fit by ordinary least squares on `decoded_rgb ≈ latent · M + b` over (final unpacked latent,
/// 8×-downsampled VAE decode) pairs — 2 prompts/seeds, 8-step Lightning at 1024², 32,768 samples,
/// R² = 0.9586.
///
/// **Refit whenever the VAE lineage changes**, with `tests/fit_preview_rgb.rs`:
///
/// ```sh
/// QWEN_IMAGE_SNAPSHOT=… cargo test -p mlx-gen-qwen-image --release \
///   --test fit_preview_rgb -- --ignored --nocapture
/// ```
///
/// That producer renders the corpus, solves the system, and prints this block ready to paste. It
/// exists because this comment used to end at "re-solving" — naming a procedure with no
/// implementation, which left the constants unreproducible and the R² above uncheckable by anyone
/// who did not fit them. The original corpus is gone, so the producer reports its delta against
/// these values rather than claiming to replay them.
///
/// A stale fit degrades preview colour only; it cannot affect the render, which never reads these.
///
/// **These are not Qwen-Image-only.** `mlx-gen-krea` reuses [`QwenVae`](crate::QwenVae) directly, so
/// the same latent space — and therefore the same fit — applies to the Krea family unchanged. A
/// second family needs its own fit only if it has its own VAE.
const RGB_FACTORS: [[f32; 3]; 16] = [
    [-0.00986379, 0.0257554, 0.211834],
    [-0.00150066, -0.00355605, 0.00219657],
    [0.0881243, 0.0565462, 0.0390654],
    [0.166173, 0.180288, 0.0838119],
    [0.0081918, -0.00272948, -0.0139806],
    [0.0276023, -0.0379166, -0.0372937],
    [-0.144053, -0.167288, -0.107295],
    [-0.0423725, -0.004423, 0.00174681],
    [-0.0705916, -0.0879479, -0.17535],
    [-0.0603724, 0.0326614, 0.0934403],
    [0.0473827, 0.121914, 0.0651104],
    [0.0138456, 0.0267495, 0.0120851],
    [-0.0844989, -0.0160223, 0.0123298],
    [-0.0162293, -0.0335703, -0.018524],
    [0.111816, 0.050061, 0.0724697],
    [0.0448471, 0.0208121, 0.0407526],
];

/// Intercept of the [`RGB_FACTORS`] fit — the mid-grey a zero latent projects to.
const RGB_BIAS: [f32; 3] = [0.406258, 0.385829, 0.287052];

/// Unpack the current Qwen latent `[1, seq, 64]` and hand it to the shared preview machinery.
///
/// Never fails the denoise: a projection error is swallowed, because a preview is decoration and
/// losing a frame must not cost the caller a render. Inert sinks return before even unpacking.
pub(crate) fn emit_preview(
    sink: &PreviewSink,
    counter: &mlx_gen::preview::PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    packed: &Array,
    width: u32,
    height: u32,
) {
    if !sink.is_active() {
        return;
    }
    mlx_gen::preview::emit_preview(sink, counter, sigmas, sigma, || {
        let unpacked = unpack_latents(packed, width, height)?;
        mlx_gen::preview::project_latents(&unpacked, &RGB_FACTORS, RGB_BIAS)
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn fixed_seed_qwen_preview_bytes_remain_stable() {
        let packed = crate::pipeline::create_noise(42, 16, 16).unwrap();
        let sigmas = [1.0_f32, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &packed, 16, 16);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (1, 1));
        assert_eq!((frames[0].image.width, frames[0].image.height), (2, 2));
        // Post-move fixture: this byte-level golden fixes both seeded MLX noise and Qwen's
        // packed→spatial channel ordering. Base-branch equivalence is separate validation evidence,
        // not something this head-only fixture can establish by itself.
        assert_eq!(
            frames[0].image.pixels,
            [120, 69, 59, 0, 0, 0, 126, 90, 115, 64, 152, 178]
        );
    }

    #[test]
    fn failed_unpack_consumes_its_schedule_position() {
        let invalid_packed = Array::zeros::<f32>(&[1, 1, 1]).unwrap();
        let valid_packed = crate::pipeline::create_noise(42, 16, 16).unwrap();
        let sigmas = [1.0_f32, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));

        emit_preview(&sink, &counter, &sigmas, sigmas[0], &invalid_packed, 16, 16);
        emit_preview(&sink, &counter, &sigmas, sigmas[0], &valid_packed, 16, 16);
        emit_preview(&sink, &counter, &sigmas, sigmas[1], &valid_packed, 16, 16);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }
}

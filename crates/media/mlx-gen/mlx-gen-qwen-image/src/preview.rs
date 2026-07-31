//! Per-step latent preview for the Qwen-Image denoise loops — the [`PreviewSink`] request seam
//! (`gen-core::runtime`).
//!
//! A **linear latent→RGB approximation** at latent resolution (`h/8 × w/8`), with **no VAE
//! forward**: one `[n,16]×[16,3]` matmul plus a clip, which is ~1 ms at 128×128 for a 1024²
//! render against tens/hundreds of ms for a decode. That cost ratio is the whole reason a preview
//! can run every step; a real decode per step would roughly double the render.
//!
//! Everything here is gated on [`PreviewSink::is_active`], so a request that carries the inert
//! default sink pays one branch per denoise evaluation and nothing else — the projection, the
//! unpack, and the allocation are all skipped.

use std::cell::Cell;

use mlx_rs::ops::{add, matmul, maximum, minimum, multiply, round};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::{Image, PreviewFrame, PreviewSink, Result};

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

/// Numbers a denoise loop's preview frames, keyed by **schedule position** rather than by
/// evaluation count.
///
/// The `predict` closure a provider hands to `run_flow_sampler` runs once per solver *evaluation*,
/// which is once per step for Euler/Lightning but twice per step for the Heun family. Counting
/// evaluations would therefore reach `total` halfway through a Heun render and then stall. Instead
/// each frame is numbered by where its `sigma` sits in the schedule, and a position already emitted
/// is skipped — one frame per step, correctly numbered, for any solver. A sigma that is not a
/// schedule entry at all (a solver evaluating at an interpolated point) falls back to "one past the
/// last emitted", so numbering stays strictly monotonic and bounded either way.
pub(crate) struct PreviewCounter {
    emitted: Cell<u32>,
    total: u32,
}

impl PreviewCounter {
    /// `sigmas` is the schedule slice the sampler will integrate: `n + 1` entries for `n` steps.
    pub(crate) fn new(sigmas: &[f32]) -> Self {
        Self {
            emitted: Cell::new(0),
            total: sigmas.len().saturating_sub(1).max(1) as u32,
        }
    }

    pub(crate) fn total(&self) -> u32 {
        self.total
    }

    /// The 1-based frame number for an evaluation at `sigma`, or `None` when this schedule position
    /// has already produced a frame.
    pub(crate) fn next(&self, sigmas: &[f32], sigma: f32) -> Option<u32> {
        let candidate = match sigmas.iter().position(|s| *s == sigma) {
            Some(index) => index as u32 + 1,
            None => self.emitted.get().saturating_add(1),
        }
        .min(self.total);
        if candidate <= self.emitted.get() {
            return None;
        }
        self.emitted.set(candidate);
        Some(candidate)
    }
}

/// Project the current packed latent `[1, seq, 64]` and hand the frame to `sink`.
///
/// Never fails the denoise: a projection error is swallowed, because a preview is decoration and
/// losing a frame must not cost the caller a render. Inert sinks return immediately.
pub(crate) fn emit_preview(
    sink: &PreviewSink,
    counter: &PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    packed: &Array,
    width: u32,
    height: u32,
) {
    if !sink.is_active() {
        return;
    }
    let Some(current) = counter.next(sigmas, sigma) else {
        return;
    };
    if let Ok(image) = project(packed, width, height) {
        sink.emit(PreviewFrame {
            current,
            total: counter.total(),
            image,
        });
    }
}

/// Packed latent → RGB8 [`Image`] at latent resolution, via the linear factors.
fn project(packed: &Array, width: u32, height: u32) -> Result<Image> {
    let x = unpack_latents(packed, width, height)?; // [1, 16, h/8, w/8]
    let shape = x.shape().to_vec();
    let (h, w) = (shape[2], shape[3]);
    let x = x.as_dtype(Dtype::Float32)?;
    let x = x.reshape(&[16, h * w])?.transpose_axes(&[1, 0])?; // [n, 16]

    let factors = Array::from_slice(&RGB_FACTORS.concat(), &[16, 3]);
    let bias = Array::from_slice(&RGB_BIAS, &[3]);
    let rgb = add(&matmul(&x, &factors)?, &bias)?; // [n, 3], roughly [0, 1]
    let rgb = minimum(&maximum(&rgb, scalar(0.0))?, scalar(1.0))?;
    let rgb = round(&multiply(&rgb, scalar(255.0))?, 0)?;
    let pixels: Vec<u8> = rgb.as_slice::<f32>().iter().map(|&v| v as u8).collect();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An 8-step schedule has 9 sigmas. Euler evaluates once per step, at sigmas[0..8].
    const SIGMAS: [f32; 9] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0];

    #[test]
    fn euler_cadence_numbers_every_step_once() {
        let counter = PreviewCounter::new(&SIGMAS);
        assert_eq!(counter.total(), 8);
        let frames: Vec<_> = SIGMAS[..8]
            .iter()
            .filter_map(|&s| counter.next(&SIGMAS, s))
            .collect();
        assert_eq!(frames, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// The bug this counter exists to prevent: a Heun-family solver evaluates twice per step, so a
    /// naive eval counter reaches `total` at the halfway point and reports "8/8" for the back half
    /// of the render. Keyed by schedule position, the second evaluation of a step is skipped.
    #[test]
    fn heun_cadence_does_not_overrun_the_step_count() {
        let counter = PreviewCounter::new(&SIGMAS);
        let mut frames = Vec::new();
        for pair in SIGMAS.windows(2).take(8) {
            // predictor at sigma[i], corrector at sigma[i + 1]
            frames.extend(counter.next(&SIGMAS, pair[0]));
            frames.extend(counter.next(&SIGMAS, pair[1]));
        }
        assert_eq!(frames, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(frames.iter().all(|&f| f <= counter.total()));
    }

    /// A solver evaluating at an interpolated sigma still gets a monotonic, bounded number.
    #[test]
    fn off_schedule_sigma_falls_back_to_monotonic_numbering() {
        let counter = PreviewCounter::new(&SIGMAS);
        assert_eq!(counter.next(&SIGMAS, 0.95), Some(1));
        assert_eq!(counter.next(&SIGMAS, 0.85), Some(2));
        // Back onto the schedule at a position already covered: skipped, never rewound.
        assert_eq!(counter.next(&SIGMAS, 0.9), None);
        assert_eq!(counter.next(&SIGMAS, 0.8), Some(3));
    }

    /// A degenerate one-entry schedule must not divide by zero or report `total = 0`.
    #[test]
    fn single_sigma_schedule_has_a_total_of_one() {
        let sigmas = [1.0_f32];
        let counter = PreviewCounter::new(&sigmas);
        assert_eq!(counter.total(), 1);
        assert_eq!(counter.next(&sigmas, 1.0), Some(1));
        assert_eq!(counter.next(&sigmas, 1.0), None);
    }
}

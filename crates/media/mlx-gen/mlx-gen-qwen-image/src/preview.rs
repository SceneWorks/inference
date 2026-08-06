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
/// MLX_GEN_QWEN_SNAPSHOT=…/bf16 QWEN_LIGHTNING_SNAPSHOT=…/models--lightx2v--Qwen-Image-Lightning/snapshots/<rev> \
///   cargo test -p mlx-gen-qwen-image --release --test fit_preview_rgb -- --ignored --nocapture
/// ```
///
/// Both variables, not one: the corpus renders through the 8-step Lightning **adapter**, because
/// "8-step Lightning" is what this block documents and the distilled schedule alone is half of it.
/// The producer panics on a missing variable rather than skipping.
///
/// That producer renders the corpus, solves the system, and prints this block ready to paste. It
/// exists because this comment used to end at "re-solving" — naming a procedure with no
/// implementation, which left the constants unreproducible and the R² above uncheckable by anyone
/// who did not fit them. The original corpus is gone, so the producer reports its delta against
/// these values rather than claiming to replay them.
///
/// **These values are validated, not merely inherited.** sc-17515 scored them unchanged against
/// `SceneWorks/qwen-image-mlx@8080a417` bf16 over 32,768 fresh samples: **R² = 0.9450, mean |ΔRGB|
/// 12.58/255**, against **R² = 0.9633, mean |ΔRGB| 10.01/255** for an in-sample re-solve on the same
/// corpus. Both are clamped the way [`project_latents`](mlx_gen::preview::project_latents) clamps
/// before the 8-bit round, and the producer computes and prints the pair from one run — so the
/// comparison is reproducible rather than asserted. (Its paste-ready block is annotated with the
/// *unclamped* re-solve, 0.9521 on that corpus. That is the number the floor gates and it is not
/// comparable to either figure above.) Giving up 0.018 of R² and 2.58/255 of mean error to a
/// best-case, in-sample refit is not grounds for re-baselining shipping constants onto a two-render
/// sample, which is why these are unchanged. Those two deltas are the producer's own
/// `a refit would buy` line — `+0.0183` and `2.58`, differenced from the UNROUNDED quantities, so
/// the mean-error gap reads 2.58 and not the 2.57 you get by subtracting the rounded 10.01 from the
/// rounded 12.58.
///
/// What the producer asserts every run is the **0.90 floor, against these constants** — not the
/// 0.9450 itself, so a slide from 0.9450 to 0.91 would still pass. That is deliberate: the floor
/// asks whether the linear projection still describes this VAE at all, which is the failure that
/// reaches a preview, and a tighter bound on a two-render corpus would flake on sampling noise. A
/// VAE re-pin that moves the lineage out from under this block fails its lane instead of silently
/// degrading previews. (The R² = 0.0114 that story opened on was a defect in the producer's host
/// readback, not in these constants — `mlx-rs` `as_slice` returns physical storage for a transpose
/// view. `project_latents` is unaffected: it performs the same reshape/transpose but consumes it
/// with a stride-aware `matmul`, and reads back only that op's contiguous output.)
///
/// A stale fit degrades preview colour only; it cannot affect the render, which never reads these.
///
/// **These are not Qwen-Image-only.** `mlx-gen-krea` reuses [`QwenVae`](crate::QwenVae) directly, so
/// the same latent space — and therefore the same fit — applies to the Krea family unchanged. A
/// second family needs its own fit only if it has its own VAE.
///
/// `pub` for one reason: `tests/fit_preview_rgb.rs` is an integration test — a separate crate — and
/// its drift gate has to score **these** values, not a copy of them. It carried a hand-typed
/// duplicate while it only printed a delta report; now that the duplicate feeds an `assert!`, a
/// divergent copy would make the weekly lane score the OLD constants and report green while previews
/// use the new ones — a false green on precisely the scenario the gate was added for. `#[doc(hidden)]`
/// because that is a test seam, not API: these were private, so this changes nothing about the
/// rendered docs.
#[doc(hidden)]
pub const RGB_FACTORS: [[f32; 3]; 16] = [
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
///
/// `pub` on the same terms, and for the same reason, as [`RGB_FACTORS`].
#[doc(hidden)]
pub const RGB_BIAS: [f32; 3] = [0.406258, 0.385829, 0.287052];

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
        project_spatial_latents(&unpacked)
    });
}

/// Project and emit a Qwen-family spatial latent `[1, 16, h, w]` without packed-token conversion.
///
/// Krea 2 reuses [`crate::QwenVae`] and keeps its denoise state in this spatial layout, so this is
/// the provider-owned reuse seam for the same fitted RGB coefficients. Callers must still own the
/// schedule counter because only the generation route knows whether the current trajectory is a
/// full schedule, an img2img tail, or a multi-phase slice. An inert sink returns before projection.
pub fn emit_spatial_preview(
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
        project_spatial_latents(latents)
    });
}

/// Project and emit a singleton-frame Qwen-VAE latent `[1, 16, 1, h, w]`.
///
/// Anima reuses [`crate::QwenVae`] and its normalized 16-channel latent space, but Cosmos keeps a
/// singleton temporal axis during still-image denoise. The reshape stays inside the shared
/// best-effort projection closure so an invalid layout loses only that decorative frame while still
/// consuming its schedule position; it can never fail the generation or be retried as a duplicate.
pub fn emit_single_frame_preview(
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
        let shape = latents.shape();
        if shape.len() != 5 || shape[0] != 1 || shape[1] != 16 || shape[2] != 1 {
            return Err(mlx_gen::Error::Msg(format!(
                "single-frame Qwen preview latent must have shape [1, 16, 1, h, w], got {shape:?}"
            )));
        }
        let spatial = latents.reshape(&[1, shape[1], shape[3], shape[4]])?;
        project_spatial_latents(&spatial)
    });
}

fn project_spatial_latents(latents: &Array) -> mlx_gen::Result<mlx_gen::Image> {
    mlx_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
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

    #[test]
    fn spatial_reuse_is_byte_identical_to_qwen_unpack_projection() {
        let packed = crate::pipeline::create_noise(42, 16, 16).unwrap();
        let spatial = unpack_latents(&packed, 16, 16).unwrap();
        let sigmas = [1.0_f32, 0.0];

        let packed_frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&packed_frames);
        let packed_sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        emit_preview(
            &packed_sink,
            &mlx_gen::preview::PreviewCounter::new(&sigmas),
            &sigmas,
            sigmas[0],
            &packed,
            16,
            16,
        );

        let spatial_frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&spatial_frames);
        let spatial_sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        emit_spatial_preview(
            &spatial_sink,
            &mlx_gen::preview::PreviewCounter::new(&sigmas),
            &sigmas,
            sigmas[0],
            &spatial,
        );

        assert_eq!(
            packed_frames.lock().unwrap()[0].image,
            spatial_frames.lock().unwrap()[0].image
        );
    }

    #[test]
    fn failed_single_frame_reshape_is_decorative_and_consumes_position() {
        let invalid_video = Array::zeros::<f32>(&[1, 16, 2, 2, 2]).unwrap();
        let valid_still = Array::zeros::<f32>(&[1, 16, 1, 2, 2]).unwrap();
        let sigmas = [1.0_f32, 0.5, 0.0];
        let counter = mlx_gen::preview::PreviewCounter::new(&sigmas);
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));

        emit_single_frame_preview(&sink, &counter, &sigmas, sigmas[0], &invalid_video);
        emit_single_frame_preview(&sink, &counter, &sigmas, sigmas[0], &valid_still);
        emit_single_frame_preview(&sink, &counter, &sigmas, sigmas[1], &valid_still);

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }
}

//! The QwenVae latent→RGB preview fit, and Qwen-Image's own per-step preview seam (epic 16948,
//! sc-16950 / sc-16952; the MLX original is epic 16624 / `mlx-gen-qwen-image/src/preview.rs`).
//!
//! This module owns the fitted constants, the spatial projection that applies them, and — for this
//! crate's own routes — the **packed** projector that unpacks first. Schedule numbering, emission,
//! dedup, and the swallow-on-failure contract live in [`candle_gen::preview`], shared by every candle
//! family (sc-16949).
//!
//! ## Qwen-Image previews are projected AFTER the unpack (sc-16952)
//!
//! Qwen-Image is one of only two candle families with a packed latent seam. Its sampler denoises in
//! the **packed token** space `[1, (H/16)·(W/16), 64]` — `create_noise` samples there and every
//! `run_flow_sampler` running latent stays there — while the fit below is defined over the *spatial*
//! VAE latent `[1, 16, H/8, W/8]`. Projecting the packed sequence directly would not merely look
//! wrong: `[1, seq, 64]` is rank 3, so it fails the `[1, C, h, w]` contract outright, and a rank-4
//! reinterpretation of it would read 64 "channels" of interleaved 2×2 patch cells at half the true
//! resolution. [`project_packed_latents`] therefore runs [`crate::pipeline::unpack_latents`] first —
//! the same inverse patchify the decode tail already applies before the VAE — and only then the fit.
//! Krea, whose denoise state is *already* spatial, keeps using [`project_spatial_latents`] directly.
//!
//! ## What the hook sees on each route
//!
//! All three shipped Qwen-Image render routes drive [`candle_gen::run_flow_sampler`], so all three
//! opt in with a projector closure rather than by restructuring a loop, and all three hand the hook
//! the sampler's running latent — which is structurally the single **target** token sequence:
//!
//! * **CFG never reaches the preview.** Every route is true-CFG, and both the positive and negative
//!   forwards plus `pipeline::compute_guided_noise` run *inside* the predict closure, which returns
//!   one combined velocity. No fused `[2, …]` batch exists anywhere in the sampler.
//! * **Edit reference tokens never reach the preview.** `edit.rs` concatenates the VAE-encoded
//!   reference latents onto the sequence axis *inside* the closure
//!   (`Tensor::cat(&[latents, static_latents], 1)`) and narrows the result back to the noise prefix,
//!   so the sampler's latent stays `[1, (H/16)·(W/16), 64]` — the image being generated, never a
//!   reference. This is the hazard that makes an edit preview show the wrong picture, and it is
//!   closed structurally rather than defensively.
//! * **The control hint never reaches the preview.** `control_fun.rs` keeps its packed 132-channel
//!   VACE context in a closure capture, constant across steps and never part of the running latent.
//!
//! ## The fit is reused, not refitted (sc-16950, re-grounded for this crate by sc-16952)
//!
//! `RGB_FACTORS` / `RGB_BIAS` are the least-squares constants epic 16624 committed at
//! `mlx-gen-qwen-image/src/preview.rs:42`, transcribed verbatim. They are ordinary numbers over a VAE
//! *latent space* with no backend in them, so the correct candle move is to reuse them — but only
//! once the reuse is grounded in **tensor bytes** rather than in a matching Rust type name. That
//! grounding is recorded in `docs/migration/evidence/sc-16950-krea-candle-preview.md` and pinned by
//! `tests::committed_fit_matches_the_mlx_source_block` plus the Krea-side provenance row
//! (`candle-gen-krea/tests/preview_real_weights.rs`):
//!
//! * `krea/Krea-2-Turbo` @ `1161245028ef398cd0a951101b2bbf486464f841` — `vae/` SHA-256
//!   `ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08`;
//! * `krea/Krea-2-Raw` @ `4ad9f4b627a647fad78b3dfeebb09f2654aeb494` — the **same** file SHA-256;
//! * `SceneWorks/qwen-image-mlx` @ `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` — `q4|q8/vae/` SHA-256
//!   `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344`, the snapshot the MLX fit was
//!   measured against.
//!
//! All **194** tensors are value-identical across the two files (126,892,531 values); the container
//! differs only in width — the published Krea `vae/` is an f32 container whose values are all exactly
//! bf16-representable (zero low-16 mantissa bits, every value), and the MLX snapshot stores those same
//! values as bf16. `latents_mean` / `latents_std`, which *define* the normalized latent space the fit
//! was measured in, are identical in both `vae/config.json` files.
//!
//! sc-16952 re-grounded the reuse for **this** crate's own routes, and landed on a strictly stronger
//! result than Krea's: every snapshot a candle Qwen-Image route loads a VAE from publishes the
//! *identical file*, byte for byte, as the fit donor —
//! `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344`, 253,806,966 bytes, with a
//! `vae/config.json` also byte-identical (`c448160dba5ce79c965cb075ee02e18d1c42eb6424f787e5869790d577b56a65`)
//! and therefore carrying the same `latents_mean` / `latents_std`. That covers `Qwen/Qwen-Image-2512`
//! (t2i, and the base the ControlNet/Fun lane reuses for its VAE), `Qwen/Qwen-Image-Edit-2511`, and
//! the packed `SceneWorks/qwen-image-mlx` / `SceneWorks/qwen-image-edit-2511-mlx` q4 and q8 tiers —
//! every tier keeps the VAE dense and unmodified. No tensor-by-tensor argument is needed when the
//! bytes are the same bytes; `tests/preview_real_weights.rs` pins it per snapshot anyway.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::Result;

/// Least-squares latent→RGB factors for the Qwen-Image VAE latent space (16 channels; row *i* maps
/// latent channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the intercept.
///
/// Fit by ordinary least squares on `decoded_rgb ≈ latent · M + b` over (final unpacked latent,
/// 8×-downsampled VAE decode) pairs — 2 prompts/seeds, 8-step Lightning at 1024², 32,768 samples,
/// R² = 0.9586. The producer is `mlx-gen-qwen-image/tests/fit_preview_rgb.rs`; candle has no producer
/// of its own **by design** — a second fit of the same latent space would be a second source of truth
/// for one set of numbers.
///
/// **These are not Qwen-Image-only.** `candle-gen-krea` reuses [`crate::vae::QwenVae`] directly, so
/// the same latent space — and therefore the same fit — applies to the Krea family unchanged, exactly
/// as `mlx-gen-krea` reuses the MLX original. A family needs its own fit only if it has its own VAE.
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

/// The QwenVae latent channel count the fit is defined over. A latent that does not carry exactly
/// this many channels is not in the fitted space, and [`project_spatial_latents`] rejects it rather
/// than projecting a mismatched map.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// Project a **spatial** QwenVae latent `[1, 16, h, w]` to a latent-resolution RGB8 preview.
///
/// This is the provider-owned reuse seam for the fitted coefficients. `candle-gen-krea` keeps its
/// denoise state in exactly this layout (`[1, 16, H/8, W/8]`, the normalized space
/// [`crate::vae::QwenVae::decode`] de-normalizes), so its routes hand the sampler's running latent
/// straight here with no unpack step.
///
/// Errors on any other layout: the caller's frame is then lost and swallowed by
/// [`candle_gen::preview::emit_preview`], which is the intended decorative-failure behaviour.
pub fn project_spatial_latents(latents: &Tensor) -> Result<Image> {
    candle_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

/// Project a **packed** Qwen-Image denoise latent `[1, (H/16)·(W/16), 64]` to a latent-resolution
/// RGB8 preview, by running the inverse 2×2 patchify first.
///
/// This is the seam every Qwen-Image render route projects through, and the reason it exists is the
/// packed latent space this family denoises in: `crate::pipeline::unpack_latents` recovers the
/// spatial `[1, 16, H/8, W/8]` the fit is defined over — exactly as the decode tail does before
/// handing the latent to the VAE — and [`project_spatial_latents`] then applies the constants.
/// `width` / `height` are the request's *image* dimensions, the same pair the route passes to
/// `crate::pipeline::create_noise`; the unpack derives the token grid from them.
///
/// Errors — a latent that is not this route's packed shape, or a `width`/`height` that does not
/// describe it — are what the shared emitter swallows to lose exactly one decorative frame.
pub fn project_packed_latents(packed: &Tensor, width: u32, height: u32) -> Result<Image> {
    let spatial = crate::pipeline::unpack_latents(packed, width, height)?;
    project_spatial_latents(&spatial)
}

/// The per-route preview hook every Qwen-Image render route hands to
/// [`candle_gen::run_flow_sampler`]: a projector closure over [`project_packed_latents`] bound to
/// this render's dimensions. The driver owns frame numbering, multi-eval dedup, and the
/// swallow-on-failure contract (sc-16949), so no route restructures its loop.
///
/// Build it **per image**: a batched t2i request runs one driver call per seed and each call must
/// start a fresh trajectory at frame 1. (The driver builds its own counter per call, so this is a
/// property of the call rather than of the hook — building the hook alongside the call keeps the two
/// impossible to separate.)
pub(crate) fn hook(sink: &PreviewSink, width: u32, height: u32) -> PreviewHook<'_> {
    PreviewHook::new(sink, move |packed: &Tensor| {
        project_packed_latents(packed, width, height)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device};
    use candle_gen::gen_core::{CancelFlag, PreviewFrame, Progress};

    use super::*;

    /// The fit is **reused**, not refitted: these are the epic-16624 constants transcribed verbatim
    /// from `mlx-gen-qwen-image/src/preview.rs`. Pinned as literals here so an edit to either copy
    /// fails rather than silently forking one latent space into two colour maps.
    #[test]
    fn committed_fit_matches_the_mlx_source_block() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(RGB_FACTORS[0], [-0.00986379, 0.0257554, 0.211834]);
        assert_eq!(RGB_FACTORS[6], [-0.144053, -0.167288, -0.107295]);
        assert_eq!(RGB_FACTORS[15], [0.0448471, 0.0208121, 0.0407526]);
        assert_eq!(RGB_BIAS, [0.406258, 0.385829, 0.287052]);
    }

    /// A zero latent projects to the intercept — the mid-grey a preview opens on before any structure
    /// has emerged, and the cheapest end-to-end check that the constants are wired the right way round.
    #[test]
    fn zero_latent_projects_to_the_intercept_grey() {
        let latents = Tensor::zeros((1, 16, 3, 5), DType::F32, &Device::Cpu).unwrap();
        let image = project_spatial_latents(&latents).unwrap();
        assert_eq!((image.width, image.height), (5, 3));
        // 0.406258·255 = 103.6, 0.385829·255 = 98.4, 0.287052·255 = 73.2
        assert_eq!(image.pixels[..3], [104, 98, 73]);
        assert_eq!(image.pixels.len(), 3 * 5 * 3);
        assert!(image.pixels.chunks_exact(3).all(|p| p == [104, 98, 73]));
    }

    /// The projection is latent-resolution, not image-resolution: a 1024² render's `[1, 16, 128, 128]`
    /// latent yields a 128² frame.
    #[test]
    fn projection_is_latent_resolution() {
        let latents = Tensor::zeros((1, 16, 128, 128), DType::F32, &Device::Cpu).unwrap();
        let image = project_spatial_latents(&latents).unwrap();
        assert_eq!((image.width, image.height), (128, 128));
    }

    /// bf16 is the candle GPU denoise dtype; the shared projection casts to f32 up front, so the
    /// Qwen-space seam must accept it rather than panicking in the matmul.
    #[test]
    fn projection_accepts_a_bf16_latent() {
        let latents = Tensor::zeros((1, 16, 2, 2), DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let image = project_spatial_latents(&latents).unwrap();
        assert_eq!(image.pixels[..3], [104, 98, 73]);
    }

    /// A latent outside the fitted space is rejected — the error is what the shared emitter swallows
    /// to lose exactly one decorative frame.
    #[test]
    fn projection_rejects_a_non_qwen_latent_layout() {
        for shape in [(1usize, 4usize, 2usize, 2usize), (1, 32, 2, 2)] {
            let latents = Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
            let error = project_spatial_latents(&latents).unwrap_err();
            assert!(error.to_string().contains("does not match latent channel"));
        }

        let packed = Tensor::zeros((1usize, 64usize, 64usize), DType::F32, &Device::Cpu).unwrap();
        let error = project_spatial_latents(&packed).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"));
    }

    // --- The packed seam: projection runs AFTER the unpack (sc-16952) ------------------------------

    /// A packed latent handed to the SPATIAL projector is rejected outright — it is rank 3, not
    /// `[1, C, h, w]`. This is the failure the packed seam exists to prevent, pinned so "just call the
    /// spatial projector from the sampler" reads as the mistake it is rather than as a shortcut.
    #[test]
    fn a_packed_latent_is_not_projectable_without_the_unpack() {
        let packed = packed_latent(WIDTH, HEIGHT);
        let error = project_spatial_latents(&packed).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"), "{error}");
    }

    /// The packed projector is exactly `unpack_latents` followed by the spatial projector — same
    /// bytes, no second code path for the constants.
    #[test]
    fn packed_projection_equals_the_spatial_projection_of_the_unpacked_latent() {
        let packed = Tensor::rand(-2f32, 2f32, (1, SEQ, 64), &Device::Cpu).unwrap();
        let spatial = crate::pipeline::unpack_latents(&packed, WIDTH, HEIGHT).unwrap();
        assert_eq!(
            spatial.dims(),
            [1, 16, HEIGHT as usize / 8, WIDTH as usize / 8]
        );

        let via_packed = project_packed_latents(&packed, WIDTH, HEIGHT).unwrap();
        let via_spatial = project_spatial_latents(&spatial).unwrap();
        assert_eq!(via_packed.pixels, via_spatial.pixels);
        assert_eq!(
            (via_packed.width, via_packed.height),
            (via_spatial.width, via_spatial.height)
        );
    }

    /// The frame is at **VAE-latent** resolution `H/8 × W/8`, not at the packed **token grid**
    /// `H/16 × W/16`. Half-resolution frames are what projecting the packed sequence as if it were
    /// spatial would silently produce, so the two are asserted apart rather than together.
    #[test]
    fn packed_projection_is_vae_latent_resolution_not_token_grid_resolution() {
        for (width, height) in [(64u32, 64u32), (1024, 1024), (1024, 768)] {
            let (lat_h, lat_w) = crate::pipeline::latent_dims(width, height);
            let packed = Tensor::zeros((1, lat_h * lat_w, 64), DType::F32, &Device::Cpu).unwrap();
            let image = project_packed_latents(&packed, width, height).unwrap();
            assert_eq!(
                (image.width, image.height),
                (width / 8, height / 8),
                "{width}x{height} must project at VAE-latent resolution"
            );
            assert_ne!(
                (image.width, image.height),
                (lat_w as u32, lat_h as u32),
                "{width}x{height} must NOT project at the packed token-grid resolution"
            );
        }
    }

    /// bf16 is the candle GPU denoise dtype and the dtype the packed running latent actually carries.
    #[test]
    fn packed_projection_accepts_a_bf16_latent() {
        let packed = packed_latent(WIDTH, HEIGHT).to_dtype(DType::BF16).unwrap();
        let image = project_packed_latents(&packed, WIDTH, HEIGHT).unwrap();
        assert_eq!(image.pixels[..3], [104, 98, 73]);
    }

    /// Dimensions that do not describe the packed latent are an error, not a reshape onto the wrong
    /// grid. The emitter swallows it and one decorative frame is lost.
    #[test]
    fn packed_projection_rejects_mismatched_dimensions() {
        let packed = packed_latent(WIDTH, HEIGHT);
        assert!(project_packed_latents(&packed, WIDTH * 2, HEIGHT).is_err());
        assert!(project_packed_latents(&packed, WIDTH, HEIGHT * 2).is_err());
    }

    // --- Driving the real sampler ------------------------------------------------------------------

    /// A small but genuinely Qwen-shaped render: 64² → a 4×4 token grid, 16 packed tokens, and an
    /// 8×8 spatial latent.
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const SEQ: usize = 16;

    fn packed_latent(width: u32, height: u32) -> Tensor {
        let (lat_h, lat_w) = crate::pipeline::latent_dims(width, height);
        Tensor::zeros((1, lat_h * lat_w, 64), DType::F32, &Device::Cpu).unwrap()
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    fn frames_of(captured: &Arc<Mutex<Vec<PreviewFrame>>>) -> Vec<(u32, u32)> {
        candle_gen::lock_recover(captured)
            .iter()
            .map(|f| (f.current, f.total))
            .collect()
    }

    /// A velocity of exactly zero: the flow-Euler step leaves the latent untouched, so the sampler's
    /// output is a pure function of its input and any byte difference is the wiring's.
    fn zero_velocity(x: &Tensor, _t: f32) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Drive the real flow sampler over `sigmas`, in the packed space and with the real schedule the
    /// routes resolve — the same driver, convention and argument order all three call sites use.
    fn run(
        sampler: Option<&str>,
        sigmas: &[f32],
        start: Tensor,
        preview: Option<&PreviewHook<'_>>,
        predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        candle_gen::run_flow_sampler(
            sampler,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            sigmas,
            start,
            7,
            &CancelFlag::new(),
            &mut |_: Progress| {},
            preview,
            predict,
        )
    }

    fn sigmas(steps: usize) -> Vec<f32> {
        crate::pipeline::qwen_sigmas(steps, WIDTH, HEIGHT)
    }

    /// Euler evaluates once per step: an N-step render emits exactly N frames, 1..=N, each carrying
    /// `total == N`.
    #[test]
    fn euler_emits_exactly_one_numbered_frame_per_step() {
        for steps in [1usize, 4, 8] {
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink, WIDTH, HEIGHT);
            run(
                None,
                &sigmas(steps),
                packed_latent(WIDTH, HEIGHT),
                Some(&hook),
                zero_velocity,
            )
            .unwrap();
            assert_eq!(
                frames_of(&captured),
                (1..=steps as u32)
                    .map(|n| (n, steps as u32))
                    .collect::<Vec<_>>(),
                "{steps}-step Euler render"
            );
        }
    }

    /// The candle-specific hazard the shared counter exists for: heun and dpmpp_sde evaluate the
    /// predict closure **twice** per outer step, so an undeduped path would emit 2N frames. The
    /// evaluation count is asserted to exceed the step count first, so a solver that silently fell
    /// back to Euler could not make this pass vacuously.
    #[test]
    fn multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step() {
        for name in ["heun", "dpmpp_sde"] {
            let steps = 6usize;
            let evaluations = std::cell::Cell::new(0usize);
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink, WIDTH, HEIGHT);
            run(
                Some(name),
                &sigmas(steps),
                packed_latent(WIDTH, HEIGHT),
                Some(&hook),
                |x, t| {
                    evaluations.set(evaluations.get() + 1);
                    zero_velocity(x, t)
                },
            )
            .unwrap();

            assert!(
                evaluations.get() > steps,
                "{name} must evaluate more than once per step for this test to mean anything \
                 (got {} evaluations for {steps} steps)",
                evaluations.get()
            );
            assert_eq!(
                frames_of(&captured),
                (1..=steps as u32)
                    .map(|n| (n, steps as u32))
                    .collect::<Vec<_>>(),
                "{name} must still emit exactly one frame per outer step"
            );
        }
    }

    /// Every emitted frame is a latent-resolution RGB8 image of the running trajectory.
    #[test]
    fn emitted_frames_are_vae_latent_resolution_rgb8() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink, 128, 64);
        let start = packed_latent(128, 64);
        run(None, &sigmas(2), start, Some(&hook), zero_velocity).unwrap();

        let frames = candle_gen::lock_recover(&captured);
        assert_eq!(frames.len(), 2);
        for frame in frames.iter() {
            assert_eq!((frame.image.width, frame.image.height), (16, 8));
            assert_eq!(frame.image.pixels.len(), 16 * 8 * 3);
        }
    }

    // --- What the hook is allowed to see -----------------------------------------------------------

    /// The CFG hazard, driven through the real sampler with a predict closure shaped like the routes':
    /// it fuses a two-leg batch internally and returns one combined velocity. The unconditional half
    /// never becomes the running latent, so it can never be projected.
    #[test]
    fn cfg_never_exposes_the_unconditional_half_to_the_preview() {
        let (sink, _captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_packed_latents(x, WIDTH, HEIGHT)
        });

        run(
            None,
            &sigmas(4),
            packed_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                // `pipeline::compute_guided_noise`'s shape: two forwards over one fused batch,
                // blended back down to a single conditional-space velocity before returning.
                let fused = Tensor::cat(&[x, x], 0)?;
                assert_eq!(fused.dims()[0], 2);
                let cond = fused.narrow(0, 0, 1)?;
                Ok(cond.zeros_like()?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter().all(|dims| dims == &[1, SEQ, 64]),
            "the hook must only ever see the single unfused conditional latent, got {seen:?}"
        );
    }

    /// The edit hazard, and the one this family actually had to close: the edit closure concatenates
    /// the static reference latents onto the SEQUENCE axis and narrows the forward's result back to
    /// the noise prefix. If the sampler's latent ever carried reference tokens the preview would show
    /// the reference image, and the sequence length is exactly what would give it away — so the seen
    /// shapes are asserted against the noise-only length, with the joint length computed and asserted
    /// different so the row cannot pass by the two happening to coincide.
    #[test]
    fn edit_previews_project_target_tokens_only() {
        let (sink, captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_packed_latents(x, WIDTH, HEIGHT)
        });

        // Two references, packed at their own condition resolution — the dual-latent sequence.
        let references = Tensor::ones((1, 40, 64), DType::F32, &Device::Cpu).unwrap();
        let joint_seq = SEQ + 40;
        assert_ne!(joint_seq, SEQ);

        run(
            None,
            &sigmas(4),
            packed_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                let joint = Tensor::cat(&[x, &references], 1)?;
                assert_eq!(joint.dims()[1], joint_seq);
                // `forward_edit_with_memory(...).narrow(1, 0, noise_seq)`.
                Ok(joint.narrow(1, 0, SEQ)?.zeros_like()?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter().all(|dims| dims == &[1, SEQ, 64]),
            "the hook must never see a reference token, got {seen:?}"
        );
        // And the frames are the target's latent size, not the joint sequence's.
        for (width, height) in candle_gen::lock_recover(&captured)
            .iter()
            .map(|f| (f.image.width, f.image.height))
        {
            assert_eq!((width, height), (WIDTH / 8, HEIGHT / 8));
        }
    }

    // --- Decorative by contract --------------------------------------------------------------------

    /// An inert sink must be byte-identical to no hook at all, and an ACTIVE sink must be too — the
    /// preview reads the latent and never writes it.
    #[test]
    fn an_inert_sink_is_byte_identical_to_an_unhooked_render() {
        let s = sigmas(6);
        let start = Tensor::rand(-1f32, 1f32, (1, SEQ, 64), &Device::Cpu).unwrap();
        let velocity = |x: &Tensor, t: f32| Ok((x * (t as f64 + 0.25))?);
        let bytes = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let bare = run(None, &s, start.clone(), None, velocity).unwrap();

        let inert = PreviewSink::default();
        let inert_hook = hook(&inert, WIDTH, HEIGHT);
        assert!(!inert_hook.is_active());
        let hooked = run(None, &s, start.clone(), Some(&inert_hook), velocity).unwrap();
        assert_eq!(
            bytes(&bare),
            bytes(&hooked),
            "an inert preview sink must not perturb a single latent byte"
        );

        let (sink, captured) = collecting_sink();
        let active_hook = hook(&sink, WIDTH, HEIGHT);
        let active = run(None, &s, start, Some(&active_hook), velocity).unwrap();
        assert_eq!(bytes(&bare), bytes(&active));
        assert_eq!(candle_gen::lock_recover(&captured).len(), 6);
    }

    /// A projection failure loses its frame and never fails the render. The realistic shape of that
    /// failure here is a hook whose dimensions do not describe the running latent.
    #[test]
    fn a_projection_failure_loses_the_frame_and_never_fails_the_render() {
        let (sink, captured) = collecting_sink();
        // A hook built for a 1024² render, handed a 64² trajectory: every unpack fails.
        let hook = hook(&sink, 1024, 1024);
        let out = run(
            None,
            &sigmas(5),
            packed_latent(WIDTH, HEIGHT),
            Some(&hook),
            zero_velocity,
        )
        .expect("a failing projection must not fail the render");

        assert_eq!(out.dims(), [1, SEQ, 64]);
        assert!(
            candle_gen::lock_recover(&captured).is_empty(),
            "no frame may be emitted when every projection fails"
        );
    }

    // --- Route inventory ---------------------------------------------------------------------------

    /// [`candle_gen::run_flow_sampler`]'s argument count before the predict closure. Pinned so a
    /// signature change — or a scanner mis-split — fails this inventory loudly instead of quietly
    /// shifting which argument "the one before the closure" names.
    const SAMPLER_ARGUMENTS_BEFORE_PREDICT: usize = 8;

    /// The arguments of every `run_flow_sampler` call in `source`, one entry per call site, covering
    /// the arguments **before** the predict closure — the window the `preview` argument sits in.
    ///
    /// Ported from sc-16950's Krea inventory. The window is bounded by the call's own bracket balance
    /// and ends at the first top-level `|`; it deliberately does not key off a closure parameter name,
    /// because a route naming that parameter something else would otherwise widen the window to the
    /// next call site (or to end of file) and let any `Some(&preview)` in the swallowed text — prose
    /// included — satisfy a route that was left dark. A missing bound is a failure, not a wider window.
    ///
    /// The match is textual, so writing the driver's name followed by an open paren in prose is read
    /// as a call site: name it without the paren in comments.
    fn sampler_call_sites(file: &str, source: &str) -> Vec<Vec<String>> {
        const CALL: &str = "run_flow_sampler(";
        let mut sites = Vec::new();
        let mut cursor = 0usize;
        while let Some(at) = source[cursor..].find(CALL) {
            let args_start = cursor + at + CALL.len();
            sites.push(sampler_call_arguments(
                file,
                sites.len(),
                &source[args_start..],
            ));
            cursor = args_start;
        }
        sites
    }

    /// The comma-separated top-level arguments of one call, given everything after its open paren.
    fn sampler_call_arguments(file: &str, index: usize, rest: &str) -> Vec<String> {
        let site = format!("{file}: run_flow_sampler call #{index}");
        let normalize = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");

        let mut args: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut depth = 1usize;
        let mut chars = rest.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                // Comments are not code: a `(` or a `|` inside one must not move the scan.
                '/' if chars.peek() == Some(&'/') => {
                    for c in chars.by_ref() {
                        if c == '\n' {
                            break;
                        }
                    }
                    current.push(' ');
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    let (mut nesting, mut prev) = (1usize, '\0');
                    for c in chars.by_ref() {
                        match (prev, c) {
                            ('/', '*') => (nesting, prev) = (nesting + 1, '\0'),
                            ('*', '/') => {
                                nesting -= 1;
                                prev = '\0';
                                if nesting == 0 {
                                    break;
                                }
                            }
                            _ => prev = c,
                        }
                    }
                    assert_eq!(nesting, 0, "{site} has an unterminated block comment");
                    current.push(' ');
                }
                // Nor are string literals.
                '"' => {
                    current.push('"');
                    let mut escaped = false;
                    let mut closed = false;
                    for c in chars.by_ref() {
                        current.push(c);
                        if escaped {
                            escaped = false;
                        } else if c == '\\' {
                            escaped = true;
                        } else if c == '"' {
                            closed = true;
                            break;
                        }
                    }
                    assert!(closed, "{site} has an unterminated string literal");
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    assert!(
                        depth > 0,
                        "{site} closes without a predict closure — the scan cannot bound its \
                         preview argument, so no assertion about that argument would mean anything"
                    );
                    current.push(ch);
                }
                ',' if depth == 1 => {
                    args.push(normalize(&current));
                    current.clear();
                }
                // The predict closure's parameter list: the argument window ends here, whatever that
                // parameter is called.
                '|' if depth == 1 => {
                    let trailing = normalize(&current);
                    assert!(
                        trailing.is_empty(),
                        "{site} has unparsed text {trailing:?} between its last argument and the \
                         predict closure — the scan cannot be trusted to have found the preview \
                         argument"
                    );
                    return args;
                }
                _ => current.push(ch),
            }
        }
        panic!("{site} is unterminated: no predict closure and no closing paren before end of file")
    }

    /// Every shipped Qwen-Image render route emits previews, pinned at the source level: base txt2img
    /// (`lib.rs`), reference edit (`edit.rs`), and the 2512-Fun ControlNet lane (`control_fun.rs`) —
    /// one sampler site each, all three passing a hook. A route left unwired shows the user nothing,
    /// and no weights-free test can otherwise reach a route that needs a 20B DiT.
    ///
    /// This is the crate-local half of the epic-16948 guard; `candle-gen-catalog`'s
    /// `preview_advertising` module carries the same counts as the family's route inventory and ties
    /// them to the advertised `supports_preview`.
    #[test]
    fn every_qwen_image_render_route_passes_a_preview_hook() {
        for (file, source) in [
            ("lib.rs", include_str!("lib.rs")),
            ("edit.rs", include_str!("edit.rs")),
            ("control_fun.rs", include_str!("control_fun.rs")),
        ] {
            let sites = sampler_call_sites(file, source);
            assert_eq!(
                sites.len(),
                1,
                "{file}: expected exactly 1 sampler call site, found {}. A new render route must \
                 pass a preview hook and be named in this inventory (and in the catalog's).",
                sites.len()
            );
            let args = &sites[0];
            assert_eq!(
                args.len(),
                SAMPLER_ARGUMENTS_BEFORE_PREDICT,
                "{file}: expected {SAMPLER_ARGUMENTS_BEFORE_PREDICT} arguments before the predict \
                 closure, parsed {args:?}"
            );
            // Positional, not `contains`: the preview is the argument immediately before the predict
            // closure, so this cannot be satisfied by the word appearing anywhere else.
            assert_eq!(
                args.last().map(String::as_str),
                Some("Some(&preview)"),
                "{file} does not pass a preview hook: {args:?}"
            );
        }
    }

    /// `pipeline.rs` owns the latent geometry and the schedules, not a denoise loop: it must hold no
    /// sampler site at all. Pinned as a negative so a future route added there cannot slip past the
    /// inventory above, which only looks at three named files.
    #[test]
    fn the_geometry_module_drives_no_sampler() {
        assert!(sampler_call_sites("pipeline.rs", include_str!("pipeline.rs")).is_empty());
    }
}

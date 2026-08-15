//! Anima's per-step latent preview seam (epic 16948, sc-16953; the MLX original is epic 16624 /
//! sc-16629).
//!
//! Anima carries **no fit of its own**. Its VAE *is* the Qwen-Image `AutoencoderKLQwenImage` — the
//! crate already reuses [`crate::vae::QwenVae`] wholesale — so the latent space its DiT denoises in is
//! the same normalized 16-channel space the epic-16624 QwenVae least-squares constants were measured
//! over. This module owns exactly one thing: the layout adaptation from Anima's **5-D Cosmos** running
//! latent to the `[1, C, h, w]` contract the shared projection takes, and then defers to
//! `candle_gen_qwen_image::preview::project_spatial_latents` for the fit itself. Schedule numbering,
//! emission, multi-eval dedup and the swallow-on-failure contract live in [`candle_gen::preview`],
//! shared by every candle family (sc-16949).
//!
//! ## The latent shape at the emission point — verified, not assumed
//!
//! Anima denoises in a **spatial** latent, but a 5-D one: `crate::pipeline::create_noise` samples
//! `[1, 16, 1, H/8, W/8]` (the Cosmos-Predict2 video layout with a length-1 temporal axis), and
//! `crate::transformer::CosmosDiT::forward` unpatchifies back to that same rank, so every
//! `candle_gen::run_flow_sampler` running latent stays 5-D from the first σ to the last. That is rank
//! 5, so handing it straight to the shared projection would fail the `[1, C, h, w]` contract outright
//! — Anima is neither Krea (already `[1, 16, H/8, W/8]`, projects directly) nor Qwen-Image (packed
//! `[1, (H/16)·(W/16), 64]`, has to run the inverse patchify first).
//!
//! [`project_single_frame_latents`] therefore drops the length-1 temporal axis — **the same
//! `squeeze` the decode tail already applies** before handing the latent to `QwenVae::decode` — and
//! only then applies the fit. Literally the same: both spell the axis
//! [`crate::config::LATENT_TEMPORAL_AXIS`], so the preview cannot come to disagree with the decode
//! about which axis it is. Because the rest of the geometry travels entirely inside the latent, the
//! projector needs no `width`/`height` argument at all: hook geometry and latent geometry are not
//! merely bound to one source, there is only one source to be bound to.
//!
//! The layout adaptation lives *here* rather than in `candle-gen-qwen-image` (where the MLX twin put
//! it) because on candle the 5-D Cosmos layout is Anima's alone: candle Qwen-Image denoises packed,
//! not 5-D. `project_spatial_latents` is the documented reuse seam for the fitted coefficients, and
//! each candle family owns its own way of reaching it.
//!
//! ## What the hook sees
//!
//! The one shipped Anima render lane (`crate::pipeline::AnimaPipeline::generate`, shared by all three
//! registered variants) drives `candle_gen::run_flow_sampler`, so it opts in with a projector closure
//! rather than by restructuring its loop, and it hands the hook the sampler's running latent — which
//! is structurally the single conditional trajectory:
//!
//! * **CFG never reaches the preview.** `anima_base` / `anima_aesthetic` run true classifier-free
//!   guidance as *two separate DiT forwards inside the predict closure*
//!   (`v_uncond + guidance·(v_cond − v_uncond)`), returning one combined velocity. No fused `[2, …]`
//!   batch exists anywhere in the sampler, so there is no unconditional half for a preview to project.
//!   `anima_turbo` is the merged CFG-free student and runs one forward.
//! * **There are no reference or control tokens to leak.** Anima is txt2img only — `load_variant`
//!   rejects `spec.control`, `spec.extra_controls` and `spec.ip_adapter` — and the text conditioning
//!   reaches the DiT as `encoder_hidden_states`, never as part of the latent.
//!
//! ## The fit is reused, not refitted — grounded in tensor bytes
//!
//! The claim is not "both crates name a type `QwenVae`". Anima publishes its VAE as a single
//! `vae/qwen_image_vae.safetensors` in the **original** Qwen naming, so it is a different *file* from
//! the fit donor; [`crate::vae::convert_vae_key`] is the rename that makes the two comparable. Under
//! that rename the transfer is exact:
//!
//! * `circlestone-labs/Anima` @ `53eec3898af698b2cf2a11379021fc9c5465d228` —
//!   `split_files/vae/qwen_image_vae.safetensors`, SHA-256
//!   `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f`, 253,806,246 bytes;
//! * `SceneWorks/qwen-image-mlx` @ `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` —
//!   `q4|q8/vae/diffusion_pytorch_model.safetensors`, SHA-256
//!   `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344`, 253,806,966 bytes, the
//!   snapshot the epic-16624 fit was measured against and the file sc-16952 pinned for every candle
//!   Qwen-Image lane.
//!
//! **194 of 194 tensors, 126,892,531 values, bit-identical** — and both files are bf16 containers, so
//! this is stronger than sc-16950's Krea comparison, which had an f32-vs-bf16 container difference to
//! argue past. The 720-byte file-size difference is the safetensors *header* alone: the longer
//! diffusers key names, and the payload re-ordering that follows from sorting different names. Pinned
//! by `tests/preview_real_weights.rs`.
//!
//! The per-channel `latents_mean` / `latents_std` that *define* the normalized space need no
//! comparison here and get none: Anima publishes no `vae/config.json`, because candle's `QwenVae`
//! carries those constants in Rust and Anima reuses that very type. The de-normalization is
//! definitionally the same code, not two files that happen to agree.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

use crate::config::LATENT_TEMPORAL_AXIS;

/// The QwenVae latent channel count the reused fit is defined over, re-exported from the crate that
/// owns the constants so Anima cannot drift from it by restating a number.
pub use candle_gen_qwen_image::preview::PREVIEW_LATENT_CHANNELS;

/// Project Anima's 5-D Cosmos running latent `[1, 16, 1, h, w]` to a latent-resolution RGB8 preview.
///
/// The length-1 temporal axis is dropped first — the same squeeze the decode tail applies before
/// `crate::vae::QwenVae::decode` — and the reused QwenVae fit is then applied by
/// `candle_gen_qwen_image::preview::project_spatial_latents`.
///
/// Errors on any other layout, including the already-squeezed `[1, 16, h, w]`: a rank-4 latent is not
/// something this family's sampler can produce, and silently accepting one would hide a real
/// regression in the denoise shape. The caller's frame is then lost and swallowed by
/// `candle_gen::preview::emit_preview`, which is the intended decorative-failure behaviour.
pub fn project_single_frame_latents(latents: &Tensor) -> Result<Image> {
    let spatial = drop_temporal_axis(latents)?;
    candle_gen_qwen_image::preview::project_spatial_latents(&spatial)
}

/// `[1, C, 1, h, w]` → `[1, C, h, w]`, rejecting anything that is not one Cosmos still frame in the
/// fitted channel space.
///
/// Written as a checked reshape rather than a bare `squeeze`, because candle's `squeeze` is a no-op on
/// an axis whose extent is not 1: a genuinely temporal `[1, 16, T>1, h, w]` latent would pass straight
/// through it and only fail later, in the shared projection, with a message about the wrong contract.
fn drop_temporal_axis(latents: &Tensor) -> Result<Tensor> {
    let dims = latents.dims();
    if dims.len() != 5
        || dims[0] != 1
        || dims[1] != PREVIEW_LATENT_CHANNELS
        || dims[LATENT_TEMPORAL_AXIS] != 1
    {
        return Err(CandleError::Msg(format!(
            "anima preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, 1, h, w], got \
             {dims:?}"
        )));
    }
    Ok(latents.squeeze(LATENT_TEMPORAL_AXIS)?)
}

/// The preview hook Anima's render lane hands to `candle_gen::run_flow_sampler`: a projector closure
/// over [`project_single_frame_latents`]. The driver owns frame numbering, multi-eval dedup and the
/// swallow-on-failure contract (sc-16949), so the denoise loop is not restructured.
///
/// Build it **per image**: a batched request runs one driver call per seed and each call must start a
/// fresh trajectory at frame 1. (The driver builds its own counter per call, so this is a property of
/// the call rather than of the hook — building the hook alongside the call keeps the two impossible to
/// separate.)
pub(crate) fn hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project_single_frame_latents)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device};
    use candle_gen::gen_core::sampling::TimestepConvention;
    use candle_gen::gen_core::{CancelFlag, PreviewFrame, Progress};

    use super::*;

    /// A small but genuinely Anima-shaped render: 512² is the advertised minimum, giving a 64×64
    /// spatial latent under the 8× VAE compression.
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;

    fn cosmos_latent(width: u32, height: u32) -> Tensor {
        Tensor::zeros(
            (
                1,
                PREVIEW_LATENT_CHANNELS,
                1,
                (height / 8) as usize,
                (width / 8) as usize,
            ),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
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

    // --- The reuse seam ---------------------------------------------------------------------------

    /// Anima adds no constants of its own: the channel count comes from the crate that owns the fit,
    /// and the projection of a zero latent is the *same* intercept grey `candle-gen-qwen-image` pins.
    /// A second copy of the coefficients here would be a second source of truth for one set of
    /// numbers, so this row asserts the shared value rather than a transcription of it.
    #[test]
    fn the_fit_is_the_shared_qwenvae_one() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);

        let latents = cosmos_latent(WIDTH, HEIGHT);
        let via_anima = project_single_frame_latents(&latents).unwrap();
        let via_qwen = candle_gen_qwen_image::preview::project_spatial_latents(
            &latents.squeeze(LATENT_TEMPORAL_AXIS).unwrap(),
        )
        .unwrap();
        assert_eq!(via_anima.pixels, via_qwen.pixels);
        // 0.406258·255 = 103.6, 0.385829·255 = 98.4, 0.287052·255 = 73.2 — the shared intercept.
        assert_eq!(via_anima.pixels[..3], [104, 98, 73]);
    }

    /// The frame is at **VAE-latent** resolution `H/8 × W/8`, matching what the decode tail feeds the
    /// VAE, and it carries one RGB triplet per latent cell.
    #[test]
    fn projection_is_latent_resolution() {
        for (width, height) in [(512u32, 512u32), (1024, 1024), (1536, 1024)] {
            let image = project_single_frame_latents(&cosmos_latent(width, height)).unwrap();
            assert_eq!((image.width, image.height), (width / 8, height / 8));
            assert_eq!(
                image.pixels.len(),
                (width / 8) as usize * (height / 8) as usize * 3
            );
        }
    }

    /// The layout gate, stated as the three ways it can be wrong. The already-squeezed rank-4 latent
    /// is included deliberately: this family's sampler cannot produce one, so accepting it would hide
    /// a real change in the denoise shape rather than tolerate a harmless variation.
    #[test]
    fn projection_rejects_every_non_cosmos_layout() {
        let shapes: &[&[usize]] = &[
            &[1, 16, 8, 8],    // already squeezed — not a shape this sampler produces
            &[1, 16, 2, 8, 8], // a genuinely temporal latent, which `squeeze` would pass through
            &[1, 4, 1, 8, 8],  // outside the fitted channel space
            &[2, 16, 1, 8, 8], // batched
            &[16, 1, 8, 8],    // rank 4, no leading batch
        ];
        for shape in shapes {
            let latents = Tensor::zeros(*shape, DType::F32, &Device::Cpu).unwrap();
            let error = project_single_frame_latents(&latents).unwrap_err();
            assert!(
                error.to_string().contains("[1, 16, 1, h, w]"),
                "{shape:?} must be rejected by the Anima layout gate, got: {error}"
            );
        }
    }

    /// bf16 is the candle GPU compute dtype. Anima keeps its latents f32 end-to-end on purpose
    /// (sc-10625), but the shared projection casts up front and the seam must not be the thing that
    /// breaks if that ever changes.
    #[test]
    fn projection_accepts_a_low_precision_latent() {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let latents = cosmos_latent(WIDTH, HEIGHT).to_dtype(dtype).unwrap();
            let image = project_single_frame_latents(&latents)
                .unwrap_or_else(|e| panic!("{dtype:?} latent failed to project: {e}"));
            assert_eq!(image.pixels[..3], [104, 98, 73]);
        }
    }

    // --- Driving the real sampler ------------------------------------------------------------------

    /// A velocity of exactly zero: the flow-Euler step leaves the latent untouched, so the sampler's
    /// output is a pure function of its input and any byte difference is the wiring's.
    fn zero_velocity(x: &Tensor, _t: f32) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Drive the real flow sampler over `sigmas`, with the same driver, convention and argument order
    /// the render lane uses.
    fn run(
        sampler: Option<&str>,
        sigmas: &[f32],
        start: Tensor,
        preview: Option<&PreviewHook<'_>>,
        predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::Sigma,
            sigmas,
            start,
            7,
            &CancelFlag::new(),
            &mut |_: Progress| {},
            preview,
            predict,
        )
    }

    /// Euler evaluates once per step: an N-step render emits exactly N frames, 1..=N, each carrying
    /// `total == N`. Driven over the crate's real schedule, not a synthetic σ array.
    #[test]
    fn euler_emits_exactly_one_numbered_frame_per_step() {
        for steps in [1usize, 4, 10, 30] {
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink);
            run(
                Some("euler"),
                &crate::anima_sigmas(steps),
                cosmos_latent(WIDTH, HEIGHT),
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
    ///
    /// `er_sde` — Anima's own default solver — rides along as a third case, so the shipped default is
    /// covered by the same row rather than only the two known multi-eval solvers.
    #[test]
    fn multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step() {
        for name in ["heun", "dpmpp_sde"] {
            let steps = 6usize;
            let evaluations = std::cell::Cell::new(0usize);
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink);
            run(
                Some(name),
                &crate::anima_sigmas(steps),
                cosmos_latent(WIDTH, HEIGHT),
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

    /// A batched request (`count` up to 8) runs one driver call per seed, and each call must start a
    /// fresh trajectory at frame 1 rather than continuing the previous image's numbering — otherwise
    /// the second image's positions are all already emitted and it silently produces no frames at all.
    ///
    /// The property belongs to the *driver*, which builds its own counter per call, so it survives the
    /// hook being reused across calls. Driven that way deliberately: reusing one hook is the shape
    /// that would break if numbering ever moved into the hook.
    #[test]
    fn each_image_of_a_batch_numbers_its_own_trajectory_from_one() {
        let steps = 4usize;
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        for _ in 0..3 {
            run(
                Some("euler"),
                &crate::anima_sigmas(steps),
                cosmos_latent(WIDTH, HEIGHT),
                Some(&hook),
                zero_velocity,
            )
            .unwrap();
        }
        let one_trajectory: Vec<_> = (1..=steps as u32).map(|n| (n, steps as u32)).collect();
        assert_eq!(
            frames_of(&captured),
            [
                one_trajectory.clone(),
                one_trajectory.clone(),
                one_trajectory
            ]
            .concat(),
            "each image in a batch must emit its own 1..=N run"
        );
    }

    /// The shipped default solver gets its own row: whatever cadence `er_sde` evaluates at, the strip
    /// is still exactly one frame per outer step.
    #[test]
    fn the_default_sampler_emits_one_frame_per_outer_step() {
        let steps = 8usize;
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        run(
            Some(crate::DEFAULT_SAMPLER),
            &crate::anima_sigmas(steps),
            cosmos_latent(WIDTH, HEIGHT),
            Some(&hook),
            zero_velocity,
        )
        .unwrap();
        assert_eq!(
            frames_of(&captured),
            (1..=steps as u32)
                .map(|n| (n, steps as u32))
                .collect::<Vec<_>>()
        );
    }

    /// Every emitted frame is a latent-resolution RGB8 image of the running trajectory, on a
    /// non-square render.
    #[test]
    fn emitted_frames_are_vae_latent_resolution_rgb8() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        run(
            None,
            &crate::anima_sigmas(3),
            cosmos_latent(1024, 512),
            Some(&hook),
            zero_velocity,
        )
        .unwrap();

        let frames = candle_gen::lock_recover(&captured);
        assert_eq!(frames.len(), 3);
        for frame in frames.iter() {
            assert_eq!((frame.image.width, frame.image.height), (128, 64));
            assert_eq!(frame.image.pixels.len(), 128 * 64 * 3);
        }
    }

    // --- What the hook is allowed to see -----------------------------------------------------------

    /// The CFG hazard, driven through the real sampler with a predict closure shaped like the render
    /// lane's: `anima_base` / `anima_aesthetic` run the conditional and unconditional DiT forwards
    /// separately *inside* the closure and blend them into one velocity. The unconditional forward's
    /// input is the same latent, and neither its output nor any fused batch ever becomes the running
    /// latent — so the hook can only ever see the single conditional trajectory.
    #[test]
    fn cfg_never_exposes_the_unconditional_half_to_the_preview() {
        let (sink, captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_single_frame_latents(x)
        });

        let guidance = 4.5f64;
        run(
            Some("euler"),
            &crate::anima_sigmas(4),
            cosmos_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                // `pipeline`'s CFG shape: two forwards, one combined velocity, no fused batch.
                let v_cond = x.zeros_like()?;
                let v_uncond = x.ones_like()?;
                Ok((&v_uncond + ((v_cond - &v_uncond)? * guidance)?)?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter()
                .all(|dims| dims == &[1, PREVIEW_LATENT_CHANNELS, 1, 64, 64]),
            "the hook must only ever see the single unfused conditional latent, got {seen:?}"
        );
        assert_eq!(frames_of(&captured).len(), 4);
    }

    // --- Decorative by contract --------------------------------------------------------------------

    /// An inert sink must be byte-identical to no hook at all, and an ACTIVE sink must be too — the
    /// preview reads the latent and never writes it.
    #[test]
    fn an_inert_sink_is_byte_identical_to_an_unhooked_render() {
        let sigmas = crate::anima_sigmas(6);
        let start = Tensor::rand(
            -1f32,
            1f32,
            (1, PREVIEW_LATENT_CHANNELS, 1, 8, 8),
            &Device::Cpu,
        )
        .unwrap();
        let velocity = |x: &Tensor, t: f32| Ok((x * (t as f64 + 0.25))?);
        let bytes = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let bare = run(None, &sigmas, start.clone(), None, velocity).unwrap();

        let inert = PreviewSink::default();
        let inert_hook = hook(&inert);
        assert!(!inert_hook.is_active());
        let hooked = run(None, &sigmas, start.clone(), Some(&inert_hook), velocity).unwrap();
        assert_eq!(
            bytes(&bare),
            bytes(&hooked),
            "an inert preview sink must not perturb a single latent byte"
        );

        let (sink, captured) = collecting_sink();
        let active_hook = hook(&sink);
        let active = run(None, &sigmas, start, Some(&active_hook), velocity).unwrap();
        assert_eq!(bytes(&bare), bytes(&active));
        assert_eq!(candle_gen::lock_recover(&captured).len(), 6);
    }

    /// A projection failure loses its frame and never fails the render. The realistic shape of that
    /// failure here is a trajectory whose latent is not the Cosmos layout the projector accepts.
    #[test]
    fn a_projection_failure_loses_the_frame_and_never_fails_the_render() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        // A rank-4 spatial latent: the shape Krea denoises in, which this projector rejects.
        let start =
            Tensor::zeros((1, PREVIEW_LATENT_CHANNELS, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let out = run(
            None,
            &crate::anima_sigmas(5),
            start,
            Some(&hook),
            zero_velocity,
        )
        .expect("a failing projection must not fail the render");

        assert_eq!(out.dims(), [1, PREVIEW_LATENT_CHANNELS, 8, 8]);
        assert!(
            candle_gen::lock_recover(&captured).is_empty(),
            "no frame may be emitted when every projection fails"
        );
    }

    // --- Route inventory ---------------------------------------------------------------------------

    /// The sampler driver whose call sites this inventory reads. Named without an open paren
    /// everywhere else in this crate's prose, because the scan below is textual.
    const DRIVER: &str = "run_flow_sampler";

    /// `run_flow_sampler`'s argument count, and the 0-based position of its `preview` argument.
    /// Pinned so a signature change — or a scanner mis-split — fails this inventory loudly instead of
    /// quietly shifting which argument is being asserted about.
    const SAMPLER_ARITY: usize = 9;
    const PREVIEW_ARGUMENT: usize = 7;

    /// The test-only attribute whose item is dropped before the scan. Spelled once, and asserted to
    /// leave no survivor behind.
    const TEST_ATTRIBUTE: &str = "#[cfg(test)]";

    /// Rust source with comments, string / char literals, and `#[cfg(test)]` items removed — so a
    /// driver name quoted in prose or in a literal is never read as a call site, a bracket inside one
    /// never moves the scan, and this very module's own test helpers do not read as shipped routes.
    fn code_only(file: &str, source: &str) -> String {
        let chars: Vec<char> = source.chars().collect();
        let mut out = String::with_capacity(source.len());
        let mut i = 0usize;
        // `Some((bracket depth, has opened its top-level block))` while consuming a test-only item.
        let mut skipping: Option<(i32, bool)> = None;
        while i < chars.len() {
            let ch = chars[i];
            if ch == '/' && chars.get(i + 1) == Some(&'/') {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            if ch == '/' && chars.get(i + 1) == Some(&'*') {
                i += 2;
                let mut nesting = 1usize;
                while i < chars.len() && nesting > 0 {
                    if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                        nesting += 1;
                        i += 2;
                    } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                        nesting -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                assert_eq!(nesting, 0, "{file}: unterminated block comment");
                continue;
            }
            if ch == '"' {
                i += 1;
                let mut escaped = false;
                let mut closed = false;
                while i < chars.len() {
                    let c = chars[i];
                    i += 1;
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        closed = true;
                        break;
                    }
                }
                assert!(closed, "{file}: unterminated string literal");
                continue;
            }
            // A `'` opens a char literal only when it closes; otherwise it is a lifetime.
            if ch == '\'' && (chars.get(i + 1) == Some(&'\\') || chars.get(i + 2) == Some(&'\'')) {
                i += 1;
                if chars.get(i) == Some(&'\\') {
                    // The backslash and the character it escapes are both part of the literal, so a
                    // `'\''` must not stop scanning at its own escaped quote; a numeric escape
                    // (`'\u{1F600}'`) then runs on to the real closing quote.
                    i += 2;
                    while i < chars.len() && chars[i] != '\'' {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
                assert_eq!(chars.get(i), Some(&'\''), "{file}: malformed char literal");
                i += 1;
                continue;
            }
            if skipping.is_none() && matches_at(&chars, i, TEST_ATTRIBUTE) {
                i += TEST_ATTRIBUTE.chars().count();
                skipping = Some((0, false));
                continue;
            }
            if let Some((depth, entered)) = skipping.as_mut() {
                match ch {
                    '(' | '[' | '{' => {
                        *depth += 1;
                        if ch == '{' && *depth == 1 {
                            *entered = true;
                        }
                    }
                    ')' | ']' | '}' => {
                        *depth -= 1;
                        assert!(*depth >= 0, "{file}: unbalanced test-only item");
                        if *depth == 0 && *entered {
                            skipping = None;
                        }
                    }
                    // A test-only item is not always a block: the attribute also applies to a `use`,
                    // a single struct field, or an enum variant, which end at `;` or `,`.
                    ';' | ',' if *depth == 0 => skipping = None,
                    _ => {}
                }
                i += 1;
                continue;
            }
            out.push(ch);
            i += 1;
        }
        assert!(skipping.is_none(), "{file}: a test-only item never closed");
        // Belt and braces: this scanner only understands the one exact spelling, so a `cfg` predicate
        // that mentions `test` in any other form must not survive silently — it would put test code
        // back into a scan that reports "no sampler site" for it.
        assert!(
            !out.contains("cfg(test"),
            "{file}: a cfg predicate mentioning `test` survived the strip — teach `code_only` \
             about it rather than scanning test code as shipped code"
        );
        out
    }

    fn matches_at(chars: &[char], at: usize, needle: &str) -> bool {
        needle
            .chars()
            .enumerate()
            .all(|(offset, c)| chars.get(at + offset) == Some(&c))
    }

    /// The top-level, comma-separated arguments of every driver call in `source`, one entry per site.
    ///
    /// The window is bounded by the call's own **bracket balance** and ends at its closing paren, so
    /// it works for a site that passes its predict closure by name — which this crate's does, and
    /// which sc-16950's first-`|` bounding rule could not parse at all. A closure's parameter list is
    /// consumed whole so its commas and pipes are never mistaken for the call's own.
    fn sampler_call_sites(file: &str, source: &str) -> Vec<Vec<String>> {
        let call = format!("{DRIVER}(");
        let code = code_only(file, source);
        let mut sites = Vec::new();
        let mut cursor = 0usize;
        while let Some(at) = code[cursor..].find(&call) {
            let args_start = cursor + at + call.len();
            let site = format!("{file}: {DRIVER} call #{}", sites.len());
            sites.push(call_arguments(&site, &code[args_start..]));
            cursor = args_start;
        }
        sites
    }

    /// The comma-separated top-level arguments of one call, given everything after its open paren.
    fn call_arguments(site: &str, rest: &str) -> Vec<String> {
        let normalize = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = rest.chars().collect();
        let mut args: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut depth = 1usize;
        let mut i = 0usize;

        while i < chars.len() {
            let ch = chars[i];
            i += 1;
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let last = normalize(&current);
                        if !last.is_empty() {
                            args.push(last);
                        }
                        return args;
                    }
                    current.push(ch);
                }
                ',' if depth == 1 => {
                    args.push(normalize(&current));
                    current.clear();
                }
                '|' if depth == 1 => {
                    while i < chars.len() && chars[i] != '|' {
                        i += 1;
                    }
                    assert!(
                        i < chars.len(),
                        "{site} has an unterminated closure parameter list"
                    );
                    i += 1;
                    current.push_str(" <closure> ");
                }
                _ => current.push(ch),
            }
        }
        panic!("{site} is unterminated: no closing paren before end of file")
    }

    /// Anima's **one** shipped render lane passes a preview hook, pinned at the source level.
    ///
    /// All three registered ids — `anima_base`, `anima_aesthetic`, `anima_turbo` — share this single
    /// `crate::pipeline::AnimaPipeline::generate` body and differ only in the DiT weights file, so one
    /// hooked site is the whole family. That is why the catalog's `PREVIEW_ROUTE_IDS` carries three
    /// rows against this crate's one: ids and render lanes are different counts, and neither may be
    /// inferred from the other.
    ///
    /// This is the crate-local half of the epic-16948 guard; `candle-gen-catalog`'s
    /// `preview_advertising` module carries the same count as the family's route inventory and ties it
    /// to the advertised `supports_preview`.
    #[test]
    fn the_render_lane_passes_a_preview_hook() {
        let sites = sampler_call_sites("pipeline.rs", include_str!("pipeline.rs"));
        assert_eq!(
            sites.len(),
            1,
            "expected exactly 1 sampler call site in pipeline.rs, found {}. A new render route \
             must pass a preview hook and be named in this inventory (and in the catalog's).",
            sites.len()
        );
        let args = &sites[0];
        assert_eq!(
            args.len(),
            SAMPLER_ARITY,
            "expected {SAMPLER_ARITY} arguments, parsed {args:?}"
        );
        // Positional, not `contains`: the preview is a specific argument, so this cannot be satisfied
        // by the word appearing anywhere else in the call.
        assert_eq!(
            args[PREVIEW_ARGUMENT].as_str(),
            "Some(&preview)",
            "the render lane does not pass a preview hook: {args:?}"
        );
    }

    /// Every other shipped module drives no sampler at all, so the single inventoried site above is
    /// the whole crate. Pinned as a negative so a future render route added elsewhere cannot slip past
    /// an inventory that only looks at one file.
    #[test]
    fn no_other_shipped_module_drives_a_sampler() {
        for (file, source) in [
            ("lib.rs", include_str!("lib.rs")),
            ("adapt.rs", include_str!("adapt.rs")),
            ("adapters.rs", include_str!("adapters.rs")),
            ("conditioner.rs", include_str!("conditioner.rs")),
            ("config.rs", include_str!("config.rs")),
            ("loader.rs", include_str!("loader.rs")),
            ("nn.rs", include_str!("nn.rs")),
            ("preview.rs", include_str!("preview.rs")),
            ("rope.rs", include_str!("rope.rs")),
            ("text_encoder.rs", include_str!("text_encoder.rs")),
            ("tokenizer.rs", include_str!("tokenizer.rs")),
            ("training.rs", include_str!("training.rs")),
            ("transformer.rs", include_str!("transformer.rs")),
            ("vae.rs", include_str!("vae.rs")),
        ] {
            assert!(
                sampler_call_sites(file, source).is_empty(),
                "{file} drives a sampler but is not in the route inventory"
            );
        }
    }

    /// The file list above must be the crate's **whole** shipped module surface, or a new module
    /// could hold an unhooked render route and the negative pin would never look at it.
    #[test]
    fn the_negative_pin_covers_every_shipped_module() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut on_disk: Vec<String> = std::fs::read_dir(&src)
            .expect("read src/")
            .map(|entry| entry.expect("dir entry").path())
            .filter(|path| path.extension().is_some_and(|e| e == "rs"))
            .map(|path| {
                path.file_name()
                    .expect("file name")
                    .to_string_lossy()
                    .into()
            })
            .collect();
        on_disk.sort();
        assert_eq!(
            on_disk,
            [
                "adapt.rs",
                "adapters.rs",
                "conditioner.rs",
                "config.rs",
                "lib.rs",
                "loader.rs",
                "nn.rs",
                "pipeline.rs",
                "preview.rs",
                "rope.rs",
                "text_encoder.rs",
                "tokenizer.rs",
                "training.rs",
                "transformer.rs",
                "vae.rs",
            ],
            "a module joined or left src/ — add it to the route inventory or to the negative pin"
        );
    }
}

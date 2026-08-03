//! Z-Image's per-step latent preview seam (epic 16948, sc-16957; the MLX original is epic 16624 /
//! `mlx-gen-z-image/src/preview.rs`).
//!
//! Schedule numbering, multi-eval dedup and the swallow-on-failure contract live in
//! [`candle_gen::preview`], shared by every candle family (sc-16949). This module owns two things: the
//! **reused** Z-Image 16-channel fit, and the layout adaptation from Z-Image's 5-D running latent to
//! the `[1, C, h, w]` contract the shared projection takes.
//!
//! ## The latent shape at the emission point — verified, not assumed
//!
//! Epic 16948's scoping flagged Z-Image as one of the two families that "project post-unpack" on MLX
//! and asked this story to verify rather than port or omit an unpack step by assumption. It is
//! verified, and the answer is that **candle Z-Image is not packed at all**. There is no
//! `unpack_latents` here and none is needed: the patchify/unpatchify pair lives entirely *inside*
//! `candle_transformers::models::z_image::transformer`'s forward, so the sampler's running latent
//! never enters the packed token space.
//!
//! | stage | shape | projectable? |
//! | --- | --- | --- |
//! | `common::seed_noise` | `[1, 16, H/8, W/8]` | not what the sampler sees |
//! | after `z_image::preprocess::prepare_inputs` (`unsqueeze(2)`) | `[1, 16, 1, H/8, W/8]` | no — rank 5 |
//! | after dropping the frame axis | `[1, 16, H/8, W/8]` | yes — the fitted space |
//!
//! So the recovery is a single squeeze of the singleton **frame** axis — the very squeeze
//! `crate::common::decode` already applies before handing the VAE its NCHW latent, spelled with the
//! same `crate::common::LATENT_FRAME_AXIS` constant (private, hence no link) so the preview and the
//! decode cannot come to
//! disagree about which axis it is. Because the rest of the geometry travels inside the latent, the
//! projector needs no `width`/`height` argument: hook geometry and latent geometry are not merely bound
//! to one source, there is only one source to bind to.
//!
//! MLX denoises the same trajectory in `[16, 1, h, w]` (no batch axis) and reaches the fitted space
//! through its own `pipeline::unpack_latents`; candle's leading batch axis is the only difference, and
//! it is why this seam could not be written by porting the MLX one.
//!
//! ## What the hook sees on each route
//!
//! Both **registered** routes (`z_image_turbo`, `z_image`) and the **base** halves of the name-driven
//! Fun-ControlNet provider drive [`candle_gen::run_flow_sampler`], so they opt in with a projector
//! closure. The three **distilled Turbo** lanes — control-staged, control-resident, and the img2img /
//! masked-edit provider — own bespoke flow-match Euler loops and emit by calling
//! [`candle_gen::preview::emit_preview_at`] directly against a crate-private `bespoke_counter`, at the
//! top of each iteration, so every lane previews the same thing: the running latent *entering* step
//! `k`, which is exactly where the shared drivers emit.
//!
//! * **CFG never reaches the preview.** Turbo is guidance-distilled (one forward, no negative prompt).
//!   The base and base-control lanes do run true CFG, but as *two separate DiT forwards inside the
//!   predict closure* blended into one velocity (`v_uncond + g·(v_cond − v_uncond)`); no fused
//!   `[2, …]` batch is ever the running latent, so there is no unconditional half to project.
//! * **The control context never reaches the preview.** `crate::control` VAE-encodes the pose skeleton
//!   once into a constant 33-channel context and injects it *inside* `forward_control`. It is a closure
//!   capture, never part of the tensor the loop integrates — which is this story's "control routes
//!   project target tokens only" criterion, closed structurally rather than by a guard.
//! * **img2img sees only the target.** `crate::edit` blends the VAE-encoded source into `x_t` *before*
//!   the loop starts; from the first emission onward there is one trajectory and it is the target's.
//!
//! ## The σ convention: this family needs no correction
//!
//! `candle_gen::run_flow_sampler` integrates a
//! [`candle_gen::gen_core::sampling::FlowModelSampling`], whose `input_scale` is exactly `1.0` at every
//! σ, so the running latent already *is* the tensor the fit was measured against and the σ-less
//! [`candle_gen::preview::PreviewHook::new`] constructor is the correct one. The three bespoke Turbo
//! loops scale nothing either — they hand `latents` straight to the DiT. sc-16954 found the **opposite**
//! for the discrete ε cohort (SDXL/Kolors denoise in k-diffusion VE σ-space and must apply
//! `1/√(σ²+1)` first, or 89.4% of the first frame clips to the rails); `tests/preview_real_weights.rs`
//! measures this family's rail-clipped fraction rather than asserting the difference in prose.
//!
//! ## The fit is reused, not refitted — and its latent space is FLUX.1's
//!
//! `RGB_FACTORS` / `RGB_BIAS` are the epic-16624 constants transcribed verbatim from
//! `mlx-gen-z-image/src/preview.rs`. They are least-squares numbers over a VAE *latent space* with no
//! backend in them; candle reuses them and deliberately ships **no producer** of its own —
//! `mlx-gen-z-image/tests/fit_preview_rgb.rs` remains the only way they are re-derived.
//!
//! The reuse is grounded in the bytes candle actually loads, and doing that turned up a finding this
//! story was asked to settle explicitly. **Z-Image's VAE is FLUX.1-dev's, byte for byte.** Both
//! `Tongyi-MAI/Z-Image-Turbo` and `Tongyi-MAI/Z-Image` publish a `vae/diffusion_pytorch_model.safetensors`
//! whose SHA-256 is `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`
//! (167,666,902 bytes, 244 tensors) — the *same file* `black-forest-labs/FLUX.1-dev` ships and the one
//! sc-16956 pinned as the FLUX.1 diffusers container. Their `vae/config.json` even names its origin:
//! `"_name_or_path": "flux-dev"`. The fit donor `SceneWorks/z-image-turbo-mlx` @
//! `bb2bc9893b3c49ae96c813350775f791a2e8bc80` `bf16/vae/model.safetensors` (SHA-256
//! `0fbab8b6…3810`, 167,666,968 bytes) is that same file re-containered: all 244 tensors byte-identical,
//! the 66-byte size difference being the safetensors header's `__metadata__` alone.
//!
//! So epic 16624 committed **two** fits over **one** latent space — this one and
//! `mlx-gen-flux/src/preview.rs`'s — measured on different render sets. That is a duplication, not a
//! contradiction: both are valid OLS solutions and either would preview either family. This module
//! keeps the Z-Image one because it is the one the story names, because it was measured on Z-Image
//! renders, and because it keeps candle's Z-Image previews byte-comparable with the MLX lane's.
//! Collapsing the two is a cross-engine decision that would change MLX preview bytes, so it is recorded
//! as a follow-up rather than taken here.
//! `tests/preview_real_weights.rs` re-derives every claim above per snapshot; the full record is
//! `docs/migration/evidence/sc-16957-z-image-candle-preview.md`.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::{PreviewCounter, PreviewHook};
use candle_gen::{CandleError, Result};

use crate::common::LATENT_FRAME_AXIS;

/// Ordinary-least-squares map from the native Z-Image VAE latent to latent-resolution RGB (row *i* maps
/// latent channel *i* to `[r, g, b]`), with [`RGB_BIAS`] the intercept.
///
/// **Reused verbatim from `mlx-gen-z-image/src/preview.rs`, not refitted.** Fit on four diverse
/// real-weight Z-Image-Turbo bf16 renders and measured on two disjoint prompt/seed holdouts, all 256²
/// with eight static-shift-3 flow-Euler steps, against 8×8-average-pooled native VAE decodes. Fit R²
/// `(R,G,B) = (0.98367, 0.97883, 0.98092)`, overall `0.98133`; holdout R²
/// `(0.94679, 0.96390, 0.89464)`, overall `0.92827`.
///
/// Donor snapshot: `SceneWorks/z-image-turbo-mlx` revision
/// `bb2bc9893b3c49ae96c813350775f791a2e8bc80`, `bf16` tier, `vae/model.safetensors`, 167,666,968 bytes,
/// SHA-256 `0fbab8b661f6ee6af81c88a6eb1501ec1f7b4b8fe4ad29803507ebe0cf863810` — whose 244 tensors are
/// byte-identical to the FLUX.1-dev diffusers container (see the module docs).
///
/// Refit — in `mlx-gen-z-image`, never here — whenever the Z-Image VAE lineage changes.
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

/// The fit's intercept — the near-neutral grey a fully-zero latent projects to. Reused with
/// [`RGB_FACTORS`].
const RGB_BIAS: [f32; 3] = [0.502_150_24, 0.483_383_92, 0.458_297_43];

/// The latent channel count the fit is defined over, derived from the committed factor table's own
/// length so nothing in this crate can drift from it by restating a number.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// The fit is the SIXTEEN-channel one, and it is defined over the channel count the rest of the crate
/// already denoises in. Compile-time, because a runtime row over constants proves nothing a `const`
/// assertion does not prove earlier.
const _: () = assert!(
    PREVIEW_LATENT_CHANNELS == 16 && PREVIEW_LATENT_CHANNELS == crate::common::LATENT_CHANNELS
);

/// Project Z-Image's 5-D running latent `[1, 16, 1, h, w]` to a latent-resolution RGB8 preview.
///
/// The singleton frame axis is dropped first — the same squeeze `crate::common::decode` applies before
/// the VAE — and the reused fit is then applied by [`candle_gen::preview::project_latents`].
///
/// Errors on any other layout, including the already-squeezed `[1, 16, h, w]`: a rank-4 latent is not
/// something this family's sampler can produce, and silently accepting one would hide a real regression
/// in the denoise shape. The caller's frame is then lost and swallowed by
/// [`candle_gen::preview::emit_preview`], which is the intended decorative-failure behaviour.
pub fn project_frame_latents(latents: &Tensor) -> Result<Image> {
    let spatial = drop_frame_axis(latents)?;
    candle_gen::preview::project_latents(&spatial, &RGB_FACTORS, RGB_BIAS)
}

/// `[1, C, 1, h, w]` → `[1, C, h, w]`, rejecting anything that is not one Z-Image still frame in the
/// fitted channel space.
///
/// Written as a checked squeeze rather than a bare one, because candle's `squeeze` is a **no-op** on an
/// axis whose extent is not 1: a `[1, 16, T>1, h, w]` latent would pass straight through it and only
/// fail later, in the shared projection, with a message about a contract this family never violates.
fn drop_frame_axis(latents: &Tensor) -> Result<Tensor> {
    let dims = latents.dims();
    if dims.len() != 5
        || dims[0] != 1
        || dims[1] != PREVIEW_LATENT_CHANNELS
        || dims[LATENT_FRAME_AXIS] != 1
    {
        return Err(CandleError::Msg(format!(
            "z-image preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, 1, h, w], got \
             {dims:?}"
        )));
    }
    Ok(latents.squeeze(LATENT_FRAME_AXIS)?)
}

/// The preview hook every shared-driver Z-Image lane hands to [`candle_gen::run_flow_sampler`]: a
/// projector closure over [`project_frame_latents`]. The driver owns frame numbering, multi-eval dedup
/// and the swallow-on-failure contract (sc-16949), so no route restructures its loop.
///
/// Build it **per image**: a batched request runs one driver call per seed and each call must start a
/// fresh trajectory at frame 1. (The driver builds its own counter per call, so this is a property of
/// the call rather than of the hook — building the hook alongside the call keeps the two impossible to
/// separate.)
pub(crate) fn hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project_frame_latents)
}

/// The frame counter a **bespoke** Z-Image loop numbers against: the three distilled-Turbo lanes
/// (control staged, control resident, img2img/edit) walk `for step in 0..n` with the scheduler's own
/// Euler step rather than driving a shared sampler, so there is no driver to build the counter for
/// them.
///
/// Keyed on the step **index** rather than on σ, which is what those loops actually iterate — they read
/// `current_timestep_normalized()` / `scheduler.timesteps[i]`, never a σ they could hand back. `steps`
/// is the count the loop reports as `Progress::Step { total }`, so the preview's `total` and the
/// progress bar's cannot disagree; on the reduced img2img schedule that is `steps − start`, and the
/// caller passes the loop-local index (`step_i − start`) to match.
pub(crate) fn bespoke_counter(steps: usize) -> PreviewCounter {
    PreviewCounter::with_steps(steps)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device};
    use candle_gen::gen_core::sampling::TimestepConvention;
    use candle_gen::gen_core::{CancelFlag, PreviewFrame, Progress};

    use super::*;

    /// A small but genuinely Z-Image-shaped render: 256² is the advertised minimum, giving a 32×32
    /// spatial latent under the 8× VAE compression.
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;

    fn frame_latent(width: u32, height: u32) -> Tensor {
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

    // --- The reused fit ---------------------------------------------------------------------------

    /// The fit is **reused**, not refitted: these are the epic-16624 constants transcribed verbatim
    /// from `mlx-gen-z-image/src/preview.rs`. Pinned as literals so an edit to either copy fails rather
    /// than silently forking one latent space into two colour maps.
    #[test]
    fn committed_fit_matches_the_mlx_source_block() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(
            RGB_FACTORS[0],
            [-0.013_211_725, 0.020_633_436, 0.050_329_126]
        );
        assert_eq!(RGB_FACTORS[6], [0.017_442_052, 0.055_714_785, 0.043_591_47]);
        assert_eq!(
            RGB_FACTORS[9],
            [0.066_185_762, 0.045_447_53, -0.031_686_028]
        );
        assert_eq!(
            RGB_FACTORS[13],
            [-0.072_754_92, -0.010_183_617, -0.074_263_78]
        );
        assert_eq!(
            RGB_FACTORS[15],
            [-0.061_362_23, -0.036_242_01, -0.029_276_784]
        );
        assert_eq!(RGB_BIAS, [0.502_150_24, 0.483_383_92, 0.458_297_43]);
        assert!(RGB_FACTORS.iter().flatten().all(|v| v.is_finite()));
        assert!(RGB_BIAS.iter().all(|v| v.is_finite()));
    }

    /// A zero latent projects to the fit's intercept — the one place the committed bias is directly
    /// observable, so a typo in [`RGB_BIAS`] cannot pass.
    #[test]
    fn a_zero_latent_projects_to_the_fit_intercept() {
        let image = project_frame_latents(&frame_latent(WIDTH, HEIGHT)).unwrap();
        assert_eq!((image.width, image.height), (32, 32));
        // 0.50215024·255 = 128.0, 0.48338392·255 = 123.3, 0.45829743·255 = 116.9
        assert_eq!(image.pixels[..3], [128, 123, 117]);
    }

    /// The frame is at **VAE-latent** resolution `W/8 × H/8`, matching what the decode tail feeds the
    /// VAE, and it carries one RGB triplet per latent cell.
    #[test]
    fn projection_is_latent_resolution() {
        for (width, height) in [(256u32, 256u32), (1024, 1024), (1536, 1024)] {
            let image = project_frame_latents(&frame_latent(width, height)).unwrap();
            assert_eq!((image.width, image.height), (width / 8, height / 8));
            assert_eq!(
                image.pixels.len(),
                (width / 8) as usize * (height / 8) as usize * 3
            );
        }
    }

    /// The layout gate, stated as the ways it can be wrong. The already-squeezed rank-4 latent is
    /// included deliberately: this family's sampler cannot produce one, so accepting it would hide a
    /// real change in the denoise shape rather than tolerate a harmless variation. So is the genuinely
    /// temporal latent, which a bare `squeeze` would pass straight through.
    #[test]
    fn projection_rejects_every_non_z_image_layout() {
        let shapes: &[&[usize]] = &[
            &[1, 16, 32, 32],    // already squeezed — not a shape this sampler produces
            &[1, 16, 2, 32, 32], // temporal: `squeeze` is a no-op on a non-1 axis
            &[1, 4, 1, 32, 32],  // the SDXL channel space
            &[1, 32, 1, 32, 32], // the FLUX.2 channel space
            &[2, 16, 1, 32, 32], // batched — a fused CFG pair would land here
            &[16, 1, 32, 32],    // the MLX layout, which has no leading batch axis
        ];
        for shape in shapes {
            let latents = Tensor::zeros(*shape, DType::F32, &Device::Cpu).unwrap();
            let error = project_frame_latents(&latents).unwrap_err();
            assert!(
                error.to_string().contains("[1, 16, 1, h, w]"),
                "{shape:?} must be rejected by the Z-Image layout gate, got: {error}"
            );
        }
    }

    /// bf16 is the candle GPU denoise dtype — Z-Image loads at bf16 regardless of the CPU default — and
    /// a bf16 matmul against f32 constants is a hard panic on candle. The shared projection casts up
    /// front; this row is what proves this seam inherits that rather than the latent's dtype.
    #[test]
    fn projection_accepts_a_low_precision_latent() {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let latents = frame_latent(WIDTH, HEIGHT).to_dtype(dtype).unwrap();
            let image = project_frame_latents(&latents)
                .unwrap_or_else(|e| panic!("{dtype:?} latent failed to project: {e}"));
            assert_eq!(image.pixels[..3], [128, 123, 117]);
        }
    }

    /// The recovery is the decode's own squeeze, not a second implementation of it: dropping the frame
    /// axis with `crate::common::LATENT_FRAME_AXIS` and projecting is the same picture as projecting
    /// the tensor `crate::common::decode` hands the VAE. Driven over a non-trivial latent so a squeeze
    /// of the wrong axis could not agree.
    #[test]
    fn the_recovery_is_the_one_the_decode_uses() {
        let latents = Tensor::rand(-2f32, 2f32, (1, 16, 1, 6, 10), &Device::Cpu).unwrap();
        let via_preview = project_frame_latents(&latents).unwrap();

        let spatial = latents.squeeze(LATENT_FRAME_AXIS).unwrap();
        assert_eq!(spatial.dims(), [1, 16, 6, 10]);
        let via_decode_shape =
            candle_gen::preview::project_latents(&spatial, &RGB_FACTORS, RGB_BIAS).unwrap();
        assert_eq!(via_preview.pixels, via_decode_shape.pixels);
        assert_eq!((via_preview.width, via_preview.height), (10, 6));

        // The squeeze is load-bearing: reading the same values as if the frame axis were the channel
        // axis is a different picture. Pinned so a refactor that squeezed axis 1 would be red.
        let wrong = latents.squeeze(1).unwrap();
        assert_eq!(
            wrong.dims(),
            [1, 16, 1, 6, 10],
            "squeeze on a non-1 axis is a no-op — which is exactly why the gate is a checked squeeze"
        );
    }

    // --- Driving the real sampler ------------------------------------------------------------------

    /// A velocity of exactly zero: the flow-Euler step leaves the latent untouched, so the sampler's
    /// output is a pure function of its input and any byte difference is the wiring's.
    fn zero_velocity(x: &Tensor, _t: f32) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Drive the real flow sampler over `sigmas`, with the same driver, convention and argument order
    /// every shared-driver Z-Image lane uses (`OneMinusSigma`, the Z-Image conditioning convention).
    fn run(
        sampler: Option<&str>,
        sigmas: &[f32],
        start: Tensor,
        preview: Option<&PreviewHook<'_>>,
        predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::OneMinusSigma,
            sigmas,
            start,
            16957,
            &CancelFlag::new(),
            &mut |_: Progress| {},
            preview,
            predict,
        )
    }

    /// Z-Image-Turbo's own schedule at this render size — the array the Turbo routes resolve, not a
    /// synthetic ramp.
    fn turbo_sigmas(steps: usize) -> Vec<f32> {
        use candle_transformers::models::z_image::scheduler::{
            FlowMatchEulerDiscreteScheduler, SchedulerConfig,
        };
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        // The Turbo construction: `Some(mu)`, which under `use_dynamic_shifting=false` applies NO
        // shift, so the sigmas stay linear (see `crate::pipeline::render`).
        scheduler.set_timesteps(steps, Some(0.0));
        let native: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();
        candle_gen::resolve_flow_schedule(None, 0.0, steps, &native)
    }

    /// Euler evaluates once per step: an N-step render emits exactly N frames, 1..=N, each carrying
    /// `total == N`.
    #[test]
    fn euler_emits_exactly_one_numbered_frame_per_step() {
        for steps in [1usize, 4, 8, 50] {
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink);
            run(
                None,
                &turbo_sigmas(steps),
                frame_latent(WIDTH, HEIGHT),
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
    /// evaluation count is asserted to exceed the step count first, so a solver that silently fell back
    /// to Euler could not make this pass vacuously.
    #[test]
    fn multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step() {
        for name in ["heun", "dpmpp_sde"] {
            let steps = 6usize;
            let evaluations = std::cell::Cell::new(0usize);
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink);
            run(
                Some(name),
                &turbo_sigmas(steps),
                frame_latent(WIDTH, HEIGHT),
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
    /// Driven with ONE hook reused across the calls deliberately: that is the shape that would break if
    /// numbering ever moved out of the driver and into the hook.
    #[test]
    fn each_image_of_a_batch_numbers_its_own_trajectory_from_one() {
        let steps = 4usize;
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        for _ in 0..3 {
            run(
                None,
                &turbo_sigmas(steps),
                frame_latent(WIDTH, HEIGHT),
                Some(&hook),
                zero_velocity,
            )
            .unwrap();
        }
        let one: Vec<_> = (1..=steps as u32).map(|n| (n, steps as u32)).collect();
        assert_eq!(
            frames_of(&captured),
            [one.clone(), one.clone(), one].concat(),
            "each image in a batch must emit its own 1..=N run"
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
            &turbo_sigmas(3),
            frame_latent(1024, 512),
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

    // --- The bespoke-loop counter ------------------------------------------------------------------

    /// The three distilled-Turbo lanes number their own frames. The counter must produce the same
    /// `1..=N / total N` shape the shared driver does, dedup a repeated index, and clamp an over-run —
    /// so a bespoke loop and a driven one are indistinguishable to the consumer.
    #[test]
    fn the_bespoke_counter_numbers_like_the_driver() {
        let counter = bespoke_counter(4);
        assert_eq!(counter.total(), 4);
        assert_eq!(
            (0..4)
                .filter_map(|i| counter.next_step(i))
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(counter.next_step(3), None, "a repeated index must not emit");
        assert_eq!(counter.next_step(9), None, "an over-run index must clamp");
    }

    /// The reduced img2img schedule: `crate::edit` runs `start..steps` and reports
    /// `total = steps − start`, so the counter is built over that difference and fed the loop-local
    /// index. Pinned because feeding it the ABSOLUTE `step_i` on a strength-shortened run would number
    /// the first frame `start + 1` and emit nothing at all once `start ≥ total`.
    #[test]
    fn the_bespoke_counter_follows_a_reduced_img2img_schedule() {
        let (steps, start) = (4usize, 2usize);
        let total = steps - start;
        let counter = bespoke_counter(total);
        assert_eq!(counter.total(), 2);
        let numbered: Vec<_> = (start..steps)
            .filter_map(|step_i| counter.next_step(step_i - start))
            .collect();
        assert_eq!(numbered, vec![1, 2]);

        // The bug this row exists for: absolute indices against the same counter.
        let naive = bespoke_counter(total);
        let wrong: Vec<_> = (start..steps).filter_map(|i| naive.next_step(i)).collect();
        assert_ne!(wrong, numbered);
    }

    /// A bespoke loop's emission, end to end: the counter plus the shared emitter reproduce the
    /// driver's contract, including the swallow-on-failure behaviour that keeps a preview decorative.
    #[test]
    fn a_bespoke_loop_emits_one_frame_per_step_and_swallows_failures() {
        let (sink, captured) = collecting_sink();
        let counter = bespoke_counter(3);
        let good = frame_latent(WIDTH, HEIGHT);
        // A rank-4 latent: the shape this projector rejects, standing in for a denoise regression.
        let bad = Tensor::zeros((1, 16, 32, 32), DType::F32, &Device::Cpu).unwrap();

        for (step, latent) in [(0usize, &good), (1, &bad), (2, &good)] {
            candle_gen::preview::emit_preview_at(&sink, &counter, step, || {
                project_frame_latents(latent)
            });
        }

        assert_eq!(
            frames_of(&captured),
            vec![(1, 3), (3, 3)],
            "the failed projection must lose its frame and keep its schedule position"
        );
    }

    // --- What the hook is allowed to see -----------------------------------------------------------

    /// The CFG hazard, driven through the real sampler with a predict closure shaped like the base
    /// lanes': two separate DiT forwards blended into one velocity, never a fused batch. The
    /// unconditional half is never the running latent, so it can never be projected.
    #[test]
    fn cfg_never_exposes_the_unconditional_half_to_the_preview() {
        let (sink, captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_frame_latents(x)
        });

        let guidance = 4.0f64;
        run(
            None,
            &turbo_sigmas(4),
            frame_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                // `render_base`'s CFG shape, with the Z-Image sign convention applied per branch.
                let v_cond = x.zeros_like()?.neg()?;
                let v_uncond = x.ones_like()?.neg()?;
                Ok((&v_uncond + ((v_cond - &v_uncond)? * guidance)?)?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter()
                .all(|dims| dims == &[1, PREVIEW_LATENT_CHANNELS, 1, 32, 32]),
            "the hook must only ever see the single unfused conditional latent, got {seen:?}"
        );
        assert_eq!(frames_of(&captured).len(), 4);
    }

    /// The control hazard in this family's shape: `crate::control` concatenates a constant 33-channel
    /// context onto the DiT input *inside* `forward_control` and returns a 16-channel velocity. Driven
    /// through the real sampler with a closure that does exactly that — the shape that WOULD leak if a
    /// route handed the joint tensor to the loop.
    #[test]
    fn the_control_context_never_reaches_the_previewed_latent() {
        let (sink, captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project_frame_latents(x)
        });

        // 16 control latent + 1 zero mask + 16 zero inpaint = the 33-channel Fun-ControlNet context.
        let context = Tensor::ones((1, 33, 1, 32, 32), DType::F32, &Device::Cpu).unwrap();
        run(
            None,
            &turbo_sigmas(4),
            frame_latent(WIDTH, HEIGHT),
            Some(&hook),
            |x, _t| {
                let joint = Tensor::cat(&[x, &context], 1)?;
                assert_eq!(joint.dims()[1], PREVIEW_LATENT_CHANNELS + 33);
                Ok(joint
                    .narrow(1, 0, PREVIEW_LATENT_CHANNELS)?
                    .zeros_like()?
                    .neg()?)
            },
        )
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter()
                .all(|dims| dims == &[1, PREVIEW_LATENT_CHANNELS, 1, 32, 32]),
            "the hook must never see the control context, got {seen:?}"
        );
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
        let sigmas = turbo_sigmas(6);
        let start = Tensor::rand(-1f32, 1f32, (1, 16, 1, 8, 8), &Device::Cpu).unwrap();
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
    /// failure here is a trajectory whose latent is not the 5-D layout the projector accepts.
    #[test]
    fn a_projection_failure_loses_the_frame_and_never_fails_the_render() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        // The rank-4 spatial latent Krea denoises in, which this projector rejects.
        let start = Tensor::zeros((1, 16, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let out = run(None, &turbo_sigmas(5), start, Some(&hook), zero_velocity)
            .expect("a failing projection must not fail the render");

        assert_eq!(out.dims(), [1, 16, 8, 8]);
        assert!(
            candle_gen::lock_recover(&captured).is_empty(),
            "no frame may be emitted when every projection fails"
        );
    }

    // --- Route inventory ---------------------------------------------------------------------------

    /// The shared sampler driver whose call sites this inventory reads. Named without an open paren
    /// everywhere else in this crate's prose, because the scan below is textual.
    const DRIVER: &str = "run_flow_sampler";

    /// The direct emission call a **bespoke** Z-Image loop makes. Same textual-scan caveat.
    const EMIT: &str = "emit_preview_at(";

    /// `run_flow_sampler`'s argument count, and the 0-based position of its `preview` argument. Pinned
    /// so a signature change — or a scanner mis-split — fails this inventory loudly instead of quietly
    /// shifting which argument is being asserted about.
    const SAMPLER_ARITY: usize = 9;
    const PREVIEW_ARGUMENT: usize = 7;

    /// `emit_preview_at`'s argument count and the 0-based position of its **sink** argument.
    const EMIT_ARITY: usize = 4;
    const SINK_ARGUMENT: usize = 0;

    /// The test-only attribute whose item is dropped before the scan. Spelled once, and asserted to
    /// leave no survivor behind.
    const TEST_ATTRIBUTE: &str = "#[cfg(test)]";

    /// Rust source with comments, string / char literals, and `#[cfg(test)]` items removed — so a
    /// driver name quoted in prose or in a literal is never read as a call site, a bracket inside one
    /// never moves the scan, and this very module's own test helpers do not read as shipped routes.
    ///
    /// Ported from `candle-gen-anima/src/preview.rs`; `candle-gen-catalog`'s `preview_advertising`
    /// module carries the hardened cross-crate version, which additionally follows the shipped module
    /// tree so an out-of-line `#[cfg(test)] mod` file (this crate has four) is never read at all.
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
            "{file}: a cfg predicate mentioning `test` survived the strip — teach `code_only` about \
             it rather than scanning test code as shipped code"
        );
        out
    }

    fn matches_at(chars: &[char], at: usize, needle: &str) -> bool {
        needle
            .chars()
            .enumerate()
            .all(|(offset, c)| chars.get(at + offset) == Some(&c))
    }

    /// The top-level, comma-separated arguments of every `call` in `source`, one entry per site.
    ///
    /// The window is bounded by the call's own **bracket balance** and ends at its closing paren, so it
    /// works for a site whose last argument is an inline closure and for one that passes a named
    /// function. A closure's parameter list is consumed whole so its commas and pipes are never
    /// mistaken for the call's own.
    fn call_sites(file: &str, call: &str, source: &str) -> Vec<Vec<String>> {
        let code = code_only(file, source);
        let mut sites = Vec::new();
        let mut cursor = 0usize;
        while let Some(at) = code[cursor..].find(call) {
            let args_start = cursor + at + call.len();
            let site = format!("{file}: {call} call #{}", sites.len());
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

    /// Every shipped Z-Image render lane emits, pinned at the source level, per file and per site.
    ///
    /// Eleven lanes, three files, two wiring layers — the widest inventory in this epic so far:
    ///
    /// * `pipeline.rs` — **four** hooked driver sites: the Turbo and base resident routes (`render`,
    ///   `render_base`) that the two registered descriptors reach, and their staged-residency twins
    ///   (`denoise_sequential`, `denoise_base_sequential`) that a `stage_residency` request reaches
    ///   instead. txt2img and img2img are the same site — img2img only changes the start step.
    /// * `control.rs` — **two** hooked driver sites (the base-mode staged and resident control lanes)
    ///   plus **two** direct emissions (the distilled Turbo lanes, which own bespoke Euler loops).
    /// * `edit.rs` — **one** direct emission, the img2img / masked-edit provider's bespoke loop.
    ///
    /// `training.rs` is the deliberate omission and is pinned as such below.
    ///
    /// This is the crate-local half of the epic-16948 guard; `candle-gen-catalog`'s
    /// `preview_advertising` module carries the same counts as the family's route inventory and ties
    /// them to the advertised `supports_preview`.
    #[test]
    fn every_shipped_render_lane_emits_a_preview() {
        for (file, source, hooked, direct) in [
            ("pipeline.rs", include_str!("pipeline.rs"), 4usize, 0usize),
            ("control.rs", include_str!("control.rs"), 2, 2),
            ("edit.rs", include_str!("edit.rs"), 0, 1),
        ] {
            let sites = call_sites(file, &format!("{DRIVER}("), source);
            assert_eq!(
                sites.len(),
                hooked,
                "{file}: expected exactly {hooked} sampler call sites, found {}. A new render route \
                 must pass a preview hook and be named in this inventory (and in the catalog's).",
                sites.len()
            );
            for (index, args) in sites.iter().enumerate() {
                assert_eq!(
                    args.len(),
                    SAMPLER_ARITY,
                    "{file}: {DRIVER} #{index} expected {SAMPLER_ARITY} arguments, parsed {args:?}"
                );
                // Positional, not `contains`: the preview is a specific argument, so this cannot be
                // satisfied by the word appearing anywhere else in the call.
                assert_eq!(
                    args[PREVIEW_ARGUMENT].as_str(),
                    "Some(&preview)",
                    "{file}: {DRIVER} #{index} does not pass a preview hook: {args:?}"
                );
            }

            let emissions = call_sites(file, EMIT, source);
            assert_eq!(
                emissions.len(),
                direct,
                "{file}: expected exactly {direct} direct emission calls, found {}",
                emissions.len()
            );
            for (index, args) in emissions.iter().enumerate() {
                assert_eq!(
                    args.len(),
                    EMIT_ARITY,
                    "{file}: {EMIT} #{index} expected {EMIT_ARITY} arguments, parsed {args:?}"
                );
                // The sink is the REQUEST's, not a local: a bespoke loop that emitted against a fresh
                // `PreviewSink::default()` would be inert and would still count as a direct emission
                // in the catalog's tally, so the crate-local half is what pins which sink it is.
                assert_eq!(
                    args[SINK_ARGUMENT].as_str(),
                    "&req.preview",
                    "{file}: {EMIT} #{index} must emit against the request's own sink: {args:?}"
                );
            }
        }
    }

    /// **The caller inventory.** sc-16955 found that a site-level assertion cannot see a caller that
    /// stops forwarding its sink — Lens has one sampler site and three callers, only two of which have
    /// a `PreviewSink`. Z-Image is structurally immune to that shape, and this row is what pins the
    /// structure rather than asserting the immunity in prose.
    ///
    /// No Z-Image lane takes a preview hook (or a sink) as a **parameter**. Every one of the seven
    /// emitting lanes takes the whole request — `&GenerationRequest` in `pipeline.rs`,
    /// `&ZImageControlRequest` in `control.rs`, `&ZImageEditRequest` in `edit.rs` — and reads
    /// `req.preview` at the site itself. A caller therefore has nothing to drop: forwarding the request
    /// is forwarding the sink, and a caller that did not forward the request could not call the lane at
    /// all.
    ///
    /// Asserted as "every hook construction and every direct emission names `req.preview`", which is
    /// exactly the property that makes the site count sufficient here.
    #[test]
    fn every_emitting_lane_reads_the_sink_off_its_own_request() {
        let mut constructions = 0usize;
        for (file, source) in [
            ("pipeline.rs", include_str!("pipeline.rs")),
            ("control.rs", include_str!("control.rs")),
            ("edit.rs", include_str!("edit.rs")),
        ] {
            let code = code_only(file, source);
            // Every hook in this crate is built by `crate::preview::hook(&req.preview)` — one
            // expression, so a lane cannot acquire a hook from anywhere else without failing here.
            for args in call_sites(file, "preview::hook(", source) {
                assert_eq!(
                    args,
                    vec!["&req.preview".to_string()],
                    "{file}: a preview hook must be built from the request's own sink, got {args:?}"
                );
                constructions += 1;
            }
            // And no lane may take a sink or a hook as a parameter, which is the shape that would
            // reintroduce the Lens hazard.
            for needle in [": &PreviewHook", ": &PreviewSink", "PreviewHook<'_>) ->"] {
                assert!(
                    !code.contains(needle),
                    "{file} declares a preview parameter ({needle:?}) — the caller inventory in this \
                     row assumes every lane reads `req.preview` at the site, so a parameter would \
                     need a per-caller assertion instead"
                );
            }
        }
        assert_eq!(
            constructions, 6,
            "expected 6 hook constructions (4 in pipeline.rs, 2 in control.rs)"
        );
    }

    /// The trainer's periodic sample render is the crate's one **deliberately dark** sampler site.
    ///
    /// It drives the sampler from a synthetic request that carries no `PreviewSink` — its result is
    /// delivered as a finished `TrainingProgress::Sample` image, not as a live denoise stream — so it
    /// passes `None` on purpose, the same decision sc-16950 recorded for Krea's trainer and sc-16954
    /// for SDXL's. Pinned positively (the argument IS `None`) rather than by omission, and declared as
    /// a `DarkSite` in the catalog's inventory too.
    #[test]
    fn the_trainer_sample_render_is_deliberately_dark() {
        let sites = call_sites(
            "training.rs",
            &format!("{DRIVER}("),
            include_str!("training.rs"),
        );
        assert_eq!(sites.len(), 1, "training.rs must drive exactly one sampler");
        assert_eq!(sites[0].len(), SAMPLER_ARITY);
        assert_eq!(
            sites[0][PREVIEW_ARGUMENT].as_str(),
            "None",
            "the trainer's sample render must stay dark; if it ever emits, remove its DarkSite row \
             from candle-gen-catalog's inventory in the same change"
        );
        assert!(
            call_sites("training.rs", EMIT, include_str!("training.rs")).is_empty(),
            "the trainer must not emit directly either"
        );
    }

    /// Every other shipped module drives no sampler and emits nothing, so the inventoried sites above
    /// are the whole crate. Pinned as a negative so a future render route added elsewhere cannot slip
    /// past an inventory that only looks at four files.
    ///
    /// `preview.rs` is included: this module's own sampler-driving helpers live under `#[cfg(test)]`
    /// and the strip above removes them, so its shipped half must scan clean.
    #[test]
    fn no_other_shipped_module_drives_a_sampler_or_emits() {
        for (file, source) in [
            ("lib.rs", include_str!("lib.rs")),
            ("adapters.rs", include_str!("adapters.rs")),
            ("base.rs", include_str!("base.rs")),
            ("comfyui.rs", include_str!("comfyui.rs")),
            ("common.rs", include_str!("common.rs")),
            ("dit.rs", include_str!("dit.rs")),
            ("memory_strategy.rs", include_str!("memory_strategy.rs")),
            ("packed_dit.rs", include_str!("packed_dit.rs")),
            ("packed_te.rs", include_str!("packed_te.rs")),
            ("preview.rs", include_str!("preview.rs")),
            ("quant.rs", include_str!("quant.rs")),
        ] {
            assert!(
                call_sites(file, &format!("{DRIVER}("), source).is_empty(),
                "{file} drives a sampler but is not in the route inventory"
            );
            assert!(
                call_sites(file, EMIT, source).is_empty(),
                "{file} emits a preview but is not in the route inventory"
            );
        }
    }

    /// The file lists above must be the crate's **whole** `src/` surface, or a new module could hold an
    /// unhooked render route and neither the inventory nor the negative pin would look at it.
    ///
    /// The four `*_validate.rs` files are named here as what they are — out-of-line `#[cfg(test)] mod`
    /// GPU-validation harnesses, which `candle-gen-catalog`'s module-tree walk deliberately excludes —
    /// so adding a fifth one is a diff here rather than a silent hole.
    #[test]
    fn the_inventory_covers_every_file_in_src() {
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
                "adapters.rs",
                "base.rs",
                "base_img2img_validate.rs", // #[cfg(test)] mod — not shipped code
                "comfyui.rs",
                "common.rs",
                "control.rs",
                "control_validate.rs", // #[cfg(test)] mod — not shipped code
                "dit.rs",
                "edit.rs",
                "edit_validate.rs", // #[cfg(test)] mod — not shipped code
                "lib.rs",
                "memory_strategy.rs",
                "packed_dit.rs",
                "packed_te.rs",
                "pipeline.rs",
                "preview.rs",
                "quant.rs",
                "training.rs",
                "turbo_img2img_validate.rs", // #[cfg(test)] mod — not shipped code
            ],
            "a module joined or left src/ — add it to the route inventory or to the negative pin"
        );
    }
}

//! SD3.5's per-step latent preview seam (epic 16948, sc-16958; the MLX original is epic 16624 /
//! `mlx-gen-sd3/src/preview.rs`).
//!
//! Schedule numbering, multi-eval dedup and the swallow-on-failure contract live in
//! [`candle_gen::preview`], shared by every candle family (sc-16949). This module owns exactly two
//! things: the **reused** SD3.5 16-channel fit, and the layout check that says the tensor being
//! projected is one of this family's latents rather than some other 16-channel family's.
//!
//! ## Every denoise lane, enumerated
//!
//! This crate has **one** shared-driver call site and no bespoke loop of any kind: a `git grep` of
//! `run_flow_sampler` / `run_curated_sampler` / `run_scm_sampler` across `candle-gen-sd3/src` returns
//! a single hit, in `pipeline.rs`'s `render_core`, and there is no hand-written `for`-loop denoise
//! beside it. That one site is what every user-reachable lane funnels through — the
//! **one-site, N-callers** shape:
//!
//! | route id | lanes reaching the site | default |
//! | --- | --- | --- |
//! | `sd3_5_large` | txt2img (true CFG); img2img / `Reference` (true CFG, reduced σ tail) | txt2img |
//! | `sd3_5_large_turbo` | txt2img (distilled, no CFG); img2img / `Reference` (distilled) | txt2img |
//! | `sd3_5_medium` | txt2img (true CFG); img2img / `Reference` (true CFG, reduced σ tail) | txt2img |
//!
//! Three registered descriptors × two lanes = six user-reachable lanes, all wired by hooking that one
//! site. The img2img lane is not a second site: `Pipeline::render` resolves the fork step and blends
//! the VAE-encoded reference into `x_t` *before* the driver call, then hands the driver the reduced
//! `sigmas[start..]` tail — so from the first emission onward there is one trajectory and it is the
//! target's, which is this story's "img2img projects the target only" criterion closed structurally.
//! `start_step == 0` (no reference) is the full schedule.
//!
//! There are no name-driven bespoke providers here and no trainer: `load_variant` refuses control and
//! IP-adapter overlays outright, so this crate ships no descriptor-less render lane and therefore no
//! deliberately dark site. `img2img_validate.rs` is `#[cfg(test)]` and does not ship.
//!
//! **CFG never reaches the preview.** Large and Medium run true CFG as *two separate MMDiT forwards
//! inside the predict closure*, blended into one velocity (`v_uncond + cfg·(v_cond − v_uncond)`); no
//! fused `[2, …]` batch is ever the running latent, so there is no unconditional half to project.
//! Turbo is guidance-distilled and runs a single forward.
//!
//! ## The latent shape at the emission point — verified, not assumed
//!
//! The story asked this to be confirmed rather than assumed, and it holds: **SD3.5 is not packed and
//! needs no unpack step.** The MMDiT patchify/unpatchify pair lives entirely inside
//! [`crate::transformer::Sd3Transformer::forward`], so the sampler's running latent never enters the
//! packed token space, and unlike Z-Image there is no frame axis to drop either.
//!
//! | stage | shape | projectable? |
//! | --- | --- | --- |
//! | seeded noise in `render_core` | `[1, 16, H/8, W/8]` | yes — the fitted space |
//! | after the img2img blend `x_t = (1−σ)·clean + σ·noise` | `[1, 16, H/8, W/8]` | yes, unchanged |
//! | what [`candle_gen::run_flow_sampler`] integrates and hands the hook | `[1, 16, H/8, W/8]` | yes |
//! | what the decode tail hands the VAE | `[1, 16, H/8, W/8]` | the same tensor |
//!
//! So hook geometry and latent geometry are not merely bound to one source — there is only one source
//! to bind to, and the projector needs no `width`/`height` argument. The channel count is taken from
//! [`crate::vae::LATENT_CHANNELS`], the very constant `render_core` builds its noise with and the VAE
//! decodes, so this module cannot come to disagree with the denoise about how wide the space is.
//!
//! ## The σ convention: this family needs no correction
//!
//! [`candle_gen::run_flow_sampler`] integrates a
//! [`candle_gen::gen_core::sampling::FlowModelSampling`], whose `input_scale` is exactly `1.0` at
//! every σ, so the running latent already *is* the tensor the fit was measured against and the σ-less
//! [`candle_gen::preview::PreviewHook::new`] constructor is the correct one — not
//! [`candle_gen::preview::PreviewHook::with_sigma`]. sc-16954 found the **opposite** for the discrete
//! ε cohort (SDXL/Kolors denoise in k-diffusion VE σ-space and must apply `1/√(σ²+1)` first, or 89.4%
//! of the first frame clips to the rails). `tests/preview_real_weights.rs` reads `input_scale` off the
//! `ModelSampling` itself and measures this family's own rail-clipped fraction rather than asserting
//! the difference in prose.
//!
//! ## The fit is reused, not refitted — and its 16-channel space is SD3.5's own
//!
//! `RGB_FACTORS` / `RGB_BIAS` (private, hence no links) are the epic-16624 constants transcribed
//! verbatim from
//! `mlx-gen-sd3/src/preview.rs`. They are least-squares numbers over a VAE *latent space* with no
//! backend in them; candle reuses them and deliberately ships **no producer** of its own —
//! `mlx-gen-sd3/tests/fit_preview_rgb.rs` remains the only way they are re-derived.
//!
//! Epic 16948 asked every 16-channel story to settle *which* 16-channel space it occupies, because
//! sc-16956 (Boogu) and sc-16957 (Z-Image) each turned out to be FLUX.1-dev's. **SD3.5 is not.** It is
//! the first genuinely distinct 16-channel space in this epic, and the distinction is in the tensors,
//! not in a config:
//!
//! * All three registered snapshots ship the **same** `vae/diffusion_pytorch_model.safetensors` —
//!   SHA-256 `8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc`, 167,666,902 bytes,
//!   244 bf16 tensors, `vae/config.json` SHA-256
//!   `58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060` (809 bytes) — across
//!   `stabilityai/stable-diffusion-3.5-large` @ `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f` (the
//!   revision the fit was measured on), `…-large-turbo` @ `ec07796fc06b096cc56de9762974a28f4c632eda`
//!   and `…-medium` @ `b940f670f0eda2d07fbb75229e779da1ad11eb80`. One fit therefore covers all three
//!   routes by artifact identity, not by inference from a shared channel count.
//! * `black-forest-labs/FLUX.1-dev` ships a VAE of the **same architecture and the same size** —
//!   167,666,902 bytes, the same 244 keys with the same shapes and the same bf16 dtype — whose SHA-256
//!   is `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`. A tensor walk settles it:
//!   **0 of 244 tensors are byte-identical**, over an identical 167,639,366-byte payload whose SHA-256
//!   is `52163be5…7bb2` for SD3.5 and `44b97a3d…bcb9` for FLUX.1-dev. Same architecture, different
//!   trained weights.
//! * The normalizations differ too, and they are part of the space's definition: SD3.5 denoises
//!   `(mean − 0.0609) · 1.5305` ([`crate::vae::SHIFT_FACTOR`] / [`crate::vae::SCALING_FACTOR`], read
//!   from its own `vae/config.json`) where FLUX.1-dev uses `0.1159` / `0.3611`.
//!
//! Two things follow, both worth stating because both were open questions the epic named. First, this
//! is **not** a third fit over the FLUX.1 latent space, so sc-17309 — which tracks the Z-Image /
//! FLUX.1 duplication — must not gain an SD3.5 row. Second, sharing the
//! [`candle_transformers::models::z_image::vae::AutoEncoderKL`] Rust type with Z-Image says nothing
//! about the space: this crate reuses that type precisely because the *architecture* is shared and the
//! *weights* and scale/shift are not, which is exactly the "matching Rust type" reasoning the epic
//! forbids grounding a reuse in.
//!
//! `tests/preview_real_weights.rs` re-derives every claim above per snapshot; the full record is
//! `docs/migration/evidence/sc-16958-sd3-candle-preview.md`.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

use crate::vae::LATENT_CHANNELS;

/// Ordinary-least-squares map from the normalized SD3.5 VAE latent to latent-resolution RGB (row *i*
/// maps latent channel *i* to `[r, g, b]`), with `RGB_BIAS` the intercept.
///
/// **Reused verbatim from `mlx-gen-sd3/src/preview.rs`, not refitted.** Fit on four diverse
/// real-weight SD3.5-Large renders and measured on two disjoint prompt/seed holdouts, all 256² with
/// eight static-shift-3 flow-Euler steps at CFG 3.5, against 8×8-average-pooled native VAE decodes.
/// Fit R² `(R,G,B) = (0.97402, 0.98248, 0.98505)`, overall `0.98031`; holdout R²
/// `(0.86452, 0.87254, 0.95631)`, overall `0.91459`.
///
/// Donor snapshot: `stabilityai/stable-diffusion-3.5-large` revision
/// `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f`, `vae/diffusion_pytorch_model.safetensors`,
/// 167,666,902 bytes, SHA-256 `8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc` —
/// the same file `…-large-turbo` and `…-medium` ship, and a **different** VAE from FLUX.1-dev's
/// despite the identical architecture and container size (see the module docs).
///
/// The projector consumes the provider's SD3-normalized `(mean − 0.0609) · 1.5305` latent, which is
/// what the sampler integrates and what the decode tail un-scales.
///
/// Refit — in `mlx-gen-sd3`, never here — whenever the SD3.5 VAE lineage changes.
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

/// The fit's intercept — the near-neutral grey a fully-zero latent projects to. Reused with
/// `RGB_FACTORS`.
const RGB_BIAS: [f32; 3] = [0.646_403_3, 0.625_806_57, 0.615_018_96];

/// The latent channel count the fit is defined over, derived from the committed factor table's own
/// length so nothing in this crate can drift from it by restating a number.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// The fit is the SIXTEEN-channel one, and it is defined over the channel count the rest of the crate
/// already denoises and decodes in. Compile-time, because a runtime row over constants proves nothing
/// a `const` assertion does not prove earlier.
const _: () = assert!(PREVIEW_LATENT_CHANNELS == 16 && PREVIEW_LATENT_CHANNELS == LATENT_CHANNELS);

/// Project SD3.5's native spatial running latent `[1, 16, h, w]` to a latent-resolution RGB8 preview.
///
/// There is nothing to recover first: this is already the `[1, C, h, w]` contract
/// [`candle_gen::preview::project_latents`] takes, and it is the same tensor the decode tail hands the
/// VAE. The only work here is rejecting a latent that is not one of this family's.
///
/// Errors on any other layout. The caller's frame is then lost and swallowed by
/// [`candle_gen::preview::emit_preview`], which is the intended decorative-failure behaviour.
pub fn project_latents(latents: &Tensor) -> Result<Image> {
    check_layout(latents)?;
    candle_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

/// Reject anything that is not one batch-1 latent in the fitted sixteen-channel space.
///
/// The shared projection would reject most of these anyway, but naming the channel count here makes
/// the failure say *SD3.5* — and catches the one case it cannot see, a rank-4 latent whose channel
/// count merely happens to match some other family's.
fn check_layout(latents: &Tensor) -> Result<()> {
    let dims = latents.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != PREVIEW_LATENT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "SD3.5 preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, h, w], got {dims:?}"
        )));
    }
    Ok(())
}

/// The preview hook every SD3.5 lane hands [`candle_gen::run_flow_sampler`]: a projector closure over
/// [`project_latents`]. The driver owns frame numbering, multi-eval dedup and the swallow-on-failure
/// contract (sc-16949), so no route restructures its loop.
///
/// Build it **per image**: a batched request runs one driver call per seed and each call must start a
/// fresh trajectory at frame 1. (The driver builds its own counter per call, so this is a property of
/// the call rather than of the hook — but building the hook alongside the render keeps the two from
/// drifting apart.)
pub(crate) fn hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project_latents)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device};
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};
    use candle_gen::gen_core::PreviewFrame;
    use candle_gen::preview::PreviewCounter;

    use super::*;

    fn zeros(shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).expect("zeros")
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    // ── The fit ───────────────────────────────────────────────────────────────────────────────────

    /// The reused table must have one finite row per latent channel, and the channel count it defines
    /// must be the one the rest of the crate denoises and decodes in.
    #[test]
    fn the_reused_fit_has_one_finite_row_per_sd3_latent_channel() {
        assert_eq!(RGB_FACTORS.len(), 16);
        assert_eq!(PREVIEW_LATENT_CHANNELS, LATENT_CHANNELS);
        assert!(RGB_FACTORS.iter().flatten().all(|v| v.is_finite()));
        assert!(RGB_BIAS.iter().all(|v| v.is_finite()));
    }

    /// The constants are the epic-16624 SD3.5 numbers, not some other family's. Spot-pinned on the
    /// bias — which is the row most likely to be copied from a neighbouring crate by accident, since
    /// every family's is a near-neutral grey — plus the two extreme factor rows.
    ///
    /// The FLUX.1 / Z-Image bias values are named in the failure message rather than imported: this
    /// crate deliberately does not depend on those crates, and the point is that SD3.5's space is its
    /// own.
    #[test]
    fn the_committed_constants_are_the_sd3_ones() {
        assert_eq!(RGB_BIAS, [0.646_403_3, 0.625_806_57, 0.615_018_96]);
        assert_ne!(
            RGB_BIAS,
            [0.495_903_46, 0.492_634_95, 0.482_487_62],
            "this is candle-gen-flux's FLUX.1 bias — SD3.5's 16-channel space is a different one"
        );
        assert_ne!(
            RGB_BIAS,
            [0.502_150_24, 0.483_383_92, 0.458_297_43],
            "this is candle-gen-z-image's bias — SD3.5's 16-channel space is a different one"
        );
        assert_eq!(RGB_FACTORS[6], [0.095_622_25, 0.048_480_41, 0.052_133_94]);
        assert_eq!(
            RGB_FACTORS[15],
            [-0.003_357_253, -0.035_778_474, -0.019_335_456]
        );
    }

    // ── The layout contract ───────────────────────────────────────────────────────────────────────

    /// The native spatial latent projects at latent resolution, with no unpack and no squeeze — the
    /// shape claim the story asked to be confirmed rather than assumed.
    #[test]
    fn the_native_spatial_latent_projects_at_latent_resolution() {
        let image = project_latents(&zeros((1, 16, 3, 5))).expect("project");
        assert_eq!((image.width, image.height), (5, 3));
        assert_eq!(image.pixels.len(), 5 * 3 * 3);
    }

    /// Anything that is not one batch-1 sixteen-channel spatial latent is refused, and the message
    /// says SD3.5. The rank-3 row is the one that matters: a packed token sequence is what a
    /// copy-paste from a FLUX-family crate would produce, and it must not project.
    #[test]
    fn a_latent_that_is_not_this_family_is_refused() {
        let packed = Tensor::zeros((1, 256, 64), DType::F32, &Device::Cpu).expect("packed");
        let five_d = Tensor::zeros((1, 16, 1, 4, 4), DType::F32, &Device::Cpu).expect("5-D");
        for bad in [
            zeros((2, 16, 4, 4)), // batched
            zeros((1, 32, 4, 4)), // FLUX.2's channel width
            zeros((1, 4, 4, 4)),  // SDXL's channel width
            packed,               // a packed FLUX-family token sequence
            five_d,               // Z-Image's 5-D frame-axis latent
        ] {
            let error = project_latents(&bad)
                .expect_err("must be refused")
                .to_string();
            assert!(
                error.contains("SD3.5 preview latent must have shape [1, 16, h, w]"),
                "got: {error}"
            );
        }
    }

    // ── The hook ──────────────────────────────────────────────────────────────────────────────────

    /// The hook emits one frame per schedule position, dedups a repeat before projecting, and reports
    /// the driver's numbering — the multi-eval contract, exercised through this family's own hook
    /// rather than only through the shared module's tests.
    #[test]
    fn the_hook_numbers_frames_and_dedups_a_repeated_position() {
        let sigmas = [1.0f32, 0.5, 0.0];
        let (sink, frames) = collecting_sink();
        let hook = hook(&sink);
        let counter = PreviewCounter::new(&sigmas);
        let latents = zeros((1, 16, 2, 2));

        hook.emit(&counter, &sigmas, sigmas[0], &latents);
        hook.emit(&counter, &sigmas, sigmas[0], &latents); // a heun-style repeat
        hook.emit(&counter, &sigmas, sigmas[1], &latents);

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.current, f.total))
                .collect::<Vec<_>>(),
            vec![(1, 2), (2, 2)],
            "a two-step schedule evaluated three times must emit exactly two frames"
        );
    }

    /// A malformed latent loses exactly one decorative frame and consumes its schedule position; the
    /// render is unaffected. The failure is swallowed by the shared emitter, never surfaced.
    #[test]
    fn a_malformed_latent_loses_one_frame_and_consumes_its_position() {
        let sigmas = [1.0f32, 0.5, 0.0];
        let (sink, frames) = collecting_sink();
        let hook = hook(&sink);
        let counter = PreviewCounter::new(&sigmas);

        hook.emit(&counter, &sigmas, sigmas[0], &zeros((1, 15, 2, 2)));
        hook.emit(&counter, &sigmas, sigmas[1], &zeros((1, 16, 2, 2)));

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 2));
    }

    /// An inert sink does no tensor work at all and does not advance the counter — the property that
    /// makes an unwatched render byte-identical to a pre-sc-16958 one.
    #[test]
    fn an_inert_sink_projects_nothing_and_advances_nothing() {
        let sigmas = [1.0f32, 0.0];
        let inert = PreviewSink::default();
        let hook = hook(&inert);
        assert!(!hook.is_active());
        let counter = PreviewCounter::new(&sigmas);
        // A latent that would panic the projection if it were ever reached is not reached.
        hook.emit(&counter, &sigmas, sigmas[0], &zeros((1, 15, 2, 2)));
        assert_eq!(counter.next(&sigmas, sigmas[0]), Some(1));
    }

    // ── The wiring, pinned against this crate's own source ────────────────────────────────────────

    /// The shipped half of `pipeline.rs` — everything ahead of its single `#[cfg(test)] mod tests`.
    ///
    /// The structural rows in that module drive [`super::hook`] over inert sinks, so a scan that
    /// counted them would read test code as shipped wiring.
    fn shipped_pipeline() -> &'static str {
        const SOURCE: &str = include_str!("pipeline.rs");
        const MARKER: &str = "\n#[cfg(test)]\n";
        assert_eq!(
            SOURCE.matches(MARKER).count(),
            1,
            "pipeline.rs must hold exactly one `#[cfg(test)]` item for this split to be sound — \
             teach this scan about the new one rather than letting it read test code as shipped code"
        );
        &SOURCE[..SOURCE.find(MARKER).expect("the marker was just counted")]
    }

    /// The shipped render lane builds its hook over the **request's** sink, and `render_core` takes
    /// that hook **by reference** rather than as an `Option`.
    ///
    /// Both halves are load-bearing and neither implies the other, because the epic's cross-crate
    /// guard cannot see either. `candle-gen-catalog`'s `preview_advertising` inventory classifies the
    /// preview argument of the `run_flow_sampler` call *inside* `render_core` — one hop past
    /// `Pipeline::render`. While that hop was an `Option`, changing the single caller argument to
    /// `None` took all six SD3.5 lanes dark while `hooked: 1`, all three `PREVIEW_ROUTE_IDS` and
    /// `supports_preview: true` went on advertising and the entire CPU suite stayed green; the only
    /// rows that would have caught it are `#[ignore]`d and CUDA-only.
    ///
    /// The `&PreviewHook` parameter makes going dark a **type error**, which is the same immunity
    /// `candle-gen-chroma` has. This row pins the parameter so that immunity cannot be undone by
    /// quietly widening it back, and pins the sink so the hook cannot be built over an inert one
    /// instead — the one way left to go dark without changing a type.
    #[test]
    fn the_render_lane_builds_its_hook_from_the_requests_sink() {
        let shipped = shipped_pipeline();

        // The sinks: exactly TWO hooks are built in shipped code — the single-pass `Pipeline::render`
        // lane and the chained denoise-pass lane (sc-20425) — and BOTH are built over the request's.
        // The count is pinned so a *third* lane cannot appear without being named in this crate's
        // inventory and in the catalog's.
        assert_eq!(
            shipped.matches("preview::hook(").count(),
            2,
            "pipeline.rs must build exactly two preview hooks in shipped code (the single-pass render \
             lane and the chained denoise-pass lane) — a further render lane must be named in this \
             crate's inventory (and in the catalog's) rather than appearing here"
        );
        assert_eq!(
            shipped.matches("preview::hook(&req.preview)").count(),
            2,
            "every render lane must build its hook over the REQUEST's sink; a hook over any other sink \
             is a dark lane that still passes every argument-shaped check"
        );

        // The parameter: read out of the declaration itself, so this cannot be satisfied by the
        // spelling appearing anywhere else in the file.
        let at = shipped
            .find("pub(crate) fn render_core(")
            .expect("render_core must be declared in pipeline.rs");
        let declaration = &shipped[at..];
        let end = declaration
            .find("\n) -> Result<Image> {")
            .expect("render_core's parameter list must end at its return type");
        let parameter = declaration[..end]
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with("preview:"))
            .expect("render_core must take a preview parameter");
        assert_eq!(
            parameter, "preview: &candle_gen::preview::PreviewHook<'_>,",
            "render_core must take its hook by reference. An `Option` here is blankable at the \
             caller, and that mutation is invisible to the catalog's route inventory — it classifies \
             the driver argument one hop further in, which would still read `Some(preview)`."
        );
    }

    // ── The σ convention ──────────────────────────────────────────────────────────────────────────

    /// SD3.5 drives `TimestepConvention::Sigma` over a `FlowModelSampling`, whose `input_scale` is
    /// identically 1.0 — so the running latent already is the tensor the fit was measured against and
    /// [`PreviewHook::new`] is the correct constructor rather than
    /// [`candle_gen::preview::PreviewHook::with_sigma`].
    ///
    /// Read off the very `ModelSampling` the driver integrates rather than asserted about the family
    /// in prose. `tests/preview_real_weights.rs` measures the consequence — the first frame's
    /// rail-clipped fraction — on this family's own prior.
    #[test]
    fn the_flow_contract_needs_no_input_scaling() {
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(
                ms.input_scale(sigma),
                1.0,
                "FlowModelSampling::input_scale must be identically 1.0; at {sigma} it is not, and \
                 SD3.5 would need PreviewHook::with_sigma"
            );
        }
    }
}

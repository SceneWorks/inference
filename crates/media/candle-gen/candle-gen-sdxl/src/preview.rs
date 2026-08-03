//! The SDXL-family per-step latent preview seam (epic 16948, sc-16954; the MLX original is epic
//! 16624 / `mlx-gen-sdxl/src/preview.rs`).
//!
//! This module owns the **4-channel** SDXL/Kolors fit — the smallest channel count in the epic, and
//! the one that most exercises the sc-16949 hoist's genericity: [`candle_gen::preview::project_latents`]
//! must have no 16-channel assumption baked in. Schedule numbering, multi-eval dedup and the
//! swallow-on-failure contract all live in [`candle_gen::preview`], shared by every candle family.
//!
//! `candle-gen-kolors` reuses these coefficients through [`project_spatial_latents`] rather than
//! restating them; `candle-gen-instantid` deliberately does **not** — see the adjudication below.
//!
//! ## The latent shape at the emission point — verified, not assumed
//!
//! Every SDXL and Kolors denoise lane runs a **rank-4 spatial** latent `[1, 4, H/8, W/8]` from the
//! first σ to the last. `crate::pipeline::Pipeline::render` builds `(1, 4, lat_h, lat_w)` directly,
//! `crate::denoise::seeded_sigma_prior` returns the same NCHW shape, and the decode tail
//! (`crate::pipeline::tiled_vae_decode`) takes exactly `[1, 4, h, w]`. So unlike Qwen-Image (packed
//! rank 3) or Anima (5-D Cosmos), SDXL needs **no layout adaptation at all** — there is no unpack
//! step to write and none is written.
//!
//! The batch axis is always 1: `req.count` is served sequentially through
//! `candle_gen::for_each_image_seed`, one fresh `[1, 4, h, w]` prior per image, and CFG never widens
//! the running latent (below).
//!
//! ## The latent *convention* at the emission point — the part that is NOT shared with the flow cohort
//!
//! [`candle_gen::run_curated_sampler`] hands the hook the **running** latent `x`, never the
//! `c_in`-scaled model input `x_in`, and documents that as the property making the hook see "the
//! tensor a family's linear RGB fit was measured against". That is true for the flow-match families
//! wired before this story — `FlowModelSampling::input_scale` is exactly `1.0` at every σ — and it is
//! **false here**.
//!
//! SDXL and Kolors denoise in k-diffusion **VE σ-space**: the prior is `unit noise · σ_max` with
//! σ_max ≈ 14.6, and `gen_core::sampling::DiscreteModelSampling::input_scale` supplies the
//! `1/√(σ²+1)` renormalization *inside* the driver. The MLX fit was measured on 12-step **ancestral
//! Euler**, whose sampler folds that renormalization into its own step — so the fit's domain is the
//! renormalized latent, not the raw VE one. Projecting `x` directly would push the early frames to
//! roughly `σ·ε` against `~0.17` slopes, clamping them to a saturated binary field instead of the
//! noise-to-image progression the fit describes.
//!
//! [`project_ve_latents`] therefore applies the family's own `input_scale` before projecting, and the
//! lanes that already hold a renormalized latent ([`project_spatial_latents`]) apply nothing. Which
//! lane is which is not a judgement call — it is read off what the lane feeds its UNet:
//!
//! | lane | running latent | projector |
//! | --- | --- | --- |
//! | `Pipeline::denoise_curated`, `denoise::denoise_curated`, Kolors `Pipeline::denoise_curated` | VE σ-space (driver applies `c_in`) | [`project_ve_latents`] |
//! | `Pipeline::denoise_lightning`, Kolors native leading-Euler | VE-like (lane applies its own `c_in` / `scale_in`) | [`project_ve_latents`] |
//! | `denoise::denoise_ip_multi_control`, `SdxlEdit::denoise_edit` | already renormalized — ancestral folds it into the step, "the UNet input is the raw latents" | [`project_spatial_latents`] |
//!
//! At the final emission σ is small, so `c_in → 1` and the two agree; the correction only ever
//! changes the early frames, which is precisely where the uncorrected projection was wrong.
//!
//! ## CFG never reaches the preview
//!
//! Every lane fuses `[uncond, cond]` **inside** its predict closure — `Tensor::cat(&[x, x], 0)` on
//! entry, `chunk(2, 0)` plus the guidance combine before returning — so the tensor the sampler
//! carries as its running latent is batch 1 at every step and no unconditional half exists for a
//! preview to project. The bespoke ancestral and leading-Euler loops keep the same discipline
//! (`latents` stays batch 1; only `x_unet` is widened). Pinned by rows that drive the real lanes.
//!
//! ## The fit is reused, not refitted — grounded in tensor bytes
//!
//! The claim being checked is not "both engines name a type `AutoEncoderKL`". SDXL and Kolors are one
//! latent space because they ship **one VAE file**: `vae/diffusion_pytorch_model.fp16.safetensors`,
//! SHA-256 `bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68`, 167,335,342 bytes, is
//! byte-identical across `stabilityai/stable-diffusion-xl-base-1.0`, `Kwai-Kolors/Kolors-diffusers`,
//! and **every** shipped tier (`bf16`/`q8`/`q4`) of the `SceneWorks/sdxl-base-mlx` and
//! `SceneWorks/kolors-mlx` re-hosts — the MLX packer mirrors the VAE dense rather than packing it.
//! That is the same hash `mlx-gen-sdxl/src/preview.rs` cites as its Kolors grounding, so the fit
//! donor's file *is* this file. Both `vae/config.json`s declare `latent_channels: 4` and
//! `scaling_factor: 0.13025`, the two numbers that define the space.
//!
//! One asymmetry is recorded rather than glossed: candle's SDXL **decode** runs the caller-staged
//! `madebyollin/sdxl-vae-fp16-fix` (`crate::loaders::load_sdxl_vae`), which is a genuine fine-tune —
//! all 248 tensors differ from the original in both encoder and decoder — whereas Kolors decodes with
//! the snapshot's own VAE. It is a documented drop-in for the *same* latent space (the UNet that
//! produces these latents is byte-identical across engines, and `VAE_SCALE` is unchanged at 0.13025),
//! so the fit's input domain is unaffected; what it could in principle move is the fit's colour
//! target. That is settled empirically rather than by assertion — the real-weight rows in
//! `tests/preview_real_weights.rs` measure convergence against the image this decoder actually
//! produces. See `docs/migration/evidence/sc-16954-sdxl-candle-preview.md`.
//!
//! ## InstantID is deliberately not wired
//!
//! `candle-gen-instantid` registers no descriptor at all — it is a `BESPOKE_UTILITY_CRATES` member
//! and `candle-gen-catalog` actively forbids it acquiring one — so there is no `supports_preview` to
//! flip and no catalog row to inventory. It reaches this crate's [`crate::denoise::denoise_curated`]
//! and [`crate::denoise::denoise_ip_multi_control`] directly; both now take a preview argument, and
//! InstantID passes `None` at every call. MLX left it unadvertised for the same reason.
//!
//! A stale or absent fit degrades preview colour only; the denoise path never reads these constants.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

/// Ordinary-least-squares map from SDXL-family VAE latents to latent-resolution RGB.
///
/// **Reused verbatim from `mlx-gen-sdxl/src/preview.rs:27`, not refitted.** These are least-squares
/// constants over a VAE *latent space*; there is no backend in them, and epic 16948 reuses every fit
/// epic 16624 committed once a family has proven it loads the same VAE bytes (above). There is
/// deliberately no candle producer of these numbers — `mlx-gen-sdxl/tests/fit_preview_rgb.rs` remains
/// the only way they are re-derived.
///
/// Fit on four diverse 512² real-weight SDXL renders (warm/cool, indoor/outdoor,
/// portrait/still-life/landscape; seeds 1663301..1663304) and evaluated on two disjoint
/// subject/palette holdouts (seeds 1663391, 1663392), all 12-step ancestral Euler at CFG 5.0, against
/// 8×8-average-pooled VAE decode targets. Fit R² `(R,G,B) = (0.91640, 0.92538, 0.91487)`, overall
/// `0.91849`; holdout R² `(0.86501, 0.84844, 0.86649)`, overall `0.86065`.
///
/// That the targets were 8×8-pooled decodes is also what fixes the fit's **domain**: an ancestral
/// Euler latent, i.e. the `1/√(σ²+1)`-renormalized one. See the module docs for why the VE lanes must
/// apply that scaling before projecting.
///
/// Refit whenever the SDXL-family VAE lineage or latent normalization changes.
const RGB_FACTORS: [[f32; 3]; 4] = [
    [0.171_078_03, 0.205_344_2, 0.213_290_84],
    [-0.128_209_89, 0.028939432, 0.044224623],
    [0.046837712, 0.052948396, 0.006_726_24],
    [-0.181_879_64, -0.124_704_68, -0.124_656_26],
];

/// The fit's intercept — the colour a fully-zero latent projects to. Reused with [`RGB_FACTORS`].
const RGB_BIAS: [f32; 3] = [0.555_939, 0.509_310_5, 0.492_320_7];

/// The SDXL-family latent channel count the fit is defined over. Derived from the committed factor
/// table's own length, so a consumer (Kolors) cannot drift from it by restating a number.
pub const PREVIEW_LATENT_CHANNELS: usize = RGB_FACTORS.len();

/// The fit is the FOUR-channel one. Compile-time, because a runtime row over constants proves nothing
/// a `const` assertion does not prove earlier and more cheaply.
const _: () = assert!(PREVIEW_LATENT_CHANNELS == 4);

/// The intercept is R > G > B — a warm grey, not a neutral one. This is why sc-16950's
/// `r_first < 0.35` correlation ceiling does not generalize and is deliberately not ported: a preview's
/// noise floor carries the fit's own channel-mean structure, so it is scene-dependent. The real-weight
/// harness uses `r_last - r_first > 0.30` with a loose `r_first < 0.75` instead.
const _: () = assert!(RGB_BIAS[0] > RGB_BIAS[1] && RGB_BIAS[1] > RGB_BIAS[2]);

/// Project an **already-renormalized** SDXL-family latent `[1, 4, h, w]` to a latent-resolution RGB8
/// preview.
///
/// This is the reuse seam `candle-gen-kolors` calls: it shares the SDXL latent space (one byte-identical
/// VAE file, `scaling_factor` 0.13025) and therefore these coefficients, and calling through here is
/// what keeps a second copy of them from existing.
///
/// "Already renormalized" means the ancestral / edit lanes, whose sampler folds the `1/√(σ²+1)` input
/// scaling into its own step so the running latent is the tensor the fit was measured on. A lane that
/// denoises in raw VE σ-space must use [`project_ve_latents`] instead.
///
/// Errors on any layout that is not one batch-1 four-channel spatial latent; the caller's frame is
/// then lost and swallowed by `candle_gen::preview::emit_preview`, the intended decorative-failure
/// behaviour.
pub fn project_spatial_latents(latents: &Tensor) -> Result<Image> {
    check_layout(latents)?;
    candle_gen::preview::project_latents(latents, &RGB_FACTORS, RGB_BIAS)
}

/// Project a **k-diffusion VE σ-space** SDXL-family latent by first applying the family's own
/// `1/√(σ²+1)` input scaling, then [`project_spatial_latents`].
///
/// `sigma` is the schedule σ the frame is being emitted at, as delivered by
/// `candle_gen::preview::PreviewHook::with_sigma`. `None` — which only a σ-less driver produces, and
/// no SDXL-family lane uses — is an error rather than an un-scaled projection: silently projecting the
/// raw VE latent is exactly the failure this function exists to prevent, and an error is swallowed
/// into a lost decorative frame rather than a wrong one.
///
/// The scaling is `gen_core::sampling::DiscreteModelSampling::input_scale`'s closed form. It is
/// spelled out here rather than taken from a `ModelSampling` handle because the bespoke Lightning and
/// Kolors leading-Euler lanes hold their own equivalent coefficient and no trait object at all; the
/// rows in `tests/preview_real_weights.rs` pin it against the real `DiscreteModelSampling`.
pub fn project_ve_latents(latents: &Tensor, sigma: Option<f32>) -> Result<Image> {
    let Some(sigma) = sigma else {
        return Err(CandleError::Msg(
            "sdxl preview: a VE-space latent needs the schedule sigma to renormalize with, but the \
             driver supplied none"
                .into(),
        ));
    };
    project_spatial_latents(&renormalize(latents, sigma)?)
}

/// `x · 1/√(σ²+1)` — the VE → fit-domain map, as `DiscreteModelSampling::input_scale` computes it.
fn renormalize(latents: &Tensor, sigma: f32) -> Result<Tensor> {
    let scale = 1.0 / ((sigma * sigma + 1.0) as f64).sqrt();
    Ok(latents.affine(scale, 0.0)?)
}

/// Reject anything that is not one batch-1 latent in the fitted four-channel space.
///
/// The shared projection would reject most of these anyway, but naming the channel count here makes
/// the failure say *SDXL* — and catches the one case it cannot see, a rank-4 latent whose channel
/// count merely happens to match some other family's.
fn check_layout(latents: &Tensor) -> Result<()> {
    let dims = latents.dims();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != PREVIEW_LATENT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "sdxl preview latent must have shape [1, {PREVIEW_LATENT_CHANNELS}, h, w], got {dims:?}"
        )));
    }
    Ok(())
}

/// The preview hook a **VE σ-space** lane hands `candle_gen::run_curated_sampler`.
///
/// Built per image: the driver starts a fresh counter per call, and building the hook alongside the
/// call keeps the two impossible to separate.
pub(crate) fn ve_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::with_sigma(sink, project_ve_latents)
}

/// The preview hook for a lane whose running latent is **already renormalized** — the ancestral and
/// edit loops. Public so `candle-gen-kolors`' bespoke providers can build the same seam.
pub fn spatial_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project_spatial_latents)
}

/// The VE-space hook, for `candle-gen-kolors`' bespoke providers, which reach
/// [`crate::denoise::denoise_curated`] in this crate rather than owning a driver call of their own.
pub fn ve_hook_for(sink: &PreviewSink) -> PreviewHook<'_> {
    ve_hook(sink)
}

#[cfg(test)]
mod tests {
    use candle_gen::candle_core::{DType, Device};

    use super::*;

    fn zeros(shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    /// A zero latent projects to the fit's intercept — the one place the committed bias is directly
    /// observable, so a typo in `RGB_BIAS` cannot pass.
    #[test]
    fn a_zero_latent_projects_to_the_fit_intercept() {
        let image = project_spatial_latents(&zeros((1, 4, 2, 3))).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        let expect: Vec<u8> = RGB_BIAS
            .iter()
            .map(|c| (c * 255.0).round() as u8)
            .collect::<Vec<_>>()
            .repeat(6);
        assert_eq!(image.pixels, expect);
    }

    #[test]
    fn projection_rejects_every_non_sdxl_layout() {
        // Rank 3 (a packed Qwen-shaped latent), batch 2, and a 16-channel spatial latent.
        let rank_three = Tensor::zeros((4, 2, 3), DType::F32, &Device::Cpu).unwrap();
        for bad in [rank_three, zeros((2, 4, 2, 3)), zeros((1, 16, 2, 3))] {
            let error = project_spatial_latents(&bad).unwrap_err().to_string();
            assert!(
                error.contains("sdxl preview latent must have shape [1, 4, h, w]"),
                "unexpected error: {error}"
            );
        }
    }

    /// The VE correction is `1/√(σ²+1)`, and it must agree with the `DiscreteModelSampling` the
    /// denoise itself integrates — not merely be some decreasing function of σ.
    #[test]
    fn ve_renormalization_matches_discrete_model_sampling_input_scale() {
        use candle_gen::gen_core::sampling::{DiscreteModelSampling, ModelSampling};
        let sched = crate::pipeline::sdxl_alpha_schedule().unwrap();
        let ms = DiscreteModelSampling::sdxl(&sched);
        for sigma in [0.0292f32, 0.5, 1.0, 4.0, 14.6] {
            let ours = 1.0 / ((sigma * sigma + 1.0) as f64).sqrt();
            let theirs = ms.input_scale(sigma) as f64;
            assert!(
                (ours - theirs).abs() < 1e-6,
                "sigma {sigma}: ours {ours} vs DiscreteModelSampling {theirs}"
            );
        }
    }

    /// At a large σ the raw VE projection saturates and the corrected one does not — the concrete
    /// reason the correction exists. At a small σ the two converge, which is why the LAST frame is
    /// unaffected either way.
    #[test]
    fn the_ve_correction_changes_early_frames_and_not_late_ones() {
        let latents = Tensor::from_vec(
            (0..4 * 4 * 4)
                .map(|i| (i % 7) as f32 - 3.0)
                .collect::<Vec<f32>>(),
            (1, 4, 4, 4),
            &Device::Cpu,
        )
        .unwrap();

        let raw = project_spatial_latents(&latents).unwrap();
        let early = project_ve_latents(&latents, Some(14.6)).unwrap();
        let late = project_ve_latents(&latents, Some(0.0292)).unwrap();

        assert_ne!(
            raw.pixels, early.pixels,
            "the correction must actually change a large-sigma frame"
        );
        assert_eq!(
            raw.pixels, late.pixels,
            "at the last schedule position c_in -> 1, so the corrected and raw projections agree"
        );

        // Saturation is the failure mode being avoided: the uncorrected large-sigma frame clips to
        // the 0/255 rails far more than the corrected one.
        let rails = |p: &[u8]| p.iter().filter(|&&v| v == 0 || v == 255).count();
        assert!(
            rails(&raw.pixels) > rails(&early.pixels),
            "uncorrected {} vs corrected {} rail pixels",
            rails(&raw.pixels),
            rails(&early.pixels)
        );
    }

    /// A σ-less driver must not silently project an un-scaled VE latent.
    #[test]
    fn ve_projection_without_a_sigma_is_an_error_not_an_unscaled_projection() {
        let error = project_ve_latents(&zeros((1, 4, 2, 3)), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("needs the schedule sigma"), "{error}");
    }
}

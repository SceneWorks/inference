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
//! `crate::denoise::seeded_sigma_prior` returns the same NCHW shape, and the decode seam
//! (`crate::pipeline::SdxlLatentDecoder`) takes exactly `[1, 4, h, w]`. So unlike Qwen-Image (packed
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
//! [`project_ve_latents`] therefore applies the family's own `input_scale` before projecting, and a
//! lane that already holds the renormalized tensor — because it was handed one, or because it just
//! computed one to feed the UNet — projects that tensor directly with [`project_spatial_latents`].
//! Which lane is which is not a judgement call: the projector is always shown the tensor the lane
//! feeds its UNet, so it is read off the lane's own model input:
//!
//! | lane | running latent | projector |
//! | --- | --- | --- |
//! | `Pipeline::denoise_curated`, `denoise::denoise_curated`, Kolors `Pipeline::denoise_curated` | VE σ-space (driver applies `c_in`) | [`project_ve_latents`] |
//! | `Pipeline::denoise_lightning`, Kolors native leading-Euler | VE-like, but the lane computes its own `c_in` / `1÷scale_in` and previews the product | [`project_spatial_latents`] on `x·c_in` |
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
///
/// Public because `candle-gen-kolors`' curated lanes reach [`crate::denoise::denoise_curated`] in
/// this crate rather than owning a driver call of their own, so they need the same seam. There is no
/// companion constructor for the already-renormalized lanes: those hold a bespoke loop and call
/// `candle_gen::preview::emit_preview_at` with [`project_spatial_latents`] directly, so a hook would
/// have no caller.
pub fn ve_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::with_sigma(sink, project_ve_latents)
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

    // ── The forwarded hook, pinned against this crate's own source ────────────────────────────────
    //
    // `denoise::denoise_curated` takes `Option<&PreviewHook>` and forwards it to the shared driver.
    // `candle-gen-catalog`'s `preview_advertising` inventory classifies the argument at the DRIVER
    // call — inside `denoise_curated`'s own body, where it reads `preview` and therefore always says
    // `hooked: 1`. It cannot see what the *callers* pass. Blanking `ip_provider.rs`'s `Some(&preview)`
    // to `None` therefore takes the IP-adapter lane preview-dark while the inventory, the route ids
    // and `supports_preview: true` all keep advertising, and the CPU suite stays green — the same
    // two-hop hole sc-16958's review found in `candle-gen-sd3` (closed there by making the parameter
    // a non-`Option`, which is not available here: `denoise_curated` is `pub` and `candle-gen-kolors`
    // / `candle-gen-instantid` reach it too, and InstantID passes `None` on purpose).
    //
    // `Pipeline::denoise_curated` and `Pipeline::denoise_lightning` need no row: each builds its hook
    // in the same body that drives the sampler, which is exactly the shape the catalog already sees.

    /// The shared curated-denoise helper the in-crate IP lane forwards its hook through. Spelled
    /// without its open paren so this module cannot match itself.
    const SHARED_DENOISE: &str = "denoise_curated";

    /// `denoise::denoise_curated`'s argument count and the position of its preview argument, both
    /// re-derived from the declaration by
    /// [`the_shared_curated_denoise_signature_pins_the_preview_argument_position`] rather than
    /// trusted — a reordered signature must fail loudly instead of shifting the pin below onto a
    /// neighbouring argument.
    const SHARED_DENOISE_ARITY: usize = 16;
    const SHARED_DENOISE_PREVIEW_AT: usize = 13;

    /// Every module this crate ships, so a caller added anywhere is seen.
    ///
    /// `ip_validate.rs` and `edit_validate.rs` are the only `src` files left out: `lib.rs` declares
    /// both under `#[cfg(test)] mod`, so neither ships.
    const MODULES: [(&str, &str); 25] = [
        ("adapters.rs", include_str!("adapters.rs")),
        ("clip.rs", include_str!("clip.rs")),
        ("conditioning.rs", include_str!("conditioning.rs")),
        ("denoise.rs", include_str!("denoise.rs")),
        ("edit_provider.rs", include_str!("edit_provider.rs")),
        ("ip_adapter.rs", include_str!("ip_adapter.rs")),
        ("ip_provider.rs", include_str!("ip_provider.rs")),
        ("ldm.rs", include_str!("ldm.rs")),
        ("lib.rs", include_str!("lib.rs")),
        ("loaders.rs", include_str!("loaders.rs")),
        ("pipeline.rs", include_str!("pipeline.rs")),
        ("preview.rs", include_str!("preview.rs")),
        ("sampler.rs", include_str!("sampler.rs")),
        ("training.rs", include_str!("training.rs")),
        ("vision_encoder.rs", include_str!("vision_encoder.rs")),
        ("weights.rs", include_str!("weights.rs")),
        ("unet/attention.rs", include_str!("unet/attention.rs")),
        ("unet/controlnet.rs", include_str!("unet/controlnet.rs")),
        ("unet/conv.rs", include_str!("unet/conv.rs")),
        ("unet/embeddings.rs", include_str!("unet/embeddings.rs")),
        ("unet/mod.rs", include_str!("unet/mod.rs")),
        ("unet/resnet.rs", include_str!("unet/resnet.rs")),
        ("unet/unet_2d.rs", include_str!("unet/unet_2d.rs")),
        (
            "unet/unet_2d_blocks.rs",
            include_str!("unet/unet_2d_blocks.rs"),
        ),
        ("unet/vae_encode.rs", include_str!("unet/vae_encode.rs")),
    ];

    /// `source` with its comments removed and its string literals left intact, so the helper's name
    /// written in prose can never be read as a call site.
    fn code_only(file: &str, source: &str) -> String {
        let chars: Vec<char> = source.chars().collect();
        let mut out = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            let (ch, next) = (chars[i], chars.get(i + 1).copied());
            // A raw string literal (`r"…"` / `r#"…"#`), whose body may hold unescaped quotes.
            if ch == 'r'
                && !i
                    .checked_sub(1)
                    .is_some_and(|p| chars[p].is_alphanumeric() || chars[p] == '_')
            {
                let mut hashes = 0usize;
                while chars.get(i + 1 + hashes) == Some(&'#') {
                    hashes += 1;
                }
                if chars.get(i + 1 + hashes) == Some(&'"') {
                    out.push(' ');
                    i += 2 + hashes;
                    loop {
                        assert!(i < chars.len(), "{file}: unterminated raw string literal");
                        if chars[i] == '"'
                            && (0..hashes).all(|h| chars.get(i + 1 + h) == Some(&'#'))
                        {
                            i += 1 + hashes;
                            break;
                        }
                        i += 1;
                    }
                    continue;
                }
            }
            match (ch, next) {
                ('/', Some('/')) => {
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1;
                    }
                    out.push(' ');
                }
                ('/', Some('*')) => {
                    let mut nesting = 0usize;
                    loop {
                        assert!(i < chars.len(), "{file}: unterminated block comment");
                        match (chars[i], chars.get(i + 1).copied()) {
                            ('/', Some('*')) => {
                                nesting += 1;
                                i += 2;
                            }
                            ('*', Some('/')) => {
                                nesting -= 1;
                                i += 2;
                                if nesting == 0 {
                                    break;
                                }
                            }
                            _ => i += 1,
                        }
                    }
                    out.push(' ');
                }
                ('"', _) => {
                    out.push('"');
                    i += 1;
                    let mut escaped = false;
                    loop {
                        assert!(i < chars.len(), "{file}: unterminated string literal");
                        let c = chars[i];
                        out.push(c);
                        i += 1;
                        if escaped {
                            escaped = false;
                        } else if c == '\\' {
                            escaped = true;
                        } else if c == '"' {
                            break;
                        }
                    }
                }
                _ => {
                    out.push(ch);
                    i += 1;
                }
            }
        }
        out
    }

    /// The comma-separated top-level arguments of one call or parameter list, given everything after
    /// its open paren. Bounded by the call's own bracket balance.
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
                '"' => {
                    current.push('"');
                    loop {
                        assert!(i < chars.len(), "{site}: unterminated string literal");
                        let c = chars[i];
                        i += 1;
                        current.push(c);
                        if c == '\\' {
                            i += 1;
                        } else if c == '"' {
                            break;
                        }
                    }
                }
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
                _ => current.push(ch),
            }
        }
        panic!("{site} is unterminated: no closing paren before end of file")
    }

    /// Every **call** to the free function `denoise::denoise_curated` in one module.
    ///
    /// The declaration (`fn denoise_curated(`) and `Pipeline`'s same-named *method*
    /// (`self.denoise_curated(`) are both skipped by their prefix — the method builds its own hook in
    /// its own body and is therefore already visible to the catalog's inventory.
    fn shared_denoise_calls(file: &str, source: &str) -> Vec<Vec<String>> {
        let code = code_only(file, source);
        let call = format!("{SHARED_DENOISE}(");
        let mut sites = Vec::new();
        let mut cursor = 0usize;
        while let Some(offset) = code[cursor..].find(&call) {
            let at = cursor + offset;
            let args_start = at + call.len();
            let before = &code[..at];
            if !before.ends_with('.') && !before.trim_end().ends_with("fn") {
                let site = format!("{file}: {SHARED_DENOISE} call #{}", sites.len());
                sites.push(call_arguments(&site, &code[args_start..]));
            }
            cursor = args_start;
        }
        sites
    }

    /// The preview argument's position is read out of `denoise_curated`'s own declaration, so a
    /// reordered or widened signature fails here instead of quietly moving the pin below onto a
    /// neighbouring argument.
    #[test]
    fn the_shared_curated_denoise_signature_pins_the_preview_argument_position() {
        let code = code_only("denoise.rs", include_str!("denoise.rs"));
        let declaration = format!("pub fn {SHARED_DENOISE}(");
        let at = code
            .find(&declaration)
            .expect("denoise.rs must declare the shared curated denoise helper");
        let parameters = call_arguments(
            "denoise.rs: the denoise_curated declaration",
            &code[at + declaration.len()..],
        );
        assert_eq!(
            parameters.len(),
            SHARED_DENOISE_ARITY,
            "parsed {parameters:?}"
        );
        assert_eq!(
            parameters[SHARED_DENOISE_PREVIEW_AT],
            "preview: Option<&candle_gen::preview::PreviewHook<'_>>",
            "parsed {parameters:?}"
        );
    }

    /// Every caller of `denoise::denoise_curated` in this crate, classified by the argument it passes
    /// in the preview slot — positionally, so the pin cannot be satisfied by the word appearing
    /// elsewhere in the call.
    ///
    /// The `#[cfg(test)]` rows in `denoise.rs` are listed rather than stripped: classifying **every**
    /// occurrence is what makes the shipped count exact, and a scan that guessed which ones were test
    /// code would be a second place to get that wrong.
    #[test]
    fn every_shipped_caller_of_the_shared_curated_denoise_passes_a_hook() {
        let mut inventory: Vec<(&str, String)> = Vec::new();
        for (file, source) in MODULES {
            for args in shared_denoise_calls(file, source) {
                assert_eq!(
                    args.len(),
                    SHARED_DENOISE_ARITY,
                    "{file}: expected {SHARED_DENOISE_ARITY} arguments with the preview argument at \
                     position {SHARED_DENOISE_PREVIEW_AT}, parsed {args:?}"
                );
                inventory.push((file, args[SHARED_DENOISE_PREVIEW_AT].clone()));
            }
        }
        let inventory: Vec<(&str, &str)> = inventory
            .iter()
            .map(|(file, argument)| (*file, argument.as_str()))
            .collect();
        assert_eq!(
            inventory,
            [
                // `denoise.rs`'s own structural rows: no request, no sink, nothing to emit into.
                ("denoise.rs", "None"),
                ("denoise.rs", "None"),
                // The one SHIPPED caller — the IP-adapter provider's curated lane. `None` here is a
                // dark render route that no other guard in this repo can see.
                ("ip_provider.rs", "Some(&preview)"),
            ],
            "the shared curated denoise gained, lost or re-classified a caller"
        );
    }
}

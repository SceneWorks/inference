//! Ideogram 4 text-to-image pipeline: Qwen3-VL text encode → flow-matching denoise → latent
//! de-normalize + unpatchify + VAE decode. Port of `Ideogram4Pipeline.__call__`.
//!
//! Two denoise modes share this pipeline, selected by whether the **unconditional** DiT is present:
//! * **Quality (asymmetric CFG, default)** — the conditional DiT runs over the full `[text ; image]`
//!   sequence; the unconditional DiT runs over the **image-only** slice with zeroed conditioning.
//!   Per step the velocities combine `v = g·pos_v + (1−g)·neg_v`.
//! * **Turbo (CFG-free single DiT, issue #488)** — `uncond` is `None` and the conditional DiT
//!   carries the ostris **TurboTime** LoRA; per step `v = pos_v` (no negative branch, guidance off),
//!   so a render costs one DiT forward over ~8 steps instead of two over ~48.
//!
//! Both Euler-step `z += v·(s−t)`. Tokenization (the Qwen3-VL chat template) is the caller's job —
//! `generate` takes `input_ids`.
//!
//! **Edit (img2img / mask inpaint, sc-6303/6330).** When an [`EditInit`] is supplied, the denoise
//! starts from a VAE-encoded source latent noised to a strength-derived step instead of pure noise
//! (img2img / Remix); an optional latent-grid mask additionally pins the keep region (mask 0) to the
//! source re-noised to each step's σ while regenerating the white region (mask 1) — the classic
//! masked-img2img inpaint on this same flow-match loop (no dedicated inpaint UNet). With no
//! `EditInit`, the path is byte-identical to the original text-to-image render.

use std::path::Path;

use mlx_rs::ops::{add, concatenate_axis, divide, multiply, split_sections, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use mlx_gen::gen_core;
use mlx_gen::gen_core::sampling::init_time_step;
use mlx_gen::image::{resize_lanczos_u8, resize_nearest_u8};
use mlx_gen::media::Image;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{flow_capture_plan, CancelFlag, Error, LatentDecoder, PreviewSink, Progress, Result};
use mlx_gen_flux2::{patchify_latents, Flux2Vae};

use crate::adapters::apply_ideogram_adapters;
use crate::config::{Ideogram4DitConfig, MAX_TEXT_TOKENS};
use crate::loader::{
    load_text_encoder, load_tokenizer, load_transformer, load_unconditional_transformer, load_vae,
};
use crate::scheduler::{make_step_intervals, preset_mu_std, LogitNormalSchedule};
use crate::text_encoder::Ideogram4TextEncoder;
use crate::transformer::Ideogram4Transformer;

/// `patch_size * ae_scale_factor` — height/width must be a multiple of this (=16).
pub const PATCH: u32 = 2;
pub const AE_SCALE: u32 = 8;
const IMAGE_POSITION_OFFSET: i32 = 65536;
const LLM_TOKEN_INDICATOR: i32 = 3;
const OUTPUT_IMAGE_INDICATOR: i32 = 2;
// `MAX_TEXT_TOKENS` (the reference `Ideogram4PipelineConfig.max_text_tokens`) is the single
// `config::MAX_TEXT_TOKENS` source (F-110) — a longer prompt is rejected by `_tokenize`.
/// Per-step guidance schedule tail: the reference `DEFAULT_GUIDANCE_SCHEDULE` drops to 3.0 for the
/// final `POLISH_STEPS` low-noise steps (the rest use the base guidance, typically 7.0).
const POLISH_STEPS: usize = 3;
const POLISH_GUIDANCE: f32 = 3.0;
/// Edit (img2img / mask inpaint) conditioning prepared **once per request** (seed-independent) by
/// [`Ideogram4Pipeline::prepare_edit`]: the BN-normalized packed source latent `[1, num_img, 128]`,
/// an optional latent-grid inpaint mask `[1, num_img, 1]` (1 = repaint, 0 = keep), and the img2img
/// strength. Passed to [`Ideogram4Pipeline::generate_edit_with_progress`] per seed.
pub struct EditInit {
    /// BN-normalized packed source latent `[1, num_img, 128]` (same space as the running `z`).
    pub z0: Array,
    /// Latent-grid inpaint mask `[1, num_img, 1]` (1.0 = repaint/white, 0.0 = keep/black). `None`
    /// for plain img2img (regenerate everywhere from the noised source).
    pub mask: Option<Array>,
    /// img2img strength in `(0, 1]` — fraction of the denoise executed from the noised source.
    pub strength: f32,
}

/// Forward evaluation seam for MLX Ideogram 4 flow-matching Euler denoising (`SW-6970`, `PIPE-7`).
///
/// Generic over conditional + optional unconditional velocity prediction so weights-free Tier 0
/// `DriverSupport` conformance tests can drive [`IdeogramEulerDriver`] over mocked forwards
/// without loading model weights.
pub trait IdeogramVelocityForward {
    /// Predict the conditional image-token velocity `[1, num_img, ch]` at clean-fraction time
    /// `t_val` and descending schedule index `schedule_step` (`num_run - 1` down to `run_from`).
    fn predict_cond(&self, z: &Array, t_val: f32, schedule_step: usize) -> Result<Array>;

    /// Predict the unconditional image-token velocity `[1, num_img, ch]` for asymmetric two-DiT
    /// CFG (`Some` in quality mode, `None` in CFG-free turbo mode).
    fn predict_uncond(&self, z: &Array, t_val: f32, schedule_step: usize) -> Result<Option<Array>>;
}

/// Typed [`FamilyDriver`] for MLX Ideogram 4 quality (asymmetric two-DiT CFG with polish tail)
/// and turbo (CFG-free single-DiT) flow-matching Euler denoising (`SW-6970`, `PIPE-7`, `PIPE-8`).
pub struct IdeogramEulerDriver<F> {
    schedule: LogitNormalSchedule,
    step_intervals: Vec<f64>,
    num_run: usize,
    run_from: usize,
    guidance: f32,
    forward: F,
    support: DriverSupport,
}

impl<F: IdeogramVelocityForward> IdeogramEulerDriver<F> {
    fn build_support(driver_id: &'static str) -> DriverSupport {
        DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons {
                renoise_from_clean:
                    "native MLX Ideogram Euler driver does not re-noise from clean boundary estimates",
                geometry_change:
                    "native MLX Ideogram Euler driver does not support geometry-changing phase boundaries",
                read_only_boundary_estimate:
                    "native MLX Ideogram Euler driver does not expose read-only boundary clean estimates",
                ..DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(
        driver_id: &'static str,
        schedule: LogitNormalSchedule,
        step_intervals: Vec<f64>,
        num_run: usize,
        run_from: usize,
        guidance: f32,
        forward: F,
    ) -> Result<Self> {
        if run_from > num_run || step_intervals.len() <= num_run {
            return Err(Error::Unsupported(format!(
                "{driver_id}: invalid Ideogram step range run_from={run_from}, num_run={num_run}, intervals={}",
                step_intervals.len()
            )));
        }
        Ok(Self {
            schedule,
            step_intervals,
            num_run,
            run_from,
            guidance,
            forward,
            support: Self::build_support(driver_id),
        })
    }

    /// Total executed outer steps (`num_run - run_from`).
    pub fn total_steps(&self) -> usize {
        self.num_run - self.run_from
    }
}

impl<F: IdeogramVelocityForward> FamilyDriver<Array> for IdeogramEulerDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> gen_core::Result<gen_core::BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        state.enter_slice(&input)?;
        let total_run = self.total_steps();
        if input.span.end > total_run {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: slice span {:?} exceeds {total_run} steps",
                self.support.driver_id, input.span
            )));
        }

        let mut z = match input.state {
            gen_core::BoundaryState::XSigma(x)
            | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };

        for outer_step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            // Ideogram walks `i = num_run - 1` down to `run_from` (inverted clean-fraction).
            let i = self.num_run - 1 - outer_step;
            let t_val = self.schedule.eval(self.step_intervals[i + 1]);
            let s_val = self.schedule.eval(self.step_intervals[i]);
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: t_val as f32,
                next_sigma: s_val as f32,
            };
            hooks.observe(&info, &z)?;

            let pos_v = self
                .forward
                .predict_cond(&z, t_val as f32, i)
                .map_err(gen_core::Error::from)?;
            let v = match self
                .forward
                .predict_uncond(&z, t_val as f32, i)
                .map_err(gen_core::Error::from)?
            {
                Some(neg_v) => {
                    // Per-step asymmetric CFG (`PIPE-8`): `v = gw·pos_v + (1−gw)·neg_v`. The
                    // reference drops to POLISH_GUIDANCE for the final POLISH_STEPS low-noise steps
                    // (`i ∈ {0, 1, 2}`); base `guidance` drives the main steps.
                    let gw_i = if i < POLISH_STEPS {
                        POLISH_GUIDANCE
                    } else {
                        self.guidance
                    };
                    gen_core::guidance::cfg_linear_blend(
                        &mlx_gen::MlxLatentOps,
                        &pos_v,
                        &neg_v,
                        gw_i,
                    )?
                }
                // Turbo (issue #488): CFG-free single DiT — velocity is `pos_v` directly.
                None => pos_v,
            };
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            let dt = Array::from_f32((s_val - t_val) as f32);
            let step_delta = multiply(&v, dt).map_err(|e| gen_core::Error::from(Error::from(e)))?;
            z = add(&z, &step_delta).map_err(|e| gen_core::Error::from(Error::from(e)))?;
            z = hooks.post_step(&info, z)?;
            hooks.materialize(&info, &z)?;
            state.record_outer_step(outer_step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(gen_core::BoundaryState::XSigma(z))
    }
}

struct ProductionIdeogramForward<'a> {
    cond: &'a Ideogram4Transformer,
    cond_prep: &'a crate::transformer::model::PreparedConditioning,
    neg: Option<(
        &'a Ideogram4Transformer,
        &'a crate::transformer::model::PreparedConditioning,
    )>,
    text_z_padding: &'a Array,
    num_text: i32,
}

impl IdeogramVelocityForward for ProductionIdeogramForward<'_> {
    fn predict_cond(&self, z: &Array, t_val: f32, _schedule_step: usize) -> Result<Array> {
        let t = Array::from_slice(&[t_val], &[1]);
        let pos_z = concatenate_axis(&[self.text_z_padding, z], 1)?;
        let pos_out = self
            .cond
            .forward_prepared(&pos_z, &t, self.cond_prep, None)?;
        // Image-token velocities = the `[num_text..seq)` suffix — a zero-copy strided split at the
        // fixed text/image boundary, vs. an arange `take_axis` gather rebuilt every step (F-152).
        Ok(split_sections(&pos_out, &[self.num_text], 1)?.swap_remove(1))
    }

    fn predict_uncond(
        &self,
        z: &Array,
        t_val: f32,
        _schedule_step: usize,
    ) -> Result<Option<Array>> {
        match self.neg {
            Some((uncond, neg_prep)) => {
                let t = Array::from_slice(&[t_val], &[1]);
                Ok(Some(uncond.forward_prepared(z, &t, neg_prep, None)?))
            }
            None => Ok(None),
        }
    }
}

/// Flow-matching interpolation `z = σ·clean + (1−σ)·noise`. **Ideogram's [`LogitNormalSchedule`] is
/// inverted from the usual flow-match σ**: `eval(0) ≈ t_max ≈ 0.999` is the *clean* end and
/// `eval(1) ≈ t_min ≈ 0.0001` is *pure noise* (the denoise loop inits from `random::normal` and the
/// first executed step's `t_val = eval(si[num_run])` is the small-σ noisy end). So a larger σ weights
/// the clean source more, the mirror of the fork's `add_noise_by_interpolation`.
fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    Ok(add(
        &multiply(clean, Array::from_f32(sigma))?,
        &multiply(noise, Array::from_f32(1.0 - sigma))?,
    )?)
}

/// Preprocess a source image onto the model's input grid: resize (Lanczos) to `width×height`,
/// normalize to `[-1,1]`, NHWC `[1,H,W,3]` f32. Mirrors flux2 `preprocess_ref_image`.
fn preprocess_source_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    // A 0×0 image with an empty buffer passes the length check but panics in the resampler —
    // reject it with the same typed error (F-084 class).
    if iw == 0 || ih == 0 {
        return Err(Error::Msg(format!(
            "ideogram edit: source image has a zero dimension ({iw}x{ih})"
        )));
    }
    if image.pixels.len()
        != mlx_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(Error::Msg(format!(
            "ideogram edit: source pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)?
    };
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// Build the latent-grid inpaint mask `[1, num_img, 1]` (f32; 1.0 = repaint/white, 0.0 = keep/black)
/// from a mask image: PIL-"L" luma → binarize at image res → nearest `patch·ae = 16×` downsample
/// (top-left of each 16×16 block, torch `nearest`'s `floor(dst·scale)`), flattened row-major to
/// match the image-token order (`j = h·grid_w + w`). Ideogram's token grid is `H/16 × W/16`, so the
/// downsample factor is 16 (vs 8 in sdxl `preprocess_mask`).
fn preprocess_mask_packed(mask: &Image, width: u32, height: u32) -> Result<Array> {
    let (w, h) = (width as usize, height as usize);
    // Request-supplied buffer: reject zero dims and a mismatched pixel buffer before the luma
    // conversion/resize index into it (F-002; the guard `preprocess_source_image` already has).
    let (mw, mh) = (mask.width as usize, mask.height as usize);
    if mw == 0 || mh == 0 || w == 0 || h == 0 {
        return Err(Error::Msg(format!(
            "ideogram edit: mask image has a zero dimension ({mw}x{mh} -> {w}x{h})"
        )));
    }
    if mask.pixels.len()
        != mlx_gen::gen_core::imageops::checked_image_buffer_len(mw, mh, 3).unwrap_or(usize::MAX)
    {
        return Err(Error::Msg(format!(
            "ideogram edit: mask pixel buffer {} != {mw}x{mh}x3",
            mask.pixels.len()
        )));
    }
    let patch = (PATCH * AE_SCALE) as usize; // 16
                                             // Nearest (not bicubic): a mask must not gain interpolated grays that flip the 0.5 binarize.
    let luma: Vec<u8> = if (mask.width as usize, mask.height as usize) == (w, h) {
        rgb_to_luma(&mask.pixels)
    } else {
        let resized = resize_nearest_u8(
            &mask.pixels,
            mask.height as usize,
            mask.width as usize,
            h,
            w,
        )?;
        let u8s: Vec<u8> = resized
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        rgb_to_luma(&u8s)
    };
    let (gh, gw) = (h / patch, w / patch);
    let mut packed = Vec::with_capacity(gh * gw);
    for ly in 0..gh {
        for lx in 0..gw {
            let v = luma[(ly * patch) * w + (lx * patch)]; // top-left of the block
            packed.push(if v as f32 / 255.0 >= 0.5 { 1.0f32 } else { 0.0 });
        }
    }
    Ok(Array::from_slice(&packed, &[1, (gh * gw) as i32, 1]))
}

/// PIL "L" grayscale luma: `round(R·299/1000 + G·587/1000 + B·114/1000)` per RGB pixel.
fn rgb_to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            let l = (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000;
            l.min(255) as u8
        })
        .collect()
}

/// The **text-encode phase** of an Ideogram 4 pipeline (sc-10840): the Qwen3-VL text encoder + the
/// tokenizer. The encoder is the phase-A component dropped first under
/// [`OffloadPolicy::Sequential`](mlx_gen::OffloadPolicy): tokenize → encode → **drop the TE** →
/// denoise/decode over the two DiTs + VAE, bounding peak unified memory to `max(TE, DiTs+VAE)`.
pub struct Ideogram4Text {
    te: Ideogram4TextEncoder,
    tok: TextTokenizer,
}

/// The **heavy render phase** of an Ideogram 4 pipeline (sc-10840): the conditional DiT, the optional
/// unconditional DiT (asymmetric-CFG negative branch; `None` in turbo), the VAE, and the DiT config.
/// Held after the text encoder is dropped under `Sequential`; both residencies run the identical
/// [`run_denoise_from_embeds`](Ideogram4Heavy::run_denoise_from_embeds) body, so a `Sequential` job is
/// byte-identical to `Resident`. The PiD overlay is NOT here — it is owned separately by the registry's
/// heavy bundle (mirroring lens), since the struct API threads it as a per-call parameter.
pub struct Ideogram4Heavy {
    cond: Ideogram4Transformer,
    /// The unconditional DiT (asymmetric-CFG negative branch). `None` in the **turbo** mode
    /// ([`load_turbo`](Self::load_turbo)) — the CFG-free single-DiT path runs the conditional DiT
    /// alone, halving resident memory.
    uncond: Option<Ideogram4Transformer>,
    vae: Flux2Vae,
    dit: Ideogram4DitConfig,
}

/// A loaded Ideogram 4 pipeline: the [`Ideogram4Text`] + [`Ideogram4Heavy`] phases (sc-10840 split)
/// held together. This is the direct struct API used by the crate's `#[ignore]`d real-weight tests;
/// the registry generator ([`crate::model`]) holds the two phases separately so it can drop the text
/// encoder mid-job for `Sequential` residency.
pub struct Ideogram4Pipeline {
    text: Ideogram4Text,
    heavy: Ideogram4Heavy,
}

impl Ideogram4Text {
    /// Load the Qwen3-VL text encoder + tokenizer (`text_encoder/` + `tokenizer/`) — the phase-A
    /// component dropped first under `Sequential`.
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            te: load_text_encoder(root)?,
            tok: load_tokenizer(root)?,
        })
    }

    /// Quantize the Qwen3-VL text encoder (group-wise affine Q4/Q8) after the dense load. Factored so
    /// the `Resident` and `Sequential` paths build a byte-identical encoder (quantization is
    /// deterministic).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.te.quantize(bits)
    }

    /// Tokenize a prompt to `input_ids` exactly as the reference `_tokenize`: wrap it in the
    /// Qwen3-VL single-user chat template ([`ChatTemplate::QwenInstruct`](mlx_gen::tokenizer::ChatTemplate::QwenInstruct))
    /// and encode with `add_special_tokens=false`. Rejects a prompt longer than `MAX_TEXT_TOKENS`.
    /// The prompt is the model's native **JSON caption** string (SceneWorks builds it); plain text
    /// is out-of-distribution.
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        let ids = self.tok.encode_chat_ids(prompt, false)?;
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(Error::Msg(format!(
                "prompt has {} tokens, exceeds max_text_tokens={MAX_TEXT_TOKENS}",
                ids.len()
            )));
        }
        Ok(ids)
    }

    /// Encode the (single, unpadded) prompt tokens to the Qwen3-VL prompt embeds `[1, num_text,
    /// llm_dim]` — the phase-A conditioning materialized before the TE is dropped under `Sequential`.
    /// Seed-independent (a forward pass, no RNG), so hoisting it out of the count loop is
    /// byte-identical to the pre-sc-10840 per-image re-encode.
    pub fn encode(&self, input_ids: &[i32]) -> Result<Array> {
        let num_text = input_ids.len() as i32;
        let ids = Array::from_slice(input_ids, &[1, num_text]);
        let attn = Array::from_slice(&vec![1i32; num_text as usize], &[1, num_text]);
        self.te.prompt_embeds(&ids, &attn) // [1, num_text, llm_dim]
    }
}

impl Ideogram4Heavy {
    /// Load the heavy render phase (2 DiTs + VAE) from a converted snapshot dir — the quality
    /// (asymmetric-CFG) mode.
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            cond: load_transformer(root)?,
            uncond: Some(load_unconditional_transformer(root)?),
            vae: load_vae(root)?,
            dit: Ideogram4DitConfig::v4(),
        })
    }

    /// Load the **turbo** heavy phase (issue #488): the conditional DiT + VAE, **without** the
    /// unconditional DiT. The caller applies the TurboTime LoRA via
    /// [`apply_adapters`](Self::apply_adapters) and runs the CFG-free path (single forward per step,
    /// guidance off). Skips loading the ~half-of-weights uncond DiT.
    pub fn load_turbo(root: &Path) -> Result<Self> {
        Ok(Self {
            cond: load_transformer(root)?,
            uncond: None,
            vae: load_vae(root)?,
            dit: Ideogram4DitConfig::v4(),
        })
    }

    /// `true` when this bundle runs the CFG-free single-DiT turbo path (no unconditional DiT).
    pub fn is_turbo(&self) -> bool {
        self.uncond.is_none()
    }

    /// Install adapters (the TurboTime LoRA, or a user Ideogram LoRA) onto the **conditional** DiT
    /// via the shared strict loader — errors on any unmatched target. Apply **after**
    /// [`quantize`](Self::quantize) (the residual is computed and added on top of the possibly
    /// quantized base, fork-faithfully). No-op for an empty spec list.
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<()> {
        if !specs.is_empty() {
            apply_ideogram_adapters(&mut self.cond, specs)?;
        }
        Ok(())
    }

    /// Quantize the heavy render components in place (group-wise affine Q4/Q8) after the dense load —
    /// the conditional DiT, the unconditional DiT (quality mode only), and the VAE — matching the
    /// flux2 family's `spec.quantize` semantics. Norms / tiny embeddings stay dense (each module's
    /// `quantize` decides). The text encoder quantizes in the phase-A [`Ideogram4Text::quantize`].
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.cond.quantize(bits)?;
        if let Some(uncond) = &mut self.uncond {
            uncond.quantize(bits)?;
        }
        self.vae.quantize(bits)?;
        Ok(())
    }

    /// VAE-encode a source image into the BN-normalized packed latent `[1, num_img, 128]` the
    /// denoise operates on — the exact inverse of `decode`'s de-normalize +
    /// unpatchify: resize → `encode_mean` → 2×2 patchify (Ideogram's `(ph,pw,c)` c-innermost order)
    /// → BN-normalize `(x − mean)/std`. Seed-independent; encode once per request.
    pub fn encode_init_latents(&self, image: &Image, height: u32, width: u32) -> Result<Array> {
        let patch = PATCH * AE_SCALE;
        let grid_h = (height / patch) as i32;
        let grid_w = (width / patch) as i32;
        let pre = preprocess_source_image(image, width, height)?; // [1, H, W, 3]
        let enc = self.vae.encode_mean(&pre)?; // [1, H/8, W/8, 32] = [1, gh·2, gw·2, 32]
                                               // Patchify to packed [1, L, 128] — inverse of decode's unpatchify (channels (ph, pw, c)).
        let packed = enc
            .reshape(&[1, grid_h, 2, grid_w, 2, 32])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[1, grid_h * grid_w, 128])?;
        // BN-normalize in packed NHWC — inverse of decode's `z·std + mean`.
        let (bn_std, bn_mean) = self.vae.bn_stats();
        let normed = divide(
            &subtract(&packed, &bn_mean.reshape(&[1, 1, 128])?)?,
            &bn_std.reshape(&[1, 1, 128])?,
        )?;
        Ok(normed)
    }

    /// Prepare the per-request [`EditInit`] (img2img / inpaint): VAE-encode the source once and
    /// build the optional latent-grid mask. The result is reused across the per-seed count loop.
    pub fn prepare_edit(
        &self,
        source: &Image,
        mask: Option<&Image>,
        strength: f32,
        height: u32,
        width: u32,
    ) -> Result<EditInit> {
        let z0 = self.encode_init_latents(source, height, width)?;
        let mask = match mask {
            Some(m) => Some(preprocess_mask_packed(m, width, height)?),
            None => None,
        };
        Ok(EditInit { z0, mask, strength })
    }

    /// PiD `from_ldm` early-stop (sc-8048) capture plan for one Ideogram generation.
    ///
    /// **Frame conversion.** Ideogram's [`LogitNormalSchedule`] is *inverted* from the usual flow-match
    /// σ (see `add_noise_by_interpolation`): `eval(si)` returns the **clean weight** (`eval(si[0])≈0.999`
    /// clean, `eval(si[num_run])≈0.0001` pure noise). PiD's [`flow_capture_plan`] wants the standard
    /// degrade σ (`0` = clean, `1` = noise) on a strictly *descending* schedule, so the executed-step
    /// trajectory maps to `σ_std = 1 − eval(si)`. The denoise runs `i = num_run−1 … 0`; the latent at the
    /// **start** of executed step `e` (`i = num_run−1−e`) sits at `σ_std = 1 − eval(si[num_run−e])`, so
    /// the descending schedule handed to the plan is `[1 − eval(si[num_run]), …, 1 − eval(si[0])]`
    /// (length `num_run + 1`, trailing ≈0). This is the exact σ trajectory
    /// [`Self::run_denoise_from_embeds`] traverses.
    ///
    /// Returns `Some((capture_sigma, run_from))` when an early stop is warranted: `capture_sigma` is the
    /// achieved standard-frame degrade σ to mint the PiD decoder at, and `run_from` is the loop's new
    /// lower bound (`run_denoise` iterates `i = num_run−1 … run_from`, so it executes `keep − 1` of the
    /// high-noise steps and stops with the latent at `capture_sigma`, skipping the clean polish tail).
    /// `None` → the clean σ=0 full-denoise path (`req.pid_capture_sigma` unset / no benefit) — the loop
    /// runs its full `0..num_run` range, byte-identical.
    pub fn fromldm_capture(
        &self,
        height: u32,
        width: u32,
        num_steps: usize,
        strength: Option<f32>,
        capture_sigma: Option<f32>,
    ) -> Option<(f32, usize)> {
        let num_run = match strength {
            Some(s) => init_time_step(num_steps, Some(s)),
            None => num_steps,
        };
        if num_run == 0 {
            return None;
        }
        let (mu_eff, std_eff) = preset_mu_std(num_steps);
        let schedule = LogitNormalSchedule::for_resolution(height, width, mu_eff, std_eff);
        let si = make_step_intervals(num_steps);
        // Descending standard-frame σ over the executed trajectory (high-noise first, trailing ≈0).
        let sigmas: Vec<f32> = (0..=num_run)
            .map(|e| 1.0 - schedule.eval(si[num_run - e]) as f32)
            .collect();
        let plan = flow_capture_plan(&sigmas, capture_sigma)?;
        // `plan.keep` counts schedule entries to keep (`keep − 1` executed steps). The loop runs
        // `i = num_run−1 … run_from`, i.e. `num_run − run_from` steps, so `run_from = num_run − (keep−1)`.
        let run_from = num_run.saturating_sub(plan.keep - 1);
        Some((plan.sigma, run_from))
    }

    /// The shared flow-matching denoise from PRE-ENCODED Qwen3-VL prompt embeds (sc-10840): the body
    /// behind the T2I (`edit == None`) and edit (`edit == Some`) paths. `te_out` is the phase-A
    /// [`Ideogram4Text::encode`] output `[1, num_text, llm_dim]` (hoisted out of this loop under
    /// `Sequential` so the encoder is dropped before the DiTs load). With `edit == None` the body is
    /// byte-identical to the original text-to-image render (pure-noise init, full step range, no mask
    /// blend). `run_from` is the PiD `from_ldm` early-stop loop lower bound (sc-8048): `0` runs the
    /// full range (byte-identical); a larger value stops the denoise early, leaving the latent at the
    /// capture σ (see [`fromldm_capture`](Self::fromldm_capture)).
    #[allow(clippy::too_many_arguments)]
    pub fn run_denoise_from_embeds(
        &self,
        te_out: &Array,
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
        edit: Option<&EditInit>,
        run_from: usize,
        pid: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        self.run_denoise_from_embeds_with_preview(
            te_out,
            height,
            width,
            num_steps,
            guidance,
            seed,
            edit,
            run_from,
            pid,
            cancel,
            on_progress,
            &PreviewSink::default(),
        )
    }

    /// Preview-aware sibling of [`Self::run_denoise_from_embeds`]. Ideogram owns this cadence
    /// because its reversed logit-normal Euler/inpaint loop does not route through the shared flow
    /// sampler. One frame is attempted at the start of each actually executed loop step.
    #[allow(clippy::too_many_arguments)]
    pub fn run_denoise_from_embeds_with_preview(
        &self,
        te_out: &Array,
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
        edit: Option<&EditInit>,
        run_from: usize,
        pid: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &PreviewSink,
    ) -> Result<Array> {
        let patch = PATCH * AE_SCALE;
        // Request-derived: reject as a typed error rather than abort the worker (F-020/L-A). The
        // RES_MIN/RES_MAX (256–2048) range is a Generator-layer guarantee (enforced by the shared
        // `Capabilities` min/max-size in the registered `validate` path, sc-6983); this lower-level
        // pipeline entry re-checks only the hard multiple-of-`patch` invariant it cannot recover from.
        if !height.is_multiple_of(patch) || !width.is_multiple_of(patch) {
            return Err(Error::Msg(format!(
                "ideogram: height/width must be multiples of {patch} (got {height}x{width})"
            )));
        }
        let grid_h = (height / patch) as i32;
        let grid_w = (width / patch) as i32;
        let num_img = grid_h * grid_w;
        // `num_text` is the phase-A embed length (single unpadded prompt → positions 0..num_text).
        let num_text = te_out.shape()[1];
        let seq = num_text + num_img;
        let llm_dim = self.dit.llm_features_dim;
        let ch = self.dit.in_channels;

        // ── Text conditioning: the pre-encoded Qwen3-VL embeds padded with image-token zeros ──
        let llm_features = concatenate_axis(&[te_out, &zeros(&[1, num_img, llm_dim])?], 1)?; // [1, seq, llm_dim]

        // ── Packed positions / segments / role indicators (host-built) ──
        let pack = Packing::build(num_text, grid_h, grid_w);
        let position_ids = Array::from_slice(&pack.position_ids, &[1, seq, 3]);
        let indicator = Array::from_slice(&pack.indicator, &[1, seq]);
        // Step-invariant conditioning (role masks + MRoPE) built ONCE per branch, not per step (F-029).
        // `segment_ids` are all `1` (a single packed segment), so the additive attention mask is
        // identically zero and is dropped (`None`) rather than built/added every forward.
        let cond_prep = self
            .cond
            .prepare(&llm_features, &position_ids, &indicator, 1, seq)?;
        // The unconditional (negative) branch runs over the image-only slice. Built only in quality
        // mode — turbo (`uncond=None`) skips it entirely, avoiding both the second DiT forward and
        // the large `neg_llm` zero tensor (num_img × 53248 f32, ~0.9 GB at 1024²).
        let neg = match self.uncond.as_ref() {
            Some(uncond) => {
                let neg_position_ids = Array::from_slice(&pack.neg_position_ids, &[1, num_img, 3]);
                let neg_indicator = Array::from_slice(&pack.neg_indicator, &[1, num_img]);
                // The uncond branch's `indicator` is image-only, so its LLM role mask is identically
                // zero → the projected LLM stream is provably zero. `prepare` still runs the projection
                // once (bit-identical, and folded into `neg_prep`), removing it from the per-step loop.
                let neg_llm = zeros(&[1, num_img, llm_dim])?;
                let neg_prep =
                    uncond.prepare(&neg_llm, &neg_position_ids, &neg_indicator, 1, num_img)?;
                Some((uncond, neg_prep))
            }
            None => None,
        };

        // ── Flow-matching schedule (mu/std from the V4 preset for this step count) ──
        // (mu, std) come from the reference V4 preset for this step count (NOT a caller-supplied
        // value): the schedule is fully determined by the step count + resolution, so there is no
        // `mu` knob on the public API (sc-6944 dropped the inert parameter).
        let (mu_eff, std_eff) = preset_mu_std(num_steps);
        let schedule = LogitNormalSchedule::for_resolution(height, width, mu_eff, std_eff);
        let si = make_step_intervals(num_steps);

        // Edit (img2img / inpaint): run only `num_run = floor(steps·strength)` of the reversed loop,
        // which skips the noisiest (smallest-σ) leading steps and starts from the source noised to
        // σ = schedule.eval(si[num_run]) (Ideogram's schedule is inverted — larger σ = cleaner, so a
        // larger `num_run`/strength → a smaller start σ → more change). T2I runs the full range from
        // pure noise. `init_time_step` floors a positive strength to ≥1 step.
        let num_run = match edit {
            Some(e) => init_time_step(num_steps, Some(e.strength)),
            None => num_steps,
        };

        // ── Init: always draw the noise (identical RNG stream); blend with the source for an edit ──
        let key = random::key(seed)?;
        let noise = random::normal::<f32>(&[1, num_img, ch], None, None, Some(&key))?;
        let z0 = match edit {
            Some(e) => {
                add_noise_by_interpolation(&e.z0, &noise, schedule.eval(si[num_run]) as f32)?
            }
            None => noise.clone(),
        };
        let text_z_padding = zeros(&[1, num_text, ch])?;

        // ── Flow-matching Euler denoise (high → low noise) via typed FamilyDriver (`SW-6970`) ──
        // PiD `from_ldm` early-stop (sc-8048): stop at `run_from` instead of `0`, so only the
        // `num_run − run_from` high-noise steps run and the latent is left at the capture σ (the clean
        // low-noise polish tail `i < run_from` is skipped). `run_from == 0` → the full range (byte-
        // identical clean path). `total_run` counts the executed steps so progress still reaches 100%.
        let total_run = num_run - run_from;
        let preview_sigmas: Vec<f32> = (run_from..=num_run)
            .rev()
            .map(|step| schedule.eval(si[step]) as f32)
            .collect();
        let previews = mlx_gen::preview::PreviewCounter::new(&preview_sigmas);
        let cfg_active = neg.is_some();
        let driver = IdeogramEulerDriver::new(
            if cfg_active {
                "mlx_ideogram_euler_quality"
            } else {
                "mlx_ideogram_euler_turbo"
            },
            schedule,
            si.clone(),
            num_run,
            run_from,
            guidance,
            ProductionIdeogramForward {
                cond: &self.cond,
                cond_prep: &cond_prep,
                neg: neg.as_ref().map(|(u, p)| (*u, p)),
                text_z_padding: &text_z_padding,
                num_text,
            },
        )?;
        let mut driver_state = DriverRunState::new();
        let mut on_observe = |info: &DriverStepInfo, latent: &Array| -> gen_core::Result<()> {
            mlx_gen_flux2::preview::emit_ideogram_preview(
                preview,
                &previews,
                &preview_sigmas,
                info.sigma,
                latent,
                grid_h,
                grid_w,
                &self.vae,
            );
            Ok(())
        };
        let mut mask_blend_hook = |info: &DriverStepInfo,
                                   latent: Array|
         -> gen_core::Result<Array> {
            // Inpaint: pin the keep region (mask 0) to the source re-noised to this step's σ
            // (= s_val = info.next_sigma, the post-step time) and regenerate the white region
            // (mask 1). At the final step s≈0 the keep region is the clean source. Mirrors sdxl
            // `InpaintBlend`; draws no RNG, so an all-white mask reduces to plain img2img.
            if let Some(e) = edit {
                if let Some(mask) = &e.mask {
                    let init_noised = add_noise_by_interpolation(&e.z0, &noise, info.next_sigma)
                        .map_err(gen_core::Error::from)?;
                    let fg = multiply(mask, &latent)
                        .map_err(|err| gen_core::Error::from(Error::from(err)))?;
                    let inv_mask = subtract(Array::from_f32(1.0), mask)
                        .map_err(|err| gen_core::Error::from(Error::from(err)))?;
                    let bg = multiply(&inv_mask, &init_noised)
                        .map_err(|err| gen_core::Error::from(Error::from(err)))?;
                    return add(&fg, &bg).map_err(|err| gen_core::Error::from(Error::from(err)));
                }
            }
            Ok(latent)
        };
        let mut on_materialize = |_info: &DriverStepInfo, latent: &Array| -> gen_core::Result<()> {
            eval([latent]).map_err(|err| gen_core::Error::from(Error::from(err)))?;
            Ok(())
        };
        let mut hooks = StepHooks::new(on_progress);
        hooks.on_observe = Some(&mut on_observe);
        if edit.is_some_and(|e| e.mask.is_some()) {
            hooks.on_post_step = Some(&mut mask_blend_hook);
        }
        hooks.on_materialize = Some(&mut on_materialize);
        let slice = SliceSpec {
            geometry: Some((width, height)),
            ..SliceSpec::full(total_run)
        };
        let out = run_mlx_driver_slices(
            &driver,
            [slice],
            gen_core::BoundaryState::XSigma(z0),
            total_run,
            seed,
            cancel,
            None,
            &mut driver_state,
            &mut hooks,
            |_, out, _| Ok(out),
        )?;
        let z = match out {
            gen_core::BoundaryState::XSigma(x)
            | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };

        mlx_gen::image_execution::sampled_native_with_rng(
            "ideogram-euler-v4",
            "clean-fraction",
            z.dtype(),
            &preview_sigmas,
            seed,
            "mlx-keyed-normal-f32-v1;initial=output-seed;no-step-draws",
        );
        let consumed_f64: Vec<_> = (run_from..=num_run)
            .rev()
            .map(|step| schedule.eval(si[step]))
            .collect();
        mlx_gen::image_execution::choice(
            "ideogram.f64_schedule",
            mlx_gen::gen_core::generator::ExecutionFact::resolved(
                serde_json::to_value(
                    mlx_gen::gen_core::generator::NumericScheduleDigest::f64_values(&consumed_f64)
                        .map_err(|error| Error::Msg(error.to_string()))?,
                )
                .map_err(|error| Error::Msg(error.to_string()))?,
            ),
        );
        mlx_gen::image_execution::scalar("ideogram.guidance", guidance);
        mlx_gen::image_execution::policy_debug("ideogram.cfg_active", &cfg_active);
        mlx_gen::image_execution::policy_debug("ideogram.polish_steps", &POLISH_STEPS);
        mlx_gen::image_execution::scalar("ideogram.polish_guidance", POLISH_GUIDANCE);
        mlx_gen::image_execution::policy_debug(
            "ideogram.mask_blend",
            &edit.is_some_and(|edit| edit.mask.is_some()),
        );
        on_progress(Progress::Decoding);
        self.decode(&z, grid_h, grid_w, pid)
    }

    /// De-normalize → unpatchify → VAE decode → RGB u8 `[H, W, 3]`. With a PiD overlay (`pid = Some`,
    /// epic 7840 sc-7847) the order-canonical raw 32-ch latent is re-packed into the FLUX-canonical
    /// space and 4×-super-resolved instead (returning a `4H × 4W` image).
    fn decode(
        &self,
        z: &Array,
        grid_h: i32,
        grid_w: i32,
        pid: Option<&dyn LatentDecoder>,
    ) -> Result<Array> {
        // The shared VAE owns both the BatchNorm de-normalization and Ideogram's distinct
        // `(ph,pw,c)` unpatch order. The result is the order-canonical raw 32-channel VAE latent.
        let latent = self.vae.unpack_ideogram_packed_latents(z, grid_h, grid_w)?;

        let decoded = match pid {
            // PiD 4× super-resolving decode (sc-7847). Ideogram's DiT packs the 128 channels as
            // (ph,pw,c), but the `flux2` PiD student was trained on the diffusers (c,ph,pw) packing.
            // Route through the order-canonical raw 32-ch latent (`latent`, the same tensor the VAE
            // decodes correctly): NCHW → FLUX-order `patchify_latents` → BN-normalize → the student's
            // input space (identical to what flux2/lens hand PiD). Output NCHW [1,3,4H,4W] → NHWC.
            Some(d) => {
                mlx_gen::ensure_decoder_layout(
                    Some(&mlx_gen::gen_core::FLUX2_PACKED_LATENT_SPACE),
                    d,
                )?;
                let nchw = latent.transpose_axes(&[0, 3, 1, 2])?; // [1,32,gh·2,gw·2]
                let patched = patchify_latents(&nchw)?; // [1,128,gh,gw] (c,ph,pw)
                let normed = self.vae.bn_normalize_nchw(&patched)?; // BN-normalize (FLUX order)
                d.decode(&normed)?.transpose_axes(&[0, 2, 3, 1])? // [1,3,4H,4W] → [1,4H,4W,3]
            }
            None => self.vae.decode(&latent)?, // [1, H, W, 3] f32, ~[-1,1]
        };
        let sh = decoded.shape();
        let (h, w) = (sh[1], sh[2]);
        let clamped = mlx_rs::ops::clip(&decoded, (&Array::from_f32(-1.0), &Array::from_f32(1.0)))?;
        let scaled = multiply(
            &add(&clamped, Array::from_f32(1.0))?,
            Array::from_f32(127.5),
        )?;
        let u8img = mlx_rs::ops::round(&scaled, None)?.as_dtype(Dtype::Uint8)?;
        Ok(u8img.reshape(&[h, w, 3])?)
    }
}

// The composing struct API (sc-10840): `Ideogram4Pipeline` holds the two phases and delegates every
// public method — tokenize/encode to [`Ideogram4Text`], everything DiT/VAE to [`Ideogram4Heavy`], and
// the `generate*` fronts to `text.encode` → `heavy.run_denoise_from_embeds`. The registry generator
// ([`crate::model`]) holds the two phases in a `Residency` instead, dropping the text encoder between
// them under `Sequential`. Both routes share the phase types, so their output is byte-identical.
impl Ideogram4Pipeline {
    /// Load all components (2 DiTs + Qwen3-VL text encoder + VAE + tokenizer) from a converted
    /// snapshot dir — the quality (asymmetric-CFG) mode.
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            text: Ideogram4Text::load(root)?,
            heavy: Ideogram4Heavy::load(root)?,
        })
    }

    /// Load the **turbo** pipeline (issue #488): the conditional DiT + Qwen3-VL TE + VAE + tokenizer,
    /// **without** the unconditional DiT. The caller applies the TurboTime LoRA via
    /// [`apply_adapters`](Self::apply_adapters) and runs the CFG-free [`generate`](Self::generate)
    /// path. Skips loading the ~half-of-weights uncond DiT.
    pub fn load_turbo(root: &Path) -> Result<Self> {
        Ok(Self {
            text: Ideogram4Text::load(root)?,
            heavy: Ideogram4Heavy::load_turbo(root)?,
        })
    }

    /// `true` when this pipeline runs the CFG-free single-DiT turbo path (no unconditional DiT).
    pub fn is_turbo(&self) -> bool {
        self.heavy.is_turbo()
    }

    /// Install adapters (the TurboTime LoRA, or a user Ideogram LoRA) onto the conditional DiT.
    /// Apply **after** [`quantize`](Self::quantize). No-op for an empty spec list.
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<()> {
        self.heavy.apply_adapters(specs)
    }

    /// Quantize the whole model in place (Q4/Q8) after the dense load — the two DiTs + VAE (heavy)
    /// and the Qwen3-VL text encoder (phase A). Each component's quantize is independent and
    /// deterministic, so this matches the pre-split whole-model quantize.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.heavy.quantize(bits)?;
        self.text.quantize(bits)?;
        Ok(())
    }

    /// Tokenize a prompt to `input_ids` (the Qwen3-VL chat template). Delegates to [`Ideogram4Text`].
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        self.text.tokenize(prompt)
    }

    /// VAE-encode a source image into the BN-normalized packed latent. Delegates to [`Ideogram4Heavy`].
    pub fn encode_init_latents(&self, image: &Image, height: u32, width: u32) -> Result<Array> {
        self.heavy.encode_init_latents(image, height, width)
    }

    /// Prepare the per-request [`EditInit`] (img2img / inpaint). Delegates to [`Ideogram4Heavy`].
    pub fn prepare_edit(
        &self,
        source: &Image,
        mask: Option<&Image>,
        strength: f32,
        height: u32,
        width: u32,
    ) -> Result<EditInit> {
        self.heavy
            .prepare_edit(source, mask, strength, height, width)
    }

    /// PiD `from_ldm` early-stop (sc-8048) capture plan. Delegates to [`Ideogram4Heavy`].
    pub fn fromldm_capture(
        &self,
        height: u32,
        width: u32,
        num_steps: usize,
        strength: Option<f32>,
        capture_sigma: Option<f32>,
    ) -> Option<(f32, usize)> {
        self.heavy
            .fromldm_capture(height, width, num_steps, strength, capture_sigma)
    }

    /// Generate one image. `input_ids`: the chat-templated prompt tokens. Returns an RGB `[H, W, 3]`
    /// `uint8` array. No progress/cancellation — see
    /// [`generate_with_progress`](Self::generate_with_progress).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
    ) -> Result<Array> {
        self.generate_with_progress(
            input_ids,
            height,
            width,
            num_steps,
            guidance,
            seed,
            None, // native VAE decode (no PiD overlay)
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// [`generate`](Self::generate) with cooperative cancellation + step/decode progress. Encodes the
    /// Qwen3-VL embeds once ([`Ideogram4Text::encode`]) then denoises/decodes over the heavy bundle
    /// ([`Ideogram4Heavy::run_denoise_from_embeds`]). Pure text-to-image (no edit conditioning).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_progress(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
        pid: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let te_out = self.text.encode(input_ids)?;
        self.heavy.run_denoise_from_embeds(
            &te_out,
            height,
            width,
            num_steps,
            guidance,
            seed,
            None,
            0, // full denoise range (no PiD early-stop)
            pid,
            cancel,
            on_progress,
        )
    }

    /// PiD `from_ldm` early-stop (sc-8048) T2I entry: as
    /// [`generate_with_progress`](Self::generate_with_progress) but stops the denoise at `run_from`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_progress_from(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
        run_from: usize,
        pid: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let te_out = self.text.encode(input_ids)?;
        self.heavy.run_denoise_from_embeds(
            &te_out,
            height,
            width,
            num_steps,
            guidance,
            seed,
            None,
            run_from,
            pid,
            cancel,
            on_progress,
        )
    }

    /// [`generate_with_progress`](Self::generate_with_progress) for an **edit** (img2img / mask
    /// inpaint, sc-6303/6330).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_edit_with_progress(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
        edit: &EditInit,
        pid: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let te_out = self.text.encode(input_ids)?;
        self.heavy.run_denoise_from_embeds(
            &te_out,
            height,
            width,
            num_steps,
            guidance,
            seed,
            Some(edit),
            0, // full denoise range (no PiD early-stop)
            pid,
            cancel,
            on_progress,
        )
    }

    /// PiD `from_ldm` early-stop (sc-8048) edit entry: as
    /// [`generate_edit_with_progress`](Self::generate_edit_with_progress) but stops at `run_from`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_edit_with_progress_from(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        seed: u64,
        edit: &EditInit,
        run_from: usize,
        pid: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let te_out = self.text.encode(input_ids)?;
        self.heavy.run_denoise_from_embeds(
            &te_out,
            height,
            width,
            num_steps,
            guidance,
            seed,
            Some(edit),
            run_from,
            pid,
            cancel,
            on_progress,
        )
    }
}

/// Device-side f32 zeros of the given shape (F-028). Previously built a host `vec![0f32; n]` and
/// copied it up — at 1024² the LLM-feature padding + `neg_llm` are `num_img × 53248` f32 (~0.9 GB
/// each, ×2 in quality mode), so materializing them on host was hundreds of MB of pointless traffic
/// per render. `mlx_rs::ops::zeros` allocates them on device with no host round-trip.
fn zeros(shape: &[i32]) -> Result<Array> {
    Ok(mlx_rs::ops::zeros::<f32>(shape)?)
}

/// Host-built packed sequence metadata: text tokens (`LLM`) then image tokens (`IMAGE`).
struct Packing {
    position_ids: Vec<i32>,
    indicator: Vec<i32>,
    neg_position_ids: Vec<i32>,
    neg_indicator: Vec<i32>,
}

impl Packing {
    fn build(num_text: i32, grid_h: i32, grid_w: i32) -> Self {
        let num_img = grid_h * grid_w;
        let mut position_ids = Vec::new();
        let mut indicator = Vec::new();
        for i in 0..num_text {
            position_ids.extend_from_slice(&[i, i, i]);
            indicator.push(LLM_TOKEN_INDICATOR);
        }
        let mut neg_position_ids = Vec::new();
        for j in 0..num_img {
            let (h, w) = (j / grid_w, j % grid_w);
            // Reference `_prepare_ids`: image positions are `[t, h, w] + IMAGE_POSITION_OFFSET` on
            // ALL THREE axes (`t_idx` is 0, so t = OFFSET). The offset keeps image positions disjoint
            // from the text positions (0..num_text) — leaving t=0 collides with the text t-axis and
            // corrupts the text→image MRoPE cross-attention (first 24 dims).
            let p = [
                IMAGE_POSITION_OFFSET,
                h + IMAGE_POSITION_OFFSET,
                w + IMAGE_POSITION_OFFSET,
            ];
            position_ids.extend_from_slice(&p);
            neg_position_ids.extend_from_slice(&p);
            indicator.push(OUTPUT_IMAGE_INDICATOR);
        }
        // NOTE: the packed sequence is a single segment (segment_ids ≡ 1), so no segment tensor is
        // stored — the DiT denoise runs with `mask = None` (F-029). The public `Ideogram4Transformer::
        // forward` still accepts an explicit `segment_ids` for the parity golden.
        Self {
            position_ids,
            indicator,
            neg_position_ids,
            neg_indicator: vec![OUTPUT_IMAGE_INDICATOR; num_img as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-002: a request-supplied mask with a short pixel buffer must be a typed error, not an OOB
    /// panic in the luma indexing (same-size path) or the resampler (resize path).
    #[test]
    fn preprocess_mask_packed_rejects_short_buffer() {
        // Same-size path: claims 32×32 but carries 8 bytes.
        let bad = Image {
            width: 32,
            height: 32,
            pixels: vec![0u8; 8],
        };
        let e = preprocess_mask_packed(&bad, 32, 32)
            .unwrap_err()
            .to_string();
        assert!(e.contains("mask pixel buffer"), "unexpected error: {e}");

        // Resize path: 16×16 claimed, short buffer, target 32×32.
        let e = preprocess_mask_packed(
            &Image {
                width: 16,
                height: 16,
                pixels: vec![0u8; 5],
            },
            32,
            32,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("mask pixel buffer"), "unexpected error: {e}");
    }

    /// F-002: a 0×0 mask (empty buffer passes a bare length check) must also be a typed error.
    #[test]
    fn preprocess_mask_packed_rejects_zero_dims() {
        let zero = Image {
            width: 0,
            height: 0,
            pixels: vec![],
        };
        let e = preprocess_mask_packed(&zero, 32, 32)
            .unwrap_err()
            .to_string();
        assert!(e.contains("zero dimension"), "unexpected error: {e}");
    }

    /// The valid path still binarizes + nearest-downsamples onto the 16× token grid.
    #[test]
    fn preprocess_mask_packed_binarizes_on_token_grid() {
        // 32×32: left half white, right half black → 2×2 grid rows [1, 0] flattened.
        let (w, h) = (32u32, 32u32);
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = if x < 16 { 255u8 } else { 0u8 };
                pixels.extend_from_slice(&[v, v, v]);
            }
        }
        let m = Image {
            width: w,
            height: h,
            pixels,
        };
        let a = preprocess_mask_packed(&m, w, h).unwrap();
        assert_eq!(a.shape(), &[1, 4, 1]);
        assert_eq!(a.as_slice::<f32>(), &[1.0, 0.0, 1.0, 0.0]);
    }

    /// F-084 class: a 0×0 source image (empty buffer passes the length check) must be a typed
    /// error instead of panicking in the resampler.
    #[test]
    fn preprocess_source_image_rejects_zero_dims() {
        let zero = Image {
            width: 0,
            height: 0,
            pixels: vec![],
        };
        let e = preprocess_source_image(&zero, 32, 32)
            .unwrap_err()
            .to_string();
        assert!(e.contains("zero dimension"), "unexpected error: {e}");
    }

    struct MockIdeogramForward {
        with_uncond: bool,
    }

    impl IdeogramVelocityForward for MockIdeogramForward {
        fn predict_cond(&self, z: &Array, t_val: f32, schedule_step: usize) -> Result<Array> {
            let scale = Array::from_f32(0.15 + 0.05 * t_val + 0.01 * (schedule_step as f32));
            let bias = Array::from_f32(0.25);
            Ok(add(multiply(z, &scale)?, &bias)?)
        }

        fn predict_uncond(
            &self,
            z: &Array,
            t_val: f32,
            schedule_step: usize,
        ) -> Result<Option<Array>> {
            if !self.with_uncond {
                return Ok(None);
            }
            let scale = Array::from_f32(-0.08 + 0.02 * t_val - 0.005 * (schedule_step as f32));
            let bias = Array::from_f32(-0.1);
            Ok(Some(add(multiply(z, &scale)?, &bias)?))
        }
    }

    fn mlx_arrays_bit_equal(a: &Array, b: &Array) -> bool {
        let a = mlx_gen::array::contiguous(a).unwrap();
        let b = mlx_gen::array::contiguous(b).unwrap();
        eval([&a, &b]).unwrap();
        let sa = a.as_slice::<f32>();
        let sb = b.as_slice::<f32>();
        sa.len() == sb.len() && sa.iter().zip(sb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn ideogram_euler_driver_quality_passes_tier0_conformance() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let (mu, std) = preset_mu_std(6);
        let schedule = LogitNormalSchedule::for_resolution(512, 512, mu, std);
        let si = make_step_intervals(6);
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 6,
            split_at: 3,
            seed: 42,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: || {
                Box::new(
                    IdeogramEulerDriver::new(
                        "mlx_ideogram_euler_quality",
                        schedule,
                        si.clone(),
                        6,
                        0,
                        7.0,
                        MockIdeogramForward { with_uncond: true },
                    )
                    .unwrap(),
                )
            },
            make_initial: || Array::from_slice(&[0.4f32, -0.7, 0.25, 1.1], &[1, 2, 2]),
            latents_equal: mlx_arrays_bit_equal,
        })
        .expect("MLX Ideogram quality driver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_carry_refusal_or_support);
        assert!(report.verified_reset_preserves_draws);
        assert!(report.verified_boundary_estimate_readonly_or_refusal);
        assert!(report.verified_renoise_and_resample_guards);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn ideogram_euler_driver_turbo_passes_tier0_conformance() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let (mu, std) = preset_mu_std(6);
        let schedule = LogitNormalSchedule::for_resolution(512, 512, mu, std);
        let si = make_step_intervals(6);
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 6,
            split_at: 2,
            seed: 99,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: || {
                Box::new(
                    IdeogramEulerDriver::new(
                        "mlx_ideogram_euler_turbo",
                        schedule,
                        si.clone(),
                        6,
                        0,
                        7.0,
                        MockIdeogramForward { with_uncond: false },
                    )
                    .unwrap(),
                )
            },
            make_initial: || Array::from_slice(&[-0.3f32, 0.8, -0.5, 0.2], &[1, 2, 2]),
            latents_equal: mlx_arrays_bit_equal,
        })
        .expect("MLX Ideogram turbo driver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn ideogram_euler_driver_preserves_polish_mask_blend_preview_and_pid_early_stop_bit_exact() {
        let num_steps = 6usize;
        let guidance = 7.0f32;
        let (mu, std) = preset_mu_std(num_steps);
        let schedule = LogitNormalSchedule::for_resolution(512, 512, mu, std);
        let si = make_step_intervals(num_steps);
        let z0_init = Array::from_slice(&[0.5f32, -0.25, 0.75, -1.0], &[1, 2, 2]);
        let clean = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[1, 2, 2]);
        let noise = Array::from_slice(&[-0.6f32, 0.9, -0.2, 0.5], &[1, 2, 2]);
        let mask = Array::from_slice(&[1.0f32, 0.0], &[1, 2, 1]);
        let mock = MockIdeogramForward { with_uncond: true };

        // 1. Quality + inpaint mask blend over full 6 steps (exercises main guidance=7.0 on
        //    i={5,4,3} and POLISH_GUIDANCE=3.0 on i={2,1,0}) vs direct pre-migration reference.
        let mut expected = z0_init.clone();
        let mut expected_sigmas = Vec::new();
        for i in (0..num_steps).rev() {
            let t_val = schedule.eval(si[i + 1]);
            let s_val = schedule.eval(si[i]);
            expected_sigmas.push(t_val as f32);
            let pos_v = mock.predict_cond(&expected, t_val as f32, i).unwrap();
            let neg_v = mock
                .predict_uncond(&expected, t_val as f32, i)
                .unwrap()
                .unwrap();
            let gw = if i < POLISH_STEPS {
                POLISH_GUIDANCE
            } else {
                guidance
            };
            let v = add(
                multiply(&pos_v, Array::from_f32(gw)).unwrap(),
                multiply(&neg_v, Array::from_f32(1.0 - gw)).unwrap(),
            )
            .unwrap();
            expected = add(
                &expected,
                multiply(&v, Array::from_f32((s_val - t_val) as f32)).unwrap(),
            )
            .unwrap();
            let init_noised = add_noise_by_interpolation(&clean, &noise, s_val as f32).unwrap();
            expected = add(
                multiply(&mask, &expected).unwrap(),
                multiply(subtract(Array::from_f32(1.0), &mask).unwrap(), &init_noised).unwrap(),
            )
            .unwrap();
            eval([&expected]).unwrap();
        }

        let driver = IdeogramEulerDriver::new(
            "mlx_ideogram_euler_quality",
            schedule,
            si.clone(),
            num_steps,
            0,
            guidance,
            MockIdeogramForward { with_uncond: true },
        )
        .unwrap();
        let cancel = CancelFlag::new();
        let mut state = DriverRunState::new();
        let mut observed_sigmas = Vec::new();
        let mut progress_seen = Vec::new();
        let mut on_progress = |p: Progress| progress_seen.push(p);
        let mut on_observe = |info: &DriverStepInfo, _z: &Array| -> gen_core::Result<()> {
            observed_sigmas.push(info.sigma);
            Ok(())
        };
        let mut on_post_step = |info: &DriverStepInfo, latent: Array| -> gen_core::Result<Array> {
            let init_noised = add_noise_by_interpolation(&clean, &noise, info.next_sigma).unwrap();
            let fg = multiply(&mask, &latent).unwrap();
            let inv = subtract(Array::from_f32(1.0), &mask).unwrap();
            let bg = multiply(&inv, &init_noised).unwrap();
            Ok(add(&fg, &bg).unwrap())
        };
        let mut on_materialize = |_info: &DriverStepInfo, latent: &Array| -> gen_core::Result<()> {
            eval([latent]).unwrap();
            Ok(())
        };
        let mut hooks = StepHooks::new(&mut on_progress);
        hooks.on_observe = Some(&mut on_observe);
        hooks.on_post_step = Some(&mut on_post_step);
        hooks.on_materialize = Some(&mut on_materialize);

        let out = run_mlx_driver_slices(
            &driver,
            [SliceSpec::full(num_steps)],
            gen_core::BoundaryState::XSigma(z0_init.clone()),
            num_steps,
            11,
            &cancel,
            None,
            &mut state,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let got = match out {
            gen_core::BoundaryState::XSigma(x) => x,
            _ => panic!("expected XSigma"),
        };
        assert!(
            mlx_arrays_bit_equal(&got, &expected),
            "MLX Ideogram driver must be bit-identical to pre-migration loop"
        );
        assert_eq!(observed_sigmas, expected_sigmas);
        assert_eq!(progress_seen.len(), num_steps);

        // 2. PiD `run_from = 2` early-stop executes only steps i={5,4,3,2} (4 steps).
        let run_from = 2usize;
        let total_run = num_steps - run_from;
        let mut expected_pid = z0_init.clone();
        for i in (run_from..num_steps).rev() {
            let t_val = schedule.eval(si[i + 1]);
            let s_val = schedule.eval(si[i]);
            let pos_v = mock.predict_cond(&expected_pid, t_val as f32, i).unwrap();
            let neg_v = mock
                .predict_uncond(&expected_pid, t_val as f32, i)
                .unwrap()
                .unwrap();
            let gw = if i < POLISH_STEPS {
                POLISH_GUIDANCE
            } else {
                guidance
            };
            let v = add(
                multiply(&pos_v, Array::from_f32(gw)).unwrap(),
                multiply(&neg_v, Array::from_f32(1.0 - gw)).unwrap(),
            )
            .unwrap();
            expected_pid = add(
                &expected_pid,
                multiply(&v, Array::from_f32((s_val - t_val) as f32)).unwrap(),
            )
            .unwrap();
        }
        let pid_driver = IdeogramEulerDriver::new(
            "mlx_ideogram_euler_quality",
            schedule,
            si,
            num_steps,
            run_from,
            guidance,
            MockIdeogramForward { with_uncond: true },
        )
        .unwrap();
        let mut pid_state = DriverRunState::new();
        let mut pid_progress = Vec::new();
        let mut on_pid_progress = |p: Progress| pid_progress.push(p);
        let mut pid_hooks = StepHooks::new(&mut on_pid_progress);
        let pid_out = run_mlx_driver_slices(
            &pid_driver,
            [SliceSpec::full(total_run)],
            gen_core::BoundaryState::XSigma(z0_init),
            total_run,
            11,
            &cancel,
            None,
            &mut pid_state,
            &mut pid_hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let got_pid = match pid_out {
            gen_core::BoundaryState::XSigma(x) => x,
            _ => panic!("expected XSigma"),
        };
        assert!(mlx_arrays_bit_equal(&got_pid, &expected_pid));
        assert_eq!(
            pid_progress,
            vec![
                Progress::Step {
                    current: 1,
                    total: 4
                },
                Progress::Step {
                    current: 2,
                    total: 4
                },
                Progress::Step {
                    current: 3,
                    total: 4
                },
                Progress::Step {
                    current: 4,
                    total: 4
                },
            ]
        );
    }
}

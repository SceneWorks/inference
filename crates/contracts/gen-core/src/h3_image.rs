//! MiniMax-H3-Image request contract: size and canvas rules, the published aspect table, the fixed
//! 6-step sigma list, and the refusals a request must pass before any backend runs. Tensor-free:
//! MLX, Candle and the shared-generation runner all read these as data.
//!
//! Sources (read 2026-11-10): `huggingface/diffusers` main,
//! `src/diffusers/modular_pipelines/minimax_h3_image/` (`before_denoise.py` canvas and validation,
//! `modular_pipeline.py` multiple/max sequence length, `decoders.py` resize back), plus the
//! MiniMaxAI/MiniMax-H3-Image model card and `pdd_config.json`.
//!
//! Upstream enforces NO maximum output area or edge: the requested size only has to be positive.
//! The 32 px stride applies to the derived generation canvas, not to the requested size, which is
//! resized back from the canvas. The card's "about 4 MP to 20 MP" range is advisory only. This
//! family is NOT bound by H3 video's `CANVAS_MAX_PIXELS` (1.03 MP) or `MAX_CANVAS_EDGE` (2016).

use crate::Error;
use thiserror::Error as ThisError;

/// Canvas multiple, pixels per token edge: VAE 16x spatial times DiT patch 2x2
/// (`canvas_multiple`, modular_pipeline.py:69).
pub const SIZE_STRIDE: u32 = 32;
/// Smallest accepted `min_generation_area` (upstream parameter range, pixels).
pub const MIN_GENERATION_AREA_MIN: u64 = 1024;
/// Largest accepted `min_generation_area`.
pub const MIN_GENERATION_AREA_MAX: u64 = 16_777_216;
/// Upstream default `min_generation_area` (2048x2048).
pub const MIN_GENERATION_AREA_DEFAULT: u64 = 4_194_304;
/// Advisory working range (not enforced; upstream has no maximum): about 4 MP to 20 MP.
pub const WORKING_AREA_MIN: u64 = 4 * 1_048_576;
/// Upper end of the advisory working range.
pub const WORKING_AREA_MAX: u64 = 20 * 1_048_576;
/// Maximum reference images for edit (model card; diffusers does not enforce a count).
pub const MAX_REFERENCES: usize = 16;
/// Prompt is truncated upstream to this many tokens.
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// The only step count the distilled sampler supports.
pub const FIXED_STEPS: u32 = 6;
/// The only guidance value (no classifier-free guidance).
pub const FIXED_GUIDANCE: f32 = 1.0;

/// The 6-step schedule from `pdd_config.json` (7 boundaries, `1.0` down to `0.0`). The scheduler
/// uses velocity euler updates and does NOT shift these sigmas.
pub const SIGMAS: [f32; 7] = [
    1.0,
    0.967_523_7,
    0.887_548_5,
    0.75,
    0.532_773_9,
    0.232_008_88,
    0.0,
];

/// [`SIGMAS`] exactly as `pdd_config.json` / diffusers' `sample_sigmas` publish them (float64).
/// The sampler derives its float32 grid from these, not from the rounded [`SIGMAS`]; see
/// [`sampler_sigmas`].
pub const SIGMAS_F64: [f64; 7] = [
    1.0,
    0.967_523_694_038_391_1,
    0.887_548_506_259_918_2,
    0.75,
    0.532_773_911_952_972_4,
    0.232_008_874_416_351_32,
    0.0,
];

/// The float32 sigma grid the released sampler steps on (`MiniMaxH3ImageSetTimestepsStep`,
/// before_denoise.py): every published sigma is rounded through its float32 timestep
/// `t = f32(1 - sigma)` and back, `sigma' = 1 - t` in float32, with the terminal `0.0` kept. Below
/// `sigma = 0.5` that round trip moves a sigma by an ulp, so this is not [`SIGMAS`].
pub fn sampler_sigmas() -> [f32; 7] {
    let mut out = [0.0f32; 7];
    for (slot, &s) in out.iter_mut().zip(SIGMAS_F64.iter()).take(6) {
        let t = (1.0f64 - s) as f32;
        *slot = 1.0f32 - t;
    }
    out
}

/// One published size: label and exact pixel dimensions (about 16 MP each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AspectEntry {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
}

/// Published "Common 16 MP sizes", including the portrait forms the card lists.
pub const ASPECT_TABLE: [AspectEntry; 8] = [
    AspectEntry {
        label: "1:1",
        width: 4096,
        height: 4096,
    },
    AspectEntry {
        label: "4:3",
        width: 4736,
        height: 3552,
    },
    AspectEntry {
        label: "3:4",
        width: 3552,
        height: 4736,
    },
    AspectEntry {
        label: "3:2",
        width: 5024,
        height: 3360,
    },
    AspectEntry {
        label: "2:3",
        width: 3360,
        height: 5024,
    },
    AspectEntry {
        label: "16:9",
        width: 5472,
        height: 3072,
    },
    AspectEntry {
        label: "9:16",
        width: 3072,
        height: 5472,
    },
    AspectEntry {
        label: "21:9",
        width: 6272,
        height: 2688,
    },
];

/// Why a H3-Image request was refused. Each variant names the limit it broke.
#[derive(Debug, Clone, PartialEq, ThisError)]
pub enum H3ImageRefusal {
    #[error("size {width}x{height} must be positive on both sides")]
    SizeNotPositive { width: u32, height: u32 },
    #[error("min_generation_area {value} is outside {MIN_GENERATION_AREA_MIN}..={MIN_GENERATION_AREA_MAX}")]
    MinGenerationAreaRange { value: u64 },
    #[error("steps {steps} refused: H3-Image is fixed at {FIXED_STEPS} steps")]
    Steps { steps: u32 },
    #[error("guidance {guidance} refused: H3-Image has no CFG; guidance must be {FIXED_GUIDANCE} or omitted")]
    Guidance { guidance: f32 },
    #[error("negative prompt refused: H3-Image does not support one")]
    NegativePrompt,
    #[error("{count} reference images refused: at most {MAX_REFERENCES} are supported")]
    TooManyReferences { count: usize },
    #[error(
        "reference image {index} is {width}x{height}: every reference needs at least \
         {MIN_REFERENCE_EDGE} pixels on each side"
    )]
    ReferenceTooSmall {
        index: usize,
        width: u32,
        height: u32,
    },
    #[error(
        "reference image {index} carries strength {strength}: H3-Image edit has no per-reference \
         strength; omit it"
    )]
    ReferenceStrength { index: usize, strength: f32 },
    #[error("sampler {name:?} refused: H3-Image samples with one of {SAMPLER_NAMES:?}")]
    UnknownSampler { name: String },
    #[error(
        "sampler {named:?} conflicts with the h3_image_sampler block, which selects {typed:?}; \
         name one sampler"
    )]
    SamplerConflict { named: String, typed: &'static str },
    #[error("lcm {field} {value} refused: it must be finite")]
    SamplerNonFinite { field: &'static str, value: f32 },
    #[error("lcm s_noise {value} refused: it must be greater than 0")]
    SamplerNoiseScale { value: f32 },
    #[error("lcm s_noise_end {value} refused: it must be 0 or greater")]
    SamplerNoiseScaleEnd { value: f32 },
    #[error("lcm noise_clip_std {value} refused: it must be 0 (off) or greater")]
    SamplerNoiseClip { value: f32 },
}

impl From<H3ImageRefusal> for Error {
    fn from(r: H3ImageRefusal) -> Self {
        Error::Unsupported(r.to_string())
    }
}

/// Python `round` (half to even) to the nearest multiple, floored at one multiple.
fn round_to_multiple(value: f64) -> u32 {
    let m = f64::from(SIZE_STRIDE);
    ((value / m).round_ties_even() * m).max(m) as u32
}

/// The `(width, height)` canvas upstream generates a `width x height` output on
/// (`get_image_canvas_size`, before_denoise.py:94). An image below `min_generation_area` is scaled
/// up by `sqrt(min_generation_area / area)` at the same aspect; either way both sides round to the
/// nearest multiple of [`SIZE_STRIDE`]. The output is resized back from this canvas to the
/// requested size when they differ (decoders.py:165).
pub fn generation_canvas(width: u32, height: u32, min_generation_area: u64) -> (u32, u32) {
    let (w, h) = (f64::from(width), f64::from(height));
    let scale = (min_generation_area as f64 / (w * h)).sqrt().max(1.0);
    (round_to_multiple(w * scale), round_to_multiple(h * scale))
}

/// Check an output size: upstream only requires it to be positive (before_denoise.py:212).
pub fn validate_size(width: u32, height: u32) -> Result<(), H3ImageRefusal> {
    if width == 0 || height == 0 {
        return Err(H3ImageRefusal::SizeNotPositive { width, height });
    }
    Ok(())
}

/// Check an optional `min_generation_area` against upstream's range (before_denoise.py:214).
pub fn validate_min_generation_area(value: Option<u64>) -> Result<(), H3ImageRefusal> {
    match value {
        Some(v) if !(MIN_GENERATION_AREA_MIN..=MIN_GENERATION_AREA_MAX).contains(&v) => {
            Err(H3ImageRefusal::MinGenerationAreaRange { value: v })
        }
        _ => Ok(()),
    }
}

/// Check the whole request. `guidance` of `None` means omitted.
pub fn validate_request(
    width: u32,
    height: u32,
    min_generation_area: Option<u64>,
    steps: u32,
    guidance: Option<f32>,
    has_negative_prompt: bool,
    reference_count: usize,
) -> Result<(), H3ImageRefusal> {
    validate_size(width, height)?;
    validate_min_generation_area(min_generation_area)?;
    if steps != FIXED_STEPS {
        return Err(H3ImageRefusal::Steps { steps });
    }
    if let Some(g) = guidance {
        if g != FIXED_GUIDANCE {
            return Err(H3ImageRefusal::Guidance { guidance: g });
        }
    }
    if has_negative_prompt {
        return Err(H3ImageRefusal::NegativePrompt);
    }
    if reference_count > MAX_REFERENCES {
        return Err(H3ImageRefusal::TooManyReferences {
            count: reference_count,
        });
    }
    Ok(())
}

// ── sampler ─────────────────────────────────────────────────────────────────────────────────────
//
// Both samplers step on the same fixed grid ([`sampler_sigmas`], six steps). ComfyUI's H3-Image
// model sampling puts its `simple` scheduler at six steps exactly on that grid, so the LCM sampler
// below is ComfyUI's `comfy/k_diffusion/sampling.py::sample_lcm` on the published sigmas.

/// The sampler names a request may give in [`crate::GenerationRequest::sampler`], in the order the
/// descriptor advertises them. `lcm` (at [`H3ImageSampler::lcm_default`]) is the default;
/// `euler` is the released deterministic sampler, selected by name or block.
pub const SAMPLER_NAMES: &[&str] = &["euler", "lcm"];
/// The default LCM noise scale at step 0.
pub const LCM_S_NOISE_DEFAULT: f32 = 1.1;
/// The default LCM noise scale at the last step; `s` ramps linearly from [`LCM_S_NOISE_DEFAULT`].
pub const LCM_S_NOISE_END_DEFAULT: f32 = 0.9;
/// LCM noise clipping is off unless asked for.
pub const LCM_NOISE_CLIP_STD_DEFAULT: f32 = 0.0;

/// How each of the six steps moves the generated rows.
///
/// The velocity `v` the H3-Image DiT returns points **toward the data**: the released Euler step
/// is `x + (σᵢ − σᵢ₊₁)·v` with `σ` falling, so the clean estimate is `x0 = x + σᵢ·v`. ComfyUI's
/// `CONST` model sampling writes the same estimate as `x - σ·v'` over its noise-ward velocity
/// `v' = -v`; the two are the same arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum H3ImageSampler {
    /// The released deterministic velocity Euler step, `x ← x + (σᵢ − σᵢ₊₁)·v`. The diffusers
    /// reference pipeline runs this sampler.
    Euler,
    /// ComfyUI `sample_lcm`: jump to the clean estimate `x0 = x + σᵢ·v`, then, unless `σᵢ₊₁` is
    /// the terminal 0, re-noise it to `σᵢ₊₁` with fresh seeded noise:
    /// `x ← σᵢ₊₁·(s·ε) + (1 − σᵢ₊₁)·x0`. `s` ramps linearly from `s_noise` at step 0 to
    /// `s_noise_end` at step `n − 1` ([`H3ImageSampler::noise_scale`]). When `noise_clip_std` is
    /// above 0, `ε` is first clamped to `±noise_clip_std · std(ε)`.
    Lcm {
        s_noise: f32,
        s_noise_end: f32,
        noise_clip_std: f32,
    },
}

/// The default sampler is [`H3ImageSampler::lcm_default`].
impl Default for H3ImageSampler {
    fn default() -> Self {
        Self::lcm_default()
    }
}

impl H3ImageSampler {
    /// The default sampler: LCM with `s_noise = 1.1` ramping to `s_noise_end = 0.9`, no clipping.
    pub const fn lcm_default() -> Self {
        Self::Lcm {
            s_noise: LCM_S_NOISE_DEFAULT,
            s_noise_end: LCM_S_NOISE_END_DEFAULT,
            noise_clip_std: LCM_NOISE_CLIP_STD_DEFAULT,
        }
    }

    /// LCM at a constant `s_noise` (`s_noise_end = s_noise`), no clipping.
    pub const fn lcm(s_noise: f32) -> Self {
        Self::Lcm {
            s_noise,
            s_noise_end: s_noise,
            noise_clip_std: LCM_NOISE_CLIP_STD_DEFAULT,
        }
    }

    /// The name [`SAMPLER_NAMES`] lists this sampler under.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Lcm { .. } => "lcm",
        }
    }

    /// The first non-finite parameter as `(field, value)`, or `None`.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        match *self {
            Self::Euler => None,
            Self::Lcm {
                s_noise,
                s_noise_end,
                noise_clip_std,
            } => [
                ("s_noise", s_noise),
                ("s_noise_end", s_noise_end),
                ("noise_clip_std", noise_clip_std),
            ]
            .into_iter()
            .find(|(_, v)| !v.is_finite()),
        }
    }

    /// Refuse a parameter the step cannot use: a non-finite value, `s_noise <= 0`, a negative
    /// `s_noise_end` (so every noised step's scale stays positive) or a negative clip.
    pub fn validate(&self) -> Result<(), H3ImageRefusal> {
        if let Some((field, value)) = self.first_nonfinite_float() {
            return Err(H3ImageRefusal::SamplerNonFinite { field, value });
        }
        if let Self::Lcm {
            s_noise,
            s_noise_end,
            noise_clip_std,
        } = *self
        {
            if s_noise <= 0.0 {
                return Err(H3ImageRefusal::SamplerNoiseScale { value: s_noise });
            }
            if s_noise_end < 0.0 {
                return Err(H3ImageRefusal::SamplerNoiseScaleEnd { value: s_noise_end });
            }
            if noise_clip_std < 0.0 {
                return Err(H3ImageRefusal::SamplerNoiseClip {
                    value: noise_clip_std,
                });
            }
        }
        Ok(())
    }

    /// The noise scale `s` at step `step` of `num_steps`:
    /// `s_noise + (s_noise_end − s_noise)·t` with `t = step / (num_steps − 1)`, and `t = 0` for a
    /// single step. `0` under Euler, which adds no noise.
    pub fn noise_scale(&self, step: usize, num_steps: usize) -> f32 {
        match *self {
            Self::Euler => 0.0,
            Self::Lcm {
                s_noise,
                s_noise_end,
                ..
            } => {
                let t = if num_steps > 1 {
                    step as f32 / (num_steps - 1) as f32
                } else {
                    0.0
                };
                s_noise + (s_noise_end - s_noise) * t
            }
        }
    }
}

/// The sampler a request runs: the typed `h3_image_sampler` block when present (validated), else
/// the `sampler` name — `None` or `"lcm"` is the default [`H3ImageSampler::lcm_default`],
/// `"euler"` is [`H3ImageSampler::Euler`]. A name that disagrees with the typed block is refused rather
/// than either one winning.
pub fn resolve_sampler(
    name: Option<&str>,
    typed: Option<H3ImageSampler>,
) -> Result<H3ImageSampler, H3ImageRefusal> {
    if let Some(name) = name.filter(|n| !SAMPLER_NAMES.contains(n)) {
        return Err(H3ImageRefusal::UnknownSampler {
            name: name.to_owned(),
        });
    }
    let sampler = match (name, typed) {
        (Some(named), Some(typed)) if named != typed.name() => {
            return Err(H3ImageRefusal::SamplerConflict {
                named: named.to_owned(),
                typed: typed.name(),
            })
        }
        (_, Some(typed)) => typed,
        (Some("euler"), None) => H3ImageSampler::Euler,
        // No name, or `"lcm"`: the only other name the check above admits.
        (_, None) => H3ImageSampler::lcm_default(),
    };
    sampler.validate()?;
    Ok(sampler)
}

/// [`resolve_sampler`] over a request's `sampler` and `h3_image_sampler` fields.
pub fn request_sampler(req: &crate::GenerationRequest) -> Result<H3ImageSampler, H3ImageRefusal> {
    resolve_sampler(req.sampler.as_deref(), req.h3_image_sampler)
}

/// The seed of the LCM noise one step draws, from the seed of the image being rendered.
///
/// `splitmix64((image_seed ⊕ "LCM_NOIS") + (step + 1)·φ)`, with `φ = 0x9E3779B97F4A7C15` and the
/// domain word `0x4C434D5F4E4F4953` (ASCII `LCM_NOIS`). A pure function, so the same seed and
/// sampler give the same image on a backend; the domain word and the finalizer keep these draws
/// apart from the initial latent (seeded with `image_seed` itself) and the edit references' draws.
/// Each backend turns the returned seed into standard-normal noise with its own generator, so the
/// noise is the same across runs of one backend, not across backends.
pub fn lcm_noise_seed(image_seed: u64, step: usize) -> u64 {
    const DOMAIN: u64 = 0x4C43_4D5F_4E4F_4953;
    let mut z =
        (image_seed ^ DOMAIN).wrapping_add((step as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The tensor operations [`sampler_step`] needs. A backend implements these over its own float32
/// tensors; the step arithmetic itself lives only in [`sampler_step`].
pub trait H3ImageStepOps {
    type Tensor;
    type Error;
    /// `x + b·y`, element-wise, float32.
    fn add_scaled(
        &mut self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        b: f32,
    ) -> Result<Self::Tensor, Self::Error>;
    /// `a·x`, element-wise, float32.
    fn scale(&mut self, x: &Self::Tensor, a: f32) -> Result<Self::Tensor, Self::Error>;
    /// Standard-normal float32 noise shaped like `like`, drawn deterministically from `seed`.
    fn gaussian_like(
        &mut self,
        like: &Self::Tensor,
        seed: u64,
    ) -> Result<Self::Tensor, Self::Error>;
    /// The sample standard deviation (divisor `n − 1`, `torch.std`'s default) over every element.
    fn std(&mut self, x: &Self::Tensor) -> Result<f32, Self::Error>;
    /// `x` clamped element-wise to `[lo, hi]`.
    fn clamp(&mut self, x: &Self::Tensor, lo: f32, hi: f32) -> Result<Self::Tensor, Self::Error>;
}

/// Where one step sits: its index, the step count, and the two grid sigmas it moves between.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerStep {
    /// The step index, `0..num_steps`.
    pub step: usize,
    /// Steps in the run.
    pub num_steps: usize,
    /// `σᵢ`, the noise level the velocity was evaluated at.
    pub sigma: f32,
    /// `σᵢ₊₁`, the noise level the step moves to (`0` after the last).
    pub sigma_next: f32,
    /// The seed of the image being rendered (the request seed plus the image's index in a
    /// multi-image request); [`lcm_noise_seed`] derives each step's noise seed from it.
    pub image_seed: u64,
}

/// One sampler step over the generated rows `x` and their velocity `v`. An edit's reference rows
/// never reach this function: the backend passes only the generated rows and keeps the reference
/// rows as they are.
///
/// * [`H3ImageSampler::Euler`]: `x + (σᵢ − σᵢ₊₁)·v`, the released update.
/// * [`H3ImageSampler::Lcm`]: `x0 = x + σᵢ·v`; returns `x0` when `σᵢ₊₁ = 0`, otherwise
///   `(1 − σᵢ₊₁)·x0 + σᵢ₊₁·s·ε` with `ε` drawn from [`lcm_noise_seed`] and clamped to
///   `±noise_clip_std·std(ε)` when the clip is on.
pub fn sampler_step<O: H3ImageStepOps + ?Sized>(
    ops: &mut O,
    sampler: &H3ImageSampler,
    at: SamplerStep,
    x: &O::Tensor,
    v: &O::Tensor,
) -> Result<O::Tensor, O::Error> {
    match *sampler {
        H3ImageSampler::Euler => ops.add_scaled(x, v, at.sigma - at.sigma_next),
        H3ImageSampler::Lcm { noise_clip_std, .. } => {
            let x0 = ops.add_scaled(x, v, at.sigma)?;
            if at.sigma_next <= 0.0 {
                return Ok(x0);
            }
            let mut eps = ops.gaussian_like(&x0, lcm_noise_seed(at.image_seed, at.step))?;
            if noise_clip_std > 0.0 {
                let c = noise_clip_std * ops.std(&eps)?;
                eps = ops.clamp(&eps, -c, c)?;
            }
            let s = sampler.noise_scale(at.step, at.num_steps);
            let kept = ops.scale(&x0, 1.0 - at.sigma_next)?;
            ops.add_scaled(&kept, &eps, at.sigma_next * s)
        }
    }
}

/// Per-row modality tag of an image row (`MINIMAX_H3_IMAGE_TAG`, modular_pipeline.py). Tags index
/// the transformer's AdaLN table, so the values are a checkpoint contract: image rows reuse H3's
/// video slot.
pub const IMAGE_TAG: i32 = 0;
/// Per-row modality tag of a text row (`MINIMAX_H3_TEXT_TAG`).
pub const TEXT_TAG: i32 = 1;
/// Family DiT parity bound (requirement E9): a backend's velocity matches the diffusers
/// `MiniMaxH3ImageTransformer2DModel` (float32 activations over the same bf16 weights) when
/// `||got - want|| / ||want||` is at most this, at step 0 and at the on-trajectory step 3. Both the
/// MLX and the Candle real-weight gates assert it.
pub const DIT_PARITY_MAX_REL_L2: f32 = 5e-2;
/// Family DiT parity bound: the flattened velocity's cosine similarity against diffusers is at
/// least this. Asserted next to [`DIT_PARITY_MAX_REL_L2`] by both backends.
pub const DIT_PARITY_MIN_COSINE: f32 = 0.995;
/// Edit-path DiT parity bound, relative to the reference's own bf16 rounding: a backend's edit
/// velocity passes when `rel-L2(ours, f32 reference)` is at most this times
/// `rel-L2(reference bf16, reference f32)`, computed per step from the same reference file.
///
/// Why: on edit inputs the deep-stack blocks amplify bf16 rounding, so upstream's own bf16 forward
/// misses the fixed [`DIT_PARITY_MAX_REL_L2`] (5.33e-2 at step 0, 4.98e-2 at step 3). Michael
/// approved this bound on 2026-11-11. The text-to-image gates keep the fixed
/// [`DIT_PARITY_MAX_REL_L2`]. Use [`edit_parity_limit`] to apply it.
pub const EDIT_DIT_PARITY_MAX_RATIO_TO_REFERENCE_BF16: f64 = 2.0;
/// A reference bf16 drift below this is not a genuine bf16 forward (bf16 rounding through a deep
/// stack is never this small), so [`edit_parity_limit`] refuses it.
pub const EDIT_DIT_MIN_PLAUSIBLE_REFERENCE_BF16_DRIFT: f64 = 1e-4;

/// The edit-path rel-L2 limit for one step: [`EDIT_DIT_PARITY_MAX_RATIO_TO_REFERENCE_BF16`] times
/// the reference's bf16-vs-f32 rel-L2 drift at that step. Refuses a non-finite drift and one below
/// [`EDIT_DIT_MIN_PLAUSIBLE_REFERENCE_BF16_DRIFT`], so the gate can never pass vacuously.
pub fn edit_parity_limit(reference_bf16_drift: f64) -> Result<f64, Error> {
    if !reference_bf16_drift.is_finite() {
        return Err(Error::Msg(format!(
            "the edit reference's bf16-vs-f32 drift is {reference_bf16_drift}; the reference must \
             contain a finite bf16 forward (diag.bf16.*)"
        )));
    }
    if reference_bf16_drift < EDIT_DIT_MIN_PLAUSIBLE_REFERENCE_BF16_DRIFT {
        return Err(Error::Msg(format!(
            "the edit reference's bf16-vs-f32 drift {reference_bf16_drift:.3e} is below \
             {EDIT_DIT_MIN_PLAUSIBLE_REFERENCE_BF16_DRIFT:.0e}: the bf16 reference is not a \
             genuine bf16 forward"
        )));
    }
    Ok(EDIT_DIT_PARITY_MAX_RATIO_TO_REFERENCE_BF16 * reference_bf16_drift)
}
/// VAE spatial compression: one latent cell per 16x16 pixels.
pub const VAE_SPATIAL_RATIO: u32 = 16;

/// The `text2image` presentation's token ids: the prompt tokenized verbatim (no chat template, no
/// special tokens) and truncated to [`MAX_PROMPT_TOKENS`] (encoders.py,
/// `MiniMaxH3ImageTextEncoderStep`). An empty presentation is refused, as upstream refuses an empty
/// prompt.
pub fn t2i_prompt_ids(mut ids: Vec<i32>) -> Result<Vec<i32>, Error> {
    if ids.is_empty() {
        return Err(Error::Unsupported(
            "MiniMax-H3-Image needs a non-empty prompt: the packed sequence has no text rows \
             without one"
                .into(),
        ));
    }
    ids.truncate(MAX_PROMPT_TOKENS);
    Ok(ids)
}

/// The `(latent_height, latent_width)` a `width x height` canvas packs to. The canvas must be a
/// [`SIZE_STRIDE`] multiple, which [`generation_canvas`] guarantees.
pub fn latent_grid(canvas_width: u32, canvas_height: u32) -> Result<(u32, u32), Error> {
    if canvas_width == 0
        || canvas_height == 0
        || !canvas_width.is_multiple_of(SIZE_STRIDE)
        || !canvas_height.is_multiple_of(SIZE_STRIDE)
    {
        return Err(Error::Msg(format!(
            "MiniMax-H3-Image canvas {canvas_width}x{canvas_height} must be a positive multiple of \
             {SIZE_STRIDE}"
        )));
    }
    Ok((
        canvas_height / VAE_SPATIAL_RATIO,
        canvas_width / VAE_SPATIAL_RATIO,
    ))
}

// ── edit (reference images) ─────────────────────────────────────────────────────────────────────
//
// Sources (read 2026-11-10): diffusers main `modular_pipelines/minimax_h3_image/`
// `before_encoder.py` (`bound_short_edge`, `MiniMaxH3ImageEditResizeStep`), `encoders.py`
// (`MiniMaxH3ImageEditTextEncoderStep`, `MiniMaxH3ImageEditVaeEncoderStep`) and
// `modular_pipeline.py` (`reference_noise_aug`).

/// The smallest side a reference may have: one canvas multiple (`before_encoder.py:123`).
pub const MIN_REFERENCE_EDGE: u32 = SIZE_STRIDE;
/// The VAE copy of a reference keeps its pixels while its short edge is within
/// `[REFERENCE_MIN_SHORT_EDGE, REFERENCE_MAX_SHORT_EDGE]` (`reference_min_short_edge`).
pub const REFERENCE_MIN_SHORT_EDGE: u32 = 1024;
/// Upper bound of the VAE copy's short edge (`reference_max_short_edge`).
pub const REFERENCE_MAX_SHORT_EDGE: u32 = 2048;
/// The text-encoder copy of a reference always has this short edge (`vision_min_short_edge` =
/// `vision_max_short_edge` = 1024).
pub const VISION_SHORT_EDGE: u32 = 1024;
/// The `t` the reference rows of an edit are noised to and held at for every step
/// (`reference_noise_aug`, modular_pipeline.py): just short of clean in H3's `t = 1` clean
/// convention.
pub const REFERENCE_NOISE_AUG: f32 = 0.999;
/// Qwen3-VL merges `2 x 2` vision patches into one token (`merge_size ** 2`).
pub const VISION_MERGE_TOKENS: usize = 4;

/// The `(width, height)` one consumer of a reference resizes it to (`bound_short_edge`,
/// before_encoder.py): unchanged when the short edge is within `[min_short_edge, max_short_edge]`,
/// otherwise scaled at the same aspect so the short edge lands on the nearer bound, both sides
/// rounded (Python `round`, half to even) to `multiple` and floored at one `multiple`.
pub fn bound_short_edge(
    width: u32,
    height: u32,
    min_short_edge: u32,
    max_short_edge: u32,
    multiple: u32,
) -> (u32, u32) {
    let short = width.min(height);
    if (min_short_edge..=max_short_edge).contains(&short) {
        return (width, height);
    }
    let bound = if short < min_short_edge {
        min_short_edge
    } else {
        max_short_edge
    };
    let scale = f64::from(bound) / f64::from(short);
    let m = f64::from(multiple);
    let side = |v: u32| ((f64::from(v) * scale / m).round_ties_even() * m).max(m) as u32;
    (side(width), side(height))
}

/// The size the VAE copy of a `width x height` reference is resized to.
pub fn reference_vae_size(width: u32, height: u32) -> (u32, u32) {
    bound_short_edge(
        width,
        height,
        REFERENCE_MIN_SHORT_EDGE,
        REFERENCE_MAX_SHORT_EDGE,
        SIZE_STRIDE,
    )
}

/// The size the text-encoder (vision tower) copy of a `width x height` reference is resized to.
pub fn reference_vision_size(width: u32, height: u32) -> (u32, u32) {
    bound_short_edge(
        width,
        height,
        VISION_SHORT_EDGE,
        VISION_SHORT_EDGE,
        SIZE_STRIDE,
    )
}

/// The ordered reference images of an edit request: every [`crate::Conditioning::Reference`] and every
/// image of every [`crate::Conditioning::MultiReference`], in request order (`<Picture 1>` is the first).
/// Zero references is a `text2image` request. Refuses a per-reference strength other than the
/// no-op `1.0`, more than [`MAX_REFERENCES`] images, and a reference under
/// [`MIN_REFERENCE_EDGE`] on either side. Other conditioning kinds are left to the descriptor's
/// allowlist.
pub fn edit_reference_images(
    conditioning: &[crate::Conditioning],
) -> Result<Vec<&crate::Image>, H3ImageRefusal> {
    let mut images = Vec::new();
    for c in conditioning {
        match c {
            crate::Conditioning::Reference { image, strength } => {
                if let Some(s) = strength.filter(|&s| s != 1.0) {
                    return Err(H3ImageRefusal::ReferenceStrength {
                        index: images.len() + 1,
                        strength: s,
                    });
                }
                images.push(image);
            }
            crate::Conditioning::MultiReference { images: many } => images.extend(many.iter()),
            _ => {}
        }
        if images.len() > MAX_REFERENCES {
            return Err(H3ImageRefusal::TooManyReferences {
                count: conditioning
                    .iter()
                    .map(|c| match c {
                        crate::Conditioning::Reference { .. } => 1,
                        crate::Conditioning::MultiReference { images } => images.len(),
                        _ => 0,
                    })
                    .sum(),
            });
        }
    }
    validate_reference_sizes(images.iter().map(|i| (i.width, i.height)))?;
    Ok(images)
}

/// Refuse a reference under [`MIN_REFERENCE_EDGE`] on either side (before_encoder.py:123).
/// `index` in the refusal is 1-based, the `<Picture N>` the prompt names it by.
pub fn validate_reference_sizes(
    sizes: impl IntoIterator<Item = (u32, u32)>,
) -> Result<(), H3ImageRefusal> {
    for (i, (width, height)) in sizes.into_iter().enumerate() {
        if width < MIN_REFERENCE_EDGE || height < MIN_REFERENCE_EDGE {
            return Err(H3ImageRefusal::ReferenceTooSmall {
                index: i + 1,
                width,
                height,
            });
        }
    }
    Ok(())
}

/// The label that precedes reference `index`'s (0-based) vision block: `"Picture {index + 1}: "`
/// with no angle brackets — the `add_vision_id` form of the processor chat template
/// (`processor/chat_template.json`), unlike the `<Picture N>` the prompt refers to it by.
pub fn edit_reference_label(index: usize) -> String {
    format!("Picture {}: ", index + 1)
}

/// The three special ids a vision block is spelled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionBlockIds {
    pub vision_start: i32,
    pub image_pad: i32,
    pub vision_end: i32,
}

/// The `edit` presentation (`MiniMaxH3ImageEditTextEncoderStep`, encoders.py): per reference, its
/// [`edit_reference_label`] tokenized verbatim and truncated to [`MAX_PROMPT_TOKENS`], then
/// `<|vision_start|>`, `vision_tokens[i]` `<|image_pad|>` and `<|vision_end|>`; then the prompt
/// tokenized verbatim and truncated to [`MAX_PROMPT_TOKENS`]. No chat template, no special tokens.
/// Returns `(ids, tags)` built in lockstep: label and prompt rows are [`TEXT_TAG`], every row of a
/// vision block is [`IMAGE_TAG`].
///
/// `encode` is the backend tokenizer (`tokenizer(text, add_special_tokens=False)`); it is called
/// once per label and once for the prompt.
pub fn edit_presentation(
    prompt: &str,
    vision_tokens: &[usize],
    ids_of: VisionBlockIds,
    mut encode: impl FnMut(&str) -> Result<Vec<i32>, Error>,
) -> Result<(Vec<i32>, Vec<i32>), Error> {
    if vision_tokens.is_empty() {
        return Err(Error::Unsupported(
            "MiniMax-H3-Image edit needs at least one reference image".into(),
        ));
    }
    if vision_tokens.len() > MAX_REFERENCES {
        return Err(H3ImageRefusal::TooManyReferences {
            count: vision_tokens.len(),
        }
        .into());
    }
    let mut ids = Vec::new();
    let mut tags = Vec::new();
    for (i, &n) in vision_tokens.iter().enumerate() {
        if n == 0 {
            return Err(Error::Msg(format!(
                "MiniMax-H3-Image edit: reference {} contributes no vision tokens",
                i + 1
            )));
        }
        let mut label = encode(&edit_reference_label(i))?;
        label.truncate(MAX_PROMPT_TOKENS);
        tags.extend(std::iter::repeat_n(TEXT_TAG, label.len()));
        ids.extend(label);
        ids.push(ids_of.vision_start);
        ids.extend(std::iter::repeat_n(ids_of.image_pad, n));
        ids.push(ids_of.vision_end);
        tags.extend(std::iter::repeat_n(IMAGE_TAG, n + 2));
    }
    let prompt_ids = encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err(Error::Unsupported(
            "MiniMax-H3-Image needs a non-empty prompt".into(),
        ));
    }
    let prompt_ids = &prompt_ids[..prompt_ids.len().min(MAX_PROMPT_TOKENS)];
    tags.extend(std::iter::repeat_n(TEXT_TAG, prompt_ids.len()));
    ids.extend_from_slice(prompt_ids);
    Ok((ids, tags))
}

/// The `(latent_height, latent_width)` of one VAE-resized `width x height` reference after the
/// encode crops (`MiniMaxH3ImageEditVaeEncoderStep`): pixels cropped at the top left to whole
/// [`VAE_SPATIAL_RATIO`] cells, then the latent cropped to whole `patch` cells.
pub fn reference_latent_grid(width: u32, height: u32, patch: (u32, u32)) -> (u32, u32) {
    let lh = height / VAE_SPATIAL_RATIO;
    let lw = width / VAE_SPATIAL_RATIO;
    (lh / patch.0 * patch.0, lw / patch.1 * patch.1)
}

/// The per-block DiT parity sweep both backends' real-weight `edit` gates print (SW-7092).
///
/// The edit reference records the residual stream after a set of blocks from two diffusers
/// forwards over the same inputs: float32 activations (what the gates assert against) and bf16
/// compute (the checkpoint as published). For each block this module compares a backend's stream
/// with both, next to the reference's own bf16-vs-f32 drift, over the whole packed sequence and
/// over each row class (`[text | reference | generated]`), and locates the largest reference
/// activation and the largest error. Host-side and tensor-free so the arithmetic is unit-tested
/// once rather than per backend. Diagnostic only: nothing here is a bound.
pub mod dit_sweep {
    use std::fmt;
    use std::ops::Range;

    /// A backend has *departed* from diffusers' bf16 compute at a block when its error against
    /// the bf16 reference exceeds this multiple of the reference's own bf16-vs-f32 drift there.
    /// Two independent bf16 roundings of similar size sit about `sqrt(2)` times one of them apart,
    /// so twice the drift is more than rounding alone accounts for.
    pub const DEPARTURE_FACTOR: f32 = 2.0;

    /// The packed sequence's row classes, in order: text rows, reference rows, generated rows.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RowRegions {
        pub text: usize,
        pub reference: usize,
        pub generated: usize,
    }

    impl RowRegions {
        /// Names of [`Self::ranges`], in order.
        pub const NAMES: [&'static str; 4] = ["all", "text", "reference", "generated"];

        pub fn seq_len(&self) -> usize {
            self.text + self.reference + self.generated
        }

        /// Row ranges of the whole sequence and of each class, in [`Self::NAMES`] order.
        pub fn ranges(&self) -> [Range<usize>; 4] {
            let (t, r) = (self.text, self.text + self.reference);
            [0..self.seq_len(), 0..t, t..r, r..self.seq_len()]
        }

        /// The class name of sequence row `row`.
        pub fn class_of(&self, row: usize) -> &'static str {
            let [_, text, reference, _] = self.ranges();
            if text.contains(&row) {
                "text"
            } else if reference.contains(&row) {
                "reference"
            } else {
                "generated"
            }
        }
    }

    /// `||got - want|| / ||want||` over rows `rows` of two row-major `[seq_len, width]` buffers,
    /// accumulated in f64. `0` when both are zero there, infinite when only `want` is.
    pub fn rel_l2_rows(got: &[f32], want: &[f32], width: usize, rows: Range<usize>) -> f32 {
        let span = rows.start * width..rows.end * width;
        let (mut diff, mut norm) = (0f64, 0f64);
        for (g, w) in got[span.clone()].iter().zip(&want[span]) {
            let e = f64::from(*g) - f64::from(*w);
            diff += e * e;
            norm += f64::from(*w) * f64::from(*w);
        }
        if norm == 0.0 {
            return if diff == 0.0 { 0.0 } else { f32::INFINITY };
        }
        (diff / norm).sqrt() as f32
    }

    /// One row class's errors at one block.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct RegionErrors {
        /// The backend against the float32-activation reference (what the gate asserts against).
        pub vs_f32: f32,
        /// The backend against the bf16-compute reference.
        pub vs_bf16: f32,
        /// The reference's own bf16-compute stream against its float32 one.
        pub drift: f32,
    }

    impl RegionErrors {
        /// `vs_bf16 / drift`: about `sqrt(2)` for a backend whose only difference from diffusers is
        /// its own bf16 rounding; a block where it jumps is where the backend computes something
        /// else. Infinite when the reference has no drift there.
        pub fn departure_ratio(&self) -> f32 {
            if self.drift > 0.0 {
                self.vs_bf16 / self.drift
            } else {
                f32::INFINITY
            }
        }
    }

    /// One element of the stream, located, with the three values there.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Located {
        pub row: usize,
        pub channel: usize,
        pub class: &'static str,
        pub f32_ref: f32,
        pub bf16_ref: f32,
        pub got: f32,
    }

    /// One block of the sweep.
    #[derive(Debug, Clone, PartialEq)]
    pub struct SweepRow {
        pub layer: usize,
        /// In [`RowRegions::NAMES`] order.
        pub regions: [RegionErrors; 4],
        /// The largest `|f32 reference|` — where a massive activation sits.
        pub peak: Located,
        /// The largest `|got - f32 reference|`.
        pub worst: Located,
    }

    impl SweepRow {
        /// Compare a backend's residual stream after block `layer` with both references. All three
        /// buffers are row-major `[seq_len, width]` (a leading batch of 1 flattens to the same).
        pub fn new(
            layer: usize,
            got: &[f32],
            f32_ref: &[f32],
            bf16_ref: &[f32],
            width: usize,
            regions: RowRegions,
        ) -> Result<Self, String> {
            let len = regions.seq_len() * width;
            if width == 0 || [got.len(), f32_ref.len(), bf16_ref.len()] != [len; 3] {
                return Err(format!(
                    "dit sweep block {layer}: buffers of {}, {} and {} values against {} rows x \
                     {width} channels",
                    got.len(),
                    f32_ref.len(),
                    bf16_ref.len(),
                    regions.seq_len()
                ));
            }
            let errors = |rows: Range<usize>| RegionErrors {
                vs_f32: rel_l2_rows(got, f32_ref, width, rows.clone()),
                vs_bf16: rel_l2_rows(got, bf16_ref, width, rows.clone()),
                drift: rel_l2_rows(bf16_ref, f32_ref, width, rows),
            };
            let argmax = |key: &dyn Fn(usize) -> f32| -> usize {
                (0..len).fold(0, |best, i| if key(i) > key(best) { i } else { best })
            };
            let locate = |i: usize| Located {
                row: i / width,
                channel: i % width,
                class: regions.class_of(i / width),
                f32_ref: f32_ref[i],
                bf16_ref: bf16_ref[i],
                got: got[i],
            };
            Ok(Self {
                layer,
                regions: regions.ranges().map(errors),
                peak: locate(argmax(&|i| f32_ref[i].abs())),
                worst: locate(argmax(&|i| (got[i] - f32_ref[i]).abs())),
            })
        }
    }

    impl fmt::Display for Located {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "row {} ({}) ch {}: f32 {:.4e} bf16 {:.4e} got {:.4e}",
                self.row, self.class, self.channel, self.f32_ref, self.bf16_ref, self.got
            )
        }
    }

    impl fmt::Display for SweepRow {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "sweep block {:>2}", self.layer)?;
            for (name, e) in RowRegions::NAMES.iter().zip(&self.regions) {
                write!(
                    f,
                    " | {name} vs-f32 {:.3e} vs-bf16 {:.3e} drift {:.3e} ratio {:.2}",
                    e.vs_f32,
                    e.vs_bf16,
                    e.drift,
                    e.departure_ratio()
                )?;
            }
            write!(
                f,
                " | peak |f32| {} | worst |got-f32| {}",
                self.peak, self.worst
            )
        }
    }

    /// Per row class (in [`RowRegions::NAMES`] order), the first block whose `vs_bf16` exceeds
    /// [`DEPARTURE_FACTOR`] times its `drift`.
    pub fn first_departure(rows: &[SweepRow]) -> [Option<usize>; 4] {
        std::array::from_fn(|c| {
            rows.iter()
                .find(|r| r.regions[c].vs_bf16 > DEPARTURE_FACTOR * r.regions[c].drift)
                .map(|r| r.layer)
        })
    }

    /// The consecutive pair of swept blocks across which the whole-sequence `vs_f32` grows by the
    /// largest factor: `(from_layer, to_layer, from, to)`.
    pub fn largest_growth(rows: &[SweepRow]) -> Option<(usize, usize, f32, f32)> {
        rows.windows(2)
            .map(|w| {
                (
                    w[0].layer,
                    w[1].layer,
                    w[0].regions[0].vs_f32,
                    w[1].regions[0].vs_f32,
                )
            })
            .filter(|(_, _, from, _)| *from > 0.0)
            .max_by(|a, b| (a.3 / a.2).total_cmp(&(b.3 / b.2)))
    }

    /// Per row class, the consecutive pair of swept blocks across which
    /// [`RegionErrors::departure_ratio`] rises the most: `(from_layer, to_layer, from, to)`.
    pub fn largest_ratio_rise(rows: &[SweepRow], class: usize) -> Option<(usize, usize, f32, f32)> {
        rows.windows(2)
            .map(|w| {
                (
                    w[0].layer,
                    w[1].layer,
                    w[0].regions[class].departure_ratio(),
                    w[1].regions[class].departure_ratio(),
                )
            })
            .filter(|(_, _, from, to)| from.is_finite() && to.is_finite())
            .max_by(|a, b| (a.3 - a.2).total_cmp(&(b.3 - b.2)))
    }

    /// The closing summary lines: per row class the first departure and the largest rise of the
    /// departure ratio, then the largest whole-sequence growth of `vs_f32`.
    pub fn summary(rows: &[SweepRow]) -> Vec<String> {
        let mut lines: Vec<String> = RowRegions::NAMES
            .iter()
            .zip(first_departure(rows))
            .zip(0..)
            .map(|((name, layer), class)| {
                let first = match layer {
                    Some(layer) => format!(
                        "sweep {name} rows: first departure from diffusers bf16 (vs-bf16 > \
                         {DEPARTURE_FACTOR} x drift) at block {layer}"
                    ),
                    None => format!(
                        "sweep {name} rows: no departure from diffusers bf16 (vs-bf16 > \
                         {DEPARTURE_FACTOR} x drift) at any swept block"
                    ),
                };
                match largest_ratio_rise(rows, class) {
                    Some((a, b, from, to)) => format!(
                        "{first}; largest ratio rise between swept blocks {a} and {b}: {from:.2} \
                         -> {to:.2}"
                    ),
                    None => first,
                }
            })
            .collect();
        if let Some((a, b, from, to)) = largest_growth(rows) {
            lines.push(format!(
                "sweep all rows: largest vs-f32 growth between swept blocks {a} and {b}: {from:.3e} \
                 -> {to:.3e} ({:.2}x)",
                to / from
            ));
        }
        lines
    }

    /// The block the edit gate re-runs alone, from the reference's own input (SW-7092): the block
    /// where both backends under-build the reference's massive activation at [`PROBE`].
    pub const ISOLATED_BLOCK: usize = 39;

    /// `(row, channel)` of the massive activation the isolation line reads: text row 74, channel
    /// 3474 — about `-2.2e6` after block 39 in the float32 edit reference.
    pub const PROBE: (usize, usize) = (74, 3474);

    impl Located {
        /// The element at `(row, channel)` of three row-major `[seq_len, width]` buffers; `None`
        /// when it lies outside any of them.
        pub fn at(
            (row, channel): (usize, usize),
            got: &[f32],
            f32_ref: &[f32],
            bf16_ref: &[f32],
            width: usize,
            regions: RowRegions,
        ) -> Option<Self> {
            let i = row.checked_mul(width)?.checked_add(channel)?;
            if channel >= width || row >= regions.seq_len() {
                return None;
            }
            Some(Self {
                row,
                channel,
                class: regions.class_of(row),
                f32_ref: *f32_ref.get(i)?,
                bf16_ref: *bf16_ref.get(i)?,
                got: *got.get(i)?,
            })
        }
    }

    /// One block run in isolation: `isolated block {layer} [{label}]`, then per row class the
    /// rel-L2 against the float32 and the bf16 reference after that block, then the [`PROBE`]
    /// element. With the reference's own input, a large error here is the block's own compute;
    /// a small one puts the full forward's error at that block on what it inherited.
    pub fn isolation_line(label: &str, row: &SweepRow, probe: &Located) -> String {
        let mut line = format!("isolated block {} [{label}]", row.layer);
        for (name, e) in RowRegions::NAMES.iter().zip(&row.regions) {
            line.push_str(&format!(
                " | {name} vs-f32 {:.3e} vs-bf16 {:.3e}",
                e.vs_f32, e.vs_bf16
            ));
        }
        line.push_str(&format!(" | probe {probe} | worst |got-f32| {}", row.worst));
        line
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const REGIONS: RowRegions = RowRegions {
            text: 2,
            reference: 3,
            generated: 1,
        };

        #[test]
        fn row_ranges_partition_the_sequence_in_packed_order() {
            let [all, text, reference, generated] = REGIONS.ranges();
            assert_eq!(all, 0..6);
            assert_eq!((text, reference, generated), (0..2, 2..5, 5..6));
            let classes: Vec<_> = (0..6).map(|r| REGIONS.class_of(r)).collect();
            assert_eq!(
                classes,
                [
                    "text",
                    "text",
                    "reference",
                    "reference",
                    "reference",
                    "generated"
                ]
            );
            let no_refs = RowRegions {
                text: 1,
                reference: 0,
                generated: 2,
            };
            assert_eq!(no_refs.ranges()[2], 1..1);
            assert_eq!(no_refs.class_of(1), "generated");
        }

        #[test]
        fn rel_l2_reads_only_the_requested_rows() {
            // Width 2, three rows; row 1 is wildly wrong, rows 0 and 2 off by 10%.
            let want = [1.0f32, 0.0, 3.0, 4.0, 0.0, 2.0];
            let got = [1.1f32, 0.0, -30.0, 40.0, 0.0, 2.2];
            assert!((rel_l2_rows(&got, &want, 2, 0..1) - 0.1).abs() < 1e-6);
            assert!((rel_l2_rows(&got, &want, 2, 2..3) - 0.1).abs() < 1e-6);
            assert!(rel_l2_rows(&got, &want, 2, 1..2) > 5.0);
            assert_eq!(rel_l2_rows(&got, &want, 2, 1..1), 0.0);
            assert_eq!(rel_l2_rows(&[1.0], &[0.0], 1, 0..1), f32::INFINITY);
        }

        #[test]
        fn a_sweep_row_splits_errors_by_class_and_locates_peak_and_worst() {
            // Width 2 over [text 2 | reference 3 | generated 1].
            let f32_ref = [
                1.0f32, 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ];
            let bf16_ref = f32_ref;
            let mut got = f32_ref;
            got[11] = 4.0; // generated row 5, channel 1: the worst error, not the peak
            let row = SweepRow::new(7, &got, &f32_ref, &bf16_ref, 2, REGIONS).unwrap();
            assert_eq!(row.layer, 7);
            let [_, text, reference, generated] = row.regions;
            assert_eq!((text.vs_f32, reference.vs_f32), (0.0, 0.0));
            assert!(generated.vs_f32 > 2.0);
            assert_eq!(generated.drift, 0.0);
            assert_eq!(
                (
                    row.peak.row,
                    row.peak.channel,
                    row.peak.class,
                    row.peak.f32_ref
                ),
                (2, 0, "reference", 50.0)
            );
            assert_eq!(
                (
                    row.worst.row,
                    row.worst.channel,
                    row.worst.class,
                    row.worst.got
                ),
                (5, 1, "generated", 4.0)
            );
            assert!(SweepRow::new(7, &got[..10], &f32_ref, &bf16_ref, 2, REGIONS).is_err());
        }

        #[test]
        fn the_isolation_line_reads_the_probe_and_every_row_class() {
            let f32_ref = [
                1.0f32, 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ];
            let bf16_ref = f32_ref;
            let mut got = f32_ref;
            got[4] = 25.0; // reference row 2, channel 0: half the massive activation
            let probe = Located::at((2, 0), &got, &f32_ref, &bf16_ref, 2, REGIONS).unwrap();
            assert_eq!(
                (probe.class, probe.f32_ref, probe.got),
                ("reference", 50.0, 25.0)
            );
            assert!(Located::at((6, 0), &got, &f32_ref, &bf16_ref, 2, REGIONS).is_none());
            assert!(Located::at((0, 2), &got, &f32_ref, &bf16_ref, 2, REGIONS).is_none());
            let row = SweepRow::new(39, &got, &f32_ref, &bf16_ref, 2, REGIONS).unwrap();
            let line = isolation_line("ref input", &row, &probe);
            assert!(line.starts_with("isolated block 39 [ref input]"), "{line}");
            for name in RowRegions::NAMES {
                assert!(line.contains(&format!("| {name} vs-f32")), "{line}");
            }
            assert!(line.contains("probe row 2 (reference) ch 0"), "{line}");
        }

        fn row(layer: usize, vs_f32: f32, vs_bf16: f32, drift: f32) -> SweepRow {
            let e = RegionErrors {
                vs_f32,
                vs_bf16,
                drift,
            };
            let at = Located {
                row: 0,
                channel: 0,
                class: "text",
                f32_ref: 0.0,
                bf16_ref: 0.0,
                got: 0.0,
            };
            SweepRow {
                layer,
                regions: [e; 4],
                peak: at,
                worst: at,
            }
        }

        #[test]
        fn departure_is_the_first_block_past_twice_the_reference_drift() {
            let rows = [
                row(0, 0.01, 0.012, 0.008),
                row(24, 0.02, 0.02, 0.01),
                row(30, 0.05, 0.05, 0.02),
                row(31, 0.06, 0.06, 0.02),
            ];
            assert_eq!(first_departure(&rows), [Some(30); 4]);
            assert_eq!(first_departure(&rows[..2]), [None; 4]);
            assert_eq!(largest_growth(&rows), Some((24, 30, 0.02, 0.05)));
            // Ratios 1.5, 2.0, 2.5, 3.0 rise by 0.5 at every pair.
            let (_, _, from, to) = largest_ratio_rise(&rows, 0).unwrap();
            assert!((to - from - 0.5).abs() < 1e-5, "{from} -> {to}");
            let jump = [
                row(0, 0.01, 0.015, 0.01),
                row(5, 0.02, 0.016, 0.01),
                row(6, 0.02, 0.06, 0.01),
            ];
            let (a, b, from, to) = largest_ratio_rise(&jump, 2).unwrap();
            assert_eq!((a, b), (5, 6));
            assert!(
                (from - 1.6).abs() < 1e-5 && (to - 6.0).abs() < 1e-5,
                "{from} -> {to}"
            );
            assert_eq!(
                row(0, 0.0, 0.1, 0.0).regions[0].departure_ratio(),
                f32::INFINITY
            );
            let lines = summary(&rows);
            assert_eq!(lines.len(), 5);
            assert!(lines[0].contains("at block 30"), "{}", lines[0]);
            assert!(lines[0].contains("largest ratio rise"), "{}", lines[0]);
            assert!(lines[4].contains("blocks 24 and 30"), "{}", lines[4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_parity_limit_scales_the_reference_bf16_drift() {
        let limit = edit_parity_limit(5.33e-2).unwrap();
        assert!((limit - 2.0 * 5.33e-2).abs() < 1e-12);
    }

    #[test]
    fn edit_parity_limit_refuses_a_missing_or_nan_drift() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(edit_parity_limit(bad).is_err());
        }
    }

    #[test]
    fn edit_parity_limit_refuses_a_tiny_drift() {
        assert!(edit_parity_limit(0.0).is_err());
        assert!(edit_parity_limit(9e-5).is_err());
        assert!(edit_parity_limit(1e-4).is_ok());
    }

    #[test]
    fn prompt_ids_truncate_at_4096_and_refuse_empty() {
        assert_eq!(t2i_prompt_ids(vec![5, 6]).unwrap(), vec![5, 6]);
        let long: Vec<i32> = (0..5000).collect();
        let kept = t2i_prompt_ids(long).unwrap();
        assert_eq!(kept.len(), MAX_PROMPT_TOKENS);
        assert_eq!(kept[4095], 4095);
        assert!(matches!(
            t2i_prompt_ids(Vec::new()),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn latent_grid_is_the_canvas_over_16() {
        assert_eq!(latent_grid(2048, 2048).unwrap(), (128, 128));
        assert_eq!(latent_grid(1664, 2496).unwrap(), (156, 104));
        assert!(latent_grid(1000, 1000).is_err());
        assert_eq!((IMAGE_TAG, TEXT_TAG), (0, 1));
    }

    #[test]
    fn table_entries_are_stride_aligned_near_16mp() {
        for e in ASPECT_TABLE {
            validate_size(e.width, e.height).unwrap();
            assert_eq!((e.width % SIZE_STRIDE, e.height % SIZE_STRIDE), (0, 0));
            let a = u64::from(e.width) * u64::from(e.height);
            assert!((15_000_000..=17_000_000).contains(&a), "{}", e.label);
        }
    }

    #[test]
    fn size_only_needs_to_be_positive() {
        validate_size(1, 1).unwrap();
        validate_size(1000, 1001).unwrap();
        // No maximum area or edge upstream, and not H3 video's 1.03 MP / 2016 caps.
        validate_size(8192, 8192).unwrap();
        assert_eq!(
            validate_size(0, 32),
            Err(H3ImageRefusal::SizeNotPositive {
                width: 0,
                height: 32
            })
        );
    }

    #[test]
    fn canvas_scales_small_images_and_rounds() {
        // At or above the minimum area: only rounding to the stride.
        assert_eq!(
            generation_canvas(4096, 4096, MIN_GENERATION_AREA_DEFAULT),
            (4096, 4096)
        );
        assert_eq!(generation_canvas(1000, 1001, 1024), (992, 992));
        // 1024x1024 with the 4 MP default scales 2x.
        assert_eq!(
            generation_canvas(1024, 1024, MIN_GENERATION_AREA_DEFAULT),
            (2048, 2048)
        );
        // min_generation_area 1024 never enlarges; tiny sizes floor at one stride.
        assert_eq!(generation_canvas(8, 8, 1024), (32, 32));
        // 512x768 (0.39 MP) at 4 MP: scale sqrt(4194304/393216)=3.266 -> 1672.5, 2508.8
        assert_eq!(
            generation_canvas(512, 768, MIN_GENERATION_AREA_DEFAULT),
            (1664, 2496)
        );
    }

    #[test]
    fn min_generation_area_range() {
        validate_min_generation_area(None).unwrap();
        validate_min_generation_area(Some(1024)).unwrap();
        validate_min_generation_area(Some(16_777_216)).unwrap();
        assert!(validate_min_generation_area(Some(1023)).is_err());
        let e = validate_min_generation_area(Some(16_777_217)).unwrap_err();
        assert!(e.to_string().contains("16777216"));
    }

    #[test]
    fn refusals_are_distinct() {
        let ok = |s, g, n, r| validate_request(2048, 2048, None, s, g, n, r);
        ok(6, None, false, 16).unwrap();
        ok(6, Some(1.0), false, 0).unwrap();
        let errs = [
            ok(8, None, false, 0).unwrap_err(),
            ok(6, Some(4.0), false, 0).unwrap_err(),
            ok(6, None, true, 0).unwrap_err(),
            ok(6, None, false, 17).unwrap_err(),
        ];
        for i in 0..errs.len() {
            for j in i + 1..errs.len() {
                assert_ne!(
                    std::mem::discriminant(&errs[i]),
                    std::mem::discriminant(&errs[j])
                );
            }
        }
        assert!(matches!(
            Error::from(errs[0].clone()),
            Error::Unsupported(_)
        ));
    }

    #[test]
    fn the_sampler_grid_is_the_float32_round_trip_of_the_published_sigmas() {
        let g = sampler_sigmas();
        for (i, (&f, &d)) in SIGMAS.iter().zip(SIGMAS_F64.iter()).enumerate() {
            assert_eq!(
                f, d as f32,
                "SIGMAS[{i}] is the float32 rounding of SIGMAS_F64[{i}]"
            );
        }
        assert_eq!(g[0], 1.0);
        assert_eq!(g[3], 0.75);
        assert_eq!(g[6], 0.0);
        assert!(g.windows(2).all(|w| w[0] > w[1]));
        // The round trip moves a sigma by at most one float32 ulp of its timestep.
        for i in 0..6 {
            let d = (g[i] - SIGMAS[i]).abs();
            assert!(d <= f32::EPSILON, "{i}: {} vs {}", g[i], SIGMAS[i]);
        }
    }

    #[test]
    fn sigmas_are_the_published_list() {
        assert_eq!(SIGMAS.len(), FIXED_STEPS as usize + 1);
        assert_eq!(SIGMAS[0], 1.0);
        assert_eq!(SIGMAS[6], 0.0);
        assert!(SIGMAS.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn bound_short_edge_keeps_in_range_and_scales_to_the_nearer_bound() {
        // In range: untouched, even off the 32 grid.
        assert_eq!(reference_vae_size(1500, 1100), (1500, 1100));
        // Below 1024: short edge scaled up to 1024, long side rounded to 32.
        assert_eq!(reference_vae_size(512, 768), (1024, 1536));
        // Above 2048: down to 2048.
        assert_eq!(reference_vae_size(4096, 3000), (2784, 2048));
        // The vision copy is pinned at a 1024 short edge both ways.
        assert_eq!(reference_vision_size(1500, 1100), (1408, 1024));
        assert_eq!(reference_vision_size(1024, 1024), (1024, 1024));
        assert_eq!(reference_vision_size(64, 32), (2048, 1024));
        // Python round is half-to-even: 1.5 multiples -> 2, 2.5 -> 2.
        assert_eq!(bound_short_edge(48, 64, 64, 64, 32), (64, 96));
        assert_eq!(bound_short_edge(80, 32, 32, 32, 32), (80, 32));
        assert_eq!(bound_short_edge(80, 16, 32, 32, 32), (160, 32));
    }

    fn img(w: u32, h: u32) -> crate::Image {
        crate::Image {
            width: w,
            height: h,
            pixels: vec![0; (w * h * 3) as usize],
        }
    }

    #[test]
    fn edit_references_flatten_in_order_and_refuse_17_small_and_strength() {
        use crate::Conditioning;
        let conditioning = vec![
            Conditioning::Reference {
                image: img(32, 40),
                strength: None,
            },
            Conditioning::MultiReference {
                images: vec![img(64, 32), img(33, 33)],
            },
            Conditioning::Reference {
                image: img(50, 50),
                strength: Some(1.0),
            },
        ];
        let refs = edit_reference_images(&conditioning).unwrap();
        let sizes: Vec<(u32, u32)> = refs.iter().map(|i| (i.width, i.height)).collect();
        assert_eq!(sizes, vec![(32, 40), (64, 32), (33, 33), (50, 50)]);
        assert!(edit_reference_images(&[]).unwrap().is_empty());

        let sixteen = vec![Conditioning::MultiReference {
            images: (0..16).map(|_| img(32, 32)).collect(),
        }];
        assert_eq!(edit_reference_images(&sixteen).unwrap().len(), 16);
        let seventeen = vec![
            sixteen[0].clone(),
            Conditioning::Reference {
                image: img(32, 32),
                strength: None,
            },
        ];
        assert_eq!(
            edit_reference_images(&seventeen).unwrap_err(),
            H3ImageRefusal::TooManyReferences { count: 17 }
        );
        let small = vec![Conditioning::MultiReference {
            images: vec![img(32, 32), img(31, 400)],
        }];
        assert_eq!(
            edit_reference_images(&small).unwrap_err(),
            H3ImageRefusal::ReferenceTooSmall {
                index: 2,
                width: 31,
                height: 400
            }
        );
        let strength = vec![Conditioning::Reference {
            image: img(32, 32),
            strength: Some(0.5),
        }];
        assert!(matches!(
            edit_reference_images(&strength).unwrap_err(),
            H3ImageRefusal::ReferenceStrength { index: 1, .. }
        ));
    }

    /// The presentation is `Picture i: ` + vision block per reference, then the prompt; vision rows
    /// are image-tagged, labels and prompt text-tagged.
    #[test]
    fn edit_presentation_matches_the_reference_layout() {
        let specials = VisionBlockIds {
            vision_start: 900,
            image_pad: 901,
            vision_end: 902,
        };
        let mut seen = Vec::new();
        let (ids, tags) = edit_presentation("make it red", &[2, 1], specials, |s| {
            seen.push(s.to_owned());
            Ok(s.bytes().take(3).map(i32::from).collect())
        })
        .unwrap();
        assert_eq!(seen, vec!["Picture 1: ", "Picture 2: ", "make it red"]);
        let p = |s: &str| s.bytes().take(3).map(i32::from).collect::<Vec<i32>>();
        let mut want = p("Picture 1: ");
        want.extend([900, 901, 901, 902]);
        want.extend(p("Picture 2: "));
        want.extend([900, 901, 902]);
        want.extend(p("make it red"));
        assert_eq!(ids, want);
        assert_eq!(tags, vec![1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]);
        assert!(edit_presentation("x", &[], specials, |_| Ok(vec![1])).is_err());
        assert!(edit_presentation("x", &[1; 17], specials, |_| Ok(vec![1])).is_err());
        assert!(edit_presentation("x", &[0], specials, |_| Ok(vec![1])).is_err());
        assert!(edit_presentation("", &[1], specials, |_| Ok(vec![])).is_err());
        // The prompt truncates at 4096 tokens.
        let (ids, _) = edit_presentation("long", &[1], specials, |s| {
            Ok(if s == "long" { vec![7; 5000] } else { vec![1] })
        })
        .unwrap();
        assert_eq!(ids.len(), 1 + 3 + MAX_PROMPT_TOKENS);
    }

    /// A host `Vec<f32>` implementation of the step ops, with a scripted "gaussian" that returns
    /// `noise` and records the seed it was asked for.
    struct HostOps {
        noise: Vec<f32>,
        seeds: Vec<u64>,
    }

    impl H3ImageStepOps for HostOps {
        type Tensor = Vec<f32>;
        type Error = ();
        fn add_scaled(&mut self, x: &Vec<f32>, y: &Vec<f32>, b: f32) -> Result<Vec<f32>, ()> {
            Ok(x.iter().zip(y).map(|(x, y)| x + b * y).collect())
        }
        fn scale(&mut self, x: &Vec<f32>, a: f32) -> Result<Vec<f32>, ()> {
            Ok(x.iter().map(|x| a * x).collect())
        }
        fn gaussian_like(&mut self, like: &Vec<f32>, seed: u64) -> Result<Vec<f32>, ()> {
            assert_eq!(like.len(), self.noise.len());
            self.seeds.push(seed);
            Ok(self.noise.clone())
        }
        fn std(&mut self, x: &Vec<f32>) -> Result<f32, ()> {
            let n = x.len() as f32;
            let mean = x.iter().sum::<f32>() / n;
            Ok((x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0)).sqrt())
        }
        fn clamp(&mut self, x: &Vec<f32>, lo: f32, hi: f32) -> Result<Vec<f32>, ()> {
            Ok(x.iter().map(|v| v.clamp(lo, hi)).collect())
        }
    }

    fn host(noise: &[f32]) -> HostOps {
        HostOps {
            noise: noise.to_vec(),
            seeds: Vec::new(),
        }
    }

    fn at(step: usize, sigma: f32, sigma_next: f32) -> SamplerStep {
        SamplerStep {
            step,
            num_steps: 6,
            sigma,
            sigma_next,
            image_seed: 42,
        }
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    }

    /// Euler is the released `x + (σᵢ − σᵢ₊₁)·v` and draws no noise.
    #[test]
    fn euler_steps_by_the_sigma_delta() {
        let mut ops = host(&[9.0, 9.0]);
        let out = sampler_step(
            &mut ops,
            &H3ImageSampler::Euler,
            at(3, 0.75, 0.5),
            &vec![1.0, 2.0],
            &vec![0.5, -1.0],
        )
        .unwrap();
        close(&out, &[1.125, 1.75]);
        assert!(ops.seeds.is_empty());
    }

    /// LCM: `x0 = x + σ·v`, then `(1 − σ')·x0 + σ'·s·ε` with step 0's `s = s_noise`; the noise is
    /// drawn from `lcm_noise_seed(image_seed, step)`.
    #[test]
    fn lcm_renoises_the_clean_estimate_to_the_next_sigma() {
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let mut ops = host(&[1.0, -2.0]);
        let out = sampler_step(
            &mut ops,
            &lcm,
            at(0, 0.75, 0.5),
            &vec![1.0, 2.0],
            &vec![0.5, -1.0],
        )
        .unwrap();
        // x0 = [1.375, 1.25]; 0.5·x0 + 0.5·1.1·[1, -2] = [1.2375, -0.475].
        close(&out, &[1.2375, -0.475]);
        assert_eq!(ops.seeds, vec![lcm_noise_seed(42, 0)]);

        // Step 2 of 6: s = 1.1 − 0.2·0.4 = 1.02.
        let mut ops = host(&[1.0, -2.0]);
        let out = sampler_step(
            &mut ops,
            &lcm,
            at(2, 0.75, 0.5),
            &vec![0.0, 0.0],
            &vec![0.0, 0.0],
        )
        .unwrap();
        close(&out, &[0.51, -1.02]);
        assert_eq!(ops.seeds, vec![lcm_noise_seed(42, 2)]);
    }

    /// The last step lands on the clean estimate and draws nothing.
    #[test]
    fn lcm_last_step_returns_the_clean_estimate() {
        let mut ops = host(&[5.0, 5.0]);
        let out = sampler_step(
            &mut ops,
            &H3ImageSampler::lcm_default(),
            at(5, 0.25, 0.0),
            &vec![1.0, 2.0],
            &vec![0.5, -1.0],
        )
        .unwrap();
        close(&out, &[1.125, 1.75]);
        assert!(ops.seeds.is_empty());
    }

    /// The clip clamps ε to `±noise_clip_std·std(ε)`, with the sample (n − 1) standard deviation.
    #[test]
    fn lcm_clip_bounds_the_noise_by_its_std() {
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.0,
            s_noise_end: 1.0,
            noise_clip_std: 1.0,
        };
        // mean 0, sample variance (9 + 9 + 1 + 1) / 3 = 20/3, std 2.5819889.
        let c = (20.0f32 / 3.0).sqrt();
        let mut ops = host(&[3.0, -3.0, 1.0, -1.0]);
        let zero = vec![0.0; 4];
        let out = sampler_step(&mut ops, &lcm, at(1, 0.75, 0.5), &zero, &zero).unwrap();
        close(&out, &[0.5 * c, -0.5 * c, 0.5, -0.5]);
        // Clip 0 is off: the same draw passes through untouched.
        let mut ops = host(&[3.0, -3.0, 1.0, -1.0]);
        let out = sampler_step(
            &mut ops,
            &H3ImageSampler::lcm(1.0),
            at(1, 0.75, 0.5),
            &zero,
            &zero,
        )
        .unwrap();
        close(&out, &[1.5, -1.5, 0.5, -0.5]);
    }

    /// `s` ramps linearly over `step / (n − 1)`, and a single step uses `s_noise`.
    #[test]
    fn noise_scale_interpolates_from_start_to_end() {
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let want = [1.1, 1.06, 1.02, 0.98, 0.94, 0.9];
        for (i, w) in want.iter().enumerate() {
            assert!((lcm.noise_scale(i, 6) - w).abs() < 1e-6, "step {i}");
        }
        assert_eq!(lcm.noise_scale(0, 1), 1.1);
        assert_eq!(H3ImageSampler::lcm(0.7).noise_scale(3, 6), 0.7);
        assert_eq!(H3ImageSampler::Euler.noise_scale(0, 6), 0.0);
    }

    /// No sampler named and no block is the default LCM at 1.1 → 0.9 with no clipping; `"lcm"`
    /// alone is the same; `"euler"` alone is the released Euler.
    #[test]
    fn the_default_sampler_is_lcm() {
        let default = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        assert_eq!(H3ImageSampler::lcm_default(), default);
        assert_eq!(H3ImageSampler::default(), default);
        assert_eq!(resolve_sampler(None, None).unwrap(), default);
        assert_eq!(
            request_sampler(&crate::GenerationRequest::default()).unwrap(),
            default
        );
        assert_eq!(resolve_sampler(Some("lcm"), None).unwrap(), default);
        assert_eq!(
            resolve_sampler(Some("euler"), None).unwrap(),
            H3ImageSampler::Euler
        );
        assert_eq!(
            resolve_sampler(Some("euler"), Some(H3ImageSampler::Euler)).unwrap(),
            H3ImageSampler::Euler
        );
        assert_eq!(
            resolve_sampler(None, Some(H3ImageSampler::Euler)).unwrap(),
            H3ImageSampler::Euler
        );
        let typed = H3ImageSampler::Lcm {
            s_noise: 1.3,
            s_noise_end: 0.7,
            noise_clip_std: 2.5,
        };
        assert_eq!(resolve_sampler(Some("lcm"), Some(typed)).unwrap(), typed);
        assert_eq!(resolve_sampler(None, Some(typed)).unwrap(), typed);
        assert_eq!(SAMPLER_NAMES, &["euler", "lcm"]);
    }

    #[test]
    fn sampler_refusals_are_typed() {
        let lcm = |s_noise, s_noise_end, noise_clip_std| {
            resolve_sampler(
                None,
                Some(H3ImageSampler::Lcm {
                    s_noise,
                    s_noise_end,
                    noise_clip_std,
                }),
            )
            .unwrap_err()
        };
        assert!(matches!(
            lcm(f32::NAN, 1.0, 0.0),
            H3ImageRefusal::SamplerNonFinite {
                field: "s_noise",
                ..
            }
        ));
        assert!(matches!(
            lcm(1.0, f32::INFINITY, 0.0),
            H3ImageRefusal::SamplerNonFinite {
                field: "s_noise_end",
                ..
            }
        ));
        assert!(matches!(
            lcm(1.0, 1.0, f32::NEG_INFINITY),
            H3ImageRefusal::SamplerNonFinite {
                field: "noise_clip_std",
                ..
            }
        ));
        assert_eq!(
            lcm(0.0, 1.0, 0.0),
            H3ImageRefusal::SamplerNoiseScale { value: 0.0 }
        );
        assert_eq!(
            lcm(-1.0, 1.0, 0.0),
            H3ImageRefusal::SamplerNoiseScale { value: -1.0 }
        );
        assert_eq!(
            lcm(1.0, -0.5, 0.0),
            H3ImageRefusal::SamplerNoiseScaleEnd { value: -0.5 }
        );
        assert_eq!(
            lcm(1.0, 1.0, -1.0),
            H3ImageRefusal::SamplerNoiseClip { value: -1.0 }
        );
        // A ramp down to 0 is admitted: every noised step's scale stays positive.
        resolve_sampler(
            None,
            Some(H3ImageSampler::Lcm {
                s_noise: 1.0,
                s_noise_end: 0.0,
                noise_clip_std: 0.0,
            }),
        )
        .unwrap();
        assert_eq!(
            resolve_sampler(Some("dpmpp_2m"), None).unwrap_err(),
            H3ImageRefusal::UnknownSampler {
                name: "dpmpp_2m".into()
            }
        );
        assert_eq!(
            resolve_sampler(Some("euler"), Some(H3ImageSampler::lcm_default())).unwrap_err(),
            H3ImageRefusal::SamplerConflict {
                named: "euler".into(),
                typed: "lcm"
            }
        );
        assert!(matches!(
            Error::from(lcm(0.0, 1.0, 0.0)),
            Error::Unsupported(_)
        ));
    }

    /// The block is refused on the shared floor by a model that does not advertise it, and its
    /// floats join the request's finiteness floor.
    #[test]
    fn only_an_advertising_model_admits_the_sampler_block() {
        let mut caps = crate::Capabilities {
            min_size: 1,
            max_size: 4096,
            max_count: 1,
            ..Default::default()
        };
        let mut req = crate::GenerationRequest {
            h3_image_sampler: Some(H3ImageSampler::lcm_default()),
            ..Default::default()
        };
        assert!(matches!(
            caps.validate_request("other_model", &req),
            Err(Error::Unsupported(_))
        ));
        caps.supports_h3_image_sampler = true;
        caps.validate_request("minimax_h3_image", &req).unwrap();
        req.h3_image_sampler = Some(H3ImageSampler::lcm(f32::NAN));
        assert_eq!(req.first_nonfinite_float().map(|(f, _)| f), Some("s_noise"));
    }

    /// The per-step noise seed is a pure function of (image seed, step), distinct per step and
    /// per image, and never the image seed itself (the initial latent's).
    #[test]
    fn lcm_noise_seeds_are_deterministic_and_distinct() {
        assert_eq!(lcm_noise_seed(7, 3), lcm_noise_seed(7, 3));
        let mut seen = std::collections::HashSet::new();
        for seed in [0u64, 1, 7, 7090, u64::MAX] {
            for step in 0..6 {
                let s = lcm_noise_seed(seed, step);
                assert_ne!(s, seed);
                assert!(seen.insert(s), "seed {seed} step {step} collides");
            }
        }
    }

    #[test]
    fn reference_latent_grid_crops_to_cells_then_patches() {
        assert_eq!(reference_latent_grid(1024, 1536, (2, 2)), (96, 64));
        // 100 px -> 6 latent cells -> 6; 90 px -> 5 -> 4.
        assert_eq!(reference_latent_grid(90, 100, (2, 2)), (6, 4));
        assert_eq!(REFERENCE_NOISE_AUG, 0.999);
    }
}

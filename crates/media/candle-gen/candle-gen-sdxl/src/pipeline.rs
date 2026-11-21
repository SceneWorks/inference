//! The candle SDXL **txt2img** pipeline (sc-3675) — the proven epic-3494 prototype
//! (`D:\sceneworks-candle-spike\src\bin\candle_sdxl.rs`) lifted out of its standalone CLI/PNG shell
//! and into the backend-neutral [`gen_core::Generator`] contract.
//!
//! What changed vs the spike, and what deliberately did **not**:
//! - **Components** (the GO-validated path): dual CLIP (CLIP-L + CLIP-bigG) loaded **f16** (sc-3674;
//!   the spike used f32) and encoded; UNet **f16**; VAE **f16** with the `madebyollin/sdxl-vae-fp16-fix`
//!   (f16 SDXL VAE NaNs without it); VAE scale **0.13025** (the diffusers SDXL value, not candle's
//!   hardcoded SD1.5 0.18215).
//! - **Perf (sc-3674)**: the UNet attention runs through fused **flash-attention** when the crate is
//!   built `--features flash-attn` AND the runtime toggle ([`crate::set_flash_attn`], default on) is
//!   set — on Blackwell sm_120 that cut steady-state from ~0.32 to ~0.21 s/step and peak VRAM ~21.6→18
//!   GiB. The build feature is the opt-in; the toggle is what the SceneWorks UI exposes.
//! - **Peak VRAM (sc-4987)**: two structural levers on top of sc-3674's 18 GiB high-water mark, both
//!   targeting torch-parity (~9 GiB) at 1024². (1) **Staged sequential load** — each CLIP encoder is
//!   loaded, run, and **dropped** before the next, and *both* are gone before the UNet/VAE even load
//!   (text embeddings are seed-independent, computed once up front), so the dual CLIP (~1.6 GiB f16)
//!   never sits resident through denoise/decode. (2) **VAE tiling** — the VAE decode at 1024² is the
//!   tallest single allocation; [`SdxlVaeDecoder::decode_tiled`] bounds it. **sc-19753 changed its
//!   shape**: it used to tile the whole decode and trapezoidally blend the seams (diffusers'
//!   `enable_vae_tiling`), which gave every tile its own `GroupNorm` statistics and its own mid-block
//!   attention neighbourhood. It now runs the globally-scoped head once and bounds each 3×3
//!   convolution on halo-expanded crops, so the bounded decode tracks the dense one. Gated by
//!   [`crate::vae_tiling_enabled`] (default on) and only *fires* above 512² output (the geometry
//!   policy lives in [`gen_core::tiling`]).
//! - **Deterministic seeding + non-ancestral scheduler (sc-3673)**: initial noise is drawn from a
//!   fixed-algorithm CPU RNG (`StdRng`) seeded by `seed` and moved to the device — NOT candle's CUDA
//!   `device.set_seed`, whose seed→noise mapping was not portable across launch environments and
//!   occasionally collapsed the sample (sc-3498). The sampler is **DDIM (eta=0)**, non-ancestral, so
//!   there is no per-step stochastic noise. Net: generation is a pure function of `(seed, request)`.
//! - **CLI/`emit_event`/PNG/sidecar removed**: progress is `on_progress(Progress::Step/Decoding)`,
//!   cancellation is `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes (no candle-specific worker code).
//! - **Weights come from `spec.weights` (the SDXL snapshot dir)**, not a hardcoded HF repo: UNet +
//!   both text encoders load from the snapshot's component subdirs. The three **model-agnostic**
//!   inputs — the fp16-VAE-fix and the CLIP-L/bigG `tokenizer.json`s — are **passed in** as
//!   [`LoadSpec::components`](gen_core::LoadSpec::components) (`vae_fp16_fix` / `tokenizer_clip_l` /
//!   `tokenizer_clip_bigg`, epic 13657 / sc-13663), never self-fetched.
//!
//! - **Component caching (sc-5037)**: the seed/prompt/resolution-independent [`Components`] (UNet +
//!   VAE) are loaded once and **cached on the generator** across `generate` calls (keyed by the
//!   flash-attn setting), so back-to-back requests skip the ~7 GiB UNet/VAE disk re-read. This is
//!   reconciled with the sc-4987 staged load rather than reverting it: CLIP stays
//!   load-on-demand-and-free (only one encoder resident at a time), and the generator computes the
//!   text embeddings *before* acquiring the cached UNet/VAE — so the cold-call ordering (CLIP freed
//!   before UNet/VAE load) and the ~8.7 GiB peak are preserved; the cache holds only UNet+VAE
//!   resident between calls (a latency win, not a peak-VRAM regression).
//!
//! - **RealVisXL (sc-3677)**: RealVisXL_V5.0 (`SG161222/RealVisXL_V5.0`) shares the SDXL architecture
//!   AND ships the standard diffusers multi-component tree with the *same* component filenames this
//!   pipeline already resolves — `unet/diffusion_pytorch_model.fp16.safetensors`,
//!   `text_encoder{,_2}/model.fp16.safetensors`. So it loads through this exact snapshot path
//!   unmodified; the single-file root checkpoints it also publishes are not needed and no single-file
//!   loader was added (the [`snapshot_file`] component layout is present, not absent). The model-
//!   agnostic VAE-fix + CLIP tokenizers and the production defaults below ([`DEFAULT_STEPS`],
//!   [`DEFAULT_GUIDANCE`], [`VAE_SCALE`]) are shared, matching the Python `SdxlDiffusersAdapter`; the
//!   one accepted sampler difference (DDIM eta=0 vs the adapter's euler_ancestral) is the sc-3673
//!   launch-portable-determinism choice. Parity is locked by `tests/conformance.rs::realvisxl_conformance`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::diffusion_schedule::{SDXL_BETA_END, SDXL_BETA_START, SDXL_TRAIN_STEPS};
use candle_gen::gen_core::sampling::{
    schedule_sigmas, AlphaSchedule, DiscreteModelSampling, LightningPolicy, SamplerPolicy,
    Scheduler,
};
use candle_gen::gen_core::tiling::{TilingConfig, VaeTiling};
use candle_gen::gen_core::{
    self, AdapterSpec, CancelFlag, GenerationRequest, Image, LoadSpec, PidWeights, Progress, Quant,
    WeightsSource,
};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::PidEngine;

/// The PiD backbone (latent-space) tag for SDXL (epic 7840 / sc-7853): SDXL's own `sdxl` VP-frame
/// student (4× SR). Kolors reuses this crate's decode seam via the same `sdxl` tag (shared VAE).
/// Re-exported (sc-8373) so `candle-gen-instantid` loads the SAME `sdxl` student — InstantID composes
/// the SDXL VAE, so there is no InstantID-specific PiD checkpoint.
pub const PID_BACKBONE: &str = "sdxl";
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;

use crate::long_prompt::{self, ChunkPlan};
use crate::SdxlVaeDecoder;
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};

// The vendored, packed-detecting SDXL UNet (sc-5165 / sc-9416): its Linear surface routes through the
// shared `candle_gen::quant` seam, so it loads a pre-quantized MLX tier (SceneWorks/sdxl-base-mlx
// q4/q8) straight from the packed parts. Aliased to avoid clashing with the stock (dense) UNet type.
use crate::unet::{sdxl_unet_config, UNet2DConditionModel as VendoredUNet};
use candle_gen::quant::{PackedConfig, MLX_GROUP_SIZE};
use rand::{rngs::StdRng, SeedableRng};
use tokenizers::Tokenizer;

/// diffusers SDXL VAE `scaling_factor` (candle's example hardcodes the SD1.5 value 0.18215 for `Xl`;
/// 0.13025 is the diffusers-correct one and is what produced correctly-exposed output in the spike).
pub(crate) const VAE_SCALE: f64 = 0.13025;
/// Production SDXL defaults (the SceneWorks `sdxl` row): 30 steps, CFG 7.0 — used when the request
/// omits them.
const DEFAULT_STEPS: usize = 30;
const DEFAULT_GUIDANCE: f64 = 7.0;
/// The sampler an **omitted** `req.sampler` resolves to (sc-10826): the curated `ddim` solver — a
/// k-diffusion DDIM (eta=0, non-ancestral) over the SDXL ε-schedule, driven by the unified
/// [`gen_core::sampling`] framework via [`Pipeline::denoise_curated`]. It **replaces** the native
/// candle-transformers `DDIMScheduler` inference loop, which rendered a ghosted, translucent
/// double-exposure (guidance-invariant) on the default path while every curated solver — including
/// this curated `ddim` — is clean. Being eta=0 / non-ancestral it keeps the sc-3673 launch-portable
/// determinism the native default targeted (generation stays a pure function of `(seed, request)`),
/// and `ddim` is part of the advertised curated vocabulary ([`candle_gen::curated_sampler_names`]),
/// so it remains a valid selection.
const DEFAULT_SAMPLER: &str = "ddim";

/// The curated solver a txt2img render resolves `req.sampler` to (sc-10826), factored out of
/// [`Pipeline::render`] so the default-routing rule is unit-testable without a GPU:
/// - `Some(LIGHTNING_SAMPLER)` ⇒ `None` — the `lightning` render owns its own few-step path.
/// - `None` (omitted) ⇒ `Some(DEFAULT_SAMPLER)` — the curated `ddim` solver (the fix: the native
///   candle-transformers DDIM loop that ghosted is gone).
/// - any other `Some(name)` ⇒ `Some(name)` verbatim — the curated solver by that name (an unknown
///   name euler-falls-back inside `run_curated_sampler`, N3).
fn resolve_sampler(sampler: Option<&str>) -> Option<&str> {
    if sampler == Some(LIGHTNING_SAMPLER) {
        None
    } else {
        Some(sampler.unwrap_or(DEFAULT_SAMPLER))
    }
}

/// The few-step **Lightning** sampler id (sc-6128) — diffusers Euler-trailing, selected per request
/// via `req.sampler` and advertised in [`crate::descriptor`]'s `samplers`. The SceneWorks worker forces
/// it for the `realvisxl_lightning` model id; distilled Lightning checkpoints (RealVisXL Lightning /
/// SDXL-Lightning) render correctly in 2–8 steps through this schedule, where DDIM at the same step
/// count produces mush.
pub(crate) const LIGHTNING_SAMPLER: &str = "lightning";
/// Lightning's few-step default when the request omits `steps` — matches `mlx-gen-sdxl`'s
/// `accel_defaults("lightning")` (4 steps, CFG off). The worker typically sends an explicit count
/// (the AC eyeballs ~5).
const LIGHTNING_DEFAULT_STEPS: usize = 4;
/// Build SDXL's ε-prediction α-cumprod schedule (`scaled_linear` β over 1000 train steps) — the
/// [`DiscreteModelSampling`] source the curated unified-sampler path integrates over. Shared by the
/// txt2img [`Pipeline::denoise_curated`] (sc-7124), the Lightning policy, and the conditioned
/// [`crate::ip_provider`] curated denoise (sc-7297), so they speak one SDXL noise schedule.
pub(crate) fn sdxl_alpha_schedule() -> Result<AlphaSchedule> {
    Ok(AlphaSchedule::scaled_linear(
        SDXL_TRAIN_STEPS,
        SDXL_BETA_START,
        SDXL_BETA_END,
    ))
}

/// Build the SDXL-**Lightning** sampler *policy* (sc-6128) for `num_steps`: diffusers
/// `EulerDiscreteScheduler(timestep_spacing="trailing", final_sigmas_type="zero")`, ε-prediction. The
/// schedule math is the backend-neutral [`gen_core::sampling::LightningPolicy`] — the **same** policy
/// the `mlx-gen-sdxl` `LightningSampler` drives, so no candle gen-core pin bump is needed and the two
/// backends share the reference trailing-spacing + interpolated σ table. The candle side is only the
/// ~5-line tensor application in [`Pipeline::denoise_lightning`].
pub(crate) fn lightning_policy(num_steps: usize) -> Result<LightningPolicy> {
    let sched = sdxl_alpha_schedule()?;
    Ok(LightningPolicy::new(&sched, SDXL_TRAIN_STEPS, num_steps))
}

/// The diffusers filename the fp16-stable SDXL VAE (`madebyollin/sdxl-vae-fp16-fix`) ships under — the
/// base SDXL VAE NaNs in f16, so this model-agnostic replacement is provisioned by the consumer as the
/// `vae_fp16_fix` component (epic 13657, sc-13663). Used to resolve a component staged as a directory
/// (`WeightsSource::Dir`) to its weight file; a component staged as a `File` is used verbatim.
pub(crate) const VAE_FIX_FILE: &str = "diffusion_pytorch_model.safetensors";

/// The SDXL lane's three **passed-in** model components (epic 13657, sc-13658 contract): the two
/// model-agnostic CLIP tokenizers and the fp16-stable VAE. Inference NEVER self-fetches these — before
/// sc-13663 they were downloaded on the render path from three pinned upstream repos
/// (`openai/clip-vit-large-patch14`, `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`,
/// `madebyollin/sdxl-vae-fp16-fix`); now the consumer stages each as a local
/// [`LoadSpec::components`](gen_core::LoadSpec::components) entry (or, for the bespoke edit/IP/InstantID
/// providers, a `*Paths` field) under these registered ids, and a missing one is a load-time contract
/// error ([`gen_core::require_component`]) — not a mid-render fetch.
pub(crate) const COMPONENT_TOKENIZER_CLIP_L: &str = "tokenizer_clip_l";
pub(crate) const COMPONENT_TOKENIZER_CLIP_BIGG: &str = "tokenizer_clip_bigg";
pub(crate) const COMPONENT_VAE_FP16_FIX: &str = "vae_fp16_fix";

/// The SDXL lane's required component ids, in the descriptor-advertised order — the single source of
/// truth for [`crate::descriptor`]'s `required_components` and the [`SdxlComponents::from_spec`] load
/// gate. Registered on [`LoadSpec::components`](gen_core::LoadSpec::components).
pub(crate) const REQUIRED_COMPONENTS: &[&str] = &[
    COMPONENT_TOKENIZER_CLIP_L,
    COMPONENT_TOKENIZER_CLIP_BIGG,
    COMPONENT_VAE_FP16_FIX,
];
/// A fused LDM checkpoint carries its own VAE, so only the two model-agnostic tokenizer assets must
/// be staged. The ordinary snapshot route continues to require the fp16-fix VAE above.
pub(crate) const LDM_REQUIRED_COMPONENTS: &[&str] =
    &[COMPONENT_TOKENIZER_CLIP_L, COMPONENT_TOKENIZER_CLIP_BIGG];

/// The caller-staged sources for the three SDXL components, resolved + validated once at load and
/// threaded to every consumption site (the txt2img tokenizers/VAE, and — for the LoadSpec-driven
/// generator + trainer — the whole render/train path) in place of the deleted render-path self-fetch. Each is the
/// raw [`WeightsSource`] the caller staged; the concrete weight file is resolved at the point of use
/// via [`resolve_tokenizer_file`] / [`resolve_vae_file`] (so a `Dir` or a direct `File` both work).
#[derive(Clone, Debug)]
pub(crate) struct SdxlComponents {
    pub(crate) tokenizer_clip_l: WeightsSource,
    pub(crate) tokenizer_clip_bigg: WeightsSource,
    pub(crate) vae_fp16_fix: Option<WeightsSource>,
}

impl SdxlComponents {
    /// Resolve + validate the three components from a [`LoadSpec`] at load time — the sc-13658 contract
    /// gate for the SDXL lane. Rejects any component key the model does not declare
    /// ([`gen_core::reject_unknown_components`], typed `Unsupported`), then requires each of the three
    /// ([`gen_core::require_component`]), producing a caller-actionable load-time error naming the
    /// model, the missing id, and the `with_component` builder — never a mid-render fetch. Used by the
    /// registered generator ([`crate::load`]) and trainer ([`crate::load_trainer`]) loads; the bespoke
    /// edit/IP/InstantID providers carry the same sources on their `*Paths` structs instead.
    pub(crate) fn from_spec(spec: &LoadSpec, model_id: &str) -> gen_core::Result<Self> {
        gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, model_id)?;
        let tokenizer_clip_l = gen_core::require_component(
            spec,
            COMPONENT_TOKENIZER_CLIP_L,
            model_id,
            "CLIP-L tokenizer",
        )?
        .clone();
        let tokenizer_clip_bigg = gen_core::require_component(
            spec,
            COMPONENT_TOKENIZER_CLIP_BIGG,
            model_id,
            "CLIP-bigG tokenizer",
        )?
        .clone();
        let vae_fp16_fix = if matches!(spec.weights, WeightsSource::File(_)) {
            spec.components.get(COMPONENT_VAE_FP16_FIX).cloned()
        } else {
            Some(
                gen_core::require_component(
                    spec,
                    COMPONENT_VAE_FP16_FIX,
                    model_id,
                    "fp16-fix VAE",
                )?
                .clone(),
            )
        };
        Ok(Self {
            tokenizer_clip_l,
            tokenizer_clip_bigg,
            vae_fp16_fix,
        })
    }
}

/// Resolve a tokenizer component [`WeightsSource`] to its `tokenizer.json` file: a `File` is used
/// verbatim; a `Dir` joins the diffusers `tokenizer.json` name (the CLIP tokenizer snapshot layout).
pub(crate) fn resolve_tokenizer_file(src: &WeightsSource) -> PathBuf {
    match src {
        WeightsSource::File(f) => f.clone(),
        WeightsSource::Dir(d) => d.join("tokenizer.json"),
    }
}

/// Resolve the VAE component [`WeightsSource`] to its `.safetensors` weight file: a `File` is used
/// verbatim; a `Dir` joins the fp16-fix VAE's diffusers filename ([`VAE_FIX_FILE`]).
pub(crate) fn resolve_vae_file(src: &WeightsSource) -> PathBuf {
    match src {
        WeightsSource::File(f) => f.clone(),
        WeightsSource::Dir(d) => d.join(VAE_FIX_FILE),
    }
}

/// The SDXL VAE's tiling geometry (sc-4987): the decoder upsamples latents ×8 spatially, and an image
/// VAE has **no temporal axis** — so temporal scale 1, non-causal (the `[B, 4, h, w]` latent is tiled
/// on the two spatial axes only, with the singleton temporal axis a no-op in [`TilingConfig::plan`]).
const SDXL_VAE_TILING: VaeTiling = VaeTiling {
    spatial_scale: 8,
    temporal_scale: 1,
    causal_temporal: false,
    // 128 ch at full resolution (block_out_channels[0]). At T=1 the write bound is ~4096x4096.
    full_res_channels: 128,
};

/// Native SDXL VAE adapter for the backend-generic latent-decoder seam. The seam always receives the
/// normalized sampler latent; this wrapper owns SDXL's `1 / VAE_SCALE` de-normalization and the
/// established optional tiled decode, so InstantID and the registered SDXL lanes no longer branch
/// around the trait for their native default.
pub struct SdxlLatentDecoder<'a> {
    vae: &'a SdxlVaeDecoder,
    decode_dtype: Option<DType>,
}

impl<'a> SdxlLatentDecoder<'a> {
    pub fn new(vae: &'a SdxlVaeDecoder) -> Self {
        Self {
            vae,
            decode_dtype: None,
        }
    }

    /// Select the dtype at the VAE boundary. Imported single-file SDXL checkpoints carry their
    /// original VAE, which is loaded in f32 to avoid the base model's unstable fp16 decode; the
    /// native snapshot route leaves the sampler latent in its existing compute dtype.
    pub fn with_decode_dtype(vae: &'a SdxlVaeDecoder, decode_dtype: DType) -> Self {
        Self {
            vae,
            decode_dtype: Some(decode_dtype),
        }
    }

    fn unscale(&self, latents: &Tensor) -> Result<Tensor> {
        let unscaled = (latents / VAE_SCALE)?;
        match self.decode_dtype {
            Some(dtype) => Ok(unscaled.to_dtype(dtype)?),
            None => Ok(unscaled),
        }
    }
}

impl LatentDecoder for SdxlLatentDecoder<'_> {
    fn input_latent_space(&self) -> Option<&candle_gen::gen_core::LatentSpace> {
        Some(&candle_gen::gen_core::SDXL_LATENT_SPACE)
    }

    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.vae.decode(&self.unscale(latents)?)
    }

    /// Bounded decode with dense-image GroupNorm semantics (sc-19753).
    ///
    /// The globally-scoped head — `post_quant_conv`, `conv_in` and the mid block's full-grid
    /// attention — runs once on the whole latent; in the tail every `GroupNorm` reduces the full
    /// layer activation and only halo-expanded 3×3 convolution work is tiled. This replaced a
    /// whole-decode `tile_blend_decode`, under which each tile normalized against its own crop and
    /// attended only to its own tokens — a different decode, not a blend artifact.
    ///
    /// `tiling.spatial.tile_px` bounds each convolution crop in output pixels. The configured
    /// overlap remains part of the public tiling contract and policy identity, but halo/core
    /// arithmetic needs no blend of whole-decode outputs.
    fn decode_tiled(
        &self,
        latents: &Tensor,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(CandleError::Canceled);
        }
        let unscaled = self.unscale(latents)?;
        let (_, _, h, w) = unscaled.dims4()?;
        if tiling.needs_tiling(SDXL_VAE_TILING, 1, h as i32, w as i32) {
            let tile_px = tiling
                .spatial
                .as_ref()
                .ok_or_else(|| {
                    CandleError::Msg("sdxl tiled decode requires spatial tiling".into())
                })?
                .tile_px;
            return self
                .vae
                .decode_tiled(&unscaled, tile_px.max(3) as usize, cancel);
        }
        self.vae.decode(&unscaled)
    }
}

/// Which of the two SDXL CLIP encoders — selects the tokenizer repo, the snapshot weights subpath,
/// and which `StableDiffusionConfig` clip config to use.
#[derive(Clone, Copy)]
pub(crate) enum Clip {
    /// CLIP-L (`text_encoder/`) — `openai/clip-vit-large-patch14` tokenizer.
    L,
    /// OpenCLIP bigG (`text_encoder_2/`) — `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` tokenizer.
    BigG,
}

impl Clip {
    /// `(tokenizer repo, snapshot weights subpath)`.
    pub(crate) fn sources(&self) -> (&'static str, &'static str) {
        match self {
            Clip::L => (
                "openai/clip-vit-large-patch14",
                "text_encoder/model.fp16.safetensors",
            ),
            Clip::BigG => (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "text_encoder_2/model.fp16.safetensors",
            ),
        }
    }

    /// The encoder's diffusers component subdir (`text_encoder` / `text_encoder_2`) — the base for its
    /// `config.json` (packed-detect) and its **packed** weight file (`model.safetensors`, not the
    /// dense `.fp16` name).
    pub(crate) fn subdir(&self) -> &'static str {
        match self {
            Clip::L => "text_encoder",
            Clip::BigG => "text_encoder_2",
        }
    }

    /// The vendored CLIP config for this encoder (sc-9527): CLIP-L (`text_encoder/`) vs OpenCLIP bigG
    /// (`text_encoder_2/`). Mirrors the stock `clip::Config::sdxl()` / `sdxl2()` the pipeline uses.
    pub(crate) fn vendored_config(&self) -> crate::clip::Config {
        match self {
            Clip::L => crate::clip::Config::sdxl(),
            Clip::BigG => crate::clip::Config::sdxl2(),
        }
    }
}

/// The two SDXL CLIP tokenizers (CLIP-L + CLIP-bigG), loaded+parsed **once** and cached on the
/// generator, reused across every `text_embeddings` call (sc-8991 / F-011) instead of re-reading the
/// `tokenizer.json` files and re-parsing them on each encode. Model-agnostic (the caller-staged
/// `tokenizer_clip_l` / `tokenizer_clip_bigg` components, sc-13663), so a single pair serves every SDXL
/// snapshot the generator renders. These carry no VRAM, so caching them does not affect the sc-4987
/// CLIP-weight peak lever.
pub(crate) struct SdxlTokenizers {
    tok_l: Tokenizer,
    tok_g: Tokenizer,
}

impl SdxlTokenizers {
    /// Load both CLIP tokenizers from the caller-staged [`WeightsSource`] components (epic 13657,
    /// sc-13663) — the `tokenizer_clip_l` / `tokenizer_clip_bigg` ids resolved by the load gate. No
    /// render-path self-fetch: the paths are provisioned by the consumer and resolved via
    /// [`resolve_tokenizer_file`]. Call once per generator.
    pub(crate) fn load(tok_l: &WeightsSource, tok_g: &WeightsSource) -> Result<Self> {
        let tok_l_file = resolve_tokenizer_file(tok_l);
        let tok_g_file = resolve_tokenizer_file(tok_g);
        let tok_l = Tokenizer::from_file(&tok_l_file).map_err(|e| {
            CandleError::Msg(format!(
                "sdxl: load CLIP-L tokenizer from {}: {e}",
                tok_l_file.display()
            ))
        })?;
        let tok_g = Tokenizer::from_file(&tok_g_file).map_err(|e| {
            CandleError::Msg(format!(
                "sdxl: load CLIP-bigG tokenizer from {}: {e}",
                tok_g_file.display()
            ))
        })?;
        #[cfg(feature = "model-inference-proof")]
        candle_gen::gen_core::model_input_proof::record_reader_paths(
            "candle-gen-sdxl::pipeline::SdxlTokenizers::load/tokenizers",
            candle_gen::gen_core::model_input_proof::InputRole::Tokenizer,
            &[&tok_l_file, &tok_g_file],
        );
        Ok(Self { tok_l, tok_g })
    }
}

/// A txt2img pipeline handle. sc-4987 made loading **staged**: this carries only the
/// `StableDiffusionConfig` (the per-request latent dims), the snapshot `root`, and the compute
/// device/dtype — the heavy components (CLIP, UNet, VAE) are loaded *inside* [`generate`] in the
/// order they are needed and dropped as soon as they are not, so the dual CLIP is freed before the
/// UNet/VAE ever allocate. (Pre-sc-4987 this struct held all four components resident at once.)
pub(crate) struct Pipeline {
    config: StableDiffusionConfig,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// LoRA/LoKr adapters to merge into the UNet at component-load time (sc-5165). Fixed for the
    /// generator's lifetime (they come from the `LoadSpec`), so they do not enter the component cache
    /// key — only flash-attn does. Empty ⇒ the stock mmap `build_unet` path (zero regression).
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), built into the cached
    /// [`Components`] so the PiD engine loads once alongside the UNet/VAE. `None` ⇒ native VAE decode.
    pid_spec: Option<PidWeights>,
    /// The caller-staged `vae_fp16_fix` component (epic 13657, sc-13663) — the fp16-stable VAE weight
    /// source, resolved in [`load_components`](Self::load_components) in place of the deleted render-path self-fetch.
    vae_fix: Option<WeightsSource>,
    ldm: Option<Arc<crate::ldm::LdmComponents>>,
    quant: Option<Quant>,
}

/// The seed- and prompt-independent heavy components (UNet + f16 VAE), `Arc`-shared so they can be
/// **cached on the generator across `generate` calls** (sc-5037) and cheaply cloned out from under
/// the cache lock for a render. SDXL's UNet/VAE are resolution-agnostic (`build_unet`/`build_vae`
/// read only the fixed `unet`/`autoencoder` sub-configs, never the latent dims), so a single cached
/// pair serves every request size; the only construction input that varies is flash-attn, which the
/// generator keys the cache on. CLIP is deliberately **not** here — it stays load-on-demand-and-free
/// (the sc-4987 peak-VRAM lever), so caching the UNet/VAE does not make the dual CLIP resident.
#[derive(Clone)]
pub(crate) struct Components {
    pub(crate) unet: SdxlUnet,
    pub(crate) vae: Arc<SdxlVaeDecoder>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
    pub(crate) pid: Option<Arc<PidEngine>>,
}

/// The SDXL denoise UNet, in one of two builds that share the txt2img `forward(x, t, ehs)` contract
/// (sc-9416):
///
/// - [`Self::Stock`] — the stock candle-transformers `UNet2DConditionModel`, built for a **dense**
///   diffusers snapshot (bf16/f16). Byte-identical to pre-sc-9416, incl. the fused flash-attention
///   path; this is the default for every dense SDXL/RealVisXL checkpoint (zero regression).
/// - [`Self::Vendored`] — the crate's vendored UNet, whose Linear surface packed-detects through the
///   shared `candle_gen::quant`. Built **only** for a pre-quantized MLX tier
///   (`SceneWorks/sdxl-base-mlx` q4/q8), where the whole attention/FF/proj/time-embed Linear surface
///   loads straight from the packed `{weight u32, scales, biases}` parts (no dense staging) and the
///   convolutions + norms stay dense. Runs the math attention (the vendored flash path is a stub).
///
/// Both are `Arc`-shared so the seed/prompt-independent UNet is cached across `generate` calls (sc-5037)
/// and cheaply cloned per render.
#[derive(Clone)]
pub(crate) enum SdxlUnet {
    Stock(Arc<UNet2DConditionModel>),
    Vendored(Arc<VendoredUNet>),
    Packed(Arc<VendoredUNet>),
}

impl SdxlUnet {
    /// The txt2img denoise forward, dispatched to whichever build. Both compute the SDXL ε-prediction
    /// `[B, 4, h, w]` for `(latents, timestep, dual-CLIP embeddings)` — the packed vendored UNet is
    /// pinned bit-identical to the stock UNet on a dense build by the vendored-vs-stock parity test.
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Stock(u) => Ok(u.forward(xs, timestep, encoder_hidden_states)?),
            Self::Vendored(u) | Self::Packed(u) => {
                Ok(u.forward(xs, timestep, encoder_hidden_states)?)
            }
        }
    }
}

/// One SDXL CLIP text encoder, in one of two builds that share the `forward(ids) -> last hidden`
/// contract (sc-9527):
///
/// - [`Self::Stock`] — the stock candle-transformers `ClipTextTransformer`, built for a **dense**
///   diffusers snapshot. Byte-identical to the pre-sc-9527 txt2img path (zero regression on every
///   dense SDXL/RealVisXL checkpoint).
/// - [`Self::Vendored`] — the crate's vendored CLIP tower, whose Linear surface packed-detects through
///   `candle_gen::quant`. Built **only** for a pre-quantized MLX tier (`SceneWorks/sdxl-base-mlx`
///   q4/q8), where every attention / MLP `Linear` loads straight from the packed
///   `{weight u32, scales, biases}` parts (no dense staging).
///
/// The vendored tower is pinned bit-identical to the stock one on a dense build by the
/// `clip::tests::vendored_dense_matches_stock` parity test.
pub(crate) enum CandleModule {
    Stock(stable_diffusion::clip::ClipTextTransformer),
    Vendored(crate::clip::ClipTextTransformer),
}

impl CandleModule {
    /// The last-hidden-state forward `ids [B, S] -> [B, S, embed_dim]`, dispatched to whichever build.
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Stock(m) => Ok(m.forward(ids)?),
            Self::Vendored(m) => Ok(m.forward(ids)?),
        }
    }
}

impl Pipeline {
    /// Build the (light) pipeline handle for the SDXL snapshot `root` at the given device/dtype (f16)
    /// and request dims. This does **no** weight I/O — the config's only request-dependent fields are
    /// the latent dims; the heavy components load lazily in [`generate`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        width: u32,
        height: u32,
        adapters: &[AdapterSpec],
        pid_spec: Option<PidWeights>,
        vae_fix: Option<WeightsSource>,
        ldm: Option<Arc<crate::ldm::LdmComponents>>,
        quant: Option<Quant>,
    ) -> Result<Self> {
        // The config's only request-dependent fields are the latent dims; the component configs
        // (clip/clip2/unet/autoencoder) are fixed for SDXL.
        let config = StableDiffusionConfig::sdxl(None, Some(height as usize), Some(width as usize));
        Ok(Self {
            config,
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
            adapters: adapters.to_vec(),
            pid_spec,
            vae_fix,
            ldm,
            quant,
        })
    }

    /// SDXL dual-CLIP conditioning: encode `prompt` (cond) and `uncond` through both encoders, stack
    /// `[uncond, cond]` on the batch axis, and concatenate the two encoders on the feature axis —
    /// shape `[2, tokens, 2048]`, cast to the compute dtype. Mirrors the spike's `text_embeddings`.
    ///
    /// **Caller obligation (sc-14195):** this is **always** batch 2, regardless of the request's
    /// guidance — the uncond row is encoded unconditionally. So every denoise lane that runs
    /// CFG-**off** (a single conditioned forward over a batch-1 latent: `lightning`, or any sampler
    /// at `guidance ≤ 1.0`) must narrow to the cond row (index 1) before the UNet forward, or the
    /// batch-1 query meets batch-2 cross-attention K/V and the UNet dies in the attention matmul.
    ///
    /// sc-4987: each encoder is loaded, run, and dropped **inside** [`encode_one`] before the next is
    /// loaded — so the two CLIP encoders are never co-resident, and both are gone when this returns.
    /// sc-5037: the generator calls this **before** acquiring the (possibly cached-resident) UNet/VAE,
    /// preserving the cold-call ordering (CLIP freed before they load); on a warm call the UNet/VAE are
    /// already resident, but only one CLIP encoder is ever resident at a time (`build_unet`+VAE ≈ 7 GiB
    /// + one CLIP ≤ 1.4 GiB stays under the denoise-time peak, so the sc-4987 high-water is preserved).
    pub(crate) fn text_embeddings(
        &self,
        toks: &SdxlTokenizers,
        prompt: &str,
        uncond: &str,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        // sc-20528: a prompt past CLIP's 77-token window is chunked, not rejected. The chunk count is
        // decided ONCE for the whole request — the two encoders are concatenated on the feature axis
        // and `[uncond, cond]` are stacked on the batch axis, so all four encodings must land on the
        // same sequence length. The negative prompt therefore takes exactly the same path as the
        // positive one (topped up with empty windows when it is the shorter of the two).
        let (plan_l, plan_g) = self.chunk_plans(toks)?;
        let chunks = long_prompt::common_chunks(
            &[(&plan_l, &toks.tok_l), (&plan_g, &toks.tok_g)],
            &[uncond, prompt],
        )?;
        let l = self.encode_one(
            Clip::L,
            &toks.tok_l,
            &plan_l,
            prompt,
            uncond,
            chunks,
            on_progress,
        )?;
        let g = self.encode_one(
            Clip::BigG,
            &toks.tok_g,
            &plan_g,
            prompt,
            uncond,
            chunks,
            on_progress,
        )?;
        Ok(Tensor::cat(&[l, g], D::Minus1)?)
    }

    pub(crate) fn text_embeddings_resident(
        &self,
        toks: &SdxlTokenizers,
        models: &(CandleModule, CandleModule),
        prompt: &str,
        uncond: &str,
    ) -> Result<Tensor> {
        let (plan_l, plan_g) = self.chunk_plans(toks)?;
        Self::record_text_representation(Clip::L, &models.0);
        Self::record_text_representation(Clip::BigG, &models.1);
        let chunks = long_prompt::common_chunks(
            &[(&plan_l, &toks.tok_l), (&plan_g, &toks.tok_g)],
            &[uncond, prompt],
        )?;
        let l = self.encode_loaded(&models.0, &toks.tok_l, &plan_l, prompt, uncond, chunks)?;
        let g = self.encode_loaded(&models.1, &toks.tok_g, &plan_g, prompt, uncond, chunks)?;
        Ok(Tensor::cat(&[l, g], D::Minus1)?)
    }

    /// The per-encoder [`ChunkPlan`] pair for this pipeline's CLIP configs (`pad_with` +
    /// `max_position_embeddings`), built from the cached tokenizers. Cheap — no weights, no tensors.
    fn chunk_plans(&self, toks: &SdxlTokenizers) -> Result<(ChunkPlan, ChunkPlan)> {
        let clip2 = self
            .config
            .clip2
            .as_ref()
            .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?;
        let plan_l = ChunkPlan::new(
            &toks.tok_l,
            self.config.clip.pad_with.as_deref(),
            self.config.clip.max_position_embeddings,
        )?;
        let plan_g = ChunkPlan::new(
            &toks.tok_g,
            clip2.pad_with.as_deref(),
            clip2.max_position_embeddings,
        )?;
        Ok((plan_l, plan_g))
    }

    /// Load one CLIP encoder, encode `[uncond, cond]` through it (each text split into `chunks`
    /// windows of `max_position_embeddings` ids, sc-20528), and return the embeddings — the encoder
    /// weights are loaded into a local and **dropped when this function returns** (sc-4987), freeing
    /// its VRAM before the next encoder / the UNet load.
    ///
    /// `plan` is this encoder's half of [`Self::chunk_plans`]; `chunks` is the request-wide window
    /// count, the SAME for both encoders and both texts, so the returned
    /// `[2, chunks·window, embed_dim]` tensors concatenate on the feature axis.
    #[allow(clippy::too_many_arguments)]
    fn encode_one(
        &self,
        which: Clip,
        tokenizer: &Tokenizer,
        plan: &ChunkPlan,
        prompt: &str,
        uncond: &str,
        chunks: usize,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let text_model = self.load_text_model(which)?;
        Self::record_text_representation(which, &text_model);
        let result = candle_gen::gen_core::with_prompt_encoding(on_progress, || {
            self.encode_loaded(&text_model, tokenizer, plan, prompt, uncond, chunks)
        });
        candle_gen::residency::synchronize_result(&self.device, result)
    }

    pub(crate) fn load_text_models(&self) -> Result<(CandleModule, CandleModule)> {
        Ok((
            self.load_text_model(Clip::L)?,
            self.load_text_model(Clip::BigG)?,
        ))
    }

    fn record_text_representation(which: Clip, model: &CandleModule) {
        let key = match which {
            Clip::L => "sdxl/clip_l/representation",
            Clip::BigG => "sdxl/clip_bigg/representation",
        };
        let representation = match model {
            CandleModule::Stock(_) => "dense",
            CandleModule::Vendored(_) => "packed_or_imported",
        };
        candle_gen::execution::policy(key, representation.into());
    }

    fn load_text_model(&self, which: Clip) -> Result<CandleModule> {
        let (_tok_repo, weights_sub) = which.sources();
        let clip_cfg = match which {
            Clip::L => &self.config.clip,
            Clip::BigG => self
                .config
                .clip2
                .as_ref()
                .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?,
        };
        // The tokenizer is now loaded+parsed ONCE on the generator (sc-8991 / F-011) from the
        // caller-staged component (sc-13663) and passed in, rather than re-read per encode. The CLIP
        // *weights* still load and drop inside this function (the sc-4987 peak-VRAM lever); only the
        // tiny tokenizer is cached.
        //
        // sc-9527 (sc-9089j follow-up to sc-9416): the MLX SDXL tiers ALSO pack the dual CLIP text
        // encoders (a `quantization` block in `text_encoder{,_2}/config.json` + `.scales`-packed Linears
        // under `model.safetensors`). The txt2img conditioning uses only each encoder's last hidden
        // state (`forward`), so we build the **vendored, packed-detecting** CLIP tower when the tier is
        // packed — every Linear (attn q/k/v/out_proj, MLP fc1/fc2) loads straight from the packed parts —
        // and the stock dense builder otherwise (byte-identical, pinned by the vendored-vs-stock parity
        // test). The `group_size` is threaded from the component config (sc-9410).
        let ldm_map = self.ldm.as_ref().map(|components| match which {
            Clip::L => &components.clip_l,
            Clip::BigG => &components.clip_bigg,
        });
        let text_model: CandleModule = if let Some(map) = ldm_map {
            let tower = if let Some(quant) = self.quant {
                // Stage the dense fused CLIP map in system RAM, then fold each projection directly
                // onto the compute device. Building the tower on the accelerator first would require
                // the dense tier to fit before Q4/Q8 could make it smaller.
                let cpu = Device::Cpu;
                let vb = VarBuilder::from_tensors(map.clone(), self.dtype, &cpu);
                let mut tower = crate::clip::ClipTextTransformer::new_gs(
                    vb,
                    &which.vendored_config(),
                    candle_gen::quant::MLX_GROUP_SIZE,
                )?;
                tower.quantize_onto(quant, &self.device)?;
                tower
            } else {
                let vb = VarBuilder::from_tensors(map.clone(), self.dtype, &self.device);
                crate::clip::ClipTextTransformer::new_gs(
                    vb,
                    &which.vendored_config(),
                    candle_gen::quant::MLX_GROUP_SIZE,
                )?
            };
            CandleModule::Vendored(tower)
        } else {
            match detect_packed_clip(&self.root, &which)? {
                Some((packed_file, group_size)) => {
                    let vs =
                        candle_gen::mmap_var_builder(&[packed_file], self.dtype, &self.device)?;
                    let tower = crate::clip::ClipTextTransformer::new_gs(
                        vs,
                        &which.vendored_config(),
                        group_size,
                    )?;
                    CandleModule::Vendored(tower)
                }
                None => {
                    // sc-3674: load CLIP at the compute dtype (f16), not the spike's F32. The fp16
                    // safetensors load directly, the forward runs f16 (diffusers loads CLIP fp16 too), and it
                    // halves the text-encoder VRAM (CLIP-bigG ~2.8→1.4 GiB) with no visible quality change.
                    // The embeddings are cast to `dtype` below.
                    let stock = stable_diffusion::build_clip_transformer(
                        clip_cfg,
                        snapshot_file(&self.root, weights_sub)?,
                        &self.device,
                        self.dtype,
                    )?;
                    CandleModule::Stock(stock)
                }
            }
        };

        Ok(text_model)
    }

    fn encode_loaded(
        &self,
        text_model: &CandleModule,
        tokenizer: &Tokenizer,
        plan: &ChunkPlan,
        prompt: &str,
        uncond: &str,
        chunks: usize,
    ) -> Result<Tensor> {
        // sc-20528: one forward per CLIP window, concatenated on the sequence axis (the A1111/compel
        // "long prompt weighting" shape). A prompt that fits produces exactly one window whose ids are
        // the tokenizer output right-padded with the pad token — bit-for-bit the pre-sc-20528 encoding,
        // so nothing about a ≤77-token render changes.
        let encode = |text: &str| -> Result<Tensor> {
            let rows = plan.rows_aligned(tokenizer, text, chunks)?;
            let mut hidden = Vec::with_capacity(rows.len());
            for row in &rows {
                let ids = Tensor::new(row.as_slice(), &self.device)?.unsqueeze(0)?;
                hidden.push(text_model.forward(&ids)?);
            }
            // Take the single window straight through rather than routing it via `cat` — the ≤77 path
            // stays the identical tensor the old code produced.
            if hidden.len() == 1 {
                Ok(hidden.remove(0))
            } else {
                Ok(Tensor::cat(&hidden, 1)?)
            }
        };

        let cond = encode(prompt)?;
        let uncond = encode(uncond)?;
        Ok(Tensor::cat(&[uncond, cond], 0)?.to_dtype(self.dtype)?)
        // `text_model` drops here, freeing this encoder's weights before the caller loads the next
        // (sc-4987). The `tokenizer` is borrowed from the generator's cache and outlives this call.
    }

    /// Load the heavy [`Components`] (UNet + f16 VAE) for the given flash-attn setting. The UNet reads
    /// from the snapshot (fused flash-attention when built `--features flash-attn` AND `use_flash_attn`
    /// — sc-3674); the f16-stable VAE (`madebyollin/sdxl-vae-fp16-fix`) resolves from the caller-staged
    /// `vae_fp16_fix` component (sc-13663). The generator owns the caching of the result across calls
    /// (sc-5037); this is the cache-miss loader.
    pub(crate) fn load_components(&self, use_flash_attn: bool) -> Result<Components> {
        let unet = self.load_unet(use_flash_attn)?;
        let (vae, pid) = self.load_decoders()?;
        Ok(Components { unet, vae, pid })
    }

    pub(crate) fn load_unet(&self, use_flash_attn: bool) -> Result<SdxlUnet> {
        // sc-9416: a **packed** MLX tier (`SceneWorks/sdxl-base-mlx` q4/q8) ships its UNet under the
        // non-`.fp16` filename with a `quantization` block in `unet/config.json` and `.scales`-packed
        // Linear weights. Detect it and load the vendored packed-detecting UNet straight from the packed
        // parts (no dense staging); every dense snapshot keeps the stock build below, unchanged.
        let unet = if let Some(ldm) = &self.ldm {
            let mut raw = ldm.unet.clone();
            let table = crate::adapters::build_sdxl_kohya_table(&raw);
            let cpu = Device::Cpu;
            let build_device = if self.quant.is_some() {
                &cpu
            } else {
                &self.device
            };
            let vs = VarBuilder::from_tensors(raw.clone(), self.dtype, build_device);
            let mut vendored = VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?;
            if !self.adapters.is_empty() {
                let linear = crate::adapters::install_additive(
                    &mut vendored,
                    &self.adapters,
                    &table,
                    build_device,
                )?;
                let conv = crate::adapters::install_additive_conv(
                    &mut vendored,
                    &self.adapters,
                    &table,
                    build_device,
                )?;
                crate::adapters::guard_each_adapter_matched(
                    &self.adapters,
                    &[&linear.applied_by_spec, &conv.applied_by_spec],
                )?;
            }
            if let Some(quant) = self.quant {
                vendored.quantize_onto(quant, &self.device)?;
            }
            raw.clear();
            SdxlUnet::Vendored(Arc::new(vendored))
        } else {
            match self.detect_packed_unet()? {
                Some((packed_file, group_size)) => {
                    // sc-11103: a packed tier WITH a distill LoRA/LoKr applies it **additively** — the packed
                    // Linears take a forward-time residual (base kept packed) and any conv LoRA folds into the
                    // dense convs (`load_packed_unet_with_adapters`), so the q4/q8 footprint survives instead
                    // of dequant-folding the FF (the retired sc-9528 path).
                    let vendored = if self.adapters.is_empty() {
                        self.load_packed_unet(&packed_file, group_size)?
                    } else {
                        self.load_packed_unet_with_adapters(&packed_file, group_size)?
                    };
                    SdxlUnet::Packed(Arc::new(vendored))
                }
                None => {
                    let unet_file =
                        snapshot_file(&self.root, "unet/diffusion_pytorch_model.fp16.safetensors")?;
                    if use_flash_attn && self.adapters.is_empty() {
                        // **Unadapted + flash only.** The fused flash-attn kernel never materializes the full
                        // `[B·H, S, S]` scores tensor, so it does not hit the i32-overflow (sc-11154 / F-081)
                        // and needs no additive seam; keep the stock candle UNet so the fused kernel is used
                        // (byte-identical to pre-sc-5165). An ADAPTED render falls through to the vendored
                        // additive path below (sc-11682): the vendored flash path is a stub, and an additive
                        // residual over a pristine (evictable) mmap base is worth more for an adapted render
                        // than the fused kernel — so the old stock fold path is retired and adapted renders
                        // always take the i32-overflow-safe vendored math path.
                        let unet =
                            self.config
                                .build_unet(unet_file, &self.device, 4, true, self.dtype)?;
                        SdxlUnet::Stock(Arc::new(unet))
                    } else {
                        // Math-path attention (the default — no `flash-attn` feature — AND every adapted
                        // render): the stock candle UNet materializes a full `[B·H, S, S]` scores tensor that
                        // overflows i32 at ≥ ~1664² (2048²: `2·10·16384² ≈ 5.4e9 > i32::MAX`) and silently
                        // corrupts on CUDA (sc-11154 / F-081). Route through the vendored UNet, whose math
                        // attention is the i32-overflow-safe `sdpa_budgeted_flat` (bit-identical to the stock
                        // forward per `vendored_unet_matches_stock_forward`); an adapter rides additively over
                        // the mmap base (sc-11682), never folded.
                        let vendored = if self.adapters.is_empty() {
                            self.load_dense_vendored_unet(&unet_file)?
                        } else {
                            self.load_dense_vendored_unet_with_adapters(&unet_file)?
                        };
                        SdxlUnet::Vendored(Arc::new(vendored))
                    }
                }
            }
        };
        Ok(unet)
    }

    pub(crate) fn load_decoders(&self) -> Result<(Arc<SdxlVaeDecoder>, Option<Arc<PidEngine>>)> {
        let vae = if let Some(ldm) = &self.ldm {
            // A generic fused A1111 checkpoint may carry the original SDXL VAE, whose fp16 decode is
            // numerically unstable. Consume the checkpoint's own VAE truthfully, but keep it at f32;
            // `decode_image` casts the latent at this boundary.
            let vs = VarBuilder::from_tensors(ldm.vae.clone(), DType::F32, &self.device);
            SdxlVaeDecoder::new(vs, 3, &sdxl_vae_config())?
        } else {
            let source = self.vae_fix.as_ref().ok_or_else(|| {
                CandleError::Msg("sdxl: snapshot load is missing the fp16-fix VAE component".into())
            })?;
            SdxlVaeDecoder::from_file(
                &resolve_vae_file(source),
                &self.device,
                self.dtype,
                &sdxl_vae_config(),
            )?
        };
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; SDXL's own `sdxl` latent-space student. `None` ⇒ native VAE.
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok((Arc::new(vae), pid))
    }

    /// Detect a pre-quantized MLX SDXL tier at this pipeline's `root` — the free
    /// [`detect_packed_unet`] (shared with the InstantID/edit/IP-Adapter UNet loader, sc-10813) keyed
    /// on `self.root`.
    fn detect_packed_unet(&self) -> Result<Option<(PathBuf, usize)>> {
        detect_packed_unet(&self.root)
    }

    /// Build the vendored packed-detecting SDXL UNet from a packed MLX-tier `unet/` checkpoint
    /// (sc-9416). One mmap'd VarBuilder feeds the whole UNet; `linear_detect` in the vendored
    /// attention/FF/proj/time-embed sites builds a quantized `QLinear` straight from each packed
    /// `{weight, scales, biases}` triple, while the convolutions + norms load dense. No adapter is
    /// installed (the packed tier is inference-only), so the four attention projections' `LoraLinear`
    /// bases are their packed `QLinear` and the forward is exactly `x·Wᵀ + b` (dequant-on-forward).
    fn load_packed_unet(&self, unet_file: &Path, _group_size: usize) -> Result<VendoredUNet> {
        let vs =
            candle_gen::mmap_var_builder(&[unet_file.to_path_buf()], self.dtype, &self.device)?;
        // The vendored `new` threads the default MLX group size (64) — validated == the config group in
        // `detect_packed_unet` — through its packed-detecting leaves; `sdxl_unet_config` is the canonical
        // 3-block SDXL geometry (`use_linear_projection = true`, matching the packed `proj_in/out`).
        Ok(VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?)
    }

    /// Build the vendored packed UNet from a packed MLX-tier checkpoint with the [`AdapterSpec`]s applied
    /// **additively** (sc-11103, the sc-9528 dequant-fold replacement). A distill LoRA on a packed tier
    /// now rides the packed Linears as a **forward-time residual** (`y = base(x) + Σ scale·((x·A)·B)`,
    /// [`crate::adapters::install_additive`]) — the u32 codes are never dequantized, so the q4/q8
    /// footprint survives (SDXL-Lightning / RealVisXL-Lightning target the FF, the bulk of the UNet).
    /// The **conv** surface stays dense on a packed tier, so a conv LoRA still **folds** into it
    /// ([`crate::adapters::fold_conv_adapters`]) at no packed cost. The additive residual equals the
    /// dense fold to f32 tolerance — the accuracy bar the packed base's own quant already accepts (the
    /// per-ULP chaos-sensitivity argument is about a *re-quantized* fold, not a residual on the frozen
    /// packed base, so it does not apply here).
    fn load_packed_unet_with_adapters(
        &self,
        unet_file: &Path,
        group_size: usize,
    ) -> Result<VendoredUNet> {
        // The vendored UNet's top-level constructor threads only the default MLX group 64 through its
        // blocks; a non-64 tier would pack/read at mismatched grids. Refuse it loudly (mirrors
        // `detect_packed_unet`) rather than mis-apply.
        crate::adapters::assert_group_size_supported(group_size)?;
        let mut raw = candle_gen::candle_core::safetensors::load(unet_file, &Device::Cpu)?;
        // Shared kohya `flattened → dotted` table for both packed adapter passes (conv fold + additive).
        let table = crate::adapters::build_sdxl_kohya_table(&raw);
        // Fold any conv-LoRA into the dense conv weights BEFORE the build; the packed Linears are left
        // untouched so they load packed and take the additive residual below.
        let conv = crate::adapters::fold_conv_adapters(&mut raw, &self.adapters, &table)?;
        // `from_tensors` serves the u32 packed Linears via the vendored seam's `get_unchecked_dtype`
        // (exactly as the mmap path) and the (conv-folded) dense weights via the vb dtype. `false` = no
        // flash-attn on the packed path.
        let vs = VarBuilder::from_tensors(raw, self.dtype, &self.device);
        let mut unet = VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?;
        // Push the LoRA/LoKr residuals onto the packed Linear leaves — the base stays packed.
        let add =
            crate::adapters::install_additive(&mut unet, &self.adapters, &table, &self.device)?;
        // A non-empty spec set that neither folded a conv nor installed a residual is a misconfiguration.
        crate::adapters::guard_each_adapter_matched(
            &self.adapters,
            &[&conv.applied_by_spec, &add.applied_by_spec],
        )?;
        Ok(unet)
    }

    /// Build the **dense** SDXL UNet through the vendored stack (sc-11154 / F-081). The vendored math
    /// attention routes through the shared i32-overflow-safe [`candle_gen::sdpa_budgeted_flat`], so the
    /// dense (no-flash) path no longer materializes an over-`i32::MAX` `[B·H, S, S]` scores tensor at
    /// the advertised 2048² envelope (the stock candle UNet does — silent CUDA corruption). The
    /// vendored copy is bit-identical to the stock forward (`vendored_unet_matches_stock_forward`), and
    /// the diffusers fp16 checkpoint layout is shared: its `linear_detect` leaves see no `.scales`
    /// siblings and load every Linear dense (the same code path the packed loader takes for un-packed
    /// tensors). No adapter is installed, so each attention projection's base is its dense Linear.
    fn load_dense_vendored_unet(&self, unet_file: &Path) -> Result<VendoredUNet> {
        let vs =
            candle_gen::mmap_var_builder(&[unet_file.to_path_buf()], self.dtype, &self.device)?;
        Ok(VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?)
    }

    /// Dense vendored UNet with the [`AdapterSpec`]s applied **additively** (sc-11682) — the adapted
    /// counterpart of [`Self::load_dense_vendored_unet`], through the i32-overflow-safe vendored stack.
    /// The bf16 base stays a **pristine mmap** (never folded into a host `from_tensors` map), so
    /// epic-10765 offload/eviction can drop-and-restore it cheaply — a fold `W += δ` mutates the weight
    /// into a non-disk-re-derivable host copy that must be pinned. The adapter rides as forward-time
    /// residuals on both the Linear ([`crate::adapters::install_additive`]) and conv
    /// ([`crate::adapters::install_additive_conv`]) surfaces; additive equals the fold to f32 tolerance
    /// (and matches the trainer's own additive forward — the ~1-ULP `(W+δ)·x` gap is the *fold*'s, not
    /// the residual's). The kohya table is read from the file header (no data copy) so community
    /// `lora_unet_<flat>` keys resolve against the mmap base.
    fn load_dense_vendored_unet_with_adapters(&self, unet_file: &Path) -> Result<VendoredUNet> {
        let table = crate::adapters::build_sdxl_kohya_table_from_file(unet_file)?;
        let vs =
            candle_gen::mmap_var_builder(&[unet_file.to_path_buf()], self.dtype, &self.device)?;
        let mut unet = VendoredUNet::new(vs, 4, 4, false, sdxl_unet_config())?;
        let lin =
            crate::adapters::install_additive(&mut unet, &self.adapters, &table, &self.device)?;
        let conv = crate::adapters::install_additive_conv(
            &mut unet,
            &self.adapters,
            &table,
            &self.device,
        )?;
        crate::adapters::guard_each_adapter_matched(
            &self.adapters,
            &[&lin.applied_by_spec, &conv.applied_by_spec],
        )?;
        Ok(unet)
    }

    /// Render `req` against pre-resolved `text_embeddings` and (caller-cached, sc-5037) `unet`/`vae`,
    /// emitting per-step progress and honoring `req.cancel`. Returns one `gen_core::Image` per
    /// `req.count` (each with seed `base_seed + index`). The denoise+decode here is unchanged from
    /// sc-4987 — only the component *ownership* moved out to the generator so it can cache them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        text_embeddings: &Tensor,
        unet: &SdxlUnet,
        vae: &SdxlVaeDecoder,
        pid: Option<&PidEngine>,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let base_seed = run.map_or_else(
            || req.seed.unwrap_or_else(gen_core::default_seed),
            |r| r.plan().base_seed(),
        );
        let pid_decoder =
            candle_gen_pid::resolve_pid_decoder(pid, req, base_seed, crate::MODEL_ID)?;
        self.render_with(
            req,
            text_embeddings,
            unet,
            run,
            on_progress,
            |latents, progress| {
                let mut do_decode = || {
                    progress(Progress::Decoding);
                    self.decode(
                        vae,
                        pid_decoder
                            .as_ref()
                            .map(|decoder| decoder as &dyn LatentDecoder),
                        &latents,
                        &req.cancel,
                        crate::denoise::decode_tiling(req.memory),
                    )
                };
                if let Some(run) = run {
                    run.decode_phase(&mut do_decode)
                } else {
                    do_decode()
                }
            },
        )
    }

    pub(crate) fn denoise_images(
        &self,
        req: &GenerationRequest,
        text_embeddings: &Tensor,
        unet: &SdxlUnet,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Tensor>> {
        if req.use_pid && self.pid_spec.is_none() {
            return Err(CandleError::Msg(
                "sdxl: PiD requested but no PiD decoder loaded".into(),
            ));
        }
        self.render_with(
            req,
            text_embeddings,
            unet,
            run,
            on_progress,
            |latents, _| Ok(latents),
        )
    }

    pub(crate) fn decode_images(
        &self,
        req: &GenerationRequest,
        latents: &[Tensor],
        vae: &SdxlVaeDecoder,
        pid: Option<&PidEngine>,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let seed = run.map_or_else(
            || req.seed.unwrap_or_else(gen_core::default_seed),
            |r| r.plan().base_seed(),
        );
        let decoder = candle_gen_pid::resolve_pid_decoder(pid, req, seed, crate::MODEL_ID)?;
        latents
            .iter()
            .map(|latent| {
                let mut do_decode = || {
                    on_progress(Progress::Decoding);
                    self.decode(
                        vae,
                        decoder.as_ref().map(|d| d as &dyn LatentDecoder),
                        latent,
                        &req.cancel,
                        crate::denoise::decode_tiling(req.memory),
                    )
                };
                if let Some(run) = run {
                    run.decode_phase(&mut do_decode)
                } else {
                    do_decode()
                }
            })
            .collect()
    }

    fn load_vae_encoder(&self) -> Result<crate::unet::VaeMomentsEncoder> {
        if let Some(ldm) = &self.ldm {
            let vs = VarBuilder::from_tensors(ldm.vae.clone(), DType::F32, &self.device);
            Ok(crate::unet::VaeMomentsEncoder::new(vs, VAE_SCALE)?)
        } else {
            let source = self.vae_fix.as_ref().ok_or_else(|| {
                CandleError::Msg("sdxl: snapshot load is missing the fp16-fix VAE component".into())
            })?;
            crate::loaders::load_sdxl_vae_encoder(source, &self.device, self.dtype)
        }
    }

    fn encode_source_image(&self, source: &Image, width: u32, height: u32) -> Result<Tensor> {
        let (iw, ih) = (source.width as usize, source.height as usize);
        if source.pixels.len()
            != candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3)
                .unwrap_or(usize::MAX)
        {
            return Err(CandleError::Msg(format!(
                "sdxl: source_image pixel buffer {} != {iw}x{ih}x3",
                source.pixels.len()
            )));
        }
        let (rw, rh) = (width as usize, height as usize);
        let resized = if (iw, ih) == (rw, rh) {
            source.pixels.iter().map(|&p| p as f32).collect()
        } else {
            candle_gen::gen_core::imageops::resize_lanczos_u8(&source.pixels, ih, iw, rh, rw)?
        };
        let data: Vec<f32> = resized.into_iter().map(|v| v / 127.5 - 1.0).collect();
        let hwc = Tensor::from_vec(data, (rh, rw, 3), &self.device)?;
        let enc_dtype = if self.ldm.is_some() {
            DType::F32
        } else {
            self.dtype
        };
        let nchw = hwc
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .to_dtype(enc_dtype)?;
        let encoder = self.load_vae_encoder()?;
        Ok(encoder.encode_mean(&nchw)?.to_dtype(DType::F32)?)
    }

    fn render_with<T>(
        &self,
        req: &GenerationRequest,
        text_embeddings: &Tensor,
        unet: &SdxlUnet,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
        mut consume: impl FnMut(Tensor, &mut dyn FnMut(Progress)) -> Result<T>,
    ) -> Result<Vec<T>> {
        // Both cached resident components and staged components enter here.
        let representation = match unet {
            SdxlUnet::Stock(_) | SdxlUnet::Vendored(_) => "dense",
            SdxlUnet::Packed(_) => "packed",
        };
        candle_gen::execution::policy("sdxl/unet/representation", representation.into());
        // sc-6128: a `lightning` request runs the few-step Euler-trailing path. Every other render —
        // the omitted-sampler default AND every explicit curated solver name (incl. `ddim`) — routes
        // through the unified curated `Sampler` over `DiscreteModelSampling` (EPS), epic 7114 P4 / sc-7124.
        //
        // sc-10826: the omitted-sampler default previously ran candle-transformers' native
        // `DDIMScheduler` inference loop, which rendered a ghosted, translucent double-exposure
        // (guidance-invariant) while every curated solver — including the curated `ddim` — is clean.
        // So the default now maps to `DEFAULT_SAMPLER` (the curated `ddim`), and the native loop is
        // gone. `ddim` no longer diverts to the native path; it takes the curated `ddim` solver like
        // every other name. Determinism is preserved (curated `ddim` is eta=0 / non-ancestral).
        let lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
        // The curated solver name for a non-lightning render: the request's name, or `ddim` by default
        // (`resolve_sampler`). `None` ⇒ a `lightning` render (its own path). An unknown name falls back
        // to euler inside `run_curated_sampler` (N3 — never a hard fail).
        let curated: Option<&str> = resolve_sampler(req.sampler.as_deref());
        let steps = req.steps.map(|s| s as usize).unwrap_or(if lightning {
            LIGHTNING_DEFAULT_STEPS
        } else {
            DEFAULT_STEPS
        });
        let guidance = req.guidance.map(|g| g as f64).unwrap_or(DEFAULT_GUIDANCE);
        candle_gen::execution::policy("sdxl/pipeline/guidance", format!("{guidance:?}").into());
        // Lightning is a distilled, classifier-free sampler — it never runs CFG (the worker sends
        // guidance 1.0 for `realvisxl_lightning`, and CFG on a CFG-free checkpoint degrades output).
        // The DDIM path honors the request guidance exactly as before.
        let use_guide = !lightning && guidance > 1.0;
        candle_gen::execution::policy("sdxl/pipeline/use_guide", format!("{use_guide:?}").into());
        let base_seed = run.map_or_else(
            || req.seed.unwrap_or_else(gen_core::default_seed),
            |r| r.plan().base_seed(),
        );
        let total = steps as u32;
        let (init_w, init_h) = match run {
            Some(run) => run.plan().phase_output_size(0),
            None => (self.config.width as u32, self.config.height as u32),
        };
        let (lat_h, lat_w) = ((init_h / 8) as usize, (init_w / 8) as usize);
        let init_clean = match &req.source_image {
            Some(source) => Some(self.encode_source_image(source, init_w, init_h)?),
            None => None,
        };

        // Lightning precompute (seed-independent): the trailing-Euler policy + the cond-only text
        // embedding (row 1 of the `[uncond, cond]` dual-CLIP stack — Lightning runs one conditioned
        // forward per step, so the uncond row is unused). Built once, reused for every image.
        let lightning_ctx = if lightning {
            Some((
                lightning_policy(steps)?,
                text_embeddings.narrow(0, 1, 1)?.contiguous()?,
            ))
        } else {
            None
        };

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            if let Some(run) = run {
                run.reset_image_denoise_phases();
            }
            // sc-3673 — deterministic, launch-portable initial noise: draw N(0,1) from a
            // fixed-algorithm CPU RNG (`StdRng`, ChaCha-based) seeded by `seed`, build the latent on
            // CPU, then move it to the compute device. This replaces candle's CUDA `device.set_seed`
            // + on-device `randn`, whose seed→noise mapping was NOT portable across launch
            // environments and occasionally collapsed the sample to garbage (sc-3498). Paired with the
            // non-ancestral solver (DDIM, or the deterministic Lightning Euler), the whole generation
            // is a pure function of `(seed, request)` — same seed ⇒ same image, any launch.
            let n = 4 * lat_h * lat_w;
            let mut rng = StdRng::seed_from_u64(candle_gen::execution::seed("sdxl/pipeline", seed));
            let noise = candle_gen::seeded_normal_vec(&mut rng, n);
            let init = Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?;

            let latents = if let Some((policy, cond)) = &lightning_ctx {
                self.denoise_lightning(
                    &init,
                    policy,
                    cond,
                    unet,
                    &req.cancel,
                    on_progress,
                    total,
                    &req.preview,
                )?
            } else if let Some(name) = curated {
                // The default path (sc-10826): `curated` is `Some` for every non-lightning render, so
                // the omitted-sampler default (→ the curated `ddim`) and every explicit curated name
                // run this one unified sampler. The native candle-transformers DDIM loop is gone.
                self.denoise_curated(
                    req,
                    name,
                    &init,
                    init_clean.as_ref(),
                    text_embeddings,
                    unet,
                    steps,
                    use_guide,
                    guidance,
                    seed,
                    run,
                    on_progress,
                )?
            } else {
                // Unreachable by construction: `curated` is `Some` whenever `lightning_ctx` is `None`
                // (a non-lightning render always resolves a curated name via `DEFAULT_SAMPLER`). A
                // typed error rather than an `unwrap`/`unreachable!` so a future routing change can't
                // silently fall through to a broken (or removed) path.
                return Err(CandleError::Msg(
                    "sdxl: no denoise path selected (neither lightning nor curated) — a routing bug"
                        .into(),
                ));
            };
            if let Some(run) = run {
                run.finish_image_denoise_phases()?;
            }

            consume(latents, on_progress)
        })
    }

    /// The SDXL-**Lightning** few-step denoise (sc-6128) — diffusers Euler-trailing, ε-prediction,
    /// **CFG-off**. Distilled Lightning checkpoints are trained classifier-free, so this runs a single
    /// conditioned UNet forward per step (no uncond batch, no CFG combine).
    ///
    /// The latents live in diffusers' un-normalized **σ-space** (kept f32, unlike the DDIM path's f16
    /// latents): the prior is `unit_noise · σ_max`, the model input is `x/√(σ²+1)` cast to the UNet
    /// dtype, and each step is the deterministic Euler update `x ← x + ε·(σ_{i+1} − σ_i)` in f32. That
    /// update is the candle tensor application of [`gen_core::sampling`]'s neutral [`LightningPolicy`]
    /// coefficients (`a_x = 1`, `a_noise = 0`), mirroring `mlx-gen-sdxl`'s `apply_step` — so the two
    /// backends share one reference schedule.
    ///
    /// `init` is the seeded unit-normal noise (CPU `StdRng` → device, f32; the sc-3673 launch-portable
    /// contract); `cond` is the cond-only text embedding `[1, T, 2048]`. Returns latents in the compute
    /// dtype (f16) for the shared [`decode`](Self::decode).
    #[allow(clippy::too_many_arguments)]
    fn denoise_lightning(
        &self,
        init: &Tensor,
        policy: &LightningPolicy,
        cond: &Tensor,
        unet: &SdxlUnet,
        cancel: &gen_core::runtime::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        total: u32,
        preview: &gen_core::PreviewSink,
    ) -> Result<Tensor> {
        // σ-space prior: unit noise · the largest σ (init_noise_sigma for trailing spacing).
        let mut latents = init.affine(policy.init_noise_scale() as f64, 0.0)?;
        // Per-step latent preview (epic 16948, sc-16954). This lane owns its loop rather than driving
        // a shared sampler, so it numbers frames itself — on the STEP INDEX, since it walks
        // `LightningPolicy` coefficients rather than a σ array. One frame per iteration, so there is
        // no multi-eval repeat to dedup; the counter still bounds and dedups on principle.
        let preview_counter = candle_gen::preview::PreviewCounter::with_steps(policy.num_steps());
        candle_gen::execution::schedule(
            "sdxl-lightning-euler",
            &(0..=policy.num_steps())
                .filter_map(|i| policy.sigma(i))
                .collect::<Vec<_>>(),
            "epsilon",
        );
        candle_gen::execution::runtime(latents.device(), latents.dtype());
        for i in 0..policy.num_steps() {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let c = policy.coeffs(i);
            // Model-input scaling x/√(σ²+1), cast to the UNet compute dtype (f16). CFG-off ⇒ batch 1.
            let x_in = latents.affine(c.c_in as f64, 0.0)?.to_dtype(self.dtype)?;
            // Preview the RENORMALIZED latent — `x·c_in`, the very coefficient this step feeds the
            // UNet, which is also the domain the reused fit was measured in. Binding it to `c.c_in`
            // rather than recomputing `1/√(σ²+1)` is what keeps the preview and the denoise from
            // coming to disagree about the scaling. Failures are swallowed by `emit_preview_at`.
            candle_gen::preview::emit_preview_at(preview, &preview_counter, i, || {
                crate::preview::project_spatial_latents(&latents.affine(c.c_in as f64, 0.0)?)
            });
            let eps = unet
                .forward(&x_in, c.timestep as f64, cond)?
                .to_dtype(DType::F32)?;
            // Euler ε-pred step in f32: x + ε·(σ_{i+1} − σ_i) (a_x = 1, a_noise = 0, deterministic).
            latents = (latents + eps.affine(c.a_out as f64, 0.0)?)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        // The shared `decode` expects the compute dtype (f16), like the DDIM loop's latents.
        Ok(latents.to_dtype(self.dtype)?)
    }

    /// The **curated** ε/DDPM denoise (epic 7114 P4, sc-7124) — an ADDITIVE option alongside the native
    /// DDIM default and Lightning. Drives the unified [`gen_core::sampling::Sampler`] (`euler` /
    /// `euler_ancestral` / `heun` / `dpmpp_2m` / `dpmpp_sde` / `uni_pc` / `lcm`) over a
    /// [`DiscreteModelSampling`] (SDXL ε-prediction, `scaled_linear` β over 1000 train steps). The
    /// `scheduler` axis (`normal` default / `karras` / `sgm_uniform` / …) picks the σ schedule via
    /// [`candle_gen::resolve_schedule`]. Latents live in k-diffusion VE σ-space (prior = unit noise ·
    /// σ_max), kept f32 (like the Lightning path); the [`DiscreteModelSampling`] recombines ε → x0 and
    /// supplies the `1/√(σ²+1)` input scaling, so the closure just runs the UNet + CFG and returns raw ε.
    /// The native DDIM/Lightning defaults are untouched, so this never affects the N1 default-parity gate.
    #[allow(clippy::too_many_arguments)]
    fn denoise_curated(
        &self,
        req: &GenerationRequest,
        sampler: &str,
        init: &Tensor,
        init_clean: Option<&Tensor>,
        text_embeddings: &Tensor,
        unet: &SdxlUnet,
        steps: usize,
        use_guide: bool,
        guidance: f64,
        seed: u64,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let sched = sdxl_alpha_schedule()?;
        let ms = DiscreteModelSampling::sdxl(&sched);
        // Native curated schedule = ComfyUI's SDXL default (`normal`); the scheduler axis overrides it.
        let native = schedule_sigmas(Scheduler::Normal, &ms, steps);
        let full_sigmas =
            candle_gen::resolve_schedule(req.scheduler.as_deref(), &ms, steps, &native);
        let (sigmas, latents): (&[f32], Tensor) = if let Some(clean) = init_clean {
            let raw_strength = req.strength.unwrap_or(crate::DEFAULT_EDIT_STRENGTH);
            let start_step = gen_core::resolve_img2img_init_time_step(
                full_sigmas.len().saturating_sub(1),
                Some(raw_strength),
            )
            .map_err(CandleError::from)?;
            let sliced = &full_sigmas[start_step..];
            if sliced.len() <= 1 {
                return Ok(clean.to_dtype(self.dtype)?);
            }
            let blended = (clean + (init * sliced[0] as f64)?)?;
            (sliced, blended)
        } else {
            // VE prior: unit noise · σ_max (sigmas[0]); kept f32 through the sampler (cast to f16 per eval).
            (&full_sigmas, (init * full_sigmas[0] as f64)?)
        };
        // Per-step latent preview (epic 16948, sc-16954). Opting in is the sc-16949 projector hook, so
        // the loop is not restructured and the driver owns frame numbering + the multi-eval dedup.
        // `ve_hook` because the running latent here is raw k-diffusion VE σ-space — see
        // `crate::preview` for why that needs the `1/√(σ²+1)` renormalization the fit was measured in.
        // Built per image: the driver starts a fresh counter per call.
        let preview = match run {
            Some(run) => crate::preview::ve_hook_for_plan(&req.preview, run.plan()),
            None => crate::preview::ve_hook(&req.preview),
        };
        // sc-14195: the conditioning must be batched to match the UNet input. `text_embeddings` is
        // ALWAYS the `[uncond, cond]` stack (uncond-first — `text_embeddings()` encodes both rows
        // unconditionally, since the uncond row is what CFG needs and it is cheap to carry), but
        // CFG-off (`guidance ≤ 1.0`, an advertised request value) runs a **single conditioned**
        // forward over a batch-1 latent. Feeding it the batch-2 stack put a batch-1 query against
        // batch-2 cross-attention K/V and killed the UNet inside the attention matmul
        // ("shape mismatch in matmul, lhs: [10, 4096, 64], rhs: [20, 64, 77]"). So narrow to the
        // cond row (index 1) exactly as the Lightning path already does — CFG-on is untouched.
        let ehs = if use_guide {
            text_embeddings.clone()
        } else {
            text_embeddings.narrow(0, 1, 1)?.contiguous()?
        };
        if req.guidance_method.as_deref() == Some("cfg_pp")
            && use_guide
            && gen_core::sampling::Solver::from_name(sampler)
                .is_some_and(gen_core::sampling::base_supports_cfgpp)
        {
            return self.denoise_cfgpp(
                req,
                sampler,
                &ms,
                sigmas,
                latents,
                &ehs,
                unet,
                guidance,
                seed,
                run,
                &preview,
                on_progress,
            );
        }
        let mut predict = |x_in: &Tensor, t: f32| -> Result<Tensor> {
            // `x_in` is already `1/√(σ²+1)`-scaled by `denoise()`; `t` is the nearest training-step
            // index the UNet embeds. CFG batches/combines exactly like the native DDIM path.
            let eps = if use_guide && crate::sequential_cfg_enabled() {
                let model_in = x_in.to_dtype(self.dtype)?;
                let uncond =
                    unet.forward(&model_in, t as f64, &ehs.narrow(0, 0, 1)?.contiguous()?)?;
                candle_gen::residency::check_cancel(&req.cancel)?;
                let cond =
                    unet.forward(&model_in, t as f64, &ehs.narrow(0, 1, 1)?.contiguous()?)?;
                (&uncond + ((cond - &uncond)? * guidance)?)?
            } else {
                let model_in = if use_guide {
                    Tensor::cat(&[x_in, x_in], 0)?
                } else {
                    x_in.clone()
                };
                let noise_pred = unet.forward(&model_in.to_dtype(self.dtype)?, t as f64, &ehs)?;
                if use_guide {
                    let chunks = noise_pred.chunk(2, 0)?;
                    (&chunks[0] + ((&chunks[1] - &chunks[0])? * guidance)?)?
                } else {
                    noise_pred
                }
            };
            // Raw ε in f32 so the DiscreteModelSampling x0 recombine + solver math stay f32.
            Ok(eps.to_dtype(DType::F32)?)
        };
        let out = match run {
            Some(run) => {
                let (run_sigmas, spans) = if init_clean.is_some() {
                    let spans = match run.plan().low_res_start() {
                        Some(low_res) if low_res.k > 0 => {
                            let p0 = low_res.k.min(sigmas.len().saturating_sub(1));
                            if p0 < sigmas.len().saturating_sub(1) {
                                vec![0..p0, p0..sigmas.len().saturating_sub(1)]
                            } else {
                                std::iter::once(0..sigmas.len().saturating_sub(1)).collect()
                            }
                        }
                        _ => std::iter::once(0..sigmas.len().saturating_sub(1)).collect(),
                    };
                    (sigmas, spans)
                } else {
                    (
                        run.plan().denoise_sigmas(),
                        run.plan().denoise_spans().to_vec(),
                    )
                };
                candle_gen::sampler::run_curated_sampler_spans(
                    Some(sampler),
                    &ms,
                    run_sigmas,
                    &spans,
                    Some(run),
                    false,
                    latents,
                    seed,
                    &req.cancel,
                    on_progress,
                    Some(&preview),
                    &mut predict,
                )?
            }
            None => candle_gen::run_curated_sampler(
                Some(sampler),
                &ms,
                sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                Some(&preview),
                &mut predict,
            )?,
        };
        // The shared `decode` expects the compute dtype (f16), like the DDIM/Lightning latents.
        Ok(out.to_dtype(self.dtype)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_cfgpp(
        &self,
        req: &GenerationRequest,
        sampler: &str,
        ms: &DiscreteModelSampling,
        sigmas: &[f32],
        mut latents: Tensor,
        ehs: &Tensor,
        unet: &SdxlUnet,
        guidance: f64,
        seed: u64,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        preview: &candle_gen::preview::PreviewHook<'_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        use candle_gen::gen_core::sampling::{base_supports_cfgpp, cfgpp_denoise, Solver};

        let base = Solver::from_name(sampler)
            .filter(|solver| base_supports_cfgpp(*solver))
            .unwrap_or(Solver::Euler);
        let ops = candle_gen::CandleLatentOps;
        let total = sigmas.len().saturating_sub(1).max(1) as u32;
        let preview_counter = candle_gen::preview::PreviewCounter::new(sigmas);
        candle_gen::execution::runtime(latents.device(), latents.dtype());
        candle_gen::execution::schedule(sampler, sigmas, "Discrete/Eps/cfg_pp");

        let spans = match run {
            Some(run) if run.plan().denoise_sigmas().len() == sigmas.len() => {
                run.plan().denoise_spans().to_vec()
            }
            Some(run) => match run.plan().low_res_start() {
                Some(low_res) if low_res.k > 0 => {
                    let p0 = low_res.k.min(sigmas.len().saturating_sub(1));
                    if p0 < sigmas.len().saturating_sub(1) {
                        vec![0..p0, p0..sigmas.len().saturating_sub(1)]
                    } else {
                        std::iter::once(0..sigmas.len().saturating_sub(1)).collect()
                    }
                }
                _ => std::iter::once(0..sigmas.len().saturating_sub(1)).collect(),
            },
            None => std::iter::once(0..sigmas.len().saturating_sub(1)).collect(),
        };

        let predict_pair = |x_in: &Tensor, t: f32| -> gen_core::Result<(Tensor, Tensor)> {
            let (eps_u, eps_c) = if crate::sequential_cfg_enabled() {
                let model_in = x_in.to_dtype(self.dtype).map_err(CandleError::from)?;
                let u = unet.forward(
                    &model_in,
                    t as f64,
                    &ehs.narrow(0, 0, 1)
                        .map_err(CandleError::from)?
                        .contiguous()
                        .map_err(CandleError::from)?,
                )?;
                candle_gen::residency::check_cancel(&req.cancel)?;
                let c = unet.forward(
                    &model_in,
                    t as f64,
                    &ehs.narrow(0, 1, 1)
                        .map_err(CandleError::from)?
                        .contiguous()
                        .map_err(CandleError::from)?,
                )?;
                (
                    u.to_dtype(DType::F32).map_err(CandleError::from)?,
                    c.to_dtype(DType::F32).map_err(CandleError::from)?,
                )
            } else {
                let model_in = Tensor::cat(&[x_in, x_in], 0)
                    .and_then(|cat| cat.to_dtype(self.dtype))
                    .map_err(CandleError::from)?;
                let noise_pred = unet
                    .forward(&model_in, t as f64, ehs)?
                    .to_dtype(DType::F32)
                    .map_err(CandleError::from)?;
                let chunks = noise_pred.chunk(2, 0).map_err(CandleError::from)?;
                (chunks[0].clone(), chunks[1].clone())
            };
            let guided = (&eps_u
                + ((&eps_c - &eps_u).map_err(CandleError::from)? * guidance)
                    .map_err(CandleError::from)?)
            .map_err(CandleError::from)?;
            Ok((guided, eps_u))
        };

        let mut denoise_fn = |x: &Tensor, sigma: f32| -> gen_core::Result<(Tensor, Tensor)> {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            candle_gen::preview::PreviewHook::emit(preview, &preview_counter, sigmas, sigma, x);
            let current = (sigmas.iter().filter(|&&s| s > sigma).count() as u32 + 1).min(total);
            on_progress(Progress::Step { current, total });
            cfgpp_denoise(&ops, ms, x, sigma, predict_pair)
        };
        latents = run_cfgpp_spans(
            &ops,
            base,
            sigmas,
            &spans,
            latents,
            |span_idx| {
                if let Some(run) = run {
                    run.begin_denoise_phase(span_idx)
                        .map_err(CandleError::from)?;
                }
                Ok(())
            },
            |span_idx, _latents| {
                if let Some(run) = run {
                    run.end_denoise_phase(span_idx).map_err(CandleError::from)?;
                }
                Ok(())
            },
            &mut denoise_fn,
            |span_idx, boundary_step, latents, state| {
                let _ = span_idx;
                if let Some(run) = run {
                    if let Some(low_res_plan) = run.plan().low_res_start().cloned() {
                        if boundary_step == low_res_plan.k {
                            let sigma_k = sigmas[boundary_step];
                            let base_seed = run.plan().base_seed();
                            let output_index = seed.wrapping_sub(base_seed) as u32;
                            let pass_index = run.plan().pass_index();
                            let x0_low = gen_core::sampling::denoise(
                                &ops,
                                ms,
                                latents,
                                sigma_k,
                                |xin, t| Ok(predict_pair(xin, t)?.0),
                            )
                            .map_err(CandleError::from)?;
                            let low_f32 = x0_low
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let dims = x0_low.dims();
                            let (b_sz, channels, low_h) = (dims[0], dims[1], dims[2]);
                            let vae_spatial_scale =
                                ((low_res_plan.low_height as usize) / low_h.max(1)).max(1);
                            let noised = gen_core::transition_clean_estimate_f32(
                                gen_core::BoundaryState::CleanEstimate {
                                    latent: &low_f32,
                                    source: gen_core::EstimateSource::EndpointEvaluation,
                                },
                                b_sz,
                                channels,
                                vae_spatial_scale as u32,
                                &low_res_plan,
                                sigma_k,
                                ms,
                                base_seed,
                                output_index,
                                pass_index,
                                0,
                            )
                            .map_err(CandleError::from)?;
                            let full_h = (low_res_plan.full_height as usize) / vae_spatial_scale;
                            let full_w = (low_res_plan.full_width as usize) / vae_spatial_scale;
                            let transitioned = Tensor::from_vec(
                                noised,
                                (b_sz, channels, full_h, full_w),
                                x0_low.device(),
                            )?;
                            state.note_geometry_change(
                                (low_res_plan.low_width, low_res_plan.low_height),
                                (low_res_plan.full_width, low_res_plan.full_height),
                            );
                            state.note_renoise_transition();
                            state
                                .apply_phase_boundary(
                                    gen_core::sampling::PhaseBoundary::Reset,
                                    true,
                                )
                                .map_err(CandleError::from)?;
                            return Ok(transitioned);
                        }
                    }
                }
                Ok(latents.clone())
            },
        )?;
        Ok(latents.to_dtype(self.dtype)?)
    }

    /// Decode latents to an RGB8 [`Image`], either through the native VAE or — when a PiD decoder
    /// resolved (epic 7840 / sc-7853) — the super-resolving PiD student (emits a larger `[1,3,4H,4W]`
    /// tensor). Both produce `[-1, 1]` pixels; [`to_image`](Self::to_image) reads the size from the
    /// tensor (never `latent*8`).
    ///
    /// **Latent convention (sc-7848 parity — NOT zero-transform on candle):** the PiD `sdxl` student
    /// trained on the **0.13025-normalized** latent — the scaled sampler output `latents`. In candle the
    /// VAE de-scale happens here in the pipeline (`latents / VAE_SCALE`) rather than inside `vae.decode`
    /// (unlike the qwen/flux families, whose VAE de-normalizes internally), so PiD gets `latents`
    /// (normalized) while the VAE gets the de-scaled raw latent. This matches MLX, where `vae.decode`
    /// de-scales internally and both paths receive that same normalized tensor.
    fn decode(
        &self,
        vae: &SdxlVaeDecoder,
        pid: Option<&dyn LatentDecoder>,
        latents: &Tensor,
        cancel: &CancelFlag,
        tiling: Option<TilingConfig>,
    ) -> Result<Image> {
        let native = if self.ldm.is_some() {
            SdxlLatentDecoder::with_decode_dtype(vae, DType::F32)
        } else {
            SdxlLatentDecoder::new(vae)
        };
        self.decode_with_tiling_gate(&native, pid, latents, cancel, tiling)
    }

    /// Production SDXL decoder dispatch after the process-global tiling gate is sampled. Kept
    /// separate from [`Self::decode`] so native/PiD selection, tiled-vs-monolithic routing, and the
    /// final RGB8 postprocess can be exercised together without constructing a real PiD engine.
    fn decode_with_tiling_gate(
        &self,
        native: &dyn LatentDecoder,
        pid: Option<&dyn LatentDecoder>,
        latents: &Tensor,
        cancel: &CancelFlag,
        tiling: Option<TilingConfig>,
    ) -> Result<Image> {
        let decoder = pid.unwrap_or(native);
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        candle_gen::ensure_decoder_compatible(
            Some(&candle_gen::gen_core::SDXL_LATENT_SPACE),
            decoder,
        )?;
        let img = match &tiling {
            Some(tiling) => decoder.decode_tiled(latents, tiling, Some(cancel))?,
            None => decoder.decode(latents)?,
        };
        self.to_image(&img)
    }

    /// Convert a decoded pixel tensor `[1, 3, H, W]` in `[-1, 1]` → RGB8 [`Image`] (`x/2 + 0.5`, clamp,
    /// ×255). Shared by the native VAE decode and the PiD super-resolving decode; the output size is
    /// read from the tensor, never assumed (PiD may be 4× the VAE-native size).
    fn to_image(&self, img: &Tensor) -> Result<Image> {
        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let scaled = (img * 255.)?;
        let img = candle_gen::round_rgb8(&scaled)?
            .i(0)?
            .to_device(&Device::Cpu)?;
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
        }
        let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        Ok(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_cfgpp_spans<L: gen_core::sampling::LatentOps + 'static>(
    ops: &L,
    base: gen_core::sampling::Solver,
    sigmas: &[f32],
    spans: &[std::ops::Range<usize>],
    mut latents: L::Latent,
    mut begin_phase: impl FnMut(usize) -> Result<()>,
    mut end_phase: impl FnMut(usize, &L::Latent) -> Result<()>,
    mut denoise_fn: impl FnMut(&L::Latent, f32) -> gen_core::Result<(L::Latent, L::Latent)>,
    mut transition_fn: impl FnMut(
        usize,
        usize,
        &L::Latent,
        &mut gen_core::sampling::SolverRunState<L>,
    ) -> Result<L::Latent>,
) -> Result<L::Latent> {
    use candle_gen::gen_core::sampling::{cfgpp_sampler_for, CfgPpSampler, SolverRunState};

    let cfgpp_sampler: Box<dyn CfgPpSampler<L>> =
        cfgpp_sampler_for::<L>(base).expect("base_supports_cfgpp holds");
    let mut state = SolverRunState::new_cfg_pp(cfgpp_sampler.base_solver(), sigmas);

    for (span_idx, span) in spans.iter().enumerate() {
        begin_phase(span_idx)?;
        latents = cfgpp_sampler
            .sample_span(
                ops,
                &mut denoise_fn,
                latents,
                sigmas,
                span.clone(),
                &mut state,
            )
            .map_err(CandleError::from)?;
        end_phase(span_idx, &latents)?;
        if span_idx + 1 < spans.len() {
            latents = transition_fn(span_idx, span.end, &latents, &mut state)?;
        }
    }
    if state.next_step() != sigmas.len().saturating_sub(1) {
        return Err(CandleError::from(gen_core::Error::Unsupported(
            "cfg_pp sampler spans do not cover the executed schedule".into(),
        )));
    }
    Ok(latents)
}

pub(crate) fn sdxl_vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// Detect a **packed** MLX-tier CLIP encoder `which` in the snapshot at `root` (sc-9527, sc-9089j
/// follow-up to sc-9416): `Some((packed_weight_file, group_size))` when
/// `text_encoder{,_2}/config.json` carries a `quantization` block ([`PackedConfig`]) AND the packed
/// weight file (`model.safetensors`, not the dense `.fp16` name) exists, else `None` — a dense
/// diffusers snapshot loads through the stock builder unchanged. A missing config (e.g. a bare
/// single-file checkpoint) is treated as dense; the downstream loader gives the precise "missing X"
/// error. `group_size` is threaded from the config (defaulting to 64 via [`PackedConfig`], never
/// silent-dense — the sc-9410 rule) into the vendored CLIP's Linear seam.
pub(crate) fn detect_packed_clip(root: &Path, which: &Clip) -> Result<Option<(PathBuf, usize)>> {
    let dir = which.subdir();
    let cfg_path = root.join(dir).join("config.json");
    if !cfg_path.is_file() {
        return Ok(None);
    }
    let bytes = candle_gen::gen_core::__model_input_reader!(
std::fs::read(&cfg_path),
"crates/media/candle-gen/candle-gen-sdxl/src/pipeline.rs::detect_packed_clip::std::fs::read@75223",
ArtifactInput,
&cfg_path,
|__model_input_path| std::fs::read(__model_input_path)
)
    .map_err(|e| CandleError::Msg(format!("sdxl: read {dir}/config.json: {e}")))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| CandleError::Msg(format!("sdxl: parse {dir}/config.json: {e}")))?;
    let Some(packed) = PackedConfig::from_config(&cfg) else {
        return Ok(None);
    };
    let file = snapshot_file(root, &format!("{dir}/model.safetensors"))?;
    Ok(Some((file, packed.group_size as usize)))
}

/// Detect a pre-quantized MLX SDXL tier at `root` (sc-9416): `Some((unet_file, group_size))` when
/// `unet/config.json` carries a `quantization` block ([`PackedConfig`]) and the packed weight file
/// (`diffusion_pytorch_model.safetensors`, not the dense `.fp16` name) exists, else `None` (a dense
/// diffusers snapshot — the stock/dense build). Errors on a packed tier whose group size the vendored
/// UNet's Linear seam does not thread (only 64 today) rather than silently repacking at the wrong grid.
///
/// Shared by the base txt2img load ([`Pipeline::load_components`], via the [`Pipeline::detect_packed_unet`]
/// method wrapper) AND the InstantID/edit/IP-Adapter vendored-UNet loader ([`crate::loaders::load_instantid_unet`],
/// sc-10813) — both take the packed vs dense fork from the SAME `unet/config.json` probe so a q4/q8 tier
/// serves the edit / inpaint / IP-Adapter lanes, not just plain txt2img.
pub(crate) fn detect_packed_unet(root: &Path) -> Result<Option<(PathBuf, usize)>> {
    let cfg_path = root.join("unet/config.json");
    if !cfg_path.is_file() {
        return Ok(None);
    }
    let bytes = candle_gen::gen_core::__model_input_reader!(
std::fs::read(&cfg_path),
"crates/media/candle-gen/candle-gen-sdxl/src/pipeline.rs::detect_packed_unet::std::fs::read@76839",
ArtifactInput,
&cfg_path,
|__model_input_path| std::fs::read(__model_input_path)
)
    .map_err(|e| CandleError::Msg(format!("sdxl: read unet/config.json: {e}")))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| CandleError::Msg(format!("sdxl: parse unet/config.json: {e}")))?;
    let Some(packed) = PackedConfig::from_config(&cfg) else {
        return Ok(None);
    };
    let group_size = packed.group_size as usize;
    // The vendored UNet's Linear seam threads the default MLX group size (64) through its leaf
    // constructors; a non-64 tier would repack on the wrong grid, so refuse it loudly. The SDXL MLX
    // tiers all pack at 64, so this never fires on a real tier.
    if group_size != MLX_GROUP_SIZE {
        // sc-9528 kept this loud reject: the vendored UNet's top-level `new` → blocks → leaves chain
        // threads only the default group 64 (the leaf `*_gs` constructors exist, but wiring a non-64
        // group through the many nested block constructors is the same infeasibility lens/sd3 hit in
        // sc-9474). A non-64 SDXL MLX tier does not exist today; refuse it rather than repack on the
        // wrong grid. The packed adapter path ([`crate::adapters::assert_group_size_supported`]) asserts
        // gs==64 for the same reason.
        return Err(CandleError::Msg(format!(
            "sdxl: packed tier group_size {group_size} unsupported (only {MLX_GROUP_SIZE}); \
             a non-64 SDXL tier needs the group threaded through the UNet blocks (sc-9528)"
        )));
    }
    let file = snapshot_file(root, "unet/diffusion_pytorch_model.safetensors")?;
    Ok(Some((file, group_size)))
}

/// Resolve a component file inside the SDXL snapshot dir, erroring clearly if absent (e.g. a
/// single-file RealVisXL checkpoint that lacks the diffusers multi-component tree — sc-3677).
pub(crate) fn snapshot_file(root: &Path, sub: &str) -> Result<PathBuf> {
    let p = root.join(sub);
    if !p.is_file() {
        return Err(CandleError::Msg(format!(
            "sdxl snapshot is missing {sub} (expected a diffusers multi-component snapshot at {})",
            root.display()
        )));
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct SdxlDecodeSpy {
        output: Tensor,
        tiled_output: Tensor,
        decode_calls: Cell<usize>,
        tiled_calls: Cell<usize>,
    }

    impl SdxlDecodeSpy {
        fn new(output: Tensor, tiled_output: Tensor) -> Self {
            Self {
                output,
                tiled_output,
                decode_calls: Cell::new(0),
                tiled_calls: Cell::new(0),
            }
        }

        fn same(output: Tensor) -> Self {
            Self::new(output.clone(), output)
        }
    }

    #[test]
    fn sdxl_schedule_constants_match_the_canonical_candle_grid() {
        let shared = sdxl_alpha_schedule().unwrap();
        let direct = AlphaSchedule::scaled_linear(SDXL_TRAIN_STEPS, SDXL_BETA_START, SDXL_BETA_END);
        assert_eq!(shared.alphas_cumprod, direct.alphas_cumprod);
    }

    impl LatentDecoder for SdxlDecodeSpy {
        fn input_latent_space(&self) -> Option<&candle_gen::gen_core::LatentSpace> {
            Some(&candle_gen::gen_core::SDXL_LATENT_SPACE)
        }

        fn decode(&self, _latents: &Tensor) -> Result<Tensor> {
            self.decode_calls.set(self.decode_calls.get() + 1);
            Ok(self.output.clone())
        }

        fn decode_tiled(
            &self,
            _latents: &Tensor,
            _tiling: &TilingConfig,
            cancel: Option<&CancelFlag>,
        ) -> Result<Tensor> {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(CandleError::Canceled);
            }
            self.tiled_calls.set(self.tiled_calls.get() + 1);
            Ok(self.tiled_output.clone())
        }
    }

    fn decode_test_pipeline(device: &Device) -> Pipeline {
        Pipeline {
            config: StableDiffusionConfig::sdxl(None, Some(128), Some(128)),
            root: PathBuf::from("/nonexistent/sdxl-decode-test"),
            device: device.clone(),
            dtype: DType::F32,
            adapters: vec![],
            pid_spec: None,
            vae_fix: Some(WeightsSource::File("/nonexistent/vae.safetensors".into())),
            ldm: None,
            quant: None,
        }
    }

    fn tiny_sdxl_vae(device: &Device) -> SdxlVaeDecoder {
        use candle_gen::candle_nn::{VarBuilder, VarMap};
        use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;

        let vars = VarMap::new();
        SdxlVaeDecoder::new(
            VarBuilder::from_varmap(&vars, DType::F32, device),
            3,
            &AutoEncoderKLConfig::default(),
        )
        .unwrap()
    }

    fn legacy_sdxl_image(vae: &SdxlVaeDecoder, latents: &Tensor) -> Image {
        use candle_gen::candle_core::IndexOp;

        let decoded = vae.decode(&(latents / VAE_SCALE).unwrap()).unwrap();
        let scaled = (((decoded / 2.0).unwrap() + 0.5).unwrap())
            .clamp(0f32, 1f32)
            .unwrap();
        let scaled = (scaled * 255.0).unwrap();
        let image = candle_gen::round_rgb8(&scaled)
            .unwrap()
            .i(0)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let (channels, height, width) = image.dims3().unwrap();
        assert_eq!(channels, 3);
        Image {
            width: width as u32,
            height: height as u32,
            pixels: image
                .permute((1, 2, 0))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
        }
    }

    #[test]
    fn imported_ldm_decoder_casts_at_the_vae_boundary() {
        let device = Device::Cpu;
        let vae = tiny_sdxl_vae(&device);
        let latents = Tensor::zeros((1, 4, 2, 2), DType::F16, &device).unwrap();

        let native = SdxlLatentDecoder::new(&vae);
        assert_eq!(native.unscale(&latents).unwrap().dtype(), DType::F16);

        let imported = SdxlLatentDecoder::with_decode_dtype(&vae, DType::F32);
        assert_eq!(imported.unscale(&latents).unwrap().dtype(), DType::F32);
    }

    /// SC-18309 N1: a real tiny SDXL VAE decoder proves that moving `1 / VAE_SCALE` into the
    /// native trait adapter leaves the no-override tensor exact, then traverses the registered
    /// [`Pipeline::decode`] route for byte-exact RGB parity and PiD selection. Explicit gate arms
    /// exercise the same production helper's monolithic/tiled dispatch and postprocess.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn decoder_seam_preserves_sdxl_default_and_pid_bytes() {
        let device = Device::Cpu;
        let vae = tiny_sdxl_vae(&device);
        let values = (0..(4 * 3 * 5))
            .map(|index| index as f32 * 0.01 - 0.3)
            .collect::<Vec<_>>();
        let latents = Tensor::from_vec(values, (1, 4, 3, 5), &device).unwrap();

        let legacy_tensor = vae.decode(&(latents.clone() / VAE_SCALE).unwrap()).unwrap();
        let via_seam = SdxlLatentDecoder::new(&vae).decode(&latents).unwrap();
        assert_eq!(
            via_seam.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            legacy_tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            "normalization ownership must not change one output bit"
        );
        let expected = legacy_sdxl_image(&vae, &latents);
        let pipeline = decode_test_pipeline(&device);
        let cancel = CancelFlag::default();
        let got = pipeline
            .decode(&vae, None, &latents, &cancel, None)
            .unwrap();
        assert_eq!(got, expected);

        let pid = SdxlDecodeSpy::same(Tensor::ones((1, 3, 4, 7), DType::F32, &device).unwrap());
        let got = pipeline
            .decode(&vae, Some(&pid), &latents, &cancel, None)
            .unwrap();
        assert_eq!((got.width, got.height), (7, 4));
        assert!(got.pixels.iter().all(|pixel| *pixel == 255));
        assert_eq!(pid.decode_calls.get() + pid.tiled_calls.get(), 1);

        let native = SdxlDecodeSpy::new(
            Tensor::full(-1.0f32, (1, 3, 2, 3), &device).unwrap(),
            Tensor::ones((1, 3, 2, 3), DType::F32, &device).unwrap(),
        );
        let monolithic = pipeline
            .decode_with_tiling_gate(&native, None, &latents, &cancel, None)
            .unwrap();
        assert!(monolithic.pixels.iter().all(|pixel| *pixel == 0));
        assert_eq!(native.decode_calls.get(), 1);
        assert_eq!(native.tiled_calls.get(), 0);

        let tiled = pipeline
            .decode_with_tiling_gate(
                &native,
                None,
                &latents,
                &cancel,
                Some(crate::memory_strategy::decode_tiling_config(None)),
            )
            .unwrap();
        assert!(tiled.pixels.iter().all(|pixel| *pixel == 255));
        assert_eq!(native.decode_calls.get(), 1);
        assert_eq!(native.tiled_calls.get(), 1);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED pipeline::tests::decoder_seam_preserves_sdxl_default_and_pid_bytes");
    }

    /// sc-9416: `detect_packed_unet` returns `Some((file, group_size))` for a snapshot whose
    /// `unet/config.json` carries a `quantization` block AND the packed weight file exists, and `None`
    /// for a dense snapshot (no block) — the packed/dense fork the base txt2img load takes. GPU-free.
    #[test]
    fn detect_packed_unet_reads_quantization_block() {
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();
        // A packed config + a (stub) packed weight file at the non-.fp16 name.
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 64}, "cross_attention_dim": 2048}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            b"stub",
        )
        .unwrap();

        let pipe = Pipeline {
            config: StableDiffusionConfig::sdxl(None, Some(1024), Some(1024)),
            root: tmp.clone(),
            device: Device::Cpu,
            dtype: DType::F32,
            adapters: vec![],
            pid_spec: None,
            vae_fix: Some(WeightsSource::File("/nonexistent/vae.safetensors".into())),
            ldm: None,
            quant: None,
        };
        let got = pipe.detect_packed_unet().unwrap();
        assert!(got.is_some(), "a quantization block ⇒ packed tier");
        assert_eq!(got.unwrap().1, 64, "group_size threaded from config");

        // A dense config (no quantization block) ⇒ None (the stock build).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"cross_attention_dim": 2048, "sample_size": 128}"#,
        )
        .unwrap();
        assert!(
            pipe.detect_packed_unet().unwrap().is_none(),
            "no quantization block ⇒ dense (stock) build"
        );

        // A bits-only block still packs (group defaults to 64, not silent-dense — the sc-9410 rule).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 8}}"#,
        )
        .unwrap();
        assert_eq!(
            pipe.detect_packed_unet().unwrap().map(|(_, g)| g),
            Some(64),
            "bits-only ⇒ packed at the default group 64"
        );
    }

    /// A packed tier whose group size is not the seam's threaded 64 is rejected loudly (sc-9416 /
    /// sc-9528) rather than silently repacking on the wrong grid.
    #[test]
    fn detect_packed_unet_rejects_non_64_group() {
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            b"stub",
        )
        .unwrap();
        let pipe = Pipeline {
            config: StableDiffusionConfig::sdxl(None, Some(1024), Some(1024)),
            root: tmp.clone(),
            device: Device::Cpu,
            dtype: DType::F32,
            adapters: vec![],
            pid_spec: None,
            vae_fix: Some(WeightsSource::File("/nonexistent/vae.safetensors".into())),
            ldm: None,
            quant: None,
        };
        assert!(
            pipe.detect_packed_unet().is_err(),
            "a non-64 group_size must be rejected, not silently mis-repacked"
        );
    }

    /// sc-10826: an **omitted** sampler must resolve to the curated `ddim` solver — the native
    /// candle-transformers `DDIMScheduler` inference loop (which rendered a ghosted, translucent
    /// double-exposure on the default path) is removed, so the default now runs the same unified
    /// curated framework every named sampler does. `lightning` keeps its own path (`None`), and an
    /// explicit curated name passes through unchanged. This pins the routing rule the ghost fix hinges
    /// on, GPU-free — the human-eyeball coherence check is the `realvisxl_lightning` GPU smoke with
    /// `RVXL_SAMPLER=` (engine default) + real CFG.
    #[test]
    fn omitted_sampler_routes_to_curated_ddim_not_native() {
        assert_eq!(
            resolve_sampler(None),
            Some("ddim"),
            "omitted ⇒ curated ddim"
        );
        assert_eq!(
            resolve_sampler(Some("ddim")),
            Some("ddim"),
            "ddim ⇒ curated ddim"
        );
        assert_eq!(
            resolve_sampler(Some("dpmpp_2m")),
            Some("dpmpp_2m"),
            "an explicit curated name passes through"
        );
        assert_eq!(
            resolve_sampler(Some(LIGHTNING_SAMPLER)),
            None,
            "lightning takes its own few-step path, not the curated framework"
        );
        // The default is a genuinely-advertised curated solver, so it never silently euler-falls-back
        // or targets a removed native path.
        assert_eq!(DEFAULT_SAMPLER, "ddim");
        assert!(
            candle_gen::curated_sampler_names().contains(&DEFAULT_SAMPLER),
            "DEFAULT_SAMPLER must be in the advertised curated menu"
        );
    }

    /// sc-3677 parity: the production txt2img values the candle lane resolves an omitted field to
    /// must match the SceneWorks `SdxlDiffusersAdapter` reference (30 steps, CFG 7.0), and the
    /// VAE un-scale must be the diffusers-correct SDXL `scaling_factor` (0.13025 — NOT candle's
    /// hardcoded SD1.5 0.18215). `sdxl` and `realvisxl` map to this one engine, so this pins the
    /// shared default surface both ids inherit. GPU-free (asserts the constants directly).
    #[test]
    fn parity_defaults_match_diffusers_adapter() {
        assert_eq!(DEFAULT_STEPS, 30);
        // float consts: compare with an epsilon (clippy's float_cmp would reject `==`).
        assert!((DEFAULT_GUIDANCE - 7.0).abs() < f64::EPSILON);
        assert!((VAE_SCALE - 0.13025).abs() < f64::EPSILON);
    }

    /// epic 13657 / sc-13663: the three render-path artifacts (the fp16-fix VAE + both CLIP
    /// tokenizers) are NOT self-fetched — they are passed-in `LoadSpec::components` resolved by the
    /// load gate. Assert the registered ids + their order match the sc-13658 registry, and that a
    /// staged source resolves to a concrete weight file for both `Dir` (join the diffusers filename)
    /// and `File` (verbatim) stagings. GPU-free.
    #[test]
    fn components_resolve_from_staged_sources() {
        assert_eq!(
            REQUIRED_COMPONENTS,
            ["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"],
        );

        // A `Dir` staging joins the well-known diffusers filename for each artifact.
        assert_eq!(
            resolve_tokenizer_file(&WeightsSource::Dir("/models/clip_l".into())),
            std::path::Path::new("/models/clip_l/tokenizer.json"),
        );
        assert_eq!(
            resolve_vae_file(&WeightsSource::Dir("/models/vae".into())),
            std::path::Path::new("/models/vae").join(VAE_FIX_FILE),
        );

        // A `File` staging is used verbatim.
        assert_eq!(
            resolve_tokenizer_file(&WeightsSource::File("/t/tokenizer.json".into())),
            std::path::Path::new("/t/tokenizer.json"),
        );
        assert_eq!(
            resolve_vae_file(&WeightsSource::File("/v/vae.safetensors".into())),
            std::path::Path::new("/v/vae.safetensors"),
        );

        // The load gate resolves each staged source, rejecting an unknown key and a missing one.
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_component("tokenizer_clip_l", WeightsSource::Dir("/clip_l".into()))
            .with_component(
                "tokenizer_clip_bigg",
                WeightsSource::Dir("/clip_bigg".into()),
            )
            .with_component(
                "vae_fp16_fix",
                WeightsSource::File("/vae.safetensors".into()),
            );
        let comps = SdxlComponents::from_spec(&spec, crate::MODEL_ID).unwrap();
        assert!(matches!(comps.vae_fp16_fix, Some(WeightsSource::File(_))));

        // Missing the VAE component → a load-time Msg naming the id + the builder.
        let mut bad = spec.clone();
        bad.components.remove("vae_fp16_fix");
        let err = SdxlComponents::from_spec(&bad, crate::MODEL_ID)
            .unwrap_err()
            .to_string();
        assert!(err.contains("vae_fp16_fix"), "names the id: {err}");
        assert!(err.contains("with_component"), "names the builder: {err}");

        // A fused checkpoint carries its own VAE. Its structural gate requires only the two
        // tokenizer assets and does not demand the model-agnostic replacement VAE.
        let fused = LoadSpec::new(WeightsSource::File("/model.safetensors".into()))
            .with_component(
                COMPONENT_TOKENIZER_CLIP_L,
                WeightsSource::Dir("/clip_l".into()),
            )
            .with_component(
                COMPONENT_TOKENIZER_CLIP_BIGG,
                WeightsSource::Dir("/clip_bigg".into()),
            );
        let fused_components = SdxlComponents::from_spec(&fused, crate::MODEL_ID).unwrap();
        assert!(fused_components.vae_fp16_fix.is_none());
    }

    /// sc-6128: the Lightning policy is diffusers `EulerDiscreteScheduler(timestep_spacing="trailing",
    /// final_sigmas_type="zero")` built from the SDXL `scaled_linear` betas. Pin the trailing timesteps
    /// (the hand-computable `round(arange(N, 0, −N/steps)) − 1`), the σ-max prior scale, and the
    /// final-step zero-σ landing — the candle wrapper of the gen-core policy (no GPU/weights).
    #[test]
    fn lightning_policy_is_trailing_euler_with_zero_final() {
        let p = lightning_policy(5).unwrap();
        assert_eq!(p.num_steps(), 5);
        // Trailing spacing for 5 steps over 1000 train timesteps: round([1000,800,600,400,200]) − 1.
        let ts: Vec<f32> = (0..5).map(|i| p.coeffs(i).timestep).collect();
        assert_eq!(ts, vec![999.0, 799.0, 599.0, 399.0, 199.0]);
        // init_noise_scale = the largest σ (σ at the near-train-end first step) — well above 1 for SDXL.
        assert!(
            p.init_noise_scale() > 10.0,
            "σ_max should be the large trailing σ, got {}",
            p.init_noise_scale()
        );
        // c_in = 1/√(σ²+1) ∈ (0, 1] and the conditioning timestep descends across the schedule.
        let c0 = p.coeffs(0);
        assert!(c0.c_in > 0.0 && c0.c_in <= 1.0);
        assert!(c0.timestep > p.coeffs(4).timestep);
        // `final_sigmas_type="zero"`: the last step's σ_{i+1} is 0, so a_out = 0 − σ_last < 0 — the
        // step drives the latent the rest of the way to the clean sample.
        assert!(
            p.coeffs(4).a_out < 0.0,
            "final a_out should bring σ→0, got {}",
            p.coeffs(4).a_out
        );
        // The deterministic Euler step injects no noise.
        assert!((0..5).all(|i| p.coeffs(i).a_noise == 0.0));
    }

    /// sc-6128: the policy guards a degenerate 0-step request (the real `steps>=1` floor is the
    /// generator's `validate`), so `lightning_policy(0)` still yields a usable 1-step schedule rather
    /// than panicking on a `/0`.
    #[test]
    fn lightning_policy_clamps_zero_steps() {
        assert_eq!(lightning_policy(0).unwrap().num_steps(), 1);
    }

    /// sc-19753: the snapshot VAE loader now builds [`SdxlVaeDecoder`] from this restated config
    /// rather than `StableDiffusionConfig::sdxl`'s private `autoencoder` block, so the restatement
    /// has to be right. It is not a new coupling — the A1111/LDM branch of `load_components` has
    /// always built its VAE from `sdxl_vae_config()`, so a wrong value here already broke that
    /// route — and upstream's copy is a literal in the pinned candle revision, unable to drift
    /// without a pin bump. This states the values so an edit here fails loudly.
    #[test]
    fn sdxl_vae_config_states_the_diffusers_sdxl_autoencoder_block() {
        let cfg = sdxl_vae_config();
        assert_eq!(cfg.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(cfg.layers_per_block, 2);
        assert_eq!(cfg.latent_channels, 4);
        assert_eq!(cfg.norm_num_groups, 32);
        assert!(cfg.use_quant_conv);
        assert!(cfg.use_post_quant_conv);
    }

    /// Below the tiling threshold (a 64² latent → 512² output, the conformance render size) the plan
    /// produces a **single** tile, so the tiled path is a no-op pass-through identical to a monolithic
    /// decode — the guarantee that 512² output is unchanged by sc-4987.
    #[test]
    fn no_tiling_below_threshold() {
        let cfg = crate::memory_strategy::decode_tiling_config(None);
        // 64² latent = 512² output: not > the 64-latent tile, so tiling must NOT fire.
        assert!(!cfg.needs_tiling(SDXL_VAE_TILING, 1, 64, 64));
        // 128² latent = 1024² output: must fire.
        assert!(cfg.needs_tiling(SDXL_VAE_TILING, 1, 128, 128));
    }

    /// F-061 / sc-9045: the bespoke `denoise::decode_image` (trainer preview, IP / edit providers) and
    /// the registered `Pipeline::decode` now share [`SdxlLatentDecoder`]. This asserts
    /// the seam's gate is a pure function of the tiling flag + latent size — so both callers make the
    /// **same** tiled-vs-monolithic decision at identical resolutions. Combined with
    /// `tile_blend_identity_roundtrip` (tiling is exact for an identity decode) and
    /// `no_tiling_below_threshold` (≤512² stays monolithic ⇒ byte-identical to a bare decode), the two
    /// SDXL lanes are guaranteed the same output for in-memory cases and the same bounded peak on large
    /// latents. A real-VAE decode-parity check runs on the GPU conformance lane (no CPU VAE fixture).
    #[test]
    fn tiled_decode_gate_is_shared_and_size_driven() {
        let cfg = crate::memory_strategy::decode_tiling_config(None);
        // The decision both trait-seam callers make is `enabled && needs_tiling`.
        // With the flag off, no latent tiles (registered + bespoke both decode monolithically).
        let gate =
            |enabled: bool, h: i32, w: i32| enabled && cfg.needs_tiling(SDXL_VAE_TILING, 1, h, w);
        assert!(
            !gate(false, 128, 128),
            "flag off ⇒ never tile (monolithic, byte-identical)"
        );
        assert!(
            !gate(true, 64, 64),
            "512² output ⇒ single tile ⇒ monolithic"
        );
        assert!(
            gate(true, 128, 128),
            "1024² output ⇒ tiled (bounded peak) on both lanes"
        );
    }

    /// A tiny SDXL-shaped UNet config (one basic + one cross-attn down block, cross-attn mid,
    /// mirrored up) so the whole [`Pipeline::render`] seam runs on CPU in milliseconds. The only
    /// dimension that matters to the CFG-batch contract is `cross_attention_dim` — the conditioning
    /// this UNet consumes is `[B, tokens, 16]`, the shape-analogue of the real `[B, 77, 2048]`.
    fn tiny_unet_cfg() -> crate::unet::UNet2DConditionModelConfig {
        crate::unet::UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                crate::unet::BlockConfig {
                    out_channels: 32,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
                crate::unet::BlockConfig {
                    out_channels: 64,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
            ],
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 16,
            use_linear_projection: false,
        }
    }

    /// sc-14195 — the CFG-off render seam. Drives [`Pipeline::render`] end to end (request →
    /// `use_guide` → curated denoise → decode) against a tiny CPU UNet + VAE, with the conditioning
    /// batched exactly the way the production [`Pipeline::text_embeddings`] always batches it: the
    /// `[uncond, cond]` stack, **regardless of guidance**.
    ///
    /// `guidance = 1.0` is an advertised, accepted request value (`validate` passes it; the whole
    /// `use_guide = guidance > 1.0` fork exists to serve it) and means *CFG off* — one conditioned
    /// UNet forward per step over a batch-1 latent. Before the fix, `render` narrowed nothing, so
    /// that batch-1 latent met the batch-2 cross-attention K/V and the UNet died inside the
    /// attention matmul: `shape mismatch in matmul, lhs: [10, 4096, 64], rhs: [20, 64, 77]` (the
    /// story's Linux/CUDA repro at 1024²). This test reproduces that on CPU at 128².
    ///
    /// Three things are pinned, because "it no longer errors" is far too weak a bar — a narrow to
    /// the WRONG row also stops erroring, and would silently render the **negative** prompt:
    ///
    /// 1. **It runs.** guidance 1.0 renders instead of shape-mismatching (the regression).
    /// 2. **It picks the cond row, not the uncond row.** Rendered with a `[A, B]` stack, CFG-off
    ///    must produce byte-identical pixels to a `[B, B]` stack and *different* pixels from an
    ///    `[A, A]` stack — which is only true if the narrow selects index 1. This is the assertion
    ///    that kills `narrow(0, 0, 1)`; the shape check alone does not.
    /// 3. **CFG-on is untouched.** guidance 7.0 still runs the batch-2 forward.
    ///
    /// Mutation-checked, all three killed: dropping the narrow fails (1), `narrow(0, 0, 1)` fails
    /// (2), and narrowing unconditionally fails (3).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn render_at_guidance_one_runs_cfg_off_without_batch_mismatch() {
        use candle_gen::candle_nn::VarMap;
        use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;

        let device = Device::Cpu;
        let dtype = DType::F32;

        let unet_vm = VarMap::new();
        let unet = SdxlUnet::Vendored(Arc::new(
            VendoredUNet::new(
                VarBuilder::from_varmap(&unet_vm, dtype, &device),
                4,
                4,
                false,
                tiny_unet_cfg(),
            )
            .unwrap(),
        ));
        let vae_vm = VarMap::new();
        let vae = SdxlVaeDecoder::new(
            VarBuilder::from_varmap(&vae_vm, dtype, &device),
            3,
            &AutoEncoderKLConfig::default(),
        )
        .unwrap();

        // 128² ⇒ a 16² latent: below the VAE tiling threshold, so the decode stays monolithic.
        let pipe = Pipeline {
            config: StableDiffusionConfig::sdxl(None, Some(128), Some(128)),
            root: PathBuf::from("/nonexistent"),
            device: device.clone(),
            dtype,
            adapters: vec![],
            pid_spec: None,
            vae_fix: Some(WeightsSource::File("/nonexistent/vae.safetensors".into())),
            ldm: None,
            quant: None,
        };

        // Two DISTINCT conditioning rows so the selected row is observable in the pixels: `row_a`
        // stands in for the uncond (negative) encoding, `row_b` for the cond (prompt) encoding.
        // `text_embeddings()` always hands `render` the `[uncond, cond]` stack — here `[A, B]`.
        let row_a = Tensor::full(0.35f32, (1, 5, 16), &device).unwrap();
        let row_b = Tensor::full(-0.80f32, (1, 5, 16), &device).unwrap();
        let stack = |top: &Tensor, bottom: &Tensor| Tensor::cat(&[top, bottom], 0).unwrap();
        let ab = stack(&row_a, &row_b); // the real layout: uncond row 0, cond row 1
        let bb = stack(&row_b, &row_b); // cond in BOTH rows
        let aa = stack(&row_a, &row_a); // uncond in BOTH rows

        let req_at = |guidance: f32, count: u32| GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            width: 128,
            height: 128,
            count,
            seed: Some(7),
            steps: Some(2),
            guidance: Some(guidance),
            ..Default::default()
        };
        let render = |guidance: f32, ehs: &Tensor| {
            pipe.render(
                &req_at(guidance, 1),
                ehs,
                &unet,
                &vae,
                None,
                None,
                &mut |_| {},
            )
        };

        // (1) CFG off (the sc-14195 repro): guidance 1.0 ⇒ one conditioned branch, batch-1 latent.
        let off = render(1.0, &ab)
            .expect("guidance 1.0 must run the CFG-off single-branch path, not shape-mismatch");
        assert_eq!(off.len(), 1);
        // 16×16 is the *fixture's* output size, not a latent dim: `AutoEncoderKLConfig::default()`
        // has a single `block_out_channels` entry, so this toy decoder does no upsampling and emits
        // the latent's 16² directly (a real SDXL VAE would emit 128² here).
        assert_eq!((off[0].width, off[0].height), (16, 16));

        // (2) The row identity — the assertion that distinguishes "narrows" from "narrows to the
        // RIGHT row". Rendering `[A, B]` must equal rendering `[B, B]` (both carry B at index 1)
        // and must differ from `[A, A]`. `narrow(0, 0, 1)` flips both and fails here.
        let off_bb = render(1.0, &bb).expect("CFG-off render with the cond row duplicated");
        assert_eq!(
            off[0].pixels, off_bb[0].pixels,
            "CFG-off must consume the COND row (index 1) — [A,B] and [B,B] agree there"
        );
        let off_aa = render(1.0, &aa).expect("CFG-off render with the uncond row duplicated");
        assert_ne!(
            off[0].pixels, off_aa[0].pixels,
            "CFG-off must NOT consume the uncond row — that would render the negative prompt"
        );

        // (3) CFG on (the default path, unchanged): guidance 7.0 ⇒ the batched uncond+cond forward.
        let on = render(7.0, &ab).expect("the default CFG path must be unaffected");
        assert_eq!(on.len(), 1);
        // A real CFG combine at 7.0 extrapolates well past the cond-only prediction, so the two
        // arms diverge. (This does NOT catch a duplicate-everything "fix": at guidance 1.0 the
        // combine collapses to `uncond + 1·(cond − uncond) = cond`, the same pixels as the narrow.
        // Assertion (2) plus the shape mismatch are what cover that direction.)
        assert_ne!(
            off[0].pixels, on[0].pixels,
            "CFG-off and CFG-on must produce different images from the same seed"
        );

        // `count > 1` at CFG-off: the narrow is rebuilt per image inside `denoise_curated`, so pin
        // that the batch loop stays consistent and each seed still yields its own image.
        let multi = pipe
            .render(&req_at(1.0, 2), &ab, &unet, &vae, None, None, &mut |_| {})
            .expect("CFG-off must also serve count > 1");
        assert_eq!(multi.len(), 2);
        assert_eq!(
            multi[0].pixels, off[0].pixels,
            "image 0 uses the base seed ⇒ identical to the count=1 render"
        );
        assert_ne!(
            multi[0].pixels, multi[1].pixels,
            "image 1 uses base_seed+1 ⇒ a different image"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED pipeline::tests::render_at_guidance_one_runs_cfg_off_without_batch_mismatch");
    }

    #[test]
    fn cfgpp_dpmpp_2m_preserves_multistep_history_across_inert_split() {
        use candle_gen::gen_core::sampling::{
            cfgpp_denoise, cfgpp_sampler_for, CpuLatentOps, DiscreteModelSampling, Solver,
        };

        let registry = crate::provider_registry().expect("candle sdxl provider registry");
        let spec = crate::tests::sdxl_spec(WeightsSource::Dir("/nonexistent/sdxl".into()));
        let req = GenerationRequest {
            prompt: "a lighthouse in a storm".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            seed: Some(7),
            guidance: Some(7.0),
            sampler: Some("dpmpp_2m".into()),
            scheduler: Some("karras".into()),
            guidance_method: Some("cfg_pp".into()),
            ..Default::default()
        };
        let unsplit_plan = registry
            .compile_render_plan(crate::MODEL_ID, &spec, &req)
            .unwrap()
            .expect("compiled sdxl cfg_pp plan");
        assert_eq!(unsplit_plan.denoise_spans(), vec![0..8]);
        let split_plan = unsplit_plan.clone().with_inert_split_at(4).unwrap();
        assert_eq!(split_plan.denoise_spans(), vec![0..4, 4..8]);
        assert!(split_plan.low_res_start().is_none());

        let sched = sdxl_alpha_schedule().unwrap();
        let ms = DiscreteModelSampling::sdxl(&sched);
        let ops = CpuLatentOps;
        let init = vec![0.4f32, -0.25, 0.15, 0.8];
        let predict_pair = |x_in: &Vec<f32>, t: f32| -> gen_core::Result<(Vec<f32>, Vec<f32>)> {
            let tf = (t * 1e-3).sin();
            let guided = x_in
                .iter()
                .enumerate()
                .map(|(i, &v)| 0.65 * v + 0.12 * tf + (i as f32) * 0.03)
                .collect();
            let uncond = x_in
                .iter()
                .enumerate()
                .map(|(i, &v)| 0.35 * v - 0.07 * tf - (i as f32) * 0.02)
                .collect();
            Ok((guided, uncond))
        };

        let run_for_plan = |plan: &gen_core::RenderPlan| -> Vec<f32> {
            let cancel = gen_core::CancelFlag::default();
            let run = gen_core::RenderRun::new(plan, &cancel, None);
            run.encode(|| Ok::<_, CandleError>(())).unwrap();
            run.begin_rendering().unwrap();
            let mut denoise_fn =
                |x: &Vec<f32>, sigma: f32| cfgpp_denoise(&ops, &ms, x, sigma, predict_pair);
            let out = run_cfgpp_spans(
                &ops,
                Solver::Dpmpp2m,
                plan.denoise_sigmas(),
                &plan.denoise_spans(),
                init.clone(),
                |phase| run.begin_denoise_phase(phase).map_err(CandleError::from),
                |phase, _| run.end_denoise_phase(phase).map_err(CandleError::from),
                &mut denoise_fn,
                |_phase, boundary_step, latents, state| {
                    if run
                        .plan()
                        .low_res_start()
                        .is_some_and(|lr| boundary_step == lr.k)
                    {
                        state.note_renoise_transition();
                        state
                            .apply_phase_boundary(gen_core::sampling::PhaseBoundary::Reset, true)
                            .map_err(CandleError::from)?;
                    }
                    Ok(latents.clone())
                },
            )
            .unwrap();
            let out = run.decode_phase(|| Ok::<_, CandleError>(out)).unwrap();
            run.finish_rendering();
            out
        };

        let unsplit_out = run_for_plan(&unsplit_plan);
        let split_out = run_for_plan(&split_plan);
        assert_eq!(
            split_out, unsplit_out,
            "inert phase split must preserve CFG++ DPM++ 2M multistep history"
        );

        // Sanity check: resetting CFG++ solver state at step 4 (as per-span `sample` did) diverges.
        let sigmas = split_plan.denoise_sigmas();
        let sampler = cfgpp_sampler_for::<CpuLatentOps>(Solver::Dpmpp2m).unwrap();
        let mut denoise_fn =
            |x: &Vec<f32>, sigma: f32| cfgpp_denoise(&ops, &ms, x, sigma, predict_pair);
        let mid = sampler
            .sample(&ops, &mut denoise_fn, init.clone(), &sigmas[0..=4])
            .unwrap();
        let reset_out = sampler
            .sample(&ops, &mut denoise_fn, mid, &sigmas[4..=8])
            .unwrap();
        assert_ne!(
            reset_out, unsplit_out,
            "per-span reset must differ from preserved multistep history"
        );
    }
}

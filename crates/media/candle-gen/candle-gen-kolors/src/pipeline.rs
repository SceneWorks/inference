//! The candle Kolors **txt2img + source-image img2img** pipeline — ChatGLM3-6B prompt encode → seeded
//! noise or a VAE-encoded reference with the strength-selected schedule tail → the SDXL-family
//! Kolors UNet (real CFG over the leading-Euler schedule) → the SDXL VAE, driven through the
//! backend-neutral [`gen_core::Generator`] contract and parity-matched to the macOS
//! `mlx-gen-kolors` provider.
//!
//! Parity choices (grounded in the mlx `model.rs` + diffusers `KolorsPipeline`):
//! - **Conditioning**: each prompt is tokenized to the fixed 256-len left-padded form and run through
//!   ChatGLM3 with its own padding mask + `position_ids`; `context = hidden[-2]` `[1, 256, 4096]`,
//!   `pooled = hidden[-1]` last-position `[1, 4096]`. The two prompts' results are CFG-batched
//!   `[uncond, cond]` (candle's chunk convention), so the encode itself stays B==1.
//! - **`time_ids`** = `(H, W, 0, 0, H, W)` per row (SDXL `_get_add_time_ids`, original == target, no crop).
//! - **Sampler**: the leading EulerDiscrete over the 1100-step `scaled_linear` schedule
//!   ([`crate::sampler`]); `scale_model_input` divides by `√(σ²+1)`, the Euler step adds `ε·(σ_next−σ)`.
//! - **CFG**: `pred = uncond + g·(cond − uncond)`; `g ≤ 1` skips the negative branch (single forward).
//! - **Deterministic seeding (sc-3673)**: initial noise from a fixed-algorithm CPU RNG (`StdRng`,
//!   ChaCha) seeded by `seed`, moved to the device — launch-portable per seed.
//!
//! Components load at **f32** (the candle port recipe — single matmul dtype; = mlx's "f32 activations
//! over bf16 weights"); the SDXL VAE is f32-stable so it needs no fp16-fix.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::sampling::{AlphaSchedule, Scheduler, Solver};
use candle_gen::gen_core::{
    self, AdapterSpec, Conditioning, GenerationRequest, Image, PidWeights, Progress,
};
use candle_gen::quant::{PackedConfig, QLinear, MLX_GROUP_SIZE};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, Result};
// The vendored, packed-detecting SDXL UNet (candle-gen-sdxl, sc-9416) — its whole Linear surface routes
// through `candle_gen::quant`, so it loads a packed MLX tier straight from the packed parts. Kolors is an
// SDXL-family UNet, so the tier's `unet/` loads into it 1:1; the two Kolors deltas (the 5632
// `add_embedding` + the external `encoder_hid_proj`) are handled outside the block stack, exactly as the
// Kolors IP-Adapter provider already does (sc-10819).
use candle_gen_pid::PidEngine;
use candle_gen_sdxl::{
    load_vendored_unet_with_adapters, UNet2DConditionModel as VendoredUNet, VaeMomentsEncoder,
};
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};

use crate::chatglm3::ChatGlmModel;
// Shared pipeline scaffolding (sc-9001 / F-021) — the time_ids / noise / decode / CFG-encode /
// curated-σ blocks that were triplicated across this pipeline + the control / IP providers.
use crate::common::{self, CuratedSetup};
use crate::config::{ChatGlmConfig, DEFAULT_GUIDANCE, DEFAULT_STEPS};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;
use crate::unet::KolorsUNet;

/// The PiD backbone (latent-space) tag for Kolors (epic 7840 / sc-7853). Kolors reuses the SDXL VAE,
/// so its latent space is `sdxl` — the same 4× SR student SDXL resolves.
const PID_BACKBONE: &str = "sdxl";

/// diffusers SDXL VAE `scaling_factor` (Kolors reuses it). The latents are divided by this before
/// decode — the diffusers-correct SDXL value (NOT candle's hardcoded SD1.5 0.18215). `pub(crate)` so
/// the IP-Adapter provider (sc-5488) shares the exact decode scale.
pub(crate) const VAE_SCALE: f64 = 0.13025;

/// Kolors' `scaled_linear` β endpoints + train-step count — the diffusers `EulerDiscreteScheduler`
/// config the native [`KolorsEulerSampler`](crate::sampler) is built from (β₁ = **0.014**, NOT SDXL's
/// 0.012; N = **1100**, NOT SDXL's 1000). The curated [`DiscreteModelSampling`] σ-table (sc-7124) is
/// built from these same values so the ε/DDPM menu integrates over Kolors' own noise schedule.
const KOLORS_BETA_START: f32 = 0.00085;
const KOLORS_BETA_END: f32 = 0.014;
const KOLORS_TRAIN_STEPS: usize = crate::sampler::NUM_TRAIN_TIMESTEPS;

/// Build Kolors' ε-prediction α-cumprod schedule (`scaled_linear` β over the 1100 train steps) — the
/// [`DiscreteModelSampling`] source the curated unified-sampler path integrates over. Shared by the
/// txt2img [`Pipeline::denoise_curated`] (sc-7124) and the conditioned [`crate::control`] /
/// [`crate::ip_provider`] curated denoises (sc-7297), so all three speak one Kolors noise schedule.
pub(crate) fn kolors_alpha_schedule() -> Result<AlphaSchedule> {
    Ok(AlphaSchedule::scaled_linear(
        KOLORS_TRAIN_STEPS,
        KOLORS_BETA_START,
        KOLORS_BETA_END,
    ))
}

/// Curated-vs-native routing shared by the three Kolors entry points — txt2img
/// ([`Pipeline::render`]), pose-control ([`crate::control`]) and IP-Adapter
/// ([`crate::ip_provider`]) — so the decision can't drift again (sc-8984: txt2img consulted only the
/// sampler axis and silently rendered a validated scheduler-only request, e.g.
/// `scheduler: Some("karras")` with the default sampler, over the native schedule).
///
/// Returns `Some(sampler_name)` — the name to drive [`candle_gen::run_curated_sampler`] with — when
/// a curated solver name (≠ the native [`DEFAULT_SAMPLER`](crate::config::DEFAULT_SAMPLER)) OR a
/// curated scheduler is requested; `None` keeps the native byte-exact leading-Euler default (N1).
/// A scheduler-only request keeps `euler_discrete` (a non-solver alias) ⇒ the curated driver's euler
/// fallback (N3). The legacy `discrete` scheduler alias is not a curated scheduler
/// ([`Scheduler::from_name`] = `None`), so it stays native.
pub(crate) fn curated_route<'a>(
    sampler: Option<&'a str>,
    scheduler: Option<&str>,
) -> Option<&'a str> {
    let sampler_curated = sampler
        .is_some_and(|n| Solver::from_name(n).is_some() && n != crate::config::DEFAULT_SAMPLER);
    let scheduler_curated = scheduler.and_then(Scheduler::from_name).is_some();
    (sampler_curated || scheduler_curated)
        .then(|| sampler.unwrap_or(crate::config::DEFAULT_SAMPLER))
}

/// A light pipeline handle: the snapshot `root` and compute device. Heavy components load via
/// [`load_components`](Self::load_components) and are owned/cached by the generator.
pub(crate) struct Pipeline {
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), built into the cached
    /// [`Components`] so the PiD engine loads once alongside the base model. `None` ⇒ native VAE decode.
    pid_spec: Option<PidWeights>,
    adapters: Vec<AdapterSpec>,
}

/// Kolors' two UNet deltas vs stock SDXL, both auto-present in the checkpoint: the `add_embedding` MLP
/// takes **5632** = pooled(4096) + 6·256 time-ids (vs SDXL's 2816), and an `encoder_hid_proj` Linear
/// projects the ChatGLM3 context (4096) to the cross-attention width (2048). The vendored SDXL UNet
/// carries the first via [`VendoredUNet::with_add_embedding`]; the second is applied outside the block
/// stack (the vendored UNet's context arrives already at 2048). Mirrors [`crate::ip_provider`].
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 5632;
const CONTEXT_DIM: usize = 4096;
const CROSS_ATTENTION_DIM: usize = 2048;

/// The Kolors denoise UNet, in one of two builds sharing the projected-context forward contract
/// (sc-10819):
///
/// - [`Self::Dense`] — the crate's [`KolorsUNet`] (stock candle-transformers cross-attn blocks + the
///   internal `encoder_hid_proj`), built for a **dense** `Kwai-Kolors/Kolors-diffusers` snapshot.
///   Byte-identical to the pre-sc-10819 txt2img path (zero regression on every dense checkpoint).
/// - [`Self::Packed`] — the vendored, packed-detecting SDXL [`VendoredUNet`] carrying the two Kolors
///   deltas (the 5632 `add_embedding` + an **external** packed-detecting `encoder_hid_proj`), built
///   **only** for a pre-quantized MLX tier (`SceneWorks/kolors-mlx` q4/q8), where the whole
///   attention/FF/proj/time-embed Linear surface loads straight from the packed
///   `{weight u32, scales, biases}` parts (no dense staging) and the convolutions + norms stay dense.
///   This is the SAME vendored stack the Kolors IP-Adapter provider renders through ([`crate::ip_provider`]),
///   minus the IP install — so a q4/q8 tier reproduces the Kolors txt2img numerics at a packed footprint.
///
/// Both are `Arc`-shared so the seed/prompt-independent UNet is cached across `generate` calls.
#[derive(Clone)]
pub(crate) enum KolorsUnet {
    Dense(Arc<KolorsUNet>),
    Packed {
        unet: Arc<VendoredUNet>,
        /// The Kolors `encoder_hid_proj` (ChatGLM3 4096 → cross-attn 2048), packed-detected (the MLX
        /// tier packs it inside `unet/`, so a bare `candle_nn::Linear` can't read it). The vendored UNet
        /// has no internal `encoder_hid_proj`, so it is applied here (like [`crate::ip_provider`]).
        encoder_hid_proj: Arc<QLinear>,
    },
}

impl KolorsUnet {
    /// Project the raw ChatGLM3 context `[B, S, 4096]` to the cross-attention width `[B, S, 2048]`.
    /// Step-invariant (prompt-only, not timestep), so the caller hoists it out of the denoise loop
    /// (sc-9040 / F-056) and feeds the result to [`Self::forward_projected`].
    fn project_context(&self, context: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(u) => Ok(u.project_context(context)?),
            Self::Packed {
                encoder_hid_proj, ..
            } => Ok(encoder_hid_proj.forward(context)?),
        }
    }

    /// Predict `eps` for one denoise step from an **already-projected** context. Both builds take the
    /// same `(model_in, timestep, encoder_hidden_states 2048-wide, pooled, time_ids)`; the packed
    /// vendored UNet routes through `forward_instantid` with no IP tokens and no ControlNet residuals —
    /// numerically a plain Kolors forward (the exact path [`crate::ip_provider`]'s base denoise uses).
    fn forward_projected(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Dense(u) => {
                Ok(u.forward_projected(xs, timestep, encoder_hidden_states, pooled, time_ids)?)
            }
            Self::Packed { unet, .. } => Ok(unet.forward_instantid(
                xs,
                timestep,
                encoder_hidden_states,
                pooled,
                time_ids,
                None, // txt2img — no ControlNet down residuals
                None, // … and no mid residual
            )?),
        }
    }
}

/// The loaded Kolors components, `Arc`-shared so the generator can cache them across `generate` calls.
/// All four are immutable in the forward (no per-call mutable state), so no interior locking is needed.
#[derive(Clone)]
pub(crate) struct Components {
    tokenizer: Arc<KolorsTokenizer>,
    chatglm: Arc<ChatGlmModel>,
    unet: KolorsUnet,
    vae: Arc<AutoEncoderKL>,
    /// The img2img-only half of the VAE. Ordinary T2I must not pay for or retain a second copy of
    /// the encoder weights, so the first reference request populates this shared read-through slot.
    vae_encoder: Arc<Mutex<Option<Arc<VaeMomentsEncoder>>>>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
    pid: Option<Arc<PidEngine>>,
}

impl Pipeline {
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        pid_spec: Option<PidWeights>,
        adapters: Vec<AdapterSpec>,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            pid_spec,
            adapters,
        }
    }

    /// Load the four heavy components from the Kolors-diffusers snapshot (`tokenizer/`, `text_encoder/`
    /// ChatGLM3-6B, `unet/` SDXL-family UNet, `vae/` SDXL VAE).
    ///
    /// **Packed q4/q8 tiers (sc-10819, epic 9083).** A pre-quantized `SceneWorks/kolors-mlx` tier packs
    /// the UNet (pack-all) + the four ChatGLM3 projections, mirroring the dense VAE (mlx-gen #659). The
    /// packing is **detected from disk** (the `quantization` block in `unet/` & `text_encoder/`
    /// `config.json`, the same probe the SDXL packed load uses), not from `LoadSpec::quantize`:
    /// - the ChatGLM3 encoder's four projections packed-detect per Linear (`.scales` sibling) with the
    ///   `group_size` threaded from `text_encoder/config.json`;
    /// - a packed `unet/` builds the vendored, packed-detecting SDXL UNet with the two Kolors deltas; a
    ///   dense `unet/` builds the stock [`KolorsUNet`] (the byte-exact default path);
    /// - the VAE stays dense f32 in every tier (the MLX packer mirrors it, not packs it).
    pub(crate) fn load_components(&self) -> Result<Components> {
        let (tokenizer, chatglm) = self.load_conditioner()?;
        let unet = self.load_unet()?;
        let vae = self.load_vae()?;
        let pid = self.load_pid()?;
        Ok(Components {
            tokenizer: Arc::new(tokenizer),
            chatglm: Arc::new(chatglm),
            unet,
            vae: Arc::new(vae),
            vae_encoder: Arc::new(Mutex::new(None)),
            pid,
        })
    }

    /// Materialize only the tokenizer and ChatGLM conditioning phase.  Staged
    /// requests use this independently so the 6B text tower is released before
    /// the UNet phase is opened.
    fn load_conditioner(&self) -> Result<(KolorsTokenizer, ChatGlmModel)> {
        let tokenizer = KolorsTokenizer::from_dir(self.root.join("tokenizer"))?;

        // ChatGLM3-6B text encoder. The four GLM projections packed-detect on their `.scales` sibling
        // (`ChatGlmModel::new_gs`); the group is threaded from `text_encoder/config.json` when packed
        // (a dense tier has no block → the default 64, ignored by the dense Linear arm).
        let te_dir = self.root.join("text_encoder");
        let te_group = detect_packed_group(&te_dir.join("config.json"))?.unwrap_or(MLX_GROUP_SIZE);
        let chatglm = ChatGlmModel::new_gs(
            ChatGlmConfig::chatglm3_6b(),
            self.f32_vb(&te_dir)?,
            te_group,
        )?;
        Ok((tokenizer, chatglm))
    }

    /// Materialize only the denoise phase, preserving the packed/dense fork.
    fn load_unet(&self) -> Result<KolorsUnet> {
        // UNet: a packed MLX tier (a `quantization` block in `unet/config.json`) builds the vendored,
        // packed-detecting SDXL UNet + the two Kolors deltas straight from the packed parts; a dense
        // snapshot builds the stock `KolorsUNet` (byte-exact default path, zero regression).
        match detect_packed_unet(&self.root)? {
            Some((unet_file, group_size)) => {
                let vs = candle_gen::mmap_var_builder(&[unet_file], DType::F32, &self.device)?;
                // The vendored UNet + the 5632 `add_embedding` (both packed-detecting via the shared
                // `candle_gen::quant` seam); `sdxl_unet_config` is the canonical 3-block SDXL geometry
                // Kolors shares. `false` = math attention (the vendored flash path is a stub).
                let vendored = load_vendored_unet_with_adapters(
                    &self.root,
                    &self.device,
                    DType::F32,
                    &self.adapters,
                    ADDITION_TIME_EMBED_DIM,
                    PROJECTION_INPUT_DIM,
                )?;
                // The Kolors `encoder_hid_proj` is packed inside `unet/` (pack-all), so it must
                // packed-detect too — a bare `candle_nn::Linear` would read the u32 codes as garbage.
                let encoder_hid_proj = QLinear::linear_detect_gs(
                    CONTEXT_DIM,
                    CROSS_ATTENTION_DIM,
                    &vs,
                    "encoder_hid_proj",
                    true,
                    group_size,
                )?;
                Ok(KolorsUnet::Packed {
                    unet: Arc::new(vendored),
                    encoder_hid_proj: Arc::new(encoder_hid_proj),
                })
            }
            None if self.adapters.is_empty() => Ok(KolorsUnet::Dense(Arc::new(KolorsUNet::new(
                self.f32_vb(&self.root.join("unet"))?,
                false,
            )?))),
            None => {
                let vs = self.f32_vb(&self.root.join("unet"))?;
                let vendored = load_vendored_unet_with_adapters(
                    &self.root,
                    &self.device,
                    DType::F32,
                    &self.adapters,
                    ADDITION_TIME_EMBED_DIM,
                    PROJECTION_INPUT_DIM,
                )?;
                let encoder_hid_proj = QLinear::linear_detect_gs(
                    CONTEXT_DIM,
                    CROSS_ATTENTION_DIM,
                    &vs,
                    "encoder_hid_proj",
                    true,
                    MLX_GROUP_SIZE,
                )?;
                Ok(KolorsUnet::Packed {
                    unet: Arc::new(vendored),
                    encoder_hid_proj: Arc::new(encoder_hid_proj),
                })
            }
        }
    }

    /// Materialize only the native F32 SDXL VAE decode phase.  bf16 snapshots
    /// intentionally still execute this exact F32 recipe.
    fn load_vae(&self) -> Result<AutoEncoderKL> {
        Ok(AutoEncoderKL::new(
            self.f32_vb(&self.root.join("vae"))?,
            3,
            3,
            sdxl_vae_config(),
        )?)
    }

    fn load_pid(&self) -> Result<Option<Arc<PidEngine>>> {
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; Kolors shares the SDXL VAE latent space (`sdxl` student).
        Ok(match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        })
    }

    /// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the ChatGLM3 encoder + UNet ship
    /// sharded or single-file).
    fn f32_vb(&self, dir: &Path) -> Result<VarBuilder<'static>> {
        candle_gen::load_sorted_mmap(dir, DType::F32, &self.device, "kolors")
    }

    fn load_vae_encoder(&self) -> Result<VaeMomentsEncoder> {
        Ok(VaeMomentsEncoder::new(
            self.f32_vb(&self.root.join("vae"))?,
            VAE_SCALE,
        )?)
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. One image per `req.count` (each at seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        ensure_not_cancelled(req)?;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let use_guide = guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (h, w) = (req.height, req.width);
        let img2img = resolve_reference(req)?;
        let init_latents = match img2img {
            Some((image, _)) => Some(self.encode_reference(components, image, w, h)?),
            None => None,
        };

        // sc-7124 (epic 7114 P4): a curated solver name (≠ the native `euler_discrete` default / None)
        // OR a curated scheduler (sc-8984) routes the unified `Sampler` over `DiscreteModelSampling`
        // (EPS) as a NEW path — the same [`curated_route`] decision as the pose-control / IP-Adapter
        // providers. The native leading-Euler default stays byte-exact (N1) — Kolors' `steps_offset=1`
        // leading timesteps can't be bit-reproduced by `DiscreteModelSampling::timestep`, so this is
        // ADDITIVE, not a replacement.
        let curated = curated_route(req.sampler.as_deref(), req.scheduler.as_deref());

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native SDXL VAE decode. Shared across `count` images (same prompt).
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            components.pid.as_deref(),
            req,
            base_seed,
            crate::MODEL_ID,
        )?;

        let sampler = match img2img {
            Some((_, strength)) => {
                KolorsEulerSampler::img2img(steps, strength).map_err(CandleError::Msg)?
            }
            None => KolorsEulerSampler::new(steps).map_err(CandleError::Msg)?,
        };

        // Conditioning is seed-independent — encode once. CFG batch is [uncond, cond] (candle's chunk
        // order); without guidance only the positive branch is built. The ChatGLM3 encode stays local
        // (it threads `components`); the shared helper owns only the identical CFG-concat convention.
        let (context, pooled, batch) = common::cfg_batch_context(
            &req.prompt,
            negative,
            use_guide,
            common::resolve_cfg_batching(req),
            |p| self.encode(components, p),
        )?;
        let time_ids = common::build_time_ids(&self.device, batch, h, w)?;

        // The Kolors `encoder_hid_proj` (ChatGLM3 4096 → cross-attention 2048) is step-invariant, so
        // project the CFG-batched context ONCE here rather than every denoise step (sc-9040 / F-056),
        // matching the pose-control / IP-Adapter providers. The projected result feeds
        // `KolorsUNet::forward_projected`.
        let encoder_hidden_states = components.unet.project_context(&context)?;

        let (lat_h, lat_w) = ((h / 8) as usize, (w / 8) as usize);
        let total = sampler.num_steps() as u32;
        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            // A cancellation raised during prompt/reference setup must be observed even when this
            // image's strength selects an empty denoise schedule.
            ensure_not_cancelled(req)?;
            let noise = common::initial_noise(&self.device, seed, lat_h, lat_w)?;

            let latents = if let Some(name) = curated {
                self.denoise_curated(
                    req,
                    name,
                    &noise,
                    init_latents.as_ref(),
                    img2img.map_or(1.0, |(_, strength)| strength),
                    components,
                    &encoder_hidden_states,
                    &pooled,
                    &time_ids,
                    steps,
                    use_guide,
                    guidance,
                    seed,
                    on_progress,
                )?
            } else {
                let mut latents = match init_latents.as_ref() {
                    Some(init) => sampler.add_noise(init, &noise)?,
                    None => (&noise * sampler.init_noise_sigma() as f64)?,
                };
                // Per-step latent preview (epic 16948, sc-16954). A bespoke loop, so it numbers its
                // own frames on the STEP INDEX -- this lane walks a `KolorsEulerSampler` timestep
                // table, not a descending sigma array. One eval per step, so nothing repeats for the
                // counter to dedup; it still bounds and dedups on principle.
                let preview_counter =
                    candle_gen::preview::PreviewCounter::with_steps(sampler.num_steps());
                for i in 0..sampler.num_steps() {
                    if req.cancel.is_cancelled() {
                        return Err(CandleError::Canceled);
                    }
                    let scaled = (&latents / sampler.scale_in(i) as f64)?;
                    // Preview `scaled` -- the very tensor this step feeds the UNet, and therefore the
                    // domain the reused fit was measured in. Binding the preview to the lane's own
                    // `scale_in` is what stops the two coming to disagree about the renormalization.
                    candle_gen::preview::emit_preview_at(&req.preview, &preview_counter, i, || {
                        crate::preview::project_spatial_latents(&scaled)
                    });
                    let model_in = if use_guide {
                        Tensor::cat(&[&scaled, &scaled], 0)?
                    } else {
                        scaled
                    };
                    let eps = components.unet.forward_projected(
                        &model_in,
                        sampler.timestep(i) as f64,
                        &encoder_hidden_states,
                        &pooled,
                        &time_ids,
                    )?;
                    let eps = if use_guide {
                        let ch = eps.chunk(2, 0)?;
                        let (uncond, cond) = (&ch[0], &ch[1]);
                        (uncond + ((cond - uncond)? * guidance as f64)?)?
                    } else {
                        eps
                    };
                    latents = (&latents + (eps * sampler.step_dt(i) as f64)?)?;
                    on_progress(Progress::Step {
                        current: i as u32 + 1,
                        total,
                    });
                }
                latents
            };

            // Zero-strength img2img executes no denoise iteration on either route, so this explicit
            // checkpoint is the final guard against decoding a canceled request.
            ensure_not_cancelled(req)?;
            on_progress(Progress::Decoding);
            common::decode(&components.vae, pid_decoder.as_ref(), &latents)
        })
    }

    /// Request-authoritative staged execution.  The conditioning tower, denoiser, and decoder are
    /// materialized in three non-overlapping phases; img2img opens its VAE encoder before
    /// conditioning.  Every phase synchronizes before it is dropped, including on cancellation or
    /// an error through [`SynchronizedPhase`]'s drop guard.  This must remain separate from the warm
    /// [`Components`] cache: a staged request must never inherit resident heavyweight components.
    pub(crate) fn render_staged(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        ensure_not_cancelled(req)?;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let use_guide = guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (h, w) = (req.height, req.width);
        let img2img = resolve_reference(req)?;

        // Edit's latent-init phase precedes ChatGLM and does not retain the VAE encoder once the
        // deterministic posterior mean has been produced.
        let init_latents = match img2img {
            Some((image, _)) => {
                let encoder = SynchronizedPhase::new(self.load_vae_encoder()?, self.device.clone());
                let latents = self.encode_reference_with(&encoder, image, w, h)?;
                encoder.release()?;
                Some(latents)
            }
            None => None,
        };

        ensure_not_cancelled(req)?;
        let conditioner = SynchronizedPhase::new(self.load_conditioner()?, self.device.clone());
        let (context, pooled, batch) = common::cfg_batch_context(
            &req.prompt,
            negative,
            use_guide,
            common::resolve_cfg_batching(req),
            |prompt| self.encode_with(&conditioner.0, &conditioner.1, prompt),
        )?;
        let time_ids = common::build_time_ids(&self.device, batch, h, w)?;
        conditioner.release()?;

        ensure_not_cancelled(req)?;
        let unet = SynchronizedPhase::new(self.load_unet()?, self.device.clone());
        let encoder_hidden_states = unet.project_context(&context)?;
        let sampler = match img2img {
            Some((_, strength)) => {
                KolorsEulerSampler::img2img(steps, strength).map_err(CandleError::Msg)?
            }
            None => KolorsEulerSampler::new(steps).map_err(CandleError::Msg)?,
        };
        let curated = curated_route(req.sampler.as_deref(), req.scheduler.as_deref());
        let (lat_h, lat_w) = ((h / 8) as usize, (w / 8) as usize);
        let total = sampler.num_steps() as u32;
        let mut latents = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            ensure_not_cancelled(req)?;
            let seed = base_seed.wrapping_add(u64::from(index));
            let noise = common::initial_noise(&self.device, seed, lat_h, lat_w)?;
            let rendered = if let Some(name) = curated {
                self.denoise_curated_with_unet(
                    req,
                    name,
                    &noise,
                    init_latents.as_ref(),
                    img2img.map_or(1.0, |(_, strength)| strength),
                    &unet,
                    &encoder_hidden_states,
                    &pooled,
                    &time_ids,
                    steps,
                    use_guide,
                    guidance,
                    seed,
                    on_progress,
                )?
            } else {
                let mut current = match init_latents.as_ref() {
                    Some(init) => sampler.add_noise(init, &noise)?,
                    None => (&noise * sampler.init_noise_sigma() as f64)?,
                };
                let preview_counter =
                    candle_gen::preview::PreviewCounter::with_steps(sampler.num_steps());
                for step in 0..sampler.num_steps() {
                    ensure_not_cancelled(req)?;
                    let scaled = (&current / sampler.scale_in(step) as f64)?;
                    candle_gen::preview::emit_preview_at(
                        &req.preview,
                        &preview_counter,
                        step,
                        || crate::preview::project_spatial_latents(&scaled),
                    );
                    let model_in = if use_guide {
                        Tensor::cat(&[&scaled, &scaled], 0)?
                    } else {
                        scaled
                    };
                    let eps = unet.forward_projected(
                        &model_in,
                        sampler.timestep(step) as f64,
                        &encoder_hidden_states,
                        &pooled,
                        &time_ids,
                    )?;
                    let eps = if use_guide {
                        let chunks = eps.chunk(2, 0)?;
                        (&chunks[0] + ((&chunks[1] - &chunks[0])? * guidance as f64)?)?
                    } else {
                        eps
                    };
                    current = (&current + (eps * sampler.step_dt(step) as f64)?)?;
                    on_progress(Progress::Step {
                        current: step as u32 + 1,
                        total,
                    });
                }
                current
            };
            latents.push(rendered);
        }
        unet.release()?;

        ensure_not_cancelled(req)?;
        // The native VAE is deliberately F32 for all physical tiers; PiD is a replacement decoder,
        // not a co-resident fourth phase.
        let decoder =
            SynchronizedPhase::new((self.load_vae()?, self.load_pid()?), self.device.clone());
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            decoder.1.as_deref(),
            req,
            base_seed,
            crate::MODEL_ID,
        )?;
        let mut images = Vec::with_capacity(latents.len());
        for latent in &latents {
            ensure_not_cancelled(req)?;
            on_progress(Progress::Decoding);
            images.push(common::decode(&decoder.0, pid_decoder.as_ref(), latent)?);
        }
        drop(pid_decoder);
        decoder.release()?;
        Ok(images)
    }

    /// The **curated** ε/DDPM denoise (epic 7114 P4, sc-7124) — an ADDITIVE option alongside the native
    /// leading-Euler default. Drives the unified [`gen_core::sampling`] solver menu (`euler` /
    /// `euler_ancestral` / `heun` / `dpmpp_2m` / `dpmpp_sde` / `uni_pc` / `lcm` / `ddim`) over a
    /// [`DiscreteModelSampling`] (Kolors ε-prediction, `scaled_linear` β over the 1100 train steps), with
    /// the `scheduler` axis (`normal` default / `karras` / `sgm_uniform` / …) picking the σ schedule via
    /// [`candle_gen::resolve_schedule`]. Latents live in k-diffusion VE σ-space (prior = unit noise ·
    /// σ_max), kept f32 like the native path; the [`DiscreteModelSampling`] recombines ε → x0 and supplies
    /// the `1/√(σ²+1)` input scaling, so the `predict` closure just runs the UNet + CFG and returns raw ε.
    ///
    /// The native leading-Euler default is untouched, so this never affects the N1 default-parity gate —
    /// Kolors' `steps_offset=1` leading timesteps aren't bit-reproducible by `DiscreteModelSampling`, so a
    /// curated request is its own (ComfyUI-style trailing/normal) path, not a re-derivation of the default.
    #[allow(clippy::too_many_arguments)]
    fn denoise_curated(
        &self,
        req: &GenerationRequest,
        sampler: &str,
        noise: &Tensor,
        init_latents: Option<&Tensor>,
        strength: f32,
        components: &Components,
        encoder_hidden_states: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
        steps: usize,
        use_guide: bool,
        guidance: f32,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        self.denoise_curated_with_unet(
            req,
            sampler,
            noise,
            init_latents,
            strength,
            &components.unet,
            encoder_hidden_states,
            pooled,
            time_ids,
            steps,
            use_guide,
            guidance,
            seed,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_curated_with_unet(
        &self,
        req: &GenerationRequest,
        sampler: &str,
        noise: &Tensor,
        init_latents: Option<&Tensor>,
        strength: f32,
        unet: &KolorsUnet,
        encoder_hidden_states: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
        steps: usize,
        use_guide: bool,
        guidance: f32,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        // Shared curated-σ setup (sc-9001): the Kolors DiscreteModelSampling + σ-table + VE-σ prior,
        // identical across the three entry points. `init` is the raw seeded noise (lifted to σ-space).
        let setup = match init_latents {
            Some(init) => {
                CuratedSetup::new_img2img(req.scheduler.as_deref(), steps, strength, noise, init)?
            }
            None => CuratedSetup::new(req.scheduler.as_deref(), steps, noise)?,
        };
        // Per-step latent preview (epic 16948, sc-16954). The sc-16949 projector hook, so the loop is
        // not restructured and the driver owns frame numbering plus the multi-eval dedup. `ve_hook`
        // because the running latent here is raw k-diffusion VE sigma-space. Built per image: the
        // driver starts a fresh counter per call.
        let preview = crate::preview::ve_hook(&req.preview);
        let out = candle_gen::run_curated_sampler(
            Some(sampler),
            &setup.model_sampling,
            &setup.sigmas,
            setup.prior,
            seed,
            &req.cancel,
            on_progress,
            Some(&preview),
            |x_in, t| -> Result<Tensor> {
                // `x_in` is already `1/√(σ²+1)`-scaled by `denoise()`; `t` is the nearest training-step
                // index the UNet embeds. CFG batches/combines exactly like the native leading-Euler path.
                let model_in = if use_guide {
                    Tensor::cat(&[x_in, x_in], 0)?
                } else {
                    x_in.clone()
                };
                let eps = unet.forward_projected(
                    &model_in,
                    t as f64,
                    encoder_hidden_states,
                    pooled,
                    time_ids,
                )?;
                let eps = if use_guide {
                    let ch = eps.chunk(2, 0)?;
                    let (uncond, cond) = (&ch[0], &ch[1]);
                    (uncond + ((cond - uncond)? * guidance as f64)?)?
                } else {
                    eps
                };
                // Raw ε in f32 so the DiscreteModelSampling x0 recombine + solver math stay f32.
                Ok(eps.to_dtype(DType::F32)?)
            },
        )?;
        // The shared `decode` consumes the compute dtype (f32 for Kolors), like the native latents.
        Ok(out.to_dtype(DType::F32)?)
    }

    /// Deterministically VAE-encode one img2img reference: LANCZOS to the render size, RGB
    /// `[0,255]` to NCHW `[-1,1]`, then the scaled posterior mean (never a device-RNG sample).
    fn encode_reference(
        &self,
        components: &Components,
        image: &Image,
        width: u32,
        height: u32,
    ) -> Result<Tensor> {
        let (in_w, in_h) = (image.width as usize, image.height as usize);
        let expected =
            gen_core::imageops::checked_image_buffer_len(in_w, in_h, 3).ok_or_else(|| {
                CandleError::Msg(format!(
                    "kolors: invalid reference dimensions {}x{}",
                    image.width, image.height
                ))
            })?;
        if image.pixels.len() != expected {
            return Err(CandleError::Msg(format!(
                "kolors: reference pixel buffer {} != {in_w}x{in_h}x3",
                image.pixels.len()
            )));
        }
        let (out_w, out_h) = (width as usize, height as usize);
        let resized = resize_lanczos_u8(&image.pixels, in_h, in_w, out_h, out_w)?;
        let data: Vec<f32> = resized
            .into_iter()
            .map(|pixel| pixel / 127.5 - 1.0)
            .collect();
        let input = Tensor::from_vec(data, (out_h, out_w, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let vae_encoder = candle_gen::cached(&components.vae_encoder, || {
            Ok::<_, CandleError>(Arc::new(VaeMomentsEncoder::new(
                self.f32_vb(&self.root.join("vae"))?,
                VAE_SCALE,
            )?))
        })?;
        Ok(vae_encoder.encode_mean(&input)?)
    }

    fn encode_reference_with(
        &self,
        vae_encoder: &VaeMomentsEncoder,
        image: &Image,
        width: u32,
        height: u32,
    ) -> Result<Tensor> {
        let (in_w, in_h) = (image.width as usize, image.height as usize);
        let expected =
            gen_core::imageops::checked_image_buffer_len(in_w, in_h, 3).ok_or_else(|| {
                CandleError::Msg(format!(
                    "kolors: invalid reference dimensions {}x{}",
                    image.width, image.height
                ))
            })?;
        if image.pixels.len() != expected {
            return Err(CandleError::Msg(format!(
                "kolors: reference pixel buffer {} != {in_w}x{in_h}x3",
                image.pixels.len()
            )));
        }
        let (out_w, out_h) = (width as usize, height as usize);
        let resized = resize_lanczos_u8(&image.pixels, in_h, in_w, out_h, out_w)?;
        let data: Vec<f32> = resized
            .into_iter()
            .map(|pixel| pixel / 127.5 - 1.0)
            .collect();
        let input = Tensor::from_vec(data, (out_h, out_w, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        Ok(vae_encoder.encode_mean(&input)?)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])` via the ChatGLM3 encoder. Stays
    /// local (not in [`crate::common`]) because it threads the cached [`Components`]; the shared
    /// [`common::cfg_batch_context`] takes this as a closure so the CFG-concat convention is the only
    /// shared piece — the ChatGLM3 tokenize/encode specifics stay per-site.
    fn encode(&self, components: &Components, prompt: &str) -> Result<(Tensor, Tensor)> {
        self.encode_with(&components.tokenizer, &components.chatglm, prompt)
    }

    fn encode_with(
        &self,
        tokenizer: &KolorsTokenizer,
        chatglm: &ChatGlmModel,
        prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        let tokens = tokenizer.encode(prompt)?;
        Ok(chatglm.encode_prompt(&tokens)?)
    }
}

/// Own one staged component group and synchronize its device work before release.  Explicit
/// `release` catches synchronization failures; `Drop` supplies the same cleanup for cancellation,
/// error, and unwind paths where there is no result channel.
pub(crate) struct SynchronizedPhase<T> {
    value: Option<T>,
    device: Device,
}

impl<T> SynchronizedPhase<T> {
    pub(crate) fn new(value: T, device: Device) -> Self {
        Self {
            value: Some(value),
            device,
        }
    }
    pub(crate) fn release(mut self) -> Result<()> {
        self.device.synchronize()?;
        drop(self.value.take());
        Ok(())
    }
}

impl<T> std::ops::Deref for SynchronizedPhase<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("staged phase released")
    }
}

impl<T> std::ops::DerefMut for SynchronizedPhase<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().expect("staged phase released")
    }
}

impl<T> Drop for SynchronizedPhase<T> {
    fn drop(&mut self) {
        if self.value.is_some() {
            let _ = self.device.synchronize();
            drop(self.value.take());
        }
    }
}

/// Cancellation gate used before setup, at each count iteration, and immediately before decode.
/// The latter two are required for img2img strength `0`, whose native and curated schedules contain
/// no model evaluations and therefore cannot rely on a denoise-loop poll.
#[inline]
fn ensure_not_cancelled(req: &GenerationRequest) -> Result<()> {
    if req.cancel.is_cancelled() {
        Err(CandleError::Canceled)
    } else {
        Ok(())
    }
}

/// Resolve Kolors' one latent-init reference. Per-reference strength wins over the request-level
/// value, then the diffusers default (0.3); more than one reference is unsupported and fails closed.
fn resolve_reference(req: &GenerationRequest) -> Result<Option<(&Image, f32)>> {
    const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.3;
    let mut reference = None;
    for conditioning in &req.conditioning {
        if let Conditioning::Reference { image, strength } = conditioning {
            if reference.is_some() {
                return Err(CandleError::Msg(
                    "kolors: multiple reference images are not supported".into(),
                ));
            }
            reference = Some((
                image,
                strength
                    .or(req.strength)
                    .unwrap_or(DEFAULT_IMG2IMG_STRENGTH)
                    .clamp(0.0, 1.0),
            ));
        }
    }
    Ok(reference)
}

/// Parse the packed `group_size` out of a component `config.json` (sc-10819): `Some(group_size)` when
/// the file carries a `quantization` block ([`PackedConfig`]), else `None` (a dense component — a
/// missing config is treated as dense; the downstream loader gives the precise "missing X" error). Used
/// for the ChatGLM3 `text_encoder/` group thread (the per-Linear `.scales` detection is
/// [`QLinear::linear_detect_gs`]'s job, so this only recovers the grid, never gates the packed path).
/// Read the physical MLX packing grid from a component config.  The conditioned
/// IP/Control providers use this too: their base snapshot is the same canonical
/// q4/q8 artifact as the registered route, so treating those paths as dense
/// would either decode u32 codes as weights or silently cross a tier.
pub(crate) fn detect_packed_group(cfg_path: &Path) -> Result<Option<usize>> {
    if !cfg_path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(cfg_path)
        .map_err(|e| CandleError::Msg(format!("kolors: read {}: {e}", cfg_path.display())))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| CandleError::Msg(format!("kolors: parse {}: {e}", cfg_path.display())))?;
    Ok(PackedConfig::from_config(&cfg).map(|p| p.group_size as usize))
}

/// Detect a pre-quantized MLX Kolors tier at `root` (sc-10819): `Some((unet_file, group_size))` when
/// `unet/config.json` carries a `quantization` block ([`PackedConfig`]) and the packed weight file
/// (`diffusion_pytorch_model.safetensors`) exists, else `None` (a dense diffusers snapshot → the stock
/// [`KolorsUNet`]). Mirrors `candle_gen_sdxl`'s `detect_packed_unet`: the Kolors UNet reuses the vendored
/// SDXL UNet, whose Linear seam threads only the default MLX group 64 through its nested blocks, so a
/// non-64 tier is refused loudly rather than repacked on the wrong grid (the `SceneWorks/kolors-mlx`
/// tiers pack at 64, so this never fires on a real tier).
fn detect_packed_unet(root: &Path) -> Result<Option<(PathBuf, usize)>> {
    let cfg_path = root.join("unet/config.json");
    let Some(group_size) = detect_packed_group(&cfg_path)? else {
        return Ok(None);
    };
    if group_size != MLX_GROUP_SIZE {
        return Err(CandleError::Msg(format!(
            "kolors: packed tier group_size {group_size} unsupported (the vendored SDXL UNet threads \
             only {MLX_GROUP_SIZE}); a non-64 tier needs the group threaded through the UNet blocks"
        )));
    }
    let file = root.join("unet/diffusion_pytorch_model.safetensors");
    if !file.is_file() {
        return Err(CandleError::Msg(format!(
            "kolors: packed tier {} declares a quantization block but the packed UNet file {} is \
             missing",
            root.display(),
            file.display()
        )));
    }
    Ok(Some((file, group_size)))
}

/// The SDXL VAE config (`stabilityai/stable-diffusion-xl-base-1.0/vae/config.json`) — Kolors reuses it.
/// `pub(crate)` so the IP-Adapter provider (sc-5488) builds the identical VAE.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// sc-8984: a scheduler-only curated request (default / absent sampler) MUST route the curated
    /// path — it was previously dropped on the floor by txt2img, silently rendering the native
    /// schedule after `validate` accepted the scheduler.
    #[test]
    fn curated_route_scheduler_only_routes_curated() {
        // Absent sampler ⇒ the default (non-solver) name drives the curated driver's euler fallback.
        assert_eq!(curated_route(None, Some("karras")), Some("euler_discrete"));
        assert_eq!(
            curated_route(Some("euler_discrete"), Some("sgm_uniform")),
            Some("euler_discrete")
        );
    }

    #[test]
    fn curated_route_sampler_axis() {
        // A curated solver routes regardless of the scheduler axis, keeping its own name.
        assert_eq!(curated_route(Some("dpmpp_2m"), None), Some("dpmpp_2m"));
        assert_eq!(curated_route(Some("heun"), Some("karras")), Some("heun"));
    }

    #[test]
    fn curated_route_native_default_stays_native() {
        // N1: the byte-exact native leading-Euler default is untouched.
        assert_eq!(curated_route(None, None), None);
        assert_eq!(curated_route(Some("euler_discrete"), None), None);
        // The legacy `discrete` scheduler alias is NOT a curated scheduler — native schedule.
        assert_eq!(curated_route(None, Some("discrete")), None);
        assert_eq!(
            curated_route(Some("euler_discrete"), Some("discrete")),
            None
        );
        // Unknown names fall back to the native default (N3 at this layer = stay native).
        assert_eq!(curated_route(Some("not_a_solver"), None), None);
        assert_eq!(curated_route(None, Some("not_a_scheduler")), None);
    }

    #[test]
    fn reference_strength_precedence_default_and_bounds_match_mlx() {
        let request = GenerationRequest {
            strength: Some(0.7),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.4),
            }],
            ..Default::default()
        };
        assert_eq!(resolve_reference(&request).unwrap().unwrap().1, 0.4);

        let request_level = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..request.clone()
        };
        assert_eq!(resolve_reference(&request_level).unwrap().unwrap().1, 0.7);

        let defaulted = GenerationRequest {
            strength: None,
            ..request_level.clone()
        };
        assert_eq!(resolve_reference(&defaulted).unwrap().unwrap().1, 0.3);

        let clamped = GenerationRequest {
            strength: Some(2.0),
            ..request_level.clone()
        };
        assert_eq!(resolve_reference(&clamped).unwrap().unwrap().1, 1.0);

        let multiple = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                },
                Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                },
            ],
            ..Default::default()
        };
        assert!(resolve_reference(&multiple).is_err());
    }

    fn zero_strength_reference(sampler: Option<&str>) -> GenerationRequest {
        GenerationRequest {
            prompt: "edit the reference".into(),
            width: 512,
            height: 512,
            steps: Some(10),
            sampler: sampler.map(str::to_owned),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.0),
            }],
            ..Default::default()
        }
    }

    fn assert_zero_strength_cancellation(request: &GenerationRequest, curated: bool) {
        let (_, strength) = resolve_reference(request).unwrap().unwrap();
        assert_eq!(strength, 0.0);
        let native_schedule =
            KolorsEulerSampler::img2img(request.steps.unwrap() as usize, strength).unwrap();
        assert_eq!(native_schedule.num_steps(), 0, "the denoise loop is empty");
        assert_eq!(
            curated_route(request.sampler.as_deref(), request.scheduler.as_deref()).is_some(),
            curated
        );

        // Production's first checkpoint handles a pre-canceled request.
        let mut pre_cancelled = request.clone();
        pre_cancelled.cancel = gen_core::CancelFlag::new();
        pre_cancelled.cancel.cancel();
        assert!(matches!(
            ensure_not_cancelled(&pre_cancelled),
            Err(CandleError::Canceled)
        ));

        // Recreate the second case with a fresh active request, then cancel after setup: the
        // per-image and pre-decode calls use this same gate and must not depend on a denoise
        // iteration existing.
        ensure_not_cancelled(request).expect("request begins active");
        request.cancel.cancel();
        assert!(matches!(
            ensure_not_cancelled(request),
            Err(CandleError::Canceled)
        ));
    }

    #[test]
    fn native_zero_strength_img2img_observes_cancellation_without_a_step() {
        let request = zero_strength_reference(None);
        assert_zero_strength_cancellation(&request, false);
    }

    #[test]
    fn curated_zero_strength_img2img_observes_cancellation_without_a_step() {
        let request = zero_strength_reference(Some("dpmpp_2m"));
        assert_zero_strength_cancellation(&request, true);
    }

    #[test]
    fn t2i_component_load_leaves_the_img2img_encoder_cache_empty() {
        // Lock the production construction boundary: the ordinary component load creates only the
        // decoder-bearing AutoEncoderKL and an empty shared slot. VaeMomentsEncoder construction is
        // confined to encode_reference's read-through miss, reached only by reference requests.
        let source = include_str!("pipeline.rs");
        let loader_start = source.find("pub(crate) fn load_components").unwrap();
        let loader_end = source[loader_start..].find("    fn f32_vb").unwrap() + loader_start;
        let loader = &source[loader_start..loader_end];
        assert!(loader.contains("vae_encoder: Arc::new(Mutex::new(None))"));
        assert!(!loader.contains("VaeMomentsEncoder::new"));

        let encode_start = source.find("    fn encode_reference").unwrap();
        let encode_end = source[encode_start..].find("    fn encode(").unwrap() + encode_start;
        let encode = &source[encode_start..encode_end];
        assert!(encode.contains("candle_gen::cached(&components.vae_encoder"));
        assert!(encode.contains("VaeMomentsEncoder::new"));
    }

    #[test]
    fn synchronized_phase_releases_on_success_error_and_panic() {
        struct Probe(std::sync::Arc<AtomicUsize>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let device = Device::Cpu;
        let drops = std::sync::Arc::new(AtomicUsize::new(0));
        SynchronizedPhase::new(Probe(drops.clone()), device.clone())
            .release()
            .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let error_path = || -> Result<()> {
            let _phase = SynchronizedPhase::new(Probe(drops.clone()), device.clone());
            Err(CandleError::Msg("injected phase error".into()))
        };
        assert!(error_path().is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);

        let unwind = std::panic::catch_unwind({
            let drops = drops.clone();
            move || {
                let _phase = SynchronizedPhase::new(Probe(drops), Device::Cpu);
                panic!("injected phase panic");
            }
        });
        assert!(unwind.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }

    /// sc-10819: `detect_packed_unet` returns `Some((file, group))` when `unet/config.json` carries a
    /// `quantization` block AND the packed weight file exists (a `SceneWorks/kolors-mlx` tier), `None`
    /// for a dense snapshot (no block), and errors on a non-64 group (the vendored SDXL UNet threads
    /// only 64). `detect_packed_group` returns the text-encoder group for the ChatGLM3 thread. GPU-free.
    #[test]
    fn detect_packed_unet_reads_quantization_block() {
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();
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

        let got = detect_packed_unet(&tmp).unwrap();
        assert!(got.is_some(), "a quantization block ⇒ packed tier");
        assert_eq!(got.unwrap().1, 64, "group_size threaded from config");

        // A bits-only block still packs (group defaults to 64, never silent-dense — the sc-9410 rule).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 8}}"#,
        )
        .unwrap();
        assert_eq!(detect_packed_unet(&tmp).unwrap().map(|(_, g)| g), Some(64));

        // A dense config (no quantization block) ⇒ None (the stock KolorsUNet build).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"cross_attention_dim": 2048}"#,
        )
        .unwrap();
        assert!(detect_packed_unet(&tmp).unwrap().is_none());

        // A non-64 group is rejected loudly rather than repacked on the wrong grid.
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();
        assert!(detect_packed_unet(&tmp).is_err());
    }

    /// `detect_packed_group` recovers the packed group for the ChatGLM3 `text_encoder/` thread, and is
    /// `None` for a dense config or an absent file (the dense fallback the loader defaults to 64).
    #[test]
    fn detect_packed_group_reads_text_encoder_config() {
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();
        let cfg = tmp.join("config.json");
        // Absent file ⇒ dense.
        assert_eq!(detect_packed_group(&cfg).unwrap(), None);
        std::fs::write(&cfg, br#"{"quantization": {"bits": 8, "group_size": 64}}"#).unwrap();
        assert_eq!(detect_packed_group(&cfg).unwrap(), Some(64));
        std::fs::write(&cfg, br#"{"hidden_size": 4096}"#).unwrap();
        assert_eq!(detect_packed_group(&cfg).unwrap(), None);
    }

    #[test]
    fn sdxl_vae_config_pins_canonical_values() {
        let c = sdxl_vae_config();
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(c.latent_channels, 4);
        assert_eq!(c.norm_num_groups, 32);
        assert!(c.use_quant_conv);
        assert!(c.use_post_quant_conv);
    }
}

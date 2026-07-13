//! The candle Kolors **txt2img** pipeline — ChatGLM3-6B prompt encode → the SDXL-family Kolors UNet
//! (real CFG over the leading-Euler schedule) → the SDXL VAE, driven through the backend-neutral
//! [`gen_core::Generator`] contract and parity-matched to the macOS `mlx-gen-kolors` provider.
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
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::{AlphaSchedule, Scheduler, Solver};
use candle_gen::gen_core::{self, GenerationRequest, Image, PidWeights, Progress};
use candle_gen::quant::{PackedConfig, QLinear, MLX_GROUP_SIZE};
// Shared per-image batch seed (`base + index`) — one home in `candle-gen` (sc-9043 / F-059).
use candle_gen::{CandleError, Result};
// The vendored, packed-detecting SDXL UNet (candle-gen-sdxl, sc-9416) — its whole Linear surface routes
// through `candle_gen::quant`, so it loads a packed MLX tier straight from the packed parts. Kolors is an
// SDXL-family UNet, so the tier's `unet/` loads into it 1:1; the two Kolors deltas (the 5632
// `add_embedding` + the external `encoder_hid_proj`) are handled outside the block stack, exactly as the
// Kolors IP-Adapter provider already does (sc-10819).
use candle_gen_pid::PidEngine;
use candle_gen_sdxl::{sdxl_unet_config, UNet2DConditionModel as VendoredUNet};
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
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853); None ⇒ native VAE decode.
    pid: Option<Arc<PidEngine>>,
}

impl Pipeline {
    pub(crate) fn load(root: &Path, device: &Device, pid_spec: Option<PidWeights>) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            pid_spec,
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

        // UNet: a packed MLX tier (a `quantization` block in `unet/config.json`) builds the vendored,
        // packed-detecting SDXL UNet + the two Kolors deltas straight from the packed parts; a dense
        // snapshot builds the stock `KolorsUNet` (byte-exact default path, zero regression).
        let unet = match detect_packed_unet(&self.root)? {
            Some((unet_file, group_size)) => {
                let vs = candle_gen::mmap_var_builder(&[unet_file], DType::F32, &self.device)?;
                // The vendored UNet + the 5632 `add_embedding` (both packed-detecting via the shared
                // `candle_gen::quant` seam); `sdxl_unet_config` is the canonical 3-block SDXL geometry
                // Kolors shares. `false` = math attention (the vendored flash path is a stub).
                let vendored = VendoredUNet::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
                    .with_add_embedding(
                        vs.clone(),
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
                KolorsUnet::Packed {
                    unet: Arc::new(vendored),
                    encoder_hid_proj: Arc::new(encoder_hid_proj),
                }
            }
            None => KolorsUnet::Dense(Arc::new(KolorsUNet::new(
                self.f32_vb(&self.root.join("unet"))?,
                false,
            )?)),
        };

        let vae = AutoEncoderKL::new(
            self.f32_vb(&self.root.join("vae"))?,
            3,
            3,
            sdxl_vae_config(),
        )?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; Kolors shares the SDXL VAE latent space (`sdxl` student).
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(Components {
            tokenizer: Arc::new(tokenizer),
            chatglm: Arc::new(chatglm),
            unet,
            vae: Arc::new(vae),
            pid,
        })
    }

    /// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the ChatGLM3 encoder + UNet ship
    /// sharded or single-file).
    fn f32_vb(&self, dir: &Path) -> Result<VarBuilder<'static>> {
        candle_gen::load_sorted_mmap(dir, DType::F32, &self.device, "kolors")
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. One image per `req.count` (each at seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let use_guide = guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (h, w) = (req.height, req.width);

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

        let sampler = KolorsEulerSampler::new(steps).map_err(CandleError::Msg)?;

        // Conditioning is seed-independent — encode once. CFG batch is [uncond, cond] (candle's chunk
        // order); without guidance only the positive branch is built. The ChatGLM3 encode stays local
        // (it threads `components`); the shared helper owns only the identical CFG-concat convention.
        let (context, pooled, batch) =
            common::cfg_batch_context(&req.prompt, negative, use_guide, |p| {
                self.encode(components, p)
            })?;
        let time_ids = common::build_time_ids(&self.device, batch, h, w)?;

        // The Kolors `encoder_hid_proj` (ChatGLM3 4096 → cross-attention 2048) is step-invariant, so
        // project the CFG-batched context ONCE here rather than every denoise step (sc-9040 / F-056),
        // matching the pose-control / IP-Adapter providers. The projected result feeds
        // `KolorsUNet::forward_projected`.
        let encoder_hidden_states = components.unet.project_context(&context)?;

        let (lat_h, lat_w) = ((h / 8) as usize, (w / 8) as usize);
        let total = sampler.num_steps() as u32;
        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let noise = common::initial_noise(&self.device, seed, lat_h, lat_w)?;

            let latents = if let Some(name) = curated {
                self.denoise_curated(
                    req,
                    name,
                    &noise,
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
                let mut latents = (&noise * sampler.init_noise_sigma() as f64)?;
                for i in 0..sampler.num_steps() {
                    if req.cancel.is_cancelled() {
                        return Err(CandleError::Canceled);
                    }
                    let scaled = (&latents / sampler.scale_in(i) as f64)?;
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

            on_progress(Progress::Decoding);
            common::decode(&components.vae, pid_decoder.as_ref(), &latents)
        })
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
        init: &Tensor,
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
        // Shared curated-σ setup (sc-9001): the Kolors DiscreteModelSampling + σ-table + VE-σ prior,
        // identical across the three entry points. `init` is the raw seeded noise (lifted to σ-space).
        let setup = CuratedSetup::new(req.scheduler.as_deref(), steps, init)?;
        let out = candle_gen::run_curated_sampler(
            Some(sampler),
            &setup.model_sampling,
            &setup.sigmas,
            setup.prior,
            seed,
            &req.cancel,
            on_progress,
            |x_in, t| -> Result<Tensor> {
                // `x_in` is already `1/√(σ²+1)`-scaled by `denoise()`; `t` is the nearest training-step
                // index the UNet embeds. CFG batches/combines exactly like the native leading-Euler path.
                let model_in = if use_guide {
                    Tensor::cat(&[x_in, x_in], 0)?
                } else {
                    x_in.clone()
                };
                let eps = components.unet.forward_projected(
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

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])` via the ChatGLM3 encoder. Stays
    /// local (not in [`crate::common`]) because it threads the cached [`Components`]; the shared
    /// [`common::cfg_batch_context`] takes this as a closure so the CFG-concat convention is the only
    /// shared piece — the ChatGLM3 tokenize/encode specifics stay per-site.
    fn encode(&self, components: &Components, prompt: &str) -> Result<(Tensor, Tensor)> {
        let tokens = components.tokenizer.encode(prompt)?;
        Ok(components.chatglm.encode_prompt(&tokens)?)
    }
}

/// Parse the packed `group_size` out of a component `config.json` (sc-10819): `Some(group_size)` when
/// the file carries a `quantization` block ([`PackedConfig`]), else `None` (a dense component — a
/// missing config is treated as dense; the downstream loader gives the precise "missing X" error). Used
/// for the ChatGLM3 `text_encoder/` group thread (the per-Linear `.scales` detection is
/// [`QLinear::linear_detect_gs`]'s job, so this only recovers the grid, never gates the packed path).
fn detect_packed_group(cfg_path: &Path) -> Result<Option<usize>> {
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

    /// sc-10819: `detect_packed_unet` returns `Some((file, group))` when `unet/config.json` carries a
    /// `quantization` block AND the packed weight file exists (a `SceneWorks/kolors-mlx` tier), `None`
    /// for a dense snapshot (no block), and errors on a non-64 group (the vendored SDXL UNet threads
    /// only 64). `detect_packed_group` returns the text-encoder group for the ChatGLM3 thread. GPU-free.
    #[test]
    fn detect_packed_unet_reads_quantization_block() {
        let tmp = std::env::temp_dir().join(format!("sc10819_detect_{}", std::process::id()));
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

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// `detect_packed_group` recovers the packed group for the ChatGLM3 `text_encoder/` thread, and is
    /// `None` for a dense config or an absent file (the dense fallback the loader defaults to 64).
    #[test]
    fn detect_packed_group_reads_text_encoder_config() {
        let tmp = std::env::temp_dir().join(format!("sc10819_group_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = tmp.join("config.json");
        // Absent file ⇒ dense.
        assert_eq!(detect_packed_group(&cfg).unwrap(), None);
        std::fs::write(&cfg, br#"{"quantization": {"bits": 8, "group_size": 64}}"#).unwrap();
        assert_eq!(detect_packed_group(&cfg).unwrap(), Some(64));
        std::fs::write(&cfg, br#"{"hidden_size": 4096}"#).unwrap();
        assert_eq!(detect_packed_group(&cfg).unwrap(), None);
        std::fs::remove_dir_all(&tmp).ok();
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

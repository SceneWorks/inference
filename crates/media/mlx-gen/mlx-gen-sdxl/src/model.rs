//! `Sdxl` — the Stable Diffusion XL implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and explicit registration under the id `"sdxl"` (the
//! SceneWorks worker's `payload.model`).
//!
//! SDXL is the in-process Apple `mlx-examples/stable_diffusion` path (vendored at
//! `_vendor/mlx_sd/`) brought into Rust — a **U-Net** generator (conv ResBlocks + spatial/cross
//! attention + time/`text_time` micro-conditioning), dual CLIP text encoders, an SDXL VAE, and a
//! discrete Euler-Ancestral sampler with real classifier-free guidance. Parity target = the
//! vendored fp16 reference path (`StableDiffusionXL.generate_latents`), validated stage-by-stage.
//!
//! Slices land incrementally (sc-2400): this module starts as the contract + capability surface;
//! [`load`] assembles components as each slice (tokenizer → text encoders → U-Net → VAE → sampler)
//! is wired and parity-proven.

use mlx_gen::{
    curated_scheduler_names, default_seed, schedule_sigmas, ActivationMemoryAnchor, AlphaSchedule,
    Capabilities, Conditioning, ConditioningKind, DiffusionSampler, DiscreteModelSampling, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LatentDecoder, LcmSampler,
    LightningSampler, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision, Progress,
    Quant, Residency, Result, Scheduler, SizeFloor, Solver, TcdSampler, WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::Dtype;
use std::path::Path;

use mlx_gen::array::scalar;
use mlx_gen::gen_core::sampling::{vp_capture_plan, VpCapturePlan};
use mlx_gen_pid::{resolve_pid_decoder_at_sigma, PidEngine};

use crate::config::DiffusionConfig;
use crate::inpaint::{preprocess_mask, InpaintBlend};
use crate::ip_adapter::IpImageEncoder;
use crate::loader;
use crate::pipeline::{
    denoise_cfgpp_with_preview, denoise_curated_with_preview, denoise_inpaint_with_preview,
    denoise_ip_with_preview, denoise_multi_control_with_preview, denoise_with_preview,
    encode_conditioning_windows, encode_init_latents, preprocess_control_image, text_time_ids,
    ControlContext, Denoiser,
};
use crate::sampler::{AncestralEuler, EulerSampler};
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::{ControlNet, UNet2DConditionModel};
use crate::vae::Autoencoder;

/// Caller-staged tokenizer snapshot used by the registry path for a fused imported checkpoint.
pub const LDM_TOKENIZER_COMPONENT: &str = "ldm_tokenizer";

/// img2img default strength (the vendored `generate_latents_from_image` default).
const DEFAULT_STRENGTH: f32 = 0.8;
/// Masked-inpaint / outpaint default strength — the worker's `SdxlDiffusersAdapter` uses 0.85 for
/// `use_inpaint`/`outpaint` (vs 0.6 for a plain edit). An explicit request strength still wins.
const INPAINT_DEFAULT_STRENGTH: f32 = 0.85;
/// Default `ip_adapter_scale` (sc-3059) when a request doesn't override it (the worker's plus-face
/// default ≈ 0.6). In IP mode the `Reference` strength field carries the IP scale.
const IP_DEFAULT_SCALE: f32 = 0.6;
/// Default per-branch `conditioning_scale` for a ControlNet `Conditioning::Control` that leaves
/// `scale = None` (F-085) — the diffusers `controlnet_conditioning_scale` full-strength default. An
/// explicit `Some(x)`, including `Some(0.0)` for an inert branch, overrides it.
const DEFAULT_CONTROLNET_SCALE: f32 = 1.0;

/// Resolve an img2img-style strength, preserving the existing default-and-clamp semantics.
fn resolve_strength(requested: Option<f32>, default: f32) -> f32 {
    requested.unwrap_or(default).clamp(0.0, 1.0)
}

/// Resolve the ancestral schedule window for an img2img or inpaint run.
fn ancestral_strength_schedule(steps: usize, max_time: f32, strength: f32) -> (usize, f32) {
    ((steps as f32 * strength) as usize, max_time * strength)
}

/// Select the strength tail of a curated sigma schedule.
fn curated_strength_schedule(full_sigmas: &[f32], steps: usize, strength: f32) -> Vec<f32> {
    let effective_steps = (steps as f32 * strength) as usize;
    let run_start = full_sigmas
        .len()
        .saturating_sub(1)
        .saturating_sub(effective_steps);
    full_sigmas[run_start..].to_vec()
}

/// The SDXL compute dtype: the U-Net + both CLIP text encoders run **fp16** (the production
/// reference `StableDiffusionXL(float16=True)`); the VAE loads f32 inside its own loader. Shared by
/// the eager (`Resident`) load and the per-generation (`Sequential`) component loaders so both build
/// byte-identical components.
const DTYPE: Dtype = Dtype::Float16;

/// SDXL-base-1.0 production defaults (the SceneWorks `MlxSdxlAdapter`): 30 inference steps,
/// CFG 7.0, native 1024². Used when a request omits the corresponding field (consumed by the
/// `generate` pipeline slice, sc-2400 S5).
pub(crate) const DEFAULT_STEPS: u32 = 30;
pub(crate) const DEFAULT_GUIDANCE: f32 = 7.0;

/// The few-step acceleration samplers (sc-2769). Selected per request via `req.sampler`; each is
/// paired with its acceleration LoRA at load (`spec.adapters`) by the caller (the SceneWorks
/// variant manifest, epic 2755) — selecting one without its LoRA loaded yields undertrained noise.
pub(crate) const ACCEL_SAMPLERS: [&str; 3] = ["lcm", "lightning", "hyper"];

/// `original_inference_steps` for the LCM/TCD timestep selection (diffusers' default).
const LCM_ORIGINAL_STEPS: usize = 50;

/// Per-variant few-step defaults `(steps, CFG, TCD eta)`, applied when the request omits `steps`/
/// `guidance`. **Locked by the sc-2758 SDXL acceleration A/B characterization** (re-tuned here per
/// sc-2907; `sdxl` and `realvisxl` came out identical, so the table keys on the sampler only). CFG is
/// 1.0 (off) for all three — Lightning/Hyper are trained CFG-free and LCM-LoRA runs at low/no CFG —
/// which also halves the per-step UNet work. Lightning's step count must match the loaded LoRA
/// (2/4/8); LCM uses a single LoRA at any step count.
fn accel_defaults(sampler: &str) -> (u32, f32, f32) {
    match sampler {
        // LCM is the weakest method and 4 steps is too soft as a default; sc-2758 locks 8 as the
        // quality floor (the LCM-LoRA is step-free, so this is a plain default, not LoRA-bound).
        "lcm" => (8, 1.0, 0.0),
        "lightning" => (4, 1.0, 0.0),
        // Hyper-SD: TCD, deterministic (eta=0) — sc-2758 locked eta=0 for the step-graded
        // (1/2/4/8-step) LoRAs, which is the default LoRA path here.
        "hyper" => (4, 1.0, 0.0),
        _ => (DEFAULT_STEPS, DEFAULT_GUIDANCE, 0.0),
    }
}

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TARGETS["sdxl"]`).
pub const MODEL_ID: &str = "sdxl";

/// PiD latent-space backbone tag (epic 7840, sc-7848): the `sdxl` student in
/// [`mlx_gen_pid::registry`] (SDXL's 4-ch, `0.13025`-affine VAE latent). The whole SDXL family
/// shares this latent space, so `mlx-gen-kolors` (and the RealVisXL variants,
/// which register under this same `"sdxl"` generator) reuse this tag rather than redeclaring it.
pub const PID_BACKBONE: &str = "sdxl";

/// SDXL's identity + capabilities — constructible without loading weights (registry
/// introspection). Capability flags are turned on as each slice lands and is parity-proven, so the
/// descriptor never advertises a path that isn't wired (avoids the false-capability trap —
/// [[false-green-gates-mask-descope]]).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::SDXL_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "sdxl",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // img2img Reference (sc-2638) + masked inpaint/outpaint (Mask, sc-3057) + tile-ControlNet
            // detail (Control, sc-3058 — requires a control checkpoint via LoadSpec::control). LoRA
            // (kohya `lora_unet_` + PEFT, sc-2639) and LoKr (sc-2640 — Rust is more capable than the
            // vendored path, which rejects LoKr) are wired.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::Control,
            ],
            supports_lora: true,
            supports_lokr: true,
            // `euler_ancestral` is the production default (full-CFG, 30-step, the bespoke vendored
            // ancestral loop); `lcm`/`lightning`/`hyper` are the few-step acceleration samplers
            // (sc-2769), each driven by its diffusers-faithful schedule and paired with an acceleration
            // LoRA at load. The remaining names (`euler`/`heun`/`dpmpp_2m`/`dpmpp_sde`/`uni_pc`/`ddim`)
            // are the unified curated solvers (epic 7114, sc-7121) — the additive k-diffusion path over
            // `DiscreteModelSampling`; selecting one (or a non-`discrete` scheduler) routes to
            // `denoise_curated` while the default stays byte-exact. A request naming any other sampler
            // is rejected in `validate_request` rather than silently downgraded.
            samplers: vec![
                "euler_ancestral",
                "lcm",
                "lightning",
                "hyper",
                "euler",
                "heun",
                "dpmpp_2m",
                "dpmpp_sde",
                "uni_pc",
                "ddim",
            ],
            // `discrete` is the native ancestral schedule; the rest are the curated σ schedulers
            // (epic 7114 scheduler axis) usable with any curated sampler.
            schedulers: {
                let mut s = vec!["discrete"];
                s.extend(curated_scheduler_names());
                s
            },
            // Plain CFG (the shared `gen_core::guidance::cfg` over `MlxLatentOps`, epic 7434 P3 sc-7443;
            // byte-identical to the retired hand form, shared by Kolors/InstantID/PuLID via
            // `denoise_core`), plus CFG++ (`cfg_pp`, sc-8256) on the curated path — gated at dispatch to a
            // CFG++-compatible base solver (euler/ddim/dpmpp_2m) + an active guidance gap.
            supported_guidance_methods: vec!["cfg", "cfg_pp"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            // On-the-fly Q4/Q8 over the U-Net + CLIP encoders + IdentityNet, conv_shortcut kept
            // dense (sc-2769 / sc-3329). Read by the worker capability advertisement (sc-3723).
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam (epic 10834); honors Sequential offload (F-176).
            supports_sequential_offload: true,
            unconditionally_engages_staged_residency: false,
            supports_preview: true,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            // Chained denoise passes are not wired for this provider (sc-20415).
            supports_denoise_passes: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
            execution: Default::default(),
            approximation: Default::default(),
        },
    }
}

/// A loaded SDXL generator: the dual CLIP encoders + tokenizer, the U-Net, the VAE, the
/// Euler-Ancestral sampler (production default), and the `alphas_cumprod` schedule the few-step
/// acceleration samplers (LCM/Lightning/Hyper) build on — assembled from a snapshot directory.
pub struct Sdxl {
    descriptor: ModelDescriptor,
    tokenizer: ClipBpeTokenizer,
    sampler: EulerSampler,
    /// DDPM `alphas_cumprod` from the SDXL `scaled_linear` betas — shared by the acceleration
    /// samplers (sc-2769). Built once at load (the ancestral `sampler` keeps its own σ table).
    alpha_schedule: AlphaSchedule,
    /// Number of loaded ControlNet branches (`spec.control` + `spec.extra_controls`), computed at
    /// load so `generate` can reject a `Control`-count mismatch under either residency (the
    /// `Sequential` seam holds loader closures, not the spec, so this is captured up front).
    control_count: usize,
    /// Component-residency strategy (epic 10834 Phase 1, sc-10839; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`] at [`load`]. `Resident` (default) holds
    /// every heavy component (both CLIP encoders + U-Net + control/IP/VAE/PiD) warm for the whole job
    /// and across jobs; `Sequential` holds only the per-phase loader closures and re-loads each
    /// component in phase order per generation (text encode → **drop the encoders** → U-Net/VAE
    /// denoise+decode), bounding peak unified memory to the largest single working set instead of the
    /// sum, at the cost of the warm cache — for Macs where the resident set would OOM. The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush once for all providers.
    residency: Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>,
    /// The load-time `OffloadPolicy` verdict, kept as the DEFAULT for a request that names no
    /// [`GenerationMemory::stage_residency`](mlx_gen::gen_core::GenerationMemory). Rung 1 is
    /// request-scoped from SC-15525 onward; the policy is no longer the authority.
    default_stage_residency: bool,
    /// Whether THIS load can execute ladder rung 4 — see
    /// [`memory_strategy::streamable`](crate::memory_strategy::streamable).
    streamable: bool,
    loaded_spec: LoadSpec,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
}

/// One correctness-only production-latent decode comparison.
///
/// This test seam deliberately carries no duration, allocator, peak-memory, or calibration field.
/// The latent is captured from the normal [`Sdxl::generate_impl`] denoise path immediately before
/// its production decode, then decoded once densely and once with the caller's candidate tiling.
#[doc(hidden)]
pub struct DecodeQualitySample {
    pub production_latent: mlx_rs::Array,
    pub dense: Image,
    pub tiled: Image,
}

type FinalLatentObserver<'a> =
    &'a mut dyn FnMut(&Autoencoder, &mlx_rs::Array, Option<&dyn LatentDecoder>) -> Result<()>;

/// The heavy render-phase components (everything but the text encoders): the U-Net, its ControlNet
/// branches / IP-Adapter, the VAE, and the optional PiD decoder. Owned by the `Resident` components
/// (held for the whole job) or by a `Sequential` generate (loaded after the encoders are dropped,
/// freed when the job ends).
pub(crate) struct SdxlHeavyOwned {
    unet: UNet2DConditionModel,
    /// ControlNet branches (sc-3058; MultiControlNet sc-3378), loaded from `LoadSpec::control` +
    /// `LoadSpec::extra_controls`. Empty when no control checkpoint was supplied. `generate` requires
    /// exactly one `Control` conditioning per loaded branch (paired by order); their residuals are
    /// summed (the diffusers `MultiControlNetModel` rule).
    controls: Vec<ControlNet>,
    /// Optional IP-Adapter image-token source (sc-3059), loaded from `LoadSpec::ip_adapter`. When
    /// present, the model is in "IP mode": a `Reference` conditioning is the image prompt (txt2img +
    /// IP), not an img2img init. The decoupled-attn K/V projections are installed into `unet`.
    ip_adapter: Option<IpImageEncoder>,
    vae: Autoencoder,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7848): loaded when the spec
    /// carries [`LoadSpec::pid`]. `Some` ⇒ a `req.use_pid` generation decodes the final SDXL latent
    /// through the `sdxl` PiD student (4× SR) instead of the VAE. `None` ⇒ the default byte-exact
    /// VAE decode.
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body is written once and
/// runs identically whether the components are held resident (borrowed out of [`ResidentComponents`])
/// or were just loaded by the `Sequential` path — mirrors candle's `DitRef` (sc-10769). Cheap refs.
struct SdxlHeavy<'a> {
    unet: &'a UNet2DConditionModel,
    controls: &'a [ControlNet],
    ip_adapter: Option<&'a IpImageEncoder>,
    vae: &'a Autoencoder,
    pid: Option<&'a PidEngine>,
}

impl SdxlHeavyOwned {
    fn as_ref(&self) -> SdxlHeavy<'_> {
        SdxlHeavy {
            unet: &self.unet,
            controls: &self.controls,
            ip_adapter: self.ip_adapter.as_ref(),
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
}

/// Construct an [`Sdxl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-diffusion-xl-base-1.0` snapshot (the diffusers multi-component tree —
/// `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `unet/`, `vae/`).
///
/// **Dtype:** the U-Net + both CLIP text encoders run **fp16**, matching the production reference
/// (`StableDiffusionXL(float16=True)`); the **VAE stays f32** (the vendored always loads the
/// autoencoder f32 — the SDXL VAE is fp16-unstable). The whole fp16 path is byte-identical to the
/// reference at the matched MLX — now 0.32.0, RE-CONFIRMED by sc-12896: the fp16 UNet golden was
/// re-dumped on the 0.32.0 non-NAX env and `unet_single_forward_matches_vendored_fp16` measures
/// 100.00% byte-exact (16384/16384). Established on 0.31.2 by sc-2721 (needs sc-2772's NAX 16-bit
/// fix + the compiled `gelu_exact`); fp16 is bit-identical across the 0.31.2→0.32.0 bump (on
/// SDXL's path the sc-12896 cross-stack drift is f32-only; dense-bf16 paths drift in other crates).
/// The lower-level `load_unet`/`load_text_encoder_*` keep an f32 path for the tight stage gates.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16 (the production \
             reference dtype) — drop the precision override"
                .into(),
        ));
    }
    if matches!(spec.weights, WeightsSource::File(_)) {
        mlx_gen::gen_core::reject_unknown_components(spec, &[LDM_TOKENIZER_COMPONENT], MODEL_ID)?;
        let tokenizer_root = match spec.components.get(LDM_TOKENIZER_COMPONENT) {
            Some(WeightsSource::Dir(path)) => path,
            Some(WeightsSource::File(path)) => {
                return Err(Error::Msg(format!(
                    "sdxl: imported fused checkpoint component '{LDM_TOKENIZER_COMPONENT}' must be a directory, got {}",
                    path.display()
                )))
            }
            None => {
                return Err(Error::Msg(format!(
                    "sdxl: imported fused checkpoint requires caller-staged '{LDM_TOKENIZER_COMPONENT}' tokenizer assets"
                )))
            }
        };
        return load_from_ldm_file(spec, tokenizer_root);
    }
    Ok(Box::new(load_concrete(spec)?))
}

/// Construct the concrete generator for the correctness-only production-latent admission harness.
/// Normal registry callers should use [`load`].
#[doc(hidden)]
pub fn load_concrete(spec: &LoadSpec) -> Result<Sdxl> {
    if spec.precision != Precision::Bf16 {
        // `Precision::Bf16` is the registry's dense sentinel; the dense path runs fp16 (the
        // production dtype). A non-default precision flag is rejected rather than silently ignored.
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16 (the production \
             reference dtype) — drop the precision override"
                .into(),
        ));
    }
    // Resolve the snapshot dir up front — a fail-fast for BOTH residencies (Sequential defers the
    // heavy component build to each generate, but a single-file source is still wrong, so reject it
    // here rather than at the first generate).
    let root = resolve_root(spec)?;

    let cfg = DiffusionConfig::sdxl_base();
    let alpha_schedule =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    // Component residency (epic 10834 Phase 1, sc-10839): the default `Resident` builds every heavy
    // component now and holds it warm; `Sequential` keeps only the spec and re-loads per generate in
    // phase order (encode → drop encoders → denoise/decode) to bound peak memory. The `Resident`
    // build is byte-identical to the pre-sc-10839 `load` — the same loaders, adapter/quant order,
    // and PiD overlay, just assembled through the shared per-phase helpers.
    let control_count = spec.control.is_some() as usize + spec.extra_controls.len();
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate. An
    // already-packed turnkey loads packed (no re-quant); `Resident` quantizes once. So warn only for
    // the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), descriptor().id)?
        {
            mlx_gen::residency::warn_sequential_requantize(descriptor().id, q.bits());
        }
    }
    let residency = build_residency(spec)?;
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(descriptor().id, spec)?;
    Ok(Sdxl {
        descriptor: descriptor(),
        default_stage_residency: crate::memory_strategy::default_stage_residency(spec),
        streamable: crate::memory_strategy::streamable(spec),
        loaded_spec: spec.clone(),
        memory_strategy,
        tokenizer: loader::load_tokenizer(root)?,
        sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE)?,
        alpha_schedule,
        control_count,
        residency,
    })
}

/// Load a fused SDXL LDM/A1111 checkpoint in memory. The fused file supplies both text encoders,
/// UNet, and VAE; only the model-agnostic tokenizer assets come from `tokenizer_root`.
pub fn load_from_ldm_file(spec: &LoadSpec, tokenizer_root: &Path) -> Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    let file_pin = match &spec.weights {
        WeightsSource::File(_) => spec
            .weights_file_pin()?
            .expect("File weights must resolve to a pin"),
        WeightsSource::Dir(_) => {
            return Err(Error::Msg(
                "sdxl LDM loader expects a fused .safetensors file".into(),
            ))
        }
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16".into(),
        ));
    }
    let residency = build_ldm_residency(spec, &file_pin)?;
    let cfg = DiffusionConfig::sdxl_base();
    let alpha_schedule =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    // A fused LDM/A1111 checkpoint has no re-openable **per-component** source — every component is
    // cut out of one file by `ldm::split_ldm_checkpoint` — so `memory_strategy::streamable` reports
    // `false` for its `WeightsSource::File` and rung 4 is declared Missing for this load rather than
    // silently executing resident. Rung 1 is a different question and is genuinely executable here:
    // the fused *file* is re-openable through the retained pin, which is what
    // [`build_ldm_residency`] keeps for both phase loaders (sc-18317).
    // The public helper retains the same prepared identity guarantee as registry dispatch: contract
    // projection reopens the fused header and tokenizer loading opens staged files, so complete both
    // under the retained source + full prepared-file guards before exposing the generator.
    let (memory_strategy, tokenizer) = spec.read_prepared_files_unchanged(|| {
        file_pin.read_unchanged(|_| {
            Ok::<_, Error>((
                crate::memory_strategy::memory_strategy_contract(descriptor().id, spec)?,
                loader::load_tokenizer(tokenizer_root)?,
            ))
        })
    })?;
    Ok(Box::new(Sdxl {
        descriptor: descriptor(),
        default_stage_residency: crate::memory_strategy::default_stage_residency(spec),
        streamable: false,
        loaded_spec: spec.clone(),
        memory_strategy,
        tokenizer,
        sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE)?,
        alpha_schedule,
        control_count: spec.control.is_some() as usize + spec.extra_controls.len(),
        residency,
    }))
}

/// The fused-LDM policy→[`Residency`] dispatch, mirroring [`build_residency`]'s single-seam shape
/// for the imported-checkpoint route (sc-18317).
///
/// **Every arm retains the pin and keeps reload loaders.** Before sc-18317 the `Resident` arm read
/// the fused file once through a valid pin, dropped both, and built `Residency::resident` — a
/// non-rebuildable owner whose loaders return errors. That made rung 1 a *declared but unexecutable*
/// capability on this route: the descriptor publishes `supports_sequential_offload` (so
/// `staged_residency_availability()` is `Selectable`), the contract builder declares
/// `StagedResidency` `Implemented` (it derives only `load_shape` and rung 4 from the spec), rung 1
/// declares no prerequisite for `validate_selection` to refuse, and the shared
/// `Capabilities::validate_request` floor does not inspect `stage_residency` at all. A staged
/// selection was therefore admitted everywhere and then refused inside `generate` by
/// `Residency::ensure_rebuildable` — the epic's core defect class.
///
/// Nothing here is new machinery: the `Sequential` arm has always re-split the fused checkpoint
/// through this same pin on every generate, and `PinnedWeightsFile::read_unchanged` documents
/// repeated reopen as its intended use ("keep this same pin for every lazy or sequential reopen").
/// The nearest neighbour agrees — `mlx-gen-z-image`'s fused ComfyUI `Checkpoint` source builds a
/// request-scoped residency whose per-phase loaders re-split one pinned file. So the `Resident` arm
/// now routes through [`Residency::from_policy_with_resident`], which keeps a **distinct warm
/// aggregate**: the warm pair is still built by ONE fused read (byte-identical composition to the
/// pre-sc-18317 eager build, `load_pid = true` for the reusable PiD superset), while the two staged
/// phase loaders re-split the file per phase exactly as `Sequential` does.
///
/// One behavioural delta, and it is deliberate: under `Resident` the warm pair is now built on first
/// use instead of at load. Structural validity of the checkpoint is still proven at load — the
/// contract projection reads and classifies every tensor header through the pin
/// (`model::fused_ldm_component_footprint`) and rejects a file missing a text encoder, U-Net or VAE
/// — and `load` is only ever reached from the job that then calls `generate`, so a tensor-level
/// failure surfaces in the same job either way. `Sequential` on this route already deferred
/// everything.
pub(crate) fn build_ldm_residency(
    spec: &LoadSpec,
    file_pin: &mlx_gen::gen_core::PinnedWeightsFile,
) -> Result<Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>> {
    let warm_file = file_pin.clone();
    let warm_spec = spec.clone();
    let text_file = file_pin.clone();
    let text_quant = spec.quantize;
    let heavy_file = file_pin.clone();
    let heavy_spec = spec.clone();
    Residency::from_policy_with_resident(
        spec.offload_policy,
        move || {
            warm_spec.read_prepared_files_unchanged(|| {
                warm_file.read_unchanged(|file| {
                    let crate::ldm::LdmComponents {
                        unet,
                        clip_l,
                        clip_bigg,
                        vae,
                    } = crate::ldm::split_ldm_checkpoint(file)?;
                    let text = build_ldm_text(clip_l, clip_bigg, warm_spec.quantize)?;
                    let heavy = build_ldm_heavy(&warm_spec, unet, vae, true)?;
                    Ok::<_, Error>((text, heavy))
                })
            })
        },
        move || {
            text_file.read_unchanged(|file| {
                let crate::ldm::LdmComponents {
                    clip_l, clip_bigg, ..
                } = crate::ldm::split_ldm_checkpoint(file)?;
                build_ldm_text(clip_l, clip_bigg, text_quant)
            })
        },
        move |load_pid| {
            heavy_spec.read_prepared_files_unchanged(|| {
                heavy_file.read_unchanged(|file| {
                    let crate::ldm::LdmComponents { unet, vae, .. } =
                        crate::ldm::split_ldm_checkpoint(file)?;
                    build_ldm_heavy(&heavy_spec, unet, vae, load_pid)
                })
            })
        },
    )
}

fn build_ldm_text(
    clip_l: mlx_gen::weights::Weights,
    clip_bigg: mlx_gen::weights::Weights,
    quantize: Option<Quant>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    // Build from lazy fused-file arrays first. The component maps are consumed and dropped by the
    // loaders, leaving only the encoder-owned handles before quantization/materialization walks one
    // projection at a time under the caller's immutable-file guard.
    let mut te1 = loader::load_text_encoder_1_from_weights(clip_l, DTYPE)?;
    let mut te2 = loader::load_text_encoder_2_from_weights(clip_bigg, DTYPE)?;
    if let Some(quant) = quantize {
        te1.quantize(quant.bits())?;
        te2.quantize(quant.bits())?;
    }
    te1.materialize_weights()?;
    te2.materialize_weights()?;
    Ok((te1, te2))
}

fn build_ldm_heavy(
    spec: &LoadSpec,
    unet_weights: mlx_gen::weights::Weights,
    vae_weights: mlx_gen::weights::Weights,
    load_pid: bool,
) -> Result<SdxlHeavyOwned> {
    // The U-Net component map is consumed by the constructor so no second full source map retains
    // arrays while the packed model is walked below. The VAE intentionally stays dense and is
    // materialized as its own bounded component under the same immutable-file guard.
    let mut unet = loader::load_unet_from_weights(unet_weights, DTYPE)?;
    vae_weights.materialize()?;
    let vae = loader::load_vae_from_weights(vae_weights)?;
    if !spec.adapters.is_empty() {
        let coverage = if std::env::var_os("SDXL_LORA_VENDORED").is_some() {
            crate::adapters::LoraCoverage::Vendored
        } else {
            crate::adapters::LoraCoverage::Complete
        };
        crate::adapters::apply_sdxl_adapters_with(&mut unet, &spec.adapters, coverage)?;
    }
    let mut controls = Vec::new();
    if let Some(source) = &spec.control {
        controls.push(loader::load_controlnet(source, DTYPE)?);
    }
    for source in &spec.extra_controls {
        controls.push(loader::load_controlnet(source, DTYPE)?);
    }
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(path)) => {
            let (encoder, pairs) = loader::load_ip_adapter(path, DTYPE)?;
            unet.install_ip_adapter(pairs)?;
            Some(encoder)
        }
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory".into(),
            ))
        }
        None => None,
    };
    if let Some(quant) = spec.quantize {
        let bits = quant.bits();
        unet.quantize(bits)?;
        for control in &mut controls {
            control.quantize(bits)?;
        }
    }
    unet.materialize_weights()?;
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|weights| PidEngine::from_spec(weights, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(SdxlHeavyOwned {
        unet,
        controls,
        ip_adapter,
        vae,
        pid,
    })
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-10839; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in
/// exactly one place. `Resident` eager-loads the dual CLIP text encoders + heavy bundle now (the heavy
/// loader with `use_pid = true`, loading any PiD overlay once and reusing it); `Sequential` captures
/// the two per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both use the
/// same [`load_text_encoders`] / [`load_heavy`], so the `Resident` composition is byte-identical to the
/// pre-seam one. The deferral is weight-free-testable: under `Sequential` this touches no component
/// weights, so a dispatch that ignored `offload_policy` would eager-load and fail the "Sequential
/// defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
) -> Result<Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoders(resolve_root(&spec_text)?, spec_text.quantize),
        move |use_pid| load_heavy(&spec_heavy, resolve_root(&spec_heavy)?, use_pid),
    )
}

/// Resolve the snapshot directory from the load spec, rejecting a single-file source (SDXL needs the
/// diffusers multi-component tree). Shared by [`load`] and the `Sequential` per-phase loaders.
fn resolve_root(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "sdxl expects a snapshot directory (tokenizer/ text_encoder/ unet/ vae/ …), not a \
             single .safetensors file"
                .into(),
        )),
    }
}

/// Load the dual CLIP text encoders (the phase-A component under `Sequential`, dropped before the
/// U-Net loads). Factored out of [`load`] so the `Resident` and `Sequential` paths build byte-
/// identical encoders: both run fp16 with the same optional Q4/Q8 (`group_size 64`) over every
/// quantizable Linear + the token Embedding, matching the sc-2604/sc-1975 quant scope.
fn load_text_encoders(
    root: &Path,
    quant: Option<Quant>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    let mut te1 = loader::load_text_encoder_1_dtype(root, DTYPE)?;
    let mut te2 = loader::load_text_encoder_2_dtype(root, DTYPE)?;
    if let Some(q) = quant {
        // F-144 (sc-11129): reject a requested-vs-packed tier mismatch up front. `quantize()` silently
        // no-ops on already-packed weights, so a Q4 request over a pre-quantized Q8 turnkey would
        // otherwise serve Q8 with no diagnostic. `needs_load_time_quant` errors on a mismatch; on a
        // matching-packed or dense snapshot it returns Ok and the quantize below stands (a no-op on the
        // already-packed encoders, a real pack on a dense snapshot).
        mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), descriptor().id)?;
        let bits = q.bits();
        te1.quantize(bits)?;
        te2.quantize(bits)?;
    }
    Ok((te1, te2))
}

/// Load the heavy render-phase components — U-Net (+ LoRA/LoKr merge + IP-Adapter install + Q4/Q8),
/// ControlNet branches (+ Q4/Q8), VAE (f32), and the optional PiD overlay — everything but the text
/// encoders. Factored out of [`load`] so the `Sequential` path can load these AFTER the encoders are
/// dropped (bounding peak to `max(encoders, U-Net+VAE)`), and the `Resident` path builds the same
/// bundle up front. The operation order matches the pre-sc-10839 `load` (adapter merge before quant),
/// and the components are independent of the text encoders, so both residencies are byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, load_pid: bool) -> Result<SdxlHeavyOwned> {
    let mut unet = loader::load_unet_dtype(root, DTYPE)?;
    if !spec.adapters.is_empty() {
        // Merge LoRA (kohya `lora_unet_` / PEFT, sc-2639) and LoKr (sc-2640) into the dense fp16
        // U-Net weights at load — the production reference merges into the `float16=True` U-Net too,
        // and merging (not a
        // forward-time residual) keeps the chaos-sensitive ancestral sampler bit-exact. Out-of-surface
        // keys (mid_block/ff/conv) are surfaced in the report, not dropped.
        //
        // Coverage (sc-2671): default to the strictly-more-correct COMPLETE surface — mid_block +
        // the GEGLU FF the vendored `lora.py` silently drops — so SDXL LoRAs apply in full, matching
        // diffusers (Michael's correctness-over-parity call, 2026-06-03). `SDXL_LORA_VENDORED` is the
        // escape hatch back to the legacy 515-module surface for byte-parity with the retired Python
        // path.
        let coverage = if std::env::var_os("SDXL_LORA_VENDORED").is_some() {
            eprintln!(
                "sdxl: SDXL_LORA_VENDORED set — restricting LoRA to the legacy vendored 515-module \
                 surface (mid_block + ff dropped; byte-parity with the retired Python path)"
            );
            crate::adapters::LoraCoverage::Vendored
        } else {
            crate::adapters::LoraCoverage::Complete
        };
        crate::adapters::apply_sdxl_adapters_with(&mut unet, &spec.adapters, coverage)?;
    }
    let vae = loader::load_vae(root)?; // VAE always f32 (vendored loads the autoencoder float16=False)

    // ControlNet branches (sc-3058; MultiControlNet sc-3378) — `spec.control` first, then each
    // `spec.extra_controls`, all at the U-Net dtype (fp16). Quantized with the U-Net below when
    // `spec.quantize` is set (the encoder-copy Linears; conv stem / cond-embedding / zero-convs stay
    // dense, matching the U-Net scope).
    let mut controls: Vec<ControlNet> = Vec::new();
    if let Some(src) = &spec.control {
        controls.push(loader::load_controlnet(src, DTYPE)?);
    }
    for src in &spec.extra_controls {
        controls.push(loader::load_controlnet(src, DTYPE)?);
    }

    // Optional IP-Adapter (sc-3059) — install the decoupled-attn K/V pairs into the still-mutable,
    // pre-quant U-Net (so they quantize with it) and keep the image-token encoder.
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => {
            let (enc, pairs) = loader::load_ip_adapter(p, DTYPE)?;
            unet.install_ip_adapter(pairs)?;
            Some(enc)
        }
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory, not a single file"
                    .into(),
            ));
        }
        None => None,
    };

    if let Some(q) = spec.quantize {
        // Q4/Q8 (group_size 64) over every quantizable Linear of the U-Net + control branches —
        // applied AFTER the adapter merge (the merge needs the dense weight; `merge_dense_delta`
        // errors on a quantized base, matching the fork's "LoRA merged pre-quantization"). The core
        // `AdaptableLinear::quantize` casts each weight to bf16 before packing (sc-2604): SDXL ships
        // fp16/fp32 on disk, and quantizing the as-loaded dtype would give drifted group scales — the
        // sc-1975 "Q8 broken on base-1.0". Convs / norms / token & position embeddings stay dense
        // (gather lookups, not matmuls). The text encoders are quantized in [`load_text_encoders`];
        // the **VAE stays f32** — its only Linears are the tiny quant/post-quant projections
        // (negligible memory), and a dense decode preserves output quality. Scope verified
        // empirically by the full `load(Q).generate()` gate (sc-2641).
        //
        // F-144 (sc-11129): reject a requested-vs-packed U-Net tier mismatch first — `quantize()`
        // no-ops on an already-packed snapshot, so a Q4 request over a pre-quantized Q8 turnkey would
        // silently serve Q8. On a matching-packed or dense snapshot this returns Ok and the quantize
        // below stands (a no-op on the packed U-Net, a real pack on the dense control branches).
        mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), descriptor().id)?;
        let bits = q.bits();
        unet.quantize(bits)?;
        for cn in &mut controls {
            cn.quantize(bits)?;
        }
    }

    // PiD decoder overlay (epic 7840, sc-7848): load the `sdxl` student + Gemma caption encoder once
    // when the spec carries it AND this generate uses it (`load_pid`, F-177) — Resident passes `true`
    // (loaded once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips the student +
    // its Gemma caption encoder entirely. Shared across the whole SDXL family (sdxl/realvisxl) — and
    // Kolors, which loads its own engine via `mlx_gen_sdxl::model::PID_BACKBONE`.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };

    // Ladder rung 4 (SC-15525, closing SC-16355) — armed LAST, and that ordering is the whole
    // correctness argument. Each `Transformer2D`'s stream captures the installed IP-Adapter K/V
    // projections and the forward-time residual adapters off its FINISHED resident blocks, so the
    // streamed and resident paths cannot disagree about which state landed where. Arming before the
    // IP install or the adapter merge would capture an empty state and silently render without the
    // image prompt.
    //
    // `memory_strategy::streamable` is the single authority for whether this load may arm at all: it
    // refuses a fused single file, an eager load shape, a load-time quantization that materializes
    // the trunk, and an adapter load that merged its delta into weights the snapshot does not carry.
    if crate::memory_strategy::streamable(spec) {
        let unet_file =
            loader::resolve_weight_file(root, "unet", "diffusion_pytorch_model", DTYPE)?;
        unet.arm_block_streams(
            &WeightsSource::File(unet_file),
            spec.quantize.map(|q| q.bits()),
        );
    }

    Ok(SdxlHeavyOwned {
        unet,
        controls,
        ip_adapter,
        vae,
        pid,
    })
}

// Written out rather than `mlx_gen::impl_generator!` because SDXL now answers the memory-strategy
// hooks (SC-15525); the macro covers only `descriptor`/`validate`/`generate`.
impl mlx_gen::gen_core::Generator for Sdxl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.loaded_spec, &self.memory_strategy, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy::begin_request(
            self.descriptor.id,
            &self.loaded_spec,
            &self.memory_strategy,
            context,
        )
    }
}

impl Sdxl {
    /// Run the ordinary production denoise path and compare the exact final latent under dense and
    /// candidate tiled decode. This is an explicit correctness-only seam for admission tooling; it
    /// does not read clocks or allocator/memory counters and does not change the production output.
    #[doc(hidden)]
    pub fn generate_decode_quality(
        &self,
        req: &GenerationRequest,
        tiling: &mlx_gen::tiling::TilingConfig,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(GenerationOutput, Vec<DecodeQualitySample>)> {
        let mut samples = Vec::new();
        let output = {
            let mut capture = |vae: &Autoencoder,
                               latent: &mlx_rs::Array,
                               pid: Option<&dyn LatentDecoder>|
             -> Result<()> {
                latent.eval()?;
                let dense =
                    crate::pipeline::decode_image_tiled(vae, latent, pid, None, Some(&req.cancel))?;
                let tiled = crate::pipeline::decode_image_tiled(
                    vae,
                    latent,
                    pid,
                    Some(tiling),
                    Some(&req.cancel),
                )?;
                samples.push(DecodeQualitySample {
                    production_latent: latent.clone(),
                    dense,
                    tiled,
                });
                Ok(())
            };
            self.generate_impl_with_final_latent_observer(req, on_progress, Some(&mut capture))?
        };
        Ok((output, samples))
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (text encode → drop the CLIP encoders under `Sequential` → load
    /// the U-Net/control/IP/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run`] seam (sc-11125), which owns the eval/drop/clear discipline, the
    /// stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.generate_impl_with_final_latent_observer(req, on_progress, None)
    }

    fn generate_impl_with_final_latent_observer(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
        mut final_latent_observer: Option<FinalLatentObserver<'_>>,
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let sampler_name = req.sampler.as_deref().unwrap_or("euler_ancestral");
        let is_accel = ACCEL_SAMPLERS.contains(&sampler_name);
        // Curated unified path (epic 7114, sc-7121): a curated solver name (other than the bespoke
        // `euler_ancestral` default + the accel profiles) OR a non-`discrete` scheduler routes to the
        // additive k-diffusion `denoise_curated` over `DiscreteModelSampling`. The ancestral default
        // with no curated knob stays on the bespoke vendored loop, byte-exact (the N1 default gate).
        let scheduler_curated = req
            .scheduler
            .as_deref()
            .and_then(Scheduler::from_name)
            .is_some();
        let sampler_curated = Solver::from_name(sampler_name).is_some()
            && !is_accel
            && sampler_name != "euler_ancestral";
        let use_curated = !is_accel && (sampler_curated || scheduler_curated);
        // F-082: the accel samplers build their own distilled few-step schedule, so a request pairing
        // one with a curated σ scheduler used to validate and then silently drop the scheduler.
        // Reject the combination instead of misreporting the request as honored.
        if is_accel && scheduler_curated {
            return Err(Error::Msg(format!(
                "sdxl: the {sampler_name:?} acceleration sampler uses its own distilled schedule \
                 and cannot honor the {:?} scheduler — drop `scheduler` (or pick a curated sampler)",
                req.scheduler.as_deref().unwrap_or_default()
            )));
        }
        // Per-variant defaults for the few-step samplers; the production defaults otherwise.
        let (def_steps, def_cfg, eta) = if is_accel {
            accel_defaults(sampler_name)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE, 0.0)
        };
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let cfg = req.guidance.unwrap_or(def_cfg);
        let cfg_on = cfg > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let reference = self.resolve_reference(req)?;
        let mask_img = self.resolve_mask(req)?;
        let max_time = self.sampler.max_time();

        // Acceleration variants are txt2img-only in v1 (epic 2755 "Image-only v1"); reject an init
        // image rather than silently ignoring it.
        if is_accel && reference.is_some() {
            return Err(Error::Msg(format!(
                "sdxl: the {sampler_name:?} acceleration sampler is txt2img-only (no img2img \
                 reference) in this build"
            )));
        }
        // Inpaint (Mask) rides the ancestral img2img path and needs an init image to blend against.
        if mask_img.is_some() {
            if is_accel {
                return Err(Error::Msg(
                    "sdxl: inpaint masks are not supported with the acceleration samplers".into(),
                ));
            }
            if use_curated {
                return Err(Error::Msg(
                    "sdxl: curated samplers/schedulers are not supported with an inpaint Mask (its \
                     per-step blend has no post-step hook in the callback Sampler) — use the default \
                     euler_ancestral"
                        .into(),
                ));
            }
            if reference.is_none() {
                return Err(Error::Msg(
                    "sdxl: inpaint requires an init image (a Reference) alongside the Mask".into(),
                ));
            }
        }
        // ControlNet (sc-3058; MultiControlNet sc-3378): each `Control` conditioning pairs, in order,
        // with a loaded control branch (`spec.control` + `spec.extra_controls`); their residuals are
        // summed. Needs the ancestral path; not combined with an inpaint mask in this build.
        let control_reqs = self.resolve_control(req)?;
        if !control_reqs.is_empty() {
            if is_accel {
                return Err(Error::Msg(
                    "sdxl: ControlNet is not supported with the acceleration samplers".into(),
                ));
            }
            if control_reqs.len() != self.control_count {
                return Err(Error::Msg(format!(
                    "sdxl: {} Control conditioning(s) passed but the model was loaded with {} control \
                     checkpoint(s) (set LoadSpec::control + extra_controls, one per Control, in order)",
                    control_reqs.len(),
                    self.control_count
                )));
            }
            if mask_img.is_some() {
                return Err(Error::Msg(
                    "sdxl: combining a ControlNet (Control) with an inpaint Mask is not supported"
                        .into(),
                ));
            }
        }

        // ── Phase A: text encode (epic 10834 Phase 1, sc-10839; sc-11125). Seed-independent (no RNG)
        // — tokenizing above the residency lifecycle, then hoisting the encode above the control/IP
        // builds and the per-image loop, is byte-identical to the F-068 order. Under `Sequential` the
        // shared seam LOADS the dual CLIP encoders, encodes, materializes, then DROPS them +
        // `clear_cache()` so their ~1 GB frees before the U-Net/control/IP bundle loads below —
        // bounding peak to `max(encoders, U-Net+VAE)`. Under `Resident` it borrows the warm encoders
        // (byte-identical to the pre-sc-10839 `encode_conditioning`).
        // sc-20528: a prompt past CLIP's 77-token context is split into windows, not truncated. The
        // window count is decided ONCE here, as the max over both CFG rows, so `[cond, uncond]` stay
        // stackable and the negative prompt takes exactly the same path as the positive one. A
        // request whose rows both fit is a single window that IS the pre-sc-20528 token batch.
        let tokens = self
            .tokenizer
            .tokenize_windows(&req.prompt, if cfg_on { Some(negative) } else { None })?;

        // ── Ladder request resolution (SC-15525) ────────────────────────────────────────────────
        // Rung 1 is request-scoped from here on: the load-time `OffloadPolicy` is only the default a
        // request that names nothing keeps.
        let stage_residency =
            crate::memory_strategy::stage_residency(req, self.default_stage_residency);
        let window_size = crate::memory_strategy::transformer_window_size(req)?;
        // Two fail-closed guards a calibration harness driving `generate` with a hand-built
        // `GenerationMemory` must still cross — it never went through `safety_check`.
        if window_size.is_some() && !self.streamable {
            return Err(Error::Unsupported(
                "sdxl: bounded transformer residency needs a DeferredMaterialization load over a \
                 snapshot directory whose U-Net stays lazy and whose adapters (if any) are \
                 replayable; this generator cannot stream its blocks"
                    .into(),
            ));
        }
        if window_size.is_some() && !stage_residency {
            return Err(Error::Unsupported(
                "sdxl: bounded transformer residency requires staged residency engaged in the same \
                 request — without the phase release both CLIP towers stay resident through the \
                 denoise and the request peak does not move"
                    .into(),
            ));
        }
        let decode_tiling =
            crate::memory_strategy::decode_tiling_for_contract(req, &self.memory_strategy)?;
        let forward_plan = crate::plan::SdxlForwardPlan::with_attention(
            crate::memory_strategy::attention_plan(req)?,
        )
        .with_window(window_size.map(|size| crate::plan::SdxlBlockWindow {
            size,
            cancel: &req.cancel,
        }));

        self.residency.run_request_scoped(
            stage_residency,
            false,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &(ClipTextEncoder, ClipTextEncoder)| {
                encode_conditioning_windows(&text.0, &text.1, &tokens)
            },
            // Materialize the conditioning + pooled while the encoders are still alive (Sequential
            // only) — MLX is lazy, so an un-evaluated output keeps the encoders referenced through the
            // graph and the drop would free nothing (cf. Wan's `encode_text_staged`).
            |enc| match enc {
                Some(enc) => Ok(mlx_rs::transforms::eval([&enc.0, &enc.1])?),
                None => Ok(()),
            },
            // ── Establish the heavy render components (U-Net + control/IP + VAE + PiD) and run the
            // denoise/decode body once against the `heavy` borrow — identical for both residencies.
            // `on_progress` is threaded through the seam (F-179) and shadows the outer sink here.
            |heavy_owned, enc, on_progress| {
        let heavy = heavy_owned.as_ref();
        let (conditioning, pooled) = enc;

        // Build the ControlNet contexts once (seed-independent): preprocess each control image to
        // [0,1] NHWC and CFG-batch it to match the U-Net input, paired by order with a loaded branch.
        let mut control_ctxs: Vec<ControlContext> = Vec::with_capacity(control_reqs.len());
        for ((image, scale), cn) in control_reqs.iter().zip(heavy.controls) {
            let img = preprocess_control_image(image, req.width, req.height)?;
            let img = if cfg_on {
                concatenate_axis(&[&img, &img], 0)?
            } else {
                img
            };
            control_ctxs.push(ControlContext {
                controlnet: cn,
                // Precompute the step-invariant conditioning embedding once per run (F-069).
                cond_embed: cn.embed_cond(&img)?,
                scale: *scale,
            });
        }

        // IP-Adapter (sc-3059): when the model carries IP weights and a Reference is present (no
        // mask/control/accel), the Reference is the image prompt (txt2img + IP), NOT an img2img init.
        // The IP scale rides the Reference `strength` field (default 0.6). Tokens are seed-independent
        // → built once, CFG-batched with a zeros uncond row so the negative pass gets no IP signal.
        let ip_mode = heavy.ip_adapter.is_some()
            && reference.is_some()
            && mask_img.is_none()
            && control_reqs.is_empty()
            && !is_accel;
        let ip_scale = reference.and_then(|(_, s)| s).unwrap_or(IP_DEFAULT_SCALE);
        let ip_tokens = if ip_mode {
            let enc = heavy.ip_adapter.expect("ip_adapter present in ip_mode");
            let (image, _) = reference.expect("reference present in ip_mode");
            let tokens = enc.tokens(image)?;
            Some(if cfg_on {
                let zeros = enc.zeros_like_tokens(tokens.dtype())?;
                concatenate_axis(&[&tokens, &zeros], 0)?
            } else {
                tokens
            })
        } else {
            None
        };

        let time_ids = text_time_ids(pooled.shape()[0]);
        let latent_shape = [1, (req.height / 8) as i32, (req.width / 8) as i32, 4];
        // img2img/inpaint init latents (the f32 VAE encode) and the inpaint mask are seed-independent
        // too (F-068). `init_latents` is Some exactly for the ancestral img2img/inpaint paths — a
        // Reference that is neither an accel run nor an IP image prompt; `mask_latent` adds the mask.
        let init_latents = match reference {
            Some((image, _)) if !is_accel && !ip_mode => Some(encode_init_latents(
                heavy.vae, image, req.width, req.height,
            )?),
            _ => None,
        };
        let mask_latent = match mask_img {
            Some(mask) if init_latents.is_some() => {
                Some(preprocess_mask(mask, req.width, req.height)?)
            }
            _ => None,
        };

        // PiD decode overlay (epic 7840, sc-7848) + `from_ldm` early-stop (sc-8049). SDXL is the lone
        // **variance-preserving** PiD student. Its two denoise paths keep the latent in DIFFERENT frames,
        // so the from_ldm capture is handled per-path:
        //   • ancestral (the default; txt2img / img2img / control / IP via `denoise_core`) stores the
        //     latent ALREADY renormalized to the VP frame `(x0+σε)/√(σ²+1)` = `√(1−σ_vp²)·x0 + σ_vp·ε`
        //     at every node (see `EulerSampler::step`/`add_noise`), so a truncated x_k is handed to PiD
        //     as-is (no rescale);
        //   • curated (opt-in k-diffusion) stores RAW VE latents `x0+σ·ε`, so a truncated x_k is mapped
        //     into the VP frame by the plan's rescale (`1/√(1+σ²)`) before decode.
        // Both stay 0.13025-normalized throughout (the loop runs in the scaled latent space `vae.decode`
        // consumes), so no extra normalization is applied. The plan is image-independent (the schedule is
        // fixed by steps / scheduler / strength), so resolve it once here and mint the decoder at the
        // achieved degrade σ. The clean σ=0 decode (`vp_plan = None`) is byte-identical to before. The
        // few-step accel path (decode-bound → no from_ldm benefit, sc-7993) and the inpaint mask-blend
        // (needs the full schedule to σ=0) keep the clean decode; a from_ldm request on either errors
        // loudly rather than silently dropping the knob.
        let vp_plan: Option<VpCapturePlan> = if req.use_pid && req.pid_capture_sigma.is_some() {
            if is_accel || mask_latent.is_some() {
                return Err(Error::Msg(format!(
                    "{}: pid_capture_sigma (from_ldm early-stop) is not supported on the SDXL {} path \
                     (it keeps the clean σ=0 decode); use the standard ancestral or curated denoise for \
                     from_ldm (sc-8049)",
                    self.descriptor.id,
                    if is_accel {
                        "few-step accel"
                    } else {
                        "inpaint mask-blend"
                    }
                )));
            }
            // Resolve the VP capture against the EXACT σ schedule this run will denoise, so `keep` and the
            // achieved degrade σ agree with the truncated trajectory. (This mirrors the per-path schedule
            // build in the count loop below — deterministic host math, no RNG, so it does not perturb the
            // ancestral noise stream.)
            let edm_sigmas = if use_curated {
                let ms = DiscreteModelSampling::sdxl(&self.alpha_schedule);
                let sched = req
                    .scheduler
                    .as_deref()
                    .and_then(Scheduler::from_name)
                    .unwrap_or(Scheduler::Normal);
                let full_sigmas = schedule_sigmas(sched, &ms, steps);
                if init_latents.is_some() {
                    let strength = resolve_strength(
                        reference.and_then(|(_, strength)| strength),
                        DEFAULT_STRENGTH,
                    );
                    curated_strength_schedule(&full_sigmas, steps, strength)
                } else {
                    full_sigmas
                }
            } else {
                let (eff, start_time) = if init_latents.is_some() {
                    let strength = resolve_strength(
                        reference.and_then(|(_, strength)| strength),
                        DEFAULT_STRENGTH,
                    );
                    ancestral_strength_schedule(steps, max_time, strength)
                } else {
                    (steps, max_time)
                };
                AncestralEuler::new(&self.sampler, eff, start_time)?.edm_sigmas()
            };
            vp_capture_plan(&edm_sigmas, req.pid_capture_sigma)
        } else {
            None
        };
        let capture_sigma = vp_plan.map(|p| p.sigma).unwrap_or(0.0);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            heavy.pid,
            req,
            base_seed,
            self.descriptor.id,
            capture_sigma,
        )?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);

        mlx_gen::diagnostics::record_phase_boundary(
            mlx_gen::diagnostics::BenchmarkPhaseBoundary::DenoiseStart,
        );
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // One image per iteration (the vendored `_run_one`, n_images=1), each with its own seed.
            let seed = base_seed.wrapping_add(i as u64);
            // Seed the global RNG up front; the hoisted conditioning/VAE encodes drew no RNG, so the
            // first draw here is the init noise (the prior / img2img add_noise) — matching the
            // reference stream.
            mlx_rs::random::seed(seed)?;

            // Curated unified-sampler path (epic 7114, sc-7121): k-diffusion VE-σ sampling over a
            // `DiscreteModelSampling`, additive alongside the bespoke ancestral default. The latents
            // live in raw σ-space; the curated solver + scheduler are selected per request. Supports
            // txt2img / img2img / ControlNet / IP-Adapter (inpaint is guarded out above).
            if use_curated {
                let ms = DiscreteModelSampling::sdxl(&self.alpha_schedule);
                let sched = req
                    .scheduler
                    .as_deref()
                    .and_then(Scheduler::from_name)
                    .unwrap_or(Scheduler::Normal);
                let full_sigmas = schedule_sigmas(sched, &ms, steps);
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                // Raw k-diffusion σ-space init: txt2img `ε·σ_max`; img2img runs the strength-tail of the
                // schedule, seeded `x₀ + ε·σ_start` (diffusers EulerDiscrete add_noise). A strength that
                // rounds to 0 effective steps leaves the schedule at `[0.0]` → the init is returned.
                let (run_sigmas, init) = if let Some(x_0) = &init_latents {
                    let strength = resolve_strength(
                        reference.and_then(|(_, strength)| strength),
                        DEFAULT_STRENGTH,
                    );
                    let rs = curated_strength_schedule(&full_sigmas, steps, strength);
                    let init = add(x_0, &multiply(&noise, scalar(rs[0]))?)?;
                    (rs, init)
                } else {
                    let init = multiply(&noise, scalar(full_sigmas[0]))?;
                    (full_sigmas, init)
                };
                // PiD from_ldm early-stop (sc-8049): truncate the curated k-diffusion schedule to the
                // VP-capture `keep` nodes so the solver stops at the achieved degrade σ; the clean path
                // (`vp_plan = None`) runs the full schedule byte-identically.
                let keep_sigmas: &[f32] = match &vp_plan {
                    Some(p) => &run_sigmas[..p.keep],
                    None => &run_sigmas,
                };
                let ip = ip_tokens.as_ref().map(|t| (t, ip_scale));
                // CFG++ (sc-8256): opt-in via `guidance_method == "cfg_pp"`, only with a CFG++-compatible
                // base solver (euler/ddim/dpmpp_2m) and an active guidance gap (`cfg > 1`). Anything else
                // — including `cfg_pp` on an incompatible sampler — falls back to the plain curated path
                // (N3, never a hard-fail), so the default is byte-untouched.
                let want_cfgpp = req.guidance_method.as_deref() == Some("cfg_pp")
                    && cfg > 1.0
                    && Solver::from_name(sampler_name)
                        .is_some_and(mlx_gen::gen_core::sampling::base_supports_cfgpp);
                let latents = if want_cfgpp {
                    denoise_cfgpp_with_preview(
                        heavy.unet,
                        Some(sampler_name),
                        &ms,
                        keep_sigmas,
                        init,
                        &conditioning,
                        &pooled,
                        &time_ids,
                        cfg,
                        &req.cancel,
                        on_progress,
                        &req.preview,
                        &control_ctxs,
                        ip,
                        None,
                        forward_plan,
                    )?
                } else {
                    denoise_curated_with_preview(
                        heavy.unet,
                        Some(sampler_name),
                        &ms,
                        keep_sigmas,
                        init,
                        &conditioning,
                        &pooled,
                        &time_ids,
                        cfg,
                        seed,
                        &req.cancel,
                        on_progress,
                        &req.preview,
                        &control_ctxs,
                        ip,
                        None,
                        forward_plan,
                    )?
                };
                // Curated latents live in RAW VE σ-space (`x0+σ·ε`); an early-stop leaves x_k at σ>0, so
                // map it into the student's VP frame with the plan's rescale (`1/√(1+σ²)`) before decode.
                // The clean path (`vp_plan = None`) leaves it byte-identical. (sc-8049)
                let latents = match &vp_plan {
                    Some(p) => multiply(&latents, scalar(p.rescale))?,
                    None => latents,
                };
                if let Some(observer) = final_latent_observer.as_deref_mut() {
                    observer(heavy.vae, &latents, pid_ref)?;
                }
                mlx_gen::diagnostics::record_phase_boundary(
                    mlx_gen::diagnostics::BenchmarkPhaseBoundary::DecodeStart,
                );
                on_progress(Progress::Decoding);
                images.push(crate::pipeline::decode_image_tiled(
                    heavy.vae,
                    &latents,
                    pid_ref,
                    decode_tiling.as_ref(),
                    Some(&req.cancel),
                )?);
                continue;
            }

            // Build the run's sampler + its seeded init latents. The denoise loop is driven entirely
            // by the sampler's own schedule (`sampler.num_steps()`), so the trait owns the per-step
            // timestep, the input scaling, and the step math.
            let (latents, sampler, blend): (
                mlx_rs::Array,
                Box<dyn DiffusionSampler + '_>,
                Option<InpaintBlend>,
            ) = if is_accel {
                // Few-step acceleration (txt2img): unit-noise prior scaled into the sampler's space.
                let s = self.build_accel_sampler(sampler_name, steps, eta, seed);
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                let lat = s.scale_initial_noise(&noise)?;
                (lat, s, None)
            } else if let (Some(x_0), Some(mask_latent)) = (&init_latents, &mask_latent) {
                // Masked inpaint (sc-3057): same ancestral img2img start, but keep the FIXED prior
                // noise so the per-step blend can pin the black (keep) region to the init noised to
                // each step's σ. Default strength 0.85 (the worker's inpaint default).
                let strength = resolve_strength(
                    reference.and_then(|(_, strength)| strength),
                    INPAINT_DEFAULT_STRENGTH,
                );
                let (eff, start_step) =
                    ancestral_strength_schedule(steps, max_time, strength);
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                let x_t = self.sampler.add_noise_with(x_0, &noise, start_step)?;
                // The kept region is noised to each step's "next" time `t_prev` (schedule[i].1).
                let t_prev: Vec<f32> = self
                    .sampler
                    .timesteps(eff, start_step)?
                    .into_iter()
                    .map(|(_, tp)| tp)
                    .collect();
                let blend = InpaintBlend::new(
                    &self.sampler,
                    mask_latent.clone(),
                    x_0.clone(),
                    noise,
                    t_prev,
                );
                (
                    x_t,
                    Box::new(AncestralEuler::new(&self.sampler, eff, start_step)?),
                    Some(blend),
                )
            } else if let Some(x_0) = &init_latents {
                // img2img (ancestral; the vendored `generate_latents_from_image`): start at
                // `max_time·strength`, run `int(steps·strength)` steps — NO min-1 floor (strength ≤
                // 1/steps ⇒ 0 steps ⇒ init returned unchanged, dodging the σ=0 ancestral `σ_up` 0/0
                // → NaN).
                let strength = resolve_strength(
                    reference.and_then(|(_, strength)| strength),
                    DEFAULT_STRENGTH,
                );
                let (eff, start_step) =
                    ancestral_strength_schedule(steps, max_time, strength);
                let x_t = self.sampler.add_noise(x_0, start_step)?;
                // PiD from_ldm early-stop (sc-8049): truncate to the VP-capture `keep` steps; the stored
                // ancestral latent is already the VP frame, so it is handed to PiD as-is. Clean path
                // (`vp_plan = None`) keeps the full schedule.
                let sampler = AncestralEuler::new(&self.sampler, eff, start_step)?;
                (
                    x_t,
                    Box::new(match &vp_plan {
                        Some(p) => sampler.truncate_to(p.keep - 1),
                        None => sampler,
                    }),
                    None,
                )
            } else {
                // txt2img (ancestral): seeded prior.
                let prior = self.sampler.sample_prior(&latent_shape)?;
                // PiD from_ldm early-stop (sc-8049): truncate to the VP-capture `keep` steps (see the
                // img2img arm); clean path (`vp_plan = None`) keeps the full schedule.
                let sampler = AncestralEuler::new(&self.sampler, steps, max_time)?;
                (
                    prior,
                    Box::new(match &vp_plan {
                        Some(p) => sampler.truncate_to(p.keep - 1),
                        None => sampler,
                    }),
                    None,
                )
            };

            let d = Denoiser::with_plan(heavy.unet, sampler.as_ref(), forward_plan);
            let latents = if let Some(tokens) = &ip_tokens {
                denoise_ip_with_preview(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    &req.preview,
                    tokens,
                    ip_scale,
                )?
            } else if !control_ctxs.is_empty() {
                denoise_multi_control_with_preview(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    &req.preview,
                    &control_ctxs,
                )?
            } else if let Some(b) = &blend {
                denoise_inpaint_with_preview(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    &req.preview,
                    b,
                )?
            } else {
                denoise_with_preview(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    &req.preview,
                )?
            };

            if let Some(observer) = final_latent_observer.as_deref_mut() {
                observer(heavy.vae, &latents, pid_ref)?;
            }

            mlx_gen::diagnostics::record_phase_boundary(
                mlx_gen::diagnostics::BenchmarkPhaseBoundary::DecodeStart,
            );
            on_progress(Progress::Decoding);
            images.push(crate::pipeline::decode_image_tiled(
                    heavy.vae,
                    &latents,
                    pid_ref,
                    decode_tiling.as_ref(),
                    Some(&req.cancel),
                )?);
        }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

impl Sdxl {
    /// Build the per-run few-step acceleration sampler (sc-2769). `name` is one of
    /// [`ACCEL_SAMPLERS`]; `steps` is the inference step count (Lightning must match the loaded
    /// LoRA's 2/4/8); `eta` is the TCD stochasticity (Hyper-SD); `seed` is the request seed driving
    /// the deterministic between-step re-noise (D6). The samplers cast the U-Net input to fp16 (the
    /// loaded compute dtype) and run their step math in f32.
    fn build_accel_sampler(
        &self,
        name: &str,
        steps: usize,
        eta: f32,
        seed: u64,
    ) -> Box<dyn DiffusionSampler> {
        let n_train = self.alpha_schedule.alphas_cumprod.len();
        let sched = self.alpha_schedule.clone();
        match name {
            "lcm" => Box::new(LcmSampler::new(
                sched,
                n_train,
                LCM_ORIGINAL_STEPS,
                steps,
                Dtype::Float16,
                seed,
            )),
            "lightning" => Box::new(LightningSampler::new(
                &sched,
                n_train,
                steps,
                Dtype::Float16,
            )),
            "hyper" => Box::new(TcdSampler::new(
                sched,
                n_train,
                LCM_ORIGINAL_STEPS,
                steps,
                eta,
                Dtype::Float16,
                seed,
            )),
            // `generate` only calls this for `name ∈ ACCEL_SAMPLERS`.
            _ => unreachable!("build_accel_sampler: {name:?} is not an acceleration sampler"),
        }
    }

    /// Extract the single img2img init image + its strength from the request's conditioning (the
    /// per-reference strength wins over `req.strength`). SDXL img2img conditions on exactly one init
    /// image, so more than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple reference images are not supported (single img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// Extract the single inpaint mask from the request's conditioning (sc-3057). White = repaint,
    /// black = keep. SDXL supports one mask; more than one is an error.
    fn resolve_mask<'a>(&self, req: &'a GenerationRequest) -> Result<Option<&'a Image>> {
        let mut mask = None;
        for c in &req.conditioning {
            if let Conditioning::Mask { image } = c {
                if mask.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple inpaint masks are not supported".into(),
                    ));
                }
                mask = Some(image);
            }
        }
        Ok(mask)
    }

    /// Collect the ControlNet control images + `conditioning_scale`s (sc-3058; MultiControlNet
    /// sc-3378), in request order. Each pairs with a loaded control branch (`spec.control` +
    /// `spec.extra_controls`); the count must match (validated in `generate`). A single `Control` is
    /// the common case; more than one runs as MultiControlNet (residuals summed).
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Vec<(&'a Image, f32)>> {
        let mut controls = Vec::new();
        for c in &req.conditioning {
            if let Conditioning::Control { image, scale, .. } = c {
                // `None` → the diffusers `controlnet_conditioning_scale` default (full strength);
                // `Some(x)` — including `Some(0.0)` for an inert branch — is used verbatim (F-085).
                controls.push((image, scale.unwrap_or(DEFAULT_CONTROLNET_SCALE)));
            }
        }
        Ok(controls)
    }
}

/// SDXL's VAE produces a `/8` latent and the three-block U-Net applies two stride-2 downsamplers
/// before mirroring them through exact skip concatenations. Each image axis must therefore be a
/// multiple of `8 * 2² = 32`; accepting only the VAE stride can produce an odd intermediate whose
/// upsampled extent no longer matches its skip tensor. `validate_request` enforces this structural
/// multiple before production denoise.
pub const SIZE_MULTIPLE: u32 = 32;

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // Shared capability floor (F-022): count/steps range, size range, negative_prompt/guidance/
    // true_cfg support gating + finiteness, sampler/scheduler/guidance_method membership, and accepted
    // conditioning kinds. Delegating to core (like Kolors, F-132) restores the `true_cfg` and
    // `guidance_method` checks this hand-rolled copy had dropped — a `cfg_pp` typo in `guidance_method`
    // previously slipped through and silently rendered plain CFG. `steps == Some(0)` is now the floor's
    // job too. The `?` keeps the typed `Error::Unsupported` for capability gaps.
    caps.validate_request(MODEL_ID, req)?;

    // SDXL-specific checks layered on top of the shared floor:
    if req.prompt.is_empty() {
        return Err(Error::Msg("sdxl: prompt must not be empty".into()));
    }
    // The /8 VAE plus two exact U-Net downsample/upsample joins require SIZE_MULTIPLE.
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "sdxl: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    match &spec.weights {
        WeightsSource::Dir(_) => mlx_gen::PerComponentBytes::from_spec_subdirs(
            spec,
            &["text_encoder", "text_encoder_2"],
            &["unet"],
            &["vae"],
        ),
        WeightsSource::File(file) => fused_ldm_component_footprint(file, spec.quantize),
    }
}

/// Header-only resident projection for a fused LDM/A1111 checkpoint.  Unlike a snapshot tree, a
/// fused file must first be classified by the exact shared LDM key remapper.  Text/UNet values are
/// retained at fp16 (or at the selected affine Q4/Q8 tier); VAE values are retained at f32.  Keys the
/// executable splitter ignores are omitted here too.
fn fused_ldm_component_footprint(
    file: &Path,
    quantize: Option<Quant>,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    use mlx_gen::gen_core::sdxl_ldm::{remap_sdxl_ldm_key, SdxlComponent, TensorRemap};

    let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(file)?;
    if headers.is_empty() {
        return Err(mlx_gen::gen_core::Error::Msg(
            "sdxl fused checkpoint contains no tensor headers".to_owned(),
        ));
    }
    let mut out = mlx_gen::PerComponentBytes::default();
    for tensor in headers {
        let Some(remap) = remap_sdxl_ldm_key(&tensor.name) else {
            continue;
        };
        let (component, targets, source_shape_is_pack_shape) = match remap {
            TensorRemap::Rename(component, target) => (component, vec![target], true),
            // Row-wise Q/K/V packing is additive, so packing the fused row stack has the same byte
            // total as packing the three remapped projections independently.
            TensorRemap::SplitQkv(component, targets) => {
                (component, targets.into_iter().collect(), true)
            }
            // Transpose/squeeze preserve element count but can change which axis is grouped. Keep
            // them dense in the estimate rather than risk understating the exact packed footprint.
            TensorRemap::Transpose(component, target) | TensorRemap::Squeeze(component, target) => {
                (component, vec![target], false)
            }
        };
        let bytes = match component {
            SdxlComponent::Vae => tensor.materialized_bytes(4)?,
            SdxlComponent::Unet | SdxlComponent::ClipL | SdxlComponent::ClipBigG => {
                let target_is_packable = source_shape_is_pack_shape
                    && tensor.shape.len() == 2
                    && targets.iter().all(|target| {
                        let Some(base) = target.strip_suffix(".weight") else {
                            return false;
                        };
                        component == SdxlComponent::Unet
                            || (!base.ends_with(".token_embedding")
                                && !base.ends_with(".position_embedding"))
                    });
                match (quantize, target_is_packable) {
                    (Some(quant), true) => {
                        let group = crate::quant::GROUP_SIZE as usize;
                        let [rows, columns] = tensor.shape.as_slice() else {
                            unreachable!("target_is_packable requires a weight target; shape is checked below")
                        };
                        if *columns < group || *columns % group != 0 {
                            tensor.materialized_bytes(2)?
                        } else {
                            let rows = u64::try_from(*rows).map_err(|_| {
                                mlx_gen::gen_core::Error::Msg(
                                    "sdxl fused projection row count overflow".to_owned(),
                                )
                            })?;
                            let columns = u64::try_from(*columns).map_err(|_| {
                                mlx_gen::gen_core::Error::Msg(
                                    "sdxl fused projection column count overflow".to_owned(),
                                )
                            })?;
                            let codes = rows
                                .checked_mul(columns)
                                .and_then(|elements| elements.checked_mul(quant.bits() as u64))
                                .map(|bits| bits / 8)
                                .ok_or_else(|| {
                                    mlx_gen::gen_core::Error::Msg(
                                        "sdxl fused quantized code size overflow".to_owned(),
                                    )
                                })?;
                            let tables = rows
                                .checked_mul(columns / group as u64)
                                .and_then(|entries| entries.checked_mul(4))
                                .ok_or_else(|| {
                                    mlx_gen::gen_core::Error::Msg(
                                        "sdxl fused quantization table size overflow".to_owned(),
                                    )
                                })?;
                            codes.checked_add(tables).ok_or_else(|| {
                                mlx_gen::gen_core::Error::Msg(
                                    "sdxl fused quantized resident size overflow".to_owned(),
                                )
                            })?
                        }
                    }
                    _ => tensor.materialized_bytes(2)?,
                }
            }
        };
        let slot = match component {
            SdxlComponent::ClipL | SdxlComponent::ClipBigG => &mut out.text_encoder,
            SdxlComponent::Unet => &mut out.dit,
            SdxlComponent::Vae => &mut out.vae,
        };
        *slot = slot.checked_add(bytes).ok_or_else(|| {
            mlx_gen::gen_core::Error::Msg("sdxl fused component byte sum overflow".to_owned())
        })?;
    }
    if out.text_encoder == 0 || out.dit == 0 || out.vae == 0 {
        return Err(mlx_gen::gen_core::Error::Msg(
            "sdxl fused checkpoint is missing a text encoder, UNet, or VAE component".to_owned(),
        ));
    }
    Ok(out)
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// The weights-free memory-strategy registration (SC-15525) — the shared registry conformance
/// suite's entry point into this provider's declaration.
pub const MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: |spec| {
            crate::memory_strategy::weights_free_memory_strategy_contract(MODEL_ID, spec)
        },
        safety_check: crate::memory_strategy::safety_check,
    };

/// The weights-free **behavioral** registration: valid fixtures plus a request scope, so the shared
/// suite can drive every declared rung's lifecycle without loading 6.6 GB of weights.
pub const MEMORY_BEHAVIOR_REGISTRATION: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

/// sc-16195 Apple-Silicon warm sweep: q8 and dense both peaked at 14.039 GiB at 1024².
/// The 14.05 GiB family ceiling is deliberately upward-rounded.
pub const ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: MODEL_ID,
        anchor: ActivationMemoryAnchor {
            bytes_1024: 15_086_072_628,
        },
    };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strength_resolution_preserves_defaults_explicit_values_and_clamping() {
        for (requested, default) in [
            (None, DEFAULT_STRENGTH),
            (None, INPAINT_DEFAULT_STRENGTH),
            (Some(0.0), DEFAULT_STRENGTH),
            (Some(0.37), DEFAULT_STRENGTH),
            (Some(1.0), DEFAULT_STRENGTH),
            (Some(-0.25), DEFAULT_STRENGTH),
            (Some(1.25), INPAINT_DEFAULT_STRENGTH),
        ] {
            let previous = requested.unwrap_or(default).clamp(0.0, 1.0);
            assert_eq!(resolve_strength(requested, default), previous);
        }
    }

    #[test]
    fn ancestral_strength_schedule_matches_previous_formula() {
        let steps = 30;
        let max_time = 999.0;
        for strength in [0.0, 0.01, 0.6, DEFAULT_STRENGTH, 0.85, 1.0] {
            let previous = ((steps as f32 * strength) as usize, max_time * strength);
            assert_eq!(
                ancestral_strength_schedule(steps, max_time, strength),
                previous
            );
        }
    }

    #[test]
    fn curated_strength_schedule_matches_previous_tail_selection() {
        let full_sigmas = vec![14.0, 8.0, 4.0, 2.0, 1.0, 0.0];
        let steps = 5;
        for strength in [0.0, 0.01, 0.2, 0.6, DEFAULT_STRENGTH, 1.0] {
            let effective_steps = (steps as f32 * strength) as usize;
            let run_start = full_sigmas
                .len()
                .saturating_sub(1)
                .saturating_sub(effective_steps);
            let previous = full_sigmas[run_start..].to_vec();
            assert_eq!(
                curated_strength_schedule(&full_sigmas, steps, strength),
                previous
            );
        }
    }

    #[test]
    fn descriptor_is_sdxl() {
        let d = descriptor();
        assert_eq!(d.id, "sdxl");
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_preview);
    }

    #[test]
    fn registered_in_core_registry() {
        // The family catalog must expose the model registration.
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == "sdxl"),
            "sdxl is not registered in mlx_gen's generator registry"
        );
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    /// sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised SDXL bucket to.
    /// Pin the value and mutation-check that a VAE-valid multiple of 8 which is not the full U-Net
    /// multiple is rejected with the stride error, and an on-stride size passes.
    #[test]
    fn size_multiple_is_the_pinned_stride() {
        assert_eq!(SIZE_MULTIPLE, 32);
        let caps = descriptor().capabilities;
        let off = validate_request(
            &caps,
            &GenerationRequest {
                prompt: "a fox".into(),
                width: 1000, // 125×8 — VAE-valid but not SIZE_MULTIPLE
                height: 1024,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(off.contains("multiples of 32"), "got: {off}");
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                prompt: "a fox".into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            }
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_explicit_zero_steps() {
        let caps = descriptor().capabilities;
        // F-073: an explicit `steps: Some(0)` would VAE-decode pure scaled noise → reject loudly.
        let zero = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_request(&caps, &zero).unwrap_err().to_string();
        assert!(err.contains("steps"), "got: {err}");
        // `steps: None` (use the production default) and an explicit positive count are accepted.
        let unset = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&caps, &unset).is_ok());
        let one = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(1),
            ..Default::default()
        };
        assert!(validate_request(&caps, &one).is_ok());
    }

    #[test]
    fn validate_rejects_unadvertised_guidance_method_and_true_cfg() {
        // F-022: the hand-rolled copy dropped the `guidance_method` membership + `true_cfg` gate. A
        // `cfg_pp` typo (e.g. "cfgpp") previously slipped through and silently rendered plain CFG.
        let caps = descriptor().capabilities;
        // Advertised methods are ["cfg", "cfg_pp"]; a typo must be rejected (typed Unsupported).
        let typo = GenerationRequest {
            prompt: "a fox".into(),
            guidance_method: Some("cfgpp".into()),
            ..Default::default()
        };
        let err = validate_request(&caps, &typo).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "cfg_pp typo should be a typed Unsupported gap, got {err:?}"
        );
        // The correct spelling passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            guidance_method: Some("cfg_pp".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &ok).is_ok());
        // SDXL doesn't support true_cfg — a request must be rejected, not ignored.
        let tcfg = GenerationRequest {
            prompt: "a fox".into(),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &tcfg).is_err());
    }

    #[test]
    fn validate_accepts_cfg_and_negative_prompt_rejects_bad_size() {
        let caps = descriptor().capabilities;
        // Real CFG + negative prompt are supported.
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // VAE-valid but U-Net-invalid size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1000,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // Out-of-range size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // The default + every wired sampler is accepted (an unset sampler defaults to ancestral): the
        // accel profiles AND the unified curated solvers (epic 7114, sc-7121).
        assert!(validate_request(&caps, &base).is_ok());
        for ok in [
            "euler_ancestral",
            "lcm",
            "lightning",
            "hyper",
            "euler",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        sampler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "sampler {ok:?} should be accepted"
            );
        }
        // An unknown sampler is rejected, not silently downgraded.
        for bad in ["plms", "dpm_fast", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    sampler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported sampler"), "got: {err}");
        }
    }

    #[test]
    fn validate_scheduler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // `discrete` (the native ancestral schedule) + every curated σ scheduler is accepted (sc-7121).
        for ok in [
            "discrete",
            "normal",
            "simple",
            "karras",
            "exponential",
            "sgm_uniform",
            "beta",
            "ddim_uniform",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        scheduler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "scheduler {ok:?} should be accepted"
            );
        }
        // diffusers timestep-spacing names are NOT curated scheduler names → rejected.
        for bad in ["leading", "trailing", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    scheduler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported scheduler"), "got: {err}");
        }
    }

    #[test]
    fn ordinary_load_does_not_treat_an_undeclared_file_as_a_snapshot() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(
            err.contains("ldm_tokenizer") || err.contains("snapshot directory"),
            "the file source must fail at an imported-route structural gate, not load as a snapshot: {err}"
        );
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that SDXL's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the single-file guard in
    // `resolve_root` passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the CLIP text encoders from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The `sequential_residency_real_weights.rs` A/B is
    // `#[ignore]`d; this is the default-run guard.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/sdxl-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(&missing_snapshot_spec(OffloadPolicy::Sequential))
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(&missing_snapshot_spec(OffloadPolicy::Resident))
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }

    // ── sc-18317: the fused-LDM route's rung-1 capability is EXECUTABLE, not merely declared.
    //
    // The defect these cover: `load_from_ldm_file` under `OffloadPolicy::Resident` used to obtain a
    // valid file pin, read through it once, drop it, and build `Residency::resident` — whose loaders
    // return errors and whose `rebuildable` flag is `false`. Nothing upstream refused a staged
    // selection on that instance (the descriptor publishes `Selectable`, the contract declares rung 1
    // `Implemented`, rung 1 has no prerequisite for `validate_selection` to check, and the shared
    // `Capabilities::validate_request` floor never inspects `stage_residency`), so the composition was
    // admitted everywhere and then refused inside `generate` by `Residency::ensure_rebuildable`.
    //
    // These run weight-free. A real fused SDXL checkpoint is ~7 GB, so the fixture is a structurally
    // valid safetensors carrying one SDXL-irrelevant tensor: `Weights::from_file` opens it, the shared
    // LDM key remapper classifies nothing, and `split_ldm_checkpoint` returns its own typed
    // "missing the … component" error. That error arriving is the proof the staged phase loader
    // re-opened the pinned fused file; the resident-only refusal arriving instead is the defect.
    // Each test mints its fixture root with `tempfile::tempdir()` and holds the `TempDir` guard for
    // its own duration: the guard IS the cleanup, including out of a panicking test (sc-17791).
    //
    /// One structurally valid safetensors file with a single F32 tensor and no SDXL/LDM keys.
    ///
    /// Deliberately valid rather than garbage: the point is to reach `split_ldm_checkpoint`'s key
    /// classification through the pin, not to probe the safetensors reader's malformed-input path.
    fn write_stub_fused_checkpoint(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("stub-fused.safetensors");
        let mut header =
            br#"{"not_an_sdxl_tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
        // Pad to an 8-byte boundary so the payload starts aligned, as the format expects.
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        std::fs::write(&path, bytes).expect("write stub fused checkpoint");
        path
    }

    fn stub_fused_spec(dir: &Path, policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::File(write_stub_fused_checkpoint(dir)))
            .with_offload_policy(policy)
    }

    /// The narrowest public reach into `ensure_rebuildable`: a resident-only owner refuses to evict.
    #[test]
    fn fused_ldm_resident_load_retains_its_reload_loaders() {
        let dir = tempfile::tempdir().expect("fixture root");
        let spec = stub_fused_spec(dir.path(), OffloadPolicy::Resident);
        let pin = spec
            .weights_file_pin()
            .expect("pin the stub checkpoint")
            .expect("a File source resolves to a pin");
        let residency = build_ldm_residency(&spec, &pin)
            .expect("the fused Resident arm must build a residency");
        let evicted = residency
            .evict_warm()
            .expect("a fused Resident load must retain reload loaders");
        assert!(
            !evicted,
            "the warm pair is built on first use, so there is nothing to evict yet"
        );
    }

    /// The whole point: a staged selection on a fused `Resident` instance reaches the real pinned
    /// phase loaders instead of being refused for having none.
    #[test]
    fn fused_ldm_staged_selection_reaches_the_real_pinned_phase_loaders() {
        let dir = tempfile::tempdir().expect("fixture root");
        let spec = stub_fused_spec(dir.path(), OffloadPolicy::Resident);
        let pin = spec
            .weights_file_pin()
            .expect("pin the stub checkpoint")
            .expect("a File source resolves to a pin");
        let residency = build_ldm_residency(&spec, &pin)
            .expect("the fused Resident arm must build a residency");
        let cancel = mlx_gen::CancelFlag::new();
        let err = residency
            .run_request_scoped(
                // The selection admission accepts today and used to refuse here.
                true,
                // Rung 4 is `Missing` on a fused file; `generate` always passes `false`.
                false,
                &cancel,
                false,
                &mut |_| {},
                |_text: &(ClipTextEncoder, ClipTextEncoder)| Ok::<(), Error>(()),
                |_| Ok(()),
                |_heavy: &SdxlHeavyOwned, (), _: &mut dyn FnMut(Progress)| Ok::<(), Error>(()),
            )
            .expect_err("the stub checkpoint carries no SDXL tensors, so the text phase must fail");
        let msg = err.to_string();
        assert!(
            !msg.contains("no reload loaders"),
            "a staged selection must reach the pinned phase loaders, not the resident-only \
             refusal: {msg}"
        );
        assert!(
            msg.contains("sdxl LDM checkpoint is missing the"),
            "the staged text phase must have re-split the pinned fused file: {msg}"
        );
    }

    /// The reach half of the chain: every layer between a request and `generate` accepts a staged
    /// selection on a fused `Resident` load, which is exactly why the instance has to back it.
    #[test]
    fn fused_ldm_staged_residency_is_declared_and_admitted_on_every_layer() {
        use mlx_gen::gen_core::{
            MemoryNumericTier, MemorySelection, MemoryStrategy, MemoryStrategySupport,
            StagedResidencyAvailability,
        };

        assert_eq!(
            descriptor().capabilities.staged_residency_availability(),
            StagedResidencyAvailability::Selectable,
            "the static descriptor publishes rung 1 as selectable for every SDXL route"
        );

        let dir = tempfile::tempdir().expect("fixture root");
        let spec = stub_fused_spec(dir.path(), OffloadPolicy::Resident);
        let contract =
            crate::memory_strategy::weights_free_memory_strategy_contract(descriptor().id, &spec)
                .expect("declaration-equivalent contract for a fused load");
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .map(|capability| &capability.support),
            Some(&MemoryStrategySupport::Implemented),
            "rung 1 is declared Implemented on a fused File + Resident load"
        );
        assert!(
            !crate::memory_strategy::streamable(&spec),
            "rung 4 stays Missing on a fused file — only rung 1 is at issue here"
        );
        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
            })
            .expect("admission accepts a staged selection on a fused load");
    }
}
#[test]
#[ignore = "real fused SDXL checkpoint + Metal; set SDXL_LDM_CHECKPOINT and SDXL_TOKENIZER_ROOT"]
fn fused_ldm_real_weight_render_is_nonconstant() {
    let checkpoint = std::env::var_os("SDXL_LDM_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .expect("set SDXL_LDM_CHECKPOINT");
    let tokenizer_root = std::env::var_os("SDXL_TOKENIZER_ROOT")
        .map(std::path::PathBuf::from)
        .expect("set SDXL_TOKENIZER_ROOT");
    let spec = LoadSpec::new(WeightsSource::File(checkpoint));
    let generator = load_from_ldm_file(&spec, &tokenizer_root).expect("load fused SDXL");
    let output = generator
        .generate(
            &GenerationRequest {
                prompt: "a red fox in a snowy forest, cinematic photograph".into(),
                negative_prompt: Some("blurry, low quality".into()),
                width: 512,
                height: 512,
                steps: Some(2),
                guidance: Some(5.0),
                seed: Some(14024),
                ..Default::default()
            },
            &mut |_| {},
        )
        .expect("render fused SDXL");
    let GenerationOutput::Images(images) = output else {
        panic!("SDXL returned non-image output");
    };
    let image = images.first().expect("one image");
    assert_eq!((image.width, image.height), (512, 512));
    let min = *image.pixels.iter().min().expect("pixels");
    let max = *image.pixels.iter().max().expect("pixels");
    assert!(max.saturating_sub(min) > 32, "render is nearly constant");
}

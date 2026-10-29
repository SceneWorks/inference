//! `KolorsGenerator` — the [`mlx_gen::Generator`] impl for Kolors, plus its [`descriptor`]/[`load`]
//! entry points and explicit registration under the id `"kolors"` (sc-3874).
//!
//! The epic-3090 ports (sc-3091–3098) gave [`crate::Kolors`] the full capability surface but only as
//! a direct struct API (which the parity tests call). This module makes Kolors **dispatchable** —
//! the SceneWorks worker's in-process registry entry — by
//! mapping [`LoadSpec`]/[`GenerationRequest`] onto that API and looping `req.count` with per-image
//! seeds + cancel + streamed progress, mirroring `mlx-gen-sdxl/src/model.rs`.
//!
//! **Registration mechanism:** the named constant below is composed by the family registry, which
//! is in turn composed by the MLX platform catalog.

use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use std::path::Path;

use mlx_gen::gen_core::sampling::{vp_capture_plan, VpCapturePlan};
use mlx_gen::{
    curated_scheduler_names, default_seed, schedule_sigmas, AlphaSchedule, Capabilities,
    Conditioning, ConditioningKind, ControlKind, DiscreteModelSampling, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LatentDecoder, LoadSpec, Modality, ModelDescriptor,
    OffloadPolicy, Progress, Quant, Residency, Result, Scheduler, Solver, StepSupport,
    WeightsSource,
};

use mlx_gen_pid::{resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_sdxl::{
    encode_init_latents, load_controlnet, ControlNet, IpImageEncoder, SdxlBlockWindow,
    SdxlForwardPlan, PID_BACKBONE,
};

use crate::ip_adapter::load_kolors_ip_adapter;
use crate::model::{KolorsHeavy, KolorsText, DEFAULT_IMG2IMG_STRENGTH, SPATIAL_SCALE};
use crate::sampler::{KolorsEulerSampler, BETA_END, BETA_START, NUM_TRAIN_TIMESTEPS};

type KolorsEncodedConditioning = ((Array, Array), Option<(Array, Array)>);

/// Registry id — the SceneWorks worker's `payload.model` for the Kolors family.
pub const MODEL_ID: &str = "kolors";

/// diffusers `KolorsPipeline` production defaults: 50 inference steps, CFG 5.0.
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 5.0;
/// Default IP-Adapter scale when a request doesn't override it (carried on the `Reference` strength
/// field in IP mode, mirroring the SDXL IP-Adapter convention).
const IP_DEFAULT_SCALE: f32 = 0.6;
/// Default ControlNet `conditioning_scale` for a `Conditioning::Control` that leaves `scale = None`
/// (F-085) — the diffusers full-strength default. An explicit `Some(x)`, including `Some(0.0)` for an
/// inert branch, overrides it.
const DEFAULT_CONTROLNET_SCALE: f32 = 1.0;
/// Default img2img init strength for the combined strict-pose tier (sc-5012) when `req.strength` is
/// unset — the torch `_run_pose` default 1.0 (at full strength the init only seeds latent
/// dimensions; identity comes from the IP-Adapter, structure from the ControlNet).
const POSE_IMG2IMG_STRENGTH: f32 = 1.0;
/// The single Kolors sampler — diffusers `EulerDiscreteScheduler` (leading), see [`KolorsEulerSampler`].
const SAMPLER: &str = "euler_discrete";
/// Kolors' `/8` VAE feeds an SDXL U-Net with two exact downsample/upsample skip joins, so both image
/// dims must be multiples of **32**. Exposed as the pinned-engine stride SceneWorks ties each
/// advertised Kolors image bucket to (sc-12612); the model and registry share the same constant.
pub const SIZE_MULTIPLE: u32 = crate::model::PRODUCTION_SPATIAL_MULTIPLE as u32;

/// Kolors' identity + capabilities — constructible without loading weights (registry
/// introspection). Advertises **only** the wired + parity-proven surface (the false-capability
/// guard): T2I + img2img (`Reference`) + ControlNet-pose (`Control`) + IP-Adapter (`Reference` in
/// IP mode) + Q8/Q4 + **LoRA/LoKr** (sc-4733 — merged into the SDXL-family U-Net at load via
/// [`crate::model::Kolors::apply_lora`], the inference complement to the Kolors trainer sc-4568).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::SDXL_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "kolors",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Kolors uses real classifier-free guidance over the ChatGLM3 conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            // Reference = img2img init (sc-3095) OR the IP-Adapter image prompt when an IP-Adapter is
            // loaded (sc-3098); Control = the Kolors ControlNet-pose branch (sc-3097).
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Control],
            // LoRA/LoKr merge into the SDXL-family U-Net at load (sc-4733).
            supports_lora: true,
            supports_lokr: true,
            // `euler_discrete` is the native leading-Euler default; the rest are the unified curated
            // solvers (epic 7114, sc-7121) — the additive k-diffusion path over `DiscreteModelSampling`.
            // Selecting one (or a non-`discrete` scheduler) routes to `Kolors::denoise_curated_latents`,
            // which now covers EVERY mode incl. the conditioned sub-providers (ControlNet-pose,
            // IP-Adapter, the combined pose tier — sc-7297, via `denoise_curated`'s control/ip support);
            // the native default stays byte-exact.
            samplers: {
                let mut s = vec![SAMPLER];
                s.extend([
                    "euler",
                    "euler_ancestral",
                    "heun",
                    "dpmpp_2m",
                    "dpmpp_sde",
                    "uni_pc",
                    "lcm",
                    "ddim",
                ]);
                s
            },
            // `discrete` is the native schedule; the rest are the curated σ schedulers (epic 7114).
            schedulers: {
                let mut s = vec!["discrete"];
                s.extend(curated_scheduler_names());
                s
            },
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            // Kolors' train-timestep count is a real upper bound: above it `step_ratio` floors to
            // 0 and every timestep collapses to 1 — silent garbage, not an error (F-124). It was
            // checked only inside `validate_request` below; declaring it makes it discoverable
            // (sc-19559).
            supported_steps: StepSupport::Range {
                min: 1,
                max: NUM_TRAIN_TIMESTEPS as u32,
            },
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Wired onto the shared `Residency` seam (epic 10834, sc-10840); honors Sequential offload
            // (F-176). The monolithic `Kolors` was split into a droppable `KolorsText` (6B ChatGLM3 +
            // tokenizer) phase and a `KolorsHeavy` (SDXL U-Net + VAE) phase (`crate::model`): under
            // `Sequential` the ChatGLM3 encoder — the dominant footprint — is encoded, materialized,
            // then dropped before the U-Net + VAE (+ ControlNet / IP-Adapter / PiD) load, bounding
            // peak unified memory to `max(ChatGLM3, U-Net+VAE)`. Kolors quantizes the U-Net + ChatGLM3
            // DENSE at load, so a `Sequential` + `quantize` load re-quantizes each generate (F-181
            // advisory in `load`).
            supports_sequential_offload: true,
            supports_preview: true,
            // sc-18317: Kolors' guidance is real classifier-free guidance, and the one convention it
            // implements is the doubled `[cond, uncond]` batch every assembly in `model.rs` builds.
            // Declaring the single mode — rather than leaving the domain `Unsupported` — is what makes
            // a planner able to select it explicitly and makes `sequential` a refusal by name instead
            // of a silently batched run.
            execution: mlx_gen::gen_core::ExecutionSurface {
                cfg_batching: mlx_gen::gen_core::CfgBatchingDomain::Modes(
                    crate::model::CFG_BATCHING_MODES.to_vec(),
                ),
                ..mlx_gen::gen_core::ExecutionSurface::default()
            },
            ..Default::default()
        },
    }
}

/// A loaded, dispatchable Kolors generator: the component-residency strategy (epic 10834, sc-10840)
/// plus the load-time presence flags the validator reads. Holds ONLY the [`Residency`] (no direct
/// component fields — a retained component would defeat the `Sequential` drop): `Resident` (default)
/// holds the ChatGLM3 encoder + U-Net + VAE (+ ControlNet / IP-Adapter / PiD) warm; `Sequential` holds
/// only the per-phase loader closures and re-loads each per generation in phase order (encode →
/// **drop the ChatGLM3 encoder** → U-Net/VAE/ControlNet/IP/PiD), bounding peak to
/// `max(ChatGLM3, U-Net+VAE)`.
pub struct KolorsGenerator {
    descriptor: ModelDescriptor,
    /// Whether a ControlNet was requested (`spec.control`). Known at load without loading weights, so
    /// the validator's mode-combination guards work even under `Sequential` (the ControlNet itself is
    /// loaded lazily in the heavy phase).
    has_control: bool,
    /// Whether an IP-Adapter was requested (`spec.ip_adapter` is a dir). Same rationale as
    /// [`has_control`](Self::has_control).
    has_ip: bool,
    /// The load-time `OffloadPolicy` verdict, kept as the DEFAULT for a request that names no
    /// [`GenerationMemory::stage_residency`](mlx_gen::gen_core::GenerationMemory). Rung 1 is
    /// request-scoped from SC-15521 onward; the policy is no longer the authority.
    default_stage_residency: bool,
    /// Whether THIS load can execute ladder rung 4 — see
    /// [`memory_strategy::streamable`](crate::memory_strategy::streamable).
    streamable: bool,
    loaded_spec: LoadSpec,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    residency: Residency<KolorsText, KolorsHeavyOwned>,
}

/// The heavy render-phase components owned by a `Resident` build or a `Sequential` generate (mirrors
/// lens's `LensHeavyOwned`): the SDXL U-Net + VAE ([`KolorsHeavy`]) plus the optionally-loaded
/// ControlNet branch, IP-Adapter image-token encoder (its decoupled-attn K/V pairs already installed
/// into the U-Net), and PiD overlay.
pub(crate) struct KolorsHeavyOwned {
    heavy: KolorsHeavy,
    control: Option<ControlNet>,
    ip_encoder: Option<IpImageEncoder>,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7848): loaded when the spec carries
    /// [`LoadSpec::pid`] AND this generate uses it (F-177). Kolors shares the SDXL VAE latent space, so
    /// it reuses the `sdxl` PiD student (via [`PID_BACKBONE`]). `Some` ⇒ a `req.use_pid` generation
    /// decodes the final latent 4× through PiD instead of the VAE; `None` ⇒ the byte-exact VAE decode.
    pid: Option<PidEngine>,
}

/// Build a [`KolorsGenerator`] from a [`LoadSpec`].
///
/// `spec.weights` is a `Kwai-Kolors/Kolors-diffusers` snapshot dir (the multi-component tree with
/// the materialized `tokenizer/tokenizer.json`). Dense runs **fp16** (the SDXL-family production
/// dtype; the VAE stays f32 via `load_vae`). `spec.quantize` ⇒ load-time Q8/Q4 (sc-3096);
/// `spec.control` ⇒ the Kolors ControlNet-Pose checkpoint (sc-3097); `spec.ip_adapter` ⇒ the
/// Kolors-IP-Adapter-Plus snapshot dir (sc-3098), whose K/V pairs are installed into the (pre-quant)
/// U-Net. `spec.adapters` (LoRA/LoKr) ⇒ merged into the dense U-Net before quantization (sc-4733).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // fp16 dense path (SDXL-family production dtype). `Precision::Bf16` is the registry's
    // "dense default / no override" sentinel here — NOT a literal bf16 request — mapped to fp16
    // for this SDXL-family loader (see the `Precision` enum note). A precision override is not
    // wired (the VAE is always f32, the rest fp16), so reject it rather than silently ignore.
    if spec.precision != mlx_gen::Precision::Bf16 {
        return Err(Error::Msg(
            "kolors: precision override is not wired; the dense path runs fp16 (SDXL-family \
             production dtype) — drop the precision override"
                .into(),
        ));
    }
    // Fail-fast the single-file source (and a File ip_adapter) up front for BOTH policies. Sequential
    // defers the component build, but these are still wrong regardless of policy.
    resolve_root(spec)?;
    if matches!(&spec.ip_adapter, Some(WeightsSource::File(_))) {
        return Err(Error::Msg(
            "kolors ip_adapter expects a Kolors-IP-Adapter-Plus snapshot directory, not a file"
                .into(),
        ));
    }
    // F-181: a `Sequential` + `quantize` load over a **dense** snapshot re-quantizes each generate
    // (repeated compute; the dense transient shrinks the memory win).
    //
    // SC-15521 narrowed the condition. The advisory used to fire on every staged quantized load,
    // including the three shipped `SceneWorks/kolors-mlx` tiers whose weights are already packed —
    // where nothing re-quantizes and the message is simply wrong. Staged residency is the ladder's
    // rung 1 and therefore now the *normal* path, so an advisory that cried wolf on every default
    // request would have trained readers to ignore it. These are the same predicates rung 4 gates
    // on, so the warning and the rung cannot disagree about which loads pack at load time.
    //
    // **Both components are checked, not just the U-Net.** `load_leaves_blocks_lazy` inspects
    // `unet/` alone, and a Sequential load re-runs the ENTIRE staged schedule per generate — encoder
    // included. On a mixed snapshot (packed `unet/`, dense `text_encoder/`) a U-Net-only gate
    // suppressed the advisory while the 6B ChatGLM3 tower genuinely re-quantized on every request,
    // which is by far the larger of the two costs on this family.
    if let Some(q) = spec.quantize {
        let repacks_each_generate = !crate::memory_strategy::load_leaves_blocks_lazy(spec)
            || !crate::memory_strategy::text_encoder_leaves_blocks_lazy(spec);
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) && repacks_each_generate {
            mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
        }
    }
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec)?;
    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        has_control: spec.control.is_some(),
        has_ip: matches!(&spec.ip_adapter, Some(WeightsSource::Dir(_))),
        default_stage_residency: crate::memory_strategy::default_stage_residency(spec),
        streamable: crate::memory_strategy::streamable(spec),
        loaded_spec: spec.clone(),
        memory_strategy,
        residency: build_residency(spec)?,
    }))
}

/// Resolve the snapshot dir (rejecting a single-file source). Shared by the entry-point fail-fast and
/// the `Sequential` per-phase loaders.
fn resolve_root(spec: &LoadSpec) -> Result<std::path::PathBuf> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p.clone()),
        WeightsSource::File(_) => Err(Error::Msg(
            "kolors expects a Kolors-diffusers snapshot directory (text_encoder/ tokenizer/ \
             unet/ vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// The policy→[`Residency`] dispatch (sc-10840), routed through the single [`Residency::from_policy`]
/// seam (F-180). `Resident` eager-loads the ChatGLM3 text phase + heavy bundle now (the heavy loader
/// with `use_pid = true`, loading any PiD overlay once and reusing it); `Sequential` captures the two
/// per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both use the same
/// [`KolorsText::load`] / [`load_heavy_owned`], so the `Resident` composition is byte-identical to the
/// pre-seam whole-model load (independent snapshot subdirs, deterministic RNG-free LoRA-merge → quant
/// → IP-install). Weight-free-testable: under `Sequential` this touches no component weights.
pub(crate) fn build_residency(spec: &LoadSpec) -> Result<Residency<KolorsText, KolorsHeavyOwned>> {
    let dtype = Dtype::Float16;
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let root = resolve_root(&spec_text)?;
            let mut text = KolorsText::load_streamed(
                &root,
                dtype,
                crate::memory_strategy::streamable(&spec_text),
            )?;
            // Q4/Q8 quantizes the 6B ChatGLM3 encoder in place after the dense load (the U-Net
            // quantizes in `load_heavy_owned`). Deterministic, so byte-identical across residencies.
            if let Some(q) = spec_text.quantize {
                text.quantize(q.bits())?;
            }
            // Ladder rung 4's `TextEncoder` scope (SC-15521) — armed after the quantize, so the
            // recorded tier is the one the resident blocks carry. Nothing is installed on a
            // `GlmBlock` at load, so unlike the U-Net there is nothing to capture; see
            // `crate::block_stream`.
            if crate::memory_strategy::streamable(&spec_text) {
                text.arm_block_stream(&root, spec_text.quantize.map(|q| q.bits()))?;
            }
            Ok(text)
        },
        move |use_pid| {
            let root = resolve_root(&spec_heavy)?;
            load_heavy_owned(&spec_heavy, &root, dtype, use_pid)
        },
    )
}

/// Load the heavy render bundle — the SDXL U-Net + VAE (+ optional ControlNet / IP-Adapter / PiD),
/// everything but the ChatGLM3 encoder. Merges any LoRA/LoKr into the dense U-Net, then quantizes it
/// (the SDXL ordering, sc-4733), then loads the ControlNet + installs the IP-Adapter K/V pairs into
/// the (possibly quantized) U-Net. The PiD student is loaded only when `use_pid` (F-177) — Resident
/// passes `true` (loaded once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips it.
fn load_heavy_owned(
    spec: &LoadSpec,
    root: &Path,
    dtype: Dtype,
    use_pid: bool,
) -> Result<KolorsHeavyOwned> {
    let mut heavy =
        KolorsHeavy::load_streamed(root, dtype, crate::memory_strategy::streamable(spec))?;
    if !spec.adapters.is_empty() {
        heavy.apply_lora(&spec.adapters)?;
    }
    if let Some(q) = spec.quantize {
        // F-144 (sc-11129): reject a requested-vs-packed tier mismatch before quantizing. `quantize()`
        // silently no-ops on already-packed weights, so a Q4 request over a pre-quantized Q8 snapshot
        // would serve Q8 with no diagnostic. Reuses the SDXL-family marker check — the Kolors U-Net is
        // the SDXL `UNet2DConditionModel` under `unet/`, the representative heavy component.
        mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), MODEL_ID)?;
        heavy.quantize_unet(q.bits())?;
    }

    let control = match &spec.control {
        Some(src) => Some(load_controlnet(src, dtype)?),
        None => None,
    };

    let ip_encoder = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => {
            let (enc, pairs) = load_kolors_ip_adapter(p, dtype)?;
            heavy.install_ip_adapter(pairs)?;
            Some(enc)
        }
        // A File ip_adapter was rejected up front in `load`.
        Some(WeightsSource::File(_)) => None,
        None => None,
    };

    // Ladder rung 4 (SC-15521) — armed LAST, and that ordering is the whole correctness argument.
    // Each `Transformer2D`'s stream captures the installed IP-Adapter K/V projections and the
    // forward-time residual adapters off its FINISHED resident blocks, so the streamed and resident
    // paths cannot disagree about which state landed where. Arming before the LoRA merge, the
    // quantize or the IP install would capture an empty state and silently render without the image
    // prompt (`mlx_gen_sdxl::block_stream` fails loudly on the IP half rather than allowing that).
    //
    // `memory_strategy::streamable` is the single authority for whether this load may arm at all: it
    // refuses an eager load shape, a load-time quantization over a dense snapshot that materializes
    // the trunk, and an adapter load that merged its delta into weights the snapshot does not carry.
    if crate::memory_strategy::streamable(spec) {
        heavy.arm_block_streams(root, spec.quantize.map(|q| q.bits()))?;
    }

    // PiD decoder overlay (epic 7840, sc-7848): load the `sdxl` student + Gemma caption encoder once
    // when the spec carries it AND this generate uses it (Kolors = SDXL VAE latent space).
    let pid = if use_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };

    Ok(KolorsHeavyOwned {
        heavy,
        control,
        ip_encoder,
        pid,
    })
}

// Written out rather than `mlx_gen::impl_generator!` because Kolors now answers the memory-strategy
// hooks (SC-15521); the macro covers only `descriptor`/`validate`/`generate`.
impl Generator for KolorsGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
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

impl KolorsGenerator {
    /// The rich-`Result` body behind [`Generator::validate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)?;
        // Mode-combination guards. The Kolors conditioning paths are mutually exclusive EXCEPT the
        // combined strict-pose tier (sc-5012): Control (the pose skeleton) + a Reference (the
        // IP-Adapter identity, which also seeds the img2img init), which is supported when BOTH a
        // ControlNet and an IP-Adapter are loaded.
        let has_ref = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let has_ctrl = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }));
        if has_ctrl && !self.has_control {
            return Err(Error::Msg(
                "kolors: a Control conditioning was passed but the model was loaded without a \
                 ControlNet (set LoadSpec::control)"
                    .into(),
            ));
        }
        // Control + Reference is the combined pose tier — allowed ONLY when an IP-Adapter is also
        // loaded (the Reference is the IP identity + img2img init). Plain Control + img2img (a
        // Reference with no IP-Adapter) is not a wired Kolors path.
        if has_ctrl && has_ref && !self.has_ip {
            return Err(Error::Msg(
                "kolors: combining ControlNet (Control) with a Reference requires an IP-Adapter (the \
                 combined pose tier — load LoadSpec::ip_adapter); plain Control + img2img is not \
                 supported in this build"
                    .into(),
            ));
        }
        // A loaded IP-Adapter + Control with no Reference can't run: the combined pass needs the
        // reference as the IP identity (and the IP image prompt is required in IP mode anyway).
        if has_ctrl && self.has_ip && !has_ref {
            return Err(Error::Msg(
                "kolors: the combined ControlNet + IP-Adapter pass requires a Reference image (the \
                 IP-Adapter identity)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    /// The staged residency lifecycle (ChatGLM3 encode pos+neg → **drop the encoder** under
    /// `Sequential` → load the U-Net/VAE/ControlNet/IP/PiD → per-mode denoise/decode → free the heavy
    /// bundle) is driven by the shared [`Residency::run`] seam (sc-10840). The per-mode denoise
    /// dispatch (curated + the five bespoke assemblies) + the PiD `from_ldm` plan run inside the render
    /// closure, byte-identical for both policies (the only global-RNG draw per image is the noise,
    /// after `random::seed(seed)` — hoisting the seed-independent encode/VAE-init out cannot perturb it).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let request = self.resolve_request(req)?;

        self.residency.run_request_scoped(
            request.stage_residency,
            false,
            &req.cancel,
            req.use_pid,
            on_progress,
            // ── Phase A: encode the positive (+ negative when CFG is on) ChatGLM3 conditioning.
            // Seed-independent; under `Sequential` the shared seam materializes these + DROPS the 6B
            // ChatGLM3 encoder before the U-Net/VAE load — bounding peak to `max(ChatGLM3, U-Net+VAE)`.
            // The negative encode is skipped when guidance is off (F-005, sc-9091): the per-mode
            // assemblies build B=1 conditioning for `cfg <= 1.0` and never read the uncond stream.
            // Ladder rung 4's `TextEncoder` scope (SC-15521) lands HERE, not in the denoise: the
            // ChatGLM3-6B tower is the largest component in the model, and at the one advertised
            // cell where the conditioning phase carries the request peak (`bf16` at `min_size`) a
            // window over these 28 blocks is what moves it. `encode_windowed` is bit-identical to
            // `encode` — same constructor, same replayed tier — so the negative encode is windowed
            // too rather than left resident beside a windowed positive.
            |text: &KolorsText| {
                let encode = |prompt: &str| match request.text_window {
                    Some(size) => text.encode_windowed(prompt, size, &req.cancel),
                    None => text.encode(prompt),
                };
                let pos = encode(&req.prompt)?;
                let neg = if request.cfg > 1.0 {
                    Some(encode(request.negative)?)
                } else {
                    None
                };
                Ok((pos, neg))
            },
            // Materialize the (context, pooled) tuples while the encoder is still alive (Sequential
            // only) — MLX is lazy, so un-evaluated outputs keep the 6B encoder referenced and the drop
            // would free nothing.
            |encoded: Option<&KolorsEncodedConditioning>| {
                let Some((pos, neg)) = encoded else {
                    return Ok(());
                };
                let mut arrays = vec![&pos.0, &pos.1];
                if let Some(n) = neg {
                    arrays.push(&n.0);
                    arrays.push(&n.1);
                }
                eval(arrays)?;
                Ok(())
            },
            // ── Phase B: the IP-token/VAE-init/PiD-plan setup + the per-image denoise/decode count
            // loop. Identical body for both residencies.
            |heavy_owned: &KolorsHeavyOwned, (pos, neg), on_progress: &mut dyn FnMut(Progress)| {
                self.render_phase(&request, heavy_owned, &pos, neg.as_ref(), on_progress)
            },
        )
    }

    /// Everything the request resolves before the residency lifecycle starts: the defaults, the
    /// typed mode, the sampling path and the memory-ladder rungs.
    fn resolve_request<'r>(&self, req: &'r GenerationRequest) -> Result<KolorsRequest<'r>> {
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let cfg = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);

        let reference = self.resolve_reference(req)?;
        let control = self.resolve_control(req)?;
        let mode = KolorsMode::resolve(self.has_ip, reference, control, req.strength)?;

        // ── Ladder request resolution (SC-15521) ────────────────────────────────────────────────
        // Rung 1 is request-scoped from here on: the load-time `OffloadPolicy` is only the default a
        // request that names nothing keeps.
        let stage_residency =
            crate::memory_strategy::stage_residency(req, self.default_stage_residency);
        let window = crate::memory_strategy::transformer_window(req)?;
        let dit_window = window.and_then(|w| w.dit());
        let text_window = window.and_then(|w| w.text_encoder());
        // Two fail-closed guards a calibration harness driving `generate` with a hand-built
        // `GenerationMemory` must still cross — it never went through `safety_check`.
        if window.is_some() && !self.streamable {
            return Err(Error::Unsupported(
                "kolors: bounded transformer residency needs a DeferredMaterialization load over a \
                 snapshot directory whose U-Net AND ChatGLM3 blocks stay lazy (a pre-quantized tier, or dense with no \
                 --quantize) and whose adapters (if any) are replayable; this generator cannot \
                 stream its blocks"
                    .into(),
            ));
        }
        if window.is_some() && !stage_residency {
            return Err(Error::Unsupported(
                "kolors: bounded transformer residency requires staged residency engaged in the \
                 same request — without the phase release the 6B ChatGLM3 encoder stays resident \
                 through the denoise and the request peak does not move"
                    .into(),
            ));
        }
        let decode_tiling =
            crate::memory_strategy::decode_tiling_for_contract(req, &self.memory_strategy)?;
        let forward_plan =
            SdxlForwardPlan::with_attention(crate::memory_strategy::attention_plan(req)?)
                .with_window(dit_window.map(|size| SdxlBlockWindow {
                    size,
                    cancel: &req.cancel,
                }));
        // sc-18317: the request's typed CFG batching selection, resolved once for every mode so the
        // six assemblies cannot disagree. An unset selection resolves to `Batched`, this family's own
        // pre-story convention, which is what keeps every existing render byte-for-byte identical;
        // any other mode was already refused by `Capabilities::validate_request` against the
        // descriptor's declared `execution` surface before we got here.
        let cfg_batching = crate::model::resolve_cfg_batching(req);

        Ok(KolorsRequest {
            req,
            steps,
            cfg,
            negative,
            base_seed,
            mode,
            use_curated: uses_curated_path(req),
            stage_residency,
            text_window,
            decode_tiling,
            forward_plan,
            cfg_batching,
        })
    }

    /// Phase B: bind the mode to the loaded ControlNet / IP encoder, encode the seed-independent
    /// init, resolve the PiD `from_ldm` plan, then denoise + decode each output in turn.
    fn render_phase(
        &self,
        request: &KolorsRequest<'_>,
        heavy_owned: &KolorsHeavyOwned,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let req = request.req;
        let heavy = &heavy_owned.heavy;
        let mode = prepare_conditioning(request.mode, heavy_owned)?;
        let init = build_init_latents(request, heavy, &mode)?;
        let vp_plan = resolve_vp_plan(request, heavy, &mode)?;
        let capture_sigma = vp_plan.map(|p| p.sigma).unwrap_or(0.0);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            heavy_owned.pid.as_ref(),
            req,
            request.base_seed,
            self.descriptor.id,
            capture_sigma,
        )?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
        let plan = KolorsRenderPlan {
            request,
            heavy,
            mode: &mode,
            init: &init,
            pos,
            neg,
            vp_plan,
        };

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = request.base_seed.wrapping_add(i as u64);
            random::seed(seed)?;
            mlx_gen_sdxl::stages::run_output(&plan, seed, on_progress, |latents, on_progress| {
                images.push(mlx_gen_sdxl::stages::decode_output(
                    heavy.vae(),
                    &latents,
                    pid_ref,
                    request.decode_tiling.as_ref(),
                    &req.cancel,
                    on_progress,
                )?);
                Ok(())
            })?;
        }
        Ok(GenerationOutput::Images(images))
    }

    /// The single img2img / IP reference image + its strength (the per-reference strength wins). One
    /// reference only; more than one is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "kolors: multiple reference images are not supported".into(),
                    ));
                }
                reference = Some((image, *strength));
            }
        }
        Ok(reference)
    }

    /// The single ControlNet control image + `conditioning_scale`. One control branch only; the
    /// Kolors ControlNet is pose-trained, so a non-pose `ControlKind` is rejected.
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Option<(&'a Image, f32)>> {
        let mut control = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, kind, scale } = c {
                if control.is_some() {
                    return Err(Error::Msg(
                        "kolors: multiple control images are not supported".into(),
                    ));
                }
                if !matches!(kind, ControlKind::Pose) {
                    return Err(Error::Msg(format!(
                        "kolors: only Pose ControlNet is wired (got {kind:?})"
                    )));
                }
                // `None` → the full-strength default; `Some(x)` — incl. `Some(0.0)` — verbatim (F-085).
                control = Some((image, scale.unwrap_or(DEFAULT_CONTROLNET_SCALE)));
            }
        }
        Ok(control)
    }
}

/// Curated unified-sampler path (epic 7114, sc-7121 + sc-7297): a curated solver name (≠ the native
/// `euler_discrete`) OR a non-`discrete` scheduler routes through the additive k-diffusion
/// `KolorsHeavy::denoise_curated_latents`. This covers EVERY mode — txt2img / img2img AND the
/// conditioned sub-providers (ControlNet-pose, IP-Adapter, the combined pose tier sc-5012). The
/// native `euler_discrete` default stays byte-exact — the legacy `denoise_*_latents` assemblies are
/// entered only when no curated knob is set (N1). Resolved from `req` (no weights).
fn uses_curated_path(req: &GenerationRequest) -> bool {
    let scheduler_curated = req
        .scheduler
        .as_deref()
        .and_then(Scheduler::from_name)
        .is_some();
    let sampler_curated = req
        .sampler
        .as_deref()
        .map(|s| Solver::from_name(s).is_some() && s != SAMPLER)
        .unwrap_or(false);
    scheduler_curated || sampler_curated
}

/// The request-level Kolors plan resolved before the residency lifecycle (sc-27336).
struct KolorsRequest<'r> {
    req: &'r GenerationRequest,
    steps: usize,
    cfg: f32,
    negative: &'r str,
    base_seed: u64,
    mode: KolorsMode<'r>,
    use_curated: bool,
    stage_residency: bool,
    text_window: Option<usize>,
    decode_tiling: Option<mlx_gen::tiling::TilingConfig>,
    forward_plan: SdxlForwardPlan<'r>,
    cfg_batching: mlx_gen::gen_core::CfgBatching,
}

/// The role the request's Reference / Control inputs play (sc-27336). Resolved once from the
/// request (`Net = ()`, `Tokens = ()`), then bound to the loaded ControlNet branch and the encoded
/// IP image tokens by [`prepare_conditioning`].
///
/// The mode matrix deliberately differs between the two denoise paths, and the enum encodes both
/// rather than unifying them: [`Self::Control`] carries the non-IP Reference (`img2img`) the
/// request may pair with it. `validate_impl` refuses that combination, but were it to reach the
/// denoise, the curated path img2img-initialises from it while the legacy `euler_discrete`
/// assemblies ignore it (only the legacy PiD capture schedule reads its strength).
#[derive(Clone, Copy, Debug)]
pub(crate) enum KolorsMode<'r, Net = (), Tokens = ()> {
    /// No Reference, no Control.
    Txt2Img,
    /// The Reference is the img2img init (no IP-Adapter loaded).
    Img2Img { image: &'r Image, strength: f32 },
    /// The Reference is the IP-Adapter image prompt (sc-3098).
    Ip {
        image: &'r Image,
        scale: f32,
        tokens: Tokens,
    },
    /// ControlNet-pose (sc-3097), plus any non-IP Reference (see the type docs).
    Control {
        net: Net,
        image: &'r Image,
        scale: f32,
        img2img: Option<(&'r Image, f32)>,
    },
    /// The combined strict-pose tier (sc-5012): pose ControlNet + IP-Adapter identity, on an
    /// img2img init from the SAME reference (the IP image) at `init_strength`.
    PoseIp {
        net: Net,
        skeleton: &'r Image,
        control_scale: f32,
        reference: &'r Image,
        ip_scale: f32,
        init_strength: f32,
        tokens: Tokens,
    },
}

impl<'r> KolorsMode<'r> {
    /// Resolve the request's mode. `ip_loaded` is whether an IP-Adapter was loaded: then the
    /// Reference is the image prompt (and required), otherwise it is the img2img init.
    pub(crate) fn resolve(
        ip_loaded: bool,
        reference: Option<(&'r Image, Option<f32>)>,
        control: Option<(&'r Image, f32)>,
        req_strength: Option<f32>,
    ) -> Result<Self> {
        let img2img = |(image, strength): (&'r Image, Option<f32>)| {
            (
                image,
                strength
                    .or(req_strength)
                    .unwrap_or(DEFAULT_IMG2IMG_STRENGTH),
            )
        };
        Ok(match (ip_loaded, reference, control) {
            (true, None, _) => {
                return Err(Error::Msg(
                    "kolors: an IP-Adapter is loaded but no Reference image was provided (the \
                     Reference is the image prompt in IP mode)"
                        .into(),
                ))
            }
            (true, Some((reference, strength)), Some((skeleton, control_scale))) => Self::PoseIp {
                net: (),
                skeleton,
                control_scale,
                reference,
                ip_scale: strength.unwrap_or(IP_DEFAULT_SCALE),
                init_strength: req_strength.unwrap_or(POSE_IMG2IMG_STRENGTH),
                tokens: (),
            },
            (true, Some((image, strength)), None) => Self::Ip {
                image,
                scale: strength.unwrap_or(IP_DEFAULT_SCALE),
                tokens: (),
            },
            (false, reference, Some((image, scale))) => Self::Control {
                net: (),
                image,
                scale,
                img2img: reference.map(img2img),
            },
            (false, Some(reference), None) => {
                let (image, strength) = img2img(reference);
                Self::Img2Img { image, strength }
            }
            (false, None, None) => Self::Txt2Img,
        })
    }
}

impl<'r, Net, Tokens> KolorsMode<'r, Net, Tokens> {
    /// The curated path's init image + strength: the pose tier seeds from the reference, img2img
    /// (and a Control's non-IP Reference) from its own image; everything else starts from noise.
    fn curated_init(&self) -> Option<(&'r Image, f32)> {
        match *self {
            Self::PoseIp {
                reference,
                init_strength,
                ..
            } => Some((reference, init_strength)),
            Self::Img2Img { image, strength }
            | Self::Control {
                img2img: Some((image, strength)),
                ..
            } => Some((image, strength)),
            Self::Txt2Img | Self::Ip { .. } | Self::Control { img2img: None, .. } => None,
        }
    }

    /// The legacy assemblies' init image: the pose tier's reference (== the IP image) or plain
    /// img2img's. A Control's non-IP Reference is NOT an init on this path.
    fn legacy_init_image(&self) -> Option<&'r Image> {
        match *self {
            Self::PoseIp { reference, .. } => Some(reference),
            Self::Img2Img { image, .. } => Some(image),
            Self::Txt2Img | Self::Ip { .. } | Self::Control { .. } => None,
        }
    }
}

/// The mode bound to the loaded ControlNet branch and the encoded IP image tokens.
type KolorsBoundMode<'r, 'h> = KolorsMode<'r, &'h ControlNet, Array>;

/// Stage 1 (sc-27336 `prepare_conditioning`): pair the Control conditioning with the loaded
/// ControlNet (a typed error, not an `expect`), then encode the IP image tokens. The tokens are
/// seed-independent — encoded once (the `denoise_ip_latents` methods CFG-batch them per image).
fn prepare_conditioning<'r, 'h>(
    mode: KolorsMode<'r>,
    heavy_owned: &'h KolorsHeavyOwned,
) -> Result<KolorsBoundMode<'r, 'h>> {
    let net = || {
        heavy_owned.control.as_ref().ok_or_else(|| {
            Error::Msg(
                "kolors: a Control conditioning needs the ControlNet loaded (`spec.control`)"
                    .into(),
            )
        })
    };
    let tokens = |image: &Image| match heavy_owned.ip_encoder.as_ref() {
        Some(encoder) => encoder.tokens(image),
        None => Err(Error::Msg(
            "kolors: IP mode needs the IP-Adapter image encoder loaded".into(),
        )),
    };
    Ok(match mode {
        KolorsMode::Txt2Img => KolorsMode::Txt2Img,
        KolorsMode::Img2Img { image, strength } => KolorsMode::Img2Img { image, strength },
        KolorsMode::Ip { image, scale, .. } => KolorsMode::Ip {
            image,
            scale,
            tokens: tokens(image)?,
        },
        KolorsMode::Control {
            image,
            scale,
            img2img,
            ..
        } => KolorsMode::Control {
            net: net()?,
            image,
            scale,
            img2img,
        },
        KolorsMode::PoseIp {
            skeleton,
            control_scale,
            reference,
            ip_scale,
            init_strength,
            ..
        } => {
            let net = net()?;
            KolorsMode::PoseIp {
                net,
                skeleton,
                control_scale,
                reference,
                ip_scale,
                init_strength,
                tokens: tokens(reference)?,
            }
        }
    })
}

/// The seed-independent init latents (F-083: the VAE encode draws no RNG, so it is hoisted above
/// the count loop), per denoise path.
enum KolorsInit {
    /// The curated path's init (`None` ⇒ the txt2img σ_max start) and its strength.
    Curated { x0: Option<Array>, strength: f32 },
    /// The legacy assemblies' init: the pose tier's (from the reference == IP image) or plain
    /// img2img's (from its reference). A Control's non-IP Reference is not encoded here.
    Legacy(Option<Array>),
}

/// The seed-independent half of stage 2 (`build_init`).
fn build_init_latents(
    request: &KolorsRequest<'_>,
    heavy: &KolorsHeavy,
    mode: &KolorsBoundMode<'_, '_>,
) -> Result<KolorsInit> {
    let (w, h) = (request.req.width, request.req.height);
    let encode = |image: &Image| encode_init_latents(heavy.vae(), image, w, h);
    Ok(if request.use_curated {
        match mode.curated_init() {
            Some((image, strength)) => KolorsInit::Curated {
                x0: Some(encode(image)?),
                strength,
            },
            None => KolorsInit::Curated {
                x0: None,
                strength: 0.0,
            },
        }
    } else {
        KolorsInit::Legacy(mode.legacy_init_image().map(encode).transpose()?)
    })
}

/// PiD decode overlay (epic 7840, sc-7848) + `from_ldm` early-stop (sc-8049). Kolors is a
/// **variance-preserving** PiD student on the shared SDXL backbone, but EVERY Kolors path stores RAW
/// variance-exploding latents `x0 + σ·ε`. So `from_ldm` is uniform: truncate the schedule to the
/// VP-capture `keep` nodes and map the captured VE latent into the student's VP frame with the
/// plan's `rescale` (`1/√(1+σ_edm²)`) before decode. The plan is image-independent, so it is
/// resolved once, against the EXACT schedule the active mode denoises, BEFORE the per-image
/// `random::seed`/noise draws (does not perturb the noise stream). `pid_capture_sigma` unset ⇒
/// `None` ⇒ byte-identical clean decode.
fn resolve_vp_plan(
    request: &KolorsRequest<'_>,
    heavy: &KolorsHeavy,
    mode: &KolorsBoundMode<'_, '_>,
) -> Result<Option<VpCapturePlan>> {
    let req = request.req;
    if !(req.use_pid && req.pid_capture_sigma.is_some()) {
        return Ok(None);
    }
    let steps = request.steps;
    let edm_sigmas: Vec<f32> = if request.use_curated {
        let sched = AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let scheduler = req
            .scheduler
            .as_deref()
            .and_then(Scheduler::from_name)
            .unwrap_or(Scheduler::Normal);
        let full_sigmas = schedule_sigmas(scheduler, &ms, steps);
        // Mirrors the curated init: combined-pose + img2img are strength-sliced; txt2img runs the
        // full schedule.
        match mode.curated_init() {
            Some((_, strength)) => mlx_gen_sdxl::stages::curated_strength_tail(
                &full_sigmas,
                steps,
                strength.clamp(0.0, 1.0),
            ),
            None => full_sigmas,
        }
    } else {
        let strength = match *mode {
            // Combined strict-pose tier: img2img at POSE_IMG2IMG_STRENGTH (or `req.strength`).
            KolorsMode::PoseIp { init_strength, .. } => Some(init_strength),
            KolorsMode::Img2Img { strength, .. }
            | KolorsMode::Control {
                img2img: Some((_, strength)),
                ..
            } => Some(strength),
            KolorsMode::Txt2Img | KolorsMode::Ip { .. } | KolorsMode::Control { .. } => None,
        };
        match strength {
            Some(strength) => KolorsEulerSampler::kolors_img2img(steps, strength, heavy.dtype())?
                .edm_sigmas()
                .to_vec(),
            // txt2img / controlnet-only / ip-only: the full native leading-Euler schedule.
            None => KolorsEulerSampler::kolors(steps, heavy.dtype())?
                .edm_sigmas()
                .to_vec(),
        }
    };
    Ok(vp_capture_plan(&edm_sigmas, req.pid_capture_sigma))
}

/// Stages 2–3 for one Kolors output, over the request's bound mode and init (sc-27336).
struct KolorsRenderPlan<'a> {
    request: &'a KolorsRequest<'a>,
    heavy: &'a KolorsHeavy,
    mode: &'a KolorsBoundMode<'a, 'a>,
    init: &'a KolorsInit,
    pos: &'a (Array, Array),
    neg: Option<&'a (Array, Array)>,
    vp_plan: Option<VpCapturePlan>,
}

impl mlx_gen_sdxl::stages::FamilyStages for KolorsRenderPlan<'_> {
    type Start = ();
    /// The output's initial noise, held through its decode exactly as the inline loop held it.
    type Retained = Array;

    /// Draw this image's initial noise. Only one global-RNG draw happens per image (the noise); the
    /// img2img VAE-encode draws none, so the per-image output stays byte-identical to the struct
    /// API.
    fn build_init(&self, _seed: u64) -> Result<((), Array)> {
        let req = self.request.req;
        let (lh, lw) = (
            req.height as i32 / SPATIAL_SCALE,
            req.width as i32 / SPATIAL_SCALE,
        );
        Ok((
            (),
            random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?,
        ))
    }

    fn denoise(
        &self,
        (): (),
        noise: &Array,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let latents = match self.init {
            KolorsInit::Curated { x0, strength } => {
                self.denoise_curated(x0.as_ref(), *strength, noise, seed, on_progress)?
            }
            KolorsInit::Legacy(x0) => self.denoise_legacy(x0.as_ref(), noise, on_progress)?,
        };
        mlx_gen_sdxl::stages::rescale_ve_capture(latents, self.vp_plan.as_ref())
    }
}

impl KolorsRenderPlan<'_> {
    /// The run-step truncation each denoise method applies (keep nodes ⇒ keep-1 denoise steps).
    fn run_steps(&self) -> Option<usize> {
        self.vp_plan.map(|p| p.keep - 1)
    }

    fn denoise_curated(
        &self,
        x0: Option<&Array>,
        strength: f32,
        noise: &Array,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let request = self.request;
        let req = request.req;
        let (control, ip) = match *self.mode {
            KolorsMode::Control {
                net, image, scale, ..
            } => (Some((net, image, scale)), None),
            KolorsMode::PoseIp {
                net,
                skeleton,
                control_scale,
                ref tokens,
                ip_scale,
                ..
            } => (
                Some((net, skeleton, control_scale)),
                Some((tokens, ip_scale)),
            ),
            KolorsMode::Ip {
                ref tokens, scale, ..
            } => (None, Some((tokens, scale))),
            KolorsMode::Txt2Img | KolorsMode::Img2Img { .. } => (None, None),
        };
        self.heavy.denoise_curated_latents_with_preview(
            req.sampler.as_deref(),
            req.scheduler.as_deref(),
            x0,
            noise,
            self.pos,
            self.neg,
            request.steps,
            strength,
            request.cfg,
            seed,
            req.height as i32,
            req.width as i32,
            control,
            ip,
            self.run_steps(),
            &req.cancel,
            on_progress,
            &req.preview,
            request.forward_plan,
            request.cfg_batching,
        )
    }

    /// The legacy (native `euler_discrete`) per-mode assemblies. Each arm binds the hoisted init it
    /// needs in its own pattern; an arm whose init is absent falls through exactly as the
    /// historical `if let` chain did.
    fn denoise_legacy(
        &self,
        x0: Option<&Array>,
        noise: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let request = self.request;
        let req = request.req;
        let (h, w) = (req.height as i32, req.width as i32);
        match (self.mode, x0) {
            (
                &KolorsMode::PoseIp {
                    net,
                    skeleton,
                    control_scale,
                    ref tokens,
                    ip_scale,
                    init_strength,
                    ..
                },
                Some(init_latents),
            ) => self.heavy.denoise_controlnet_ip_latents_with_preview(
                net,
                tokens,
                init_latents,
                noise,
                skeleton,
                self.pos,
                self.neg,
                request.steps,
                init_strength,
                request.cfg,
                control_scale,
                ip_scale,
                h,
                w,
                self.run_steps(),
                &req.cancel,
                on_progress,
                &req.preview,
                request.forward_plan,
                request.cfg_batching,
            ),
            (
                &KolorsMode::Control {
                    net, image, scale, ..
                }
                | &KolorsMode::PoseIp {
                    net,
                    skeleton: image,
                    control_scale: scale,
                    ..
                },
                _,
            ) => self.heavy.denoise_controlnet_latents_with_preview(
                net,
                noise,
                image,
                self.pos,
                self.neg,
                request.steps,
                request.cfg,
                scale,
                h,
                w,
                self.run_steps(),
                &req.cancel,
                on_progress,
                &req.preview,
                request.forward_plan,
                request.cfg_batching,
            ),
            (KolorsMode::Ip { tokens, scale, .. }, _) => {
                self.heavy.denoise_ip_latents_with_preview(
                    tokens,
                    noise,
                    self.pos,
                    self.neg,
                    request.steps,
                    request.cfg,
                    *scale,
                    h,
                    w,
                    self.run_steps(),
                    &req.cancel,
                    on_progress,
                    &req.preview,
                    request.forward_plan,
                    request.cfg_batching,
                )
            }
            (&KolorsMode::Img2Img { strength, .. }, Some(x0)) => {
                self.heavy.denoise_img2img_latents_with_preview(
                    x0,
                    noise,
                    self.pos,
                    self.neg,
                    request.steps,
                    strength,
                    request.cfg,
                    h,
                    w,
                    self.run_steps(),
                    &req.cancel,
                    on_progress,
                    &req.preview,
                    request.forward_plan,
                    request.cfg_batching,
                )
            }
            (KolorsMode::Img2Img { .. } | KolorsMode::Txt2Img, _) => {
                self.heavy.denoise_latents_with_preview(
                    noise,
                    self.pos,
                    self.neg,
                    request.steps,
                    request.cfg,
                    h,
                    w,
                    self.run_steps(),
                    &req.cancel,
                    on_progress,
                    &req.preview,
                    request.forward_plan,
                    request.cfg_batching,
                )
            }
        }
    }
}

/// Capability-driven request validation (unit-testable without loaded weights).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // Shared capability contract: count/size range, negative_prompt/guidance/true_cfg support,
    // sampler, scheduler, and conditioning kinds. Delegating to core keeps Kolors from drifting
    // out of sync with the descriptor (F-132); this was previously a hand-rolled copy that had
    // already lost the negative_prompt/guidance/true_cfg/scheduler checks.
    caps.validate_request(MODEL_ID, req)?;

    // Kolors-specific checks layered on top of the shared contract:
    if req.prompt.is_empty() {
        return Err(Error::Msg("kolors: prompt must not be empty".into()));
    }
    // `steps == 0` divides by zero in `KolorsEulerSampler::new` (`num_train_timesteps / num_steps`) —
    // now rejected by the shared floor above (F-007). `steps > 1100` (the train-timestep count) makes
    // `step_ratio == 0` so every timestep collapses to 1 — a silent-garbage upper bound the floor
    // doesn't know about, so keep it here (F-124). `None` falls back to DEFAULT_STEPS.
    if let Some(steps) = req.steps {
        if steps as usize > NUM_TRAIN_TIMESTEPS {
            return Err(Error::Msg(format!(
                "kolors: steps must be in 1..={NUM_TRAIN_TIMESTEPS} (got {steps})"
            )));
        }
    }
    // The /8 VAE plus two exact U-Net downsample/upsample joins require SIZE_MULTIPLE.
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "kolors: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
//
/// Per-component bytes for this snapshot — **the decoder projected to its resident width, not its
/// stored one** (sc-15839).
///
/// `text_encoder/` and `unet/` are priced from stored bytes because that is what they cost resident:
/// the ChatGLM3 tower and the U-Net load at their snapshot precision (fp16, or already-packed q4/q8
/// codes, which `PerComponentBytes` sums verbatim).
///
/// The VAE is different, and the difference is a factor of two. [`mlx_gen_sdxl::load_vae`] does
/// `cast_all(Dtype::Float32)` **unconditionally** — the SDXL VAE is fp16-unstable, so it runs f32
/// even when everything around it is fp16 — while every cached Kolors tier ships only
/// `vae/diffusion_pytorch_model.fp16.safetensors` (159.56 MiB). A stored-bytes footprint therefore
/// declared 159.56 MiB for a component that is 319.11 MiB resident, at **all three** tiers, and the
/// worker sizes budgets off these facts. Projecting through the shared
/// [`projected_safetensors_bytes`](mlx_gen::asset_facts::projected_safetensors_bytes) primitive —
/// the same one z-image, krea, qwen-image and mage use — keeps that arithmetic out of this file.
///
/// The file is resolved by [`mlx_gen_sdxl::resolve_vae_weight_file`] rather than summed over `vae/`,
/// so the footprint sizes exactly the file the resident load opens even on a snapshot that happens
/// to cache both the f32 master and the fp16 variant.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mut components = mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["unet"],
        &["vae"],
    )?;
    let WeightsSource::Dir(root) = &spec.weights else {
        // Unreachable: `from_spec_subdirs` already rejected a single-file checkpoint.
        return Ok(components);
    };
    components.vae = projected_vae_bytes(root)?;
    Ok(components)
}

/// The VAE's **resident** bytes: every tensor of the file [`mlx_gen_sdxl::load_vae`] opens,
/// materialized f32.
pub(crate) fn projected_vae_bytes(root: &std::path::Path) -> mlx_gen::gen_core::Result<u64> {
    let file = mlx_gen_sdxl::resolve_vae_weight_file(root)
        .map_err(|e| mlx_gen::gen_core::Error::Msg(format!("kolors: {e}")))?;
    mlx_gen::asset_facts::projected_safetensors_bytes(&file, |_| {
        mlx_gen::asset_facts::ResidentProjection::Float32
    })
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// The weights-free memory-strategy registration (SC-15521) — the shared registry conformance
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
/// suite can drive every declared rung's lifecycle without loading a 6 GB tier.
pub const MEMORY_BEHAVIOR_REGISTRATION: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-27336: the typed mode matrix, weights-free. The curated and legacy paths keep their
    /// historical difference for a Control + non-IP Reference request (refused by `validate_impl`,
    /// but encoded rather than unified): curated img2img-initialises from it, legacy ignores it.
    #[test]
    fn mode_matrix_keeps_the_curated_and_legacy_inits_distinct() {
        let reference = Image {
            width: 1,
            height: 1,
            pixels: vec![1, 2, 3],
        };
        let skeleton = Image::default();
        let err = KolorsMode::resolve(true, None, None, None).unwrap_err();
        assert!(
            err.to_string().contains("no Reference image was provided"),
            "{err}"
        );
        let err = KolorsMode::resolve(true, None, Some((&skeleton, 1.0)), None).unwrap_err();
        assert!(
            err.to_string().contains("no Reference image was provided"),
            "{err}"
        );

        assert!(matches!(
            KolorsMode::resolve(false, None, None, Some(0.2)).unwrap(),
            KolorsMode::Txt2Img
        ));
        let img2img =
            KolorsMode::resolve(false, Some((&reference, None)), None, Some(0.4)).unwrap();
        assert!(matches!(img2img, KolorsMode::Img2Img { strength, .. } if strength == 0.4));
        assert_eq!(img2img.curated_init().map(|(_, s)| s), Some(0.4));
        assert!(img2img.legacy_init_image().is_some());
        let defaulted = KolorsMode::resolve(false, Some((&reference, None)), None, None).unwrap();
        assert!(matches!(defaulted, KolorsMode::Img2Img { strength, .. }
            if strength == DEFAULT_IMG2IMG_STRENGTH));

        let ip = KolorsMode::resolve(true, Some((&reference, None)), None, Some(0.9)).unwrap();
        assert!(matches!(ip, KolorsMode::Ip { scale, .. } if scale == IP_DEFAULT_SCALE));
        assert_eq!(ip.curated_init().map(|(_, s)| s), None);
        assert!(ip.legacy_init_image().is_none());

        // The pose tier: IP scale from the reference, init strength from the request.
        let pose = KolorsMode::resolve(
            true,
            Some((&reference, Some(0.3))),
            Some((&skeleton, 0.7)),
            None,
        )
        .unwrap();
        assert!(
            matches!(pose, KolorsMode::PoseIp { ip_scale, control_scale, init_strength, .. }
            if ip_scale == 0.3 && control_scale == 0.7 && init_strength == POSE_IMG2IMG_STRENGTH)
        );
        assert_eq!(pose.curated_init().map(|(i, _)| i), Some(&reference));
        assert_eq!(pose.legacy_init_image(), Some(&reference));

        let control_only = KolorsMode::resolve(false, None, Some((&skeleton, 1.0)), None).unwrap();
        assert!(control_only.curated_init().is_none());
        assert!(control_only.legacy_init_image().is_none());
        let control_ref = KolorsMode::resolve(
            false,
            Some((&reference, Some(0.6))),
            Some((&skeleton, 1.0)),
            None,
        )
        .unwrap();
        assert!(
            matches!(control_ref, KolorsMode::Control { img2img: Some((_, s)), .. } if s == 0.6)
        );
        assert_eq!(control_ref.curated_init(), Some((&reference, 0.6)));
        assert!(control_ref.legacy_init_image().is_none());
    }
    use crate::sampler::KolorsEulerSampler;

    #[test]
    fn descriptor_is_kolors() {
        let d = descriptor();
        assert_eq!(d.id, "kolors");
        assert_eq!(d.family, "kolors");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_preview);
        assert!(
            d.capabilities.supports_lora,
            "Kolors LoRA is wired (sc-4733)"
        );
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(!d.capabilities.accepts(ConditioningKind::Mask));
    }

    #[test]
    fn registered_in_family_catalog() {
        // The family catalog resolves "kolors" and reaches the loader without real weights.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/kolors".into()));
        let err = match crate::provider_registry().unwrap().load("kolors", &spec) {
            Ok(_) => panic!("bogus weights dir must fail to load"),
            Err(e) => e.to_string(),
        };
        assert!(
            !err.contains("no generator registered"),
            "kolors should resolve in the registry; got: {err}"
        );
    }

    #[test]
    fn validate_rejects_bad_steps() {
        // `steps == 0` would divide by zero in the sampler — rejected by the shared floor (F-007,
        // message "steps must be >= 1"). `steps > NUM_TRAIN_TIMESTEPS` collapses every timestep to
        // 1 — also the shared floor now, from the advertised range (sc-19559), which is derived
        // from the same `NUM_TRAIN_TIMESTEPS` Kolors' own F-124 check uses. Both must be rejected;
        // `None` and an in-range count pass.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        // steps == 0 is rejected by the floor with its own message.
        let zero = GenerationRequest {
            steps: Some(0),
            ..base.clone()
        };
        let err = validate_request(&caps, &zero).unwrap_err().to_string();
        assert!(err.contains("steps must be >= 1"), "steps=0 got: {err}");
        // steps > NUM_TRAIN_TIMESTEPS is now rejected by the SHARED floor, from the range this
        // descriptor advertises (sc-19559) — the same constant Kolors' own check below uses, so
        // the two cannot disagree. Asserting the shared message rather than the provider's is the
        // point: the bound moved onto the advertised surface, where a consumer can read it.
        let over = GenerationRequest {
            steps: Some(NUM_TRAIN_TIMESTEPS as u32 + 1),
            ..base.clone()
        };
        let err = validate_request(&caps, &over).unwrap_err().to_string();
        // Asserted by the SHARED guard's own unique wording, not by `1..=1100`: BOTH messages
        // contain that substring (`kolors: steps must be in 1..=1100 (got 1101)` is the provider's),
        // so the short form would not say which one fired — and which one fires is the whole claim.
        assert!(
            err.contains(&format!(
                "is outside this model's supported range 1..={NUM_TRAIN_TIMESTEPS}"
            )),
            "over-max must be refused by the SHARED range guard, got: {err}"
        );
        for ok in [None, Some(1), Some(50), Some(NUM_TRAIN_TIMESTEPS as u32)] {
            let req = GenerationRequest {
                steps: ok,
                ..base.clone()
            };
            assert!(validate_request(&caps, &req).is_ok(), "steps={ok:?}");
        }
    }

    /// Kolors' own F-124 over-max branch is **shadowed, not dead** — and this proves it still works.
    ///
    /// `validate_rejects_bad_steps` above shows the shared range guard now fires first for a real
    /// `descriptor()`, which leaves the provider's `steps must be in 1..=N` branch unreachable from
    /// that path and its wording asserted nowhere. Deleting it would be wrong: it is the backstop
    /// for a descriptor whose advertised range regressed to `Unconstrained`, which is a one-line
    /// edit away. `validate_request` takes `caps` by argument, so that exact regression is
    /// constructible — and under it the request must STILL be refused, by Kolors' own message.
    #[test]
    fn the_provider_over_max_branch_still_refuses_when_the_advertised_range_regresses() {
        let unadvertised = Capabilities {
            supported_steps: StepSupport::Unconstrained,
            ..descriptor().capabilities
        };
        let over = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            steps: Some(NUM_TRAIN_TIMESTEPS as u32 + 1),
            ..Default::default()
        };
        let err = validate_request(&unadvertised, &over)
            .expect_err("an over-max step count must be refused even with no advertised range")
            .to_string();
        assert!(
            err.contains(&format!(
                "kolors: steps must be in 1..={NUM_TRAIN_TIMESTEPS}"
            )),
            "the PROVIDER's F-124 branch must be what refuses this: {err}"
        );
        // And it must not over-reach: the same unadvertised caps still admit the boundary count.
        let at_max = GenerationRequest {
            steps: Some(NUM_TRAIN_TIMESTEPS as u32),
            ..over.clone()
        };
        assert!(validate_request(&unadvertised, &at_max).is_ok());
    }

    #[test]
    fn validate_ties_size_multiple_to_pinned_stride() {
        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Kolors
        // bucket to. Pin the value and mutation-check that a VAE-valid size which is not the full
        // U-Net multiple is rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 32);
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            height: 1024,
            ..Default::default()
        };
        let off_stride = validate_request(
            &caps,
            &GenerationRequest {
                width: 1000, // 125×8 — VAE-valid but not SIZE_MULTIPLE
                ..base.clone()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            off_stride.contains("multiples of 32"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                width: 1024, // 32×32 — on-stride
                ..base.clone()
            },
        )
        .is_ok());
    }

    #[test]
    fn sampler_rejects_zero_steps() {
        // The defensive guard in `KolorsEulerSampler::new` (reached via `kolors`) returns a typed error
        // rather than panicking on the divide-by-zero (F-124).
        let err = match KolorsEulerSampler::kolors(0, mlx_rs::Dtype::Float32) {
            Ok(_) => panic!("num_steps == 0 must error, not build a sampler"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("num_steps must be >= 1"), "got: {err}");
    }

    /// **sc-18317 — the declared CFG-batching domain is reachable, and the two domains this lane has
    /// no mechanism for are refused.**
    ///
    /// The production `validate` delegates the shared contract to `Capabilities::validate_request`
    /// (asserted separately), so admitting/refusing here is admitting/refusing on the `generate` path.
    /// `Batched` is admitted because `model::cfg_conditioning` genuinely consumes it; `Sequential` is refused because the
    /// ChatGLM3 assembly and the shared SDXL denoise runs guidance as one batch; and the cadence / FFN-chunk domains are refused because this
    /// family has no such mechanism at all — the truthful state, not an oversight.
    #[test]
    fn execution_domains_are_declared_exactly_where_this_lane_consumes_them() {
        let caps = descriptor().capabilities;
        assert!(caps.execution.declaration_errors().is_empty());
        assert!(caps.execution.cfg_batching.is_supported());
        assert!(!caps.execution.graph_eval_cadence_blocks.is_supported());
        assert!(!caps.execution.ffn_chunk_rows.is_supported());

        let request = |memory| mlx_gen::gen_core::GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            count: 1,
            steps: Some(1),
            memory: Some(memory),
            ..Default::default()
        };

        caps.validate_request(
            "kolors",
            &request(mlx_gen::gen_core::GenerationMemory {
                cfg_batching: Some(mlx_gen::gen_core::CfgBatching::Batched),
                ..Default::default()
            }),
        )
        .expect("the implemented mode must be admitted");

        for (label, memory) in [
            (
                "cfg_batching",
                mlx_gen::gen_core::GenerationMemory {
                    cfg_batching: Some(mlx_gen::gen_core::CfgBatching::Sequential),
                    ..Default::default()
                },
            ),
            (
                "graph_eval_cadence",
                mlx_gen::gen_core::GenerationMemory {
                    graph_eval_cadence: Some(mlx_gen::gen_core::GraphEvalCadence::EVERY_BLOCK),
                    ..Default::default()
                },
            ),
            (
                "ffn_chunk",
                mlx_gen::gen_core::GenerationMemory {
                    ffn_chunk: Some(mlx_gen::gen_core::FfnChunk::new(2048).unwrap()),
                    ..Default::default()
                },
            ),
        ] {
            let error = caps
                .validate_request("kolors", &request(memory))
                .expect_err("an unimplemented execution selection must be refused");
            assert!(
                matches!(error, mlx_gen::gen_core::Error::Unsupported(_)),
                "{label} must be a capability gap: {error:?}"
            );
            let message = error.to_string();
            assert!(message.contains(label), "{label}: {message}");
            assert!(
                message.contains("unset"),
                "{label} refusal must name the remedy: {message}"
            );
        }

        // And an unset selection still validates — the default-preservation half.
        caps.validate_request(
            "kolors",
            &request(mlx_gen::gen_core::GenerationMemory::default()),
        )
        .expect("an unset execution selection must validate");
    }

    #[test]
    fn validate_delegates_to_core_capability_checks() {
        // F-132: `validate_request` now delegates the shared contract to `Capabilities::validate_request`
        // rather than re-implementing it. Assert the checks the hand-rolled copy had dropped now fire:
        // an unsupported scheduler and a `true_cfg` the descriptor doesn't advertise.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };

        let bad_scheduler = GenerationRequest {
            scheduler: Some("ddim".into()),
            ..base.clone()
        };
        assert!(
            validate_request(&caps, &bad_scheduler).is_err(),
            "unsupported scheduler must be rejected (delegated to core)"
        );

        let bad_true_cfg = GenerationRequest {
            true_cfg: Some(4.0),
            ..base.clone()
        };
        assert!(
            validate_request(&caps, &bad_true_cfg).is_err(),
            "true_cfg must be rejected — Kolors advertises supports_true_cfg=false"
        );

        // The advertised scheduler still passes.
        let good = GenerationRequest {
            scheduler: Some("discrete".into()),
            ..base
        };
        assert!(validate_request(&caps, &good).is_ok());
    }

    #[test]
    fn validate_accepts_curated_samplers_and_schedulers() {
        // epic 7114 (sc-7121): the unified curated solver + scheduler menu is advertised additively
        // alongside the native `euler_discrete`/`discrete` and accepted by the shared capability check.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        for s in [
            "euler_discrete",
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "lcm",
            "ddim",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        sampler: Some(s.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "sampler {s:?} should be accepted"
            );
        }
        for s in [
            "discrete",
            "normal",
            "karras",
            "sgm_uniform",
            "beta",
            "ddim_uniform",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        scheduler: Some(s.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "scheduler {s:?} should be accepted"
            );
        }
        // Unknown names are still rejected (delegated to the shared contract).
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                sampler: Some("nonsense".into()),
                ..base.clone()
            }
        )
        .is_err());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                scheduler: Some("leading".into()),
                ..base
            }
        )
        .is_err());
    }

    #[test]
    fn advertises_lora_adapters() {
        // sc-4733: LoRA/LoKr are wired — merged into the SDXL-family U-Net at load. The descriptor
        // advertises both (the real-weight merge + scale=0≡base parity is `tests/lora_parity.rs`).
        assert!(descriptor().capabilities.supports_lora);
        assert!(descriptor().capabilities.supports_lokr);
    }

    #[test]
    fn advertises_sequential_offload() {
        // Kolors now honors the shared Residency seam (epic 10834, sc-10840).
        assert!(descriptor().capabilities.supports_sequential_offload);
    }

    // ── Sequential residency (epic 10834, sc-10840): weight-free proof the dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent snapshot dir; the discriminator is
    // deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the ChatGLM3 encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The real-weights A/B is `#[ignore]`d; this runs by default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/kolors-residency-test-snapshot".into(),
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
            !msg.contains("single .safetensors file"),
            "expected an eager-load failure, not the up-front single-file guard: {msg}"
        );
    }
}

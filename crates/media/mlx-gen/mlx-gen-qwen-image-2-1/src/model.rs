//! `QwenImage21` — the Qwen-Image 2.1 text-to-image implementation of [`mlx_gen::Generator`],
//! plus its [`descriptor`]/[`load`] entry points and the explicit registration constant.
//!
//! [`load`] assembles the model from a `Qwen/Qwen-Image-2.1` snapshot directory (see
//! [`crate::loader`]) and [`QwenImage21::generate`] runs prompt → image: template + tokenize →
//! Qwen3-VL conditioning (drop the system prefix) → seeded packed noise → resolution-shifted
//! flow-match Euler denoise (true CFG when a negative prompt and a scale above 1 are given) →
//! unpack → denormalise → RGBA decode → RGB composited over white.

use std::path::Path;

use gen_core_family::qwen_image_2_1::pag::{
    qwen_image_2_1_pag_capability, qwen_image_2_1_turbo_pag_capability,
};
use mlx_gen::gen_core::{
    ExecutionSurface, GenerationReport, PrefixKvCacheDomain, PrefixKvCacheMode, PrefixKvCacheReport,
};
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, gen_core, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor,
    OutputChannels, Precision, Progress, Quant, Residency, Result, SizeFloor,
};

use crate::config::{
    SavedSigmaSchedule, SchedulerConfig, TransformerConfig, DEFAULT_STEPS, DEFAULT_TRUE_CFG,
    PRESETS, SIZE_MULTIPLE, VAE_SCALE_FACTOR,
};
use crate::loader;
use crate::pipeline::{
    create_noise, decode_rgb, decode_rgba, denoise_reported_with_pag_and_run, encode_references,
    joint_branch, prepare_conditioning_references, DenoiseInputs, DenoiseReport, JointBranch,
    KvCacheMode, KvCachePath, ReferenceConditioning,
};
use crate::reference::{collect_references, PreparedReference};
use crate::scheduler;
use crate::text_encoder::{system_prompt_drop_count, QwenImage21TextEncoder};
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// Registry id for Qwen-Image 2.1 (the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "qwen_image_2_1";
/// Registry id for the distilled Qwen-Image 2.1 **Turbo** checkpoint (sc-25900): the same
/// architecture, presets and licence as [`MODEL_ID`], sampled over the few-step sigma list its
/// snapshot saves instead of the resolution-shifted schedule.
pub const TURBO_MODEL_ID: &str = "qwen_image_2_1_turbo";
/// Descriptor family — distinct from the 2512 crate's `qwen-image`.
pub const FAMILY: &str = "qwen-image-2-1";

/// The label the family's shared code (transformer, quant, reference, adapters, memory pricing)
/// starts its refusals with: the base engine id's.
const BASE_LABEL: &str = "qwen_image_2_1: ";

/// E3 (sc-25909) — **a refusal reached through engine id `id` names `id`**, one rule on both
/// backends. The family's shared code labels its refusals with the base id (it serves both and
/// is not told which); every public entry point of a Turbo-id object — the generator's load,
/// `validate`, `generate`, `generate_with_report` and request scope, and the Turbo memory
/// registration — rewrites a leading `qwen_image_2_1: ` to `qwen_image_2_1_turbo: ` on the way
/// out. A base-id refusal, and a message that does not start with the base label (one that
/// already names an id, or a backend error), passes through unchanged.
pub(crate) fn label_message(id: &str, message: String) -> String {
    if id == TURBO_MODEL_ID {
        if let Some(rest) = message.strip_prefix(BASE_LABEL) {
            return format!("{id}: {rest}");
        }
    }
    message
}

/// [`label_message`] over this crate's error type.
pub(crate) fn labelled(id: &str, error: Error) -> Error {
    match error {
        Error::Msg(message) => Error::Msg(label_message(id, message)),
        Error::Unsupported(message) => Error::Unsupported(label_message(id, message)),
        Error::GeometryRefused {
            reason,
            requested_width,
            requested_height,
            alternative,
        } => Error::GeometryRefused {
            reason: label_message(id, reason),
            requested_width,
            requested_height,
            alternative,
        },
        other => other,
    }
}

/// [`label_message`] over gen-core's error type.
pub(crate) fn labelled_core(id: &str, error: gen_core::Error) -> gen_core::Error {
    match error {
        gen_core::Error::Msg(message) => gen_core::Error::Msg(label_message(id, message)),
        gen_core::Error::Unsupported(message) => {
            gen_core::Error::Unsupported(label_message(id, message))
        }
        gen_core::Error::GeometryRefused {
            reason,
            requested_width,
            requested_height,
            alternative,
        } => gen_core::Error::GeometryRefused {
            reason: label_message(id, reason),
            requested_width,
            requested_height,
            alternative,
        },
        other => other,
    }
}
/// Smallest side upstream's pipeline can render: one vision slot (a 2×2 latent) at 16 px/token.
pub const MIN_SIZE: u32 = 32;

/// Which released checkpoint of the family an engine id serves. The two share every component
/// architecture and differ only in how the denoise schedule is obtained ([`Schedule`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// `Qwen/Qwen-Image-2.1` — [`MODEL_ID`], the resolution-shifted 40-step schedule.
    Base,
    /// `Qwen/Qwen-Image-2.1-Turbo` — [`TURBO_MODEL_ID`], the snapshot's saved sigmas.
    Turbo,
}

impl Variant {
    /// The engine id this variant registers under.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Base => MODEL_ID,
            Self::Turbo => TURBO_MODEL_ID,
        }
    }

    /// The Hugging Face repository this variant's snapshot is released as.
    pub const fn upstream_hf_repo(self) -> &'static str {
        match self {
            Self::Base => crate::UPSTREAM_HF_REPO,
            Self::Turbo => crate::UPSTREAM_TURBO_HF_REPO,
        }
    }

    /// The pinned revision of [`Self::upstream_hf_repo`] this crate is frozen against.
    pub const fn upstream_hf_revision(self) -> &'static str {
        match self {
            Self::Base => crate::UPSTREAM_HF_REVISION,
            Self::Turbo => crate::UPSTREAM_TURBO_HF_REVISION,
        }
    }

    /// The variant an engine id names, or `None` for an id this crate does not serve.
    pub fn from_id(id: &str) -> Option<Self> {
        [Self::Base, Self::Turbo]
            .into_iter()
            .find(|variant| variant.id() == id)
    }

    /// Descriptor for this variant.
    pub fn descriptor(self) -> ModelDescriptor {
        match self {
            Self::Base => descriptor(),
            Self::Turbo => descriptor_turbo(),
        }
    }
}

/// How one loaded checkpoint obtains its denoise schedule — read from the snapshot at load.
#[derive(Clone, Debug, PartialEq)]
pub enum Schedule {
    /// The base checkpoint: `linspace` → resolution-dependent `mu` shift → terminal stretch
    /// ([`scheduler::sigmas`]).
    Shifted(SchedulerConfig),
    /// The distilled checkpoint: the snapshot's saved sigmas, resampled to the requested step
    /// count ([`scheduler::resample_saved_sigmas`]) — no shift, no stretch.
    Saved(SavedSigmaSchedule),
}

impl Schedule {
    /// Frozen production schedule for weights-free `RenderPlan` compilation.
    pub fn production(variant: Variant) -> Self {
        match variant {
            Variant::Base => Self::Shifted(SchedulerConfig::production()),
            Variant::Turbo => Self::Saved(SavedSigmaSchedule {
                sample_sigmas: vec![
                    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
                ],
                num_train_timesteps: 1000,
            }),
        }
    }

    /// The schedule a snapshot carries for `variant`. Each variant refuses the other's snapshot by
    /// name instead of running the wrong schedule over it: the base id refuses a snapshot that
    /// saves `sample_sigmas` ([`loader::refuse_saved_sigma_snapshot`]), and the Turbo id refuses
    /// one that does not, or whose scheduler config is not the static identity shift
    /// ([`loader::load_saved_sigma_schedule`]).
    pub fn load(variant: Variant, root: &Path) -> Result<Self> {
        match variant {
            Variant::Base => {
                loader::refuse_saved_sigma_snapshot(root)?;
                loader::load_scheduler_config(root).map(Self::Shifted)
            }
            Variant::Turbo => loader::load_saved_sigma_schedule(root).map(Self::Saved),
        }
    }

    /// The step count a request that names none runs: upstream's 40 for the base checkpoint, one
    /// step per saved sigma for the distilled one.
    pub fn default_steps(&self) -> u32 {
        match self {
            Self::Shifted(_) => DEFAULT_STEPS,
            Self::Saved(saved) => saved.default_steps(),
        }
    }

    /// The native schedule for one request (length `steps + 1`, trailing `0.0`) and the `mu` a
    /// curated scheduler is built over: the resolution shift for the base checkpoint, `ln(1) = 0`
    /// for the distilled one's static unit shift.
    pub fn native_sigmas(&self, steps: usize, width: u32, height: u32) -> Result<(Vec<f32>, f32)> {
        match self {
            Self::Shifted(cfg) => {
                let tokens = scheduler::image_tokens(width, height);
                Ok((
                    scheduler::sigmas(cfg, steps, tokens)?,
                    scheduler::mu_for_tokens(cfg, tokens) as f32,
                ))
            }
            Self::Saved(saved) => Ok((
                scheduler::resample_saved_sigmas(&saved.sample_sigmas, steps)?,
                0.0,
            )),
        }
    }
}

/// The distilled Turbo checkpoint's identity + capabilities (sc-25900): [`descriptor`] under
/// [`TURBO_MODEL_ID`]. Nothing else differs on the advertised surface — the checkpoint shares the
/// base architecture, presets, reference/RGBA routes, adapter host and tiers, and upstream still
/// accepts a negative prompt with `true_cfg_scale > 1` on it. Its own defaults are not descriptor
/// fields: the step count is the number of sigmas the snapshot saves ([`Schedule::default_steps`])
/// and true CFG is off unless asked for ([`DEFAULT_TRUE_CFG`]).
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = TURBO_MODEL_ID;
    d.capabilities.pag = Some(qwen_image_2_1_turbo_pag_capability());
    d
}

/// Qwen-Image 2.1's identity + capabilities — constructible without loading weights.
pub fn descriptor() -> ModelDescriptor {
    let max_size = PRESETS
        .iter()
        .map(|p| p.width.max(p.height))
        .max()
        .unwrap_or(2048);
    ModelDescriptor {
        // The Qwen3-VL tower is loaded from the snapshot's own `text_encoder/`; substituting one
        // through `LoadSpec::text_encoder` is not advertised for this route.
        encoder_contract: None,
        denoiser_output_latent_space: Some(&gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // True CFG: a negative prompt plus `true_cfg`/`guidance` above 1 runs the negative
            // branch (upstream default 1.0 = off).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Reference conditioning (sc-24110). `Reference` and `MultiReference` reach the same
            // upstream call — one ordered list of one to ten condition images — so both kinds are
            // advertised and flattened in request order, matching the seam the 2512 edit provider
            // already exposes to SceneWorks routing. `Mask` is deliberately **not** advertised:
            // upstream has no mask input (see UPSTREAM.md), and a mask travels as an ordinary
            // extra reference the prompt names.
            // `ReferenceRgba` (sc-24111) joins them: upstream converts every condition image
            // to RGBA anyway, so a reference that already carries alpha is the general case and
            // an RGB one the opaque special case. Advertising it is what lets a transparent layer
            // be edited (its alpha reaches the VAE encode) instead of being flattened at the
            // contract boundary.
            conditioning: vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::ReferenceRgba,
                gen_core::ConditioningKind::MultiReference,
            ],
            // Native transparency (sc-24111). The VAE is four-channel in and out on **every**
            // route behind this descriptor — T2I and reference alike — so there is no sub-path
            // that would have to fabricate an alpha to honour the flag. Whether a given render is
            // actually transparent is decided by the prompt, not by this bit; see UPSTREAM.md.
            supports_alpha_output: true,
            // LoRA + LoKr (sc-24156): installed onto every DiT Linear through the
            // transformer's `AdaptableHost`, stacked and mixed via the shared strict seam, as
            // residuals after any Q4/Q8 — so on every tier and on every route (T2I and the
            // reference/edit path share the one DiT `load_heavy` builds).
            supports_lora: true,
            supports_lokr: true,
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
            min_size: MIN_SIZE,
            max_size,
            max_count: 8,
            mac_only: true,
            // Both affine tiers ship as **pre-quantized** snapshots ([`crate::convert`]); against
            // one, `spec.quantize` only selects the tier on disk. Against a dense snapshot a Q4/Q8
            // request still quantizes the DiT and the Qwen3 tower at load (the tower's is a typed
            // refusal for a geometry the group size cannot cover). A non-affine request, a request
            // that disagrees with a packed tier, or no request against one is a typed refusal
            // ([`crate::quant::needs_load_time_quant`]), never a silent mis-serve.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // No `component_precision_floors`: a tier is a whole-pipeline contract, so every
            // packable component runs the tier the caller selected (`crate::quant`).
            requires_sigma_shift: true,
            supports_sequential_offload: true,
            // Both sides must be multiples of 32 px (16× VAE × 2×2 vision slot).
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            // The prefix KV cache (sc-25905) is a request-selected execution domain: both modes
            // are implemented, an unset field is the prefill path, and the shared floor refuses
            // nothing this declares.
            execution: ExecutionSurface {
                prefix_kv_cache: PrefixKvCacheDomain::Modes(
                    crate::memory_strategy::PREFIX_KV_CACHE_MODES.to_vec(),
                ),
                ..ExecutionSurface::default()
            },
            supports_low_res_start: true,
            pag: Some(qwen_image_2_1_pag_capability()),
            ..Default::default()
        },
    }
}

/// A loaded Qwen-Image 2.1 generator.
pub struct QwenImage21 {
    descriptor: ModelDescriptor,
    /// The checkpoint this generator serves — its engine id and schedule family.
    variant: Variant,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops, derived from the tokenizer.
    drop_count: usize,
    schedule: Schedule,
    residency: Residency<QwenImage21TextEncoder, Heavy>,
    /// The spec this generator was loaded from — the tier and residency policy the shared
    /// memory-strategy seam checks a request selection against (sc-24114).
    spec: LoadSpec,
    /// The loaded memory contract ([`crate::memory_strategy::memory_strategy_contract`] over the
    /// snapshot on disk), published through [`Generator::memory_strategy_contract`] so the
    /// SceneWorks worker's optimized selections reach this generator instead of the trait's
    /// "has not adopted the shared memory-strategy contract" default.
    memory_strategy: gen_core::MemoryProviderContract,
}

/// The one refusal both Qwen-Image 2.1 backends raise for a `LoadSpec::precision` override. The
/// wording is shared with `candle_gen_qwen_image_2_1::PRECISION_OVERRIDE_REFUSAL` by literal (the
/// two provider crates cannot share a symbol without routing it through gen-core); the tests on
/// both sides pin this exact text so the two cannot drift.
pub const PRECISION_OVERRIDE_REFUSAL: &str = "qwen_image_2_1: components load at the backend's \
    own compute dtype (the checkpoint's bf16 on MLX, CUDA and Metal; f32 on the candle CPU lane); \
    drop the precision override";

/// The heavy render-phase components — everything but the text encoder.
pub(crate) struct Heavy {
    transformer: QwenImage21Transformer,
    vae: QwenImage21Vae,
}

/// Construct a [`QwenImage21`] from a [`LoadSpec`] whose `weights` is a `Qwen/Qwen-Image-2.1`
/// snapshot directory — dense, or one of the pre-quantized tiers [`crate::convert`] produces.
///
/// `spec.quantize` (Q4/Q8) **selects a tier**. Against a pre-quantized snapshot it is a pure
/// selector: every projection packed-detects off its `{base}.scales` sibling and no quantization
/// pass runs, so a Q4 tier lands at ~4 bits/weight with no dense bf16 transient. Against a dense
/// snapshot it still means "quantize the DiT at load", the historical behaviour. A request that
/// disagrees with the tier on disk is a hard error rather than a silent mis-serve — see
/// [`crate::quant::needs_load_time_quant`].
///
/// `Resident` (default) holds every component warm; `Sequential` loads the text encoder, encodes,
/// drops it, then loads the DiT + VAE — bounding peak memory to `max(text encoder, DiT + VAE)`.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Variant::Base, spec)
}

/// [`load`] for the distilled checkpoint: `spec.weights` is a `Qwen/Qwen-Image-2.1-Turbo` snapshot
/// directory (dense or a pre-quantized tier of one), registered as [`TURBO_MODEL_ID`].
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Variant::Turbo, spec)
}

/// The one load path both engine ids share. `variant` picks the id the generator answers to and
/// the schedule family read from the snapshot ([`Schedule::load`]) — a snapshot of the other
/// variant is a typed refusal, raised before any weight is read.
pub fn load_variant(variant: Variant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant_unlabelled(variant, spec).map_err(|error| labelled(variant.id(), error))
}

fn load_variant_unlabelled(variant: Variant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    gen_core::reject_unknown_components(spec, &[], variant.id())?;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(PRECISION_OVERRIDE_REFUSAL.into()));
    }
    if spec.text_encoder.is_some() {
        return Err(Error::Unsupported(
            "qwen_image_2_1: the Qwen3-VL text encoder is loaded from the snapshot's own \
             text_encoder/; LoadSpec::text_encoder substitution is not advertised"
                .into(),
        ));
    }
    if let Some(q) = spec.quantize {
        if !matches!(q, Quant::Q4 | Quant::Q8) {
            return Err(Error::Unsupported(crate::quant::non_affine_quant_refusal(
                q,
            )));
        }
    }
    let root = loader::snapshot_root(&spec.weights)?;
    // The schedule first: it is two small JSON reads, and it is where a snapshot of the other
    // variant (base under the Turbo id, or the reverse) is refused by name.
    let schedule = Schedule::load(variant, root)?;
    // Resolve the request against the tier actually on disk before anything is read, so a
    // mismatched pair fails with the actionable message rather than rendering the wrong tier.
    crate::quant::needs_load_time_quant(root, spec.quantize)?;
    // The loaded contract is priced from the snapshot on disk BEFORE any weight is read, so a
    // tier/request disagreement is the same refusal whether it is asked for through `load` or
    // through the memory registration.
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(variant.id(), spec)?;
    // `Resident` builds the DiT (and installs the adapters) right here; `Sequential` defers that to
    // every generate, where an unmatched adapter key would fail each request instead of the load.
    // Resolve the stack now so both policies refuse a bad key at load, by name.
    if spec.offload_policy == gen_core::OffloadPolicy::Sequential && !spec.adapters.is_empty() {
        validate_adapters_on_lazy_dit(spec)?;
    }
    let tokenizer = loader::load_tokenizer(root)?;
    let drop_count = system_prompt_drop_count(&tokenizer)?;
    let residency = build_residency(spec)?;
    Ok(Box::new(QwenImage21 {
        descriptor: match variant {
            Variant::Base => descriptor(),
            Variant::Turbo => descriptor_turbo(),
        },
        variant,
        tokenizer,
        drop_count,
        schedule,
        residency,
        spec: spec.clone(),
        memory_strategy,
    }))
}

/// Whether one request runs staged: the request's admitted `GenerationMemory::stage_residency`
/// (written by the memory-strategy scope) OR a `Sequential` load policy. Mirrors the candle twin.
pub(crate) fn stage_residency_for(spec: &LoadSpec, req: &GenerationRequest) -> bool {
    req.memory.is_some_and(|memory| memory.stage_residency)
        || spec.offload_policy == gen_core::OffloadPolicy::Sequential
}

fn build_residency(spec: &LoadSpec) -> Result<Residency<QwenImage21TextEncoder, Heavy>> {
    let text_spec = spec.clone();
    let heavy_spec = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoder(&text_spec),
        move |_use_pid| load_heavy(&heavy_spec),
    )
}

/// The Qwen3 language tower at the tier's declared text-encoder width.
///
/// A packed tier loads packed. A dense snapshot with a Q4/Q8 request is quantized here to
/// [`crate::quant::Tier::text_encoder_bits`] — which is the selected tier, because a tier is a
/// whole-pipeline contract — so a load-time tier and the installable tier of the same name hold the
/// same resident layout rather than two different ones.
///
/// A geometry the single declared group size cannot cover is a **typed refusal**, not a silent
/// dense tower: serving a "q4" load whose text encoder is secretly bf16 is the mixed-tier failure
/// this route does not have.
fn load_text_encoder(spec: &LoadSpec) -> Result<QwenImage21TextEncoder> {
    let root = loader::snapshot_root(&spec.weights)?;
    let mut encoder = loader::load_text_encoder(root)?;
    if crate::quant::needs_load_time_quant(root, spec.quantize)? {
        if let Some(bits) = crate::quant::Tier::from_selected(spec.quantize)
            .and_then(crate::quant::Tier::text_encoder_bits)
        {
            if !encoder.quantize(bits)? {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1: this snapshot's Qwen3 tower is {} wide with a {} SwiGLU \
                     intermediate, and a tier is written at group {}; a load-time Q{bits} tier \
                     would leave the tower dense, which is not the tier that was asked for. Point \
                     at a pre-quantized snapshot, or drop the quantize request.",
                    encoder.hidden_size(),
                    encoder.intermediate_size(),
                    crate::quant::GROUP_SIZE,
                )));
            }
        }
    }
    Ok(encoder)
}

/// Run the exact `load_heavy` adapter install — same quantize decision, same strict seam, same
/// key→module map — against a DiT that is only a lazy graph, and drop it unevaluated.
///
/// MLX safetensors loads, the load-time quantize and the adapter install are all lazy graph
/// construction; nothing here calls `eval` or `materialize_accessed` (which
/// [`loader::load_transformer`] does), so no DiT weight is read and a `Sequential` load keeps its
/// `max(text encoder, DiT + VAE)` bound. What it does do is resolve every adapter key against the
/// real module tree, so an unmatched key fails here with the same named error the `Resident`
/// install raises.
fn validate_adapters_on_lazy_dit(spec: &LoadSpec) -> Result<()> {
    let root: &Path = loader::snapshot_root(&spec.weights)?;
    let mut probe = loader::load_transformer_lazy(root)?;
    if crate::quant::needs_load_time_quant(root, spec.quantize)? {
        let bits = spec
            .quantize
            .expect("needs_load_time_quant is false without a requested tier")
            .bits();
        probe.quantize(bits)?;
    }
    crate::adapters::apply_qwen_image_2_1_adapters(&mut probe, &spec.adapters)?;
    Ok(())
}

fn load_heavy(spec: &LoadSpec) -> Result<Heavy> {
    #[cfg(test)]
    let _diagnostic_bounds = crate::q4_diagnostic::materialization_bounds();
    let root: &Path = loader::snapshot_root(&spec.weights)?;
    let mut transformer = loader::load_transformer(root)?;
    // A pre-quantized tier is already packed — `quantize` would be a no-op over packed weights, so
    // the guard decides rather than the call silently doing nothing.
    if crate::quant::needs_load_time_quant(root, spec.quantize)? {
        let bits = spec
            .quantize
            .expect("needs_load_time_quant is false without a requested tier")
            .bits();
        transformer.quantize(bits)?;
    }
    // LoRA/LoKr (sc-24156): installed AFTER quantization, as forward-time residuals over the dense
    // or packed base, so a tier never has to re-pack an adapted Linear and every tier takes the
    // same adapters. Strict: an adapter key that resolves to no DiT Linear fails the load by name.
    // Every route (T2I and the reference/edit path) renders through this one DiT.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_image_2_1_adapters(&mut transformer, &spec.adapters)?;
    }
    #[cfg(test)]
    crate::q4_diagnostic::install_variant(&mut transformer, spec)?;
    let vae = loader::load_vae(root)?;
    Ok(Heavy { transformer, vae })
}

/// Separate test-only first-step prediction; all render replays finish before
/// this capture is enabled, so numeric observation cannot alter their lifecycle.
#[cfg(test)]
pub(crate) fn diagnostic_first_forward(spec: &LoadSpec, req: &GenerationRequest) -> Result<()> {
    let root = loader::snapshot_root(&spec.weights)?;
    let tokenizer = loader::load_tokenizer(root)?;
    let drop = system_prompt_drop_count(&tokenizer)?;
    let schedule = Schedule::load(Variant::Base, root)?;
    let params = resolve_run_params(&schedule, req)?;
    assert!(
        !params.use_negative,
        "fixed diagnostic uses the default positive branch"
    );
    let te = load_text_encoder(spec)?;
    let branches = assemble_reference_branches(&te, &tokenizer, req, drop, params.use_negative)?;
    let heavy = load_heavy(spec)?;
    assert_eq!(heavy.transformer.compute_dtype(), mlx_rs::Dtype::Float32);
    let references = encode_references(&heavy.vae, &branches.references)?;
    let noise = create_noise(
        params.base_seed,
        req.width,
        req.height,
        heavy.transformer.config().in_channels,
    )?;
    let _bounds = crate::memory_strategy::AllocatorBounds::enter(
        10_505_507_280 + 6_759_417,
        crate::memory_strategy::derived::request_transient_budget_bytes(
            crate::quant::Tier::from_selected(spec.quantize).unwrap_or(crate::quant::Tier::Bf16),
            req.width,
            req.height,
            crate::memory_strategy::derived::TABLE_CONDITIONING_TOKENS,
            req.memory_reference_count(),
            params.use_negative,
            None,
        ),
    );
    let velocity = heavy.transformer.forward_joint(
        &branches.pos.text,
        &crate::pipeline::joint_images(&references, &noise),
        params.sigmas[0],
        &branches.pos.layout,
    )?;
    mlx_rs::transforms::eval([&velocity])?;
    assert_eq!(velocity.dtype(), mlx_rs::Dtype::Float32);
    Ok(())
}

impl Generator for QwenImage21 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// The registered contract, priced from the loaded snapshot — the same one
    /// [`crate::memory_strategy::MEMORY_REGISTRATION`] builds for this spec.
    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.spec, &self.memory_strategy, context)
    }

    /// The registered executable seam (`crate::memory_strategy::begin_request_for`) on
    /// the loaded generator: the scope's `configure_request` writes the admitted
    /// `GenerationMemory` onto the request, which [`Self::generate`] then honours — staged
    /// residency through `Residency::run_request_scoped`, bounded decode through
    /// [`crate::pipeline::decode_tiling`].
    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request_for(
            self.variant.id(),
            &self.spec,
            &self.memory_strategy,
            context,
        )
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request_with_schedule(
            self.variant,
            &self.descriptor.capabilities,
            &self.schedule,
            req,
        )
        .map_err(|error| labelled(self.variant.id(), error).into())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        mlx_gen::image_execution::without_capture(|| self.generate_observed(req, on_progress))
    }

    /// The render plus which prefix-KV-cache path it ran and the bytes its caches held
    /// (`GenerationReport::prefix_kv_cache`, sc-25905) — the primary implementation;
    /// [`Self::generate`] is this without the report.
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        mlx_gen::image_execution::without_capture(|| {
            self.generate_with_report_observed(req, on_progress)
        })
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        if let Some(plan) = compile_trajectory_for(self.variant, &self.schedule, req)? {
            let plan = plan.bind_load(&self.spec);
            return self.generate_planned_with_execution_report(req, &plan, None, on_progress);
        }
        mlx_gen::image_execution::complete_image_report_attempt(
            self.descriptor(),
            &req.cancel,
            || {
                mlx_gen::image_execution::policy_debug(
                    "QwenImage21.loaded_sequential_residency",
                    &self.residency.is_sequential(),
                );
                mlx_gen::image_execution::policy_debug(
                    "QwenImage21.load_precision_policy",
                    &self.spec.precision,
                );
                mlx_gen::image_execution::policy_debug(
                    "QwenImage21.load_quantization_policy",
                    &self.spec.quantize,
                );
                mlx_gen::image_execution::applied_adapter_controls(
                    "QwenImage21.applied_adapter_controls",
                    &self.spec.adapters,
                );
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                self.generate_with_report_observed(req, &mut progress)
            },
        )
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        mlx_gen::image_execution::without_capture(|| {
            self.generate_planned_observed(req, plan, scope, on_progress)
        })
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        let mut scope = scope;
        mlx_gen::image_execution::complete_image_report_attempt(
            self.descriptor(),
            &req.cancel,
            || {
                mlx_gen::image_execution::policy_debug(
                    "QwenImage21.loaded_sequential_residency",
                    &self.residency.is_sequential(),
                );
                mlx_gen::image_execution::policy_debug(
                    "QwenImage21.load_precision_policy",
                    &self.spec.precision,
                );
                mlx_gen::image_execution::policy_debug(
                    "QwenImage21.load_quantization_policy",
                    &self.spec.quantize,
                );
                mlx_gen::image_execution::applied_adapter_controls(
                    "QwenImage21.applied_adapter_controls",
                    &self.spec.adapters,
                );
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                self.generate_planned_observed(req, plan, scope.take(), &mut progress)
            },
        )
    }
}

impl QwenImage21 {
    fn generate_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let report = self.generate_with_report_observed(req, on_progress)?;
        report.output.ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{}: the render produced no output",
                self.variant.id()
            ))
        })
    }

    fn generate_with_report_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        if let Some(plan) = compile_trajectory_for(self.variant, &self.schedule, req)? {
            let plan = plan.bind_load(&self.spec);
            return self.generate_planned_observed(req, &plan, None, on_progress);
        }
        self.generate_impl(req, None, on_progress)
            .map_err(|error| labelled(self.variant.id(), error).into())
    }

    fn generate_planned_observed(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        plan.check_request(&self.descriptor, req)
            .map_err(|error| labelled_core(self.variant.id(), error))?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.spec))
            .map_err(|error| labelled_core(self.variant.id(), error))?;
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let mut report = self
            .generate_impl(req, Some(&run), on_progress)
            .map_err(|error| labelled(self.variant.id(), error))?;
        let slot_cfg = if self.spec.adapters.is_empty() {
            "base_dit"
        } else {
            "base_dit+adapters"
        };
        report.low_res_start = plan
            .low_res_start_execution_report("f32", 4, slot_cfg)
            .map_err(|error| labelled_core(self.variant.id(), error))?;
        Ok(report)
    }
}

/// The resolved run parameters of one request.
pub(crate) struct RunParams {
    pub(crate) true_cfg: f32,
    pub(crate) use_negative: bool,
    pub(crate) base_seed: u64,
    pub(crate) sigmas: Vec<f32>,
}

pub(crate) fn resolve_run_params(
    schedule: &Schedule,
    req: &GenerationRequest,
) -> Result<RunParams> {
    let steps = req.steps.unwrap_or(schedule.default_steps()) as usize;
    let (native, mu) = schedule.native_sigmas(steps, req.width, req.height)?;
    let sigmas = mlx_gen::resolve_flow_schedule_with_shift(
        req.scheduler.as_deref(),
        req.scheduler_shift,
        mu,
        steps,
        &native,
    )?;
    // `true_cfg_scale`: honour a caller-supplied `true_cfg`, then `guidance`, then the upstream
    // default (off). Guidance engages only with a negative prompt, as upstream's `do_true_cfg`.
    let true_cfg = req.true_cfg.or(req.guidance).unwrap_or(DEFAULT_TRUE_CFG);
    mlx_gen::image_execution::scalar("qwen-image-2-1.true_cfg", true_cfg);
    let has_negative = req
        .negative_prompt
        .as_deref()
        .is_some_and(|n| !n.trim().is_empty());
    Ok(RunParams {
        true_cfg,
        use_negative: true_cfg > 1.0 && has_negative,
        base_seed: req.seed.unwrap_or_else(default_seed),
        sigmas,
    })
}

impl QwenImage21 {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.validate(req)?;
        let resolved_pag = self
            .descriptor
            .capabilities
            .resolve_pag(self.variant.id(), req)?;
        mlx_gen::image_execution::policy_debug("qwen-image-2-1.pag", &resolved_pag);
        let params = resolve_run_params(&self.schedule, req)?;
        let actual_steps = params.sigmas.len().saturating_sub(1) as u32;
        let tiling = crate::pipeline::decode_tiling(req);
        mlx_gen::image_execution::policy_debug("qwen-image-2-1.tiling", &tiling);
        // sc-24114: request-scoped allocator bounds — cache limit = the derived transient budget
        // (freed buffers above it go back to the OS instead of pooling to the device's working-set
        // line, which is what let the footprint reach 110 GB), memory limit = resident + budget
        // (uncredited defence in depth against the eval pipeline's run-ahead). Restored on drop.
        // The resident term credits the installed adapter stack (sc-24156) — its residual factors
        // and any materialized LyCORIS deltas, exactly as the loaded contract prices them in
        // `overlay_bytes` — so an adapter load does not run under a limit sized for the bare base.
        let resident_bytes = derived_resident_bytes(
            self.variant,
            self.spec.quantize,
            self.memory_strategy.asset_facts.overlay_bytes,
        );
        #[cfg(test)]
        let resident_bytes =
            resident_bytes.saturating_add(crate::q4_diagnostic::resident_surcharge());
        // sc-25905: the prefix KV cache is a request selection, and the bound installed is the
        // bound of the mode selected — the cache-shed transient for a prefill render, and for a
        // cached one the same transient plus the priced cache bytes as *retained* memory (live
        // arrays the request holds, so they raise the memory limit and leave the buffer-cache
        // limit alone). Never a cached render under a prefill bound.
        let kv_cache = crate::memory_strategy::prefix_kv_cache_for(req);
        let tier = crate::quant::Tier::from_selected(self.spec.quantize)
            .unwrap_or(crate::quant::Tier::Bf16);
        let priced_cache_bytes = match kv_cache {
            PrefixKvCacheMode::Cached => {
                crate::memory_strategy::derived::prefix_kv_cache_demand_bytes(
                    tier,
                    req.width,
                    req.height,
                    req.memory_reference_count(),
                    params.use_negative,
                )
            }
            PrefixKvCacheMode::Prefill => 0,
        };
        let _bounds = crate::memory_strategy::AllocatorBounds::enter_retaining(
            resident_bytes,
            // sc-25909: at the loaded tier — a packed tier's denoise is priced at the f32 it
            // computes in, not at bf16.
            crate::memory_strategy::derived::request_transient_budget_bytes(
                tier,
                req.width,
                req.height,
                crate::memory_strategy::derived::TABLE_CONDITIONING_TOKENS,
                req.memory_reference_count(),
                params.use_negative,
                tiling
                    .as_ref()
                    .and_then(|t| t.spatial)
                    .map(|s| s.tile_px as u32),
            ),
            priced_cache_bytes,
        );
        let drop = self.drop_count;
        // Staged residency is a per-request execution decision (sc-24114): the shared worker
        // selects it through the memory-strategy scope, which writes `GenerationMemory` onto the
        // request. A `Sequential` load policy stages every request whatever the request says, so
        // the two are OR'd — a request can stage a warm-loaded generator, never un-stage a
        // sequential one.
        let stage_residency = stage_residency_for(&self.spec, req);
        mlx_gen::image_execution::policy_debug("qwen-image-2-1.stage_residency", &stage_residency);
        let (phase0_w, phase0_h) = run
            .map(|r| r.plan().phase_output_size(0))
            .unwrap_or((req.width, req.height));
        // What the denoise loops did, written as they run (one report per image; they agree).
        let mut ran: Option<PrefixKvCacheReport> = None;
        let mut warnings = Vec::new();
        let output = self.residency.run_request_scoped(
            stage_residency,
            false,
            &req.cancel,
            false,
            on_progress,
            |te: &QwenImage21TextEncoder| {
                if let Some(run) = run {
                    run.encode(|| {
                        assemble_reference_branches(
                            te,
                            &self.tokenizer,
                            req,
                            drop,
                            params.use_negative,
                        )
                    })
                } else {
                    assemble_reference_branches(te, &self.tokenizer, req, drop, params.use_negative)
                }
            },
            |_| Ok(()),
            |heavy, branches, on_progress| {
                let ReferenceBranches {
                    pos,
                    neg,
                    references,
                } = branches;
                let channels = heavy.transformer.config().in_channels;
                // A selection this checkpoint cannot honour is refused, never run as the other
                // mode: the cache is only valid on a `causal_condition` model (both released
                // checkpoints are), and a caller that priced a cached render must not be handed a
                // prefill one in silence.
                if kv_cache.is_cached() && !heavy.transformer.config().causal_condition {
                    return Err(Error::Unsupported(format!(
                        "{}: prefix_kv_cache=cached was selected, but this checkpoint is not \
                         `causal_condition` and has no prefix to cache. Select prefill, or leave \
                         memory.prefix_kv_cache unset.",
                        self.variant.id()
                    )));
                }
                // sc-25905: a cached render is held to what it was priced at. The real prefix is
                // known only now (tokenized prompt rows, fitted reference grids), so the caches
                // this request would hold are computed from the real layouts at the model's own
                // compute dtype and compared with the priced figure — the same arithmetic over
                // the published bounds at the tier's width. Past it, the request runs prefill:
                // never cached above its price, never a failed render.
                let overflow = if kv_cache.is_cached() {
                    let mut prefixes = vec![pos.layout.prefix_len() as u64];
                    prefixes.extend(neg.as_ref().map(|neg| neg.layout.prefix_len() as u64));
                    crate::memory_strategy::prefix_kv_cache_overflow(
                        self.variant.id(),
                        heavy.transformer.config(),
                        heavy.transformer.compute_dtype(),
                        tier,
                        &prefixes,
                        req.memory_reference_count(),
                        crate::memory_strategy::derived::image_tokens(req.width, req.height),
                    )?
                } else {
                    None
                };
                if let Some(overflow) = overflow {
                    warnings.push(gen_core::GenerationWarning {
                        code: crate::memory_strategy::PREFIX_KV_CACHE_OVERFLOW_WARNING.to_owned(),
                        message: format!(
                            "{}: prefix_kv_cache=cached was selected, but this request's caches \
                             would hold {} bytes against the {} bytes priced for it (the prompt \
                             or a reference exceeds the priced prefix bound); rendering prefill",
                            self.variant.id(),
                            overflow.actual_bytes,
                            overflow.priced_bytes
                        ),
                    });
                }
                let loop_mode = if kv_cache.is_cached() && overflow.is_none() {
                    KvCacheMode::On
                } else {
                    KvCacheMode::Off
                };
                // The condition latents: both branches share one reference encode.
                let reference_latents = encode_references(&heavy.vae, &references)?;
                let conditioning = (!references.is_empty()).then(|| ReferenceConditioning {
                    latents: &reference_latents,
                    layout: &pos.layout,
                    negative_layout: neg.as_ref().map(|neg| &neg.layout),
                });
                if let Some(run) = run {
                    run.begin_rendering()?;
                }
                // ONE decode per image either way — upstream always decodes four channels and
                // has no transparency flag — so this branch chooses only whether the alpha is
                // carried (`Rgba`) or applied over white (`Rgb`, the default). The latents,
                // the denoise and the VAE call are identical.
                let rgba_out = req.output_channels == OutputChannels::Rgba;
                let mut images = Vec::with_capacity(if rgba_out { 0 } else { req.count as usize });
                let mut rgba_images =
                    Vec::with_capacity(if rgba_out { req.count as usize } else { 0 });
                for i in 0..req.count {
                    if let Some(run) = run {
                        run.reset_image_denoise_phases();
                    }
                    let seed = params.base_seed.wrapping_add(i as u64);
                    let latents = create_noise(seed, phase0_w, phase0_h, channels)?;
                    let mut report = DenoiseReport::default();
                    let latents = denoise_reported_with_pag_and_run(
                        DenoiseInputs {
                            transformer: &heavy.transformer,
                            sigmas: &params.sigmas,
                            latents,
                            prompt_embeds: &pos.text,
                            negative_embeds: neg.as_ref().map(|neg| &neg.text),
                            true_cfg_scale: params.true_cfg,
                            width: req.width,
                            height: req.height,
                            sampler: req.sampler.as_deref(),
                            seed,
                            cancel: &req.cancel,
                            references: conditioning.as_ref().map(|c| ReferenceConditioning {
                                latents: c.latents,
                                layout: c.layout,
                                negative_layout: c.negative_layout,
                            }),
                            // The one switch (sc-25903), driven by the request's selection
                            // (sc-25905): `Off` is the prefill path, byte-for-byte.
                            kv_cache: loop_mode,
                        },
                        resolved_pag.as_ref(),
                        run,
                        &mut report,
                        on_progress,
                    )?;
                    if let Some(run) = run {
                        run.finish_image_denoise_phases()?;
                    }
                    ran = Some(PrefixKvCacheReport {
                        path: match report.kv_cache {
                            KvCachePath::Cached => PrefixKvCacheMode::Cached,
                            KvCachePath::Prefill => PrefixKvCacheMode::Prefill,
                        },
                        cache_bytes: report.kv_cache_bytes as u64,
                        overflow,
                    });
                    on_progress(Progress::Decoding);
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    // The denoise's pooled buffers are the wrong sizes for the decoder; shed them
                    // so the decode starts from `active` rather than `active + pool` (a pure
                    // memory hint — the latents are materialized by the sampler's final eval).
                    mlx_rs::memory::clear_cache();
                    let mut decode_body = || -> Result<()> {
                        if rgba_out {
                            rgba_images.push(decode_rgba(
                                &heavy.vae,
                                &latents,
                                req.width,
                                req.height,
                                tiling.as_ref(),
                                Some(&req.cancel),
                            )?);
                        } else {
                            images.push(decode_rgb(
                                &heavy.vae,
                                &latents,
                                req.width,
                                req.height,
                                tiling.as_ref(),
                                Some(&req.cancel),
                            )?);
                        }
                        Ok(())
                    };
                    if let Some(run) = run {
                        run.decode_phase(decode_body)?;
                    } else {
                        decode_body()?;
                    }
                    // The decoded image is host bytes now; nothing the decoder pooled is reused by
                    // the next image's denoise, so return it before the next step or the caller.
                    mlx_rs::memory::clear_cache();
                }
                if let Some(run) = run {
                    run.finish_rendering();
                }
                if rgba_out {
                    Ok(GenerationOutput::ImagesRgba(rgba_images))
                } else {
                    Ok(GenerationOutput::Images(images))
                }
            },
        )?;
        Ok(GenerationReport {
            output: Some(output),
            artifacts: None,
            warnings,
            prefix_kv_cache: ran,
            pag: resolved_pag
                .as_ref()
                .map(|p| p.execution_report(actual_steps)),
            low_res_start: None,
        })
    }
}

/// The derived resident set one request's allocator bounds are built on: `variant`'s resident
/// weights at the loaded tier ([`crate::memory_strategy::derived::resident_weights_for`] — the
/// Turbo checkpoint's bf16 autoencoder is half the base's f32 one) plus the installed adapter
/// overlay.
pub(crate) fn derived_resident_bytes(
    variant: Variant,
    quantize: Option<Quant>,
    overlay_bytes: u64,
) -> u64 {
    crate::memory_strategy::derived::resident_weights_for(
        variant,
        crate::quant::Tier::from_selected(quantize).unwrap_or(crate::quant::Tier::Bf16),
    )
    .resident_total()
    .saturating_add(overlay_bytes)
}

/// The conditioning one render carries, assembled while the text encoder is resident: the
/// positive (and, with true CFG, negative) [`JointBranch`] and the ordered prepared references.
pub(crate) struct ReferenceBranches {
    pub(crate) pos: JointBranch,
    pub(crate) neg: Option<JointBranch>,
    pub(crate) references: Vec<PreparedReference>,
}

/// The render path's whole text-side assembly for `req`: the ordered reference list
/// ([`collect_references`]), host-preprocessed against the snapshot's own Qwen3-VL geometry
/// ([`prepare_conditioning_references`]; empty ⇒ the text-to-image route, unchanged), each
/// prompt encoded with those references and assembled by [`joint_branch`] at the request's size —
/// a different prompt is a different text length, hence one layout per branch. The text rows are
/// forced while the encoder is alive, so a Sequential drop cannot leave an unevaluated graph
/// pointing at freed weights.
///
/// Factored out of `generate` so the edit trainer's equivalence test (sc-24161) compares against
/// the code the render path actually runs, not a re-implementation of it.
pub(crate) fn assemble_reference_branches(
    te: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    req: &GenerationRequest,
    drop: usize,
    use_negative: bool,
) -> Result<ReferenceBranches> {
    let images = collect_references(req)?;
    let references = prepare_conditioning_references(te, &images)?;
    let branch = |prompt: &str| -> Result<JointBranch> {
        let conditioning = te.encode_conditioning(tokenizer, prompt, drop, &references)?;
        joint_branch(&conditioning, &references, req.width, req.height)
    };
    let pos = branch(&req.prompt)?;
    let neg = if use_negative {
        Some(branch(req.negative_prompt.as_deref().unwrap_or(""))?)
    } else {
        None
    };
    match &neg {
        Some(neg) => mlx_rs::transforms::eval([&pos.text, &neg.text])?,
        None => mlx_rs::transforms::eval([&pos.text])?,
    }
    Ok(ReferenceBranches {
        pos,
        neg,
        references,
    })
}

/// Capability-driven request validation: the shared floor (count, size range + 32-px grid,
/// negative/guidance support, sampler/scheduler membership, finiteness, `steps >= 1`) plus the base
/// checkpoint's own `steps >= 2` (the terminal-sigma stretch is undefined at one step). The
/// distilled checkpoint has no stretch, so one step is a real schedule there (`[σ_0, 0]`).
#[cfg(test)]
pub(crate) fn validate_request(
    variant: Variant,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    validate_request_with_schedule(variant, caps, &Schedule::production(variant), req)
}

pub(crate) fn validate_request_with_schedule(
    variant: Variant,
    caps: &Capabilities,
    schedule: &Schedule,
    req: &GenerationRequest,
) -> Result<()> {
    // The reference list first, so a caller sees the actionable message (the `Mask` workaround,
    // the one-to-ten window) rather than the shared floor's generic "unsupported conditioning",
    // and so a bad count is refused before any weight is touched.
    collect_references(req)?;
    caps.validate_request(variant.id(), req)?;
    if req.prompt.trim().is_empty() && req.negative_prompt.is_none() {
        // Upstream renders an empty prompt as a single space; accept it, but a whitespace-only
        // prompt with nothing else is almost certainly a caller bug, so say so.
        return Err(Error::Msg(
            "qwen_image_2_1: the prompt is empty (upstream would condition on a single space)"
                .into(),
        ));
    }
    if let (Variant::Base, Some(steps)) = (variant, req.steps) {
        if steps < 2 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: steps must be >= 2 (got {steps}); the terminal-sigma stretch is undefined at one step"
            )));
        }
    }
    if let Some(low_res) = &req.low_res_start {
        if !req.conditioning.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start conflicts with pass-0 reference/edit conditioning",
                variant.id()
            )));
        }
        let desc = variant.descriptor();
        let executed_steps = req.steps.unwrap_or(schedule.default_steps()) as usize;
        low_res.resolve_for_descriptor(&desc, req, executed_steps, VAE_SCALE_FACTOR)?;
    }
    Ok(())
}

pub fn compile_trajectory_for(
    variant: Variant,
    schedule: &Schedule,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    let desc = variant.descriptor();
    validate_request_with_schedule(variant, &desc.capabilities, schedule, req)
        .map_err(|error| labelled(variant.id(), error))?;
    if req
        .conditioning
        .iter()
        .any(|c| matches!(c, gen_core::Conditioning::ReferenceRgba { .. }))
    {
        return Ok(None);
    }
    let params =
        resolve_run_params(schedule, req).map_err(|error| labelled(variant.id(), error))?;
    let solver = gen_core::sampling::canonical_solver_or_euler(req.sampler.as_deref());
    let executed_steps = params.sigmas.len().saturating_sub(1);
    let pass = gen_core::render::PackedImagePass {
        pixel_stride: VAE_SCALE_FACTOR,
        token_width: TransformerConfig::production().in_channels as u32,
        sampler: solver.name(),
        guidance: if params.use_negative {
            params.true_cfg
        } else {
            1.0
        },
        sigmas: params.sigmas,
        executed_steps,
        capture_sigma: 0.0,
    };
    let plan = if req.conditioning.is_empty() {
        gen_core::RenderPlan::base_image(&desc, req, pass)?
    } else {
        gen_core::RenderPlan::packed_image_trajectory(&desc, req, pass)?
    };
    Ok(Some(plan))
}

pub(crate) fn validate_plan_load_spec(variant: Variant, spec: &LoadSpec) -> Result<()> {
    gen_core::reject_unknown_components(spec, &[], variant.id())?;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(PRECISION_OVERRIDE_REFUSAL.into()));
    }
    if spec.text_encoder.is_some() {
        return Err(Error::Unsupported(
            "qwen_image_2_1: the Qwen3-VL text encoder is loaded from the snapshot's own \
             text_encoder/; LoadSpec::text_encoder substitution is not advertised"
                .into(),
        ));
    }
    if let Some(q) = spec.quantize {
        if !matches!(q, Quant::Q4 | Quant::Q8) {
            return Err(Error::Unsupported(crate::quant::non_affine_quant_refusal(
                q,
            )));
        }
    }
    if matches!(spec.weights, mlx_gen::WeightsSource::File(_)) {
        return Err(Error::Msg(
            "qwen_image_2_1 expects a snapshot directory, not a single file".into(),
        ));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
        || spec.pid.is_some()
    {
        return Err(Error::Unsupported(
            "qwen_image_2_1: control, IP-Adapter, identity, and PiD overlays are not supported"
                .into(),
        ));
    }
    Ok(())
}

pub fn compile_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    validate_plan_load_spec(Variant::Base, spec)
        .map_err(|error| labelled(Variant::Base.id(), error))?;
    compile_trajectory_for(
        Variant::Base,
        &Schedule::production(Variant::Base),
        req,
    )
    .map(|opt| opt.map(|plan| plan.bind_load(spec)))
}

pub fn compile_render_plan_turbo(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    validate_plan_load_spec(Variant::Turbo, spec)
        .map_err(|error| labelled(Variant::Turbo.id(), error))?;
    compile_trajectory_for(
        Variant::Turbo,
        &Schedule::production(Variant::Turbo),
        req,
    )
    .map(|opt| opt.map(|plan| plan.bind_load(spec)))
}

pub const RENDER_PLAN_REGISTRATION: gen_core::RenderPlanRegistration =
    gen_core::RenderPlanRegistration {
        provider_id: MODEL_ID,
        compile: compile_render_plan,
    };

pub const TURBO_RENDER_PLAN_REGISTRATION: gen_core::RenderPlanRegistration =
    gen_core::RenderPlanRegistration {
        provider_id: TURBO_MODEL_ID,
        compile: compile_render_plan_turbo,
    };

pub(crate) fn component_footprint(spec: &LoadSpec) -> gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

mlx_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo;
    footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{OffloadPolicy, WeightsSource};

    fn req(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red fox".into(),
            width,
            height,
            ..Default::default()
        }
    }

    /// A Turbo request's allocator bounds are built on the Turbo checkpoint's resident set — its
    /// bf16 autoencoder — not on the base checkpoint's, at every tier, with the adapter overlay
    /// on top. The base figure is the base table's, unchanged. (The transient budget beside it is
    /// `request_transient_budget_bytes`, the same for both ids: no activation term is
    /// variant-dependent.)
    ///
    /// *Mutation that reds this:* `derived_resident_bytes` pricing every variant from the base
    /// table (`resident_weights(tier)`).
    #[test]
    fn a_request_is_bounded_on_its_own_checkpoints_resident_set() {
        use crate::memory_strategy::derived;
        for (quant, tier) in [
            (None, crate::quant::Tier::Bf16),
            (Some(Quant::Q8), crate::quant::Tier::Q8),
            (Some(Quant::Q4), crate::quant::Tier::Q4),
        ] {
            let base = derived_resident_bytes(Variant::Base, quant, 0);
            let turbo = derived_resident_bytes(Variant::Turbo, quant, 0);
            assert_eq!(base, derived::resident_weights(tier).resident_total());
            assert_eq!(
                base - turbo,
                derived::VAE_BYTES - derived::TURBO_VAE_BYTES,
                "{tier:?}"
            );
            assert_eq!(
                derived_resident_bytes(Variant::Turbo, quant, 4096),
                turbo + 4096
            );
        }
    }

    /// The two load-spec refusals both backends share **by literal** (sc-24114): the exact text
    /// is pinned here and, character for character, in `candle_gen_qwen_image_2_1`'s tests, so
    /// the wording cannot drift on either side without one of the two pins going red. Both are the
    /// typed `Unsupported`, not `Msg`, and both are raised before the snapshot is touched.
    #[test]
    fn precision_and_non_affine_quant_refusals_share_the_candle_wording() {
        assert_eq!(
            PRECISION_OVERRIDE_REFUSAL,
            "qwen_image_2_1: components load at the backend's own compute dtype (the checkpoint's \
             bf16 on MLX, CUDA and Metal; f32 on the candle CPU lane); drop the precision override"
        );
        assert_eq!(
            crate::quant::non_affine_quant_refusal(Quant::Nvfp4),
            "qwen_image_2_1: Nvfp4 is not a Qwen-Image 2.1 affine tier (Q4/Q8)"
        );

        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1-not-on-disk".into()));
        let mut fp32 = spec.clone();
        fp32.precision = Precision::Fp32;
        match load(&fp32) {
            Err(Error::Unsupported(message)) => assert_eq!(message, PRECISION_OVERRIDE_REFUSAL),
            other => panic!(
                "a precision override must be a typed Unsupported, got {:?}",
                other.err()
            ),
        }
        match load(&spec.with_quant(Quant::Nvfp4)) {
            Err(Error::Unsupported(message)) => {
                assert_eq!(
                    message,
                    crate::quant::non_affine_quant_refusal(Quant::Nvfp4)
                );
            }
            other => panic!(
                "a non-affine tier must be a typed Unsupported, got {:?}",
                other.err()
            ),
        }
    }

    /// A request stages when the memory-strategy scope said so OR the load policy is
    /// `Sequential`; a plain request on a `Resident` load does not.
    ///
    /// *Mutation that reds this:* dropping either arm of the `||` in `stage_residency_for`.
    #[test]
    fn stage_residency_is_the_request_selection_or_the_sequential_policy() {
        use mlx_gen::gen_core::GenerationMemory;

        let resident = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let sequential = resident
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential);
        let plain = req(64, 64);
        let mut staged = req(64, 64);
        staged.memory = Some(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        });
        let mut unstaged = req(64, 64);
        unstaged.memory = Some(GenerationMemory::default());

        assert!(!stage_residency_for(&resident, &plain));
        assert!(!stage_residency_for(&resident, &unstaged));
        assert!(stage_residency_for(&resident, &staged));
        assert!(stage_residency_for(&sequential, &plain));
        assert!(stage_residency_for(&sequential, &staged));
    }

    #[test]
    fn descriptor_is_distinct_from_the_2512_route() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_2_1");
        assert_eq!(d.family, "qwen-image-2-1");
        assert_eq!(d.modality, Modality::Image);
        assert_eq!(d.capabilities.max_size, 2752);
        assert_eq!(d.capabilities.min_size, 32);
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 32 }
        );
        assert!(d.capabilities.supports_true_cfg);
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::ReferenceRgba,
                gen_core::ConditioningKind::MultiReference,
            ],
            "the reference/edit route is the same upstream call as text-to-image, and a \
             transparent reference is an ordinary reference that kept its alpha (sc-24111)"
        );
        assert!(
            d.capabilities.supports_alpha_output,
            "the VAE is four-channel on every route behind this descriptor (sc-24111)"
        );
        assert!(
            !d.capabilities.accepts(gen_core::ConditioningKind::Mask),
            "upstream has no mask input; a mask travels as an ordinary extra reference"
        );
        assert_eq!(d.denoiser_output_latent_space.map(|s| s.channels), Some(64));
        assert!(gen_core::registry::model_descriptor_errors(&d).is_empty());
    }

    #[test]
    fn every_preset_validates_and_off_grid_sizes_do_not() {
        let caps = descriptor().capabilities;
        for p in PRESETS {
            assert!(
                validate_request(Variant::Base, &caps, &req(p.width, p.height)).is_ok(),
                "{}",
                p.ratio
            );
        }
        let err = validate_request(Variant::Base, &caps, &req(1000, 1024))
            .unwrap_err()
            .to_string();
        assert!(err.contains("32"), "{err}");
        assert!(validate_request(Variant::Base, &caps, &req(4096, 4096)).is_err());
        assert!(validate_request(Variant::Base, &caps, &req(32, 32)).is_ok());
    }

    #[test]
    fn steps_below_two_and_empty_prompts_are_refused() {
        let caps = descriptor().capabilities;
        let mut r = req(2048, 2048);
        r.steps = Some(1);
        let err = validate_request(Variant::Base, &caps, &r)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
        r.steps = Some(2);
        assert!(validate_request(Variant::Base, &caps, &r).is_ok());
        r.prompt = "   ".into();
        let err = validate_request(Variant::Base, &caps, &r)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prompt is empty"), "{err}");
    }

    #[test]
    fn reference_shapes_are_refused_at_validate_before_any_weight_loads() {
        use mlx_gen::gen_core::{Conditioning, Image};

        let caps = descriptor().capabilities;
        let image = |n: u32| Image {
            width: n,
            height: n,
            pixels: vec![0; (n * n * 3) as usize],
        };
        let mut r = req(2048, 2048);
        assert!(
            validate_request(Variant::Base, &caps, &r).is_ok(),
            "no conditioning is T2I"
        );

        r.conditioning = vec![Conditioning::MultiReference {
            images: (0..11).map(|_| image(8)).collect(),
        }];
        let err = validate_request(Variant::Base, &caps, &r)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at most 10"), "{err}");

        // The `Mask` refusal must be this route's actionable message, not the shared floor's
        // generic "unsupported conditioning" — which is why the reference check runs first.
        r.conditioning = vec![Conditioning::Mask { image: image(8) }];
        let err = validate_request(Variant::Base, &caps, &r)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no mask input"), "{err}");
        assert!(err.contains("extra reference"), "{err}");

        r.conditioning = vec![Conditioning::MultiReference {
            images: (0..10).map(|_| image(8)).collect(),
        }];
        assert!(
            validate_request(Variant::Base, &caps, &r).is_ok(),
            "ten references is the documented boundary"
        );
    }

    #[test]
    fn run_params_follow_upstream_defaults() {
        let s = Schedule::Shifted(SchedulerConfig::production());
        let p = resolve_run_params(&s, &req(2048, 2048)).unwrap();
        assert_eq!(p.sigmas.len(), 41, "40 default steps + the trailing 0");
        assert_eq!(p.true_cfg, 1.0);
        assert!(!p.use_negative);
        let mut r = req(2048, 2048);
        r.negative_prompt = Some("blurry".into());
        r.guidance = Some(3.0);
        let p = resolve_run_params(&s, &r).unwrap();
        assert!(p.use_negative);
        assert_eq!(p.true_cfg, 3.0);
        r.true_cfg = Some(1.0);
        let p = resolve_run_params(&s, &r).unwrap();
        assert!(
            !p.use_negative,
            "true_cfg 1.0 disables guidance even with a negative"
        );
    }

    /// The released Turbo `sample_sigmas` (`Qwen/Qwen-Image-2.1-Turbo` @ `UPSTREAM_TURBO_HF_REVISION`).
    const TURBO_SAVED: [f64; 8] = [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
    ];

    fn turbo_schedule() -> Schedule {
        Schedule::Saved(SavedSigmaSchedule {
            sample_sigmas: TURBO_SAVED.to_vec(),
            num_train_timesteps: 1000,
        })
    }

    fn saved_then_zero() -> Vec<f32> {
        TURBO_SAVED
            .iter()
            .map(|sigma| *sigma as f32)
            .chain([0.0])
            .collect()
    }

    /// The sigma list `generate` hands the sampler (`RunParams::sigmas` goes into
    /// `DenoiseInputs::sigmas` verbatim) for the distilled checkpoint: with no step count, and
    /// with the saved length named explicitly, it is exactly the snapshot's eight saved sigmas and
    /// a trailing 0 — at every resolution, because there is no `mu` shift and no terminal stretch.
    ///
    /// *Mutations that red this:* defaulting to the base 40 steps (`DEFAULT_STEPS`); routing the
    /// saved sigmas through `scheduler::sigmas` (shift + stretch) instead of the resample.
    #[test]
    fn turbo_denoises_over_the_saved_sigmas_by_default_and_at_the_saved_length() {
        let schedule = turbo_schedule();
        assert_eq!(schedule.default_steps(), 8);
        for (width, height) in [(2048, 2048), (2752, 1536), (512, 512), (32, 32)] {
            let mut r = req(width, height);
            let default = resolve_run_params(&schedule, &r).unwrap();
            assert_eq!(
                default.sigmas,
                saved_then_zero(),
                "{width}x{height} default"
            );
            r.steps = Some(8);
            let explicit = resolve_run_params(&schedule, &r).unwrap();
            assert_eq!(
                explicit.sigmas,
                saved_then_zero(),
                "{width}x{height} steps=8"
            );
        }
        // True CFG is off by default and still engages on request, as upstream's pipeline allows.
        let mut r = req(2048, 2048);
        let p = resolve_run_params(&schedule, &r).unwrap();
        assert_eq!(p.true_cfg, 1.0);
        assert!(!p.use_negative);
        r.negative_prompt = Some("blurry".into());
        r.true_cfg = Some(2.0);
        assert!(resolve_run_params(&schedule, &r).unwrap().use_negative);
    }

    /// Twelve steps on the distilled checkpoint is the 12-point resample of the saved curve
    /// (`scheduler::resample_saved_sigmas`, pinned value for value in `scheduler::tests`), and the
    /// same request on the base checkpoint is still its own resolution-shifted schedule.
    ///
    /// *Mutation that reds this:* ignoring `req.steps` on the saved-sigma arm (upstream's
    /// behaviour — the steps knob would be dead).
    #[test]
    fn turbo_resamples_other_step_counts_and_the_base_schedule_is_unchanged() {
        let mut r = req(2048, 2048);
        r.steps = Some(12);
        let turbo = resolve_run_params(&turbo_schedule(), &r).unwrap();
        assert_eq!(
            turbo.sigmas,
            [
                1.0, 0.98563533, 0.970362, 0.95418, 0.9358107, 0.9161107, 0.89508, 0.861792,
                0.79827667, 0.704534, 0.5112233, 0.27637867, 0.0
            ]
        );

        let cfg = SchedulerConfig::production();
        let base = resolve_run_params(&Schedule::Shifted(cfg.clone()), &r).unwrap();
        assert_eq!(
            base.sigmas,
            scheduler::sigmas_for_image(&cfg, 12, 2048, 2048).unwrap()
        );
        assert_eq!(base.sigmas.len(), 13);
        assert!((base.sigmas[11] - 0.02).abs() < 1e-6, "terminal stretch");
        assert_ne!(base.sigmas, turbo.sigmas);
        // The base default is still upstream's 40.
        assert_eq!(
            Schedule::Shifted(SchedulerConfig::production()).default_steps(),
            40
        );
    }

    /// A curated `scheduler` name on the distilled checkpoint goes through the same shared seam
    /// as the base one (`resolve_flow_schedule`), built over an **unshifted** flow model
    /// (`mu = ln(shift) = 0`): it replaces the saved curve rather than re-spacing it. A name the
    /// seam does not recognise keeps the saved sigmas (`validate` refuses such a name before it
    /// gets here; the fallback is the seam's own never-hard-fail rule).
    ///
    /// *Mutation that reds this:* passing the base resolution `mu` to the curated builder.
    #[test]
    fn turbo_curated_schedulers_are_built_unshifted_and_native_names_keep_the_saved_sigmas() {
        let schedule = turbo_schedule();
        let mut r = req(2048, 2048);
        for native in ["flow_match_euler", "not-a-scheduler"] {
            r.scheduler = Some(native.into());
            assert_eq!(
                resolve_run_params(&schedule, &r).unwrap().sigmas,
                saved_then_zero(),
                "{native}"
            );
        }
        r.scheduler = Some("simple".into());
        let curated = resolve_run_params(&schedule, &r).unwrap().sigmas;
        assert_eq!(
            curated,
            mlx_gen::resolve_flow_schedule(Some("simple"), 0.0, 8, &saved_then_zero())
        );
        assert_ne!(curated, saved_then_zero());
    }

    /// One step is a real schedule on the distilled checkpoint (`[1, 0]`, no terminal stretch to
    /// divide by) and stays refused on the base one.
    #[test]
    fn turbo_accepts_one_step_and_the_base_checkpoint_still_refuses_it() {
        let caps = descriptor_turbo().capabilities;
        let mut r = req(2048, 2048);
        r.steps = Some(1);
        validate_request(Variant::Turbo, &caps, &r).unwrap();
        assert_eq!(
            resolve_run_params(&turbo_schedule(), &r).unwrap().sigmas,
            [1.0, 0.0]
        );
        let err = validate_request(Variant::Base, &descriptor().capabilities, &r)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
        r.steps = Some(0);
        let err = validate_request(Variant::Turbo, &caps, &r)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("qwen_image_2_1_turbo: steps must be >= 1"),
            "{err}"
        );
    }

    /// The Turbo descriptor is the base descriptor under its own id with Turbo-tuned PAG
    /// defaults (`default_stop = 0.75`, `recommended_scale = 1.2`).
    #[test]
    fn turbo_descriptor_mirrors_the_base_descriptor_under_its_own_id() {
        let (base, turbo) = (descriptor(), descriptor_turbo());
        assert_eq!(turbo.id, "qwen_image_2_1_turbo");
        assert_eq!(base.id, "qwen_image_2_1");
        assert_eq!(
            base.capabilities.pag,
            Some(gen_core_family::qwen_image_2_1::pag::qwen_image_2_1_pag_capability())
        );
        assert_eq!(
            turbo.capabilities.pag,
            Some(gen_core_family::qwen_image_2_1::pag::qwen_image_2_1_turbo_pag_capability())
        );
        let mut turbo_normalized = turbo;
        turbo_normalized.id = base.id;
        turbo_normalized.capabilities.pag = base.capabilities.pag.clone();
        assert_eq!(format!("{turbo_normalized:?}"), format!("{base:?}"));
        assert!(gen_core::registry::model_descriptor_errors(&descriptor_turbo()).is_empty());
        assert_eq!(Variant::from_id("qwen_image_2_1"), Some(Variant::Base));
        assert_eq!(
            Variant::from_id("qwen_image_2_1_turbo"),
            Some(Variant::Turbo)
        );
        assert_eq!(Variant::from_id("qwen_image"), None);
    }

    #[test]
    fn load_rejects_single_file_and_unwired_overlays() {
        let err = load(&LoadSpec::new(WeightsSource::File(
            "/tmp/q.safetensors".into(),
        )))
        .err()
        .expect("a single file is refused")
        .to_string();
        assert!(err.contains("snapshot directory"), "{err}");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_text_encoder(WeightsSource::Dir("/elsewhere".into()));
        let err = load(&spec)
            .err()
            .expect("substitution is refused")
            .to_string();
        assert!(err.contains("text_encoder"), "{err}");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Q8);
        let err = load(&spec).err().expect("missing snapshot").to_string();
        assert!(!err.contains("not an MLX affine tier"), "{err}");
        assert!(
            load(
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
                    .with_offload_policy(OffloadPolicy::Sequential)
            )
            .is_err(),
            "a missing snapshot fails at the tokenizer, for both policies"
        );
    }

    #[test]
    fn descriptors_advertise_qwen21_pag_and_validate_bounds() {
        let base = descriptor();
        let turbo = descriptor_turbo();
        let zero = GenerationRequest {
            prompt: "a red fox".into(),
            pag: Some(mlx_gen::gen_core::PagConfig {
                scale: 0.0,
                layers: Some(vec![16]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_request(Variant::Base, &base.capabilities, &zero).is_ok());
        assert!(mlx_gen::gen_core::resolve_pag(
            base.capabilities.pag.as_ref(),
            MODEL_ID,
            zero.effective_pag().unwrap()
        )
        .unwrap()
        .is_none());

        let oob = GenerationRequest {
            prompt: "a red fox".into(),
            pag: Some(mlx_gen::gen_core::PagConfig {
                scale: 0.0,
                layers: Some(vec![32]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_request(Variant::Turbo, &turbo.capabilities, &oob).is_err());

        let resolved_turbo = mlx_gen::gen_core::resolve_pag(
            turbo.capabilities.pag.as_ref(),
            TURBO_MODEL_ID,
            Some(&mlx_gen::gen_core::PagConfig {
                scale: 1.2,
                ..Default::default()
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved_turbo.active_steps(8), 6);
        resolved_turbo
            .execution_report(8)
            .validate_with_capability(turbo.capabilities.pag.as_ref().unwrap())
            .unwrap();
    }

    #[test]
    fn tier0_render_plan_and_low_res_start_conformance() {
        use mlx_gen::gen_core::{Conditioning, Image, LowResStartSpec};

        let base_desc = descriptor();
        let turbo_desc = descriptor_turbo();
        assert!(base_desc.capabilities.supports_low_res_start);
        assert!(turbo_desc.capabilities.supports_low_res_start);

        let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "/tmp/qwen-image-2-1",
        )));
        let base_req = GenerationRequest {
            prompt: "a red fox".into(),
            width: 1024,
            height: 1024,
            steps: Some(20),
            low_res_start: Some(LowResStartSpec::new(6, Some(0.5))),
            ..Default::default()
        };
        let base_plan = compile_render_plan(&spec, &base_req)
            .expect("base plan compiles")
            .expect("some base plan");
        assert_eq!(base_plan.phase_output_size(0), (512, 512));
        assert_eq!(base_plan.phase_output_size(1), (1024, 1024));
        assert_eq!(base_plan.denoise_spans(), vec![0..6, 6..20]);

        let turbo_req = GenerationRequest {
            prompt: "a red fox".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            low_res_start: Some(LowResStartSpec::new(3, Some(0.5))),
            ..Default::default()
        };
        let turbo_plan = compile_render_plan_turbo(&spec, &turbo_req)
            .expect("turbo plan compiles")
            .expect("some turbo plan");
        assert_eq!(turbo_plan.phase_output_size(0), (512, 512));
        assert_eq!(turbo_plan.phase_output_size(1), (1024, 1024));
        assert_eq!(turbo_plan.denoise_spans(), vec![0..3, 3..8]);

        // Reference / edit conditioning must refuse low_res_start with typed Unsupported.
        let edit_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![128; 64 * 64 * 3],
                },
                strength: None,
            }],
            ..base_req
        };
        assert!(matches!(
            compile_render_plan(&spec, &edit_req),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            compile_render_plan_turbo(&spec, &edit_req),
            Err(gen_core::Error::Unsupported(_))
        ));
    }
}

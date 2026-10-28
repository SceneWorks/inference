//! The **`minimax_h3` generator** (sc-17147): prompt → real video frames plus a real synchronized
//! stereo soundtrack.
//!
//! `t2va` on the `transformer` partition. The checkpoint is guidance-distilled — no negative prompt,
//! no guidance scale, one transformer forward per step.
//!
//! # Phases, and why nothing is held resident
//!
//! The three heavy components are **66.7 GB** of Qwen3-VL-32B text encoder, **62 GB** of DiT (38.7 GB
//! once the AdaLN projections are evicted — sc-17145) and **10 GB** of video VAE. Holding any two of
//! them at once does not fit a sensible budget, so `MiniMaxH3::generate_impl` builds each phase's
//! component, uses it, drops it and drains MLX's allocator cache before the next — the staged shape
//! the LTX provider uses, applied to a much larger spread. `load` therefore holds **paths**, not
//! tensors: it validates that every partition the render needs is present and defers the reads.
//!
//! # Geometry: `frames` is a lattice point, `duration` is a request
//!
//! Two request fields reach the same quantity and they are deliberately treated differently:
//!
//! * **`frames`** names a point on the model's own `17n + 5` lattice. An off-lattice value is
//!   **rejected** ([`crate::pipeline::resolve_geometry`]) — SceneWorks normalizes dimensions
//!   upstream, so a gate that silently refits is a gate that can never be observed to fire, and at
//!   video scale a refit is three quarters of a second of picture the caller never asked for.
//! * **`duration`** is a continuous seconds value with no lattice of its own, so it is aligned
//!   **upward** to the next legal frame count ([`crate::pipeline::align_frames_for_duration`]) — the
//!   reference ships `align_num_frames` for exactly this — and the alignment is bounded by the same
//!   5-15 s range rather than being a way around it.
//!
//! When both are present `frames` wins, because it is the exact one.

use std::path::{Path, PathBuf};

use mlx_rs::Dtype;

use mlx_gen::gen_core::{
    effective_reference_image_short_edge, reject_unknown_components,
    validate_reference_image_short_edge, Capabilities, Conditioning, ConditioningKind,
    GenerationOutput, GenerationRequest, Generator, KeyframeRef, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Quant, StepSupport, WeightsSource,
};
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::weights::Weights;
use mlx_gen::{default_seed, Error, Result};
use mlx_gen_boogu::vision::VisionTower;

use crate::audio_config::{MiniMaxH3AudioVaeConfig, AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::audio_vae::MiniMaxH3AudioVae;
use crate::denoise::{JointSchedule, AUDIO_SIGMA_SHIFT, MIN_INFERENCE_STEPS, VIDEO_SIGMA_SHIFT};
use crate::dit::adaln::AdaLnResidency;
use crate::dit::model::{JointDit, MiniMaxH3Dit};
use crate::pipeline::{
    fit_audio_to_video, fl2va_layout, frames_to_images, initial_latents, prepend_condition_rows,
    render_latents_with_observer, resolve_geometry, revert_pixel_normalization, t2va_layout,
    RequestGeometry, MAX_CANVAS_EDGE, MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, SPATIAL_STRIDE,
};
use crate::reference::{Ref2VaReference, Ref2VaReferences, VideoReference};
use crate::text_encoder::{
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, LM_PREFIX, VISION_PREFIX,
};
use crate::vae::MiniMaxH3VideoVae;

/// The published provider id.
pub const MODEL_ID: &str = "minimax_h3";

/// Stable worker-facing boundary for the FL2VA Qwen3-VL presentation. The wording is deliberately
/// specific: SceneWorks uses the `MiniMax-H3 MLX I2V` prefix to distinguish this known
/// process-poisoning path from T2VA, Ref2VA, other MLX families, and ordinary allocation errors.
const FL2VA_GROUNDED_QWEN3_VL_PHASE: &str =
    "MiniMax-H3 MLX I2V grounded Qwen3-VL vision/text conditioning";

/// Stable worker-facing boundary for the FL2VA keyframe VAE presentation.
const FL2VA_KEYFRAME_VAE_PHASE: &str = "MiniMax-H3 MLX I2V keyframe VAE conditioning";

/// Attach an actionable FL2VA phase to device/loader failures without degrading the typed request
/// outcomes the shared generator contract relies on. In particular, cancellation, unsupported
/// conditioning, and geometry refusal must not become generic backend failures merely because the
/// request carries a keyframe.
fn in_fl2va_phase<T>(phase: &'static str, result: Result<T>) -> Result<T> {
    result.map_err(|error| match error {
        Error::Mlx(exception) => Error::Msg(format!("{phase} failed: MLX op failed: {exception}")),
        Error::Msg(message) => Error::Msg(format!("{phase} failed: {message}")),
        typed => typed,
    })
}

/// Model evaluations a request runs when it names no step count.
///
/// The reference declares no default; 50 is what the sc-17242 spike rendered at and what the model
/// card's own examples use. **Evaluations**, not `MiniMaxH3Scheduler`'s `num_inference_steps` —
/// that count includes the terminal `σ = 0` the model is never evaluated at, so the schedule is
/// built with `steps + 1` (see [`crate::denoise::schedule`]).
pub const DEFAULT_STEPS: u32 = 50;

/// Upper bound on requested steps. A provider-local guard: every step is a full 33 B forward and the
/// AdaLN cache grows with the distinct-timestep count, so an unbounded value is a resource hazard
/// rather than a slow render.
pub const MAX_STEPS: u32 = 200;

/// Quantization group size the shared vision tower is built with — the same value every other
/// consumer of `mlx-gen-boogu`'s tower passes.
const VISION_GROUP_SIZE: i32 = 64;

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        // Both `None`, and both deliberate (sc-17137 main sync):
        //   * `encoder_contract`: like every shipped video provider (Wan, LTX, Mochi, SVD,
        //     SeedVR2), the text encoder is not advertised as substitutable.
        //   * `denoiser_output_latent_space`: the denoiser's video half is a 24-channel latent on
        //     the 17-frame clip lattice — `ceil(clip/vae_ratio_t) - token_drop` tokens per clip,
        //     decoded twice with a seam cross-fade (`crate::chunking`). No `LatentTemporalLaw`
        //     variant expresses that mapping, and per the field's contract `None` makes every
        //     external decoder fail closed rather than let a channel-count match imply
        //     compatibility. The paired ViT decoder + audio VAE are internal to this crate.
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "minimax_h3",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch exists anywhere in the checkpoint.
            supports_negative_prompt: false,
            // **One kind covers all three keyframe shapes.** `Keyframe` carries a `frame_idx`, and
            // first-only / last-only / first+last are one, one and two of them — see
            // [`keyframe_anchors`] for why last-frame-only is a payload shape rather than a mode
            // of its own.
            //
            // The three `ref2va` kinds (sc-17149) are the omni-reference surface on
            // `transformer_ref`. They are three kinds rather than one because gen-core has no
            // heterogeneous-reference variant and `conditioning` is an **ordered** `Vec`, so the
            // request's own order carries the semantics `ref2va` needs — see
            // [`crate::reference`] on why order is not incidental, and [`request_references`] for
            // why the video one is `ReferenceVideo` and not `VideoClip` or `VideoSync`.
            //
            // `VideoClip` is deliberately **not** advertised. Nothing here reads it, and leaving it
            // in the allowlist would let an in-context clip through to a model that has no in-context
            // clip mechanism; default-deny turns it into the typed `Error::Unsupported` instead.
            conditioning: vec![
                ConditioningKind::Keyframe,
                ConditioningKind::Reference,
                ConditioningKind::ReferenceVideo,
                ConditioningKind::ReferenceAudio,
            ],
            // sc-18724. Every attention/FFN projection in the 50-block stack and the 2-block token
            // refiner is an `AdaptableLinear` on **every** tier, so the lightx2v turbo LoRAs fold as
            // forward-time residuals with no dequantization — see [`crate::adapters`], which also
            // owns the per-file alpha resolution those checkpoints need.
            supports_lora: true,
            // LoKr routes through the shared LyCORIS seam on the same host
            // ([`crate::adapters::apply_minimax_h3_adapters`]); no MiniMax-H3 LoKr is published, but
            // the surface is real rather than declared, so this is not advertising a stub.
            supports_lokr: true,
            // The tiers exist and load (sc-17150) — packed offline by `crate::convert` and staged
            // as the [`DIT_COMPONENT`]. `spec.quantize` is reconciled against the staged tier's
            // marker rather than triggering a load-time quantize; see [`reconcile_tier`].
            supported_quants: &[Quant::Q4, Quant::Q8],
            min_size: SPATIAL_STRIDE,
            // The widest edge `crate::keyframe::resolve_canvas_size` can put a picture on, at the
            // 4:1 aspect ceiling. A per-edge cap is NOT the area budget — `CANVAS_MAX_PIXELS` is
            // checked as a product by `resolve_geometry` and still refuses 1536x1536 / 1344x1344,
            // which sit inside this ceiling on both edges. See `MAX_CANVAS_EDGE` (sc-17152).
            max_size: MAX_CANVAS_EDGE,
            max_count: 1,
            // The provider's step bound, advertised rather than hidden (sc-19559). `validate`
            // below still refuses the same counts; this is what makes the bound readable
            // weights-free, which matters most for a model whose text encoder is ~53 GB.
            supported_steps: StepSupport::Range {
                min: 1,
                max: MAX_STEPS,
            },
            mac_only: true,
            // The TE → DiT → VAE phase order is hardcoded in `generate_impl`, not a load-time
            // default a `GenerationMemory` block could switch off, so every generate stages
            // physically whatever residency the request selects. This is why
            // [`crate::memory_strategy`] models the floor as `PhaseEnvelope` — `max(TE, DiT, VAE)`
            // and never the sum, because the three are never co-resident. Independent of
            // `supports_sequential_offload` above: the staging is unconditional, but no *selectable*
            // Sequential control is wired onto the shared residency seam.
            unconditionally_engages_staged_residency: true,
            // No enhancement surface exists in this crate: `enhance_prompt` is ignored, so
            // advertising it would describe a toggle a UI must not present as effective.
            supports_prompt_enhancement: false,
            // The audio surface describes a *selectable* audio request (voice / language / rate),
            // which a video model has none of — the soundtrack rides `GenerationOutput::Video`.
            audio_sample_rates: vec![],
            ..Default::default()
        },
    }
}

/// Which MiniMax-H3 **task** a request is, and therefore which DiT partition it loads.
///
/// # The two checkpoints are structurally indistinguishable
///
/// `transformer/` and `transformer_ref/` ship **the same `config.json`** (byte-identical) and **the
/// same 638 tensor names**. Nothing structural separates them — not a shape check, not a
/// key-mapping proof, not a config diff. Only the *values* differ. That is the same hazard class
/// [`crate::layout`] documents for the gated-FFN half-swap: a port that loads the wrong one gets a
/// model that runs, produces plausible video, and is wrong.
///
/// So the partition is selected by this typed enum rather than by a bare string at the call site,
/// and `tests/ref2va_checkpoint.rs` pins the selection **against the real bytes** of a tensor that
/// differs between the two (`proj_in.bias`), because that is the only thing that can tell them
/// apart.
///
/// # Why the task is not simply "are there keyframes"
///
/// `t2va` and `fl2va` share `transformer/`; only `ref2va` moves. A request carrying **both**
/// keyframes and references is refused rather than silently resolved to one of them — the two
/// condition the video stream through different mechanisms at different rotary times, and picking
/// one would discard conditioning the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxH3Task {
    /// Text only. `transformer/`.
    T2va,
    /// First / last frame keyframes. `transformer/`.
    Fl2va,
    /// Multi-modal references. **`transformer_ref/`.**
    Ref2va,
}

/// The DiT partition a `t2va` / `fl2va` request loads.
pub const BASE_DIT_PARTITION: &str = "transformer";

/// The DiT partition a `ref2va` request loads.
pub const REFERENCE_DIT_PARTITION: &str = "transformer_ref";

impl MiniMaxH3Task {
    /// The snapshot subdirectory this task's DiT is read from.
    pub fn partition(self) -> &'static str {
        match self {
            Self::T2va | Self::Fl2va => BASE_DIT_PARTITION,
            Self::Ref2va => REFERENCE_DIT_PARTITION,
        }
    }

    /// Whether this task reads the vision tower (shard 14) as part of its presentation.
    ///
    /// `t2va` does not; `fl2va` runs the tower over its keyframes and `ref2va` over its image and
    /// video references. An audio reference contributes **no** vision block, so a `ref2va` request
    /// whose only visual references are absent cannot occur — [`crate::reference`] refuses an
    /// audio-only list precisely so this stays true.
    pub fn needs_vision_tower(self) -> bool {
        matches!(self, Self::Fl2va | Self::Ref2va)
    }

    /// Derive the task from what the request actually conditions on.
    ///
    /// `has_keyframes` and `has_references` are passed rather than read off the request so this
    /// stays a pure, exhaustively testable function — the story requires the selection to be
    /// covered by a test, and a function that reaches into a `GenerationRequest` can only be
    /// tested by building one.
    pub fn resolve(has_keyframes: bool, has_references: bool) -> Result<Self> {
        match (has_keyframes, has_references) {
            (false, false) => Ok(Self::T2va),
            (true, false) => Ok(Self::Fl2va),
            (false, true) => Ok(Self::Ref2va),
            (true, true) => Err(Error::Msg(format!(
                "{MODEL_ID}: a request carries both keyframes and references, which are different \
                 tasks on different checkpoints — `fl2va` pins a literal frame of the generated \
                 clip from `transformer/`, `ref2va` conditions on unpositioned references from \
                 `transformer_ref/`. Send one or the other."
            ))),
        }
    }
}

/// The loaded generator — paths and precision, deliberately no tensors. See the module docs.
pub struct MiniMaxH3 {
    #[cfg(test)]
    pub(crate) fixture: Option<crate::veda_fixture::Fixture>,
    pub(crate) veda_component: Option<PathBuf>,
    veda_report: crate::veda::ReportSink,
    descriptor: ModelDescriptor,
    root: PathBuf,
    /// The directory holding the DiT's `config.json` and shards — `root.join("transformer")` in the
    /// **flat** upstream layout, or the staged [`DIT_COMPONENT`] directory in the **split** layout
    /// where the tiered DiT comes from `SceneWorks/minimax-h3-mlx/{tier}/transformer` while every
    /// shared component still comes from the upstream root. See [`resolve_dit_dir`].
    dit_dir: PathBuf,
    /// The directory holding the text encoder's shards — `root.join("text_encoder")` in the flat
    /// upstream layout, or the staged [`TEXT_ENCODER_COMPONENT`] when a **packed** TE tier is
    /// installed (sc-19120). See [`resolve_text_encoder_dir`].
    text_encoder_dir: PathBuf,
    dtype: Dtype,
    /// Adapter files to fold into whichever DiT partition a render maps (sc-18724). Held as specs,
    /// not as loaded factors: `t2va`/`fl2va` and `ref2va` run different checkpoints, and the DiT is
    /// mapped and released per render, so the install belongs beside the load rather than here.
    adapters: Vec<AdapterSpec>,
    /// The load shape this generator was loaded at — the same value
    /// [`crate::memory_strategy::contract_for`] resolves the contract at, captured so the runtime
    /// admission ([`Self::requested_transformer_window`]) and the declaration cannot disagree:
    /// rung 4 is `Implemented` on a [`LoadShape::DeferredMaterialization`] spec and `Missing` on a
    /// resident one, and a request the contract refuses must be refused here too (sc-18662).
    load_shape: mlx_gen::gen_core::LoadShape,
    /// The provider's memory-strategy contract, resolved at load and **published through
    /// [`Generator::memory_strategy_contract`]** (sc-18650).
    ///
    /// Not an `Option`, unlike the ltx/wan twins: [`crate::memory_strategy::contract_for`] is total
    /// — every load surface this loader admits has a contract — so there is no "loaded route with no
    /// calibrated contract" arm to model, and an `Option` here could only ever express a bug.
    ///
    /// Resolved at [`Self::load_shape`] by construction (`contract_for` reads `spec.load_shape`, the
    /// same field `load` copies into `load_shape`), so the declaration and the runtime admission in
    /// [`Self::requested_transformer_window`] cannot disagree about whether rung 4 exists.
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    /// The numeric tier this route loaded at, for the safety check's tier axis. See
    /// [`crate::memory_strategy::numeric_tier`].
    memory_tier: mlx_gen::gen_core::MemoryNumericTier,
}

/// The staged-component id for the **tiered** DiT directory (sc-17150).
///
/// Both tiered components — this and [`TEXT_ENCODER_COMPONENT`] — are redirected individually so a
/// `q4` install holds one 18.8 GB DiT and one packed text encoder while the two VAEs and the
/// tokenizer still resolve against the upstream root, with no second copy of anything.
///
/// Deliberately **not** in [`ModelDescriptor::required_components`]: it is needed only for a
/// non-`bf16` tier, and a flat upstream snapshot must keep loading with nothing staged. That is the
/// sensenova `distill_lora` convention — a conditionally-needed component is declared to
/// [`reject_unknown_components`] per load, not advertised as a universal requirement.
pub const DIT_COMPONENT: &str = "transformer";

/// The staged-component id for the **tiered text encoder** (sc-19120).
///
/// # Why this component became tiered, when it was explicitly not before
///
/// The Qwen3-VL-32B condition encoder was shipped dense in every tier on the reasoning that a
/// rehost bought nothing: its 14 shards are byte-identical to `Qwen/Qwen3-VL-32B-Instruct`
/// (66_714_912_872 B, 14/14 SHA-256), so a mirror would add only bytes. That reasoning is intact
/// **for a dense mirror** and is superseded for a packed one — a packed tier is a *derived*
/// artifact that upstream does not publish and cannot be sourced from Qwen at any revision.
///
/// What changed is the measurement, not the argument: the ~53 GB process high-water this model was
/// believed to have for activation reasons is
/// [this stage](crate::memory_strategy::CONDITIONING_STAGE_PEAK_BYTES), running **before** the DiT
/// is mapped, and it was dense at every tier — so the DiT's real 40.43 → 11.63 GB tiering was
/// invisible underneath it.
///
/// # It is independent of the DiT's tier, on purpose
///
/// `reconcile_tier` makes `spec.quantize` an assertion about the staged DiT. This component is
/// deliberately **not** held to that same assertion (see `reconcile_text_encoder`): the shipped
/// manifest pairs one dense text encoder with all three DiT tiers, and coupling the two would break
/// every existing install the moment a `q4` DiT was requested. The tier of what is staged here is
/// whatever the `{base}.scales` in it say, and the loaders auto-detect it.
pub const TEXT_ENCODER_COMPONENT: &str = "text_encoder";

/// Resolve the DiT directory: the staged [`DIT_COMPONENT`] if the caller provided one, else
/// `root/transformer` (the flat upstream layout).
fn resolve_dit_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(DIT_COMPONENT) {
        Some(WeightsSource::Dir(p)) => p.clone(),
        // A single file is not a component directory; fall back and let the config probe below
        // produce the actionable "missing transformer/config.json" error against the root.
        _ => root.join(DIT_COMPONENT),
    }
}

/// Resolve the text-encoder directory: the staged [`TEXT_ENCODER_COMPONENT`] if the caller provided
/// one, else `root/text_encoder`.
fn resolve_text_encoder_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(p)) => p.clone(),
        _ => root.join(TEXT_ENCODER_COMPONENT),
    }
}

/// Validate the staged text encoder's quantization marker, if it carries one.
///
/// Unlike [`reconcile_tier`] this does **not** compare against `spec.quantize` — see
/// [`TEXT_ENCODER_COMPONENT`] for why the two components' tiers are independent. What it does check
/// is the one thing a packed component can get wrong in a way no loader can detect: the **group
/// size**.
///
/// [`mlx_gen::quant::packed_bits`] *infers* the bit width from the weight/scales column ratio
/// **assuming** [`crate::text_encoder::GROUP_SIZE`]. A component packed at a different group size
/// therefore does not fail cleanly — Mage's q8 vision tower (packed at 32, read at 64) derived
/// "bits 16" from a perfectly good artifact and was diagnosed as a bad upload (sc-15154). Reading
/// the declared group size and rejecting a mismatch by name turns that into one actionable line.
fn reconcile_text_encoder(te_dir: &Path) -> Result<()> {
    let Some(declared) = mlx_gen::quant::packed_quant_group_size_at(te_dir)? else {
        // Dense, or packed without a declared group size — the loaders auto-detect on `.scales`
        // and `packed_bits` will reject anything that does not divide cleanly at our own width.
        return Ok(());
    };
    if declared != crate::text_encoder::GROUP_SIZE {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: the staged text encoder in {} declares quantization group_size {declared}, \
             but this engine reads packed text-encoder weights at {} — a mismatched group size does \
             not fail cleanly (the inferred bit width comes out legal-looking and wrong), so it is \
             rejected here; re-pack the tier at {}",
            te_dir.display(),
            crate::text_encoder::GROUP_SIZE,
            crate::text_encoder::GROUP_SIZE
        )));
    }
    Ok(())
}

/// Reconcile a caller's `spec.quantize` against the tier actually staged on disk.
///
/// **MiniMax-H3 never quantizes at load.** Every tier ships pre-quantized (`crate::convert`), and
/// the reason is not tidiness: quantizing the DiT at load would materialize its 66_280_430_080
/// dense bytes *and* the growing packed output at once, which is the install-time peak this model
/// cannot afford on any Mac it targets. So `spec.quantize` is an **assertion** about which tier was
/// staged, not an instruction — the mochi / LTX `split_model.json` shape, read here from the
/// `quantization` marker [`mlx_gen::quant::write_quantized_config`] writes into each tier's
/// `config.json`.
///
/// A disagreement is a hard error rather than a silent downgrade, because an unmarked packed
/// component and a genuinely dense one are indistinguishable to a loader that guesses.
fn reconcile_tier(dit_dir: &Path, requested: Option<mlx_gen::gen_core::Quant>) -> Result<()> {
    let packed = mlx_gen::quant::packed_quant_bits_at(dit_dir)?;
    match (packed, requested) {
        (Some(bits), Some(q)) if bits != q.bits() => Err(Error::Msg(format!(
            "{MODEL_ID}: spec.quantize={q:?} (bits {}) disagrees with the staged tier's \
             config.json quantization marker (bits {bits}) in {} — the on-disk tier is \
             authoritative; stage the tier you asked for",
            q.bits(),
            dit_dir.display()
        ))),
        // Packed and agreed (or asserted nothing): the loader builds it packed from `.scales`.
        (Some(_), _) => Ok(()),
        (None, Some(q)) => Err(Error::Unsupported(format!(
            "{MODEL_ID}: spec.quantize={q:?} but {} carries no quantization marker — MiniMax-H3 \
             does not quantize at load (the DiT's 66_280_430_080 dense bytes plus the packed \
             output will not co-reside); stage the pre-quantized tier's `transformer` directory as \
             the '{DIT_COMPONENT}' component",
            dit_dir.display()
        ))),
        (None, None) => Ok(()),
    }
}

/// Reject a request this model cannot serve, before any weight is read.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if let Some(veda) = &req.veda {
        veda.validate(50, req.steps.unwrap_or(DEFAULT_STEPS) as usize)
            .map_err(Error::Msg)?;
    }
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{MODEL_ID}: prompt must not be empty")));
    }
    caps.validate_request(MODEL_ID, req)?;
    if let Some(steps) = req.steps {
        if steps == 0 || steps > MAX_STEPS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: steps must be in 1..={MAX_STEPS}, got {steps}"
            )));
        }
    }
    // The video sigma-shift override (sc-18729), refused **here** rather than 20 minutes downstream
    // where `SigmaSchedule::with_shift` would finally reject it — `validate` is the only gate that
    // runs before the 53 GB text encoder maps. The shared float floor in `caps.validate_request`
    // already rejects NaN/±inf for every knob; this adds the sign, which is specific to a shift
    // (`σ' = s·σ / (1 + (s − 1)·σ)` has a pole and flips sign for `s <= 0`).
    if let Some(shift) = req.scheduler_shift {
        if shift <= 0.0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: scheduler_shift is the video sigma shift and must be positive, got \
                 {shift} (the base checkpoint ships {VIDEO_SIGMA_SHIFT}; the 768p 4-step turbo \
                 variant is trained at 6.0)"
            )));
        }
    }
    // **The `ref2va` reference gate, at the engine boundary** (sc-17149). Every cap — 9 images,
    // 3 clips, 3 audio, 12 combined — plus the audio-is-never-alone rule and the both-tasks-at-once
    // refusal, all before a single weight is read.
    //
    // Enforced here and not only at the API for the reason the whole epic exists: a pipeline handed
    // conditioning it cannot consume does not fail, it renders a plausible clip that silently
    // ignored the thirteenth reference. `Ref2VaReferences` is constructible only through its
    // validating constructor, so the render path cannot reach a reference list that skipped this.
    let references = request_references(req)?;
    MiniMaxH3Task::resolve(!req.keyframes().is_empty(), references.is_some())?;

    // sc-23402 — the reference-image short edge, refused here for the same reason every gate above
    // is: `validate` is the only thing that runs before the 53 GB text encoder maps. The check is
    // unconditional rather than ref2va-only, so a value typed onto a request that carries no image
    // reference is still caught rather than silently inert.
    validate_reference_image_short_edge(MODEL_ID, req)?;

    // sc-19571 — the conditioning-strength refusal runs at the request boundary, not 20 minutes
    // into a render, for the same reason every other gate above does.
    reject_keyframe_strength(&req.keyframes())?;

    // The geometry gate itself — the same call `generate` makes, so `validate` and the render agree
    // by construction rather than by two copies of the lattice arithmetic.
    request_geometry(req).map(|_| ())
}

/// Map a request's keyframes onto MiniMax-H3's two anchor slots, in packed order.
///
/// # The last-frame-only decision (sc-17148)
///
/// The story asks whether last-frame-only should be a **new SceneWorks mode** or an **optional
/// field on `first_last_frame`**. It is the latter, and this function is where that is expressed:
/// there is no third mode, only a `Keyframe` payload whose `frame_idx` names which end it anchors.
///
/// Three reasons, in order of weight:
///
/// 1. **Upstream does not model it as a separate task either.** `MiniMaxH3Blocks._workflow_map`
///    has one `fl2va` entry with two accepted signatures (`image`, or `last_image`), and
///    `before_encoder.py` filters a fixed `(("first", image), ("last", last_image))` pair by
///    presence. All four shapes fall out of one ordered tuple. A third mode would model as
///    disjoint something the reference models as one mechanism with an optional slot.
/// 2. **The information is already carried.** `frame_idx` distinguishes the two ends losslessly. A
///    new mode would add a manifest entry (sc-17158), an allow-list entry and routing across six
///    surfaces (sc-17159), and a Video Studio affordance (sc-17161) — all to re-encode one bit
///    that the existing payload already has.
/// 3. **The genuine difference is a prompting one, not a plumbing one.** Upstream's own
///    `VIDEO_PROMPT_WRITING_GUIDE` names I2VA / L2VA / FL2VA as distinct tasks because they take
///    **different prompt preambles**. That is UI copy and preset territory, and it is well served
///    by a preset over `first_last_frame` rather than by a mode the engine has to branch on.
///
/// What follows for the surfaces this story does not own: `first_last_frame` must accept a payload
/// with **only** the last slot filled. sc-17159 owns making that reachable; the engine accepts it
/// today.
///
/// # The index convention
///
/// `frame_idx` 0 is the first frame. The last is `num_frames - 1` **or** `-1`, because a caller
/// that knows only "the end of the clip" should not have to resolve the frame count first. Any
/// other index is **rejected**: the model has exactly two anchor slots, and silently snapping a
/// mid-clip index to the nearest end would condition on something the caller did not ask for.
/// **Refuse a conditioning strength rather than ignore it** (sc-19571).
///
/// `Conditioning::Keyframe` carries a `strength` that its docs define as a `1 − strength` denoise
/// mask, and SceneWorks exposes it as the first/last-frame conditioning sliders. MiniMax-H3 has no
/// such mask: an anchor is mixed at the checkpoint's own trained-in
/// [`KEYFRAME_NOISE_AUG`](crate::conditioning::KEYFRAME_NOISE_AUG_T) = `0.999` and its rows are told
/// they sit at exactly that `t` (`PackedLayout::row_timesteps`). That number is a property of how
/// the released model was trained, not a knob — `crate::conditioning`'s own docs say conditioning an
/// anchor anywhere else is off-distribution — so there is nothing here a caller-supplied strength
/// could weight without inventing a regime the checkpoint never saw.
///
/// The sc-19571 rule is that a control either works or is refused with a clear error, never
/// silently dropped. This is the refusal, and it names the mechanism so the caller can tell the
/// difference between "unsupported here" and "you typed it wrong".
fn reject_keyframe_strength(keyframes: &[KeyframeRef<'_>]) -> Result<()> {
    for kf in keyframes {
        if kf.strength != 1.0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: keyframe conditioning strength is not supported (got {}) — a \
                 MiniMax-H3 anchor is held at the checkpoint's own trained-in noise augmentation \
                 t = {}, with no denoise mask to weight; use strength 1.0",
                kf.strength,
                crate::conditioning::KEYFRAME_NOISE_AUG_T
            )));
        }
    }
    Ok(())
}

pub(crate) fn keyframe_anchors(
    keyframes: &[KeyframeRef<'_>],
    num_frames: i32,
) -> Result<Vec<crate::dit::positions::KeyframeAnchor>> {
    use crate::dit::positions::KeyframeAnchor;
    reject_keyframe_strength(keyframes)?;
    let mut first = None;
    let mut last = None;
    for kf in keyframes {
        let slot = if kf.frame_idx == 0 {
            &mut first
        } else if kf.frame_idx == -1 || kf.frame_idx == num_frames - 1 {
            &mut last
        } else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a keyframe must anchor the FIRST frame (index 0) or the LAST (index \
                 {} or -1); MiniMax-H3 has exactly two anchor slots and no mid-clip conditioning, \
                 got index {}",
                num_frames - 1,
                kf.frame_idx
            )));
        };
        if slot.is_some() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: two keyframes anchor the same end of the clip (index {})",
                kf.frame_idx
            )));
        }
        *slot = Some(kf.image);
    }
    let mut anchors = Vec::with_capacity(2);
    if first.is_some() {
        anchors.push(KeyframeAnchor::First);
    }
    if last.is_some() {
        anchors.push(KeyframeAnchor::Last);
    }
    Ok(anchors)
}

/// The request's keyframe images in packed order — first, then last.
///
/// Positional with [`keyframe_anchors`]: the caller may supply them in any order, and both
/// functions sort them into the reference's `(first, last)` order so `anchors[i]` always describes
/// `images[i]`.
pub(crate) fn keyframe_images<'a>(
    keyframes: &[KeyframeRef<'a>],
    num_frames: i32,
) -> Result<Vec<&'a mlx_gen::media::Image>> {
    let mut first = None;
    let mut last = None;
    for kf in keyframes {
        if kf.frame_idx == 0 {
            first = Some(kf.image);
        } else if kf.frame_idx == -1 || kf.frame_idx == num_frames - 1 {
            last = Some(kf.image);
        }
    }
    // Re-run the validation so the two functions cannot disagree about what is legal.
    keyframe_anchors(keyframes, num_frames)?;
    Ok(first.into_iter().chain(last).collect())
}

/// Map a request's **ordered** conditioning onto `ref2va`'s reference list.
///
/// Returns `None` when the request carries no reference of any modality, which is what makes this
/// the `t2va` / `fl2va` discriminator as well.
///
/// # Order is taken from the request and never re-sorted
///
/// [`GenerationRequest::conditioning`] is a `Vec`, and `ref2va`'s order is semantic — it fixes the
/// `"<Picture i>"` / `"<Audio j>"` / `"<Video k>"` labels **and** advances the shared rotary clock.
/// So this walks the vector once and preserves position. Grouping by modality first (the obvious
/// implementation, and what [`keyframe_images`] legitimately does for `fl2va`'s two fixed slots)
/// would silently rewrite the request.
///
/// # The `ReferenceVideo` decision (sc-17149), and the two carriers it replaced
///
/// A `ref2va` video reference maps to [`Conditioning::ReferenceVideo`] — a carrier added to
/// gen-core for this task, rather than either of the two video variants that already existed. The
/// story asks for this to be deliberate, so:
///
/// * **`VideoSync` is the wrong meaning.** Its contract is video→audio Foley — "the whole-clip
///   visual condition an audio decoder attends to, to synthesize a synchronized soundtrack for a
///   *silent clip*", and its own docs say the frames "are **not** spliced into a video latent".
///   Both halves are false here: a `ref2va` video reference is VAE-encoded into video latent rows,
///   and it conditions a *generated* clip rather than scoring a supplied one. Advertising
///   `VideoSync` would also advertise a Foley capability MiniMax-H3 does not have, and the kind is
///   what routing reads.
/// * **`VideoClip` is the right *mechanism* but the wrong *vocabulary*.** Its latent handling does
///   describe a reference block — VAE-encoded, appended as extra rows, never written by the denoise
///   loop. But its payload is `{frames, frame_idx, strength}`, and a reference can use exactly one
///   of those three fields:
///     * `frame_idx` is a **position in the generated timeline**, which is the defining thing a
///       reference does not have (that is the whole difference from a keyframe);
///     * `strength` is a `1 − strength` denoise mask, and reference rows are fully pinned at the
///       checkpoint's own conditioning timestep ([`crate::denoise::KEYFRAME_NOISE_AUG`] for visual
///       rows, [`crate::denoise::REFERENCE_AUDIO_TIMESTEP`] for audio), never caller-selectable.
///
///   An earlier revision of this function did ride `VideoClip` and rejected both fields unless they
///   were `0` and `1.0`. That shipped a request vocabulary in which two of three fields were traps,
///   and it still could not carry the two things a reference actually needs — see below.
///
/// # What `VideoClip` could not carry, and why that mattered more than the traps
///
/// * **The clip's own frame rate.** [`VideoReference::fps`] is required data, not a hint:
///   MiniMax-H3 resamples every reference onto its own 24 fps by dropping and duplicating whole
///   frames, so a rate that was lost is a reference conditioned on **at the wrong speed with
///   nothing to raise about it**. `VideoClip` has no rate, so the old mapping read the
///   request-level `req.fps` — which [`request_geometry`] independently *rejects* unless it is
///   exactly [`crate::denoise::MINIMAX_H3_FPS`]. Between them, a reference's rate could only ever
///   resolve to 24.0: a 30 fps reference clip was silently treated as 24 fps, which is precisely
///   the failure [`crate::reference`] says the field exists to prevent. `req.fps` is the rate of
///   the *generated output*; a reference's is the rate of *supplied input media*, and for a
///   reference — which does not bind the output geometry at all — those are different quantities.
/// * **The clip's own soundtrack.** [`VideoReference::audio`] is conditioned on as that reference's
///   own, rotary-aligned with its video rows and sharing their origin. `VideoClip` has no audio
///   field, so a soundtrack could only arrive as a separate [`Conditioning::ReferenceAudio`] — legal,
///   but a *standalone* reference with its own rotary slot, which also consumes one of the
///   [`crate::reference::MAX_AUDIO_REFERENCES`] cap slots a video's own soundtrack does not. A
///   different request, silently substituted for the one the caller meant.
///
/// [`Conditioning::ReferenceVideo`] carries `{frames, fps, audio}` — the engine type's own shape —
/// so both reach the packer and neither field-trap exists to reject. A standalone
/// [`Conditioning::ReferenceAudio`] remains available and remains a genuinely different request.
///
/// The SceneWorks-side payload and mode-reachability work rides sc-17160 / sc-17159.
pub(crate) fn request_references(req: &GenerationRequest) -> Result<Option<Ref2VaReferences>> {
    let mut refs: Vec<Ref2VaReference> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => {
                refs.push(Ref2VaReference::Image(image.clone()));
            }
            Conditioning::ReferenceAudio { audio, .. } => {
                refs.push(Ref2VaReference::Audio(crate::reference::AudioReference {
                    audio: audio.clone(),
                }));
            }
            Conditioning::ReferenceVideo { frames, fps, audio } => {
                refs.push(Ref2VaReference::Video(VideoReference {
                    frames: frames.clone(),
                    // The clip's own rate, carried by the variant. Range-checked by
                    // `Ref2VaReferences::new`, which owns every ref2va rule at the engine boundary.
                    fps: f64::from(*fps),
                    audio: audio.clone(),
                }));
            }
            // Keyframes belong to `fl2va` and are resolved by `keyframe_anchors`; everything else
            // is refused upstream by `Capabilities::accepts`, which default-denies any kind this
            // descriptor does not advertise.
            _ => {}
        }
    }
    if refs.is_empty() {
        return Ok(None);
    }
    Ref2VaReferences::new(refs).map(Some)
}

/// Resolve the request's geometry: the canvas, and the frame count from `frames` (exact) or
/// `duration` (aligned).
pub(crate) fn request_geometry(req: &GenerationRequest) -> Result<RequestGeometry> {
    // The shared resolver the Candle engine and allocation admission (`gen_core::veda::workspace`)
    // use, so the three can never resolve one request to different frame counts (sc-26691).
    let frames = mlx_gen::gen_core::h3_memory::resolve_request_frames(req.frames, req.duration)?;
    let frames = i32::try_from(frames)
        .map_err(|_| Error::Msg(format!("{MODEL_ID}: frames {frames} does not fit an i32")))?;
    if let Some(fps) = req.fps {
        if f64::from(fps) != crate::denoise::MINIMAX_H3_FPS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the released model generates at {} fps only, got {fps}",
                crate::denoise::MINIMAX_H3_FPS
            )));
        }
    }
    resolve_geometry(req.width, req.height, frames)
}

/// Load the generator from a `MiniMaxAI/MiniMax-H3` snapshot root.
///
/// Validates that every partition the `t2va` render reads is present — `transformer/`,
/// `text_encoder/`, `tokenizer/`, `vae/`, `audio_vae/` — and reads none of them. An absent partition
/// is a **load-time** error with the missing path named; this loader never derives a cache location
/// and never self-fetches (epic 13657).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: expected a MiniMax-H3 snapshot directory (the dir holding \
                 `transformer/`), not a single file"
            )))
        }
    };
    reject_unknown_components(
        spec,
        &[
            DIT_COMPONENT,
            TEXT_ENCODER_COMPONENT,
            crate::veda::contract::COMPONENT,
        ],
        MODEL_ID,
    )?;
    // Adapter paths are probed at load so a mistyped one fails here rather than 20 minutes into a
    // render, after the DiT is mapped. The factors themselves are read and installed per render,
    // onto the task's own DiT (sc-18724) — `t2va`/`fl2va` and `ref2va` are different checkpoints.
    for adapter in &spec.adapters {
        if !adapter.path.is_file() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: adapter {} does not exist",
                adapter.path.display()
            )));
        }
        // Two shared-spec knobs no MiniMax-H3 path can honor. Surfaced rather than silently ignored,
        // per their own contract: `pass_scales` is LTX's per-distilled-stage strength and this
        // denoise is single-pass, `moe_expert` addresses a dual-expert Wan MoE and this DiT is
        // single-stream.
        if adapter.pass_scales.is_some() {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} sets per-pass scales, which only LTX's two-stage denoise \
                 has; this model runs one transformer pass per step — use a uniform `scale`",
                adapter.path.display()
            )));
        }
        if adapter.moe_expert.is_some() {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} targets a MoE expert; this DiT is single-stream and has \
                 none",
                adapter.path.display()
            )));
        }
    }
    // The DiT is the one tiered component, so it is probed at its resolved location rather than
    // under the root — a split install has no `root/transformer` at all.
    let dit_dir = resolve_dit_dir(&root, spec);
    let dit_config = dit_dir.join("config.json");
    if !dit_config.is_file() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: missing {} — stage the tier's `transformer` directory as the \
             '{DIT_COMPONENT}' component, or point at a snapshot root that holds `transformer/`",
            dit_config.display()
        )));
    }
    reconcile_tier(&dit_dir, spec.quantize)?;
    // `ref2va` is a first-class task of this engine, not an optional extra, so its checkpoint is
    // probed at load like every other partition. A snapshot missing it fails here, naming the path
    // — rather than 20 minutes into a render when the reference arm finally reaches for it.
    //
    // Probed as `dit_dir`'s SIBLING, not under the root: the reference DiT is tiered exactly like
    // the base one (`crate::convert` pre-quantizes `transformer/` or `transformer_ref/` alike), so
    // on a split install it sits at `{tier}/transformer_ref` and there is no `root/transformer_ref`
    // to find. See [`MiniMaxH3::task_dit_dir`], which resolves the same way at denoise time.
    let reference_dit = dit_dir.with_file_name(REFERENCE_DIT_PARTITION);
    let reference_config = reference_dit.join("config.json");
    if !reference_config.is_file() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: missing {} — `ref2va` reads its own DiT partition, which is tiered \
             alongside the base one; stage the tier's `{REFERENCE_DIT_PARTITION}` directory next \
             to its `{BASE_DIT_PARTITION}`, or point at a snapshot root that holds \
             `{REFERENCE_DIT_PARTITION}/`",
            reference_config.display()
        )));
    }
    reconcile_tier(&reference_dit, spec.quantize)?;
    // The text encoder is the second tiered component (sc-19120), so it is probed at its resolved
    // location like the DiT and not under the root: a split install that stages a packed TE has no
    // `root/text_encoder` at all.
    let text_encoder_dir = resolve_text_encoder_dir(&root, spec);
    let te_config = text_encoder_dir.join("config.json");
    if !te_config.is_file() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: missing {} — stage the tier's `text_encoder` directory as the \
             '{TEXT_ENCODER_COMPONENT}' component, or point at a snapshot root that holds \
             `text_encoder/`",
            te_config.display()
        )));
    }
    reconcile_text_encoder(&text_encoder_dir)?;
    for (partition, probe) in [
        ("vae", "config.json"),
        ("audio_vae", "config.json"),
        ("tokenizer", "tokenizer.json"),
        // The audio VAE's *constructor* arguments live only in the three FL2VA source documents;
        // the repackaged root config carries none of them (see `MiniMaxH3AudioVaeConfig`).
        ("FL2VA/audio_vae", "config.json"),
        ("FL2VA/audio_vae", "config.yaml"),
        ("FL2VA/audio_vae", "metadata.json"),
    ] {
        let path = root.join(partition).join(probe);
        if !path.is_file() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: missing {} in the snapshot root {}",
                path.strip_prefix(&root).unwrap_or(&path).display(),
                root.display()
            )));
        }
    }
    let dtype = match spec.precision {
        Precision::Fp32 => Dtype::Float32,
        _ => Dtype::Bfloat16,
    };
    // Resolve the same source-aware contract used by registry preflight and retain it on the
    // generator. This reads component metadata, source headers and configs, never tensor payloads.
    let memory_strategy = crate::memory_strategy::contract_for(spec).map_err(Error::from)?;
    let memory_tier = crate::memory_strategy::numeric_tier(spec);
    Ok(Box::new(MiniMaxH3 {
        descriptor: descriptor(),
        #[cfg(test)]
        fixture: None,
        veda_component: match spec.components.get(crate::veda::contract::COMPONENT) {
            Some(WeightsSource::File(path)) => Some(path.clone()),
            _ => None,
        },
        veda_report: Default::default(),
        root,
        dit_dir,
        text_encoder_dir,
        dtype,
        adapters: spec.adapters.clone(),
        load_shape: spec.load_shape,
        memory_strategy,
        memory_tier,
    }))
}

/// Release a component's device memory for real: drop the Rust handle, then drain MLX's allocator
/// cache so the buffers go back to the system rather than migrating active → cache.
///
/// The drain is the **shared retried** one ([`mlx_gen::residency::drain_allocator_cache`]), not a
/// bare `clear_cache()`. sc-17145 measured a single sweep leaving a straggler counted as active in
/// roughly one run in five under load, because the Metal command buffer that used the weights had
/// not retired by the time `eval` returned; every phase boundary in [`MiniMaxH3::generate_impl`]
/// and [`MiniMaxH3::generate_ref2va`] hands 10–67 GB back through this function, so the straggler
/// is a whole component's worth.
fn release<T>(component: T) {
    drop(component);
    mlx_gen::residency::drain_allocator_cache();
}

impl MiniMaxH3 {
    /// The snapshot root this generator reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The resolved DiT directory — `root/transformer` on a flat install, or the staged
    /// [`DIT_COMPONENT`] on a tiered one.
    pub fn dit_dir(&self) -> &Path {
        &self.dit_dir
    }

    /// The DiT directory a given task denoises from.
    ///
    /// `ref2va` reads its own [`REFERENCE_DIT_PARTITION`] checkpoint (sc-17149) and every tier
    /// carries both partitions side by side (sc-17150 — `crate::convert` pre-quantizes
    /// `transformer/` and `transformer_ref/` identically), so the reference partition is resolved
    /// as [`Self::dit_dir`]'s **sibling** rather than against the snapshot root. That is the only
    /// form that works on both layouts: a split install has no `root/transformer_ref` at all, and a
    /// flat one has `dit_dir == root/transformer`, whose sibling is exactly `root/transformer_ref`.
    /// `load` probes the same path, so a snapshot missing it fails at load rather than here.
    fn task_dit_dir(&self, task: MiniMaxH3Task) -> PathBuf {
        match task.partition() {
            BASE_DIT_PARTITION => self.dit_dir.clone(),
            partition => self.dit_dir.with_file_name(partition),
        }
    }

    /// Read only the selected partition's JSON header, before any conditioner/VAE weights.
    /// The resolved tier/sibling directory is the same one the eventual DiT loader uses.
    pub(crate) fn task_dit_config(
        &self,
        task: MiniMaxH3Task,
    ) -> Result<crate::dit::MiniMaxH3DitConfig> {
        let path = self.task_dit_dir(task).join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::Msg(format!("{MODEL_ID}: no task config for selected partition `{}` at {}; reimport with its config or select a partition that provides one", task.partition(), path.display()))
            } else {
                Error::Msg(format!("{MODEL_ID}: cannot read task config for selected partition `{}` at {}: {e}; verify the source and permissions", task.partition(), path.display()))
            }
        })?;
        crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
    }

    /// The block stack's precision.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Map `task`'s DiT partition and fold in every configured adapter (sc-18724).
    ///
    /// The **single** seam both denoise paths go through, so a LoRA cannot be honored on `t2va` and
    /// forgotten on `ref2va`. The install happens here rather than in [`load`] because the two tasks
    /// run different checkpoints and the DiT is mapped and released per render — and it happens
    /// *before* `JointDit::new`, whose `PrecomputeAndEvict` consumes the model. That ordering is
    /// safe: nothing in the published turbo set targets `adaln_proj`, and
    /// [`crate::adapters`] cannot reach it anyway, so the eviction lever is untouched.
    ///
    /// The install is **strict**: an unmatched target, or a spec list that matched nothing at all,
    /// fails the render here rather than producing a plausible one the LoRA barely touched. The
    /// returned report can therefore carry nothing a caller has to act on, and is dropped.
    fn load_task_dit(&self, task: MiniMaxH3Task) -> Result<MiniMaxH3Dit> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return crate::veda_fixture::dit();
        }
        let mut dit = MiniMaxH3Dit::load_dir(self.task_dit_dir(task), self.dtype)?;
        if !self.adapters.is_empty() {
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.adapters)?;
        }
        Ok(dit)
    }

    /// The rung-4 admission for one request (sc-18662): `Some(window)` routes the render through
    /// the deferred loaders ([`MiniMaxH3Dit::load_dir_deferred`],
    /// [`MiniMaxH3TextEncoder::from_dir_deferred`]), `None` is the resident staged path.
    ///
    /// The guards mirror the contract exactly, because **declaration is not reachability in either
    /// direction**: a request the contract's `validate_selection` refuses must be refused here even
    /// when it arrives without going through a request scope, and a request the contract admits
    /// must not be silently downgraded to the resident path.
    ///
    /// That mirror is a **standing obligation on two files**, and it was broken between sc-18662 and
    /// sc-18650: the adapter guard below existed here with no counterpart in the declaration, so an
    /// adapter-carrying deferred load advertised rung 4 `Implemented`, was admitted by the safety
    /// check, opened a request scope, had `stream_transformer_blocks` set from that engagement by
    /// `MemoryProviderContract::generation_memory` — and was then refused right here. Every guard
    /// below now names the declaration clause it mirrors; adding a guard without one reopens that
    /// loop.
    ///
    /// * The generator must have been loaded at [`LoadShape::DeferredMaterialization`] — the rung's
    ///   first shared prerequisite, and half of what
    ///   [`crate::memory_strategy::streamable`] resolves `Implemented` on. On a resident load the
    ///   rung is `Missing` and streaming is a typed refusal, not a fallback.
    /// * Adapters are refused. A window rebuilds each block from the staged directory per step, so
    ///   the forward-time residual factors sc-18724 installs would be dropped from every windowed
    ///   block — a plausible render the adapter barely touched, which is the exact failure
    ///   `apply_minimax_h3_adapters`' strict matching exists to prevent. Mirrored by the other half
    ///   of [`crate::memory_strategy::streamable`], which must live in the declaration because
    ///   `MemoryRunContext` has no adapter axis for a route gate to read.
    /// * The component scope must be [`TransformerComponent::Both`] — the only declared arm.
    ///   `None` defaults to `Dit` by the shared convention (SC-15794), and `Dit` is deliberately
    ///   outside this provider's published domain: declaring the DiT alone would leave the
    ///   conditioning phase, the taller stage at every tier, with no lever (AC3).
    /// * The window size must be inside the published singleton domain
    ///   ([`crate::memory_strategy::TRANSFORMER_WINDOW_SIZE`]) — 1 is the measured residency floor
    ///   on every text-encoder tier, so any other value would advertise a lever that only trades
    ///   memory away.
    ///   (Until sc-17153 this said the parameter was *inert* above 1. It is not: the packed
    ///   encoders spread 2.17-3.12x across `[1, 5, 10, 50]`. The domain is unchanged, its reason is
    ///   not — see that constant.)
    fn requested_transformer_window(&self, req: &GenerationRequest) -> Result<Option<u32>> {
        use mlx_gen::gen_core::LoadShape;
        let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
            return Ok(None);
        };
        if self.load_shape != LoadShape::DeferredMaterialization {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer streaming (rung 4) requires a \
                 LoadShape::DeferredMaterialization load; this generator was loaded resident, \
                 where the rung is declared Missing"
            )));
        }
        if !self.adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer streaming rebuilds blocks from the staged tier per \
                 window, so a configured adapter would be silently dropped from every windowed \
                 block; run adapters on the resident path"
            )));
        }
        let component = memory
            .transformer_window_component
            .unwrap_or(mlx_gen::gen_core::TransformerComponent::Dit);
        if component != mlx_gen::gen_core::TransformerComponent::Both {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer streaming is declared for TransformerComponent::Both \
                 only — got {component:?}. The conditioning stage is the taller of the two at \
                 every tier, so a Dit-only window would leave the request peak untouched"
            )));
        }
        let size = memory
            .transformer_window_size
            .unwrap_or(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE);
        if size != crate::memory_strategy::TRANSFORMER_WINDOW_SIZE {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer window {size} is outside the published domain \
                 [{}] — 1 is the measured residency floor on every text-encoder tier, so larger \
                 windows only raise the peak this rung exists to bound and are not advertised",
                crate::memory_strategy::TRANSFORMER_WINDOW_SIZE
            )));
        }
        Ok(Some(size))
    }

    /// [`Self::load_task_dit`]'s rung-4 twin: defer the 600 block tensors and hold only the I/O
    /// projections and the refiner. Adapters were already refused by
    /// [`Self::requested_transformer_window`]; the assertion keeps the two seams from drifting.
    fn load_task_dit_deferred(&self, task: MiniMaxH3Task) -> Result<MiniMaxH3Dit> {
        assert!(
            self.adapters.is_empty(),
            "{MODEL_ID}: a deferred DiT load with adapters configured — \
             requested_transformer_window must refuse this before the load is reached"
        );
        MiniMaxH3Dit::load_dir_deferred(self.task_dit_dir(task), self.dtype)
    }

    /// Map the text-encoder shards a presentation needs, out of the **resolved** text-encoder
    /// directory — so a staged packed tier ([`TEXT_ENCODER_COMPONENT`], sc-19120) is mapped instead
    /// of the dense root component.
    ///
    /// The window itself lives with the encoder ([`crate::text_encoder::map_shards`]), which is
    /// also what the tier measurement drives, so a per-stage memory figure is taken from this exact
    /// mapping rather than from a copy of it.
    fn map_te_shards(&self, with_vision: bool) -> Result<Weights> {
        crate::text_encoder::map_shards(&self.text_encoder_dir, with_vision)
    }

    /// Build the request's text encoder at the residency [`Self::requested_transformer_window`]
    /// admitted: resident over the mapped shards, or deferred with a per-window rebuild (sc-18662).
    ///
    /// One constructor for all three conditioning paths, so `t2va` cannot stream while `fl2va` /
    /// `ref2va` silently stay resident — the windowed walk lives in the encoder's own `run_layers`,
    /// which every forward variant routes through.
    fn build_te(
        &self,
        w: Option<&Weights>,
        cfg: &MiniMaxH3TeConfig,
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
    ) -> Result<MiniMaxH3TextEncoder> {
        match window {
            None => MiniMaxH3TextEncoder::from_weights(
                w.expect("a resident encoder builds from the mapped shards"),
                LM_PREFIX,
                cfg,
            ),
            Some(window) => {
                let mut te = MiniMaxH3TextEncoder::from_dir_deferred(
                    &self.text_encoder_dir,
                    LM_PREFIX,
                    cfg,
                )?;
                te.set_block_window(window as usize, cancel.clone())?;
                Ok(te)
            }
        }
    }

    /// Encode the prompt and immediately release the 66.7 GB text encoder.
    fn encode_prompt(
        &self,
        prompt: &str,
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
    ) -> Result<mlx_rs::Array> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return Ok(crate::veda_fixture::context());
        }
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) = tok.encode_prompt(prompt)?;

        // Under a deferred load the map is skipped entirely: `from_dir_deferred` reads only
        // `embed_tokens`, and a retained shard map would keep every layer tensor reachable —
        // the retained-view hazard `block_stream`'s docs name.
        let w = match window {
            None => Some(self.map_te_shards(false)?),
            Some(_) => None,
        };
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        let context = te.forward(&ids, &mask)?;
        // Force it BEFORE the encoder is dropped: under lazy evaluation the context is a graph node
        // holding every weight it was computed from, so dropping first would free nothing and the
        // first denoise step would re-materialize 66.7 GB.
        mlx_rs::transforms::eval([&context])?;
        release((te, w));
        Ok(context)
    }

    /// The `fl2va` presentation: run the Qwen3-VL **vision tower** over the keyframes, splice their
    /// embeddings into the `"<Picture i>: "` + vision-block presentation, and return the context
    /// together with its **per-row modality tags**.
    ///
    /// This is one of the two paths a keyframe takes. The other is the VAE encode in
    /// [`crate::conditioning`]; the sc-17242 spike established that the reference runs **both**,
    /// and neither substitutes for the other — the tower supplies semantic context to the prompt
    /// stream, the VAE supplies the pixel-space anchor the video rows are conditioned on.
    fn encode_prompt_grounded(
        &self,
        prompt: &str,
        keyframes: &[&mlx_gen::media::Image],
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
    ) -> Result<(mlx_rs::Array, Vec<i32>)> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let mut w = self.map_te_shards(true)?;

        let vision = VisionTower::from_weights(
            &w,
            crate::text_encoder::minimax_h3_vision_config(),
            VISION_PREFIX,
            VISION_GROUP_SIZE,
        )?;
        // sc-23402: the tower's 351 source tensors are the read set of `w` at this point — force
        // and GPU-verify them (sc-22414) before the first tower kernel touches them cold. Peak
        // -neutral: the tower is resident for the whole `run_vision` either way.
        w.materialize_accessed()?;
        let grounded = crate::text_encoder::run_vision(&vision, keyframes)?;
        // Force the tower's output BEFORE dropping it, and drop its tensors out of `w` too — the
        // same discipline `encode_prompt` documents. Under lazy evaluation `grounded` is a graph
        // node holding every weight it was computed from, so releasing the tower while `w` still
        // maps `model.visual.*` frees nothing and the splice below would re-materialize it.
        let mut forced: Vec<&mlx_rs::Array> = grounded.embeds.iter().collect();
        forced.extend(grounded.deepstack.iter().flatten());
        mlx_rs::transforms::eval(forced)?;
        release(vision);
        w.remove_prefix(VISION_PREFIX);

        let (ids, mask, tags) = tok.encode_fl2va(prompt, &grounded.counts)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        // Under a window the mapped shards are released before the encoder is built: the deferred
        // loader reopens the staged directory per window, and a retained shard map would keep every
        // layer tensor reachable across the whole windowed walk.
        let w = match window {
            None => Some(w),
            Some(_) => {
                release(w);
                None
            }
        };
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        let context = te.forward_with_images(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        mlx_rs::transforms::eval([&context])?;
        release((te, w, grounded));

        // The tags describe the presentation row for row; a mismatch would mis-tag every row after
        // the divergence and is silent, because both lengths build a runnable sequence.
        if context.shape()[1] != tags.len() as i32 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the grounded context has {} rows but {} modality tags",
                context.shape()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        *self.veda_report.borrow_mut() = mlx_gen::gen_core::VedaExecutionReport {
            requested: req.veda.as_ref().is_some_and(|v| v.enabled),
            ..Default::default()
        };
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let geometry = request_geometry(req)?;
        let evaluations = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        // `num_inference_steps` counts the terminal σ = 0, at which the model is never evaluated.
        //
        // **The video shift is a per-request knob** (sc-18729). The base checkpoint's published
        // 12.0 is the default, but the distilled turbo variants are trained against their own
        // shift and are simply wrong on 12.0 — `lightx2v/Minimax-h3-Turbo`'s own
        // `DIFFUSERS_SETUP_AND_INFERENCE.md` passes `--video-shift 6` for the 4-step 768p file,
        // and its model-specs table lists the training shifts per variant (12/3 for the 544p
        // 4-step and 8-step files, 6/3 for the 768p one). `scheduler_shift` is the shared
        // request-level home for exactly this (`mlx-gen-wan`, `mlx-gen-scail2` and
        // `mlx-gen-sensenova` already read it), so a turbo render needs no bespoke surface.
        //
        // Only the **video** shift is overridable, because only the video shift moves across the
        // published set: every documented variant keeps audio at 3.0, and the reference's
        // `--audio-shift` is a separate flag this knob is not. `AUDIO_SIGMA_SHIFT` is therefore
        // passed explicitly rather than defaulted, so the pair this call builds is visible at the
        // call site instead of hiding behind `JointSchedule::new`'s "both at their own published
        // shifts".
        let video_shift = req.scheduler_shift.unwrap_or(VIDEO_SIGMA_SHIFT);
        let schedule = JointSchedule::with_shifts(evaluations + 1, video_shift, AUDIO_SIGMA_SHIFT)?;
        if schedule.num_evals() == 0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a {evaluations}-step schedule collapsed to no model evaluations \
                 (num_inference_steps must be >= {MIN_INFERENCE_STEPS})"
            )));
        }
        let seed = req.seed.unwrap_or_else(default_seed);

        // --- 1. conditioning ----------------------------------------------------------------
        // `t2va` and `fl2va` diverge here and stay diverged through the latent prep — deliberately,
        // as the reference does. Zero keyframes is `t2va` at the WEIGHTS level (one `transformer`
        // partition) and a different block path at every other level.
        let keyframes = req.keyframes();
        let references = request_references(req)?;
        // **The checkpoint decision.** Resolved once, here, and carried to the single
        // `MiniMaxH3Dit::load` call site — the two partitions are byte-different and structurally
        // identical, so this is the only thing standing between a `ref2va` request and a plausible
        // render off the wrong 66 GB. See [`MiniMaxH3Task`].
        let task = MiniMaxH3Task::resolve(!keyframes.is_empty(), references.is_some())?;
        let patch = self.task_dit_config(task)?.patch_size;
        // The rung-4 admission (sc-18662), resolved once and carried to every heavy phase — the
        // conditioning and denoise residency must agree, or half the request streams while the
        // other half quietly holds a resident stack.
        let window = self.requested_transformer_window(req)?;
        if let Some(refs) = &references {
            return self.generate_ref2va(
                req,
                refs,
                task,
                &geometry,
                &schedule,
                seed,
                window,
                on_progress,
            );
        }
        let anchors = keyframe_anchors(&keyframes, geometry.joint.num_frames)?;
        // Fitted ONCE, here, and shared by both keyframe paths. The vision tower and the VAE
        // encode must see the same pixels — resizing separately per path would let a resampling
        // difference put the two conditioning signals fractionally out of register, which is
        // exactly the class of divergence nothing downstream can detect.
        let fitted = if anchors.is_empty() {
            Vec::new()
        } else {
            let images = keyframe_images(&keyframes, geometry.joint.num_frames)?;
            crate::keyframe::fit_keyframes(&images, geometry.width, geometry.height)?
        };

        // The two stage boundaries this model's residency seam actually has (sc-19120). H3 is a
        // load→use→drop seam through and through, so `Progress::Loading` is the event its phases
        // were specified for — `mlx-gen-krea-realtime` emits the same pair. Two things depend on
        // it: a UI that would otherwise sit silent through a multi-GB text-encoder map with no
        // `Step` to show, and `tests/te_tier_generate_stages.rs`, which resets the MLX peak at each
        // boundary and so can attribute a high-water to the stage that set it. That attribution is
        // the whole reason this story exists: for three measurements the conditioning stage's peak
        // was read as the DiT's.
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder));
        let (context, text_tags) = if anchors.is_empty() {
            let context = self.encode_prompt(&req.prompt, window, &req.cancel)?;
            let tags = vec![crate::denoise::TEXT_TAG; context.shape()[1] as usize];
            (context, tags)
        } else {
            let refs: Vec<&mlx_gen::media::Image> = fitted.iter().collect();
            in_fl2va_phase(
                FL2VA_GROUNDED_QWEN3_VL_PHASE,
                self.encode_prompt_grounded(&req.prompt, &refs, window, &req.cancel),
            )?
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 1b. keyframe conditioning latents (fl2va) ----------------------------------------
        // The VAE encode, the second of the keyframe's two paths. Done before the DiT is mapped so
        // the ~10 GB VAE and the ~62 GB DiT are never both resident.
        let condition_rows = if anchors.is_empty() {
            None
        } else {
            in_fl2va_phase(
                FL2VA_KEYFRAME_VAE_PHASE,
                (|| {
                    let pixels: Vec<mlx_rs::Array> = fitted
                        .iter()
                        .map(crate::keyframe::keyframe_to_vae_pixels)
                        .collect::<Result<_>>()?;
                    let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
                    let rows = crate::conditioning::build_condition_rows(
                        &vae,
                        &pixels,
                        &anchors,
                        patch,
                        &crate::conditioning::KeyframeNoise::Seeded,
                    )?;
                    if let Some(r) = &rows {
                        // Force before the VAE is dropped, for the same lazy-evaluation reason the
                        // context is forced above.
                        mlx_rs::transforms::eval([r])?;
                    }
                    release((vae, pixels));
                    Ok(rows)
                })(),
            )?
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. denoise ---------------------------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer));
        let dit = match window {
            None => self.load_task_dit(task)?,
            Some(_) => self.load_task_dit_deferred(task)?,
        };
        debug_assert_eq!(patch, dit.config().patch_size);
        let layout = if anchors.is_empty() {
            t2va_layout(&geometry, context.shape()[1], patch)?
        } else {
            fl2va_layout(&geometry, &text_tags, &anchors, patch)?
        };
        let (video_rows, audio_rows) = initial_latents(&geometry, patch, seed)?;
        // The anchors LEAD the video row stream; the scheduler then writes only the tail.
        let video_rows = prepend_condition_rows(&layout, condition_rows.as_ref(), &video_rows)?;
        let adaln = crate::denoise::adaln_schedule(&schedule)?;
        // The 26.02 GB lever: project the whole schedule's modulation and release `adaln_proj`.
        // Every timestep this run evaluates at is enumerated in `adaln`, so the eviction is safe.
        // Under rung 4 the same projection pass runs windowed instead — a deferred load never held
        // the projections, so there is nothing to evict (`JointDit::new_windowed`).
        let mut model = match window {
            None => JointDit::new(
                dit,
                layout.clone(),
                &context,
                adaln,
                AdaLnResidency::PrecomputeAndEvict,
            )?,
            Some(w) => JointDit::new_windowed(
                dit,
                layout.clone(),
                &context,
                adaln,
                AdaLnResidency::PrecomputeAndEvict,
                w as usize,
                req.cancel.clone(),
            )?,
        };
        if let Some(request) = req.veda.as_ref().filter(|v| v.enabled) {
            #[cfg(test)]
            if let Some(fixture) = &self.fixture {
                if fixture.predictor {
                    model.prepare_veda_fixture(
                        request,
                        self.veda_report.clone(),
                        fixture.fail_self_test,
                    );
                } else {
                    model.prepare_veda(
                        request,
                        self.veda_component.as_deref(),
                        self.veda_report.clone(),
                    );
                }
            } else {
                model.prepare_veda(
                    request,
                    self.veda_component.as_deref(),
                    self.veda_report.clone(),
                );
            }
            #[cfg(not(test))]
            model.prepare_veda(
                request,
                self.veda_component.as_deref(),
                self.veda_report.clone(),
            );
        }
        release(context);

        let total = schedule.num_evals() as u32;
        let mut on_step = |completed: usize| {
            on_progress(Progress::Step {
                current: completed as u32,
                total,
            });
        };
        let mut terminal_probe = req
            .h3_terminal_telemetry
            .then(mlx_gen::gen_core::h3_terminal_telemetry::Collector::default);
        let mut observe_step =
            |index, elapsed| {
                if let Some(probe) = terminal_probe.as_mut() {
                    probe.record(index, elapsed, serde_json::json!({
                    "method": "mlx_process_allocator_cumulative_highwater_no_reset",
                    "allocatorActiveSampleBytes": mlx_rs::memory::get_active_memory() as u64,
                    "allocatorCacheSampleBytes": mlx_rs::memory::get_cache_memory() as u64,
                    "allocatorPeakBytesCumulative": mlx_rs::memory::get_peak_memory() as u64,
                }));
                }
            };
        let rendered = render_latents_with_observer(
            &mut model,
            &layout,
            &schedule,
            &video_rows,
            &audio_rows,
            patch,
            &req.cancel,
            &mut on_step,
            req.h3_terminal_telemetry
                .then_some(&mut observe_step as &mut dyn FnMut(usize, std::time::Duration)),
        )?;
        if let Some(probe) = terminal_probe {
            self.veda_report.borrow_mut().terminal_telemetry = Some(probe.finish("mlx"));
        }

        release((model, video_rows, audio_rows));

        // --- 3. decode ----------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let frames = self.decode_video(req, &rendered.video)?;
        let audio = self.decode_audio(&rendered.audio)?;
        let audio = fit_audio_to_video(audio, &geometry)?;
        if audio.sample_rate != AUDIO_SAMPLE_RATE || audio.channels != AUDIO_OUTPUT_CHANNELS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the decoded soundtrack is {} Hz / {} channels, expected \
                 {AUDIO_SAMPLE_RATE} / {AUDIO_OUTPUT_CHANNELS}",
                audio.sample_rate, audio.channels
            )));
        }
        if frames.len() != geometry.joint.num_frames as usize {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: crate::denoise::MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
    }

    /// The **`ref2va` render** — ordered multi-modal references on the `transformer_ref`
    /// checkpoint.
    ///
    /// The phase order is the same staged one `t2va` / `fl2va` use, for the same memory reason, but
    /// it has **three** heavy phases rather than two: the 66.7 GB conditioner, then the VAEs, then
    /// the 66 GB `transformer_ref`. Each is released before the next is mapped, and the DiT is
    /// mapped last so it is never resident alongside the conditioner. sc-17151 owns making that
    /// residency contract enforced rather than merely observed.
    #[allow(clippy::too_many_arguments)]
    fn generate_ref2va(
        &self,
        req: &GenerationRequest,
        references: &Ref2VaReferences,
        task: MiniMaxH3Task,
        geometry: &RequestGeometry,
        schedule: &JointSchedule,
        seed: u64,
        window: Option<u32>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        use crate::conditioning::{
            encode_reference_condition, keyframe_condition_rows, reference_audio_rows,
            reference_clip_to_vae_pixels, KeyframeNoise,
        };
        use crate::dit::positions::ReferenceLatentGeometry;
        use crate::pipeline::{prepend_condition_audio_rows, ref2va_layout_with_audio_channels};

        let patch = self.task_dit_config(task)?.patch_size;

        // --- 0. normalize every reference onto the model's own rates and resolutions ------------
        // References do NOT bind the canvas: the geometry was already resolved (16:9 by default)
        // and each reference is put on its own resolution here.
        //
        // The image short edge is the request's effective one (sc-23402) — validated in
        // `validate_request` before any weight was read, so this resolve cannot admit a value the
        // gate refused. It sizes the reference and its latent rows only; the canvas above is
        // untouched by it.
        let reference_short_edge = effective_reference_image_short_edge(req) as i32;
        // sc-26691: the image resize mode (default `match`) and the upstream soundtrack bound.
        let resize_mode = mlx_gen::gen_core::effective_reference_image_resize_mode(req);
        let num_frames = geometry.joint.num_frames as usize;
        let truncate = |a: &mlx_gen::media::AudioTrack| {
            mlx_gen::gen_core::truncate_reference_audio(
                a,
                num_frames,
                crate::denoise::MINIMAX_H3_FPS,
            )
        };
        let mut normalized: Vec<Ref2VaReference> = Vec::with_capacity(references.len());
        for r in references.as_slice() {
            normalized.push(match r {
                Ref2VaReference::Image(img) => {
                    Ref2VaReference::Image(crate::reference::resize_reference_image(
                        img,
                        SPATIAL_STRIDE as i32,
                        geometry.width,
                        geometry.height,
                        resize_mode,
                        reference_short_edge,
                    )?)
                }
                Ref2VaReference::Video(v) => Ref2VaReference::Video(VideoReference {
                    frames: crate::reference::normalize_reference_clip(
                        &v.frames,
                        v.fps,
                        geometry.joint.num_frames as usize,
                        SPATIAL_STRIDE as i32,
                        crate::pipeline::CANVAS_SHORT_EDGE as i32,
                        i64::from(crate::pipeline::CANVAS_MAX_PIXELS),
                        crate::denoise::MINIMAX_H3_FPS,
                    )?,
                    fps: crate::denoise::MINIMAX_H3_FPS,
                    audio: v.audio.as_ref().map(truncate),
                }),
                Ref2VaReference::Audio(a) => {
                    Ref2VaReference::Audio(crate::reference::AudioReference {
                        audio: truncate(&a.audio),
                    })
                }
            });
        }
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 1. the conditioner: vision tower over the visual references, then the 66.7 GB LM ---
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder));
        let (context, text_tags) =
            self.encode_prompt_ref2va(&req.prompt, &normalized, window, &req.cancel)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. the VAEs: one conditioning latent per visual reference, one soundtrack per -------
        //        audio-bearing reference. Done before the DiT is mapped.
        let mut condition_rows: Vec<mlx_rs::Array> = Vec::new();
        let mut audio_rows_per_ref: Vec<mlx_rs::Array> = Vec::new();
        let mut geometries: Vec<ReferenceLatentGeometry> = Vec::new();
        let mut reference_audio_channels = Vec::new();
        {
            let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
            let audio_enc = self.load_audio_encoder()?;
            for (i, r) in normalized.iter().enumerate() {
                // Audio FIRST, mirroring the packed row order.
                let num_audio_latents = if let Some(track) = r.audio() {
                    let wave = crate::pipeline::audio_track_to_encoder_input(track)?;
                    let posterior = audio_enc.encode(&wave)?;
                    // The MODE, never a sample — a soundtrack is deterministic conditioning.
                    let normed = audio_enc.normalize(posterior.mode())?;
                    // `[B, latent_channels, latents]` -> `[channels, latents, features]`.
                    let t = normed.transpose_axes(&[0, 2, 1])?;
                    reference_audio_channels.push(t.shape()[0]);
                    let latents = t.shape()[1];
                    audio_rows_per_ref.push(reference_audio_rows(&t)?);
                    latents
                } else {
                    reference_audio_channels.push(0);
                    0
                };

                let (frames, height, width) = match r {
                    Ref2VaReference::Audio(_) => (0, 0, 0),
                    Ref2VaReference::Image(img) => {
                        let pixels = reference_clip_to_vae_pixels(std::slice::from_ref(img))?;
                        let c =
                            encode_reference_condition(&vae, &pixels, &KeyframeNoise::Seeded, i)?;
                        let s = c.shape().to_vec();
                        condition_rows.push(keyframe_condition_rows(
                            &c,
                            patch,
                            &KeyframeNoise::Seeded,
                            i,
                        )?);
                        (s[2], s[3], s[4])
                    }
                    Ref2VaReference::Video(v) => {
                        // Snap DOWN to the VAE's `17n + 5` lattice so nothing is padded.
                        let keep = crate::conditioning::snap_reference_frames_down(v.frames.len())
                            .min(v.frames.len());
                        let pixels = reference_clip_to_vae_pixels(&v.frames[..keep])?;
                        let c =
                            encode_reference_condition(&vae, &pixels, &KeyframeNoise::Seeded, i)?;
                        let s = c.shape().to_vec();
                        condition_rows.push(keyframe_condition_rows(
                            &c,
                            patch,
                            &KeyframeNoise::Seeded,
                            i,
                        )?);
                        (s[2], s[3], s[4])
                    }
                };
                geometries.push(ReferenceLatentGeometry {
                    kind: r.kind(),
                    num_latent_frames: frames,
                    latent_height: height,
                    latent_width: width,
                    num_audio_latents,
                });
            }
            // Force BEFORE the VAEs are dropped: under lazy evaluation these rows are graph nodes
            // holding every weight they were computed from, so dropping first frees nothing.
            let mut forced: Vec<&mlx_rs::Array> = condition_rows.iter().collect();
            forced.extend(audio_rows_per_ref.iter());
            if !forced.is_empty() {
                mlx_rs::transforms::eval(forced)?;
            }
            release((vae, audio_enc));
        }
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let condition_video = (!condition_rows.is_empty())
            .then(|| mlx_rs::ops::concatenate_axis(&condition_rows, 1))
            .transpose()?;
        let condition_audio = (!audio_rows_per_ref.is_empty())
            .then(|| mlx_rs::ops::concatenate_axis(&audio_rows_per_ref, 1))
            .transpose()?;

        // --- 3. denoise on `transformer_ref` ----------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer));
        let dit = match window {
            None => self.load_task_dit(task)?,
            Some(_) => self.load_task_dit_deferred(task)?,
        };
        debug_assert_eq!(patch, dit.config().patch_size);
        let layout = ref2va_layout_with_audio_channels(
            geometry,
            &text_tags,
            &geometries,
            &reference_audio_channels,
            patch,
        )?;
        let (video_rows, audio_rows) = initial_latents(geometry, patch, seed)?;
        let video_rows = prepend_condition_rows(&layout, condition_video.as_ref(), &video_rows)?;
        let audio_rows =
            prepend_condition_audio_rows(&layout, condition_audio.as_ref(), &audio_rows)?;
        let adaln = crate::denoise::adaln_schedule(schedule)?;
        let mut model = match window {
            None => JointDit::new(
                dit,
                layout.clone(),
                &context,
                adaln,
                AdaLnResidency::PrecomputeAndEvict,
            )?,
            Some(w) => JointDit::new_windowed(
                dit,
                layout.clone(),
                &context,
                adaln,
                AdaLnResidency::PrecomputeAndEvict,
                w as usize,
                req.cancel.clone(),
            )?,
        };
        if let Some(request) = req.veda.as_ref().filter(|v| v.enabled) {
            model.prepare_veda(
                request,
                self.veda_component.as_deref(),
                self.veda_report.clone(),
            );
        }
        release(context);

        let total = schedule.num_evals() as u32;
        let mut on_step = |completed: usize| {
            on_progress(Progress::Step {
                current: completed as u32,
                total,
            });
        };
        let mut terminal_probe = req
            .h3_terminal_telemetry
            .then(mlx_gen::gen_core::h3_terminal_telemetry::Collector::default);
        let mut observe_step =
            |index, elapsed| {
                if let Some(probe) = terminal_probe.as_mut() {
                    probe.record(index, elapsed, serde_json::json!({
                    "method": "mlx_process_allocator_cumulative_highwater_no_reset",
                    "allocatorActiveSampleBytes": mlx_rs::memory::get_active_memory() as u64,
                    "allocatorCacheSampleBytes": mlx_rs::memory::get_cache_memory() as u64,
                    "allocatorPeakBytesCumulative": mlx_rs::memory::get_peak_memory() as u64,
                }));
                }
            };
        let rendered = render_latents_with_observer(
            &mut model,
            &layout,
            schedule,
            &video_rows,
            &audio_rows,
            patch,
            &req.cancel,
            &mut on_step,
            req.h3_terminal_telemetry
                .then_some(&mut observe_step as &mut dyn FnMut(usize, std::time::Duration)),
        )?;
        if let Some(probe) = terminal_probe {
            self.veda_report.borrow_mut().terminal_telemetry = Some(probe.finish("mlx"));
        }

        release((model, video_rows, audio_rows));

        // --- 4. decode --------------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let frames = self.decode_video(req, &rendered.video)?;
        let audio = self.decode_audio(&rendered.audio)?;
        let audio = fit_audio_to_video(audio, geometry)?;
        if frames.len() != geometry.joint.num_frames as usize {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame ref2va request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: crate::denoise::MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
    }

    /// The audio VAE **encoder**, built from the same three FL2VA source documents the decoder
    /// reads its constructor arguments from.
    fn load_audio_encoder(&self) -> Result<crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder> {
        let source = self.root.join("FL2VA").join("audio_vae");
        let read = |name: &str| -> Result<String> {
            let path = source.join(name);
            mlx_gen::gen_core::__model_input_choose!(
                std::fs::read_to_string(&path),
                mlx_gen::gen_core::model_input_proof::read_configuration_to_string(&path)
            )
            .map_err(|e| Error::Msg(format!("{MODEL_ID}: reading {}: {e}", path.display())))
        };
        let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )?;
        let mut w = Weights::from_dir(self.root.join("audio_vae"))?;
        let encoder = crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::from_weights(
            &mut w,
            &cfg,
            Dtype::Float32,
        )?;
        // sc-23402: force + GPU-verify the read set (sc-22414) before the first encode.
        w.materialize_accessed()?;
        Ok(encoder)
    }

    /// The `ref2va` presentation: the vision tower over every **visual** reference, spliced into
    /// the `<Picture i>` / `<Audio j>` / `<Video k>` presentation.
    ///
    /// A waveform never reaches the tower — a standalone audio reference contributes its label and
    /// nothing else, which is why the tower's sources and the presentation entries are built in one
    /// pass rather than zipped afterwards.
    fn encode_prompt_ref2va(
        &self,
        prompt: &str,
        references: &[Ref2VaReference],
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
    ) -> Result<(mlx_rs::Array, Vec<i32>)> {
        use crate::reference::{
            sample_video_condition_frames, ReferencePresentation, VIDEO_SAMPLE_FPS,
            VISION_TEMPORAL_PATCH,
        };

        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let mut w = self.map_te_shards(true)?;

        // The tower's sources, in **sequence order** — the order the pad runs appear in, which is
        // what `forward_with_references` consumes them in.
        let mut sources: Vec<crate::text_encoder::VisionSource<'_>> = Vec::new();
        // Per reference: how many tower sources it contributes, and its block timestamps.
        let mut plan: Vec<(usize, Vec<f64>)> = Vec::new();
        for r in references {
            match r {
                Ref2VaReference::Audio(_) => plan.push((0, Vec::new())),
                Ref2VaReference::Image(img) => {
                    sources.push(crate::text_encoder::VisionSource::Image(img));
                    plan.push((1, Vec::new()));
                }
                Ref2VaReference::Video(v) => {
                    let (indices, timestamps) = sample_video_condition_frames(
                        v.frames.len(),
                        crate::denoise::MINIMAX_H3_FPS,
                        VIDEO_SAMPLE_FPS,
                        VISION_TEMPORAL_PATCH,
                    )?;
                    // One tower input per merged group — the block the tokenizer presents — not one
                    // per sampled frame. A trailing partial group repeats its last frame, exactly
                    // as its timestamp label does.
                    let groups =
                        crate::reference::condition_frame_groups(&indices, VISION_TEMPORAL_PATCH);
                    if groups.len() != timestamps.len() {
                        return Err(Error::Msg(format!(
                            "{MODEL_ID}: a reference clip sampled {} frame group(s) for {} block \
                             timestamp(s)",
                            groups.len(),
                            timestamps.len()
                        )));
                    }
                    for &(first, second) in &groups {
                        sources.push(crate::text_encoder::VisionSource::FramePair(
                            &v.frames[first],
                            &v.frames[second],
                        ));
                    }
                    plan.push((groups.len(), timestamps));
                }
            }
        }
        if sources.is_empty() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a ref2va request carries no visual reference for the conditioner — a \
                 waveform never reaches it"
            )));
        }

        let vision = VisionTower::from_weights(
            &w,
            crate::text_encoder::minimax_h3_vision_config(),
            VISION_PREFIX,
            VISION_GROUP_SIZE,
        )?;
        // sc-23402: same as `encode_prompt_grounded` — the tower is the FIRST big load of a
        // `ref2va` request and the packed token table the second; both are verified at their
        // load boundary now, the table in `MiniMaxH3TextEncoder::from_weights`.
        w.materialize_accessed()?;
        let grounded = crate::text_encoder::run_vision_sources(&vision, &sources)?;
        let mut forced: Vec<&mlx_rs::Array> = grounded.embeds.iter().collect();
        forced.extend(grounded.deepstack.iter().flatten());
        mlx_rs::transforms::eval(forced)?;
        release(vision);
        w.remove_prefix(VISION_PREFIX);

        // The presentation, from the tower's per-source counts.
        let mut cursor = 0usize;
        let mut presentation: Vec<ReferencePresentation> = Vec::with_capacity(references.len());
        for (r, (n_sources, timestamps)) in references.iter().zip(&plan) {
            match r {
                Ref2VaReference::Audio(_) => presentation.push(ReferencePresentation::Audio),
                Ref2VaReference::Image(_) => {
                    presentation.push(ReferencePresentation::Image {
                        num_tokens: grounded.counts[cursor],
                    });
                    cursor += 1;
                }
                Ref2VaReference::Video(v) => {
                    // One block per merged frame pair; every block of one clip shares a token count.
                    let num_tokens = grounded.counts[cursor];
                    presentation.push(ReferencePresentation::Video {
                        num_tokens,
                        timestamps: timestamps.clone(),
                        has_audio: v.audio.is_some(),
                    });
                    cursor += n_sources;
                }
            }
        }

        let (ids, mask, tags) = tok.encode_ref2va(prompt, &presentation)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        // Same release-before-build as `encode_prompt_grounded`: a retained shard map would defeat
        // the windowed walk's release.
        let w = match window {
            None => Some(w),
            Some(_) => {
                release(w);
                None
            }
        };
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        let context = te.forward_with_references(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        mlx_rs::transforms::eval([&context])?;
        release((te, w, grounded));

        if context.shape()[1] != tags.len() as i32 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the ref2va context has {} rows but {} modality tags",
                context.shape()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }

    /// Decode the video latent, at the tiling the request's rung-2 selection admits.
    ///
    /// **The tiling is resolved BEFORE the 10.42 GB VAE is mapped**, deliberately. An unadmitted
    /// geometry is a caller error, and answering it with a file-load error (or worse, after paying
    /// the load) would hide which of the two actually failed. It is also what lets the reachability
    /// test prove this call site exists without a snapshot: the refusal fires on a root that has no
    /// weights at all.
    fn decode_video(
        &self,
        req: &GenerationRequest,
        latents: &mlx_rs::Array,
    ) -> Result<Vec<mlx_gen::media::Image>> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return crate::veda_fixture::decode_video(req, latents);
        }
        let tiling = crate::pipeline::decode_tiling_for(req)?;
        let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?.with_tiling(tiling);
        let decoded = vae.decode(latents)?;
        let pixels = revert_pixel_normalization(&decoded)?;
        mlx_rs::transforms::eval([&pixels])?;
        release(vae);
        frames_to_images(&pixels)
    }

    fn decode_audio(&self, latents: &mlx_rs::Array) -> Result<mlx_gen::media::AudioTrack> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return crate::veda_fixture::decode_audio(latents);
        }
        // The geometry comes from the three FL2VA source documents the reference's own
        // `from_pretrained` reads — parsed, not defaulted, so a variant snapshot cannot silently
        // run at this one's numbers. The weights come from the repackaged `audio_vae/`, which ships
        // the same tensors under the same names.
        let source = self.root.join("FL2VA").join("audio_vae");
        let read = |name: &str| -> Result<String> {
            let path = source.join(name);
            mlx_gen::gen_core::__model_input_choose!(
                std::fs::read_to_string(&path),
                mlx_gen::gen_core::model_input_proof::read_configuration_to_string(&path)
            )
            .map_err(|e| Error::Msg(format!("{MODEL_ID}: reading {}: {e}", path.display())))
        };
        let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )?;
        let mut w = Weights::from_dir(self.root.join("audio_vae"))?;
        let vae = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32)?;
        // sc-23402: force + GPU-verify the read set (sc-22414) before the decode consumes it.
        w.materialize_accessed()?;
        let track = vae.decode_audio_track(latents)?;
        release((vae, w));
        Ok(track)
    }
}

/// The video VAE decodes a *whole* clip in one call, so the memory contract is the same one
/// `MiniMaxH3VideoVae::decode_temporal` already owns.
const _: () = {
    assert!(MIN_DURATION_SECONDS < MAX_DURATION_SECONDS);
};

/// Hand-written rather than `mlx_gen::impl_generator!`, for the three memory-strategy methods the
/// macro cannot emit (sc-18650) — the ltx/wan shape.
///
/// The macro emits **only** `descriptor` / `validate` / `generate`, so while this provider used it
/// the other three slots held their trait defaults: `memory_strategy_contract` answered `None`, and
/// downstream that is the "has not adopted the shared contract" state — SceneWorks' video admission
/// reads exactly this seam and skips the request-geometry gate entirely when it is `None`. A 33B
/// joint-AV model was therefore rendering ungated while `crate::memory_strategy` built a complete,
/// calibrated contract that nothing ever asked for.
///
/// The first three bodies below are the same delegation the macro emitted, verbatim.
///
/// # Why all three overrides land together
///
/// They are not independent. The defaults are mutually consistent only for a provider publishing
/// **no** contract:
///
/// * `memory_strategy_safety_check`'s default rejects every non-`Resident` selection outright;
/// * `begin_memory_strategy_request`'s default hard-errors ("advertises an implemented optimized
///   memory strategy but does not open a request scope") for any contract declaring an implemented
///   optimized rung — and this one declares two, `StagedResidency` and `BoundedDecode`, plus
///   `BoundedTransformerResidency` on a deferred load.
///
/// So publishing the contract alone would turn today's working-but-ungated renders into hard
/// errors. The rung semantics are not restated here: all three bodies delegate to
/// [`crate::memory_strategy`], whose `strategies()` is the single declaration and whose
/// `route_gate` is the single geometry predicate.
impl Generator for MiniMaxH3 {
    fn veda_execution_report(&self) -> Option<mlx_gen::gen_core::VedaExecutionReport> {
        Some(self.veda_report.borrow().clone())
    }

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

    fn veda_workspace_bound(
        &self,
        req: mlx_gen::gen_core::veda::workspace::Request<'_>,
    ) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::veda::workspace::WorkspaceBound>> {
        crate::memory_strategy::veda_workspace_bound_for_contract(req, &self.memory_strategy)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    /// Defense in depth over the shared worker's selection. Delegates to the provider's own
    /// admission, which is `standard_memory_strategy_safety_check` (the capability table, the owned
    /// parameter domains and the numeric tier) **plus** this family's `route_gate` — the lattice,
    /// stride, canvas, batch and `use_pid` predicates the render itself enforces.
    ///
    /// **Two axes, not one.** The route gate can only judge what a `MemoryRunContext` carries —
    /// geometry, tier, parameters — so everything that is a property of the *load* rather than the
    /// request is judged by the capability table instead, from a contract resolved at load time.
    /// The deferred-shape and no-adapter prerequisites of rung 4 are both in that half
    /// ([`crate::memory_strategy::streamable`]). A selection this accepts is one `generate` can run
    /// only while both halves stay complete; sc-18650 fixed the case where the second was missing.
    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check_at_tier(
            &self.memory_strategy,
            self.memory_tier,
            context,
        )
    }

    /// Open the request scope the accepted selection needs, re-running the safety check first so a
    /// caller that skipped it cannot open a scope over a refused selection. The production entry
    /// point, so its terminal cleanup drains the MLX allocator cache.
    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy::begin_request(&self.memory_strategy, self.memory_tier, context)
    }
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
pub(crate) mod tests;

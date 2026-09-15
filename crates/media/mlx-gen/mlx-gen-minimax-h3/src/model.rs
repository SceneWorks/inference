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
    reject_unknown_components, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, KeyframeRef, LoadSpec, Modality, ModelDescriptor, Precision,
    Progress, Quant, StepSupport, WeightsSource,
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
    render_latents, resolve_geometry, revert_pixel_normalization, t2va_layout, RequestGeometry,
    MAX_CANVAS_EDGE, MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, SMALLEST_LEGAL_FRAMES,
    SPATIAL_STRIDE,
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
    let frames = match (req.frames, req.duration) {
        (Some(frames), _) => i32::try_from(frames)
            .map_err(|_| Error::Msg(format!("{MODEL_ID}: frames {frames} does not fit an i32")))?,
        (None, Some(seconds)) => crate::pipeline::align_frames_for_duration(seconds)?,
        (None, None) => SMALLEST_LEGAL_FRAMES,
    };
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
    reject_unknown_components(spec, &[DIT_COMPONENT, TEXT_ENCODER_COMPONENT], MODEL_ID)?;
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
    // sc-18650. Resolved here, at the one point that holds the spec, and carried on the generator so
    // `Generator::memory_strategy_contract` has something to publish. `contract_for` only stats the
    // component directories it has already probed above — no weight file is opened.
    let memory_strategy = crate::memory_strategy::contract_for(spec).map_err(Error::from)?;
    let memory_tier = crate::memory_strategy::numeric_tier(spec);
    Ok(Box::new(MiniMaxH3 {
        descriptor: descriptor(),
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
                        crate::pipeline::PATCH_SIZE,
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
        let patch = dit.config().patch_size;
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
        release(context);

        let total = schedule.num_evals() as u32;
        let mut on_step = |completed: usize| {
            on_progress(Progress::Step {
                current: completed as u32,
                total,
            });
        };
        let rendered = render_latents(
            &mut model,
            &layout,
            &schedule,
            &video_rows,
            &audio_rows,
            patch,
            &req.cancel,
            &mut on_step,
        )?;
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
        use crate::pipeline::{prepend_condition_audio_rows, ref2va_layout};

        // --- 0. normalize every reference onto the model's own rates and resolutions ------------
        // References do NOT bind the canvas: the geometry was already resolved (16:9 by default)
        // and each reference is put on its own resolution here.
        let mut normalized: Vec<Ref2VaReference> = Vec::with_capacity(references.len());
        for r in references.as_slice() {
            normalized.push(match r {
                Ref2VaReference::Image(img) => Ref2VaReference::Image(
                    crate::reference::normalize_reference_image(img, SPATIAL_STRIDE as i32)?,
                ),
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
                    audio: v.audio.clone(),
                }),
                Ref2VaReference::Audio(a) => Ref2VaReference::Audio(a.clone()),
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
                    let latents = t.shape()[1];
                    audio_rows_per_ref.push(reference_audio_rows(&t)?);
                    latents
                } else {
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
                            crate::pipeline::PATCH_SIZE,
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
                            crate::pipeline::PATCH_SIZE,
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
        let patch = dit.config().patch_size;
        let layout = ref2va_layout(geometry, &text_tags, &geometries, patch)?;
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
        release(context);

        let total = schedule.num_evals() as u32;
        let mut on_step = |completed: usize| {
            on_progress(Progress::Step {
                current: completed as u32,
                total,
            });
        };
        let rendered = render_latents(
            &mut model,
            &layout,
            schedule,
            &video_rows,
            &audio_rows,
            patch,
            &req.cancel,
            &mut on_step,
        )?;
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
            std::fs::read_to_string(&path)
                .map_err(|e| Error::Msg(format!("{MODEL_ID}: reading {}: {e}", path.display())))
        };
        let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )?;
        let mut w = Weights::from_dir(self.root.join("audio_vae"))?;
        crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::from_weights(
            &mut w,
            &cfg,
            Dtype::Float32,
        )
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
        let mut sources: Vec<&mlx_gen::media::Image> = Vec::new();
        // Per reference: how many tower sources it contributes, and its block timestamps.
        let mut plan: Vec<(usize, Vec<f64>)> = Vec::new();
        for r in references {
            match r {
                Ref2VaReference::Audio(_) => plan.push((0, Vec::new())),
                Ref2VaReference::Image(img) => {
                    sources.push(img);
                    plan.push((1, Vec::new()));
                }
                Ref2VaReference::Video(v) => {
                    let (indices, timestamps) = sample_video_condition_frames(
                        v.frames.len(),
                        crate::denoise::MINIMAX_H3_FPS,
                        VIDEO_SAMPLE_FPS,
                        VISION_TEMPORAL_PATCH,
                    )?;
                    for &i in &indices {
                        sources.push(&v.frames[i]);
                    }
                    plan.push((indices.len(), timestamps));
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
        let grounded = crate::text_encoder::run_vision(&vision, &sources)?;
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
        let tiling = crate::pipeline::decode_tiling_for(req)?;
        let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?.with_tiling(tiling);
        let decoded = vae.decode(latents)?;
        let pixels = revert_pixel_normalization(&decoded)?;
        mlx_rs::transforms::eval([&pixels])?;
        release(vae);
        frames_to_images(&pixels)
    }

    fn decode_audio(&self, latents: &mlx_rs::Array) -> Result<mlx_gen::media::AudioTrack> {
        // The geometry comes from the three FL2VA source documents the reference's own
        // `from_pretrained` reads — parsed, not defaulted, so a variant snapshot cannot silently
        // run at this one's numbers. The weights come from the repackaged `audio_vae/`, which ships
        // the same tensors under the same names.
        let source = self.root.join("FL2VA").join("audio_vae");
        let read = |name: &str| -> Result<String> {
            let path = source.join(name);
            std::fs::read_to_string(&path)
                .map_err(|e| Error::Msg(format!("{MODEL_ID}: reading {}: {e}", path.display())))
        };
        let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )?;
        let mut w = Weights::from_dir(self.root.join("audio_vae"))?;
        let vae = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32)?;
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
mod tests {
    use super::*;
    use mlx_gen::gen_core::CancelFlag;
    use mlx_gen::runtime::{AdapterKind, MoeExpert};

    #[test]
    fn fl2va_phase_boundaries_preserve_the_actionable_metal_timeout() {
        const TIMEOUT: &str = "[METAL] Command buffer execution failed: Caused GPU Timeout Error \
             (00000002:kIOGPUCommandBufferCallbackErrorTimeout)";

        for phase in [FL2VA_GROUNDED_QWEN3_VL_PHASE, FL2VA_KEYFRAME_VAE_PHASE] {
            let error = in_fl2va_phase::<()>(phase, Err(Error::Msg(TIMEOUT.into())))
                .expect_err("injected Metal timeout must remain a failure")
                .to_string();
            assert!(error.starts_with(phase), "missing FL2VA phase: {error}");
            assert!(
                error.contains("kIOGPUCommandBufferCallbackErrorTimeout"),
                "the original driver classification was hidden: {error}"
            );
        }
    }

    #[test]
    fn fl2va_phase_boundaries_do_not_stringify_typed_request_outcomes() {
        assert_eq!(
            in_fl2va_phase(FL2VA_KEYFRAME_VAE_PHASE, Ok(3.25_f32))
                .expect("a successful phase remains numerically transparent"),
            3.25
        );
        assert!(matches!(
            in_fl2va_phase::<()>(FL2VA_GROUNDED_QWEN3_VL_PHASE, Err(Error::Canceled)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            in_fl2va_phase::<()>(
                FL2VA_KEYFRAME_VAE_PHASE,
                Err(Error::Unsupported("fixture".into()))
            ),
            Err(Error::Unsupported(message)) if message == "fixture"
        ));
        assert!(matches!(
            in_fl2va_phase::<()>(
                FL2VA_KEYFRAME_VAE_PHASE,
                Err(Error::MissingTensor("fixture.weight".into()))
            ),
            Err(Error::MissingTensor(key)) if key == "fixture.weight"
        ));
    }

    fn request(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            width,
            height,
            cancel: CancelFlag::default(),
            ..Default::default()
        }
    }

    /// A generator over a snapshot that does not exist, at an explicit load shape.
    ///
    /// The contract is resolved from a spec carrying **that same shape** rather than being pinned
    /// here, because `load` resolves the two from one `LoadSpec` and a fixture that let them
    /// disagree would be testing a state the loader cannot produce (sc-18650).
    fn generator_with(dit_dir: &str, load_shape: mlx_gen::gen_core::LoadShape) -> MiniMaxH3 {
        let spec =
            LoadSpec::new(WeightsSource::Dir(PathBuf::from("/snap"))).with_load_shape(load_shape);
        MiniMaxH3 {
            descriptor: descriptor(),
            root: PathBuf::from("/snap"),
            dit_dir: PathBuf::from(dit_dir),
            text_encoder_dir: PathBuf::from("/snap/text_encoder"),
            dtype: Dtype::Bfloat16,
            adapters: Vec::new(),
            load_shape,
            memory_strategy: crate::memory_strategy::contract_for(&spec).expect("contract"),
            memory_tier: crate::memory_strategy::numeric_tier(&spec),
        }
    }

    /// The shape rung 4 is `Implemented` at. NOT `LoadSpec::new`'s default — that is
    /// `EagerMaterialization`; a caller must ask for the deferred shape, and the resident-load
    /// refusal below constructs the eager twin explicitly.
    fn generator_at(dit_dir: &str) -> MiniMaxH3 {
        generator_with(
            dit_dir,
            mlx_gen::gen_core::LoadShape::DeferredMaterialization,
        )
    }

    /// Write the structurally complete **weights-free** snapshot [`load`] probes for.
    ///
    /// Every path here is stat'd or parsed as JSON by the loader; not one tensor is read. The
    /// shards are **sparse** files of the measured component sizes — `safetensors_path_bytes`
    /// stats rather than parses — so the contract resolved off this tree carries the family's real
    /// asset facts at no disk cost. That is what lets the assertions below distinguish the
    /// production contract from the zero-footprint weights-free one.
    fn weightless_snapshot(root: &Path) {
        for (dir, probe) in [
            (BASE_DIT_PARTITION, "config.json"),
            (REFERENCE_DIT_PARTITION, "config.json"),
            ("text_encoder", "config.json"),
            ("vae", "config.json"),
            ("audio_vae", "config.json"),
            ("tokenizer", "tokenizer.json"),
            ("FL2VA/audio_vae", "config.json"),
            ("FL2VA/audio_vae", "config.yaml"),
            ("FL2VA/audio_vae", "metadata.json"),
        ] {
            let dir = root.join(dir);
            std::fs::create_dir_all(&dir).expect("component dir");
            std::fs::write(dir.join(probe), b"{}").expect("probe document");
        }
        for (component, bytes) in [
            (BASE_DIT_PARTITION, crate::memory_strategy::DIT_BF16_BYTES),
            ("text_encoder", crate::memory_strategy::TEXT_ENCODER_BYTES),
            ("vae", crate::memory_strategy::VIDEO_VAE_BYTES),
            ("audio_vae", crate::memory_strategy::AUDIO_VAE_BYTES),
        ] {
            let shard = std::fs::File::create(root.join(component).join("model.safetensors"))
                .expect("shard");
            shard.set_len(bytes).expect("sparse shard");
        }
    }

    /// **The loaded generator publishes the provider's memory-strategy contract** (sc-18650).
    ///
    /// This is the seam SceneWorks admission reads: `Generator::memory_strategy_contract` on the
    /// boxed generator `load` hands back. `mlx_gen::impl_generator!` emits only `descriptor` /
    /// `validate` / `generate`, so before this story the vtable slot resolved to the trait default
    /// `None` and every MiniMax-H3 render skipped request-geometry admission entirely — while
    /// `memory_strategy::contract_for` built a full contract that nothing ever asked for.
    ///
    /// Asserted through `dyn Generator` rather than on the concrete struct, because the default is
    /// what a `Box<dyn Generator>` would dispatch to.
    ///
    /// Both load shapes are driven, because rung 4's support is the one entry that varies with the
    /// shape — a single-shape test would pass against a contract pinned to either constant.
    #[test]
    fn a_loaded_generator_publishes_the_providers_memory_strategy_contract() {
        use mlx_gen::gen_core::{LoadShape, MemoryStrategy, MemoryStrategySupport};
        let root = tempfile::tempdir().expect("fixture root");
        weightless_snapshot(root.path());

        for (load_shape, rung4) in [
            (
                LoadShape::DeferredMaterialization,
                MemoryStrategySupport::Implemented,
            ),
            (
                LoadShape::EagerMaterialization,
                MemoryStrategySupport::Missing,
            ),
        ] {
            let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()))
                .with_load_shape(load_shape);
            let generator = load(&spec).expect("weights-free load");
            let published = generator
                .memory_strategy_contract()
                .expect("a loaded generator must publish its memory-strategy contract");

            // It is the provider's own contract for THIS spec — not some other shape's, and not the
            // resident-only compatibility default the trait falls back to.
            assert_eq!(
                published,
                &crate::memory_strategy::contract_for(&spec).expect("contract")
            );
            assert_eq!(published.load_shape, load_shape);
            // ...and the PRODUCTION one. The weights-free fixture contract differs from it only in
            // asset facts, so every assertion below would hold against it too; this is the one that
            // proves `load` resolved the snapshot it had just probed.
            assert_ne!(
                published,
                &crate::memory_strategy::weights_free_contract(&spec).expect("fixture contract"),
                "the published contract must carry the resolved snapshot's asset facts"
            );
            assert_eq!(
                published.asset_facts.transformer_bytes,
                crate::memory_strategy::DIT_BF16_BYTES
            );

            // The rungs it advertises are `memory_strategy::strategies()`'s, for this load shape.
            let support = |strategy: MemoryStrategy| {
                published
                    .capability(strategy)
                    .unwrap_or_else(|| panic!("{strategy:?} must appear in the strategy table"))
                    .support
                    .clone()
            };
            for implemented in [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ] {
                assert_eq!(
                    support(implemented),
                    MemoryStrategySupport::Implemented,
                    "{implemented:?}"
                );
            }
            // Rung 3 is measured inapplicable on MLX for this family, at every shape.
            assert!(
                matches!(
                    support(MemoryStrategy::BoundedAttention),
                    MemoryStrategySupport::StructurallyNotApplicable { .. }
                ),
                "rung 3 must stay structurally inapplicable, got {:?}",
                support(MemoryStrategy::BoundedAttention)
            );
            // Rung 4 exists only where its shared prerequisite — the deferred shape — does.
            assert_eq!(
                support(MemoryStrategy::BoundedTransformerResidency),
                rung4,
                "{load_shape:?}"
            );
        }
    }

    /// **The published safety check and request scope refuse what this provider cannot run**
    /// (sc-18650).
    ///
    /// Publishing a contract without these two is worse than publishing nothing: the trait's own
    /// `memory_strategy_safety_check` default would accept any selection the capability table
    /// allows **without** this family's `route_gate`, and its `begin_memory_strategy_request`
    /// default hard-errors for a contract declaring implemented optimized rungs. So each arm here
    /// is a live guard, not a restatement of `memory_strategy`'s own tests:
    ///
    /// * the accepted arm dies if `begin_memory_strategy_request` is dropped (the default errors);
    /// * the geometry arms die if `memory_strategy_safety_check` is dropped (the default runs no
    ///   route gate and admits both);
    /// * the rung-4 arm is the same context judged by two generators, so it cannot pass by
    ///   accident: it fails if the shape stops reaching the declaration.
    #[test]
    fn the_published_admission_refuses_selections_this_provider_cannot_serve() {
        use mlx_gen::gen_core::{LoadShape, MemorySafetyDecision, MemoryStrategy};
        let root = tempfile::tempdir().expect("fixture root");
        weightless_snapshot(root.path());
        let spec_at = |load_shape| {
            LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf())).with_load_shape(load_shape)
        };
        let deferred_spec = spec_at(LoadShape::DeferredMaterialization);
        let deferred = load(&deferred_spec).expect("weights-free load");
        let contract = deferred.memory_strategy_contract().expect("contract");

        // The provider's own fixture builder supplies the contexts, so every parameter is one this
        // contract published — a hand-rolled context would risk testing the parameter validator
        // instead of the rung.
        let fixture_for = |strategy| {
            crate::memory_strategy::registered_fixture(&deferred_spec, contract, strategy)
                .expect("fixtures")
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("{strategy:?} must publish at least one route fixture"))
        };

        // --- accepted: the legal rung-2 selection admits, and opens a real scope ----------------
        let admitted = fixture_for(MemoryStrategy::BoundedDecode).context;
        assert_eq!(
            deferred.memory_strategy_safety_check(&admitted),
            MemorySafetyDecision::Accept
        );
        assert!(
            deferred
                .begin_memory_strategy_request(&admitted)
                .expect("an admitted selection must open a scope, not error")
                .is_some(),
            "an implemented optimized rung must hand back a live request scope"
        );

        // --- refused: a rung this LOADED route does not have ------------------------------------
        // The identical context, judged by the eager twin whose contract declares rung 4 `Missing`.
        let streamed = fixture_for(MemoryStrategy::BoundedTransformerResidency).context;
        assert_eq!(
            deferred.memory_strategy_safety_check(&streamed),
            MemorySafetyDecision::Accept,
            "the deferred load implements rung 4"
        );
        let eager = load(&spec_at(LoadShape::EagerMaterialization)).expect("weights-free load");
        assert!(
            matches!(
                eager.memory_strategy_safety_check(&streamed),
                MemorySafetyDecision::Reject { .. }
            ),
            "a resident load must refuse block streaming, not silently accept it"
        );
        assert!(
            eager.begin_memory_strategy_request(&streamed).is_err(),
            "a refused selection must not open a scope"
        );

        // --- refused: a geometry the render itself would reject ---------------------------------
        // Off the `17n + 5` lattice. Only `route_gate` knows this, and only the override runs it.
        let mut off_lattice = admitted.clone();
        off_lattice.geometry.frames += 1;
        let MemorySafetyDecision::Reject { reason } =
            deferred.memory_strategy_safety_check(&off_lattice)
        else {
            panic!("an off-lattice frame count must be refused");
        };
        assert!(reason.contains("17n+5"), "{reason}");
        assert!(
            deferred
                .begin_memory_strategy_request(&off_lattice)
                .is_err(),
            "a refused geometry must not open a scope"
        );

        // --- refused: a decode tile outside the PUBLISHED PARAMETER DOMAIN -----------------------
        //
        // Deliberately not titled "`validate_decode_geometry` runs": it does not, and cannot, from
        // here. `contract.validate_selection` checks the owned parameter domains *before*
        // `route_gate` is ever called, and rung 2's published domain is the same singleton
        // `validate_decode_geometry` admits — so no value exists that clears one and trips the
        // other, and every out-of-domain tile is refused by the domain check. That predicate's own
        // seam is the request scope's `configure_decode`, which `memory_strategy`'s rung-2 chain
        // test drives end to end. What this arm pins is the domain check itself, by its message.
        //
        // Only the edge is moved: the fixture selection already carries both published parameters,
        // so this is a one-axis mutation of a complete, admitted selection rather than a selection
        // with a hole in it — and the assertion names which axis was refused.
        let mut retiled = admitted;
        retiled.selection.parameters.decode_tile_edge =
            Some(crate::memory_strategy::DECODE_TILE_EDGE + SPATIAL_STRIDE);
        let MemorySafetyDecision::Reject { reason } =
            deferred.memory_strategy_safety_check(&retiled)
        else {
            panic!("an out-of-domain decode tile must be refused");
        };
        assert!(
            reason.contains("outside the declared production candidates"),
            "{reason}"
        );

        // --- refused: rung 4 on an ADAPTER-CARRYING deferred load (sc-18650) ----------------------
        //
        // The blocker this arm exists for. `requested_transformer_window` has always refused a
        // windowed render with adapters, but until this story the contract did not know: `streamable`
        // read the load shape alone, so this exact spec published rung 4 `Implemented`, accepted the
        // selection, opened a scope, and only then hit a hard `Unsupported` inside `generate`.
        //
        // `MemoryRunContext` carries no adapter axis, so this cannot be a route-gate refusal — the
        // declaration is the only seam that can see it, which is why the assertion is on `support`
        // as well as on the two admission seams.
        let lora = root.path().join("turbo.safetensors");
        std::fs::write(&lora, b"\x00").expect("adapter file");
        let adapted_spec =
            spec_at(LoadShape::DeferredMaterialization).with_adapters(vec![AdapterSpec {
                path: lora,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }]);
        let adapted = load(&adapted_spec).expect("weights-free load with an adapter");
        let adapted_contract = adapted.memory_strategy_contract().expect("contract");
        assert_eq!(
            adapted_contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("rung 4 must appear in the strategy table")
                .support,
            mlx_gen::gen_core::MemoryStrategySupport::Missing,
            "an adapter load must declare rung 4 Missing — the window would drop the factors"
        );
        assert!(
            matches!(
                adapted.memory_strategy_safety_check(&streamed),
                MemorySafetyDecision::Reject { .. }
            ),
            "an adapter load must refuse block streaming at the safety check"
        );
        assert!(
            adapted.begin_memory_strategy_request(&streamed).is_err(),
            "an adapter load must not open a streaming scope"
        );
        // ...and the adapter's resident bytes are charged rather than declared away (sc-18650).
        assert_eq!(
            adapted_contract.asset_facts.overlay_bytes, 1,
            "the forward-time residual factors are resident for the whole render"
        );
        assert_eq!(
            contract.asset_facts.overlay_bytes, 0,
            "a render with no adapter has no overlay"
        );
    }

    /// **The rung-4 admission mirrors the contract in both directions** (sc-18662).
    ///
    /// `requested_transformer_window` is the single seam every render's residency decision goes
    /// through — `generate_impl` resolves it once and threads it to the conditioning, the DiT load
    /// and the `JointDit` construction on all three routes. Each guard is mutated individually:
    /// a streamed request must come back `Some(1)` (not `None` — a silent downgrade to the
    /// resident path would be this test's false green), and each refusal must name its reason.
    #[test]
    fn transformer_streaming_is_admitted_and_refused_exactly_as_declared() {
        use mlx_gen::gen_core::{GenerationMemory, LoadShape, TransformerComponent};
        let streamed = |memory: GenerationMemory| GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            memory: Some(memory),
            ..Default::default()
        };
        let stream_request = GenerationMemory {
            stage_residency: true,
            stream_transformer_blocks: true,
            transformer_window_size: Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: Some(TransformerComponent::Both),
            ..Default::default()
        };
        let generator = generator_at("/snap/transformer");

        // The happy path admits the published window — `Some`, never a silent `None`.
        assert_eq!(
            generator
                .requested_transformer_window(&streamed(stream_request))
                .expect("the declared selection must be admitted"),
            Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE)
        );
        // An untouched request is byte-for-byte unaffected: no memory block, no window.
        assert_eq!(
            generator
                .requested_transformer_window(&request(576, 320))
                .expect("a request without a memory block is the resident path"),
            None
        );
        // `stream_transformer_blocks: false` is the resident path even with parameters present.
        assert_eq!(
            generator
                .requested_transformer_window(&streamed(GenerationMemory {
                    stream_transformer_blocks: false,
                    ..stream_request
                }))
                .expect("an unstreamed selection is the resident path"),
            None
        );

        // A resident load genuinely does not have the rung (its contract declares `Missing`).
        let resident = generator_with("/snap/transformer", LoadShape::EagerMaterialization);
        let refusal = resident
            .requested_transformer_window(&streamed(stream_request))
            .expect_err("a resident load must refuse streaming, not downgrade it")
            .to_string();
        assert!(refusal.contains("DeferredMaterialization"), "{refusal}");

        // Adapters would be silently dropped from every windowed block — typed refusal.
        let adapted = MiniMaxH3 {
            adapters: vec![AdapterSpec {
                path: PathBuf::from("/nonexistent.safetensors"),
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            ..generator_at("/snap/transformer")
        };
        let refusal = adapted
            .requested_transformer_window(&streamed(stream_request))
            .expect_err("a streamed render with adapters must be refused")
            .to_string();
        assert!(refusal.contains("adapter"), "{refusal}");

        // The component defaults to `Dit` (SC-15794), which is deliberately outside this
        // provider's published domain — so both the explicit and the defaulted forms refuse.
        for component in [None, Some(TransformerComponent::Dit)] {
            let refusal = generator
                .requested_transformer_window(&streamed(GenerationMemory {
                    transformer_window_component: component,
                    ..stream_request
                }))
                .expect_err("a Dit-only window must be refused (AC3)")
                .to_string();
            assert!(refusal.contains("Both"), "{refusal}");
        }

        // A window outside the measured singleton domain is an advertised lever that does nothing.
        let refusal = generator
            .requested_transformer_window(&streamed(GenerationMemory {
                transformer_window_size: Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE + 1),
                ..stream_request
            }))
            .expect_err("an out-of-domain window must be refused")
            .to_string();
        assert!(
            refusal.contains("outside the published domain"),
            "{refusal}"
        );
        // `None` means the provider's own default and is admitted.
        assert_eq!(
            generator
                .requested_transformer_window(&streamed(GenerationMemory {
                    transformer_window_size: None,
                    ..stream_request
                }))
                .expect("a defaulted window size uses the provider's constant"),
            Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE)
        );
    }

    /// **`decode_video` consults the request's rung-2 selection, and does so before it maps the
    /// 10.42 GB VAE** (sc-18660 AC4).
    ///
    /// This is the call-site half of reachability. `memory_strategy` proves the selection travels
    /// as far as `pipeline::decode_tiling_for`; nothing there can prove `decode_video` *calls* it —
    /// a doc comment saying so is not evidence. This drives the real method on a root with no
    /// weights at all and pins **which** error comes back:
    ///
    /// * an unadmitted geometry ⇒ the tiling refusal, proving the resolution ran;
    /// * an admitted geometry ⇒ the missing-`config.json` error, proving the refusal is not a
    ///   blanket failure and that the method proceeds to the load.
    ///
    /// Deleting the `decode_tiling_for` call turns the first arm into the second and fails here.
    #[test]
    fn decode_video_resolves_the_requests_tiling_before_it_loads_the_vae() {
        use mlx_gen::gen_core::GenerationMemory;
        let generator = generator_at("/snap/transformer");
        let latents = mlx_rs::Array::from_slice(&[0.0f32; 24], &[1, 24, 1, 1, 1]);
        let request = |edge: u32, overlap: u32| GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(edge),
                decode_overlap: Some(overlap),
                ..Default::default()
            }),
            ..Default::default()
        };

        let refused = generator
            .decode_video(&request(128, 32), &latents)
            .expect_err("an unadmitted tile must be refused")
            .to_string();
        assert!(
            refused.contains("bounded decode admits only the reference geometry"),
            "expected the rung-2 refusal, got: {refused}"
        );
        assert!(
            !refused.contains("config.json"),
            "the refusal must fire BEFORE the VAE load, got: {refused}"
        );

        // A starved overlap takes the same path — the corruption it causes is cross-frame and no
        // memory number could see it, so admission is the only place it can be stopped.
        let starved = generator
            .decode_video(
                &request(crate::memory_strategy::DECODE_TILE_EDGE, 0),
                &latents,
            )
            .expect_err("a zero overlap must be refused")
            .to_string();
        assert!(starved.contains("bounded decode admits only the reference geometry"));

        // The control arm: an ADMITTED geometry gets past the resolution and fails at the load,
        // which is what proves the refusal above is specific rather than a blanket rejection.
        let loaded = generator
            .decode_video(
                &request(
                    crate::memory_strategy::DECODE_TILE_EDGE,
                    crate::memory_strategy::DECODE_OVERLAP,
                ),
                &latents,
            )
            .expect_err("there are no weights at /snap")
            .to_string();
        assert!(
            loaded.contains("config.json"),
            "an admitted geometry must proceed to the load, got: {loaded}"
        );
    }

    /// `ref2va` reads its partition from **beside the resolved DiT**, not from under the root.
    ///
    /// The sc-17149 / sc-17150 seam. Before tiering, `ref2va` loaded `root.join("transformer_ref")`
    /// and that was right because `dit_dir` was always `root/transformer`. On a tiered install the
    /// DiT is staged outside the snapshot root entirely — `root/transformer_ref` does not exist, and
    /// resolving there would either fail at load or, worse, find a stale dense checkpoint next to a
    /// `q4` base and silently mix tiers across two branches of the same render.
    #[test]
    fn the_reference_partition_follows_the_tier_not_the_root() {
        // Flat upstream layout: the sibling IS the root-relative path, so nothing regresses.
        let flat = generator_at("/snap/transformer");
        assert_eq!(
            flat.task_dit_dir(MiniMaxH3Task::Ref2va),
            PathBuf::from("/snap/transformer_ref")
        );
        assert_eq!(
            flat.task_dit_dir(MiniMaxH3Task::T2va),
            PathBuf::from("/snap/transformer")
        );

        // Tiered/split layout: both partitions come from the SAME tier directory, and neither is
        // under `/snap`. This is the case `root.join(partition)` got wrong.
        let tiered = generator_at("/tiers/q4/transformer");
        assert_eq!(
            tiered.task_dit_dir(MiniMaxH3Task::Ref2va),
            PathBuf::from("/tiers/q4/transformer_ref")
        );
        assert_eq!(
            tiered.task_dit_dir(MiniMaxH3Task::T2va),
            PathBuf::from("/tiers/q4/transformer")
        );
        assert_eq!(
            tiered.task_dit_dir(MiniMaxH3Task::Fl2va),
            tiered.task_dit_dir(MiniMaxH3Task::T2va),
            "fl2va shares the base partition"
        );

        // The two partitions are never the same directory — the whole point of the split.
        for dir in ["/snap/transformer", "/tiers/q8/transformer"] {
            let g = generator_at(dir);
            assert_ne!(
                g.task_dit_dir(MiniMaxH3Task::Ref2va),
                g.task_dit_dir(MiniMaxH3Task::T2va)
            );
        }
    }

    /// The descriptor's shape: a Mac-only video model with no guidance surface and no conditioning.
    #[test]
    fn the_descriptor_declares_a_guidance_free_video_model() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.backend, "mlx");
        assert!(matches!(d.modality, Modality::Video));
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert_eq!(d.capabilities.max_count, 1);
        assert_eq!(d.capabilities.min_size, SPATIAL_STRIDE);
        assert!(d.capabilities.mac_only);
        assert!(
            d.capabilities.audio_sample_rates.is_empty(),
            "the audio surface describes an audio REQUEST, which a video model has none of"
        );
    }

    /// **The geometry gate fires from `validate`, before any weight is read.** This is the
    /// acceptance criterion's "rejected, not silently refit" at the contract boundary.
    #[test]
    fn validate_rejects_off_lattice_geometry() {
        let caps = descriptor().capabilities;

        // An explicit off-lattice frame count.
        for frames in [123u32, 125, 129, 200] {
            let req = GenerationRequest {
                frames: Some(frames),
                ..request(576, 320)
            };
            let e = validate_request(&caps, &req).unwrap_err().to_string();
            assert!(e.contains("17n + 5"), "{frames}: {e}");
        }
        // A canvas that is 16-aligned but not 32-aligned.
        let e = validate_request(&caps, &request(592, 320))
            .unwrap_err()
            .to_string();
        assert!(e.contains("multiple of 32"), "{e}");

        // ...and the legal request passes.
        validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                ..request(576, 320)
            },
        )
        .unwrap();
    }

    /// Every resolution SceneWorks advertises for `minimax_h3` and `minimax_h3_ref`.
    ///
    /// Copied from `config/manifests/builtin.models.jsonc` — both partitions declare this exact
    /// list, and it drives the resolution `<select>` in `apps/web/src/resolutionOverride.js`. A
    /// bucket the menu offers and the engine refuses is a submit error the user cannot avoid, so
    /// this list is the acceptance criterion rather than an illustration.
    const ADVERTISED_BUCKETS: [(u32, u32); 9] = [
        (1536, 672),
        (672, 1536),
        (1344, 768),
        (768, 1344),
        (1024, 768),
        (768, 1024),
        (768, 768),
        (576, 320),
        (320, 576),
    ];

    /// **The per-edge ceiling is the widest canvas the model's own resolver emits** (sc-17152).
    ///
    /// The number is *derived here*, not restated: a maintained literal inside the test that exists
    /// to catch a wrong literal proves nothing. The sweep walks the model's whole legal aspect
    /// range and asks [`crate::keyframe::resolve_canvas_size`] — the arithmetic that turns a
    /// reference's ratio into a canvas — for each canvas, then asserts `max_size` equals the widest
    /// edge that came back. A ceiling below it would refuse a canvas the model resolves to on its
    /// own; above it would advertise an edge nothing can produce.
    #[test]
    fn the_capability_ceiling_is_the_widest_canvas_the_resolver_emits() {
        const STEPS: u32 = 40_000;
        let mut widest = 0u32;
        for i in 0..=STEPS {
            let ratio = crate::keyframe::MIN_ASPECT_RATIO
                + (crate::keyframe::MAX_ASPECT_RATIO - crate::keyframe::MIN_ASPECT_RATIO)
                    * f64::from(i)
                    / f64::from(STEPS);
            let (h, w) = crate::keyframe::resolve_canvas_size(
                ratio,
                1.0,
                SPATIAL_STRIDE as i32,
                crate::pipeline::CANVAS_SHORT_EDGE as i32,
                i64::from(crate::pipeline::CANVAS_MAX_PIXELS),
            )
            .unwrap();
            widest = widest.max(w as u32).max(h as u32);
        }
        assert_eq!(
            descriptor().capabilities.max_size,
            widest,
            "max_size must be the widest edge `resolve_canvas_size` can emit across 1:4..=4:1"
        );
        assert_eq!(descriptor().capabilities.max_size, MAX_CANVAS_EDGE);

        // The advertised menu must fit under it — this is the defect the story was filed for.
        let advertised = ADVERTISED_BUCKETS
            .iter()
            .flat_map(|&(w, h)| [w, h])
            .max()
            .unwrap();
        assert!(
            advertised <= descriptor().capabilities.max_size,
            "SceneWorks advertises a {advertised} px edge; the engine ceiling is {}",
            descriptor().capabilities.max_size
        );
    }

    /// **Every advertised bucket validates** — the user-visible half of sc-17152.
    #[test]
    fn every_advertised_resolution_bucket_validates() {
        let caps = descriptor().capabilities;
        for (w, h) in ADVERTISED_BUCKETS {
            // The manifest list must itself be inside the checkpoint's envelope, or the engine is
            // right to refuse and the manifest is the thing that is wrong.
            assert!(
                u64::from(w) * u64::from(h) <= u64::from(crate::pipeline::CANVAS_MAX_PIXELS),
                "{w}x{h} is over the area budget"
            );
            validate_request(
                &caps,
                &GenerationRequest {
                    frames: Some(124),
                    ..request(w, h)
                },
            )
            .unwrap_or_else(|e| panic!("{w}x{h} is advertised but refused: {e}"));
        }
    }

    /// **Raising the per-edge ceiling did not widen the area envelope** (sc-17152).
    ///
    /// Both shapes here are *inside* `max_size` on both edges and 32-aligned, so the per-edge gate
    /// and the stride gate are both satisfied and only the area gate can refuse them — asserted
    /// explicitly, because a refusal that came from the edge gate would make this test pass while
    /// proving the opposite of what it claims. The error is named rather than checked with
    /// `is_err`.
    #[test]
    fn an_over_area_canvas_is_still_refused_by_the_area_gate() {
        let caps = descriptor().capabilities;
        for (w, h) in [(1536u32, 1536u32), (1344u32, 1344u32)] {
            assert!(
                w <= caps.max_size && h <= caps.max_size,
                "{w}x{h} must be INSIDE the per-edge ceiling or this proves nothing"
            );
            assert!(w.is_multiple_of(SPATIAL_STRIDE) && h.is_multiple_of(SPATIAL_STRIDE));
            assert!(u64::from(w) * u64::from(h) > u64::from(crate::pipeline::CANVAS_MAX_PIXELS));

            let e = validate_request(
                &caps,
                &GenerationRequest {
                    frames: Some(124),
                    ..request(w, h)
                },
            )
            .unwrap_err()
            .to_string();
            assert!(e.contains("canvas budget"), "{w}x{h}: {e}");
            assert!(
                !e.contains("outside supported range"),
                "{w}x{h} must be refused by the AREA gate, not the per-edge gate: {e}"
            );
        }

        // The one shape the raised ceiling newly admits: the 4:1 canvas the resolver itself emits.
        // The short edge is derived — the ceiling divides the area budget exactly, which is what
        // makes `MAX_CANVAS_EDGE x short` sit at the budget rather than over it.
        let short = crate::pipeline::CANVAS_MAX_PIXELS / MAX_CANVAS_EDGE;
        assert_eq!(short * MAX_CANVAS_EDGE, crate::pipeline::CANVAS_MAX_PIXELS);
        assert!(short.is_multiple_of(SPATIAL_STRIDE));
        for (w, h) in [(MAX_CANVAS_EDGE, short), (short, MAX_CANVAS_EDGE)] {
            validate_request(
                &caps,
                &GenerationRequest {
                    frames: Some(124),
                    ..request(w, h)
                },
            )
            .unwrap_or_else(|e| panic!("{w}x{h} is the resolver's own 4:1 canvas: {e}"));
        }

        // …and one stride past it on the long edge is outside the per-edge ceiling again.
        let e = validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                ..request(MAX_CANVAS_EDGE + SPATIAL_STRIDE, short)
            },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("outside supported range"), "{e}");
    }

    /// sc-18729 — **the video sigma shift is a per-request override, and the default is a byte-exact
    /// no-op.**
    ///
    /// Three claims, because two of them are individually satisfiable by a wrong implementation:
    ///
    /// 1. `scheduler_shift: None` builds exactly what `JointSchedule::new` builds. An implementation
    ///    that quietly defaulted to, say, 6.0 would still "honor the knob" by every other check
    ///    here — this is the arm that pins the base checkpoint's schedule unchanged.
    /// 2. `Some(6.0)` moves the **video** grid and leaves the **audio** grid alone. Asserting only
    ///    that something changed would pass an implementation that fed one shift to both modalities,
    ///    which is the sc-17146 defect: it moves audio 1.81e-1 at cosine 0.9846.
    /// 3. A non-positive shift is refused at `validate`, not 20 minutes later inside
    ///    `SigmaSchedule::with_shift` — `validate` is the only gate that runs before the 53 GB text
    ///    encoder maps.
    #[test]
    fn the_video_sigma_shift_is_overridable_and_defaults_to_the_published_grid() {
        let caps = descriptor().capabilities;

        // (1) The default path is the published pair, exactly.
        let default = JointSchedule::with_shifts(9, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT).unwrap();
        assert_eq!(
            default,
            JointSchedule::new(9).unwrap(),
            "an unset scheduler_shift must reproduce the base checkpoint's schedule exactly"
        );

        // (2) The turbo 768p shift moves the video grid and only the video grid.
        let turbo = JointSchedule::with_shifts(5, 6.0, AUDIO_SIGMA_SHIFT).unwrap();
        assert_ne!(
            turbo.video(),
            default.video(),
            "shift 6.0 must produce a different video sigma grid than 12.0"
        );
        assert_eq!(
            turbo.audio().shift(),
            AUDIO_SIGMA_SHIFT,
            "the audio shift must stay at its published 3.0 — the override is video-only"
        );

        // (3) Refused at the contract boundary, with the knob named.
        for bad in [0.0f32, -1.0] {
            let req = GenerationRequest {
                frames: Some(124),
                scheduler_shift: Some(bad),
                ..request(576, 320)
            };
            let e = validate_request(&caps, &req).unwrap_err().to_string();
            assert!(
                e.contains("scheduler_shift") && e.contains("positive"),
                "shift {bad}: {e}"
            );
        }
        // ...and the documented turbo value passes.
        validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                scheduler_shift: Some(6.0),
                ..request(576, 320)
            },
        )
        .expect("the documented 768p turbo shift must validate");
    }

    /// `frames` is exact and `duration` is aligned — the split, pinned in both directions.
    #[test]
    fn frames_are_exact_and_duration_is_aligned() {
        // 5.3 s has no lattice point; it aligns UP to 141 frames.
        let by_duration = GenerationRequest {
            duration: Some(5.3),
            ..request(576, 320)
        };
        assert_eq!(
            request_geometry(&by_duration).unwrap().joint.num_frames,
            141
        );

        // The same clip requested as 127 frames is refused rather than aligned.
        let by_frames = GenerationRequest {
            frames: Some(127),
            ..request(576, 320)
        };
        assert!(request_geometry(&by_frames).is_err());

        // `frames` wins when both are given.
        let both = GenerationRequest {
            frames: Some(124),
            duration: Some(15.0),
            ..request(576, 320)
        };
        assert_eq!(request_geometry(&both).unwrap().joint.num_frames, 124);

        // No frames, no duration: the shortest legal render.
        assert_eq!(
            request_geometry(&request(576, 320))
                .unwrap()
                .joint
                .num_frames,
            SMALLEST_LEGAL_FRAMES
        );
    }

    /// The model generates at 24 fps and says so rather than quietly retiming.
    #[test]
    fn a_foreign_frame_rate_is_rejected() {
        let req = GenerationRequest {
            fps: Some(30),
            ..request(576, 320)
        };
        let e = request_geometry(&req).unwrap_err().to_string();
        assert!(e.contains("24 fps"), "{e}");
    }

    fn image(w: u32, h: u32) -> mlx_gen::media::Image {
        mlx_gen::media::Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    fn image_ref() -> Conditioning {
        Conditioning::Reference {
            image: image(64, 64),
            strength: None,
        }
    }

    fn track() -> mlx_gen::media::AudioTrack {
        mlx_gen::media::AudioTrack {
            samples: vec![0.0; 64],
            // The engine ships no resampler, so `Ref2VaReferences::new` admits only this rate.
            sample_rate: AUDIO_SAMPLE_RATE,
            channels: 1,
            stems: Vec::new(),
        }
    }

    fn clip_ref() -> Conditioning {
        Conditioning::ReferenceVideo {
            frames: vec![image(64, 64)],
            fps: 24.0,
            audio: None,
        }
    }

    fn audio_ref() -> Conditioning {
        Conditioning::ReferenceAudio {
            audio: track(),
            strength: None,
        }
    }

    fn with(conditioning: Vec<Conditioning>) -> GenerationRequest {
        GenerationRequest {
            frames: Some(124),
            conditioning,
            ..request(576, 320)
        }
    }

    /// **The task selects the checkpoint, and only `ref2va` moves.**
    ///
    /// The two partitions are byte-different but structurally identical — same config, same 638
    /// tensor names — so this mapping is the *only* thing standing between a `ref2va` request and a
    /// plausible render off the wrong 66 GB of weights.
    #[test]
    fn the_task_picks_the_partition_and_only_ref2va_moves() {
        assert_eq!(
            MiniMaxH3Task::resolve(false, false).unwrap(),
            MiniMaxH3Task::T2va
        );
        assert_eq!(
            MiniMaxH3Task::resolve(true, false).unwrap(),
            MiniMaxH3Task::Fl2va
        );
        assert_eq!(
            MiniMaxH3Task::resolve(false, true).unwrap(),
            MiniMaxH3Task::Ref2va
        );

        assert_eq!(MiniMaxH3Task::T2va.partition(), "transformer");
        assert_eq!(MiniMaxH3Task::Fl2va.partition(), "transformer");
        assert_eq!(MiniMaxH3Task::Ref2va.partition(), "transformer_ref");
        // Stated as an inequality too: if someone "simplifies" the match into one arm, the equality
        // assertions above could still be satisfied by a constant.
        assert_ne!(
            MiniMaxH3Task::Ref2va.partition(),
            MiniMaxH3Task::T2va.partition(),
            "ref2va must NOT read the base checkpoint"
        );

        // The vision tower is read by both grounded tasks and by neither text-only one.
        assert!(!MiniMaxH3Task::T2va.needs_vision_tower());
        assert!(MiniMaxH3Task::Fl2va.needs_vision_tower());
        assert!(MiniMaxH3Task::Ref2va.needs_vision_tower());

        // Both at once is refused rather than resolved to one of them.
        let e = MiniMaxH3Task::resolve(true, true).unwrap_err().to_string();
        assert!(e.contains("both keyframes and references"), "{e}");
    }

    /// **Every `ref2va` cap is enforced at the engine boundary**, by `validate`, before a weight is
    /// read — the story's second acceptance criterion.
    #[test]
    fn the_reference_caps_are_enforced_by_validate() {
        let caps = descriptor().capabilities;
        let err = |c: Vec<Conditioning>| {
            validate_request(&caps, &with(c))
                .expect_err("expected a rejection")
                .to_string()
        };

        // 10 images.
        let e = err(vec![image_ref(); 10]);
        assert!(e.contains("at most 9 image"), "{e}");
        // 4 clips.
        let e = err(vec![clip_ref(); 4]);
        assert!(e.contains("at most 3 video"), "{e}");
        // 4 audio (paired, so the audio-alone rule is not what fires).
        let mut four_audio = vec![image_ref()];
        four_audio.extend(vec![audio_ref(); 4]);
        let e = err(four_audio);
        assert!(e.contains("at most 3 audio"), "{e}");
        // 13 total, with every per-modality cap satisfied.
        let mut thirteen = vec![image_ref(); 9];
        thirteen.extend(vec![clip_ref(); 3]);
        thirteen.push(audio_ref());
        let e = err(thirteen);
        assert!(e.contains("at most 12 references in total"), "{e}");

        // …and the legal saturated shapes are ACCEPTED. Without these arms every assertion above
        // would still pass with a gate that rejected everything.
        let mut twelve = vec![image_ref(); 9];
        twelve.extend(vec![clip_ref(); 2]);
        twelve.push(audio_ref());
        validate_request(&caps, &with(twelve)).expect("9 + 2 + 1 = 12 is legal");
        validate_request(&caps, &with(vec![image_ref(); 9])).expect("9 images is legal");
        validate_request(&caps, &with(vec![clip_ref(); 3])).expect("3 clips is legal");
    }

    /// An in-context clip is **refused**, not silently ignored (sc-17149). This is the arm that
    /// makes dropping `ConditioningKind::VideoClip` from the descriptor a real gate rather than a
    /// cosmetic edit: `request_references` no longer reads the variant, so without default-deny a
    /// `VideoClip` would sail through and render a plausible clip that dropped the caller's
    /// conditioning entirely.
    #[test]
    fn an_in_context_video_clip_is_refused_as_unsupported() {
        let caps = descriptor().capabilities;
        let e = validate_request(
            &caps,
            &with(vec![Conditioning::VideoClip {
                frames: vec![image(64, 64)],
                frame_idx: 7,
                strength: 0.5,
            }]),
        )
        .unwrap_err();
        assert!(
            matches!(e, Error::Unsupported(_)),
            "an unadvertised kind must be the typed Unsupported, got {e:?}"
        );
    }

    /// A reference clip carries **its own frame rate**, and the rate survives the mapping.
    ///
    /// This is the hole the old `VideoClip` carrier could not close: it had no rate, so the mapping
    /// read `req.fps` — which `request_geometry` rejects unless it is exactly 24.0. A 30 fps
    /// reference was therefore conditioned on as 24 fps with nothing raised. Asserting `30.0`
    /// *while the request's own fps stays unset* is what pins that the two are separate quantities.
    #[test]
    fn a_reference_clip_carries_its_own_frame_rate() {
        let req = with(vec![Conditioning::ReferenceVideo {
            frames: vec![image(64, 64)],
            fps: 30.0,
            audio: None,
        }]);
        assert!(req.fps.is_none(), "the request's own output rate is unset");
        let refs = request_references(&req).unwrap().unwrap();
        match &refs.as_slice()[0] {
            Ref2VaReference::Video(v) => assert!(
                (v.fps - 30.0).abs() < f64::EPSILON,
                "the clip's own rate must survive, got {}",
                v.fps
            ),
            other => panic!("expected a video reference, got {other:?}"),
        }
        // And it is legal: a reference does not bind the output rate, so a 30 fps reference on a
        // 24 fps render is a valid request rather than a geometry conflict.
        validate_request(&descriptor().capabilities, &req).expect("a 30 fps reference is legal");
    }

    /// A reference clip's **own soundtrack** reaches the engine on the clip, not as a standalone
    /// audio reference — the distinction the whole carrier exists for.
    ///
    /// The two assertions are one claim each and both are needed: the soundtrack is *present* on the
    /// video reference, and it did *not* also become a reference of its own (which is what sending
    /// it as a separate `ReferenceAudio` would have done, consuming an audio cap slot and taking its
    /// own rotary origin instead of sharing the clip's).
    #[test]
    fn a_reference_clips_own_soundtrack_rides_the_clip() {
        let req = with(vec![Conditioning::ReferenceVideo {
            frames: vec![image(64, 64)],
            fps: 24.0,
            audio: Some(track()),
        }]);
        let refs = request_references(&req).unwrap().unwrap();
        assert_eq!(refs.as_slice().len(), 1, "one reference, not two");
        match &refs.as_slice()[0] {
            Ref2VaReference::Video(v) => {
                assert!(v.audio.is_some(), "the soundtrack must ride the clip")
            }
            other => panic!("expected a video reference, got {other:?}"),
        }
        assert_eq!(
            refs.audio_count(),
            0,
            "a clip's own soundtrack must not consume a standalone audio-reference slot"
        );
        validate_request(&descriptor().capabilities, &req)
            .expect("a lone video reference with its own soundtrack is legal");

        // The same waveform sent as a *standalone* reference is a different request: two
        // references, and one of them does consume an audio slot.
        let standalone = with(vec![clip_ref(), audio_ref()]);
        let refs = request_references(&standalone).unwrap().unwrap();
        assert_eq!(refs.as_slice().len(), 2);
        assert_eq!(refs.audio_count(), 1);
    }

    /// A request carrying keyframes **and** references is refused at the boundary — the two are
    /// different tasks on different checkpoints.
    #[test]
    fn keyframes_and_references_together_are_refused() {
        let caps = descriptor().capabilities;
        let e = validate_request(
            &caps,
            &with(vec![
                Conditioning::Keyframe {
                    image: image(576, 320),
                    frame_idx: 0,
                    strength: 1.0,
                },
                image_ref(),
            ]),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("both keyframes and references"), "{e}");
    }

    /// sc-19571 — **refuse, do not ignore.** MiniMax-H3 anchors at the checkpoint's trained-in
    /// `KEYFRAME_NOISE_AUG_T`, so `Keyframe.strength` has nothing to weight here; a request asking
    /// for a partial pin must be told so rather than rendered as a full one. Gated in BOTH places a
    /// request can arrive: `validate_request` (before the 53 GB text encoder maps) and
    /// `keyframe_anchors` (the render's own path), so neither can be the only one carrying it.
    ///
    /// Mutation guard: delete the `reject_keyframe_strength` call in `validate_request` and the
    /// first `unwrap_err` panics; delete the one in `keyframe_anchors` and the second does.
    #[test]
    fn a_keyframe_conditioning_strength_is_refused_not_ignored() {
        let caps = descriptor().capabilities;
        let kf = |strength: f32| Conditioning::Keyframe {
            image: image(576, 320),
            frame_idx: 0,
            strength,
        };
        // A full pin — the only thing the checkpoint expresses — is admitted.
        assert!(validate_request(&caps, &with(vec![kf(1.0)])).is_ok());

        let e = validate_request(&caps, &with(vec![kf(0.6)]))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("conditioning strength is not supported") && e.contains("0.999"),
            "{e}"
        );

        // …and the render path refuses it independently of validate.
        let img = image(576, 320);
        let refs = [KeyframeRef {
            image: &img,
            frame_idx: 0,
            strength: 0.6,
        }];
        let e = keyframe_anchors(&refs, 124).unwrap_err().to_string();
        assert!(e.contains("conditioning strength is not supported"), "{e}");
        // `keyframe_images` re-runs the same validation, so it cannot disagree.
        assert!(keyframe_images(&refs, 124).is_err());
    }

    /// The reference list keeps **request order** across modalities — order is semantic for
    /// `ref2va`, and re-sorting it would silently rewrite the request.
    #[test]
    fn the_reference_list_preserves_request_order_across_modalities() {
        let req = with(vec![clip_ref(), image_ref(), audio_ref(), image_ref()]);
        let refs = request_references(&req).unwrap().unwrap();
        let kinds: Vec<_> = refs.as_slice().iter().map(|r| r.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                crate::reference::ReferenceKind::Video,
                crate::reference::ReferenceKind::Image,
                crate::reference::ReferenceKind::Audio,
                crate::reference::ReferenceKind::Image,
            ],
            "grouping by modality would silently reorder a ref2va request"
        );
        // A request with no reference at all is `None`, which is what makes this the t2va/fl2va
        // discriminator too.
        assert!(request_references(&request(576, 320)).unwrap().is_none());
    }

    /// An empty prompt and an out-of-range step count are contract errors.
    #[test]
    fn malformed_requests_are_rejected() {
        let caps = descriptor().capabilities;
        let blank = GenerationRequest {
            prompt: "   ".into(),
            ..request(576, 320)
        };
        assert!(validate_request(&caps, &blank).is_err());
        let too_many = GenerationRequest {
            steps: Some(MAX_STEPS + 1),
            ..request(576, 320)
        };
        assert!(validate_request(&caps, &too_many).is_err());
        let none = GenerationRequest {
            steps: Some(0),
            ..request(576, 320)
        };
        assert!(validate_request(&caps, &none).is_err());
    }

    /// A missing partition is a load error naming the path, not a mid-render surprise.
    #[test]
    fn a_snapshot_missing_a_partition_fails_at_load() {
        let dir = tempfile::tempdir().unwrap();
        // `Box<dyn Generator>` is not `Debug`, so the error is destructured rather than unwrapped.
        let message = |spec: &LoadSpec| match load(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("this spec must not load"),
        };
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let e = message(&spec);
        assert!(e.contains("transformer"), "{e}");
        // A single file is the wrong shape entirely.
        let file = LoadSpec::new(WeightsSource::File(dir.path().join("x.safetensors")));
        let e = message(&file);
        assert!(e.contains("directory"), "{e}");
    }

    /// sc-18724 — the adapter surface is **declared and reachable**, and the two shared-spec knobs
    /// this model cannot honor are refused rather than silently ignored.
    ///
    /// `supports_lora` alone proves nothing: before this slice `load` rejected every adapter spec
    /// outright, so a `true` there would have advertised a path no request could take.
    #[test]
    fn the_adapter_surface_is_declared_and_the_unusable_knobs_are_refused() {
        let caps = descriptor().capabilities;
        assert!(caps.supports_lora);
        assert!(caps.supports_lokr);

        let dir = tempfile::tempdir().unwrap();
        let message = |spec: &LoadSpec| match load(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("this spec must not load"),
        };
        let missing = dir.path().join("nope.safetensors");
        let with = |adapter: AdapterSpec| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
            spec.adapters = vec![adapter];
            spec
        };

        // A mistyped path fails at load, naming the file — not 20 minutes into a render.
        let e = message(&with(AdapterSpec::new(
            missing.clone(),
            1.0,
            AdapterKind::Lora,
        )));
        assert!(e.contains("does not exist"), "{e}");
        assert!(e.contains("nope.safetensors"), "{e}");

        // A real file gets past the adapter gate and fails on the (absent) snapshot instead, which
        // is what proves the gate no longer refuses adapters wholesale.
        let real = dir.path().join("real.safetensors");
        std::fs::write(&real, b"not-really-safetensors").unwrap();
        let e = message(&with(AdapterSpec::new(
            real.clone(),
            1.0,
            AdapterKind::Lora,
        )));
        assert!(
            e.contains("transformer") && !e.contains("adapter"),
            "an existing adapter must fall through to the snapshot probe; got {e}"
        );

        // LTX's per-pass strengths and Wan's MoE expert selector have no meaning here.
        let mut per_pass = AdapterSpec::new(real.clone(), 1.0, AdapterKind::Lora);
        per_pass.pass_scales = Some(vec![1.0, 0.5]);
        let e = message(&with(per_pass));
        assert!(e.contains("per-pass scales"), "{e}");
        let mut expert = AdapterSpec::new(real, 1.0, AdapterKind::Lora);
        expert.moe_expert = Some(MoeExpert::High);
        let e = message(&with(expert));
        assert!(e.contains("MoE expert"), "{e}");
    }

    /// sc-18724 — the render-path seam: `load_task_dit` really folds the configured adapters into
    /// the DiT it maps, on the partition the task selects. The install itself is gated in
    /// `tests/turbo_lora.rs`; what this adds is that the **generator** reaches it at all.
    ///
    /// `MINIMAX_H3_TURBO_DIT` is a `transformer/` directory (any tier) and `MINIMAX_H3_TURBO_LORA`
    /// the downloaded `lightx2v/Minimax-h3-Turbo` dir.
    #[test]
    #[ignore = "needs a real MiniMax-H3 transformer/ (MINIMAX_H3_TURBO_DIT) + the turbo LoRA \
                (MINIMAX_H3_TURBO_LORA) + Metal"]
    fn the_render_path_folds_the_configured_adapters() {
        use mlx_gen::adapters::AdaptableHost;

        let dit_dir = std::env::var("MINIMAX_H3_TURBO_DIT").unwrap_or_default();
        assert!(
            !dit_dir.is_empty(),
            "MINIMAX_H3_TURBO_DIT must point at a MiniMax-H3 `transformer/` directory"
        );
        let lora_dir = std::env::var("MINIMAX_H3_TURBO_LORA").unwrap_or_default();
        assert!(
            !lora_dir.is_empty(),
            "MINIMAX_H3_TURBO_LORA must point at a lightx2v/Minimax-h3-Turbo snapshot dir"
        );
        let lora =
            PathBuf::from(lora_dir).join("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors");
        assert!(lora.is_file(), "missing {}", lora.display());

        let mut gen = generator_at(&dit_dir);
        gen.adapters = vec![AdapterSpec::new(lora, 1.0, AdapterKind::Lora)];
        let mut dit = gen
            .load_task_dit(MiniMaxH3Task::T2va)
            .unwrap_or_else(|e| panic!("load_task_dit: {e}"));

        // Every one of the 312 published targets carries exactly one residual, reached through the
        // generator rather than by calling the installer directly.
        let cfg = dit.config().clone();
        let mut adapted = 0usize;
        for path in crate::adapters::adapter_target_paths(&cfg) {
            let segs: Vec<&str> = path.split('.').collect();
            let lin = dit
                .adaptable_mut(&segs)
                .unwrap_or_else(|| panic!("unreachable target {path}"));
            assert_eq!(lin.adapters().len(), 1, "{path}");
            adapted += 1;
        }
        println!("[render seam] {adapted} module(s) folded via MiniMaxH3::load_task_dit");
        assert_eq!(adapted, 312);
    }

    /// The requested step count is model EVALUATIONS; the schedule adds the terminal σ = 0.
    #[test]
    fn the_requested_step_count_is_evaluations() {
        let s = JointSchedule::new(DEFAULT_STEPS as usize + 1).unwrap();
        assert_eq!(s.num_evals(), DEFAULT_STEPS as usize);
        assert_eq!(DEFAULT_STEPS, 50);
    }
}

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

// sc-27342: the descriptor, load path, request resolution, prompt encoding and the two
// generation bodies live in sibling modules; the provider type, its task routing and the
// Generator surface stay here.
mod descriptor;
mod generate;
mod load;
mod prompt;
mod ref2va;
mod request;
pub use descriptor::*;
pub use load::*;
pub(crate) use request::*;

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
/// [`GenerationReport::warnings`](mlx_gen::gen_core::GenerationReport::warnings) code: the text
/// encoder's GPU view of its weights needed re-reads before it verified (sc-23053, sc-27451).
pub const GPU_VISIBILITY_RECOVERED: &str = "gpu_visibility_recovered";

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

    /// The render plus its non-fatal warnings (sc-27451): GPU weight-visibility re-reads the text
    /// encoder needed before its token table verified (sc-23053) are reported as a
    /// [`GPU_VISIBILITY_RECOVERED`] warning, not only on stderr. The count is the same process-wide
    /// `coherence::retries` delta the encoder logs.
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::GenerationReport> {
        let before = mlx_gen::coherence::retries();
        let output = self.generate_impl(req, on_progress)?;
        let retries = mlx_gen::coherence::retries().saturating_sub(before);
        let mut report = mlx_gen::gen_core::GenerationReport::from_output(output);
        if retries != 0 {
            report.warnings.push(mlx_gen::gen_core::GenerationWarning {
                code: GPU_VISIBILITY_RECOVERED.to_owned(),
                message: format!(
                    "GPU weight visibility recovered after {retries} verification retries; \
                     generation was not retried"
                ),
            });
        }
        Ok(report)
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

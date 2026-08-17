//! The **`minimax_h3` generator on candle/CUDA** (sc-17156): prompt → real video frames plus a real
//! synchronized stereo soundtrack, `t2va` and `fl2va`.
//!
//! The Windows/Linux sibling of `mlx_gen_minimax_h3::model`. The checkpoint is guidance-distilled —
//! no negative prompt, no guidance scale, one transformer forward per step.
//!
//! # Phases, and why nothing is held resident
//!
//! The three heavy components are **66.71 GB** of Qwen3-VL-32B text encoder, **66.28 GB** of DiT
//! (40.43 GB once the AdaLN projections are evicted) and **10.42 GB** of video VAE. Holding any two
//! of them at once does not fit a card that exists, so `MiniMaxH3::generate_impl` builds each
//! phase's component, uses it, drops it and synchronizes the device before the next.
//! [`load`](MiniMaxH3::load) therefore holds **paths**, not tensors: it validates that every
//! partition the render needs is present and defers the reads.
//!
//! # The offload policy is FORCED, and that is what makes the VRAM gate honest
//!
//! [`OffloadPolicy`] is advisory in `gen-core` — "a provider that has not wired it falls back to
//! `Resident`". This provider inverts that: [`OFFLOAD_POLICY`] is `Sequential` **whatever the caller
//! asks for**, and [`MiniMaxH3::offload_policy`] reports what the render will actually do rather
//! than what was requested.
//!
//! That is not a convenience. `candle.vramGbByTier` in SceneWorks' manifest is a *measured*
//! sequential-phase peak, and the worker admits a card by comparing its free VRAM against it. If a
//! caller could ask for `Resident` and get it, the same manifest number would admit a render whose
//! true peak is the **sum** of the three components rather than their max — 143 GB against a
//! published ~40 GB. The number is only truthful because the load is pinned to match it, so the pin
//! is enforced here and **two** guards in this module's `tests` watch it, because the policy field
//! and the staging it stands for are separate facts:
//!
//! * `the_offload_policy_is_forced_sequential_even_when_resident_is_requested` pins the reported
//!   policy — a `load` that started echoing `spec.offload_policy` reds.
//! * `every_heavy_component_is_released_before_the_next_one_is_mapped` pins the **staging itself**.
//!   `self.offload` is read by no render path, so the first guard alone stays green if every
//!   `drop` / `release_device_memory` below is deleted. The second one reads this file's own
//!   production source and reds on the deletion of any single one of them. See its doc comment for
//!   what a source scan can and cannot see.
//!
//! # Geometry: `frames` is a lattice point, `duration` is a request
//!
//! Two request fields reach the same quantity and are deliberately treated differently:
//!
//! * **`frames`** names a point on the model's own `17n + 5` lattice. An off-lattice value is
//!   **rejected** ([`crate::pipeline::resolve_geometry`]) — SceneWorks normalizes dimensions
//!   upstream, so a gate that silently refits is a gate that can never be observed to fire, and at
//!   video scale a refit is three quarters of a second of picture the caller never asked for.
//! * **`duration`** is a continuous seconds value with no lattice of its own, so it is aligned
//!   **upward** to the next legal frame count ([`crate::pipeline::align_frames_for_duration`]).
//!
//! When both are present `frames` wins, because it is the exact one.
//!
//! # `ref2va` is here (sc-17157), and it moves the checkpoint
//!
//! A request carrying image / video / audio **references** is a third task, [`MiniMaxH3Task::Ref2va`],
//! and it denoises from a **different 66 GB partition** — [`REFERENCE_DIT_PARTITION`]. The two
//! partitions ship the same `config.json` and the same 638 tensor names and differ only in their
//! values, so [`MiniMaxH3Task`] is the single thing standing between a reference request and a
//! plausible render off the wrong checkpoint. See that type.
//!
//! `generate_ref2va` has **four** conditioning phases rather than three. The DiT is mapped **after
//! the conditioner and before the decode VAEs** — it is not the last component of the render, the
//! decode video VAE and the audio VAE follow it — and only ONE of the two 66 GB partitions is ever
//! mapped, because [`MiniMaxH3Task::partition`] picks exactly one.
//!
//! # `transformer_ref` is required **per task**, not per snapshot
//!
//! The reference partition is checked when a request resolves to [`MiniMaxH3Task::Ref2va`], at the
//! engine boundary and before any weight is read — not unconditionally in [`load`]. The reason is
//! **which artifacts an off-Mac install can actually obtain**, and it is stated below as three
//! checkable facts about the catalog rather than as a claim about what any engine permits.
//!
//! The MLX lane's `load` opens `transformer/config.json` **and** `transformer_ref/config.json` on
//! every load regardless of task. That is safe *there* because SceneWorks' `builtin.models.jsonc`
//! ships `{q4,q8,bf16}/transformer_ref/*` as per-tier `coRequisite` rows — part of the minimum
//! loadable set, not an optional Ref2VA extra. Off-Mac it would not be safe, because:
//!
//! 1. **every one of those `transformer_ref` rows is `platforms: ["macos"]`**;
//! 2. the off-Mac artifact set sc-19558 defines (a raw upstream snapshot: `text_encoder`,
//!    `transformer`, `vae`, `audio_vae`, plus the `FL2VA/audio_vae` config triple) **carries no
//!    `transformer_ref` rows at all**; and
//! 3. SceneWorks' `crates/sceneworks-worker/src/video_jobs/minimax_h3.rs` is
//!    `#[cfg(target_os = "macos")]` end to end, so there is no off-Mac dispatch arm today.
//!
//! So an off-Mac install has no catalog route to the partition, and a base-only snapshot is the
//! ordinary off-Mac shape rather than a damaged one (an interrupted download and a declined
//! co-requisite reach it too). Probing at `load` would fail **every** off-Mac load, taking `t2va`
//! and `fl2va` offline over a directory the catalog never offered.
//!
//! **This is exactly why the per-request refusal has to be loud.** `ref2va` is fully ported on this
//! lane — sc-17157 landed it, and [`descriptor`] advertises all three reference
//! [`ConditioningKind`]s, so [`Generator::validate`] genuinely admits such a request. The reference
//! arm is therefore reachable through this crate's own API on a snapshot that cannot serve it, and
//! the alternative to refusing is a `ref2va` request silently rendering off `transformer/` —
//! plausible video, wrong checkpoint. The other two tasks keep working.
//!
//! > **Two retracted premises, recorded so neither is reintroduced.**
//! >
//! > 1. This decision was once justified as "`SceneWorks/minimax-h3-mlx` publishes no
//! >    `q4/transformer_ref` (sc-19517)". False today: the sc-19573 manifest block ships
//! >    `q4/transformer_ref/*` with an exact hosted byte count (18.78 GB) at revision `137ce668`,
//! >    and the manifest flags the older reasoning as having come "from a premise the engine has
//! >    never honoured".
//! > 2. A first attempt at replacing it said the off-Mac rows are absent because "this provider
//! >    default-denies `ref2va` at its conditioning allowlist until sc-17157 lands the port". Also
//! >    false, and contradicted a few hundred lines below: sc-17157 **landed** (`32204c935`,
//! >    2026-08-15), and this provider advertises and admits `ref2va`. The manifest comment that
//! >    claim was paraphrased from predates the port and is itself stale, which is why the reasoning
//! >    above rests only on where the rows are and what dispatch exists — both directly checkable —
//! >    and asserts no mechanism. **Why the rows are still absent is a catalog question**, so
//! >    confirm it with the catalog owner before adding them; do not infer it here.
//! >
//! > Both behaviours were and remain correct; only the stated causes were wrong. See [`crate::tier`]
//! > for the full reconciliation.
//!
//! # The turbo LoRA seam IS here (sc-18728), and the declaration still gates nothing
//!
//! [`crate::adapters`] is the candle twin of `mlx_gen_minimax_h3::adapters`, installed through the
//! single [`MiniMaxH3::load_task_dit`] seam. `supports_lora` is now `true` — but it is worth keeping
//! the original finding on the record, because it is what shapes the code around it:
//!
//! **`supports_lora`, `supports_lokr` and `supported_quants` are declarations gen-core reads on no
//! path.** No validator and no registry code inspects them. They are advertisements to a
//! weights-free consumer, not gates. So flipping `supports_lora` to `true` grants nothing and
//! flipping it to `false` would forbid nothing; the behaviour is entirely in
//! [`crate::adapters::apply_minimax_h3_adapters`], which is strict about unmatched targets and
//! about a file that folded nothing. Symmetrically, `supports_lokr: false` is enforced twice in
//! real code — by kind in [`load`] and by file content in the installer — precisely because the
//! flag itself cannot refuse anything.
//!
//! The quant tiers and every unread `LoadSpec` slot are still refused by [`load`]: the explicit
//! `spec.quantize` check plus `reject_unread_slots` for the seven slots this provider does not
//! read. Without them a spec carrying any of those loaded clean and rendered with the caller's knob
//! silently discarded. See [`load`] for the full note.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    reject_unknown_components, Capabilities, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, MemoryProviderContract, Modality, ModelDescriptor, OffloadPolicy, Precision,
    Progress, Quant, SizeFloor, StepSupport, WeightsSource,
};
use candle_gen::gen_core::{AdapterSpec, ConditioningKind, Image};
use candle_gen::{CandleError, Result};

use crate::audio_config::{AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::denoise::{JointSchedule, AUDIO_SIGMA_SHIFT, MINIMAX_H3_FPS, VIDEO_SIGMA_SHIFT};
use crate::dit::adaln::AdaLnResidency;
use crate::dit::model::{JointDit, MiniMaxH3Dit};
use crate::dit::positions::KeyframeAnchor;
use crate::pipeline::{
    align_frames_for_duration, fit_audio_to_video, fl2va_layout, frames_to_images, initial_latents,
    prepend_condition_rows, render_latents, resolve_geometry, revert_pixel_normalization,
    t2va_layout, RequestGeometry, CANVAS_SHORT_EDGE, MAX_CANVAS_EDGE, PATCH_SIZE, SPATIAL_STRIDE,
};
use crate::reference::{Ref2VaReference, Ref2VaReferences, VideoReference};
use crate::text_encoder::{
    lm_prefixes, load_vision_tower, MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer,
    LM_PREFIX, TE_STORE_DTYPE,
};
use crate::vae::MiniMaxH3VideoVae;
use crate::MODEL_ID;

/// Model evaluations a request runs when it names no step count.
///
/// The reference declares no default; 50 is what the sc-17242 spike rendered at and what the model
/// card's own examples use. **Evaluations**, not `num_inference_steps` — that count includes the
/// terminal `σ = 0` the model is never evaluated at, so the schedule is built with `steps + 1`.
pub const DEFAULT_STEPS: u32 = 50;

/// Upper bound on requested steps. A provider-local guard: every step is a full 33 B forward and the
/// AdaLN cache grows with the distinct-timestep count, so an unbounded value is a resource hazard
/// rather than a slow render.
pub const MAX_STEPS: u32 = 200;

/// **The residency this provider runs at, regardless of what a caller asked for.**
///
/// See the module docs: the manifest's measured `vramGbByTier` is a sequential-phase peak, so the
/// number is only truthful if the load cannot be talked out of the staging it was measured under.
pub const OFFLOAD_POLICY: OffloadPolicy = OffloadPolicy::Sequential;

/// The DiT partition a `t2va` / `fl2va` render denoises from.
pub const BASE_DIT_PARTITION: &str = "transformer";

/// The DiT partition a `ref2va` render denoises from.
pub const REFERENCE_DIT_PARTITION: &str = "transformer_ref";

/// The component directories **every** task reads that are **tier-agnostic**, so they always resolve
/// against the snapshot root.
///
/// `text_encoder` and [`BASE_DIT_PARTITION`] used to be here too. They are not any more (sc-20267):
/// both are tiered and both may be staged *outside* the root as component overrides, so requiring
/// them against the root would fail every split-tier install. They are required against the dirs
/// [`crate::tier::MiniMaxH3TierPaths`] resolved instead — see [`MiniMaxH3::load`]. The check itself is
/// unchanged; only the directory it runs on is.
///
/// [`REFERENCE_DIT_PARTITION`] is deliberately **not** here — see [`task_component_dirs`].
const TIER_AGNOSTIC_COMPONENT_DIRS: [&str; 2] = ["vae", "audio_vae"];

/// The `LoadSpec::components` keys this provider **recognizes** as staging overrides (sc-20267).
///
/// The two tiered components, and only those: the manifest ships per-tier `transformer` and
/// `text_encoder` subtrees, and staging them is how a caller selects a tier. Everything else is still
/// refused by `reject_unknown_components` — a caller who stages a `vae` believing it will be read gets
/// a typed error rather than a silently ignored directory.
///
/// **Recognized is not required.** [`descriptor`]'s `required_components` stays empty: an unstaged
/// component resolves to the flat `root/<name>` layout, which is the ordinary upstream-snapshot shape.
///
/// [`REFERENCE_DIT_PARTITION`] is deliberately absent — it is never staged directly, it is derived as
/// the resolved DiT's sibling. Accepting it as a key would let a caller stage a reference partition
/// from a *different* tier than the base one.
const KNOWN_COMPONENTS: &[&str] = &[
    crate::tier::DIT_COMPONENT,
    crate::tier::TEXT_ENCODER_COMPONENT,
];

/// The component directories a given task reads, on top of [`TIER_AGNOSTIC_COMPONENT_DIRS`] and the
/// two tiered components.
///
/// # Why the reference partition is task-gated rather than load-gated
///
/// `ref2va` is a first-class task, so a snapshot missing `transformer_ref/` must fail **before any
/// weight is read**, naming the path, rather than twenty minutes into a render when the reference
/// arm finally reaches for it. The obvious implementation — put it in
/// [`TIER_AGNOSTIC_COMPONENT_DIRS`] — is wrong, and its blast radius is larger than it looks: the
/// `transformer_ref` rows SceneWorks' manifest ships are **all `platforms: ["macos"]`**, and the
/// off-Mac artifact set carries none, so an off-Mac install has no catalog route to the partition and
/// a base-only snapshot is the *ordinary* off-Mac shape rather than a broken one. Requiring the
/// reference partition at `load` would fail provider construction outright and take `t2va` and
/// `fl2va` offline with it, on every platform this lane serves, for a snapshot that serves both
/// perfectly well.
///
/// (Two retracted premises are recorded in the module docs — the sc-19517 "not published" claim, and
/// a later "default-denies `ref2va` until sc-17157" claim that is equally false since sc-17157
/// landed. The conclusion rests on the manifest rows and the absent off-Mac dispatch arm, both
/// checkable, and asserts no engine mechanism. See also [`crate::tier`].)
///
/// So the check runs where the task is known: [`Generator::validate`], which
/// [`MiniMaxH3::generate_impl`] calls first, and which runs before the geometry is resolved and long
/// before the text encoder is mapped. Fail-loud is preserved for the request that actually needs the
/// partition; the requests that do not are unaffected.
fn task_component_dirs(task: MiniMaxH3Task) -> &'static [&'static str] {
    match task {
        MiniMaxH3Task::T2va | MiniMaxH3Task::Fl2va => &[],
        MiniMaxH3Task::Ref2va => &[REFERENCE_DIT_PARTITION],
    }
}

/// The caller's staged override for `component`, if it is a directory.
///
/// A single *file* is not a component directory, so it falls back to `None` and lets the resolver
/// produce the actionable "missing `<component>/config.json`" error against the root —
/// `mlx_gen_minimax_h3::model`'s `resolve_dit_dir` makes the identical choice for the identical
/// reason.
fn staged_component_dir<'a>(spec: &'a LoadSpec, component: &str) -> Option<&'a Path> {
    match spec.components.get(component) {
        Some(WeightsSource::Dir(p)) => Some(p.as_path()),
        _ => None,
    }
}

/// Map `spec.quantize` onto the tier this load will **assert**.
///
/// # `None` must NOT become [`crate::tier::Tier::Bf16`]
///
/// `spec.quantize` is an `Option<Quant>` whose `None` means *"the caller asserted nothing"*, and that
/// is a genuine third state. MLX honours it: `mlx_gen_minimax_h3::model::reconcile_tier` admits a
/// packed tier under `(Some(_), None)` and only refuses a *dense* component under an explicit
/// request. [`crate::tier::Tier::Bf16`] is not that state — it is a positive assertion of denseness,
/// and `require_dit` refuses a packed tier under it. So `None` maps to `None` here and the reconcile
/// is skipped entirely; the loaders auto-detect the tier from `{base}.scales` regardless, so nothing
/// downstream needs to be told. Defaulting `None` to `Bf16` would turn "I did not ask" into "I demand
/// dense" and reject every packed install that loads fine on MLX today — the trap PR 2 of sc-20267
/// left a warning about on the [`crate::tier::Tier`] enum.
///
/// # `Nvfp4` is refused rather than mapped
///
/// [`Quant::Nvfp4`] reports `bits() == 4`, so mapping it to [`crate::tier::Tier::Q4`] would let an
/// NVFP4 request reconcile *cleanly* against a `q4` marker and render the int4-affine tier while the
/// caller believed they had selected NVFP4. That is the silent-numerics-swap epic 11037's SC#5
/// forbids, so it is a typed refusal at the registry boundary instead. This model publishes no NVFP4
/// tier — `mlx_gen_minimax_h3::convert` packs MLX affine `q4`/`q8` only — so there is nothing to
/// route it to.
fn requested_tier(spec: &LoadSpec) -> candle_gen::gen_core::Result<Option<crate::tier::Tier>> {
    match spec.quantize {
        None => Ok(None),
        Some(Quant::Q4) => Ok(Some(crate::tier::Tier::Q4)),
        Some(Quant::Q8) => Ok(Some(crate::tier::Tier::Q8)),
        Some(q @ Quant::Nvfp4) => Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: spec.quantize={q:?} is not a published MiniMax-H3 tier. This family ships \
             MLX affine `q4` and `q8` only (packed offline by the MLX lane's `convert`); NVFP4 is a \
             distinct tier with different numerics (E2M1 elements over FP8 block scales), and \
             because it also reports 4 bits it would otherwise reconcile silently against a `q4` \
             marker and render int4-affine under an NVFP4 request. Request `Q4` or `Q8`."
        ))),
    }
}

/// A component directory is present **and carries at least one `.safetensors` shard**.
///
/// `is_dir()` alone was the gate here, and it admitted an empty or shard-less `transformer_ref/` —
/// which is exactly the mid-render failure the load-time check exists to eliminate. A snapshot whose
/// download was interrupted leaves precisely that shape behind.
fn require_component(root: &Path, component: &str) -> Result<()> {
    require_component_shards(&root.join(component), component)
}

/// [`require_component`] against an **already-resolved** directory.
///
/// The tiered components are staged outside the snapshot root, so the check cannot be expressed as
/// `root.join(component)` for them. Identical check, identical messages — only the path arithmetic
/// moves to the caller.
fn require_component_shards(dir: &Path, component: &str) -> Result<()> {
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: the snapshot has no `{component}/` component at {}",
            dir.display()
        )));
    }
    let shards = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("{MODEL_ID}: read {}: {e}", dir.display())))?
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
        })
        .count();
    if shards == 0 {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: the `{component}/` component at {} carries no `.safetensors` shard — an \
             empty component directory loads clean and fails in the middle of the render",
            dir.display()
        )));
    }
    Ok(())
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
/// and `tests/ref2va_checkpoint.rs` pins the selection — structurally without weights, and against
/// the real bytes of a tensor that differs between the two under `#[ignore]`.
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

impl MiniMaxH3Task {
    /// The snapshot subdirectory this task's DiT is read from.
    pub fn partition(self) -> &'static str {
        match self {
            Self::T2va | Self::Fl2va => BASE_DIT_PARTITION,
            Self::Ref2va => REFERENCE_DIT_PARTITION,
        }
    }

    /// Whether this task reads the vision tower as part of its presentation.
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
            (true, true) => Err(CandleError::Msg(format!(
                "{MODEL_ID}: a request carries both keyframes and references, which are different \
                 tasks on different checkpoints — `fl2va` pins a literal frame of the generated \
                 clip from `transformer/`, `ref2va` conditions on unpositioned references from \
                 `transformer_ref/`. Send one or the other."
            ))),
        }
    }
}

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "minimax_h3",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch exists anywhere in the checkpoint.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // **One kind covers all three keyframe shapes.** `Keyframe` carries a `frame_idx`, and
            // first-only / last-only / first+last are one, one and two of them.
            //
            // The three `ref2va` kinds (sc-17157) are the omni-reference surface on
            // `transformer_ref`. They are three kinds rather than one because gen-core has no
            // heterogeneous-reference variant and `conditioning` is an **ordered** `Vec`, so the
            // request's own order carries the semantics `ref2va` needs — see
            // [`crate::reference`] on why order is not incidental, and [`request_references`] for
            // why the video one is `ReferenceVideo` and not `VideoClip` or `VideoSync`.
            //
            // `VideoClip` is deliberately **not** advertised. Nothing here reads it, and leaving it
            // in the allowlist would let an in-context clip through to a model that has no
            // in-context clip mechanism; default-deny turns it into the typed
            // `Error::Unsupported` instead.
            conditioning: vec![
                ConditioningKind::Keyframe,
                ConditioningKind::Reference,
                ConditioningKind::ReferenceVideo,
                ConditioningKind::ReferenceAudio,
            ],
            // **The turbo-LoRA seam is ported** (sc-18728): `crate::adapters` is the candle twin
            // of `mlx_gen_minimax_h3::adapters`, installed on the job-local DiT through
            // `load_task_dit` — which every task, `ref2va` included, goes through. This bit is a
            // declaration gen-core reads on no path, so it is an advertisement to a weights-free
            // consumer; the enforcement is `crate::adapters::apply_minimax_h3_adapters`, which is
            // strict about both unmatched targets and a file that matched nothing.
            supports_lora: true,
            // **LoKr stays false, and here that is enforced twice.** This DiT's adapter seam
            // installs `scale·((x·A)·B)` residuals only; a Kronecker delta is a different
            // operation, so `load` refuses `AdapterKind::Lokr` and the installer refuses a file
            // carrying `lokr_*` factors regardless of how it was declared.
            supports_lokr: false,
            // **The tier loader is ported** (sc-20267): the pre-quantized tiers are packed offline
            // by the MLX lane's `convert`, and this lane now *reads* them. `crate::tier` resolves
            // the per-tier component directories and reconciles the request against the on-disk
            // `quantization` marker; `crate::quant` builds each packed Linear straight from the MLX
            // affine triple, so a `q4` DiT loads at its own footprint with no dense bf16 transient.
            //
            // **`Nvfp4` is deliberately absent, and that is not an oversight.** It is a distinct
            // creative-choice tier (epic 11037, sc-11042 Option A) with different numerics — E2M1
            // elements over FP8 block scales, not an int4 affine pack — and `Quant::Nvfp4.bits()`
            // reports `4`, so admitting it would let an NVFP4 request reconcile cleanly against a
            // `q4` marker and render the wrong tier silently. `load` refuses it by name.
            //
            // Like `supports_lora`, this is a declaration gen-core reads on no path; the enforcement
            // is `crate::tier`'s reconcile plus the `Nvfp4` refusal in [`load`].
            supported_quants: &[Quant::Q4, Quant::Q8],
            // **The discovery signal, and it is true here.** gen-core makes
            // `OffloadPolicy::Sequential` advisory, so an unwired engine silently stays resident and
            // the fallback is invisible from outside; this bit is what a consumer reads to know
            // which it got. `candle-gen-krea` states the rule as a lockstep contract — "never flip
            // this on ahead of the wiring", because advertising a lane that would really run
            // resident makes the fit-gate under-predict its peak and admit a job that then OOMs.
            //
            // The wiring is here and is *stronger* than honoring: `generate_impl` releases every
            // heavy component and synchronizes the device before mapping the next, and
            // `OFFLOAD_POLICY` forces `Sequential` so a caller cannot even opt out. Leaving this
            // `false` was the divergence worth closing — SceneWorks' manifest already declares
            // `supportsSequentialOffload: true`, and a `false` bit here would tell the worker to
            // size this render at the ~143 GB SUM of the components rather than their max.
            //
            // Note this is a *separate* question from the memory-ladder rung-1 declaration in
            // `crate::memory_strategy`, which stays `Missing` pending its behavior seam (sc-18660).
            // That declaration gates ladder ADMISSION; this bit describes the residency lifecycle.
            supports_sequential_offload: true,
            max_count: 1,
            // The provider's step bound, advertised rather than hidden (sc-19559) — the candle
            // twin of `mlx-gen-minimax-h3`'s declaration, from this lane's own `MAX_STEPS`.
            supported_steps: StepSupport::Range {
                min: 1,
                max: MAX_STEPS,
            },
            component_precision_floors: &[],
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: SPATIAL_STRIDE,
            // The widest edge `crate::keyframe::resolve_canvas_size` can put a picture on, at the
            // 4:1 aspect ceiling. A per-edge cap is NOT the area budget — `CANVAS_MAX_PIXELS` is
            // checked as a product by `resolve_geometry` and still refuses 1536x1536 / 1344x1344,
            // which sit inside this ceiling on both edges. See `MAX_CANVAS_EDGE` (sc-17152).
            max_size: MAX_CANVAS_EDGE,
            // The stride is advertised, not hidden in provider code, so a weights-free consumer can
            // predict that an unaligned canvas is REFUSED rather than quietly refit.
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SPATIAL_STRIDE,
            },
            ..Default::default()
        },
    }
}

/// The loaded `minimax_h3` provider — **paths only**.
///
/// Nothing heavy is resident between renders, and nothing heavy is resident between *phases* of one
/// render either. That is the whole memory design, so the struct that carries it holds no tensors to
/// make the property structural rather than a habit.
pub struct MiniMaxH3 {
    descriptor: ModelDescriptor,
    root: PathBuf,
    /// The per-tier component directories this load reads (sc-20267) — resolved once from the
    /// caller's staged component overrides, so every later render reads the same tier.
    ///
    /// Held **beside** `root` rather than replacing it: the VAEs, tokenizer and `FL2VA/` are
    /// tier-agnostic and still resolve against the root, which is exactly what
    /// [`crate::tier::MiniMaxH3TierPaths::shared_root`] records.
    tiers: crate::tier::MiniMaxH3TierPaths,
    /// The tier the caller **asserted**, or `None` when `spec.quantize` was absent.
    ///
    /// `None` is a third state, not a synonym for [`crate::tier::Tier::Bf16`]: it means the caller
    /// asserted nothing, so the reconcile is skipped and the loaders auto-detect from `.scales`.
    /// Mapping it onto `Bf16` would turn "I did not ask" into "I demand dense" and reject every
    /// packed install that loads fine on MLX today — see [`requested_tier`].
    tier: Option<crate::tier::Tier>,
    device: Device,
    dtype: DType,
    offload: OffloadPolicy,
    contract: MemoryProviderContract,
    /// The staged adapter specs, held as **paths and strengths** rather than tensors — the same
    /// discipline the rest of this struct keeps. They are read once per render, in
    /// [`MiniMaxH3::load_task_dit`], onto the job-local DiT.
    adapters: Vec<AdapterSpec>,
}

impl MiniMaxH3 {
    /// Validate the snapshot and record the paths a render will read.
    ///
    /// # The tier is resolved here, once (sc-20267)
    ///
    /// The DiT and text encoder come from the caller's staged component overrides when present, so
    /// they are **not** necessarily under `root`. The tier-agnostic components — the two VAEs, the
    /// tokenizer, `FL2VA/` — always are. Resolving once and holding
    /// [`crate::tier::MiniMaxH3TierPaths`] is what keeps a later render from re-deriving a different
    /// answer, which is how the DiT and the vision tower beside it could end up on different tiers.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        // **The two tiered components are now RECOGNIZED staging keys** (sc-20267). The allowlist was
        // `&[]`, which refused every component override — including the two the manifest's per-tier
        // subtrees exist to be staged as, so a tier could not have been handed to this provider at all.
        // They are recognized, not *required*: `descriptor().required_components` stays empty, because
        // an unstaged component falls back to the flat `root/<name>` layout rather than failing.
        reject_unknown_components(spec, KNOWN_COMPONENTS, MODEL_ID)?;
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(p) => p
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| p.clone()),
        };
        // The one mapping, shared with the registry entry point. `requested_tier` reports its refusal
        // as the contract-load-bearing `gen_core::Error::Unsupported`, which `CandleError` has no
        // variant for — so on this path (which returns `CandleError`) the type is flattened to `Msg`.
        // That is not a weaker check, only a weaker error *type*: [`load`] runs the same mapping first
        // and is where a registry caller gets the typed refusal.
        let tier = requested_tier(spec).map_err(|e| CandleError::Msg(e.to_string()))?;
        let tiers = crate::tier::MiniMaxH3TierPaths::resolve(
            &root,
            staged_component_dir(spec, crate::tier::DIT_COMPONENT),
            staged_component_dir(spec, crate::tier::TEXT_ENCODER_COMPONENT),
        );
        // The tier-agnostic components, against the root — unchanged.
        for component in TIER_AGNOSTIC_COMPONENT_DIRS {
            require_component(&root, component)?;
        }
        // The two tiered components, against the dirs that were actually resolved. `require_dit`
        // additionally reconciles the on-disk `quantization` marker against an asserted tier and
        // validates the declared group size; the shard probe is this crate's own, and catches the
        // interrupted-download shape a `config.json`-only check would pass.
        require_component_shards(&tiers.dit_dir, BASE_DIT_PARTITION)?;
        require_component_shards(&tiers.text_encoder_dir, crate::tier::TEXT_ENCODER_COMPONENT)?;
        match tier {
            Some(t) => tiers.require_dit(t)?,
            // Nothing was asserted, so there is no tier to reconcile against — but the group size
            // is still validated, because a component packed at a group this engine does not read
            // derives a legal-looking, wrong bit width rather than failing cleanly (sc-15154).
            None => tiers.require_dit_unasserted()?,
        }
        tiers.require_text_encoder()?;
        let device = candle_gen::default_device()?;
        Ok(Self {
            descriptor: descriptor(),
            root,
            tiers,
            tier,
            device,
            // The DiT ships mixed f32/bf16 top-level tensors; bf16 is the block store, matching the
            // published checkpoint rather than widening it.
            dtype: DType::BF16,
            // **Not `spec.offload_policy`.** See [`OFFLOAD_POLICY`] and the module docs.
            offload: OFFLOAD_POLICY,
            contract: crate::memory_strategy::contract_for(spec)?,
            adapters: spec.adapters.clone(),
        })
    }

    /// The residency this provider will actually run at.
    ///
    /// Reported rather than echoed: a caller reading this back gets the enforced policy, so
    /// "requested Resident, got Sequential" is observable instead of silent.
    pub fn offload_policy(&self) -> OffloadPolicy {
        self.offload
    }

    /// The snapshot root a render reads from.
    ///
    /// The **tier-agnostic** components resolve against this: the two VAEs, the tokenizer and
    /// `FL2VA/`. The DiT and text encoder do not — see [`Self::tier_paths`].
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The per-tier component directories this load resolved (sc-20267).
    ///
    /// `pub` because the property "the DiT and the vision tower beside it came from the *same* staged
    /// tier" is not assertable from outside the crate otherwise, and a source scan that the call sites
    /// *look* right is not evidence — `tests/tier_resolution.rs` reads these back off a loaded
    /// provider.
    pub fn tier_paths(&self) -> &crate::tier::MiniMaxH3TierPaths {
        &self.tiers
    }

    /// The tier the caller asserted through `spec.quantize`, or `None` when they asserted nothing.
    ///
    /// `None` is **not** [`crate::tier::Tier::Bf16`] — see `requested_tier`.
    pub fn requested_tier(&self) -> Option<crate::tier::Tier> {
        self.tier
    }

    /// **The single place a request becomes a checkpoint choice.**
    ///
    /// Returns the task *and* the reference list together because they are one decision: the list's
    /// presence is what makes the task `ref2va`, and re-deriving either half separately is how the
    /// two would drift. [`Generator::validate`] and `Self::generate_impl` (private, hence not
    /// linked) both consume exactly this value, and `tests/ref2va_checkpoint.rs` asserts that
    /// `MiniMaxH3Task::resolve` is called from nowhere else in the crate — so a weights-free test
    /// that drives this function is testing the render path's own decision rather than a parallel
    /// restatement of it.
    pub fn resolve_task(
        &self,
        req: &GenerationRequest,
    ) -> Result<(MiniMaxH3Task, Option<Ref2VaReferences>)> {
        let references = request_references(req)?;
        let task = MiniMaxH3Task::resolve(!req.keyframes().is_empty(), references.is_some())?;
        for component in task_component_dirs(task) {
            debug_assert_eq!(
                *component, REFERENCE_DIT_PARTITION,
                "the only task-gated component is the reference partition; a new one needs its own \
                 resolution rule rather than the sibling rule below"
            );
            // The reference partition is the resolved DiT dir's **SIBLING**, not `root/transformer_ref`
            // (sc-20267): a split tier install stages `transformer` outside the snapshot and has no
            // `root/transformer_ref` at all, so resolving against the root would report a missing
            // directory while the correct one sat next to the staged DiT.
            let dir = &self.tiers.reference_dit_dir;
            require_component_shards(dir, component).map_err(|e| {
                CandleError::Msg(format!(
                    "{e} — this is the {task:?} task's own checkpoint, and it is not part of the \
                     off-Mac artifact set: every `{REFERENCE_DIT_PARTITION}` row in the model \
                     catalog is macOS-only, so this snapshot has no route to it. The other tasks are \
                     unaffected."
                ))
            })?;
            // ...and it must be the SAME tier as the base partition. `crate::convert` packs
            // `transformer/` and `transformer_ref/` alike, so a mismatch here is a broken install
            // rather than a legal mixed one.
            match self.tier {
                Some(t) => self.tiers.require_reference_dit(t)?,
                None => self.tiers.require_reference_dit_unasserted()?,
            }
        }
        Ok((task, references))
    }

    /// The device every phase materializes on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The adapter specs this provider was loaded with.
    pub fn adapters(&self) -> &[AdapterSpec] {
        &self.adapters
    }

    /// The **resolved directory** for one DiT partition (sc-20267).
    ///
    /// `transformer` is the staged tier dir (else `root/transformer`); `transformer_ref` is that
    /// dir's SIBLING, never `root/transformer_ref` — see [`crate::tier`] on why the sibling rule
    /// serves both the flat and the split-tier layout. Any other name is a programming error rather
    /// than a bad install: [`MiniMaxH3Task::partition`] is the only producer of this argument and it
    /// returns one of the two constants.
    fn partition_dir(&self, partition: &str) -> Result<&Path> {
        if partition == BASE_DIT_PARTITION {
            Ok(&self.tiers.dit_dir)
        } else if partition == REFERENCE_DIT_PARTITION {
            Ok(&self.tiers.reference_dit_dir)
        } else {
            Err(CandleError::Msg(format!(
                "{MODEL_ID}: `{partition}` is not a DiT partition of this model — the two are \
                 `{BASE_DIT_PARTITION}` and `{REFERENCE_DIT_PARTITION}`"
            )))
        }
    }

    /// **The single render seam that maps a DiT partition and folds the staged adapters onto it**
    /// (sc-18728).
    ///
    /// One function rather than an install at each call site, so a future `ref2va` denoise path
    /// (sc-17157) cannot acquire a DiT that skipped the adapters — the MLX sibling landed the same
    /// single seam for exactly that reason. The DiT is **job-local**: residuals are pushed onto this
    /// render's own copy, never onto a shared resident base, so two concurrent renders at different
    /// strengths cannot see each other's fold.
    ///
    /// The install is strict — an unmatched target, or a file that folded nothing, is an error
    /// rather than a partial render — and it runs **before** `JointDit` precomputes the AdaLN
    /// tables. That ordering is required rather than incidental: `adaln_proj` is deliberately
    /// unreachable to adapters ([`crate::dit::block::DitBlock::adaptable_mut`]), so the precompute
    /// must observe the same projection whether or not a LoRA is staged.
    /// **`pub` on purpose.** The property "a staged adapter reaches the DiT this render denoises
    /// with" cannot be asserted from outside the crate unless this seam is callable, and a source
    /// scan asserting the call site *looks* right is not evidence that it folds anything —
    /// `tests/turbo_lora.rs::the_render_seam_folds_the_staged_adapter_onto_the_dit` calls this and
    /// reads the residual back off the returned model.
    ///
    /// The partition is mapped to its **resolved tier directory** through `Self::partition_dir`,
    /// so a staged tier's DiT is denoised from rather than the root's.
    pub fn load_task_dit(&self, partition: &str) -> Result<MiniMaxH3Dit> {
        let mut dit =
            MiniMaxH3Dit::load_from_dir(self.partition_dir(partition)?, &self.device, self.dtype)?;
        if !self.adapters.is_empty() {
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.adapters)?;
        }
        Ok(dit)
    }

    /// Resolve the request's geometry: `frames` wins over `duration`, and the canvas falls back to
    /// the checkpoint's own 16:9 default when the caller names none.
    fn request_geometry(&self, req: &GenerationRequest) -> Result<RequestGeometry> {
        let frames = match (req.frames, req.duration) {
            (Some(f), _) => f as usize,
            (None, Some(seconds)) => align_frames_for_duration(seconds)?,
            (None, None) => align_frames_for_duration(MIN_DEFAULT_SECONDS)?,
        };
        let (width, height) = if req.width == 0 || req.height == 0 {
            (
                CANVAS_SHORT_EDGE * 16 / 9 / SPATIAL_STRIDE * SPATIAL_STRIDE,
                CANVAS_SHORT_EDGE,
            )
        } else {
            (req.width, req.height)
        };
        resolve_geometry(width, height, frames)
    }

    /// **Phase 1 (t2va): encode the prompt and release the 66.7 GB conditioner.**
    ///
    /// The prefix trim is what keeps this to the 50 layers the tap runs: `lm_prefixes` names
    /// `embed_tokens` and layers `0..50`, so the header-only mmap never materializes layers 50-63,
    /// the final norm or `lm_head`.
    fn encode_prompt(&self, prompt: &str) -> Result<candle_gen::candle_core::Tensor> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) = tok.encode_prompt(prompt, &self.device)?;
        let cfg = MiniMaxH3TeConfig::from_component_dir(&self.tiers.text_encoder_dir)?;
        let shards =
            candle_gen::loader::sorted_safetensors(&self.tiers.text_encoder_dir, "minimax-h3 te")?;
        let prefixes = lm_prefixes(LM_PREFIX, &cfg);
        let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
        let w =
            candle_gen::Weights::from_files_filtered(&shards, &self.device, TE_STORE_DTYPE, &refs)?;
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_STORE_DTYPE)?;
        drop(w);
        let context = te.forward(&ids, &mask)?;
        // candle is eager, so `context` is already materialized here and dropping the encoder
        // genuinely frees its weights — there is no lazy graph holding them alive.
        drop(te);
        crate::dit::release_device_memory(&self.device)?;
        Ok(context)
    }

    /// **Phase 1 (fl2va): the vision-grounded presentation.**
    ///
    /// Runs the Qwen3-VL vision tower over the fitted keyframes, splices their embeddings into the
    /// `"<Picture i>: "` presentation, and returns the context together with its **per-row modality
    /// tags** — a vision block's rows are tagged *video* and address a different block of the AdaLN
    /// modulation table than the text rows around them.
    ///
    /// This is one of the two paths a keyframe takes. The other is the VAE encode in
    /// [`crate::conditioning`]; the reference runs **both**, and neither substitutes for the other —
    /// the tower supplies semantic context to the prompt stream, the VAE supplies the pixel-space
    /// anchor the video rows are conditioned on.
    fn encode_prompt_grounded(
        &self,
        prompt: &str,
        keyframes: &[&Image],
    ) -> Result<(candle_gen::candle_core::Tensor, Vec<u32>)> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let cfg = MiniMaxH3TeConfig::from_component_dir(&self.tiers.text_encoder_dir)?;

        let vision = load_vision_tower(&self.tiers.text_encoder_dir, &self.device)?;
        let grounded = crate::text_encoder::run_vision(&vision, keyframes, &self.device)?;
        drop(vision);
        crate::dit::release_device_memory(&self.device)?;

        let (ids, mask, tags) = tok.encode_fl2va(prompt, &grounded.counts, &self.device)?;
        let shards =
            candle_gen::loader::sorted_safetensors(&self.tiers.text_encoder_dir, "minimax-h3 te")?;
        let prefixes = lm_prefixes(LM_PREFIX, &cfg);
        let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
        let w =
            candle_gen::Weights::from_files_filtered(&shards, &self.device, TE_STORE_DTYPE, &refs)?;
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_STORE_DTYPE)?;
        drop(w);
        let context = te.forward_with_images(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        drop(te);
        drop(grounded);
        crate::dit::release_device_memory(&self.device)?;

        // The tags describe the presentation row for row; a mismatch would mis-tag every row after
        // the divergence and is silent, because both lengths build a runnable sequence.
        if context.dims()[1] != tags.len() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: the grounded context has {} rows but {} modality tags",
                context.dims()[1],
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
        Generator::validate(self, req)?;
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let geometry = self.request_geometry(req)?;
        let evaluations = req.steps.unwrap_or(DEFAULT_STEPS);
        if evaluations > MAX_STEPS {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: {evaluations} steps exceeds the provider bound of {MAX_STEPS}"
            )));
        }
        // `num_inference_steps` counts the terminal σ = 0, at which the model is never evaluated.
        //
        // **The video shift is a per-request knob.** The base checkpoint's published 12.0 is the
        // default, but the distilled turbo variants are trained against their own shift and are
        // simply wrong on 12.0. Only the **video** shift is overridable, because only the video
        // shift moves across the published set: every documented variant keeps audio at 3.0.
        let video_shift = req.scheduler_shift.unwrap_or(VIDEO_SIGMA_SHIFT);
        let schedule =
            JointSchedule::with_shifts(evaluations as usize + 1, video_shift, AUDIO_SIGMA_SHIFT)?;
        let seed = req.seed.unwrap_or(0);

        // --- 1. conditioning -----------------------------------------------------------------
        let keyframes = req.keyframes();
        // **The checkpoint decision.** Resolved once, here, and carried to BOTH
        // `MiniMaxH3Dit::load` call sites — the base arm below and the `ref2va` arm in
        // `generate_ref2va` — each of which takes its partition from `task.partition()` and from
        // nothing else. The two partitions are byte-different and structurally identical, so this
        // is the only thing standing between a `ref2va` request and a plausible render off the
        // wrong 66 GB. See [`MiniMaxH3Task`] and `tests/ref2va_checkpoint.rs`.
        let (task, references) = self.resolve_task(req)?;
        if let Some(refs) = &references {
            return self.generate_ref2va(req, refs, task, &geometry, &schedule, seed, on_progress);
        }
        let anchors = keyframe_anchors(&keyframes, geometry.joint.num_frames)?;
        // Fitted ONCE, here, and shared by both keyframe paths. The vision tower and the VAE encode
        // must see the same pixels — resizing separately per path would let a resampling difference
        // put the two conditioning signals fractionally out of register, which is exactly the class
        // of divergence nothing downstream can detect.
        let fitted = if anchors.is_empty() {
            Vec::new()
        } else {
            let images: Vec<&Image> = keyframes.iter().map(|k| k.image).collect();
            crate::keyframe::fit_keyframes(&images, geometry.width, geometry.height)?
        };

        let (context, text_tags) = if anchors.is_empty() {
            let context = self.encode_prompt(&req.prompt)?;
            let tags = vec![crate::denoise::TEXT_TAG; context.dims()[1]];
            (context, tags)
        } else {
            let refs: Vec<&Image> = fitted.iter().collect();
            self.encode_prompt_grounded(&req.prompt, &refs)?
        };
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 1b. keyframe conditioning latents (fl2va) ----------------------------------------
        // The VAE encode, the second of the keyframe's two paths. Done before the DiT is mapped so
        // the ~10 GB VAE and the ~66 GB DiT are never both resident.
        let condition_rows = if anchors.is_empty() {
            None
        } else {
            let pixels = fitted
                .iter()
                .map(|f| crate::keyframe::keyframe_to_vae_pixels(f, &self.device))
                .collect::<Result<Vec<_>>>()?;
            let vae = MiniMaxH3VideoVae::load(&self.root, &self.device, self.dtype)?;
            let rows = crate::conditioning::build_condition_rows(
                &vae,
                &pixels,
                &anchors,
                PATCH_SIZE,
                &crate::conditioning::KeyframeNoise::Seeded,
                &self.device,
            )?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            rows
        };
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 2. denoise -----------------------------------------------------------------------
        // Through the adapter seam, at the partition THIS task selected — so `ref2va` (sc-17157)
        // gets the staged LoRA folded onto `transformer_ref` exactly as `t2va` / `fl2va` get it on
        // `transformer`. A bare `MiniMaxH3Dit::load` here would render a reference job with the
        // adapter silently dropped.
        let dit = self.load_task_dit(task.partition())?;
        let patch = dit.config().patch_size;
        let layout = if anchors.is_empty() {
            t2va_layout(&geometry, context.dims()[1], patch, &self.device)?
        } else {
            fl2va_layout(&geometry, &text_tags, &anchors, patch, &self.device)?
        };
        let (video_rows, audio_rows) = initial_latents(&geometry, patch, seed, &self.device)?;
        // The anchors LEAD the video row stream; the scheduler then writes only the tail.
        let video_rows = prepend_condition_rows(&layout, condition_rows.as_ref(), &video_rows)?;
        let adaln = crate::denoise::adaln_schedule(&schedule)?;
        // The 26.02 GB lever: project the whole schedule's modulation and release `adaln_proj`.
        // Every timestep this run evaluates at is enumerated in `adaln`, so the eviction is safe.
        let mut model = JointDit::new(
            dit,
            layout.clone(),
            &context,
            adaln,
            AdaLnResidency::PrecomputeAndEvict,
        )?;
        drop(context);

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
            &self.device,
            &req.cancel,
            &mut on_step,
        )?;
        drop(model);
        drop(video_rows);
        drop(audio_rows);
        crate::dit::release_device_memory(&self.device)?;

        // --- 3. decode ------------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let frames = {
            let vae = MiniMaxH3VideoVae::load_decode_only(&self.root, &self.device, self.dtype)?;
            let video = vae.decode(&rendered.video)?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            frames_to_images(&revert_pixel_normalization(&video)?)?
        };
        let audio = {
            let (cfg, w) = self.load_audio_vae()?;
            let vae = crate::audio_vae::MiniMaxH3AudioVae::from_weights(
                &w,
                &cfg,
                &self.device,
                self.dtype,
            )?;
            drop(w);
            let track = vae.decode_audio_track(&rendered.audio)?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            track
        };
        let audio = finish_audio(audio, &geometry)?;
        if frames.len() != geometry.joint.num_frames {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
    }

    /// The **`ref2va` render** — ordered multi-modal references on the `transformer_ref`
    /// checkpoint (sc-17157).
    ///
    /// The phase order is the same staged one `t2va` / `fl2va` use, for the same memory reason, but
    /// it has **four** conditioning phases rather than three: the vision tower, the 66.7 GB
    /// conditioner, then the two VAEs, then the 66.3 GB `transformer_ref`. Each is released before
    /// the next is mapped, so the DiT is never resident alongside the conditioner. It is **not**
    /// the last thing mapped — the decode video VAE and the audio VAE follow it — but it is the
    /// last *66 GB* thing, and only ONE of the two 66 GB partitions is ever mapped by a render,
    /// because [`MiniMaxH3Task::partition`] picks exactly one.
    #[allow(clippy::too_many_arguments)]
    fn generate_ref2va(
        &self,
        req: &GenerationRequest,
        references: &Ref2VaReferences,
        task: MiniMaxH3Task,
        geometry: &RequestGeometry,
        schedule: &JointSchedule,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        use crate::conditioning::{
            encode_reference_condition, keyframe_condition_rows, reference_audio_rows,
            reference_clip_to_vae_pixels, KeyframeNoise,
        };
        use crate::dit::positions::ReferenceLatentGeometry;
        use crate::pipeline::{
            audio_track_to_encoder_input, prepend_condition_audio_rows, ref2va_layout,
            CANVAS_MAX_PIXELS,
        };

        // **The task is the caller's resolved one, and it must be the reference task.** The
        // parameter carries `resolve_task`'s decision rather than re-deriving it, but a `&self`
        // method taking a `MiniMaxH3Task` will accept any of the three, and passing `T2va` here
        // would denoise a reference request off `transformer/` — the exact failure this whole story
        // exists to prevent. A hard refusal, not a `debug_assert`: release builds strip those.
        if task != MiniMaxH3Task::Ref2va {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: generate_ref2va was handed {task:?}, which denoises from `{}` — a \
                 reference render must be `Ref2va` on `{REFERENCE_DIT_PARTITION}`",
                task.partition()
            )));
        }

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
                        geometry.joint.num_frames,
                        SPATIAL_STRIDE as i32,
                        CANVAS_SHORT_EDGE as i32,
                        i64::from(CANVAS_MAX_PIXELS),
                        MINIMAX_H3_FPS,
                    )?,
                    fps: MINIMAX_H3_FPS,
                    audio: v.audio.clone(),
                }),
                Ref2VaReference::Audio(a) => Ref2VaReference::Audio(a.clone()),
            });
        }
        // **Normalization must not lose a soundtrack.** A clip's own audio rides through
        // `normalize_reference_clip` by clone, and a reference that arrived without it conditions
        // the render on fewer audio rows *and* emits one fewer `<Audio j>` label — a shorter,
        // perfectly runnable sequence. `Ref2VaReferences::audio_label_count` is the boundary's own
        // count of audio-bearing references, so this is the two counts checked against each other
        // rather than each side trusting itself.
        let audio_blocks = normalized.iter().filter(|r| r.audio().is_some()).count();
        if audio_blocks != references.audio_label_count() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: normalization left {audio_blocks} audio-bearing references of the \
                 {} the request carried",
                references.audio_label_count()
            )));
        }
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 1. the conditioner: vision tower over the visual references, then the 66.7 GB LM ---
        let (context, text_tags) = self.encode_prompt_ref2va(&req.prompt, &normalized)?;
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 2. the VAEs: one conditioning latent per visual reference, one soundtrack per -------
        //        audio-bearing reference. Done before the DiT is mapped.
        let mut condition_rows: Vec<candle_gen::candle_core::Tensor> = Vec::new();
        let mut audio_rows_per_ref: Vec<candle_gen::candle_core::Tensor> = Vec::new();
        let mut geometries: Vec<ReferenceLatentGeometry> = Vec::new();
        {
            let vae = MiniMaxH3VideoVae::load(&self.root, &self.device, self.dtype)?;
            let audio_enc = self.load_audio_encoder()?;
            for (i, r) in normalized.iter().enumerate() {
                // Audio FIRST, mirroring the packed row order.
                let num_audio_latents = if let Some(track) = r.audio() {
                    let wave = audio_track_to_encoder_input(track, &self.device)?;
                    let posterior = audio_enc.encode(&wave)?;
                    // The MODE, never a sample — a soundtrack is deterministic conditioning.
                    let normed = audio_enc.normalize(posterior.mode())?;
                    // `[B, latent_channels, latents]` -> `[channels, latents, features]`.
                    let t = normed.permute((0, 2, 1))?.contiguous()?;
                    let latents = t.dims()[1];
                    audio_rows_per_ref.push(reference_audio_rows(&t)?);
                    latents
                } else {
                    0
                };

                let (frames, height, width) = match r {
                    Ref2VaReference::Audio(_) => (0, 0, 0),
                    Ref2VaReference::Image(img) => {
                        let pixels =
                            reference_clip_to_vae_pixels(std::slice::from_ref(img), &self.device)?;
                        let c = encode_reference_condition(
                            &vae,
                            &pixels,
                            &KeyframeNoise::Seeded,
                            i,
                            &self.device,
                        )?;
                        let s = c.dims().to_vec();
                        condition_rows.push(keyframe_condition_rows(
                            &c,
                            PATCH_SIZE,
                            &KeyframeNoise::Seeded,
                            i,
                            &self.device,
                        )?);
                        (s[2], s[3], s[4])
                    }
                    Ref2VaReference::Video(v) => {
                        // Snap DOWN to the VAE's `17n + 5` lattice so nothing is padded.
                        //
                        // The `.min` is what keeps a clip from being over-read, and it is safe
                        // ONLY because `normalize_reference_clip` refuses a normalized clip shorter
                        // than `MIN_REFERENCE_CLIP_FRAMES` (one whole chunk, 22). Without that
                        // floor a 13..21-frame clip — 13 is `sample_video_condition_frames`' own
                        // minimum — would reach the VAE OFF the lattice, where nothing in this
                        // crate says what the encoder does.
                        let keep = crate::conditioning::snap_reference_frames_down(v.frames.len())
                            .min(v.frames.len());
                        // A hard refusal, not a `debug_assert!` — for the same reason stated at the
                        // task check above: release builds strip those, so the one configuration
                        // that would actually encode off the lattice is the one with no check. The
                        // two upstream floors (`validate` and `normalize_reference_clip`) make this
                        // unreachable; that is the argument for it being cheap, not for it being
                        // absent.
                        if keep < crate::reference::MIN_REFERENCE_CLIP_FRAMES {
                            return Err(CandleError::Msg(format!(
                                "{MODEL_ID}: reference {i} snapped to {keep} frames, below the \
                                 {}-frame `17n + 5` floor — the normalized clip reached the \
                                 encoder off the lattice",
                                crate::reference::MIN_REFERENCE_CLIP_FRAMES
                            )));
                        }
                        let pixels = reference_clip_to_vae_pixels(&v.frames[..keep], &self.device)?;
                        let c = encode_reference_condition(
                            &vae,
                            &pixels,
                            &KeyframeNoise::Seeded,
                            i,
                            &self.device,
                        )?;
                        let s = c.dims().to_vec();
                        condition_rows.push(keyframe_condition_rows(
                            &c,
                            PATCH_SIZE,
                            &KeyframeNoise::Seeded,
                            i,
                            &self.device,
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
            drop(vae);
            drop(audio_enc);
            crate::dit::release_device_memory(&self.device)?;
        }
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let condition_video = if condition_rows.is_empty() {
            None
        } else {
            Some(candle_gen::candle_core::Tensor::cat(&condition_rows, 1)?.contiguous()?)
        };
        let condition_audio = if audio_rows_per_ref.is_empty() {
            None
        } else {
            Some(candle_gen::candle_core::Tensor::cat(&audio_rows_per_ref, 1)?.contiguous()?)
        };

        // --- 3. denoise on `transformer_ref` ----------------------------------------------------
        // Through `load_task_dit`, NOT a bare `MiniMaxH3Dit::load`. sc-17157 and sc-18728 landed on
        // separate branches and met here: the reference render is the exact case the seam's doc
        // comment names — a denoise path that maps its own DiT would render a `ref2va` job with the
        // caller's staged LoRA silently dropped, `Ok` and with no error. `tests/turbo_lora.rs`
        // covers the seam; the staging table below tracks this anchor.
        let dit = self.load_task_dit(task.partition())?;
        let patch = dit.config().patch_size;
        let layout = ref2va_layout(geometry, &text_tags, &geometries, patch, &self.device)?;
        let (video_rows, audio_rows) = initial_latents(geometry, patch, seed, &self.device)?;
        let video_rows = prepend_condition_rows(&layout, condition_video.as_ref(), &video_rows)?;
        let audio_rows =
            prepend_condition_audio_rows(&layout, condition_audio.as_ref(), &audio_rows)?;
        let adaln = crate::denoise::adaln_schedule(schedule)?;
        let mut model = JointDit::new(
            dit,
            layout.clone(),
            &context,
            adaln,
            AdaLnResidency::PrecomputeAndEvict,
        )?;
        drop(context);

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
            &self.device,
            &req.cancel,
            &mut on_step,
        )?;
        drop(model);
        drop(video_rows);
        drop(audio_rows);
        crate::dit::release_device_memory(&self.device)?;

        // --- 4. decode --------------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let frames = {
            let vae = MiniMaxH3VideoVae::load_decode_only(&self.root, &self.device, self.dtype)?;
            let video = vae.decode(&rendered.video)?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            frames_to_images(&revert_pixel_normalization(&video)?)?
        };
        let audio = {
            let (cfg, w) = self.load_audio_vae()?;
            let vae = crate::audio_vae::MiniMaxH3AudioVae::from_weights(
                &w,
                &cfg,
                &self.device,
                self.dtype,
            )?;
            drop(w);
            let track = vae.decode_audio_track(&rendered.audio)?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            track
        };
        let audio = finish_audio(audio, geometry)?;
        if frames.len() != geometry.joint.num_frames {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame ref2va request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
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
    ) -> Result<(candle_gen::candle_core::Tensor, Vec<u32>)> {
        use crate::reference::{
            sample_video_condition_frames, ReferencePresentation, VIDEO_SAMPLE_FPS,
            VISION_TEMPORAL_PATCH,
        };

        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let cfg = MiniMaxH3TeConfig::from_component_dir(&self.tiers.text_encoder_dir)?;

        // The tower's sources, in **sequence order** — the order the pad runs appear in, which is
        // what `forward_with_references` consumes them in.
        let mut sources: Vec<&Image> = Vec::new();
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
                        MINIMAX_H3_FPS,
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
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: a ref2va request carries no visual reference for the conditioner — a \
                 waveform never reaches it"
            )));
        }

        let vision = load_vision_tower(&self.tiers.text_encoder_dir, &self.device)?;
        let grounded = crate::text_encoder::run_vision(&vision, &sources, &self.device)?;
        drop(vision);
        crate::dit::release_device_memory(&self.device)?;

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

        let (ids, mask, tags) = tok.encode_ref2va(prompt, &presentation, &self.device)?;
        let shards =
            candle_gen::loader::sorted_safetensors(&self.tiers.text_encoder_dir, "minimax-h3 te")?;
        let prefixes = lm_prefixes(LM_PREFIX, &cfg);
        let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
        let w =
            candle_gen::Weights::from_files_filtered(&shards, &self.device, TE_STORE_DTYPE, &refs)?;
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_STORE_DTYPE)?;
        drop(w);
        let context = te.forward_with_references(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        drop(te);
        drop(grounded);
        crate::dit::release_device_memory(&self.device)?;

        if context.dims()[1] != tags.len() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: the ref2va context has {} rows but {} modality tags",
                context.dims()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }
}

impl MiniMaxH3 {
    /// The audio VAE **encoder**, built from the same three FL2VA source documents the decoder
    /// reads its constructor arguments from.
    ///
    /// Loaded at [`crate::audio_vae_encoder::ENCODER_DTYPE`] — **f32, never `self.dtype`**.
    /// diffusers pins this component with `_keep_in_fp32_modules = ["encoder", …, "mean_proj",
    /// "logs_proj"]` — the weight-normed convolutions and Snake activations degrade audibly under
    /// bf16 — and a reference soundtrack encoded at the wrong precision conditions the render with
    /// no diagnostic. `the_audio_encoder_is_pinned_to_f32_not_the_provider_dtype` scans this body.
    fn load_audio_encoder(&self) -> Result<crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder> {
        use crate::audio_vae_encoder::ENCODER_DTYPE;
        let cfg = self.audio_vae_config()?;
        let shards = candle_gen::loader::sorted_safetensors(
            &self.root.join("audio_vae"),
            "minimax-h3 audio vae",
        )?;
        let w = candle_gen::Weights::from_files(&shards, &self.device, ENCODER_DTYPE)?;
        crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::from_weights(
            &w,
            &cfg,
            &self.device,
            ENCODER_DTYPE,
        )
    }

    /// The audio VAE's config triple plus its weight map.
    ///
    /// The config comes from the FL2VA **source triple** (`config.json` + `config.yaml` +
    /// `metadata.json`) when the snapshot ships it, because that is where the reference's
    /// constructor kwargs actually live; the diffusers-repackaged `audio_vae/config.json` alone is
    /// not sufficient (see `crate::audio_config`). A snapshot without the triple falls back to the
    /// declared defaults, which those documents are asserted to reproduce exactly.
    fn load_audio_vae(
        &self,
    ) -> Result<(
        crate::audio_config::MiniMaxH3AudioVaeConfig,
        candle_gen::Weights,
    )> {
        let cfg = self.audio_vae_config()?;
        let shards = candle_gen::loader::sorted_safetensors(
            &self.root.join("audio_vae"),
            "minimax-h3 audio vae",
        )?;
        let w = candle_gen::Weights::from_files(&shards, &self.device, self.dtype)?;
        Ok((cfg, w))
    }

    /// The audio VAE's constructor arguments, shared by the decode and encode halves.
    ///
    /// Split out so the two halves cannot disagree about the geometry — `encoder_rates` fixes the
    /// hop length the encode half pads to *and* the `decoder_rates` the decode half upsamples by,
    /// and two independent readers of the same triple is exactly how those drift.
    fn audio_vae_config(&self) -> Result<crate::audio_config::MiniMaxH3AudioVaeConfig> {
        let src = self.root.join("FL2VA").join("audio_vae");
        if !src.join("config.yaml").is_file() {
            return Ok(crate::audio_config::MiniMaxH3AudioVaeConfig::default());
        }
        let read = |name: &str| -> Result<String> {
            std::fs::read_to_string(src.join(name)).map_err(|e| {
                CandleError::Msg(format!(
                    "{MODEL_ID}: read {}: {e}",
                    src.join(name).display()
                ))
            })
        };
        crate::audio_config::MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )
    }
}

/// Fit the decoded soundtrack to the clip and **check what came back**, for every generate arm.
///
/// Hoisted rather than written twice. The two arms had diverged on exactly this: the base arm
/// checked the decoded rate and channel count, the `ref2va` arm did not — so a decode that returned
/// mono, or 16 kHz, produced a silently wrong soundtrack on one path and a typed error on the other.
/// Two generate arms disagreeing about a sanity assertion is this epic's signature shape, so the
/// assertion has one home.
fn finish_audio(
    audio: candle_gen::gen_core::AudioTrack,
    geometry: &RequestGeometry,
) -> Result<candle_gen::gen_core::AudioTrack> {
    let audio = fit_audio_to_video(audio, geometry)?;
    if audio.sample_rate != AUDIO_SAMPLE_RATE || audio.channels != AUDIO_OUTPUT_CHANNELS {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: the decoded soundtrack is {} Hz / {} channels, expected \
             {AUDIO_SAMPLE_RATE} / {AUDIO_OUTPUT_CHANNELS}",
            audio.sample_rate, audio.channels
        )));
    }
    Ok(audio)
}

/// Duration a request that names neither `frames` nor `duration` renders at — the lattice floor.
const MIN_DEFAULT_SECONDS: f32 = 5.1667;

/// **Refuse a conditioning strength rather than ignore it** (sc-19571).
///
/// `Conditioning::Keyframe` carries a `strength` that its docs define as a `1 − strength` denoise
/// mask, and SceneWorks exposes it as the first/last-frame conditioning sliders. MiniMax-H3 has no
/// such mask: an anchor is mixed at the checkpoint's own trained-in
/// [`KEYFRAME_NOISE_AUG_T`](crate::conditioning::KEYFRAME_NOISE_AUG_T) = `0.999` and its rows are
/// told they sit at exactly that `t`. That number is a property of how the released model was
/// trained, not a knob, so there is nothing a caller-supplied strength could weight without
/// inventing a regime the checkpoint never saw.
///
/// Byte-for-byte the same rule as the MLX lane's `reject_keyframe_strength` — a control that one
/// backend refuses and the other silently drops is the divergence this epic keeps paying for.
fn reject_keyframe_strength(keyframes: &[candle_gen::gen_core::KeyframeRef<'_>]) -> Result<()> {
    for k in keyframes {
        if k.strength != 1.0 {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: keyframe conditioning strength is not supported (got {}) — a \
                 MiniMax-H3 anchor is held at the checkpoint's own trained-in noise augmentation \
                 t = {}, with no denoise mask to weight; use strength 1.0",
                k.strength,
                crate::conditioning::KEYFRAME_NOISE_AUG_T
            )));
        }
    }
    Ok(())
}

/// Which end of the clip each keyframe anchors, from the request's `frame_idx` values.
///
/// `fl2va` has exactly two slots — the first frame and the last — and the anchors are **positional**
/// against the fitted keyframe list, so a request that anchors the same end twice, or names a frame
/// index that is neither end, is a typed error rather than a render that quietly moved it.
pub fn keyframe_anchors(
    keyframes: &[candle_gen::gen_core::KeyframeRef<'_>],
    num_frames: usize,
) -> Result<Vec<KeyframeAnchor>> {
    reject_keyframe_strength(keyframes)?;
    let last = num_frames.saturating_sub(1);
    let mut out = Vec::with_capacity(keyframes.len());
    for k in keyframes {
        let idx = k.frame_idx.max(0) as usize;
        let anchor = if idx == 0 {
            KeyframeAnchor::First
        } else if idx == last {
            KeyframeAnchor::Last
        } else {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: a keyframe anchors frame {idx}, but fl2va has exactly two slots — the \
                 FIRST frame (0) and the LAST ({last}). An interior keyframe has no conditioning \
                 row to occupy"
            )));
        };
        out.push(anchor);
    }
    // `[First, First]` is arity-legal and semantically impossible; `crate::conditioning` rejects it
    // too, but catching it here names the request field the caller got wrong.
    if out.len() == 2 && out[0] == out[1] {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: both keyframes anchor {:?}; fl2va's two slots are the FIRST and the LAST \
             frame",
            out[0]
        )));
    }
    Ok(out)
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
/// implementation, and what [`keyframe_anchors`] legitimately does for `fl2va`'s two fixed slots)
/// would silently rewrite the request.
///
/// # The `ReferenceVideo` decision, and the two carriers it replaced
///
/// A `ref2va` video reference maps to `Conditioning::ReferenceVideo` — a carrier added to gen-core
/// for this task, rather than either of the two video variants that already existed:
///
/// * **`VideoSync` is the wrong meaning.** Its contract is video→audio Foley — the whole-clip
///   visual condition an audio decoder attends to for a *silent clip* — and its own docs say the
///   frames "are **not** spliced into a video latent". Both halves are false here.
/// * **`VideoClip` is the right *mechanism* but the wrong *vocabulary*.** Its payload is
///   `{frames, frame_idx, strength}`, and a reference can use none of the latter two: `frame_idx`
///   is a position in the generated timeline, which is exactly what a reference does not have, and
///   `strength` is a `1 − strength` denoise mask, where reference rows are fully pinned at the
///   checkpoint's own conditioning timestep. It also cannot carry a clip's **own frame rate** or
///   **own soundtrack**, both of which [`crate::reference::VideoReference`] requires: a rate that
///   was lost is a reference conditioned on at the wrong speed with nothing to raise about it.
pub(crate) fn request_references(req: &GenerationRequest) -> Result<Option<Ref2VaReferences>> {
    use candle_gen::gen_core::Conditioning;

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
            // is refused upstream by `Capabilities::validate_request`, which default-denies any
            // kind this descriptor does not advertise.
            _ => {}
        }
    }
    if refs.is_empty() {
        return Ok(None);
    }
    Ref2VaReferences::new(refs).map(Some)
}

impl Generator for MiniMaxH3 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.trim().is_empty() {
            return Err(candle_gen::gen_core::Error::Msg(format!(
                "{MODEL_ID}: prompt must not be empty"
            )));
        }
        // **The `ref2va` reference gate, at the engine boundary** (sc-17157). Every cap — 9 images,
        // 3 clips, 3 audio, 12 combined — plus the audio-is-never-alone rule and the
        // both-tasks-at-once refusal, all before a single weight is read.
        //
        // Enforced here and not only at the API for the reason the whole epic exists: a pipeline
        // handed conditioning it cannot consume does not fail, it renders a plausible clip that
        // silently ignored the thirteenth reference. `Ref2VaReferences` is constructible only
        // through its validating constructor, so the render path cannot reach a reference list
        // that skipped this.
        //
        // This is also where the task's OWN checkpoint is required to be present: `ref2va` needs
        // `transformer_ref/`, and this runs before a single weight is read. See
        // [`task_component_dirs`] for why that is not a `load`-time check.
        let (_, references) = self.resolve_task(req)?;

        // sc-19571 — the conditioning-strength refusal runs at the request boundary, not deep
        // inside a render that has already mapped the text encoder.
        reject_keyframe_strength(&req.keyframes())?;

        // **The area budget runs HERE, not only inside `generate`** (sc-17152).
        //
        // `Capabilities::max_size` is a per-edge bound and can never constrain a product, so it was
        // never the area gate — it only *looked* like one while it sat at 1344, because a square
        // inside the budget was arithmetically impossible above it. Raising it to
        // [`MAX_CANVAS_EDGE`] removes that accident, and without this call `1536 x 1536` (2.3x the
        // budget) would pass `validate` and be refused only once `request_geometry` ran, deep
        // inside a render that has already mapped the text encoder.
        //
        // The MLX provider has always run the gate at its validate boundary
        // (`mlx-gen-minimax-h3::model::validate_request`); this is the Candle lane converging on it
        // rather than a new rule. `request_geometry` is the same helper `generate` resolves
        // through, so the two cannot disagree about what is renderable.
        //
        // It runs *after* `resolve_task` so sc-17157's error ordering is unchanged — the task's own
        // checkpoint is still reported before any geometry-dependent refusal — and the resolved
        // geometry is reused by the clip-length floor below rather than resolved a second time.
        let geometry = self.request_geometry(req)?;

        // **The clip-length floor, on the same side of the boundary as everything else** (sc-17157).
        //
        // `MIN_REFERENCE_CLIP_FRAMES` was enforced only inside `normalize_reference_clip`, whose
        // sole production caller is `generate_ref2va`. So `validate` ADMITTED a short clip that
        // `generate` then refused — precisely the late failure this method exists to prevent, and
        // one the crate's own tests had frozen in place by building 4-frame clips and asserting
        // they validate.
        //
        // # Why here and not in `resolve_task`
        //
        // The surviving frame count depends on the request's `num_frames`, so the check needs the
        // geometry. Putting it in `resolve_task` would mean calling `request_geometry` there, and
        // `resolve_task` is reached from `generate_impl` too — geometry would be resolved twice per
        // render, and a t2va/fl2va request carrying no clip at all would start failing INSIDE task
        // selection on an error that has nothing to do with which checkpoint it needs.
        //
        // `validate` is where request admission already lives, and the "single decision point"
        // property `resolve_task` is documented for is about the CHECKPOINT, which is untouched.
        // The reference caps — 9 images, 3 clips, 12 total, audio-never-alone — are likewise
        // enforced outside `resolve_task`, in `Ref2VaReferences::new`. So this follows the crate's
        // existing shape rather than departing from it, and error ordering stays legible: the
        // task's own checkpoint first, then the geometry-dependent admission.
        if let Some(references) = references {
            for r in references.as_slice() {
                if let Ref2VaReference::Video(v) = r {
                    crate::reference::normalized_clip_frame_count(
                        v.frames.len(),
                        v.fps,
                        geometry.joint.num_frames,
                        MINIMAX_H3_FPS,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.contract)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<GenerationOutput> {
        Ok(self.generate_impl(req, on_progress)?)
    }
}

/// Registry entry point.
///
/// # The load-time refusals, and why the descriptor flags cannot stand in for them
///
/// [`descriptor`] sets `supports_lokr` to `false` and `supported_quants` to `&[]`. **Those fields
/// are declarations that gen-core reads on no path.** `Capabilities::validate_request` gates the
/// conditioning allowlist (which is what actually default-denies `VideoClip` — **not** the `ref2va`
/// kinds, which [`descriptor`] advertises and `validate` admits since sc-17157 landed the port) but
/// never looks at `supports_lora` or `supported_quants`, and `ProviderRegistryBuilder` does not inspect a
/// `LoadSpec` either — so without the checks below, a spec carrying `quantize`, a LoKr adapter, or a
/// per-pass / MoE adapter knob loaded clean and rendered with the caller's knob silently discarded.
///
/// `reject_unknown_components` does not cover this: it validates `spec.components`, a different
/// field from `spec.adapters` and `spec.quantize`.
///
/// Since sc-18728 `spec.adapters` is **read** rather than refused — see
/// [`MiniMaxH3::load_task_dit`] — so the checks here narrowed to the three adapter-shaped knobs this
/// lane still cannot honor: `AdapterKind::Lokr`, `pass_scales` (LTX's two-stage schedule) and
/// `moe_expert` (Wan's dual-expert selector). Each is a field a caller can set on an
/// otherwise-valid LoRA, and each would otherwise be dropped without a word.
///
/// This is the `candle-gen-mochi` idiom — refuse at the registry entry point, which returns
/// `gen_core::Result` and can therefore carry the contract-load-bearing
/// [`candle_gen::gen_core::Error::Unsupported`] that `CandleError` has no variant for.
///
/// # Every other `LoadSpec` slot is decided too
///
/// The checks below were the ones this crate shipped first, but "declared and unenforced" is a
/// *class*, not two instances. `reject_unread_slots` closes the rest of it, and the three slots
/// that are **not** refused are each accounted for there rather than left to inference.
pub fn load(spec: &LoadSpec) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    for a in &spec.adapters {
        // **LoKr is still refused**, by kind here and by file content in
        // [`crate::adapters::apply_minimax_h3_adapters`]. This DiT's seam installs
        // `scale·((x·A)·B)` residuals; a Kronecker delta is a different operation, so accepting one
        // would be a wrong fold rather than a weak one. Refused at the registry entry point because
        // that is the boundary carrying `Error::Unsupported`.
        if a.kind == candle_gen::gen_core::AdapterKind::Lokr {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} is staged as LoKr; the candle lane installs LoRA residuals only. \
                 Use a LoRA export of the same adapter.",
                a.path.display()
            )));
        }
        // `pass_scales` is LTX's per-distilled-stage knob and `moe_expert` is Wan's dual-expert
        // selector. This checkpoint has one denoise pass and one expert, so either is a knob that
        // would be silently discarded — the exact class `reject_unread_slots` exists to close, one
        // field deeper.
        if a.pass_scales.is_some() {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} sets `pass_scales`, which only LTX's two-stage denoise \
                 reads. MiniMax-H3 runs ONE forward per step, so a per-pass schedule has nowhere \
                 to apply and would be silently dropped.",
                a.path.display()
            )));
        }
        if a.moe_expert.is_some() {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} names a MoE expert, which only the Wan2.2 A14B dual-expert \
                 denoiser reads. This checkpoint is single-stream, so the selection would be \
                 silently dropped.",
                a.path.display()
            )));
        }
    }
    // `spec.quantize` is no longer refused wholesale (sc-20267): `Q4` and `Q8` are served by
    // `crate::tier` + `crate::quant`. What survives is the ONE tier this family does not publish —
    // `requested_tier` refuses `Nvfp4` by name, and it is run *here* as well as inside
    // `MiniMaxH3::load` because this is the boundary that carries the typed `Error::Unsupported`.
    requested_tier(spec)?;
    reject_unread_slots(spec)?;
    Ok(Box::new(MiniMaxH3::load(spec)?))
}

/// **Refuse every `LoadSpec` slot this provider does not read**, naming the ones the caller set.
///
/// The same defect class as the `adapters` / `quantize` checks above, one field over. A caller that
/// stages a ControlNet, an IP-Adapter, a PiD decoder, a face-ID bundle or an external text-encoder
/// snapshot for this model gets weights that **nothing in this crate ever opens** — grep `src/` and
/// `tests/` for any of these fields and the only hits are this function and the guard that proves
/// it. Silently discarding them is precisely what this provider was already caught doing with
/// `spec.adapters`.
///
/// This is the `candle-gen-flux2` accumulate-and-name idiom (`edit_provider.rs`), which reports
/// *every* offending slot in one message rather than only the first: a caller fixing them one round
/// trip at a time is a worse contract than one that lists them. The refusal is
/// `Error::Unsupported`, matching the sibling refusals in `candle-gen-mochi`, `candle-gen-wan` and
/// `candle-gen-ltx` for `control` / `extra_controls` / `ip_adapter`.
///
/// # The three slots that are deliberately NOT here
///
/// * **`spec.components`** — already refused, by `reject_unknown_components` inside
///   [`MiniMaxH3::load`], against this model's (empty) `required_components`.
/// * **`spec.offload_policy`** — deliberately *overridden*, not ignored: [`OFFLOAD_POLICY`] wins and
///   [`MiniMaxH3::offload_policy`] reports the enforced value, so the override is observable rather
///   than silent. Refusing a `Resident` request would break every caller that never asked for
///   staging; see the module docs.
/// * **`spec.load_shape`** — same shape. `crate::memory_strategy::LOAD_SHAPE` is pinned to the
///   loader this crate actually has and published on the memory contract whatever the spec says,
///   which `load_shape_is_pinned_to_the_loader_not_taken_from_the_spec` asserts. sc-18662 changes
///   the loader, not this decision.
fn reject_unread_slots(spec: &LoadSpec) -> candle_gen::gen_core::Result<()> {
    let mut unread: Vec<&str> = Vec::new();
    if spec.control.is_some() {
        unread.push("control");
    }
    if !spec.extra_controls.is_empty() {
        unread.push("extra_controls");
    }
    if spec.ip_adapter.is_some() {
        unread.push("ip_adapter");
    }
    if spec.pid.is_some() {
        unread.push("pid");
    }
    if spec.identity.is_some() {
        unread.push("identity");
    }
    if spec.text_encoder.is_some() {
        unread.push("text_encoder");
    }
    // `Precision::Bf16` is gen-core's "no override" sentinel, not a literal request, so only `Fp32`
    // is an ask. This provider hardcodes `DType::BF16` to match the published checkpoint's block
    // store; gen-core's own field docs say a provider that has not wired the override "rejects it
    // at `load`", which is what this is.
    if spec.precision != Precision::Bf16 {
        unread.push("precision");
    }
    if unread.is_empty() {
        return Ok(());
    }
    Err(candle_gen::gen_core::Error::Unsupported(format!(
        "{MODEL_ID}: the candle lane reads none of these LoadSpec slots — {}. No code in this \
         crate opens them, so honoring the load would render as though they were never set. \
         `control`/`extra_controls`/`ip_adapter` need the conditioning branches this lane has not \
         ported; `pid` and `identity` have no seam in this latent space; `text_encoder` is \
         co-located under the snapshot root and is not relocatable here; `precision` is pinned to \
         the checkpoint's bf16 block store. Drop the slot or pick a model that reads it.",
        unread.join(", ")
    )))
}

candle_gen::register_generators! {
    pub const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    /// Every component directory a flat (untiered) snapshot must carry — the four
    /// [`MiniMaxH3::load`] ends up requiring when nothing is staged, since an unstaged tier resolves
    /// to `root/transformer` and `root/text_encoder`.
    ///
    /// The **flat-layout** statement of the requirement: what a snapshot must hold to load with no
    /// component overrides, which is what the load-refusal tests drive and what the staging helpers
    /// build. Production iterates [`TIER_AGNOSTIC_COMPONENT_DIRS`] plus the two *resolved* tier dirs
    /// instead, so a staged tier is never held to the root.
    ///
    /// Declared **inside** this module rather than at file scope with a `#[cfg(test)]` attribute:
    /// `production_source` splits model.rs at the first `\n#[cfg(test)]\n`, so a file-scope one would
    /// truncate every source-scan guard in this module to the code above it — which is exactly what
    /// `body_of` caught when it was written that way.
    const REQUIRED_COMPONENT_DIRS: [&str; 4] = [
        crate::tier::TEXT_ENCODER_COMPONENT,
        BASE_DIT_PARTITION,
        "vae",
        "audio_vae",
    ];

    use super::*;

    /// **The offload policy is forced, not echoed** — the tripwire the measured `vramGbByTier`
    /// depends on.
    ///
    /// `OffloadPolicy` is advisory in gen-core, so the plausible regression is a `load` that reads
    /// `spec.offload_policy` "like every other provider does". That would leave the manifest number
    /// admitting a card against a sequential peak while the render ran resident — a ~3.5x
    /// under-estimate that fails as an OOM in the middle of a 20-minute render, not at admission.
    ///
    /// Asserting `Sequential` alone would be a false green (it is also the value a `Sequential`
    /// request would echo), so the load below asks for **`Resident`** and the assertion is that it
    /// did not get it.
    #[test]
    fn the_offload_policy_is_forced_sequential_even_when_resident_is_requested() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let spec =
            LoadSpec::new(WeightsSource::Dir(root)).with_offload_policy(OffloadPolicy::Resident);
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);

        let model = MiniMaxH3::load(&spec).unwrap();
        assert_eq!(
            model.offload_policy(),
            OffloadPolicy::Sequential,
            "a Resident request must NOT be honored: the manifest's measured vramGbByTier is a \
             sequential-phase peak"
        );
        assert_eq!(OFFLOAD_POLICY, OffloadPolicy::Sequential);
        assert_ne!(model.offload_policy(), spec.offload_policy);
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

    /// A generator over a staged-but-empty snapshot — enough to drive `validate`, which reads no
    /// weights.
    ///
    /// It stages through [`staged_root`] rather than bare `create_dir_all`: sc-19573's
    /// [`require_component`] refuses a component directory carrying no `.safetensors` shard, so a
    /// directory-only root no longer loads at all.
    fn validator() -> MiniMaxH3 {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let model = MiniMaxH3::load(&LoadSpec::new(WeightsSource::Dir(root))).unwrap();
        // The tempdir may drop here: for the `t2va` requests this drives, `validate` touches no
        // filesystem — `task_component_dirs(T2va)` is empty — which is the point.
        model
    }

    fn request(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            width,
            height,
            frames: Some(124),
            ..Default::default()
        }
    }

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
                CANVAS_SHORT_EDGE as i32,
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
        let model = validator();
        for (w, h) in ADVERTISED_BUCKETS {
            // The manifest list must itself be inside the checkpoint's envelope, or the engine is
            // right to refuse and the manifest is the thing that is wrong.
            assert!(
                u64::from(w) * u64::from(h) <= u64::from(crate::pipeline::CANVAS_MAX_PIXELS),
                "{w}x{h} is over the area budget"
            );
            model
                .validate(&request(w, h))
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
    ///
    /// This also pins that the area gate runs from **`validate`** on this lane. Before sc-17152 it
    /// did not: `validate` ran only the shared capability floor, and `resolve_geometry` was reached
    /// only once `generate` resolved geometry. That was invisible while `max_size` was 1344,
    /// because no square inside a 1344 ceiling can exceed the area budget.
    #[test]
    fn an_over_area_canvas_is_still_refused_by_the_area_gate() {
        let model = validator();
        let caps = &model.descriptor().capabilities;
        for (w, h) in [(1536u32, 1536u32), (1344u32, 1344u32)] {
            assert!(
                w <= caps.max_size && h <= caps.max_size,
                "{w}x{h} must be INSIDE the per-edge ceiling or this proves nothing"
            );
            assert!(w.is_multiple_of(SPATIAL_STRIDE) && h.is_multiple_of(SPATIAL_STRIDE));
            assert!(u64::from(w) * u64::from(h) > u64::from(crate::pipeline::CANVAS_MAX_PIXELS));

            let e = model.validate(&request(w, h)).unwrap_err().to_string();
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
            model
                .validate(&request(w, h))
                .unwrap_or_else(|e| panic!("{w}x{h} is the resolver's own 4:1 canvas: {e}"));
        }

        // …and one stride past it on the long edge is outside the per-edge ceiling again.
        let e = model
            .validate(&request(MAX_CANVAS_EDGE + SPATIAL_STRIDE, short))
            .unwrap_err()
            .to_string();
        assert!(e.contains("outside supported range"), "{e}");
    }

    /// This file's **production** source: the test module removed, then full-line comments stripped.
    ///
    /// Both exclusions are load-bearing, and each was observed to matter:
    ///
    /// * **The test module goes first.** This module's own staging table names every anchor the
    ///   guard searches for, so scanning the whole file finds the guard's own string literals and
    ///   the scan proves nothing about the code. The MLX sibling drops it for the same reason.
    /// * **Then comments.** Prose legitimately narrates the call sites — the module docs discuss
    ///   `drop` and the device release directly — so counting comment lines would make the gate
    ///   track the documentation rather than the code.
    fn production_source() -> String {
        let whole = include_str!("model.rs");
        let production = whole
            .split_once("\n#[cfg(test)]\n")
            .map_or(whole, |(before, _)| before);
        assert!(
            production.len() < whole.len(),
            "expected a `#[cfg(test)]` module in model.rs; if it moved, this scan is reading the \
             whole file and would match the test module's own anchor literals"
        );
        production
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The brace-matched body of one function, located by a signature that must be unique.
    fn body_of(src: &str, signature: &str) -> String {
        let start = src.find(signature).unwrap_or_else(|| {
            panic!(
                "model.rs no longer contains `{signature}`. This guard would otherwise silently \
                 stop watching a renamed function, so it fails instead of passing vacuously"
            )
        });
        assert!(
            !src[start + signature.len()..].contains(signature),
            "`{signature}` appears more than once; this scan reads only the first"
        );
        let open = start
            + src[start..]
                .find('{')
                .expect("a function signature is followed by a body");
        let mut depth = 0usize;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return src[open..open + i + 1].to_owned();
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces while extracting `{signature}`")
    }

    /// The explicit device release. Dropping the Rust binding alone is not enough — candle
    /// hands the pages back to its own allocator, so without this the next phase maps on top.
    const RELEASE: &str = "crate::dit::release_device_memory(";

    /// The two bindings every text-encoder phase holds: the weight map and the tower built
    /// from it. The tower is built *before* the map is dropped, necessarily, so a phase is a
    /// group of maps and a group of releases rather than one pair.
    const TE_MAPS: &[&str] = &[
        "Weights::from_files_filtered",
        "MiniMaxH3TextEncoder::from_weights",
    ];

    /// One staged phase: a name for the message, the source anchors that MAP its component,
    /// and the source anchors that RELEASE it.
    type Phase = (
        &'static str,
        &'static [&'static str],
        &'static [&'static str],
    );

    /// **Every heavy-component map anchor this file may contain.**
    ///
    /// Read by `every_heavy_map_is_inside_a_staged_phase`, which asserts each occurrence in the
    /// production source is inside a tabled staging function or inside a delegated helper. A new
    /// heavy component must be added here *and* to [`staged_phases`], or the two disagree loudly.
    const MAP_ANCHORS: &[&str] = &[
        // `load_from_dir` since sc-20267 — the DiT is mapped from the RESOLVED tier directory, so the
        // anchor moved with it. `MiniMaxH3Dit::load(` is the flat-layout wrapper and no longer appears
        // on any render path; `the_dit_is_mapped_from_the_resolved_tier_dir` pins that, so this scan
        // cannot go back to watching a name production stopped using.
        "MiniMaxH3Dit::load_from_dir(",
        "MiniMaxH3VideoVae::load(",
        "MiniMaxH3VideoVae::load_decode_only(",
        "load_vision_tower(",
        "Weights::from_files(",
        "Weights::from_files_filtered(",
        "MiniMaxH3TextEncoder::from_weights(",
        "MiniMaxH3AudioVae::from_weights(",
        "MiniMaxH3AudioVaeEncoder::from_weights(",
    ];

    /// Helpers that are themselves a tabled map anchor: the phase that calls them owns their
    /// staging, so the maps *inside* them are covered by the caller's entry.
    ///
    /// `load_task_dit` is the third (sc-18728): it is the single seam that maps a DiT partition and
    /// folds any staged LoRA onto it, so both render paths name `self.load_task_dit(` as their map
    /// anchor and the inner `MiniMaxH3Dit::load` lives one call deeper. Exempting it here is only
    /// safe because the pairing below is asserted in both directions — a helper listed here that no
    /// phase actually names as a map would be an uncovered hole, and reds.
    const DELEGATED_HELPERS: &[&str] = &[
        "fn load_audio_encoder(",
        "fn load_audio_vae(",
        "fn load_task_dit(",
    ];

    /// **The staged phases of every staging function in this file, in source order** — the table
    /// `every_heavy_component_is_released_before_the_next_one_is_mapped` walks.
    ///
    /// `the_offload_policy_is_forced_sequential_even_when_resident_is_requested` asserts
    /// `MiniMaxH3::offload_policy()`, and that is a real guard against a `load` that starts echoing
    /// `spec.offload_policy`. It is also *all* it is: `self.offload` is read by nothing on any
    /// render path. Delete every `drop` and every device release from `generate_impl` and that test
    /// stays green, while the manifest's measured `vramGbByTier` silently stops describing the
    /// render — the true peak becomes the ~143 GB **sum** of the components instead of their max,
    /// and the worker admits cards against a number 3.5x too small.
    ///
    /// So the guard walks the staged phases of the **five** staging functions in this file's own
    /// production source. Each phase must map its component, drop it, and release the device —
    /// **before** the next phase maps anything. Deleting any single `drop(...)`, or any single
    /// `crate::dit::release_device_memory(...)`, makes it red on its own; each was mutated
    /// individually to prove it, not as a set.
    ///
    /// # The table is not trusted to be complete
    ///
    /// A hardcoded phase table is only as exhaustive as whoever last edited it, so
    /// `every_heavy_map_is_inside_a_staged_phase` reads the same production source for **every**
    /// heavy-component map anchor in [`MAP_ANCHORS`] and asserts each occurrence falls inside a
    /// tabled function — or inside a helper that is itself a tabled map anchor. A sixth staging
    /// function, or a heavy map hidden in a new helper like `load_audio_encoder`, reds there rather
    /// than being silently uncovered.
    ///
    /// # What a source scan can and cannot see
    ///
    /// It **cannot** observe device bytes. It proves the releases are written and ordered, not that
    /// the allocator returned pages, and it says nothing about a component held alive by a path it
    /// does not name. The thing that *would* observe that is a weights-free behavioral seam — the
    /// MLX sibling's `MemoryBehaviorRegistration` shape, with per-route fixtures a harness executes
    /// without a checkpoint — and that is sc-18660's scope. Nothing in this crate can execute
    /// `generate_impl` without a ~143 GB snapshot, so this is the strongest guard available at this
    /// size: deliberately blunt, in exchange for being exact about the regression that was live.
    fn staged_phases() -> [(&'static str, &'static [Phase]); 5] {
        [
            (
                "fn encode_prompt(",
                &[(
                    "the 66.7 GB Qwen3-VL-32B text encoder",
                    TE_MAPS,
                    &["drop(w);", "drop(te);"],
                )],
            ),
            (
                "fn encode_prompt_grounded(",
                &[
                    (
                        "the Qwen3-VL vision tower",
                        &["load_vision_tower("],
                        &["drop(vision);"],
                    ),
                    (
                        "the 66.7 GB Qwen3-VL-32B text encoder",
                        TE_MAPS,
                        &["drop(w);", "drop(te);", "drop(grounded);"],
                    ),
                ],
            ),
            (
                "fn generate_impl(",
                &[
                    (
                        "the 10.4 GB video VAE, keyframe encode",
                        &["MiniMaxH3VideoVae::load("],
                        &["drop(vae);"],
                    ),
                    (
                        // `load_task_dit` is the single seam that maps the partition AND folds any
                        // staged LoRA (sc-18728), so the anchor tracks it rather than the inner
                        // `MiniMaxH3Dit::load` — which is now one call deeper and outside this
                        // function's body, where a scan of `generate_impl` cannot see it.
                        "the 66.3 GB DiT",
                        &["self.load_task_dit("],
                        &["drop(model);"],
                    ),
                    (
                        "the 10.4 GB video VAE, decode",
                        &["MiniMaxH3VideoVae::load_decode_only("],
                        &["drop(vae);"],
                    ),
                    (
                        "the audio VAE, decode",
                        &["self.load_audio_vae(", "MiniMaxH3AudioVae::from_weights("],
                        &["drop(w);", "drop(vae);"],
                    ),
                ],
            ),
            // The `ref2va` presentation (sc-17157). Same two components as `encode_prompt_grounded`
            // — the tower, then the LM — but a separate function, so a release deleted here would
            // not be caught by that entry.
            (
                "fn encode_prompt_ref2va(",
                &[
                    (
                        "the Qwen3-VL vision tower",
                        &["load_vision_tower("],
                        &["drop(vision);"],
                    ),
                    (
                        "the 66.7 GB Qwen3-VL-32B text encoder",
                        TE_MAPS,
                        &["drop(w);", "drop(te);", "drop(grounded);"],
                    ),
                ],
            ),
            // The `ref2va` render. **The DiT phase is the one that matters for sc-17157**: it maps
            // ONE 66 GB partition, chosen by `MiniMaxH3Task::partition()`, and drops it before the
            // decode VAEs. The two partitions are byte-identical in size, so a build that mapped
            // both would double the sequential peak the manifest admits cards against.
            (
                "fn generate_ref2va(",
                &[
                    (
                        "the 10.4 GB video VAE + the audio VAE encoder, reference encode",
                        &["MiniMaxH3VideoVae::load(", "self.load_audio_encoder("],
                        &["drop(vae);", "drop(audio_enc);"],
                    ),
                    (
                        // Same seam as `generate_impl`'s DiT phase: `load_task_dit` maps the
                        // partition AND folds any staged LoRA, so the inner `MiniMaxH3Dit::load`
                        // is one call deeper and outside this function's body.
                        "the 66.3 GB task DiT",
                        &["self.load_task_dit("],
                        &["drop(model);"],
                    ),
                    (
                        "the 10.4 GB video VAE, decode",
                        &["MiniMaxH3VideoVae::load_decode_only("],
                        &["drop(vae);"],
                    ),
                    (
                        "the audio VAE, decode",
                        &["self.load_audio_vae(", "MiniMaxH3AudioVae::from_weights("],
                        &["drop(w);", "drop(vae);"],
                    ),
                ],
            ),
        ]
    }

    /// **The staging itself is pinned — not merely the policy field.** See [`staged_phases`] and
    /// the long note above it for what this scan can and cannot see.
    #[test]
    fn every_heavy_component_is_released_before_the_next_one_is_mapped() {
        let staged = staged_phases();
        let src = production_source();
        for (signature, phases) in staged {
            let body = body_of(&src, signature);
            // An unbalanced brace inside a string literal defeats the matcher, and a body that ran
            // long would scan a sibling's staging as its own. Fail loudly on that rather than
            // passing on borrowed evidence. (Overrunning the last function instead hits EOF, which
            // `body_of` reports as unbalanced braces.)
            for (other, _) in staged {
                assert!(
                    other == signature || !body.contains(other),
                    "the extracted body of `{signature}` ran into `{other}`"
                );
            }

            let mut cursor = 0usize;
            for (phase, maps, releases) in phases {
                let mut first_map = usize::MAX;
                for m in *maps {
                    let at = cursor
                        + body[cursor..].find(m).unwrap_or_else(|| {
                            panic!(
                                "{signature} / {phase}: `{m}` no longer appears after the previous \
                                 phase's device release. Either the component is gone or the phase \
                                 order changed, and the measured sequential peak describes neither"
                            )
                        });
                    first_map = first_map.min(at);
                }
                let mut last_release = first_map;
                for r in *releases {
                    let at = first_map
                        + body[first_map..].find(r).unwrap_or_else(|| {
                            panic!(
                                "{signature} / {phase}: `{r}` is missing after the component is \
                                 mapped. A heavy component mapped and never dropped turns the \
                                 sequential peak into a SUM, and the manifest's measured \
                                 vramGbByTier admits cards against the max"
                            )
                        });
                    last_release = last_release.max(at);
                }
                cursor = last_release
                    + body[last_release..].find(RELEASE).unwrap_or_else(|| {
                        panic!(
                            "{signature} / {phase}: no `{RELEASE}` between this phase's last drop \
                             and the end of the function"
                        )
                    })
                    + RELEASE.len();
            }

            // Exhaustive for this function: a phase added without a release must not be able to
            // hide behind the phases that have one.
            let found = body.matches(RELEASE).count();
            assert_eq!(
                found,
                phases.len(),
                "{signature} has {found} `{RELEASE}` call(s) for {} pinned phase(s) — if a phase \
                 was added, add it to the table in this test so it is pinned too",
                phases.len()
            );
        }
    }

    /// **Every render arm evicts the AdaLN projections, so the contract's exclusion is honest.**
    ///
    /// `memory_strategy::adaln_component` declares 26.02 GB of DiT weights as
    /// `PrecomputedThenEvicted` and states that `crate::model` passes
    /// [`AdaLnResidency::PrecomputeAndEvict`] as a *literal*, so nothing a request carries can
    /// re-resident them. That was a one-arm claim when sc-18665 wrote it. sc-17157 then added a
    /// **second** `JointDit::new` call site — `generate_ref2va`, which denoises the reference
    /// partition — and a doc comment cannot see it. An arm that passed
    /// [`AdaLnResidency::Resident`] would leave the contract declaring a saving that render does
    /// not deliver, which is the OOM direction and exactly the half-of-a-pair-moved shape sc-17150
    /// and sc-19008 both shipped.
    ///
    /// So: scan the production source, and require every `JointDit::new(` construction to carry the
    /// evicting literal. The count is derived, not maintained — a third arm is covered the moment
    /// it is written.
    #[test]
    fn every_joint_dit_construction_evicts_the_adaln_projections() {
        const CONSTRUCTION: &str = "JointDit::new(";
        const EVICTING: &str = "AdaLnResidency::PrecomputeAndEvict";

        let src = production_source();
        let sites: Vec<usize> = src.match_indices(CONSTRUCTION).map(|(i, _)| i).collect();
        assert!(
            sites.len() >= 2,
            "expected at least the base and ref2va render arms to construct a `{CONSTRUCTION}`, \
             found {}. If the constructor was renamed this guard stopped watching anything",
            sites.len()
        );
        for start in sites {
            // The residency is the last argument, so the literal lives between the constructor and
            // the closing paren of that call.
            let tail = &src[start..];
            let end = tail
                .find(")?")
                .expect("a JointDit::new call is closed and fallible");
            assert!(
                tail[..end].contains(EVICTING),
                "a `{CONSTRUCTION}` at byte {start} does not pass `{EVICTING}`. \
                 memory_strategy::adaln_component declares the eviction unconditionally, so an arm \
                 that keeps the projections resident makes the published contract over-declare its \
                 saving by up to 26.02 GB:\n{}",
                &tail[..end]
            );
        }
    }

    /// **No heavy component is mapped outside a staged phase.**
    ///
    /// The table in [`staged_phases`] is hand-written, so on its own it pins only what someone
    /// remembered to list. This closes that: every occurrence of every anchor in [`MAP_ANCHORS`]
    /// must fall inside one of the tabled function bodies, or inside a [`DELEGATED_HELPERS`] body
    /// whose caller lists it as a map anchor. A sixth staging function — or a heavy map dropped
    /// into a new helper, which is exactly the shape `load_audio_encoder` has — reds here.
    ///
    /// Both halves of the pairing are asserted: every delegated helper must really be named as a
    /// map anchor by some phase, so the exemption list cannot quietly grow into a hole.
    #[test]
    fn every_heavy_map_is_inside_a_staged_phase() {
        let src = production_source();
        let staged = staged_phases();

        // The covered spans: the tabled function bodies, plus the delegated helpers'.
        let mut covered: Vec<String> = staged
            .iter()
            .map(|(signature, _)| body_of(&src, signature))
            .collect();
        for helper in DELEGATED_HELPERS {
            // `fn load_audio_vae(` -> `self.load_audio_vae(`, the anchor a phase must name.
            let anchor = format!("self.{}", helper.trim_start_matches("fn "));
            let named = staged
                .iter()
                .flat_map(|(_, phases)| phases.iter())
                .any(|(_, maps, _)| maps.contains(&anchor.as_str()));
            assert!(
                named,
                "`{helper}` is exempted as a delegated helper but no phase names `{anchor}` as a \
                 map — the exemption would then be an uncovered hole"
            );
            covered.push(body_of(&src, helper));
        }

        for anchor in MAP_ANCHORS {
            let total = src.matches(anchor).count();
            let inside: usize = covered
                .iter()
                .map(|body| body.matches(anchor).count())
                .sum();
            assert!(
                total > 0,
                "`{anchor}` no longer appears in model.rs; if the component moved, this scan is \
                 watching nothing"
            );
            assert_eq!(
                inside,
                total,
                "{} of {total} `{anchor}` occurrence(s) in model.rs sit OUTSIDE every staged \
                 phase. A heavy component mapped outside the staging is invisible to the phase \
                 table and turns the measured sequential peak into a sum",
                total - inside
            );
        }
    }

    /// **The audio VAE's encode half is pinned to f32 and does NOT follow the provider's dtype.**
    ///
    /// diffusers' `_keep_in_fp32_modules` is the contract, and the whole of it lives at one call
    /// site. The plausible regression is a tidy-up to `self.dtype` — which is `BF16` — producing a
    /// reference soundtrack encoded at the wrong precision with no diagnostic and no shape change.
    /// Asserting `posterior.dtype() == F32` in the parity suite cannot see this: that follows from
    /// the dtype the *test* passed in.
    #[test]
    fn the_audio_encoder_is_pinned_to_f32_not_the_provider_dtype() {
        use crate::audio_vae_encoder::ENCODER_DTYPE;
        assert_eq!(ENCODER_DTYPE, DType::F32);
        // ...and the provider's own dtype is genuinely a different one, or the pin is vacuous.
        let tmp = tempfile::tempdir().unwrap();
        let model = MiniMaxH3::load(&LoadSpec::new(WeightsSource::Dir(staged_root(&tmp)))).unwrap();
        assert_eq!(model.dtype, DType::BF16);
        assert_ne!(model.dtype, ENCODER_DTYPE);

        // Three: the `use`, the weight map, and the encoder itself. Both consumers must read it —
        // a map at bf16 feeding an f32 encoder is the same defect one layer up.
        let body = body_of(&production_source(), "fn load_audio_encoder(");
        assert_eq!(
            body.matches("ENCODER_DTYPE").count(),
            3,
            "both the weight map and the encoder must be built at ENCODER_DTYPE: {body}"
        );
        assert!(
            !body.contains("self.dtype"),
            "load_audio_encoder must not read the provider's bf16 store dtype: {body}"
        );
    }

    /// A staged snapshot root with every unconditionally-required component dir — enough for
    /// `load` to succeed.
    ///
    /// Each directory carries a **shard**, because a component dir is only satisfied by one: an
    /// empty directory was accepted until sc-17157's review, which is the same mid-render failure
    /// the load-time check exists to eliminate.
    fn staged_root(tmp: &tempfile::TempDir) -> PathBuf {
        let root = tmp.path().to_path_buf();
        for c in REQUIRED_COMPONENT_DIRS {
            stage_component(&root, c);
        }
        root
    }

    /// One component directory with a single (empty) `.safetensors` shard and a **dense**
    /// `config.json` in it.
    ///
    /// The `config.json` is not decoration: since sc-20267 `MiniMaxH3::load` resolves the tier and
    /// requires each tiered component's `config.json` — the marker file whose `quantization` block
    /// says which tier is staged, and the file `MiniMaxH3Dit::load_from_dir` would open at render time
    /// anyway. `mlx_gen_minimax_h3::model::load` probes it eagerly too. A component with no
    /// `config.json` is the interrupted-download shape, so a fixture without one is not a
    /// well-formed snapshot.
    fn stage_component(root: &Path, component: &str) {
        stage_component_at(&root.join(component), None);
    }

    /// [`stage_component`] at an explicit directory, optionally carrying a **packed** tier marker at
    /// `bits` and [`candle_gen::quant::MLX_GROUP_SIZE`].
    ///
    /// `None` writes a dense config (no `quantization` block) — the `bf16` tier's shape.
    fn stage_component_at(dir: &Path, packed_bits: Option<i32>) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), []).unwrap();
        let config = match packed_bits {
            Some(bits) => format!(
                r#"{{"quantization": {{"bits": {bits}, "group_size": {}}}}}"#,
                candle_gen::quant::MLX_GROUP_SIZE
            ),
            None => r#"{"num_layers": 50}"#.to_owned(),
        };
        std::fs::write(dir.join("config.json"), config).unwrap();
    }

    /// A staged **packed tier** DiT directory outside the snapshot root, as a component override — the
    /// split-install shape the manifest's per-tier subtrees produce.
    fn staged_tier_dit(tmp: &tempfile::TempDir, tier: &str, bits: i32) -> PathBuf {
        let dir = tmp.path().join("tiers").join(tier).join(BASE_DIT_PARTITION);
        stage_component_at(&dir, Some(bits));
        dir
    }

    /// A staged root that also carries the `ref2va` reference partition.
    fn staged_root_with_reference(tmp: &tempfile::TempDir) -> PathBuf {
        let root = staged_root(tmp);
        stage_component(&root, REFERENCE_DIT_PARTITION);
        root
    }

    /// A `LoadSpec` adapter spec at `kind`, with the two model-specific knobs optional.
    fn adapter_spec(
        kind: candle_gen::gen_core::AdapterKind,
        pass_scales: Option<Vec<f32>>,
        moe_expert: Option<candle_gen::gen_core::MoeExpert>,
    ) -> AdapterSpec {
        AdapterSpec {
            path: PathBuf::from("/turbo.safetensors"),
            scale: 1.0,
            kind,
            pass_scales,
            moe_expert,
        }
    }

    /// **A staged LoRA is CARRIED to the render, not dropped and not refused** (sc-18728).
    ///
    /// The predecessor of this test asserted the opposite — that `spec.adapters` was refused —
    /// because the seam did not exist. Now that it does, the property worth guarding flips: `load`
    /// must accept the spec **and** the loaded provider must still be holding it, because "accepted
    /// then discarded" is exactly the silent failure the old refusal existed to prevent, and it is
    /// indistinguishable from success at the `load` boundary alone.
    #[test]
    fn a_staged_lora_survives_load_rather_than_being_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);

        // Control: without adapters the same root loads and holds none.
        let bare = MiniMaxH3::load(&LoadSpec::new(WeightsSource::Dir(root.clone()))).unwrap();
        assert!(bare.adapters().is_empty());

        let spec = LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![adapter_spec(
            candle_gen::gen_core::AdapterKind::Lora,
            None,
            None,
        )]);
        load(&spec).expect("a LoRA spec must load now that the seam is ported");

        let model = MiniMaxH3::load(&spec).unwrap();
        assert_eq!(
            model.adapters().len(),
            1,
            "the spec's adapters must be RETAINED — a provider that accepted and forgot them would \
             render at the base checkpoint with no error at all"
        );
        assert_eq!(model.adapters()[0].scale, 1.0);
    }

    /// **The three adapter-shaped knobs this lane still cannot honor are refused by name.**
    ///
    /// Each is a field a caller can set on an otherwise-valid LoRA spec, and each is read by nothing
    /// in this crate. They are asserted **individually**, not as a set: a single "some knob is
    /// refused" arm would stay green if two of the three checks were deleted.
    ///
    /// The control at the top is what makes the arms attributable — the same path, the same scale,
    /// everything identical except the knob under test, loads cleanly.
    #[test]
    fn lokr_and_the_two_foreign_adapter_knobs_are_each_refused_individually() {
        use candle_gen::gen_core::{AdapterKind, MoeExpert};
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let with =
            |a: AdapterSpec| LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![a]);

        // Control: a plain LoRA with neither knob set loads.
        load(&with(adapter_spec(AdapterKind::Lora, None, None)))
            .expect("the plain LoRA control must load — otherwise the arms prove nothing");

        for (spec, needle) in [
            (adapter_spec(AdapterKind::Lokr, None, None), "LoKr"),
            (
                adapter_spec(AdapterKind::Lora, Some(vec![1.0, 0.5]), None),
                "pass_scales",
            ),
            (
                adapter_spec(AdapterKind::Lora, None, Some(MoeExpert::High)),
                "MoE expert",
            ),
        ] {
            // `Box<dyn Generator>` is not `Debug`, so `expect_err` is unavailable.
            let err = match load(&with(spec)) {
                Ok(_) => panic!("{needle} must be refused, not silently dropped"),
                Err(e) => e,
            };
            assert!(
                matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
                "must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(needle),
                "the refusal must name the knob it refused; wanted {needle:?}, got {msg}"
            );
        }
    }

    /// A solid RGB image of the given extent, for the reference-request fixtures.
    fn ref_image(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    fn ref_track() -> candle_gen::gen_core::AudioTrack {
        candle_gen::gen_core::AudioTrack {
            samples: vec![0.0; 64],
            sample_rate: 32_000,
            channels: 1,
            stems: Vec::new(),
        }
    }

    fn ref_conditioning() -> candle_gen::gen_core::Conditioning {
        candle_gen::gen_core::Conditioning::Reference {
            image: ref_image(64, 64),
            strength: Some(1.0),
        }
    }

    /// **All three reference modalities BIND, individually and combined** — the descriptor
    /// advertises them and `Generator::validate` admits them (sc-17157).
    ///
    /// The control is the point: before this story every one of these was a typed `Unsupported`,
    /// so a build that reverted the descriptor entry fails here rather than silently dropping
    /// Ref2VA back to a refusal.
    #[test]
    fn every_reference_modality_binds_individually_and_combined() {
        use candle_gen::gen_core::Conditioning;

        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root_with_reference(&tmp);
        let generator = load(&LoadSpec::new(WeightsSource::Dir(root))).unwrap();

        let base = GenerationRequest {
            prompt: "a cellist on a rooftop".into(),
            width: 1344,
            height: 768,
            ..Default::default()
        };
        generator.validate(&base).expect("the bare request");

        let image = ref_conditioning();
        let video = Conditioning::ReferenceVideo {
            // 24, not 4: at 24 fps source == kept, and `MIN_REFERENCE_CLIP_FRAMES` is 22.
            // A 4-frame clip is REFUSED by `validate` now, which is the point of the floor.
            frames: vec![ref_image(64, 64); 24],
            fps: 24.0,
            audio: None,
        };
        let audio = Conditioning::ReferenceAudio {
            audio: ref_track(),
            strength: None,
        };

        // Individually. Audio is never alone (`crate::reference`), so it is paired with an image —
        // and that pairing rule is asserted on its own below.
        for (name, conditioning) in [
            ("image", vec![image.clone()]),
            ("video", vec![video.clone()]),
            ("audio", vec![image.clone(), audio.clone()]),
        ] {
            let req = GenerationRequest {
                conditioning,
                ..base.clone()
            };
            generator
                .validate(&req)
                .unwrap_or_else(|e| panic!("a {name} reference must bind: {e}"));
        }

        // Combined, at the per-modality caps: 9 images + 2 clips + 1 audio = 12, the combined cap.
        let mut combined = vec![image.clone(); 9];
        combined.extend(vec![video.clone(); 2]);
        combined.push(audio.clone());
        assert_eq!(combined.len(), 12);
        generator
            .validate(&GenerationRequest {
                conditioning: combined,
                ..base.clone()
            })
            .expect("all three modalities combined, exactly at the 12-reference cap");

        // An audio-only request is refused — a waveform never reaches the conditioner.
        let e = generator
            .validate(&GenerationRequest {
                conditioning: vec![audio],
                ..base.clone()
            })
            .expect_err("audio alone leaves the visual stream unconditioned");
        assert!(e.to_string().contains("cannot be used on its own"), "{e}");

        // Keyframes and references together are two different checkpoints, so the request is
        // refused rather than silently resolved to one of them.
        let keyframe = Conditioning::Keyframe {
            image: ref_image(64, 64),
            frame_idx: 0,
            strength: 1.0,
        };
        let e = generator
            .validate(&GenerationRequest {
                conditioning: vec![keyframe, image],
                ..base
            })
            .expect_err("fl2va and ref2va are different checkpoints");
        assert!(
            e.to_string().contains("both keyframes and references"),
            "{e}"
        );
    }

    /// **Over-cap is rejected at the ENGINE boundary**, with a message naming the cap it tripped.
    ///
    /// Each pass sends exactly one reference over one cap against a control at that same cap, so a
    /// refusal cannot be credited to a different gate — and the `12 = 9 + 2 + 1` control above
    /// shows the combined cap is not simply refusing everything.
    #[test]
    fn an_over_cap_reference_request_is_rejected_at_the_engine_boundary() {
        use candle_gen::gen_core::Conditioning;

        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root_with_reference(&tmp);
        let generator = load(&LoadSpec::new(WeightsSource::Dir(root))).unwrap();
        let base = GenerationRequest {
            prompt: "a cellist on a rooftop".into(),
            width: 1344,
            height: 768,
            ..Default::default()
        };

        let image = || ref_conditioning();
        let video = || Conditioning::ReferenceVideo {
            // 24, not 4: at 24 fps source == kept, and `MIN_REFERENCE_CLIP_FRAMES` is 22.
            // A 4-frame clip is REFUSED by `validate` now, which is the point of the floor.
            frames: vec![ref_image(64, 64); 24],
            fps: 24.0,
            audio: None,
        };
        let audio = || Conditioning::ReferenceAudio {
            audio: ref_track(),
            strength: None,
        };

        // (name, at-cap request, over-cap request, the substring the refusal must carry)
        let mut over_total = vec![image(); 9];
        over_total.extend(vec![video(); 3]);
        over_total.extend(vec![audio(); 3]);
        let mut at_total = vec![image(); 9];
        at_total.extend(vec![video(); 2]);
        at_total.push(audio());

        let mut audio_at = vec![image()];
        audio_at.extend(vec![audio(); 3]);
        let mut audio_over = vec![image()];
        audio_over.extend(vec![audio(); 4]);

        for (name, at_cap, over_cap, expect) in [
            (
                "images",
                vec![image(); 9],
                vec![image(); 10],
                "at most 9 image",
            ),
            (
                "clips",
                vec![video(); 3],
                vec![video(); 4],
                "at most 3 video",
            ),
            ("audio", audio_at, audio_over, "at most 3 audio"),
            (
                "combined",
                at_total,
                over_total,
                "at most 12 references in total",
            ),
        ] {
            generator
                .validate(&GenerationRequest {
                    conditioning: at_cap,
                    ..base.clone()
                })
                .unwrap_or_else(|e| {
                    panic!("{name}: the AT-cap request must be admitted, else the gate is off by one: {e}")
                });
            let e = match generator.validate(&GenerationRequest {
                conditioning: over_cap,
                ..base.clone()
            }) {
                Ok(()) => panic!("{name}: the over-cap request must be refused"),
                Err(e) => e.to_string(),
            };
            assert!(e.contains(expect), "{name}: {e}");
        }
    }

    /// **The task selects the checkpoint, and only `ref2va` moves.**
    ///
    /// The two partitions are structurally indistinguishable — same config, same 638 tensor names —
    /// so this mapping is the *only* thing standing between a `ref2va` request and a plausible
    /// render off the wrong 66 GB.
    #[test]
    fn the_task_picks_the_partition_and_only_ref2va_moves() {
        assert_eq!(MiniMaxH3Task::T2va.partition(), BASE_DIT_PARTITION);
        assert_eq!(MiniMaxH3Task::Fl2va.partition(), BASE_DIT_PARTITION);
        assert_eq!(MiniMaxH3Task::Ref2va.partition(), REFERENCE_DIT_PARTITION);
        assert_ne!(BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION);

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
        let e = MiniMaxH3Task::resolve(true, true).unwrap_err().to_string();
        assert!(e.contains("both keyframes and references"), "{e}");

        // Only the reference task runs the tower over references; `t2va` has no visual source.
        assert!(!MiniMaxH3Task::T2va.needs_vision_tower());
        assert!(MiniMaxH3Task::Fl2va.needs_vision_tower());
        assert!(MiniMaxH3Task::Ref2va.needs_vision_tower());
    }

    /// `request_references` preserves the request's **order** across modalities, because that order
    /// fixes the presentation labels and advances the shared rotary clock.
    ///
    /// The assertion is on the interleaved sequence, not on counts: a grouping implementation (the
    /// obvious one) produces the same counts and a different request.
    #[test]
    fn the_reference_list_keeps_the_requests_order_across_modalities() {
        use candle_gen::gen_core::Conditioning;

        let req = GenerationRequest {
            prompt: "p".into(),
            conditioning: vec![
                Conditioning::ReferenceVideo {
                    // 30 at 30 fps normalizes to 24 frames, clearing the 22-frame floor.
                    frames: vec![ref_image(64, 64); 30],
                    fps: 30.0,
                    audio: None,
                },
                ref_conditioning(),
                Conditioning::ReferenceAudio {
                    audio: ref_track(),
                    strength: None,
                },
                ref_conditioning(),
            ],
            ..Default::default()
        };
        let refs = request_references(&req).unwrap().expect("references");
        let kinds: Vec<crate::reference::ReferenceKind> =
            refs.as_slice().iter().map(|r| r.kind()).collect();
        use crate::reference::ReferenceKind::{Audio, Image as I, Video};
        assert_eq!(kinds, vec![Video, I, Audio, I]);

        // The clip's OWN rate survives the mapping — `VideoClip` could not carry it, and a rate
        // silently defaulted to 24 plays a 30 fps reference 25% fast with nothing to raise.
        match &refs.as_slice()[0] {
            crate::reference::Ref2VaReference::Video(v) => assert_eq!(v.fps, 30.0),
            other => panic!("expected a clip, got {other:?}"),
        }

        // A request with no reference of any modality is `None`, which is the t2va/fl2va
        // discriminator.
        assert!(request_references(&GenerationRequest {
            prompt: "p".into(),
            ..Default::default()
        })
        .unwrap()
        .is_none());
    }

    /// **A `q4` request RESOLVES the `q4` tier** — the inversion of the refusal this crate shipped
    /// before sc-20267, which turned every tier request into `Error::Unsupported`.
    ///
    /// This asserts on the **seam**, with non-default evidence, rather than on "it loaded": a load
    /// that silently ignored `spec.quantize` would also return `Ok`. So the staged tier dir is placed
    /// OUTSIDE the snapshot root and both halves are read back — `tier_paths().dit_dir` must be the
    /// staged path (not `root/transformer`, which is also present and dense), and `requested_tier()`
    /// must be `Q4`. A resolver that fell back to the root would produce a different path, and a
    /// mapping that dropped the request would produce `None`.
    ///
    /// **Resolution and reconcile only** — the name says so deliberately. The staged shards here are
    /// empty stubs, so nothing is decoded and no packed tensor is built; what is proven is that the
    /// request selects the tier, the on-disk marker is read, and a disagreeing tier is refused. The
    /// packed *load* is proven where weights exist: `tests/quant_policy.rs` drives real MLX triples
    /// through `crate::quant`, and the boogu tower tests drive the vision half.
    #[test]
    fn a_q4_request_resolves_and_reconciles_the_q4_tier() {
        let tmp = tempfile::tempdir().unwrap();
        // A complete flat DENSE snapshot, plus a packed q4 DiT staged elsewhere. The two disagree, so
        // whichever one the provider resolved is observable.
        let root = staged_root(&tmp);
        let staged = staged_tier_dit(&tmp, "q4", 4);

        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        spec.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(staged.clone()),
        );
        // **Through the REGISTRY entry point first.** That is where the deleted refusal lived, so this
        // is the assertion that actually inverts the old test — a bare `MiniMaxH3::load` call still
        // passes with the wholesale `spec.quantize` refusal put back, which is the mutation that caught
        // this test's first draft.
        load(&spec).expect("a q4 request must be ADMITTED at the registry entry point");

        let provider =
            MiniMaxH3::load(&spec).expect("a q4 request over a staged q4 tier must load");

        assert_eq!(
            provider.requested_tier(),
            Some(crate::tier::Tier::Q4),
            "the request must reach the resolver as an asserted tier, not be dropped"
        );
        assert_eq!(
            provider.tier_paths().dit_dir,
            staged,
            "the DiT must come from the STAGED tier dir, not from root/transformer"
        );
        assert_ne!(
            provider.tier_paths().dit_dir,
            root.join(BASE_DIT_PARTITION),
            "a resolver that fell back to the root would render the dense checkpoint under a q4 request"
        );
        // The reference partition is the staged dir's SIBLING, so a split install is probed where the
        // directory actually is rather than under the snapshot root.
        assert_eq!(
            provider.tier_paths().reference_dit_dir,
            staged.with_file_name(REFERENCE_DIT_PARTITION)
        );
        // ...and the tier-agnostic components still resolve against the root.
        assert_eq!(provider.tier_paths().shared_root, root);

        // The marker really was read: the same staged dir under a q8 request is refused, so the
        // reconcile is doing work rather than accepting anything.
        let mut q8 = LoadSpec::new(WeightsSource::Dir(root)).with_quant(Quant::Q8);
        q8.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(staged),
        );
        let err = match MiniMaxH3::load(&q8) {
            Ok(_) => panic!("a q8 request over a staged q4 tier must be refused"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("on-disk tier is authoritative"),
            "{err}"
        );
    }

    /// **`spec.quantize == None` is the third state, and it must NOT become `Tier::Bf16`.**
    ///
    /// A packed tier staged with no assertion loads — which is what MLX does
    /// (`reconcile_tier` admits `(Some(_), None)`) and what every pre-sc-19120 install needs, since
    /// the loaders recover the tier from `{base}.scales` regardless. Mapping `None` onto `Bf16` would
    /// make this a refusal and reject those installs.
    #[test]
    fn an_unasserted_request_admits_a_packed_tier_rather_than_demanding_dense() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let staged = staged_tier_dit(&tmp, "q4", 4);

        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(staged.clone()),
        );
        assert!(spec.quantize.is_none(), "the premise of this test");
        let provider =
            MiniMaxH3::load(&spec).expect("a packed tier under NO assertion must be admitted");
        assert_eq!(
            provider.requested_tier(),
            None,
            "an absent request must stay absent — Bf16 is a positive assertion of denseness"
        );
        assert_eq!(provider.tier_paths().dit_dir, staged);

        // The group size is still validated on the unasserted path, because a mismatched group derives
        // a legal-looking wrong bit width instead of failing cleanly (sc-15154).
        let bad = tmp
            .path()
            .join("tiers")
            .join("bad")
            .join(BASE_DIT_PARTITION);
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("model-00001-of-00001.safetensors"), []).unwrap();
        std::fs::write(
            bad.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(staged_root(&tmp)));
        spec.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(bad),
        );
        let err = match MiniMaxH3::load(&spec) {
            Ok(_) => panic!("a group-32 tier must be refused even with no asserted tier"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("group_size 32"), "{err}");
    }

    /// **`Nvfp4` is refused by name, not mapped onto the `q4` tier.**
    ///
    /// It reports `bits() == 4`, so mapping it to `Tier::Q4` would let it reconcile *cleanly* against
    /// a `q4` marker and render int4-affine under an NVFP4 request — the silent numerics swap epic
    /// 11037's SC#5 forbids. The refusal must be the typed `Unsupported` at the registry boundary.
    #[test]
    fn an_nvfp4_request_is_refused_rather_than_reconciled_against_a_q4_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let staged = staged_tier_dit(&tmp, "q4", 4);

        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_quant(Quant::Nvfp4);
        spec.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(staged),
        );
        let err = match load(&spec) {
            Ok(_) => {
                panic!("an NVFP4 request must be refused — this family publishes no NVFP4 tier")
            }
            Err(e) => e,
        };
        assert!(
            matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
            "must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("Nvfp4"),
            "the refusal must name the tier: {msg}"
        );
        assert!(
            msg.contains("E2M1") || msg.contains("different numerics"),
            "the refusal must say WHY it is not a q4 substitute: {msg}"
        );
        // And the advertised set agrees with the refusal.
        assert!(!descriptor()
            .capabilities
            .supported_quants
            .contains(&Quant::Nvfp4));
    }

    /// **The DiT is mapped from the RESOLVED tier dir, and the flat wrapper is off the render path.**
    ///
    /// `MAP_ANCHORS` watches `MiniMaxH3Dit::load_from_dir(`. If a render path ever went back to the
    /// `root`-joining `MiniMaxH3Dit::load(` wrapper, that scan would silently stop covering it *and*
    /// a staged tier would be ignored in favour of `root/transformer`. Both halves are asserted:
    /// the wrapper does not appear in production source, and `partition_dir` really returns the
    /// staged dirs.
    #[test]
    fn the_dit_is_mapped_from_the_resolved_tier_dir() {
        let src = production_source();
        assert!(
            src.contains("MiniMaxH3Dit::load_from_dir("),
            "the render path must map the DiT from a resolved directory"
        );
        assert!(
            !src.contains("MiniMaxH3Dit::load("),
            "the `root`-joining wrapper must not appear on any render path — it would ignore a \
             staged tier and fall back to root/transformer"
        );

        // ...and the mapping really is the staged dir, per partition.
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let staged = staged_tier_dit(&tmp, "q4", 4);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        spec.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(staged.clone()),
        );
        let provider = MiniMaxH3::load(&spec).expect("load");
        assert_eq!(provider.partition_dir(BASE_DIT_PARTITION).unwrap(), staged);
        assert_eq!(
            provider.partition_dir(REFERENCE_DIT_PARTITION).unwrap(),
            staged.with_file_name(REFERENCE_DIT_PARTITION),
            "the reference partition is the staged DiT's sibling"
        );
        assert!(
            provider.partition_dir("vae").is_err(),
            "a non-partition name is a programming error, not a silent fallback"
        );
    }

    /// **The DiT, the text encoder and the VAEs may come from THREE DIFFERENT roots.**
    ///
    /// Not a hypothetical: the SceneWorks catalog's off-Mac tier install stages the DiT and the text
    /// encoder from the `SceneWorks/minimax-h3-mlx` rehost while the VAEs and tokenizer come from
    /// `MiniMaxAI/MiniMax-H3`, because only the two large components are published per tier. So a tier
    /// load must never assume one shared snapshot root, and the two tiered components must be resolved
    /// **independently of each other** (sc-19120) as well as of the root.
    #[test]
    fn the_two_tiered_components_resolve_from_separate_roots() {
        let tmp = tempfile::tempdir().unwrap();
        // The snapshot root carries ONLY the tier-agnostic components — no transformer/, no
        // text_encoder/. This is the off-Mac split-install shape.
        let root = tmp.path().join("shared");
        for c in TIER_AGNOSTIC_COMPONENT_DIRS {
            stage_component(&root, c);
        }
        let dit = tmp
            .path()
            .join("rehost")
            .join("q4")
            .join(BASE_DIT_PARTITION);
        stage_component_at(&dit, Some(4));
        // The TE is packed at q8 while the DiT is q4 — a legal configuration, because the engine
        // deliberately does NOT couple the two tiers (sc-19120). If it did, this load would fail.
        let te = tmp
            .path()
            .join("rehost")
            .join("q8")
            .join(crate::tier::TEXT_ENCODER_COMPONENT);
        stage_component_at(&te, Some(8));

        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        spec.components.insert(
            crate::tier::DIT_COMPONENT.to_owned(),
            WeightsSource::Dir(dit.clone()),
        );
        spec.components.insert(
            crate::tier::TEXT_ENCODER_COMPONENT.to_owned(),
            WeightsSource::Dir(te.clone()),
        );
        let provider = MiniMaxH3::load(&spec)
            .expect("three roots must load — the catalog's off-Mac tier install is exactly this");

        assert_eq!(provider.tier_paths().dit_dir, dit);
        assert_eq!(provider.tier_paths().text_encoder_dir, te);
        assert_eq!(provider.tier_paths().shared_root, root);
        assert!(
            !root.join(BASE_DIT_PARTITION).exists(),
            "the premise: the root really has no DiT, so a root-relative resolver would have failed"
        );
        // The asserted q4 applies to the DiT only. The TE's own tier is auto-detected from its
        // `.scales`, which is why `require_text_encoder` takes no `Tier`.
        assert_eq!(provider.requested_tier(), Some(crate::tier::Tier::Q4));
        assert_eq!(
            crate::tier::MiniMaxH3TierPaths::staged_bits(&te).unwrap(),
            Some(8),
            "a q8 text encoder beside a q4 DiT is legal and is NOT reconciled against it"
        );
    }

    /// A **dense** snapshot under an explicit `q4` request is refused — H3 does not quantize at load.
    #[test]
    fn a_tier_request_over_a_dense_snapshot_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let spec = LoadSpec::new(WeightsSource::Dir(root)).with_quant(Quant::Q4);
        let err = match MiniMaxH3::load(&spec) {
            Ok(_) => panic!("a q4 request over a dense snapshot cannot be satisfied"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("does not quantize at load"),
            "{err}"
        );
    }

    /// **Every `LoadSpec` slot this provider does not read is refused, and each is proven alone.**
    ///
    /// `spec.adapters` and `spec.quantize` were the two this crate shipped first, but "declared and
    /// unenforced" is a *class*. `control`, `extra_controls` and `ip_adapter` are explicitly refused
    /// by `candle-gen-mochi`, `candle-gen-wan` and `candle-gen-ltx` — the very sibling idiom this
    /// provider cites — and H3 reads none of them, nor `pid`, `identity`, `text_encoder` or a
    /// non-default `precision`. Staging any of them produced a clean load and a render that behaved
    /// as though the caller had never set it.
    ///
    /// Each pass mutates exactly ONE slot against a control that loads the SAME fixture with the
    /// slot unset, so a refusal cannot be credited to a different slot or to a broken root. The
    /// message assertion reads the generated slot LIST, not the static prose (which names several
    /// slots and would satisfy a substring check even with the guard deleted) — so deleting any one
    /// `if` in `reject_unread_slots` fails exactly one pass.
    #[test]
    fn every_unread_loadspec_slot_is_refused_and_names_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let src = || WeightsSource::Dir(root.clone());

        for slot in [
            "control",
            "extra_controls",
            "ip_adapter",
            "pid",
            "identity",
            "text_encoder",
            "precision",
        ] {
            // The control, re-run every pass: the same root with the slot unset must still load.
            load(&LoadSpec::new(src())).unwrap_or_else(|e| {
                panic!(
                    "{slot}: the bare spec must load — otherwise the refusal proves nothing: {e}"
                )
            });

            let mut spec = LoadSpec::new(src());
            match slot {
                "control" => spec.control = Some(src()),
                "extra_controls" => spec.extra_controls = vec![src()],
                "ip_adapter" => spec.ip_adapter = Some(src()),
                "pid" => {
                    spec.pid = Some(candle_gen::gen_core::PidWeights {
                        checkpoint: src(),
                        gemma: src(),
                    })
                }
                "identity" => {
                    spec.identity = Some(candle_gen::gen_core::IdentityWeights {
                        encoder: Some(src()),
                        eva: None,
                        face_dir: None,
                    })
                }
                "text_encoder" => spec.text_encoder = Some(src()),
                // `Precision::Bf16` is gen-core's "no override" sentinel, so `Fp32` is the only ask.
                "precision" => spec.precision = Precision::Fp32,
                other => unreachable!("unhandled slot `{other}`"),
            }

            let err = match load(&spec) {
                Ok(_) => panic!(
                    "`{slot}` was set and the load SUCCEEDED — nothing in this crate reads it, so \
                     the render would silently ignore the caller's weights"
                ),
                Err(e) => e,
            };
            assert!(
                matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
                "{slot}: must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("slots — {slot}.")),
                "the refusal must NAME the offending slot in its list so a caller knows what to \
                 drop: {msg}"
            );
        }
    }

    /// The three `LoadSpec` slots that are decided but deliberately **not** refused stay that way.
    ///
    /// `offload_policy` and `load_shape` are *overridden and reported*, not ignored — refusing them
    /// would break every caller that never asked for staging — and `components` is refused a layer
    /// down by `reject_unknown_components`. This pins that a bare spec carrying the non-default
    /// residency knobs still loads, so the refusal list above cannot quietly grow to cover them.
    #[test]
    fn the_residency_knobs_are_overridden_rather_than_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Resident)
            .with_load_shape(candle_gen::gen_core::LoadShape::DeferredMaterialization);
        load(&spec).unwrap_or_else(|e| {
            panic!("the residency knobs are honored by override, not refused, but load failed: {e}")
        });

        // ...and an unknown named component IS refused, one layer down.
        let unknown = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_component("transformer_ref", WeightsSource::Dir(root));
        assert!(
            load(&unknown).is_err(),
            "an unrecognized named component must be refused by reject_unknown_components"
        );
    }

    /// A snapshot missing any one of the unconditionally-required component directories fails at
    /// `load`, naming the **path** — not twenty minutes later inside a phase.
    ///
    /// The assertion is on `root.join(missing)`, not on the bare component name: the refusal
    /// message used to interpolate `{REQUIRED_COMPONENT_DIRS:?}`, which names every component, so
    /// `e.contains(missing)` could not fail whatever the loader did.
    ///
    /// An **empty** component directory is refused too, and by the same check. `is_dir()` alone was
    /// the gate, so a shard-less `transformer_ref/` — the shape an interrupted download leaves —
    /// loaded clean and failed mid-render, which is precisely what this check exists to prevent.
    #[test]
    fn a_missing_or_empty_component_fails_the_load_and_names_its_path() {
        for missing in REQUIRED_COMPONENT_DIRS {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().to_path_buf();
            for c in REQUIRED_COMPONENT_DIRS.iter().filter(|c| **c != missing) {
                stage_component(&root, c);
            }
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            let path = root.join(missing).display().to_string();

            let e = match MiniMaxH3::load(&spec) {
                Ok(_) => panic!("a snapshot without `{missing}/` must not load"),
                Err(e) => e.to_string(),
            };
            assert!(e.contains(&path), "{missing} absent: {e}");

            // Present but EMPTY: still refused, and the message says why.
            std::fs::create_dir_all(root.join(missing)).unwrap();
            let e = match MiniMaxH3::load(&spec) {
                Ok(_) => panic!("an empty `{missing}/` must not load — it fails mid-render"),
                Err(e) => e.to_string(),
            };
            assert!(
                e.contains(&path) && e.contains("no `.safetensors` shard"),
                "{e}"
            );

            // Control: the same root with a shard in it loads, so the refusals are attributable.
            stage_component(&root, missing);
            MiniMaxH3::load(&spec).expect("the control must load, else the refusals prove nothing");
        }
    }

    /// **`transformer_ref` is required of the `ref2va` REQUEST, not of the snapshot** (sc-17157).
    ///
    /// The blast radius is the point. A base-only install is the *ordinary* off-Mac shape: every
    /// `transformer_ref` row in SceneWorks' manifest is `platforms: ["macos"]` and the off-Mac
    /// artifact set carries none, so such an install has no catalog route to the partition.
    /// Requiring it in `load` would fail provider construction outright and take `t2va` and `fl2va`
    /// offline with it — a whole model offline for a capability neither of them uses. So this asserts
    /// both halves: such a snapshot **loads and serves the other two tasks**, and a reference request
    /// against it is refused at the engine boundary, before a weight is read, naming the missing
    /// directory.
    ///
    /// The refusal has to be loud precisely because `ref2va` **is** ported here (sc-17157, landed):
    /// `descriptor` advertises all three reference kinds and `validate` admits them, so the arm is
    /// reachable on a snapshot that cannot serve it.
    ///
    /// (Two retracted premises are recorded in the module docs: the sc-19517 "not published" claim,
    /// and a later "default-denies `ref2va` until sc-17157" claim. Neither is asserted here.)
    #[test]
    fn a_ref2va_request_needs_the_reference_partition_but_the_other_tasks_do_not() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        assert!(!root.join(REFERENCE_DIT_PARTITION).exists());

        let generator = load(&LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("a base-only snapshot must still construct the provider");

        let base = GenerationRequest {
            prompt: "a cellist on a rooftop".into(),
            width: 1344,
            height: 768,
            ..Default::default()
        };
        generator.validate(&base).expect("t2va must still work");
        generator
            .validate(&GenerationRequest {
                conditioning: vec![candle_gen::gen_core::Conditioning::Keyframe {
                    image: ref_image(64, 64),
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..base.clone()
            })
            .expect("fl2va must still work");

        // ...and the reference request is refused, naming the path it needs.
        let e = match generator.validate(&GenerationRequest {
            conditioning: vec![ref_conditioning()],
            ..base.clone()
        }) {
            Ok(()) => panic!("a ref2va request must not be admitted without transformer_ref/"),
            Err(e) => e.to_string(),
        };
        let path = root.join(REFERENCE_DIT_PARTITION).display().to_string();
        assert!(e.contains(&path), "the refusal must name the path: {e}");
        // The refusal attributes the gap to the ARTIFACT fact — the catalog rows are macOS-only —
        // and names the partition. Both negative assertions pin retracted premises OUT of
        // user-facing text: sc-19517's "not published", and the later "until sc-17157 lands the
        // port" (sc-17157 landed, and this provider advertises `ref2va`).
        assert!(e.contains("macOS-only"), "{e}");
        assert!(e.contains(REFERENCE_DIT_PARTITION), "{e}");
        assert!(
            !e.contains("sc-19517"),
            "the retracted hosting premise must not reappear in user-facing text: {e}"
        );
        assert!(
            !e.contains("sc-17157"),
            "sc-17157 LANDED — a refusal must not describe the port as pending: {e}"
        );

        // Control: stage the partition and the very same request is admitted.
        stage_component(&root, REFERENCE_DIT_PARTITION);
        load(&LoadSpec::new(WeightsSource::Dir(root)))
            .unwrap()
            .validate(&GenerationRequest {
                conditioning: vec![ref_conditioning()],
                ..base
            })
            .expect("with transformer_ref/ present the request is admitted");
    }

    /// **A `ref2va` REQUEST resolves to the reference partition** — the request→loader half of the
    /// checkpoint decision, weights-free.
    ///
    /// `tests/ref2va_checkpoint.rs` proves the DiT loader honours the partition it is handed and
    /// that both load sites take theirs from `task.partition()` and from nowhere else. This is the
    /// other end of that chain: a real [`GenerationRequest`] carrying each reference kind, driven
    /// through the **same** `resolve_task` that `generate_impl` consumes. Neither half alone is the
    /// claim; together they are.
    #[test]
    fn a_reference_request_resolves_to_the_reference_partition() {
        use candle_gen::gen_core::Conditioning;

        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root_with_reference(&tmp);
        let model = MiniMaxH3::load(&LoadSpec::new(WeightsSource::Dir(root))).unwrap();
        let base = GenerationRequest {
            prompt: "a cellist on a rooftop".into(),
            width: 1344,
            height: 768,
            ..Default::default()
        };

        for (name, conditioning) in [
            ("image", vec![ref_conditioning()]),
            (
                "video",
                vec![Conditioning::ReferenceVideo {
                    // 24 frames at 24 fps — see the floor note above.
                    frames: vec![ref_image(64, 64); 24],
                    fps: 24.0,
                    audio: None,
                }],
            ),
            (
                "audio",
                vec![
                    ref_conditioning(),
                    Conditioning::ReferenceAudio {
                        audio: ref_track(),
                        strength: None,
                    },
                ],
            ),
        ] {
            let (task, references) = model
                .resolve_task(&GenerationRequest {
                    conditioning,
                    ..base.clone()
                })
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(task, MiniMaxH3Task::Ref2va, "{name}");
            assert!(references.is_some(), "{name}");
            assert_eq!(
                task.partition(),
                REFERENCE_DIT_PARTITION,
                "{name}: a reference request must denoise from the reference partition"
            );
        }

        // The negative halves: without references the very same call stays on the base partition.
        let (task, references) = model.resolve_task(&base).unwrap();
        assert_eq!(task, MiniMaxH3Task::T2va);
        assert!(references.is_none());
        assert_eq!(task.partition(), BASE_DIT_PARTITION);

        let (task, _) = model
            .resolve_task(&GenerationRequest {
                conditioning: vec![Conditioning::Keyframe {
                    image: ref_image(64, 64),
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..base
            })
            .unwrap();
        assert_eq!(task, MiniMaxH3Task::Fl2va);
        assert_eq!(task.partition(), BASE_DIT_PARTITION);
    }

    /// **`validate` refuses exactly the clips `generate` would refuse** (sc-17157).
    ///
    /// `MIN_REFERENCE_CLIP_FRAMES` used to be enforced only inside `normalize_reference_clip`,
    /// which runs deep inside `generate_ref2va`. `validate` therefore ADMITTED a short clip that
    /// `generate` then rejected, after the video VAE and the audio encoder had been mapped — the
    /// late-failure shape this whole story removes. Four tests in this module had frozen the defect
    /// in place by building 4-frame clips and asserting they validate; they now build clips that
    /// clear the floor, and this is the negative half they were missing.
    ///
    /// The last loop drives BOTH sides on the same inputs, so "the boundary and the render agree"
    /// is executed rather than asserted.
    #[test]
    fn validate_refuses_a_clip_shorter_than_the_vae_chunk_floor() {
        use candle_gen::gen_core::Conditioning;

        use crate::reference::MIN_REFERENCE_CLIP_FRAMES;

        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root_with_reference(&tmp);
        let model = MiniMaxH3::load(&LoadSpec::new(WeightsSource::Dir(root.clone()))).unwrap();
        let generator = load(&LoadSpec::new(WeightsSource::Dir(root))).unwrap();
        let base = GenerationRequest {
            prompt: "a cellist on a rooftop".into(),
            width: 1344,
            height: 768,
            ..Default::default()
        };
        let clip = |frames: usize, fps: f32| GenerationRequest {
            conditioning: vec![Conditioning::ReferenceVideo {
                frames: vec![ref_image(64, 64); frames],
                fps,
                audio: None,
            }],
            ..base.clone()
        };

        // The default geometry is well above the floor, so it is the FLOOR that bites below and not
        // the `num_frames` truncation.
        let num_frames = model.request_geometry(&base).unwrap().joint.num_frames;
        assert!(num_frames > MIN_REFERENCE_CLIP_FRAMES, "{num_frames}");

        // Clips that normalize BELOW 22, including the 13..=21 window the floor exists for —
        // `sample_video_condition_frames`' own minimum is 13, so that window was reachable.
        for (frames, fps) in [(4usize, 24.0f32), (21, 24.0), (4, 30.0), (26, 30.0)] {
            let e = match generator.validate(&clip(frames, fps)) {
                Ok(()) => panic!(
                    "{frames} frames at {fps} fps normalizes below the floor and must be refused \
                     by validate, not by generate"
                ),
                Err(e) => e.to_string(),
            };
            assert!(
                e.contains(&format!("{MIN_REFERENCE_CLIP_FRAMES}-frame floor"))
                    && e.contains("17n + 5"),
                "{frames}@{fps}: the refusal must name the floor it hit: {e}"
            );
        }

        // Controls: AT the floor is admitted, so this is a floor and not a ban on clips.
        for (frames, fps) in [(22usize, 24.0f32), (24, 24.0), (28, 30.0)] {
            generator.validate(&clip(frames, fps)).unwrap_or_else(|e| {
                panic!("{frames}@{fps} clears the floor and must validate: {e}")
            });
        }

        // **The two sides agree on the same input.** What `validate` admits the render path admits,
        // and what it refuses the render path refuses. That gap IS the defect, so it is the thing
        // asserted rather than each side's behavior separately.
        for (frames, fps) in [(4usize, 24.0f32), (21, 24.0), (22, 24.0), (30, 30.0)] {
            let by_validate = generator.validate(&clip(frames, fps)).is_ok();
            let by_render = crate::reference::normalize_reference_clip(
                &vec![ref_image(64, 64); frames],
                f64::from(fps),
                num_frames,
                SPATIAL_STRIDE as i32,
                CANVAS_SHORT_EDGE as i32,
                i64::from(crate::pipeline::CANVAS_MAX_PIXELS),
                MINIMAX_H3_FPS,
            )
            .is_ok();
            assert_eq!(
                by_validate, by_render,
                "{frames}@{fps}: validate and the render path disagree — that gap IS the defect"
            );
        }
    }

    /// The descriptor advertises what this lane can actually do — and, just as load-bearing, does
    /// **not** advertise what it cannot.
    #[test]
    fn the_descriptor_advertises_only_the_ported_surface() {
        let d = descriptor();
        assert_eq!(d.id, "minimax_h3");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        // Guidance-distilled.
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        // `fl2va` plus all three `ref2va` kinds (sc-17157) — and NOT `VideoClip`, which this model
        // has no in-context-clip mechanism for and which default-deny turns into a typed
        // `Unsupported`.
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Keyframe,
                ConditioningKind::Reference,
                ConditioningKind::ReferenceVideo,
                ConditioningKind::ReferenceAudio,
            ]
        );
        assert!(!d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::VideoClip));
        // **The turbo LoRA seam IS ported** (sc-18728) — and this bit is only an advertisement:
        // gen-core reads `supports_lora` on no path, so what a consumer can rely on is
        // `crate::adapters`, exercised end-to-end in `tests/turbo_lora.rs`.
        assert!(d.capabilities.supports_lora);
        // LoKr stays false and is enforced twice in real code (by kind in `load`, by file content
        // in the installer) precisely because this flag refuses nothing on its own.
        assert!(!d.capabilities.supports_lokr);
        // **The tier loader IS ported** (sc-20267) — `crate::tier` resolves the per-tier component
        // dirs and `crate::quant` builds the packed bases. Like `supports_lora` this is only an
        // advertisement; the enforcement is the reconcile plus the `Nvfp4` refusal in `load`.
        assert_eq!(
            d.capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8],
            "the two tiers `mlx_gen_minimax_h3::convert` publishes and this lane reads"
        );
        // `Nvfp4` is deliberately absent: it reports 4 bits, so advertising it would let an NVFP4
        // request reconcile against a `q4` marker and render the wrong tier silently.
        assert!(!d.capabilities.supported_quants.contains(&Quant::Nvfp4));
        // The residency lifecycle IS wired — and forced — so the discovery signal says so. This is
        // the bit a consumer reads to know whether `OffloadPolicy::Sequential` bounds the peak here
        // or is a silent no-op; `false` would tell the fit-gate to size this render at the SUM of
        // the components. It moves in lockstep with `OFFLOAD_POLICY` and the staging tripwire.
        assert!(d.capabilities.supports_sequential_offload);
        assert_eq!(OFFLOAD_POLICY, OffloadPolicy::Sequential);
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 32 }
        );
    }

    /// Keyframe anchors come from `frame_idx` and only the two ends exist. An interior index is a
    /// typed error naming the field, not a keyframe quietly moved to frame 0.
    #[test]
    fn keyframe_anchors_admit_only_the_two_ends() {
        let img = Image {
            width: 32,
            height: 32,
            pixels: vec![0; 32 * 32 * 3],
        };
        let kf = |idx: i32| candle_gen::gen_core::KeyframeRef {
            image: &img,
            frame_idx: idx,
            strength: 1.0,
        };
        assert_eq!(keyframe_anchors(&[], 124).unwrap(), Vec::new());
        assert_eq!(
            keyframe_anchors(&[kf(0)], 124).unwrap(),
            vec![KeyframeAnchor::First]
        );
        // A last-frame-only request is a payload shape, not a mode of its own.
        assert_eq!(
            keyframe_anchors(&[kf(123)], 124).unwrap(),
            vec![KeyframeAnchor::Last]
        );
        assert_eq!(
            keyframe_anchors(&[kf(0), kf(123)], 124).unwrap(),
            vec![KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        // An interior frame has no conditioning row to occupy.
        let e = keyframe_anchors(&[kf(60)], 124).unwrap_err().to_string();
        assert!(e.contains("two slots"), "{e}");
        // The same end twice is arity-legal and semantically impossible.
        let e = keyframe_anchors(&[kf(0), kf(0)], 124)
            .unwrap_err()
            .to_string();
        assert!(e.contains("both keyframes anchor"), "{e}");
    }

    /// sc-19571 — **refuse, do not ignore.** MiniMax-H3 anchors at the checkpoint's trained-in
    /// `KEYFRAME_NOISE_AUG_T`, so `Keyframe.strength` has nothing to weight here; a request that
    /// asks for a partial pin must be told so rather than rendered as a full one. This gate lives in
    /// `keyframe_anchors`, which BOTH `validate` and the render route through, so the two cannot
    /// disagree.
    ///
    /// Mutation guard: delete the `reject_keyframe_strength` call at the top of `keyframe_anchors`
    /// and the two `unwrap_err`s below panic while the full-pin assertions still pass.
    #[test]
    fn keyframe_anchors_refuse_a_conditioning_strength() {
        let img = Image {
            width: 32,
            height: 32,
            pixels: vec![0; 32 * 32 * 3],
        };
        let kf = |idx: i32, strength: f32| candle_gen::gen_core::KeyframeRef {
            image: &img,
            frame_idx: idx,
            strength,
        };
        // A full pin — the only thing the checkpoint expresses — is admitted.
        assert_eq!(
            keyframe_anchors(&[kf(0, 1.0), kf(123, 1.0)], 124).unwrap(),
            vec![KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        // Anything else is refused, on either slot, with the mechanism named.
        let e = keyframe_anchors(&[kf(0, 0.6)], 124)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("conditioning strength is not supported") && e.contains("0.999"),
            "{e}"
        );
        let e = keyframe_anchors(&[kf(0, 1.0), kf(123, 0.2)], 124)
            .unwrap_err()
            .to_string();
        assert!(e.contains("conditioning strength is not supported"), "{e}");
    }

    /// The step bound is a real gate, not a comment: every step is a full 33 B forward.
    #[test]
    fn the_step_bound_is_enforced() {
        assert_eq!(DEFAULT_STEPS, 50);
        assert_eq!(MAX_STEPS, 200);
        const { assert!(DEFAULT_STEPS < MAX_STEPS) };
    }
}

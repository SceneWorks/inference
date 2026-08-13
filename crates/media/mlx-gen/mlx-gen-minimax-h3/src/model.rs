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
    Progress, Quant, SizeFloor, WeightsSource,
};
use mlx_gen::weights::Weights;
use mlx_gen::{default_seed, Error, Result};
use mlx_gen_boogu::vision::VisionTower;

use crate::audio_config::{MiniMaxH3AudioVaeConfig, AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::audio_vae::MiniMaxH3AudioVae;
use crate::denoise::{JointSchedule, MIN_INFERENCE_STEPS};
use crate::dit::adaln::AdaLnResidency;
use crate::dit::model::{JointDit, MiniMaxH3Dit};
use crate::pipeline::{
    fit_audio_to_video, fl2va_layout, frames_to_images, initial_latents, prepend_condition_rows,
    render_latents, resolve_geometry, revert_pixel_normalization, t2va_layout, RequestGeometry,
    MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, SMALLEST_LEGAL_FRAMES, SPATIAL_STRIDE,
};
use crate::reference::{Ref2VaReference, Ref2VaReferences, VideoReference};
use crate::text_encoder::{
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, LM_PREFIX, VISION_PREFIX,
};
use crate::vae::MiniMaxH3VideoVae;

/// The published provider id.
pub const MODEL_ID: &str = "minimax_h3";

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

/// Text-encoder shards the layer-50 tap needs. Shards 13-14 hold only the never-executed tail
/// (layers 50-63 and `lm_head`), so mapping the whole 66.7 GB component would read 15 GB for
/// nothing.
const TE_SHARDS: std::ops::RangeInclusive<u32> = 1..=12;

/// The shard holding the Qwen3-VL **vision tower**. All 351 `model.visual.*` tensors live in
/// shard 14 and nowhere else, so [`TE_SHARDS`] excludes it for `t2va` and `fl2va` must add it back.
const VISION_SHARD: u32 = 14;

/// Quantization group size the shared vision tower is built with — the same value every other
/// consumer of `mlx-gen-boogu`'s tower passes.
const VISION_GROUP_SIZE: i32 = 64;

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "minimax_h3",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch exists anywhere in the checkpoint.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
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
            supports_lora: false,
            supports_lokr: false,
            // The tiers exist and load (sc-17150) — packed offline by `crate::convert` and staged
            // as the [`DIT_COMPONENT`]. `spec.quantize` is reconciled against the staged tier's
            // marker rather than triggering a load-time quantize; see [`reconcile_tier`].
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            samplers: Vec::new(),
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: SPATIAL_STRIDE,
            max_size: 1344,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_preview: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // The audio surface describes a *selectable* audio request (voice / language / rate),
            // which a video model has none of — the soundtrack rides `GenerationOutput::Video`.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
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
    dtype: Dtype,
}

/// The staged-component id for the **tiered** DiT directory (sc-17150).
///
/// Only the DiT is tiered: the text encoder, both VAEs and the tokenizer are dense in every tier and
/// are shared as a co-requisite, so they always resolve against the upstream root. Redirecting just
/// this one component is what lets a `q4` install hold one 18.8 GB DiT alongside the shared 66.7 GB
/// text encoder without a second copy of anything.
///
/// Deliberately **not** in [`ModelDescriptor::required_components`]: it is needed only for a
/// non-`bf16` tier, and a flat upstream snapshot must keep loading with nothing staged. That is the
/// sensenova `distill_lora` convention — a conditionally-needed component is declared to
/// [`reject_unknown_components`] per load, not advertised as a universal requirement.
pub const DIT_COMPONENT: &str = "transformer";

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
pub(crate) fn keyframe_anchors(
    keyframes: &[KeyframeRef<'_>],
    num_frames: i32,
) -> Result<Vec<crate::dit::positions::KeyframeAnchor>> {
    use crate::dit::positions::KeyframeAnchor;
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
    reject_unknown_components(spec, &[DIT_COMPONENT], MODEL_ID)?;
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: adapters are not supported"
        )));
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
    for (partition, probe) in [
        ("vae", "config.json"),
        ("audio_vae", "config.json"),
        ("text_encoder", "config.json"),
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
    Ok(Box::new(MiniMaxH3 {
        descriptor: descriptor(),
        root,
        dit_dir,
        dtype,
    }))
}

/// Release a component's device memory for real: drop the Rust handle, then drain MLX's allocator
/// cache so the buffers go back to the system rather than migrating active → cache.
fn release<T>(component: T) {
    drop(component);
    mlx_rs::memory::clear_cache();
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

    /// Map the text-encoder shards a presentation needs.
    ///
    /// `t2va` reads shards 1-12 ([`TE_SHARDS`]). `fl2va` additionally needs **shard 14**, which is
    /// where all 351 `model.visual.*` tensors live — the vision tower is not spread across the
    /// component, it sits entirely in the last shard alongside the never-executed decoder tail.
    /// That is why the `t2va` window could stop at 12 and why a keyframe request cannot.
    fn map_te_shards(&self, with_vision: bool) -> Result<Weights> {
        let dir = self.root.join("text_encoder");
        let mut w = Weights::empty();
        let shards: Vec<u32> = if with_vision {
            TE_SHARDS.chain(std::iter::once(VISION_SHARD)).collect()
        } else {
            TE_SHARDS.collect()
        };
        for i in shards {
            let shard = format!("model-{i:05}-of-00014.safetensors");
            let part = Weights::from_file(dir.join(&shard))?;
            let keys: Vec<String> = part.keys().map(str::to_owned).collect();
            for k in keys {
                // Shard 14 also holds layers 50-63 and `lm_head`, which are never executed. Keep
                // only what the tower needs so the fl2va path does not carry 15 GB for nothing.
                if with_vision && i == VISION_SHARD && !k.starts_with(VISION_PREFIX) {
                    continue;
                }
                let t = part.require(&k)?.clone();
                w.insert(k, t);
            }
        }
        Ok(w)
    }

    /// Encode the prompt and immediately release the 66.7 GB text encoder.
    fn encode_prompt(&self, prompt: &str) -> Result<mlx_rs::Array> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) = tok.encode_prompt(prompt)?;

        let w = self.map_te_shards(false)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)?;
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
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)?;
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
        let schedule = JointSchedule::new(evaluations + 1)?;
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
        if let Some(refs) = &references {
            return self.generate_ref2va(req, refs, task, &geometry, &schedule, seed, on_progress);
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

        let (context, text_tags) = if anchors.is_empty() {
            let context = self.encode_prompt(&req.prompt)?;
            let tags = vec![crate::denoise::TEXT_TAG; context.shape()[1] as usize];
            (context, tags)
        } else {
            let refs: Vec<&mlx_gen::media::Image> = fitted.iter().collect();
            self.encode_prompt_grounded(&req.prompt, &refs)?
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
                // Force before the VAE is dropped, for the same lazy-evaluation reason the context
                // is forced above.
                mlx_rs::transforms::eval([r])?;
            }
            release((vae, pixels));
            rows
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. denoise ---------------------------------------------------------------------
        let dit = MiniMaxH3Dit::load_dir(self.task_dit_dir(task), self.dtype)?;
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
        let mut model = JointDit::new(
            dit,
            layout.clone(),
            &context,
            adaln,
            AdaLnResidency::PrecomputeAndEvict,
        )?;
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
        let frames = self.decode_video(&rendered.video)?;
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
        let (context, text_tags) = self.encode_prompt_ref2va(&req.prompt, &normalized)?;
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
        let dit = MiniMaxH3Dit::load_dir(self.task_dit_dir(task), self.dtype)?;
        let patch = dit.config().patch_size;
        let layout = ref2va_layout(geometry, &text_tags, &geometries, patch)?;
        let (video_rows, audio_rows) = initial_latents(geometry, patch, seed)?;
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
        let frames = self.decode_video(&rendered.video)?;
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
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)?;
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

    fn decode_video(&self, latents: &mlx_rs::Array) -> Result<Vec<mlx_gen::media::Image>> {
        let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
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

mlx_gen::impl_generator!(MiniMaxH3 {
    validate: |s, req| validate_request(&s.descriptor.capabilities, req),
    generate: generate_impl,
});

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::CancelFlag;

    fn request(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            width,
            height,
            cancel: CancelFlag::default(),
            ..Default::default()
        }
    }

    fn generator_at(dit_dir: &str) -> MiniMaxH3 {
        MiniMaxH3 {
            descriptor: descriptor(),
            root: PathBuf::from("/snap"),
            dit_dir: PathBuf::from(dit_dir),
            dtype: Dtype::Bfloat16,
        }
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
            sample_rate: 24_000,
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

    /// The requested step count is model EVALUATIONS; the schedule adds the terminal σ = 0.
    #[test]
    fn the_requested_step_count_is_evaluations() {
        let s = JointSchedule::new(DEFAULT_STEPS as usize + 1).unwrap();
        assert_eq!(s.num_evals(), DEFAULT_STEPS as usize);
        assert_eq!(DEFAULT_STEPS, 50);
    }
}

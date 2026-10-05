//! The `Trainer` contract — LoRA/LoKr fine-tuning of a registered model (epic 3039), the training
//! analog of [`Generator`](crate::generator::Generator). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! SceneWorks owns the training *product* surface (datasets, plan normalization, validation, the
//! queue) in Rust; this is the *execution* surface that replaces the Python kernel for the
//! MLX-native families. The worker maps its normalized `TrainingPlan` onto a [`TrainingRequest`]
//! and calls [`Trainer::train`] — exactly as it maps `ImageRequest` → `GenerationRequest` and calls
//! `Generator::generate`. mlx-gen owns these shapes; it does not depend on the SceneWorks contract.
//!
//! The spike (sc-3042) proved the per-family training mechanism (functional autograd over an
//! external LoRA factor map, re-injected as `Adapter::Lora`, stepped with `keyed_value_and_grad` +
//! AdamW). This module is the family-agnostic glue around it: the config/progress/request shapes,
//! the [`schedule`] LR helpers, dataset bucketing, and checkpointing. Each family crate implements
//! [`Trainer`] (Z-Image in sc-3044) and publishes a [`crate::registry::TrainerRegistration`].

// The pure LR-schedule policy lives here (gen-core); the MLX training kernels
// (checkpoint/dataset/lora/optim, incl. `TrainOptimizer`) stay in mlx-gen's `train` module.
pub mod aux_schedule;
pub mod resume;
pub mod schedule;

use std::path::PathBuf;

pub use aux_schedule::{combine_step_terms, plan_step, AuxAlternation, StepPlan};
pub use schedule::LrSchedule;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::generator::Modality;
use crate::media::Image;
use crate::runtime::CancelFlag;

/// Adapter network parameterization (mirrors SceneWorks `network_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NetworkType {
    /// Standard low-rank `A·B` adapter.
    #[default]
    Lora,
    /// LyCORIS Kronecker-product adapter (LoKr); `decompose_factor` is the block-split knob.
    Lokr,
}

impl NetworkType {
    /// Parse the free-form contract string; unknown / empty → `Lora`.
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "lokr" => NetworkType::Lokr,
            _ => NetworkType::Lora,
        }
    }
}

/// Concrete training hyperparameters — the engine-side mirror of SceneWorks' `TrainingConfig`
/// (with the typed equivalents of the fields its Python kernel reads out of the plan's `advanced`
/// bag: LR schedule, warmup, timestep sampling, loss type, LoKr decompose factor, target modules).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingConfig {
    /// LoRA/LoKr rank (network dimension).
    pub rank: u32,
    /// LoRA alpha; the residual is scaled by `alpha/rank`.
    pub alpha: f32,
    pub learning_rate: f32,
    /// Total training micro-steps (forward/backward passes; an optimizer update fires every
    /// `gradient_accumulation` of them).
    pub steps: u32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    /// Gradient (activation) checkpointing: recompute each transformer block's activations during
    /// the backward pass instead of retaining them, bounding the first-step working set (sc-4874 —
    /// without it a production-resolution run can exceed unified memory and the OS hard-kills the
    /// worker). This is the engine-side home of the SceneWorks "Gradient Checkpointing" toggle (which
    /// was previously a no-op on the Rust path). Strictly opt-in — never auto-enabled; a run that
    /// would exceed the memory budget with it off is refused up front by the family trainer's
    /// pre-flight guard (a catchable error recommending this flag). Numerically it changes nothing
    /// (grads are bit-identical to the dense path); the cost is recompute time.
    pub gradient_checkpointing: bool,
    /// Training compute dtype for the model forward/backward: `"bf16"` (default — halves the
    /// activation working set; the ecosystem-standard mixed precision, sc-4887) or `"f32"` (full
    /// precision, the pre-sc-4887 behavior). The trainable adapter factors, loss, gradients, and
    /// optimizer state stay f32 either way (master-weights pattern); only the frozen base weights
    /// and the activation stream are cast. Unrecognized values mean f32.
    pub train_dtype: String,
    /// Square training resolution edge in pixels; bucketed down to a multiple of 32.
    pub resolution: u32,
    /// Adapter-checkpoint cadence, in micro-steps (`0` = no intermediate checkpoints).
    pub save_every: u32,
    pub seed: u64,
    /// Optimizer name, kept a free string to stay engine-agnostic (`adamw`/`adam`/`lion`/…); the
    /// family trainer maps it to an mlx-rs optimizer. Prodigy/Rose are not in mlx-rs (sc-3048).
    pub optimizer: String,
    pub weight_decay: f32,
    pub lr_scheduler: LrSchedule,
    pub lr_warmup_steps: u32,
    pub network_type: NetworkType,
    /// LoKr block-split factor (`-1` = auto). Ignored for plain LoRA.
    pub decompose_factor: i32,
    /// LoRA target module suffixes (e.g. `["to_q","to_k","to_v","to_out.0"]`); empty = the family
    /// default set.
    pub lora_target_modules: Vec<String>,
    /// ControlNet control type for a control-branch training run (`"pose"`/`"canny"`/`"depth"`/…);
    /// `None` for LoRA/LoKr training. Selects the branch's conditioning semantics and is recorded in
    /// the produced overlay's metadata so the model catalog / registration describes it correctly
    /// (rather than a hardcoded label). A control-branch trainer requires it set.
    pub control_type: Option<String>,
    /// **Full base-model fine-tune** selector (F-6, epic 14034 / sc-14056): when `true`, the family
    /// trainer updates *every* DiT weight rather than injecting a LoRA/LoKr adapter, producing a full
    /// fine-tuned checkpoint instead of an adapter. `false` (the default) is the LoRA/LoKr path, so it
    /// is additive — every existing `..Default::default()` caller and the whole conformance surface is
    /// unaffected (the same additive shape as [`control_type`](Self::control_type)). Only a trainer that
    /// advertises [`TrainerDescriptor::supports_full_finetune`] (Mage-Flow-Base today) may honor it; an
    /// adapter-only trainer must **reject** the request via the shared
    /// [`validate_full_finetune_request`] floor — it must never silently train a LoRA instead (F-006 /
    /// F-055). Because the dense retained-graph backward holds the whole model,
    /// a full fine-tune is memory-heavy: the platform/tier gate (SceneWorks) bounds it, and
    /// production-resolution runs additionally need gradient checkpointing (not yet ported, sc-14989).
    pub full_finetune: bool,
    /// Flow-match timestep sampling distribution (`sigmoid`/`linear`/`uniform`/…) — the *noise*
    /// schedule, distinct from `lr_scheduler`.
    pub timestep_type: String,
    /// Timestep sampling bias (`balanced`/`high_noise`/`low_noise`/…).
    pub timestep_bias: String,
    /// Loss type (`mse`/`mae`/…); families default to MSE on the velocity target.
    pub loss_type: String,
    /// Trigger word baked into captions / surfaced on the output adapter.
    pub trigger_word: Option<String>,
    /// Preview-sample cadence, in micro-steps (`0` = no preview samples). At each multiple the
    /// trainer renders preview images from the **in-progress adapter** (installed exactly as a train
    /// step installs it) so the user can watch the LoRA learn — the engine-side home of the
    /// SceneWorks "Sample cadence" control. The Python trainer did this; the native port dropped it
    /// (sc-5637). Samples are emitted via the [`TrainingProgress::Sample`] event on the existing
    /// `on_progress` callback — no extra trainer method.
    pub sample_every: u32,
    /// The prompts rendered at each [`sample_every`](Self::sample_every) cadence (the family trainer
    /// caps how many it renders per cadence — typically ≤4). Empty disables sampling regardless of
    /// `sample_every`. Encoded once during the dataset-caching pass (before families that free their
    /// text encoder post-cache do so), then reused for every cadence.
    pub sample_prompts: Vec<String>,
    /// Inference (denoise) steps per preview sample.
    pub sample_steps: u32,
    /// Guidance scale for preview samples. Guidance-distilled families (z-image-turbo,
    /// lens-turbo, …) ignore it / render at their fixed schedule; CFG families (sdxl, kolors) honor it.
    pub sample_guidance_scale: f32,
    /// Model-specific training options forwarded losslessly by the product layer.
    ///
    /// The common fields above remain the portable contract. Families with a genuinely richer
    /// strategy surface use this bag instead of overloading unrelated scalar fields. LTX-2.5, for
    /// example, carries its video/audio generation flags, intrinsic/reference conditions, and
    /// validation CFG/STG/modality-guidance recipe here. A trainer must parse and validate the keys
    /// it consumes and reject an unsupported requested mode; it must never silently collapse one
    /// workflow into another.
    pub model_options: JsonMap<String, JsonValue>,
    /// Mid-schedule **resume** (sc-9560 / F-125): when `true`, the family trainer looks in
    /// [`output_dir`](TrainingRequest::output_dir) for the latest resume snapshot written by a prior
    /// interrupted run of the **same** output adapter (`file_name`) at [`save_every`](Self::save_every)
    /// — its trainable factors, optimizer state, and step/update index — and continues from there
    /// instead of restarting at step 0. `false` (the default) always trains from scratch. Requires
    /// `save_every > 0` on the interrupted run to have produced a snapshot; a run whose target `steps`
    /// is already reached by the snapshot is a no-op. Resume is bit-exact when a snapshot lands on an
    /// optimizer-update boundary (always for `gradient_accumulation = 1`; for `> 1`, use a `save_every`
    /// that is a multiple of it) — the in-flight accumulation buffer is not snapshotted.
    pub resume: bool,
    /// **Weight noising** (epic 2123, sc-24826) — relative-mode perturbation of the trainable
    /// adapter factors, ported from ai-toolkit-perceptual. After every **real optimizer update**
    /// (not every gradient-accumulation micro-step) each adapter tensor `w` (LoRA A/B, LoKr
    /// factors — never a base weight) is permanently perturbed by
    /// `w += N(0, 1) · weight_noise_sigma · rms(w)`, with the noise drawn from an RNG derived from
    /// [`seed`](Self::seed) and the update index, so a seeded run is reproducible (and resume is
    /// bit-exact). The upstream suggested strength when enabled is `0.0125`.
    ///
    /// `0.0` (the default) is **off**: the trainer takes no extra RNG draws and produces exactly the
    /// adapter it did before this field existed. A non-zero sigma is refused (typed
    /// [`crate::Error::Unsupported`]) by any trainer whose
    /// [`TrainerDescriptor::techniques`] does not declare
    /// [`weight_noise`](TrainingTechniques::weight_noise), and by every trainer for a
    /// [`full_finetune`](Self::full_finetune) run — see [`validate_training_techniques`].
    pub weight_noise_sigma: f32,
    /// **Depth anchoring** (epic 2123, sc-2125) — an auxiliary perceptual loss that keeps the
    /// adapter's predicted geometry consistent with the training image: the model's x0 prediction
    /// is decoded with the family's small differentiable decoder, run through a frozen
    /// Depth-Anything-V2, and compared (scale-and-shift-invariant L1 + multi-scale gradient
    /// matching) against the depth of the training image's own encode→decode round trip, cached
    /// once per image. Off by default ([`AuxLossSchedule::weight`] `0`); refused (typed
    /// [`crate::Error::Unsupported`]) by any trainer whose [`TrainerDescriptor::techniques`] does
    /// not declare [`depth_anchoring`](TrainingTechniques::depth_anchoring) — see
    /// [`validate_training_techniques`].
    pub depth_anchoring: DepthAnchoringConfig,
    /// Directory holding the family's **small differentiable x0 decoder** (TAEF1 for the Flux-VAE
    /// 16-channel families, TAESD/TAESDXL for SD/SDXL, …) — the decoder every decoded-x0
    /// perceptual loss (depth anchoring, and the identity/body losses that reuse the same shared
    /// path) runs the model's x0 prediction through. Required (a typed refusal otherwise) whenever
    /// such a loss is enabled; ignored when none is. `None` by default.
    pub perceptual_decoder_dir: Option<PathBuf>,
}

/// Schedule of one **auxiliary perceptual loss** (epic 2123 E8: depth anchoring here; the
/// identity, landmark, body and latent losses reuse it): its weight, the noise-level window it
/// fires in, and how it alternates with the diffusion loss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxLossSchedule {
    /// Loss weight. `0` (the default) is **off** — the loss is never computed and the run trains
    /// exactly as it would without it.
    pub weight: f32,
    /// Inclusive lower bound of the noise-level window the loss fires in — the flow-match `σ`
    /// (or the normalized timestep `t / T` for ε/v-prediction families): `0` = clean, `1` = pure
    /// noise.
    pub t_min: f32,
    /// Inclusive upper bound of the noise-level window (see [`t_min`](Self::t_min)).
    pub t_max: f32,
    /// Alternation period, counted per image and per optimizer update (see
    /// [`aux_schedule::AuxAlternation`]):
    /// - `1` — the weighted aux loss is **added** to the diffusion loss on every in-window step;
    /// - `n ≥ 2` — every `n`-th update of each image is an **aux-only** update on which the
    ///   diffusion loss contributes **zero**; the others are diffusion-only. An aux-only update
    ///   samples its noise level inside `[t_min, t_max]`, so no step is wasted. `2` (the default)
    ///   is the upstream strict alternation.
    pub every_n: u32,
}

impl AuxLossSchedule {
    /// The off schedule: weight `0`, full window `[0, 1]`, strict alternation.
    pub const OFF: Self = Self {
        weight: 0.0,
        t_min: 0.0,
        t_max: 1.0,
        every_n: 2,
    };

    /// Whether the loss is turned on (`weight > 0`).
    pub fn is_enabled(&self) -> bool {
        self.weight > 0.0
    }

    /// Whether noise level `t` lies inside the inclusive window.
    pub fn in_window(&self, t: f32) -> bool {
        t >= self.t_min && t <= self.t_max
    }

    /// Reject a malformed schedule: a non-finite or negative weight, a window outside `[0, 1]` or
    /// with `t_min > t_max`, or `every_n == 0`. `name` labels the error.
    pub fn validate(&self, name: &str) -> Result<(), String> {
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(format!(
                "{name} weight must be a finite value >= 0, got {}",
                self.weight
            ));
        }
        let unit = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
        if !unit(self.t_min) || !unit(self.t_max) || self.t_min > self.t_max {
            return Err(format!(
                "{name} timestep window [{}, {}] must satisfy 0 <= t_min <= t_max <= 1",
                self.t_min, self.t_max
            ));
        }
        if self.every_n == 0 {
            return Err(format!("{name} alternation period every_n must be >= 1"));
        }
        Ok(())
    }
}

impl Default for AuxLossSchedule {
    fn default() -> Self {
        Self::OFF
    }
}

/// Which Depth-Anything-V2 checkpoint depth anchoring runs (all three share one module graph).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DepthModelSize {
    /// ViT-S/14 (~25M params) — the default and the upstream-calibrated choice.
    #[default]
    Small,
    /// ViT-B/14 (~98M params).
    Base,
    /// ViT-L/14 (~335M params) — much larger gradients; upstream suggests a far smaller weight.
    Large,
}

impl DepthModelSize {
    /// Parse the contract string (`small`/`base`/`large`, case-insensitive); `None` otherwise.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "small" => Some(Self::Small),
            "base" => Some(Self::Base),
            "large" => Some(Self::Large),
            _ => None,
        }
    }

    /// The contract string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Base => "base",
            Self::Large => "large",
        }
    }
}

/// [`TrainingConfig::depth_anchoring`] — the depth-anchoring loss schedule plus the frozen
/// Depth-Anything-V2 checkpoint it runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DepthAnchoringConfig {
    /// Weight / window / alternation. Off by default.
    pub schedule: AuxLossSchedule,
    /// Which DA2 checkpoint [`model_dir`](Self::model_dir) holds.
    pub model_size: DepthModelSize,
    /// Directory holding the DA2 `*-hf` checkpoint (`model.safetensors`). Required (a typed
    /// refusal otherwise) when the loss is enabled.
    pub model_dir: Option<PathBuf>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-4,
            steps: 1000,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: false,
            train_dtype: "bf16".to_string(),
            resolution: 1024,
            save_every: 250,
            seed: 0,
            optimizer: "adamw".to_string(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::Constant,
            lr_warmup_steps: 0,
            network_type: NetworkType::Lora,
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            control_type: None,
            // Full base fine-tune is OFF by default (F-6): a caller that does not opt in trains a
            // LoRA/LoKr adapter exactly as before. The worker sets it from the plan for a full-tune run.
            full_finetune: false,
            timestep_type: "sigmoid".to_string(),
            timestep_bias: "balanced".to_string(),
            loss_type: "mse".to_string(),
            trigger_word: None,
            // Preview sampling is OFF by default: a caller that does not opt in (every conformance
            // profile, every existing test that builds via `..Default::default()`) trains exactly as
            // before. The worker sets these explicitly from the plan's `advanced` bag (sc-5637).
            sample_every: 0,
            sample_prompts: Vec::new(),
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            model_options: JsonMap::new(),
            // Resume is OFF by default (F-125): a caller that does not opt in trains from scratch,
            // exactly as before. The worker sets it from the plan when re-running an interrupted job.
            resume: false,
            // Weight noising is OFF by default (epic 2123 E1): a caller that does not opt in trains
            // exactly as before.
            weight_noise_sigma: 0.0,
            // Depth anchoring is OFF by default (epic 2123 E1): weight 0, no aux model loaded.
            depth_anchoring: DepthAnchoringConfig::default(),
            perceptual_decoder_dir: None,
        }
    }
}

/// One training example. Paths are resolved by the caller (the worker resolves the dataset's
/// absolute image paths).
///
/// Three shapes share this one struct (all additive — every existing trainer's behaviour is
/// unchanged by the later ones):
///
/// * **captioned** ([`captioned`](Self::captioned)) — `image_path` + `caption`, the LoRA/LoKr
///   text-to-image case;
/// * **control pair** ([`with_control`](Self::with_control)) — plus a `control_image_path`, the
///   ControlNet case;
/// * **edit pair** ([`edit_pair`](Self::edit_pair), sc-24161) — `image_path` is the **target**
///   (the edited result), `caption` is the **edit instruction**, and
///   [`reference_image_paths`](Self::reference_image_paths) is the **ordered** list of source /
///   reference images the instruction refers to ("the first image", "the second image", …).
///   Order is semantic and is preserved end to end. Only a trainer whose descriptor advertises
///   [`TrainerDescriptor::max_reference_images`] `> 0` may train on edit pairs; every other trainer
///   refuses them through the shared [`validate_edit_request`] floor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingItem {
    /// The image the model learns to produce — for an edit pair, the edit's **target**.
    pub image_path: PathBuf,
    /// The caption — for an edit pair, the **edit instruction**.
    pub caption: String,
    /// Optional per-item control-conditioning image (the ControlNet case): a rendered condition
    /// aligned to `image_path` — a pose skeleton, canny edge map, depth map, … `None` is the LoRA
    /// case, which every existing trainer ignores. A control-branch trainer requires it present on
    /// every item (its `validate` rejects the request otherwise).
    pub control_image_path: Option<PathBuf>,
    /// Model-specific per-example inputs forwarded losslessly by the product layer.
    ///
    /// Most image trainers leave this empty. Rich multimodal trainers use it for resolved sidecar
    /// paths and shape metadata that cannot be represented by the legacy image/control-image pair.
    /// LTX-2.5, for example, consumes preprocessed video/audio latents plus optional intrinsic
    /// masks and reference latents from this bag. A family must validate every field it consumes
    /// and reject missing inputs for the selected workflow; it must never substitute the image
    /// path or silently fall back to another workflow.
    pub model_options: JsonMap<String, JsonValue>,
    /// The **ordered** reference images of an instruction-edit pair (sc-24161): the source image(s)
    /// the `caption` instruction edits or composes, in the order the instruction names them. Empty
    /// for every captioned / control item (the default), which every trainer treats exactly as
    /// before. Non-empty makes the item an edit pair, which only an edit-capable trainer
    /// ([`TrainerDescriptor::max_reference_images`] `> 0`) accepts — and then only up to that cap;
    /// see [`validate_edit_request`].
    pub reference_image_paths: Vec<PathBuf>,
}

impl TrainingItem {
    /// A captioned item with no control conditioning (the LoRA case) — the common constructor that
    /// keeps callers insulated from the optional `control_image_path` field.
    pub fn captioned(image_path: PathBuf, caption: String) -> Self {
        Self {
            image_path,
            caption,
            control_image_path: None,
            model_options: JsonMap::new(),
            reference_image_paths: Vec::new(),
        }
    }

    /// A captioned item paired with a control-conditioning image (the ControlNet case).
    pub fn with_control(image_path: PathBuf, caption: String, control_image_path: PathBuf) -> Self {
        Self {
            image_path,
            caption,
            control_image_path: Some(control_image_path),
            model_options: JsonMap::new(),
            reference_image_paths: Vec::new(),
        }
    }

    /// An instruction-edit pair (sc-24161): the `target_image_path` the model learns to produce
    /// from the **ordered** `reference_image_paths` under the edit `instruction`. The order of
    /// `reference_image_paths` is kept exactly as given.
    pub fn edit_pair(
        target_image_path: PathBuf,
        instruction: String,
        reference_image_paths: Vec<PathBuf>,
    ) -> Self {
        Self {
            image_path: target_image_path,
            caption: instruction,
            control_image_path: None,
            model_options: JsonMap::new(),
            reference_image_paths,
        }
    }

    /// `true` when this item is an instruction-edit pair (it carries at least one reference).
    pub fn is_edit_pair(&self) -> bool {
        !self.reference_image_paths.is_empty()
    }
}

/// A training run: the dataset, the hyperparameters, and where to write the adapter. The base model
/// is supplied at `load`-time via the [`LoadSpec`](crate::runtime::LoadSpec) (mirroring inference:
/// `load(id, spec)` then `generate(req)`), so it is not repeated here.
#[derive(Clone, Debug)]
pub struct TrainingRequest {
    pub items: Vec<TrainingItem>,
    pub config: TrainingConfig,
    /// Absolute directory the adapter (and any intermediate checkpoints) are written into.
    pub output_dir: PathBuf,
    /// Output adapter file name, e.g. `my_style.safetensors`.
    pub file_name: String,
    /// Trigger words surfaced on the produced adapter.
    pub trigger_words: Vec<String>,
    /// Cooperative cancellation, polled between steps (mirrors `GenerationRequest`).
    pub cancel: CancelFlag,
}

/// A progress event streamed during a long [`Trainer::train`] — the training analog of
/// [`Progress`](crate::runtime::Progress), with bands matching the kernel's
/// prepare→load→cache→train→checkpoint→save lifecycle.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingProgress {
    /// Resolving the dataset / building buckets.
    Preparing,
    /// Loading the (frozen) base model weights.
    LoadingModel,
    /// Encoding + caching VAE latents and prompt embeddings: item `current` of `total` (1-based).
    Caching { current: u32, total: u32 },
    /// Optimizer micro-step `step` of `total` (1-based) with the latest scalar `loss`.
    Training { step: u32, total: u32, loss: f32 },
    /// An intermediate adapter checkpoint was written at micro-step `step`.
    Checkpoint { step: u32 },
    /// A preview sample image, rendered from the in-progress adapter at micro-step `step` (sc-5637).
    /// `index`/`total` are the 1-based position within this cadence's prompt set; `prompt` is the
    /// rendered prompt; `image` is the decoded 8-bit RGB bitmap the consumer persists/streams (e.g.
    /// the worker writes it as a project asset and appends it to the job result the Training Studio
    /// renders). Emitted only when `config.sample_every > 0` and `config.sample_prompts` is non-empty.
    /// Consumers that don't care can ignore it — it is interleaved with `Training` and does not
    /// affect step/loss accounting.
    Sample {
        step: u32,
        index: u32,
        total: u32,
        prompt: String,
        image: Image,
    },
    /// Writing the final adapter.
    Saving,
}

/// What a [`Trainer::train`] produced.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingOutput {
    /// Absolute path to the final adapter safetensors.
    pub adapter_path: PathBuf,
    /// Micro-steps actually run (may be < `config.steps` if cancelled).
    pub steps: u32,
    /// The last training loss observed.
    pub final_loss: f32,
}

/// Identity + capabilities of a trainer (drives `validate` and consumer introspection). The
/// training analog of [`ModelDescriptor`](crate::generator::ModelDescriptor).
#[derive(Clone, Copy, Debug)]
pub struct TrainerDescriptor {
    /// Registry id, e.g. `"z_image_turbo"` (matches the generator id of the same base model).
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this trainer ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub modality: Modality,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    /// Whether this trainer can train a ControlNet-style **control branch** (the sc-10163
    /// `control_type` / per-item `control_image_path` contract), as opposed to a plain LoRA/LoKr
    /// adapter. The worker reads this to know whether a control-training plan is serviceable, and the
    /// shared [`validate_control_request`] floor uses it to decide whether a control request is a
    /// capability gap (reject typed) or must be checked for per-item control images. `false` for every
    /// LoRA/LoKr-only trainer shipped today — they must *reject* a control request, not silently train
    /// a plain adapter (F-006 / F-055).
    pub supports_control: bool,
    /// Whether this trainer can run a **full base fine-tune** (the sc-14056 / epic-14034
    /// [`TrainingConfig::full_finetune`] contract) — updating *every* base weight and emitting a full
    /// fine-tuned checkpoint — as opposed to injecting a LoRA/LoKr adapter. The shared
    /// [`validate_full_finetune_request`] floor uses it to decide whether a `full_finetune` request is
    /// a capability gap (reject typed). `false` for every adapter-only trainer shipped today — they
    /// must *reject* a full-fine-tune request, not silently train a LoRA adapter and hand back
    /// something the caller did not ask for (F-006 / F-055, the same class the control floor guards).
    pub supports_full_finetune: bool,
    /// The most **ordered reference images** one instruction-edit training item may carry
    /// (sc-24161) — the model's own reference cap, read from the same fact its render path
    /// enforces (Qwen-Image 2.1: 10), never a second hard-coded number. `0` means the trainer
    /// cannot train on edit pairs at all: the shared [`validate_edit_request`] floor then refuses
    /// any request carrying [`TrainingItem::reference_image_paths`] with a typed
    /// [`crate::Error::Unsupported`] instead of silently training a text-to-image adapter on the
    /// targets (the F-055 class). `0` for every trainer shipped before sc-24161.
    pub max_reference_images: u32,
    /// Which optional **training techniques** (epic 2123) this trainer actually implements. The
    /// shared [`validate_training_techniques`] floor refuses a request that turns on a technique
    /// the trainer does not declare — a typed [`crate::Error::Unsupported`] before any work, never a
    /// silently ignored knob. [`TrainingTechniques::NONE`] for a trainer that implements none.
    pub techniques: TrainingTechniques,
}

/// Per-technique support flags for the optional training techniques of epic 2123 (weight noising
/// and depth anchoring today; gradient noise, aspect buckets, masked loss and the perceptual
/// identity/body/latent losses join here as their stories land). Each flag gates exactly one
/// [`TrainingConfig`] knob through [`validate_training_techniques`].
///
/// Non-supporting descriptors spell [`TrainingTechniques::NONE`], so a new flag defaults to
/// *unsupported* everywhere without touching them; a supporting descriptor names the flags it
/// implements (and, once more than one flag exists, completes the rest with `..NONE`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingTechniques {
    /// Honors [`TrainingConfig::weight_noise_sigma`] (relative-mode adapter weight noising).
    pub weight_noise: bool,
    /// Honors [`TrainingConfig::depth_anchoring`] (decoded-x0 Depth-Anything-V2 anchoring loss).
    pub depth_anchoring: bool,
}

impl TrainingTechniques {
    /// No optional technique supported — every technique knob must stay at its off value.
    pub const NONE: Self = Self {
        weight_noise: false,
        depth_anchoring: false,
    };
}

/// The shared **training-technique floor** (epic 2123 E3/E5) — every trainer's `validate` *and*
/// `train` entry point calls it before any expensive work, so a requested technique the trainer
/// does not implement is refused instead of silently ignored.
///
/// - `weight_noise_sigma` not finite or negative ⇒ [`crate::Error::Msg`] (malformed request).
/// - `weight_noise_sigma > 0` on a trainer whose [`TrainerDescriptor::techniques`] lacks
///   [`weight_noise`](TrainingTechniques::weight_noise) ⇒ typed [`crate::Error::Unsupported`].
/// - `weight_noise_sigma > 0` with [`TrainingConfig::full_finetune`] ⇒ typed
///   [`crate::Error::Unsupported`]: weight noise perturbs adapter factors only and must never touch
///   base weights (E5).
/// - a malformed [`TrainingConfig::depth_anchoring`] schedule ⇒ [`crate::Error::Msg`].
/// - depth anchoring enabled on a trainer whose descriptor lacks
///   [`depth_anchoring`](TrainingTechniques::depth_anchoring) ⇒ typed [`crate::Error::Unsupported`].
/// - depth anchoring enabled without a [`DepthAnchoringConfig::model_dir`] or a
///   [`TrainingConfig::perceptual_decoder_dir`] ⇒ [`crate::Error::Msg`] naming the missing model.
/// - every technique off ⇒ no-op.
pub fn validate_training_techniques(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> crate::Result<()> {
    let sigma = req.config.weight_noise_sigma;
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(crate::Error::Msg(format!(
            "{}: weight_noise_sigma must be a finite value >= 0, got {sigma}",
            desc.id
        )));
    }
    if sigma > 0.0 {
        if req.config.full_finetune {
            return Err(crate::Error::Unsupported(format!(
                "{}: weight noising (weight_noise_sigma {sigma}) perturbs adapter factors only and \
                 cannot be combined with a full base fine-tune",
                desc.id
            )));
        }
        if !desc.techniques.weight_noise {
            return Err(crate::Error::Unsupported(format!(
                "{}: weight noising (weight_noise_sigma {sigma}) is not supported by this trainer",
                desc.id
            )));
        }
    }
    let depth = &req.config.depth_anchoring;
    depth
        .schedule
        .validate("depth anchoring")
        .map_err(|m| crate::Error::Msg(format!("{}: {m}", desc.id)))?;
    if depth.schedule.is_enabled() {
        if !desc.techniques.depth_anchoring {
            return Err(crate::Error::Unsupported(format!(
                "{}: depth anchoring (weight {}) is not supported by this trainer",
                desc.id, depth.schedule.weight
            )));
        }
        if depth.model_dir.is_none() {
            return Err(crate::Error::Msg(format!(
                "{}: depth anchoring needs the Depth-Anything-V2 {} checkpoint \
                 (depth_anchoring.model_dir is unset)",
                desc.id,
                depth.model_size.as_str()
            )));
        }
        if req.config.perceptual_decoder_dir.is_none() {
            return Err(crate::Error::Msg(format!(
                "{}: depth anchoring needs the family's small x0 decoder \
                 (perceptual_decoder_dir is unset)",
                desc.id
            )));
        }
    }
    Ok(())
}

/// The shared control-training validation floor (F-006) — the training analog of
/// [`Capabilities::validate_request`](crate::generator::Capabilities::validate_request). A family
/// trainer's `validate` calls this so the "control_type set ⇒ every item carries a control image"
/// invariant (and the "unsupported ⇒ typed reject" rule) lives in one place instead of being
/// re-implemented per family — the recurring fixes-don't-travel gap sc-10163 opened.
///
/// - `control_type` unset ⇒ the plain LoRA/LoKr path: no-op.
/// - `control_type` set on a trainer that does **not** advertise
///   [`supports_control`](TrainerDescriptor::supports_control) ⇒ typed
///   [`crate::Error::Unsupported`] (the
///   LoRA-only trainers must reject a control request, not silently produce a non-control adapter).
/// - `control_type` set on a control-capable trainer ⇒ every [`TrainingItem`] must carry a
///   `control_image_path`, else [`crate::Error::Msg`].
pub fn validate_control_request(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> crate::Result<()> {
    let Some(control_type) = req.config.control_type.as_deref() else {
        return Ok(());
    };
    if !desc.supports_control {
        return Err(crate::Error::Unsupported(format!(
            "{}: control-branch training (control_type {control_type:?}) is not supported by this \
             trainer",
            desc.id
        )));
    }
    if let Some(idx) = req
        .items
        .iter()
        .position(|it| it.control_image_path.is_none())
    {
        return Err(crate::Error::Msg(format!(
            "{}: control_type {control_type:?} requires a control image on every training item \
             (item {idx} has none)",
            desc.id
        )));
    }
    Ok(())
}

/// The shared **full-base-fine-tune** validation floor (F-006, sc-14056) — the exact analog of
/// [`validate_control_request`] for the [`TrainingConfig::full_finetune`] selector. A family
/// trainer's `validate` calls it so the "unsupported ⇒ typed reject" rule lives in one place instead
/// of each adapter-only trainer independently deciding to ignore the flag.
///
/// - `full_finetune == false` ⇒ the plain LoRA/LoKr path: no-op.
/// - `full_finetune == true` on a trainer that does **not** advertise
///   [`supports_full_finetune`](TrainerDescriptor::supports_full_finetune) ⇒ typed
///   [`crate::Error::Unsupported`]. Silently training an adapter instead would hand the caller a
///   ~100 MB LoRA where they asked for a fine-tuned base checkpoint, and the consumer's memory gate,
///   output registration, and UI would all describe the wrong artifact — the F-055 silent-adapter
///   class, which this repo already refuses for `control_type`.
/// - `full_finetune == true` on a full-tune-capable trainer ⇒ ok.
pub fn validate_full_finetune_request(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> crate::Result<()> {
    if req.config.full_finetune && !desc.supports_full_finetune {
        return Err(crate::Error::Unsupported(format!(
            "{}: full base fine-tune (full_finetune) is not supported by this trainer — it can only \
             train a LoRA/LoKr adapter",
            desc.id
        )));
    }
    Ok(())
}

/// The shared **instruction-edit dataset** floor (sc-24161) — the analog of
/// [`validate_control_request`] for [`TrainingItem::reference_image_paths`]. Every family trainer's
/// `validate` calls it, so "a trainer that cannot use references refuses an edit dataset" and the
/// per-model reference cap live in one place.
///
/// - no item carries references ⇒ the plain captioned / control path: no-op.
/// - any item carries references on a trainer whose
///   [`max_reference_images`](TrainerDescriptor::max_reference_images) is `0` ⇒ typed
///   [`crate::Error::Unsupported`] — never a text-to-image adapter silently trained on the targets.
/// - an edit-capable trainer, but some item has **no** references (a mixed dataset) ⇒
///   [`crate::Error::Msg`] naming the item: an edit run conditions every step on its references, so
///   a reference-less item has no defined training input.
/// - an item also carries a `control_image_path` ⇒ [`crate::Error::Msg`]: edit pairs and control
///   pairs are different workflows.
/// - an item carries more references than the cap ⇒ [`crate::Error::Msg`] naming the item, its
///   count and the cap.
/// - an edit pair with an empty (whitespace-only) instruction `caption` ⇒ [`crate::Error::Msg`]: the
///   instruction is the edit's whole conditioning.
///
/// It also carries the **item-shape floor** every trainer gets for free (and the reason
/// [`TrainingItem`] can derive `Default` safely): any item whose `image_path` — or any of whose
/// `reference_image_paths` — is empty is refused with [`crate::Error::Msg`] naming the item, so a
/// `..Default::default()` literal that forgot a path fails `validate` instead of the run.
pub fn validate_edit_request(desc: &TrainerDescriptor, req: &TrainingRequest) -> crate::Result<()> {
    for (idx, item) in req.items.iter().enumerate() {
        if item.image_path.as_os_str().is_empty() {
            return Err(crate::Error::Msg(format!(
                "{}: item {idx} has an empty image_path",
                desc.id
            )));
        }
        if let Some(r) = item
            .reference_image_paths
            .iter()
            .position(|p| p.as_os_str().is_empty())
        {
            return Err(crate::Error::Msg(format!(
                "{}: item {idx} has an empty path at reference {r}",
                desc.id
            )));
        }
    }
    let Some(first_edit) = req.items.iter().position(TrainingItem::is_edit_pair) else {
        return Ok(());
    };
    let cap = desc.max_reference_images as usize;
    if cap == 0 {
        return Err(crate::Error::Unsupported(format!(
            "{}: instruction-edit training (item {first_edit} carries ordered reference images) is \
             not supported by this trainer — it trains on captioned images only, and would \
             otherwise silently learn text-to-image from the edit targets",
            desc.id
        )));
    }
    for (idx, item) in req.items.iter().enumerate() {
        let count = item.reference_image_paths.len();
        if count == 0 {
            return Err(crate::Error::Msg(format!(
                "{}: an instruction-edit dataset needs at least one reference image on every item \
                 (item {idx} has none); split captioned and edit items into separate runs",
                desc.id
            )));
        }
        if item.control_image_path.is_some() {
            return Err(crate::Error::Msg(format!(
                "{}: item {idx} carries both reference images (edit pair) and a control image \
                 (control pair); an item is one or the other",
                desc.id
            )));
        }
        if count > cap {
            return Err(crate::Error::Msg(format!(
                "{}: item {idx} carries {count} reference images, but this model accepts at most \
                 {cap} reference images per edit",
                desc.id
            )));
        }
        if item.caption.trim().is_empty() {
            return Err(crate::Error::Msg(format!(
                "{}: item {idx} is an edit pair with an empty instruction caption",
                desc.id
            )));
        }
    }
    Ok(())
}

/// `model_options` keys naming a workflow an image LoRA/LoKr trainer does not read from
/// `model_options`: ordered reference images (instruction-edit training travels as
/// `TrainingItem::reference_image_paths`, under the shared edit floor) and control conditioning.
/// Every other key — the worker's `mixedPrecision`, `cacheLatents`, `networkType`, `sampleEvery`, … —
/// is not the trainer's to read and is ignored.
pub const REFERENCE_CONTROL_MODEL_OPTIONS: [&str; 8] = [
    "references",
    "referenceImages",
    "reference_images",
    "referenceImagePaths",
    "controlType",
    "control_type",
    "controlImage",
    "control_image",
];

/// Whether a [`REFERENCE_CONTROL_MODEL_OPTIONS`] value actually selects a workflow. The worker's
/// `advanced` map carries these keys in their *off* state too — `null`, `[]`, `""` (or whitespace),
/// `{}`, `false`, `"none"` — and those select nothing.
pub fn model_option_selects_something(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null | JsonValue::Bool(false) => false,
        JsonValue::String(s) => {
            let s = s.trim();
            !s.is_empty() && !s.eq_ignore_ascii_case("none")
        }
        JsonValue::Array(items) => !items.is_empty(),
        JsonValue::Object(map) => !map.is_empty(),
        JsonValue::Bool(true) | JsonValue::Number(_) => true,
    }
}

/// The first [`REFERENCE_CONTROL_MODEL_OPTIONS`] key that selects something, on the config's
/// `model_options` first and then on each item's.
pub fn selected_reference_control_model_option(req: &TrainingRequest) -> Option<&'static str> {
    let refused = |options: &JsonMap<String, JsonValue>| {
        REFERENCE_CONTROL_MODEL_OPTIONS
            .iter()
            .find(|key| {
                options
                    .get(**key)
                    .is_some_and(model_option_selects_something)
            })
            .copied()
    };
    refused(&req.config.model_options).or_else(|| {
        req.items
            .iter()
            .find_map(|item| refused(&item.model_options))
    })
}

/// Refuse (as [`crate::Error::Msg`], `label`-prefixed) a request whose `model_options` select
/// reference / control conditioning ([`selected_reference_control_model_option`]) rather than
/// silently training without it.
pub fn refuse_reference_control_model_options(
    label: &str,
    req: &TrainingRequest,
) -> crate::Result<()> {
    match selected_reference_control_model_option(req) {
        None => Ok(()),
        Some(key) => Err(crate::Error::Msg(format!(
            "{label}: model_options `{key}` selects reference/control conditioning, which this \
             trainer does not read from model_options (instruction-edit references travel as \
             TrainingItem::reference_image_paths); refusing rather than silently training \
             without it"
        ))),
    }
}

/// The shared **load-overlay floor** for image LoRA/LoKr trainers (sc-24163): a control,
/// extra-control, IP-adapter or identity overlay on the trainer's
/// [`LoadSpec`](crate::runtime::LoadSpec) is not part of the trained adapter, so it is a typed
/// [`crate::Error::Unsupported`] rather than silently dropped. `label` names the trainer.
pub fn refuse_trainer_load_overlays(
    label: &str,
    spec: &crate::runtime::LoadSpec,
) -> crate::Result<()> {
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
    {
        return Err(crate::Error::Unsupported(format!(
            "{label}: control / IP-adapter / identity overlays are not part of text-to-image \
             LoRA/LoKr training"
        )));
    }
    Ok(())
}

/// A LoRA/LoKr trainer for one model family — the training analog of
/// [`Generator`](crate::generator::Generator). `train` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread) and takes `&mut self` because training mutates the
/// adapter parameters and optimizer state. The request carries a cancel flag and `on_progress`
/// streams the lifecycle.
pub trait Trainer {
    /// Identity + capabilities (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &TrainerDescriptor;

    /// Reject a request this trainer cannot serve (LoKr when unsupported, empty dataset,
    /// unresolvable target modules, …) before doing expensive work.
    fn validate(&self, req: &TrainingRequest) -> crate::Result<()>;

    /// Run training to completion (or until `req.cancel` trips), writing the adapter to
    /// `req.output_dir`.
    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> crate::Result<TrainingOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_item_ctors_set_control() {
        let img = PathBuf::from("a.png");
        let lora = TrainingItem::captioned(img.clone(), "a cat".into());
        assert_eq!(
            lora.control_image_path, None,
            "captioned = no control (LoRA)"
        );

        let ctrl =
            TrainingItem::with_control(img.clone(), "a cat".into(), PathBuf::from("a.pose.png"));
        assert_eq!(ctrl.control_image_path, Some(PathBuf::from("a.pose.png")));
        assert_eq!(ctrl.image_path, img);
    }

    #[test]
    fn config_default_has_no_control_type() {
        // Additive: LoRA callers building via `..Default::default()` get `control_type: None` and
        // are unaffected; only a control-branch trainer sets it.
        assert_eq!(TrainingConfig::default().control_type, None);
    }

    #[test]
    fn full_finetune_is_opt_in_and_the_flag_discriminates() {
        // Additive (sc-14056): every `..Default::default()` caller and the whole conformance surface
        // trains a LoRA/LoKr adapter; only a full-base-fine-tune run opts in.
        //
        // This deliberately does NOT just assert the default value — a bare
        // `assert!(!TrainingConfig::default().full_finetune)` is a constant that passes no matter what
        // the floor does with the field, and would still pass if `validate_full_finetune_request` were
        // deleted. Instead assert the *behavioral* difference the flag must make: on one and the same
        // adapter-only trainer, two requests differing ONLY by this field land on opposite sides of
        // the floor. Flip the default, or stop reading the field, and exactly one half breaks.
        let desc = trainer_desc(false);
        let items = vec![TrainingItem::captioned(
            PathBuf::from("a.png"),
            "a cat".into(),
        )];

        let adapter_req = train_req(None, items);
        assert!(
            !adapter_req.config.full_finetune,
            "the default request must be the adapter path"
        );
        assert!(
            validate_full_finetune_request(&desc, &adapter_req).is_ok(),
            "an adapter-only trainer must accept the default (non-full) request"
        );

        let mut full_req = adapter_req;
        full_req.config.full_finetune = true;
        assert!(
            validate_full_finetune_request(&desc, &full_req).is_err(),
            "flipping only `full_finetune` must change the outcome — otherwise the field is inert"
        );
    }

    #[test]
    fn validate_training_techniques_floor() {
        // sc-24826 (epic 2123 E3/E5): weight noising is refused unless the descriptor declares it,
        // refused with a full fine-tune, and malformed sigmas are refused outright.
        let items = vec![TrainingItem::captioned(
            PathBuf::from("a.png"),
            "a cat".into(),
        )];
        let plain = trainer_desc(false);
        let mut noisy_desc = trainer_desc_with(false, true);
        noisy_desc.techniques.weight_noise = true;

        // Off (the default) ⇒ no-op everywhere.
        let off = train_req(None, items);
        assert_eq!(off.config.weight_noise_sigma, 0.0);
        assert!(validate_training_techniques(&plain, &off).is_ok());
        assert!(validate_training_techniques(&noisy_desc, &off).is_ok());

        // On ⇒ typed Unsupported on a trainer that does not declare it; accepted where declared.
        let mut on = off.clone();
        on.config.weight_noise_sigma = 0.0125;
        let err = validate_training_techniques(&plain, &on).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m) if m.contains("weight noising")),
            "{err:?}"
        );
        assert!(validate_training_techniques(&noisy_desc, &on).is_ok());

        // On + full fine-tune ⇒ refused even where weight noise is declared (E5).
        let mut full = on.clone();
        full.config.full_finetune = true;
        let err = validate_training_techniques(&noisy_desc, &full).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m) if m.contains("full base fine-tune")),
            "{err:?}"
        );

        // Malformed ⇒ Msg, regardless of support.
        for bad in [-0.01f32, f32::NAN, f32::INFINITY] {
            let mut r = off.clone();
            r.config.weight_noise_sigma = bad;
            let err = validate_training_techniques(&noisy_desc, &r).unwrap_err();
            assert!(matches!(err, crate::Error::Msg(_)), "{bad}: {err:?}");
        }
    }

    #[test]
    fn validate_training_techniques_depth_anchoring_floor() {
        // sc-2125 (epic 2123 E3): depth anchoring is refused unless the descriptor declares it,
        // needs both aux models named, and a malformed schedule is refused outright.
        let items = vec![TrainingItem::captioned(
            PathBuf::from("a.png"),
            "a cat".into(),
        )];
        let plain = trainer_desc(false);
        let mut depth_desc = trainer_desc(false);
        depth_desc.techniques.depth_anchoring = true;

        // Off (the default) ⇒ no-op everywhere.
        let off = train_req(None, items);
        assert_eq!(off.config.depth_anchoring.schedule, AuxLossSchedule::OFF);
        assert!(!off.config.depth_anchoring.schedule.is_enabled());
        assert!(validate_training_techniques(&plain, &off).is_ok());

        let mut on = off.clone();
        on.config.depth_anchoring.schedule.weight = 0.1;
        on.config.depth_anchoring.model_dir = Some(PathBuf::from("/m/da2"));
        on.config.perceptual_decoder_dir = Some(PathBuf::from("/m/taef1"));
        let err = validate_training_techniques(&plain, &on).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m) if m.contains("depth anchoring")),
            "{err:?}"
        );
        assert!(validate_training_techniques(&depth_desc, &on).is_ok());

        // Missing aux models ⇒ a message naming the missing model.
        let mut no_da2 = on.clone();
        no_da2.config.depth_anchoring.model_dir = None;
        let err = validate_training_techniques(&depth_desc, &no_da2).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("Depth-Anything-V2")));
        let mut no_dec = on.clone();
        no_dec.config.perceptual_decoder_dir = None;
        let err = validate_training_techniques(&depth_desc, &no_dec).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("decoder")));

        // Malformed schedules ⇒ Msg, regardless of support.
        let bad = [
            AuxLossSchedule {
                weight: -0.1,
                ..AuxLossSchedule::OFF
            },
            AuxLossSchedule {
                weight: f32::NAN,
                ..AuxLossSchedule::OFF
            },
            AuxLossSchedule {
                t_min: 0.6,
                t_max: 0.4,
                ..AuxLossSchedule::OFF
            },
            AuxLossSchedule {
                t_max: 1.5,
                ..AuxLossSchedule::OFF
            },
            AuxLossSchedule {
                every_n: 0,
                ..AuxLossSchedule::OFF
            },
        ];
        for schedule in bad {
            let mut r = on.clone();
            r.config.depth_anchoring.schedule = AuxLossSchedule {
                weight: if schedule.weight == 0.0 {
                    0.1
                } else {
                    schedule.weight
                },
                ..schedule
            };
            let err = validate_training_techniques(&depth_desc, &r).unwrap_err();
            assert!(matches!(err, crate::Error::Msg(_)), "{schedule:?}: {err:?}");
        }
    }

    #[test]
    fn aux_loss_schedule_window_is_inclusive() {
        let s = AuxLossSchedule {
            weight: 1.0,
            t_min: 0.2,
            t_max: 0.8,
            every_n: 2,
        };
        assert!(s.in_window(0.2) && s.in_window(0.8) && s.in_window(0.5));
        assert!(!s.in_window(0.19) && !s.in_window(0.81));
    }

    #[test]
    fn depth_model_size_round_trips() {
        for size in [
            DepthModelSize::Small,
            DepthModelSize::Base,
            DepthModelSize::Large,
        ] {
            assert_eq!(DepthModelSize::parse(size.as_str()), Some(size));
        }
        assert_eq!(
            DepthModelSize::parse(" LARGE "),
            Some(DepthModelSize::Large)
        );
        assert_eq!(DepthModelSize::parse("giant"), None);
        assert_eq!(DepthModelSize::default(), DepthModelSize::Small);
    }

    fn trainer_desc(supports_control: bool) -> TrainerDescriptor {
        trainer_desc_with(supports_control, false)
    }

    fn trainer_desc_with(
        supports_control: bool,
        supports_full_finetune: bool,
    ) -> TrainerDescriptor {
        TrainerDescriptor {
            id: "fam_trainer",
            family: "fam",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control,
            supports_full_finetune,
            max_reference_images: 0,
            techniques: TrainingTechniques::NONE,
        }
    }

    fn train_req(control_type: Option<&str>, items: Vec<TrainingItem>) -> TrainingRequest {
        TrainingRequest {
            items,
            config: TrainingConfig {
                control_type: control_type.map(str::to_owned),
                ..Default::default()
            },
            output_dir: PathBuf::from("/out"),
            file_name: "a.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::default(),
        }
    }

    #[test]
    fn validate_control_request_floor() {
        // F-006: the shared control-training floor.
        let img = PathBuf::from("a.png");
        let ctrl = PathBuf::from("a.pose.png");

        // control_type unset ⇒ no-op on both trainer kinds.
        let lora_items = vec![TrainingItem::captioned(img.clone(), "a cat".into())];
        assert!(validate_control_request(
            &trainer_desc(false),
            &train_req(None, lora_items.clone())
        )
        .is_ok());

        // control_type set on a NON-control trainer ⇒ typed Unsupported (must reject, not silently
        // train a plain adapter — the F-055 class).
        let err = validate_control_request(
            &trainer_desc(false),
            &train_req(Some("pose"), lora_items.clone()),
        )
        .unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "control on a LoRA-only trainer is a capability gap → Unsupported, got {err:?}"
        );

        // control_type set on a control-capable trainer, but an item lacks a control image ⇒ Msg.
        let err =
            validate_control_request(&trainer_desc(true), &train_req(Some("pose"), lora_items))
                .unwrap_err();
        assert!(
            matches!(&err, crate::Error::Msg(_)) && err.to_string().contains("control image"),
            "got {err:?}"
        );

        // control_type set on a control-capable trainer with every item carrying a control image ⇒ ok.
        let ctrl_items = vec![TrainingItem::with_control(img, "a cat".into(), ctrl)];
        assert!(validate_control_request(
            &trainer_desc(true),
            &train_req(Some("pose"), ctrl_items)
        )
        .is_ok());
    }

    #[test]
    fn validate_full_finetune_request_floor() {
        // F-006 (sc-14056): the shared full-base-fine-tune floor, mirroring the control floor above.
        let items = vec![TrainingItem::captioned(
            PathBuf::from("a.png"),
            "a cat".into(),
        )];
        let full_req = |desc_items: Vec<TrainingItem>| {
            let mut req = train_req(None, desc_items);
            req.config.full_finetune = true;
            req
        };

        // full_finetune unset ⇒ no-op on BOTH trainer kinds (additive; the whole existing surface).
        for supports_full in [false, true] {
            assert!(
                validate_full_finetune_request(
                    &trainer_desc_with(false, supports_full),
                    &train_req(None, items.clone())
                )
                .is_ok(),
                "a non-full request must pass regardless of supports_full_finetune"
            );
        }

        // full_finetune set on a trainer that does NOT advertise it ⇒ typed Unsupported. It must
        // REJECT, not fall through and silently train a LoRA adapter — the F-055 class.
        let err = validate_full_finetune_request(
            &trainer_desc_with(false, false),
            &full_req(items.clone()),
        )
        .unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "a full fine-tune on an adapter-only trainer is a capability gap → Unsupported, got \
             {err:?}"
        );
        assert!(
            err.to_string().contains("full_finetune"),
            "the rejection must name the unsupported capability, got {err}"
        );

        // full_finetune set on a full-tune-capable trainer ⇒ ok. This is the half that proves the
        // floor reads the descriptor flag rather than rejecting every full request outright.
        assert!(
            validate_full_finetune_request(&trainer_desc_with(false, true), &full_req(items))
                .is_ok(),
            "a trainer advertising supports_full_finetune must be allowed through the floor"
        );
    }

    fn edit_desc(max_reference_images: u32) -> TrainerDescriptor {
        TrainerDescriptor {
            max_reference_images,
            ..trainer_desc(false)
        }
    }

    /// sc-24161: the edit-pair item shape keeps its references in the order given, is
    /// distinguishable from the captioned/control shapes, and the older constructors (and
    /// `Default`) carry no references — so every pre-existing caller is unaffected.
    #[test]
    fn edit_pair_keeps_its_references_in_order_and_older_shapes_carry_none() {
        let refs: Vec<PathBuf> = ["c.png", "a.png", "b.png"]
            .iter()
            .map(PathBuf::from)
            .collect();
        let item = TrainingItem::edit_pair(
            PathBuf::from("target.png"),
            "put the cat from the second image on the sofa of the first".into(),
            refs.clone(),
        );
        assert_eq!(item.image_path, PathBuf::from("target.png"));
        assert_eq!(
            item.caption,
            "put the cat from the second image on the sofa of the first"
        );
        assert_eq!(item.reference_image_paths, refs, "order is semantic");
        assert!(item.is_edit_pair());
        assert_eq!(item.control_image_path, None);

        // Round trip through a request (the shape the worker hands every trainer) and a clone:
        // the order survives untouched.
        let req = train_req(None, vec![item.clone()]);
        assert_eq!(req.clone().items[0].reference_image_paths, refs);
        assert_eq!(req.items[0], item);

        let captioned = TrainingItem::captioned(PathBuf::from("a.png"), "a cat".into());
        let control = TrainingItem::with_control(
            PathBuf::from("a.png"),
            "a cat".into(),
            PathBuf::from("a.pose.png"),
        );
        for older in [&captioned, &control, &TrainingItem::default()] {
            assert!(older.reference_image_paths.is_empty());
            assert!(!older.is_edit_pair());
        }
    }

    /// sc-24161: a trainer that cannot use references (`max_reference_images == 0`, every trainer
    /// shipped before the edit trainer) refuses an edit dataset with a typed `Unsupported` — never
    /// a silently trained text-to-image adapter — and passes a captioned dataset untouched.
    #[test]
    fn a_non_edit_trainer_refuses_an_edit_dataset() {
        let edit = vec![TrainingItem::edit_pair(
            PathBuf::from("t.png"),
            "make it blue".into(),
            vec![PathBuf::from("r.png")],
        )];
        let err = validate_edit_request(&edit_desc(0), &train_req(None, edit)).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "an edit dataset on a non-edit trainer is a capability gap → Unsupported, got {err:?}"
        );
        assert!(err.to_string().contains("instruction-edit"), "{err}");

        let captioned = vec![TrainingItem::captioned(
            PathBuf::from("a.png"),
            "a cat".into(),
        )];
        for cap in [0, 10] {
            assert!(
                validate_edit_request(&edit_desc(cap), &train_req(None, captioned.clone())).is_ok(),
                "a captioned dataset is the floor's no-op (cap {cap})"
            );
        }
    }

    /// sc-24161: the reference cap. Exactly `max_reference_images` passes; one more is refused
    /// with a message naming the cap and the offending item.
    #[test]
    fn the_reference_cap_is_enforced_and_named() {
        let refs = |n: usize| -> Vec<PathBuf> {
            (0..n).map(|i| PathBuf::from(format!("r{i}.png"))).collect()
        };
        let item =
            |n: usize| TrainingItem::edit_pair(PathBuf::from("t.png"), "compose".into(), refs(n));
        let desc = edit_desc(10);
        assert!(validate_edit_request(&desc, &train_req(None, vec![item(1)])).is_ok());
        assert!(validate_edit_request(&desc, &train_req(None, vec![item(10)])).is_ok());
        let err =
            validate_edit_request(&desc, &train_req(None, vec![item(1), item(11)])).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("at most 10"), "{msg}");
        assert!(msg.contains("item 1 carries 11"), "{msg}");
    }

    /// sc-24161: an edit run is all-edit — a reference-less item in it, or an item that is both an
    /// edit pair and a control pair, is refused by name.
    #[test]
    fn mixed_and_hybrid_edit_datasets_are_refused() {
        let edit = TrainingItem::edit_pair(
            PathBuf::from("t.png"),
            "make it blue".into(),
            vec![PathBuf::from("r.png")],
        );
        let plain = TrainingItem::captioned(PathBuf::from("a.png"), "a cat".into());
        let err =
            validate_edit_request(&edit_desc(10), &train_req(None, vec![edit.clone(), plain]))
                .unwrap_err()
                .to_string();
        assert!(err.contains("item 1 has none"), "{err}");

        let mut hybrid = edit;
        hybrid.control_image_path = Some(PathBuf::from("pose.png"));
        let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![hybrid]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("one or the other"), "{err}");
    }

    /// sc-24161: the item-shape floor that makes `TrainingItem: Default` safe — a forgotten
    /// (empty) `image_path` or reference path is refused by name on EVERY trainer (cap 0 or not),
    /// and an edit pair needs a non-empty instruction. A captioned item may keep an empty caption
    /// (trigger-word-only datasets), so that is NOT refused.
    #[test]
    fn empty_paths_and_empty_edit_instructions_are_refused() {
        for cap in [0, 10] {
            let forgotten = TrainingItem {
                caption: "a cat".into(),
                ..Default::default()
            };
            let err = validate_edit_request(&edit_desc(cap), &train_req(None, vec![forgotten]))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("item 0 has an empty image_path"),
                "cap {cap}: {err}"
            );
        }

        let empty_ref = TrainingItem::edit_pair(
            PathBuf::from("t.png"),
            "make it blue".into(),
            vec![PathBuf::from("r.png"), PathBuf::new()],
        );
        let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![empty_ref]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty path at reference 1"), "{err}");

        let silent = TrainingItem::edit_pair(
            PathBuf::from("t.png"),
            "  ".into(),
            vec![PathBuf::from("r.png")],
        );
        let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![silent]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty instruction"), "{err}");

        let trigger_only = TrainingItem::captioned(PathBuf::from("a.png"), String::new());
        assert!(
            validate_edit_request(&edit_desc(0), &train_req(None, vec![trigger_only])).is_ok(),
            "an empty caption on a captioned item stays legal"
        );
    }

    fn plain_request() -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem::captioned(
                PathBuf::from("a.png"),
                "a cat".into(),
            )],
            config: TrainingConfig::default(),
            output_dir: PathBuf::from("out"),
            file_name: "out.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    /// *Mutation that reds this:* `model_option_selects_something` treating every non-null value as
    /// present, or dropping a key from [`REFERENCE_CONTROL_MODEL_OPTIONS`].
    #[test]
    fn reference_control_model_options_refuse_on_values_and_ignore_off_values() {
        let mut req = plain_request();
        let off = serde_json::json!({
            "mixedPrecision": "bf16",
            "references": [],
            "referenceImages": "",
            "reference_images": "  ",
            "referenceImagePaths": {},
            "controlType": null,
            "control_type": "none",
            "controlImage": false,
            "control_image": "None",
        });
        req.config.model_options = off.as_object().unwrap().clone();
        req.items[0].model_options = off.as_object().unwrap().clone();
        refuse_reference_control_model_options("t", &req).unwrap();
        for key in REFERENCE_CONTROL_MODEL_OPTIONS {
            let mut on = req.clone();
            on.items[0]
                .model_options
                .insert(key.into(), serde_json::json!("x"));
            let err = refuse_reference_control_model_options("t", &on)
                .unwrap_err()
                .to_string();
            assert!(err.starts_with("t: model_options `"), "{err}");
            assert!(err.contains(&format!("`{key}`")), "{err}");
        }
    }

    /// *Mutation that reds this:* dropping any arm of the `||` in `refuse_trainer_load_overlays`.
    #[test]
    fn trainer_load_overlays_are_typed_refusals() {
        use crate::runtime::{IdentityWeights, LoadSpec, WeightsSource};

        let dir = || WeightsSource::Dir(PathBuf::from("/x"));
        let base = LoadSpec::new(dir());
        refuse_trainer_load_overlays("t", &base).unwrap();
        let mut identity = base.clone();
        identity.identity = Some(IdentityWeights::default());
        for spec in [
            base.clone().with_control(dir()),
            base.clone().with_extra_control(dir()),
            base.clone().with_ip_adapter(dir()),
            identity,
        ] {
            match refuse_trainer_load_overlays("t", &spec) {
                Err(crate::Error::Unsupported(message)) => assert_eq!(
                    message,
                    "t: control / IP-adapter / identity overlays are not part of text-to-image \
                     LoRA/LoKr training"
                ),
                other => panic!("an overlay must be a typed Unsupported, got {other:?}"),
            }
        }
    }
}

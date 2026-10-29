//! The training request, progress, output and descriptor shapes, and the `Trainer` trait.

use super::*;

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
///   refuses them through the shared `gen_core_train::validate_edit_request` floor.
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
    /// see `gen_core_train::validate_edit_request`.
    pub reference_image_paths: Vec<PathBuf>,
    /// The item's **subject mask** (epic 2123, sc-24828): a single-channel image the same size as
    /// [`image_path`](Self::image_path), white (255) = subject, black (0) = background, with soft
    /// edges in between. Read only when [`TrainingConfig::subject_mask_loss`] is on — then every
    /// item must carry one (an edit pair's mask covers its **target**); `None` otherwise.
    pub subject_mask_path: Option<PathBuf>,
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
            subject_mask_path: None,
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
            subject_mask_path: None,
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
            subject_mask_path: None,
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

/// What [`Trainer::train_with_report`] returns: the [`TrainingOutput`] plus the run's non-fatal
/// conditions (a preview sample that failed to render, …) in the order they occurred (sc-27451).
/// The training analog of [`GenerationReport`](crate::generator::GenerationReport).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingReport {
    pub output: TrainingOutput,
    /// Empty when there is nothing to report.
    pub warnings: Vec<crate::generator::GenerationWarning>,
}

/// [`TrainingReport::warnings`] code for a periodic preview sample
/// ([`TrainingProgress::Sample`]) that failed to render; the run continues without it.
pub const PREVIEW_SAMPLE_FAILED: &str = "preview_sample_failed";

/// The [`PREVIEW_SAMPLE_FAILED`] warning for prompt `index` (1-based) of the step-`step` preview.
pub fn preview_sample_failed(
    step: u32,
    index: u32,
    error: &dyn std::fmt::Display,
) -> crate::generator::GenerationWarning {
    crate::generator::GenerationWarning {
        code: PREVIEW_SAMPLE_FAILED.to_owned(),
        message: format!("preview sample {index} at step {step} failed to render: {error}"),
    }
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
    /// shared `gen_core_train::validate_control_request` floor uses it to decide whether a control request is a
    /// capability gap (reject typed) or must be checked for per-item control images. `false` for every
    /// LoRA/LoKr-only trainer shipped today — they must *reject* a control request, not silently train
    /// a plain adapter (F-006 / F-055).
    pub supports_control: bool,
    /// Whether this trainer can run a **full base fine-tune** (the sc-14056 / epic-14034
    /// [`TrainingConfig::full_finetune`] contract) — updating *every* base weight and emitting a full
    /// fine-tuned checkpoint — as opposed to injecting a LoRA/LoKr adapter. The shared
    /// `gen_core_train::validate_full_finetune_request` floor uses it to decide whether a `full_finetune` request is
    /// a capability gap (reject typed). `false` for every adapter-only trainer shipped today — they
    /// must *reject* a full-fine-tune request, not silently train a LoRA adapter and hand back
    /// something the caller did not ask for (F-006 / F-055, the same class the control floor guards).
    pub supports_full_finetune: bool,
    /// The most **ordered reference images** one instruction-edit training item may carry
    /// (sc-24161) — the model's own reference cap, read from the same fact its render path
    /// enforces (Qwen-Image 2.1: 10), never a second hard-coded number. `0` means the trainer
    /// cannot train on edit pairs at all: the shared `gen_core_train::validate_edit_request` floor then refuses
    /// any request carrying [`TrainingItem::reference_image_paths`] with a typed
    /// [`crate::Error::Unsupported`] instead of silently training a text-to-image adapter on the
    /// targets (the F-055 class). `0` for every trainer shipped before sc-24161.
    pub max_reference_images: u32,
    /// Which optional **training techniques** (epic 2123) this trainer actually implements. The
    /// shared `gen_core_train::validate_training_techniques` floor refuses a request that turns on a technique
    /// the trainer does not declare — a typed [`crate::Error::Unsupported`] before any work, never a
    /// silently ignored knob. [`TrainingTechniques::NONE`] for a trainer that implements none.
    pub techniques: TrainingTechniques,
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

    /// **Train and report** (sc-27451): the [`TrainingReport`] — the output plus the run's non-fatal
    /// warnings (e.g. a preview sample that failed to render), which [`train`](Self::train) has no
    /// channel for. Additive for the reason
    /// [`Generator::generate_with_report`](crate::generator::Generator::generate_with_report) is: a
    /// default-implemented method instead of a new [`TrainingProgress`] variant or
    /// [`TrainingOutput`] field, whose exhaustive matches and literals span consumers. The default
    /// wraps [`train`](Self::train) with no warnings; a trainer that reports warnings overrides this
    /// and derives `train` from it.
    fn train_with_report(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> crate::Result<TrainingReport> {
        self.train(req, on_progress).map(|output| TrainingReport {
            output,
            warnings: Vec::new(),
        })
    }
}

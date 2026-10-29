//! Training hyperparameters: `TrainingConfig`, auxiliary-loss configs and schedules, and the memory budget.

use super::*;

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
    /// **Gradient noise** (epic 2123, sc-24827) — annealed Gaussian noise on the trainable adapter
    /// gradients, the `neelakantan` mode of ai-toolkit-perceptual (Neelakantan et al. 2015). On
    /// every **real optimizer update** `t` (0-based update index, not the micro-step), after the
    /// window's gradients are averaged and globally norm-clipped and **before** the optimizer step
    /// (upstream's placement: "after clip so the clip doesn't eat the noise"), every adapter gradient
    /// `g` gets `g += N(0, 1) · σ_t` with `σ_t = gradient_noise_eta / (1 + t)^gradient_noise_gamma`
    /// (see [`gradient_noise_std`]). The noise is drawn from an RNG derived from
    /// [`seed`](Self::seed) and the update index on a stream independent of weight noising (E4).
    /// Upstream's suggested strength when enabled is `0.01`.
    ///
    /// `0.0` (the default) is **off**: no extra RNG draws, the adapter is exactly the pre-sc-24827
    /// one. A non-zero eta is refused (typed [`crate::Error::Unsupported`]) by any trainer whose
    /// [`TrainerDescriptor::techniques`] does not declare
    /// [`gradient_noise`](TrainingTechniques::gradient_noise), and for a
    /// [`full_finetune`](Self::full_finetune) run (it perturbs adapter gradients only) — see
    /// [`validate_training_techniques`].
    pub gradient_noise_eta: f32,
    /// Annealing exponent `γ` of [`gradient_noise_eta`](Self::gradient_noise_eta)'s schedule
    /// `σ_t = η / (1 + t)^γ`. Default `0.55` (the paper's / upstream's default). Must be finite and
    /// `>= 0` (a negative exponent would grow the noise without bound); ignored while eta is `0`.
    pub gradient_noise_gamma: f32,
    /// **Multi-resolution buckets with per-bucket repeat counts** (epic 2123, sc-2127), ported from
    /// ai-toolkit-perceptual's dataset `resolution: [..]` + `num_repeats: [..]` lists. Each dataset
    /// item is cached once **per bucket** (square-cropped to that bucket's edge, which the trainer
    /// floors to its latent stride exactly as it does [`resolution`](Self::resolution)), and every
    /// epoch visits each item `repeats` times per bucket — so buckets `512/768/1024` with repeats
    /// `16/4/1` train on a `16:4:1` per-image sample mix at those three latent sizes. The order is
    /// a shuffle seeded from [`seed`](Self::seed) and the epoch index (E4), see [`BucketSchedule`].
    ///
    /// Empty (the default) is **off**: the trainer caches one latent per item at
    /// [`resolution`](Self::resolution) and walks the cache round-robin exactly as before. A single
    /// bucket is never shuffled, so a single bucket equal to today's resolution reproduces today's
    /// sample order. A non-empty list is refused (typed [`crate::Error::Unsupported`]) by any
    /// trainer whose [`TrainerDescriptor::techniques`] does not declare
    /// [`resolution_buckets`](TrainingTechniques::resolution_buckets); a malformed list (a zero
    /// resolution or repeat count, a resolution off the [`RESOLUTION_BUCKET_STRIDE`], a duplicate
    /// resolution, more than [`MAX_RESOLUTION_BUCKETS`]) is
    /// refused by [`validate_training_techniques`]. Memory pre-flights size for
    /// [`max_training_resolution`](Self::max_training_resolution) (E7).
    pub resolution_buckets: Vec<ResolutionBucket>,
    /// Aspect-assigned rectangular buckets. Mutually exclusive with the legacy square list.
    pub aspect_ratio_buckets: Vec<AspectRatioBucket>,
    /// Keep frozen VAE outputs in run-owned RAM; false encodes samples on demand.
    pub cache_latents: bool,
    /// Keep frozen caption embeddings in run-owned RAM; false encodes captions on demand.
    pub cache_text_embeddings: bool,
    /// Caller-selected GPU/unified-memory ceiling. None uses the physical device budget;
    /// Some(0) refuses. A caller cap can lower discovery, never replace or raise it.
    pub memory_limit_bytes: Option<u64>,
    /// Operational ceiling as a fraction of the actual selected device's physical TOTAL bytes,
    /// not its free memory, safety allowance or recommended working set. None is off; a present
    /// value must be finite and 0 < fraction <= 1. Intersects both byte cap and live admission.
    /// MOSS consumes this field; it is not part of numerical resume identity.
    pub memory_limit_fraction: Option<f32>,
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
    /// **Subject-masked loss weighting** (epic 2123, sc-24828) — weight the per-element training
    /// loss by each item's subject mask ([`TrainingItem::subject_mask_path`]), ported from
    /// ai-toolkit-perceptual. The mask is cropped/resized exactly like its image, area-averaged
    /// down to the latent grid (so a latent cell straddling the subject edge gets a fractional
    /// mask value `m`), and turned into the weight map
    /// `w = background_weight + (subject_weight − background_weight) · m`, which multiplies the
    /// per-element loss **before** the mean reduction (see [`subject_mask`]).
    /// The divisor stays the full element count, so the gradient (and effective learning rate)
    /// scales by about `background_weight + (subject_weight − background_weight) · coverage`
    /// — with `background_weight = 0` and a small subject, proportionally lower.
    ///
    /// `None` (the default) is **off**: no mask is read and the loss is exactly the unweighted
    /// mean it was before this field existed. `Some` is refused (typed
    /// [`crate::Error::Unsupported`]) by a trainer whose [`TrainerDescriptor::techniques`] does not
    /// declare [`subject_mask_loss`](TrainingTechniques::subject_mask_loss), and refused when any
    /// item lacks a mask — see [`validate_training_techniques`].
    pub subject_mask_loss: Option<SubjectMaskLoss>,
    /// **Body losses** (epic 2123, sc-24832) — three decoded-x0 auxiliary perceptual losses on the
    /// shared path: ViTPose+ bone-length **proportions**, HybrIK SMPL-beta **shape**, and Sapiens
    /// surface **normals**, each with its own [`AuxLossSchedule`] (see [`body`]). Every loss is
    /// off by default; each is refused (typed [`crate::Error::Unsupported`]) by a trainer whose
    /// [`TrainerDescriptor::techniques`] does not declare its flag
    /// ([`body_proportion_loss`](TrainingTechniques::body_proportion_loss),
    /// [`body_shape_loss`](TrainingTechniques::body_shape_loss),
    /// [`normal_loss`](TrainingTechniques::normal_loss)) — see [`validate_training_techniques`].
    /// An image with no detected person is skipped by all three.
    pub body_losses: BodyLossesConfig,
    /// **ArcFace identity loss** (epic 2123, sc-24831) — an auxiliary perceptual loss on the shared
    /// decoded-x0 path: the face of the decoded x0 prediction (cropped with the box SCRFD found on
    /// the image's own encode→decode round trip, once per image) is embedded by a frozen ArcFace and
    /// pulled toward the image's reference embedding (or the dataset mean): upstream's bias-centred
    /// cosine (both embeddings minus the mean embedding of 200 noise images, re-normalized),
    /// `loss = max(0, 1 − cos / clean)` (`clean` = 1 per-image, the image's own clean score in
    /// dataset-average mode), scaled by the step's noise level. An image with no detected face is
    /// skipped; a step whose live cosine is at or below [`IdentityLossConfig::min_cos`]
    /// contributes zero (no push on a hallucinated non-face). Off by
    /// default ([`AuxLossSchedule::weight`] `0`); refused (typed [`crate::Error::Unsupported`]) by a
    /// trainer whose [`TrainerDescriptor::techniques`] does not declare
    /// [`identity_loss`](TrainingTechniques::identity_loss).
    pub identity_loss: IdentityLossConfig,
    /// **Face-landmark loss** (epic 2123, sc-24831) — an auxiliary perceptual loss on the shared
    /// decoded-x0 path: a frozen MediaPipe FaceMesh-v2 predicts the 478 landmarks of the decoded x0
    /// face crop (same reference-time SCRFD box as the identity loss) and the region-weighted
    /// (jaw ×3, lips ×2, eyes+nose ×1) mean landmark distance to the reference's normalized
    /// landmarks is the loss, scaled by the step's noise level and — when the identity loss is on
    /// — gated by its cosine (a frame at or below `min_cos` contributes zero). Off by default;
    /// refused by a trainer that does not declare
    /// [`face_landmark_loss`](TrainingTechniques::face_landmark_loss).
    pub face_landmark_loss: FaceLandmarkLossConfig,
    /// Directory holding the **face-analysis stack** both face losses detect with —
    /// `scrfd_10g.safetensors` (SCRFD detector) and `arcface_iresnet100.safetensors` (the ArcFace
    /// embedder the identity loss runs), the SceneWorks `instantid_face_stack` bundle. Required (a
    /// typed refusal otherwise) whenever [`identity_loss`](Self::identity_loss) or
    /// [`face_landmark_loss`](Self::face_landmark_loss) is enabled. `None` by default.
    pub face_analysis_dir: Option<PathBuf>,
    /// **VAE perceptual anchor** (epic 2123, sc-24833) — an auxiliary perceptual loss ported from
    /// ai-toolkit-perceptual `toolkit/vae_anchor.py`: the model's x0 prediction is decoded with the
    /// family's small differentiable decoder (the shared decoded-x0 path, E8), re-encoded by a
    /// frozen **FLUX.2 VAE encoder**, and its multi-scale encoder features (the output of each of
    /// the four resolution levels' last resnet and of the mid block's second resnet) are compared
    /// with a per-level `1 - cosine` against the same features of the training image's clean round
    /// trip, cached once per image. Off by default ([`AuxLossSchedule::weight`] `0`); refused
    /// (typed [`crate::Error::Unsupported`]) by any trainer whose
    /// [`TrainerDescriptor::techniques`] does not declare
    /// [`vae_anchor_loss`](TrainingTechniques::vae_anchor_loss) — see
    /// [`validate_training_techniques`].
    pub vae_anchor: VaeAnchorConfig,
    /// **E-LatentLPIPS** (epic 2123, sc-24833) — the learned latent-space perceptual metric of
    /// Kang et al. (ECCV 2024, `mingukkang/elatentlpips`) between the model's x0 prediction and the
    /// training image's clean latent, ported from ai-toolkit-perceptual's
    /// `latent_perceptual_loss_weight`. It runs on the latent directly (no decode) with the weights
    /// calibrated for the trainer's latent family ([`LatentLpipsFamily`]). Off by default; refused
    /// (typed [`crate::Error::Unsupported`]) by any trainer whose
    /// [`TrainerDescriptor::techniques`] does not declare
    /// [`latent_lpips_loss`](TrainingTechniques::latent_lpips_loss) — which no trainer whose latent
    /// family has no published E-LatentLPIPS weights declares.
    pub latent_lpips: LatentLpipsConfig,
    /// **Training adapter** (sc-25213) — a frozen LoRA `.safetensors` applied to the base model for
    /// the training forward/backward only, ai-toolkit's `assistant_lora_path`. Its use is the
    /// step-distilled Z-Image-Turbo base: ostris' de-distill adapter is applied to the base while
    /// the user's LoRA trains, so the user's adapter learns the concept rather than undoing the
    /// distillation, and is then removed again — preview samples render without it, and the saved
    /// adapter carries **only** the trainable keys, so the LoRA applies to the bare distilled base at
    /// generation time exactly as the previews showed.
    ///
    /// `None` (the default) is **off**: nothing is read and the trainer trains on the bare base
    /// exactly as before this field existed. `Some` is refused (typed
    /// [`crate::Error::Unsupported`]) by any trainer whose [`TrainerDescriptor::techniques`] does
    /// not declare [`training_adapter`](TrainingTechniques::training_adapter), and for a
    /// [`full_finetune`](Self::full_finetune) run (the adapter would be baked into the fine-tuned
    /// weights) — see [`validate_training_techniques`]. A file that cannot be read fails the run
    /// when the trainer loads it, naming the path.
    pub training_adapter: Option<PathBuf>,
}

/// Where [`IdentityLossConfig`]'s per-image target embedding comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IdentityReferenceMode {
    /// Every image's target is the L2-normalized mean of all face-bearing images' reference
    /// embeddings (upstream `identity_loss_use_average`, its default) — pulls each sample toward
    /// the subject's identity rather than toward one photo's pose/lighting.
    #[default]
    DatasetAverage,
    /// Every image's target is its own reference embedding.
    PerImage,
}

impl IdentityReferenceMode {
    /// Parse the contract string (`dataset_average`/`per_image`, case-insensitive); `None` otherwise.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "dataset_average" => Some(Self::DatasetAverage),
            "per_image" => Some(Self::PerImage),
            _ => None,
        }
    }

    /// The contract string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DatasetAverage => "dataset_average",
            Self::PerImage => "per_image",
        }
    }
}

/// [`TrainingConfig::identity_loss`] — the ArcFace identity loss schedule and its gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityLossConfig {
    /// Weight / window / alternation. Off by default.
    pub schedule: AuxLossSchedule,
    /// A step whose live cosine `cos(embed(x0_face), reference)` is `<= min_cos` contributes zero
    /// identity loss (upstream `identity_loss_min_cos`, default `0.2`). In `[-1, 1]`.
    pub min_cos: f32,
    /// Per-image or dataset-average target embedding.
    pub reference_mode: IdentityReferenceMode,
}

impl IdentityLossConfig {
    /// Upstream `identity_loss_min_cos` default.
    pub const DEFAULT_MIN_COS: f32 = 0.2;
    /// Inclusive bounds of [`min_cos`](Self::min_cos).
    pub const MIN_COS_RANGE: (f32, f32) = (-1.0, 1.0);
}

impl Default for IdentityLossConfig {
    fn default() -> Self {
        Self {
            schedule: PIXEL_PERCEPTUAL_SCHEDULE,
            min_cos: Self::DEFAULT_MIN_COS,
            reference_mode: IdentityReferenceMode::default(),
        }
    }
}

/// [`TrainingConfig::face_landmark_loss`] — the FaceMesh landmark loss schedule plus the frozen
/// FaceMesh-v2 checkpoint it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceLandmarkLossConfig {
    /// Weight / window / alternation. Off by default ([`PIXEL_PERCEPTUAL_SCHEDULE`]).
    pub schedule: AuxLossSchedule,
    /// Directory holding the converted FaceMesh-v2 program checkpoint
    /// (`face_landmarks_detector.safetensors`). Required (a typed refusal otherwise) when the loss is
    /// enabled.
    pub model_dir: Option<PathBuf>,
}

impl Default for FaceLandmarkLossConfig {
    fn default() -> Self {
        Self {
            schedule: PIXEL_PERCEPTUAL_SCHEDULE,
            model_dir: None,
        }
    }
}

/// The weights of [`TrainingConfig::subject_mask_loss`] (epic 2123, sc-24828). A latent cell with
/// mask value `m ∈ [0, 1]` (1 = subject) weighs `background_weight + (subject_weight −
/// background_weight) · m` in the loss. `background_weight = 0` drops the background from the loss
/// (and from the gradient) entirely; `background_weight = subject_weight = 1` is the unweighted loss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubjectMaskLoss {
    /// Weight of a pure-background latent cell, in `[0, 1]`.
    pub background_weight: f32,
    /// Weight of a pure-subject latent cell, in `(0, 1]`.
    pub subject_weight: f32,
}

impl SubjectMaskLoss {
    /// Inclusive bounds of [`background_weight`](Self::background_weight).
    pub const BACKGROUND_WEIGHT_RANGE: (f32, f32) = (0.0, 1.0);
    /// Bounds of [`subject_weight`](Self::subject_weight): exclusive lower, inclusive upper (a
    /// zero subject weight would train on nothing but background — the inverse of the technique).
    pub const SUBJECT_WEIGHT_RANGE: (f32, f32) = (0.0, 1.0);

    /// The loss weight of a latent cell whose (area-averaged) mask value is `m`.
    pub fn weight(&self, m: f32) -> f32 {
        self.background_weight + (self.subject_weight - self.background_weight) * m
    }

    /// Refuse malformed weights with a message naming the field.
    pub fn validate(&self, label: &str) -> crate::Result<()> {
        let (bg_lo, bg_hi) = Self::BACKGROUND_WEIGHT_RANGE;
        let bg = self.background_weight;
        if !bg.is_finite() || bg < bg_lo || bg > bg_hi {
            return Err(crate::Error::Msg(format!(
                "{label}: subject_mask_loss.background_weight must be in [{bg_lo}, {bg_hi}], got {bg}"
            )));
        }
        let (s_lo, s_hi) = Self::SUBJECT_WEIGHT_RANGE;
        let s = self.subject_weight;
        if !s.is_finite() || s <= s_lo || s > s_hi {
            return Err(crate::Error::Msg(format!(
                "{label}: subject_mask_loss.subject_weight must be in ({s_lo}, {s_hi}], got {s}"
            )));
        }
        Ok(())
    }
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
    /// Alternation period, counted in optimizer updates (see [`aux_schedule::AltKey::claims`]):
    /// - `1` — the weighted aux loss is **added** to the diffusion loss on every in-window step;
    /// - `n ≥ 2` — about every `n`-th optimizer update is an **aux-only** update on which the
    ///   diffusion loss contributes **zero** (never two in a row); the others are diffusion-only.
    ///   The phase drifts per pass over the data, so every image gets both kinds. An aux-only
    ///   update samples its noise level inside `[t_min, t_max]`, so no step is wasted. `2` (the
    ///   default) is upstream's every-other-update alternation.
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

/// Which E-LatentLPIPS weight set matches a trainer's latent space (epic 2123, the latent
/// perceptual loss): the trainer names its family in the aux-loss builder context; `None` there
/// means no E-LatentLPIPS weights exist for that latent space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentLpipsFamily {
    Sd15,
    Sd21,
    Sdxl,
    Sd3,
    Flux,
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
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAnchoringConfig {
    /// Weight / window / alternation. Off by default ([`PIXEL_PERCEPTUAL_SCHEDULE`]).
    pub schedule: AuxLossSchedule,
    /// Which DA2 checkpoint [`model_dir`](Self::model_dir) holds.
    pub model_size: DepthModelSize,
    /// Directory holding the DA2 `*-hf` checkpoint (`model.safetensors`). Required (a typed
    /// refusal otherwise) when the loss is enabled.
    pub model_dir: Option<PathBuf>,
}

impl Default for DepthAnchoringConfig {
    fn default() -> Self {
        Self {
            schedule: PIXEL_PERCEPTUAL_SCHEDULE,
            model_size: DepthModelSize::default(),
            model_dir: None,
        }
    }
}

/// The default schedule of the **decoded-x0 (pixel-space)** auxiliary losses whose upstream window
/// was `[0, 1]` — depth anchoring, the identity and face-landmark losses and the body-proportion
/// loss (sc-2124; body shape / normal keep upstream's [`BODY_SURFACE_SCHEDULE`]) — once given a
/// weight: the low-noise window `[0, 0.5]` the latent losses use ([`LATENT_PERCEPTUAL_SCHEDULE`])
/// and upstream's every-other-update alternation (`every_n = 2`). Near pure noise the decoded x0
/// is mush, so an aux-only update there trains the adapter toward the frozen model's reading of
/// noise; the real-weight A/B saw `[0, 1]` destroy likeness on SDXL and hurt Z-Image. Weight `0`
/// ⇒ off. An explicit window is honoured unchanged.
pub const PIXEL_PERCEPTUAL_SCHEDULE: AuxLossSchedule = AuxLossSchedule {
    weight: 0.0,
    t_min: 0.0,
    t_max: 0.5,
    every_n: 2,
};

/// The default schedule of the **body shape and surface-normal** losses (sc-2124): upstream
/// ai-toolkit-perceptual's `[0.4, 0.8]` window, every-other-update alternation. Unlike the other
/// decoded-x0 losses (upstream `[0, 1]`, here [`PIXEL_PERCEPTUAL_SCHEDULE`]) this window already
/// excludes the top of the noise range, so it is kept. Weight `0` ⇒ off.
pub const BODY_SURFACE_SCHEDULE: AuxLossSchedule = AuxLossSchedule {
    weight: 0.0,
    t_min: 0.4,
    t_max: 0.8,
    every_n: 2,
};

/// The upstream (ai-toolkit-perceptual) default schedule of the two latent-space perceptual losses
/// (sc-24833) once given a weight: window `[0, 0.5]` (`*_loss_min_t` / `*_loss_max_t`) and
/// **additive** (`every_n = 1`) — upstream adds both terms to the diffusion loss on every in-window
/// step rather than alternating. Weight `0` ⇒ off.
pub const LATENT_PERCEPTUAL_SCHEDULE: AuxLossSchedule = AuxLossSchedule {
    weight: 0.0,
    t_min: 0.0,
    t_max: 0.5,
    every_n: 1,
};

/// [`TrainingConfig::vae_anchor`] — the VAE-anchor loss schedule plus the frozen FLUX.2 VAE whose
/// encoder it runs (sc-24833).
#[derive(Clone, Debug, PartialEq)]
pub struct VaeAnchorConfig {
    /// Weight / window / alternation. Off by default ([`LATENT_PERCEPTUAL_SCHEDULE`]).
    pub schedule: AuxLossSchedule,
    /// Directory holding the FLUX.2 VAE in the diffusers layout
    /// (`diffusion_pytorch_model.safetensors`; only `encoder.*` tensors are read). Required (a
    /// typed refusal otherwise) when the loss is enabled.
    pub model_dir: Option<PathBuf>,
}

impl Default for VaeAnchorConfig {
    fn default() -> Self {
        Self {
            schedule: LATENT_PERCEPTUAL_SCHEDULE,
            model_dir: None,
        }
    }
}

/// [`TrainingConfig::latent_lpips`] — the E-LatentLPIPS loss schedule plus the directory holding the
/// published weights (sc-24833). The trainer picks the file for its own [`LatentLpipsFamily`].
#[derive(Clone, Debug, PartialEq)]
pub struct LatentLpipsConfig {
    /// Weight / window / alternation. Off by default ([`LATENT_PERCEPTUAL_SCHEDULE`]).
    pub schedule: AuxLossSchedule,
    /// Directory holding the E-LatentLPIPS checkpoints (a `Mingguksky/elatentlpips` snapshot — see
    /// [`LatentLpipsFamily::checkpoint_file`]). Required (a typed refusal otherwise) when the loss
    /// is enabled.
    pub model_dir: Option<PathBuf>,
}

impl Default for LatentLpipsConfig {
    fn default() -> Self {
        Self {
            schedule: LATENT_PERCEPTUAL_SCHEDULE,
            model_dir: None,
        }
    }
}

/// [`LatentLpipsFamily`] — the latent spaces E-LatentLPIPS publishes calibrated weights for (`Mingguksky/elatentlpips`,
/// sc-24833). A trainer whose VAE latent space is none of these (FLUX.2, Qwen-Image / Wan, LTX
/// latents) has no matching weights and does not declare
/// [`latent_lpips_loss`](TrainingTechniques::latent_lpips_loss), so the floor refuses the loss.
///
/// | family | latent | model-space latent | trainers |
/// |---|---|---|---|
/// | `Sd15` | SD 1.x VAE, 4 ch | `z * 0.18215` | none |
/// | `Sd21` | SD 2.x VAE, 4 ch | `z * 0.18215` | none |
/// | `Sdxl` | SDXL VAE, 4 ch | `z * 0.13025` | SDXL / Illustrious, Kolors |
/// | `Sd3` | SD3 VAE, 16 ch | `(z - 0.0609) * 1.5305` | SD3.5 |
/// | `Flux` | FLUX.1 VAE, 16 ch | `(z - 0.1159) * 0.3611` | Z-Image |
///
/// The network consumes the **model-space** latent the diffusion model trains on (upstream calls it
/// with `normalize=False` on exactly those latents).
impl LatentLpipsFamily {
    /// The upstream encoder name (`sd15` / `sd21` / `sdxl` / `sd3` / `flux`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sd15 => "sd15",
            Self::Sd21 => "sd21",
            Self::Sdxl => "sdxl",
            Self::Sd3 => "sd3",
            Self::Flux => "flux",
        }
    }

    /// Latent channels the network's first convolution consumes.
    pub fn latent_channels(self) -> usize {
        match self {
            Self::Sd15 | Self::Sd21 | Self::Sdxl => 4,
            Self::Sd3 | Self::Flux => 16,
        }
    }

    /// The published checkpoint's path relative to [`LatentLpipsConfig::model_dir`].
    pub fn checkpoint_file(self) -> String {
        format!("elatentlpips_ckpt/{}_latest_vgg16_tuned.pth", self.as_str())
    }
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
            // Gradient noise is OFF by default (epic 2123 E1); gamma carries the upstream default
            // so turning eta on alone gives the paper's schedule.
            gradient_noise_eta: 0.0,
            gradient_noise_gamma: DEFAULT_GRADIENT_NOISE_GAMMA,
            // Resolution buckets are OFF by default (epic 2123 E1): one bucket at `resolution`,
            // walked round-robin exactly as before.
            resolution_buckets: Vec::new(),
            aspect_ratio_buckets: Vec::new(),
            cache_latents: true,
            cache_text_embeddings: true,
            memory_limit_bytes: None,
            memory_limit_fraction: None,
            // Subject-masked loss is OFF by default (epic 2123 E1): no mask is read and the loss is
            // the plain unweighted mean.
            subject_mask_loss: None,
            // Identity / face-landmark losses are OFF by default (epic 2123 E1): weight 0, no face
            // model loaded.
            identity_loss: IdentityLossConfig::default(),
            face_landmark_loss: FaceLandmarkLossConfig::default(),
            face_analysis_dir: None,
            // Body losses are OFF by default (epic 2123 E1): all three weights 0, no model loaded.
            body_losses: BodyLossesConfig::default(),
            // The latent-space perceptual losses are OFF by default (epic 2123 E1, sc-24833).
            vae_anchor: VaeAnchorConfig::default(),
            latent_lpips: LatentLpipsConfig::default(),
            // No training adapter by default (sc-25213): the trainer trains on the bare base.
            training_adapter: None,
        }
    }
}

/// Intersect a caller ceiling with actual device discovery; zero is never a guard bypass.
pub fn effective_memory_budget(physical: u64, cap: Option<u64>) -> crate::Result<u64> {
    if physical == 0 || cap == Some(0) {
        return Err(crate::Error::Unsupported(
            "training memory budget must be positive".into(),
        ));
    }
    Ok(cap.map_or(physical, |limit| physical.min(limit)))
}

/// Resolve an operational fraction against the selected device's TOTAL physical bytes, then
/// intersect its independent live/safety budget and optional absolute cap. Integer IEEE-754
/// arithmetic floors exactly, including large u64 totals, rather than rounding a cap upward.
pub fn effective_memory_budget_with_fraction(
    available_budget: u64,
    physical_total: Option<u64>,
    byte_cap: Option<u64>,
    fraction: Option<f32>,
) -> crate::Result<u64> {
    let budget = effective_memory_budget(available_budget, byte_cap)?;
    let Some(fraction) = fraction else {
        return Ok(budget);
    };
    validate_memory_limit_fraction(Some(fraction))?;
    let total = physical_total.filter(|n| *n > 0).ok_or_else(|| {
        crate::Error::Unsupported(
            "training memory fraction requires the actual selected physical total".into(),
        )
    })?;
    let bits = fraction.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let mantissa = if exponent == 0 {
        bits & 0x7fffff
    } else {
        (bits & 0x7fffff) | (1 << 23)
    };
    let shift = if exponent == 0 { 149 } else { 150 - exponent };
    let cap = if shift >= 128 {
        0
    } else {
        (u128::from(total) * u128::from(mantissa)) >> shift
    };
    let cap = u64::try_from(cap)
        .map_err(|_| crate::Error::Unsupported("training memory fraction overflow".into()))?;
    effective_memory_budget(budget, Some(cap))
}

pub(super) fn validate_memory_limit_fraction(fraction: Option<f32>) -> crate::Result<()> {
    if fraction.is_some_and(|f| !f.is_finite() || f <= 0.0 || f > 1.0) {
        return Err(crate::Error::Unsupported(
            "training memory fraction must be finite and 0 < fraction <= 1; None means off".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod memory_budget_tests;

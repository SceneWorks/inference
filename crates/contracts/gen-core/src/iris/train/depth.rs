//! Backend-neutral **depth training contract** of Iris-3B (epic sc-25678, story sc-25687): the
//! supervised direct-regression fine-tune that produces a `depth/` export — everything about a depth
//! run that is not a tensor. The MLX (`mlx-gen-iris::train::depth`) and Candle
//! (`candle-gen-iris::train::depth`) trainers read this one definition, so both backends see the
//! same request surface, the same aligned batches (same paired transforms, same positional draws,
//! same normalized targets), the same loss definition, the same checkpoints and the same export.
//! It extends [`super`] (the generation training contract): the positional seeding, the ranged
//! [`DataWalk`], the optimizer plan / routing / LR law, clipping, EMA, the durable checkpoint
//! layout ([`publish_checkpoint`](super::publish_checkpoint), retention, `latest`) and the micro-step
//! units are the same code.
//!
//! ## Source fidelity
//!
//! The frozen release (`speridlabs/iris-3b` @ [`super::super::UPSTREAM_CODE_REVISION`]) ships **no**
//! downstream training loop. The model, the export and the inference forward are upstream code
//! (`iris3b/downstream/depth.py` `IrisDepth`, `scripts/export_downstream.py depth`); the recipe is
//! the paper's §5.2 (arXiv 2610.09450v1, "Recipe." / "Data."). Every choice below names its source;
//! where the paper is silent the conventional choice is the default **and** an option (the
//! `UPSTREAM.md` depth-training rows list them):
//!
//! | choice | source | default | option |
//! | --- | --- | --- | --- |
//! | model: widened input projections, zero 4th channel, 1×1 reducer, empty prompt, one forward at `t = num_train_timesteps` | `IrisDepth.forward` | — | — |
//! | target: per-image relative log depth, 2nd/98th percentiles of the **valid** GT mapped to −1 / +1 | paper §5.2 + `depth.py` docstring | 2 / 98 | `low_percentile`, `high_percentile` |
//! | target clamp to [−1, 1] | paper silent (Marigold clips; the stated map does not) | off | `clip_target` |
//! | loss `1/|M| Σ_{p∈M} |d̂−d| + 5‖∇d̂−∇d‖₁` (Eq. 5) | paper §5.2 | weight 5 | `gradient_weight` |
//! | ∇: forward differences in x and y, a pair counted when both pixels are valid | paper silent (∇ undefined) | — (documented) | — |
//! | `|M|` over the whole micro-batch | Eq. 5 has one `M` | `valid_pixels` | `loss_reduction: per_image` |
//! | AdamW β (0.9, 0.999), wd 0.01, LR 3e-5 constant, no warmup, clip 1.0, batch 32, 10K steps, raw weights | paper §5.2 | clip 1.0, betas, no EMA, raw export | [`apply_paper_recipe`] sets the rest |
//! | random horizontal flips | paper §5.2 | p = 0.5 | `hflip_prob` |
//! | crop / resize | paper silent (native 480×640 / 352×1216) | no resample at the native size, else cover-resize; center crop | `resize: none`, `crop: random` |
//! | invalid / beyond-far-plane pixels masked from the loss | paper §5.2 | per-item mask + `max_depth` | — |
//! | reducer init from a generation parent | paper silent; `nn.Conv2d` default in `IrisDepth.__init__` | torch default | `reducer_init: channel_mean` |
//! | widened-channel init from a generation parent | paper silent (the channel is always zero) | zeros | — |
//!
//! The 9:1 Hypersim / Virtual KITTI 2 mixing is dataset composition (the SceneWorks dataset owns
//! it). A batch has one training size ([`DepthAugmentPlan`]); sources of another size are resized
//! (bicubic RGB, nearest depth + mask) to cover it and cropped — paired, so RGB, depth and mask stay
//! aligned pixel for pixel.
//!
//! ## Dataset items
//!
//! Each [`TrainingItem`] is one aligned example: `image_path` is the RGB image (PNG / JPEG) and
//! `model_options["depth"]` ([`ITEM_DEPTH_KEY`]) an object
//!
//! | key | meaning |
//! | --- | --- |
//! | `path` | numerical depth, same `W × H` as the image: `.npy` (`f4`/`f8`/`u2`/`u1`/`i4`, shape `[H, W]`, `[1, H, W]` or `[H, W, 1]`) or a 16-/8-bit grayscale `.png` |
//! | `mask_path` | optional validity mask (`.npy` or `.png`), non-zero = valid |
//! | `scale` | optional factor applied to the stored values (e.g. `0.01` for centimetres); default 1 |
//! | `min_depth` | optional: values `<= min_depth` are invalid (default 0 — `log` needs `d > 0`) |
//! | `max_depth` | optional far plane: values `> max_depth` are invalid (paper: 65 m / 80 m) |
//!
//! A pixel is valid when the mask allows it, the scaled depth is finite, `> min_depth` and
//! `<= max_depth`. Captions are not read (the task runs on the empty prompt): a non-empty caption is
//! refused rather than ignored.
//!
//! ## Invalid examples (explicit policy, `invalid_examples`)
//!
//! An item is **invalid** when it has fewer than `min_valid_pixels` valid pixels or its valid log
//! depths have a degenerate 2nd–98th percentile range (`< 1e-6`, e.g. a constant plane): the
//! normalization is undefined. `reject` (default) refuses the run at the dataset pass naming the item;
//! `skip` drops it from the walk with a warning (an all-invalid dataset is refused). After the
//! random crop a row can still lose its support: such a row's mask is zeroed, so it contributes
//! nothing; a micro-batch with no valid pixel has loss 0 and zero gradient (it still counts as a
//! micro-step).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Map as JsonMap, Value as JsonValue};
use sha2::{Digest, Sha256};

use super::{
    batch_seed, mix_seed, scale_lr, CheckpointState, DataWalk, HostRng, LrShape, MixedPrecision,
    MuonAdjustLr, OptimizerKind, OptimizerPlan, ResumeDataPolicy, WeightsSelect, OPTIONS_KEY,
};
use crate::iris::{IrisConfig, IrisTask, ModelConfig, DEPTH_MODEL_ID};
use crate::train::{LrSchedule, TrainingConfig, TrainingItem, TrainingRequest};
use crate::{Error, Result};

/// Registry id of the Iris depth trainer (the same id as the depth estimator it trains for).
pub const DEPTH_TRAINER_ID: &str = DEPTH_MODEL_ID;
/// `TrainingItem::model_options` key carrying an item's depth target (see the module docs).
pub const ITEM_DEPTH_KEY: &str = "depth";
/// `LoadSpec::components` key of an `empty_prompt.safetensors` file: the empty-prompt conditioning
/// a run that starts from the **generation** backbone needs (a `depth/` export carries its own).
pub const EMPTY_PROMPT_COMPONENT: &str = "empty_prompt";
/// The export's key prefix of the widened backbone (`IrisDepth.pixel`).
pub const EXPORT_BACKBONE_PREFIX: &str = "pixel.";
/// `IrisDepth.depth_reducer` (`nn.Conv2d(in_channels, 1, 1)`), `[1, C, 1, 1]` and `[1]`.
pub const REDUCER_WEIGHT: &str = "depth_reducer.weight";
pub const REDUCER_BIAS: &str = "depth_reducer.bias";
/// Eq. 5's gradient-term weight.
pub const DEFAULT_GRADIENT_WEIGHT: f64 = 5.0;
/// The normalization percentiles (paper §5.2: "the 2nd and 98th percentiles of the valid GT").
pub const DEFAULT_LOW_PERCENTILE: f64 = 2.0;
pub const DEFAULT_HIGH_PERCENTILE: f64 = 98.0;
/// Paper §5.2: AdamW β = (0.9, 0.999).
pub const DEFAULT_BETAS: (f64, f64) = (0.9, 0.999);
/// Paper §5.2: gradient clipping at 1.0.
pub const DEFAULT_GRADIENT_CLIP: f64 = 1.0;
/// Paper §5.2: random horizontal flips (probability ½, the conventional flip).
pub const DEFAULT_HFLIP_PROB: f64 = 0.5;
/// The least `min_valid_pixels` (two values make a percentile range).
pub const MIN_VALID_PIXELS_FLOOR: usize = 2;
/// A 2nd–98th percentile log-depth range below this is degenerate (no normalization exists).
pub const DEGENERATE_LOG_RANGE: f64 = 1e-6;

/// Paper §5.2 recipe values (`apply_paper_recipe`).
pub const PAPER_LEARNING_RATE: f32 = 3e-5;
pub const PAPER_WEIGHT_DECAY: f32 = 0.01;
pub const PAPER_GLOBAL_BATCH: u32 = 32;
pub const PAPER_STEPS: u32 = 10_000;

/// Set the shared-config fields of the paper's depth recipe (§5.2): full fine-tune, AdamW, LR 3e-5
/// constant, no warmup, weight decay 0.01, global batch 32 (as `batch_size` 32, accumulation 1),
/// 10K steps, L1 loss. The Iris-options defaults already carry β, clipping and raw-weights export.
/// The caller still sets the size, cadences, seed and precision.
pub fn apply_paper_recipe(cfg: &mut TrainingConfig) {
    cfg.full_finetune = true;
    cfg.optimizer = "adamw".into();
    cfg.learning_rate = PAPER_LEARNING_RATE;
    cfg.weight_decay = PAPER_WEIGHT_DECAY;
    cfg.lr_scheduler = LrSchedule::Constant;
    cfg.lr_warmup_steps = 0;
    cfg.batch_size = PAPER_GLOBAL_BATCH;
    cfg.gradient_accumulation = 1;
    cfg.steps = PAPER_STEPS;
    cfg.loss_type = "l1".into();
}

// =============================================================================================
// The plan
// =============================================================================================

/// Where the trained tensors start (decided by the trainer from its `LoadSpec::weights`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepthInit {
    /// A `depth/` export (the release's or a previous native run's): every tensor as stored.
    Export,
    /// The **generation** backbone (the paper's parent): input projections widened (zero columns
    /// for the always-zero 4th channel), reducer initialised per [`ReducerInit`].
    Parent,
}

/// `resize`: how a source of another size reaches the training size before the crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeMode {
    /// Scale to cover the training size ([`cover_size`]; bicubic RGB, nearest depth + mask).
    Cover,
    /// Never resample: crop at native resolution (a smaller source is refused).
    None,
}

/// `crop`: how a batch row is cut to the training size after the covering resize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropMode {
    Center,
    Random,
}

/// `loss_reduction`: the `1/|M|` of Eq. 5 over a micro-batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossReduction {
    /// One `M` = every valid pixel of the micro-batch (Eq. 5 read literally; the default).
    ValidPixels,
    /// Eq. 5 per image, then the mean over the images that have a valid pixel.
    PerImage,
}

/// `invalid_examples`: what an invalid item does (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidPolicy {
    Reject,
    Skip,
}

/// `reducer_init` for a [`DepthInit::Parent`] run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducerInit {
    /// `nn.Conv2d`'s default (`kaiming_uniform(a=√5)` weight and the fan-in bias bound: both
    /// `U(±1/√C)`) — what `IrisDepth.__init__` constructs.
    TorchDefault,
    /// `w = 1/C`, `b = 0`: depth starts as the mean of the parent's output channels.
    ChannelMean,
}

/// The paired spatial transform of a batch row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthAugmentPlan {
    /// The training size (every row of every batch).
    pub height: usize,
    pub width: usize,
    pub resize: ResizeMode,
    pub crop: CropMode,
    pub hflip_prob: f64,
}

/// The depth loss and target normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthLossPlan {
    pub gradient_weight: f64,
    pub low_percentile: f64,
    pub high_percentile: f64,
    pub clip_target: bool,
    pub reduction: LossReduction,
}

impl Default for DepthLossPlan {
    fn default() -> Self {
        Self {
            gradient_weight: DEFAULT_GRADIENT_WEIGHT,
            low_percentile: DEFAULT_LOW_PERCENTILE,
            high_percentile: DEFAULT_HIGH_PERCENTILE,
            clip_target: false,
            reduction: LossReduction::ValidPixels,
        }
    }
}

/// Depth previews: every `every` optimizer steps the first `items` dataset items (center crop, no
/// flip) are predicted from the in-progress weights and reported as `TrainingProgress::Sample`
/// images (upstream's `colorize`, near = bright); the `prompt` field carries the item's image path.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthPreviewPlan {
    pub every: u32,
    pub items: usize,
    pub weights: WeightsSelect,
}

/// The fully resolved, validated plan of one depth training run.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthTrainPlan {
    pub augment: DepthAugmentPlan,
    pub loss: DepthLossPlan,
    pub batch_size: usize,
    pub grad_accum: usize,
    /// Optimizer-step horizon (`TrainingConfig::steps / grad_accum`).
    pub max_steps: u32,
    pub num_epochs: Option<u32>,
    pub seed: u64,
    pub mixed_precision: MixedPrecision,
    pub optimizer: OptimizerPlan,
    pub schedule: LrShape,
    pub warmup_steps: u64,
    pub gradient_clip: f64,
    pub ema: Option<f64>,
    pub nan_loss_tolerance: u32,
    pub invalid: InvalidPolicy,
    pub min_valid_pixels: usize,
    pub reducer_init: ReducerInit,
    /// `load_from`: a previous depth run's checkpoint directory or a depth export directory whose
    /// tensors replace the init (weights only; fresh optimizer).
    pub load_from: Option<PathBuf>,
    pub save_every: u32,
    pub keep_last_checkpoints: usize,
    pub milestone_steps: Vec<u64>,
    pub preview: DepthPreviewPlan,
    pub resume: bool,
    pub resume_from: Option<PathBuf>,
    pub resume_data_policy: ResumeDataPolicy,
    pub override_lr_on_resume: bool,
    pub export_weights: WeightsSelect,
}

/// The `model_options["iris"]` keys of a depth run (anything else is refused).
///
/// | key | default |
/// | --- | --- |
/// | `depth_size` `[h, w]` | `[resolution, resolution]` |
/// | `resize` (`cover` / `none`), `crop` (`center` / `random`), `hflip_prob` | `cover`, `center`, 0.5 |
/// | `gradient_weight`, `low_percentile`, `high_percentile`, `clip_target`, `loss_reduction` (`valid_pixels` / `per_image`) | 5, 2, 98, false, `valid_pixels` |
/// | `invalid_examples` (`reject` / `skip`), `min_valid_pixels` | `reject`, 2 |
/// | `reducer_init` (`torch_default` / `channel_mean`) | `torch_default` |
/// | `betas`, `gradient_clip` | `[0.9, 0.999]`, 1.0 |
/// | `muon_momentum`, `muon_nesterov`, `muon_adjust_lr`, `auto_lr`, `base_batch_size` | as generation |
/// | `ema_enabled`, `ema_decay` | false, 0.9999 |
/// | `num_epochs`, `nan_loss_tolerance` | unset, 20 |
/// | `keep_last_checkpoints`, `milestone_steps` | 0, `[]` |
/// | `load_from`, `resume_from`, `resume_data_policy`, `override_lr_on_resume` | unset, unset, `exact`, false |
/// | `preview_items`, `preview_weights` | 4 (capped by the dataset), `raw` |
/// | `export_weights` | `raw` (the paper evaluates the raw weights) |
pub const DEPTH_OPTION_KEYS: [&str; 32] = [
    "depth_size",
    "resize",
    "crop",
    "hflip_prob",
    "gradient_weight",
    "low_percentile",
    "high_percentile",
    "clip_target",
    "loss_reduction",
    "invalid_examples",
    "min_valid_pixels",
    "reducer_init",
    "betas",
    "gradient_clip",
    "muon_momentum",
    "muon_nesterov",
    "muon_adjust_lr",
    "auto_lr",
    "base_batch_size",
    "ema_enabled",
    "ema_decay",
    "num_epochs",
    "nan_loss_tolerance",
    "keep_last_checkpoints",
    "milestone_steps",
    "load_from",
    "resume_from",
    "resume_data_policy",
    "override_lr_on_resume",
    "preview_items",
    "preview_weights",
    "export_weights",
];

/// The default number of preview items.
pub const DEFAULT_PREVIEW_ITEMS: usize = 4;

struct Opts(JsonMap<String, JsonValue>);

impl Opts {
    fn from_config(cfg: &TrainingConfig) -> Result<Self> {
        let map = match cfg.model_options.get(OPTIONS_KEY) {
            None | Some(JsonValue::Null) => JsonMap::new(),
            Some(JsonValue::Object(m)) => m.clone(),
            Some(other) => {
                return Err(Error::Msg(format!(
                "iris depth training: model_options.{OPTIONS_KEY} must be an object, got {other}"
            )))
            }
        };
        for key in map.keys() {
            if !DEPTH_OPTION_KEYS.contains(&key.as_str()) {
                return Err(Error::Unsupported(format!(
                    "iris depth training: model_options.{OPTIONS_KEY}.{key} is not an Iris depth \
                     training option (known: {})",
                    DEPTH_OPTION_KEYS.join(", ")
                )));
            }
        }
        for key in cfg.model_options.keys() {
            if key != OPTIONS_KEY {
                return Err(Error::Unsupported(format!(
                    "iris depth training: model_options.{key} is not read by the depth trainer \
                     (its options live under model_options.{OPTIONS_KEY})"
                )));
            }
        }
        Ok(Self(map))
    }
    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key).filter(|v| !v.is_null())
    }
    fn f64(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                Error::Msg(format!(
                    "iris depth training: {key} must be a finite number, got {v}"
                ))
            }),
        }
    }
    fn u64(&self, key: &str) -> Result<Option<u64>> {
        self.get(key)
            .map(|v| {
                v.as_u64().ok_or_else(|| {
                    Error::Msg(format!(
                        "iris depth training: {key} must be a non-negative integer, got {v}"
                    ))
                })
            })
            .transpose()
    }
    fn bool(&self, key: &str, default: bool) -> Result<bool> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_bool().ok_or_else(|| {
                Error::Msg(format!(
                    "iris depth training: {key} must be a bool, got {v}"
                ))
            }),
        }
    }
    fn string(&self, key: &str) -> Result<Option<String>> {
        self.get(key)
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris depth training: {key} must be a string, got {v}"
                    ))
                })
            })
            .transpose()
    }
    fn choice<T: Copy>(&self, key: &str, default: T, table: &[(&str, T)]) -> Result<T> {
        match self.string(key)? {
            None => Ok(default),
            Some(s) => table
                .iter()
                .find(|(name, _)| *name == s)
                .map(|(_, v)| *v)
                .ok_or_else(|| {
                    let names: Vec<&str> = table.iter().map(|(n, _)| *n).collect();
                    Error::Msg(format!(
                        "iris depth training: {key} {s:?} ({})",
                        names.join(" | ")
                    ))
                }),
        }
    }
}

fn weights_select(opts: &Opts, key: &str, default: WeightsSelect) -> Result<WeightsSelect> {
    opts.choice(
        key,
        default,
        &[("raw", WeightsSelect::Raw), ("ema", WeightsSelect::Ema)],
    )
}

impl DepthTrainPlan {
    /// Resolve and validate a request against the depth model's config (the export's or the
    /// generation parent's — both carry the same `model` section). Every request field the depth
    /// objective cannot honour is refused by name, never ignored.
    pub fn resolve(req: &TrainingRequest, config: &IrisConfig) -> Result<Self> {
        let cfg = &req.config;
        let opts = Opts::from_config(cfg)?;
        let unsupported =
            |what: String| Err(Error::Unsupported(format!("iris depth training: {what}")));

        if !cfg.full_finetune {
            return unsupported(
                "the depth task trains the full model (paper §5.2: \"we fully fine-tune\"); set \
                 full_finetune — LoRA / LoKr depth adapters are not a depth export"
                    .into(),
            );
        }
        if cfg.gradient_checkpointing {
            return unsupported("gradient (activation) checkpointing is not implemented".into());
        }
        let loss = cfg.loss_type.trim().to_ascii_lowercase();
        if !matches!(loss.as_str(), "" | "l1" | "depth_l1") {
            return unsupported(format!(
                "loss_type {:?} — the depth objective is Eq. 5's masked L1 + gradient L1 \
                 (loss_type l1)",
                cfg.loss_type
            ));
        }
        // The flow-sampling knobs have no meaning for a single deterministic forward: only their
        // shared-contract defaults (which select nothing) are accepted.
        let ts = cfg.timestep_type.trim().to_ascii_lowercase();
        if !matches!(ts.as_str(), "" | "sigmoid") {
            return unsupported(format!(
                "timestep_type {:?} — depth runs one forward at t = num_train_timesteps",
                cfg.timestep_type
            ));
        }
        let tb = cfg.timestep_bias.trim().to_ascii_lowercase();
        if !matches!(tb.as_str(), "" | "balanced") {
            return unsupported(format!(
                "timestep_bias {:?} — depth runs one forward at t = num_train_timesteps",
                cfg.timestep_bias
            ));
        }
        if !cfg.sample_prompts.is_empty() {
            return unsupported(
                "sample_prompts — depth previews predict dataset items (preview_items), they take \
                 no prompt"
                    .into(),
            );
        }
        if !cfg.lora_target_modules.is_empty() {
            return unsupported("lora_target_modules — depth training is a full fine-tune".into());
        }
        if cfg.trigger_word.is_some() || !req.trigger_words.is_empty() {
            return unsupported(
                "trigger words — the depth task is conditioned on the empty prompt".into(),
            );
        }
        let mixed_precision = match cfg.train_dtype.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => MixedPrecision::Bf16,
            "f32" | "fp32" | "float32" | "no" => MixedPrecision::Fp32,
            other => return unsupported(format!("train_dtype {other:?} (bf16 | fp32)")),
        };
        let schedule = match cfg.lr_scheduler {
            LrSchedule::Constant => LrShape::Constant,
            LrSchedule::Cosine => LrShape::Cosine,
            LrSchedule::Linear => {
                return unsupported("lr_scheduler linear (constant | cosine)".into())
            }
        };
        let kind = match cfg.optimizer.trim().to_ascii_lowercase().as_str() {
            "adamw" => OptimizerKind::AdamW,
            "muon" => OptimizerKind::Muon,
            other => return unsupported(format!("optimizer {other:?} (adamw | muon)")),
        };
        if !(cfg.learning_rate.is_finite() && cfg.learning_rate >= 0.0) {
            return Err(Error::Msg(
                "iris depth training: learning_rate must be >= 0".into(),
            ));
        }
        if !(cfg.weight_decay.is_finite() && cfg.weight_decay >= 0.0) {
            return Err(Error::Msg(
                "iris depth training: weight_decay must be >= 0".into(),
            ));
        }
        if cfg.steps == 0 {
            return Err(Error::Msg("iris depth training: steps must be >= 1".into()));
        }
        let batch_size = cfg.batch_size.max(1) as usize;
        let grad_accum = cfg.gradient_accumulation.max(1) as usize;
        for (field, value) in [
            ("steps", cfg.steps),
            ("save_every", cfg.save_every),
            ("sample_every", cfg.sample_every),
        ] {
            if !(value as usize).is_multiple_of(grad_accum) {
                return Err(Error::Msg(format!(
                    "iris depth training: {field} = {value} micro-steps is not a multiple of \
                     gradient_accumulation = {grad_accum}"
                )));
            }
        }
        let accum = grad_accum as u32;

        // ---- size ---------------------------------------------------------------------------
        let patch = config.model.patch_size;
        let (height, width) = match opts.get("depth_size") {
            None => (cfg.resolution as usize, cfg.resolution as usize),
            Some(JsonValue::Array(a)) if a.len() == 2 => {
                let side = |v: &JsonValue| v.as_u64().map(|x| x as usize);
                match (side(&a[0]), side(&a[1])) {
                    (Some(h), Some(w)) => (h, w),
                    _ => {
                        return Err(Error::Msg(
                            "iris depth training: depth_size must be [height, width] integers"
                                .into(),
                        ))
                    }
                }
            }
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris depth training: depth_size must be [height, width], got {v}"
                )))
            }
        };
        for (name, side) in [("height", height), ("width", width)] {
            if side == 0 || !side.is_multiple_of(patch) {
                return Err(Error::Msg(format!(
                    "iris depth training: the training {name} {side} must be a positive multiple \
                     of the backbone patch_size {patch} (resolution / depth_size)"
                )));
            }
        }
        let hflip_prob = opts.f64("hflip_prob", DEFAULT_HFLIP_PROB)?;
        if !(0.0..=1.0).contains(&hflip_prob) {
            return Err(Error::Msg(
                "iris depth training: hflip_prob must be in [0, 1]".into(),
            ));
        }
        let augment = DepthAugmentPlan {
            height,
            width,
            resize: opts.choice(
                "resize",
                ResizeMode::Cover,
                &[("cover", ResizeMode::Cover), ("none", ResizeMode::None)],
            )?,
            crop: opts.choice(
                "crop",
                CropMode::Center,
                &[("center", CropMode::Center), ("random", CropMode::Random)],
            )?,
            hflip_prob,
        };

        // ---- loss ---------------------------------------------------------------------------
        let loss = DepthLossPlan {
            gradient_weight: opts.f64("gradient_weight", DEFAULT_GRADIENT_WEIGHT)?,
            low_percentile: opts.f64("low_percentile", DEFAULT_LOW_PERCENTILE)?,
            high_percentile: opts.f64("high_percentile", DEFAULT_HIGH_PERCENTILE)?,
            clip_target: opts.bool("clip_target", false)?,
            reduction: opts.choice(
                "loss_reduction",
                LossReduction::ValidPixels,
                &[
                    ("valid_pixels", LossReduction::ValidPixels),
                    ("per_image", LossReduction::PerImage),
                ],
            )?,
        };
        if loss.gradient_weight < 0.0 {
            return Err(Error::Msg(
                "iris depth training: gradient_weight must be >= 0".into(),
            ));
        }
        if !(0.0 <= loss.low_percentile
            && loss.low_percentile < loss.high_percentile
            && loss.high_percentile <= 100.0)
        {
            return Err(Error::Msg(
                "iris depth training: need 0 <= low_percentile < high_percentile <= 100".into(),
            ));
        }
        let invalid = opts.choice(
            "invalid_examples",
            InvalidPolicy::Reject,
            &[
                ("reject", InvalidPolicy::Reject),
                ("skip", InvalidPolicy::Skip),
            ],
        )?;
        let min_valid_pixels = opts
            .u64("min_valid_pixels")?
            .map(|v| v as usize)
            .unwrap_or(MIN_VALID_PIXELS_FLOOR);
        if min_valid_pixels < MIN_VALID_PIXELS_FLOOR {
            return Err(Error::Msg(format!(
                "iris depth training: min_valid_pixels must be >= {MIN_VALID_PIXELS_FLOOR}"
            )));
        }

        // ---- optimizer ----------------------------------------------------------------------
        let betas = match opts.get("betas") {
            None => DEFAULT_BETAS,
            Some(JsonValue::Array(b)) if b.len() == 2 => {
                let f = |v: &JsonValue| v.as_f64().filter(|x| (0.0..1.0).contains(x));
                match (f(&b[0]), f(&b[1])) {
                    (Some(a), Some(c)) => (a, c),
                    _ => {
                        return Err(Error::Msg(
                            "iris depth training: betas must be in [0, 1)".into(),
                        ))
                    }
                }
            }
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris depth training: betas must be a pair, got {v}"
                )))
            }
        };
        let muon_momentum = opts.f64("muon_momentum", 0.95)?;
        if !(0.0..1.0).contains(&muon_momentum) {
            return Err(Error::Msg(
                "iris depth training: muon_momentum must be in [0, 1)".into(),
            ));
        }
        let muon_adjust_lr = opts.choice(
            "muon_adjust_lr",
            MuonAdjustLr::RmsNorm,
            &[
                ("rms_norm", MuonAdjustLr::RmsNorm),
                ("spectral_norm", MuonAdjustLr::SpectralNorm),
                ("none", MuonAdjustLr::None),
            ],
        )?;
        let auto_lr = opts.string("auto_lr")?.unwrap_or_else(|| "none".into());
        let base_batch_size = opts.u64("base_batch_size")?.unwrap_or(256) as usize;
        if base_batch_size == 0 {
            return Err(Error::Msg(
                "iris depth training: base_batch_size must be >= 1".into(),
            ));
        }
        let base_lr = cfg.learning_rate as f64;
        let lr = scale_lr(base_lr, &auto_lr, batch_size * grad_accum, base_batch_size)?;
        let optimizer = OptimizerPlan {
            kind,
            base_lr,
            lr,
            betas,
            weight_decay: cfg.weight_decay as f64,
            muon_momentum,
            muon_nesterov: opts.bool("muon_nesterov", true)?,
            muon_adjust_lr,
            auto_lr,
            base_batch_size,
        };
        let gradient_clip = opts.f64("gradient_clip", DEFAULT_GRADIENT_CLIP)?;
        if gradient_clip <= 0.0 {
            return Err(Error::Msg(
                "iris depth training: gradient_clip must be > 0".into(),
            ));
        }
        let ema = if opts.bool("ema_enabled", false)? {
            let d = opts.f64("ema_decay", 0.9999)?;
            if !(0.0..=1.0).contains(&d) {
                return Err(Error::Msg(
                    "iris depth training: ema_decay must be in [0, 1]".into(),
                ));
            }
            Some(d)
        } else {
            if opts.get("ema_decay").is_some() {
                return Err(Error::Msg(
                    "iris depth training: ema_decay needs ema_enabled".into(),
                ));
            }
            None
        };
        let num_epochs = opts.u64("num_epochs")?.map(|v| v as u32);
        if num_epochs == Some(0) {
            return Err(Error::Msg(
                "iris depth training: num_epochs must be >= 1".into(),
            ));
        }
        let resume_data_policy = opts.choice(
            "resume_data_policy",
            ResumeDataPolicy::Exact,
            &[
                ("exact", ResumeDataPolicy::Exact),
                ("new_phase", ResumeDataPolicy::NewPhase),
            ],
        )?;
        let preview_weights = weights_select(&opts, "preview_weights", WeightsSelect::Raw)?;
        let export_weights = weights_select(&opts, "export_weights", WeightsSelect::Raw)?;
        for (key, w) in [
            ("preview_weights", preview_weights),
            ("export_weights", export_weights),
        ] {
            if w == WeightsSelect::Ema && ema.is_none() {
                return Err(Error::Msg(format!(
                    "iris depth training: {key} ema needs ema_enabled"
                )));
            }
        }
        let milestone_steps = match opts.get("milestone_steps") {
            None => Vec::new(),
            Some(JsonValue::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        Error::Msg("iris depth training: milestone_steps must be integers".into())
                    })
                })
                .collect::<Result<_>>()?,
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris depth training: milestone_steps must be a list, got {v}"
                )))
            }
        };
        let preview_items = opts
            .u64("preview_items")?
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_PREVIEW_ITEMS);
        Ok(Self {
            augment,
            loss,
            batch_size,
            grad_accum,
            max_steps: cfg.steps / accum,
            num_epochs,
            seed: cfg.seed,
            mixed_precision,
            optimizer,
            schedule,
            warmup_steps: cfg.lr_warmup_steps.div_ceil(accum) as u64,
            gradient_clip,
            ema,
            nan_loss_tolerance: opts.u64("nan_loss_tolerance")?.unwrap_or(20) as u32,
            invalid,
            min_valid_pixels,
            reducer_init: opts.choice(
                "reducer_init",
                ReducerInit::TorchDefault,
                &[
                    ("torch_default", ReducerInit::TorchDefault),
                    ("channel_mean", ReducerInit::ChannelMean),
                ],
            )?,
            load_from: opts.string("load_from")?.map(PathBuf::from),
            save_every: cfg.save_every / accum,
            keep_last_checkpoints: opts.u64("keep_last_checkpoints")?.unwrap_or(0) as usize,
            milestone_steps,
            preview: DepthPreviewPlan {
                every: if preview_items == 0 {
                    0
                } else {
                    cfg.sample_every / accum
                },
                items: preview_items,
                weights: preview_weights,
            },
            resume: cfg.resume,
            resume_from: opts.string("resume_from")?.map(PathBuf::from),
            resume_data_policy,
            override_lr_on_resume: opts.bool("override_lr_on_resume", false)?,
            export_weights,
        })
    }

    /// The identity any resume must match (the trained state's shape and routing).
    pub fn state_identity(&self) -> String {
        format!(
            "task=depth;artifact=full;optimizer={};ema={}",
            self.optimizer.kind.as_str(),
            self.ema.is_some()
        )
    }

    /// The data-shaping identity an `exact` resume additionally requires.
    pub fn data_identity(&self) -> String {
        let a = &self.augment;
        let l = &self.loss;
        format!(
            "size={}x{};resize={:?};crop={:?};hflip={};batch={};accum={};seed={};grad_w={};\
             pct={}-{};clip_target={};reduction={:?};invalid={:?};min_valid={}",
            a.height,
            a.width,
            a.resize,
            a.crop,
            a.hflip_prob,
            self.batch_size,
            self.grad_accum,
            self.seed,
            l.gradient_weight,
            l.low_percentile,
            l.high_percentile,
            l.clip_target,
            l.reduction,
            self.invalid,
            self.min_valid_pixels
        )
    }
}

/// Check a request's items before any work: every item names its depth target, carries no
/// caption, control image, references or subject mask, and no other per-item option.
pub fn validate_depth_items(req: &TrainingRequest) -> Result<()> {
    for item in &req.items {
        let name = item.image_path.display();
        if !item.caption.trim().is_empty() {
            return Err(Error::Unsupported(format!(
                "iris depth training: item {name} has a caption; the depth task is conditioned on \
                 the empty prompt and reads no caption (send an empty caption)"
            )));
        }
        if item.control_image_path.is_some() || item.subject_mask_path.is_some() {
            return Err(Error::Unsupported(format!(
                "iris depth training: item {name} carries a control image / subject mask; a depth \
                 item is an RGB image + its depth target (model_options.{ITEM_DEPTH_KEY})"
            )));
        }
        if let Some(key) = item.model_options.keys().find(|k| *k != ITEM_DEPTH_KEY) {
            return Err(Error::Unsupported(format!(
                "iris depth training: item {name} option {key:?} is not read (only \
                 {ITEM_DEPTH_KEY:?})"
            )));
        }
        DepthItemSpec::from_item(item)?;
    }
    Ok(())
}

// =============================================================================================
// Item files
// =============================================================================================

/// One item's depth target description (`model_options["depth"]`).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthItemSpec {
    pub path: PathBuf,
    pub mask_path: Option<PathBuf>,
    pub scale: f64,
    pub min_depth: f64,
    pub max_depth: Option<f64>,
}

const ITEM_KEYS: [&str; 5] = ["path", "mask_path", "scale", "min_depth", "max_depth"];

impl DepthItemSpec {
    pub fn from_item(item: &TrainingItem) -> Result<Self> {
        let name = item.image_path.display();
        let obj = match item.model_options.get(ITEM_DEPTH_KEY) {
            Some(JsonValue::Object(m)) => m,
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris depth training: item {name} model_options.{ITEM_DEPTH_KEY} must be an \
                     object, got {other}"
                )))
            }
            None => {
                return Err(Error::Msg(format!(
                    "iris depth training: item {name} has no depth target \
                     (model_options.{ITEM_DEPTH_KEY}.path)"
                )))
            }
        };
        if let Some(k) = obj.keys().find(|k| !ITEM_KEYS.contains(&k.as_str())) {
            return Err(Error::Unsupported(format!(
                "iris depth training: item {name} depth key {k:?} is unknown (known: {})",
                ITEM_KEYS.join(", ")
            )));
        }
        let path = obj.get("path").and_then(JsonValue::as_str).ok_or_else(|| {
            Error::Msg(format!(
                "iris depth training: item {name} depth.path must be a string"
            ))
        })?;
        let mask_path = match obj.get("mask_path") {
            None | Some(JsonValue::Null) => None,
            Some(v) => Some(PathBuf::from(v.as_str().ok_or_else(|| {
                Error::Msg(format!(
                    "iris depth training: item {name} depth.mask_path must be a string"
                ))
            })?)),
        };
        let num = |k: &str| -> Result<Option<f64>> {
            match obj.get(k) {
                None | Some(JsonValue::Null) => Ok(None),
                Some(v) => v
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .map(Some)
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "iris depth training: item {name} depth.{k} must be a finite number"
                        ))
                    }),
            }
        };
        let scale = num("scale")?.unwrap_or(1.0);
        if scale <= 0.0 {
            return Err(Error::Msg(format!(
                "iris depth training: item {name} depth.scale must be > 0"
            )));
        }
        let min_depth = num("min_depth")?.unwrap_or(0.0);
        if min_depth < 0.0 {
            return Err(Error::Msg(format!(
                "iris depth training: item {name} depth.min_depth must be >= 0"
            )));
        }
        let max_depth = num("max_depth")?;
        if max_depth.is_some_and(|m| m <= min_depth) {
            return Err(Error::Msg(format!(
                "iris depth training: item {name} depth.max_depth must exceed min_depth"
            )));
        }
        Ok(Self {
            path: PathBuf::from(path),
            mask_path,
            scale,
            min_depth,
            max_depth,
        })
    }
}

/// A 2-D numeric array read from a `.npy` or grayscale `.png` (row-major `h × w`, as f64).
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f64>,
}

/// Read a numeric plane: NumPy `.npy` (v1–v3, C order, `f4 f8 u1 u2 i4 u4 i2 b1`, shape `[H, W]`,
/// `[1, H, W]` or `[H, W, 1]`) or an 8-/16-bit grayscale PNG (by extension).
pub fn read_plane(path: &Path) -> Result<Plane> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match ext.as_str() {
        "npy" => read_npy(path),
        "png" => read_png_plane(path),
        other => Err(Error::Msg(format!(
            "iris depth training: {} — unsupported depth/mask format {other:?} (.npy | .png)",
            path.display()
        ))),
    }
}

fn read_png_plane(path: &Path) -> Result<Plane> {
    let img = image::open(path).map_err(|e| {
        Error::Msg(format!(
            "iris depth training: decode {}: {e}",
            path.display()
        ))
    })?;
    let (width, height) = (img.width() as usize, img.height() as usize);
    let values: Vec<f64> = match img {
        image::DynamicImage::ImageLuma16(b) => b.into_raw().into_iter().map(f64::from).collect(),
        image::DynamicImage::ImageLuma8(b) => b.into_raw().into_iter().map(f64::from).collect(),
        other => {
            return Err(Error::Msg(format!(
            "iris depth training: {} is a {:?} PNG; a depth / mask plane must be single-channel \
                 (8- or 16-bit grayscale)",
            path.display(),
            other.color()
        )))
        }
    };
    Ok(Plane {
        width,
        height,
        values,
    })
}

/// A minimal `.npy` reader (no pickles, no object arrays).
pub fn read_npy(path: &Path) -> Result<Plane> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Msg(format!("iris depth training: read {}: {e}", path.display())))?;
    parse_npy(&bytes)
        .map_err(|e| Error::Msg(format!("iris depth training: {}: {e}", path.display())))
}

fn parse_npy(bytes: &[u8]) -> std::result::Result<Plane, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated header".into());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + header_len)
            .ok_or("truncated header")?,
    )
    .map_err(|_| "header is not text")?;
    let field = |key: &str| -> std::result::Result<&str, String> {
        let at = header
            .find(&format!("'{key}'"))
            .ok_or(format!("header has no {key}"))?;
        let rest = &header[at + key.len() + 2..];
        let colon = rest.find(':').ok_or("malformed header")?;
        Ok(rest[colon + 1..].trim_start())
    };
    let descr = field("descr")?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|d| d.split('\'').next())
        .ok_or("malformed descr")?;
    if field("fortran_order")?.starts_with("True") {
        return Err("fortran_order arrays are not supported (save C order)".into());
    }
    let shape_text = field("shape")?;
    let shape_text = shape_text
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or("malformed shape")?;
    let shape: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| format!("bad dim {s:?}")))
        .collect::<std::result::Result<_, _>>()?;
    let (height, width) = match shape.as_slice() {
        [h, w] | [1, h, w] | [h, w, 1] => (*h, *w),
        other => return Err(format!("shape {other:?} is not a plane ([H, W])")),
    };
    let n = height * width;
    let data = &bytes[start + header_len..];
    let little = |d: &str| d.starts_with('<') || d.starts_with('|');
    if !little(descr) {
        return Err(format!("dtype {descr:?} is not little-endian"));
    }
    let kind = &descr[1..];
    let size = match kind {
        "f8" | "i8" | "u8" => 8,
        "f4" | "i4" | "u4" => 4,
        "u2" | "i2" | "f2" => 2,
        "u1" | "i1" | "b1" => 1,
        other => return Err(format!("unsupported dtype {other:?}")),
    };
    if data.len() < n * size {
        return Err(format!(
            "holds {} data bytes, the {height}x{width} {kind} plane needs {}",
            data.len(),
            n * size
        ));
    }
    let values: Vec<f64> = data[..n * size]
        .chunks_exact(size)
        .map(|c| match kind {
            "f8" => f64::from_le_bytes(c.try_into().unwrap()),
            "i8" => i64::from_le_bytes(c.try_into().unwrap()) as f64,
            "u8" => u64::from_le_bytes(c.try_into().unwrap()) as f64,
            "f4" => f32::from_le_bytes(c.try_into().unwrap()) as f64,
            "i4" => i32::from_le_bytes(c.try_into().unwrap()) as f64,
            "u4" => u32::from_le_bytes(c.try_into().unwrap()) as f64,
            "u2" => u16::from_le_bytes(c.try_into().unwrap()) as f64,
            "i2" => i16::from_le_bytes(c.try_into().unwrap()) as f64,
            "f2" => half_to_f64(u16::from_le_bytes(c.try_into().unwrap())),
            "i1" => c[0] as i8 as f64,
            _ => c[0] as f64,
        })
        .collect();
    Ok(Plane {
        width,
        height,
        values,
    })
}

fn half_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 if frac == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
    }
}

/// One aligned example at its source size.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthSample {
    pub width: usize,
    pub height: usize,
    /// `h · w · 3` RGB8.
    pub rgb: Vec<u8>,
    /// `h · w` scaled depth (`stored · scale`; meaningful only where `valid`).
    pub depth: Vec<f32>,
    pub valid: Vec<bool>,
}

/// Decode one item: the RGB image, its depth plane and (optional) mask — all the same `W × H` —
/// and its validity (see the module docs).
pub fn load_depth_sample(item: &TrainingItem) -> Result<DepthSample> {
    let spec = DepthItemSpec::from_item(item)?;
    let img = image::open(&item.image_path)
        .map_err(|e| {
            Error::Msg(format!(
                "iris depth training: decode {}: {e}",
                item.image_path.display()
            ))
        })?
        .to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let plane = read_plane(&spec.path)?;
    let same = |p: &Plane, what: &str, path: &Path| -> Result<()> {
        if (p.width, p.height) != (width, height) {
            return Err(Error::Msg(format!(
                "iris depth training: {what} {} is {}x{}, its image {} is {width}x{height} — RGB, \
                 depth and mask must be aligned pixel for pixel",
                path.display(),
                p.width,
                p.height,
                item.image_path.display()
            )));
        }
        Ok(())
    };
    same(&plane, "depth", &spec.path)?;
    let mask = match &spec.mask_path {
        Some(p) => {
            let m = read_plane(p)?;
            same(&m, "mask", p)?;
            Some(m.values)
        }
        None => None,
    };
    let mut depth = Vec::with_capacity(width * height);
    let mut valid = Vec::with_capacity(width * height);
    for (i, &v) in plane.values.iter().enumerate() {
        let d = v * spec.scale;
        let ok = mask.as_ref().is_none_or(|m| m[i] != 0.0)
            && d.is_finite()
            && d > spec.min_depth
            && spec.max_depth.is_none_or(|m| d <= m);
        depth.push(if ok { d as f32 } else { 0.0 });
        valid.push(ok);
    }
    Ok(DepthSample {
        width,
        height,
        rgb: img.into_raw(),
        depth,
        valid,
    })
}

// =============================================================================================
// Target normalization
// =============================================================================================

/// The per-image normalized target: `log(d)` over the valid pixels with the `low`/`high`
/// percentiles (numpy `linear`) mapped to −1 / +1 — `2·(log d − p_lo)/(p_hi − p_lo) − 1` — optionally
/// clamped to [−1, 1]; invalid pixels are 0. `None` when fewer than `min_valid` pixels are valid or
/// the percentile range is below [`DEGENERATE_LOG_RANGE`].
pub fn normalize_log_depth(
    depth: &[f32],
    valid: &[bool],
    loss: &DepthLossPlan,
    min_valid: usize,
) -> Option<Vec<f32>> {
    let logs: Vec<f32> = depth
        .iter()
        .zip(valid)
        .filter(|(_, &v)| v)
        .map(|(&d, _)| d.ln())
        .collect();
    if logs.len() < min_valid.max(MIN_VALID_PIXELS_FLOOR) {
        return None;
    }
    let mut sorted = logs;
    sorted.sort_by(f32::total_cmp);
    let lo = crate::iris::depth::percentile(&sorted, loss.low_percentile) as f64;
    let hi = crate::iris::depth::percentile(&sorted, loss.high_percentile) as f64;
    let range = hi - lo;
    if range.is_nan() || range < DEGENERATE_LOG_RANGE {
        return None;
    }
    Some(
        depth
            .iter()
            .zip(valid)
            .map(|(&d, &v)| {
                if !v {
                    return 0.0;
                }
                let t = 2.0 * ((d.ln() as f64) - lo) / range - 1.0;
                let t = if loss.clip_target {
                    t.clamp(-1.0, 1.0)
                } else {
                    t
                };
                t as f32
            })
            .collect(),
    )
}

/// Why an item is invalid for training (`None` = trainable): see [`normalize_log_depth`].
pub fn invalid_reason(
    sample: &DepthSample,
    loss: &DepthLossPlan,
    min_valid: usize,
) -> Option<String> {
    let n = sample.valid.iter().filter(|&&v| v).count();
    if n < min_valid {
        return Some(format!(
            "{n} valid depth pixels (min_valid_pixels = {min_valid})"
        ));
    }
    normalize_log_depth(&sample.depth, &sample.valid, loss, min_valid)
        .is_none()
        .then(|| {
            format!(
                "its valid log depths have a degenerate {}–{} percentile range (< \
                 {DEGENERATE_LOG_RANGE}): no relative normalization exists",
                loss.low_percentile, loss.high_percentile
            )
        })
}

// =============================================================================================
// Paired transforms and batches
// =============================================================================================

/// The covering resize of a `w × h` source for a `tw × th` training size: the source as is when it
/// already is that size, else scaled by `max(th/h, tw/w)` with each side rounded (half-even) and
/// at least the target.
pub fn cover_size(h: usize, w: usize, th: usize, tw: usize) -> (usize, usize) {
    if (h, w) == (th, tw) {
        return (h, w);
    }
    let s = (th as f64 / h as f64).max(tw as f64 / w as f64);
    let side = |v: usize, t: usize| ((v as f64 * s).round_ties_even() as usize).max(t);
    (side(h, th), side(w, tw))
}

/// The size a `w × h` source has before the crop under `aug.resize`: [`cover_size`], or the source
/// itself (`resize: none`, which needs a source at least the training size).
pub fn covered_size(h: usize, w: usize, aug: &DepthAugmentPlan) -> Result<(usize, usize)> {
    match aug.resize {
        ResizeMode::Cover => Ok(cover_size(h, w, aug.height, aug.width)),
        ResizeMode::None if h >= aug.height && w >= aug.width => Ok((h, w)),
        ResizeMode::None => Err(Error::Msg(format!(
            "iris depth training: a {w}x{h} source is smaller than the {}x{} training size and \
             resize is none",
            aug.width, aug.height
        ))),
    }
}

/// One row's random draws (positional: the batch key's stream, rows in order; a random crop draws
/// `top` then `left`, every row draws its flip).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDraw {
    pub top: usize,
    pub left: usize,
    pub flip: bool,
}

fn round_half_even(v: f64) -> usize {
    v.round_ties_even() as usize
}

/// Draw one row's crop offsets and flip for a covered `rh × rw` source.
pub fn draw_row(rng: &mut HostRng, aug: &DepthAugmentPlan, rh: usize, rw: usize) -> RowDraw {
    let (top, left) = match aug.crop {
        CropMode::Center => (
            round_half_even((rh - aug.height) as f64 / 2.0),
            round_half_even((rw - aug.width) as f64 / 2.0),
        ),
        CropMode::Random => {
            let top = rng.below(rh - aug.height + 1);
            let left = rng.below(rw - aug.width + 1);
            (top, left)
        }
    };
    let flip = rng.uniform() < aug.hflip_prob;
    RowDraw { top, left, flip }
}

/// PIL `NEAREST` source index of output `dst` (`floor((dst + ½) · in/out)`, clamped).
fn nearest_index(dst: usize, in_len: usize, out_len: usize) -> usize {
    (((dst as f64 + 0.5) * in_len as f64 / out_len as f64) as usize).min(in_len - 1)
}

/// One transformed row at the training size: RGB `[3, h, w]` in `[−1, 1]`, the scaled depth, the
/// validity (all aligned).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthRow {
    pub rgb_chw: Vec<f32>,
    pub depth: Vec<f32>,
    pub valid: Vec<bool>,
}

/// The paired spatial transform: the covering resize (RGB PIL-bicubic; depth and mask nearest, so
/// no depth is ever invented across a hole or an edge), the crop at `draw`, then the flip — the same
/// geometry for all three planes.
pub fn transform_sample(
    sample: &DepthSample,
    aug: &DepthAugmentPlan,
    draw: RowDraw,
) -> Result<DepthRow> {
    let (h, w) = (sample.height, sample.width);
    let (rh, rw) = covered_size(h, w, aug)?;
    let resized = (rh, rw) != (h, w);
    let rgb: Vec<f32> = if resized {
        crate::imageops::resize_bicubic_u8(&sample.rgb, h, w, rh, rw)?
    } else {
        sample.rgb.iter().map(|&v| v as f32).collect()
    };
    let rows: Vec<usize> = (0..rh).map(|y| nearest_index(y, h, rh)).collect();
    let cols: Vec<usize> = (0..rw).map(|x| nearest_index(x, w, rw)).collect();
    let (th, tw) = (aug.height, aug.width);
    if draw.top + th > rh || draw.left + tw > rw {
        return Err(Error::Msg(format!(
            "iris depth training: crop {}+{th} x {}+{tw} outside the {rh}x{rw} covered source",
            draw.top, draw.left
        )));
    }
    let mut rgb_chw = vec![0f32; 3 * th * tw];
    let mut depth = vec![0f32; th * tw];
    let mut valid = vec![false; th * tw];
    for y in 0..th {
        for x in 0..tw {
            let sx = if draw.flip { tw - 1 - x } else { x };
            let (ry, rx) = (draw.top + y, draw.left + sx);
            let o = y * tw + x;
            for c in 0..3 {
                rgb_chw[c * th * tw + o] = rgb[(ry * rw + rx) * 3 + c] / 127.5 - 1.0;
            }
            let src = if resized {
                rows[ry] * w + cols[rx]
            } else {
                ry * w + rx
            };
            depth[o] = sample.depth[src];
            valid[o] = sample.valid[src];
        }
    }
    Ok(DepthRow {
        rgb_chw,
        depth,
        valid,
    })
}

/// One micro-batch, host-side (row-major, `B` rows of the training size).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthBatch {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    /// `[B, 3, H, W]` in `[−1, 1]`.
    pub rgb: Vec<f32>,
    /// `[B, 1, H, W]` normalized relative log depth (0 where invalid).
    pub target: Vec<f32>,
    /// `[B, 1, H, W]` 1.0 = valid, 0.0 = masked out.
    pub valid: Vec<f32>,
    /// Valid pixels per row (0 for a row whose normalization was undefined after the crop).
    pub valid_count: Vec<usize>,
}

impl DepthBatch {
    /// The per-row weights `w_i` of the loss `Σ_i w_i · num_i`, where `num_i` is row `i`'s masked sum
    /// `Σ_{p∈M_i} |d̂−d| + λ‖∇d̂−∇d‖₁`: `1/|M|` under [`LossReduction::ValidPixels`], `1/(|M_i| · k)`
    /// over the `k` rows with support under [`LossReduction::PerImage`] (0 for a row without).
    pub fn row_weights(&self, reduction: LossReduction) -> Vec<f32> {
        match reduction {
            LossReduction::ValidPixels => {
                let total: usize = self.valid_count.iter().sum();
                let w = if total == 0 { 0.0 } else { 1.0 / total as f64 };
                vec![w as f32; self.batch]
            }
            LossReduction::PerImage => {
                let k = self.valid_count.iter().filter(|&&c| c > 0).count();
                self.valid_count
                    .iter()
                    .map(|&c| {
                        if c == 0 {
                            0.0
                        } else {
                            (1.0 / (c as f64 * k as f64)) as f32
                        }
                    })
                    .collect()
            }
        }
    }
}

/// Assemble rows (already transformed) into a batch, normalizing each row's target.
pub fn batch_from_rows(
    rows: Vec<DepthRow>,
    aug: &DepthAugmentPlan,
    loss: &DepthLossPlan,
    min_valid: usize,
) -> DepthBatch {
    let (h, w) = (aug.height, aug.width);
    let b = rows.len();
    let mut out = DepthBatch {
        batch: b,
        height: h,
        width: w,
        rgb: Vec::with_capacity(b * 3 * h * w),
        target: Vec::with_capacity(b * h * w),
        valid: Vec::with_capacity(b * h * w),
        valid_count: Vec::with_capacity(b),
    };
    for row in rows {
        out.rgb.extend_from_slice(&row.rgb_chw);
        match normalize_log_depth(&row.depth, &row.valid, loss, min_valid) {
            Some(t) => {
                out.target.extend_from_slice(&t);
                out.valid
                    .extend(row.valid.iter().map(|&v| if v { 1.0f32 } else { 0.0 }));
                out.valid_count
                    .push(row.valid.iter().filter(|&&v| v).count());
            }
            None => {
                // The crop left this row without a normalizable support: it contributes nothing.
                out.target.extend(std::iter::repeat_n(0.0, h * w));
                out.valid.extend(std::iter::repeat_n(0.0, h * w));
                out.valid_count.push(0);
            }
        }
    }
    out
}

/// Micro-batch `pos` of `epoch`: walk positions → `kept` dataset indices, decode, the paired
/// transforms with the batch key's draws ([`batch_seed`]), normalized targets.
pub fn assemble_batch(
    req: &TrainingRequest,
    plan: &DepthTrainPlan,
    kept: &[usize],
    walk: &DataWalk,
    epoch: u32,
    pos: usize,
) -> Result<DepthBatch> {
    let mut rng = HostRng::new(batch_seed(plan.seed, epoch, pos));
    let mut rows = Vec::new();
    for i in walk.batch(pos) {
        let sample = load_depth_sample(&req.items[kept[i]])?;
        let (rh, rw) = covered_size(sample.height, sample.width, &plan.augment)?;
        let draw = draw_row(&mut rng, &plan.augment, rh, rw);
        rows.push(transform_sample(&sample, &plan.augment, draw)?);
    }
    Ok(batch_from_rows(
        rows,
        &plan.augment,
        &plan.loss,
        plan.min_valid_pixels,
    ))
}

/// The deterministic preview row of an item: center crop, no flip.
pub fn preview_row(item: &TrainingItem, plan: &DepthTrainPlan) -> Result<DepthRow> {
    let sample = load_depth_sample(item)?;
    let aug = DepthAugmentPlan {
        crop: CropMode::Center,
        hflip_prob: 0.0,
        ..plan.augment
    };
    let (rh, rw) = covered_size(sample.height, sample.width, &aug)?;
    let draw = RowDraw {
        top: round_half_even((rh - aug.height) as f64 / 2.0),
        left: round_half_even((rw - aug.width) as f64 / 2.0),
        flip: false,
    };
    transform_sample(&sample, &aug, draw)
}

/// The dataset pass: decode and check every item (one `on_item(i)` call each, after it is checked;
/// cancellation is polled before each). Returns the dataset indices the walk trains on: all of them
/// under [`InvalidPolicy::Reject`] (the first invalid item refuses the run), the valid ones under
/// [`InvalidPolicy::Skip`] (each skip is warned on stderr; nothing left is a refusal).
pub fn depth_dataset_pass(
    req: &TrainingRequest,
    plan: &DepthTrainPlan,
    mut on_item: impl FnMut(usize),
) -> Result<Vec<usize>> {
    let mut kept = Vec::with_capacity(req.items.len());
    for (i, item) in req.items.iter().enumerate() {
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let sample = load_depth_sample(item)?;
        covered_size(sample.height, sample.width, &plan.augment)
            .map_err(|e| Error::Msg(format!("{e} (item {})", item.image_path.display())))?;
        match invalid_reason(&sample, &plan.loss, plan.min_valid_pixels) {
            None => kept.push(i),
            Some(why) => match plan.invalid {
                InvalidPolicy::Reject => {
                    return Err(Error::Msg(format!(
                        "iris depth training: item {} is not a valid depth example: {why} \
                         (invalid_examples = reject; use skip to drop such items)",
                        item.image_path.display()
                    )))
                }
                InvalidPolicy::Skip => eprintln!(
                    "iris depth training: warning — skipping item {}: {why}",
                    item.image_path.display()
                ),
            },
        }
        on_item(i);
    }
    if kept.is_empty() {
        return Err(Error::Msg(
            "iris depth training: no item of the dataset is a valid depth example".into(),
        ));
    }
    Ok(kept)
}

/// The dataset fingerprint of a depth request: [`crate::train::resume::request_fingerprint`] plus
/// every item's depth options and the bytes of its depth and mask files.
pub fn depth_dataset_fingerprint(req: &TrainingRequest) -> Result<String> {
    let base = crate::train::resume::request_fingerprint(req)?;
    let mut h = Sha256::new();
    h.update(base.as_bytes());
    for item in &req.items {
        let spec = DepthItemSpec::from_item(item)?;
        if let Some(m) = item.model_options.get(ITEM_DEPTH_KEY) {
            h.update(m.to_string().as_bytes());
        }
        for p in std::iter::once(&spec.path).chain(spec.mask_path.as_ref()) {
            let bytes = std::fs::read(p).map_err(|e| {
                Error::Msg(format!("iris depth training: read {}: {e}", p.display()))
            })?;
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
        h.update([0u8]);
    }
    Ok(format!("{:x}", h.finalize()))
}

// =============================================================================================
// Loss (host reference — the backends compute the same on tensors)
// =============================================================================================

/// Row `i`'s masked sum `num_i = Σ_{p∈M_i} |d̂_p − d_p| + λ (Σ |Δx d̂ − Δx d| + Σ |Δy d̂ − Δy d|)`
/// with forward differences counted where both pixels are valid. f64 host reference.
pub fn depth_loss_row_sums(pred: &[f32], batch: &DepthBatch, gradient_weight: f64) -> Vec<f64> {
    let (h, w) = (batch.height, batch.width);
    (0..batch.batch)
        .map(|b| {
            let o = b * h * w;
            let (p, t, v) = (
                &pred[o..o + h * w],
                &batch.target[o..o + h * w],
                &batch.valid[o..o + h * w],
            );
            let mut value = 0f64;
            let mut grad = 0f64;
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    value += (p[i] as f64 - t[i] as f64).abs() * v[i] as f64;
                    if x + 1 < w {
                        let j = i + 1;
                        let d = (p[j] as f64 - p[i] as f64) - (t[j] as f64 - t[i] as f64);
                        grad += d.abs() * (v[i] * v[j]) as f64;
                    }
                    if y + 1 < h {
                        let j = i + w;
                        let d = (p[j] as f64 - p[i] as f64) - (t[j] as f64 - t[i] as f64);
                        grad += d.abs() * (v[i] * v[j]) as f64;
                    }
                }
            }
            value + gradient_weight * grad
        })
        .collect()
}

/// Eq. 5 on a batch (host reference): `Σ_i w_i · num_i` ([`DepthBatch::row_weights`]).
pub fn depth_loss_host(pred: &[f32], batch: &DepthBatch, loss: &DepthLossPlan) -> f64 {
    let sums = depth_loss_row_sums(pred, batch, loss.gradient_weight);
    batch
        .row_weights(loss.reduction)
        .iter()
        .zip(sums)
        .map(|(&w, s)| w as f64 * s)
        .sum()
}

/// `∂ loss / ∂ d̂` of [`depth_loss_host`] (subgradient `sign(0) = 0`, torch's `abs` backward).
pub fn depth_loss_grad_host(pred: &[f32], batch: &DepthBatch, loss: &DepthLossPlan) -> Vec<f64> {
    let (h, w) = (batch.height, batch.width);
    let weights = batch.row_weights(loss.reduction);
    let sign = |v: f64| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let mut g = vec![0f64; pred.len()];
    for (b, &rw) in weights.iter().enumerate() {
        let o = b * h * w;
        let rw = rw as f64;
        for y in 0..h {
            for x in 0..w {
                let i = o + y * w + x;
                let (p, t, v) = (&pred, &batch.target, &batch.valid);
                g[i] += rw * sign(p[i] as f64 - t[i] as f64) * v[i] as f64;
                for j in [(x + 1 < w).then_some(i + 1), (y + 1 < h).then_some(i + w)]
                    .into_iter()
                    .flatten()
                {
                    let d = (p[j] as f64 - p[i] as f64) - (t[j] as f64 - t[i] as f64);
                    let s = rw * loss.gradient_weight * sign(d) * (v[i] * v[j]) as f64;
                    g[j] += s;
                    g[i] -= s;
                }
            }
        }
    }
    g
}

// =============================================================================================
// Model tensors, init, export
// =============================================================================================

/// Every trained tensor of the depth model `(key, shape)`: the backbone's (the native loader's key
/// names, input projections widened by [`super::super::depth::DEPTH_EXTRA_INPUT_CHANNELS`]) plus
/// [`REDUCER_WEIGHT`] / [`REDUCER_BIAS`]. The export prefixes the backbone keys with
/// [`EXPORT_BACKBONE_PREFIX`].
pub fn depth_tensor_shapes(cfg: &ModelConfig) -> Vec<(String, Vec<usize>)> {
    let extra = crate::iris::depth::DEPTH_EXTRA_INPUT_CHANNELS;
    let p = cfg.patch_size;
    let mut out = super::backbone_tensor_shapes(cfg);
    for (k, s) in out.iter_mut() {
        match k.as_str() {
            "s_embedder.proj.weight" => s[1] = p * p * (cfg.in_channels + extra),
            "pixel_embedder.proj.weight" => s[1] = cfg.in_channels + extra,
            _ => {}
        }
    }
    out.push((REDUCER_WEIGHT.into(), vec![1, cfg.in_channels, 1, 1]));
    out.push((REDUCER_BIAS.into(), vec![1]));
    out
}

/// The input projections a parent init widens: `(key, columns per input channel)` — the patch
/// embedder's channel-major `c·p·p` vector and the per-pixel embedder. The new channel's columns are
/// appended (it is concatenated after RGB).
pub fn widened_projections(cfg: &ModelConfig) -> [(&'static str, usize); 2] {
    [
        ("s_embedder.proj.weight", cfg.patch_size * cfg.patch_size),
        ("pixel_embedder.proj.weight", 1),
    ]
}

/// The reducer's initial `([C] weight, [1] bias)` for a parent init — host draws keyed on the run
/// seed, so both backends start from the same values.
pub fn reducer_init_values(
    kind: ReducerInit,
    in_channels: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    match kind {
        ReducerInit::ChannelMean => (vec![1.0 / in_channels as f32; in_channels], vec![0.0]),
        ReducerInit::TorchDefault => {
            let bound = 1.0 / (in_channels as f64).sqrt();
            let mut rng = HostRng::new(mix_seed(&[seed, 0x6465_7074_685f_7265]));
            let mut u = || ((rng.uniform() * 2.0 - 1.0) * bound) as f32;
            let w: Vec<f32> = (0..in_channels).map(|_| u()).collect();
            let b = vec![u()];
            (w, b)
        }
    }
}

/// The depth export's `config.yaml` (`scripts/export_downstream.py`): the source config's `model`,
/// `text_encoder` and `flow` sections verbatim (every other top-level section — a previous `task`,
/// training sections — dropped) plus `task: {name: depth}`. Checked to resolve as a depth export.
pub fn depth_export_config_yaml(source: &str) -> Result<String> {
    let mut out = String::new();
    let mut keep = false;
    for line in source.lines() {
        let top = !line.is_empty() && !line.starts_with([' ', '\t', '-', '#']);
        if top {
            let key = line.split(':').next().unwrap_or("").trim();
            keep = matches!(key, "model" | "text_encoder" | "flow");
        }
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str("task:\n  name: depth\n");
    let settings = crate::iris::downstream::parse_task_settings(&out)?;
    if settings.map(|s| s.name) != Some("depth".into()) {
        return Err(Error::Msg(
            "iris depth export: the rendered config.yaml does not carry task depth".into(),
        ));
    }
    IrisConfig::parse(&out)?;
    Ok(out)
}

/// The depth export's `model.safetensors` metadata: `format = pt` plus the trained-model stamps
/// ([`super::task_model_metadata`] with `irisTask = depth`).
pub fn depth_export_metadata(
    weights: WeightsSelect,
    steps: u64,
    base_identity: &str,
) -> BTreeMap<String, String> {
    super::task_model_metadata(IrisTask::Depth, weights, steps, base_identity)
}

/// Check a depth checkpoint against the resuming run: the task, the trained state and the base
/// (the base supplies the empty-prompt conditioning) always; an `exact` resume also the data
/// shaping and the dataset.
pub fn check_depth_resume(
    saved: &CheckpointState,
    plan: &DepthTrainPlan,
    dataset_fingerprint: &str,
    base_identity: &str,
) -> Result<()> {
    if saved.task != IrisTask::Depth {
        return Err(Error::Msg(format!(
            "iris depth resume: the checkpoint belongs to the {} task",
            saved.task.name()
        )));
    }
    if saved.state_identity != plan.state_identity() {
        return Err(Error::Msg(format!(
            "iris depth resume: the checkpoint trained a different state ({}) than this run ({})",
            saved.state_identity,
            plan.state_identity()
        )));
    }
    if saved.base_identity != base_identity {
        return Err(Error::Msg(
            "iris depth resume: the checkpoint was trained from a different base (its empty-prompt \
             conditioning and init differ)"
                .into(),
        ));
    }
    if plan.resume_data_policy == ResumeDataPolicy::Exact {
        if saved.data_identity != plan.data_identity() {
            return Err(Error::Msg(format!(
                "iris depth resume: data shaping differs (saved {}, requested {}); \
                 resume_data_policy new_phase starts the current dataset fresh",
                saved.data_identity,
                plan.data_identity()
            )));
        }
        if saved.dataset_fingerprint != dataset_fingerprint {
            return Err(Error::Msg(
                "iris depth resume: the dataset differs from the checkpoint's; resume_data_policy \
                 new_phase starts the current dataset fresh"
                    .into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

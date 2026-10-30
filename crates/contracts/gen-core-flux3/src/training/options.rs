//! Strict family options and the actually implemented source mechanism matrix.
//! Unknown/unsupported intent fails before loading, never becomes a different trainer.
use super::{
    policy::{optimizer_update_budget, SourceLrKind, SourceLrSchedule},
    Weighting,
};
use gen_core::{
    train::LrSchedule, train::NetworkType, train::TrainingConfig, train::TrainingRequest,
};
use gen_core::{Error, Result};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 trainer: {detail}"))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingPrecision {
    F32,
    Bf16,
}
impl TrainingPrecision {
    pub fn parse(value: &str) -> Result<Self> {
        match value {"f32"|"fp32"|"float32"=>Ok(Self::F32),"bf16"|"bfloat16"=>Ok(Self::Bf16),_=>Err(invalid("training precision must be F32 or BF16; FP16 backprop requires a native gradient scaler and is not advertised"))}
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}
/// The `TrainingConfig.model_options` key holding this family's source options (the
/// convention the other strict family trainers use, e.g. `model_options["iris"]`). Every key
/// inside it must be one this trainer implements; an unknown one is refused. Keys OUTSIDE it
/// belong to the host's product bag (SceneWorks forwards its whole `advanced` object, whose
/// common fields are already mapped onto the typed `TrainingConfig` the parse checks above).
pub const OPTIONS_KEY: &str = "flux3";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workflow {
    Captioned,
    PairedEdit,
}
/// The native prefix of the projections that process ONLY reference (image-condition) tokens.
pub const REFERENCE_ONLY_PREFIX: &str = "content_mode_blocks.image_cond.";
impl Workflow {
    /// Whether the LoRA factor named `factor` (a native target or `{native}.lora_{A,B}.weight`)
    /// is reached by this workflow's graph, i.e. receives a gradient. A captioned run has no
    /// reference tokens, so the `image_cond` projections never enter its graph: PyTorch leaves
    /// their `.grad` as `None` and AdamW skips them entirely (no moment, no step count, no
    /// decoupled decay). Candle's autograd yields the same partition (its GradStore omits them,
    /// pinned by a runner test); MLX's functional gradient returns zeros instead, so it must
    /// apply this predicate rather than step and decay factors the source never touches.
    pub fn trains(self, factor: &str) -> bool {
        match self {
            Self::PairedEdit => true,
            Self::Captioned => !factor.starts_with(REFERENCE_ONLY_PREFIX),
        }
    }
}
#[derive(Clone, Debug)]
pub struct TrainingOptions {
    pub precision: TrainingPrecision,
    pub workflow: Workflow,
    pub caption_dropout: f32,
    pub max_sequence_length: usize,
    pub center_crop: bool,
    pub random_flip: bool,
    pub weighting: Weighting,
    pub logit_mean: f32,
    pub logit_std: f32,
    pub mode_scale: f32,
    pub max_grad_norm: f32,
    pub scheduler: SourceLrKind,
    pub cycles: u32,
    pub power: f64,
}
impl TrainingOptions {
    pub fn parse(req: &TrainingRequest) -> Result<Self> {
        let cfg = &req.config;
        if cfg.rank == 0
            || !cfg.alpha.is_finite()
            || cfg.alpha <= 0.0
            || !cfg.learning_rate.is_finite()
            || cfg.learning_rate <= 0.0
            || !cfg.weight_decay.is_finite()
            || cfg.weight_decay < 0.0
            || cfg.steps == 0
            || cfg.batch_size == 0
            || cfg.gradient_accumulation == 0
        {
            return Err(invalid("rank/alpha/rate/steps/batch/accumulation must be positive and finite; decay nonnegative"));
        }
        if cfg.network_type != NetworkType::Lora || cfg.full_finetune || cfg.control_type.is_some()
        {
            return Err(invalid("only plain additive LoRA on the frozen dense base is implemented; LoKr/full fine-tune/control are not source-compatible outputs"));
        }
        if !cfg.optimizer.trim().eq_ignore_ascii_case("adamw") {
            return Err(invalid("this source-correct profile implements AdamW; 8-bit aliases/other optimizer profiles must not silently become full-precision AdamW"));
        }
        if !cfg.loss_type.trim().eq_ignore_ascii_case("mse") {
            return Err(invalid(
                "the pinned source objective is weighted velocity MSE, not MAE/another loss",
            ));
        }
        if cfg.gradient_checkpointing {
            return Err(invalid("this dense autograd profile does not yet implement activation recomputation; disable gradient checkpointing rather than silently ignoring it"));
        }
        if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() {
            return Err(invalid("in-training sample rendering is not advertised by this profile; final adapters load through ordinary generation"));
        }
        if cfg.resolution < 64 || cfg.resolution > 4096 || !cfg.resolution.is_multiple_of(16) {
            return Err(invalid(
                "training resolution must be a supported positive64..4096 multiple of16",
            ));
        }
        if req.items.is_empty() {
            return Err(invalid("training dataset is empty"));
        }
        let reference_count = req.items[0].reference_image_paths.len();
        if reference_count > 1
            || req
                .items
                .iter()
                .any(|item| item.reference_image_paths.len() != reference_count)
        {
            return Err(invalid("choose captioned examples OR uniformly paired single-reference edit examples; mixed/many-reference training is not the pinned recipe"));
        }
        if req.items.iter().any(|item| {
            item.control_image_path.is_some()
                || item.subject_mask_path.is_some()
                || !item.model_options.is_empty()
        }) {
            return Err(invalid(
                "unimplemented per-example control/mask/model options must not be ignored",
            ));
        }
        let workflow = if reference_count == 1 {
            Workflow::PairedEdit
        } else {
            Workflow::Captioned
        };
        let empty = serde_json::Map::new();
        let bag = match cfg.model_options.get(OPTIONS_KEY) {
            None => &empty,
            Some(serde_json::Value::Object(bag)) => bag,
            Some(_) => {
                return Err(invalid(format!(
                    "model_options.{OPTIONS_KEY} must be an object of source options"
                )))
            }
        };
        const ALLOWED: &[&str] = &[
            "mode",
            "caption_dropout",
            "max_sequence_length",
            "weighting_scheme",
            "logit_mean",
            "logit_std",
            "mode_scale",
            "center_crop",
            "random_flip",
            "max_grad_norm",
            "lr_scheduler",
            "lr_num_cycles",
            "lr_power",
            "with_prior_preservation",
            "lora_dropout",
            "use_rslora",
            "use_dora",
            "rank_pattern",
            "alpha_pattern",
        ];
        for key in bag.keys() {
            if !ALLOWED.contains(&key.as_str()) {
                return Err(invalid(format!("unsupported model option `{key}`")));
            }
        }
        let number = |key: &str, default: f64| -> Result<f64> {
            match bag.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| invalid(format!("{key} must be finite numeric"))),
            }
        };
        let boolean = |key: &str, default: bool| -> Result<bool> {
            match bag.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| invalid(format!("{key} must be boolean"))),
            }
        };
        for key in ["with_prior_preservation", "use_rslora", "use_dora"] {
            if boolean(key, false)? {
                return Err(invalid(format!(
                    "{key} is not an implemented training mechanism; refusing substitution"
                )));
            }
        }
        if number("lora_dropout", 0.0)? != 0.0 {
            return Err(invalid(
                "nonzero LoRA network dropout is not implemented by the native residual seam",
            ));
        }
        for key in ["rank_pattern", "alpha_pattern"] {
            if let Some(value) = bag.get(key) {
                if !value.as_object().is_some_and(|map| map.is_empty()) {
                    return Err(invalid(format!(
                        "nonempty {key} is not plain global-rank/alpha LoRA"
                    )));
                }
            }
        }
        if let Some(value) = bag.get("mode") {
            let requested = match value.as_str() {
                Some("captioned" | "text_to_image") => Workflow::Captioned,
                Some("edit" | "image_to_image") => Workflow::PairedEdit,
                _ => return Err(invalid("invalid training mode")),
            };
            if requested != workflow {
                return Err(invalid(
                    "selected training mode and dataset reference inputs disagree",
                ));
            }
        }
        let caption_dropout = number("caption_dropout", 0.0)? as f32;
        if !(0.0..=1.0).contains(&caption_dropout) {
            return Err(invalid("caption_dropout must be in[0,1]"));
        }
        let max_sequence_length = match bag.get("max_sequence_length") {
            None => 512,
            Some(value) => usize::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| invalid("max_sequence_length must be positive integer"))?,
            )
            .map_err(|_| invalid("max_sequence_length overflows"))?,
        };
        if max_sequence_length == 0 || max_sequence_length > i32::MAX as usize {
            return Err(invalid(
                "max_sequence_length must be positive and fit native positions",
            ));
        }
        if cfg.timestep_bias != "balanced" {
            return Err(invalid("use explicit source logit_mean/std rather than an unimplemented generic timestep bias"));
        }
        let default_weighting = match cfg.timestep_type.as_str() {
            "uniform" | "linear" => "none",
            "sigmoid" | "logit_normal" => "logit_normal",
            "mode" => "mode",
            "cosmap" => "cosmap",
            "sigma_sqrt" => "sigma_sqrt",
            _ => return Err(invalid("unsupported timestep distribution")),
        };
        let weighting = Weighting::parse(match bag.get("weighting_scheme") {
            None => default_weighting,
            Some(value) => value
                .as_str()
                .ok_or_else(|| invalid("weighting_scheme must be string"))?,
        })?;
        let logit_mean = number("logit_mean", 0.0)? as f32;
        let logit_std = number("logit_std", 1.0)? as f32;
        let mode_scale = number("mode_scale", 1.29)? as f32;
        let max_grad_norm = number("max_grad_norm", 1.0)? as f32;
        if !logit_mean.is_finite()
            || !logit_std.is_finite()
            || logit_std < 0.0
            || !mode_scale.is_finite()
            || !max_grad_norm.is_finite()
            || max_grad_norm < 0.0
        {
            return Err(invalid(
                "invalid source density/gradient clipping parameters",
            ));
        }
        let generic = match cfg.lr_scheduler {
            LrSchedule::Constant => {
                if cfg.lr_warmup_steps == 0 {
                    SourceLrKind::Constant
                } else {
                    SourceLrKind::ConstantWithWarmup
                }
            }
            LrSchedule::Linear => SourceLrKind::Linear,
            LrSchedule::Cosine => SourceLrKind::Cosine,
        };
        let scheduler = match bag.get("lr_scheduler") {
            None => generic,
            Some(value) => SourceLrKind::parse(
                value
                    .as_str()
                    .ok_or_else(|| invalid("lr_scheduler must be string"))?,
            )?,
        };
        let cycles = match bag.get("lr_num_cycles") {
            None => 1,
            Some(value) => u32::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| invalid("lr_num_cycles must be nonnegative integer"))?,
            )
            .map_err(|_| invalid("lr_num_cycles overflows"))?,
        };
        let power = number("lr_power", 1.0)?;
        Ok(Self {
            precision: TrainingPrecision::parse(&cfg.train_dtype)?,
            workflow,
            caption_dropout,
            max_sequence_length,
            center_crop: boolean("center_crop", false)?,
            random_flip: boolean("random_flip", false)?,
            weighting,
            logit_mean,
            logit_std,
            mode_scale,
            max_grad_norm,
            scheduler,
            cycles,
            power,
        })
    }
    pub fn schedule(
        &self,
        cfg: &TrainingConfig,
        batches_per_epoch: u64,
    ) -> Result<SourceLrSchedule> {
        let schedule = SourceLrSchedule {
            kind: self.scheduler,
            warmup: cfg.lr_warmup_steps as u64,
            total_updates: optimizer_update_budget(
                cfg.steps as u64,
                cfg.gradient_accumulation,
                batches_per_epoch,
            )?,
            cycles: self.cycles,
            power: self.power,
            base_lr: cfg.learning_rate as f64,
        };
        schedule.validate()?;
        Ok(schedule)
    }
}

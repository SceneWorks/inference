//! The **source-to-native training-configuration map** of the Iris generation trainer (story
//! sc-25689), implemented: upstream's typed config tree (`src/iris3b/config.py`, merged as
//! structured defaults ← experiment YAML ← dotlist overrides, exactly as `load_config` does) is
//! translated into the native request — the shared [`TrainingConfig`] plus the Iris options bag
//! ([`super::OPTIONS_KEY`]). Every public upstream key (every leaf of [`DEFAULTS_YAML`], and the
//! trainer CLI's `--config` / `--expected-nodes` / `--prestage` / overrides) has one
//! [`Disposition`] in [`UPSTREAM_OPTIONS`]; an unknown key is refused (OmegaConf's struct mode
//! raises too), a refused key is refused unless it is left at its default, and nothing is
//! silently ignored.
//!
//! The four official generation stages (`configs/iris3b/{stage1_256, stage2_512, stage3_1024,
//! sft_1024}.yaml`) are vendored verbatim ([`STAGE_PRESETS`]) and resolve through the same
//! translation into **editable** native presets ([`stage_preset_config`]) that carry their stage,
//! parent stage and data identity. A stage's parent checkpoint and its dataset are the run's own
//! inputs (`resume_from` and the request items), so the vendored placeholders
//! (`data.data_dirs: [/path/to/wids/dataset]`, `train.resume_from: /path/to/...pth`) are cleared by
//! explicit overrides when a preset is applied.

use std::collections::BTreeMap;

use serde_json::{json, Map as JsonMap, Value as JsonValue};

use super::super::yaml::{self, Value};
use super::OPTIONS_KEY;
use crate::train::{LrSchedule, TrainingConfig};
use crate::{Error, Result};

/// `OmegaConf.structured(Config)` at the pinned revision (single-line strings).
pub const DEFAULTS_YAML: &str = include_str!("upstream/defaults.yaml");

/// One official generation stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagePreset {
    /// The stage id (the config file stem).
    pub id: &'static str,
    /// The stage this one continues from (`resume_data_policy: new_phase` off its final checkpoint).
    pub parent: Option<&'static str>,
    /// The vendored YAML, verbatim.
    pub yaml: &'static str,
}

/// The four official stages, in curriculum order (256 → 512 → 1024 → SFT).
pub const STAGE_PRESETS: [StagePreset; 4] = [
    StagePreset {
        id: "stage1_256",
        parent: None,
        yaml: include_str!("upstream/stage1_256.yaml"),
    },
    StagePreset {
        id: "stage2_512",
        parent: Some("stage1_256"),
        yaml: include_str!("upstream/stage2_512.yaml"),
    },
    StagePreset {
        id: "stage3_1024",
        parent: Some("stage2_512"),
        yaml: include_str!("upstream/stage3_1024.yaml"),
    },
    StagePreset {
        id: "sft_1024",
        parent: Some("stage3_1024"),
        yaml: include_str!("upstream/sft_1024.yaml"),
    },
];

/// The stage preset named `id`.
pub fn stage_preset(id: &str) -> Result<&'static StagePreset> {
    STAGE_PRESETS.iter().find(|s| s.id == id).ok_or_else(|| {
        Error::Msg(format!(
            "iris training: unknown stage preset {id:?} (known: {})",
            STAGE_PRESETS
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })
}

/// How the native trainer treats one upstream key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// Carried into the native request (the field / option it lands in).
    Honored(&'static str),
    /// A framework / performance flag whose native counterpart is named here; it never changes the
    /// optimizer update.
    Mapped(&'static str),
    /// Never read by upstream's trainer either (an inference-time or launcher setting).
    Inert(&'static str),
    /// Fixed by a provisioned resource's `config.yaml`: a value that is set must equal the
    /// resource's (the trainer refuses a mismatch at plan time).
    Resource(&'static str),
    /// Refused (typed) unless left at its default.
    Refused(&'static str),
}

/// One upstream option and its disposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpstreamOption {
    pub key: &'static str,
    pub disposition: Disposition,
}

macro_rules! opts {
    ($( $k:literal => $d:expr ),* $(,)?) => { &[ $( UpstreamOption { key: $k, disposition: $d } ),* ] };
}
use Disposition::{Honored as H, Inert as I, Mapped as M, Refused as R, Resource as Rs};

const ARCH: &str = "the trained architecture is the backbone resource's config.yaml (random init \
                    reads it); a set value must equal it — provision a backbone config with that \
                    architecture to train one";
const TE: &str = "the frozen conditioning is the text-encoder resource and the backbone's \
                  text_encoder section; a set value must equal it";

/// Every public upstream training option and the CLI surface, with its native disposition.
pub const UPSTREAM_OPTIONS: &[UpstreamOption] = opts![
    "name" => H("iris.run_name (state.json, metrics.jsonl)"),
    "work_dir" => M("the request's output_dir (checkpoints under <output_dir>/<stem>.checkpoints)"),
    "wandb_project" => H("iris.tracking.project (metrics.jsonl header)"),
    "report_to" => M("metrics are written to <checkpoints>/metrics.jsonl; uploading them to wandb / \
                      tensorboard is the product layer's (inference links no network client)"),
    "tags" => H("iris.tracking.tags (metrics.jsonl header)"),
    "model.block" => Rs(ARCH), "model.dual_depth" => Rs(ARCH), "model.final_block_text" => Rs(ARCH),
    "model.hidden_size" => Rs(ARCH), "model.depth" => Rs(ARCH), "model.num_heads" => Rs(ARCH),
    "model.num_kv_heads" => Rs(ARCH), "model.gated_attention" => Rs(ARCH),
    "model.sandwich_norm" => Rs(ARCH), "model.patch_size" => Rs(ARCH),
    "model.in_channels" => Rs(ARCH), "model.mlp_ratio" => Rs(ARCH), "model.qkv_bias" => Rs(ARCH),
    "model.qk_norm" => Rs(ARCH), "model.norm_eps" => Rs(ARCH),
    "model.attn_backend" => M("sdpa | torch_flash | torch_cudnn | fa3 | fa4 select a torch attention \
                               kernel; natively MLX's fused SDPA / Candle's SDPA compute the same \
                               attention"),
    "model.modulation" => Rs(ARCH), "model.modulation_rank" => Rs(ARCH),
    "model.timestep_max_period" => Rs(ARCH), "model.adaln_zero_init" => Rs(ARCH),
    "model.rope_theta" => Rs(ARCH), "model.rope_scale" => Rs(ARCH),
    "model.rope_aspect" => H("iris.rope_aspect (square | isotropic; the trained and exported image \
                              RoPE law — stage 1 trains square)"),
    "model.rope_frame_pairs" => Rs(ARCH), "model.rope_frame_theta" => Rs(ARCH),
    "model.text_rope" => Rs(ARCH), "model.text_rope_theta" => Rs(ARCH),
    "model.text_abs_pos_embed" => Rs(ARCH), "model.text_dim" => Rs(ARCH),
    "model.text_len" => Rs(ARCH), "model.text_adapter" => Rs(ARCH),
    "model.text_lap_num_layers" => Rs(ARCH), "model.text_lap_num_heads" => Rs(ARCH),
    "model.text_lap_mlp_ratio" => Rs(ARCH),
    "model.repa_layer" => H("iris.repa.layer"),
    "model.pixel.enabled" => Rs(ARCH), "model.pixel.depth" => Rs(ARCH),
    "model.pixel.hidden_size" => Rs(ARCH), "model.pixel.attn_hidden_size" => Rs(ARCH),
    "model.pixel.num_heads" => Rs(ARCH), "model.pixel.mlp_ratio" => Rs(ARCH),
    "model.pixel.modulation" => Rs(ARCH), "model.pixel.abs_pos_embed" => Rs(ARCH),
    "flow.num_train_timesteps" => Rs("the 1000-point grid of the backbone's flow section"),
    "flow.shift" => H("iris.flow_shift"),
    "flow.timestep_sampler" => H("TrainingConfig::timestep_type"),
    "flow.logit_mean" => H("iris.logit_mean"), "flow.logit_std" => H("iris.logit_std"),
    "flow.prediction" => H("iris.prediction"),
    "flow.x_pred_sigma_min" => H("iris.x_pred_sigma_min"),
    "flow.shift_law" => H("iris.shift_law"), "flow.shift_base_tokens" => H("iris.shift_base_tokens"),
    "text_encoder.name" => Rs(TE), "text_encoder.pretrained" => Rs(TE), "text_encoder.dim" => Rs(TE),
    "text_encoder.max_length" => Rs(TE), "text_encoder.dtype" => Rs(TE),
    "text_encoder.attn_implementation" => M("a Hugging Face attention backend switch; the shared \
                                             native decoder runs the same attention"),
    "text_encoder.hidden_layers" => Rs(TE),
    "text_encoder.null_embed_dir" => M("the on-disk null-embedding memo; natively the null is encoded \
                                        once per run in memory (bit-identical: the encoder is \
                                        frozen and deterministic)"),
    "text_encoder.compile" => M("torch.compile of the frozen decoder; natively the decoder runs eagerly \
                                 (same math)"),
    "text_encoder.on_caption_overflow" => H("iris.on_caption_overflow"),
    "repa.weight" => H("iris.repa.weight"), "repa.variant" => H("iris.repa.variant"),
    "repa.spatial_norm_gamma" => H("iris.repa.spatial_norm_gamma"),
    "repa.teacher_source" => H("iris.repa.teacher_source (hub: the native DINOv2 teacher; hf / \
                                DINOv3: refused, see TeacherArch)"),
    "repa.teacher_hub" => H("iris.repa.teacher_hub (facebookresearch/dinov2 only)"),
    "repa.teacher" => H("iris.repa.teacher (dinov2_vit{s,b,l,g}14[_reg])"),
    "repa.teacher_weights" => H("iris.repa.teacher_weights: the teacher resource directory \
                                 (LoadSpec component repa_teacher); never downloaded"),
    "repa.teacher_dim" => H("iris.repa.teacher_dim"),
    "repa.proj_hidden_dim" => H("iris.repa.proj_hidden_dim"),
    "repa.teacher_image_size" => H("iris.repa.teacher_image_size"),
    "repa.teacher_patch_size" => H("iris.repa.teacher_patch_size"),
    "repa.teacher_match_student" => H("iris.repa.teacher_match_student"),
    "data.data_dirs" => R("native training reads the request's TrainingItems (the product layer \
                           resolves WIDS shards into items)"),
    "data.type" => H("iris.data_type (pixel | pixel_multiscale; the synthetic smoke datasets are \
                      refused)"),
    "data.image_size" => H("TrainingConfig::resolution"),
    "data.shape_policy" => H("iris.shape_policy"),
    "data.aspect_ratio_bucket" => H("iris.aspect_ratio_bucket"),
    "data.shape_align" => H("iris.shape_align"), "data.shape_max_ratio" => H("iris.shape_max_ratio"),
    "data.caption_field" => H("iris.caption_field"), "data.caption_fields" => H("iris.caption_fields"),
    "data.val_data_dirs" => R("the holdout set is the request items flagged model_options.split = \
                               \"holdout\""),
    "data.num_workers" => M("dataloader worker processes; natively the host decodes each batch in \
                             process"),
    "train.batch_size" => H("TrainingConfig::batch_size (per rank)"),
    "train.grad_accum_steps" => H("TrainingConfig::gradient_accumulation"),
    "train.expected_global_batch" => H("iris.expected_global_batch"),
    "train.num_epochs" => H("iris.num_epochs"),
    "train.max_steps" => H("TrainingConfig::steps (= max_steps × grad_accum micro-steps; 0 = the \
                            epoch horizon)"),
    "train.mixed_precision" => H("TrainingConfig::train_dtype (bf16 | fp16 | no)"),
    "train.activation_checkpointing" => H("iris.activation_checkpointing"),
    "train.ac_selective_every" => H("iris.ac_selective_every"),
    "train.gradient_clip" => H("iris.gradient_clip"), "train.text_dropout" => H("iris.text_dropout"),
    "train.seed" => H("TrainingConfig::seed"),
    "train.optimizer.name" => H("TrainingConfig::optimizer"),
    "train.optimizer.lr" => H("TrainingConfig::learning_rate"),
    "train.optimizer.betas" => H("iris.betas"),
    "train.optimizer.weight_decay" => H("TrainingConfig::weight_decay"),
    "train.optimizer.muon_momentum" => H("iris.muon_momentum"),
    "train.optimizer.muon_nesterov" => H("iris.muon_nesterov"),
    "train.optimizer.muon_adjust_lr" => H("iris.muon_adjust_lr"),
    "train.optimizer.auto_lr" => H("iris.auto_lr"),
    "train.optimizer.base_batch_size" => H("iris.base_batch_size"),
    "train.optimizer.schedule" => H("TrainingConfig::lr_scheduler"),
    "train.optimizer.warmup_steps" => H("TrainingConfig::lr_warmup_steps (× grad_accum)"),
    "train.optimizer.scale_warmup_by_world" => H("iris.scale_warmup_by_world"),
    "train.ema.enabled" => H("iris.ema_enabled"), "train.ema.decay" => H("iris.ema_decay"),
    "train.dist.strategy" => H("iris.dist.strategy"), "train.dist.sharding" => H("iris.dist.sharding"),
    "train.dist.mesh.dp_shard" => H("iris.dist.mesh.dp_shard"),
    "train.dist.mesh.dp_replicate" => H("iris.dist.mesh.dp_replicate"),
    "train.dist.param_dtype" => H("iris.dist.param_dtype"),
    "train.dist.reduce_dtype" => H("iris.dist.reduce_dtype"),
    "train.dist.reshard_after_forward" => H("iris.dist.reshard_after_forward"),
    "train.dist.checkpoint_io" => H("iris.dist.checkpoint_io"),
    "train.dist.ddp.find_unused_parameters" => H("iris.dist.ddp.find_unused_parameters"),
    "train.dist.ddp.gradient_as_bucket_view" => H("iris.dist.ddp.gradient_as_bucket_view"),
    "train.dist.ddp.batched_grad_copy" => H("iris.dist.ddp.batched_grad_copy"),
    "train.dist.ddp.skip_all_reduce_unused_params" => H("iris.dist.ddp.skip_all_reduce_unused_params"),
    "train.dist.ddp.bucket_cap_mb" => H("iris.dist.ddp.bucket_cap_mb"),
    "train.dist.ddp.forward_sync_buffers" => H("iris.dist.ddp.forward_sync_buffers"),
    "train.load_from" => H("iris.load_from (a native checkpoint / exported model dir; a torch .pth is \
                            refused — export it with scripts/export_checkpoint.py)"),
    "train.resume_from" => H("iris.resume_from (\"latest\" = TrainingConfig::resume)"),
    "train.resume_data_world_size" => R("migrates torch checkpoints written before data_position was \
                                         persisted; every native checkpoint records its data position"),
    "train.resume_data_policy" => H("iris.resume_data_policy"),
    "train.resume_optimizer" => H("iris.resume_optimizer"),
    "train.override_lr_on_resume" => H("iris.override_lr_on_resume"),
    "train.save_every_steps" => H("TrainingConfig::save_every (× grad_accum)"),
    "train.save_every_epochs" => H("iris.save_every_epochs"),
    "train.keep_last_checkpoints" => H("iris.keep_last_checkpoints"),
    "train.milestone_steps" => H("iris.milestone_steps"),
    "train.log_every" => H("iris.log_every"),
    "train.log_block_grad_norms" => H("iris.log_block_grad_norms"),
    "train.sample_every_steps" => H("TrainingConfig::sample_every (× grad_accum) + \
                                     iris.preview_at_step_one"),
    "train.val_every_steps" => H("iris.val_every_steps"),
    "train.val_samples" => H("iris.val_samples"),
    "train.validation_prompts" => H("iris.upstream_validation_prompts (true: the default seven; \
                                     a list: the config's own — every prompt, uncapped)"),
    "train.nan_loss_tolerance" => H("iris.nan_loss_tolerance"),
    "train.perf.foreach_ema" => M("multi-tensor EMA kernels; the native EMA update is the same \
                                   elementwise blend"),
    "train.perf.fused_adamw" => M("torch's fused AdamW kernel; the native AdamW is the same update"),
    "train.perf.skip_dropped_text" => M("dropped rows skip the encoder; natively a dropped row always \
                                         takes the null states without encoding (identical: the \
                                         encoder consumes no randomness)"),
    "train.perf.lazy_sync" => H("iris.perf.lazy_sync (the non-finite guard on the clipped grad norm)"),
    "train.perf.compile" => M("torch.compile (numerics within training noise); natively eager"),
    "train.perf.compile_mode" => M("torch.compile mode; natively eager"),
    "train.perf.compile_scope" => M("torch.compile scope; natively eager"),
    "train.perf.compile_dynamic" => M("torch.compile shape dynamism; natively eager"),
    "train.perf.compile_fullgraph" => M("torch.compile fullgraph; natively eager"),
    "train.perf.recompile_limit" => M("dynamo cache budget; natively eager (no recompilation)"),
    "sample.steps" => I("scripts/sample.py only; the trainer's previews use 100 steps"),
    "sample.order" => H("iris.preview_order"),
    "sample.cfg_scale" => I("scripts/sample.py only; the trainer's previews use CFG 3"),
    "sample.cfg_interval" => I("scripts/sample.py only; the trainer's previews use (0, 1)"),
    "sample.negative_prompt" => I("scripts/sample.py only; the trainer's previews pass none"),
    "cli.--config" => H("translate_upstream's yaml"),
    "cli.--expected-nodes" => H("iris.dist.world.expected_nodes"),
    "cli.--prestage" => M("fills node-local torch caches (hub teacher, shape-plan keys); natively the \
                           teacher is a provisioned resource and shape keys are computed in the \
                           run's preparing pass — Trainer::validate is the dry run"),
    "cli.overrides" => H("translate_upstream's dotlist overrides"),
];

/// The disposition of `key`.
pub fn disposition(key: &str) -> Option<Disposition> {
    UPSTREAM_OPTIONS
        .iter()
        .find(|o| o.key == key)
        .map(|o| o.disposition)
}

// =============================================================================================
// The merged tree
// =============================================================================================

fn leaves(v: &Value, prefix: &str, out: &mut BTreeMap<String, Value>) {
    match v {
        Value::Map(entries) => {
            for (k, child) in entries {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                if matches!(child, Value::Map(_)) {
                    leaves(child, &key, out);
                } else {
                    out.insert(key, child.clone());
                }
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

/// The flattened upstream defaults (`dotted.key → value`).
pub fn default_leaves() -> Result<BTreeMap<String, Value>> {
    let tree = yaml::parse(DEFAULTS_YAML)?;
    let mut out = BTreeMap::new();
    leaves(&tree, "", &mut out);
    Ok(out)
}

/// Parse one dotlist value (`key=value`) the way OmegaConf's `from_dotlist` reads YAML scalars.
fn parse_override_value(text: &str) -> Result<Value> {
    let t = text.trim();
    if t.starts_with('[') {
        return yaml::parse_flow_list(t, 0);
    }
    let parsed = yaml::parse(&format!("v: {t}"))?;
    Ok(parsed
        .get("v")
        .cloned()
        .unwrap_or(Value::Scalar("null".into())))
}

/// Merge `defaults ← yaml ← overrides` into flat leaves; unknown keys are refused.
pub fn merged_leaves(
    config_yaml: Option<&str>,
    overrides: &[String],
) -> Result<(BTreeMap<String, Value>, Vec<String>)> {
    let mut out = default_leaves()?;
    let mut explicit = Vec::new();
    if let Some(text) = config_yaml {
        let tree = yaml::parse(text)?;
        let mut user = BTreeMap::new();
        leaves(&tree, "", &mut user);
        for (k, v) in user {
            if !out.contains_key(&k) {
                return Err(Error::Unsupported(format!(
                    "iris training: upstream config key {k:?} is not part of iris3b.config.Config"
                )));
            }
            explicit.push(k.clone());
            out.insert(k, v);
        }
    }
    for o in overrides {
        let (k, v) = o
            .split_once('=')
            .ok_or_else(|| Error::Msg(format!("iris training: override {o:?} is not key=value")))?;
        let k = k.trim().to_string();
        if !out.contains_key(&k) {
            return Err(Error::Unsupported(format!(
                "iris training: override key {k:?} is not part of iris3b.config.Config"
            )));
        }
        explicit.push(k.clone());
        out.insert(k, parse_override_value(v)?);
    }
    Ok((out, explicit))
}

/// A YAML scalar or list as JSON (`null`/`~` → null, booleans, integers, floats, strings).
pub fn to_json(v: &Value) -> JsonValue {
    match v {
        Value::Scalar(s) => scalar_json(s),
        Value::List(items) => JsonValue::Array(items.iter().map(to_json).collect()),
        Value::Map(entries) => JsonValue::Object(
            entries
                .iter()
                .map(|(k, v)| (k.clone(), to_json(v)))
                .collect(),
        ),
    }
}

fn scalar_json(s: &str) -> JsonValue {
    match s {
        "null" | "~" | "None" => return JsonValue::Null,
        "true" | "True" => return JsonValue::Bool(true),
        "false" | "False" => return JsonValue::Bool(false),
        _ => {}
    }
    if let Ok(i) = s.parse::<i64>() {
        return json!(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        if s.contains(['.', 'e', 'E']) {
            return json!(f);
        }
    }
    JsonValue::String(s.to_string())
}

// =============================================================================================
// Translation
// =============================================================================================

/// The native request an upstream config resolves to.
#[derive(Clone, Debug, PartialEq)]
pub struct Translation {
    /// The shared config with `model_options["iris"]` filled.
    pub config: TrainingConfig,
    /// The upstream keys that were set explicitly (YAML or overrides).
    pub explicit: Vec<String>,
    /// The mapped (framework / performance) keys whose value differs from upstream's default, with
    /// the native semantic each maps to.
    pub mapped: Vec<(String, String)>,
}

/// Translate an upstream training config (`--config` YAML text and dotlist `overrides`) into the
/// native request. `base` supplies the fields upstream does not describe (`network_type`, rank, …);
/// the translation always trains the full model, as upstream does.
pub fn translate_upstream(
    config_yaml: Option<&str>,
    overrides: &[String],
    base: TrainingConfig,
) -> Result<Translation> {
    let (leaves, explicit) = merged_leaves(config_yaml, overrides)?;
    let defaults = default_leaves()?;
    let get = |k: &str| -> &Value { leaves.get(k).expect("every key has a default") };
    let j = |k: &str| to_json(get(k));
    let u = |k: &str| -> Result<u64> {
        j(k).as_u64().ok_or_else(|| {
            Error::Msg(format!(
                "iris training: upstream {k} must be a non-negative integer, got {}",
                j(k)
            ))
        })
    };
    let f = |k: &str| -> Result<f64> {
        j(k).as_f64().ok_or_else(|| {
            Error::Msg(format!(
                "iris training: upstream {k} must be a number, got {}",
                j(k)
            ))
        })
    };
    let s = |k: &str| -> String {
        match j(k) {
            JsonValue::String(v) => v,
            JsonValue::Null => String::new(),
            other => other.to_string(),
        }
    };
    let is_default = |k: &str| leaves.get(k) == defaults.get(k);

    // Refused keys: only their defaults are accepted.
    let mut mapped = Vec::new();
    for o in UPSTREAM_OPTIONS {
        if o.key.starts_with("cli.") {
            continue;
        }
        match o.disposition {
            Disposition::Refused(reason) if !is_default(o.key) => {
                return Err(Error::Unsupported(format!(
                    "iris training: upstream {} = {} is refused: {reason}",
                    o.key,
                    j(o.key)
                )))
            }
            Disposition::Mapped(m) if !is_default(o.key) => {
                mapped.push((o.key.to_string(), m.to_string()))
            }
            _ => {}
        }
    }

    let mut cfg = base;
    let accum = u("train.grad_accum_steps")?.max(1) as u32;
    cfg.full_finetune = true;
    cfg.batch_size = u("train.batch_size")? as u32;
    cfg.gradient_accumulation = accum;
    cfg.steps = (u("train.max_steps")? as u32).saturating_mul(accum);
    cfg.seed = u("train.seed")?;
    cfg.train_dtype = s("train.mixed_precision");
    cfg.gradient_checkpointing = false;
    cfg.optimizer = s("train.optimizer.name");
    cfg.learning_rate = f("train.optimizer.lr")? as f32;
    cfg.weight_decay = f("train.optimizer.weight_decay")? as f32;
    cfg.lr_scheduler = match s("train.optimizer.schedule").as_str() {
        "constant" => LrSchedule::Constant,
        "cosine" => LrSchedule::Cosine,
        other => {
            return Err(Error::Msg(format!(
                "iris training: unknown lr schedule '{other}' (constant | cosine)"
            )))
        }
    };
    cfg.lr_warmup_steps = (u("train.optimizer.warmup_steps")? as u32).saturating_mul(accum);
    cfg.resolution = u("data.image_size")? as u32;
    cfg.save_every = (u("train.save_every_steps")? as u32).saturating_mul(accum);
    cfg.sample_every = (u("train.sample_every_steps")? as u32).saturating_mul(accum);
    cfg.sample_steps = 100; // VALIDATION_STEPS
    cfg.sample_guidance_scale = 3.0; // VALIDATION_CFG_SCALE
    cfg.timestep_type = s("flow.timestep_sampler");
    cfg.timestep_bias = "balanced".into();
    cfg.loss_type = "mse".into();
    let resume_from = j("train.resume_from");
    cfg.resume = resume_from.as_str() == Some("latest");

    let mut o = JsonMap::new();
    let mut put = |k: &str, v: JsonValue| {
        o.insert(k.to_string(), v);
    };
    put("run_name", j("name"));
    put(
        "tracking",
        json!({"project": j("wandb_project"), "tags": j("tags"), "report_to": j("report_to")}),
    );
    // data
    let data_type = s("data.type");
    match data_type.as_str() {
        "pixel" | "pixel_multiscale" => {}
        "synthetic" | "synthetic_multiscale" => {
            return Err(Error::Unsupported(format!(
                "iris training: data.type {data_type} is upstream's random smoke-test dataset; \
                 native runs train on the request items"
            )))
        }
        other => {
            return Err(Error::Msg(format!(
                "iris training: unknown data.type {other:?}"
            )))
        }
    }
    put("data_type", json!(data_type));
    put("shape_policy", j("data.shape_policy"));
    put("aspect_ratio_bucket", j("data.aspect_ratio_bucket"));
    put("shape_align", j("data.shape_align"));
    put("shape_max_ratio", j("data.shape_max_ratio"));
    put("caption_field", j("data.caption_field"));
    put("caption_fields", j("data.caption_fields"));
    // flow
    put("flow_shift", j("flow.shift"));
    put("logit_mean", j("flow.logit_mean"));
    put("logit_std", j("flow.logit_std"));
    put("prediction", j("flow.prediction"));
    put("x_pred_sigma_min", j("flow.x_pred_sigma_min"));
    put("shift_law", j("flow.shift_law"));
    put("shift_base_tokens", j("flow.shift_base_tokens"));
    put("on_caption_overflow", j("text_encoder.on_caption_overflow"));
    // the architecture / conditioning keys the YAML set explicitly (checked against the resources)
    let mut model = JsonMap::new();
    let mut text_encoder = JsonMap::new();
    for k in &explicit {
        if let Some(rest) = k.strip_prefix("model.") {
            if matches!(rest, "attn_backend") {
                continue;
            }
            if rest == "rope_aspect" || rest == "repa_layer" {
                continue;
            }
            model.insert(rest.to_string(), j(k));
        } else if let Some(rest) = k.strip_prefix("text_encoder.") {
            if matches!(
                rest,
                "attn_implementation" | "null_embed_dir" | "compile" | "on_caption_overflow"
            ) {
                continue;
            }
            text_encoder.insert(rest.to_string(), j(k));
        } else if k == "flow.num_train_timesteps" {
            put("num_train_timesteps", j(k));
        }
    }
    if !model.is_empty() {
        put("architecture", JsonValue::Object(model));
    }
    if !text_encoder.is_empty() {
        put("text_encoder", JsonValue::Object(text_encoder));
    }
    // Always the merged value: a key left at upstream's default must resolve to upstream's default,
    // never inherit the provisioned backbone's (a stage-1 export carries `rope_aspect: square`).
    put("rope_aspect", j("model.rope_aspect"));
    // REPA
    let mut repa = JsonMap::new();
    for k in [
        "weight",
        "variant",
        "spatial_norm_gamma",
        "teacher_source",
        "teacher_hub",
        "teacher",
        "teacher_weights",
        "teacher_dim",
        "proj_hidden_dim",
        "teacher_image_size",
        "teacher_patch_size",
        "teacher_match_student",
    ] {
        repa.insert(k.into(), j(&format!("repa.{k}")));
    }
    repa.insert("layer".into(), j("model.repa_layer"));
    put("repa", JsonValue::Object(repa));
    // train loop
    put("expected_global_batch", j("train.expected_global_batch"));
    put("num_epochs", j("train.num_epochs"));
    put(
        "activation_checkpointing",
        j("train.activation_checkpointing"),
    );
    put("ac_selective_every", j("train.ac_selective_every"));
    put("gradient_clip", j("train.gradient_clip"));
    put("text_dropout", j("train.text_dropout"));
    put("betas", j("train.optimizer.betas"));
    put("muon_momentum", j("train.optimizer.muon_momentum"));
    put("muon_nesterov", j("train.optimizer.muon_nesterov"));
    put("muon_adjust_lr", j("train.optimizer.muon_adjust_lr"));
    put("auto_lr", j("train.optimizer.auto_lr"));
    put("base_batch_size", j("train.optimizer.base_batch_size"));
    put(
        "scale_warmup_by_world",
        j("train.optimizer.scale_warmup_by_world"),
    );
    put("ema_enabled", j("train.ema.enabled"));
    put("ema_decay", j("train.ema.decay"));
    put(
        "dist",
        json!({
            "strategy": j("train.dist.strategy"),
            "sharding": j("train.dist.sharding"),
            "mesh": {"dp_shard": j("train.dist.mesh.dp_shard"),
                     "dp_replicate": j("train.dist.mesh.dp_replicate")},
            "param_dtype": j("train.dist.param_dtype"),
            "reduce_dtype": j("train.dist.reduce_dtype"),
            "reshard_after_forward": j("train.dist.reshard_after_forward"),
            "checkpoint_io": j("train.dist.checkpoint_io"),
            "ddp": {
                "find_unused_parameters": j("train.dist.ddp.find_unused_parameters"),
                "gradient_as_bucket_view": j("train.dist.ddp.gradient_as_bucket_view"),
                "batched_grad_copy": j("train.dist.ddp.batched_grad_copy"),
                "skip_all_reduce_unused_params": j("train.dist.ddp.skip_all_reduce_unused_params"),
                "bucket_cap_mb": j("train.dist.ddp.bucket_cap_mb"),
                "forward_sync_buffers": j("train.dist.ddp.forward_sync_buffers"),
            }
        }),
    );
    if !j("train.load_from").is_null() {
        put("load_from", j("train.load_from"));
    }
    if let Some(p) = resume_from.as_str().filter(|p| *p != "latest") {
        put("resume_from", json!(p));
    }
    put("resume_data_policy", j("train.resume_data_policy"));
    put("resume_optimizer", j("train.resume_optimizer"));
    put("override_lr_on_resume", j("train.override_lr_on_resume"));
    put("save_every_epochs", j("train.save_every_epochs"));
    put("keep_last_checkpoints", j("train.keep_last_checkpoints"));
    put("milestone_steps", j("train.milestone_steps"));
    put("log_every", j("train.log_every"));
    put("log_block_grad_norms", j("train.log_block_grad_norms"));
    put("val_every_steps", j("train.val_every_steps"));
    put("val_samples", j("train.val_samples"));
    put("nan_loss_tolerance", j("train.nan_loss_tolerance"));
    put(
        "perf",
        json!({
            "foreach_ema": j("train.perf.foreach_ema"),
            "fused_adamw": j("train.perf.fused_adamw"),
            "skip_dropped_text": j("train.perf.skip_dropped_text"),
            "lazy_sync": j("train.perf.lazy_sync"),
            "compile": j("train.perf.compile"),
            "compile_mode": j("train.perf.compile_mode"),
            "compile_scope": j("train.perf.compile_scope"),
            "compile_dynamic": j("train.perf.compile_dynamic"),
            "compile_fullgraph": j("train.perf.compile_fullgraph"),
            "recompile_limit": j("train.perf.recompile_limit"),
        }),
    );
    // previews: upstream renders the raw weights at step 1 and every cadence, every validation
    // prompt (the seven defaults unless the config names others) — uncapped, carried through
    // `upstream_validation_prompts` (true = the defaults, a list = the config's own).
    put("preview_weights", json!("raw"));
    put("preview_at_step_one", json!(true));
    put("preview_order", j("sample.order"));
    let prompts = j("train.validation_prompts");
    cfg.sample_prompts = Vec::new();
    if is_default("train.validation_prompts") {
        put("upstream_validation_prompts", json!(true));
    } else {
        put("upstream_validation_prompts", prompts);
    }
    // Every carried value is type-checked by `IrisTrainPlan::resolve`.
    cfg.model_options
        .insert(OPTIONS_KEY.into(), JsonValue::Object(o));
    Ok(Translation {
        config: cfg,
        explicit,
        mapped,
    })
}

fn json_eq(a: &JsonValue, b: &JsonValue) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// Check the architecture / conditioning keys a translated config set explicitly
/// (`iris.architecture`, `iris.text_encoder`, `iris.num_train_timesteps`) against the provisioned
/// backbone resource: each must equal the resource's value ([`Disposition::Resource`]).
pub fn check_resources(
    config: &super::IrisConfig,
    defaults: &super::TrainFlowDefaults,
    architecture: Option<&JsonValue>,
    text_encoder: Option<&JsonValue>,
    num_train_timesteps: Option<&JsonValue>,
) -> Result<()> {
    let m = &config.model;
    let p = &m.pixel;
    let native = |k: &str| -> Option<JsonValue> {
        Some(match k {
            "block" => json!(m.block),
            "dual_depth" => json!(m.dual_depth),
            "final_block_text" => json!(m.final_block_text),
            "hidden_size" => json!(m.hidden_size),
            "depth" => json!(m.depth),
            "num_heads" => json!(m.num_heads),
            "num_kv_heads" => json!(m.num_kv_heads),
            "gated_attention" => json!(m.gated_attention),
            "sandwich_norm" => json!(m.sandwich_norm),
            "patch_size" => json!(m.patch_size),
            "in_channels" => json!(m.in_channels),
            "mlp_ratio" => json!(m.mlp_ratio),
            "qkv_bias" => json!(m.qkv_bias),
            "qk_norm" => json!(m.qk_norm),
            "norm_eps" => json!(m.norm_eps),
            "modulation" => json!(m.modulation),
            "modulation_rank" => json!(defaults.modulation_rank),
            "timestep_max_period" => json!(m.timestep_max_period),
            "adaln_zero_init" => json!(defaults.adaln_zero_init),
            "rope_theta" => json!(m.rope_theta),
            "rope_scale" => json!(m.rope_scale),
            "rope_frame_pairs" => json!(m.rope_frame_pairs),
            "rope_frame_theta" => json!(defaults.rope_frame_theta),
            "text_rope" => json!(m.text_rope),
            "text_rope_theta" => json!(m.text_rope_theta),
            "text_abs_pos_embed" => json!(m.text_abs_pos_embed),
            "text_dim" => json!(m.text_dim),
            "text_len" => json!(m.text_len),
            "text_adapter" => json!(m.text_adapter),
            "text_lap_num_layers" => json!(m.text_lap_num_layers),
            "text_lap_num_heads" => json!(m.text_lap_num_heads),
            "text_lap_mlp_ratio" => json!(m.text_lap_mlp_ratio),
            "pixel.enabled" => json!(p.enabled),
            "pixel.depth" => json!(p.depth),
            "pixel.hidden_size" => json!(p.hidden_size),
            "pixel.attn_hidden_size" => json!(p.attn_hidden_size),
            "pixel.num_heads" => json!(p.num_heads),
            "pixel.mlp_ratio" => json!(p.mlp_ratio),
            "pixel.modulation" => json!(p.modulation),
            "pixel.abs_pos_embed" => json!(p.abs_pos_embed),
            _ => return None,
        })
    };
    let t = &config.text_encoder;
    let native_te = |k: &str| -> Option<JsonValue> {
        Some(match k {
            "name" => json!(t.name),
            "pretrained" => json!(t.pretrained),
            "dim" => json!(t.dim),
            "max_length" => json!(t.max_length),
            "dtype" => json!(t.dtype),
            "hidden_layers" => json!(t.hidden_layers),
            _ => return None,
        })
    };
    for (section, raw, lookup) in [
        (
            "model",
            architecture,
            &native as &dyn Fn(&str) -> Option<JsonValue>,
        ),
        ("text_encoder", text_encoder, &native_te),
    ] {
        let Some(raw) = raw.filter(|v| !v.is_null()) else {
            continue;
        };
        let obj = raw.as_object().ok_or_else(|| {
            Error::Msg(format!(
                "iris training: the {section} overrides must be an object"
            ))
        })?;
        for (k, v) in obj {
            let have = lookup(k).ok_or_else(|| {
                Error::Unsupported(format!(
                    "iris training: {section}.{k} is not an upstream {section} key the trainer \
                     checks"
                ))
            })?;
            let same = match (v.as_array(), have.as_array()) {
                (Some(a), Some(b)) => {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_eq(x, y))
                }
                _ => json_eq(v, &have),
            };
            if !same {
                return Err(Error::Unsupported(format!(
                    "iris training: {section}.{k} = {v} differs from the provisioned resource's \
                     {have}: {}",
                    if section == "model" { ARCH } else { TE }
                )));
            }
        }
    }
    if let Some(v) = num_train_timesteps.filter(|v| !v.is_null()) {
        if !json_eq(v, &json!(config.flow.num_train_timesteps)) {
            return Err(Error::Unsupported(format!(
                "iris training: flow.num_train_timesteps = {v} differs from the backbone's {}",
                config.flow.num_train_timesteps
            )));
        }
    }
    Ok(())
}

/// The editable native config of an official stage: the vendored YAML translated with the
/// placeholder dataset and parent-checkpoint paths cleared (`data.data_dirs=[]`,
/// `train.resume_from=null`), stamped with `stage` / `parent_stage` and its data identity. Edit any
/// field or option before sending; set `resume_from` to the parent stage's checkpoint to continue
/// the curriculum (the preset already carries the stage's `resume_data_policy`).
pub fn stage_preset_config(id: &str, base: TrainingConfig) -> Result<TrainingConfig> {
    let preset = stage_preset(id)?;
    let t = translate_upstream(
        Some(preset.yaml),
        &[
            "data.data_dirs=[]".to_string(),
            "train.resume_from=null".to_string(),
        ],
        base,
    )?;
    let mut cfg = t.config;
    let opts = cfg
        .model_options
        .get_mut(OPTIONS_KEY)
        .and_then(JsonValue::as_object_mut)
        .expect("translated");
    opts.insert("stage".into(), json!(preset.id));
    opts.insert("parent_stage".into(), json!(preset.parent));
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(cfg: &TrainingConfig) -> &JsonMap<String, JsonValue> {
        cfg.model_options[OPTIONS_KEY].as_object().unwrap()
    }

    #[test]
    fn the_registry_covers_every_upstream_key() {
        let keys: Vec<String> = default_leaves().unwrap().into_keys().collect();
        for k in &keys {
            assert!(
                disposition(k).is_some(),
                "upstream key {k} has no disposition"
            );
        }
        for o in UPSTREAM_OPTIONS {
            assert!(
                o.key.starts_with("cli.") || keys.iter().any(|k| k == o.key),
                "registry key {} is not an upstream key",
                o.key
            );
        }
        assert_eq!(keys.len() + 4, UPSTREAM_OPTIONS.len());
    }

    #[test]
    fn stage_presets_translate_their_upstream_values() {
        let s1 = stage_preset_config("stage1_256", TrainingConfig::default()).unwrap();
        let o = opts(&s1);
        assert_eq!(s1.resolution, 256);
        assert_eq!(s1.batch_size, 16);
        assert_eq!(s1.steps, 365_000);
        assert_eq!(s1.optimizer, "muon");
        assert_eq!(o["flow_shift"], json!(2.0));
        assert_eq!(o["repa"]["weight"], json!(0.5));
        assert_eq!(o["rope_aspect"], json!("square"));
        assert_eq!(o["expected_global_batch"], json!(1024));
        assert_eq!(o["dist"]["strategy"], json!("fsdp2"));
        assert_eq!(o["dist"]["mesh"]["dp_shard"], json!(8));
        assert_eq!(o["perf"]["lazy_sync"], json!(true));
        assert_eq!(o["stage"], json!("stage1_256"));
        assert_eq!(o["parent_stage"], JsonValue::Null);
        assert_eq!(o["keep_last_checkpoints"], json!(3));
        assert_eq!(s1.save_every, 5000);
        let s2 = stage_preset_config("stage2_512", TrainingConfig::default()).unwrap();
        let o2 = opts(&s2);
        assert_eq!(o2["shape_policy"], json!("bucket"));
        assert_eq!(o2["aspect_ratio_bucket"], json!("shared21-512"));
        assert_eq!(o2["resume_data_policy"], json!("new_phase"));
        assert_eq!(o2["parent_stage"], json!("stage1_256"));
        assert_eq!(o2["repa"]["weight"], json!(0.0));
        assert!(o2.get("resume_from").is_none());
        let s3 = stage_preset_config("stage3_1024", TrainingConfig::default()).unwrap();
        assert_eq!(s3.learning_rate, 5e-5);
        assert_eq!(opts(&s3)["override_lr_on_resume"], json!(true));
        assert_eq!(opts(&s3)["aspect_ratio_bucket"], json!("shared21-1024"));
        let sft = stage_preset_config("sft_1024", TrainingConfig::default()).unwrap();
        assert_eq!(sft.learning_rate, 4e-5);
        assert_eq!(sft.lr_warmup_steps, 0);
        assert_eq!(sft.steps, 665_000);
        assert!(stage_preset_config("stage4", TrainingConfig::default()).is_err());
    }

    #[test]
    fn refused_keys_are_refused_only_off_default() {
        assert!(translate_upstream(None, &[], TrainingConfig::default()).is_ok());
        for o in [
            "data.data_dirs=[/x]",
            "data.val_data_dirs=[/y]",
            "train.resume_data_world_size=8",
            "data.type=synthetic",
        ] {
            let e = translate_upstream(None, &[o.to_string()], TrainingConfig::default());
            assert!(matches!(e, Err(Error::Unsupported(_))), "{o}");
        }
        assert!(
            translate_upstream(None, &["train.bogus=1".into()], TrainingConfig::default()).is_err()
        );
        assert!(
            translate_upstream(Some("train:\n  nope: 1\n"), &[], TrainingConfig::default())
                .is_err()
        );
    }

    #[test]
    fn honored_keys_change_the_request_and_mapped_keys_are_reported() {
        let base = translate_upstream(None, &[], TrainingConfig::default()).unwrap();
        // Every Honored key, perturbed off its default, must reach the request.
        let perturbed: BTreeMap<&str, &str> = [
            ("name", "x"),
            ("wandb_project", "p"),
            ("tags", "[a]"),
            ("model.rope_aspect", "square"),
            ("model.repa_layer", "4"),
            ("flow.shift", "2.0"),
            ("flow.timestep_sampler", "uniform"),
            ("flow.logit_mean", "0.5"),
            ("flow.logit_std", "2.0"),
            ("flow.prediction", "x"),
            ("flow.x_pred_sigma_min", "0.1"),
            ("flow.shift_law", "sd3"),
            ("flow.shift_base_tokens", "1024"),
            ("text_encoder.on_caption_overflow", "error"),
            ("repa.weight", "0.5"),
            ("repa.variant", "irepa"),
            ("repa.spatial_norm_gamma", "0.5"),
            ("repa.teacher_source", "hf"),
            ("repa.teacher_hub", "x/y"),
            ("repa.teacher", "dinov2_vits14"),
            ("repa.teacher_weights", "/w"),
            ("repa.teacher_dim", "384"),
            ("repa.proj_hidden_dim", "64"),
            ("repa.teacher_image_size", "448"),
            ("repa.teacher_patch_size", "16"),
            ("repa.teacher_match_student", "true"),
            ("data.type", "pixel_multiscale"),
            ("data.image_size", "256"),
            ("data.shape_policy", "area"),
            ("data.aspect_ratio_bucket", "shared21-1024"),
            ("data.shape_align", "64"),
            ("data.shape_max_ratio", "2.0"),
            ("data.caption_field", "alt"),
            ("data.caption_fields", "[a, b]"),
            ("train.batch_size", "3"),
            ("train.grad_accum_steps", "2"),
            ("train.expected_global_batch", "64"),
            ("train.num_epochs", "7"),
            ("train.max_steps", "9"),
            ("train.mixed_precision", "no"),
            ("train.activation_checkpointing", "full"),
            ("train.ac_selective_every", "3"),
            ("train.gradient_clip", "1.0"),
            ("train.text_dropout", "0.2"),
            ("train.seed", "9"),
            ("train.optimizer.name", "adamw"),
            ("train.optimizer.lr", "0.5"),
            ("train.optimizer.betas", "[0.8, 0.9]"),
            ("train.optimizer.weight_decay", "0.1"),
            ("train.optimizer.muon_momentum", "0.9"),
            ("train.optimizer.muon_nesterov", "false"),
            ("train.optimizer.muon_adjust_lr", "none"),
            ("train.optimizer.auto_lr", "sqrt"),
            ("train.optimizer.base_batch_size", "64"),
            ("train.optimizer.schedule", "cosine"),
            ("train.optimizer.warmup_steps", "5"),
            ("train.optimizer.scale_warmup_by_world", "false"),
            ("train.ema.enabled", "false"),
            ("train.ema.decay", "0.5"),
            ("train.dist.strategy", "fsdp2"),
            ("train.dist.sharding", "hybrid"),
            ("train.dist.mesh.dp_shard", "2"),
            ("train.dist.mesh.dp_replicate", "2"),
            ("train.dist.param_dtype", "fp32"),
            ("train.dist.reduce_dtype", "bf16"),
            ("train.dist.reshard_after_forward", "false"),
            ("train.dist.checkpoint_io", "dcp"),
            ("train.dist.ddp.find_unused_parameters", "true"),
            ("train.dist.ddp.gradient_as_bucket_view", "true"),
            ("train.dist.ddp.batched_grad_copy", "true"),
            ("train.dist.ddp.skip_all_reduce_unused_params", "true"),
            ("train.dist.ddp.bucket_cap_mb", "50"),
            ("train.dist.ddp.forward_sync_buffers", "false"),
            ("train.load_from", "/ckpt"),
            ("train.resume_from", "latest"),
            ("train.resume_data_policy", "new_phase"),
            ("train.resume_optimizer", "core"),
            ("train.override_lr_on_resume", "true"),
            ("train.save_every_steps", "7"),
            ("train.save_every_epochs", "2"),
            ("train.keep_last_checkpoints", "2"),
            ("train.milestone_steps", "[4]"),
            ("train.log_every", "3"),
            ("train.log_block_grad_norms", "true"),
            ("train.sample_every_steps", "8"),
            ("train.val_every_steps", "7"),
            ("train.val_samples", "4"),
            ("train.validation_prompts", "[a cat]"),
            ("train.nan_loss_tolerance", "3"),
            ("train.perf.lazy_sync", "true"),
            ("sample.order", "1"),
        ]
        .into_iter()
        .collect();
        for o in UPSTREAM_OPTIONS {
            if let Disposition::Honored(_) = o.disposition {
                if o.key.starts_with("cli.") {
                    continue;
                }
                assert!(
                    perturbed.contains_key(o.key),
                    "honored {} not exercised",
                    o.key
                );
            }
        }
        for (k, v) in &perturbed {
            let o = format!("{k}={v}");
            let t = translate_upstream(None, &[o.to_string()], TrainingConfig::default()).unwrap();
            assert_ne!(t.config, base.config, "{o} changed nothing");
        }
        let t = translate_upstream(
            None,
            &["train.perf.compile=true".into()],
            TrainingConfig::default(),
        )
        .unwrap();
        assert_eq!(t.mapped.len(), 1);
        assert_eq!(t.mapped[0].0, "train.perf.compile");
        let t = translate_upstream(
            None,
            &["model.hidden_size=64".into()],
            TrainingConfig::default(),
        )
        .unwrap();
        assert_eq!(opts(&t.config)["architecture"]["hidden_size"], json!(64));
    }
}

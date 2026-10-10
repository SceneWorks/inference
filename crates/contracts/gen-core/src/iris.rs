//! Backend-neutral contract for the **Iris-3B** family (epic sc-25678): task identity, the
//! per-task resource layout, the release `config.yaml`, the Qwen3-VL conditioning window, the
//! FlowDPM-Solver++ step plan and the generation control surface.
//!
//! Everything here is host-side (no tensors), so the MLX provider (`mlx-gen-iris`), a Candle twin
//! and the SceneWorks worker read one definition. The tensor work lives in the backend crates.
//!
//! Frozen upstream: `speridlabs/iris-3b` code @ [`UPSTREAM_CODE_REVISION`], weights @
//! [`UPSTREAM_WEIGHTS_REVISION`], text encoder `Qwen/Qwen3-VL-4B-Instruct` @
//! [`TEXT_ENCODER_REVISION`].

use std::path::{Path, PathBuf};

use crate::generator::GenerationRequest;
use crate::runtime::{LoadSpec, WeightsSource};
use crate::{Error, Result};

/// GitHub `speridlabs/iris-3b` commit the port mirrors (model, pipeline, text conditioning, solver).
pub const UPSTREAM_CODE_REVISION: &str = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc";
/// Hugging Face `speridlabs/iris-3b` revision (weights + `config.yaml` for every task).
pub const UPSTREAM_WEIGHTS_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";
/// Hugging Face repository of the weights.
pub const UPSTREAM_WEIGHTS_REPO: &str = "speridlabs/iris-3b";
/// The generation task's text encoder repository.
pub const TEXT_ENCODER_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
/// Pinned text-encoder revision — tokenizer, config and shards are taken together from it.
pub const TEXT_ENCODER_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

/// Registry id of the Iris-3B text-to-image route (the SceneWorks worker's `payload.model`).
pub const GENERATION_MODEL_ID: &str = "iris_3b";
/// Descriptor family shared by every Iris task and backend.
pub const FAMILY: &str = "iris";

/// `LoadSpec::components` key carrying the Qwen3-VL text-encoder directory of the generation task.
pub const TEXT_ENCODER_COMPONENT: &str = "text_encoder";
/// Backbone config file, as `scripts/export_checkpoint.py` writes it.
pub const BACKBONE_CONFIG_FILE: &str = "config.yaml";
/// Backbone weights file (the `IrisDiT` state dict with upstream key names).
pub const BACKBONE_WEIGHTS_FILE: &str = "model.safetensors";

/// Release sampling defaults (`SampleConfig` / `scripts/sample.py`).
pub const DEFAULT_STEPS: u32 = 100;
/// DPM-Solver++ order (`sample.order`). Not a request knob in this release surface.
pub const DEFAULT_SOLVER_ORDER: usize = 2;
/// Classifier-free guidance scale (`sample.cfg_scale`).
pub const DEFAULT_CFG_SCALE: f32 = 3.0;
/// CFG gate: guidance applies at model-evaluation times strictly inside this interval.
pub const DEFAULT_CFG_INTERVAL: (f64, f64) = (0.0, 1.0);
/// Default output side (`--height` / `--width`).
pub const DEFAULT_SIZE: u32 = 1024;
/// Output sides are admitted on this pixel grid (the release's `model.patch_size`). A backbone
/// whose patch does not divide it is refused at load, so an admitted size always patchifies.
pub const SIZE_MULTIPLE: u32 = 16;

/// The three tasks of the release. Each is its own backbone checkpoint; only generation conditions
/// on text, so only generation names the text encoder as a resource (E4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrisTask {
    /// Text-to-image: the repo-root backbone + Qwen3-VL-4B.
    Generation,
    /// Monocular depth (`depth/`): one forward pass with the shipped empty-prompt states.
    Depth,
    /// Restoration / upscaling (`upscaler/`): one forward pass with the shipped empty-prompt states.
    Restoration,
}

impl IrisTask {
    /// Sub-directory of the weights repo holding this task's `config.yaml` + `model.safetensors`.
    pub fn backbone_subdir(self) -> &'static str {
        match self {
            IrisTask::Generation => "",
            IrisTask::Depth => "depth",
            IrisTask::Restoration => "upscaler",
        }
    }

    /// Whether the task's resources include the Qwen3-VL text encoder.
    pub fn uses_text_encoder(self) -> bool {
        matches!(self, IrisTask::Generation)
    }

    pub fn name(self) -> &'static str {
        match self {
            IrisTask::Generation => "generation",
            IrisTask::Depth => "depth",
            IrisTask::Restoration => "restoration",
        }
    }
}

/// The resolved, existence-checked resources of the generation task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationResources {
    /// Directory with `config.yaml` + `model.safetensors`.
    pub backbone_dir: PathBuf,
    /// The `Qwen/Qwen3-VL-4B-Instruct` snapshot directory.
    pub text_encoder_dir: PathBuf,
}

/// Files the text-encoder resource must carry besides its safetensors.
pub const TEXT_ENCODER_REQUIRED_FILES: [&str; 3] =
    ["config.json", "tokenizer.json", "tokenizer_config.json"];

impl GenerationResources {
    /// Resolve and check the generation task's resources from a load spec: `spec.weights` is the
    /// backbone directory, `spec.components["text_encoder"]` the Qwen3-VL snapshot. Every missing
    /// piece is a load-time error naming the path and the resource it belongs to.
    pub fn from_spec(spec: &LoadSpec, model_id: &str) -> Result<Self> {
        crate::control::reject_unknown_components(spec, &[TEXT_ENCODER_COMPONENT], model_id)?;
        let backbone_dir = match &spec.weights {
            WeightsSource::Dir(dir) => dir.clone(),
            WeightsSource::File(file) => {
                return Err(Error::Msg(format!(
                    "{model_id}: the generation backbone resource must be a directory holding \
                     {BACKBONE_CONFIG_FILE} + {BACKBONE_WEIGHTS_FILE} (the {UPSTREAM_WEIGHTS_REPO} \
                     root), not the single file {}",
                    file.display()
                )))
            }
        };
        let text_encoder_dir = match spec.components.get(TEXT_ENCODER_COMPONENT) {
            Some(WeightsSource::Dir(dir)) => dir.clone(),
            Some(WeightsSource::File(file)) => {
                return Err(Error::Msg(format!(
                    "{model_id}: the '{TEXT_ENCODER_COMPONENT}' resource must be the \
                     {TEXT_ENCODER_REPO} snapshot directory, not the single file {}",
                    file.display()
                )))
            }
            None => {
                return Err(Error::Msg(format!(
                    "{model_id}: the generation task needs its text encoder — stage the \
                     {TEXT_ENCODER_REPO}@{TEXT_ENCODER_REVISION} snapshot directory as the \
                     '{TEXT_ENCODER_COMPONENT}' component"
                )))
            }
        };
        let resources = Self {
            backbone_dir,
            text_encoder_dir,
        };
        resources.check(model_id)?;
        Ok(resources)
    }

    /// Existence check of every file the generation task reads. Shard completeness is checked by
    /// the backend when it resolves the shard index.
    pub fn check(&self, model_id: &str) -> Result<()> {
        require_dir(&self.backbone_dir, model_id, "generation backbone")?;
        for name in [BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE] {
            require_file(
                &self.backbone_dir.join(name),
                model_id,
                "generation backbone",
            )?;
        }
        require_dir(&self.text_encoder_dir, model_id, "text encoder")?;
        for name in TEXT_ENCODER_REQUIRED_FILES {
            require_file(&self.text_encoder_dir.join(name), model_id, "text encoder")?;
        }
        let index = self.text_encoder_dir.join("model.safetensors.index.json");
        let single = self.text_encoder_dir.join("model.safetensors");
        if !index.is_file() && !single.is_file() {
            return Err(Error::Msg(format!(
                "{model_id}: the text encoder resource {} has neither \
                 model.safetensors.index.json nor model.safetensors — the {TEXT_ENCODER_REPO} \
                 snapshot is incomplete",
                self.text_encoder_dir.display()
            )));
        }
        Ok(())
    }
}

fn require_dir(dir: &Path, model_id: &str, resource: &str) -> Result<()> {
    if dir.is_dir() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "{model_id}: the {resource} resource directory {} does not exist",
            dir.display()
        )))
    }
}

fn require_file(path: &Path, model_id: &str, resource: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "{model_id}: the {resource} resource is incomplete — {} is missing",
            path.display()
        )))
    }
}

// ---------------------------------------------------------------------------------------------
// config.yaml
// ---------------------------------------------------------------------------------------------

/// `model.pixel` (`PixelStageConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct PixelStageConfig {
    pub enabled: bool,
    pub depth: usize,
    pub hidden_size: usize,
    pub attn_hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub modulation: String,
    pub abs_pos_embed: bool,
}

/// `model` (`ModelConfig`) — every field that changes the inference graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub block: String,
    pub dual_depth: usize,
    pub final_block_text: String,
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub gated_attention: bool,
    pub sandwich_norm: bool,
    pub patch_size: usize,
    pub in_channels: usize,
    pub mlp_ratio: f64,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub norm_eps: f64,
    pub modulation: String,
    pub timestep_max_period: f64,
    pub rope_theta: f64,
    pub rope_scale: f64,
    pub rope_aspect: String,
    pub rope_frame_pairs: usize,
    pub text_rope: bool,
    pub text_rope_theta: f64,
    pub text_abs_pos_embed: bool,
    pub text_dim: usize,
    pub text_len: usize,
    pub text_adapter: String,
    pub text_lap_num_layers: usize,
    pub text_lap_num_heads: usize,
    pub text_lap_mlp_ratio: f64,
    pub pixel: PixelStageConfig,
}

/// `text_encoder` (`TextEncoderConfig`) — the conditioning contract the backbone was trained on.
#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderConfig {
    pub name: String,
    pub pretrained: String,
    pub dim: usize,
    pub max_length: usize,
    pub dtype: String,
    /// 1-based post-block hidden-state indices, sorted and unique.
    pub hidden_layers: Vec<usize>,
    pub on_caption_overflow: String,
}

/// `flow` (`FlowConfig`) — the inference-relevant part.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub prediction: String,
    pub shift_law: String,
}

/// The parsed `config.yaml` of one task backbone (`model` / `text_encoder` / `flow` sections).
#[derive(Clone, Debug, PartialEq)]
pub struct IrisConfig {
    pub model: ModelConfig,
    pub text_encoder: TextEncoderConfig,
    pub flow: FlowConfig,
}

impl IrisConfig {
    /// Read `dir/config.yaml`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(BACKBONE_CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("iris: read {}: {e}", path.display())))?;
        Self::parse(&text).map_err(|e| Error::Msg(format!("iris: {}: {e}", path.display())))
    }

    /// Parse the YAML text. Keys absent from the file take upstream's structured defaults (the
    /// release `ModelConfig()` / `TextEncoderConfig()` / `FlowConfig()`), exactly as
    /// `iris3b.config.inference_config` merges a checkpoint's sections over the schema.
    pub fn parse(text: &str) -> Result<Self> {
        let root = yaml::parse(text)?;
        let model = root.get("model");
        let pixel = model.and_then(|m| m.get("pixel"));
        let te = root.get("text_encoder");
        let flow = root.get("flow");
        let m = Section(model);
        let p = Section(pixel);
        let t = Section(te);
        let f = Section(flow);
        let num_kv_heads = match model.and_then(|m| m.get("num_kv_heads")) {
            None => Some(5),
            Some(v) if v.is_null() => None,
            Some(v) => Some(v.as_usize("model.num_kv_heads")?),
        };
        Ok(Self {
            model: ModelConfig {
                block: m.string("block", "single_stream")?,
                dual_depth: m.usize("dual_depth", 8)?,
                final_block_text: m.string("final_block_text", "keep")?,
                hidden_size: m.usize("hidden_size", 2560)?,
                depth: m.usize("depth", 24)?,
                num_heads: m.usize("num_heads", 20)?,
                num_kv_heads,
                gated_attention: m.bool("gated_attention", true)?,
                sandwich_norm: m.bool("sandwich_norm", true)?,
                patch_size: m.usize("patch_size", 16)?,
                in_channels: m.usize("in_channels", 3)?,
                mlp_ratio: m.f64("mlp_ratio", 4.0)?,
                qkv_bias: m.bool("qkv_bias", false)?,
                qk_norm: m.bool("qk_norm", true)?,
                norm_eps: m.f64("norm_eps", 1e-6)?,
                modulation: m.string("modulation", "shared_bias")?,
                timestep_max_period: m.f64("timestep_max_period", 10.0)?,
                rope_theta: m.f64("rope_theta", 10_000.0)?,
                rope_scale: m.f64("rope_scale", 16.0)?,
                rope_aspect: m.string("rope_aspect", "isotropic")?,
                rope_frame_pairs: m.usize("rope_frame_pairs", 0)?,
                text_rope: m.bool("text_rope", true)?,
                text_rope_theta: m.f64("text_rope_theta", 10_000.0)?,
                text_abs_pos_embed: m.bool("text_abs_pos_embed", true)?,
                text_dim: m.usize("text_dim", 2560)?,
                text_len: m.usize("text_len", 300)?,
                text_adapter: m.string("text_adapter", "lap_blocks2")?,
                text_lap_num_layers: m.usize("text_lap_num_layers", 12)?,
                text_lap_num_heads: m.usize("text_lap_num_heads", 32)?,
                text_lap_mlp_ratio: m.f64("text_lap_mlp_ratio", 1.3)?,
                pixel: PixelStageConfig {
                    enabled: p.bool("enabled", true)?,
                    depth: p.usize("depth", 4)?,
                    hidden_size: p.usize("hidden_size", 16)?,
                    attn_hidden_size: p.usize("attn_hidden_size", 1280)?,
                    num_heads: p.usize("num_heads", 10)?,
                    mlp_ratio: p.f64("mlp_ratio", 4.0)?,
                    modulation: p.string("modulation", "post")?,
                    abs_pos_embed: p.bool("abs_pos_embed", true)?,
                },
            },
            text_encoder: TextEncoderConfig {
                name: t.string("name", "qwen3_vl")?,
                pretrained: t.string("pretrained", TEXT_ENCODER_REPO)?,
                dim: t.usize("dim", 2560)?,
                max_length: t.usize("max_length", 300)?,
                dtype: t.string("dtype", "bfloat16")?,
                hidden_layers: match te.and_then(|v| v.get("hidden_layers")) {
                    None => vec![2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                    Some(v) => v.as_usize_list("text_encoder.hidden_layers")?,
                },
                on_caption_overflow: t.string("on_caption_overflow", "warn")?,
            },
            flow: FlowConfig {
                num_train_timesteps: f.usize("num_train_timesteps", 1000)?,
                shift: f.f64("shift", 4.0)?,
                prediction: f.string("prediction", "v")?,
                shift_law: f.string("shift_law", "none")?,
            },
        })
    }

    /// Refuse any config the native graph does not implement, naming the key — never a silent
    /// mis-render. The supported set is exactly the released architecture's switches; every width
    /// and depth is read from the file.
    pub fn validate_supported(&self) -> Result<()> {
        let m = &self.model;
        let checks: [(&str, bool, String); 25] = [
            ("model.block", m.block == "single_stream", m.block.clone()),
            (
                // `dual_depth: 0` renames the shared modulation core (`adaln_shared`); the native
                // graph implements the hybrid dual → single-stream trunk only.
                "model.dual_depth",
                1 <= m.dual_depth && m.dual_depth <= m.depth,
                m.dual_depth.to_string(),
            ),
            (
                "model.patch_size",
                m.patch_size > 0 && (SIZE_MULTIPLE as usize).is_multiple_of(m.patch_size),
                m.patch_size.to_string(),
            ),
            (
                "model.final_block_text",
                matches!(m.final_block_text.as_str(), "keep" | "drop"),
                m.final_block_text.clone(),
            ),
            (
                "model.modulation",
                m.modulation == "shared_bias",
                m.modulation.clone(),
            ),
            (
                "model.gated_attention",
                m.gated_attention,
                m.gated_attention.to_string(),
            ),
            (
                "model.sandwich_norm",
                m.sandwich_norm,
                m.sandwich_norm.to_string(),
            ),
            ("model.qkv_bias", !m.qkv_bias, m.qkv_bias.to_string()),
            ("model.qk_norm", m.qk_norm, m.qk_norm.to_string()),
            (
                "model.rope_aspect",
                m.rope_aspect == "isotropic",
                m.rope_aspect.clone(),
            ),
            (
                "model.rope_frame_pairs",
                m.rope_frame_pairs == 0,
                m.rope_frame_pairs.to_string(),
            ),
            ("model.text_rope", m.text_rope, m.text_rope.to_string()),
            (
                "model.text_abs_pos_embed",
                m.text_abs_pos_embed,
                m.text_abs_pos_embed.to_string(),
            ),
            (
                "model.text_adapter",
                m.text_adapter == "lap_blocks2",
                m.text_adapter.clone(),
            ),
            (
                "model.pixel.enabled",
                m.pixel.enabled,
                m.pixel.enabled.to_string(),
            ),
            (
                "model.pixel.modulation",
                m.pixel.modulation == "post",
                m.pixel.modulation.clone(),
            ),
            (
                "model.pixel.abs_pos_embed",
                m.pixel.abs_pos_embed,
                m.pixel.abs_pos_embed.to_string(),
            ),
            (
                "flow.prediction",
                self.flow.prediction == "v",
                self.flow.prediction.clone(),
            ),
            (
                "flow.shift_law",
                self.flow.shift_law == "none",
                self.flow.shift_law.clone(),
            ),
            (
                "text_encoder.name",
                self.text_encoder.name == "qwen3_vl",
                self.text_encoder.name.clone(),
            ),
            (
                // The Qwen3-VL tower always computes in bf16 (the release's dtype).
                "text_encoder.dtype",
                self.text_encoder.dtype == "bfloat16",
                self.text_encoder.dtype.clone(),
            ),
            (
                // The DiT reads exactly the window the encoder produces.
                "text_encoder.max_length",
                self.text_encoder.max_length == m.text_len,
                self.text_encoder.max_length.to_string(),
            ),
            (
                "text_encoder.on_caption_overflow",
                matches!(
                    self.text_encoder.on_caption_overflow.as_str(),
                    "warn" | "silent"
                ),
                self.text_encoder.on_caption_overflow.clone(),
            ),
            (
                "model.text_lap_num_layers",
                m.text_lap_num_layers == self.text_encoder.hidden_layers.len(),
                m.text_lap_num_layers.to_string(),
            ),
            (
                "model.text_dim",
                m.text_dim == self.text_encoder.dim,
                m.text_dim.to_string(),
            ),
        ];
        for (key, ok, value) in checks {
            if !ok {
                return Err(Error::Unsupported(format!(
                    "iris: {key} = {value} is not implemented by the native port (the released \
                     Iris-3B architecture is the supported set)"
                )));
            }
        }
        let layers = &self.text_encoder.hidden_layers;
        if layers.is_empty() || layers.windows(2).any(|w| w[0] >= w[1]) || layers[0] < 1 {
            return Err(Error::Msg(format!(
                "iris: text_encoder.hidden_layers must be sorted, unique and 1-based, got {layers:?}"
            )));
        }
        let heads_ok = m.hidden_size.is_multiple_of(m.num_heads)
            && m.num_kv_heads
                .is_none_or(|kv| kv > 0 && m.num_heads.is_multiple_of(kv))
            && m.pixel.attn_hidden_size.is_multiple_of(m.pixel.num_heads)
            && m.text_dim.is_multiple_of(m.text_lap_num_heads);
        if !heads_ok {
            return Err(Error::Msg(
                "iris: head counts do not divide the configured widths".into(),
            ));
        }
        Ok(())
    }

    /// `model.text_len` is the window the DiT reads; the encoder must produce at least that many.
    pub fn conditioning_window(&self) -> usize {
        self.text_encoder.max_length.min(self.model.text_len)
    }
}

struct Section<'a>(Option<&'a yaml::Value>);

impl Section<'_> {
    fn raw(&self, key: &str) -> Option<&yaml::Value> {
        self.0.and_then(|v| v.get(key))
    }
    fn string(&self, key: &str, default: &str) -> Result<String> {
        Ok(self
            .raw(key)
            .map(|v| v.as_str(key).map(str::to_owned))
            .transpose()?
            .unwrap_or_else(|| default.to_owned()))
    }
    fn usize(&self, key: &str, default: usize) -> Result<usize> {
        self.raw(key)
            .map(|v| v.as_usize(key))
            .transpose()
            .map(|v| v.unwrap_or(default))
    }
    fn f64(&self, key: &str, default: f64) -> Result<f64> {
        self.raw(key)
            .map(|v| v.as_f64(key))
            .transpose()
            .map(|v| v.unwrap_or(default))
    }
    fn bool(&self, key: &str, default: bool) -> Result<bool> {
        self.raw(key)
            .map(|v| v.as_bool(key))
            .transpose()
            .map(|v| v.unwrap_or(default))
    }
}

/// A strict reader for the YAML subset OmegaConf writes for these configs: block mappings by
/// indentation, `key: scalar`, and `- scalar` sequences. Anything else (flow collections, anchors,
/// multi-line scalars) is an error rather than a guess.
mod yaml {
    use crate::{Error, Result};

    #[derive(Clone, Debug, PartialEq)]
    pub enum Value {
        Map(Vec<(String, Value)>),
        List(Vec<Value>),
        Scalar(String),
    }

    impl Value {
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        pub fn is_null(&self) -> bool {
            matches!(self, Value::Scalar(s) if matches!(s.as_str(), "null" | "~" | "None"))
        }
        fn scalar(&self, key: &str) -> Result<&str> {
            match self {
                Value::Scalar(s) => Ok(s),
                _ => Err(Error::Msg(format!("{key}: expected a scalar"))),
            }
        }
        pub fn as_str(&self, key: &str) -> Result<&str> {
            self.scalar(key)
        }
        pub fn as_usize(&self, key: &str) -> Result<usize> {
            let s = self.scalar(key)?;
            s.parse().map_err(|_| {
                Error::Msg(format!("{key}: expected a non-negative integer, got {s:?}"))
            })
        }
        pub fn as_f64(&self, key: &str) -> Result<f64> {
            let s = self.scalar(key)?;
            s.parse()
                .map_err(|_| Error::Msg(format!("{key}: expected a number, got {s:?}")))
        }
        pub fn as_bool(&self, key: &str) -> Result<bool> {
            match self.scalar(key)? {
                "true" | "True" => Ok(true),
                "false" | "False" => Ok(false),
                s => Err(Error::Msg(format!("{key}: expected true/false, got {s:?}"))),
            }
        }
        pub fn as_usize_list(&self, key: &str) -> Result<Vec<usize>> {
            match self {
                Value::List(items) => items.iter().map(|v| v.as_usize(key)).collect(),
                _ => Err(Error::Msg(format!("{key}: expected a list"))),
            }
        }
    }

    struct Line<'a> {
        indent: usize,
        text: &'a str,
        number: usize,
    }

    pub fn parse(text: &str) -> Result<Value> {
        let lines: Vec<Line> = text
            .lines()
            .enumerate()
            .filter_map(|(i, raw)| {
                let body = strip_comment(raw).trim_end();
                let trimmed = body.trim_start();
                (!trimmed.is_empty() && trimmed != "---").then(|| Line {
                    indent: body.len() - trimmed.len(),
                    text: trimmed,
                    number: i + 1,
                })
            })
            .collect();
        let mut pos = 0;
        let value = parse_map(&lines, &mut pos, 0)?;
        if pos != lines.len() {
            return Err(Error::Msg(format!(
                "yaml line {}: unexpected indentation",
                lines[pos].number
            )));
        }
        Ok(value)
    }

    fn strip_comment(line: &str) -> &str {
        let mut quote = None;
        for (i, c) in line.char_indices() {
            match (quote, c) {
                (None, '\'' | '"') => quote = Some(c),
                (Some(q), c) if c == q => quote = None,
                (None, '#') if i == 0 || line[..i].ends_with(' ') => return &line[..i],
                _ => {}
            }
        }
        line
    }

    fn unquote(s: &str) -> Result<String> {
        let s = s.trim();
        if s.starts_with('{') || s.starts_with('[') || s.starts_with('&') || s.starts_with('*') {
            return Err(Error::Msg(format!("unsupported yaml value {s:?}")));
        }
        for q in ['\'', '"'] {
            if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
                return Ok(s[1..s.len() - 1].to_owned());
            }
        }
        Ok(s.to_owned())
    }

    fn parse_map(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Value> {
        let mut entries = Vec::new();
        while *pos < lines.len() {
            let line = &lines[*pos];
            if line.indent < indent {
                break;
            }
            if line.indent > indent || line.text.starts_with("- ") || line.text == "-" {
                return Err(Error::Msg(format!(
                    "yaml line {}: expected a `key:` at indent {indent}",
                    line.number
                )));
            }
            let (key, rest) = line.text.split_once(':').ok_or_else(|| {
                Error::Msg(format!("yaml line {}: expected `key: value`", line.number))
            })?;
            let key = unquote(key)?;
            if entries.iter().any(|(k, _)| k == &key) {
                return Err(Error::Msg(format!(
                    "yaml line {}: duplicate key {key:?}",
                    line.number
                )));
            }
            *pos += 1;
            let rest = rest.trim();
            let value = if !rest.is_empty() {
                Value::Scalar(unquote(rest)?)
            } else if *pos < lines.len()
                && lines[*pos].indent >= indent
                && (lines[*pos].text.starts_with("- ") || lines[*pos].text == "-")
            {
                parse_list(lines, pos, lines[*pos].indent)?
            } else if *pos < lines.len() && lines[*pos].indent > indent {
                parse_map(lines, pos, lines[*pos].indent)?
            } else {
                Value::Scalar("null".into())
            };
            entries.push((key, value));
        }
        Ok(Value::Map(entries))
    }

    fn parse_list(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Value> {
        let mut items = Vec::new();
        while *pos < lines.len() && lines[*pos].indent == indent {
            let line = &lines[*pos];
            let Some(item) = line.text.strip_prefix('-') else {
                break;
            };
            let item = item.trim();
            if item.is_empty() || item.contains(": ") {
                return Err(Error::Msg(format!(
                    "yaml line {}: only scalar list items are supported",
                    line.number
                )));
            }
            items.push(Value::Scalar(unquote(item)?));
            *pos += 1;
        }
        Ok(Value::List(items))
    }
}

// ---------------------------------------------------------------------------------------------
// Qwen3-VL conditioning window
// ---------------------------------------------------------------------------------------------

/// `iris3b/text/qwen3_vl.py` `_PROMPT_PREFIX`, verbatim.
pub const PROMPT_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, \
    shape, size, texture, quantity, text, spatial relationships of the objects and \
    background:<|im_end|>\n<|im_start|>user\n";
/// `iris3b/text/qwen3_vl.py` `_PROMPT_SUFFIX`, verbatim.
pub const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// One prompt's assembled encoder input, as upstream's `Qwen3VLTextEncoder._run` builds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextWindow {
    /// The **real** tokens: prefix ‖ caption (truncated to the budget) ‖ suffix. Upstream right-pads
    /// this to `prefix + max_length` with the pad id; the tower is causal and the pads are trailing,
    /// so the real rows are identical with or without the pads and the backend runs only these.
    pub input_ids: Vec<i32>,
    /// Leading template tokens dropped from the hidden states (`start = len(prefix)`).
    pub prefix_len: usize,
    /// `[max_length]` 0/1 mask of the conditioning window (`attention_mask[:, start:stop]`).
    pub mask: Vec<i32>,
    /// Caption tokens dropped by the budget (upstream's overflow accounting).
    pub truncated_tokens: usize,
}

impl TextWindow {
    /// Real tokens inside the window (`caption + suffix`).
    pub fn window_tokens(&self) -> usize {
        self.input_ids.len() - self.prefix_len
    }
}

/// Assemble the window: the caption is cut to `max_length - len(suffix)` tokens **before** the
/// suffix is appended, so the assistant-turn marker can never be the part that is cut.
pub fn assemble_window(
    prefix_ids: &[i32],
    caption_ids: &[i32],
    suffix_ids: &[i32],
    max_length: usize,
) -> Result<TextWindow> {
    if prefix_ids.is_empty() || suffix_ids.is_empty() {
        return Err(Error::Msg(
            "iris: the Qwen3-VL prompt template produced an empty token prefix or suffix".into(),
        ));
    }
    if max_length <= suffix_ids.len() {
        return Err(Error::Msg(format!(
            "iris: text_encoder.max_length={max_length} leaves no room for a caption: the \
             assistant-turn suffix alone is {} tokens",
            suffix_ids.len()
        )));
    }
    let budget = max_length - suffix_ids.len();
    let n = caption_ids.len().min(budget);
    let mut input_ids = Vec::with_capacity(prefix_ids.len() + n + suffix_ids.len());
    input_ids.extend_from_slice(prefix_ids);
    input_ids.extend_from_slice(&caption_ids[..n]);
    input_ids.extend_from_slice(suffix_ids);
    let mut mask = vec![0i32; max_length];
    mask[..n + suffix_ids.len()].fill(1);
    Ok(TextWindow {
        input_ids,
        prefix_len: prefix_ids.len(),
        mask,
        truncated_tokens: caption_ids.len() - n,
    })
}

// ---------------------------------------------------------------------------------------------
// FlowDPM-Solver++
// ---------------------------------------------------------------------------------------------

/// `sigma' = shift·sigma / (1 + (shift − 1)·sigma)`.
pub fn shift_sigma(sigma: f64, shift: f64) -> f64 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// `FlowDPMSolver.time_grid`: `1 − linspace(1, 0.001, steps + 1)`, shifted, descending, ending at
/// exactly 0. Computed in f64 like upstream (`torch.linspace(..., dtype=float64)`).
pub fn time_grid(steps: usize, shift: f64) -> Vec<f64> {
    let n = steps + 1;
    let mut grid: Vec<f64> = (0..n)
        .map(|i| {
            // torch.linspace(start, end, n): start + i·step for the first half, end − (n−1−i)·step
            // for the second (its symmetric evaluation), step = (end − start)/(n − 1).
            let (start, end) = (1.0f64, 0.001f64);
            let lin = if n == 1 {
                start
            } else {
                let step = (end - start) / (n - 1) as f64;
                if i < n / 2 {
                    start + step * i as f64
                } else {
                    end - step * (n - 1 - i) as f64
                }
            };
            shift_sigma(1.0 - lin, shift)
        })
        .collect();
    grid.reverse();
    grid
}

fn lambda(t: f64) -> f64 {
    if t <= 0.0 {
        f64::INFINITY
    } else {
        ((1.0 - t) / t).ln()
    }
}

/// One solver step's update, with every scalar already in the f32 the tensors are multiplied by.
///
/// * `First`: `x ← cx·x − c0·x0`
/// * `Second`: `x ← cx·x − c0·x0 − c1·d`, `d = (x0 − x0_prev) / r0`
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DpmUpdate {
    First { cx: f32, c0: f32 },
    Second { cx: f32, c0: f32, c1: f32, r0: f32 },
}

/// One step of the plan: evaluate the model at `s`, then apply `update` to reach `t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpmStep {
    pub s: f64,
    pub t: f64,
    pub update: DpmUpdate,
}

impl DpmStep {
    /// Model time fed to the network (`t_model = s · num_timesteps`, f32).
    pub fn model_time(&self, num_timesteps: usize) -> f32 {
        (self.s * num_timesteps as f64) as f32
    }
    /// The `x0 = x − s·v` coefficient.
    pub fn s_f32(&self) -> f32 {
        self.s as f32
    }
}

/// The full multistep schedule of `FlowDPMSolver.sample(steps, order, shift)`: order ramps up over
/// the first steps and back down at the end (`lower_order_final`), so the terminal update is the
/// exact `x ← x0` projection. NFE == steps.
pub fn dpm_solver_plan(steps: usize, order: usize, shift: f64) -> Result<Vec<DpmStep>> {
    if steps == 0 {
        return Err(Error::Msg("iris: steps must be >= 1".into()));
    }
    if !(1..=2).contains(&order) {
        return Err(Error::Unsupported(format!(
            "iris: DPM-Solver++ order {order} is not implemented (upstream implements 1 and 2)"
        )));
    }
    let grid = time_grid(steps, shift);
    let mut plan = Vec::with_capacity(steps);
    for i in 1..=steps {
        let (s, t) = (grid[i - 1], grid[i]);
        let step_order = i.min(order).min(steps + 1 - i);
        let update = if step_order == 1 {
            let h = lambda(t) - lambda(s);
            let phi1 = (-h).exp_m1();
            DpmUpdate::First {
                cx: (t / s) as f32,
                c0: ((1.0 - t) * phi1) as f32,
            }
        } else {
            let s1 = grid[i - 2];
            let (lam_t, lam_0, lam_1) = (lambda(t), lambda(s), lambda(s1));
            let (h, h0) = (lam_t - lam_0, lam_0 - lam_1);
            let r0 = h0 / h;
            let phi1 = (-h).exp_m1();
            DpmUpdate::Second {
                cx: (t / s) as f32,
                c0: ((1.0 - t) * phi1) as f32,
                c1: (0.5 * (1.0 - t) * phi1) as f32,
                r0: r0 as f32,
            }
        };
        plan.push(DpmStep { s, t, update });
    }
    Ok(plan)
}

/// Whether CFG runs at model time `s` (`lo < s < hi`, upstream's strict gate).
pub fn cfg_active(cfg_scale: f32, s: f64, interval: (f64, f64)) -> bool {
    cfg_scale != 1.0 && interval.0 < s && s < interval.1
}

// ---------------------------------------------------------------------------------------------
// Generation control surface
// ---------------------------------------------------------------------------------------------

/// The resolved generation controls of one request (release defaults filled in).
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationParams {
    pub steps: usize,
    pub cfg_scale: f32,
    /// The CFG unconditional prompt — `""` (the training dropout null) unless a negative prompt is
    /// given. Encoded with the same template as the positive prompt.
    pub negative_prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
}

impl GenerationParams {
    /// Resolve the request's controls. `default_seed` supplies the seed when none is given.
    pub fn resolve(req: &GenerationRequest, default_seed: u64) -> Self {
        Self {
            steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            cfg_scale: req.guidance.unwrap_or(DEFAULT_CFG_SCALE),
            negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
            seed: req.seed.unwrap_or(default_seed),
            width: req.width,
            height: req.height,
        }
    }

    /// Whether the run needs the unconditional branch at all (`cfg_scale != 1`).
    pub fn uses_cfg(&self) -> bool {
        self.cfg_scale != 1.0
    }
}

/// Refuse every request field the Iris generation route does not honour, by name. The shared
/// `Capabilities::validate_request` floor already polices size, count, steps, negative prompt,
/// guidance/true_cfg support, sampler/scheduler/guidance-method membership and conditioning kinds;
/// this covers the remaining per-request knobs, so none is ever accepted and ignored.
pub fn reject_unhonored_generation_controls(model_id: &str, req: &GenerationRequest) -> Result<()> {
    let set: [(&str, bool); 25] = [
        ("scheduler_shift", req.scheduler_shift.is_some()),
        ("timestep_to_start_cfg", req.timestep_to_start_cfg.is_some()),
        ("guidance_eta", req.guidance_eta.is_some()),
        ("guidance_momentum", req.guidance_momentum.is_some()),
        (
            "guidance_norm_threshold",
            req.guidance_norm_threshold.is_some(),
        ),
        ("strength", req.strength.is_some()),
        ("control_scale", req.control_scale.is_some()),
        ("text_style_gain", req.text_style_gain.is_some()),
        ("image_guidance", req.image_guidance.is_some()),
        ("frames", req.frames.is_some()),
        ("fps", req.fps.is_some()),
        ("duration", req.duration.is_some()),
        ("video_mode", req.video_mode.is_some()),
        ("trim_first_frames", req.trim_first_frames.is_some()),
        (
            "reference_image_short_edge",
            req.reference_image_short_edge.is_some(),
        ),
        ("motion_bucket_id", req.motion_bucket_id.is_some()),
        ("noise_aug_strength", req.noise_aug_strength.is_some()),
        ("decode_chunk_size", req.decode_chunk_size.is_some()),
        ("conditioning_fps", req.conditioning_fps.is_some()),
        ("softness", req.softness.is_some()),
        ("use_pid", req.use_pid),
        ("pid_capture_sigma", req.pid_capture_sigma.is_some()),
        ("use_uncensored_enhancer", req.use_uncensored_enhancer),
        ("phases", req.phases.is_some()),
        ("audio", req.audio.is_some()),
    ];
    for (field, is_set) in set {
        if is_set {
            return Err(Error::Unsupported(format!(
                "{model_id}: `{field}` is not a control of the Iris-3B generation route; drop it \
                 from the request"
            )));
        }
    }
    // At `cfg_scale == 1` the unconditional branch is never evaluated, so a negative prompt would
    // be accepted and silently dropped.
    let negative = req
        .negative_prompt
        .as_deref()
        .is_some_and(|p| !p.is_empty());
    if negative && req.guidance.unwrap_or(DEFAULT_CFG_SCALE) == 1.0 {
        return Err(Error::Unsupported(format!(
            "{model_id}: `negative_prompt` has no effect at guidance 1.0 (classifier-free guidance \
             is off); raise guidance or drop the negative prompt"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE_CONFIG: &str = "model:
  block: single_stream
  dual_depth: 8
  final_block_text: keep
  hidden_size: 2560
  depth: 24
  num_heads: 20
  num_kv_heads: 5
  gated_attention: true
  sandwich_norm: true
  patch_size: 16
  in_channels: 3
  mlp_ratio: 4.0
  qkv_bias: false
  qk_norm: true
  norm_eps: 1.0e-06
  attn_backend: sdpa
  modulation: shared_bias
  modulation_rank: 64
  timestep_max_period: 10.0
  adaln_zero_init: true
  rope_theta: 10000.0
  rope_scale: 16.0
  rope_aspect: isotropic
  rope_frame_pairs: 0
  rope_frame_theta: 10.0
  text_rope: true
  text_rope_theta: 10000.0
  text_abs_pos_embed: true
  text_dim: 2560
  text_len: 300
  text_adapter: lap_blocks2
  text_lap_num_layers: 12
  text_lap_num_heads: 32
  text_lap_mlp_ratio: 1.3
  repa_layer: 10
  pixel:
    enabled: true
    depth: 4
    hidden_size: 16
    attn_hidden_size: 1280
    num_heads: 10
    mlp_ratio: 4.0
    modulation: post
    abs_pos_embed: true
text_encoder:
  name: qwen3_vl
  pretrained: Qwen/Qwen3-VL-4B-Instruct
  dim: 2560
  max_length: 300
  dtype: bfloat16
  attn_implementation: sdpa
  hidden_layers:
  - 2
  - 5
  - 8
  - 11
  - 14
  - 17
  - 20
  - 23
  - 26
  - 29
  - 32
  - 35
  compile: false
  on_caption_overflow: warn
flow:
  num_train_timesteps: 1000
  shift: 4.0
  timestep_sampler: logit_normal
  logit_mean: 0.0
  logit_std: 1.0
  prediction: v
  x_pred_sigma_min: 0.05
  shift_law: none
  shift_base_tokens: 256
";

    #[test]
    fn release_config_parses_and_equals_the_schema_defaults() {
        let parsed = IrisConfig::parse(RELEASE_CONFIG).unwrap();
        parsed.validate_supported().unwrap();
        // The release config is exactly `ModelConfig()` etc., so an empty file must parse to the
        // same thing (the inference_config merge-over-defaults law).
        let defaults = IrisConfig::parse("model:\n  block: single_stream\n").unwrap();
        assert_eq!(parsed, defaults);
        assert_eq!(parsed.model.num_kv_heads, Some(5));
        assert_eq!(parsed.text_encoder.hidden_layers.len(), 12);
        assert_eq!(parsed.conditioning_window(), 300);
    }

    #[test]
    fn unsupported_switches_are_typed_refusals() {
        let text = RELEASE_CONFIG.replace("rope_aspect: isotropic", "rope_aspect: square");
        let err = IrisConfig::parse(&text)
            .unwrap()
            .validate_supported()
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(m) if m.contains("model.rope_aspect")));
        let text = RELEASE_CONFIG.replace("prediction: v", "prediction: x");
        let err = IrisConfig::parse(&text)
            .unwrap()
            .validate_supported()
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(m) if m.contains("flow.prediction")));
        for (from, to, key) in [
            ("dtype: bfloat16", "dtype: float16", "text_encoder.dtype"),
            (
                "max_length: 300",
                "max_length: 256",
                "text_encoder.max_length",
            ),
            ("dual_depth: 8", "dual_depth: 0", "model.dual_depth"),
            ("patch_size: 16", "patch_size: 32", "model.patch_size"),
        ] {
            let text = RELEASE_CONFIG.replace(from, to);
            assert_ne!(text, RELEASE_CONFIG, "{from}");
            let err = IrisConfig::parse(&text)
                .unwrap()
                .validate_supported()
                .unwrap_err();
            assert!(
                matches!(&err, Error::Unsupported(m) if m.contains(key)),
                "{key}: {err:?}"
            );
        }
    }

    #[test]
    fn yaml_reader_rejects_what_it_does_not_understand() {
        assert!(IrisConfig::parse("model: {a: 1}\n").is_err());
        assert!(IrisConfig::parse("model:\n  depth: x\n").is_err());
        assert!(IrisConfig::parse("model:\n  depth: 1\n  depth: 2\n").is_err());
        let null_kv = IrisConfig::parse("model:\n  num_kv_heads: null\n").unwrap();
        assert_eq!(null_kv.model.num_kv_heads, None);
    }

    #[test]
    fn window_truncates_the_caption_and_keeps_the_suffix() {
        let w = assemble_window(&[1, 2], &[10, 11, 12, 13, 14], &[8, 9], 5).unwrap();
        assert_eq!(w.input_ids, [1, 2, 10, 11, 12, 8, 9]);
        assert_eq!(w.mask, [1, 1, 1, 1, 1]);
        assert_eq!(w.truncated_tokens, 2);
        assert_eq!(w.window_tokens(), 5);
        let empty = assemble_window(&[1, 2], &[], &[8, 9], 5).unwrap();
        assert_eq!(empty.input_ids, [1, 2, 8, 9]);
        assert_eq!(empty.mask, [1, 1, 0, 0, 0]);
        assert!(assemble_window(&[1], &[3], &[8, 9], 2).is_err());
    }

    #[test]
    fn solver_plan_ramps_order_and_ends_with_the_exact_projection() {
        let plan = dpm_solver_plan(5, 2, 4.0).unwrap();
        let orders: Vec<_> = plan
            .iter()
            .map(|s| matches!(s.update, DpmUpdate::Second { .. }) as u8 + 1)
            .collect();
        assert_eq!(orders, [1, 2, 2, 2, 1]);
        let last = plan.last().unwrap();
        assert_eq!(last.t, 0.0);
        // x ← 0·x + 1·x0 at t = 0
        assert_eq!(last.update, DpmUpdate::First { cx: 0.0, c0: -1.0 });
        let grid = time_grid(100, 4.0);
        assert_eq!(grid.len(), 101);
        assert_eq!(grid[100], 0.0);
        assert!((grid[0] - shift_sigma(0.999, 4.0)).abs() < 1e-15);
        assert!(grid.windows(2).all(|w| w[0] > w[1]));
        assert!(dpm_solver_plan(3, 3, 4.0).is_err());
    }

    #[test]
    fn cfg_gate_is_strictly_inside_the_interval() {
        assert!(cfg_active(3.0, 0.5, DEFAULT_CFG_INTERVAL));
        assert!(!cfg_active(3.0, 1.0, DEFAULT_CFG_INTERVAL));
        assert!(!cfg_active(3.0, 0.0, DEFAULT_CFG_INTERVAL));
        assert!(!cfg_active(1.0, 0.5, DEFAULT_CFG_INTERVAL));
    }

    #[test]
    fn unhonored_controls_are_refused_by_name() {
        let ok = GenerationRequest::default();
        reject_unhonored_generation_controls("iris_3b", &ok).unwrap();
        let req = GenerationRequest {
            scheduler_shift: Some(2.0),
            ..Default::default()
        };
        let err = reject_unhonored_generation_controls("iris_3b", &req).unwrap_err();
        assert!(matches!(err, Error::Unsupported(m) if m.contains("scheduler_shift")));
    }

    #[test]
    fn params_take_the_release_defaults() {
        let p = GenerationParams::resolve(&GenerationRequest::default(), 7);
        assert_eq!(p.steps, 100);
        assert_eq!(p.cfg_scale, 3.0);
        assert_eq!(p.negative_prompt, "");
        assert_eq!(p.seed, 7);
        assert!(p.uses_cfg());
    }

    #[test]
    fn tasks_name_their_resources() {
        assert!(IrisTask::Generation.uses_text_encoder());
        assert!(!IrisTask::Depth.uses_text_encoder());
        assert!(!IrisTask::Restoration.uses_text_encoder());
        assert_eq!(IrisTask::Restoration.backbone_subdir(), "upscaler");
        for sha in [
            UPSTREAM_CODE_REVISION,
            UPSTREAM_WEIGHTS_REVISION,
            TEXT_ENCODER_REVISION,
        ] {
            assert_eq!(sha.len(), 40);
            assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn missing_resources_are_load_errors_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let err = GenerationResources::from_spec(&spec, "iris_3b").unwrap_err();
        assert!(err.to_string().contains("text_encoder"), "{err}");
        let mut spec = spec;
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(dir.path().join("te")),
        );
        let err = GenerationResources::from_spec(&spec, "iris_3b").unwrap_err();
        assert!(err.to_string().contains("config.yaml"), "{err}");
    }
}

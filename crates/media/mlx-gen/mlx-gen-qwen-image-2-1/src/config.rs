//! Frozen upstream configuration: the production component geometry (the `config.json` values of
//! `Qwen/Qwen-Image-2.1` @ [`crate::UPSTREAM_HF_REVISION`]), the size presets and defaults from the
//! upstream README, and the `config.json` readers the loader uses so a miniature parity snapshot and
//! the production snapshot go through one code path.

use std::path::Path;

use mlx_gen::{Error, Result};
use serde_json::Value;

/// Both image dims must be multiples of **32** px: the VAE compresses 16× and the transformer
/// expands each Qwen3-VL vision slot into a 2×2 group of latent tokens, so the latent grid must be
/// even on both axes (`QwenImage21Pipeline`: `multiple_of = vae_scale_factor * 2`).
pub const SIZE_MULTIPLE: u32 = 32;
/// Pixels per latent token per side (`AutoencoderKLQwenImage21.scale_factor_spatial`).
pub const VAE_SCALE_FACTOR: u32 = 16;
/// Upstream default denoising steps (`num_inference_steps=40` in every README example).
pub const DEFAULT_STEPS: u32 = 40;
/// Upstream default `true_cfg_scale`: 2.1 is meant to be sampled **without** guidance; a negative
/// prompt plus a scale above 1 enables real classifier-free guidance (two forwards per step).
pub const DEFAULT_TRUE_CFG: f32 = 1.0;
/// Condition images the unified model accepts (README: "Support up to 10 reference images").
pub const MAX_REFERENCE_IMAGES: usize = 10;
/// The fixed system instruction of the prompt templates (`QwenImage21Pipeline.sys_prompt`).
pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";
/// `QwenImage21Pipeline.__call__`'s `output_resolution` default — the side length every condition
/// image is fitted to before it reaches the vision tower and the VAE.
pub const OUTPUT_RESOLUTION: u32 = 1024;
/// Qwen3-VL processor patch side (`processor/preprocessor_config.json` `patch_size`).
pub const VISION_PATCH_SIZE: usize = 16;
/// Qwen3-VL processor 2×2 patch merge (`merge_size`) — one `<|image_pad|>` slot per merged block,
/// which the DiT expands into a 2×2 group of latent tokens.
pub const VISION_MERGE_SIZE: usize = 2;
/// Latent tokens one `<|image_pad|>` vision slot stands for
/// (`transformer_qwenimage21._IMG_TOKENS_PER_SLOT`).
pub const IMAGE_TOKENS_PER_SLOT: usize = VISION_MERGE_SIZE * VISION_MERGE_SIZE;

/// One upstream aspect-ratio preset (README `aspect_ratios`, `(width, height)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizePreset {
    pub ratio: &'static str,
    pub width: u32,
    pub height: u32,
}

/// The seven upstream presets, `1:1` (the default) first.
pub const PRESETS: [SizePreset; 7] = [
    SizePreset {
        ratio: "1:1",
        width: 2048,
        height: 2048,
    },
    SizePreset {
        ratio: "4:3",
        width: 2400,
        height: 1792,
    },
    SizePreset {
        ratio: "3:4",
        width: 1792,
        height: 2400,
    },
    SizePreset {
        ratio: "3:2",
        width: 2528,
        height: 1696,
    },
    SizePreset {
        ratio: "2:3",
        width: 1696,
        height: 2528,
    },
    SizePreset {
        ratio: "16:9",
        width: 2752,
        height: 1536,
    },
    SizePreset {
        ratio: "9:16",
        width: 1536,
        height: 2752,
    },
];

fn read_json(path: &Path) -> Result<Value> {
    let bytes = mlx_gen::gen_core::__model_input_reader!(
        std::fs::read(path),
        "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/config.rs::read_json::std::fs::read@3295",
        ArtifactInput,
        path,
        |__model_input_path| std::fs::read(__model_input_path)
    )
    .map_err(|e| Error::Msg(format!("qwen_image_2_1: read {}: {e}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::Msg(format!("qwen_image_2_1: parse {}: {e}", path.display())))
}

fn field<'a>(v: &'a Value, key: &str, ctx: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| Error::Msg(format!("qwen_image_2_1: {ctx}: missing `{key}`")))
}

fn u(v: &Value, key: &str, ctx: &str) -> Result<usize> {
    field(v, key, ctx)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| Error::Msg(format!("qwen_image_2_1: {ctx}: `{key}` is not an integer")))
}

fn f_or(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(Value::as_f64)
        .map_or(default, |x| x as f32)
}

fn u_or(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |x| x as usize)
}

fn b_or(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// The even `[t, h, w]` split of `half = head_dim / 2` the sibling Qwen3-VL ports use when a
/// snapshot declares no `mrope_section` — `t` takes the remainder so the three always sum to `half`.
pub(crate) fn even_mrope_section(half: usize) -> [usize; 3] {
    [half - 2 * (half / 3), half / 3, half / 3]
}

/// A **present** `rope_scaling.mrope_section`, validated: exactly three non-negative integers that
/// sum to `half`. Anything else is a typed refusal, never a silent fallback to
/// [`even_mrope_section`] — a wrong section rotates every condition token with the wrong geometry
/// and no later check would notice (sc-24114).
pub(crate) fn parse_mrope_section(section: &Value, half: usize, ctx: &str) -> Result<[usize; 3]> {
    let refuse = |why: &str| {
        Error::Msg(format!(
            "qwen_image_2_1: {ctx}: `mrope_section` {section} is invalid ({why}); it must be three \
             non-negative integers summing to head_dim / 2 = {half}. Fix the snapshot's \
             text_encoder/config.json, or omit the field for the even split."
        ))
    };
    let items = section.as_array().ok_or_else(|| refuse("not an array"))?;
    if items.len() != 3 {
        return Err(refuse("wrong arity"));
    }
    let mut parsed = [0usize; 3];
    for (slot, item) in parsed.iter_mut().zip(items) {
        *slot = item
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| refuse("a non-integer entry"))?;
    }
    if parsed.iter().sum::<usize>() != half {
        return Err(refuse("wrong sum"));
    }
    Ok(parsed)
}

fn f32_list(v: &Value, key: &str, ctx: &str) -> Result<Vec<f32>> {
    field(v, key, ctx)?
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_f64)
                .map(|x| x as f32)
                .collect()
        })
        .ok_or_else(|| Error::Msg(format!("qwen_image_2_1: {ctx}: `{key}` is not a list")))
}

/// `transformer/config.json` (`QwenImage21Transformer2DModel`).
#[derive(Clone, Debug, PartialEq)]
pub struct TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub context_in_dim: usize,
    pub mlp_ratio: usize,
    pub axes_dims_rope: [usize; 3],
    pub eps: f32,
    /// Text and condition-image tokens modulate from `t = 0` (an extra timestep row) instead of the
    /// sampled timestep. `true` for the released checkpoint.
    pub causal_condition: bool,
}

impl TransformerConfig {
    /// The released `Qwen/Qwen-Image-2.1` DiT: 32 single-stream layers, 32 heads × 128, ~7B.
    pub fn production() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            num_layers: 32,
            attention_head_dim: 128,
            num_attention_heads: 32,
            context_in_dim: 4096,
            mlp_ratio: 3,
            axes_dims_rope: [16, 56, 56],
            eps: 1e-6,
            causal_condition: true,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        Self::from_value(&read_json(path)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let ctx = "transformer/config.json";
        let axes = field(v, "axes_dims_rope", ctx)?
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
            .filter(|a| a.len() == 3)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "qwen_image_2_1: {ctx}: `axes_dims_rope` must be three integers"
                ))
            })?;
        let cfg = Self {
            in_channels: u(v, "in_channels", ctx)?,
            out_channels: u_or(v, "out_channels", u(v, "in_channels", ctx)?),
            num_layers: u(v, "num_layers", ctx)?,
            attention_head_dim: u(v, "attention_head_dim", ctx)?,
            num_attention_heads: u(v, "num_attention_heads", ctx)?,
            context_in_dim: u(v, "context_in_dim", ctx)?,
            mlp_ratio: u_or(v, "mlp_ratio", 3),
            axes_dims_rope: [axes[0] as usize, axes[1] as usize, axes[2] as usize],
            eps: f_or(v, "eps", 1e-6),
            causal_condition: b_or(v, "causal_condition", true),
        };
        if u_or(v, "patch_size", 1) != 1 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: only `patch_size: 1` is supported (2.1 consumes latents unpatched)"
            )));
        }
        if cfg.axes_dims_rope.iter().sum::<usize>() != cfg.attention_head_dim {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: axes_dims_rope {:?} must sum to attention_head_dim {}",
                cfg.axes_dims_rope, cfg.attention_head_dim
            )));
        }
        Ok(cfg)
    }
}

/// The language tower of `text_encoder/config.json` (`Qwen3VLForConditionalGeneration` →
/// `text_config`, `model_type: qwen3_vl_text`).
#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Interleaved M-RoPE section `[t, h, w]` (`rope_scaling.mrope_section`, sums to
    /// `head_dim / 2`). With a text-only prompt all three position rows are the token index, so
    /// the interleave collapses to plain 1-D RoPE — which is why the text-to-image path is
    /// unchanged by the reference route (sc-24110).
    pub mrope_section: [usize; 3],
}

impl TextEncoderConfig {
    /// The released Qwen3-VL-8B language tower (36 layers, 4096 hidden, GQA 32/8 × 128).
    pub fn production() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 4096,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 12_288,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        Self::from_value(&read_json(path)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let ctx = "text_encoder/config.json";
        let text = field(v, "text_config", ctx)?;
        let rope = text
            .get("rope_parameters")
            .or_else(|| text.get("rope_scaling"))
            .cloned()
            .unwrap_or(Value::Null);
        let rope_theta = rope
            .get("rope_theta")
            .and_then(Value::as_f64)
            .or_else(|| text.get("rope_theta").and_then(Value::as_f64))
            .ok_or_else(|| Error::Msg(format!("qwen_image_2_1: {ctx}: missing `rope_theta`")))?;
        if text.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: Qwen3-VL text attention is bias-free; `attention_bias: true` is not this architecture"
            )));
        }
        let hidden_size = u(text, "hidden_size", ctx)?;
        let num_attention_heads = u(text, "num_attention_heads", ctx)?;
        let head_dim = u_or(text, "head_dim", hidden_size / num_attention_heads.max(1));
        // `mrope_section` is `[t, h, w]` and sums to `head_dim / 2`. A snapshot that OMITS it
        // falls back to the even split the sibling Qwen3-VL ports use (which is also what a
        // text-only prompt collapses to); a snapshot that CARRIES one that is malformed — wrong
        // arity, a non-integer entry, or a sum that is not `head_dim / 2` — is refused rather than
        // silently re-split: the image-conditioned path would otherwise rotate every condition
        // token with the wrong geometry and nothing downstream would notice (sc-24114).
        let half = head_dim / 2;
        let mrope_section = match rope.get("mrope_section") {
            None | Some(Value::Null) => even_mrope_section(half),
            Some(section) => parse_mrope_section(section, half, ctx)?,
        };
        Ok(Self {
            vocab_size: u(text, "vocab_size", ctx)?,
            hidden_size,
            num_hidden_layers: u(text, "num_hidden_layers", ctx)?,
            num_attention_heads,
            num_key_value_heads: u_or(text, "num_key_value_heads", num_attention_heads),
            head_dim,
            intermediate_size: u(text, "intermediate_size", ctx)?,
            rms_norm_eps: f_or(text, "rms_norm_eps", 1e-6),
            rope_theta: rope_theta as f32,
            mrope_section,
        })
    }
}

/// The Qwen3-VL **vision** tower geometry plus the host-side processor settings the reference
/// route needs (`text_encoder/config.json` → `vision_config`, and `processor/`'s image-processor
/// block). Parsed here so the tiny parity snapshot — which deliberately ships a miniature tower
/// and a miniature pixel budget — drives exactly the production code path.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    /// The tower itself, in `mlx-llm`'s shared Qwen3-VL form.
    pub tower: mlx_llm::models::Qwen3VLVisionConfig,
    /// Host preprocessing (patch/merge/temporal geometry, pixel budget, mean/std).
    pub processor: mlx_llm::image::Qwen35ImageProcessor,
}

impl VisionConfig {
    pub fn from_value(
        text_encoder_config: &Value,
        processor_config: Option<&Value>,
    ) -> Result<Self> {
        let tower = mlx_llm::models::Qwen3VLVisionConfig::from_json(text_encoder_config)
            .map_err(|e| Error::Msg(format!("qwen_image_2_1: text_encoder/config.json: {e}")))?;
        let mut processor = mlx_llm::image::Qwen35ImageProcessor {
            patch_size: tower.patch_size as usize,
            temporal_patch_size: tower.temporal_patch_size as usize,
            merge_size: tower.spatial_merge_size as usize,
            ..Default::default()
        };
        // `processor/preprocessor_config.json`, or the `image_processor` block of
        // `processor/processor_config.json`. Absent, the released geometry stands.
        if let Some(block) = processor_config.and_then(|v| v.get("image_processor").or(Some(v))) {
            // `size.shortest_edge` / `size.longest_edge` is how both the released and the
            // miniature processor serialise the budget; the bare `min_pixels` / `max_pixels`
            // spelling is the older `Qwen2VLImageProcessor` form and wins when present.
            let size = block.get("size");
            for (bare, edge, out) in [
                ("min_pixels", "shortest_edge", &mut processor.min_pixels),
                ("max_pixels", "longest_edge", &mut processor.max_pixels),
            ] {
                if let Some(x) = block
                    .get(bare)
                    .and_then(Value::as_u64)
                    .or_else(|| size.and_then(|s| s.get(edge)).and_then(Value::as_u64))
                {
                    *out = x as usize;
                }
            }
            for (key, out) in [
                ("image_mean", &mut processor.mean),
                ("image_std", &mut processor.std),
            ] {
                if let Some(values) = block.get(key).and_then(Value::as_array) {
                    for (slot, value) in out.iter_mut().zip(values) {
                        if let Some(x) = value.as_f64() {
                            *slot = x as f32;
                        }
                    }
                }
            }
        }
        if processor.min_pixels == 0 || processor.max_pixels < processor.min_pixels {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: the image processor's pixel budget is empty \
                 ([{}, {}])",
                processor.min_pixels, processor.max_pixels
            )));
        }
        Ok(Self { tower, processor })
    }

    /// `output_resolution` for this snapshot: upstream's 1024-px default, clamped into the side
    /// lengths the snapshot's own processor accepts without `smart_resize` rebinding the grid, and
    /// snapped down onto the 32-px grid.
    ///
    /// Upstream takes `output_resolution` as a pipeline argument; gen-core's request carries no
    /// field for it, so the default is derived rather than passed. On the released snapshot the
    /// budget is `[256², 4096²]` and this is exactly 1024. The miniature parity snapshot's budget
    /// is three orders of magnitude smaller, and deriving the value is what lets it exercise this
    /// path at all.
    pub fn output_resolution(&self) -> u32 {
        let multiple = SIZE_MULTIPLE as f64;
        let floor = (self.processor.min_pixels as f64).sqrt();
        let ceiling = (self.processor.max_pixels as f64).sqrt();
        let target = f64::from(OUTPUT_RESOLUTION).min(ceiling).max(floor);
        let snapped = (target / multiple).floor() * multiple;
        (snapped.max(multiple) as u32).max(SIZE_MULTIPLE)
    }
}

/// `vae/config.json` (`AutoencoderKLQwenImage21`).
#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Per encoder stage: whether the (video) downsample is temporal. Single-frame images never
    /// run the temporal convolutions, but the flag still selects the parameter-free shortcut shape.
    pub temperal_downsample: Vec<bool>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub scale_factor_spatial: usize,
}

impl VaeConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        Self::from_value(&read_json(path)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let ctx = "vae/config.json";
        let base_dim = u(v, "base_dim", ctx)?;
        let dim_mult: Vec<usize> = field(v, "dim_mult", ctx)?
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|x| x as usize)
                    .collect()
            })
            .ok_or_else(|| {
                Error::Msg(format!("qwen_image_2_1: {ctx}: `dim_mult` is not a list"))
            })?;
        let temperal_downsample: Vec<bool> = field(v, "temperal_downsample", ctx)?
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_bool).collect())
            .ok_or_else(|| {
                Error::Msg(format!(
                    "qwen_image_2_1: {ctx}: `temperal_downsample` is not a list"
                ))
            })?;
        let cfg = Self {
            base_dim,
            decoder_base_dim: u_or(v, "decoder_base_dim", base_dim),
            z_dim: u(v, "z_dim", ctx)?,
            dim_mult,
            num_res_blocks: u(v, "num_res_blocks", ctx)?,
            temperal_downsample,
            in_channels: u_or(v, "in_channels", 4),
            out_channels: u_or(v, "out_channels", 4),
            latents_mean: f32_list(v, "latents_mean", ctx)?,
            latents_std: f32_list(v, "latents_std", ctx)?,
            scale_factor_spatial: u_or(v, "scale_factor_spatial", 16),
        };
        if !b_or(v, "is_residual", true) {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: only the residual (`is_residual: true`) autoencoder is ported"
            )));
        }
        if v.get("patch_size").is_some_and(|p| !p.is_null()) {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: `patch_size` must be null (2.1's VAE is unpatched)"
            )));
        }
        if cfg.latents_mean.len() != cfg.z_dim || cfg.latents_std.len() != cfg.z_dim {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: latents_mean/std must have z_dim = {} entries",
                cfg.z_dim
            )));
        }
        if cfg.temperal_downsample.len() + 1 != cfg.dim_mult.len() {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: temperal_downsample needs dim_mult.len() - 1 entries"
            )));
        }
        let spatial = 1usize << (cfg.dim_mult.len() - 1);
        if spatial != cfg.scale_factor_spatial {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: {} stages give {spatial}x spatial compression, config says {}",
                cfg.dim_mult.len(),
                cfg.scale_factor_spatial
            )));
        }
        Ok(cfg)
    }
}

/// `scheduler/scheduler_config.json` — the resolution-dependent exponential time shift with the
/// terminal-sigma stretch (`FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting`).
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerConfig {
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub base_shift: f32,
    pub max_shift: f32,
    /// `None` when the config omits/zeroes `shift_terminal` (no stretch).
    pub shift_terminal: Option<f32>,
    pub num_train_timesteps: usize,
}

impl SchedulerConfig {
    /// The released scheduler config (`base 256 / max 8192`, `0.5 → 0.9`, terminal `0.02`).
    pub fn production() -> Self {
        Self {
            base_image_seq_len: 256,
            max_image_seq_len: 8192,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: Some(0.02),
            num_train_timesteps: 1000,
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        Self::from_value(&read_json(path)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let ctx = "scheduler/scheduler_config.json";
        if !b_or(v, "use_dynamic_shifting", false) {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {ctx}: only `use_dynamic_shifting: true` (resolution-shifted) schedules are ported"
            )));
        }
        if let Some(why) = unported_scheduler_flag(v) {
            return Err(Error::Msg(format!("qwen_image_2_1: {ctx}: {why}")));
        }
        let terminal = f_or(v, "shift_terminal", 0.0);
        Ok(Self {
            // The pipeline's `calculate_shift` defaults (256 / 4096 / 0.5 / 1.15) apply when a
            // key is absent from the config.
            base_image_seq_len: u_or(v, "base_image_seq_len", 256),
            max_image_seq_len: u_or(v, "max_image_seq_len", 4096),
            base_shift: f_or(v, "base_shift", 0.5),
            max_shift: f_or(v, "max_shift", 1.15),
            shift_terminal: (terminal != 0.0).then_some(terminal),
            num_train_timesteps: u_or(v, "num_train_timesteps", 1000),
        })
    }
}

/// The `FlowMatchEulerDiscreteScheduler` switch a scheduler config sets that neither released
/// checkpoint does, as the reason it is refused — `None` when it sets none. Both readers refuse the
/// same switches; each raises the refusal under **its own** engine id and in its own error class:
/// the resolution-shifted reader ([`SchedulerConfig`]) as it always has, the saved-sigma reader
/// ([`SavedSigmaSchedule`]) as the typed `Unsupported` every other "this snapshot is not
/// Turbo-shaped" refusal is.
fn unported_scheduler_flag(v: &Value) -> Option<String> {
    if v.get("time_shift_type")
        .and_then(Value::as_str)
        .is_some_and(|t| t != "exponential")
    {
        return Some("only the exponential time shift is ported".to_owned());
    }
    [
        "use_karras_sigmas",
        "use_exponential_sigmas",
        "use_beta_sigmas",
        "invert_sigmas",
        "stochastic_sampling",
    ]
    .into_iter()
    .find(|flag| b_or(v, flag, false))
    .map(|flag| format!("`{flag}: true` is not the released schedule"))
}

/// `model_index.json` key under which a distilled checkpoint saves its sampling sigmas
/// (`QwenImage21Pipeline.config.sample_sigmas`, diffusers PR 14950).
pub const SAMPLE_SIGMAS_KEY: &str = "sample_sigmas";

/// The saved sampling sigmas of a `model_index.json`, terminal `0` excluded, exactly as parsed
/// (`f64`). `Ok(None)` when the key is absent or `null` — the base checkpoint, which has no saved
/// schedule. A key that is **present** is validated rather than tolerated: a non-empty list of
/// finite numbers in `(0, 1]`, strictly descending. Anything else is a typed refusal, because a
/// silently dropped entry would shorten the distilled schedule and still render.
pub fn sample_sigmas(model_index: &Value) -> Result<Option<Vec<f64>>> {
    let ctx = "model_index.json";
    let refuse = |why: &str| {
        Error::Msg(format!(
            "qwen_image_2_1_turbo: {ctx}: `{SAMPLE_SIGMAS_KEY}` is invalid ({why}); it must be a \
             non-empty, strictly descending list of sigmas in (0, 1] without the terminal 0"
        ))
    };
    let list = match model_index.get(SAMPLE_SIGMAS_KEY) {
        None | Some(Value::Null) => return Ok(None),
        Some(list) => list.as_array().ok_or_else(|| refuse("not a list"))?,
    };
    if list.is_empty() {
        return Err(refuse("empty"));
    }
    let mut sigmas = Vec::with_capacity(list.len());
    for item in list {
        let sigma = item.as_f64().ok_or_else(|| refuse("a non-numeric entry"))?;
        if !(sigma > 0.0 && sigma <= 1.0) {
            return Err(refuse("an entry outside (0, 1]"));
        }
        if sigmas.last().is_some_and(|previous| sigma >= *previous) {
            return Err(refuse("not strictly descending"));
        }
        sigmas.push(sigma);
    }
    Ok(Some(sigmas))
}

/// The schedule of a **distilled** snapshot (`Qwen/Qwen-Image-2.1-Turbo`): the sampling sigmas its
/// `model_index.json` saves, used unchanged because its `scheduler/scheduler_config.json` turns
/// every scheduler transform off — `use_dynamic_shifting: false` (no resolution-dependent `mu`
/// shift), `shift: 1.0` (the static shift `shift·σ / (1 + (shift − 1)·σ)` is the identity) and
/// `shift_terminal: null` (no terminal stretch).
///
/// A separate type from [`SchedulerConfig`] on purpose: the two checkpoints share no schedule
/// arithmetic, so a saved-sigma snapshot cannot reach the `mu`-shift formula (or the reverse) by
/// construction. [`crate::scheduler::resample_saved_sigmas`] turns it into the per-request list.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedSigmaSchedule {
    /// `model_index.json` → `sample_sigmas`: descending, terminal `0` excluded, as parsed.
    pub sample_sigmas: Vec<f64>,
    pub num_train_timesteps: usize,
}

impl SavedSigmaSchedule {
    /// The checkpoint's own step count — one denoising step per saved sigma.
    pub fn default_steps(&self) -> u32 {
        self.sample_sigmas.len() as u32
    }

    /// From a snapshot's `model_index.json` and `scheduler/scheduler_config.json`. Refuses, naming
    /// the mismatch, a snapshot that saves no sigmas (a base checkpoint) and a scheduler config
    /// that would transform them (dynamic shifting, a non-unit static shift, a terminal stretch).
    pub fn from_values(model_index: &Value, scheduler_config: &Value) -> Result<Self> {
        let ctx = "scheduler/scheduler_config.json";
        let sample_sigmas = sample_sigmas(model_index)?.ok_or_else(|| {
            Error::Unsupported(format!(
                "qwen_image_2_1_turbo: model_index.json carries no `{SAMPLE_SIGMAS_KEY}`, so this \
                 is not a distilled Qwen-Image-2.1-Turbo snapshot; load a base Qwen-Image 2.1 \
                 snapshot as `qwen_image_2_1`"
            ))
        })?;
        let v = scheduler_config;
        if b_or(v, "use_dynamic_shifting", false) {
            return Err(Error::Unsupported(format!(
                "qwen_image_2_1_turbo: {ctx}: `use_dynamic_shifting: true` is the base Qwen-Image \
                 2.1 resolution-shifted schedule, but model_index.json saves `{SAMPLE_SIGMAS_KEY}`; \
                 the saved sigmas are only ported with the static shift (`use_dynamic_shifting: \
                 false`)"
            )));
        }
        if let Some(why) = unported_scheduler_flag(v) {
            return Err(Error::Unsupported(format!(
                "qwen_image_2_1_turbo: {ctx}: {why}"
            )));
        }
        let unit_or_absent = |key: &str, unit: f64| match v.get(key) {
            None | Some(Value::Null) => true,
            Some(value) => value.as_f64() == Some(unit),
        };
        if !unit_or_absent("shift", 1.0) {
            return Err(Error::Unsupported(format!(
                "qwen_image_2_1_turbo: {ctx}: only `shift: 1.0` is ported for a saved-sigma \
                 schedule (got {}); any other static shift would move the saved sigmas",
                v["shift"]
            )));
        }
        if !unit_or_absent("shift_terminal", 0.0) {
            return Err(Error::Unsupported(format!(
                "qwen_image_2_1_turbo: {ctx}: `shift_terminal` must be null for a saved-sigma \
                 schedule (got {}); a terminal stretch would move the saved sigmas",
                v["shift_terminal"]
            )));
        }
        Ok(Self {
            sample_sigmas,
            num_train_timesteps: u_or(v, "num_train_timesteps", 1000),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_32_multiples_and_1x1_is_default() {
        assert_eq!(PRESETS[0].ratio, "1:1");
        assert_eq!((PRESETS[0].width, PRESETS[0].height), (2048, 2048));
        for p in PRESETS {
            assert_eq!(p.width % SIZE_MULTIPLE, 0, "{}", p.ratio);
            assert_eq!(p.height % SIZE_MULTIPLE, 0, "{}", p.ratio);
        }
        assert_eq!(PRESETS.len(), 7);
    }

    #[test]
    fn production_configs_match_the_pinned_snapshot() {
        let t = TransformerConfig::from_value(&serde_json::json!({
            "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56], "context_in_dim": 4096,
            "in_channels": 64, "num_attention_heads": 32, "num_layers": 32, "out_channels": 64,
            "patch_size": 1, "mlp_ratio": 3, "eps": 1e-06, "causal_condition": true
        }))
        .unwrap();
        assert_eq!(t, TransformerConfig::production());
        assert_eq!(t.inner_dim(), 4096);

        let s = SchedulerConfig::from_value(&serde_json::json!({
            "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": false,
            "max_image_seq_len": 8192, "max_shift": 0.9, "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": 0.02, "stochastic_sampling": false,
            "time_shift_type": "exponential", "use_beta_sigmas": false,
            "use_dynamic_shifting": true, "use_exponential_sigmas": false,
            "use_karras_sigmas": false
        }))
        .unwrap();
        assert_eq!(s, SchedulerConfig::production());

        let te = TextEncoderConfig::from_value(&serde_json::json!({
            "model_type": "qwen3_vl",
            "text_config": {
                "attention_bias": false, "head_dim": 128, "hidden_size": 4096,
                "intermediate_size": 12288, "num_attention_heads": 32, "num_hidden_layers": 36,
                "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
                "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
                "rope_theta": 5000000, "vocab_size": 151936
            }
        }))
        .unwrap();
        assert_eq!(te, TextEncoderConfig::production());
    }

    #[test]
    fn config_readers_reject_unported_variants() {
        let err = TransformerConfig::from_value(&serde_json::json!({
            "attention_head_dim": 16, "axes_dims_rope": [4, 6, 6], "context_in_dim": 8,
            "in_channels": 4, "num_attention_heads": 2, "num_layers": 1, "patch_size": 2
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("patch_size"), "{err}");

        let err = SchedulerConfig::from_value(&serde_json::json!({"use_dynamic_shifting": false}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("use_dynamic_shifting"), "{err}");

        let err = VaeConfig::from_value(&serde_json::json!({
            "base_dim": 4, "z_dim": 8, "dim_mult": [1, 1], "num_res_blocks": 1,
            "temperal_downsample": [false],
            "latents_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "latents_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "scale_factor_spatial": 16
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("spatial compression"), "{err}");
    }

    /// `Qwen/Qwen-Image-2.1-Turbo` @ `UPSTREAM_TURBO_HF_REVISION`: `model_index.json`'s saved
    /// sigmas and `scheduler/scheduler_config.json`, verbatim.
    fn turbo_model_index() -> Value {
        serde_json::json!({
            "_class_name": "QwenImage21Pipeline",
            "sample_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        })
    }

    fn turbo_scheduler_config() -> Value {
        serde_json::json!({
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": false,
            "max_image_seq_len": 8192, "max_shift": 0.9, "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": null, "stochastic_sampling": false,
            "time_shift_type": "exponential", "use_beta_sigmas": false,
            "use_dynamic_shifting": false, "use_exponential_sigmas": false,
            "use_karras_sigmas": false
        })
    }

    /// The released Turbo pair parses to its eight saved sigmas — and the base reader still
    /// refuses that same scheduler config, so the base checkpoint's contract is unchanged.
    ///
    /// *Mutation that reds this:* dropping an entry while parsing (`filter_map`), or reading the
    /// list from `scheduler_config.json` instead of `model_index.json`.
    #[test]
    fn the_released_turbo_snapshot_parses_to_its_eight_saved_sigmas() {
        let saved =
            SavedSigmaSchedule::from_values(&turbo_model_index(), &turbo_scheduler_config())
                .unwrap();
        assert_eq!(
            saved.sample_sigmas,
            [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        );
        assert_eq!(saved.default_steps(), 8);
        assert_eq!(saved.num_train_timesteps, 1000);
        // A scheduler config with no keys is `FlowMatchEulerDiscreteScheduler`'s defaults: static,
        // unit shift, no terminal.
        assert_eq!(
            SavedSigmaSchedule::from_values(&turbo_model_index(), &serde_json::json!({})).unwrap(),
            saved
        );

        let err = SchedulerConfig::from_value(&turbo_scheduler_config())
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "qwen_image_2_1: scheduler/scheduler_config.json: only `use_dynamic_shifting: true` \
             (resolution-shifted) schedules are ported"
        );
    }

    /// A saved-sigma schedule refuses, by name, every snapshot shape that would not use the saved
    /// sigmas unchanged.
    ///
    /// *Mutations that red this:* accepting a missing `sample_sigmas` (the base snapshot under the
    /// Turbo id), or dropping any one of the dynamic-shift / `shift` / `shift_terminal` guards.
    #[test]
    fn a_saved_sigma_schedule_refuses_snapshots_that_would_transform_or_lack_the_sigmas() {
        let with = |key: &str, value: Value| {
            let mut config = turbo_scheduler_config();
            config[key] = value;
            SavedSigmaSchedule::from_values(&turbo_model_index(), &config)
        };
        // The base snapshot: no saved sigmas at all (absent, or an explicit null).
        for index in [
            serde_json::json!({"_class_name": "QwenImage21Pipeline"}),
            serde_json::json!({"sample_sigmas": null}),
            Value::Null,
        ] {
            match SavedSigmaSchedule::from_values(&index, &turbo_scheduler_config()) {
                Err(Error::Unsupported(err)) => {
                    assert!(err.contains("carries no `sample_sigmas`"), "{err}");
                    assert!(err.contains("`qwen_image_2_1`"), "{err}");
                }
                other => panic!("a snapshot without saved sigmas must be refused: {other:?}"),
            }
        }
        for (key, value, needle) in [
            (
                "use_dynamic_shifting",
                serde_json::json!(true),
                "`use_dynamic_shifting: true`",
            ),
            ("shift", serde_json::json!(3.0), "only `shift: 1.0`"),
            ("shift", serde_json::json!("1.0"), "only `shift: 1.0`"),
            (
                "shift_terminal",
                serde_json::json!(0.02),
                "`shift_terminal` must be null",
            ),
        ] {
            match with(key, value.clone()) {
                Err(Error::Unsupported(err)) => assert!(err.contains(needle), "{key}: {err}"),
                other => panic!("{key} = {value} must be a typed refusal: {other:?}"),
            }
        }
        let err = with("use_karras_sigmas", serde_json::json!(true))
            .unwrap_err()
            .to_string();
        assert!(err.contains("`use_karras_sigmas: true`"), "{err}");
        // The two spellings of "no terminal stretch" and of the unit shift are accepted.
        assert!(with("shift_terminal", serde_json::json!(0.0)).is_ok());
        assert!(with("shift", serde_json::json!(1)).is_ok());
    }

    /// Every way a snapshot can fail to be a Turbo snapshot is the **typed** refusal, raised under
    /// the Turbo id — the scheduler switches neither checkpoint sets included (sc-25902; the
    /// candle twin's test of the same name). The base reader keeps its own label and error class
    /// for those same switches.
    ///
    /// *Mutations that red this:* raising the unported-switch refusal as `Error::Msg`, or under
    /// the base `qwen_image_2_1:` label, from the saved-sigma reader.
    #[test]
    fn a_snapshot_that_is_not_turbo_shaped_is_refused_by_name() {
        let refusal = |index: Value, scheduler: Value| match SavedSigmaSchedule::from_values(
            &index, &scheduler,
        ) {
            Err(Error::Unsupported(message)) => message,
            other => panic!("expected a typed Unsupported refusal, got {other:?}"),
        };
        for index in [
            Value::Null,
            serde_json::json!({"_class_name": "QwenImage21Pipeline"}),
            serde_json::json!({"sample_sigmas": null}),
        ] {
            let text = refusal(index, turbo_scheduler_config());
            assert!(text.starts_with("qwen_image_2_1_turbo: "), "{text}");
            assert!(text.contains("no `sample_sigmas`"), "{text}");
            assert!(text.contains("`qwen_image_2_1`"), "{text}");
        }
        for (key, value, needle) in [
            (
                "use_dynamic_shifting",
                serde_json::json!(true),
                "use_dynamic_shifting",
            ),
            ("shift", serde_json::json!(3.0), "`shift: 1.0`"),
            ("shift", serde_json::json!("3.0"), "`shift: 1.0`"),
            ("shift_terminal", serde_json::json!(0.02), "shift_terminal"),
            ("shift_terminal", serde_json::json!(false), "shift_terminal"),
            (
                "use_karras_sigmas",
                serde_json::json!(true),
                "`use_karras_sigmas: true` is not the released schedule",
            ),
            (
                "use_exponential_sigmas",
                serde_json::json!(true),
                "`use_exponential_sigmas: true`",
            ),
            (
                "use_beta_sigmas",
                serde_json::json!(true),
                "`use_beta_sigmas: true`",
            ),
            (
                "invert_sigmas",
                serde_json::json!(true),
                "`invert_sigmas: true`",
            ),
            (
                "stochastic_sampling",
                serde_json::json!(true),
                "`stochastic_sampling: true`",
            ),
            (
                "time_shift_type",
                serde_json::json!("linear"),
                "only the exponential time shift is ported",
            ),
        ] {
            let mut scheduler = turbo_scheduler_config();
            scheduler[key] = value.clone();
            let text = refusal(turbo_model_index(), scheduler.clone());
            assert!(
                text.starts_with("qwen_image_2_1_turbo: scheduler/scheduler_config.json: "),
                "{key} = {value}: {text}"
            );
            assert!(text.contains(needle), "{key} = {value}: {text}");
        }

        // The base reader's refusal of the same switches is unchanged: its own id, `Error::Msg`.
        for (key, value, want) in [
            (
                "use_karras_sigmas",
                serde_json::json!(true),
                "qwen_image_2_1: scheduler/scheduler_config.json: `use_karras_sigmas: true` is \
                 not the released schedule",
            ),
            (
                "time_shift_type",
                serde_json::json!("linear"),
                "qwen_image_2_1: scheduler/scheduler_config.json: only the exponential time \
                 shift is ported",
            ),
        ] {
            let mut scheduler = turbo_scheduler_config();
            scheduler["use_dynamic_shifting"] = serde_json::json!(true);
            scheduler[key] = value;
            match SchedulerConfig::from_value(&scheduler) {
                Err(Error::Msg(message)) => assert_eq!(message, want),
                other => panic!("{key}: the base reader's refusal changed: {other:?}"),
            }
        }
    }

    /// A present `sample_sigmas` is validated, never silently repaired.
    ///
    /// *Mutation that reds this:* collecting with `filter_map(Value::as_f64)` — the non-numeric
    /// case would then load as a seven-step schedule.
    #[test]
    fn present_sample_sigmas_are_validated() {
        assert_eq!(sample_sigmas(&serde_json::json!({})).unwrap(), None);
        assert_eq!(
            sample_sigmas(&serde_json::json!({"sample_sigmas": [1.0, 0.5]})).unwrap(),
            Some(vec![1.0, 0.5])
        );
        for (list, why) in [
            (serde_json::json!([]), "empty"),
            (serde_json::json!("1.0, 0.5"), "not a list"),
            (serde_json::json!([1.0, "0.9", 0.5]), "a non-numeric entry"),
            (
                serde_json::json!([1.0, 0.5, 0.0]),
                "an entry outside (0, 1]",
            ),
            (serde_json::json!([1.5, 0.5]), "an entry outside (0, 1]"),
            (
                serde_json::json!([1.0, 0.5, 0.5]),
                "not strictly descending",
            ),
            (serde_json::json!([0.5, 1.0]), "not strictly descending"),
        ] {
            let err = sample_sigmas(&serde_json::json!({"sample_sigmas": list}))
                .unwrap_err()
                .to_string();
            assert!(err.contains("`sample_sigmas`"), "{list}: {err}");
            assert!(err.contains(why), "{list}: expected `{why}` in {err}");
        }
    }

    /// One `text_config` with `rope_scaling` replaced by `rope`.
    fn text_encoder_with_rope(rope: serde_json::Value) -> Result<TextEncoderConfig> {
        TextEncoderConfig::from_value(&serde_json::json!({
            "model_type": "qwen3_vl",
            "text_config": {
                "attention_bias": false, "head_dim": 128, "hidden_size": 4096,
                "intermediate_size": 12288, "num_attention_heads": 32, "num_hidden_layers": 36,
                "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
                "rope_scaling": rope,
                "rope_theta": 5000000, "vocab_size": 151936
            }
        }))
    }

    /// An **absent** `mrope_section` falls back to the even split; a **present** one is taken as
    /// written when valid and is a typed refusal — never a silent even split — when it is not.
    ///
    /// *Mutation that reds this:* restoring the `.unwrap_or(even)` fallback for a present-but-
    /// invalid section (any of the four malformed cases below would then load as `[44, 42, 42]`).
    #[test]
    fn mrope_section_falls_back_only_when_absent_and_refuses_when_present_and_invalid() {
        // Absent (no `rope_scaling` block at all, a block without the key, or an explicit null):
        // the even split of head_dim / 2 = 64 -> [22, 21, 21].
        for rope in [
            serde_json::Value::Null,
            serde_json::json!({"rope_type": "default"}),
            serde_json::json!({"rope_type": "default", "mrope_section": null}),
        ] {
            let te = text_encoder_with_rope(rope.clone()).unwrap();
            assert_eq!(
                te.mrope_section,
                [22, 21, 21],
                "absent section under {rope}"
            );
            assert_eq!(te.mrope_section, even_mrope_section(64));
        }

        // Present and valid: taken as written, even when it is not the even split.
        let te =
            text_encoder_with_rope(serde_json::json!({"mrope_section": [24, 20, 20]})).unwrap();
        assert_eq!(te.mrope_section, [24, 20, 20]);
        let te =
            text_encoder_with_rope(serde_json::json!({"mrope_section": [32, 16, 16]})).unwrap();
        assert_eq!(te.mrope_section, [32, 16, 16]);

        // Present and invalid: refused, naming the field and the constraint.
        for (section, why) in [
            (serde_json::json!([24, 20]), "wrong arity"),
            (serde_json::json!([24, 20, 20, 0]), "wrong arity"),
            (serde_json::json!([24, 20, 21]), "wrong sum"),
            (serde_json::json!([24, "20", 20]), "a non-integer entry"),
            (serde_json::json!([24, -20, 60]), "a non-integer entry"),
            (serde_json::json!(64), "not an array"),
        ] {
            let err = text_encoder_with_rope(serde_json::json!({"mrope_section": section}))
                .unwrap_err()
                .to_string();
            assert!(err.contains("`mrope_section`"), "{section}: {err}");
            assert!(err.contains(why), "{section}: expected `{why}` in {err}");
            assert!(err.contains("= 64"), "{section}: {err}");
        }
    }
}

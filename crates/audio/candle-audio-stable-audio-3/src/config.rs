//! Typed parsing for the two upstream Stable Audio 3 `model_config.json` shapes.
//!
//! The defaults in this module mirror the constructors at frozen upstream commit
//! `124e8a799f57a1f665495ecb72e547d0a62867f1`, not Rust zero values. This matters most for
//! `dyt`: `TransformerResamplingBlock` defaults it to `true`, so the medium embedded SAME and the
//! standalone SAME configurations are equivalent even when one spelling omits the key.
//!
//! Training dictionaries are intentionally not modeled. Serde's normal unknown-field tolerance
//! accepts them (and future training-only additions) while inference fields remain typed.

use std::collections::BTreeMap;
use std::path::Path;

use candle_audio::{AudioError, Result};
use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}
fn default_one_f64() -> f64 {
    1.0
}
fn default_two() -> usize {
    2
}
fn default_32() -> usize {
    32
}
fn default_128() -> usize {
    128
}
fn default_256() -> usize {
    256
}
fn default_768() -> usize {
    768
}
fn default_4096() -> usize {
    4096
}
fn default_c_mults() -> Vec<usize> {
    vec![1, 2, 4, 8]
}
fn default_strides() -> Vec<usize> {
    vec![2, 4, 8, 8]
}
fn default_depths() -> Vec<usize> {
    vec![3, 3, 3, 3]
}
fn default_sinusoidal_blocks() -> Vec<usize> {
    vec![0, 0, 0, 0]
}
fn default_timestep_features_dim() -> usize {
    256
}
fn default_t5_model() -> String {
    "google/t5gemma-b-b-ul2".into()
}
fn default_t5_max_length() -> usize {
    128
}
fn default_padding_mode() -> PaddingMode {
    PaddingMode::Zero
}
fn default_number_max() -> f64 {
    1.0
}
fn default_base_shift() -> f64 {
    0.5
}
fn default_max_shift() -> f64 {
    1.15
}
fn default_anchor_length() -> usize {
    2000
}
fn default_anchor_logsnr() -> f64 {
    -6.2
}
fn default_logsnr_end() -> f64 {
    2.0
}

/// Complete parsed snapshot configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StableAudioConfig {
    pub model_type: ModelType,
    pub sample_rate: u32,
    pub sample_size: usize,
    #[serde(default = "default_two")]
    pub audio_channels: usize,
    pub model: ModelConfig,
}

impl StableAudioConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        let parsed: Self = serde_json::from_str(&text)
            .map_err(|e| AudioError::Msg(format!("parse {}: {e}", path.display())))?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn validate(&self) -> Result<()> {
        let shape_matches = matches!(
            (&self.model_type, &self.model),
            (ModelType::DiffusionCondInpaint, ModelConfig::Diffusion(_))
                | (ModelType::Autoencoder, ModelConfig::Autoencoder(_))
        );
        if !shape_matches {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 model_type {:?} does not match its model object",
                self.model_type
            )));
        }
        if self.audio_channels != 2 {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 expects stereo audio_channels=2, got {}",
                self.audio_channels
            )));
        }
        match &self.model {
            ModelConfig::Diffusion(model) => model.validate(),
            ModelConfig::Autoencoder(model) => model.validate(),
        }
    }

    pub fn autoencoder(&self) -> &AutoencoderConfig {
        match &self.model {
            ModelConfig::Diffusion(model) => &model.pretransform.config,
            ModelConfig::Autoencoder(model) => model,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    DiffusionCondInpaint,
    Autoencoder,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelConfig {
    Diffusion(Box<DiffusionModelConfig>),
    Autoencoder(Box<AutoencoderConfig>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionModelConfig {
    pub pretransform: AutoencoderPretransform,
    pub conditioning: ConditioningConfig,
    pub diffusion: DiffusionConfig,
    pub io_channels: usize,
}

impl DiffusionModelConfig {
    fn validate(&self) -> Result<()> {
        if self.pretransform.kind != PretransformType::Autoencoder {
            return Err(AudioError::Msg(
                "Stable Audio 3 diffusion pretransform must have type autoencoder".into(),
            ));
        }
        self.pretransform.config.validate()?;
        if self.io_channels != self.diffusion.config.io_channels {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 outer io_channels {} != DiT io_channels {}",
                self.io_channels, self.diffusion.config.io_channels
            )));
        }
        if self.conditioning.cond_dim != self.diffusion.config.cond_token_dim {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 conditioner dim {} != DiT cond_token_dim {}",
                self.conditioning.cond_dim, self.diffusion.config.cond_token_dim
            )));
        }
        self.conditioning.validate()?;
        self.diffusion.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoencoderPretransform {
    #[serde(rename = "type")]
    pub kind: PretransformType,
    #[serde(default)]
    pub iterate_batch: bool,
    #[serde(default)]
    pub chunked: bool,
    #[serde(default)]
    pub enable_grad: bool,
    #[serde(default = "default_one_f64")]
    pub scale: f64,
    pub config: AutoencoderConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PretransformType {
    Autoencoder,
    Patched,
}

/// SAME autoencoder configuration, either standalone or nested under the diffusion pretransform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoencoderConfig {
    pub pretransform: PatchedPretransform,
    pub encoder: EncoderComponent,
    pub decoder: DecoderComponent,
    pub bottleneck: BottleneckComponent,
    pub latent_dim: usize,
    pub downsampling_ratio: usize,
    pub io_channels: usize,
    #[serde(default)]
    pub in_channels: Option<usize>,
    #[serde(default)]
    pub out_channels: Option<usize>,
}

impl AutoencoderConfig {
    fn validate(&self) -> Result<()> {
        if self.pretransform.kind != PretransformType::Patched {
            return Err(AudioError::Msg(
                "Stable Audio 3 inner pretransform must have type patched".into(),
            ));
        }
        let encoder = &self.encoder.config;
        let decoder = &self.decoder.config;
        if encoder.c_mults.len() != encoder.strides.len()
            || encoder.c_mults.len() != encoder.transformer_depths.len()
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 encoder stage arrays have different lengths".into(),
            ));
        }
        if decoder.c_mults.len() != decoder.strides.len()
            || decoder.c_mults.len() != decoder.transformer_depths.len()
            || decoder.c_mults.len() != decoder.sinusoidal_blocks.len()
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 decoder stage arrays have different lengths".into(),
            ));
        }
        if encoder.latent_dim != self.latent_dim || decoder.latent_dim != self.latent_dim {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 latent dimensions disagree: model={}, encoder={}, decoder={}",
                self.latent_dim, encoder.latent_dim, decoder.latent_dim
            )));
        }
        let ratio: usize = encoder.strides.iter().product();
        let patched_ratio = ratio.saturating_mul(self.pretransform.config.patch_size);
        if patched_ratio != self.downsampling_ratio {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 downsampling_ratio {} != patch_size {} × encoder strides {:?}",
                self.downsampling_ratio, self.pretransform.config.patch_size, encoder.strides
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchedPretransform {
    #[serde(rename = "type")]
    pub kind: PretransformType,
    #[serde(default)]
    pub enable_grad: bool,
    pub config: PatchedConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchedConfig {
    pub patch_size: usize,
    pub channels: usize,
    #[serde(default = "default_one")]
    pub oversampling: usize,
    #[serde(default)]
    pub postfilter_channels: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoencoderModuleType {
    TaaeV2,
    Same,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderComponent {
    #[serde(rename = "type")]
    pub kind: AutoencoderModuleType,
    #[serde(default = "default_true")]
    pub requires_grad: bool,
    pub config: EncoderConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderComponent {
    #[serde(rename = "type")]
    pub kind: AutoencoderModuleType,
    #[serde(default = "default_true")]
    pub requires_grad: bool,
    #[serde(default)]
    pub soft_clip: bool,
    pub config: DecoderConfig,
}

/// Fields accepted by upstream `SAMEEncoder`, including kwargs consumed by each resampling block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_two")]
    pub in_channels: usize,
    #[serde(default = "default_128")]
    pub channels: usize,
    #[serde(default = "default_32")]
    pub latent_dim: usize,
    #[serde(default = "default_c_mults")]
    pub c_mults: Vec<usize>,
    #[serde(default = "default_strides")]
    pub strides: Vec<usize>,
    #[serde(default = "default_depths")]
    pub transformer_depths: Vec<usize>,
    #[serde(default)]
    pub sliding_window: Option<Vec<usize>>,
    #[serde(default)]
    pub checkpointing: bool,
    #[serde(default)]
    pub conformer: bool,
    #[serde(default)]
    pub layer_scale: bool,
    #[serde(default)]
    pub causal: bool,
    #[serde(default = "default_true")]
    pub differential: bool,
    #[serde(default)]
    pub variable_stride: bool,
    #[serde(default)]
    pub mask_noise: f64,
    #[serde(default)]
    pub conv_mapping: bool,
    #[serde(default)]
    pub freeze_backbone: bool,
    #[serde(default = "default_128")]
    pub dim_heads: usize,
    #[serde(default = "default_128")]
    pub chunk_size: usize,
    #[serde(default)]
    pub chunk_midpoint_shift: bool,
    #[serde(default = "default_true")]
    pub dyt: bool,
    #[serde(default)]
    pub feat_scale: bool,
    #[serde(default = "default_three")]
    pub ff_mult: f64,
    #[serde(default = "default_true")]
    pub mapping_bias: bool,
    #[serde(default)]
    pub cross_attn: bool,
    #[serde(default)]
    pub use_flash: bool,
    #[serde(default)]
    pub use_snake: bool,
    #[serde(default)]
    pub use_dilated_conv: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default)]
    pub enable_inner_layer_dropout: bool,
    #[serde(default = "default_mapping_style")]
    pub mapping_style: String,
}

fn default_three() -> f64 {
    3.0
}
fn default_mapping_style() -> String {
    "none".into()
}

/// Fields accepted by upstream `SAMEDecoder`, including resampling-block kwargs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderConfig {
    #[serde(default = "default_two")]
    pub out_channels: usize,
    #[serde(default = "default_128")]
    pub channels: usize,
    #[serde(default = "default_32")]
    pub latent_dim: usize,
    #[serde(default = "default_c_mults")]
    pub c_mults: Vec<usize>,
    #[serde(default = "default_strides")]
    pub strides: Vec<usize>,
    #[serde(default = "default_depths")]
    pub transformer_depths: Vec<usize>,
    #[serde(default = "default_sinusoidal_blocks")]
    pub sinusoidal_blocks: Vec<usize>,
    #[serde(default)]
    pub sliding_window: Option<Vec<usize>>,
    #[serde(default)]
    pub checkpointing: bool,
    #[serde(default)]
    pub conformer: bool,
    #[serde(default)]
    pub layer_scale: bool,
    #[serde(default)]
    pub causal: bool,
    #[serde(default = "default_true")]
    pub differential: bool,
    #[serde(default)]
    pub variable_stride: bool,
    #[serde(default)]
    pub mask_noise: f64,
    #[serde(default)]
    pub conv_mapping: bool,
    #[serde(default)]
    pub freeze_backbone: bool,
    #[serde(default = "default_128")]
    pub dim_heads: usize,
    #[serde(default = "default_128")]
    pub chunk_size: usize,
    #[serde(default)]
    pub chunk_midpoint_shift: bool,
    #[serde(default = "default_true")]
    pub dyt: bool,
    #[serde(default)]
    pub feat_scale: bool,
    #[serde(default = "default_three")]
    pub ff_mult: f64,
    #[serde(default = "default_true")]
    pub mapping_bias: bool,
    #[serde(default)]
    pub cross_attn: bool,
    #[serde(default)]
    pub use_flash: bool,
    #[serde(default)]
    pub use_snake: bool,
    #[serde(default)]
    pub use_dilated_conv: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default)]
    pub enable_inner_layer_dropout: bool,
    #[serde(default = "default_mapping_style")]
    pub mapping_style: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BottleneckComponent {
    #[serde(rename = "type")]
    pub kind: BottleneckType,
    #[serde(default = "default_true")]
    pub requires_grad: bool,
    pub config: BottleneckConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BottleneckType {
    Softnorm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BottleneckConfig {
    #[serde(default = "default_32")]
    pub dim: usize,
    #[serde(default)]
    pub noise_augment_dim: usize,
    #[serde(default)]
    pub noise_regularize: bool,
    #[serde(default)]
    pub auto_scale: bool,
    #[serde(default)]
    pub freeze: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditioningConfig {
    pub configs: Vec<ConditionerConfig>,
    pub cond_dim: usize,
    #[serde(default)]
    pub default_keys: BTreeMap<String, String>,
    #[serde(default)]
    pub pre_encoded_keys: Vec<String>,
}

impl ConditioningConfig {
    fn validate(&self) -> Result<()> {
        if self.cond_dim != 768 {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 shipped conditioning expects cond_dim=768, got {}",
                self.cond_dim
            )));
        }
        if !self.pre_encoded_keys.is_empty() || !self.default_keys.is_empty() {
            return Err(AudioError::Msg(
                "Stable Audio 3 pre-encoded/default conditioner branches are unsupported".into(),
            ));
        }
        if self.configs.len() != 2 {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 expects exactly prompt and seconds_total conditioners, got {}",
                self.configs.len()
            )));
        }
        let mut prompt = false;
        let mut seconds = false;
        for conditioner in &self.configs {
            match conditioner {
                ConditionerConfig::T5gemma { id, config } if id == "prompt" => {
                    prompt = true;
                    if config.max_length != 256
                        || config.padding_mode != PaddingMode::Learned
                        || config.enable_grad
                        || config.project_out
                    {
                        return Err(AudioError::Msg(
                            "Stable Audio 3 prompt conditioner must be the shipped 256-token \
                             learned-padding, unprojected T5Gemma encoder"
                                .into(),
                        ));
                    }
                }
                ConditionerConfig::Number { id, config } if id == "seconds_total" => {
                    seconds = true;
                    if config.min_val != 0.0
                        || config.max_val != 384.0
                        || config.fourier_features_type != FourierFeaturesType::Expo
                    {
                        return Err(AudioError::Msg(
                            "Stable Audio 3 seconds_total must use the shipped [0,384] Expo \
                             NumberConditioner"
                                .into(),
                        ));
                    }
                }
                _ => {
                    return Err(AudioError::Msg(
                        "Stable Audio 3 supports only prompt T5Gemma and seconds_total number \
                         conditioning"
                            .into(),
                    ));
                }
            }
        }
        if !prompt || !seconds {
            return Err(AudioError::Msg(
                "Stable Audio 3 requires prompt and seconds_total conditioners".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerConfig {
    T5gemma { id: String, config: T5GemmaConfig },
    Number { id: String, config: NumberConfig },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct T5GemmaConfig {
    #[serde(default = "default_t5_model")]
    pub model_name: String,
    #[serde(default = "default_t5_max_length")]
    pub max_length: usize,
    #[serde(default)]
    pub enable_grad: bool,
    #[serde(default)]
    pub project_out: bool,
    #[serde(default = "default_padding_mode")]
    pub padding_mode: PaddingMode,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub subfolder: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaddingMode {
    None,
    Zero,
    Learned,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumberConfig {
    #[serde(default)]
    pub min_val: f64,
    #[serde(default = "default_number_max")]
    pub max_val: f64,
    #[serde(default)]
    pub fourier_features_type: FourierFeaturesType,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FourierFeaturesType {
    #[default]
    Learned,
    Expo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionConfig {
    #[serde(default)]
    pub cross_attention_cond_ids: Vec<String>,
    #[serde(default)]
    pub global_cond_ids: Vec<String>,
    #[serde(default)]
    pub input_concat_ids: Vec<String>,
    #[serde(default)]
    pub local_add_cond_ids: Vec<String>,
    #[serde(default)]
    pub modular_local_cond_configs: Vec<ModularLocalConditioningConfig>,
    #[serde(default)]
    pub prepend_cond_ids: Vec<String>,
    #[serde(rename = "type")]
    pub kind: DiffusionModuleType,
    #[serde(default)]
    pub diffusion_objective: DiffusionObjective,
    #[serde(default)]
    pub distribution_shift_options: Option<DistributionShiftConfig>,
    #[serde(default)]
    pub sampling_distribution_shift_options: Option<DistributionShiftConfig>,
    #[serde(default)]
    pub mask_padding_attention: bool,
    #[serde(default)]
    pub use_effective_length_for_schedule: bool,
    pub config: DitConfig,
}

impl DiffusionConfig {
    fn validate(&self) -> Result<()> {
        if self.config.num_heads == 0
            || !self.config.embed_dim.is_multiple_of(self.config.num_heads)
        {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 DiT embed_dim {} not divisible by num_heads {}",
                self.config.embed_dim, self.config.num_heads
            )));
        }
        if self.cross_attention_cond_ids != ["prompt", "seconds_total"]
            || self.global_cond_ids != ["seconds_total"]
            || self.local_add_cond_ids != ["inpaint_mask", "inpaint_masked_input"]
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 conditioning routes must be cross=[prompt,seconds_total], \
                 global=[seconds_total], local=[inpaint_mask,inpaint_masked_input]"
                    .into(),
            ));
        }
        if !self.input_concat_ids.is_empty()
            || !self.prepend_cond_ids.is_empty()
            || !self.modular_local_cond_configs.is_empty()
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 input-concat, prepend, and modular-local conditioning branches \
                 are unsupported"
                    .into(),
            ));
        }
        if !self.mask_padding_attention || !self.use_effective_length_for_schedule {
            return Err(AudioError::Msg(
                "Stable Audio 3 shipped DiTs require mask_padding_attention and \
                 use_effective_length_for_schedule"
                    .into(),
            ));
        }
        self.config.validate_shipped()
    }

    /// Effective inference-time shift. Upstream supplies this LogSNR setting when the optional
    /// `sampling_distribution_shift_options` dictionary is absent.
    pub fn effective_sampling_shift(&self) -> DistributionShiftConfig {
        self.sampling_distribution_shift_options
            .clone()
            .unwrap_or_else(DistributionShiftConfig::sampling_default)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModularLocalConditioningConfig {
    pub id: String,
    pub dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionModuleType {
    Dit,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionObjective {
    #[default]
    V,
    RfDenoiser,
    RectifiedFlow,
}

/// The complete constructor dictionary consumed by `DiffusionTransformer` and
/// `ContinuousTransformer` for the frozen checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DitConfig {
    #[serde(default = "default_32")]
    pub io_channels: usize,
    #[serde(default = "default_one")]
    pub patch_size: usize,
    #[serde(default = "default_768")]
    pub embed_dim: usize,
    #[serde(default)]
    pub cond_token_dim: usize,
    #[serde(default = "default_true")]
    pub project_cond_tokens: bool,
    #[serde(default)]
    pub global_cond_dim: usize,
    #[serde(default = "default_true")]
    pub project_global_cond: bool,
    #[serde(default)]
    pub input_concat_dim: usize,
    #[serde(default)]
    pub prepend_cond_dim: usize,
    #[serde(default = "default_12")]
    pub depth: usize,
    #[serde(default = "default_8")]
    pub num_heads: usize,
    #[serde(default)]
    pub transformer_type: TransformerType,
    #[serde(default)]
    pub global_cond_type: GlobalConditioningType,
    #[serde(default)]
    pub timestep_cond_type: TimestepConditioningType,
    #[serde(default)]
    pub timestep_embed_dim: Option<usize>,
    #[serde(default)]
    pub timestep_features_type: FourierFeaturesType,
    #[serde(default = "default_timestep_features_dim")]
    pub timestep_features_dim: usize,
    #[serde(default)]
    pub timestep_features_logsnr: bool,
    #[serde(default)]
    pub local_add_cond_dim: Option<usize>,
    #[serde(default)]
    pub attn_kwargs: AttentionConfig,
    #[serde(default)]
    pub norm_type: NormType,
    #[serde(default)]
    pub norm_kwargs: NormConfig,
    #[serde(default)]
    pub ff_kwargs: FeedForwardConfig,
    #[serde(default)]
    pub num_memory_tokens: usize,
    #[serde(default = "default_true")]
    pub rotary_pos_emb: bool,
    #[serde(default)]
    pub cross_attn_rotary_pos_emb: bool,
    #[serde(default = "default_true")]
    pub zero_init_branch_outputs: bool,
    #[serde(default)]
    pub conformer: bool,
    #[serde(default)]
    pub causal: bool,
    #[serde(default)]
    pub use_sinusoidal_emb: bool,
    #[serde(default)]
    pub use_abs_pos_emb: bool,
    #[serde(default = "default_abs_pos_length")]
    pub abs_pos_emb_max_length: usize,
    #[serde(default)]
    pub sliding_window: Option<Vec<usize>>,
    #[serde(default = "default_minus_one")]
    pub final_cross_attn_ix: isize,
    #[serde(default)]
    pub layer_scale: bool,
}

fn default_one() -> usize {
    1
}
fn default_8() -> usize {
    8
}
fn default_12() -> usize {
    12
}
fn default_abs_pos_length() -> usize {
    10_000
}
fn default_minus_one() -> isize {
    -1
}

impl DitConfig {
    fn validate_shipped(&self) -> Result<()> {
        let small = self.embed_dim == 1024
            && self.depth == 20
            && self.num_heads == 16
            && !self.attn_kwargs.differential;
        let medium = self.embed_dim == 1536
            && self.depth == 24
            && self.num_heads == 24
            && self.attn_kwargs.differential;
        if !small && !medium {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 supports only shipped small 1024x20 ordinary or medium \
                 1536x24 differential DiTs, got {}x{} heads={} differential={}",
                self.embed_dim, self.depth, self.num_heads, self.attn_kwargs.differential
            )));
        }
        if self.io_channels != 256
            || self.patch_size != 1
            || self.cond_token_dim != 768
            || self.global_cond_dim != 768
            || self.local_add_cond_dim != Some(257)
            || self.input_concat_dim != 0
            || self.prepend_cond_dim != 0
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 shipped DiT dimensions are io=256, patch=1, cond/global=768, \
                 local=257 with no input/prepend concat"
                    .into(),
            ));
        }
        if self.transformer_type != TransformerType::ContinuousTransformer
            || self.global_cond_type != GlobalConditioningType::AdaLn
            || self.timestep_cond_type != TimestepConditioningType::Global
            || self.timestep_embed_dim.is_some()
            || self.timestep_features_type != FourierFeaturesType::Expo
            || self.timestep_features_dim != 256
            || self.timestep_features_logsnr
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 requires continuous AdaLN global conditioning and direct \
                 256-dimensional Expo timestep features"
                    .into(),
            ));
        }
        if !self.project_cond_tokens
            || !self.project_global_cond
            || self.attn_kwargs.qk_norm != QkNorm::Rms
            || self.attn_kwargs.qk_norm_eps != 1e-6
            || self.attn_kwargs.feat_scale
            || self.norm_type != NormType::RmsNorm
            || !self.norm_kwargs.force_fp32
            || self.norm_kwargs.fix_scale
            || self.norm_kwargs.eps != 1e-5
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 requires projected conditioning, RMS block/QK norms, and \
                 force-fp32 block normalization"
                    .into(),
            ));
        }
        if self.ff_kwargs.mult != 4.0
            || self.ff_kwargs.no_bias
            || !self.ff_kwargs.glu
            || self.ff_kwargs.use_conv
            || !self.ff_kwargs.zero_init_output
            || self.ff_kwargs.sinusoidal
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 requires the shipped biased linear GLU feed-forward".into(),
            ));
        }
        if self.num_memory_tokens != 64
            || !self.rotary_pos_emb
            || self.cross_attn_rotary_pos_emb
            || !self.zero_init_branch_outputs
            || self.conformer
            || self.causal
            || self.use_sinusoidal_emb
            || self.use_abs_pos_emb
            || self.sliding_window.is_some()
            || self.final_cross_attn_ix != -1
            || self.layer_scale
        {
            return Err(AudioError::Msg(
                "Stable Audio 3 supports only 64-memory-token global RoPE blocks with every \
                 layer cross-attending and no conformer/positional/sliding/layer-scale branches"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformerType {
    #[default]
    ContinuousTransformer,
    MmTransformer,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlobalConditioningType {
    #[default]
    #[serde(rename = "prepend")]
    Prepend,
    #[serde(rename = "adaLN")]
    AdaLn,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestepConditioningType {
    #[default]
    Global,
    InputConcat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionConfig {
    #[serde(default)]
    pub qk_norm: QkNorm,
    #[serde(default = "default_qk_eps")]
    pub qk_norm_eps: f64,
    #[serde(default)]
    pub differential: bool,
    #[serde(default)]
    pub feat_scale: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            qk_norm: QkNorm::None,
            qk_norm_eps: default_qk_eps(),
            differential: false,
            feat_scale: false,
        }
    }
}

fn default_qk_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QkNorm {
    L2,
    Ln,
    Rms,
    Dyt,
    #[default]
    None,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    #[default]
    LayerNorm,
    RmsNorm,
    Dyt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormConfig {
    #[serde(default)]
    pub fix_scale: bool,
    #[serde(default)]
    pub force_fp32: bool,
    #[serde(default = "default_norm_eps")]
    pub eps: f64,
}

impl Default for NormConfig {
    fn default() -> Self {
        Self {
            fix_scale: false,
            force_fp32: false,
            eps: default_norm_eps(),
        }
    }
}

fn default_norm_eps() -> f64 {
    1e-5
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedForwardConfig {
    #[serde(default = "default_four")]
    pub mult: f64,
    #[serde(default)]
    pub no_bias: bool,
    #[serde(default = "default_true")]
    pub glu: bool,
    #[serde(default)]
    pub use_conv: bool,
    #[serde(default = "default_three_usize")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_true")]
    pub zero_init_output: bool,
    #[serde(default)]
    pub sinusoidal: bool,
}

impl Default for FeedForwardConfig {
    fn default() -> Self {
        Self {
            mult: default_four(),
            no_bias: false,
            glu: true,
            use_conv: false,
            conv_kernel_size: default_three_usize(),
            zero_init_output: true,
            sinusoidal: false,
        }
    }
}

fn default_four() -> f64 {
    4.0
}
fn default_three_usize() -> usize {
    3
}

/// Typed union of all upstream distribution-shift constructors. Fields not used by the selected
/// `kind` retain their upstream defaults, which makes minimal and explicit dictionaries identical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionShiftConfig {
    #[serde(rename = "type", default)]
    pub kind: DistributionShiftType,
    #[serde(default = "default_256")]
    pub min_length: usize,
    #[serde(default = "default_4096")]
    pub max_length: usize,
    #[serde(default = "default_base_shift")]
    pub base_shift: f64,
    #[serde(default = "default_max_shift")]
    pub max_shift: f64,
    #[serde(default)]
    pub use_sine: bool,
    #[serde(default = "default_one_f64")]
    pub alpha_min: f64,
    #[serde(default = "default_one_f64")]
    pub alpha_max: f64,
    #[serde(default = "default_anchor_length")]
    pub anchor_length: usize,
    #[serde(default = "default_anchor_logsnr")]
    pub anchor_logsnr: f64,
    #[serde(default = "default_one_f64")]
    pub rate: f64,
    #[serde(default = "default_logsnr_end")]
    pub logsnr_end: f64,
}

impl DistributionShiftConfig {
    pub fn sampling_default() -> Self {
        Self {
            kind: DistributionShiftType::Logsnr,
            rate: 0.0,
            ..Self::default()
        }
    }
}

impl Default for DistributionShiftConfig {
    fn default() -> Self {
        Self {
            kind: DistributionShiftType::Full,
            min_length: default_256(),
            max_length: default_4096(),
            base_shift: default_base_shift(),
            max_shift: default_max_shift(),
            use_sine: false,
            alpha_min: 1.0,
            alpha_max: 1.0,
            anchor_length: default_anchor_length(),
            anchor_logsnr: default_anchor_logsnr(),
            rate: 1.0,
            logsnr_end: default_logsnr_end(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistributionShiftType {
    None,
    Flux,
    #[default]
    Full,
    Logsnr,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_defaults_are_values_not_false_or_null() {
        let encoder: EncoderConfig = serde_json::from_str("{}").unwrap();
        let explicit: EncoderConfig = serde_json::from_str(
            r#"{
                "in_channels": 2, "channels": 128, "latent_dim": 32,
                "c_mults": [1,2,4,8], "strides": [2,4,8,8],
                "transformer_depths": [3,3,3,3], "checkpointing": false,
                "conformer": false, "layer_scale": false, "causal": false,
                "differential": true, "variable_stride": false, "mask_noise": 0,
                "conv_mapping": false, "freeze_backbone": false, "dim_heads": 128,
                "chunk_size": 128, "chunk_midpoint_shift": false, "dyt": true,
                "feat_scale": false, "ff_mult": 3, "mapping_bias": true,
                "cross_attn": false, "use_flash": false, "use_snake": false,
                "use_dilated_conv": false, "conv_bias": true,
                "enable_inner_layer_dropout": false, "mapping_style": "none"
            }"#,
        )
        .unwrap();
        assert_eq!(encoder, explicit);
        assert!(encoder.dyt, "upstream SAME default is DyT");
        assert!(encoder.differential);

        let shift: DistributionShiftConfig =
            serde_json::from_str(r#"{"min_length":256,"max_length":4096}"#).unwrap();
        assert_eq!(shift.kind, DistributionShiftType::Full);
        assert_eq!(shift, DistributionShiftConfig::default());
    }

    #[test]
    fn decoder_minimal_and_explicit_upstream_defaults_are_identical() {
        let minimal: DecoderConfig = serde_json::from_str("{}").unwrap();
        let explicit: DecoderConfig = serde_json::from_str(
            r#"{
                "out_channels": 2, "channels": 128, "latent_dim": 32,
                "c_mults": [1,2,4,8], "strides": [2,4,8,8],
                "transformer_depths": [3,3,3,3],
                "sinusoidal_blocks": [0,0,0,0], "checkpointing": false,
                "conformer": false, "layer_scale": false, "causal": false,
                "differential": true, "variable_stride": false, "mask_noise": 0,
                "conv_mapping": false, "freeze_backbone": false, "dim_heads": 128,
                "chunk_size": 128, "chunk_midpoint_shift": false, "dyt": true,
                "feat_scale": false, "ff_mult": 3, "mapping_bias": true,
                "cross_attn": false, "use_flash": false, "use_snake": false,
                "use_dilated_conv": false, "conv_bias": true,
                "enable_inner_layer_dropout": false, "mapping_style": "none"
            }"#,
        )
        .unwrap();
        assert_eq!(minimal, explicit);
        assert!(minimal.dyt, "upstream SAME decoder default is DyT");
    }

    #[test]
    fn training_only_unknown_fields_remain_tolerated() {
        let value = serde_json::json!({
            "model_type": "autoencoder",
            "sample_rate": 44100,
            "sample_size": 24576,
            "audio_channels": 2,
            "model": {
                "pretransform": {"type":"patched","config":{"patch_size":256,"channels":2}},
                "encoder": {"type":"same","config":{"in_channels":512,"latent_dim":256,"c_mults":[6],"strides":[16],"transformer_depths":[6]}},
                "decoder": {"type":"same","config":{"out_channels":512,"latent_dim":256,"c_mults":[6],"strides":[16],"transformer_depths":[6],"sinusoidal_blocks":[0]}},
                "bottleneck": {"type":"softnorm","config":{"dim":256}},
                "latent_dim":256,"downsampling_ratio":4096,"io_channels":2
            },
            "training": {"future_only": {"arbitrary": [1,2,3]}}
        });
        let parsed: StableAudioConfig = serde_json::from_value(value).unwrap();
        parsed.validate().unwrap();
        assert!(parsed.autoencoder().encoder.config.dyt);
    }
}

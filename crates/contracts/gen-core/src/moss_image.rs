//! Frozen MOSS-Image request and execution-receipt contracts carried by
//! [`crate::GenerationRequest`] and the [`Generator`](crate::Generator) MOSS entry points. No
//! tensors, model acquisition, or backend execution. The schedule, guidance, DMD and package
//! helpers live in `gen_core_moss_image::moss_image` (sc-27316). Source authority and precision
//! policies are recorded in `docs/models/moss-image.md`.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// Shared byte authority for both native backends.
pub use crate::moss_image_assets::{
    moss_image_asset, verify_moss_image_asset, MossImageAssetIdentity, MOSS_IMAGE_ASSETS,
};
use crate::{Error, GenerationRequest, Image, Result};

pub const BASE_REPO: &str = "OpenMOSS-Team/MOSS-Image-Base";
pub const BASE_REVISION: &str = "1ceede80badddd6a7812069619edaa5633587192";
pub const SFT_REPO: &str = "OpenMOSS-Team/MOSS-Image-SFT";
pub const SFT_REVISION: &str = "fc5de0d56fc910cbca6f6625a70f5f582b26deb1";
pub const DMD_REPO: &str = "lodestones/MOSS-Image-SFT-8-steps-dmd";
pub const DMD_REVISION: &str = "5ff314b4c1470876b114b810146085db8a181a6e";
pub const PIPELINE_SHA256: &str =
    "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31";
pub const LICENSE: &str = "Apache-2.0";
/// Optional immutable components. Include their exact pins and acceleration policy in cache keys.
/// Exact SFT snapshot supplying shared encoder/tokenizer/VAE/scheduler for either DiT variant.
pub const MOSS_SHARED_SNAPSHOT_COMPONENT: &str = "moss-shared-snapshot";
pub const DMD_LORA_COMPONENT: &str = "moss-dmd-lora";
pub const DMD_TIME_COMPONENT: &str = "moss-dmd-time";
pub const DMD_DATASPACE_COMPONENT: &str = "moss-dmd-dataspace";
pub const DMD_TRANSFORMER_COMPONENT: &str = "moss-dmd-transformer";
pub const DMD_PROJECTIONS: [&str; 6] = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"];
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageVariant {
    Base,
    Sft,
}
impl MossImageVariant {
    pub fn provider_id(self) -> &'static str {
        match self {
            Self::Base => "moss_image_base",
            Self::Sft => "moss_image_sft",
        }
    }
    pub fn repository(self) -> &'static str {
        match self {
            Self::Base => BASE_REPO,
            Self::Sft => SFT_REPO,
        }
    }
    pub fn revision(self) -> &'static str {
        match self {
            Self::Base => BASE_REVISION,
            Self::Sft => SFT_REVISION,
        }
    }
    pub fn from_provider_id(id: &str) -> Option<Self> {
        match id {
            "moss_image_base" => Some(Self::Base),
            "moss_image_sft" => Some(Self::Sft),
            _ => None,
        }
    }
    pub fn validate_identity(self, repo: &str, revision: &str) -> Result<()> {
        require(
            repo == self.repository() && revision == self.revision(),
            "checkpoint identity differs from the frozen Base/SFT package",
        )
    }
}
/// The two author-published DMD execution policies. Neither implies equal output to the other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossDmdPrecision {
    /// BF16 SFT backbone plus unmerged FP32 low-rank residuals, time MLP, shared/head modulation.
    #[default]
    UnmergedBf16,
    /// The published merged distilled transformer loaded AND executed in FP32, without autocast.
    FullFp32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MossImageAcceleration {
    #[default]
    Off,
    Dmd(MossDmdPrecision),
}
#[derive(Clone, Debug, PartialEq)]
pub struct MossSkipLayerGuidance {
    pub layers: Vec<u32>,
    pub scale: f32,
    pub start: f32,
    pub stop: f32,
}
/// Model-specific controls; ordinary requests remain default-OFF even when DMD assets are cached.
#[derive(Clone, Debug, PartialEq)]
pub struct MossImageOptions {
    pub acceleration: MossImageAcceleration,
    pub max_sequence_length: u32,
    /// None uses author default `[22]` for ordinary inference. Some(empty) disables skipping.
    pub uncond_skip_layers: Option<Vec<u32>>,
    pub slg: Option<MossSkipLayerGuidance>,
    /// None uses the frozen training grid shift 1.0.
    pub training_shift: Option<f32>,
}
impl Default for MossImageOptions {
    fn default() -> Self {
        Self {
            acceleration: MossImageAcceleration::Off,
            max_sequence_length: 4096,
            uncond_skip_layers: None,
            slg: None,
            training_shift: None,
        }
    }
}
impl MossImageOptions {
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            acceleration: _,
            max_sequence_length: _,
            uncond_skip_layers: _,
            slg,
            training_shift,
        } = self;
        if let Some(x) = training_shift.filter(|x| !x.is_finite()) {
            return Some(("moss_image.training_shift", x));
        }
        if let Some(MossSkipLayerGuidance {
            layers: _,
            scale,
            start,
            stop,
        }) = slg
        {
            for (key, x) in [
                ("moss_image.slg.scale", *scale),
                ("moss_image.slg.start", *start),
                ("moss_image.slg.stop", *stop),
            ] {
                if !x.is_finite() {
                    return Some((key, x));
                }
            }
        }
        None
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct MossImageResolved {
    pub variant: MossImageVariant,
    pub acceleration: MossImageAcceleration,
    pub steps: u32,
    pub guidance: f32,
    pub shift: f32,
    pub training_shift: f32,
    pub max_sequence_length: u32,
    pub uncond_skip_layers: Vec<u32>,
    pub slg: Option<MossSkipLayerGuidance>,
}
impl MossImageResolved {
    pub fn resolve(variant: MossImageVariant, request: &GenerationRequest) -> Result<Self> {
        let options = request.moss_image.clone().unwrap_or_default();
        require(
            options.first_nonfinite_float().is_none(),
            "nonfinite MOSS control",
        )?;
        require(
            request.first_nonfinite_float().is_none(),
            "nonfinite request control",
        )?;
        require(
            request.width > 0
                && request.height > 0
                && request.width.is_multiple_of(16)
                && request.height.is_multiple_of(16),
            "dimensions must be positive multiples of 16",
        )?;
        require(request.count > 0, "image count must be positive")?;
        require(
            (1..=4096).contains(&options.max_sequence_length),
            "context limit must be in 1..=4096",
        )?;
        require(
            request.conditioning.is_empty(),
            "reference/edit/control conditioning is not supported",
        )?;
        require(
            request.sampler.as_deref().is_none_or(|v| v == "euler"),
            "only the published Euler sampler is supported",
        )?;
        require(
            request
                .scheduler
                .as_deref()
                .is_none_or(|v| v == "moss_training_grid"),
            "only the MOSS training-grid scheduler is supported",
        )?;
        require(
            request.true_cfg.is_none()
                && request.timestep_to_start_cfg.is_none()
                && request.cfg_interval.is_none()
                && request.phases.is_none(),
            "generic CFG scheduling and phases are not MOSS controls",
        )?;
        require(
            request
                .guidance_method
                .as_deref()
                .is_none_or(|v| v == "cfg")
                && request.guidance_eta.is_none()
                && request.guidance_momentum.is_none()
                && request.guidance_norm_threshold.is_none(),
            "only published MOSS CFG/SLG guidance is supported",
        )?;
        let accelerated = matches!(options.acceleration, MossImageAcceleration::Dmd(_));
        let steps = request.steps.unwrap_or(if accelerated { 8 } else { 50 });
        let guidance = request
            .guidance
            .unwrap_or(if accelerated { 0.0 } else { 3.0 });
        let shift = request.scheduler_shift.unwrap_or(5.0);
        let training_shift = options.training_shift.unwrap_or(1.0);
        require(
            guidance.is_finite()
                && guidance >= 0.0
                && shift.is_finite()
                && shift > 0.0
                && training_shift > 0.0,
            "guidance must be nonnegative and shifts positive",
        )?;
        require(steps > 0, "steps must be positive")?;
        if let Some(layers) = &options.uncond_skip_layers {
            validate_layers(layers)?;
        }
        if let Some(slg) = &options.slg {
            validate_layers(&slg.layers)?;
            require(
                slg.scale >= 0.0 && 0.0 <= slg.start && slg.start <= slg.stop && slg.stop <= 1.0,
                "SLG requires nonnegative scale and 0 <= start <= stop <= 1",
            )?;
        }
        if accelerated {
            require(
                variant == MossImageVariant::Sft,
                "DMD requires the exact SFT parent",
            )?;
            require(
                (1..=8).contains(&steps),
                "DMD steps must select a trained prefix in 1..=8",
            )?;
            require(
                request.guidance.is_none_or(|g| g == 0.0)
                    && request.negative_prompt.as_deref().is_none_or(str::is_empty),
                "DMD does not support CFG or negative prompts",
            )?;
            require(
                request.sampler.is_none()
                    && request.scheduler.is_none()
                    && request.scheduler_shift.is_none()
                    && options.training_shift.is_none()
                    && options.slg.is_none()
                    && options.uncond_skip_layers.is_none()
                    && request.guidance_method.is_none(),
                "DMD uses fixed step codes, no Euler, layer skipping or SLG",
            )?;
        }
        Ok(Self {
            variant,
            acceleration: options.acceleration,
            steps,
            guidance,
            shift,
            training_shift,
            max_sequence_length: options.max_sequence_length,
            uncond_skip_layers: if accelerated {
                vec![]
            } else {
                options.uncond_skip_layers.unwrap_or_else(|| vec![22])
            },
            slg: options.slg,
        })
    }
    pub fn skipping_active(&self, step: usize, actual_steps: usize) -> bool {
        if actual_steps == 0 || step >= actual_steps {
            return false;
        }
        let (start, stop) = self
            .slg
            .as_ref()
            .map(|s| (s.start, s.stop))
            .unwrap_or((0.0, 1.0));
        let fraction = step as f32 / actual_steps as f32;
        start <= fraction && fraction < stop
    }
}
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Unsupported(format!("MOSS-Image: {message}")))
    }
}
fn validate_layers(layers: &[u32]) -> Result<()> {
    require(
        layers.iter().all(|i| *i < 42),
        "skip layers must be zero-based indices in 0..42",
    )
}
/// Checked CPU representation for native advanced entry points. Supplied embeddings remain F32
/// until transformer projection; binary masks compact each sample before the DiT forward.
#[derive(Clone, Copy, Debug)]
pub struct MossImageEmbeddings<'a> {
    pub values: &'a [f32],
    pub batch: usize,
    pub sequence: usize,
    pub mask: Option<&'a [u8]>,
}
impl MossImageEmbeddings<'_> {
    pub fn validate(&self, max_sequence_length: u32) -> Result<()> {
        let count = self
            .batch
            .checked_mul(self.sequence)
            .and_then(|n| n.checked_mul(4096));
        require(
            self.batch > 0
                && self.sequence > 0
                && self.sequence <= max_sequence_length as usize
                && count == Some(self.values.len()),
            "embeddings must be [B,S,4096] within context limit",
        )?;
        require(
            self.values.iter().all(|v| v.is_finite()),
            "nonfinite embedding",
        )?;
        if let Some(mask) = self.mask {
            require(
                mask.len() == self.batch * self.sequence && mask.iter().all(|v| *v <= 1),
                "attention mask must be binary [B,S]",
            )?;
            require(
                mask.chunks_exact(self.sequence).all(|row| row.contains(&1)),
                "each embedding row needs a valid token",
            )?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageOutputKind {
    #[default]
    Images,
    Latents,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct MossImageAdvancedInputs<'a> {
    pub positive: Option<MossImageEmbeddings<'a>>,
    pub negative: Option<MossImageEmbeddings<'a>>,
    /// Per-prompt negative strings; replaces the generic broadcast negative prompt.
    pub negative_prompts: Option<&'a [String]>,
    /// Author NCHW [prompt_batch * image_count,128,height/16,width/16], FP32.
    pub noise: Option<&'a [f32]>,
    pub output: MossImageOutputKind,
}
impl MossImageAdvancedInputs<'_> {
    pub fn validate(
        &self,
        request: &GenerationRequest,
        resolved: &MossImageResolved,
    ) -> Result<usize> {
        let batch = if let Some(positive) = self.positive {
            require(
                request.prompt.is_empty() && request.prompt_batch.is_empty(),
                "supply prompts or positive embeddings, not both",
            )?;
            positive.validate(resolved.max_sequence_length)?;
            positive.batch
        } else {
            request.prompt_batch.len().max(1)
        };
        if let Some(negative) = self.negative {
            negative.validate(resolved.max_sequence_length)?;
            require(
                negative.batch == batch
                    && request.negative_prompt.as_deref().is_none_or(str::is_empty),
                "negative embeddings must match positive batch and replace negative text",
            )?;
            require(
                matches!(resolved.acceleration, MossImageAcceleration::Off),
                "DMD does not consume negative embeddings",
            )?;
        }
        if let Some(negatives) = self.negative_prompts {
            require(
                negatives.len() == batch
                    && self.negative.is_none()
                    && request.negative_prompt.is_none(),
                "negative prompt batch must match positive batch and replace other negative input",
            )?;
            require(
                matches!(resolved.acceleration, MossImageAcceleration::Off),
                "DMD does not consume negative prompts",
            )?;
        }
        let images = batch
            .checked_mul(request.count as usize)
            .ok_or_else(|| Error::Unsupported("MOSS batch overflow".into()))?;
        if let Some(noise) = self.noise {
            let n = images
                .checked_mul(128)
                .and_then(|n| n.checked_mul(request.height as usize / 16))
                .and_then(|n| n.checked_mul(request.width as usize / 16));
            require(
                n == Some(noise.len()) && noise.iter().all(|v| v.is_finite()),
                "supplied noise must be finite FP32 with exact NCHW shape",
            )?;
        }
        Ok(images)
    }
}
#[derive(Clone, Debug)]
pub enum MossImageAdvancedOutput {
    Images(Vec<Image>),
    Latents { values: Vec<f32>, shape: [usize; 4] },
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AccelerationWire {
    Off,
    UnmergedBf16,
    FullFp32,
}
impl Serialize for MossImageAcceleration {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Off => AccelerationWire::Off,
            Self::Dmd(MossDmdPrecision::UnmergedBf16) => AccelerationWire::UnmergedBf16,
            Self::Dmd(MossDmdPrecision::FullFp32) => AccelerationWire::FullFp32,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for MossImageAcceleration {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Ok(match AccelerationWire::deserialize(deserializer)? {
            AccelerationWire::Off => Self::Off,
            AccelerationWire::UnmergedBf16 => Self::Dmd(MossDmdPrecision::UnmergedBf16),
            AccelerationWire::FullFp32 => Self::Dmd(MossDmdPrecision::FullFp32),
        })
    }
}
impl<'de> Deserialize<'de> for MossSkipLayerGuidance {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            layers: Vec<u32>,
            scale: f32,
            start: f32,
            stop: f32,
        }
        let w = Wire::deserialize(deserializer)?;
        let out = Self {
            layers: w.layers,
            scale: w.scale,
            start: w.start,
            stop: w.stop,
        };
        if out.layers.iter().any(|i| *i >= 42)
            || !out.scale.is_finite()
            || out.scale < 0.0
            || !out.start.is_finite()
            || !out.stop.is_finite()
            || !(0.0 <= out.start && out.start <= out.stop && out.stop <= 1.0)
        {
            return Err(serde::de::Error::custom("invalid MOSS SLG controls"));
        }
        Ok(out)
    }
}
impl<'de> Deserialize<'de> for MossImageOptions {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        fn context() -> u32 {
            4096
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            #[serde(default)]
            acceleration: MossImageAcceleration,
            #[serde(default = "context")]
            max_sequence_length: u32,
            #[serde(default)]
            uncond_skip_layers: Option<Vec<u32>>,
            #[serde(default)]
            slg: Option<MossSkipLayerGuidance>,
            #[serde(default)]
            training_shift: Option<f32>,
        }
        let w = Wire::deserialize(deserializer)?;
        let out = Self {
            acceleration: w.acceleration,
            max_sequence_length: w.max_sequence_length,
            uncond_skip_layers: w.uncond_skip_layers,
            slg: w.slg,
            training_shift: w.training_shift,
        };
        // A mode's inner controls can be checked without knowing Base/SFT; the provider resolver
        // additionally enforces variant and generic-request compatibility.
        MossImageResolved::resolve(
            MossImageVariant::Sft,
            &GenerationRequest {
                moss_image: Some(out.clone()),
                ..Default::default()
            },
        )
        .map_err(serde::de::Error::custom)?;
        Ok(out)
    }
}
impl MossImageResolved {
    /// Validate externally decoded effective controls against the same canonical request resolver.
    pub fn validate(&self) -> Result<()> {
        let ordinary = matches!(self.acceleration, MossImageAcceleration::Off);
        let expected = Self::resolve(
            self.variant,
            &GenerationRequest {
                steps: Some(self.steps),
                guidance: Some(self.guidance),
                scheduler_shift: ordinary.then_some(self.shift),
                moss_image: Some(MossImageOptions {
                    acceleration: self.acceleration,
                    max_sequence_length: self.max_sequence_length,
                    uncond_skip_layers: ordinary.then(|| self.uncond_skip_layers.clone()),
                    slg: self.slg.clone(),
                    training_shift: ordinary.then_some(self.training_shift),
                }),
                ..Default::default()
            },
        )?;
        require(
            *self == expected,
            "resolved controls are inconsistent with their execution mode",
        )
    }
}
impl<'de> Deserialize<'de> for MossImageResolved {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            variant: MossImageVariant,
            acceleration: MossImageAcceleration,
            steps: u32,
            guidance: f32,
            shift: f32,
            training_shift: f32,
            max_sequence_length: u32,
            uncond_skip_layers: Vec<u32>,
            slg: Option<MossSkipLayerGuidance>,
        }
        let w = Wire::deserialize(deserializer)?;
        let out = Self {
            variant: w.variant,
            acceleration: w.acceleration,
            steps: w.steps,
            guidance: w.guidance,
            shift: w.shift,
            training_shift: w.training_shift,
            max_sequence_length: w.max_sequence_length,
            uncond_skip_layers: w.uncond_skip_layers,
            slg: w.slg,
        };
        out.validate().map_err(serde::de::Error::custom)?;
        Ok(out)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageAssetRole {
    Transformer,
    TextEncoder,
    Tokenizer,
    Scheduler,
    Vae,
    DmdLora,
    DmdTime,
    DmdDataspace,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageAssetReceipt {
    pub role: MossImageAssetRole,
    pub repository: String,
    pub revision: String,
    pub relative_path: String,
    pub sha256: String,
}
impl MossImageAssetReceipt {
    /// Construct from an approved logical asset and retained pin, independent of physical basename.
    /// Frozen repository/revision/path, size and digest must all authenticate; labels alone fail.
    pub fn from_pin(
        role: MossImageAssetRole,
        repository: &str,
        revision: &str,
        relative_path: &str,
        pin: &crate::PinnedWeightsFile,
    ) -> Result<Self> {
        // sc-26325 / F-282: explicit file components may use content-addressed filenames.
        // Bind the logical approved identity to actual bytes, never to a lexical suffix.
        verify_moss_image_asset(pin, repository, revision, relative_path)?;
        Ok(Self {
            role,
            repository: repository.into(),
            revision: revision.into(),
            relative_path: relative_path.into(),
            sha256: hex_digest(pin.content_sha256()),
        })
    }
}
fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageDtype {
    Bf16,
    F32,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MossImageSamplingReceipt {
    Euler {
        sigmas: Vec<f32>,
    },
    Dmd {
        step_codes: Vec<f32>,
        mix_matrix_sha256: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageSampleReceipt {
    /// Actual effective engine seed. None denotes caller-supplied noise, which overrides a seed.
    pub seed: Option<u64>,
    /// FP32 little-endian NCHW bytes before upload. No accelerator readback is required.
    pub noise_sha256: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageConditioningSource {
    Text,
    Embeddings,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageConditioningInputReceipt {
    pub source: MossImageConditioningSource,
    /// Text input bytes, or supplied FP32 LE embedding bytes followed by mask bytes; no raw prompt.
    pub sha256: String,
    pub sequence_length: u32,
    pub valid_tokens: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageConditioningReceipt {
    pub positive: MossImageConditioningInputReceipt,
    pub negative: Option<MossImageConditioningInputReceipt>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageExecutionDevice {
    Cpu,
    Mlx,
    Cuda,
    Metal,
}
/// Construction data. Only `MossImageExecutionReport::new` produces a validated completed receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageExecutionData {
    pub execution_device: MossImageExecutionDevice,
    pub resolved: MossImageResolved,
    pub assets: Vec<MossImageAssetReceipt>,
    pub sampling: MossImageSamplingReceipt,
    /// Ordered prompt-major, image-minor, matching output order.
    pub samples: Vec<MossImageSampleReceipt>,
    pub conditioning: Vec<MossImageConditioningReceipt>,
    pub images_per_prompt: u32,
    pub latent_shape: [usize; 4],
    pub output_kind: MossImageOutputKind,
    pub component_dtypes: BTreeMap<String, MossImageDtype>,
    #[serde(with = "adapter_receipt_wire")]
    pub adapter_reports: Vec<crate::AdapterApplyReport>,
}
/// Validated actual completed execution, exposed by the additive Generator getter. The provider
/// clears this before every attempt and publishes only after final success/cancellation checks.
/// Consistency validation does not execute a model or prove that arbitrary claimed hashes are real;
/// production builders must consume retained verified pins and actual execution facts.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct MossImageExecutionReport(MossImageExecutionData);
impl MossImageExecutionReport {
    pub fn new(data: MossImageExecutionData) -> Result<Self> {
        validate_execution_receipt(&data)?;
        Ok(Self(data))
    }
    pub fn data(&self) -> &MossImageExecutionData {
        &self.0
    }
    pub fn into_data(self) -> MossImageExecutionData {
        self.0
    }
}
impl<'de> Deserialize<'de> for MossImageExecutionReport {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Self::new(MossImageExecutionData::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
mod adapter_receipt_wire {
    use super::*;
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Wire {
        adapter_path: std::path::PathBuf,
        applied: usize,
        skipped: Vec<String>,
    }
    pub fn serialize<S: serde::Serializer>(
        reports: &[crate::AdapterApplyReport],
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        reports
            .iter()
            .map(|r| Wire {
                adapter_path: r.adapter_path.clone(),
                applied: r.applied,
                skipped: r.skipped.clone(),
            })
            .collect::<Vec<_>>()
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Vec<crate::AdapterApplyReport>, D::Error> {
        Ok(Vec::<Wire>::deserialize(deserializer)?
            .into_iter()
            .map(|r| crate::AdapterApplyReport {
                adapter_path: r.adapter_path,
                applied: r.applied,
                skipped: r.skipped,
            })
            .collect())
    }
}
fn receipt_asset_paths(mode: MossImageAcceleration) -> BTreeMap<(MossImageAssetRole, String), ()> {
    let mut result = BTreeMap::new();
    for path in SNAPSHOT_FILES.iter().copied().filter(|p| *p != "LICENSE") {
        let role = if path.starts_with("transformer/") {
            MossImageAssetRole::Transformer
        } else if path.starts_with("text_encoder/") {
            MossImageAssetRole::TextEncoder
        } else if path.starts_with("tokenizer/") {
            MossImageAssetRole::Tokenizer
        } else if path.starts_with("scheduler/") {
            MossImageAssetRole::Scheduler
        } else {
            MossImageAssetRole::Vae
        };
        if matches!(mode, MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32))
            && path.starts_with("transformer/diffusion_pytorch_model-000")
        {
            continue;
        }
        result.insert((role, path.into()), ());
    }
    if matches!(mode, MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)) {
        for i in 1..=4 {
            result.insert(
                (
                    MossImageAssetRole::Transformer,
                    format!("transformer/diffusion_pytorch_model-{i:05}-of-00004.safetensors"),
                ),
                (),
            );
        }
    }
    match mode {
        MossImageAcceleration::Off => (),
        MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => {
            result.insert((MossImageAssetRole::DmdLora, "lora.safetensors".into()), ());
            result.insert(
                (
                    MossImageAssetRole::DmdTime,
                    "time_embedding.safetensors".into(),
                ),
                (),
            );
            result.insert(
                (
                    MossImageAssetRole::DmdDataspace,
                    "dataspace.safetensors".into(),
                ),
                (),
            );
        }
        MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => {
            result.insert(
                (
                    MossImageAssetRole::DmdDataspace,
                    "dataspace.safetensors".into(),
                ),
                (),
            );
        }
    }
    result
}
fn validate_execution_receipt(data: &MossImageExecutionData) -> Result<()> {
    data.resolved.validate()?;
    let r = &data.resolved;
    let expected = receipt_asset_paths(r.acceleration);
    let mut seen = BTreeSet::new();
    for asset in &data.assets {
        let key = (asset.role, asset.relative_path.clone());
        require(
            expected.contains_key(&key) && seen.insert(key),
            "receipt has unexpected/duplicate consumed assets",
        )?;
        let (repo, revision) = match asset.role {
            MossImageAssetRole::Transformer
                if matches!(
                    r.acceleration,
                    MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)
                ) =>
            {
                (DMD_REPO, DMD_REVISION)
            }
            MossImageAssetRole::Transformer => (r.variant.repository(), r.variant.revision()),
            MossImageAssetRole::DmdLora
            | MossImageAssetRole::DmdTime
            | MossImageAssetRole::DmdDataspace => (DMD_REPO, DMD_REVISION),
            _ => (SFT_REPO, SFT_REVISION),
        };
        require(
            asset.repository == repo && asset.revision == revision && valid_digest(&asset.sha256),
            "receipt source identity/digest mismatch",
        )?;
    }
    require(
        seen.len() == expected.len(),
        "receipt omits consumed config/shard/tokenizer/auxiliary assets",
    )?;
    match (&data.sampling, r.acceleration) {
        (MossImageSamplingReceipt::Euler { sigmas }, MossImageAcceleration::Off) => require(
            *sigmas == moss_image_schedule(r.steps, r.shift, r.training_shift)?,
            "receipt sigma schedule differs from effective controls",
        )?,
        (
            MossImageSamplingReceipt::Dmd {
                step_codes,
                mix_matrix_sha256,
            },
            MossImageAcceleration::Dmd(_),
        ) => require(
            step_codes.len() == r.steps as usize
                && step_codes.iter().all(|v| v.is_finite())
                && valid_digest(mix_matrix_sha256),
            "receipt DMD prefix/matrix digest is invalid",
        )?,
        _ => {
            return Err(Error::Unsupported(
                "MOSS receipt sampler/mode mismatch".into(),
            ))
        }
    }
    let batch = data.conditioning.len();
    let count = batch.checked_mul(data.images_per_prompt as usize);
    require(
        batch > 0
            && data.images_per_prompt > 0
            && count == Some(data.samples.len())
            && data.latent_shape[0] == data.samples.len()
            && data.latent_shape[1] == 128
            && data.latent_shape[2] > 0
            && data.latent_shape[3] > 0,
        "receipt batch/output geometry mismatch",
    )?;
    require(
        data.latent_shape
            .iter()
            .try_fold(1usize, |n, v| n.checked_mul(*v))
            .is_some(),
        "receipt latent shape overflows",
    )?;
    require(
        data.samples.iter().all(|s| valid_digest(&s.noise_sha256)),
        "receipt noise digest is malformed",
    )?;
    for sample in &data.conditioning {
        for input in std::iter::once(&sample.positive).chain(sample.negative.as_ref()) {
            require(
                valid_digest(&input.sha256)
                    && input.sequence_length > 0
                    && input.sequence_length <= r.max_sequence_length
                    && input.valid_tokens > 0
                    && input.valid_tokens <= input.sequence_length,
                "receipt conditioning digest/length mismatch",
            )?;
        }
        require(
            sample.negative.is_some()
                == (matches!(r.acceleration, MossImageAcceleration::Off) && r.guidance != 1.0),
            "receipt negative conditioning does not match executed CFG",
        )?;
    }
    let mut expected_types: BTreeMap<String, MossImageDtype> = [
        ("transformer", MossImageDtype::Bf16),
        ("text_encoder", MossImageDtype::Bf16),
        ("vae", MossImageDtype::F32),
        ("latents", MossImageDtype::F32),
        ("time_embedding", MossImageDtype::Bf16),
        ("shared_block_modulation", MossImageDtype::Bf16),
        ("video_head_modulation", MossImageDtype::Bf16),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    if let MossImageAcceleration::Dmd(precision) = r.acceleration {
        for key in [
            "time_embedding",
            "shared_block_modulation",
            "video_head_modulation",
        ] {
            expected_types.insert(key.into(), MossImageDtype::F32);
        }
        if precision == MossDmdPrecision::FullFp32 {
            expected_types.insert("transformer".into(), MossImageDtype::F32);
        } else {
            expected_types.insert("lora_residual".into(), MossImageDtype::F32);
        }
    }
    if data.execution_device == MossImageExecutionDevice::Cpu {
        expected_types.insert("text_encoder".into(), MossImageDtype::F32);
    }
    require(
        data.component_dtypes == expected_types,
        "receipt component precision differs from the published execution policy",
    )?;
    if matches!(
        r.acceleration,
        MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16)
    ) {
        require(
            data.adapter_reports.len() == 1
                && data.adapter_reports[0].applied == 42 * 6
                && data.adapter_reports[0].skipped.is_empty()
                && !data.adapter_reports[0].adapter_path.as_os_str().is_empty(),
            "receipt must record complete unmerged adapter installation",
        )?;
    } else {
        require(
            data.adapter_reports.is_empty(),
            "ordinary/full-FP32 receipt must not claim an unmerged adapter installation",
        )?;
    }
    Ok(())
}
impl Serialize for MossSkipLayerGuidance {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        MossImageResolved::resolve(
            MossImageVariant::Sft,
            &GenerationRequest {
                moss_image: Some(MossImageOptions {
                    slg: Some(self.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .map_err(serde::ser::Error::custom)?;
        serde_json::json!({"layers":self.layers,"scale":self.scale,"start":self.start,"stop":self.stop}).serialize(serializer)
    }
}
impl Serialize for MossImageOptions {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        MossImageResolved::resolve(
            MossImageVariant::Sft,
            &GenerationRequest {
                moss_image: Some(self.clone()),
                ..Default::default()
            },
        )
        .map_err(serde::ser::Error::custom)?;
        serde_json::json!({"acceleration":self.acceleration,"max_sequence_length":self.max_sequence_length,"uncond_skip_layers":self.uncond_skip_layers,"slg":self.slg,"training_shift":self.training_shift}).serialize(serializer)
    }
}
impl Serialize for MossImageResolved {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        serde_json::json!({"variant":self.variant,"acceleration":self.acceleration,"steps":self.steps,"guidance":self.guidance,"shift":self.shift,"training_shift":self.training_shift,"max_sequence_length":self.max_sequence_length,"uncond_skip_layers":self.uncond_skip_layers,"slg":self.slg}).serialize(serializer)
    }
}

/// Training V2 grid: FP32 shift, nearest training sigma (ties choose lower), unique consecutive.
/// Returns sigmas including terminal zero; transformer times are sigmas[..len-1] * 1000.
pub fn moss_image_schedule(steps: u32, shift: f32, training_shift: f32) -> Result<Vec<f32>> {
    require(
        steps > 0
            && shift.is_finite()
            && shift > 0.0
            && training_shift.is_finite()
            && training_shift > 0.0,
        "invalid schedule steps/shifts",
    )?;
    fn grid(count: usize, shift: f32) -> impl DoubleEndedIterator<Item = f32> {
        let lower = shift / (1000.0 + shift - 1.0);
        (0..count).map(move |i| {
            // sc-26322: PyTorch 2.9.1 CPU linspace computes an FP32 step, anchoring
            // the second half at the end. FP64 interpolation can change the nearest grid index.
            let step = (lower - 1.0) / (count - 1).max(1) as f32;
            let v = if count == 1 {
                1.0
            } else if i < count / 2 {
                1.0 + step * i as f32
            } else {
                lower - step * (count - i - 1) as f32
            };
            shift * v / (1.0 + (shift - 1.0) * v)
        })
    }
    let training = grid(1000, training_shift).rev().collect::<Vec<_>>();
    require(
        training
            .iter()
            .all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0),
        "training shift produces a nonfinite/invalid grid",
    )?;
    let mut output = Vec::new();
    for sigma in grid(steps as usize, shift) {
        require(
            sigma.is_finite() && sigma > 0.0 && sigma <= 1.0,
            "inference shift produces a nonfinite/invalid grid",
        )?;
        let snapped = nearest_training_sigma(&training, sigma);
        if output.last() != Some(&snapped) {
            output.push(snapped);
        }
    }
    output.push(0.0);
    Ok(output)
}
/// Files whose contents the caller must pin before cache identity / provider loading. A repo/revision
/// label alone is not a content receipt. The sharded indexes must additionally bind every shard.
pub const SNAPSHOT_FILES: &[&str] = &[
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "LICENSE",
];

#[doc(hidden)]
pub fn nearest_training_sigma(training: &[f32], sigma: f32) -> f32 {
    let hi = training
        .partition_point(|value| *value < sigma)
        .min(training.len() - 1);
    let lo = hi.saturating_sub(1);
    if (sigma - training[lo]).abs() <= (training[hi] - sigma).abs() {
        training[lo]
    } else {
        training[hi]
    }
}
/// Backend providers implement this beside Generator; no placeholder executor is registered.
pub trait MossImageAdvancedGenerator {
    fn generate_moss_advanced(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        on_progress: &mut dyn FnMut(crate::Progress),
    ) -> Result<MossImageAdvancedOutput>;
}

#[cfg(test)]
mod execution_receipt_tests {
    use super::*;
    use crate::Progress;
    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    fn example(mode: MossImageAcceleration) -> MossImageExecutionData {
        let resolved = MossImageResolved::resolve(
            MossImageVariant::Sft,
            &GenerationRequest {
                moss_image: Some(MossImageOptions {
                    acceleration: mode,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let assets = receipt_asset_paths(mode)
            .into_keys()
            .map(|(role, path)| {
                let (repo, rev) = if matches!(
                    role,
                    MossImageAssetRole::DmdLora
                        | MossImageAssetRole::DmdTime
                        | MossImageAssetRole::DmdDataspace
                ) || (role == MossImageAssetRole::Transformer
                    && mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32))
                {
                    (DMD_REPO, DMD_REVISION)
                } else {
                    (SFT_REPO, SFT_REVISION)
                };
                MossImageAssetReceipt {
                    role,
                    repository: repo.into(),
                    revision: rev.into(),
                    relative_path: path,
                    sha256: HASH.into(),
                }
            })
            .collect();
        let sampling = if mode == MossImageAcceleration::Off {
            MossImageSamplingReceipt::Euler {
                sigmas: moss_image_schedule(
                    resolved.steps,
                    resolved.shift,
                    resolved.training_shift,
                )
                .unwrap(),
            }
        } else {
            MossImageSamplingReceipt::Dmd {
                step_codes: vec![1.; resolved.steps as usize],
                mix_matrix_sha256: HASH.into(),
            }
        };
        let input = MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: HASH.into(),
            sequence_length: 27,
            valid_tokens: 27,
        };
        let conditioning = vec![MossImageConditioningReceipt {
            positive: input.clone(),
            negative: (mode == MossImageAcceleration::Off).then_some(input),
        }];
        let mut component_dtypes: BTreeMap<String, MossImageDtype> = [
            ("transformer", MossImageDtype::Bf16),
            ("text_encoder", MossImageDtype::Bf16),
            ("vae", MossImageDtype::F32),
            ("latents", MossImageDtype::F32),
            ("time_embedding", MossImageDtype::Bf16),
            ("shared_block_modulation", MossImageDtype::Bf16),
            ("video_head_modulation", MossImageDtype::Bf16),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
        let mut adapter_reports = vec![];
        if let MossImageAcceleration::Dmd(policy) = mode {
            for key in [
                "time_embedding",
                "shared_block_modulation",
                "video_head_modulation",
            ] {
                component_dtypes.insert(key.into(), MossImageDtype::F32);
            }
            if policy == MossDmdPrecision::UnmergedBf16 {
                component_dtypes.insert("lora_residual".into(), MossImageDtype::F32);
                adapter_reports.push(crate::AdapterApplyReport {
                    adapter_path: "/fixture/lora.safetensors".into(),
                    applied: 252,
                    skipped: vec![],
                });
            } else {
                component_dtypes.insert("transformer".into(), MossImageDtype::F32);
            }
        }
        MossImageExecutionData {
            execution_device: MossImageExecutionDevice::Cuda,
            resolved,
            assets,
            sampling,
            samples: vec![MossImageSampleReceipt {
                seed: Some(173),
                noise_sha256: HASH.into(),
            }],
            conditioning,
            images_per_prompt: 1,
            latent_shape: [1, 128, 64, 64],
            output_kind: MossImageOutputKind::Images,
            component_dtypes,
            adapter_reports,
        }
    }
    #[test]
    fn author_fp32_nondefault_schedules_and_nearest_indices() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/nondefault_schedules.json"
        ))
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = moss_image_schedule(
                c["steps"].as_u64().unwrap() as u32,
                c["shift"].as_f64().unwrap() as f32,
                c["training_shift"].as_f64().unwrap() as f32,
            )
            .unwrap();
            let expected = c["sigmas"].as_array().unwrap();
            assert_eq!(got.len(), expected.len());
            for (i, (a, b)) in got.iter().zip(expected).enumerate() {
                let b = b.as_f64().unwrap() as f32;
                assert!((*a - b).abs() < 1.3e-7, "case={c}, step={i}, {a} != {b}");
            }
            // A whole training-grid point is ~1e-3, far outside the one-ULP scalar tolerance.
            if c["training_shift"] == 1.0 && c["steps"] == 7 {
                let training = moss_image_schedule(1000, 1.0, 1.0).unwrap();
                let mut sorted = training[..training.len() - 1].to_vec();
                sorted.reverse();
                let indices = got[..got.len() - 1]
                    .iter()
                    .map(|v| sorted.iter().position(|x| x == v).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(indices, [999, 833, 666, 500, 333, 166, 0]);
                assert_eq!(got[5], 0.167_000_01_f32);
            }
            assert_eq!(got[0], 1.0);
            assert_eq!(*got.last().unwrap(), 0.0);
        }
    }
    #[test]
    fn canonical_options_wire_is_strict_and_preserves_null_vs_empty_skips() {
        let defaults: MossImageOptions = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(defaults, MossImageOptions::default());
        let none: MossImageOptions =
            serde_json::from_value(serde_json::json!({"uncond_skip_layers":null})).unwrap();
        let empty: MossImageOptions =
            serde_json::from_value(serde_json::json!({"uncond_skip_layers":[]})).unwrap();
        assert_eq!(none.uncond_skip_layers, None);
        assert_eq!(empty.uncond_skip_layers, Some(vec![]));
        for (wire, mode) in [
            ("off", MossImageAcceleration::Off),
            (
                "unmerged_bf16",
                MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            ),
            (
                "full_fp32",
                MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
            ),
        ] {
            let options: MossImageOptions =
                serde_json::from_value(serde_json::json!({"acceleration":wire})).unwrap();
            assert_eq!(options.acceleration, mode);
            assert_eq!(
                serde_json::to_value(&options).unwrap()["acceleration"],
                wire
            );
            assert_eq!(
                serde_json::from_value::<MossImageOptions>(serde_json::to_value(&options).unwrap())
                    .unwrap(),
                options
            );
        }
        for bad in [
            serde_json::json!({"acceleration":true}),
            serde_json::json!({"acceleration":"dmd"}),
            serde_json::json!({"max_sequence_length":0}),
            serde_json::json!({"uncond_skip_layers":[42]}),
            serde_json::json!({"unknown":false}),
            serde_json::json!({"training_shift":-1}),
            serde_json::json!({"acceleration":"unmerged_bf16","training_shift":1}),
            serde_json::json!({"slg":{"layers":[1],"scale":3,"start":0,"stop":1,"unknown":0}}),
        ] {
            assert!(serde_json::from_value::<MossImageOptions>(bad).is_err());
        }
        let nonfinite = MossImageOptions {
            training_shift: Some(f32::NAN),
            ..Default::default()
        };
        assert!(serde_json::to_value(nonfinite).is_err());
    }
    #[test]
    fn receipt_round_trips_all_policies_and_refuses_crossed_execution_facts() {
        for mode in [
            MossImageAcceleration::Off,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        ] {
            let data = example(mode);
            let report = MossImageExecutionReport::new(data.clone()).unwrap();
            let json = serde_json::to_value(&report).unwrap();
            assert_eq!(
                serde_json::from_value::<MossImageExecutionReport>(json.clone()).unwrap(),
                report
            );
            let mut bad = json;
            bad["not_a_real_field"] = true.into();
            assert!(serde_json::from_value::<MossImageExecutionReport>(bad).is_err());
            let mutations: &[fn(&mut MossImageExecutionData)] = &[
                |d| {
                    d.assets.pop();
                },
                |d| {
                    d.assets[0].revision = "wrong".into();
                },
                |d| {
                    d.assets[0].sha256 = "bad".into();
                },
                |d| {
                    d.latent_shape[0] = 2;
                },
                |d| {
                    d.samples.clear();
                },
                |d| {
                    d.conditioning[0].positive.valid_tokens = 0;
                },
                |d| {
                    d.component_dtypes
                        .insert("vae".into(), MossImageDtype::Bf16);
                },
                |d| {
                    d.resolved.variant = MossImageVariant::Base;
                },
            ];
            for mutate in mutations {
                let mut d = data.clone();
                mutate(&mut d);
                assert!(MossImageExecutionReport::new(d).is_err());
            }
            let mut crossed = data.clone();
            crossed.sampling = match crossed.sampling {
                MossImageSamplingReceipt::Euler { .. } => MossImageSamplingReceipt::Dmd {
                    step_codes: vec![1.; 8],
                    mix_matrix_sha256: HASH.into(),
                },
                _ => MossImageSamplingReceipt::Euler {
                    sigmas: vec![1., 0.],
                },
            };
            assert!(MossImageExecutionReport::new(crossed).is_err());
        }
        let mut cpu = example(MossImageAcceleration::Off);
        cpu.execution_device = MossImageExecutionDevice::Cpu;
        assert!(MossImageExecutionReport::new(cpu.clone()).is_err());
        cpu.component_dtypes
            .insert("text_encoder".into(), MossImageDtype::F32);
        MossImageExecutionReport::new(cpu).unwrap();
        let mut mixed = example(MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16));
        mixed.adapter_reports[0]
            .skipped
            .push("layers.2.proj.q".into());
        assert!(MossImageExecutionReport::new(mixed).is_err());
        let mut bad = example(MossImageAcceleration::Off);
        if let MossImageSamplingReceipt::Euler { sigmas } = &mut bad.sampling {
            sigmas[1] = 0.5;
        }
        assert!(MossImageExecutionReport::new(bad).is_err());
    }
    #[test]
    fn asset_receipt_uses_approved_logical_identity_for_renamed_file_sc26325() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("content-addressed-cache.safetensors");
        std::fs::write(
            &file,
            include_bytes!("../tests/fixtures/moss_image/transformer_config.json"),
        )
        .unwrap();
        let pin = crate::PinnedWeightsFile::pin(&file).unwrap();
        let receipt = MossImageAssetReceipt::from_pin(
            MossImageAssetRole::Transformer,
            SFT_REPO,
            SFT_REVISION,
            "transformer/config.json",
            &pin,
        )
        .unwrap();
        assert_eq!(receipt.relative_path, "transformer/config.json");
        assert_eq!(receipt.sha256, hex_digest(pin.content_sha256()));
        assert!(!pin.loader_path().ends_with("transformer/config.json"));
        for (repo, rev, path) in [
            (SFT_REPO, "main", "transformer/config.json"),
            (SFT_REPO, DMD_REVISION, "transformer/config.json"),
            (DMD_REPO, DMD_REVISION, "dataspace.safetensors"),
        ] {
            assert!(MossImageAssetReceipt::from_pin(
                MossImageAssetRole::Transformer,
                repo,
                rev,
                path,
                &pin
            )
            .is_err());
        }
        let mut foreign =
            include_bytes!("../tests/fixtures/moss_image/transformer_config.json").to_vec();
        foreign[0] = b'X'; // Same size forces digest validation rather than a size-only rejection.
        std::fs::write(&file, &foreign).unwrap();
        assert!(MossImageAssetReceipt::from_pin(
            MossImageAssetRole::Transformer,
            SFT_REPO,
            SFT_REVISION,
            "transformer/config.json",
            &pin
        )
        .is_err());
        let bad = crate::PinnedWeightsFile::pin(&file).unwrap();
        assert!(MossImageAssetReceipt::from_pin(
            MossImageAssetRole::Transformer,
            SFT_REPO,
            SFT_REVISION,
            "transformer/config.json",
            &bad
        )
        .is_err());
    }
    #[test]
    fn advanced_dyn_entry_point_forwards_progress_and_typed_cancel() {
        struct Fake;
        impl MossImageAdvancedGenerator for Fake {
            fn generate_moss_advanced(
                &self,
                req: &GenerationRequest,
                inputs: MossImageAdvancedInputs<'_>,
                progress: &mut dyn FnMut(Progress),
            ) -> Result<MossImageAdvancedOutput> {
                let resolved = MossImageResolved::resolve(MossImageVariant::Sft, req)?;
                inputs.validate(req, &resolved)?;
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                progress(Progress::Step {
                    current: 1,
                    total: 1,
                });
                Ok(MossImageAdvancedOutput::Latents {
                    values: vec![0.; 128],
                    shape: [1, 128, 1, 1],
                })
            }
        }
        let fake: &dyn MossImageAdvancedGenerator = &Fake;
        let request = GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        };
        let mut steps = vec![];
        fake.generate_moss_advanced(
            &request,
            MossImageAdvancedInputs {
                output: MossImageOutputKind::Latents,
                ..Default::default()
            },
            &mut |p| {
                if let Progress::Step { current, total } = p {
                    steps.push((current, total))
                }
            },
        )
        .unwrap();
        assert_eq!(steps, [(1, 1)]);
        request.cancel.cancel();
        assert!(matches!(
            fake.generate_moss_advanced(
                &request,
                MossImageAdvancedInputs::default(),
                &mut |_| panic!("cancelled callback")
            ),
            Err(Error::Canceled)
        ));
    }
}

//! Frozen MOSS-Image contracts. No tensors, model acquisition, or backend execution.
//!
//! Source authority and precision policies are recorded in `docs/models/moss-image.md`.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    weightsmeta::{Dtype, TensorView},
    Error, GenerationRequest, Image, Result,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    /// None uses author default [22] for ordinary inference. Some(empty) disables skipping.
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

/// Finalized dense DiT configuration shared by Base and SFT; reject other model architectures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MossImageConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub latent_channels: usize,
}
impl Default for MossImageConfig {
    fn default() -> Self {
        Self {
            dim: 3072,
            ffn_dim: 8192,
            num_heads: 24,
            num_layers: 42,
            text_dim: 4096,
            freq_dim: 256,
            latent_channels: 128,
        }
    }
}
impl MossImageConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let expected = Self::default();
        for (key, number) in [
            ("dim", expected.dim),
            ("ffn_dim", expected.ffn_dim),
            ("num_heads", expected.num_heads),
            ("num_layers", expected.num_layers),
            ("text_dim", expected.text_dim),
            ("freq_dim", expected.freq_dim),
            ("latent_channels", expected.latent_channels),
        ] {
            require(
                value[key].as_u64() == Some(number as u64),
                &format!("unexpected transformer {key}"),
            )?;
        }
        require(
            value["rope_base"].as_f64() == Some(10000.0),
            "unexpected RoPE base",
        )?;
        Ok(expected)
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
            // torch.linspace returns the start for one element; compute fractions in f64 then cast.
            let v = if count == 1 {
                1.0
            } else {
                (1.0 + (lower as f64 - 1.0) * i as f64 / (count - 1) as f64) as f32
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
fn nearest_training_sigma(training: &[f32], sigma: f32) -> f32 {
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
/// Author time features: FP64 frequencies/phase, cosine half then sine half, cast once to FP32.
pub fn moss_time_features(timestep: f32) -> Result<[f32; 256]> {
    require(timestep.is_finite(), "nonfinite timestep")?;
    let mut values = [0.0; 256];
    for i in 0..128 {
        let phase = timestep as f64 * 10000_f64.powf(-(i as f64) / 128.0);
        values[i] = phase.cos() as f32;
        values[i + 128] = phase.sin() as f32;
    }
    Ok(values)
}
/// FP32 guidance, including the author CFG=1 SLG special case. Inputs must have equal lengths.
pub fn moss_guidance(
    conditional: &[f32],
    unconditional: Option<&[f32]>,
    skipped: Option<&[f32]>,
    cfg: f32,
    slg_scale: f32,
) -> Result<Vec<f32>> {
    require(
        cfg.is_finite() && cfg >= 0.0 && slg_scale.is_finite() && slg_scale >= 0.0,
        "invalid guidance",
    )?;
    if cfg != 1.0 {
        require(
            unconditional.is_some(),
            "CFG requires a negative prediction",
        )?;
    }
    for branch in [unconditional, skipped].into_iter().flatten() {
        require(
            branch.len() == conditional.len() && branch.iter().all(|v| v.is_finite()),
            "invalid prediction branch",
        )?;
    }
    require(
        conditional.iter().all(|v| v.is_finite()),
        "nonfinite prediction",
    )?;
    let output: Vec<f32> = conditional
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let base = if cfg != 1.0 {
                let u = unconditional.unwrap()[i];
                u + cfg * (c - u)
            } else if slg_scale > 0.0 {
                skipped.map_or(c, |s| s[i])
            } else {
                c
            };
            base + skipped.map_or(0.0, |s| slg_scale * (c - s[i]))
        })
        .collect();
    require(
        output.iter().all(|v| v.is_finite()),
        "nonfinite guided prediction",
    )?;
    Ok(output)
}

/// Checked CPU representation for native advanced entry points. Embeddings are converted to the
/// backend encoder dtype on upload; binary masks compact each sample before the DiT forward.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
/// Backend providers implement this beside Generator; no placeholder executor is registered.
pub trait MossImageAdvancedGenerator {
    fn generate_moss_advanced(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
    ) -> Result<MossImageAdvancedOutput>;
}

/// Exact frozen DMD tensor coverage. Views retain caller-owned payloads; validation never merges.
pub fn dmd_lora_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut expected = BTreeMap::new();
    for index in 0..42 {
        for name in DMD_PROJECTIONS {
            let input = if name == "ffn_down" { 8192 } else { 3072 };
            let output = if name == "ffn_up_gate" { 16384 } else { 3072 };
            let prefix = format!("diffusion_model.layers.{index}.proj.{name}");
            expected.insert(format!("{prefix}.lora_down.weight"), vec![128, input]);
            expected.insert(format!("{prefix}.lora_up.weight"), vec![output, 128]);
            expected.insert(format!("{prefix}.alpha"), vec![]);
        }
    }
    expected
}
pub fn dmd_replacement_shapes() -> BTreeMap<String, Vec<usize>> {
    [
        ("time_embedding.0.weight", vec![3072, 256]),
        ("time_embedding.0.bias", vec![3072]),
        ("time_embedding.2.weight", vec![3072, 3072]),
        ("time_embedding.2.bias", vec![3072]),
        ("shared_block_modulation.bias", vec![18432]),
        ("video_head_modulation.bias", vec![6144]),
    ]
    .into_iter()
    .map(|(k, s)| (k.into(), s))
    .collect()
}
pub fn validate_dmd_tensors(
    tensors: &BTreeMap<String, TensorView<'_>>,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    require(
        tensors.keys().collect::<BTreeSet<_>>() == expected.keys().collect::<BTreeSet<_>>(),
        "DMD tensor keys must be consumed exactly",
    )?;
    for (name, shape) in expected {
        let view = &tensors[name];
        let elements = shape.iter().try_fold(1usize, |n, v| n.checked_mul(*v));
        require(
            view.dtype == Dtype::F32
                && (view.shape == shape
                    || (name.ends_with(".alpha")
                        && view.shape.iter().try_fold(1usize, |n, v| n.checked_mul(*v))
                            == Some(1)))
                && elements.and_then(|n| n.checked_mul(4)) == Some(view.data.len()),
            &format!("DMD tensor shape/dtype mismatch: {name}"),
        )?;
        for bytes in view.data.chunks_exact(4) {
            let v = f32::from_le_bytes(bytes.try_into().unwrap());
            if !v.is_finite() || (name.ends_with(".alpha") && v != 128.0) {
                return Err(Error::Unsupported(format!(
                    "MOSS-Image: DMD tensor must be finite (alpha=128): {name}"
                )));
            }
        }
    }
    Ok(())
}
/// Validate all three unmerged files together, bound to the exact SFT parent and adapter package.
pub fn validate_dmd_package(
    parent_repo: &str,
    parent_revision: &str,
    package_repo: &str,
    package_revision: &str,
    lora: &BTreeMap<String, TensorView<'_>>,
    replacements: &BTreeMap<String, TensorView<'_>>,
    dataspace: &BTreeMap<String, TensorView<'_>>,
) -> Result<()> {
    MossImageVariant::Sft.validate_identity(parent_repo, parent_revision)?;
    require(
        package_repo == DMD_REPO && package_revision == DMD_REVISION,
        "DMD package identity mismatch",
    )?;
    validate_dmd_tensors(lora, &dmd_lora_shapes())?;
    validate_dmd_tensors(replacements, &dmd_replacement_shapes())?;
    validate_dmd_tensors(
        dataspace,
        &[
            ("latent_mix.Q".into(), vec![128, 128]),
            ("step_codes".into(), vec![8]),
        ]
        .into_iter()
        .collect(),
    )
}
/// Fixed-noise DMD input x=z+Q*x0. NCHW, Q indexed [output,input], all arithmetic FP32.
pub fn moss_dmd_input(
    noise: &[f32],
    clean: &[f32],
    q: &[f32],
    batch: usize,
    spatial: usize,
) -> Result<Vec<f32>> {
    let size = batch.checked_mul(128).and_then(|n| n.checked_mul(spatial));
    require(
        batch > 0
            && spatial > 0
            && size == Some(noise.len())
            && clean.len() == noise.len()
            && q.len() == 128 * 128,
        "DMD shape mismatch",
    )?;
    require(
        noise.iter().chain(clean).chain(q).all(|v| v.is_finite()),
        "DMD input must be finite FP32",
    )?;
    let mut input = noise.to_vec();
    for b in 0..batch {
        for out in 0..128 {
            for p in 0..spatial {
                let mut mixed = 0.0f32;
                for channel in 0..128 {
                    mixed += q[out * 128 + channel] * clean[(b * 128 + channel) * spatial + p];
                }
                input[(b * 128 + out) * spatial + p] += mixed;
            }
        }
    }
    require(
        input.iter().all(|v| v.is_finite()),
        "nonfinite DMD mixed input",
    )?;
    Ok(input)
}
/// Fixed-noise clean update x0=z-v; deliberately independent of the previous x0 and step size.
pub fn moss_dmd_update(noise: &[f32], velocity: &[f32]) -> Result<Vec<f32>> {
    require(
        noise.len() == velocity.len(),
        "DMD prediction shape mismatch",
    )?;
    let output: Vec<f32> = noise.iter().zip(velocity).map(|(z, v)| z - v).collect();
    require(
        noise
            .iter()
            .chain(velocity)
            .chain(&output)
            .all(|v| v.is_finite()),
        "nonfinite DMD update",
    )?;
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
pub const MOSS_VAE_DECODER_CHANNELS: [usize; 4] = [96, 192, 384, 384];
pub const MOSS_TEXT_ENCODER_PREFIX: &str = "language_model";
/// Optional package components required by this exact load mode. These must be included in ordinary
/// LoadSpec prepared identities; switching the mode requires a distinct load/cache entry.
pub fn moss_dmd_components(acceleration: MossImageAcceleration) -> &'static [&'static str] {
    match acceleration {
        MossImageAcceleration::Off => &[],
        MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => &[
            DMD_LORA_COMPONENT,
            DMD_TIME_COMPONENT,
            DMD_DATASPACE_COMPONENT,
        ],
        MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => {
            &[DMD_TRANSFORMER_COMPONENT, DMD_DATASPACE_COMPONENT]
        }
    }
}

/// Native scheduler configuration; incompatible diffusion schedules never fall back silently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MossImageSchedulerConfig {
    pub shift: f32,
    pub training_shift: f32,
}
impl Default for MossImageSchedulerConfig {
    fn default() -> Self {
        Self {
            shift: 5.0,
            training_shift: 1.0,
        }
    }
}
impl MossImageSchedulerConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        require(
            value["_class_name"] == "MOSSImageScheduler"
                && value["num_train_timesteps"].as_u64() == Some(1000),
            "expected the MOSS 1000-step training scheduler",
        )?;
        for flag in [
            "invert_sigmas",
            "stochastic_sampling",
            "use_dynamic_shifting",
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
        ] {
            require(
                value.get(flag).is_none_or(|v| v.as_bool() == Some(false)),
                "alternate scheduler flags are unsupported",
            )?;
        }
        let shift = value["shift"]
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| Error::Unsupported("MOSS scheduler shift missing".into()))?;
        let training_shift = value["training_shift"]
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| Error::Unsupported("MOSS training shift missing".into()))?;
        moss_image_schedule(1, shift, training_shift)?;
        Ok(Self {
            shift,
            training_shift,
        })
    }
}
pub fn moss_dmd_step_codes(codes: &[f32], steps: u32) -> Result<&[f32]> {
    require(
        codes.len() == 8 && codes.iter().all(|v| v.is_finite()) && (1..=8).contains(&steps),
        "DMD requires eight finite codes and a trained prefix in 1..=8",
    )?;
    Ok(&codes[..steps as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/moss_image/goldens.json")).unwrap()
    }
    fn dmd_request() -> GenerationRequest {
        GenerationRequest {
            moss_image: Some(MossImageOptions {
                acceleration: MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    #[test]
    fn defaults_are_ordinary_for_both_models_and_mode_transitions() {
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let r = MossImageResolved::resolve(variant, &GenerationRequest::default()).unwrap();
            assert_eq!(r.acceleration, MossImageAcceleration::Off);
            assert_eq!(
                (r.steps, r.guidance, r.shift, r.training_shift),
                (50, 3.0, 5.0, 1.0)
            );
            assert_eq!(r.uncond_skip_layers, [22]);
        }
        let mut req = dmd_request();
        assert_eq!(
            MossImageResolved::resolve(MossImageVariant::Sft, &req)
                .unwrap()
                .steps,
            8
        );
        req.moss_image = None;
        assert_eq!(
            MossImageResolved::resolve(MossImageVariant::Sft, &req)
                .unwrap()
                .steps,
            50
        );
        assert!(moss_dmd_components(MossImageAcceleration::Off).is_empty());
    }
    #[test]
    fn dmd_rejects_every_incompatible_control_and_accepts_both_published_policies() {
        assert!(MossImageResolved::resolve(MossImageVariant::Base, &dmd_request()).is_err());
        for policy in [MossDmdPrecision::UnmergedBf16, MossDmdPrecision::FullFp32] {
            for steps in 1..=8 {
                let mut req = dmd_request();
                req.steps = Some(steps);
                req.moss_image.as_mut().unwrap().acceleration = MossImageAcceleration::Dmd(policy);
                assert!(MossImageResolved::resolve(MossImageVariant::Sft, &req).is_ok());
            }
        }
        let mutations: &[fn(&mut GenerationRequest)] = &[
            |r| r.steps = Some(0),
            |r| r.steps = Some(9),
            |r| r.guidance = Some(1.0),
            |r| r.negative_prompt = Some("bad".into()),
            |r| r.sampler = Some("euler".into()),
            |r| r.scheduler = Some("moss_training_grid".into()),
            |r| r.scheduler_shift = Some(5.0),
            |r| r.true_cfg = Some(3.0),
            |r| r.guidance_method = Some("cfg".into()),
            |r| r.moss_image.as_mut().unwrap().training_shift = Some(1.0),
            |r| r.moss_image.as_mut().unwrap().uncond_skip_layers = Some(vec![]),
            |r| {
                r.moss_image.as_mut().unwrap().slg = Some(MossSkipLayerGuidance {
                    layers: vec![24],
                    scale: 3.0,
                    start: 0.0,
                    stop: 1.0,
                })
            },
        ];
        for mutate in mutations {
            let mut r = dmd_request();
            mutate(&mut r);
            assert!(
                MossImageResolved::resolve(MossImageVariant::Sft, &r).is_err(),
                "{r:?}"
            );
        }
    }
    #[test]
    fn ordinary_controls_validate_boundaries_and_nonfinite_floor() {
        let mut r = GenerationRequest {
            width: 1280,
            height: 768,
            guidance: Some(1.0),
            moss_image: Some(MossImageOptions {
                max_sequence_length: 1,
                uncond_skip_layers: Some(vec![]),
                slg: Some(MossSkipLayerGuidance {
                    layers: vec![0, 41],
                    scale: 3.0,
                    start: 0.25,
                    stop: 0.75,
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let resolved = MossImageResolved::resolve(MossImageVariant::Base, &r).unwrap();
        assert!(!resolved.skipping_active(0, 4));
        assert!(resolved.skipping_active(1, 4));
        assert!(!resolved.skipping_active(3, 4));
        r.moss_image.as_mut().unwrap().slg.as_mut().unwrap().scale = f32::NAN;
        assert_eq!(r.first_nonfinite_float().unwrap().0, "moss_image.slg.scale");
        assert!(MossImageResolved::resolve(MossImageVariant::Base, &r).is_err());
        r.moss_image = None;
        r.width = 1023;
        assert!(MossImageResolved::resolve(MossImageVariant::Base, &r).is_err());
        r.width = 1024;
        r.scheduler = Some("flux2".into());
        assert!(MossImageResolved::resolve(MossImageVariant::Base, &r).is_err());
    }
    #[test]
    fn frozen_config_and_asset_identities_cannot_select_a_sibling() {
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/transformer_config.json"
        ))
        .unwrap();
        assert_eq!(
            MossImageConfig::from_json(&config).unwrap(),
            MossImageConfig::default()
        );
        let mut changed = config;
        changed["num_layers"] = 41.into();
        assert!(MossImageConfig::from_json(&changed).is_err());
        assert!(MossImageVariant::Sft
            .validate_identity(BASE_REPO, BASE_REVISION)
            .is_err());
        assert!(MossImageVariant::Sft
            .validate_identity(SFT_REPO, SFT_REVISION)
            .is_ok());
        let vae: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/moss_image/vae_config.json"))
                .unwrap();
        assert_eq!(
            vae["decoder_block_out_channels"],
            serde_json::json!(MOSS_VAE_DECODER_CHANNELS)
        );
        let encoder: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/text_encoder_config.json"
        ))
        .unwrap();
        assert_eq!(encoder["architectures"][0], "Qwen3_5Model");
        assert_eq!(encoder["text_config"]["hidden_size"], 4096);
    }
    #[test]
    fn author_schedule_and_f64_time_goldens() {
        let fixture = fixture();
        let expected = fixture["author_schedule_50"]["sigmas"].as_array().unwrap();
        let actual = moss_image_schedule(50, 5.0, 1.0).unwrap();
        assert_eq!(actual.len(), expected.len());
        // The recorded torch float32 linspace backend can vary by one ULP; snap indices cannot.
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (*a - e.as_f64().unwrap() as f32).abs() < 1.3e-7,
                "{a} != {e}"
            );
        }
        assert_eq!(nearest_training_sigma(&[0.0, 0.5, 1.0], 0.25), 0.0);
        assert_eq!(nearest_training_sigma(&[0.0, 0.5, 1.0], 0.75), 0.5);
        assert_eq!(moss_image_schedule(1, 5.0, 1.0).unwrap(), [1.0, 0.0]);
        let many = moss_image_schedule(2000, 5.0, 1.0).unwrap();
        assert!(many.len() <= 1001 && many.windows(2).all(|p| p[0] > p[1]));
        for args in [
            (0, 5., 1.),
            (50, 0., 1.),
            (50, f32::NAN, 1.),
            (50, 5., f32::INFINITY),
        ] {
            assert!(moss_image_schedule(args.0, args.1, args.2).is_err());
        }
        let actual = moss_time_features(987.0).unwrap();
        for (a, e) in actual
            .iter()
            .zip(fixture["time_features"]["values"].as_array().unwrap())
        {
            assert!((*a - e.as_f64().unwrap() as f32).abs() < 2e-7);
        }
        assert!(moss_time_features(f32::NAN).is_err());
    }
    #[test]
    fn guidance_pins_cfg_one_slg_and_normal_cfg() {
        assert_eq!(
            moss_guidance(&[4.], Some(&[1.]), Some(&[2.]), 3., 2.).unwrap(),
            [14.]
        );
        assert_eq!(
            moss_guidance(&[4.], None, Some(&[2.]), 1., 3.).unwrap(),
            [8.]
        );
        assert_eq!(moss_guidance(&[4.], None, None, 1., 0.).unwrap(), [4.]);
        assert!(moss_guidance(&[4.], None, None, 3., 0.).is_err());
        assert!(moss_guidance(&[f32::NAN], None, None, 1., 0.).is_err());
    }
    #[test]
    fn nonsymmetric_mix_and_fixed_noise_recurrence_match_hand_fixture() {
        let f = fixture();
        let mut z = vec![0.; 256];
        z[0] = 2.;
        z[2] = -1.; // spatial=2, channel offsets 0 and2
        let mut q = vec![0.; 128 * 128];
        q[0] = 0.5;
        q[1] = 2.;
        q[128] = -1.;
        q[129] = 0.25;
        let mut clean = vec![0.; 256];
        for (i, code) in [0.125, 0.75].into_iter().enumerate() {
            let x = moss_dmd_input(&z, &clean, &q, 1, 2).unwrap();
            assert_eq!(x[0], f["dmd"]["inputs"][i][0].as_f64().unwrap() as f32);
            assert_eq!(x[2], f["dmd"]["inputs"][i][1].as_f64().unwrap() as f32);
            let v = x.iter().map(|x| x * code).collect::<Vec<_>>();
            clean = moss_dmd_update(&z, &v).unwrap();
            assert_eq!(clean[0], f["dmd"]["clean"][i][0].as_f64().unwrap() as f32);
            assert_eq!(clean[2], f["dmd"]["clean"][i][1].as_f64().unwrap() as f32);
        }
        assert!(moss_dmd_input(&z, &clean, &q, 2, 2).is_err());
        q[0] = f32::NAN;
        assert!(moss_dmd_input(&z, &clean, &q, 1, 2).is_err());
        assert!(moss_dmd_update(&[f32::INFINITY], &[1.]).is_err());
    }
    #[test]
    fn strict_dmd_tensor_coverage_shapes_values_and_replacements() {
        let shapes = dmd_lora_shapes();
        assert_eq!(shapes.len(), 42 * 6 * 3);
        assert_eq!(
            shapes["diffusion_model.layers.41.proj.ffn_up_gate.lora_up.weight"],
            [16384, 128]
        );
        assert_eq!(
            shapes["diffusion_model.layers.41.proj.ffn_down.lora_down.weight"],
            [128, 8192]
        );
        let replacements = dmd_replacement_shapes();
        assert_eq!(replacements.len(), 6);
        assert_eq!(replacements["shared_block_modulation.bias"], [18432]);
        assert_eq!(replacements["video_head_modulation.bias"], [6144]);
        let bytes = 128f32.to_le_bytes();
        let expected = [("x.alpha".into(), vec![])].into_iter().collect();
        let mut views = BTreeMap::new();
        views.insert(
            "x.alpha".into(),
            TensorView {
                dtype: Dtype::F32,
                shape: &[],
                data: &bytes,
            },
        );
        validate_dmd_tensors(&views, &expected).unwrap();
        views.insert(
            "extra".into(),
            TensorView {
                dtype: Dtype::F32,
                shape: &[],
                data: &bytes,
            },
        );
        assert!(validate_dmd_tensors(&views, &expected).is_err());
        views.remove("extra");
        let nan = f32::NAN.to_le_bytes();
        views.insert(
            "x.alpha".into(),
            TensorView {
                dtype: Dtype::F32,
                shape: &[],
                data: &nan,
            },
        );
        assert!(validate_dmd_tensors(&views, &expected).is_err());
        views.insert(
            "x.alpha".into(),
            TensorView {
                dtype: Dtype::BF16,
                shape: &[],
                data: &bytes,
            },
        );
        assert!(validate_dmd_tensors(&views, &expected).is_err());
        assert!(validate_dmd_package(
            BASE_REPO,
            BASE_REVISION,
            DMD_REPO,
            DMD_REVISION,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new()
        )
        .is_err());
    }
    #[test]
    fn advanced_embeddings_masks_batch_supplied_noise_and_latent_output() {
        let values = vec![0.; 2 * 3 * 4096];
        let mask = [1, 1, 0, 1, 0, 0];
        let mut r = GenerationRequest {
            count: 2,
            width: 16,
            height: 32,
            ..Default::default()
        };
        let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &r).unwrap();
        let noise = vec![0.; 4 * 128 * 2];
        let mut inputs = MossImageAdvancedInputs {
            positive: Some(MossImageEmbeddings {
                values: &values,
                batch: 2,
                sequence: 3,
                mask: Some(&mask),
            }),
            noise: Some(&noise),
            output: MossImageOutputKind::Latents,
            ..Default::default()
        };
        assert_eq!(inputs.validate(&r, &resolved).unwrap(), 4);
        let bad_mask = [1, 1, 0, 0, 0, 0];
        inputs.positive.as_mut().unwrap().mask = Some(&bad_mask);
        assert!(inputs.validate(&r, &resolved).is_err());
        let bad_mask = [1, 2, 0, 1, 0, 0];
        inputs.positive.as_mut().unwrap().mask = Some(&bad_mask);
        assert!(inputs.validate(&r, &resolved).is_err());
        inputs.positive.as_mut().unwrap().mask = Some(&mask);
        r.prompt = "text conflicts".into();
        assert!(inputs.validate(&r, &resolved).is_err());
        r.prompt.clear();
        inputs.noise = Some(&noise[..noise.len() - 1]);
        assert!(inputs.validate(&r, &resolved).is_err());
    }
}

#[cfg(test)]
mod package_tests {
    use super::*;
    #[test]
    fn source_fixture_hashes_scheduler_and_step_codes_are_pinned() {
        use sha2::{Digest, Sha256};
        let gold: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/moss_image/goldens.json"))
                .unwrap();
        for (path, bytes) in [
            (
                "transformer/config.json",
                include_bytes!("../tests/fixtures/moss_image/transformer_config.json").as_slice(),
            ),
            (
                "scheduler/scheduler_config.json",
                include_bytes!("../tests/fixtures/moss_image/scheduler_scheduler_config.json")
                    .as_slice(),
            ),
            (
                "text_encoder/config.json",
                include_bytes!("../tests/fixtures/moss_image/text_encoder_config.json").as_slice(),
            ),
            (
                "vae/config.json",
                include_bytes!("../tests/fixtures/moss_image/vae_config.json").as_slice(),
            ),
        ] {
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                gold["source"]["small_file_sha256"][path].as_str().unwrap()
            );
        }
        let mut config: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/scheduler_scheduler_config.json"
        ))
        .unwrap();
        assert_eq!(
            MossImageSchedulerConfig::from_json(&config).unwrap(),
            MossImageSchedulerConfig::default()
        );
        config["use_dynamic_shifting"] = true.into();
        assert!(MossImageSchedulerConfig::from_json(&config).is_err());
        assert_eq!(
            moss_dmd_step_codes(&[0., 1., 2., 3., 4., 5., 6., 7.], 3).unwrap(),
            [0., 1., 2.]
        );
        assert!(moss_dmd_step_codes(&[0.; 7], 3).is_err());
        assert!(moss_dmd_step_codes(&[f32::NAN; 8], 3).is_err());
        assert!(moss_dmd_step_codes(&[0.; 8], 9).is_err());
    }
    #[test]
    fn entire_unmerged_package_accepts_exact_coverage_and_detects_missing_bias() {
        let lora_shapes = dmd_lora_shapes();
        let replacement_shapes = dmd_replacement_shapes();
        let dataspace_shapes: BTreeMap<String, Vec<usize>> = [
            ("latent_mix.Q".into(), vec![128, 128]),
            ("step_codes".into(), vec![8]),
        ]
        .into_iter()
        .collect();
        // Reuse one zero payload across independent views rather than allocating model weights.
        let zeros = vec![0u8; 3072 * 3072 * 4];
        let alpha = 128f32.to_le_bytes();
        fn views<'a>(
            shapes: &'a BTreeMap<String, Vec<usize>>,
            zeros: &'a [u8],
            alpha: &'a [u8],
        ) -> BTreeMap<String, TensorView<'a>> {
            shapes
                .iter()
                .map(|(key, shape)| {
                    let bytes = shape.iter().product::<usize>() * 4;
                    (
                        key.clone(),
                        TensorView {
                            dtype: Dtype::F32,
                            shape: shape.as_slice(),
                            data: if key.ends_with(".alpha") {
                                alpha
                            } else {
                                &zeros[..bytes]
                            },
                        },
                    )
                })
                .collect()
        }
        let lora = views(&lora_shapes, &zeros, &alpha);
        let mut replacements = views(&replacement_shapes, &zeros, &alpha);
        let ds = views(&dataspace_shapes, &zeros, &alpha);
        validate_dmd_package(
            SFT_REPO,
            SFT_REVISION,
            DMD_REPO,
            DMD_REVISION,
            &lora,
            &replacements,
            &ds,
        )
        .unwrap();
        replacements.remove("video_head_modulation.bias");
        assert!(validate_dmd_package(
            SFT_REPO,
            SFT_REVISION,
            DMD_REPO,
            DMD_REVISION,
            &lora,
            &replacements,
            &ds
        )
        .is_err());
        assert!(validate_dmd_package(
            SFT_REPO,
            SFT_REVISION,
            DMD_REPO,
            "wrong-revision",
            &lora,
            &replacements,
            &ds
        )
        .is_err());
    }
    #[test]
    fn controls_are_refused_by_other_provider_routes_on_shared_floor() {
        let caps = crate::Capabilities {
            max_count: 8,
            max_size: 2048,
            ..Default::default()
        };
        let request = GenerationRequest {
            moss_image: Some(MossImageOptions::default()),
            ..Default::default()
        };
        assert!(caps
            .validate_request("flux2_klein_4b", &request)
            .unwrap_err()
            .to_string()
            .contains("MOSS controls"));
    }
}

//! Portable MOSS-Image option resolution. This crate owns the native and browser policy rules;
//! it has no generator, tensor, filesystem, or host dependencies.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MossPolicyError(&'static str);
impl MossPolicyError {
    pub fn message(self) -> &'static str {
        self.0
    }
}
impl std::fmt::Display for MossPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MOSS-Image: {}", self.0)
    }
}
impl std::error::Error for MossPolicyError {}
type Result<T> = std::result::Result<T, MossPolicyError>;

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(MossPolicyError(message))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageVariant {
    Base,
    Sft,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossDmdPrecision {
    #[default]
    UnmergedBf16,
    FullFp32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MossImageAcceleration {
    #[default]
    Off,
    Dmd(MossDmdPrecision),
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

#[derive(Clone, Debug, PartialEq)]
pub struct MossSkipLayerGuidance {
    pub layers: Vec<u32>,
    pub scale: f32,
    pub start: f32,
    pub stop: f32,
}
impl MossSkipLayerGuidance {
    pub fn validate(&self) -> Result<()> {
        validate_layers(&self.layers)?;
        require(
            self.scale.is_finite()
                && self.scale >= 0.0
                && self.start.is_finite()
                && self.stop.is_finite()
                && 0.0 <= self.start
                && self.start <= self.stop
                && self.stop <= 1.0,
            "SLG requires nonnegative scale and 0 <= start <= stop <= 1",
        )
    }
}
impl Serialize for MossSkipLayerGuidance {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        // Preserve the native serializer's resolver order: nonfinite MOSS controls precede
        // invalid skip layers, then SLG range checks.
        MossImageResolved::resolve(
            MossImageVariant::Sft,
            &MossPolicyRequest {
                moss_image: Some(MossImageOptions {
                    slg: Some(self.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .map_err(serde::ser::Error::custom)?;
        #[derive(Serialize)]
        struct Wire<'a> {
            layers: &'a [u32],
            scale: f32,
            start: f32,
            stop: f32,
        }
        Wire {
            layers: &self.layers,
            scale: self.scale,
            start: self.start,
            stop: self.stop,
        }
        .serialize(serializer)
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
        let wire = Wire::deserialize(deserializer)?;
        let value = Self {
            layers: wire.layers,
            scale: wire.scale,
            start: wire.start,
            stop: wire.stop,
        };
        value
            .validate()
            .map_err(|_| serde::de::Error::custom("invalid MOSS SLG controls"))?;
        Ok(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MossImageOptions {
    pub acceleration: MossImageAcceleration,
    pub max_sequence_length: u32,
    pub uncond_skip_layers: Option<Vec<u32>>,
    pub slg: Option<MossSkipLayerGuidance>,
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
        if let Some(x) = self.training_shift.filter(|x| !x.is_finite()) {
            return Some(("moss_image.training_shift", x));
        }
        if let Some(slg) = &self.slg {
            for (name, value) in [
                ("moss_image.slg.scale", slg.scale),
                ("moss_image.slg.start", slg.start),
                ("moss_image.slg.stop", slg.stop),
            ] {
                if !value.is_finite() {
                    return Some((name, value));
                }
            }
        }
        None
    }
    pub fn validate(&self) -> Result<()> {
        require(
            self.first_nonfinite_float().is_none(),
            "nonfinite MOSS control",
        )?;
        require(
            (1..=4096).contains(&self.max_sequence_length),
            "context limit must be in 1..=4096",
        )?;
        if let Some(layers) = &self.uncond_skip_layers {
            validate_layers(layers)?;
        }
        if let Some(slg) = &self.slg {
            slg.validate()?;
        }
        Ok(())
    }
}
impl Serialize for MossImageOptions {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        MossImageResolved::resolve(
            MossImageVariant::Sft,
            &MossPolicyRequest {
                moss_image: Some(self.clone()),
                ..Default::default()
            },
        )
        .map_err(serde::ser::Error::custom)?;
        #[derive(Serialize)]
        struct Wire<'a> {
            acceleration: MossImageAcceleration,
            max_sequence_length: u32,
            uncond_skip_layers: &'a Option<Vec<u32>>,
            slg: &'a Option<MossSkipLayerGuidance>,
            training_shift: Option<f32>,
        }
        Wire {
            acceleration: self.acceleration,
            max_sequence_length: self.max_sequence_length,
            uncond_skip_layers: &self.uncond_skip_layers,
            slg: &self.slg,
            training_shift: self.training_shift,
        }
        .serialize(serializer)
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
        let wire = Wire::deserialize(deserializer)?;
        let value = Self {
            acceleration: wire.acceleration,
            max_sequence_length: wire.max_sequence_length,
            uncond_skip_layers: wire.uncond_skip_layers,
            slg: wire.slg,
            training_shift: wire.training_shift,
        };
        MossImageResolved::resolve(
            MossImageVariant::Sft,
            &MossPolicyRequest {
                moss_image: Some(value.clone()),
                ..Default::default()
            },
        )
        .map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

/// The MOSS fields of a generation request. Presence flags represent non-MOSS carriers rejected
/// by this family; a native adapter checks its full request's other float controls first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MossPolicyRequest {
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub scheduler_shift: Option<f32>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub guidance_method: Option<String>,
    pub guidance_eta: Option<f32>,
    pub guidance_momentum: Option<f32>,
    pub guidance_norm_threshold: Option<f32>,
    pub true_cfg: Option<f32>,
    pub timestep_to_start_cfg: Option<u32>,
    pub cfg_interval: Option<(f32, f32)>,
    pub phases_present: bool,
    pub conditioning_present: bool,
    pub negative_prompt: Option<String>,
    pub moss_image: Option<MossImageOptions>,
}
impl Default for MossPolicyRequest {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            count: 1,
            steps: None,
            guidance: None,
            scheduler_shift: None,
            sampler: None,
            scheduler: None,
            guidance_method: None,
            guidance_eta: None,
            guidance_momentum: None,
            guidance_norm_threshold: None,
            true_cfg: None,
            timestep_to_start_cfg: None,
            cfg_interval: None,
            phases_present: false,
            conditioning_present: false,
            negative_prompt: None,
            moss_image: None,
        }
    }
}
impl MossPolicyRequest {
    fn has_nonfinite_float(&self) -> bool {
        [
            self.guidance,
            self.scheduler_shift,
            self.guidance_eta,
            self.guidance_momentum,
            self.guidance_norm_threshold,
            self.true_cfg,
            self.cfg_interval.map(|v| v.0),
            self.cfg_interval.map(|v| v.1),
        ]
        .into_iter()
        .flatten()
        .any(|v| !v.is_finite())
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
    pub fn resolve(variant: MossImageVariant, request: &MossPolicyRequest) -> Result<Self> {
        let options = request.moss_image.clone().unwrap_or_default();
        require(
            options.first_nonfinite_float().is_none(),
            "nonfinite MOSS control",
        )?;
        require(!request.has_nonfinite_float(), "nonfinite request control")?;
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
            !request.conditioning_present,
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
                && !request.phases_present,
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
            slg.validate()?;
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

    pub fn validate(&self) -> Result<()> {
        let ordinary = matches!(self.acceleration, MossImageAcceleration::Off);
        let expected = Self::resolve(
            self.variant,
            &MossPolicyRequest {
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
impl Serialize for MossImageResolved {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        #[derive(Serialize)]
        struct Wire<'a> {
            variant: MossImageVariant,
            acceleration: MossImageAcceleration,
            steps: u32,
            guidance: f32,
            shift: f32,
            training_shift: f32,
            max_sequence_length: u32,
            uncond_skip_layers: &'a [u32],
            slg: &'a Option<MossSkipLayerGuidance>,
        }
        Wire {
            variant: self.variant,
            acceleration: self.acceleration,
            steps: self.steps,
            guidance: self.guidance,
            shift: self.shift,
            training_shift: self.training_shift,
            max_sequence_length: self.max_sequence_length,
            uncond_skip_layers: &self.uncond_skip_layers,
            slg: &self.slg,
        }
        .serialize(serializer)
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
        let wire = Wire::deserialize(deserializer)?;
        let value = Self {
            variant: wire.variant,
            acceleration: wire.acceleration,
            steps: wire.steps,
            guidance: wire.guidance,
            shift: wire.shift,
            training_shift: wire.training_shift,
            max_sequence_length: wire.max_sequence_length,
            uncond_skip_layers: wire.uncond_skip_layers,
            slg: wire.slg,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

fn validate_layers(layers: &[u32]) -> Result<()> {
    require(
        layers.iter().all(|i| *i < 42),
        "skip layers must be zero-based indices in 0..42",
    )
}

/// Apply the author's edit defaults to a copy of the request. Other request fields remain intact.
pub fn apply_edit_defaults(request: &MossPolicyRequest) -> Result<MossPolicyRequest> {
    let mut request = request.clone();
    let options = request.moss_image.get_or_insert_with(Default::default);
    require(
        options.acceleration == MossImageAcceleration::Off
            && options.slg.is_none()
            && options
                .uncond_skip_layers
                .as_ref()
                .is_none_or(Vec::is_empty),
        "MOSS edit requires ordinary Base/SFT, full-layer CFG, and no SLG; edit+DMD is unvalidated",
    )?;
    require(
        !request.conditioning_present,
        "MOSS edit references must be supplied once through typed edit inputs",
    )?;
    options.uncond_skip_layers = Some(vec![]);
    request.steps.get_or_insert(30);
    request.guidance.get_or_insert(4.0);
    request.scheduler_shift.get_or_insert(5.0);
    Ok(request)
}

//! Shared FLUX3 training policy, pinned to diffusers' DreamBooth LoRA scripts.
//!
//! This is NOT the inference flow-map schedule. Both native trainers consume these
//! scalar decisions while keeping all differentiable tensor operations on their backend.
//! The reference is inference-independent: captions are dropped per instance, not
//! duplicated into the inference base checkpoint's two CFG conditioning rows.
use std::collections::{BTreeMap, BTreeSet};

use super::{adapters::native_target, DitConfig};
use crate::{Error, Result};

/// Apache-2.0 training source, distinct from the inference-only BFL code pin.
pub const TRAINING_REFERENCE_REVISION: &str = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941";
pub const TRAINING_REFERENCE_REPO: &str = "huggingface/diffusers";
pub const TRAINING_REFERENCE_SCRIPTS: [&str; 2] = [
    "examples/dreambooth/train_dreambooth_lora_flux3_klein.py",
    "examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py",
];
/// `FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT, use_dynamic_shifting=False)`.
pub const FLOW_SHIFT: f32 = 6.93;
pub const TRAIN_TIMESTEPS: usize = 1000;
/// Source PEFT defaults, not an implicit gain fallback when loading external adapters.
pub const DEFAULT_RANK: u32 = 4;
pub const DEFAULT_ALPHA: f32 = 4.0;
/// Every default PEFT target (mode-specific AND shared blocks).
pub const DEFAULT_TARGETS: [&str; 6] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.linear_in",
    "ff.linear_out",
];

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 training: {detail}"))
}

/// Actual descending discrete scheduler entry. Index 999 is positive, not zero.
pub fn training_sigma(index: usize) -> Result<f32> {
    if index >= TRAIN_TIMESTEPS {
        return Err(invalid("timestep index must be in 0..1000"));
    }
    let sigma = (TRAIN_TIMESTEPS - index) as f32 / TRAIN_TIMESTEPS as f32;
    Ok(FLOW_SHIFT * sigma / (1.0 + (FLOW_SHIFT - 1.0) * sigma))
}

/// The source converts `u * num_train_timesteps` to an integer, without rounding.
pub fn density_index(u: f32) -> Result<usize> {
    if !u.is_finite() || !(0.0..1.0).contains(&u) {
        return Err(invalid("timestep density must be finite in [0,1)"));
    }
    let index = (u * TRAIN_TIMESTEPS as f32) as usize;
    if index >= TRAIN_TIMESTEPS {
        return Err(invalid("timestep density rounded beyond schedule"));
    }
    Ok(index)
}

/// Upstream couples a density selection and a loss weighting under this one option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    None,
    SigmaSqrt,
    LogitNormal,
    Mode,
    Cosmap,
}

impl Weighting {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "sigma_sqrt" => Ok(Self::SigmaSqrt),
            "logit_normal" => Ok(Self::LogitNormal),
            "mode" => Ok(Self::Mode),
            "cosmap" => Ok(Self::Cosmap),
            _ => Err(invalid(format!("unknown weighting_scheme `{name}`"))),
        }
    }

    /// Consume an independently drawn uniform `[0,1)` or standard normal value.
    /// RNG ownership belongs to the trainer so its cursor can be checkpointed.
    pub fn density(
        self,
        uniform: f32,
        normal: f32,
        mean: f32,
        std: f32,
        mode_scale: f32,
    ) -> Result<f32> {
        let u = match self {
            Self::LogitNormal => {
                if !normal.is_finite() || !mean.is_finite() || !std.is_finite() || std < 0.0 {
                    return Err(invalid(
                        "logit_normal requires finite normal/mean and nonnegative std",
                    ));
                }
                1.0 / (1.0 + (-(mean + std * normal)).exp())
            }
            Self::Mode => {
                density_index(uniform)?;
                if !mode_scale.is_finite() {
                    return Err(invalid("mode_scale must be finite"));
                }
                1.0 - uniform
                    - mode_scale
                        * ((std::f32::consts::PI * uniform / 2.0).cos().powi(2) - 1.0 + uniform)
            }
            _ => {
                density_index(uniform)?;
                uniform
            }
        };
        density_index(u)?;
        Ok(u)
    }

    /// `compute_loss_weighting_for_sd3`. Use this on f32 loss, never on latent inputs.
    pub fn loss_weight(self, sigma: f32) -> Result<f32> {
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) || sigma == 0.0 {
            return Err(invalid("training sigma must be finite in (0,1]"));
        }
        Ok(match self {
            Self::SigmaSqrt => sigma.powf(-2.0),
            Self::Cosmap => {
                2.0 / (std::f32::consts::PI * (1.0 - 2.0 * sigma + 2.0 * sigma.powi(2)))
            }
            _ => 1.0,
        })
    }
}

/// The empty prompt must be encoded at the ORIGINAL caption's window length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionChoice {
    Original,
    EmptyAtWindow(usize),
}

pub fn caption_choice(
    draw: f32,
    probability: f32,
    prior: bool,
    window: usize,
) -> Result<CaptionChoice> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) || window == 0 {
        return Err(invalid(
            "caption dropout requires probability in [0,1] and nonempty window",
        ));
    }
    if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return Err(invalid("caption dropout draw must be in [0,1)"));
    }
    Ok(if !prior && draw < probability {
        CaptionChoice::EmptyAtWindow(window)
    } else {
        CaptionChoice::Original
    })
}

/// An explicit target mapping reused for factor initialization, checkpoint validation,
/// and diffusers adapter export. Shapes refer to base `[output,input]` projections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraTarget {
    pub native: String,
    pub diffusers: String,
    pub output: usize,
    pub input: usize,
}

/// Expand ONLY the source-supported suffixes across all mode/shared blocks. Use S6's
/// generation mapping as the authority, not a second fuzzy name-conversion implementation.
pub fn lora_targets(cfg: &DitConfig, requested: &[String]) -> Result<Vec<LoraTarget>> {
    if cfg.flow_map {
        return Err(invalid(
            "adaptation requires the undistilled base checkpoint, not a flow map",
        ));
    }
    let selected: Vec<&str> = if requested.is_empty() {
        DEFAULT_TARGETS.to_vec()
    } else {
        requested.iter().map(String::as_str).collect()
    };
    let mut unique = BTreeSet::new();
    for suffix in &selected {
        if !DEFAULT_TARGETS.contains(suffix) || !unique.insert(*suffix) {
            return Err(invalid(format!(
                "unknown or repeated LoRA target `{suffix}`"
            )));
        }
    }
    let mut parents = Vec::new();
    for mode in ["image", "image_cond"] {
        for block in 0..cfg.depth {
            parents.push(format!("mode_blocks.{mode}.{block}"));
        }
    }
    for block in 0..cfg.depth {
        parents.push(format!("txt_mode_blocks.{block}"));
    }
    for block in 0..cfg.depth_single {
        parents.push(format!("single_transformer_blocks.{block}"));
    }
    let mut targets = Vec::new();
    for parent in parents {
        for suffix in &selected {
            let leaf = if suffix.starts_with("ff.") {
                suffix.to_string()
            } else {
                format!("attn.{suffix}")
            };
            let diffusers = format!("transformer.{parent}.{leaf}");
            let (native, output, input) = native_target(&diffusers, cfg)?;
            targets.push(LoraTarget {
                native,
                diffusers,
                output,
                input,
            });
        }
    }
    if targets.is_empty() {
        return Err(invalid("base config has no trainable blocks"));
    }
    Ok(targets)
}

/// A trajectory identity, separate from a checkpoint directory name. The caller supplies
/// observed artifact digests, NEVER Hub metadata as a downloaded-file attestation.
/// `configuration` must contain the complete normalized training configuration, including
/// optimizer/scheduler, rank/alpha, scaling, precision/backend and RNG algorithm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumeIdentity {
    pub base_repo: String,
    pub base_revision: String,
    pub artifacts: BTreeMap<String, String>,
    pub dataset: String,
    pub preprocessing: String,
    pub configuration: BTreeMap<String, String>,
    pub factors: BTreeMap<String, Vec<usize>>,
}

impl ResumeIdentity {
    pub fn validate(&self) -> Result<()> {
        let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
        if self.base_repo.is_empty() || !hex(&self.base_revision, 40) {
            return Err(invalid(
                "resume base_repo/base_revision is missing or malformed",
            ));
        }
        if self.artifacts.is_empty()
            || self
                .artifacts
                .iter()
                .any(|(k, v)| k.is_empty() || !hex(v, 64))
        {
            return Err(invalid("resume artifacts require actual SHA256 identities"));
        }
        if !hex(&self.dataset, 64) || !hex(&self.preprocessing, 64) {
            return Err(invalid(
                "resume dataset/preprocessing identity is missing or malformed",
            ));
        }
        if self.factors.is_empty()
            || self
                .factors
                .iter()
                .any(|(k, v)| k.is_empty() || v.len() != 2 || v.contains(&0))
        {
            return Err(invalid(
                "resume factors require an exact nonempty key/shape table",
            ));
        }
        for key in [
            "optimizer",
            "learning_rate",
            "rank",
            "alpha",
            "scaling",
            "precision",
            "backend",
            "scheduler",
            "accumulation",
            "rng",
        ] {
            if self.configuration.get(key).is_none_or(|v| v.is_empty()) {
                return Err(invalid(format!("resume configuration is missing `{key}`")));
            }
        }
        Ok(())
    }

    /// Fail BEFORE allocating optimizer/training tensors. Never reset on mismatch.
    pub fn ensure_matches(&self, saved: &Self) -> Result<()> {
        self.validate()?;
        saved.validate()?;
        macro_rules! same {
            ($field:ident) => {
                if self.$field != saved.$field {
                    return Err(invalid(concat!(
                        "resume ",
                        stringify!($field),
                        " mismatch; use the original run inputs or start a new output directory"
                    )));
                }
            };
        }
        same!(base_repo);
        same!(base_revision);
        same!(artifacts);
        same!(dataset);
        same!(preprocessing);
        same!(configuration);
        same!(factors);
        Ok(())
    }

    /// Canonical, versioned metadata for the existing native checkpoint container.
    pub fn to_metadata(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::json!({
            "schema": 1, "training_reference": TRAINING_REFERENCE_REVISION,
            "base_repo": self.base_repo, "base_revision": self.base_revision,
            "artifacts": self.artifacts, "dataset": self.dataset,
            "preprocessing": self.preprocessing, "configuration": self.configuration,
            "factors": self.factors,
        })
        .to_string())
    }

    pub fn from_metadata(text: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| invalid(format!("invalid resume metadata: {e}")))?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid("resume metadata must be an object"))?;
        const KEYS: [&str; 9] = [
            "schema",
            "training_reference",
            "base_repo",
            "base_revision",
            "artifacts",
            "dataset",
            "preprocessing",
            "configuration",
            "factors",
        ];
        if object.len() != KEYS.len()
            || object.keys().any(|k| !KEYS.contains(&k.as_str()))
            || value["schema"] != 1
            || value["training_reference"] != TRAINING_REFERENCE_REVISION
        {
            return Err(invalid(
                "unknown/incomplete resume schema or training reference",
            ));
        }
        let string = |key: &str| {
            value[key]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| invalid(format!("resume {key} must be a string")))
        };
        let identity = Self {
            base_repo: string("base_repo")?,
            base_revision: string("base_revision")?,
            artifacts: serde_json::from_value(value["artifacts"].clone())
                .map_err(|e| invalid(format!("resume artifacts: {e}")))?,
            dataset: string("dataset")?,
            preprocessing: string("preprocessing")?,
            configuration: serde_json::from_value(value["configuration"].clone())
                .map_err(|e| invalid(format!("resume configuration: {e}")))?,
            factors: serde_json::from_value(value["factors"].clone())
                .map_err(|e| invalid(format!("resume factors: {e}")))?,
        };
        identity.validate()?;
        Ok(identity)
    }
}

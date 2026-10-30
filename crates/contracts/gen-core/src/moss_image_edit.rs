//! Frozen MOSS edit adapter identity and the tensor-neutral edit inputs/receipt the
//! [`Generator`](crate::Generator) MOSS edit entry points exchange. Reference sizing, request
//! mapping and package validation live in `gen_core_moss_image::moss_image_edit` (sc-27316).

use crate::moss_image::{MossImageAcceleration, MossImageAdvancedInputs, MossImageExecutionReport};
use crate::{Error, Image, Result};
use serde::{Deserialize, Serialize};

pub const EDIT_COMPONENT: &str = "moss-image-edit";
pub const EDIT_REPOSITORY: &str = "chinoll/Moss-Image-Edit";
pub const EDIT_REVISION: &str = "3be9c40037e1b89b17aa85ff6f8ee805872aa558";
pub const EDIT_WEIGHTS_SHA256: &str =
    "17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf";
pub const EDIT_CONFIG_SHA256: &str =
    "b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669";
pub const EDIT_CONTRACT: &str = "ref-prefix-only-t0__target-text-original-moss-v1";
/// The publisher has not declared a weights license. Code licensing is not weight licensing.
pub const EDIT_WEIGHTS_LICENSE: Option<&str> = None;
#[derive(Clone, Copy, Debug)]
pub struct MossImageEditInputs<'a> {
    /// One to four ordered RGB slots. Repeated images retain separate ordinals.
    pub references: &'a [Image],
    pub reference_max_edge: u32,
    pub use_kv_cache: bool,
    pub advanced: MossImageAdvancedInputs<'a>,
}
impl Default for MossImageEditInputs<'_> {
    fn default() -> Self {
        Self {
            references: &[],
            reference_max_edge: 512,
            use_kv_cache: true,
            advanced: MossImageAdvancedInputs::default(),
        }
    }
}
impl MossImageEditInputs<'_> {
    pub fn validate(&self) -> Result<Vec<[u32; 2]>> {
        if !(1..=4).contains(&self.references.len()) {
            return Err(Error::Unsupported(
                "MOSS edit requires one to four ordered reference slots".into(),
            ));
        }
        self.references
            .iter()
            .map(|image| {
                let expected = (image.width as usize)
                    .checked_mul(image.height as usize)
                    .and_then(|n| n.checked_mul(3));
                if expected != Some(image.pixels.len()) || image.width == 0 || image.height == 0 {
                    return Err(Error::Unsupported(
                        "MOSS reference requires nonempty exact RGB bytes".into(),
                    ));
                }
                reference_dimensions(image.width, image.height, self.reference_max_edge)
            })
            .collect()
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageEditReferenceReceipt {
    pub original_dimensions: [u32; 2],
    pub encoded_dimensions: [u32; 2],
    pub pixels_sha256: String,
    pub resized_pixels_sha256: String,
    pub latent_sha256: String,
}
/// Completed edit evidence. `base` records the ordinary backbone, effective controls and noise;
/// the extra fields identify the adapter and reference conditioning actually consumed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageEditExecutionReport {
    pub base: MossImageExecutionReport,
    pub adapter_path: std::path::PathBuf,
    pub edit_contract: String,
    pub adapter_repository: String,
    pub adapter_revision: String,
    pub adapter_sha256: String,
    pub config_sha256: String,
    pub applied_projections: usize,
    pub references: Vec<MossImageEditReferenceReceipt>,
    pub reference_max_edge: u32,
    pub use_kv_cache: bool,
}
impl MossImageEditExecutionReport {
    pub fn validate(&self) -> Result<()> {
        let digest = |v: &str| {
            v.len() == 64
                && v.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        let r = &self.base.data().resolved;
        if self.adapter_repository != EDIT_REPOSITORY
            || self.adapter_revision != EDIT_REVISION
            || self.adapter_sha256 != EDIT_WEIGHTS_SHA256
            || self.config_sha256 != EDIT_CONFIG_SHA256
            || self.edit_contract != EDIT_CONTRACT
            || self.applied_projections != 252
            || self.adapter_path.as_os_str().is_empty()
            || r.acceleration != MossImageAcceleration::Off
            || !r.uncond_skip_layers.is_empty()
            || r.slg.is_some()
            || !(1..=4).contains(&self.references.len())
        {
            return Err(Error::Unsupported(
                "invalid MOSS edit execution identity or policy".into(),
            ));
        }
        for reference in &self.references {
            if reference.encoded_dimensions
                != reference_dimensions(
                    reference.original_dimensions[0],
                    reference.original_dimensions[1],
                    self.reference_max_edge,
                )?
                || !digest(&reference.pixels_sha256)
                || !digest(&reference.resized_pixels_sha256)
                || !digest(&reference.latent_sha256)
            {
                return Err(Error::Unsupported(
                    "invalid MOSS edit reference receipt".into(),
                ));
            }
        }
        Ok(())
    }
}
impl<'de> Deserialize<'de> for MossImageEditExecutionReport {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            base: MossImageExecutionReport,
            adapter_path: std::path::PathBuf,
            edit_contract: String,
            adapter_repository: String,
            adapter_revision: String,
            adapter_sha256: String,
            config_sha256: String,
            applied_projections: usize,
            references: Vec<MossImageEditReferenceReceipt>,
            reference_max_edge: u32,
            use_kv_cache: bool,
        }
        let w = Wire::deserialize(deserializer)?;
        let report = Self {
            base: w.base,
            adapter_path: w.adapter_path,
            edit_contract: w.edit_contract,
            adapter_repository: w.adapter_repository,
            adapter_revision: w.adapter_revision,
            adapter_sha256: w.adapter_sha256,
            config_sha256: w.config_sha256,
            applied_projections: w.applied_projections,
            references: w.references,
            reference_max_edge: w.reference_max_edge,
            use_kv_cache: w.use_kv_cache,
        };
        report.validate().map_err(serde::de::Error::custom)?;
        Ok(report)
    }
}

/// Upstream no-crop resize, nearest /16 rounding (ties upward), never aspect-ratio crop.
pub fn reference_dimensions(width: u32, height: u32, max_edge: u32) -> Result<[u32; 2]> {
    if width == 0 || height == 0 || !(16..=4096).contains(&max_edge) {
        return Err(Error::Unsupported(
            "MOSS reference edge must be in 16..=4096 and source nonempty".into(),
        ));
    }
    let limit = max_edge / 16 * 16;
    let scale = (limit as f64 / width.max(height) as f64).min(1.0);
    let aligned = |n: u32| (((n as f64 * scale / 16.0 + 0.5).floor() as u32) * 16).clamp(16, limit);
    Ok([aligned(width), aligned(height)])
}

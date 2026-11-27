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

/// Owned checked CPU embeddings carried inside [`MossImageEditRequest`] on
/// [`crate::MossImageExtension`].
#[derive(Clone, Debug, PartialEq)]
pub struct MossImageOwnedEmbeddings {
    pub values: Vec<f32>,
    pub batch: usize,
    pub sequence: usize,
    pub mask: Option<Vec<u8>>,
}

impl MossImageOwnedEmbeddings {
    pub fn as_borrowed(&self) -> crate::moss_image::MossImageEmbeddings<'_> {
        crate::moss_image::MossImageEmbeddings {
            values: &self.values,
            batch: self.batch,
            sequence: self.sequence,
            mask: self.mask.as_deref(),
        }
    }

    pub fn from_borrowed(value: crate::moss_image::MossImageEmbeddings<'_>) -> Self {
        Self {
            values: value.values.to_vec(),
            batch: value.batch,
            sequence: value.sequence,
            mask: value.mask.map(|m| m.to_vec()),
        }
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        self.values
            .iter()
            .copied()
            .find(|v| !v.is_finite())
            .map(|v| ("moss_image.edit.advanced.embeddings", v))
    }
}

/// Owned advanced inputs for [`MossImageEditRequest`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MossImageAdvancedRequest {
    pub positive: Option<MossImageOwnedEmbeddings>,
    pub negative: Option<MossImageOwnedEmbeddings>,
    pub negative_prompts: Option<Vec<String>>,
    pub noise: Option<Vec<f32>>,
    pub output: crate::moss_image::MossImageOutputKind,
}

impl MossImageAdvancedRequest {
    pub fn as_borrowed(&self) -> MossImageAdvancedInputs<'_> {
        MossImageAdvancedInputs {
            positive: self.positive.as_ref().map(MossImageOwnedEmbeddings::as_borrowed),
            negative: self.negative.as_ref().map(MossImageOwnedEmbeddings::as_borrowed),
            negative_prompts: self.negative_prompts.as_deref(),
            noise: self.noise.as_deref(),
            output: self.output,
        }
    }

    pub fn from_borrowed(value: MossImageAdvancedInputs<'_>) -> Self {
        Self {
            positive: value.positive.map(MossImageOwnedEmbeddings::from_borrowed),
            negative: value.negative.map(MossImageOwnedEmbeddings::from_borrowed),
            negative_prompts: value.negative_prompts.map(|v| v.to_vec()),
            noise: value.noise.map(|v| v.to_vec()),
            output: value.output,
        }
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        if let Some(v) = self
            .positive
            .as_ref()
            .and_then(MossImageOwnedEmbeddings::first_nonfinite_float)
        {
            return Some(v);
        }
        if let Some(v) = self
            .negative
            .as_ref()
            .and_then(MossImageOwnedEmbeddings::first_nonfinite_float)
        {
            return Some(v);
        }
        self.noise
            .as_ref()
            .and_then(|noise| noise.iter().copied().find(|v| !v.is_finite()))
            .map(|v| ("moss_image.edit.advanced.noise", v))
    }
}

/// Owned MOSS-Image-Edit payload carried inside [`crate::MossImageExtension::edit`] so native
/// MOSS edit executes through the unified [`crate::Generator`] entry points without family-named
/// trait methods (`PIPE-12`, `REG-4`).
#[derive(Clone, Debug, PartialEq)]
pub struct MossImageEditRequest {
    /// One to four ordered RGB slots. Repeated images retain separate ordinals.
    pub references: Vec<Image>,
    pub reference_max_edge: u32,
    pub use_kv_cache: bool,
    pub advanced: MossImageAdvancedRequest,
}

impl Default for MossImageEditRequest {
    fn default() -> Self {
        Self {
            references: Vec::new(),
            reference_max_edge: 512,
            use_kv_cache: true,
            advanced: MossImageAdvancedRequest::default(),
        }
    }
}

impl MossImageEditRequest {
    pub fn as_inputs(&self) -> MossImageEditInputs<'_> {
        MossImageEditInputs {
            references: &self.references,
            reference_max_edge: self.reference_max_edge,
            use_kv_cache: self.use_kv_cache,
            advanced: self.advanced.as_borrowed(),
        }
    }

    pub fn from_inputs(inputs: MossImageEditInputs<'_>) -> Self {
        Self {
            references: inputs.references.to_vec(),
            reference_max_edge: inputs.reference_max_edge,
            use_kv_cache: inputs.use_kv_cache,
            advanced: MossImageAdvancedRequest::from_borrowed(inputs.advanced),
        }
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        self.advanced.first_nonfinite_float()
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pag: Option<crate::PagExecutionReport>,
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
        if self.pag != self.base.data().pag || self.pag.is_some() != r.pag.is_some() {
            return Err(Error::Unsupported(
                "invalid MOSS edit PAG execution report".into(),
            ));
        }
        if let Some(pag) = &self.pag {
            pag.validate_with_capability(&crate::moss_image::moss_image_pag_capability())
                .map_err(|e| Error::Unsupported(format!("MOSS-Image: {e}")))?;
            let actual_steps = match &self.base.data().sampling {
                crate::moss_image::MossImageSamplingReceipt::Euler { sigmas } => {
                    sigmas.len().saturating_sub(1) as u32
                }
                crate::moss_image::MossImageSamplingReceipt::Dmd { step_codes, .. } => {
                    step_codes.len() as u32
                }
            };
            if pag.total_steps != actual_steps
                || r.pag
                    .as_ref()
                    .map(|resolved_pag| resolved_pag.execution_report(actual_steps))
                    .as_ref()
                    != Some(pag)
            {
                return Err(Error::Unsupported(
                    "invalid MOSS edit PAG execution report".into(),
                ));
            }
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
            #[serde(default)]
            pag: Option<crate::PagExecutionReport>,
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
            pag: w.pag,
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::moss_image::{
        moss_image_schedule, MossImageAssetReceipt, MossImageAssetRole,
        MossImageConditioningInputReceipt, MossImageConditioningReceipt,
        MossImageConditioningSource, MossImageDtype, MossImageExecutionData,
        MossImageExecutionDevice, MossImageOptions, MossImageOutputKind, MossImageResolved,
        MossImageSampleReceipt, MossImageSamplingReceipt, MossImageVariant, SFT_REPO, SFT_REVISION,
        SNAPSHOT_FILES,
    };
    use crate::{GenerationRequest, PagConfig};

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn build_edit_report(pag: Option<PagConfig>) -> MossImageEditExecutionReport {
        let req = GenerationRequest {
            steps: Some(30),
            guidance: Some(4.0),
            features: crate::FeatureParams {
                pag,
                ..Default::default()
            },
            extensions: crate::FamilyExtensions {
                moss_image: Some(crate::MossImageExtension::from_options(MossImageOptions {
                    uncond_skip_layers: Some(vec![]),
                    ..Default::default()
                })),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &req).unwrap();
        let sigmas = moss_image_schedule(resolved.steps, resolved.shift, resolved.training_shift)
            .expect("schedule");
        let actual_steps = (sigmas.len() - 1) as u32;
        let pag_exec = resolved
            .pag
            .as_ref()
            .map(|resolved_pag| resolved_pag.execution_report(actual_steps));
        let assets = SNAPSHOT_FILES
            .iter()
            .copied()
            .filter(|p| *p != "LICENSE")
            .map(|p| {
                let role = if p.starts_with("transformer/") {
                    MossImageAssetRole::Transformer
                } else if p.starts_with("text_encoder/") {
                    MossImageAssetRole::TextEncoder
                } else if p.starts_with("tokenizer/") {
                    MossImageAssetRole::Tokenizer
                } else if p.starts_with("scheduler/") {
                    MossImageAssetRole::Scheduler
                } else {
                    MossImageAssetRole::Vae
                };
                MossImageAssetReceipt {
                    role,
                    repository: SFT_REPO.into(),
                    revision: SFT_REVISION.into(),
                    relative_path: p.into(),
                    sha256: HASH.into(),
                }
            })
            .collect();
        let cond_input = MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: HASH.into(),
            sequence_length: 8,
            valid_tokens: 8,
        };
        let component_dtypes: BTreeMap<String, MossImageDtype> = [
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
        let base = MossImageExecutionReport::new(MossImageExecutionData {
            execution_device: MossImageExecutionDevice::Cuda,
            selected_stream_device: None,
            resolved,
            assets,
            sampling: MossImageSamplingReceipt::Euler { sigmas },
            samples: vec![MossImageSampleReceipt {
                seed: Some(7),
                noise_sha256: HASH.into(),
            }],
            conditioning: vec![MossImageConditioningReceipt {
                positive: cond_input.clone(),
                negative: Some(cond_input),
            }],
            images_per_prompt: 1,
            latent_shape: [1, 128, 64, 64],
            output_kind: MossImageOutputKind::Images,
            component_dtypes,
            adapter_reports: vec![],
            pag: pag_exec.clone(),
        })
        .unwrap();
        MossImageEditExecutionReport {
            base,
            adapter_path: "/fixture/moss_edit.safetensors".into(),
            edit_contract: EDIT_CONTRACT.into(),
            adapter_repository: EDIT_REPOSITORY.into(),
            adapter_revision: EDIT_REVISION.into(),
            adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
            config_sha256: EDIT_CONFIG_SHA256.into(),
            applied_projections: 252,
            references: vec![MossImageEditReferenceReceipt {
                original_dimensions: [512, 512],
                encoded_dimensions: [512, 512],
                pixels_sha256: HASH.into(),
                resized_pixels_sha256: HASH.into(),
                latent_sha256: HASH.into(),
            }],
            reference_max_edge: 512,
            use_kv_cache: true,
            pag: pag_exec,
        }
    }

    #[test]
    fn edit_report_pag_round_trips_and_rejects_both_base_vs_edit_mismatch_directions() {
        let no_pag = build_edit_report(None);
        no_pag.validate().unwrap();
        let no_pag_wire = serde_json::to_value(&no_pag).unwrap();
        assert!(no_pag_wire.get("pag").is_none());
        assert_eq!(
            serde_json::from_value::<MossImageEditExecutionReport>(no_pag_wire.clone()).unwrap(),
            no_pag
        );

        let with_pag = build_edit_report(Some(PagConfig {
            scale: 2.0,
            layers: Some(vec![20]),
            ..Default::default()
        }));
        with_pag.validate().unwrap();
        let pag_exec = with_pag.pag.clone().expect("edit PAG report");
        let with_pag_wire = serde_json::to_value(&with_pag).unwrap();
        assert_eq!(
            serde_json::from_value::<MossImageEditExecutionReport>(with_pag_wire.clone()).unwrap(),
            with_pag
        );

        // Mismatch 1: base.pag == Some(_), edit.pag == None (silently dropping executed PAG).
        let mut dropped_edit_pag = with_pag.clone();
        dropped_edit_pag.pag = None;
        assert!(matches!(
            dropped_edit_pag.validate(),
            Err(Error::Unsupported(_))
        ));
        let mut omitted_wire = with_pag_wire.clone();
        omitted_wire.as_object_mut().unwrap().remove("pag");
        assert!(serde_json::from_value::<MossImageEditExecutionReport>(omitted_wire).is_err());
        let mut null_wire = with_pag_wire;
        null_wire["pag"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<MossImageEditExecutionReport>(null_wire).is_err());

        // Mismatch 2: base.pag == None, edit.pag == Some(_) (fabricating PAG on a non-PAG run).
        let mut fabricated_edit_pag = no_pag;
        fabricated_edit_pag.pag = Some(pag_exec.clone());
        assert!(matches!(
            fabricated_edit_pag.validate(),
            Err(Error::Unsupported(_))
        ));
        let mut fabricated_wire = no_pag_wire;
        fabricated_wire["pag"] = serde_json::to_value(&pag_exec).unwrap();
        assert!(serde_json::from_value::<MossImageEditExecutionReport>(fabricated_wire).is_err());

        // Mismatch 3: base.pag == Some(p1), edit.pag == Some(p2) with differing fields.
        let mut differing_edit_pag = with_pag;
        differing_edit_pag.pag.as_mut().unwrap().scale = 3.5;
        assert!(matches!(
            differing_edit_pag.validate(),
            Err(Error::Unsupported(_))
        ));
    }
}

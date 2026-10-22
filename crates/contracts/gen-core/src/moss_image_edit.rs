//! Frozen MOSS edit adapter and tensor-neutral execution inputs.
use crate::moss_image::{MossImageAcceleration, MossImageAdvancedInputs, MossImageExecutionReport};
use crate::{Error, GenerationRequest, Image, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const EDIT_COMPONENT: &str = "moss-image-edit";
pub const EDIT_REPOSITORY: &str = "chinoll/Moss-Image-Edit";
pub const EDIT_REVISION: &str = "e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11";
pub const EDIT_WEIGHTS_SHA256: &str =
    "17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf";
pub const EDIT_CONFIG_SHA256: &str =
    "b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669";
pub const EDIT_CONTRACT: &str = "ref-prefix-only-t0__target-text-original-moss-v1";
/// The publisher has not declared a weights license. Code licensing is not weight licensing.
pub const EDIT_WEIGHTS_LICENSE: Option<&str> = None;

#[derive(Clone, Copy, Debug)]
pub struct MossImageEditInputs<'a> {
    /// Ordered RGB images. The first delivery accepts exactly one reference.
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
        if self.references.len() != 1 {
            return Err(Error::Unsupported(
                "MOSS edit currently requires exactly one ordered reference".into(),
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
/// Resolve edit-only defaults without mutating an ordinary request or accepting img2img strength.
pub fn edit_request(request: &GenerationRequest) -> Result<GenerationRequest> {
    let mut request = request.clone();
    let options = request.moss_image.get_or_insert_with(Default::default);
    if options.acceleration != MossImageAcceleration::Off
        || options.slg.is_some()
        || options
            .uncond_skip_layers
            .as_ref()
            .is_some_and(|v| !v.is_empty())
    {
        return Err(Error::Unsupported("MOSS edit requires ordinary Base/SFT, full-layer CFG, and no SLG; edit+DMD is unvalidated".into()));
    }
    if !request.conditioning.is_empty() {
        return Err(Error::Unsupported(
            "MOSS edit references must be supplied once through typed edit inputs".into(),
        ));
    }
    options.uncond_skip_layers = Some(vec![]);
    request.steps.get_or_insert(30);
    request.guidance.get_or_insert(4.0);
    request.scheduler_shift.get_or_insert(5.0);
    Ok(request)
}
pub fn edit_adapter_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    for layer in 0..42 {
        for (projection, input, output) in [
            ("q", 3072, 3072),
            ("k", 3072, 3072),
            ("v", 3072, 3072),
            ("o", 3072, 3072),
            ("ffn_up_gate", 3072, 16384),
            ("ffn_down", 8192, 3072),
        ] {
            shapes.insert(
                format!("layers.{layer}.proj.{projection}.lora_A"),
                vec![64, input],
            );
            shapes.insert(
                format!("layers.{layer}.proj.{projection}.lora_B"),
                vec![output, 64],
            );
        }
    }
    shapes
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
            || self.references.len() != 1
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edit_contract_rejects_combination_and_preserves_controls() {
        let r = GenerationRequest {
            seed: Some(13),
            width: 640,
            height: 384,
            ..Default::default()
        };
        let edit = edit_request(&r).unwrap();
        assert_eq!(
            (
                edit.steps,
                edit.guidance,
                edit.seed,
                edit.width,
                edit.height
            ),
            (Some(30), Some(4.0), Some(13), 640, 384)
        );
        assert_eq!(edit.moss_image.unwrap().uncond_skip_layers, Some(vec![]));
        assert!(r.moss_image.is_none());
        let mut invalid = r;
        invalid.moss_image = Some(crate::moss_image::MossImageOptions {
            acceleration: MossImageAcceleration::Dmd(Default::default()),
            ..Default::default()
        });
        assert!(edit_request(&invalid).is_err());
        assert_eq!(reference_dimensions(999, 333, 512).unwrap(), [512, 176]);
        assert_eq!(reference_dimensions(8, 40, 512).unwrap(), [16, 48]);
        assert!(reference_dimensions(100, 100, 0).is_err());
        assert_eq!(edit_adapter_shapes().len(), 504);
        assert_eq!(
            edit_adapter_shapes()["layers.41.proj.ffn_down.lora_A"],
            [64, 8192]
        );
    }
}

/// Pillow-compatible 8-bit RGB separable Lanczos resampling. Intermediate pixels and signed
/// coefficients use Pillow's 22-bit fixed-point rounding, including the horizontal uint8 pass.
pub fn resize_reference(image: &Image, max_edge: u32) -> Result<Image> {
    let [width, height] = reference_dimensions(image.width, image.height, max_edge)?;
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|n| n.checked_mul(3));
    if expected != Some(image.pixels.len()) {
        return Err(Error::Unsupported("invalid reference RGB length".into()));
    }
    fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let support = 3.0 * filter_scale;
        let sinc = |v: f64| {
            if v == 0.0 {
                1.0
            } else {
                let p = v * std::f64::consts::PI;
                p.sin() / p
            }
        };
        (0..output)
            .map(|i| {
                let center = (i as f64 + 0.5) * scale;
                let start = ((center - support + 0.5) as isize).max(0) as usize;
                let end = ((center + support + 0.5) as usize).min(input);
                let weights: Vec<f64> = (start..end)
                    .map(|x| {
                        let v = (x as f64 - center + 0.5) / filter_scale;
                        if (-3.0..3.0).contains(&v) {
                            sinc(v) * sinc(v / 3.0)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let sum: f64 = weights.iter().sum();
                (
                    start,
                    weights
                        .into_iter()
                        .map(|w| {
                            let v = w / sum * (1 << 22) as f64;
                            (v + if v < 0.0 { -0.5 } else { 0.5 }) as i32
                        })
                        .collect(),
                )
            })
            .collect()
    }
    let horizontal = coefficients(image.width as usize, width as usize);
    let vertical = coefficients(image.height as usize, height as usize);
    let clip = |sum: i64| ((sum + (1 << 21)) >> 22).clamp(0, 255) as u8;
    let mut intermediate = vec![0u8; width as usize * image.height as usize * 3];
    for y in 0..image.height as usize {
        for (x, (start, weights)) in horizontal.iter().enumerate() {
            for c in 0..3 {
                let sum = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        image.pixels[(y * image.width as usize + start + i) * 3 + c] as i64
                            * *w as i64
                    })
                    .sum();
                intermediate[(y * width as usize + x) * 3 + c] = clip(sum);
            }
        }
    }
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for (y, (start, weights)) in vertical.iter().enumerate() {
        for x in 0..width as usize {
            for c in 0..3 {
                let sum = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        intermediate[((start + i) * width as usize + x) * 3 + c] as i64 * *w as i64
                    })
                    .sum();
                pixels[(y * width as usize + x) * 3 + c] = clip(sum);
            }
        }
    }
    Ok(Image {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod resize_tests {
    use super::*;
    #[test]
    fn pillow_frozen_rgb_resize_matches_every_byte() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/tiny_edit/resize_fixture.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let number = |k: &str| case[k].as_u64().unwrap() as u32;
            let bytes = |k: &str| {
                case[k]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect::<Vec<_>>()
            };
            let original = Image {
                width: number("width"),
                height: number("height"),
                pixels: bytes("input_rgb"),
            };
            let resized = resize_reference(&original, number("max_edge")).unwrap();
            assert_eq!(
                (resized.width, resized.height),
                (number("output_width"), number("output_height"))
            );
            assert_eq!(
                resized.pixels,
                bytes("expected_rgb"),
                "{}x{} edge{}",
                original.width,
                original.height,
                number("max_edge")
            );
        }
    }
}

/// Validate the complete frozen A/B layout before any backend tensor allocation.
pub fn validate_edit_adapter_headers(
    headers: &[crate::weightsmeta::SafetensorsTensorHeader],
) -> Result<()> {
    let expected = edit_adapter_shapes();
    let mut seen = std::collections::BTreeSet::new();
    for header in headers {
        if !seen.insert(header.name.clone())
            || expected.get(&header.name) != Some(&header.shape)
            || header.dtype != crate::weightsmeta::Dtype::BF16
            || header.data_bytes != header.shape.iter().map(|n| *n as u64).product::<u64>() * 2
        {
            return Err(Error::Unsupported(format!(
                "invalid MOSS edit tensor: {}",
                header.name
            )));
        }
    }
    if seen.len() != expected.len() {
        return Err(Error::Unsupported(
            "MOSS edit requires all 252 complete rank64 A/B projection pairs".into(),
        ));
    }
    Ok(())
}
/// Complete config identity/semantics, including the exact module and target sets.
pub fn validate_edit_config(value: &serde_json::Value) -> Result<()> {
    use serde_json::json;
    let modules = value["modules"]
        .as_object()
        .ok_or_else(|| Error::Unsupported("missing MOSS edit modules".into()))?;
    let targets = value["targets"]
        .as_array()
        .ok_or_else(|| Error::Unsupported("missing MOSS edit targets".into()))?;
    let shapes = edit_adapter_shapes();
    let expected: std::collections::BTreeSet<_> = shapes
        .keys()
        .filter_map(|k| k.strip_suffix(".lora_A").map(str::to_string))
        .collect();
    let target_set: std::collections::BTreeSet<_> = targets
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    if value["format"] != "moss-edit-lora"
        || value["format_version"] != 1
        || value["metadata"]["edit_contract"] != EDIT_CONTRACT
        || value["weights_sha256"] != EDIT_WEIGHTS_SHA256
        || modules
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            != expected
        || target_set != expected
        || targets.len() != 252
        || value["metadata"]["model_config"]
            != json!({"dim":3072,"ffn_dim":8192,"freq_dim":256,"latent_channels":128,"num_heads":24,"num_layers":42,"rope_base":10000.0,"text_dim":4096})
    {
        return Err(Error::Unsupported(
            "invalid frozen MOSS edit configuration".into(),
        ));
    }
    for name in expected {
        let module = &modules[&name];
        if module["rank"] != 64
            || module["alpha"] != 64.0
            || module["dropout"] != 0.0
            || module["param_dtype"] != "bfloat16"
            || module["in_features"] != shapes[&format!("{name}.lora_A")][1]
            || module["out_features"] != shapes[&format!("{name}.lora_B")][0]
        {
            return Err(Error::Unsupported(format!(
                "invalid MOSS edit module configuration: {name}"
            )));
        }
    }
    Ok(())
}
#[cfg(test)]
mod package_tests {
    use super::*;
    use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};
    #[test]
    fn frozen_package_config_and_all_504_headers_are_exact() {
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/tiny_edit/adapter_config.json"
        ))
        .unwrap();
        validate_edit_config(&config).unwrap();
        for change in [
            "rank",
            "alpha",
            "dropout",
            "param_dtype",
            "in_features",
            "out_features",
        ] {
            let mut corrupt = config.clone();
            corrupt["modules"]["layers.0.proj.q"][change] = serde_json::json!(-1);
            assert!(validate_edit_config(&corrupt).is_err(), "{change}");
        }
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/tiny_edit/adapter_header.json"
        ))
        .unwrap();
        let headers: Vec<_> = fixture
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| k.as_str() != "__metadata__")
            .map(|(name, value)| SafetensorsTensorHeader {
                name: name.clone(),
                dtype: if value["dtype"] == "BF16" {
                    Dtype::BF16
                } else {
                    Dtype::F32
                },
                shape: value["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
                data_bytes: value["data_offsets"][1].as_u64().unwrap()
                    - value["data_offsets"][0].as_u64().unwrap(),
            })
            .collect();
        validate_edit_adapter_headers(&headers).unwrap();
        let mut missing = headers.clone();
        missing.pop();
        assert!(validate_edit_adapter_headers(&missing).is_err());
        let mut duplicate = headers.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(validate_edit_adapter_headers(&duplicate).is_err());
        let mut wrong_dtype = headers.clone();
        wrong_dtype[0].dtype = Dtype::F32;
        assert!(validate_edit_adapter_headers(&wrong_dtype).is_err());
        let mut wrong_shape = headers;
        wrong_shape[0].shape[0] += 1;
        assert!(validate_edit_adapter_headers(&wrong_shape).is_err());
    }
}

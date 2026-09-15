//! Exact configuration and safetensors-header contracts for multimodal conditioning towers.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::weightsmeta::Dtype;
use crate::{EncoderConfigFloat, EncoderContract, Error, Result, SafetensorsTensorHeader};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionEncoderArchitecture {
    Qwen3Vl,
    Qwen2_5Vl,
    /// FLUX.2-dev's Mistral3 wrapper: a Pixtral tower plus the separately named multimodal
    /// projector that maps vision features into the Mistral language width.
    PixtralMistral3,
}

/// The exact vision half of a multimodal conditioning encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionEncoderContract {
    pub architecture: VisionEncoderArchitecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub output_width: usize,
    pub hidden_activation: &'static str,
    /// Effective rotary base consumed by the vision runtime. Published Qwen configs commonly omit
    /// this field, in which case both supported architectures use the Transformers default (10_000).
    pub rope_theta: EncoderConfigFloat,
    /// Effective LayerNorm/RMSNorm epsilon consumed by the vision runtime. The serialized field name
    /// differs by architecture; omission means the architecture default (1e-6).
    pub normalization_eps: EncoderConfigFloat,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub num_position_embeddings: Option<usize>,
    pub deepstack_visual_indexes: &'static [usize],
    pub window_size: Option<usize>,
    pub full_attention_block_indexes: &'static [usize],
}

impl VisionEncoderContract {
    pub fn validate_definition(&self, language: &EncoderContract) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.output_width == 0
            || !self.rope_theta.get().is_finite()
            || self.rope_theta.get() <= 0.0
            || !self.normalization_eps.get().is_finite()
            || self.normalization_eps.get() <= 0.0
            || self.patch_size == 0
            || self.temporal_patch_size == 0
            || self.spatial_merge_size == 0
            || self.in_channels == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.output_width != language.hidden_size
            || self
                .deepstack_visual_indexes
                .iter()
                .any(|&layer| layer >= self.num_hidden_layers)
            || self
                .full_attention_block_indexes
                .iter()
                .any(|&layer| layer >= self.num_hidden_layers)
        {
            return Err(Error::Unsupported(format!(
                "invalid multimodal vision contract: {self:?}; language_hidden_size={}",
                language.hidden_size
            )));
        }
        match self.architecture {
            VisionEncoderArchitecture::Qwen3Vl
                if self.num_position_embeddings.is_some()
                    && !self.deepstack_visual_indexes.is_empty()
                    && self.window_size.is_none()
                    && self.full_attention_block_indexes.is_empty() => {}
            VisionEncoderArchitecture::Qwen2_5Vl
                if self.num_position_embeddings.is_none()
                    && self.deepstack_visual_indexes.is_empty()
                    && self.window_size.is_some()
                    && !self.full_attention_block_indexes.is_empty() => {}
            VisionEncoderArchitecture::PixtralMistral3
                if self.num_position_embeddings.is_none()
                    && self.deepstack_visual_indexes.is_empty()
                    && self.window_size.is_none()
                    && self.full_attention_block_indexes.is_empty()
                    && self.temporal_patch_size == 1 => {}
            _ => {
                return Err(Error::Unsupported(format!(
                    "vision contract fields do not match architecture {:?}",
                    self.architecture
                )))
            }
        }
        Ok(())
    }

    pub(crate) fn validate_config(
        &self,
        root: &Value,
        path: &Path,
        language: &EncoderContract,
    ) -> Result<()> {
        self.validate_definition(language)?;
        if self.architecture == VisionEncoderArchitecture::PixtralMistral3 {
            return self.validate_pixtral_mistral3_config(root, path, language);
        }
        let vision = root
            .get("vision_config")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                self.mismatch(path, "vision_config", "object", "missing or non-object")
            })?;
        let vision = Value::Object(vision.clone());
        let model_type = match self.architecture {
            VisionEncoderArchitecture::Qwen3Vl => "qwen3_vl",
            VisionEncoderArchitecture::Qwen2_5Vl => "qwen2_5_vl",
            VisionEncoderArchitecture::PixtralMistral3 => unreachable!("handled above"),
        };
        self.expect_str(&vision, path, "model_type", model_type)?;
        self.expect_str(&vision, path, "hidden_act", self.hidden_activation)?;
        self.expect_effective_f64(
            &vision,
            path,
            "rope_theta",
            self.rope_theta.get(),
            10_000.0,
            &[
                &["rope_theta"],
                &["rope_parameters", "rope_theta"],
                &["rope_scaling", "rope_theta"],
            ],
        )?;
        let normalization_fields: &[&[&str]] = match self.architecture {
            VisionEncoderArchitecture::Qwen3Vl => &[&["layer_norm_eps"], &["norm_eps"]],
            VisionEncoderArchitecture::Qwen2_5Vl => &[&["rms_norm_eps"], &["norm_eps"]],
            VisionEncoderArchitecture::PixtralMistral3 => unreachable!("handled above"),
        };
        self.expect_effective_f64(
            &vision,
            path,
            "normalization_eps",
            self.normalization_eps.get(),
            1e-6,
            normalization_fields,
        )?;
        for (field, expected) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("depth", self.num_hidden_layers),
            ("num_heads", self.num_attention_heads),
            ("out_hidden_size", self.output_width),
            ("patch_size", self.patch_size),
            ("temporal_patch_size", self.temporal_patch_size),
            ("spatial_merge_size", self.spatial_merge_size),
            ("in_channels", self.in_channels),
        ] {
            self.expect_usize(&vision, path, field, expected)?;
        }
        if let Some(expected) = self.num_position_embeddings {
            self.expect_usize(&vision, path, "num_position_embeddings", expected)?;
        }
        if let Some(expected) = self.window_size {
            self.expect_usize(&vision, path, "window_size", expected)?;
        }
        self.expect_usize_array(
            &vision,
            path,
            "deepstack_visual_indexes",
            self.deepstack_visual_indexes,
        )?;
        self.expect_usize_array(
            &vision,
            path,
            "fullatt_block_indexes",
            self.full_attention_block_indexes,
        )?;
        Ok(())
    }

    fn validate_pixtral_mistral3_config(
        &self,
        root: &Value,
        path: &Path,
        language: &EncoderContract,
    ) -> Result<()> {
        self.expect_str(root, path, "model_type", "mistral3")?;
        self.expect_usize(root, path, "image_token_index", 10)?;
        self.expect_bool(root, path, "multimodal_projector_bias", false)?;
        self.expect_str(root, path, "projector_hidden_act", "gelu")?;
        self.expect_usize(root, path, "spatial_merge_size", self.spatial_merge_size)?;
        self.expect_i64(root, path, "vision_feature_layer", -1)?;

        let vision = root
            .get("vision_config")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                self.mismatch(path, "vision_config", "object", "missing or non-object")
            })?;
        let vision = Value::Object(vision.clone());
        self.expect_str(&vision, path, "vision_config.model_type", "pixtral")?;
        self.expect_str(
            &vision,
            path,
            "vision_config.hidden_act",
            self.hidden_activation,
        )?;
        for (field, expected) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("head_dim", self.hidden_size / self.num_attention_heads),
            ("patch_size", self.patch_size),
            ("num_channels", self.in_channels),
        ] {
            self.expect_usize(
                &vision,
                path,
                match field {
                    "hidden_size" => "vision_config.hidden_size",
                    "intermediate_size" => "vision_config.intermediate_size",
                    "num_hidden_layers" => "vision_config.num_hidden_layers",
                    "num_attention_heads" => "vision_config.num_attention_heads",
                    "head_dim" => "vision_config.head_dim",
                    "patch_size" => "vision_config.patch_size",
                    "num_channels" => "vision_config.num_channels",
                    _ => unreachable!(),
                },
                expected,
            )?;
        }
        self.expect_f64(&vision, path, "vision_config.attention_dropout", 0.0)?;
        self.expect_effective_f64(
            &vision,
            path,
            "vision_config.rope_theta",
            self.rope_theta.get(),
            10_000.0,
            &[
                &["rope_theta"],
                &["rope_parameters", "rope_theta"],
                &["rope_scaling", "rope_theta"],
            ],
        )?;
        self.expect_effective_f64(
            &vision,
            path,
            "vision_config.normalization_eps",
            self.normalization_eps.get(),
            1e-5,
            &[&["rms_norm_eps"], &["norm_eps"]],
        )?;
        if self.output_width != language.hidden_size {
            return Err(self.mismatch(
                path,
                "multi_modal_projector.output_width",
                language.hidden_size,
                self.output_width,
            ));
        }
        Ok(())
    }

    pub fn validate_tensor_headers(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
    ) -> Result<()> {
        let expected = self.expected_headers()?;
        let owns_tensor = |name: &str| match self.architecture {
            VisionEncoderArchitecture::Qwen3Vl | VisionEncoderArchitecture::Qwen2_5Vl => {
                name.starts_with("visual.")
            }
            VisionEncoderArchitecture::PixtralMistral3 => {
                name.starts_with("vision_tower.") || name.starts_with("multi_modal_projector.")
            }
        };
        let actual = headers
            .iter()
            .filter(|header| owns_tensor(&header.name))
            .map(|header| (header.name.as_str(), header))
            .collect::<BTreeMap<_, _>>();
        for (name, shape) in &expected {
            let header = actual.get(name.as_str()).ok_or_else(|| {
                self.mismatch(
                    path,
                    "vision_tensor",
                    format!("{name} {shape:?}"),
                    "missing",
                )
            })?;
            if header.shape != *shape {
                return Err(self.mismatch(
                    path,
                    "vision_tensor_shape",
                    format!("{name} {shape:?}"),
                    format!("{:?}", header.shape),
                ));
            }
            if !matches!(header.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32) {
                return Err(self.mismatch(
                    path,
                    "vision_tensor_dtype",
                    "F16, BF16, or F32",
                    format!("{name}={:?}", header.dtype),
                ));
            }
        }
        let packed = headers
            .iter()
            .filter(|header| {
                owns_tensor(&header.name)
                    && (header.dtype == Dtype::U32
                        || header.name.ends_with(".scales")
                        || header.name.ends_with(".biases"))
            })
            .map(|header| header.name.clone())
            .collect::<Vec<_>>();
        if !packed.is_empty() {
            return Err(self.mismatch(
                path,
                "vision_packing",
                "dense vision tower",
                format!("packed tensors {packed:?}"),
            ));
        }
        Ok(())
    }

    /// Exact on-disk tensor surface materialized by the corresponding vision constructors.
    pub fn expected_headers(&self) -> Result<Vec<(String, Vec<usize>)>> {
        let h = self.hidden_size;
        let i = self.intermediate_size;
        let merged = h
            .checked_mul(self.spatial_merge_size)
            .and_then(|value| value.checked_mul(self.spatial_merge_size))
            .ok_or_else(|| Error::Unsupported("vision merged width overflow".into()))?;
        let mut tensors = vec![(
            "visual.patch_embed.proj.weight".into(),
            vec![
                h,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ],
        )];
        match self.architecture {
            VisionEncoderArchitecture::Qwen3Vl => {
                tensors.extend([
                    ("visual.patch_embed.proj.bias".into(), vec![h]),
                    (
                        "visual.pos_embed.weight".into(),
                        vec![
                            self.num_position_embeddings.expect("definition validated"),
                            h,
                        ],
                    ),
                ]);
                for layer in 0..self.num_hidden_layers {
                    let base = format!("visual.blocks.{layer}");
                    tensors.extend([
                        (format!("{base}.norm1.weight"), vec![h]),
                        (format!("{base}.norm1.bias"), vec![h]),
                        (format!("{base}.norm2.weight"), vec![h]),
                        (format!("{base}.norm2.bias"), vec![h]),
                        (format!("{base}.attn.qkv.weight"), vec![3 * h, h]),
                        (format!("{base}.attn.qkv.bias"), vec![3 * h]),
                        (format!("{base}.attn.proj.weight"), vec![h, h]),
                        (format!("{base}.attn.proj.bias"), vec![h]),
                        (format!("{base}.mlp.linear_fc1.weight"), vec![i, h]),
                        (format!("{base}.mlp.linear_fc1.bias"), vec![i]),
                        (format!("{base}.mlp.linear_fc2.weight"), vec![h, i]),
                        (format!("{base}.mlp.linear_fc2.bias"), vec![h]),
                    ]);
                }
                for base in std::iter::once("visual.merger".to_owned()).chain(
                    (0..self.deepstack_visual_indexes.len())
                        .map(|index| format!("visual.deepstack_merger_list.{index}")),
                ) {
                    let norm = if base == "visual.merger" { h } else { merged };
                    tensors.extend([
                        (format!("{base}.norm.weight"), vec![norm]),
                        (format!("{base}.norm.bias"), vec![norm]),
                        (format!("{base}.linear_fc1.weight"), vec![merged, merged]),
                        (format!("{base}.linear_fc1.bias"), vec![merged]),
                        (
                            format!("{base}.linear_fc2.weight"),
                            vec![self.output_width, merged],
                        ),
                        (format!("{base}.linear_fc2.bias"), vec![self.output_width]),
                    ]);
                }
            }
            VisionEncoderArchitecture::Qwen2_5Vl => {
                for layer in 0..self.num_hidden_layers {
                    let base = format!("visual.blocks.{layer}");
                    tensors.extend([
                        (format!("{base}.norm1.weight"), vec![h]),
                        (format!("{base}.norm2.weight"), vec![h]),
                        (format!("{base}.attn.qkv.weight"), vec![3 * h, h]),
                        (format!("{base}.attn.qkv.bias"), vec![3 * h]),
                        (format!("{base}.attn.proj.weight"), vec![h, h]),
                        (format!("{base}.attn.proj.bias"), vec![h]),
                        (format!("{base}.mlp.gate_proj.weight"), vec![i, h]),
                        (format!("{base}.mlp.gate_proj.bias"), vec![i]),
                        (format!("{base}.mlp.up_proj.weight"), vec![i, h]),
                        (format!("{base}.mlp.up_proj.bias"), vec![i]),
                        (format!("{base}.mlp.down_proj.weight"), vec![h, i]),
                        (format!("{base}.mlp.down_proj.bias"), vec![h]),
                    ]);
                }
                tensors.extend([
                    ("visual.merger.ln_q.weight".into(), vec![h]),
                    ("visual.merger.mlp.0.weight".into(), vec![merged, merged]),
                    ("visual.merger.mlp.0.bias".into(), vec![merged]),
                    (
                        "visual.merger.mlp.2.weight".into(),
                        vec![self.output_width, merged],
                    ),
                    ("visual.merger.mlp.2.bias".into(), vec![self.output_width]),
                ]);
            }
            VisionEncoderArchitecture::PixtralMistral3 => {
                tensors = vec![
                    (
                        "vision_tower.patch_conv.weight".into(),
                        vec![h, self.in_channels, self.patch_size, self.patch_size],
                    ),
                    ("vision_tower.ln_pre.weight".into(), vec![h]),
                ];
                for layer in 0..self.num_hidden_layers {
                    let base = format!("vision_tower.transformer.layers.{layer}");
                    tensors.extend([
                        (format!("{base}.attention_norm.weight"), vec![h]),
                        (format!("{base}.ffn_norm.weight"), vec![h]),
                        (format!("{base}.attention.q_proj.weight"), vec![h, h]),
                        (format!("{base}.attention.k_proj.weight"), vec![h, h]),
                        (format!("{base}.attention.v_proj.weight"), vec![h, h]),
                        (format!("{base}.attention.o_proj.weight"), vec![h, h]),
                        (format!("{base}.feed_forward.gate_proj.weight"), vec![i, h]),
                        (format!("{base}.feed_forward.up_proj.weight"), vec![i, h]),
                        (format!("{base}.feed_forward.down_proj.weight"), vec![h, i]),
                    ]);
                }
                tensors.extend([
                    ("multi_modal_projector.norm.weight".into(), vec![h]),
                    (
                        "multi_modal_projector.patch_merger.merging_layer.weight".into(),
                        vec![h, merged],
                    ),
                    (
                        "multi_modal_projector.linear_1.weight".into(),
                        vec![self.output_width, h],
                    ),
                    (
                        "multi_modal_projector.linear_2.weight".into(),
                        vec![self.output_width, self.output_width],
                    ),
                ]);
            }
        }
        Ok(tensors)
    }

    fn expect_str(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: &str,
    ) -> Result<()> {
        let key = field.rsplit('.').next().expect("field has a key");
        match config.get(key).and_then(Value::as_str) {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn expect_usize(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: usize,
    ) -> Result<()> {
        let key = field.rsplit('.').next().expect("field has a key");
        match config
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
        {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn expect_i64(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: i64,
    ) -> Result<()> {
        let key = field.rsplit('.').next().expect("field has a key");
        match config.get(key).and_then(Value::as_i64) {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn expect_bool(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: bool,
    ) -> Result<()> {
        let key = field.rsplit('.').next().expect("field has a key");
        match config.get(key).and_then(Value::as_bool) {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn expect_f64(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
    ) -> Result<()> {
        let key = field.rsplit('.').next().expect("field has a key");
        match config.get(key).and_then(Value::as_f64) {
            Some(actual) if actual.to_bits() == expected.to_bits() => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn expect_usize_array(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: &[usize],
    ) -> Result<()> {
        if expected.is_empty() && config.get(field).is_none() {
            return Ok(());
        }
        let actual = config
            .get(field)
            .and_then(Value::as_array)
            .and_then(|values| {
                values
                    .iter()
                    .map(|v| v.as_u64().and_then(|v| usize::try_from(v).ok()))
                    .collect::<Option<Vec<_>>>()
            });
        match actual {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => {
                Err(self.mismatch(path, field, format!("{expected:?}"), format!("{actual:?}")))
            }
            None => Err(self.mismatch(
                path,
                field,
                format!("{expected:?}"),
                "missing or non-integer array",
            )),
        }
    }

    /// Validate every authored alias, or the architecture default when all aliases are absent/null.
    /// This is deliberately stricter than picking the first alias: conflicting declarations must not
    /// validate while a backend silently consumes a different one.
    fn expect_effective_f64(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
        default: f64,
        candidates: &[&[&str]],
    ) -> Result<()> {
        let mut found = false;
        for candidate in candidates {
            let Some(value) = json_path(config, candidate) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_f64().ok_or_else(|| {
                self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={value}", candidate.join(".")),
                )
            })?;
            found = true;
            if actual.to_bits() != expected.to_bits() {
                return Err(self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={actual}", candidate.join(".")),
                ));
            }
        }
        if found || default.to_bits() == expected.to_bits() {
            Ok(())
        } else {
            Err(self.mismatch(
                path,
                field,
                expected,
                format!("architecture default {default}"),
            ))
        }
    }

    fn mismatch(
        &self,
        path: &Path,
        field: &'static str,
        expected: impl std::fmt::Display,
        actual: impl std::fmt::Display,
    ) -> Error {
        Error::Unsupported(format!(
            "vision encoder contract mismatch at {}: field `{field}` expected {expected}, got {actual}",
            path.display()
        ))
    }
}

fn json_path<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(root, |value, field| value.get(field))
}

//! Native Prism GGUF packed-tensor ingestion.
//!
//! PQ2_0/PTQ1_0 are transcoded directly to MLX's affine two-bit words. This is a packed-to-packed
//! operation; it does not pass through the generic dense GGUF converter.

use std::collections::{BTreeMap, BTreeSet};

use core_llm::{PrismHadamardMetadata, PrismPackedKind, PrismPackedMatrixRef, PrismTransformRole};
use mlx_rs::Array;

use crate::error::{Error, Result};
use crate::gguf::{GgufFile, MetaValue, TensorInfo};
use crate::primitives::prism::{PrismEmbedding, PrismLinear};
use crate::primitives::Weights;
use crate::prism::PrismMlxPack;

/// One GGUF tensor transcoded to MLX's packed affine representation.
pub struct PrismAffineParts {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub signs: Array,
    pub block: i32,
    pub role: PrismTransformRole,
}

/// In-memory native GGUF load product. Packed tensors remain two-bit MLX arrays.
pub struct LoadedPrismGguf {
    pub config: serde_json::Value,
    pub weights: Weights,
    pub pack: PrismMlxPack,
    pub tokenizer: core_llm::Tokenizer,
    pub template: Box<dyn core_llm::ChatTemplate>,
    pub template_capabilities: (bool, bool, bool, bool),
    pub stop_tokens: Vec<i32>,
}

/// Load a published Qwen35 Prism GGUF directly, without the generic dense conversion pipeline.
pub fn load(file: &GgufFile) -> Result<LoadedPrismGguf> {
    if file.meta_str("general.architecture") != Some("qwen35") {
        return Err(Error::Unsupported(
            "native Prism GGUF requires general.architecture=qwen35".into(),
        ));
    }
    let raw_hadamard = hadamard_metadata(file)?;
    let mut canonical_hadamard = raw_hadamard.clone();
    canonical_hadamard.forward_weight_names = raw_hadamard
        .forward_weight_names
        .iter()
        .map(|name| map_weight_name(name))
        .collect::<Result<_>>()?;
    canonical_hadamard.inverse_weight_names = raw_hadamard
        .inverse_weight_names
        .iter()
        .map(|name| map_weight_name(name))
        .collect::<Result<_>>()?;

    let mut tensors = std::collections::HashMap::new();
    let mut module_specs = Vec::new();
    let mut seen_packed = BTreeSet::new();
    for info in &file.tensors {
        let canonical = map_weight_name(&info.name)?;
        if matches!(info.ggml_type, 142 | 143) {
            let parts = transcode_tensor(file, info, &info.name, &raw_hadamard)?;
            let base = canonical.strip_suffix(".weight").ok_or_else(|| {
                Error::Config(format!("packed GGUF tensor `{canonical}` is not a weight"))
            })?;
            tensors.insert(canonical.clone(), parts.weight);
            tensors.insert(format!("{base}.scales"), parts.scales);
            tensors.insert(format!("{base}.biases"), parts.biases);
            tensors.insert(format!("{base}.signs"), parts.signs);
            let module_path = base
                .strip_prefix("language_model.")
                .ok_or_else(|| Error::Config(format!("invalid Prism key `{base}`")))?;
            module_specs.push((
                module_path.to_owned(),
                parts.role == PrismTransformRole::Inverse,
            ));
            seen_packed.insert(info.name.clone());
        } else {
            let values = crate::gguf::dequantize(
                info.ggml_type,
                file.tensor_data(info)?,
                info.num_elements(),
            )?;
            let shape = info
                .shape
                .iter()
                .map(|&n| {
                    i32::try_from(n).map_err(|_| Error::Config("GGUF dimension overflow".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            tensors.insert(canonical, Array::from_slice(&values, &shape));
        }
    }
    let declared: BTreeSet<_> = raw_hadamard
        .forward_weight_names
        .union(&raw_hadamard.inverse_weight_names)
        .cloned()
        .collect();
    if seen_packed != declared {
        return Err(Error::Config(
            "GGUF Prism packed tensors and transform manifest differ".into(),
        ));
    }
    let pack = PrismMlxPack::from_gguf(canonical_hadamard, module_specs)?;
    let config = reconstruct_config(file, &tensors)?;
    let reconstructed = match crate::gguf::tokenizer::reconstruct(file)? {
        crate::gguf::tokenizer::TokenizerOutcome::Reconstructed(t) => t,
        crate::gguf::tokenizer::TokenizerOutcome::Unsupported(reason) => {
            return Err(Error::Unsupported(format!("GGUF tokenizer: {reason}")))
        }
        crate::gguf::tokenizer::TokenizerOutcome::Absent => {
            return Err(Error::Config(
                "GGUF Prism tokenizer metadata is absent".into(),
            ))
        }
    };
    let tokenizer = core_llm::Tokenizer::from_json(&reconstructed.tokenizer_json.to_string())
        .map_err(|e| Error::Config(e.to_string()))?;
    let template =
        core_llm::JinjaChatTemplate::from_tokenizer_config(&reconstructed.tokenizer_config_json)
            .map_err(|e| Error::Config(e.to_string()))?;
    let template_capabilities = (
        template.source().contains("enable_thinking"),
        template.source().contains("reasoning_effort"),
        template.source().contains("preserve_thinking"),
        template.source().contains("tool_call"),
    );
    let mut stop_tokens = Vec::new();
    for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id"] {
        if let Some(token) = file.meta_u64(key).and_then(|n| i32::try_from(n).ok()) {
            if !stop_tokens.contains(&token) {
                stop_tokens.push(token);
            }
        }
    }
    Ok(LoadedPrismGguf {
        config,
        weights: Weights::from_map(tensors),
        pack,
        tokenizer,
        template: Box::new(template),
        template_capabilities,
        stop_tokens,
    })
}

fn map_weight_name(name: &str) -> Result<String> {
    let mapped = match name {
        "token_embd.weight" => "model.embed_tokens.weight".to_owned(),
        "output_norm.weight" => "model.norm.weight".to_owned(),
        "output.weight" => "lm_head.weight".to_owned(),
        _ => {
            let rest = name.strip_prefix("blk.").ok_or_else(|| {
                Error::Unsupported(format!("GGUF Prism tensor `{name}` has no Qwen35 mapping"))
            })?;
            let (layer, suffix) = rest.split_once('.').ok_or_else(|| {
                Error::Unsupported(format!("GGUF Prism tensor `{name}` has no Qwen35 mapping"))
            })?;
            layer.parse::<usize>().map_err(|_| {
                Error::Unsupported(format!("GGUF Prism tensor `{name}` has an invalid layer"))
            })?;
            let suffix = match suffix {
                "attn_norm.weight" => "input_layernorm.weight",
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
                "attn_gate.weight" => "linear_attn.in_proj_z.weight",
                "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
                "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
                "ssm_out.weight" => "linear_attn.out_proj.weight",
                "ssm_norm.weight" => "linear_attn.norm.weight",
                "ssm_a" => "linear_attn.A_log",
                "ssm_dt.bias" => "linear_attn.dt_bias",
                "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
                _ => {
                    return Err(Error::Unsupported(format!(
                        "GGUF Prism tensor `{name}` has no Qwen35 mapping"
                    )))
                }
            };
            format!("model.layers.{layer}.{suffix}")
        }
    };
    Ok(format!("language_model.{mapped}"))
}

fn reconstruct_config(
    file: &GgufFile,
    tensors: &std::collections::HashMap<String, Array>,
) -> Result<serde_json::Value> {
    let q = |key: &str| format!("qwen35.{key}");
    let u = |key: &str| -> Result<u64> {
        file.meta_u64(&q(key))
            .ok_or_else(|| Error::Config(format!("GGUF missing `{}`", q(key))))
    };
    let f = |key: &str| -> Result<f64> {
        file.meta_f64(&q(key))
            .ok_or_else(|| Error::Config(format!("GGUF missing `{}`", q(key))))
    };
    let embed = tensors
        .get("language_model.model.embed_tokens.weight")
        .ok_or_else(|| Error::MissingTensor("language_model.model.embed_tokens.weight".into()))?;
    let head_dim = u("attention.key_length")?;
    let rope_dim = u("rope.dimension_count")?;
    let value_heads = u("ssm.time_step_rank")?;
    let key_heads = u("ssm.group_count")?;
    let inner = u("ssm.inner_size")?;
    let state = u("ssm.state_size")?;
    if value_heads == 0
        || key_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || !inner.is_multiple_of(value_heads)
    {
        return Err(Error::Config("GGUF Prism has invalid GDN geometry".into()));
    }
    let section = required(file, "qwen35.rope.dimension_sections")?
        .as_array()
        .ok_or_else(|| Error::Config("GGUF Qwen35 rope sections must be an array".into()))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect::<Vec<_>>();
    if section.len() != 3 {
        return Err(Error::Config(
            "GGUF Qwen35 rope sections must have length 3".into(),
        ));
    }
    Ok(serde_json::json!({
        "model_type":"prism_hadamard_qwen35",
        "text_config":{
            "model_type":"qwen3_5_text",
            "hidden_size":u("embedding_length")?,
            "intermediate_size":u("feed_forward_length")?,
            "num_hidden_layers":u("block_count")?,
            "num_attention_heads":u("attention.head_count")?,
            "num_key_value_heads":u("attention.head_count_kv")?,
            "head_dim":head_dim,
            "vocab_size":embed.shape()[0],
            "rms_norm_eps":f("attention.layer_norm_rms_epsilon")?,
            "max_position_embeddings":u("context_length")?,
            "full_attention_interval":u("full_attention_interval")?,
            "linear_num_value_heads":value_heads,
            "linear_num_key_heads":key_heads,
            "linear_value_head_dim":inner/value_heads,
            "linear_key_head_dim":state,
            "linear_conv_kernel_dim":u("ssm.conv_kernel")?,
            "mtp_num_hidden_layers":0,
            "mtp_use_dedicated_embeddings":false,
            "tie_word_embeddings":false,
            "rope_parameters":{
                "rope_theta":f("rope.freq_base")?,
                "partial_rotary_factor":rope_dim as f64/head_dim as f64,
                "mrope_section":section
            }
        }
    }))
}

/// Validated Hadamard contract embedded in a published Prism GGUF.
pub fn hadamard_metadata(file: &GgufFile) -> Result<PrismHadamardMetadata> {
    eq_u64(file, "prism.hadamard.version", 1)?;
    eq_str(
        file,
        "prism.hadamard.transform",
        "normalized-sylvester-walsh-hadamard",
    )?;
    eq_str(file, "prism.hadamard.axis", "input-last-dimension")?;
    eq_str(file, "prism.hadamard.sign_mode", "explicit")?;
    let block_size = usize::try_from(
        required(file, "prism.hadamard.block_size")?
            .as_u64()
            .ok_or_else(|| {
                Error::Config("GGUF Prism Hadamard block size must be an integer".into())
            })?,
    )
    .map_err(|_| Error::Config("GGUF Prism Hadamard block size is too large".into()))?;
    let widths = usize_values(required(file, "prism.hadamard.sign_widths")?)?;
    let values = required(file, "prism.hadamard.sign_values")?
        .as_array()
        .ok_or_else(|| Error::Config("GGUF Prism signs must be an array".into()))?;
    let mut signs_by_width = BTreeMap::new();
    let mut at = 0usize;
    for width in widths {
        let end = at
            .checked_add(width)
            .ok_or_else(|| Error::Config("GGUF Prism sign length overflow".into()))?;
        let signs = values
            .get(at..end)
            .ok_or_else(|| Error::Config("GGUF Prism signs are truncated".into()))?
            .iter()
            .map(|v| match v.as_f64() {
                Some(-1.0) => Ok(-1),
                Some(1.0) => Ok(1),
                _ => Err(Error::Config("GGUF Prism signs must be -1 or +1".into())),
            })
            .collect::<Result<Vec<_>>>()?;
        if signs_by_width.insert(width, signs).is_some() {
            return Err(Error::Config(
                "GGUF Prism has a duplicate sign width".into(),
            ));
        }
        at = end;
    }
    if at != values.len() {
        return Err(Error::Config(
            "GGUF Prism signs have trailing values".into(),
        ));
    }
    let metadata = PrismHadamardMetadata {
        block_size,
        signs_by_width,
        forward_weight_names: string_values(required(file, "prism.hadamard.weight_names")?)?,
        inverse_weight_names: string_values(required(
            file,
            "prism.hadamard.inverse_weight_names",
        )?)?,
        gdn_v_grouped: required(file, "prism.hadamard.gdn_v_grouped")?.as_bool() == Some(true),
    };
    if !metadata.gdn_v_grouped {
        return Err(Error::Config(
            "published Prism GGUF requires grouped GDN tensors; refusing an ungrouped pack".into(),
        ));
    }
    metadata
        .validate()
        .map_err(|e| Error::Config(e.to_string()))?;
    Ok(metadata)
}

/// Transcode one packed GGUF matrix to an MLX-native packed linear operator.
pub fn linear_from_tensor(
    file: &GgufFile,
    info: &TensorInfo,
    canonical_weight_name: &str,
    metadata: &PrismHadamardMetadata,
) -> Result<PrismLinear> {
    let parts = transcode_tensor(file, info, canonical_weight_name, metadata)?;
    if parts.role != PrismTransformRole::Forward {
        return Err(Error::Config(format!(
            "GGUF Prism `{canonical_weight_name}` is not a forward matrix"
        )));
    }
    PrismLinear::new(
        canonical_weight_name,
        parts.weight,
        parts.scales,
        parts.biases,
        parts.signs,
        parts.block,
    )
}

/// Transcode the inverse-rotated packed token embedding.
pub fn embedding_from_tensor(
    file: &GgufFile,
    info: &TensorInfo,
    canonical_weight_name: &str,
    metadata: &PrismHadamardMetadata,
) -> Result<PrismEmbedding> {
    let parts = transcode_tensor(file, info, canonical_weight_name, metadata)?;
    if parts.role != PrismTransformRole::Inverse {
        return Err(Error::Config(format!(
            "GGUF Prism `{canonical_weight_name}` is not the inverse embedding"
        )));
    }
    PrismEmbedding::new(
        canonical_weight_name,
        parts.weight,
        parts.scales,
        parts.biases,
        parts.signs,
        parts.block,
    )
}

pub fn transcode_tensor(
    file: &GgufFile,
    info: &TensorInfo,
    name: &str,
    metadata: &PrismHadamardMetadata,
) -> Result<PrismAffineParts> {
    if info.shape.len() != 2 {
        return Err(Error::Config(format!(
            "GGUF Prism `{}` must be a matrix, got {:?}",
            info.name, info.shape
        )));
    }
    let kind = PrismPackedKind::from_ggml_type(info.ggml_type)
        .map_err(|e| Error::Config(e.to_string()))?;
    let rows = info.shape[0];
    let width = info.shape[1];
    let raw = file.tensor_data(info)?;
    let matrix = PrismPackedMatrixRef::from_gguf(kind, &[width, rows], raw)
        .map_err(|e| Error::Config(e.to_string()))?;
    let transform = metadata
        .classify_weight(name, width)
        .map_err(|e| Error::Config(e.to_string()))?;
    let signs = transform
        .signs
        .ok_or_else(|| Error::Config(format!("GGUF Prism packed tensor `{name}` is undeclared")))?;
    let groups = width / 128;
    let mut words = vec![0u32; rows * width / 16];
    let mut scales = vec![0f32; rows * groups];
    let mut biases = vec![0f32; rows * groups];
    for row in 0..rows {
        matrix
            .transcode_affine_row_into(
                row,
                &mut words[row * width / 16..(row + 1) * width / 16],
                &mut scales[row * groups..(row + 1) * groups],
                &mut biases[row * groups..(row + 1) * groups],
            )
            .map_err(|e| Error::Config(e.to_string()))?;
    }
    let shape = |a: usize, b: usize| -> Result<[i32; 2]> {
        Ok([
            i32::try_from(a).map_err(|_| Error::Config("Prism row count overflow".into()))?,
            i32::try_from(b).map_err(|_| Error::Config("Prism width overflow".into()))?,
        ])
    };
    Ok(PrismAffineParts {
        weight: Array::from_slice(&words, &shape(rows, width / 16)?),
        scales: Array::from_slice(&scales, &shape(rows, groups)?)
            .as_dtype(mlx_rs::Dtype::Float16)?,
        biases: Array::from_slice(&biases, &shape(rows, groups)?)
            .as_dtype(mlx_rs::Dtype::Float16)?,
        signs: Array::from_slice(
            &signs.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &[i32::try_from(width).map_err(|_| Error::Config("Prism width overflow".into()))?],
        ),
        block: i32::try_from(metadata.block_size)
            .map_err(|_| Error::Config("Prism block size overflow".into()))?,
        role: transform.role,
    })
}

fn required<'a>(file: &'a GgufFile, key: &str) -> Result<&'a MetaValue> {
    file.meta(key)
        .ok_or_else(|| Error::Config(format!("GGUF missing `{key}`")))
}

fn eq_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    if required(file, key)?.as_u64() != Some(expected) {
        return Err(Error::Config(format!("GGUF `{key}` must be {expected}")));
    }
    Ok(())
}

fn eq_str(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    if required(file, key)?.as_str() != Some(expected) {
        return Err(Error::Config(format!("GGUF `{key}` must be `{expected}`")));
    }
    Ok(())
}

fn usize_values(value: &MetaValue) -> Result<Vec<usize>> {
    value
        .as_array()
        .ok_or_else(|| Error::Config("GGUF Prism integer metadata must be an array".into()))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::Config("GGUF Prism width is invalid".into()))
        })
        .collect()
}

fn string_values(value: &MetaValue) -> Result<BTreeSet<String>> {
    value
        .as_array()
        .ok_or_else(|| Error::Config("GGUF Prism names must be an array".into()))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::Config("GGUF Prism name is not a string".into()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{LoadSpec, Message, Sampling, TextLlm, TextLlmRequest};

    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    fn key(out: &mut Vec<u8>, name: &str, ty: u32) {
        string(out, name);
        out.extend_from_slice(&ty.to_le_bytes());
    }
    fn string_meta(out: &mut Vec<u8>, name: &str, value: &str) {
        key(out, name, 8);
        string(out, value);
    }
    fn u32_meta(out: &mut Vec<u8>, name: &str, value: u32) {
        key(out, name, 4);
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn array_header(out: &mut Vec<u8>, name: &str, element: u32, len: usize) {
        key(out, name, 9);
        out.extend_from_slice(&element.to_le_bytes());
        out.extend_from_slice(&(len as u64).to_le_bytes());
    }

    fn one_matrix(kind: PrismPackedKind) -> GgufFile {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&10u64.to_le_bytes());
        string_meta(&mut out, "general.architecture", "qwen35");
        u32_meta(&mut out, "prism.hadamard.version", 1);
        u32_meta(&mut out, "prism.hadamard.block_size", 128);
        string_meta(
            &mut out,
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        );
        string_meta(&mut out, "prism.hadamard.axis", "input-last-dimension");
        string_meta(&mut out, "prism.hadamard.sign_mode", "explicit");
        key(&mut out, "prism.hadamard.gdn_v_grouped", 7);
        out.push(1);
        array_header(&mut out, "prism.hadamard.weight_names", 8, 1);
        string(&mut out, "output.weight");
        array_header(&mut out, "prism.hadamard.inverse_weight_names", 8, 0);
        array_header(&mut out, "prism.hadamard.sign_widths", 4, 1);
        out.extend_from_slice(&128u32.to_le_bytes());
        // sign_values is the final metadata entry, replacing the count above: metadata count is 11.
        // Patch the header count before appending it.
        out[16..24].copy_from_slice(&11u64.to_le_bytes());
        array_header(&mut out, "prism.hadamard.sign_values", 6, 128);
        for _ in 0..128 {
            out.extend_from_slice(&1.0f32.to_le_bytes());
        }
        string(&mut out, "output.weight");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&128u64.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&kind.ggml_type().to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        match kind {
            PrismPackedKind::Pq2_0 => {
                out.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
                out.extend_from_slice(&[0x49; 32]);
            }
            PrismPackedKind::Ptq1_0 => {
                out.extend_from_slice(&(0..26).map(|n| (n * 7) as u8).collect::<Vec<_>>());
                out.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
            }
        }
        GgufFile::parse(out).unwrap()
    }

    #[test]
    fn both_published_gguf_types_transcode_to_native_mlx_parity() {
        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let file = one_matrix(kind);
            let metadata = hadamard_metadata(&file).unwrap();
            let info = &file.tensors[0];
            let linear = linear_from_tensor(&file, info, "output.weight", &metadata).unwrap();
            let x = (0..128)
                .map(|i| (i as f32 - 40.0) / 128.0)
                .collect::<Vec<_>>();
            let got = linear.forward(&Array::from_slice(&x, &[1, 128])).unwrap();
            mlx_rs::transforms::eval([&got]).unwrap();

            let matrix =
                PrismPackedMatrixRef::from_gguf(kind, &[128, 1], file.tensor_data(info).unwrap())
                    .unwrap();
            let mut dense = vec![0.0; 128];
            matrix.decode_row_into(0, &mut dense).unwrap();
            let mut rotated = x.clone();
            core_llm::apply_hadamard_forward_in_place(&mut rotated, &[1; 128], 128, None).unwrap();
            let expected: f32 = rotated.iter().zip(dense).map(|(a, b)| a * b).sum();
            assert!((got.item::<f32>() - expected).abs() < 2e-3, "{kind:?}");
        }
    }

    fn meta_string(name: &str, value: &str) -> Vec<u8> {
        let mut out = Vec::new();
        string_meta(&mut out, name, value);
        out
    }
    fn meta_u32(name: &str, value: u32) -> Vec<u8> {
        let mut out = Vec::new();
        u32_meta(&mut out, name, value);
        out
    }
    fn meta_f32(name: &str, value: f32) -> Vec<u8> {
        let mut out = Vec::new();
        key(&mut out, name, 6);
        out.extend_from_slice(&value.to_le_bytes());
        out
    }
    fn meta_bool(name: &str, value: bool) -> Vec<u8> {
        let mut out = Vec::new();
        key(&mut out, name, 7);
        out.push(value as u8);
        out
    }
    fn meta_strings(name: &str, values: &[String]) -> Vec<u8> {
        let mut out = Vec::new();
        array_header(&mut out, name, 8, values.len());
        for value in values {
            string(&mut out, value);
        }
        out
    }
    fn meta_u32s(name: &str, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        array_header(&mut out, name, 4, values.len());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }
    fn meta_i32s(name: &str, values: &[i32]) -> Vec<u8> {
        let mut out = Vec::new();
        array_header(&mut out, name, 5, values.len());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }
    fn meta_f32s(name: &str, values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        array_header(&mut out, name, 6, values.len());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    struct TinyTensor {
        name: String,
        width: usize,
        rows: usize,
        ty: u32,
        data: Vec<u8>,
    }

    fn packed_data(kind: PrismPackedKind, rows: usize, width: usize) -> Vec<u8> {
        let blocks = rows * width / 128;
        let mut out = Vec::with_capacity(blocks * kind.block_bytes());
        for n in 0..blocks {
            match kind {
                PrismPackedKind::Pq2_0 => {
                    out.extend_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
                    out.extend_from_slice(&[0x55; 32]);
                }
                PrismPackedKind::Ptq1_0 => {
                    out.extend((0..26).map(|i| ((i + n) * 13) as u8));
                    out.extend_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
                }
            }
        }
        out
    }

    fn tiny_provider_file(kind: PrismPackedKind) -> tempfile::NamedTempFile {
        let packed = [
            ("output.weight", 4, 128),
            ("token_embd.weight", 4, 128),
            ("blk.0.attn_q.weight", 256, 128),
            ("blk.0.attn_k.weight", 64, 128),
            ("blk.0.attn_v.weight", 64, 128),
            ("blk.0.attn_output.weight", 128, 128),
            ("blk.0.ffn_gate.weight", 128, 128),
            ("blk.0.ffn_up.weight", 128, 128),
            ("blk.0.ffn_down.weight", 128, 128),
        ];
        let mut tensors = packed
            .iter()
            .map(|&(name, rows, width)| TinyTensor {
                name: name.into(),
                width,
                rows,
                ty: kind.ggml_type(),
                data: packed_data(kind, rows, width),
            })
            .collect::<Vec<_>>();
        for (name, len) in [
            ("output_norm.weight", 128),
            ("blk.0.attn_norm.weight", 128),
            ("blk.0.post_attention_norm.weight", 128),
            ("blk.0.attn_q_norm.weight", 64),
            ("blk.0.attn_k_norm.weight", 64),
        ] {
            tensors.push(TinyTensor {
                name: name.into(),
                width: len,
                rows: 1,
                ty: 0,
                data: vec![0; len * 4],
            });
        }
        let forward = packed
            .iter()
            .filter(|(name, _, _)| *name != "token_embd.weight")
            .map(|(name, _, _)| (*name).to_owned())
            .collect::<Vec<_>>();
        let mut metadata = vec![
            meta_string("general.architecture", "qwen35"),
            meta_u32(
                "general.file_type",
                if kind == PrismPackedKind::Pq2_0 {
                    141
                } else {
                    143
                },
            ),
            meta_u32("prism.hadamard.version", 1),
            meta_u32("prism.hadamard.block_size", 128),
            meta_string(
                "prism.hadamard.transform",
                "normalized-sylvester-walsh-hadamard",
            ),
            meta_string("prism.hadamard.axis", "input-last-dimension"),
            meta_string("prism.hadamard.sign_mode", "explicit"),
            meta_bool("prism.hadamard.gdn_v_grouped", true),
            meta_strings("prism.hadamard.weight_names", &forward),
            meta_strings(
                "prism.hadamard.inverse_weight_names",
                &["token_embd.weight".into()],
            ),
            meta_u32s("prism.hadamard.sign_widths", &[128]),
            meta_f32s("prism.hadamard.sign_values", &[1.0; 128]),
        ];
        for (key_name, value) in [
            ("embedding_length", 128),
            ("feed_forward_length", 128),
            ("block_count", 1),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 64),
            ("context_length", 128),
            ("full_attention_interval", 1),
            ("ssm.time_step_rank", 2),
            ("ssm.group_count", 1),
            ("ssm.inner_size", 128),
            ("ssm.state_size", 64),
            ("ssm.conv_kernel", 4),
            ("rope.dimension_count", 64),
        ] {
            metadata.push(meta_u32(&format!("qwen35.{key_name}"), value));
        }
        metadata.extend([
            meta_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
            meta_f32("qwen35.rope.freq_base", 10_000_000.0),
            meta_u32s("qwen35.rope.dimension_sections", &[11, 11, 10]),
            meta_string("tokenizer.ggml.model", "gpt2"),
            meta_string("tokenizer.ggml.pre", "qwen35"),
            meta_strings(
                "tokenizer.ggml.tokens",
                &[
                    "<unk>".into(),
                    "hello".into(),
                    "world".into(),
                    "<eos>".into(),
                ],
            ),
            meta_strings("tokenizer.ggml.merges", &[]),
            meta_i32s("tokenizer.ggml.token_type", &[3, 1, 1, 3]),
            meta_u32("tokenizer.ggml.bos_token_id", 2),
            meta_u32("tokenizer.ggml.eos_token_id", 3),
            meta_string("tokenizer.chat_template", "<unk>"),
        ]);

        let mut image = Vec::new();
        image.extend_from_slice(b"GGUF");
        image.extend_from_slice(&3u32.to_le_bytes());
        image.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        image.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for entry in metadata {
            image.extend(entry);
        }
        let mut offset = 0usize;
        for tensor in &tensors {
            string(&mut image, &tensor.name);
            let vector = tensor.rows == 1 && tensor.ty == 0;
            image.extend_from_slice(&(if vector { 1u32 } else { 2u32 }).to_le_bytes());
            image.extend_from_slice(&(tensor.width as u64).to_le_bytes());
            if !vector {
                image.extend_from_slice(&(tensor.rows as u64).to_le_bytes());
            }
            image.extend_from_slice(&tensor.ty.to_le_bytes());
            image.extend_from_slice(&(offset as u64).to_le_bytes());
            offset += tensor.data.len();
            offset = offset.next_multiple_of(32);
        }
        while !image.len().is_multiple_of(32) {
            image.push(0);
        }
        for tensor in tensors {
            image.extend(tensor.data);
            while !image.len().is_multiple_of(32) {
                image.push(0);
            }
        }
        let mut file = tempfile::Builder::new().suffix(".gguf").tempfile().unwrap();
        std::io::Write::write_all(&mut file, &image).unwrap();
        file
    }

    #[test]
    fn direct_gguf_provider_loads_and_generates_for_both_packed_types() {
        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let file = tiny_provider_file(kind);
            let spec = LoadSpec::dense(file.path().display().to_string());
            assert!(crate::provider::can_load(&spec));
            let provider = crate::provider::LlamaProvider::load(&spec).unwrap();
            assert!(provider.is_quantized());
            let request = TextLlmRequest {
                messages: vec![Message::user("ignored")],
                sampling: Sampling::greedy(),
                max_new_tokens: 2,
                ..Default::default()
            };
            let output = provider.generate(&request, &mut |_| {}).unwrap();
            assert_eq!(output.usage.generated_tokens, 2, "{kind:?}");
        }
    }
}

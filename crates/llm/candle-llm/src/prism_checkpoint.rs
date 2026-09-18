//! Strict loader for published Prism/Bonsai MLX affine-2 snapshots.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device};
use core_llm::{GdnLayout, PrismHadamardMetadata, PrismPackedKind};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::GdnRowMap;
use crate::primitives::{PrismPackedWeight, PrismRegistry, Weights};
use crate::prism_gguf::{RawGguf, RawValue};

pub(crate) struct PrismMlxCheckpoint {
    pub weights: Weights,
    pub registry: PrismRegistry,
}

pub(crate) struct PrismGgufCheckpoint {
    pub weights: Weights,
    pub registry: PrismRegistry,
    pub config_json: Value,
    pub stop_tokens: Vec<i32>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    tokenizer_json: String,
}

impl PrismGgufCheckpoint {
    pub fn is_prism(path: &Path) -> Result<bool> {
        let raw = RawGguf::open(path)?;
        Ok(raw
            .metadata
            .get("general.architecture")
            .and_then(RawValue::string)
            == Some("qwen35")
            && raw
                .tensors
                .values()
                .any(|tensor| matches!(tensor.ggml_type, 142 | 143)))
    }

    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        let mut raw = RawGguf::open(path)?;
        if raw
            .metadata
            .get("general.architecture")
            .and_then(RawValue::string)
            != Some("qwen35")
        {
            return Err(Error::Config(
                "Prism GGUF requires general.architecture=qwen35".into(),
            ));
        }
        let config_json = gguf_config(&raw)?;
        let (tokenizer_json, stop_tokens, chat_template, bos_token, eos_token) =
            gguf_tokenizer(&raw)?;
        let group_count = meta_u(&raw, "qwen35.ssm.group_count")? as usize;
        let time_step_rank = meta_u(&raw, "qwen35.ssm.time_step_rank")? as usize;
        let inner = meta_u(&raw, "qwen35.ssm.inner_size")? as usize;
        let gdn = GdnLayout::from_ssm_out(inner, time_step_rank, group_count)
            .map_err(|e| Error::Config(format!("Prism GGUF GDN geometry: {e}")))?;
        let metadata = gguf_hadamard(&raw)?;
        let mut dense = HashMap::new();
        let mut packed = HashMap::new();
        let mut names = raw.tensors.keys().cloned().collect::<Vec<_>>();
        names.sort();
        for source_name in names {
            let info = raw
                .tensors
                .get(&source_name)
                .cloned()
                .expect("name cloned from map");
            let target = map_gguf_name(&source_name).ok_or_else(|| {
                Error::Unsupported(format!("unmapped Prism GGUF tensor {source_name}"))
            })?;
            let data = raw.read_tensor(&source_name)?;
            match info.ggml_type {
                142 | 143 => {
                    if info.dimensions.len() != 2 {
                        return Err(Error::Config(format!(
                            "packed GGUF tensor {source_name} is not 2-D"
                        )));
                    }
                    let kind = if info.ggml_type == 142 {
                        PrismPackedKind::Pq2_0
                    } else {
                        PrismPackedKind::Ptq1_0
                    };
                    let row_map = gguf_row_map(&source_name, inner, time_step_rank, group_count);
                    let activation_gdn = source_name.ends_with(".ssm_out.weight").then_some(gdn);
                    let weight = PrismPackedWeight::from_gguf(
                        target.clone(),
                        kind,
                        &info.dimensions,
                        data,
                        &metadata,
                        row_map,
                        activation_gdn,
                        device,
                    )?;
                    packed.insert(target, Arc::new(weight));
                }
                0 | 30 => {
                    let mut values = decode_plain(info.ggml_type, &data)?;
                    let mut shape = info.dimensions.clone();
                    shape.reverse();
                    if let Some(map) =
                        gguf_row_map(&source_name, inner, time_step_rank, group_count)
                    {
                        reorder_dense_rows(&mut values, &shape, map)?;
                    }
                    dense.insert(
                        target,
                        candle_core::Tensor::from_vec(values, shape, device)?,
                    );
                }
                ty => {
                    return Err(Error::Unsupported(format!(
                        "Prism GGUF tensor {source_name} type {ty}"
                    )))
                }
            }
        }
        if packed.len() != metadata.forward_weight_names.len() + metadata.inverse_weight_names.len()
        {
            return Err(Error::Config(format!(
                "Prism GGUF packed count {} does not match transform metadata {}",
                packed.len(),
                metadata.forward_weight_names.len() + metadata.inverse_weight_names.len()
            )));
        }
        Ok(Self {
            weights: Weights::from_map(dense, device.clone()),
            registry: PrismRegistry::new(packed)?,
            config_json,
            stop_tokens,
            chat_template,
            bos_token,
            eos_token,
            tokenizer_json,
        })
    }

    pub fn tokenizer(&self) -> Result<core_llm::Tokenizer> {
        core_llm::Tokenizer::from_json(&self.tokenizer_json)
            .map_err(|e| Error::Config(format!("Prism GGUF tokenizer: {e}")))
    }
}

impl PrismMlxCheckpoint {
    pub fn open(dir: &Path, device: &Device, config: &Value) -> Result<Self> {
        validate_config(config)?;
        let metadata = read_metadata(dir)?;
        let modules = config
            .get("modules")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config("Prism config missing modules array".into()))?;
        let loaded = Weights::from_dir(dir, device)?;
        let mut tensors = loaded.into_map();
        let mut packed = HashMap::with_capacity(modules.len());
        let mut declared = BTreeSet::new();
        for module in modules {
            let path = module
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Config("Prism module missing path".into()))?;
            if !declared.insert(path.to_string()) {
                return Err(Error::Config(format!("duplicate Prism module {path}")));
            }
            if module.get("block").and_then(Value::as_u64) != Some(metadata.block_size as u64)
                || module.get("dtype").and_then(Value::as_str) != Some("float16")
            {
                return Err(Error::Config(format!(
                    "Prism module {path} does not match frozen block/dtype contract"
                )));
            }
            let base = format!("language_model.{path}");
            let weight_key = format!("{base}.weight");
            let scales_key = format!("{base}.scales");
            let biases_key = format!("{base}.biases");
            let words = tensors
                .remove(&weight_key)
                .ok_or_else(|| Error::MissingTensor(weight_key.clone()))?;
            let scales = tensors
                .remove(&scales_key)
                .ok_or_else(|| Error::MissingTensor(scales_key.clone()))?;
            let biases = tensors
                .remove(&biases_key)
                .ok_or_else(|| Error::MissingTensor(biases_key.clone()))?;
            let weight = PrismPackedWeight::from_mlx_affine2(
                weight_key.clone(),
                words,
                scales,
                &biases,
                &metadata,
                None,
                None,
            )?;
            packed.insert(weight_key, Arc::new(weight));
        }
        if packed.len() != metadata.forward_weight_names.len() + metadata.inverse_weight_names.len()
        {
            return Err(Error::Config(format!(
                "Prism module count {} != Hadamard tensor count {}",
                packed.len(),
                metadata.forward_weight_names.len() + metadata.inverse_weight_names.len()
            )));
        }
        if let Some(name) = tensors.keys().find(|name| {
            name.ends_with(".scales")
                || name.ends_with(".biases")
                || tensors
                    .get(*name)
                    .is_some_and(|tensor| tensor.dtype() == DType::U32)
        }) {
            return Err(Error::Config(format!(
                "undeclared Prism packed tensor {name}; refusing a partial load"
            )));
        }
        Ok(Self {
            weights: Weights::from_map(tensors, device.clone()),
            registry: PrismRegistry::new(packed)?,
        })
    }
}

fn validate_config(config: &Value) -> Result<()> {
    let q = config
        .get("quantization")
        .ok_or_else(|| Error::Config("Prism config missing quantization contract".into()))?;
    if config.get("schema_version").and_then(Value::as_u64) != Some(2)
        || config.get("model_type").and_then(Value::as_str) != Some("prism_hadamard_qwen35")
        || q.get("bits").and_then(Value::as_u64) != Some(2)
        || q.get("group_size").and_then(Value::as_u64) != Some(128)
        || q.get("mode").and_then(Value::as_str) != Some("affine")
    {
        return Err(Error::Config(
            "unsupported Prism config (requires schema 2 qwen35 affine-2 group-128)".into(),
        ));
    }
    Ok(())
}

fn read_metadata(dir: &Path) -> Result<PrismHadamardMetadata> {
    let path = dir.join("hadamard.json");
    let value: Value = serde_json::from_slice(&std::fs::read(&path)?)
        .map_err(|error| Error::Config(format!("parse {}: {error}", path.display())))?;
    let exact = |key: &str, expected: &str| -> Result<()> {
        if value.get(key).and_then(Value::as_str) == Some(expected) {
            Ok(())
        } else {
            Err(Error::Config(format!("invalid Prism metadata {key}")))
        }
    };
    if value.get("prism.hadamard.version").and_then(Value::as_u64) != Some(1) {
        return Err(Error::Config(
            "unsupported Prism Hadamard metadata version".into(),
        ));
    }
    exact(
        "prism.hadamard.transform",
        "normalized-sylvester-walsh-hadamard",
    )?;
    exact("prism.hadamard.axis", "input-last-dimension")?;
    exact("prism.hadamard.sign_mode", "explicit")?;
    let block_size = value
        .get("prism.hadamard.block_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Config("Prism metadata missing block size".into()))?
        as usize;
    let names = |key: &str| -> Result<BTreeSet<String>> {
        value
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config(format!("Prism metadata missing {key}")))?
            .iter()
            .map(|item| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    Error::Config(format!("Prism metadata {key} contains a non-string"))
                })
            })
            .collect()
    };
    let widths = value
        .get("prism.hadamard.sign_widths")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("Prism metadata missing sign widths".into()))?;
    let values = value
        .get("prism.hadamard.sign_values")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("Prism metadata missing sign values".into()))?;
    let mut offset = 0usize;
    let mut signs_by_width = BTreeMap::new();
    for width in widths {
        let width = width
            .as_u64()
            .ok_or_else(|| Error::Config("invalid Prism sign width".into()))?
            as usize;
        let end = offset
            .checked_add(width)
            .ok_or_else(|| Error::Config("Prism sign width overflow".into()))?;
        let slice = values
            .get(offset..end)
            .ok_or_else(|| Error::Config("truncated Prism sign values".into()))?;
        let signs = slice
            .iter()
            .map(|sign| {
                sign.as_i64()
                    .and_then(|v| i8::try_from(v).ok())
                    .ok_or_else(|| Error::Config("invalid Prism sign value".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if signs_by_width.insert(width, signs).is_some() {
            return Err(Error::Config(format!("duplicate Prism sign width {width}")));
        }
        offset = end;
    }
    if offset != values.len() {
        return Err(Error::Config("extra Prism sign values".into()));
    }
    let metadata = PrismHadamardMetadata {
        block_size,
        signs_by_width,
        forward_weight_names: names("prism.hadamard.weight_names")?,
        inverse_weight_names: names("prism.hadamard.inverse_weight_names")?,
        gdn_v_grouped: value
            .get("prism.hadamard.gdn_v_grouped")
            .and_then(Value::as_bool)
            .ok_or_else(|| Error::Config("Prism metadata missing gdn_v_grouped".into()))?,
    };
    metadata
        .validate()
        .map_err(|error| Error::Config(format!("Prism Hadamard metadata: {error}")))?;
    Ok(metadata)
}

fn meta_u(raw: &RawGguf, key: &str) -> Result<u64> {
    raw.metadata
        .get(key)
        .and_then(RawValue::u64)
        .ok_or_else(|| Error::Config(format!("Prism GGUF missing integer metadata {key}")))
}

fn meta_f(raw: &RawGguf, key: &str) -> Result<f64> {
    raw.metadata
        .get(key)
        .and_then(RawValue::f64)
        .ok_or_else(|| Error::Config(format!("Prism GGUF missing float metadata {key}")))
}

fn gguf_config(raw: &RawGguf) -> Result<Value> {
    let heads = meta_u(raw, "qwen35.attention.head_count")?;
    let hidden = meta_u(raw, "qwen35.embedding_length")?;
    let head_dim = meta_u(raw, "qwen35.attention.key_length")?;
    let groups = meta_u(raw, "qwen35.ssm.group_count")?;
    let rank = meta_u(raw, "qwen35.ssm.time_step_rank")?;
    let inner = meta_u(raw, "qwen35.ssm.inner_size")?;
    let state = meta_u(raw, "qwen35.ssm.state_size")?;
    if state != inner / rank || !rank.is_multiple_of(groups) {
        return Err(Error::Config(
            "Prism GGUF inconsistent SSM dimensions".into(),
        ));
    }
    let sections = raw
        .metadata
        .get("qwen35.rope.dimension_sections")
        .and_then(RawValue::array)
        .ok_or_else(|| Error::Config("Prism GGUF missing RoPE dimension sections".into()))?;
    if sections.len() < 3 {
        return Err(Error::Config("Prism GGUF needs three RoPE sections".into()));
    }
    let section = |i: usize| {
        sections[i]
            .u64()
            .ok_or_else(|| Error::Config("invalid RoPE section".into()))
    };
    let vocab = raw
        .tensors
        .get("token_embd.weight")
        .and_then(|t| t.dimensions.get(1))
        .copied()
        .ok_or_else(|| Error::Config("Prism GGUF missing token embedding".into()))?;
    Ok(serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": hidden,
            "intermediate_size": meta_u(raw, "qwen35.feed_forward_length")?,
            "num_hidden_layers": meta_u(raw, "qwen35.block_count")?,
            "num_attention_heads": heads,
            "num_key_value_heads": meta_u(raw, "qwen35.attention.head_count_kv")?,
            "head_dim": head_dim, "vocab_size": vocab,
            "rms_norm_eps": meta_f(raw, "qwen35.attention.layer_norm_rms_epsilon")?,
            "full_attention_interval": meta_u(raw, "qwen35.full_attention_interval")?,
            "linear_num_value_heads": rank, "linear_num_key_heads": groups,
            "linear_key_head_dim": state, "linear_value_head_dim": state,
            "linear_conv_kernel_dim": meta_u(raw, "qwen35.ssm.conv_kernel")?,
            "max_position_embeddings": meta_u(raw, "qwen35.context_length")?,
            "partial_rotary_factor": (meta_u(raw, "qwen35.rope.dimension_count")? as f64) / (head_dim as f64),
            "rope_parameters": {"rope_theta": meta_f(raw, "qwen35.rope.freq_base")?, "mrope_section": [section(0)?, section(1)?, section(2)?]},
            "tie_word_embeddings": !raw.tensors.contains_key("output.weight"),
            "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": false
        }
    }))
}

fn gguf_tokenizer(
    raw: &RawGguf,
) -> Result<(
    String,
    Vec<i32>,
    Option<String>,
    Option<String>,
    Option<String>,
)> {
    let string_array = |key: &str| -> Result<Vec<String>> {
        raw.metadata
            .get(key)
            .and_then(RawValue::array)
            .ok_or_else(|| Error::Config(format!("Prism GGUF missing {key}")))?
            .iter()
            .map(|v| {
                v.string()
                    .map(str::to_string)
                    .ok_or_else(|| Error::Config(format!("invalid {key}")))
            })
            .collect()
    };
    let tokens = string_array("tokenizer.ggml.tokens")?;
    let merges = string_array("tokenizer.ggml.merges")?;
    if tokens.is_empty() || merges.is_empty() {
        return Err(Error::Unsupported(
            "Prism GGUF needs embedded BPE tokens and merges".into(),
        ));
    }
    let types = raw
        .metadata
        .get("tokenizer.ggml.token_type")
        .and_then(RawValue::array)
        .map(|a| {
            a.iter()
                .map(|v| match v {
                    RawValue::I(x) => *x as i32,
                    RawValue::U(x) => *x as i32,
                    _ => 1,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (id, token) in tokens.iter().enumerate() {
        vocab.insert(token.clone(), serde_json::json!(id));
        if matches!(types.get(id).copied().unwrap_or(1), 3 | 4) {
            added.push(serde_json::json!({"id":id,"content":token,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}));
        }
    }
    let doc = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,
        "normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
        "post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":vocab,"merges":merges}});
    let id = |key: &str| {
        raw.metadata
            .get(key)
            .and_then(RawValue::u64)
            .and_then(|v| i32::try_from(v).ok())
    };
    let eos = id("tokenizer.ggml.eos_token_id");
    let eot = id("tokenizer.ggml.eot_token_id");
    let mut stop = Vec::new();
    if let Some(v) = eos {
        stop.push(v);
    }
    if let Some(v) = eot {
        if !stop.contains(&v) {
            stop.push(v);
        }
    }
    let token_at = |id: Option<i32>| {
        id.and_then(|v| usize::try_from(v).ok())
            .and_then(|v| tokens.get(v).cloned())
    };
    Ok((
        doc.to_string(),
        stop,
        raw.metadata
            .get("tokenizer.chat_template")
            .and_then(RawValue::string)
            .map(str::to_string),
        token_at(id("tokenizer.ggml.bos_token_id")),
        token_at(eos),
    ))
}

fn gguf_hadamard(raw: &RawGguf) -> Result<PrismHadamardMetadata> {
    let exact = |key: &str, expected: &str| -> Result<()> {
        if raw.metadata.get(key).and_then(RawValue::string) == Some(expected) {
            Ok(())
        } else {
            Err(Error::Config(format!("invalid Prism GGUF metadata {key}")))
        }
    };
    if meta_u(raw, "prism.hadamard.version")? != 1 {
        return Err(Error::Config(
            "unsupported Prism GGUF Hadamard version".into(),
        ));
    }
    exact(
        "prism.hadamard.transform",
        "normalized-sylvester-walsh-hadamard",
    )?;
    exact("prism.hadamard.axis", "input-last-dimension")?;
    exact("prism.hadamard.sign_mode", "explicit")?;
    let list_names = |key: &str| -> Result<BTreeSet<String>> {
        raw.metadata
            .get(key)
            .and_then(RawValue::array)
            .ok_or_else(|| Error::Config(format!("missing {key}")))?
            .iter()
            .map(|v| {
                v.string()
                    .ok_or_else(|| Error::Config(format!("invalid {key}")))
                    .and_then(|name| {
                        map_gguf_name(name)
                            .ok_or_else(|| Error::Config(format!("unmapped transform name {name}")))
                    })
            })
            .collect()
    };
    let widths = raw
        .metadata
        .get("prism.hadamard.sign_widths")
        .and_then(RawValue::array)
        .ok_or_else(|| Error::Config("missing Prism GGUF sign widths".into()))?;
    let values = raw
        .metadata
        .get("prism.hadamard.sign_values")
        .and_then(RawValue::array)
        .ok_or_else(|| Error::Config("missing Prism GGUF signs".into()))?;
    let mut offset = 0usize;
    let mut signs_by_width = BTreeMap::new();
    for width in widths {
        let width = width
            .u64()
            .ok_or_else(|| Error::Config("invalid sign width".into()))?
            as usize;
        let end = offset
            .checked_add(width)
            .ok_or_else(|| Error::Config("sign width overflow".into()))?;
        let signs = values
            .get(offset..end)
            .ok_or_else(|| Error::Config("truncated signs".into()))?
            .iter()
            .map(|v| match v {
                RawValue::I(x) => {
                    i8::try_from(*x).map_err(|_| Error::Config("invalid sign".into()))
                }
                _ => Err(Error::Config("invalid sign".into())),
            })
            .collect::<Result<Vec<_>>>()?;
        signs_by_width.insert(width, signs);
        offset = end;
    }
    if offset != values.len() {
        return Err(Error::Config("extra signs".into()));
    }
    let metadata = PrismHadamardMetadata {
        block_size: meta_u(raw, "prism.hadamard.block_size")? as usize,
        signs_by_width,
        forward_weight_names: list_names("prism.hadamard.weight_names")?,
        inverse_weight_names: list_names("prism.hadamard.inverse_weight_names")?,
        gdn_v_grouped: raw
            .metadata
            .get("prism.hadamard.gdn_v_grouped")
            .and_then(RawValue::bool)
            .ok_or_else(|| Error::Config("missing gdn_v_grouped".into()))?,
    };
    metadata
        .validate()
        .map_err(|e| Error::Config(format!("Prism GGUF Hadamard metadata: {e}")))?;
    Ok(metadata)
}

fn map_gguf_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("language_model.model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("language_model.model.norm.weight".into()),
        "output.weight" => return Some("language_model.lm_head.weight".into()),
        _ => {}
    }
    let (layer, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let target = match suffix {
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_a" => "linear_attn.A_log",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        _ => return None,
    };
    Some(format!("language_model.model.layers.{layer}.{target}"))
}

fn gguf_row_map(name: &str, inner: usize, rank: usize, groups: usize) -> Option<GdnRowMap> {
    let reps = rank / groups;
    let unit = inner / rank;
    if name.ends_with(".attn_qkv.weight") || name.ends_with(".ssm_conv1d.weight") {
        Some(GdnRowMap {
            prefix: 2 * groups * unit,
            groups,
            repetitions: reps,
            unit,
        })
    } else if name.ends_with(".attn_gate.weight")
        || name.ends_with(".ssm_alpha.weight")
        || name.ends_with(".ssm_beta.weight")
    {
        Some(GdnRowMap {
            prefix: 0,
            groups,
            repetitions: reps,
            unit: if name.contains("ssm_alpha") || name.contains("ssm_beta") {
                1
            } else {
                unit
            },
        })
    } else {
        None
    }
}

fn decode_plain(ty: u32, data: &[u8]) -> Result<Vec<f32>> {
    match ty {
        0 if data.len().is_multiple_of(4) => Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()),
        30 if data.len().is_multiple_of(2) => Ok(data
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect()),
        _ => Err(Error::Config("invalid plain GGUF tensor bytes".into())),
    }
}

fn reorder_dense_rows(values: &mut Vec<f32>, shape: &[usize], map: GdnRowMap) -> Result<()> {
    if shape.len() != 2 {
        return Err(Error::Config(
            "GGUF row permutation requires 2-D tensor".into(),
        ));
    }
    let (rows, cols) = (shape[0], shape[1]);
    map.validate(rows)?;
    let old = values.clone();
    for dst in 0..rows {
        let src = map.source_row(dst)?;
        values[dst * cols..(dst + 1) * cols].copy_from_slice(&old[src * cols..(src + 1) * cols]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Tensor;
    use serde_json::json;

    use super::*;

    fn config() -> Value {
        json!({
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "modules": [{"path": "model.embed_tokens", "block": 128, "embedding": true, "dtype": "float16"}]
        })
    }

    fn write_fixture(dir: &Path, bad_bias: bool) {
        let device = Device::Cpu;
        let base = "language_model.model.embed_tokens";
        let mut tensors = HashMap::new();
        tensors.insert(
            format!("{base}.weight"),
            Tensor::zeros((2, 8), DType::U32, &device).unwrap(),
        );
        tensors.insert(
            format!("{base}.scales"),
            Tensor::ones((2, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{base}.biases"),
            Tensor::full(if bad_bias { 0f32 } else { -1f32 }, (2, 1), &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        std::fs::write(
            dir.join("hadamard.json"),
            serde_json::to_vec(&json!({
                "prism.hadamard.version": 1,
                "prism.hadamard.block_size": 128,
                "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
                "prism.hadamard.axis": "input-last-dimension",
                "prism.hadamard.sign_mode": "explicit",
                "prism.hadamard.weight_names": [],
                "prism.hadamard.inverse_weight_names": [format!("{base}.weight")],
                "prism.hadamard.sign_widths": [128],
                "prism.hadamard.sign_values": vec![1; 128],
                "prism.hadamard.gdn_v_grouped": true
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn strict_mlx_checkpoint_splits_packed_from_auxiliary_weights() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), false);
        let checkpoint = PrismMlxCheckpoint::open(dir.path(), &Device::Cpu, &config()).unwrap();
        assert_eq!(checkpoint.registry.len(), 1);
        assert!(checkpoint
            .registry
            .contains("language_model.model.embed_tokens.weight"));
        assert!(checkpoint.weights.is_empty());
    }

    #[test]
    fn strict_mlx_checkpoint_rejects_malformed_affine_metadata() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), true);
        let error = match PrismMlxCheckpoint::open(dir.path(), &Device::Cpu, &config()) {
            Ok(_) => panic!("malformed affine metadata was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("biases exactly equal"));
    }

    #[test]
    fn compact_checkpoint_executes_qwen_prefill_and_cached_decode() {
        use crate::models::{Qwen35Config, Qwen35Model};

        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let projection_paths = [
            "lm_head",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ];
        let mut modules = projection_paths
            .iter()
            .map(|path| json!({"path": path, "block": 128, "embedding": false, "dtype": "float16"}))
            .collect::<Vec<_>>();
        modules.push(json!({"path": "model.embed_tokens", "block": 128, "embedding": true, "dtype": "float16"}));
        let config = json!({
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "modules": modules,
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": 128, "intermediate_size": 128,
                "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                "head_dim": 128, "vocab_size": 128, "rms_norm_eps": 1e-6,
                "full_attention_interval": 1, "linear_num_value_heads": 1,
                "linear_num_key_heads": 1, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "partial_rotary_factor": 0.25, "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": false
            }
        });
        let mut tensors = HashMap::new();
        for path in projection_paths {
            let rows = if path.ends_with("q_proj") { 256 } else { 128 };
            let base = format!("language_model.{path}");
            tensors.insert(
                format!("{base}.weight"),
                Tensor::zeros((rows, 8), DType::U32, &device).unwrap(),
            );
            tensors.insert(
                format!("{base}.scales"),
                Tensor::ones((rows, 1), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{base}.biases"),
                Tensor::full(-1f32, (rows, 1), &device).unwrap(),
            );
        }
        let embed = "language_model.model.embed_tokens";
        tensors.insert(
            format!("{embed}.weight"),
            Tensor::zeros((128, 8), DType::U32, &device).unwrap(),
        );
        tensors.insert(
            format!("{embed}.scales"),
            Tensor::ones((128, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{embed}.biases"),
            Tensor::full(-1f32, (128, 1), &device).unwrap(),
        );
        for name in [
            "language_model.model.norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.post_attention_layernorm.weight",
            "language_model.model.layers.0.self_attn.q_norm.weight",
            "language_model.model.layers.0.self_attn.k_norm.weight",
        ] {
            tensors.insert(
                name.into(),
                Tensor::zeros((128,), DType::F32, &device).unwrap(),
            );
        }
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
        let forward = projection_paths
            .iter()
            .map(|path| format!("language_model.{path}.weight"))
            .collect::<Vec<_>>();
        std::fs::write(dir.path().join("hadamard.json"), serde_json::to_vec(&json!({
            "prism.hadamard.version": 1, "prism.hadamard.block_size": 128,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension", "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": forward,
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [128], "prism.hadamard.sign_values": vec![1; 128],
            "prism.hadamard.gdn_v_grouped": true
        })).unwrap()).unwrap();
        let checkpoint = PrismMlxCheckpoint::open(dir.path(), &device, &config).unwrap();
        let qcfg = Qwen35Config::from_json(&config).unwrap();
        let model = Qwen35Model::from_prism_weights(
            &checkpoint.weights,
            "language_model.model",
            qcfg,
            &checkpoint.registry,
            DType::F32,
        )
        .unwrap();
        let mut cache = model.new_cache();
        let prefill = Tensor::from_vec(vec![1i64, 2], (1, 2), &device).unwrap();
        let logits = model.decode_logits(&prefill, &mut cache, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 128]);
        assert!(logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
        let next = Tensor::from_vec(vec![3i64], (1, 1), &device).unwrap();
        assert_eq!(
            model.decode_logits(&next, &mut cache, 2).unwrap().dims(),
            &[1, 128]
        );
    }

    #[test]
    fn gguf_name_and_grouped_row_mapping_match_qwen_layout() {
        assert_eq!(
            map_gguf_name("blk.7.ssm_out.weight").as_deref(),
            Some("language_model.model.layers.7.linear_attn.out_proj.weight")
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_output.weight").as_deref(),
            Some("language_model.model.layers.3.self_attn.o_proj.weight")
        );
        let map = GdnRowMap {
            prefix: 0,
            groups: 2,
            repetitions: 3,
            unit: 1,
        };
        let mut values = vec![0., 1., 2., 3., 4., 5.];
        reorder_dense_rows(&mut values, &[6, 1], map).unwrap();
        assert_eq!(values, vec![0., 2., 4., 1., 3., 5.]);
    }
}

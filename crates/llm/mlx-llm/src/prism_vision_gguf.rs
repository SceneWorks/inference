//! Native loader for Prism's separable Qwen3-VL GGUF projector.
//!
//! The language GGUF and projector GGUF are independent artifacts. The provider opens this module
//! only for the exact `LoadSpec::projector_source` selected by the caller; it never guesses between
//! adjacent BF16 and Q8_0 projector variants.

use std::collections::{HashMap, HashSet};

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use crate::error::{Error, Result};
use crate::gguf::{dequantize, GgufFile, TensorInfo};
use crate::models::Qwen35VisionConfig;
use crate::primitives::Weights;

const DEPTH: usize = 27;
const ALLOWED_TYPES: &[u32] = &[0, 1, 8, 30]; // F32, F16, Q8_0, BF16.

pub(crate) struct LoadedPrismVisionGguf {
    pub(crate) weights: Weights,
    pub(crate) config: Qwen35VisionConfig,
}

pub(crate) fn load(file: &GgufFile) -> Result<LoadedPrismVisionGguf> {
    validate(file)?;
    let mut tensors = HashMap::with_capacity(file.tensors.len() - 1);
    let mut seen = HashSet::with_capacity(file.tensors.len());
    let mut patch = [None, None];

    for info in &file.tensors {
        if !seen.insert(info.name.as_str()) {
            return Err(Error::Config(format!(
                "Prism projector has duplicate tensor `{}`",
                info.name
            )));
        }
        if !ALLOWED_TYPES.contains(&info.ggml_type) {
            return Err(Error::Unsupported(format!(
                "Prism projector tensor `{}` uses unsupported GGML type {}",
                info.name, info.ggml_type
            )));
        }
        let expected = projector_shape(&info.name).ok_or_else(|| {
            Error::Config(format!("unexpected Prism projector tensor `{}`", info.name))
        })?;
        if info.shape != expected {
            return Err(Error::Config(format!(
                "Prism projector tensor `{}` shape {:?}, expected {expected:?}",
                info.name, info.shape
            )));
        }
        let array = decode_tensor(file, info)?;
        match info.name.as_str() {
            "v.patch_embd.weight" => patch[0] = Some(array),
            "v.patch_embd.weight.1" => patch[1] = Some(array),
            name => {
                let key = projector_key(name).ok_or_else(|| {
                    Error::Config(format!("unexpected Prism projector tensor `{name}`"))
                })?;
                if tensors.insert(key, array).is_some() {
                    return Err(Error::Config(format!(
                        "Prism projector maps multiple tensors to `{name}`"
                    )));
                }
            }
        }
    }

    let [first, second] = patch
        .map(|part| part.ok_or_else(|| Error::MissingTensor("v.patch_embd.weight[.1]".into())));
    let first = first?.reshape(&[1152, 3, 1, 16, 16])?;
    let second = second?.reshape(&[1152, 3, 1, 16, 16])?;
    let patch = concatenate_axis(&[&first, &second], 2)?;
    tensors.insert("vision_tower.patch_embed.proj.weight".into(), patch);

    // 27 blocks × 12 tensors + 6 merger + patch weight/bias + position = 333 HF tensors.
    if tensors.len() != 333 {
        return Err(Error::Config(format!(
            "Prism projector mapped {} tensors, expected 333",
            tensors.len()
        )));
    }

    Ok(LoadedPrismVisionGguf {
        weights: Weights::from_map(tensors),
        config: Qwen35VisionConfig {
            depth: DEPTH,
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            in_channels: 3,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 5120,
            num_position_embeddings: 2304,
            deepstack_visual_indexes: Vec::new(),
        },
    })
}

/// Validate the projector header and tensor table without reading or expanding tensor payloads.
pub(crate) fn validate(file: &GgufFile) -> Result<()> {
    validate_metadata(file)?;
    let mut type_counts = std::collections::BTreeMap::<u32, usize>::new();
    let mut seen = HashSet::with_capacity(file.tensors.len());
    for info in &file.tensors {
        *type_counts.entry(info.ggml_type).or_default() += 1;
        if !seen.insert(info.name.as_str()) {
            return Err(Error::Config(format!(
                "Prism projector has duplicate tensor `{}`",
                info.name
            )));
        }
        let expected = projector_shape(&info.name).ok_or_else(|| {
            Error::Config(format!("unexpected Prism projector tensor `{}`", info.name))
        })?;
        if info.shape != expected {
            return Err(Error::Config(format!(
                "Prism projector tensor `{}` shape {:?}, expected {expected:?}",
                info.name, info.shape
            )));
        }
    }
    let bf16 = [(0, 224), (30, 110)].into_iter().collect();
    let q8 = [(0, 224), (1, 27), (8, 83)].into_iter().collect();
    let expected_file_type = if type_counts == bf16 {
        32
    } else if type_counts == q8 {
        7
    } else {
        return Err(Error::Config(format!(
            "Prism projector tensor-type inventory {type_counts:?} is neither the published BF16 \
             layout {{0: 224, 30: 110}} nor Q8_0 layout {{0: 224, 1: 27, 8: 83}}"
        )));
    };
    expect_u64(file, "general.file_type", expected_file_type)?;
    if file.tensors.len() != 334 {
        return Err(Error::Config(format!(
            "Prism projector contains {} tensors, expected 334",
            file.tensors.len()
        )));
    }
    Ok(())
}

fn decode_tensor(file: &GgufFile, info: &TensorInfo) -> Result<Array> {
    let values = dequantize(info.ggml_type, file.tensor_data(info)?, info.num_elements())?;
    let shape = info
        .shape
        .iter()
        .map(|&dim| {
            i32::try_from(dim).map_err(|_| Error::Config("projector dimension overflow".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Array::from_slice(&values, &shape))
}

fn validate_metadata(file: &GgufFile) -> Result<()> {
    expect_str(file, "general.architecture", "clip")?;
    expect_str(file, "general.type", "mmproj")?;
    expect_str(file, "clip.projector_type", "qwen3vl_merger")?;
    expect_bool(file, "clip.has_vision_encoder", true)?;
    expect_bool(file, "clip.use_gelu", true)?;
    expect_u64(file, "clip.vision.block_count", 27)?;
    expect_u64(file, "clip.vision.embedding_length", 1152)?;
    expect_u64(file, "clip.vision.attention.head_count", 16)?;
    expect_f64(file, "clip.vision.attention.layer_norm_epsilon", 1.0e-6)?;
    expect_u64(file, "clip.vision.feed_forward_length", 4304)?;
    expect_u64(file, "clip.vision.projection_dim", 5120)?;
    expect_u64(file, "clip.vision.image_size", 768)?;
    expect_f64_array(file, "clip.vision.image_mean", &[0.5, 0.5, 0.5])?;
    expect_f64_array(file, "clip.vision.image_std", &[0.5, 0.5, 0.5])?;
    expect_u64(file, "clip.vision.patch_size", 16)?;
    expect_u64(file, "clip.vision.spatial_merge_size", 2)?;
    let deepstack = file
        .meta("clip.vision.is_deepstack_layers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Config("Prism projector missing deepstack metadata".into()))?;
    if deepstack.len() != DEPTH || deepstack.iter().any(|v| v.as_bool() != Some(false)) {
        return Err(Error::Config(
            "Prism projector deepstack metadata must contain 27 false entries".into(),
        ));
    }
    Ok(())
}

fn expect_str(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    if file.meta_str(key) != Some(expected) {
        return Err(Error::Config(format!(
            "Prism projector `{key}` must be `{expected}`"
        )));
    }
    Ok(())
}

fn expect_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    if file.meta_u64(key) != Some(expected) {
        return Err(Error::Config(format!(
            "Prism projector `{key}` must be {expected}"
        )));
    }
    Ok(())
}

fn expect_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    if file.meta(key).and_then(|v| v.as_bool()) != Some(expected) {
        return Err(Error::Config(format!(
            "Prism projector `{key}` must be {expected}"
        )));
    }
    Ok(())
}

fn expect_f64(file: &GgufFile, key: &str, expected: f64) -> Result<()> {
    let actual = file.meta(key).and_then(|v| v.as_f64());
    if actual.is_none_or(|value| (value - expected).abs() > 1.0e-12) {
        return Err(Error::Config(format!(
            "Prism projector `{key}` must be {expected}"
        )));
    }
    Ok(())
}

fn expect_f64_array(file: &GgufFile, key: &str, expected: &[f64]) -> Result<()> {
    let valid = file
        .meta(key)
        .and_then(|v| v.as_array())
        .is_some_and(|values| {
            values.len() == expected.len()
                && values.iter().zip(expected).all(|(value, expected)| {
                    value
                        .as_f64()
                        .is_some_and(|value| (value - expected).abs() <= f32::EPSILON as f64)
                })
        });
    if !valid {
        return Err(Error::Config(format!(
            "Prism projector `{key}` does not match the published preprocessing vector"
        )));
    }
    Ok(())
}

fn projector_key(name: &str) -> Option<String> {
    let leaf = match name {
        "v.patch_embd.bias" => "patch_embed.proj.bias".to_owned(),
        "v.position_embd.weight" => "pos_embed.weight".to_owned(),
        "v.post_ln.weight" => "merger.norm.weight".to_owned(),
        "v.post_ln.bias" => "merger.norm.bias".to_owned(),
        "mm.0.weight" => "merger.linear_fc1.weight".to_owned(),
        "mm.0.bias" => "merger.linear_fc1.bias".to_owned(),
        "mm.2.weight" => "merger.linear_fc2.weight".to_owned(),
        "mm.2.bias" => "merger.linear_fc2.bias".to_owned(),
        _ => {
            let rest = name.strip_prefix("v.blk.")?;
            let (layer, tail) = rest.split_once('.')?;
            let _: usize = layer.parse().ok()?;
            let mapped = match tail {
                "attn_out.weight" => "attn.proj.weight",
                "attn_out.bias" => "attn.proj.bias",
                "attn_qkv.weight" => "attn.qkv.weight",
                "attn_qkv.bias" => "attn.qkv.bias",
                "ffn_up.weight" => "mlp.linear_fc1.weight",
                "ffn_up.bias" => "mlp.linear_fc1.bias",
                "ffn_down.weight" => "mlp.linear_fc2.weight",
                "ffn_down.bias" => "mlp.linear_fc2.bias",
                "ln1.weight" => "norm1.weight",
                "ln1.bias" => "norm1.bias",
                "ln2.weight" => "norm2.weight",
                "ln2.bias" => "norm2.bias",
                _ => return None,
            };
            format!("blocks.{layer}.{mapped}")
        }
    };
    Some(format!("vision_tower.{leaf}"))
}

fn projector_shape(name: &str) -> Option<Vec<usize>> {
    let shape = match name {
        "v.patch_embd.weight" | "v.patch_embd.weight.1" => vec![1152, 3, 16, 16],
        "v.patch_embd.bias" => vec![1152],
        "v.position_embd.weight" => vec![2304, 1152],
        "v.post_ln.weight" | "v.post_ln.bias" => vec![1152],
        "mm.0.weight" => vec![4608, 4608],
        "mm.0.bias" => vec![4608],
        "mm.2.weight" => vec![5120, 4608],
        "mm.2.bias" => vec![5120],
        _ => {
            let rest = name.strip_prefix("v.blk.")?;
            let (layer, tail) = rest.split_once('.')?;
            if layer.parse::<usize>().ok()? >= DEPTH {
                return None;
            }
            match tail {
                "attn_out.weight" => vec![1152, 1152],
                "attn_out.bias" => vec![1152],
                "attn_qkv.weight" => vec![3456, 1152],
                "attn_qkv.bias" => vec![3456],
                "ffn_up.weight" => vec![4304, 1152],
                "ffn_up.bias" => vec![4304],
                "ffn_down.weight" => vec![1152, 4304],
                "ffn_down.bias" => vec![1152],
                "ln1.weight" | "ln1.bias" | "ln2.weight" | "ln2.bias" => vec![1152],
                _ => return None,
            }
        }
    };
    Some(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_projector_tensor_family_without_aliasing_patch_halves() {
        assert_eq!(
            projector_key("v.blk.26.attn_qkv.weight").as_deref(),
            Some("vision_tower.blocks.26.attn.qkv.weight")
        );
        assert_eq!(
            projector_key("v.blk.0.ffn_down.bias").as_deref(),
            Some("vision_tower.blocks.0.mlp.linear_fc2.bias")
        );
        assert_eq!(
            projector_key("mm.2.weight").as_deref(),
            Some("vision_tower.merger.linear_fc2.weight")
        );
        assert!(projector_key("v.patch_embd.weight").is_none());
        assert!(projector_key("v.patch_embd.weight.1").is_none());
        assert!(projector_key("unknown.weight").is_none());
        assert_eq!(
            projector_shape("v.blk.0.ffn_down.weight"),
            Some(vec![1152, 4304])
        );
        assert!(projector_shape("v.blk.27.ln1.weight").is_none());
    }
}

//! Strict loader for the qwen3vl_merger vision projector paired with Bonsai GGUF language files.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::models::{Qwen35VisionConfig, Qwen35VisionModel};
use crate::primitives::Weights;

const PREFIX: &str = "vision_tower";

/// Fully loaded native GGUF vision sidecar. Matrix Q8_0 tensors remain compact in the returned
/// model; only scalar/norm/bias tables and the small patch/position tensors are dense.
pub(crate) struct PrismVisionGguf {
    pub model: Qwen35VisionModel,
    pub config: Qwen35VisionConfig,
}

impl PrismVisionGguf {
    pub fn open(path: &Path, device: &Device, language_hidden_size: usize) -> Result<Self> {
        let mut file = File::open(path)?;
        let content = Content::read(&mut file)?;
        let cfg = validate_metadata(&content, language_hidden_size)?;
        let expected = expected_tensors(&cfg);
        let actual = content
            .tensor_infos
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
        if actual != expected_names {
            let missing = expected_names
                .difference(&actual)
                .cloned()
                .collect::<Vec<_>>();
            let extra = actual
                .difference(&expected_names)
                .cloned()
                .collect::<Vec<_>>();
            return Err(Error::Config(format!(
                "Bonsai mmproj tensor set mismatch (missing {missing:?}, extra {extra:?})"
            )));
        }

        let file_type = meta_u(&content, "general.file_type")?;
        if !matches!(file_type, 7 | 32) {
            return Err(Error::Config(format!(
                "Bonsai mmproj file_type {file_type} unsupported (expected Q8_0=7 or BF16=32)"
            )));
        }
        let mut dense = HashMap::new();
        let mut packed = HashMap::new();
        for (source, spec) in &expected {
            if source.starts_with("v.patch_embd.weight") {
                continue;
            }
            let info = content
                .tensor_infos
                .get(source)
                .expect("tensor set equality checked");
            if info.shape.dims() != spec.shape.as_slice() {
                return Err(Error::Config(format!(
                    "Bonsai mmproj tensor {source} shape {:?}, expected {:?}",
                    info.shape.dims(),
                    spec.shape
                )));
            }
            validate_dtype(source, info.ggml_dtype, spec.matrix, file_type)?;
            let q = content.tensor(&mut file, source, device)?;
            if spec.matrix && info.ggml_dtype == GgmlDType::Q8_0 {
                packed.insert(spec.target.clone(), q);
            } else {
                dense.insert(
                    spec.target.clone(),
                    q.dequantize(device)?.to_dtype(DType::F32)?,
                );
            }
        }

        // llama.cpp serializes the two temporal Conv3d slices separately as `[out,C,H,W]`.
        // Stack at T to recover the upstream `[out,C,T,H,W]` flatten order.
        let patch0 = read_patch_half(&content, &mut file, "v.patch_embd.weight", &cfg, device)?;
        let patch1 = read_patch_half(&content, &mut file, "v.patch_embd.weight.1", &cfg, device)?;
        dense.insert(
            format!("{PREFIX}.patch_embed.proj.weight"),
            Tensor::stack(&[&patch0, &patch1], 2)?,
        );

        let weights = Weights::from_map(dense, device.clone());
        let model =
            Qwen35VisionModel::from_gguf_weights(&weights, PREFIX, cfg.clone(), &mut packed)?;
        Ok(Self { model, config: cfg })
    }
}

#[derive(Clone)]
struct TensorSpec {
    target: String,
    shape: Vec<usize>,
    matrix: bool,
}

fn insert(
    out: &mut BTreeMap<String, TensorSpec>,
    source: impl Into<String>,
    target: impl Into<String>,
    shape: impl Into<Vec<usize>>,
    matrix: bool,
) {
    out.insert(
        source.into(),
        TensorSpec {
            target: target.into(),
            shape: shape.into(),
            matrix,
        },
    );
}

fn expected_tensors(cfg: &Qwen35VisionConfig) -> BTreeMap<String, TensorSpec> {
    let h = cfg.hidden_size as usize;
    let inter = cfg.intermediate_size as usize;
    let merge = cfg.merge_dim() as usize;
    let out_h = cfg.out_hidden_size as usize;
    let mut out = BTreeMap::new();
    for i in 0..cfg.depth {
        let src = |leaf: &str| format!("v.blk.{i}.{leaf}");
        let dst = |leaf: &str| format!("{PREFIX}.blocks.{i}.{leaf}");
        for (s, d, rows, cols) in [
            ("attn_qkv", "attn.qkv", 3 * h, h),
            ("attn_out", "attn.proj", h, h),
            ("ffn_up", "mlp.linear_fc1", inter, h),
            ("ffn_down", "mlp.linear_fc2", h, inter),
        ] {
            insert(
                &mut out,
                src(&format!("{s}.weight")),
                dst(&format!("{d}.weight")),
                vec![rows, cols],
                true,
            );
            insert(
                &mut out,
                src(&format!("{s}.bias")),
                dst(&format!("{d}.bias")),
                vec![rows],
                false,
            );
        }
        for (s, d) in [("ln1", "norm1"), ("ln2", "norm2")] {
            insert(
                &mut out,
                src(&format!("{s}.weight")),
                dst(&format!("{d}.weight")),
                vec![h],
                false,
            );
            insert(
                &mut out,
                src(&format!("{s}.bias")),
                dst(&format!("{d}.bias")),
                vec![h],
                false,
            );
        }
    }
    insert(
        &mut out,
        "mm.0.weight",
        format!("{PREFIX}.merger.linear_fc1.weight"),
        vec![merge, merge],
        true,
    );
    insert(
        &mut out,
        "mm.0.bias",
        format!("{PREFIX}.merger.linear_fc1.bias"),
        vec![merge],
        false,
    );
    insert(
        &mut out,
        "mm.2.weight",
        format!("{PREFIX}.merger.linear_fc2.weight"),
        vec![out_h, merge],
        true,
    );
    insert(
        &mut out,
        "mm.2.bias",
        format!("{PREFIX}.merger.linear_fc2.bias"),
        vec![out_h],
        false,
    );
    insert(
        &mut out,
        "v.post_ln.weight",
        format!("{PREFIX}.merger.norm.weight"),
        vec![h],
        false,
    );
    insert(
        &mut out,
        "v.post_ln.bias",
        format!("{PREFIX}.merger.norm.bias"),
        vec![h],
        false,
    );
    insert(
        &mut out,
        "v.position_embd.weight",
        format!("{PREFIX}.pos_embed.weight"),
        vec![cfg.num_position_embeddings as usize, h],
        false,
    );
    insert(
        &mut out,
        "v.patch_embd.bias",
        format!("{PREFIX}.patch_embed.proj.bias"),
        vec![h],
        false,
    );
    let half = vec![
        h,
        cfg.in_channels as usize,
        cfg.patch_size as usize,
        cfg.patch_size as usize,
    ];
    insert(&mut out, "v.patch_embd.weight", "", half.clone(), false);
    insert(&mut out, "v.patch_embd.weight.1", "", half, false);
    out
}

fn read_patch_half(
    content: &Content,
    file: &mut File,
    name: &str,
    cfg: &Qwen35VisionConfig,
    device: &Device,
) -> Result<Tensor> {
    let info = content.tensor_infos.get(name).expect("tensor set checked");
    let expected = [
        cfg.hidden_size as usize,
        cfg.in_channels as usize,
        cfg.patch_size as usize,
        cfg.patch_size as usize,
    ];
    if info.shape.dims() != expected || info.ggml_dtype != GgmlDType::F32 {
        return Err(Error::Config(format!(
            "Bonsai mmproj tensor {name} must be F32 {:?}, got {:?} {:?}",
            expected,
            info.ggml_dtype,
            info.shape.dims()
        )));
    }
    Ok(content
        .tensor(file, name, device)?
        .dequantize(device)?
        .to_dtype(DType::F32)?)
}

fn validate_dtype(name: &str, dtype: GgmlDType, matrix: bool, file_type: u64) -> Result<()> {
    let valid = if !matrix {
        dtype == GgmlDType::F32
    } else if file_type == 32 {
        dtype == GgmlDType::BF16
    } else {
        matches!(dtype, GgmlDType::Q8_0 | GgmlDType::F16)
    };
    if valid {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "Bonsai mmproj tensor {name} has invalid {dtype:?} for file_type {file_type}"
        )))
    }
}

fn meta<'a>(content: &'a Content, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| Error::Config(format!("Bonsai mmproj missing metadata {key}")))
}

fn meta_u(content: &Content, key: &str) -> Result<u64> {
    meta(content, key)?
        .to_u64()
        .map_err(|_| Error::Config(format!("Bonsai mmproj metadata {key} is not unsigned")))
}

fn meta_bool(content: &Content, key: &str) -> Result<bool> {
    meta(content, key)?
        .to_bool()
        .map_err(|_| Error::Config(format!("Bonsai mmproj metadata {key} is not boolean")))
}

fn meta_str<'a>(content: &'a Content, key: &str) -> Result<&'a str> {
    meta(content, key)?
        .to_string()
        .map(String::as_str)
        .map_err(|_| Error::Config(format!("Bonsai mmproj metadata {key} is not a string")))
}

fn value_f32(value: &Value) -> Option<f32> {
    match value {
        Value::F32(v) => Some(*v),
        Value::F64(v) => Some(*v as f32),
        _ => None,
    }
}

fn meta_f32(content: &Content, key: &str) -> Result<f32> {
    value_f32(meta(content, key)?)
        .ok_or_else(|| Error::Config(format!("Bonsai mmproj metadata {key} is not numeric")))
}

fn meta_f32_array(content: &Content, key: &str) -> Result<Vec<f32>> {
    meta(content, key)?
        .to_vec()
        .map_err(|_| Error::Config(format!("Bonsai mmproj metadata {key} is not an array")))?
        .iter()
        .map(|v| {
            value_f32(v)
                .ok_or_else(|| Error::Config(format!("Bonsai mmproj metadata {key} is not numeric")))
        })
        .collect()
}

fn validate_metadata(content: &Content, language_hidden_size: usize) -> Result<Qwen35VisionConfig> {
    if meta_str(content, "general.architecture")? != "clip"
        || meta_str(content, "general.type")? != "mmproj"
        || !meta_bool(content, "clip.has_vision_encoder")?
        || meta_str(content, "clip.projector_type")? != "qwen3vl_merger"
        || !meta_bool(content, "clip.use_gelu")?
    {
        return Err(Error::Config(
            "projector is not a CLIP qwen3vl_merger vision sidecar".into(),
        ));
    }
    let depth = meta_u(content, "clip.vision.block_count")? as usize;
    let hidden = meta_u(content, "clip.vision.embedding_length")? as usize;
    let heads = meta_u(content, "clip.vision.attention.head_count")? as usize;
    let intermediate = meta_u(content, "clip.vision.feed_forward_length")? as usize;
    let image_size = meta_u(content, "clip.vision.image_size")? as usize;
    let patch = meta_u(content, "clip.vision.patch_size")? as usize;
    let merge = meta_u(content, "clip.vision.spatial_merge_size")? as usize;
    let projection = meta_u(content, "clip.vision.projection_dim")? as usize;
    let eps = meta_f32(content, "clip.vision.attention.layer_norm_epsilon")?;
    let mean = meta_f32_array(content, "clip.vision.image_mean")?;
    let std = meta_f32_array(content, "clip.vision.image_std")?;
    if depth == 0
        || hidden == 0
        || heads == 0
        || !hidden.is_multiple_of(heads)
        || intermediate == 0
        || patch == 0
        || image_size == 0
        || !image_size.is_multiple_of(patch)
        || merge == 0
        || projection != language_hidden_size
        || (eps - 1e-6).abs() > 1e-10
        || mean != [0.5, 0.5, 0.5]
        || std != [0.5, 0.5, 0.5]
    {
        return Err(Error::Config(
            "Bonsai mmproj geometry is inconsistent with its language model".into(),
        ));
    }
    let deepstack = meta(content, "clip.vision.is_deepstack_layers")?
        .to_vec()
        .map_err(|_| Error::Config("clip.vision.is_deepstack_layers is not an array".into()))?;
    if deepstack.len() != depth
        || deepstack
            .iter()
            .any(|v| v.to_bool().ok().is_none_or(|enabled| enabled))
    {
        return Err(Error::Config(
            "Bonsai qwen3vl_merger must declare one false DeepStack flag per block".into(),
        ));
    }
    let side = image_size / patch;
    Ok(Qwen35VisionConfig {
        depth,
        hidden_size: hidden as i32,
        num_heads: heads as i32,
        intermediate_size: intermediate as i32,
        in_channels: 3,
        patch_size: patch as i32,
        temporal_patch_size: 2,
        spatial_merge_size: merge as i32,
        out_hidden_size: projection as i32,
        num_position_embeddings: (side * side) as i32,
        deepstack_visual_indexes: Vec::new(),
    })
}

//! Shared forward-time LoRA/LoKr installer for DiTs whose adapter keys resolve to the same dotted
//! paths exposed by their [`AdaptLinear`] visitor. The frozen base remains dense or packed; adapters
//! are stacked as additive residuals. Family-specific visitors stay in their provider crates.

use std::collections::{BTreeMap, HashSet};

use candle_core::{DType, Device, Result, Tensor};
use gen_core::weightsmeta as wmeta;
use gen_core::{AdapterKind, AdapterSpec};

use crate::train::lora::{parse_lokr_metadata, LoraAdapterMeta};
use crate::train::merge::{read_adapter, read_scalar, AdapterFile, LoraTriple, Role};
use crate::{CandleError, Result as CResult};

use super::{AdaptLinear, LokrFactors};

const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

#[derive(Debug, Default, PartialEq, Eq)]
pub struct AdditiveAdapterReport {
    pub applied: usize,
    pub skipped_targets: Vec<String>,
    pub skipped_keys: usize,
}

struct PendingLora {
    a: Tensor,
    b: Tensor,
    scale: f64,
    source: usize,
}

struct PendingLokr {
    w1: Option<Tensor>,
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
    scale: f64,
    source: usize,
}

#[derive(Clone, Copy)]
enum UpSlice {
    Chunk { count: usize, index: usize },
    Range { start: usize, len: usize },
}

#[derive(Clone)]
struct LoraCandidate {
    key: String,
    up_slice: Option<UpSlice>,
    down_chunks: Option<(usize, usize)>,
}

fn direct_candidate(key: impl Into<String>) -> LoraCandidate {
    LoraCandidate {
        key: key.into(),
        up_slice: None,
        down_chunks: None,
    }
}

fn split_candidate(
    key: impl Into<String>,
    up_slice: UpSlice,
    count: usize,
    index: usize,
) -> LoraCandidate {
    LoraCandidate {
        key: key.into(),
        up_slice: Some(up_slice),
        down_chunks: Some((count, index)),
    }
}

fn block_parts<'a>(path: &'a str, root: &str) -> Option<(usize, &'a str)> {
    let rest = path.strip_prefix(root)?.strip_prefix('.')?;
    let (index, suffix) = rest.split_once('.')?;
    Some((index.parse().ok()?, suffix))
}

/// Family aliases whose source checkpoint keeps BFL/ComfyUI fused projections while the host uses
/// diffusers split projections. This mirrors the MLX adapter router for FLUX.1/2.
fn bfl_candidates(family: &str, path: &str, out_features: usize) -> Vec<LoraCandidate> {
    let mut out = Vec::new();
    if family.starts_with("flux2") {
        for (source, target) in [
            ("img_in", "x_embedder"),
            ("txt_in", "context_embedder"),
            (
                "time_in.in_layer",
                "time_guidance_embed.timestep_embedder.linear_1",
            ),
            (
                "time_in.out_layer",
                "time_guidance_embed.timestep_embedder.linear_2",
            ),
            (
                "double_stream_modulation_img.lin",
                "double_stream_modulation_img.linear",
            ),
            (
                "double_stream_modulation_txt.lin",
                "double_stream_modulation_txt.linear",
            ),
            (
                "single_stream_modulation.lin",
                "single_stream_modulation.linear",
            ),
            ("final_layer.linear", "proj_out"),
        ] {
            if path == target {
                out.push(direct_candidate(source));
            }
        }
        if let Some((index, suffix)) = block_parts(path, "transformer_blocks") {
            let mapping = [
                ("attn.to_out.0", "img_attn.proj"),
                ("attn.to_add_out", "txt_attn.proj"),
                ("ff.linear_in", "img_mlp.0"),
                ("ff.linear_out", "img_mlp.2"),
                ("ff_context.linear_in", "txt_mlp.0"),
                ("ff_context.linear_out", "txt_mlp.2"),
            ];
            for (target, source) in mapping {
                if suffix == target {
                    out.push(direct_candidate(format!("double_blocks.{index}.{source}")));
                    out.push(direct_candidate(format!(
                        "lora_unet_double_blocks_{index}_{}",
                        source.replace('.', "_")
                    )));
                }
            }
            for (stream, targets) in [
                ("img", ["attn.to_q", "attn.to_k", "attn.to_v"]),
                (
                    "txt",
                    ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                ),
            ] {
                for (part, target) in targets.into_iter().enumerate() {
                    if suffix == target {
                        let dotted = format!("double_blocks.{index}.{stream}_attn.qkv");
                        out.push(split_candidate(
                            dotted,
                            UpSlice::Chunk {
                                count: 3,
                                index: part,
                            },
                            3,
                            part,
                        ));
                        out.push(split_candidate(
                            format!("lora_unet_double_blocks_{index}_{stream}_attn_qkv"),
                            UpSlice::Chunk {
                                count: 3,
                                index: part,
                            },
                            3,
                            part,
                        ));
                    }
                }
            }
        }
        if let Some((index, suffix)) = block_parts(path, "single_transformer_blocks") {
            for (target, source) in [
                ("attn.to_qkv_mlp_proj", "linear1"),
                ("attn.to_out", "linear2"),
            ] {
                if suffix == target {
                    out.push(direct_candidate(format!("single_blocks.{index}.{source}")));
                    out.push(direct_candidate(format!(
                        "lora_unet_single_blocks_{index}_{source}"
                    )));
                }
            }
        }
    } else if family.starts_with("flux") {
        if let Some((index, suffix)) = block_parts(path, "transformer_blocks") {
            let mapping = [
                ("attn.to_out.0", "img_attn_proj"),
                ("attn.to_add_out", "txt_attn_proj"),
                ("ff.net.0.proj", "img_mlp_0"),
                ("ff.net.2", "img_mlp_2"),
                ("ff_context.net.0.proj", "txt_mlp_0"),
                ("ff_context.net.2", "txt_mlp_2"),
                ("norm1.linear", "img_mod_lin"),
                ("norm1_context.linear", "txt_mod_lin"),
            ];
            for (target, source) in mapping {
                if suffix == target {
                    out.push(direct_candidate(format!(
                        "lora_unet_double_blocks_{index}_{source}"
                    )));
                }
            }
            for (stream, targets) in [
                ("img", ["attn.to_q", "attn.to_k", "attn.to_v"]),
                (
                    "txt",
                    ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                ),
            ] {
                for (part, target) in targets.into_iter().enumerate() {
                    if suffix == target {
                        out.push(split_candidate(
                            format!("lora_unet_double_blocks_{index}_{stream}_attn_qkv"),
                            UpSlice::Chunk {
                                count: 3,
                                index: part,
                            },
                            3,
                            part,
                        ));
                    }
                }
            }
        }
        if let Some((index, suffix)) = block_parts(path, "single_transformer_blocks") {
            match suffix {
                "attn.to_q" | "attn.to_k" | "attn.to_v" | "proj_mlp" => {
                    let (part, start) = match suffix {
                        "attn.to_q" => (0, 0),
                        "attn.to_k" => (1, out_features),
                        "attn.to_v" => (2, 2 * out_features),
                        _ => (3, 3 * (out_features / 4)),
                    };
                    out.push(split_candidate(
                        format!("lora_unet_single_blocks_{index}_linear1"),
                        UpSlice::Range {
                            start,
                            len: out_features,
                        },
                        4,
                        part,
                    ));
                }
                "proj_out" => out.push(direct_candidate(format!(
                    "lora_unet_single_blocks_{index}_linear2"
                ))),
                "norm.linear" => out.push(direct_candidate(format!(
                    "lora_unet_single_blocks_{index}_modulation_lin"
                ))),
                _ => {}
            }
        }
    }
    out
}

/// Wan community adapters commonly use the native model namespace while the Candle host exposes
/// diffusers names. Dense Wan merges translate source keys before lookup; additive dense/packed
/// installs instead walk the host, so offer the equivalent native spelling as a candidate.
fn wan_candidates(family: &str, path: &str) -> Vec<LoraCandidate> {
    if !family.starts_with("wan") {
        return Vec::new();
    }
    let mut native = path
        .replace(".attn1.", ".self_attn.")
        .replace(".attn2.", ".cross_attn.");
    if let Some(base) = native.strip_suffix(".ffn.net.0.proj") {
        native = format!("{base}.ffn.0");
    } else if let Some(base) = native.strip_suffix(".ffn.net.2") {
        native = format!("{base}.ffn.2");
    } else {
        for (diffusers, wan) in [
            (".to_q", ".q"),
            (".to_k", ".k"),
            (".to_v", ".v"),
            (".to_out.0", ".o"),
        ] {
            if let Some(base) = native.strip_suffix(diffusers) {
                native = format!("{base}{wan}");
                break;
            }
        }
    }
    (native != path)
        .then(|| direct_candidate(native))
        .into_iter()
        .collect()
}

fn strip_prefix(key: &str) -> &str {
    for prefix in wmeta::COMMON_LORA_PREFIXES {
        if let Some(rest) = key.strip_prefix(prefix) {
            return rest;
        }
    }
    key
}

fn classify_lora_key(key: &str) -> Option<(String, Role)> {
    let key = strip_prefix(key);
    for (suffix, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".lora_down", Role::Down),
        (".lora_up", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = key.strip_suffix(suffix) {
            return Some((path.to_string(), role));
        }
    }
    None
}

fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    for suffix in LOKR_SUFFIXES {
        if let Some(path) = key.strip_suffix(suffix) {
            return Some((strip_prefix(path).to_string(), &suffix[1..]));
        }
    }
    None
}

fn resolve_lora(
    file: &AdapterFile,
    scale: f32,
    source: usize,
    pending: &mut BTreeMap<String, Vec<PendingLora>>,
    skipped: &mut usize,
) -> CResult<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, tensor) in &file.tensors {
        match classify_lora_key(key) {
            Some((path, Role::Down)) => {
                triples.entry(path).or_default().down = Some(tensor.clone())
            }
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(tensor.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, "alpha", tensor)?)
            }
            None => *skipped += 1,
        }
    }
    let metadata = LoraAdapterMeta::from_file_metadata(&file.meta);
    for (path, triple) in triples {
        let (Some(down), Some(up)) = (triple.down, triple.up) else {
            *skipped += 1;
            continue;
        };
        if down.rank() != 2 || up.rank() != 2 {
            *skipped += 1;
            continue;
        }
        let (meta_alpha, meta_rank) = metadata
            .as_ref()
            .map_or((None, None), |m| m.effective(&path));
        let rank = meta_rank.unwrap_or(down.dim(0)? as f32) as f64;
        if rank == 0.0 {
            *skipped += 1;
            continue;
        }
        let alpha = triple.alpha.or(meta_alpha).unwrap_or(rank as f32) as f64;
        let a = down.to_dtype(DType::F32)?.t()?.contiguous()?;
        let b = (up.to_dtype(DType::F32)?.t()?.contiguous()? * (alpha / rank))?;
        pending.entry(path).or_default().push(PendingLora {
            a,
            b,
            scale: scale as f64,
            source,
        });
    }
    Ok(())
}

fn resolve_lokr(
    file: &AdapterFile,
    scale: f32,
    source: usize,
    pending: &mut BTreeMap<String, Vec<PendingLokr>>,
    skipped: &mut usize,
) -> CResult<()> {
    let (rank, alpha) = parse_lokr_metadata(
        file.meta.get("rank").map(String::as_str),
        file.meta.get("alpha").map(String::as_str),
    )?;
    let full_scale = alpha as f64 / rank as f64 * scale as f64;
    if !full_scale.is_finite() {
        return Err(CandleError::Msg(format!(
            "LoKr derived scale must be finite, got {full_scale}"
        )));
    }
    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, tensor) in &file.tensors {
        match classify_lokr_key(key) {
            Some((path, factor)) => {
                grouped
                    .entry(path)
                    .or_default()
                    .insert(factor, tensor.clone());
            }
            None => *skipped += 1,
        }
    }
    for (path, factors) in grouped {
        pending.entry(path).or_default().push(PendingLokr {
            w1: factors.get("lokr_w1").cloned(),
            w1_a: factors.get("lokr_w1_a").cloned(),
            w1_b: factors.get("lokr_w1_b").cloned(),
            w2: factors.get("lokr_w2").cloned(),
            w2_a: factors.get("lokr_w2_a").cloned(),
            w2_b: factors.get("lokr_w2_b").cloned(),
            scale: full_scale,
            source,
        });
    }
    Ok(())
}

/// Installs a stack on a model that exposes its adaptable projections using canonical dotted keys.
/// `visit` must walk every adaptable projection exactly once. A non-empty stack that matches no
/// projection is an error, as are declared/file kind mismatches and adapter formats without a safe
/// additive representation.
pub fn install_dotted_adapters(
    family: &str,
    specs: &[AdapterSpec],
    device: &Device,
    visit: impl FnOnce(&mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>) -> Result<()>,
) -> CResult<AdditiveAdapterReport> {
    let mut loras: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut lokrs: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
    let mut report = AdditiveAdapterReport::default();
    for (source, spec) in specs.iter().enumerate() {
        if !spec.scale.is_finite() {
            return Err(CandleError::Msg(format!(
                "{family}: adapter {} scale must be finite, got {}",
                spec.path.display(),
                spec.scale
            )));
        }
        if let Some(expert) = spec.moe_expert {
            return Err(CandleError::Msg(format!(
                "{family}: adapter {} targets the {expert:?} MoE expert, but this model has a single denoiser",
                spec.path.display()
            )));
        }
        let file = read_adapter(&spec.path)?;
        let keys_lokr = wmeta::keys_contain_lokr(file.tensors.keys().map(String::as_str));
        let keys_loha = wmeta::keys_contain_loha(file.tensors.keys().map(String::as_str));
        if keys_loha {
            return Err(CandleError::Msg(format!(
                "{family}: LoHa has no packed-safe additive representation: {}",
                spec.path.display()
            )));
        }
        if keys_lokr && !file.declares_lokr() {
            return Err(CandleError::Msg(format!(
                "{family}: untagged LyCORIS LoKr cannot be scaled truthfully on an additive tier: {}",
                spec.path.display()
            )));
        }
        match (spec.kind, file.declares_lokr()) {
            (AdapterKind::Lora, true) => {
                return Err(CandleError::Msg(format!(
                    "{family}: adapter {} was declared LoRA but its metadata says networkType=lokr",
                    spec.path.display()
                )));
            }
            (AdapterKind::Lokr, false) => {
                return Err(CandleError::Msg(format!(
                    "{family}: adapter {} was declared LoKr but does not declare networkType=lokr",
                    spec.path.display()
                )));
            }
            (AdapterKind::Lokr, true) => resolve_lokr(
                &file,
                spec.scale,
                source,
                &mut lokrs,
                &mut report.skipped_keys,
            )?,
            (AdapterKind::Lora, false) => resolve_lora(
                &file,
                spec.scale,
                source,
                &mut loras,
                &mut report.skipped_keys,
            )?,
        }
    }

    let mut matched = HashSet::new();
    let mut applied_sources = HashSet::new();
    let mut visitor = |path: &str, linear: &mut AdaptLinear| -> Result<()> {
        let (out_features, in_features) = linear.base_shape();
        let kohya = format!("lora_unet_{}", path.replace('.', "_"));
        let mut candidates = vec![direct_candidate(path), direct_candidate(kohya.clone())];
        candidates.extend(bfl_candidates(family, path, out_features));
        candidates.extend(wan_candidates(family, path));
        for candidate in candidates {
            let Some(items) = loras.get(&candidate.key) else {
                continue;
            };
            matched.insert(candidate.key.clone());
            for item in items {
                let mut a = item.a.clone();
                let mut b = item.b.clone();
                if let Some(slice) = candidate.up_slice {
                    let (start, len) = match slice {
                        UpSlice::Chunk { count, index } => {
                            if b.dim(1)? % count != 0 {
                                return Err(candle_core::Error::Msg(format!(
                                    "{family}: fused adapter target `{}` has {} output rows, not divisible by {count}",
                                    candidate.key,
                                    b.dim(1)?
                                )));
                            }
                            let len = b.dim(1)? / count;
                            (index * len, len)
                        }
                        UpSlice::Range { start, len } => (start, len),
                    };
                    if start + len > b.dim(1)? {
                        return Err(candle_core::Error::Msg(format!(
                            "{family}: fused adapter target `{}` is too short for output slice {start}..{}",
                            candidate.key,
                            start + len
                        )));
                    }
                    b = b.narrow(1, start, len)?;
                }
                if let Some((count, index)) = candidate.down_chunks {
                    let rank = a.dim(1)?;
                    if rank % count == 0 && b.dim(0)? == rank / count {
                        let len = rank / count;
                        a = a.narrow(1, index * len, len)?;
                    }
                }
                if a.dims()[0] != in_features || b.dims()[1] != out_features {
                    // sc-11045 fix round (MAJOR 7): on an NVFP4 host a mis-shaped factor is a
                    // typed refusal, never a silent skip — there is no fallback regime that could
                    // have rendered it, so "skipped" would mean "silently un-adapted".
                    if linear.is_nvfp4() {
                        return Err(candle_core::Error::Msg(format!(
                            "{family}: LoRA factors for `{}` reconstruct [in={}, out={}], which \
                             does not compose against the NVFP4 base [out={out_features}, \
                             in={in_features}]; an NVFP4 base refuses a mismatched factor rather \
                             than skipping it",
                            candidate.key,
                            a.dims()[0],
                            b.dims()[1],
                        )));
                    }
                    report.skipped_keys += 1;
                    continue;
                }
                // Routed through the checked push on an NVFP4 host (`push_lora` delegates to
                // `push_lora_checked` there, sc-21483/sc-11045), so dtype/device admission is
                // typed at install.
                linear
                    .push_lora(a.to_device(device)?, b.to_device(device)?, item.scale)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                report.applied += 1;
                applied_sources.insert(item.source);
            }
        }
        let mut lokr_candidates = vec![direct_candidate(path), direct_candidate(kohya.as_str())];
        lokr_candidates.extend(bfl_candidates(family, path, out_features));
        lokr_candidates.extend(wan_candidates(family, path));
        for candidate in lokr_candidates {
            let Some(items) = lokrs.get(&candidate.key) else {
                continue;
            };
            matched.insert(candidate.key.clone());
            for item in items {
                let built = match candidate.up_slice {
                    Some(UpSlice::Chunk { count: _, index }) => LokrFactors::build_sliced(
                        item.scale,
                        in_features,
                        (index * out_features, out_features),
                        item.w1.as_ref(),
                        item.w1_a.as_ref(),
                        item.w1_b.as_ref(),
                        item.w2.as_ref(),
                        None,
                        item.w2_a.as_ref(),
                        item.w2_b.as_ref(),
                    ),
                    Some(UpSlice::Range { start, len }) => {
                        if len != out_features {
                            return Err(candle_core::Error::Msg(format!(
                                "{family}: fused LoKr target `{}` output slice has length {len}, expected host width {out_features}",
                                candidate.key
                            )));
                        }
                        LokrFactors::build_sliced(
                            item.scale,
                            in_features,
                            (start, len),
                            item.w1.as_ref(),
                            item.w1_a.as_ref(),
                            item.w1_b.as_ref(),
                            item.w2.as_ref(),
                            None,
                            item.w2_a.as_ref(),
                            item.w2_b.as_ref(),
                        )
                    }
                    None => LokrFactors::build(
                        item.scale,
                        (out_features, in_features),
                        item.w1.as_ref(),
                        item.w1_a.as_ref(),
                        item.w1_b.as_ref(),
                        item.w2.as_ref(),
                        None,
                        item.w2_a.as_ref(),
                        item.w2_b.as_ref(),
                    ),
                };
                let Some(factors) =
                    built.map_err(|error| candle_core::Error::Msg(error.to_string()))?
                else {
                    return Err(candle_core::Error::Msg(format!(
                        "{family}: LoKr target `{}` has no packed-safe structured form",
                        candidate.key
                    )));
                };
                // Routed through the checked push on an NVFP4 host (`push_lokr_structured`
                // delegates to the checked form there, sc-21483/sc-11045): a factor set that does
                // not reconstruct the base shape is a typed refusal at install.
                linear
                    .push_lokr_structured(
                        factors
                            .to_device(device)
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))?,
                    )
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                report.applied += 1;
                applied_sources.insert(item.source);
            }
        }
        Ok(())
    };
    visit(&mut visitor)?;

    for path in loras.keys().chain(lokrs.keys()) {
        if !matched.contains(path) {
            report.skipped_targets.push(path.clone());
        }
    }
    if let Some((_source, spec)) = specs
        .iter()
        .enumerate()
        .find(|(source, _)| !applied_sources.contains(source))
    {
        return Err(CandleError::Msg(format!(
            "{family}: adapter {} matched no compatible model projection; every selected adapter must apply (expected dotted PEFT/diffusers LoRA or PEFT-stamped LoKr keys matching the model visitor)",
            spec.path.display()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Linear;
    use std::collections::HashMap;

    fn write_lora(path: &std::path::Path, target: &str, in_dim: usize, out_dim: usize) {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            format!("{target}.lora_A.weight"),
            Tensor::ones((1, in_dim), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{target}.lora_B.weight"),
            Tensor::ones((out_dim, 1), DType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    fn write_lokr(path: &std::path::Path, target: &str, w1: Tensor, w2: Tensor, alpha: &str) {
        let tensors = HashMap::from([
            (format!("{target}.lokr_w1"), w1),
            (format!("{target}.lokr_w2"), w2),
        ]);
        safetensors::serialize_to_file(
            tensors.into_iter().collect::<Vec<_>>(),
            Some(HashMap::from([
                ("networkType".to_string(), "lokr".to_string()),
                ("rank".to_string(), "1".to_string()),
                ("alpha".to_string(), alpha.to_string()),
            ])),
            path,
        )
        .unwrap();
    }

    #[test]
    fn dotted_classification_strips_namespaces_and_preserves_stack_targets() {
        assert_eq!(
            classify_lora_key("transformer.layers.0.attn.q.lora_A.weight"),
            Some(("layers.0.attn.q".into(), Role::Down))
        );
        assert_eq!(
            classify_lora_key("diffusion_model.layers.0.attn.q.lora_up.weight"),
            Some(("layers.0.attn.q".into(), Role::Up))
        );
        assert_eq!(
            classify_lokr_key("transformer.layers.0.attn.q.lokr_w2_b"),
            Some(("layers.0.attn.q".into(), "lokr_w2_b"))
        );
    }

    #[test]
    fn installs_a_stacked_lora_on_dense_and_fails_closed_on_zero_match() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first.safetensors");
        let second = temp.path().join("second.safetensors");
        write_lora(&first, "layers.0.proj", 2, 2);
        write_lora(&second, "layers.0.proj", 2, 2);
        let specs = vec![
            AdapterSpec::new(first, 0.5, AdapterKind::Lora),
            AdapterSpec::new(second, 1.0, AdapterKind::Lora),
        ];
        let device = Device::Cpu;
        let base = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let mut linear = AdaptLinear::from_dense(Linear::new(base, None), 2, 2);
        let report = install_dotted_adapters("fixture", &specs, &device, |visitor| {
            visitor("layers.0.proj", &mut linear)
        })
        .unwrap();
        assert_eq!(report.applied, 2);
        assert!(linear.is_adapted());

        let missing = temp.path().join("missing.safetensors");
        write_lora(&missing, "layers.9.missing", 2, 2);
        let mut untouched = AdaptLinear::from_dense(
            Linear::new(Tensor::zeros((2, 2), DType::F32, &device).unwrap(), None),
            2,
            2,
        );
        install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(missing, 1.0, AdapterKind::Lora)],
            &device,
            |visitor| visitor("layers.0.proj", &mut untouched),
        )
        .expect_err("an adapter that reaches no model projection must fail closed");

        let partial = temp.path().join("partial-stack-miss.safetensors");
        write_lora(&partial, "layers.9.missing", 2, 2);
        let mut partially_adapted = AdaptLinear::from_dense(
            Linear::new(Tensor::zeros((2, 2), DType::F32, &device).unwrap(), None),
            2,
            2,
        );
        let error = install_dotted_adapters(
            "fixture",
            &[
                specs[0].clone(),
                AdapterSpec::new(partial.clone(), 1.0, AdapterKind::Lora),
            ],
            &device,
            |visitor| visitor("layers.0.proj", &mut partially_adapted),
        )
        .expect_err("one valid adapter must not hide a later zero-match adapter");
        assert!(error.to_string().contains(&partial.display().to_string()));
    }

    /// **sc-11045 fix round (MAJOR 7): a mis-shaped factor against an NVFP4 host is a typed
    /// refusal at install, never a silent `skipped_keys` bump.** The scenario the silent skip
    /// hides is exactly this one: an adapter whose file also carries a well-shaped factor for a
    /// sibling projection "applies" (the whole-spec zero-match check passes) while the mis-shaped
    /// key is dropped — a partially, silently un-adapted NVFP4 render (epic E6).
    ///
    /// # Mutation
    ///
    /// Remove the `linear.is_nvfp4()` refusal in the visitor's shape check (restore the plain
    /// `skipped_keys += 1; continue`): the install below succeeds with one silently skipped key
    /// and the `unwrap_err` goes red.
    #[test]
    fn a_mis_shaped_factor_against_an_nvfp4_host_refuses_instead_of_skipping() {
        use super::super::{ActPrecision, Nvfp4Linear};

        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("nvfp4-mixed-shapes.safetensors");
        // One well-shaped target ("layers.0.good": 64→32) and one mis-shaped ("layers.0.bad":
        // written 16→32 against a 64→32 host) in the SAME file.
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        for (target, in_dim) in [("layers.0.good", 64usize), ("layers.0.bad", 16usize)] {
            tensors.insert(
                format!("{target}.lora_A.weight"),
                Tensor::ones((1, in_dim), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{target}.lora_B.weight"),
                Tensor::ones((32, 1), DType::F32, &device).unwrap(),
            );
        }
        safetensors::serialize_to_file(tensors.into_iter().collect::<Vec<_>>(), None, &adapter)
            .unwrap();

        let nvfp4_host = |out_dim: usize, in_dim: usize| {
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| ((i % 17) as f32 - 8.0) / 11.0)
                .collect();
            let w = Tensor::from_vec(w, (out_dim, in_dim), &device).unwrap();
            let lin = Nvfp4Linear::from_dense(&w, None, &device, ActPrecision::W4A16).unwrap();
            AdaptLinear::from_nvfp4(lin)
        };
        let mut good = nvfp4_host(32, 64);
        let mut bad = nvfp4_host(32, 64);
        let error = install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)],
            &device,
            |visitor| {
                visitor("layers.0.good", &mut good)?;
                visitor("layers.0.bad", &mut bad)
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("NVFP4 base") && error.contains("layers.0.bad"),
            "the refusal must name the NVFP4 base and the offending key: {error}"
        );
        assert!(
            !bad.is_adapted(),
            "nothing may be attached to the mis-matched host"
        );
    }

    #[test]
    fn installs_on_a_packed_base_and_rejects_declared_kind_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("packed.safetensors");
        write_lora(&adapter, "proj", 64, 2);
        let device = Device::Cpu;
        let codes = Tensor::zeros((2, 8), DType::U32, &device).unwrap();
        let scales = Tensor::ones((2, 1), DType::F32, &device).unwrap();
        let biases = Tensor::zeros((2, 1), DType::F32, &device).unwrap();
        let packed =
            super::super::QLinear::from_packed(&codes, &scales, &biases, None, &device).unwrap();
        let mut linear = AdaptLinear::from_packed(packed, 64, 2);
        let report = install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora)],
            &device,
            |visitor| visitor("proj", &mut linear),
        )
        .unwrap();
        assert_eq!(report.applied, 1);
        assert!(linear.is_adapted());

        let mut dense = AdaptLinear::from_dense(
            Linear::new(Tensor::zeros((2, 64), DType::F32, &device).unwrap(), None),
            64,
            2,
        );
        let err = install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(adapter, 1.0, AdapterKind::Lokr)],
            &device,
            |visitor| visitor("proj", &mut dense),
        )
        .unwrap_err();
        assert!(err.to_string().contains("declared LoKr"));

        let mut dense = AdaptLinear::from_dense(
            Linear::new(Tensor::zeros((2, 64), DType::F32, &device).unwrap(), None),
            64,
            2,
        );
        let routed = AdapterSpec::new(
            temp.path().join("packed.safetensors"),
            1.0,
            AdapterKind::Lora,
        )
        .with_moe_expert(gen_core::MoeExpert::High);
        let err = install_dotted_adapters("fixture", &[routed], &device, |visitor| {
            visitor("proj", &mut dense)
        })
        .unwrap_err();
        assert!(err.to_string().contains("single denoiser"));
    }

    #[test]
    fn kohya_flattened_target_resolves_to_the_dotted_visitor_path() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("kohya.safetensors");
        write_lora(&adapter, "lora_unet_layers_0_proj", 2, 2);
        let device = Device::Cpu;
        let mut linear = AdaptLinear::from_dense(
            Linear::new(Tensor::zeros((2, 2), DType::F32, &device).unwrap(), None),
            2,
            2,
        );
        let report = install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)],
            &device,
            |visitor| visitor("layers.0.proj", &mut linear),
        )
        .unwrap();
        assert_eq!(report.applied, 1);
    }

    #[test]
    fn flux2_bfl_fused_qkv_fans_out_to_split_host_projections() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("flux2-bfl-qkv.safetensors");
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight".to_string(),
            Tensor::ones((1, 2), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight".to_string(),
            Tensor::arange(0f32, 6f32, &device)
                .unwrap()
                .reshape((6, 1))
                .unwrap(),
        );
        candle_core::safetensors::save(&tensors, &adapter).unwrap();
        let specs = [AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)];
        let make = || {
            AdaptLinear::from_dense(
                Linear::new(Tensor::zeros((2, 2), DType::F32, &device).unwrap(), None),
                2,
                2,
            )
        };
        let (mut q, mut k, mut v) = (make(), make(), make());
        let report = install_dotted_adapters("flux2", &specs, &device, |visitor| {
            visitor("transformer_blocks.0.attn.to_q", &mut q)?;
            visitor("transformer_blocks.0.attn.to_k", &mut k)?;
            visitor("transformer_blocks.0.attn.to_v", &mut v)
        })
        .unwrap();
        assert_eq!(report.applied, 3);
        assert!(q.is_adapted() && k.is_adapted() && v.is_adapted());
    }

    #[test]
    fn flux1_and_flux2_bfl_fused_qkv_lokr_fan_out_structurally() {
        let temp = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let w1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3, 1), &device).unwrap();
        let w2 = Tensor::eye(2, DType::F32, &device).unwrap();
        for (family, target) in [
            ("flux", "lora_unet_double_blocks_0_img_attn_qkv"),
            ("flux2", "double_blocks.0.img_attn.qkv"),
        ] {
            let adapter = temp
                .path()
                .join(format!("{family}-bfl-qkv-lokr.safetensors"));
            write_lokr(&adapter, target, w1.clone(), w2.clone(), "1");
            let specs = [AdapterSpec::new(adapter, 1.0, AdapterKind::Lokr)];
            let make = || {
                AdaptLinear::from_dense(
                    Linear::new(Tensor::zeros((2, 2), DType::F32, &device).unwrap(), None),
                    2,
                    2,
                )
            };
            let (mut q, mut k, mut v) = (make(), make(), make());
            let report = install_dotted_adapters(family, &specs, &device, |visitor| {
                visitor("transformer_blocks.0.attn.to_q", &mut q)?;
                visitor("transformer_blocks.0.attn.to_k", &mut k)?;
                visitor("transformer_blocks.0.attn.to_v", &mut v)
            })
            .unwrap();
            assert_eq!(
                report.applied, 3,
                "{family} fused LoKr must fan out to q/k/v"
            );
            let x = Tensor::ones((1, 2), DType::F32, &device).unwrap();
            let outputs = [
                q.forward(&x).unwrap(),
                k.forward(&x).unwrap(),
                v.forward(&x).unwrap(),
            ];
            for (index, output) in outputs.iter().enumerate() {
                let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let want = (index + 1) as f32;
                assert!(values.iter().all(|value| (*value - want).abs() < 1e-6));
            }
        }
    }

    #[test]
    fn lokr_rejects_non_finite_metadata_and_user_scale() {
        let temp = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let w1 = Tensor::ones((1, 1), DType::F32, &device).unwrap();
        let w2 = Tensor::ones((2, 2), DType::F32, &device).unwrap();
        let bad_meta = temp.path().join("bad-meta.safetensors");
        write_lokr(&bad_meta, "proj", w1.clone(), w2.clone(), "inf");
        let make = || {
            AdaptLinear::from_dense(
                Linear::new(Tensor::zeros((2, 2), DType::F32, &device).unwrap(), None),
                2,
                2,
            )
        };
        let mut linear = make();
        let error = install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(bad_meta, 1.0, AdapterKind::Lokr)],
            &device,
            |visitor| visitor("proj", &mut linear),
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("invalid metadata scale") && message.contains("alpha must be finite")
        );

        let valid = temp.path().join("valid.safetensors");
        write_lokr(&valid, "proj", w1, w2, "1");
        let mut linear = make();
        let error = install_dotted_adapters(
            "fixture",
            &[AdapterSpec::new(valid, f32::INFINITY, AdapterKind::Lokr)],
            &device,
            |visitor| visitor("proj", &mut linear),
        )
        .unwrap_err();
        assert!(error.to_string().contains("scale must be finite"));
    }
}

//! Strict FLUX3 LoRA admission and the explicit inverse of diffusers' frozen
//! `_convert_non_diffusers_flux3_lora_to_diffusers` (7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941).
//! Identity metadata is mandatory: `base_model`, `base_revision`. A consistent explicit global
//! alpha must be declared by native alpha/lora_alpha or PEFT metadata; no alpha=rank fallback.
//! Tensor values are read only by the native backend after this tensor-neutral preflight.
use super::{
    DitConfig, BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION, DISTILLED_WEIGHTS_REPO,
    DISTILLED_WEIGHTS_REVISION,
};
use gen_core::{AdapterKind, AdapterSpec, Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

#[derive(Clone, Debug)]
pub struct Projection {
    pub target: String,
    pub down: String,
    pub up: String,
    pub rank: usize,
    pub scale: f64,
}

fn unsupported(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 LoRA: {detail}"))
}

/// Resolve a native or diffusers module stem, never a fuzzy suffix/alias match.
pub fn native_target(stem: &str, cfg: &DitConfig) -> Result<(String, usize, usize)> {
    let stem = stem.strip_prefix("transformer.").unwrap_or(stem);
    let stem = stem.strip_prefix("diffusion_model.").unwrap_or(stem);
    let (parent, leaf) = if let Some(rest) = stem
        .strip_prefix("mode_blocks.")
        .or_else(|| stem.strip_prefix("content_mode_blocks."))
    {
        let mut fields = rest.splitn(3, '.');
        let mode = fields.next().unwrap_or("");
        let block = fields
            .next()
            .unwrap_or("")
            .parse::<usize>()
            .map_err(|_| unsupported(format!("invalid block in {stem}")))?;
        if !["image", "image_cond"].contains(&mode) || block >= cfg.depth {
            return Err(unsupported(format!("unknown mode/block {stem}")));
        }
        (
            format!("content_mode_blocks.{mode}.{block}"),
            fields.next().unwrap_or(""),
        )
    } else if let Some(rest) = stem.strip_prefix("txt_mode_blocks.") {
        let (block, leaf) = rest
            .split_once('.')
            .ok_or_else(|| unsupported(format!("invalid target {stem}")))?;
        let block = block
            .parse::<usize>()
            .map_err(|_| unsupported(format!("invalid block in {stem}")))?;
        if block >= cfg.depth {
            return Err(unsupported(format!("unknown text block {stem}")));
        }
        (format!("txt_mode_blocks.{block}"), leaf)
    } else if let Some(rest) = stem
        .strip_prefix("single_transformer_blocks.")
        .or_else(|| stem.strip_prefix("single_blocks."))
    {
        let (block, leaf) = rest
            .split_once('.')
            .ok_or_else(|| unsupported(format!("invalid target {stem}")))?;
        let block = block
            .parse::<usize>()
            .map_err(|_| unsupported(format!("invalid block in {stem}")))?;
        if block >= cfg.depth_single {
            return Err(unsupported(format!("unknown single block {stem}")));
        }
        (format!("single_blocks.{block}"), leaf)
    } else {
        return Err(unsupported(format!("unsupported target {stem}; use mode_blocks or single_transformer_blocks attention/MLP projections")));
    };
    let (leaf, output, input) = match leaf {
        "q_proj" | "attn.to_q" => ("q_proj", cfg.hidden, cfg.hidden),
        "k_proj" | "attn.to_k" => ("k_proj", cfg.hidden, cfg.hidden),
        "v_proj" | "attn.to_v" => ("v_proj", cfg.hidden, cfg.hidden),
        "attn_out" | "attn.to_out.0" => ("attn_out", cfg.hidden, cfg.hidden),
        "mlp_in" | "ff.linear_in" => ("mlp_in", 2 * cfg.mlp_hidden(), cfg.hidden),
        "mlp_out" | "ff.linear_out" => ("mlp_out", cfg.hidden, cfg.mlp_hidden()),
        _ => return Err(unsupported(format!("unsupported projection {stem}"))),
    };
    Ok((format!("{parent}.{leaf}"), output, input))
}

/// Header-only admission, run before device initialization (including sequential loads).
/// Every tensor must belong to exactly one A/B pair. No partial installs or ignored keys.
pub fn inspect(spec: &AdapterSpec, cfg: &DitConfig) -> Result<Vec<Projection>> {
    let mut file =
        File::open(&spec.path).map_err(|e| unsupported(format!("{}: {e}", spec.path.display())))?;
    let file_len = file.metadata()?.len();
    let header = read_header_json(&mut file, file_len, &spec.path)?;
    inspect_header(spec, cfg, &header)
}

/// The crate-wide bounded header read (sc-27314), then the FLUX3 16 MiB family cap
/// ([`super::checkpoint::MAX_HEADER_BYTES`]) before any header buffer is allocated.
fn read_header_json(
    reader: &mut impl Read,
    file_len: u64,
    path: &Path,
) -> Result<serde_json::Value> {
    let (len, _data_start) =
        gen_core::weightsmeta::read_safetensors_header_len(reader, file_len, path)
            .map_err(|e| unsupported(format!("invalid safetensors header: {e}")))?;
    if len == 0 || len as u64 > super::checkpoint::MAX_HEADER_BYTES {
        return Err(unsupported("invalid/oversized safetensors header"));
    }
    let mut header = vec![0; len];
    reader
        .read_exact(&mut header)
        .map_err(|e| unsupported(format!("truncated header: {e}")))?;
    serde_json::from_slice(&header).map_err(|e| unsupported(format!("invalid header JSON: {e}")))
}

/// Only plain additive LoRA: honor explicit native/PEFT alpha, never invent alpha=rank or
/// silently reinterpret algorithms with different gain/orientation/per-target semantics.
fn plain_metadata(meta: &serde_json::Value) -> Result<(f64, Vec<usize>)> {
    let peft = meta
        .get("lora_adapter_metadata")
        .map(|raw| {
            let text = raw
                .as_str()
                .ok_or_else(|| unsupported("lora_adapter_metadata must be a JSON string"))?;
            let value: serde_json::Value = serde_json::from_str(text)
                .map_err(|e| unsupported(format!("invalid lora_adapter_metadata: {e}")))?;
            let object = value
                .as_object()
                .ok_or_else(|| unsupported("lora_adapter_metadata must contain an object"))?;
            // Pinned diffusers 7ecb1e1 `_save_lora_weights` packs the transformer's PEFT config
            // with `_pack_dict_with_prefix(metadata, "transformer")` (lora_base.py:327,1105) and
            // its loader strips exactly that prefix (peft.py:212). Normalize ONLY that prefix;
            // another component's keys (text_encoder.* etc.) cannot describe this DiT adapter.
            let mut normalized = serde_json::Map::new();
            for (key, field) in object {
                let bare = key.strip_prefix("transformer.").unwrap_or(key);
                if bare.contains('.') || bare.is_empty() {
                    return Err(unsupported(format!(
                        "lora_adapter_metadata key {key:?} is not a transformer PEFT field; only the diffusers `transformer.` prefix is supported"
                    )));
                }
                if let Some(previous) = normalized.insert(bare.to_owned(), field.clone()) {
                    if &previous != field {
                        return Err(unsupported(format!(
                            "lora_adapter_metadata declares {bare:?} twice with different values"
                        )));
                    }
                }
            }
            Ok(serde_json::Value::Object(normalized))
        })
        .transpose()?;
    let mut alpha: Option<f64> = None;
    let mut ranks = Vec::new();
    for section in std::iter::once(meta).chain(peft.iter()) {
        for key in ["networkType", "network_type", "peft_type"] {
            if let Some(kind) = section.get(key) {
                if !kind
                    .as_str()
                    .is_some_and(|kind| kind.eq_ignore_ascii_case("lora"))
                {
                    return Err(unsupported(format!(
                        "{key} must name plain LoRA, got {kind}"
                    )));
                }
            }
        }
        let args = section
            .get("ss_network_args")
            .map(|raw| {
                let text = raw
                    .as_str()
                    .ok_or_else(|| unsupported("ss_network_args must be a JSON string"))?;
                let value = serde_json::from_str::<serde_json::Value>(text)
                    .map_err(|e| unsupported(format!("invalid ss_network_args: {e}")))?;
                if !value.is_object() {
                    return Err(unsupported("ss_network_args must contain an object"));
                }
                Ok(value)
            })
            .transpose()?;
        let algo = args
            .as_ref()
            .and_then(|args| args.get("algo"))
            .and_then(serde_json::Value::as_str);
        if algo.is_some_and(|algo| !algo.eq_ignore_ascii_case("lora")) {
            return Err(unsupported("ss_network_args.algo must name plain LoRA"));
        }
        if let Some(module) = section.get("ss_network_module") {
            let plain = module.as_str().is_some_and(|module| {
                matches!(module, "lora" | "networks.lora")
                    || (module == "lycoris.kohya"
                        && algo.is_some_and(|algo| algo.eq_ignore_ascii_case("lora")))
            });
            if !plain {
                return Err(unsupported(
                    "ss_network_module does not identify plain LoRA",
                ));
            }
        }
        for key in ["use_rslora", "use_dora", "fan_in_fan_out"] {
            if let Some(value) = section.get(key) {
                let enabled = value.as_bool().or_else(|| {
                    match value.as_str()?.trim().to_ascii_lowercase().as_str() {
                        "false" => Some(false),
                        "true" => Some(true),
                        _ => None,
                    }
                });
                if enabled != Some(false) {
                    return Err(unsupported(format!(
                        "{key} is not supported; export plain additive LoRA"
                    )));
                }
            }
        }
        for key in ["rank_pattern", "alpha_pattern"] {
            if let Some(value) = section.get(key) {
                let value = if let Some(text) = value.as_str() {
                    serde_json::from_str(text)
                        .map_err(|e| unsupported(format!("invalid {key}: {e}")))?
                } else {
                    value.clone()
                };
                if !value.is_null() && !value.as_object().is_some_and(|object| object.is_empty()) {
                    return Err(unsupported(format!("per-target {key} is not supported; export a resolved global alpha/rank adapter")));
                }
            }
        }
        for key in ["lora_alpha", "alpha", "ss_network_alpha"] {
            if let Some(value) = section.get(key) {
                let value = value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse::<f64>().ok())
                    .ok_or_else(|| unsupported(format!("{key} must be numeric")))?;
                if !value.is_finite() || value <= 0.0 || alpha.is_some_and(|alpha| alpha != value) {
                    return Err(unsupported(format!(
                        "{key} must be finite, positive and consistent with every declared alpha"
                    )));
                }
                alpha = Some(value);
            }
        }
        for key in ["rank", "r", "ss_network_dim"] {
            if let Some(value) = section.get(key) {
                let rank = value
                    .as_u64()
                    .and_then(|rank| usize::try_from(rank).ok())
                    .or_else(|| value.as_str()?.trim().parse::<usize>().ok())
                    .filter(|rank| *rank > 0)
                    .ok_or_else(|| unsupported(format!("{key} must be a positive integer")))?;
                ranks.push(rank);
            }
        }
    }
    Ok((
        alpha.ok_or_else(|| {
            unsupported(
                "explicit alpha/lora_alpha (native or PEFT metadata) is required; no rank fallback",
            )
        })?,
        ranks,
    ))
}

/// Every `__metadata__` key admission reads. A backend that receives tensors and metadata from
/// one loader (MLX `Weights`) rebuilds the header from exactly these, so validation and the
/// installed tensors come from the same read rather than a second open of the file.
pub const ADMISSION_METADATA_KEYS: &[&str] = &[
    "base_model",
    "base_revision",
    "lora_adapter_metadata",
    "networkType",
    "network_type",
    "peft_type",
    "ss_network_args",
    "ss_network_module",
    "use_rslora",
    "use_dora",
    "fan_in_fan_out",
    "rank_pattern",
    "alpha_pattern",
    "lora_alpha",
    "alpha",
    "ss_network_alpha",
    "rank",
    "r",
    "ss_network_dim",
];

/// Admission over an in-memory safetensors buffer: the SAME bytes the caller then hands to its
/// tensor loader, so the validated header and the installed tensors cannot diverge (no TOCTOU).
pub fn inspect_bytes(spec: &AdapterSpec, cfg: &DitConfig, bytes: &[u8]) -> Result<Vec<Projection>> {
    let header = read_header_json(&mut &bytes[..], bytes.len() as u64, &spec.path)?;
    inspect_header(spec, cfg, &header)
}

/// Admission over a header value the caller assembled from its own loaded tensors + metadata.
pub fn inspect_header(
    spec: &AdapterSpec,
    cfg: &DitConfig,
    header: &serde_json::Value,
) -> Result<Vec<Projection>> {
    if spec.kind != AdapterKind::Lora
        || !spec.scale.is_finite()
        || spec.pass_scales.is_some()
        || spec.moe_expert.is_some()
    {
        return Err(unsupported(
            "requires LoRA, finite explicit scale, no pass/expert override",
        ));
    }
    let (repo, revision) = if cfg.flow_map {
        (DISTILLED_WEIGHTS_REPO, DISTILLED_WEIGHTS_REVISION)
    } else {
        (BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION)
    };
    let meta = &header["__metadata__"];
    for (key, want) in [("base_model", repo), ("base_revision", revision)] {
        if meta[key].as_str() != Some(want) {
            return Err(unsupported(format!("{key} must be {want:?}; got {}. Export identity metadata for the actual base checkpoint", meta[key])));
        }
    }
    let (alpha, declared_ranks) = plain_metadata(meta)?;
    let object = header
        .as_object()
        .ok_or_else(|| unsupported("header must be an object"))?;
    let mut pairs: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    for key in object.keys().filter(|k| k.as_str() != "__metadata__") {
        let (stem, down) = if let Some(s) = key.strip_suffix(".lora_A.weight") {
            (s, true)
        } else if let Some(s) = key.strip_suffix(".lora_B.weight") {
            (s, false)
        } else {
            return Err(unsupported(format!(
                "unrecognized tensor {key}; expected lora_A.weight/lora_B.weight"
            )));
        };
        let pair = pairs.entry(stem.to_owned()).or_default();
        if down {
            pair.0 = Some(key.clone());
        } else {
            pair.1 = Some(key.clone());
        }
    }
    if pairs.is_empty() {
        return Err(unsupported("adapter contains no LoRA projections"));
    }
    let mut seen = BTreeSet::new();
    let mut plan = Vec::new();
    for (stem, (down, up)) in pairs {
        let down = down.ok_or_else(|| unsupported(format!("{stem}: missing lora_A.weight")))?;
        let up = up.ok_or_else(|| unsupported(format!("{stem}: missing lora_B.weight")))?;
        let (target, output, input) = native_target(&stem, cfg)?;
        if !seen.insert(target.clone()) {
            return Err(unsupported(format!("duplicate mapped target {target}")));
        }
        let shape = |key: &str| -> Result<Vec<usize>> {
            if !matches!(header[key]["dtype"].as_str(), Some("F32" | "F16" | "BF16")) {
                return Err(unsupported(format!(
                    "{key}: requires floating-point weights"
                )));
            }
            header[key]["shape"]
                .as_array()
                .ok_or_else(|| unsupported(format!("{key}: missing shape")))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|v| v as usize)
                        .ok_or_else(|| unsupported(format!("{key}: invalid shape")))
                })
                .collect()
        };
        let a = shape(&down)?;
        let b = shape(&up)?;
        if a.len() != 2 || b.len() != 2 || a[0] == 0 || a[1] != input || b != [output, a[0]] {
            return Err(unsupported(format!("{target}: incompatible rank/shape A{a:?}, B{b:?}; expected [rank,{input}] and [{output},rank]")));
        }
        if declared_ranks.iter().any(|rank| *rank != a[0]) {
            return Err(unsupported(format!(
                "{target}: every declared rank must match actual tensor rank {}",
                a[0]
            )));
        }
        let scale = f64::from(spec.scale) * alpha / a[0] as f64;
        if !scale.is_finite() || !(scale as f32).is_finite() {
            return Err(unsupported(format!("{target}: alpha/rank/scale overflow")));
        }
        plan.push(Projection {
            target,
            down,
            up,
            rank: a[0],
            scale,
        });
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> DitConfig {
        DitConfig {
            hidden: 8,
            heads: 2,
            depth: 2,
            depth_single: 3,
            mlp_ratio: 2.0,
            axes_dim: vec![4],
            theta: 10000.0,
            image_in_channels: 4,
            image_cond_in_channels: 4,
            text_in_channels: 8,
            out_modes: vec!["image".into()],
            vec_in_dim: None,
            image_rope_scale: 0.5,
            flow_map: false,
        }
    }
    fn header() -> serde_json::Value {
        serde_json::json!({
            "__metadata__": {"base_model": BASE_WEIGHTS_REPO, "base_revision": BASE_WEIGHTS_REVISION, "lora_alpha": "6"},
            "transformer.mode_blocks.image.1.attn.to_q.lora_A.weight": {"shape": [2,8], "dtype": "F32"},
            "transformer.mode_blocks.image.1.attn.to_q.lora_B.weight": {"shape": [8,2], "dtype": "F32"}
        })
    }
    #[test]
    fn exact_mapping_and_nondefault_scale() {
        let spec = AdapterSpec::new("adapter.safetensors".into(), -0.25, AdapterKind::Lora);
        let p = inspect_header(&spec, &config(), &header()).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].target, "content_mode_blocks.image.1.q_proj");
        assert_eq!(p[0].rank, 2);
        assert_eq!(p[0].scale, -0.75);
        let mut stamped = header();
        stamped["__metadata__"]["rank"] = "2".into();
        assert_eq!(
            inspect_header(&spec, &config(), &stamped).unwrap()[0].rank,
            2
        );
        for (source, want) in [
            (
                "single_transformer_blocks.2.ff.linear_in",
                "single_blocks.2.mlp_in",
            ),
            (
                "txt_mode_blocks.0.attn.to_out.0",
                "txt_mode_blocks.0.attn_out",
            ),
            (
                "diffusion_model.content_mode_blocks.image_cond.0.mlp_out",
                "content_mode_blocks.image_cond.0.mlp_out",
            ),
        ] {
            assert_eq!(native_target(source, &config()).unwrap().0, want);
        }
        for bad in [
            "mode_blocks.text.0.attn.to_q",
            "single_transformer_blocks.3.attn.to_q",
            "txt_mode_blocks.0.norm",
            "foo.q_proj",
        ] {
            assert!(matches!(
                native_target(bad, &config()),
                Err(Error::Unsupported(_))
            ));
        }
    }
    /// Header exactly as pinned diffusers 7ecb1e1 `Flux3LoraLoaderMixin.save_lora_weights` writes it:
    /// `{"format":"pt","lora_adapter_metadata": json.dumps(_pack_dict_with_prefix(peft_config.to_dict(),
    /// "transformer"), indent=2, sort_keys=True)}` plus `transformer.`-packed tensors; SceneWorks'
    /// identity stamps are the only additions.
    fn diffusers_saved_header(
        patch: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    ) -> serde_json::Value {
        let mut peft = serde_json::json!({
            "alpha_pattern": {}, "auto_mapping": null, "base_model_name_or_path": null, "bias": "none",
            "corda_config": null, "eva_config": null, "exclude_modules": null, "fan_in_fan_out": false,
            "inference_mode": false, "init_lora_weights": "gaussian", "layer_replication": null,
            "layers_pattern": null, "layers_to_transform": null, "loftq_config": {}, "lora_alpha": 6,
            "lora_bias": false, "lora_dropout": 0.0, "megatron_config": null, "megatron_core": "megatron.core",
            "modules_to_save": null, "peft_type": "LORA", "r": 2, "rank_pattern": {}, "revision": null,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "linear_in", "linear_out"],
            "task_type": null, "trainable_token_indices": null, "use_dora": false, "use_rslora": false
        }).as_object().unwrap().clone();
        patch(&mut peft);
        let packed: serde_json::Map<String, serde_json::Value> = peft
            .into_iter()
            .map(|(k, v)| (format!("transformer.{k}"), v))
            .collect();
        let mut h = header();
        let meta = h["__metadata__"].as_object_mut().unwrap();
        meta.remove("lora_alpha");
        meta.insert("format".into(), "pt".into());
        meta.insert(
            "lora_adapter_metadata".into(),
            serde_json::to_string_pretty(&packed).unwrap().into(),
        );
        h
    }

    #[test]
    fn diffusers_saved_transformer_prefixed_metadata_is_normalized_and_still_guarded() {
        let spec = AdapterSpec::new("adapter.safetensors".into(), -0.25, AdapterKind::Lora);
        let plan = inspect_header(&spec, &config(), &diffusers_saved_header(|_| {})).unwrap();
        assert_eq!(
            (plan[0].rank, plan[0].scale),
            (2, -0.75),
            "prefixed lora_alpha 6 / r 2 is honoured"
        );
        let refused = |h: serde_json::Value, why: &str| {
            assert!(
                matches!(
                    inspect_header(&spec, &config(), &h),
                    Err(Error::Unsupported(_))
                ),
                "{why}"
            );
        };
        refused(
            diffusers_saved_header(|p| {
                p.insert("use_rslora".into(), true.into());
            }),
            "prefixed rsLoRA",
        );
        refused(
            diffusers_saved_header(|p| {
                p.insert("use_dora".into(), true.into());
            }),
            "prefixed DoRA",
        );
        refused(
            diffusers_saved_header(|p| {
                p.insert("alpha_pattern".into(), serde_json::json!({"to_q": 12}));
            }),
            "prefixed alpha_pattern",
        );
        refused(
            diffusers_saved_header(|p| {
                p.insert("rank_pattern".into(), serde_json::json!({"to_q": 8}));
            }),
            "prefixed rank_pattern",
        );
        refused(
            diffusers_saved_header(|p| {
                p.insert("r".into(), 4.into());
            }),
            "prefixed rank contradicts factors",
        );
        let mut contradiction = diffusers_saved_header(|_| {});
        contradiction["__metadata__"]["lora_alpha"] = "12".into();
        refused(contradiction, "top-level alpha contradicts prefixed alpha");
        let mut foreign = diffusers_saved_header(|_| {});
        let mut blob: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
            foreign["__metadata__"]["lora_adapter_metadata"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        // A benign value that is otherwise consistent: only the component prefix is wrong.
        blob.insert("text_encoder.r".into(), 2.into());
        foreign["__metadata__"]["lora_adapter_metadata"] =
            serde_json::to_string(&blob).unwrap().into();
        refused(foreign, "another component's prefix");
        let mut mixed = diffusers_saved_header(|_| {});
        let mut blob: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
            mixed["__metadata__"]["lora_adapter_metadata"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        blob.insert("use_dora".into(), true.into());
        mixed["__metadata__"]["lora_adapter_metadata"] =
            serde_json::to_string(&blob).unwrap().into();
        refused(mixed, "bare and prefixed copies disagree");
    }

    #[test]
    fn plain_native_and_peft_metadata_never_silently_switch_scaling_algorithms() {
        let spec = AdapterSpec::new("adapter.safetensors".into(), -0.25, AdapterKind::Lora);
        let mut native = header();
        native["__metadata__"]
            .as_object_mut()
            .unwrap()
            .remove("lora_alpha");
        native["__metadata__"]["alpha"] = "6".into();
        native["__metadata__"]["ss_network_module"] = "networks.lora".into();
        assert_eq!(
            inspect_header(&spec, &config(), &native).unwrap()[0].scale,
            -0.75
        );
        let blob = serde_json::json!({"peft_type":"LORA", "r":2, "lora_alpha":6, "use_rslora":false, "use_dora":false, "fan_in_fan_out":false, "rank_pattern":{}, "alpha_pattern":{}});
        let mut peft = header();
        peft["__metadata__"]
            .as_object_mut()
            .unwrap()
            .remove("lora_alpha");
        peft["__metadata__"]["lora_adapter_metadata"] = blob.to_string().into();
        assert_eq!(
            inspect_header(&spec, &config(), &peft).unwrap()[0].scale,
            -0.75
        );
        for field in [
            "use_rslora",
            "use_dora",
            "fan_in_fan_out",
            "alpha_pattern",
            "rank_pattern",
        ] {
            let mut changed = blob.clone();
            changed[field] = if field.ends_with("pattern") {
                serde_json::json!({"mode_blocks.image.1.attn.to_q":12})
            } else {
                true.into()
            };
            let mut h = header();
            h["__metadata__"]["lora_adapter_metadata"] = changed.to_string().into();
            assert!(
                matches!(
                    inspect_header(&spec, &config(), &h),
                    Err(Error::Unsupported(_))
                ),
                "{field}"
            );
        }
        for (key, value) in [("ss_network_module", "networks.lokr"), ("r", "3")] {
            let mut h = header();
            h["__metadata__"][key] = value.into();
            assert!(
                matches!(
                    inspect_header(&spec, &config(), &h),
                    Err(Error::Unsupported(_))
                ),
                "{key}"
            );
        }
        let mut h = header();
        h["__metadata__"]["networkType"] = "lokr".into();
        assert!(matches!(
            inspect_header(&spec, &config(), &h),
            Err(Error::Unsupported(_))
        ));
        let mut h = header();
        h["__metadata__"]["alpha"] = "12".into();
        assert!(matches!(
            inspect_header(&spec, &config(), &h),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn incompatible_identity_rank_alpha_and_unconsumed_weights_fail_typed() {
        let spec = AdapterSpec::new("adapter.safetensors".into(), 0.3, AdapterKind::Lora);
        let mut mutations = Vec::new();
        for (key, value) in [
            ("base_model", "wrong"),
            ("base_revision", "wrong"),
            ("lora_alpha", "NaN"),
            ("lora_alpha", "0"),
            ("rank", "3"),
            ("rank", "0"),
        ] {
            let mut h = header();
            h["__metadata__"][key] = value.into();
            mutations.push(h);
        }
        let mut h = header();
        h["transformer.mode_blocks.image.1.attn.to_q.lora_B.weight"]["shape"] =
            serde_json::json!([7, 2]);
        mutations.push(h);
        let mut h = header();
        h["ignored.weight"] = serde_json::json!({"shape":[1], "dtype":"F32"});
        mutations.push(h);
        for h in mutations {
            assert!(matches!(
                inspect_header(&spec, &config(), &h),
                Err(Error::Unsupported(_))
            ));
        }
        let mut distilled = config();
        distilled.flow_map = true;
        assert!(matches!(
            inspect_header(&spec, &distilled, &header()),
            Err(Error::Unsupported(_))
        ));
    }
}

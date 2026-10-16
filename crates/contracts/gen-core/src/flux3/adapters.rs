//! Strict FLUX3 LoRA admission and the explicit inverse of diffusers' frozen
//! `_convert_non_diffusers_flux3_lora_to_diffusers` (7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941).
//! Identity metadata is mandatory: `base_model`, `base_revision`, `lora_alpha`.
//! Tensor values are read only by the native backend after this tensor-neutral preflight.
use super::{
    DitConfig, BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION, DISTILLED_WEIGHTS_REPO,
    DISTILLED_WEIGHTS_REVISION,
};
use crate::{AdapterKind, AdapterSpec, Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
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
    if spec.kind != AdapterKind::Lora
        || !spec.scale.is_finite()
        || spec.pass_scales.is_some()
        || spec.moe_expert.is_some()
    {
        return Err(unsupported(
            "requires LoRA, finite explicit scale, no pass/expert override",
        ));
    }
    let mut file =
        File::open(&spec.path).map_err(|e| unsupported(format!("{}: {e}", spec.path.display())))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .map_err(|e| unsupported(format!("invalid safetensors header: {e}")))?;
    let len = u64::from_le_bytes(len);
    if len == 0 || len > 16 * 1024 * 1024 {
        return Err(unsupported("invalid/oversized safetensors header"));
    }
    let mut header = vec![0; len as usize];
    file.read_exact(&mut header)
        .map_err(|e| unsupported(format!("truncated header: {e}")))?;
    let header: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| unsupported(format!("invalid header JSON: {e}")))?;
    inspect_header(spec, cfg, &header)
}

fn inspect_header(
    spec: &AdapterSpec,
    cfg: &DitConfig,
    header: &serde_json::Value,
) -> Result<Vec<Projection>> {
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
    let alpha: f64 = meta["lora_alpha"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| unsupported("missing numeric lora_alpha metadata"))?;
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(unsupported("lora_alpha must be finite and positive"));
    }
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
    #[test]
    fn incompatible_identity_rank_alpha_and_unconsumed_weights_fail_typed() {
        let spec = AdapterSpec::new("adapter.safetensors".into(), 0.3, AdapterKind::Lora);
        let mut mutations = Vec::new();
        for (key, value) in [
            ("base_model", "wrong"),
            ("base_revision", "wrong"),
            ("lora_alpha", "NaN"),
            ("lora_alpha", "0"),
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

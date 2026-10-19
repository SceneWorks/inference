//! Exact-parent, transformer-only portable adapters. The supplied DMD stack is independent.
use crate::transformer::Transformer;
use mlx_gen::{
    adapters::AdaptableHost,
    gen_core::{self, moss_image::MossImageVariant, moss_image_training as contract},
    runtime::{AdapterKind, AdapterSpec},
    Error, Result,
};

pub(crate) fn apply(
    host: &mut Transformer,
    variant: MossImageVariant,
    specs: &[AdapterSpec],
) -> Result<Vec<gen_core::AdapterApplyReport>> {
    let allowed = host.adaptable_paths();
    let mut expected_counts = Vec::new();
    // Validate the entire collection before the first install. No partial/incompatible application.
    for spec in specs {
        if !spec.scale.is_finite() || spec.pass_scales.is_some() || spec.moe_expert.is_some() {
            return Err(Error::Unsupported(
                "MOSS adapters require a finite uniform scale without pass/expert overrides".into(),
            ));
        }
        let meta = gen_core::weightsmeta::safetensors_file_metadata(&spec.path)?;
        contract::validate_metadata(variant, &meta)?;
        let lokr = meta.get("networkType").is_some_and(|v| v == "lokr");
        if !matches!(
            (&spec.kind, lokr),
            (AdapterKind::Lora, false) | (AdapterKind::Lokr, true)
        ) {
            return Err(Error::Unsupported(
                "MOSS adapter kind does not match its safetensors metadata".into(),
            ));
        }
        let tensors = gen_core::weightsmeta::safetensors_path_tensor_headers(&spec.path)?;
        if tensors.is_empty() {
            return Err(Error::Unsupported("empty MOSS adapter".into()));
        }
        let suffixes: &[&str] = if lokr {
            &[".lokr_w1", ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b"]
        } else {
            &[".lora_A.weight", ".lora_B.weight", ".alpha"]
        };
        let mut targets = std::collections::BTreeSet::new();
        let mut ranks = std::collections::BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
        for tensor in tensors {
            if tensor.dtype != gen_core::weightsmeta::Dtype::F32 {
                return Err(Error::Unsupported(format!(
                    "MOSS adapter tensor {} must be FP32",
                    tensor.name
                )));
            }
            let Some((path, factor)) =
                gen_core::weightsmeta::split_factor_key(&tensor.name, suffixes)
            else {
                return Err(Error::Unsupported(format!(
                    "unsupported MOSS adapter tensor {}",
                    tensor.name
                )));
            };
            targets.insert(path.to_owned());
            if !allowed.iter().any(|p| p == path) {
                return Err(Error::Unsupported(format!(
                    "MOSS adapter target {path} is not a transformer projection"
                )));
            }
            if !lokr {
                let parts: Vec<_> = path.split('.').collect();
                let facts = host
                    .adaptable_facts(&parts)
                    .ok_or_else(|| Error::Unsupported(format!("unknown MOSS projection {path}")))?;
                let shape = &tensor.shape;
                let pair = ranks.entry(path.to_owned()).or_default();
                match factor {
                    "lora_A.weight"
                        if shape.len() == 2
                            && shape[0] > 0
                            && shape[1] == facts.base_shape[1] as usize =>
                    {
                        pair.0 = Some(shape[0])
                    }
                    "lora_B.weight"
                        if shape.len() == 2
                            && shape[1] > 0
                            && shape[0] == facts.base_shape[0] as usize =>
                    {
                        pair.1 = Some(shape[1])
                    }
                    "alpha" if shape.iter().product::<usize>() == 1 => {}
                    _ => {
                        return Err(Error::Unsupported(format!(
                            "MOSS adapter tensor {} has incompatible shape {:?}",
                            tensor.name, shape
                        )))
                    }
                }
            }
        }
        if ranks
            .values()
            .any(|(a, b)| matches!((a,b),(Some(a),Some(b)) if a!=b))
        {
            return Err(Error::Unsupported("MOSS LoRA A/B ranks disagree".into()));
        }
        expected_counts.push(targets.len());
    }
    let mut reports = Vec::new();
    for (spec, expected) in specs.iter().zip(expected_counts) {
        let weights = mlx_gen::weights::Weights::from_file(&spec.path)?;
        for key in weights.keys() {
            if !weights.require(key)?.is_finite()?.all(None)?.item::<bool>() {
                return Err(Error::Unsupported(format!(
                    "MOSS adapter has nonfinite tensor {key}"
                )));
            }
        }
        let r = match spec.kind {
            AdapterKind::Lora => {
                mlx_gen::adapters::loader::apply_lora_peft(host, &weights, spec.scale, None)?
            }
            AdapterKind::Lokr => mlx_gen::adapters::loader::apply_lokr(host, &weights, spec.scale)?,
        };
        if r.applied != expected || !r.unmatched_paths.is_empty() {
            return Err(Error::Unsupported(format!(
                "MOSS adapter could not apply every target: {:?}",
                r.unmatched_paths
            )));
        }
        reports.push(gen_core::AdapterApplyReport {
            adapter_path: spec.path.clone(),
            applied: r.applied,
            skipped: Vec::new(),
        });
    }
    let mut bytes = 0u64;
    for path in &allowed {
        let parts: Vec<_> = path.split('.').collect();
        let linear = host
            .adaptable_mut(&parts)
            .ok_or_else(|| Error::Msg("MOSS adapter path disappeared".into()))?;
        linear.materialize_adapters()?;
        for adapter in linear.adapters() {
            bytes += match adapter {
                mlx_gen::adapters::Adapter::Lora { a, b, .. } => (a.nbytes() + b.nbytes()) as u64,
                mlx_gen::adapters::Adapter::Lokr { delta, .. } => delta.nbytes() as u64,
                mlx_gen::adapters::Adapter::LokrStructured { .. } => {
                    return Err(Error::Unsupported(
                        "MOSS requires dense LoKr residuals".into(),
                    ))
                }
            };
        }
    }
    host.custom_owned_bytes = bytes;
    Ok(reports)
}

/// Header-derived adapter working set, not a measurement. Dense LoKr additionally owns a full
/// FP32 [out,in] delta per targeted projection; counting its small file alone is unsafe.
pub(crate) fn estimated_bytes(
    specs: &[AdapterSpec],
    config: &gen_core::moss_image::MossImageConfig,
) -> Result<u64> {
    let allowed = contract::target_paths(config.num_layers, &[])?;
    let mut bytes = 0u64;
    for spec in specs {
        let mut paths = std::collections::BTreeSet::new();
        for header in gen_core::weightsmeta::safetensors_path_tensor_headers(&spec.path)? {
            bytes = bytes
                .checked_add(header.data_bytes)
                .ok_or_else(|| Error::Unsupported("MOSS adapter byte overflow".into()))?;
            if spec.kind == AdapterKind::Lokr {
                let (path, _) = gen_core::weightsmeta::split_factor_key(
                    &header.name,
                    &gen_core::weightsmeta::LOKR_SUFFIXES,
                )
                .ok_or_else(|| {
                    Error::Unsupported(format!("unknown MOSS LoKr factor {}", header.name))
                })?;
                if !allowed.iter().any(|p| p == path) {
                    return Err(Error::Unsupported(format!(
                        "unknown MOSS LoKr target {path}"
                    )));
                }
                paths.insert(path.to_owned());
            }
        }
        for path in paths {
            let (input, output) = if path.ends_with(".ffn_up_gate") {
                (config.dim, 2 * config.ffn_dim)
            } else if path.ends_with(".ffn_down") {
                (config.ffn_dim, config.dim)
            } else {
                (config.dim, config.dim)
            };
            let delta = (input as u64)
                .checked_mul(output as u64)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| Error::Unsupported("MOSS LoKr delta size overflow".into()))?;
            bytes = bytes
                .checked_add(delta)
                .ok_or_else(|| Error::Unsupported("MOSS adapter byte overflow".into()))?;
        }
    }
    Ok(bytes)
}

/// Materialize under the model lifecycle's exact prepared tokens, never repin a changed adapter
/// during a mode switch. Lazy factor reads complete before the post-verification.
pub(crate) fn apply_pinned(
    host: &mut Transformer,
    variant: MossImageVariant,
    specs: &[AdapterSpec],
    pins: &[gen_core::PinnedWeightsFile],
) -> Result<Vec<gen_core::AdapterApplyReport>> {
    if pins.len() != specs.len()
        || pins
            .iter()
            .zip(specs)
            .any(|(pin, spec)| pin.loader_path() != spec.path)
    {
        return Err(Error::Unsupported(
            "MOSS adapter source-token set does not match the requested stack".into(),
        ));
    }
    for pin in pins {
        pin.verify_unchanged()?;
    }
    let reports = apply(host, variant, specs)?;
    for pin in pins {
        pin.verify_unchanged()?;
    }
    Ok(reports)
}

//! Inference-side LTX-2.3 LoRA loading.
//!
//! The native trainer writes PEFT factors for the video DiT attention projections. Inference keeps
//! both dense and MLX-packed bases unchanged and attaches those factors through candle-gen's shared
//! additive [`LoraLinear`](candle_gen::train::lora::LoraLinear) core. This avoids materializing a
//! 22B dense transformer merely to fold a small adapter and preserves q4/q8 residency.

use std::collections::{BTreeMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::{weightsmeta, AdapterKind, AdapterSpec};
use candle_gen::train::lora::LoraAdapterMeta;
use candle_gen::train::merge::{no_target_matched, read_adapter, read_scalar, LoraTriple, Role};
use candle_gen::{CandleError, Result};

use crate::transformer::AvDiT;

const PATH_PREFIXES: [&str; 6] = [
    "base_model.model.diffusion_model.",
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "",
];

/// Result of installing one or more LTX LoRAs.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct AdditiveReport {
    /// Number of `(adapter file, projection)` residuals installed.
    pub applied: usize,
    /// Non-LoRA keys ignored while parsing (metadata lives in the safetensors header, not here).
    pub skipped_keys: usize,
}

struct PendingLora {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

fn strip_path_prefix(path: &str) -> &str {
    PATH_PREFIXES
        .iter()
        .find_map(|prefix| path.strip_prefix(prefix))
        .unwrap_or(path)
}

fn is_video_attention_target(path: &str) -> bool {
    let mut parts = path.split('.');
    matches!(parts.next(), Some("transformer_blocks"))
        && parts
            .next()
            .is_some_and(|part| part.parse::<usize>().is_ok())
        && matches!(parts.next(), Some("attn1" | "attn2"))
        && matches!(
            (parts.next(), parts.next(), parts.next()),
            (Some("to_q" | "to_k" | "to_v"), None, None) | (Some("to_out"), Some("0"), None)
        )
}

fn classify_lora_key(key: &str) -> Option<(String, Role)> {
    let key = strip_path_prefix(key);
    for (suffix, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = key.strip_suffix(suffix) {
            return Some((path.to_string(), role));
        }
    }
    None
}

fn effective_scale(spec: &AdapterSpec) -> Result<f64> {
    if spec.moe_expert.is_some() {
        return Err(CandleError::Msg(format!(
            "ltx: adapter {} specifies a Wan MoE expert; LTX has one video DiT",
            spec.path.display()
        )));
    }
    match &spec.pass_scales {
        None => Ok(spec.scale as f64),
        Some(scales) if scales.len() == 1 => Ok(scales[0] as f64),
        Some(scales) => Err(CandleError::Msg(format!(
            "ltx: adapter {} has {} pass_scales entries, but candle LTX-2.3 uses one distilled denoise pass",
            spec.path.display(),
            scales.len()
        ))),
    }
}

/// Read and install LoRA adapters on the canonical video attention surface.
///
/// A resolved module outside the exact training surface is an error, as is an incomplete or
/// shape-mismatched pair. Thus a non-empty adapter can never silently render as the base model.
pub fn install_ltx_adapters(dit: &mut AvDiT, specs: &[AdapterSpec]) -> Result<AdditiveReport> {
    let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut report = AdditiveReport::default();

    for spec in specs {
        if spec.kind == AdapterKind::Lokr {
            return Err(CandleError::Msg(format!(
                "ltx: LoKr adapters are not supported; use a LoRA adapter. Offending file: {}",
                spec.path.display()
            )));
        }
        let scale = effective_scale(spec)?;
        let file = read_adapter(&spec.path)?;
        if file.declares_lokr()
            || weightsmeta::keys_contain_lokr(file.tensors.keys().map(String::as_str))
        {
            return Err(CandleError::Msg(format!(
                "ltx: adapter {} contains LoKr factors, which this provider does not support; use LoRA",
                spec.path.display()
            )));
        }

        let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
        for (key, tensor) in &file.tensors {
            match classify_lora_key(key) {
                Some((path, Role::Down)) => {
                    triples.entry(path).or_default().down = Some(tensor.clone())
                }
                Some((path, Role::Up)) => {
                    triples.entry(path).or_default().up = Some(tensor.clone())
                }
                Some((path, Role::Alpha)) => {
                    triples.entry(path).or_default().alpha =
                        Some(read_scalar(key, "alpha", tensor)?)
                }
                None => report.skipped_keys += 1,
            }
        }
        let metadata = LoraAdapterMeta::from_file_metadata(&file.meta);
        for (path, triple) in triples {
            if !is_video_attention_target(&path) {
                return Err(CandleError::Msg(format!(
                    "ltx: LoRA target `{path}` is outside the canonical video attention surface \
                     `transformer_blocks.N.attn{{1,2}}.{{to_q,to_k,to_v,to_out.0}}`"
                )));
            }
            let (down, up) = match (triple.down, triple.up) {
                (Some(down), Some(up)) => (down, up),
                _ => {
                    return Err(CandleError::Msg(format!(
                        "ltx: LoRA target `{path}` has an incomplete A/B factor pair"
                    )))
                }
            };
            let (rank, in_features) = down.dims2().map_err(|_| {
                CandleError::Msg(format!("ltx: LoRA target `{path}` A factor must be rank-2"))
            })?;
            let (out_features, up_rank) = up.dims2().map_err(|_| {
                CandleError::Msg(format!("ltx: LoRA target `{path}` B factor must be rank-2"))
            })?;
            if rank == 0 || up_rank != rank {
                return Err(CandleError::Msg(format!(
                    "ltx: LoRA target `{path}` has incompatible factor shapes {:?} / {:?}",
                    down.dims(),
                    up.dims()
                )));
            }
            let (meta_alpha, meta_rank) = metadata
                .as_ref()
                .map_or((None, None), |meta| meta.effective(&path));
            let scale_rank = meta_rank.unwrap_or(rank as f32);
            if scale_rank <= 0.0 || !scale_rank.is_finite() {
                return Err(CandleError::Msg(format!(
                    "ltx: LoRA target `{path}` has invalid metadata rank {scale_rank}"
                )));
            }
            let alpha = triple.alpha.or(meta_alpha).unwrap_or(scale_rank);
            if !alpha.is_finite() {
                return Err(CandleError::Msg(format!(
                    "ltx: LoRA target `{path}` has non-finite alpha"
                )));
            }
            let a = down.to_dtype(DType::F32)?.t()?.contiguous()?;
            let b =
                (up.to_dtype(DType::F32)?.t()?.contiguous()? * (alpha as f64 / scale_rank as f64))?;
            debug_assert_eq!(a.dims(), &[in_features, rank]);
            debug_assert_eq!(b.dims(), &[rank, out_features]);
            pending
                .entry(path)
                .or_default()
                .push(PendingLora { a, b, scale });
        }
    }

    let device = dit.device().clone();
    let mut matched = HashSet::new();
    dit.visit_video_adaptable_mut(&mut |loaded_path, linear| {
        let path = strip_path_prefix(loaded_path);
        if let Some(residuals) = pending.get(path) {
            let (out_features, in_features) = linear.base_shape();
            for residual in residuals {
                if residual.a.dims()[0] != in_features
                    || residual.b.dims()[0] != residual.a.dims()[1]
                    || residual.b.dims()[1] != out_features
                {
                    return Err(candle_gen::candle_core::Error::Msg(format!(
                        "ltx: LoRA target `{path}` factor shapes {:?} / {:?} do not match base [{out_features}, {in_features}]",
                        residual.a.dims(),
                        residual.b.dims()
                    )));
                }
                linear.push_additive_lora(
                    residual.a.to_device(&device)?,
                    residual.b.to_device(&device)?,
                    residual.scale,
                );
                report.applied += 1;
            }
            matched.insert(path.to_string());
        }
        Ok(())
    })?;

    let unmatched: Vec<&str> = pending
        .keys()
        .filter(|path| !matched.contains(path.as_str()))
        .map(String::as_str)
        .collect();
    if !unmatched.is_empty() {
        return Err(CandleError::Msg(format!(
            "ltx: adapter target(s) did not match the loaded DiT: {}",
            unmatched.join(", ")
        )));
    }
    if !specs.is_empty() && report.applied == 0 {
        return Err(no_target_matched(
            "ltx",
            "expected PEFT `<path>.lora_A/B.weight` keys over \
             `transformer_blocks.N.attn{1,2}.{to_q,to_k,to_v,to_out.0}`",
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping_is_exact() {
        for path in [
            "transformer_blocks.0.attn1.to_q",
            "transformer_blocks.47.attn2.to_k",
            "transformer_blocks.3.attn1.to_v",
            "transformer_blocks.9.attn2.to_out.0",
        ] {
            let key = format!("model.diffusion_model.{path}.lora_A.weight");
            assert_eq!(
                classify_lora_key(&key),
                Some((path.to_string(), Role::Down))
            );
            assert!(is_video_attention_target(path));
        }
        for path in [
            "transformer_blocks.0.audio_attn1.to_q",
            "transformer_blocks.0.attn1.to_gate_logits",
            "transformer_blocks.0.ff.net.0.proj",
            "transformer_blocks.x.attn1.to_q",
        ] {
            assert!(!is_video_attention_target(path), "{path}");
        }
    }
}

//! Inference-side LTX-2.3 LoRA / PEFT-LoKr loading.
//!
//! The native trainer writes PEFT factors for video attention, while the official Eros distill LoRA
//! also targets feed-forward, gated attention, audio, and cross-modal projections. Inference keeps
//! both dense and MLX-packed bases unchanged and attaches all resolved factors through candle-gen's shared
//! additive [`LoraLinear`](candle_gen::train::lora::LoraLinear) core. LoKr uses the allocation-free
//! Kronecker vec-trick. This avoids materializing a 22B dense transformer merely to fold a small
//! adapter and preserves q4/q8 residency. Untagged third-party LyCORIS LoKr is admitted with its
//! per-module scale. LyCORIS LoHa and full-rank diff patches are rejected explicitly because they
//! have no truthful allocation-free representation on the packed host.

use std::collections::{BTreeMap, HashSet};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::{weightsmeta, AdapterKind, AdapterSpec};
use candle_gen::quant::LokrFactors;
use candle_gen::train::lora::{parse_lokr_metadata, LoraAdapterMeta};
use candle_gen::train::merge::{
    has_diff_patch_keys, parse_lokr_thirdparty, read_adapter, read_scalar, AdapterFile, LoraTriple,
    Role,
};
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

/// Result of installing one or more LTX LoRA/LoKr adapters.
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
    pass_scales: Vec<f64>,
    /// Index of the selected [`AdapterSpec`] that supplied this residual.
    source: usize,
}

#[derive(Default)]
struct LokrGroup {
    w1: Option<Tensor>,
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
}

struct PendingLokr {
    factors: LokrGroup,
    /// Alpha/rank and any LyCORIS module scale. User pass strengths remain separate so the
    /// structured residual can select stage one or two without rebuilding the DiT.
    factor_scale: f64,
    pass_scales: Vec<f64>,
    /// Index of the selected [`AdapterSpec`] that supplied this residual.
    source: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterFormat {
    Lora,
    Lokr,
    ThirdPartyLokr,
}

fn classify_format(spec: &AdapterSpec, file: &AdapterFile) -> Result<AdapterFormat> {
    if weightsmeta::keys_contain_loha(file.tensors.keys().map(String::as_str)) {
        return Err(CandleError::Msg(format!(
            "ltx: LoHa factors have no allocation-free structured residual on a packed LTX \
             projection; use LoRA/PEFT-LoKr instead. Offending file: {}",
            spec.path.display()
        )));
    }
    let has_lokr_keys = weightsmeta::keys_contain_lokr(file.tensors.keys().map(String::as_str));
    if spec.kind == AdapterKind::Lokr && !file.declares_lokr() && !has_lokr_keys {
        return Err(CandleError::Msg(format!(
            "ltx: adapter {} was declared LoKr but contains no lokr_w1/w2 factors",
            spec.path.display()
        )));
    }
    if file.declares_lokr() && !has_lokr_keys {
        return Err(CandleError::Msg(format!(
            "ltx: adapter {} declares LoKr but contains no lokr_w1/w2 factors",
            spec.path.display()
        )));
    }
    Ok(if file.declares_lokr() {
        AdapterFormat::Lokr
    } else if has_lokr_keys {
        AdapterFormat::ThirdPartyLokr
    } else {
        AdapterFormat::Lora
    })
}

fn ensure_no_diff_patch(spec: &AdapterSpec) -> Result<()> {
    if has_diff_patch_keys(&spec.path)? {
        return Err(CandleError::Msg(format!(
            "ltx: full-rank diff-patch adapters cannot be applied without materializing every \
             adapted 22B projection; use a LoRA/PEFT-LoKr adapter. Offending file: {}",
            spec.path.display()
        )));
    }
    Ok(())
}

fn strip_path_prefix(path: &str) -> &str {
    PATH_PREFIXES
        .iter()
        .find_map(|prefix| path.strip_prefix(prefix))
        .unwrap_or(path)
}

/// Map published diffusers/PEFT LTX module names onto Candle's provider namespace.
///
/// `cond_safe` uses the rootless `diffusion_model.` namespace while Candle's dense and split-tier
/// builders are rooted at `model.diffusion_model.`. The two layouts describe the same AudioVideo
/// tree; normalize the wrapper before routing rather than treating the 1,660 non-video projections
/// as unknown. Keep the Lightricks projection spelling (`to_out.0`, `ff.net.0.proj`, `ff.net.2`) —
/// the tier's `to_out`/`ff.proj_*` spellings are already translated by its VarBuilder remapper and
/// must never leak into adapter matching.
fn normalize_ltx_adapter_path(path: &str) -> String {
    strip_path_prefix(path).to_string()
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
            return Some((normalize_ltx_adapter_path(path), role));
        }
    }
    None
}

fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    let key = strip_path_prefix(key);
    for suffix in [
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_w1",
        "lokr_w2",
    ] {
        if let Some(path) = key.strip_suffix(&format!(".{suffix}")) {
            return Some((normalize_ltx_adapter_path(path), suffix));
        }
    }
    None
}

/// User strengths for the distilled LTX stages. A scalar remains uniform for third-party adapters;
/// Eros supplies the explicit two-stage `[stage1, stage2]` vector.
fn effective_pass_scales(spec: &AdapterSpec) -> Result<Vec<f64>> {
    if spec.moe_expert.is_some() {
        return Err(CandleError::Msg(format!(
            "ltx: adapter {} specifies a Wan MoE expert; LTX has one video DiT",
            spec.path.display()
        )));
    }
    let scales: Vec<f64> = match &spec.pass_scales {
        None => vec![spec.scale as f64],
        Some(scales) if matches!(scales.len(), 1 | 2) => {
            scales.iter().copied().map(f64::from).collect()
        }
        Some(scales) => {
            return Err(CandleError::Msg(format!(
            "ltx: adapter {} has {} pass_scales entries; LTX-2.3 accepts a uniform single scale \
                 or its two-stage [stage1, stage2] shape",
            spec.path.display(),
            scales.len()
        )))
        }
    };
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(CandleError::Msg(format!(
            "ltx: adapter {} effective pass scales must be finite",
            spec.path.display()
        )));
    }
    Ok(scales)
}

/// Read and install LoRA / PEFT-LoKr adapters on the complete LTX AudioVideo projection surface.
///
/// A resolved module outside the exact training surface is an error, as is an incomplete or
/// shape-mismatched pair. Thus a non-empty adapter can never silently render as the base model.
pub fn install_ltx_adapters(dit: &mut AvDiT, specs: &[AdapterSpec]) -> Result<AdditiveReport> {
    let mut pending: BTreeMap<String, Vec<PendingLora>> = BTreeMap::new();
    let mut pending_lokr: BTreeMap<String, Vec<PendingLokr>> = BTreeMap::new();
    let mut report = AdditiveReport::default();

    for (source, spec) in specs.iter().enumerate() {
        ensure_no_diff_patch(spec)?;
        let pass_scales = effective_pass_scales(spec)?;
        let file = read_adapter(&spec.path)?;
        let format = classify_format(spec, &file)?;
        if format == AdapterFormat::Lokr {
            let (rank, alpha) = parse_lokr_metadata(
                file.meta.get("rank").map(String::as_str),
                file.meta.get("alpha").map(String::as_str),
            )?;
            let factor_scale = alpha as f64 / rank as f64;
            let mut groups: BTreeMap<String, LokrGroup> = BTreeMap::new();
            for (key, tensor) in &file.tensors {
                if let Some((path, factor)) = classify_lokr_key(key) {
                    let group = groups.entry(path).or_default();
                    match factor {
                        "lokr_w1" => group.w1 = Some(tensor.clone()),
                        "lokr_w1_a" => group.w1_a = Some(tensor.clone()),
                        "lokr_w1_b" => group.w1_b = Some(tensor.clone()),
                        "lokr_w2" => group.w2 = Some(tensor.clone()),
                        "lokr_w2_a" => group.w2_a = Some(tensor.clone()),
                        "lokr_w2_b" => group.w2_b = Some(tensor.clone()),
                        _ => unreachable!(),
                    }
                } else {
                    report.skipped_keys += 1;
                }
            }
            for (path, factors) in groups {
                pending_lokr.entry(path).or_default().push(PendingLokr {
                    factors,
                    factor_scale,
                    pass_scales: pass_scales.clone(),
                    source,
                });
            }
            continue;
        }
        if format == AdapterFormat::ThirdPartyLokr {
            for (raw_path, group) in parse_lokr_thirdparty(&file)? {
                let path = normalize_ltx_adapter_path(&raw_path);
                let lycoris_scale = if let Some(factor) = group.w1_a.as_ref() {
                    let rank = factor.dims()[1] as f64;
                    group.alpha.map_or(1.0, |alpha| alpha as f64 / rank)
                } else if let Some(factor) = group.w2_a.as_ref() {
                    let rank = factor.dims()[1] as f64;
                    group.alpha.map_or(1.0, |alpha| alpha as f64 / rank)
                } else {
                    // When both Kronecker legs are full LyCORIS forces alpha=rank, hence scale one.
                    1.0
                };
                pending_lokr.entry(path).or_default().push(PendingLokr {
                    factors: LokrGroup {
                        w1: group.w1,
                        w1_a: group.w1_a,
                        w1_b: group.w1_b,
                        w2: group.w2,
                        w2_a: group.w2_a,
                        w2_b: group.w2_b,
                    },
                    factor_scale: lycoris_scale,
                    pass_scales: pass_scales.clone(),
                    source,
                });
            }
            continue;
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
            pending.entry(path).or_default().push(PendingLora {
                a,
                b,
                pass_scales: pass_scales.clone(),
                source,
            });
        }
    }

    let device = dit.device().clone();
    let mut matched = HashSet::new();
    // Keep the originating selected spec through parsing and target installation. A stack-level
    // aggregate is insufficient: one valid Eros/distill adapter must not make a second, unrelated
    // selected adapter look applied merely because both were merged into the same pending maps.
    let mut applied_by_spec = vec![0usize; specs.len()];
    dit.visit_adaptable_mut(&mut |loaded_path, linear| {
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
                linear.push_additive_lora_per_pass(
                    residual.a.to_device(&device)?,
                    residual.b.to_device(&device)?,
                    residual.pass_scales.clone(),
                )?;
                report.applied += 1;
                applied_by_spec[residual.source] += 1;
            }
            matched.insert(path.to_string());
        }
        if let Some(residuals) = pending_lokr.get(path) {
            let (out_features, in_features) = linear.base_shape();
            for residual in residuals {
                let group = &residual.factors;
                let factors = LokrFactors::build(
                    residual.factor_scale,
                    (out_features, in_features),
                    group.w1.as_ref(),
                    group.w1_a.as_ref(),
                    group.w1_b.as_ref(),
                    group.w2.as_ref(),
                    None,
                    group.w2_a.as_ref(),
                    group.w2_b.as_ref(),
                )
                .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?
                .ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(format!(
                        "ltx: LoKr target `{path}` has no allocation-free 2-D structured form"
                    ))
                })?;
                let factors = factors
                    .to_device(&device)
                    .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
                linear.push_additive_lokr_per_pass(factors, residual.pass_scales.clone())?;
                report.applied += 1;
                applied_by_spec[residual.source] += 1;
            }
            matched.insert(path.to_string());
        }
        Ok(())
    })?;

    if let Some((source, spec)) = specs
        .iter()
        .enumerate()
        .find(|(source, _)| applied_by_spec[*source] == 0)
    {
        return Err(CandleError::Msg(format!(
            "ltx: selected adapter #{} ({}) applied zero projection residuals; expected PEFT \
             `<path>.lora_A/B.weight` or `<path>.lokr_w1/w2` keys over the LTX-2.3 AudioVideo \
             transformer projection surface",
            source + 1,
            spec.path.display()
        )));
    }

    let unmatched: Vec<&str> = pending
        .keys()
        .chain(pending_lokr.keys())
        .filter(|path| !matched.contains(path.as_str()))
        .map(String::as_str)
        .collect();
    if !unmatched.is_empty() {
        return Err(CandleError::Msg(format!(
            "ltx: adapter target(s) did not match the loaded DiT: {}",
            unmatched.join(", ")
        )));
    }
    debug_assert_eq!(report.applied, applied_by_spec.into_iter().sum::<usize>());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::{Linear, Module};
    use candle_gen::train::lora::LoraLinear;

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
        }
        for path in [
            "transformer_blocks.0.audio_attn1.to_q",
            "transformer_blocks.0.attn1.to_gate_logits",
            "transformer_blocks.0.ff.net.0.proj",
            "transformer_blocks.x.attn1.to_q",
        ] {
            assert!(classify_lora_key(&format!("{path}.lora_A.weight")).is_some());
        }
    }

    /// The published cond_safe adapter has 1,660 paired LoRA targets / 3,320 tensors. Its
    /// diffusers `diffusion_model.` wrapper is intentionally different from Candle's
    /// `model.diffusion_model.` builder root, but after normalization the complete AudioVideo
    /// surface must be present. The historical video-only visitor covered 768 tensors; accepting
    /// any such partial map would recreate the noise-render regression.
    #[test]
    fn cond_safe_key_mapping_is_complete_over_all_3320_tensors() {
        let mut provider_paths = BTreeSet::new();
        for stream in ["", "audio_"] {
            provider_paths.insert(format!("{stream}patchify_proj"));
            provider_paths.insert(format!("{stream}proj_out"));
            for controller in [
                format!("{stream}adaln_single"),
                format!("{stream}prompt_adaln_single"),
                if stream.is_empty() {
                    "av_ca_video_scale_shift_adaln_single".into()
                } else {
                    "av_ca_audio_scale_shift_adaln_single".into()
                },
                if stream.is_empty() {
                    "av_ca_a2v_gate_adaln_single".into()
                } else {
                    "av_ca_v2a_gate_adaln_single".into()
                },
            ] {
                provider_paths.insert(format!("{controller}.emb.timestep_embedder.linear_1"));
                provider_paths.insert(format!("{controller}.emb.timestep_embedder.linear_2"));
                provider_paths.insert(format!("{controller}.linear"));
            }
        }
        for block in 0..48 {
            for attention in [
                "attn1",
                "attn2",
                "audio_attn1",
                "audio_attn2",
                "audio_to_video_attn",
                "video_to_audio_attn",
            ] {
                for projection in ["to_q", "to_k", "to_v", "to_out.0", "to_gate_logits"] {
                    provider_paths.insert(format!(
                        "transformer_blocks.{block}.{attention}.{projection}"
                    ));
                }
            }
            for feed_forward in ["ff", "audio_ff"] {
                provider_paths.insert(format!(
                    "transformer_blocks.{block}.{feed_forward}.net.0.proj"
                ));
                provider_paths.insert(format!("transformer_blocks.{block}.{feed_forward}.net.2"));
            }
        }
        assert_eq!(provider_paths.len(), 1_660);

        let mut mapped = BTreeSet::new();
        let mut mapped_tensors = 0;
        for path in &provider_paths {
            for suffix in ["lora_A.weight", "lora_B.weight"] {
                let key = format!("diffusion_model.{path}.{suffix}");
                let (mapped_path, _) = classify_lora_key(&key).expect("published PEFT key");
                assert!(
                    provider_paths.contains(&mapped_path),
                    "unmapped target: {key}"
                );
                mapped.insert(mapped_path);
                mapped_tensors += 1;
            }
        }
        assert_eq!(mapped_tensors, 3_320);
        assert_eq!(mapped, provider_paths);
    }

    #[test]
    fn eros_pass_scales_preserve_both_distilled_stages() {
        let mut spec = AdapterSpec::new("eros.safetensors".into(), 0.25, AdapterKind::Lora);
        spec.pass_scales = Some(vec![0.8, 0.2]);
        assert_eq!(
            effective_pass_scales(&spec).unwrap(),
            vec![0.8_f32 as f64, 0.2_f32 as f64]
        );

        spec.pass_scales = Some(vec![0.6]);
        assert_eq!(effective_pass_scales(&spec).unwrap(), vec![0.6_f32 as f64]);

        spec.pass_scales = Some(vec![0.6, 0.4, 0.2]);
        let error = effective_pass_scales(&spec).unwrap_err().to_string();
        assert!(error.contains("3 pass_scales") && error.contains("[stage1, stage2]"));
    }

    #[test]
    fn effective_pass_scales_reject_non_finite_entries() {
        let spec = AdapterSpec::new("nan.safetensors".into(), f32::NAN, AdapterKind::Lora);
        assert!(effective_pass_scales(&spec)
            .unwrap_err()
            .to_string()
            .contains("effective pass scales must be finite"));

        let mut spec = AdapterSpec::new("inf.safetensors".into(), 1.0, AdapterKind::Lora);
        spec.pass_scales = Some(vec![f32::INFINITY, 0.25]);
        assert!(effective_pass_scales(&spec)
            .unwrap_err()
            .to_string()
            .contains("effective pass scales must be finite"));
    }

    #[test]
    fn peft_lokr_routes_to_structured_residual_and_moves_forward() {
        let dev = Device::Cpu;
        let spec = AdapterSpec::new("peft-lokr.safetensors".into(), 1.0, AdapterKind::Lokr);
        let file = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.0.attn1.to_q.lokr_w1".into(),
                    Tensor::from_vec(vec![2f32], (1, 1), &dev).unwrap(),
                ),
                (
                    "transformer_blocks.0.attn1.to_q.lokr_w2".into(),
                    Tensor::from_vec(vec![3f32], (1, 1), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::from([
                ("networkType".into(), "lokr".into()),
                ("rank".into(), "1".into()),
                ("alpha".into(), "1".into()),
            ]),
        };
        assert_eq!(classify_format(&spec, &file).unwrap(), AdapterFormat::Lokr);
        for key in file.tensors.keys() {
            let (path, _factor) = classify_lokr_key(key).expect("canonical LoKr factor");
            assert_eq!(path, "transformer_blocks.0.attn1.to_q");
        }

        let factors = LokrFactors::build(
            1.0,
            (1, 1),
            file.tensors.get("transformer_blocks.0.attn1.to_q.lokr_w1"),
            None,
            None,
            file.tensors.get("transformer_blocks.0.attn1.to_q.lokr_w2"),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("1x1 LoKr is structured");
        let base = Tensor::zeros((1, 1), DType::F32, &dev).unwrap();
        let mut linear = LoraLinear::from_linear(
            Linear::new(base, None),
            1,
            1,
            "transformer_blocks.0.attn1.to_q".into(),
        );
        linear.push_additive_lokr(factors);
        let out = linear
            .forward(&Tensor::from_vec(vec![5f32], (1, 1), &dev).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, [30.0]);
    }

    #[test]
    fn lycoris_lokr_is_admitted_while_loha_is_rejected_explicitly() {
        let dev = Device::Cpu;
        let spec = AdapterSpec::new("lycoris.safetensors".into(), 1.0, AdapterKind::Lora);
        let loha = AdapterFile {
            tensors: HashMap::from([(
                "transformer_blocks.0.attn1.to_q.hada_w1_a".into(),
                Tensor::ones((1, 1), DType::F32, &dev).unwrap(),
            )]),
            meta: HashMap::new(),
        };
        let error = classify_format(&spec, &loha).unwrap_err().to_string();
        assert!(error.contains("LoHa") && error.contains("allocation-free"));

        let lokr = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.0.attn1.to_q.lokr_w1".into(),
                    Tensor::ones((1, 1), DType::F32, &dev).unwrap(),
                ),
                (
                    "transformer_blocks.0.attn1.to_q.lokr_w2".into(),
                    Tensor::ones((1, 1), DType::F32, &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        assert_eq!(
            classify_format(&spec, &lokr).unwrap(),
            AdapterFormat::ThirdPartyLokr
        );
        let parsed = parse_lokr_thirdparty(&lokr).unwrap();
        let group = parsed
            .get("transformer_blocks.0.attn1.to_q")
            .expect("dotted LyCORIS target");
        let factors = LokrFactors::build(
            1.0,
            (1, 1),
            group.w1.as_ref(),
            group.w1_a.as_ref(),
            group.w1_b.as_ref(),
            group.w2.as_ref(),
            None,
            group.w2_a.as_ref(),
            group.w2_b.as_ref(),
        )
        .unwrap();
        assert!(factors.is_some(), "LyCORIS LoKr has a structured host form");
    }

    #[test]
    fn diff_patch_is_detected_for_loud_rejection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full-rank.safetensors");
        let tensors: HashMap<String, Tensor> = HashMap::from([(
            "transformer_blocks.0.attn1.to_q.diff".to_string(),
            Tensor::ones((1, 1), DType::F32, &Device::Cpu).unwrap(),
        )]);
        candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
        let error = ensure_no_diff_patch(&spec).unwrap_err().to_string();
        assert!(error.contains("diff-patch") && error.contains("full-rank"));
    }
}

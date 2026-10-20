//! Exact-parent, transformer-only portable adapters. The supplied DMD stack is independent.
use crate::{precision::linear_fp32, transformer::Transformer};
use mlx_gen::{
    adapters::{AdaptableHost, Adapter},
    gen_core::{self, moss_image::MossImageVariant, moss_image_training as contract},
    runtime::{AdapterKind, AdapterSpec},
    Error, Result,
};
use mlx_rs::{ops::multiply, transforms::eval, Array, Dtype};
use std::collections::BTreeMap;

// Same raw-factor admission as Candle, independent of PEFT's alpha folding on reload.
// Check every unscaled FP32 contraction (including decomposed legs) before success/export.
fn contraction_bound(weight: &Array, input: f64) -> Result<f64> {
    let columns = weight.shape()[1] as usize;
    let norm = crate::pipeline::logical_f32_readback(weight)?
        .chunks_exact(columns)
        .map(|row| row.iter().map(|v| f64::from(v.abs())).sum::<f64>())
        .fold(0., f64::max);
    let bound = norm * input;
    if !bound.is_finite() || bound > f32::MAX as f64 {
        return Err(Error::Unsupported(
            "MOSS unscaled factor contraction overflows FP32; reduce factor magnitude".into(),
        ));
    }
    Ok(bound)
}
fn validate_lora_factors(a: &Array, b: &Array, scale: f32) -> Result<()> {
    finite(a, "LoRA A")?;
    finite(b, "LoRA B")?;
    if scale == 0. {
        return Ok(()); // Disabled adapters skip contractions, but source finiteness still applies.
    }
    let raw = contraction_bound(b, contraction_bound(a, 1.)?)?;
    if raw * f64::from(scale.abs()) > f32::MAX as f64 {
        return Err(Error::Unsupported(
            "MOSS scaled LoRA contraction overflows FP32; reduce factor magnitude or strength"
                .into(),
        ));
    }
    Ok(())
}
#[derive(Clone)]
enum Factor {
    Full(Array),
    LowRank { a: Array, b: Array },
}
impl Factor {
    fn from_group(group: &BTreeMap<String, Array>, name: &str) -> Result<Self> {
        match (
            group.get(name),
            group.get(&format!("{name}_a")),
            group.get(&format!("{name}_b")),
        ) {
            (Some(w), None, None) => Ok(Self::Full(w.clone())),
            (None, Some(a), Some(b)) => Ok(Self::LowRank {
                a: a.clone(),
                b: b.clone(),
            }),
            _ => Err(Error::Unsupported(format!(
                "MOSS {name} has incomplete or competing representations"
            ))),
        }
    }
    fn shape(&self) -> (i32, i32) {
        match self {
            Self::Full(w) => (w.shape()[0], w.shape()[1]),
            Self::LowRank { a, b } => (a.shape()[0], b.shape()[1]),
        }
    }
    fn arrays(&self) -> Vec<&Array> {
        match self {
            Self::Full(w) => vec![w],
            Self::LowRank { a, b } => vec![a, b],
        }
    }
    fn rebound(&self, arrays: &[Array], cursor: &mut usize) -> Self {
        match self {
            Self::Full(_) => {
                let w = arrays[*cursor].clone();
                *cursor += 1;
                Self::Full(w)
            }
            Self::LowRank { .. } => {
                let a = arrays[*cursor].clone();
                let b = arrays[*cursor + 1].clone();
                *cursor += 2;
                Self::LowRank { a, b }
            }
        }
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Full(w) => linear_fp32(x, w),
            Self::LowRank { a, b } => linear_fp32(&linear_fp32(x, b)?, a),
        }
    }
    fn forward_bound(&self, input: f64) -> Result<f64> {
        match self {
            Self::Full(w) => contraction_bound(w, input),
            Self::LowRank { a, b } => contraction_bound(a, contraction_bound(b, input)?),
        }
    }
    fn effective_max(&self) -> Result<f64> {
        let product = match self {
            Self::Full(w) => w.clone(),
            Self::LowRank { a, b } => linear_fp32(&b.t(), a)?.t(),
        };
        finite(&product, "reconstructed LoKr factor")?;
        Ok(product.abs()?.max(None)?.item::<f32>() as f64)
    }
}

/// Raw trainable factors, not a reconstructed [out,in] delta. The same structured FP32 GEMVs
/// execute during training, checkpoint recomputation and inference reload.
#[derive(Clone)]
pub(crate) struct Lokr {
    first: Factor,
    second: Factor,
    scale: f32,
}
impl Lokr {
    pub(crate) fn from_group(
        group: &BTreeMap<String, Array>,
        shape: &[i32],
        scale: f32,
    ) -> Result<Self> {
        if shape.len() != 2 || !scale.is_finite() {
            return Err(Error::Unsupported(
                "MOSS LoKr needs a matrix and finite effective scale".into(),
            ));
        }
        let shapes = group
            .iter()
            .map(|(k, v)| (k.clone(), v.shape().iter().map(|d| *d as usize).collect()))
            .collect();
        contract::lokr_layout(&shapes, shape[0] as usize, shape[1] as usize)?;
        Ok(Self {
            first: Factor::from_group(group, "lokr_w1")?,
            second: Factor::from_group(group, "lokr_w2")?,
            scale,
        })
    }
    pub(crate) fn arrays(&self) -> Vec<&Array> {
        let mut arrays = self.first.arrays();
        arrays.extend(self.second.arrays());
        arrays
    }
    pub(crate) fn rebound(&self, arrays: &[Array], cursor: &mut usize) -> Self {
        Self {
            first: self.first.rebound(arrays, cursor),
            second: self.second.rebound(arrays, cursor),
            scale: self.scale,
        }
    }
    pub(crate) fn residual(&self, x: &Array) -> Result<Option<Array>> {
        if self.scale == 0. {
            return Ok(None);
        }
        let (a, c) = self.first.shape();
        let (b, d) = self.second.shape();
        let mut output_shape = x.shape().to_vec();
        let input = output_shape
            .pop()
            .ok_or_else(|| Error::Msg("MOSS LoKr input is scalar".into()))?;
        if input != c * d {
            return Err(Error::Msg("MOSS LoKr input axis mismatch".into()));
        }
        let rows = i32::try_from(x.size() / input as usize)
            .map_err(|_| Error::Unsupported("MOSS LoKr row count overflow".into()))?;
        let x = x.as_dtype(Dtype::Float32)?.reshape(&[rows, c, d])?;
        // Choose the smaller structured intermediate. min(c*b,d*a)<=sqrt(input*output), so
        // no target-sized Kronecker matrix (or pathological c*b intermediate) is allocated.
        let y = if i64::from(c) * i64::from(b) <= i64::from(d) * i64::from(a) {
            self.first
                .forward(&self.second.forward(&x)?.transpose_axes(&[0, 2, 1])?)?
                .transpose_axes(&[0, 2, 1])?
        } else {
            self.second.forward(
                &self
                    .first
                    .forward(&x.transpose_axes(&[0, 2, 1])?)?
                    .transpose_axes(&[0, 2, 1])?,
            )?
        };
        output_shape.push(a * b);
        Ok(Some(multiply(
            y.reshape(&output_shape)?,
            Array::from_f32(self.scale),
        )?))
    }
    pub(crate) fn materialize_and_validate(&self) -> Result<()> {
        for array in self.arrays() {
            eval([array])?;
            finite(array, "LoKr factor")?;
        }
        let (a, c) = self.first.shape();
        let (b, d) = self.second.shape();
        let raw_bound = if self.scale == 0. {
            0.
        } else if i64::from(c) * i64::from(b) <= i64::from(d) * i64::from(a) {
            self.first.forward_bound(self.second.forward_bound(1.)?)?
        } else {
            self.second.forward_bound(self.first.forward_bound(1.)?)?
        };
        if raw_bound * f64::from(self.scale.abs()) > f32::MAX as f64 {
            return Err(Error::Unsupported(
                "MOSS scaled LoKr contraction overflows FP32; reduce factor magnitude or strength"
                    .into(),
            ));
        }
        // Kronecker coefficient maxima factor exactly. Validate derived (possibly decomposed)
        // factors and their effective product, not just finite source tensors.
        let first = self.first.effective_max()?;
        let second = self.second.effective_max()?;
        let envelope = first * second * (self.scale as f64).abs();
        if !envelope.is_finite() || envelope > f32::MAX as f64 {
            return Err(Error::Unsupported("MOSS LoKr factors overflow the FP32 effective adapter; reduce factor magnitude or strength".into()));
        }
        Ok(())
    }
}
fn finite(array: &Array, label: &str) -> Result<()> {
    if !array.is_finite()?.all(None)?.item::<bool>() {
        return Err(Error::Unsupported(format!("MOSS nonfinite {label}")));
    }
    Ok(())
}
/// Check the actual installed/folded arrays before reporting application success.
pub(crate) fn validate_lora(adapter: &Adapter) -> Result<()> {
    match adapter {
        Adapter::Lora { a, b, scale } => {
            // Shared Adapter stores [in,rank] / [rank,out], unlike raw PEFT [rank,in] /
            // [out,rank]. Validate the actual contraction axes, not the storage row axes.
            validate_lora_factors(&a.t(), &b.t(), *scale)?;
            finite(&multiply(b, Array::from_f32(*scale))?, "scaled LoRA B")
        }
        _ => Err(Error::Unsupported(
            "MOSS LoKr must retain raw structured factors, not dense reconstructed deltas".into(),
        )),
    }
}

pub(crate) fn apply(
    host: &mut Transformer,
    variant: MossImageVariant,
    specs: &[AdapterSpec],
) -> Result<Vec<gen_core::AdapterApplyReport>> {
    let allowed = host.adaptable_paths();
    let mut validated = Vec::new();
    // Header/metadata validation for the entire stack precedes installation.
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
        let scale = if lokr {
            let (rank, alpha) = contract::lokr_scaling(&meta)?;
            let scale = (alpha / rank) * spec.scale;
            if !scale.is_finite() {
                return Err(Error::Unsupported(
                    "MOSS LoKr effective scale overflows FP32".into(),
                ));
            }
            scale
        } else {
            spec.scale
        };
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&spec.path)?;
        if headers.is_empty() {
            return Err(Error::Unsupported("empty MOSS adapter".into()));
        }
        let suffixes: &[&str] = if lokr {
            &gen_core::weightsmeta::LOKR_SUFFIXES
        } else {
            &[".lora_A.weight", ".lora_B.weight", ".alpha"]
        };
        let mut groups = BTreeMap::<String, BTreeMap<String, Vec<usize>>>::new();
        for header in headers {
            if header.dtype != gen_core::weightsmeta::Dtype::F32 {
                return Err(Error::Unsupported(format!(
                    "MOSS adapter tensor {} must be FP32",
                    header.name
                )));
            }
            let (path, factor) = gen_core::weightsmeta::split_factor_key(&header.name, suffixes)
                .ok_or_else(|| {
                    Error::Unsupported(format!("unsupported MOSS adapter tensor {}", header.name))
                })?;
            if !allowed.iter().any(|p| p == path) {
                return Err(Error::Unsupported(format!(
                    "MOSS adapter target {path} is not a transformer projection"
                )));
            }
            groups
                .entry(path.to_owned())
                .or_default()
                .insert(factor.to_owned(), header.shape);
        }
        for (path, group) in &groups {
            let parts: Vec<_> = path.split('.').collect();
            let facts = host
                .adaptable_facts(&parts)
                .ok_or_else(|| Error::Unsupported(format!("unknown MOSS projection {path}")))?;
            if lokr {
                contract::lokr_layout(
                    group,
                    facts.base_shape[0] as usize,
                    facts.base_shape[1] as usize,
                )?;
            } else {
                let mut rank_a = None;
                let mut rank_b = None;
                for (factor, shape) in group {
                    match factor.as_str() {
                        "lora_A.weight"
                            if shape.len() == 2
                                && shape[0] > 0
                                && shape[1] == facts.base_shape[1] as usize =>
                        {
                            rank_a = Some(shape[0])
                        }
                        "lora_B.weight"
                            if shape.len() == 2
                                && shape[1] > 0
                                && shape[0] == facts.base_shape[0] as usize =>
                        {
                            rank_b = Some(shape[1])
                        }
                        "alpha" if shape.iter().product::<usize>() == 1 => {}
                        _ => {
                            return Err(Error::Unsupported(format!(
                            "MOSS adapter tensor {path}.{factor} has incompatible shape {shape:?}"
                        )))
                        }
                    }
                }
                if matches!((rank_a,rank_b),(Some(a),Some(b)) if a!=b) {
                    return Err(Error::Unsupported("MOSS LoRA A/B ranks disagree".into()));
                }
            }
        }
        validated.push((groups, scale));
    }
    let mut reports = Vec::new();
    for (spec, (groups, scale)) in specs.iter().zip(validated) {
        let weights = mlx_gen::weights::Weights::from_file(&spec.path)?;
        for key in weights.keys() {
            finite(weights.require(key)?, &format!("source tensor {key}"))?;
        }
        let applied = match spec.kind {
            AdapterKind::Lora => {
                // Validate RAW portable factors too: alpha folding must not make this backend
                // accept a file that overflows Candle's otherwise identical raw execution.
                for path in groups.keys() {
                    validate_lora_factors(
                        weights.require(&format!("{path}.lora_A.weight"))?,
                        weights.require(&format!("{path}.lora_B.weight"))?,
                        if scale == 0. { 0. } else { 1. },
                    )?;
                }
                let report =
                    mlx_gen::adapters::loader::apply_lora_peft(host, &weights, scale, None)?;
                if report.applied != groups.len() || !report.unmatched_paths.is_empty() {
                    return Err(Error::Unsupported(
                        "MOSS adapter did not apply every declared target".into(),
                    ));
                }
                report.applied
            }
            AdapterKind::Lokr => {
                for (path, group) in &groups {
                    let parts: Vec<_> = path.split('.').collect();
                    let facts = host
                        .adaptable_facts(&parts)
                        .ok_or_else(|| Error::Msg("MOSS adapter path disappeared".into()))?;
                    let tensors = group
                        .keys()
                        .map(|factor| {
                            Ok((
                                factor.clone(),
                                weights.require(&format!("{path}.{factor}"))?.clone(),
                            ))
                        })
                        .collect::<Result<BTreeMap<_, _>>>()?;
                    host.push_lokr(path, Lokr::from_group(&tensors, &facts.base_shape, scale)?)?;
                }
                groups.len()
            }
        };
        reports.push(gen_core::AdapterApplyReport {
            adapter_path: spec.path.clone(),
            applied,
            skipped: Vec::new(),
        });
    }
    host.materialize_custom()?;
    Ok(reports)
}

/// Structured LoKr retains only the supplied FP32 factors; no full [out,in] delta is allocated.
/// Header alpha scalars are conservatively included, although the executable stack stores scalars.
pub(crate) fn estimated_bytes(
    specs: &[AdapterSpec],
    _config: &gen_core::moss_image::MossImageConfig,
) -> Result<u64> {
    specs.iter().try_fold(0u64, |bytes, spec| {
        gen_core::weightsmeta::safetensors_path_tensor_headers(&spec.path)?
            .iter()
            .try_fold(bytes, |total, header| {
                total
                    .checked_add(header.data_bytes)
                    .ok_or_else(|| Error::Unsupported("MOSS adapter byte overflow".into()))
            })
    })
}

/// Materialize under the model lifecycle's exact prepared tokens, never repin changed files.
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
            "MOSS adapter source-token set does not match requested stack".into(),
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

/// The producer validates the installed effective tensors before publishing an adapter artifact,
/// including a final optimizer update that has not yet been used by another forward pass.
pub(crate) fn save_validated(
    host: &mut Transformer,
    adapter: &mlx_gen::train::lora::TrainAdapter,
    params: &mlx_gen::train::lora::LoraParams,
    cfg: &mlx_gen::TrainingConfig,
    extra: &[(&str, &str)],
    path: &std::path::Path,
) -> Result<()> {
    // Validate raw portable PEFT tensors before the shared installer folds alpha into B.
    // Reload and producer must agree even when folding would hide raw FP32 overflow.
    if let mlx_gen::train::lora::TrainAdapter::Lora { targets } = adapter {
        for target in targets {
            let a_key = format!("{}.lora_a", target.path);
            let b_key = format!("{}.lora_b", target.path);
            let a = params
                .get(a_key.as_str())
                .ok_or_else(|| Error::MissingTensor(a_key.clone()))?;
            let b = params
                .get(b_key.as_str())
                .ok_or_else(|| Error::MissingTensor(b_key.clone()))?;
            validate_lora_factors(a, b, cfg.alpha / cfg.rank as f32)?;
        }
    }
    host.install_training_adapter(adapter, params, cfg.alpha, cfg.rank as f32)?;
    host.materialize_custom()?;
    adapter.save_with_meta(
        params,
        cfg.alpha,
        cfg.rank as f32,
        cfg.decompose_factor,
        "",
        extra,
        path,
    )
}

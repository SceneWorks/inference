//! Portable exact-parent adapters. Raw FP32 factors serve training, recomputation and generation;
//! no projection-sized Kronecker delta is ever constructed.
use crate::{loader::tensor_f32, transformer::Transformer};
use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::gen_core::{self, moss_image::MossImageVariant};
use candle_gen::{CandleError, Result};
use gen_core_moss_image::moss_image_training as contract;
use std::collections::BTreeMap;

fn invalid(message: impl Into<String>) -> CandleError {
    CandleError::Unsupported(format!("MOSS adapter: {}", message.into()))
}
pub(crate) fn finite(t: &Tensor, label: &str) -> Result<()> {
    if tensor_f32(t)?.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("nonfinite {label}")));
    }
    Ok(())
}
// Bound each actual FP32 contraction for inputs with magnitude <= 1. A finite scaled
// coefficient is insufficient: the implementation applies scale only AFTER the raw factors.
// Absolute row sums also cover cancellation and each decomposed leg, without a dense delta.
fn contraction_bound(weight: &Tensor, input: f64) -> Result<f64> {
    let (_, columns) = weight.dims2()?;
    let norm = tensor_f32(weight)?
        .chunks_exact(columns)
        .map(|row| row.iter().map(|v| f64::from(v.abs())).sum::<f64>())
        .fold(0., f64::max);
    let bound = norm * input;
    if !bound.is_finite() || bound > f32::MAX as f64 {
        return Err(invalid(
            "unscaled factor contraction overflows FP32; reduce factor magnitude",
        ));
    }
    Ok(bound)
}
#[derive(Clone)]
pub(crate) enum Factor {
    Full(Tensor),
    LowRank { a: Tensor, b: Tensor },
}
impl Factor {
    pub(crate) fn from_group(group: &BTreeMap<String, Tensor>, name: &str) -> Result<Self> {
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
            _ => Err(invalid(format!(
                "incomplete or competing {name} representations"
            ))),
        }
    }
    fn detached(&self) -> Self {
        match self {
            Self::Full(w) => Self::Full(w.detach()),
            Self::LowRank { a, b } => Self::LowRank {
                a: a.detach(),
                b: b.detach(),
            },
        }
    }
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Full(w) => (w.dims()[0], w.dims()[1]),
            Self::LowRank { a, b } => (a.dims()[0], b.dims()[1]),
        }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Full(w) => Ok(Linear::new(w.clone(), None).forward(&x.contiguous()?)?),
            Self::LowRank { a, b } => Ok(Linear::new(a.clone(), None)
                .forward(&Linear::new(b.clone(), None).forward(&x.contiguous()?)?)?),
        }
    }
    fn forward_bound(&self, input: f64) -> Result<f64> {
        match self {
            Self::Full(w) => contraction_bound(w, input),
            Self::LowRank { a, b } => contraction_bound(a, contraction_bound(b, input)?),
        }
    }
    fn tensors(&self) -> Vec<&Tensor> {
        match self {
            Self::Full(w) => vec![w],
            Self::LowRank { a, b } => vec![a, b],
        }
    }
    fn effective_max(&self) -> Result<f64> {
        let matrix = match self {
            Self::Full(w) => w.clone(),
            Self::LowRank { a, b } => a.matmul(b)?,
        };
        finite(&matrix, "derived LoKr factor")?;
        Ok(tensor_f32(&matrix)?
            .iter()
            .map(|v| f64::from(v.abs()))
            .fold(0., f64::max))
    }
}
#[derive(Clone)]
pub(crate) enum Residual {
    Lora {
        down: Tensor,
        up: Tensor,
        scale: f64,
    },
    Lokr {
        first: Factor,
        second: Factor,
        scale: f64,
    },
}
impl Residual {
    pub(crate) fn detached(&self) -> Self {
        match self {
            Self::Lora { down, up, scale } => Self::Lora {
                down: down.detach(),
                up: up.detach(),
                scale: *scale,
            },
            Self::Lokr {
                first,
                second,
                scale,
            } => Self::Lokr {
                first: first.detached(),
                second: second.detached(),
                scale: *scale,
            },
        }
    }
    pub(crate) fn from_group(
        group: &BTreeMap<String, Tensor>,
        output: usize,
        input: usize,
        lokr: bool,
        scale: f64,
    ) -> Result<Self> {
        if !scale.is_finite() || scale.abs() > f32::MAX as f64 {
            return Err(invalid("effective scale overflows FP32"));
        }
        if group.values().any(|t| t.dtype() != DType::F32) {
            return Err(invalid("factors must be FP32"));
        }
        let residual = if lokr {
            let shapes = group
                .iter()
                .map(|(k, v)| (k.clone(), v.dims().to_vec()))
                .collect();
            contract::lokr_layout(&shapes, output, input)?;
            Self::Lokr {
                first: Factor::from_group(group, "lokr_w1")?,
                second: Factor::from_group(group, "lokr_w2")?,
                scale,
            }
        } else {
            if group
                .keys()
                .any(|k| !["lora_A.weight", "lora_B.weight", "alpha"].contains(&k.as_str()))
            {
                return Err(invalid("unknown LoRA factor"));
            }
            let down = group
                .get("lora_A.weight")
                .ok_or_else(|| invalid("missing LoRA A"))?;
            let up = group
                .get("lora_B.weight")
                .ok_or_else(|| invalid("missing LoRA B"))?;
            let rank = down.dims().first().copied().unwrap_or(0);
            if rank == 0 || down.dims() != [rank, input] || up.dims() != [output, rank] {
                return Err(invalid("LoRA A/B shape or rank mismatch"));
            }
            Self::Lora {
                down: down.clone(),
                up: up.clone(),
                scale,
            }
        };
        residual.validate()?;
        Ok(residual)
    }
    pub(crate) fn tensors(&self) -> Vec<&Tensor> {
        match self {
            Self::Lora { down, up, .. } => vec![down, up],
            Self::Lokr { first, second, .. } => {
                let mut v = first.tensors();
                v.extend(second.tensors());
                v
            }
        }
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.tensors()
            .iter()
            .map(|t| (t.elem_count() * 4) as u64)
            .sum()
    }
    pub(crate) fn validate(&self) -> Result<()> {
        for tensor in self.tensors() {
            finite(tensor, "source factor")?;
        }
        let raw_bound = if self.scale() == 0. {
            0. // Forward skips a disabled residual before any contraction.
        } else {
            match self {
                Self::Lora { down, up, .. } => contraction_bound(up, contraction_bound(down, 1.)?)?,
                Self::Lokr { first, second, .. } => {
                    let (a, c) = first.shape();
                    let (b, d) = second.shape();
                    if c * b <= d * a {
                        first.forward_bound(second.forward_bound(1.)?)?
                    } else {
                        second.forward_bound(first.forward_bound(1.)?)?
                    }
                }
            }
        };
        if raw_bound * self.scale().abs() > f32::MAX as f64 {
            return Err(invalid(
                "scaled factor contraction overflows FP32; reduce factor magnitude or strength",
            ));
        }
        let envelope = match self {
            Self::Lora { down, up, scale } => {
                let max = |t: &Tensor| -> Result<f64> {
                    Ok(tensor_f32(t)?
                        .iter()
                        .map(|v| f64::from(v.abs()))
                        .fold(0., f64::max))
                };
                max(down)? * max(up)? * down.dim(0)? as f64 * scale.abs()
            }
            Self::Lokr {
                first,
                second,
                scale,
            } => first.effective_max()? * second.effective_max()? * scale.abs(),
        };
        if !envelope.is_finite() || envelope > f32::MAX as f64 {
            return Err(invalid("derived effective adapter overflows FP32"));
        }
        Ok(())
    }
    pub(crate) fn scale(&self) -> f64 {
        match self {
            Self::Lora { scale, .. } | Self::Lokr { scale, .. } => *scale,
        }
    }
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Option<Tensor>> {
        let scale = self.scale();
        if scale == 0. {
            return Ok(None);
        }
        let x = x.to_dtype(DType::F32)?;
        let value = match self {
            Self::Lora { down, up, .. } => {
                let rank = Linear::new(down.clone(), None).forward(&x.contiguous()?)?;
                if down.dim(0)? < up.dim(0)? && scale != 1.0 {
                    return Ok(Some(
                        Linear::new(up.clone(), None).forward(&(rank * scale)?)?,
                    ));
                }
                Linear::new(up.clone(), None).forward(&rank)?
            }
            Self::Lokr { first, second, .. } => {
                let (a, c) = first.shape();
                let (b, d) = second.shape();
                let mut shape = x.dims().to_vec();
                let input = shape.pop().ok_or_else(|| invalid("scalar input"))?;
                if input != c * d {
                    return Err(invalid("LoKr input dimension mismatch"));
                }
                let rows = x.elem_count() / input;
                let x = x.reshape((rows, c, d))?;
                let y = if c * b <= d * a {
                    first
                        .forward(&second.forward(&x)?.transpose(1, 2)?)?
                        .transpose(1, 2)?
                } else {
                    second.forward(&first.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?)?
                };
                shape.push(a * b);
                y.contiguous()?.reshape(shape)?
            }
        };
        Ok(Some(if scale == 1.0 {
            value
        } else {
            (value * scale)?
        }))
    }
}

/// Validate the entire supplied stack before installing anything. Metadata and header validation
/// precede tensor allocation; source tokens are checked before and after consuming their bytes.
pub(crate) fn apply(
    host: &mut Transformer,
    variant: MossImageVariant,
    specs: &[gen_core::AdapterSpec],
    pins: &[gen_core::PinnedWeightsFile],
) -> Result<Vec<gen_core::AdapterApplyReport>> {
    if pins.len() != specs.len()
        || pins
            .iter()
            .zip(specs)
            .any(|(p, s)| p.loader_path() != s.path)
    {
        return Err(invalid("source-token set mismatch"));
    }
    let shapes = host.projection_shapes();
    let mut pending = Vec::new();
    for (spec, pin) in specs.iter().zip(pins) {
        pin.verify_unchanged()?;
        if !spec.scale.is_finite() || spec.pass_scales.is_some() || spec.moe_expert.is_some() {
            return Err(invalid(
                "requires finite uniform strength without pass/expert overrides",
            ));
        }
        let meta = gen_core::weightsmeta::safetensors_file_metadata(&spec.path)?;
        contract::validate_metadata(variant, &meta)?;
        let lokr = match (&spec.kind, meta.get("networkType").map(String::as_str)) {
            (gen_core::AdapterKind::Lora, Some("lora")) => false,
            (gen_core::AdapterKind::Lokr, Some("lokr")) => true,
            _ => return Err(invalid("kind disagrees with networkType")),
        };
        let (metadata_rank, metadata_alpha) = contract::lokr_scaling(&meta)?;
        let suffixes: &[&str] = if lokr {
            &gen_core::weightsmeta::LOKR_SUFFIXES
        } else {
            &[".lora_A.weight", ".lora_B.weight", ".alpha"]
        };
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&spec.path)?;
        if headers.is_empty() {
            return Err(invalid("empty adapter"));
        }
        let mut groups = BTreeMap::<String, BTreeMap<String, Vec<usize>>>::new();
        for header in headers {
            if header.dtype != gen_core::weightsmeta::Dtype::F32 {
                return Err(invalid(format!("{} must be FP32", header.name)));
            }
            let (path, factor) = gen_core::weightsmeta::split_factor_key(&header.name, suffixes)
                .ok_or_else(|| invalid(format!("unknown tensor {}", header.name)))?;
            if !shapes.contains_key(path) {
                return Err(invalid(format!("unknown transformer projection {path}")));
            }
            groups
                .entry(path.into())
                .or_default()
                .insert(factor.into(), header.shape);
        }
        for (path, group) in &groups {
            let &(output, input) = &shapes[path];
            if lokr {
                contract::lokr_layout(group, output, input)?;
            } else {
                let a = group
                    .get("lora_A.weight")
                    .ok_or_else(|| invalid("orphan LoRA B/alpha"))?;
                let b = group
                    .get("lora_B.weight")
                    .ok_or_else(|| invalid("orphan LoRA A/alpha"))?;
                if a.len() != 2
                    || a[0] == 0
                    || a[1] != input
                    || b != &vec![output, a[0]]
                    || group
                        .get("alpha")
                        .is_some_and(|s| s.iter().product::<usize>() != 1)
                {
                    return Err(invalid("LoRA header shape mismatch"));
                }
            }
        }
        let weights =
            candle_gen::weights::Weights::from_file(&spec.path, host.device(), DType::F32)?;
        let mut projections = Vec::new();
        for (path, group) in groups {
            let tensors = group
                .keys()
                .map(|key| Ok((key.clone(), weights.require(&format!("{path}.{key}"))?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            let (output, input) = shapes[&path];
            let scale = if lokr {
                f64::from((metadata_alpha / metadata_rank) * spec.scale)
            } else {
                // PEFT per-target alpha is optional only when the file supplies valid global alpha.
                let rank = tensors["lora_A.weight"].dim(0)? as f32;
                let alpha = if let Some(t) = tensors.get("alpha") {
                    let v = tensor_f32(t)?;
                    if v.len() != 1 {
                        return Err(invalid("invalid alpha scalar"));
                    }
                    v[0]
                } else {
                    meta.get("alpha")
                        .ok_or_else(|| invalid("missing LoRA alpha"))?
                        .parse::<f32>()
                        .map_err(|_| invalid("malformed LoRA alpha"))?
                };
                if !alpha.is_finite() || rank != metadata_rank {
                    return Err(invalid(
                        "LoRA alpha must be finite and tensor rank must match metadata",
                    ));
                }
                f64::from((alpha / rank) * spec.scale)
            };
            projections.push((
                path,
                Residual::from_group(&tensors, output, input, lokr, scale)?,
            ));
        }
        pin.verify_unchanged()?;
        pending.push((spec.path.clone(), projections));
    }
    let mut reports = Vec::new();
    for (path, projections) in pending {
        let applied = projections.len();
        for (target, residual) in projections {
            host.push_residual(&target, residual)?;
        }
        reports.push(gen_core::AdapterApplyReport {
            adapter_path: path,
            applied,
            skipped: vec![],
        });
    }
    Ok(reports)
}

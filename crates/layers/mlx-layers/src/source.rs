//! Retained source declarations and measured representations for a single MLX weight map.
use crate::{Error, Result};
use gen_core::checkpoint_codec::*;
use gen_core::checkpoint_facts::{ExecutionRepresentation, NativeExecutionCapability};
use mlx_rs::{Array, Dtype};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
};

pub struct MlxCodecResidency;
impl CodecResidencyPolicy for MlxCodecResidency {
    fn residency(
        &self,
        _: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        shape: &[usize],
    ) -> ResidencyMode {
        if shape.len() == 2
            && matches!(
                spec,
                TensorCodecSpec::ConvRotInt8 { .. }
                    | TensorCodecSpec::ScalarFp8 {
                        full_precision_matrix_mult: false,
                        ..
                    }
                    | TensorCodecSpec::Mxfp8 {
                        full_precision_matrix_mult: false,
                        ..
                    }
            )
        {
            ResidencyMode::Packed
        } else {
            ResidencyMode::Dense
        }
    }
}
pub fn capability() -> NativeExecutionCapability {
    NativeExecutionCapability::new([
        gen_core::convrot_int8::CODEC_ID,
        FP8_E4M3_SCALAR_CODEC.codec_id,
        FP8_E5M2_SCALAR_CODEC.codec_id,
        MXFP8_CODEC.codec_id,
    ])
}

pub fn registry() -> &'static CheckpointCodecRegistry {
    static REGISTRY: OnceLock<CheckpointCodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied())
            .expect("valid shared codec registry")
    })
}

#[derive(Debug)]
struct Row {
    accounting: Arc<gen_core::CheckpointReadAccounting>,
    key: String,
    logical_shape: Vec<usize>,
    // The constructor verifies the actual handles it keeps before enabling packed accounting.
    packed_bytes: AtomicU64,
    packed_observed: AtomicBool,
    has_weight_scale: bool,
    has_input_scale: bool,
}
#[derive(Clone, Debug)]
pub struct Observation(Arc<Row>);
impl Observation {
    pub fn bind(&self, codes: &Array, scale: &Array, input: Option<&Array>) -> Result<()> {
        let row = &self.0;
        let bytes = codes.nbytes()
            + if row.has_weight_scale {
                scale.nbytes()
            } else {
                0
            }
            + if row.has_input_scale {
                input.ok_or("missing measured activation scale")?.nbytes()
            } else {
                0
            };
        row.packed_bytes.store(bytes as u64, Ordering::Relaxed);
        Ok(())
    }
    pub(crate) fn pending(&self) -> bool {
        !self.0.packed_observed.load(Ordering::Relaxed)
    }
    pub fn record_packed(&self) -> Result<()> {
        let row = &self.0;
        if !row.packed_observed.load(Ordering::Relaxed) {
            let bytes = row.packed_bytes.load(Ordering::Relaxed);
            if bytes != 0 {
                row.accounting
                    .record(&row.key, ExecutionRepresentation::NativePacked, bytes)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                row.packed_observed.store(true, Ordering::Relaxed);
            }
        }
        Ok(())
    }
    pub fn materialized(&self, array: &Array) -> Result<()> {
        if self.0.packed_bytes.load(Ordering::Relaxed) != 0
            && matches!(array.dtype(), Dtype::Uint8 | Dtype::Int8)
        {
            return self.record_packed();
        }
        if matches!(
            array.dtype(),
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
        ) {
            let row = &self.0;
            let shape: Vec<_> = array.shape().iter().map(|&d| d as usize).collect();
            let width = array.item_size();
            // Adapter normalization may reshape a dense buffer without changing its elements.
            let shape = if row.logical_shape.iter().product::<usize>() == array.size() {
                &row.logical_shape
            } else {
                &shape
            };
            row.accounting
                .declare_dense_layout(&row.key, &shape, width)
                .map_err(|e| Error::Msg(e.to_string()))?;
            row.accounting
                .record(
                    &row.key,
                    ExecutionRepresentation::DenseFallback,
                    array.nbytes() as u64,
                )
                .map_err(|e| Error::Msg(e.to_string()))?;
        }
        Ok(())
    }
}

/// Raw and logical readers share observations. Only numeric accounting is retained in a model
/// scope snapshot. Arrays and their lifetime remain under the provider's own pin and eviction policy.
#[derive(Default, Clone, Debug)]
pub struct SourceReads {
    rows: HashMap<String, Observation>,
    accounting: Option<Arc<gen_core::CheckpointReadAccounting>>,
}
impl SourceReads {
    pub fn new(plan: Arc<LogicalWeightPlan>, paths: &[PathBuf]) -> Self {
        let accounting = gen_core::CheckpointReadAccounting::new(plan.clone(), capability(), paths);
        let rows = plan
            .tensors
            .iter()
            .map(|t| {
                let (has_weight_scale, has_input_scale) = match &t.codec {
                    TensorCodecSpec::ScalarFp8 {
                        scale, input_scale, ..
                    } => (
                        matches!(scale, ScalarScaleSource::Companion { .. }),
                        input_scale.is_some(),
                    ),
                    TensorCodecSpec::ConvRotInt8 { .. } | TensorCodecSpec::Mxfp8 { .. } => {
                        (true, false)
                    }
                    _ => (false, false),
                };
                (
                    t.logical_key.clone(),
                    Observation(Arc::new(Row {
                        accounting: accounting.clone(),
                        key: t.logical_key.clone(),
                        logical_shape: t.shape.clone(),
                        packed_bytes: AtomicU64::new(0),
                        packed_observed: AtomicBool::new(false),
                        has_weight_scale,
                        has_input_scale,
                    })),
                )
            })
            .collect();
        Self {
            rows,
            accounting: Some(accounting),
        }
    }
    pub fn get(&self, key: &str) -> Option<Observation> {
        self.rows.get(key).cloned()
    }
    pub fn alias(&mut self, from: &str, to: &str) {
        if let Some(row) = self.get(from) {
            self.rows.insert(to.to_owned(), row);
        }
    }
    pub fn materialized(&self, key: &str, array: &Array) -> Result<()> {
        if let Some(obs) = self.rows.get(key) {
            obs.materialized(array)?;
        }
        Ok(())
    }
    pub fn receipt(&self) -> Option<LogicalWeightReceipt> {
        self.accounting.as_ref().map(|a| a.receipt())
    }
}

pub fn prepare_sources(
    tensors: &mut HashMap<String, Array>,
    paths: &[PathBuf],
    tables: &[String],
) -> Result<SourceReads> {
    use gen_core::comfy_quant::ComfyQuantFormat;
    if tables.is_empty()
        && !tensors
            .values()
            .any(|v| v.dtype() == Dtype::Uint8 || v.dtype() == Dtype::Int8)
    {
        return Ok(SourceReads::default());
    }
    let Some(plan) = plan_projection_sources(
        paths,
        |k| tensors.contains_key(k),
        tables,
        registry(),
        &MlxCodecResidency,
    )
    .map_err(|e| Error::Msg(e.to_string()))?
    else {
        return Ok(SourceReads::default());
    };
    for t in &plan.tensors {
        let key = &t.logical_key;
        let Some(array) = tensors.get(key).cloned() else {
            return Err(format!("{key}: selected source disappeared").into());
        };
        let base = key.strip_suffix(".weight");
        let (format, scale, sx) = match &t.codec {
            TensorCodecSpec::ConvRotInt8 { geometry, .. } => {
                let base = base.ok_or("missing ConvRot projection base")?;
                tensors
                    .entry(format!("{base}.comfy_quant"))
                    .or_insert_with(|| {
                        crate::convrot_int8::descriptor_array(geometry.rotation_group)
                    });
                continue;
            }
            TensorCodecSpec::Mxfp8 { scale, .. } => (ComfyQuantFormat::Mxfp8, Some(scale), None),
            TensorCodecSpec::ScalarFp8 {
                scale, input_scale, ..
            } => (
                if t.encoding == WeightEncoding::Fp8E5M2 {
                    ComfyQuantFormat::Float8E5M2
                } else {
                    ComfyQuantFormat::Float8E4M3Fn
                },
                match scale {
                    ScalarScaleSource::Unit => None,
                    ScalarScaleSource::Companion { physical_key } => Some(physical_key),
                },
                input_scale.as_ref(),
            ),
            _ => continue,
        };
        let get = |k: &String| {
            tensors
                .get(k)
                .cloned()
                .ok_or_else(|| Error::Msg(format!("{key}: missing companion {k}")))
        };
        let sw = scale.map(get).transpose()?;
        let sx = sx.map(get).transpose()?;
        if t.residency.mode == ResidencyMode::Dense || base.is_none() {
            let value = if array.ndim() == 2 {
                crate::fp8::Fp8Linear::new(format, array, sw, sx, &t.shape, None)?.dense_weight()?
            } else {
                let mut value = crate::fp8::decoded_codes(&array, format)?;
                if let Some(scale) = sw {
                    value = mlx_rs::ops::multiply(&value, &scale)?;
                }
                value.as_dtype(Dtype::Bfloat16)?
            };
            tensors.insert(key.clone(), value);
            if let Some(base) = base {
                for suffix in [".comfy_quant", ".weight_scale", ".input_scale"] {
                    tensors.remove(&format!("{base}{suffix}"));
                }
            }
        } else if let Some(base) = base {
            let shape = (format == ComfyQuantFormat::Mxfp8).then_some(t.shape.as_slice());
            tensors.insert(
                format!("{base}.comfy_quant"),
                crate::fp8::descriptor_array(format, false, shape),
            );
            tensors.insert(
                format!("{base}.weight_scale"),
                sw.unwrap_or_else(crate::fp8::unit_scale),
            );
            if let Some(sx) = sx {
                tensors.insert(format!("{base}.input_scale"), sx);
            }
        }
    }
    Ok(SourceReads::new(Arc::new(plan), paths))
}

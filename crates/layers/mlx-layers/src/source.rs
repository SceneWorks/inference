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
        Arc, Mutex, OnceLock, Weak,
    },
};

pub fn metal_available() -> bool {
    static METAL: OnceLock<bool> = OnceLock::new();
    *METAL.get_or_init(|| {
        let mut available = false;
        // SAFETY: MLX writes a Boolean availability probe; no stream or array is allocated.
        unsafe { mlx_sys::mlx_metal_is_available(&mut available) == 0 && available }
    })
}

pub struct MlxCodecResidency;
impl CodecResidencyPolicy for MlxCodecResidency {
    fn residency_for_key(
        &self,
        key: &str,
        codec: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        shape: &[usize],
    ) -> ResidencyMode {
        if key.ends_with(".weight") || key.ends_with("_weight") {
            self.residency(codec, spec, shape)
        } else {
            ResidencyMode::Dense
        }
    }
    fn residency(
        &self,
        _: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        shape: &[usize],
    ) -> ResidencyMode {
        if metal_available()
            && shape.len() == 2
            && matches!(
                spec,
                TensorCodecSpec::ConvRotInt8 { .. }
                    | TensorCodecSpec::RowwiseFp8 { .. }
                    | TensorCodecSpec::ScalarFp8 { .. }
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
    if !metal_available() {
        return NativeExecutionCapability::new([]);
    }
    NativeExecutionCapability::new([
        gen_core::convrot_int8::CODEC_ID,
        FP8_E4M3_SCALAR_CODEC.codec_id,
        FP8_E5M2_SCALAR_CODEC.codec_id,
        MXFP8_CODEC.codec_id,
        FP8_E4M3_ROWWISE_CODEC.codec_id,
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
    mx_layout: Option<([usize; 2], bool)>,
    // The constructor verifies the actual handles it keeps before enabling packed accounting.
    packed_bytes: AtomicU64,
    packed_observed: AtomicBool,
    has_weight_scale: bool,
    has_input_scale: bool,
    slice: Option<(usize, usize, usize)>,
    children: Mutex<Vec<Weak<Row>>>,
}
#[derive(Clone, Debug)]
pub struct Observation(Arc<Row>);
impl Observation {
    pub fn mxfp8_geometry(&self) -> Option<([usize; 2], Option<&[usize]>)> {
        let (stored, declared) = self.0.mx_layout?;
        Some((stored, declared.then_some(&self.0.logical_shape)))
    }

    /// Each expert selected from a rank-three source has its own materialization boundary.
    pub fn expert(&self, index: usize, rows: usize, cols: usize) -> Result<Self> {
        if self.0.logical_shape.len() != 3
            || self.0.logical_shape[1..] != [rows, cols]
            || index >= self.0.logical_shape[0]
        {
            return Err("expert source geometry changed since planning".into());
        }
        let row = Arc::new(Row {
            accounting: self.0.accounting.clone(),
            key: self.0.key.clone(),
            logical_shape: vec![rows, cols],
            mx_layout: None,
            packed_bytes: AtomicU64::new(0),
            packed_observed: AtomicBool::new(false),
            has_weight_scale: self.0.has_weight_scale,
            has_input_scale: self.0.has_input_scale,
            slice: Some((index * rows, rows, cols)),
            children: Default::default(),
        });
        self.0
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(&row));
        Ok(Self(row))
    }

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
                match row.slice {
                    Some((start, rows, cols)) => row
                        .accounting
                        .record_shared_matrix_slice(&row.key, start, rows, cols, bytes),
                    None => row.accounting.record(
                        &row.key,
                        ExecutionRepresentation::NativePacked,
                        bytes,
                    ),
                }
                .map_err(|e| Error::Msg(e.to_string()))?;
                row.packed_observed.store(true, Ordering::Relaxed);
            }
        }
        Ok(())
    }
    pub fn materialized(&self, array: &Array) -> Result<()> {
        if array.ndim() == 3 && array.dtype() == Dtype::Uint8 {
            for child in self
                .0
                .children
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .filter_map(Weak::upgrade)
            {
                Self(child).record_packed()?;
            }
        }
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
                .declare_dense_layout(&row.key, shape, width)
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
    rotations: Arc<Mutex<HashMap<usize, Array>>>,
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
                    TensorCodecSpec::ConvRotInt8 { .. }
                    | TensorCodecSpec::Mxfp8 { .. }
                    | TensorCodecSpec::RowwiseFp8 { .. } => (true, false),
                    _ => (false, false),
                };
                (
                    t.logical_key.clone(),
                    Observation(Arc::new(Row {
                        accounting: accounting.clone(),
                        key: t.logical_key.clone(),
                        logical_shape: t.shape.clone(),
                        mx_layout: match &t.codec {
                            TensorCodecSpec::Mxfp8 {
                                stored_shape,
                                logical_shape_declared,
                                ..
                            } => Some((*stored_shape, *logical_shape_declared)),
                            _ => None,
                        },
                        packed_bytes: AtomicU64::new(0),
                        packed_observed: AtomicBool::new(false),
                        has_weight_scale,
                        has_input_scale,
                        slice: None,
                        children: Default::default(),
                    })),
                )
            })
            .collect();
        Self {
            rows,
            accounting: Some(accounting),
            rotations: Default::default(),
        }
    }
    pub fn rotation(&self, group: usize, build: impl FnOnce() -> Array) -> Array {
        self.rotations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(group)
            .or_insert_with(build)
            .clone()
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
    pub fn native_projection_report(&self) -> Vec<core_llm::ProjectionReport> {
        self.accounting
            .as_ref()
            .map(|a| a.native_projection_report())
            .unwrap_or_default()
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
    prepare_sources_with_experts(tensors, paths, tables, |_| false)
}

pub fn prepare_sources_with_experts(
    tensors: &mut HashMap<String, Array>,
    paths: &[PathBuf],
    tables: &[String],
    expert_key: impl Fn(&str) -> bool,
) -> Result<SourceReads> {
    prepare_sources_with_layout(tensors, paths, tables, expert_key, |_, _| Dtype::Bfloat16)
}

/// Compile the same file-backed operator layout for load admission and materialization.
pub fn projection_plan(
    paths: &[PathBuf],
    selected: impl Fn(&str) -> bool,
    tables: &[String],
    expert_key: impl Fn(&str) -> bool,
    coefficient_key: impl Fn(&str) -> bool,
) -> Result<Option<LogicalWeightPlan>> {
    struct Layout<E, C> {
        expert: E,
        coefficient: C,
    }
    impl<E: Fn(&str) -> bool, C: Fn(&str) -> bool> CodecResidencyPolicy for Layout<E, C> {
        fn residency(
            &self,
            c: &CheckpointCodecRegistration,
            spec: &TensorCodecSpec,
            shape: &[usize],
        ) -> ResidencyMode {
            MlxCodecResidency.residency(c, spec, shape)
        }
        fn residency_for_key(
            &self,
            key: &str,
            c: &CheckpointCodecRegistration,
            spec: &TensorCodecSpec,
            shape: &[usize],
        ) -> ResidencyMode {
            if (self.coefficient)(key) {
                return ResidencyMode::Dense;
            }
            if shape.len() == 3
                && (self.expert)(key)
                && metal_available()
                && matches!(spec, TensorCodecSpec::ScalarFp8 { .. })
            {
                ResidencyMode::Packed
            } else if (self.expert)(key) {
                self.residency(c, spec, shape)
            } else {
                MlxCodecResidency.residency_for_key(key, c, spec, shape)
            }
        }
    }
    plan_projection_sources(
        paths,
        selected,
        tables,
        registry(),
        &Layout {
            expert: expert_key,
            coefficient: coefficient_key,
        },
    )
    .map_err(|e| Error::Msg(e.to_string()))
}

/// Conservative additional load buffers of the selected encoded source operands. Physical
/// codes, scales and biases are priced by the checkpoint plan. A device's allocator supplies
/// the footprint of one allocation, including rounding. Counts are derived from the shared
/// constructors and coefficient decode, without evaluating a tensor.
pub fn load_auxiliary_bytes(
    plan: &LogicalWeightPlan,
    selected: impl Fn(&str) -> bool,
    allocation: impl Fn(u64) -> Option<u64>,
) -> Option<u64> {
    auxiliary_bytes(plan, selected, allocation, true)
}

/// Additional operands kept by a completed MLX source load, beyond the physical checkpoint
/// rows. A source owns one rotation per geometry and uses the thread's shared FP8 unit scale.
pub fn retained_auxiliary_bytes(
    plan: &LogicalWeightPlan,
    selected: impl Fn(&str) -> bool,
    allocation: impl Fn(u64) -> Option<u64>,
) -> Option<u64> {
    auxiliary_bytes(plan, selected, allocation, false)
}

fn auxiliary_bytes(
    plan: &LogicalWeightPlan,
    selected: impl Fn(&str) -> bool,
    allocation: impl Fn(u64) -> Option<u64>,
    during_load: bool,
) -> Option<u64> {
    let mut extra = 0u64;
    let mut needs_unit = false;
    let mut rotations = std::collections::BTreeSet::new();
    let scales: HashMap<_, _> = plan
        .companions
        .iter()
        .map(|c| (c.physical_key.as_str(), c.source_bytes))
        .collect();
    let mut add = |bytes: u64, allocations: u64| -> Option<()> {
        extra = extra.checked_add(allocation(bytes)?.checked_mul(allocations)?)?;
        Some(())
    };
    for row in plan.tensors.iter().filter(|r| selected(&r.physical_key)) {
        // Only these codecs construct auxiliary buffers in bind_layout_with_dtypes.
        if !matches!(
            row.codec,
            TensorCodecSpec::ConvRotInt8 { .. }
                | TensorCodecSpec::ScalarFp8 { .. }
                | TensorCodecSpec::RowwiseFp8 { .. }
                | TensorCodecSpec::Mxfp8 { .. }
        ) {
            continue;
        }
        if row.residency.mode == ResidencyMode::Packed {
            if during_load {
                add(gen_core::convrot_int8::MAX_DESCRIPTOR_BYTES as u64, 1)?;
            }
            match &row.codec {
                TensorCodecSpec::ConvRotInt8 { geometry, .. }
                    if rotations.insert(geometry.rotation_group) =>
                {
                    // from_slice copies the host rotation into its one shared MLX array.
                    let group = geometry.rotation_group as u64;
                    add(
                        group.checked_mul(group).and_then(|n| n.checked_mul(4))?,
                        if during_load { 2 } else { 1 },
                    )?;
                }
                TensorCodecSpec::ScalarFp8 {
                    scale, input_scale, ..
                } => {
                    let weight_scale = match scale {
                        ScalarScaleSource::Unit => None,
                        ScalarScaleSource::Companion { physical_key } => Some(physical_key),
                    };
                    needs_unit |= weight_scale.is_none() || input_scale.is_none();
                    for key in [weight_scale, input_scale.as_ref()].into_iter().flatten() {
                        let stored = *scales.get(key.as_str())?;
                        if stored < 4 {
                            // Legacy floating scalar scales are promoted to F32. Their
                            // source bytes are already in the physical plan.
                            add(if during_load { 4 } else { 4 - stored }, 1)?;
                        }
                    }
                }
                TensorCodecSpec::RowwiseFp8 { .. } | TensorCodecSpec::Mxfp8 { .. } => {
                    needs_unit = true
                }
                _ => {}
            }
        } else if during_load {
            // E4M3 decode can create three byte masks, two decoded F32 buffers and one
            // scaled F32 result before the BF16 coefficient buffer. E5M2's two U16
            // shifts also fit this bound. Price the padded source extent for MXFP8.
            let elements = row.source_bytes;
            // Scalar constants for the decode predicates, NaN replacement and scales.
            add(4, 8)?;
            add(elements, 3)?;
            add(elements.checked_mul(4)?, 3)?;
            add(elements.checked_mul(2)?, 1)?;
            if let TensorCodecSpec::Mxfp8 { scale, .. } = &row.codec {
                // Unswizzle, exponent conversion, subtraction, power, NaN selection.
                add(scales.get(scale.as_str())?.checked_mul(4)?, 6)?;
            }
        }
    }
    if needs_unit {
        add(4, 1)?;
    }
    Some(extra)
}

/// Physical operands retained by an MLX owner. Native operators bind a floating bias at its
/// original width. An encoded bias follows its independently declared coefficient precision.
pub fn projected_bytes_from_plan(
    plan: &LogicalWeightPlan,
    headers: &[gen_core::weightsmeta::SafetensorsTensorHeader],
    dense_width: impl Fn(&gen_core::weightsmeta::SafetensorsTensorHeader) -> u64,
) -> gen_core::Result<std::collections::BTreeMap<String, u64>> {
    use gen_core::weightsmeta::Dtype as Stored;
    let mut bytes =
        gen_core::checkpoint_codec::projected_bytes_from_plan(plan, headers, &dense_width)?;
    let bias_keys: std::collections::HashSet<_> = plan
        .tensors
        .iter()
        .filter(|r| r.residency.mode == ResidencyMode::Packed && bytes[&r.physical_key] != 0)
        .map(|row| {
            let key = &row.physical_key;
            if let Some(base) = key.strip_suffix("_weight") {
                format!("{base}_bias")
            } else {
                format!("{}.bias", key.strip_suffix(".weight").unwrap_or(key))
            }
        })
        .collect();
    for header in headers.iter().filter(|h| bias_keys.contains(&h.name)) {
        let compute = dense_width(header);
        let width = match header.dtype {
            _ if compute == 0 => 0,
            Stored::F32 => 4,
            Stored::BF16 | Stored::F16 => 2,
            Stored::F8_E4M3 | Stored::F8_E5M2 => compute,
            _ => continue,
        };
        bytes.insert(header.name.clone(), header.materialized_bytes(width)?);
    }
    Ok(bytes)
}

/// Determine dense consumer precision from the original storage handles, before decoding either
/// operand of a mixed-precision module. Encoded matrices selected for native execution stay packed.
pub fn prepare_sources_with_layout(
    tensors: &mut HashMap<String, Array>,
    paths: &[PathBuf],
    tables: &[String],
    expert_key: impl Fn(&str) -> bool,
    dense_dtype: impl Fn(&str, &HashMap<String, Array>) -> Dtype,
) -> Result<SourceReads> {
    prepare_sources_for_consumers(tensors, paths, tables, expert_key, |_| false, dense_dtype)
}

/// Retain operator operands according to their declared use. A matrix-shaped coefficient table
/// (for example multidimensional LayerNorm weights) must be decoded before elementwise use.
pub fn prepare_sources_for_consumers(
    tensors: &mut HashMap<String, Array>,
    paths: &[PathBuf],
    tables: &[String],
    expert_key: impl Fn(&str) -> bool,
    coefficient_key: impl Fn(&str) -> bool,
    dense_dtype: impl Fn(&str, &HashMap<String, Array>) -> Dtype,
) -> Result<SourceReads> {
    if tables.is_empty()
        && !tensors
            .values()
            .any(|v| v.dtype() == Dtype::Uint8 || v.dtype() == Dtype::Int8)
    {
        return Ok(SourceReads::default());
    }
    let Some(plan) = projection_plan(
        paths,
        |k| tensors.contains_key(k),
        tables,
        &expert_key,
        coefficient_key,
    )?
    else {
        return Ok(SourceReads::default());
    };
    let dtypes: HashMap<_, _> = plan
        .tensors
        .iter()
        .filter(|row| row.residency.mode == ResidencyMode::Dense)
        .map(|row| {
            (
                row.logical_key.as_str(),
                dense_dtype(&row.logical_key, tensors),
            )
        })
        .collect();
    bind_layout_with_dtypes(tensors, &plan, expert_key, |key| {
        dtypes.get(key).copied().unwrap_or(Dtype::Bfloat16)
    })?;
    Ok(SourceReads::new(Arc::new(plan), paths))
}

/// Attach validated physical layouts to an existing tensor map without creating a load receipt.
/// Memory pricing uses the same row geometry with shape-only, neutral array placeholders.
pub fn bind_source_layout(
    tensors: &mut HashMap<String, Array>,
    plan: &LogicalWeightPlan,
    expert_key: impl Fn(&str) -> bool,
) -> Result<()> {
    bind_layout_with_dtypes(tensors, plan, expert_key, |_| Dtype::Bfloat16)
}

fn bind_layout_with_dtypes(
    tensors: &mut HashMap<String, Array>,
    plan: &LogicalWeightPlan,
    expert_key: impl Fn(&str) -> bool,
    dense_dtype: impl Fn(&str) -> Dtype,
) -> Result<()> {
    use gen_core::comfy_quant::ComfyQuantFormat;
    for t in &plan.tensors {
        let key = &t.logical_key;
        let Some(array) = tensors.get(key).cloned() else {
            return Err(format!("{key}: selected source disappeared").into());
        };
        let base = key
            .strip_suffix(".weight")
            .or_else(|| key.strip_suffix("_weight"))
            .or_else(|| expert_key(key).then_some(key.as_str()));
        let (format, scale, sx) = match &t.codec {
            TensorCodecSpec::ConvRotInt8 { geometry, .. } => {
                if t.residency.mode != ResidencyMode::Packed {
                    return Err(format!(
                        "{key}: ConvRot requires the Metal signed integer projection operator"
                    )
                    .into());
                }
                let base = base.ok_or("missing ConvRot projection base")?;
                tensors.insert(
                    format!("{base}.comfy_quant"),
                    crate::convrot_int8::descriptor_array(geometry.rotation_group),
                );
                continue;
            }
            TensorCodecSpec::RowwiseFp8 { scale } => {
                (ComfyQuantFormat::Fp8Rowwise, Some(scale), None)
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
        // Keep one normalized spelling for consumed producer aliases. A downstream component
        // remap must not apply a legacy scale to a value already bound or decoded here.
        let physical_base = key.strip_suffix(".weight").unwrap_or(key);
        for suffix in [".scale_weight", ".scale_input"] {
            tensors.remove(&format!("{physical_base}{suffix}"));
        }
        if t.residency.mode == ResidencyMode::Dense || base.is_none() {
            let dtype = dense_dtype(key);
            let value = if array.ndim() == 2 {
                crate::fp8::Fp8Linear::new(format, array, sw, sx, &t.shape, None)?
                    .dense_weight_as(dtype)?
            } else {
                let mut value = crate::fp8::decoded_codes(&array, format)?;
                if let Some(scale) = sw {
                    value = mlx_rs::ops::multiply(&value, &scale)?;
                }
                value.as_dtype(dtype)?
            };
            tensors.insert(key.clone(), value);
            for suffix in [".comfy_quant", ".weight_scale", ".input_scale"] {
                tensors.remove(&format!("{}{suffix}", base.unwrap_or(physical_base)));
            }
        } else if let Some(base) = base {
            let shape = (format == ComfyQuantFormat::Mxfp8).then_some(t.shape.as_slice());
            tensors.insert(
                format!("{base}.comfy_quant"),
                crate::fp8::descriptor_array(format, t.codec.full_precision_matrix_mult(), shape),
            );
            tensors.insert(
                format!("{base}.weight_scale"),
                sw.unwrap_or_else(crate::fp8::unit_scale),
            );
            if let Some(sx) = sx {
                tensors.insert(format!("{base}.input_scale"), sx);
            } else {
                tensors.remove(&format!("{base}.input_scale"));
            }
        }
    }
    Ok(())
}

/// Make a row-major weight view before exposing its bytes to a persistent cache key.
pub fn contiguous_weight(x: &mlx_rs::Array) -> crate::Result<mlx_rs::Array> {
    let stream = mlx_rs::StreamOrDevice::default();
    // SAFETY: MLX initializes an owned handle on success; the empty handle is released on failure.
    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status =
            mlx_sys::mlx_contiguous(&mut result, x.as_ptr(), false, stream.as_ref().as_ptr());
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(crate::Error::Msg(
                "mlx_contiguous failed while making a row-major copy".into(),
            ));
        }
        Ok(mlx_rs::Array::from_ptr(result))
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn selected_native_rotations_charge_one_shared_allocation_pair() {
        let row = |base: &str| {
            let shape = vec![32, 64];
            LogicalTensorPlan {
                logical_key: format!("{base}.weight"),
                physical_key: format!("{base}.weight"),
                encoding: WeightEncoding::Int8,
                codec_id: gen_core::convrot_int8::CODEC_ID,
                resident_encoding: WeightEncoding::DenseBf16,
                codec: TensorCodecSpec::ConvRotInt8 {
                    scale: format!("{base}.weight_scale"),
                    geometry: gen_core::convrot_int8::ConvRotInt8::from_geometry(
                        64,
                        &shape,
                        &[32, 1],
                    )
                    .unwrap(),
                },
                shape,
                source_bytes: 32 * 64,
                residency: PlannedResidency {
                    mode: ResidencyMode::Packed,
                    resident_bytes: 32 * 64,
                },
                transform: None,
            }
        };
        let plan = LogicalWeightPlan {
            mapping_id: "test",
            tensors: vec![row("a"), row("b")],
            companions: vec![],
            source_bytes: 2 * 32 * 64,
        };
        let headers = [
            (
                "a.weight",
                gen_core::weightsmeta::Dtype::I8,
                vec![32, 64],
                2048,
            ),
            ("a.bias", gen_core::weightsmeta::Dtype::F32, vec![32], 128),
            (
                "b.weight",
                gen_core::weightsmeta::Dtype::I8,
                vec![32, 64],
                2048,
            ),
            (
                "b.bias",
                gen_core::weightsmeta::Dtype::F8_E4M3,
                vec![32],
                32,
            ),
        ]
        .map(|(name, dtype, shape, data_bytes)| {
            gen_core::weightsmeta::SafetensorsTensorHeader {
                name: name.into(),
                dtype,
                shape,
                data_bytes,
            }
        });
        let physical = projected_bytes_from_plan(&plan, &headers, |_| 2).unwrap();
        assert_eq!(physical["a.bias"], 128);
        assert_eq!(physical["b.bias"], 64);
        let descriptor = gen_core::convrot_int8::MAX_DESCRIPTOR_BYTES as u64;
        let rotation = 64 * 64 * 4 * 2;
        assert_eq!(
            retained_auxiliary_bytes(&plan, |_| true, Some),
            Some(rotation / 2)
        );
        assert_eq!(
            load_auxiliary_bytes(&plan, |_| true, Some),
            Some(2 * descriptor + rotation)
        );
        assert_eq!(
            load_auxiliary_bytes(&plan, |key| key == "b.weight", Some),
            Some(descriptor + rotation)
        );
        assert_eq!(load_auxiliary_bytes(&plan, |_| false, Some), Some(0));
        assert_eq!(
            load_auxiliary_bytes(&plan, |_| true, |_| Some(u64::MAX)),
            None
        );
    }
}

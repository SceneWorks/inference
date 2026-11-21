//! Per-reader, backend-independent measurements. The observer retains the plan and byte counts;
//! backend tensor handles, mapped files, and workspaces remain owned by their provider.
use crate::checkpoint_codec::{
    CodecResidencyReport, LogicalReadMaterialization, LogicalWeightPlan, LogicalWeightReceipt,
    RegimeDemotion, ResidencyMode, TensorCodecSpec, WeightEncoding,
};
use crate::checkpoint_facts::{
    CheckpointWeightFacts, ExecutionRepresentation, NativeExecutionCapability,
};
use crate::{CheckpointReceiptScope, Error, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State {
    rows: BTreeMap<usize, (ExecutionRepresentation, u64)>,
    dense_layouts: BTreeMap<usize, (Vec<usize>, usize)>,
}

pub struct CheckpointReadAccounting {
    plan: Arc<LogicalWeightPlan>,
    capability: NativeExecutionCapability,
    indices: HashMap<String, usize>,
    state: Mutex<State>,
}
impl std::fmt::Debug for CheckpointReadAccounting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointReadAccounting")
            .field("mapping", &self.plan.mapping_id)
            .finish_non_exhaustive()
    }
}
impl CheckpointReadAccounting {
    pub fn new(
        plan: Arc<LogicalWeightPlan>,
        capability: NativeExecutionCapability,
        paths: &[PathBuf],
    ) -> Arc<Self> {
        let indices = plan
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.logical_key.clone(), i))
            .collect();
        let tracker = Arc::new(Self {
            plan,
            capability,
            indices,
            state: Default::default(),
        });
        let observed = tracker.clone();
        CheckpointReceiptScope::register(paths, move || observed.facts());
        tracker
    }

    pub fn capability(&self) -> &NativeExecutionCapability {
        &self.capability
    }

    fn index(&self, key: &str) -> Result<usize> {
        self.indices
            .get(key)
            .copied()
            .ok_or_else(|| Error::Msg(format!("unknown measured checkpoint row {key:?}")))
    }

    /// Specify the requested dense compute width and unpadded logical geometry before materialization.
    /// This is an independent operator contract; the later measurement must agree with these bytes.
    pub fn declare_dense_layout(
        &self,
        key: &str,
        shape: &[usize],
        element_bytes: usize,
    ) -> Result<()> {
        let i = self.index(key)?;
        let row = &self.plan.tensors[i];
        if !matches!(element_bytes, 2 | 4 | 8) {
            return Err(format!("{key}: unsupported dense compute width {element_bytes}").into());
        }
        if shape != row.shape {
            let allowed = match &row.codec {
                TensorCodecSpec::Mxfp8 { stored_shape, .. } => {
                    shape.len() == 2
                        && crate::mxfp8_padded_shape([shape[0], shape[1]]) == *stored_shape
                }
                TensorCodecSpec::Nvfp4 { .. } => {
                    shape.len() == 2
                        && row.shape.len() == 2
                        && shape
                            .iter()
                            .zip(&row.shape)
                            .all(|(v, stored)| *v > 0 && v <= stored && stored - v < 16)
                }
                _ => false,
            };
            if !allowed {
                return Err(format!(
                    "{key}: dense shape {shape:?} disagrees with planned geometry {:?}",
                    row.shape
                )
                .into());
            }
        }
        let bytes = shape
            .iter()
            .try_fold(element_bytes, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| Error::Msg(format!("{key}: dense layout exceeds address space")))?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((old, width)) = state.dense_layouts.get(&i) {
            if old != shape || *width != element_bytes {
                return Err(format!("{key}: contradictory dense layout declarations").into());
            }
            return Ok(());
        }
        if state.rows.get(&i).is_some_and(|(r, n)| {
            *r == ExecutionRepresentation::DenseFallback && *n != bytes as u64
        }) {
            return Err(
                format!("{key}: declared layout contradicts an existing measurement").into(),
            );
        }
        state
            .dense_layouts
            .insert(i, (shape.to_vec(), element_bytes));
        Ok(())
    }

    /// Record the representation the constructed operator actually retains. A repeated read
    /// or evicted window replaces its own measurement and never increments the tensor count.
    pub fn record(&self, key: &str, repr: ExecutionRepresentation, bytes: u64) -> Result<()> {
        let i = self.index(key)?;
        let row = &self.plan.tensors[i];
        if repr == ExecutionRepresentation::NativePacked
            && row.residency.mode != ResidencyMode::Packed
        {
            return Err(format!("{key}: a dense plan cannot claim packed execution").into());
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((prev, n)) = state.rows.get(&i) {
            if *prev == repr && *n != bytes {
                return Err(
                    format!("{key}: changed {repr:?} residency from {n} to {bytes} bytes").into(),
                );
            }
            if row.residency.mode == ResidencyMode::Packed
                && *prev == ExecutionRepresentation::DenseFallback
                && repr == ExecutionRepresentation::NativePacked
            {
                // A subsequent read can re-stage the packed source. A constructed dense consumer
                // remains authoritative until its source pin/model is replaced.
                return Ok(());
            }
        }
        state.rows.insert(i, (repr, bytes));
        Ok(())
    }

    pub fn materialized_bytes(&self, key: &str) -> Option<u64> {
        let i = self.indices.get(key)?;
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .rows
            .get(i)
            .map(|(_, n)| *n)
    }

    fn receipt_for(&self, state: &State) -> LogicalWeightReceipt {
        let mut cells: BTreeMap<_, CodecResidencyReport> = BTreeMap::new();
        let mut owners = BTreeMap::new();
        let mut demotions = Vec::new();
        for (&i, &(representation, bytes)) in &state.rows {
            let t = &self.plan.tensors[i];
            let id = (t.codec_id, representation);
            owners.insert(t.physical_key.as_str(), id);
            let report = cells.entry(id).or_insert(CodecResidencyReport {
                codec_id: t.codec_id,
                representation,
                tensor_count: 0,
                source_bytes: 0,
                resident_bytes: 0,
            });
            report.tensor_count += 1;
            report.resident_bytes = report.resident_bytes.saturating_add(bytes);
            report.source_bytes = report.source_bytes.saturating_add(t.source_bytes);
            if t.residency.mode == ResidencyMode::Packed
                && representation == ExecutionRepresentation::DenseFallback
            {
                demotions.push(RegimeDemotion {
                    logical_key: t.logical_key.clone(),
                    codec_id: t.codec_id,
                    resident_bytes: bytes,
                });
            }
        }
        for c in &self.plan.companions {
            if let Some(cell) = owners
                .get(c.owner_physical_key.as_str())
                .and_then(|id| cells.get_mut(id))
            {
                cell.source_bytes = cell.source_bytes.saturating_add(c.source_bytes);
            }
        }
        let count = state.rows.len();
        let materialization = if count == 0 {
            LogicalReadMaterialization::Deferred
        } else if count == self.plan.tensors.len() {
            LogicalReadMaterialization::Materialized
        } else {
            LogicalReadMaterialization::Partial
        };
        LogicalWeightReceipt {
            mapping_id: self.plan.mapping_id,
            tensor_count: count,
            source_bytes: cells.values().map(|c| c.source_bytes).sum(),
            materialization,
            residency: cells.into_values().collect(),
            demotions,
        }
    }

    pub fn receipt(&self) -> LogicalWeightReceipt {
        self.receipt_for(&self.state.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn facts(&self) -> Result<CheckpointWeightFacts> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let receipt = self.receipt_for(&state);
        let mut plan = std::borrow::Cow::Borrowed(self.plan.as_ref());
        for (&i, (shape, width)) in &state.dense_layouts {
            let row = &mut plan.to_mut().tensors[i];
            row.shape = shape.clone();
            row.resident_encoding = match width {
                2 if row.resident_encoding == WeightEncoding::DenseF16 => WeightEncoding::DenseF16,
                2 => WeightEncoding::DenseBf16,
                4 => WeightEncoding::DenseF32,
                8 => WeightEncoding::DenseF64,
                _ => unreachable!("validated dense width"),
            };
            if row.residency.mode == ResidencyMode::Dense {
                row.residency.resident_bytes =
                    shape.iter().fold(*width as u64, |n, &d| n * d as u64);
            }
        }
        CheckpointWeightFacts::new(plan.as_ref(), self.capability.clone(), receipt)
            .map_err(|e| Error::Msg(e.to_string()))
    }
}

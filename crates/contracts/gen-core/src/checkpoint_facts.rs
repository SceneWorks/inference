//! **Three separate, correlated facts about one loaded checkpoint** (epic 11037, sc-21484).
//!
//! A quantized import has three questions that consumers routinely collapse into one, and the
//! collapse is exactly how a dense BF16 fallback ends up labelled "NVFP4" in a user-visible model
//! fact:
//!
//! 1. **What is the source?** — the immutable binding and topology of the artifact that was
//!    imported. SceneWorks owns the semantic half (`ImportPlanV1`); this crate's half is the
//!    [`SourceBinding`] token, taken from the *verified* [`PinnedWeightsFile`] the load actually
//!    read.
//! 2. **What codecs does the source store?** — the device-independent inventory compiled into a
//!    [`LogicalWeightPlan`]: per codec row, how many logical tensors and how many source bytes.
//!    A checkpoint whose projections are stored `nvfp4-v1` says so here **on every host**,
//!    including hosts that cannot execute NVFP4 natively.
//! 3. **What did this host actually materialize?** — the measured
//!    [`LogicalWeightReceipt`], now split per **[`ExecutionRepresentation`]** so a
//!    `nvfp4-v1` row that ran the packed W4A4 operand and a `nvfp4-v1` row that decoded to dense
//!    BF16 are *different rows*, not one indistinguishable total.
//!
//! [`CheckpointWeightFacts`] carries all three together and **validates** their correlation, so
//! the three can be read separately without any of them drifting from the others:
//!
//! * The receipt must be the receipt of *this* plan (same `mapping_id`).
//! * Every codec the receipt reports must be one the source inventory declares — a receipt cannot
//!   *alias* the source (report `int8-per-row-v1`, or a `q4` tier, for a file stored `nvfp4-v1`).
//! * A [`ExecutionRepresentation::NativePacked`] row must be backed **both** by a plan that priced
//!   that many packed tensors **and** by a host [`NativeExecutionCapability`] that lists the codec.
//!   A non-native host runs the declared dense fallback and its receipt cannot label the run
//!   native, because there is no capability to license the label and no planned packed pricing to
//!   measure against.
//! * Nothing may exceed the plan: the receipt is measured over what has materialized *so far*, so
//!   every count and byte total is `<=` the plan's, and equal once the load is complete
//!   ([`CheckpointWeightFacts::is_complete`]).
//!
//! # Why the source inventory counts a physical tensor once
//!
//! [`SourceCodecEntry::source_bytes`] sums each **distinct physical key** once, while
//! `tensor_count` counts **logical** tensors. Those differ whenever one stored tensor feeds several
//! logical outputs (a fused QKV projection split at read time), and summing per logical row would
//! report a file larger than it is. Resident bytes stay attributed per logical row, because that
//! is what actually occupies memory.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::checkpoint_codec::{
    CodecResidencyReport, LogicalReadMaterialization, LogicalWeightPlan, LogicalWeightReceipt,
    ResidencyMode,
};
use crate::runtime::{FileStatFingerprint, PinnedWeightsFile};

/// The representation a codec row was **actually materialized as** on this host.
///
/// This is the fact the plan's [`ResidencyMode`] *predicts* and the receipt *reports*. They are
/// deliberately separate types: a `ResidencyMode` is a plan-time pricing decision, an
/// `ExecutionRepresentation` is a measured outcome, and the whole point of the pair is that a
/// consumer can tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutionRepresentation {
    /// The stored packing itself is the execution operand — packed fp8 E4M3 GEMM, NVFP4 W4A4.
    NativePacked,
    /// The stored bytes were decoded to the codec's dense resident encoding (BF16 for every
    /// quantized row here) and execute dense.
    DenseFallback,
}

impl ExecutionRepresentation {
    /// The stable wire label. SceneWorks renders these into asset metadata and model facts, so they
    /// are part of the handoff contract and do not change with refactors.
    pub fn label(self) -> &'static str {
        match self {
            Self::NativePacked => "native-packed",
            Self::DenseFallback => "dense-fallback",
        }
    }

    /// The representation a plan's residency decision predicts.
    pub fn from_residency(mode: ResidencyMode) -> Self {
        match mode {
            ResidencyMode::Packed => Self::NativePacked,
            ResidencyMode::Dense => Self::DenseFallback,
        }
    }
}

impl fmt::Display for ExecutionRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What **this host** can execute in a codec's stored packing — a host fact, stated without any
/// checkpoint in hand.
///
/// Backends render their own residency policy into this shape (candle-gen's
/// `CandleCodecResidency::native_execution_capability`), so a consumer across the worker boundary
/// reads one form regardless of backend. A host below the NVFP4 `sm_120` floor — including
/// datacenter `sm_100`, which is outside this leg's kernel — lists no `nvfp4-v1` here, and that
/// absence is what makes a native label unrepresentable rather than merely unlikely.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeExecutionCapability {
    native_codec_ids: BTreeSet<&'static str>,
}

impl NativeExecutionCapability {
    /// The dense-only host: no codec executes in its stored packing.
    pub const fn dense_only() -> Self {
        Self {
            native_codec_ids: BTreeSet::new(),
        }
    }

    /// Declare the codec rows this host executes natively.
    pub fn new(codec_ids: impl IntoIterator<Item = &'static str>) -> Self {
        Self {
            native_codec_ids: codec_ids.into_iter().collect(),
        }
    }

    /// Whether this host executes `codec_id` in its stored packing.
    pub fn executes_natively(&self, codec_id: &str) -> bool {
        self.native_codec_ids.contains(codec_id)
    }

    /// The declared codec ids, sorted.
    pub fn native_codec_ids(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.native_codec_ids.iter().copied()
    }

    /// Whether this host executes nothing natively (the dense-fallback host).
    pub fn is_dense_only(&self) -> bool {
        self.native_codec_ids.is_empty()
    }
}

/// The verified identity of the artifact a load actually read.
///
/// Taken from a [`PinnedWeightsFile`] whose [`PinnedWeightsFile::ensure_unchanged`] passes at the
/// moment the facts are assembled, so the codec inventory and the receipt are tied to a source that
/// still is what it was when the plan was compiled — not merely to a path string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceBinding {
    canonical_path: PathBuf,
    target_fingerprint: FileStatFingerprint,
}

impl SourceBinding {
    /// Re-verify the pin and capture its identity. Fails exactly when
    /// [`PinnedWeightsFile::ensure_unchanged`] does.
    pub fn verify(pin: &PinnedWeightsFile) -> crate::Result<Self> {
        pin.ensure_unchanged()?;
        Ok(Self {
            canonical_path: pin.canonical_target_path().to_path_buf(),
            target_fingerprint: pin.target_fingerprint().clone(),
        })
    }

    /// The canonical target path the pin resolved.
    pub fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    /// Mutation-sensitive identity of the resolved target.
    pub fn target_fingerprint(&self) -> &FileStatFingerprint {
        &self.target_fingerprint
    }
}

/// One codec row of the **source inventory**: what the file stores, plus what this load's plan
/// priced for it.
///
/// The first two fields are device-independent source facts. The `planned_*` fields are this
/// device's pricing, kept on the same row so a consumer can say "the source is `nvfp4-v1`, and on
/// this host N of its M tensors execute packed" without joining two collections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCodecEntry {
    pub codec_id: &'static str,
    /// Logical tensors stored in this codec.
    pub tensor_count: usize,
    /// Source bytes of those tensors plus their companions, each **distinct physical tensor**
    /// counted once.
    pub source_bytes: u64,
    /// How many of `tensor_count` this load's plan priced [`ResidencyMode::Packed`].
    pub planned_native_packed_tensors: usize,
    /// Planned resident bytes of the packed subset — weight rows plus the companions those rows
    /// retain.
    pub planned_native_packed_resident_bytes: u64,
    /// Planned resident bytes of the dense-fallback subset, same accounting.
    pub planned_dense_resident_bytes: u64,
}

impl SourceCodecEntry {
    /// Logical tensors this load's plan priced dense.
    pub fn planned_dense_tensors(&self) -> usize {
        self.tensor_count
            .saturating_sub(self.planned_native_packed_tensors)
    }

    /// Planned tensors and resident bytes for one representation.
    fn planned(&self, representation: ExecutionRepresentation) -> (usize, u64) {
        match representation {
            ExecutionRepresentation::NativePacked => (
                self.planned_native_packed_tensors,
                self.planned_native_packed_resident_bytes,
            ),
            ExecutionRepresentation::DenseFallback => (
                self.planned_dense_tensors(),
                self.planned_dense_resident_bytes,
            ),
        }
    }
}

/// The complete source-codec inventory of one compiled plan, sorted by codec id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCodecSummary {
    pub mapping_id: &'static str,
    pub entries: Vec<SourceCodecEntry>,
    /// Every byte of the file's data region, from the plan.
    pub source_bytes: u64,
    /// Logical tensors in the plan.
    pub tensor_count: usize,
}

impl SourceCodecSummary {
    /// Compile the inventory from a plan. Pure — it reads the plan and nothing else.
    pub fn of(plan: &LogicalWeightPlan) -> Self {
        // Codec row → accumulator. Source bytes are accumulated over *distinct physical keys*, so
        // a fused physical tensor feeding several logical outputs is counted once.
        let mut entries: BTreeMap<&'static str, SourceCodecEntry> = BTreeMap::new();
        let mut codec_by_owner: BTreeMap<&str, &'static str> = BTreeMap::new();
        let mut counted_physical: BTreeSet<&str> = BTreeSet::new();
        for tensor in &plan.tensors {
            codec_by_owner.insert(tensor.physical_key.as_str(), tensor.codec_id);
            let entry = entries
                .entry(tensor.codec_id)
                .or_insert_with(|| SourceCodecEntry {
                    codec_id: tensor.codec_id,
                    tensor_count: 0,
                    source_bytes: 0,
                    planned_native_packed_tensors: 0,
                    planned_native_packed_resident_bytes: 0,
                    planned_dense_resident_bytes: 0,
                });
            entry.tensor_count += 1;
            if counted_physical.insert(tensor.physical_key.as_str()) {
                entry.source_bytes = entry.source_bytes.saturating_add(tensor.source_bytes);
            }
            match tensor.residency.mode {
                ResidencyMode::Packed => {
                    entry.planned_native_packed_tensors += 1;
                    entry.planned_native_packed_resident_bytes = entry
                        .planned_native_packed_resident_bytes
                        .saturating_add(tensor.residency.resident_bytes);
                }
                ResidencyMode::Dense => {
                    entry.planned_dense_resident_bytes = entry
                        .planned_dense_resident_bytes
                        .saturating_add(tensor.residency.resident_bytes);
                }
            }
        }
        // A companion belongs to its owner's codec row, and to the representation its owner was
        // priced in: a packed load retains its scales, a dense decode consumes them (zero bytes).
        let packed_owners: BTreeSet<&str> = plan
            .tensors
            .iter()
            .filter(|tensor| tensor.residency.mode == ResidencyMode::Packed)
            .map(|tensor| tensor.physical_key.as_str())
            .collect();
        for companion in &plan.companions {
            let Some(codec_id) = codec_by_owner.get(companion.owner_physical_key.as_str()) else {
                continue;
            };
            let Some(entry) = entries.get_mut(codec_id) else {
                continue;
            };
            if counted_physical.insert(companion.physical_key.as_str()) {
                entry.source_bytes = entry.source_bytes.saturating_add(companion.source_bytes);
            }
            if packed_owners.contains(companion.owner_physical_key.as_str()) {
                entry.planned_native_packed_resident_bytes = entry
                    .planned_native_packed_resident_bytes
                    .saturating_add(companion.resident_bytes);
            } else {
                entry.planned_dense_resident_bytes = entry
                    .planned_dense_resident_bytes
                    .saturating_add(companion.resident_bytes);
            }
        }
        Self {
            mapping_id: plan.mapping_id,
            entries: entries.into_values().collect(),
            source_bytes: plan.source_bytes,
            tensor_count: plan.tensor_count(),
        }
    }

    /// The inventory row for one codec, or `None` when the source stores nothing in it.
    pub fn entry(&self, codec_id: &str) -> Option<&SourceCodecEntry> {
        self.entries.iter().find(|entry| entry.codec_id == codec_id)
    }

    /// Whether the source stores any tensor in this codec — the **source** question, answered the
    /// same on every host.
    pub fn declares(&self, codec_id: &str) -> bool {
        self.entry(codec_id).is_some()
    }
}

/// Why a set of checkpoint facts is not self-consistent. Every variant names the codec (or the
/// mapping) it fails on; none of them is recoverable by rounding — each one means a consumer was
/// about to be told something untrue about the load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointWeightFactsError {
    /// The receipt was produced by a different plan than the inventory.
    MappingMismatch {
        plan_mapping_id: &'static str,
        receipt_mapping_id: &'static str,
    },
    /// The receipt reports a codec the source does not store — the receipt is *aliasing* the
    /// source.
    UnplannedCodec { codec_id: &'static str },
    /// The receipt labels tensors natively packed that this host cannot execute natively.
    NativeWithoutCapability {
        codec_id: &'static str,
        tensor_count: usize,
    },
    /// The receipt reports more tensors in a representation than the plan priced for it. With a
    /// packed representation and a zero-packed plan this is precisely "a dense fallback labelled
    /// native".
    RepresentationExceedsPlan {
        codec_id: &'static str,
        representation: ExecutionRepresentation,
        reported_tensors: usize,
        planned_tensors: usize,
    },
    /// The receipt measured more resident bytes in a representation than the plan priced for it.
    ResidentBytesExceedPlan {
        codec_id: &'static str,
        representation: ExecutionRepresentation,
        measured_bytes: u64,
        planned_bytes: u64,
    },
    /// The receipt accounts for more source bytes than the whole file has.
    SourceBytesExceedPlan {
        measured_bytes: u64,
        planned_bytes: u64,
    },
    /// A materialized receipt reports more logical tensors than the plan contains.
    TensorCountExceedsPlan { reported: usize, planned: usize },
    /// A deferred read reported residency rows. Deferred means nothing was evaluated, so there is
    /// nothing measured to report; inventing rows there is the same class of untruth as labelling
    /// a dense row native.
    DeferredReadReportedResidency { rows: usize },
}

impl fmt::Display for CheckpointWeightFactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MappingMismatch {
                plan_mapping_id,
                receipt_mapping_id,
            } => write!(
                f,
                "checkpoint facts: the receipt was produced under mapping \
                 {receipt_mapping_id:?} but the source inventory was compiled under \
                 {plan_mapping_id:?}; these are facts about two different loads"
            ),
            Self::UnplannedCodec { codec_id } => write!(
                f,
                "checkpoint facts: the receipt reports codec {codec_id:?}, which the source \
                 inventory does not declare; a receipt may report how the source was materialized, \
                 never re-label what the source is"
            ),
            Self::NativeWithoutCapability {
                codec_id,
                tensor_count,
            } => write!(
                f,
                "checkpoint facts: the receipt labels {tensor_count} tensor(s) of codec \
                 {codec_id:?} `{}`, but this host declares no native execution for that codec — a \
                 non-native host runs the declared dense fallback and its receipt must say so",
                ExecutionRepresentation::NativePacked
            ),
            Self::RepresentationExceedsPlan {
                codec_id,
                representation,
                reported_tensors,
                planned_tensors,
            } => write!(
                f,
                "checkpoint facts: the receipt reports {reported_tensors} tensor(s) of codec \
                 {codec_id:?} as `{representation}`, but the plan priced only {planned_tensors} \
                 there; the receipt measures what materialized and cannot exceed what was planned"
            ),
            Self::ResidentBytesExceedPlan {
                codec_id,
                representation,
                measured_bytes,
                planned_bytes,
            } => write!(
                f,
                "checkpoint facts: codec {codec_id:?} measured {measured_bytes} resident byte(s) \
                 as `{representation}` against {planned_bytes} planned; planned and measured \
                 residency must agree, and a measurement above the plan is double-counting"
            ),
            Self::SourceBytesExceedPlan {
                measured_bytes,
                planned_bytes,
            } => write!(
                f,
                "checkpoint facts: the receipt accounts for {measured_bytes} source byte(s) but \
                 the plan's file holds {planned_bytes}"
            ),
            Self::TensorCountExceedsPlan { reported, planned } => write!(
                f,
                "checkpoint facts: the receipt reports {reported} materialized tensor(s) against a \
                 plan of {planned}"
            ),
            Self::DeferredReadReportedResidency { rows } => write!(
                f,
                "checkpoint facts: a deferred read reported {rows} residency row(s); a deferred \
                 read evaluated nothing and must report no measured residency"
            ),
        }
    }
}

impl std::error::Error for CheckpointWeightFactsError {}

impl From<CheckpointWeightFactsError> for crate::Error {
    fn from(error: CheckpointWeightFactsError) -> Self {
        crate::Error::Unsupported(error.to_string())
    }
}

/// The three correlated facts about one loaded checkpoint, validated against each other.
///
/// Construct with [`CheckpointWeightFacts::new`]; there is no field-wise constructor, so a
/// consumer can never be handed an unvalidated set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointWeightFacts {
    source_binding: Option<SourceBinding>,
    capability: NativeExecutionCapability,
    source: SourceCodecSummary,
    receipt: LogicalWeightReceipt,
}

impl CheckpointWeightFacts {
    /// Assemble and validate. `capability` is the **host's** native-execution declaration; the
    /// receipt may not claim a native representation the capability does not license.
    pub fn new(
        plan: &LogicalWeightPlan,
        capability: NativeExecutionCapability,
        receipt: LogicalWeightReceipt,
    ) -> Result<Self, CheckpointWeightFactsError> {
        let source = SourceCodecSummary::of(plan);
        validate(&source, &capability, &receipt)?;
        Ok(Self {
            source_binding: None,
            capability,
            source,
            receipt,
        })
    }

    /// Tie these facts to the verified source binding they describe. Re-verifies the pin, so a file
    /// that changed under the load refuses here rather than shipping facts about bytes that are
    /// gone.
    pub fn with_verified_source(mut self, pin: &PinnedWeightsFile) -> crate::Result<Self> {
        self.source_binding = Some(SourceBinding::verify(pin)?);
        Ok(self)
    }

    /// The verified source binding, when the loader supplied one.
    pub fn source_binding(&self) -> Option<&SourceBinding> {
        self.source_binding.as_ref()
    }

    /// This host's native-execution declaration.
    pub fn capability(&self) -> &NativeExecutionCapability {
        &self.capability
    }

    /// Fact 2: what the source stores.
    pub fn source(&self) -> &SourceCodecSummary {
        &self.source
    }

    /// Fact 3: what actually materialized, split per [`ExecutionRepresentation`].
    pub fn receipt(&self) -> &LogicalWeightReceipt {
        &self.receipt
    }

    /// The measured rows, sorted by `(codec_id, representation)`.
    pub fn materialized(&self) -> &[CodecResidencyReport] {
        &self.receipt.residency
    }

    /// The measured row for one `(codec, representation)` pair.
    pub fn materialized_as(
        &self,
        codec_id: &str,
        representation: ExecutionRepresentation,
    ) -> Option<&CodecResidencyReport> {
        self.receipt
            .residency
            .iter()
            .find(|row| row.codec_id == codec_id && row.representation == representation)
    }

    /// Whether **any** tensor of this codec actually executed in its stored packing. This is the
    /// question a product fact must ask before saying "native NVFP4"; it is false on every host
    /// that took the dense fallback, however the source is stored.
    pub fn executes_natively(&self, codec_id: &str) -> bool {
        self.materialized_as(codec_id, ExecutionRepresentation::NativePacked)
            .is_some_and(|row| row.tensor_count > 0)
    }

    /// Total measured resident bytes across every row.
    pub fn resident_bytes(&self) -> u64 {
        self.receipt.resident_bytes()
    }

    /// Whether the receipt now covers the plan's whole tensor surface. Only then do planned and
    /// measured residency have to be *equal* rather than merely bounded.
    pub fn is_complete(&self) -> bool {
        self.receipt.tensor_count == self.source.tensor_count
    }
}

fn validate(
    source: &SourceCodecSummary,
    capability: &NativeExecutionCapability,
    receipt: &LogicalWeightReceipt,
) -> Result<(), CheckpointWeightFactsError> {
    if source.mapping_id != receipt.mapping_id {
        return Err(CheckpointWeightFactsError::MappingMismatch {
            plan_mapping_id: source.mapping_id,
            receipt_mapping_id: receipt.mapping_id,
        });
    }
    if receipt.materialization == LogicalReadMaterialization::Deferred
        && !receipt.residency.is_empty()
    {
        return Err(CheckpointWeightFactsError::DeferredReadReportedResidency {
            rows: receipt.residency.len(),
        });
    }
    if receipt.tensor_count > source.tensor_count {
        return Err(CheckpointWeightFactsError::TensorCountExceedsPlan {
            reported: receipt.tensor_count,
            planned: source.tensor_count,
        });
    }
    if receipt.source_bytes > source.source_bytes {
        return Err(CheckpointWeightFactsError::SourceBytesExceedPlan {
            measured_bytes: receipt.source_bytes,
            planned_bytes: source.source_bytes,
        });
    }
    for row in &receipt.residency {
        let Some(entry) = source.entry(row.codec_id) else {
            return Err(CheckpointWeightFactsError::UnplannedCodec {
                codec_id: row.codec_id,
            });
        };
        if row.representation == ExecutionRepresentation::NativePacked
            && row.tensor_count > 0
            && !capability.executes_natively(row.codec_id)
        {
            return Err(CheckpointWeightFactsError::NativeWithoutCapability {
                codec_id: row.codec_id,
                tensor_count: row.tensor_count,
            });
        }
        let (planned_tensors, planned_bytes) = entry.planned(row.representation);
        if row.tensor_count > planned_tensors {
            return Err(CheckpointWeightFactsError::RepresentationExceedsPlan {
                codec_id: row.codec_id,
                representation: row.representation,
                reported_tensors: row.tensor_count,
                planned_tensors,
            });
        }
        if row.resident_bytes > planned_bytes {
            return Err(CheckpointWeightFactsError::ResidentBytesExceedPlan {
                codec_id: row.codec_id,
                representation: row.representation,
                measured_bytes: row.resident_bytes,
                planned_bytes,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_codec::{
        CompanionRole, CompanionTensorPlan, LogicalTensorPlan, PlannedResidency, TensorCodecSpec,
        WeightEncoding, DENSE_BF16_CODEC, INT8_PER_ROW_CODEC, NVFP4_CODEC,
    };

    const MAPPING: &str = "facts-test-v1";

    fn nvfp4_tensor(key: &str, mode: ResidencyMode, resident: u64) -> LogicalTensorPlan {
        LogicalTensorPlan {
            logical_key: key.to_owned(),
            physical_key: key.to_owned(),
            encoding: WeightEncoding::UInt8,
            shape: vec![64, 64],
            source_bytes: 2048,
            codec_id: NVFP4_CODEC.codec_id,
            resident_encoding: WeightEncoding::DenseBf16,
            codec: TensorCodecSpec::Nvfp4 {
                block_scale: format!("{key}_scale"),
                global_scale: format!("{key}_scale_2"),
                input_scale: None,
                stored_shape: [64, 64],
                logical_shape: [64, 64],
                logical_shape_declared: true,
                full_precision_matrix_mult: false,
            },
            residency: PlannedResidency {
                mode,
                resident_bytes: resident,
            },
        }
    }

    fn dense_tensor(key: &str) -> LogicalTensorPlan {
        LogicalTensorPlan {
            logical_key: key.to_owned(),
            physical_key: key.to_owned(),
            encoding: WeightEncoding::DenseBf16,
            shape: vec![8, 8],
            source_bytes: 128,
            codec_id: DENSE_BF16_CODEC.codec_id,
            resident_encoding: WeightEncoding::DenseBf16,
            codec: TensorCodecSpec::Dense,
            residency: PlannedResidency {
                mode: ResidencyMode::Dense,
                resident_bytes: 128,
            },
        }
    }

    fn scale_companion(owner: &str, resident: u64) -> CompanionTensorPlan {
        CompanionTensorPlan {
            physical_key: format!("{owner}_scale"),
            role: CompanionRole::WeightScale,
            owner_physical_key: owner.to_owned(),
            source_bytes: 256,
            resident_bytes: resident,
        }
    }

    /// A plan with one packed NVFP4 row, one dense-fallback NVFP4 row, and a dense BF16 row.
    fn mixed_plan() -> LogicalWeightPlan {
        LogicalWeightPlan {
            mapping_id: MAPPING,
            tensors: vec![
                nvfp4_tensor("packed", ResidencyMode::Packed, 2048),
                nvfp4_tensor("fallback", ResidencyMode::Dense, 8192),
                dense_tensor("plain"),
            ],
            companions: vec![
                scale_companion("packed", 256),
                scale_companion("fallback", 0),
            ],
            source_bytes: 2048 + 2048 + 128 + 256 + 256,
        }
    }

    fn nvfp4_capability() -> NativeExecutionCapability {
        NativeExecutionCapability::new([NVFP4_CODEC.codec_id])
    }

    fn row(
        codec_id: &'static str,
        representation: ExecutionRepresentation,
        tensor_count: usize,
        source_bytes: u64,
        resident_bytes: u64,
    ) -> CodecResidencyReport {
        CodecResidencyReport {
            codec_id,
            representation,
            tensor_count,
            source_bytes,
            resident_bytes,
        }
    }

    /// The receipt a complete materialization of [`mixed_plan`] measures.
    fn honest_receipt() -> LogicalWeightReceipt {
        LogicalWeightReceipt {
            mapping_id: MAPPING,
            tensor_count: 3,
            source_bytes: 2048 + 2048 + 128 + 256 + 256,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![
                row(
                    DENSE_BF16_CODEC.codec_id,
                    ExecutionRepresentation::DenseFallback,
                    1,
                    128,
                    128,
                ),
                row(
                    NVFP4_CODEC.codec_id,
                    ExecutionRepresentation::DenseFallback,
                    1,
                    2048 + 256,
                    8192,
                ),
                row(
                    NVFP4_CODEC.codec_id,
                    ExecutionRepresentation::NativePacked,
                    1,
                    2048 + 256,
                    2048 + 256,
                ),
            ],
        }
    }

    /// **The three facts are separate and correlated (epic E8).** The source inventory says
    /// `nvfp4-v1` regardless of representation; the receipt splits the same codec into the row that
    /// executed packed and the row that fell back dense.
    #[test]
    fn source_inventory_and_receipt_answer_different_questions_about_one_codec() {
        let plan = mixed_plan();
        let facts =
            CheckpointWeightFacts::new(&plan, nvfp4_capability(), honest_receipt()).expect("valid");

        // Fact 2 — the source. Two logical tensors are stored NVFP4, whatever the host does.
        let entry = facts
            .source()
            .entry(NVFP4_CODEC.codec_id)
            .expect("the source declares nvfp4");
        assert_eq!(entry.tensor_count, 2);
        assert_eq!(entry.planned_native_packed_tensors, 1);
        assert_eq!(entry.planned_dense_tensors(), 1);
        // Each distinct physical tensor once: two weights (2048 each) + two scales (256 each).
        assert_eq!(entry.source_bytes, 2048 + 2048 + 256 + 256);
        assert!(facts.source().declares(NVFP4_CODEC.codec_id));
        assert!(!facts.source().declares(INT8_PER_ROW_CODEC.codec_id));

        // Fact 3 — what materialized, split per representation.
        assert_eq!(
            facts
                .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::NativePacked)
                .expect("a packed row")
                .resident_bytes,
            2048 + 256
        );
        assert_eq!(
            facts
                .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::DenseFallback)
                .expect("a dense-fallback row")
                .resident_bytes,
            8192
        );
        assert!(facts.executes_natively(NVFP4_CODEC.codec_id));
        assert!(facts.is_complete());
        // E4: planned and measured residency agree once the whole surface materialized.
        assert_eq!(facts.resident_bytes(), plan.resident_bytes());
    }

    /// **AC3: a non-native host executes the declared dense fallback and its receipt never labels
    /// the run native.** The same file, planned dense-only, is still `nvfp4-v1` at the source.
    #[test]
    fn a_dense_only_host_declares_the_nvfp4_source_and_a_dense_fallback_receipt() {
        let mut plan = mixed_plan();
        // The dense-only plan: nothing is priced packed and no scale stays resident.
        plan.tensors[0] = nvfp4_tensor("packed", ResidencyMode::Dense, 8192);
        plan.companions[0] = scale_companion("packed", 0);
        let receipt = LogicalWeightReceipt {
            mapping_id: MAPPING,
            tensor_count: 3,
            source_bytes: plan.source_bytes,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![
                row(
                    DENSE_BF16_CODEC.codec_id,
                    ExecutionRepresentation::DenseFallback,
                    1,
                    128,
                    128,
                ),
                row(
                    NVFP4_CODEC.codec_id,
                    ExecutionRepresentation::DenseFallback,
                    2,
                    2048 + 2048 + 256 + 256,
                    8192 + 8192,
                ),
            ],
        };
        let facts = CheckpointWeightFacts::new(
            &plan,
            NativeExecutionCapability::dense_only(),
            receipt.clone(),
        )
        .expect("a dense-only load is valid");
        assert!(facts.source().declares(NVFP4_CODEC.codec_id));
        assert!(!facts.executes_natively(NVFP4_CODEC.codec_id));
        assert!(facts.capability().is_dense_only());
        assert!(facts
            .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::NativePacked)
            .is_none());
    }

    /// **The mutation this contract exists to kill (1/2): label a dense receipt native.** The plan
    /// priced nothing packed and the host declares no capability, so both guards fire — the
    /// capability one first, because "this host cannot do that at all" is the stronger diagnosis.
    #[test]
    fn labelling_a_dense_fallback_receipt_native_refuses() {
        let mut plan = mixed_plan();
        plan.tensors[0] = nvfp4_tensor("packed", ResidencyMode::Dense, 8192);
        plan.companions[0] = scale_companion("packed", 0);
        let mutated = LogicalWeightReceipt {
            mapping_id: MAPPING,
            tensor_count: 1,
            source_bytes: 2048 + 256,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![row(
                NVFP4_CODEC.codec_id,
                // The lie: the bytes were decoded dense, the row says native.
                ExecutionRepresentation::NativePacked,
                1,
                2048 + 256,
                8192,
            )],
        };
        let error = CheckpointWeightFacts::new(
            &plan,
            NativeExecutionCapability::dense_only(),
            mutated.clone(),
        )
        .expect_err("a dense fallback cannot be labelled native");
        assert_eq!(
            error,
            CheckpointWeightFactsError::NativeWithoutCapability {
                codec_id: NVFP4_CODEC.codec_id,
                tensor_count: 1,
            }
        );

        // …and the plan-pricing guard is independently sufficient: even on a host that *does*
        // declare NVFP4 capability, a plan that priced no packed tensor has nothing native to
        // measure.
        let error = CheckpointWeightFacts::new(&plan, nvfp4_capability(), mutated)
            .expect_err("a plan that priced no packed row cannot yield a native receipt");
        assert_eq!(
            error,
            CheckpointWeightFactsError::RepresentationExceedsPlan {
                codec_id: NVFP4_CODEC.codec_id,
                representation: ExecutionRepresentation::NativePacked,
                reported_tensors: 1,
                planned_tensors: 0,
            }
        );
    }

    /// **The mutation this contract exists to kill (2/2): alias the source to another codec.**
    /// Re-labelling an NVFP4 row `int8-per-row-v1` (the shape a "this is really q4" product fact
    /// would take) is refused: a receipt reports how the source was materialized and never
    /// re-states what the source is.
    #[test]
    fn aliasing_the_source_codec_in_the_receipt_refuses() {
        let plan = mixed_plan();
        let mutated = LogicalWeightReceipt {
            mapping_id: MAPPING,
            tensor_count: 1,
            source_bytes: 2048 + 256,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![row(
                INT8_PER_ROW_CODEC.codec_id,
                ExecutionRepresentation::DenseFallback,
                1,
                2048 + 256,
                8192,
            )],
        };
        let error = CheckpointWeightFacts::new(&plan, nvfp4_capability(), mutated)
            .expect_err("the receipt cannot re-label the source codec");
        assert_eq!(
            error,
            CheckpointWeightFactsError::UnplannedCodec {
                codec_id: INT8_PER_ROW_CODEC.codec_id,
            }
        );
    }

    /// **E4, no double-counting.** A measured row above its planned residency is refused rather
    /// than passed on as a bigger footprint.
    #[test]
    fn a_measurement_above_the_plans_pricing_refuses() {
        let plan = mixed_plan();
        let mut receipt = honest_receipt();
        receipt.residency[2].resident_bytes += 1;
        let error = CheckpointWeightFacts::new(&plan, nvfp4_capability(), receipt)
            .expect_err("measured residency cannot exceed the plan");
        assert_eq!(
            error,
            CheckpointWeightFactsError::ResidentBytesExceedPlan {
                codec_id: NVFP4_CODEC.codec_id,
                representation: ExecutionRepresentation::NativePacked,
                measured_bytes: 2048 + 256 + 1,
                planned_bytes: 2048 + 256,
            }
        );
    }

    /// A partially materialized read is valid and simply `!is_complete()` — the receipt is honest
    /// about the instant, so the bounds are `<=`, not `==`.
    #[test]
    fn a_partial_read_is_valid_and_incomplete() {
        let plan = mixed_plan();
        let partial = LogicalWeightReceipt {
            mapping_id: MAPPING,
            tensor_count: 1,
            source_bytes: 2048 + 256,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![row(
                NVFP4_CODEC.codec_id,
                ExecutionRepresentation::NativePacked,
                1,
                2048 + 256,
                2048 + 256,
            )],
        };
        let facts = CheckpointWeightFacts::new(&plan, nvfp4_capability(), partial).expect("valid");
        assert!(!facts.is_complete());
        assert!(facts.executes_natively(NVFP4_CODEC.codec_id));
    }

    /// A receipt from a different mapping is not a receipt for this plan.
    #[test]
    fn a_receipt_from_another_mapping_refuses() {
        let plan = mixed_plan();
        let mut receipt = honest_receipt();
        receipt.mapping_id = "some-other-mapping-v1";
        let error = CheckpointWeightFacts::new(&plan, nvfp4_capability(), receipt)
            .expect_err("mapping ids must agree");
        assert_eq!(
            error,
            CheckpointWeightFactsError::MappingMismatch {
                plan_mapping_id: MAPPING,
                receipt_mapping_id: "some-other-mapping-v1",
            }
        );
    }

    /// A deferred read has evaluated nothing, so residency rows there would be invented.
    #[test]
    fn a_deferred_read_may_not_report_residency() {
        let plan = mixed_plan();
        let mut receipt = honest_receipt();
        receipt.materialization = LogicalReadMaterialization::Deferred;
        let error = CheckpointWeightFacts::new(&plan, nvfp4_capability(), receipt)
            .expect_err("a deferred read reports no residency");
        assert_eq!(
            error,
            CheckpointWeightFactsError::DeferredReadReportedResidency { rows: 3 }
        );
    }

    /// One stored tensor feeding several logical outputs is counted **once** in source bytes and
    /// several times in tensor/resident counts — the invariant a declarative fused-tensor transform
    /// (sc-21547) needs from this summary.
    #[test]
    fn a_fused_physical_tensor_contributes_its_source_bytes_once() {
        let mut fused_a = nvfp4_tensor("fused.q", ResidencyMode::Dense, 8192);
        fused_a.physical_key = "fused.qkv".to_owned();
        let mut fused_b = nvfp4_tensor("fused.k", ResidencyMode::Dense, 8192);
        fused_b.physical_key = "fused.qkv".to_owned();
        let plan = LogicalWeightPlan {
            mapping_id: MAPPING,
            tensors: vec![fused_a, fused_b],
            companions: vec![],
            source_bytes: 2048,
        };
        let entry = SourceCodecSummary::of(&plan)
            .entry(NVFP4_CODEC.codec_id)
            .expect("nvfp4 row")
            .clone();
        assert_eq!(entry.tensor_count, 2, "two logical outputs");
        assert_eq!(entry.source_bytes, 2048, "one physical tensor's bytes");
        assert_eq!(entry.planned_dense_resident_bytes, 8192 + 8192);
    }

    /// The wire labels SceneWorks renders are fixed by this test, not by whatever the enum's
    /// `Debug` happens to print.
    #[test]
    fn execution_representation_labels_are_the_wire_contract() {
        assert_eq!(
            ExecutionRepresentation::NativePacked.label(),
            "native-packed"
        );
        assert_eq!(
            ExecutionRepresentation::DenseFallback.label(),
            "dense-fallback"
        );
        assert_eq!(
            ExecutionRepresentation::from_residency(ResidencyMode::Packed),
            ExecutionRepresentation::NativePacked
        );
        assert_eq!(
            ExecutionRepresentation::from_residency(ResidencyMode::Dense),
            ExecutionRepresentation::DenseFallback
        );
    }
}

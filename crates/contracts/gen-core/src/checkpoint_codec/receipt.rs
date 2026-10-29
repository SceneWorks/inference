//! Residency reports and the logical weight receipt.

/// What one codec left resident after a read, **in one execution representation**.
///
/// A codec that materialized some tensors natively packed and others through its dense fallback —
/// the mixed-hardware NVFP4 case, where a padded or mis-aligned layer falls back on the same device
/// that runs its siblings packed — reports **two** rows, one per
/// [`ExecutionRepresentation`](crate::checkpoint_facts::ExecutionRepresentation). Collapsing them
/// into one total is exactly how a dense BF16 decode ends up labelled native (sc-21484).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecResidencyReport {
    pub codec_id: &'static str,
    /// The representation these tensors were actually materialized as.
    pub representation: crate::checkpoint_facts::ExecutionRepresentation,
    pub tensor_count: usize,
    /// Bytes read from the source file for these tensors (from the plan; companions included).
    pub source_bytes: u64,
    /// Bytes the decoded tensors occupy resident, measured from the backend arrays after decode
    /// (retained companions included).
    pub resident_bytes: u64,
}

/// Whether a read evaluated its arrays. A deferred read (block-streamed or bounded-quantized
/// loaders) leaves payloads lazy on purpose and must not report invented resident bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalReadMaterialization {
    /// Every planned tensor was evaluated; residency reports are measured.
    Materialized,
    /// **Some** planned tensors were evaluated; residency reports cover exactly those and no more
    /// (sc-11045 fix round). This is what a block-streamed load's front-only snapshot honestly is:
    /// labelling it [`Self::Materialized`] would claim "every planned tensor was evaluated" about a
    /// read that deliberately deferred its transformer blocks, and a consumer comparing the receipt
    /// against the plan's full pricing would then diagnose under-reporting where there is none.
    /// Planned-vs-measured residency is bounded (`<=`) here, never forced equal.
    Partial,
    /// Payloads remain lazy; residency reports are absent by construction.
    Deferred,
}

/// One logical row the provider **demoted** from the plan's `Packed` pricing to a dense-BF16
/// execution after construction settled its real regime (sc-11045 fix round, epic E5).
///
/// The plan prices residency from checkpoint geometry and device floors alone; a provider's role
/// table (or a transparent construction-time fallback — no fused quantizer, a staging failure) can
/// still serve a `Packed`-priced row as a dense BF16 weight (W4A16). This record is the **explicit,
/// typed accounting** that reconciles the receipt with the plan: the named row materialized
/// [`crate::checkpoint_facts::ExecutionRepresentation::DenseFallback`] holding `resident_bytes` of
/// dense weight instead of the plan's packed pricing. `CheckpointWeightFacts::new` validates every
/// demotion against the plan — an unknown key, a codec mismatch, or a demotion of a row the plan
/// never priced packed is a typed refusal, never a tolerance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegimeDemotion {
    /// The logical key of the demoted plan row.
    pub logical_key: String,
    /// The codec the plan stores that row in.
    pub codec_id: &'static str,
    /// Bytes the demoted row actually holds resident — the dense weight the fallback materialized,
    /// measured from the constructed layer, never copied from the plan.
    pub resident_bytes: u64,
}

/// The receipt a backend reader returns with the logical weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalWeightReceipt {
    pub mapping_id: &'static str,
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub materialization: LogicalReadMaterialization,
    /// One report per codec actually used; empty when deferred.
    pub residency: Vec<CodecResidencyReport>,
    /// Rows the provider demoted from the plan's `Packed` pricing to a dense-BF16 execution
    /// ([`RegimeDemotion`]); empty on a load whose construction honoured every packed pricing.
    pub demotions: Vec<RegimeDemotion>,
}

impl LogicalWeightReceipt {
    pub fn resident_bytes(&self) -> u64 {
        self.residency
            .iter()
            .map(|report| report.resident_bytes)
            .sum()
    }
}

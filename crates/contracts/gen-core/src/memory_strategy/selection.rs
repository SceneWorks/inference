//! The caller's numeric tier, selection, geometry, budget and mode vocabulary, and the rejection payload.

use super::*;

/// Numeric tier is one immutable axis of a selection. Strategies cannot change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryNumericTier {
    pub precision: Precision,
    pub quant: Option<Quant>,
    /// The provider-declared component floors active for this numeric tier. Part of evidence/cache
    /// identity: a uniform q4 run and a q4 run with q8 components are different numeric regimes.
    pub component_precision_floors: &'static [crate::ComponentPrecisionFloor],
}

/// Shared-worker selection presented to the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySelection {
    pub strategy: MemoryStrategy,
    pub parameters: MemoryStrategyParameters,
    pub tier: MemoryNumericTier,
}

/// Request geometry used by evidence and cache identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryGeometry {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub frames: u32,
    /// Number of reference images consumed by this provider forward pass. This is request
    /// geometry, not overlay identity; zero means the request has no reference input.
    pub reference_count: u32,
}

/// Shape of the reference input consumed by one provider request.
///
/// This is intentionally separate from [`MemoryGeometry::reference_count`].  A count says how
/// many references are present; it does not say whether the provider consumed images, a clip, a
/// mask, or another conditioning carrier.  Evidence for one carrier must never price another.
/// `Other` keeps the contract extensible without silently collapsing future shapes into one key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryReferenceShape {
    None,
    Image,
    Video,
    Mask,
    Other(String),
}

impl MemoryReferenceShape {
    /// Stable spelling at the persisted evidence/protocol boundary.
    pub fn as_key(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Image => "image",
            Self::Video => "video",
            Self::Mask => "mask",
            Self::Other(shape) => shape,
        }
    }

    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Canonical live-budget accounting owned by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// The platform-enforced capacity available to this process or worker, in bytes. This and every
    /// other budget field are one caller snapshot taken immediately before request admission.
    pub total_bytes: u64,
    /// Bytes already charged against [`total_bytes`](Self::total_bytes) at that snapshot, including
    /// unrelated live allocations and any request-owned resident assets loaded before admission.
    /// A caller must not subtract these bytes again from `total_bytes` before constructing the
    /// budget.
    pub committed_bytes: u64,
    /// Memory a backend is holding but could return under pressure — chiefly an allocator's
    /// freed-buffer reuse cache.
    ///
    /// **Creditable only if something will actually reclaim it before the platform's killer fires.**
    /// [`effective_bytes`](Self::effective_bytes) adds this back into the budget, which assumes a
    /// reclaimer exists. That assumption holds where the OS applies backpressure and fails where it
    /// terminates the process instead.
    ///
    /// iOS is the failing case, measured: jetsam reads `phys_footprint` — which counts an allocator's
    /// reuse cache in full — and does not ask anyone to release anything first. An MLX process there
    /// sat at `active + cache` pinned to the per-process cap across an entire decode, with ~3.9 GB
    /// cached and reclaimable, and was killed with 4 MiB of headroom. Every byte counted here was
    /// genuinely reclaimable and none of it was reclaimed, because nothing asked.
    ///
    /// What makes it creditable is bounding the allocator's own cache limit so the *allocator*
    /// reclaims continuously rather than waiting to be asked (`runtime_ios`'s
    /// `bound_mlx_to_platform_limits`). A caller that cannot name such a mechanism should pass `0`
    /// here rather than credit memory that will still be resident at the moment of the kill.
    /// Reclaimable bytes are a subset of `committed_bytes`, never a separate pool: the shared safety
    /// gate rejects `reclaimable_bytes > committed_bytes` rather than inventing capacity that was not
    /// first charged to the snapshot.
    pub reclaimable_bytes: u64,
    /// Capacity deliberately unavailable to the request, in the same process-residency currency as
    /// `total_bytes` and `committed_bytes`.
    pub reserved_headroom_bytes: u64,
}

impl MemoryBudget {
    /// Effective request budget. Reserved headroom is removed from the currently free plus
    /// reclaimable memory, while the total-minus-headroom ceiling prevents reclaimable accounting
    /// from exceeding physical capacity. Signed intermediate arithmetic preserves an existing
    /// overcommit deficit instead of saturating it away before reclamation is credited.
    pub fn effective_bytes(self) -> u64 {
        let ceiling = self
            .total_bytes
            .saturating_sub(self.reserved_headroom_bytes);
        let available = i128::from(self.total_bytes) - i128::from(self.committed_bytes)
            + i128::from(self.reclaimable_bytes)
            - i128::from(self.reserved_headroom_bytes);
        available.clamp(0, i128::from(ceiling)) as u64
    }

    /// Whether the request's **incremental** live demand fits; exact boundaries are accepted.
    ///
    /// `incremental_live_demand_bytes` is the additional live allocation above this budget's
    /// committed snapshot, not an absolute request high-water and not process footprint. Evidence
    /// records an absolute request live peak; before calling `fits`, the caller subtracts only the
    /// request-owned resident bytes already included in `committed_bytes`. Unrelated committed bytes
    /// are already charged by [`effective_bytes`](Self::effective_bytes) and must not be subtracted
    /// from the request peak.
    pub fn fits(self, incremental_live_demand_bytes: u64) -> bool {
        incremental_live_demand_bytes <= self.effective_bytes()
    }

    /// Total platform capacity required by this snapshot and an incremental request demand.
    ///
    /// This is the user-facing inverse of [`fits`](Self::fits): already committed bytes remain
    /// charged, valid reclaim credit is removed, then reserved headroom and new live demand are
    /// added. Saturation keeps an invalidly large input conservative.
    pub fn required_total_bytes(self, incremental_live_demand_bytes: u64) -> u64 {
        self.committed_bytes
            .saturating_sub(self.reclaimable_bytes.min(self.committed_bytes))
            .saturating_add(self.reserved_headroom_bytes)
            .saturating_add(incremental_live_demand_bytes)
    }
}

/// Cold/warm state is request-scoped and never implies reuse of a prior selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCacheState {
    Cold,
    Warm,
}

/// Provenance class under which the caller admitted the selected peak.
///
/// This is deliberately distinct from calibration identity. A conservative estimate is allowed to
/// drive an implemented optimization when a route has no model-memory calibration, but it must be
/// named explicitly; an absent calibration never turns an arbitrary hand-built optimized context
/// into an authorized estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryOptimizationAuthority {
    /// Resident baseline; no optimized claim is made.
    #[default]
    Resident,
    /// Exact/stale measured evidence that remains bound to the provider calibration handshake.
    Calibrated,
    /// Caller-synthesized conservative estimate admitted behind the estimate margin.
    Estimated,
}

/// Advertised request surface. Optimized evidence never transfers between these modes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryMode {
    TextToImage,
    ImageToImage,
    Edit,
    Other(String),
}

impl MemoryMode {
    /// Stable spelling at the persisted evidence/protocol boundary.
    pub fn as_key(&self) -> &str {
        match self {
            Self::TextToImage => "text_to_image",
            Self::ImageToImage => "image_to_image",
            Self::Edit => "edit",
            Self::Other(mode) => mode,
        }
    }
}

/// Truthful caller-facing pre-OOM rejection payload. The caller may populate a measured smaller
/// geometry; it must not invent advice from unknown or unverified evidence. The contract defines
/// this vocabulary even when an in-process provider does not construct it directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRejection {
    pub available_bytes: u64,
    pub minimum_verified_peak_bytes: Option<u64>,
    pub smaller_verified_geometry: Option<MemoryGeometry>,
    pub exclusion_reason: MemoryEvidenceVerdict,
}

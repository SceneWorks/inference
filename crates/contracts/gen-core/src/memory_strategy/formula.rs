//! Peak-memory formula shapes, window materialization, and per-backend realization.

use super::*;

/// Formula inputs whose coefficients live in manifest/generated evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryFormulaVariable {
    AssetBytes,
    PixelCount,
    LatentPixelCount,
    BatchCount,
    FrameCount,
    ConditioningTokenCount,
    OverlayBytes,
    DecodeTileArea,
    AttentionChunkSize,
    TransformerWindowSize,
}

/// Provider-owned shape of the peak-memory formula. Coefficients do not live here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryFormulaKind {
    /// Generic MLX-compatible seam: materialized asset bytes plus calibrated headroom.
    AssetBytesPlusHeadroom,
    /// One affine expression over the declared request/strategy variables.
    Affine {
        variables: Vec<MemoryFormulaVariable>,
    },
    /// Maximum of independently calibrated lifecycle phase expressions (the generalized Krea
    /// phase-curve shape, also suitable for request-aware provider estimators such as Mage).
    PhaseEnvelope {
        phases: Vec<MemoryPhase>,
        variables: Vec<MemoryFormulaVariable>,
    },
    /// A phase envelope which separately declares resident component contributions. This is a new
    /// formula shape instead of a field on [`MemoryAssetFacts`], so every existing provider keeps
    /// its source-compatible contract literal and adopts the component axis only by choosing this
    /// variant.
    ComponentPhaseEnvelope {
        phases: Vec<MemoryPhase>,
        variables: Vec<MemoryFormulaVariable>,
        resident_components: Vec<MemoryResidentComponent>,
    },
}

impl MemoryFormulaKind {
    /// Whether this formula consumes `variable`.
    pub fn uses(&self, variable: MemoryFormulaVariable) -> bool {
        match self {
            Self::AssetBytesPlusHeadroom => variable == MemoryFormulaVariable::AssetBytes,
            Self::Affine { variables }
            | Self::PhaseEnvelope { variables, .. }
            | Self::ComponentPhaseEnvelope { variables, .. } => variables.contains(&variable),
        }
    }

    pub(super) fn resident_components(&self) -> &[MemoryResidentComponent] {
        match self {
            Self::ComponentPhaseEnvelope {
                resident_components,
                ..
            } => resident_components,
            _ => &[],
        }
    }
}

/// What one rung-4 window materialization physically does on a realization (SC-16090).
///
/// This is the invariant that makes the ladder's shared cost order *true* rather than assumed. Rung 4
/// re-materializes the whole transformer once per **forward**, so a window's cost is multiplied by
/// `n_blocks × forwards`: 240 times for a 30-block stack over an 8-step schedule, and 480 where true
/// CFG runs a second forward per step. Whatever a window does is paid every one of those times, so only
/// work that is *irreducibly* per-window belongs there.
///
/// # Why this is declared rather than assumed
///
/// SC-15791 measured Candle's rung 4 at ~8.0 s/step against MLX's ~0.309 s on the identical file, and
/// ~100% of the ~26× is a host-side format conversion sitting in the per-window path. Per 97.1 MiB
/// block, medians on a host whose run-to-run spread reaches ±38%:
///
/// - **~204 ms** for the leg the spike labels *host read + repack* — a mapped read plus the MLX affine
///   triple → GGML `Q4_1` conversion. The read is inside that number and is not separated out; on a
///   cold page cache it is the part that dominates, since Candle re-reads the tier every step.
/// - **62.5 ms** attributable to a device→host round trip: the packed triple is loaded onto the CUDA
///   device and `repack::q4_parts` pulls all three straight back with `to_device(&Cpu)`. This figure is
///   a **residual** (266.8 production − 204.3 corrected), not an independently timed leg.
/// - The PCIe transfer itself: **unresolvable against a ±34.9 ms noise floor.** The only irreducibly
///   per-window part is a rounding error.
///
/// The conversion is a pure deterministic function of bytes that never change, recomputed once per
/// block per forward for one answer.
///
/// So the difference between the two backends is not device residency, PCIe, or unified memory. It is
/// that MLX's window materialization reads bytes the accelerator can consume as they sit on disk,
/// and Candle's does not — yet. Nothing about Candle prevents it: the conversion is
/// content-addressed by the source tensor and can be done once, ahead of the render, into a
/// device-format artifact a window then maps and copies.
///
/// # The obligation
///
/// A rung-4 realization SHOULD be [`Self::DeviceFormatTransfer`]. [`Self::HostFormatConversion`] is
/// admitted only as a **transitional** state that names the story removing it, because a rung whose
/// per-window path recomputes model-proportional work is not the rung the ladder costs.
///
/// # Where this lives, and why not on `MemoryStrategy`
///
/// SC-16090's brief was to layer additively on [`MemoryProviderContract`] "rather than patching the
/// bare enum". The prohibition is on [`MemoryStrategy`]: that enum is contract-owned precisely so no
/// provider can opt out of arithmetic, and a rung-4 field there would change the rung for MLX too. This
/// fact is the opposite kind — a property of one provider's loader — so it is declared on the
/// realization and read through [`MemoryProviderContract::window_materialization`], mirroring how
/// [`MemoryProviderContract::engages`] layers over [`MemoryStrategy::engages`].
///
/// It is a **required** field rather than an `Option` or a defaulted one, which does mean every existing
/// construction site had to change. That is the intended cost: a default is an opt-out that reads as a
/// declaration, and the whole point is that a Candle provider adopting rung 4 cannot stay silent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryWindowMaterialization {
    /// A window is a transfer of bytes **already in the form the accelerator consumes** — a mapped
    /// read plus, where the memory is not unified, a host-to-device copy. No format conversion, no
    /// device→host round trip, no re-derivation of anything. Cost is bounded by the transfer, so the
    /// rung costs what the ladder says it costs.
    ///
    /// MLX/Metal realizations satisfy this structurally: the mmap *is* the residency.
    DeviceFormatTransfer,
    /// A window performs a **host-side format conversion** whose cost is proportional to the block's
    /// weights, on every window of every step. Non-conforming, and admitted only transitionally.
    ///
    /// `converts` names the conversion in the terms the code uses, so a reviewer can find it.
    /// `owner_story` names the story that removes it — required, and the reason this variant is not a
    /// silent opt-out: an existing provider may keep shipping while a *new* one cannot quietly
    /// reproduce the defect without pointing at its fix.
    HostFormatConversion {
        converts: String,
        owner_story: String,
    },
}

impl MemoryWindowMaterialization {
    /// Whether this realization meets rung 4's per-window cost obligation.
    pub const fn is_conforming(&self) -> bool {
        matches!(self, Self::DeviceFormatTransfer)
    }
}

/// Backend-specific realization without imposing CUDA transfer language on unified memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryBackendRealization {
    CandleCuda {
        device_residency: bool,
        host_backed_weights: bool,
        host_to_device_block_materialization: bool,
        /// What one rung-4 window materialization does here (SC-16090). Stated per provider rather
        /// than per backend: it is a property of the loader a provider's streamed path calls, and two
        /// Candle providers can differ.
        ///
        /// Deliberately has **no default**. A defaulted or `Option` field would let a provider adopt
        /// rung 4 without answering, which is the silent state this exists to remove — so a new Candle
        /// provider cannot compile until it states one. What the contract cannot do is *verify* the
        /// answer: a provider that types `DeviceFormatTransfer` over a converting loader has made a
        /// false declaration, and nothing here catches that. The compile-forced choice buys a
        /// deliberate statement a reviewer can check against the loader, not a proof.
        block_materialization: MemoryWindowMaterialization,
    },
    MlxMetal {
        bounded_wired_residency: bool,
        lazy_or_mmap_materialization: bool,
        explicit_evaluation_and_synchronization: bool,
        cache_eviction: bool,
    },
}

/// Stable backend identity used by evidence and caller-side selection.
///
/// This deliberately excludes the realization capability flags above. Those flags describe one
/// loaded provider and are covered by its calibration identity; the evidence-key axis is the
/// canonical backend family carried by persisted calibration records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryBackend {
    Candle,
    Mlx,
}

impl MemoryBackend {
    /// Stable spelling at the persisted evidence/protocol boundary.
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Mlx => "mlx",
        }
    }
}

impl MemoryBackendRealization {
    pub const fn backend_kind(&self) -> MemoryBackend {
        match self {
            Self::CandleCuda { .. } => MemoryBackend::Candle,
            Self::MlxMetal { .. } => MemoryBackend::Mlx,
        }
    }

    pub const fn backend_id(&self) -> &'static str {
        self.backend_kind().as_key()
    }

    /// This realization's rung-4 window materialization (SC-16090) — the backend-neutral question,
    /// with a per-realization answer. This is the shape the contract layers everything backend-varying
    /// in: one shared question asked of every backend, not a forked ladder.
    ///
    /// `None` means **unstated**, which is a third condition and deliberately not folded into
    /// [`MemoryWindowMaterialization::HostFormatConversion`]: an MLX realization that declares neither
    /// lazy nor mmap materialization has not told us what its windows do, and encoding that as a
    /// *declared host conversion* would both assert something unmeasured and demand an `owner_story`
    /// field `MlxMetal` does not have — leaving the provider with an error whose only named fix is
    /// impossible. Its actual fix is to declare `lazy_or_mmap_materialization`.
    ///
    /// An MLX/Metal realization that does declare it answers
    /// [`MemoryWindowMaterialization::DeviceFormatTransfer`], because there the mapped file *is* the
    /// residency and there is no conversion seam to violate the obligation.
    pub fn window_materialization(&self) -> Option<&MemoryWindowMaterialization> {
        const MAPPED: &MemoryWindowMaterialization =
            &MemoryWindowMaterialization::DeviceFormatTransfer;
        match self {
            Self::CandleCuda {
                block_materialization,
                ..
            } => Some(block_materialization),
            Self::MlxMetal {
                lazy_or_mmap_materialization: true,
                ..
            } => Some(MAPPED),
            Self::MlxMetal { .. } => None,
        }
    }
}

/// Provider-owned calibration identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCalibrationIdentity {
    pub abi: u32,
    pub fingerprint: String,
    /// Exact intra-phase materialization shape covered by this calibration.
    pub load_shape: LoadShape,
}

impl MemoryCalibrationIdentity {
    pub fn new(fingerprint: impl Into<String>, load_shape: LoadShape) -> Self {
        Self {
            abi: MEMORY_CALIBRATION_ABI,
            fingerprint: fingerprint.into(),
            load_shape,
        }
    }
}

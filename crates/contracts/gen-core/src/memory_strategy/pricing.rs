//! Plan-based multi-phase / multi-stream live-set memory pricing (`MEM-3`, `§3.9`),
//! equivalent-only admission (`MEM-4`), scope identity lifecycle verification (`MEM-5`),
//! and the `B4` six-buffer `low_res_start` transition workspace / real-evidence posture (`§6.12`, `E4`, `SW-6941`).
//!
//! Nothing here executes tensor operations or invents a model-ID capability table:
//! concurrent streams and joint workspaces are summed (never maxed), shared weights and aliases
//! are deduplicated by canonical allocation/component ID, budget domains remain strictly separate
//! (`DeviceVram`, `HostRam`, `UnifiedMemory`), and whole-request high-water evidence copied across
//! phase fields keeps its whole-request basis so weights and already-included workspaces are never
//! double-counted.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Error, EstimateSource, FeatureClass, MemoryBudget, MemoryGeometry, MemoryPhase,
    MemoryRequestScope, MemoryRunOutcome, MemoryStrategy, RenderPlan, RenderRun,
    ResolvedLowResStartPlan, Result,
};

// -------------------------------------------------------------------------------------------------
// MEM-3 (§3.9): Budget Domains, Per-Phase / Per-Stream Geometry, Concurrent Live-Set Pricing,
// Alias Deduplication, and Aggregate Evidence Basis Classification
// -------------------------------------------------------------------------------------------------

/// Physical memory budget domain (`MEM-3`, `§3.9`).
///
/// Separate domains are never silently summed or converted into one another:
/// - `DeviceVram`: discrete accelerator memory (e.g., CUDA VRAM).
/// - `HostRam`: host system RAM (e.g., CPU-offloaded weights or pinned staging buffers).
/// - `UnifiedMemory`: Apple Silicon / unified pool where CPU and GPU allocations draw from one physical budget, counted once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryBudgetDomain {
    DeviceVram,
    HostRam,
    UnifiedMemory,
}

impl MemoryBudgetDomain {
    pub const ALL: [Self; 3] = [Self::DeviceVram, Self::HostRam, Self::UnifiedMemory];

    pub const fn as_key(self) -> &'static str {
        match self {
            Self::DeviceVram => "device_vram",
            Self::HostRam => "host_ram",
            Self::UnifiedMemory => "unified_memory",
        }
    }
}

/// Output materialization mode for a phase or stream (`MEM-3`, `§3.9`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StreamOutputMode {
    /// Whole output is buffered simultaneously in memory before handoff.
    #[default]
    Buffered,
    /// Output is emitted/consumed incrementally in bounded chunks or tiles.
    Streamed { chunk_bytes: u64 },
}

/// Per-stream, per-phase physical geometry (`MEM-3`, `§3.9`).
///
/// Captures spatial area (`width * height`), latent frame count, audio token/duration geometry,
/// batch size, reference count and per-reference pixel canvases, and output buffering mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamPhaseGeometry {
    pub stream_id: String,
    pub width: u32,
    pub height: u32,
    pub latent_frames: u32,
    pub audio_tokens: u64,
    pub audio_duration_millis: u64,
    pub batch: u32,
    pub reference_count: u32,
    pub reference_canvases: Vec<(u32, u32)>,
    pub output_mode: StreamOutputMode,
}

impl StreamPhaseGeometry {
    /// Construct an image-stream phase geometry.
    pub fn image(stream_id: impl Into<String>, width: u32, height: u32, batch: u32) -> Self {
        Self {
            stream_id: stream_id.into(),
            width,
            height,
            latent_frames: 1,
            audio_tokens: 0,
            audio_duration_millis: 0,
            batch,
            reference_count: 0,
            reference_canvases: Vec::new(),
            output_mode: StreamOutputMode::Buffered,
        }
    }

    /// Spatial pixel area (`width * height`).
    pub const fn pixel_area(&self) -> u64 {
        (self.width as u64).saturating_mul(self.height as u64)
    }

    /// Packed token count for a spatial `pixel_stride` (`(width / stride) * (height / stride) * latent_frames`).
    pub fn packed_tokens(&self, pixel_stride: u32) -> Result<u64> {
        if pixel_stride == 0 {
            return Err(Error::Unsupported(
                "StreamPhaseGeometry::packed_tokens: pixel_stride must be non-zero".into(),
            ));
        }
        let w = u64::from(self.width / pixel_stride);
        let h = u64::from(self.height / pixel_stride);
        let f = u64::from(self.latent_frames.max(1));
        w.checked_mul(h)
            .and_then(|area| area.checked_mul(f))
            .ok_or_else(|| Error::Unsupported("stream packed token count overflow".into()))
    }

    /// Total reference pixel area across all attached reference canvases.
    pub fn total_reference_pixel_area(&self) -> Result<u64> {
        self.reference_canvases
            .iter()
            .try_fold(0_u64, |acc, &(rw, rh)| {
                let area = u64::from(rw).checked_mul(u64::from(rh)).ok_or_else(|| {
                    Error::Unsupported("reference canvas pixel area overflow".into())
                })?;
                acc.checked_add(area)
                    .ok_or_else(|| Error::Unsupported("reference total pixel area overflow".into()))
            })
    }
}

/// A named weight, overlay, workspace, or buffer allocation in a specific [`MemoryBudgetDomain`] (`MEM-3`, `§3.9`).
///
/// Allocations that share an `alias_group` (e.g. a shared weight trunk or an in-place aliased view)
/// are deduplicated within each simultaneously live set so shared weights and aliases are counted
/// exactly once (`max` over the alias group if sizes differ, or the single shared size).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricedAllocation {
    pub id: String,
    pub alias_group: Option<String>,
    pub domain: MemoryBudgetDomain,
    pub bytes: u64,
}

impl PricedAllocation {
    pub fn new(id: impl Into<String>, domain: MemoryBudgetDomain, bytes: u64) -> Self {
        Self {
            id: id.into(),
            alias_group: None,
            domain,
            bytes,
        }
    }

    pub fn with_alias_group(mut self, alias_group: impl Into<String>) -> Self {
        self.alias_group = Some(alias_group.into());
        self
    }
}

/// Deduplicate and sum a set of simultaneously live [`PricedAllocation`]s in `domain` (`MEM-3`, `§3.9`).
///
/// - Allocations in other budget domains contribute `0` to `domain`.
/// - Allocations sharing the same `alias_group` (or the same `id` when `alias_group` is `None`)
///   are counted once at their shared byte footprint (`max` if views have different spans of the
///   same underlying storage).
/// - Distinct live allocations are summed with checked arithmetic.
pub fn sum_deduplicated_live_bytes<'a>(
    domain: MemoryBudgetDomain,
    allocations: impl IntoIterator<Item = &'a PricedAllocation>,
) -> Result<u64> {
    let mut by_key: BTreeMap<&str, u64> = BTreeMap::new();
    for alloc in allocations {
        if alloc.domain != domain {
            continue;
        }
        let key = alloc.alias_group.as_deref().unwrap_or(alloc.id.as_str());
        let entry = by_key.entry(key).or_insert(0);
        *entry = (*entry).max(alloc.bytes);
    }
    by_key.values().try_fold(0_u64, |acc, &bytes| {
        acc.checked_add(bytes)
            .ok_or_else(|| Error::Msg(format!("live-set byte sum overflow in {:?}", domain)))
    })
}

/// Classification of what a calibration or runtime evidence peak value actually includes (`MEM-3`, `§3.9`).
///
/// Prevents two classic pricing defects:
/// 1. Adding resident weights, ordinary latents, or stream totals a second time onto a `TotalPeak`
///    whole-request high-water mark that was copied into phase fields (`conditioning_peak_bytes`,
///    `denoise_peak_bytes`, `decode_peak_bytes`).
/// 2. Silently reinterpreting an ambiguous or unclassified evidence record when a new recipe
///    introduces additional live allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidencePeakBasis {
    /// Phase-scoped activation peak (`ActivationPeak`): records only the phase's transient
    /// activation/workspace demand above resident weights (unless flags state otherwise).
    ActivationPeak {
        includes_resident_weights: bool,
        includes_resident_overlays: bool,
        includes_ordinary_latents: bool,
        includes_concurrent_streams: bool,
        includes_joint_workspace: bool,
    },
    /// Whole-request live high-water mark (`TotalPeak`): even when copied across multiple phase
    /// fields in an evidence manifest, this value is a single request-wide high-water mark.
    TotalPeak {
        copied_across_phase_fields: bool,
        includes_resident_weights: bool,
        includes_resident_overlays: bool,
        includes_ordinary_latents: bool,
        includes_concurrent_streams: bool,
        includes_joint_workspace: bool,
    },
    /// Evidence basis is ambiguous or unverified for the requested decomposition; callers must
    /// follow the route's unpriced-evidence fail-closed posture rather than guessing.
    Ambiguous,
}

impl EvidencePeakBasis {
    /// Canonical whole-request high-water evidence copied across phase fields (such as the
    /// SC-15823 Candle `flux1_dev` q4 `1024x1024` T2I record where `17,651,074,970 B` is stored
    /// in `conditioning_peak_bytes`, `denoise_peak_bytes`, `decode_peak_bytes`, and
    /// `observed_peak_bytes`).
    pub const WHOLE_REQUEST_HIGH_WATER_COPIED_TO_PHASES: Self = Self::TotalPeak {
        copied_across_phase_fields: true,
        includes_resident_weights: true,
        includes_resident_overlays: true,
        includes_ordinary_latents: true,
        includes_concurrent_streams: true,
        includes_joint_workspace: true,
    };

    /// Pure phase-local activation workspace excluding resident weights and overlays.
    pub const PHASE_LOCAL_ACTIVATION_ONLY: Self = Self::ActivationPeak {
        includes_resident_weights: false,
        includes_resident_overlays: false,
        includes_ordinary_latents: true,
        includes_concurrent_streams: false,
        includes_joint_workspace: false,
    };

    /// Classify a 3-phase evidence tuple `(conditioning, denoise, decode, total_observed)`.
    ///
    /// When all present phase values equal `total_observed` (a non-zero whole-request high-water
    /// copied into each phase field), returns [`Self::WHOLE_REQUEST_HIGH_WATER_COPIED_TO_PHASES`]
    /// so phase envelope consumers never mistake it for three independent phase-local activation
    /// peaks or add resident weights a second time.
    pub const fn classify_phase_peaks(
        conditioning_peak: Option<u64>,
        denoise_peak: Option<u64>,
        decode_peak: Option<u64>,
        total_observed_peak: u64,
    ) -> Self {
        if total_observed_peak == 0 {
            return Self::Ambiguous;
        }
        let cond_eq = match conditioning_peak {
            Some(v) => v == total_observed_peak,
            None => true,
        };
        let den_eq = match denoise_peak {
            Some(v) => v == total_observed_peak,
            None => true,
        };
        let dec_eq = match decode_peak {
            Some(v) => v == total_observed_peak,
            None => true,
        };
        let any_present =
            conditioning_peak.is_some() || denoise_peak.is_some() || decode_peak.is_some();
        if any_present && cond_eq && den_eq && dec_eq {
            Self::WHOLE_REQUEST_HIGH_WATER_COPIED_TO_PHASES
        } else if any_present {
            Self::ActivationPeak {
                includes_resident_weights: true,
                includes_resident_overlays: true,
                includes_ordinary_latents: true,
                includes_concurrent_streams: true,
                includes_joint_workspace: true,
            }
        } else {
            Self::TotalPeak {
                copied_across_phase_fields: false,
                includes_resident_weights: true,
                includes_resident_overlays: true,
                includes_ordinary_latents: true,
                includes_concurrent_streams: true,
                includes_joint_workspace: true,
            }
        }
    }

    pub const fn is_whole_request_total(self) -> bool {
        matches!(self, Self::TotalPeak { .. })
    }

    pub const fn includes_resident_weights(self) -> Option<bool> {
        match self {
            Self::ActivationPeak {
                includes_resident_weights,
                ..
            }
            | Self::TotalPeak {
                includes_resident_weights,
                ..
            } => Some(includes_resident_weights),
            Self::Ambiguous => None,
        }
    }

    pub const fn includes_resident_overlays(self) -> Option<bool> {
        match self {
            Self::ActivationPeak {
                includes_resident_overlays,
                ..
            }
            | Self::TotalPeak {
                includes_resident_overlays,
                ..
            } => Some(includes_resident_overlays),
            Self::Ambiguous => None,
        }
    }

    pub const fn includes_concurrent_streams(self) -> Option<bool> {
        match self {
            Self::ActivationPeak {
                includes_concurrent_streams,
                ..
            }
            | Self::TotalPeak {
                includes_concurrent_streams,
                ..
            } => Some(includes_concurrent_streams),
            Self::Ambiguous => None,
        }
    }

    pub const fn includes_joint_workspace(self) -> Option<bool> {
        match self {
            Self::ActivationPeak {
                includes_joint_workspace,
                ..
            }
            | Self::TotalPeak {
                includes_joint_workspace,
                ..
            } => Some(includes_joint_workspace),
            Self::Ambiguous => None,
        }
    }
}

/// One phase operation `op` in a planned render (`MEM-3`, `§3.9`):
///
/// `operation_peak(op) = resident_weights(op) + resident_overlays(op) + activation_peak(op) + retained_state(op)`
/// where concurrent stream activations and any joint cross-stream workspace live during `op` are
/// **summed, never maxed**, and aliased/shared allocations are counted once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseOperationLiveSet {
    pub operation_id: String,
    pub phase: MemoryPhase,
    pub pass_index: usize,
    pub phase_index: usize,
    pub stream_geometries: Vec<StreamPhaseGeometry>,
    pub resident_weights: Vec<PricedAllocation>,
    pub resident_overlays: Vec<PricedAllocation>,
    /// Per-stream activation peaks live concurrently during this operation (summed across streams).
    pub concurrent_stream_activations: Vec<PricedAllocation>,
    /// Joint workspace (e.g. joint audio-video attention or conditioning projection) live alongside the streams.
    pub joint_workspaces: Vec<PricedAllocation>,
    /// Retained state kept live across steps/phases during this operation (e.g. KV cache, solver history, reference latents).
    pub retained_state: Vec<PricedAllocation>,
    /// Optional aggregate evidence peak and its basis when pricing from a calibrated record.
    pub evidence_peak: Option<(MemoryBudgetDomain, u64, EvidencePeakBasis)>,
}

impl PhaseOperationLiveSet {
    /// Compute `operation_peak(op)` in `domain` (`MEM-3`, `§3.9`).
    ///
    /// - Refuses `EvidencePeakBasis::Ambiguous` with a typed [`Error::Unsupported`].
    /// - Never double-adds weights, overlays, stream totals, or joint workspace that the evidence
    ///   basis declares are already included.
    /// - Sums concurrent streams + joint workspace + resident weights + resident overlays + retained state,
    ///   deduplicating shared alias groups.
    pub fn operation_peak_bytes(&self, domain: MemoryBudgetDomain) -> Result<u64> {
        if let Some((ev_domain, ev_peak, basis)) = self.evidence_peak {
            if ev_domain == domain {
                return match basis {
                    EvidencePeakBasis::Ambiguous => Err(Error::Unsupported(format!(
                        "operation '{}': evidence peak basis is Ambiguous in {:?}; refusing to guess decomposition",
                        self.operation_id, domain
                    ))),
                    EvidencePeakBasis::ActivationPeak {
                        includes_resident_weights,
                        includes_resident_overlays,
                        includes_ordinary_latents: _,
                        includes_concurrent_streams,
                        includes_joint_workspace,
                    }
                    | EvidencePeakBasis::TotalPeak {
                        copied_across_phase_fields: _,
                        includes_resident_weights,
                        includes_resident_overlays,
                        includes_ordinary_latents: _,
                        includes_concurrent_streams,
                        includes_joint_workspace,
                    } => {
                        let mut extra = Vec::new();
                        if !includes_resident_weights {
                            extra.extend(self.resident_weights.iter());
                        }
                        if !includes_resident_overlays {
                            extra.extend(self.resident_overlays.iter());
                        }
                        if !includes_concurrent_streams {
                            extra.extend(self.concurrent_stream_activations.iter());
                        }
                        if !includes_joint_workspace {
                            extra.extend(self.joint_workspaces.iter());
                        }
                        extra.extend(self.retained_state.iter());
                        let extra_bytes = sum_deduplicated_live_bytes(domain, extra)?;
                        ev_peak.checked_add(extra_bytes).ok_or_else(|| {
                            Error::Msg(format!(
                                "operation '{}' peak overflow in {:?}",
                                self.operation_id, domain
                            ))
                        })
                    }
                };
            }
        }

        let all = self
            .resident_weights
            .iter()
            .chain(self.resident_overlays.iter())
            .chain(self.concurrent_stream_activations.iter())
            .chain(self.joint_workspaces.iter())
            .chain(self.retained_state.iter());
        sum_deduplicated_live_bytes(domain, all)
    }
}

/// What kind of transition mechanism executes at a phase or pass boundary (`MEM-3`, `§3.9`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionMechanismKind {
    /// Parameter-free latent-only spatial unpack/resize/repack/renoise transition (`low_res_start`).
    LatentSpatialTransition,
    /// Learned-component transition (e.g., VAE decode/encode round-trip, learned latent upsampler,
    /// or endpoint denoiser evaluation). Must price the learned component's execution workspace
    /// above latent-only buffer deltas!
    LearnedComponentTransition {
        /// Minimum non-zero workspace floor required for the learned component evaluation.
        learned_component_workspace_bytes: u64,
    },
}

/// One transition `t` between neighboring phases or passes (`MEM-3`, `§3.9`):
///
/// `transition_peak(t) = resident_weights(t) + resident_overlays(t) + transition_workspace_peak(t) + live_handoff_buffers(t)`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionOperationLiveSet {
    pub transition_id: String,
    pub from_operation_id: String,
    pub to_operation_id: String,
    pub mechanism: TransitionMechanismKind,
    pub resident_weights: Vec<PricedAllocation>,
    pub resident_overlays: Vec<PricedAllocation>,
    /// Workspace of the component or evaluation actually executing at the transition (e.g., endpoint forward or VAE decode).
    pub transition_workspaces: Vec<PricedAllocation>,
    ///Simultaneously live handoff buffers at the boundary (e.g., the 6 fp32 transition buffers + retained conditioning).
    pub live_handoff_buffers: Vec<PricedAllocation>,
}

impl TransitionOperationLiveSet {
    /// Compute `transition_peak(t)` in `domain` (`MEM-3`, `§3.9`).
    ///
    /// For a [`TransitionMechanismKind::LearnedComponentTransition`], enforces that the priced
    /// transition workspace strictly exceeds latent-only handoff buffers by at least the learned
    /// component workspace floor.
    pub fn transition_peak_bytes(&self, domain: MemoryBudgetDomain) -> Result<u64> {
        let workspace_only =
            sum_deduplicated_live_bytes(domain, self.transition_workspaces.iter())?;
        if let TransitionMechanismKind::LearnedComponentTransition {
            learned_component_workspace_bytes,
        } = self.mechanism
        {
            let device_or_unified = domain == MemoryBudgetDomain::DeviceVram
                || domain == MemoryBudgetDomain::UnifiedMemory;
            if device_or_unified
                && (learned_component_workspace_bytes == 0
                    || workspace_only < learned_component_workspace_bytes)
            {
                return Err(Error::Unsupported(format!(
                    "transition '{}': learned transition workspace ({workspace_only} B) must include learned component workspace (>= {learned_component_workspace_bytes} B > 0) above latent-only handoff deltas",
                    self.transition_id
                )));
            }
        }
        let all = self
            .resident_weights
            .iter()
            .chain(self.resident_overlays.iter())
            .chain(self.transition_workspaces.iter())
            .chain(self.live_handoff_buffers.iter());
        sum_deduplicated_live_bytes(domain, all)
    }
}

/// Complete multi-phase, multi-stream live-set memory plan for a request (`MEM-3`, `§3.9`).
///
/// `request_peak(domain) = max(max_op operation_peak(op, domain), max_t transition_peak(t, domain))`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequestLiveSetPlan {
    pub operations: Vec<PhaseOperationLiveSet>,
    pub transitions: Vec<TransitionOperationLiveSet>,
}

/// Priced request peak breakdown for one [`MemoryBudgetDomain`] (`MEM-3`, `§3.9`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricedDomainPeak {
    pub domain: MemoryBudgetDomain,
    pub max_operation_peak_bytes: u64,
    pub max_transition_peak_bytes: u64,
    pub request_peak_bytes: u64,
    pub dominant_stage_id: String,
}

impl RequestLiveSetPlan {
    /// Price the request peak in `domain`:
    /// `request_peak = max(max_op operation_peak(op), max_t transition_peak(t))`.
    ///
    /// Verifies that every transition references valid neighboring operations in `self.operations`
    /// so a request bound always covers both neighboring phases and each transition operation.
    pub fn price_domain(&self, domain: MemoryBudgetDomain) -> Result<PricedDomainPeak> {
        if self.operations.is_empty() {
            return Err(Error::Unsupported(
                "RequestLiveSetPlan::price_domain: plan must contain at least one phase operation"
                    .into(),
            ));
        }
        let op_ids: BTreeSet<&str> = self
            .operations
            .iter()
            .map(|op| op.operation_id.as_str())
            .collect();
        for transition in &self.transitions {
            if !op_ids.contains(transition.from_operation_id.as_str())
                || !op_ids.contains(transition.to_operation_id.as_str())
            {
                return Err(Error::Unsupported(format!(
                    "transition '{}' references unknown neighboring operation(s) '{}' -> '{}'",
                    transition.transition_id,
                    transition.from_operation_id,
                    transition.to_operation_id
                )));
            }
        }

        let mut max_op = 0_u64;
        let mut max_trans = 0_u64;
        let mut dominant = self.operations[0].operation_id.clone();
        let mut best_overall = 0_u64;

        for op in &self.operations {
            let peak = op.operation_peak_bytes(domain)?;
            if peak > max_op {
                max_op = peak;
            }
            if peak >= best_overall {
                best_overall = peak;
                dominant = op.operation_id.clone();
            }
        }

        for tr in &self.transitions {
            let peak = tr.transition_peak_bytes(domain)?;
            if peak > max_trans {
                max_trans = peak;
            }
            if peak > best_overall {
                best_overall = peak;
                dominant = tr.transition_id.clone();
            }
        }

        Ok(PricedDomainPeak {
            domain,
            max_operation_peak_bytes: max_op,
            max_transition_peak_bytes: max_trans,
            request_peak_bytes: max_op.max(max_trans),
            dominant_stage_id: dominant,
        })
    }

    /// Price all budget domains that appear in the plan, keeping `DeviceVram`, `HostRam`, and
    /// `UnifiedMemory` strictly separate (`MEM-3`, `§3.9`).
    pub fn price_all_domains(&self) -> Result<BTreeMap<MemoryBudgetDomain, PricedDomainPeak>> {
        let mut map = BTreeMap::new();
        for domain in MemoryBudgetDomain::ALL {
            let peak = self.price_domain(domain)?;
            if peak.request_peak_bytes > 0 {
                map.insert(domain, peak);
            }
        }
        Ok(map)
    }
}

// -------------------------------------------------------------------------------------------------
// MEM-4: Admission Equivalent-Only Rule
// -------------------------------------------------------------------------------------------------

/// One candidate adjustment considered by memory admission to make a request fit (`MEM-4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionAdjustmentCandidate {
    pub id: String,
    pub feature_class: FeatureClass,
    pub strategy: MemoryStrategy,
    /// Whether this candidate would alter the loaded or requested quantization tier.
    pub changes_quantization_tier: bool,
    /// Whether this candidate adds or alters trajectory passes/steps (e.g. `low_res_start`, `hires_fix`).
    pub alters_recipe_trajectory: bool,
    /// Resulting incremental peak demand in bytes if applied.
    pub incremental_peak_bytes: u64,
}

/// Typed refusal reason when equivalent-only admission cannot fit a request (`MEM-4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EquivalentAdmissionRefusal {
    /// A candidate was rejected because it is not `FeatureClass::Equivalent` or changes quantization tier.
    ForbiddenNonEquivalentAdjustment {
        candidate_id: String,
        feature_class: FeatureClass,
        changes_quantization_tier: bool,
        alters_recipe_trajectory: bool,
    },
    /// Every legal `FeatureClass::Equivalent` adjustment still exceeds the effective budget.
    InsufficientEquivalentAdjustments {
        effective_budget_bytes: u64,
        lowest_equivalent_peak_bytes: Option<u64>,
        forbidden_candidates_skipped: Vec<String>,
    },
}

impl std::fmt::Display for EquivalentAdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForbiddenNonEquivalentAdjustment {
                candidate_id,
                feature_class,
                changes_quantization_tier,
                alters_recipe_trajectory,
            } => write!(
                f,
                "admission candidate '{candidate_id}' is forbidden by MEM-4 (class={}, changes_quantization_tier={changes_quantization_tier}, alters_recipe_trajectory={alters_recipe_trajectory}); admission may only select FeatureClass::Equivalent adjustments",
                feature_class.as_str()
            ),
            Self::InsufficientEquivalentAdjustments {
                effective_budget_bytes,
                lowest_equivalent_peak_bytes,
                forbidden_candidates_skipped,
            } => write!(
                f,
                "equivalent-only admission adjustments are insufficient for effective budget {effective_budget_bytes} B (lowest equivalent peak={lowest_equivalent_peak_bytes:?}, skipped non-equivalent candidates={forbidden_candidates_skipped:?})"
            ),
        }
    }
}

/// Validate that an explicit admission adjustment satisfies `MEM-4`:
/// - `feature_class == FeatureClass::Equivalent`
/// - `!changes_quantization_tier`
/// - `!alters_recipe_trajectory`
pub fn validate_equivalent_admission_candidate(
    candidate: &AdmissionAdjustmentCandidate,
) -> std::result::Result<(), EquivalentAdmissionRefusal> {
    if candidate.feature_class != FeatureClass::Equivalent
        || candidate.changes_quantization_tier
        || candidate.alters_recipe_trajectory
    {
        return Err(
            EquivalentAdmissionRefusal::ForbiddenNonEquivalentAdjustment {
                candidate_id: candidate.id.clone(),
                feature_class: candidate.feature_class,
                changes_quantization_tier: candidate.changes_quantization_tier,
                alters_recipe_trajectory: candidate.alters_recipe_trajectory,
            },
        );
    }
    Ok(())
}

/// Select the first (least-cost) `FeatureClass::Equivalent` adjustment that fits `budget` (`MEM-4`).
///
/// - Never selects a `FeatureClass::Recipe` candidate (such as `low_res_start` or adding passes),
///   even if its peak would fit `budget`.
/// - Never selects a `FeatureClass::Diagnostic` candidate.
/// - Never selects a candidate that changes the quantization tier (`changes_quantization_tier == true`).
/// - If no legal `FeatureClass::Equivalent` candidate fits, refuses with a typed
///   [`EquivalentAdmissionRefusal::InsufficientEquivalentAdjustments`].
pub fn select_equivalent_only_admission(
    budget: MemoryBudget,
    candidates: &[AdmissionAdjustmentCandidate],
) -> std::result::Result<&AdmissionAdjustmentCandidate, EquivalentAdmissionRefusal> {
    let mut lowest_equivalent_peak: Option<u64> = None;
    let mut forbidden_skipped = Vec::new();

    for candidate in candidates {
        if validate_equivalent_admission_candidate(candidate).is_err() {
            forbidden_skipped.push(candidate.id.clone());
            continue;
        }
        lowest_equivalent_peak = Some(match lowest_equivalent_peak {
            Some(prev) => prev.min(candidate.incremental_peak_bytes),
            None => candidate.incremental_peak_bytes,
        });
        if budget.fits(candidate.incremental_peak_bytes) {
            return Ok(candidate);
        }
    }

    Err(
        EquivalentAdmissionRefusal::InsufficientEquivalentAdjustments {
            effective_budget_bytes: budget.effective_bytes(),
            lowest_equivalent_peak_bytes: lowest_equivalent_peak,
            forbidden_candidates_skipped: forbidden_skipped,
        },
    )
}

// -------------------------------------------------------------------------------------------------
// B4 Closure Witness (§6.12, E4, SW-6941): Six-Buffer Transition Ceiling, Exact-Fit / One-Byte-Under
// Fixture, and Fail-Closed Real-Evidence Posture
// -------------------------------------------------------------------------------------------------

/// Exact measured SC-15823 Candle `flux1_dev` q4 `1024x1024` T2I (batch 1, 8 steps, seed 42,
/// `Resident`) ordinary peak in bytes (`17,651,074,970 B`).
pub const CANDLE_FLUX1_DEV_Q4_1024_ORDINARY_PEAK_BYTES: u64 = 17_651_074_970;

/// Reduced fp32 latent buffer bytes at `512x512` (`1 x 16 x 64 x 64 x 4 B = 262,144 B`).
pub const FLUX1_LOW_RES_512_FP32_LATENT_BYTES: u64 = 262_144;

/// Full fp32 latent buffer bytes at `1024x1024` (`1 x 16 x 128 x 128 x 4 B = 1,048,576 B`).
pub const FLUX1_FULL_RES_1024_FP32_LATENT_BYTES: u64 = 1_048_576;

/// Six-buffer fp32 transition workspace ceiling at `512x512 -> 1024x1024` (`2 * 262,144 + 4 * 1,048,576 = 4,718,592 B`, `4.5 MiB`).
pub const FLUX1_LOW_RES_1024_SIX_BUFFER_CEILING_BYTES: u64 = 4_718_592;

/// Exact conditional peak for SC-15823 Candle `flux1_dev` q4 `1024x1024` + 6-buffer transition ceiling
/// (`17,651,074,970 + 4,718,592 = 17,655,793,562 B`).
pub const CANDLE_FLUX1_DEV_Q4_1024_CONDITIONAL_LOW_RES_PEAK_BYTES: u64 =
    CANDLE_FLUX1_DEV_Q4_1024_ORDINARY_PEAK_BYTES + FLUX1_LOW_RES_1024_SIX_BUFFER_CEILING_BYTES;

/// Exact breakdown of the six simultaneously live fp32 transition buffers for `low_res_start` (`§6.12`, `B4`).
///
/// The six buffers are:
/// 1. `reduced_packed_source_bytes`: reduced packed source/clean estimate in fp32 (`B * C * H_low * W_low * 4`)
/// 2. `reduced_unpacked_clean_bytes`: reduced unpacked `[B, C, H_low, W_low]` clean estimate in fp32
/// 3. `full_resized_clean_bytes`: full-resolution `[B, C, H_full, W_full]` bilinear-resized clean estimate in fp32
/// 4. `full_transition_noise_bytes`: independent full-resolution `[B, C, H_full, W_full]` Gaussian transition noise in fp32
/// 5. `full_re_noised_blend_bytes`: full-resolution `[B, C, H_full, W_full]` blended `x(sigma_K)` in fp32
/// 6. `full_repacked_destination_bytes`: full-resolution packed destination `[B, T_full, C * P^2]` in fp32 (before/during cast)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowResTransitionSixBufferCeiling {
    pub batch: u32,
    pub latent_channels: u32,
    pub vae_spatial_scale: u32,
    pub low_width: u32,
    pub low_height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub reduced_latent_fp32_bytes: u64,
    pub full_latent_fp32_bytes: u64,
    pub reduced_packed_source_bytes: u64,
    pub reduced_unpacked_clean_bytes: u64,
    pub full_resized_clean_bytes: u64,
    pub full_transition_noise_bytes: u64,
    pub full_re_noised_blend_bytes: u64,
    pub full_repacked_destination_bytes: u64,
    pub total_six_buffer_ceiling_bytes: u64,
}

impl LowResTransitionSixBufferCeiling {
    /// Compute the exact six-buffer fp32 transition workspace ceiling for a resolved `low_res_start` plan (`§6.12`, `B4`).
    pub fn compute(
        plan: &ResolvedLowResStartPlan,
        batch: u32,
        latent_channels: u32,
        vae_spatial_scale: u32,
    ) -> Result<Self> {
        if batch == 0 || latent_channels == 0 || vae_spatial_scale == 0 {
            return Err(Error::Unsupported(
                "LowResTransitionSixBufferCeiling::compute: batch, latent_channels, and vae_spatial_scale must be non-zero".into(),
            ));
        }
        if !plan.low_width.is_multiple_of(vae_spatial_scale)
            || !plan.low_height.is_multiple_of(vae_spatial_scale)
            || !plan.full_width.is_multiple_of(vae_spatial_scale)
            || !plan.full_height.is_multiple_of(vae_spatial_scale)
        {
            return Err(Error::Unsupported(
                "LowResTransitionSixBufferCeiling::compute: dimensions must be multiples of vae_spatial_scale".into(),
            ));
        }
        let low_lat_w = u64::from(plan.low_width / vae_spatial_scale);
        let low_lat_h = u64::from(plan.low_height / vae_spatial_scale);
        let full_lat_w = u64::from(plan.full_width / vae_spatial_scale);
        let full_lat_h = u64::from(plan.full_height / vae_spatial_scale);
        let b_c = u64::from(batch)
            .checked_mul(u64::from(latent_channels))
            .ok_or_else(|| Error::Msg("six-buffer batch*channels overflow".into()))?;

        let reduced_latent_fp32_bytes = b_c
            .checked_mul(low_lat_h)
            .and_then(|v| v.checked_mul(low_lat_w))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Msg("six-buffer reduced latent byte overflow".into()))?;

        let full_latent_fp32_bytes = b_c
            .checked_mul(full_lat_h)
            .and_then(|v| v.checked_mul(full_lat_w))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Msg("six-buffer full latent byte overflow".into()))?;

        let two_reduced = reduced_latent_fp32_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Msg("six-buffer 2*reduced overflow".into()))?;
        let four_full = full_latent_fp32_bytes
            .checked_mul(4)
            .ok_or_else(|| Error::Msg("six-buffer 4*full overflow".into()))?;
        let total_six_buffer_ceiling_bytes = two_reduced
            .checked_add(four_full)
            .ok_or_else(|| Error::Msg("six-buffer total ceiling overflow".into()))?;

        Ok(Self {
            batch,
            latent_channels,
            vae_spatial_scale,
            low_width: plan.low_width,
            low_height: plan.low_height,
            full_width: plan.full_width,
            full_height: plan.full_height,
            reduced_latent_fp32_bytes,
            full_latent_fp32_bytes,
            reduced_packed_source_bytes: reduced_latent_fp32_bytes,
            reduced_unpacked_clean_bytes: reduced_latent_fp32_bytes,
            full_resized_clean_bytes: full_latent_fp32_bytes,
            full_transition_noise_bytes: full_latent_fp32_bytes,
            full_re_noised_blend_bytes: full_latent_fp32_bytes,
            full_repacked_destination_bytes: full_latent_fp32_bytes,
            total_six_buffer_ceiling_bytes,
        })
    }
}

/// Trace of the four required increments that must be covered to price a `low_res_start` transition
/// on top of an ordinary single-phase T2I peak (`§6.12`, `E4`, `SW-6941`):
/// 1. `small_grid_forward_covered_by_ordinary_peak`: whether phase-0's forward at `(low_w, low_h)` is
///    provably `<=` the full-grid denoise forward peak covered by `ordinary_t2i_peak_bytes` (by
///    monotonic attention/MLP token area `low_tokens < full_tokens`).
/// 2. `retained_phase1_context_bytes`: extra retained buffers kept live across phase 0 / transition /
///    phase 1 (e.g. phase-1 positional IDs `img_ids` and `PreparedDit` conditioning when phase 0's
///    prepared conditioning is also retained).
/// 3. `solver_snapshot_handoff_bytes`: any solver clean-estimate snapshot or boundary state retained
///    at step `K` (`PreviousEvaluation` cache or `x_sigma_k` input alongside `x0_low`).
/// 4. `endpoint_evaluation_workspace_covered_by_ordinary_peak`: whether the read-only endpoint
///    evaluation at `sigma_K` (when `EstimateSource::EndpointEvaluation` is active) executes at
///    `(low_w, low_h)` before the 6-buffer fp32 resize/renoise allocations and is therefore bounded
///    within the ordinary full-grid forward peak.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowResTransitionCoverageTrace {
    pub ordinary_t2i_peak_bytes: u64,
    pub evidence_basis: EvidencePeakBasis,
    pub estimate_source: EstimateSource,
    pub small_grid_forward_covered_by_ordinary_peak: bool,
    pub endpoint_evaluation_workspace_covered_by_ordinary_peak: bool,
    pub retained_phase1_context_bytes: u64,
    pub solver_snapshot_handoff_bytes: u64,
    pub six_buffer_ceiling: LowResTransitionSixBufferCeiling,
}

impl LowResTransitionCoverageTrace {
    /// Total incremental bytes above `ordinary_t2i_peak_bytes` when both the small-grid forward and
    /// any read-only endpoint evaluation workspace are covered by `ordinary_t2i_peak_bytes`.
    pub fn total_incremental_bytes_above_ordinary(&self) -> Result<u64> {
        if !self.small_grid_forward_covered_by_ordinary_peak {
            return Err(Error::Unsupported(
                "low_res_start small-grid forward is not proven covered by the ordinary full-grid peak".into(),
            ));
        }
        if self.estimate_source == EstimateSource::EndpointEvaluation
            && !self.endpoint_evaluation_workspace_covered_by_ordinary_peak
        {
            return Err(Error::Unsupported(
                "low_res_start read-only endpoint evaluation workspace is not proven covered by the ordinary full-grid peak".into(),
            ));
        }
        self.six_buffer_ceiling
            .total_six_buffer_ceiling_bytes
            .checked_add(self.retained_phase1_context_bytes)
            .and_then(|v| v.checked_add(self.solver_snapshot_handoff_bytes))
            .ok_or_else(|| Error::Msg("low_res_start incremental byte bound overflow".into()))
    }

    /// Total bounded request peak (`ordinary_t2i_peak_bytes + total_incremental_bytes_above_ordinary()`).
    pub fn total_bounded_peak_bytes(&self) -> Result<u64> {
        let delta = self.total_incremental_bytes_above_ordinary()?;
        self.ordinary_t2i_peak_bytes
            .checked_add(delta)
            .ok_or_else(|| Error::Msg("low_res_start total bounded peak overflow".into()))
    }
}

/// Posture for pricing a `low_res_start` request against an ordinary T2I evidence cell (`§6.12`, `E4`, `SW-6941`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowResTransitionPricingPosture {
    /// Conditional / bounded-increment mode (`§6.12` exact-fit / one-byte-under witness):
    /// ordinary peak coverage, retained buffers, snapshot, and endpoint workspace are bounded within
    /// `ordinary_t2i_peak_bytes + six_buffer_ceiling_bytes` (with zero unpriced retained delta).
    ConditionalSixBufferIncrement,
    /// Real-evidence posture (`§6.12` fail-closed rule):
    /// when the evidence cell only records an unclassified whole-request high-water mark (`TotalPeak`
    /// copied across phase fields) and no [`LowResTransitionCoverageTrace`] has qualified the cell's
    /// additional live increments (small-grid forward, retained state, snapshot, endpoint workspace),
    /// admission **refuses `low_res_start` with a typed reason before loading** while keeping
    /// ordinary T2I admission completely unchanged.
    UnqualifiedRealEvidence,
    /// Qualified conservative upper bound covering the small-grid forward, retained phase-1 context,
    /// solver snapshot handoff, endpoint evaluation workspace, and the 6-buffer transition ceiling.
    QualifiedConservativeBound(LowResTransitionCoverageTrace),
}

/// Typed refusal reason when `low_res_start` admission is refused before loading (`§6.12`, `E4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowResTransitionAdmissionRefusal {
    /// The cell's evidence only records a whole-request high-water mark (`TotalPeak` copied across
    /// phases) and the new recipe's additional live increments (small-grid forward, retained buffers,
    /// solver snapshot, endpoint evaluation workspace) are unclassified/unpriced for this cell.
    UnpricedTransitionIncrementsOnWholeRequestEvidence {
        route_id: String,
        ordinary_t2i_peak_bytes: u64,
        six_buffer_ceiling_bytes: u64,
        evidence_basis: EvidencePeakBasis,
    },
    /// A coverage trace was supplied, but either the small-grid forward or the endpoint evaluation
    /// workspace was not covered by the ordinary peak.
    IncompleteCoverageProof { route_id: String, reason: String },
    /// The priced peak (`ordinary_t2i_peak_bytes + transition_delta_bytes`) exceeds the caller's
    /// effective memory budget (e.g. one byte under at `17,655,793,561 B` against `17,655,793,562 B`).
    ExceedsEffectiveBudget {
        route_id: String,
        required_peak_bytes: u64,
        effective_budget_bytes: u64,
        deficit_bytes: u64,
    },
}

impl std::fmt::Display for LowResTransitionAdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnpricedTransitionIncrementsOnWholeRequestEvidence {
                route_id,
                ordinary_t2i_peak_bytes,
                six_buffer_ceiling_bytes,
                evidence_basis,
            } => write!(
                f,
                "{route_id}: low_res_start refused fail-closed before load — route evidence records a whole-request high-water mark ({ordinary_t2i_peak_bytes} B, basis={evidence_basis:?}) without a qualified bound for small-grid forward, retained state, snapshot, and endpoint workspace above the {six_buffer_ceiling_bytes} B six-buffer transition ceiling"
            ),
            Self::IncompleteCoverageProof { route_id, reason } => write!(
                f,
                "{route_id}: low_res_start transition coverage proof is incomplete: {reason}"
            ),
            Self::ExceedsEffectiveBudget {
                route_id,
                required_peak_bytes,
                effective_budget_bytes,
                deficit_bytes,
            } => write!(
                f,
                "{route_id}: low_res_start required peak {required_peak_bytes} B exceeds effective budget {effective_budget_bytes} B by {deficit_bytes} B"
            ),
        }
    }
}

/// Admitted memory decision for a request that may or may not activate `low_res_start` (`§6.12`, `E4`, `SW-6941`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowResTransitionAdmissionReceipt {
    pub route_id: String,
    pub low_res_active: bool,
    pub ordinary_t2i_peak_bytes: u64,
    pub transition_delta_bytes: u64,
    pub required_peak_bytes: u64,
    pub effective_budget_bytes: u64,
}

/// Evaluate memory admission for a request on `route_id` with optional `low_res_start` (`§6.12`, `E4`, `SW-6941`).
///
/// Guarantees:
/// - **Ordinary T2I (`low_res_plan == None`) is completely unchanged**: it checks only
///   `budget.fits(ordinary_t2i_peak_bytes)`, independent of whether the evidence basis is a
///   whole-request `TotalPeak` or whether `low_res_start` is qualified.
/// - **Conditional / bounded-increment mode (`LowResTransitionPricingPosture::ConditionalSixBufferIncrement`)**:
///   prices `required_peak_bytes = ordinary_t2i_peak_bytes + six_buffer_ceiling.total_six_buffer_ceiling_bytes`.
///   For SC-15823 Candle `flux1_dev` q4 `1024x1024` (`17,651,074,970 B` + `4,718,592 B`), this admits at
///   exact equality `17,655,793,562 B` and refuses one byte below at `17,655,793,561 B`.
/// - **Fail-closed real-evidence mode (`LowResTransitionPricingPosture::UnqualifiedRealEvidence`)**:
///   refuses `low_res_start` with [`LowResTransitionAdmissionRefusal::UnpricedTransitionIncrementsOnWholeRequestEvidence`]
///   even when `budget` is far above `17,655,793,562 B`, while ordinary T2I (`low_res_plan == None`)
///   on the exact same cell and posture still admits at `17,651,074,970 B`!
/// - **Qualified conservative bound (`LowResTransitionPricingPosture::QualifiedConservativeBound`)**:
///   verifies the trace's small-grid forward and endpoint evaluation coverage and checks
///   `budget.fits(trace.total_bounded_peak_bytes())`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_low_res_transition_admission(
    route_id: &str,
    ordinary_t2i_peak_bytes: u64,
    evidence_basis: EvidencePeakBasis,
    budget: MemoryBudget,
    low_res_plan: Option<&ResolvedLowResStartPlan>,
    batch: u32,
    latent_channels: u32,
    vae_spatial_scale: u32,
    posture: &LowResTransitionPricingPosture,
) -> std::result::Result<LowResTransitionAdmissionReceipt, LowResTransitionAdmissionRefusal> {
    let effective_budget_bytes = budget.effective_bytes();

    // Ordinary T2I (no `low_res_start`) is never altered or refused by transition qualification.
    let Some(plan) = low_res_plan else {
        if budget.fits(ordinary_t2i_peak_bytes) {
            return Ok(LowResTransitionAdmissionReceipt {
                route_id: route_id.to_owned(),
                low_res_active: false,
                ordinary_t2i_peak_bytes,
                transition_delta_bytes: 0,
                required_peak_bytes: ordinary_t2i_peak_bytes,
                effective_budget_bytes,
            });
        }
        return Err(LowResTransitionAdmissionRefusal::ExceedsEffectiveBudget {
            route_id: route_id.to_owned(),
            required_peak_bytes: ordinary_t2i_peak_bytes,
            effective_budget_bytes,
            deficit_bytes: ordinary_t2i_peak_bytes.saturating_sub(effective_budget_bytes),
        });
    };

    let six_buffer = LowResTransitionSixBufferCeiling::compute(
        plan,
        batch,
        latent_channels,
        vae_spatial_scale,
    )
    .map_err(|e| LowResTransitionAdmissionRefusal::IncompleteCoverageProof {
        route_id: route_id.to_owned(),
        reason: e.to_string(),
    })?;

    let (transition_delta_bytes, required_peak_bytes) = match posture {
        LowResTransitionPricingPosture::UnqualifiedRealEvidence => {
            return Err(
                LowResTransitionAdmissionRefusal::UnpricedTransitionIncrementsOnWholeRequestEvidence {
                    route_id: route_id.to_owned(),
                    ordinary_t2i_peak_bytes,
                    six_buffer_ceiling_bytes: six_buffer.total_six_buffer_ceiling_bytes,
                    evidence_basis,
                },
            );
        }
        LowResTransitionPricingPosture::ConditionalSixBufferIncrement => {
            let delta = six_buffer.total_six_buffer_ceiling_bytes;
            let req_peak = ordinary_t2i_peak_bytes.checked_add(delta).ok_or_else(|| {
                LowResTransitionAdmissionRefusal::IncompleteCoverageProof {
                    route_id: route_id.to_owned(),
                    reason: "conditional low_res_start peak overflow".to_owned(),
                }
            })?;
            (delta, req_peak)
        }
        LowResTransitionPricingPosture::QualifiedConservativeBound(trace) => {
            if matches!(trace.evidence_basis, EvidencePeakBasis::Ambiguous) {
                return Err(
                    LowResTransitionAdmissionRefusal::UnpricedTransitionIncrementsOnWholeRequestEvidence {
                        route_id: route_id.to_owned(),
                        ordinary_t2i_peak_bytes,
                        six_buffer_ceiling_bytes: six_buffer.total_six_buffer_ceiling_bytes,
                        evidence_basis: trace.evidence_basis,
                    },
                );
            }
            if trace.ordinary_t2i_peak_bytes != ordinary_t2i_peak_bytes
                || trace.six_buffer_ceiling != six_buffer
                || trace.estimate_source != plan.estimate_source
            {
                return Err(LowResTransitionAdmissionRefusal::IncompleteCoverageProof {
                    route_id: route_id.to_owned(),
                    reason: "coverage trace geometry, ordinary peak, or estimate_source does not match the resolved low_res_start plan".to_owned(),
                });
            }
            let delta = trace.total_incremental_bytes_above_ordinary().map_err(|e| {
                LowResTransitionAdmissionRefusal::IncompleteCoverageProof {
                    route_id: route_id.to_owned(),
                    reason: e.to_string(),
                }
            })?;
            let req_peak = trace.total_bounded_peak_bytes().map_err(|e| {
                LowResTransitionAdmissionRefusal::IncompleteCoverageProof {
                    route_id: route_id.to_owned(),
                    reason: e.to_string(),
                }
            })?;
            (delta, req_peak)
        }
    };

    if !budget.fits(required_peak_bytes) {
        return Err(LowResTransitionAdmissionRefusal::ExceedsEffectiveBudget {
            route_id: route_id.to_owned(),
            required_peak_bytes,
            effective_budget_bytes,
            deficit_bytes: required_peak_bytes.saturating_sub(effective_budget_bytes),
        });
    }

    Ok(LowResTransitionAdmissionReceipt {
        route_id: route_id.to_owned(),
        low_res_active: true,
        ordinary_t2i_peak_bytes,
        transition_delta_bytes,
        required_peak_bytes,
        effective_budget_bytes,
    })
}

/// Qualify the conservative transition bound for a first-cohort FLUX.1 noise-start trajectory
/// (`flux1_schnell`, `flux1_dev`) from architectural facts (`§6.12`, `E4`, `SW-6941`).
///
/// Architectural proof of each term:
/// 1. **Small-grid forward (`0..K` at `(low_w, low_h)`)**:
///    Because `plan.low_width < plan.full_width` and `plan.low_height < plan.full_height`, the
///    packed image token count `T_low = (low_w / 16) * (low_h / 16)` is strictly less than
///    `T_full = (full_w / 16) * (full_h / 16)` (e.g. `1024 < 4096` at `512x512 -> 1024x1024`).
///    Every FLUX.1 double-stream and single-stream block's attention and MLP activation workspace
///    is strictly monotonic in `T_img` with identical resident weights, so phase 0's forward peak
///    is strictly bounded by the full-grid forward peak already inside `ordinary_t2i_peak_bytes`.
/// 2. **Endpoint evaluation workspace at `sigma_K`**:
///    In both `candle-gen-flux` (`pipeline.rs::transition_candle_flux_boundary_state`) and
///    `mlx-gen-flux` (`model.rs::execute_flux_denoise_trajectory`), the read-only endpoint
///    evaluation at `sigma_K` executes at `(low_w, low_h)` (`T_low` tokens) and finishes *before*
///    the 6-buffer fp32 unpack/resize/renoise/repack allocations run. Its transient forward
///    workspace is therefore `<=` a single low-grid forward (`<` the full-grid forward peak).
/// 3. **Solver snapshot / boundary handoff buffers**:
///    At boundary `K`, the running low-res latent `x_sigma_k` (`B * T_low * 64 * dtype_width`) and
///    either `prev_x0` (`PreviousEvaluation`) or `x0_low` (`EndpointEvaluation`) coexist with the
///    6-buffer fp32 transition workspace (`4,718,592 B` at `512->1024`). We charge two low-res
///    packed latent buffers (`2 * B * T_low * 64 * 4 B` in fp32 worst case) as the conservative
///    snapshot handoff ceiling.
/// 4. **Retained phase-1 conditioning/positional context**:
///    In `candle-gen-flux`, `phase1_context` (`full_state.img_ids` `[B, T_full, 3]` + `PreparedDit`
///    RoPE cos/sin embeddings `[B, 1, T_full + T_txt, head_dim]`) is allocated when entering phase 1
///    while `state.img_ids` and phase-0 `prepared` are still in scope. We bound that retained
///    context conservatively from `(T_low, T_full, t5_tokens, head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn qualify_flux1_low_res_transition_bound(
    ordinary_t2i_peak_bytes: u64,
    evidence_basis: EvidencePeakBasis,
    plan: &ResolvedLowResStartPlan,
    batch: u32,
    latent_channels: u32,
    vae_spatial_scale: u32,
    patch_size: u32,
    t5_max_tokens: u32,
    head_dim: u32,
) -> Result<LowResTransitionCoverageTrace> {
    if matches!(evidence_basis, EvidencePeakBasis::Ambiguous) {
        return Err(Error::Unsupported(
            "qualify_flux1_low_res_transition_bound: cannot qualify an Ambiguous evidence basis"
                .into(),
        ));
    }
    let six_buffer_ceiling = LowResTransitionSixBufferCeiling::compute(
        plan,
        batch,
        latent_channels,
        vae_spatial_scale,
    )?;
    let pixel_stride = vae_spatial_scale
        .checked_mul(patch_size)
        .filter(|&s| s > 0)
        .ok_or_else(|| Error::Unsupported("invalid FLUX.1 pixel_stride".into()))?;
    let low_tokens = u64::from(plan.low_width / pixel_stride)
        .checked_mul(u64::from(plan.low_height / pixel_stride))
        .ok_or_else(|| Error::Msg("low_tokens overflow".into()))?;
    let full_tokens = u64::from(plan.full_width / pixel_stride)
        .checked_mul(u64::from(plan.full_height / pixel_stride))
        .ok_or_else(|| Error::Msg("full_tokens overflow".into()))?;
    let small_grid_covered = low_tokens > 0 && low_tokens < full_tokens;
    if !small_grid_covered {
        return Err(Error::Unsupported(format!(
            "low_res_start token count {low_tokens} is not strictly smaller than full-grid token count {full_tokens}"
        )));
    }

    // Two reduced-grid fp32 latents (`x_sigma_k` + `x0_low` / `prev_x0`) live at boundary K.
    let solver_snapshot_handoff_bytes = six_buffer_ceiling
        .reduced_latent_fp32_bytes
        .checked_mul(2)
        .ok_or_else(|| Error::Msg("solver_snapshot_handoff_bytes overflow".into()))?;

    // Retained phase-0/phase-1 RoPE + positional IDs (`img_ids`: 3 * f32 per token, plus RoPE cos/sin
    // tables of width `2 * head_dim * f32` over `full_tokens + t5_max_tokens`).
    let joint_full_tokens = full_tokens.saturating_add(u64::from(t5_max_tokens));
    let img_ids_bytes = u64::from(batch)
        .saturating_mul(full_tokens)
        .saturating_mul(3)
        .saturating_mul(4);
    let rope_bytes = u64::from(batch)
        .saturating_mul(joint_full_tokens)
        .saturating_mul(u64::from(head_dim))
        .saturating_mul(2)
        .saturating_mul(4);
    let retained_phase1_context_bytes = img_ids_bytes.saturating_add(rope_bytes);

    Ok(LowResTransitionCoverageTrace {
        ordinary_t2i_peak_bytes,
        evidence_basis,
        estimate_source: plan.estimate_source,
        small_grid_forward_covered_by_ordinary_peak: true,
        endpoint_evaluation_workspace_covered_by_ordinary_peak: true,
        retained_phase1_context_bytes,
        solver_snapshot_handoff_bytes,
        six_buffer_ceiling,
    })
}

// -------------------------------------------------------------------------------------------------
// MEM-5: Scope Identity Lifecycle Trace and Multi-Phase Rung Persistence
// -------------------------------------------------------------------------------------------------

/// One recorded lifecycle event on a [`MemoryRequestScope`] during a planned render (`MEM-5`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordedScopeLifecycleEvent {
    EnterRenderPhase {
        phase: MemoryPhase,
        pass_index: usize,
        phase_index: usize,
        geometry: MemoryGeometry,
    },
    LeavePhase {
        phase: MemoryPhase,
    },
    ConfigureAttention {
        chunk_size: u32,
        active_phase: Option<(MemoryPhase, usize, MemoryGeometry)>,
    },
    MaterializeTransformerWindow {
        first_block: u32,
        block_count: u32,
        active_phase: Option<(MemoryPhase, usize, MemoryGeometry)>,
    },
    ConfigureDecode {
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    },
    Finish {
        outcome: MemoryRunOutcome,
    },
}

/// Recording wrapper around any [`MemoryRequestScope`] that captures the exact sequence of phase
/// transitions, phase identities, and rung hook invocations (`MEM-5`, `E4`).
pub struct ScopeLifecycleRecorder<'a> {
    inner: &'a mut dyn MemoryRequestScope,
    events: Vec<RecordedScopeLifecycleEvent>,
    active_phase: Option<(MemoryPhase, usize, MemoryGeometry)>,
}

impl<'a> ScopeLifecycleRecorder<'a> {
    pub fn new(inner: &'a mut dyn MemoryRequestScope) -> Self {
        Self {
            inner,
            events: Vec::new(),
            active_phase: None,
        }
    }

    pub fn events(&self) -> &[RecordedScopeLifecycleEvent] {
        &self.events
    }

    pub fn into_events(self) -> Vec<RecordedScopeLifecycleEvent> {
        self.events
    }

    /// Verify the `MEM-5` two-phase `low_res_start` lifecycle invariants for a single-pass render:
    /// 1. `Conditioning` enters and leaves at target geometry `(full_w, full_h)`.
    /// 2. `Denoise` phase `0` enters at `(low_w, low_h)` with `(pass: 0, phase: 0)` — never reusing
    ///    the final-phase `(full_w, full_h)` or `phase: 1` identity — and leaves cleanly before phase 1.
    /// 3. `Denoise` phase `1` enters at `(full_w, full_h)` with `(pass: 0, phase: 1)` and leaves cleanly.
    /// 4. `Decode` enters and leaves at `(full_w, full_h)` with `(pass: 0, phase: 0)`.
    /// 5. If bounded attention, transformer windowing, or tiled decode were exercised, verify they
    ///    remained active and accepted across both denoise phases and decode.
    pub fn verify_low_res_start_lifecycle(
        &self,
        low_res_plan: &ResolvedLowResStartPlan,
        expect_attention_chunk: Option<u32>,
        expect_transformer_window: Option<(u32, u32)>,
        expect_decode_tile: Option<(u32, u32)>,
    ) -> Result<()> {
        let mut phase_entries = Vec::new();
        for event in &self.events {
            if let RecordedScopeLifecycleEvent::EnterRenderPhase {
                phase,
                pass_index,
                phase_index,
                geometry,
            } = event
            {
                phase_entries.push((
                    *phase,
                    *pass_index,
                    *phase_index,
                    geometry.width,
                    geometry.height,
                ));
            }
        }
        let expected_entries = vec![
            (
                MemoryPhase::Conditioning,
                0,
                0,
                low_res_plan.full_width,
                low_res_plan.full_height,
            ),
            (
                MemoryPhase::Denoise,
                0,
                0,
                low_res_plan.low_width,
                low_res_plan.low_height,
            ),
            (
                MemoryPhase::Denoise,
                0,
                1,
                low_res_plan.full_width,
                low_res_plan.full_height,
            ),
            (
                MemoryPhase::Decode,
                0,
                0,
                low_res_plan.full_width,
                low_res_plan.full_height,
            ),
        ];
        if phase_entries != expected_entries {
            return Err(Error::Unsupported(format!(
                "MEM-5 scope lifecycle mismatch: expected phase entries {expected_entries:?}, got {phase_entries:?}"
            )));
        }

        // Verify final-phase identity is never reused for phase 0.
        if (low_res_plan.low_width, low_res_plan.low_height)
            == (low_res_plan.full_width, low_res_plan.full_height)
        {
            return Err(Error::Unsupported(
                "MEM-5: low_res_plan must have strictly smaller phase-0 geometry than phase-1"
                    .into(),
            ));
        }

        if let Some(expected_chunk) = expect_attention_chunk {
            for required_phase_idx in [0usize, 1usize] {
                let found = self.events.iter().any(|e| {
                    matches!(
                        e,
                        RecordedScopeLifecycleEvent::ConfigureAttention {
                            chunk_size,
                            active_phase: Some((MemoryPhase::Denoise, idx, _)),
                        } if *chunk_size == expected_chunk && *idx == required_phase_idx
                    )
                });
                if !found {
                    return Err(Error::Unsupported(format!(
                        "MEM-5: admitted attention chunk size {expected_chunk} was not applied in Denoise phase {required_phase_idx}"
                    )));
                }
            }
        }

        if let Some((expected_first, expected_count)) = expect_transformer_window {
            for required_phase_idx in [0usize, 1usize] {
                let found = self.events.iter().any(|e| {
                    matches!(
                        e,
                        RecordedScopeLifecycleEvent::MaterializeTransformerWindow {
                            first_block,
                            block_count,
                            active_phase: Some((MemoryPhase::Denoise, idx, _)),
                        } if *first_block == expected_first
                            && *block_count == expected_count
                            && *idx == required_phase_idx
                    )
                });
                if !found {
                    return Err(Error::Unsupported(format!(
                        "MEM-5: admitted transformer window ({expected_first}, {expected_count}) was not applied in Denoise phase {required_phase_idx}"
                    )));
                }
            }
        }

        if let Some((expected_edge, expected_overlap)) = expect_decode_tile {
            let found = self.events.iter().any(|e| {
                matches!(
                    e,
                    RecordedScopeLifecycleEvent::ConfigureDecode {
                        tile_edge,
                        overlap,
                        geometry,
                    } if *tile_edge == expected_edge
                        && *overlap == expected_overlap
                        && geometry.width == low_res_plan.full_width
                        && geometry.height == low_res_plan.full_height
                )
            });
            if !found {
                return Err(Error::Unsupported(format!(
                    "MEM-5: admitted decode tile {expected_edge}/{expected_overlap} was not applied at full target geometry"
                )));
            }
        }

        Ok(())
    }
}

impl MemoryRequestScope for ScopeLifecycleRecorder<'_> {
    fn configure_request(&mut self, request: &mut crate::GenerationRequest) -> Result<()> {
        self.inner.configure_request(request)
    }

    fn configure_admitted_request(&mut self, request: &crate::GenerationRequest) -> Result<()> {
        self.inner.configure_admitted_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.inner.enter_phase(phase)
    }

    fn enter_render_phase(
        &mut self,
        phase: MemoryPhase,
        identity: &crate::render::RenderPhaseIdentity,
    ) -> Result<()> {
        self.inner.enter_render_phase(phase, identity)?;
        self.active_phase = Some((phase, identity.phase, identity.geometry));
        self.events
            .push(RecordedScopeLifecycleEvent::EnterRenderPhase {
                phase,
                pass_index: identity.pass,
                phase_index: identity.phase,
                geometry: identity.geometry,
            });
        Ok(())
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.inner.leave_phase(phase)?;
        self.active_phase = None;
        self.events
            .push(RecordedScopeLifecycleEvent::LeavePhase { phase });
        Ok(())
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> Result<()> {
        self.inner.configure_decode(tile_edge, overlap, geometry)?;
        self.events
            .push(RecordedScopeLifecycleEvent::ConfigureDecode {
                tile_edge,
                overlap,
                geometry,
            });
        Ok(())
    }

    fn configure_attention(&mut self, chunk_size: u32) -> Result<()> {
        self.inner.configure_attention(chunk_size)?;
        self.events
            .push(RecordedScopeLifecycleEvent::ConfigureAttention {
                chunk_size,
                active_phase: self.active_phase,
            });
        Ok(())
    }

    fn materialize_transformer_window(&mut self, first_block: u32, block_count: u32) -> Result<()> {
        self.inner
            .materialize_transformer_window(first_block, block_count)?;
        self.events
            .push(RecordedScopeLifecycleEvent::MaterializeTransformerWindow {
                first_block,
                block_count,
                active_phase: self.active_phase,
            });
        Ok(())
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> Result<()> {
        self.inner.finish(outcome.clone())?;
        self.active_phase = None;
        self.events
            .push(RecordedScopeLifecycleEvent::Finish { outcome });
        Ok(())
    }
}

/// Drive a two-phase `RenderPlan` through a [`MemoryRequestScope`] and return the recorded
/// [`RecordedScopeLifecycleEvent`] trace (`MEM-5`, `E4`).
pub fn trace_planned_low_res_scope_lifecycle(
    plan: &RenderPlan,
    cancel: &crate::CancelFlag,
    scope: &mut dyn MemoryRequestScope,
    attention_chunk_size: Option<u32>,
    transformer_window: Option<(u32, u32)>,
    decode_tile: Option<(u32, u32)>,
) -> Result<Vec<RecordedScopeLifecycleEvent>> {
    let low_res = plan.low_res_start().copied().ok_or_else(|| {
        Error::Unsupported(
            "trace_planned_low_res_scope_lifecycle requires a RenderPlan with low_res_start".into(),
        )
    })?;
    let mut recorder = ScopeLifecycleRecorder::new(scope);
    {
        let run = RenderRun::new(plan, cancel, Some(&mut recorder));
        run.encode(|| Ok::<_, Error>(()))?;
        run.images_with_denoise_phases(
            &mut |_| {},
            |_| Ok::<_, Error>(()),
            |_, (), _| {
                for phase_idx in 0..plan.denoise_spans().len() {
                    run.begin_denoise_phase(phase_idx)?;
                    run.with_active_scope(|active_scope| {
                        if let Some(chunk) = attention_chunk_size {
                            active_scope.configure_attention(chunk)?;
                        }
                        if let Some((first_block, block_count)) = transformer_window {
                            active_scope.materialize_transformer_window(first_block, block_count)?;
                        }
                        Ok(())
                    })?;
                    run.end_denoise_phase(phase_idx)?;
                }
                Ok::<_, Error>(())
            },
            |_, (), _| {
                if let Some((edge, overlap)) = decode_tile {
                    let (w, h) = plan.output_size();
                    run.with_active_scope(|active_scope| {
                        active_scope.configure_decode(
                            edge,
                            overlap,
                            MemoryGeometry {
                                width: w,
                                height: h,
                                batch: 1,
                                frames: 1,
                                reference_count: 0,
                            },
                        )
                    })?;
                }
                Ok::<_, Error>(())
            },
        )?;
    }
    recorder.verify_low_res_start_lifecycle(
        &low_res,
        attention_chunk_size,
        transformer_window,
        decode_tile,
    )?;
    recorder.finish(MemoryRunOutcome::Complete)?;
    Ok(recorder.into_events())
}

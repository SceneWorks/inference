//! Header-only, per-phase COUNTED pricing of one iOS request: the architecture facts the iOS
//! anchor law (epic 16774, `mlx-gen-ios-catalog::anchor`) scales a measured device anchor by.
//!
//! Every iOS provider crate exposes `ios_pricing(&IosPricingRequest) -> Result<IosPricing>` built
//! from the SAME `ios_quantum` formulas its per-quantum guards price (component load floors,
//! resident component bytes under the request's regime, and each phase's largest counted quantum
//! scratch), and `ios_quantum_phase(label) -> Option<IosPhase>`, which attributes each of its
//! quantum labels to the phase it runs in (a device capture splits its kernel lifetime peak by
//! phase through those labels). Pricing reads safetensors headers and small configs only: it never
//! builds a graph or reads a tensor payload, so it runs on device at admission time.
//!
//! Prices are NOT admission. Admission is a measured device anchor plus the anchor law; a price
//! only says how a measured residue scales between the anchor's regime/geometry and a request's.

use crate::gen_core::{GenerationMemory, LoadSpec, MemoryGeometry, MemoryMode, MemorySelection};

/// Reborrow an engine capacity check across a backend-neutral source-reader boundary.
/// The original engine error stays owned here (never stringified or down-converted): a core
/// consumer cannot swallow a denial by returning `Ok`. Ordinary core/work errors retain their
/// existing typed conversion at the call site. This grants no source authority or capacity.
pub fn with_core_source_capacity<T>(
    next: &mut dyn FnMut(u64) -> crate::Result<()>,
    work: impl FnOnce(&mut dyn FnMut(u64) -> crate::gen_core::Result<()>) -> crate::Result<T>,
) -> crate::Result<T> {
    let mut original_error = None;
    let result = {
        let mut core_next = |bytes| {
            if original_error.is_some() {
                return Err(crate::gen_core::Error::Unsupported(
                    "source metadata capacity already refused".into(),
                ));
            }
            match next(bytes) {
                Ok(()) => Ok(()),
                Err(error) => {
                    original_error = Some(error);
                    Err(crate::gen_core::Error::Unsupported(
                        "source metadata capacity refused".into(),
                    ))
                }
            }
        };
        work(&mut core_next)
    };
    match original_error {
        Some(error) => Err(error),
        None => result,
    }
}

/// The execution phases a request's memory peak is attributed to. Load floors are their own phase
/// (a load's peak is the component's stored bytes plus its read scratch, not an activation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IosPhase {
    /// A component load floor (`ios_quantum::before_load*`), whichever component it is.
    Load,
    /// Text / image-prompt conditioning (text encoders, CLIP vision, conditioners).
    Conditioning,
    /// An image ENCODE: an img2img / inpaint init latent, a control hint, a reference image.
    Encode,
    /// The denoiser (DiT / U-Net / ControlNet blocks, sampler steps).
    Denoise,
    /// The decode to pixels (VAE, tile assembly, RGB conversion), and post-processing tiles.
    Decode,
}

impl IosPhase {
    pub const ALL: [IosPhase; 5] = [
        IosPhase::Load,
        IosPhase::Conditioning,
        IosPhase::Encode,
        IosPhase::Denoise,
        IosPhase::Decode,
    ];

    /// Stable key used in evidence rows.
    pub fn as_key(self) -> &'static str {
        match self {
            IosPhase::Load => "load",
            IosPhase::Conditioning => "conditioning",
            IosPhase::Encode => "encode",
            IosPhase::Denoise => "denoise",
            IosPhase::Decode => "decode",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.as_key() == key)
    }
}

/// The counted price of ONE phase: the component bytes resident in it under the request's regime,
/// plus its LARGEST counted quantum (live input + scratch, the same bytes its guard checks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosPhasePrice {
    pub phase: IosPhase,
    /// Component (weight) bytes resident throughout this phase under the request's regime.
    pub resident_bytes: u64,
    /// The largest counted quantum of the phase (its guard's priced scratch), bytes.
    pub scratch_bytes: u64,
    /// The quantum label that binds `scratch_bytes` (the anchor law calibrates a measured residue
    /// against a counted formula only where the request's binding quantum is the anchor's).
    pub binding: String,
}

impl IosPhasePrice {
    pub fn peak_bytes(&self) -> u64 {
        self.resident_bytes.saturating_add(self.scratch_bytes)
    }
}

/// Actual operands of ONE coincident binding/epoch/window. Zero fields mean not applicable only
/// when the provider supplied this basis; `None` on a window is unverified, never inferred fixed.
/// No generic area exponent or independent maxima are applied to these operands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IosWorkBasis {
    pub forward_rows: u64,
    pub token_count: u64,
    pub model_width: u32,
    pub model_height: u32,
    pub source_plane_bytes: u64,
    pub processed_plane_bytes: u64,
    pub output_plane_bytes: u64,
    pub patch_width: u32,
    pub patch_height: u32,
}

/// One potentially binding physical-liveness domain. Counts are source-owned ledgers, NOT an
/// automatic physical bound or subtractable lower bound. Only a provider's `IosBounder` supplies
/// a certified transfer/full-physical bound. Equal labels/counts/classes alone prove nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosWindowPrice {
    pub phase: IosPhase,
    /// Stable semantic binding/epoch/window class, not an invocation number. Legacy `take` is None.
    pub class: Option<String>,
    pub label: String,
    /// Structural resident components at this same point.
    pub resident_bytes: u64,
    /// Coincident ambient held owners, excluding storage already counted in resident/scratch.
    pub held_bytes: u64,
    /// The guarded operation's counted live-input/workspace upper, not a physical measurement.
    pub scratch_bytes: u64,
    pub basis: Option<IosWorkBasis>,
    /// Source storage/liveness provenance domain, NOT a human signoff or proof-by-string.
    pub owner_domain: String,
}

impl IosWindowPrice {
    pub fn logical_peak_bytes(&self) -> u64 {
        self.resident_bytes
            .saturating_add(self.held_bytes)
            .saturating_add(self.scratch_bytes)
    }
    pub fn profile_class(&self) -> Option<&str> {
        self.class.as_deref().filter(|class| {
            !class.is_empty() && self.basis.is_some() && !self.owner_domain.is_empty()
        })
    }
}

/// Bounded ACTUAL metadata/check facts attached by the catalog to ANCHOR pricing only. Providers
/// map their real observed callback/window domains; no planned logical replay belongs here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosObservedMetadata {
    pub provider: String,
    pub kind: String,
    pub fields: std::collections::BTreeMap<String, u64>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosObservedCheck {
    pub phase: Option<IosPhase>,
    pub label: String,
    /// Last actual mark of each producer/kind at this check; indices address `metadata`.
    /// Co-occurrence/epoch correspondence is proved by provider CODE, not a generic string match.
    pub context: Vec<u32>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IosAnchorObservation {
    pub domain: IosMemoryDomain,
    pub coordinate: IosMemoryCoordinate,
    pub baseline_bytes: u64,
    pub metadata: Vec<IosObservedMetadata>,
    pub checks: Vec<IosObservedCheck>,
    /// A gap invalidates phase attribution; the law uses the whole-run kernel upper instead.
    pub telemetry_complete: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosAnchorWitness {
    pub capture: u32,
    pub check: u32,
}

/// Memory-counter domain of a certificate/measurement. Native allocator proofs do not include
/// Rust/host/process allocator/driver demand and MUST NOT authorize process-growth admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IosMemoryDomain {
    #[default]
    Unspecified,
    ProcessGrowth,
    NativeAllocatorGrowth,
    NativeAllocatorAbsolute,
}

/// Baseline coordinate is part of the certificate, not an implicit zero/foreign-inventory credit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IosMemoryCoordinate {
    #[default]
    Unspecified,
    Absolute,
    BeforeRequestConstruction,
    BeforeModelLoad,
    WindowEntry,
}

/// A provider-certified correspondence candidate. The signed upper must hold at EVERY matched
/// instant covered by its classes, not be a difference of independently maximized logical uppers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosAnchorCandidate {
    pub anchor_class: String,
    pub delta_growth_upper: i128,
    /// Optional SOURCE-PROVED co-resident growth lower bound in the anchor's baseline domain.
    /// None is not a zero-baseline assumption: it means this transfer does not subtract such a
    /// bound. A logical resident/scratch upper must never be supplied as this lower bound.
    pub physical_anchor_lower_bound: Option<u64>,
    /// Actual check/metadata witnesses in `anchor_pricing.observations`. Provider CODE validates
    /// their source epoch/window correspondence; the consumer checks existence/label/context.
    pub witnesses: Vec<IosAnchorWitness>,
}

/// Returned only by source-owned proof callbacks. Each record covers the WHOLE request class;
/// every plausible anchor correspondence belongs in `anchor_candidates` (the law takes MAX).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IosWindowBound {
    Transfer {
        domain: IosMemoryDomain,
        coordinate: IosMemoryCoordinate,
        request_class: String,
        anchor_candidates: Vec<IosAnchorCandidate>,
    },
    /// COMPLETE unmargined growth upper in the named domain. ProcessGrowth includes co-residency,
    /// host, allocator/cache and workspace. Native-only certificates never authorize process use.
    /// A raw guard scratch or logical upper sum alone is NOT this certificate.
    RawPhysical {
        domain: IosMemoryDomain,
        coordinate: IosMemoryCoordinate,
        request_class: String,
        growth_upper: u64,
    },
}

/// Proof authority remains provider CODE using actual contexts/ledgers/bases. No backend dependency
/// is introduced here, and no class string/numeric equality is accepted as a transfer proof.
pub type IosBounder = fn(
    anchor_request: &IosPricingRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> crate::Result<Vec<IosWindowBound>>;

/// The MLX stream family an execution's operations actually ran on. Allocation laws are lowering
/// specific: a certificate audited for one stream family never certifies another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IosExecutionStream {
    Cpu,
    Metal,
}

/// Facts about the execution being CERTIFIED, stated at the admission boundary by the caller that
/// owns that execution (SDK/runtime/capture harness). Never sampled from whichever thread happens
/// to evaluate a pure pricing/bounding function, and never defaulted when absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IosExecutionFacts {
    /// Effective stream of every wrapper op, including stream-less frontend casts.
    pub stream: IosExecutionStream,
    /// Whether the execution runs inside the iOS quantum scope (`ios_quantum::enabled()`), which
    /// selects graph branches (bounded attention, tiled encode) the pricing assumes.
    pub ios_quantum_scope: bool,
    /// Explicit host byte budget for EXACT schedule/epoch analysis (checked with
    /// `ResolvedFlowControls::check_capture_schedule_bytes` before any schedule is built). A
    /// request whose schedule exceeds it is refused, never clamped.
    pub analysis_budget_bytes: u64,
    /// The execution owner's statement of host allocations made by the callbacks IT installs
    /// (`on_progress` closures, an `ios_quantum` listener). `None` is unstated: a provider whose
    /// host certificate depends on it refuses, naming the missing term (never zero).
    pub callback_host: Option<IosCallbackHost>,
}

/// Host allocations made by caller-owned callbacks, stated by the caller that installs them and
/// derived from THAT caller's source. The provider multiplies the per-event terms by its own
/// source-derived event counts; provider-side `Progress`/formatting work stays provider-owned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IosCallbackHost {
    /// Every installed callback is source-demonstrably non-allocating (e.g. `|_| {}`), and no
    /// quantum listener is installed.
    NonAllocating,
    /// Source-derived cumulative upper (no drop credit): bytes allocated once at installation,
    /// plus per progress event, plus per preview-sink event, plus per quantum mark delivered to
    /// a listener, plus per BYTE of each delivered Checked/Synchronized label (a listener that
    /// copies the full label). The provider supplies each event count and its own source label
    /// lengths; no label is capped or truncated to make this finite.
    Bounded {
        fixed_bytes: u64,
        per_progress_event_bytes: u64,
        per_preview_event_bytes: u64,
        per_quantum_mark_bytes: u64,
        per_quantum_label_byte_bytes: u64,
    },
}

/// A NOT-source-proved premise a provider's bounds depend on (e.g. a retirement ordering asserted
/// rather than proved). Carried beside the bounds so a consumer can never use them silently.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IosCertificatePremise {
    /// Stable identifier an explicit, user-approved admission policy names to accept it.
    pub id: String,
    /// What is assumed and why it is not proved.
    pub statement: String,
}

/// Bounds plus every premise they rest on. Empty `premises` means fully source-proved bounds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IosCertifiedBounds {
    pub bounds: Vec<IosWindowBound>,
    pub premises: Vec<IosCertificatePremise>,
    /// The provider's request-scoped HOST growth outside the MLX native counter, a SEPARATE
    /// domain from `bounds` (never folded into a native delta here).
    pub host: IosHostGrowth,
}

impl IosCertifiedBounds {
    /// Fully source-proved bounds (no premise), with no host-growth claim.
    pub fn proven(bounds: Vec<IosWindowBound>) -> Self {
        Self {
            bounds,
            premises: Vec::new(),
            host: IosHostGrowth::Undeclared,
        }
    }

    /// Fully source-proved native bounds plus a source-derived request host-growth upper and the
    /// request's explicitly REQUESTED OS thread-stack bytes (a separate physical quantity) for
    /// EVERY distinct native request class (the same cumulative request uppers per class).
    pub fn proven_with_host(
        bounds: Vec<IosWindowBound>,
        host_growth_upper: u64,
        os_stack_request_bytes: u64,
    ) -> Self {
        let mut classes = std::collections::BTreeSet::new();
        let host = bounds
            .iter()
            .filter_map(|bound| match bound {
                IosWindowBound::Transfer {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    request_class,
                    ..
                } => Some(request_class.clone()),
                _ => None,
            })
            .filter(|class| classes.insert(class.clone()))
            .map(|request_class| IosHostGrowthBound {
                request_class,
                growth_upper: host_growth_upper,
                os_stack_request_bytes,
            })
            .collect();
        Self {
            bounds,
            premises: Vec::new(),
            host: IosHostGrowth::Declared(host),
        }
    }
}

/// A provider's claim about request-scoped HOST (Rust heap) growth outside the MLX native
/// allocator counter. Separate domain: it never alters native certificates or measured evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum IosHostGrowth {
    /// No host-growth claim. The native-to-process model's stated fixed overhead is the only
    /// allowance for non-native growth (the original activity-26569 premise, unchanged).
    #[default]
    Undeclared,
    /// A finite source/request-derived cumulative upper (bytes, no drop credit) on ALL the
    /// request's host allocations outside the MLX counter, per native request class. Mapping requires exactly one entry for every
    /// native transfer class and refuses missing, duplicate or foreign entries; it is consumed
    /// only under the separately approved host-extended premise.
    Declared(Vec<IosHostGrowthBound>),
    /// The provider HAS request-scoped host allocations outside the MLX counter but no finite
    /// bound for at least one of them yet (the named reason). Native-domain use is unaffected;
    /// any mapping into the process domain refuses with this reason, never treating it as zero.
    Unproved(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosHostGrowthBound {
    pub request_class: String,
    /// Rust heap: finite source/request-derived cumulative host allocation upper.
    pub growth_upper: u64,
    /// Separate PHYSICAL quantity: OS thread-stack bytes the request's code explicitly requests
    /// (e.g. `Builder::stack_size`), never part of `growth_upper` or any native total.
    pub os_stack_request_bytes: u64,
}

/// The stable id of the activity-26569 native-to-process premise.
pub const NATIVE_TO_PROCESS_PREMISE_ID: &str = "activity-26569/target-device-native-to-process";

/// The stable id of the HOST-EXTENDED native-to-process premise (PROPOSED, unapproved). It is a
/// distinct premise: approving [`NATIVE_TO_PROCESS_PREMISE_ID`] never approves this one.
pub const NATIVE_TO_PROCESS_HOST_PREMISE_ID: &str =
    "activity-26569/target-device-native-to-process-with-host-v1";

/// The stable id of the Rust TOOLCHAIN requested-layout premise (unapproved by default): host
/// growth uppers that rest on toolchain container-growth or repr(Rust) layout rules rather than
/// on source guarantees carry it, so a consumer can never use them silently.
pub const RUST_TOOLCHAIN_LAYOUT_PREMISE_ID: &str = "rust-toolchain/requested-host-layout-v1";

/// The toolchain requested-layout premise a host-growth upper rests on, for policy approval.
pub fn rust_toolchain_layout_premise() -> IosCertificatePremise {
    IosCertificatePremise {
        id: RUST_TOOLCHAIN_LAYOUT_PREMISE_ID.into(),
        statement: "on the authority-pinned Rust toolchain (1.96.0 / std ac68): (a) String::from \
                    / to_string of a str allocate exactly its length; (b) a Vec grown by push or \
                    collect to n elements requests at most (4n + 8) elements cumulatively; (c) an \
                    insert-only BTreeMap allocates at most entries + 1 nodes, each no larger than \
                    an internal node; (d) a repr(Rust) struct is no larger than its fields each \
                    rounded to the maximum field alignment, and a repr(Rust) enum no larger than \
                    one tag word plus its widest payload. NOT source-proved"
            .into(),
    }
}

/// Activity 26569's TARGET-DEVICE ENGINEERING MODEL, as an explicit premise and transform, stated
/// in exactly the ANCHOR-RELATIVE form the transform applies: for the anchor and request of one
/// transfer (same family, device, build, coordinate), the process (`phys_footprint`) growth delta
/// is bounded by the native-allocator growth delta plus `overhead_bytes`. A bound on each
/// execution's process growth by its own native growth does NOT imply this (a native-heavy anchor
/// can make the native delta small while the process delta is large), so the premise names the
/// delta form itself. It is NOT source-proved. It is applied only under an explicit,
/// user-approved policy; it rewrites certificate DELTAS (never measured evidence); its consumer
/// must validate it against both measured endpoint anchors and still enforce the anchors' stated
/// margin and the live headroom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosNativeToProcessModel {
    /// Stated per-window process overhead beyond native growth, in bytes.
    pub overhead_bytes: u64,
    /// Where the model comes from (device/build/capture identities), recorded with the premise.
    pub provenance: String,
}

impl IosNativeToProcessModel {
    /// The premise this model asserts, with its parameters, for policy approval and recording.
    pub fn premise(&self) -> IosCertificatePremise {
        IosCertificatePremise {
            id: NATIVE_TO_PROCESS_PREMISE_ID.into(),
            statement: format!(
                "for each transfer of one family: process_growth(request) - process_growth(anchor) \
                 <= native_growth(request) - native_growth(anchor) + {} B per window \
                 (anchor-relative engineering model: {})",
                self.overhead_bytes, self.provenance
            ),
        }
    }

    /// The host-extended premise, for policy approval and recording beside [`Self::premise`].
    pub fn host_premise(&self) -> IosCertificatePremise {
        IosCertificatePremise {
            id: NATIVE_TO_PROCESS_HOST_PREMISE_ID.into(),
            statement: format!(
                "for each transfer of one family: process_growth(request) - process_growth(anchor) \
                 <= native_growth(request) - native_growth(anchor) + host_growth_upper(request) + \
                 os_stack_request(request) + {} B per window, where os_stack_request is the \
                 provider's explicitly requested worker thread-stack bytes (assumed to bound their \
                 physical charge) and host_growth_upper is the provider's FINITE \
                 source/request-derived cumulative upper on ALL request-scoped host allocations \
                 outside the MLX counter (requested layouts, parse/tokenizer/schedule bookkeeping \
                 and container growth, the latter under stated toolchain assumptions); the \
                 anchor's own host growth is not subtracted, and only unmodelled physical, driver \
                 and allocator residual remains inside the stated overhead (anchor-relative \
                 engineering model: {})",
                self.overhead_bytes, self.provenance
            ),
        }
    }

    /// Map native-allocator TRANSFER certificates into the process domain. Process-domain bounds
    /// pass unchanged; any other domain, a raw native bound, or overflow refuses: a native raw
    /// total has no anchor-relative form this model is stated for.
    pub fn to_process(&self, bounds: &[IosWindowBound]) -> crate::Result<Vec<IosWindowBound>> {
        self.map_native(bounds, |_| Ok(0))
    }

    /// [`Self::to_process`] with a DECLARED source-derived host term: each native transfer's
    /// delta gains the stated overhead PLUS its class's host upper (added, never subtracted, and
    /// the native delta is retained). Every native transfer class needs exactly one host entry;
    /// a missing, duplicate or foreign entry refuses (host evidence is never silently zero).
    /// An undeclared host claim is the original premise exactly.
    pub fn to_process_with_host(
        &self,
        bounds: &[IosWindowBound],
        host: &IosHostGrowth,
    ) -> crate::Result<Vec<IosWindowBound>> {
        let entries = match host {
            IosHostGrowth::Undeclared => return self.to_process(bounds),
            IosHostGrowth::Unproved(reason) => {
                return Err(crate::Error::Unsupported(format!(
                    "host growth unproved: {reason}"
                )))
            }
            IosHostGrowth::Declared(entries) => entries,
        };
        let refuse = |why: String| crate::Error::Unsupported(format!("host growth: {why}"));
        let mut by_class = std::collections::BTreeMap::new();
        for entry in entries {
            let bytes = entry
                .growth_upper
                .checked_add(entry.os_stack_request_bytes)
                .ok_or_else(|| refuse("host term overflows".into()))?;
            if by_class
                .insert(entry.request_class.as_str(), bytes)
                .is_some()
            {
                return Err(refuse(format!(
                    "duplicate entry for {}",
                    entry.request_class
                )));
            }
        }
        let native: std::collections::BTreeSet<&str> = bounds
            .iter()
            .filter_map(|bound| match bound {
                IosWindowBound::Transfer {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    request_class,
                    ..
                } => Some(request_class.as_str()),
                _ => None,
            })
            .collect();
        if let Some(foreign) = by_class.keys().find(|class| !native.contains(*class)) {
            return Err(refuse(format!(
                "{foreign} is not a native transfer class (a process bound already includes host)"
            )));
        }
        self.map_native(bounds, |class| {
            by_class
                .get(class)
                .map(|&bytes| i128::from(bytes))
                .ok_or_else(|| refuse(format!("no source-derived host upper for {class}")))
        })
    }

    fn map_native(
        &self,
        bounds: &[IosWindowBound],
        host: impl Fn(&str) -> crate::Result<i128>,
    ) -> crate::Result<Vec<IosWindowBound>> {
        let overhead = i128::from(self.overhead_bytes);
        bounds
            .iter()
            .map(|bound| match bound {
                IosWindowBound::Transfer {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    coordinate,
                    request_class,
                    anchor_candidates,
                } => {
                    let extra = overhead.checked_add(host(request_class)?).ok_or_else(|| {
                        crate::Error::Unsupported("native-to-process delta overflows".into())
                    })?;
                    Ok(IosWindowBound::Transfer {
                        domain: IosMemoryDomain::ProcessGrowth,
                        coordinate: *coordinate,
                        request_class: request_class.clone(),
                        anchor_candidates: anchor_candidates
                            .iter()
                            .map(|candidate| {
                                Ok(IosAnchorCandidate {
                                    delta_growth_upper: candidate
                                        .delta_growth_upper
                                        .checked_add(extra)
                                        .ok_or_else(|| {
                                            crate::Error::Unsupported(
                                                "native-to-process delta overflows".into(),
                                            )
                                        })?,
                                    // A native lower bound is not a process lower bound.
                                    physical_anchor_lower_bound: None,
                                    ..candidate.clone()
                                })
                            })
                            .collect::<crate::Result<Vec<_>>>()?,
                    })
                }
                bound if bound_is_process(bound) => Ok(bound.clone()),
                _ => Err(crate::Error::Unsupported(
                    "native-to-process model maps only native TRANSFER certificates".into(),
                )),
            })
            .collect()
    }
}

fn bound_is_process(bound: &IosWindowBound) -> bool {
    matches!(
        bound,
        IosWindowBound::Transfer {
            domain: IosMemoryDomain::ProcessGrowth,
            ..
        } | IosWindowBound::RawPhysical {
            domain: IosMemoryDomain::ProcessGrowth,
            ..
        }
    )
}

/// The payload-free replay a catalog anchor is priced by (no GenerationRequest carrier): the
/// recorded canonical recipe token and the anchor's typed conditioning roles.
#[derive(Clone, Copy, Debug)]
pub struct IosReplay<'a> {
    pub token: &'a str,
    pub true_reference_count: u32,
    pub mask_count: u32,
    pub control_count: u32,
    /// The anchor's recorded provider-typed source facts, when its provider records them.
    pub source: Option<IosReplaySource<'a>>,
}

/// Opaque provider-typed source facts recorded with a payload-free anchor (for SDXL, its
/// `SamplingSourceFacts`): only that provider's bounder downcasts them, and a mismatched type
/// refuses there. Recorded capture metadata only; never constructed from a request or payload.
#[derive(Clone, Copy)]
pub struct IosReplaySource<'a>(pub &'a (dyn std::any::Any + Send + Sync));

impl std::fmt::Debug for IosReplaySource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IosReplaySource(..)")
    }
}

/// One side of a transfer: its pricing request plus, for a payload-free anchor, its replay facts.
/// A bounder resolves its execution plan from `pricing.generation` when present, else from the
/// replay token; neither absent refuses.
#[derive(Clone, Copy, Debug)]
pub struct IosBoundRequest<'a> {
    pub pricing: &'a IosPricingRequest<'a>,
    pub replay: Option<IosReplay<'a>>,
}

/// A provider bounder bound to explicit execution facts. Returning `Err` refuses the request; a
/// provider that cannot certify the stated facts must refuse exactly, not return partial bounds.
/// Premise-dependent bounds MUST name their premises; they are consumed only under an explicit
/// policy approving every named premise.
pub type IosCertifiedBounder = fn(
    anchor_request: &IosBoundRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosBoundRequest<'_>,
    request_pricing: &IosPricing,
    execution: &IosExecutionFacts,
) -> crate::Result<IosCertifiedBounds>;

/// Actual boundary snapshot, supplied by SDK/capture/test caller at the declared point, NEVER
/// read at an arbitrary time by a pure pricing/normalization helper. Bytes alone do not prove
/// inventory cancellation; provider code must validate source-compatible owner/birth facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosMemoryBaseline {
    pub domain: IosMemoryDomain,
    pub coordinate: IosMemoryCoordinate,
    pub bytes: u64,
    /// Actual counter/query method and boundary provenance, not a human approval marker.
    pub source: String,
}

/// Every potentially binding candidate plus a backwards-compatible phase winner summary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IosPricing {
    pub phases: Vec<IosPhasePrice>,
    pub windows: Vec<IosWindowPrice>,
    /// Bounded actual capture witnesses, attached by catalog/CPU capture code to ANCHOR pricing.
    /// Analytic providers leave this empty; pure planned replay must never populate it.
    pub observations: Vec<IosAnchorObservation>,
    /// Real REQUEST boundary fact. Anchor baselines remain per-capture in `observations` so
    /// differing repeat baselines are never collapsed into a fabricated single zero/value.
    pub request_baseline: Option<IosMemoryBaseline>,
    /// Actual execution facts for phases that do not run (e.g. no init encode / zero denoise).
    /// Never fill this automatically just because an expected window was forgotten.
    pub absent_phases: std::collections::BTreeMap<IosPhase, String>,
}

impl IosPricing {
    /// The phase's price, if the request runs it.
    pub fn phase(&self, phase: IosPhase) -> Option<&IosPhasePrice> {
        self.phases.iter().find(|p| p.phase == phase)
    }

    /// The whole-run counted peak: phases run in sequence, so the worst phase.
    pub fn peak_bytes(&self) -> u64 {
        self.phases
            .iter()
            .map(IosPhasePrice::peak_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Fold one quantum into its phase, keeping the phase's worst (largest resident + scratch)
    /// quantum. Providers build their pricing with this so a phase is always its worst quantum.
    pub fn take(&mut self, phase: IosPhase, resident_bytes: u64, scratch_bytes: u64, label: &str) {
        self.windows.push(IosWindowPrice {
            phase,
            class: None,
            label: label.into(),
            resident_bytes,
            held_bytes: 0,
            scratch_bytes,
            basis: None,
            owner_domain: String::new(),
        });
        self.take_summary(phase, resident_bytes, scratch_bytes, label);
    }

    /// Preserve the WHOLE candidate before reducing the compatibility summary. Held owners are
    /// folded into the summary's resident bucket; guard scratch retains its original meaning.
    pub fn take_window(&mut self, window: IosWindowPrice) {
        self.take_summary(
            window.phase,
            window.resident_bytes.saturating_add(window.held_bytes),
            window.scratch_bytes,
            &window.label,
        );
        self.windows.push(window);
    }

    pub fn phase_not_run(&mut self, phase: IosPhase, reason: &str) {
        self.absent_phases.insert(phase, reason.into());
    }

    fn take_summary(
        &mut self,
        phase: IosPhase,
        resident_bytes: u64,
        scratch_bytes: u64,
        label: &str,
    ) {
        let candidate = IosPhasePrice {
            phase,
            resident_bytes,
            scratch_bytes,
            binding: label.to_string(),
        };
        match self.phases.iter_mut().find(|p| p.phase == phase) {
            Some(existing) if existing.peak_bytes() >= candidate.peak_bytes() => {}
            Some(existing) => *existing = candidate,
            None => {
                self.phases.push(candidate);
                self.phases.sort_by_key(|p| p.phase);
            }
        }
    }
}

/// What a provider prices: the request's load (tier, components), route, geometry, the selected
/// ladder parameters and the execution levers actually written onto the request, and the encoded
/// text window. Exactly the memory-relevant facts of the iOS evidence key.
#[derive(Clone, Copy, Debug)]
pub struct IosPricingRequest<'a> {
    /// The registered memory-strategy provider id (e.g. `sana_sprint_1600m`, `iris_3b_restore`).
    pub provider: &'a str,
    pub spec: &'a LoadSpec,
    pub mode: &'a MemoryMode,
    pub geometry: MemoryGeometry,
    pub selection: &'a MemorySelection,
    pub execution: &'a GenerationMemory,
    /// Actual post-caller-policy request, or a faithful capture replay. Dynamic sampling providers
    /// must reject missing context; header/default diagnostics construct an explicit request.
    /// Fixed-provider laws may omit it only when their price genuinely does not depend on it.
    pub generation: Option<&'a crate::gen_core::GenerationRequest>,
    /// Actual transform request when transform controls affect host/tensor liveness. Mutually
    /// exclusive with `generation`; fixed-provider diagnostics may omit both only explicitly.
    pub transform: Option<&'a crate::gen_core::TransformRequest>,
    /// Actual Iris depth request, including original source and resolution projection.
    pub depth: Option<&'a crate::gen_core::iris::depth::DepthRequest>,
    /// Encoded text window in tokens where it is request-dependent (SDXL CLIP windows x 77); 0
    /// where the provider pads to a fixed, artifact-determined window.
    pub token_window: u32,
}

impl IosPricingRequest<'_> {
    /// Context carriers are mutually exclusive. Absence is permitted only for explicitly fixed
    /// diagnostic laws; dynamic providers separately require their actual matching request.
    pub fn validate_work_context(&self) -> crate::Result<()> {
        let count = u8::from(self.generation.is_some())
            + u8::from(self.transform.is_some())
            + u8::from(self.depth.is_some());
        if count > 1 {
            return Err(crate::Error::Unsupported(
                "iOS pricing work contexts are mutually exclusive".into(),
            ));
        }
        Ok(())
    }
}

/// A provider's pricing entry point.
pub type IosPricer = fn(&IosPricingRequest<'_>) -> crate::Result<IosPricing>;
/// A provider's quantum-label to phase attribution.
pub type IosQuantumPhase = fn(&str) -> Option<IosPhase>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_keeps_its_worst_quantum_and_the_run_its_worst_phase() {
        let mut p = IosPricing::default();
        p.take(IosPhase::Decode, 100, 10, "a");
        p.take(IosPhase::Decode, 100, 50, "b");
        p.take(IosPhase::Decode, 100, 20, "c");
        p.take(IosPhase::Denoise, 300, 1, "d");
        assert_eq!(p.phase(IosPhase::Decode).unwrap().binding, "b");
        assert_eq!(p.peak_bytes(), 301);
        assert_eq!(
            p.phases.iter().map(|p| p.phase).collect::<Vec<_>>(),
            [IosPhase::Denoise, IosPhase::Decode]
        );
        assert!(p.phase(IosPhase::Encode).is_none());
        for phase in IosPhase::ALL {
            assert_eq!(IosPhase::from_key(phase.as_key()), Some(phase));
        }
    }
}

#[cfg(test)]
mod native_to_process_tests {
    use super::*;

    fn transfer(domain: IosMemoryDomain, delta: i128) -> IosWindowBound {
        IosWindowBound::Transfer {
            domain,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            request_class: "r".into(),
            anchor_candidates: vec![IosAnchorCandidate {
                anchor_class: "a".into(),
                delta_growth_upper: delta,
                physical_anchor_lower_bound: Some(7),
                witnesses: vec![IosAnchorWitness {
                    capture: 0,
                    check: 0,
                }],
            }],
        }
    }

    /// HOST: the model states EXACTLY the anchor-relative delta premise it applies (a per-run
    /// process <= native bound would not justify transforming a signed delta: anchor 1000/100,
    /// request 1100/1100 MiB satisfy it yet the native delta 100 under-bounds the process delta
    /// 1000). Native transfers gain the stated overhead and lose their native lower bound;
    /// process bounds pass unchanged; a native raw total is refused.
    #[test]
    fn the_native_to_process_model_states_and_applies_the_anchor_relative_premise() {
        let model = IosNativeToProcessModel {
            overhead_bytes: 8,
            provenance: "test".into(),
        };
        let premise = model.premise();
        assert_eq!(premise.id, NATIVE_TO_PROCESS_PREMISE_ID);
        assert!(premise
            .statement
            .contains("process_growth(request) - process_growth(anchor)"));
        assert!(premise
            .statement
            .contains("native_growth(request) - native_growth(anchor) + 8 B"));
        let mapped = model
            .to_process(&[transfer(IosMemoryDomain::NativeAllocatorGrowth, -3)])
            .unwrap();
        let IosWindowBound::Transfer {
            domain,
            anchor_candidates,
            ..
        } = &mapped[0]
        else {
            panic!("transfer expected")
        };
        assert_eq!(*domain, IosMemoryDomain::ProcessGrowth);
        assert_eq!(anchor_candidates[0].delta_growth_upper, 5);
        assert_eq!(anchor_candidates[0].physical_anchor_lower_bound, None);
        let process = transfer(IosMemoryDomain::ProcessGrowth, 4);
        assert_eq!(
            model.to_process(std::slice::from_ref(&process)).unwrap(),
            vec![process]
        );
        let raw = IosWindowBound::RawPhysical {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            request_class: "r".into(),
            growth_upper: 1,
        };
        assert!(model.to_process(&[raw]).is_err());
    }

    /// HOST: a DECLARED source-derived host term is ADDED to each native transfer (native delta
    /// and fixed overhead retained); missing, duplicate and foreign entries refuse rather than
    /// zero; an undeclared claim is exactly the original premise; process bounds never gain host;
    /// and the host-extended premise is a distinct id with its own statement.
    #[test]
    fn the_host_extended_model_adds_only_declared_source_host_growth() {
        let model = IosNativeToProcessModel {
            overhead_bytes: 8,
            provenance: "test".into(),
        };
        let host = model.host_premise();
        assert_eq!(host.id, NATIVE_TO_PROCESS_HOST_PREMISE_ID);
        assert_ne!(host.id, NATIVE_TO_PROCESS_PREMISE_ID);
        assert!(host.statement.contains("os_stack_request(request) + 8 B"));
        let native = transfer(IosMemoryDomain::NativeAllocatorGrowth, -3);
        let delta = |bounds: &[IosWindowBound]| match &bounds[0] {
            IosWindowBound::Transfer {
                domain: IosMemoryDomain::ProcessGrowth,
                anchor_candidates,
                ..
            } => anchor_candidates[0].delta_growth_upper,
            other => panic!("{other:?}"),
        };
        let entry = |class: &str, bytes| IosHostGrowthBound {
            request_class: class.into(),
            growth_upper: bytes,
            os_stack_request_bytes: 0,
        };
        let declared = IosHostGrowth::Declared(vec![entry("r", 100)]);
        let slice = std::slice::from_ref(&native);
        assert_eq!(
            delta(&model.to_process_with_host(slice, &declared).unwrap()),
            105
        );
        assert_eq!(
            model
                .to_process_with_host(slice, &IosHostGrowth::Undeclared)
                .unwrap(),
            model.to_process(slice).unwrap()
        );
        assert!(model
            .to_process_with_host(slice, &IosHostGrowth::Declared(vec![]))
            .is_err());
        assert!(model
            .to_process_with_host(
                slice,
                &IosHostGrowth::Declared(vec![entry("r", 1), entry("r", 2)])
            )
            .is_err());
        assert!(model
            .to_process_with_host(
                slice,
                &IosHostGrowth::Declared(vec![entry("r", 1), entry("other", 2)])
            )
            .is_err());
        let process = transfer(IosMemoryDomain::ProcessGrowth, 4);
        assert!(model
            .to_process_with_host(
                std::slice::from_ref(&process),
                &IosHostGrowth::Declared(vec![entry("r", 1)])
            )
            .is_err());
        let huge = IosHostGrowth::Declared(vec![entry("r", u64::MAX)]);
        let top = transfer(IosMemoryDomain::NativeAllocatorGrowth, i128::MAX - 1);
        assert!(model
            .to_process_with_host(std::slice::from_ref(&top), &huge)
            .is_err());
        let unproved = IosHostGrowth::Unproved("tokenizer upper unproved".into());
        let refusal = model.to_process_with_host(slice, &unproved).unwrap_err();
        assert!(refusal.to_string().contains("tokenizer upper unproved"));
        let certified =
            IosCertifiedBounds::proven_with_host(vec![native.clone(), native.clone()], 9, 0);
        assert_eq!(
            certified.host,
            IosHostGrowth::Declared(vec![entry("r", 9)]),
            "one entry per distinct native class"
        );
        // The requested OS stack is a separate field that the mapping adds beside the heap term.
        let with_stack = IosHostGrowth::Declared(vec![IosHostGrowthBound {
            request_class: "r".into(),
            growth_upper: 100,
            os_stack_request_bytes: 1000,
        }]);
        assert_eq!(
            delta(&model.to_process_with_host(slice, &with_stack).unwrap()),
            1105
        );
    }
}

#[cfg(test)]
mod source_capacity_bridge_tests {
    use super::with_core_source_capacity;
    use crate::Error;

    #[test]
    fn original_callback_error_wins_even_if_core_work_recovers() {
        let mut calls = 0;
        let mut next = |_| {
            calls += 1;
            Err(Error::GeometryRefused {
                reason: "original callback".into(),
                requested_width: 17,
                requested_height: 23,
                alternative: Some((11, 13)),
            })
        };
        let result = with_core_source_capacity(&mut next, |core_next| {
            assert!(core_next(3).is_err());
            assert!(core_next(5).is_err());
            Ok(7u8)
        });
        assert_eq!(
            calls, 1,
            "a denied check stays sticky without re-invoking its owner"
        );
        match result.unwrap_err() {
            Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            } => {
                assert_eq!(reason, "original callback");
                assert_eq!(
                    (requested_width, requested_height, alternative),
                    (17, 23, Some((11, 13)))
                );
            }
            other => panic!("callback error changed domain: {other:?}"),
        }
    }

    #[test]
    fn core_and_work_io_errors_keep_their_typed_domains() {
        let mut next = |_| Ok(());
        let result: crate::Result<()> = with_core_source_capacity(&mut next, |core_next| {
            core_next(3)?;
            Err(crate::gen_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "original core IO",
            ))
            .into())
        });
        match result.unwrap_err() {
            Error::Io(error) => {
                assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
                assert_eq!(error.to_string(), "original core IO");
            }
            other => panic!("core error changed domain: {other:?}"),
        }
        let result: crate::Result<()> =
            with_core_source_capacity(&mut next, |_| Err(Error::Canceled));
        assert!(matches!(result, Err(Error::Canceled)));
    }

    #[test]
    fn successful_core_checks_reborrow_the_same_serial_callback() {
        let mut calls = Vec::new();
        let mut next = |bytes| {
            calls.push(bytes);
            Ok(())
        };
        let value = with_core_source_capacity(&mut next, |core_next| {
            core_next(3)?;
            core_next(5)?;
            Ok(7u8)
        })
        .unwrap();
        assert_eq!(value, 7);
        assert_eq!(calls, [3, 5]);
    }
}

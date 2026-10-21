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
}

impl IosCertifiedBounds {
    /// Fully source-proved bounds (no premise).
    pub fn proven(bounds: Vec<IosWindowBound>) -> Self {
        Self {
            bounds,
            premises: Vec::new(),
        }
    }
}

/// The stable id of the activity-26569 native-to-process premise.
pub const NATIVE_TO_PROCESS_PREMISE_ID: &str = "activity-26569/target-device-native-to-process";

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

    /// Map native-allocator TRANSFER certificates into the process domain. Process-domain bounds
    /// pass unchanged; any other domain, a raw native bound, or overflow refuses: a native raw
    /// total has no anchor-relative form this model is stated for.
    pub fn to_process(&self, bounds: &[IosWindowBound]) -> crate::Result<Vec<IosWindowBound>> {
        let overhead = i128::from(self.overhead_bytes);
        bounds
            .iter()
            .map(|bound| match bound {
                IosWindowBound::Transfer {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    coordinate,
                    request_class,
                    anchor_candidates,
                } => Ok(IosWindowBound::Transfer {
                    domain: IosMemoryDomain::ProcessGrowth,
                    coordinate: *coordinate,
                    request_class: request_class.clone(),
                    anchor_candidates: anchor_candidates
                        .iter()
                        .map(|candidate| {
                            Ok(IosAnchorCandidate {
                                delta_growth_upper: candidate
                                    .delta_growth_upper
                                    .checked_add(overhead)
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
                }),
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
}

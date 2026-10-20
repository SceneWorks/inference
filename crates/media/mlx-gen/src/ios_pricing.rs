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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IosMemoryDomain {
    Unspecified,
    ProcessGrowth,
    NativeAllocatorGrowth,
    NativeAllocatorAbsolute,
}
impl Default for IosMemoryDomain {
    fn default() -> Self {
        Self::Unspecified
    }
}

/// Baseline coordinate is part of the certificate, not an implicit zero/foreign-inventory credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IosMemoryCoordinate {
    Unspecified,
    Absolute,
    BeforeRequestConstruction,
    BeforeModelLoad,
    WindowEntry,
}
impl Default for IosMemoryCoordinate {
    fn default() -> Self {
        Self::Unspecified
    }
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

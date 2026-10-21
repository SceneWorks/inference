//! Correlated, source-certified iOS anchor transfer (epic 16774).
//!
//! For each request window, provider CODE proves at every corresponding instant
//! `F_request - F_anchor <= delta_growth_upper`. The consumer takes the MAX of
//! `U_anchor + delta` over ALL plausible covered candidates, then the MAX over
//! request windows. Logical resident/held/scratch uppers are neither physical
//! lower bounds nor certificates. Matching labels, counts or endpoint results
//! never create a transfer. There is no fitted ratio, area discount, independent
//! maxima subtraction, or borrowed overhead for an unobserved phase.
//!
//! All certified entrypoints return an UNMARGINED upper in the named coordinate.
//! Process admission uses ProcessGrowth, never a native-only/absolute proof, and
//! adds one explicit `with_margin` after complete coverage/identity validation.
//! Capture/catalog code must validate physical baseline compatibility before
//! constructing `MeasuredAnchor`. Its phase readings may be lifetime highwater
//! uppers, not binding measurements: transfers conservatively use the whole-run
//! physical upper for EVERY window, including gaps in phase attribution.

use crate::ios_pricing::{
    IosMemoryCoordinate, IosMemoryDomain, IosPhase, IosPricing, IosWindowBound, IosWindowPrice,
};
use std::collections::{BTreeMap, BTreeSet};

/// Historical endpoint diagnostic threshold, not proof of a transfer's soundness.
pub const CONSISTENCY_TOLERANCE: f64 = 0.126;

/// Validated upper in the explicit counter domain and baseline coordinate.
/// Growth is normalized over that capture's compatible starting footprint;
/// `NativeAllocatorAbsolute/Absolute` is explicitly absolute, never inferred
/// from a zero baseline. A masked lifetime peak is not an exact measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredAnchor {
    pub domain: IosMemoryDomain,
    pub coordinate: IosMemoryCoordinate,
    pub demand: u64,
    /// Diagnostic physical uppers only; not assumed to coincide with logical winners.
    pub phases: BTreeMap<IosPhase, u64>,
    /// Diagnostic geometry; never used to infer a physical scaling exponent.
    pub area: u64,
}

/// Output pixels, retained for caller diagnostics (not a transfer certificate).
pub fn area(width: u32, height: u32, batch: u32) -> u64 {
    u64::from(width) * u64::from(height) * u64::from(batch.max(1))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotDerivable {
    Uncertified,
    DomainMismatch,
    Coverage(String),
    InvalidProfile(String),
    InvalidWitness(String),
    InvalidMeasurement,
    Arithmetic,
}

impl std::fmt::Display for NotDerivable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uncertified => f.write_str("no source-certified physical window bounds"),
            Self::DomainMismatch => {
                f.write_str("unspecified or incompatible iOS memory-counter domain")
            }
            Self::Coverage(s) => write!(f, "incomplete iOS window coverage: {s}"),
            Self::InvalidProfile(s) => write!(f, "invalid iOS window profile: {s}"),
            Self::InvalidWitness(s) => write!(f, "invalid actual iOS observation: {s}"),
            Self::InvalidMeasurement => f.write_str("invalid physical growth measurement"),
            Self::Arithmetic => f.write_str("physical growth bound is negative or overflows"),
        }
    }
}
impl std::error::Error for NotDerivable {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedPhase {
    pub domain: IosMemoryDomain,
    pub coordinate: IosMemoryCoordinate,
    pub phase: IosPhase,
    pub bytes: u64,
    /// Compatibility field: no logical scratch calibration is performed.
    pub calibrated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Derivation {
    pub domain: IosMemoryDomain,
    pub coordinate: IosMemoryCoordinate,
    pub demand: u64,
    pub phases: Vec<DerivedPhase>,
}

/// Add the explicit admission margin ONCE, after reducing the complete bound.
pub fn with_margin(growth_upper: u64, margin: u64) -> Result<u64, NotDerivable> {
    growth_upper
        .checked_add(margin)
        .ok_or(NotDerivable::Arithmetic)
}

/// Lifetime-kernel highwater provenance, kept distinct from an exact run peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelPeakQuality {
    AdvancedDuringRun,
    CurrentAttainedUpper,
    MaskedUpper,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelGrowth {
    pub starting_current: u64,
    pub starting_highwater: u64,
    pub ending_highwater: u64,
    pub run_current_peak: u64,
    pub query_count: u64,
    pub growth_upper: u64,
    pub quality: KernelPeakQuality,
}

impl KernelGrowth {
    /// Inputs are actual same-task kernel queries, including the starting current
    /// footprint. Query validity/provenance is the capture owner's responsibility.
    pub fn from_queries(
        starting_current: u64,
        starting_highwater: u64,
        ending_highwater: u64,
        run_current_peak: u64,
        query_count: u64,
    ) -> Result<Self, NotDerivable> {
        if query_count == 0
            || starting_highwater < starting_current
            || ending_highwater < starting_highwater
            || run_current_peak < starting_current
            || run_current_peak > ending_highwater
        {
            return Err(NotDerivable::InvalidMeasurement);
        }
        let quality = if ending_highwater > starting_highwater {
            KernelPeakQuality::AdvancedDuringRun
        } else if run_current_peak == ending_highwater {
            KernelPeakQuality::CurrentAttainedUpper
        } else {
            KernelPeakQuality::MaskedUpper
        };
        Ok(Self {
            starting_current,
            starting_highwater,
            ending_highwater,
            run_current_peak,
            query_count,
            growth_upper: ending_highwater
                .checked_sub(starting_current)
                .ok_or(NotDerivable::Arithmetic)?,
            quality,
        })
    }

    /// Only an actually attained physical upper is exact. Complete request-key
    /// identity is separately checked by the catalog, never by logical DTO equality.
    pub fn exact_growth(&self) -> Result<u64, NotDerivable> {
        if self.quality == KernelPeakQuality::MaskedUpper {
            Err(NotDerivable::InvalidMeasurement)
        } else {
            Ok(self.growth_upper)
        }
    }
}

fn profile_class(window: &IosWindowPrice) -> Result<&str, NotDerivable> {
    window
        .profile_class()
        .ok_or_else(|| NotDerivable::InvalidProfile(window.label.clone()))
}

fn bound_class(bound: &IosWindowBound) -> &str {
    match bound {
        IosWindowBound::Transfer { request_class, .. }
        | IosWindowBound::RawPhysical { request_class, .. } => request_class,
    }
}

fn validate_measurement_coordinate(
    domain: IosMemoryDomain,
    coordinate: IosMemoryCoordinate,
) -> Result<(), NotDerivable> {
    if domain == IosMemoryDomain::Unspecified
        || coordinate == IosMemoryCoordinate::Unspecified
        || (domain == IosMemoryDomain::NativeAllocatorAbsolute)
            != (coordinate == IosMemoryCoordinate::Absolute)
    {
        Err(NotDerivable::DomainMismatch)
    } else {
        Ok(())
    }
}

fn bound_coordinate(bound: &IosWindowBound) -> IosMemoryCoordinate {
    match bound {
        IosWindowBound::Transfer { coordinate, .. }
        | IosWindowBound::RawPhysical { coordinate, .. } => *coordinate,
    }
}

fn bound_domain(bound: &IosWindowBound) -> IosMemoryDomain {
    match bound {
        IosWindowBound::Transfer { domain, .. } | IosWindowBound::RawPhysical { domain, .. } => {
            *domain
        }
    }
}

fn bound_for<'a>(
    bounds: &'a [IosWindowBound],
    class: &str,
) -> Result<&'a IosWindowBound, NotDerivable> {
    let mut found = bounds.iter().filter(|bound| bound_class(bound) == class);
    let bound = found
        .next()
        .ok_or_else(|| NotDerivable::Coverage(class.into()))?;
    if found.next().is_some() {
        return Err(NotDerivable::Coverage(format!(
            "duplicate certificate for {class}"
        )));
    }
    Ok(bound)
}

fn request_windows<'a>(
    request: &'a IosPricing,
    class: &str,
) -> Result<Vec<&'a IosWindowPrice>, NotDerivable> {
    let windows: Vec<_> = request
        .windows
        .iter()
        .filter(|window| window.class.as_deref() == Some(class))
        .collect();
    if windows.is_empty() {
        return Err(NotDerivable::Coverage(class.into()));
    }
    for window in &windows {
        profile_class(window)?;
        if request.absent_phases.contains_key(&window.phase) {
            return Err(NotDerivable::InvalidProfile(format!(
                "{} both runs and is absent",
                window.phase.as_key()
            )));
        }
    }
    // A class may cover repeated invocations, but not unrelated phases.
    if windows
        .iter()
        .any(|window| window.phase != windows[0].phase)
    {
        return Err(NotDerivable::InvalidProfile(format!(
            "class {class} spans phases"
        )));
    }
    Ok(windows)
}

fn candidate_upper(
    anchor: &MeasuredAnchor,
    at_anchor: &IosPricing,
    candidate: &crate::ios_pricing::IosAnchorCandidate,
) -> Result<u64, NotDerivable> {
    if anchor.phases.values().any(|upper| *upper > anchor.demand) {
        return Err(NotDerivable::InvalidMeasurement);
    }
    let windows: Vec<_> = at_anchor
        .windows
        .iter()
        .filter(|window| window.class.as_deref() == Some(candidate.anchor_class.as_str()))
        .collect();
    if windows.is_empty() || candidate.witnesses.is_empty() {
        return Err(NotDerivable::Coverage(candidate.anchor_class.clone()));
    }
    for window in &windows {
        profile_class(window)?;
        if at_anchor.absent_phases.contains_key(&window.phase) {
            return Err(NotDerivable::InvalidProfile(candidate.anchor_class.clone()));
        }
    }
    if windows
        .iter()
        .any(|window| window.phase != windows[0].phase)
    {
        return Err(NotDerivable::InvalidProfile(candidate.anchor_class.clone()));
    }
    for witness in &candidate.witnesses {
        let observation = at_anchor
            .observations
            .get(witness.capture as usize)
            .ok_or_else(|| NotDerivable::InvalidWitness(candidate.anchor_class.clone()))?;
        if observation.domain != anchor.domain || observation.coordinate != anchor.coordinate {
            return Err(NotDerivable::DomainMismatch);
        }
        let check = observation
            .checks
            .get(witness.check as usize)
            .ok_or_else(|| NotDerivable::InvalidWitness(candidate.anchor_class.clone()))?;
        // These are integrity checks, NOT the proof of correspondence. That proof
        // belongs to the source callback which supplied this certificate.
        if !windows.iter().any(|window| {
            window.label == check.label && check.phase.is_none_or(|phase| window.phase == phase)
        }) || check.context.is_empty()
        {
            return Err(NotDerivable::InvalidWitness(candidate.anchor_class.clone()));
        }
        let mut seen = BTreeSet::new();
        for index in &check.context {
            let metadata = observation
                .metadata
                .get(*index as usize)
                .ok_or_else(|| NotDerivable::InvalidWitness(candidate.anchor_class.clone()))?;
            if metadata.provider.is_empty() || metadata.kind.is_empty() || !seen.insert(*index) {
                return Err(NotDerivable::InvalidWitness(candidate.anchor_class.clone()));
            }
        }
    }
    // Never subtract a logical upper. The optional lower is source-proved and
    // already accounted for in the signed delta; here it is only checked for
    // consistency with the physical upper in the same baseline domain.
    if candidate
        .physical_anchor_lower_bound
        .is_some_and(|lower| lower > anchor.demand)
    {
        return Err(NotDerivable::InvalidMeasurement);
    }
    let sum = i128::from(anchor.demand)
        .checked_add(candidate.delta_growth_upper)
        .ok_or(NotDerivable::Arithmetic)?;
    u64::try_from(sum).map_err(|_| NotDerivable::Arithmetic)
}

/// Bound ONE certified class only. This intentionally makes no claim about other
/// classes, a complete phase, or admission. Provider native forward proofs use it
/// while sampler/host windows remain separately unproved.
pub fn derive_window_certified(
    anchor: &MeasuredAnchor,
    at_anchor: &IosPricing,
    request: &IosPricing,
    bounds: &[IosWindowBound],
    request_class: &str,
) -> Result<u64, NotDerivable> {
    validate_measurement_coordinate(anchor.domain, anchor.coordinate)?;
    request_windows(request, request_class)?;
    let bound = bound_for(bounds, request_class)?;
    if bound_domain(bound) != anchor.domain || bound_coordinate(bound) != anchor.coordinate {
        return Err(NotDerivable::DomainMismatch);
    }
    match bound {
        IosWindowBound::RawPhysical { growth_upper, .. } => Ok(*growth_upper),
        IosWindowBound::Transfer {
            anchor_candidates, ..
        } => {
            if anchor_candidates.is_empty() {
                return Err(NotDerivable::Coverage(request_class.into()));
            }
            let mut upper = None;
            for candidate in anchor_candidates {
                let bytes = candidate_upper(anchor, at_anchor, candidate)?;
                upper = Some(upper.map_or(bytes, |previous: u64| previous.max(bytes)));
            }
            upper.ok_or_else(|| NotDerivable::Coverage(request_class.into()))
        }
    }
}

/// Bound EVERY window of one phase. Missing phases are not zero: only an
/// explicit actual `phase_not_run` execution fact permits a zero result.
pub fn derive_phase_certified(
    anchor: &MeasuredAnchor,
    at_anchor: &IosPricing,
    request: &IosPricing,
    bounds: &[IosWindowBound],
    phase: IosPhase,
) -> Result<DerivedPhase, NotDerivable> {
    validate_measurement_coordinate(anchor.domain, anchor.coordinate)?;
    let mut classes = BTreeSet::new();
    for window in request
        .windows
        .iter()
        .filter(|window| window.phase == phase)
    {
        classes.insert(profile_class(window)?);
    }
    if classes.is_empty() {
        if request.phases.iter().any(|price| price.phase == phase)
            || request
                .absent_phases
                .get(&phase)
                .is_none_or(|reason| reason.is_empty())
        {
            return Err(NotDerivable::Coverage(phase.as_key().into()));
        }
        return Ok(DerivedPhase {
            domain: anchor.domain,
            coordinate: anchor.coordinate,
            phase,
            bytes: 0,
            calibrated: false,
        });
    }
    if request.absent_phases.contains_key(&phase) {
        return Err(NotDerivable::InvalidProfile(format!(
            "{} both runs and is absent",
            phase.as_key()
        )));
    }
    let mut bytes = 0;
    for class in classes {
        bytes = bytes.max(derive_window_certified(
            anchor, at_anchor, request, bounds, class,
        )?);
    }
    Ok(DerivedPhase {
        domain: anchor.domain,
        coordinate: anchor.coordinate,
        phase,
        bytes,
        calibrated: false,
    })
}

/// Complete unmargined request bound. Every phase must have certified windows or
/// an explicit actual absence. Duplicate certificates cannot stand in for missing
/// classes. Extra certificates for nonexistent classes are also rejected.
pub fn derive_certified(
    anchor: &MeasuredAnchor,
    at_anchor: &IosPricing,
    request: &IosPricing,
    bounds: &[IosWindowBound],
) -> Result<Derivation, NotDerivable> {
    if request.windows.is_empty() {
        return Err(NotDerivable::Coverage("empty request window set".into()));
    }
    for bound in bounds {
        request_windows(request, bound_class(bound))?;
    }
    let phases: Vec<_> = IosPhase::ALL
        .into_iter()
        .map(|phase| derive_phase_certified(anchor, at_anchor, request, bounds, phase))
        .collect::<Result<_, _>>()?;
    let demand = phases
        .iter()
        .map(|phase| phase.bytes)
        .max()
        .ok_or_else(|| NotDerivable::Coverage("empty request".into()))?;
    Ok(Derivation {
        domain: anchor.domain,
        coordinate: anchor.coordinate,
        demand,
        phases,
    })
}

/// Legacy logical pricing has no proof authority. Kept source-compatible while
/// callers migrate to provider callbacks; it cannot silently reuse the old law.
pub fn derive(
    _anchor: &MeasuredAnchor,
    _at_anchor: &IosPricing,
    _request: &IosPricing,
    _request_area: u64,
) -> Result<Derivation, NotDerivable> {
    Err(NotDerivable::Uncertified)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Consistency {
    pub predicted: u64,
    pub measured: u64,
    pub relative_error: f64,
}
impl Consistency {
    pub fn flagged(&self) -> bool {
        self.relative_error.abs() > CONSISTENCY_TOLERANCE
    }
    /// Endpoint disagreement diagnoses a defect; endpoint agreement is not proof.
    pub fn fails_closed(&self) -> bool {
        self.predicted < self.measured
    }
}

/// Legacy consistency cannot invent source certificates.
pub fn consistency(
    _optimized: &MeasuredAnchor,
    _at_optimized: &IosPricing,
    _resident: &MeasuredAnchor,
    _at_resident: &IosPricing,
) -> Result<Consistency, NotDerivable> {
    Err(NotDerivable::Uncertified)
}

/// Endpoint diagnostic only, using the SAME complete source-certified consumer.
pub fn consistency_certified(
    optimized: &MeasuredAnchor,
    at_optimized: &IosPricing,
    resident: &MeasuredAnchor,
    at_resident: &IosPricing,
    bounds: &[IosWindowBound],
) -> Result<Consistency, NotDerivable> {
    if optimized.domain != resident.domain || optimized.coordinate != resident.coordinate {
        return Err(NotDerivable::DomainMismatch);
    }
    let predicted = derive_certified(optimized, at_optimized, at_resident, bounds)?.demand;
    let measured = resident.demand;
    Ok(Consistency {
        predicted,
        measured,
        relative_error: if measured == 0 {
            if predicted == 0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (predicted as f64 - measured as f64) / measured as f64
        },
    })
}

/// MLX CPU-fixture measurements, not physical-device evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredRun {
    pub demand: u64,
    pub phases: BTreeMap<IosPhase, u64>,
}
impl MeasuredRun {
    pub fn anchor(&self, area: u64) -> MeasuredAnchor {
        MeasuredAnchor {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            demand: self.demand,
            phases: self.phases.clone(),
            area,
        }
    }
}

/// Measure a tiny CPU fixture's MLX active-memory intervals for provider proofs.
/// This existing observer is NOT a kernel-physical capture.
#[cfg(not(target_os = "ios"))]
pub fn measure_phases<T>(
    phase_of: crate::ios_pricing::IosQuantumPhase,
    body: impl FnOnce() -> crate::Result<T>,
) -> crate::Result<(T, MeasuredRun)> {
    use crate::ios_quantum::QuantumEvent;
    use std::cell::RefCell;
    use std::rc::Rc;
    crate::residency::drain_allocator_cache();
    let baseline = mlx_rs::memory::get_active_memory() as u64;
    mlx_rs::memory::reset_peak_memory();
    type State = (Option<IosPhase>, bool, BTreeMap<IosPhase, u64>);
    let state: Rc<RefCell<State>> = Rc::new(RefCell::new((None, false, BTreeMap::new())));
    let peak_now = || {
        (mlx_rs::memory::get_peak_memory() as u64).max(mlx_rs::memory::get_active_memory() as u64)
    };
    let sink = Rc::clone(&state);
    let record = move |label: &str, opens: bool| {
        let peak = peak_now().saturating_sub(baseline);
        mlx_rs::memory::reset_peak_memory();
        let mut s = sink.borrow_mut();
        let here = phase_of(label);
        let target = if s.1 { s.0 } else { here };
        if let Some(phase) = target {
            let slot = s.2.entry(phase).or_insert(0);
            *slot = (*slot).max(peak);
        }
        s.0 = here;
        s.1 = opens;
    };
    let record = Rc::new(record);
    let observer = {
        let record = Rc::clone(&record);
        move |event: &QuantumEvent| match event {
            QuantumEvent::Checked { label, .. } => record(label, true),
            QuantumEvent::Synchronized { label, .. } => record(label, false),
        }
    };
    let value = crate::ios_quantum::with_cpu_diagnostic_observer(observer, body)?;
    let tail = peak_now().saturating_sub(baseline);
    let mut s = state.borrow_mut();
    if let Some(phase) = s.0 {
        let slot = s.2.entry(phase).or_insert(0);
        *slot = (*slot).max(tail);
    }
    let phases = std::mem::take(&mut s.2);
    let demand = phases.values().copied().max().unwrap_or(0).max(tail);
    Ok((value, MeasuredRun { demand, phases }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ios_pricing::{
        IosAnchorCandidate, IosAnchorObservation, IosAnchorWitness, IosObservedCheck,
        IosObservedMetadata, IosWorkBasis,
    };
    use IosPhase::*;

    // Synthetic certificates test only consumer algebra/integrity. They are NOT
    // provider proofs, device evidence, or authorization for production callbacks.
    fn window(
        class: &str,
        phase: IosPhase,
        resident: u64,
        held: u64,
        scratch: u64,
    ) -> IosWindowPrice {
        IosWindowPrice {
            phase,
            class: Some(class.into()),
            label: format!("op:{class}"),
            resident_bytes: resident,
            held_bytes: held,
            scratch_bytes: scratch,
            basis: Some(IosWorkBasis::default()),
            owner_domain: "synthetic-ledger".into(),
        }
    }
    fn pricing(windows: Vec<IosWindowPrice>) -> IosPricing {
        let mut pricing = IosPricing::default();
        for window in windows {
            pricing.take_window(window);
        }
        for phase in IosPhase::ALL {
            if pricing.phase(phase).is_none() {
                pricing.phase_not_run(phase, "synthetic actual omission");
            }
        }
        pricing
    }
    fn observe(pricing: &mut IosPricing) {
        let checks = pricing
            .windows
            .iter()
            .map(|window| IosObservedCheck {
                phase: Some(window.phase),
                label: window.label.clone(),
                context: vec![0],
            })
            .collect();
        pricing.observations.push(IosAnchorObservation {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
            baseline_bytes: 17,
            metadata: vec![IosObservedMetadata {
                provider: "synthetic".into(),
                kind: "actual-test-context".into(),
                fields: BTreeMap::new(),
            }],
            checks,
            telemetry_complete: true,
        });
    }
    fn anchor(demand: u64) -> MeasuredAnchor {
        MeasuredAnchor {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
            demand,
            phases: BTreeMap::new(),
            area: 1,
        }
    }
    fn candidate(class: &str, delta: i128, check: u32) -> IosAnchorCandidate {
        IosAnchorCandidate {
            anchor_class: class.into(),
            delta_growth_upper: delta,
            physical_anchor_lower_bound: None,
            witnesses: vec![IosAnchorWitness { capture: 0, check }],
        }
    }
    fn transfer(class: &str, candidates: Vec<IosAnchorCandidate>) -> IosWindowBound {
        IosWindowBound::Transfer {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
            request_class: class.into(),
            anchor_candidates: candidates,
        }
    }
    fn simple() -> (MeasuredAnchor, IosPricing, IosPricing, Vec<IosWindowBound>) {
        let mut at = pricing(vec![window("a", Denoise, 0, 0, 100)]);
        observe(&mut at);
        let req = pricing(vec![window("r", Denoise, 0, 0, 50)]);
        (
            anchor(80),
            at,
            req,
            vec![transfer("r", vec![candidate("a", -35, 0)])],
        )
    }

    #[test]
    fn raw_ratio_counterexample_uses_certified_delta_not_kappa() {
        let (a, at, req, bounds) = simple();
        // Logical 100 -> 50 and physical 80 -> 45: the old .8 ratio gave 40.
        assert_eq!(derive_certified(&a, &at, &req, &bounds).unwrap().demand, 45);
        assert_eq!(derive(&a, &at, &req, 1), Err(NotDerivable::Uncertified));
    }

    #[test]
    fn native_and_process_domains_never_substitute_for_each_other() {
        let (mut a, mut at, req, mut bounds) = simple();
        let IosWindowBound::Transfer { domain, .. } = &mut bounds[0] else {
            unreachable!()
        };
        *domain = IosMemoryDomain::NativeAllocatorGrowth;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        a.domain = IosMemoryDomain::NativeAllocatorGrowth;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        at.observations[0].domain = IosMemoryDomain::NativeAllocatorGrowth;
        let native = derive_certified(&a, &at, &req, &bounds).unwrap();
        assert_eq!(native.demand, 45);
        assert_eq!(native.domain, IosMemoryDomain::NativeAllocatorGrowth);
        assert!(native
            .phases
            .iter()
            .all(|phase| phase.domain == native.domain));
        let raw = vec![IosWindowBound::RawPhysical {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
            request_class: "r".into(),
            growth_upper: 45,
        }];
        assert_eq!(
            derive_certified(&a, &at, &req, &raw),
            Err(NotDerivable::DomainMismatch)
        );
        a.domain = IosMemoryDomain::Unspecified;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        assert_eq!(
            MeasuredRun {
                demand: 1,
                phases: BTreeMap::new()
            }
            .anchor(1)
            .domain,
            IosMemoryDomain::NativeAllocatorGrowth
        );
    }

    #[test]
    fn baseline_coordinates_are_explicit_and_checked_on_used_observations() {
        let (mut a, mut at, req, mut bounds) = simple();
        let IosWindowBound::Transfer { coordinate, .. } = &mut bounds[0] else {
            unreachable!()
        };
        *coordinate = IosMemoryCoordinate::BeforeModelLoad;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        a.coordinate = IosMemoryCoordinate::BeforeModelLoad;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        at.observations[0].coordinate = IosMemoryCoordinate::BeforeModelLoad;
        at.observations[0].baseline_bytes = 123;
        let result = derive_certified(&a, &at, &req, &bounds).unwrap();
        assert_eq!(result.demand, 45);
        assert_eq!(result.coordinate, IosMemoryCoordinate::BeforeModelLoad);
        assert!(result
            .phases
            .iter()
            .all(|phase| phase.coordinate == result.coordinate));
        assert_eq!(at.observations[0].baseline_bytes, 123); // Never silently treated as zero.
        a.coordinate = IosMemoryCoordinate::Unspecified;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        let IosWindowBound::Transfer { coordinate, .. } = &mut bounds[0] else {
            unreachable!()
        };
        *coordinate = IosMemoryCoordinate::Unspecified;
        at.observations[0].coordinate = IosMemoryCoordinate::Unspecified;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        assert_eq!(
            MeasuredRun {
                demand: 1,
                phases: BTreeMap::new()
            }
            .anchor(1)
            .coordinate,
            IosMemoryCoordinate::BeforeModelLoad
        );
    }

    #[test]
    fn absolute_native_channel_requires_explicit_absolute_coordinate() {
        for (domain, coordinate) in [
            (
                IosMemoryDomain::ProcessGrowth,
                IosMemoryCoordinate::Absolute,
            ),
            (
                IosMemoryDomain::NativeAllocatorAbsolute,
                IosMemoryCoordinate::BeforeModelLoad,
            ),
        ] {
            let (mut a, mut at, req, mut bounds) = simple();
            a.domain = domain;
            a.coordinate = coordinate;
            at.observations[0].domain = domain;
            at.observations[0].coordinate = coordinate;
            let IosWindowBound::Transfer {
                domain: bound_domain,
                coordinate: bound_coordinate,
                ..
            } = &mut bounds[0]
            else {
                unreachable!()
            };
            *bound_domain = domain;
            *bound_coordinate = coordinate;
            assert_eq!(
                derive_certified(&a, &at, &req, &bounds),
                Err(NotDerivable::DomainMismatch)
            );
        }
        let (mut a, mut at, req, mut bounds) = simple();
        a.coordinate = IosMemoryCoordinate::Absolute;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        a.domain = IosMemoryDomain::NativeAllocatorAbsolute;
        a.coordinate = IosMemoryCoordinate::BeforeModelLoad;
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::DomainMismatch)
        );
        a.coordinate = IosMemoryCoordinate::Absolute;
        let IosWindowBound::Transfer {
            domain, coordinate, ..
        } = &mut bounds[0]
        else {
            unreachable!()
        };
        *domain = IosMemoryDomain::NativeAllocatorAbsolute;
        *coordinate = IosMemoryCoordinate::Absolute;
        at.observations[0].domain = IosMemoryDomain::NativeAllocatorAbsolute;
        at.observations[0].coordinate = IosMemoryCoordinate::Absolute;
        let result = derive_certified(&a, &at, &req, &bounds).unwrap();
        assert_eq!(result.domain, IosMemoryDomain::NativeAllocatorAbsolute);
        assert_eq!(result.coordinate, IosMemoryCoordinate::Absolute);
        assert_eq!(result.demand, 45);
        let raw = vec![IosWindowBound::RawPhysical {
            domain: IosMemoryDomain::NativeAllocatorAbsolute,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            request_class: "r".into(),
            growth_upper: 45,
        }];
        assert_eq!(
            derive_certified(&a, &at, &req, &raw),
            Err(NotDerivable::DomainMismatch)
        );
    }

    #[test]
    fn asymmetric_baselines_are_normalized_before_signed_growth_transfer() {
        let a_query = KernelGrowth::from_queries(20, 30, 100, 95, 2).unwrap();
        let r_query = KernelGrowth::from_queries(70, 90, 115, 110, 2).unwrap();
        assert_eq!(a_query.growth_upper, 80);
        assert_eq!(r_query.growth_upper, 45);
        let delta = i128::from(r_query.exact_growth().unwrap())
            - i128::from(a_query.exact_growth().unwrap());
        assert_eq!(delta, -35); // NOT the +15 difference of absolute highwaters.
        let (a, at, req, _) = simple();
        assert_eq!(
            derive_certified(
                &a,
                &at,
                &req,
                &[transfer("r", vec![candidate("a", delta, 0)])]
            )
            .unwrap()
            .demand,
            r_query.exact_growth().unwrap()
        );
        assert!(KernelGrowth::from_queries(120, 100, 115, 120, 2).is_err());
    }

    #[test]
    fn asymmetric_signed_transfer_is_request_minus_anchor_in_both_directions() {
        let (a, at, req, bounds) = simple();
        assert_eq!(derive_certified(&a, &at, &req, &bounds).unwrap().demand, 45);
        let mut reverse_anchor = req.clone();
        observe(&mut reverse_anchor);
        let reverse = vec![transfer("a", vec![candidate("r", 35, 0)])];
        assert_eq!(
            derive_certified(&anchor(45), &reverse_anchor, &at, &reverse)
                .unwrap()
                .demand,
            80
        );
        // Neither swapping pricing inputs with the old class mapping nor negating
        // delta can accidentally give the same answer for this asymmetric pair.
        assert!(derive_certified(&a, &req, &at, &bounds).is_err());
        let wrong_sign = vec![transfer("r", vec![candidate("a", 35, 0)])];
        assert_eq!(
            derive_certified(&a, &at, &req, &wrong_sign).unwrap().demand,
            115
        );
    }

    #[test]
    fn coincident_held_body_switch_is_not_difference_of_maxima() {
        let mut at = pricing(vec![
            window("a0", Denoise, 90, 10, 0),
            window("a1", Denoise, 99, 0, 0),
        ]);
        observe(&mut at);
        let req = pricing(vec![
            window("r0", Denoise, 90, 0, 0),
            window("r1", Denoise, 99, 10, 0),
        ]);
        let bounds = vec![
            transfer("r0", vec![candidate("a0", -10, 0)]),
            transfer("r1", vec![candidate("a1", 10, 1)]),
        ];
        // MAX held is 10 in both requests, but the actual peak switches 100 -> 109.
        let result = derive_certified(&anchor(100), &at, &req, &bounds).unwrap();
        assert_eq!(result.demand, 110); // Conservative whole-run upper + pointwise delta.
        assert!(result.demand >= 109);
    }

    #[test]
    fn ambiguous_anchor_correspondences_take_max_not_cheapest() {
        let mut at = pricing(vec![
            window("a0", Decode, 0, 0, 100),
            window("a1", Decode, 0, 0, 99),
        ]);
        observe(&mut at);
        let req = pricing(vec![window("r", Decode, 0, 0, 1)]);
        let bounds = vec![transfer(
            "r",
            vec![candidate("a0", -30, 0), candidate("a1", 9, 1)],
        )];
        assert_eq!(
            derive_certified(&anchor(100), &at, &req, &bounds)
                .unwrap()
                .demand,
            109
        );
    }

    #[test]
    fn every_request_window_not_only_the_phase_winner_is_covered() {
        let (a, at, mut req, mut bounds) = simple();
        req.take_window(window("other", Denoise, 0, 0, 1));
        assert!(derive_phase_certified(&a, &at, &req, &bounds, Denoise).is_err());
        assert_eq!(
            derive_window_certified(&a, &at, &req, &bounds, "r").unwrap(),
            45
        );
        bounds.push(IosWindowBound::RawPhysical {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
            request_class: "other".into(),
            growth_upper: 120,
        });
        assert_eq!(
            derive_phase_certified(&a, &at, &req, &bounds, Denoise)
                .unwrap()
                .bytes,
            120
        );
    }

    #[test]
    fn equal_logical_labels_and_counts_are_not_transfer_authority() {
        let (a, at, _, _) = simple();
        assert!(derive_certified(&a, &at, &at, &[]).is_err());
        assert_eq!(derive(&a, &at, &at, a.area), Err(NotDerivable::Uncertified));
    }

    #[test]
    fn margin_is_applied_after_the_complete_maximum_not_stacked_reserves() {
        let a = anchor(100);
        let req = pricing(vec![
            window("first", Conditioning, 10000, 10000, 10000),
            window("second", Decode, 20000, 20000, 20000),
        ]);
        let bounds = vec![
            IosWindowBound::RawPhysical {
                domain: IosMemoryDomain::ProcessGrowth,
                coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
                request_class: "first".into(),
                growth_upper: 100,
            },
            IosWindowBound::RawPhysical {
                domain: IosMemoryDomain::ProcessGrowth,
                coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
                request_class: "second".into(),
                growth_upper: 109,
            },
        ];
        let result = derive_certified(&a, &IosPricing::default(), &req, &bounds).unwrap();
        assert_eq!(result.demand, 109);
        assert_eq!(with_margin(result.demand, 7).unwrap(), 116);
        assert_eq!(
            result
                .phases
                .iter()
                .find(|phase| phase.phase == Decode)
                .unwrap()
                .bytes,
            109
        );
    }

    #[test]
    fn duplicate_certificates_do_not_replace_a_missing_class() {
        let (a, at, mut req, mut bounds) = simple();
        req.take_window(window("missing", Denoise, 0, 0, 1));
        bounds.push(bounds[0].clone());
        assert!(derive_certified(&a, &at, &req, &bounds).is_err());
    }

    #[test]
    fn empty_or_unknown_candidates_fail_even_if_other_candidates_are_valid() {
        let (a, at, req, _) = simple();
        assert!(derive_certified(&a, &at, &req, &[transfer("r", vec![])]).is_err());
        let bounds = vec![transfer(
            "r",
            vec![candidate("a", 0, 0), candidate("unknown", -10, 0)],
        )];
        assert!(derive_certified(&a, &at, &req, &bounds).is_err());
    }

    #[test]
    fn unverified_profiles_and_missing_actual_context_fail() {
        let (a, at, req, bounds) = simple();
        for field in 0..3 {
            let mut bad = req.clone();
            match field {
                0 => bad.windows[0].class = None,
                1 => bad.windows[0].basis = None,
                _ => bad.windows[0].owner_domain.clear(),
            }
            assert!(derive_certified(&a, &at, &bad, &bounds).is_err());
        }
        let mut bad = at.clone();
        bad.observations.clear();
        assert!(derive_certified(&a, &bad, &req, &bounds).is_err());
        let mut bad = at.clone();
        bad.observations[0].checks[0].context.clear();
        assert!(derive_certified(&a, &bad, &req, &bounds).is_err());
    }

    #[test]
    fn witness_indices_label_phase_and_metadata_are_checked() {
        let (a, at, req, bounds) = simple();
        for defect in 0..6 {
            let mut bad = at.clone();
            match defect {
                0 => bad.observations[0].checks.clear(),
                1 => bad.observations[0].checks[0].context[0] = 1,
                2 => bad.observations[0].checks[0].label = "unobserved".into(),
                3 => bad.observations[0].checks[0].phase = Some(Decode),
                4 => bad.observations[0].metadata[0].provider.clear(),
                _ => bad.observations[0].checks[0].context.push(0),
            }
            assert!(
                derive_certified(&a, &bad, &req, &bounds).is_err(),
                "defect {defect}"
            );
        }
    }

    #[test]
    fn signed_addition_rejects_negative_and_both_overflow_domains() {
        let (a, at, req, _) = simple();
        for delta in [-81, i128::MIN, i128::MAX, i128::from(u64::MAX)] {
            assert_eq!(
                derive_certified(
                    &a,
                    &at,
                    &req,
                    &[transfer("r", vec![candidate("a", delta, 0)])]
                ),
                Err(NotDerivable::Arithmetic)
            );
        }
        assert_eq!(
            derive_certified(
                &a,
                &at,
                &req,
                &[transfer("r", vec![candidate("a", -80, 0)])]
            )
            .unwrap()
            .demand,
            0
        );
    }

    #[test]
    fn physical_lower_is_not_invented_or_subtracted_twice() {
        let (a, at, req, mut bounds) = simple();
        let IosWindowBound::Transfer {
            anchor_candidates, ..
        } = &mut bounds[0]
        else {
            unreachable!()
        };
        anchor_candidates[0].physical_anchor_lower_bound = Some(80);
        assert_eq!(derive_certified(&a, &at, &req, &bounds).unwrap().demand, 45);
        let IosWindowBound::Transfer {
            anchor_candidates, ..
        } = &mut bounds[0]
        else {
            unreachable!()
        };
        anchor_candidates[0].physical_anchor_lower_bound = Some(81);
        assert_eq!(
            derive_certified(&a, &at, &req, &bounds),
            Err(NotDerivable::InvalidMeasurement)
        );
    }

    #[test]
    fn identity_delta_reproduces_valid_measurement_and_margin_once() {
        let (a, at, req, _) = simple();
        let bounds = vec![transfer("r", vec![candidate("a", 0, 0)])];
        let measured = KernelGrowth::from_queries(20, 30, 100, 95, 2).unwrap();
        assert_eq!(measured.exact_growth().unwrap(), a.demand);
        let demand = derive_certified(&a, &at, &req, &bounds).unwrap().demand;
        assert_eq!(demand, 80);
        assert_eq!(with_margin(demand, 7).unwrap(), 87);
        assert_eq!(with_margin(u64::MAX, 1), Err(NotDerivable::Arithmetic));
    }

    #[test]
    fn larger_quantum_and_new_phase_need_complete_physical_certificates() {
        let (a, at, mut req, bounds) = simple();
        req.take_window(window("encode", Encode, 1, 1, 1000));
        req.absent_phases.remove(&Encode);
        assert!(derive_certified(&a, &at, &req, &bounds).is_err());
        let mut certified = bounds;
        certified.push(IosWindowBound::RawPhysical {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeRequestConstruction,
            request_class: "encode".into(),
            growth_upper: 1500,
        });
        assert_eq!(
            derive_certified(&a, &at, &req, &certified).unwrap().demand,
            1500
        );
    }

    #[test]
    fn actual_zero_denoise_is_explicit_not_missing_data() {
        let a = anchor(0);
        let mut req = IosPricing::default();
        assert!(derive_phase_certified(&a, &req, &req, &[], Denoise).is_err());
        req.phase_not_run(Denoise, "actual zero-step schedule");
        assert_eq!(
            derive_phase_certified(&a, &req, &req, &[], Denoise)
                .unwrap()
                .bytes,
            0
        );
        assert!(derive_certified(&a, &req, &req, &[]).is_err());
        req.take_window(window("contradiction", Denoise, 0, 0, 1));
        assert!(derive_phase_certified(&a, &req, &req, &[], Denoise).is_err());
    }

    #[test]
    fn masked_old_kernel_peak_is_upper_not_exact_and_gaps_use_whole_run() {
        let masked = KernelGrowth::from_queries(20, 120, 120, 90, 4).unwrap();
        assert_eq!(masked.growth_upper, 100);
        assert_eq!(masked.quality, KernelPeakQuality::MaskedUpper);
        assert!(masked.exact_growth().is_err());
        let attained = KernelGrowth::from_queries(20, 120, 120, 120, 4).unwrap();
        assert_eq!(attained.exact_growth().unwrap(), 100);
        assert!(KernelGrowth::from_queries(20, 120, 120, 90, 0).is_err());
        assert!(KernelGrowth::from_queries(20, 10, 120, 90, 4).is_err());
        let (mut a, mut at, req, _) = simple();
        a.demand = masked.growth_upper;
        a.phases.insert(Denoise, 1); // Not a binding measurement.
        at.observations[0].telemetry_complete = false;
        at.observations[0].checks[0].phase = None;
        assert_eq!(
            derive_certified(&a, &at, &req, &[transfer("r", vec![candidate("a", 0, 0)])])
                .unwrap()
                .demand,
            100
        );
        a.phases.insert(Denoise, 101);
        assert_eq!(
            derive_certified(&a, &at, &req, &[transfer("r", vec![candidate("a", 0, 0)])]),
            Err(NotDerivable::InvalidMeasurement)
        );
    }
}

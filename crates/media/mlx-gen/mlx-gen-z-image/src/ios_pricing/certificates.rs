//! Z-Image native-active-counter certificates, from the pinned CPU-stream source graph.
//!
//! The request side is a checked cumulative sum of source-enumerated allocator requests. The
//! anchor side subtracts NOTHING: measured nonnegative anchor growth is merely an additive
//! reference with real capture witnesses. These are NOT physical task-footprint certificates.
use super::{header_dtype, ios_quantum_phase, Component, Facts, Model, Route, VaeOwnerHistory};
use mlx_gen::ios_pricing::{
    IosAnchorCandidate, IosAnchorWitness, IosMemoryCoordinate, IosMemoryDomain, IosPhase,
    IosPricing, IosWindowBound,
};
use mlx_gen::sampler::flow_sampling::ResolvedFlowSampling;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;
use std::collections::{BTreeMap, BTreeSet};
#[path = "certificates/cpu.rs"]
mod cpu;
#[path = "certificates/vae_graph.rs"]
mod vae_graph;
#[path = "certificates/z_graph.rs"]
mod z_graph;
use cpu::{add, mul, Graph, Tensor};

#[derive(Debug, Default)]
pub(crate) struct WindowCertificates {
    pub bounds: Vec<IosWindowBound>,
    pub blocked: Vec<(String, String)>,
}
impl WindowCertificates {
    pub fn blocked_failures(&self, case: &str, selection: &str, tier: &str) -> Vec<String> {
        self.blocked
            .iter()
            .map(|(class, reason)| {
                format!("{case} / {selection} / {tier}: NotDerivable {class}: {reason}")
            })
            .collect()
    }
}

/// Source checkpoints, not allocation maxima borrowed from logical pricing. The same source
/// checkpoint may occur in many callbacks/outputs; retaining its greatest cumulative prefix covers
/// every occurrence without taking any eval/drop/phase-transition retirement credit.
#[derive(Clone)]
struct Program {
    graph: Graph,
    checkpoints: BTreeMap<(IosPhase, String, usize), u64>,
    points: BTreeMap<String, usize>,
    ios_quantum_scope: bool,
}
impl Program {
    fn new(page: u64, ios_quantum_scope: bool) -> Result<Self> {
        Ok(Self {
            graph: Graph::new(page)?,
            checkpoints: BTreeMap::new(),
            points: BTreeMap::new(),
            ios_quantum_scope,
        })
    }
    fn mark(&mut self, label: &str) -> Result<()> {
        let point = *self.points.entry(label.into()).or_default();
        self.mark_at(label, point)?;
        *self.points.get_mut(label).unwrap() = point
            .checked_add(1)
            .ok_or_else(|| Error::Msg("Z source point overflow".into()))?;
        Ok(())
    }
    fn mark_at(&mut self, label: &str, point: usize) -> Result<()> {
        let phase = ios_quantum_phase(label)
            .ok_or_else(|| Error::Msg(format!("unmapped Z source label {label}")))?;
        self.checkpoints
            .entry((phase, label.into(), point))
            .and_modify(|n| *n = (*n).max(self.graph.upper()))
            .or_insert(self.graph.upper());
        Ok(())
    }
    fn reset_phase_points(&mut self, phase: IosPhase) {
        self.points
            .retain(|label, _| ios_quantum_phase(label) != Some(phase));
    }
}

/// Actual caller-owned heavy-construction ordering. The original CPU fixture quantizes Control
/// before VAE load; production loads VAE first. This changes cumulative prefixes, not model roles.
#[derive(Clone, Copy, Debug)]
pub enum HeavyLoadOrder {
    QuantizeBeforeVae,
    VaeBeforeQuantize,
}

/// Caller-asserted facts of the EXECUTION being certified, never sampled from the thread that
/// happens to evaluate this pure producer. `ios_quantum_scope` selects the same graph branches as
/// `ios_quantum::enabled()` at runtime (bounded VAE attention, >256px tiled encode); the canonical
/// request classes price exactly that scoped graph, so an unscoped run is refused, not re-priced.
#[derive(Clone, Copy, Debug)]
pub struct CertifiedExecution {
    pub load_order: HeavyLoadOrder,
    pub ios_quantum_scope: bool,
    /// Every wrapper op of the certified run (incl. stream-less frontend casts) used the CPU stream:
    /// global default device/stream AND any mlx-rs task-local default override.
    pub cpu_stream: bool,
}

/// The exact refusals for execution facts this producer cannot certify. Shared by the public
/// provider bounder so it refuses BEFORE reading any snapshot header.
pub(crate) fn certified_execution_refusal(execution: CertifiedExecution) -> Result<()> {
    if !execution.cpu_stream {
        // The pinned CPU lowering (explicit-GEMM conv, composed SDPA/norm fallbacks, native
        // Quantize, default-stream wrapper casts) is the authority ONLY for a CPU-stream run.
        return Err(Error::Unsupported("Z certificate CPU lowering requires the certified execution's effective wrapper/default stream to be CPU".into()));
    }
    if !execution.ios_quantum_scope {
        return Err(Error::Unsupported("Z certificate: canonical request classes price the iOS-quantum-scope graph; an unscoped execution has different VAE graph branches".into()));
    }
    Ok(())
}

/// `sampling` MUST be the real resolved and provider-windowed plan used by execution, not an
/// allocation-class representative or a schedule recreated from a logical price/step multiplier.
/// In particular the original fixture's manual init tail differs from production strength rules.
pub(crate) fn window_certificates(
    model: &Model,
    route: &Route,
    sampling: &ResolvedFlowSampling,
    execution: CertifiedExecution,
    at_anchor: &IosPricing,
    request: &IosPricing,
) -> Result<WindowCertificates> {
    certified_execution_refusal(execution)?;
    sampling.epochs()?; // validate the exact supplied factory schedule; no invented step bound.
    let page = mlx_gen::ios_quantum::native_metal_page_bytes().ok_or_else(|| {
        Error::Unsupported(
            "Z active-counter certificate: actual linked Metal vm_page_size unavailable".into(),
        )
    })?;
    let mut prefixes = BTreeMap::<(IosPhase, String, usize), u64>::new();
    model
        .vae_sources
        .visit_constructed(model.facts.quant_bits, &mut |vae, owners| {
            let mut program = Program::new(page, execution.ios_quantum_scope)?;
            z_graph::request(
                &mut program,
                model,
                route,
                sampling,
                execution.load_order,
                vae,
                &owners,
            )?;
            for (key, value) in program.checkpoints {
                prefixes
                    .entry(key)
                    .and_modify(|n| *n = (*n).max(value))
                    .or_insert(value);
            }
            Ok(())
        })?;
    bind_prefixes(route, at_anchor, request, &prefixes)
}

fn bind_prefixes(
    route: &Route,
    anchor: &IosPricing,
    request: &IosPricing,
    prefixes: &BTreeMap<(IosPhase, String, usize), u64>,
) -> Result<WindowCertificates> {
    let mut result = WindowCertificates::default();
    for window in &request.windows {
        let class = window
            .class
            .as_deref()
            .ok_or_else(|| Error::Msg("Z certificate request has no source class".into()))?;
        let expected = format!(
            "z-image/{}/{}/{}/source-op-",
            route.mode.key(),
            window.phase.as_key(),
            window.label
        );
        let point = class
            .strip_prefix(&expected)
            .and_then(|suffix| suffix.split('/').next())
            .and_then(|n| n.parse::<usize>().ok());
        let Some(point) = point else {
            result.blocked.push((
                class.into(),
                "request class does not identify this route's source operation".into(),
            ));
            continue;
        };
        let Some(&upper) = prefixes.get(&(window.phase, window.label.clone(), point)) else {
            result.blocked.push((class.into(),format!("no evaluated source checkpoint for {} point {point} in the supplied resolved execution plan",window.label)));
            continue;
        };
        let mut candidates = Vec::new();
        let mut classes = BTreeSet::new();
        for aw in anchor.windows.iter().filter(|aw| aw.phase == window.phase) {
            let Some(anchor_class) = aw.class.as_ref() else {
                continue;
            };
            if !anchor_class.starts_with(&format!("z-image/{}/", route.mode.key()))
                || !classes.insert(anchor_class.clone())
            {
                continue;
            }
            let mut witnesses = Vec::new();
            for (capture, observation) in anchor.observations.iter().enumerate() {
                if observation.domain != IosMemoryDomain::NativeAllocatorGrowth
                    || observation.coordinate != IosMemoryCoordinate::BeforeModelLoad
                {
                    continue;
                }
                for (check, observed) in observation.checks.iter().enumerate() {
                    if observed.label != aw.label
                        || observed.phase.is_some_and(|phase| phase != aw.phase)
                        || observed.context.is_empty()
                    {
                        continue;
                    }
                    // Label matching alone is not the transfer proof. U_request - 0 is independent
                    // of the anchor's graph; these checks only establish the genuine additive
                    // anchor coordinate and actual context being referenced, with no lower credit.
                    let mut seen = BTreeSet::new();
                    if observed.context.iter().all(|&i| {
                        seen.insert(i)
                            && observation
                                .metadata
                                .get(i as usize)
                                .is_some_and(|m| !m.provider.is_empty() && !m.kind.is_empty())
                    }) {
                        witnesses.push(IosAnchorWitness {
                            capture: u32::try_from(capture)
                                .map_err(|_| Error::Msg("anchor capture index overflow".into()))?,
                            check: u32::try_from(check)
                                .map_err(|_| Error::Msg("anchor check index overflow".into()))?,
                        });
                    }
                }
            }
            if !witnesses.is_empty() {
                candidates.push(IosAnchorCandidate {
                    anchor_class: anchor_class.clone(),
                    delta_growth_upper: i128::from(upper),
                    physical_anchor_lower_bound: None,
                    witnesses,
                });
            }
        }
        if candidates.is_empty() {
            result.blocked.push((class.into(),"anchor has no actual same-phase native-growth BeforeModelLoad check with valid nonempty metadata context".into()));
        } else {
            result.bounds.push(IosWindowBound::Transfer {
                domain: IosMemoryDomain::NativeAllocatorGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                request_class: class.into(),
                anchor_candidates: candidates,
            });
        }
    }
    Ok(result)
}

//! Weights-free campaign protocol. One selected case per fresh process; no implicit fallbacks.
//!
//! The campaign measures ANCHORS only (the recorded memory-anchor rule: two measured extremes per
//! family, everything in between derived — `mlx_gen_ios_catalog::anchor`): per provider route x
//! tier x mode, the fully RESIDENT rung-0 composition and the fully OPTIMIZED deepest-rung,
//! cheapest-selection composition, both at the [`reference_edge`] with [`ANCHOR_STEPS`] denoise
//! steps. Two validation cases ride along for ONE model: the optimized anchor at full steps (does a
//! short-step capture see the full-step peak?) and at the [`VALIDATION_EDGE`] (an ADMITTED run the
//! anchor law predicts before it is measured).
use std::collections::{BTreeMap, BTreeSet};

/// The selection token of a strategy with no declared parameters (and of a rung the provider does
/// not implement: admission refuses it, typed).
pub const BASE_SELECTION: &str = "base";

/// Denoise steps of an anchor capture: the memory peak of a denoise step does not grow with the
/// step count (each step's activations are released before the next), so one step is the
/// footprint of N. [`Purpose::ValidationSteps`] checks that claim on one model.
pub const ANCHOR_STEPS: u32 = 1;

/// Repeats of each optimized anchor: their spread is the admission margin at the low end.
pub const OPTIMIZED_REPEATS: u32 = 3;

/// The second geometry of the low-end validation case (the anchors sit at [`reference_edge`]).
pub const VALIDATION_EDGE: u32 = 1024;

/// The anchors' reference geometry: the smallest square edge every iOS image provider renders
/// natively and the one most likely to complete for BOTH anchors on an 8 GB phone (every phase's
/// activations are smallest there; resident weights do not depend on it). Supra2 renders 256².
pub fn reference_edge(provider: &str) -> u32 {
    if provider.contains("supra") {
        256
    } else {
        512
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surface {
    pub provider: String,
    pub routes: Vec<String>,
    pub modes: Vec<String>,
    /// Per `(tier, rung)`, the provider's own declared selection space as tokens, QUALITY FIRST
    /// (`MemoryProviderContract::selection_space`; only rungs the contract IMPLEMENTS have one).
    /// Absent: the single [`BASE_SELECTION`].
    pub selections: BTreeMap<(String, u8), Vec<String>>,
}

impl Surface {
    /// The first (most resident) selection token of `(tier, rung)`.
    pub fn first_selection(&self, tier: &str, rung: u8) -> String {
        self.selections
            .get(&(tier.to_string(), rung))
            .and_then(|v| v.first().cloned())
            .unwrap_or_else(|| BASE_SELECTION.into())
    }

    /// The fully optimized `(rung, selection)` of a tier: the deepest rung the provider implements
    /// above rung 0, at its LAST (cheapest) selection; `None` when it implements none.
    pub fn optimized(&self, tier: &str) -> Option<(u8, String)> {
        (1..=4u8).rev().find_map(|rung| {
            self.selections
                .get(&(tier.to_string(), rung))
                .and_then(|v| v.last())
                .map(|token| (rung, token.clone()))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub provider: String,
    pub route: String,
    pub tier: String,
    pub edge: u32,
    pub mode: String,
    pub rung: u8,
    /// The cell's memory selection (tile edge, overlap, attention chunk, window size and scope) as
    /// a token of the provider's declared space; executed exactly as selected.
    pub selection: String,
    /// Denoise steps the case runs (`0`: the provider's own default — full steps).
    pub steps: u32,
}
impl Case {
    pub fn key(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}/{}/{}/{}",
            self.provider,
            self.route,
            self.tier,
            self.edge,
            self.mode,
            self.rung,
            self.selection,
            self.steps
        )
    }
    pub fn parse(value: &str) -> Result<Self, String> {
        let p: Vec<_> = value.split('/').collect();
        if p.len() != 8
            || p.iter().any(|p| {
                p.is_empty()
                    || !p
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            })
        {
            return Err(
                "case must be provider/route/{bf16,q8,q4}/edge/mode/rung/selection/steps".into(),
            );
        }
        let edge = p[3].parse::<u32>().map_err(|_| "invalid edge")?;
        let rung = p[5].parse::<u8>().map_err(|_| "invalid rung")?;
        let steps = p[7].parse::<u32>().map_err(|_| "invalid steps")?;
        if !["bf16", "q8", "q4"].contains(&p[2])
            || ![256, 512, 768, 1024, 1280].contains(&edge)
            || rung > 4
            || steps > 100
        {
            return Err("tier, geometry, rung or steps outside finite campaign domain".into());
        }
        Ok(Self {
            provider: p[0].into(),
            route: p[1].into(),
            tier: p[2].into(),
            edge,
            mode: p[4].into(),
            rung,
            selection: p[6].into(),
            steps,
        })
    }
}

/// Why a case is in the campaign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Purpose {
    /// The family's fully resident (rung 0) anchor; its pre-allocation refusal is evidence.
    Resident,
    /// The family's fully optimized anchor; must complete, captured [`OPTIMIZED_REPEATS`] times.
    Optimized,
    /// One model's optimized anchor at full steps: is the short-step peak the full-step peak?
    ValidationSteps,
    /// One model's optimized configuration at [`VALIDATION_EDGE`], run ADMITTED once the anchors are
    /// committed: the anchor law's prediction against the measurement.
    ValidationGeometry,
}

impl Purpose {
    pub fn label(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Optimized => "optimized",
            Self::ValidationSteps => "validation_steps",
            Self::ValidationGeometry => "validation_geometry",
        }
    }
}

/// One planned campaign case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    pub case: Case,
    pub purpose: Purpose,
    pub repeats: u32,
}

/// The anchor campaign, derived from the runner's own catalog (never a hand list): per provider x
/// route x tier x registered mode, the resident and the optimized anchor at the reference
/// geometry and [`ANCHOR_STEPS`]; plus the two validation cases on the FIRST family (catalog order,
/// lowest tier) that has an optimized anchor. A family whose provider implements no rung above 0
/// has only its resident anchor (and so nothing derivable). A new provider/mode is visible
/// automatically; admission, never enumeration, decides whether it can run.
pub fn enumerate(surfaces: &[Surface]) -> Vec<Planned> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut validation: Option<Case> = None;
    for s in surfaces {
        let edge = reference_edge(&s.provider);
        for route in &s.routes {
            for tier in ["q4", "q8", "bf16"] {
                for mode in &s.modes {
                    let case = |rung: u8, selection: String, steps: u32| Case {
                        provider: s.provider.clone(),
                        route: route.clone(),
                        tier: tier.into(),
                        edge,
                        mode: mode.clone(),
                        rung,
                        selection,
                        steps,
                    };
                    let mut push = |c: Case, purpose, repeats| {
                        if seen.insert(c.key()) {
                            out.push(Planned {
                                case: c,
                                purpose,
                                repeats,
                            });
                        }
                    };
                    push(
                        case(0, s.first_selection(tier, 0), ANCHOR_STEPS),
                        Purpose::Resident,
                        1,
                    );
                    if let Some((rung, selection)) = s.optimized(tier) {
                        let optimized = case(rung, selection, ANCHOR_STEPS);
                        validation.get_or_insert_with(|| optimized.clone());
                        push(optimized, Purpose::Optimized, OPTIMIZED_REPEATS);
                    }
                }
            }
        }
    }
    if let Some(anchor) = validation {
        out.push(Planned {
            case: Case {
                steps: 0,
                ..anchor.clone()
            },
            purpose: Purpose::ValidationSteps,
            repeats: 1,
        });
        if anchor.edge != VALIDATION_EDGE {
            out.push(Planned {
                case: Case {
                    edge: VALIDATION_EDGE,
                    ..anchor
                },
                purpose: Purpose::ValidationGeometry,
                repeats: 1,
            });
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Successful,
    AdmittedRefused,
    DiagnosticSuccess,
    DiagnosticFailure,
    /// A bounded DEVICE CAPTURE whose quantum refused before constructing its graph: the
    /// "next size refused before jetsam" evidence. Never a production pass.
    CaptureRefused,
    Skipped,
}
impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Self::Successful => "successful",
            Self::AdmittedRefused => "admitted_refused",
            Self::DiagnosticSuccess => "diagnostic_success",
            Self::DiagnosticFailure => "diagnostic_failure",
            Self::CaptureRefused => "capture_refused",
            Self::Skipped => "skipped",
        }
    }
    pub fn production_success(self) -> bool {
        self == Self::Successful
    }
}

/// The adapter MUST complete admission before entering `execute`. Tests use the identical
/// orchestration with a fake provider, rather than testing a second implementation of the gate.
#[derive(Debug)]
pub enum AdmissionError {
    Refused(String),
    Failure(String),
}
impl From<String> for AdmissionError {
    fn from(e: String) -> Self {
        Self::Failure(e)
    }
}
impl From<&str> for AdmissionError {
    fn from(e: &str) -> Self {
        Self::Failure(e.into())
    }
}

/// Why execution stopped. A quantum refusal is typed by the bounded-quantum mechanism itself
/// (`ios_quantum::take_refusal`), never by parsing the error message.
#[derive(Debug)]
pub enum ExecuteError {
    QuantumRefused(String),
    /// A post-admission facade/capacity refusal; the provider records its typed kind.
    Refused(String),
    Failure(String),
}
impl From<String> for ExecuteError {
    fn from(e: String) -> Self {
        Self::Failure(e)
    }
}
impl From<&str> for ExecuteError {
    fn from(e: &str) -> Self {
        Self::Failure(e.into())
    }
}

pub trait Provider {
    /// `diagnostic` selects the bounded CAPTURE path: no evidence row is required, only bounded
    /// per-quantum execution. Otherwise admission against the reviewed evidence table.
    fn admit(&mut self, case: &Case, diagnostic: bool) -> Result<(), AdmissionError>;
    fn execute(&mut self, case: &Case) -> Result<(), ExecuteError>;
}
pub fn run(provider: &mut impl Provider, case: &Case, diagnostic: bool) -> (Status, String) {
    if let Err(error) = provider.admit(case, diagnostic) {
        return match error {
            AdmissionError::Refused(reason) => (
                if diagnostic {
                    Status::DiagnosticFailure
                } else {
                    Status::AdmittedRefused
                },
                reason,
            ),
            AdmissionError::Failure(reason) => (Status::DiagnosticFailure, reason),
        };
    }
    match provider.execute(case) {
        Ok(()) => (
            if diagnostic {
                Status::DiagnosticSuccess
            } else {
                Status::Successful
            },
            "completed".into(),
        ),
        Err(ExecuteError::QuantumRefused(e)) if diagnostic => (Status::CaptureRefused, e),
        Err(ExecuteError::QuantumRefused(e)) => (
            Status::DiagnosticFailure,
            format!("admitted case hit a quantum refusal (admission evidence is stale): {e}"),
        ),
        Err(ExecuteError::Refused(e)) if !diagnostic => (Status::AdmittedRefused, e),
        Err(ExecuteError::Refused(e)) | Err(ExecuteError::Failure(e)) => {
            (Status::DiagnosticFailure, e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn surface() -> Surface {
        Surface {
            provider: "new_provider".into(),
            routes: vec!["new_provider".into()],
            modes: vec!["text".into(), "reference".into(), "mask".into()],
            selections: BTreeMap::new(),
        }
    }
    #[test]
    fn public_case_parse_roundtrip_and_finite_domain() {
        let c = Case::parse("iris_3b/iris_3b/q8/768/reference/4/t512_dit/1").unwrap();
        assert_eq!(c.edge, 768);
        assert_eq!(c.rung, 4);
        assert_eq!(c.tier, "q8");
        assert_eq!(c.selection, "t512_dit");
        assert_eq!(c.steps, 1);
        assert_eq!(Case::parse(&c.key()).unwrap(), c);
        for s in [
            "p/p/q4/0/text/1/base/1",
            "p/p/q4/999999999/text/1/base/1",
            "p/p/q4/512/text/5/base/1",
            "p/../q4/512/text/0/base/1",
            "p/p/q5/512/text/0/base/1",
            "p/p/q4/512/text/0/base",
            "p/p/q4/512/text/0/base/1/extra",
            "p/p/q4/512/text/0/t5.12/1",
            "p/p/q4/512/text/0/base/101",
            "p/p/q4/512/text/0/base/x",
        ] {
            assert!(Case::parse(s).is_err(), "{s}");
        }
    }

    /// The campaign is the ANCHOR SHAPE, derived from the catalog: per provider x route x tier x
    /// mode exactly one resident anchor (rung 0, the first selection) and, where the provider
    /// implements a rung above 0, exactly one optimized anchor (the deepest such rung, its LAST
    /// selection, repeated), both at the reference edge with short steps; plus exactly one
    /// full-steps and one second-geometry validation case on the first optimized family. Nothing
    /// else — no edge ladder, no selection ladder, no rung in between.
    #[test]
    fn the_campaign_enumerates_exactly_the_anchor_cases() {
        let mut a = surface();
        for tier in ["q4", "q8", "bf16"] {
            a.selections
                .insert((tier.into(), 2), vec!["t768".into(), "t512".into()]);
            a.selections.insert(
                (tier.into(), 4),
                vec!["t768_w2".into(), "t512_w2".into(), "s_t256_w1".into()],
            );
        }
        let mut supra = surface();
        supra.provider = "supra2_img".into();
        supra.routes = vec!["supra2_img".into()];
        let planned = enumerate(&[a.clone(), a.clone(), supra.clone()]);
        let keys: BTreeSet<_> = planned.iter().map(|p| p.case.key()).collect();
        assert_eq!(keys.len(), planned.len(), "no duplicate case");
        let anchors: Vec<_> = planned
            .iter()
            .filter(|p| matches!(p.purpose, Purpose::Resident | Purpose::Optimized))
            .collect();
        for s in [&a, &supra] {
            for route in &s.routes {
                for tier in ["q4", "q8", "bf16"] {
                    for mode in &s.modes {
                        let family: Vec<_> = anchors
                            .iter()
                            .filter(|p| {
                                p.case.provider == s.provider
                                    && &p.case.route == route
                                    && p.case.tier == tier
                                    && &p.case.mode == mode
                            })
                            .collect();
                        let resident: Vec<_> = family
                            .iter()
                            .filter(|p| p.purpose == Purpose::Resident)
                            .collect();
                        assert_eq!(resident.len(), 1);
                        assert_eq!(resident[0].case.rung, 0);
                        assert_eq!(resident[0].case.selection, s.first_selection(tier, 0));
                        assert_eq!(resident[0].repeats, 1);
                        let optimized: Vec<_> = family
                            .iter()
                            .filter(|p| p.purpose == Purpose::Optimized)
                            .collect();
                        match s.optimized(tier) {
                            Some((rung, selection)) => {
                                assert_eq!(optimized.len(), 1);
                                assert_eq!(optimized[0].case.rung, rung);
                                assert_eq!(optimized[0].case.selection, selection);
                                assert_eq!(optimized[0].repeats, OPTIMIZED_REPEATS);
                            }
                            None => assert!(optimized.is_empty()),
                        }
                        for p in &family {
                            assert_eq!(p.case.edge, reference_edge(&s.provider));
                            assert_eq!(p.case.steps, ANCHOR_STEPS);
                        }
                    }
                }
            }
        }
        // The optimized anchor is the deepest implemented rung at its cheapest selection.
        assert_eq!(a.optimized("q4"), Some((4, "s_t256_w1".into())));
        assert_eq!(supra.optimized("q4"), None);
        assert!(anchors
            .iter()
            .all(|p| p.case.provider != "supra2_img" || p.case.edge == 256));
        // Exactly one of each validation case, both on the FIRST optimized family.
        let first = anchors
            .iter()
            .find(|p| p.purpose == Purpose::Optimized)
            .unwrap();
        let of = |purpose| {
            planned
                .iter()
                .filter(|p| p.purpose == purpose)
                .collect::<Vec<_>>()
        };
        let steps = of(Purpose::ValidationSteps);
        assert_eq!(steps.len(), 1);
        assert_eq!(
            steps[0].case,
            Case {
                steps: 0,
                ..first.case.clone()
            }
        );
        let geometry = of(Purpose::ValidationGeometry);
        assert_eq!(geometry.len(), 1);
        assert_eq!(
            geometry[0].case,
            Case {
                edge: VALIDATION_EDGE,
                ..first.case.clone()
            }
        );
        assert_eq!(
            planned.len(),
            anchors.len() + 2,
            "anchors and the two validation cases only"
        );
        // A catalog with no optimized family has no validation case.
        assert!(enumerate(&[supra])
            .iter()
            .all(|p| p.purpose == Purpose::Resident));
    }
    struct Fake {
        admitted: bool,
        fail: bool,
        quantum: bool,
        refused: bool,
        loads: usize,
    }
    impl Provider for Fake {
        fn admit(&mut self, _: &Case, _: bool) -> Result<(), AdmissionError> {
            if self.admitted {
                Ok(())
            } else {
                Err(AdmissionError::Refused("no bound".into()))
            }
        }
        fn execute(&mut self, _: &Case) -> Result<(), ExecuteError> {
            self.loads += 1;
            if self.refused {
                Err(ExecuteError::Refused(
                    "live capacity recheck refused".into(),
                ))
            } else if self.quantum {
                Err(ExecuteError::QuantumRefused("SANA block refused".into()))
            } else if self.fail {
                Err("bad output".into())
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn next_up_refuses_before_weight_load_and_no_false_diagnostic_pass() {
        let c = enumerate(&[surface()]).remove(0).case;
        let mut p = Fake {
            admitted: false,
            fail: false,
            quantum: false,
            refused: false,
            loads: 0,
        };
        assert_eq!(run(&mut p, &c, false).0, Status::AdmittedRefused);
        assert_eq!(p.loads, 0);
        assert_eq!(run(&mut p, &c, true).0, Status::DiagnosticFailure);
        assert_eq!(p.loads, 0);
        p.admitted = true;
        assert_eq!(run(&mut p, &c, false).0, Status::Successful);
        assert_eq!(p.loads, 1);
        let diagnostic = run(&mut p, &c, true).0;
        assert_eq!(diagnostic, Status::DiagnosticSuccess);
        assert!(!diagnostic.production_success());
        p.fail = true;
        assert_eq!(run(&mut p, &c, false).0, Status::DiagnosticFailure);
        assert!(!Status::Skipped.production_success());
    }
    #[test]
    fn a_capture_quantum_refusal_is_typed_capture_evidence_never_a_pass() {
        let c = enumerate(&[surface()]).remove(0).case;
        let mut p = Fake {
            admitted: true,
            fail: false,
            quantum: true,
            refused: false,
            loads: 0,
        };
        let (status, reason) = run(&mut p, &c, true);
        assert_eq!(status, Status::CaptureRefused);
        assert_eq!(status.label(), "capture_refused");
        assert!(!status.production_success());
        assert!(reason.contains("SANA block refused"));
        // In admitted (product) mode the same refusal means the evidence was stale: a failure.
        assert_eq!(run(&mut p, &c, false).0, Status::DiagnosticFailure);
        // A post-admission facade/capacity refusal stays a typed admitted refusal.
        p.refused = true;
        assert_eq!(run(&mut p, &c, false).0, Status::AdmittedRefused);
        assert_eq!(run(&mut p, &c, true).0, Status::DiagnosticFailure);
    }
}

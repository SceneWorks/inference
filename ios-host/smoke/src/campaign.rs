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
/// step count (each step's activations are released before the next), so a short run is the
/// footprint of a long one. Two steps rather than one so the capture also crosses a step-to-step
/// release (and no sampler sees a degenerate one-sigma schedule). [`Purpose::ValidationSteps`]
/// checks the claim on one model against the provider's full default schedule.
pub const ANCHOR_STEPS: u32 = 2;

/// Repeats of each optimized anchor: their spread is the admission margin at the low end. Two
/// suffice: the margin is `max(spread, 32 MiB floor)`, so a third repeat could only widen a spread
/// the floor already backstops, at a third of the campaign's cost.
pub const OPTIMIZED_REPEATS: u32 = 2;

/// The geometry of the additive-MODE validation case (a ControlNet request, run admitted).
pub const VALIDATION_MODE_EDGE: u32 = 1024;

/// The second geometry of the low-end validation case (the anchors sit at [`reference_edge`]): the
/// farthest ladder size from the anchor, derived downward.
pub const VALIDATION_EDGE: u32 = 512;

/// The anchors' reference geometry: the LARGEST ladder edge (1024²; Supra2 renders 256² only).
///
/// * The fully optimized configuration must fit at every ladder size, so it completes here by
///   requirement — and this is the tightest cell, where an accurate low end matters most: it is
///   measured, not derived.
/// * Derivation from it is safe and tight. Measured on the desktop MLX lane (SceneWorks
///   `docs/sc-16195`, 7 tiers x 5 geometries), the activation transient is linear in area above
///   1024² (≤ 2.0% error from a 1024² anchor) and floored below it (a smaller request is never
///   credited, so it errs large). From a 512² anchor the same data shows a 1024²-knee law would
///   UNDER-predict 1024² by 49–70%, and the safe linear law over-predict it by 19–105%.
/// * The resident anchor may be refused before allocation here; that typed refusal is its evidence.
pub fn reference_edge(provider: &str) -> u32 {
    if provider.contains("supra") {
        256
    } else {
        1024
    }
}

/// One provider-resolved allocation-graph representative, not a sampler/scheduler/float grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureRecipe {
    pub token: Option<String>,
    pub memory_class: Option<String>,
    /// Actual normalized configured graph; separate from base architecture and sampling class.
    pub configured_variant: Option<String>,
    pub steps: u32,
    /// An existing unsupported combination, visible in listing but not a completed-anchor target.
    pub refusal: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surface {
    pub provider: String,
    pub routes: Vec<String>,
    pub modes: Vec<String>,
    /// Canonical recipe representatives by (route, tier, mode), resolved on actual configuration
    /// when required. Unresolved templates remain explicit refusals, never fabricated classes.
    pub recipes: BTreeMap<(String, String, String), Vec<CaptureRecipe>>,
    /// Per `(tier, rung)`, the provider's own declared selection space as tokens, QUALITY FIRST
    /// (`MemoryProviderContract::selection_space`; only rungs the contract IMPLEMENTS have one).
    /// Absent: the single [`BASE_SELECTION`].
    pub selections: BTreeMap<(String, String, u8), Vec<String>>,
}

impl Surface {
    fn recipes_for(&self, route: &str, tier: &str, mode: &str) -> Vec<CaptureRecipe> {
        self.recipes
            .get(&(route.into(), tier.into(), mode.into()))
            .cloned()
            .unwrap_or_else(|| {
                vec![CaptureRecipe {
                    token: None,
                    memory_class: None,
                    configured_variant: None,
                    steps: ANCHOR_STEPS,
                    refusal: None,
                }]
            })
            .into_iter()
            .filter(|recipe| recipe.refusal.is_none())
            .collect()
    }

    /// The first (most resident) selection token of `(tier, rung)`.
    pub fn first_selection(&self, route: &str, tier: &str, rung: u8) -> String {
        self.selections
            .get(&(route.to_string(), tier.to_string(), rung))
            .and_then(|v| v.first().cloned())
            .unwrap_or_else(|| BASE_SELECTION.into())
    }

    /// The fully optimized `(rung, selection)` of a tier: the deepest rung the provider implements
    /// above rung 0, at its LAST (cheapest) selection; `None` when it implements none.
    /// Whether the provider serves `tier` at all: its contract declares a selection space for some
    /// rung at that tier (a route with no declared rung at a tier — e.g. a dense-only task asked for
    /// q4 — is not an anchor family; the listing reports it as unserved).
    pub fn serves(&self, tier: &str) -> bool {
        self.selections.keys().any(|(_, t, _)| t == tier)
    }

    pub fn serves_route(&self, route: &str, tier: &str) -> bool {
        self.selections
            .keys()
            .any(|(r, t, _)| r == route && t == tier)
    }

    pub fn optimized(&self, route: &str, tier: &str) -> Option<(u8, String)> {
        (1..=4u8).rev().find_map(|rung| {
            self.selections
                .get(&(route.to_string(), tier.to_string(), rung))
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
    /// Optional canonical sampling recipe token. Syntax is checked here; its semantics must be
    /// validated by the provider's real resolver before the case can execute.
    pub recipe: Option<String>,
}
impl Case {
    pub fn key(&self) -> String {
        let base = format!(
            "{}/{}/{}/{}/{}/{}/{}/{}",
            self.provider,
            self.route,
            self.tier,
            self.edge,
            self.mode,
            self.rung,
            self.selection,
            self.steps
        );
        self.recipe
            .as_ref()
            .map_or_else(|| base.clone(), |recipe| format!("{base}/{recipe}"))
    }
    pub fn parse(value: &str) -> Result<Self, String> {
        let p: Vec<_> = value.split('/').collect();
        if !matches!(p.len(), 8 | 9)
            || p.iter().any(|p| {
                p.is_empty()
                    || !p
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            })
        {
            return Err(
                "case must be provider/route/{bf16,q8,q4}/edge/mode/rung/selection/steps[/recipe]"
                    .into(),
            );
        }
        let edge = p[3].parse::<u32>().map_err(|_| "invalid edge")?;
        let rung = p[5].parse::<u8>().map_err(|_| "invalid rung")?;
        let steps = p[7].parse::<u32>().map_err(|_| "invalid steps")?;
        if !["bf16", "q8", "q4"].contains(&p[2])
            || ![256, 512, 768, 1024, 1280].contains(&edge)
            || rung > 4
        {
            return Err("tier, geometry or rung outside campaign domain".into());
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
            recipe: p.get(8).map(|s| (*s).into()),
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
    /// A component-additive MODE (a ControlNet request at [`VALIDATION_MODE_EDGE`]) derived from its
    /// family's text anchor, run ADMITTED: the additive-mode derivation checked on the device.
    ValidationMode,
}

impl Purpose {
    pub fn label(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Optimized => "optimized",
            Self::ValidationSteps => "validation_steps",
            Self::ValidationGeometry => "validation_geometry",
            Self::ValidationMode => "validation_mode",
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

/// How the runner groups the catalog into anchor FAMILIES: a `(provider, route, tier)` into an
/// architecture group (on the host, the base-component architecture digest of the provisioned
/// slot — checkpoints of one architecture share a group; weights-free, every route is its own), and
/// a `(provider, campaign mode)` into its anchor variant (`base` for a mode the provider proved
/// component-additive, else the mode itself).
pub struct Grouping<'a> {
    pub architecture: &'a dyn Fn(&str, &str, &str) -> String,
    /// `(provider, route, campaign mode, tier, actual existing recipe)` → variant.
    /// Route/token context lets configured role selection reuse the real provider resolver.
    pub variant: &'a dyn Fn(&str, &str, &str, &str, &CaptureRecipe) -> String,
}

/// One anchor family: its group key, the member `provider/route/mode` cells it prices (all derived
/// except the anchor's own), and the cell the anchors are captured on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Family {
    pub architecture: String,
    pub tier: String,
    pub variant: String,
    pub recipe: Option<String>,
    pub anchor_steps: u32,
    /// `(provider, route, mode)` the anchors run on: the first text-mode member, else the first.
    pub anchor: (String, String, String),
    pub members: Vec<String>,
}

/// The weights-free grouping: no checkpoint collapse (each `(provider, route)` is its own
/// architecture group), variants from `variant`.
pub fn per_route<'a>(
    variant: &'a dyn Fn(&str, &str, &str, &str, &CaptureRecipe) -> String,
) -> Grouping<'a> {
    Grouping {
        architecture: &|provider: &str, route: &str, _tier: &str| format!("{provider}/{route}"),
        variant,
    }
}

/// The anchor families of the catalog under `grouping`, in catalog order.
pub fn families(surfaces: &[Surface], grouping: &Grouping<'_>) -> Vec<Family> {
    let mut out: Vec<Family> = Vec::new();
    for s in surfaces {
        for route in &s.routes {
            for tier in ["q4", "q8", "bf16"]
                .into_iter()
                .filter(|t| s.serves_route(route, t))
            {
                let architecture = (grouping.architecture)(&s.provider, route, tier);
                for mode in &s.modes {
                    for recipe in s.recipes_for(route, tier, mode) {
                        let base_variant =
                            (grouping.variant)(&s.provider, route, mode, tier, &recipe);
                        let mut variant = recipe
                            .memory_class
                            .as_ref()
                            .map_or(base_variant.clone(), |class| {
                                format!("{base_variant}/{class}")
                            });
                        if let Some(configured) = &recipe.configured_variant {
                            variant.push_str("/configured=");
                            variant.push_str(configured);
                        }
                        let member = format!("{}/{route}/{mode}", s.provider);
                        match out.iter_mut().find(|f| {
                            f.architecture == architecture && f.tier == tier && f.variant == variant
                        }) {
                            Some(f) => {
                                if mode == "text" && f.anchor.2 != "text" {
                                    f.anchor = (s.provider.clone(), route.clone(), mode.clone());
                                    f.recipe = recipe.token.clone();
                                    f.anchor_steps = recipe.steps;
                                }
                                if !f.members.contains(&member) {
                                    f.members.push(member);
                                }
                            }
                            None => out.push(Family {
                                architecture: architecture.clone(),
                                tier: tier.into(),
                                variant,
                                recipe: recipe.token.clone(),
                                anchor_steps: recipe.steps,
                                anchor: (s.provider.clone(), route.clone(), mode.clone()),
                                members: vec![member],
                            }),
                        }
                    }
                }
            }
        }
    }
    out
}

/// The anchor campaign, derived from the runner's own catalog (never a hand list): per anchor
/// FAMILY ([`families`]) the resident and the optimized anchor at the reference geometry and
/// [`ANCHOR_STEPS`], on the family's anchor cell; plus the validation cases — the full-steps and
/// second-geometry cases on the FIRST optimized family, and the additive-MODE case (a ControlNet
/// request at [`VALIDATION_MODE_EDGE`], run admitted) on the first q4 family that derives a
/// `control` member. A family whose anchor provider implements no rung above 0 has only its
/// resident anchor. A new provider/mode is visible automatically.
pub fn plan(surfaces: &[Surface], grouping: &Grouping<'_>) -> (Vec<Family>, Vec<Planned>) {
    let families = families(surfaces, grouping);
    let surface = |provider: &str| surfaces.iter().find(|s| s.provider == provider);
    let mut out = Vec::new();
    let mut validation: Option<Case> = None;
    let mut mode_validation: Option<Case> = None;
    for f in &families {
        let (provider, route, mode) = &f.anchor;
        let Some(s) = surface(provider) else { continue };
        let case = |rung: u8, selection: String| Case {
            provider: provider.clone(),
            route: route.clone(),
            tier: f.tier.clone(),
            edge: reference_edge(provider),
            mode: mode.clone(),
            rung,
            selection,
            steps: f.anchor_steps,
            recipe: f.recipe.clone(),
        };
        out.push(Planned {
            case: case(0, s.first_selection(&f.anchor.1, &f.tier, 0)),
            purpose: Purpose::Resident,
            repeats: 1,
        });
        if let Some((rung, selection)) = s.optimized(&f.anchor.1, &f.tier) {
            let optimized = case(rung, selection);
            validation.get_or_insert_with(|| optimized.clone());
            if mode_validation.is_none() && f.tier == "q4" && mode == "text" {
                if let Some(member) = f.members.iter().find(|m| m.ends_with("/control")) {
                    let mut parts = member.splitn(3, '/');
                    let (p, r) = (
                        parts.next().unwrap_or_default(),
                        parts.next().unwrap_or_default(),
                    );
                    mode_validation = Some(Case {
                        provider: p.into(),
                        route: r.into(),
                        edge: VALIDATION_MODE_EDGE,
                        mode: "control".into(),
                        ..optimized.clone()
                    });
                }
            }
            out.push(Planned {
                case: optimized,
                purpose: Purpose::Optimized,
                repeats: OPTIMIZED_REPEATS,
            });
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
    if let Some(case) = mode_validation {
        out.push(Planned {
            case,
            purpose: Purpose::ValidationMode,
            repeats: 1,
        });
    }
    (families, out)
}

/// [`plan`] under the weights-free [`per_route`] grouping with every mode additive (the campaign
/// shape the unit tests hold).
pub fn enumerate(surfaces: &[Surface]) -> Vec<Planned> {
    let variant = |_: &str, _: &str, _: &str, _: &str, _: &CaptureRecipe| "base".to_string();
    plan(surfaces, &per_route(&variant)).1
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
            recipes: BTreeMap::new(),
            // Every tier served at rung 0 (its parameterless selection).
            selections: ["q4", "q8", "bf16"]
                .into_iter()
                .map(|t| {
                    (
                        ("new_provider".into(), t.to_string(), 0),
                        vec![BASE_SELECTION.to_string()],
                    )
                })
                .collect(),
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
        assert!(c.recipe.is_none());
        assert_eq!(
            Case::parse("p/p/q4/512/text/0/base/101").unwrap().steps,
            101
        );
        assert_eq!(Case::parse("p/p/q4/512/text/0/base/0").unwrap().steps, 0);
        let maximum_steps = Case::parse("p/p/q4/512/text/0/base/4294967295").unwrap();
        assert_eq!(maximum_steps.steps, u32::MAX);
        assert_eq!(Case::parse(&maximum_steps.key()).unwrap(), maximum_steps);
        let long = format!("sdxl/sdxl/q4/1024/text/4/base/3/sdxl1-{}", "ab".repeat(512));
        let recipe_case = Case::parse(&long).unwrap();
        assert_eq!(recipe_case.key(), long);
        assert_eq!(recipe_case.recipe.as_deref(), long.rsplit('/').next());
        // Syntax alone is not semantic approval: the capture adapter validates the real recipe.
        for s in [
            "p/p/q4/0/text/1/base/1",
            "p/p/q4/999999999/text/1/base/1",
            "p/p/q4/512/text/5/base/1",
            "p/../q4/512/text/0/base/1",
            "p/p/q5/512/text/0/base/1",
            "p/p/q4/512/text/0/base",
            "p/p/q4/512/text/0/base/1/extra/extra",
            "p/p/q4/512/text/0/t5.12/1",
            "p/p/q4/512/text/0/base/4294967296",
            "p/p/q4/512/text/0/base/x",
        ] {
            assert!(Case::parse(s).is_err(), "{s}");
        }
    }

    /// The campaign is the ANCHOR SHAPE, derived from the catalog: per FAMILY (architecture group
    /// x tier x variant) exactly one resident anchor (rung 0, first selection, once) and, where the
    /// anchor provider implements a rung above 0, exactly one optimized anchor (the deepest rung,
    /// its LAST selection, repeated), both on the family's text member at the reference edge with
    /// short steps; checkpoints sharing an architecture collapse into one family, additive modes
    /// join their text family, a non-additive mode is its own; plus exactly one full-steps, one
    /// second-geometry and one additive-mode validation case. Nothing else.
    #[test]
    fn the_campaign_enumerates_exactly_the_anchor_families() {
        let mut a = surface();
        a.routes = vec!["base".into(), "finetune".into(), "distilled".into()];
        a.modes = vec![
            "text".into(),
            "reference".into(),
            "control".into(),
            "batch".into(),
        ];
        a.selections.clear();
        for route in &a.routes {
            for tier in ["q4", "q8", "bf16"] {
                a.selections
                    .insert((route.clone(), tier.into(), 0), vec![BASE_SELECTION.into()]);
                a.selections.insert(
                    (route.clone(), tier.into(), 2),
                    vec!["t768".into(), "t512".into()],
                );
                a.selections.insert(
                    (route.clone(), tier.into(), 4),
                    vec!["t768_w2".into(), "t512_w2".into(), "s_t256_w1".into()],
                );
            }
        }
        let mut supra = surface();
        supra.provider = "supra2_img".into();
        supra.routes = vec!["supra2_img".into()];
        supra.selections = BTreeMap::from([(
            ("supra2_img".into(), "q4".into(), 0),
            vec![BASE_SELECTION.into()],
        )]);
        // `base` and `finetune` share every header; `distilled` does not. `batch` is non-additive.
        let architecture = |provider: &str, route: &str, _tier: &str| {
            if route == "distilled" {
                format!("{provider}/distilled")
            } else {
                provider.to_string()
            }
        };
        let variant = |_: &str, _: &str, mode: &str, _: &str, _: &CaptureRecipe| {
            if mode == "batch" {
                mode.to_string()
            } else {
                "base".to_string()
            }
        };
        let grouping = Grouping {
            architecture: &architecture,
            variant: &variant,
        };
        let (families, planned) = plan(&[a.clone(), supra.clone()], &grouping);
        // new_provider: {shared, distilled} x 3 tiers x {base, batch}; supra2: 1 tier x 1 variant
        // (its three additive modes in one family).
        let keys: BTreeSet<_> = planned.iter().map(|p| p.case.key()).collect();
        assert_eq!(keys.len(), planned.len(), "no duplicate case");
        for f in &families {
            let anchors: Vec<_> = planned
                .iter()
                .filter(|p| {
                    matches!(p.purpose, Purpose::Resident | Purpose::Optimized)
                        && (
                            p.case.provider.clone(),
                            p.case.route.clone(),
                            p.case.mode.clone(),
                        ) == f.anchor
                        && p.case.tier == f.tier
                })
                .collect();
            let s = if f.anchor.0 == "supra2_img" {
                &supra
            } else {
                &a
            };
            let resident: Vec<_> = anchors
                .iter()
                .filter(|p| p.purpose == Purpose::Resident)
                .collect();
            assert_eq!(resident.len(), 1, "{f:?}");
            assert_eq!(resident[0].case.rung, 0);
            assert_eq!(resident[0].repeats, 1);
            let optimized: Vec<_> = anchors
                .iter()
                .filter(|p| p.purpose == Purpose::Optimized)
                .collect();
            match s.optimized(&f.anchor.1, &f.tier) {
                Some((rung, selection)) => {
                    assert_eq!(optimized.len(), 1);
                    assert_eq!(
                        (optimized[0].case.rung, &optimized[0].case.selection),
                        (rung, &selection)
                    );
                    assert_eq!(optimized[0].repeats, OPTIMIZED_REPEATS);
                }
                None => assert!(optimized.is_empty()),
            }
            for p in &anchors {
                assert_eq!(p.case.edge, reference_edge(&p.case.provider));
                assert_eq!(p.case.steps, ANCHOR_STEPS);
            }
            // The anchor runs on a text member when the family has one.
            if f.members.iter().any(|m| m.ends_with("/text")) {
                assert_eq!(f.anchor.2, "text");
            }
        }
        let of = |arch: &str, tier: &str, variant: &str| {
            families
                .iter()
                .find(|f| f.architecture == arch && f.tier == tier && f.variant == variant)
                .unwrap()
        };
        let shared = of("new_provider", "q4", "base");
        assert_eq!(
            shared.members,
            [
                "new_provider/base/text",
                "new_provider/base/reference",
                "new_provider/base/control",
                "new_provider/finetune/text",
                "new_provider/finetune/reference",
                "new_provider/finetune/control"
            ],
            "the fine-tune collapses into the base architecture; additive modes join the text family"
        );
        assert_eq!(of("new_provider/distilled", "q4", "base").members.len(), 3);
        assert_eq!(of("new_provider", "q4", "batch").anchor.2, "batch");
        assert_eq!(families.len(), 2 * 3 * 2 + 1);
        assert!(planned
            .iter()
            .filter(|p| p.case.provider == "supra2_img")
            .all(|p| p.case.edge == 256 && p.case.tier == "q4"));
        // Exactly one of each validation case: steps and geometry on the first optimized family,
        // the additive-mode case on the first q4 family deriving a `control` member.
        let first = planned
            .iter()
            .find(|p| p.purpose == Purpose::Optimized)
            .unwrap();
        let only = |purpose| {
            let v: Vec<_> = planned.iter().filter(|p| p.purpose == purpose).collect();
            assert_eq!(v.len(), 1, "{purpose:?}");
            v[0].case.clone()
        };
        assert_eq!(
            only(Purpose::ValidationSteps),
            Case {
                steps: 0,
                ..first.case.clone()
            }
        );
        assert_eq!(
            only(Purpose::ValidationGeometry),
            Case {
                edge: VALIDATION_EDGE,
                ..first.case.clone()
            }
        );
        assert_eq!(
            only(Purpose::ValidationMode),
            Case {
                edge: VALIDATION_MODE_EDGE,
                mode: "control".into(),
                ..first.case.clone()
            }
        );
        let anchors = planned
            .iter()
            .filter(|p| matches!(p.purpose, Purpose::Resident | Purpose::Optimized))
            .count();
        assert_eq!(
            planned.len(),
            anchors + 3,
            "anchors and the three validation cases only"
        );
        // The weights-free grouping never collapses routes.
        let variant = |_: &str, _: &str, _: &str, _: &str, _: &CaptureRecipe| "base".to_string();
        assert_eq!(families_count(&[a], &per_route(&variant)), 3 * 3);
        assert!(enumerate(&[supra])
            .iter()
            .all(|p| p.purpose == Purpose::Resident));
    }

    fn families_count(surfaces: &[Surface], grouping: &Grouping<'_>) -> usize {
        families(surfaces, grouping).len()
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

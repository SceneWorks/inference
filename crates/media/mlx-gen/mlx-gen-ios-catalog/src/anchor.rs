//! The iOS anchor law (epic 16774): a measured device anchor plus analytic derivation is the whole
//! admission basis, exactly as the recorded rule states for the desktop lanes (SceneWorks
//! `docs/memory-strategy-contract.md`: "the anchor + analytic derivation is the complete,
//! self-sufficient admission basis"; "every non-anchor rung and geometry is now derived
//! analytically from the anchor … rather than measured"; `crates/sceneworks-core/src/
//! memory_anchor.rs`, the image derivation law). This is an in-repo port of the same LAW and the
//! same failure mode, not of that code: the architecture facts here are the providers' own counted
//! iOS pricing (`mlx_gen::ios_pricing`), i.e. the very formulas the per-quantum guards check.
//!
//! Per phase `p` of the request:
//!
//! ```text
//!   residue_p  = M_p − C_p(anchor regime)                 measured growth minus resident components
//!   s_p        = min(residue_p, S_p(anchor))              the part the counted formula accounts for
//!   overhead_p = residue_p − s_p                          the part no formula prices (driver, allocator)
//!   D_p        = C_p(request regime)                      the request's own resident components
//!              + overhead_p × λ(area)                     unpriced overhead, area-scaled (below)
//!              + S_p(request) × κ_p                       the request's counted activation, calibrated
//!   κ_p        = s_p / S_p(anchor)  if the request's binding quantum IS the anchor's, else 1
//!   demand     = max_p D_p                                phases run in sequence
//! ```
//!
//! * `M_p` is the anchor's measured kernel-footprint growth in phase `p` (an unattributed phase is
//!   priced at the whole-run growth, an upper bound). `C_p` / `S_p` are the counted resident bytes
//!   and largest counted quantum of the phase under each regime and geometry.
//! * `κ_p ≤ 1` is the measured calibration of a counted formula: where the formula over-prices
//!   what the device measured, the request on THE SAME quantum is priced at the measured ratio, so
//!   admission near the anchor is the measurement, not the formula's conservative scratch. A
//!   different binding quantum gets its formula unscaled (erring large).
//! * `λ(area) = max(A_request, 1024²) / max(A_anchor, 1024²)`: the unpriced overhead is linear in
//!   area above 1024² and FLOORED below (measured on the desktop MLX lane: the activation transient
//!   is proportional to megapixels above 1024², max 1.97% deviation, with a floor below — sc-16195).
//! * DOMAIN: a phase whose measured growth is below the components its regime holds resident has
//!   left the law's domain (the counters did not see the weights) and is NOT derivable — never
//!   clamped to zero. A request phase the anchor never ran is not derivable either.
//!
//! At the anchor's own configuration the law returns the anchor's measurement exactly
//! (`C + (r − s) + s = M`). With both anchors measured, the optimized anchor predicts the resident
//! one: that is the law's free self-check ([`consistency`]).

use crate::evidence::AnchorRow;
use mlx_gen::ios_pricing::{IosPhase, IosPricing};
use std::collections::BTreeMap;

/// The area knee of the unpriced-overhead scaler: linear above, floor below.
pub const AREA_KNEE: u64 = 1024 * 1024;

/// Tolerance of the two-anchor self-check: the measured same-cell RECAPTURE spread of the desktop
/// MLX active peak (SceneWorks `MLX_ACTIVE_RECAPTURE_SPREAD`, 0.1260…), i.e. how far two captures of
/// one cell are measured to disagree. A law prediction off by more than the measurement's own spread
/// is the law failing, not noise.
pub const CONSISTENCY_TOLERANCE: f64 = 0.126;

/// A completed anchor's measurement, in growth over its capture's starting footprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredAnchor {
    /// Whole-run growth (the largest across repeats).
    pub demand: u64,
    /// Per-phase growth (the largest across repeats); a missing phase is priced at `demand`.
    pub phases: BTreeMap<IosPhase, u64>,
    /// Output pixels of the anchor's reference geometry (width × height × batch).
    pub area: u64,
}

impl MeasuredAnchor {
    /// From a completed anchor row; `None` for a refused one.
    pub fn from_row(row: &AnchorRow) -> Option<Self> {
        Some(Self {
            demand: row.demand_bytes()?,
            phases: row.phase_demands(),
            area: area(row.key.width, row.key.height, row.key.batch),
        })
    }

    fn growth(&self, phase: IosPhase) -> u64 {
        self.phases.get(&phase).copied().unwrap_or(self.demand)
    }
}

/// Output pixels of a geometry.
pub fn area(width: u32, height: u32, batch: u32) -> u64 {
    u64::from(width) * u64::from(height) * u64::from(batch.max(1))
}

/// Why a request is not derivable from an anchor (admission refuses it `no_evidence`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotDerivable {
    /// The request runs a phase the anchor's configuration does not.
    PhaseNotAnchored(IosPhase),
    /// The anchor measured less growth in `phase` than the components its regime holds resident.
    OutsideDomain {
        phase: IosPhase,
        measured: u64,
        resident: u64,
    },
}

impl std::fmt::Display for NotDerivable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PhaseNotAnchored(p) => write!(f, "the anchor never ran phase {}", p.as_key()),
            Self::OutsideDomain {
                phase,
                measured,
                resident,
            } => write!(
                f,
                "phase {} measured {measured} B below its {resident} B resident components \
                 (outside the anchor law's domain)",
                phase.as_key()
            ),
        }
    }
}

/// One derived phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedPhase {
    pub phase: IosPhase,
    pub bytes: u64,
    /// Whether the activation term was calibrated by the anchor (same binding quantum).
    pub calibrated: bool,
}

/// A derived demand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Derivation {
    pub demand: u64,
    pub phases: Vec<DerivedPhase>,
}

fn mul_div_ceil(value: u64, num: u64, den: u64) -> u64 {
    if den == 0 {
        return value;
    }
    let product = u128::from(value) * u128::from(num);
    u64::try_from(product.div_ceil(u128::from(den))).unwrap_or(u64::MAX)
}

/// The unpriced-overhead area scaler `λ` applied to `bytes`.
pub fn scale_overhead(bytes: u64, request_area: u64, anchor_area: u64) -> u64 {
    mul_div_ceil(
        bytes,
        request_area.max(AREA_KNEE),
        anchor_area.max(AREA_KNEE),
    )
}

/// Derive `request`'s demand from `anchor`, measured in the configuration `at_anchor` prices.
pub fn derive(
    anchor: &MeasuredAnchor,
    at_anchor: &IosPricing,
    request: &IosPricing,
    request_area: u64,
) -> Result<Derivation, NotDerivable> {
    let mut phases = Vec::with_capacity(request.phases.len());
    for wanted in &request.phases {
        let measured_at = at_anchor
            .phase(wanted.phase)
            .ok_or(NotDerivable::PhaseNotAnchored(wanted.phase))?;
        let measured = anchor.growth(wanted.phase);
        let residue = measured.checked_sub(measured_at.resident_bytes).ok_or(
            NotDerivable::OutsideDomain {
                phase: wanted.phase,
                measured,
                resident: measured_at.resident_bytes,
            },
        )?;
        let priced = residue.min(measured_at.scratch_bytes);
        let overhead = residue - priced;
        let calibrated = wanted.binding == measured_at.binding && measured_at.scratch_bytes > 0;
        let activation = if calibrated {
            mul_div_ceil(wanted.scratch_bytes, priced, measured_at.scratch_bytes)
        } else {
            wanted.scratch_bytes
        };
        let bytes = wanted
            .resident_bytes
            .saturating_add(scale_overhead(overhead, request_area, anchor.area))
            .saturating_add(activation);
        phases.push(DerivedPhase {
            phase: wanted.phase,
            bytes,
            calibrated,
        });
    }
    Ok(Derivation {
        demand: phases.iter().map(|p| p.bytes).max().unwrap_or(0),
        phases,
    })
}

/// The two-anchor self-check: the optimized anchor's derivation of the resident anchor's own
/// configuration against what the resident anchor measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Consistency {
    pub predicted: u64,
    pub measured: u64,
    /// `(predicted − measured) / measured`.
    pub relative_error: f64,
}

impl Consistency {
    /// Beyond the tolerance in either direction: reported with the anchors.
    pub fn flagged(&self) -> bool {
        self.relative_error.abs() > CONSISTENCY_TOLERANCE
    }

    /// The law UNDER-predicts a measured anchor beyond the tolerance: derivations of this family are
    /// unsafe and fail closed. (Over-prediction is flagged but every derivation then errs large.)
    pub fn fails_closed(&self) -> bool {
        self.relative_error < -CONSISTENCY_TOLERANCE
    }
}

/// [`Consistency`] of a family whose two anchors both completed.
pub fn consistency(
    optimized: &MeasuredAnchor,
    at_optimized: &IosPricing,
    resident: &MeasuredAnchor,
    at_resident: &IosPricing,
) -> Result<Consistency, NotDerivable> {
    let predicted = derive(optimized, at_optimized, at_resident, resident.area)?.demand;
    let measured = resident.demand.max(1);
    Ok(Consistency {
        predicted,
        measured,
        relative_error: (predicted as f64 - measured as f64) / measured as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::ios_pricing::IosPhase::*;

    const MIB: u64 = 1 << 20;

    fn pricing(phases: &[(IosPhase, u64, u64, &str)]) -> IosPricing {
        let mut p = IosPricing::default();
        for &(phase, resident, scratch, label) in phases {
            p.take(phase, resident * MIB, scratch * MIB, label);
        }
        p
    }

    fn anchor(demand: u64, phases: &[(IosPhase, u64)], edge: u32) -> MeasuredAnchor {
        MeasuredAnchor {
            demand: demand * MIB,
            phases: phases.iter().map(|&(p, b)| (p, b * MIB)).collect(),
            area: area(edge, edge, 1),
        }
    }

    /// A staged, tiled anchor at 512²: text 1200 resident + 60 scratch, denoise 1900 + 240, decode
    /// 600 + 700. Measured growth: text 1300, denoise 2250 (110 MiB unpriced overhead over the
    /// formula), decode 1000 (the decode formula over-prices: 400 measured of 700 priced).
    fn optimized_512() -> (MeasuredAnchor, IosPricing) {
        (
            anchor(
                2250,
                &[(Conditioning, 1300), (Denoise, 2250), (Decode, 1000)],
                512,
            ),
            pricing(&[
                (Conditioning, 1200, 60, "te layer"),
                (Denoise, 1900, 240, "dit block"),
                (Decode, 600, 700, "vae tile"),
            ]),
        )
    }

    #[test]
    fn the_derivation_reproduces_the_anchor_exactly_at_its_own_key() {
        let (a, p) = optimized_512();
        let d = derive(&a, &p, &p, a.area).unwrap();
        assert_eq!(d.demand, a.demand);
        for phase in &d.phases {
            assert_eq!(phase.bytes, a.phases[&phase.phase], "{:?}", phase.phase);
            assert!(phase.calibrated);
        }
        // An unattributed phase is priced at the whole-run growth and still reproduces it.
        let blind = MeasuredAnchor {
            phases: BTreeMap::new(),
            ..a.clone()
        };
        assert_eq!(derive(&blind, &p, &p, a.area).unwrap().demand, a.demand);
    }

    #[test]
    fn the_derivation_is_monotone_in_geometry_and_in_rung_savings() {
        let (a, p) = optimized_512();
        // Geometry: the counted activations grow with size (same binding quanta).
        let at = |edge: u64| {
            let s = edge * edge / (512 * 512);
            pricing(&[
                (Conditioning, 1200, 60, "te layer"),
                (Denoise, 1900, 240 * s, "dit block"),
                (Decode, 600, 700 + 50 * s, "vae tile"),
            ])
        };
        let mut previous = 0;
        for edge in [512u32, 768, 1024, 1536, 2048] {
            let d = derive(&a, &p, &at(u64::from(edge)), area(edge, edge, 1))
                .unwrap()
                .demand;
            assert!(d >= previous, "{edge}: {d} < {previous}");
            previous = d;
        }
        // Rung savings: each lever removed (more resident, bigger tile) prices at least as high.
        let rungs = [
            p.clone(),
            pricing(&[
                (Conditioning, 1200, 60, "te layer"),
                (Denoise, 1900, 240, "dit block"),
                (Decode, 600, 1400, "vae tile"),
            ]),
            pricing(&[
                (Conditioning, 1200, 60, "te layer"),
                (Denoise, 3100, 240, "dit block"),
                (Decode, 3100, 1400, "vae tile"),
            ]),
            pricing(&[
                (Conditioning, 4200, 60, "te layer"),
                (Denoise, 4200, 240, "dit block"),
                (Decode, 4200, 2600, "vae untiled"),
            ]),
        ];
        let derived: Vec<u64> = rungs
            .iter()
            .map(|r| derive(&a, &p, r, a.area).unwrap().demand)
            .collect();
        assert!(derived.windows(2).all(|w| w[0] <= w[1]), "{derived:?}");
        // The calibration applies only on the anchor's own binding quantum: the untiled decode is
        // priced at its formula, unscaled.
        let resident = derive(&a, &p, &rungs[3], a.area).unwrap();
        let decode = resident.phases.iter().find(|p| p.phase == Decode).unwrap();
        assert!(!decode.calibrated);
        assert_eq!(decode.bytes, (4200 + 2600) * MIB);
    }

    #[test]
    fn the_unpriced_overhead_is_floored_below_1024_and_linear_in_area_above() {
        // Below the knee: flat.
        assert_eq!(
            scale_overhead(100 * MIB, area(768, 768, 1), area(512, 512, 1)),
            100 * MIB
        );
        assert_eq!(
            scale_overhead(100 * MIB, area(1024, 1024, 1), area(512, 512, 1)),
            100 * MIB
        );
        // Above: proportional to megapixels over the knee (or over the anchor when it is above).
        assert_eq!(
            scale_overhead(100 * MIB, area(2048, 2048, 1), area(512, 512, 1)),
            400 * MIB
        );
        assert_eq!(
            scale_overhead(100 * MIB, area(1536, 1536, 1), area(512, 512, 1)),
            225 * MIB
        );
        assert_eq!(
            scale_overhead(100 * MIB, area(1024, 1024, 1), area(2048, 2048, 1)),
            25 * MIB
        );
        // Through the law: the denoise overhead (110 MiB) is carried flat to 1024², x4 at 2048².
        let (a, p) = optimized_512();
        let d = |edge: u32| {
            derive(&a, &p, &p, area(edge, edge, 1))
                .unwrap()
                .phases
                .into_iter()
                .find(|p| p.phase == Denoise)
                .unwrap()
                .bytes
        };
        assert_eq!(d(1024), d(512));
        assert_eq!(d(2048) - d(512), 3 * 110 * MIB);
    }

    #[test]
    fn a_phase_outside_the_domain_or_never_anchored_is_not_derivable() {
        let (mut a, p) = optimized_512();
        a.phases.insert(Denoise, 1800 * MIB);
        assert_eq!(
            derive(&a, &p, &p, a.area),
            Err(NotDerivable::OutsideDomain {
                phase: Denoise,
                measured: 1800 * MIB,
                resident: 1900 * MIB,
            })
        );
        let (a, p) = optimized_512();
        let with_encode = pricing(&[
            (Conditioning, 1200, 60, "te layer"),
            (Encode, 600, 300, "vae encode"),
            (Denoise, 1900, 240, "dit block"),
            (Decode, 600, 700, "vae tile"),
        ]);
        assert_eq!(
            derive(&a, &p, &with_encode, a.area),
            Err(NotDerivable::PhaseNotAnchored(Encode))
        );
    }

    #[test]
    fn two_anchors_check_each_other_and_an_under_prediction_fails_closed() {
        let (opt, p_opt) = optimized_512();
        let p_res = pricing(&[
            (Conditioning, 4200, 60, "te layer"),
            (Denoise, 4200, 240, "dit block"),
            (Decode, 4200, 1400, "vae untiled"),
        ]);
        // The optimized anchor predicts: conditioning 4200+40+60 = 4300, denoise 4200+110+240 =
        // 4550, decode 4200 + 0 + 1400 = 5600 → 5600 MiB.
        let consistent = anchor(5500, &[], 512);
        let c = consistency(&opt, &p_opt, &consistent, &p_res).unwrap();
        assert_eq!(c.predicted, 5600 * MIB);
        assert!(!c.flagged() && !c.fails_closed());
        // The resident anchor measured 7000: the law under-predicts by 20% → fail closed.
        let under = consistency(&opt, &p_opt, &anchor(7000, &[], 512), &p_res).unwrap();
        assert!(under.flagged() && under.fails_closed());
        // It measured 4000: the law over-predicts by 40% → flagged, but errs large.
        let over = consistency(&opt, &p_opt, &anchor(4000, &[], 512), &p_res).unwrap();
        assert!(over.flagged() && !over.fails_closed());
    }
}

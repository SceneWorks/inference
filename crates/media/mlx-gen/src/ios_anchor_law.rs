//! The iOS anchor law (epic 16774), shared by admission (`mlx-gen-ios-catalog`) and the providers'
//! CPU additivity proofs: a measured device anchor plus analytic derivation is the whole
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
//!              + overhead_p × λ                           unpriced overhead, area-scaled (below)
//!              + S_p(request) × κ_p                       the request's counted activation, calibrated
//!   κ_p        = s_p / S_p(anchor)  if the request's binding quantum IS the anchor's, else 1
//!   demand     = max_p D_p                                phases run in sequence
//! ```
//!
//! * `M_p` is the anchor's measured kernel-footprint growth in phase `p` (an unattributed phase is
//!   priced at the whole-run growth, an upper bound). `C_p` / `S_p` are the counted resident bytes
//!   and largest counted quantum of the phase under each regime and geometry.
//! * `κ_p ≤ 1` is the measured calibration of a counted formula: where the formula over-prices
//!   what the device measured, the request on THE SAME quantum at or below the anchor's size is
//!   priced at the measured ratio, so admission near the anchor is the measurement, not the
//!   formula's conservative scratch. A different binding quantum, or the same quantum LARGER than
//!   at the anchor (the measured/priced ratio rises with size — Iris's CPU proofs show 0.75 at the
//!   anchor, 0.82 at batch 2, 0.95 at 16x area), gets its formula unscaled (an upper bound).
//! * `λ = max(A_request, A_anchor) / A_anchor`: the unpriced overhead is linear in area ABOVE the
//!   anchor and FLOORED below it (never credited for a smaller request). Measured on the desktop
//!   MLX lane (SceneWorks `docs/sc-16195`: the activation transient of 7 tiers at 5 geometries) the
//!   transient is proportional to megapixels above 1024² (≤ 2.0% from a 1024² anchor) and has a
//!   floor below it. The iOS anchors sit at 1024², where this is exactly "linear above 1024², floor
//!   below". It is written relative to the anchor rather than to a fixed 1024² knee because the
//!   knee form, applied from an anchor below 1024², holds the overhead flat up to 1024² — which
//!   the same sweep shows UNDER-predicts 512² → 1024² by 49–70%; anchor-relative linear growth
//!   over-predicts it instead (test `the_area_law_against_the_measured_desktop_sweep`).
//! * DOMAIN: a phase whose measured growth is below the components its regime holds resident has
//!   left the law's domain (the counters did not see the weights) and is NOT derivable — never
//!   clamped to zero.
//! * EXTRA PHASES. A component-additive mode (img2img / reference / mask, ControlNet, IP-Adapter)
//!   adds components — re-added through `C_p(request)` — and may add a phase the anchor never ran
//!   (the VAE / control-hint ENCODE). That phase is priced by its guarded counted formula at full
//!   price plus the largest unpriced overhead the anchor measured. Whether a mode IS additive is a
//!   per-provider claim each provider proves on its CPU fixtures with [`measure_phases`] (measured
//!   mode peak ≤ the peak derived from the text-mode run); a mode whose proof fails is its own anchor
//!   family.
//!
//! At the anchor's own configuration the law returns the anchor's measurement exactly
//! (`C + (r − s) + s = M`). With both anchors measured, the optimized anchor predicts the resident
//! one: that is the law's free self-check ([`consistency`]).

use crate::ios_pricing::{IosPhase, IosPricing};
use std::collections::BTreeMap;

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

/// The unpriced-overhead area scaler `λ = max(A_request, A_anchor) / A_anchor` applied to `bytes`.
pub fn scale_overhead(bytes: u64, request_area: u64, anchor_area: u64) -> u64 {
    mul_div_ceil(bytes, request_area.max(anchor_area), anchor_area.max(1))
}

/// Derive `request`'s demand from `anchor`, measured in the configuration `at_anchor` prices.
pub fn derive(
    anchor: &MeasuredAnchor,
    at_anchor: &IosPricing,
    request: &IosPricing,
    request_area: u64,
) -> Result<Derivation, NotDerivable> {
    // The anchor's measured residue per phase it ran: (priced part, unpriced overhead).
    let mut measured = BTreeMap::new();
    for price in &at_anchor.phases {
        let growth = anchor.growth(price.phase);
        let residue =
            growth
                .checked_sub(price.resident_bytes)
                .ok_or(NotDerivable::OutsideDomain {
                    phase: price.phase,
                    measured: growth,
                    resident: price.resident_bytes,
                })?;
        let priced = residue.min(price.scratch_bytes);
        measured.insert(price.phase, (priced, residue - priced));
    }
    // A phase the anchor never ran (a mode's extra phase — the img2img / reference / control-hint
    // ENCODE) is priced by its guarded counted formula at full price (κ = 1), plus the largest
    // unpriced overhead the anchor measured in any phase (the process's own allocator/driver
    // share, area-scaled): component-additive, and erring large.
    let worst_overhead = measured.values().map(|(_, o)| *o).max().unwrap_or(0);
    let mut phases = Vec::with_capacity(request.phases.len());
    for wanted in &request.phases {
        let (bytes, calibrated) = match (at_anchor.phase(wanted.phase), measured.get(&wanted.phase))
        {
            (Some(measured_at), Some(&(priced, overhead))) => {
                // The device's calibration of a formula holds at and BELOW the size it was measured
                // at: the measured/priced ratio of a quantum rises with its size (fixed costs the
                // formula over-prices weigh less in a bigger quantum), so a ratio learned on a
                // smaller quantum would under-price a larger one. Above the anchor's quantum the
                // formula — an upper bound — prices it unscaled.
                let calibrated = wanted.binding == measured_at.binding
                    && measured_at.scratch_bytes > 0
                    && wanted.scratch_bytes <= measured_at.scratch_bytes;
                let activation = if calibrated {
                    mul_div_ceil(wanted.scratch_bytes, priced, measured_at.scratch_bytes)
                } else {
                    wanted.scratch_bytes
                };
                (
                    wanted
                        .resident_bytes
                        .saturating_add(scale_overhead(overhead, request_area, anchor.area))
                        .saturating_add(activation),
                    calibrated,
                )
            }
            _ => (
                wanted
                    .resident_bytes
                    .saturating_add(scale_overhead(worst_overhead, request_area, anchor.area))
                    .saturating_add(wanted.scratch_bytes),
                false,
            ),
        };
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

/// A CPU-fixture run's measured growth, total and per phase: the host stand-in for a device anchor
/// capture, for the providers' additivity proofs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredRun {
    /// Largest MLX active-memory growth over the run's starting active memory.
    pub demand: u64,
    /// The same, per phase (attributed through the quantum marks exactly as a device capture is).
    pub phases: BTreeMap<IosPhase, u64>,
}

impl MeasuredRun {
    /// This run as an anchor at `area` output pixels.
    pub fn anchor(&self, area: u64) -> MeasuredAnchor {
        MeasuredAnchor {
            demand: self.demand,
            phases: self.phases.clone(),
            area,
        }
    }
}

/// Run `body` (a load + generate on a tiny CPU fixture, inside
/// `ios_quantum::with_cpu_diagnostic_headroom`) and measure its MLX active-memory growth per
/// phase. At every quantum check / synchronization the MLX peak since the previous mark is read and
/// reset; growth inside an opened quantum belongs to that quantum's phase (`phase_of(label)`), glue
/// after a synchronization to the next mark's. Unavailable on iOS (a device capture reads the kernel
/// footprint instead).
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
    // (phase of the interval in progress, whether the last mark opened a quantum)
    type State = (Option<IosPhase>, bool, BTreeMap<IosPhase, u64>);
    let state: Rc<RefCell<State>> = Rc::new(RefCell::new((None, false, BTreeMap::new())));
    // `reset_peak_memory` zeroes the peak: an interval with no allocation reads 0, so the reading
    // is never below what is active now.
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
    // The tail after the last mark belongs to the last mark's phase.
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
    use crate::ios_pricing::IosPhase::*;

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

    /// The calibration applies at and below the anchor's quantum size only: the same binding
    /// quantum priced LARGER than at the anchor gets its formula unscaled.
    #[test]
    fn the_calibration_never_scales_a_quantum_above_the_anchors_size() {
        let (a, p) = optimized_512();
        let decode = |scratch: u64| {
            let r = pricing(&[
                (Conditioning, 1200, 60, "te layer"),
                (Denoise, 1900, 240, "dit block"),
                (Decode, 600, scratch, "vae tile"),
            ]);
            derive(&a, &p, &r, a.area)
                .unwrap()
                .phases
                .into_iter()
                .find(|x| x.phase == Decode)
                .unwrap()
        };
        // Below: 350 × 400/700 = 200 calibrated.
        let below = decode(350);
        assert!(below.calibrated);
        assert_eq!(below.bytes, (600 + 200) * MIB);
        // Above: 1400 at full price, never 800.
        let above = decode(1400);
        assert!(!above.calibrated);
        assert_eq!(above.bytes, (600 + 1400) * MIB);
    }

    #[test]
    fn the_unpriced_overhead_is_floored_below_the_anchor_and_linear_in_area_above_it() {
        // A 1024² anchor (the iOS reference geometry): flat below, proportional above.
        let k = area(1024, 1024, 1);
        assert_eq!(scale_overhead(100 * MIB, area(512, 512, 1), k), 100 * MIB);
        assert_eq!(scale_overhead(100 * MIB, area(768, 768, 1), k), 100 * MIB);
        assert_eq!(scale_overhead(100 * MIB, k, k), 100 * MIB);
        assert_eq!(scale_overhead(100 * MIB, area(2048, 2048, 1), k), 400 * MIB);
        assert_eq!(scale_overhead(100 * MIB, area(1536, 1536, 1), k), 225 * MIB);
        // Batch multiplies the area.
        assert_eq!(scale_overhead(100 * MIB, area(1024, 1024, 2), k), 200 * MIB);
        // Through the law from a 512² anchor: the denoise overhead (110 MiB) grows with the area
        // above the anchor — never held flat below a fixed knee.
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
        assert_eq!(d(256), d(512));
        assert_eq!(d(1024) - d(512), 3 * 110 * MIB);
    }

    /// VALIDATION against measured data: SceneWorks `docs/sc-16195/resolution-sweep.json`, the
    /// desktop MLX activation transient (bytes) of 7 tiers at 5 geometries. Taken as if NOTHING were
    /// priced (the whole transient as unpriced overhead — the law's worst case), the area law from
    /// a 1024² anchor is within 2.0% above it and errs large below it; it NEVER under-predicts from
    /// either anchor geometry. The 1024²-knee form from a 512² anchor would under-predict by up to
    /// 70.3% — which is why λ is anchor-relative.
    #[test]
    fn the_area_law_against_the_measured_desktop_sweep() {
        const GEOMETRIES: [(u32, u32); 5] = [
            (512, 512),
            (1024, 1024),
            (1024, 1536),
            (1152, 2048),
            (2048, 2048),
        ];
        #[rustfmt::skip]
        const SWEEP: [(&str, [u64; 5]); 6] = [
            ("illustrious_xl_v1 q8", [4598180324, 15074781464, 22569658648, 33837312280, 59121494296]),
            ("krea_2_turbo bf16/q8", [4208132200, 8232829032, 12330385512, 18484609128, 32863912040]),
            ("lens q4", [4731765416, 15080292984, 22577119864, 33847739000, 59138802296]),
            ("qwen_image q8", [4204138088, 8226047592, 12323079784, 18476516968, 32853984616]),
            ("sdxl bf16", [4598011364, 15074497816, 22569293080, 33836864792, 59120817432]),
            ("z_image_turbo q4", [4477811836, 15078788688, 22574042704, 33842302544, 59127860816]),
        ];
        let areas = GEOMETRIES.map(|(w, h)| area(w, h, 1));
        let error =
            |predicted: u64, measured: u64| (predicted as f64 - measured as f64) / measured as f64;
        let (mut worst_above_1024, mut worst_from_512, mut knee_from_512) = (0f64, 0f64, 0f64);
        for (name, transient) in SWEEP {
            for anchor in [0usize, 1] {
                for (i, &measured) in transient.iter().enumerate() {
                    let predicted = scale_overhead(transient[anchor], areas[i], areas[anchor]);
                    let e = error(predicted, measured);
                    assert!(
                        e >= -1e-9,
                        "{name}: anchor {anchor} under-predicts geometry {i} by {e}"
                    );
                    if anchor == 1 && i > 1 {
                        worst_above_1024 = worst_above_1024.max(e);
                    }
                    if anchor == 0 && i > 0 {
                        worst_from_512 = worst_from_512.max(e);
                        let knee = mul_div_ceil(
                            transient[0],
                            areas[i].max(areas[1]),
                            areas[0].max(areas[1]),
                        );
                        knee_from_512 = knee_from_512.min(error(knee, measured));
                    }
                }
            }
        }
        assert!(worst_above_1024 < 0.021, "{worst_above_1024}");
        assert!(worst_from_512 < 1.05, "{worst_from_512}");
        assert!(knee_from_512 < -0.70, "{knee_from_512}");
    }

    #[test]
    fn a_phase_outside_the_domain_is_not_derivable_and_an_extra_phase_is_priced_additively() {
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
        // A mode's extra ENCODE phase (component-additive): its resident components + its counted
        // formula at full price + the anchor's largest unpriced overhead (denoise, 110 MiB).
        let (a, p) = optimized_512();
        let with_encode = pricing(&[
            (Conditioning, 1200, 60, "te layer"),
            (Encode, 600, 300, "vae encode"),
            (Denoise, 1900, 240, "dit block"),
            (Decode, 600, 700, "vae tile"),
        ]);
        let d = derive(&a, &p, &with_encode, a.area).unwrap();
        let encode = d.phases.iter().find(|x| x.phase == Encode).unwrap();
        assert_eq!(encode.bytes, (600 + 300 + 110) * MIB);
        assert!(!encode.calibrated);
        assert_eq!(d.demand, a.demand, "the anchored phases are unchanged");
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

/// The CPU stand-in for a device capture attributes growth to the phase of the quantum it ran in.
#[cfg(all(test, target_os = "macos"))]
#[test]
fn measure_phases_attributes_growth_to_each_quantums_phase() {
    const CHILD: &str = "MLX_IOS_MEASURE_PHASES_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ios_anchor_law::measure_phases_attributes_growth_to_each_quantums_phase",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::ios_quantum::quantum_metal_guard::forbid_metal());
    fn phase_of(label: &str) -> Option<IosPhase> {
        match label {
            "small" => Some(IosPhase::Conditioning),
            "big" => Some(IosPhase::Denoise),
            _ => None,
        }
    }
    let ((), run) = crate::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || {
        measure_phases(phase_of, || {
            let ones = |n: i32| mlx_rs::ops::ones::<f32>(&[n]);
            let a = crate::ios_quantum::run("small", 1 << 20, || {
                Ok(ones(1 << 16)?.multiply(mlx_rs::Array::from_f32(2.0))?)
            })?;
            let b = crate::ios_quantum::run("big", 1 << 24, || {
                Ok(ones(1 << 21)?.multiply(mlx_rs::Array::from_f32(3.0))?)
            })?;
            drop((a, b));
            Ok(())
        })
    })
    .unwrap();
    let small = run.phases[&IosPhase::Conditioning];
    let big = run.phases[&IosPhase::Denoise];
    assert!(small >= 1 << 18, "{small}");
    assert!(big >= 1 << 23, "{big}");
    assert!(big > small);
    assert_eq!(run.demand, big.max(small));
}

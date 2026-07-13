//! System-aware **memory-adaptation policy** (sc-11750) — the backend-neutral escalation core that
//! decides WHICH constrained-memory levers a generation engages, and at what setting, from a device
//! budget plus shape-derived peak estimates.
//!
//! This generalizes the Wan `estimate → gate → size` seam (sc-4998; the VAE-tiling half now lives in
//! [`crate::tiling::budgeted_plan`]) one level up: a real render has **more than one** lever, they cost
//! different amounts (a residency flip is nearly free; resolution reduction costs quality), and they
//! target **different peaks** (the resident footprint, the denoise steady state, the decode spike). The
//! requirement (sc-11750) is that a large-memory machine pays *zero* overhead — full res, `Resident`,
//! bf16 branch, single-pass decode — while a constrained machine engages levers only as far as needed,
//! in cost order, at the lightest sufficient setting.
//!
//! [`plan_memory_adaptation`] is that decision. It is **pure** (the caller injects the budget and a
//! shape-derived peak estimator, exactly like `budgeted_plan` injects `safe_gib` + `peak_cost`), so it
//! keeps gen-core's zero-tensor-dep / Linux-buildable invariant and both backends (mlx-gen on Metal,
//! candle-gen on CUDA) share one escalation ladder. The per-lane peak constants live in the caller's
//! estimator; this layer holds only the **order** and the budget comparison.
//!
//! ## The ladder (cheapest first — the sc-11750 order)
//! 1. [`Lever::SequentialResidency`] — drop the phase-A text/vision encoder before the heavy load, so
//!    it no longer co-resides with the DiT/VAE. Frees `text_resident_gib` from **both** the denoise and
//!    decode peaks. Nearly free (only the cross-request weight cache is lost), so it is tried first.
//! 2. [`Lever::VaeDecodeTiling`] — split the VAE decode into tiles so its spike drops toward the
//!    un-tileable floor (resident VAE + output buffers). Reduces the **decode** peak only; the actual
//!    tile is then sized by [`crate::tiling::budgeted_plan`].
//! 3. [`Lever::BranchQuant`] — pack the control branch to the base tier at **load time** (it cannot be
//!    re-packed mid-render). Reduces the **denoise** peak only.
//! 4. [`Lever::ResolutionReduction`] — the last resort, because it is the only lever that costs image
//!    quality. Re-estimates *both* peaks at a smaller resolution scale, engaged only when levers 1–3 at
//!    full resolution still don't fit.

use crate::runtime::OffloadPolicy;

/// One constrained-memory lever, in cost order (cheapest / least-quality-cost first). The `u8`
/// discriminants make the cost order explicit and let a test assert the engaged levers came out
/// ascending (never a more-expensive lever before a cheaper one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lever {
    /// Drop the text/vision encoder before the heavy load (`OffloadPolicy::Sequential`).
    SequentialResidency = 0,
    /// Tile the VAE decode so its spike drops toward the resident-VAE + output-buffer floor.
    VaeDecodeTiling = 1,
    /// Pack the control branch to the base quant tier at load time.
    BranchQuant = 2,
    /// Reduce the render resolution (the only quality-costing lever) — last resort.
    ResolutionReduction = 3,
}

/// The stage peaks of one generation at a given resolution scale, in GiB — all shape-derived estimates
/// the caller computes, and all **excluding** the phase-A text encoder (its footprint is carried once by
/// [`LaneLevers::text_resident_gib`], since the residency lever is what adds or removes it). A render
/// "fits" when every peak here, plus the text term when `Resident`, is within the safe budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StagePeaks {
    /// Denoise-stage peak: the heavy resident weights (DiT base + bf16 branch) plus the per-step
    /// activation working set. Excludes the text encoder.
    pub denoise_ex_text_gib: f64,
    /// Single-pass VAE-decode peak: the resident VAE (+ any co-resident heavy weights) plus the
    /// full-output decode spike. Excludes the text encoder.
    pub decode_ex_text_gib: f64,
    /// The least a **tiled** decode can peak at (resident VAE + output buffers + one minimal tile) —
    /// the floor [`Lever::VaeDecodeTiling`] drives toward, excluding the text encoder. `None` when the
    /// lane cannot tile its decode (then the tiling lever is unavailable). Scales with resolution like
    /// the other peaks, so the caller recomputes it per scale.
    pub decode_tiled_floor_ex_text_gib: Option<f64>,
}

/// Which levers a lane offers the policy, and how much each saves. A lever the lane cannot apply is
/// signalled by its "unavailable" sentinel (`text_resident_gib == 0` / `!supports_sequential`,
/// `branch_quant_saves_gib == 0`, an empty `resolution_scales`, or a `None`
/// [`StagePeaks::decode_tiled_floor_ex_text_gib`]); the policy simply skips it.
#[derive(Clone, Copy, Debug)]
pub struct LaneLevers<'a> {
    /// The phase-A text/vision encoder's resident footprint (GiB) — the amount
    /// [`Lever::SequentialResidency`] frees from **both** stage peaks by dropping it before the heavy
    /// load. `0.0` ⇒ the lane has no separable text phase to drop.
    pub text_resident_gib: f64,
    /// Whether the lane actually honors `OffloadPolicy::Sequential` (a provider that has not wired it
    /// treats `Sequential` as `Resident`, so the lever would be inert). `false` ⇒ residency unavailable
    /// even if `text_resident_gib > 0`.
    pub supports_sequential: bool,
    /// GiB [`Lever::BranchQuant`] removes from the peaks by packing the resident control branch bf16 →
    /// base tier. The branch is resident through **both** stages (it injects during denoise, sits
    /// allocated during decode), so this is subtracted from both the denoise and decode peaks — the
    /// tiled decode floor included. `0.0` ⇒ no separable branch to pack (or it is already packed).
    pub branch_quant_saves_gib: f64,
    /// Candidate resolution scales in `(0, 1)`, **largest first** (e.g. `[0.75, 0.5]`), tried only as
    /// the last resort. Empty ⇒ the lane will not reduce resolution. The policy re-estimates the stage
    /// peaks at each via the `peaks_at` closure.
    pub resolution_scales: &'a [f64],
}

impl LaneLevers<'_> {
    /// Whether the residency lever is actually usable (a text phase exists *and* the lane honors
    /// `Sequential`).
    fn residency_available(&self) -> bool {
        self.supports_sequential && self.text_resident_gib > 0.0
    }
}

/// The decision [`plan_memory_adaptation`] returns: the settings a lane applies to fit the budget, plus
/// the cost-ordered list of levers that had to engage and whether the render fits at all.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPlan {
    /// The residency to load under. `Sequential` iff [`Lever::SequentialResidency`] engaged.
    pub residency: OffloadPolicy,
    /// Whether to tile the VAE decode ([`Lever::VaeDecodeTiling`]). When `true` the caller sizes the
    /// actual tile with [`crate::tiling::budgeted_plan`] against `safe_gib`; when `false` it runs the
    /// single-pass decode.
    pub tile_decode: bool,
    /// Whether to pack the control branch to the base tier at load ([`Lever::BranchQuant`]).
    pub quantize_branch: bool,
    /// The resolution scale to render at — `1.0` unless [`Lever::ResolutionReduction`] engaged, then
    /// the largest candidate scale that fits.
    pub resolution_scale: f64,
    /// The projected concurrent peak (GiB) after the engaged levers — `max(denoise, decode)` at the
    /// chosen settings. When `tile_decode` is set the decode term is the tiled *floor* (the best case);
    /// `budgeted_plan` then picks the largest tile whose real peak stays ≤ `safe_gib`.
    pub projected_peak_gib: f64,
    /// The levers that engaged, in ascending cost order. Empty ⇒ the fast path (nothing engaged): full
    /// res, `Resident`, bf16 branch, single-pass decode — the large-memory-machine case.
    pub engaged: Vec<Lever>,
    /// Whether the render fits the budget with the chosen settings. `false` ⇒ even the smallest scale
    /// with every lever engaged still peaks over `safe_gib`; the caller surfaces a catchable error
    /// (with `projected_peak_gib`) rather than letting the OS/GPU kill the process mid-render.
    pub feasible: bool,
}

impl MemoryPlan {
    /// The zero-overhead plan: nothing engaged (full res, `Resident`, bf16 branch, single-pass decode).
    fn none(projected_peak_gib: f64) -> Self {
        Self {
            residency: OffloadPolicy::Resident,
            tile_decode: false,
            quantize_branch: false,
            resolution_scale: 1.0,
            projected_peak_gib,
            engaged: Vec::new(),
            feasible: true,
        }
    }
}

/// The outcome of engaging levers 1–3 at one resolution scale: the resulting stage peaks and which
/// levers were needed. Internal to the escalation.
struct Attempt {
    denoise: f64,
    decode: f64,
    engaged: Vec<Lever>,
}

impl Attempt {
    fn fits(&self, safe_gib: f64) -> bool {
        self.denoise <= safe_gib && self.decode <= safe_gib
    }
    fn peak(&self) -> f64 {
        self.denoise.max(self.decode)
    }
}

/// Engage the cost-ordered levers 1–3 (residency → decode tiling → branch quant) at a fixed resolution
/// scale, each **only** when the peak it targets is over budget, and report the resulting peaks. This is
/// the "lightest sufficient setting" core: a lever that isn't needed (or isn't available) is skipped.
fn engage_levers_at(peaks: StagePeaks, levers: &LaneLevers<'_>, safe_gib: f64) -> Attempt {
    let text = levers.text_resident_gib;
    let mut engaged = Vec::new();

    // Baseline: `Resident`, so the text encoder co-resides through both stages.
    let mut sequential = false;
    let mut denoise = peaks.denoise_ex_text_gib + text;
    let mut decode = peaks.decode_ex_text_gib + text;

    // Lever 1 — Sequential residency (cheapest): drop the text encoder from both peaks. Engage when
    // either stage is over budget and the lane offers it.
    if (denoise > safe_gib || decode > safe_gib) && levers.residency_available() {
        sequential = true;
        denoise = peaks.denoise_ex_text_gib;
        decode = peaks.decode_ex_text_gib;
        engaged.push(Lever::SequentialResidency);
    }
    // The text term still co-resides through the decode when residency did NOT engage.
    let resident_text = if sequential { 0.0 } else { text };

    // Lever 2 — VAE decode tiling: drop the decode peak to its tiled floor when the decode stage is
    // still over budget. The floor is the best a tile can do; `budgeted_plan` sizes the real tile.
    if decode > safe_gib {
        if let Some(floor) = peaks.decode_tiled_floor_ex_text_gib {
            decode = floor + resident_text;
            engaged.push(Lever::VaeDecodeTiling);
        }
    }

    // Lever 3 — branch quant: pack the branch to the base tier at load. The branch is a **resident
    // weight**, held through both the denoise (where it injects) and the decode (where it sits idle but
    // still allocated) stages, so packing it lowers BOTH peaks — including the tiled decode floor, which
    // carries the resident branch. Its primary trigger is the denoise steady state (sc-11750), but it
    // also engages when the post-tiling decode is still over budget (tiling shrank the spike but the
    // resident weights are still too heavy). Cost-ordered after tiling: a spike big enough to tile is
    // handled by the free-quality tiling lever first.
    if (denoise > safe_gib || decode > safe_gib) && levers.branch_quant_saves_gib > 0.0 {
        denoise = (denoise - levers.branch_quant_saves_gib).max(0.0);
        decode = (decode - levers.branch_quant_saves_gib).max(0.0);
        engaged.push(Lever::BranchQuant);
    }

    Attempt {
        denoise,
        decode,
        engaged,
    }
}

/// Turn an [`Attempt`] at a given scale into a full [`MemoryPlan`], marking resolution reduction when
/// the scale is below full.
fn plan_from_attempt(attempt: Attempt, scale: f64, feasible: bool) -> MemoryPlan {
    let projected_peak_gib = attempt.peak();
    let mut engaged = attempt.engaged;
    let sequential = engaged.contains(&Lever::SequentialResidency);
    let tile_decode = engaged.contains(&Lever::VaeDecodeTiling);
    let quantize_branch = engaged.contains(&Lever::BranchQuant);
    if scale < 1.0 {
        engaged.push(Lever::ResolutionReduction);
    }
    MemoryPlan {
        residency: if sequential {
            OffloadPolicy::Sequential
        } else {
            OffloadPolicy::Resident
        },
        tile_decode,
        quantize_branch,
        resolution_scale: scale,
        projected_peak_gib,
        engaged,
        feasible,
    }
}

/// Decide the memory-adaptation plan for one generation (sc-11750). Given the safe device budget, the
/// levers the lane offers, and a `peaks_at(scale)` estimator that returns the shape-derived
/// [`StagePeaks`] at a resolution scale, returns the [`MemoryPlan`]: the lightest cost-ordered set of
/// levers that fits the budget.
///
/// The escalation:
///   1. **Fast path** — if full resolution fits `Resident` with no levers (the large-memory machine),
///      return [`MemoryPlan`] with `engaged` empty: zero overhead, full quality.
///   2. Otherwise engage levers 1–3 (residency → decode tiling → branch quant) at **full resolution**,
///      each only when the peak it targets is over budget. If that fits, done — no quality cost.
///   3. Only if full-res-with-all-levers still doesn't fit, try each `resolution_scales` candidate
///      (largest first), re-engaging levers 1–3 at that scale, and take the first that fits.
///   4. If nothing fits even at the smallest scale, return the smallest-scale attempt with
///      `feasible = false` so the caller errors *before* the render (with `projected_peak_gib`).
///
/// `peaks_at` is called with `1.0` for full resolution and with each entry of
/// [`LaneLevers::resolution_scales`]; a lane with no resolution lever supplies an empty slice and
/// `peaks_at` is only ever called with `1.0`.
pub fn plan_memory_adaptation(
    safe_gib: f64,
    levers: LaneLevers<'_>,
    peaks_at: impl Fn(f64) -> StagePeaks,
) -> MemoryPlan {
    // Fast path: full res, Resident, no levers. This is the whole point — a machine with headroom pays
    // nothing (no tiling passes, no dequant-on-forward, no quality loss).
    let full = peaks_at(1.0);
    let text = levers.text_resident_gib;
    let resident_denoise = full.denoise_ex_text_gib + text;
    let resident_decode = full.decode_ex_text_gib + text;
    if resident_denoise <= safe_gib && resident_decode <= safe_gib {
        return MemoryPlan::none(resident_denoise.max(resident_decode));
    }

    // Escalate. Try full resolution with levers 1–3 first (no quality cost), then each smaller scale.
    // `1.0` leads so a fit there never reduces resolution; the smaller scales are the last resort.
    let full_attempt = engage_levers_at(full, &levers, safe_gib);
    if full_attempt.fits(safe_gib) {
        return plan_from_attempt(full_attempt, 1.0, true);
    }

    let mut last = full_attempt;
    let mut last_scale = 1.0;
    for &scale in levers.resolution_scales {
        let attempt = engage_levers_at(peaks_at(scale), &levers, safe_gib);
        if attempt.fits(safe_gib) {
            return plan_from_attempt(attempt, scale, true);
        }
        last = attempt;
        last_scale = scale;
    }

    // Nothing fit — return the smallest-scale, fully-engaged attempt as infeasible so the caller can
    // surface a catchable over-budget error with the projected peak.
    plan_from_attempt(last, last_scale, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lane profile with every lever available, and a peak model that scales the activation/decode
    /// terms with the resolution scale² (area) while the resident-weight terms stay fixed — the shape a
    /// real render has. `dn0`/`dc0` are the full-res ex-text denoise/decode peaks; `floor0` the tiled
    /// decode floor; `wt`/`vae` the resolution-independent resident weight terms inside them.
    struct Model {
        dn_weight: f64,   // res-independent denoise resident (DiT + branch), ex-text
        dn_act: f64,      // full-res denoise activation term (scales with area)
        dc_vae: f64,      // res-independent decode resident (VAE), ex-text
        dc_spike: f64,    // full-res decode spike (scales with area)
        floor_extra: f64, // tiled floor add over dc_vae (min tile + output buffers, scales with area)
        tileable: bool,
    }

    impl Model {
        fn peaks_at(&self, scale: f64) -> StagePeaks {
            let area = scale * scale;
            StagePeaks {
                denoise_ex_text_gib: self.dn_weight + self.dn_act * area,
                decode_ex_text_gib: self.dc_vae + self.dc_spike * area,
                decode_tiled_floor_ex_text_gib: self
                    .tileable
                    .then_some(self.dc_vae + self.floor_extra * area),
            }
        }
    }

    // A Krea-control-shaped model (candle #480 ballpark, ex-text): denoise ≈ base+branch (~9) + ~11
    // activations = ~20; decode ≈ VAE (~1) + ~30 spike = ~31; tiled floor ≈ VAE + ~4 = ~5. Text ~8.
    fn krea_model() -> Model {
        Model {
            dn_weight: 9.0,
            dn_act: 11.0,
            dc_vae: 1.0,
            dc_spike: 30.0,
            floor_extra: 4.0,
            tileable: true,
        }
    }

    const SCALES: [f64; 2] = [0.75, 0.5];

    fn krea_levers() -> LaneLevers<'static> {
        LaneLevers {
            text_resident_gib: 8.0,
            supports_sequential: true,
            branch_quant_saves_gib: 4.6, // bf16 branch ~6.6 → q4 ~2
            resolution_scales: &SCALES,
        }
    }

    /// Large-memory machine (128 GB → ~100 GiB safe): NOTHING engages. The fast-path regression guard.
    #[test]
    fn large_memory_engages_nothing() {
        let m = krea_model();
        let plan = plan_memory_adaptation(100.0, krea_levers(), |s| m.peaks_at(s));
        assert!(plan.engaged.is_empty(), "no lever should engage: {plan:?}");
        assert_eq!(plan.residency, OffloadPolicy::Resident);
        assert!(!plan.tile_decode);
        assert!(!plan.quantize_branch);
        assert_eq!(plan.resolution_scale, 1.0);
        assert!(plan.feasible);
    }

    /// The engaged levers always come out in ascending cost order (never quant before residency, etc.).
    #[test]
    fn engaged_levers_are_cost_ordered() {
        let m = krea_model();
        // A tight budget that forces several levers.
        let plan = plan_memory_adaptation(12.0, krea_levers(), |s| m.peaks_at(s));
        let mut sorted = plan.engaged.clone();
        sorted.sort();
        assert_eq!(
            plan.engaged, sorted,
            "levers must be in cost order: {plan:?}"
        );
    }

    /// A budget just under the resident peak but above the Sequential peak: residency ALONE fixes it —
    /// the cheapest lever, and no tiling/quant/res engaged.
    #[test]
    fn residency_alone_when_it_suffices() {
        let m = krea_model();
        // Full-res decode ex-text = 31; +8 text = 39 resident. Sequential decode = 31. Pick a budget
        // ≥ max(denoise_ex_text=20, decode_ex_text=31) but < resident peak (39). 32 works.
        let plan = plan_memory_adaptation(32.0, krea_levers(), |s| m.peaks_at(s));
        assert_eq!(plan.engaged, vec![Lever::SequentialResidency], "{plan:?}");
        assert_eq!(plan.residency, OffloadPolicy::Sequential);
        assert!(!plan.tile_decode && !plan.quantize_branch);
        assert_eq!(plan.resolution_scale, 1.0);
        assert!(plan.feasible && plan.projected_peak_gib <= 32.0);
    }

    /// Decode over budget but denoise fine after residency: tiling engages, quant does NOT (denoise
    /// already fits). Proves each lever targets its own peak.
    #[test]
    fn tiles_decode_without_quantizing_when_denoise_fits() {
        let m = krea_model();
        // Sequential: denoise=20, decode=31, tiled floor=5. Budget 22: denoise(20)≤22 fits, decode(31)
        // >22 → tile to floor(5). Quant should NOT engage (denoise already ≤22).
        let plan = plan_memory_adaptation(22.0, krea_levers(), |s| m.peaks_at(s));
        assert!(plan.tile_decode, "decode must tile: {plan:?}");
        assert!(
            !plan.quantize_branch,
            "branch quant must NOT engage when denoise already fits: {plan:?}"
        );
        assert_eq!(
            plan.engaged,
            vec![Lever::SequentialResidency, Lever::VaeDecodeTiling]
        );
        assert!(plan.feasible);
    }

    /// A budget that forces residency + tiling + quant but NOT resolution (they fit at full res).
    #[test]
    fn quantizes_branch_before_touching_resolution() {
        let m = krea_model();
        // Sequential denoise=20, decode→floor 5. Budget 16: denoise(20)>16 → quant → 20-4.6=15.4≤16.
        // decode floor(5)≤16. Fits at full res, so NO resolution reduction.
        let plan = plan_memory_adaptation(16.0, krea_levers(), |s| m.peaks_at(s));
        assert!(plan.quantize_branch, "branch must quantize: {plan:?}");
        assert_eq!(
            plan.resolution_scale, 1.0,
            "must not reduce res yet: {plan:?}"
        );
        assert!(!plan.engaged.contains(&Lever::ResolutionReduction));
        assert!(plan.feasible);
    }

    /// A budget that levers 1–3 can't reach at full res, but a smaller resolution can: resolution is the
    /// LAST resort and only then engages.
    #[test]
    fn resolution_is_last_resort() {
        let m = krea_model();
        // Full-res Sequential+quant denoise = 20-4.6 = 15.4; decode floor 5. Budget 13: denoise 15.4>13
        // even quantized at full res → must drop resolution. At 0.75²: dn_act 11·0.5625=6.19 → denoise
        // ex-text 9+6.19=15.19, -4.6 quant = 10.59 ≤13; decode floor 1+4·0.5625=3.25 ≤13. Fits at 0.75.
        let plan = plan_memory_adaptation(13.0, krea_levers(), |s| m.peaks_at(s));
        assert!(
            plan.engaged.contains(&Lever::ResolutionReduction),
            "resolution must engage: {plan:?}"
        );
        assert_eq!(
            plan.resolution_scale, 0.75,
            "largest sufficient scale: {plan:?}"
        );
        assert!(plan.feasible);
        // Everything cheaper than resolution was already spent.
        assert!(plan.residency == OffloadPolicy::Sequential && plan.quantize_branch);
    }

    /// The smallest sufficient scale is chosen: a budget that 0.75 can't reach but 0.5 can lands on 0.5,
    /// not smaller — and never skips 0.75 if 0.75 would have worked.
    #[test]
    fn picks_largest_sufficient_resolution() {
        let m = krea_model();
        // Choose a budget where 0.75 fails but 0.5 works. At 0.5²=0.25: denoise 9+11·0.25=11.75, quant
        // →7.15; decode floor 1+4·0.25=2. Budget 8: 0.5 denoise 7.15≤8 ✓. 0.75 denoise 10.59>8 ✗.
        let plan = plan_memory_adaptation(8.0, krea_levers(), |s| m.peaks_at(s));
        assert_eq!(plan.resolution_scale, 0.5, "{plan:?}");
        assert!(plan.feasible);
    }

    /// Infeasible: even the smallest scale with every lever over budget → `feasible = false` and the
    /// projected peak is reported for the caller's catchable error.
    #[test]
    fn infeasible_reports_projected_peak() {
        let m = krea_model();
        // A budget below the tiled decode floor at the smallest scale is unreachable.
        let plan = plan_memory_adaptation(1.5, krea_levers(), |s| m.peaks_at(s));
        assert!(!plan.feasible, "must be infeasible: {plan:?}");
        assert!(plan.projected_peak_gib > 1.5);
    }

    /// A lane with NO levers available (a plain lane: no text phase, no branch, no tiling, no res
    /// candidates) either fits at full res with nothing, or reports infeasible — it never fabricates a
    /// lever it doesn't have.
    #[test]
    fn lane_without_levers_never_fabricates_one() {
        let peaks = StagePeaks {
            denoise_ex_text_gib: 20.0,
            decode_ex_text_gib: 10.0,
            decode_tiled_floor_ex_text_gib: None,
        };
        let bare = LaneLevers {
            text_resident_gib: 0.0,
            supports_sequential: false,
            branch_quant_saves_gib: 0.0,
            resolution_scales: &[],
        };
        // Fits: nothing engaged.
        let ok = plan_memory_adaptation(25.0, bare, |_| peaks);
        assert!(ok.engaged.is_empty() && ok.feasible);
        // Over budget with no lever to pull: infeasible, not a phantom tiling/quant.
        let no = plan_memory_adaptation(15.0, bare, |_| peaks);
        assert!(!no.feasible, "{no:?}");
        assert!(no.engaged.is_empty(), "no lever exists to engage: {no:?}");
    }

    /// Residency unavailable (provider hasn't wired Sequential) ⇒ the text term stays co-resident and
    /// the policy leans on the other levers instead of a lever that would be inert.
    #[test]
    fn skips_residency_when_unwired() {
        let m = krea_model();
        let levers = LaneLevers {
            supports_sequential: false,
            ..krea_levers()
        };
        // Resident peak (decode 31 + text 8 = 39) is over 35, so escalation runs — but residency can't
        // engage, so the policy tiles the decode (text stays co-resident) instead of a no-op lever.
        let plan = plan_memory_adaptation(35.0, levers, |s| m.peaks_at(s));
        assert!(
            !plan.engaged.contains(&Lever::SequentialResidency),
            "must not claim an unwired residency lever: {plan:?}"
        );
        assert_eq!(plan.residency, OffloadPolicy::Resident);
        assert!(plan.tile_decode && plan.feasible, "{plan:?}");
    }
}

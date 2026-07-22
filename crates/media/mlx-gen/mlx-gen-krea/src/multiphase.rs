//! Multi-phase Krea denoise primitive (epic 13879, sc-13884) — the host-side decomposition that lets
//! one Krea render run an ordered list of **phases** within ONE denoise trajectory over ONE coherent
//! global sigma schedule. Each phase owns a contiguous slice of that shared schedule (its step count),
//! its own guidance (true-CFG on/off), and its own active adapter stack. The canonical workflow is
//! "*N* steps Raw with true-CFG on, then *M* steps Raw+turbo-LoRA with CFG off", with the split freely
//! varied — but the primitive is general over any ordered phase list.
//!
//! # The correctness crux: ONE global schedule, contiguous slices
//!
//! The whole point of a multi-phase render (vs. running two independent renders and stitching) is that
//! the latent AND the sigma trajectory flow **continuously** across every phase boundary. So the sigma
//! schedule is computed ONCE for the *total* step budget (the sum of the phases' steps), and each phase
//! is handed a contiguous, inclusive index slice of it. Phase *i*'s slice ends at the exact schedule
//! index phase *i+1*'s slice begins — the shared boundary sigma — so resuming phase *i+1* from phase
//! *i*'s output latent at that same sigma is a seamless continuation, never a reset. Computing an
//! independent schedule per phase would restart sigma at each boundary and cause a seam/reset artifact;
//! [`resolve_phase_slices`] is the guard that this never happens (its contiguity + shared-boundary
//! contract is pinned by the unit tests).
//!
//! # What this module owns (host math) vs. what the driver owns (GPU)
//!
//! This module is pure host arithmetic + validation — no tensors, no device — so it is fully unit
//! testable without weights: schedule slicing ([`resolve_phase_slices`]), per-phase guidance/CFG-branch
//! selection ([`phase_uses_cfg`] / [`any_phase_uses_cfg`]), per-phase adapter-set resolution
//! ([`resolve_phase_adapters`]), and the whole-request resolution + validation
//! ([`resolve_phases`]). The [`KreaHeavy`](crate::pipeline::KreaHeavy) render driver consumes the
//! resolved plan and drives `run_flow_sampler` over each slice from the running latent, selecting the
//! two-forward (CFG) or single-forward body per phase — the only GPU-bound part.

use mlx_gen::{Error, GenerationPhase, Result};

/// A contiguous slice of the ONE shared global sigma schedule owned by one phase — an **inclusive**
/// index range `[start, end]` into a schedule of length `total_steps + 1`. Running the flow sampler
/// over `schedule[start..=end]` performs exactly `end - start` Euler steps and leaves the latent at
/// `schedule[end]`; the next phase's slice starts at that same index (`start == prev.end`), so the
/// latent and sigma continue with no reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseSlice {
    /// First schedule index of this phase (the sigma the phase resumes from).
    pub start: usize,
    /// Last schedule index of this phase, **inclusive** (the sigma the phase reaches; also the next
    /// phase's `start`).
    pub end: usize,
}

impl PhaseSlice {
    /// The number of Euler denoise steps this phase runs (`end - start`).
    pub fn steps(&self) -> usize {
        self.end - self.start
    }

    /// The inclusive Rust range that indexes the shared schedule for this phase — `schedule[start..=end]`
    /// is the sub-slice the flow sampler runs.
    pub fn range(&self) -> std::ops::RangeInclusive<usize> {
        self.start..=self.end
    }
}

/// Slice the ONE global schedule into the contiguous, shared-boundary phase windows for `phase_steps`.
///
/// The returned slices, in order, partition the schedule of length `sum(phase_steps) + 1` such that:
/// slice 0 starts at index 0; every slice's `end` equals the next slice's `start` (the shared boundary
/// — this is what keeps the sigma trajectory continuous); and the last slice's `end` equals the total
/// step budget (the schedule's final index, i.e. σ = 0). Every schedule step index is covered by
/// exactly one phase, with no gap, no overlap, and no reset.
///
/// Errors when `phase_steps` is empty (a multi-phase render needs ≥ 1 phase) or any phase has 0 steps
/// (an empty phase is a malformed request). `model` names the caller for the error message.
pub fn resolve_phase_slices(phase_steps: &[usize], model: &str) -> Result<Vec<PhaseSlice>> {
    if phase_steps.is_empty() {
        return Err(Error::Msg(format!(
            "{model}: a multi-phase render requires at least one phase"
        )));
    }
    let mut slices = Vec::with_capacity(phase_steps.len());
    let mut cursor = 0usize;
    for (i, &steps) in phase_steps.iter().enumerate() {
        if steps == 0 {
            return Err(Error::Msg(format!(
                "{model}: phase {i} has 0 steps (each phase must run at least one step)"
            )));
        }
        let start = cursor;
        cursor += steps;
        slices.push(PhaseSlice { start, end: cursor });
    }
    Ok(slices)
}

/// The total step budget of a phase list — the sum of every phase's steps, which is the length (minus
/// one) of the ONE global schedule built for the whole trajectory. `steps` on the flat request is
/// ignored in favor of this when phases are present.
pub fn total_phase_steps(phases: &[GenerationPhase]) -> usize {
    phases.iter().map(|p| p.steps as usize).sum()
}

/// Whether a phase runs the **true-CFG** (two forwards per step: conditional + unconditional) path.
/// Mirrors the single-phase Raw selector (`cfg = guidance > 0`): a strictly-positive guidance engages
/// classifier-free guidance; `0.0` (and, upstream, `None`) collapses to a single conditional forward.
pub fn phase_uses_cfg(guidance: f32) -> bool {
    guidance > 0.0
}

/// Whether **any** resolved phase uses the CFG path — the discriminator for whether the shared render
/// plan must carry the unconditional prep (`prep_neg`). The negative context/prep is built once iff at
/// least one phase needs it; a phase with `guidance == 0` simply never consults it.
pub fn any_phase_uses_cfg(phases: &[ResolvedPhase]) -> bool {
    phases.iter().any(|p| phase_uses_cfg(p.guidance))
}

/// A resolved reference to one load-time adapter activated by a phase: the adapter's index in the
/// loaded stack plus its effective per-phase weight (`None` = use the adapter's load-time scale).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedPhaseAdapter {
    /// Index into the load-time adapter stack (bounds-checked against the loaded count).
    pub index: usize,
    /// Per-phase weight override, or `None` to use the adapter's load-time scale.
    pub weight: Option<f32>,
}

/// Resolve one phase's adapter references against the size of the loaded adapter stack, bounds-checking
/// every index. An **empty** result means the phase runs the bare base model (no adapters). Errors on
/// an index ≥ `loaded_adapter_count` — a request naming an adapter the model was never loaded with
/// (surfaced loudly, never silently dropped, matching the load-time adapter seam's strictness).
pub fn resolve_phase_adapters(
    phase: &GenerationPhase,
    loaded_adapter_count: usize,
    model: &str,
) -> Result<Vec<ResolvedPhaseAdapter>> {
    let mut out = Vec::with_capacity(phase.adapters.len());
    for pa in &phase.adapters {
        if pa.adapter >= loaded_adapter_count {
            return Err(Error::Msg(format!(
                "{model}: phase adapter index {} is out of range — the model was loaded with {} \
                 adapter(s) (indices 0..{})",
                pa.adapter,
                loaded_adapter_count,
                loaded_adapter_count.saturating_sub(1)
            )));
        }
        out.push(ResolvedPhaseAdapter {
            index: pa.adapter,
            weight: pa.weight,
        });
    }
    Ok(out)
}

/// The fully-resolved plan for one phase: its schedule slice, its guidance (already defaulted from the
/// request/model), and its resolved, bounds-checked adapter set. The render driver reads exactly this.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPhase {
    /// This phase's contiguous slice of the shared global schedule.
    pub slice: PhaseSlice,
    /// This phase's guidance (`> 0` ⇒ true-CFG two-forward path; `0.0` ⇒ single conditional forward).
    pub guidance: f32,
    /// The load-time adapters active during this phase (empty ⇒ base-only).
    pub adapters: Vec<ResolvedPhaseAdapter>,
}

/// Resolve a request's `phases` into an ordered [`ResolvedPhase`] plan over ONE shared schedule.
///
/// Validates the list (non-empty, every phase ≥ 1 step), slices the shared schedule ([`resolve_phase_slices`]),
/// defaults each phase's guidance (`None` ⇒ `default_guidance`, the request/model guidance), and
/// resolves + bounds-checks each phase's adapter references ([`resolve_phase_adapters`]). The result
/// carries everything the [`KreaHeavy`](crate::pipeline::KreaHeavy) driver needs to run the trajectory,
/// with the total step budget being the sum of the phase slices (`resolved.last().slice.end`).
///
/// `default_guidance` is the guidance a phase with `None` inherits (the Raw default, or `0.0` on a
/// CFG-free variant). `loaded_adapter_count` is how many adapters the model was loaded with (from
/// `LoadSpec::adapters`), against which phase adapter indices are checked.
pub fn resolve_phases(
    phases: &[GenerationPhase],
    default_guidance: f32,
    loaded_adapter_count: usize,
    model: &str,
) -> Result<Vec<ResolvedPhase>> {
    let phase_steps: Vec<usize> = phases.iter().map(|p| p.steps as usize).collect();
    let slices = resolve_phase_slices(&phase_steps, model)?;
    let mut resolved = Vec::with_capacity(phases.len());
    for (phase, slice) in phases.iter().zip(slices) {
        resolved.push(ResolvedPhase {
            slice,
            guidance: phase.guidance.unwrap_or(default_guidance),
            adapters: resolve_phase_adapters(phase, loaded_adapter_count, model)?,
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::PhaseAdapter;

    fn phase(steps: u32, guidance: Option<f32>, adapters: Vec<PhaseAdapter>) -> GenerationPhase {
        GenerationPhase {
            steps,
            guidance,
            adapters,
        }
    }

    /// The crux: given a total split into phases, the slices are contiguous, cover the ONE schedule
    /// exactly once in order, with no gap/overlap — and each phase resumes at the SAME sigma the prior
    /// phase reached. Pinned against concrete schedule values so the test would fail if a phase ever
    /// recomputed its own schedule (which would reset the boundary sigma).
    #[test]
    fn phase_slices_are_contiguous_and_share_boundary_sigmas() {
        // total = 3 + 2 + 4 = 9 → the ONE global schedule has 10 sigmas (len = total + 1).
        let slices = resolve_phase_slices(&[3, 2, 4], "krea_2_raw").unwrap();
        assert_eq!(
            slices,
            vec![
                PhaseSlice { start: 0, end: 3 },
                PhaseSlice { start: 3, end: 5 },
                PhaseSlice { start: 5, end: 9 },
            ]
        );

        // Coverage: first starts at 0, last ends at total (σ = 0), and each end == next start (no gap,
        // no overlap, no reset).
        assert_eq!(slices.first().unwrap().start, 0);
        assert_eq!(slices.last().unwrap().end, 9);
        for pair in slices.windows(2) {
            assert_eq!(
                pair[0].end, pair[1].start,
                "phases must share the boundary index"
            );
        }
        // Step counts sum back to the total budget.
        let total: usize = slices.iter().map(PhaseSlice::steps).sum();
        assert_eq!(total, 9);

        // A concrete descending schedule of length total + 1. The BOUNDARY sigma each phase resumes
        // from is literally the same array element the prior phase ended on — the property that a
        // per-phase recomputed schedule would break.
        let schedule: [f32; 10] = [1.0, 0.9, 0.78, 0.64, 0.5, 0.36, 0.22, 0.12, 0.05, 0.0];
        // Phase 0 runs schedule[0..=3] (σ 1.0 → 0.64); phase 1 must RESUME at σ 0.64, not restart at 1.0.
        assert_eq!(schedule[slices[0].end], schedule[slices[1].start]);
        assert_eq!(schedule[slices[0].end], 0.64);
        // Phase 1 runs schedule[3..=5] (σ 0.64 → 0.36); phase 2 resumes at σ 0.36 and runs to σ 0.
        assert_eq!(schedule[slices[1].end], schedule[slices[2].start]);
        assert_eq!(schedule[slices[1].end], 0.36);
        assert_eq!(schedule[slices[2].end], 0.0);

        // Every schedule STEP index (0..total) belongs to exactly one phase (partition, no double-cover).
        let mut covered = vec![0usize; 9];
        for s in &slices {
            for c in &mut covered[s.start..s.end] {
                *c += 1;
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "each step covered exactly once: {covered:?}"
        );
    }

    /// A single phase covering the whole budget degenerates to the ordinary single-trajectory render
    /// (one slice spanning the entire schedule).
    #[test]
    fn single_phase_spans_the_whole_schedule() {
        let slices = resolve_phase_slices(&[52], "krea_2_raw").unwrap();
        assert_eq!(slices, vec![PhaseSlice { start: 0, end: 52 }]);
        assert_eq!(slices[0].steps(), 52);
    }

    /// An empty phase list and a 0-step phase are both rejected (malformed multi-phase requests).
    #[test]
    fn phase_slices_reject_empty_and_zero_step() {
        let empty = resolve_phase_slices(&[], "krea_2_raw")
            .unwrap_err()
            .to_string();
        assert!(empty.contains("at least one phase"), "got: {empty}");
        let zero = resolve_phase_slices(&[4, 0, 2], "krea_2_raw")
            .unwrap_err()
            .to_string();
        assert!(zero.contains("phase 1 has 0 steps"), "got: {zero}");
    }

    /// Per-phase guidance selects the discriminating branch: `> 0` ⇒ CFG (two forwards), `0.0` ⇒
    /// single conditional forward. `None` inherits the caller's default (here a positive Raw default),
    /// so it engages CFG.
    #[test]
    fn phase_guidance_selects_the_cfg_branch() {
        assert!(phase_uses_cfg(3.5));
        assert!(!phase_uses_cfg(0.0));
        // A phase with None guidance defaults to the Raw guidance (3.5 > 0 → CFG) via resolve_phases.
        let resolved = resolve_phases(
            &[phase(20, None, vec![]), phase(8, Some(0.0), vec![])],
            3.5,
            0,
            "krea_2_raw",
        )
        .unwrap();
        assert_eq!(resolved[0].guidance, 3.5);
        assert!(
            phase_uses_cfg(resolved[0].guidance),
            "phase 0 (None → 3.5) is CFG"
        );
        assert!(
            !phase_uses_cfg(resolved[1].guidance),
            "phase 1 (0.0) is CFG-off"
        );
        // The plan needs prep_neg because phase 0 uses CFG, even though phase 1 does not.
        assert!(any_phase_uses_cfg(&resolved));

        // An all-CFG-off plan needs no negative prep.
        let cfg_off = resolve_phases(
            &[phase(4, Some(0.0), vec![]), phase(4, Some(0.0), vec![])],
            0.0,
            0,
            "krea_2_turbo",
        )
        .unwrap();
        assert!(!any_phase_uses_cfg(&cfg_off));
    }

    /// Per-phase adapter resolution: base-only (empty), a weight override, a load-time-scale default
    /// (`None`), and an out-of-range index rejected loudly.
    #[test]
    fn phase_adapters_resolve_and_bounds_check() {
        // Base-only phase → empty resolved set.
        let base = resolve_phase_adapters(&phase(20, None, vec![]), 2, "krea_2_raw").unwrap();
        assert!(base.is_empty());

        // Weight override + load-time default against a model loaded with 2 adapters.
        let p = phase(
            8,
            Some(0.0),
            vec![
                PhaseAdapter {
                    adapter: 0,
                    weight: Some(0.8),
                },
                PhaseAdapter {
                    adapter: 1,
                    weight: None,
                },
            ],
        );
        let got = resolve_phase_adapters(&p, 2, "krea_2_raw").unwrap();
        assert_eq!(
            got,
            vec![
                ResolvedPhaseAdapter {
                    index: 0,
                    weight: Some(0.8)
                },
                ResolvedPhaseAdapter {
                    index: 1,
                    weight: None
                },
            ]
        );

        // Index 1 with only ONE adapter loaded → out-of-range error.
        let oor = resolve_phase_adapters(
            &phase(
                8,
                Some(0.0),
                vec![PhaseAdapter {
                    adapter: 1,
                    weight: None,
                }],
            ),
            1,
            "krea_2_raw",
        )
        .unwrap_err()
        .to_string();
        assert!(oor.contains("out of range"), "got: {oor}");
        assert!(oor.contains("loaded with 1 adapter"), "got: {oor}");
    }

    /// End-to-end resolution of the canonical workflow: 20 steps Raw CFG-on base-only, then 8 steps
    /// Raw+turbo-LoRA CFG-off, over ONE 28-step schedule.
    #[test]
    fn resolve_phases_end_to_end_raw_then_turbo_lora() {
        let phases = vec![
            phase(20, None, vec![]),
            phase(
                8,
                Some(0.0),
                vec![PhaseAdapter {
                    adapter: 0,
                    weight: Some(1.0),
                }],
            ),
        ];
        assert_eq!(total_phase_steps(&phases), 28);
        let resolved = resolve_phases(&phases, 3.5, 1, "krea_2_raw").unwrap();
        assert_eq!(resolved.len(), 2);
        // Phase 0: steps 0..=20, CFG on, base-only.
        assert_eq!(resolved[0].slice, PhaseSlice { start: 0, end: 20 });
        assert_eq!(resolved[0].guidance, 3.5);
        assert!(resolved[0].adapters.is_empty());
        // Phase 1: steps 20..=28, CFG off, turbo LoRA #0 at weight 1.0 — resuming from phase 0's latent
        // at the SHARED boundary index 20.
        assert_eq!(resolved[1].slice, PhaseSlice { start: 20, end: 28 });
        assert_eq!(resolved[0].slice.end, resolved[1].slice.start);
        assert!(!phase_uses_cfg(resolved[1].guidance));
        assert_eq!(
            resolved[1].adapters,
            vec![ResolvedPhaseAdapter {
                index: 0,
                weight: Some(1.0)
            }]
        );
        assert!(any_phase_uses_cfg(&resolved));
    }
}

//! Typed bespoke-loop driver contract (`PIPE-7`, `PIPE-14`, `DEC-13`, `DEC-19`, `B3`).
//!
//! Essential bespoke denoising loops (native SDXL ancestral/acceleration, native Kolors
//! leading-Euler, Wan native UniPC, FLUX.3 flow-map, Sana Sprint SCM, Boogu/MOSS DMD, Iris
//! FlowDPM, Ideogram, Supra2, etc.) run under the shared runner through [`FamilyDriver`] rather
//! than open-coding ad-hoc cancellation, progress, boundary, and state-ownership machinery.
//!
//! Every [`FamilyDriver`] is generic over its forward seam so a weights-free Tier 0 conformance
//! suite ([`conformance::verify_driver_conformance`]) can verify every capability and typed
//! refusal declared in [`DriverSupport`] without loading weights or running inference compute.

use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};
use std::thread::ThreadId;

use serde::{Deserialize, Serialize};

use crate::{BoundaryState, CancelFlag, Error, Progress, RenderRun, Result};

/// Distinguishes ordinary trajectory evaluations from read-only boundary estimates (`PIPE-3`/`4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverEvaluationKind {
    /// Normal trajectory step evaluation; may update solver history, stochastic draws, and
    /// emit outer-step progress.
    Trajectory,
    /// Read-only boundary clean-estimate evaluation; MUST NOT advance outer progress, mutate
    /// solver/guidance history, commit autoregressive state, or consume solver RNG draws.
    BoundaryEstimate,
}

impl DriverEvaluationKind {
    pub const fn is_read_only_boundary(self) -> bool {
        matches!(self, Self::BoundaryEstimate)
    }
}

/// Solver/driver numerical history rule at a slice entry boundary (`PIPE-3`).
///
/// Independent of stochastic RNG draw coordinates: [`Self::Reset`] clears numerical history
/// without resetting `stochastic_draw_index` or unrelated driver caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverHistoryPolicy {
    /// Preserve numerical history across a compatible `Continue` boundary (e.g. Wan native UniPC).
    Carry,
    /// Clear numerical history before executing this slice.
    Reset,
}

/// Declared transition entering a [`FamilyDriver`] slice (`PIPE-4`, `PIPE-7`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceTransition {
    /// Initial entry at the start of a pass trajectory.
    Start,
    /// Contiguous continuation at the previous slice's endpoint sigma and geometry.
    Continue,
    /// Re-noise from a clean estimate (`BoundaryState::CleanEstimate`); never accepts noisy
    /// `BoundaryState::XSigma` (prevents double-noising).
    Renoise,
    /// Geometry-changing transition from a clean estimate (e.g. low-res start).
    Resample,
}

/// Actionable typed `DEC-13` refusal messages for unsupported [`DriverSupport`] capabilities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverRefusalReasons {
    pub arbitrary_slice: &'static str,
    pub carry_history: &'static str,
    pub reset_history: &'static str,
    pub renoise_from_clean: &'static str,
    pub geometry_change: &'static str,
    pub read_only_boundary_estimate: &'static str,
}

impl Default for DriverRefusalReasons {
    fn default() -> Self {
        Self {
            arbitrary_slice: "driver does not support sub-schedule slicing",
            carry_history: "driver does not support carrying numerical history across slices",
            reset_history: "driver does not support resetting numerical history mid-trajectory",
            renoise_from_clean: "driver does not support re-noising from a clean boundary estimate",
            geometry_change: "driver does not support geometry-changing phase boundaries",
            read_only_boundary_estimate:
                "driver does not support read-only boundary clean-estimate evaluation",
        }
    }
}

/// Primary behavioural capability declaration for a [`FamilyDriver`] (`AP5`, `PIPE-7`, `DEC-13`).
///
/// Every `true` capability and every `false` typed refusal is verified in Tier 0 over a fake
/// forward seam via [`conformance::verify_driver_conformance`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverSupport {
    pub driver_id: &'static str,
    /// Supports arbitrary contiguous half-open slices `[start, end)` of the pass schedule.
    pub arbitrary_slice: bool,
    /// Supports carrying numerical solver/driver history across a contiguous `Continue` boundary.
    pub carry_history: bool,
    /// Supports resetting numerical history at a slice boundary without resetting RNG coordinates.
    pub reset_history: bool,
    /// Supports entering a slice via [`SliceTransition::Renoise`] from [`BoundaryState::CleanEstimate`].
    pub renoise_from_clean: bool,
    /// Supports entering a slice via [`SliceTransition::Resample`] with changed spatial geometry.
    pub geometry_change: bool,
    /// Supports read-only [`DriverEvaluationKind::BoundaryEstimate`] without mutating state/RNG.
    pub read_only_boundary_estimate: bool,
    /// Supports capturing `GuidedPrimaryX0` (`EstimateSource::PreviousEvaluation`) without an extra forward.
    pub previous_estimate_capture: bool,
    /// True when the driver draws from a process-global RNG (such as MLX global `random::seed` /
    /// `random::normal`) and therefore requires serialized whole-trajectory ownership (`B3`/`PIPE-14`).
    pub uses_global_rng: bool,
    /// Typed `DEC-13` refusal messages emitted when an unsupported capability is requested.
    pub refusal_reasons: DriverRefusalReasons,
}

impl DriverSupport {
    /// Validate a [`SliceInput`] against this driver's declared capabilities and invariants.
    ///
    /// Enforces `PIPE-3`/`PIPE-4`/`PIPE-7`/`DEC-13` uniformly so no driver silently ignores an
    /// unsupported transition, incompatible `Carry`, or out-of-bounds slice.
    pub fn validate_slice<L>(
        &self,
        input: &SliceInput<'_, L>,
        state: &DriverRunState,
    ) -> Result<()> {
        state.assert_owner_thread()?;
        if input.span.start > input.span.end || input.span.end > input.total_steps {
            return Err(Error::Unsupported(format!(
                "{}: invalid slice {:?} for total_steps={}",
                self.driver_id, input.span, input.total_steps
            )));
        }

        let is_full_span = input.span.start == 0 && input.span.end == input.total_steps;
        if !is_full_span && !self.arbitrary_slice {
            return Err(Error::Unsupported(format!(
                "{}: {}",
                self.driver_id, self.refusal_reasons.arbitrary_slice
            )));
        }

        // Solver history NEVER crosses a geometry change or re-noise (PIPE-3 / PIPE-4).
        if matches!(
            input.transition,
            SliceTransition::Renoise | SliceTransition::Resample
        ) && input.history == DriverHistoryPolicy::Carry
        {
            return Err(Error::Unsupported(format!(
                "{}: solver history cannot be carried across a {:?} boundary",
                self.driver_id, input.transition
            )));
        }

        match input.transition {
            SliceTransition::Start => {
                if input.span.start != 0 && !self.arbitrary_slice {
                    return Err(Error::Unsupported(format!(
                        "{}: {}",
                        self.driver_id, self.refusal_reasons.arbitrary_slice
                    )));
                }
            }
            SliceTransition::Continue => {
                if let Some(prev_end) = state.last_span_end {
                    if prev_end != input.span.start {
                        return Err(Error::Unsupported(format!(
                            "{}: Continue slice starts at {} but previous slice ended at {prev_end}",
                            self.driver_id, input.span.start
                        )));
                    }
                }
                if let (Some(prev_geom), Some(next_geom)) = (state.last_geometry, input.geometry) {
                    if prev_geom != next_geom {
                        return Err(Error::Unsupported(format!(
                            "{}: Continue transition cannot change geometry from {prev_geom:?} to {next_geom:?}",
                            self.driver_id
                        )));
                    }
                }
                match input.history {
                    DriverHistoryPolicy::Carry if !self.carry_history => {
                        return Err(Error::Unsupported(format!(
                            "{}: {}",
                            self.driver_id, self.refusal_reasons.carry_history
                        )));
                    }
                    DriverHistoryPolicy::Reset if !self.reset_history => {
                        return Err(Error::Unsupported(format!(
                            "{}: {}",
                            self.driver_id, self.refusal_reasons.reset_history
                        )));
                    }
                    _ => {}
                }
            }
            SliceTransition::Renoise => {
                if !self.renoise_from_clean {
                    return Err(Error::Unsupported(format!(
                        "{}: {}",
                        self.driver_id, self.refusal_reasons.renoise_from_clean
                    )));
                }
                if !matches!(input.state, BoundaryState::CleanEstimate { .. }) {
                    return Err(Error::Unsupported(format!(
                        "{}: Renoise transition requires BoundaryState::CleanEstimate (refused noisy/route state to prevent double-noising)",
                        self.driver_id
                    )));
                }
            }
            SliceTransition::Resample => {
                if !self.geometry_change {
                    return Err(Error::Unsupported(format!(
                        "{}: {}",
                        self.driver_id, self.refusal_reasons.geometry_change
                    )));
                }
                if !matches!(input.state, BoundaryState::CleanEstimate { .. }) {
                    return Err(Error::Unsupported(format!(
                        "{}: Resample transition requires BoundaryState::CleanEstimate",
                        self.driver_id
                    )));
                }
            }
        }

        if input.request_boundary_estimate && !self.read_only_boundary_estimate {
            return Err(Error::Unsupported(format!(
                "{}: {}",
                self.driver_id, self.refusal_reasons.read_only_boundary_estimate
            )));
        }

        Ok(())
    }
}

/// Per-evaluation coordinate metadata handed to [`StepHooks`] (`PIPE-6`, `DEC-19`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriverStepInfo {
    pub pass: usize,
    pub phase: usize,
    /// 0-based absolute outer step index in the enclosing pass schedule.
    pub outer_step: usize,
    /// Total executed outer steps in the enclosing pass (drives monotone `Progress::Step`).
    pub total_outer_steps: usize,
    /// 0-based evaluation ordinal within `outer_step`.
    pub eval_in_step: u32,
    pub kind: DriverEvaluationKind,
    /// Schedule coordinate (sigma or timestep) entering this evaluation.
    pub sigma: f32,
    /// Next schedule coordinate leaving this step.
    pub next_sigma: f32,
}

impl DriverStepInfo {
    /// Progress tick for a completed outer step (monotone `1..=total_outer_steps`, `DEC-19`).
    pub fn completed_progress(self) -> Progress {
        Progress::Step {
            current: (self.outer_step + 1).min(self.total_outer_steps.max(1)) as u32,
            total: self.total_outer_steps.max(1) as u32,
        }
    }
}

/// Typed entry contract for one [`FamilyDriver::run_slice`] invocation (`PIPE-7`).
pub struct SliceInput<'a, L> {
    pub pass: usize,
    pub phase: usize,
    /// Half-open absolute outer-step span `[start, end)` within the enclosing pass schedule.
    pub span: Range<usize>,
    /// Total executed outer steps in the enclosing pass.
    pub total_steps: usize,
    /// Typed entry boundary latent state.
    pub state: BoundaryState<L>,
    /// Declared boundary transition entering this slice.
    pub transition: SliceTransition,
    /// Declared solver/driver history policy entering this slice.
    pub history: DriverHistoryPolicy,
    /// Optional spatial `(width, height)` for this slice (checked across geometry changes).
    pub geometry: Option<(u32, u32)>,
    /// Sample seed for this trajectory (`PIPE-11`).
    pub seed: u64,
    /// Cooperative cancellation flag checked before every evaluation and boundary.
    pub cancel: &'a CancelFlag,
    /// When `true`, emit a read-only [`BoundaryState::CleanEstimate`] after completing `span`.
    pub request_boundary_estimate: bool,
}

impl<'a, L> SliceInput<'a, L> {
    /// Build a single-slice full-trajectory input `[0, total_steps)` from a noisy initial state.
    pub fn full_trajectory(
        latent: L,
        total_steps: usize,
        seed: u64,
        cancel: &'a CancelFlag,
    ) -> Self {
        Self {
            pass: 0,
            phase: 0,
            span: 0..total_steps,
            total_steps,
            state: BoundaryState::XSigma(latent),
            transition: SliceTransition::Start,
            history: DriverHistoryPolicy::Reset,
            geometry: None,
            seed,
            cancel,
            request_boundary_estimate: false,
        }
    }

    /// Attach spatial geometry to this slice input.
    pub fn with_geometry(mut self, width: u32, height: u32) -> Self {
        self.geometry = Some((width, height));
        self
    }

    /// Attach pass and phase ordinals to this slice input.
    pub fn with_pass_phase(mut self, pass: usize, phase: usize) -> Self {
        self.pass = pass;
        self.phase = phase;
        self
    }
}

/// Descriptor for one slice driven by [`run_driver_slices`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceSpec {
    pub pass: usize,
    pub phase: usize,
    pub span: Range<usize>,
    pub transition: SliceTransition,
    pub history: DriverHistoryPolicy,
    pub geometry: Option<(u32, u32)>,
    pub request_boundary_estimate: bool,
}

impl SliceSpec {
    /// Convenience spec for a single full pass `[0, total_steps)`.
    pub fn full(total_steps: usize) -> Self {
        Self {
            pass: 0,
            phase: 0,
            span: 0..total_steps,
            transition: SliceTransition::Start,
            history: DriverHistoryPolicy::Reset,
            geometry: None,
            request_boundary_estimate: false,
        }
    }

    /// Build planned trajectory slices from an active [`RenderRun`]'s compiled `denoise_spans()`,
    /// falling back to a single full-trajectory slice when `run` is `None` or the compiled spans
    /// do not partition `0..total_steps`.
    ///
    /// Preserves every compiled phase index so [`run_driver_slices`] enters and finishes every
    /// phase (including inert splits created via `RenderPlan::with_inert_split_at`).
    pub fn from_render_run(run: Option<&RenderRun<'_, '_>>, total_steps: usize) -> Vec<Self> {
        match run {
            Some(run) => {
                let plan_spans = run.plan().denoise_spans();
                let covered: usize = plan_spans.iter().map(|s| s.len()).sum();
                if covered == total_steps && !plan_spans.is_empty() {
                    plan_spans
                        .into_iter()
                        .enumerate()
                        .map(|(phase, span)| Self {
                            pass: run.plan().pass_index(),
                            phase,
                            span,
                            transition: if phase == 0 {
                                SliceTransition::Start
                            } else {
                                SliceTransition::Continue
                            },
                            history: if phase == 0 {
                                DriverHistoryPolicy::Reset
                            } else {
                                DriverHistoryPolicy::Carry
                            },
                            geometry: Some(run.plan().phase_output_size(phase)),
                            request_boundary_estimate: false,
                        })
                        .collect()
                } else {
                    vec![Self::full(total_steps)]
                }
            }
            None => vec![Self::full(total_steps)],
        }
    }
}

/// Request-owned, owner-local execution state that outlives individual phase slices (`PIPE-3`, `B3`).
///
/// `!Send` and `!Sync` by construction: loaded providers, hooks, scratch, and mutable trajectory
/// state stay bound to their owning executor thread (`PIPE-14`).
pub struct DriverRunState {
    pub completed_outer_steps: usize,
    pub internal_evaluations: u64,
    pub boundary_evaluations: u64,
    /// Absolute stochastic draw index across the pass; preserved independently of numerical
    /// history resets (`PIPE-3`).
    pub stochastic_draw_index: u64,
    pub history_resets: u32,
    pub last_span_end: Option<usize>,
    pub last_geometry: Option<(u32, u32)>,
    owner_thread: ThreadId,
    history: Option<Box<dyn Any>>,
    cache: Option<Box<dyn Any>>,
    _owner_local: PhantomData<Rc<()>>,
}

impl Default for DriverRunState {
    fn default() -> Self {
        Self::new()
    }
}

impl DriverRunState {
    /// Create a fresh per-request execution state bound to the calling executor thread.
    pub fn new() -> Self {
        Self {
            completed_outer_steps: 0,
            internal_evaluations: 0,
            boundary_evaluations: 0,
            stochastic_draw_index: 0,
            history_resets: 0,
            last_span_end: None,
            last_geometry: None,
            owner_thread: std::thread::current().id(),
            history: None,
            cache: None,
            _owner_local: PhantomData,
        }
    }

    /// The executor thread that owns this request state.
    pub fn owner_thread(&self) -> ThreadId {
        self.owner_thread
    }

    /// Verify that execution remains on the owning thread (`B3` / `PIPE-14`).
    pub fn assert_owner_thread(&self) -> Result<()> {
        let current = std::thread::current().id();
        if current != self.owner_thread {
            return Err(Error::Unsupported(format!(
                "driver run state accessed from thread {current:?} instead of owner {:?}",
                self.owner_thread
            )));
        }
        Ok(())
    }

    /// Clear numerical solver/driver history without modifying `stochastic_draw_index`,
    /// `completed_outer_steps`, or unrelated caches (`PIPE-3`).
    pub fn reset_history(&mut self) {
        self.history = None;
        self.history_resets = self.history_resets.saturating_add(1);
    }

    /// Invalidate geometry- or configuration-dependent driver cache entries.
    pub fn invalidate_cache(&mut self) {
        self.cache = None;
    }

    /// Store type-erased, owner-local numerical solver/driver history.
    pub fn set_history<T: 'static>(&mut self, value: T) {
        self.history = Some(Box::new(value));
    }

    /// Borrow type-erased numerical history if present and matching `T`.
    pub fn history_ref<T: 'static>(&self) -> Option<&T> {
        self.history
            .as_deref()
            .and_then(<dyn Any>::downcast_ref::<T>)
    }

    /// Mutably borrow type-erased numerical history if present and matching `T`.
    pub fn history_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.history
            .as_deref_mut()
            .and_then(<dyn Any>::downcast_mut::<T>)
    }

    /// Take type-erased numerical history if present and matching `T`.
    pub fn take_history<T: 'static>(&mut self) -> Option<T> {
        let boxed = self.history.take()?;
        match boxed.downcast::<T>() {
            Ok(value) => Some(*value),
            Err(original) => {
                self.history = Some(original);
                None
            }
        }
    }

    /// Store an owner-local driver cache entry.
    pub fn set_cache<T: 'static>(&mut self, value: T) {
        self.cache = Some(Box::new(value));
    }

    /// Borrow the owner-local driver cache entry if present and matching `T`.
    pub fn cache_ref<T: 'static>(&self) -> Option<&T> {
        self.cache.as_deref().and_then(<dyn Any>::downcast_ref::<T>)
    }

    /// Mutably borrow the owner-local driver cache entry if present and matching `T`.
    pub fn cache_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.cache
            .as_deref_mut()
            .and_then(<dyn Any>::downcast_mut::<T>)
    }

    /// Advance the absolute stochastic draw coordinate and return the draw index used.
    pub fn next_draw_index(&mut self) -> u64 {
        let index = self.stochastic_draw_index;
        self.stochastic_draw_index = self.stochastic_draw_index.saturating_add(1);
        index
    }

    /// Record one model evaluation (internal accounting; `BoundaryEstimate` also increments
    /// `boundary_evaluations` without touching `completed_outer_steps`).
    pub fn record_evaluation(&mut self, kind: DriverEvaluationKind) {
        self.internal_evaluations = self.internal_evaluations.saturating_add(1);
        if kind.is_read_only_boundary() {
            self.boundary_evaluations = self.boundary_evaluations.saturating_add(1);
        }
    }

    /// Record completion of one outer step (`DEC-19`).
    pub fn record_outer_step(&mut self, outer_step: usize) {
        self.completed_outer_steps = self.completed_outer_steps.saturating_add(1);
        self.last_span_end = Some(outer_step + 1);
    }

    /// Apply declared slice-entry state transitions (history reset / cache invalidation on
    /// geometry change) after [`DriverSupport::validate_slice`] succeeds.
    pub fn enter_slice<L>(&mut self, input: &SliceInput<'_, L>) -> Result<()> {
        self.assert_owner_thread()?;
        let geometry_changed = matches!(
            (self.last_geometry, input.geometry),
            (Some(prev), Some(next)) if prev != next
        ) || input.transition == SliceTransition::Resample;

        if geometry_changed {
            self.invalidate_cache();
        }
        if (input.history == DriverHistoryPolicy::Reset
            || matches!(
                input.transition,
                SliceTransition::Renoise | SliceTransition::Resample
            ))
            && (self.history.is_some()
                || matches!(
                    input.transition,
                    SliceTransition::Continue
                        | SliceTransition::Renoise
                        | SliceTransition::Resample
                ))
        {
            self.reset_history();
        }
        if let Some(geometry) = input.geometry {
            self.last_geometry = Some(geometry);
        }
        Ok(())
    }

    /// Record slice exit endpoint (`span.end`).
    pub fn finish_slice(&mut self, span_end: usize) {
        self.last_span_end = Some(span_end);
    }
}

/// Callback type for per-step latent observation (`StepHooks::on_observe`).
pub type ObserveHook<'a, L> = &'a mut dyn FnMut(&DriverStepInfo, &L) -> Result<()>;
/// Callback type for per-step post-step latent transforms (`StepHooks::on_post_step`).
pub type PostStepHook<'a, L> = &'a mut dyn FnMut(&DriverStepInfo, L) -> Result<L>;
/// Callback type for per-step materialization/quantum barriers (`StepHooks::on_materialize`).
pub type MaterializeHook<'a, L> = &'a mut dyn FnMut(&DriverStepInfo, &L) -> Result<()>;

/// Request-scoped step hooks supplied to a [`FamilyDriver`] (`PIPE-6`, `HOOK-1`, `DEC-19`).
///
/// Inactive hooks are `None` and cost nothing when unused (`HOOK-5`).
pub struct StepHooks<'a, L> {
    /// Outer-step progress callback (never invoked on read-only boundary estimates).
    pub on_progress: &'a mut dyn FnMut(Progress),
    /// Optional per-step latent observation hook (e.g. preview emission, diagnostics).
    pub on_observe: Option<ObserveHook<'a, L>>,
    /// Optional post-step latent transform (e.g. inpaint mask blend, I2V pin, H3 row freeze).
    pub on_post_step: Option<PostStepHook<'a, L>>,
    /// Optional per-step materialization / quantum barrier (e.g. MLX `eval` + iOS quantum).
    pub on_materialize: Option<MaterializeHook<'a, L>>,
}

impl<'a, L> StepHooks<'a, L> {
    /// Construct hooks with progress only and all optional step hooks inactive.
    pub fn new(on_progress: &'a mut dyn FnMut(Progress)) -> Self {
        Self {
            on_progress,
            on_observe: None,
            on_post_step: None,
            on_materialize: None,
        }
    }

    /// Emit outer-step progress for a completed step (`DEC-19`).
    pub fn emit_progress(&mut self, info: &DriverStepInfo) {
        if !info.kind.is_read_only_boundary() {
            (self.on_progress)(info.completed_progress());
        }
    }

    /// Invoke `on_observe` when present and not a read-only boundary estimate.
    pub fn observe(&mut self, info: &DriverStepInfo, latent: &L) -> Result<()> {
        if !info.kind.is_read_only_boundary() {
            if let Some(hook) = self.on_observe.as_deref_mut() {
                hook(info, latent)?;
            }
        }
        Ok(())
    }

    /// Invoke `on_post_step` when present and not a read-only boundary estimate.
    pub fn post_step(&mut self, info: &DriverStepInfo, latent: L) -> Result<L> {
        if !info.kind.is_read_only_boundary() {
            if let Some(hook) = self.on_post_step.as_deref_mut() {
                return hook(info, latent);
            }
        }
        Ok(latent)
    }

    /// Invoke `on_materialize` when present.
    pub fn materialize(&mut self, info: &DriverStepInfo, latent: &L) -> Result<()> {
        if let Some(hook) = self.on_materialize.as_deref_mut() {
            hook(info, latent)?;
        }
        Ok(())
    }
}

/// Typed escape hatch for essential bespoke denoising loops (`PIPE-7`).
///
/// Concrete implementations are generic over a forward-evaluation seam so Tier 0 conformance
/// tests can verify every claim in [`DriverSupport`] without loading model weights.
pub trait FamilyDriver<L> {
    /// Primary capability declaration verified by [`conformance::verify_driver_conformance`].
    fn support(&self) -> &DriverSupport;

    /// Execute one schedule slice under request-owned `state` and `hooks`.
    fn run_slice(
        &self,
        input: SliceInput<'_, L>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, L>,
    ) -> Result<BoundaryState<L>>;
}

/// Drive a [`FamilyDriver`] across one or more planned slices, binding [`RenderRun`] phase
/// scopes, cancellation, owner-thread enforcement, and boundary transitions (`PIPE-3`/`4`/`7`).
#[allow(clippy::too_many_arguments)]
pub fn run_driver_slices<L, D: FamilyDriver<L> + ?Sized>(
    driver: &D,
    slices: impl IntoIterator<Item = SliceSpec>,
    initial_state: BoundaryState<L>,
    total_steps: usize,
    seed: u64,
    cancel: &CancelFlag,
    run: Option<&RenderRun<'_, '_>>,
    state: &mut DriverRunState,
    hooks: &mut StepHooks<'_, L>,
    mut on_boundary: impl FnMut(
        usize,
        BoundaryState<L>,
        &mut DriverRunState,
    ) -> Result<BoundaryState<L>>,
) -> Result<BoundaryState<L>> {
    state.assert_owner_thread()?;
    let specs: Vec<SliceSpec> = slices.into_iter().collect();
    if specs.is_empty() {
        return Ok(initial_state);
    }

    let mut current = initial_state;
    let last_idx = specs.len() - 1;
    for (idx, spec) in specs.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(run) = run {
            run.begin_denoise_phase(spec.phase)?;
        }
        let input = SliceInput {
            pass: spec.pass,
            phase: spec.phase,
            span: spec.span.clone(),
            total_steps,
            state: current,
            transition: spec.transition,
            history: spec.history,
            geometry: spec.geometry,
            seed,
            cancel,
            request_boundary_estimate: spec.request_boundary_estimate,
        };
        driver.support().validate_slice(&input, state)?;
        let out = driver.run_slice(input, state, hooks)?;
        if let Some(run) = run {
            run.end_denoise_phase(spec.phase)?;
        }
        current = if idx < last_idx {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            on_boundary(idx, out, state)?
        } else {
            out
        };
    }
    Ok(current)
}

// =================================================================================================
// B3 / PIPE-14: Global-RNG Whole-Trajectory Serialization Gate
// =================================================================================================

thread_local! {
    /// Addresses of [`GlobalRngTrajectoryGate`] instances currently locked by this thread.
    /// Allows nested helpers on the owning thread (e.g., InstantID calling SDXL `seeded_prior`
    /// followed by `denoise_ip_multi_control`) to re-enter the same gate without
    /// self-deadlocking, while still acquiring the lock for any distinct gate instance and
    /// blocking every other thread until the outer trajectory finishes.
    static HELD_GATES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Serializes whole trajectories (`seed → draws → materialization`) for providers that draw from
/// a process-global RNG such as MLX's `mlx_rs::random::seed` / `random::normal` (`B3`, `PIPE-14`).
///
/// Reentrant on the currently owning thread for the same gate instance; cross-thread callers wait
/// until the active trajectory finishes and materializes all draws, guaranteeing zero cross-request
/// RNG interleaving.
pub struct GlobalRngTrajectoryGate {
    lock: Mutex<()>,
}

impl GlobalRngTrajectoryGate {
    pub const fn new() -> Self {
        Self {
            lock: Mutex::new(()),
        }
    }

    /// Acquire a whole-trajectory lease on the calling thread.
    ///
    /// Nested calls on the same owning thread for the same gate instance succeed without
    /// re-locking; acquiring a distinct gate instance still locks that gate's mutex, and any
    /// other thread blocks until the outermost lease drops.
    pub fn acquire(&self) -> GlobalRngTrajectoryLease<'_> {
        let gate_addr = self as *const Self as usize;
        let already_held = HELD_GATES.with(|gates| gates.borrow().contains(&gate_addr));
        if already_held {
            return GlobalRngTrajectoryLease {
                gate_addr,
                _guard: None,
            };
        }
        // Recover from a poisoned mutex so a panic in one failed trajectory does not permanently
        // wedge unrelated subsequent requests (`PIPE-15`).
        let guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        HELD_GATES.with(|gates| gates.borrow_mut().push(gate_addr));
        GlobalRngTrajectoryLease {
            gate_addr,
            _guard: Some(guard),
        }
    }

    /// Execute `body` while holding a whole-trajectory lease.
    pub fn run_trajectory<T>(&self, body: impl FnOnce() -> T) -> T {
        let _lease = self.acquire();
        body()
    }
}

impl Default for GlobalRngTrajectoryGate {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII lease returned by [`GlobalRngTrajectoryGate::acquire`]. `!Send` so the lock is always
/// released on the thread that acquired it.
pub struct GlobalRngTrajectoryLease<'a> {
    gate_addr: usize,
    _guard: Option<MutexGuard<'a, ()>>,
}

impl Drop for GlobalRngTrajectoryLease<'_> {
    fn drop(&mut self) {
        if self._guard.is_some() {
            HELD_GATES.with(|gates| {
                let mut held = gates.borrow_mut();
                if let Some(pos) = held.iter().rposition(|&addr| addr == self.gate_addr) {
                    held.swap_remove(pos);
                } else {
                    debug_assert!(false, "global RNG trajectory gate address missing on drop");
                }
            });
        }
    }
}

// =================================================================================================
// Reusable Tier 0 DriverSupport Conformance Harness (PIPE-7, AP5, DEC-13, DEC-19, B3)
// =================================================================================================

/// Reusable weights-free Tier 0 conformance harness for every [`FamilyDriver`] (`SW-6966`,
/// `SW-6968`, `SW-6969`, `SW-6970`).
pub mod conformance {
    use super::*;

    /// Configuration for [`verify_driver_conformance`].
    pub struct ConformanceCase<L, MakeDriver, MakeInitial, LatentsEqual>
    where
        MakeDriver: FnMut() -> Box<dyn FamilyDriver<L>>,
        MakeInitial: FnMut() -> L,
        LatentsEqual: Fn(&L, &L) -> bool,
    {
        /// Total outer steps `N >= 2` exercised by the conformance suite.
        pub total_steps: usize,
        /// Split point `1 <= split_at < total_steps` used for slice/carry/reset/boundary checks.
        pub split_at: usize,
        /// Base sample seed for deterministic trajectory comparison.
        pub seed: u64,
        /// Primary spatial geometry `(width, height)` for single-geometry tests.
        pub base_geometry: (u32, u32),
        /// Alternate spatial geometry used when checking `SliceTransition::Resample`.
        pub alt_geometry: (u32, u32),
        /// Factory constructing a fresh weights-free [`FamilyDriver`] over a mocked forward seam.
        pub make_driver: MakeDriver,
        /// Factory constructing a deterministic initial latent `L`.
        pub make_initial: MakeInitial,
        /// Exact equality predicate over `L` (e.g., bit-exact `f32::to_bits` comparison).
        pub latents_equal: LatentsEqual,
    }

    /// Summary of verified Tier 0 conformance checks returned by [`verify_driver_conformance`].
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ConformanceReport {
        pub driver_id: &'static str,
        pub total_steps: usize,
        pub unsplit_evaluations: u64,
        pub unsplit_draws: u64,
        pub verified_inert_split: bool,
        pub verified_carry_refusal_or_support: bool,
        pub verified_reset_preserves_draws: bool,
        pub verified_boundary_estimate_readonly_or_refusal: bool,
        pub verified_renoise_and_resample_guards: bool,
        pub verified_cancellation: bool,
    }

    fn extract_latent<L>(state: BoundaryState<L>) -> L {
        match state {
            BoundaryState::XSigma(l)
            | BoundaryState::CleanEstimate { latent: l, .. }
            | BoundaryState::RouteSpecific { state: l, .. } => l,
        }
    }

    /// Verify every capability and typed refusal declared in `driver.support()` over a
    /// weights-free mocked forward seam (`PIPE-7`, `DEC-13`, `DEC-19`, `B3`).
    pub fn verify_driver_conformance<L, MakeDriver, MakeInitial, LatentsEqual>(
        mut case: ConformanceCase<L, MakeDriver, MakeInitial, LatentsEqual>,
    ) -> Result<ConformanceReport>
    where
        L: Clone,
        MakeDriver: FnMut() -> Box<dyn FamilyDriver<L>>,
        MakeInitial: FnMut() -> L,
        LatentsEqual: Fn(&L, &L) -> bool,
    {
        if case.total_steps < 2 || case.split_at == 0 || case.split_at >= case.total_steps {
            return Err(Error::Unsupported(
                "conformance harness requires total_steps >= 2 and 1 <= split_at < total_steps"
                    .into(),
            ));
        }

        let cancel = CancelFlag::default();

        // 1. Full unsplit trajectory: monotone outer-step progress 1..=N, internal evals >= N.
        let driver = (case.make_driver)();
        let support = driver.support().clone();
        let mut unsplit_state = DriverRunState::new();
        let mut unsplit_progress = Vec::new();
        let mut observe_steps = Vec::new();
        let unsplit_out = {
            let mut on_progress = |p: Progress| unsplit_progress.push(p);
            let mut on_observe = |info: &DriverStepInfo, _l: &L| {
                observe_steps.push(info.outer_step);
                Ok(())
            };
            let mut hooks = StepHooks::new(&mut on_progress);
            hooks.on_observe = Some(&mut on_observe);
            let input = SliceInput::full_trajectory(
                (case.make_initial)(),
                case.total_steps,
                case.seed,
                &cancel,
            )
            .with_geometry(case.base_geometry.0, case.base_geometry.1);
            driver.run_slice(input, &mut unsplit_state, &mut hooks)?
        };
        let expected_progress: Vec<Progress> = (1..=case.total_steps)
            .map(|i| Progress::Step {
                current: i as u32,
                total: case.total_steps as u32,
            })
            .collect();
        if unsplit_progress != expected_progress {
            return Err(Error::Msg(format!(
                "{}: expected monotone progress {:?}, got {:?}",
                support.driver_id, expected_progress, unsplit_progress
            )));
        }
        if unsplit_state.completed_outer_steps != case.total_steps {
            return Err(Error::Msg(format!(
                "{}: completed_outer_steps={} != total_steps={}",
                support.driver_id, unsplit_state.completed_outer_steps, case.total_steps
            )));
        }
        if unsplit_state.internal_evaluations < case.total_steps as u64 {
            return Err(Error::Msg(format!(
                "{}: internal_evaluations={} < total_steps={}",
                support.driver_id, unsplit_state.internal_evaluations, case.total_steps
            )));
        }
        let unsplit_latent = extract_latent(unsplit_out);

        // 2. Slicing & Carry/Reset conformance.
        let mut verified_inert_split = false;
        let mut verified_reset_preserves_draws = false;
        let driver = (case.make_driver)();
        if support.arbitrary_slice {
            if support.carry_history {
                let mut split_state = DriverRunState::new();
                let mut split_progress = Vec::new();
                let split_out = {
                    let mut on_progress = |p: Progress| split_progress.push(p);
                    let mut hooks = StepHooks::new(&mut on_progress);
                    run_driver_slices(
                        driver.as_ref(),
                        [
                            SliceSpec {
                                pass: 0,
                                phase: 0,
                                span: 0..case.split_at,
                                transition: SliceTransition::Start,
                                history: DriverHistoryPolicy::Reset,
                                geometry: Some(case.base_geometry),
                                request_boundary_estimate: false,
                            },
                            SliceSpec {
                                pass: 0,
                                phase: 1,
                                span: case.split_at..case.total_steps,
                                transition: SliceTransition::Continue,
                                history: DriverHistoryPolicy::Carry,
                                geometry: Some(case.base_geometry),
                                request_boundary_estimate: false,
                            },
                        ],
                        BoundaryState::XSigma((case.make_initial)()),
                        case.total_steps,
                        case.seed,
                        &cancel,
                        None,
                        &mut split_state,
                        &mut hooks,
                        |_, out, _| Ok(out),
                    )?
                };
                if split_progress != expected_progress {
                    return Err(Error::Msg(format!(
                        "{}: split progress {:?} != unsplit {:?}",
                        support.driver_id, split_progress, expected_progress
                    )));
                }
                if split_state.stochastic_draw_index != unsplit_state.stochastic_draw_index {
                    return Err(Error::Msg(format!(
                        "{}: split draws {} != unsplit draws {}",
                        support.driver_id,
                        split_state.stochastic_draw_index,
                        unsplit_state.stochastic_draw_index
                    )));
                }
                let split_latent = extract_latent(split_out);
                if !(case.latents_equal)(&split_latent, &unsplit_latent) {
                    return Err(Error::Msg(format!(
                        "{}: inert Carry split diverged from unsplit trajectory",
                        support.driver_id
                    )));
                }
                verified_inert_split = true;
            } else {
                // Must refuse Continue + Carry with typed Error::Unsupported.
                let mut st = DriverRunState::new();
                let mut on_progress = |_: Progress| {};
                let mut hooks = StepHooks::new(&mut on_progress);
                let mid = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: 0..case.split_at,
                        total_steps: case.total_steps,
                        state: BoundaryState::XSigma((case.make_initial)()),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                )?;
                let err = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: case.split_at..case.total_steps,
                        total_steps: case.total_steps,
                        state: mid,
                        transition: SliceTransition::Continue,
                        history: DriverHistoryPolicy::Carry,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                );
                if !matches!(err, Err(Error::Unsupported(_))) {
                    return Err(Error::Msg(format!(
                        "{}: expected typed Unsupported refusal for Continue+Carry when carry_history=false",
                        support.driver_id
                    )));
                }
            }

            if support.reset_history {
                let driver = (case.make_driver)();
                let mut st = DriverRunState::new();
                let mut progress = Vec::new();
                let mut on_progress = |p: Progress| progress.push(p);
                let mut hooks = StepHooks::new(&mut on_progress);
                let mid = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: 0..case.split_at,
                        total_steps: case.total_steps,
                        state: BoundaryState::XSigma((case.make_initial)()),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                )?;
                let draws_after_first = st.stochastic_draw_index;
                let resets_before = st.history_resets;
                let _ = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: case.split_at..case.total_steps,
                        total_steps: case.total_steps,
                        state: mid,
                        transition: SliceTransition::Continue,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                )?;
                if st.history_resets <= resets_before {
                    return Err(Error::Msg(format!(
                        "{}: Continue+Reset did not record a numerical history reset",
                        support.driver_id
                    )));
                }
                if st.stochastic_draw_index < draws_after_first {
                    return Err(Error::Msg(format!(
                        "{}: Continue+Reset rewound stochastic_draw_index ({} < {})",
                        support.driver_id, st.stochastic_draw_index, draws_after_first
                    )));
                }
                if st.stochastic_draw_index != unsplit_state.stochastic_draw_index {
                    return Err(Error::Msg(format!(
                        "{}: Continue+Reset changed total stochastic_draw_index ({} != unsplit {})",
                        support.driver_id,
                        st.stochastic_draw_index,
                        unsplit_state.stochastic_draw_index
                    )));
                }
                if progress != expected_progress {
                    return Err(Error::Msg(format!(
                        "{}: Continue+Reset progress {:?} != expected {:?}",
                        support.driver_id, progress, expected_progress
                    )));
                }
                verified_reset_preserves_draws = true;
            } else {
                // Must refuse Continue + Reset with typed Error::Unsupported.
                let driver = (case.make_driver)();
                let mut st = DriverRunState::new();
                let mut on_progress = |_: Progress| {};
                let mut hooks = StepHooks::new(&mut on_progress);
                let mid = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: 0..case.split_at,
                        total_steps: case.total_steps,
                        state: BoundaryState::XSigma((case.make_initial)()),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                )?;
                let err = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: case.split_at..case.total_steps,
                        total_steps: case.total_steps,
                        state: mid,
                        transition: SliceTransition::Continue,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                );
                if !matches!(err, Err(Error::Unsupported(_))) {
                    return Err(Error::Msg(format!(
                        "{}: expected typed Unsupported refusal for Continue+Reset when reset_history=false",
                        support.driver_id
                    )));
                }
            }
        } else {
            // Must refuse a partial slice with typed Error::Unsupported.
            let mut st = DriverRunState::new();
            let mut on_progress = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut on_progress);
            let err = driver.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 0,
                    span: 0..case.split_at,
                    total_steps: case.total_steps,
                    state: BoundaryState::XSigma((case.make_initial)()),
                    transition: SliceTransition::Start,
                    history: DriverHistoryPolicy::Reset,
                    geometry: Some(case.base_geometry),
                    seed: case.seed,
                    cancel: &cancel,
                    request_boundary_estimate: false,
                },
                &mut st,
                &mut hooks,
            );
            if !matches!(err, Err(Error::Unsupported(_))) {
                return Err(Error::Msg(format!(
                    "{}: expected typed Unsupported refusal for partial slice when arbitrary_slice=false",
                    support.driver_id
                )));
            }
        }

        // 3. Read-only boundary estimate conformance (or typed refusal).
        let boundary_span = if support.arbitrary_slice {
            0..case.split_at
        } else {
            0..case.total_steps
        };
        let expected_boundary_ticks = boundary_span.len();
        if support.read_only_boundary_estimate {
            // Baseline run over `boundary_span` without requesting a boundary estimate so we can
            // prove the boundary estimate is strictly read-only with respect to solver RNG draws,
            // outer steps, history resets, observer/post-step hooks, and numerical solver history.
            let baseline_driver = (case.make_driver)();
            let mut baseline_state = DriverRunState::new();
            let mut baseline_progress = Vec::new();
            let mut baseline_observe_count = 0usize;
            let mut baseline_post_step_count = 0usize;
            let baseline_out = {
                let mut on_progress = |p: Progress| baseline_progress.push(p);
                let mut on_observe = |_info: &DriverStepInfo, _l: &L| {
                    baseline_observe_count += 1;
                    Ok(())
                };
                let mut on_post_step = |_info: &DriverStepInfo, l: L| {
                    baseline_post_step_count += 1;
                    Ok(l)
                };
                let mut hooks = StepHooks::new(&mut on_progress);
                hooks.on_observe = Some(&mut on_observe);
                hooks.on_post_step = Some(&mut on_post_step);
                baseline_driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: boundary_span.clone(),
                        total_steps: case.total_steps,
                        state: BoundaryState::XSigma((case.make_initial)()),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut baseline_state,
                    &mut hooks,
                )?
            };

            let driver = (case.make_driver)();
            let mut st = DriverRunState::new();
            let mut progress = Vec::new();
            let mut boundary_observe_count = 0usize;
            let mut boundary_post_step_count = 0usize;
            let out = {
                let mut on_progress = |p: Progress| progress.push(p);
                let mut on_observe = |_info: &DriverStepInfo, _l: &L| {
                    boundary_observe_count += 1;
                    Ok(())
                };
                let mut on_post_step = |_info: &DriverStepInfo, l: L| {
                    boundary_post_step_count += 1;
                    Ok(l)
                };
                let mut hooks = StepHooks::new(&mut on_progress);
                hooks.on_observe = Some(&mut on_observe);
                hooks.on_post_step = Some(&mut on_post_step);
                driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: boundary_span,
                        total_steps: case.total_steps,
                        state: BoundaryState::XSigma((case.make_initial)()),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: true,
                    },
                    &mut st,
                    &mut hooks,
                )?
            };
            if !matches!(out, BoundaryState::CleanEstimate { .. }) {
                return Err(Error::Msg(format!(
                    "{}: read_only_boundary_estimate=true must return BoundaryState::CleanEstimate",
                    support.driver_id
                )));
            }
            if progress.len() != expected_boundary_ticks || progress != baseline_progress {
                return Err(Error::Msg(format!(
                    "{}: boundary estimate altered outer progress ({:?} != baseline {:?})",
                    support.driver_id, progress, baseline_progress
                )));
            }
            if st.boundary_evaluations == 0 {
                return Err(Error::Msg(format!(
                    "{}: read_only_boundary_estimate=true did not account boundary_evaluations",
                    support.driver_id
                )));
            }
            if st.stochastic_draw_index != baseline_state.stochastic_draw_index {
                return Err(Error::Msg(format!(
                    "{}: boundary estimate consumed solver RNG draws ({} != baseline {})",
                    support.driver_id,
                    st.stochastic_draw_index,
                    baseline_state.stochastic_draw_index
                )));
            }
            if st.completed_outer_steps != baseline_state.completed_outer_steps {
                return Err(Error::Msg(format!(
                    "{}: boundary estimate advanced completed_outer_steps ({} != baseline {})",
                    support.driver_id,
                    st.completed_outer_steps,
                    baseline_state.completed_outer_steps
                )));
            }
            if st.history_resets != baseline_state.history_resets {
                return Err(Error::Msg(format!(
                    "{}: boundary estimate mutated history_resets ({} != baseline {})",
                    support.driver_id, st.history_resets, baseline_state.history_resets
                )));
            }
            if boundary_observe_count != baseline_observe_count {
                return Err(Error::Msg(format!(
                    "{}: boundary estimate invoked on_observe ({} != baseline {})",
                    support.driver_id, boundary_observe_count, baseline_observe_count
                )));
            }
            if boundary_post_step_count != baseline_post_step_count {
                return Err(Error::Msg(format!(
                    "{}: boundary estimate invoked on_post_step ({} != baseline {})",
                    support.driver_id, boundary_post_step_count, baseline_post_step_count
                )));
            }
            if support.arbitrary_slice && support.carry_history {
                // Continuing the rest of the pass from the baseline XSigma using `st` (which
                // underwent the boundary estimate) must reproduce `unsplit_latent` bit-for-bit,
                // proving the boundary estimate did not mutate carried solver history.
                let mut cont_progress = Vec::new();
                let continued_out = {
                    let mut on_progress = |p: Progress| cont_progress.push(p);
                    let mut hooks = StepHooks::new(&mut on_progress);
                    driver.run_slice(
                        SliceInput {
                            pass: 0,
                            phase: 1,
                            span: case.split_at..case.total_steps,
                            total_steps: case.total_steps,
                            state: baseline_out,
                            transition: SliceTransition::Continue,
                            history: DriverHistoryPolicy::Carry,
                            geometry: Some(case.base_geometry),
                            seed: case.seed,
                            cancel: &cancel,
                            request_boundary_estimate: false,
                        },
                        &mut st,
                        &mut hooks,
                    )?
                };
                let continued_latent = extract_latent(continued_out);
                if !(case.latents_equal)(&continued_latent, &unsplit_latent) {
                    return Err(Error::Msg(format!(
                        "{}: boundary estimate mutated carried solver history (continuation diverged from unsplit trajectory)",
                        support.driver_id
                    )));
                }
            }
        } else {
            let driver = (case.make_driver)();
            let mut st = DriverRunState::new();
            let mut on_progress = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut on_progress);
            let boundary_res = driver.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 0,
                    span: boundary_span,
                    total_steps: case.total_steps,
                    state: BoundaryState::XSigma((case.make_initial)()),
                    transition: SliceTransition::Start,
                    history: DriverHistoryPolicy::Reset,
                    geometry: Some(case.base_geometry),
                    seed: case.seed,
                    cancel: &cancel,
                    request_boundary_estimate: true,
                },
                &mut st,
                &mut hooks,
            );
            if !matches!(boundary_res, Err(Error::Unsupported(_))) {
                return Err(Error::Msg(format!(
                    "{}: expected typed Unsupported refusal when read_only_boundary_estimate=false",
                    support.driver_id
                )));
            }
        }

        // 4. Renoise / Resample guards:
        //    - Carry across Renoise or Resample MUST always refuse.
        //    - Noisy XSigma handed to Renoise MUST always refuse (prevents double-noising).
        //    - Unsupported Renoise/Resample MUST refuse with Error::Unsupported.
        let driver = (case.make_driver)();
        let guard_span = if support.arbitrary_slice {
            case.split_at..case.total_steps
        } else {
            0..case.total_steps
        };
        {
            let mut st = DriverRunState::new();
            let mut on_progress = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut on_progress);
            let noisy_renoise = driver.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: guard_span.clone(),
                    total_steps: case.total_steps,
                    state: BoundaryState::XSigma((case.make_initial)()),
                    transition: SliceTransition::Renoise,
                    history: DriverHistoryPolicy::Reset,
                    geometry: Some(case.base_geometry),
                    seed: case.seed,
                    cancel: &cancel,
                    request_boundary_estimate: false,
                },
                &mut st,
                &mut hooks,
            );
            if !matches!(noisy_renoise, Err(Error::Unsupported(_))) {
                return Err(Error::Msg(format!(
                    "{}: Renoise with BoundaryState::XSigma must refuse to prevent double-noising",
                    support.driver_id
                )));
            }

            let carry_resample = driver.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: guard_span.clone(),
                    total_steps: case.total_steps,
                    state: BoundaryState::CleanEstimate {
                        latent: (case.make_initial)(),
                        source: crate::EstimateSource::EndpointEvaluation,
                    },
                    transition: SliceTransition::Resample,
                    history: DriverHistoryPolicy::Carry,
                    geometry: Some(case.alt_geometry),
                    seed: case.seed,
                    cancel: &cancel,
                    request_boundary_estimate: false,
                },
                &mut st,
                &mut hooks,
            );
            if !matches!(carry_resample, Err(Error::Unsupported(_))) {
                return Err(Error::Msg(format!(
                    "{}: Carry across Resample must refuse",
                    support.driver_id
                )));
            }

            if !support.renoise_from_clean {
                let res = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: guard_span.clone(),
                        total_steps: case.total_steps,
                        state: BoundaryState::CleanEstimate {
                            latent: (case.make_initial)(),
                            source: crate::EstimateSource::EndpointEvaluation,
                        },
                        transition: SliceTransition::Renoise,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.base_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                );
                if !matches!(res, Err(Error::Unsupported(_))) {
                    return Err(Error::Msg(format!(
                        "{}: expected typed Unsupported refusal when renoise_from_clean=false",
                        support.driver_id
                    )));
                }
            }
            if !support.geometry_change {
                let res = driver.run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: guard_span,
                        total_steps: case.total_steps,
                        state: BoundaryState::CleanEstimate {
                            latent: (case.make_initial)(),
                            source: crate::EstimateSource::EndpointEvaluation,
                        },
                        transition: SliceTransition::Resample,
                        history: DriverHistoryPolicy::Reset,
                        geometry: Some(case.alt_geometry),
                        seed: case.seed,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut st,
                    &mut hooks,
                );
                if !matches!(res, Err(Error::Unsupported(_))) {
                    return Err(Error::Msg(format!(
                        "{}: expected typed Unsupported refusal when geometry_change=false",
                        support.driver_id
                    )));
                }
            }
        }

        // 5. Cancellation: pre-cancelled returns Error::Canceled with 0 evaluations & 0 progress;
        //    mid-trajectory cancel after step 0 stops before completing all steps.
        let driver = (case.make_driver)();
        {
            let pre_cancel = CancelFlag::default();
            pre_cancel.cancel();
            let mut st = DriverRunState::new();
            let mut ticks = 0usize;
            let mut on_progress = |_: Progress| ticks += 1;
            let mut hooks = StepHooks::new(&mut on_progress);
            let res = driver.run_slice(
                SliceInput::full_trajectory(
                    (case.make_initial)(),
                    case.total_steps,
                    case.seed,
                    &pre_cancel,
                )
                .with_geometry(case.base_geometry.0, case.base_geometry.1),
                &mut st,
                &mut hooks,
            );
            if !matches!(res, Err(Error::Canceled)) || ticks != 0 || st.internal_evaluations != 0 {
                return Err(Error::Msg(format!(
                    "{}: pre-cancelled trajectory must return Error::Canceled with 0 evals/ticks (evals={}, ticks={})",
                    support.driver_id, st.internal_evaluations, ticks
                )));
            }
        }
        {
            let mid_cancel = CancelFlag::default();
            let mid_cancel_hook = mid_cancel.clone();
            let mut st = DriverRunState::new();
            let mut ticks = 0usize;
            let mut on_progress = |_: Progress| {
                ticks += 1;
                mid_cancel_hook.cancel();
            };
            let mut hooks = StepHooks::new(&mut on_progress);
            let res = driver.run_slice(
                SliceInput::full_trajectory(
                    (case.make_initial)(),
                    case.total_steps,
                    case.seed,
                    &mid_cancel,
                )
                .with_geometry(case.base_geometry.0, case.base_geometry.1),
                &mut st,
                &mut hooks,
            );
            if !matches!(res, Err(Error::Canceled)) || ticks != 1 {
                return Err(Error::Msg(format!(
                    "{}: mid-trajectory cancel after step 1 must return Error::Canceled at tick 1 (got ticks={ticks})",
                    support.driver_id
                )));
            }
        }

        Ok(ConformanceReport {
            driver_id: support.driver_id,
            total_steps: case.total_steps,
            unsplit_evaluations: unsplit_state.internal_evaluations,
            unsplit_draws: unsplit_state.stochastic_draw_index,
            verified_inert_split,
            verified_carry_refusal_or_support: true,
            verified_reset_preserves_draws,
            verified_boundary_estimate_readonly_or_refusal: true,
            verified_renoise_and_resample_guards: true,
            verified_cancellation: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::conformance::{verify_driver_conformance, ConformanceCase};
    use super::*;
    use std::sync::{Arc, Barrier};

    struct SyntheticDriver<F> {
        support: DriverSupport,
        forward: F,
    }

    impl<F: Fn(&[f32], usize, DriverEvaluationKind) -> Vec<f32>> FamilyDriver<Vec<f32>>
        for SyntheticDriver<F>
    {
        fn support(&self) -> &DriverSupport {
            &self.support
        }

        fn run_slice(
            &self,
            input: SliceInput<'_, Vec<f32>>,
            state: &mut DriverRunState,
            hooks: &mut StepHooks<'_, Vec<f32>>,
        ) -> Result<BoundaryState<Vec<f32>>> {
            self.support.validate_slice(&input, state)?;
            state.enter_slice(&input)?;
            let mut latent = match input.state {
                BoundaryState::XSigma(x)
                | BoundaryState::CleanEstimate { latent: x, .. }
                | BoundaryState::RouteSpecific { state: x, .. } => x,
            };

            let mut acc = state.take_history::<f32>().unwrap_or(0.0);
            for step in input.span.clone() {
                if input.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let info = DriverStepInfo {
                    pass: input.pass,
                    phase: input.phase,
                    outer_step: step,
                    total_outer_steps: input.total_steps,
                    eval_in_step: 0,
                    kind: DriverEvaluationKind::Trajectory,
                    sigma: (input.total_steps - step) as f32,
                    next_sigma: (input.total_steps - step - 1) as f32,
                };
                hooks.observe(&info, &latent)?;
                let pred = (self.forward)(&latent, step, DriverEvaluationKind::Trajectory);
                state.record_evaluation(DriverEvaluationKind::Trajectory);
                let draw_idx = state.next_draw_index();
                acc += 0.1 * (step as f32 + 1.0);
                for (x, p) in latent.iter_mut().zip(pred.iter()) {
                    *x = *x * 0.9 + *p * 0.1 + acc * 0.01 + (draw_idx as f32) * 1e-4;
                }
                latent = hooks.post_step(&info, latent)?;
                hooks.materialize(&info, &latent)?;
                state.record_outer_step(step);
                hooks.emit_progress(&info);
            }
            state.set_history(acc);
            state.finish_slice(input.span.end);

            if input.request_boundary_estimate {
                if input.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let est = (self.forward)(
                    &latent,
                    input.span.end,
                    DriverEvaluationKind::BoundaryEstimate,
                );
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: est,
                    source: crate::EstimateSource::EndpointEvaluation,
                });
            }
            Ok(BoundaryState::XSigma(latent))
        }
    }

    #[test]
    fn conformance_harness_verifies_full_featured_and_restricted_drivers() {
        let full_support = DriverSupport {
            driver_id: "synthetic_full",
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: true,
            geometry_change: true,
            read_only_boundary_estimate: true,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons::default(),
        };
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 42,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: || {
                Box::new(SyntheticDriver {
                    support: full_support.clone(),
                    forward: |x: &[f32], step: usize, _kind| {
                        x.iter().map(|&v| v * 0.5 + step as f32 * 0.25).collect()
                    },
                })
            },
            make_initial: || vec![1.0, -0.5, 0.25, 0.75],
            latents_equal: |a, b| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("full-featured synthetic driver passes conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_reset_preserves_draws);
        assert_eq!(report.unsplit_evaluations, 4);
        assert_eq!(report.unsplit_draws, 4);

        let restricted_support = DriverSupport {
            driver_id: "synthetic_restricted",
            arbitrary_slice: false,
            carry_history: false,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: true,
            refusal_reasons: DriverRefusalReasons::default(),
        };
        let restricted = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 99,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: || {
                Box::new(SyntheticDriver {
                    support: restricted_support.clone(),
                    forward: |x: &[f32], step: usize, _kind| {
                        x.iter().map(|&v| v + step as f32).collect()
                    },
                })
            },
            make_initial: || vec![0.1, 0.2, 0.3, 0.4],
            latents_equal: |a, b| a == b,
        })
        .expect("restricted synthetic driver passes conformance with typed refusals");
        assert!(!restricted.verified_inert_split);
        assert!(restricted.verified_cancellation);
    }

    #[test]
    fn global_rng_trajectory_gate_serializes_whole_trajectories_without_interleaving() {
        // B3 / PIPE-14 (§6.10): concurrent submissions with seed 11 and seed 22 on a global-RNG
        // provider must execute as two atomic trajectories (seed -> draws -> materialize) with
        // each result matching its isolated single-thread trace bit-for-bit.
        #[derive(Default)]
        struct FakeGlobalRng {
            state: u64,
            events: Vec<(u64, &'static str)>,
        }
        impl FakeGlobalRng {
            fn seed(&mut self, seed: u64) {
                self.state = seed;
                self.events.push((seed, "seed"));
            }
            fn draw(&mut self, tag: u64) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.events.push((tag, "draw"));
                self.state
            }
            fn materialize(&mut self, tag: u64) {
                self.events.push((tag, "materialize"));
            }
        }

        let isolated = |seed: u64| -> Vec<u64> {
            let mut rng = FakeGlobalRng::default();
            rng.seed(seed);
            let mut out = Vec::new();
            for _ in 0..4 {
                out.push(rng.draw(seed));
            }
            rng.materialize(seed);
            out
        };
        let want_11 = isolated(11);
        let want_22 = isolated(22);

        let gate = Arc::new(GlobalRngTrajectoryGate::new());
        let fake_rng = Arc::new(Mutex::new(FakeGlobalRng::default()));
        let start = Arc::new(Barrier::new(2));

        let spawn_worker = |seed: u64| {
            let gate = Arc::clone(&gate);
            let fake_rng = Arc::clone(&fake_rng);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                gate.run_trajectory(|| {
                    // Nested reentrant acquisition on the owning thread must not deadlock.
                    gate.run_trajectory(|| {
                        fake_rng.lock().unwrap().seed(seed);
                        let mut draws = Vec::with_capacity(4);
                        for _ in 0..4 {
                            draws.push(fake_rng.lock().unwrap().draw(seed));
                            std::thread::yield_now();
                        }
                        fake_rng.lock().unwrap().materialize(seed);
                        draws
                    })
                })
            })
        };

        let t1 = spawn_worker(11);
        let t2 = spawn_worker(22);
        let got_11 = t1.join().unwrap();
        let got_22 = t2.join().unwrap();
        assert_eq!(got_11, want_11, "seed 11 must match isolated trace");
        assert_eq!(got_22, want_22, "seed 22 must match isolated trace");

        let events = &fake_rng.lock().unwrap().events;
        assert_eq!(events.len(), 12);
        let first_seed = events[0].0;
        let second_seed = events[6].0;
        assert_ne!(first_seed, second_seed);
        assert!(
            events[0..6].iter().all(|(s, _)| *s == first_seed),
            "first trajectory must complete seed->draws->materialize without interleaving: {events:?}"
        );
        assert!(
            events[6..12].iter().all(|(s, _)| *s == second_seed),
            "second trajectory must complete seed->draws->materialize without interleaving: {events:?}"
        );
        assert_eq!(events[0].1, "seed");
        assert_eq!(events[5].1, "materialize");
        assert_eq!(events[6].1, "seed");
        assert_eq!(events[11].1, "materialize");
    }

    #[test]
    fn global_rng_trajectory_gate_keeps_distinct_gates_mutually_exclusive_across_threads() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let gate_a = Arc::new(GlobalRngTrajectoryGate::new());
        let gate_b = Arc::new(GlobalRngTrajectoryGate::new());
        let t2_inside_b = Arc::new(Barrier::new(2));
        let t1_attempting_b = Arc::new(AtomicBool::new(false));
        let active_in_b = Arc::new(AtomicUsize::new(0));
        let max_in_b = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        let t2_gate_b = Arc::clone(&gate_b);
        let t2_barrier = Arc::clone(&t2_inside_b);
        let t2_attempting = Arc::clone(&t1_attempting_b);
        let t2_active = Arc::clone(&active_in_b);
        let t2_max = Arc::clone(&max_in_b);
        let t2_events = Arc::clone(&events);
        let t2 = std::thread::spawn(move || {
            t2_gate_b.run_trajectory(|| {
                let now = t2_active.fetch_add(1, Ordering::SeqCst) + 1;
                t2_max.fetch_max(now, Ordering::SeqCst);
                t2_events.lock().unwrap().push("t2_enter_b");
                // Synchronize once T2 definitely owns `gate_b` and T1 already owns `gate_a`.
                t2_barrier.wait();
                while !t2_attempting.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                for _ in 0..64 {
                    std::thread::yield_now();
                }
                assert_eq!(
                    t2_active.load(Ordering::SeqCst),
                    1,
                    "T1 holding gate_a must not bypass gate_b's mutex"
                );
                t2_events.lock().unwrap().push("t2_exit_b");
                t2_active.fetch_sub(1, Ordering::SeqCst);
            });
        });

        let t1_gate_a = Arc::clone(&gate_a);
        let t1_gate_b = Arc::clone(&gate_b);
        let t1_barrier = Arc::clone(&t2_inside_b);
        let t1_attempting = Arc::clone(&t1_attempting_b);
        let t1_active = Arc::clone(&active_in_b);
        let t1_max = Arc::clone(&max_in_b);
        let t1_events = Arc::clone(&events);
        let t1 = std::thread::spawn(move || {
            let lease_a = t1_gate_a.acquire();
            t1_barrier.wait();
            t1_attempting.store(true, Ordering::SeqCst);
            let lease_b = t1_gate_b.acquire();
            let now = t1_active.fetch_add(1, Ordering::SeqCst) + 1;
            t1_max.fetch_max(now, Ordering::SeqCst);
            t1_events.lock().unwrap().push("t1_enter_b");
            // Re-entering `gate_b` on the owning thread remains non-deadlocking.
            t1_gate_b.run_trajectory(|| {
                t1_events.lock().unwrap().push("t1_reenter_b");
            });
            // Dropping `gate_a` before `gate_b` must keep `gate_b` locked and reentrant.
            drop(lease_a);
            t1_gate_b.run_trajectory(|| {
                t1_events.lock().unwrap().push("t1_still_holds_b");
            });
            t1_events.lock().unwrap().push("t1_exit_b");
            t1_active.fetch_sub(1, Ordering::SeqCst);
            drop(lease_b);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(max_in_b.load(Ordering::SeqCst), 1);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "t2_enter_b",
                "t2_exit_b",
                "t1_enter_b",
                "t1_reenter_b",
                "t1_still_holds_b",
                "t1_exit_b",
            ]
        );
    }

    #[test]
    fn conformance_harness_catches_reset_draw_wipe_missing_reset_refusal_and_dirty_boundary_estimate(
    ) {
        #[derive(Clone, Copy)]
        enum Fault {
            None,
            WipeDrawsOnReset,
            IgnoreResetRefusal,
            BoundaryConsumesDraw,
            BoundaryAdvancesOuterStep,
            BoundaryInvokesObserve,
            BoundaryInvokesPostStep,
            BoundaryResetsHistory,
            BoundaryMutatesHistory,
        }

        struct FaultyDriver {
            support: DriverSupport,
            fault: Fault,
        }

        impl FamilyDriver<Vec<f32>> for FaultyDriver {
            fn support(&self) -> &DriverSupport {
                &self.support
            }

            fn run_slice(
                &self,
                input: SliceInput<'_, Vec<f32>>,
                state: &mut DriverRunState,
                hooks: &mut StepHooks<'_, Vec<f32>>,
            ) -> Result<BoundaryState<Vec<f32>>> {
                if !(matches!(self.fault, Fault::IgnoreResetRefusal)
                    && input.transition == SliceTransition::Continue
                    && input.history == DriverHistoryPolicy::Reset)
                {
                    self.support.validate_slice(&input, state)?;
                }
                state.enter_slice(&input)?;
                if matches!(self.fault, Fault::WipeDrawsOnReset)
                    && input.transition == SliceTransition::Continue
                    && input.history == DriverHistoryPolicy::Reset
                {
                    state.stochastic_draw_index = 0;
                }
                let mut latent = match input.state {
                    BoundaryState::XSigma(x)
                    | BoundaryState::CleanEstimate { latent: x, .. }
                    | BoundaryState::RouteSpecific { state: x, .. } => x,
                };
                let mut acc = state.take_history::<f32>().unwrap_or(0.0);
                for step in input.span.clone() {
                    if input.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let info = DriverStepInfo {
                        pass: input.pass,
                        phase: input.phase,
                        outer_step: step,
                        total_outer_steps: input.total_steps,
                        eval_in_step: 0,
                        kind: DriverEvaluationKind::Trajectory,
                        sigma: (input.total_steps - step) as f32,
                        next_sigma: (input.total_steps - step - 1) as f32,
                    };
                    hooks.observe(&info, &latent)?;
                    state.record_evaluation(DriverEvaluationKind::Trajectory);
                    let draw_idx = state.next_draw_index();
                    acc += 0.1 * (step as f32 + 1.0);
                    for x in &mut latent {
                        *x = *x * 0.9 + acc * 0.01 + (draw_idx as f32) * 1e-4;
                    }
                    latent = hooks.post_step(&info, latent)?;
                    state.record_outer_step(step);
                    hooks.emit_progress(&info);
                }
                state.set_history(acc);
                state.finish_slice(input.span.end);

                if input.request_boundary_estimate {
                    state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                    let fake_traj_info = DriverStepInfo {
                        pass: input.pass,
                        phase: input.phase,
                        outer_step: input.span.end,
                        total_outer_steps: input.total_steps,
                        eval_in_step: 0,
                        kind: DriverEvaluationKind::Trajectory,
                        sigma: 0.0,
                        next_sigma: 0.0,
                    };
                    match self.fault {
                        Fault::BoundaryConsumesDraw => {
                            let _ = state.next_draw_index();
                        }
                        Fault::BoundaryAdvancesOuterStep => {
                            state.record_outer_step(input.span.end);
                        }
                        Fault::BoundaryInvokesObserve => {
                            hooks.observe(&fake_traj_info, &latent)?;
                        }
                        Fault::BoundaryInvokesPostStep => {
                            latent = hooks.post_step(&fake_traj_info, latent)?;
                        }
                        Fault::BoundaryResetsHistory => {
                            state.reset_history();
                        }
                        Fault::BoundaryMutatesHistory => {
                            state.set_history(acc + 42.0);
                        }
                        _ => {}
                    }
                    return Ok(BoundaryState::CleanEstimate {
                        latent,
                        source: crate::EstimateSource::EndpointEvaluation,
                    });
                }
                Ok(BoundaryState::XSigma(latent))
            }
        }

        let base_support = DriverSupport {
            driver_id: "faulty_driver",
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: true,
            geometry_change: true,
            read_only_boundary_estimate: true,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons::default(),
        };

        let run_case = |support: DriverSupport, fault: Fault| {
            verify_driver_conformance(ConformanceCase {
                total_steps: 4,
                split_at: 2,
                seed: 7,
                base_geometry: (512, 512),
                alt_geometry: (1024, 1024),
                make_driver: move || {
                    Box::new(FaultyDriver {
                        support: support.clone(),
                        fault,
                    })
                },
                make_initial: || vec![1.0, -0.5, 0.25, 0.75],
                latents_equal: |a, b| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                },
            })
        };

        // Sanity: clean driver passes, including when `reset_history=false` with proper refusal.
        assert!(run_case(base_support.clone(), Fault::None).is_ok());
        let no_reset_support = DriverSupport {
            reset_history: false,
            ..base_support.clone()
        };
        let no_reset_report =
            run_case(no_reset_support.clone(), Fault::None).expect("typed refusal should pass");
        assert!(!no_reset_report.verified_reset_preserves_draws);

        // Every injected fault must be caught by `verify_driver_conformance`.
        for (support, fault, expected_fragment) in [
            (
                base_support.clone(),
                Fault::WipeDrawsOnReset,
                "stochastic_draw_index",
            ),
            (
                no_reset_support,
                Fault::IgnoreResetRefusal,
                "Continue+Reset",
            ),
            (
                base_support.clone(),
                Fault::BoundaryConsumesDraw,
                "consumed solver RNG draws",
            ),
            (
                base_support.clone(),
                Fault::BoundaryAdvancesOuterStep,
                "completed_outer_steps",
            ),
            (
                base_support.clone(),
                Fault::BoundaryInvokesObserve,
                "on_observe",
            ),
            (
                base_support.clone(),
                Fault::BoundaryInvokesPostStep,
                "on_post_step",
            ),
            (
                base_support.clone(),
                Fault::BoundaryResetsHistory,
                "history_resets",
            ),
            (
                base_support,
                Fault::BoundaryMutatesHistory,
                "carried solver history",
            ),
        ] {
            let err = run_case(support, fault).expect_err("injected fault must fail conformance");
            let msg = err.to_string();
            assert!(
                msg.contains(expected_fragment),
                "expected error mentioning {expected_fragment:?}, got {msg:?}"
            );
        }
    }
}

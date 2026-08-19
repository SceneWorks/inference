//! **Chained denoise passes** — the versioned `advanced.denoisePasses` request contract
//! (epic 20414, sc-20415).
//!
//! A denoise pass is a *complete, self-contained denoise run* over the current latent. A request
//! carries an **ordered** list of them; pass *n+1* picks up the latent pass *n* produced and denoises
//! it again with its **own** step count, sampler, scheduler, `denoise` fraction, guidance and
//! adapter weights.
//!
//! # The runtime invariant
//!
//! This module owns the request/plan contract only — sc-20418 wires the execution. The invariant the
//! execution must satisfy is stated here, once, so both sides are written against the same words:
//!
//! 1. **Later passes operate on the current latent.** Pass *n+1* starts from the latent pass *n*
//!    ended on. There is **no VAE round trip** between passes — no decode to pixels and no re-encode,
//!    so no VAE reconstruction loss and no resample is introduced at a pass boundary.
//! 2. **Each pass gets its own fresh schedule.** A pass builds a brand-new sigma schedule from its
//!    own [`scheduler`](DenoisePass::scheduler) and [`steps`](DenoisePass::steps), then enters it at
//!    the point named by its [`denoise`](DenoisePass::denoise) fraction (`denoise == 1.0` ⇒ the full
//!    trajectory from the top; `denoise == 0.2` ⇒ the last 20 % of it). Schedules are **not** shared
//!    or spliced across passes.
//! 3. **Solver history is reset at every pass boundary.** Multistep solvers (`dpmpp_2m`, `uni_pc`, …)
//!    carry state from previous evaluations; that state belongs to the schedule it was accumulated
//!    on. A pass therefore starts with an empty history — carrying it across a schedule reset would
//!    mix derivatives taken at incomparable sigmas.
//! 4. **Each pass has its own noise stream.** The per-pass seed is derived deterministically from the
//!    job seed and the pass index ([`denoise_pass_seed`]) and is recorded in the resolved plan, so a
//!    replay reproduces every pass exactly.
//!
//! # Denoise passes vs. Krea RAW multi-phase denoise
//!
//! [`GenerationRequest::phases`](crate::GenerationRequest::phases) (epic 13879, sc-13884) and
//! [`GenerationRequest::denoise_passes`](crate::GenerationRequest::denoise_passes) look similar and
//! are **not** the same mechanism. They are mutually exclusive in one request, and the shared floor
//! rejects a request that sets both.
//!
//! | | `phases` — RAW **multi-phase denoise** | `denoisePasses` — **denoise passes** |
//! |---|---|---|
//! | Schedule | ONE global sigma schedule, shared by every phase | a **fresh** schedule per pass |
//! | A phase/pass is | a contiguous **slice** of that one schedule | a complete denoise run of its own |
//! | Boundary | resumes at the sigma the prior phase reached — no reset | full reset: new schedule, new solver history |
//! | Total steps | the **sum** of the phases' steps | each pass runs its own `steps` |
//! | Sampler | one, request-level | per pass |
//! | Scheduler | one, request-level (deliberately not per-phase) | per pass |
//! | `denoise` fraction | none — the trajectory is continuous | per pass, in `(0, 1]` |
//! | Adapters, empty list | the **bare base model** (no adapters) | the load-time stack at its load-time scales |
//! | Adapters, non-empty | the exact set of adapters active for the phase | **weight overrides** for the named adapters only |
//! | Scope | the Krea RAW family | model-agnostic |
//!
//! The adapter row is the trap: a phase's `adapters` list *selects* which of the load-time stack is
//! active, so `vec![]` means "run bare base". A pass's `adapters` list *overrides weights*, so
//! `vec![]` means "leave the load-time stack exactly as it was loaded". Both reuse the same
//! [`PhaseAdapter`] reference type — the type is a `(load-time index, optional weight)` pair in both
//! mechanisms; only the meaning of the empty list differs.
//!
//! # Resolution
//!
//! [`resolve_denoise_plan`] turns a request into a [`ResolvedDenoisePlan`]: every pass field is
//! **explicit**, resolved `pass value → top-level request value → model default`, with the per-pass
//! seed already derived. Execution and metadata therefore never re-read the request or depend on
//! implicit UI state.
//!
//! A request with **no** `denoisePasses` resolves through the very same function to a **one-pass**
//! plan carrying the top-level request's own sampler/scheduler/steps/guidance/seed — that identity is
//! what makes the legacy path and the new path one code path rather than two (see
//! `single_pass_plan_matches_top_level_request` in this module's tests).
//!
//! # Wire format
//!
//! The JSON is the SceneWorks `advanced.denoisePasses` array, camelCase, and is decoded/encoded here
//! by hand ([`denoise_passes_from_json`] / [`denoise_passes_to_json`]) because gen-core deliberately
//! carries no `serde` derive. Decoding is **strict**: an unknown key is an error, which is what makes
//! the round-trip provably lossless rather than silently lossy. Cross-language fixtures live at
//! `crates/contracts/gen-core/tests/fixtures/denoise_passes/` and are consumed by both repositories.
//!
//! One deliberate deviation from "natural JSON types": in a **resolved plan**, `jobSeed` and every
//! per-pass `seed` are decimal **strings**. Seeds are full-range `u64` — [`denoise_pass_seed`]
//! finishes with splitmix64, so a derived pass seed is above 2^53 in ~99.9% of cases — and the other
//! side of this contract is JavaScript, whose `JSON.parse` silently rounds such a number to the
//! nearest `f64`. A truncated seed raises no error and replays as a *different image*, so the plan
//! moves seeds as text and [`ResolvedDenoisePlan::from_json`] refuses a bare number outright.

use std::fmt;

use crate::error::Error;
use crate::generator::PhaseAdapter;
use crate::sampling::schedulers::Scheduler;
use crate::sampling::solvers::Solver;

use serde_json::{Map, Value};

/// The version of this contract, stamped into every [`ResolvedDenoisePlan`] JSON.
///
/// A resolved plan is a **persisted** artifact: it is stored beside a render so the render can be
/// replayed exactly. Without a version stamp, a plan written by a later build — one that changed the
/// seed derivation, redefined what `denoise` indexes, or altered the resolution ladder — would be
/// read by an older build as if it meant the same thing, and would replay to a *different image*
/// with no error raised anywhere. [`ResolvedDenoisePlan::from_json`] therefore refuses a plan whose
/// version it does not implement rather than silently mis-reading it.
///
/// The **request** side deliberately carries no version. `advanced.denoisePasses` is a bare array
/// with no natural carrier, its decoding is strict (an unknown key is an error, not a silent drop),
/// and its compatibility rule needs no negotiation: absent means the ordinary single-pass render,
/// for every build that has ever existed.
///
/// Bump this when a change would make an older build mis-read a newer plan. Adding a *new optional*
/// per-pass field an older build may ignore is not such a change; changing [`denoise_pass_seed`],
/// the meaning of `denoise`, or the resolution ladder is.
pub const DENOISE_PASS_CONTRACT_VERSION: u32 = 1;

/// The largest number of chained passes one request may carry.
///
/// A sanity cap, not a model limit: each pass is a full denoise run, so a pathological list is an
/// unbounded, cancel-only-recoverable job — the same footgun class the flat
/// `steps` sanity cap closes. Real recipes use two or three passes.
pub const MAX_DENOISE_PASSES: usize = 16;

/// The [`denoise`](DenoisePass::denoise) fraction used when a pass does not name one: the **full**
/// trajectory. This is the value that makes a one-pass plan identical to the legacy top-level
/// request.
pub const DEFAULT_PASS_DENOISE: f32 = 1.0;

/// The odd 64-bit constant (`⌊2⁶⁴ / φ⌋`) that keys one pass's noise stream away from the next.
///
/// The same golden-ratio constant the per-step noise subkey uses
/// ([`CpuLatentOps::randn_like`](crate::sampling::latent_ops::CpuLatentOps)); sharing it is
/// deliberate — one repository-wide domain-separation constant, mixed with a different index.
pub const DENOISE_PASS_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The seed for pass `pass_index` of a job seeded with `job_seed`.
///
/// **Pass 0 returns `job_seed` unchanged.** That is load-bearing, not an oversight: a one-pass plan
/// must be bit-identical to the legacy single-trajectory render of the same request, and every
/// existing replay payload names only the job seed. Later passes are separated by mixing the salt in
/// at the pass index and running a splitmix64 finalizer, so pass 1 and pass 2 do not merely differ by
/// a constant offset (adjacent seeds are a real correlation source for the per-step subkey, which is
/// itself an additive derivation).
///
/// Deterministic and pure: the resolved value is recorded in
/// [`ResolvedDenoisePass::seed`] so an exact replay never has to re-derive it.
#[inline]
#[must_use]
pub fn denoise_pass_seed(job_seed: u64, pass_index: usize) -> u64 {
    if pass_index == 0 {
        return job_seed;
    }
    let mut z = job_seed.wrapping_add(DENOISE_PASS_SEED_SALT.wrapping_mul(pass_index as u64));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// =================================================================================================
// Request-side types
// =================================================================================================

/// One pass of a [chained denoise](self). Every field except the adapter list is optional and
/// inherits from the top-level request (then the model default) during
/// [resolution](resolve_denoise_plan).
///
/// Additive and `Default`-able like the rest of the request surface:
/// `DenoisePass { steps: Some(10), ..Default::default() }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DenoisePass {
    /// Denoise steps for **this pass alone** (each pass runs its own schedule, so these are not
    /// slices of a shared budget). Must be `>= 1` when present; `None` inherits
    /// [`GenerationRequest::steps`](crate::GenerationRequest::steps), then the model default.
    pub steps: Option<u32>,
    /// Canonical sampler id for this pass (e.g. `"euler"`), validated against the model's advertised
    /// [`Capabilities::samplers`](crate::Capabilities::samplers) menu — or, with no model in hand,
    /// against the curated [`Solver`] registry. `None` inherits
    /// [`GenerationRequest::sampler`](crate::GenerationRequest::sampler), then the model default.
    pub sampler: Option<String>,
    /// Canonical scheduler id for this pass (e.g. `"normal"`), validated against the model's
    /// advertised [`Capabilities::schedulers`](crate::Capabilities::schedulers) menu — or, with no
    /// model in hand, against the curated [`Scheduler`] registry. `None` inherits
    /// [`GenerationRequest::scheduler`](crate::GenerationRequest::scheduler), then the model default.
    pub scheduler: Option<String>,
    /// How much of this pass's freshly-built schedule to actually run, in `(0, 1]`. `1.0` runs the
    /// whole trajectory from the top; `0.2` enters at the last 20 % of it (the refinement case).
    /// `None` ⇒ [`DEFAULT_PASS_DENOISE`]. Zero is rejected — a zero-denoise pass is a no-op that
    /// costs a full model load, which is a malformed request rather than a useful one.
    ///
    /// Distinct from [`GenerationRequest::strength`](crate::GenerationRequest::strength): `strength`
    /// is the img2img fidelity of the *conditioning image*, a whole-request concept; `denoise` is
    /// this pass's entry point into its own schedule.
    pub denoise: Option<f32>,
    /// Guidance override for this pass. `None` inherits
    /// [`GenerationRequest::guidance`](crate::GenerationRequest::guidance), then the model default.
    /// Joins the request finiteness floor.
    pub guidance: Option<f32>,
    /// Pass-local **weight overrides** for the load-time adapter stack
    /// ([`LoadSpec::adapters`](crate::LoadSpec::adapters), by index).
    ///
    /// An **empty** list means the load-time stack runs at its load-time scales — *not* "no
    /// adapters". That is the opposite of
    /// [`GenerationPhase::adapters`](crate::GenerationPhase::adapters), which selects the active set;
    /// see the [module docs](self) for the full comparison. An entry with `weight: None` explicitly
    /// pins the load-time scale for this pass (a no-op that documents intent); an entry with
    /// `weight: Some(w)` scales that adapter's contribution to `w` for this pass only — the
    /// "adapter off for pass 1, on for pass 2" recipe is `weight: Some(0.0)` then `Some(1.0)`.
    pub adapters: Vec<PhaseAdapter>,
}

// =================================================================================================
// Errors
// =================================================================================================

/// Which field of a denoise-pass request a [`DenoisePassError`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoisePassField {
    /// The `denoisePasses` array itself (arity, mutual exclusion, capability).
    DenoisePasses,
    /// [`DenoisePass::steps`].
    Steps,
    /// [`DenoisePass::sampler`].
    Sampler,
    /// [`DenoisePass::scheduler`].
    Scheduler,
    /// [`DenoisePass::denoise`].
    Denoise,
    /// [`DenoisePass::guidance`].
    Guidance,
    /// [`DenoisePass::adapters`] — the list as a whole.
    Adapters,
    /// The `adapter` index of one [`PhaseAdapter`] entry.
    AdapterIndex,
    /// The `weight` of one [`PhaseAdapter`] entry.
    AdapterWeight,
}

impl DenoisePassField {
    /// The field's name **as it appears on the wire** (camelCase, matching the SceneWorks request).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::DenoisePasses => "denoisePasses",
            Self::Steps => "steps",
            Self::Sampler => "sampler",
            Self::Scheduler => "scheduler",
            Self::Denoise => "denoise",
            Self::Guidance => "guidance",
            Self::Adapters => "adapters",
            Self::AdapterIndex => "adapters.adapter",
            Self::AdapterWeight => "adapters.weight",
        }
    }
}

impl fmt::Display for DenoisePassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What is wrong with the field a [`DenoisePassError`] names.
#[derive(Clone, Debug, PartialEq)]
pub enum DenoisePassIssue {
    /// The request set both `phases` and `denoisePasses`. They are different mechanisms over the same
    /// denoise (see the [module docs](self)) and there is no defined composition of the two.
    MutuallyExclusiveWithPhases,
    /// The model does not advertise
    /// [`Capabilities::supports_denoise_passes`](crate::Capabilities::supports_denoise_passes).
    Unsupported,
    /// `denoisePasses: []`. Send at least one pass, or omit the field for the ordinary single-pass
    /// render.
    Empty,
    /// More passes than [`MAX_DENOISE_PASSES`].
    TooMany { count: usize, max: usize },
    /// An integer field that must be `>= 1` was zero.
    NotPositive,
    /// An integer field exceeded its sanity cap.
    AboveMax { value: u64, max: u64 },
    /// A float field fell outside its half-open/closed range.
    OutOfRange { value: f32, min: f32, max: f32 },
    /// A float field was NaN or ±Inf. Rejected for the same reason the flat request floor rejects it
    /// (F-053): a non-finite knob poisons the whole denoise silently.
    NotFinite { value: f32 },
    /// A sampler/scheduler id that is not in the supported set.
    UnknownId { id: String, supported: Vec<String> },
    /// An adapter index past the end of the load-time stack.
    AdapterIndexOutOfRange { index: usize, loaded: usize },
    /// Two entries in one pass override the same load-time adapter — contradictory, so rejected
    /// rather than resolved by last-write-wins.
    DuplicateAdapter { index: usize },
    /// The JSON was structurally wrong (wrong type, unknown key, non-integer where an integer was
    /// required).
    Malformed { detail: String },
}

impl DenoisePassIssue {
    /// A stable, machine-readable discriminant for this issue.
    ///
    /// The cross-language fixtures name expected failures by this code, so a SceneWorks-side test can
    /// assert the *same* rejection this crate makes without depending on the English message.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::MutuallyExclusiveWithPhases => "mutuallyExclusiveWithPhases",
            Self::Unsupported => "unsupported",
            Self::Empty => "empty",
            Self::TooMany { .. } => "tooMany",
            Self::NotPositive => "notPositive",
            Self::AboveMax { .. } => "aboveMax",
            Self::OutOfRange { .. } => "outOfRange",
            Self::NotFinite { .. } => "notFinite",
            Self::UnknownId { .. } => "unknownId",
            Self::AdapterIndexOutOfRange { .. } => "adapterIndexOutOfRange",
            Self::DuplicateAdapter { .. } => "duplicateAdapter",
            Self::Malformed { .. } => "malformed",
        }
    }
}

impl fmt::Display for DenoisePassIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MutuallyExclusiveWithPhases => f.write_str(
                "`phases` and `denoisePasses` are mutually exclusive — send one or the other",
            ),
            Self::Unsupported => f.write_str("this model does not support chained denoise passes"),
            Self::Empty => f.write_str(
                "must not be empty — send at least one pass, or omit the field for a single-pass \
                 render",
            ),
            Self::TooMany { count, max } => write!(f, "{count} passes exceeds the maximum {max}"),
            Self::NotPositive => f.write_str("must be >= 1"),
            Self::AboveMax { value, max } => write!(f, "{value} exceeds the sanity cap {max}"),
            Self::OutOfRange { value, min, max } => {
                write!(f, "{value} is outside the range ({min}, {max}]")
            }
            Self::NotFinite { value } => write!(f, "must be finite (got {value})"),
            Self::UnknownId { id, supported } => {
                write!(f, "unknown id {id:?} (supported: {supported:?})")
            }
            Self::AdapterIndexOutOfRange { index, loaded } => write!(
                f,
                "adapter index {index} is out of range of the {loaded} loaded adapter(s)"
            ),
            Self::DuplicateAdapter { index } => {
                write!(
                    f,
                    "adapter index {index} is overridden more than once in this pass"
                )
            }
            Self::Malformed { detail } => f.write_str(detail),
        }
    }
}

/// A denoise-pass contract violation, naming the **pass index** and the **invalid field**.
///
/// Both are machine-readable ([`pass_index`](Self::pass_index) / [`field`](Self::field)) rather than
/// only present in the message, so a consumer UI can highlight the offending control instead of
/// re-parsing prose. [`path`](Self::path) renders the JSON pointer-ish path the error is about, e.g.
/// `advanced.denoisePasses[1].denoise`.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoisePassError {
    pass_index: Option<usize>,
    field: DenoisePassField,
    issue: DenoisePassIssue,
}

impl DenoisePassError {
    /// An error about one specific pass.
    #[must_use]
    pub fn at(pass_index: usize, field: DenoisePassField, issue: DenoisePassIssue) -> Self {
        Self {
            pass_index: Some(pass_index),
            field,
            issue,
        }
    }

    /// An error about the `denoisePasses` array as a whole (no single pass is at fault).
    #[must_use]
    pub fn whole(field: DenoisePassField, issue: DenoisePassIssue) -> Self {
        Self {
            pass_index: None,
            field,
            issue,
        }
    }

    /// The zero-based index of the offending pass, or `None` when the whole array is at fault.
    #[must_use]
    pub fn pass_index(&self) -> Option<usize> {
        self.pass_index
    }

    /// The offending field.
    #[must_use]
    pub fn field(&self) -> DenoisePassField {
        self.field
    }

    /// What is wrong with it.
    #[must_use]
    pub fn issue(&self) -> &DenoisePassIssue {
        &self.issue
    }

    /// The request path this error is about, e.g. `advanced.denoisePasses[1].denoise`.
    #[must_use]
    pub fn path(&self) -> String {
        match (self.pass_index, self.field) {
            (None, DenoisePassField::DenoisePasses) => "advanced.denoisePasses".to_string(),
            (None, field) => format!("advanced.denoisePasses.{field}"),
            (Some(i), DenoisePassField::DenoisePasses) => format!("advanced.denoisePasses[{i}]"),
            (Some(i), field) => format!("advanced.denoisePasses[{i}].{field}"),
        }
    }

    /// Whether this is a **capability gap** (the model/backend cannot do it) rather than a malformed
    /// value. Capability gaps map to the typed [`Error::Unsupported`] so a consumer can distinguish
    /// them (F-008); everything else maps to [`Error::Msg`].
    #[must_use]
    pub fn is_capability_gap(&self) -> bool {
        matches!(
            self.issue,
            DenoisePassIssue::Unsupported | DenoisePassIssue::UnknownId { .. }
        )
    }

    /// Lift into the crate error type, prefixed with the model descriptor `id` the way the rest of
    /// the shared request floor does.
    #[must_use]
    pub fn into_gen_error(self, id: &str) -> Error {
        let message = format!("{id}: {self}");
        if self.is_capability_gap() {
            Error::Unsupported(message)
        } else {
            Error::Msg(message)
        }
    }
}

impl fmt::Display for DenoisePassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path(), self.issue)
    }
}

impl std::error::Error for DenoisePassError {}

impl From<DenoisePassError> for Error {
    fn from(err: DenoisePassError) -> Self {
        let message = err.to_string();
        if err.is_capability_gap() {
            Error::Unsupported(message)
        } else {
            Error::Msg(message)
        }
    }
}

/// Shorthand for a fallible denoise-pass contract operation.
pub type DenoisePassResult<T> = std::result::Result<T, DenoisePassError>;

// =================================================================================================
// Validation context
// =================================================================================================

/// What a denoise-pass list is validated **against**.
///
/// Built from a model's [`Capabilities`](crate::Capabilities) via
/// [`Capabilities::denoise_pass_context`](crate::Capabilities::denoise_pass_context) at the request
/// boundary, or via [`registry_only`](Self::registry_only) when there is no model in hand (a
/// SceneWorks-side or tooling pre-check, which can still catch every structural and range error and
/// any id outside the curated registry).
#[derive(Clone, Copy, Debug)]
pub struct DenoisePassContext<'a> {
    /// Whether the model advertises chained denoise passes at all. `false` ⇒ every non-empty list is
    /// [`DenoisePassIssue::Unsupported`].
    pub supports_denoise_passes: bool,
    /// The model's advertised sampler menu. `None` ⇒ validate against the curated [`Solver`]
    /// registry instead (model menus legitimately carry native ids beyond the curated set, so a
    /// menu, when present, is authoritative).
    pub samplers: Option<&'a [&'a str]>,
    /// The model's advertised scheduler menu; `None` ⇒ the curated [`Scheduler`] registry.
    pub schedulers: Option<&'a [&'a str]>,
    /// How many adapters the load spec provisioned, if known. `None` ⇒ the per-pass adapter index
    /// bound is not checked here (the model re-checks it when it resolves the plan against its real
    /// stack).
    pub loaded_adapters: Option<usize>,
    /// Upper sanity cap on a single pass's `steps`.
    pub max_steps: u32,
}

impl DenoisePassContext<'_> {
    /// A model-free context: structure, ranges, and ids against the **curated** [`Solver`] /
    /// [`Scheduler`] registries.
    #[must_use]
    pub fn registry_only() -> Self {
        Self {
            supports_denoise_passes: true,
            samplers: None,
            schedulers: None,
            loaded_adapters: None,
            max_steps: crate::generator::MAX_STEPS,
        }
    }

    /// This context with the load-time adapter count filled in.
    #[must_use]
    pub fn with_loaded_adapters(mut self, loaded: usize) -> Self {
        self.loaded_adapters = Some(loaded);
        self
    }
}

/// Every canonical sampler id the curated registry knows.
#[must_use]
pub fn curated_sampler_ids() -> Vec<&'static str> {
    Solver::ALL.iter().map(|s| s.name()).collect()
}

/// Every canonical scheduler id the curated registry knows.
#[must_use]
pub fn curated_scheduler_ids() -> Vec<&'static str> {
    Scheduler::ALL.iter().map(|s| s.name()).collect()
}

// =================================================================================================
// Validation
// =================================================================================================

/// Validate a denoise-pass list **before execution**.
///
/// `phases_present` is whether the same request also set
/// [`GenerationRequest::phases`](crate::GenerationRequest::phases) — the mutual-exclusion check.
///
/// Checks, in order, stopping at the first failure: mutual exclusion → model capability → arity
/// (empty / [`MAX_DENOISE_PASSES`]) → per pass, in index order: `steps` positive and under the cap,
/// `denoise` finite and in `(0, 1]`, `guidance` finite, `sampler` and `scheduler` known, adapter
/// indices in range and not duplicated, adapter weights finite.
///
/// Every error names the pass index and the field ([`DenoisePassError`]).
pub fn validate_denoise_passes(
    passes: &[DenoisePass],
    phases_present: bool,
    ctx: &DenoisePassContext<'_>,
) -> DenoisePassResult<()> {
    if phases_present {
        return Err(DenoisePassError::whole(
            DenoisePassField::DenoisePasses,
            DenoisePassIssue::MutuallyExclusiveWithPhases,
        ));
    }
    if !ctx.supports_denoise_passes {
        return Err(DenoisePassError::whole(
            DenoisePassField::DenoisePasses,
            DenoisePassIssue::Unsupported,
        ));
    }
    if passes.is_empty() {
        return Err(DenoisePassError::whole(
            DenoisePassField::DenoisePasses,
            DenoisePassIssue::Empty,
        ));
    }
    if passes.len() > MAX_DENOISE_PASSES {
        return Err(DenoisePassError::whole(
            DenoisePassField::DenoisePasses,
            DenoisePassIssue::TooMany {
                count: passes.len(),
                max: MAX_DENOISE_PASSES,
            },
        ));
    }
    for (index, pass) in passes.iter().enumerate() {
        validate_one_pass(index, pass, ctx)?;
    }
    Ok(())
}

fn validate_one_pass(
    index: usize,
    pass: &DenoisePass,
    ctx: &DenoisePassContext<'_>,
) -> DenoisePassResult<()> {
    if let Some(steps) = pass.steps {
        if steps == 0 {
            return Err(DenoisePassError::at(
                index,
                DenoisePassField::Steps,
                DenoisePassIssue::NotPositive,
            ));
        }
        if steps > ctx.max_steps {
            return Err(DenoisePassError::at(
                index,
                DenoisePassField::Steps,
                DenoisePassIssue::AboveMax {
                    value: u64::from(steps),
                    max: u64::from(ctx.max_steps),
                },
            ));
        }
    }
    if let Some(denoise) = pass.denoise {
        if !denoise.is_finite() {
            return Err(DenoisePassError::at(
                index,
                DenoisePassField::Denoise,
                DenoisePassIssue::NotFinite { value: denoise },
            ));
        }
        // `(0, 1]` — a zero-denoise pass is a full model run that changes nothing, and > 1 has no
        // meaning against a schedule normalized to its own top.
        if denoise <= 0.0 || denoise > 1.0 {
            return Err(DenoisePassError::at(
                index,
                DenoisePassField::Denoise,
                DenoisePassIssue::OutOfRange {
                    value: denoise,
                    min: 0.0,
                    max: 1.0,
                },
            ));
        }
    }
    if let Some(guidance) = pass.guidance {
        if !guidance.is_finite() {
            return Err(DenoisePassError::at(
                index,
                DenoisePassField::Guidance,
                DenoisePassIssue::NotFinite { value: guidance },
            ));
        }
    }
    if let Some(sampler) = &pass.sampler {
        check_id(
            index,
            DenoisePassField::Sampler,
            sampler,
            ctx.samplers,
            Solver::from_name(sampler).is_some(),
            curated_sampler_ids,
        )?;
    }
    if let Some(scheduler) = &pass.scheduler {
        check_id(
            index,
            DenoisePassField::Scheduler,
            scheduler,
            ctx.schedulers,
            Scheduler::from_name(scheduler).is_some(),
            curated_scheduler_ids,
        )?;
    }
    let mut seen: Vec<usize> = Vec::with_capacity(pass.adapters.len());
    for entry in &pass.adapters {
        if let Some(loaded) = ctx.loaded_adapters {
            if entry.adapter >= loaded {
                return Err(DenoisePassError::at(
                    index,
                    DenoisePassField::AdapterIndex,
                    DenoisePassIssue::AdapterIndexOutOfRange {
                        index: entry.adapter,
                        loaded,
                    },
                ));
            }
        }
        if seen.contains(&entry.adapter) {
            return Err(DenoisePassError::at(
                index,
                DenoisePassField::Adapters,
                DenoisePassIssue::DuplicateAdapter {
                    index: entry.adapter,
                },
            ));
        }
        seen.push(entry.adapter);
        if let Some(weight) = entry.weight {
            if !weight.is_finite() {
                return Err(DenoisePassError::at(
                    index,
                    DenoisePassField::AdapterWeight,
                    DenoisePassIssue::NotFinite { value: weight },
                ));
            }
        }
    }
    Ok(())
}

fn check_id(
    index: usize,
    field: DenoisePassField,
    id: &str,
    menu: Option<&[&str]>,
    in_curated_registry: bool,
    curated: fn() -> Vec<&'static str>,
) -> DenoisePassResult<()> {
    let ok = match menu {
        Some(menu) => menu.contains(&id),
        None => in_curated_registry,
    };
    if ok {
        return Ok(());
    }
    let supported: Vec<String> = match menu {
        Some(menu) => menu.iter().map(|s| (*s).to_string()).collect(),
        None => curated().iter().map(|s| (*s).to_string()).collect(),
    };
    Err(DenoisePassError::at(
        index,
        field,
        DenoisePassIssue::UnknownId {
            id: id.to_string(),
            supported,
        },
    ))
}

// =================================================================================================
// Resolution
// =================================================================================================

/// The model's own fallbacks, for the last rung of the `pass → request → model` resolution ladder.
///
/// The model supplies these because only it knows them; the resolver's job is to make them
/// **explicit in the plan** so nothing downstream re-derives a default.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoiseDefaults {
    /// The model's default step count (used when neither the pass nor the request names one).
    pub steps: u32,
    /// The model's default canonical sampler id.
    pub sampler: String,
    /// The model's default canonical scheduler id.
    pub scheduler: String,
    /// The model's default guidance, or `None` for a model with no guidance axis.
    pub guidance: Option<f32>,
}

impl DenoiseDefaults {
    /// The usual constructor.
    #[must_use]
    pub fn new(steps: u32, sampler: impl Into<String>, scheduler: impl Into<String>) -> Self {
        Self {
            steps,
            sampler: sampler.into(),
            scheduler: scheduler.into(),
            guidance: None,
        }
    }

    /// These defaults with a default guidance.
    #[must_use]
    pub fn with_guidance(mut self, guidance: f32) -> Self {
        self.guidance = Some(guidance);
        self
    }
}

/// One fully-resolved pass: no `Option` that could still be filled in from somewhere else, and the
/// seed already derived. This is what execution (sc-20418) and the replay metadata consume.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDenoisePass {
    /// Zero-based position in the chain — also the pass-seed index.
    pub index: usize,
    /// Steps this pass runs on its own schedule.
    pub steps: u32,
    /// Canonical sampler id.
    pub sampler: String,
    /// Canonical scheduler id.
    pub scheduler: String,
    /// Entry point into this pass's schedule, in `(0, 1]`.
    pub denoise: f32,
    /// Guidance for this pass. Still `Option` after resolution because a model with no guidance axis
    /// genuinely has none — `None` means "this model's implicit no-guidance path", not "look it up".
    pub guidance: Option<f32>,
    /// The deterministic per-pass seed ([`denoise_pass_seed`]), recorded for exact replay. Written to
    /// JSON as a decimal **string** — it is full-range `u64` and a bare number would be truncated by
    /// any JavaScript reader.
    pub seed: u64,
    /// Pass-local adapter weight overrides, verbatim from the request (an empty list leaves the
    /// load-time stack at its load-time scales — see [`DenoisePass::adapters`]).
    pub adapters: Vec<PhaseAdapter>,
}

/// The explicit, ordered plan a request resolves to.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDenoisePlan {
    /// The passes, in execution order. Never empty.
    pub passes: Vec<ResolvedDenoisePass>,
    /// The job seed every pass seed was derived from — recorded so a plan is self-describing for
    /// replay without also carrying the originating request. Written to JSON as a decimal **string**,
    /// like every seed in the plan.
    pub job_seed: u64,
}

impl ResolvedDenoisePlan {
    /// How many passes the plan runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Always `false` — a resolved plan has at least one pass. Present because clippy requires it
    /// alongside [`len`](Self::len).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Whether this plan is the ordinary single-pass render (the legacy shape).
    #[must_use]
    pub fn is_single_pass(&self) -> bool {
        self.passes.len() == 1
    }

    /// The total step budget across every pass.
    #[must_use]
    pub fn total_steps(&self) -> u64 {
        self.passes.iter().map(|p| u64::from(p.steps)).sum()
    }
}

/// Resolve a request into an explicit [`ResolvedDenoisePlan`].
///
/// `job_seed` is the request's **already-resolved** seed — callers do
/// `req.seed.unwrap_or_else(gen_core::default_seed)` **once** per generation, because
/// [`default_seed`](crate::default_seed) is time-derived and would otherwise differ between the plan
/// and the render (F-089).
///
/// A request with no [`denoise_passes`](crate::GenerationRequest::denoise_passes) resolves to a
/// **one-pass** plan built from the top-level request — the legacy path and the chained path share
/// this one function, so they cannot drift.
///
/// Validation runs first; an invalid request never yields a plan.
pub fn resolve_denoise_plan(
    req: &crate::GenerationRequest,
    job_seed: u64,
    defaults: &DenoiseDefaults,
    ctx: &DenoisePassContext<'_>,
) -> DenoisePassResult<ResolvedDenoisePlan> {
    let implicit = [DenoisePass::default()];
    let passes: &[DenoisePass] = match &req.denoise_passes {
        Some(passes) => {
            validate_denoise_passes(passes, req.phases.is_some(), ctx)?;
            passes
        }
        // The legacy single-trajectory request IS a one-pass chain whose every field inherits.
        None => &implicit,
    };
    let resolved = passes
        .iter()
        .enumerate()
        .map(|(index, pass)| ResolvedDenoisePass {
            index,
            steps: pass.steps.or(req.steps).unwrap_or(defaults.steps),
            sampler: pass
                .sampler
                .clone()
                .or_else(|| req.sampler.clone())
                .unwrap_or_else(|| defaults.sampler.clone()),
            scheduler: pass
                .scheduler
                .clone()
                .or_else(|| req.scheduler.clone())
                .unwrap_or_else(|| defaults.scheduler.clone()),
            denoise: pass.denoise.unwrap_or(DEFAULT_PASS_DENOISE),
            guidance: pass.guidance.or(req.guidance).or(defaults.guidance),
            seed: denoise_pass_seed(job_seed, index),
            adapters: pass.adapters.clone(),
        })
        .collect();
    Ok(ResolvedDenoisePlan {
        passes: resolved,
        job_seed,
    })
}

// =================================================================================================
// JSON (the cross-language wire format)
// =================================================================================================

const PASS_KEYS: [&str; 6] = [
    "steps",
    "sampler",
    "scheduler",
    "denoise",
    "guidance",
    "adapters",
];
const ADAPTER_KEYS: [&str; 2] = ["adapter", "weight"];

fn malformed(index: Option<usize>, field: DenoisePassField, detail: String) -> DenoisePassError {
    match index {
        Some(i) => DenoisePassError::at(i, field, DenoisePassIssue::Malformed { detail }),
        None => DenoisePassError::whole(field, DenoisePassIssue::Malformed { detail }),
    }
}

/// The shortest decimal that round-trips this `f32`, as a JSON number.
///
/// Emitting `f64::from(0.2f32)` would write `0.20000000298023224` — valid JSON, but it turns a
/// human-authored fixture into noise and makes byte-level round-trip comparison impossible. `{:?}`
/// on an `f32` is Rust's shortest round-tripping form (`0.2`, `1.0`, `3.4028235e38`). Non-finite
/// values cannot be represented in JSON at all; they are rejected by validation long before they
/// reach here, and encode as `null` if they somehow do.
fn json_f32(value: f32) -> Value {
    serde_json::from_str::<Value>(&format!("{value:?}")).unwrap_or(Value::Null)
}

/// A seed as a **decimal string** — the resolved plan's wire form for `seed` and `jobSeed`.
///
/// Seeds are full-range `u64`. [`default_seed`](crate::default_seed) is nanoseconds since the epoch
/// (~1.7e18 today) and [`denoise_pass_seed`] finishes with splitmix64, so a derived pass seed is
/// uniform over the whole `u64` range and exceeds 2^53 in ~99.9% of cases. A bare JSON number above
/// 2^53 does not survive a JavaScript `JSON.parse`: it is rounded to the nearest `f64` **silently**,
/// which replays as a *different image* with no error anywhere. The plan is a cross-language
/// artifact, so its seeds are written as strings, where every reader is exact.
fn json_seed(value: u64) -> Value {
    Value::String(value.to_string())
}

/// Read a plan seed written by [`json_seed`].
///
/// Deliberately **string-only**: a bare JSON number here is exactly the shape a JavaScript producer
/// emits after `JSON.parse` has already truncated it, so accepting it would turn a corrupted seed
/// into a plausible-looking plan. Refusing it is the same policy `contractVersion` enforces — fail
/// loudly rather than replay a different image.
fn seed_from_json(index: Option<usize>, key: &str, value: &Value) -> DenoisePassResult<u64> {
    let Some(text) = value.as_str() else {
        return Err(malformed(
            index,
            DenoisePassField::DenoisePasses,
            format!(
                "`{key}` must be a u64 written as a decimal STRING (got {}) — a bare JSON number \
                 loses seeds above 2^53 in every JavaScript reader",
                json_type_name(value)
            ),
        ));
    };
    text.parse::<u64>().map_err(|err| {
        malformed(
            index,
            DenoisePassField::DenoisePasses,
            format!("`{key}` string {text:?} is not a decimal u64: {err}"),
        )
    })
}

/// Read `advanced.denoisePasses` out of an `advanced` object.
///
/// A missing key, or an explicit `null`, is `Ok(None)` — the ordinary request with no chained passes.
///
/// A **null `advanced` block itself** is `Ok(None)` for the same reason: `advanced` is optional on
/// the request, and a producer that serializes an absent optional as `"advanced": null` (rather than
/// omitting the key) is describing the same request as one that omits it. Rejecting that shape would
/// make this contract retroactively invalidate payloads written before it existed. Any other
/// non-object — an array, a number, a string — is still malformed: those are structurally wrong,
/// not merely absent.
pub fn denoise_passes_from_advanced_json(
    advanced: &Value,
) -> DenoisePassResult<Option<Vec<DenoisePass>>> {
    if advanced.is_null() {
        return Ok(None);
    }
    let Some(object) = advanced.as_object() else {
        return Err(malformed(
            None,
            DenoisePassField::DenoisePasses,
            format!(
                "`advanced` must be a JSON object (got {})",
                json_type_name(advanced)
            ),
        ));
    };
    match object.get("denoisePasses") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => denoise_passes_from_json(value).map(Some),
    }
}

/// Decode a `denoisePasses` JSON **array** into typed passes.
///
/// Strict by design: an unknown key, a wrong JSON type, or a non-integer `steps`/`adapter` is a
/// [`DenoisePassIssue::Malformed`] naming the pass index and field. Strictness is what makes the
/// round-trip lossless — a tolerated unknown key would be silently dropped on re-encode.
///
/// An explicit `null` for an optional field is accepted and means "absent"; re-encoding omits it,
/// which is the one deliberate normalization the format performs.
pub fn denoise_passes_from_json(value: &Value) -> DenoisePassResult<Vec<DenoisePass>> {
    let Some(array) = value.as_array() else {
        return Err(malformed(
            None,
            DenoisePassField::DenoisePasses,
            format!(
                "`denoisePasses` must be a JSON array (got {})",
                json_type_name(value)
            ),
        ));
    };
    array
        .iter()
        .enumerate()
        .map(|(index, item)| pass_from_json(index, item))
        .collect()
}

fn pass_from_json(index: usize, value: &Value) -> DenoisePassResult<DenoisePass> {
    let Some(object) = value.as_object() else {
        return Err(malformed(
            Some(index),
            DenoisePassField::DenoisePasses,
            format!(
                "a pass must be a JSON object (got {})",
                json_type_name(value)
            ),
        ));
    };
    reject_unknown_keys(index, DenoisePassField::DenoisePasses, object, &PASS_KEYS)?;
    Ok(DenoisePass {
        steps: opt_u32(index, DenoisePassField::Steps, object.get("steps"))?,
        sampler: opt_string(index, DenoisePassField::Sampler, object.get("sampler"))?,
        scheduler: opt_string(index, DenoisePassField::Scheduler, object.get("scheduler"))?,
        denoise: opt_f32(index, DenoisePassField::Denoise, object.get("denoise"))?,
        guidance: opt_f32(index, DenoisePassField::Guidance, object.get("guidance"))?,
        adapters: adapters_from_json(index, object.get("adapters"))?,
    })
}

fn adapters_from_json(index: usize, value: Option<&Value>) -> DenoisePassResult<Vec<PhaseAdapter>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(array) = value.as_array() else {
        return Err(malformed(
            Some(index),
            DenoisePassField::Adapters,
            format!(
                "`adapters` must be a JSON array (got {})",
                json_type_name(value)
            ),
        ));
    };
    array
        .iter()
        .map(|item| {
            let Some(object) = item.as_object() else {
                return Err(malformed(
                    Some(index),
                    DenoisePassField::Adapters,
                    format!(
                        "an adapter override must be a JSON object (got {})",
                        json_type_name(item)
                    ),
                ));
            };
            reject_unknown_keys(index, DenoisePassField::Adapters, object, &ADAPTER_KEYS)?;
            let adapter = object.get("adapter").ok_or_else(|| {
                malformed(
                    Some(index),
                    DenoisePassField::AdapterIndex,
                    "an adapter override must name an `adapter` index".to_string(),
                )
            })?;
            let adapter = adapter.as_u64().ok_or_else(|| {
                malformed(
                    Some(index),
                    DenoisePassField::AdapterIndex,
                    format!(
                        "`adapter` must be a non-negative integer (got {})",
                        json_type_name(adapter)
                    ),
                )
            })?;
            let adapter = usize::try_from(adapter).map_err(|_| {
                malformed(
                    Some(index),
                    DenoisePassField::AdapterIndex,
                    format!("`adapter` index {adapter} does not fit this platform's usize"),
                )
            })?;
            Ok(PhaseAdapter {
                adapter,
                weight: opt_f32(index, DenoisePassField::AdapterWeight, object.get("weight"))?,
            })
        })
        .collect()
}

fn reject_unknown_keys(
    index: usize,
    field: DenoisePassField,
    object: &Map<String, Value>,
    allowed: &[&str],
) -> DenoisePassResult<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(malformed(
                Some(index),
                field,
                format!("unknown field {key:?} (known fields: {allowed:?})"),
            ));
        }
    }
    Ok(())
}

fn opt_u32(
    index: usize,
    field: DenoisePassField,
    value: Option<&Value>,
) -> DenoisePassResult<Option<u32>> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    let raw = value.as_u64().ok_or_else(|| {
        malformed(
            Some(index),
            field,
            format!(
                "`{field}` must be a non-negative integer (got {})",
                json_type_name(value)
            ),
        )
    })?;
    u32::try_from(raw).map(Some).map_err(|_| {
        malformed(
            Some(index),
            field,
            format!("`{field}` value {raw} does not fit in a u32"),
        )
    })
}

fn opt_f32(
    index: usize,
    field: DenoisePassField,
    value: Option<&Value>,
) -> DenoisePassResult<Option<f32>> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    let raw = value.as_f64().ok_or_else(|| {
        malformed(
            Some(index),
            field,
            format!(
                "`{field}` must be a JSON number (got {})",
                json_type_name(value)
            ),
        )
    })?;
    Ok(Some(raw as f32))
}

fn opt_string(
    index: usize,
    field: DenoisePassField,
    value: Option<&Value>,
) -> DenoisePassResult<Option<String>> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    value.as_str().map(|s| Some(s.to_string())).ok_or_else(|| {
        malformed(
            Some(index),
            field,
            format!(
                "`{field}` must be a JSON string (got {})",
                json_type_name(value)
            ),
        )
    })
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Encode typed passes back to the canonical `denoisePasses` JSON array.
///
/// Absent optionals are **omitted** rather than written as `null`, so
/// `denoise_passes_from_json(denoise_passes_to_json(x)) == x` for every value this crate produces.
#[must_use]
pub fn denoise_passes_to_json(passes: &[DenoisePass]) -> Value {
    Value::Array(passes.iter().map(pass_to_json).collect())
}

fn pass_to_json(pass: &DenoisePass) -> Value {
    let mut object = Map::new();
    if let Some(steps) = pass.steps {
        object.insert("steps".to_string(), Value::from(steps));
    }
    if let Some(sampler) = &pass.sampler {
        object.insert("sampler".to_string(), Value::from(sampler.as_str()));
    }
    if let Some(scheduler) = &pass.scheduler {
        object.insert("scheduler".to_string(), Value::from(scheduler.as_str()));
    }
    if let Some(denoise) = pass.denoise {
        object.insert("denoise".to_string(), json_f32(denoise));
    }
    if let Some(guidance) = pass.guidance {
        object.insert("guidance".to_string(), json_f32(guidance));
    }
    if !pass.adapters.is_empty() {
        object.insert(
            "adapters".to_string(),
            Value::Array(pass.adapters.iter().map(adapter_to_json).collect()),
        );
    }
    Value::Object(object)
}

fn adapter_to_json(adapter: &PhaseAdapter) -> Value {
    let mut object = Map::new();
    object.insert("adapter".to_string(), Value::from(adapter.adapter));
    if let Some(weight) = adapter.weight {
        object.insert("weight".to_string(), json_f32(weight));
    }
    Value::Object(object)
}

impl ResolvedDenoisePlan {
    /// Encode the plan as JSON — the replay/metadata form SceneWorks persists.
    ///
    /// `jobSeed` and each pass's `seed` are written as decimal **strings**, not JSON numbers. Seeds
    /// are full-range `u64` (`denoise_pass_seed` finishes with splitmix64, so a derived pass seed
    /// exceeds 2^53 in ~99.9% of cases) and a bare JSON number above 2^53 is silently rounded by
    /// JavaScript's `JSON.parse` — which replays as a *different image* with no error raised
    /// anywhere. [`from_json`](Self::from_json) is string-only on the way back for the same reason.
    /// Every other field is its natural JSON type.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let passes: Vec<Value> = self
            .passes
            .iter()
            .map(|pass| {
                let mut object = Map::new();
                object.insert("index".to_string(), Value::from(pass.index));
                object.insert("steps".to_string(), Value::from(pass.steps));
                object.insert("sampler".to_string(), Value::from(pass.sampler.as_str()));
                object.insert(
                    "scheduler".to_string(),
                    Value::from(pass.scheduler.as_str()),
                );
                object.insert("denoise".to_string(), json_f32(pass.denoise));
                if let Some(guidance) = pass.guidance {
                    object.insert("guidance".to_string(), json_f32(guidance));
                }
                object.insert("seed".to_string(), json_seed(pass.seed));
                object.insert(
                    "adapters".to_string(),
                    Value::Array(pass.adapters.iter().map(adapter_to_json).collect()),
                );
                Value::Object(object)
            })
            .collect();
        let mut root = Map::new();
        root.insert(
            "contractVersion".to_string(),
            Value::from(DENOISE_PASS_CONTRACT_VERSION),
        );
        root.insert("jobSeed".to_string(), json_seed(self.job_seed));
        root.insert("passes".to_string(), Value::Array(passes));
        Value::Object(root)
    }

    /// Decode a plan previously written by [`to_json`](Self::to_json).
    pub fn from_json(value: &Value) -> DenoisePassResult<Self> {
        let Some(object) = value.as_object() else {
            return Err(malformed(
                None,
                DenoisePassField::DenoisePasses,
                format!(
                    "a plan must be a JSON object (got {})",
                    json_type_name(value)
                ),
            ));
        };
        // Version first: a plan from a build this one does not implement must be refused before any
        // of its fields are interpreted, because the whole risk is interpreting them wrongly.
        let version = object
            .get("contractVersion")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                malformed(
                    None,
                    DenoisePassField::DenoisePasses,
                    "a plan must carry an integer `contractVersion`".to_string(),
                )
            })?;
        if version != u64::from(DENOISE_PASS_CONTRACT_VERSION) {
            return Err(malformed(
                None,
                DenoisePassField::DenoisePasses,
                format!(
                    "plan `contractVersion` {version} is not implemented by this build (expected \
                     {DENOISE_PASS_CONTRACT_VERSION})"
                ),
            ));
        }
        let job_seed = seed_from_json(
            None,
            "jobSeed",
            object.get("jobSeed").ok_or_else(|| {
                malformed(
                    None,
                    DenoisePassField::DenoisePasses,
                    "a plan must carry a `jobSeed`".to_string(),
                )
            })?,
        )?;
        let Some(array) = object.get("passes").and_then(Value::as_array) else {
            return Err(malformed(
                None,
                DenoisePassField::DenoisePasses,
                "a plan must carry a `passes` array".to_string(),
            ));
        };
        let passes = array
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let Some(object) = item.as_object() else {
                    return Err(malformed(
                        Some(index),
                        DenoisePassField::DenoisePasses,
                        format!(
                            "a resolved pass must be a JSON object (got {})",
                            json_type_name(item)
                        ),
                    ));
                };
                let need = |key: &str| {
                    object.get(key).ok_or_else(|| {
                        malformed(
                            Some(index),
                            DenoisePassField::DenoisePasses,
                            format!("a resolved pass must carry `{key}`"),
                        )
                    })
                };
                let steps = opt_u32(index, DenoisePassField::Steps, Some(need("steps")?))?
                    .expect("a present `steps` decodes to Some");
                let sampler = opt_string(index, DenoisePassField::Sampler, Some(need("sampler")?))?
                    .expect("a present `sampler` decodes to Some");
                let scheduler =
                    opt_string(index, DenoisePassField::Scheduler, Some(need("scheduler")?))?
                        .expect("a present `scheduler` decodes to Some");
                let denoise = opt_f32(index, DenoisePassField::Denoise, Some(need("denoise")?))?
                    .expect("a present `denoise` decodes to Some");
                let seed = seed_from_json(Some(index), "seed", need("seed")?)?;
                Ok(ResolvedDenoisePass {
                    index,
                    steps,
                    sampler,
                    scheduler,
                    denoise,
                    guidance: opt_f32(index, DenoisePassField::Guidance, object.get("guidance"))?,
                    seed,
                    adapters: adapters_from_json(index, object.get("adapters"))?,
                })
            })
            .collect::<DenoisePassResult<Vec<_>>>()?;
        Ok(Self { passes, job_seed })
    }
}

/// Compare two JSON values, allowing numbers to differ by at most `tolerance`.
///
/// Exists because the contract's floats are `f32` and a fixture is authored in decimal: an exact
/// `Value` comparison would make an f32→f64 widening (`0.2` → `0.20000000298023224`) look like a
/// contract break. Structure, key sets and string values are compared **exactly** — only numbers are
/// given slack.
#[must_use]
pub fn json_equal_within(left: &Value, right: &Value, tolerance: f64) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() <= tolerance,
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(a, b)| json_equal_within(a, b, tolerance))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, a)| {
                    b.get(key)
                        .is_some_and(|b| json_equal_within(a, b, tolerance))
                })
        }
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{Capabilities, GenerationPhase, GenerationRequest};

    fn defaults() -> DenoiseDefaults {
        DenoiseDefaults::new(20, "euler", "normal").with_guidance(3.5)
    }

    fn ctx() -> DenoisePassContext<'static> {
        DenoisePassContext::registry_only()
    }

    fn pass(steps: u32, sampler: &str, scheduler: &str, denoise: f32) -> DenoisePass {
        DenoisePass {
            steps: Some(steps),
            sampler: Some(sampler.to_string()),
            scheduler: Some(scheduler.to_string()),
            denoise: Some(denoise),
            ..Default::default()
        }
    }

    // --- seed domain separation -------------------------------------------------------------

    #[test]
    fn pass_zero_seed_is_the_job_seed() {
        for seed in [0_u64, 1, 42, u64::MAX] {
            assert_eq!(denoise_pass_seed(seed, 0), seed);
        }
    }

    #[test]
    fn later_pass_seeds_are_separated_and_deterministic() {
        let seeds: Vec<u64> = (0..8).map(|i| denoise_pass_seed(1234, i)).collect();
        // Deterministic.
        assert_eq!(
            seeds,
            (0..8)
                .map(|i| denoise_pass_seed(1234, i))
                .collect::<Vec<_>>()
        );
        // Distinct.
        for (i, a) in seeds.iter().enumerate() {
            for b in seeds.iter().skip(i + 1) {
                assert_ne!(a, b, "pass seeds must not collide: {seeds:?}");
            }
        }
        // Not a plain offset ladder (the correlation the splitmix finalizer exists to break).
        assert_ne!(
            seeds[2].wrapping_sub(seeds[1]),
            seeds[3].wrapping_sub(seeds[2])
        );
    }

    #[test]
    fn pass_seeds_differ_between_jobs() {
        assert_ne!(denoise_pass_seed(1, 1), denoise_pass_seed(2, 1));
    }

    // --- validation: pass index + field ------------------------------------------------------

    #[test]
    fn phases_and_denoise_passes_are_mutually_exclusive() {
        let err = validate_denoise_passes(&[pass(10, "euler", "normal", 1.0)], true, &ctx())
            .expect_err("both set must be rejected");
        assert_eq!(err.pass_index(), None);
        assert_eq!(err.field(), DenoisePassField::DenoisePasses);
        assert_eq!(err.issue(), &DenoisePassIssue::MutuallyExclusiveWithPhases);
        assert_eq!(err.path(), "advanced.denoisePasses");
    }

    #[test]
    fn empty_pass_array_is_rejected() {
        let err = validate_denoise_passes(&[], false, &ctx()).expect_err("empty must be rejected");
        assert_eq!(err.issue(), &DenoisePassIssue::Empty);
        assert_eq!(err.field(), DenoisePassField::DenoisePasses);
    }

    #[test]
    fn too_many_passes_are_rejected() {
        let passes = vec![pass(1, "euler", "normal", 1.0); MAX_DENOISE_PASSES + 1];
        let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("cap must bite");
        assert_eq!(
            err.issue(),
            &DenoisePassIssue::TooMany {
                count: MAX_DENOISE_PASSES + 1,
                max: MAX_DENOISE_PASSES
            }
        );
    }

    #[test]
    fn zero_steps_names_its_pass_and_field() {
        let passes = vec![
            pass(10, "euler", "normal", 1.0),
            DenoisePass {
                steps: Some(0),
                ..Default::default()
            },
        ];
        let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("steps 0 rejected");
        assert_eq!(err.pass_index(), Some(1));
        assert_eq!(err.field(), DenoisePassField::Steps);
        assert_eq!(err.issue(), &DenoisePassIssue::NotPositive);
        assert_eq!(err.path(), "advanced.denoisePasses[1].steps");
        assert!(err.to_string().contains("denoisePasses[1].steps"), "{err}");
    }

    #[test]
    fn steps_above_the_sanity_cap_are_rejected() {
        let passes = vec![DenoisePass {
            steps: Some(u32::MAX),
            ..Default::default()
        }];
        let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("cap must bite");
        assert_eq!(err.field(), DenoisePassField::Steps);
        assert!(matches!(err.issue(), DenoisePassIssue::AboveMax { .. }));
    }

    #[test]
    fn denoise_outside_the_half_open_unit_range_is_rejected() {
        for bad in [0.0_f32, -0.5, 1.5] {
            let passes = vec![
                pass(10, "euler", "normal", 1.0),
                pass(4, "euler", "normal", bad),
            ];
            let err =
                validate_denoise_passes(&passes, false, &ctx()).expect_err("denoise range bites");
            assert_eq!(err.pass_index(), Some(1), "for {bad}");
            assert_eq!(err.field(), DenoisePassField::Denoise);
            assert_eq!(
                err.issue(),
                &DenoisePassIssue::OutOfRange {
                    value: bad,
                    min: 0.0,
                    max: 1.0
                }
            );
        }
        // The closed upper bound is legal.
        assert!(validate_denoise_passes(&[pass(4, "euler", "normal", 1.0)], false, &ctx()).is_ok());
    }

    #[test]
    fn non_finite_pass_floats_are_rejected_with_their_index() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let passes = vec![
                pass(10, "euler", "normal", 1.0),
                DenoisePass {
                    guidance: Some(bad),
                    ..pass(4, "euler", "normal", 0.2)
                },
            ];
            let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("NaN/Inf bites");
            assert_eq!(err.pass_index(), Some(1));
            assert_eq!(err.field(), DenoisePassField::Guidance);
        }
        let passes = vec![DenoisePass {
            denoise: Some(f32::NAN),
            ..Default::default()
        }];
        let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("NaN denoise bites");
        assert_eq!(err.field(), DenoisePassField::Denoise);
        assert!(matches!(err.issue(), DenoisePassIssue::NotFinite { .. }));
    }

    #[test]
    fn unknown_sampler_and_scheduler_ids_are_capability_gaps() {
        let passes = vec![pass(10, "definitely_not_a_sampler", "normal", 1.0)];
        let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("unknown id bites");
        assert_eq!(err.pass_index(), Some(0));
        assert_eq!(err.field(), DenoisePassField::Sampler);
        assert!(err.is_capability_gap());
        assert!(
            matches!(Error::from(err), Error::Unsupported(_)),
            "a capability gap must stay typed (F-008)"
        );

        let passes = vec![pass(10, "euler", "definitely_not_a_scheduler", 1.0)];
        let err = validate_denoise_passes(&passes, false, &ctx()).expect_err("unknown id bites");
        assert_eq!(err.field(), DenoisePassField::Scheduler);
    }

    #[test]
    fn a_model_menu_is_authoritative_over_the_curated_registry() {
        let menu: &[&'static str] = &["euler"];
        let restricted = DenoisePassContext {
            samplers: Some(menu),
            ..DenoisePassContext::registry_only()
        };
        // `heun` is in the curated registry but not on this model's menu.
        assert!(Solver::from_name("heun").is_some());
        let err = validate_denoise_passes(&[pass(4, "heun", "normal", 1.0)], false, &restricted)
            .expect_err("off-menu sampler bites");
        assert_eq!(err.field(), DenoisePassField::Sampler);
        assert!(
            validate_denoise_passes(&[pass(4, "euler", "normal", 1.0)], false, &restricted).is_ok()
        );
    }

    #[test]
    fn a_model_that_does_not_advertise_passes_rejects_them() {
        let unsupported = DenoisePassContext {
            supports_denoise_passes: false,
            ..DenoisePassContext::registry_only()
        };
        let err = validate_denoise_passes(&[pass(10, "euler", "normal", 1.0)], false, &unsupported)
            .expect_err("capability gate bites");
        assert_eq!(err.issue(), &DenoisePassIssue::Unsupported);
        assert!(err.is_capability_gap());
    }

    #[test]
    fn adapter_refs_are_bounds_checked_and_deduplicated() {
        let with_adapters = |entries: Vec<PhaseAdapter>| DenoisePass {
            adapters: entries,
            ..pass(4, "euler", "normal", 0.5)
        };
        let bounded = ctx().with_loaded_adapters(2);
        let err = validate_denoise_passes(
            &[with_adapters(vec![PhaseAdapter {
                adapter: 5,
                weight: Some(1.0),
            }])],
            false,
            &bounded,
        )
        .expect_err("out-of-range index bites");
        assert_eq!(err.field(), DenoisePassField::AdapterIndex);
        assert_eq!(
            err.issue(),
            &DenoisePassIssue::AdapterIndexOutOfRange {
                index: 5,
                loaded: 2
            }
        );

        let err = validate_denoise_passes(
            &[with_adapters(vec![
                PhaseAdapter {
                    adapter: 0,
                    weight: Some(1.0),
                },
                PhaseAdapter {
                    adapter: 0,
                    weight: Some(0.5),
                },
            ])],
            false,
            &bounded,
        )
        .expect_err("duplicate override bites");
        assert_eq!(err.field(), DenoisePassField::Adapters);
        assert_eq!(
            err.issue(),
            &DenoisePassIssue::DuplicateAdapter { index: 0 }
        );

        // Unknown load-time stack ⇒ the index bound is the model's to check later.
        assert!(validate_denoise_passes(
            &[with_adapters(vec![PhaseAdapter {
                adapter: 5,
                weight: None
            }])],
            false,
            &ctx()
        )
        .is_ok());

        let err = validate_denoise_passes(
            &[with_adapters(vec![PhaseAdapter {
                adapter: 0,
                weight: Some(f32::NAN),
            }])],
            false,
            &bounded,
        )
        .expect_err("NaN weight bites");
        assert_eq!(err.field(), DenoisePassField::AdapterWeight);
    }

    // --- resolution --------------------------------------------------------------------------

    /// AC 3: a one-pass plan resolves equivalently to the current top-level request.
    ///
    /// The comparison is field-by-field with an explicit numeric tolerance, and it is run for a
    /// matrix of top-level requests — a fully-specified one, a bare one that falls all the way
    /// through to the model defaults, and a partially-specified one.
    #[test]
    fn single_pass_plan_matches_top_level_request() {
        const TOL: f32 = 1e-6;
        let cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                seed: Some(9_876_543_210),
                steps: Some(28),
                sampler: Some("dpmpp_2m".to_string()),
                scheduler: Some("karras".to_string()),
                guidance: Some(4.5),
                ..Default::default()
            },
            // Nothing set: every field must fall through to the model defaults.
            GenerationRequest::default(),
            // Partially set: steps + guidance from the request, sampler/scheduler from the model.
            GenerationRequest {
                steps: Some(12),
                guidance: Some(1.25),
                ..Default::default()
            },
        ];
        for legacy in cases {
            let job_seed = legacy.seed.unwrap_or(4_242);
            let implicit = resolve_denoise_plan(&legacy, job_seed, &defaults(), &ctx())
                .expect("the legacy request resolves");

            // The same request, re-expressed as an EXPLICIT one-pass chain that names exactly what
            // the top level named (and inherits exactly what it did not).
            let explicit_req = GenerationRequest {
                denoise_passes: Some(vec![DenoisePass {
                    steps: legacy.steps,
                    sampler: legacy.sampler.clone(),
                    scheduler: legacy.scheduler.clone(),
                    denoise: Some(1.0),
                    guidance: legacy.guidance,
                    adapters: Vec::new(),
                }]),
                ..legacy.clone()
            };
            let explicit = resolve_denoise_plan(&explicit_req, job_seed, &defaults(), &ctx())
                .expect("the explicit one-pass request resolves");

            assert!(implicit.is_single_pass());
            assert!(explicit.is_single_pass());
            assert_eq!(implicit.job_seed, explicit.job_seed);
            let (a, b) = (&implicit.passes[0], &explicit.passes[0]);
            assert_eq!(a.index, b.index);
            assert_eq!(a.steps, b.steps);
            assert_eq!(a.sampler, b.sampler);
            assert_eq!(a.scheduler, b.scheduler);
            assert_eq!(a.adapters, b.adapters);
            // The seed of pass 0 is the job seed itself — an existing replay payload names only it.
            assert_eq!(a.seed, job_seed);
            assert_eq!(b.seed, job_seed);
            assert!(
                (a.denoise - b.denoise).abs() <= TOL,
                "denoise {} vs {}",
                a.denoise,
                b.denoise
            );
            match (a.guidance, b.guidance) {
                (None, None) => {}
                (Some(x), Some(y)) => assert!((x - y).abs() <= TOL, "guidance {x} vs {y}"),
                (x, y) => panic!("guidance presence differs: {x:?} vs {y:?}"),
            }
            // ... and the resolved values ARE the top-level request's own, not something new.
            assert_eq!(a.steps, legacy.steps.unwrap_or(defaults().steps));
            assert_eq!(
                a.sampler,
                legacy.sampler.clone().unwrap_or(defaults().sampler)
            );
            assert_eq!(
                a.scheduler,
                legacy.scheduler.clone().unwrap_or(defaults().scheduler)
            );
            let expected_guidance = legacy.guidance.or(defaults().guidance);
            assert_eq!(a.guidance.is_some(), expected_guidance.is_some());
            if let (Some(x), Some(y)) = (a.guidance, expected_guidance) {
                assert!((x - y).abs() <= TOL);
            }
            assert!((a.denoise - DEFAULT_PASS_DENOISE).abs() <= TOL);
        }
    }

    #[test]
    fn resolution_ladder_is_pass_then_request_then_model() {
        let req = GenerationRequest {
            steps: Some(30),
            sampler: Some("heun".to_string()),
            scheduler: Some("karras".to_string()),
            guidance: Some(7.0),
            denoise_passes: Some(vec![
                // Pass 0 names nothing → the request's own values.
                DenoisePass::default(),
                // Pass 1 overrides everything.
                DenoisePass {
                    steps: Some(4),
                    sampler: Some("ddim".to_string()),
                    scheduler: Some("simple".to_string()),
                    denoise: Some(0.2),
                    guidance: Some(1.0),
                    adapters: vec![PhaseAdapter {
                        adapter: 0,
                        weight: Some(0.8),
                    }],
                },
            ]),
            ..Default::default()
        };
        let plan = resolve_denoise_plan(&req, 7, &defaults(), &ctx()).expect("resolves");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.total_steps(), 34);
        assert_eq!(plan.passes[0].steps, 30);
        assert_eq!(plan.passes[0].sampler, "heun");
        assert_eq!(plan.passes[0].scheduler, "karras");
        assert_eq!(plan.passes[0].guidance, Some(7.0));
        assert_eq!(plan.passes[0].denoise, 1.0);
        assert_eq!(plan.passes[0].seed, 7);
        assert_eq!(plan.passes[1].steps, 4);
        assert_eq!(plan.passes[1].sampler, "ddim");
        assert_eq!(plan.passes[1].scheduler, "simple");
        assert_eq!(plan.passes[1].guidance, Some(1.0));
        assert_eq!(plan.passes[1].denoise, 0.2);
        assert_eq!(plan.passes[1].seed, denoise_pass_seed(7, 1));
        assert_ne!(plan.passes[1].seed, 7);

        // The model default is the LAST rung: with a bare request, every pass gets it.
        let bare = GenerationRequest {
            denoise_passes: Some(vec![DenoisePass::default(), DenoisePass::default()]),
            ..Default::default()
        };
        let plan = resolve_denoise_plan(&bare, 7, &defaults(), &ctx()).expect("resolves");
        for pass in &plan.passes {
            assert_eq!(pass.steps, defaults().steps);
            assert_eq!(pass.sampler, defaults().sampler);
            assert_eq!(pass.scheduler, defaults().scheduler);
            assert_eq!(pass.guidance, defaults().guidance);
        }
    }

    #[test]
    fn resolution_validates_before_it_plans() {
        let req = GenerationRequest {
            denoise_passes: Some(vec![DenoisePass {
                denoise: Some(2.0),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let err = resolve_denoise_plan(&req, 1, &defaults(), &ctx()).expect_err("invalid");
        assert_eq!(err.field(), DenoisePassField::Denoise);

        let both = GenerationRequest {
            phases: Some(vec![GenerationPhase {
                steps: 4,
                ..Default::default()
            }]),
            denoise_passes: Some(vec![DenoisePass::default()]),
            ..Default::default()
        };
        let err = resolve_denoise_plan(&both, 1, &defaults(), &ctx()).expect_err("invalid");
        assert_eq!(err.issue(), &DenoisePassIssue::MutuallyExclusiveWithPhases);
    }

    // --- capabilities wiring ------------------------------------------------------------------

    #[test]
    fn capabilities_supply_the_validation_context() {
        let caps = Capabilities {
            supports_denoise_passes: true,
            samplers: vec!["euler", "ddim"],
            schedulers: vec!["normal"],
            ..Default::default()
        };
        let ctx = caps.denoise_pass_context(Some(1));
        assert!(ctx.supports_denoise_passes);
        assert_eq!(ctx.samplers, Some(&["euler", "ddim"][..]));
        assert_eq!(ctx.schedulers, Some(&["normal"][..]));
        assert_eq!(ctx.loaded_adapters, Some(1));

        // A model with no advertised menu falls back to the curated registry rather than rejecting
        // every id.
        let bare = Capabilities {
            supports_denoise_passes: true,
            ..Default::default()
        };
        assert_eq!(bare.denoise_pass_context(None).samplers, None);
    }

    // --- JSON ---------------------------------------------------------------------------------

    #[test]
    fn json_round_trips_losslessly() {
        let passes = vec![
            pass(10, "euler", "normal", 1.0),
            DenoisePass {
                steps: Some(4),
                sampler: Some("dpmpp_2m".to_string()),
                scheduler: Some("karras".to_string()),
                denoise: Some(0.2),
                guidance: Some(2.5),
                adapters: vec![
                    PhaseAdapter {
                        adapter: 0,
                        weight: Some(0.75),
                    },
                    PhaseAdapter {
                        adapter: 1,
                        weight: None,
                    },
                ],
            },
            DenoisePass::default(),
        ];
        let json = denoise_passes_to_json(&passes);
        assert_eq!(denoise_passes_from_json(&json).expect("decodes"), passes);
        // ... and the encoding is canonical: encoding the decode reproduces the same bytes.
        assert_eq!(
            serde_json::to_string(&denoise_passes_to_json(
                &denoise_passes_from_json(&json).expect("decodes")
            ))
            .expect("encodes"),
            serde_json::to_string(&json).expect("encodes")
        );
    }

    #[test]
    fn f32_values_encode_as_their_shortest_decimal() {
        let json = denoise_passes_to_json(&[DenoisePass {
            denoise: Some(0.2),
            guidance: Some(1.0),
            ..Default::default()
        }]);
        let text = serde_json::to_string(&json).expect("encodes");
        assert!(text.contains("\"denoise\":0.2"), "{text}");
        assert!(text.contains("\"guidance\":1.0"), "{text}");
    }

    #[test]
    fn explicit_nulls_decode_as_absent() {
        let json = serde_json::json!([{
            "steps": 10,
            "sampler": null,
            "scheduler": null,
            "denoise": null,
            "guidance": null,
            "adapters": null,
        }]);
        assert_eq!(
            denoise_passes_from_json(&json).expect("decodes"),
            vec![DenoisePass {
                steps: Some(10),
                ..Default::default()
            }]
        );
    }

    #[test]
    fn unknown_json_keys_are_rejected_with_their_pass_index() {
        let json = serde_json::json!([
            { "steps": 10 },
            { "steps": 4, "sampeler": "euler" },
        ]);
        let err = denoise_passes_from_json(&json).expect_err("typo must be rejected");
        assert_eq!(err.pass_index(), Some(1));
        assert!(matches!(err.issue(), DenoisePassIssue::Malformed { .. }));
        assert!(err.to_string().contains("sampeler"), "{err}");
    }

    #[test]
    fn wrong_json_types_are_rejected_with_their_field() {
        let cases: Vec<(Value, DenoisePassField)> = vec![
            (
                serde_json::json!([{ "steps": "ten" }]),
                DenoisePassField::Steps,
            ),
            (
                serde_json::json!([{ "sampler": 3 }]),
                DenoisePassField::Sampler,
            ),
            (
                serde_json::json!([{ "denoise": "half" }]),
                DenoisePassField::Denoise,
            ),
            (
                serde_json::json!([{ "adapters": [{ "adapter": -1 }] }]),
                DenoisePassField::AdapterIndex,
            ),
            (
                serde_json::json!([{ "adapters": [{ "weight": 0.5 }] }]),
                DenoisePassField::AdapterIndex,
            ),
        ];
        for (json, field) in cases {
            let err = denoise_passes_from_json(&json).expect_err("bad type must be rejected");
            assert_eq!(err.field(), field, "for {json}");
            assert_eq!(err.pass_index(), Some(0));
        }
        let err = denoise_passes_from_json(&serde_json::json!({})).expect_err("not an array");
        assert_eq!(err.pass_index(), None);
    }

    #[test]
    fn advanced_block_reader_treats_absent_and_null_alike() {
        assert_eq!(
            denoise_passes_from_advanced_json(&serde_json::json!({})).expect("decodes"),
            None
        );
        assert_eq!(
            denoise_passes_from_advanced_json(&serde_json::json!({ "denoisePasses": null }))
                .expect("decodes"),
            None
        );
        assert_eq!(
            denoise_passes_from_advanced_json(&serde_json::json!({ "denoisePasses": [] }))
                .expect("decodes"),
            Some(Vec::new())
        );
        assert!(denoise_passes_from_advanced_json(&serde_json::json!([])).is_err());
    }

    /// A null `advanced` block is the same request as an omitted one. A legacy producer that
    /// serializes its absent optional as `"advanced": null` predates this contract, so treating it as
    /// malformed would retroactively break payloads that were valid when they were written.
    /// Structurally wrong shapes stay rejected — the tolerance is for *absence*, not for anything
    /// that merely fails to be an object.
    #[test]
    fn a_null_advanced_block_is_the_absent_state_not_a_malformed_one() {
        assert_eq!(
            denoise_passes_from_advanced_json(&Value::Null).expect("a null `advanced` decodes"),
            None
        );
        for wrong in [
            serde_json::json!([]),
            serde_json::json!(0),
            serde_json::json!("advanced"),
            serde_json::json!(true),
        ] {
            let err = denoise_passes_from_advanced_json(&wrong)
                .expect_err("a non-null non-object `advanced` is malformed");
            assert_eq!(err.pass_index(), None, "for {wrong}");
            assert_eq!(err.field(), DenoisePassField::DenoisePasses, "for {wrong}");
        }
    }

    #[test]
    fn resolved_plans_round_trip_through_json() {
        let req = GenerationRequest {
            steps: Some(20),
            sampler: Some("euler".to_string()),
            scheduler: Some("normal".to_string()),
            guidance: Some(3.5),
            denoise_passes: Some(vec![
                pass(10, "euler", "normal", 1.0),
                DenoisePass {
                    adapters: vec![PhaseAdapter {
                        adapter: 0,
                        weight: Some(0.8),
                    }],
                    ..pass(4, "dpmpp_2m", "karras", 0.2)
                },
            ]),
            ..Default::default()
        };
        let plan =
            resolve_denoise_plan(&req, 1_234_567_890, &defaults(), &ctx()).expect("resolves");
        let json = plan.to_json();
        assert_eq!(
            ResolvedDenoisePlan::from_json(&json).expect("decodes"),
            plan
        );
        assert!(json_equal_within(
            &ResolvedDenoisePlan::from_json(&json)
                .expect("decodes")
                .to_json(),
            &json,
            0.0,
        ));
    }

    #[test]
    fn a_plan_from_an_unimplemented_contract_version_is_refused() {
        let plan = ResolvedDenoisePlan {
            job_seed: 5,
            passes: vec![ResolvedDenoisePass {
                index: 0,
                steps: 10,
                sampler: "euler".to_string(),
                scheduler: "normal".to_string(),
                denoise: 1.0,
                guidance: None,
                seed: 5,
                adapters: Vec::new(),
            }],
        };
        let mut json = plan.to_json();
        assert_eq!(
            json.get("contractVersion").and_then(Value::as_u64),
            Some(u64::from(DENOISE_PASS_CONTRACT_VERSION)),
            "every emitted plan is stamped"
        );
        assert_eq!(
            ResolvedDenoisePlan::from_json(&json).expect("decodes"),
            plan
        );

        // A plan from a future build must be refused, not reinterpreted.
        json["contractVersion"] = Value::from(DENOISE_PASS_CONTRACT_VERSION + 1);
        let err = ResolvedDenoisePlan::from_json(&json).expect_err("future version refused");
        assert!(err.to_string().contains("contractVersion"), "{err}");

        // ... and so must an unstamped one: it predates the guarantee, so nothing warrants assuming
        // it means what this build means.
        json.as_object_mut()
            .expect("object")
            .remove("contractVersion");
        assert!(ResolvedDenoisePlan::from_json(&json).is_err());
    }

    /// Plan seeds cross the language boundary as decimal **strings**.
    ///
    /// The other end of this contract is JavaScript, where `JSON.parse` silently rounds an integer
    /// above 2^53 to the nearest `f64`. A per-pass seed is a splitmix64 output, so it is above 2^53
    /// essentially always — the checked-in reference recipe already carries `5145724004617983535`,
    /// which a bare-number reader would hand back as `5145724004617983000` and replay as a different
    /// image with no error raised. So: emit strings, and refuse a bare number on read rather than
    /// accept the exact shape a truncating producer emits.
    #[test]
    fn plan_seeds_are_json_strings_so_they_survive_a_javascript_reader() {
        // The real pass-1 seed of the reference recipe — comfortably past 2^53.
        let derived = denoise_pass_seed(1_234_567_890, 1);
        assert!(
            derived > (1_u64 << 53),
            "the fixture premise: derived seeds exceed JavaScript's exact-integer range ({derived})"
        );
        let plan = ResolvedDenoisePlan {
            job_seed: u64::MAX,
            passes: vec![ResolvedDenoisePass {
                index: 0,
                steps: 10,
                sampler: "euler".to_string(),
                scheduler: "normal".to_string(),
                denoise: 1.0,
                guidance: None,
                seed: derived,
                adapters: Vec::new(),
            }],
        };
        let json = plan.to_json();
        assert_eq!(
            json.get("jobSeed").and_then(Value::as_str),
            Some(u64::MAX.to_string().as_str()),
            "jobSeed is a decimal string"
        );
        assert_eq!(
            json["passes"][0].get("seed").and_then(Value::as_str),
            Some(derived.to_string().as_str()),
            "a pass seed is a decimal string"
        );
        // Full-range round-trip: the exact bits come back, which is the whole point.
        let decoded = ResolvedDenoisePlan::from_json(&json).expect("decodes");
        assert_eq!(decoded, plan);
        assert_eq!(decoded.job_seed, u64::MAX);
        assert_eq!(decoded.passes[0].seed, derived);

        // A bare JSON number is refused on both keys — it is exactly what a truncating reader
        // re-emits, so accepting it would launder a corrupted seed into a plausible plan.
        for key in ["jobSeed", "seed"] {
            let mut broken = json.clone();
            let slot = if key == "jobSeed" {
                &mut broken["jobSeed"]
            } else {
                &mut broken["passes"][0]["seed"]
            };
            *slot = Value::from(42_u64);
            let err = ResolvedDenoisePlan::from_json(&broken)
                .expect_err("a numeric seed must be refused");
            assert!(err.to_string().contains("decimal STRING"), "{key}: {err}");
        }

        // A string that is not a decimal u64 is refused too, rather than silently becoming 0.
        for bad in ["", "-1", "18446744073709551616", "0x10", "1e9", " 7"] {
            let mut broken = json.clone();
            broken["jobSeed"] = Value::from(bad);
            assert!(
                ResolvedDenoisePlan::from_json(&broken).is_err(),
                "{bad:?} must not decode"
            );
        }
    }

    #[test]
    fn json_equal_within_gives_slack_to_numbers_only() {
        let a = serde_json::json!({ "denoise": 0.2, "sampler": "euler" });
        let b = serde_json::json!({ "denoise": 0.20000000298023224, "sampler": "euler" });
        assert!(json_equal_within(&a, &b, 1e-6));
        assert!(!json_equal_within(&a, &b, 0.0));
        let c = serde_json::json!({ "denoise": 0.2, "sampler": "heun" });
        assert!(!json_equal_within(&a, &c, 1.0));
        let d = serde_json::json!({ "denoise": 0.2 });
        assert!(!json_equal_within(&a, &d, 1.0), "key sets differ");
    }
}

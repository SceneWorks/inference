//! The scheduler function library (epic 7114, P1, sc-7116): pure `(ModelSampling, steps) -> sigma
//! array` builders — the σ-schedule half of the unified framework. Each is backend-neutral host math,
//! independent of prediction type, and faithfully ports ComfyUI's `comfy/samplers.py::calculate_sigmas`
//! plus the `comfy/k_diffusion` `get_sigmas_karras` / `get_sigmas_exponential`. The curated set from
//! epic decision 2: `normal`, `simple`, `karras`, `exponential`, `sgm_uniform`, `beta`, `ddim_uniform`.
//!
//! The sampler ([`super::unified::Sampler`]) integrates the latents down the returned schedule. Every
//! schedule is DESCENDING and ends with a trailing `0.0` terminal node (the per-builder note gives the
//! length). The analytic builders (`karras`/`exponential`) take an explicit `[σ_min, σ_max]`; the
//! model-native builders read [`ModelSampling::sigma_table`] / [`ModelSampling::timestep`] /
//! [`ModelSampling::sigma`]. The dispatcher derives `σ_min`/`σ_max` from the model's σ table (its
//! smallest positive / largest entry), matching ComfyUI (for flux the table's smallest positive entry
//! is the real `σ_min`, not 0).
//!
//! # Advanced multi-pass schedules (epic 20414, sc-20416)
//!
//! [`Scheduler::LinearQuadratic`] and [`Scheduler::BongTangent`] are the two *chained-denoise-pass*
//! schedules. They are host math like everything else here, so MLX and candle consume byte-identical
//! coefficients through the one [`schedule_sigmas`] seam. They are deliberately **not** in
//! [`Scheduler::ALL`] — that constant is the menu every adopted engine advertises, and these two are
//! gated to the model families that have been explicitly validated for them
//! ([`Scheduler::ADVANCED_PASS`]; see the backends' `advanced_pass_scheduler_names()`).
//!
//! ## `linear_quadratic`
//!
//! The linear-quadratic t-schedule of Meta's *Movie Gen* (arXiv:2410.13720), as written down in
//! Hugging Face **diffusers** (Apache-2.0) `pipelines/mochi/pipeline_mochi.py` and reused by
//! LTX-Video. Over `N` steps with `L = linear_steps` (default `N / 2`), `Q = N − L` and
//! `τ = threshold_noise` (default `0.025`), the *denoised fraction* `f` rises from 0 to 1:
//!
//! ```text
//! f(i) = i·τ/L                            i = 0 .. L−1     (linear segment)
//! f(i) = a·i² + b·i + c                   i = L .. N       (quadratic segment)
//!   d = L − τ·N,   a = d/(L·Q²),   b = τ/L − 2d/Q²,   c = a·L²
//! σ(i) = 1 − f(i)
//! ```
//!
//! Those coefficients are exactly the ones pinned by `f(L) = τ`, `f′(L) = τ/L` and `f(N) = 1`, so the
//! join is C¹-smooth and the schedule finishes fully denoised — all three are asserted by the tests.
//! Most steps land in the high-noise region and the schedule then accelerates, which is the point.
//!
//! ## `bong_tangent`
//!
//! The tangent-based schedule popularised by the RES4LYF ComfyUI extension. RES4LYF is
//! licence-incompatible with this Apache-2.0 crate, so **no RES4LYF source is vendored, fetched or
//! consulted**; the schedule is re-derived from its published description — an arctangent sigmoid in
//! the step index, affinely renormalised so the endpoints are hit exactly:
//!
//! ```text
//! raw(x) = ((2/π)·atan(−slope·(x − pivot)) + 1)/2
//! σ(x)   = (raw(x) − raw(N)) / (raw(0) − raw(N)) · (start − end) + end        x = 0 .. N
//! ```
//!
//! **Documented ambiguity.** RES4LYF publishes no defaults for `slope`/`pivot`, and its formula
//! indexes `x` in *absolute* steps — which makes the curve shape depend on the step count (a 4-step
//! schedule degenerates to nearly linear while a 50-step one is a sharp sigmoid). Because epic
//! 20414's reference recipe runs this schedule on a **4-step** pass, gen-core fixes *step-relative*
//! defaults so the shape is the same at every length: `pivot = 0.6·N` and `slope = 6.0/N` (i.e. slope
//! `6.0` in normalised `u = x/N` units), `start = 1.0`, `end = 0.0`. The equation itself is kept in
//! the published absolute-index parameterisation, so [`bong_tangent_sigmas`] can still reproduce any
//! absolute-index variant a caller asks for.

use super::ModelSampling;

/// The scheduler vocabulary. The first eight are the curated menu (epic 7114 decision 2):
/// `Normal`/`Simple` are the model's native schedule sampled two ways; `Karras`/`Exponential` are
/// analytic σ ramps; `SgmUniform` is `Normal` with the SGM endpoint convention; `Beta` biases steps
/// toward the schedule ends; `DdimUniform` is the DDIM stride; `Beta57` is the RES4LYF
/// `beta_scheduler(α=0.5, β=0.7)` (richer texture/detail).
///
/// `LinearQuadratic`/`BongTangent` are the epic-20414 advanced multi-pass schedules — implemented
/// here like every other schedule, but held out of [`Scheduler::ALL`] and advertised only by
/// explicitly validated families ([`Scheduler::ADVANCED_PASS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Normal,
    Simple,
    Karras,
    Exponential,
    SgmUniform,
    Beta,
    DdimUniform,
    Beta57,
    LinearQuadratic,
    BongTangent,
}

impl Scheduler {
    /// Parse the canonical lowercase name (the UI / recipe vocabulary). Unknown -> `None` (callers
    /// fall back to the model default + emit an event, N3).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "normal" => Self::Normal,
            "simple" => Self::Simple,
            "karras" => Self::Karras,
            "exponential" => Self::Exponential,
            "sgm_uniform" => Self::SgmUniform,
            "beta" => Self::Beta,
            "ddim_uniform" => Self::DdimUniform,
            "beta57" => Self::Beta57,
            "linear_quadratic" => Self::LinearQuadratic,
            "bong_tangent" => Self::BongTangent,
            _ => return None,
        })
    }

    /// The canonical lowercase name (round-trips with [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Simple => "simple",
            Self::Karras => "karras",
            Self::Exponential => "exponential",
            Self::SgmUniform => "sgm_uniform",
            Self::Beta => "beta",
            Self::DdimUniform => "ddim_uniform",
            Self::Beta57 => "beta57",
            Self::LinearQuadratic => "linear_quadratic",
            Self::BongTangent => "bong_tangent",
        }
    }

    /// Every scheduler in the curated vocabulary, in menu order. This is the menu every adopted
    /// engine advertises through the backends' `curated_scheduler_names()`, so it is a compatibility
    /// surface: the epic-20414 additions live in [`Self::ADVANCED_PASS`] instead, and this list stays
    /// exactly the eight ids (and the exact output) it had before.
    pub const ALL: [Scheduler; 8] = [
        Self::Normal,
        Self::Simple,
        Self::Karras,
        Self::Exponential,
        Self::SgmUniform,
        Self::Beta,
        Self::DdimUniform,
        Self::Beta57,
    ];

    /// The advanced multi-pass scheduler vocabulary (epic 20414), in menu order. Gated: a family
    /// only advertises these once it has been explicitly validated for them, by appending the
    /// backends' `advanced_pass_scheduler_names()` to its [`crate::generator::Capabilities`]
    /// `schedulers` list. Everything else keeps the [`Self::ALL`] menu untouched.
    pub const ADVANCED_PASS: [Scheduler; 2] = [Self::LinearQuadratic, Self::BongTangent];

    /// Every scheduler gen-core can build — [`Self::ALL`] followed by [`Self::ADVANCED_PASS`]. Use
    /// this for exhaustive dispatcher coverage, never as a capability menu.
    pub const EVERY: [Scheduler; 10] = [
        Self::Normal,
        Self::Simple,
        Self::Karras,
        Self::Exponential,
        Self::SgmUniform,
        Self::Beta,
        Self::DdimUniform,
        Self::Beta57,
        Self::LinearQuadratic,
        Self::BongTangent,
    ];

    /// Whether this id is one of the gated [`Self::ADVANCED_PASS`] schedules.
    pub fn is_advanced_pass(self) -> bool {
        matches!(self, Self::LinearQuadratic | Self::BongTangent)
    }
}

/// Build the σ schedule for `steps` denoise steps under `scheduler` — ComfyUI `calculate_sigmas`. The
/// result is descending with a trailing `0.0`.
pub fn schedule_sigmas(scheduler: Scheduler, ms: &dyn ModelSampling, steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    match scheduler {
        Scheduler::Karras => {
            let (lo, hi) = table_sigma_range(ms);
            karras_sigmas(steps, lo, hi, 7.0)
        }
        Scheduler::Exponential => {
            let (lo, hi) = table_sigma_range(ms);
            exponential_sigmas(steps, lo, hi)
        }
        Scheduler::Normal => normal_sigmas(ms, steps, false),
        Scheduler::SgmUniform => normal_sigmas(ms, steps, true),
        Scheduler::Simple => simple_sigmas(ms, steps),
        Scheduler::DdimUniform => ddim_uniform_sigmas(ms, steps),
        Scheduler::Beta => beta_sigmas(ms, steps, 0.6, 0.6),
        // beta57 (RES4LYF / Anima community): the same beta_scheduler as `Beta` but with `α = 0.5`,
        // `β = 0.7` — biasing more steps toward the high-noise phase for richer texture/detail. Verified
        // against RES4LYF's ClownsharkBatwing scheduler and the standalone forge-beta57-scheduler
        // (`stats.beta.ppf(q, 0.5, 0.7)`), both identical to ComfyUI `beta_scheduler` with those params.
        Scheduler::Beta57 => beta_sigmas(ms, steps, 0.5, 0.7),
        // epic 20414 advanced multi-pass schedules. Both are analytic ramps over the normalised
        // [1, 0] noise fraction, scaled onto the model's σ_max exactly as ComfyUI scales its own
        // linear-quadratic node — so they land correctly for flow (σ_max = 1), ε/DDPM and EDM alike.
        Scheduler::LinearQuadratic => {
            let (_, hi) = table_sigma_range(ms);
            linear_quadratic_raw(
                steps,
                hi,
                LINEAR_QUADRATIC_THRESHOLD_NOISE,
                default_linear_steps(steps),
            )
        }
        Scheduler::BongTangent => {
            let (_, hi) = table_sigma_range(ms);
            bong_tangent_raw(
                steps,
                hi,
                BONG_TANGENT_SLOPE / steps as f64,
                BONG_TANGENT_PIVOT * steps as f64,
                1.0,
                0.0,
            )
        }
    }
}

/// The checked twin of [`schedule_sigmas`]: rejects a degenerate `steps` up front and rejects a
/// schedule that came out non-finite or non-descending, instead of silently clamping.
///
/// [`schedule_sigmas`] keeps its infallible `steps.max(1)` contract because every adopted engine
/// depends on it (an unset/unknown scheduler must never fail a generation — the N3 rule). Callers
/// that resolve a *user-selected* schedule — the epic-20414 per-pass scheduler knob is the first —
/// should use this instead so a bad step count or a NaN-producing parameter surfaces as a typed
/// error at the seam rather than as a black image several minutes later.
pub fn try_schedule_sigmas(
    scheduler: Scheduler,
    ms: &dyn ModelSampling,
    steps: usize,
) -> Result<Vec<f32>, ScheduleError> {
    if steps == 0 {
        return Err(ScheduleError::DegenerateSteps {
            scheduler: scheduler.name(),
            steps,
        });
    }
    let sigmas = schedule_sigmas(scheduler, ms, steps);
    validate_schedule(scheduler.name(), &sigmas)?;
    Ok(sigmas)
}

/// A schedule this module refused to produce. Typed so the request layer can report *which* knob was
/// wrong rather than stringifying a generic message; converts into [`crate::Error::Msg`] (the same
/// class as the request-layer `steps must be >= 1` floor — a capability gap is
/// [`crate::Error::Unsupported`] and is raised by `Capabilities::validate_request`, not here).
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    /// `steps` was zero (or otherwise below the schedule's floor).
    #[error("scheduler {scheduler}: steps must be >= 1, got {steps}")]
    DegenerateSteps {
        scheduler: &'static str,
        steps: usize,
    },

    /// A schedule parameter was outside the range the closed form is defined on.
    #[error("scheduler {scheduler}: invalid {parameter}: {reason}")]
    InvalidParameter {
        scheduler: &'static str,
        parameter: &'static str,
        reason: String,
    },

    /// A produced σ was NaN or infinite.
    #[error("scheduler {scheduler}: sigma[{index}] is not finite ({value})")]
    NonFinite {
        scheduler: &'static str,
        index: usize,
        value: f32,
    },

    /// The schedule violated the descending / positive / trailing-zero contract.
    #[error("scheduler {scheduler}: schedule is not strictly descending to 0 at index {index} ({previous} -> {current})")]
    NotDescending {
        scheduler: &'static str,
        index: usize,
        previous: f32,
        current: f32,
    },
}

impl From<ScheduleError> for crate::Error {
    fn from(e: ScheduleError) -> Self {
        crate::Error::Msg(e.to_string())
    }
}

/// Enforce the module-wide schedule contract: at least two nodes, every value finite, non-increasing,
/// strictly positive before the terminal node, and an exact `0.0` terminal node. (Non-increasing
/// rather than strictly decreasing because `simple` legitimately repeats a σ-table entry when `steps`
/// exceeds the table's resolution; the advanced schedules are checked for *strict* descent by their
/// own tests.)
fn validate_schedule(scheduler: &'static str, sigmas: &[f32]) -> Result<(), ScheduleError> {
    if sigmas.len() < 2 {
        return Err(ScheduleError::DegenerateSteps {
            scheduler,
            steps: sigmas.len().saturating_sub(1),
        });
    }
    for (index, &value) in sigmas.iter().enumerate() {
        if !value.is_finite() {
            return Err(ScheduleError::NonFinite {
                scheduler,
                index,
                value,
            });
        }
    }
    if sigmas[0] <= 0.0 {
        return Err(ScheduleError::NotDescending {
            scheduler,
            index: 0,
            previous: sigmas[0],
            current: sigmas[0],
        });
    }
    for index in 1..sigmas.len() {
        let (previous, current) = (sigmas[index - 1], sigmas[index]);
        let terminal = index + 1 == sigmas.len();
        let ok = current <= previous
            && if terminal {
                current == 0.0
            } else {
                current > 0.0
            };
        if !ok {
            return Err(ScheduleError::NotDescending {
                scheduler,
                index,
                previous,
                current,
            });
        }
    }
    Ok(())
}

/// The (smallest-positive, largest) entries of the model's discrete σ table — ComfyUI feeds these to
/// the analytic schedulers as `σ_min`/`σ_max` (for flux the smallest positive table entry is the real
/// `σ_min`, not the `0.0` the flow ModelSampling reports for the clean end).
fn table_sigma_range(ms: &dyn ModelSampling) -> (f32, f32) {
    let table = ms.sigma_table();
    let hi = table.last().copied().unwrap_or(1.0);
    let lo = table
        .iter()
        .copied()
        .find(|&s| s > 0.0)
        .unwrap_or((hi * 1e-3).max(1e-4));
    (lo, hi)
}

#[inline]
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

// =================================================================================================
// Analytic σ ramps (prediction-independent; take explicit endpoints).
// =================================================================================================

/// Karras et al. (2022) σ schedule: `(σ_max^{1/ρ} + ramp·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ` over
/// `ramp = linspace(0, 1, n)`, with a trailing `0.0`. `ρ = 7` is the ComfyUI default. Length `n + 1`.
pub fn karras_sigmas(n: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let n = n.max(1);
    let rho = rho as f64;
    let min_inv = (sigma_min.max(0.0) as f64).powf(1.0 / rho);
    let max_inv = (sigma_max as f64).powf(1.0 / rho);
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let ramp = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            (max_inv + ramp * (min_inv - max_inv)).powf(rho) as f32
        })
        .collect();
    out.push(0.0);
    out
}

/// Exponential σ schedule: `exp(linspace(ln σ_max, ln σ_min, n))`, with a trailing `0.0`. Length
/// `n + 1` (geometric spacing in σ).
pub fn exponential_sigmas(n: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
    let n = n.max(1);
    let lmin = (sigma_min.max(1e-12) as f64).ln();
    let lmax = (sigma_max.max(1e-12) as f64).ln();
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let f = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            (lmax + (lmin - lmax) * f).exp() as f32
        })
        .collect();
    out.push(0.0);
    out
}

// =================================================================================================
// Advanced multi-pass σ ramps (epic 20414, sc-20416). Analytic, prediction-independent, scaled onto
// the model's σ_max. Derivations and the documented ambiguity are in the module docs.
// =================================================================================================

/// `threshold_noise` default for `linear_quadratic` — the denoised fraction reached at the end of the
/// linear segment. The value Movie Gen / diffusers / LTX-Video all use.
pub const LINEAR_QUADRATIC_THRESHOLD_NOISE: f64 = 0.025;

/// `slope` default for `bong_tangent`, in NORMALISED units (per unit of `x / steps`) — the dispatcher
/// passes `BONG_TANGENT_SLOPE / steps` so the curve keeps its shape at any step count.
pub const BONG_TANGENT_SLOPE: f64 = 6.0;

/// `pivot` default for `bong_tangent`, as a FRACTION of the step count — the dispatcher passes
/// `BONG_TANGENT_PIVOT * steps`, putting the sigmoid's knee 60% of the way down the schedule.
pub const BONG_TANGENT_PIVOT: f64 = 0.6;

/// The default `linear_steps` for `linear_quadratic`: half the schedule, floored at 1 so the closed
/// form stays defined (`steps == 1` is special-cased by the builder and never reads this).
fn default_linear_steps(steps: usize) -> usize {
    (steps / 2).max(1)
}

/// Movie Gen / diffusers **linear-quadratic** σ schedule (epic 20414). `sigma_max` scales the
/// normalised `[1, 0]` noise-fraction ramp onto the model's σ range. Length `n + 1`, descending, with
/// an exact trailing `0.0`.
///
/// `threshold_noise` is the denoised fraction reached at the end of the linear segment
/// ([`LINEAR_QUADRATIC_THRESHOLD_NOISE`]); `linear_steps` is how many of the `n` steps that segment
/// spans (`None` = half the schedule, floored at 1). See the module docs for the closed form and the
/// three constraints its coefficients encode.
///
/// Errors when `n == 0`, when `linear_steps` is outside `1 ..= n − 1`, when `threshold_noise` is not
/// finite, or when the resulting schedule is not a finite descending ramp (which a caller-supplied
/// `threshold_noise ≥ linear_steps / n` can cause — the quadratic segment inverts).
pub fn linear_quadratic_sigmas(
    n: usize,
    sigma_max: f32,
    threshold_noise: f64,
    linear_steps: Option<usize>,
) -> Result<Vec<f32>, ScheduleError> {
    const NAME: &str = "linear_quadratic";
    if n == 0 {
        return Err(ScheduleError::DegenerateSteps {
            scheduler: NAME,
            steps: n,
        });
    }
    if !threshold_noise.is_finite() || threshold_noise <= 0.0 || threshold_noise >= 1.0 {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "threshold_noise",
            reason: format!("must be finite and in (0, 1), got {threshold_noise}"),
        });
    }
    if !sigma_max.is_finite() || sigma_max <= 0.0 {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "sigma_max",
            reason: format!("must be finite and > 0, got {sigma_max}"),
        });
    }
    let linear = linear_steps.unwrap_or_else(|| default_linear_steps(n));
    if n > 1 && !(1..n).contains(&linear) {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "linear_steps",
            reason: format!("must satisfy 1 <= linear_steps < steps ({n}), got {linear}"),
        });
    }
    let sigmas = linear_quadratic_raw(n, sigma_max, threshold_noise, linear);
    validate_schedule(NAME, &sigmas)?;
    Ok(sigmas)
}

/// The unchecked closed form behind [`linear_quadratic_sigmas`] — the dispatcher's path, where
/// `n >= 1` and the defaults are known good.
fn linear_quadratic_raw(n: usize, sigma_max: f32, threshold_noise: f64, linear: usize) -> Vec<f32> {
    let n = n.max(1);
    // n == 1 has no room for both segments (`linear` would be 0); the limit of the closed form is a
    // single full-noise node plus the terminal, which is also what ComfyUI's node returns.
    if n == 1 {
        return vec![sigma_max, 0.0];
    }
    let (nf, lf) = (n as f64, linear as f64);
    let quad = nf - lf;
    let tau = threshold_noise;
    let d = lf - tau * nf;
    let a = d / (lf * quad * quad);
    let b = tau / lf - 2.0 * d / (quad * quad);
    let c = a * lf * lf;

    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64;
            let frac = if i < linear {
                x * tau / lf
            } else {
                a * x * x + b * x + c
            };
            ((1.0 - frac) * sigma_max as f64) as f32
        })
        .collect();
    // `a·n² + b·n + c == 1` exactly in exact arithmetic; push the terminal rather than let f64
    // residue leave a 1e-16 tail where the contract wants a hard 0.
    out.push(0.0);
    out
}

/// RES4LYF-style **bong_tangent** σ schedule (epic 20414) — an arctangent sigmoid in the step index,
/// affinely renormalised so `σ(0) = start·sigma_max` and `σ(n) = end`. Length `n + 1`, descending,
/// with an exact trailing `0.0`.
///
/// `slope` and `pivot` are in ABSOLUTE step-index units (the published parameterisation); the
/// dispatcher supplies the step-relative defaults [`BONG_TANGENT_SLOPE`] `/ n` and
/// [`BONG_TANGENT_PIVOT`] `* n`. See the module docs for why.
///
/// Errors when `n == 0`, when a parameter is non-finite, when `start <= end`, when the sigmoid does
/// not decrease across the schedule (`slope <= 0`), or when the result is not a finite descending
/// ramp.
pub fn bong_tangent_sigmas(
    n: usize,
    sigma_max: f32,
    slope: f64,
    pivot: f64,
    start: f64,
    end: f64,
) -> Result<Vec<f32>, ScheduleError> {
    const NAME: &str = "bong_tangent";
    if n == 0 {
        return Err(ScheduleError::DegenerateSteps {
            scheduler: NAME,
            steps: n,
        });
    }
    for (parameter, value) in [
        ("sigma_max", sigma_max as f64),
        ("slope", slope),
        ("pivot", pivot),
        ("start", start),
        ("end", end),
    ] {
        if !value.is_finite() {
            return Err(ScheduleError::InvalidParameter {
                scheduler: NAME,
                parameter,
                reason: format!("must be finite, got {value}"),
            });
        }
    }
    if sigma_max <= 0.0 {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "sigma_max",
            reason: format!("must be > 0, got {sigma_max}"),
        });
    }
    if slope <= 0.0 {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "slope",
            reason: format!("must be > 0 so the schedule descends, got {slope}"),
        });
    }
    if start <= end {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "start",
            reason: format!("must be greater than end ({end}), got {start}"),
        });
    }
    // The module contract is a schedule that terminates on an exact `0.0` node, so a non-zero `end`
    // is rejected here rather than surfacing later as an opaque "not descending to 0".
    if end != 0.0 {
        return Err(ScheduleError::InvalidParameter {
            scheduler: NAME,
            parameter: "end",
            reason: format!("must be 0.0 (schedules terminate on a zero node), got {end}"),
        });
    }
    let sigmas = bong_tangent_raw(n, sigma_max, slope, pivot, start, end);
    validate_schedule(NAME, &sigmas)?;
    Ok(sigmas)
}

/// The unchecked closed form behind [`bong_tangent_sigmas`] — the dispatcher's path.
fn bong_tangent_raw(
    n: usize,
    sigma_max: f32,
    slope: f64,
    pivot: f64,
    start: f64,
    end: f64,
) -> Vec<f32> {
    let n = n.max(1);
    let raw = |x: f64| ((2.0 / std::f64::consts::PI) * (-slope * (x - pivot)).atan() + 1.0) / 2.0;
    let (hi, lo) = (raw(0.0), raw(n as f64));
    let span = hi - lo;
    let scale = start - end;
    let mut out: Vec<f32> = (0..n)
        .map(|i| (((raw(i as f64) - lo) / span * scale + end) * sigma_max as f64) as f32)
        .collect();
    // `σ(n) == end` by construction; push it exactly so the terminal node is a hard 0.
    out.push((end * sigma_max as f64) as f32);
    out
}

// =================================================================================================
// Model-native schedules (read the ModelSampling timestep<->sigma map / σ table).
// =================================================================================================

/// ComfyUI `normal_scheduler`: evenly-spaced timesteps between `timestep(σ_max)` and `timestep(σ_min)`,
/// mapped back through [`ModelSampling::sigma`], with a trailing `0.0`. `sgm = true` is the SGM-uniform
/// endpoint convention (`linspace(start, end, steps + 1)[:-1]`). Length `steps + 1`.
pub fn normal_sigmas(ms: &dyn ModelSampling, steps: usize, sgm: bool) -> Vec<f32> {
    let steps = steps.max(1);
    let start = ms.timestep(ms.sigma_max());
    let end = ms.timestep(ms.sigma_min());
    let timesteps: Vec<f32> = (0..steps)
        .map(|i| {
            let f = if sgm {
                // linspace(start, end, steps + 1)[:-1] -> fraction i / steps.
                i as f32 / steps as f32
            } else if steps == 1 {
                0.0
            } else {
                // linspace(start, end, steps) -> fraction i / (steps - 1).
                i as f32 / (steps - 1) as f32
            };
            lerp(start, end, f)
        })
        .collect();
    let mut sigs: Vec<f32> = timesteps.iter().map(|&t| ms.sigma(t)).collect();
    sigs.push(0.0);
    sigs
}

/// ComfyUI `simple_scheduler`: the native σ table sub-sampled by a fixed stride from the noisy end,
/// `table[-(1 + ⌊x·len/steps⌋)]` for `x in 0..steps`, with a trailing `0.0`. Length `steps + 1`.
pub fn simple_sigmas(ms: &dyn ModelSampling, steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    let table = ms.sigma_table();
    let n = table.len();
    if n == 0 {
        return vec![0.0];
    }
    let ss = n as f64 / steps as f64;
    let mut sigs: Vec<f32> = (0..steps)
        .map(|x| {
            let from_end = 1 + (x as f64 * ss) as usize;
            let idx = n.saturating_sub(from_end);
            table[idx.min(n - 1)]
        })
        .collect();
    sigs.push(0.0);
    sigs
}

/// ComfyUI `ddim_scheduler`: a uniform DDIM stride through the native σ table from index 1 upward,
/// reversed to descending, with a trailing `0.0`. The `table[1] ≈ 0` guard bumps `steps` (a near-zero
/// second node, as some schedules have). Length is `~steps + 1` (stride-dependent).
pub fn ddim_uniform_sigmas(ms: &dyn ModelSampling, steps: usize) -> Vec<f32> {
    let mut steps = steps.max(1);
    let table = ms.sigma_table();
    let n = table.len();
    if n <= 1 {
        return vec![0.0];
    }
    if table[1].abs() < 1e-5 {
        steps += 1;
    }
    let ss = (n / steps).max(1);
    let mut sigs: Vec<f32> = Vec::new();
    let mut x = 1usize;
    while x < n {
        sigs.push(table[x]);
        x += ss;
    }
    sigs.reverse();
    sigs.push(0.0);
    sigs
}

/// ComfyUI `beta_scheduler`: timesteps drawn from the inverse Beta CDF (`α = β = 0.6` default, biasing
/// toward the schedule ends), mapped to native σ-table indices with consecutive-duplicate removal, then
/// a trailing `0.0`. Length is `≤ steps + 1` (after dedup).
pub fn beta_sigmas(ms: &dyn ModelSampling, steps: usize, alpha: f64, beta: f64) -> Vec<f32> {
    let steps = steps.max(1);
    let table = ms.sigma_table();
    let n = table.len();
    if n == 0 {
        return vec![0.0];
    }
    let total = n.saturating_sub(1) as f64;
    let mut sigs: Vec<f32> = Vec::new();
    let mut last_t: i64 = -1;
    for i in 0..steps {
        // ts = 1 - linspace(0, 1, steps, endpoint=False)[i] = 1 - i/steps.
        let ts = 1.0 - i as f64 / steps as f64;
        let t = (beta_ppf(ts, alpha, beta) * total).round() as i64;
        if t != last_t {
            let idx = (t.max(0) as usize).min(n - 1);
            sigs.push(table[idx]);
        }
        last_t = t;
    }
    sigs.push(0.0);
    sigs
}

// =================================================================================================
// Inverse Beta CDF (scipy.stats.beta.ppf) — self-contained host f64, no deps. Regularized incomplete
// beta via Lanczos lnΓ + the Numerical-Recipes `betacf` continued fraction, inverted by bisection.
// =================================================================================================

/// Lanczos approximation of `ln Γ(x)` (g = 7), with the reflection formula for `x < 0.5`.
// The coefficients are the canonical published g=7 Lanczos set, kept verbatim for auditing (the
// extra digits round harmlessly into f64) — same rationale as `sampling.rs::compute_mu`.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + G + 0.5;
        let mut a = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Continued-fraction core of the regularized incomplete beta (Numerical Recipes `betacf`, modified
/// Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-13 {
            break;
        }
    }
    h
}

/// The regularized incomplete beta `I_x(a, b)` (monotone increasing in `x`, `I_0 = 0`, `I_1 = 1`).
fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse Beta CDF: the `x ∈ [0, 1]` with `I_x(a, b) = p` (scipy `beta.ppf`), by bisection on the
/// monotone `reg_inc_beta`.
fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if reg_inc_beta(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{
        AlphaSchedule, DiscreteModelSampling, EdmModelSampling, FlowModelSampling,
        TimestepConvention,
    };

    fn sdxl() -> DiscreteModelSampling {
        DiscreteModelSampling::sdxl(&AlphaSchedule::scaled_linear(1000, 0.00085, 0.012))
    }

    fn is_descending_to_zero(sigs: &[f32]) -> bool {
        sigs.len() >= 2
            && *sigs.last().unwrap() == 0.0
            && sigs.windows(2).all(|w| w[0] >= w[1])
            && sigs[..sigs.len() - 1].iter().all(|&s| s > 0.0)
    }

    #[test]
    fn karras_endpoints_and_shape() {
        let s = karras_sigmas(5, 0.1, 10.0, 7.0);
        assert_eq!(s.len(), 6); // n + 1
        assert!((s[0] - 10.0).abs() < 1e-4, "first {}", s[0]);
        assert!((s[4] - 0.1).abs() < 1e-4, "last-nonzero {}", s[4]);
        assert_eq!(s[5], 0.0);
        assert!(is_descending_to_zero(&s));
    }

    #[test]
    fn exponential_is_geometric() {
        // linspace(ln10, ln0.1, 3).exp() = [10, 1, 0.1], + trailing 0.
        let s = exponential_sigmas(3, 0.1, 10.0);
        assert_eq!(s.len(), 4);
        for (g, w) in s.iter().zip([10.0_f32, 1.0, 0.1, 0.0]) {
            assert!((g - w).abs() < 1e-4, "got {g} want {w}");
        }
    }

    #[test]
    fn normal_and_sgm_shapes_and_monotonic() {
        let ms = sdxl();
        let normal = normal_sigmas(&ms, 10, false);
        let sgm = normal_sigmas(&ms, 10, true);
        assert_eq!(normal.len(), 11);
        assert_eq!(sgm.len(), 11);
        assert!(is_descending_to_zero(&normal), "normal {normal:?}");
        assert!(is_descending_to_zero(&sgm), "sgm {sgm:?}");
        // normal starts at the model's max sigma; sgm omits the very first endpoint -> starts lower.
        assert!(normal[0] >= sgm[0]);
    }

    #[test]
    fn simple_shape_and_monotonic() {
        let ms = sdxl();
        let s = simple_sigmas(&ms, 8);
        assert_eq!(s.len(), 9);
        assert!(is_descending_to_zero(&s), "{s:?}");
        // First sampled sigma is the noisy-end of the table.
        assert!((s[0] - ms.sigma_max()).abs() / ms.sigma_max() < 1e-3);
    }

    #[test]
    fn ddim_and_beta_monotonic_to_zero() {
        let ms = sdxl();
        let ddim = ddim_uniform_sigmas(&ms, 10);
        let beta = beta_sigmas(&ms, 10, 0.6, 0.6);
        assert!(is_descending_to_zero(&ddim), "ddim {ddim:?}");
        assert!(is_descending_to_zero(&beta), "beta {beta:?}");
        assert!(*beta.first().unwrap() <= ms.sigma_max() * 1.0001);
    }

    #[test]
    fn beta_ppf_matches_known_values() {
        // Symmetric Beta(0.6, 0.6): median at 0.5; CDF round-trips its own ppf.
        assert!((beta_ppf(0.5, 0.6, 0.6) - 0.5).abs() < 1e-6);
        for &p in &[0.1_f64, 0.3, 0.7, 0.9] {
            let x = beta_ppf(p, 0.6, 0.6);
            assert!((reg_inc_beta(0.6, 0.6, x) - p).abs() < 1e-5, "p={p} x={x}");
        }
        // Beta(1,1) is uniform: ppf(p) == p.
        assert!((beta_ppf(0.42, 1.0, 1.0) - 0.42).abs() < 1e-5);
    }

    #[test]
    fn dispatcher_covers_all_schedulers_on_every_model_sampling() {
        let sdxl = sdxl();
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        let edm = EdmModelSampling::svd();
        let models: [&dyn ModelSampling; 3] = [&sdxl, &flow, &edm];
        for ms in models {
            for sched in Scheduler::EVERY {
                let sigs = schedule_sigmas(sched, ms, 12);
                assert!(
                    is_descending_to_zero(&sigs),
                    "{} produced non-monotone/zero-terminated schedule: {sigs:?}",
                    sched.name()
                );
            }
        }
    }

    #[test]
    fn scheduler_name_roundtrip() {
        for s in Scheduler::EVERY {
            assert_eq!(Scheduler::from_name(s.name()), Some(s));
        }
        assert_eq!(Scheduler::from_name("nope"), None);
        assert_eq!(Scheduler::from_name("beta57"), Some(Scheduler::Beta57));
        assert_eq!(
            Scheduler::ALL.len(),
            8,
            "curated scheduler set is 8 (added beta57)"
        );
    }

    /// AC3 regression guard. The curated menu is a compatibility surface — every adopted engine
    /// advertises it verbatim — so the epic-20414 additions must NOT leak into it, and each curated
    /// id must still produce exactly the bytes it produced before. The second half re-derives the
    /// eight schedules from their own builders and compares byte-for-byte with the dispatcher.
    #[test]
    fn curated_scheduler_ids_and_output_are_unchanged() {
        assert_eq!(
            Scheduler::ALL.map(|s| s.name()),
            [
                "normal",
                "simple",
                "karras",
                "exponential",
                "sgm_uniform",
                "beta",
                "ddim_uniform",
                "beta57",
            ],
            "the curated scheduler menu is frozen — advanced ids belong in ADVANCED_PASS"
        );
        assert!(
            Scheduler::ALL.iter().all(|s| !s.is_advanced_pass()),
            "no curated id may be flagged advanced"
        );

        let sdxl = sdxl();
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        let edm = EdmModelSampling::svd();
        let models: [&dyn ModelSampling; 3] = [&sdxl, &flow, &edm];
        for ms in models {
            for steps in [1_usize, 4, 10, 20] {
                let (lo, hi) = table_sigma_range(ms);
                for (sched, want) in [
                    (Scheduler::Karras, karras_sigmas(steps, lo, hi, 7.0)),
                    (Scheduler::Exponential, exponential_sigmas(steps, lo, hi)),
                    (Scheduler::Normal, normal_sigmas(ms, steps, false)),
                    (Scheduler::SgmUniform, normal_sigmas(ms, steps, true)),
                    (Scheduler::Simple, simple_sigmas(ms, steps)),
                    (Scheduler::DdimUniform, ddim_uniform_sigmas(ms, steps)),
                    (Scheduler::Beta, beta_sigmas(ms, steps, 0.6, 0.6)),
                    (Scheduler::Beta57, beta_sigmas(ms, steps, 0.5, 0.7)),
                ] {
                    assert_eq!(
                        schedule_sigmas(sched, ms, steps),
                        want,
                        "{} changed at steps={steps}",
                        sched.name()
                    );
                }
            }
        }
    }

    // =============================================================================================
    // epic 20414 / sc-20416 — linear_quadratic + bong_tangent
    // =============================================================================================

    fn is_strictly_descending_to_zero(sigs: &[f32]) -> bool {
        sigs.len() >= 2
            && *sigs.last().unwrap() == 0.0
            && sigs.iter().all(|s| s.is_finite())
            && sigs.windows(2).all(|w| w[0] > w[1])
            && sigs[..sigs.len() - 1].iter().all(|&s| s > 0.0)
    }

    #[test]
    fn advanced_pass_ids_are_gated_out_of_the_curated_menu() {
        assert_eq!(
            Scheduler::ADVANCED_PASS.map(|s| s.name()),
            ["linear_quadratic", "bong_tangent"]
        );
        for s in Scheduler::ADVANCED_PASS {
            assert!(
                s.is_advanced_pass(),
                "{} must be flagged advanced",
                s.name()
            );
            assert!(
                !Scheduler::ALL.contains(&s),
                "{} must stay out of the curated menu",
                s.name()
            );
            assert!(Scheduler::EVERY.contains(&s));
        }
        assert_eq!(Scheduler::EVERY.len(), Scheduler::ALL.len() + 2);
    }

    /// AC1/AC2: exact endpoints, strict monotone descent, finiteness and determinism at every
    /// representative length — including the epic-20414 KreaMania 10-step and 4-step passes.
    #[test]
    fn advanced_schedules_are_finite_monotone_and_deterministic() {
        let sdxl = sdxl();
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        let edm = EdmModelSampling::svd();
        let models: [&dyn ModelSampling; 3] = [&sdxl, &flow, &edm];
        for ms in models {
            let sigma_max = table_sigma_range(ms).1;
            for steps in [1_usize, 2, 4, 8, 10, 20, 30, 50] {
                for sched in Scheduler::ADVANCED_PASS {
                    let got = schedule_sigmas(sched, ms, steps);
                    assert_eq!(got.len(), steps + 1, "{} len at {steps}", sched.name());
                    assert!(
                        is_strictly_descending_to_zero(&got),
                        "{} at {steps}: {got:?}",
                        sched.name()
                    );
                    assert_eq!(got[0], sigma_max, "{} start at {steps}", sched.name());
                    assert_eq!(
                        got,
                        schedule_sigmas(sched, ms, steps),
                        "{} is not deterministic",
                        sched.name()
                    );
                }
            }
        }
    }

    /// The advanced schedules are pure shape × σ_max — the property that lets the goldens be
    /// generated once against a σ_max = 1 flow model and still pin the ε/DDPM and EDM cases.
    #[test]
    fn advanced_schedules_scale_by_sigma_max() {
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        assert_eq!(table_sigma_range(&flow).1, 1.0);
        let sdxl = sdxl();
        let sigma_max = table_sigma_range(&sdxl).1;
        for steps in [4_usize, 10, 20] {
            for sched in Scheduler::ADVANCED_PASS {
                let unit = schedule_sigmas(sched, &flow, steps);
                let scaled = schedule_sigmas(sched, &sdxl, steps);
                for (i, (u, s)) in unit.iter().zip(&scaled).enumerate() {
                    let want = u * sigma_max;
                    assert!(
                        (s - want).abs() <= 1e-4 * sigma_max,
                        "{} [{i}] at {steps}: {s} != {want}",
                        sched.name()
                    );
                }
            }
        }
    }

    /// The three constraints the published linear-quadratic coefficients encode: value continuity,
    /// slope continuity, and a fully-denoised finish. Checked on the denoised-fraction `1 - σ`.
    #[test]
    fn linear_quadratic_matches_its_published_constraints() {
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        for steps in [2_usize, 4, 10, 20, 50] {
            let linear = default_linear_steps(steps);
            let s = schedule_sigmas(Scheduler::LinearQuadratic, &flow, steps);
            let frac: Vec<f64> = s.iter().map(|&x| 1.0 - x as f64).collect();
            let tau = LINEAR_QUADRATIC_THRESHOLD_NOISE;
            // Linear segment: f(i) = i·τ/L exactly.
            for (i, f) in frac.iter().enumerate().take(linear) {
                assert!(
                    (f - i as f64 * tau / linear as f64).abs() < 1e-6,
                    "linear segment at {i} of {steps}: {f}"
                );
            }
            // Value continuity at the join, and a fully-denoised finish.
            assert!(
                (frac[linear] - tau).abs() < 1e-6,
                "join value at {steps}: {}",
                frac[linear]
            );
            assert!(
                (frac[steps] - 1.0).abs() < 1e-6,
                "finish at {steps}: {}",
                frac[steps]
            );
            // Slope continuity: the first quadratic increment matches the linear step size.
            if steps > 2 {
                let quad_slope = frac[linear + 1] - frac[linear];
                assert!(
                    quad_slope >= tau / linear as f64,
                    "quadratic must not start slower than the linear segment at {steps}"
                );
            }
        }
    }

    /// bong_tangent is an arctangent sigmoid pinned to its endpoints — the defining property of the
    /// published form: whatever the slope/pivot, σ(0) and σ(n) land exactly.
    #[test]
    fn bong_tangent_pins_its_endpoints_for_any_slope_and_pivot() {
        for &(slope, pivot) in &[(0.05, 3.0), (0.5, 6.0), (3.0, 8.0), (12.0, 0.5)] {
            let s = bong_tangent_sigmas(12, 1.0, slope, pivot, 1.0, 0.0)
                .expect("valid slope/pivot must build");
            assert_eq!(s.len(), 13);
            assert_eq!(s[0], 1.0, "slope={slope} pivot={pivot}");
            assert_eq!(s[12], 0.0);
            assert!(is_strictly_descending_to_zero(&s), "{s:?}");
        }
        // The step-relative defaults keep the SHAPE constant across step counts: the midpoint of a
        // 4-step and a 40-step schedule agree, which an absolute-index slope would not do.
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        let short = schedule_sigmas(Scheduler::BongTangent, &flow, 4);
        let long = schedule_sigmas(Scheduler::BongTangent, &flow, 40);
        assert!(
            (short[2] - long[20]).abs() < 1e-5,
            "shape drifted with step count: {} vs {}",
            short[2],
            long[20]
        );
    }

    /// AC5: degenerate step counts and out-of-range parameters are typed errors, not clamps.
    #[test]
    fn advanced_schedules_reject_degenerate_input_with_typed_errors() {
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        for sched in Scheduler::EVERY {
            assert!(
                matches!(
                    try_schedule_sigmas(sched, &flow, 0),
                    Err(ScheduleError::DegenerateSteps { steps: 0, .. })
                ),
                "{} must reject steps=0",
                sched.name()
            );
            assert!(
                try_schedule_sigmas(sched, &flow, 4).is_ok(),
                "{}",
                sched.name()
            );
        }

        assert!(matches!(
            linear_quadratic_sigmas(0, 1.0, 0.025, None),
            Err(ScheduleError::DegenerateSteps { .. })
        ));
        assert!(matches!(
            linear_quadratic_sigmas(10, 1.0, f64::NAN, None),
            Err(ScheduleError::InvalidParameter {
                parameter: "threshold_noise",
                ..
            })
        ));
        assert!(matches!(
            linear_quadratic_sigmas(10, 1.0, 0.025, Some(0)),
            Err(ScheduleError::InvalidParameter {
                parameter: "linear_steps",
                ..
            })
        ));
        assert!(matches!(
            linear_quadratic_sigmas(10, 1.0, 0.025, Some(10)),
            Err(ScheduleError::InvalidParameter {
                parameter: "linear_steps",
                ..
            })
        ));
        assert!(matches!(
            linear_quadratic_sigmas(10, f32::NAN, 0.025, None),
            Err(ScheduleError::InvalidParameter {
                parameter: "sigma_max",
                ..
            })
        ));
        // τ >= L/n inverts the quadratic segment; the finite/descending guard catches it.
        assert!(matches!(
            linear_quadratic_sigmas(10, 1.0, 0.9, None),
            Err(ScheduleError::NotDescending { .. })
        ));

        assert!(matches!(
            bong_tangent_sigmas(0, 1.0, 0.6, 6.0, 1.0, 0.0),
            Err(ScheduleError::DegenerateSteps { .. })
        ));
        assert!(matches!(
            bong_tangent_sigmas(10, 1.0, 0.0, 6.0, 1.0, 0.0),
            Err(ScheduleError::InvalidParameter {
                parameter: "slope",
                ..
            })
        ));
        assert!(matches!(
            bong_tangent_sigmas(10, 1.0, 0.6, f64::INFINITY, 1.0, 0.0),
            Err(ScheduleError::InvalidParameter {
                parameter: "pivot",
                ..
            })
        ));
        assert!(matches!(
            bong_tangent_sigmas(10, 1.0, 0.6, 6.0, 0.0, 0.0),
            Err(ScheduleError::InvalidParameter {
                parameter: "start",
                ..
            })
        ));
        assert!(matches!(
            bong_tangent_sigmas(10, 1.0, 0.6, 6.0, 1.0, 0.1),
            Err(ScheduleError::InvalidParameter {
                parameter: "end",
                ..
            })
        ));

        // A typed schedule error crosses into the crate error as a message, never as Unsupported
        // (which is reserved for the capability gap `Capabilities::validate_request` raises).
        let err: crate::Error = ScheduleError::DegenerateSteps {
            scheduler: "bong_tangent",
            steps: 0,
        }
        .into();
        assert!(matches!(err, crate::Error::Msg(_)), "{err:?}");
        assert!(err.to_string().contains("bong_tangent"));
    }

    /// A short hand-derived spot check, independent of the JSON goldens: the epic-20414 10-step
    /// linear-quadratic pass, computed from the closed form by hand (τ = 0.025, L = 5, Q = 5 →
    /// a = 0.038, b = −0.375, c = 0.95).
    #[test]
    fn linear_quadratic_ten_step_matches_hand_derivation() {
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        let got = schedule_sigmas(Scheduler::LinearQuadratic, &flow, 10);
        let want = [
            1.0_f32, 0.995, 0.99, 0.985, 0.98, 0.975, 0.932, 0.813, 0.618, 0.347, 0.0,
        ];
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "[{i}] got {g} want {w}");
        }
    }

    #[test]
    fn beta57_is_beta_scheduler_with_alpha_half_beta_seven_tenths() {
        // beta57 (RES4LYF / forge-beta57-scheduler) is EXACTLY beta_scheduler(α=0.5, β=0.7) — a
        // distinct schedule from the α=β=0.6 `beta`, verified against both reference implementations.
        let ms = sdxl();
        let got = schedule_sigmas(Scheduler::Beta57, &ms, 12);
        let want = beta_sigmas(&ms, 12, 0.5, 0.7);
        assert_eq!(
            got, want,
            "beta57 must dispatch to beta_sigmas(α=0.5, β=0.7)"
        );
        assert!(is_descending_to_zero(&got), "beta57 {got:?}");
        // It genuinely differs from the α=β=0.6 `beta` schedule.
        assert_ne!(got, beta_sigmas(&ms, 12, 0.6, 0.6));
    }
}

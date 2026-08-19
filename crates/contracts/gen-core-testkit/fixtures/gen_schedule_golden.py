#!/usr/bin/env python3
"""Regenerate the INDEPENDENT golden sigma schedules for the gen-core `linear_quadratic` and
`bong_tangent` schedulers (epic 20414, sc-20416).

    Regenerate with:   python3 crates/contracts/gen-core-testkit/fixtures/gen_schedule_golden.py
    (requires:         nothing -- Python 3 stdlib only, no numpy, no torch, no network)

WHY THIS SCRIPT EXISTS
----------------------
`schedule_golden.json` must be an INDEPENDENT witness of the two schedules, not a re-print of the
Rust implementation. This generator transcribes the published closed-form equations directly (see
the per-schedule derivations below) in plain f64 Python and commits only the COMPUTED numbers. The
Rust port in `gen-core/src/sampling/schedulers.rs` is written from the same equations, so a typo on
either side shows up as a fixture mismatch rather than as two copies of the same mistake.

Unlike `gen-core/tests/fixtures/gen_sde_solver_golden.py` this script FETCHES NOTHING and vendors
nothing: both schedules are closed-form, so there is no reference implementation to import.


linear_quadratic
----------------
The linear-quadratic t-schedule of Meta's Movie Gen (arXiv:2410.13720, appendix "linear-quadratic
t-schedule") as it is written down in Hugging Face `diffusers` -- Apache-2.0, the same licence as
this repository -- in `pipelines/mochi/pipeline_mochi.py::linear_quadratic_schedule`, and reused by
LTX-Video. Nothing is copied from a GPL/AGPL custom-node repository.

Let N = steps, L = linear_steps (default N // 2), Q = N - L, tau = threshold_noise (default 0.025).
The schedule is a "denoised fraction" f(i) that rises from 0 to 1 over i = 0..N:

    f(i) = i * tau / L                              for i in 0..L-1        (linear segment)
    f(i) = a*i^2 + b*i + c                          for i in L..N          (quadratic segment)
      with  d = L - tau*N,  a = d / (L * Q^2),  b = tau/L - 2*d/Q^2,  c = a * L^2

Those three coefficients are exactly the ones pinned by the three natural constraints, which this
script asserts numerically for every case it emits:

    f(L)  = tau        (value continuity at the join)
    f'(L) = tau / L    (slope continuity at the join)
    f(N)  = 1          (the schedule finishes fully denoised)

The sigma schedule is sigma(i) = 1 - f(i), i = 0..N -- descending, sigma(0) = 1, sigma(N) = 0,
length N + 1, which is precisely gen-core's schedule convention. `diffusers` emits the same numbers
as a length-N list and lets `FlowMatchEulerDiscreteScheduler` append the terminal 0.

N == 1 is degenerate (L = 0 -> division by zero) and is defined as [1, 0], the limit of the above.


bong_tangent
------------
The tangent-based schedule popularised by the RES4LYF ComfyUI extension. RES4LYF's own source is
license-incompatible with this Apache-2.0 repository and is deliberately NOT consulted, vendored or
fetched; the schedule is re-derived here from its published DESCRIPTION -- an arctangent sigmoid in
the step index, affinely renormalised so the schedule hits its endpoints exactly:

    raw(x)   = ((2/pi) * atan(-slope * (x - pivot)) + 1) / 2       (a decreasing sigmoid in x)
    sigma(x) = (raw(x) - raw(N)) / (raw(0) - raw(N)) * (start - end) + end,   x = 0..N

`raw` is a smooth monotone decreasing map; the affine renormalisation pins sigma(0) = start and
sigma(N) = end regardless of slope/pivot, which is the defining property of the published form.

DEFINITIONAL AMBIGUITY (documented choice, see the Rust module docs): RES4LYF publishes no
defaults for `slope`/`pivot`, and its formula indexes x in ABSOLUTE steps, which makes the curve
shape depend on the step count (a 4-step schedule degenerates to nearly linear while a 50-step one
is a sharp sigmoid). Because epic 20414's reference recipe uses this schedule on a 4-step pass, we
fix STEP-RELATIVE defaults so the curve has the same shape at every length:

    pivot = 0.6 * N          (the knee sits 60% of the way down the schedule)
    slope = 6.0 / N          (i.e. slope is 6.0 in normalised u = x/N units)
    start = 1.0, end = 0.0

The published equation is kept verbatim in the absolute-index parameterisation, so a caller that
wants RES4LYF's exact absolute-index behaviour can pass its own slope/pivot.


WHAT IS EMITTED
---------------
Both schedules over a sigma_max = 1.0 model (gen-core's `FlowModelSampling`), which is the pure
shape of the schedule; the Rust suite separately proves that a model with sigma_max != 1 yields the
same shape scaled by sigma_max. Step counts cover degenerate (1, 2), the epic 20414 KreaMania
recipe (10-step pass 1, 4-step pass 2), and production lengths (20, 30, 50).


CURATED SCHEDULES (sc-20418 adjacent fix)
-----------------------------------------
The curated eight (`normal` / `simple` / `karras` / `exponential` / `sgm_uniform` / `beta` /
`ddim_uniform` / `beta57`) are ALSO pinned here over the same flow model. Before sc-20418 their
only regression guard recomputed both sides from the same Rust builders (gen-core's
`curated_scheduler_ids_and_output_are_unchanged`), which pins the dispatcher wiring but is blind to
a builder-body edit. These entries are the durable byte-pin: transcriptions of the published closed
forms (Karras et al. 2022 eq. 5; geometric/log-linear spacing; ComfyUI's normal / sgm_uniform /
simple / ddim / beta samplings over the model sigma table; the Beta-CDF timestep draw, with the
same Lanczos-lnGamma + Numerical-Recipes-betacf + bisection inverse the Rust port derives from)
in plain f64 Python.

The flow model's discrete sigma table (used by simple / ddim_uniform / beta / beta57) is gen-core's
default `ModelSampling::sigma_table`: 1000 nodes, linear in sigma from sigma_min = 1/1000 to
sigma_max = 1.0. Its Rust form interpolates in f32; the committed f64 numbers differ by < 1e-7,
inside the fixture's 1e-6 tolerance.
"""
import json
import math
from pathlib import Path

# The gen-core defaults these goldens are generated for. Keep in lockstep with
# `schedulers.rs::{LINEAR_QUADRATIC_THRESHOLD_NOISE, BONG_TANGENT_SLOPE, BONG_TANGENT_PIVOT}`.
LINEAR_QUADRATIC_THRESHOLD_NOISE = 0.025
BONG_TANGENT_SLOPE = 6.0  # normalised (per unit of x/N)
BONG_TANGENT_PIVOT = 0.6  # fraction of N

STEP_COUNTS = [1, 2, 4, 8, 10, 20, 30, 50]


def linear_quadratic(n, threshold_noise=LINEAR_QUADRATIC_THRESHOLD_NOISE, linear_steps=None):
    """Movie Gen / diffusers linear-quadratic sigma schedule. Returns n + 1 descending sigmas."""
    if n < 1:
        raise ValueError("steps must be >= 1")
    if n == 1:
        return [1.0, 0.0]
    if linear_steps is None:
        linear_steps = n // 2
    lin, tau = linear_steps, threshold_noise
    quad = n - lin
    if not 1 <= lin < n:
        raise ValueError("linear_steps must satisfy 1 <= linear_steps < steps")

    d = lin - tau * n
    a = d / (lin * quad**2)
    b = tau / lin - 2.0 * d / quad**2
    c = a * lin**2

    def quadratic(i):
        return a * i * i + b * i + c

    # The three constraints the published coefficients encode -- asserted, never assumed.
    assert math.isclose(quadratic(lin), tau, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(2.0 * a * lin + b, tau / lin, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(quadratic(n), 1.0, rel_tol=1e-12, abs_tol=1e-12)

    frac = [i * tau / lin for i in range(lin)]
    frac += [quadratic(i) for i in range(lin, n)]
    sigmas = [1.0 - f for f in frac]
    sigmas.append(0.0)  # == 1 - quadratic(n), exactly zero rather than 1e-16 of float residue
    return sigmas


def bong_tangent(n, slope=None, pivot=None, start=1.0, end=0.0):
    """Arctangent sigmoid schedule, endpoint-pinned. Returns n + 1 descending sigmas."""
    if n < 1:
        raise ValueError("steps must be >= 1")
    if slope is None:
        slope = BONG_TANGENT_SLOPE / n
    if pivot is None:
        pivot = BONG_TANGENT_PIVOT * n

    def raw(x):
        return ((2.0 / math.pi) * math.atan(-slope * (x - pivot)) + 1.0) / 2.0

    hi, lo = raw(0), raw(n)
    span = hi - lo
    if not span > 0.0:
        raise ValueError("degenerate slope/pivot: the sigmoid does not decrease over the schedule")
    scale = start - end
    sigmas = [(raw(x) - lo) / span * scale + end for x in range(n)]
    sigmas.append(end)
    return sigmas


# --- Curated schedules (sc-20418): the flow model surface the Rust builders read -----------------

FLOW_NUM_TIMESTEPS = 1000
FLOW_SIGMA_MIN = 1.0 / FLOW_NUM_TIMESTEPS
FLOW_SIGMA_MAX = 1.0


def flow_sigma_table():
    """gen-core `ModelSampling::sigma_table` default over the unshifted flow model: 1000 nodes,
    ascending, linear from sigma_min to sigma_max (timestep(s) = s, sigma(t) = t at mu = 0)."""
    n = FLOW_NUM_TIMESTEPS
    lo, hi = FLOW_SIGMA_MIN, FLOW_SIGMA_MAX
    return [lo + (hi - lo) * (i / (n - 1)) for i in range(n)]


def karras(n, sigma_min=FLOW_SIGMA_MIN, sigma_max=FLOW_SIGMA_MAX, rho=7.0):
    """Karras et al. (2022) eq. 5, trailing 0. Length n + 1."""
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    out = []
    for i in range(n):
        ramp = 0.0 if n == 1 else i / (n - 1)
        out.append((max_inv + ramp * (min_inv - max_inv)) ** rho)
    out.append(0.0)
    return out


def exponential(n, sigma_min=FLOW_SIGMA_MIN, sigma_max=FLOW_SIGMA_MAX):
    """Geometric (log-linear) spacing, trailing 0. Length n + 1."""
    lmin, lmax = math.log(sigma_min), math.log(sigma_max)
    out = []
    for i in range(n):
        f = 0.0 if n == 1 else i / (n - 1)
        out.append(math.exp(lmax + (lmin - lmax) * f))
    out.append(0.0)
    return out


def normal(n, sgm=False):
    """ComfyUI normal / sgm_uniform over the flow model: timesteps lerped from timestep(sigma_max)
    to timestep(sigma_min) (identity maps at mu = 0), trailing 0. Length n + 1."""
    start, end = FLOW_SIGMA_MAX, FLOW_SIGMA_MIN
    out = []
    for i in range(n):
        if sgm:
            f = i / n
        else:
            f = 0.0 if n == 1 else i / (n - 1)
        out.append(start + (end - start) * f)
    out.append(0.0)
    return out


def simple(n):
    """ComfyUI simple_scheduler: the sigma table sub-sampled by a fixed stride from the noisy end,
    trailing 0. Length n + 1."""
    table = flow_sigma_table()
    ss = len(table) / n
    out = []
    for x in range(n):
        from_end = 1 + int(x * ss)
        idx = max(len(table) - from_end, 0)
        out.append(table[min(idx, len(table) - 1)])
    out.append(0.0)
    return out


def ddim_uniform(n):
    """ComfyUI ddim_scheduler: a uniform stride through the table from index 1 upward, reversed,
    trailing 0. Length is stride-dependent (~n + 1)."""
    table = flow_sigma_table()
    steps = n
    if abs(table[1]) < 1e-5:
        steps += 1
    ss = max(len(table) // steps, 1)
    out = []
    x = 1
    while x < len(table):
        out.append(table[x])
        x += ss
    out.reverse()
    out.append(0.0)
    return out


# Inverse Beta CDF: the same Lanczos-lnGamma + Numerical-Recipes betacf + bisection construction
# the Rust port uses, in plain f64, so both sides compute the identical table index.
_LANCZOS_C = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]


def ln_gamma(x):
    if x < 0.5:
        return math.log(math.pi) - math.log(abs(math.sin(math.pi * x))) - ln_gamma(1.0 - x)
    x -= 1.0
    t = x + 7.0 + 0.5
    a = _LANCZOS_C[0]
    for i in range(1, 9):
        a += _LANCZOS_C[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def betacf(a, b, x):
    fpmin = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < 1e-13:
            break
    return h


def reg_inc_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def beta_ppf(p, a, b):
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if reg_inc_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def beta_schedule(n, alpha, beta):
    """ComfyUI beta_scheduler: inverse-Beta-CDF timesteps mapped to table indices with
    consecutive-duplicate removal, trailing 0. Length <= n + 1."""
    table = flow_sigma_table()
    total = len(table) - 1
    out = []
    last_t = -1
    for i in range(n):
        ts = 1.0 - i / n
        # Rust f64 `.round()` is half-away-from-zero; the argument is non-negative here.
        t = int(beta_ppf(ts, alpha, beta) * total + 0.5)
        if t != last_t:
            out.append(table[min(max(t, 0), len(table) - 1)])
        last_t = t
    out.append(0.0)
    return out


# Every curated id, with its builder and whether its output length is exactly steps + 1.
CURATED = [
    ("normal", lambda n: normal(n, sgm=False), True),
    ("simple", simple, True),
    ("karras", karras, True),
    ("exponential", exponential, True),
    ("sgm_uniform", lambda n: normal(n, sgm=True), True),
    ("beta", lambda n: beta_schedule(n, 0.6, 0.6), False),
    ("ddim_uniform", ddim_uniform, False),
    ("beta57", lambda n: beta_schedule(n, 0.5, 0.7), False),
]


def main():
    cases = []
    for n in STEP_COUNTS:
        note = ""
        if n == 10:
            note = "epic 20414 KreaMania V6 pass 1 (10 steps)"
        elif n == 4:
            note = "epic 20414 KreaMania V6 pass 2 (4 steps)"
        elif n == 1:
            note = "degenerate single-step schedule"
        cases.append(
            {
                "scheduler": "linear_quadratic",
                "model": "flow",
                "steps": n,
                "note": note,
                "sigmas": linear_quadratic(n),
            }
        )
        cases.append(
            {
                "scheduler": "bong_tangent",
                "model": "flow",
                "steps": n,
                "note": note,
                "sigmas": bong_tangent(n),
            }
        )

    exact_length = {"linear_quadratic", "bong_tangent"}
    for name, build, length_preserving in CURATED:
        if length_preserving:
            exact_length.add(name)
        for n in STEP_COUNTS:
            cases.append(
                {
                    "scheduler": name,
                    "model": "flow",
                    "steps": n,
                    "note": "curated byte-pin (sc-20418)",
                    "sigmas": build(n),
                }
            )

    for case in cases:
        sig = case["sigmas"]
        if case["scheduler"] in exact_length:
            assert len(sig) == case["steps"] + 1, case
        assert len(sig) >= 2, case
        assert sig[-1] == 0.0, case
        assert sig[0] > 0.0, case
        assert all(math.isfinite(s) for s in sig), case
        assert all(sig[i] > sig[i + 1] for i in range(len(sig) - 1)), case

    doc = {
        "meta": {
            "purpose": (
                "Independent golden sigma schedules for the gen-core schedulers: the advanced "
                "linear_quadratic + bong_tangent pair (epic 20414, sc-20416) and the curated "
                "eight (byte-pinned in sc-20418)."
            ),
            "generator": "crates/contracts/gen-core-testkit/fixtures/gen_schedule_golden.py",
            "model": (
                "gen_core::sampling::FlowModelSampling (sigma_max = 1.0); a model with a different "
                "sigma_max yields the same shape scaled by sigma_max."
            ),
            "linear_quadratic_reference": (
                "Meta Movie Gen (arXiv:2410.13720) linear-quadratic t-schedule as written in "
                "diffusers (Apache-2.0) pipelines/mochi/pipeline_mochi.py::linear_quadratic_schedule"
            ),
            "bong_tangent_reference": (
                "Arctangent sigmoid schedule popularised by the RES4LYF ComfyUI extension, "
                "re-derived from its published description; no RES4LYF source is vendored, fetched "
                "or consulted (licence-incompatible with this Apache-2.0 repository)."
            ),
            "defaults": {
                "linear_quadratic": {
                    "threshold_noise": LINEAR_QUADRATIC_THRESHOLD_NOISE,
                    "linear_steps": "steps // 2",
                },
                "bong_tangent": {
                    "slope": f"{BONG_TANGENT_SLOPE} / steps",
                    "pivot": f"{BONG_TANGENT_PIVOT} * steps",
                    "start": 1.0,
                    "end": 0.0,
                },
            },
            "convention": (
                "Descending, length steps + 1, sigmas[0] == sigma_max, trailing exact 0.0 -- the "
                "gen-core schedule contract."
            ),
            "tolerance": "1e-6 absolute (the Rust port emits f32 from f64 intermediates)",
        },
        "cases": cases,
    }

    out = Path(__file__).with_name("schedule_golden.json")
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

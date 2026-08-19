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

    for case in cases:
        sig = case["sigmas"]
        assert len(sig) == case["steps"] + 1
        assert sig[0] == 1.0 and sig[-1] == 0.0
        assert all(math.isfinite(s) for s in sig)
        assert all(sig[i] > sig[i + 1] for i in range(len(sig) - 1))

    doc = {
        "meta": {
            "purpose": (
                "Independent golden sigma schedules for the gen-core linear_quadratic and "
                "bong_tangent schedulers (epic 20414, sc-20416)."
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

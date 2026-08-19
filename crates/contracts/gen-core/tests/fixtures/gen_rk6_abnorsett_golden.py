#!/usr/bin/env python3
"""Regenerate the INDEPENDENT float64 golden trajectories for the gen-core `rk6_7s` and
`abnorsett_4m` solvers (sc-20417).

    Regenerate with:   python3 gen-core/tests/fixtures/gen_rk6_abnorsett_golden.py
    (pure standard library -- no third-party dependencies)

PROVENANCE (this is a from-the-literature implementation, NOT a port of RES4LYF)
--------------------------------------------------------------------------------
* `rk6_7s`: Butcher's classical 6th-order, 7-stage explicit Runge-Kutta tableau
  (J. C. Butcher, "On Runge-Kutta processes of high order", J. Austral. Math. Soc. 4
  (1964), pp. 179-194). The RES4LYF ComfyUI extension's `<order>_<stages>` naming does
  not pin one specific published 6(7) tableau; the repo deliberately commits to Butcher
  1964 and this generator self-verifies ALL rooted-tree order conditions through order 6
  in exact rational arithmetic before writing anything.
* `abnorsett_4m`: 4th-order exponential Adams-Bashforth (Norsett-type) multistep in the
  half-log-SNR time lambda = -ln(sigma), where the probability-flow ODE is the semilinear
  x' = -x + x0hat(lambda). The linear part is solved exactly; x0hat is extrapolated by the
  Lagrange polynomial through up to 4 history nodes integrated exactly against the
  exponential kernel (S. P. Norsett, "An A-stable modification of the Adams-Bashforth
  methods", LNM 109, Springer 1969; Hochbruck & Ostermann, "Exponential integrators",
  Acta Numerica 19 (2010), sec. 2). Lower-order startup ramp: step n runs at order
  min(n+1, 4); the terminal step to sigma=0 (h -> inf) is the order-1 limit = the final
  denoised estimate. The generator self-verifies the weights by their defining quadrature
  property (polynomial exactness against Simpson integration) on every step it takes.

WHAT IT COMPUTES
----------------
Both solvers are driven in float64 over the same nonlinear toy denoiser the Rust suite
uses (`denoised = 0.5*x + 0.25*sin(1.3*x) + 0.2*sigma*cos(0.7*x) - 0.1*sigma`), over
diffusers static-shift-3.0 flow schedules (sigma = 3t/(1+2t), t = linspace(1, 1/n, n),
trailing 0.0), with every sigma pre-rounded to f32 so the Rust f32 loader sees the exact
same grid. The recorded trajectory is the latent handed to EVERY model call (for rk6_7s
that includes each of the 7 stage latents per interior step) plus the final latent.
The Rust port carries latents in f32, so the tests compare within an f32-appropriate
tolerance (RK_GOLDEN_TOL in solvers.rs).
"""
import json
import math
import struct
from fractions import Fraction as Frac
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rk6_abnorsett_golden.json"


def f32(v: float) -> float:
    """Round a float64 to the nearest f32, returned as float64."""
    return struct.unpack("f", struct.pack("f", v))[0]


# --- schedules: diffusers static shift 3.0 (sigma' = 3t/(1+2t)), f32-rounded ------------
def shift3_sigmas(n: int) -> list:
    ts = [1.0 + (1.0 / n - 1.0) * i / (n - 1) for i in range(n)]
    sig = [f32(3.0 * t / (1.0 + 2.0 * t)) for t in ts]
    return sig + [0.0]


# --- the toy denoiser (mirrors solvers.rs `toy_denoised`, computed in float64) ----------
def toy_denoised(x, sigma):
    return [
        0.5 * xi + 0.25 * math.sin(1.3 * xi) + 0.2 * sigma * math.cos(0.7 * xi) - 0.1 * sigma
        for xi in x
    ]


# --- rk6_7s: Butcher (1964) 6(7) tableau ------------------------------------------------
C = [Frac(0), Frac(1, 3), Frac(2, 3), Frac(1, 3), Frac(1, 2), Frac(1, 2), Frac(1)]
A = [
    [],
    [Frac(1, 3)],
    [Frac(0), Frac(2, 3)],
    [Frac(1, 12), Frac(1, 3), Frac(-1, 12)],
    [Frac(-1, 16), Frac(9, 8), Frac(-3, 16), Frac(-3, 8)],
    [Frac(0), Frac(9, 8), Frac(-3, 8), Frac(-3, 4), Frac(1, 2)],
    [Frac(9, 44), Frac(-9, 11), Frac(63, 44), Frac(18, 11), Frac(0), Frac(-16, 11)],
]
B = [Frac(11, 120), Frac(0), Frac(27, 40), Frac(27, 40), Frac(-4, 15), Frac(-4, 15), Frac(11, 120)]


def verify_tableau():
    """All rooted-tree order conditions through order 6, exact rational arithmetic."""

    def trees(n, memo={}):
        if n in memo:
            return memo[n]
        if n == 1:
            memo[1] = [()]
            return memo[1]
        result = set()

        def parts(total, mx):
            if total == 0:
                yield []
                return
            for p in range(min(total, mx), 0, -1):
                for rest in parts(total - p, p):
                    yield [p] + rest

        for part in parts(n - 1, n - 1):
            pools = [trees(k) for k in part]

            def pick(idx, chosen):
                if idx == len(pools):
                    result.add(tuple(sorted(chosen)))
                    return
                for t in pools[idx]:
                    pick(idx + 1, chosen + [t])

            pick(0, [])
        memo[n] = sorted(result)
        return memo[n]

    def order(t):
        return 1 + sum(order(s) for s in t)

    def gamma(t):
        g = order(t)
        for s in t:
            g *= gamma(s)
        return g

    def gvec(t):
        if not t:
            return [Frac(1)] * 7
        sub = [gvec(s) for s in t]
        out = []
        for i in range(7):
            prod = Frac(1)
            for gv in sub:
                prod *= sum((A[i][j] * gv[j] for j in range(len(A[i]))), Frac(0))
            out.append(prod)
        return out

    checked = 0
    for p in range(1, 7):
        for t in trees(p):
            gv = gvec(t)
            phi = sum((B[i] * gv[i] for i in range(7)), Frac(0))
            assert phi == Frac(1, gamma(t)), f"order-{p} condition failed for tree {t}"
            checked += 1
    assert checked == 37, checked
    # teeth: not order 7
    fails7 = sum(
        1
        for t in trees(7)
        if sum((B[i] * gvec(t)[i] for i in range(7)), Frac(0)) != Frac(1, gamma(t))
    )
    assert fails7 > 0, "tableau unexpectedly satisfies all order-7 conditions"
    for i in range(7):
        assert sum(A[i], Frac(0)) == C[i], f"row {i} sum != c"


CF = [float(v) for v in C]
AF = [[float(v) for v in row] for row in A]
BF = [float(v) for v in B]


def run_rk6(x_init, sigmas, record):
    x = list(x_init)
    for i in range(len(sigmas) - 1):
        sigma, s_next = sigmas[i], sigmas[i + 1]
        if sigma == 0.0 or s_next == 0.0:
            record.append(list(x))
            x = toy_denoised(x, sigma)
            continue
        h = s_next - sigma
        ks = []
        for j in range(7):
            xj = list(x)
            for l, kl in enumerate(ks):
                w = h * AF[j][l]
                if w != 0.0:
                    xj = [a + w * b for a, b in zip(xj, kl)]
            sj = sigma + CF[j] * h
            record.append(list(xj))
            x0j = toy_denoised(xj, sj)
            ks.append([(a - b) / sj for a, b in zip(xj, x0j)])
        for j, kj in enumerate(ks):
            w = h * BF[j]
            if w != 0.0:
                x = [a + w * b for a, b in zip(x, kj)]
    return x


# --- abnorsett_4m -----------------------------------------------------------------------
def exp_kernel_integrals(h, k):
    """I_m(h) = int_0^h e^(tau-h) tau^m dtau, m = 0..k-1 (series for h < 1, recursion above)."""
    out = []
    if h < 1.0:
        for m in range(k):
            m_fact = math.factorial(m)
            s, num, denom = 0.0, 1.0, float(math.factorial(m + 1))
            for i in range(20):
                s += num / denom
                num *= -h
                denom *= m + 2 + i
            out.append(h ** (m + 1) * m_fact * s)
    else:
        prev = -math.expm1(-h)
        out.append(prev)
        for m in range(1, k):
            cur = h**m - m * prev
            out.append(cur)
            prev = cur
    return out


def simpson_exp_moment(h, m, n=20000):
    dt = h / n
    f = lambda t: math.exp(t - h) * (1.0 if m == 0 else t**m)
    s = f(0.0) + f(h)
    for i in range(1, n):
        s += (4.0 if i % 2 == 1 else 2.0) * f(i * dt)
    return s * dt / 3.0


def exp_ab_weights(deltas, h, self_check=True):
    k = len(deltas)
    ints = exp_kernel_integrals(h, k)
    ws = []
    for j in range(k):
        coef = [1.0]
        denom = 1.0
        for l, dl in enumerate(deltas):
            if l == j:
                continue
            denom *= deltas[j] - dl
            nxt = [0.0] * (len(coef) + 1)
            for m, cm in enumerate(coef):
                nxt[m + 1] += cm
                nxt[m] -= cm * dl
            coef = nxt
        ws.append(sum(cm * im for cm, im in zip(coef, ints)) / denom)
    if self_check:
        # defining quadrature property, against independent Simpson integration
        for m in range(k):
            got = sum(w * (1.0 if m == 0 else d**m) for w, d in zip(ws, deltas))
            want = simpson_exp_moment(h, m)
            assert abs(got - want) < 1e-10 * max(abs(want), 1e-3), (deltas, h, m, got, want)
    return ws


def run_abnorsett(x_init, sigmas, record, order=4):
    x = list(x_init)
    history = []  # (lambda, x0), oldest first
    for i in range(len(sigmas) - 1):
        sigma, s_next = sigmas[i], sigmas[i + 1]
        record.append(list(x))
        x0 = toy_denoised(x, sigma)
        if sigma == 0.0:
            x = x0
            continue
        lam = -math.log(max(sigma, 1e-12))
        if history and history[-1][0] >= lam:
            history = []
        history.append((lam, x0))
        if len(history) > order:
            history.pop(0)
        if s_next == 0.0:
            x = x0
            continue
        h = -math.log(max(s_next, 1e-12)) - lam
        deltas = [l - lam for l, _ in reversed(history)]
        ws = exp_ab_weights(deltas, h)
        decay = math.exp(-h)
        nxt = [decay * v for v in x]
        for j, w in enumerate(ws):
            x0j = history[len(history) - 1 - j][1]
            nxt = [a + w * b for a, b in zip(nxt, x0j)]
        x = nxt
    return x


def main():
    verify_tableau()
    x_init = [f32(v) for v in (0.3, -1.1, 2.0, 0.05)]
    cases = {}

    # AC2: the exact 10-step rk6_7s reference-recipe sequence.
    sig10 = shift3_sigmas(10)
    rec = []
    final = run_rk6(x_init, sig10, rec)
    rec.append(final)
    assert len(rec) == 9 * 7 + 1 + 1 == 65, len(rec)
    cases["rk6_7s_10step"] = {"sigmas": sig10, "x_init": x_init, "trajectory": rec}

    # AC2: the exact 4-step abnorsett_4m reference-recipe sequence (startup orders 1,2,3 + terminal).
    sig4 = shift3_sigmas(4)
    rec = []
    final = run_abnorsett(x_init, sig4, rec)
    rec.append(final)
    assert len(rec) == 5, len(rec)
    cases["abnorsett_4m_4step"] = {"sigmas": sig4, "x_init": x_init, "trajectory": rec}

    # 8-step abnorsett so the full-order (k=4) interior steps genuinely engage.
    sig8 = shift3_sigmas(8)
    rec = []
    final = run_abnorsett(x_init, sig8, rec)
    rec.append(final)
    assert len(rec) == 9, len(rec)
    cases["abnorsett_4m_8step"] = {"sigmas": sig8, "x_init": x_init, "trajectory": rec}

    payload = {
        "comment": (
            "Float64 golden trajectories for the sc-20417 rk6_7s (Butcher 1964 6(7) tableau) and "
            "abnorsett_4m (4th-order exponential Adams-Bashforth / Norsett multistep) solvers. "
            "Generated by gen_rk6_abnorsett_golden.py (pure stdlib; self-verifies all rooted-tree "
            "order conditions and the exponential-quadrature weight property before writing). "
            "Trajectories record the latent at EVERY model call plus the final latent."
        ),
        **cases,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

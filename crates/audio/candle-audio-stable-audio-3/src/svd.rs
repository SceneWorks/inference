//! A deterministic host-side singular value decomposition, for the `-xs` adapter family.
//!
//! # Why this exists at all
//!
//! Four of the eight Stable Audio 3 adapter types (`lora-xs`, `dora-rows-xs`, `dora-cols-xs`,
//! `bora-xs`) reconstruct their delta as `U @ M @ Vᵀ`, where `U` and `V` are the top-`r` singular
//! vectors of the **base weight** rather than tensors stored in the adapter file. Upstream computes
//! them with `torch.linalg.svd`. **Candle has no `linalg.svd` on any backend**, so applying an
//! `-xs` adapter in this repository requires an SVD written here.
//!
//! # Determinism is the whole design constraint
//!
//! A singular vector is only defined up to sign. Upstream canonicalizes by signing each `U` column
//! by its largest-magnitude entry, and `V` must follow the same flip or the reconstructed delta is
//! *negated* on that rank — an adapter that silently does the opposite of what it was trained to do
//! rather than one that fails. So the sign rule is load-bearing, and a decomposition whose sign or
//! ordering depends on the accelerator is a correctness hazard, not a performance one.
//!
//! Two decisions follow, and they are the reason this module is plain `f64` slices rather than
//! [`candle_audio::candle_core::Tensor`] work:
//!
//! * **It never runs on the accelerator.** Every rotation is host arithmetic on `Vec<f64>`. There
//!   is no Metal/CUDA reduction whose summation order could vary, so CPU, CUDA and Metal execute
//!   *the identical instruction sequence* on the identical inputs. Cross-platform agreement is a
//!   property of the construction, not a bound to be measured.
//! * **It is exact, not iterative-approximate.** One-sided Jacobi converges to machine precision
//!   with a fixed cyclic sweep order and no random start, so there is no seed, no tolerance
//!   negotiation, and no spectral-gap dependence.
//!
//! Jacobi was chosen on **cost of implementation, not on determinism**, and this module should not
//! be read as claiming it is the only deterministic option — it is not. Golub–Kahan
//! bidiagonalization followed by implicit-shift QR is fully deterministic and asymptotically much
//! cheaper, and a blocked or vectorized Jacobi keeps this exact arithmetic while only reordering
//! it. What one-sided Jacobi bought was a short, dependency-free implementation whose sign
//! convention is easy to match to upstream's. Those deterministic-and-faster alternatives are
//! recorded on `sc-15551`. What *would* reintroduce the drift this module exists to avoid is the
//! **randomized** family, and Lanczos with a random start: there the result depends on a seed and
//! on the spectral gap.
//!
//! # The cost, stated plainly
//!
//! One-sided Jacobi is `O(sweeps · n² · m)` for an `m × n` input. That is genuinely expensive on a
//! 1.45B-parameter checkpoint and it is **not** amortized — see [`jacobi_svd_top_k`] for measured
//! numbers and the tracked follow-up. Only targets an `-xs` adapter actually matches are decomposed;
//! the other seven types never enter this module.

/// The largest number of cyclic Jacobi sweeps attempted before the decomposition is declared
/// converged regardless.
///
/// Well above the classical bound for double precision — one-sided Jacobi converges quadratically
/// and six to eight sweeps is typical — so in practice the sweep loop exits on its own
/// `!rotated` condition and this cap is never the thing that stops it. It exists so a pathological
/// input terminates rather than spins, and it is a **constant rather than an adaptive budget** so
/// the operation count is a function of the input shape alone: a cap that varied with the data
/// would make the result depend on how the data happened to be laid out.
const MAX_SWEEPS: usize = 30;

/// Off-diagonal tolerance, relative to the geometric mean of the two column norms.
///
/// `f64::EPSILON` is `2.22e-16`; this is deliberately a few ulps above it so the sweep terminates
/// instead of chasing rounding noise.
const TOLERANCE: f64 = 1e-15;

/// A rank-`k` truncated singular value decomposition, in host memory, column-major-free row-major
/// layout.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedSvd {
    /// `[m, k]` row-major: the top-`k` left singular vectors, sign-canonicalized.
    pub u: Vec<f64>,
    /// `[k]`: the top-`k` singular values, descending.
    pub s: Vec<f64>,
    /// `[n, k]` row-major: the top-`k` right singular vectors, carrying `u`'s sign flips.
    pub v: Vec<f64>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Decompose `a` (`[m, n]`, row-major) and return its top-`k` singular triplets.
///
/// # Sign canonicalization
///
/// Each returned `u` column is signed by its largest-magnitude entry: if that entry is negative the
/// whole column is negated **and so is the matching `v` column**, which leaves `u sᵀ vᵀ` unchanged
/// while making the pair unique. Ties in magnitude resolve to the **lowest row index**, and an
/// entry of exactly `0.0` is treated as non-negative (no flip) — both rules are arbitrary but must
/// be *fixed*, because the alternative is a column whose sign depends on iteration order.
///
/// # Ordering
///
/// Columns are sorted by singular value descending. Exactly-equal singular values keep their
/// pre-sort column order (the sort is stable), so a matrix with a repeated singular value still
/// decomposes identically on every run — even though the corresponding subspace is genuinely
/// rotation-ambiguous. That ambiguity is a property of the matrix, not of this implementation, and
/// no sign rule can remove it; see the module docs.
///
/// # Cost — measured, not estimated
///
/// Apple M-series, `--release`, `f64`, `k = 8`:
///
/// | shape | wall |
/// |---|---|
/// | `128 x 128` | 0.048 s |
/// | `256 x 256` | 0.99 s |
/// | `768 x 256` | 1.90 s |
/// | `512 x 512` | 10.9 s |
/// | `1024 x 1024` | 112.7 s |
///
/// The observed growth is roughly `n^3.4` — `n³` from the algorithm plus a slowly rising sweep
/// count. Truncating to `k` does **not** reduce it: one-sided Jacobi orthogonalizes every column
/// before any can be discarded.
///
/// The consequence, stated plainly rather than buried: `stable_audio_3_small_*` has a 1024-wide
/// DiT, so an `-xs` adapter covering its attention stack is a **multi-hour** cold start, and
/// `stable_audio_3_medium` (1536 x 24) is worse. The `-xs` math is correct and gated at every
/// scale; only the wall clock makes a full-DiT `-xs` adapter impractical today. The conditioner's
/// `[768, 256]` Linear, at 1.9 s, is comfortably practical and is what the real-weight `-xs` case
/// exercises. An accelerated **deterministic** truncated decomposition — or an explicit safe
/// precomputed-bases contract, which is what upstream's `--svd_bases_path` training flag caches —
/// is the fix, and is filed rather than hidden. Nothing here trades determinism for speed.
pub fn jacobi_svd_top_k(a: &[f64], m: usize, n: usize, k: usize) -> Result<TruncatedSvd, String> {
    if m == 0 || n == 0 {
        return Err(format!("svd needs a non-empty matrix, got {m}x{n}"));
    }
    if a.len() != m * n {
        return Err(format!(
            "svd input has {} elements, expected {m}x{n} = {}",
            a.len(),
            m * n
        ));
    }
    if k == 0 || k > m.min(n) {
        return Err(format!(
            "svd rank {k} is outside 1..={} for a {m}x{n} matrix",
            m.min(n)
        ));
    }

    // One-sided Jacobi rotates *columns*, so its per-sweep cost is quadratic in the column count
    // and only linear in the row count. Decomposing the transpose when the input is wide is a pure
    // win and changes nothing else: the transpose's left vectors are this matrix's right vectors.
    let (u, s, v) = if m >= n {
        one_sided_jacobi(a, m, n)
    } else {
        let at = transpose(a, m, n);
        let (ut, s, vt) = one_sided_jacobi(&at, n, m);
        (vt, s, ut)
    };

    // Both branches produce `u` as `[m, min(m, n)]` and `v` as `[n, min(m, n)]`, row-major, with
    // columns already sorted descending. Truncate to `k`, then canonicalize.
    let stride = m.min(n);
    let mut out_u = vec![0.0_f64; m * k];
    let mut out_v = vec![0.0_f64; n * k];
    let mut out_s = vec![0.0_f64; k];
    for j in 0..k {
        out_s[j] = s[j];
        // The largest-|·| entry of this `u` column decides the sign of the whole triplet. `>` (not
        // `>=`) makes a magnitude tie resolve to the lowest row index; a pivot of exactly `0.0` is
        // not negative, so it does not flip.
        let mut best = -1.0_f64;
        let mut pivot = 0.0_f64;
        for i in 0..m {
            let value = u[i * stride + j];
            if value.abs() > best {
                best = value.abs();
                pivot = value;
            }
        }
        let flip = pivot < 0.0;
        for i in 0..m {
            let value = u[i * stride + j];
            out_u[i * k + j] = if flip { -value } else { value };
        }
        for i in 0..n {
            let value = v[i * stride + j];
            out_v[i * k + j] = if flip { -value } else { value };
        }
    }

    Ok(TruncatedSvd {
        u: out_u,
        s: out_s,
        v: out_v,
        m,
        n,
        k,
    })
}

fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// One-sided Jacobi on a tall-or-square `[m, n]` matrix (`m >= n`).
///
/// Returns `(u, s, v)` with `u` `[m, n]` row-major, `s` `[n]` descending, `v` `[n, n]` row-major.
/// Rank-deficient columns (a singular value that rotates to exactly zero) produce a zero `u`
/// column; the caller's rank check keeps those out of the truncated result in every well-formed
/// case, and leaving them zero rather than fabricating an arbitrary orthogonal completion means a
/// degenerate input yields a zero delta rather than a plausible wrong one.
fn one_sided_jacobi(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert!(m >= n);
    let mut b = a.to_vec(); // `[m, n]`, mutated in place into `U * S`.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let mut app = 0.0_f64;
                let mut aqq = 0.0_f64;
                let mut apq = 0.0_f64;
                for i in 0..m {
                    let bp = b[i * n + p];
                    let bq = b[i * n + q];
                    app += bp * bp;
                    aqq += bq * bq;
                    apq += bp * bq;
                }
                if apq == 0.0 || app == 0.0 || aqq == 0.0 {
                    continue;
                }
                if apq.abs() <= TOLERANCE * (app * aqq).sqrt() {
                    continue;
                }
                rotated = true;
                // Classical two-sided-equivalent rotation for the 2x2 Gram block.
                let zeta = (aqq - app) / (2.0 * apq);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let bp = b[i * n + p];
                    let bq = b[i * n + q];
                    b[i * n + p] = c * bp - s * bq;
                    b[i * n + q] = s * bp + c * bq;
                }
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // Columns of `b` are now mutually orthogonal; their norms are the singular values.
    let mut s = vec![0.0_f64; n];
    for j in 0..n {
        let mut acc = 0.0_f64;
        for i in 0..m {
            let value = b[i * n + j];
            acc += value * value;
        }
        s[j] = acc.sqrt();
    }

    // Stable descending sort by singular value; ties keep their original column index.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        s[y].partial_cmp(&s[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });

    let mut u_out = vec![0.0_f64; m * n];
    let mut v_out = vec![0.0_f64; n * n];
    let mut s_out = vec![0.0_f64; n];
    for (target, &source) in order.iter().enumerate() {
        let sigma = s[source];
        s_out[target] = sigma;
        if sigma > 0.0 {
            for i in 0..m {
                u_out[i * n + target] = b[i * n + source] / sigma;
            }
        }
        for i in 0..n {
            v_out[i * n + target] = v[i * n + source];
        }
    }
    (u_out, s_out, v_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(svd: &TruncatedSvd) -> Vec<f64> {
        let mut out = vec![0.0_f64; svd.m * svd.n];
        for i in 0..svd.m {
            for j in 0..svd.n {
                let mut acc = 0.0;
                for r in 0..svd.k {
                    acc += svd.u[i * svd.k + r] * svd.s[r] * svd.v[j * svd.k + r];
                }
                out[i * svd.n + j] = acc;
            }
        }
        out
    }

    fn deterministic_matrix(m: usize, n: usize) -> Vec<f64> {
        // A fixed, non-symmetric, full-rank-ish fill with no repeated singular values.
        (0..m * n)
            .map(|idx| {
                let i = (idx / n) as f64;
                let j = (idx % n) as f64;
                ((i + 1.0) * 0.37).sin() * ((j + 1.0) * 0.91).cos() + (i - j) * 0.013
            })
            .collect()
    }

    #[test]
    fn a_full_rank_decomposition_reconstructs_the_input() {
        for (m, n) in [(6, 4), (4, 6), (5, 5), (1, 3), (3, 1)] {
            let a = deterministic_matrix(m, n);
            let k = m.min(n);
            let svd = jacobi_svd_top_k(&a, m, n, k).expect("decompose");
            let back = reconstruct(&svd);
            for (index, (expected, got)) in a.iter().zip(back.iter()).enumerate() {
                assert!(
                    (expected - got).abs() < 1e-10,
                    "{m}x{n} element {index}: {expected} vs {got}"
                );
            }
        }
    }

    #[test]
    fn singular_values_are_descending_and_match_the_gram_spectrum() {
        let (m, n) = (7, 5);
        let a = deterministic_matrix(m, n);
        let svd = jacobi_svd_top_k(&a, m, n, n).expect("decompose");
        for pair in svd.s.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "singular values not descending: {:?}",
                svd.s
            );
        }
        // Sum of squared singular values is the Frobenius norm squared.
        let frobenius: f64 = a.iter().map(|value| value * value).sum();
        let spectrum: f64 = svd.s.iter().map(|value| value * value).sum();
        assert!(
            (frobenius - spectrum).abs() < 1e-9,
            "frobenius {frobenius} vs spectrum {spectrum}"
        );
    }

    #[test]
    fn u_and_v_columns_are_orthonormal() {
        let (m, n) = (9, 6);
        let a = deterministic_matrix(m, n);
        let svd = jacobi_svd_top_k(&a, m, n, n).expect("decompose");
        for p in 0..n {
            for q in 0..n {
                let mut du = 0.0;
                let mut dv = 0.0;
                for i in 0..m {
                    du += svd.u[i * n + p] * svd.u[i * n + q];
                }
                for i in 0..n {
                    dv += svd.v[i * n + p] * svd.v[i * n + q];
                }
                let expected = if p == q { 1.0 } else { 0.0 };
                assert!((du - expected).abs() < 1e-10, "u[{p}]·u[{q}] = {du}");
                assert!((dv - expected).abs() < 1e-10, "v[{p}]·v[{q}] = {dv}");
            }
        }
    }

    /// The sign rule is the whole reason this module is not a thin wrapper: a flipped `u` column
    /// that `v` does not follow negates that rank's contribution to the delta, which is a
    /// *misapplied* adapter rather than a failed one.
    ///
    /// This asserts the rule directly on the returned vectors, so it is blind to nothing: the
    /// largest-magnitude entry of every `u` column must be non-negative.
    #[test]
    fn every_u_column_is_signed_by_its_largest_magnitude_entry() {
        for (m, n) in [(8, 5), (5, 8), (6, 6)] {
            let a = deterministic_matrix(m, n);
            let k = m.min(n);
            let svd = jacobi_svd_top_k(&a, m, n, k).expect("decompose");
            for j in 0..k {
                let mut best = -1.0_f64;
                let mut pivot = 0.0_f64;
                for i in 0..m {
                    let value = svd.u[i * k + j];
                    if value.abs() > best {
                        best = value.abs();
                        pivot = value;
                    }
                }
                assert!(
                    pivot >= 0.0,
                    "{m}x{n} column {j} pivot is negative ({pivot}); canonicalization did not run"
                );
            }
        }
    }

    /// Negating a *row* of the input flips the sign of some `u` entries. The canonicalization must
    /// still hold, and — this is the discriminating half — the reconstruction must still be exact,
    /// which is what proves `v` followed `u`'s flip instead of being canonicalized independently.
    #[test]
    fn a_sign_flip_that_v_does_not_follow_would_break_the_reconstruction() {
        let (m, n) = (6, 4);
        let mut a = deterministic_matrix(m, n);
        for j in 0..n {
            a[2 * n + j] *= -1.0;
        }
        let svd = jacobi_svd_top_k(&a, m, n, n).expect("decompose");
        let back = reconstruct(&svd);
        for (expected, got) in a.iter().zip(back.iter()) {
            assert!((expected - got).abs() < 1e-10, "{expected} vs {got}");
        }
        for j in 0..n {
            let mut best = -1.0;
            let mut pivot = 0.0;
            for i in 0..m {
                let value = svd.u[i * n + j];
                if value.abs() > best {
                    best = value.abs();
                    pivot = value;
                }
            }
            assert!(pivot >= 0.0);
        }
    }

    /// Determinism, asserted as bit equality rather than as a bound. Same input, two calls, and a
    /// third from a transposed-then-untransposed route that exercises the wide branch.
    #[test]
    fn repeated_decompositions_are_bit_identical() {
        let (m, n) = (7, 4);
        let a = deterministic_matrix(m, n);
        let first = jacobi_svd_top_k(&a, m, n, 3).expect("decompose");
        let second = jacobi_svd_top_k(&a, m, n, 3).expect("decompose");
        assert_eq!(first, second, "two decompositions of one input disagree");

        let wide = deterministic_matrix(4, 7);
        let a_wide = jacobi_svd_top_k(&wide, 4, 7, 3).expect("decompose");
        let b_wide = jacobi_svd_top_k(&wide, 4, 7, 3).expect("decompose");
        assert_eq!(a_wide, b_wide);
    }

    #[test]
    fn truncation_keeps_the_leading_subspace() {
        let (m, n) = (8, 6);
        let a = deterministic_matrix(m, n);
        let full = jacobi_svd_top_k(&a, m, n, 6).expect("decompose");
        let truncated = jacobi_svd_top_k(&a, m, n, 2).expect("decompose");
        assert_eq!(truncated.k, 2);
        for r in 0..2 {
            assert!((full.s[r] - truncated.s[r]).abs() < 1e-12);
            for i in 0..m {
                assert!((full.u[i * 6 + r] - truncated.u[i * 2 + r]).abs() < 1e-12);
            }
            for i in 0..n {
                assert!((full.v[i * 6 + r] - truncated.v[i * 2 + r]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn malformed_shapes_are_refused() {
        assert!(jacobi_svd_top_k(&[1.0, 2.0], 0, 2, 1).is_err());
        assert!(jacobi_svd_top_k(&[1.0, 2.0], 1, 2, 3).is_err());
        assert!(jacobi_svd_top_k(&[1.0, 2.0], 1, 2, 0).is_err());
        assert!(jacobi_svd_top_k(&[1.0, 2.0, 3.0], 2, 2, 1).is_err());
    }
}

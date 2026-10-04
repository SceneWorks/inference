//! Independent f64 Kronecker oracle and componentwise bf16 forward-error budgets.
//! MLX Metal bf16 GEMMs accumulate in f32 and round their output to bf16. Each
//! budget multiplies the sum of absolute products, so cancellation cannot make
//! a whole-model peak an accidentally permissive or impossible residual bound.

pub fn rounding_fraction(c: usize, d: usize, structured: bool) -> f64 {
    let u = 2f64.powi(-8); // bf16 round-to-nearest unit roundoff
    let u32 = 2f64.powi(-24);
    let gamma = |n: usize| n as f64 * u32 / (1.0 - n as f64 * u32);
    // Dense: one rounded delta and one rounded GEMM output.
    // Structured: two rounded factors and two rounded GEMM outputs.
    // Three f32 operations also cover kron/scale versus the scale-first form.
    let reductions = if structured {
        (1.0 + gamma(c)) * (1.0 + gamma(d))
    } else {
        1.0 + gamma(c * d)
    };
    (1.0 + u).powi(if structured { 4 } else { 2 }) * reductions * (1.0 + gamma(3)) - 1.0
}

pub fn oracle(
    w1: &[f32],
    w2: &[f32],
    x: &[f32],
    shape: [usize; 4],
    scale: f32,
) -> (Vec<f64>, Vec<f64>) {
    let [a, b, c, d] = shape;
    let n = x.len() / (c * d);
    assert_eq!(w1.len(), a * c);
    assert_eq!(w2.len(), b * d);
    let mut values = vec![0.0; n * a * b];
    let mut absolute_products = values.clone();
    for row in 0..n {
        for p in 0..a {
            for q in 0..b {
                let out = row * a * b + p * b + q;
                for r in 0..c {
                    for s in 0..d {
                        let product = w1[p * c + r] as f64
                            * w2[q * d + s] as f64
                            * x[row * c * d + r * d + s] as f64
                            * scale as f64;
                        values[out] += product;
                        absolute_products[out] += product.abs();
                    }
                }
            }
        }
    }
    (values, absolute_products)
}

pub fn within_bound(got: &[f32], want: &[f64], products: &[f64], fraction: f64) -> bool {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).zip(products).all(|((&g, &w), &p)| {
        g.is_finite() && (g as f64 - w).abs() <= fraction * p + f32::MIN_POSITIVE as f64
    })
}

#[test]
fn componentwise_budget_rejects_scale_drop_and_transpose() {
    let w1 = [0.137, -0.219, 0.473, 0.311];
    let w2 = [0.417, 0.239, -0.527, 0.163];
    let x = [0.75, -0.25, 0.5, 0.125];
    let (want, products) = oracle(&w1, &w2, &x, [2, 2, 2, 2], 0.625);
    let fraction = rounding_fraction(2, 2, true);
    let correct: Vec<_> = want.iter().map(|&v| v as f32).collect();
    assert!(within_bound(&correct, &want, &products, fraction));
    for mutant in [
        vec![0.0; 4],
        oracle(&w1, &w2, &x, [2, 2, 2, 2], 1.0)
            .0
            .iter()
            .map(|&v| v as f32)
            .collect(),
        oracle(&[w1[0], w1[2], w1[1], w1[3]], &w2, &x, [2, 2, 2, 2], 0.625)
            .0
            .iter()
            .map(|&v| v as f32)
            .collect(),
    ] {
        assert!(!within_bound(&mutant, &want, &products, fraction));
    }
}

#[test]
fn cancellation_uses_absolute_products_instead_of_output_peak() {
    let (want, products) = oracle(&[1.0], &[1.0, -1.0], &[1.0, 1.0], [1, 1, 1, 2], 1.0);
    assert_eq!(want, [0.0]);
    assert_eq!(products, [2.0]);
    assert!(within_bound(
        &[0.0078125],
        &want,
        &products,
        rounding_fraction(1, 2, true)
    ));
    assert!(!within_bound(
        &[0.5],
        &want,
        &products,
        rounding_fraction(1, 2, true)
    ));
}

#[test]
fn materialized_and_structured_bf16_are_distinct_within_their_budgets() {
    // A CPU arithmetic counterexample to treating both bf16 representations as
    // byte-identical. The native gate measures actual Metal residuals separately.
    let bf16 =
        |x: f32| f32::from_bits((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000);
    let w1 = [0.137f32, -0.219, 0.473, 0.311];
    let w2 = [0.417f32, 0.239, -0.527, 0.163];
    let x = [0.75f32, -0.25, 0.5, 0.125];
    let scale = 0.625;
    let mut dense = vec![0.0; 4];
    let mut structured = dense.clone();
    for p in 0..2 {
        let mut intermediate = [0.0; 2];
        for s in 0..2 {
            intermediate[s] = bf16((0..2).map(|r| bf16(w1[p * 2 + r]) * x[r * 2 + s]).sum());
        }
        for q in 0..2 {
            dense[p * 2 + q] = bf16(
                (0..2)
                    .flat_map(|r| {
                        (0..2).map(move |s| {
                            x[r * 2 + s] * bf16(w1[p * 2 + r] * w2[q * 2 + s] * scale)
                        })
                    })
                    .sum(),
            );
            structured[p * 2 + q] = bf16(
                (0..2)
                    .map(|s| intermediate[s] * bf16(w2[q * 2 + s] * scale))
                    .sum(),
            );
        }
    }
    assert_ne!(
        dense, structured,
        "the arithmetic counterexample must discriminate representations"
    );
    let (want, products) = oracle(&w1, &w2, &x, [2, 2, 2, 2], scale);
    assert!(within_bound(
        &dense,
        &want,
        &products,
        rounding_fraction(2, 2, false)
    ));
    assert!(within_bound(
        &structured,
        &want,
        &products,
        rounding_fraction(2, 2, true)
    ));
}

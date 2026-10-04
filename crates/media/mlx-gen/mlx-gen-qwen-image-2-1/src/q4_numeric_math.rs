//! Tensor-free arithmetic and allocation checks for the test-only Q4 diagnostic.
pub const BASE_RESIDENT: u64 = 10_505_507_280;
pub const FACTOR_ALLOWANCE: u64 = 20_278_251;
pub const TRANSIENT: u64 = 11_142_168_576;
pub const EXPECTED_ELEMENTS: u64 = 6_979_321_856;
pub const EXPECTED_TARGETS: usize = 224;

#[derive(Clone, Copy, Debug)]
pub struct Pricing {
    pub elements: u64,
    pub bf16_delta: u64,
    pub f32_widening: u64,
    pub construction_scratch: u64,
    pub resident: u64,
    pub active: u64,
    pub physical: u64,
}

pub fn price(shapes: &[(u64, u64)]) -> Result<Pricing, String> {
    if shapes.len() != EXPECTED_TARGETS
        || shapes
            .iter()
            .any(|&(o, i)| !matches!((o, i), (4096, 4096) | (12288, 4096) | (4096, 12288)))
    {
        return Err("numeric donor has unexpected target count/geometry".into());
    }
    let elements: u64 = shapes.iter().map(|&(o, i)| o * i).sum();
    if elements != EXPECTED_ELEMENTS {
        return Err("numeric donor differs from frozen224-target geometry".into());
    }
    let largest = shapes.iter().map(|&(o, i)| o * i).max().unwrap();
    let bf16_delta = elements * 2;
    let f32_widening = elements * 4;
    // Pinned mlx-rs48ff5e78 builds MLX0.32.0: transforms.cpp permits10
    // active tasks before finalizing/waiting. Eleven slots remain conservative;
    // the alternate installer ALSO synchronizes the construction stream and
    // clears free cache per target, so this is not an all-stream queue claim.
    let construction_scratch = 11 * largest * (4 + 4 + 2);
    let resident = BASE_RESIDENT + FACTOR_ALLOWANCE + bf16_delta + f32_widening;
    let active = resident + construction_scratch + TRANSIENT;
    Ok(Pricing {
        elements,
        bf16_delta,
        f32_widening,
        construction_scratch,
        resident,
        active,
        physical: active + TRANSIENT,
    })
}

pub fn gamma(terms: usize) -> f64 {
    let nu = terms as f64 * (2f64).powi(-24);
    assert!(nu < 1.0);
    nu / (1.0 - nu)
}

/// A componentwise floating point dot-product bound against exact f64 products.
/// The absolute term covers subnormal rounding, not a fitted model tolerance.
pub fn dot_bound(terms: usize, absolute_products: f64) -> f64 {
    gamma(terms + 1) * absolute_products + (terms + 1) as f64 * f32::MIN_POSITIVE as f64
}

pub fn within(got: f64, expected: f64, bound: f64) -> bool {
    got.is_finite() && expected.is_finite() && bound.is_finite() && (got - expected).abs() <= bound
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shapes() -> Vec<(u64, u64)> {
        [(4096, 4096, 128), (12288, 4096, 64), (4096, 12288, 32)]
            .into_iter()
            .flat_map(|(o, i, n)| std::iter::repeat_n((o, i), n))
            .collect()
    }
    #[test]
    fn materialization_widening_retirement_and_free_cache_are_all_priced() {
        let p = price(&shapes()).unwrap();
        assert_eq!(p.bf16_delta, 13_958_643_712);
        assert_eq!(p.f32_widening, 27_917_287_424);
        assert_eq!(p.construction_scratch, 5_536_481_280);
        assert_eq!(p.resident, 52_401_716_667);
        assert_eq!(p.active, 69_080_366_523);
        assert_eq!(p.physical, 80_222_535_099);
        // Omitting widening, retirement or cache cannot reproduce the frozen envelope.
        for missing in [p.f32_widening, p.construction_scratch, TRANSIENT] {
            assert_ne!(p.physical - missing, 80_222_535_099);
        }
        let mut altered = shapes();
        altered.pop();
        assert!(price(&altered).is_err());
        let mut altered = shapes();
        altered[0] = (4096, 4095);
        assert!(price(&altered).is_err());
    }
    #[test]
    fn f32_component_bound_rejects_drop_scale_and_axis_mutants() {
        let x = [0.25f32, -0.5, 0.75, 1.0];
        let w = [0.125f32, 0.25, -0.5, 0.75];
        let expected: f64 = x.iter().zip(w).map(|(&a, b)| a as f64 * b as f64).sum();
        let products: f64 = x
            .iter()
            .zip(w)
            .map(|(&a, b)| (a as f64 * b as f64).abs())
            .sum();
        let actual: f32 = x.iter().zip(w).map(|(&a, b)| a * b).sum();
        let bound = dot_bound(4, products);
        assert!(within(actual as f64, expected, bound));
        assert!(!within(0.0, expected, bound));
        assert!(!within(actual as f64 * 0.5, expected, bound));
        let wrong_axis: f32 = x.iter().rev().zip(w).map(|(&a, b)| a * b).sum();
        assert!(!within(wrong_axis as f64, expected, bound));
    }
}

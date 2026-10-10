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

/// Apple MPP relaxed_precision permits input truncation without specifying its
/// width/rounding. Finite discrepancies are observations, never a bound verdict.
pub const ARITHMETIC_VERDICT: &str = "UNPROVEN_RELAXED_NAX_PRECISION";
#[derive(Clone, Copy, Debug)]
pub struct ArithmeticObservation {
    pub actual: f64,
    pub reference: f64,
    pub absolute_difference: f64,
    pub verdict: &'static str,
}
pub fn observe(actual: f64, reference: f64) -> Result<ArithmeticObservation, String> {
    let absolute_difference = (actual - reference).abs();
    if !actual.is_finite() || !reference.is_finite() || !absolute_difference.is_finite() {
        return Err("nonfinite numerical observation".into());
    }
    Ok(ArithmeticObservation {
        actual,
        reference,
        absolute_difference,
        verdict: ARITHMETIC_VERDICT,
    })
}

pub struct ExportSchema {
    pub factors: Vec<(String, String)>, // serialized key, exact trained key
    pub alphas: Vec<String>,
}

/// The LoRA writer adds one scalar alpha tensor per target. LoKr stores alpha
/// only in string metadata. Neither kind may add/drop any trained factor.
pub fn export_schema(
    params: &[String],
    paths: &[String],
    lora: bool,
    saved: &[String],
) -> Result<ExportSchema, String> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut factors = BTreeMap::new();
    for key in params {
        let serialized = if lora {
            if let Some(path) = key.strip_suffix(".lora_a") {
                format!("{path}.lora_A.weight")
            } else if let Some(path) = key.strip_suffix(".lora_b") {
                format!("{path}.lora_B.weight")
            } else {
                return Err(format!("unexpected trained LoRA key {key}"));
            }
        } else {
            key.clone()
        };
        if factors.insert(serialized, key.clone()).is_some() {
            return Err("duplicate trained factor key".into());
        }
    }
    let alphas: Vec<_> = if lora {
        paths.iter().map(|p| format!("{p}.alpha")).collect()
    } else {
        vec![]
    };
    let expected: BTreeSet<_> = factors.keys().chain(alphas.iter()).cloned().collect();
    let actual: BTreeSet<_> = saved.iter().cloned().collect();
    if expected.len() != factors.len() + alphas.len()
        || actual.len() != saved.len()
        || actual != expected
    {
        return Err("serialized factor/alpha keys differ from exact export schema".into());
    }
    Ok(ExportSchema {
        factors: factors.into_iter().collect(),
        alphas,
    })
}

pub fn validate_alpha(
    actual: f32,
    expected: f32,
    shape: &[i32],
    f32_dtype: bool,
) -> Result<(), String> {
    if !f32_dtype || shape != [1] || !actual.is_finite() || actual != expected {
        return Err("exported scalar alpha value/shape/dtype differs from training config".into());
    }
    Ok(())
}

pub fn validate_export_metadata(
    network: &str,
    rank: u32,
    alpha: f32,
    actual: (Option<&str>, Option<&str>, Option<&str>),
) -> Result<(), String> {
    if actual
        != (
            Some(network),
            Some(rank.to_string().as_str()),
            Some(alpha.to_string().as_str()),
        )
    {
        return Err("exported network/rank/alpha metadata differs from training config".into());
    }
    Ok(())
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
    fn relaxed_gemm_discrepancies_never_claim_a_proven_bound() {
        for (actual, reference, difference) in
            [(1.0, 1.0, 0.0), (1.125, 1.0, 0.125), (-2.0, 3.0, 5.0)]
        {
            let observation = observe(actual, reference).unwrap();
            assert_eq!(observation.actual, actual);
            assert_eq!(observation.reference, reference);
            assert_eq!(observation.absolute_difference, difference);
            assert_eq!(observation.verdict, "UNPROVEN_RELAXED_NAX_PRECISION");
            assert_ne!(observation.verdict, "PASS");
        }
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(observe(invalid, 1.0).is_err());
            assert!(observe(1.0, invalid).is_err());
        }
        assert!(observe(f64::MAX, -f64::MAX).is_err());
    }
    #[test]
    fn exact_export_schema_distinguishes_alphas_and_rejects_drop_extra_or_renamed_factors() {
        let params = vec!["block.lora_a".into(), "block.lora_b".into()];
        let paths = vec!["block".into()];
        let saved = vec![
            "block.lora_A.weight".into(),
            "block.lora_B.weight".into(),
            "block.alpha".into(),
        ];
        let schema = export_schema(&params, &paths, true, &saved).unwrap();
        assert_eq!(schema.factors.len(), 2);
        assert_eq!(schema.alphas, ["block.alpha"]);
        assert!(schema.factors.iter().all(|(_, key)| params.contains(key)));
        for dropped in 0..saved.len() {
            let mut altered = saved.clone();
            altered.remove(dropped);
            assert!(export_schema(&params, &paths, true, &altered).is_err());
        }
        let mut altered = saved.clone();
        altered.push("block.unexpected_param".into());
        assert!(export_schema(&params, &paths, true, &altered).is_err());
        let mut altered = saved.clone();
        altered[0] = "other.lora_A.weight".into();
        assert!(export_schema(&params, &paths, true, &altered).is_err());
        let mut altered = saved.clone();
        altered.push(saved[0].clone());
        assert!(export_schema(&params, &paths, true, &altered).is_err());
        let params = vec!["block.lokr_w1".into(), "block.lokr_w2".into()];
        assert!(export_schema(&params, &paths, false, &params)
            .unwrap()
            .alphas
            .is_empty());
        let mut altered = params.clone();
        altered.push("block.alpha".into());
        assert!(export_schema(&params, &paths, false, &altered).is_err());
    }
    #[test]
    fn alpha_contract_rejects_wrong_value_shape_dtype_and_nonfinite() {
        assert!(validate_alpha(2.5, 2.5, &[1], true).is_ok());
        assert!(validate_alpha(2.0, 2.5, &[1], true).is_err());
        assert!(validate_alpha(2.5, 2.5, &[1, 1], true).is_err());
        assert!(validate_alpha(2.5, 2.5, &[1], false).is_err());
        assert!(validate_alpha(f32::NAN, 2.5, &[1], true).is_err());
        assert!(
            validate_export_metadata("lora", 4, 2.5, (Some("lora"), Some("4"), Some("2.5")))
                .is_ok()
        );
        for metadata in [
            (Some("lokr"), Some("4"), Some("2.5")),
            (Some("lora"), Some("16"), Some("2.5")),
            (Some("lora"), Some("4"), Some("2")),
            (Some("lora"), Some("4"), None),
        ] {
            assert!(validate_export_metadata("lora", 4, 2.5, metadata).is_err());
        }
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

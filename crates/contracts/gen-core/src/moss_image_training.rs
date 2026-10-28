//! Portable MOSS adapter contract shared by native backends. The named sampler adapts the
//! published *pretraining* final Stage3 recipe (arXiv:2610.23374v1, Appendix A.2 Eq4), not an
//! unpublished official LoRA/SFT recipe. Posterior-mean caching is a SceneWorks choice.
pub mod admission;
pub mod recovery;

use crate::moss_image::MossImageVariant;
use crate::{Error, Result};
use std::collections::BTreeMap;

pub const FORMAT: &str = "moss_transformer_adapter_v1";
pub const SAMPLER: &str = "moss_stage3_v1";
pub const POSTERIOR: &str = "posterior_mean_v1";
pub const PROJECTIONS: [&str; 6] = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"];

pub fn target_paths(layers: usize, suffixes: &[String]) -> Result<Vec<String>> {
    let all: Vec<_> = (0..layers)
        .flat_map(|i| {
            PROJECTIONS
                .into_iter()
                .map(move |p| format!("layers.{i}.proj.{p}"))
        })
        .collect();
    for suffix in suffixes {
        if !all
            .iter()
            .any(|p| p == suffix || p.ends_with(&format!(".{suffix}")))
        {
            return Err(Error::Unsupported(format!(
                "MOSS target {suffix:?} matches no transformer projection"
            )));
        }
    }
    Ok(all
        .into_iter()
        .filter(|p| {
            suffixes.is_empty()
                || suffixes
                    .iter()
                    .any(|s| p == s || p.ends_with(&format!(".{s}")))
        })
        .collect())
}

pub fn metadata(variant: MossImageVariant, sampler: &str) -> BTreeMap<String, String> {
    [
        ("sceneworks_adapter_format", FORMAT),
        ("model_family", "moss_image"),
        ("family", "moss-image"),
        ("baseModel", variant.provider_id()),
        ("base_model", variant.provider_id()),
        ("base_repository", variant.repository()),
        ("base_revision", variant.revision()),
        ("shared_revision", crate::moss_image::SFT_REVISION),
        ("vae_posterior", POSTERIOR),
        ("training_sampler", sampler),
        ("training_objective", "unweighted_flow_velocity_v1"),
        ("adapter_compute", "fp32_residual_v1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

pub fn validate_metadata(variant: MossImageVariant, meta: &BTreeMap<String, String>) -> Result<()> {
    for (key, expected) in metadata(variant, "") {
        if key == "training_sampler" {
            continue;
        }
        if meta.get(&key) != Some(&expected) {
            return Err(Error::Unsupported(format!("MOSS adapter {key} must be {expected:?}; use an adapter trained for this exact checkpoint")));
        }
    }
    if meta.get("training_sampler").is_none_or(|s| s.is_empty()) {
        return Err(Error::Unsupported(
            "MOSS adapter has no sampler provenance".into(),
        ));
    }
    Ok(())
}

// Abramowitz-Stegun normal CDF (absolute error < 8e-8). The table is normalized at a=.95,
// rescaled to all 1000 bins, then mixed with rho=.01 uniform mass; never clip a draw at .95.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-0.5 * x * x).exp();
    let tail = d
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}
fn stage3_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    1.0 - normal_cdf((((1.0 - x) / x).ln() - (-1.09)) / 0.95)
}
/// Inverse CDF for sigma_j=j/1000, j=1..1000. `unit` is a uniform draw in [0,1).
pub fn stage3_sigma(unit: f64) -> Result<f32> {
    if !unit.is_finite() || !(0.0..1.0).contains(&unit) {
        return Err(Error::Unsupported(
            "MOSS sampler draw must lie in [0,1)".into(),
        ));
    }
    let denominator = stage3_cdf(0.95);
    for j in 1..=1000 {
        let cumulative =
            0.99 * stage3_cdf(0.95 * j as f64 / 1000.0) / denominator + 0.01 * j as f64 / 1000.0;
        if unit < cumulative || j == 1000 {
            return Ok(j as f32 / 1000.0);
        }
    }
    unreachable!()
}
/// Host-only SplitMix draw, identical across backend/device and independent of other RNG use.
pub fn unit_draw(seed: u64) -> f64 {
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x >> 11) as f64) / 9007199254740992.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_parent_and_transformer_only_selection() {
        let paths = target_paths(42, &[]).unwrap();
        assert_eq!(paths.len(), 252);
        assert_eq!(target_paths(42, &["ffn_up_gate".into()]).unwrap().len(), 42);
        assert!(target_paths(42, &["text_encoder.q_proj".into()]).is_err());
        assert!(target_paths(42, &["q".into(), "typo".into()]).is_err());
        let meta = metadata(MossImageVariant::Base, SAMPLER);
        validate_metadata(MossImageVariant::Base, &meta).unwrap();
        assert!(validate_metadata(MossImageVariant::Sft, &meta).is_err());
        let mut bad = meta.clone();
        bad.insert("base_revision".into(), "wrong".into());
        assert!(validate_metadata(MossImageVariant::Base, &bad).is_err());
    }
    #[test]
    fn stage3_is_discrete_rescaled_truncated_mixture_not_generic_logit_normal() {
        assert_eq!(stage3_sigma(0.0).unwrap(), 0.001);
        assert_eq!(stage3_sigma(1.0 - f64::EPSILON).unwrap(), 1.0);
        assert!(stage3_sigma(1.0).is_err());
        assert!(stage3_sigma(f64::NAN).is_err());
        // Independent Eq4 cumulative probability at bin750, using high-precision CDF values.
        // F(.7125)=.433197471807974, F(.95)=.974535402131586.
        let boundary = 0.99 * 0.42385339658609217 / 0.9745333332861987 + 0.0075;
        assert_eq!(stage3_sigma(boundary - 1e-5).unwrap(), 0.750);
        assert_eq!(stage3_sigma(boundary + 1e-5).unwrap(), 0.751);
        let a = stage3_sigma(unit_draw(17)).unwrap();
        assert_eq!(a, stage3_sigma(unit_draw(17)).unwrap());
        assert_ne!(a, stage3_sigma(unit_draw(18)).unwrap());
        assert!((a * 1000.0 - (a * 1000.0).round()).abs() < 0.0001);
    }
}

/// Required portable LoKr scaling metadata. Unlike permissive third-party parsing, an invalid
/// rank/alpha cannot silently change the strength of a SceneWorks-native adapter.
pub fn lokr_scaling(meta: &BTreeMap<String, String>) -> Result<(f32, f32)> {
    let rank = meta
        .get("rank")
        .ok_or_else(|| Error::Unsupported("MOSS LoKr metadata is missing rank".into()))?
        .parse::<u32>()
        .map_err(|_| Error::Unsupported("MOSS LoKr rank must be a positive integer".into()))?;
    let alpha = meta
        .get("alpha")
        .ok_or_else(|| Error::Unsupported("MOSS LoKr metadata is missing alpha".into()))?
        .parse::<f32>()
        .map_err(|_| Error::Unsupported("MOSS LoKr alpha must be finite".into()))?;
    if rank == 0 || !alpha.is_finite() {
        return Err(Error::Unsupported(
            "MOSS LoKr requires positive rank and finite alpha".into(),
        ));
    }
    Ok((rank as f32, alpha))
}

/// Logical dimensions of one Kronecker factor; decomposed factors preserve both raw matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LokrFactorShape {
    pub rows: usize,
    pub cols: usize,
    pub inner: Option<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LokrLayout {
    pub first: LokrFactorShape,
    pub second: LokrFactorShape,
}
/// Exactly one complete representation per factor, with explicit row AND column compatibility.
/// Equal element counts never justify reshaping a crossed Kronecker geometry.
pub fn lokr_layout(
    factors: &BTreeMap<String, Vec<usize>>,
    output: usize,
    input: usize,
) -> Result<LokrLayout> {
    if factors.keys().any(|k| {
        ![
            "lokr_w1",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2",
            "lokr_w2_a",
            "lokr_w2_b",
        ]
        .contains(&k.as_str())
    }) {
        return Err(Error::Unsupported(
            "MOSS LoKr has an unconsumed factor".into(),
        ));
    }
    let factor = |prefix: &str| -> Result<LokrFactorShape> {
        let full = factors.get(prefix);
        let a = factors.get(&format!("{prefix}_a"));
        let b = factors.get(&format!("{prefix}_b"));
        let matrix = |shape: &Vec<usize>| shape.len() == 2 && shape.iter().all(|d| *d > 0);
        match (full,a,b) {
            (Some(w),None,None) if matrix(w)=>Ok(LokrFactorShape{rows:w[0],cols:w[1],inner:None}),
            (None,Some(a),Some(b)) if matrix(a)&&matrix(b)&&a[1]==b[0]=>Ok(LokrFactorShape{rows:a[0],cols:b[1],inner:Some(a[1])}),
            _=>Err(Error::Unsupported(format!("MOSS {prefix} requires exactly full OR a complete shape-compatible a/b pair; orphan/mixed factors are not applied"))),
        }
    };
    let first = factor("lokr_w1")?;
    let second = factor("lokr_w2")?;
    if first.rows.checked_mul(second.rows) != Some(output)
        || first.cols.checked_mul(second.cols) != Some(input)
    {
        return Err(Error::Unsupported(format!(
            "MOSS LoKr reconstructed axes do not match [{output},{input}]"
        )));
    }
    Ok(LokrLayout { first, second })
}

#[cfg(test)]
mod review_validation_tests {
    use super::*;
    fn valid() -> BTreeMap<String, Vec<usize>> {
        [
            ("lokr_w1".into(), vec![2, 2]),
            ("lokr_w2".into(), vec![1536, 1536]),
        ]
        .into_iter()
        .collect()
    }
    #[test]
    fn lokr_validates_both_axes_not_equal_element_count() {
        lokr_layout(&valid(), 3072, 3072).unwrap();
        let mut crossed = valid();
        crossed.insert("lokr_w2".into(), vec![768, 3072]);
        assert!(lokr_layout(&crossed, 3072, 3072).is_err());
    }
    #[test]
    fn every_kronecker_factor_has_exactly_one_complete_representation() {
        for factor in ["lokr_w1", "lokr_w2"] {
            let mut mixed = valid();
            let shape = mixed[factor].clone();
            mixed.insert(format!("{factor}_a"), vec![shape[0], 1]);
            assert!(
                lokr_layout(&mixed, 3072, 3072).is_err(),
                "orphan {factor}_a"
            );
            mixed.insert(format!("{factor}_b"), vec![1, shape[1]]);
            assert!(
                lokr_layout(&mixed, 3072, 3072).is_err(),
                "competing full and decomposed {factor}"
            );
            mixed.remove(factor);
            lokr_layout(&mixed, 3072, 3072).unwrap();
            mixed.remove(&format!("{factor}_a"));
            assert!(
                lokr_layout(&mixed, 3072, 3072).is_err(),
                "orphan {factor}_b"
            );
        }
    }
    #[test]
    fn portable_lokr_scaling_never_defaults_bad_metadata() {
        let mut meta = BTreeMap::from([("rank".into(), "4".into()), ("alpha".into(), "6".into())]);
        assert_eq!(lokr_scaling(&meta).unwrap(), (4., 6.));
        for rank in ["0", "-1", "oops", "NaN", "inf", "4.5"] {
            meta.insert("rank".into(), rank.into());
            assert!(lokr_scaling(&meta).is_err(), "rank {rank}");
        }
        meta.insert("rank".into(), "4".into());
        for alpha in ["oops", "NaN", "inf", "1e999"] {
            meta.insert("alpha".into(), alpha.into());
            assert!(lokr_scaling(&meta).is_err(), "alpha {alpha}");
        }
        meta.remove("alpha");
        assert!(lokr_scaling(&meta).is_err());
        meta.insert("alpha".into(), "6".into());
        meta.remove("rank");
        assert!(lokr_scaling(&meta).is_err());
    }
}

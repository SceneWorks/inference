//! Portable MOSS adapter contract shared by native backends. The named sampler adapts the
//! published *pretraining* final Stage3 recipe (arXiv:2610.23374v1, Appendix A.2 Eq4), not an
//! unpublished official LoRA/SFT recipe. Posterior-mean caching is a SceneWorks choice.
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

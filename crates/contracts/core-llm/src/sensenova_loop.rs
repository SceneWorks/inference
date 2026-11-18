//! Execution metadata for the released SenseNova-Loop-9B checkpoint. Both
//! backends bind weights by physical layer and cache slots by visit order.
use serde_json::Value;

mod weights;
pub use weights::{target_tensors, TARGET_SHARDS};

pub const MODEL_REVISION: &str = "3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6";

/// Original sidecars shared byte-for-byte by BF16 and both published native tiers.
/// Derived snapshots append provenance separately from these model inputs.
pub fn validate_sidecars(root: &std::path::Path) -> Result<(), String> {
    for (name, expected) in [
        (
            "tokenizer.json",
            "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        ),
        (
            "tokenizer_config.json",
            "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
        ),
        (
            "generation_config.json",
            "f288bb4281580138395c6acc7b3adb62f45c21530bc0708cb2cf9edf093f6a4f",
        ),
        (
            "chat_template.jinja",
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        ),
        (
            "processor_config.json",
            "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        ),
        (
            "preprocessor_config.json",
            "3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2",
        ),
    ] {
        let path = root.join(name);
        verify_digest(&path, expected)?;
    }
    Ok(())
}

/// Verify the 11 source shards before assigning the frozen revision to a derived tier.
pub fn validate_source_shards(root: &std::path::Path) -> Result<(), String> {
    for &(name, expected) in TARGET_SHARDS {
        verify_digest(&root.join(name), expected)?;
    }
    Ok(())
}

fn verify_digest(path: &std::path::Path, expected: &str) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("SenseNova-Loop-9B: cannot open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut digest)
        .map_err(|e| format!("SenseNova-Loop-9B: cannot hash {}: {e}", path.display()))?;
    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        return Err(format!("SenseNova-Loop-9B: {} SHA-256 {actual} differs from release {MODEL_REVISION} ({expected})", path.display()));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SenseNovaLoop(());

impl SenseNovaLoop {
    pub const START: usize = 4;
    pub const END: usize = 28;
    pub const STREAMS: usize = 2;
    pub const ITERATIONS: usize = 20;
    pub const VISITS: usize = 56;

    /// Recognize the wrapper before descending into its ordinary Qwen text config.
    pub fn from_config(config: &Value) -> Result<Option<Self>, String> {
        let model = config["model_type"].as_str();
        if model != Some("sensenova_loop")
            && config.get("loop_settings").is_none()
            && config.get("mhc").is_none()
            && !config["architectures"].as_array().is_some_and(|names| {
                names.iter().any(|name| {
                    name.as_str()
                        .is_some_and(|s| s.starts_with("SenseNovaLoop"))
                })
            })
        {
            return Ok(None);
        }
        validate_fields(config, &release_config(), "")?;
        for name in ["loop_settings", "mhc"] {
            let expected_keys = if name == "mhc" { 6 } else { 4 };
            if config[name].as_object().map(|o| o.len()) != Some(expected_keys) {
                return Err(format!("SenseNova-Loop-9B: unsupported {name} fields"));
            }
        }
        let rope = &config["text_config"]["rope_parameters"];
        if rope.as_object().map(|o| o.len()) != Some(7) {
            return Err("SenseNova-Loop-9B: unsupported YaRN options".into());
        }
        Ok(Some(Self(())))
    }

    pub fn visits(self) -> impl Iterator<Item = usize> {
        (0..Self::END).chain(Self::START..32)
    }

    pub fn mhc(self, physical: usize) -> bool {
        (Self::START..Self::END).contains(&physical)
    }

    /// Transformers YaRN uses its default 32/1 rotation thresholds and applies
    /// its attention factor to both FP32 trigonometric tables before the cast.
    pub fn yarn_frequencies(self) -> (Vec<f32>, f32) {
        let dim = 64.;
        let theta = 10_000_000_f64;
        let correction = |rotations: f64| {
            dim * (262_144. / (rotations * std::f64::consts::TAU)).ln() / (2. * theta.ln())
        };
        let low = correction(32.).floor();
        let high = correction(1.).ceil();
        let frequencies = (0..32)
            .map(|i| {
                let extrapolated = 1. / (theta as f32).powf(i as f32 / 32.);
                let ramp = ((i as f64 - low) / (high - low)).clamp(0., 1.) as f32;
                extrapolated * (1. - ramp) + extrapolated * 0.5 * ramp
            })
            .collect();
        (frequencies, 1. + 0.1 * 2_f32.ln())
    }
}

fn validate_fields(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    if let Some(fields) = expected.as_object() {
        for (name, value) in fields {
            validate_fields(&actual[name], value, &format!("{path}/{name}"))?;
        }
    } else if actual != expected {
        return Err(format!(
            "SenseNova-Loop-9B requires {path}={expected}, found {actual}"
        ));
    }
    Ok(())
}

fn release_config() -> Value {
    serde_json::from_str(include_str!("sensenova_loop.json"))
        .expect("release geometry is valid JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn release_visits_and_yarn() {
        let plan = SenseNovaLoop::from_config(&release_config())
            .unwrap()
            .unwrap();
        let visits: Vec<_> = plan.visits().collect();
        assert_eq!(visits.len(), 56);
        assert_eq!(&visits[4..28], &visits[28..52]);
        assert_eq!(visits.iter().filter(|&&i| i % 4 == 3).count(), 14);
        assert_eq!(visits.iter().filter(|&&i| plan.mhc(i)).count(), 48);
        let (freq, scale) = plan.yarn_frequencies();
        assert_eq!(freq.len(), 32);
        assert_eq!(freq[0], 1.);
        assert!((freq[31] - 0.5 * 10_000_000_f32.powf(-31. / 32.)).abs() < 1e-12);
        assert!((scale - 1.0693147).abs() < 1e-6);
    }

    #[test]
    fn refuses_partial_or_incompatible_release() {
        for path in [
            "/model_type",
            "/loop_settings/eval_count",
            "/mhc/iterations",
            "/text_config/rope_parameters/mrope_section",
            "/vision_config/patch_size",
            "/text_config/layer_types/2",
        ] {
            let mut cfg = release_config();
            *cfg.pointer_mut(path).unwrap() = Value::Null;
            assert!(SenseNovaLoop::from_config(&cfg)
                .unwrap_err()
                .contains(path.split('/').nth(1).unwrap()));
        }
        assert_eq!(
            SenseNovaLoop::from_config(&json!({"model_type":"qwen3_5"})),
            Ok(None)
        );
    }
}

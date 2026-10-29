//! The flow-match trainers' shared request vocabulary, request floor and timestep sampler
//! (sc-27335, sc-27324).
//!
//! Every MLX image/video LoRA trainer accepts the same free-form `timestep_type` / `timestep_bias` /
//! `loss_type` spellings, rejects the same malformed runs, and draws its per-step flow-match timestep
//! with the same port of the SceneWorks `sample_training_timestep`. They used to carry one copy
//! each, and the copies had already drifted (the weighted sampler's second key was `seed ^ φ` in
//! some families and `seed.wrapping_add(φ)` in others). One definition here; a family keeps only
//! what is genuinely its own (extra timestep types, a schedule shift, a band remap).

use mlx_rs::random;

use super::{TrainOptimizer, TrainingConfig, TrainingRequest};
use crate::Result;

/// Recognized `timestep_type` values (`sigmoid` is the fall-through default).
pub const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];

/// Recognized `timestep_bias` values (`balanced`/`none`/`neutral` are neutral).
pub const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];

/// Recognized `loss_type` values (`mse`/`l2` = MSE, `mae`/`l1` = MAE).
pub const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// The weighted sampler's second-key offset (the 32-bit golden-ratio constant).
const WEIGHTED_KEY_OFFSET: u64 = 0x9E37_79B9;

/// Normalize a free-form config string the way the trainers' parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
pub fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Refuse a config `value` for `field` whose normalized spelling is not in `supported`, so a typo
/// cannot silently fall back to a default sampler or loss (sc-4041). `prefix` names the trainer
/// (`"<id> trainer"`).
pub fn check_vocab(prefix: &str, field: &str, value: &str, supported: &[&str]) -> Result<()> {
    if supported.contains(&normalize_cfg(value).as_str()) {
        return Ok(());
    }
    Err(format!(
        "{prefix}: {field} '{value}' is not recognized (supported: {})",
        supported.join(", ")
    )
    .into())
}

/// The run-shape floor every adapter trainer shares: a non-empty dataset, a non-zero rank, **non-zero
/// steps** (sc-4040 — a 0-step run would otherwise fall through to the save and write a no-op `B = 0`
/// identity adapter) and an optimizer MLX training implements.
pub fn validate_run_shape(prefix: &str, req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err(format!("{prefix}: dataset is empty").into());
    }
    if cfg.rank == 0 {
        return Err(format!("{prefix}: rank must be > 0").into());
    }
    validate_steps_and_optimizer(prefix, cfg)
}

/// The steps/optimizer half of [`validate_run_shape`], for a trainer whose rank floor is conditional.
pub fn validate_steps_and_optimizer(prefix: &str, cfg: &TrainingConfig) -> Result<()> {
    if cfg.steps == 0 {
        return Err(format!("{prefix}: steps must be > 0").into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "{prefix}: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    Ok(())
}

/// The sampler/loss vocabulary floor: `timestep_bias` and `loss_type` against the shared lists, and
/// `timestep_type` against `timestep_types` ([`TIMESTEP_TYPES`] unless the family samples more).
pub fn validate_sampling_vocab(
    prefix: &str,
    cfg: &TrainingConfig,
    timestep_types: &[&str],
) -> Result<()> {
    check_vocab(prefix, "timestep_type", &cfg.timestep_type, timestep_types)?;
    check_vocab(
        prefix,
        "timestep_bias",
        &cfg.timestep_bias,
        &TIMESTEP_BIASES,
    )?;
    check_vocab(prefix, "loss_type", &cfg.loss_type, &LOSS_TYPES)
}

/// [`validate_run_shape`] then [`validate_sampling_vocab`] over the shared [`TIMESTEP_TYPES`]: the
/// complete capability-free request floor of a trainer with no family-specific checks.
pub fn validate_common_request(prefix: &str, req: &TrainingRequest) -> Result<()> {
    validate_run_shape(prefix, req)?;
    validate_sampling_vocab(prefix, &req.config, &TIMESTEP_TYPES)
}

/// Apply a `timestep_bias` to a unit timestep: `high` → `√t`, `low` → `t²`, anything else neutral.
pub fn apply_timestep_bias(t: f32, timestep_bias: &str) -> f32 {
    match normalize_cfg(timestep_bias).as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    }
}

/// Clamp a unit timestep to the open interval the flow-match trainers train on, `[1e-3, 1−1e-3]`.
pub fn clamp_unit_timestep(t: f32) -> f32 {
    t.clamp(1e-3, 1.0 - 1e-3)
}

/// Sample a biased, **unclamped** flow-match unit timestep — a faithful port of the SceneWorks
/// `sample_training_timestep`: `sigmoid(randn)` by default, `uniform` for `linear`/`uniform`,
/// `(uniform + sigmoid(randn))/2` for `weighted`; then [`apply_timestep_bias`]. Deterministic in
/// `seed`.
///
/// The weighted branch's second key is `seed.wrapping_add(φ)` (Fibonacci-hash mixing) rather than an
/// XOR-sibling seed, whose two draws can correlate depending on `key()`'s splat (sc-5308). Families that
/// used the XOR form before sc-27324 draw a different (equally distributed) weighted stream now.
///
/// Use this directly only when a family transforms the unit timestep before clamping (a schedule
/// shift); otherwise use [`sample_timestep`].
pub fn sample_unit_timestep(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let t = match normalize_cfg(timestep_type).as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed.wrapping_add(WEIGHTED_KEY_OFFSET))?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    Ok(apply_timestep_bias(t, timestep_bias))
}

/// [`sample_unit_timestep`] clamped by [`clamp_unit_timestep`]: the per-step flow-match timestep
/// (interpolation coefficient) `t ∈ [1e-3, 1−1e-3]`. Deterministic in `seed`.
pub fn sample_timestep(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    Ok(clamp_unit_timestep(sample_unit_timestep(
        timestep_type,
        timestep_bias,
        seed,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancelFlag, TrainingItem};
    use std::path::PathBuf;

    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem {
                image_path: PathBuf::from("/tmp/x.png"),
                caption: "a swatch".into(),
                ..Default::default()
            }],
            config: TrainingConfig {
                rank: 4,
                steps: 2,
                ..Default::default()
            },
            output_dir: PathBuf::from("/tmp/vocab_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn normalize_cfg_folds_case_space_and_dash() {
        assert_eq!(normalize_cfg("  High-Noise "), "high_noise");
        assert_eq!(normalize_cfg("favor low noise"), "favor_low_noise");
    }

    #[test]
    fn common_request_floor_keeps_the_family_prefix_and_messages() {
        validate_common_request("fixture trainer", &request()).unwrap();

        let mut empty = request();
        empty.items.clear();
        let error = validate_common_request("fixture trainer", &empty).unwrap_err();
        assert_eq!(error.to_string(), "fixture trainer: dataset is empty");

        let mut typo = request();
        typo.config.timestep_type = "sigmiod".into();
        let error = validate_common_request("fixture trainer", &typo).unwrap_err();
        assert_eq!(
            error.to_string(),
            "fixture trainer: timestep_type 'sigmiod' is not recognized (supported: sigmoid, \
             linear, uniform, weighted)"
        );

        let mut loss = request();
        loss.config.loss_type = "huber".into();
        assert!(validate_common_request("fixture trainer", &loss).is_err());

        let mut steps = request();
        steps.config.steps = 0;
        let error = validate_common_request("fixture trainer", &steps).unwrap_err();
        assert_eq!(error.to_string(), "fixture trainer: steps must be > 0");
    }

    #[test]
    fn timestep_bias_is_applied_after_normalization() {
        assert_eq!(apply_timestep_bias(0.25, "High"), 0.5);
        assert_eq!(apply_timestep_bias(0.5, "low-noise"), 0.25);
        assert_eq!(apply_timestep_bias(0.5, "balanced"), 0.5);
    }

    #[test]
    fn sample_timestep_is_deterministic_and_in_range() {
        for ttype in ["sigmoid", "linear", "uniform", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                for seed in [0_u64, 7, 0x9E37_79B9, u64::MAX] {
                    let a = sample_timestep(ttype, bias, seed).unwrap();
                    let b = sample_timestep(ttype, bias, seed).unwrap();
                    assert_eq!(a.to_bits(), b.to_bits(), "{ttype}/{bias}/{seed}");
                    assert!(
                        (1e-3..=1.0 - 1e-3).contains(&a),
                        "{ttype}/{bias}/{seed}: {a}"
                    );
                }
            }
        }
    }
}

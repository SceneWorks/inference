//! The flow-match LoRA trainers' shared request vocabulary and checks (sc-27413): the
//! `timestep_type` / `timestep_bias` / `loss_type` spellings, the config-string normalizer, the
//! capability-free request validation, the shared gen-core trainer floors, and the unit-timestep
//! sampler — previously one verbatim copy per MLX family trainer.

use mlx_rs::random;

use crate::train::{TrainOptimizer, TrainerDescriptor, TrainingRequest};
use crate::Result;

/// Recognized `timestep_type` values (`sigmoid` is the default the sampler falls back to).
pub const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values — the high/low-noise tilts [`sample_unit_timestep`] branches on
/// plus the neutral default (`balanced`/`none`/`neutral`).
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
/// Recognized `loss_type` values.
pub const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// Normalize a free-form config string the way the trainers' parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings a run would.
pub fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free flow-match request validation, run before any expensive load: rejects an empty
/// dataset, zero rank, **zero steps** (a 0-step run would write a no-op `B = 0` identity adapter), an
/// optimizer MLX training does not provide, and an unrecognized `timestep_type` / `timestep_bias` /
/// `loss_type` (rather than silently falling back to a default). `label` prefixes every message (e.g.
/// `"krea trainer"`), so each family's error text is unchanged.
pub fn validate_flow_match_request(req: &TrainingRequest, label: &str) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err(format!("{label}: dataset is empty").into());
    }
    if cfg.rank == 0 {
        return Err(format!("{label}: rank must be > 0").into());
    }
    if cfg.steps == 0 {
        return Err(format!("{label}: steps must be > 0").into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "{label}: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    validate_flow_vocabulary(req, label, &TIMESTEP_TYPES)
}

/// The `timestep_type` / `timestep_bias` / `loss_type` half of [`validate_flow_match_request`], over a
/// caller-supplied `timestep_types` vocabulary.
pub fn validate_flow_vocabulary(
    req: &TrainingRequest,
    label: &str,
    timestep_types: &[&str],
) -> Result<()> {
    let cfg = &req.config;
    if !timestep_types.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(format!(
            "{label}: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            timestep_types.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "{label}: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "{label}: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

/// The gen-core trainer floors every LoRA-only flow trainer runs ahead of its own request check: a
/// control-branch request, a `full_finetune` request, an undeclared training technique, and an
/// edit-pair request the descriptor does not declare are each a typed refusal, never silently
/// trained as a plain adapter.
pub fn validate_shared_floors(
    descriptor: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    gen_core::train::validate_control_request(descriptor, req)?;
    gen_core::train::validate_full_finetune_request(descriptor, req)?;
    gen_core::train::validate_training_techniques(descriptor, req)?;
    gen_core::train::validate_edit_request(descriptor, req)
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `t ∈ [1e-3, 1−1e-3]` — a
/// port of the SceneWorks `sample_training_timestep`: `sigmoid(randn)` by default, `uniform` for
/// linear, `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√t`, `low` → `t²`.
/// Deterministic in `seed`.
pub fn sample_unit_timestep(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t = match ttype.as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed ^ 0x9E37_79B9)?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    let bias = timestep_bias
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_");
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

//! Exact-source scalar LR/clip/tail policy oracle. No weights or accelerator needed.
use gen_core_flux3::training::policy::{
    accumulation_loss_scale, optimizer_update_budget, source_clip_multiplier, SourceLrKind,
    SourceLrSchedule,
};
fn golden() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/flux3_training_policy_golden.json")).unwrap()
}

#[test]
fn all_source_learning_rate_schedules_match_actual_frozen_functions() {
    let golden = golden();
    for case in golden["schedules"].as_array().unwrap() {
        let schedule = SourceLrSchedule {
            kind: SourceLrKind::parse(case["name"].as_str().unwrap()).unwrap(),
            warmup: case["warmup"].as_u64().unwrap(),
            total_updates: case["total_updates"].as_u64().unwrap(),
            cycles: case["cycles"].as_u64().unwrap() as u32,
            power: case["power"].as_f64().unwrap(),
            base_lr: case["base_lr"].as_f64().unwrap(),
        };
        schedule.validate().unwrap();
        for (step, expected) in case["multipliers"].as_array().unwrap().iter().enumerate() {
            let got = schedule.multiplier(step as u64).unwrap();
            assert!(
                (got - expected.as_f64().unwrap()).abs() < 2e-12,
                "{} step{step}: {got} != {expected}",
                case["name"]
            );
        }
    }
    assert!(SourceLrKind::parse("typo").is_err());
}

#[test]
fn clipping_uses_source_epsilon_even_at_the_norm_boundary() {
    for case in golden()["clips"].as_array().unwrap() {
        let norm = case["norm"].as_f64().unwrap() as f32;
        let maximum = case["max_norm"].as_f64().unwrap() as f32;
        let coefficient = source_clip_multiplier(norm, maximum).unwrap();
        assert_eq!(
            coefficient.to_bits(),
            (case["coefficient"].as_f64().unwrap() as f32).to_bits()
        );
        for (input, expected) in case["gradient"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["clipped"].as_array().unwrap())
        {
            assert_eq!(
                (input.as_f64().unwrap() as f32 * coefficient).to_bits(),
                (expected.as_f64().unwrap() as f32).to_bits()
            );
        }
    }
    assert!(source_clip_multiplier(f32::NAN, 1.0).is_err());
}

#[test]
fn short_accumulation_tail_keeps_configured_divisor_and_epoch_flush_count() {
    let golden = golden();
    let updates = golden["three_micro_accumulation_two"].as_array().unwrap();
    let scale = accumulation_loss_scale(2).unwrap();
    assert_eq!(
        (1.0 + 3.0) * scale,
        updates[0]["accumulated_gradient"].as_f64().unwrap() as f32
    );
    assert_eq!(
        9.0 * scale,
        updates[1]["accumulated_gradient"].as_f64().unwrap() as f32
    );
    assert_ne!(
        9.0 * scale,
        9.0,
        "a short tail is NOT averaged by its actual pending count"
    );
    assert_eq!(
        optimizer_update_budget(6, 2, 3).unwrap(),
        4,
        "each three-batch epoch flushes its short tail"
    );
    assert_eq!(optimizer_update_budget(7, 2, 3).unwrap(), 5);
    assert!(accumulation_loss_scale(0).is_err());
    assert!(optimizer_update_budget(3, 2, 0).is_err());
}

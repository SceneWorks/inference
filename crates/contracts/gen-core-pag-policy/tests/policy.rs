use std::borrow::Cow;

use gen_core_pag_policy::{
    pag_apply_slice, resolve_pag, PagCapability, PagConfig, PagExecutionReport, PagPolicyErrorKind,
    DEFAULT_PAG_MAX_SCALE,
};
use serde_json::json;

mod support;

fn sample_capability() -> PagCapability {
    PagCapability {
        num_blocks: 42,
        block_domain: Cow::Borrowed("layers"),
        default_layers: vec![20],
        default_start: 0.0,
        default_stop: 1.0,
        recommended_scale: 2.5,
        max_scale: DEFAULT_PAG_MAX_SCALE,
    }
}

#[test]
fn portable_native_and_wasm_corpus() {
    assert_eq!(support::run(), 16);
}

#[test]
fn nonfinite_floats_are_identified_in_canonical_order_and_rejected() {
    let cap = sample_capability();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let c1 = PagConfig {
            scale: bad,
            start: Some(bad),
            stop: Some(bad),
            adaptive_scale: Some(bad),
            layers: None,
        };
        assert_eq!(
            c1.first_nonfinite_float().map(|(f, _)| f),
            Some("pag.scale")
        );
        assert!(cap.resolve(Some(&c1)).is_err());

        let c2 = PagConfig {
            scale: 1.0,
            start: Some(bad),
            stop: Some(bad),
            adaptive_scale: Some(bad),
            layers: None,
        };
        assert_eq!(
            c2.first_nonfinite_float().map(|(f, _)| f),
            Some("pag.start")
        );
        assert!(cap.resolve(Some(&c2)).is_err());

        let c3 = PagConfig {
            scale: 1.0,
            start: Some(0.0),
            stop: Some(bad),
            adaptive_scale: Some(bad),
            layers: None,
        };
        assert_eq!(c3.first_nonfinite_float().map(|(f, _)| f), Some("pag.stop"));
        assert!(cap.resolve(Some(&c3)).is_err());

        let c4 = PagConfig {
            scale: 1.0,
            start: Some(0.0),
            stop: Some(1.0),
            adaptive_scale: Some(bad),
            layers: None,
        };
        assert_eq!(
            c4.first_nonfinite_float().map(|(f, _)| f),
            Some("pag.adaptive_scale")
        );
        assert!(cap.resolve(Some(&c4)).is_err());
    }
}

#[test]
fn execution_report_records_resolved_parameters_and_validates_consistency() {
    let cap = sample_capability();
    let cfg = PagConfig {
        scale: 3.0,
        layers: Some(vec![35, 12, 35]),
        start: Some(0.25),
        stop: Some(0.75),
        adaptive_scale: Some(1.0),
    };
    let resolved = cap.resolve(Some(&cfg)).unwrap();
    assert_eq!(resolved.layers, vec![12, 35]);
    assert!(resolved.perturbs_block(12, 1, 4));
    assert!(!resolved.perturbs_block(20, 1, 4));
    assert!(!resolved.perturbs_block(12, 0, 4));
    assert_eq!(resolved.step_layers(0, 4), &[] as &[u32]);
    assert_eq!(resolved.step_layers(1, 4), &[12, 35]);

    let report = resolved.execution_report(4);
    assert_eq!(
        report,
        PagExecutionReport {
            layers: vec![12, 35],
            active_steps: 2,
            total_steps: 4,
            scale: 3.0,
            adaptive_scale: 1.0,
            start: 0.25,
            stop: 0.75,
        }
    );
    report.validate_with_capability(&cap).unwrap();
    let encoded = serde_json::to_value(&report).unwrap();
    assert_eq!(
        serde_json::from_value::<PagExecutionReport>(encoded).unwrap(),
        report
    );

    let mut unsorted = report.clone();
    unsorted.layers = vec![35, 12];
    assert!(unsorted.validate_with_capability(&cap).is_err());

    let mut wrong_steps = report.clone();
    wrong_steps.active_steps = 3;
    assert!(wrong_steps.validate_with_capability(&cap).is_err());

    let mut out_of_range = report;
    out_of_range.layers = vec![42];
    assert!(out_of_range.validate_with_capability(&cap).is_err());
}

#[test]
fn resolve_pag_helper_maps_noop_and_capability_gaps() {
    let cap = sample_capability();
    assert_eq!(
        resolve_pag(Some(&cap), "moss_image_sft", None).unwrap(),
        None
    );
    assert_eq!(
        resolve_pag(
            Some(&cap),
            "moss_image_sft",
            Some(&PagConfig {
                scale: 0.0,
                layers: Some(vec![0, 41]),
                ..Default::default()
            })
        )
        .unwrap(),
        None
    );
    let unsupported = resolve_pag(
        None,
        "flux1_schnell",
        Some(&PagConfig {
            scale: 2.0,
            ..Default::default()
        }),
    )
    .unwrap_err();
    assert_eq!(unsupported.kind(), PagPolicyErrorKind::Unsupported);

    // Unknown wire fields are rejected cleanly.
    assert!(serde_json::from_value::<PagConfig>(json!({"scale": 2.0, "extra": 1})).is_err());
}

#[test]
fn slice_combine_covers_two_pass_three_pass_and_zero_scale() {
    let cond = [2.0_f32, 4.0, -1.0];
    let uncond = [1.0_f32, 0.0, 1.0];
    let perturbed = [0.5_f32, 2.0, -2.0];

    // 2-pass: eps_base == eps_cond -> cond + s * (cond - perturbed)
    let two_pass = pag_apply_slice(&cond, &cond, &perturbed, 2.0).unwrap();
    assert_eq!(two_pass, vec![5.0, 8.0, 1.0]);

    // 3-pass: eps_base == eps_cfg -> cfg + s * (cond - perturbed)
    let cfg: Vec<f32> = cond
        .iter()
        .zip(&uncond)
        .map(|(&c, &u)| u + 3.0 * (c - u))
        .collect();
    let three_pass = pag_apply_slice(&cfg, &cond, &perturbed, 1.5).unwrap();
    let expected: Vec<f32> = (0..3)
        .map(|i| cfg[i] + 1.5 * (cond[i] - perturbed[i]))
        .collect();
    assert_eq!(three_pass, expected);

    // Zero scale identity.
    assert_eq!(pag_apply_slice(&cfg, &cond, &perturbed, 0.0).unwrap(), cfg);
    assert!(pag_apply_slice(&cfg, &cond, &perturbed, -0.5).is_err());
    assert!(pag_apply_slice(&cfg, &cond, &perturbed, f32::NAN).is_err());
    assert!(pag_apply_slice(&cfg[..2], &cond, &perturbed, 1.0).is_err());
}

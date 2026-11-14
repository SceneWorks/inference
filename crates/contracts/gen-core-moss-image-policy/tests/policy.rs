use gen_core_moss_image_policy::{
    apply_edit_defaults, MossDmdPrecision, MossImageAcceleration, MossImageOptions,
    MossImageResolved, MossImageVariant, MossPolicyRequest, MossSkipLayerGuidance,
};
use serde_json::{json, Value};

// These cases preserve gen-core's existing MOSS resolver and edit_request behavior. The same
// serde-only policy crate is compiled for native and wasm consumers.
#[test]
fn ordinary_defaults_and_explicit_controls() {
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        let default = MossImageResolved::resolve(variant, &MossPolicyRequest::default()).unwrap();
        assert_eq!(
            (default.steps, default.guidance, default.shift),
            (50, 3.0, 5.0)
        );
        assert_eq!(default.training_shift, 1.0);
        assert_eq!(default.max_sequence_length, 4096);
        assert_eq!(default.uncond_skip_layers, [22]);
        assert_eq!(default.acceleration, MossImageAcceleration::Off);

        let request = MossPolicyRequest {
            steps: Some(17),
            guidance: Some(4.5),
            scheduler_shift: Some(6.0),
            moss_image: Some(MossImageOptions {
                max_sequence_length: 1024,
                uncond_skip_layers: Some(vec![]),
                slg: Some(MossSkipLayerGuidance {
                    layers: vec![1, 41],
                    scale: 2.0,
                    start: 0.25,
                    stop: 0.75,
                }),
                training_shift: Some(1.5),
                ..Default::default()
            }),
            ..Default::default()
        };
        let resolved = MossImageResolved::resolve(variant, &request).unwrap();
        assert_eq!(
            (resolved.steps, resolved.guidance, resolved.shift),
            (17, 4.5, 6.0)
        );
        assert_eq!(resolved.training_shift, 1.5);
        assert_eq!(resolved.max_sequence_length, 1024);
        assert!(resolved.uncond_skip_layers.is_empty());
        assert!(!resolved.skipping_active(4, 17));
        assert!(resolved.skipping_active(5, 17));
        assert!(!resolved.skipping_active(13, 17));
        let wire = serde_json::to_value(&resolved).unwrap();
        assert_eq!(
            serde_json::from_value::<MossImageResolved>(wire).unwrap(),
            resolved
        );
    }
}

#[test]
fn dmd_defaults_and_refusals() {
    for precision in [MossDmdPrecision::UnmergedBf16, MossDmdPrecision::FullFp32] {
        let request = MossPolicyRequest {
            moss_image: Some(MossImageOptions {
                acceleration: MossImageAcceleration::Dmd(precision),
                ..Default::default()
            }),
            ..Default::default()
        };
        let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &request).unwrap();
        assert_eq!((resolved.steps, resolved.guidance), (8, 0.0));
        assert!(resolved.uncond_skip_layers.is_empty());
        assert!(MossImageResolved::resolve(MossImageVariant::Base, &request).is_err());
        for changed in [
            MossPolicyRequest {
                steps: Some(9),
                ..request.clone()
            },
            MossPolicyRequest {
                negative_prompt: Some("bad".into()),
                ..request.clone()
            },
            MossPolicyRequest {
                sampler: Some("euler".into()),
                ..request.clone()
            },
            MossPolicyRequest {
                guidance: Some(1.0),
                ..request.clone()
            },
            MossPolicyRequest {
                scheduler_shift: Some(5.0),
                ..request.clone()
            },
        ] {
            assert!(MossImageResolved::resolve(MossImageVariant::Sft, &changed).is_err());
        }
        let wire = serde_json::to_value(&resolved).unwrap();
        assert_eq!(
            serde_json::from_value::<MossImageResolved>(wire).unwrap(),
            resolved
        );
    }
}

#[test]
fn serde_numeric_and_nested_fields_match_native_contract() {
    let integer_guidance: MossPolicyRequest = serde_json::from_value(json!({
        "guidance": 4,
        "moss_image": {"uncond_skip_layers": [], "slg": {
            "layers": [0, 41], "scale": 2, "start": 0, "stop": 1
        }}
    }))
    .unwrap();
    assert_eq!(integer_guidance.guidance, Some(4.0));
    assert_eq!(
        integer_guidance
            .moss_image
            .as_ref()
            .unwrap()
            .uncond_skip_layers,
        Some(vec![])
    );
    let none: MossImageOptions =
        serde_json::from_value(json!({"uncond_skip_layers": null})).unwrap();
    let empty: MossImageOptions =
        serde_json::from_value(json!({"uncond_skip_layers": []})).unwrap();
    assert_eq!(none.uncond_skip_layers, None);
    assert_eq!(empty.uncond_skip_layers, Some(vec![]));
    for invalid in [
        json!({"guidance": "4"}),
        json!({"width": 1024.5}),
        json!({"moss_image": {"max_sequence_length": "4096"}}),
        json!({"moss_image": {"max_sequence_length": 0}}),
        json!({"moss_image": {"uncond_skip_layers": [42]}}),
        json!({"moss_image": {"slg": {"layers": [1], "scale": 2, "start": 0, "stop": 1, "unknown": 0}}}),
        json!({"moss_image": {"acceleration": "unmerged_bf16", "training_shift": 1}}),
    ] {
        assert!(serde_json::from_value::<MossPolicyRequest>(invalid).is_err());
    }
    let nonfinite = MossPolicyRequest {
        guidance: Some(f32::NAN),
        ..Default::default()
    };
    assert!(MossImageResolved::resolve(MossImageVariant::Sft, &nonfinite).is_err());
    let mut inconsistent =
        MossImageResolved::resolve(MossImageVariant::Sft, &MossPolicyRequest::default()).unwrap();
    inconsistent.steps = 0;
    assert!(serde_json::to_value(inconsistent).is_err());
}

#[test]
fn edit_defaults_preserve_other_fields_and_reject_incompatible_modes() {
    let request = MossPolicyRequest {
        width: 640,
        height: 384,
        count: 2,
        ..Default::default()
    };
    let edit = apply_edit_defaults(&request).unwrap();
    assert_eq!((edit.width, edit.height, edit.count), (640, 384, 2));
    assert_eq!(
        (edit.steps, edit.guidance, edit.scheduler_shift),
        (Some(30), Some(4.0), Some(5.0))
    );
    assert_eq!(edit.moss_image.unwrap().uncond_skip_layers, Some(vec![]));
    assert!(request.moss_image.is_none());
    let explicit = MossPolicyRequest {
        steps: Some(11),
        guidance: Some(2.0),
        scheduler_shift: Some(7.0),
        ..request.clone()
    };
    let preserved = apply_edit_defaults(&explicit).unwrap();
    assert_eq!(
        (
            preserved.steps,
            preserved.guidance,
            preserved.scheduler_shift
        ),
        (Some(11), Some(2.0), Some(7.0))
    );
    for invalid in [
        MossPolicyRequest {
            conditioning_present: true,
            ..request.clone()
        },
        MossPolicyRequest {
            moss_image: Some(MossImageOptions {
                acceleration: MossImageAcceleration::Dmd(Default::default()),
                ..Default::default()
            }),
            ..request.clone()
        },
        MossPolicyRequest {
            moss_image: Some(MossImageOptions {
                uncond_skip_layers: Some(vec![22]),
                ..Default::default()
            }),
            ..request.clone()
        },
        MossPolicyRequest {
            moss_image: Some(MossImageOptions {
                slg: Some(MossSkipLayerGuidance {
                    layers: vec![],
                    scale: 1.0,
                    start: 0.0,
                    stop: 1.0,
                }),
                ..Default::default()
            }),
            ..request.clone()
        },
    ] {
        assert!(apply_edit_defaults(&invalid).is_err());
    }
    let wire: Value = serde_json::to_value(preserved).unwrap();
    assert_eq!(wire["moss_image"]["uncond_skip_layers"], json!([]));
}

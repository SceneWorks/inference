//! sc-26237: original authoring data must not become the rounded conditioning table.
use gen_core_flux3::{layouts::*, pup, references};
use serde_json::json;

fn output() -> references::OutputGeometry {
    references::OutputGeometry {
        canvas: (112, 80),
        crop: Some([5, 5, 106, 74]),
    }
}

#[test]
fn request_layout_defaults_absent_and_is_scoped_to_flux3() {
    use gen_core::{Capabilities, GenerationRequest};
    let mut request = GenerationRequest::default();
    assert!(request.flux3_layout.is_none());
    let caps = Capabilities {
        min_size: 64,
        max_size: 4096,
        max_count: 8,
        ..Default::default()
    };
    assert!(caps.validate_request("other_family", &request).is_ok());
    request.flux3_layout = Some(
        Layout::from_json(json!({"version":1,"boxes":[{"id":"a","bbox":[0,0,1,1]}]})).unwrap(),
    );
    assert!(caps
        .validate_request("other_family", &request)
        .unwrap_err()
        .to_string()
        .contains("flux3_layout"));
    assert!(caps
        .validate_request_skip_size("other_family", &request)
        .unwrap_err()
        .to_string()
        .contains("flux3_layout"));
    assert!(caps
        .validate_request(gen_core::flux3::DISTILLED_MODEL_ID, &request)
        .is_ok());
    assert!(caps
        .validate_request(gen_core::flux3::BASE_MODEL_ID, &request)
        .is_ok());
}

#[test]
fn independent_frozen_bbox_source_rows_match_exact_python_json() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/flux3-layouts/source-boxes.json")).unwrap();
    assert_eq!(fixture["source_sha256"], pup::BBOX_PY_SHA256);
    let plan = references::plan(
        &[(101, 69), (80, 64)],
        references::ReferenceOutputSize {
            width: Some(112),
            height: Some(80),
        },
    )
    .unwrap();
    let refs = [
        ReferenceBinding::oriented("A", (101, 69)),
        ReferenceBinding::oriented("B", (80, 64)),
    ];
    for case in fixture["cases"].as_array().unwrap() {
        let layout = Layout::from_json(case["input"].clone()).unwrap();
        let compiled = layout.compile(&refs, Some(&plan), &output()).unwrap();
        assert_eq!(
            compiled.rows_json(),
            case["expected_python_json"].as_str().unwrap()
        );
        assert_eq!(layout.to_json(), &case["input"]);
    }
}

#[test]
fn placement_and_explicit_removal_roundtrip_without_f32_or_null_coalescing() {
    let original = json!({"version":1,"boxes":[
        {"id":"first_stable","desc":"猫 \"[red]\"","bbox":[0.0,0.123456789012345,1.0,1.25]},
        {"id":"remove_other","desc":"remove","from":"ref_image_0","reference_id":"asset-B","src_bbox":[0.1,0.2,0.3,0.4],"tgt_bbox":null},
        {"id":"move","desc":"move","from":"ref_image_1","reference_id":"asset-A","src_bbox":[0,0,1,1],"tgt_bbox":[0.2,0.3,0.8,0.9]}
    ]});
    let layout = Layout::from_json(original.clone()).unwrap();
    let plan = references::plan(
        &[(101, 69), (80, 64)],
        references::ReferenceOutputSize {
            width: Some(112),
            height: Some(80),
        },
    )
    .unwrap();
    // Reordered references deliberately disagree with the stale upstream ordinal in authoring.
    let refs = [
        ReferenceBinding::oriented("asset-A", (101, 69)),
        ReferenceBinding::oriented("asset-B", (80, 64)),
    ];
    let compiled = layout.compile(&refs, Some(&plan), &output()).unwrap();
    assert_eq!(layout.to_json(), &original);
    assert_eq!(
        Layout::parse(&layout.to_json().to_string()).unwrap(),
        layout
    );
    assert_eq!(compiled.boxes[1].from.as_deref(), Some("ref_image_1"));
    assert_eq!(compiled.boxes[2].from.as_deref(), Some("ref_image_0"));
    assert_eq!(compiled.boxes[1].tgt_bbox, None);
    assert!(compiled.rows_json().contains("\"tgt_bbox\": null"));
    assert_eq!(
        compiled.boxes[0].bbox.as_ref().unwrap()[1],
        0.123456789012345
    );
    assert_eq!(
        compiled.rows[0].get("bbox").unwrap(),
        &pup::loads("[0, 123, 1000, 1000]").unwrap()
    );
    assert_eq!(original["boxes"][0]["bbox"][3], 1.25); // source grid clamps only the derived row
}

#[test]
fn absent_and_null_are_distinct_in_storage_but_source_derived_rows_agree() {
    let absent = Layout::from_json(
        json!({"version":1,"boxes":[{"id":"gone","from":"ref_image_0","src_bbox":[0,0,1,1]}]}),
    )
    .unwrap();
    let explicit = Layout::from_json(json!({"version":1,"boxes":[{"id":"gone","from":"ref_image_0","src_bbox":[0,0,1,1],"tgt_bbox":null}]})).unwrap();
    assert_ne!(absent, explicit);
    let plan = references::plan(&[(101, 69)], Default::default()).unwrap();
    let refs = [ReferenceBinding::oriented("asset-A", (101, 69))];
    assert_eq!(
        absent.compile(&refs, Some(&plan), &output()).unwrap().rows,
        explicit
            .compile(&refs, Some(&plan), &output())
            .unwrap()
            .rows
    );
    assert!(absent.to_json()["boxes"][0].get("tgt_bbox").is_none());
    assert!(explicit.to_json()["boxes"][0]["tgt_bbox"].is_null());
}

#[test]
fn explicit_output_pixels_include_crop_offset_without_changing_original() {
    let original = json!({"version":1,"boxes":[{"id":"edge","desc":"full visible image","bbox":[0,0,69,101],"bbox_space":"output"}]});
    let layout = Layout::from_json(original.clone()).unwrap();
    let compiled = layout.compile(&[], None, &output()).unwrap();
    assert_eq!(
        compiled.boxes[0].bbox.as_deref(),
        Some([5.0 / 80.0, 5.0 / 112.0, 74.0 / 80.0, 106.0 / 112.0].as_slice())
    );
    assert_eq!(layout.to_json(), &original);
    let normalized =
        Layout::from_json(json!({"version":1,"boxes":[{"id":"edge","bbox":[0,0,1,1]}]})).unwrap();
    assert_eq!(
        normalized.compile(&[], None, &output()).unwrap().rows[0]
            .get("bbox")
            .unwrap(),
        &pup::loads("[0, 0, 1000, 1000]").unwrap()
    );
}

#[test]
fn original_exif_and_fitted_canvas_coordinates_map_to_same_oriented_source() {
    let plan = references::plan(&[(69, 101)], Default::default()).unwrap();
    let refs = [ReferenceBinding {
        id: "photo".into(),
        original_size: (101, 69),
        orientation: 6,
    }];
    let g = &plan.geometry[0];
    let canvas = g.canvas_region;
    let cases = [
        ("original", json!([0, 0, 69, 101])),
        ("oriented", json!([0, 0, 101, 69])),
        ("fitted", json!([0, 0, g.fitted_size.1, g.fitted_size.0])),
        (
            "reference_canvas",
            json!([canvas[1], canvas[0], canvas[3], canvas[2]]),
        ),
    ];
    for (space, coords) in cases {
        let doc = json!({"version":1,"boxes":[{"id":"a","from":"ref_image_0","reference_id":"photo","src_bbox":coords,"src_space":space,"tgt_bbox":null}]});
        let layout = Layout::from_json(doc).unwrap();
        let compiled = layout.compile(&refs, Some(&plan), &plan.output).unwrap();
        assert_eq!(
            compiled.boxes[0].src_bbox.as_deref(),
            Some([0.0, 0.0, 1.0, 1.0].as_slice()),
            "{space}"
        );
    }
}

#[test]
fn all_exif_orientations_transform_as_rectangles_not_just_two_corners() {
    let original = [10.0, 20.0, 30.0, 50.0];
    let expected = [
        [10., 20., 30., 50.],
        [10., 50., 30., 80.],
        [30., 50., 50., 80.],
        [30., 20., 50., 50.],
        [20., 10., 50., 30.],
        [20., 30., 50., 50.],
        [50., 30., 80., 50.],
        [50., 10., 80., 30.],
    ];
    for orientation in 1..=8 {
        assert_eq!(
            orient_box(original, (100, 60), orientation).unwrap(),
            expected[orientation as usize - 1]
        );
    }
}

#[test]
fn reject_invalid_before_any_conditioning_and_preserve_finite_outside_coordinates() {
    for row in [
        json!({"id":"a","bbox":[0,0,0,1]}),
        json!({"id":"a","bbox":[0,0,1]}),
        json!({"id":"a","bbox":[0,false,1,1]}),
        json!({"id":"a","bbox":[0,0,1,1],"src_bbox":[0,0,1,1]}),
        json!({"id":"a","from":"ref_image_5","src_bbox":[0,0,1,1]}),
        json!({"id":"a","from":"target","src_bbox":[0,0,1,1],"tgt_bbox":null}),
        json!({"id":"a","bbox":[0,0,1,1],"bbox_space":"guess"}),
        json!({"id":"a","bbox":[0,0,1,1],"units":"pixels"}),
        json!({"id":"a","bbox":[0,0,1,1],"src_space":"original"}),
        json!({"id":"a","from":"target","src_bbox":null,"src_space":"original","tgt_bbox":[0,0,1,1]}),
    ] {
        let result = Layout::from_json(json!({"version":1,"boxes":[row]}))
            .and_then(|l| l.compile(&[], None, &output()));
        assert!(result.is_err());
    }
    assert!(Layout::parse(r#"{"version":1,"boxes":[{"id":"a","bbox":[NaN,0,1,1]}]}"#).is_err());
    assert!(Layout::from_json(
        json!({"version":1,"boxes":[{"id":"same","bbox":[0,0,1,1]},{"id":"same","bbox":[0,0,1,1]}]})
    )
    .is_err());
    let outside =
        Layout::from_json(json!({"version":1,"boxes":[{"id":"a","bbox":[-0.1,-2,1.2,2]}]}))
            .unwrap();
    assert_eq!(
        outside.compile(&[], None, &output()).unwrap().rows[0]
            .get("bbox")
            .unwrap(),
        &pup::loads("[0, 0, 1000, 1000]").unwrap()
    );
}

#[test]
fn source_half_even_grid_and_complete_caption_bound() {
    let layout = Layout::from_json(json!({"version":1,"boxes":[{"id":"ties","bbox":[0.0005,0.0015,0.0025,0.0035],"desc":"猫"}]})).unwrap();
    let compiled = layout.compile(&[], None, &output()).unwrap();
    assert_eq!(
        compiled.rows[0].get("bbox").unwrap(),
        &pup::loads("[0, 2, 2, 4]").unwrap()
    );
    assert!(compiled.caption(&"猫".repeat(16000)).is_err());
    assert!(compiled.caption("scene").unwrap().starts_with("scene ["));
}

#[test]
fn live_request_locks_layout_before_prompt_seed_for_both_reference_modes() {
    use gen_core::{Conditioning, GenerationRequest, Image};
    for reference in [false, true] {
        let mut req = GenerationRequest {
            prompt: "  move cat\n".into(),
            width: 112,
            height: 80,
            flux3_layout: Some(Layout::from_json(if reference {
                json!({"version":1,"boxes":[{"id":"cat","desc":"cat","from":"ref_image_0","src_bbox":[0.1,0.2,0.6,0.7],"tgt_bbox":null}]})
            } else {
                json!({"version":1,"boxes":[{"id":"cat","desc":"cat","bbox":[0.1,0.2,0.6,0.7]}]})
            }).unwrap()),
            ..Default::default()
        };
        if reference {
            req.conditioning.push(Conditioning::Reference {
                image: Image {
                    width: 80,
                    height: 64,
                    pixels: vec![17; 80 * 64 * 3],
                },
                strength: None,
            });
        }
        let original = req.flux3_layout.clone().unwrap();
        let input = references::upsampler_input(&req).unwrap();
        assert_eq!(
            input.mode,
            if reference {
                pup::UpsampleMode::EditingWithBoxes
            } else {
                pup::UpsampleMode::TextWithBoxes
            }
        );
        let compiled = compile_request(&req).unwrap().unwrap();
        let outcome = pup::upsample(None, &input, &Default::default(), &req.cancel, &mut |_| {
            panic!("bypass must not infer")
        })
        .unwrap();
        assert_eq!(outcome.caption, compiled.caption(&req.prompt).unwrap());
        assert_ne!(
            gen_core_flux3::prompt_seed(&outcome.caption),
            gen_core_flux3::prompt_seed(&req.prompt)
        );
        assert_eq!(outcome.report.original_prompt, req.prompt);
        assert_eq!(
            outcome.report.locked_annotations.as_deref(),
            Some(compiled.rows_json().as_str())
        );
        assert_eq!(req.flux3_layout.as_ref(), Some(&original));
    }
}

#[test]
fn malformed_layout_reference_binding_fails_before_admission_and_pup() {
    use gen_core::{Conditioning, GenerationRequest, Image};
    let mut req = GenerationRequest {
        flux3_layout: Some(Layout::from_json(json!({"version":1,"boxes":[{"id":"cat","from":"ref_image_0","reference_id":"selected","src_bbox":[0,0,1,1],"tgt_bbox":null}]})).unwrap()),
        conditioning: vec![Conditioning::Reference {image:Image {width:80,height:64,pixels:vec![19;80*64*3]},strength:None}],
        ..Default::default()
    };
    assert!(matches!(
        references::request_plan(&req),
        Err(gen_core::Error::Unsupported(_))
    ));
    req.flux3_layout_bindings = vec![ReferenceBinding::oriented("selected", (79, 64))];
    assert!(references::request_plan(&req).is_err());
    assert!(references::upsampler_input(&req).is_err());
    req.flux3_layout_bindings[0].original_size = (80, 64);
    assert!(references::request_plan(&req).unwrap().is_some());
    req.flux3_layout = None;
    assert!(references::request_plan(&req).is_err());
}

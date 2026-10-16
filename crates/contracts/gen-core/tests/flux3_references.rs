//! Frozen pipeline.py / generate.py reference geometry, not an img2img noise-init policy.
use gen_core::flux3::references::{prepare, reference_canvas, ReferenceOutputSize};
use gen_core::Image;

fn image(w: u32, h: u32, tag: u8) -> Image {
    Image {
        width: w,
        height: h,
        pixels: (0..w * h)
            .flat_map(|i| [tag, (i % 251) as u8, (i / w % 251) as u8])
            .collect(),
    }
}

#[test]
fn admission_checks_resolved_auto_canvas_then_normalizes_for_generic_scope() {
    use gen_core::flux3::references::guard_memory_scope;
    use gen_core::{
        Conditioning, GenerationRequest, MemoryGeometry, MemoryPhase, MemoryRequestScope,
        MemoryRunOutcome,
    };
    use std::{cell::Cell, rc::Rc};
    struct Scope {
        admitted: MemoryGeometry,
        seen: Rc<Cell<usize>>,
    }
    impl MemoryRequestScope for Scope {
        fn configure_request(&mut self, req: &mut GenerationRequest) -> gen_core::Result<()> {
            self.seen.set(self.seen.get() + 1);
            assert_eq!(
                (req.width, req.height),
                (self.admitted.width, self.admitted.height)
            );
            Ok(())
        }
        fn enter_phase(&mut self, _: MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn leave_phase(&mut self, _: MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _: u32) -> gen_core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(&mut self, _: u32, _: u32) -> gen_core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, _: MemoryRunOutcome) -> gen_core::Result<()> {
            Ok(())
        }
    }
    let request = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: image(2032, 64, 17),
            strength: None,
        }],
        flux3_reference_output: Some(ReferenceOutputSize::default()),
        ..Default::default()
    };
    assert_eq!((request.width, request.height), (1024, 1024));
    let mut wrong = request.clone();
    let admitted = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: 1,
    };
    let seen = Rc::new(Cell::new(0));
    let mut scope = guard_memory_scope(
        Box::new(Scope {
            admitted,
            seen: seen.clone(),
        }),
        admitted,
    );
    let err = scope.configure_request(&mut wrong).unwrap_err();
    assert!(err
        .to_string()
        .contains("resolve owned references before memory admission"));
    assert_eq!(seen.get(), 0);
    assert_eq!((wrong.width, wrong.height), (1024, 1024));
    let admitted = MemoryGeometry {
        width: 2032,
        height: 64,
        ..admitted
    };
    let mut correct = request;
    let mut scope = guard_memory_scope(
        Box::new(Scope {
            admitted,
            seen: seen.clone(),
        }),
        admitted,
    );
    scope.configure_request(&mut correct).unwrap();
    assert_eq!(seen.get(), 1);
    assert_eq!((correct.width, correct.height), (2032, 64));
    assert_eq!(
        correct.flux3_reference_output,
        Some(ReferenceOutputSize::default())
    );
    let admitted = MemoryGeometry {
        width: 1024,
        height: 1024,
        reference_count: 0,
        ..admitted
    };
    let mut legacy = GenerationRequest::default();
    let mut scope = guard_memory_scope(
        Box::new(Scope {
            admitted,
            seen: seen.clone(),
        }),
        admitted,
    );
    scope.configure_request(&mut legacy).unwrap();
    assert_eq!((legacy.width, legacy.height), (1024, 1024));
    assert_eq!(legacy.flux3_reference_output, None);
}

#[test]
fn owned_file_exif_precedes_ae_rgb_and_pup_white_forks_for_one_and_ten() {
    use gen_core::flux3::references::{
        decode_reference_file, prepare_borrowed, request_images, upsampler_input,
    };
    use gen_core::{Conditioning, GenerationRequest};
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flux3-references");
    let decoded = decode_reference_file(&root.join("exif6.png")).unwrap();
    assert!(
        matches!(&decoded,Conditioning::ReferenceRgba {image,..} if (image.width,image.height)==(67,83))
    );
    for count in [1, 10] {
        let req = GenerationRequest {
            conditioning: vec![decoded.clone(); count],
            flux3_reference_output: Some(ReferenceOutputSize::default()),
            ..Default::default()
        };
        let ae = request_images(&req).unwrap();
        assert_eq!(ae[0].pixels, std::fs::read(root.join("exif6.rgb")).unwrap());
        let prepared = prepare_borrowed(
            &ae.iter().map(|i| i.as_ref()).collect::<Vec<_>>(),
            ReferenceOutputSize::default(),
        )
        .unwrap();
        assert_eq!(
            prepared.images.len(),
            count,
            "repeated assets remain separate references"
        );
        let fixture = if count == 1 {
            "exif6.prepared.rgb"
        } else {
            "exif6.multi.rgb"
        };
        for image in &prepared.images {
            assert_eq!(image.pixels, std::fs::read(root.join(fixture)).unwrap());
        }
        if count == 1 {
            assert_eq!(prepared.output.crop, Some([6, 6, 73, 89]));
        } else {
            assert_eq!(prepared.geometry[0].source_region, [1, 1, 65, 81]);
        }
        let pup = upsampler_input(&req).unwrap();
        assert_eq!(pup.references.len(), count);
        for image in pup.references {
            assert_eq!(
                image.to_rgb8().unwrap().2,
                std::fs::read(root.join("exif6.pup.rgb")).unwrap()
            );
        }
    }
    assert!(decode_reference_file(&root.join("missing-reference.png")).is_err());
    let empty = tempfile::NamedTempFile::new().unwrap();
    assert!(decode_reference_file(empty.path()).is_err());
}

#[test]
fn rgba_forks_before_ae_dropalpha_and_pup_white_composite_without_reordering() {
    use gen_core::flux3::references::{request_images, request_plan, upsampler_input};
    use gen_core::{Conditioning, GenerationRequest, RgbaImage};
    let mut req = GenerationRequest {
        flux3_reference_output: Some(ReferenceOutputSize::default()),
        ..Default::default()
    };
    for n in 0..10 {
        if n % 2 == 0 {
            req.conditioning.push(Conditioning::Reference {
                image: image(64, 80, n),
                strength: None,
            });
        } else {
            let pixels = (0..64 * 80)
                .flat_map(|i| [n, 31, 223, [0, 128, 255][i % 3]])
                .collect();
            req.conditioning.push(Conditioning::ReferenceRgba {
                image: RgbaImage {
                    width: 64,
                    height: 80,
                    pixels,
                },
                strength: None,
            });
        }
    }
    let ae = request_images(&req).unwrap();
    assert_eq!(
        ae.iter().map(|i| i.pixels[0]).collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
    assert_eq!(&ae[1].pixels[..9], &[1, 31, 223, 1, 31, 223, 1, 31, 223]);
    assert_eq!(
        request_plan(&req).unwrap().unwrap().conditioning_tokens,
        200
    );
    let pup = upsampler_input(&req).unwrap();
    assert_eq!(pup.references.len(), 10);
    let (_, _, white) = pup.references[1].to_rgb8().unwrap();
    assert_eq!(&white[..3], &[255, 255, 255]);
    assert_eq!(&white[3..6], &[128, 143, 239]);
    assert_eq!(&white[6..9], &[1, 31, 223]);
    req.conditioning = vec![req.conditioning[1].clone()];
    assert_eq!(request_images(&req).unwrap()[0].pixels[0], 1);
    assert_eq!(upsampler_input(&req).unwrap().references.len(), 1);
    req.conditioning = vec![Conditioning::ReferenceRgba {
        image: RgbaImage {
            width: 64,
            height: 64,
            pixels: vec![0; 3],
        },
        strength: None,
    }];
    assert!(request_plan(&req).is_err());
    req.conditioning = vec![Conditioning::ReferenceRgba {
        image: RgbaImage {
            width: u32::MAX,
            height: u32::MAX,
            pixels: vec![],
        },
        strength: None,
    }];
    assert!(
        request_plan(&req).is_err(),
        "checked RGBA buffer extent must reject, never overflow/panic"
    );
}

#[test]
fn single_centred_padding_and_default_output_crop() {
    let source = image(83, 67, 11);
    let p = prepare(
        std::slice::from_ref(&source),
        ReferenceOutputSize::default(),
    )
    .unwrap();
    assert_eq!((p.images[0].width, p.images[0].height), (96, 80));
    assert_eq!(p.output.canvas, (96, 80));
    assert_eq!(p.output.crop, Some([6, 6, 89, 73]));
    assert_eq!(p.geometry[0].source_size, (83, 67));
    assert_eq!(p.geometry[0].source_region, [0, 0, 83, 67]);
    assert_eq!(p.geometry[0].canvas_region, [6, 6, 89, 73]);
    assert_eq!(
        &p.images[0].pixels[(6 * 96 + 6) * 3..(6 * 96 + 7) * 3],
        &source.pixels[..3]
    );
    // reflect (not edge/symmetric) at the left/top corner.
    assert_eq!(
        &p.images[0].pixels[..3],
        &source.pixels[(6 * 83 + 6) * 3..(6 * 83 + 7) * 3]
    );
    let explicit = prepare(
        &[source],
        ReferenceOutputSize {
            width: Some(128),
            height: None,
        },
    )
    .unwrap();
    assert_eq!(explicit.output.canvas, (128, 80));
    assert_eq!(explicit.output.crop, None);
}

#[test]
fn tiny_single_uses_symmetric_padding_and_never_zero_dimensions() {
    let p = prepare(&[image(3, 1, 17)], ReferenceOutputSize::default()).unwrap();
    assert_eq!(p.output.canvas, (64, 64));
    assert_eq!(p.output.crop, Some([30, 31, 33, 32]));
    assert_eq!(
        &p.images[0].pixels[(31 * 64 + 29) * 3..(31 * 64 + 30) * 3],
        &[17, 0, 0]
    );
    assert_eq!(
        &p.images[0].pixels[(31 * 64 + 33) * 3..(31 * 64 + 34) * 3],
        &[17, 2, 0]
    );
}

#[test]
fn ordered_ten_references_are_not_sorted_or_deduplicated() {
    let refs: Vec<_> = (0..10).map(|n| image(83 + n, 67 + n, n as u8)).collect();
    let p = prepare(&refs, ReferenceOutputSize::default()).unwrap();
    assert_eq!(p.output.canvas, (80, 64));
    assert_eq!(p.output.crop, None);
    assert_eq!(
        p.images.iter().map(|i| i.pixels[0]).collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
    assert_eq!(p.geometry[0].source_region, [1, 1, 81, 65]);
    let mut reversed = refs.clone();
    reversed.reverse();
    let r = prepare(&reversed, ReferenceOutputSize::default()).unwrap();
    assert_eq!(
        r.images.iter().map(|i| i.pixels[0]).collect::<Vec<_>>(),
        (0..10).rev().collect::<Vec<_>>()
    );
    assert_eq!(r.output.canvas, (80, 64));
    let repeat = prepare(
        &[refs[0].clone(), refs[0].clone()],
        ReferenceOutputSize::default(),
    )
    .unwrap();
    assert_eq!(repeat.images.len(), 2);
    assert_eq!(repeat.images[0], repeat.images[1]);
}

#[test]
fn typed_output_intent_and_unsupported_strength_are_not_silently_ignored() {
    use gen_core::flux3::references::{crop_output, request_images, request_plan, upsampler_input};
    use gen_core::{Conditioning, GenerationRequest};
    let source = image(83, 67, 9);
    let mut req = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: source.clone(),
            strength: None,
        }],
        ..Default::default()
    };
    // Legacy dimensions remain explicit even when they equal a common UI default.
    assert_eq!(
        request_plan(&req).unwrap().unwrap().output.canvas,
        (1024, 1024)
    );
    assert_eq!(request_plan(&req).unwrap().unwrap().output.crop, None);
    req.flux3_reference_output = Some(ReferenceOutputSize::default());
    let auto = request_plan(&req).unwrap().unwrap();
    assert_eq!(auto.output.canvas, (96, 80));
    assert_eq!(auto.output.crop, Some([6, 6, 89, 73]));
    let prepared = prepare(
        std::slice::from_ref(&source),
        ReferenceOutputSize::default(),
    )
    .unwrap();
    let actual = crop_output(prepared.images[0].clone(), &auto.output).unwrap();
    assert_eq!(actual, source);
    // PUP sees originals and editing mode, never the padded conditioning canvas.
    let pup = upsampler_input(&req).unwrap();
    assert_eq!(
        pup.mode,
        gen_core::flux3::pup::UpsampleMode::infer(1, false)
    );
    assert!(matches!(
        &pup.references[0],
        gen_core::flux3::pup::ReferenceImage::Rgb8 {
            width: 83,
            height: 67,
            ..
        }
    ));
    for explicit in [
        ReferenceOutputSize {
            width: Some(128),
            height: None,
        },
        ReferenceOutputSize {
            width: None,
            height: Some(128),
        },
        ReferenceOutputSize {
            width: Some(128),
            height: Some(128),
        },
    ] {
        req.flux3_reference_output = Some(explicit);
        let p = request_plan(&req).unwrap().unwrap();
        assert_eq!(
            p.output.canvas,
            (explicit.width.unwrap_or(96), explicit.height.unwrap_or(80))
        );
        assert_eq!(p.output.crop, None);
    }
    req.strength = Some(1.0);
    assert!(request_plan(&req)
        .unwrap_err()
        .to_string()
        .contains("strength"));
    req.strength = None;
    req.conditioning = vec![Conditioning::Reference {
        image: source,
        strength: Some(0.5),
    }];
    assert!(request_images(&req).is_err());
    req.conditioning.clear();
    assert!(request_plan(&req).is_err());
    req.flux3_reference_output = None;
    assert!(request_plan(&req).unwrap().is_none());
}

#[test]
fn memory_route_requires_matching_reference_count_and_rejects_other_family_intent() {
    use gen_core::flux3::references::validate_memory_route;
    use gen_core::{Capabilities, GenerationRequest, MemoryMode};
    for n in [1, 10] {
        assert!(validate_memory_route(&MemoryMode::Edit, n).is_ok());
        assert!(validate_memory_route(&MemoryMode::ImageToImage, n).is_ok());
        assert!(validate_memory_route(&MemoryMode::TextToImage, n).is_err());
    }
    for n in [0, 11] {
        assert!(validate_memory_route(&MemoryMode::Edit, n).is_err());
    }
    assert!(validate_memory_route(&MemoryMode::TextToImage, 0).is_ok());
    let req = GenerationRequest {
        flux3_reference_output: Some(ReferenceOutputSize::default()),
        ..Default::default()
    };
    let caps = Capabilities {
        min_size: 64,
        max_size: 4096,
        max_count: 8,
        ..Default::default()
    };
    let err = caps.validate_request("other_family", &req).unwrap_err();
    assert!(err.to_string().contains("flux3_reference_output"));
    let err = caps
        .validate_request_skip_size("other_family", &req)
        .unwrap_err();
    assert!(err.to_string().contains("flux3_reference_output"));
    let err = caps
        .validate_request_audio("other_family", &req)
        .unwrap_err();
    assert!(err.to_string().contains("flux3_reference_output"));
}

#[test]
fn canvas_budget_includes_padding_and_rounds_ties_even() {
    assert_eq!(reference_canvas(83, 67).unwrap(), ((83, 67), (96, 80)));
    assert_eq!(
        reference_canvas(8192, 4096).unwrap(),
        ((2867, 1434), (2880, 1440))
    );
    assert_eq!(reference_canvas(1, 9000).unwrap(), ((1, 4096), (64, 4096)));
    for (w, h) in [(4096, 4096), (3000, 1400), (100000, 1)] {
        let (_, (cw, ch)) = reference_canvas(w, h).unwrap();
        assert!(cw * ch <= 4 * 1024 * 1024);
        assert!((64..=4096).contains(&cw) && (64..=4096).contains(&ch));
    }
}

#[test]
fn frozen_pillow_oracles_match_every_rgb_byte_including_lanczos_and_padding() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flux3-references");
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("reference-preprocess.json")).unwrap())
            .unwrap();
    assert_eq!(data["upstream"], gen_core::flux3::UPSTREAM_CODE_REVISION);
    for row in data["cases"].as_array().unwrap() {
        let w = row["input"][0].as_u64().unwrap() as u32;
        let h = row["input"][1].as_u64().unwrap() as u32;
        let pixels = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    [
                        ((x * 17 + y * 7) % 256) as u8,
                        ((x * 3 + y * 19) % 256) as u8,
                        ((x * 29 + y * 11) % 256) as u8,
                    ]
                })
            })
            .collect();
        let source = Image {
            width: w,
            height: h,
            pixels,
        };
        let images = if row["multi"].as_bool().unwrap() {
            vec![source.clone(), source]
        } else {
            vec![source]
        };
        let prepared = prepare(&images, ReferenceOutputSize::default()).unwrap();
        let output = &prepared.images[0];
        assert_eq!(output.width as u64, row["output"][0].as_u64().unwrap());
        assert_eq!(output.height as u64, row["output"][1].as_u64().unwrap());
        let expected = std::fs::read(root.join(row["file"].as_str().unwrap())).unwrap();
        assert_eq!(output.pixels.len(), expected.len());
        let mismatch = output
            .pixels
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{} first differing RGB byte", row["name"]);
    }
}

#[test]
fn ordered_negative_l_axis_and_source_derived_context_cost() {
    use gen_core::flux3::references::{conditioning_ids, plan};
    let ids = conditioning_ids(2, 3, 9).unwrap();
    assert_eq!(
        ids,
        vec![
            [0., 0., 0., -10.],
            [0., 0., 0.5, -10.],
            [0., 0., 1., -10.],
            [0., 0.5, 0., -10.],
            [0., 0.5, 0.5, -10.],
            [0., 0.5, 1., -10.]
        ]
    );
    assert!(conditioning_ids(2, 3, 10).is_err());
    let p = plan(&[(4096, 4096)], ReferenceOutputSize::default()).unwrap();
    assert_eq!(p.output.canvas, (2048, 2048));
    assert_eq!(p.conditioning_tokens, 16384);
    let p = plan(&[(1024, 1024); 10], ReferenceOutputSize::default()).unwrap();
    assert_eq!(p.conditioning_tokens, 40960);
}

#[test]
fn invalid_set_or_member_fails_before_preparing_any_images() {
    assert!(prepare(&[], ReferenceOutputSize::default()).is_err());
    assert!(prepare(&vec![image(64, 64, 1); 11], ReferenceOutputSize::default()).is_err());
    for invalid in [
        Image::default(),
        Image {
            width: 16,
            height: 16,
            pixels: vec![0; 767],
        },
        Image {
            width: 16,
            height: 16,
            pixels: vec![0; 769],
        },
    ] {
        assert!(prepare(&[image(83, 67, 1), invalid], ReferenceOutputSize::default()).is_err());
    }
    assert!(prepare(
        &[image(15, 64, 1), image(64, 64, 2)],
        ReferenceOutputSize::default()
    )
    .is_err());
    assert!(prepare(
        &[image(83, 67, 1)],
        ReferenceOutputSize {
            width: Some(65),
            height: None
        }
    )
    .is_err());
}

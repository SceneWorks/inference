//! The gen-core generator contract on the committed miniature package, through the explicit
//! provider catalog's production load path (`provider_registry().load(id, spec)` — the
//! composition the SceneWorks worker reaches through `candle-gen-catalog` / `runtime-cuda`):
//! validate honesty, progress (`Step 1..=N` then exactly one `Decoding`), typed cancellation,
//! seeded determinism, both residency policies, the component overrides, typed errors for a
//! missing resource and for the controls the flow map cannot honour. Mirrors the MLX twin's
//! `generator_contract` case for case — the same request contract on both backends.

use candle_gen::gen_core::flux3::{AUTOENCODER_COMPONENT, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    CancelFlag, Error as CoreError, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    OffloadPolicy, Progress, WeightsSource,
};

use crate::common::{tiny_autoencoder, tiny_package, tiny_text_encoder};

const ID: &str = "flux3_klein";

fn spec(policy: OffloadPolicy) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(tiny_package())).with_offload_policy(policy)
}

fn load(policy: OffloadPolicy) -> Box<dyn Generator> {
    candle_gen_flux3::provider_registry()
        .unwrap()
        .load(ID, &spec(policy))
        .expect("the tiny package loads through the catalog path")
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: "a red fox in the snow".to_owned(),
        width: 80,
        height: 64,
        steps: 3,
        seed: 42,
        cancel_steps: 6,
    }
}

fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the snow".to_owned(),
        width: 80,
        height: 64,
        steps: Some(3),
        seed: Some(42),
        ..Default::default()
    }
}

#[test]
fn public_reference_edits_honor_auto_crop_explicit_sizes_and_fail_before_progress() {
    use candle_gen::gen_core::flux3::references::ReferenceOutputSize;
    use candle_gen::gen_core::{Conditioning, Image, Precision, RgbaImage};
    let refs: Vec<Conditioning> = (0..10)
        .map(|n| {
            let (w, h) = if n % 2 == 0 { (83, 67) } else { (67, 99) };
            let rgb = (0..w * h)
                .flat_map(|p| [n * 19, (p % 251) as u8, (p / w % 251) as u8])
                .collect::<Vec<_>>();
            if n % 2 == 0 {
                Conditioning::ReferenceRgba {
                    image: RgbaImage {
                        width: w,
                        height: h,
                        pixels: rgb
                            .chunks_exact(3)
                            .enumerate()
                            .flat_map(|(i, p)| [p[0], p[1], p[2], [0, 128, 255][i % 3]])
                            .collect(),
                    },
                    strength: None,
                }
            } else {
                Conditioning::Reference {
                    image: Image {
                        width: w,
                        height: h,
                        pixels: rgb,
                    },
                    strength: None,
                }
            }
        })
        .collect();
    for id in [ID, "flux3_klein_base"] {
        let mut loaded = if id == ID {
            spec(OffloadPolicy::Resident)
        } else {
            let mut spec = LoadSpec::new(WeightsSource::Dir(
                crate::common::fixtures().join("tiny-base-snapshot"),
            ));
            spec.components.insert(
                TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(tiny_text_encoder()),
            );
            spec.components.insert(
                AUTOENCODER_COMPONENT.into(),
                WeightsSource::File(tiny_autoencoder()),
            );
            spec
        };
        loaded.precision = Precision::Fp32;
        let g = candle_gen_flux3::provider_registry()
            .unwrap()
            .load(id, &loaded)
            .unwrap();
        let render = |req: &GenerationRequest| {
            let GenerationOutput::Images(mut images) = g.generate(req, &mut |_| {}).unwrap() else {
                panic!("image output required")
            };
            assert_eq!(images.len(), 1);
            images.remove(0)
        };
        let mut edit = GenerationRequest {
            width: 1024,
            height: 1024,
            steps: Some(1),
            conditioning: vec![refs[0].clone()],
            flux3_reference_output: Some(ReferenceOutputSize::default()),
            ..request()
        };
        if id != "flux3_klein" {
            edit.guidance = Some(1.0);
        }
        g.validate(&edit).unwrap();
        let image = render(&edit);
        assert_eq!(
            (image.width, image.height),
            (83, 67),
            "single-auto result removes only the source padding"
        );
        let exif = candle_gen::gen_core::flux3::references::decode_reference_file(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../contracts/gen-core/tests/fixtures/flux3-references/exif6.png"),
        )
        .unwrap();
        edit.conditioning = vec![exif.clone()];
        let oriented = render(&edit);
        assert_eq!(
            (oriented.width, oriented.height),
            (67, 83),
            "EXIF orientation is applied before auto canvas/crop"
        );
        edit.conditioning = vec![exif; 10];
        let oriented_ten = render(&edit);
        assert_eq!(
            (oriented_ten.width, oriented_ten.height),
            (64, 80),
            "ten EXIF references use upstream multi-reference crop"
        );
        let mut narrow = edit.clone();
        narrow.conditioning = vec![
            Conditioning::Reference {
                image: Image {
                    width: 48,
                    height: 64,
                    pixels: vec![19; 48 * 64 * 3],
                },
                strength: None,
            },
            refs[0].clone(),
        ];
        narrow.flux3_reference_output = Some(ReferenceOutputSize::default());
        // The ordinary memory scope normalizes numeric geometry BEFORE model validation.
        // Its 48px canvas must not be re-rejected by the descriptor T2I minimum.
        let plan = candle_gen::gen_core::flux3::references::normalize_request(&mut narrow)
            .unwrap()
            .unwrap();
        assert_eq!(plan.output.canvas, (48, 64));
        g.validate(&narrow).unwrap();
        let narrow_image = render(&narrow);
        assert_eq!((narrow_image.width, narrow_image.height), (48, 64));
        let mut narrow_t2i = narrow;
        narrow_t2i.conditioning.clear();
        narrow_t2i.flux3_reference_output = None;
        assert!(
            g.validate(&narrow_t2i).is_err(),
            "legacy no-reference minimum is unchanged"
        );
        let formats = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../contracts/gen-core/tests/fixtures/flux3-image-input");
        for file in [
            "mode-luma16.png",
            "mode-rgb16.png",
            "mode-palette-alpha.png",
            "mode-rgb-trns.png",
        ] {
            edit.conditioning = vec![
                candle_gen::gen_core::flux3::references::decode_reference_file(&formats.join(file))
                    .unwrap(),
            ];
            let normalized = candle_gen::gen_core::flux3::references::request_images(&edit)
                .unwrap()[0]
                .as_ref()
                .clone();
            let encoded = render(&edit);
            assert_eq!(
                (encoded.width, encoded.height),
                (4, 3),
                "{file}: source auto output crop"
            );
            let mut canonical = edit.clone();
            canonical.conditioning = vec![Conditioning::Reference {
                image: normalized,
                strength: None,
            }];
            assert_eq!(
                encoded.pixels,
                render(&canonical).pixels,
                "{file}: encoded input equals source-normalized RGB through the public provider"
            );
        }
        edit.conditioning = vec![refs[0].clone()];
        for (intent, size) in [
            (
                ReferenceOutputSize {
                    width: Some(128),
                    height: None,
                },
                (128, 80),
            ),
            (
                ReferenceOutputSize {
                    width: None,
                    height: Some(128),
                },
                (96, 128),
            ),
        ] {
            edit.flux3_reference_output = Some(intent);
            let image = render(&edit);
            assert_eq!(
                (image.width, image.height),
                size,
                "either explicit side suppresses output crop"
            );
        }
        edit.conditioning = refs.clone();
        edit.flux3_reference_output = Some(ReferenceOutputSize::default());
        let ten = render(&edit);
        assert_eq!((ten.width, ten.height), (80, 64));
        let mut text_only = edit.clone();
        text_only.width = 80;
        text_only.height = 64;
        text_only.conditioning.clear();
        text_only.flux3_reference_output = None;
        let plain = render(&text_only);
        assert_ne!(
            ten.pixels, plain.pixels,
            "ordinary Generator must consume references, not silently render T2I"
        );
        edit.conditioning.reverse();
        let reverse = render(&edit);
        assert_eq!(
            (reverse.width, reverse.height),
            (64, 96),
            "first prepared reference controls auto geometry, never reordered"
        );
        let mut bad = edit.clone();
        bad.conditioning.push(refs[0].clone());
        let mut progress = Vec::new();
        assert!(g.generate(&bad, &mut |p| progress.push(p)).is_err());
        assert!(
            progress.is_empty(),
            "excess references fail before any generation phase"
        );
        bad = edit.clone();
        bad.strength = Some(1.0);
        assert!(g.generate(&bad, &mut |p| progress.push(p)).is_err());
        assert!(
            progress.is_empty(),
            "unsupported strength cannot become silent T2I"
        );
        bad = edit.clone();
        bad.conditioning = vec![Conditioning::ReferenceRgba {
            image: RgbaImage::default(),
            strength: None,
        }];
        assert!(g.generate(&bad, &mut |p| progress.push(p)).is_err());
        assert!(
            progress.is_empty(),
            "invalid pixel carriers fail before encoding/PUP/render"
        );
    }
}

#[test]
fn effective_caption_is_recorded_before_encoding_and_prompt_derived_seeding() {
    use candle_gen::gen_core::{PromptEnhancementReport, PromptEnhancementSink};
    use std::sync::{Arc, Mutex};
    let mut loaded = spec(OffloadPolicy::Resident);
    loaded.components.insert(
        "prompt_upsampler".into(),
        WeightsSource::Dir(crate::common::fixtures().join("tiny-pup-snapshot")),
    );
    let generator = candle_gen_flux3::provider_registry()
        .unwrap()
        .load(ID, &loaded)
        .unwrap();
    assert!(
        generator
            .memory_strategy_contract()
            .unwrap()
            .asset_facts
            .overlay_bytes
            > 0
    );
    let reports: Arc<Mutex<Vec<PromptEnhancementReport>>> = Arc::default();
    let sink = reports.clone();
    let mut enhanced = request();
    enhanced.prompt = "a red bicycle leaning against a brick wall".into();
    enhanced.seed = None;
    enhanced.enhance_prompt = true;
    enhanced.enhance_max_tokens = Some(12);
    enhanced.prompt_enhancement = PromptEnhancementSink::new(move |r| sink.lock().unwrap().push(r));
    let first = generator.generate(&enhanced, &mut |_| {}).unwrap();
    let effective = {
        let captured = reports.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_ne!(captured[0].effective_prompt, enhanced.prompt);
        captured[0].effective_prompt.clone()
    };
    let replay = GenerationRequest {
        prompt: effective,
        enhance_prompt: false,
        enhance_max_tokens: None,
        prompt_enhancement: Default::default(),
        ..enhanced
    };
    let second = generator.generate(&replay, &mut |_| {}).unwrap();
    match (first, second) {
        (GenerationOutput::Images(a), GenerationOutput::Images(b)) => {
            assert_eq!(a[0].pixels, b[0].pixels)
        }
        other => panic!("expected images from both native requests: {other:?}"),
    }
}

#[test]
fn gen_core_conformance_resident() {
    gen_core_testkit::conformance(|| load(OffloadPolicy::Resident), &profile());
}

#[test]
fn gen_core_conformance_sequential() {
    gen_core_testkit::conformance(|| load(OffloadPolicy::Sequential), &profile());
}

#[test]
fn progress_is_steps_then_one_decoding_and_output_is_rgb8() {
    let g = load(OffloadPolicy::Resident);
    let mut events = Vec::new();
    let out = g
        .generate(&request(), &mut |p| events.push(format!("{p:?}")))
        .unwrap();
    assert_eq!(
        events,
        [
            "ConditioningComplete",
            "Step { current: 1, total: 3 }",
            "Step { current: 2, total: 3 }",
            "Step { current: 3, total: 3 }",
            "Decoding",
        ]
    );
    match out {
        GenerationOutput::Images(images) => {
            assert_eq!(images.len(), 1);
            assert_eq!((images[0].width, images[0].height), (80, 64));
            assert_eq!(images[0].pixels.len(), 80 * 64 * 3);
        }
        other => panic!("expected images, got {other:?}"),
    }
    // `count` renders one image after another from `seed + i`: the second image of a count-2
    // request is the single image of `seed + 1`.
    let two = g
        .generate(
            &GenerationRequest {
                count: 2,
                ..request()
            },
            &mut |_| {},
        )
        .unwrap();
    let one = g
        .generate(
            &GenerationRequest {
                seed: Some(43),
                ..request()
            },
            &mut |_| {},
        )
        .unwrap();
    match (two, one) {
        (GenerationOutput::Images(two), GenerationOutput::Images(one)) => {
            assert_eq!(two.len(), 2);
            assert_eq!(two[1].pixels, one[0].pixels);
            assert_ne!(two[0].pixels, two[1].pixels);
        }
        _ => panic!("expected images"),
    }
}

/// An unseeded request is seeded from its caption (`generate.py`'s `sha256(caption)`): it
/// reproduces, and equals the explicitly seeded request at that seed.
#[test]
fn an_unseeded_request_is_seeded_from_its_prompt() {
    let g = load(OffloadPolicy::Resident);
    let unseeded = GenerationRequest {
        seed: None,
        ..request()
    };
    let a = g.generate(&unseeded, &mut |_| {}).unwrap();
    let b = g.generate(&unseeded, &mut |_| {}).unwrap();
    let explicit = g
        .generate(
            &GenerationRequest {
                seed: Some(candle_gen::gen_core::flux3::prompt_seed(&unseeded.prompt)),
                ..request()
            },
            &mut |_| {},
        )
        .unwrap();
    let other = g
        .generate(
            &GenerationRequest {
                seed: Some(candle_gen::gen_core::flux3::prompt_seed("another caption")),
                ..request()
            },
            &mut |_| {},
        )
        .unwrap();
    match (a, b, explicit, other) {
        (
            GenerationOutput::Images(a),
            GenerationOutput::Images(b),
            GenerationOutput::Images(e),
            GenerationOutput::Images(o),
        ) => {
            assert_eq!(a[0].pixels, b[0].pixels, "reproduces per prompt");
            assert_eq!(a[0].pixels, e[0].pixels, "is the sha256(caption) seed");
            assert_ne!(a[0].pixels, o[0].pixels, "another caption, another seed");
        }
        _ => panic!("expected images"),
    }
}

/// A pre-tripped cancel refuses before any work; the request's own flag tripped mid-run surfaces
/// as the typed `Canceled`, never a partial image.
#[test]
fn cancellation_is_typed() {
    let g = load(OffloadPolicy::Resident);
    let cancel = CancelFlag::new();
    cancel.cancel();
    let req = GenerationRequest {
        cancel: cancel.clone(),
        ..request()
    };
    match g.generate(&req, &mut |_| {}) {
        Err(CoreError::Canceled) => {}
        other => panic!("expected Canceled, got {other:?}"),
    }
    let cancel = CancelFlag::new();
    let req = GenerationRequest {
        cancel: cancel.clone(),
        steps: Some(6),
        ..request()
    };
    let flag = cancel.clone();
    match g.generate(&req, &mut |p| {
        if matches!(p, Progress::Step { current: 2, .. }) {
            flag.cancel();
        }
    }) {
        Err(CoreError::Canceled) => {}
        other => panic!("expected Canceled mid-run, got {other:?}"),
    }
}

/// The flow map's typed refusals reach the caller through `validate` and `generate` alike, as
/// `Unsupported` (a capability the route does not do) or `Msg` (a malformed value) — never a
/// silent fallback.
#[test]
fn unhonourable_controls_are_refused_at_validation() {
    let g = load(OffloadPolicy::Resident);
    g.validate(&request()).unwrap();
    for (what, req, unsupported) in [
        (
            "guidance",
            GenerationRequest {
                guidance: Some(3.5),
                ..request()
            },
            true,
        ),
        // The descriptor advertises no guidance axis at all: 1.0 is refused like any other value,
        // by `Capabilities::validate_request` and by `GenerationParams::resolve` alike.
        (
            "guidance 1.0",
            GenerationRequest {
                guidance: Some(1.0),
                ..request()
            },
            true,
        ),
        (
            "negative prompt",
            GenerationRequest {
                negative_prompt: Some("blurry".into()),
                ..request()
            },
            true,
        ),
        (
            "off-grid size",
            GenerationRequest {
                width: 72,
                ..request()
            },
            false,
        ),
        (
            "size under the 64-px floor",
            GenerationRequest {
                width: 48,
                ..request()
            },
            false,
        ),
        (
            "sampler",
            GenerationRequest {
                sampler: Some("dpmpp_2m".into()),
                ..request()
            },
            true,
        ),
        (
            "scheduler",
            GenerationRequest {
                scheduler: Some("beta".into()),
                ..request()
            },
            true,
        ),
    ] {
        let err = g.validate(&req).expect_err(what);
        if unsupported {
            assert!(matches!(err, CoreError::Unsupported(_)), "{what}: {err:?}");
        } else {
            assert!(
                matches!(err, CoreError::Unsupported(_) | CoreError::Msg(_)),
                "{what}: {err:?}"
            );
        }
        assert!(g.generate(&req, &mut |_| {}).is_err(), "{what}: generate");
    }
}

/// Explicit `text_encoder` / `autoencoder` components override the bundled copies; a package
/// without them and without overrides is a load error naming the missing resource; an unknown
/// component key is refused.
#[test]
fn components_override_the_bundled_copies() {
    let registry = candle_gen_flux3::provider_registry().unwrap();
    // A package directory holding only the DiT + config: the bundled copies are absent.
    let dir = tempfile::tempdir().unwrap();
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(tiny_package().join(name), dir.path().join(name)).unwrap();
    }
    let bare = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let err = match registry.load(ID, &bare) {
        Err(err) => err,
        Ok(_) => panic!("no autoencoder: the bare package must not load"),
    };
    assert!(err.to_string().contains("ae.safetensors"), "{err}");
    let mut spec = bare.clone();
    spec.components.insert(
        AUTOENCODER_COMPONENT.into(),
        WeightsSource::File(tiny_autoencoder()),
    );
    let err = match registry.load(ID, &spec) {
        Err(err) => err,
        Ok(_) => panic!("no text encoder: the package must not load without one"),
    };
    assert!(err.to_string().contains("text encoder"), "{err}");
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    let g = registry.load(ID, &spec).expect("overrides load");
    let out = g.generate(&request(), &mut |_| {}).unwrap();
    let bundled = load(OffloadPolicy::Resident)
        .generate(&request(), &mut |_| {})
        .unwrap();
    match (out, bundled) {
        (GenerationOutput::Images(a), GenerationOutput::Images(b)) => {
            assert_eq!(a[0].pixels, b[0].pixels, "same components, same image");
        }
        _ => panic!("expected images"),
    }
    let mut unknown = spec.clone();
    unknown
        .components
        .insert("upsampler".into(), WeightsSource::Dir(tiny_text_encoder()));
    match registry.load(ID, &unknown) {
        Err(CoreError::Unsupported(m)) => assert!(m.contains("upsampler"), "{m}"),
        other => panic!("expected a typed refusal, got {:?}", other.err()),
    }
}

/// A CFG (base) package is a typed refusal at load, never sampled as a flow map; a quantized
/// package likewise (sc-26241).
#[test]
fn base_and_quantized_packages_are_typed_refusals() {
    let registry = candle_gen_flux3::provider_registry().unwrap();
    let file = LoadSpec::new(WeightsSource::File(
        tiny_package().join("model.safetensors"),
    ));
    assert!(
        matches!(registry.load(ID, &file), Err(CoreError::Unsupported(message)) if message.contains("package directory"))
    );
    assert!(
        matches!(registry.memory_strategy_contract(ID, &file), Err(CoreError::Unsupported(message)) if message.contains("package directory"))
    );
    let config = std::fs::read_to_string(tiny_package().join("config.json")).unwrap();
    for (what, needle, edit) in [
        (
            "base checkpoint",
            "CFG checkpoint",
            config
                .replace("\"flow_map\": true", "\"flow_map\": false")
                .replace("\"guidance\": null", "\"guidance\": 3.5")
                // A well-formed base package (frozen 50/3.5) so the ROUTE mismatch is what refuses.
                .replace("\"steps\": 4", "\"steps\": 50"),
        ),
        (
            "quantized variant",
            "quantization",
            config.replace("\"quantization\": null", "\"quantization\": \"fp8r\""),
        ),
    ] {
        assert_ne!(edit, config, "{what}: the edit must change the config");
        let dir = tempfile::tempdir().unwrap();
        for name in ["model.safetensors", "ae.safetensors"] {
            std::fs::copy(tiny_package().join(name), dir.path().join(name)).unwrap();
        }
        std::fs::write(dir.path().join("config.json"), edit).unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(tiny_text_encoder()),
        );
        match registry.load(ID, &spec) {
            Err(CoreError::Unsupported(m)) => assert!(m.contains(needle), "{what}: {m}"),
            other => panic!("{what}: expected a typed refusal, got {:?}", other.err()),
        }
    }
}

#[test]
fn base_cfg_adapter_loads_through_the_public_provider_for_both_residencies() {
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Precision};
    let fixture = crate::common::fixtures();
    let registry = candle_gen_flux3::provider_registry().unwrap();
    for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
        let mut loaded = LoadSpec::new(WeightsSource::Dir(fixture.join("tiny-base-snapshot")))
            .with_offload_policy(policy);
        loaded.precision = Precision::Fp32;
        loaded.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(tiny_text_encoder()),
        );
        loaded.components.insert(
            AUTOENCODER_COMPONENT.into(),
            WeightsSource::File(tiny_autoencoder()),
        );
        loaded.adapters.push(AdapterSpec::new(
            fixture.join("flux3_base_lora.safetensors"),
            -0.25,
            AdapterKind::Lora,
        ));
        let generator = registry.load("flux3_klein_base", &loaded).unwrap();
        assert_eq!(generator.descriptor().id, "flux3_klein_base");
        assert!(generator.descriptor().capabilities.supports_guidance);
        assert!(!generator.descriptor().capabilities.supports_true_cfg);
        assert!(generator.descriptor().capabilities.supports_negative_prompt);
        assert!(generator.descriptor().capabilities.supports_lora);
        assert_eq!(
            generator.memory_strategy_contract().unwrap().provider_id,
            "flux3_klein_base"
        );
        let request = GenerationRequest {
            guidance: Some(2.25),
            negative_prompt: Some("a cat on a roof".into()),
            ..request()
        };
        generator.validate(&request).unwrap();
        for guidance in [0.0f32, 1.0, 3.5] {
            let mut standard = request.clone();
            standard.guidance = Some(guidance);
            generator.validate(&standard).unwrap();
        }
        let mut separate = request.clone();
        separate.true_cfg = Some(4.25);
        assert!(matches!(
            generator.validate(&separate),
            Err(CoreError::Unsupported(_))
        ));
        let mut progress = Vec::new();
        let GenerationOutput::Images(images) = generator
            .generate(&request, &mut |p| progress.push(p))
            .unwrap()
        else {
            panic!("image output required")
        };
        assert_eq!(images.len(), 1);
        assert_eq!((images[0].width, images[0].height), (80, 64));
        assert_eq!(images[0].pixels.len(), 80 * 64 * 3);
        assert!(progress.iter().any(|p| matches!(
            p,
            Progress::Step {
                current: 3,
                total: 3
            }
        )));
        let reports = generator.adapter_apply_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].applied, 6);
        assert!(reports[0].skipped.is_empty());
        loaded.adapters[0].scale = f32::NAN;
        assert!(matches!(
            registry.load("flux3_klein_base", &loaded),
            Err(CoreError::Unsupported(_))
        ));
    }
}

#[test]
fn callback_cancel_at_conditioning_boundary_stops_before_denoise() {
    for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
        let generator = load(policy);
        let req = request();
        let mut events = Vec::new();
        let result = generator.generate(&req, &mut |event| {
            events.push(event);
            if event == Progress::ConditioningComplete {
                req.cancel.cancel();
            }
        });
        assert!(
            matches!(result, Err(CoreError::Canceled)),
            "{policy:?}: {result:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|&&p| p == Progress::ConditioningComplete)
                .count(),
            1
        );
        assert!(
            !events
                .iter()
                .any(|p| matches!(p, Progress::Step { .. } | Progress::Decoding)),
            "cancellation at the boundary must stop before the first forward: {events:?}"
        );
    }
}

//! The gen-core generator contract on the committed miniature package, through the explicit
//! provider catalog's production load path (`provider_registry().load(id, spec)` — the
//! composition the SceneWorks worker reaches through `mlx-gen-catalog`): validate honesty,
//! progress (`Step 1..=N` then exactly one `Decoding`), typed cancellation, seeded determinism,
//! both residency policies, the component overrides, typed errors for a missing resource and for
//! the controls the flow map cannot honour.

use mlx_gen::gen_core::flux3::{AUTOENCODER_COMPONENT, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{Error as CoreError, Progress};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
};

use crate::common::{cpu_process, tiny_autoencoder, tiny_package, tiny_text_encoder};

const ID: &str = "flux3_klein";

fn spec(policy: OffloadPolicy) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(tiny_package())).with_offload_policy(policy)
}

fn load(policy: OffloadPolicy) -> Box<dyn mlx_gen::Generator> {
    mlx_gen_flux3::provider_registry()
        .unwrap()
        .load(ID, &spec(policy))
        .expect("the tiny package loads through the catalog path")
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: "a red fox in the snow".to_owned(),
        width: 48,
        height: 32,
        steps: 3,
        seed: 42,
        cancel_steps: 6,
    }
}

fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the snow".to_owned(),
        width: 48,
        height: 32,
        steps: Some(3),
        seed: Some(42),
        ..Default::default()
    }
}

#[test]
fn gen_core_conformance_resident() {
    cpu_process(
        "generator_contract::gen_core_conformance_resident",
        gen_core_conformance_resident_body,
    );
}

fn gen_core_conformance_resident_body() {
    gen_core_testkit::conformance(|| load(OffloadPolicy::Resident), &profile());
}

#[test]
fn gen_core_conformance_sequential() {
    cpu_process(
        "generator_contract::gen_core_conformance_sequential",
        gen_core_conformance_sequential_body,
    );
}

fn gen_core_conformance_sequential_body() {
    gen_core_testkit::conformance(|| load(OffloadPolicy::Sequential), &profile());
}

#[test]
fn progress_is_steps_then_one_decoding_and_output_is_rgb8() {
    cpu_process(
        "generator_contract::progress_is_steps_then_one_decoding_and_output_is_rgb8",
        progress_is_steps_then_one_decoding_and_output_is_rgb8_body,
    );
}

fn progress_is_steps_then_one_decoding_and_output_is_rgb8_body() {
    let g = load(OffloadPolicy::Resident);
    let mut events = Vec::new();
    let out = g
        .generate(&request(), &mut |p| events.push(format!("{p:?}")))
        .unwrap();
    assert_eq!(
        events,
        [
            "Step { current: 1, total: 3 }",
            "Step { current: 2, total: 3 }",
            "Step { current: 3, total: 3 }",
            "Decoding",
        ]
    );
    match out {
        GenerationOutput::Images(images) => {
            assert_eq!(images.len(), 1);
            assert_eq!((images[0].width, images[0].height), (48, 32));
            assert_eq!(images[0].pixels.len(), 48 * 32 * 3);
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

/// A pre-tripped cancel refuses before any work; the request's own flag tripped mid-run surfaces
/// as the typed `Canceled`, never a partial image.
#[test]
fn cancellation_is_typed() {
    cpu_process(
        "generator_contract::cancellation_is_typed",
        cancellation_is_typed_body,
    );
}

fn cancellation_is_typed_body() {
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

/// The flow map's typed refusals reach the caller through `validate` and `generate` alike.
#[test]
fn unhonourable_controls_are_refused_at_validation() {
    cpu_process(
        "generator_contract::unhonourable_controls_are_refused_at_validation",
        unhonourable_controls_are_refused_at_validation_body,
    );
}

fn unhonourable_controls_are_refused_at_validation_body() {
    let g = load(OffloadPolicy::Resident);
    g.validate(&request()).unwrap();
    for (what, req) in [
        (
            "guidance",
            GenerationRequest {
                guidance: Some(3.5),
                ..request()
            },
        ),
        (
            "negative prompt",
            GenerationRequest {
                negative_prompt: Some("blurry".into()),
                ..request()
            },
        ),
        (
            "off-grid size",
            GenerationRequest {
                width: 40,
                ..request()
            },
        ),
        (
            "sampler",
            GenerationRequest {
                sampler: Some("dpmpp_2m".into()),
                ..request()
            },
        ),
    ] {
        let err = g.validate(&req).expect_err(what);
        assert!(
            matches!(err, CoreError::Unsupported(_) | CoreError::Msg(_)),
            "{what}: {err:?}"
        );
        assert!(g.generate(&req, &mut |_| {}).is_err(), "{what}: generate");
    }
}

/// Explicit `text_encoder` / `autoencoder` components override the bundled copies; a package
/// without them and without overrides is a load error naming the missing resource; an unknown
/// component key is refused.
#[test]
fn components_override_the_bundled_copies() {
    cpu_process(
        "generator_contract::components_override_the_bundled_copies",
        components_override_the_bundled_copies_body,
    );
}

fn components_override_the_bundled_copies_body() {
    let registry = mlx_gen_flux3::provider_registry().unwrap();
    // A package directory holding only the DiT + config: the bundled copies are absent.
    let dir = tempfile::tempdir().unwrap();
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(tiny_package().join(name), dir.path().join(name)).unwrap();
    }
    let bare = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let err = registry.load(ID, &bare).err().expect("no autoencoder");
    assert!(err.to_string().contains("ae.safetensors"), "{err}");
    let mut spec = bare.clone();
    spec.components.insert(
        AUTOENCODER_COMPONENT.into(),
        WeightsSource::File(tiny_autoencoder()),
    );
    let err = registry.load(ID, &spec).err().expect("no text encoder");
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

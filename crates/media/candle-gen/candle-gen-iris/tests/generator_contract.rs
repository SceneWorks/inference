//! The gen-core generator contract on the committed miniature snapshot, through the explicit
//! provider catalog's production load path (`provider_registry().load(id, spec)` — the composition
//! the SceneWorks worker reaches through `candle-gen-catalog` / `runtime-cuda`): validate honesty,
//! progress (`Step 1..=N` then exactly one `Decoding`), typed cancellation (pre-tripped and mid-run,
//! never a partial image), seeded determinism, both residency policies, typed errors for
//! missing/incomplete task resources (the control surface itself is `controls_contract`). Mirrors the MLX
//! twin's `generator_contract` case for case.

use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use candle_gen::gen_core::{
    CancelFlag, Error as CoreError, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    OffloadPolicy, Progress, WeightsSource,
};

use crate::common::{tiny_backbone, tiny_text_encoder};

const ID: &str = "iris_3b";

fn spec(policy: OffloadPolicy) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone())).with_offload_policy(policy);
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    spec
}

fn load(policy: OffloadPolicy) -> Box<dyn Generator> {
    candle_gen_iris::provider_registry()
        .unwrap()
        .load(ID, &spec(policy))
        .expect("the tiny snapshot loads through the catalog path")
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: "a red fox in the snow".to_owned(),
        width: 32,
        height: 16,
        steps: 3,
        seed: 42,
        cancel_steps: 6,
    }
}

fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the snow".to_owned(),
        width: 32,
        height: 16,
        steps: Some(3),
        seed: Some(42),
        ..Default::default()
    }
}

/// The load gate on the catalog's production path: with the declared `text_encoder` component
/// removed, `load` fails naming it; an unrecognized component key is refused. The miniature spec
/// otherwise loads, so neither half can pass on an unrelated load error.
#[test]
fn missing_or_unknown_component_fails_at_load() {
    let registry = candle_gen_iris::provider_registry().unwrap();
    gen_core_testkit::check_component_load_gate(
        |spec| registry.load(ID, spec),
        &spec(OffloadPolicy::Resident),
        candle_gen_iris::model::descriptor().required_components,
    )
    .expect("Iris load must gate on every declared required component");
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
            "Step { current: 1, total: 3 }",
            "Step { current: 2, total: 3 }",
            "Step { current: 3, total: 3 }",
            "Decoding",
        ]
    );
    match out {
        GenerationOutput::Images(images) => {
            assert_eq!(images.len(), 1);
            assert_eq!((images[0].width, images[0].height), (32, 16));
            assert_eq!(images[0].pixels.len(), 32 * 16 * 3);
        }
        other => panic!("expected Images, got {other:?}"),
    }
}

#[test]
fn count_folds_into_one_bar_with_distinct_seeds() {
    let g = load(OffloadPolicy::Resident);
    let mut steps = Vec::new();
    let mut req = request();
    req.count = 2;
    let out = g
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                steps.push((current, total));
            }
        })
        .unwrap();
    assert_eq!(steps, (1..=6).map(|c| (c, 6)).collect::<Vec<_>>());
    let GenerationOutput::Images(images) = out else {
        panic!("expected images");
    };
    assert_eq!(images.len(), 2);
    assert_ne!(images[0].pixels, images[1].pixels, "seed + i per image");
}

#[test]
fn a_seed_reproduces_and_bf16_is_the_default_compute() {
    let g = load(OffloadPolicy::Resident);
    let render = || match g.generate(&request(), &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut images) => images.remove(0).pixels,
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(render(), render());
    assert_eq!(
        candle_gen_iris::compute_dtype(&spec(OffloadPolicy::Resident)),
        candle_gen::candle_core::DType::BF16
    );
}

#[test]
fn mid_run_cancel_is_typed_and_returns_no_image() {
    let g = load(OffloadPolicy::Resident);
    let mut req = request();
    req.steps = Some(6);
    let cancel = CancelFlag::new();
    req.cancel = cancel.clone();
    let mut seen = 0;
    let result = g.generate(&req, &mut |p| {
        if let Progress::Step { current, .. } = p {
            seen = current;
            if current == 2 {
                cancel.cancel();
            }
        }
    });
    assert!(
        matches!(result, Err(CoreError::Canceled)),
        "{:?}",
        result.err()
    );
    assert_eq!(seen, 2, "no step after the flag trips");
}

#[test]
fn missing_or_incomplete_resources_are_load_errors() {
    let registry = candle_gen_iris::provider_registry().unwrap();
    // no text encoder component
    let bare = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    let err = registry.load(ID, &bare).err().expect("must fail");
    assert!(err.to_string().contains(TEXT_ENCODER_COMPONENT), "{err}");
    // a text encoder directory without its tokenizer
    let dir = tempfile::tempdir().unwrap();
    for name in ["config.json", "model.safetensors", "tokenizer_config.json"] {
        std::fs::copy(tiny_text_encoder().join(name), dir.path().join(name)).unwrap();
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(dir.path().to_path_buf()),
    );
    let err = registry.load(ID, &spec).err().expect("must fail");
    assert!(err.to_string().contains("tokenizer.json"), "{err}");
    // a backbone directory without its weights
    let bb = tempfile::tempdir().unwrap();
    std::fs::copy(
        tiny_backbone().join("config.yaml"),
        bb.path().join("config.yaml"),
    )
    .unwrap();
    let mut spec = self::spec(OffloadPolicy::Resident);
    spec.weights = WeightsSource::Dir(bb.path().to_path_buf());
    let err = registry.load(ID, &spec).err().expect("must fail");
    assert!(err.to_string().contains("model.safetensors"), "{err}");
    // an unknown component key
    let mut unknown = self::spec(OffloadPolicy::Resident);
    unknown
        .components
        .insert("vae".into(), WeightsSource::Dir(tiny_backbone()));
    assert!(registry.load(ID, &unknown).is_err());
}

#[test]
fn a_truncated_text_encoder_shard_is_a_load_error_under_both_policies() {
    // An indexed snapshot whose index names a shard that is not on disk: `Resident` refuses at load,
    // `Sequential` (which defers the encoder to the first request) refuses at generate — both typed
    // errors naming the shard, never a panic or a silent fallback.
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ] {
        std::fs::copy(tiny_text_encoder().join(name), dir.path().join(name)).unwrap();
    }
    std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"metadata":{},"weight_map":{"model.language_model.norm.weight":"model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();
    let registry = candle_gen_iris::provider_registry().unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(dir.path().to_path_buf()),
    );
    let err = registry.load(ID, &spec).err().expect("resident must fail");
    assert!(err.to_string().contains("model-00001-of-00001"), "{err}");

    let spec = spec.with_offload_policy(OffloadPolicy::Sequential);
    let g = registry
        .load(ID, &spec)
        .expect("sequential defers the encoder");
    let err = g
        .generate(&request(), &mut |_| {})
        .expect_err("sequential must fail at generate");
    assert!(err.to_string().contains("model-00001-of-00001"), "{err}");
}

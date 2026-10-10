//! The gen-core generator contract on the committed miniature snapshot, through the explicit
//! provider catalog's production load path (`provider_registry().load(id, spec)` — the composition
//! the SceneWorks worker reaches through `mlx-gen-catalog`): validate honesty, progress (`Step
//! 1..=N` then exactly one `Decoding`), typed cancellation (pre-tripped and mid-run, never a partial
//! image), seeded determinism, the CFG-off render, both residency policies, typed errors for
//! missing/incomplete task resources and for controls the route does not honour.

use mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use mlx_gen::gen_core::{Error as CoreError, Progress};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
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

fn load(policy: OffloadPolicy) -> Box<dyn mlx_gen::Generator> {
    mlx_gen_iris::provider_registry()
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
fn controls_the_route_does_not_honour_are_refused() {
    let g = load(OffloadPolicy::Resident);
    let mut req = request();
    req.sampler = Some("euler".into());
    assert!(matches!(g.validate(&req), Err(CoreError::Unsupported(_))));
    let mut req = request();
    req.scheduler_shift = Some(3.0);
    assert!(
        matches!(g.validate(&req), Err(CoreError::Unsupported(m)) if m.contains("scheduler_shift"))
    );
    let mut req = request();
    req.true_cfg = Some(4.0);
    assert!(g.validate(&req).is_err());
    let mut req = request();
    req.width = 40; // not a multiple of the 16-px patch
    assert!(g.validate(&req).is_err());
    // CFG off: the negative prompt would never be evaluated.
    let mut req = request();
    req.guidance = Some(1.0);
    req.negative_prompt = Some("blurry".into());
    assert!(
        matches!(g.validate(&req), Err(CoreError::Unsupported(m)) if m.contains("negative_prompt"))
    );
    req.negative_prompt = Some(String::new());
    g.validate(&req).unwrap();
}

#[test]
fn missing_or_incomplete_resources_are_load_errors() {
    let registry = mlx_gen_iris::provider_registry().unwrap();
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
    // an unknown component key
    let mut unknown = self::spec(OffloadPolicy::Resident);
    unknown
        .components
        .insert("vae".into(), WeightsSource::Dir(tiny_backbone()));
    assert!(registry.load(ID, &unknown).is_err());
}

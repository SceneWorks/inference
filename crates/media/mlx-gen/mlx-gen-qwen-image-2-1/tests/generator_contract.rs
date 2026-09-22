//! The gen-core generator contract on the committed miniature snapshot, through the explicit
//! provider catalog's production load path (`provider_registry().load(id, spec)` — the same
//! composition the SceneWorks worker reaches via `mlx-gen-catalog`): validate honesty, progress
//! (`Step 1..=N` then exactly one `Decoding`), typed cancellation (pre-tripped and mid-run),
//! seeded determinism, the CFG-off render, both residency policies, and actionable errors.
//! Runs by default: the tiny snapshot is ~500 KB and a render takes well under a second.

use mlx_gen::gen_core::{Error as CoreError, Progress};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};

use crate::common::tiny_snapshot;

const ID: &str = "qwen_image_2_1";

fn spec(policy: OffloadPolicy) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(tiny_snapshot())).with_offload_policy(policy)
}

fn load(policy: OffloadPolicy) -> Box<dyn mlx_gen::Generator> {
    mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(ID, &spec(policy))
        .expect("the tiny snapshot loads through the catalog path")
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
        steps: 3,
        seed: 42,
        cancel_steps: 6,
    }
}

fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
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
            assert_eq!((images[0].width, images[0].height), (32, 32));
            assert_eq!(
                images[0].pixels.len(),
                32 * 32 * 3,
                "RGB8, alpha composited away"
            );
        }
        other => panic!("expected images, got {other:?}"),
    }
}

#[test]
fn sequential_policy_reports_loading_phases_and_matches_resident_pixels() {
    let resident = load(OffloadPolicy::Resident);
    let sequential = load(OffloadPolicy::Sequential);
    let a = resident.generate(&request(), &mut |_| {}).unwrap();
    let mut loading = 0;
    let b = sequential
        .generate(&request(), &mut |p| {
            if matches!(p, Progress::Loading(_)) {
                loading += 1;
            }
        })
        .unwrap();
    assert!(
        loading >= 2,
        "Sequential emits a Loading event per phase, got {loading}"
    );
    let (GenerationOutput::Images(a), GenerationOutput::Images(b)) = (a, b) else {
        panic!("images expected");
    };
    assert_eq!(
        a[0].pixels, b[0].pixels,
        "residency policy never changes the render"
    );
}

#[test]
fn count_and_seed_reach_the_runtime() {
    let g = load(OffloadPolicy::Resident);
    let mut req = request();
    req.count = 2;
    let GenerationOutput::Images(images) = g.generate(&req, &mut |_| {}).unwrap() else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 2);
    assert_ne!(
        images[0].pixels, images[1].pixels,
        "count > 1 advances the seed"
    );
    let mut second = request();
    second.seed = Some(43);
    let GenerationOutput::Images(seeded) = g.generate(&second, &mut |_| {}).unwrap() else {
        panic!("images expected");
    };
    assert_eq!(
        images[1].pixels,
        seeded.pixels_of(0),
        "seed + 1 == second image of the batch"
    );
}

trait PixelsOf {
    fn pixels_of(&self, i: usize) -> Vec<u8>;
}

impl PixelsOf for Vec<mlx_gen::Image> {
    fn pixels_of(&self, i: usize) -> Vec<u8> {
        self[i].pixels.clone()
    }
}

#[test]
fn guidance_with_a_negative_prompt_changes_the_render() {
    let g = load(OffloadPolicy::Resident);
    let base = g.generate(&request(), &mut |_| {}).unwrap();
    let mut req = request();
    req.negative_prompt = Some("blurry low quality photo".to_owned());
    req.true_cfg = Some(2.5);
    let guided = g.generate(&req, &mut |_| {}).unwrap();
    let (GenerationOutput::Images(a), GenerationOutput::Images(b)) = (base, guided) else {
        panic!("images expected");
    };
    assert_ne!(a[0].pixels, b[0].pixels);
    // `true_cfg = 1.0` (the upstream default) leaves the negative prompt inert.
    req.true_cfg = Some(1.0);
    let GenerationOutput::Images(off) = g.generate(&req, &mut |_| {}).unwrap() else {
        panic!("images expected");
    };
    assert_eq!(a[0].pixels, off[0].pixels);
}

#[test]
fn invalid_requests_fail_with_actionable_errors_before_any_work() {
    let g = load(OffloadPolicy::Resident);
    let mut req = request();
    req.width = 48;
    let err = g.validate(&req).unwrap_err().to_string();
    assert!(err.contains("32"), "grid error names the multiple: {err}");
    let mut req = request();
    req.steps = Some(1);
    let err = g.validate(&req).unwrap_err().to_string();
    assert!(err.contains("steps must be >= 2"), "{err}");
    let mut req = request();
    req.sampler = Some("nonsense".into());
    assert!(g.validate(&req).is_err());
    let mut req = request();
    req.width = 4096;
    req.height = 4096;
    assert!(g.validate(&req).is_err());
}

#[test]
fn a_pre_tripped_cancel_is_typed_and_a_mid_run_cancel_stops_within_a_step() {
    let g = load(OffloadPolicy::Resident);
    let cancel = CancelFlag::new();
    cancel.cancel();
    let mut req = request();
    req.cancel = cancel;
    assert!(matches!(
        g.generate(&req, &mut |_| {}),
        Err(CoreError::Canceled)
    ));

    let cancel = CancelFlag::new();
    let mut req = request();
    req.steps = Some(6);
    req.cancel = cancel.clone();
    let mut seen = 0;
    let result = g.generate(&req, &mut |p| {
        if let Progress::Step { .. } = p {
            seen += 1;
            cancel.cancel();
        }
    });
    assert!(matches!(result, Err(CoreError::Canceled)), "{result:?}");
    assert!(
        seen <= 2,
        "stopped within a step of the trip, saw {seen} steps"
    );
}

#[test]
fn load_time_q8_quantizes_the_dit_and_still_renders() {
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let g = registry
        .load(ID, &spec(OffloadPolicy::Resident).with_quant(Quant::Q8))
        .expect("Q8 load-time quantization of the DiT");
    let GenerationOutput::Images(images) = g.generate(&request(), &mut |_| {}).unwrap() else {
        panic!("images expected");
    };
    assert_eq!(images[0].pixels.len(), 32 * 32 * 3);
}

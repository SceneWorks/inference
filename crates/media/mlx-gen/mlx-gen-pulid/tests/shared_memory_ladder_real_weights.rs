//! SC-15527 real-Metal conformance for the PuLID-FLUX memory ladder.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    standard_memory_behavior_context, MemoryBehaviorRoute, MemoryMode, MemoryNumericTier,
    MemoryRunOutcome, MemoryStrategy, MemoryStrategySupport,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, IdentityWeights, LoadShape, LoadSpec,
    OffloadPolicy, Quant, WeightsSource,
};
use mlx_gen_face::FaceAnalysis;

fn env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}"))
}

fn pixels(output: GenerationOutput) -> Vec<u8> {
    match output {
        GenerationOutput::Images(images) => {
            assert_eq!((images[0].width, images[0].height), (1024, 1024));
            assert!(images[0].pixels.iter().any(|&value| value != 0));
            images[0].pixels.clone()
        }
        other => panic!("expected images, got {other:?}"),
    }
}

fn mean_abs_delta(left: &[u8], right: &[u8]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| a.abs_diff(b) as f64)
        .sum::<f64>()
        / left.len() as f64
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let l = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let r = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (l * r)
}

fn face_stack(root: &Path) -> FaceAnalysis {
    FaceAnalysis::load(
        &Weights::from_file(root.join("scrfd_10g.safetensors")).unwrap(),
        &Weights::from_file(root.join("arcface_iresnet100.safetensors")).unwrap(),
    )
    .unwrap()
    .with_parser(&Weights::from_file(root.join("bisenet_parsing.safetensors")).unwrap())
    .unwrap()
}

#[test]
#[ignore = "requires exact cached PuLID/FLUX artifacts and a Metal device"]
fn ladder_preserves_identity_quality_and_reduces_request_peak() {
    let face_dir = env("PULID_FACE_DIR");
    let mut optimized_spec = LoadSpec::new(WeightsSource::Dir(env("PULID_FLUX_Q4_ROOT")))
        .with_quant(Quant::Q4)
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    optimized_spec.identity = Some(IdentityWeights {
        encoder: Some(WeightsSource::File(env("PULID_ENCODER"))),
        eva: Some(WeightsSource::File(env("PULID_EVA"))),
        face_dir: Some(WeightsSource::Dir(face_dir.clone())),
    });

    let source = image::open(env("PULID_REFERENCE_IMAGE"))
        .expect("reference image")
        .to_rgb8();
    let face = mlx_gen::media::Image {
        width: source.width(),
        height: source.height(),
        pixels: source.into_raw(),
    };
    let request = GenerationRequest {
        prompt: "a portrait photograph of the same person, looking at the camera".into(),
        width: 1024,
        height: 1024,
        steps: Some(1),
        guidance: Some(4.0),
        seed: Some(42),
        conditioning: vec![Conditioning::Reference {
            image: face.clone(),
            strength: Some(1.0),
        }],
        ..Default::default()
    };
    assert!(request.phases.is_none());

    // Like-for-like Resident request baseline, measured after load with the same Q4 identity stack.
    let mut resident_spec = optimized_spec.clone();
    resident_spec.offload_policy = OffloadPolicy::Resident;
    resident_spec.load_shape = LoadShape::EagerMaterialization;
    let resident =
        mlx_gen_pulid::pulid_flux::load_pulid_flux(&resident_spec).expect("resident load");
    mlx_rs::memory::reset_peak_memory();
    let resident_pixels = pixels(
        resident
            .generate(&request, &mut |_| {})
            .expect("resident render"),
    );
    let resident_peak = mlx_rs::memory::get_peak_memory();
    println!(
        "PULID_RESIDENT_PEAK_GIB {:.3}",
        resident_peak as f64 / 1024_f64.powi(3)
    );
    drop(resident);
    mlx_rs::memory::clear_cache();

    let model =
        mlx_gen_pulid::pulid_flux::load_pulid_flux(&optimized_spec).expect("optimized load");
    let contract = model
        .memory_strategy_contract()
        .expect("PuLID memory contract");
    assert_eq!(
        contract
            .calibration
            .as_ref()
            .expect("exact calibration")
            .fingerprint,
        mlx_gen_pulid::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
    );
    for strategy in MemoryStrategy::ALL {
        assert_eq!(
            contract.capability(strategy).unwrap().support,
            MemoryStrategySupport::Implemented
        );
    }

    let tier = MemoryNumericTier {
        precision: optimized_spec.precision,
        quant: optimized_spec.quantize,
        component_precision_floors: &[],
    };
    let mut full_pixels = Vec::new();
    let mut full_peak = 0_usize;
    let mut full_context = None;

    // Serially exercise every constrained rung; each arm uses the shared selector/request scope.
    for strategy in [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ] {
        let context = standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::ImageToImage,
                reference_count: 1,
                use_pid: false,
                // PuLID is a single-phase request even when the load policy is Sequential.
                has_phases: false,
                overlay: Some("identity".into()),
            },
        )
        .unwrap();
        assert_eq!(
            model.memory_strategy_safety_check(&context),
            mlx_gen::gen_core::MemorySafetyDecision::Accept
        );
        let mut arm_request = request.clone();
        let mut scope = model
            .begin_memory_strategy_request(&context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut arm_request).unwrap();
        mlx_rs::memory::reset_peak_memory();
        let arm_pixels = pixels(
            model
                .generate(&arm_request, &mut |_| {})
                .expect("optimized render"),
        );
        let peak = mlx_rs::memory::get_peak_memory();
        let delta = mean_abs_delta(&resident_pixels, &arm_pixels);
        assert!(
            delta <= 3.0,
            "{strategy:?} drifted from Resident: mean u8 delta {delta}"
        );
        println!(
            "PULID_{strategy:?}_PEAK_GIB {:.3} MEAN_DELTA {delta:.4}",
            peak as f64 / 1024_f64.powi(3)
        );
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        if strategy == MemoryStrategy::BoundedTransformerResidency {
            full_pixels = arm_pixels;
            full_peak = peak;
            full_context = Some(context);
        }
    }
    assert!(
        full_peak < resident_peak,
        "full ladder did not reduce request peak: {full_peak} >= {resident_peak}"
    );

    let context = full_context.unwrap();

    // ArcFace needs a minimally converged image, so validate identity separately at four denoise
    // steps while preserving the calibrated 1024x1024 full-ladder route. The Q4 floor matches the
    // provider's existing real-weight quantization conformance test.
    let mut identity_request = request.clone();
    identity_request.steps = Some(4);
    let mut identity_scope = model
        .begin_memory_strategy_request(&context)
        .unwrap()
        .unwrap();
    identity_scope
        .configure_request(&mut identity_request)
        .unwrap();
    let identity_pixels = pixels(
        model
            .generate(&identity_request, &mut |_| {})
            .expect("streamed identity-quality render"),
    );
    identity_scope.finish(MemoryRunOutcome::Complete).unwrap();

    // ArcFace identity quality on the optimized output, using the exact admitted face stack.
    let analyzer = face_stack(&face_dir);
    let reference = analyzer
        .analyze(&face.pixels, face.height as usize, face.width as usize)
        .unwrap();
    let generated = analyzer.analyze(&identity_pixels, 1024, 1024).unwrap();
    let generated_face = generated
        .first()
        .expect("no face detected in the four-step streamed identity render");
    let identity_cosine = cosine(&reference[0].embedding, &generated_face.embedding);
    assert!(
        identity_cosine > 0.30,
        "streamed identity quality collapsed: cosine {identity_cosine}"
    );
    println!("PULID_STREAMED_IDENTITY_COSINE {identity_cosine:.4}");

    // Fault cleanup + fresh-request recovery at the first materialized block.
    let mut fault = request.clone();
    let mut fault_scope = model
        .begin_memory_strategy_request(&context)
        .unwrap()
        .unwrap();
    fault_scope.configure_request(&mut fault).unwrap();
    let memory = fault.memory.as_mut().unwrap();
    memory.calibration_fault_harness_authorized = true;
    memory.calibration_error_phase = Some(mlx_gen::gen_core::MemoryPhase::Denoise);
    assert!(model.generate(&fault, &mut |_| {}).is_err());
    fault_scope
        .finish(MemoryRunOutcome::Error {
            message: "expected rung-4 fault".into(),
        })
        .unwrap();

    // True-CFG uses two streamed identity injectors; it must recover after the fault and complete.
    let mut cfg = request.clone();
    cfg.true_cfg = Some(2.0);
    cfg.negative_prompt = Some("blurry, deformed".into());
    cfg.timestep_to_start_cfg = Some(0);
    let mut cfg_scope = model
        .begin_memory_strategy_request(&context)
        .unwrap()
        .unwrap();
    cfg_scope.configure_request(&mut cfg).unwrap();
    let cfg_pixels = pixels(
        model
            .generate(&cfg, &mut |_| {})
            .expect("streamed true-CFG render"),
    );
    let cfg_changed = cfg_pixels
        .iter()
        .zip(&full_pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        cfg_changed > full_pixels.len() / 100,
        "true-CFG changed only {cfg_changed} bytes"
    );
    println!("PULID_STREAMED_TRUE_CFG_CHANGED_BYTES {cfg_changed}");
    cfg_scope.finish(MemoryRunOutcome::Complete).unwrap();

    // Mutation control: id_weight=0 must materially change the streamed output.
    let mut zero = request.clone();
    if let Conditioning::Reference { strength, .. } = &mut zero.conditioning[0] {
        *strength = Some(0.0);
    }
    let mut zero_scope = model
        .begin_memory_strategy_request(&context)
        .unwrap()
        .unwrap();
    zero_scope.configure_request(&mut zero).unwrap();
    let zero_pixels = pixels(
        model
            .generate(&zero, &mut |_| {})
            .expect("zero-identity render"),
    );
    let changed = full_pixels
        .iter()
        .zip(&zero_pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        changed > full_pixels.len() / 100,
        "identity injection changed only {changed} bytes"
    );
    println!("PULID_STREAMED_IDENTITY_CHANGED_BYTES {changed}");
    zero_scope.finish(MemoryRunOutcome::Complete).unwrap();
}

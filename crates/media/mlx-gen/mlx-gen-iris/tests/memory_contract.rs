//! The memory-strategy contract of the three Iris inference tasks on their **loaded** models
//! (sc-25690), on the committed miniature resources: each loaded model publishes exactly the
//! contract its registration builds for the same spec, and the one optimized rung the family
//! implements — generation's staged residency — is executed end to end through the shared
//! request-scope seam, not merely declared.

use mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use mlx_gen::gen_core::{
    self, GenerationMemory, MemoryBehaviorRoute, MemoryMode, MemoryNumericTier, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport,
};
use mlx_gen::{
    GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Precision, WeightsSource,
};
use mlx_gen_iris::memory_strategy::{
    DEPTH_MEMORY_REGISTRATION, MEMORY_REGISTRATION, RESTORE_MEMORY_REGISTRATION,
};

use crate::common::{cpu_process, fixtures, tiny_backbone, tiny_text_encoder};

fn generation_spec(policy: OffloadPolicy) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone())).with_offload_policy(policy);
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    spec
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

const STEPS: [&str; 4] = [
    "Step { current: 1, total: 3 }",
    "Step { current: 2, total: 3 }",
    "Step { current: 3, total: 3 }",
    "Decoding",
];
const STAGED_LOADS: [&str; 2] = ["Loading(TextEncoder)", "Loading(Renderer)"];

fn render(generator: &dyn mlx_gen::Generator, req: &GenerationRequest) -> (Vec<String>, Vec<u8>) {
    let mut events = Vec::new();
    let out = generator
        .generate(req, &mut |p| events.push(format!("{p:?}")))
        .expect("the miniature render succeeds");
    match out {
        GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            (events, images.remove(0).pixels)
        }
        other => panic!("expected Images, got {other:?}"),
    }
}

/// A warm (`Resident`) generator runs ONE request staged when the shared seam selects
/// `StagedResidency`, renders the same pixels, and rebuilds its warm pair on the next unstaged
/// request.
///
/// The staged request is observable without measuring memory: only the staged order loads the two
/// components inside the request, text encoder first and the backbone after the encode.
///
/// *Mutation that reds this:* `Iris3b::generate_impl` calling `Residency::run` (the load policy's
/// default) instead of `run_request_scoped(stage_residency_for(..))` — the staged request then
/// loads nothing and reports only its steps.
#[test]
fn a_staged_selection_stages_a_warm_generator_and_renders_the_same_image() {
    cpu_process(
        "memory_contract::a_staged_selection_stages_a_warm_generator_and_renders_the_same_image",
        a_staged_selection_stages_a_warm_generator_and_renders_the_same_image_body,
    );
}

fn a_staged_selection_stages_a_warm_generator_and_renders_the_same_image_body() {
    let spec = generation_spec(OffloadPolicy::Resident);
    let generator = mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .expect("the tiny snapshot loads through the catalog path");
    let contract = generator
        .memory_strategy_contract()
        .expect("the loaded generator publishes its memory contract")
        .clone();
    // The loaded contract is the registered one for this spec, priced from the same resources.
    assert_eq!(contract, (MEMORY_REGISTRATION.contract)(&spec).unwrap());
    assert!(contract.asset_facts.conditioning_bytes > 0);
    assert!(contract.asset_facts.transformer_bytes > 0);

    let (warm_events, warm) = render(generator.as_ref(), &request());
    assert_eq!(warm_events, STEPS, "a warm request loads nothing");

    let mut context = gen_core::standard_memory_behavior_context(
        &contract,
        MemoryStrategy::StagedResidency,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )
    .unwrap();
    context.geometry.width = 32;
    context.geometry.height = 16;
    assert_eq!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    );
    let mut staged = request();
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .unwrap()
        .expect("an implemented optimized rung opens a request scope");
    scope.configure_request(&mut staged).unwrap();
    assert!(staged.memory.is_some_and(|memory| memory.stage_residency));
    let (staged_events, staged_pixels) = render(generator.as_ref(), &staged);
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    drop(scope);
    assert_eq!(
        staged_events,
        [&STAGED_LOADS[..1], &STAGED_LOADS[1..], &STEPS[..]].concat(),
        "a staged request loads the encoder, encodes, then loads the backbone"
    );
    assert_eq!(staged_pixels, warm, "staging must not change the render");

    // The staged request evicted the warm pair; the next unstaged request rebuilds it.
    let (rebuild_events, rebuilt) = render(generator.as_ref(), &request());
    assert_eq!(&rebuild_events[..2], &STAGED_LOADS);
    assert_eq!(rebuilt, warm);
    let (again_events, again) = render(generator.as_ref(), &request());
    assert_eq!(again_events, STEPS, "the warm pair is back");
    assert_eq!(again, warm);

    // A control the route has no mechanism for is refused at validation, not accepted and ignored;
    // the staged request the scope wrote validates.
    generator.validate(&staged).unwrap();
    for memory in [
        GenerationMemory {
            tile_vae_decode: true,
            ..Default::default()
        },
        GenerationMemory {
            chunk_attention: true,
            ..Default::default()
        },
        GenerationMemory {
            stream_transformer_blocks: true,
            ..Default::default()
        },
    ] {
        let mut unhonoured = request();
        unhonoured.memory = Some(memory);
        match generator.validate(&unhonoured) {
            Err(gen_core::Error::Unsupported(message)) => {
                assert!(message.contains("GenerationMemory::"), "{message}")
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
    }

    // A sequential load stages every request, whatever the request says.
    let sequential = mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &generation_spec(OffloadPolicy::Sequential))
        .unwrap();
    let (sequential_events, sequential_pixels) = render(sequential.as_ref(), &request());
    assert_eq!(&sequential_events[..2], &STAGED_LOADS);
    assert_eq!(sequential_pixels, warm);
}

/// The encoder-free tasks publish their loaded contract too: the resident baseline, priced from
/// the export on disk, identical to what the registration builds for the same spec, with the
/// registered safety check bound to the precision the model computes in. A deferred load shape —
/// which no Iris loader has — is refused at load on all three routes.
///
/// *Mutations that red this:* a loader pricing at a precision other than its spec's (the restorer
/// pricing bf16 for an FP32 load); a loaded safety check that ignores the model's precision; a
/// loader that accepts a deferred spec.
#[test]
fn the_encoder_free_tasks_publish_their_registered_contract_when_loaded() {
    cpu_process(
        "memory_contract::the_encoder_free_tasks_publish_their_registered_contract_when_loaded",
        the_encoder_free_tasks_publish_their_registered_contract_when_loaded_body,
    );
}

fn resident_context(
    contract: &gen_core::MemoryProviderContract,
    mode: &str,
    precision: Precision,
) -> MemoryRunContext {
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::Resident,
        MemoryNumericTier {
            precision,
            quant: None,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other(mode.to_owned()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )
    .unwrap();
    context.geometry.width = 32;
    context.geometry.height = 16;
    context
}

fn the_encoder_free_tasks_publish_their_registered_contract_when_loaded_body() {
    let depth_dir = fixtures().join("tiny-depth");
    let restore_dir = fixtures().join("tiny-snapshot/upscaler");
    for precision in [Precision::Bf16, Precision::Fp32] {
        let other = if precision == Precision::Fp32 {
            Precision::Bf16
        } else {
            Precision::Fp32
        };
        let mut depth_spec = LoadSpec::new(WeightsSource::Dir(depth_dir.clone()));
        depth_spec.precision = precision;
        let depth = mlx_gen_iris::depth::load_depth(&depth_spec).unwrap();
        assert_eq!(
            *depth.memory_strategy_contract(),
            (DEPTH_MEMORY_REGISTRATION.contract)(&depth_spec).unwrap()
        );
        assert_eq!(
            depth.memory_strategy_contract().provider_id,
            "iris_3b_depth"
        );

        let mut restore_spec = LoadSpec::new(WeightsSource::Dir(restore_dir.clone()));
        restore_spec.precision = precision;
        let restorer = mlx_gen_iris::IrisRestorer::load(&restore_spec).unwrap();
        assert_eq!(
            *restorer.memory_strategy_contract(),
            (RESTORE_MEMORY_REGISTRATION.contract)(&restore_spec).unwrap()
        );
        assert_eq!(
            restorer.memory_strategy_contract().provider_id,
            "iris_3b_restore"
        );

        for contract in [
            depth.memory_strategy_contract(),
            restorer.memory_strategy_contract(),
        ] {
            gen_core_testkit::memory_strategy_conformance(contract);
            assert!(contract.asset_facts.transformer_bytes > 0);
            assert!(contract.asset_facts.conditioning_bytes > 0);
            assert_eq!(
                contract.load_shape,
                gen_core::LoadShape::EagerMaterialization
            );
            assert_eq!(
                contract.architecture_facts.activation_dtype_width,
                Some(if precision == Precision::Fp32 { 4 } else { 2 })
            );
            // Resident only: nothing optimized is implemented on an encoder-free route.
            assert!(contract.strategies.iter().all(|capability| {
                (capability.strategy == MemoryStrategy::Resident)
                    == (capability.support == MemoryStrategySupport::Implemented)
            }));
        }

        // The loaded safety checks: the route's own mode at the model's own precision.
        let accept = MemorySafetyDecision::Accept;
        let depth_contract = depth.memory_strategy_contract();
        assert_eq!(
            depth.memory_strategy_safety_check(&resident_context(
                depth_contract,
                "depth",
                precision
            )),
            accept
        );
        let restore_contract = restorer.memory_strategy_contract();
        assert_eq!(
            restorer.memory_strategy_safety_check(&resident_context(
                restore_contract,
                "restoration",
                precision
            )),
            accept
        );
        for rejected in [
            depth.memory_strategy_safety_check(&resident_context(depth_contract, "depth", other)),
            depth.memory_strategy_safety_check(&resident_context(
                depth_contract,
                "restoration",
                precision,
            )),
            restorer.memory_strategy_safety_check(&resident_context(
                restore_contract,
                "restoration",
                other,
            )),
            restorer.memory_strategy_safety_check(&resident_context(
                restore_contract,
                "depth",
                precision,
            )),
        ] {
            assert!(
                matches!(rejected, MemorySafetyDecision::Reject { .. }),
                "{rejected:?}"
            );
        }
    }

    // No loader has a deferred-materialization load shape: each refuses one by name.
    let deferred = gen_core::LoadShape::DeferredMaterialization;
    let refusal = |error: String| assert!(error.contains("load_shape"), "{error}");
    let spec = LoadSpec::new(WeightsSource::Dir(depth_dir)).with_load_shape(deferred);
    match mlx_gen_iris::depth::load_depth(&spec) {
        Err(mlx_gen::Error::Unsupported(message)) => refusal(message),
        other => panic!("depth: expected a typed refusal, got {:?}", other.err()),
    }
    let spec = LoadSpec::new(WeightsSource::Dir(restore_dir)).with_load_shape(deferred);
    match mlx_gen_iris::IrisRestorer::load(&spec) {
        Err(mlx_gen::Error::Unsupported(message)) => refusal(message),
        other => panic!(
            "restoration: expected a typed refusal, got {:?}",
            other.err()
        ),
    }
    let spec = generation_spec(OffloadPolicy::Resident).with_load_shape(deferred);
    match mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
    {
        Err(gen_core::Error::Unsupported(message)) => refusal(message),
        other => panic!(
            "generation: expected a typed refusal, got {:?}",
            other.err()
        ),
    }
}

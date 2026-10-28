//! sc-25690: the two optimized rungs the generation contract declares, witnessed on the production
//! path — the catalog's `load`, the loaded generator's own memory-strategy seam, then `generate` on
//! the committed miniature snapshot. Each rung is selected the way the SceneWorks worker selects
//! it: the admitted context opens the request scope, the scope writes the request's memory block,
//! and the render must then *do* what was selected.

use candle_gen::attention::chunk_probe;
use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use candle_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Generator, LoadPhase, LoadSpec,
    MemoryBehaviorRoute, MemoryMode, MemoryNumericTier, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, OffloadPolicy, Precision, Progress, WeightsSource,
};

use crate::common::{tiny_backbone, tiny_text_encoder};

const ID: &str = "iris_3b";

fn spec(precision: Precision) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()))
        .with_offload_policy(OffloadPolicy::Resident);
    spec.precision = precision;
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    spec
}

fn load(precision: Precision) -> Box<dyn Generator> {
    candle_gen_iris::provider_registry()
        .unwrap()
        .load(ID, &spec(precision))
        .expect("the tiny snapshot loads through the catalog path")
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

/// The admitted context of `strategy` for [`request`]'s geometry, from the loaded contract.
fn context(
    generator: &dyn Generator,
    precision: Precision,
    strategy: MemoryStrategy,
) -> MemoryRunContext {
    let contract = generator
        .memory_strategy_contract()
        .expect("the loaded generator publishes its memory contract");
    let spec = spec(precision);
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
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
    context.geometry.width = request().width;
    context.geometry.height = request().height;
    assert_eq!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    );
    context
}

/// Render `request`, returning the image bytes and the component loads it reported.
fn render(generator: &dyn Generator, request: &GenerationRequest) -> (Vec<u8>, Vec<LoadPhase>) {
    let mut loads = Vec::new();
    let output = generator
        .generate(request, &mut |progress| {
            if let Progress::Loading(phase) = progress {
                loads.push(phase);
            }
        })
        .unwrap();
    let GenerationOutput::Images(mut images) = output else {
        panic!("expected images");
    };
    (images.remove(0).pixels, loads)
}

/// Rung 1 on a **warm-loaded** generator: the scope's request stages — the text encoder is loaded,
/// encoded and dropped, then the backbone is loaded — while a plain request reuses the warm pair
/// and loads nothing. The staged render is the same image.
///
/// *Mutation that reds this:* `stage_residency_for` ignoring the request
/// (`spec.offload_policy == Sequential` alone): the staged request then reports no load.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_admitted_staged_request_stages_a_warm_generator() {
    let generator = load(Precision::Bf16);
    let (resident, loads) = render(generator.as_ref(), &request());
    assert!(
        loads.is_empty(),
        "a warm resident request loads nothing: {loads:?}"
    );

    let context = context(
        generator.as_ref(),
        Precision::Bf16,
        MemoryStrategy::StagedResidency,
    );
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .unwrap()
        .expect("an implemented optimized rung opens a request scope");
    let mut staged_request = request();
    scope.configure_request(&mut staged_request).unwrap();
    assert_eq!(
        staged_request.memory,
        Some(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        })
    );
    let (staged, loads) = render(generator.as_ref(), &staged_request);
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    assert_eq!(
        loads,
        [LoadPhase::TextEncoder, LoadPhase::Renderer],
        "a staged request loads each component in its own phase"
    );
    assert_eq!(staged, resident, "staging changes residency, not the image");

    // The staged request evicted the warm pair, so the next resident request reloads it once...
    let (_, loads) = render(generator.as_ref(), &request());
    assert_eq!(loads, [LoadPhase::TextEncoder, LoadPhase::Renderer]);
    // ...and is warm again after that.
    let (again, loads) = render(generator.as_ref(), &request());
    assert!(loads.is_empty(), "{loads:?}");
    assert_eq!(again, resident);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED memory_strategy::an_admitted_staged_request_stages_a_warm_generator");
}

/// Rung 3: the scope's request carries the admitted attention budget, and the render's attention
/// kernels chunk under it. The published budgets are far above this miniature's scores tensors, so
/// the engagement itself is shown at a budget small enough to bind here — the same request field,
/// through the same `generate`.
///
/// Rendered on the FP32 path, where a chunked GEMM differs from the single pass by f32 rounding
/// only; under the bf16 policy every projection re-rounds, which turns the same ulp into a bf16
/// step and makes a pixel bound a statement about bf16, not about the chunking.
///
/// *Mutation that reds this:* `attention_budget_for` returning `None` (the render then runs every
/// attention as a single launch).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_admitted_attention_budget_bounds_the_render() {
    let generator = load(Precision::Fp32);
    chunk_probe::reset();
    let (unbounded, _) = render(generator.as_ref(), &request());
    assert_eq!(
        chunk_probe::max_chunk_count(),
        1,
        "with no selection every attention is one launch"
    );

    // The scope writes the published budget the selection named.
    let context = context(
        generator.as_ref(),
        Precision::Fp32,
        MemoryStrategy::BoundedAttention,
    );
    let admitted = context
        .selection
        .parameters
        .attention_chunk_size
        .expect("a bounded-attention selection names its budget");
    assert_eq!(
        admitted,
        candle_gen_iris::memory_strategy::ATTENTION_CHUNK_SIZE
    );
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .unwrap()
        .unwrap();
    let mut bounded_request = request();
    scope.configure_request(&mut bounded_request).unwrap();
    scope.configure_attention(admitted).unwrap();
    assert_eq!(
        bounded_request.memory,
        Some(GenerationMemory {
            chunk_attention: true,
            attention_chunk_size: Some(admitted),
            ..Default::default()
        })
    );
    chunk_probe::reset();
    let (at_published, _) = render(generator.as_ref(), &bounded_request);
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    assert_eq!(
        chunk_probe::max_chunk_count(),
        1,
        "the published budget exceeds every scores tensor of the miniature"
    );
    assert_eq!(at_published, unbounded);

    // A budget that binds on the miniature: 64 score elements is less than one query row of the
    // CFG batch, so the kernels split rows and chunk queries.
    let mut binding_request = request();
    binding_request.memory = Some(GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(64),
        ..Default::default()
    });
    chunk_probe::reset();
    let (bounded, _) = render(generator.as_ref(), &binding_request);
    assert!(
        chunk_probe::max_chunk_count() > 1,
        "the request's budget must reach the attention kernels"
    );
    // Chunking changes the GEMM shapes, not the math: the images agree to rounding.
    let worst = bounded
        .iter()
        .zip(&unbounded)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(worst <= 1, "bounded attention moved a pixel by {worst}");

    // The scope does not leak past its request.
    chunk_probe::reset();
    let (after, _) = render(generator.as_ref(), &request());
    assert_eq!(chunk_probe::max_chunk_count(), 1);
    assert_eq!(after, unbounded);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED memory_strategy::an_admitted_attention_budget_bounds_the_render");
}

/// A memory control this route has no mechanism for is refused, never accepted and ignored.
///
/// *Mutation that reds this:* dropping `refuse_unhonored_memory` from `validate_impl`.
#[test]
fn an_unhonoured_memory_control_is_refused() {
    let generator = load(Precision::Bf16);
    for memory in [
        GenerationMemory {
            tile_vae_decode: true,
            ..Default::default()
        },
        GenerationMemory {
            stream_transformer_blocks: true,
            ..Default::default()
        },
        GenerationMemory {
            chunk_attention: true,
            ..Default::default()
        },
    ] {
        let mut request = request();
        request.memory = Some(memory);
        let error = generator
            .validate(&request)
            .expect_err("an unhonoured memory control must be refused");
        assert!(error.to_string().contains("GenerationMemory"), "{error}");
    }
    let mut honoured = request();
    honoured.memory = Some(GenerationMemory {
        stage_residency: true,
        chunk_attention: true,
        attention_chunk_size: Some(64),
        ..Default::default()
    });
    generator.validate(&honoured).unwrap();
}

/// Declared bound on the bf16 render's largest single-channel pixel change under a binding
/// attention budget, in RGB8 levels.
const BF16_BOUNDED_MAX_ABS_LEVELS: u8 = 8;
/// Declared bound on the bf16 render's mean absolute pixel change under a binding attention
/// budget, in RGB8 levels. Not scale-invariant: it bounds the size of the change, not its shape.
const BF16_BOUNDED_MEAN_ABS_LEVELS: f64 = 1.0;

/// Rung 3 on the tier the contract actually witnesses (bf16). The chunked kernels compute the same
/// attention to f32 rounding (`nn` unit tests, and the FP32 render above), but under the bf16
/// policy every projection re-rounds its output to bf16, so an f32 ulp of GEMM-order difference can
/// become one bf16 step and travel through the remaining blocks and solver steps. The render is
/// therefore not bit-identical; it is held to a declared bound instead, on two metrics: the largest
/// single-channel change and the mean absolute change, both in RGB8 levels, over three seeds and
/// three binding budgets on the miniature snapshot.
///
/// *Mutation that reds this:* dropping the per-row mask narrowing in `nn::bounded_attention`, or
/// chunking the key axis instead of the query axis (either changes the attention, not its
/// rounding).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_bf16_render_under_a_binding_attention_budget_stays_inside_the_declared_bound() {
    let generator = load(Precision::Bf16);
    let mut worst_max = 0u8;
    let mut worst_mean = 0f64;
    for seed in [42u64, 7, 1234] {
        let mut plain = request();
        plain.seed = Some(seed);
        chunk_probe::reset();
        let (unbounded, _) = render(generator.as_ref(), &plain);
        assert_eq!(chunk_probe::max_chunk_count(), 1);
        for budget in [64u32, 512, 4096] {
            let mut bounded_request = plain.clone();
            bounded_request.memory = Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(budget),
                ..Default::default()
            });
            chunk_probe::reset();
            let (bounded, _) = render(generator.as_ref(), &bounded_request);
            assert!(
                chunk_probe::max_chunk_count() > 1,
                "budget {budget} must bind on the miniature"
            );
            assert_eq!(bounded.len(), unbounded.len());
            let max = bounded
                .iter()
                .zip(&unbounded)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let mean = bounded
                .iter()
                .zip(&unbounded)
                .map(|(a, b)| f64::from(a.abs_diff(*b)))
                .sum::<f64>()
                / bounded.len() as f64;
            worst_max = worst_max.max(max);
            worst_mean = worst_mean.max(mean);
        }
    }
    assert!(
        worst_max <= BF16_BOUNDED_MAX_ABS_LEVELS,
        "largest pixel change {worst_max} exceeds the declared {BF16_BOUNDED_MAX_ABS_LEVELS} levels"
    );
    assert!(
        worst_mean <= BF16_BOUNDED_MEAN_ABS_LEVELS,
        "mean pixel change {worst_mean:.4} exceeds the declared {BF16_BOUNDED_MEAN_ABS_LEVELS} levels"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED memory_strategy::a_bf16_render_under_a_binding_attention_budget_stays_inside_the_declared_bound");
}

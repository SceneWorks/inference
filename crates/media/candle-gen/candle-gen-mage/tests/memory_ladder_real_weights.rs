//! Authoritative CUDA implementation evidence for the shared Mage-Flow memory ladder (SC-15813).
//!
//! One invocation exercises one rung in a fresh process so CUDA allocator caching cannot blur the
//! measured high-water marks. The workflow runs the resident baseline first, then every advertised
//! optimized rung. All optimized outputs must remain byte-identical to that baseline.

#![cfg(feature = "cuda")]

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadShape, LoadSpec, MemoryBehaviorRoute, MemoryMode,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, Quant, WeightsSource,
};
use candle_gen::testkit::VramProbe;
use candle_gen_mage::{memory_strategy, REGISTRATION};
use sha2::{Digest, Sha256};

fn rung() -> MemoryStrategy {
    match std::env::var("MAGE_MEMORY_RUNG")
        .expect("set MAGE_MEMORY_RUNG to resident, staged, attention, or blocks")
        .as_str()
    {
        "resident" => MemoryStrategy::Resident,
        "staged" => MemoryStrategy::StagedResidency,
        "attention" => MemoryStrategy::BoundedAttention,
        "blocks" => MemoryStrategy::BoundedTransformerResidency,
        value => panic!("unsupported MAGE_MEMORY_RUNG={value}"),
    }
}

#[test]
#[ignore = "requires CANDLE_MAGE_SNAPSHOT and a physical idle CUDA runner"]
fn representative_route_exercises_advertised_rung() {
    let root = std::env::var_os("CANDLE_MAGE_SNAPSHOT")
        .map(Into::into)
        .expect("set CANDLE_MAGE_SNAPSHOT to a complete Mage-Flow snapshot");
    let out = std::env::var("MAGE_MEMORY_OUT").expect("set MAGE_MEMORY_OUT");
    let strategy = rung();
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_quant(Quant::Q4)
        .with_load_shape(LoadShape::DeferredMaterialization);

    assert!(
        candle_gen::testkit::reset_cuda_mempool_high_water(0),
        "reset CUDA live-allocation high-water"
    );
    let mut probe = VramProbe::start_rendered().assert_idle(1.0);
    let load_phase = probe.phase();
    let generator = (REGISTRATION.load)(&spec).expect("registered Mage load");
    probe.end_load(load_phase);

    let contract = generator
        .memory_strategy_contract()
        .expect("Mage CUDA memory contract");
    let tier = memory_strategy::resolved_numeric_tier(&spec).expect("numeric q4 tier");
    let context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )
    .expect("Mage memory context");
    assert_eq!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept,
        "shared admission must accept the representative route"
    );
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .expect("begin memory request")
        .expect("Mage memory request scope");
    let mut request = GenerationRequest {
        prompt: "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug".into(),
        width: 1024,
        height: 1024,
        steps: Some(20),
        guidance: Some(5.0),
        seed: Some(42),
        ..Default::default()
    };
    scope
        .configure_request(&mut request)
        .expect("configure admitted request");

    let generation_phase = probe.phase();
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("Mage memory-rung generation");
    probe.end_gen(generation_phase);
    scope
        .finish(MemoryRunOutcome::Complete)
        .expect("finish memory request");
    let report = probe.report().assert_trustworthy(1.0);
    let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
        .expect("read CUDA live-allocation high-water");
    assert!(live_peak_bytes > 0, "CUDA live peak must be positive");

    let image = match output {
        GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
        GenerationOutput::Images(images) => panic!("expected one image, got {}", images.len()),
        _ => panic!("expected image output"),
    };
    assert_eq!((image.width, image.height), (1024, 1024));
    assert_eq!(image.pixels.len(), 1024 * 1024 * 3);
    std::fs::write(&out, &image.pixels).expect("write raw RGB output");

    if strategy != MemoryStrategy::Resident {
        let reference_path = std::env::var("MAGE_MEMORY_REFERENCE")
            .expect("set MAGE_MEMORY_REFERENCE for optimized rungs");
        let reference = std::fs::read(&reference_path)
            .unwrap_or_else(|error| panic!("read MAGE_MEMORY_REFERENCE={reference_path}: {error}"));
        assert_eq!(
            image.pixels, reference,
            "{strategy:?} changed the deterministic resident output"
        );
    }

    let output_sha256 = format!("{:x}", Sha256::digest(&image.pixels));
    eprintln!(
        "MAGE_MEMORY_EVIDENCE strategy={strategy:?} composition={:?} peak_bytes={live_peak_bytes} output_sha256={output_sha256} gpu={} {report}",
        contract.engaged_composition(strategy),
        candle_gen::testkit::probe_gpu(),
    );
}

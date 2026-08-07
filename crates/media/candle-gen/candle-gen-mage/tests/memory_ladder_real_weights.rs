//! Authoritative CUDA implementation evidence for the shared Mage-Flow memory ladder (SC-15813).
//!
//! One invocation exercises one rung in a fresh process so CUDA allocator caching cannot blur the
//! measured high-water marks. The workflow runs the resident baseline first, then every advertised
//! optimized rung. The fixture spans three fixed seeds so a parity verdict is not tuned to one lucky
//! render; residency-only rungs remain exact, while query-chunked attention reports its measured RGB
//! drift before applying the provider-owned contract.

#![cfg(all(feature = "cuda", feature = "testkit"))]

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadShape, LoadSpec, MemoryBehaviorRoute, MemoryMode,
    MemoryParityContract, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, Quant,
    WeightsSource,
};
use candle_gen::testkit::VramProbe;
use candle_gen_mage::{memory_strategy, REGISTRATION};
use sha2::{Digest, Sha256};

// The dedicated Windows CUDA runner's desktop/driver baseline is about 1.4 GiB. Keep a finite
// cleanliness gate (and subtract the sampled baseline from the report), but do not reject the
// otherwise-idle runner before Mage loads. This matches the 2 GiB ceiling used by other physical
// GPU probes in this workspace.
const MAX_IDLE_BASELINE_GB: f64 = 2.0;

struct EvidenceCase {
    cohort: &'static str,
    name: &'static str,
    prompt: &'static str,
    seed: u64,
}

const EVIDENCE_CASES: [EvidenceCase; 6] = [
    EvidenceCase {
        cohort: "calibration",
        name: "animal-still-life",
        prompt: "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug",
        seed: 42,
    },
    EvidenceCase {
        cohort: "calibration",
        name: "wide-landscape",
        prompt: "a wide alpine lake at sunrise, snow peaks reflected in still water, documentary photograph",
        seed: 314_159,
    },
    EvidenceCase {
        cohort: "calibration",
        name: "graphic-poster",
        prompt: "a bold geometric travel poster of a red lighthouse, flat colors, crisp screen print texture",
        seed: 271_828,
    },
    EvidenceCase {
        cohort: "holdout",
        name: "architectural-interior",
        prompt: "a sunlit brutalist library interior with long concrete shadows, architectural photography",
        seed: 1_618_033,
    },
    EvidenceCase {
        cohort: "holdout",
        name: "macro-nature",
        prompt: "macro photograph of a dew-covered blue butterfly on a fern, shallow depth of field",
        seed: 57_721,
    },
    EvidenceCase {
        cohort: "holdout",
        name: "night-street",
        prompt: "a rainy city street at night with warm shop lights reflected on pavement, cinematic photograph",
        seed: 1_414_213,
    },
];

// Query-row chunking preserves each row's complete K/V reduction but changes GEMM M, so Candle's
// shared SDPA contract explicitly promises numerical rather than bit parity. An 8.0 RGB8 RMSE ceiling
// is an independently chosen >=30 dB PSNR quality floor (8.0 is slightly stricter than the 8.063
// RMSE equivalent). Every calibration and holdout image must pass separately; no cohort averaging can
// dilute a bad render. Staged residency and block materialization itself remain exact.
const ATTENTION_RMSE_MAX: f64 = 8.0;

fn parity_contract(strategy: MemoryStrategy) -> MemoryParityContract {
    if strategy >= MemoryStrategy::BoundedAttention {
        MemoryParityContract::Tolerance {
            metric: "rgb8_rmse_per_image".to_owned(),
            maximum_error: ATTENTION_RMSE_MAX,
        }
    } else {
        MemoryParityContract::Exact
    }
}

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

fn parity_metrics(reference: &[u8], candidate: &[u8]) -> (f64, u8, f64, f64, f64) {
    assert_eq!(reference.len(), candidate.len(), "RGB parity shape");
    assert!(!reference.is_empty(), "RGB parity requires pixels");
    let mut changed = 0u64;
    let mut maximum = 0u8;
    let mut absolute_sum = 0u64;
    let mut square_sum = 0u64;
    for (&resident, &optimized) in reference.iter().zip(candidate) {
        let error = resident.abs_diff(optimized);
        changed += u64::from(error != 0);
        maximum = maximum.max(error);
        absolute_sum += u64::from(error);
        square_sum += u64::from(error) * u64::from(error);
    }
    let count = reference.len() as f64;
    let mean = absolute_sum as f64 / count;
    let rmse = (square_sum as f64 / count).sqrt();
    let psnr = if rmse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (255.0 / rmse).log10()
    };
    (changed as f64 / count, maximum, mean, rmse, psnr)
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
    let mut probe = VramProbe::start_rendered().assert_idle(MAX_IDLE_BASELINE_GB);
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
    let mut pixels = Vec::with_capacity(EVIDENCE_CASES.len() * 1024 * 1024 * 3);
    for case in &EVIDENCE_CASES {
        assert_eq!(
            generator.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept,
            "shared admission must accept calibration case {}",
            case.name,
        );
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("begin memory request")
            .expect("Mage memory request scope");
        let mut request = GenerationRequest {
            prompt: case.prompt.into(),
            width: 1024,
            height: 1024,
            steps: Some(20),
            guidance: Some(5.0),
            seed: Some(case.seed),
            ..Default::default()
        };
        scope
            .configure_request(&mut request)
            .expect("configure admitted request");

        if strategy >= MemoryStrategy::BoundedAttention {
            candle_gen::attention::chunk_probe::reset();
        }
        if strategy == MemoryStrategy::BoundedTransformerResidency {
            candle_gen_mage::transformer::block_window_probe::reset();
        }
        let generation_phase = probe.phase();
        let output = generator
            .generate(&request, &mut |_| {})
            .expect("Mage memory-rung generation");
        probe.end_gen(generation_phase);
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("finish memory request");
        if strategy >= MemoryStrategy::BoundedAttention {
            assert!(
                candle_gen::attention::chunk_probe::max_chunk_count() > 1,
                "{}: bounded attention did not split the score tensor",
                case.name,
            );
        }
        if strategy == MemoryStrategy::BoundedTransformerResidency {
            assert!(
                candle_gen_mage::transformer::block_window_probe::materialized_windows()
                    >= memory_strategy::TRANSFORMER_BLOCKS as usize,
                "{}: transformer block windows were not materialized",
                case.name,
            );
        }
        let image = match output {
            GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
            GenerationOutput::Images(images) => panic!("expected one image, got {}", images.len()),
            _ => panic!("expected image output"),
        };
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.pixels.len(), 1024 * 1024 * 3);
        pixels.extend_from_slice(&image.pixels);
    }
    let report = probe.report().assert_trustworthy(MAX_IDLE_BASELINE_GB);
    let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
        .expect("read CUDA live-allocation high-water");
    assert!(live_peak_bytes > 0, "CUDA live peak must be positive");

    let bytes_per_case = 1024 * 1024 * 3;
    assert_eq!(pixels.len(), EVIDENCE_CASES.len() * bytes_per_case);
    std::fs::write(&out, &pixels).expect("write concatenated raw RGB outputs");

    if strategy != MemoryStrategy::Resident {
        let reference_path = std::env::var("MAGE_MEMORY_REFERENCE")
            .expect("set MAGE_MEMORY_REFERENCE for optimized rungs");
        let reference = std::fs::read(&reference_path)
            .unwrap_or_else(|error| panic!("read MAGE_MEMORY_REFERENCE={reference_path}: {error}"));
        assert_eq!(reference.len(), pixels.len(), "resident cohort shape");
        for (index, case) in EVIDENCE_CASES.iter().enumerate() {
            let range = index * bytes_per_case..(index + 1) * bytes_per_case;
            let resident = &reference[range.clone()];
            let candidate = &pixels[range];
            let (changed_fraction, max_abs, mean_abs, rmse, psnr_db) =
                parity_metrics(resident, candidate);
            eprintln!(
                "MAGE_MEMORY_PARITY cohort={} case={} seed={} strategy={strategy:?} contract={:?} changed_fraction={changed_fraction:.12} max_abs={max_abs} mean_abs={mean_abs:.12} rmse={rmse:.12} psnr_db={psnr_db:.12}",
                case.cohort, case.name, case.seed, parity_contract(strategy),
            );
            match parity_contract(strategy) {
                MemoryParityContract::Exact => assert_eq!(
                    candidate, resident,
                    "{strategy:?} changed resident output for {}",
                    case.name,
                ),
                MemoryParityContract::Tolerance { maximum_error, .. } => assert!(
                    rmse <= maximum_error,
                    "{strategy:?} RGB8 RMSE {rmse:.6} exceeded {maximum_error:.6} for {}",
                    case.name,
                ),
                MemoryParityContract::Golden { .. } => unreachable!("Mage uses no golden contract"),
            }
        }
    }

    if strategy == MemoryStrategy::BoundedTransformerResidency {
        let attention_path = std::env::var("MAGE_MEMORY_ATTENTION_REFERENCE")
            .expect("set MAGE_MEMORY_ATTENTION_REFERENCE for block-streaming isolation");
        let attention = std::fs::read(&attention_path).unwrap_or_else(|error| {
            panic!("read MAGE_MEMORY_ATTENTION_REFERENCE={attention_path}: {error}")
        });
        assert_eq!(attention.len(), pixels.len(), "attention cohort shape");
        for (index, case) in EVIDENCE_CASES.iter().enumerate() {
            let range = index * bytes_per_case..(index + 1) * bytes_per_case;
            let baseline = &attention[range.clone()];
            let candidate = &pixels[range];
            let (_, max_abs, mean_abs, rmse, _) = parity_metrics(baseline, candidate);
            eprintln!(
                "MAGE_BLOCK_ISOLATION cohort={} case={} seed={} max_abs={max_abs} mean_abs={mean_abs:.12} rmse={rmse:.12}",
                case.cohort, case.name, case.seed,
            );
            assert_eq!(
                candidate, baseline,
                "block streaming changed bounded-attention output for {}",
                case.name,
            );
        }
    }

    let output_sha256 = format!("{:x}", Sha256::digest(&pixels));
    eprintln!(
        "MAGE_MEMORY_EVIDENCE strategy={strategy:?} composition={:?} peak_bytes={live_peak_bytes} output_sha256={output_sha256} gpu={} {report}",
        contract.engaged_composition(strategy),
        candle_gen::testkit::probe_gpu(),
    );
}

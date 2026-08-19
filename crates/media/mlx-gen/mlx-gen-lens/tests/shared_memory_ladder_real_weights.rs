//! SC-15512 serial real-Metal proof for Lens / Lens-Turbo's shared image-memory ladder.
//!
//! No artifact is fetched. Run exactly one provider/tier process at a time:
//!
//! Evidence recorded 2026-08-03: the exact cached q4 `lens` artifact passed every arm and terminal
//! recovery at 1024²/one step (30.497 GiB resident versus 4.696 GiB rung 4). The exact cached bf16
//! `lens_turbo` artifact timed out in the resident-cold baseline before any ARM result, so this
//! runner does not establish full-ladder calibration evidence for Lens-Turbo or dense bf16.
//!
//! ```text
//! LENS_LADDER_PROVIDER=lens LENS_LADDER_TIER=q4 LENS_LADDER_ROOT=/.../lens-mlx/.../q4 \
//! cargo test -p mlx-gen-lens --release --test shared_memory_ladder_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use mlx_gen::gen_core::{
    GenerationMemory, MemoryPhase, ProviderRegistryBuilder, TransformerComponent,
};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Generator, LoadShape, LoadSpec, Quant, WeightsSource,
};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("set {key}; see the test documentation"))
}

fn provider() -> String {
    let value = env("LENS_LADDER_PROVIDER");
    assert!(value == "lens" || value == "lens_turbo");
    value
}

fn tier() -> Option<Quant> {
    match env("LENS_LADDER_TIER").as_str() {
        "bf16" => None,
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        other => panic!("LENS_LADDER_TIER must be bf16, q4, or q8; got {other}"),
    }
}

fn edge() -> u32 {
    std::env::var("LENS_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024)
}

fn spec() -> LoadSpec {
    let root = PathBuf::from(env("LENS_LADDER_ROOT"));
    assert!(
        root.is_dir(),
        "Lens root does not exist: {}",
        root.display()
    );
    let mut spec =
        LoadSpec::new(WeightsSource::Dir(root)).with_load_shape(LoadShape::DeferredMaterialization);
    if let Some(quant) = tier() {
        spec = spec.with_quant(quant);
    }
    spec
}

fn load() -> Box<dyn Generator> {
    let registry = mlx_gen_lens::register_providers(ProviderRegistryBuilder::new())
        .build()
        .expect("build Lens-only registry");
    registry
        .load(&provider(), &spec())
        .expect("load exact cached Lens artifact")
}

fn base_request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a weathered red fox standing in a snowy pine clearing at dawn, detailed photograph, soft natural light".into(),
        negative_prompt: Some("blurry, distorted, low contrast".into()),
        width: edge(),
        height: edge(),
        count: 1,
        steps: Some(1),
        guidance: Some(if provider() == "lens_turbo" { 1.0 } else { 5.0 }),
        seed: Some(15512),
        ..Default::default()
    }
}

fn resident_memory() -> GenerationMemory {
    GenerationMemory::default()
}

fn staged_memory() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..Default::default()
    }
}

fn decode_memory() -> GenerationMemory {
    GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(mlx_gen_lens::memory_strategy::DECODE_TILE_EDGE),
        decode_overlap: Some(mlx_gen_lens::memory_strategy::DECODE_OVERLAP),
        ..Default::default()
    }
}

fn attention_memory() -> GenerationMemory {
    GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(mlx_gen_lens::memory_strategy::ATTENTION_CHUNK_SIZE),
        ..decode_memory()
    }
}

fn rung4_memory() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        stream_transformer_blocks: true,
        transformer_window_size: Some(mlx_gen_lens::memory_strategy::TEXT_ENCODER_WINDOW),
        transformer_window_component: Some(TransformerComponent::Both),
        ..attention_memory()
    }
}

#[derive(Debug)]
struct Run {
    image: mlx_gen::Image,
    peak: usize,
    active: usize,
    cache: usize,
    post_clear_active: usize,
    post_clear_cache: usize,
    wall: Duration,
}

fn run(
    generator: &dyn Generator,
    memory: GenerationMemory,
) -> Result<Run, mlx_gen::gen_core::Error> {
    clear_cache();
    reset_peak_memory();
    let mut request = base_request();
    request.memory = Some(memory);
    let started = Instant::now();
    let output = generator.generate(&request, &mut |_| {})?;
    let image = match output {
        GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
        other => panic!("expected one image, got {other:?}"),
    };
    let active = get_active_memory();
    let cache = get_cache_memory();
    clear_cache();
    Ok(Run {
        image,
        peak: get_peak_memory(),
        active,
        cache,
        post_clear_active: get_active_memory(),
        post_clear_cache: get_cache_memory(),
        wall: started.elapsed(),
    })
}

fn drift(a: &mlx_gen::Image, b: &mlx_gen::Image) -> (u8, f64, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0_u8;
    let mut sum = 0_f64;
    let mut dot = 0_f64;
    let mut aa = 0_f64;
    let mut bb = 0_f64;
    for (&x, &y) in a.pixels.iter().zip(&b.pixels) {
        max = max.max(x.abs_diff(y));
        sum += x.abs_diff(y) as f64;
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    (
        max,
        sum / a.pixels.len() as f64,
        dot / (aa.sqrt() * bb.sqrt()).max(1e-12),
    )
}

fn assert_typed_cancel(error: mlx_gen::gen_core::Error) {
    assert!(
        matches!(error, mlx_gen::gen_core::Error::Canceled),
        "{error:?}"
    );
}

#[test]
#[ignore = "needs exact cached Lens weights and exclusive Apple/Metal access"]
fn shared_ladder_terminal_result() {
    clear_cache();
    let process_baseline_active = get_active_memory();
    let process_baseline_cache = get_cache_memory();
    let generator = load();
    // SC-18605: the printed identity has to come from the contract this run actually loaded. The
    // measured constant used to be printed unconditionally, so a `lens_turbo` or non-Q4 run — routes
    // this runner accepts by env, and which SC-15800 explicitly did *not* calibrate — would stamp the
    // measured `lens` Q4 fingerprint onto its own RESULT line. Those routes now declare the ladder,
    // so the mislabeling went from latent to reachable.
    let fingerprint = generator
        .memory_strategy_contract()
        .and_then(|contract| contract.calibration.as_ref())
        .map(|identity| identity.fingerprint.clone())
        .unwrap_or_else(|| format!("uncalibrated-estimate:{}:{:?}", provider(), tier()));
    assert_eq!(
        fingerprint == mlx_gen_lens::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        provider() == "lens" && tier() == Some(Quant::Q4),
        "only the exact measured lens Q4 route may report the shared-ladder calibration identity"
    );
    let arms = [
        ("resident", resident_memory()),
        ("staged", staged_memory()),
        ("decode", decode_memory()),
        ("attention", attention_memory()),
        ("rung4", rung4_memory()),
    ];
    let mut runs = Vec::new();
    for (name, memory) in arms {
        let cold = run(generator.as_ref(), memory).unwrap_or_else(|e| panic!("{name} cold: {e}"));
        let warm = run(generator.as_ref(), memory).unwrap_or_else(|e| panic!("{name} warm: {e}"));
        let (max, mean, correlation) = drift(&cold.image, &warm.image);
        assert_eq!(max, 0, "{name} cold/warm changed pixels");
        println!(
            "ARM name={name} cold_peak_bytes={} warm_peak_bytes={} active_bytes={} cache_bytes={} post_clear_active_bytes={} post_clear_cache_bytes={} wall_ms={} max_delta={max} mean_delta={mean:.6} correlation={correlation:.8}",
            cold.peak,
            warm.peak,
            warm.active,
            warm.cache,
            warm.post_clear_active,
            warm.post_clear_cache,
            warm.wall.as_millis(),
        );
        runs.push((name, warm));
    }

    let resident = &runs[0].1;
    for (name, run) in &runs[1..] {
        let (max, mean, correlation) = drift(&resident.image, &run.image);
        assert!(correlation > 0.90, "{name} correlation {correlation}");
        println!(
            "DRIFT name={name} max_delta={max} mean_delta={mean:.6} correlation={correlation:.8}"
        );
    }

    // Timed cooperative cancellation must cross a physical bounded-attention/block boundary as the
    // typed cancellation variant, then the same cached generator must recover.
    let mut cancel_request = base_request();
    cancel_request.memory = Some(rung4_memory());
    let cancel = cancel_request.cancel.clone();
    let delay_ms = std::env::var("LENS_LADDER_CANCEL_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(500_u64);
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(delay_ms));
        cancel.cancel();
    });
    let error = generator
        .generate(&cancel_request, &mut |_| {})
        .expect_err("timed cancellation must interrupt rung4");
    timer.join().unwrap();
    assert_typed_cancel(error);
    clear_cache();
    let cancel_post_clear = (get_active_memory(), get_cache_memory());
    let cancel_recovery = run(generator.as_ref(), rung4_memory()).expect("cancel recovery");
    println!(
        "TERMINAL_RESULT kind=cancel post_clear_active_bytes={} post_clear_cache_bytes={} recovery_peak_bytes={} recovery_post_clear_active_bytes={} recovery_post_clear_cache_bytes={}",
        cancel_post_clear.0,
        cancel_post_clear.1,
        cancel_recovery.peak,
        cancel_recovery.post_clear_active,
        cancel_recovery.post_clear_cache,
    );

    // Decode-boundary injection proves DiT shedding/error cleanup and a clean follow-up request.
    let mut fault = rung4_memory();
    fault.calibration_fault_harness_authorized = true;
    fault.calibration_error_phase = Some(MemoryPhase::Decode);
    let mut fault_request = base_request();
    fault_request.memory = Some(fault);
    let error = generator
        .generate(&fault_request, &mut |_| {})
        .expect_err("decode fault must surface");
    assert!(error.to_string().contains("calibration fault at Decode"));
    clear_cache();
    let fault_post_clear = (get_active_memory(), get_cache_memory());
    let fault_recovery = run(generator.as_ref(), rung4_memory()).expect("fault recovery");
    println!(
        "TERMINAL_RESULT kind=decode_fault post_clear_active_bytes={} post_clear_cache_bytes={} recovery_peak_bytes={} recovery_post_clear_active_bytes={} recovery_post_clear_cache_bytes={}",
        fault_post_clear.0,
        fault_post_clear.1,
        fault_recovery.peak,
        fault_recovery.post_clear_active,
        fault_recovery.post_clear_cache,
    );
    let rung4 = &runs[4].1;
    assert_eq!(drift(&rung4.image, &cancel_recovery.image).0, 0);
    assert_eq!(drift(&rung4.image, &fault_recovery.image).0, 0);

    let resident_peak = resident.peak;
    let rung4_peak = rung4.peak;
    let clean_floor = (rung4.post_clear_active, rung4.post_clear_cache);
    let staged_floor = (runs[1].1.post_clear_active, runs[1].1.post_clear_cache);
    assert_eq!(
        clean_floor, staged_floor,
        "rung4 must return to the clean staged allocator floor"
    );
    assert_eq!(
        cancel_post_clear, clean_floor,
        "cancel retained allocations"
    );
    assert_eq!(
        fault_post_clear, clean_floor,
        "decode fault retained allocations"
    );
    assert_eq!(
        (
            cancel_recovery.post_clear_active,
            cancel_recovery.post_clear_cache
        ),
        clean_floor,
        "cancel recovery retained allocations"
    );
    assert_eq!(
        (
            fault_recovery.post_clear_active,
            fault_recovery.post_clear_cache
        ),
        clean_floor,
        "decode-fault recovery retained allocations"
    );
    drop(generator);
    clear_cache();
    let final_active = get_active_memory();
    let final_cache = get_cache_memory();
    assert_eq!(
        (final_active, final_cache),
        clean_floor,
        "dropping the generator must not retain more than the clean rung4 floor"
    );
    println!(
        "TERMINAL_RESULT status=pass provider={} tier={:?} fingerprint={} load_shape={:?} size={} resident_peak_bytes={} rung4_peak_bytes={} cancel_recovery_peak_bytes={} decode_fault_recovery_peak_bytes={} process_baseline_active_bytes={} process_baseline_cache_bytes={} post_clear_active_bytes={} post_clear_cache_bytes={}",
        provider(),
        tier(),
        fingerprint,
        LoadShape::DeferredMaterialization,
        edge(),
        resident_peak,
        rung4_peak,
        cancel_recovery.peak,
        fault_recovery.peak,
        process_baseline_active,
        process_baseline_cache,
        final_active,
        final_cache,
    );
    println!(
        "RESULT status=pass provider={} tier={:?} fingerprint={} size={} decode_edge={} decode_overlap={} attention_chunk_size={} transformer_window_size={} transformer_component=both resident_peak_gib={:.3} rung4_peak_gib={:.3}",
        provider(),
        tier(),
        fingerprint,
        edge(),
        mlx_gen_lens::memory_strategy::DECODE_TILE_EDGE,
        mlx_gen_lens::memory_strategy::DECODE_OVERLAP,
        mlx_gen_lens::memory_strategy::ATTENTION_CHUNK_SIZE,
        mlx_gen_lens::memory_strategy::TEXT_ENCODER_WINDOW,
        resident_peak as f64 / GIB,
        rung4_peak as f64 / GIB,
    );
}

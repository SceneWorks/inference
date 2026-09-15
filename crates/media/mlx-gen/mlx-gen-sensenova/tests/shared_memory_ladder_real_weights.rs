//! SC-15513 serial real-Metal runner for SenseNova-U1 quality/fast Q8.
//!
//! Run one artifact at a time; this test never fetches weights:
//!
//! ```text
//! SENSENOVA_LADDER_PROVIDER=sensenova_u1_8b SENSENOVA_LADDER_ROOT=/.../q8 \
//! cargo test -p mlx-gen-sensenova --release --test integration shared_memory_ladder_real_weights:: \
//!   -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use mlx_gen::attention::AttentionBudget;
use mlx_gen::gen_core::{GenerationMemory, MemoryPhase, TransformerComponent};
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Generator, Image, LoadShape, LoadSpec,
    Quant, WeightsSource,
};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("set {key}; see test documentation"))
}

fn provider() -> String {
    let provider = env("SENSENOVA_LADDER_PROVIDER");
    assert!(
        provider == mlx_gen_sensenova::MODEL_ID || provider == mlx_gen_sensenova::MODEL_ID_FAST
    );
    provider
}

fn root() -> PathBuf {
    let root = PathBuf::from(env("SENSENOVA_LADDER_ROOT"));
    assert!(
        root.is_dir(),
        "artifact root does not exist: {}",
        root.display()
    );
    root
}

fn spec(shape: LoadShape) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root()))
        .with_quant(Quant::Q8)
        .with_load_shape(shape)
}

fn load(shape: LoadShape) -> Box<dyn Generator> {
    let spec = spec(shape);
    mlx_gen_sensenova::provider_registry()
        .unwrap()
        .load(&provider(), &spec)
        .expect("load cached SenseNova Q8")
}

fn request(memory: GenerationMemory) -> GenerationRequest {
    let size = std::env::var("SENSENOVA_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    let mode = std::env::var("SENSENOVA_LADDER_MODE").unwrap_or_else(|_| "t2i".to_owned());
    let conditioning = match mode.as_str() {
        "t2i" => Vec::new(),
        "edit" => vec![Conditioning::Reference {
            image: Image {
                width: 256,
                height: 256,
                pixels: (0..256 * 256 * 3)
                    .map(|index| (index % 256) as u8)
                    .collect(),
            },
            strength: None,
        }],
        other => panic!("SENSENOVA_LADDER_MODE must be t2i or edit, got {other}"),
    };
    GenerationRequest {
        prompt: "a red fox in a snowy pine clearing at dawn, detailed photograph".to_owned(),
        width: size,
        height: size,
        count: 1,
        steps: Some(1),
        guidance: Some(1.0),
        seed: Some(15513),
        true_cfg: (mode == "edit").then_some(1.0),
        conditioning,
        memory: Some(memory),
        ..Default::default()
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

fn run(generator: &dyn Generator, memory: GenerationMemory) -> Run {
    clear_cache();
    reset_peak_memory();
    let started = Instant::now();
    let output = generator.generate(&request(memory), &mut |_| {}).unwrap();
    let image = match output {
        GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
        other => panic!("expected one image, got {other:?}"),
    };
    let active = get_active_memory();
    let cache = get_cache_memory();
    let peak = get_peak_memory();
    clear_cache();
    Run {
        image,
        peak,
        active,
        cache,
        post_clear_active: get_active_memory(),
        post_clear_cache: get_cache_memory(),
        wall: started.elapsed(),
    }
}

fn rung4_memory() -> GenerationMemory {
    GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(mlx_gen_sensenova::memory_strategy::ATTENTION_CHUNK_SIZE),
        stream_transformer_blocks: true,
        transformer_window_size: Some(mlx_gen_sensenova::memory_strategy::TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TransformerComponent::Dit),
        ..Default::default()
    }
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
        let (x, y) = (f64::from(x), f64::from(y));
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

#[test]
#[ignore = "needs exact cached SenseNova Q8 weights and exclusive Apple/Metal access"]
fn serial_resident_attention_and_rung4() {
    clear_cache();
    let artifact_identity = mlx_gen_sensenova::memory_strategy::verified_runner_artifact(
        &provider(),
        &spec(LoadShape::EagerMaterialization),
    )
    .expect("verify exact provider artifact and eager calibrated contract before generation");
    let deferred_artifact_identity = mlx_gen_sensenova::memory_strategy::verified_runner_artifact(
        &provider(),
        &spec(LoadShape::DeferredMaterialization),
    )
    .expect("verify exact provider artifact and deferred calibrated contract before generation");
    assert_eq!(artifact_identity, deferred_artifact_identity);
    let size = request(GenerationMemory::default()).width as i32;
    let mode = std::env::var("SENSENOVA_LADDER_MODE").unwrap_or_else(|_| "t2i".to_owned());
    let image_queries = size / 32;
    let image_queries = image_queries * image_queries;
    let planned_rows_upper_bound = AttentionBudget::from_score_elements(
        u64::from(mlx_gen_sensenova::memory_strategy::ATTENTION_CHUNK_SIZE),
        true,
    )
    .query_block(1, 32, image_queries, image_queries);
    assert!(
        planned_rows_upper_bound < image_queries,
        "published bounded-attention arm does not chunk at {size}x{size}"
    );

    let eager = load(LoadShape::EagerMaterialization);
    let resident = run(eager.as_ref(), GenerationMemory::default());
    let attention = run(
        eager.as_ref(),
        GenerationMemory {
            chunk_attention: true,
            attention_chunk_size: Some(mlx_gen_sensenova::memory_strategy::ATTENTION_CHUNK_SIZE),
            ..Default::default()
        },
    );
    drop(eager);
    clear_cache();

    let deferred = load(LoadShape::DeferredMaterialization);
    let rung4 = run(deferred.as_ref(), rung4_memory());

    for (name, run) in [
        ("resident", &resident),
        ("attention", &attention),
        ("rung4", &rung4),
    ] {
        println!(
            "ARM provider={} name={name} peak_bytes={} active_bytes={} cache_bytes={} post_clear_active_bytes={} post_clear_cache_bytes={} wall_ms={}",
            provider(),
            run.peak,
            run.active,
            run.cache,
            run.post_clear_active,
            run.post_clear_cache,
            run.wall.as_millis(),
        );
    }
    for (name, run) in [("attention", &attention), ("rung4", &rung4)] {
        let (max, mean, correlation) = drift(&resident.image, &run.image);
        println!(
            "DRIFT provider={} name={name} max_delta={max} mean_delta={mean:.6} correlation={correlation:.8}",
            provider()
        );
        assert!(correlation > 0.99, "{name} correlation {correlation}");
    }
    assert!(
        rung4.peak < resident.peak,
        "rung4 did not reduce request peak: {} >= {}",
        rung4.peak,
        resident.peak
    );

    // Timed cancellation must remain typed, release the partial window, and allow a clean reuse of
    // the same deferred generator.
    let cancel_request = request(rung4_memory());
    let cancel = cancel_request.cancel.clone();
    let cancel_ms = std::env::var("SENSENOVA_LADDER_CANCEL_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(500_u64);
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(cancel_ms));
        cancel.cancel();
    });
    let error = deferred
        .generate(&cancel_request, &mut |_| {})
        .expect_err("timed cancellation must interrupt generation");
    timer.join().unwrap();
    assert!(
        matches!(error, mlx_gen::gen_core::Error::Canceled),
        "{error:?}"
    );
    clear_cache();
    let cancel_floor = (get_active_memory(), get_cache_memory());
    let cancel_recovery = run(deferred.as_ref(), rung4_memory());
    println!(
        "TERMINAL_RESULT provider={} kind=cancel post_clear_active_bytes={} post_clear_cache_bytes={} recovery_peak_bytes={} recovery_post_clear_active_bytes={} recovery_post_clear_cache_bytes={}",
        provider(),
        cancel_floor.0,
        cancel_floor.1,
        cancel_recovery.peak,
        cancel_recovery.post_clear_active,
        cancel_recovery.post_clear_cache,
    );

    // Deterministic, explicitly authorized fault after the first Gen block has been reconstructed
    // from the lazy snapshot view. The shared run_windowed error path must drain/drop it, and the
    // cached generator must recover on the next request.
    let mut fault = rung4_memory();
    fault.authorize_calibration_fault(MemoryPhase::Denoise);
    let error = deferred
        .generate(&request(fault), &mut |_| {})
        .expect_err("stream fault must surface");
    assert!(error
        .to_string()
        .contains("calibration fault after Gen block materialization"));
    clear_cache();
    let fault_floor = (get_active_memory(), get_cache_memory());
    let fault_recovery = run(deferred.as_ref(), rung4_memory());
    println!(
        "TERMINAL_RESULT provider={} kind=stream_fault post_clear_active_bytes={} post_clear_cache_bytes={} recovery_peak_bytes={} recovery_post_clear_active_bytes={} recovery_post_clear_cache_bytes={}",
        provider(),
        fault_floor.0,
        fault_floor.1,
        fault_recovery.peak,
        fault_recovery.post_clear_active,
        fault_recovery.post_clear_cache,
    );
    assert!(drift(&rung4.image, &cancel_recovery.image).2 > 0.9999);
    assert!(drift(&rung4.image, &fault_recovery.image).2 > 0.9999);
    assert_eq!(
        cancel_floor, fault_floor,
        "terminal paths retained different allocations"
    );
    assert_eq!(
        cancel_floor,
        (rung4.post_clear_active, rung4.post_clear_cache),
        "terminal paths did not return to the clean rung4 allocator floor"
    );

    println!(
        "RESULT status=pass provider={} tier=q8 artifact_sha256={} mode={} size={} resident_peak_bytes={} attention_peak_bytes={} rung4_peak_bytes={} attention_query_block_rows_upper_bound={} window_size={} cancel_post_clear_active_bytes={} cancel_post_clear_cache_bytes={} fault_post_clear_active_bytes={} fault_post_clear_cache_bytes={}",
        provider(),
        artifact_identity,
        mode,
        size,
        resident.peak,
        attention.peak,
        rung4.peak,
        planned_rows_upper_bound,
        mlx_gen_sensenova::memory_strategy::TRANSFORMER_WINDOW_SIZE,
        cancel_floor.0,
        cancel_floor.1,
        fault_floor.0,
        fault_floor.1,
    );
}

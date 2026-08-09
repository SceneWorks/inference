//! SC-15806 real-weight proof for one generator serving a mixed warm/staged request sequence.
//!
//! ```text
//! MLX_GEN_ZIMAGE_SNAPSHOT=<q4 tier dir> ZIMAGE_SIZE=512 ZIMAGE_STEPS=1 \
//!   cargo test -p mlx-gen-z-image --release --test request_scoped_residency_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use common::tier_snapshot as snapshot;
use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadPhase, LoadSpec, OffloadPolicy, Progress,
    Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
use std::time::{Duration, Instant};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn spec() -> LoadSpec {
    let snapshot = snapshot();
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone()))
        // Both legacy values are intentionally ignored by Z-Image after SC-15806. Keep Resident here
        // to prove a request can still select staged execution on a generator constructed from it.
        .with_offload_policy(OffloadPolicy::Resident);
    match snapshot.file_name().and_then(|name| name.to_str()) {
        Some("q4") => spec = spec.with_quant(Quant::Q4),
        Some("q8") => spec = spec.with_quant(Quant::Q8),
        _ => {}
    }
    spec
}

fn request(stage_residency: bool) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: env_u32("ZIMAGE_SIZE", 512),
        height: env_u32("ZIMAGE_SIZE", 512),
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_STEPS", 1)),
        memory: Some(GenerationMemory {
            stage_residency,
            ..Default::default()
        }),
        ..Default::default()
    }
}

struct Run {
    image: Image,
    peak_bytes: usize,
    retained_bytes: usize,
    wall: Duration,
    text_loads: usize,
    heavy_loads: usize,
}

fn run(generator: &dyn mlx_gen::Generator, stage_residency: bool) -> Run {
    clear_cache();
    reset_peak_memory();
    let started = Instant::now();
    let mut text_loads = 0;
    let mut heavy_loads = 0;
    let output = generator
        .generate(&request(stage_residency), &mut |progress| match progress {
            Progress::Loading(LoadPhase::TextEncoder) => text_loads += 1,
            Progress::Loading(LoadPhase::Renderer) => heavy_loads += 1,
            _ => {}
        })
        .expect("generation");
    let wall = started.elapsed();
    let peak_bytes = get_peak_memory();
    clear_cache();
    let retained_bytes = get_active_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    Run {
        image,
        peak_bytes,
        retained_bytes,
        wall,
        text_loads,
        heavy_loads,
    }
}

fn print_run(label: &str, run: &Run) {
    println!(
        "  {label:<20} peak={:.3} GiB retained={:.3} GiB wall={:.3}s loads={}+{}",
        run.peak_bytes as f64 / GIB,
        run.retained_bytes as f64 / GIB,
        run.wall.as_secs_f64(),
        run.text_loads,
        run.heavy_loads,
    );
}

#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn one_generator_serves_warm_staged_warm_with_identical_pixels() {
    let generator = mlx_gen_z_image::load(&spec()).expect("load request-scoped generator");

    // Populate the warm pair once, then measure a true warm-cache request.
    let prime = run(generator.as_ref(), false);
    let warm = run(generator.as_ref(), false);
    // The same generator must evict that pair and execute phase-staged.
    let staged = run(generator.as_ref(), true);
    // Staged leaves no warm pair; the next request rebuilds one and returns to warm execution.
    let warm_after_staged = run(generator.as_ref(), false);

    println!("\nSC-15806 one-generator request-scoped residency");
    print_run("warm", &warm);
    print_run("staged", &staged);
    print_run("warm after staged", &warm_after_staged);
    println!(
        "  mixed W->S->W         peak={:.3} GiB wall={:.3}s",
        [
            warm.peak_bytes,
            staged.peak_bytes,
            warm_after_staged.peak_bytes
        ]
        .into_iter()
        .max()
        .unwrap_or_default() as f64
            / GIB,
        (warm.wall + staged.wall + warm_after_staged.wall).as_secs_f64(),
    );

    assert_eq!((prime.text_loads, prime.heavy_loads), (1, 1));
    assert_eq!(
        (warm.text_loads, warm.heavy_loads),
        (0, 0),
        "a warm request reloaded components"
    );
    assert_eq!(
        (staged.text_loads, staged.heavy_loads),
        (1, 1),
        "the staged request did not run both phase loaders"
    );
    assert_eq!(
        (warm_after_staged.text_loads, warm_after_staged.heavy_loads),
        (1, 1),
        "the post-staged warm request reused a component that should have been released"
    );
    for (label, image) in [
        ("staged", &staged.image),
        ("warm after staged", &warm_after_staged.image),
    ] {
        assert_eq!(
            warm.image.pixels, image.pixels,
            "{label} changed output pixels"
        );
    }
    assert!(
        staged.peak_bytes < warm.peak_bytes,
        "staged peak {:.3} GiB did not beat warm peak {:.3} GiB",
        staged.peak_bytes as f64 / GIB,
        warm.peak_bytes as f64 / GIB,
    );
}

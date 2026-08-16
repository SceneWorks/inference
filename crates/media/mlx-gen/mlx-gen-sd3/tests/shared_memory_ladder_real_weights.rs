//! SC-15522 / SC-18606 hardware-gated SD3.5 shared memory-ladder evidence.
//!
//! `SD3_LADDER_SNAPSHOT` names the exact content-pinned SC-15522 Large BF16 snapshot and is the
//! only one that binds the measured calibration fingerprint. SC-18606 made rung 4 reachable for
//! Large-Turbo and Medium through the variant-generic structural admission, so
//! `SD3_TURBO_LADDER_SNAPSHOT` and `SD3_MEDIUM_LADDER_SNAPSHOT` supply their evidence lanes. These
//! tests are `#[ignore]`d and were NOT run in the SC-18606 code change; the epic's terminal
//! evidence phase owns them.

use std::path::PathBuf;

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec, OffloadPolicy,
    Progress, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("SD3_LADDER_SNAPSHOT")
            .expect("set SD3_LADDER_SNAPSHOT to the exact cached SD3.5-Large BF16 snapshot"),
    )
}

/// The two SC-18606 lanes. Absent env var ⇒ that variant's evidence is simply not being collected
/// on this host; the test says so out loud rather than passing silently.
fn optional_snapshot(variable: &str) -> Option<PathBuf> {
    std::env::var(variable).ok().map(PathBuf::from)
}

fn spec_at(root: &std::path::Path, deferred: bool) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(if deferred {
            LoadShape::DeferredMaterialization
        } else {
            LoadShape::EagerMaterialization
        })
}

fn spec(deferred: bool) -> LoadSpec {
    spec_at(&snapshot(), deferred)
}

fn size() -> u32 {
    std::env::var("SD3_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(768)
}

/// Per-variant sampling recipe. Large-Turbo is ADD-distilled with guidance baked in and advertises
/// neither `supports_guidance` nor `supports_negative_prompt`, so sending either is a validation
/// error rather than a memory finding — the trio cannot share one request literal.
fn request_for(provider: &str, memory: GenerationMemory) -> GenerationRequest {
    let turbo = provider == "sd3_5_large_turbo";
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: (!turbo).then(|| "blurry, low quality".to_owned()),
        guidance: (!turbo).then_some(if provider == "sd3_5_medium" { 5.0 } else { 3.5 }),
        width: size(),
        height: size(),
        steps: Some(1),
        seed: Some(1234),
        memory: Some(memory),
        ..Default::default()
    }
}

struct Run {
    conditioning: usize,
    denoise: usize,
    decode: usize,
    image: Image,
}

impl Run {
    fn peak(&self) -> usize {
        self.conditioning.max(self.denoise).max(self.decode)
    }
}

fn run(memory: GenerationMemory) -> Run {
    run_provider("sd3_5_large", &snapshot(), memory)
}

fn run_provider(provider: &str, root: &std::path::Path, memory: GenerationMemory) -> Run {
    let generator = mlx_gen_sd3::provider_registry()
        .expect("SD3 registry")
        .load(provider, &spec_at(root, memory.stream_transformer_blocks))
        .unwrap_or_else(|error| panic!("load {provider}: {error}"));
    clear_cache();
    reset_peak_memory();
    let mut conditioning = 0;
    let mut denoise = 0;
    let mut progress = |event| match event {
        Progress::Step { current: 1, .. } => {
            conditioning = get_peak_memory();
            reset_peak_memory();
        }
        Progress::Decoding if denoise == 0 => {
            denoise = get_peak_memory();
            reset_peak_memory();
        }
        _ => {}
    };
    let output = generator
        .generate(&request_for(provider, memory), &mut progress)
        .expect("generate SD3.5 image");
    let decode = get_peak_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected image output, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    assert!(conditioning > 0 && denoise > 0 && decode > 0);
    Run {
        conditioning,
        denoise,
        decode,
        image,
    }
}

fn full_ladder() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        tile_vae_decode: true,
        chunk_attention: true,
        stream_transformer_blocks: true,
        decode_tile_edge: Some(mlx_gen_sd3::memory_strategy::DECODE_TILE_EDGE),
        decode_overlap: Some(mlx_gen_sd3::memory_strategy::DECODE_OVERLAP),
        attention_chunk_size: Some(mlx_gen_sd3::memory_strategy::ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(mlx_gen_sd3::memory_strategy::TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TransformerComponent::Dit),
        ..Default::default()
    }
}

fn delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0;
    let mut sum = 0_u64;
    for (left, right) in a.pixels.iter().zip(&b.pixels) {
        let d = left.abs_diff(*right);
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / a.pixels.len() as f64)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

fn reference(size: u32) -> Image {
    let edge = size as usize;
    let mut pixels = Vec::with_capacity(edge * edge * 3);
    for y in 0..edge {
        for x in 0..edge {
            pixels.extend_from_slice(&[
                ((x * 255) / edge) as u8,
                ((y * 255) / edge) as u8,
                (((x + y) * 127) / edge) as u8,
            ]);
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
    }
}

#[test]
#[ignore = "needs the exact SD3.5-Large BF16 cache snapshot and Apple/Metal"]
fn full_ladder_executes_single_reference_img2img() {
    let edge = size();
    let render = |deferred, memory| {
        let generator = mlx_gen_sd3::provider_registry()
            .expect("SD3 registry")
            .load("sd3_5_large", &spec(deferred))
            .expect("load SD3.5 Large");
        let request = GenerationRequest {
            prompt: "turn this gradient into a painted mountain landscape".into(),
            negative_prompt: Some("blurry".into()),
            guidance: Some(3.5),
            width: edge,
            height: edge,
            steps: Some(2),
            seed: Some(4321),
            conditioning: vec![Conditioning::Reference {
                image: reference(edge),
                strength: Some(0.5),
            }],
            memory: Some(memory),
            ..Default::default()
        };
        clear_cache();
        reset_peak_memory();
        let GenerationOutput::Images(mut images) = generator
            .generate(&request, &mut |_| {})
            .expect("SD3.5 img2img")
        else {
            panic!("expected image output");
        };
        let peak = get_peak_memory();
        let image = images.pop().expect("one image");
        drop(generator);
        clear_cache();
        (image, peak)
    };
    let (staged, staged_peak) = render(
        false,
        GenerationMemory {
            stage_residency: true,
            ..Default::default()
        },
    );
    let (full, full_peak) = render(true, full_ladder());
    assert_eq!((full.width, full.height), (edge, edge));
    let (max, mean_delta) = delta(&staged, &full);
    assert!(max <= 64 && mean_delta < 4.0);
    assert!(full_peak < staged_peak);
    let mean = full
        .pixels
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / full.pixels.len() as f64;
    assert!(mean > 2.0 && mean < 253.0, "degenerate img2img output");
    println!(
        "RESULT status=pass provider=sd3_5_large route=single-reference-img2img ladder=full staged_peak_gib={:.3} full_peak_gib={:.3} max_delta={max} mean_delta={mean_delta:.6} output_mean={mean:.3}",
        gib(staged_peak), gib(full_peak)
    );
}

#[test]
#[ignore = "needs the exact SD3.5-Large BF16 cache snapshot and Apple/Metal"]
fn full_shared_ladder_reduces_peak_and_preserves_output() {
    assert!(snapshot().components().any(|component| {
        component.as_os_str() == mlx_gen_sd3::memory_strategy::CALIBRATED_REVISION
    }));
    let latent_edge = (size() / 8) as i32;
    let plan = mlx_gen::tiling::TilingConfig::spatial_only(
        mlx_gen_sd3::memory_strategy::DECODE_TILE_EDGE as i32,
        mlx_gen_sd3::memory_strategy::DECODE_OVERLAP as i32,
    )
    .plan(
        mlx_gen::tiling::VaeTiling::QWEN_IMAGE,
        1,
        latent_edge,
        latent_edge,
    );
    assert!(plan.h.len() > 1 && plan.w.len() > 1);
    let registry = mlx_gen_sd3::provider_registry().expect("SD3 registry");
    let contract = registry
        .memory_strategy_contract("sd3_5_large", &spec(true))
        .expect("query exact Large contract")
        .expect("Large memory contract");
    assert_eq!(
        contract.calibration.as_ref().unwrap().fingerprint,
        mlx_gen_sd3::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
    );
    for strategy in mlx_gen::gen_core::MemoryStrategy::ALL {
        assert_eq!(
            contract.capability(strategy).unwrap().support,
            mlx_gen::gen_core::MemoryStrategySupport::Implemented,
            "exact deferred Large must implement {strategy:?}"
        );
    }

    let arms = [
        (
            "staged",
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            },
        ),
        (
            "bounded-decode",
            GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(mlx_gen_sd3::memory_strategy::DECODE_TILE_EDGE),
                decode_overlap: Some(mlx_gen_sd3::memory_strategy::DECODE_OVERLAP),
                ..Default::default()
            },
        ),
        (
            "bounded-attention",
            GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                decode_tile_edge: Some(mlx_gen_sd3::memory_strategy::DECODE_TILE_EDGE),
                decode_overlap: Some(mlx_gen_sd3::memory_strategy::DECODE_OVERLAP),
                attention_chunk_size: Some(mlx_gen_sd3::memory_strategy::ATTENTION_CHUNK_SIZE),
                ..Default::default()
            },
        ),
        ("bounded-transformer", full_ladder()),
    ];
    let mut runs = Vec::new();
    for (name, memory) in arms {
        let run = run(memory);
        println!(
            "ARM name={name} conditioning_gib={:.3} denoise_gib={:.3} decode_gib={:.3} request_gib={:.3}",
            gib(run.conditioning), gib(run.denoise), gib(run.decode), gib(run.peak())
        );
        runs.push((name, run));
    }
    let staged = &runs[0].1;
    let tiled = &runs[1].1;
    let (decode_max, decode_mean) = delta(&staged.image, &tiled.image);
    assert!(decode_max <= 64 && decode_mean < 4.0);
    for (name, run) in runs.iter().skip(2) {
        let (max, mean) = delta(&tiled.image, &run.image);
        assert!(max <= 1, "{name} changed denoise: max={max} mean={mean}");
    }
    let full = &runs.last().unwrap().1;
    assert!(full.peak() < staged.peak());
    println!(
        "RESULT status=pass provider=sd3_5_large tier=bf16 revision={} fingerprint={} staged_peak_gib={:.3} full_peak_gib={:.3} decode_max_delta={} decode_mean_delta={:.6}",
        mlx_gen_sd3::memory_strategy::CALIBRATED_REVISION,
        mlx_gen_sd3::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        gib(staged.peak()),
        gib(full.peak()),
        decode_max,
        decode_mean,
    );
}

/// SC-18606: the same resident-versus-selected evidence for the two variants that had no rung-4
/// route before this story. Their contracts carry NO measured calibration — the structural
/// admission is identity-pinned, not content-pinned — so the printed RESULT line records the
/// admission and authority explicitly rather than implying Large's evidence covers them.
#[test]
#[ignore = "needs the SD3.5 Large-Turbo / Medium BF16 snapshots and Apple/Metal"]
fn structurally_admitted_variants_reduce_peak_and_preserve_output() {
    let lanes = [
        ("sd3_5_large_turbo", "SD3_TURBO_LADDER_SNAPSHOT"),
        ("sd3_5_medium", "SD3_MEDIUM_LADDER_SNAPSHOT"),
    ];
    let mut collected = 0;
    for (provider, variable) in lanes {
        let Some(root) = optional_snapshot(variable) else {
            println!("RESULT status=skip provider={provider} reason={variable}-unset");
            continue;
        };
        collected += 1;
        let registry = mlx_gen_sd3::provider_registry().expect("SD3 registry");
        let contract = registry
            .memory_strategy_contract(provider, &spec_at(&root, true))
            .expect("query contract")
            .expect("memory contract");
        for strategy in mlx_gen::gen_core::MemoryStrategy::ALL {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                mlx_gen::gen_core::MemoryStrategySupport::Implemented,
                "{provider} deferred route must implement {strategy:?}"
            );
        }
        assert!(
            contract.calibration.is_none(),
            "{provider} must not claim measured calibration from a structural admission"
        );

        let staged = run_provider(
            provider,
            &root,
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            },
        );
        let full = run_provider(provider, &root, full_ladder());
        let (max, mean) = delta(&staged.image, &full.image);
        assert!(
            max <= 64 && mean < 4.0,
            "{provider} ladder changed the image: max={max} mean={mean}"
        );
        assert!(
            full.peak() < staged.peak(),
            "{provider} full ladder did not reduce peak: staged={} full={}",
            staged.peak(),
            full.peak()
        );
        println!(
            "RESULT status=pass provider={provider} tier=bf16 admission=structural authority=estimated staged_peak_gib={:.3} full_peak_gib={:.3} max_delta={max} mean_delta={mean:.6}",
            gib(staged.peak()),
            gib(full.peak()),
        );
    }
    assert!(
        collected > 0,
        "set SD3_TURBO_LADDER_SNAPSHOT and/or SD3_MEDIUM_LADDER_SNAPSHOT to collect this evidence"
    );
}

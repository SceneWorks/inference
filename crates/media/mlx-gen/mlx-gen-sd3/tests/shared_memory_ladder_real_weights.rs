//! SC-15522 hardware-gated SD3.5-Large shared memory-ladder evidence.

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

fn spec(deferred: bool) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(snapshot()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(if deferred {
            LoadShape::DeferredMaterialization
        } else {
            LoadShape::EagerMaterialization
        })
}

fn size() -> u32 {
    std::env::var("SD3_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(768)
}

fn request(memory: GenerationMemory) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(3.5),
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
    let generator = mlx_gen_sd3::provider_registry()
        .expect("SD3 registry")
        .load("sd3_5_large", &spec(memory.stream_transformer_blocks))
        .expect("load SD3.5 Large");
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
        .generate(&request(memory), &mut progress)
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

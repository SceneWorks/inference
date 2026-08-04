//! SC-15518 hardware-gated FLUX.2 Klein shared memory-ladder evidence.
//!
//! The first three arms use the same staged BF16 snapshot with eager transformer materialization.
//! The final arm changes only the load shape and request memory to use a one-block DiT window.  Every
//! arm is a separate serial load/run/drop cycle so the Metal device is never contended.

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
        std::env::var("FLUX2_KLEIN_LADDER_SNAPSHOT")
            .expect("set FLUX2_KLEIN_LADDER_SNAPSHOT to the exact cached BF16 snapshot directory"),
    )
}

fn true_v2_assembly() -> PathBuf {
    required_path("FLUX2_KLEIN_TRUE_V2_ASSEMBLY")
}

fn required_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn spec_for(root: PathBuf, deferred: bool) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(if deferred {
            LoadShape::DeferredMaterialization
        } else {
            LoadShape::EagerMaterialization
        })
}

fn spec(deferred: bool) -> LoadSpec {
    spec_for(snapshot(), deferred)
}

fn request(memory: GenerationMemory) -> GenerationRequest {
    let size = ladder_size();
    let steps = std::env::var("FLUX2_KLEIN_LADDER_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        count: 1,
        steps: Some(steps),
        seed: Some(1234),
        memory: Some(memory),
        ..Default::default()
    }
}

fn ladder_size() -> u32 {
    std::env::var("FLUX2_KLEIN_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(768)
}

struct Run {
    conditioning: usize,
    denoise: usize,
    decode: usize,
    seconds: f64,
    image: Image,
}

impl Run {
    fn request_peak(&self) -> usize {
        self.conditioning.max(self.denoise).max(self.decode)
    }
}

fn run(memory: GenerationMemory) -> Run {
    run_with_spec(spec(memory.stream_transformer_blocks), memory)
}

fn run_with_spec(spec: LoadSpec, memory: GenerationMemory) -> Run {
    let generator = mlx_gen_flux2::provider_registry()
        .expect("FLUX.2 registry")
        .load(mlx_gen_flux2::FLUX2_KLEIN_9B_ID, &spec)
        .expect("load FLUX.2 Klein BF16");
    clear_cache();
    reset_peak_memory();
    let start = std::time::Instant::now();
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
        .expect("generate FLUX.2 Klein image");
    let decode = get_peak_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected image output, got {other:?}"),
    };
    let seconds = start.elapsed().as_secs_f64();
    drop(generator);
    clear_cache();
    assert!(conditioning > 0 && denoise > 0 && decode > 0);
    Run {
        conditioning,
        denoise,
        decode,
        seconds,
        image,
    }
}

#[test]
#[ignore = "needs the exact assembled FLUX.2 Klein True V2 BF16 artifact and Apple/Metal"]
fn full_ladder_runs_the_exact_true_v2_family_artifact() {
    let root = true_v2_assembly();
    let load = spec_for(root.clone(), true);
    let contract = mlx_gen_flux2::provider_registry()
        .expect("FLUX.2 registry")
        .memory_strategy_contract(mlx_gen_flux2::FLUX2_KLEIN_9B_ID, &load)
        .expect("query True V2 memory contract")
        .expect("True V2 memory contract");
    let calibration = contract.calibration.expect("True V2 calibration identity");
    assert_eq!(
        calibration.fingerprint,
        "flux2-klein-9b-bf16-mlx-shared-ladder-true-two-t2i-v1"
    );
    let staged = run_with_spec(
        spec_for(root, false),
        GenerationMemory {
            stage_residency: true,
            ..Default::default()
        },
    );
    let full = run_with_spec(load, full_ladder_memory());
    let (max, mean_delta) = image_delta(&staged.image, &full.image);
    assert!(
        max <= 1,
        "True V2 ladder changed output: max={max} mean={mean_delta:.6}"
    );
    assert!(
        full.request_peak() < staged.request_peak(),
        "True V2 full ladder did not reduce request peak"
    );
    let mean = full
        .image
        .pixels
        .iter()
        .map(|pixel| f64::from(*pixel))
        .sum::<f64>()
        / full.image.pixels.len() as f64;
    assert!(mean > 2.0 && mean < 253.0, "True V2 output is degenerate");
    println!(
        "RESULT status=pass provider={} family=true-v2 fingerprint={} staged_peak_gib={:.3} full_peak_gib={:.3} seconds={:.2}",
        mlx_gen_flux2::FLUX2_KLEIN_9B_ID,
        calibration.fingerprint,
        gib(staged.request_peak()),
        gib(full.request_peak()),
        full.seconds,
    );
}

fn image_delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0;
    let mut total = 0_u64;
    for (left, right) in a.pixels.iter().zip(&b.pixels) {
        let delta = left.abs_diff(*right);
        max = max.max(delta);
        total += u64::from(delta);
    }
    (max, total as f64 / a.pixels.len() as f64)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

fn full_ladder_memory() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        tile_vae_decode: true,
        chunk_attention: true,
        stream_transformer_blocks: true,
        decode_tile_edge: Some(mlx_gen_flux2::memory_strategy::DECODE_TILE_EDGE),
        decode_overlap: Some(mlx_gen_flux2::memory_strategy::DECODE_OVERLAP),
        attention_chunk_size: Some(mlx_gen_flux2::memory_strategy::ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(mlx_gen_flux2::memory_strategy::TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TransformerComponent::Dit),
        ..Default::default()
    }
}

fn synthetic_reference(size: u32, seed: usize) -> Image {
    let edge = size as usize;
    let mut pixels = Vec::with_capacity(edge * edge * 3);
    for y in 0..edge {
        for x in 0..edge {
            pixels.extend_from_slice(&[
                (((x + seed * 31) * 255) / edge) as u8,
                ((y * 255) / edge) as u8,
                (((x + y + seed * 17) * 127) / edge) as u8,
            ]);
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
    }
}

fn render_edit(provider_id: &str, references: usize, deferred: bool) -> Image {
    let generator = mlx_gen_flux2::provider_registry()
        .expect("FLUX.2 registry")
        .load(provider_id, &spec(deferred))
        .expect("load FLUX.2 Klein edit route");
    let size = 256;
    let conditioning = if references == 1 {
        vec![Conditioning::Reference {
            image: synthetic_reference(size, 0),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference {
            images: (0..references)
                .map(|index| synthetic_reference(size, index))
                .collect(),
        }]
    };
    let mut memory = full_ladder_memory();
    if !deferred {
        memory.stream_transformer_blocks = false;
        memory.transformer_window_size = None;
        memory.transformer_window_component = None;
    }
    let request = GenerationRequest {
        prompt: "make the scene look like a cold winter morning".into(),
        width: size,
        height: size,
        count: 1,
        steps: Some(2),
        seed: Some(1234),
        conditioning,
        memory: Some(memory),
        ..Default::default()
    };
    let GenerationOutput::Images(mut images) = generator
        .generate(&request, &mut |_| {})
        .expect("generate windowed edit")
    else {
        panic!("expected image output");
    };
    drop(generator);
    clear_cache();
    images.pop().expect("one edit image")
}

#[test]
#[ignore = "needs the exact FLUX.2 Klein BF16 cache snapshot and Apple/Metal"]
fn full_shared_ladder_reduces_request_peak_and_preserves_output() {
    assert!(
        snapshot().components().any(|component| {
            component.as_os_str() == mlx_gen_flux2::memory_strategy::KLEIN_CALIBRATED_REVISION
        }),
        "runner must be bound to the calibrated HF revision"
    );
    let raw_latent_edge = (ladder_size() / 8) as i32;
    let plan = mlx_gen::tiling::TilingConfig::spatial_only(
        mlx_gen_flux2::memory_strategy::DECODE_TILE_EDGE as i32,
        mlx_gen_flux2::memory_strategy::DECODE_OVERLAP as i32,
    )
    .plan(
        mlx_gen::tiling::VaeTiling {
            spatial_scale: 8,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: 128,
        },
        1,
        raw_latent_edge,
        raw_latent_edge,
    );
    assert!(
        plan.h.len() > 1 && plan.w.len() > 1,
        "native decode evidence must physically execute multiple tiles"
    );
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
                decode_tile_edge: Some(mlx_gen_flux2::memory_strategy::DECODE_TILE_EDGE),
                decode_overlap: Some(mlx_gen_flux2::memory_strategy::DECODE_OVERLAP),
                ..Default::default()
            },
        ),
        (
            "bounded-attention",
            GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                decode_tile_edge: Some(mlx_gen_flux2::memory_strategy::DECODE_TILE_EDGE),
                decode_overlap: Some(mlx_gen_flux2::memory_strategy::DECODE_OVERLAP),
                attention_chunk_size: Some(mlx_gen_flux2::memory_strategy::ATTENTION_CHUNK_SIZE),
                ..Default::default()
            },
        ),
        ("bounded-transformer", full_ladder_memory()),
    ];
    let mut runs = Vec::new();
    for (name, memory) in arms {
        let run = run(memory);
        println!(
            "ARM name={name} conditioning_gib={:.3} denoise_gib={:.3} decode_gib={:.3} request_gib={:.3} seconds={:.2}",
            gib(run.conditioning), gib(run.denoise), gib(run.decode), gib(run.request_peak()), run.seconds,
        );
        runs.push((name, run));
    }
    let staged = &runs[0].1;
    let tiled = &runs[1].1;
    let (decode_max, decode_mean) = image_delta(&staged.image, &tiled.image);
    assert!(
        decode_mean < 4.0 && decode_max <= 64,
        "multi-tile VAE decode exceeded the calibrated 1.57% mean / 25% max byte envelope: max={decode_max} mean={decode_mean:.6}"
    );
    for (name, run) in runs.iter().skip(2) {
        let (max, mean) = image_delta(&tiled.image, &run.image);
        assert!(
            max <= 1,
            "{name} changed denoise output: max={max} mean={mean:.6}"
        );
    }
    let full = &runs.last().unwrap().1;
    assert!(
        full.request_peak() < staged.request_peak(),
        "full ladder did not reduce request peak: full={:.3} staged={:.3}",
        gib(full.request_peak()),
        gib(staged.request_peak())
    );
    println!(
        "RESULT status=pass provider={} tier=bf16 revision={} fingerprint={} staged_peak_gib={:.3} full_peak_gib={:.3} decode_max_delta={} decode_mean_delta={:.6}",
        mlx_gen_flux2::FLUX2_KLEIN_9B_ID,
        mlx_gen_flux2::memory_strategy::KLEIN_CALIBRATED_REVISION,
        mlx_gen_flux2::memory_strategy::KLEIN_MEMORY_CALIBRATION_FINGERPRINT,
        gib(staged.request_peak()), gib(full.request_peak()),
        decode_max,
        decode_mean,
    );
}

#[test]
#[ignore = "needs the exact FLUX.2 Klein BF16 cache snapshot and Apple/Metal"]
fn full_ladder_covers_single_and_multi_edit_routes() {
    let single = render_edit(mlx_gen_flux2::FLUX2_KLEIN_9B_EDIT_ID, 1, true);
    let multi = render_edit(mlx_gen_flux2::FLUX2_KLEIN_9B_EDIT_ID, 2, true);
    for (name, image) in [("single", single), ("multi", multi)] {
        let mean = image
            .pixels
            .iter()
            .map(|pixel| f64::from(*pixel))
            .sum::<f64>()
            / image.pixels.len() as f64;
        assert!(mean > 2.0 && mean < 253.0, "{name} edit is degenerate");
    }
    println!(
        "RESULT status=pass provider=flux2_klein_9b routes=single-edit,multi-reference steps=2"
    );
}

#[test]
#[ignore = "needs exact FLUX.2 Klein + PiD/Gemma weights and Apple/Metal"]
fn full_ladder_composes_with_the_pid_decode_domain() {
    let input_edge = 768;
    let output_edge = input_edge * 4;
    let pid_edge = mlx_gen_pid::DecodeRoutes::pid_edges()[0];
    assert!(
        output_edge > pid_edge,
        "PiD evidence must physically execute multiple tiles"
    );
    let spec = spec(true).with_pid(
        WeightsSource::File(required_path("FLUX2_KLEIN_LADDER_PID")),
        WeightsSource::Dir(required_path("FLUX2_KLEIN_LADDER_GEMMA")),
    );
    let generator = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load(mlx_gen_flux2::FLUX2_KLEIN_9B_ID, &spec)
        .expect("load exact Klein artifact with PiD");
    let mut memory = full_ladder_memory();
    memory.decode_tile_edge = Some(mlx_gen_pid::DecodeRoutes::pid_edges()[0]);
    memory.decode_overlap = Some(mlx_gen_pid::DecodeRoutes::pid_overlap());
    let request = GenerationRequest {
        prompt: "a mountain valley landscape at golden hour".into(),
        width: input_edge,
        height: input_edge,
        count: 1,
        steps: Some(1),
        seed: Some(7),
        use_pid: true,
        memory: Some(memory),
        ..Default::default()
    };
    let GenerationOutput::Images(images) = generator
        .generate(&request, &mut |_| {})
        .expect("generate with streamed DiT and PiD decode")
    else {
        panic!("expected image output");
    };
    let image = &images[0];
    assert_eq!((image.width, image.height), (output_edge, output_edge));
    let (min, max) = image
        .pixels
        .iter()
        .fold((u8::MAX, u8::MIN), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    assert!(max.saturating_sub(min) > 40, "PiD ladder output is flat");
    println!(
        "RESULT status=pass provider=flux2_klein_9b route=pid decode_edge={} decode_overlap={} transformer_window=1 output={}x{}",
        mlx_gen_pid::DecodeRoutes::pid_edges()[0],
        mlx_gen_pid::DecodeRoutes::pid_overlap(),
        image.width,
        image.height,
    );
}

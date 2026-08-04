//! SC-15517 representative real-Metal proof for the Krea pose-control composition.
//!
//! Both arms keep staged residency and the already-verified 512/64 Qwen-VAE tiled decode. The
//! treatment additionally enables shared query-chunked attention in the seven resident pose blocks
//! and the 28-block base DiT, then materializes the reopenable base DiT one block at a time. This makes
//! the A/B attributable to the two newly wired control mechanisms while preserving the pose overlay.

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec,
    OffloadPolicy, Progress, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn base() -> PathBuf {
    std::env::var("KREA_CONTROL_BASE")
        .map(PathBuf::from)
        .expect("set KREA_CONTROL_BASE to the exact prepacked q4 Krea snapshot directory")
}

fn overlay() -> PathBuf {
    std::env::var("KREA_CONTROL_OVERLAY")
        .map(PathBuf::from)
        .expect("set KREA_CONTROL_OVERLAY to the exact pose-control safetensors file")
}

fn pose() -> Image {
    let size = 512_u32;
    let mut pixels = Vec::with_capacity((size * size * 3) as usize);
    for y in 0..size {
        for x in 0..size {
            pixels.push((x * 255 / size) as u8);
            pixels.push((y * 255 / size) as u8);
            pixels.push(((x + y) * 127 / (2 * size)) as u8);
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
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

fn run(full: bool) -> Run {
    let spec = LoadSpec::new(WeightsSource::Dir(base()))
        .with_control(WeightsSource::File(overlay()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    let generator = mlx_gen_krea::provider_registry()
        .expect("Krea registry")
        .load("krea_2_turbo_control", &spec)
        .expect("load Krea pose control");
    let request = GenerationRequest {
        prompt: "a person standing in a studio, photograph".into(),
        width: 1024,
        height: 1024,
        count: 1,
        steps: Some(1),
        seed: Some(1234),
        conditioning: vec![Conditioning::Control {
            image: pose(),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        memory: Some(GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            chunk_attention: full,
            stream_transformer_blocks: full,
            decode_tile_edge: Some(mlx_gen_krea::memory_strategy::DECODE_TILE_EDGE),
            decode_overlap: Some(mlx_gen_krea::memory_strategy::DECODE_OVERLAP),
            attention_chunk_size: full
                .then_some(mlx_gen_krea::memory_strategy::ATTENTION_CHUNK_SIZE),
            transformer_window_size: full
                .then_some(mlx_gen_krea::memory_strategy::TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: full.then_some(TransformerComponent::Dit),
            ..Default::default()
        }),
        ..Default::default()
    };

    clear_cache();
    reset_peak_memory();
    let mut conditioning = 0;
    let mut denoise = 0;
    let output = generator
        .generate(&request, &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                conditioning = get_peak_memory();
                reset_peak_memory();
            }
            Progress::Decoding if denoise == 0 => {
                denoise = get_peak_memory();
                reset_peak_memory();
            }
            _ => {}
        })
        .expect("generate control image");
    let decode = get_peak_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    Run {
        conditioning,
        denoise,
        decode,
        image,
    }
}

fn image_delta(left: &Image, right: &Image) -> (u8, f64) {
    assert_eq!((left.width, left.height), (right.width, right.height));
    let mut max = 0_u8;
    let mut total = 0_u64;
    for (left, right) in left.pixels.iter().zip(&right.pixels) {
        let delta = left.abs_diff(*right);
        max = max.max(delta);
        total += u64::from(delta);
    }
    (max, total as f64 / left.pixels.len() as f64)
}

#[test]
#[ignore = "needs cached prepacked q4 Krea weights, pose overlay, and Apple/Metal"]
fn pose_control_attention_and_base_window_reduce_peak_without_changing_output() {
    let tiled = run(false);
    let full = run(true);
    let (max_delta, mean_delta) = image_delta(&tiled.image, &full.image);
    println!(
        "CONTROL baseline conditioning={:.3} denoise={:.3} decode={:.3} request={:.3}",
        tiled.conditioning as f64 / GIB,
        tiled.denoise as f64 / GIB,
        tiled.decode as f64 / GIB,
        tiled.peak() as f64 / GIB,
    );
    println!(
        "CONTROL full conditioning={:.3} denoise={:.3} decode={:.3} request={:.3}",
        full.conditioning as f64 / GIB,
        full.denoise as f64 / GIB,
        full.decode as f64 / GIB,
        full.peak() as f64 / GIB,
    );
    assert!(
        max_delta <= 1,
        "control attention/window changed output: max {max_delta}, mean {mean_delta:.6}"
    );
    assert!(
        full.peak() < tiled.peak(),
        "control full ladder did not reduce request peak"
    );
    println!(
        "RESULT status=pass provider=krea_2_turbo_control fingerprint={} baseline_peak_gib={:.3} full_peak_gib={:.3} max_channel_delta={} mean_channel_delta={:.6}",
        mlx_gen_krea::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        tiled.peak() as f64 / GIB,
        full.peak() as f64 / GIB,
        max_delta,
        mean_delta,
    );
}

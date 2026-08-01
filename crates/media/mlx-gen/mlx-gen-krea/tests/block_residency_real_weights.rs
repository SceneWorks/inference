//! SC-16352 real-weight Krea 2 MLX request-peak attribution.
//!
//! The control is the same Sequential + deferred load with a resident 28-block forward. The treatment
//! changes only `GenerationMemory::stream_transformer_blocks`, so a lower request peak is attributable
//! to block residency rather than phase staging or loader shape. `KREA_RUNG4_TIER=q4|q8|bf16` selects
//! the exact turnkey directory. `KREA_RUNG4_ADAPTER=/path/to/lora.safetensors` additionally proves
//! that per-block low-rank replay preserves output.
//!
//! ```text
//! KREA_RUNG4_SNAPSHOT=/path/to/krea-2-turbo-mlx/snapshot \
//! KREA_RUNG4_TIER=q4 \
//! cargo test -p mlx-gen-krea --release --test block_residency_real_weights -- --ignored --nocapture
//! ```

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec,
    OffloadPolicy, Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("KREA_RUNG4_SNAPSHOT")
            .expect("set KREA_RUNG4_SNAPSHOT to the explicit Krea turnkey snapshot root"),
    )
}

fn tier() -> (&'static str, Option<Quant>) {
    match std::env::var("KREA_RUNG4_TIER").as_deref() {
        Ok("q8") => ("q8", Some(Quant::Q8)),
        Ok("bf16") => ("bf16", None),
        _ => ("q4", Some(Quant::Q4)),
    }
}

fn spec() -> LoadSpec {
    let (tier_dir, quant) = tier();
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot().join(tier_dir)))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if let Ok(path) = std::env::var("KREA_RUNG4_ADAPTER") {
        spec.adapters = vec![AdapterSpec::new(path.into(), 1.0, AdapterKind::Lora)];
    }
    spec
}

fn request(stream: bool) -> GenerationRequest {
    let size = std::env::var("KREA_RUNG4_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let steps = std::env::var("KREA_RUNG4_STEPS")
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
        memory: Some(GenerationMemory {
            stage_residency: true,
            stream_transformer_blocks: stream,
            transformer_window_size: stream.then_some(1),
            transformer_window_component: stream.then_some(TransformerComponent::Dit),
            ..Default::default()
        }),
        ..Default::default()
    }
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

fn run(stream: bool) -> Run {
    let generator = mlx_gen_krea::provider_registry()
        .expect("Krea registry")
        .load("krea_2_turbo", &spec())
        .expect("load krea_2_turbo");
    let req = request(stream);
    clear_cache();
    reset_peak_memory();
    let start = std::time::Instant::now();
    let mut conditioning = 0usize;
    let mut denoise = 0usize;
    let mut on_progress = |progress| match progress {
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
        .generate(&req, &mut on_progress)
        .expect("generate Krea image");
    let decode = get_peak_memory();
    let seconds = start.elapsed().as_secs_f64();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
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

fn image_delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0_u8;
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

#[test]
#[ignore = "needs real Krea weights and Apple/Metal"]
fn window_one_reduces_the_full_request_peak_against_the_resident_attribution_control() {
    let baseline = run(false);
    let windowed = run(true);
    let (max_delta, mean_delta) = image_delta(&baseline.image, &windowed.image);
    println!(
        "\nKrea 2 MLX rung-4 request-peak attribution ({})",
        tier().0
    );
    println!("                conditioning   denoise   decode   REQUEST   seconds");
    println!(
        "resident control {:>10.3} {:>9.3} {:>8.3} {:>9.3} {:>9.2}",
        gib(baseline.conditioning),
        gib(baseline.denoise),
        gib(baseline.decode),
        gib(baseline.request_peak()),
        baseline.seconds
    );
    println!(
        "window=1        {:>10.3} {:>9.3} {:>8.3} {:>9.3} {:>9.2}",
        gib(windowed.conditioning),
        gib(windowed.denoise),
        gib(windowed.decode),
        gib(windowed.request_peak()),
        windowed.seconds
    );
    println!("image delta max={max_delta}, mean={mean_delta:.6}");
    assert!(
        windowed.request_peak() < baseline.request_peak(),
        "rung 4 must reduce REQUEST peak: window {:.3} GiB, control {:.3} GiB",
        gib(windowed.request_peak()),
        gib(baseline.request_peak())
    );
    assert!(
        max_delta <= 1,
        "windowed output changed: max channel delta {max_delta}, mean {mean_delta:.6}"
    );
}

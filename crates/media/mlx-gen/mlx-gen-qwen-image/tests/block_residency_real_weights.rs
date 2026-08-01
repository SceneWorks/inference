//! SC-16353 — real-weight request-peak measurement for Qwen-Image MLX rung 4.
//!
//! This is deliberately one serialized test: MLX peak counters are process-global and two model
//! loads must never overlap. It records the previous staged-residency composition, then compares a
//! same-stream unbounded 60-block attribution control with genuinely bounded 8/4/2/1 cadences.
//!
//! ```text
//! QWEN_IMAGE_SNAPSHOT=<explicit bf16/q8/q4 tier dir> QWEN_RUNG4_TIER=q8 \
//!   cargo test -p mlx-gen-qwen-image --release --test block_residency_real_weights \
//!   -- --ignored --nocapture
//! ```

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec, OffloadPolicy, Progress,
    Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const BLOCKS: u32 = 60;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("QWEN_IMAGE_SNAPSHOT").expect(
        "set QWEN_IMAGE_SNAPSHOT to an explicit Qwen-Image tier directory; the test never fetches",
    ))
}

fn tier() -> Option<Quant> {
    match std::env::var("QWEN_RUNG4_TIER").as_deref() {
        Ok("bf16") => None,
        Ok("q4") => Some(Quant::Q4),
        Ok("q8") => Some(Quant::Q8),
        Ok(other) => panic!("QWEN_RUNG4_TIER must be bf16, q4, or q8; got {other}"),
        Err(_) => Some(Quant::Q8),
    }
}

fn tier_label() -> &'static str {
    match tier() {
        None => "bf16",
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        Some(other) => panic!("unsupported Qwen rung-4 tier {other:?}"),
    }
}

fn spec(policy: OffloadPolicy, shape: LoadShape) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()))
        .with_offload_policy(policy)
        .with_load_shape(shape);
    if let Some(quant) = tier() {
        spec = spec.with_quant(quant);
    }
    spec
}

fn request(memory: Option<GenerationMemory>) -> GenerationRequest {
    let size = env_u32("QWEN_RUNG4_SIZE", 1024);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: size,
        height: size,
        count: 1,
        seed: Some(16_353),
        steps: Some(env_u32("QWEN_RUNG4_STEPS", 2)),
        memory,
        ..Default::default()
    }
}

fn staged_memory() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..Default::default()
    }
}

fn window_memory(window: u32) -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        stream_transformer_blocks: true,
        transformer_window_size: Some(window),
        transformer_window_component: Some(TransformerComponent::Dit),
        ..Default::default()
    }
}

struct Run {
    arm: String,
    conditioning_bytes: usize,
    denoise_bytes: usize,
    decode_bytes: usize,
    request_bytes: usize,
    elapsed_secs: f64,
    image: Image,
}

fn run(arm: impl Into<String>, spec: LoadSpec, memory: Option<GenerationMemory>) -> Run {
    let arm = arm.into();
    let generator = mlx_gen_qwen_image::provider_registry()
        .expect("registry")
        .load("qwen_image", &spec)
        .unwrap_or_else(|error| panic!("load {arm}: {error:?}"));
    let req = request(memory);

    clear_cache();
    reset_peak_memory();
    let started = std::time::Instant::now();
    let mut conditioning_bytes = 0;
    let mut denoise_bytes = 0;
    let mut progress = |event: Progress| match event {
        Progress::Step { current: 1, .. } if conditioning_bytes == 0 => {
            conditioning_bytes = get_peak_memory();
            reset_peak_memory();
        }
        Progress::Decoding if denoise_bytes == 0 => {
            denoise_bytes = get_peak_memory();
            reset_peak_memory();
        }
        _ => {}
    };
    let output = generator
        .generate(&req, &mut progress)
        .unwrap_or_else(|error| panic!("generate {arm}: {error:?}"));
    let decode_bytes = get_peak_memory();
    let elapsed_secs = started.elapsed().as_secs_f64();
    let request_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);
    let image = match output {
        GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            images.pop().unwrap()
        }
        other => panic!("expected image output, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    assert!(conditioning_bytes > 0 && denoise_bytes > 0 && decode_bytes > 0);
    println!(
        "RESULT {{\"story\":16353,\"tier\":\"{}\",\"arm\":\"{}\",\"size\":{},\"steps\":{},\"conditioning_bytes\":{},\"denoise_bytes\":{},\"decode_bytes\":{},\"request_peak_bytes\":{},\"elapsed_secs\":{:.3}}}",
        tier_label(),
        arm,
        req.width,
        req.steps.unwrap(),
        conditioning_bytes,
        denoise_bytes,
        decode_bytes,
        request_bytes,
        elapsed_secs,
    );
    Run {
        arm,
        conditioning_bytes,
        denoise_bytes,
        decode_bytes,
        request_bytes,
        elapsed_secs,
        image,
    }
}

fn image_delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0;
    let mut sum = 0_u64;
    for (left, right) in a.pixels.iter().zip(&b.pixels) {
        let delta = left.abs_diff(*right);
        max = max.max(delta);
        sum += u64::from(delta);
    }
    (max, sum as f64 / a.pixels.len() as f64)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

#[test]
#[ignore = "needs an explicit real Qwen-Image tier + Apple/Metal GPU"]
fn bounded_window_is_distinct_from_the_unbounded_stream_control() {
    assert!(
        request(None).steps.unwrap() >= 2,
        "the production Qwen sampler needs at least two steps"
    );

    // Whole-model control: no staged component release and no block stream. This attributes the
    // broad effect of the staged/deferred load shape, but not the block-window lever by itself.
    let resident = run(
        "fully-resident-control",
        spec(OffloadPolicy::Resident, LoadShape::EagerMaterialization),
        None,
    );
    // Qwen's currently available pre-rung-4 composition. Rungs 2/3 are formally Missing on this
    // provider, so this arm is intentionally not mislabeled as the story's required rung-3 baseline.
    let previous = run(
        "prior-available-composition",
        spec(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
        Some(staged_memory()),
    );

    // Same stream/load-shape control with one all-covering range. `BlockPlan` explicitly defines
    // this as unbounded; it must never be eligible as the published rung-4 domain.
    let unbounded_stream = run(
        "deferred-unbounded-window-60-control",
        spec(
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        ),
        Some(window_memory(BLOCKS)),
    );
    let mut windows = Vec::new();
    for window in [8, 4, 2, 1] {
        windows.push(run(
            format!("rung4-window-{window}"),
            spec(
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            ),
            Some(window_memory(window)),
        ));
    }
    let published_window = mlx_gen_qwen_image::memory_strategy::TRANSFORMER_WINDOW_SIZE;
    assert!(
        mlx_gen::block_residency::BlockPlan::new(BLOCKS as usize, published_window as usize)
            .unwrap()
            .is_bounded(),
        "the published rung-4 window must actually bound the stack"
    );
    let declared = windows
        .iter()
        .find(|run| run.arm == format!("rung4-window-{published_window}"))
        .expect("published window arm");

    let (resident_max, resident_mean) = image_delta(&resident.image, &previous.image);
    assert!(
        resident_max <= 1,
        "staged attribution baseline changed the resident image (max {resident_max}/255, mean {resident_mean:.5})"
    );

    for arm in &windows {
        let (max, mean) = image_delta(&arm.image, &previous.image);
        assert!(
            max <= 1,
            "{} changed the rendered image (max {max}/255, mean {mean:.5})",
            arm.arm
        );
    }
    let (unbounded_max, unbounded_mean) = image_delta(&unbounded_stream.image, &previous.image);
    assert!(
        unbounded_max <= 1,
        "same-stream unbounded control changed the image (max {unbounded_max}/255, mean {unbounded_mean:.5})"
    );
    assert!(
        resident.request_bytes >= previous.request_bytes,
        "fully-resident control {:.3} GiB unexpectedly fell below staged {:.3} GiB",
        gib(resident.request_bytes),
        gib(previous.request_bytes),
    );
    assert!(
        declared.denoise_bytes < (unbounded_stream.denoise_bytes as f64 * 0.95) as usize,
        "published bounded window did not materially lower denoise peak versus the same-stream unbounded control: {:.3} -> {:.3} GiB",
        gib(unbounded_stream.denoise_bytes),
        gib(declared.denoise_bytes),
    );

    // The card's non-negotiable: a phase bound is not a request saving. Compare the published arm
    // directly with the same-stream unbounded control inside the committed 1%/64-MiB counter band.
    // Qwen is conditioning-bound at 1024², so this is expected to record `does-not-move`.
    let request_floor = windows
        .iter()
        .map(|run| run.request_bytes)
        .chain(std::iter::once(unbounded_stream.request_bytes))
        .min()
        .expect("stream rows");
    let request_slack = ((request_floor as f64 * 0.01) as usize).max(64 * 1024 * 1024);
    assert!(
        declared.request_bytes <= unbounded_stream.request_bytes.saturating_add(request_slack)
            && unbounded_stream.request_bytes <= declared.request_bytes.saturating_add(request_slack),
        "published bounded window unexpectedly moved the request peak outside the committed noise band: {:.3} vs {:.3} GiB",
        gib(unbounded_stream.request_bytes),
        gib(declared.request_bytes),
    );
    let lowest_denoise = windows
        .iter()
        .min_by_key(|run| run.denoise_bytes)
        .expect("bounded rows");
    assert_eq!(
        lowest_denoise.arm,
        format!("rung4-window-{published_window}"),
        "the published singleton must be the measured denoise-floor cadence"
    );
    println!(
        "REQUEST_VERDICT {{\"story\":16353,\"tier\":\"{}\",\"finding\":\"does-not-move\",\"unbounded_stream_bytes\":{},\"bounded_bytes\":{},\"request_slack_bytes\":{}}}",
        tier_label(),
        unbounded_stream.request_bytes,
        declared.request_bytes,
        request_slack,
    );

    // The cadence sweep still reports every genuinely bounded point, but no timing heuristic can
    // promote a noisier >1 value merely because it is faster.
    let candidate = windows
        .iter()
        .find(|run| run.arm == format!("rung4-window-{published_window}"))
        .expect("published bounded candidate");
    let request_floor = windows
        .iter()
        .map(|run| run.request_bytes)
        .min()
        .expect("window rows");
    let request_slack = ((request_floor as f64 * 0.01) as usize).max(64 * 1024 * 1024);
    println!(
        "CANDIDATE {{\"story\":16353,\"tier\":\"{}\",\"arm\":\"{}\",\"request_floor_bytes\":{},\"request_slack_bytes\":{},\"reason\":\"only cadence that materially lowers denoise versus the same-stream unbounded control; request peak does not move\"}}",
        tier_label(),
        candidate.arm,
        request_floor,
        request_slack,
    );

    println!(
        "SUMMARY tier={} resident={:.3}GiB prior_available={:.3}GiB unbounded_stream={:.3}GiB bounded_w{}={:.3}GiB (conditioning={:.3}, denoise={:.3}, decode={:.3}) prior_to_bounded_delta={:.3}GiB ({:.1}%) candidate={} elapsed={:.1}s",
        tier_label(),
        gib(resident.request_bytes),
        gib(previous.request_bytes),
        gib(unbounded_stream.request_bytes),
        published_window,
        gib(declared.request_bytes),
        gib(declared.conditioning_bytes),
        gib(declared.denoise_bytes),
        gib(declared.decode_bytes),
        gib(previous.request_bytes.saturating_sub(declared.request_bytes)),
        100.0 * previous.request_bytes.saturating_sub(declared.request_bytes) as f64
            / previous.request_bytes as f64,
        candidate.arm,
        resident.elapsed_secs + previous.elapsed_secs + unbounded_stream.elapsed_secs + windows.iter().map(|run| run.elapsed_secs).sum::<f64>(),
    );
}

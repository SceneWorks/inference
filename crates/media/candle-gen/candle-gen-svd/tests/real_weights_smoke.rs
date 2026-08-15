#![cfg(feature = "cuda")]

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    CancelFlag, Conditioning, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadPhase, LoadSpec, OffloadPolicy, Progress, WeightsSource,
};
use candle_gen::testkit::{cuda_mempool_used_high_bytes, reset_cuda_mempool_high_water};

const CUDA_DEVICE: i32 = 0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn source_image(width: u32, height: u32) -> Image {
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = (y * width as usize + x) * 3;
            pixels[i] = (x * 255 / width as usize) as u8;
            pixels[i + 1] = (y * 255 / height as usize) as u8;
            pixels[i + 2] = ((x + y) % 251) as u8;
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn request(width: u32, height: u32, frames: u32, chunk: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        width,
        height,
        frames: Some(frames),
        steps: Some(steps),
        fps: Some(7),
        conditioning_fps: Some(7),
        decode_chunk_size: Some(chunk),
        seed: Some(42),
        conditioning: vec![Conditioning::Reference {
            image: source_image(width, height),
            strength: None,
        }],
        ..Default::default()
    }
}

fn load_sequential() -> Box<dyn Generator> {
    let snapshot = PathBuf::from(
        std::env::var("SVD_SNAPSHOT").expect("set SVD_SNAPSHOT to the immutable SVD-XT snapshot"),
    );
    let spec =
        LoadSpec::new(WeightsSource::Dir(snapshot)).with_offload_policy(OffloadPolicy::Sequential);
    let generator = candle_gen_svd::provider_registry()
        .expect("svd registry")
        .load("svd_xt", &spec)
        .expect("load sequential svd_xt");
    assert!(
        generator
            .descriptor()
            .capabilities
            .supports_sequential_offload
    );
    generator
}

#[derive(Debug, Default)]
struct StagePeaks {
    conditioning: u64,
    unet: u64,
    decode: u64,
}

#[derive(Debug)]
struct ProfileOutput {
    frames: Vec<Image>,
    fps: u32,
    peaks: StagePeaks,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

fn assert_frames(frames: &[Image], expected: u32, width: u32, height: u32) {
    assert_eq!(frames.len(), expected as usize);
    assert!(frames
        .iter()
        .all(|frame| (frame.width, frame.height) == (width, height)));
    assert!(
        frames
            .iter()
            .all(|frame| frame.pixels.iter().min() != frame.pixels.iter().max()),
        "every decoded frame must be non-constant"
    );
}

/// Run one complete profile and split the driver's continuous live-allocation high-water at the
/// sequential component boundaries. `Progress::Loading(Renderer)` occurs twice for SVD: first when
/// conditioning has synchronized/dropped and the UNet starts loading, then when the UNet has
/// synchronized/dropped and the decode VAE starts loading.
fn run_profile(
    label: &str,
    generator: &dyn Generator,
    request: &GenerationRequest,
) -> ProfileOutput {
    assert!(
        reset_cuda_mempool_high_water(CUDA_DEVICE),
        "CUDA default-pool USED_MEM_HIGH reset must work for credible isolated stage metrics"
    );
    let mut peaks = StagePeaks::default();
    let mut renderer_loads = 0usize;
    let started = Instant::now();
    let output = generator
        .generate(request, &mut |progress| {
            let live_high = cuda_mempool_used_high_bytes(CUDA_DEVICE).unwrap_or(0);
            eprintln!(
                "[[SVD_PROGRESS]] {{\"profile\":\"{label}\",\"progress\":\"{progress:?}\",\
                 \"liveMemHighGib\":{:.3}}}",
                gib(live_high)
            );
            if progress == Progress::Loading(LoadPhase::Renderer) {
                let completed = live_high;
                match renderer_loads {
                    0 => peaks.conditioning = completed,
                    1 => peaks.unet = completed,
                    _ => panic!("svd_xt emitted more than two Renderer load boundaries"),
                }
                renderer_loads += 1;
                assert!(
                    reset_cuda_mempool_high_water(CUDA_DEVICE),
                    "stage high-water reset must succeed"
                );
            }
        })
        .unwrap_or_else(|error| panic!("{label} generation failed: {error}"));
    peaks.decode = cuda_mempool_used_high_bytes(CUDA_DEVICE).unwrap_or(0);
    let wall = started.elapsed();
    assert_eq!(
        renderer_loads, 2,
        "sequential SVD must expose conditioner → UNet → VAE boundaries"
    );
    assert!(
        peaks.conditioning > 0 && peaks.unet > 0 && peaks.decode > 0,
        "all stage USED_MEM_HIGH values must be nonzero: {peaks:?}"
    );

    let GenerationOutput::Video { frames, fps, audio } = output else {
        panic!("svd_xt must return video");
    };
    assert!(audio.is_none());
    let expected_frames = request.frames.unwrap();
    assert_frames(&frames, expected_frames, request.width, request.height);
    println!(
        "[[SVD_32GB]] {{\"profile\":\"{label}\",\"width\":{},\"height\":{},\"frames\":{},\
         \"decodeChunk\":{},\"steps\":{},\"fps\":{},\"wallSeconds\":{:.3},\
         \"conditioningMemHighGib\":{:.3},\"unetMemHighGib\":{:.3},\
         \"decodeMemHighGib\":{:.3},\"overallMemHighGib\":{:.3}}}",
        request.width,
        request.height,
        expected_frames,
        request.decode_chunk_size.unwrap(),
        request.steps.unwrap(),
        fps,
        wall.as_secs_f64(),
        gib(peaks.conditioning),
        gib(peaks.unet),
        gib(peaks.decode),
        gib(peaks.conditioning.max(peaks.unet).max(peaks.decode)),
    );
    ProfileOutput { frames, fps, peaks }
}

/// Mandatory sc-14625 target-hardware gate. Run only on a real 32 GB CUDA card:
///
/// `cargo test --release --features cuda -p candle-gen-svd --test real_weights_smoke
/// default_25_frame_profile_then_8_frame_control -- --ignored --nocapture --test-threads=1`
///
/// It records the exact workload and per-stage `CU_MEMPOOL_ATTR_USED_MEM_HIGH`, then reruns the
/// proven 8-frame control twice. The repeated same-seed output witnesses determinism, second-job
/// health, and that resetting the high-water prevents stale peak telemetry.
#[test]
#[ignore = "requires SVD_SNAPSHOT, real weights, and a physical 32 GB CUDA device"]
fn default_25_frame_profile_then_8_frame_control() {
    let generator = load_sequential();
    let default = request(1024, 576, 25, 8, 25);
    let default_out = run_profile("default", generator.as_ref(), &default);
    assert_eq!(default_out.fps, 7);

    let control = request(1024, 576, 8, 1, 12);
    let control_a = run_profile("control-a", generator.as_ref(), &control);
    let control_b = run_profile("control-b", generator.as_ref(), &control);
    assert_eq!(control_a.fps, control_b.fps);
    assert_eq!(control_a.frames.len(), control_b.frames.len());
    for (a, b) in control_a.frames.iter().zip(&control_b.frames) {
        assert_eq!(
            a.pixels, b.pixels,
            "same-seed control rerun must be byte deterministic"
        );
    }
    assert!(
        control_b.peaks.unet <= default_out.peaks.unet,
        "a reset 8-frame control cannot inherit the larger default run's UNet high-water"
    );
    // sc-19556: `default_out.wall > Duration::ZERO && control_b.wall > Duration::ZERO` was DELETED
    // here rather than replaced. It could only fail on a non-monotonic clock, and everything it
    // reached for — that both profiles genuinely ran every stage — is already asserted inside
    // `run_profile`, for every profile it is called on and more strictly than a duration could:
    // `renderer_loads == 2`, `peaks.conditioning > 0 && peaks.unet > 0 && peaks.decode > 0`, and
    // `assert_frames(&frames, expected_frames, ..)`. Deleting it left `ProfileOutput.wall` with no
    // reader at all, which is `-D dead-code` on the CUDA lane, so the FIELD is gone too. The
    // `[[SVD_32GB]]` line still reports `wallSeconds`: it prints `run_profile`'s local `wall`, and
    // never read the struct field.
}

/// Error and mid-denoise cancellation must both leave the same generator healthy for a later job.
/// The shared three-stage seam synchronizes before every component drop on these paths; this
/// real-weight test witnesses the CUDA allocator/stream behavior rather than only the mock ordering.
#[test]
#[ignore = "requires SVD_SNAPSHOT, real weights, and a CUDA device"]
fn failure_and_cancel_are_followed_by_a_healthy_second_job() {
    let generator = load_sequential();

    let valid = request(256, 256, 2, 1, 2);
    {
        let _impossible_budget = EnvRestore::set("SVD_VAE_BUDGET_GIB", "0.000001");
        assert!(
            generator.generate(&valid, &mut |_| {}).is_err(),
            "an impossible positive decode budget must fail after the VAE loads"
        );
    }

    let mut canceled = request(256, 256, 2, 1, 3);
    let cancel = CancelFlag::new();
    canceled.cancel = cancel.clone();
    let canceled_result = generator.generate(&canceled, &mut |progress| {
        if matches!(progress, Progress::Step { current: 1, .. }) {
            cancel.cancel();
        }
    });
    assert!(matches!(canceled_result, Err(Error::Canceled)));

    let recovery = request(256, 256, 2, 1, 2);
    let recovered = run_profile("post-failure-cancel", generator.as_ref(), &recovery);
    assert_eq!(recovered.frames.len(), 2);
}

struct EnvRestore {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn frame_psnr(a: &[Image], b: &[Image]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut squared = 0f64;
    let mut count = 0usize;
    for (a, b) in a.iter().zip(b) {
        assert_eq!((a.width, a.height), (b.width, b.height));
        assert_eq!(a.pixels.len(), b.pixels.len());
        for (&a, &b) in a.pixels.iter().zip(&b.pixels) {
            let delta = a as f64 - b as f64;
            squared += delta * delta;
            count += 1;
        }
    }
    let mse = squared / count as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Real-weight tiled-vs-monolithic correctness gate. A 6 GiB injected decode budget forces a spatial
/// tile below the 512² output while 1000 GiB selects monolithic. This catches attention/overlap seams
/// that a constant-field synthetic stitch test cannot see.
#[test]
#[ignore = "requires SVD_SNAPSHOT, real weights, and a CUDA device"]
fn forced_spatial_tiling_matches_monolithic_with_acceptable_psnr() {
    let generator = load_sequential();
    let request = request(512, 512, 2, 1, 2);
    let budget = EnvRestore::set("SVD_VAE_BUDGET_GIB", "1000");
    let monolithic = run_profile("parity-monolithic", generator.as_ref(), &request);
    std::env::set_var("SVD_VAE_BUDGET_GIB", "6");
    let tiled = run_profile("parity-tiled", generator.as_ref(), &request);
    drop(budget);

    let psnr = frame_psnr(&monolithic.frames, &tiled.frames);
    println!("[[SVD_TILE_PARITY]] {{\"psnrDb\":{psnr:.3}}}");
    assert!(
        psnr >= 20.0,
        "forced spatial tiling must remain visually faithful to monolithic decode; PSNR {psnr:.2} dB"
    );
}

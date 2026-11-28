#![cfg(feature = "cuda")]

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_gen::candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_gen::candle_core::Tensor, ()>::Err(()));
};

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    CancelFlag, Conditioning, Error, FamilyExtensions, GenerationOutput, GenerationRequest,
    Generator, Image, LoadPhase, LoadSpec, OffloadPolicy, Progress, SvdExtension, WeightsSource,
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
        seed: Some(42),
        conditioning: vec![Conditioning::Reference {
            image: source_image(width, height),
            strength: None,
        }],
        extensions: FamilyExtensions {
            svd: Some(SvdExtension {
                conditioning_fps: Some(7),
                decode_chunk_size: Some(chunk),
                ..Default::default()
            }),
            ..Default::default()
        },
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
        request
            .extensions
            .svd
            .as_ref()
            .and_then(|s| s.decode_chunk_size)
            .unwrap(),
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED default_25_frame_profile_then_8_frame_control");
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

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED failure_and_cancel_are_followed_by_a_healthy_second_job"
    );
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED forced_spatial_tiling_matches_monolithic_with_acceptable_psnr");
}

/// Resets the SW-6132 job-level lossy-attention switch even when an assertion unwinds.
struct LossyAttentionJob;

impl LossyAttentionJob {
    fn set(opt_ins: candle_gen::fused_attention::LossyAttentionOptIns) -> Self {
        candle_gen::fused_attention::set_lossy_attention(opt_ins).expect("a valid selection");
        Self
    }
}

impl Drop for LossyAttentionJob {
    fn drop(&mut self) {
        candle_gen::fused_attention::set_lossy_attention(
            candle_gen::fused_attention::LossyAttentionOptIns::OFF,
        )
        .unwrap();
    }
}

/// SW-6132 (SW-F287 S7): the shared, job-level lossy-attention opt-in on an **f32** video model.
/// SVD's UNet runs f32 (see `dense_dtype`) with 64-wide heads; the shared dispatch runs every
/// eligible call at bf16 (counted `lossy_adapted.f32_cast`) and, for Sol-Attn, zero-pads the heads
/// to 128 (`lossy_adapted.head_pad`), so the opt-in serves this model instead of refusing it. A
/// fixed-seed 8-frame 512² clip (10 steps) with the switch off, `int8`, `sol` at top-k 0.25 and a
/// top-k 1.0 Sol-Attn control (every block routed): every call is
/// offered and counted (served + refused = all calls), INT8 serves every call (none is masked,
/// every head is 64 wide), Sol-Attn serves the self-attention and refuses the cross-attention, each
/// opt-in clip stays non-degenerate and its PSNR against the off clip is printed (recorded, not
/// gated). `SVD_LOSSY_FRAMES_DIR=<dir>` saves the first and last frame of each clip as PNG for
/// review.
///
/// `SVD_SNAPSHOT=<snapshot> cargo test --release --features cuda,model-inference-proof
/// -p candle-gen-svd --test real_weights_smoke svd_xt_job_level_lossy_attention_smoke -- --ignored
/// --nocapture --test-threads=1`
#[test]
#[ignore = "requires SVD_SNAPSHOT, real weights, and a CUDA device"]
fn svd_xt_job_level_lossy_attention_smoke() {
    use candle_gen::fused_attention::{
        attention_path_counts, lossy_attention, Int8AttentionConfig, LossyAttentionOptIns,
        SolConfig,
    };
    let generator = load_sequential();
    let clip = request(512, 512, 8, 1, 10);
    let run = |label: &str, opt_ins: LossyAttentionOptIns| {
        let job = LossyAttentionJob::set(opt_ins);
        let before = attention_path_counts();
        let out = run_profile(label, generator.as_ref(), &clip);
        let counts = attention_path_counts().since(&before);
        drop(job);
        assert_eq!(
            lossy_attention(),
            LossyAttentionOptIns::OFF,
            "reset at job end"
        );
        println!("[[SVD_LOSSY_ATTENTION]] {label}: {counts:?}");
        (out, counts)
    };
    let (off, off_counts) = run("off", LossyAttentionOptIns::OFF);
    assert_eq!((off_counts.int8, off_counts.sol), (0, 0), "{off_counts:?}");
    assert_eq!(off_counts.int8_refusals.total(), 0, "{off_counts:?}");
    assert_eq!(off_counts.sol_refusals.total(), 0, "{off_counts:?}");
    assert!(off_counts.total() > 0, "{off_counts:?}");

    let save = |label: &str, frames: &[Image]| {
        let Some(dir) = std::env::var_os("SVD_LOSSY_FRAMES_DIR") else {
            return;
        };
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("frames dir");
        for (i, frame) in [
            (0, &frames[0]),
            (frames.len() - 1, &frames[frames.len() - 1]),
        ] {
            image::RgbImage::from_raw(frame.width, frame.height, frame.pixels.clone())
                .expect("RGB frame")
                .save(dir.join(format!("svd_{label}_{i:02}.png")))
                .expect("save frame");
        }
    };
    // Non-degenerate: every frame has real spatial variation (not a flat / NaN-black frame).
    let varied = |frames: &[Image]| {
        frames.iter().all(|f| {
            let (lo, hi) = f
                .pixels
                .iter()
                .fold((u8::MAX, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
            hi - lo > 32
        })
    };
    save("off", &off.frames);
    assert!(varied(&off.frames), "the off clip is non-degenerate");

    let (int8, int8_counts) = run(
        "job-int8",
        LossyAttentionOptIns {
            int8: Some(Int8AttentionConfig::default()),
            sol: None,
        },
    );
    assert_eq!(int8_counts.sol, 0, "{int8_counts:?}");
    assert_eq!(
        int8_counts.int8 + int8_counts.int8_refusals.total(),
        int8_counts.total(),
        "every call is offered to INT8 and counted: {int8_counts:?}"
    );
    assert_eq!(
        int8_counts.int8,
        int8_counts.total(),
        "unmasked 64-wide f32 calls: INT8 serves every one at bf16: {int8_counts:?}"
    );
    assert_eq!(
        int8_counts.lossy_adapted.f32_cast, int8_counts.int8,
        "every INT8 call is an f32 call cast to bf16: {int8_counts:?}"
    );
    assert_eq!(int8_counts.lossy_adapted.head_pad, 0, "{int8_counts:?}");
    save("int8", &int8.frames);
    assert!(varied(&int8.frames), "the INT8 clip is non-degenerate");
    println!(
        "[[SVD_LOSSY_ATTENTION]] job int8: {} INT8 calls of {} (f32 cast {}), refusals {:?}, \
         PSNR vs off {:.2} dB",
        int8_counts.int8,
        int8_counts.total(),
        int8_counts.lossy_adapted.f32_cast,
        int8_counts.int8_refusals,
        frame_psnr(&int8.frames, &off.frames)
    );

    // Sol-Attn at the product's sparse budget, and at top-k 1.0 (every key block routed: dense
    // attention up to the kernel's INT8 arithmetic) — the control that separates the f32 cast /
    // head pad adaptation (top-k 1.0 tracks the off clip like INT8 does) from the sparsity loss.
    for (label, ratio) in [("sol", 0.25f32), ("sol-dense", 1.0)] {
        let (sol, sol_counts) = run(
            label,
            LossyAttentionOptIns {
                int8: None,
                sol: Some(SolConfig::top_k(ratio)),
            },
        );
        assert!(
            sol_counts.sol > 0,
            "Sol-Attn serves the f32 64-wide self-attention: {sol_counts:?}"
        );
        assert_eq!(
            sol_counts.sol + sol_counts.sol_refusals.total(),
            sol_counts.total(),
            "every call is offered to Sol-Attn and counted: {sol_counts:?}"
        );
        assert_eq!(
            sol_counts.sol_refusals.total(),
            sol_counts.sol_refusals.cross_attention,
            "only the cross-attention is refused: {sol_counts:?}"
        );
        assert_eq!(
            (
                sol_counts.lossy_adapted.f32_cast,
                sol_counts.lossy_adapted.head_pad
            ),
            (sol_counts.sol, sol_counts.sol),
            "every Sol-Attn call is cast to bf16 and padded to 128: {sol_counts:?}"
        );
        save(label, &sol.frames);
        assert!(varied(&sol.frames), "the {label} clip is non-degenerate");
        println!(
            "[[SVD_LOSSY_ATTENTION]] job {label} (top-k {ratio}): {} Sol-Attn calls of {} \
             (f32 cast {}, head pad {}), refusals {:?}, PSNR vs off {:.2} dB",
            sol_counts.sol,
            sol_counts.total(),
            sol_counts.lossy_adapted.f32_cast,
            sol_counts.lossy_adapted.head_pad,
            sol_counts.sol_refusals,
            frame_psnr(&sol.frames, &off.frames)
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED svd_xt_job_level_lossy_attention_smoke");
}

//! sc-7845 e2e: the **integrated** PiD decode path for Krea 2 Turbo — load the real turnkey snapshot
//! with a PiD decoder overlay (`LoadSpec::with_pid`) and run `Generator::generate` once for the VAE
//! baseline and once with `use_pid`, proving Krea's own `decode_latents` hook routes the live
//! denoised latent through the shared `LatentDecoder` seam into a 4× super-resolved PiD image. Krea
//! reuses the Qwen-Image `QwenVae`, so it shares the `qwenimage` PiD student validated in sc-7843.
//!
//! `#[ignore]`d — needs the Krea Turbo snapshot (env `KREA_TURBO_DIR`, else the published
//! `SceneWorks/krea-2-turbo-mlx` `q8` root in the HF cache), the converted PiD checkpoint (env
//! `PID_QWEN_SAFETENSORS`, else `tools/golden/pid/qwenimage_2kto4k.safetensors`), and a
//! `gemma-2-2b-it` snapshot dir (env `PID_GEMMA_DIR`, else the HF cache).
//!
//! ```sh
//! cargo test -p mlx-gen-krea --release --test integration pid_decode_real_weights:: -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec, OffloadPolicy, Progress,
    WeightsSource,
};
use mlx_gen_krea::load;
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn krea_snapshot() -> PathBuf {
    env_path("KREA_TURBO_DIR").unwrap_or_else(|| panic!("set KREA_TURBO_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"))
}

fn pid_checkpoint() -> PathBuf {
    env_path("PID_QWEN_SAFETENSORS").unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/qwenimage_2kto4k.safetensors"
        ))
    })
}

fn gemma_dir() -> PathBuf {
    env_path("PID_GEMMA_DIR").unwrap_or_else(|| panic!("set PID_GEMMA_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"))
}

fn stats(img: &Image) -> (u8, u8, f64) {
    let (mut lo, mut hi) = (255u8, 0u8);
    let mut sum = 0u64;
    for &p in &img.pixels {
        lo = lo.min(p);
        hi = hi.max(p);
        sum += p as u64;
    }
    (lo, hi, sum as f64 / img.pixels.len() as f64)
}

fn save_png(img: &Image, path: &str) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs the Krea Turbo snapshot + converted PiD checkpoint + gemma-2-2b-it"]
fn krea_turbo_pid_decode_vs_vae() {
    let size: u32 = std::env::var("KREA_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let spec = LoadSpec::new(WeightsSource::Dir(krea_snapshot())).with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!("loading Krea 2 Turbo (+PiD overlay), size={size} ...");
    let t = Instant::now();
    let model = load(&spec).expect("load Krea + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let base = GenerationRequest {
        prompt: "a red fox sitting in a snowy pine forest at dawn, photorealistic".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        ..Default::default()
    };

    // --- VAE baseline ---
    let t = Instant::now();
    let vae_img = match model.generate(&base, &mut |_| {}).expect("vae generate") {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    };
    let vae_dt = t.elapsed().as_secs_f32();
    let (vlo, vhi, vmu) = stats(&vae_img);
    eprintln!(
        "VAE: {}x{} in {vae_dt:.2}s  range [{vlo},{vhi}] mean {vmu:.1}",
        vae_img.width, vae_img.height
    );
    assert_eq!(vae_img.width, size, "VAE width == native");

    // --- PiD path ---
    let pid_req = GenerationRequest {
        use_pid: true,
        ..base.clone()
    };
    let t = Instant::now();
    let pid_img = match model.generate(&pid_req, &mut |_| {}).expect("pid generate") {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    };
    let pid_dt = t.elapsed().as_secs_f32();
    let (plo, phi, pmu) = stats(&pid_img);
    eprintln!(
        "PiD: {}x{} in {pid_dt:.2}s  range [{plo},{phi}] mean {pmu:.1}",
        pid_img.width, pid_img.height
    );

    assert_eq!(pid_img.width, size * 4, "PiD width == 4× native");
    assert_eq!(pid_img.height, size * 4, "PiD height == 4× native");
    assert!(phi as i32 - plo as i32 > 40, "PiD output near-flat");

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(dir);
    save_png(&vae_img, &format!("{dir}/krea_vae_{}.png", vae_img.width));
    save_png(&pid_img, &format!("{dir}/krea_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/krea_vae_{}.png + krea_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}

struct LadderRun {
    conditioning: usize,
    denoise: usize,
    decode: usize,
    image: Image,
}

impl LadderRun {
    fn peak(&self) -> usize {
        self.conditioning.max(self.denoise).max(self.decode)
    }
}

fn run_pid_ladder(model: &dyn mlx_gen::Generator, size: u32, full: bool) -> LadderRun {
    let request = GenerationRequest {
        prompt: "a red fox sitting in a snowy pine forest at dawn, photorealistic".into(),
        width: size,
        height: size,
        count: 1,
        steps: Some(1),
        seed: Some(7),
        use_pid: true,
        memory: Some(GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            chunk_attention: full,
            stream_transformer_blocks: full,
            decode_tile_edge: Some(mlx_gen_pid::DecodeRoutes::pid_edges()[0]),
            decode_overlap: Some(mlx_gen_pid::DecodeRoutes::pid_overlap()),
            attention_chunk_size: full
                .then_some(mlx_gen_krea::block_memory_strategy::ATTENTION_CHUNK_SIZE),
            transformer_window_size: full
                .then_some(mlx_gen_krea::block_memory_strategy::TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: full.then_some(TransformerComponent::Dit),
            ..Default::default()
        }),
        ..Default::default()
    };
    clear_cache();
    reset_peak_memory();
    let mut conditioning = 0;
    let mut denoise = 0;
    let output = model
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
        .expect("generate Krea PiD ladder image");
    let decode = get_peak_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    clear_cache();
    LadderRun {
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
#[ignore = "needs cached q4 Krea, Qwen PiD, Gemma weights, and Apple/Metal"]
fn krea_pid_full_ladder_exercises_multitile_decode_attention_and_window() {
    let size: u32 = std::env::var("KREA_PID_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(768);
    assert!(
        size * 4 > mlx_gen_pid::DecodeRoutes::pid_edges()[0],
        "PiD output must exceed the selected tile edge"
    );
    let spec = LoadSpec::new(WeightsSource::Dir(krea_snapshot()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization)
        .with_pid(
            WeightsSource::File(pid_checkpoint()),
            WeightsSource::Dir(gemma_dir()),
        );
    let model = load(&spec).expect("load streamable Krea + PiD");
    let baseline = run_pid_ladder(model.as_ref(), size, false);
    let full = run_pid_ladder(model.as_ref(), size, true);
    let (max_delta, mean_delta) = image_delta(&baseline.image, &full.image);
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    eprintln!(
        "PID_LADDER baseline conditioning={:.3} denoise={:.3} decode={:.3} request={:.3}",
        baseline.conditioning as f64 / GIB,
        baseline.denoise as f64 / GIB,
        baseline.decode as f64 / GIB,
        baseline.peak() as f64 / GIB,
    );
    eprintln!(
        "PID_LADDER full conditioning={:.3} denoise={:.3} decode={:.3} request={:.3}",
        full.conditioning as f64 / GIB,
        full.denoise as f64 / GIB,
        full.decode as f64 / GIB,
        full.peak() as f64 / GIB,
    );
    assert!(
        max_delta <= 1,
        "PiD attention/window changed output: max {max_delta}, mean {mean_delta:.6}"
    );
    assert!(
        full.peak() <= baseline.peak(),
        "the full PiD composition raised request peak"
    );
    eprintln!(
        "RESULT status=pass provider=krea_2_turbo route=pid native_size={} pid_size={} decode_edge={} decode_overlap={} baseline_peak_gib={:.3} full_peak_gib={:.3} max_channel_delta={} mean_channel_delta={:.6}",
        size,
        size * 4,
        mlx_gen_pid::DecodeRoutes::pid_edges()[0],
        mlx_gen_pid::DecodeRoutes::pid_overlap(),
        baseline.peak() as f64 / GIB,
        full.peak() as f64 / GIB,
        max_delta,
        mean_delta,
    );
}

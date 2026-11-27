//! SW-6132 (SW-F287 S7) — the shared, job-level lossy-attention opt-in on a real **video** model
//! with no engine option of its own: SeedVR2-3B video upscaling at bf16, with the same switch a
//! product worker sets for any Candle model (`candle_gen::set_lossy_attention`, re-exported as
//! `runtime_cuda::attention::set_lossy_attention`). An 8-frame 256² -> 512² clip (one chunk), with
//! the switch off, `int8` and `sol`:
//!
//! - off: no INT8 / Sol-Attn route and no opt-in refusal (the default is exact routing);
//! - `int8`: the INT8 kernel serves the DiT's window attention (calls > 0); every call is offered
//!   (INT8 calls + INT8 refusals = all calls), so nothing it cannot serve is silent;
//! - `sol`: every call is offered to Sol-Attn as well (Sol-Attn calls + refusals = all calls); the
//!   count of each is printed;
//! - both opt-in clips keep every frame non-degenerate; their PSNR against the off clip is printed
//!   (recorded, not gated — SeedVR2's quality bar is the owner's from the saved frames);
//! - the switch is off again after every clip (the worker's job-end contract).
//!
//! ```text
//! set SEEDVR2_CKPT=<numz/SeedVR2_comfyUI snapshot dir>
//! cargo test -p candle-gen-seedvr2 --features cuda,model-inference-proof --release --test integration
//!   lossy_attention_job_real_weights:: -- --ignored --nocapture --test-threads=1
//! ```

use candle_gen::candle_core::DType;
use candle_gen::fused_attention::{
    attention_path_counts, lossy_attention, set_lossy_attention, AttentionPathCounts,
    Int8AttentionConfig, LossyAttentionOptIns, SolConfig,
};
use candle_gen::gen_core::Image;
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;

const DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";
const SRC: usize = 256;
const TGT: usize = 512;
const FRAMES: usize = 8;

/// A deterministic structured frame translated by `(dx, dy)`: a clip with coherent motion.
fn frame(dx: i32, dy: i32) -> Image {
    let mut pixels = vec![0u8; SRC * SRC * 3];
    let s = SRC as i32;
    for y in 0..SRC {
        for x in 0..SRC {
            let i = (y * SRC + x) * 3;
            let sx = ((x as i32 - dx).rem_euclid(s)) as usize;
            let sy = ((y as i32 - dy).rem_euclid(s)) as usize;
            let check = (((sx / 12) + (sy / 12)) % 2) as u8 * 90;
            let c = SRC as f32 / 2.0;
            let ring = (((sx as f32 - c).powi(2) + (sy as f32 - c).powi(2)).sqrt() * 0.18).sin();
            pixels[i] = (sx * 255 / SRC) as u8;
            pixels[i + 1] = 40 + check;
            pixels[i + 2] = (((ring + 1.0) * 0.5) * 255.0) as u8;
        }
    }
    Image {
        width: SRC as u32,
        height: SRC as u32,
        pixels,
    }
}

/// Resets the job switch even when an assertion unwinds.
struct JobSwitch;
impl JobSwitch {
    fn set(opt_ins: LossyAttentionOptIns) -> Self {
        set_lossy_attention(opt_ins).expect("a valid selection");
        Self
    }
}
impl Drop for JobSwitch {
    fn drop(&mut self) {
        set_lossy_attention(LossyAttentionOptIns::OFF).unwrap();
    }
}

fn upscale(
    pipe: &Seedvr2Pipeline,
    clip: &[Image],
    opt_ins: LossyAttentionOptIns,
) -> (Vec<Image>, AttentionPathCounts, f64) {
    let job = JobSwitch::set(opt_ins);
    let before = attention_path_counts();
    let t0 = std::time::Instant::now();
    let out = pipe
        .generate_video(clip, TGT, TGT, 42, 0.0, None, None, None)
        .expect("generate_video");
    let secs = t0.elapsed().as_secs_f64();
    let counts = attention_path_counts().since(&before);
    drop(job);
    assert_eq!(
        lossy_attention(),
        LossyAttentionOptIns::OFF,
        "reset at job end"
    );
    assert_eq!(out.len(), clip.len(), "frame count preserved");
    for (i, f) in out.iter().enumerate() {
        assert_eq!((f.width, f.height), (TGT as u32, TGT as u32));
        assert!(
            f.pixels.iter().min() != f.pixels.iter().max(),
            "frame {i} is constant"
        );
    }
    (out, counts, secs)
}

fn psnr(a: &[Image], b: &[Image]) -> f64 {
    let (mut se, mut n) = (0f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        se += x
            .pixels
            .iter()
            .zip(&y.pixels)
            .map(|(&p, &q)| (p as f64 - q as f64).powi(2))
            .sum::<f64>();
        n += x.pixels.len();
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn save(frame: &Image, name: &str) {
    let dir = std::env::var("SEEDVR2_SMOKE_ARTIFACT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("seedvr2_lossy_attention_job");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn seedvr2_3b_job_level_lossy_attention_smoke() {
    let ckpt = std::env::var("SEEDVR2_CKPT")
        .expect("set SEEDVR2_CKPT to a numz/SeedVR2_comfyUI checkpoint dir");
    let device = candle_gen::default_device().expect("device");
    assert!(device.is_cuda(), "the opt-in kernels are CUDA kernels");
    let pipe = Seedvr2Pipeline::load(
        &ckpt,
        DIT_FILE,
        &DitConfig::seedvr2_3b(),
        DType::BF16,
        &device,
    )
    .expect("load SeedVR2-3B");
    let clip: Vec<Image> = (0..FRAMES as i32).map(|t| frame(2 * t, t)).collect();

    let (off, off_counts, off_s) = upscale(&pipe, &clip, LossyAttentionOptIns::OFF);
    assert_eq!((off_counts.int8, off_counts.sol), (0, 0), "{off_counts:?}");
    assert_eq!(off_counts.int8_refusals.total(), 0, "{off_counts:?}");
    assert_eq!(off_counts.sol_refusals.total(), 0, "{off_counts:?}");
    assert!(off_counts.total() > 0, "{off_counts:?}");

    let (int8, int8_counts, int8_s) = upscale(
        &pipe,
        &clip,
        LossyAttentionOptIns {
            int8: Some(Int8AttentionConfig::default()),
            sol: None,
        },
    );
    assert!(int8_counts.int8 > 0, "{int8_counts:?}");
    assert_eq!(
        int8_counts.int8 + int8_counts.int8_refusals.total(),
        int8_counts.total(),
        "every call offered to INT8: {int8_counts:?}"
    );

    let (sol, sol_counts, sol_s) = upscale(
        &pipe,
        &clip,
        LossyAttentionOptIns {
            int8: None,
            sol: Some(SolConfig::top_k(0.25)),
        },
    );
    assert_eq!(
        sol_counts.sol + sol_counts.sol_refusals.total(),
        sol_counts.total(),
        "every call offered to Sol-Attn: {sol_counts:?}"
    );

    let (db_int8, db_sol) = (psnr(&int8, &off), psnr(&sol, &off));
    for (clip, name) in [(&off, "off"), (&int8, "job_int8"), (&sol, "job_sol_0.25")] {
        save(&clip[0], &format!("{name}_frame0"));
    }
    eprintln!(
        "[seedvr2 job] {FRAMES}x{SRC}^2 -> {TGT}^2 bf16: off {off_s:.2}s; int8 {int8_s:.2}s, {} INT8 \
         of {} calls, refusals {:?}, PSNR vs off {db_int8:.2} dB; sol {sol_s:.2}s, {} Sol-Attn of \
         {} calls, refusals {:?}, PSNR vs off {db_sol:.2} dB (recorded)",
        int8_counts.int8,
        int8_counts.total(),
        int8_counts.int8_refusals,
        sol_counts.sol,
        sol_counts.total(),
        sol_counts.sol_refusals
    );
    assert!(!db_int8.is_nan() && !db_sol.is_nan());
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED lossy_attention_job_real_weights::seedvr2_3b_job_level_lossy_attention_smoke"
    );
}

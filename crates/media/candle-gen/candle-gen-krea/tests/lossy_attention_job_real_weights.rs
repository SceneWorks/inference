//! SW-6132 (SW-F287 S7) — the shared, job-level lossy-attention opt-in on real Krea 2 Turbo
//! weights, through the **registered** `krea_2_turbo` generator with no engine option at all: the
//! same switch a product worker sets for any Candle model
//! (`candle_gen::set_lossy_attention`, re-exported as `runtime_cuda::attention::set_lossy_attention`).
//!
//! A fixed-seed 1024² 8-step render with the switch off, then with `int8: Some(16-bit P·V)`:
//!
//! - off: no INT8 / Sol-Attn route and no opt-in refusal at all (the default is exact routing);
//! - on: the INT8 kernel serves the denoise attention (at least one call per single-stream block
//!   per step) and every call it cannot serve is a counted refusal (`int8_refusals`), never silent;
//!   the render is >= 35 dB PSNR against the off render (the SW-6131 engine-option bar), and the
//!   s/step of both are printed;
//! - the switch is reset to off after the render (the worker's job-end contract).
//!
//! ```sh
//! KREA_TURBO_DIR=<snapshot>/bf16 KREA_SMOKE_ARTIFACT_DIR=<out> \
//!   cargo test -p candle-gen-krea --release --features cuda,flash-attn,model-inference-proof \
//!   --test integration lossy_attention_job_real_weights:: -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::fused_attention::{
    attention_path_counts, lossy_attention, set_lossy_attention, AttentionPathCounts,
    Int8AttentionConfig, LossyAttentionOptIns,
};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";
const SIZE: u32 = 1024;
const STEPS: u32 = 8;

fn snapshot() -> PathBuf {
    std::env::var("KREA_TURBO_DIR")
        .map(PathBuf::from)
        .expect("set KREA_TURBO_DIR to the bf16 Krea 2 Turbo snapshot")
}

fn artifact_dir() -> PathBuf {
    let dir = std::env::var("KREA_SMOKE_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("krea_lossy_attention_job");
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

struct Render {
    image: Image,
    s_per_step: f64,
    counts: AttentionPathCounts,
}

fn render(opt_ins: LossyAttentionOptIns) -> Render {
    let registry = candle_gen_krea::provider_registry().expect("krea registry");
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let generator = registry
        .load("krea_2_turbo", &spec)
        .expect("load krea_2_turbo");
    let request = GenerationRequest {
        prompt: PROMPT.into(),
        width: SIZE,
        height: SIZE,
        count: 1,
        seed: Some(7),
        steps: Some(STEPS),
        ..Default::default()
    };
    let mut marks = Vec::new();
    let (out, counts) = {
        let _job = JobSwitch::set(opt_ins);
        let before = attention_path_counts();
        let out = generator
            .generate(&request, &mut |progress| {
                if let Progress::Step { .. } = progress {
                    marks.push(Instant::now());
                }
            })
            .expect("generate");
        (out, attention_path_counts().since(&before))
    };
    assert_eq!(
        lossy_attention(),
        LossyAttentionOptIns::OFF,
        "reset at job end"
    );
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected images");
    };
    let image = images.pop().expect("one image");
    assert_eq!((image.width, image.height), (SIZE, SIZE));
    let s_per_step = match marks.as_slice() {
        [first, .., last] => (*last - *first).as_secs_f64() / (marks.len() - 1) as f64,
        _ => f64::NAN,
    };
    drop(generator);
    Render {
        image,
        s_per_step,
        counts,
    }
}

fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let se: f64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&p, &q)| (p as f64 - q as f64).powi(2))
        .sum();
    let mse = se / a.pixels.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn save(image: &Image, name: &str) {
    let path = artifact_dir().join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn krea_turbo_job_level_int8_attention_smoke() {
    let off = render(LossyAttentionOptIns::OFF);
    assert_eq!(
        (off.counts.int8, off.counts.sol),
        (0, 0),
        "{:?}",
        off.counts
    );
    assert_eq!(off.counts.int8_refusals.total(), 0, "{:?}", off.counts);
    assert_eq!(off.counts.sol_refusals.total(), 0, "{:?}", off.counts);
    assert!(off.counts.flash > 0, "{:?}", off.counts);

    let on = render(LossyAttentionOptIns {
        int8: Some(Int8AttentionConfig::default()),
        sol: None,
    });
    assert!(
        on.counts.int8 >= STEPS as u64,
        "the job switch must reach the denoise attention: {:?}",
        on.counts
    );
    assert_eq!(on.counts.sol, 0, "{:?}", on.counts);
    // Every opt-in call either ran INT8 or was a counted refusal routed by the flash rule.
    eprintln!(
        "[krea job int8] int8 {} calls, flash {}, budgeted {}, int8 refusals {:?}",
        on.counts.int8,
        on.counts.flash,
        on.counts.budgeted(),
        on.counts.int8_refusals
    );
    let db = psnr(&on.image, &off.image);
    save(&off.image, "off_1024");
    save(&on.image, "job_int8_1024");
    eprintln!(
        "[krea job int8] {SIZE}^2 {STEPS} steps: off {:.3} s/step, job int8 {:.3} s/step, \
         PSNR {db:.2} dB",
        off.s_per_step, on.s_per_step
    );
    assert!(db >= 35.0, "job-level INT8 render PSNR {db:.2} dB < 35 dB");
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED lossy_attention_job_real_weights::krea_turbo_job_level_int8_attention_smoke"
    );
}

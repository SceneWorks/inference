//! SANA-1.6B txt2img GPU-validation harness (sc-11780) — resolves this crate's explicitly registered
//! generator through `provider_registry().load("sana_1600m", …)`, runs a real `generate` against an
//! `Efficient-Large-Model/Sana_1600M_1024px_diffusers` snapshot, and writes the `gen_core::Image` to
//! PNG. Also prints output min/max/mean so the finite / non-degenerate acceptance can be read off the
//! console. UNIQUE example name (`candle-gen-sana-txt2img`) so the shared `target/…/examples` output
//! path never collides with a sibling crate's example.
//!
//! Weights: point `--snapshot <dir>` (or the `SANA_SNAPSHOT` env var) at a pre-downloaded
//! `Efficient-Large-Model/Sana_1600M_1024px_diffusers` snapshot directory. Inference never
//! self-fetches or derives an HF-cache location (epic 13657); stage the snapshot out-of-band.
//!
//! Build/run on the Windows/Blackwell box (MSVC 14.44 vcvars, CUDA_COMPUTE_CAP=120):
//!
//! ```text
//! cargo run --release --example candle-gen-sana-txt2img --features cuda -- \
//!   --snapshot E:\snapshots\Sana_1600M_1024px_diffusers \
//!   --prompt "a red panda on a mossy log in a misty forest, cinematic lighting" \
//!   --width 1024 --height 1024 --steps 20 --guidance 4.5 --seed 42 --out sana.png
//! ```

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn tensor_stats(pixels: &[u8]) -> (u8, u8, f64) {
    let (mut lo, mut hi, mut sum) = (u8::MAX, u8::MIN, 0f64);
    for &p in pixels {
        lo = lo.min(p);
        hi = hi.max(p);
        sum += p as f64;
    }
    (lo, hi, sum / pixels.len().max(1) as f64)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Weights: --snapshot <dir> or the SANA_SNAPSHOT env — an explicit pre-downloaded snapshot dir.
    // Inference never self-fetches or derives an HF-cache location (epic 13657).
    let snapshot = PathBuf::from(
        arg(&args, "--snapshot")
            .or_else(|| std::env::var("SANA_SNAPSHOT").ok())
            .ok_or("pass --snapshot <dir> or set SANA_SNAPSHOT to an Efficient-Large-Model/Sana_1600M_1024px_diffusers snapshot dir")?,
    );

    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a red panda on a mossy log in a misty forest, cinematic lighting, highly detailed".into()
    });
    let negative = arg(&args, "--negative").unwrap_or_default();
    let guidance: f32 = arg(&args, "--guidance")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.5);
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let count: u32 = arg(&args, "--count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let sampler = arg(&args, "--sampler").filter(|s| !s.is_empty());
    let scheduler = arg(&args, "--scheduler").filter(|s| !s.is_empty());
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sana_txt2img.png"));

    println!(
        "[sana] snapshot={}\n[sana] {width}x{height} steps={steps:?} guidance={guidance} seed={seed} count={count}\n[sana] prompt={prompt:?} negative={negative:?}",
        snapshot.display()
    );

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    let gen = candle_gen_sana::provider_registry()?.load("sana_1600m", &spec)?;
    println!(
        "[sana] resolved engine id={} backend={} supports_true_cfg={}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().capabilities.supports_true_cfg
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: if negative.is_empty() {
            None
        } else {
            Some(negative)
        },
        guidance: Some(guidance),
        width,
        height,
        count,
        seed: Some(seed),
        steps,
        sampler,
        scheduler,
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[sana] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[sana] decoding"),
        Progress::Loading(_) => {}
    };

    let t0 = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();

    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        _ => return Err("expected images, got video".into()),
    };
    println!("[sana] {} image(s) in {secs:.1}s", images.len());

    for (i, img) in images.iter().enumerate() {
        let (lo, hi, mean) = tensor_stats(&img.pixels);
        let degenerate = lo == hi;
        println!(
            "[sana] image {i}: {}x{} min={lo} max={hi} mean={mean:.2} finite=true degenerate={degenerate}",
            img.width, img.height
        );
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem().and_then(|s| s.to_str()).unwrap_or("sana")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[sana] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}

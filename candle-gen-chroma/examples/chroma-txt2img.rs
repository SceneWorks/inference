//! Chroma txt2img smoke driver — exercises the full candle-gen seam end-to-end on a real GPU:
//! `gen_core::registry::load("chroma1_hd"|"chroma1_base"|"chroma1_flash", …)` resolves THIS crate's
//! inventory-registered generator, runs [`Generator::generate`] against a local Chroma snapshot, and
//! writes each `gen_core::Image` to PNG.
//!
//! The human-eyeball check behind sc-5484 (the worker, not this example, owns asset writes in
//! production). Build with the CUDA backend on the Windows/Blackwell box:
//!
//! ```text
//! cargo run --release --example chroma-txt2img --features cuda -- \
//!   --snapshot "C:\Users\…\models--lodestones--Chroma1-HD\snapshots\<hash>" \
//!   --model hd --prompt "a photo of a rusty robot holding a lit candle" --seed 42 --out out.png
//! ```
//!
//! The snapshot is a Chroma diffusers tree (`tokenizer/`, `text_encoder/`, `transformer/`, `vae/`).
//! `--model hd|base` defaults to 28 steps / true_cfg 4.0; `--model flash` to 8 steps / true_cfg 1.0.

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("CHROMA_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set CHROMA_SNAPSHOT) pointing at a Chroma snapshot")?;
    let model = arg(&args, "--model").unwrap_or_else(|| "hd".into());
    let engine = match model.as_str() {
        "hd" => "chroma1_hd",
        "base" => "chroma1_base",
        "flash" => "chroma1_flash",
        other => return Err(format!("--model must be hd|base|flash (got {other:?})").into()),
    };
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed"
            .into()
    });
    let negative = arg(&args, "--negative");
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let true_cfg: Option<f32> = arg(&args, "--true-cfg").and_then(|s| s.parse().ok());
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
    let repeat: u32 = arg(&args, "--repeat")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("chroma_smoke.png"));

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] engine={engine} {width}x{height} steps={steps:?} true_cfg={true_cfg:?} seed={seed} count={count}\n[smoke] prompt={prompt:?}"
    );

    // Force-link the provider so its `inventory::submit!` registrations survive the linker.
    candle_gen_chroma::force_link();

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load(engine, &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: negative,
        width,
        height,
        count,
        seed: Some(seed),
        steps,
        true_cfg,
        ..Default::default()
    };

    let mut call_secs: Vec<f32> = Vec::with_capacity(repeat as usize);
    let mut images = Vec::new();
    for call in 0..repeat {
        let mut on_progress = |p: Progress| match p {
            Progress::Step { current, total } => {
                print!(
                    "\r[smoke] call {}/{repeat} step {current}/{total}   ",
                    call + 1
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Progress::Decoding => println!("\n[smoke] call {}/{repeat} decoding", call + 1),
            // Additive Sequential-residency load signal (sc-11126); no-op in this smoke example.
            Progress::Loading(_) => {}
        };
        let t_call = std::time::Instant::now();
        let output = gen.generate(&req, &mut on_progress)?;
        call_secs.push(t_call.elapsed().as_secs_f32());
        images = match output {
            GenerationOutput::Images(imgs) => imgs,
            GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
        };
    }
    let gen_s = *call_secs.last().unwrap();
    println!("[smoke] {} image(s) in {gen_s:.1}s total", images.len());

    for (i, img) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("chroma_smoke")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[smoke] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}

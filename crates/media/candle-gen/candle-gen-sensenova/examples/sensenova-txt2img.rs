//! SenseNova-U1 txt2img smoke driver — exercises the candle-gen seam end-to-end on a real GPU and
//! writes the produced `gen_core::Image` (RGB8) to PNG so the render can be eyeballed.
//!
//! Since sc-14249 (epic 9083) the `--snapshot` may be **any tier of the SceneWorks turnkey** — a
//! packed `q4/`/`q8/` subdir or the dense `bf16/` one — not just a dense snapshot. The provider
//! packed-detects per projection off the weights on disk, so the tier is chosen purely by which dir
//! you point at, and `--vram-probe` is how the per-tier `candle.vramGbByTier` rows are measured.
//!
//! ```text
//! cargo run -p candle-gen-sensenova --features cuda --release --example sensenova-txt2img -- \
//!   --snapshot "E:\huggingface\hub\models--SceneWorks--sensenova-u1-8b-mlx\snapshots\<hash>\q4" \
//!   --prompt "a fox reading a book by candlelight" --steps 8 --seed 42 --vram-probe --out q4.png
//! ```
//!
//! `--fast` selects the 8-step distilled id (`sensenova_u1_8b_fast`), which additionally exercises the
//! distill-LoRA merge / pre-merged-marker skip. Use `--vram-probe` only on an otherwise-idle GPU: the
//! report is a device-level `nvidia-smi` delta and prints its baseline so contamination is visible.

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("SENSENOVA_SNAPSHOT").ok())
        .ok_or(
            "pass --snapshot <dir> (or set SENSENOVA_SNAPSHOT) pointing at a SenseNova-U1-8B-MoT \
             snapshot, or at one q4/q8/bf16 tier of the SceneWorks turnkey",
        )?;
    let prompt =
        arg(&args, "--prompt").unwrap_or_else(|| "a fox reading a book by candlelight".to_string());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sensenova-out.png"));
    // `--fast` / SENSENOVA_FAST selects the 8-step distilled variant. Steps default to `None` so each
    // variant's own default applies (fast = 8 @ CFG 1.0, base = 50 @ CFG 4.0).
    let fast = args.iter().any(|a| a == "--fast") || std::env::var("SENSENOVA_FAST").is_ok();
    let steps: Option<u32> = arg(&args, "--steps")
        .or_else(|| std::env::var("SENSENOVA_STEPS").ok())
        .and_then(|s| s.parse().ok());

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} steps={} seed={seed} fast={fast}\n\
         [smoke] prompt={prompt:?}",
        steps
            .map(|s| s.to_string())
            .unwrap_or_else(|| "default".into())
    );

    // sc-9094 per-tier VRAM probe (shared `candle_gen::testkit::VramProbe`): `--vram-probe [--gpu n]`
    // brackets load / steady / overall-peak, which is how the manifest's per-tier
    // `candle.vramGbByTier` + `minMemoryGb` rows are derived. Run it on an idle GPU — the report
    // prints its baseline, and `assert_trustworthy` exists for a harness that must not publish a
    // contaminated number.
    let vram_gpu: Option<usize> = if args.iter().any(|a| a == "--vram-probe") {
        Some(
            arg(&args, "--gpu")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        )
    } else {
        None
    };
    let mut probe = vram_gpu.map(candle_gen::testkit::VramProbe::start);

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let load_phase = probe.as_ref().map(|p| p.phase());
    let t_load = std::time::Instant::now();
    let generator = if fast {
        candle_gen_sensenova::load_fast(&spec)?
    } else {
        candle_gen_sensenova::load(&spec)?
    };

    let req = GenerationRequest {
        prompt,
        width,
        height,
        steps,
        seed: Some(seed),
        count: 1,
        ..Default::default()
    };
    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[smoke] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[smoke] decoding"),
        Progress::Loading(phase) => println!("[smoke] loading {phase:?}"),
    };
    // This provider is LAZY: the weights land on the device inside the first `generate`, not in
    // `load`. So the load phase cannot be closed before it — closing both phases after the single
    // generate means `load-peak` and `overall-peak` bracket the same span here. The honest reading
    // for a lazy provider, and the reason `steady` is the figure to compare across tiers.
    let out_img = generator.generate(&req, &mut on_progress)?;
    if let (Some(p), Some(ph)) = (probe.as_mut(), load_phase) {
        p.end_load(ph);
        let gen_phase = p.phase();
        p.end_gen(gen_phase);
    }
    println!(
        "\n[smoke] generate took {:.2}s",
        t_load.elapsed().as_secs_f32()
    );

    let GenerationOutput::Images(images) = out_img else {
        return Err("expected image output".into());
    };
    let img = images.into_iter().next().ok_or("no image produced")?;
    // Per-pixel std — the cheap degenerate-render tell (a stub / black / flat frame has std ~0).
    let mean = img.pixels.iter().map(|&v| v as f64).sum::<f64>() / img.pixels.len() as f64;
    let var = img
        .pixels
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / img.pixels.len() as f64;
    println!("[smoke] pixel mean {mean:.2} std {:.2}", var.sqrt());

    if let Some(p) = probe.as_ref() {
        println!("[smoke] VRAM {}", p.report());
    }

    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels)
        .ok_or("pixel buffer size mismatch")?;
    buf.save(&out)?;
    println!("[smoke] wrote {}", out.display());
    Ok(())
}

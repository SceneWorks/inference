//! GPU-validation harness for the candle Boogu provider: load a snapshot by engine id, render one
//! image from a prompt, write a PNG.
//!
//! ```text
//! cargo run -p candle-gen-boogu --example boogu-txt2img --features cuda --release -- \
//!   boogu_image D:\models\Boogu-Image-0.1-Base "a red apple on a wooden table" 1024 1024 0 42 out.png
//! ```
//! Arg order: <model_id> <snapshot_dir> <prompt> [width] [height] [steps(0=default)] [seed] [out.png]
//!            [sampler] [scheduler]  — curated names (sc-9009); omit for the engine's native path.

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).cloned().unwrap_or_else(|| "boogu_image".into());
    let snapshot = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "D:/models/Boogu-Image-0.1-Base".into());
    let prompt = a
        .get(3)
        .cloned()
        .unwrap_or_else(|| "a red apple on a wooden table".into());
    let width: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 → engine default
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a
        .get(8)
        .cloned()
        .unwrap_or_else(|| "boogu_render.png".into());
    let sampler = a.get(9).cloned().filter(|s| !s.is_empty());
    let scheduler = a.get(10).cloned().filter(|s| !s.is_empty());

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.into()));
    let mut probe = candle_gen::testkit::VramProbe::start_rendered().assert_idle(2.0);
    let load_phase = probe.phase();
    let gen = candle_gen_boogu::provider_registry()?.load(&model, &spec)?;
    probe.end_load(load_phase);

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: if steps == 0 { None } else { Some(steps) },
        sampler,
        scheduler,
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            eprintln!("step {current}/{total}");
        }
        Progress::Decoding => eprintln!("decoding…"),
        Progress::Loading(phase) => eprintln!("loading {phase:?}"),
    };

    let gen_phase = probe.phase();
    let GenerationOutput::Images(images) = gen.generate(&req, &mut on_progress)? else {
        return Err("expected images".into());
    };
    probe.end_gen(gen_phase);
    let report = probe.report();
    eprintln!(
        "vram baseline={:.2} GB load-peak={:.2} GB steady={:.2} GB overall-peak={:.2} GB",
        report.baseline_gb, report.load_peak_gb, report.steady_gb, report.peak_gb
    );
    let img = images.into_iter().next().ok_or("no image")?;

    let buf =
        image::RgbImage::from_raw(img.width, img.height, img.pixels).ok_or("bad image buffer")?;
    buf.save(&out)?;
    eprintln!("wrote {out} ({}x{})", width, height);
    Ok(())
}

//! Offline sc-25328 comparison against installed SeedVR2; no product changes.
use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen_seedvr2::{config::DitConfig, pipeline::Seedvr2Pipeline};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn arg(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    Ok(args
        .windows(2)
        .find(|a| a[0] == name)
        .ok_or_else(|| format!("missing {name}"))?[1]
        .clone())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
        return Err("assign CUDA_VISIBLE_DEVICES explicitly before this offline experiment".into());
    }
    if !cfg!(feature = "cuda") {
        return Err("build with --features cuda".into());
    }
    let root = PathBuf::from(arg("--root")?);
    let bytes = std::fs::read(arg("--rgb24")?)?;
    let out = PathBuf::from(arg("--out")?);
    if bytes.len() != 39 * 512 * 288 * 3 {
        return Err("requires exactly39 frames512x288RGB24".into());
    }
    let frames: Vec<_> = bytes
        .chunks_exact(512 * 288 * 3)
        .map(|pixels| Image {
            width: 512,
            height: 288,
            pixels: pixels.to_vec(),
        })
        .collect();
    let device = Device::new_cuda(0)?;
    let start = Instant::now();
    let pipe = Seedvr2Pipeline::load(
        root,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        DType::BF16,
        &device,
    )?;
    device.synchronize()?;
    let load_seconds = start.elapsed().as_secs_f64();
    println!(
        "SeedVR2 3B BF16 load: {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let cancel = CancelFlag::new();
    let mut progress = |done, total| println!("SeedVR2 completed {done}/{total}");
    let output = pipe.generate_video(
        &frames,
        1024,
        576,
        444,
        0.,
        None,
        Some(&cancel),
        Some(&mut progress),
    )?;
    if output.len() != 39
        || output
            .iter()
            .any(|image| image.width != 1024 || image.height != 576)
    {
        return Err("baseline output shape/count mismatch".into());
    }
    let mut file = std::fs::File::create(&out)?;
    for image in output {
        file.write_all(&image.pixels)?;
    }
    device.synchronize()?;
    let video_seconds = start.elapsed().as_secs_f64();
    std::fs::write(
        out.with_extension("stages.json"),
        format!(
            r#"{{"backend":"candle/cuda","model":"SeedVR2 3B","tier":"BF16","seed":444,"frames":39,"width":1024,"height":576,"load_seconds":{load_seconds},"video_seconds":{video_seconds}}}"#
        ),
    )?;
    println!(
        "SeedVR2 3B BF16 video: {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

//! Real-weight checks of the restoration transform (`iris_3b_restore`, sc-25683) on the build's
//! device (`cuda:0` under `--features cuda`) against the pinned `speridlabs/iris-3b` `upscaler/`
//! export. `#[ignore]`d: they need the ~12 GB export and a GPU. Inputs from the environment:
//!
//! * `IRIS_UPSCALER_DIR` — the `upscaler/` folder of the pinned snapshot.
//! * `IRIS_OUT` — a directory the restored PNGs are written into.
//! * `IRIS_REAL_COMPUTE=fp32` — upstream's FP32 path instead of the release's bf16 autocast policy.
//! * `IRIS_VRAM_PROBE=1` — sample device VRAM with `nvidia-smi` across load and restore.
//!
//! Parity is against the committed upstream reference
//! `mlx-gen-iris/tests/fixtures/iris_restoration_real_small.safetensors`
//! (`tools/dump_iris_restoration_realweight.py`, `IRIS_RESTORE_CASE=small`): the native bf16 path is
//! held to 1.5x upstream's OWN bf16-autocast-vs-FP32 mean RGB8 distance on the same input; FP32 to a
//! 0.5-level mean.
//!
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! restoration_real_weights:: --test-threads 1`

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    Image, InputSizing, LoadSpec, Precision, Progress, TargetSize, Transform, TransformRequest,
    WeightsSource,
};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};

use crate::common::{fixture, host_f32, Fixture};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn fp32() -> bool {
    std::env::var("IRIS_REAL_COMPUTE").as_deref() == Ok("fp32")
}

fn rgb8(golden: &Fixture, key: &str) -> Vec<u8> {
    host_f32(golden.require(key))
        .iter()
        .map(|&v| v as u8)
        .collect()
}

fn rgb8_distance(a: &[u8], b: &[u8]) -> (u8, f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut sum, mut sq) = (0u8, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += d as f64;
        sq += (d as f64) * (d as f64);
    }
    let n = a.len() as f64;
    let psnr = if sq == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / (sq / n)).log10()
    };
    (max, sum / n, psnr)
}

fn save_png(image: &Image, name: &str) {
    let path = env_dir("IRIS_OUT").join(name);
    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .unwrap()
        .save(&path)
        .unwrap();
    eprintln!("[[IRIS_RESTORE_CANDLE]] wrote {}", path.display());
}

/// One production-path restoration through the catalog: load, validate, apply with per-tile
/// progress, a VRAM line when probing.
fn restore_once(req: &TransformRequest, label: &str) -> (Image, u32) {
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_UPSCALER_DIR")));
    if fp32() {
        spec.precision = Precision::Fp32;
    }
    let probe_vram =
        std::env::var("IRIS_VRAM_PROBE").is_ok_and(|v| v == "1") && used_mib(probe_gpu()).is_some();
    let mut probe = probe_vram.then(VramProbe::start_rendered);
    let start = Instant::now();
    let load_phase = probe.as_ref().map(VramProbe::phase);
    let restorer = candle_gen_iris::provider_registry()
        .unwrap()
        .load_transform("iris_3b_restore", &spec)
        .unwrap();
    if let (Some(p), Some(phase)) = (probe.as_mut(), load_phase) {
        p.end_load(phase);
    }
    eprintln!(
        "[[IRIS_RESTORE_CANDLE]] {label} device={} precision={} load={:.1}s",
        if cfg!(feature = "cuda") {
            "cuda:0"
        } else {
            "cpu"
        },
        if fp32() { "fp32" } else { "bf16" },
        start.elapsed().as_secs_f32()
    );
    restorer.validate(req).unwrap();
    let start = Instant::now();
    let gen_phase = probe.as_ref().map(VramProbe::phase);
    let mut tiles = 0;
    let image = restorer
        .apply(req, &mut |p| {
            if let Progress::Step { current, total } = p {
                tiles = current;
                eprintln!(
                    "[[IRIS_RESTORE_CANDLE]] {label} tile {current}/{total} at {:.1}s",
                    start.elapsed().as_secs_f32()
                );
            }
        })
        .unwrap();
    if let (Some(p), Some(phase)) = (probe.as_mut(), gen_phase) {
        p.end_gen(phase);
    }
    eprintln!(
        "[[IRIS_RESTORE_CANDLE]] {label} {}x{} -> {}x{} in {:.1}s ({tiles} tiles)",
        req.image.width,
        req.image.height,
        image.width,
        image.height,
        start.elapsed().as_secs_f32()
    );
    if let Some(p) = &probe {
        eprintln!("[[IRIS_RESTORE_CANDLE]] {label} vram {}", p.report());
    }
    (image, tiles)
}

fn small_input(golden: &Fixture) -> Image {
    let dims = golden.require("input").dims().to_vec();
    Image {
        width: dims[1] as u32,
        height: dims[0] as u32,
        pixels: rgb8(golden, "input"),
    }
}

/// 64x48 at 4x = 256x192 (enlarged to 1365x1024 → two overlapping 1024-px tiles, resized back,
/// colour-fixed) against upstream's committed RGB8 output.
#[test]
#[ignore = "needs the upscaler export (IRIS_UPSCALER_DIR) and IRIS_OUT"]
fn real_small_restoration_matches_upstream() {
    let golden = fixture("iris_restoration_real_small.safetensors");
    let req = TransformRequest {
        image: small_input(&golden),
        target: TargetSize::Scale(4.0),
        ..Default::default()
    };
    let (image, tiles) = restore_once(&req, "small");
    assert_eq!((image.width, image.height, tiles), (256, 192, 2));
    save_png(&image, "iris_restore_small_x4.png");
    let want = rgb8(&golden, "fp32/output");
    let (own_max, own_mean, own_psnr) = rgb8_distance(&rgb8(&golden, "bf16/output"), &want);
    let (max, mean, psnr) = rgb8_distance(&image.pixels, &want);
    eprintln!(
        "[[IRIS_RESTORE_CANDLE]] small: native vs upstream fp32 RGB8 max {max} mean {mean:.3} PSNR \
         {psnr:.1} dB (upstream's own bf16 vs fp32: max {own_max} mean {own_mean:.3} PSNR \
         {own_psnr:.1} dB)"
    );
    let bound = if fp32() { 0.5 } else { 1.5 * own_mean };
    assert!(mean <= bound, "mean RGB8 distance {mean:.3} exceeds {bound:.3}");
}

/// Release-scale timing render: the small reference input nearest-upsampled 8x to 512x384, then the
/// default 4x restoration (2048x1536, six 1024-px tiles fused). Report-only beyond geometry,
/// progress and a non-degenerate image — the parity gate is the small case.
#[test]
#[ignore = "needs the upscaler export (IRIS_UPSCALER_DIR) and IRIS_OUT"]
fn real_release_scale_restoration_renders() {
    let golden = fixture("iris_restoration_real_small.safetensors");
    let small = small_input(&golden);
    let (w, h) = (small.width * 8, small.height * 8);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = (((y / 8) * small.width + x / 8) * 3) as usize;
            pixels.extend_from_slice(&small.pixels[i..i + 3]);
        }
    }
    let req = TransformRequest {
        image: Image {
            width: w,
            height: h,
            pixels,
        },
        input_sizing: InputSizing::Budgeted,
        ..Default::default()
    };
    let (image, tiles) = restore_once(&req, "release-scale");
    assert_eq!((image.width, image.height, tiles), (2048, 1536, 6));
    save_png(&image, "iris_restore_release_scale_x4.png");
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / image.pixels.len() as f64;
    let std = (image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / image.pixels.len() as f64)
        .sqrt();
    eprintln!("[[IRIS_RESTORE_CANDLE]] release-scale mean {mean:.1} std {std:.1}");
    assert!(mean > 5.0 && mean < 250.0 && std > 5.0, "degenerate image");
}

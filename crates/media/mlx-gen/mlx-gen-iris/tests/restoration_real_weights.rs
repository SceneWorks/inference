//! Real-weight checks of the restoration transform (`iris_3b_restore`, sc-25683) against the pinned
//! `speridlabs/iris-3b` `upscaler/` export. `#[ignore]`d: they need the ~12 GB export and the GPU.
//! Inputs come from the environment (no machine path is baked in):
//!
//! * `IRIS_UPSCALER_DIR` — the `upscaler/` folder of the pinned snapshot.
//! * `IRIS_RESTORE_GOLDEN` — the machine-local upstream reference from
//!   `tools/dump_iris_restoration_realweight.py` (`IRIS_RESTORE_CASE=large`; large-case test only).
//! * `IRIS_REAL_COMPUTE=fp32` — upstream's FP32 path instead of the release's bf16 autocast policy.
//! * `IRIS_OUT` — optional directory the restored PNGs are written into.
//!
//! Tolerances: the native bf16 path is held to upstream's OWN bf16-autocast distance from its FP32
//! output on the same input (both recorded by the oracle), with 50 % headroom for the different
//! bf16 kernels; FP32 is held to a fixed RGB8 bound (summation order only).
//!
//! Run each under an external memory guard:
//! `cargo test -p mlx-gen-iris --release --test integration -- --ignored restoration_real_weights:: --test-threads 1`

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen::gen_core::iris::restoration::{plan_request, restore_detailed, TileGeometry};
use mlx_gen::weights::Weights;
use mlx_gen::{
    Image, InputSizing, LoadSpec, Precision, Progress, TargetSize, TransformRequest, WeightsSource,
};
use mlx_gen_iris::IrisRestorer;

use crate::common::{fixtures, host_f32};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn fp32() -> bool {
    std::env::var("IRIS_REAL_COMPUTE").as_deref() == Ok("fp32")
}

fn spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_UPSCALER_DIR")));
    if fp32() {
        spec.precision = Precision::Fp32;
    }
    spec
}

fn rgb8(golden: &Weights, key: &str) -> Vec<u8> {
    host_f32(golden.require(key).unwrap())
        .iter()
        .map(|&v| v as u8)
        .collect()
}

fn input(golden: &Weights) -> Image {
    let shape = golden.require("input").unwrap().shape().to_vec();
    Image {
        width: shape[1] as u32,
        height: shape[0] as u32,
        pixels: rgb8(golden, "input"),
    }
}

/// `(max, mean)` absolute RGB8 difference and PSNR (dB).
fn rgb8_distance(a: &[u8], b: &[u8]) -> (u8, f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut max = 0u8;
    let mut sum = 0f64;
    let mut sq = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += d as f64;
        sq += (d as f64) * (d as f64);
    }
    let n = a.len() as f64;
    let mse = sq / n;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    (max, sum / n, psnr)
}

fn save_png(image: &Image, name: &str) {
    if let Ok(dir) = std::env::var("IRIS_OUT") {
        let path = Path::new(&dir).join(name);
        image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        eprintln!("wrote {}", path.display());
    }
}

/// Hold the native RGB8 output to upstream's FP32 output: within 1.5x upstream's own bf16-vs-fp32
/// mean distance on bf16 (the release policy), within 0.5 levels mean on FP32.
fn assert_parity(label: &str, got: &[u8], golden: &Weights) {
    let want = rgb8(golden, "fp32/output");
    let upstream_bf16 = rgb8(golden, "bf16/output");
    let (own_max, own_mean, own_psnr) = rgb8_distance(&upstream_bf16, &want);
    let (max, mean, psnr) = rgb8_distance(got, &want);
    eprintln!(
        "{label}: native vs upstream fp32 RGB8 max {max} mean {mean:.3} PSNR {psnr:.1} dB \
         (upstream's own bf16 vs fp32: max {own_max} mean {own_mean:.3} PSNR {own_psnr:.1} dB)"
    );
    let bound = if fp32() { 0.5 } else { 1.5 * own_mean };
    assert!(
        mean <= bound,
        "{label}: mean RGB8 distance {mean:.3} exceeds {bound:.3}"
    );
}

/// The large case: 512x384 at 4x = 2048x1536, six overlapping 1024-px tiles fused on real weights,
/// against the machine-local upstream golden.
#[test]
#[ignore = "needs the upscaler export (IRIS_UPSCALER_DIR) + IRIS_RESTORE_GOLDEN"]
fn real_restoration_matches_upstream() {
    let golden = Weights::from_file(env_dir("IRIS_RESTORE_GOLDEN")).unwrap();
    let t0 = Instant::now();
    let restorer = IrisRestorer::load(&spec()).unwrap();
    eprintln!("load: {:.1}s", t0.elapsed().as_secs_f64());
    let req = TransformRequest {
        image: input(&golden),
        target: TargetSize::Scale(4.0),
        input_sizing: InputSizing::Original,
        ..Default::default()
    };
    let plan = restorer.plan(&req).unwrap();
    assert_eq!(
        plan,
        plan_request(&req, TileGeometry::RELEASE, "iris_3b_restore").unwrap(),
        "the release-geometry preview equals the loaded export's plan"
    );
    assert_eq!(plan.forward_count(), 6);
    let t0 = Instant::now();
    let mut ticks = Vec::new();
    let out = restore_detailed(
        &req.image,
        &plan,
        0.5,
        &req.cancel,
        &mut |p| {
            if let Progress::Step { current, total } = p {
                eprintln!(
                    "tile {current}/{total} at {:.1}s",
                    t0.elapsed().as_secs_f64()
                );
                ticks.push(current);
            }
        },
        &mut |tile, h, w| restorer.velocity(tile, h, w).map_err(Into::into),
    )
    .unwrap();
    eprintln!(
        "restore {}x{} -> {}x{}: {:.1}s",
        plan.source.width,
        plan.source.height,
        plan.output.width,
        plan.output.height,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(ticks, [1, 2, 3, 4, 5, 6]);
    let want = host_f32(golden.require("fp32/restored").unwrap());
    let mut max = 0f32;
    let mut sum = 0f64;
    for (a, b) in out.restored.data.iter().zip(&want) {
        let d = (a - b).abs();
        max = max.max(d);
        sum += d as f64;
    }
    eprintln!(
        "restored float vs upstream fp32: max {max:.3e} mean {:.3e}",
        sum / want.len() as f64
    );
    let image = out.restored.to_rgb8();
    save_png(&image, "iris_restore_large_x4.png");
    assert_parity("large", &image.pixels, &golden);
}

/// The small case (committed upstream reference): 64x48 at 4x = 256x192, enlarged to 1365x1024 (two
/// tiles) and resized back, through the registry's production load path.
#[test]
#[ignore = "needs the upscaler export (IRIS_UPSCALER_DIR)"]
fn real_small_restoration_matches_upstream() {
    let golden =
        Weights::from_file(fixtures().join("iris_restoration_real_small.safetensors")).unwrap();
    let registry = mlx_gen_iris::provider_registry().unwrap();
    let t0 = Instant::now();
    let restorer = registry.load_transform("iris_3b_restore", &spec()).unwrap();
    eprintln!("load: {:.1}s", t0.elapsed().as_secs_f64());
    let req = TransformRequest {
        image: input(&golden),
        target: TargetSize::Scale(4.0),
        ..Default::default()
    };
    restorer.validate(&req).unwrap();
    let t0 = Instant::now();
    let mut steps = 0;
    let image = restorer
        .apply(&req, &mut |p| {
            steps += matches!(p, Progress::Step { .. }) as u32
        })
        .unwrap();
    eprintln!(
        "restore 64x48 -> {}x{}: {:.1}s, {steps} tiles",
        image.width,
        image.height,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!((image.width, image.height, steps), (256, 192, 2));
    save_png(&image, "iris_restore_small_x4.png");
    assert_parity("small", &image.pixels, &golden);
}

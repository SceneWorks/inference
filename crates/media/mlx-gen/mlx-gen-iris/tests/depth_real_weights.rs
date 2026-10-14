//! Real-weight depth checks against the pinned `speridlabs/iris-3b` `depth/` export, on a real
//! photograph that ships in this repository (`crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg`)
//! at upstream's default `max_side` (1024). `#[ignore]`d: they need the 12 GB export and the GPU.
//! Inputs come from the environment:
//!
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (the test reads its `depth/`).
//! * `IRIS_DEPTH_REAL_GOLDEN` — the machine-local full-resolution upstream reference from
//!   `tools/dump_iris_depth_realweight.py` (`real_depth_matches_upstream_full_resolution` only).
//! * `IRIS_OUT` — optional; the predicted map (`.npy`) and its previews (`.png`) are written there.
//!
//! Run each under an external memory guard:
//! `cargo test -p mlx-gen-iris --release --test integration -- --ignored depth_real_weights::`

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::depth::{
    colorize_inferno, near_bright_control_image, plan_depth_input, prepare_depth_input, DepthMap,
    DepthOutput, DepthRequest, DepthResolution, IrisDepthEstimator,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Image, LoadSpec, WeightsSource};
use mlx_rs::Array;

use crate::common::{errors, fixture, host_f32};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn photo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../_vendor/mage_flow/assets/dog.jpg")
}

fn decode_photo() -> Image {
    let rgb = image::open(photo())
        .expect("the repo photo decodes")
        .to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
}

fn load_real() -> Box<dyn IrisDepthEstimator> {
    let dir = env_dir("IRIS_WEIGHTS_DIR").join("depth");
    let start = std::time::Instant::now();
    let model = mlx_gen_iris::depth::load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
    eprintln!("depth load: {:.1}s", start.elapsed().as_secs_f32());
    model
}

fn estimate(model: &dyn IrisDepthEstimator, image: Image) -> DepthOutput {
    let start = std::time::Instant::now();
    let out = model
        .estimate(
            &DepthRequest {
                image,
                resolution: DepthResolution::default(),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    eprintln!(
        "depth estimate: {:.1}s ({}x{} source, {}x{} model, {})",
        start.elapsed().as_secs_f32(),
        out.metadata.source_width,
        out.metadata.source_height,
        out.metadata.model_width,
        out.metadata.model_height,
        out.metadata.compute_dtype
    );
    assert!(out.map.values.iter().all(|v| v.is_finite()));
    out
}

/// `F.avg_pool2d(depth, k)` (floor mode) of an `H × W` map.
fn pool(map: &DepthMap, k: usize) -> Vec<f32> {
    let (w, h) = (map.width as usize, map.height as usize);
    let (pw, ph) = (w / k, h / k);
    let mut out = vec![0f32; pw * ph];
    for (i, o) in out.iter_mut().enumerate() {
        let (py, px) = (i / pw, i % pw);
        let mut sum = 0f64;
        for y in py * k..(py + 1) * k {
            for x in px * k..(px + 1) * k {
                sum += map.values[y * w + x] as f64;
            }
        }
        *o = (sum / (k * k) as f64) as f32;
    }
    out
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa * sbb).sqrt()
}

/// The committed off-host reference (`iris_depth_real_reference.safetensors`): the upstream FP32
/// map pooled over 16×16 blocks. Bounds: the release's bf16 policy vs upstream FP32 on the same
/// photo — see `UPSTREAM.md` (measured numbers).
fn check_against_reference(map: &DepthMap) {
    let reference = fixture("iris_depth_real_reference.safetensors");
    let k: usize = reference.metadata("pool").unwrap().parse().unwrap();
    let height: u32 = reference.metadata("height").unwrap().parse().unwrap();
    let width: u32 = reference.metadata("width").unwrap().parse().unwrap();
    assert_eq!((map.width, map.height), (width, height));
    let want = host_f32(reference.require("pooled").unwrap());
    let got = pool(map, k);
    let (mut max_abs, mut sum) = (0f32, 0f32);
    for (a, b) in got.iter().zip(&want) {
        max_abs = max_abs.max((a - b).abs());
        sum += (a - b).abs();
    }
    let mean = sum / want.len() as f32;
    let r = pearson(&got, &want);
    eprintln!("pooled {k}x{k} vs upstream: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} pearson={r:.6}");
    // Upstream's own bf16-autocast distance on this photo (pooled, recorded in the reference):
    // max 4.6e-2, mean 2.6e-3, pearson 0.99999. The native bf16 maps measured max 4.2e-2 (PIL
    // pixels) / 7.0e-2 (`image`-crate JPEG decode, LSB-different pixels), mean 2.3e-3 / 3.3e-3.
    assert!(r >= 0.9995, "pearson {r}");
    assert!(mean <= 1e-2, "mean |Δ| {mean}");
    assert!(max_abs <= 0.15, "max |Δ| {max_abs}");
}

/// Write the raw map as `.npy` (upstream `scripts/depth.py`'s file) plus both previews.
fn write_evidence(map: &DepthMap, stem: &str) {
    let Ok(out) = std::env::var("IRIS_OUT") else {
        return;
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        map.height, map.width
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut npy = b"\x93NUMPY\x01\x00".to_vec();
    npy.extend((header.len() as u16).to_le_bytes());
    npy.extend(header.as_bytes());
    npy.extend(map.values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(out.join(format!("{stem}.npy")), npy).unwrap();
    for (name, img) in [
        ("near-bright", near_bright_control_image(map)),
        ("inferno", colorize_inferno(map)),
    ] {
        image::RgbImage::from_raw(img.width, img.height, img.pixels)
            .unwrap()
            .save(out.join(format!("{stem}-{name}.png")))
            .unwrap();
    }
}

/// The production path (bf16, GPU) on the PIL-decoded pixels of the photo — upstream's exact input
/// — against the full-resolution upstream FP32 reference: the model input is identical, the raw
/// output and the source-size map within the release bf16 policy's distance.
#[test]
#[ignore = "needs the real depth export (IRIS_WEIGHTS_DIR) + IRIS_DEPTH_REAL_GOLDEN"]
fn real_depth_matches_upstream_full_resolution() {
    let golden = Weights::from_file(env_dir("IRIS_DEPTH_REAL_GOLDEN")).unwrap();
    let pixels = golden.require("image").unwrap();
    let sh = pixels.shape();
    let image = Image {
        width: sh[1] as u32,
        height: sh[0] as u32,
        pixels: host_f32(pixels).iter().map(|&v| v as u8).collect(),
    };
    let plan = plan_depth_input(
        image.width,
        image.height,
        DepthResolution::default(),
        mlx_gen::gen_core::iris::IrisConfig::from_dir(&env_dir("IRIS_WEIGHTS_DIR").join("depth"))
            .unwrap()
            .model
            .patch_size,
    )
    .unwrap();
    let input = prepare_depth_input(&image, &plan).unwrap();
    assert_eq!(
        input,
        host_f32(golden.require("input").unwrap()),
        "model input"
    );
    let model = load_real();
    let out = estimate(model.as_ref(), image);
    let want = golden.require("depth").unwrap();
    let got = Array::from_slice(
        &out.map.values,
        &[out.map.height as i32, out.map.width as i32],
    );
    let (max_abs, peak, mean) = errors(&got, want);
    let r = pearson(&out.map.values, &host_f32(want));
    eprintln!(
        "real depth (bf16) vs upstream fp32: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} \
         peak={peak:.3e} pearson={r:.6}"
    );
    // upstream's own bf16-vs-fp32 on this photo: mean 2.9e-3, max 0.21 (edge pixels); measured
    // mean 2.5e-3, max 0.31 at the same fur edges, pearson 0.99999
    assert!(r >= 0.9995, "pearson {r}");
    assert!(mean <= 1e-2, "mean |Δ| {mean}");
    check_against_reference(&out.map);
    write_evidence(&out.map, "dog-golden-input");
}

/// The off-host form (no golden): the photo decoded by the `image` crate (not PIL — JPEG decoders
/// may differ by an LSB), the production path, the committed pooled reference.
#[test]
#[ignore = "needs the real depth export (IRIS_WEIGHTS_DIR)"]
fn real_depth_matches_the_upstream_reference() {
    let model = load_real();
    let out = estimate(model.as_ref(), decode_photo());
    check_against_reference(&out.map);
    write_evidence(&out.map, "dog");
}

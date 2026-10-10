//! Real-weight depth check on the build's device (`cuda:0` under `--features cuda`) against the
//! pinned `speridlabs/iris-3b` `depth/` export, on a real photograph that ships in this repository
//! (`crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg`) at upstream's default `max_side`
//! (1024), compared with the committed upstream FP32 reference
//! (`mlx-gen-iris/tests/fixtures/iris_depth_real_reference.safetensors`, produced by
//! `crates/media/mlx-gen/tools/dump_iris_depth_realweight.py`). `#[ignore]`d: it needs the 12 GB
//! export and a GPU.
//!
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (the test reads its `depth/`).
//! * `IRIS_OUT` — optional; the predicted map (`.npy`) and its previews (`.png`) are written there.
//! * `IRIS_VRAM_PROBE=1` — sample device VRAM with `nvidia-smi` across load and estimate.
//!
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! depth_real_weights:: --test-threads 1`

use std::path::{Path, PathBuf};

use candle_gen::gen_core::iris::depth::{
    colorize_inferno, near_bright_control_image, DepthMap, DepthRequest, DepthResolution,
};
use candle_gen::gen_core::{Image, LoadSpec, Progress, WeightsSource};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};

use crate::common::{fixture, host_f32};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn decode_photo() -> Image {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg");
    let rgb = image::open(path).expect("the repo photo decodes").to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
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

/// The production path (bf16 on CUDA) on the repo photo vs the committed upstream FP32 reference,
/// pooled over 16×16 source blocks. Bounds: see `UPSTREAM.md` (the MLX twin's measured distance on
/// the same photo).
#[test]
#[ignore = "needs the real depth export (IRIS_WEIGHTS_DIR) and a GPU"]
fn real_depth_matches_the_upstream_reference() {
    let dir = env_dir("IRIS_WEIGHTS_DIR").join("depth");
    let probe_vram =
        std::env::var("IRIS_VRAM_PROBE").is_ok_and(|v| v == "1") && used_mib(probe_gpu()).is_some();
    let mut probe = probe_vram.then(VramProbe::start_rendered);

    let start = std::time::Instant::now();
    let load_phase = probe.as_ref().map(VramProbe::phase);
    let model = candle_gen_iris::depth::load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
    if let (Some(p), Some(phase)) = (probe.as_mut(), load_phase) {
        p.end_load(phase);
    }
    eprintln!(
        "[[IRIS_DEPTH_CANDLE]] device={} load={:.1}s",
        if cfg!(feature = "cuda") {
            "cuda:0"
        } else {
            "cpu"
        },
        start.elapsed().as_secs_f32()
    );

    let start = std::time::Instant::now();
    let gen_phase = probe.as_ref().map(VramProbe::phase);
    let mut steps = Vec::new();
    let out = model
        .estimate(
            &DepthRequest {
                image: decode_photo(),
                resolution: DepthResolution::default(),
                ..Default::default()
            },
            &mut |p| steps.push(p),
        )
        .unwrap();
    if let (Some(p), Some(phase)) = (probe.as_mut(), gen_phase) {
        p.end_gen(phase);
    }
    let m = &out.metadata;
    eprintln!(
        "[[IRIS_DEPTH_CANDLE]] estimate {}x{} (model {}x{}, {}): {:.1}s",
        m.source_width,
        m.source_height,
        m.model_width,
        m.model_height,
        m.compute_dtype,
        start.elapsed().as_secs_f32()
    );
    if let Some(p) = &probe {
        eprintln!("[[IRIS_DEPTH_CANDLE]] vram {}", p.report());
    }
    assert_eq!(
        steps,
        [Progress::Step {
            current: 1,
            total: 1
        }]
    );
    assert!(out.map.values.iter().all(|v| v.is_finite()));

    let reference = fixture("iris_depth_real_reference.safetensors");
    let k: usize = reference.meta("pool").parse().unwrap();
    let height: u32 = reference.meta("height").parse().unwrap();
    let width: u32 = reference.meta("width").parse().unwrap();
    assert_eq!((out.map.width, out.map.height), (width, height));
    let want = host_f32(reference.require("pooled"));
    let got = pool(&out.map, k);
    let (mut max_abs, mut sum) = (0f32, 0f32);
    for (a, b) in got.iter().zip(&want) {
        max_abs = max_abs.max((a - b).abs());
        sum += (a - b).abs();
    }
    let mean = sum / want.len() as f32;
    let r = pearson(&got, &want);
    eprintln!(
        "[[IRIS_DEPTH_CANDLE]] pooled {k}x{k} vs upstream fp32: max|Δ|={max_abs:.3e} \
         mean|Δ|={mean:.3e} pearson={r:.6}"
    );
    write_evidence(&out.map, "dog-candle");
    // Upstream's own bf16-autocast distance on this photo (pooled, recorded in the reference):
    // max 4.6e-2, mean 2.6e-3, pearson 0.99999. The native bf16 maps measured max 4.2e-2 (PIL
    // pixels) / 7.0e-2 (`image`-crate JPEG decode, LSB-different pixels), mean 2.3e-3 / 3.3e-3.
    assert!(r >= 0.9995, "pearson {r}");
    assert!(mean <= 1e-2, "mean |Δ| {mean}");
    assert!(max_abs <= 0.15, "max |Δ| {max_abs}");
}

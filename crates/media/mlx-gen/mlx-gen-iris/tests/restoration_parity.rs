//! Restoration (`iris_3b_restore`, sc-25683) vs upstream `iris3b/downstream/restoration.py` on the
//! committed fixtures (`tools/dump_iris_restoration.py`):
//!
//! * the host-side pixel path (shared with the Candle twin through `gen_core::iris::restoration`):
//!   torch's `scale_factor` bicubic (fractional scale on odd sides, a downscale, a non-dyadic 1.3),
//!   the antialiased bicubic enlarge/shrink, the Gaussian fusion window (32 and the release 1024),
//!   the wavelet colour fix on an image smaller than its largest dilation, tile positions, and
//!   `fit_budget`'s PIL Lanczos (bit-exact, sha256 of the bytes);
//! * the whole restorer on the miniature `upscaler/` export, FP32 on the MLX CPU stream (true f32,
//!   like the oracle): small-image enlarge → tile → resize back, multi-tile fusion, 1×, a fractional
//!   scale, a single tile, a portrait 3×, the budgeted input path and a non-dyadic 1.3× (planned
//!   output size and enlarge branch exact); colour fix on and off.
//!
//! Tolerances: the pixel ops are f32 on both sides in the same order — 1e-5 of peak (measured in
//! the assertion output). The end-to-end float images carry the backbone's FP32 summation-order
//! distance (`dit_parity`: ~3e-5 of peak) scaled by sigma — 1e-4; the RGB8 outputs may differ by
//! one level where a value sits on a rounding boundary.

use std::path::PathBuf;

use mlx_gen::gen_core::iris::restoration::{
    bicubic_aa_resize, bicubic_scale_factor, fit_budget_dims, fit_budget_image, gaussian_window,
    plan_request, restore_detailed, tile_positions, wavelet_color_fix, Dims, Planes, TileGeometry,
    INPUT_BUDGET,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, Error, Image, InputSizing, LoadSpec, Precision, Progress, TargetSize, Transform,
    TransformRequest, WeightsSource,
};
use mlx_gen_iris::IrisRestorer;
use mlx_rs::Array;
use sha2::{Digest, Sha256};

use crate::common::{fixture, fixtures, host_f32, on_cpu, tiny_backbone};

fn golden() -> Weights {
    fixture("iris_restoration_golden.safetensors")
}

fn tiny_export() -> PathBuf {
    fixtures().join("tiny-snapshot/upscaler")
}

fn planes(a: &Array) -> Planes {
    let s = a.shape();
    let (c, h, w) = match s.len() {
        4 => (s[1], s[2], s[3]),
        3 => (s[0], s[1], s[2]),
        _ => panic!("unexpected shape {s:?}"),
    };
    Planes {
        channels: c as usize,
        height: h as usize,
        width: w as usize,
        data: host_f32(a),
    }
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max_abs = 0f32;
    let mut sum = 0f32;
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        assert!(d.is_finite(), "{name}: non-finite ({a} vs {b})");
        max_abs = max_abs.max(d);
        sum += d;
    }
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    eprintln!(
        "{name}: max|Δ|={max_abs:.3e} mean|Δ|={:.3e} bound={:.3e}",
        sum / got.len().max(1) as f32,
        tol * peak
    );
    assert!(
        max_abs <= tol * peak,
        "{name}: max|Δ|={max_abs:.3e} > {tol:.0e}·{peak}"
    );
}

fn formula_rgb(width: u32, height: u32) -> Image {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height as u64 {
        for x in 0..width as u64 {
            for c in 0..3u64 {
                pixels.push(((x * 7 + y * 13 + c * 101 + (x * y) % 97) % 256) as u8);
            }
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn meta(g: &Weights, key: &str) -> serde_json::Value {
    serde_json::from_str(g.metadata(key).expect("fixture metadata")).expect("metadata json")
}

#[test]
fn bicubic_upsample_matches_torch_scale_factor() {
    let g = golden();
    let cases = meta(&g, "bicubic");
    for (name, scale) in cases.as_object().unwrap() {
        let input = planes(g.require(&format!("ops/bicubic/{name}/in")).unwrap());
        let want = planes(g.require(&format!("ops/bicubic/{name}/out")).unwrap());
        let got = bicubic_scale_factor(&input, scale.as_f64().unwrap());
        assert_eq!((got.height, got.width), (want.height, want.width), "{name}");
        close(&format!("bicubic {name}"), &got.data, &want.data, 1e-5);
    }
}

#[test]
fn antialiased_resize_matches_torch() {
    let g = golden();
    for name in ["up", "down", "width_only", "down_odd"] {
        let input = planes(g.require(&format!("ops/aa/{name}/in")).unwrap());
        let want = planes(g.require(&format!("ops/aa/{name}/out")).unwrap());
        let got = bicubic_aa_resize(&input, want.height, want.width);
        close(&format!("aa {name}"), &got.data, &want.data, 1e-5);
    }
}

#[test]
fn fusion_window_and_tile_grid_match_upstream() {
    let g = golden();
    close(
        "window 32",
        &gaussian_window(32),
        &host_f32(g.require("ops/window/32").unwrap()),
        1e-6,
    );
    let big = gaussian_window(1024);
    let row: Vec<f32> = big[512 * 1024..513 * 1024].to_vec();
    close(
        "window 1024 row 512",
        &row,
        &host_f32(g.require("ops/window/1024/centre_row").unwrap()),
        1e-6,
    );
    let col: Vec<f32> = (0..1024).map(|r| big[r * 1024 + 511]).collect();
    close(
        "window 1024 col 511",
        &col,
        &host_f32(g.require("ops/window/1024/centre_col").unwrap()),
        1e-6,
    );
    let sub: Vec<f32> = (0..1024)
        .step_by(37)
        .flat_map(|r| (0..1024).step_by(41).map(move |c| (r, c)))
        .map(|(r, c)| big[r * 1024 + c])
        .collect();
    close(
        "window 1024 subsample",
        &sub,
        &host_f32(g.require("ops/window/1024/sub").unwrap()),
        1e-6,
    );
    for case in meta(&g, "tile_positions").as_array().unwrap() {
        let v: Vec<u64> = case.as_array().unwrap()[..3]
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        let want: Vec<u32> = case[3]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            tile_positions(v[0] as u32, v[1] as u32, v[2] as u32),
            want,
            "{case}"
        );
    }
}

#[test]
fn wavelet_colour_fix_matches_upstream_at_the_boundary() {
    let g = golden();
    let content = planes(g.require("ops/wavelet/content").unwrap());
    let reference = planes(g.require("ops/wavelet/reference").unwrap());
    let want = planes(g.require("ops/wavelet/out").unwrap());
    let got = wavelet_color_fix(&content, &reference).unwrap();
    close("wavelet colour fix", &got.data, &want.data, 1e-5);
}

#[test]
fn fit_budget_is_pil_lanczos_bit_exact() {
    let g = golden();
    for case in meta(&g, "fit_budget").as_array().unwrap() {
        let size = |k: &str| {
            let v = case[k].as_array().unwrap();
            Dims::new(v[0].as_u64().unwrap() as u32, v[1].as_u64().unwrap() as u32)
        };
        let (input, output) = (size("input"), size("output"));
        let to = fit_budget_dims(input, INPUT_BUDGET);
        assert_eq!(to, output, "budget dims of {input:?}");
        let resized = fit_budget_image(&formula_rgb(input.width, input.height), to).unwrap();
        let digest = Sha256::digest(&resized.pixels);
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, case["sha256"].as_str().unwrap(), "pixels of {input:?}");
    }
}

fn fp32_restorer() -> IrisRestorer {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_export()));
    spec.precision = Precision::Fp32;
    IrisRestorer::load(&spec).expect("the miniature export loads")
}

#[test]
fn restoration_matches_upstream_end_to_end_in_fp32() {
    on_cpu(|| {
        let restorer = fp32_restorer();
        let g = golden();
        for case in meta(&g, "cases").as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let dims = case["input"].as_array().unwrap();
            let (w, h) = (
                dims[0].as_u64().unwrap() as u32,
                dims[1].as_u64().unwrap() as u32,
            );
            let pixels: Vec<u8> = host_f32(g.require(&format!("e2e/{name}/input")).unwrap())
                .iter()
                .map(|&v| v as u8)
                .collect();
            let budgeted = case["budgeted"].as_bool().unwrap();
            let color_fix = case["color_fix"].as_bool().unwrap();
            let req = TransformRequest {
                image: Image {
                    width: w,
                    height: h,
                    pixels,
                },
                target: TargetSize::Scale(case["scale"].as_f64().unwrap() as f32),
                input_sizing: if budgeted {
                    InputSizing::Budgeted
                } else {
                    InputSizing::Original
                },
                color_fix: Some(color_fix),
                ..Default::default()
            };
            let plan = restorer.plan(&req).unwrap();
            let out_dims = case["output"].as_array().unwrap();
            assert_eq!(
                (plan.output.width as u64, plan.output.height as u64),
                (out_dims[0].as_u64().unwrap(), out_dims[1].as_u64().unwrap()),
                "{name}: planned output geometry"
            );
            assert_eq!(
                plan.enlarged,
                case["enlarged"].as_bool().unwrap(),
                "{name}: upstream's `min(out_size) <= tile` enlarge branch"
            );
            let mut steps = 0;
            let out = restore_detailed(
                &req.image,
                &plan,
                0.5,
                &req.cancel,
                &mut |p| steps += matches!(p, Progress::Step { .. }) as usize,
                &mut |tile, th, tw| restorer.velocity(tile, th, tw).map_err(Into::into),
            )
            .unwrap();
            assert_eq!(
                steps,
                plan.forward_count(),
                "{name}: one progress step per tile"
            );
            let tiled = planes(g.require(&format!("e2e/{name}/tiled")).unwrap());
            close(
                &format!("{name} fused tiles"),
                &out.fused.data,
                &tiled.data,
                1e-4,
            );
            if color_fix {
                let fixed = planes(g.require(&format!("e2e/{name}/fixed")).unwrap());
                close(
                    &format!("{name} colour-fixed"),
                    &out.restored.data,
                    &fixed.data,
                    1e-4,
                );
            }
            // The full Transform path (budget, quantization) against upstream's RGB8 output.
            let image = restorer.apply(&req, &mut |_| {}).unwrap();
            let want: Vec<u8> = host_f32(g.require(&format!("e2e/{name}/output")).unwrap())
                .iter()
                .map(|&v| v as u8)
                .collect();
            assert_eq!(
                (image.width, image.height),
                (plan.output.width, plan.output.height)
            );
            let max = image
                .pixels
                .iter()
                .zip(&want)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let off = image
                .pixels
                .iter()
                .zip(&want)
                .filter(|(a, b)| a != b)
                .count();
            eprintln!("{name}: RGB8 max|Δ|={max} differing={off}/{}", want.len());
            assert!(max <= 1, "{name}: RGB8 differs by {max}");
        }
    });
}

#[test]
fn release_geometry_preview_is_backend_free() {
    // What SceneWorks shows before running: the planner with the release geometry, no weights.
    let req = TransformRequest {
        image: formula_rgb(512, 384),
        ..Default::default()
    };
    let plan = plan_request(&req, TileGeometry::RELEASE, "iris_3b_restore").unwrap();
    assert_eq!(plan.output, Dims::new(2048, 1536));
    assert_eq!(plan.forward_count(), 6);
}

#[test]
fn cancel_between_tiles_is_typed_and_yields_no_image() {
    on_cpu(|| {
        let restorer = fp32_restorer();
        let cancel = CancelFlag::new();
        let req = TransformRequest {
            image: formula_rgb(40, 24),
            target: TargetSize::Scale(4.0),
            input_sizing: InputSizing::Original,
            cancel: cancel.clone(),
            ..Default::default()
        };
        assert!(restorer.plan(&req).unwrap().forward_count() > 2);
        let mut steps = 0;
        let err = restorer
            .apply(&req, &mut |p| {
                if matches!(p, Progress::Step { .. }) {
                    steps += 1;
                    if steps == 2 {
                        cancel.cancel();
                    }
                }
            })
            .unwrap_err();
        assert!(matches!(err, mlx_gen::gen_core::Error::Canceled), "{err:?}");
        assert_eq!(steps, 2);
    });
}

#[test]
fn wrong_task_artifacts_are_refused() {
    // The generation backbone (no `task` section).
    let err = IrisRestorer::load(&LoadSpec::new(WeightsSource::Dir(tiny_backbone())))
        .err()
        .expect("a generation backbone is not a restoration export");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("generation")),
        "{err}"
    );
    // A depth export (`task.name: depth`).
    let dir = tempfile::tempdir().unwrap();
    let config = std::fs::read_to_string(tiny_export().join("config.yaml")).unwrap();
    let depth = config.replace("name: restoration", "name: depth");
    assert_ne!(depth, config);
    std::fs::write(dir.path().join("config.yaml"), depth).unwrap();
    let err = IrisRestorer::load(&LoadSpec::new(WeightsSource::Dir(dir.path().into())))
        .err()
        .expect("a depth export is not a restoration export");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("depth")),
        "{err}"
    );
    // The text encoder as a component (restoration never loads it).
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_export()));
    spec.components.insert(
        "text_encoder".into(),
        WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
    );
    let err = IrisRestorer::load(&spec).err().expect("no text encoder");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("text_encoder")),
        "{err}"
    );
    // The text encoder snapshot staged as the export itself.
    let err = IrisRestorer::load(&LoadSpec::new(WeightsSource::Dir(
        fixtures().join("tiny-snapshot/text_encoder"),
    )))
    .err()
    .expect("a text encoder is not a restoration export");
    assert!(err.to_string().contains("config.yaml"), "{err}");
}

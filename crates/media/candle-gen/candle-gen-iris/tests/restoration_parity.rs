//! Restoration (`iris_3b_restore`, sc-25683) vs upstream `iris3b/downstream/restoration.py` on the
//! MLX twin's committed fixtures (`tools/dump_iris_restoration.py`): the whole restorer on the
//! miniature `upscaler/` export, FP32 on the Candle CPU lane (true f32, like the oracle) — small-image
//! enlarge → tile → resize back, multi-tile Gaussian fusion, 1×, a fractional scale, a single tile,
//! a portrait 3×, the budgeted input path and a non-dyadic 1.3× (planned output size and enlarge
//! branch exact); colour fix on and off. The host-side pixel ops are shared with the MLX twin
//! (`gen_core::iris::restoration`) and held op by op there.
//!
//! Tolerances (the MLX twin's): the float images carry the backbone's FP32 summation-order distance
//! (`dit_parity`) scaled by sigma — 1e-4 of peak; the RGB8 outputs may differ by one level where a
//! value sits on a rounding boundary.

use std::path::PathBuf;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::iris::restoration::{restore_detailed, Planes};
use candle_gen::gen_core::{
    CancelFlag, Image, InputSizing, LoadSpec, Precision, Progress, TargetSize, Transform,
    TransformRequest, WeightsSource,
};
use candle_gen::CandleError as Error;
use candle_gen_iris::IrisRestorer;

use crate::common::{fixture, fixtures, host_f32, tiny_backbone};

fn tiny_export() -> PathBuf {
    fixtures().join("tiny-snapshot/upscaler")
}

fn planes(a: &Tensor) -> Planes {
    let s = a.dims();
    let (c, h, w) = match s.len() {
        4 => (s[1], s[2], s[3]),
        3 => (s[0], s[1], s[2]),
        _ => panic!("unexpected shape {s:?}"),
    };
    Planes {
        channels: c,
        height: h,
        width: w,
        data: host_f32(a),
    }
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        assert!(d.is_finite(), "{name}: non-finite ({a} vs {b})");
        max_abs = max_abs.max(d);
    }
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    eprintln!("{name}: max|Δ|={max_abs:.3e} bound={:.3e}", tol * peak);
    assert!(
        max_abs <= tol * peak,
        "{name}: max|Δ|={max_abs:.3e} > {tol:.0e}·{peak}"
    );
}

fn fp32_restorer() -> IrisRestorer {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_export()));
    spec.precision = Precision::Fp32;
    IrisRestorer::load_on(&spec, &Device::Cpu).expect("the miniature export loads")
}

fn rgb8(t: &Tensor) -> Vec<u8> {
    host_f32(t).iter().map(|&v| v as u8).collect()
}

#[test]
fn restoration_matches_upstream_end_to_end_in_fp32() {
    let restorer = fp32_restorer();
    let g = fixture("iris_restoration_golden.safetensors");
    let cases: serde_json::Value = serde_json::from_str(g.meta("cases")).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let dims = case["input"].as_array().unwrap();
        let (w, h) = (
            dims[0].as_u64().unwrap() as u32,
            dims[1].as_u64().unwrap() as u32,
        );
        let color_fix = case["color_fix"].as_bool().unwrap();
        let req = TransformRequest {
            image: Image {
                width: w,
                height: h,
                pixels: rgb8(g.require(&format!("e2e/{name}/input"))),
            },
            target: TargetSize::Scale(case["scale"].as_f64().unwrap() as f32),
            input_sizing: if case["budgeted"].as_bool().unwrap() {
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
        let tiled = planes(g.require(&format!("e2e/{name}/tiled")));
        close(
            &format!("{name} fused tiles"),
            &out.fused.data,
            &tiled.data,
            1e-4,
        );
        if color_fix {
            let fixed = planes(g.require(&format!("e2e/{name}/fixed")));
            close(
                &format!("{name} colour-fixed"),
                &out.restored.data,
                &fixed.data,
                1e-4,
            );
        }
        let image = restorer.apply(&req, &mut |_| {}).unwrap();
        let want = rgb8(g.require(&format!("e2e/{name}/output")));
        let out_dims = case["output"].as_array().unwrap();
        assert_eq!(
            (image.width as u64, image.height as u64),
            (out_dims[0].as_u64().unwrap(), out_dims[1].as_u64().unwrap()),
            "{name}: output geometry"
        );
        let max = image
            .pixels
            .iter()
            .zip(&want)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        eprintln!("{name}: RGB8 max|Δ|={max}");
        assert!(max <= 1, "{name}: RGB8 differs by {max}");
    }
}

#[test]
fn cancel_between_tiles_is_typed_and_yields_no_image() {
    let restorer = fp32_restorer();
    let cancel = CancelFlag::new();
    let req = TransformRequest {
        image: Image {
            width: 40,
            height: 24,
            pixels: vec![100; 40 * 24 * 3],
        },
        target: TargetSize::Scale(4.0),
        input_sizing: InputSizing::Original,
        cancel: cancel.clone(),
        ..Default::default()
    };
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
    assert!(
        matches!(err, candle_gen::gen_core::Error::Canceled),
        "{err:?}"
    );
    assert_eq!(steps, 2);
}

#[test]
fn wrong_task_artifacts_are_refused() {
    let load = |spec: &LoadSpec| IrisRestorer::load_on(spec, &Device::Cpu).err();
    let err = load(&LoadSpec::new(WeightsSource::Dir(tiny_backbone()))).expect("generation");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("generation")),
        "{err}"
    );
    let dir = tempfile::tempdir().unwrap();
    let config = std::fs::read_to_string(tiny_export().join("config.yaml")).unwrap();
    std::fs::write(
        dir.path().join("config.yaml"),
        config.replace("name: restoration", "name: depth"),
    )
    .unwrap();
    let err = load(&LoadSpec::new(WeightsSource::Dir(dir.path().into()))).expect("depth");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("depth")),
        "{err}"
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_export()));
    spec.components.insert(
        "text_encoder".into(),
        WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
    );
    let err = load(&spec).expect("text encoder component");
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("text_encoder")),
        "{err}"
    );
}

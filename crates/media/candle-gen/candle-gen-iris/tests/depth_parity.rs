//! The depth task vs upstream `iris3b.downstream.depth.DepthPredictor` on the miniature export —
//! the same fixtures (`tiny-depth`, `iris_depth_golden.safetensors`, `tools/dump_iris_depth.py`) the
//! MLX twin's `depth_parity` reads, case by case: odd / small / non-square sources, capped vs native
//! max side, the round-half-even size law, no-resize sources.
//!
//! Stages and tolerances (of peak):
//! * model input (PIL Lanczos + `[-1, 1]`, shared `gen_core` host code): **exact**;
//! * raw `[1, 1, h, w]` output, FP32 on the Candle CPU: 1e-4 (summation order only);
//! * `[H, W]` map after the bilinear resize back: 1e-4;
//! * bf16 release policy (bf16 operands, f32 accumulation on CPU): 5e-2.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::iris::depth::{
    plan_depth_input, prepare_depth_input, DepthRequest, DepthResolution, IrisDepthEstimator,
};
use candle_gen::gen_core::iris::downstream::TaskExport;
use candle_gen::gen_core::iris::{IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{Image, LoadSpec, Progress, WeightsSource};
use candle_gen_iris::depth::{load, load_depth_export, DEPTH_MODEL_ID};

use crate::common::{assert_close, fixture, fixtures, host_f32, tiny_backbone, Fixture};

fn export() -> TaskExport {
    TaskExport::from_dir(
        &fixtures().join("tiny-depth"),
        IrisTask::Depth,
        DEPTH_MODEL_ID,
    )
    .unwrap()
}

fn case(golden: &Fixture, name: &str) -> (Image, DepthResolution) {
    let img = golden.require(&format!("{name}/image"));
    let (h, w, _) = img.dims3().unwrap();
    let pixels = img.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let max_side: u32 = golden.meta(&format!("{name}/max_side")).parse().unwrap();
    let resolution = if max_side == 0 {
        DepthResolution::Native
    } else {
        DepthResolution::Capped(max_side)
    };
    (
        Image {
            width: w as u32,
            height: h as u32,
            pixels,
        },
        resolution,
    )
}

#[test]
fn depth_matches_upstream_stage_by_stage() {
    let golden = fixture("iris_depth_golden.safetensors");
    let model = load_depth_export(export(), DType::F32, &Device::Cpu).unwrap();
    let patch = model.export().config.model.patch_size;
    let names: Vec<String> = serde_json::from_str(golden.meta("cases")).unwrap();
    for name in &names {
        let (image, resolution) = case(&golden, name);
        let plan = plan_depth_input(image.width, image.height, resolution, patch).unwrap();
        let want_input = golden.require(&format!("{name}/input"));
        let (_, _, h, w) = want_input.dims4().unwrap();
        assert_eq!(
            (plan.model_height as usize, plan.model_width as usize),
            (h, w),
            "{name}: size law"
        );
        let input = prepare_depth_input(&image, &plan).unwrap();
        assert_eq!(input, host_f32(want_input), "{name}: model input");
        let raw = model
            .forward(&Tensor::from_slice(&input, (1, 3, h, w), &Device::Cpu).unwrap())
            .unwrap();
        assert_close(
            &format!("{name}/raw"),
            &raw,
            golden.require(&format!("{name}/raw")),
            1e-4,
        );
        let mut progress = Vec::new();
        let out = model
            .estimate(
                &DepthRequest {
                    image: image.clone(),
                    resolution,
                    ..Default::default()
                },
                &mut |p| progress.push(p),
            )
            .unwrap();
        assert_eq!(
            progress,
            [Progress::Step {
                current: 1,
                total: 1
            }]
        );
        let got = Tensor::from_vec(
            out.map.values.clone(),
            (out.map.height as usize, out.map.width as usize),
            &Device::Cpu,
        )
        .unwrap();
        assert_close(
            &format!("{name}/depth"),
            &got,
            golden.require(&format!("{name}/depth")),
            1e-4,
        );
        let m = &out.metadata;
        assert_eq!(m.model_id, DEPTH_MODEL_ID);
        assert_eq!((m.backend, m.compute_dtype), ("candle", "float32"));
        assert_eq!(m.resolution, resolution);
        assert_eq!(m.output_resample == "bilinear", plan.resized(), "{name}");
    }
}

#[test]
fn bf16_release_policy_tracks_upstream() {
    let golden = fixture("iris_depth_golden.safetensors");
    let model = load_depth_export(export(), DType::BF16, &Device::Cpu).unwrap();
    let raw = model.forward(golden.require("native_odd/input")).unwrap();
    assert_close("bf16 raw", &raw, golden.require("native_odd/raw"), 5e-2);
}

#[test]
fn the_depth_task_resolves_only_its_own_closure() {
    let refusal = |spec: &LoadSpec| match load(spec) {
        Err(candle_gen::gen_core::Error::Unsupported(m)) => m,
        Err(other) => panic!("expected a typed refusal, got {other}"),
        Ok(_) => panic!("expected a typed refusal, got a model"),
    };
    let dir = fixtures().join("tiny-depth");
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
    );
    assert!(refusal(&spec).contains("text_encoder"));
    let spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    assert!(refusal(&spec).contains("generation checkpoint"));
    let model = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
    assert_eq!(model.backend(), "candle");
}

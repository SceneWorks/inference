//! The depth task vs upstream `iris3b.downstream.depth.DepthPredictor` on the miniature export
//! (`tests/fixtures/tiny-depth`, `tools/dump_iris_depth.py`), case by case: odd / small / non-square
//! sources, capped vs native max side, the round-half-even size law, no-resize sources.
//!
//! Stages and tolerances (of peak):
//! * model input (PIL Lanczos + `[-1, 1]`): **exact** — `gen_core::imageops` is PIL's fixed-point
//!   resampler and the mapping is the same f32 op sequence;
//! * raw `[1, 1, h, w]` output, FP32 on the MLX CPU stream: 1e-4 (summation order only, the
//!   generation `dit_parity` gate);
//! * `[H, W]` map after the bilinear resize back: 1e-4;
//! * presentation adapters: `colorize_inferno` vs upstream `colorize` within ≤ 4 RGB levels.
//!
//! The bf16 release policy (upstream's CUDA autocast) is held to 5e-2 on the GPU stream.

use mlx_gen::gen_core::iris::depth::{
    colorize_inferno, plan_depth_input, prepare_depth_input, DepthMap, DepthRequest,
    DepthResolution, IrisDepthEstimator,
};
use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::{IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::weights::Weights;
use mlx_gen::{Image, LoadSpec, Precision, Progress, WeightsSource};
use mlx_gen_iris::depth::{load, load_depth, load_depth_export, DEPTH_MODEL_ID};
use mlx_rs::{Array, Dtype};

use crate::common::{assert_close, fixture, fixtures, host_f32, on_cpu, tiny_backbone};

fn export_dir() -> std::path::PathBuf {
    fixtures().join("tiny-depth")
}

struct Case {
    name: String,
    image: Image,
    resolution: DepthResolution,
}

fn cases(golden: &Weights) -> Vec<Case> {
    let names: Vec<String> = serde_json::from_str(golden.metadata("cases").unwrap()).unwrap();
    names
        .into_iter()
        .map(|name| {
            let img = golden.require(&format!("{name}/image")).unwrap();
            let sh = img.shape();
            let pixels: Vec<u8> = host_f32(img).iter().map(|&v| v as u8).collect();
            let max_side: u32 = golden
                .metadata(&format!("{name}/max_side"))
                .unwrap()
                .parse()
                .unwrap();
            Case {
                image: Image {
                    width: sh[1] as u32,
                    height: sh[0] as u32,
                    pixels,
                },
                resolution: if max_side == 0 {
                    DepthResolution::Native
                } else {
                    DepthResolution::Capped(max_side)
                },
                name,
            }
        })
        .collect()
}

fn map_array(map: &DepthMap) -> Array {
    Array::from_slice(&map.values, &[map.height as i32, map.width as i32])
}

#[test]
fn depth_matches_upstream_stage_by_stage() {
    on_cpu(|| {
        let golden = fixture("iris_depth_golden.safetensors");
        let spec = {
            let mut spec = LoadSpec::new(WeightsSource::Dir(export_dir()));
            spec.precision = Precision::Fp32;
            spec
        };
        let model = load_depth(&spec).unwrap();
        let patch = model.export().config.model.patch_size;
        for case in cases(&golden) {
            let name = &case.name;
            // preprocessing: exact
            let plan =
                plan_depth_input(case.image.width, case.image.height, case.resolution, patch)
                    .unwrap();
            let want_input = golden.require(&format!("{name}/input")).unwrap();
            let shape = want_input.shape();
            assert_eq!(
                (plan.model_height as i32, plan.model_width as i32),
                (shape[2], shape[3]),
                "{name}: size law"
            );
            let input = prepare_depth_input(&case.image, &plan).unwrap();
            assert_eq!(input, host_f32(want_input), "{name}: model input");
            // the forward
            let raw = model.forward(&Array::from_slice(&input, shape)).unwrap();
            assert_close(
                &format!("{name}/raw"),
                &raw,
                golden.require(&format!("{name}/raw")).unwrap(),
                1e-4,
            );
            // end to end at the source size
            let mut progress = Vec::new();
            let out = model
                .estimate(
                    &DepthRequest {
                        image: case.image.clone(),
                        resolution: case.resolution,
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
            assert_eq!(
                (out.map.width, out.map.height),
                (case.image.width, case.image.height)
            );
            let want = golden.require(&format!("{name}/depth")).unwrap();
            assert_close(&format!("{name}/depth"), &map_array(&out.map), want, 1e-4);
            let m = &out.metadata;
            assert_eq!(m.model_id, DEPTH_MODEL_ID);
            assert_eq!((m.backend, m.compute_dtype), ("mlx", "float32"));
            assert_eq!(m.resolution, case.resolution);
            assert_eq!(m.input_resample == "lanczos", plan.resized(), "{name}");
            assert_eq!(m.config_sha256, model.export().config_sha256);

            // upstream `colorize` on upstream's own map: the adapter is exact up to the f32
            // percentile arithmetic, which can move a pixel by one colormap index
            let upstream_map = DepthMap {
                width: case.image.width,
                height: case.image.height,
                values: host_f32(want),
            };
            let ours = colorize_inferno(&upstream_map);
            let theirs: Vec<u8> = host_f32(golden.require(&format!("{name}/colorize")).unwrap())
                .iter()
                .map(|&v| v as u8)
                .collect();
            let worst = ours
                .pixels
                .iter()
                .zip(&theirs)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            assert!(worst <= 4, "{name}: colorize differs by {worst}");
        }
    });
}

#[test]
fn bf16_release_policy_tracks_upstream() {
    let golden = fixture("iris_depth_golden.safetensors");
    let export = TaskExport::from_dir(&export_dir(), IrisTask::Depth, DEPTH_MODEL_ID).unwrap();
    let model = load_depth_export(export, Dtype::Bfloat16).unwrap();
    let input = golden.require("native_odd/input").unwrap();
    let raw = model.forward(input).unwrap();
    assert_close(
        "bf16 raw",
        &raw,
        golden.require("native_odd/raw").unwrap(),
        5e-2,
    );
}

#[test]
fn the_depth_task_resolves_only_its_own_closure() {
    let refusal = |spec: &LoadSpec| match load(spec) {
        Err(mlx_gen::gen_core::Error::Unsupported(m)) => m,
        Err(other) => panic!("expected a typed refusal, got {other}"),
        Ok(_) => panic!("expected a typed refusal, got a model"),
    };
    // a text encoder handed to the depth task
    let mut spec = LoadSpec::new(WeightsSource::Dir(export_dir()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
    );
    assert!(refusal(&spec).contains("text_encoder"));
    // the generation checkpoint handed to the depth task
    let spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    assert!(refusal(&spec).contains("generation checkpoint"));
    // and the depth export loads (no text encoder anywhere in its closure)
    let model = load(&LoadSpec::new(WeightsSource::Dir(export_dir()))).unwrap();
    assert_eq!(model.backend(), "mlx");
}

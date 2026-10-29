//! sc-25681: every public upstream generation control vs upstream `iris3b.sampling.generate` on the
//! miniature snapshot (`tools/dump_iris_controls.py`), table-driven over the oracle's case list —
//! solver order 1, shift 2, CFG interval (0.3, 0.8), a negative prompt, CFG off, a portrait canvas,
//! a two-prompt batch, the `prediction: x` checkpoint reading, and (sc-25690) a square canvas, a
//! second (2:1) landscape canvas beside `base`'s 3:2 one, and a changed step count (8 for 5) — each
//! from the oracle's injected noise. Every control's golden is also checked to sit measurably away from the release-controls
//! `base` golden, so a native path that accepted a control and ignored it could not pass.
//!
//! Tolerance: the `e2e_parity` bound (6e-2 of peak) — the backbone and solver run FP32 on the MLX
//! CPU stream, the text tower bf16 on both sides (1–2 bf16 ulps apart, amplified by CFG).

use gen_core_iris::{GenerationParams, IrisConfig};
use mlx_gen::{CancelFlag, GenerationRequest, PreviewSink};
use mlx_gen_iris::{denoise, encode, load_backbone, IrisTextEncoder};
use mlx_rs::Dtype;
use serde_json::Value;

use crate::common::{
    assert_close, cpu_process, errors, fixture, on_cpu, tiny_backbone, tiny_config,
    tiny_text_encoder,
};

const TOL: f32 = 6e-2;

/// The native request for one oracle case — every upstream argument through its request field.
pub fn case_request(case: &Value, config: &IrisConfig) -> GenerationRequest {
    let prompts: Vec<String> = case["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap().to_owned())
        .collect();
    let (prompt, prompt_batch) = if prompts.len() == 1 {
        (prompts[0].clone(), Vec::new())
    } else {
        (String::new(), prompts)
    };
    let interval = case["cfg_interval"].as_array().unwrap();
    let (lo, hi) = (
        interval[0].as_f64().unwrap() as f32,
        interval[1].as_f64().unwrap() as f32,
    );
    let shift = case["shift"].as_f64().unwrap();
    let negative = case["negative_prompt"].as_str().unwrap();
    GenerationRequest {
        prompt,
        prompt_batch,
        width: case["width"].as_u64().unwrap() as u32,
        height: case["height"].as_u64().unwrap() as u32,
        steps: Some(case["steps"].as_u64().unwrap() as u32),
        sampler: match case["order"].as_u64().unwrap() {
            1 => Some("euler".into()),
            _ => Some("dpmpp_2m".into()),
        },
        scheduler_shift: (shift != config.flow.shift).then_some(shift as f32),
        cfg_interval: ((lo, hi) != (0.0, 1.0)).then_some((lo, hi)),
        guidance: Some(case["cfg_scale"].as_f64().unwrap() as f32),
        negative_prompt: (!negative.is_empty()).then(|| negative.to_owned()),
        seed: Some(0),
        ..Default::default()
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn every_control_matches_upstream_on_the_miniature() {
    cpu_process(
        "controls_parity::every_control_matches_upstream_on_the_miniature",
        || {
            every_control_matches_upstream_on_the_miniature_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED controls_parity::every_control_matches_upstream_on_the_miniature");
        },
    );
}

fn every_control_matches_upstream_on_the_miniature_body() {
    on_cpu(|| {
        let base_config = tiny_config();
        let te = IrisTextEncoder::load(&tiny_text_encoder(), &base_config.text_encoder).unwrap();
        let dit = load_backbone(&tiny_backbone(), &base_config, Dtype::Float32).unwrap();
        let golden = fixture("iris_controls_golden.safetensors");
        let cases: Vec<Value> = serde_json::from_str(golden.metadata("cases").unwrap()).unwrap();
        let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "base",
                "order1",
                "shift2",
                "interval",
                "negative",
                "cfg_off",
                "portrait",
                "batch",
                "prediction_x",
                "square",
                "landscape",
                "steps"
            ],
            "the oracle's case table"
        );
        // The canvases the table must span (epic AT2): square, portrait and two landscapes, plus a
        // step count other than `base`'s — read from the oracle's own case rows, so a regenerated
        // table that lost one fails here rather than passing on fewer cases.
        let canvas = |name: &str| {
            let case = cases.iter().find(|c| c["name"] == name).unwrap();
            (
                case["width"].as_u64().unwrap(),
                case["height"].as_u64().unwrap(),
                case["steps"].as_u64().unwrap(),
            )
        };
        assert_eq!(canvas("base"), (12, 8, 5));
        assert_eq!(canvas("square"), (8, 8, 5));
        assert_eq!(canvas("portrait"), (8, 12, 5));
        assert_eq!(canvas("landscape"), (16, 8, 5));
        assert_eq!(canvas("steps"), (12, 8, 8));
        let base_image = golden.require("base/image").unwrap();
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let mut config = base_config.clone();
            config.flow.prediction = case["prediction"].as_str().unwrap().to_owned();
            config.validate_supported().unwrap();
            let req = case_request(case, &config);
            gen_core_iris::validate_generation_request("iris_3b", &req, &config).unwrap();
            let params = GenerationParams::resolve(&req, 0, &config).unwrap();
            assert_eq!(
                (
                    params.width as u64,
                    params.height as u64,
                    params.steps as u64
                ),
                canvas(name),
                "{name}: the resolved canvas and step count are the oracle's"
            );
            let conditioning = encode(&te, &params).unwrap();
            assert_eq!(conditioning.cond.len(), params.prompts.len());
            let mut steps = Vec::new();
            let image = denoise(
                &dit,
                &config.flow,
                &conditioning,
                golden.require(&format!("{name}/noise")).unwrap(),
                &params,
                &CancelFlag::new(),
                |i| steps.push(i),
                &PreviewSink::default(),
            )
            .unwrap();
            assert_eq!(steps, (1..=params.steps).collect::<Vec<_>>(), "{name}");
            let want = golden.require(&format!("{name}/image")).unwrap();
            assert_close(&format!("controls {name}"), &image, want, TOL);
            if name != "base" && want.shape() == base_image.shape() {
                let (moved, _, _) = errors(want, base_image);
                assert!(
                    moved > 2.0 * TOL,
                    "{name}: the golden moves only {moved:.3e} from base — not discriminative"
                );
            }
        }
    });
}

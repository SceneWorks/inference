//! sc-25681: every public upstream generation control vs upstream `iris3b.sampling.generate` on the
//! miniature snapshot (`tools/dump_iris_controls.py`), table-driven over the oracle's case list —
//! solver order 1, shift 2, CFG interval (0.3, 0.8), a negative prompt, CFG off, a portrait canvas,
//! a two-prompt batch, and the `prediction: x` checkpoint reading — each from the oracle's injected
//! noise. Every control's golden is also checked to sit measurably away from the release-controls
//! `base` golden, so a native path that accepted a control and ignored it could not pass.
//!
//! Reference and tolerance: the Candle CPU lane runs the text tower in f32 (no CPU half GEMM, see
//! `text_parity`), so each case is held to the oracle's **fp32-tower** render of the same case
//! (`<case>/image_f32_tower`) at 2e-4 of peak (measured ≤ 5.8e-5, `prediction_x`) — the FP32 backbone + solver + f32 tower against
//! upstream's own. (Against the bf16-tower golden the distance is upstream's own bf16-vs-fp32 tower
//! distance, measured by the oracle at up to 7.3e-2 on `prediction_x`.) The MLX twin's
//! `controls_parity`, case for case.

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::{GenerationParams, IrisConfig};
use candle_gen::gen_core::{CancelFlag, GenerationRequest, PreviewSink};
use candle_gen_iris::{denoise, encode, load_backbone, IrisTextEncoder};
use serde_json::Value;

use crate::common::{
    assert_close, cpu, errors, fixture, tiny_backbone, tiny_config, tiny_text_encoder,
};

const TOL: f32 = 2e-4;
/// The discrimination margin is measured against the bf16 goldens' spread (the MLX bound).
const SPREAD: f32 = 6e-2;

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
fn every_control_matches_upstream_on_the_miniature() {
    {
        let base_config = tiny_config();
        let te =
            IrisTextEncoder::load(&tiny_text_encoder(), &base_config.text_encoder, &cpu()).unwrap();
        let dit = load_backbone(&tiny_backbone(), &base_config, DType::F32, &cpu()).unwrap();
        let golden = fixture("iris_controls_golden.safetensors");
        let cases: Vec<Value> = serde_json::from_str(golden.meta("cases")).unwrap();
        assert_eq!(cases.len(), 9, "the oracle's case table");
        let base_image = golden.require("base/image");
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let mut config = base_config.clone();
            config.flow.prediction = case["prediction"].as_str().unwrap().to_owned();
            config.validate_supported().unwrap();
            let req = case_request(case, &config);
            candle_gen::gen_core::iris::validate_generation_request("iris_3b", &req, &config)
                .unwrap();
            let params = GenerationParams::resolve(&req, 0, &config).unwrap();
            let conditioning = encode(&te, &params).unwrap();
            assert_eq!(conditioning.cond.len(), params.prompts.len());
            let mut steps = Vec::new();
            let image = denoise(
                &dit,
                &config.flow,
                &conditioning,
                golden.require(&format!("{name}/noise")),
                &params,
                &CancelFlag::new(),
                |i| steps.push(i),
                &PreviewSink::default(),
            )
            .unwrap();
            assert_eq!(steps, (1..=params.steps).collect::<Vec<_>>(), "{name}");
            assert_close(
                &format!("controls {name} (f32 tower)"),
                &image,
                golden.require(&format!("{name}/image_f32_tower")),
                TOL,
            );
            let want = golden.require(&format!("{name}/image"));
            if name != "base" && want.dims() == base_image.dims() {
                let (moved, _, _) = errors(want, base_image);
                assert!(
                    moved > 2.0 * SPREAD,
                    "{name}: the golden moves only {moved:.3e} from base — not discriminative"
                );
            }
        }
    }
}

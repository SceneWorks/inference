//! Raw decoder parity, distinct from the public TextLlm Content stream. The latter suppresses
//! tokenizer-marked control tokens and EOS, so comparing its events directly to raw HF ids shifts
//! the sequence at `<|vision_end|>` / `</think>` and falsely reports a numerical mismatch.
//! This CPU-only miniature covers every raw text/bbox step, including suppressed ids, with the
//! frozen generated-only presence policy. Multimodal public-stream coverage lives in the media crate.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_llm::models::{Qwen35Config, Qwen35Model};
use candle_llm::primitives::sampler::{sample, SamplingParams, SplitMix64};
use candle_llm::primitives::Weights;

#[test]
fn frozen_pup_raw_text_decoder_matches_every_teacher_forced_step() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../media/mlx-gen/mlx-gen-flux3/tests/fixtures");
    let dir = fixtures.join("tiny-pup-snapshot");
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let cfg = Qwen35Config::from_json(&config).unwrap();
    let tensors =
        candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu).unwrap();
    let weights = Weights::from_map(tensors, Device::Cpu);
    let model =
        Qwen35Model::from_weights_dtype(&weights, "model.language_model", cfg, None, DType::F32)
            .unwrap();
    let golden =
        candle_core::safetensors::load(fixtures.join("flux3_pup_golden.safetensors"), &Device::Cpu)
            .unwrap();
    for case in ["t2i", "t2i_bbox"] {
        let ids = golden[&format!("ids_{case}")].to_vec1::<i32>().unwrap();
        let generated = golden[&format!("generated_{case}")]
            .to_vec1::<i32>()
            .unwrap();
        let expected_logits = golden[&format!("logits_{case}")]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let prompt: Vec<u32> = ids.iter().map(|&x| u32::try_from(x).unwrap()).collect();
        let mut cache = model.new_cache();
        let input = Tensor::new(prompt.as_slice(), &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mut logits = model.decode_logits(&input, &mut cache, 0).unwrap();
        assert_eq!(logits.dtype(), DType::F32);
        let first = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(first.len(), expected_logits.len());
        let max_error = first
            .iter()
            .zip(&expected_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_error < 1e-4,
            "{case}: raw f32 prefill max error {max_error}"
        );
        let params = SamplingParams {
            presence_penalty: 1.5,
            presence_penalty_generated_only: true,
            ..Default::default()
        }
        .for_prompt(ids.len());
        let mut history = ids;
        let mut rng = SplitMix64::new(0);
        assert_eq!(
            generated.len(),
            12,
            "both text oracle runs exhaust their raw decode budget"
        );
        for (step, &expected_id) in generated.iter().enumerate() {
            let actual_id = sample(&logits, &history, &params, &mut rng, None).unwrap();
            assert_eq!(
                actual_id, expected_id,
                "{case} raw step {step}; do not filter protocol ids"
            );
            history.push(expected_id);
            if step + 1 < generated.len() {
                let next = Tensor::new(&[u32::try_from(expected_id).unwrap()], &Device::Cpu)
                    .unwrap()
                    .reshape((1, 1))
                    .unwrap();
                logits = model
                    .decode_logits(&next, &mut cache, (prompt.len() + step) as i32)
                    .unwrap();
            }
        }
    }
}

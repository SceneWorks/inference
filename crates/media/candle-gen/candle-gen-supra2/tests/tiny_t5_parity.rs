//! The Candle text path ([`encode_tokens`] over candle's T5 v1.1 encoder) on the MLX twin's tiny
//! T5 fixture (`tiny_t5_parity.safetensors`: HF `T5EncoderModel`, 2 blocks × 4 heads, gated-GELU,
//! a 10-token context with 4 padded positions). The real-token rows are held to HF's masked
//! full-length encode; the padded rows are zeros (inert under the DiT's cross-attention key mask —
//! `tiny_dit_parity::masked_context_rows_cannot_move_the_velocity`).

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_supra2::encode_tokens;
use candle_transformers::models::t5::{Config, T5EncoderModel};
use gen_core_checkpoint::supra2::Supra2Tokens;

use crate::support::{devices, host, report, tol};

#[test]
fn tiny_t5_real_rows_match_hf_masked_encoder() {
    // The fixture's `T5Config` (tools/dump_supra2_reference.py `dump_tiny_t5`).
    let cfg: Config = serde_json::from_value(serde_json::json!({
        "vocab_size": 64, "d_model": 32, "d_kv": 8, "d_ff": 48, "num_layers": 2,
        "num_heads": 4, "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128, "dropout_rate": 0.0,
        "layer_norm_epsilon": 1e-6, "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu", "is_encoder_decoder": true,
        "pad_token_id": 0, "eos_token_id": 1
    }))
    .unwrap();
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/mlx-gen-supra2/tests/fixtures/tiny_t5_parity.safetensors");
    for (name, device) in devices() {
        let all = candle_gen::candle_core::safetensors::load(&path, &device).unwrap();
        let weights: HashMap<String, Tensor> = all
            .iter()
            .filter_map(|(k, v)| k.strip_prefix("w.").map(|n| (n.to_string(), v.clone())))
            .collect();
        let mut t5 =
            T5EncoderModel::load(VarBuilder::from_tensors(weights, DType::F32, &device), &cfg)
                .unwrap();
        let ids: Vec<i32> = host(&all["in.ids"]).iter().map(|&v| v as i32).collect();
        let mask: Vec<i32> = host(&all["in.mask"]).iter().map(|&v| v as i32).collect();
        let real = mask.iter().filter(|&&m| m == 1).count();
        assert!(real < ids.len(), "the fixture must carry padding");
        let tokens = Supra2Tokens {
            ids,
            mask,
            prompt_tokens: real,
            truncated: false,
        };
        let got = host(&encode_tokens(&mut t5, &tokens, 32, &device).unwrap());
        let want = host(&all["out.hidden"]);
        assert_eq!(got.len(), want.len());
        // Declared (max_rel, mean_rel): the MLX twin's tiny-T5 bound, 1e-5 / 1e-6, on the CPU;
        // 10x on an accelerator (reduction order only).
        report(
            &format!("tiny_t5.real_rows.{name}"),
            &got[..real * 32],
            &want[..real * 32],
            tol(&device, (1e-5, 1e-6), (1e-4, 1e-5)),
        );
        assert!(got[real * 32..].iter().all(|&v| v == 0.0), "{name}");
    }
}

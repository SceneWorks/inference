//! Qwen3-1.7B as a **text encoder** (sc-12841) — the reference `Qwen3TextEncoder` wraps the
//! causal LM and takes `hidden_states[-1]`, i.e. the full-sequence hidden states **after the
//! final RMSNorm**, as the DiT's text conditioning.
//!
//! The stack is the audio commons' stateless [`candle_audio::qwen3::TextEncoder`] (sc-27270; see
//! it for why this is not `candle_transformers::models::qwen3`). The checkpoint is the causal LM,
//! so the stack sits under `model.` (`model.embed_tokens` / `model.layers.N.*` / `model.norm`).

use candle_audio::candle_core::Result as CandleResult;
use candle_audio::qwen3::BlockConfig;
use candle_nn::VarBuilder;

use crate::config::TextEncoderConfig;

/// The stateless Qwen3 text encoder — the audio commons' shared stack (sc-27270).
pub use candle_audio::qwen3::TextEncoder as Qwen3Encoder;

/// Build the encoder from the checkpoint's text-encoder config. The weights sit under `model.`.
pub fn load(cfg: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Qwen3Encoder> {
    let block = BlockConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        attention_bias: cfg.attention_bias,
    };
    Qwen3Encoder::new(
        &block,
        cfg.vocab_size,
        cfg.num_hidden_layers,
        vb.pp("model"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarMap;

    fn tiny_cfg() -> TextEncoderConfig {
        serde_json::from_str(
            r#"{"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 4, "attention_bias": false, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "use_sliding_window": false}"#,
        )
        .unwrap()
    }

    fn tiny_encoder(seed_scale: f64) -> Qwen3Encoder {
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let enc = load(&cfg, vb).unwrap();
        // Give the zero-initialized VarMap deterministic non-trivial values.
        for (i, (_, var)) in varmap.data().lock().unwrap().iter().enumerate() {
            let t = var.as_tensor();
            let n: usize = t.shape().elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|j| (((i * 31 + j * 17) % 13) as f64 * 0.03 * seed_scale - 0.18) as f32)
                .collect();
            let new = Tensor::from_vec(vals, t.shape(), &Device::Cpu).unwrap();
            var.set(&new).unwrap();
        }
        enc
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn encode_is_causal_and_stateless() {
        let enc = tiny_encoder(1.0);
        let full = enc.encode(&[1, 2, 3, 4]).unwrap();
        let prefix = enc.encode(&[1, 2]).unwrap();
        // Causality: the first two rows of the longer encode equal the prefix encode —
        // the property that legitimizes unpadded encoding (see module docs).
        let a = full
            .i((0, 0..2))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = prefix.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "causal prefix mismatch: {x} vs {y}");
        }
        // Statelessness: re-encoding yields identical values (no hidden KV accumulation).
        let again = enc.encode(&[1, 2, 3, 4]).unwrap();
        let a2 = again.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let a1 = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a1, a2);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen3::tests::encode_is_causal_and_stateless");
    }
}

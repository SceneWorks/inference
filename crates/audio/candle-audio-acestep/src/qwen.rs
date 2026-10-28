//! Qwen3-Embedding-0.6B as ACE-Step's **text encoder** (sc-12842).
//!
//! Two entry points, both stateless full-sequence forwards (no KV-cache accumulation across
//! prompts), on the audio commons' shared [`candle_audio::qwen3::TextEncoder`] (sc-27270):
//!
//! - [`Qwen3Encoder::encode`] — the full causal stack + final RMSNorm, returning the
//!   post-norm hidden states the reference feeds as the DiT **prompt** conditioning.
//! - [`Qwen3Encoder::embed`] — the embedding-layer token lookup only, returning the raw token
//!   embeddings the condition encoder's lyric encoder contextualizes (the reference lyric path).
//!
//! The Qwen3-Embedding-0.6B safetensors store the bare `Qwen3Model` (no `model.` prefix):
//! `embed_tokens.weight` / `layers.N.*` / `norm.weight`.

use candle_audio::candle_core::Result as CandleResult;
use candle_audio::qwen3::BlockConfig;
use candle_nn::VarBuilder;

use crate::config::TextEncoderConfig;

/// The stateless Qwen3 text encoder — the audio commons' shared stack (sc-27270).
pub use candle_audio::qwen3::TextEncoder as Qwen3Encoder;

/// Build the encoder from the checkpoint's text-encoder config. The weights sit at the root.
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
    Qwen3Encoder::new(&block, cfg.vocab_size, cfg.num_hidden_layers, vb)
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
                "rope_theta": 1000000.0, "use_sliding_window": false}"#,
        )
        .unwrap()
    }

    fn tiny_encoder() -> Qwen3Encoder {
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let enc = load(&cfg, vb).unwrap();
        for (i, (_, var)) in varmap.data().lock().unwrap().iter().enumerate() {
            let t = var.as_tensor();
            let n: usize = t.shape().elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|j| (((i * 31 + j * 17) % 13) as f64 * 0.03 - 0.18) as f32)
                .collect();
            let new = Tensor::from_vec(vals, t.shape(), &Device::Cpu).unwrap();
            var.set(&new).unwrap();
        }
        enc
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn encode_is_causal_and_stateless() {
        let enc = tiny_encoder();
        let full = enc.encode(&[1, 2, 3, 4]).unwrap();
        let prefix = enc.encode(&[1, 2]).unwrap();
        let a = full
            .i((0, 0..2))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = prefix.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "causal prefix mismatch {x} vs {y}");
        }
        let again = enc.encode(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            full.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            again.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen::tests::encode_is_causal_and_stateless");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn embed_is_a_pure_lookup() {
        let enc = tiny_encoder();
        let e = enc.embed(&[1, 2, 3]).unwrap();
        assert_eq!(e.dims(), &[1, 3, 16]);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen::tests::embed_is_a_pure_lookup");
    }
}

//! Pipeline (layer-split) multi-GPU sharding tests (story 7263, part b).
//!
//! ## What these prove
//! - **Drop-in parity**: a model loaded *sharded* (`CausalLm::from_dir_sharded`) onto a single
//!   device is **token-for-token identical** to an ordinary load — the sharded `Weights` loader,
//!   per-layer device inference, and the cross-device hand-off in the decoder loop change nothing
//!   when every layer happens to share one device. This runs on CPU, no GPU, no download.
//! - **Real multi-GPU** (`#[ignore]`, needs 2 CUDA devices): the same model split across `cuda:0`
//!   and `cuda:1` runs end-to-end and its greedy output **matches the single-GPU run bit-for-bit**
//!   (the boundary hand-off is a lossless device copy; the per-layer kernels are identical), with the
//!   layers verified to actually span both GPUs.
//!
//! The point of sharding is **capacity**, not speed: it lets a model too large for one card (e.g. on
//! 2×24GB consumer GPUs) run by putting contiguous layer blocks on each GPU. The sharded loader
//! streams each file through host memory, so no single GPU ever holds more than its own shard.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use core_llm::Tokenizer;

use candle_llm::config::ModelConfig;
use candle_llm::decode::{generate, CancelFlag, GenerationConfig};
use candle_llm::models::CausalLm;
use candle_llm::primitives::sampler::SamplingParams;
use candle_llm::primitives::{SplitMix64, TokenRng, Weights};

/// Greedy, fixed seed, no stop tokens — bit-exactness across load layouts is the point.
fn gen_config(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

fn greedy(model: &CausalLm, prompt: &[i32], max_new: usize) -> Vec<i32> {
    generate(
        model,
        prompt,
        &gen_config(max_new),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens
}

// ---- Synthetic 4-layer CPU model (no download) ---------------------------------------------------

const VOCAB: usize = 48;
const HIDDEN: usize = 32;
const INTER: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;
const LAYERS: usize = 4;

fn randn(shape: (usize, usize), rng: &mut SplitMix64) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::from_vec(vec![1.0f32; d], (d,), &Device::Cpu).unwrap()
}

/// Write a tiny 4-layer `llama` snapshot from deterministic random weights and return its directory
/// (kept on disk so it can be loaded several ways; the caller removes it).
fn build_tiny_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("candle-llm-shard-{}-{uniq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let cfg = format!(
        r#"{{
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": {HIDDEN}, "intermediate_size": {INTER}, "num_hidden_layers": {LAYERS},
            "num_attention_heads": {NUM_HEADS}, "num_key_value_heads": {NUM_KV_HEADS},
            "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 0
        }}"#
    );
    std::fs::write(dir.join("config.json"), cfg).unwrap();

    let mut rng = SplitMix64::new(0x5A6E_D101);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        randn((VOCAB, HIDDEN), &mut rng),
    );
    w.insert("model.norm.weight".into(), ones(HIDDEN));
    w.insert("lm_head.weight".into(), randn((VOCAB, HIDDEN), &mut rng));

    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        w.insert(p("input_layernorm.weight"), ones(HIDDEN));
        w.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        w.insert(
            p("self_attn.q_proj.weight"),
            randn((q_dim, HIDDEN), &mut rng),
        );
        w.insert(
            p("self_attn.k_proj.weight"),
            randn((kv_dim, HIDDEN), &mut rng),
        );
        w.insert(
            p("self_attn.v_proj.weight"),
            randn((kv_dim, HIDDEN), &mut rng),
        );
        w.insert(
            p("self_attn.o_proj.weight"),
            randn((HIDDEN, q_dim), &mut rng),
        );
        w.insert(p("mlp.gate_proj.weight"), randn((INTER, HIDDEN), &mut rng));
        w.insert(p("mlp.up_proj.weight"), randn((INTER, HIDDEN), &mut rng));
        w.insert(p("mlp.down_proj.weight"), randn((HIDDEN, INTER), &mut rng));
    }

    candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
    dir
}

/// Sharded-onto-one-device is a perfect drop-in for an ordinary load: same loader output, same
/// inferred per-layer device, same forward (the cross-device hand-off is a no-op on one device).
#[test]
fn sharded_single_device_matches_plain_cpu() {
    let dir = build_tiny_dir();

    let plain = CausalLm::from_weights(
        &Weights::from_dir(&dir, &Device::Cpu).unwrap(),
        "",
        ModelConfig::from_dir(&dir).unwrap(),
    )
    .unwrap();

    let sharded = CausalLm::from_dir_sharded(
        &dir,
        ModelConfig::from_dir(&dir).unwrap(),
        DType::F32,
        &[Device::Cpu],
    )
    .unwrap();

    assert_eq!(
        sharded.layer_devices().len(),
        LAYERS,
        "one device entry per layer"
    );
    assert!(sharded.layer_devices().iter().all(|d| d.is_cpu()));

    let prompt: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7];
    let plain_out = greedy(&plain, &prompt, 16);
    let sharded_out = greedy(&sharded, &prompt, 16);
    assert!(!plain_out.is_empty());
    assert_eq!(
        sharded_out, plain_out,
        "a sharded load onto one device must be token-for-token identical to a plain load"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ---- Real multi-GPU variant (#[ignore]) ----------------------------------------------------------

mod real {
    use super::*;

    fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
        tok.encode(text, true)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect()
    }

    /// Split a real model across `cuda:0` + `cuda:1` and confirm it runs end-to-end with output
    /// identical to the single-GPU load. Skips cleanly if a second GPU or the model isn't available.
    #[test]
    #[ignore = "needs 2 CUDA GPUs + CANDLE_LLM_TEST_MODEL"]
    fn shards_across_two_gpus_and_matches_single() {
        let Some(dir) = std::env::var("CANDLE_LLM_TEST_MODEL")
            .ok()
            .filter(|v| !v.is_empty())
        else {
            eprintln!("skip: set CANDLE_LLM_TEST_MODEL");
            return;
        };
        let (Ok(d0), Ok(d1)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
            eprintln!("skip: needs two CUDA devices");
            return;
        };

        let sharded = CausalLm::from_dir_sharded(
            &dir,
            ModelConfig::from_dir(&dir).unwrap(),
            DType::BF16,
            &[d0.clone(), d1.clone()],
        )
        .unwrap();

        // The layers must actually span both GPUs (a contiguous split).
        let devs = sharded.layer_devices();
        assert!(
            devs.iter().any(|d| d.same_device(&d0)) && devs.iter().any(|d| d.same_device(&d1)),
            "sharded layers must live on both cuda:0 and cuda:1"
        );

        // Single-GPU baseline (sharded onto one device == ordinary load), same dtype path.
        let single = CausalLm::from_dir_sharded(
            &dir,
            ModelConfig::from_dir(&dir).unwrap(),
            DType::BF16,
            std::slice::from_ref(&d0),
        )
        .unwrap();

        let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
        let prompt = encode(&tok, "The capital of France is");

        let out_single = greedy(&single, &prompt, 24);
        let out_sharded = greedy(&sharded, &prompt, 24);
        assert!(!out_sharded.is_empty(), "sharded model produced no tokens");
        println!(
            "[shard] {}",
            tok.decode(
                &out_sharded.iter().map(|&t| t as u32).collect::<Vec<_>>(),
                true
            )
            .unwrap()
            .replace('\n', " ")
        );
        assert_eq!(
            out_sharded, out_single,
            "2-GPU sharded output must match the single-GPU run token-for-token"
        );
    }
}

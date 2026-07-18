//! Multi-head Latent Attention smoke test on a tiny synthetic DeepSeek-V2 model (story 7288).
//!
//! Builds a 2-layer `deepseek_v2` snapshot from deterministic random weights — exercising the MLA
//! attention path (low-rank q/kv projections, the decoupled YaRN-RoPE key sub-vector), the fine-
//! grained MoE FFN with a leading dense layer (`first_k_dense_replace`), and the growing KV cache —
//! and checks it prefills and decodes finite, correctly-shaped logits across cached steps. Runs on
//! CPU with no model download, so it guards the MLA wiring in CI alongside the real-weights breadth
//! test (`CANDLE_LLM_DEEPSEEK_MODEL`, see `tests/breadth.rs`).

use std::collections::HashMap;

use candle_core::{Device, Tensor};

use candle_llm::primitives::{input_ids, SplitMix64, TokenRng, Weights};
use candle_llm::{CausalLm, Error, ModelConfig};

const VOCAB: usize = 32;
const HIDDEN: usize = 32;
const NUM_HEADS: usize = 2;
const QK_NOPE: usize = 16;
const QK_ROPE: usize = 8;
const V_HEAD: usize = 16;
const KV_LORA: usize = 24;
const MOE_INTER: usize = 16;
const N_ROUTED: usize = 4;
const N_SHARED: usize = 1;
const DENSE_INTER: usize = 32;

fn randn(shape: (usize, usize), rng: &mut SplitMix64) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::from_vec(vec![1.0f32; d], (d,), &Device::Cpu).unwrap()
}

/// Write a tiny `deepseek_v2` snapshot to a temp dir and load it. `q_lora_rank` selects the query
/// path: `None` ⇒ a full `q_proj` (DeepSeek-V2-Lite); `Some(r)` ⇒ the low-rank `q_a → norm → q_b`.
fn load_tiny_deepseek(tag: &str, q_lora_rank: Option<usize>) -> CausalLm {
    let q_head = QK_NOPE + QK_ROPE;
    let dir = std::env::temp_dir().join(format!("candle-llm-mla-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let q_lora_json = match q_lora_rank {
        Some(r) => r.to_string(),
        None => "null".to_string(),
    };
    let config = format!(
        r#"{{
            "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
            "hidden_size": {HIDDEN}, "intermediate_size": {DENSE_INTER}, "num_hidden_layers": 2,
            "num_attention_heads": {NUM_HEADS}, "num_key_value_heads": {NUM_HEADS},
            "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 999,
            "q_lora_rank": {q_lora_json}, "kv_lora_rank": {KV_LORA},
            "qk_nope_head_dim": {QK_NOPE}, "qk_rope_head_dim": {QK_ROPE}, "v_head_dim": {V_HEAD},
            "n_routed_experts": {N_ROUTED}, "num_experts_per_tok": 2, "n_shared_experts": {N_SHARED},
            "moe_intermediate_size": {MOE_INTER}, "first_k_dense_replace": 1, "norm_topk_prob": false,
            "routed_scaling_factor": 1.0,
            "rope_scaling": {{ "type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
                "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 64 }}
        }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();

    let mut rng = SplitMix64::new(0xD2C0FFEE);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        randn((VOCAB, HIDDEN), &mut rng),
    );
    w.insert("model.norm.weight".into(), ones(HIDDEN));
    w.insert("lm_head.weight".into(), randn((VOCAB, HIDDEN), &mut rng));

    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        w.insert(p("input_layernorm.weight"), ones(HIDDEN));
        w.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));

        // MLA projections.
        match q_lora_rank {
            None => {
                w.insert(
                    p("self_attn.q_proj.weight"),
                    randn((NUM_HEADS * q_head, HIDDEN), &mut rng),
                );
            }
            Some(r) => {
                w.insert(p("self_attn.q_a_proj.weight"), randn((r, HIDDEN), &mut rng));
                w.insert(p("self_attn.q_a_layernorm.weight"), ones(r));
                w.insert(
                    p("self_attn.q_b_proj.weight"),
                    randn((NUM_HEADS * q_head, r), &mut rng),
                );
            }
        }
        w.insert(
            p("self_attn.kv_a_proj_with_mqa.weight"),
            randn((KV_LORA + QK_ROPE, HIDDEN), &mut rng),
        );
        w.insert(p("self_attn.kv_a_layernorm.weight"), ones(KV_LORA));
        w.insert(
            p("self_attn.kv_b_proj.weight"),
            randn((NUM_HEADS * (QK_NOPE + V_HEAD), KV_LORA), &mut rng),
        );
        w.insert(
            p("self_attn.o_proj.weight"),
            randn((HIDDEN, NUM_HEADS * V_HEAD), &mut rng),
        );

        if i == 0 {
            // Leading dense layer (first_k_dense_replace = 1).
            w.insert(
                p("mlp.gate_proj.weight"),
                randn((DENSE_INTER, HIDDEN), &mut rng),
            );
            w.insert(
                p("mlp.up_proj.weight"),
                randn((DENSE_INTER, HIDDEN), &mut rng),
            );
            w.insert(
                p("mlp.down_proj.weight"),
                randn((HIDDEN, DENSE_INTER), &mut rng),
            );
        } else {
            // MoE layer: router + routed experts + an ungated shared expert (plural key).
            w.insert(p("mlp.gate.weight"), randn((N_ROUTED, HIDDEN), &mut rng));
            for e in 0..N_ROUTED {
                let ep = |s: &str| p(&format!("mlp.experts.{e}.{s}"));
                w.insert(ep("gate_proj.weight"), randn((MOE_INTER, HIDDEN), &mut rng));
                w.insert(ep("up_proj.weight"), randn((MOE_INTER, HIDDEN), &mut rng));
                w.insert(ep("down_proj.weight"), randn((HIDDEN, MOE_INTER), &mut rng));
            }
            let shared_inter = N_SHARED * MOE_INTER;
            w.insert(
                p("mlp.shared_experts.gate_proj.weight"),
                randn((shared_inter, HIDDEN), &mut rng),
            );
            w.insert(
                p("mlp.shared_experts.up_proj.weight"),
                randn((shared_inter, HIDDEN), &mut rng),
            );
            w.insert(
                p("mlp.shared_experts.down_proj.weight"),
                randn((HIDDEN, shared_inter), &mut rng),
            );
        }
    }

    candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let weights = Weights::from_dir(&dir, &Device::Cpu).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    model
}

fn assert_finite(logits: &Tensor) {
    let host = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(host.len(), VOCAB);
    assert!(
        host.iter().all(|v| v.is_finite()),
        "non-finite logits: {host:?}"
    );
}

fn argmax(logits: &Tensor) -> i32 {
    let host = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    host.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

#[test]
fn deepseek_config_without_kv_lora_rank_returns_config_error() {
    let config = serde_json::json!({
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": HIDDEN,
        "intermediate_size": DENSE_INTER,
        "num_hidden_layers": 1,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false
    });
    let cfg = ModelConfig::from_json(&config).unwrap();
    assert_eq!(cfg.architecture.family(), "deepseek_v2");
    assert!(cfg.mla.is_none());

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".into(),
        Tensor::zeros((VOCAB, HIDDEN), candle_core::DType::F32, &Device::Cpu).unwrap(),
    );
    tensors.insert("model.norm.weight".into(), ones(HIDDEN));
    tensors.insert(
        "lm_head.weight".into(),
        Tensor::zeros((VOCAB, HIDDEN), candle_core::DType::F32, &Device::Cpu).unwrap(),
    );

    let err = CausalLm::from_weights(&Weights::from_map(tensors, Device::Cpu), "", cfg)
        .err()
        .expect("missing kv_lora_rank must reject the config");
    assert!(
        matches!(&err, Error::Config(message) if message.contains("kv_lora_rank")),
        "expected actionable config error, got {err:?}"
    );
}

/// Drive a prefill + several cached decode steps through the MLA path and check every step yields
/// finite `[1, vocab]` logits and that the KV cache grows one position per step.
fn run_decode(model: &CausalLm) {
    assert!(model.config().is_mla(), "model should report MLA");
    assert_eq!(model.config().architecture.family(), "deepseek_v2");

    let prompt = [1i32, 2, 3, 4, 5];
    let ids = input_ids(&prompt, model.device()).unwrap();
    let mut cache = model.new_cache();

    let logits = model.decode_logits(&ids, &mut cache, 0).unwrap();
    assert_eq!(logits.dims(), &[1, VOCAB]);
    assert_finite(&logits);

    let mut next = argmax(&logits);
    for step_i in 0..4 {
        let offset = prompt.len() as i32 + step_i;
        let step = input_ids(&[next], model.device()).unwrap();
        let logits = model.decode_logits(&step, &mut cache, offset).unwrap();
        assert_eq!(logits.dims(), &[1, VOCAB]);
        assert_finite(&logits);
        next = argmax(&logits);
    }

    use candle_llm::primitives::KvCache;
    assert_eq!(cache.offset(), prompt.len() as i32 + 4);
}

#[test]
fn mla_full_q_proj_prefills_and_decodes() {
    // DeepSeek-V2-Lite shape: a full q_proj (no query LoRA).
    let model = load_tiny_deepseek("fullq", None);
    run_decode(&model);
}

#[test]
fn mla_query_lora_prefills_and_decodes() {
    // The larger DeepSeek-V2 shape: a low-rank q_a → q_a_layernorm → q_b query path.
    let model = load_tiny_deepseek("qlora", Some(20));
    run_decode(&model);
}

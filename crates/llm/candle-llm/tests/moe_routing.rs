//! sc-24440: Mixture-of-Experts routing runs on the device, through one shared router.
//!
//! Every MoE family the crate serves — Qwen2-MoE and DeepSeek-V2 through the generic `CausalLm`,
//! Qwen3.6-MoE (`qwen3_5_moe`) through `Qwen35Model` — reaches the same routing module. This suite
//! pins what that module must preserve and what it must remove:
//!
//! * **Greedy parity with the pre-change router.** `moe_greedy_candle.json` holds the greedy
//!   continuations each fixture produced under the router this story replaced (softmax and top-k
//!   read back to the host every layer, per-expert token lists, `index_add` scatter). They were
//!   captured by running this file on the base branch (`f4ce41e50`) *before* the router changed,
//!   so a match is parity with the old router — not the new router agreeing with itself.
//! * **No host read in the router.** A decode step on an MoE fixture, dispatched as on a GPU
//!   ([`with_device_dispatch`]), runs every MoE layer through the stacked device dispatch
//!   ([`moe_device_dispatch_count`]) and issues zero counted device→host transfers
//!   ([`host_sync_count`]). The counter only sees reads that note themselves: the old Qwen3.6-MoE
//!   router counted its read, the old `CausalLm` one (Qwen2-MoE, DeepSeek-V2) did not, so the
//!   device-dispatch count is what separates the old router from the new for those two. A read
//!   hidden inside the new router itself is the source lint's to catch
//!   (`primitives::moe::tests::host_reads_go_through_read_routes`): Candle exposes no hook on a
//!   CPU tensor read.
//! * **No MoE graph refusal.** [`StepModel::graph_support`] no longer names
//!   `moe_router_host_read` for an MoE model.
//!
//! * **On CUDA (the `windows-cuda` lane), the indexed dispatch.** Every fixture — dense, Q8_0,
//!   Q4_0, Q4_K, and NVFP4 on an sm_120 device — decodes greedily through the CUDA-graph runner
//!   with its experts dispatched by the indexed kernels, token-identical to the same request
//!   dispatched grouped (the per-expert path); the graph is captured and replayed (a host read in
//!   the step would refuse it), and a decode step reads nothing back and gathers no expert matrix.
//!
//! Regenerate the goldens (only ever against a known-good tree, and say so in the commit):
//!
//! ```text
//! SC24440_WRITE_MOE_GREEDY=1 cargo test -p candle-llm --test moe_routing
//! ```

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use serde_json::{json, Map, Value};

use candle_llm::config::ModelConfig;
use candle_llm::decode::StepModel;
use candle_llm::models::{CausalLm, Qwen35Config, Qwen35Model};
use candle_llm::primitives::moe::{moe_device_dispatch_count, with_device_dispatch};
use candle_llm::primitives::{
    host_sync_count, input_ids, QuantSpec, SplitMix64, TokenRng, Weights,
};

/// The prompt every fixture prefills (a multi-token forward: the prefill routing shape).
const PROMPT: [i32; 6] = [3, 1, 4, 1, 5, 9];
/// Greedy single-token decode steps after the prefill (the decode routing shape).
const GREEDY_STEPS: usize = 24;

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../testdata/architectures/moe_greedy_candle.json")
}

/// Small deterministic weight in `[-0.2, 0.2)`.
fn randn(shape: &[usize], rng: &mut SplitMix64) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::from_vec(vec![1.0f32; d], (d,), &Device::Cpu).unwrap()
}

/// The dims a llama-family MoE fixture is built at.
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    vocab: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    moe_inter: usize,
    shared_inter: usize,
}

/// A Qwen2-MoE decoder: Qwen2 attention (q/k/v bias) + a sigmoid-gated shared expert.
fn qwen2_moe(d: Dims, norm_topk_prob: bool, seed: u64) -> (Value, HashMap<String, Tensor>) {
    let mut rng = SplitMix64::new(seed);
    let mut m = HashMap::new();
    let kv = d.heads / 2;
    let (qd, kvd) = (d.heads * d.head_dim, kv * d.head_dim);
    m.insert(
        "model.embed_tokens.weight".into(),
        randn(&[d.vocab, d.hidden], &mut rng),
    );
    m.insert("model.norm.weight".into(), ones(d.hidden));
    m.insert(
        "lm_head.weight".into(),
        randn(&[d.vocab, d.hidden], &mut rng),
    );
    for i in 0..d.layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(d.hidden));
        m.insert(p("post_attention_layernorm.weight"), ones(d.hidden));
        m.insert(
            p("self_attn.q_proj.weight"),
            randn(&[qd, d.hidden], &mut rng),
        );
        m.insert(
            p("self_attn.k_proj.weight"),
            randn(&[kvd, d.hidden], &mut rng),
        );
        m.insert(
            p("self_attn.v_proj.weight"),
            randn(&[kvd, d.hidden], &mut rng),
        );
        m.insert(
            p("self_attn.o_proj.weight"),
            randn(&[d.hidden, qd], &mut rng),
        );
        m.insert(p("self_attn.q_proj.bias"), randn(&[qd], &mut rng));
        m.insert(p("self_attn.k_proj.bias"), randn(&[kvd], &mut rng));
        m.insert(p("self_attn.v_proj.bias"), randn(&[kvd], &mut rng));
        m.insert(
            p("mlp.gate.weight"),
            randn(&[d.experts, d.hidden], &mut rng),
        );
        for e in 0..d.experts {
            let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
            m.insert(
                ep("gate_proj.weight"),
                randn(&[d.moe_inter, d.hidden], &mut rng),
            );
            m.insert(
                ep("up_proj.weight"),
                randn(&[d.moe_inter, d.hidden], &mut rng),
            );
            m.insert(
                ep("down_proj.weight"),
                randn(&[d.hidden, d.moe_inter], &mut rng),
            );
        }
        for (name, shape) in [
            ("gate_proj", [d.shared_inter, d.hidden]),
            ("up_proj", [d.shared_inter, d.hidden]),
            ("down_proj", [d.hidden, d.shared_inter]),
        ] {
            m.insert(
                p(&format!("mlp.shared_expert.{name}.weight")),
                randn(&shape, &mut rng),
            );
        }
        m.insert(
            p("mlp.shared_expert_gate.weight"),
            randn(&[1, d.hidden], &mut rng),
        );
    }
    let config = json!({
        "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
        "hidden_size": d.hidden, "intermediate_size": 2 * d.hidden,
        "num_hidden_layers": d.layers, "num_attention_heads": d.heads,
        "num_key_value_heads": kv, "vocab_size": d.vocab, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0, "tie_word_embeddings": false,
        "num_experts": d.experts, "num_experts_per_tok": d.top_k,
        "norm_topk_prob": norm_topk_prob, "moe_intermediate_size": d.moe_inter,
        "shared_expert_intermediate_size": d.shared_inter
    });
    (config, m)
}

/// A DeepSeek-V2 decoder: MLA attention, a leading dense layer, then MoE layers with an ungated
/// shared expert and a non-unit `routed_scaling_factor` (the un-normalized routing branch).
fn deepseek_v2(seed: u64) -> (Value, HashMap<String, Tensor>) {
    let (hidden, vocab, layers) = (32usize, 64usize, 3usize);
    let (heads, qk_nope, qk_rope, v_head, kv_lora) = (2usize, 16usize, 8usize, 16usize, 24usize);
    let q_head = qk_nope + qk_rope;
    let (n_routed, moe_inter, n_shared, dense_inter) = (8usize, 16usize, 2usize, 32usize);
    let mut rng = SplitMix64::new(seed);
    let mut m = HashMap::new();
    m.insert(
        "model.embed_tokens.weight".into(),
        randn(&[vocab, hidden], &mut rng),
    );
    m.insert("model.norm.weight".into(), ones(hidden));
    m.insert("lm_head.weight".into(), randn(&[vocab, hidden], &mut rng));
    for i in 0..layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(hidden));
        m.insert(p("post_attention_layernorm.weight"), ones(hidden));
        m.insert(
            p("self_attn.q_proj.weight"),
            randn(&[heads * q_head, hidden], &mut rng),
        );
        m.insert(
            p("self_attn.kv_a_proj_with_mqa.weight"),
            randn(&[kv_lora + qk_rope, hidden], &mut rng),
        );
        m.insert(p("self_attn.kv_a_layernorm.weight"), ones(kv_lora));
        m.insert(
            p("self_attn.kv_b_proj.weight"),
            randn(&[heads * (qk_nope + v_head), kv_lora], &mut rng),
        );
        m.insert(
            p("self_attn.o_proj.weight"),
            randn(&[hidden, heads * v_head], &mut rng),
        );
        if i == 0 {
            m.insert(
                p("mlp.gate_proj.weight"),
                randn(&[dense_inter, hidden], &mut rng),
            );
            m.insert(
                p("mlp.up_proj.weight"),
                randn(&[dense_inter, hidden], &mut rng),
            );
            m.insert(
                p("mlp.down_proj.weight"),
                randn(&[hidden, dense_inter], &mut rng),
            );
        } else {
            m.insert(p("mlp.gate.weight"), randn(&[n_routed, hidden], &mut rng));
            for e in 0..n_routed {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                m.insert(
                    ep("gate_proj.weight"),
                    randn(&[moe_inter, hidden], &mut rng),
                );
                m.insert(ep("up_proj.weight"), randn(&[moe_inter, hidden], &mut rng));
                m.insert(
                    ep("down_proj.weight"),
                    randn(&[hidden, moe_inter], &mut rng),
                );
            }
            let shared_inter = n_shared * moe_inter;
            m.insert(
                p("mlp.shared_experts.gate_proj.weight"),
                randn(&[shared_inter, hidden], &mut rng),
            );
            m.insert(
                p("mlp.shared_experts.up_proj.weight"),
                randn(&[shared_inter, hidden], &mut rng),
            );
            m.insert(
                p("mlp.shared_experts.down_proj.weight"),
                randn(&[hidden, shared_inter], &mut rng),
            );
        }
    }
    let config = json!({
        "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
        "hidden_size": hidden, "intermediate_size": dense_inter, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": heads, "vocab_size": vocab,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "q_lora_rank": null, "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
        "n_routed_experts": n_routed, "num_experts_per_tok": 3, "n_shared_experts": n_shared,
        "moe_intermediate_size": moe_inter, "first_k_dense_replace": 1,
        "norm_topk_prob": false, "routed_scaling_factor": 2.5
    });
    (config, m)
}

/// A Qwen3.6-MoE (`qwen3_5_moe`) hybrid decoder: three Gated-DeltaNet layers and one full-attention
/// layer, every FFN a sparse MoE block (fused `experts.gate_up_proj` checkpoint layout).
fn qwen35_moe(seed: u64) -> (Value, HashMap<String, Tensor>) {
    let (h, layers, vocab, experts, mi, si) = (32usize, 4usize, 64usize, 8usize, 16usize, 16usize);
    let (heads, kv_heads, head_dim) = (4usize, 2usize, 8usize);
    let (lv_heads, lk_heads, lk_dim, lv_dim, conv_k) = (4usize, 2usize, 4usize, 4usize, 4usize);
    let key_dim = lk_dim * lk_heads;
    let value_dim = lv_dim * lv_heads;
    let conv_dim = key_dim * 2 + value_dim;
    let pfx = "model.language_model";
    let mut rng = SplitMix64::new(seed);
    let mut m = HashMap::new();
    let mut t = |m: &mut HashMap<String, Tensor>, key: String, shape: &[usize]| {
        m.insert(key, randn(shape, &mut rng));
    };
    t(&mut m, format!("{pfx}.embed_tokens.weight"), &[vocab, h]);
    t(&mut m, format!("{pfx}.norm.weight"), &[h]);
    t(&mut m, "lm_head.weight".into(), &[vocab, h]);
    for i in 0..layers {
        let lp = |s: &str| format!("{pfx}.layers.{i}.{s}");
        t(&mut m, lp("input_layernorm.weight"), &[h]);
        t(&mut m, lp("post_attention_layernorm.weight"), &[h]);
        t(
            &mut m,
            lp("mlp.experts.gate_up_proj"),
            &[experts, 2 * mi, h],
        );
        t(&mut m, lp("mlp.experts.down_proj"), &[experts, h, mi]);
        t(&mut m, lp("mlp.gate.weight"), &[experts, h]);
        t(&mut m, lp("mlp.shared_expert.gate_proj.weight"), &[si, h]);
        t(&mut m, lp("mlp.shared_expert.up_proj.weight"), &[si, h]);
        t(&mut m, lp("mlp.shared_expert.down_proj.weight"), &[h, si]);
        t(&mut m, lp("mlp.shared_expert_gate.weight"), &[1, h]);
        if i < 3 {
            t(&mut m, lp("linear_attn.in_proj_qkv.weight"), &[conv_dim, h]);
            t(&mut m, lp("linear_attn.in_proj_z.weight"), &[value_dim, h]);
            t(&mut m, lp("linear_attn.in_proj_a.weight"), &[lv_heads, h]);
            t(&mut m, lp("linear_attn.in_proj_b.weight"), &[lv_heads, h]);
            t(
                &mut m,
                lp("linear_attn.conv1d.weight"),
                &[conv_dim, 1, conv_k],
            );
            t(&mut m, lp("linear_attn.A_log"), &[lv_heads]);
            t(&mut m, lp("linear_attn.dt_bias"), &[lv_heads]);
            t(&mut m, lp("linear_attn.norm.weight"), &[lv_dim]);
            t(&mut m, lp("linear_attn.out_proj.weight"), &[h, value_dim]);
        } else {
            t(
                &mut m,
                lp("self_attn.q_proj.weight"),
                &[heads * head_dim * 2, h],
            );
            t(
                &mut m,
                lp("self_attn.k_proj.weight"),
                &[kv_heads * head_dim, h],
            );
            t(
                &mut m,
                lp("self_attn.v_proj.weight"),
                &[kv_heads * head_dim, h],
            );
            t(
                &mut m,
                lp("self_attn.o_proj.weight"),
                &[h, heads * head_dim],
            );
            t(&mut m, lp("self_attn.q_norm.weight"), &[head_dim]);
            t(&mut m, lp("self_attn.k_norm.weight"), &[head_dim]);
        }
    }
    let config = json!({
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": h, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5, "max_position_embeddings": 128,
            "tie_word_embeddings": false, "full_attention_interval": 4,
            "linear_num_value_heads": lv_heads, "linear_num_key_heads": lk_heads,
            "linear_key_head_dim": lk_dim, "linear_value_head_dim": lv_dim,
            "linear_conv_kernel_dim": conv_k,
            "num_experts": experts, "num_experts_per_tok": 2,
            "moe_intermediate_size": mi, "shared_expert_intermediate_size": si
        },
        "vision_config": { "model_type": "qwen3_5" }
    });
    (config, m)
}

const TINY: Dims = Dims {
    hidden: 32,
    heads: 4,
    head_dim: 8,
    vocab: 64,
    layers: 2,
    experts: 8,
    top_k: 2,
    moe_inter: 16,
    shared_inter: 32,
};

/// Wide enough that every projection's input dim is a whole number of GGML blocks (and MLX
/// quantization groups), so the quantized expert banks are exercised too.
const WIDE: Dims = Dims {
    hidden: 64,
    heads: 4,
    head_dim: 16,
    vocab: 64,
    layers: 2,
    experts: 8,
    top_k: 2,
    moe_inter: 64,
    shared_inter: 64,
};

/// The host argmax of a `[1, vocab]` logits row (the test's own read, not the model's).
fn argmax(logits: &Tensor) -> i32 {
    let row = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn causal(config: &Value, weights: &HashMap<String, Tensor>, quant: Option<QuantSpec>) -> CausalLm {
    let cfg = ModelConfig::from_json(config).expect("config parses");
    let w = Weights::from_map(weights.clone(), Device::Cpu);
    CausalLm::from_weights_with(&w, "", cfg, quant).expect("model builds")
}

fn qwen35(config: &Value, weights: &HashMap<String, Tensor>) -> Qwen35Model {
    let cfg = Qwen35Config::from_json(config).expect("config parses");
    assert!(cfg.moe.is_some(), "the fixture is the MoE variant");
    let w = Weights::from_map(weights.clone(), Device::Cpu);
    Qwen35Model::from_weights(&w, "model.language_model", cfg).expect("model builds")
}

/// Prefill [`PROMPT`], then decode [`GREEDY_STEPS`] tokens greedily, one per step.
fn greedy_causal(model: &CausalLm) -> Vec<i32> {
    let mut cache = model.new_cache();
    let dev = Device::Cpu;
    let mut tok = argmax(
        &model
            .decode_logits(&input_ids(&PROMPT, &dev).unwrap(), &mut cache, 0)
            .unwrap(),
    );
    let mut out = vec![tok];
    for step in 0..GREEDY_STEPS {
        let offset = (PROMPT.len() + step) as i32;
        let logits = model
            .decode_logits(&input_ids(&[tok], &dev).unwrap(), &mut cache, offset)
            .unwrap();
        tok = argmax(&logits);
        out.push(tok);
    }
    out
}

fn greedy_qwen35(model: &Qwen35Model) -> Vec<i32> {
    let mut cache = model.new_cache();
    let dev = Device::Cpu;
    let mut tok = argmax(
        &model
            .decode_logits(&input_ids(&PROMPT, &dev).unwrap(), &mut cache, 0)
            .unwrap(),
    );
    let mut out = vec![tok];
    for step in 0..GREEDY_STEPS {
        let offset = (PROMPT.len() + step) as i32;
        let logits = model
            .decode_logits(&input_ids(&[tok], &dev).unwrap(), &mut cache, offset)
            .unwrap();
        tok = argmax(&logits);
        out.push(tok);
    }
    out
}

/// Every fixture's greedy continuation, keyed by fixture name.
fn greedy_cases() -> Vec<(&'static str, Vec<i32>)> {
    let (q2, q2w) = qwen2_moe(TINY, false, 0x2444_0001);
    let (q2n, q2nw) = qwen2_moe(WIDE, true, 0x2444_0002);
    let (ds, dsw) = deepseek_v2(0x2444_0003);
    let (q35, q35w) = qwen35_moe(0x2444_0004);
    vec![
        ("qwen2_moe", greedy_causal(&causal(&q2, &q2w, None))),
        ("qwen2_moe_norm", greedy_causal(&causal(&q2n, &q2nw, None))),
        (
            "qwen2_moe_norm_q8",
            greedy_causal(&causal(&q2n, &q2nw, Some(QuantSpec::q8()))),
        ),
        (
            "qwen2_moe_norm_q4",
            greedy_causal(&causal(&q2n, &q2nw, Some(QuantSpec::q4()))),
        ),
        ("deepseek_v2", greedy_causal(&causal(&ds, &dsw, None))),
        ("qwen35_moe", greedy_qwen35(&qwen35(&q35, &q35w))),
    ]
}

/// **AC1.** Every MoE family's greedy tokens are identical to the pre-change router's — through
/// the CPU's grouped dispatch and through the stacked device dispatch a GPU step takes.
#[test]
fn moe_greedy_tokens_match_the_pre_change_router() {
    let cases = greedy_cases();
    if std::env::var("SC24440_WRITE_MOE_GREEDY").is_ok() {
        let mut doc = Map::new();
        doc.insert(
            "_note".into(),
            Value::String(
                "Generated by `SC24440_WRITE_MOE_GREEDY=1 cargo test -p candle-llm --test \
                 moe_routing` on the base branch f4ce41e50, before sc-24440 moved MoE routing \
                 onto the device: these are the pre-change router's greedy tokens."
                    .into(),
            ),
        );
        for (name, tokens) in &cases {
            doc.insert((*name).into(), json!(tokens));
        }
        std::fs::write(
            golden_path(),
            serde_json::to_string_pretty(&doc).unwrap() + "\n",
        )
        .unwrap();
        eprintln!("wrote {}", golden_path().display());
        return;
    }
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_path()).expect("the MoE greedy goldens are committed"),
    )
    .unwrap();
    let device = with_device_dispatch(greedy_cases);
    for (name, got) in cases.iter().chain(&device) {
        let want: Vec<i32> = golden[*name]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: missing from the MoE greedy goldens"))
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(
            got, &want,
            "{name}: greedy tokens diverged from the pre-change router"
        );
    }
}

/// Prefill [`PROMPT`] through `step` (`tokens`, `offset`), then count the host syncs and MoE
/// device dispatches of one decode step — both dispatched as a GPU would dispatch them.
fn decode_step_counts(mut step: impl FnMut(&[i32], i32)) -> (u64, u64) {
    with_device_dispatch(|| {
        step(&PROMPT, 0);
        let (syncs, dispatches) = (host_sync_count(), moe_device_dispatch_count());
        step(&[7], PROMPT.len() as i32);
        (
            host_sync_count() - syncs,
            moe_device_dispatch_count() - dispatches,
        )
    })
}

/// **AC2.** A decode step on an MoE fixture routes every MoE layer on the device and reads nothing
/// back to the host (see the module docs for what each counter can and cannot see).
#[test]
fn an_moe_decode_step_reads_nothing_back_to_the_host() {
    let dev = Device::Cpu;
    let (q2, q2w) = qwen2_moe(TINY, false, 0x2444_0001);
    let (ds, dsw) = deepseek_v2(0x2444_0003);
    // Qwen2-MoE: every layer sparse; DeepSeek-V2: the first layer dense.
    for (name, model, moe_layers) in [
        ("qwen2_moe", causal(&q2, &q2w, None), TINY.layers as u64),
        ("deepseek_v2", causal(&ds, &dsw, None), 2),
    ] {
        let mut cache = model.new_cache();
        let counts = decode_step_counts(|tokens, offset| {
            model
                .decode_logits(&input_ids(tokens, &dev).unwrap(), &mut cache, offset)
                .unwrap();
        });
        assert_eq!(
            counts,
            (0, moe_layers),
            "{name}: (host syncs, MoE layers dispatched on the device) of a decode step"
        );
    }

    let (q35, q35w) = qwen35_moe(0x2444_0004);
    let model = qwen35(&q35, &q35w);
    let mut cache = model.new_cache();
    let counts = decode_step_counts(|tokens, offset| {
        model
            .decode_logits(&input_ids(tokens, &dev).unwrap(), &mut cache, offset)
            .unwrap();
    });
    assert_eq!(
        counts,
        (0, 4),
        "qwen35_moe: (host syncs, MoE layers dispatched on the device) of a decode step"
    );
}

/// **AC3.** `graph_support()` no longer refuses an MoE model for its router.
#[test]
fn moe_models_are_not_refused_for_a_router_host_read() {
    let (q2, q2w) = qwen2_moe(TINY, false, 0x2444_0001);
    let (ds, dsw) = deepseek_v2(0x2444_0003);
    let (q35, q35w) = qwen35_moe(0x2444_0004);
    let reasons = [
        ("qwen2_moe", causal(&q2, &q2w, None).graph_support()),
        ("deepseek_v2", causal(&ds, &dsw, None).graph_support()),
        ("qwen35_moe", qwen35(&q35, &q35w).graph_support()),
    ];
    for (name, reason) in reasons {
        assert_ne!(
            reason,
            Err("moe_router_host_read"),
            "{name}: still refused for its router"
        );
    }
}

/// sc-24440 part 2: the indexed MoE dispatch on CUDA, end to end. Each fixture runs twice on the
/// device: eagerly with every MoE step dispatched grouped (routes read back, each expert on its
/// own tokens — the path the indexed kernels replace), and through the CUDA-graph runner with
/// the default dispatch (indexed). The greedy tokens are identical, the decode steps are
/// captured and replayed, and one eager decode step on the indexed path issues no host sync, runs
/// every MoE layer indexed and gathers no expert matrix.
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_llm::decode::graph::cuda_graphs_policy_guard;
    use candle_llm::decode::{
        generate_step, CancelFlag, GenerationConfig, GraphRunner, StepModel, StepRequest,
    };
    use candle_llm::device::select_device;
    use candle_llm::primitives::moe::{
        moe_expert_gather_count, moe_indexed_dispatch_count, with_grouped_dispatch,
    };
    use candle_llm::primitives::ProjectionFormat;

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        let mut config = GenerationConfig {
            max_new_tokens,
            seed: Some(0),
            stop_tokens: Vec::new(),
            ..Default::default()
        };
        config.sampling.temperature = 0.0;
        config
    }

    fn on(device: &Device, weights: &HashMap<String, Tensor>) -> Weights {
        let moved = weights
            .iter()
            .map(|(k, v)| (k.clone(), v.to_device(device).unwrap()))
            .collect();
        Weights::from_map(moved, device.clone())
    }

    /// Run the comparison on one model (`moe_layers` MoE layers).
    fn check<M: StepModel>(name: &str, model: &M, moe_layers: u64) {
        assert_eq!(model.graph_support(), Ok(()), "{name}: graph support");
        let config = greedy(GREEDY_STEPS);
        let (grouped, _) = with_grouped_dispatch(|| {
            generate_step(
                model,
                &PROMPT,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
        })
        .unwrap();
        let gathers = moe_expert_gather_count();
        let runner = GraphRunner::new(model);
        let (indexed, record) = generate_step(
            &runner,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        eprintln!(
            "[moe_routing] {name}: {} — {:?}",
            record.cuda_graphs.describe(),
            indexed.tokens
        );
        assert_eq!(
            indexed.tokens, grouped.tokens,
            "{name}: indexed dispatch diverged from the grouped dispatch"
        );
        assert_eq!(record.cuda_graphs.fallback_reason, None, "{name}");
        assert!(record.cuda_graphs.replayed > 0, "{name}: no graph replayed");
        assert_eq!(
            moe_expert_gather_count(),
            gathers,
            "{name}: an expert was gathered"
        );

        // One eager decode step on the indexed path.
        let mut cache = model.new_cache_for(PROMPT.len() + 4, 0).unwrap();
        model
            .forward_step(&mut cache, StepRequest::last(&PROMPT))
            .unwrap();
        model.device().synchronize().unwrap();
        let before = (
            host_sync_count(),
            moe_indexed_dispatch_count(),
            moe_expert_gather_count(),
        );
        model
            .forward_step(&mut cache, StepRequest::last(&[7]))
            .unwrap();
        assert_eq!(
            (
                host_sync_count() - before.0,
                moe_indexed_dispatch_count() - before.1,
                moe_expert_gather_count() - before.2,
            ),
            (0, moe_layers, 0),
            "{name}: (host syncs, indexed MoE layers, expert gathers) of one decode step"
        );
    }

    /// A Qwen2-MoE whose every projection input is a whole Q4_K super-block (256), so a Q4 load
    /// stores its experts Q4_K (the narrower fixtures fall back to Q4_0).
    const K256: Dims = Dims {
        hidden: 256,
        heads: 4,
        head_dim: 64,
        vocab: 64,
        layers: 2,
        experts: 8,
        top_k: 2,
        moe_inter: 256,
        shared_inter: 256,
    };

    #[test]
    fn moe_fixtures_decode_indexed_on_cuda_token_identical_to_the_grouped_dispatch() {
        let _graphs = cuda_graphs_policy_guard(Some(true));
        let device = match select_device() {
            Ok(d @ Device::Cuda(_)) => d,
            _ => {
                eprintln!("skipping: no CUDA device");
                return;
            }
        };
        let causal_on = |config: &Value, w: &HashMap<String, Tensor>, quant| {
            let cfg = ModelConfig::from_json(config).unwrap();
            CausalLm::from_weights_with(&on(&device, w), "", cfg, quant).unwrap()
        };
        let (q2, q2w) = qwen2_moe(TINY, false, 0x2444_0001);
        let (q2n, q2nw) = qwen2_moe(WIDE, true, 0x2444_0002);
        let (qk, qkw) = qwen2_moe(K256, true, 0x2444_0005);
        let (ds, dsw) = deepseek_v2(0x2444_0003);
        let (q35, q35w) = qwen35_moe(0x2444_0004);
        check("qwen2_moe", &causal_on(&q2, &q2w, None), TINY.layers as u64);
        check(
            "qwen2_moe_norm",
            &causal_on(&q2n, &q2nw, None),
            WIDE.layers as u64,
        );
        check(
            "qwen2_moe_norm_q8",
            &causal_on(&q2n, &q2nw, Some(QuantSpec::q8())),
            WIDE.layers as u64,
        );
        check(
            "qwen2_moe_norm_q4 (Q4_0)",
            &causal_on(&q2n, &q2nw, Some(QuantSpec::q4())),
            WIDE.layers as u64,
        );
        check(
            "qwen2_moe_k256_q4 (Q4_K)",
            &causal_on(&qk, &qkw, Some(QuantSpec::q4())),
            K256.layers as u64,
        );
        // (DeepSeek-V2's MLA projections are 24 wide in this fixture — no whole GGML block — so
        // it is checked dense only, as the CPU goldens do.)
        check("deepseek_v2", &causal_on(&ds, &dsw, None), 2);
        let qwen35 = Qwen35Model::from_weights(
            &on(&device, &q35w),
            "model.language_model",
            Qwen35Config::from_json(&q35).unwrap(),
        )
        .unwrap();
        check("qwen35_moe", &qwen35, 4);
        match ProjectionFormat::nvfp4(&device) {
            Ok(format) => {
                let cfg = ModelConfig::from_json(&q2n).unwrap();
                let model =
                    CausalLm::from_weights_format(&on(&device, &q2nw), "", cfg, Some(&format))
                        .unwrap();
                check("qwen2_moe_norm_nvfp4", &model, WIDE.layers as u64);
            }
            Err(why) => {
                candle_quant_kernels::skip_without_sm120(&format!("NVFP4 MoE fixture ({why})"))
            }
        }
    }
}

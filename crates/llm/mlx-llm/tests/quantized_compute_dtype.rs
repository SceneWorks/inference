//! sc-20671: a stored quantized checkpoint runs — and caches K/V — in the loader's compute dtype
//! whatever dtype it stores its affine `scales`/`biases` in.
//!
//! mlx-community checkpoints such as `Llama-3.2-3B-Instruct-4bit` store scales, biases and norms as
//! F16. MLX's affine `quantized_matmul` returns `promote_types(x, scales)`, and BF16 activations
//! against F16 scales promote to F32. A loader that keeps the stored scale dtype therefore runs the
//! whole decoder, and its KV cache, in F32 (4 bytes). These fixtures write a tiny synthetic
//! pre-quantized Llama snapshot to disk and load it through the production route
//! (`Weights::from_dir` → `ModelConfig::from_json` → `CausalLm::from_weights_with`). They then
//! assert the per-layer activations, every cached K/V tensor, and the logits stay BF16. The
//! BF16-scale twins are the control: casting to the compute dtype must not change a checkpoint
//! already stored in it.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};
use serde_json::json;

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::quant::QuantizedLinear;
use mlx_llm::primitives::sampler::{SplitMix64, TokenRng};
use mlx_llm::primitives::{input_ids, Weights};

use crate::common::Fixture;

const HIDDEN: i32 = 64;
const VOCAB: i32 = 32;
const HEAD_DIM: i32 = 16;
const LAYERS: usize = 2;
const HEADS: i32 = 4;
const KV: i32 = 2;
const INTER: i32 = 128;
const GROUP: i32 = 32;
const COMPUTE: Dtype = Dtype::Bfloat16;

fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Array::from_slice(&data, shape)
}

/// Store `dense` under `base` as MLX packed Q4 parts whose `scales`/`biases` carry `stored`.
fn packed(tensors: &mut HashMap<String, Array>, base: &str, dense: &Array, stored: Dtype) {
    let q = QuantizedLinear::quantize(&dense.as_dtype(stored).unwrap(), GROUP, 4, None).unwrap();
    assert_eq!(q.scales.dtype(), stored);
    assert_eq!(q.biases.dtype(), stored);
    tensors.insert(format!("{base}.weight"), q.weight);
    tensors.insert(format!("{base}.scales"), q.scales);
    tensors.insert(format!("{base}.biases"), q.biases);
}

/// A tiny pre-quantized Llama snapshot on disk: every projection and the token embedding packed
/// with `stored` scales, norms stored in `stored` too (the mlx-community layout). `tied` shares the
/// packed embedding as the LM head; otherwise the head is its own packed tensor.
fn write_snapshot(stored: Dtype, tied: bool) -> Fixture {
    let mut rng = SplitMix64::new(0x2067_1000);
    let mut tensors = HashMap::new();
    let norm = || {
        Array::ones::<f32>(&[HIDDEN])
            .unwrap()
            .as_dtype(stored)
            .unwrap()
    };
    packed(
        &mut tensors,
        "model.embed_tokens",
        &randn(&[VOCAB, HIDDEN], &mut rng),
        stored,
    );
    if !tied {
        packed(
            &mut tensors,
            "lm_head",
            &randn(&[VOCAB, HIDDEN], &mut rng),
            stored,
        );
    }
    tensors.insert("model.norm.weight".into(), norm());
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        tensors.insert(p("input_layernorm.weight"), norm());
        tensors.insert(p("post_attention_layernorm.weight"), norm());
        for (name, out, inp) in [
            ("self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.k_proj", KV * HEAD_DIM, HIDDEN),
            ("self_attn.v_proj", KV * HEAD_DIM, HIDDEN),
            ("self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM),
            ("mlp.gate_proj", INTER, HIDDEN),
            ("mlp.up_proj", INTER, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTER),
        ] {
            packed(
                &mut tensors,
                &p(name),
                &randn(&[out, inp], &mut rng),
                stored,
            );
        }
    }
    let dir = Fixture::new("mlx-quantized-compute-dtype-", None);
    let refs: Vec<_> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
    let config = json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HEAD_DIM,
        "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
        "tie_word_embeddings": tied,
        "quantization": { "group_size": GROUP, "bits": 4 }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    dir
}

/// Load `dir` through the production route and assert every activation, cached K/V tensor and the
/// logits are in the compute dtype after a prefill plus one cached decode step.
fn assert_runs_in_compute_dtype(dir: &Fixture, label: &str) {
    let weights = Weights::from_dir(dir.root()).unwrap();
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let cfg = ModelConfig::from_json(&value).unwrap();
    let model = CausalLm::from_weights_with(&weights, "", cfg, None).unwrap();
    assert!(model.is_quantized(), "{label}: stored parts load packed");

    let mut probe = model.new_cache();
    let states = model
        .hidden_states(&input_ids(&[1, 2, 3, 4]), &mut probe, 0)
        .unwrap();
    assert_eq!(states.len(), LAYERS + 1);
    for (i, state) in states.iter().enumerate() {
        assert_eq!(state.dtype(), COMPUTE, "{label}: hidden state {i}");
    }

    let mut cache = model.new_cache();
    let prefill = model
        .decode_logits(&input_ids(&[1, 2, 3, 4]), &mut cache, 0)
        .unwrap();
    let step = model
        .decode_logits(&input_ids(&[5]), &mut cache, 4)
        .unwrap();
    for logits in [&prefill, &step] {
        assert_eq!(logits.shape(), &[1, VOCAB]);
        assert_eq!(logits.dtype(), COMPUTE, "{label}: logits");
    }
    for layer in 0..LAYERS {
        let (keys, values) = cache.peek(layer).unwrap().expect("layer cached");
        assert_eq!(keys.dtype(), COMPUTE, "{label}: layer {layer} keys");
        assert_eq!(values.dtype(), COMPUTE, "{label}: layer {layer} values");
    }
    assert_eq!(
        cache.element_bytes().unwrap(),
        Some(2),
        "{label}: dense KV element width"
    );
}

#[test]
fn f16_scale_snapshot_runs_and_caches_in_compute_dtype() {
    for tied in [true, false] {
        let dir = write_snapshot(Dtype::Float16, tied);
        assert_runs_in_compute_dtype(&dir, &format!("F16 scales, tied={tied}"));
    }
}

#[test]
fn bf16_scale_snapshot_runs_and_caches_in_compute_dtype() {
    for tied in [true, false] {
        let dir = write_snapshot(Dtype::Bfloat16, tied);
        assert_runs_in_compute_dtype(&dir, &format!("BF16 scales, tied={tied}"));
    }
}

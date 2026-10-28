//! Synthetic dense Qwen3-VL precision contracts. Supported-host MLX tests only;
//! these are not a claim of local Windows compilation or real model weights.
use super::*;
use std::collections::HashMap;

fn tiny() -> (Weights, ModelConfig) {
    let config = ModelConfig::from_json(&serde_json::json!({
        "model_type":"qwen3_vl", "architectures":["Qwen3VLForConditionalGeneration"],
        "text_config": {"model_type":"qwen3_vl_text", "hidden_size":16, "intermediate_size":32,
            "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1, "head_dim":8,
            "vocab_size":32, "rms_norm_eps":1e-6, "tie_word_embeddings":false,
            "rope_parameters":{"rope_type":"default","rope_theta":5000000.0,"mrope_section":[1,1,2]},
            "max_position_embeddings":512}
    })).unwrap();
    let mut weights = HashMap::new();
    let data = |shape: &[i32], phase: f32| {
        let n: i32 = shape.iter().product();
        let values = (0..n)
            .map(|i| ((i as f32 + phase) * 0.173).sin() * 0.071)
            .collect::<Vec<_>>();
        Array::from_slice(&values, shape)
    };
    let ones = |n| Array::ones::<f32>(&[n]).unwrap();
    weights.insert(
        "model.language_model.embed_tokens.weight".into(),
        data(&[32, 16], 0.3),
    );
    weights.insert("model.language_model.norm.weight".into(), ones(16));
    weights.insert("lm_head.weight".into(), data(&[32, 16], 0.7));
    for layer in 0..2 {
        let prefix = |key: &str| format!("model.language_model.layers.{layer}.{key}");
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            weights.insert(prefix(name), ones(16));
        }
        for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            weights.insert(prefix(name), ones(8));
        }
        for (name, shape) in [
            ("self_attn.q_proj.weight", [16, 16]),
            ("self_attn.k_proj.weight", [8, 16]),
            ("self_attn.v_proj.weight", [8, 16]),
            ("self_attn.o_proj.weight", [16, 16]),
            ("mlp.gate_proj.weight", [32, 16]),
            ("mlp.up_proj.weight", [32, 16]),
            ("mlp.down_proj.weight", [16, 32]),
        ] {
            weights.insert(prefix(name), data(&shape, layer as f32 + 0.2));
        }
    }
    (Weights::from_map(weights), config)
}
fn bits(array: &Array) -> Vec<u32> {
    let flat = array
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    mlx_rs::transforms::eval([&flat]).unwrap();
    flat.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
}

#[test]
fn legacy_constructor_and_explicit_bf16_are_identical() {
    let (weights, cfg) = tiny();
    let legacy = CausalLm::from_weights(&weights, "", cfg.clone()).unwrap();
    let explicit = CausalLm::from_weights_dtype(&weights, "", cfg, Dtype::Bfloat16).unwrap();
    assert_eq!(legacy.compute_dtype(), Dtype::Bfloat16);
    assert_eq!(explicit.compute_dtype(), Dtype::Bfloat16);
    assert_eq!(legacy.cache_fingerprint(), explicit.cache_fingerprint());
    let ids = Array::from_slice(&[1i32, 3, 5, 7], &[1, 4]);
    let mut a = legacy.new_cache();
    let mut b = explicit.new_cache();
    assert_eq!(
        bits(&legacy.decode_logits(&ids, &mut a, 0).unwrap()),
        bits(&explicit.decode_logits(&ids, &mut b, 0).unwrap())
    );
    for layer in 0..2 {
        let (ak, av) = a.peek(layer).unwrap().unwrap();
        let (bk, bv) = b.peek(layer).unwrap().unwrap();
        assert_eq!(bits(&ak), bits(&bk));
        assert_eq!(bits(&av), bits(&bv));
    }
}

#[test]
fn explicit_precision_changes_real_weights_outputs_rope_and_cache_identity() {
    let (weights, cfg) = tiny();
    let baseline = CausalLm::from_weights(&weights, "", cfg.clone()).unwrap();
    let baseline_identity = baseline.cache_fingerprint();
    let ids = Array::from_slice(&[1i32, 3, 5, 7], &[1, 4]);
    let mut baseline_cache = baseline.new_cache();
    let baseline_values = bits(
        &baseline
            .decode_logits(&ids, &mut baseline_cache, 0)
            .unwrap(),
    );
    let mut identities = std::collections::BTreeSet::from([baseline_identity]);
    for dtype in [Dtype::Float32, Dtype::Float16] {
        let model = CausalLm::from_weights_dtype(&weights, "", cfg.clone(), dtype).unwrap();
        assert_eq!(model.compute_dtype(), dtype);
        assert!(
            model
                .param_groups()
                .into_iter()
                .flatten()
                .all(|array| array.dtype() == dtype),
            "not just a reported dtype"
        );
        assert_eq!(model.embed(&ids).unwrap().dtype(), dtype);
        let (cos, sin) = model.rope_tables(&[0, 1, 2, 3], 1, 4).unwrap();
        assert_eq!(cos.dtype(), dtype);
        assert_eq!(sin.dtype(), dtype);
        let mut cache = model.new_cache();
        let output = model.decode_logits(&ids, &mut cache, 0).unwrap();
        assert_eq!(output.dtype(), dtype);
        let (k, v) = cache.peek(0).unwrap().unwrap();
        assert_eq!(k.dtype(), dtype);
        assert_eq!(v.dtype(), dtype);
        assert!(
            identities.insert(model.cache_fingerprint()),
            "different compute/cache precision must not alias"
        );
        assert_ne!(
            bits(&output),
            baseline_values,
            "precasting alone must not be erased by a hidden BF16 cast"
        );
    }
}

#[test]
fn training_frequency_profile_is_explicit_and_separates_nondefault_cache_math() {
    let (weights, cfg) = tiny();
    for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
        let ordinary = CausalLm::from_weights_dtype(&weights, "", cfg.clone(), dtype).unwrap();
        let training = CausalLm::from_weights_dtype(&weights, "", cfg.clone(), dtype)
            .unwrap()
            .with_training_frequency_storage()
            .unwrap();
        assert_eq!(ordinary.training_frequency_storage_dtype(), None);
        assert_eq!(training.training_frequency_storage_dtype(), Some(dtype));
        let (normal, _) = ordinary.rope_tables(&[0, 127, 511], 1, 3).unwrap();
        let (rounded, _) = training.rope_tables(&[0, 127, 511], 1, 3).unwrap();
        if dtype == Dtype::Float32 {
            assert_eq!(ordinary.cache_fingerprint(), training.cache_fingerprint());
            assert_eq!(bits(&normal), bits(&rounded));
        } else {
            assert_ne!(ordinary.cache_fingerprint(), training.cache_fingerprint());
            assert_ne!(
                bits(&normal),
                bits(&rounded),
                "actual frequency-row cache must be rebuilt, not just relabelled"
            );
        }
    }
}

#[test]
fn unsupported_precision_or_quantized_teacher_is_rejected_before_loading() {
    let (_, cfg) = tiny();
    let empty = Weights::from_map(HashMap::new());
    assert!(matches!(
        CausalLm::from_weights_dtype(&empty, "", cfg.clone(), Dtype::Uint32),
        Err(Error::Unsupported(_))
    ));
    let mut packed = cfg;
    packed.quantization = Some(QuantSpec::q4());
    assert!(matches!(
        CausalLm::from_weights_dtype(&empty, "", packed, Dtype::Float32),
        Err(Error::Unsupported(_))
    ));
}

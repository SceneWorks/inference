//! Synthetic CPU profile tests; no decoder forward requires unsupported CPU BF16 GEMM.
use super::*;
use std::collections::HashMap;
fn tiny(dtype: DType) -> CausalLm {
    let cfg=ModelConfig::from_json(&serde_json::json!({"model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],"text_config":{"model_type":"qwen3_vl_text","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-6,"rope_theta":5000000.0,"tie_word_embeddings":false}})).unwrap();
    let mut map = HashMap::new();
    let value = |shape: &[usize]| Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap();
    map.insert(
        "model.language_model.embed_tokens.weight".into(),
        value(&[32, 16]),
    );
    map.insert("model.language_model.norm.weight".into(), value(&[16]));
    map.insert("lm_head.weight".into(), value(&[32, 16]));
    for key in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        map.insert(format!("model.language_model.layers.0.{key}"), value(&[16]));
    }
    for key in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
        map.insert(format!("model.language_model.layers.0.{key}"), value(&[8]));
    }
    for (key, shape) in [
        ("self_attn.q_proj.weight", [16, 16]),
        ("self_attn.k_proj.weight", [8, 16]),
        ("self_attn.v_proj.weight", [8, 16]),
        ("self_attn.o_proj.weight", [16, 16]),
        ("mlp.gate_proj.weight", [32, 16]),
        ("mlp.up_proj.weight", [32, 16]),
        ("mlp.down_proj.weight", [16, 32]),
    ] {
        map.insert(
            format!("model.language_model.layers.0.{key}"),
            value(&shape),
        );
    }
    CausalLm::from_weights_dtype(&Weights::from_map(map, Device::Cpu), "", cfg, None, dtype)
        .unwrap()
}
fn values(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}
#[test]
fn training_profile_rebuilds_device_schedule_and_changes_profile_identity_only_explicitly() {
    let original = tiny(DType::BF16);
    let original_key = original.precision_profile_key();
    let trained = tiny(DType::BF16).with_training_frequency_storage().unwrap();
    assert_eq!(original.training_frequency_storage_dtype(), None);
    assert_eq!(
        trained.training_frequency_storage_dtype(),
        Some(DType::BF16)
    );
    assert_ne!(original_key, trained.precision_profile_key());
    let positions = [0i32, 127, 511];
    let (host_cos, host_sin) = trained
        .rope
        .cos_sin_at(&positions, DType::F32, &Device::Cpu)
        .unwrap();
    let positions = Tensor::new(&positions.map(|p| p as f32), &Device::Cpu).unwrap();
    let (device_cos, device_sin) = trained.device_rope.cos_sin(&positions, DType::F32).unwrap();
    assert_eq!(values(&host_cos), values(&device_cos));
    assert_eq!(values(&host_sin), values(&device_sin));
    assert_eq!(
        tiny(DType::BF16).precision_profile_key(),
        original_key,
        "old defaults must remain unchanged"
    );
}

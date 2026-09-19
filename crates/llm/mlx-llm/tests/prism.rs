//! Tiny executable Prism/Bonsai provider fixture. The tensors stay packed through load and decode.

use core_llm::{LoadSpec, Message, Sampling, StreamEvent, TextLlm, TextLlmRequest};
use mlx_llm::LlamaProvider;
use mlx_rs::{Array, Dtype};
use serde_json::json;

use crate::common::{assert_fixture_is_self_removing, Fixture};

const TOKENIZER: &str = r#"{
 "version":"1.0","added_tokens":[],"normalizer":null,
 "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
 "model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"world":2,"ok":3},"unk_token":"<unk>"}
}"#;

fn packed(tensors: &mut Vec<(String, Array)>, path: &str, rows: i32, width: i32) {
    let words = vec![0x5555_5555u32; (rows * width / 16) as usize]; // code 1 => zero
    let affine = Array::from_slice(
        &vec![1.0f32; (rows * width / 128) as usize],
        &[rows, width / 128],
    )
    .as_dtype(Dtype::Float16)
    .unwrap();
    tensors.extend([
        (
            format!("{path}.weight"),
            Array::from_slice(&words, &[rows, width / 16]),
        ),
        (format!("{path}.scales"), affine.clone()),
        (
            format!("{path}.biases"),
            Array::from_slice(
                &vec![-1.0f32; (rows * width / 128) as usize],
                &[rows, width / 128],
            )
            .as_dtype(Dtype::Float16)
            .unwrap(),
        ),
        (
            format!("{path}.signs"),
            Array::ones::<f32>(&[width]).unwrap(),
        ),
    ]);
}

fn write_fixture() -> Fixture {
    let dir = Fixture::new("mlx-prism-provider-", None);
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER).unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":99}"#).unwrap();
    let paths = [
        ("lm_head", false, 4, 128),
        ("model.embed_tokens", true, 4, 128),
        ("model.layers.0.self_attn.q_proj", false, 256, 128),
        ("model.layers.0.self_attn.k_proj", false, 64, 128),
        ("model.layers.0.self_attn.v_proj", false, 64, 128),
        ("model.layers.0.self_attn.o_proj", false, 128, 128),
        ("model.layers.0.mlp.gate_proj", false, 128, 128),
        ("model.layers.0.mlp.up_proj", false, 128, 128),
        ("model.layers.0.mlp.down_proj", false, 128, 128),
    ];
    let config = json!({
        "schema_version":2,"model_type":"prism_hadamard_qwen35",
        "base_model_type":"qwen3_5","tensor_namespace":"mlx-vlm-qwen3_5",
        "requires_runtime":"runtime/artifact.py","hadamard_config":"hadamard.json",
        "gdn_activation_layout":"grouped","components":{"text":true,"vision":true,"mtp":false},
        "quantization":{"bits":2,"group_size":128,"mode":"affine"},
        "image_token_id":248056,"video_token_id":248057,
        "vision_config":{
            "depth":1,"hidden_size":8,"num_heads":2,"intermediate_size":16,
            "in_channels":3,"patch_size":2,"temporal_patch_size":2,
            "spatial_merge_size":2,"out_hidden_size":128,
            "num_position_embeddings":16,"deepstack_visual_indexes":[]
        },
        "text_config":{
            "model_type":"qwen3_5_text","hidden_size":128,"num_hidden_layers":1,
            "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
            "head_dim":64,"vocab_size":4,"rms_norm_eps":0.000001,
            "rope_theta":10000000.0,"partial_rotary_factor":1.0,
            "max_position_embeddings":128,"tie_word_embeddings":false,
            "full_attention_interval":1,"linear_num_value_heads":2,"linear_num_key_heads":1,
            "linear_key_head_dim":64,"linear_value_head_dim":64,"linear_conv_kernel_dim":4,
            "mtp_num_hidden_layers":0,"mtp_use_dedicated_embeddings":false
        },
        "modules": paths.iter().map(|(path, embedding, _, _)| json!({
            "path":path,"block":128,"embedding":embedding,"dtype":"float16"
        })).collect::<Vec<_>>()
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let forward = paths
        .iter()
        .filter(|(_, embedding, _, _)| !embedding)
        .map(|(path, _, _, _)| format!("language_model.{path}.weight"))
        .collect::<Vec<_>>();
    let hadamard = json!({
        "prism.hadamard.version":1,"prism.hadamard.block_size":128,
        "prism.hadamard.transform":"normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis":"input-last-dimension","prism.hadamard.sign_mode":"explicit",
        "prism.hadamard.weight_names":forward,
        "prism.hadamard.inverse_weight_names":["language_model.model.embed_tokens.weight"],
        "prism.hadamard.sign_widths":[128],"prism.hadamard.sign_values":vec![1.0;128],
        "prism.hadamard.gdn_v_grouped":true
    });
    std::fs::write(
        dir.join("hadamard.json"),
        serde_json::to_vec(&hadamard).unwrap(),
    )
    .unwrap();

    let z = |shape: &[i32]| Array::zeros::<f32>(shape).unwrap();
    let p = "language_model.model.layers.0";
    let mut tensors = vec![
        ("language_model.model.norm.weight".into(), z(&[128])),
        (format!("{p}.input_layernorm.weight"), z(&[128])),
        (format!("{p}.post_attention_layernorm.weight"), z(&[128])),
        (format!("{p}.self_attn.q_norm.weight"), z(&[64])),
        (format!("{p}.self_attn.k_norm.weight"), z(&[64])),
    ];
    let v = "vision_tower";
    tensors.extend([
        (format!("{v}.patch_embed.proj.weight"), z(&[8, 3, 2, 2, 2])),
        (format!("{v}.patch_embed.proj.bias"), z(&[8])),
        (format!("{v}.pos_embed.weight"), z(&[16, 8])),
        (
            format!("{v}.blocks.0.norm1.weight"),
            Array::ones::<f32>(&[8]).unwrap(),
        ),
        (format!("{v}.blocks.0.norm1.bias"), z(&[8])),
        (
            format!("{v}.blocks.0.norm2.weight"),
            Array::ones::<f32>(&[8]).unwrap(),
        ),
        (format!("{v}.blocks.0.norm2.bias"), z(&[8])),
        (format!("{v}.blocks.0.attn.qkv.weight"), z(&[24, 8])),
        (format!("{v}.blocks.0.attn.qkv.bias"), z(&[24])),
        (format!("{v}.blocks.0.attn.proj.weight"), z(&[8, 8])),
        (format!("{v}.blocks.0.attn.proj.bias"), z(&[8])),
        (format!("{v}.blocks.0.mlp.linear_fc1.weight"), z(&[16, 8])),
        (format!("{v}.blocks.0.mlp.linear_fc1.bias"), z(&[16])),
        (format!("{v}.blocks.0.mlp.linear_fc2.weight"), z(&[8, 16])),
        (format!("{v}.blocks.0.mlp.linear_fc2.bias"), z(&[8])),
        (
            format!("{v}.merger.norm.weight"),
            Array::ones::<f32>(&[8]).unwrap(),
        ),
        (format!("{v}.merger.norm.bias"), z(&[8])),
        (format!("{v}.merger.linear_fc1.weight"), z(&[32, 32])),
        (format!("{v}.merger.linear_fc1.bias"), z(&[32])),
        (format!("{v}.merger.linear_fc2.weight"), z(&[128, 32])),
        (format!("{v}.merger.linear_fc2.bias"), z(&[128])),
    ]);
    for (path, _, rows, width) in paths {
        packed(&mut tensors, &format!("language_model.{path}"), rows, width);
    }
    let refs: Vec<_> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn provider_loads_and_generates_with_native_packed_operators() {
    let dir = write_fixture();
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    assert!(provider.is_quantized());
    assert!(provider.has_deferred_prism_vision());
    assert!(provider.descriptor().capabilities.supports_vision);
    assert!(provider.descriptor().capabilities.supports_video);
    let sampling = provider
        .descriptor()
        .capabilities
        .model_sampling_defaults
        .expect("Bonsai model-card sampling presets");
    assert_eq!(sampling.thinking.temperature, 1.0);
    assert_eq!(sampling.thinking.top_p, 0.95);
    assert_eq!(sampling.thinking.top_k, 20);
    assert_eq!(sampling.non_thinking.temperature, 0.7);
    assert_eq!(sampling.non_thinking.top_p, 0.8);
    assert_eq!(sampling.non_thinking.top_k, 20);
    assert_eq!(sampling.non_thinking.presence_penalty, 1.5);
    assert_eq!(provider.descriptor().family, "prism_hadamard_qwen35");
    let request = TextLlmRequest {
        messages: vec![Message::user("hello world")],
        sampling: Sampling::greedy(),
        max_new_tokens: 2,
        ..Default::default()
    };
    let mut streamed = 0;
    let output = provider
        .generate(&request, &mut |event| {
            if matches!(event, StreamEvent::Token { .. }) {
                streamed += 1;
            }
        })
        .unwrap();
    assert_eq!(output.usage.generated_tokens, 2);
    assert_eq!(streamed, 2);
    assert_fixture_is_self_removing(dir);
}

#[test]
fn embedded_vision_snapshot_rejects_external_projector_association() {
    let dir = write_fixture();
    let plain = LoadSpec::dense(dir.to_str().unwrap());
    assert!(mlx_llm::provider::can_load(&plain));
    assert!(mlx_llm::provider::weightless_vision(&plain));

    let paired = plain.with_projector(dir.join("projector.gguf").to_str().unwrap());
    assert!(!mlx_llm::provider::can_load(&paired));
    assert!(!mlx_llm::provider::weightless_vision(&paired));
    let error = LlamaProvider::load(&paired)
        .err()
        .expect("embedded snapshots must fail closed instead of ignoring a projector");
    assert!(error
        .to_string()
        .contains("projector_source is only valid for a separable Prism GGUF"));
    assert_fixture_is_self_removing(dir);
}

#[test]
fn malformed_explicit_sign_metadata_fails_closed() {
    let dir = write_fixture();
    let path = dir.join("hadamard.json");
    let mut h: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    h["prism.hadamard.sign_values"][0] = json!(0.0);
    std::fs::write(path, serde_json::to_vec(&h).unwrap()).unwrap();
    let error = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap()))
        .err()
        .expect("malformed metadata must fail");
    assert!(error.to_string().contains("signs must be -1 or +1"));
}

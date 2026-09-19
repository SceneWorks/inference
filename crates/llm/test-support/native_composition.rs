// Shared deterministic native checkpoint. Identity embeddings plus a transition LM head make
// reasoning, answer, EOS, and rejected speculative drafts observable without pretrained weights.
use core_llm::{
    Channel, Constraint, Content, ImageRef, LoadSpec, Message, MtpMode, Sampling, StreamEvent,
    TextLlm, TextLlmRequest, ThinkingMode,
};
use serde_json::json;

fn snapshot(qwen: bool, vision: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut config = json!({"model_type": if qwen {"qwen3_5"} else {"qwen3_vl"},
        "image_token_id": 7, "video_token_id": 8,
        "text_config": {"model_type":if qwen {"qwen3_5_text"} else {"qwen3_vl_text"},
        "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":8,"vocab_size":16,"rms_norm_eps":0.000001,
        "rope_theta":10000.0,"partial_rotary_factor":1.0,"max_position_embeddings":4096,
        "tie_word_embeddings":false,"eos_token_id":6,"full_attention_interval":1,
        "linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":8,
        "linear_value_head_dim":8,"linear_conv_kernel_dim":4,"mtp_num_hidden_layers":if qwen {1} else {0},
        "rope_parameters":{"rope_type":"default","mrope_section":[2,1,1],"partial_rotary_factor":1.0}}});
    if vision {
        config["vision_config"] = json!({"depth":0,"hidden_size":16,"num_heads":2,
            "intermediate_size":32,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
            "out_hidden_size":16,"num_position_embeddings":1024,"deepstack_visual_indexes":[]});
    }
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        r#"{"eos_token_id":6}"#,
    )
    .unwrap();
    let words = [
        "unk",
        "<think>",
        "reason",
        "</think>",
        "{",
        "}",
        "eos",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "a",
        "b",
        "c",
        "d",
        "e",
    ];
    let vocab: serde_json::Map<String, serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), json!(i)))
        .collect();
    let added: Vec<_> = words
        .iter()
        .enumerate()
        .map(|(id, w)| {
            json!({"id":id,"content":w,
        "single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false})
        })
        .collect();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        json!({"version":"1.0","added_tokens":added,
        "normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,
        "decoder":{"type":"Fuse"},"model":{"type":"WordLevel","vocab":vocab,"unk_token":"unk"}})
        .to_string(),
    )
    .unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"),json!({"chat_template":
        "{% for message in messages %}{{ message['content'] }}{% endfor %}{% if enable_thinking is defined and not enable_thinking %} </think>{% else %} <think>{% endif %}"}).to_string()).unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut add = |name: String, shape: &[usize], values: Option<Vec<f32>>, ones: bool| {
        let len: usize = shape.iter().product();
        let start = data.len();
        for value in values.unwrap_or_else(|| vec![if ones { 1.0 } else { 0.0 }; len]) {
            data.extend(value.to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
    };
    let mut identity = vec![0.0; 256];
    for i in 0..16 {
        identity[i * 16 + i] = 1.0;
    }
    add(
        "model.language_model.embed_tokens.weight".into(),
        &[16, 16],
        Some(identity),
        false,
    );
    add(
        "model.language_model.norm.weight".into(),
        &[16],
        None,
        !qwen,
    );
    let mut head = vec![0.0; 256];
    for (from, to) in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 11),
        (11, 6),
        (6, 6),
    ] {
        head[to * 16 + from] = 10.0;
    }
    add("lm_head.weight".into(), &[16, 16], Some(head), false);
    let layers = if qwen {
        vec!["model.language_model.layers.0", "mtp.layers.0"]
    } else {
        vec!["model.language_model.layers.0"]
    };
    for prefix in layers {
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{prefix}.{norm}.weight"), &[16], None, !qwen);
        }
        for (key, shape) in [
            ("q_proj", [if qwen { 32 } else { 16 }, 16]),
            ("k_proj", [8, 16]),
            ("v_proj", [8, 16]),
            ("o_proj", [16, 16]),
        ] {
            add(
                format!("{prefix}.self_attn.{key}.weight"),
                &shape,
                None,
                false,
            );
        }
        for norm in ["q_norm", "k_norm"] {
            add(
                format!("{prefix}.self_attn.{norm}.weight"),
                &[8],
                None,
                !qwen,
            );
        }
        for (key, shape) in [
            ("gate_proj", [32, 16]),
            ("up_proj", [32, 16]),
            ("down_proj", [16, 32]),
        ] {
            add(format!("{prefix}.mlp.{key}.weight"), &shape, None, false);
        }
    }
    if qwen {
        add("mtp.fc.weight".into(), &[16, 32], None, false);
        for norm in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"] {
            add(format!("mtp.{norm}.weight"), &[16], None, false);
        }
    }
    if vision {
        for (name, shape) in [
            ("patch_embed.proj.weight", vec![16, 3, 2, 16, 16]),
            ("pos_embed.weight", vec![1024, 16]),
            ("merger.norm.weight", vec![16]),
            ("merger.norm.bias", vec![16]),
            ("merger.linear_fc1.weight", vec![64, 64]),
            ("merger.linear_fc2.weight", vec![16, 64]),
        ] {
            add(
                format!("model.visual.{name}"),
                &shape,
                None,
                name == "merger.norm.weight",
            );
        }
    }
    let mut encoded = serde_json::to_vec(&header).unwrap();
    while !encoded.len().is_multiple_of(8) {
        encoded.push(b' ');
    }
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend(data);
    std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    dir
}

#[test]
fn native_reasoning_json_stop_and_timing_routes() {
    for qwen in [false, true] {
        for media in 0..3 {
            let vision = media > 0;
            let dir = snapshot(qwen, vision);
            let provider =
                LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string())).unwrap();
            for mode in [
                ThinkingMode::Auto,
                ThinkingMode::Enabled,
                ThinkingMode::Disabled,
            ] {
                for mtp in if qwen {
                    vec![MtpMode::Off, MtpMode::Enabled { draft_tokens: 3 }]
                } else {
                    vec![MtpMode::Off]
                } {
                    let mut message = Message::user("unk");
                    if vision {
                        let image = ImageRef::new(32, 32, vec![127; 32 * 32 * 3]).unwrap();
                        let content = if media == 1 {
                            Content::Image(image)
                        } else {
                            Content::Video(
                                core_llm::VideoRef::new(vec![image.clone(), image], vec![0.0, 1.0])
                                    .unwrap(),
                            )
                        };
                        message.content.insert(0, content);
                    }
                    let mut request = TextLlmRequest {
                        messages: vec![message],
                        max_new_tokens: 12,
                        sampling: Sampling::greedy(),
                        thinking: mode,
                        mtp,
                        constraint: Some(Constraint::Json),
                        ..Default::default()
                    };
                    let mut content = String::new();
                    let mut thinking = String::new();
                    let out = provider
                        .generate(&request, &mut |event| {
                            if let StreamEvent::Token { text, channel, .. } = event {
                                match channel {
                                    Channel::Content => content.push_str(&text),
                                    Channel::Thinking => thinking.push_str(&text),
                                }
                            }
                        })
                        .unwrap();
                    assert_eq!(
                        out.text, "{}",
                        "qwen={qwen} vision={vision} mode={mode:?} mtp={mtp:?}"
                    );
                    assert_eq!(content, out.text);
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&content).unwrap(),
                        json!({})
                    );
                    assert_eq!(
                        thinking,
                        if mode == ThinkingMode::Disabled {
                            ""
                        } else {
                            "reason"
                        }
                    );
                    assert_eq!(out.thinking.as_deref().unwrap_or(""), thinking);
                    assert!(
                        out.timings.is_some(),
                        "every real decoder branch reports phase measurements"
                    );
                    request.constraint = None;
                    request.stop = vec!["reason".into(), "{}".into(), "{}tail".into()];
                    let out = provider.generate(&request, &mut |_| {}).unwrap();
                    assert!(out.text.is_empty());
                    assert_eq!(out.finish_reason, Some(core_llm::FinishReason::Stop));
                    assert_eq!(
                        out.thinking.as_deref().unwrap_or(""),
                        thinking,
                        "reasoning stop text must not stop answer generation"
                    );
                }
            }
        }
    }
}

#[test]
fn native_resource_rejects_within_window_before_allocating() {
    const CHILD: &str = "SCENEWORKS_NATIVE_ADMISSION_FIXTURE";
    if std::env::var_os(CHILD).is_none() {
        let name = std::thread::current().name().unwrap().to_owned();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture"])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    // Isolated child process: the budget override cannot affect concurrent integration tests.
    for qwen in [false, true] {
        let dir = snapshot(qwen, true);
        std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
        let provider =
            LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string())).unwrap();
        let mut message = Message::user("unk");
        message.content.insert(
            0,
            Content::Image(ImageRef::new(32, 32, vec![127; 32 * 32 * 3]).unwrap()),
        );
        let request = TextLlmRequest {
            messages: vec![message],
            max_new_tokens: 64,
            thinking: ThinkingMode::Disabled,
            sampling: Sampling::greedy(),
            ..Default::default()
        };
        std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, "1");
        let error = provider
            .generate(&request, &mut |_| {
                panic!("rejected request emitted a token")
            })
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("bytes of native workspace but only 1 bytes are available"),
            "{error}"
        );
        assert!(
            !error.contains("context window"),
            "must reject a within-window request by memory"
        );
        let error = LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string()))
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("bytes of native workspace but only 1 bytes are available"),
            "{error}"
        );
        std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, "invalid-budget");
        let error = provider
            .generate(&request, &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsigned byte count"), "{error}");
    }
}

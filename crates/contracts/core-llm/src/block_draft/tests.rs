use super::*;

/// `tencent/Qwen3.8-27B-DSpine-b16`'s published `config.json` (Apache-2.0), verbatim but for the
/// training-only teacher keys, which the inference contract ignores.
fn dspine_b16() -> Value {
    serde_json::json!({
        "architectures": ["DSpineDraftModel"],
        "attention_bias": false,
        "block_size": 16,
        "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
        "dflash_config": {
            "draft_block_causal": true,
            "mask_token_id": 248070,
            "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "kappa": 10.0, "rank": 512},
            "selected_prefix_decode_mode": "serial",
            "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": true,
            "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
            "selected_prefix_terminal": false,
            "target_layer_ids": [1, 16, 31, 46, 61]
        },
        "draft_vocab_size": 248320,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention"],
        "max_position_embeddings": 262144,
        "max_window_layers": 5,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 64,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "rope_scaling": null,
        "rope_theta": 10000000,
        "sliding_window": 8192,
        "tie_word_embeddings": false,
        "use_sliding_window": true,
        "vocab_size": 248320
    })
}

/// The Qwen3.8-27B target the draft was trained for.
fn qwen38_27b() -> BlockDraftTarget {
    BlockDraftTarget {
        hidden_size: 5120,
        num_layers: 64,
        vocab_size: 248320,
        tokenizer_vocab: 248077,
    }
}

#[test]
fn the_published_dspine_config_parses_to_its_inference_contract() {
    let cfg = BlockDraftConfig::from_json(&dspine_b16()).unwrap();
    assert_eq!(cfg.format, BlockDraftFormat::DSpine);
    assert_eq!(cfg.target_layer_ids, [1, 16, 31, 46, 61]);
    assert_eq!(cfg.mask_token_id, 248070);
    assert_eq!(cfg.sliding_window, Some(8192));
    assert_eq!(
        cfg.context_rows(),
        Some(8191),
        "a query sees the 8191 rows before it"
    );
    assert_eq!(cfg.head_topk, 16);
    assert_eq!(cfg.max_depth(), 15, "block 16 = anchor + 15 drafts");
    assert_eq!(cfg.tap_width(), 5 * 5120);
    assert!(matches!(
        cfg.head,
        BlockDraftHead::DSpine {
            code_size: 1024,
            bridge_rank: 512,
            ..
        }
    ));
    assert!(cfg.mismatches(&qwen38_27b()).is_empty());
}

#[test]
fn architectures_select_the_format_and_an_ordinary_decoder_is_not_a_block_draft() {
    assert_eq!(
        BlockDraftFormat::from_config(&dspine_b16()),
        Some(BlockDraftFormat::DSpine)
    );
    let ordinary = serde_json::json!({"architectures": ["Qwen3ForCausalLM"]});
    assert_eq!(BlockDraftFormat::from_config(&ordinary), None);
    assert_eq!(BlockDraftFormat::from_config(&serde_json::json!({})), None);
    let err = BlockDraftConfig::from_json(&ordinary).unwrap_err();
    assert_eq!(err.kind(), "format");
}

#[test]
fn every_target_mismatch_is_refused_with_its_typed_reason() {
    let cfg = BlockDraftConfig::from_json(&dspine_b16()).unwrap();
    let kinds = |t: BlockDraftTarget| -> Vec<&'static str> {
        cfg.mismatches(&t)
            .iter()
            .map(BlockDraftRefusal::kind)
            .collect()
    };
    let base = qwen38_27b();
    assert_eq!(
        kinds(BlockDraftTarget {
            hidden_size: 4096,
            ..base
        }),
        ["hidden_size"]
    );
    assert_eq!(
        kinds(BlockDraftTarget {
            vocab_size: 151_936,
            tokenizer_vocab: 151_669,
            ..base
        }),
        ["vocabulary", "mask_token"],
        "a smaller vocabulary also drops the mask token"
    );
    assert_eq!(
        kinds(BlockDraftTarget {
            tokenizer_vocab: 248_000,
            ..base
        }),
        ["mask_token"]
    );
    assert_eq!(
        kinds(BlockDraftTarget {
            num_layers: 48,
            ..base
        }),
        ["target_layers"]
    );
    let mut odd = cfg.clone();
    odd.num_attention_heads = 30;
    assert_eq!(
        odd.mismatches(&base)
            .iter()
            .map(BlockDraftRefusal::kind)
            .collect::<Vec<_>>(),
        ["attention_geometry"]
    );
    let mut repeated = cfg.clone();
    repeated.target_layer_ids = vec![1, 1, 31];
    assert_eq!(
        repeated.mismatches(&base)[0].to_string(),
        "block draft: target_layers: target_layer_ids [1, 1, 31] (built for 64 target layers) \
         do not fit a 64-layer target"
    );
    let reason = block_draft_refusal_reason(&cfg.mismatches(&BlockDraftTarget {
        hidden_size: 4096,
        ..base
    }));
    assert_eq!(
        reason,
        "block draft: hidden_size: hidden_size 5120 != target 4096"
    );
}

#[test]
fn a_config_outside_the_dspine_contract_is_refused_by_name() {
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut raw = dspine_b16();
        edit(&mut raw);
        BlockDraftConfig::from_json(&raw).unwrap_err().to_string()
    };
    assert!(
        refused(&|v| v["dflash_config"]["conv_kernel_size"] = 4.into())
            .contains("`conv_kernel_size`")
    );
    assert!(refused(
        &|v| v["dflash_config"]["selected_prefix_feature_layers"] = serde_json::json!([1, 2, 3, 4])
    )
    .contains("every draft layer must carry a bridge write"));
    assert!(
        refused(&|v| v["layer_types"][0] = "full_attention".into()).contains("one sliding window")
    );
    assert!(
        refused(&|v| v["code_bridge_second_pass"]["head_topk"] = 8.into())
            .contains("refinement widths disagree")
    );
    assert!(
        refused(&|v| v["dflash_config"]["selected_prefix_decode_mode"] = "parallel".into())
            .contains("selected_prefix_decode_mode")
    );
    assert!(
        refused(&|v| v["rope_parameters"]["rope_type"] = "yarn".into()).contains("default RoPE")
    );
    let full = {
        let mut raw = dspine_b16();
        raw["use_sliding_window"] = false.into();
        raw["layer_types"] = serde_json::json!(vec!["full_attention"; 5]);
        raw.as_object_mut().unwrap().remove("sliding_window");
        BlockDraftConfig::from_json(&raw).unwrap()
    };
    assert_eq!(full.sliding_window, None);
    assert_eq!(full.context_rows(), None);
}

#[test]
fn stored_tensors_are_checked_by_name_and_shape_and_none_go_unused() {
    let cfg = BlockDraftConfig::from_json(&dspine_b16()).unwrap();
    let tensors = cfg.tensors();
    // The published checkpoint's 64 tensors: 5 top-level, 11 per layer × 5, 11 bridge.
    assert_eq!(tensors.len(), 5 + 11 * 5 + 11);
    let names: Vec<&str> = tensors.iter().map(|(n, _)| n.as_str()).collect();
    let shape_of = |n: &str| tensors.iter().find(|(m, _)| m == n).map(|(_, s)| s.clone());
    cfg.check_tensors(shape_of, names.iter().copied()).unwrap();
    assert_eq!(shape_of("fc.weight"), Some(vec![5120, 25600]));
    assert_eq!(
        shape_of("selected_prefix_bridge.codebook"),
        Some(vec![248320, 1024])
    );
    let err = cfg
        .check_tensors(
            |n| match n {
                "fc.weight" => Some(vec![5120, 20480]),
                "norm.weight" => None,
                other => shape_of(other),
            },
            names.iter().copied().chain(["lm_head.weight"]),
        )
        .unwrap_err();
    assert_eq!(err.kind(), "tensors");
    let text = err.to_string();
    assert!(text.contains("`fc.weight` is [5120, 20480], expected [5120, 25600]"));
    assert!(text.contains("`norm.weight` is missing"));
    assert!(text.contains("unused tensors lm_head.weight"));
}

#[test]
fn precision_defaults_to_eight_bit_and_prices_below_the_bf16_payload() {
    assert_eq!(
        BlockDraftPrecision::from_request(None),
        Ok(BlockDraftPrecision::Q8)
    );
    assert_eq!(
        BlockDraftPrecision::from_request(Some(crate::Quantize::Q4)),
        Ok(BlockDraftPrecision::Q4)
    );
    assert_eq!(
        BlockDraftPrecision::from_request(Some(crate::Quantize::Nvfp4))
            .unwrap_err()
            .kind(),
        "precision"
    );
    let cfg = BlockDraftConfig::from_json(&dspine_b16()).unwrap();
    let bf16: u64 = cfg
        .tensors()
        .iter()
        .map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>() * 2)
        .sum();
    // The published BF16 checkpoint is 4.05 GB of tensor payload.
    assert!((4_000_000_000..4_100_000_000).contains(&bf16), "{bf16}");
    let (q8, q4) = (
        cfg.load_bytes(BlockDraftPrecision::Q8),
        cfg.load_bytes(BlockDraftPrecision::Q4),
    );
    assert!(q4 < q8 && q8 < bf16, "q4 {q4} < q8 {q8} < bf16 {bf16}");
    // 8-bit: ~1.73 G quantized weights at 1 byte + 1/16 byte of scales/biases, ~0.6 GB of BF16
    // bridge and norms, plus a layer's 640 MB of BF16 projections converting and the previous
    // layer's still on their way back from the driver.
    assert!((3_600_000_000..3_800_000_000).contains(&q8), "{q8}");
}

/// The load is priced by the groups it materializes: every stored tensor read by exactly one
/// group (the checks, each layer, the head), and the peak is the resident draft plus two of the
/// largest group's windows (one converting, one released and outstanding) — for the published
/// draft a layer's seven BF16 projections, not the whole 3.46 GB of them.
#[test]
fn the_load_is_priced_by_the_groups_it_materializes() {
    let cfg = BlockDraftConfig::from_json(&dspine_b16()).unwrap();
    let groups = cfg.load_groups();
    assert_eq!(groups.len(), cfg.num_hidden_layers + 2);
    let mut read: Vec<&str> = groups
        .iter()
        .flat_map(|g| g.tensors.iter().map(String::as_str))
        .collect();
    read.sort_unstable();
    let mut stored: Vec<String> = cfg.tensors().into_iter().map(|(n, _)| n).collect();
    stored.sort_unstable();
    assert_eq!(read, stored, "every tensor in exactly one group");
    assert!(groups[0].tensors.iter().any(|t| t.ends_with("out.weight")));
    assert!(groups[1..=cfg.num_hidden_layers]
        .iter()
        .enumerate()
        .all(|(i, g)| g
            .tensors
            .iter()
            .all(|t| t.starts_with(&format!("layers.{i}.")))));
    let layer_projections: u64 = cfg
        .tensors()
        .iter()
        .filter(|(n, _)| n.starts_with("layers.0.") && cfg.quantized_matrices().contains(n))
        .map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>() * 2)
        .sum();
    let largest = groups.iter().map(|g| g.window).max().unwrap();
    assert_eq!(largest, layer_projections);
    assert!((630_000_000..650_000_000).contains(&largest), "{largest}");
    for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
        assert_eq!(
            cfg.load_bytes(precision),
            cfg.resident_bytes(precision) + 2 * largest
        );
    }
}

#[test]
fn request_bytes_stop_growing_past_the_draft_window() {
    let cfg = BlockDraftConfig::from_json(&dspine_b16()).unwrap();
    let at = |prompt: usize| cfg.request_bytes(prompt, 256, 4).unwrap();
    assert!(at(100) < at(4_000));
    assert_eq!(at(20_000), at(200_000), "E5: the window caps every term");
    let width_cost = cfg.request_bytes(100, 256, 15).unwrap() - at(100);
    assert!(width_cost > 0);
}

#[test]
fn the_advertisement_is_bounded_by_the_verify_depth_and_the_trained_block() {
    let depths = crate::DecodeBackend::Mlx.defaults().recommended_depths;
    let cap = block_draft_capabilities(8, 15, &depths);
    assert_eq!(cap.proposer, crate::SpeculativeProposer::BlockDraft);
    assert_eq!((cap.max_depth, cap.recommended_depth), (8, 4));
    let shallow = block_draft_capabilities(8, 2, &depths);
    assert_eq!((shallow.max_depth, shallow.recommended_depth), (2, 2));
}

// ---- DFlash2 (SW-7194). ----

/// `incoai/Qwen3.8-27B-DFlash2`'s published `config.json` (Apache-2.0; the SceneWorks re-host
/// `SceneWorks/qwen3.8-27b-dflash2` @ `9030fabf` carries the same bytes), verbatim.
fn dflash2_27b() -> Value {
    serde_json::json!({
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": null,
        "is_causal": false,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
        },
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 17408,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention"],
        "max_position_embeddings": 262144,
        "max_window_layers": 5,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 64,
        "pad_token_id": 248044,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 2048,
        "tie_word_embeddings": false,
        "transformers_version": "5.15.0",
        "use_cache": true,
        "use_sliding_window": true,
        "vocab_size": 248320
    })
}

#[test]
fn the_published_dflash2_config_parses_to_its_inference_contract() {
    let cfg = BlockDraftConfig::from_json(&dflash2_27b()).unwrap();
    assert_eq!(cfg.format, BlockDraftFormat::DFlash2);
    assert_eq!(cfg.format.label(), "dflash2");
    assert_eq!(cfg.target_layer_ids, [5, 19, 33, 47, 61]);
    assert_eq!(cfg.mask_token_id, 248070);
    assert_eq!(cfg.sliding_window, Some(2048));
    assert_eq!(cfg.context_rows(), Some(2047), "E5: the 2,048-token window");
    assert_eq!(cfg.max_depth(), 7, "block 8 = anchor + 7 drafts");
    assert_eq!(cfg.head_topk, 16, "selector_top_k");
    assert_eq!(cfg.rope_theta, 1e7);
    assert_eq!(
        cfg.head,
        BlockDraftHead::DFlash2 {
            conv_kernel_size: 2,
            conv_group_size: 16,
            selector_rank: 256,
            causal: false,
            input_embedding_scale: 1.0,
            output_multiplier: 1.0,
            final_logit_softcapping: None,
        }
    );
    assert!(cfg.mismatches(&qwen38_27b()).is_empty());
}

/// AC2: the format is the checkpoint's `architectures`, and a DFlash2 draft trained for another
/// target is refused with every typed reason.
#[test]
fn architectures_select_dflash2_and_a_draft_for_another_target_is_refused_by_type() {
    assert_eq!(
        BlockDraftFormat::from_config(&dflash2_27b()),
        Some(BlockDraftFormat::DFlash2)
    );
    assert_eq!(
        BlockDraftFormat::from_config(&dspine_b16()),
        Some(BlockDraftFormat::DSpine)
    );
    let cfg = BlockDraftConfig::from_json(&dflash2_27b()).unwrap();
    // Qwen3.6-35B-A3B: 2048 wide, 40 layers, the same vocabulary.
    let other = BlockDraftTarget {
        hidden_size: 2048,
        num_layers: 40,
        ..qwen38_27b()
    };
    let refusals = cfg.mismatches(&other);
    assert_eq!(
        refusals
            .iter()
            .map(BlockDraftRefusal::kind)
            .collect::<Vec<_>>(),
        ["hidden_size", "target_layers"]
    );
    assert_eq!(
        block_draft_refusal_reason(&refusals),
        "block draft: hidden_size: hidden_size 5120 != target 2048; block draft: target_layers: \
         target_layer_ids [5, 19, 33, 47, 61] (built for 64 target layers) do not fit a 40-layer \
         target"
    );
    // A Qwen3 (151k vocabulary) target: the head and the mask token do not fit either.
    let qwen3 = BlockDraftTarget {
        vocab_size: 151_936,
        tokenizer_vocab: 151_669,
        ..qwen38_27b()
    };
    assert_eq!(
        cfg.mismatches(&qwen3)
            .iter()
            .map(BlockDraftRefusal::kind)
            .collect::<Vec<_>>(),
        ["vocabulary", "mask_token"]
    );
}

#[test]
fn a_config_outside_the_dflash2_contract_is_refused_by_name() {
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut raw = dflash2_27b();
        edit(&mut raw);
        let err = BlockDraftConfig::from_json(&raw).unwrap_err();
        assert_eq!(err.kind(), "format");
        err.to_string()
    };
    assert!(
        refused(&|v| v["dflash_config"]["selected_prefix_code_bridge"] =
            serde_json::json!({"code_size": 8}))
        .contains("a head a DFlash2 draft does not have")
    );
    assert!(
        refused(&|v| v["dflash_config"]["conv_group_size"] = 48.into())
            .contains("convolution groups")
    );
    assert!(refused(&|v| {
        v["dflash_config"]
            .as_object_mut()
            .unwrap()
            .remove("selector_rank");
    })
    .contains("`selector_rank`"));
    assert!(refused(&|v| v["is_causal"] = "no".into()).contains("`is_causal`"));
    assert!(
        refused(&|v| v["layer_types"][2] = "full_attention".into()).contains("one sliding window")
    );
    assert!(
        refused(&|v| v["rope_parameters"]["rope_type"] = "yarn".into()).contains("default RoPE")
    );
    assert!(
        refused(&|v| v["dflash_config"]["selector_top_k"] = 300_000.into())
            .contains("exceeds the 248320-token vocabulary")
    );
    // `is_causal` unset follows the layer type (sliding layers are causal), as the reference.
    let mut raw = dflash2_27b();
    raw.as_object_mut().unwrap().remove("is_causal");
    let cfg = BlockDraftConfig::from_json(&raw).unwrap();
    assert!(matches!(
        cfg.head,
        BlockDraftHead::DFlash2 { causal: true, .. }
    ));
    // A DSpine config naming DFlash2's heads stays refused under its own contract.
    let mut dspine = dspine_b16();
    dspine["dflash_config"]["selector_rank"] = 256.into();
    assert!(BlockDraftConfig::from_json(&dspine)
        .unwrap_err()
        .to_string()
        .contains("a head a DSpine draft does not have"));
}

/// The published checkpoint's 81 tensors: 6 top-level, 15 per layer × 5, at the shapes its
/// header declares.
#[test]
fn the_dflash2_tensor_list_is_the_published_checkpoints() {
    let cfg = BlockDraftConfig::from_json(&dflash2_27b()).unwrap();
    let tensors = cfg.tensors();
    assert_eq!(tensors.len(), 6 + 15 * 5);
    let shape_of = |n: &str| tensors.iter().find(|(m, _)| m == n).map(|(_, s)| s.clone());
    for (name, shape) in [
        ("fc.weight", vec![5120, 25600]),
        (
            "candidate_selector.hidden_projection.weight",
            vec![256, 5120],
        ),
        ("candidate_selector.predecessor_codebook", vec![248320, 256]),
        ("candidate_selector.successor_codebook", vec![248320, 256]),
        ("layers.4.attention_conv.base_kernel", vec![2, 2, 5120]),
        (
            "layers.4.attention_conv.kernel_projection.weight",
            vec![1280, 5120],
        ),
        ("layers.0.mlp_conv.base_kernel", vec![2, 2, 5120]),
        (
            "layers.0.mlp_conv.kernel_projection.weight",
            vec![1280, 5120],
        ),
        ("layers.2.self_attn.o_proj.weight", vec![5120, 4096]),
    ] {
        assert_eq!(shape_of(name), Some(shape), "{name}");
    }
    let names: Vec<&str> = tensors.iter().map(|(n, _)| n.as_str()).collect();
    cfg.check_tensors(shape_of, names.iter().copied()).unwrap();
    // A DSpine marker in a DFlash2 checkpoint is an unused tensor.
    let err = cfg
        .check_tensors(shape_of, names.iter().copied().chain(["mask_embedding"]))
        .unwrap_err();
    assert!(err.to_string().contains("unused tensors mask_embedding"));
    let bf16: u64 = tensors
        .iter()
        .map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>() * 2)
        .sum();
    // The published `model.safetensors` is 3,848,817,896 bytes: this payload plus its header.
    assert_eq!(bf16, 3_848_808_960);
}

/// E8: DFlash2 quantizes `fc`, every layer projection, both convolutions' kernel projections and
/// the selector's hidden projection; it loads layer by layer (no checks group), and is priced as
/// its resident arrays plus two of the largest group's windows (a layer's nine BF16 matrices).
#[test]
fn the_dflash2_load_is_priced_by_the_groups_it_materializes() {
    let cfg = BlockDraftConfig::from_json(&dflash2_27b()).unwrap();
    let quantized = cfg.quantized_matrices();
    assert_eq!(quantized.len(), 1 + 9 * 5 + 1);
    assert!(quantized.contains(&"layers.3.mlp_conv.kernel_projection.weight".to_string()));
    assert!(quantized.contains(&"candidate_selector.hidden_projection.weight".to_string()));
    assert!(!quantized
        .iter()
        .any(|q| q.contains("codebook") || q.contains("base_kernel")));
    let groups = cfg.load_groups();
    assert_eq!(groups.len(), cfg.num_hidden_layers + 1);
    let mut read: Vec<&str> = groups
        .iter()
        .flat_map(|g| g.tensors.iter().map(String::as_str))
        .collect();
    read.sort_unstable();
    let mut stored: Vec<String> = cfg.tensors().into_iter().map(|(n, _)| n).collect();
    stored.sort_unstable();
    assert_eq!(read, stored, "every tensor in exactly one group");
    assert!(groups[..cfg.num_hidden_layers]
        .iter()
        .enumerate()
        .all(|(i, g)| g.tensors.len() == 15
            && g.tensors
                .iter()
                .all(|t| t.starts_with(&format!("layers.{i}.")))));
    let largest = groups.iter().map(|g| g.window).max().unwrap();
    assert_eq!(largest, 665_845_760, "a layer's nine BF16 matrices");
    assert_eq!(cfg.resident_bytes(BlockDraftPrecision::Q8), 2_164_124_160);
    assert_eq!(cfg.resident_bytes(BlockDraftPrecision::Q4), 1_265_625_600);
    for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
        assert_eq!(
            cfg.load_bytes(precision),
            cfg.resident_bytes(precision) + 2 * largest
        );
    }
    let at = |prompt: usize| cfg.request_bytes(prompt, 256, 7).unwrap();
    assert!(at(100) < at(1_500));
    assert_eq!(
        at(5_000),
        at(50_000),
        "E5: the 2,048-token window caps every term"
    );
}

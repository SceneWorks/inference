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
    assert_eq!((cfg.code_size, cfg.bridge_rank), (1024, 512));
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

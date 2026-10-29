use super::*;
use serde_json::json;

/// The MLX row of the decode-defaults table, read directly (not through the provider's own
/// reader) so a provider that stops following the table is caught.
const MLX_ROW: &core_llm::DecodeDefaults = core_llm::DecodeBackend::Mlx.defaults();
pub(crate) fn admit_any_transition() -> crate::primitives::DenseTransitionAdmission {
    crate::primitives::DenseTransitionAdmission::new(|_| Ok(()))
}

#[cfg(feature = "campaign")]
#[derive(Default)]
struct RecordingLifecycleObserver(Vec<(&'static str, u64)>);

#[cfg(feature = "campaign")]
impl crate::decode::observe::Observer for RecordingLifecycleObserver {
    fn phase(&mut self, _name: &'static str) {}

    fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}

    fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, _element_bytes: u64) {
        self.0.push(("persistent", bytes));
    }

    fn release_event(&mut self, _kind: &'static str, _role: &'static str, bytes: u64) {
        self.0.push(("released", bytes));
    }
}

#[cfg(feature = "campaign")]
#[test]
fn cancellation_capture_replays_persistent_then_release_without_allocation_aliasing() {
    let mut capture = CacheLifecycleCapture::default();
    crate::decode::observe::Observer::cache_snapshot(&mut capture, 64, 8, 8, 4);
    crate::decode::observe::Observer::release_event(&mut capture, "cache_release", "cache", 64);
    assert!(matches!(
        capture.events.first(),
        Some(CapturedCacheLifecycle::Snapshot(64, 8, 8, 4))
    ));
    let mut observer = RecordingLifecycleObserver::default();
    capture.replay(&mut observer);
    assert_eq!(observer.0, vec![("persistent", 64), ("released", 64)]);
}

#[cfg(feature = "campaign")]
#[test]
fn campaign_cancellation_probe_is_bounded_content_without_tools_or_thinking() {
    let request = campaign_cancellation_probe_request();
    assert_eq!(request.messages.len(), 1);
    assert!(request.tools.is_empty());
    assert!(request.stop.is_empty());
    assert!(request.constraint.is_none());
    assert_eq!(request.thinking, ThinkingMode::Disabled);
    assert_eq!(request.max_new_tokens, 16);
    assert!(request.sampling.is_greedy());
}

#[test]
fn qwen35_dense_checkpoint_selects_one_decoder_root() {
    let select = |keys: &[&str]| qwen35_dense_prefix(|key| keys.contains(&key));
    assert_eq!(
        select(&["model.language_model.embed_tokens.weight"]).unwrap(),
        "model.language_model"
    );
    assert_eq!(
        select(&["model.embed_tokens.weight", "mtp.fc.weight"]).unwrap(),
        "model"
    );
    assert!(select(&[]).is_err());
    assert!(select(&[
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight"
    ])
    .is_err());
}

fn frozen_dense_qwen35_config() -> Qwen35Config {
    Qwen35Config::from_json(&json!({
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 17_408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248_320,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4
    }))
    .unwrap()
}

#[test]
fn fused_request_estimate_tracks_chunked_attention_and_last_row_logits() {
    // Frozen Qwen3.8 parent geometry. This prompt size reproduces the campaign's long-context
    // scale: eager prompt-squared scores dominate hundreds of GB, while MLX's fused full and
    // vector kernels materialize no score matrix (priced conservatively as one 8-query-row score
    // tile per call) and the decoder projects one final row to the vocabulary.
    let geometry = LlmMemoryGeometry {
        query_heads: 40,
        kv_heads: 4,
        head_dim: 128,
        layers: 64,
        element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, false),
        score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
        hidden_size: 5120,
        intermediate_size: 17_408,
        vocab_size: 248_320,
        recurrent_bytes: 0,
    };
    let fused =
        estimate_mlx_request_bytes(29_600, 128, geometry, 0, 0, MlxWorkspaceContract::Chunked)
            .unwrap();
    let eager =
        estimate_mlx_request_bytes(29_600, 128, geometry, 0, 0, MlxWorkspaceContract::Eager)
            .unwrap();
    // sc-24446 prices MLX's in-flight window (eleven command buffers, each up to 50 MB plus
    // the crossing op — here a 2 GB `prompt × inter` MLP activation), so the fused bound grew
    // from ~36 GB; it stays an order of magnitude under the quadratic eager one.
    assert!(
        fused * 8 < eager,
        "bounded MLX peak: {fused} vs eager {eager}"
    );
    assert!(eager > 400_000_000_000, "quadratic eager peak: {eager}");
    assert_eq!(
        eager,
        core_llm::estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap()
            + mlx_runtime_request_bytes(29_600, 29_728, 0, 29_600, geometry, true).unwrap()
            + retired_growth(29_600, 128, geometry),
        "non-fused paths retain the shared fail-closed estimate, plus MLX's runtime terms and \
             the dense growth's retired pre-growth buffers"
    );
}

/// sc-20682: what every dense MLX estimate adds to the shared estimate's unrounded K/V term —
/// the dense cache's block rounding and its window of pre-growth copies.
fn dense_growth(prompt: usize, new_tokens: u32, geometry: LlmMemoryGeometry) -> u64 {
    let total = prompt as u64 + u64::from(new_tokens);
    crate::kv_policy::dense_final_kv_bytes(geometry.kv_shape(), total).unwrap()
        - geometry.kv_shape().dense_bytes(total).unwrap()
}

/// The pre-growth buffers a dense MLX estimate adds beside its runtime block padding: the
/// dense growth past the block-rounded capacity (sc-20682 review).
fn retired_growth(prompt: usize, new_tokens: u32, geometry: LlmMemoryGeometry) -> u64 {
    let total = prompt as u64 + u64::from(new_tokens);
    let capacity = total.div_ceil(256).max(1) * 256;
    crate::kv_policy::dense_final_kv_bytes(geometry.kv_shape(), total).unwrap()
        - geometry.kv_shape().dense_bytes(capacity).unwrap()
}

/// The shared tiled estimate with the attention term `sdpa`'s routing implies for `geometry`.
fn routed_estimate(prompt: usize, new_tokens: u32, geometry: LlmMemoryGeometry) -> u64 {
    let attention = prefill_attention_tile_bytes(
        prompt as u64,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.score_element_bytes,
    )
    .unwrap();
    core_llm::estimate_tiled_request_bytes_with_recurrent_copies(
        prompt, new_tokens, geometry, 0, 0, attention, 1,
    )
    .unwrap()
}

/// sc-20676: the chunked contract prices the tile `sdpa` actually runs — a `rows × k_len` mask
/// slice for a full-kernel head dim, a per-head score set over `rows · gqa ≤ 32` rows otherwise
/// (Phi-3's head dim 96 used to be priced as 8-row tiles while running one unfused call).
#[test]
fn chunked_contract_prices_the_routed_attention_tile() {
    let with_head_dim = |head_dim, query_heads, kv_heads| LlmMemoryGeometry {
        query_heads,
        kv_heads,
        head_dim,
        layers: 32,
        element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, false),
        score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
        hidden_size: 3072,
        intermediate_size: 8192,
        vocab_size: 32_064,
        recurrent_bytes: 0,
    };
    let prompt = 16_384usize;
    for (head_dim, query_heads, kv_heads) in [(128, 24, 8), (96, 32, 32), (256, 16, 2)] {
        let geometry = with_head_dim(head_dim, query_heads, kv_heads);
        let estimate =
            estimate_mlx_request_bytes(prompt, 16, geometry, 0, 0, MlxWorkspaceContract::Chunked)
                .unwrap();
        assert_eq!(
            estimate,
            routed_estimate(prompt, 16, geometry)
                + mlx_runtime_request_bytes(
                    prompt,
                    prompt as u64 + 16,
                    0,
                    (prompt as u64).min(SDPA_MAX_FUSED_QLEN as u64),
                    geometry,
                    true,
                )
                .unwrap()
                + retired_growth(prompt, 16, geometry),
            "head_dim {head_dim}"
        );
    }
    // The full-kernel block's mask slice (16k × 2048 × 4 B) outweighs the old 8-row tile.
    let block = routed_estimate(prompt, 16, with_head_dim(128, 24, 8));
    let old_tile =
        core_llm::estimate_chunked_request_bytes(prompt, 16, with_head_dim(128, 24, 8), 0, 0, 8)
            .unwrap();
    assert!(block > old_tile);
}

#[test]
fn fused_request_estimate_is_checked_and_preserves_mtp_and_media_costs() {
    let geometry = LlmMemoryGeometry {
        query_heads: 8,
        kv_heads: 2,
        head_dim: 64,
        layers: 4,
        element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, false),
        score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
        hidden_size: 256,
        intermediate_size: 512,
        vocab_size: 1024,
        recurrent_bytes: 4096,
    };
    let plain =
        estimate_mlx_request_bytes(128, 16, geometry, 0, 0, MlxWorkspaceContract::Chunked).unwrap();
    let media =
        estimate_mlx_request_bytes(128, 16, geometry, 123_456, 0, MlxWorkspaceContract::Chunked)
            .unwrap();
    let mtp =
        estimate_mlx_request_bytes(128, 16, geometry, 0, 3, MlxWorkspaceContract::Chunked).unwrap();
    assert_eq!(media - plain, 123_456);
    assert!(mtp > plain);
    assert!(estimate_mlx_request_bytes(
        usize::MAX,
        u32::MAX,
        geometry,
        0,
        3,
        MlxWorkspaceContract::Chunked,
    )
    .is_none());
}

#[test]
fn qwen35_prism_estimate_covers_frozen_allocator_peaks_and_rejects_remote_long_context() {
    let config = frozen_dense_qwen35_config();
    let geometry = LlmMemoryGeometry {
        query_heads: 24,
        kv_heads: 4,
        head_dim: 256,
        layers: 64,
        element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, true),
        score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
        hidden_size: 5120,
        intermediate_size: 17_408,
        vocab_size: 248_320,
        recurrent_bytes: 207_618_048,
    };
    let contract = MlxWorkspaceContract::Qwen35 {
        config: &config,
        prism: true,
    };
    let arithmetic = estimate_mlx_request_bytes(27, 128, geometry, 0, 0, contract).unwrap();
    let context_64 = estimate_mlx_request_bytes(1_187, 128, geometry, 0, 0, contract).unwrap();
    let context_512 = estimate_mlx_request_bytes(9_251, 128, geometry, 0, 0, contract).unwrap();
    let context_2048 = estimate_mlx_request_bytes(36_899, 128, geometry, 0, 0, contract).unwrap();

    // Frozen local probe at 02ed7e958: arithmetic raised the process-global allocator peak to
    // 9,321,652,712 from a captured post-load active 8,602,141,760 (a 719,510,952-byte lower
    // bound). Subtracting the stable post-request residency 8,548,422,952 gives 773,229,760;
    // price that stricter observed bound even though it is not an immediately-pre-request
    // sample. Frozen campaign 35461924246 supplies the two longer-context deltas below.
    assert!(arithmetic >= 773_229_760, "estimate: {arithmetic}");
    assert!(context_64 >= 4_091_010_468, "estimate: {context_64}");
    assert!(context_512 >= 22_885_373_536, "estimate: {context_512}");
    // The Prism hold (see `priced_compute_element_bytes`) exceeds the compute-width estimate
    // by exactly the K/V, activation and last-row logit terms at the width difference; every
    // other term is width-independent.
    let compute_width = LlmMemoryGeometry {
        element_bytes: dtype_bytes(Dtype::Bfloat16),
        score_element_bytes: 4,
        ..geometry
    };
    let delta = geometry.element_bytes - compute_width.element_bytes;
    for (prompt, estimate) in [
        (27_u64, arithmetic),
        (1_187, context_64),
        (9_251, context_512),
        (36_899, context_2048),
    ] {
        let g = geometry;
        let kv = (prompt + 128) * g.layers * g.kv_heads * g.head_dim * delta * 2;
        let activations =
            prompt * (g.intermediate_size * 3 + g.hidden_size * 8) * delta + g.vocab_size * delta;
        let at_compute_width =
            estimate_mlx_request_bytes(prompt as usize, 128, compute_width, 0, 0, contract)
                .unwrap();
        let growth = dense_growth(prompt as usize, 128, g)
            - dense_growth(prompt as usize, 128, compute_width);
        assert_eq!(
            estimate - at_compute_width,
            kv + activations + growth,
            "prompt {prompt}"
        );
    }
    assert!(arithmetic < context_64 && context_64 < context_512 && context_512 < context_2048);
    assert!(
        core_llm::admit_request_memory(context_2048, 47_922_610_176).is_err(),
        "the 36,899-token request must not be admitted on the failed remote's observed budget"
    );
}

#[test]
fn qwen35_workspace_estimate_is_checked_and_keeps_ordinary_requests_available() {
    let config = frozen_dense_qwen35_config();
    let geometry = LlmMemoryGeometry {
        query_heads: 24,
        kv_heads: 4,
        head_dim: 256,
        layers: 64,
        element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, true),
        score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
        hidden_size: 5120,
        intermediate_size: 17_408,
        vocab_size: 248_320,
        recurrent_bytes: 207_618_048,
    };
    let contract = MlxWorkspaceContract::Qwen35 {
        config: &config,
        prism: true,
    };
    // A one-token request exercises the pre-evaluation recurrence branch without applying a
    // 256-step floor. A normal 128-token prompt remains admitted on an 8-GB machine.
    let scalar = estimate_mlx_request_bytes(1, 1, geometry, 0, 0, contract).unwrap();
    let ordinary = estimate_mlx_request_bytes(128, 128, geometry, 0, 0, contract).unwrap();
    for (prompt, new_tokens, estimate) in [(1, 1, scalar), (128, 128, ordinary)] {
        assert_eq!(
            estimate,
            routed_estimate(prompt, new_tokens, geometry)
                + estimate_qwen35_workspace_extra_bytes(prompt, &config, true).unwrap()
                + dense_growth(prompt, new_tokens, geometry)
                + MLX_REQUEST_WAKE_BYTES,
            "prompt {prompt}"
        );
    }
    assert!(scalar < ordinary);
    assert!(core_llm::admit_request_memory(ordinary, 8_000_000_000).is_ok());
    assert!(
        estimate_mlx_request_bytes(usize::MAX, u32::MAX, geometry, u64::MAX, 3, contract,)
            .is_none()
    );
}

#[test]
fn reasoning_boundary_commits_same_token_json_in_release_builds() {
    let table = core_llm::ConstraintDecodeTable {
        pieces: vec!["</think>{}".into(), String::new(), "{".into()],
        special: [1].into_iter().collect(),
    };
    let mut mask = JsonMask::new(&table, [1], true);
    assert!(mask.allowed()[0]);
    mask.accept(0);
    assert!(
        mask.allowed()[1],
        "same-token answer must commit even with debug assertions off"
    );
    assert!(
        !mask.allowed()[2],
        "completed answer cannot begin a second JSON value"
    );
    let mark = mask.checkpoint();
    mask.rewind(mark);
    assert!(
        mask.allowed()[1],
        "MTP rewind must replay phase and same-token JSON"
    );
}

#[test]
fn json_constraint_waits_for_split_reasoning_close_and_rewinds_phase() {
    let table = ConstraintDecodeTable {
        pieces: vec![
            "reason".into(),
            "</thi".into(),
            "nk>".into(),
            "{".into(),
            "x".into(),
            "</think>x".into(),
            String::new(),
        ],
        special: [6].into_iter().collect(),
    };
    let mut mask = JsonMask::new(&table, [6], true);
    assert!(mask.allowed()[0]);
    assert!(
        !mask.allowed()[5],
        "invalid same-token JSON suffix is masked"
    );
    assert!(!mask.allowed()[6], "EOS cannot end an open reasoning block");
    mask.accept(0);
    let checkpoint = mask.checkpoint();
    mask.accept(1);
    mask.accept(2);
    assert!(mask.allowed()[3], "JSON begins only after split </think>");
    assert!(!mask.allowed()[4]);
    mask.rewind(checkpoint);
    assert!(mask.allowed()[1], "rewind restores the reasoning phase");
    assert!(!mask.allowed()[6]);

    let mut disabled = JsonMask::new(&table, [6], false);
    assert!(disabled.allowed()[3], "disabled thinking starts in JSON");
    assert!(!disabled.allowed()[0]);
}

#[test]
fn reasoning_controls_are_advertised_only_when_template_names_them() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        "{{ enable_thinking }} {{ tool_call }}",
    )
    .unwrap();
    let (_, thinking, effort, preserve, tools, _) = load_chat_template(dir.path());
    assert!(thinking);
    assert!(!effort);
    assert!(!preserve);
    assert!(tools);

    std::fs::write(
        dir.path().join("chat_template.jinja"),
        "{{ enable_thinking }} {{ reasoning_effort }} {{ preserve_thinking }}",
    )
    .unwrap();
    let (_, thinking, effort, preserve, tools, _) = load_chat_template(dir.path());
    assert!(thinking);
    assert!(effort);
    assert!(preserve);
    assert!(!tools);
}

#[test]
fn frozen_qwen38_generation_config_uses_both_official_stop_tokens() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/generation_config.json"
        )),
    )
    .unwrap();

    assert_eq!(eos_token_ids(dir.path()), vec![248046, 248044]);
}

fn qwen36_wrapper() -> serde_json::Value {
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": { "model_type": "qwen3_5_text" },
        "vision_config": { "model_type": "qwen3_5", "depth": 27 }
    })
}
fn llava_wrapper() -> serde_json::Value {
    json!({
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        "text_config": { "model_type": "llama" },
        "vision_config": { "model_type": "siglip_vision_model" }
    })
}
fn qwen3vl_wrapper() -> serde_json::Value {
    json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": { "model_type": "qwen3_vl_text" },
        "vision_config": { "model_type": "qwen3_vl", "depth": 27 }
    })
}

#[test]
fn text_provider_claims_qwen36_but_cedes_llava() {
    // Qwen3.6 (Qwen-VL wrapper): claimed by the text provider, served text-only.
    assert!(can_load_value(&qwen36_wrapper()));
    // Qwen3-VL (Qwen-VL wrapper): also claimed by `mlx-llama` (its nested standard Qwen3 decoder).
    assert!(can_load_value(&qwen3vl_wrapper()));
    // LLaVA: ceded to the JoyCaption vision provider.
    assert!(!can_load_value(&llava_wrapper()));
    // Plain text models (no vision_config) are unaffected.
    assert!(can_load_value(
        &json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })
    ));
    assert!(can_load_value(
        &json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })
    ));
}

#[test]
fn tool_format_distinguishes_tagged_and_llama_bare_json() {
    assert_eq!(
        tool_call_format("emit <tool_call>...</tool_call>"),
        Some(ToolCallFormat::Tagged)
    );
    assert_eq!(
        tool_call_format(
            "message.tool_calls; please respond with JSON for a function call exactly"
        ),
        Some(ToolCallFormat::BareJson)
    );
    assert_eq!(tool_call_format("message.tool_calls only"), None);

    let tools = [core_llm::ToolSpec::new(
        "record_baseline_fact",
        "Record a fact",
        json!({
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"]
        }),
    )];
    let calls = bare_json_tool_calls(
        r#"{"name":"record_baseline_fact","parameters":{"fact":"SC20671 structured fixture"}}"#,
        &tools,
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "record_baseline_fact");
    assert_eq!(
        calls[0].arguments.get("fact"),
        Some(&json!("SC20671 structured fixture"))
    );
    assert!(bare_json_tool_calls("ordinary answer", &tools).is_empty());
}

#[test]
fn weightless_vision_advertises_qwen_vl_wrappers_only() {
    // The sc-8077 weightless vision gate: `mlx-llama` advertises vision (pre-load, config.json
    // only) for a Qwen-VL wrapper so a model-first vision-required load resolves here.
    // Qwen3-VL: a vision_config + qwen3_vl arch ⇒ vision-capable.
    assert!(weightless_vision_value(&qwen3vl_wrapper()));
    // Qwen3.6 hybrid (qwen3_5) Qwen-VL wrapper: likewise vision-capable.
    assert!(weightless_vision_value(&qwen36_wrapper()));
    // A LLaVA snapshot is ceded to JoyCaption (can_load=false here) ⇒ NOT advertised, so a
    // Qwen3-VL load can never be mistaken for / misrouted via this provider's vision path.
    assert!(!weightless_vision_value(&llava_wrapper()));
    // Plain text checkpoints (no vision_config) ⇒ NOT vision-capable (the static descriptor
    // already reports supports_vision=false; the probe must not flip it on).
    assert!(!weightless_vision_value(
        &json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })
    ));
    assert!(!weightless_vision_value(
        &json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })
    ));
}

/// Locate a staged Qwen3-VL-8B-Instruct snapshot for the chat-template oracle from the explicit
/// passed-in `QWEN3VL_SNAPSHOT` env path. Inference never self-fetches or derives a cache location
/// (epic 13657). `None` ⇒ the gated tests self-skip cleanly (CI sets the var for this story).
fn qwen3vl_snapshot_dir() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("QWEN3VL_SNAPSHOT").ok()?);
    path.exists().then_some(path)
}

/// Build the representative chat-message sets the oracle pins (text-only, system+user, single
/// image, multi-image/mixed, multi-turn). Image content is a `Content::Image` (a 1×1 black pixel
/// placeholder); the byte-match exercises the same `substitute_image_placeholders` path the
/// provider uses, so the rendered single `<|image_pad|>` per image must match HF.
fn oracle_messages(case: &str) -> Vec<Message> {
    let img = || Content::Image(ImageRef::new(1, 1, vec![0, 0, 0]).unwrap());
    let user = |content: Vec<Content>| Message {
        role: core_llm::Role::User,
        content,
        thinking: None,
        tool_calls: Vec::new(),
    };
    let sys = |t: &str| Message {
        role: core_llm::Role::System,
        content: vec![Content::text(t)],
        thinking: None,
        tool_calls: Vec::new(),
    };
    let asst = |t: &str| Message {
        role: core_llm::Role::Assistant,
        content: vec![Content::text(t)],
        thinking: None,
        tool_calls: Vec::new(),
    };
    match case {
        "text_only" => vec![user(vec![Content::text("What is the capital of France?")])],
        "system_user_text" => {
            vec![
                sys("You are a helpful assistant."),
                user(vec![Content::text("Hello!")]),
            ]
        }
        "single_image" => vec![user(vec![img(), Content::text("Describe this image.")])],
        "multi_image_mixed" => vec![user(vec![
            Content::text("Compare:"),
            img(),
            Content::text("and"),
            img(),
            Content::text("please."),
        ])],
        "multi_turn" => vec![
            user(vec![Content::text("Hi")]),
            asst("Hello there!"),
            user(vec![Content::text("How are you?")]),
        ],
        other => panic!("unknown oracle case {other}"),
    }
}

/// Byte-match oracle (sc-8075 AC #1): the engine's chat-template + image-placeholder + tokenize
/// path must reproduce the pinned HF `apply_chat_template` prompt token ids **exactly** for the
/// representative messages — text-only, single-image, and multi-image/mixed. The single
/// `<|image_pad|>` (151655) per image is what the processor emits *before* patch-count expansion;
/// these fixtures pin that pre-expansion prompt. Self-skips cleanly when the snapshot is absent.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn chat_template_byte_matches_hf_processor() {
    let Some(dir) = qwen3vl_snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        crate::model_inference_device::record_model_inference_prerequisite_missing("required work unavailable in chat_template_byte_matches_hf_processor; see preceding skip diagnostic");
        return;
    };
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../models/testdata/qwen3vl_chat_template_oracle.json"
    ))
    .expect("parse chat-template oracle");

    // Sanity: the dispatch and config see Qwen3-VL (and parse the nested 256K-context text decoder).
    let cfg_value = read_config_value(&dir).expect("read config.json");
    assert_eq!(
        Architecture::from_config(&cfg_value).unwrap(),
        Architecture::Qwen3Vl,
        "snapshot must dispatch to Qwen3-VL"
    );
    let cfg = ModelConfig::from_json(&cfg_value).expect("parse Qwen3-VL text config");
    assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
    assert_eq!(cfg.max_position_embeddings, 262144, "256K context");

    let (template, _, _, _, _, _) = load_chat_template(&dir);
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

    for (case, expected) in oracle["cases"].as_object().unwrap() {
        let want: Vec<i32> = expected["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        // Drive the exact provider path: substitute image content with the Qwen-VL placeholder,
        // render the model's own Jinja template (add_generation_prompt), then tokenize without
        // auto special tokens.
        let messages = oracle_messages(case);
        let substituted = substitute_vision_placeholders(&messages, 2)
            .expect("the oracle cases carry no audio, so substitution cannot fail");
        let prompt = template
            .render_with(
                &substituted,
                &RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: None,
                    reasoning_effort: None,
                    preserve_thinking: None,
                    tools: &[],
                    date_string: None,
                },
            )
            .expect("render");
        assert_eq!(
            prompt,
            expected["text"].as_str().unwrap(),
            "case {case}: rendered prompt string must byte-match HF"
        );
        let got: Vec<i32> = tokenizer
            .encode(&prompt, false)
            .expect("encode")
            .into_iter()
            .map(|id| id as i32)
            .collect();
        assert_eq!(
            got, want,
            "case {case}: tokenized prompt ids must byte-match HF processor"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED provider::tests::chat_template_byte_matches_hf_processor"
    );
}

#[test]
fn prompt_opens_thinking_matches_template_modes() {
    // Qwen3.6 thinking/auto generation prompt: opens the block, leaves it unclosed.
    assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
    // Disabled mode renders a *closed* empty block: must not prime.
    assert!(!prompt_opens_thinking(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    // A prior closed reasoning turn followed by a fresh open block still opens.
    assert!(prompt_opens_thinking(
        "<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"
    ));
    // No reasoning markers at all (non-thinking template).
    assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
}

// --- Qwen3-VL video Text–Timestamp-Alignment oracle (tools/gen_qwen3vl_video_oracle.py) --------

fn qwen3vl_video_oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("../models/testdata/qwen3vl_video_oracle.json")).unwrap()
}

/// **Merged per-frame timestamps match `Qwen3VLProcessor._calculate_timestamps`.** Given the
/// sampled `frames_indices` + `fps`, the per-temporal-patch averaged timestamps must equal the HF
/// reference exactly — the values that feed the `<{t:.1f} seconds>` Text–Timestamp-Alignment tags.
#[test]
fn video_merged_timestamps_match_hf_reference() {
    let j = qwen3vl_video_oracle();
    let fps = j["fps"].as_f64().unwrap() as f32;
    let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
    let indices: Vec<f32> = j["frames_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    // Per-sample timestamps are `idx / fps` (matching the reference, which averages these).
    let per_sample: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
    let got = merged_frame_timestamps(&per_sample, temporal);
    let want: Vec<f32> = j["merged_timestamps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(got.len(), want.len(), "merged timestamp count vs HF");
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-5, "merged timestamp {g} vs HF {w}");
    }
}

#[test]
fn invalid_public_video_timestamps_fail_before_prompt_rendering() {
    let frame = || ImageRef::new(1, 1, vec![0, 0, 0]).unwrap();
    for timestamps in [vec![0.0, f32::NAN], vec![0.5, 0.25], vec![-0.1, 0.0]] {
        let video = VideoRef {
            frames: vec![frame(), frame()],
            timestamps,
        };
        let messages = vec![Message {
            role: core_llm::Role::User,
            content: vec![Content::Video(video)],
            thinking: None,
            tool_calls: Vec::new(),
        }];
        let error = substitute_vision_placeholders(&messages, 2)
            .expect_err("invalid timestamp sequence accepted");
        assert!(error.to_string().contains("timestamp"));
    }
    let count_mismatch = vec![Message {
        role: core_llm::Role::User,
        content: vec![Content::Video(VideoRef {
            frames: vec![frame(), frame()],
            timestamps: vec![0.0],
        })],
        thinking: None,
        tool_calls: Vec::new(),
    }];
    assert!(substitute_vision_placeholders(&count_mismatch, 2)
        .unwrap_err()
        .to_string()
        .contains("one timestamp per frame"));
}

#[test]
fn expanded_text_image_video_and_mtp_budgets_share_the_context_gate() {
    for expanded_prompt in [48usize, 52, 60] {
        validate_context_window(64, expanded_prompt, (64 - expanded_prompt) as u32).unwrap();
        let error = validate_context_window(64, expanded_prompt, (64 - expanded_prompt + 1) as u32)
            .expect_err("one-token context overflow accepted");
        assert!(error.to_string().contains("exceeds context window 64"));
    }
    assert!(validate_context_window(usize::MAX, usize::MAX, 1)
        .unwrap_err()
        .to_string()
        .contains("overflow"));
}

/// **The per-frame placeholder string matches `Qwen3VLProcessor.replace_video_token` (collapsed
/// form).** The engine emits **one** `<|video_pad|>` per frame and expands it to `frame_seqlen`
/// copies after tokenizing (exactly the image path's pattern, where the chat template renders a
/// single `<|image_pad|>`). The reference `replace_video_token` writes the *already-expanded*
/// string (`frame_seqlen` `<|video_pad|>` per frame). Collapsing each consecutive `<|video_pad|>`
/// run of the reference string to one token must yield exactly [`video_placeholder_text`]: same
/// `<{t:.1f} seconds>` Text–Timestamp-Alignment tags, same per-frame vision framing.
#[test]
fn video_placeholder_string_matches_hf_reference() {
    let j = qwen3vl_video_oracle();
    let fps = j["fps"].as_f64().unwrap() as f32;
    let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
    let indices: Vec<f32> = j["frames_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    // Build a synthetic VideoRef with one 1x1 frame per sampled index carrying its `idx/fps`
    // timestamp (the frame pixels are irrelevant to the placeholder string).
    let frames: Vec<ImageRef> = indices
        .iter()
        .map(|_| ImageRef::new(1, 1, vec![0, 0, 0]).unwrap())
        .collect();
    let timestamps: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
    let video = VideoRef::new(frames, timestamps).unwrap();
    let got = video_placeholder_text(&video, temporal);

    // Collapse the reference string's `<|video_pad|>` runs to a single token per frame.
    let pad = "<|video_pad|>";
    let mut collapsed = j["placeholder_text"].as_str().unwrap().to_string();
    while collapsed.contains(&format!("{pad}{pad}")) {
        collapsed = collapsed.replace(&format!("{pad}{pad}"), pad);
    }
    assert_eq!(
            got, collapsed,
            "collapsed Text–Timestamp-Alignment placeholder string must byte-match HF replace_video_token"
        );
    // The timestamp tags themselves must appear verbatim (the core of Text–Timestamp Alignment).
    for t in j["merged_timestamps"].as_array().unwrap() {
        let tag = format!("<{:.1} seconds>", t.as_f64().unwrap());
        assert!(
            got.contains(&tag),
            "placeholder must carry the `{tag}` timestamp tag: {got}"
        );
    }
}

/// **The per-frame `<|video_pad|>` expansion matches the HF id stream.** Tokenizing the reference
/// placeholder string yields one `<|video_pad|>` per frame; expanding each to `frame_seqlen`
/// copies (the merged patch count the ViT emits per frame) must reproduce the exact id stream the
/// processor produces — same per-frame vision framing, same timestamp tokens, same counts. This is
/// the video analogue of `expand_vision_placeholders` for images, but with `grid_t` runs.
#[test]
fn video_token_expansion_matches_hf_reference() {
    let j = qwen3vl_video_oracle();
    let expanded_hf: Vec<i32> = j["expanded_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let vid = j["video_token_id"].as_i64().unwrap() as i32;
    let grid_t = j["grid_t"].as_u64().unwrap() as usize;
    let frame_seqlen = j["frame_seqlen"].as_u64().unwrap() as usize;
    let expected_video_tokens = j["expected_video_tokens"].as_u64().unwrap() as usize;

    // The merged-token count per frame, and total, must agree with the HF processor.
    assert_eq!(
        grid_t * frame_seqlen,
        expected_video_tokens,
        "total video tokens vs HF"
    );
    assert_eq!(
        expanded_hf.iter().filter(|&&x| x == vid).count(),
        expected_video_tokens,
        "video tokens in HF id stream"
    );

    // Reconstruct the *raw* (pre-expansion) ids: collapse each consecutive `<|video_pad|>` run
    // back to a single placeholder. The HF stream has `grid_t` such runs (one per frame), each of
    // `frame_seqlen` tokens; collapsing recovers one `<|video_pad|>` per frame.
    let mut raw = Vec::new();
    let mut i = 0usize;
    let mut runs = 0usize;
    while i < expanded_hf.len() {
        if expanded_hf[i] == vid {
            raw.push(vid);
            runs += 1;
            while i < expanded_hf.len() && expanded_hf[i] == vid {
                i += 1;
            }
        } else {
            raw.push(expanded_hf[i]);
            i += 1;
        }
    }
    assert_eq!(runs, grid_t, "one <|video_pad|> run per frame (grid_t)");

    // Expanding each per-frame placeholder to `frame_seqlen` reproduces the HF id stream exactly.
    let counts = vec![frame_seqlen; grid_t];
    let expanded = crate::models::qwen35::expand_vision_placeholders(&raw, vid, &counts).unwrap();
    assert_eq!(expanded, expanded_hf, "expanded video ids vs HF processor");
}

/// **The video M-RoPE positions over the oracle grid are well-formed and per-frame-reset.** Feed
/// the expanded video id stream + the `video_grid_thw` through `mrope_positions_mm`: the temporal
/// row must reset to the frame's cursor at each frame (Qwen3-VL's synthetic time axis splits each
/// `[t,h,w]` into `t` per-frame `[1,h,w]` blocks). The HF-pinned exact-row check lives in
/// `qwen35::qwen3vl_mrope_video_matches_hf_reference`; here we confirm the provider's video grid
/// drives the same path consistently.
#[test]
fn video_mrope_positions_split_frames() {
    let j = qwen3vl_video_oracle();
    let expanded_hf: Vec<i32> = j["expanded_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let vid = j["video_token_id"].as_i64().unwrap() as i32;
    let img = j["video_token_id"].as_i64().unwrap() as i32 - 1; // a distinct unused image id
    let g = j["video_grid_thw"].as_array().unwrap()[0]
        .as_array()
        .unwrap();
    let grid = [
        g[0].as_i64().unwrap() as i32,
        g[1].as_i64().unwrap() as i32,
        g[2].as_i64().unwrap() as i32,
    ];
    let merge = j["merge"].as_i64().unwrap() as i32;

    let (t, h, w, _delta) =
        crate::models::deepstack::mrope_positions_mm(&expanded_hf, &[], img, &[grid], vid, merge)
            .unwrap();
    assert_eq!(t.len(), expanded_hf.len());
    // Each frame's video tokens share one temporal index (gt = 1 per frame after the split), and
    // the two frames sit at *different* temporal positions (the cursor advances between them).
    let frame_temporals: Vec<i32> = expanded_hf
        .iter()
        .zip(&t)
        .filter_map(|(&id, &tt)| (id == vid).then_some(tt))
        .collect();
    let distinct: std::collections::BTreeSet<i32> = frame_temporals.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        grid[0] as usize,
        "one distinct temporal index per frame"
    );
    // h/w spans are bounded by the per-frame grid (h/merge, w/merge).
    let max_w = (grid[2] / merge) - 1;
    let frame_ws: Vec<i32> = expanded_hf
        .iter()
        .zip(&w)
        .zip(&t)
        .filter_map(|((&id, &ww), &tt)| (id == vid).then_some(ww - tt))
        .collect();
    assert!(
        frame_ws.iter().all(|&rel| (0..=max_w).contains(&rel)),
        "w within per-frame grid"
    );
    let _ = h;
}

// ---- The speculative engine behind the provider (epic sc-24432, story sc-24434). ----

/// A whitespace WordLevel tokenizer over `t0..t{vocab-1}`, so every id decodes to a distinct
/// piece and the streamed deltas are the generated ids one-for-one.
fn word_tokenizer(vocab: usize) -> Tokenizer {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
    Tokenizer::from_json(&format!(
        r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
        entries.join(", ")
    ))
    .unwrap()
}

fn causal_provider() -> LlamaProvider {
    LlamaProvider::from_parts(
        crate::decode::engine::tests::causal(),
        word_tokenizer(24),
        vec![],
    )
}

fn qwen35_mtp_provider() -> LlamaProvider {
    let model = crate::decode::engine::tests::qwen35(true);
    let descriptor = descriptor_for_qwen35(model.config());
    LlamaProvider {
        descriptor,
        model: Decoder::Qwen35(Box::new(model)),
        architecture: Architecture::Qwen35,
        campaign_family: None,
        kv_family: None,
        tokenizer: word_tokenizer(50),
        template: Box::new(Llama3Template),
        tool_call_format: None,
        stop_tokens: Vec::new(),
        constraint_table: OnceCell::new(),
        vision: None,
        gemma4: None,
        campaign_prefix_cache: RefCell::new(None),
        execution_fence: LlamaProvider::new_execution_fence(),
        campaign_template_date: None,
        kv_reader: OnceCell::new(),
        kv_model_name: String::new(),
        _prism_vision_weights: None,
        prefix: RefCell::new(PrefixCache::with_budget(
            core_llm::defaults::MLX.prefix_cache_bytes,
        )),
        speculative_default: core_llm::defaults::MLX.speculative,
        draft: None,
        load_report: core_llm::LoadReport::default(),
    }
}

/// The MoE hybrid (`qwen3_5_moe`), which admission prices on the eager contract.
fn qwen35_moe_provider() -> LlamaProvider {
    use crate::models::qwen35::tests::{cfg_json_moe, synthetic_weights};
    let cfg = Qwen35Config::from_json(&cfg_json_moe()).unwrap();
    let model =
        Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap();
    LlamaProvider {
        descriptor: descriptor_for_qwen35(model.config()),
        model: Decoder::Qwen35(Box::new(model)),
        architecture: Architecture::Qwen35,
        campaign_family: None,
        kv_family: None,
        tokenizer: word_tokenizer(50),
        template: Box::new(Llama3Template),
        tool_call_format: None,
        stop_tokens: Vec::new(),
        constraint_table: OnceCell::new(),
        vision: None,
        gemma4: None,
        campaign_prefix_cache: RefCell::new(None),
        execution_fence: LlamaProvider::new_execution_fence(),
        campaign_template_date: None,
        kv_reader: OnceCell::new(),
        kv_model_name: String::new(),
        _prism_vision_weights: None,
        prefix: RefCell::new(PrefixCache::with_budget(
            core_llm::defaults::MLX.prefix_cache_bytes,
        )),
        speculative_default: core_llm::defaults::MLX.speculative,
        draft: None,
        load_report: core_llm::LoadReport::default(),
    }
}

fn spec_request(speculative: core_llm::Speculative) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user("t3 t9 t4 t11 t3 t9 t4 t11 t3 t9 t4 t11")],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(3),
        speculative: Some(speculative),
        ..Default::default()
    }
}

/// Generate, returning the output and the token ids the stream carried.
fn run(provider: &LlamaProvider, req: &TextLlmRequest) -> (TextLlmOutput, Vec<u32>) {
    let mut ids = Vec::new();
    let out = provider
        .generate(req, &mut |ev| {
            if let CoreEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .unwrap();
    (out, ids)
}

/// The plain, non-engine token-at-a-time loop over the provider's own decoder and its own
/// rendering of `req` — the reference `off` is held to.
fn plain_loop(provider: &LlamaProvider, req: &TextLlmRequest) -> Vec<u32> {
    let prompt = provider
        .template
        .render_with(
            &req.messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
                tools: &req.tools,
                date_string: provider.campaign_template_date,
            },
        )
        .unwrap();
    let ids: Vec<i32> = provider
        .tokenizer
        .encode(&prompt, false)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect();
    let config = GenerationConfig {
        max_new_tokens: req.max_new_tokens as usize,
        sampling: SamplingParams::from(&req.sampling),
        seed: req.seed,
        stop_tokens: provider.stop_tokens.clone(),
    };
    crate::decode::generate(
        &provider.model,
        &ids,
        &config,
        &crate::decode::CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens
    .into_iter()
    .map(|t| t as u32)
    .collect()
}

/// AC1 + AC2: `{proposer: prompt_lookup}` on the Causal decoder and `{proposer: mtp}` on the
/// Qwen35 MTP decoder emit exactly the greedy tokens of `off`; `off` itself is the plain,
/// non-engine loop over the same decoder and prompt; and every generation carries a populated
/// `DecodeReport` naming the proposer, its depth, the verify steps behind the mean accepted
/// length and the sampler path.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn speculative_requests_emit_the_off_tokens_and_off_is_the_plain_loop() {
    use core_llm::Speculative;
    // The absolute accounting of a budget-bound run without stop tokens: the first token
    // comes from the prefill and each verify step commits its accepted drafts plus one; every
    // forward is the prefill forward (one, also when the hybrid's prefix cache snapshots its
    // conversation boundary inside it, sc-24446), a verify step or a recovery replay.
    let accounting = |label: &str, ids: &[u32], report: &core_llm::DecodeReport| {
        assert_eq!(
            ids.len() as u64,
            1 + report.verify_steps + report.accepted_tokens,
            "{label}: {report:?}"
        );
        assert!(
            matches!(report.prefill_forwards, 1 | 2),
            "{label}: {report:?}"
        );
        assert_eq!(
            report.target_forwards,
            report.prefill_forwards
                + report.verify_steps
                + report.replay_forwards
                + report.discarded_forwards,
            "{label}: {report:?}"
        );
    };
    for (label, provider, proposer, depth) in [
        (
            "causal prompt lookup",
            causal_provider(),
            SpeculativeProposer::PromptLookup,
            4,
        ),
        (
            "qwen35 prompt lookup",
            qwen35_mtp_provider(),
            SpeculativeProposer::PromptLookup,
            4,
        ),
        (
            "qwen35 mtp",
            qwen35_mtp_provider(),
            SpeculativeProposer::Mtp,
            3,
        ),
    ] {
        let off_req = spec_request(Speculative::Off);
        let (off, off_ids) = run(&provider, &off_req);
        let plain = plain_loop(&provider, &off_req);
        assert_eq!(off_ids, plain, "{label}: off != the plain loop");
        assert_eq!(off.text, provider.tokenizer.decode(&plain, true).unwrap());
        assert_eq!(off_ids.len(), 16, "{label}: runs to the budget");
        assert!(
            provider.stop_tokens.is_empty(),
            "{label}: accounting needs no stop tokens"
        );
        let report = off.decode.expect("off reports its decode path");
        accounting(label, &off_ids, &report);
        assert_eq!(report.path, "step_model", "{label}");
        assert_eq!(report.proposer, ProposerKind::None, "{label}");
        assert_eq!(report.draft_tokens, None, "{label}");
        assert_eq!(report.verify_steps, 15, "{label}");
        assert_eq!(report.sampler, "device", "{label}");
        // E3: no graph runner on MLX — `none`, never `eager`; and the fused primitives that
        // actually ran: the hybrid's Gated DeltaNet recurrence on the GPU stream is the fused
        // Metal kernel, the causal fixture has no fused primitive.
        assert_eq!(report.graph_path, "none", "{label}");
        let fused = if matches!(provider.model, Decoder::Qwen35(_)) {
            "fused"
        } else {
            "none"
        };
        assert_eq!(report.fused_primitives.path, fused, "{label}");
        assert_eq!(report.fused_primitives.reason, None, "{label}");
        assert!(
            report.fallbacks.is_empty(),
            "{label}: {:?}",
            report.fallbacks
        );
        assert!(off.mtp.is_none(), "{label}");
        assert!(off.timings.is_some(), "{label}: text decode is timed");

        let (spec, spec_ids) = run(
            &provider,
            &spec_request(Speculative::proposer(proposer, depth)),
        );
        assert_eq!(spec_ids, off_ids, "{label}: speculative != off");
        assert_eq!(spec.text, off.text, "{label}");
        let report = spec.decode.expect("a speculative run reports");
        accounting(label, &spec_ids, &report);
        // The lookup fixture accepts drafts, so its pin covers full-acceptance steps (the
        // MTP fixture's adversarial head never has one accepted).
        if proposer == SpeculativeProposer::PromptLookup {
            assert!(report.accepted_tokens > 0, "{label}: {report:?}");
        }
        assert_eq!(report.proposer, ProposerKind::from(proposer), "{label}");
        assert_eq!(report.path, ProposerKind::from(proposer).label(), "{label}");
        assert_eq!(report.draft_tokens, Some(depth), "{label}");
        assert!(
            report.proposed_tokens > 0,
            "{label}: the proposer ran: {report:?}"
        );
        assert!(
            report.verify_steps > 0 && report.verify_steps <= 15,
            "{label}"
        );
        assert_eq!(
            report.mean_accepted_length(),
            Some(report.accepted_tokens as f64 / report.verify_steps as f64),
            "{label}"
        );
        assert_eq!(report.sampler, "device", "{label}");
        assert!(
            report.fallbacks.is_empty(),
            "{label}: {:?}",
            report.fallbacks
        );
        assert_eq!(
            spec.mtp.is_some(),
            proposer == SpeculativeProposer::Mtp,
            "{label}"
        );
        if matches!(provider.model, Decoder::Qwen35(_)) {
            // AC3 (sc-24435): the hybrid rejected drafts and recovered every rejection
            // through its checkpoint ring — no replay forward.
            assert!(
                report.accepted_tokens < report.proposed_tokens,
                "{label}: a draft was rejected: {report:?}"
            );
            assert_eq!(report.replay_forwards, 0, "{label}: {report:?}");
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::speculative_requests_emit_the_off_tokens_and_off_is_the_plain_loop");
}

/// sc-24446 (E5): `auto` reaches the engine with its acceptance monitor and an explicit
/// proposer does not. The synthetic hybrid's random MTP head never drafts a token the target
/// accepts, so `auto` (MTP at the recommended depth) is demoted after its first window and
/// the report says where, while the same proposer asked for explicitly runs to the end; both
/// stream exactly `off`'s tokens.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn auto_is_demoted_and_an_explicit_proposer_is_not() {
    use core_llm::Speculative;
    let provider = qwen35_mtp_provider();
    let request = |speculative| TextLlmRequest {
        max_new_tokens: 48,
        ..spec_request(speculative)
    };
    let (_, off_ids) = run(&provider, &request(Speculative::Off));
    // Untimed (the static threshold decides, no plain probe): the timed monitor's decisions
    // are pinned on a deterministic clock in the engine's tests.
    let (auto, auto_ids) =
        core_llm::with_decode_clock(None, || run(&provider, &request(Speculative::Auto)));
    let explicit = Speculative::proposer(SpeculativeProposer::Mtp, MLX_ROW.recommended_depths.mtp);
    let (asked, asked_ids) = run(&provider, &request(explicit));
    assert_eq!(auto_ids, off_ids, "a demoted run decodes plainly");
    assert_eq!(asked_ids, off_ids);
    let auto = auto.decode.unwrap();
    let asked = asked.decode.unwrap();
    assert_eq!(
        (auto.proposer, asked.proposer),
        (ProposerKind::Mtp, ProposerKind::Mtp)
    );
    assert_eq!(
        (auto.accepted_tokens, asked.accepted_tokens),
        (0, 0),
        "the fixture head never pays"
    );
    assert_eq!(
        auto.speculative_demoted_at,
        Some(1 + u64::from(core_llm::ACCEPTANCE_PROBE_VERIFIES)),
        "{auto:?}"
    );
    assert_eq!(asked.speculative_demoted_at, None, "{asked:?}");
    assert!(
        auto.proposed_tokens < asked.proposed_tokens,
        "auto stopped proposing: {} vs {}",
        auto.proposed_tokens,
        asked.proposed_tokens
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::auto_is_demoted_and_an_explicit_proposer_is_not");
}

/// AC2: the speculative resolution's fallback reaches `DecodeReport::fallbacks` through
/// `generate` — `auto` against a descriptor advertising no proposer, and (E2, the sc-24432
/// feature-end review) an explicit proposer the model does not advertise, which validates and
/// decodes plainly with the reason named rather than being refused — driven end to end;
/// dropping the resolution's fallback anywhere between `resolve_speculative` and the output
/// turns this red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_resolution_fallback_reaches_the_decode_report() {
    use core_llm::Speculative;
    let mut provider = causal_provider();
    let (auto, _) = run(&provider, &spec_request(Speculative::Auto));
    let report = auto.decode.unwrap();
    assert_eq!(
        report.proposer,
        ProposerKind::PromptLookup,
        "auto without a head"
    );
    assert_eq!(
        report.draft_tokens,
        Some(MLX_ROW.recommended_depths.prompt_lookup)
    );
    assert!(report.fallbacks.is_empty());

    provider.descriptor.capabilities.speculative.clear();
    let (auto, ids) = run(&provider, &spec_request(Speculative::Auto));
    let (off, off_ids) = run(&provider, &spec_request(Speculative::Off));
    assert_eq!(ids, off_ids, "the fallback decodes plainly");
    assert_eq!(auto.text, off.text);
    let report = auto.decode.unwrap();
    assert_eq!(report.proposer, ProposerKind::None);
    assert_eq!(
        report.fallbacks,
        vec![
            "speculative: auto found no proposer this model can run (it advertises none; \
                 decoded without a proposer)"
                .to_string()
        ]
    );

    // Unadvertised explicit proposers on the stock causal provider (no head, no draft).
    let provider = causal_provider();
    let (off, off_ids) = run(&provider, &spec_request(Speculative::Off));
    for proposer in [SpeculativeProposer::Mtp, SpeculativeProposer::DraftModel] {
        let req = spec_request(Speculative::proposer(proposer, 2));
        provider
            .validate(&req)
            .unwrap_or_else(|e| panic!("{proposer}: refused, not run plain: {e}"));
        let (out, ids) = run(&provider, &req);
        assert_eq!(ids, off_ids, "{proposer}: the plain path's tokens");
        assert_eq!(out.text, off.text);
        let report = out.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::None, "{proposer}");
        assert_eq!(
            report.fallbacks,
            vec![format!(
                "speculative: `{proposer}` is not available for this model (this model \
                     does not advertise it; decoded without a proposer)"
            )]
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::the_resolution_fallback_reaches_the_decode_report");
}

/// E2: a plan the request cannot run falls back to plain decoding with the reason named,
/// never silently and never as a failure.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_speculative_route_names_every_downgrade() {
    let run = |proposer, depth| SpeculativePlan::Run { proposer, depth };
    let causal = causal_provider();
    let hybrid = qwen35_mtp_provider();
    let route = |p: &LlamaProvider, plan, gemma4| {
        let mut why = Vec::new();
        let route = p.speculative_route(plan, gemma4, &mut why);
        (route, why)
    };

    assert_eq!(
        route(&causal, SpeculativePlan::Off, false),
        (SpeculativeRoute::Plain, vec![])
    );
    assert_eq!(
        route(&causal, run(SpeculativeProposer::PromptLookup, 4), false),
        (SpeculativeRoute::PromptLookup { width: 4 }, vec![])
    );
    assert_eq!(
        route(&hybrid, run(SpeculativeProposer::Mtp, 3), false),
        (SpeculativeRoute::Mtp { width: 3 }, vec![])
    );
    let (r, why) = route(&causal, run(SpeculativeProposer::Mtp, 3), false);
    assert_eq!(r, SpeculativeRoute::Plain);
    assert!(
        why[0].contains("`mtp` has no loaded MTP predictor"),
        "{why:?}"
    );
    let (r, why) = route(&hybrid, run(SpeculativeProposer::Mtp, 3), true);
    assert_eq!(r, SpeculativeRoute::Plain);
    assert!(why[0].contains("Gemma 4 multimodal"), "{why:?}");
    let (r, why) = route(&hybrid, run(SpeculativeProposer::DraftModel, 2), false);
    assert_eq!(r, SpeculativeRoute::Plain);
    assert!(
        why[0].contains("`draft_model` has no draft model"),
        "{why:?}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::the_speculative_route_names_every_downgrade");
}

/// The causal fixture with its own first layer resident as the draft model (sc-24436).
fn causal_provider_with_draft() -> LlamaProvider {
    let mut provider = causal_provider();
    provider.draft = Some(ResidentDraft {
        model: Decoder::Causal(Box::new(crate::decode::engine::tests::causal_model(24, 1))),
        proposable: 24,
        width: 24,
        context: 0,
    });
    let max_depth = core_llm::verify_depth_bound(&provider.descriptor.capabilities);
    provider
        .descriptor
        .capabilities
        .speculative
        .push(core_llm::draft_model_capabilities(
            max_depth,
            &MLX_ROW.recommended_depths,
        ));
    provider
}

/// sc-24436: with a resident draft `draft_model` routes onto the engine for every prompt
/// shape and decodes exactly `off`'s greedy stream with a report naming it; admission prices
/// the draft's own prefill and cache on top of the target's (E7).
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_resident_draft_routes_draft_model_and_is_priced() {
    use core_llm::Speculative;
    let provider = causal_provider_with_draft();
    let plan = SpeculativePlan::Run {
        proposer: SpeculativeProposer::DraftModel,
        depth: 3,
    };
    for gemma4 in [false, true] {
        let mut why = Vec::new();
        assert_eq!(
            provider.speculative_route(plan, gemma4, &mut why),
            SpeculativeRoute::DraftModel { width: 3 }
        );
        assert!(why.is_empty(), "{why:?}");
    }
    let (draft, ids) = run(
        &provider,
        &spec_request(Speculative::proposer(SpeculativeProposer::DraftModel, 3)),
    );
    let (off, off_ids) = run(&provider, &spec_request(Speculative::Off));
    assert_eq!(ids, off_ids);
    assert_eq!(draft.text, off.text);
    let report = draft.decode.unwrap();
    assert_eq!(report.proposer, ProposerKind::DraftModel);
    assert_eq!(report.draft_tokens, Some(3));
    assert!(report.proposed_tokens > 0 && report.fallbacks.is_empty());

    let price = |route| {
        provider
            .speculative_request_bytes(route, None, 64, 32, 0)
            .unwrap()
    };
    let lookup = price(SpeculativeRoute::PromptLookup { width: 4 });
    let with_draft = price(SpeculativeRoute::DraftModel { width: 4 });
    let draft_model = &provider.draft.as_ref().unwrap().model;
    let draft_request = estimate_mlx_request_bytes(
        64,
        32 + 4,
        draft_model.memory_geometry(),
        0,
        0,
        draft_model.workspace_contract(),
    )
    .unwrap();
    assert!(draft_request > 0);
    assert_eq!(
        with_draft,
        lookup + draft_request,
        "the target's verify overshoot plus the draft's own request"
    );
    // Without a resident draft nothing extra is priced (and the route never forms).
    assert_eq!(
        causal_provider().speculative_request_bytes(
            SpeculativeRoute::DraftModel { width: 4 },
            None,
            64,
            32,
            0
        ),
        Some(lookup)
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_resident_draft_routes_draft_model_and_is_priced");
}

/// sc-24436 E2: a `draft_model` route whose request outruns the draft's context window — the
/// prompt (the Gemma 4 expansion included, at its second check), the budget and a step's
/// `K + 1` positions — runs `auto` (prompt lookup on this target) by name; within it the
/// route stands, and an unbounded draft never falls back.
#[test]
fn a_draft_route_past_the_draft_context_runs_auto_by_name() {
    let mut provider = causal_provider_with_draft();
    provider.draft.as_mut().unwrap().context = 20;
    let draft = SpeculativeRoute::DraftModel { width: 3 };
    for gemma4 in [false, true] {
        let mut why = Vec::new();
        assert_eq!(
            provider.fit_draft_route(draft, 10, 6, gemma4, &mut why),
            draft
        );
        assert!(why.is_empty(), "{why:?}");
        let fallen = provider.fit_draft_route(draft, 10, 7, gemma4, &mut why);
        assert_eq!(
            fallen,
            SpeculativeRoute::PromptLookup {
                width: MLX_ROW.recommended_depths.prompt_lookup as usize
            }
        );
        assert_eq!(why.len(), 1);
        assert!(
            why[0].contains("exceeds the draft model's context window 20"),
            "{why:?}"
        );
        // Other routes are untouched.
        let lookup = SpeculativeRoute::PromptLookup { width: 3 };
        assert_eq!(
            provider.fit_draft_route(lookup, 10, 700, gemma4, &mut why),
            lookup
        );
    }
    provider.draft.as_mut().unwrap().context = 0;
    let mut why = Vec::new();
    assert_eq!(
        provider.fit_draft_route(draft, 10, 7000, false, &mut why),
        draft
    );
}

/// sc-24436 E2/E7 through the real load (the sc-24432 feature-end review; the Candle twin
/// is `a_draft_refused_by_load_admission_is_named_in_the_load_fallbacks`): a budget that fits
/// the target but not target + draft refuses the draft by name in BOTH the load report's
/// `draft` and its `fallbacks`, and the target loads alone.
#[test]
fn a_draft_refused_by_load_admission_is_named_in_the_load_fallbacks() {
    let root = crate::test_fixture::Fixture::new("mlx-llm-draft-refused-", None);
    let fixture = core_llm_testkit::write_draft_model_fixture(&root).unwrap();
    let spec = fixture.spec_with_draft();
    let target = crate::load_memory::required_bytes(&spec).unwrap();
    let draft =
        crate::load_memory::required_bytes(&LoadSpec::dense(fixture.draft.to_string_lossy()))
            .unwrap();
    let provider = with_load_budget(target + draft - 1, || LlamaProvider::load(&spec))
        .expect("the target fits and loads alone");
    let report = provider.load_report().unwrap();
    let refused = report.draft.clone().expect("the named draft is reported");
    let refusal = refused.refusal.clone().expect("refused by load admission");
    assert!(refusal.starts_with("draft model:"), "{refusal}");
    assert_eq!(refused.source, fixture.draft.to_string_lossy());
    assert!(
        report.fallbacks.contains(&refusal),
        "the refused draft is named in the load fallbacks: {:?}",
        report.fallbacks
    );
    assert!(provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::DraftModel)
        .is_none());
}

/// sc-24436 E7: a named draft is admitted at load only beside the target; without room — or
/// when it cannot be priced — it is refused by name and the target still loads; a target
/// that does not fit is refused exactly as before.
#[test]
fn a_draft_is_admitted_at_load_only_beside_the_target() {
    let root = crate::test_fixture::Fixture::new("mlx-llm-draft-admission-", None);
    let fixture = core_llm_testkit::write_draft_model_fixture(&root).unwrap();
    let spec = fixture.spec_with_draft();
    let target = crate::load_memory::required_bytes(&spec).unwrap();
    let draft =
        crate::load_memory::required_bytes(&LoadSpec::dense(fixture.draft.to_string_lossy()))
            .unwrap();
    assert!(
        draft > 0 && draft < target,
        "the draft is the smaller model"
    );

    match DraftPlan::admit(&spec, target, target + draft).unwrap() {
        DraftPlan::Load(draft_spec) => {
            assert_eq!(draft_spec.source, fixture.draft.to_string_lossy());
            assert_eq!(draft_spec.draft_source, None);
        }
        _ => panic!("room for both loads the draft"),
    }
    match DraftPlan::admit(&spec, target, target + draft - 1).unwrap() {
        DraftPlan::Refused(report) => {
            assert!(!report.is_resident());
            assert!(report.refusal.unwrap().starts_with("draft model:"));
        }
        _ => panic!("no room beside the target refuses the draft"),
    }
    assert!(
        DraftPlan::admit(&spec, target, target - 1).is_err(),
        "the target alone"
    );
    let missing = LoadSpec::dense(fixture.target.to_string_lossy())
        .with_draft(root.join("no-such-draft").to_string_lossy());
    match DraftPlan::admit(&missing, target, u64::MAX).unwrap() {
        DraftPlan::Refused(report) => {
            assert!(report.refusal.unwrap().contains("cannot be priced"))
        }
        _ => panic!("an unpriceable draft is refused"),
    }
    let unnamed = LoadSpec::dense(fixture.target.to_string_lossy());
    assert!(matches!(
        DraftPlan::admit(&unnamed, target, target).unwrap(),
        DraftPlan::None
    ));

    // End to end, an unloadable draft never fails the target's load.
    let provider = LlamaProvider::load(&missing).unwrap();
    let report = provider.load_report().unwrap().draft.unwrap();
    assert!(!report.is_resident());
    assert!(provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::DraftModel)
        .is_none());
}

/// Both decoder families advertise prompt lookup (the engine runs it on either) at their
/// geometry's depth; the Qwen3.8 head keeps its MTP advertisement — the same finite depth on
/// the legacy field and the per-proposer list (sc-24438) — and a depth past the bound is
/// admitted (it is clamped at generate, by name).
#[test]
fn every_decoder_advertises_prompt_lookup() {
    use core_llm::Speculative;
    for provider in [causal_provider(), qwen35_mtp_provider()] {
        let caps = &provider.descriptor().capabilities;
        let lookup = caps.proposer(SpeculativeProposer::PromptLookup).unwrap();
        // The fixtures' head dims (4, 8) are not vector-kernel dims: the 8-row bound.
        assert_eq!(lookup.max_depth, SPECULATIVE_MAX_DEPTH);
        assert_eq!(
            lookup.recommended_depth,
            MLX_ROW.recommended_depths.prompt_lookup
        );
        let too_deep = spec_request(Speculative::proposer(
            SpeculativeProposer::PromptLookup,
            lookup.max_depth + 1,
        ));
        provider.validate(&too_deep).unwrap();
    }
    let hybrid = qwen35_mtp_provider();
    let caps = &hybrid.descriptor().capabilities;
    let mtp = caps.proposer(SpeculativeProposer::Mtp).unwrap();
    assert_eq!(
        (mtp.max_depth, mtp.recommended_depth),
        (SPECULATIVE_MAX_DEPTH, MLX_ROW.recommended_depths.mtp)
    );
    assert_eq!(
        caps.mtp,
        Some(core_llm::MtpCapabilities {
            max_draft_tokens: mtp.max_depth,
            recommended_draft_tokens: mtp.recommended_depth,
        }),
        "the legacy field agrees with the proposer list"
    );
    assert!(caps.speculative.contains(&mtp));
    assert!(causal_provider()
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_none());
}

/// sc-24438 AC1 (E4): a loaded model's advertised depth is its attention geometry's — one less
/// than the rows MLX's decode kernel verifies in one call — for both proposers and on the
/// legacy MTP field: the shipped Qwen3.8-27B config (24q/4kv/hd 256) → 4, a 35B-A3B-shaped
/// MoE config (16q/2kv/hd 256) → 3, a gqa-4 hd-128 Llama → 7, a gqa-8 one → 3, Gemma 4 the
/// narrower of its sliding / full layers, DeepSeek MLA (192/128, head-expanded keys) → 7.
#[test]
fn the_advertised_depth_is_the_attention_geometrys() {
    let depths = |caps: &TextLlmCapabilities| {
        let lookup = caps.proposer(SpeculativeProposer::PromptLookup).unwrap();
        assert!(lookup.recommended_depth <= lookup.max_depth);
        let mtp = caps.proposer(SpeculativeProposer::Mtp).map(|m| {
            assert_eq!(
                caps.mtp,
                Some(core_llm::MtpCapabilities {
                    max_draft_tokens: m.max_depth,
                    recommended_draft_tokens: m.recommended_depth,
                })
            );
            m.max_depth
        });
        (lookup.max_depth, mtp)
    };
    let qwen38: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/reference/qwen38/config.json"
    )))
    .unwrap();
    let dense = Qwen35Config::from_json(&qwen38).unwrap();
    assert_eq!(
        (dense.num_heads, dense.num_kv_heads, dense.head_dim),
        (24, 4, 256)
    );
    assert_eq!(
        depths(&descriptor_for_qwen35(&dense).capabilities),
        (4, Some(4))
    );
    let mut a3b = qwen38.clone();
    let text = a3b["text_config"].as_object_mut().unwrap();
    text.insert("num_attention_heads".into(), json!(16));
    text.insert("num_key_value_heads".into(), json!(2));
    text.insert("model_type".into(), json!("qwen3_5_moe_text"));
    text.insert("num_experts".into(), json!(256));
    text.insert("num_experts_per_tok".into(), json!(8));
    text.insert("moe_intermediate_size".into(), json!(512));
    text.insert("shared_expert_intermediate_size".into(), json!(512));
    let a3b = Qwen35Config::from_json(&a3b).unwrap();
    assert!(a3b.moe.is_some());
    assert_eq!(
        depths(&descriptor_for_qwen35(&a3b).capabilities),
        (3, Some(3))
    );

    let llama = |heads: i32, kv: i32, extra: serde_json::Value| {
        let mut v = json!({
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": heads * 128, "intermediate_size": 256, "num_hidden_layers": 2,
            "num_attention_heads": heads, "num_key_value_heads": kv,
            "vocab_size": 32, "rms_norm_eps": 1e-5, "max_position_embeddings": 64
        });
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        depths(&descriptor_for(&ModelConfig::from_json(&v).unwrap()).capabilities)
    };
    assert_eq!(llama(32, 8, json!({})), (7, None));
    assert_eq!(llama(64, 8, json!({})), (3, None));
    assert_eq!(
        llama(
            16,
            16,
            json!({"model_type": "deepseek_v2", "architectures": ["DeepseekV2ForCausalLM"],
                       "kv_lora_rank": 64, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
                       "v_head_dim": 128})
        ),
        (7, None)
    );

    // Gemma 4: sliding hd 256 over 2 KV heads (gqa 8 → 4 rows) beats the full layers' hd 512.
    let gemma4 = |kv: i32| {
        let cfg = ModelConfig {
            num_heads: 16,
            gemma4: Some(Box::new(crate::config::Gemma4Config {
                layer_types: vec![
                    crate::config::LayerAttentionType::Sliding,
                    crate::config::LayerAttentionType::Full,
                ],
                sliding: crate::config::LayerAttention {
                    head_dim: 256,
                    num_kv_heads: kv,
                    rope_type: crate::config::RopeType::Default,
                    rope_theta: 1e4,
                    partial_rotary_factor: 1.0,
                    rope_factor: 1.0,
                    sliding_window: Some(512),
                    k_eq_v: false,
                },
                full: crate::config::LayerAttention {
                    head_dim: 512,
                    num_kv_heads: 1,
                    rope_type: crate::config::RopeType::Default,
                    rope_theta: 1e6,
                    partial_rotary_factor: 1.0,
                    rope_factor: 1.0,
                    sliding_window: None,
                    k_eq_v: true,
                },
                bidirectional: None,
                num_kv_shared_layers: 0,
                use_double_wide_mlp: false,
            })),
            num_layers: 2,
            ..crate::decode::engine::tests::tiny_llama(8).config().clone()
        };
        depths(&descriptor_for(&cfg).capabilities)
    };
    assert_eq!(gemma4(2), (3, None));
    assert_eq!(gemma4(8), (7, None));
}

/// sc-24438 AC1: the advertised max depth is finite and backend-true (on these fixtures'
/// head dims, the 8-row bound) and a request above it runs at it, with the clamp named in
/// `DecodeReport::fallbacks`, emitting exactly `off`'s stream: prompt lookup on both
/// decoders, and MTP (the new option and the legacy field) on the Qwen3.8 head.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_too_deep_request_runs_at_the_advertised_max_and_names_the_clamp() {
    use core_llm::{MtpMode, Speculative};
    let clamp = |proposer: &str, asked: u32, max: u32| {
        vec![format!(
            "speculative: `{proposer}` depth {asked} clamped to {max} (advertised 1..={max})"
        )]
    };
    let check = |provider: &LlamaProvider, req: TextLlmRequest, kind, max, label: &str| {
        let (_, off_ids) = run(provider, &spec_request(Speculative::Off));
        let (out, ids) = run(provider, &req);
        let report = out.decode.unwrap();
        assert_eq!(report.proposer, kind, "{label}");
        assert_eq!(report.draft_tokens, Some(max), "{label}");
        assert_eq!(ids, off_ids, "{label}: the clamped run is greedy-exact");
        report.fallbacks
    };
    let max = |provider: &LlamaProvider, proposer| {
        provider
            .descriptor()
            .capabilities
            .proposer(proposer)
            .unwrap()
            .max_depth
    };

    for (label, provider) in [
        ("causal", causal_provider()),
        ("qwen35", qwen35_mtp_provider()),
    ] {
        let max = max(&provider, SpeculativeProposer::PromptLookup);
        let asked = max + 5;
        let req = spec_request(Speculative::proposer(
            SpeculativeProposer::PromptLookup,
            asked,
        ));
        let fallbacks = check(&provider, req, ProposerKind::PromptLookup, max, label);
        assert_eq!(fallbacks, clamp("prompt_lookup", asked, max), "{label}");
    }

    let hybrid = qwen35_mtp_provider();
    let mtp_max = max(&hybrid, SpeculativeProposer::Mtp);
    let new = spec_request(Speculative::proposer(SpeculativeProposer::Mtp, 40));
    let fallbacks = check(&hybrid, new, ProposerKind::Mtp, mtp_max, "mtp");
    assert_eq!(fallbacks, clamp("mtp", 40, mtp_max));
    let mut legacy = spec_request(Speculative::Off);
    legacy.speculative = None;
    legacy.mtp = Some(MtpMode::Enabled { draft_tokens: 40 });
    let fallbacks = check(&hybrid, legacy, ProposerKind::Mtp, mtp_max, "legacy");
    assert_eq!(fallbacks, clamp("mtp", 40, mtp_max));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_too_deep_request_runs_at_the_advertised_max_and_names_the_clamp");
}

/// sc-24438 AC2: a sparse-MoE Qwen35 snapshot carrying an MTP head — its predictor layer a
/// sparse-MoE block, as the 35B-A3B ships it, in the fused (Qwen3.6) and the per-expert
/// (Qwen3.5) expert layout — loads through the production load path, advertises MTP on both
/// fields, and **runs** it: `{mtp}` at depths 1, 3 and the advertised max, the legacy
/// `enabled 3` and `auto` each report the MTP proposer with drafts proposed, and emit exactly
/// `off`'s stream.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_moe_snapshot_with_an_mtp_head_runs_mtp() {
    use crate::models::qwen35::tests::{cfg_json_moe_mtp, seeded_moe_mtp_tensors};
    use core_llm::{MtpMode, Speculative};
    for fused in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let mut config = cfg_json_moe_mtp();
        config["model_type"] = json!("qwen3_5_moe");
        config.as_object_mut().unwrap().remove("vision_config");
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let entries: Vec<String> = (0..50).map(|i| format!("\"t{i}\": {i}")).collect();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            format!(
                r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                    "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                    "decoder": null,
                    "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
                entries.join(", ")
            ),
        )
        .unwrap();
        let cfg = Qwen35Config::from_json(&config).unwrap();
        let tensors = seeded_moe_mtp_tensors(&cfg, fused);
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();

        let label = if fused { "fused" } else { "per-expert" };
        let provider = LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string()))
            .expect("a MoE checkpoint with an MTP head loads");
        let caps = &provider.descriptor().capabilities;
        let mtp = caps.proposer(SpeculativeProposer::Mtp).expect(label);
        assert_eq!(
            caps.mtp,
            Some(core_llm::MtpCapabilities {
                max_draft_tokens: mtp.max_depth,
                recommended_draft_tokens: mtp.recommended_depth,
            }),
            "{label}"
        );

        let (_, off_ids) = run(&provider, &spec_request(Speculative::Off));
        let mut legacy = spec_request(Speculative::Off);
        legacy.speculative = None;
        legacy.mtp = Some(MtpMode::Enabled { draft_tokens: 3 });
        // Epic AT1: the explicit option at depths 1, 3 and the advertised max.
        let explicit = |depth| {
            (
                depth,
                spec_request(Speculative::proposer(SpeculativeProposer::Mtp, depth)),
            )
        };
        for (case, (depth, req)) in [
            ("explicit 1", explicit(1)),
            ("explicit 3", explicit(3)),
            ("explicit max", explicit(mtp.max_depth)),
            ("legacy", (3, legacy)),
            // `auto` runs at the advertised recommended depth (sc-24446), not a literal.
            (
                "auto",
                (mtp.recommended_depth, spec_request(Speculative::Auto)),
            ),
        ] {
            provider.validate(&req).unwrap();
            let (out, ids) = run(&provider, &req);
            let report = out.decode.unwrap();
            assert_eq!(report.proposer, ProposerKind::Mtp, "{label} {case}");
            assert_eq!(report.draft_tokens, Some(depth), "{label} {case}");
            assert!(
                report.proposed_tokens > 0,
                "{label} {case}: the head drafted"
            );
            assert!(report.fallbacks.is_empty(), "{label} {case}");
            assert_eq!(ids, off_ids, "{label} {case}: greedy-exact");
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_moe_snapshot_with_an_mtp_head_runs_mtp"
    );
}

/// Admission prices what each route holds: prompt lookup on a softmax decoder pays only its
/// verify overshoot (it rolls back by truncation). On the hybrid every speculative route pays
/// the DeltaNet checkpoint ring its rollback holds (E7, sc-24435) — prompt lookup the ring plus
/// the overshoot, MTP the ring plus its predictor cache and verify rows — and the recurrent
/// state once: no step-start snapshot, no clone/replay copies.
#[test]
fn speculative_admission_prices_the_rollback_the_route_holds() {
    let causal = causal_provider();
    let price =
        |p: &LlamaProvider, route| p.speculative_request_bytes(route, None, 64, 32, 0).unwrap();
    let off = price(&causal, SpeculativeRoute::Plain);
    let lookup = price(&causal, SpeculativeRoute::PromptLookup { width: 4 });
    let g = causal.model.memory_geometry();
    let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
    let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
    assert_eq!(
        lookup,
        off + 4 * (kv_position + rows),
        "truncation: overshoot only"
    );

    let hybrid = qwen35_mtp_provider();
    let Decoder::Qwen35(model) = &hybrid.model else {
        unreachable!("the hybrid fixture")
    };
    let off = price(&hybrid, SpeculativeRoute::Plain);
    let g = hybrid.model.memory_geometry();
    let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
    let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
    let kv = (64 + 32) * kv_position;
    for width in [1usize, 3, 8] {
        let ring = model.checkpoint_ring_bytes(width).unwrap();
        let w = width as u64;
        assert!(ring > 0);
        assert_eq!(
            price(&hybrid, SpeculativeRoute::PromptLookup { width }),
            off + ring + w * (kv_position + rows),
            "lookup {width}: the ring plus the overshoot"
        );
        assert_eq!(
            price(&hybrid, SpeculativeRoute::Mtp { width }),
            off + ring + 2 * kv + w * rows,
            "mtp {width}: the ring plus the predictor cache and verify rows"
        );
    }

    // The MoE hybrid prices on the eager contract, with the same once-charged recurrent term.
    let moe = qwen35_moe_provider();
    let Decoder::Qwen35(model) = &moe.model else {
        unreachable!("the MoE fixture")
    };
    assert!(matches!(
        moe.model.workspace_contract(),
        MlxWorkspaceContract::Eager
    ));
    let off = price(&moe, SpeculativeRoute::Plain);
    let g = moe.model.memory_geometry();
    let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
    let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
    let kv = (64 + 32) * kv_position;
    for width in [1usize, 3, 8] {
        let ring = model.checkpoint_ring_bytes(width).unwrap();
        let w = width as u64;
        assert_eq!(
            price(&moe, SpeculativeRoute::PromptLookup { width }),
            off + ring + w * (kv_position + rows),
            "moe lookup {width}"
        );
        // The generic runtime terms also hold the verify width's K/V positions and their
        // block padding (sc-24446).
        let positions = |extra: u64| w + kv_block_padding(96 + extra).unwrap();
        assert_eq!(
            price(&moe, SpeculativeRoute::Mtp { width }),
            off + ring
                + 2 * kv
                + w * rows
                + (positions(w) - (kv_block_padding(96).unwrap())) * kv_position,
            "moe mtp {width}: the ring and the live state charged once"
        );
    }

    // A draft model (sc-24436) on the hybrid target verifies like prompt lookup: the ring
    // plus the overshoot. A hybrid draft holds its own proposal window in its ring, armed for
    // the same width: its request prices that ring, never a snapshot.
    for width in [1usize, 4] {
        assert_eq!(
            price(&hybrid, SpeculativeRoute::DraftModel { width }),
            price(&hybrid, SpeculativeRoute::PromptLookup { width }),
            "hybrid target, draft {width}: the ring plus the overshoot"
        );
    }
    let mut drafted = causal_provider();
    drafted.draft = Some(ResidentDraft {
        model: Decoder::Qwen35(Box::new(crate::decode::engine::tests::qwen35(false))),
        proposable: 24,
        width: 24,
        context: 0,
    });
    let Some(Decoder::Qwen35(draft)) = drafted.draft.as_ref().map(|d| &d.model) else {
        unreachable!("the hybrid draft")
    };
    for width in [1usize, 4] {
        let mut geometry = drafted.draft.as_ref().unwrap().model.memory_geometry();
        geometry.recurrent_bytes += draft.checkpoint_ring_bytes(width).unwrap();
        let draft_request = estimate_mlx_request_bytes(
            64,
            32 + width as u32,
            geometry,
            0,
            0,
            drafted.draft.as_ref().unwrap().model.workspace_contract(),
        )
        .unwrap();
        assert_eq!(
            price(&drafted, SpeculativeRoute::DraftModel { width }),
            price(&causal, SpeculativeRoute::PromptLookup { width }) + draft_request,
            "hybrid draft {width}: its request with its ring, no snapshot"
        );
    }
}

// ---- Prism/Bonsai companion MTP head (epic sc-24432, story sc-24444). ----

/// A Prism snapshot directory and, when `head` names a variant, a companion head beside it.
struct PrismLoad {
    target: crate::test_fixture::Fixture,
    head: crate::test_fixture::Fixture,
}

fn prism_load(head_text: Option<serde_json::Value>) -> PrismLoad {
    let target = crate::test_fixture::Fixture::new("mlx-prism-target-", None);
    // Seed 6 decodes a varied greedy run (not a single repeated id) through the chat template.
    crate::synthetic::write_prism_snapshot(&target, 6);
    let head = crate::test_fixture::Fixture::new("mlx-prism-head-", None);
    let text =
        head_text.unwrap_or_else(|| crate::synthetic::prism_qwen35_config()["text_config"].clone());
    crate::synthetic::write_companion_head(&head, &text, 5, None);
    PrismLoad { target, head }
}

fn prism_request(speculative: core_llm::Speculative) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user("t3 t9 t40 t11 t3 t9 t40 t11 t7")],
        sampling: Sampling::greedy(),
        max_new_tokens: 20,
        seed: Some(3),
        speculative: Some(speculative),
        ..Default::default()
    }
}

/// E2 through the real load (sc-24432 feature-end review; the Candle twin is
/// `a_configured_mtp_head_the_snapshot_cannot_run_is_a_named_load_fallback`): a snapshot whose
/// config declares a native MTP head it stores no tensor of loads the target plain, names the
/// `mtp:` fallback in the load report, advertises no MTP and decodes.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_configured_mtp_head_the_snapshot_does_not_carry_is_a_named_load_fallback() {
    let target = crate::test_fixture::Fixture::new("mlx-mtp-absent-", None);
    crate::synthetic::write_prism_snapshot(&target, 6);
    let path = target.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
    std::fs::write(&path, config.to_string()).unwrap();
    let provider = LlamaProvider::load(&LoadSpec::dense(target.to_str().unwrap()))
        .expect("the target still loads (E2)");
    let fallbacks = provider.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(
        fallbacks[0].starts_with("mtp: ") && fallbacks[0].contains("stores no `mtp.*` tensor"),
        "{fallbacks:?}"
    );
    assert!(provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_none());
    assert!(provider.descriptor().capabilities.mtp.is_none());
    let (_, ids) = run(&provider, &prism_request(core_llm::Speculative::Off));
    assert_eq!(ids.len(), 20);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_configured_mtp_head_the_snapshot_does_not_carry_is_a_named_load_fallback");
}

/// Per step of `req`'s plain greedy run on `provider`'s hybrid decoder: whether the step is a
/// BF16 near-tie — its top-2 logit margin within two BF16 ulps at the top logit's magnitude,
/// where a different (and equally correct) BF16 reduction order, a multi-row verify or a
/// prefix-hit suffix prefill, may pick the other token. Stepped token by token on the run's own
/// stream, the reference `off` is.
fn prism_near_ties(provider: &LlamaProvider, req: &TextLlmRequest, off: &[u32]) -> Vec<bool> {
    let (_, mut ids) = provider.render_prompt(req, &req.messages).unwrap();
    let Decoder::Qwen35(model) = &provider.model else {
        panic!("a Prism snapshot loads the hybrid decoder");
    };
    let mut cache = model.new_cache();
    let mut logits = model
        .decode_logits(&crate::primitives::input_ids(&ids), &mut cache, 0)
        .unwrap();
    let mut near = Vec::with_capacity(off.len());
    for &token in off {
        let mut row = crate::primitives::nn::to_f32_host(&logits).unwrap();
        row.sort_by(|a, b| b.total_cmp(a));
        let ulp = (2f32).powi(row[0].abs().max(f32::MIN_POSITIVE).log2().floor() as i32 - 7);
        near.push(row[0] - row[1] <= 2.0 * ulp);
        ids.push(token as i32);
        let offset = cache.offset();
        logits = model
            .decode_logits(
                &crate::primitives::input_ids(&[token as i32]),
                &mut cache,
                offset,
            )
            .unwrap();
    }
    near
}

/// `ids` is `off`'s greedy run up to its first BF16 near-tie ([`prism_near_ties`]) and may
/// part from it only there: the first differing step must be a near-tie of `off`'s, and with
/// no near-tie the runs are identical.
fn assert_bf16_greedy_parity(label: &str, ids: &[u32], off: &[u32], near_ties: &[bool]) {
    assert_eq!(ids.len(), off.len(), "{label}: length");
    match ids.iter().zip(off).position(|(a, b)| a != b) {
        None => {}
        Some(step) => assert!(
            near_ties[step],
            "{label}: diverged at step {step}, which is not a BF16 near-tie of the off run \
                 (near-ties at {:?}): {ids:?} vs {off:?}",
            near_ties
                .iter()
                .enumerate()
                .filter_map(|(i, &n)| n.then_some(i))
                .collect::<Vec<_>>()
        ),
    }
}

/// Story sc-24444 AC2: a Prism snapshot loaded with a companion head advertises MTP, and
/// `{proposer: mtp}` at depths 1, 3 and the advertised max — through the real load path and the provider's
/// engine — emits exactly `off`'s greedy tokens, with the report naming the head as the
/// proposer that ran.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn prism_with_a_companion_head_decodes_mtp_with_the_off_tokens() {
    use core_llm::Speculative;
    let fx = prism_load(None);
    let spec =
        LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
    let provider = LlamaProvider::load(&spec).unwrap();
    assert_eq!(provider.descriptor().family, "prism_hadamard_qwen35");
    let report = provider.load_report().unwrap();
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    // The companion head advertises the same backend-true depth a native head does (sc-24438).
    let advertised = provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .expect("the attached head is advertised");
    assert_eq!(advertised.recommended_depth, MLX_ROW.recommended_depths.mtp);

    let (off, off_ids) = run(&provider, &prism_request(Speculative::Off));
    assert_eq!(off_ids.len(), 20);
    let distinct: std::collections::BTreeSet<_> = off_ids.iter().collect();
    assert!(
        distinct.len() > 2,
        "fixture must not be degenerate: {off_ids:?}"
    );
    assert_eq!(
        off_ids,
        plain_loop(&provider, &prism_request(Speculative::Off))
    );
    assert_eq!(off.decode.unwrap().proposer, ProposerKind::None);
    // Since sc-20671 the Prism decoder computes in BF16: the head's verify (a multi-row
    // forward) and a prefix-hit prefill keep `off`'s tokens up to a BF16 near-tie — this
    // fixture has one at step 16 — and may part only there. The f32 twin is strict
    // (`prism_companion_head_in_f32_keeps_the_off_tokens_exactly`).
    let near_ties = prism_near_ties(&provider, &prism_request(Speculative::Off), &off_ids);
    for depth in [1, 3, advertised.max_depth] {
        let req = prism_request(Speculative::proposer(SpeculativeProposer::Mtp, depth));
        let (out, ids) = run(&provider, &req);
        assert_bf16_greedy_parity(&format!("depth {depth}"), &ids, &off_ids, &near_ties);
        let report = out.decode.expect("the engine reports its path");
        assert_eq!(report.proposer, ProposerKind::Mtp, "depth {depth}");
        assert_eq!(report.draft_tokens, Some(depth), "depth {depth}");
        assert!(
            report.proposed_tokens > 0,
            "depth {depth}: the head drafted"
        );
        assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    }

    // Without the head the same snapshot has no MTP, and loads the same greedy tokens.
    let bare = LlamaProvider::load(&LoadSpec::dense(fx.target.to_str().unwrap())).unwrap();
    assert!(bare.load_report().unwrap().fallbacks.is_empty());
    assert!(bare
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_none());
    assert_eq!(run(&bare, &prism_request(Speculative::Off)).1, off_ids);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::prism_with_a_companion_head_decodes_mtp_with_the_off_tokens");
}

/// The f32 twin of the BF16 companion-head parity above: the same Prism fixture computing in
/// f32 keeps `off`'s greedy tokens exactly over the whole run — every MTP depth, `auto`, and a
/// prefix-cache hit — so the BF16 run's only allowance is its rounding.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn prism_companion_head_in_f32_keeps_the_off_tokens_exactly() {
    use core_llm::Speculative;
    crate::models::qwen35::with_compute_dtype(Dtype::Float32, || {
        let fx = prism_load(None);
        let spec =
            LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
        let provider = LlamaProvider::load(&spec).unwrap();
        let advertised = provider
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .expect("the attached head is advertised");
        let (_, off_ids) = run(&provider, &prism_request(Speculative::Off));
        assert_eq!(off_ids.len(), 20);
        assert_eq!(
            off_ids,
            plain_loop(&provider, &prism_request(Speculative::Off))
        );
        let (again, again_ids) = run(&provider, &prism_request(Speculative::Off));
        assert!(
            again.decode.unwrap().prefix_hit_tokens > 0,
            "a prefix-cache hit"
        );
        assert_eq!(again_ids, off_ids, "the prefix-cache hit");
        for depth in [1, 3, advertised.max_depth] {
            let req = prism_request(Speculative::proposer(SpeculativeProposer::Mtp, depth));
            assert_eq!(run(&provider, &req).1, off_ids, "depth {depth}");
        }
        assert_eq!(
            run(&provider, &prism_request(Speculative::Auto)).1,
            off_ids,
            "auto"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::prism_companion_head_in_f32_keeps_the_off_tokens_exactly");
    })
}

/// Story sc-24444 AC2 / E2: a head whose geometry does not match, a missing head, and a head
/// named for a non-Qwen3.5 target are each refused with a named load fallback; the model still
/// loads, advertises no MTP and decodes plainly.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn an_unattachable_companion_head_is_a_named_load_fallback_never_a_load_failure() {
    use core_llm::Speculative;
    let mut text = crate::synthetic::prism_qwen35_config()["text_config"].clone();
    text["intermediate_size"] = json!(128);
    let fx = prism_load(Some(text));
    let spec =
        LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
    let provider = LlamaProvider::load(&spec).expect("the target still loads (E2)");
    let fallbacks = provider.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(fallbacks[0].starts_with("mtp_head: "), "{fallbacks:?}");
    assert!(
        fallbacks[0].contains("intermediate_size 128 != target 256"),
        "{fallbacks:?}"
    );
    assert!(
        fallbacks[0].contains("the model loaded without a companion head"),
        "{fallbacks:?}"
    );
    assert!(provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_none());
    let (out, ids) = run(&provider, &prism_request(Speculative::Off));
    assert_eq!(ids.len(), 20);
    assert!(out.decode.unwrap().fallbacks.is_empty());
    // An explicit MTP request (not advertised) decodes plainly with the reason named (E2,
    // never refused), and `auto` names why it ran no MTP by choosing prompt lookup.
    let (mtp, mtp_ids) = run(
        &provider,
        &prism_request(Speculative::proposer(SpeculativeProposer::Mtp, 3)),
    );
    // BF16 (sc-20671): this run's prefill is a prefix-cache hit of the first's, which keeps
    // its tokens up to a near-tie (`prism_near_ties`).
    let near_ties = prism_near_ties(&provider, &prism_request(Speculative::Off), &ids);
    assert_bf16_greedy_parity("the plain path's tokens", &mtp_ids, &ids, &near_ties);
    let mtp = mtp.decode.unwrap();
    assert_eq!(mtp.proposer, ProposerKind::None);
    assert_eq!(mtp.fallbacks.len(), 1, "{:?}", mtp.fallbacks);
    assert!(
        mtp.fallbacks[0].starts_with("speculative: `mtp` is not available"),
        "{:?}",
        mtp.fallbacks
    );
    let (auto, auto_ids) = run(&provider, &prism_request(Speculative::Auto));
    assert_bf16_greedy_parity("auto", &auto_ids, &ids, &near_ties);
    assert_eq!(auto.decode.unwrap().proposer, ProposerKind::PromptLookup);

    let missing = LoadSpec::dense(fx.target.to_str().unwrap())
        .with_mtp_head(fx.target.join("no-such-head").to_str().unwrap());
    let provider = LlamaProvider::load(&missing).expect("a missing head is not fatal");
    let fallbacks = provider.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(fallbacks[0].contains("is not a directory"), "{fallbacks:?}");

    let mut causal = causal_provider();
    causal.attach_mtp_head(Some(fx.head.as_ref()));
    let fallbacks = &causal.load_report.fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(
        fallbacks[0].contains("attach to Qwen3.5/3.8-family"),
        "{fallbacks:?}"
    );
    assert!(causal.descriptor.capabilities.mtp.is_none());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::an_unattachable_companion_head_is_a_named_load_fallback_never_a_load_failure");
}

/// Story sc-24444 E7: the head's resident bytes are its safetensors payload plus the header's
/// norm intermediates, and load admission adds them to the target's: a budget that fits the
/// target but not target + head refuses the head by name and nothing else. The head's
/// per-request cache is priced by request admission on the `mtp` route.
/// Story sc-24444 E7 with sc-24436/sc-24437: the companion head, a draft model and the
/// prefix cache are admitted together — the head on top of the target and the admitted
/// draft, the prefix cache in what they leave. A budget that fits target + draft but not the
/// head too keeps the draft, refuses the head by name and gives the cache the rest; one that
/// fits all three plus `x` bytes settles the prefix cache at exactly `x`.
#[test]
fn companion_head_draft_and_prefix_cache_are_admitted_together() {
    let fx = prism_load(None);
    let head: &Path = fx.head.as_ref();
    let head_bytes = crate::load_memory::companion_head_bytes(head).unwrap();
    let target =
        crate::load_memory::required_bytes(&LoadSpec::dense(fx.target.to_str().unwrap())).unwrap();
    let mut spec = LoadSpec::dense(fx.target.to_str().unwrap())
        .with_mtp_head(fx.head.to_str().unwrap())
        .with_draft(fx.target.to_str().unwrap());
    spec.prefix_cache_bytes = Some(u64::MAX / 4);
    let draft =
        crate::load_memory::required_bytes(&draft_load_spec(&spec, fx.target.to_str().unwrap()))
            .unwrap();

    let short = with_load_budget(target + draft + head_bytes - 1, || {
        LlamaProvider::load(&spec)
    })
    .expect("target + draft fit");
    let report = short.load_report().unwrap();
    assert!(report.draft.as_ref().unwrap().is_resident(), "{report:?}");
    assert_eq!(report.fallbacks.len(), 1, "{:?}", report.fallbacks);
    assert!(
        report.fallbacks[0].starts_with("mtp_head: refused by load admission"),
        "{:?}",
        report.fallbacks
    );
    assert_eq!(
        report.prefix_cache_bytes,
        Some(head_bytes - 1),
        "the refused head's room is the cache's"
    );

    let x = 4096;
    let all = with_load_budget(target + draft + head_bytes + x, || {
        LlamaProvider::load(&spec)
    })
    .expect("all three fit");
    let report = all.load_report().unwrap();
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    assert!(report.draft.as_ref().unwrap().is_resident(), "{report:?}");
    assert_eq!(report.prefix_cache_bytes, Some(x));
    assert!(all
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_some());
}

#[test]
fn companion_head_bytes_are_counted_by_load_and_request_admission() {
    let fx = prism_load(None);
    let head: &Path = fx.head.as_ref();
    let head_bytes = crate::load_memory::companion_head_bytes(head).unwrap();
    let payload = core_llm::checkpoint_payload_bytes(head).unwrap();
    // Seven 1-D norm vectors (4 × 128 + 2 × 64 + 128 wide) at four intermediate bytes each,
    // and one 16 KiB page of rounding per stored tensor and per norm result (sc-24446).
    let norms = (5 * 128 + 2 * 64) * 4;
    let tensors = crate::load_memory::safetensors_headers(head)
        .unwrap()
        .iter()
        .flat_map(|h| h.as_object().unwrap().keys())
        .filter(|k| *k != "__metadata__")
        .count() as u64;
    assert_eq!(head_bytes, payload + norms + (tensors + 7) * 16 * 1024);

    let target =
        crate::load_memory::required_bytes(&LoadSpec::dense(fx.target.to_str().unwrap())).unwrap();
    let with_head =
        LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
    assert_eq!(
        crate::load_memory::required_bytes(&with_head).unwrap(),
        target,
        "the target's own price is unchanged; the head is admitted on top of it"
    );
    assert!(admit_companion_head(head, target, target + head_bytes).is_ok());
    let refused = admit_companion_head(head, target, target + head_bytes - 1).unwrap_err();
    assert!(
        refused.starts_with("mtp_head: refused by load admission"),
        "{refused}"
    );
    assert!(refused.contains(&head_bytes.to_string()), "{refused}");

    // Through the real load: a budget that fits the target but not target + head refuses the
    // head by name and loads the target without MTP; one more byte admits both.
    let short = with_load_budget(target + head_bytes - 1, || LlamaProvider::load(&with_head))
        .expect("the target fits and still loads (E2)");
    let fallbacks = short.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(
        fallbacks[0].starts_with("mtp_head: refused by load admission"),
        "{fallbacks:?}"
    );
    assert!(short
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_none());
    let fits = with_load_budget(target + head_bytes, || LlamaProvider::load(&with_head))
        .expect("target + head fit");
    // Nothing beside target + head is left for the prefix cache: it settles to 0 bytes,
    // named (E2), and the head attaches.
    let load = fits.load_report().unwrap();
    assert_eq!(load.prefix_cache_bytes, Some(0));
    assert_eq!(
        load.fallbacks,
        core_llm::prefix_budget_fallback(core_llm::DecodeBackend::Mlx, None, 0)
            .into_iter()
            .collect::<Vec<_>>()
    );
    assert!(fits
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .is_some());

    let provider = LlamaProvider::load(&with_head).unwrap();
    // The prefix-cache snapshot of an MTP request carries the head's KV too (sc-24437).
    let snapshot = |mtp| provider.prefix_snapshot_bytes(Some(64), mtp).unwrap();
    assert!(
        snapshot(true) > snapshot(false) + 4 * 128,
        "the head's KV is priced"
    );
    let price = |route| {
        provider
            .speculative_request_bytes(route, None, 64, 32, 0)
            .unwrap()
    };
    assert!(
        price(SpeculativeRoute::Mtp { width: 3 }) > price(SpeculativeRoute::Plain),
        "the head's cache and rollback are priced on the mtp route"
    );
}

/// E1/E8: the backend-neutral greedy parity suite (core-llm-testkit) — the one Candle runs —
/// holds on both MLX decoders: prompt lookup at depths 1, 3, the recommended 4 and the max,
/// `auto`, and on the Qwen3.8 head MTP at depths 1 and 3, each emitting exactly `off`'s
/// stream with a report naming its proposer; and something was actually drafted and accepted.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_backend_neutral_parity_suite_holds_on_both_decoders() {
    use core_llm::Speculative;
    use core_llm_testkit::{BenchPrompt, ParityCase, PromptClass};
    let prompts = vec![
        BenchPrompt::user(
            "repeat",
            PromptClass::Predictable,
            "t3 t9 t4 t11 t3 t9 t4 t11 t3 t9 t4 t11",
        ),
        BenchPrompt::user("sparse", PromptClass::OpenEnded, "t5 t8 t1 t20 t13"),
    ];
    let lookup = |depth| ParityCase {
        speculative: Speculative::proposer(SpeculativeProposer::PromptLookup, depth),
        expect_proposer: ProposerKind::PromptLookup,
    };
    let mtp = |depth| ParityCase {
        speculative: Speculative::proposer(SpeculativeProposer::Mtp, depth),
        expect_proposer: ProposerKind::Mtp,
    };
    let auto = |expect_proposer| ParityCase {
        speculative: Speculative::Auto,
        expect_proposer,
    };
    let depths = [
        1,
        3,
        MLX_ROW.recommended_depths.prompt_lookup,
        SPECULATIVE_MAX_DEPTH,
    ];
    for (label, provider, cases) in [
        (
            "causal",
            causal_provider(),
            depths
                .map(lookup)
                .into_iter()
                .chain([auto(ProposerKind::PromptLookup)])
                .collect::<Vec<_>>(),
        ),
        (
            "qwen35 mtp",
            qwen35_mtp_provider(),
            depths
                .map(lookup)
                .into_iter()
                .chain([mtp(1), mtp(3), auto(ProposerKind::Mtp)])
                .collect(),
        ),
    ] {
        let rows =
            core_llm_testkit::check_speculative_greedy_parity(&provider, &prompts, &cases, 20)
                .unwrap_or_else(|failures| panic!("{label}: {failures}"));
        assert_eq!(rows.len(), prompts.len() * cases.len(), "{label}");
        assert!(
            rows.iter().all(|r| r.report.fallbacks.is_empty()),
            "{label}: nothing fell back"
        );
        let drafted = |kind| {
            rows.iter()
                .filter(|r| r.report.proposer == kind)
                .any(|r| r.report.proposed_tokens > 0)
        };
        assert!(
            drafted(ProposerKind::PromptLookup),
            "{label}: lookup drafted"
        );
        assert!(
            rows.iter().any(|r| r.report.accepted_tokens > 0),
            "{label}: some draft was accepted"
        );
        if label == "qwen35 mtp" {
            assert!(drafted(ProposerKind::Mtp), "{label}: the head drafted");
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::the_backend_neutral_parity_suite_holds_on_both_decoders");
}

/// Epic AT1 at a production SDPA geometry: the shared fixture's hybrid target — 6 query heads
/// over 1 KV head at head dim 64, a GQA group of 6 — beside its hybrid draft advertises every
/// proposer at its geometry's depth (the vector kernel takes `32 / 6 = 5` query rows → 4
/// drafts, below the 8-row ceiling), and every proposer at depths 1, 3 and that max emits
/// `off`'s greedy stream. Every head-dim-64 attention call ran a fused MLX kernel, never the
/// fallback: the prefill on the full kernel and each verify on the vector kernel, a max-depth
/// verify (5 rows) in one call — the kernels `vector_verify_rows` / `speculative_max_depth`
/// are justified by, which the narrow-head-dim fixtures never reach.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn every_proposer_is_greedy_exact_on_the_vector_kernel_geometry() {
    use crate::primitives::attention::kernel_tally::{record, SdpaKernel};
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    let provider = LlamaProvider::load(
        &LoadSpec::dense(fixture.hybrid_target.display().to_string())
            .with_draft(fixture.hybrid_draft.display().to_string()),
    )
    .unwrap();
    let Decoder::Qwen35(model) = &provider.model else {
        panic!("the hybrid target loads as the Qwen35 decoder");
    };
    let geometry = qwen35_attention_geometry(model.config());
    assert_eq!(geometry, (6, 1, 64, 64));
    let caps = &provider.descriptor().capabilities;
    let cases = core_llm_testkit::advertised_parity_cases(caps);
    let prompts = core_llm_testkit::draft_model_prompts();
    let (rows, calls) = record(|| {
        core_llm_testkit::check_speculative_greedy_parity(&provider, &prompts, &cases, 24)
    });
    let rows = rows.unwrap_or_else(|failures| panic!("{failures}"));
    assert!(rows.iter().all(|r| r.report.fallbacks.is_empty()));
    for kind in [
        ProposerKind::Mtp,
        ProposerKind::PromptLookup,
        ProposerKind::DraftModel,
    ] {
        assert!(
            rows.iter()
                .any(|r| r.report.proposer == kind && r.report.proposed_tokens > 0),
            "{kind:?} drafted"
        );
    }

    // The kernels every head-dim-64 call — the target's and its MTP head's — reached.
    let wide: Vec<_> = calls.iter().filter(|c| c.head_dim == 64).collect();
    let fallback: Vec<_> = wide
        .iter()
        .filter(|c| c.kernel == SdpaKernel::Fallback)
        .collect();
    assert!(fallback.is_empty(), "fell back: {fallback:?}");
    assert!(
        wide.iter().any(|c| c.kernel == SdpaKernel::Full),
        "a prefill on the full kernel"
    );
    let deepest = rows
        .iter()
        .filter_map(|r| r.report.draft_tokens)
        .max()
        .unwrap();
    assert!(
        wide.iter()
            .any(|c| c.kernel == SdpaKernel::Vector && c.q_len == 1 + deepest as i32),
        "a max-depth ({deepest}) verify in one vector-kernel call"
    );

    // And that max is the geometry's: every proposer advertises it, the table covers it.
    let max = speculative_max_depth([geometry]);
    assert_eq!(max, 4, "the vector kernel's 5 rows, not the 8-row ceiling");
    assert_eq!(deepest, max);
    for proposer in SpeculativeProposer::ALL {
        assert_eq!(
            caps.proposer(proposer).unwrap().max_depth,
            max,
            "{proposer:?}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::every_proposer_is_greedy_exact_on_the_vector_kernel_geometry");
}

// ---- The cross-turn prefix cache (sc-24437). ----

fn chat(messages: Vec<Message>, speculative: core_llm::Speculative) -> TextLlmRequest {
    TextLlmRequest {
        messages,
        sampling: Sampling::greedy(),
        max_new_tokens: 8,
        seed: Some(3),
        speculative: Some(speculative),
        ..Default::default()
    }
}

fn rendered_ids(provider: &LlamaProvider, req: &TextLlmRequest, generation: bool) -> Vec<i32> {
    let text = provider
        .template
        .render_with(
            &req.messages,
            &RenderOptions {
                add_generation_prompt: generation,
                enable_thinking: req.enable_thinking_kwarg(),
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
                tools: &req.tools,
                date_string: provider.campaign_template_date,
            },
        )
        .unwrap();
    provider
        .tokenizer
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

fn shared(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// AC1 on the softmax family through the provider: turn 2 re-renders turn 1 and its reply, so
/// it restores the shared run of turn 1's cached `prompt + reply` (N), and decodes exactly
/// what a cold run (the non-engine plain loop, and a fresh provider) decodes.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_causal_second_turn_reports_its_prefix_hit_and_matches_a_cold_run() {
    use core_llm::Speculative;
    let provider = causal_provider();
    let user = Message::user("t3 t9 t4 t11 t3 t9");
    let turn1 = chat(vec![user.clone()], Speculative::Off);
    let (out1, ids1) = run(&provider, &turn1);
    assert_eq!(out1.decode.as_ref().unwrap().prefix_hit_tokens, 0, "cold");
    assert_eq!(out1.decode.as_ref().unwrap().prefix_cache.path, "miss");
    assert_eq!(provider.prefix_cache_stats().inserted, 1);

    let turn2 = chat(
        vec![
            user,
            Message::assistant(out1.text.clone()),
            Message::user("t5 t7"),
        ],
        Speculative::Off,
    );
    let p1 = rendered_ids(&provider, &turn1, true);
    let p2 = rendered_ids(&provider, &turn2, true);
    // Turn 1's cache holds its prompt and every reply token but the last (never fed).
    let mut held = p1.clone();
    held.extend(ids1.iter().map(|&t| t as i32));
    held.truncate(p1.len() + ids1.len() - 1);
    let n = shared(&held, &p2).min(p2.len() - 1);
    assert!(n >= p1.len() - 1, "turn 2 extends turn 1's prompt: {n}");

    let (out2, ids2) = run(&provider, &turn2);
    let report = out2.decode.unwrap();
    assert_eq!(report.prefix_hit_tokens, n as u64);
    assert_eq!(
        (
            report.prefix_cache.path.as_str(),
            report.prefix_cache.reason.as_deref()
        ),
        ("hit", None)
    );
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    assert_eq!(ids2, plain_loop(&provider, &turn2));
    assert_eq!(ids2, run(&causal_provider(), &turn2).1);
    assert!(provider.prefix_cache_resident_bytes() <= provider.prefix_cache_budget());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_causal_second_turn_reports_its_prefix_hit_and_matches_a_cold_run");
}

/// AC2 through the provider: the hybrid snapshots at the end of turn 1's rendered
/// conversation; turn 2 (and turn 3, re-using the same entry after turn 2 decoded past it)
/// restores the recurrent state there and matches a cold run — plain and with the MTP head
/// resuming its warm-up from the stored boundary.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_qwen35_second_turn_restores_the_boundary_state_and_matches_a_cold_run() {
    use core_llm::Speculative;
    for speculative in [
        Speculative::Off,
        Speculative::proposer(SpeculativeProposer::Mtp, 3),
    ] {
        let provider = qwen35_mtp_provider();
        let user = Message::user("t3 t9 t4 t11 t3 t9 t4 t11");
        let turn1 = chat(vec![user.clone()], speculative);
        let (out1, _) = run(&provider, &turn1);
        assert_eq!(out1.decode.as_ref().unwrap().prefix_hit_tokens, 0);
        let boundary = rendered_ids(&provider, &turn1, false).len();
        assert_eq!(provider.prefix.borrow().keys().len(), 1);
        assert_eq!(provider.prefix.borrow().keys()[0].len(), boundary);
        // The admitted estimate bounds what the snapshot holds (E7).
        let estimate = provider
            .prefix_snapshot_bytes(Some(boundary), speculative != Speculative::Off)
            .unwrap();
        let held = provider.prefix_cache_resident_bytes();
        assert!(held > 0 && held <= estimate, "{held} > {estimate}");

        for next in ["t5 t7", "t8 t2 t6"] {
            let turn = chat(
                vec![
                    user.clone(),
                    Message::assistant(out1.text.clone()),
                    Message::user(next),
                ],
                speculative,
            );
            let (out, ids) = run(&provider, &turn);
            let report = out.decode.unwrap();
            assert_eq!(
                report.prefix_hit_tokens, boundary as u64,
                "{speculative:?}: restored at the boundary"
            );
            assert_eq!(report.prefix_cache.path, "hit");
            assert_eq!(report.proposer, speculative_kind(speculative));
            assert_eq!(ids, plain_loop(&provider, &turn), "{speculative:?}");
            assert_eq!(ids, run(&qwen35_mtp_provider(), &turn).1, "{speculative:?}");
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_qwen35_second_turn_restores_the_boundary_state_and_matches_a_cold_run");
}

fn speculative_kind(speculative: core_llm::Speculative) -> ProposerKind {
    match speculative {
        core_llm::Speculative::Proposer { proposer, .. } => proposer.into(),
        _ => ProposerKind::None,
    }
}

/// A tiny llama snapshot on disk for the load-admission tests.
fn tiny_prefix_snapshot() -> crate::test_fixture::Fixture {
    let dir = crate::test_fixture::Fixture::new("mlx-llm-prefix-", None);
    std::fs::write(
        dir.join("config.json"),
        r#"{"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 16,
                "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
                "eos_token_id": 999}"#,
    )
    .unwrap();
    let vocab: Vec<String> = (0..16).map(|i| format!("\"t{i}\": {i}")).collect();
    std::fs::write(
        dir.join("tokenizer.json"),
        format!(
            r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                    "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                    "decoder": null,
                    "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
            vocab.join(", ")
        ),
    )
    .unwrap();
    let w = |shape: &[i32]| Array::ones::<f32>(shape).unwrap();
    let mut arrays = vec![
        ("model.embed_tokens.weight".to_string(), w(&[16, 8])),
        ("model.norm.weight".to_string(), w(&[8])),
        ("lm_head.weight".to_string(), w(&[16, 8])),
    ];
    for (name, shape) in [
        ("input_layernorm.weight", vec![8]),
        ("post_attention_layernorm.weight", vec![8]),
        ("self_attn.q_proj.weight", vec![8, 8]),
        ("self_attn.k_proj.weight", vec![4, 8]),
        ("self_attn.v_proj.weight", vec![4, 8]),
        ("self_attn.o_proj.weight", vec![8, 8]),
        ("mlp.gate_proj.weight", vec![16, 8]),
        ("mlp.up_proj.weight", vec![16, 8]),
        ("mlp.down_proj.weight", vec![8, 16]),
    ] {
        arrays.push((format!("model.layers.0.{name}"), w(&shape)));
    }
    let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, a)| (k.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
    dir
}

/// Sets the operational memory override for one scope (the crate's tests run on one thread,
/// `.cargo/config.toml`, so no other test observes it).
struct MemoryOverride;

impl MemoryOverride {
    fn set(bytes: u64) -> Self {
        std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, bytes.to_string());
        Self
    }
}

impl Drop for MemoryOverride {
    fn drop(&mut self) {
        std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
    }
}

/// Tiny synthetic Llama whose head dimension the packed Metal reader supports.
fn tiny_packed_capable_model() -> CausalLm {
    tiny_causal_model(2, 1, 64)
}

/// Tiny synthetic two-layer Llama (hidden 128) with the given attention geometry.
pub(crate) fn tiny_causal_model(heads: i32, kv_heads: i32, head_dim: i32) -> CausalLm {
    synthetic_causal_model(128, 2, heads, kv_heads, head_dim)
}

/// A synthetic Llama of `layers` layers and `hidden` width (vocabulary 32) with the given
/// attention geometry; random weights from a fixed seed.
pub(crate) fn synthetic_causal_model(
    hidden: i32,
    layers: usize,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
) -> CausalLm {
    synthetic_causal_model_with_rope(hidden, layers, heads, kv_heads, head_dim, 10000.0)
}

/// [`synthetic_causal_model`] with an explicit RoPE base (same weights for every base).
pub(crate) fn synthetic_causal_model_with_rope(
    hidden: i32,
    layers: usize,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
) -> CausalLm {
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    let cfg = crate::config::ModelConfig {
        hidden_size: hidden,
        intermediate_size: 64,
        num_layers: layers,
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        rope_theta,
        rope_scaling: None,
        tie_word_embeddings: false,
        architecture: crate::config::Architecture::Llama,
        max_position_embeddings: 0,
        quantization: None,
        moe: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        query_pre_attn_scalar: None,
        partial_rotary_factor: 1.0,
        mla: None,
        yarn: None,
        mrope_section: None,
        gemma4: None,
        activation_role: Default::default(),
    };
    let mut rng = SplitMix64::new(0x5c20676);
    let mut randn = |shape: &[i32]| {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Array::from_slice(&data, shape)
    };
    let (h, v, inter) = (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size);
    let (qd, kvd) = (
        cfg.num_heads * cfg.head_dim,
        cfg.num_kv_heads * cfg.head_dim,
    );
    let mut m = std::collections::HashMap::new();
    m.insert("model.embed_tokens.weight".to_string(), randn(&[v, h]));
    m.insert(
        "model.norm.weight".into(),
        Array::ones::<f32>(&[h]).unwrap(),
    );
    m.insert("lm_head.weight".into(), randn(&[v, h]));
    for i in 0..cfg.num_layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(
            p("input_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        );
        m.insert(
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        );
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h]));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h]));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h]));
        m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd]));
        m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h]));
        m.insert(p("mlp.up_proj.weight"), randn(&[inter, h]));
        m.insert(p("mlp.down_proj.weight"), randn(&[h, inter]));
    }
    CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap()
}

/// sc-20671: admission prices K/V, activations and logits at the width the decoder actually
/// caches (its compute dtype), and only the eager score term at F32. The Prism/Bonsai path is
/// held at F32 until its frozen allocator peaks are re-taken.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn memory_geometry_prices_kv_at_the_cached_width() {
    let decoder = Decoder::Causal(Box::new(tiny_packed_capable_model()));
    let geometry = decoder.memory_geometry();
    let Decoder::Causal(model) = &decoder else {
        unreachable!()
    };
    let mut cache = model.new_cache();
    model
        .decode_logits(&input_ids(&[1, 2, 3]), &mut cache, 0)
        .unwrap();
    assert_eq!(cache.element_bytes().unwrap(), Some(geometry.element_bytes));
    assert_eq!(geometry.element_bytes, dtype_bytes(model.compute_dtype()));
    assert_eq!(
        geometry.score_element_bytes,
        dtype_bytes(Dtype::Float32),
        "sdpa_eager scores stay F32"
    );
    assert_eq!(
        priced_compute_element_bytes(Dtype::Bfloat16, true),
        dtype_bytes(Dtype::Float32),
        "Prism admission holds the pre-fix width until its peaks are re-taken"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::memory_geometry_prices_kv_at_the_cached_width");
}

#[derive(Default)]
struct CompressedArmCapture {
    snapshots: Vec<u64>,
    releases: Vec<u64>,
    reconstructions: usize,
    evidence: Vec<crate::primitives::PackedCacheEvidence>,
    fallbacks: Vec<(String, String)>,
    storage_tokens: Vec<u64>,
}

impl crate::decode::observe::Observer for CompressedArmCapture {
    fn phase(&mut self, _name: &'static str) {}
    fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
    fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, _element: u64) {
        self.snapshots.push(bytes);
    }
    fn release_event(&mut self, _kind: &'static str, _role: &'static str, bytes: u64) {
        self.releases.push(bytes);
    }
    fn dense_reconstruction(&mut self, _bytes: u64) {
        self.reconstructions += 1;
    }
    fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
        self.evidence.push(evidence.clone());
    }
    fn dense_fallback(&mut self, operation: &str, reason: &str) {
        self.fallbacks.push((operation.into(), reason.into()));
    }
    fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
        self.storage_tokens.push(storage.tokens);
    }
}

/// A compressed prefix hit imports the reused dense prefix into the arm's compressed cache by
/// quantize-on-append: the suffix prefill and decode run on the fused reader over the imported
/// history, nothing is reconstructed or recorded as a fallback, and the compressed cache never
/// re-enters the dense prefix store. A refused compressed selection keeps the dense seed and
/// records its reason instead.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn compressed_prefix_hit_imports_the_reused_prefix_into_the_compressed_cache() {
    let model = tiny_packed_capable_model();
    // Longer than one 32-token group, so the import packs a K group and keeps a residual. The
    // ids stay inside the 32-token vocabulary: MLX's embedding gather is not bounds-checked,
    // so an out-of-vocabulary id reads whatever the allocator last left past the table and
    // the logits then depend on which test ran before.
    let prompt = (0..40).map(|i| i % 31 + 1).collect::<Vec<i32>>();
    let vocab = model.config().vocab_size;
    assert!(prompt.iter().all(|id| (0..vocab).contains(id)));
    let config = GenerationConfig {
        max_new_tokens: 2,
        seed: Some(0),
        ..Default::default()
    };
    let cancel = CancelFlag::new();
    let seeded_store = || {
        let mut store = crate::decode::PrefixCache::with_budget(u64::MAX);
        crate::decode::generate_cached(&model, &prompt, &config, &cancel, &mut |_| {}, &mut store)
            .unwrap();
        assert_eq!(store.len(), 1);
        store
    };

    let arm = crate::decode::observe::CompressedKvMethod::GroupAffine
        .arm()
        .unwrap();
    let mut store = seeded_store();
    let mut capture = CompressedArmCapture::default();
    let output = crate::decode::prefix::generate_cached_with_observer(
        &model,
        &prompt,
        &config,
        &cancel,
        &mut |_| {},
        &mut store,
        None,
        None,
        Some(&mut capture),
        Some(&arm),
    )
    .unwrap();
    assert_eq!(output.tokens.len(), 2);
    assert_eq!(store.stats().hits, 1);
    assert!(capture.fallbacks.is_empty(), "{:?}", capture.fallbacks);
    let [evidence] = capture.evidence.as_slice() else {
        panic!("the compressed cache must export evidence once");
    };
    assert!(evidence.accepted_direct_calls > 0);
    assert!(evidence.fallback_reasons.is_empty() && !evidence.dense_active);
    assert_eq!(evidence.full_cache_dequantizations, 0);
    assert_eq!(capture.reconstructions, 0);
    // 39 imported prefix tokens (a whole-prompt match recomputes the last) plus the suffix.
    assert_eq!(capture.storage_tokens.first(), Some(&40));
    assert_eq!(store.len(), 1, "the compressed cache is not stored back");

    let refused = crate::decode::observe::CompressedKvArm::with_reader(
        crate::decode::observe::CompressedKvMethod::GroupAffine,
        crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
            crate::primitives::OpaqueCompiledKernel::new(
                "sc20676-prefix-refused",
                "cpu",
                0,
                std::sync::Arc::new(()),
            ),
        )),
    );
    let mut store = seeded_store();
    let mut capture = CompressedArmCapture::default();
    crate::decode::prefix::generate_cached_with_observer(
        &model,
        &prompt,
        &config,
        &cancel,
        &mut |_| {},
        &mut store,
        None,
        None,
        Some(&mut capture),
        Some(&refused),
    )
    .unwrap();
    assert!(
        matches!(capture.fallbacks.as_slice(), [(operation, _)] if operation == "cache-selection"),
        "{:?}",
        capture.fallbacks
    );
    assert!(
        capture.evidence.is_empty(),
        "the dense seed carries the request"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::compressed_prefix_hit_imports_the_reused_prefix_into_the_compressed_cache");
}

/// Prefill logits, packed evidence and fallbacks of one observed generation.
#[derive(Default)]
struct PrefillLogitsCapture {
    prefill: Vec<f32>,
    evidence: Vec<crate::primitives::PackedCacheEvidence>,
    fallbacks: Vec<(String, String)>,
}

impl crate::decode::observe::Observer for PrefillLogitsCapture {
    fn phase(&mut self, _name: &'static str) {}
    fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
    fn logits(&mut self, stage: &'static str, values: &[f32]) {
        if stage == "prefill" {
            self.prefill = values.to_vec();
        }
    }
    fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
        self.evidence.push(evidence.clone());
    }
    fn dense_fallback(&mut self, operation: &str, reason: &str) {
        self.fallbacks.push((operation.into(), reason.into()));
    }
}

/// Multi-turn prompt-cache reuse on the 8-bit compressed arm agrees with dense (SC-20671
/// `multiTurnPromptCache` diagnosis). Turn 2 extends turn 1's prompt and answer; the dense
/// store holds turn 1. The dense hit, the compressed hit (prefix imported by quantize-on-append
/// at the reused offset) and a cold compressed turn 2 must choose the same tokens, with turn-2
/// prefill logits within 8-bit rounding of the dense hit. A misplaced, dropped or re-quantized
/// imported prefix moves the compressed logits by far more than that.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn compressed_8bit_multi_turn_prompt_cache_reuse_agrees_with_dense() {
    let model = tiny_packed_capable_model();
    let vocab = model.config().vocab_size;
    // Turn 1 spans two 32-token groups plus a residual; ids stay inside the vocabulary.
    let turn1 = (0..70).map(|i| (i * 7) % 31 + 1).collect::<Vec<i32>>();
    let config = GenerationConfig {
        max_new_tokens: 6,
        seed: Some(0),
        ..Default::default()
    };
    let cancel = CancelFlag::new();
    let arm = crate::decode::observe::CompressedKvMethod::GroupAffine8
        .arm()
        .unwrap();
    let turn1_store = || {
        let mut store = crate::decode::PrefixCache::with_budget(u64::MAX);
        let out = crate::decode::generate_cached(
            &model,
            &turn1,
            &config,
            &cancel,
            &mut |_| {},
            &mut store,
        )
        .unwrap();
        (store, out.tokens)
    };
    let (_, answer) = turn1_store();
    let mut turn2 = turn1.clone();
    turn2.extend_from_slice(&answer);
    turn2.extend((0..9).map(|i| (i * 5) % 31 + 1));
    assert!(turn2.iter().all(|id| (0..vocab).contains(id)));

    let turn2_run =
        |store: &mut crate::decode::PrefixCache,
         compressed: Option<&crate::decode::observe::CompressedKvArm>| {
            let mut capture = PrefillLogitsCapture::default();
            let out = crate::decode::prefix::generate_cached_with_observer(
                &model,
                &turn2,
                &config,
                &cancel,
                &mut |_| {},
                store,
                None,
                None,
                Some(&mut capture),
                compressed,
            )
            .unwrap();
            (out.tokens, capture)
        };
    let (mut dense_store, _) = turn1_store();
    let (dense_tokens, dense) = turn2_run(&mut dense_store, None);
    let (mut hit_store, _) = turn1_store();
    let (hit_tokens, hit) = turn2_run(&mut hit_store, Some(&arm));
    let mut cold_store = crate::decode::PrefixCache::with_budget(u64::MAX);
    let (cold_tokens, cold) = turn2_run(&mut cold_store, Some(&arm));

    assert_eq!(dense_store.stats().hits, 1);
    assert_eq!(
        hit_store.stats().hits,
        1,
        "turn 2 must reuse turn 1's prefix"
    );
    assert_eq!(cold_store.stats().hits, 0);
    for (label, capture) in [("hit", &hit), ("cold", &cold)] {
        assert!(
            capture.fallbacks.is_empty(),
            "{label}: {:?}",
            capture.fallbacks
        );
        assert!(
            !capture.evidence.is_empty()
                && capture
                    .evidence
                    .iter()
                    .all(|e| e.accepted_direct_calls > 0 && !e.dense_active),
            "{label} must decode on the fused reader"
        );
    }
    let max_abs = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let scale = dense.prefill.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(scale > 0.0);
    let hit_error = max_abs(&hit.prefill, &dense.prefill) / scale;
    let cold_error = max_abs(&cold.prefill, &dense.prefill) / scale;
    eprintln!("mtpc: 8-bit turn-2 prefill relative max error hit {hit_error} cold {cold_error}");
    assert!(
        hit_error < 0.02 && cold_error < 0.02,
        "8-bit turn-2 prefill logits drifted from dense: hit {hit_error}, cold {cold_error}"
    );
    assert_eq!(hit_tokens, dense_tokens, "compressed reuse vs dense turn 2");
    assert_eq!(cold_tokens, dense_tokens, "compressed cold vs dense turn 2");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::compressed_8bit_multi_turn_prompt_cache_reuse_agrees_with_dense");
}

/// A tiny packed-capable on-disk snapshot (head dim 64, 32-word vocabulary, a 2048-token
/// window, no reachable stop token) the provider loads like a real one.
fn tiny_packed_capable_snapshot() -> tempfile::TempDir {
    tiny_snapshot(json!({"architectures": ["LlamaForCausalLM"]}), 2048, false)
}

/// [`tiny_packed_capable_snapshot`] with the config's architecture `identity` keys, a
/// `window`-token context, and (for Qwen3) unit per-head q/k RMSNorm weights.
pub(crate) fn tiny_snapshot(
    identity: serde_json::Value,
    window: u64,
    qk_norm: bool,
) -> tempfile::TempDir {
    tiny_snapshot_with(identity, window, qk_norm, SnapshotGains::DEFAULT)
}

/// Weight gains of a [`tiny_snapshot_with`] fixture over its unit-scale random draws.
#[derive(Clone, Copy, Debug)]
struct SnapshotGains {
    /// Scale of the query/key projections (of the q/k RMSNorm weights on a Qwen3 fixture):
    /// sharper attention.
    qk: f32,
    /// Scale of the value/output projections: attention's share of the residual.
    vo: f32,
    /// Weight of the embeddings in the output head: how strongly the current token decides
    /// the next one.
    head_align: f32,
}

impl SnapshotGains {
    const DEFAULT: Self = Self {
        qk: 1.0,
        vo: 1.0,
        head_align: 4.0,
    };
}

/// [`tiny_snapshot`] with explicit weight `gains`; the random draws are the same for every
/// gain, so [`SnapshotGains::DEFAULT`] reproduces [`tiny_snapshot`] exactly.
fn tiny_snapshot_with(
    identity: serde_json::Value,
    window: u64,
    qk_norm: bool,
    gains: SnapshotGains,
) -> tempfile::TempDir {
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    let dir = tempfile::tempdir().unwrap();
    let mut vocab = serde_json::Map::new();
    for (id, word) in ["<unk>", "<|", "|>", "user", "assistant", "eot_id"]
        .into_iter()
        .map(String::from)
        .chain((6..32).map(|id| format!("w{id}")))
        .enumerate()
    {
        vocab.insert(word, json!(id));
    }
    let tokenizer = json!({
        "version": "1.0", "added_tokens": [], "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" }, "post_processor": null, "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" },
    });
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let mut config = json!({
        "hidden_size": 128,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 64, "vocab_size": 32, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "tie_word_embeddings": false,
        "max_position_embeddings": window, "eos_token_id": 99,
    });
    for (key, value) in identity.as_object().unwrap() {
        config[key] = value.clone();
    }
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let mut rng = SplitMix64::new(0x5c20671);
    let mut randn = |shape: &[i32]| {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Array::from_slice(&data, shape)
    };
    let mut arrays = vec![
        (
            "model.embed_tokens.weight".to_string(),
            randn(&[32, 128]) * 4.0f32,
        ),
        (
            "model.norm.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        ),
    ];
    // A head partly aligned with the embeddings gives decisive next-token margins (a random
    // head over 32 words leaves bf16-rounding ties that flip even dense against dense), while
    // the attention path still moves the argmax (2-bit K/V flips positions).
    let head = &arrays[0].1 * gains.head_align + randn(&[32, 128]);
    arrays.push(("lm_head.weight".into(), head));
    // The identity may deepen the decoder (`num_hidden_layers`) and widen its attention
    // (`num_attention_heads`, `num_key_value_heads`, `head_dim`); every layer gets weights of
    // that geometry.
    let dim = |key: &str| i32::try_from(config[key].as_u64().unwrap()).unwrap();
    let (q_width, kv_width, head) = (
        dim("num_attention_heads") * dim("head_dim"),
        dim("num_key_value_heads") * dim("head_dim"),
        dim("head_dim"),
    );
    for i in 0..config["num_hidden_layers"].as_u64().unwrap() {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        arrays.extend([
            (
                p("input_layernorm.weight"),
                Array::ones::<f32>(&[128]).unwrap(),
            ),
            (
                p("post_attention_layernorm.weight"),
                Array::ones::<f32>(&[128]).unwrap(),
            ),
            (
                p("self_attn.q_proj.weight"),
                randn(&[q_width, 128]) * gains.qk,
            ),
            (
                p("self_attn.k_proj.weight"),
                randn(&[kv_width, 128]) * gains.qk,
            ),
            (
                p("self_attn.v_proj.weight"),
                randn(&[kv_width, 128]) * gains.vo,
            ),
            (
                p("self_attn.o_proj.weight"),
                randn(&[128, q_width]) * gains.vo,
            ),
            (p("mlp.gate_proj.weight"), randn(&[64, 128])),
            (p("mlp.up_proj.weight"), randn(&[64, 128])),
            (p("mlp.down_proj.weight"), randn(&[128, 64])),
        ]);
        if qk_norm {
            arrays.extend([
                (
                    p("self_attn.q_norm.weight"),
                    Array::ones::<f32>(&[head]).unwrap() * gains.qk,
                ),
                (
                    p("self_attn.k_norm.weight"),
                    Array::ones::<f32>(&[head]).unwrap() * gains.qk,
                ),
            ]);
        }
    }
    let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, a)| (k.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();
    dir
}

/// Campaign prompts must not depend on the day a row runs. The pinned Llama 3.2 template dates
/// its system header (`Today Date:`) from `strftime_now`; under two mocked "now" days the
/// campaign provider renders the multi-turn fixture's turn 2 to identical ids, dated with the
/// campaign constant, while a production load keeps the template's own clock dating.
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn campaign_multi_turn_prompt_ids_do_not_depend_on_the_date() {
    let snapshot = tiny_packed_capable_snapshot();
    let pinned: serde_json::Value = serde_json::from_str(include_str!(
        "../models/testdata/llama32_tokenizer_config_template.json"
    ))
    .unwrap();
    std::fs::write(
        snapshot.path().join("tokenizer_config.json"),
        pinned.to_string(),
    )
    .unwrap();
    // The tiny vocabulary plus the date header's words, so a changed date changes the ids.
    let tokenizer_path = snapshot.path().join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_path).unwrap()).unwrap();
    let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    for word in [
        "Today", "Date", "03", "04", "26", "Jul", "Oct", "2024", "2026",
    ] {
        let id = vocab.len();
        vocab.insert(word.into(), json!(id));
    }
    std::fs::write(&tokenizer_path, tokenizer.to_string()).unwrap();
    let spec = core_llm::LoadSpec::dense(snapshot.path().to_string_lossy().to_string());
    let campaign = LlamaProvider::load_for_campaign(&spec).unwrap();
    let production = LlamaProvider::load(&spec).unwrap();
    let turn2 = TextLlmRequest {
        messages: vec![
            Message::text(Role::User, "w6 w7 w8 Repeat the stable baseline fact."),
            Message::text(Role::Assistant, "w9 w10 w11"),
            Message::text(Role::User, crate::campaign::MULTI_TURN_FIXTURE_FOLLOW_UP),
        ],
        ..Default::default()
    };
    // 2026-10-03 and 2026-10-04, UTC: a row run either side of midnight.
    let (day_one, day_two) = (20_729 * 86_400, 20_730 * 86_400);
    let render = |provider: &LlamaProvider, now| {
        core_llm::template::with_template_clock(now, || {
            provider.render_prompt(&turn2, &turn2.messages).unwrap()
        })
    };
    let (text_one, ids_one) = render(&campaign, day_one);
    let (text_two, ids_two) = render(&campaign, day_two);
    assert_eq!(ids_one, ids_two, "campaign prompt ids moved with the date");
    assert_eq!(text_one, text_two);
    let header = format!(
        "Today Date: {}",
        crate::campaign::CAMPAIGN_CHAT_TEMPLATE_DATE_STRING
    );
    assert!(text_one.contains(&header), "{text_one}");
    let (production_one, production_ids_one) = render(&production, day_one);
    let (production_two, production_ids_two) = render(&production, day_two);
    assert!(
        production_one.contains("Today Date: 03 Oct 2026"),
        "{production_one}"
    );
    assert!(
        production_two.contains("Today Date: 04 Oct 2026"),
        "{production_two}"
    );
    assert_ne!(production_ids_one, production_ids_two);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::campaign_multi_turn_prompt_ids_do_not_depend_on_the_date");
}

/// SC-20671 contract v4 multi-turn prompt-cache fixture on the product provider path, at
/// 8-bit. Turn 1 runs and stores its K/V; turn 2 (turn 1's conversation, answer and a
/// follow-up) is served by a prompt-cache hit in both arms — the compressed arm importing the
/// prefix — and the compressed arm, teacher-forced on the dense turn-2 forced continuation,
/// agrees at >= 0.999 of 1024 positions. A turn 2 the cache did not serve is refused.
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn multi_turn_prompt_cache_fixture_agrees_at_8_bit_and_requires_a_turn_two_hit() {
    let snapshot = tiny_packed_capable_snapshot();
    let provider = LlamaProvider::load_for_campaign(&core_llm::LoadSpec::dense(
        snapshot.path().to_string_lossy().to_string(),
    ))
    .unwrap();
    let words = (0..70)
        .map(|i| format!("w{}", (i * 7) % 26 + 6))
        .collect::<Vec<_>>()
        .join(" ");
    let turn1 = TextLlmRequest {
        messages: vec![Message::text(Role::User, words)],
        sampling: core_llm::Sampling::greedy(),
        max_new_tokens: 8,
        seed: Some(0),
        ..Default::default()
    };
    let follow_up = "w7 w9 w11 w13 w15 w17 w19 w21 w23";
    let arm = crate::decode::observe::CompressedKvMethod::GroupAffine8
        .arm()
        .unwrap();

    let (reference, dense_turns) = provider
        .campaign_multi_turn_forced_continuation(&turn1, follow_up, 1024, None, None)
        .unwrap();
    assert_eq!(reference.len(), 1024, "every stop token is decoded through");
    // SC-20669 dense multi-turn noise-floor control: the same flow scored. The dense reference
    // reproduces the unscored stream, and a dense session teacher-forced on it through the
    // same cache-hit path scores every turn-2 position.
    let (scored, scored_turns) = provider
        .campaign_multi_turn_scored_continuation(&turn1, follow_up, 1024, None)
        .unwrap();
    assert_eq!(scored.choices, reference);
    assert_eq!(scored.stream_probabilities.len(), 1024);
    assert_eq!(scored_turns, dense_turns);
    let (forced, forced_turns) = provider
        .campaign_multi_turn_scored_continuation(&turn1, follow_up, 1024, Some(&reference))
        .unwrap();
    assert_eq!(forced.choices, reference);
    assert_eq!(forced.stream_probabilities, scored.stream_probabilities);
    assert_eq!(forced_turns, dense_turns);
    let (choices, compressed_turns) = provider
        .campaign_multi_turn_forced_continuation(
            &turn1,
            follow_up,
            1024,
            Some(&arm),
            Some(&reference),
        )
        .unwrap();
    dense_turns.validate("dense").unwrap();
    assert!(
        dense_turns.turn2.reused_prefix_tokens >= dense_turns.turn1.prompt_tokens,
        "turn 2 reuses all of turn 1's prompt: {dense_turns:?}"
    );
    assert_eq!(compressed_turns, dense_turns, "the same flow in both arms");
    let evidence =
        crate::campaign::multi_turn_forced_continuation_evidence(&reference, &choices).unwrap();
    eprintln!(
        "mtpc v4: 8-bit turn-2 teacher-forced agreement {} ({}/{}; flips {:?})",
        evidence.agreement, evidence.matches, evidence.tokens, evidence.first_flip_positions
    );
    assert!(
        evidence.agreement >= crate::campaign::COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN,
        "{evidence:?}"
    );

    // The observed fixture: both arms' turn 2 is the same prompt-cache hit.
    let observed = |packed: Option<&crate::decode::observe::CompressedKvArm>| {
        let mut capture = CompressedArmCapture::default();
        let (output, turns) = provider
            .campaign_multi_turn_observed(&turn1, follow_up, &mut |_| {}, &mut capture, packed)
            .unwrap();
        assert!(capture.fallbacks.is_empty(), "{:?}", capture.fallbacks);
        (output.text, turns, capture.evidence)
    };
    let (dense_text, dense_observed, dense_evidence) = observed(None);
    let (compressed_text, compressed_observed, compressed_evidence) = observed(Some(&arm));
    assert_eq!(dense_observed, dense_turns);
    assert_eq!(compressed_observed, dense_turns);
    assert!(dense_evidence.is_empty(), "the dense arm stays dense");
    assert!(
        !compressed_evidence.is_empty()
            && compressed_evidence
                .iter()
                .all(|e| e.accepted_direct_calls > 0 && !e.dense_active),
        "turn 2 decodes on the fused reader over the imported prefix"
    );
    assert!(!dense_text.is_empty() && !compressed_text.is_empty());

    // The 2048-token window cannot hold turn 2 of a ~1100-token turn 1 plus the 1024-token
    // continuation: refused before decoding, never shortened.
    let long_turn1 = TextLlmRequest {
        messages: vec![Message::text(
            Role::User,
            (0..1_100)
                .map(|i| format!("w{}", (i * 5) % 26 + 6))
                .collect::<Vec<_>>()
                .join(" "),
        )],
        ..turn1.clone()
    };
    let error = provider
        .campaign_multi_turn_forced_continuation(&long_turn1, follow_up, 1024, None, None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("fewer than the 1024-token forced continuation"),
        "{error}"
    );

    // A turn 2 the cache did not serve is refused.
    let mut store = crate::decode::PrefixCache::with_budget(u64::MAX);
    let (answer, _, turn1_ids) = provider.campaign_turn_one(&turn1, &mut store).unwrap();
    let turn2 = LlamaProvider::campaign_turn_two_request(&turn1, &answer, follow_up);
    let (_, turn2_ids) = provider.render_prompt(&turn2, &turn2.messages).unwrap();
    let missed = crate::campaign::PromptCacheTurn {
        prompt_tokens: turn2_ids.len() as u64,
        prompt_sha256: crate::campaign::token_stream_sha256(&turn2_ids),
        cache_hit: false,
        reused_prefix_tokens: 0,
    };
    let error = LlamaProvider::require_turn_two_hit(&missed, &turn1_ids, &turn2_ids)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("not served by a prompt-cache hit"),
        "{error}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::multi_turn_prompt_cache_fixture_agrees_at_8_bit_and_requires_a_turn_two_hit");
}

/// A packed cache whose retained reader fails to bind still records the selection's own
/// reason rather than only the later generic dense-attention fallback.
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
fn compressed_campaign_decoder_records_a_refused_reader_binding() {
    let model = tiny_packed_capable_model();
    let reader = crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
        crate::primitives::OpaqueCompiledKernel::new(
            "sc20676rx-refused",
            "cpu",
            0,
            std::sync::Arc::new(()),
        ),
    ));
    let arm = crate::decode::observe::CompressedKvArm::with_reader(
        crate::decode::observe::CompressedKvMethod::GroupAffine,
        reader,
    );
    let decoder = PackedCampaignDecoder {
        model: &model,
        arm: &arm,
        selection_fallbacks: RefCell::new(Vec::new()),
    };
    let cache = decoder.make_cache();
    assert!(
        cache.packed_evidence().is_some(),
        "a refused binding still yields the packed cache"
    );
    let fallbacks = decoder.selection_fallbacks.into_inner();
    assert!(
        matches!(fallbacks.as_slice(), [reason] if reason.contains("packed reader rejected before mutation")),
        "{fallbacks:?}"
    );
}

/// The compressed campaign decoder runs the whole observed generation (prefill and decode) on
/// the packed cache through the fused reader: no fallback, no dense reconstruction, and the
/// persistent KV it reports is the packed storage it later releases.
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn compressed_campaign_decoder_keeps_prefill_and_decode_on_the_fused_reader() {
    let model = tiny_packed_capable_model();
    let arm = crate::decode::observe::CompressedKvMethod::GroupAffine
        .arm()
        .unwrap();
    let decoder = PackedCampaignDecoder {
        model: &model,
        arm: &arm,
        selection_fallbacks: RefCell::new(Vec::new()),
    };
    let mut capture = CompressedArmCapture::default();
    let config = GenerationConfig {
        max_new_tokens: 4,
        seed: Some(0),
        ..Default::default()
    };
    let output = crate::decode::generate_with_observer(
        &decoder,
        &[1, 2, 3, 4, 5],
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
        None,
        Some(&mut capture),
    )
    .unwrap();
    assert_eq!(output.tokens.len(), 4);
    assert!(decoder.selection_fallbacks.borrow().is_empty());
    let [evidence] = capture.evidence.as_slice() else {
        panic!("one compressed cache must export evidence once");
    };
    // Two layers: one fused prefill call plus three fused decode steps each.
    assert_eq!(evidence.accepted_direct_calls, 2 * 4);
    assert!(evidence.fallback_reasons.is_empty() && !evidence.dense_active);
    assert_eq!(evidence.full_cache_dequantizations, 0);
    assert_eq!(capture.reconstructions, 0);
    assert!(!capture.snapshots.is_empty());
    assert_eq!(capture.releases, vec![*capture.snapshots.last().unwrap()]);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::compressed_campaign_decoder_keeps_prefill_and_decode_on_the_fused_reader");
}

/// SC-20671 prefill attribution through the real decode loop, at 2- and 4-bit. An empty
/// cache's first multi-row step attends on dense SDPA, so no output reads the packed store;
/// the cache must still materialize the store it reports at the step's commit. Left lazy, the
/// prefill-peak sample holds the step's dense K/V instead of the store, and the campaign's
/// `baseline + persistent KV` floor fails whenever that dense K/V is smaller than the
/// block-allocated store (Mac2 A2 qwen-short 4-bit: 80 prompt tokens, active growth
/// 9,504,184 B for a 12,845,056 B store; 2-bit passed only because its 9,175,040 B store
/// equals the 80-token dense K/V).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn compressed_prefill_sample_holds_the_reported_packed_store() {
    #[derive(Default)]
    struct PrefillCapture {
        phase: Option<&'static str>,
        prefill_active: Option<u64>,
        prefill_kv: u64,
        element_bytes: u64,
    }
    impl crate::decode::observe::Observer for PrefillCapture {
        fn phase(&mut self, name: &'static str) {
            self.phase = Some(name);
            if name == "prefill-peak" {
                self.prefill_active = Some(mlx_rs::memory::get_active_memory() as u64);
            }
        }
        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
        fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, element: u64) {
            if self.phase == Some("prefill-peak") {
                self.prefill_kv = self.prefill_kv.max(bytes);
                self.element_bytes = element;
            }
        }
    }
    let model = tiny_packed_capable_model();
    // 40 tokens: past the fused-SDPA row limit (the dense first step), one flushed 32-token
    // group and an 8-token residual.
    let prompt = (0..40).map(|i| (i % 31) + 1).collect::<Vec<i32>>();
    let config = GenerationConfig {
        max_new_tokens: 2,
        seed: Some(0),
        ..Default::default()
    };
    for method in crate::decode::observe::CompressedKvMethod::ALL {
        let arm = method.arm().unwrap();
        let decoder = PackedCampaignDecoder {
            model: &model,
            arm: &arm,
            selection_fallbacks: RefCell::new(Vec::new()),
        };
        let run = |observer: Option<&mut dyn crate::decode::observe::Observer>| {
            crate::decode::generate_with_observer(
                &decoder,
                &prompt,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
                observer,
            )
            .unwrap();
        };
        // Warm the reader and the lazily created weights so the baseline is steady. Tests run
        // one at a time (`.cargo/config.toml`), so the process-global counter is ours.
        run(None);
        let baseline = mlx_rs::memory::get_active_memory() as u64;
        let mut capture = PrefillCapture::default();
        run(Some(&mut capture));
        assert!(
            decoder.selection_fallbacks.borrow().is_empty(),
            "{method:?} selected the packed cache"
        );
        // The failing scenario: the prompt's dense K + V (2 layers, 1 KV head, D64, at the
        // cache's element width) is smaller than the reported store.
        let dense_prompt_kv = 2 * 2 * 40 * 64 * capture.element_bytes;
        assert!(
            dense_prompt_kv < capture.prefill_kv,
            "{method:?}: dense {dense_prompt_kv} B vs store {} B",
            capture.prefill_kv
        );
        let growth = capture.prefill_active.unwrap().saturating_sub(baseline);
        assert!(
            growth >= capture.prefill_kv,
            "{method:?}: prefill-peak active growth {growth} B does not hold the reported \
                 packed store {} B",
            capture.prefill_kv
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::compressed_prefill_sample_holds_the_reported_packed_store");
}

/// SC-20671 storage reconciliation through the real decode loop, at 2- and 4-bit: a prompt
/// that crosses a 256-token block boundary leaves a pending residual, and the decode that
/// follows (including a group flush) stays inside the last block, so the device share is a
/// plateau. The coordinate storage the campaign observer keeps must be the plateau's latest
/// instant: its device share is the largest persistent snapshot and its length the largest
/// live length (the Mac2 32k single-shot row recorded the prefill instant, 32836 tokens,
/// against a kvLength of 32843).
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn compressed_coordinate_storage_reconciles_across_a_block_plateau() {
    #[derive(Default)]
    struct StorageCapture {
        snapshots: Vec<(u64, u64)>,
        storage: Vec<crate::primitives::CompressedCacheStorage>,
    }
    impl crate::decode::observe::Observer for StorageCapture {
        fn phase(&mut self, _name: &'static str) {}
        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
        fn cache_snapshot(&mut self, bytes: u64, tokens: u64, _capacity: u64, _element: u64) {
            self.snapshots.push((bytes, tokens));
        }
        fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
            self.storage.push(*storage);
        }
    }
    let model = tiny_packed_capable_model();
    // 300 prompt tokens: one full block plus 44 (a 12-token residual after 32-token groups).
    let prompt = (0..300).map(|i| (i % 31) + 1).collect::<Vec<i32>>();
    for method in crate::decode::observe::CompressedKvMethod::ALL {
        let arm = method.arm().unwrap();
        let decoder = PackedCampaignDecoder {
            model: &model,
            arm: &arm,
            selection_fallbacks: RefCell::new(Vec::new()),
        };
        let mut capture = StorageCapture::default();
        let config = GenerationConfig {
            max_new_tokens: 30,
            seed: Some(0),
            ..Default::default()
        };
        crate::decode::generate_with_observer(
            &decoder,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            Some(&mut capture),
        )
        .unwrap();
        let mut observer = crate::campaign::ProductObserver::new();
        observer.begin_coordinate_operation();
        for storage in &capture.storage {
            crate::decode::observe::Observer::compressed_storage(&mut observer, storage);
        }
        observer.end_coordinate_operation();
        let peak = observer.coordinate_storage_peak().unwrap();
        let persistent = capture
            .snapshots
            .iter()
            .map(|(bytes, _)| *bytes)
            .max()
            .unwrap();
        let kv_length = capture
            .snapshots
            .iter()
            .map(|(_, tokens)| *tokens)
            .max()
            .unwrap();
        // The scenario is the failing one: several instants share the peak device share.
        let plateau = capture
            .storage
            .iter()
            .filter(|storage| storage.device_bytes() == persistent)
            .map(|storage| storage.tokens)
            .collect::<Vec<_>>();
        assert!(
            plateau.len() > 1 && plateau[0] < kv_length,
            "{method:?} {plateau:?}"
        );
        assert!(
            kv_length > 300 && kv_length < 512,
            "{method:?} decode stays in block two"
        );
        crate::campaign::coordinate_storage_reconciles(
            peak.device_code_bytes,
            Some(peak.device_bytes()),
            peak.tokens,
            persistent,
            kv_length,
        )
        .unwrap_or_else(|error| panic!("{method:?}: {error}"));
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::compressed_coordinate_storage_reconciles_across_a_block_plateau");
}

/// SC-20671 steady decode on a tiny model: both arms decode exactly the fixed length through
/// stop tokens (every vocabulary id is declared one), and a compressed arm whose reader is
/// refused fails closed instead of timing a dense decode under the compressed label.
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn campaign_steady_decode_is_fixed_length_on_both_arms_and_refuses_a_dense_compressed_arm() {
    let model = Decoder::Causal(Box::new(tiny_packed_capable_model()));
    let every_id = (0..32).collect::<Vec<i32>>();
    let prompt = [1, 2, 3, 4, 5];
    let arm = crate::decode::observe::CompressedKvMethod::GroupAffine
        .arm()
        .unwrap();
    for compressed in [None, Some(&arm)] {
        let measured =
            campaign_steady_decode_on(&model, &prompt, 8, &every_id, compressed).unwrap();
        assert_eq!(measured.prompt_tokens, 5);
        assert_eq!(measured.generated_tokens, 8);
        assert_eq!(measured.timed_tokens, 7);
        assert_eq!(
            measured.forced_stop_tokens, 8,
            "every token was a forced stop"
        );
    }
    let refused = crate::decode::observe::CompressedKvArm::with_reader(
        crate::decode::observe::CompressedKvMethod::GroupAffine,
        crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
            crate::primitives::OpaqueCompiledKernel::new(
                "sc20671-refused",
                "cpu",
                0,
                std::sync::Arc::new(()),
            ),
        )),
    );
    let error = campaign_steady_decode_on(&model, &prompt, 8, &every_id, Some(&refused))
        .unwrap_err()
        .to_string();
    assert!(error.contains("fused compressed reader"), "{error}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::campaign_steady_decode_is_fixed_length_on_both_arms_and_refuses_a_dense_compressed_arm");
}

/// SC-20671 forced continuation on a tiny model: the dense arm continues greedily through
/// every stop token; teacher-forcing the dense arm on its own continuation reproduces it at
/// every position; the compressed arm is teacher-forced on the dense stream wholly on its fused
/// reader; a position forced off the compressed arm's choice is a recorded flip; and a refused
/// compressed reader fails closed.
#[cfg(feature = "campaign")]
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn campaign_forced_continuation_teacher_forces_the_compressed_arm_on_the_dense_stream() {
    let model = Decoder::Causal(Box::new(tiny_packed_capable_model()));
    let every_id = (0..32).collect::<Vec<i32>>();
    let prompt = [1, 2, 3, 4, 5];
    let arm = crate::decode::observe::CompressedKvMethod::GroupAffine
        .arm()
        .unwrap();
    let scored = |compressed, stream: Option<&[i32]>| {
        campaign_forced_decode_on(&model, &prompt, 64, &every_id, compressed, stream, true)
    };
    let forced =
        |compressed, stream: Option<&[i32]>| scored(compressed, stream).map(|decode| decode.tokens);
    let reference = forced(None, None).unwrap();
    assert_eq!(reference.len(), 64, "every stop token is decoded through");
    assert_eq!(forced(None, Some(&reference)).unwrap(), reference);
    // Scored on the same stream, the dense arm teacher-forced on its own continuation has
    // exactly its free-running likelihood; the compressed arm scores those same tokens.
    let dense = scored(None, None).unwrap().stream_probabilities;
    assert_eq!(dense.len(), 64);
    assert_eq!(
        scored(None, Some(&reference)).unwrap().stream_probabilities,
        dense
    );
    let compressed = scored(Some(&arm), Some(&reference))
        .unwrap()
        .stream_probabilities;
    assert_eq!(compressed.len(), 64);
    assert!(compressed
        .iter()
        .all(|probability| *probability > 0.0 && *probability <= 1.0));
    // The dense noise-floor control decodes the same stream after a chunked prefill: an exact
    // dense computation in another order, so it agrees with the one-shot continuation and
    // scores the same tokens (here identically: the tiny model's chunks are exact).
    for chunk in [1, 2, 3, 4, 64] {
        let control =
            campaign_chunked_prefill_decode_on(&model, &prompt, &reference, &every_id, chunk)
                .unwrap();
        assert_eq!(control.tokens, reference, "chunk {chunk}");
        assert_eq!(control.stream_probabilities.len(), 64);
        let likelihood = crate::campaign::StreamLikelihood::from_probabilities(
            &dense,
            &control.stream_probabilities,
        )
        .unwrap();
        assert!(
            (likelihood.candidate - likelihood.reference).abs() < 1e-3,
            "chunk {chunk}: {likelihood:?}"
        );
    }
    assert!(campaign_chunked_prefill_decode_on(&model, &prompt, &reference, &every_id, 0).is_err());
    // Run 37021783368: the control's cache is ONE allocation for the prompt and stream, so a
    // prompt longer than the default 256-position block retires no growth buffer (the
    // default cache retired one per growth into MLX's buffer cache, past the row's cap).
    let long_prompt = (0..600).map(|index| index % 31).collect::<Vec<_>>();
    let mut cache = chunked_prefill_cache(&model, long_prompt.len() + 64).unwrap();
    KvCache::record_events(&mut cache);
    chunked_prefill(&model, &mut cache, &long_prompt, 64).unwrap();
    assert_eq!(cache.offset(), 600);
    assert!(
        cache
            .events()
            .iter()
            .all(|event| event.operation != "dense_block_growth_retired_buffer"),
        "the chunked control retired a dense growth buffer"
    );
    let mut grown = match &model {
        Decoder::Causal(model) => model.new_cache(),
        Decoder::Qwen35(_) => unreachable!(),
    };
    KvCache::record_events(&mut grown);
    chunked_prefill(&model, &mut grown, &long_prompt, 64).unwrap();
    assert!(
        grown
            .events()
            .iter()
            .any(|event| event.operation == "dense_block_growth_retired_buffer"),
        "the default cache grows (the defect the sized cache removes)"
    );
    let choices = forced(Some(&arm), Some(&reference)).unwrap();
    let evidence = crate::campaign::forced_continuation_evidence(&reference, &choices).unwrap();
    assert_eq!(evidence.tokens, 64);
    assert_eq!(evidence.matches + evidence.flip_count, 64);
    // Force position 10 off the compressed arm's own choice there: a guaranteed flip.
    let mut off = reference.clone();
    off[10] = (choices[10] + 1) % 32;
    let off_choices = forced(Some(&arm), Some(&off)).unwrap();
    assert_eq!(
        off_choices[..10],
        choices[..10],
        "the prefix before 10 is unchanged"
    );
    let miss = crate::campaign::forced_continuation_evidence(&off, &off_choices).unwrap();
    assert!(miss.first_flip_positions.contains(&10), "{miss:?}");
    assert!(miss.agreement < 1.0);
    assert!(forced(Some(&arm), Some(&reference[..63])).is_err());
    let refused = crate::decode::observe::CompressedKvArm::with_reader(
        crate::decode::observe::CompressedKvMethod::GroupAffine,
        crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
            crate::primitives::OpaqueCompiledKernel::new(
                "sc20671-refused",
                "cpu",
                0,
                std::sync::Arc::new(()),
            ),
        )),
    );
    let error = forced(Some(&refused), Some(&reference))
        .unwrap_err()
        .to_string();
    assert!(error.contains("fused compressed reader"), "{error}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::campaign_forced_continuation_teacher_forces_the_compressed_arm_on_the_dense_stream");
}

// ---- sc-20679: production compressed KV ----

/// One greedy product generation over `words` user words at `policy`, through the
/// production entry point (`TextLlm::generate`).
fn kv_generate(
    provider: &LlamaProvider,
    words: usize,
    policy: core_llm::KvCompressionPolicy,
    max_new_tokens: u32,
) -> TextLlmOutput {
    kv_generate_words(provider, words, (7, 26), policy, max_new_tokens).0
}

/// [`kv_generate`] over the word pattern `w{(i · step) % modulo + 6}`, also returning the
/// rendered prompt ids and the generated token ids as the stream emitted them.
fn kv_generate_words(
    provider: &LlamaProvider,
    words: usize,
    (step, modulo): (usize, usize),
    policy: core_llm::KvCompressionPolicy,
    max_new_tokens: u32,
) -> (TextLlmOutput, Vec<i32>, Vec<i32>) {
    let text = (0..words)
        .map(|i| format!("w{}", (i * step) % modulo + 6))
        .collect::<Vec<_>>()
        .join(" ");
    let request = TextLlmRequest {
        messages: vec![Message::text(Role::User, text)],
        sampling: core_llm::Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        kv_compression: policy,
        ..Default::default()
    };
    let (_, prompt_ids) = provider.render_prompt(&request, &request.messages).unwrap();
    let mut stream = Vec::new();
    let output = provider
        .generate(&request, &mut |event| {
            if let CoreEvent::Token { id, .. } = event {
                stream.push(id as i32);
            }
        })
        .unwrap();
    (output, prompt_ids, stream)
}

/// A tiny snapshot's provider exactly as production loads it: its synthetic architecture is
/// no measured row's, so it has no compressed-KV table family.
fn load_tiny_unarmed(snapshot: &tempfile::TempDir) -> LlamaProvider {
    LlamaProvider::load(&core_llm::LoadSpec::dense(
        snapshot.path().to_string_lossy().to_string(),
    ))
    .unwrap()
}

/// [`load_tiny_unarmed`], test-armed: the provider plans its KV cache as its decoder's
/// dispatch family ([`core_llm::kv_model_family`]), as if the tiny fixture were that family's
/// measured architecture, so the compressed path can run on synthetic weights.
fn load_tiny(snapshot: &tempfile::TempDir) -> LlamaProvider {
    let mut provider = load_tiny_unarmed(snapshot);
    provider.kv_family = crate::kv_policy::family_for(provider.campaign_family);
    provider
}

/// Gains that make the tiny fixture's generation depend on its context: sharp attention whose
/// output dominates the residual, and a head only weakly tied to the current token. With the
/// default gains every prompt decodes the same single token, which no reader could get wrong.
/// Sharper attention (qk ≥ 3) amplifies 8-bit key rounding past the parity bound, so the
/// gains stay in the band where 8 bits holds and 2/4 bits do not.
const CONTEXT_SENSITIVE_GAINS: SnapshotGains = SnapshotGains {
    qk: 2.0,
    vo: 3.0,
    head_align: 1.0,
};

/// AC3: a request at the family's qualified minimum context runs wholly on the fused
/// compressed reader through the production entry point, and every token it emits is the
/// dense model's choice at that position (its dense logit within 8-bit rounding of the dense
/// maximum) — on a fixture whose dense output demonstrably depends on the context. The same
/// provider keeps a short request, and an un-opted request, dense with their reasons.
fn assert_compressed_matches_dense(
    identity: serde_json::Value,
    qk_norm: bool,
    family: core_llm::KvModelFamily,
) {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == family)
        .unwrap();
    let min = usize::try_from(row.min_context_tokens).unwrap();
    let snapshot = tiny_snapshot_with(
        identity,
        row.min_context_tokens + 1024,
        qk_norm,
        CONTEXT_SENSITIVE_GAINS,
    );
    let provider = load_tiny(&snapshot);
    const NEW: u32 = 24;
    const PATTERN: (usize, usize) = (7, 26);

    let dense = kv_generate(&provider, min, Policy::Off, NEW);
    let (compressed, prompt_ids, stream) =
        kv_generate_words(&provider, min, PATTERN, Policy::Qualified, NEW);

    // Sensitivity control: the dense output is context-dependent and varied, so a reader
    // that corrupts the history changes what it should emit.
    let control = kv_generate_words(&provider, min, (5, 13), Policy::Off, NEW).0;
    assert_ne!(control.text, dense.text, "the fixture ignores its context");
    let distinct = |text: &str| {
        text.split_whitespace()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    assert!(distinct(&dense.text) >= 4, "{}", dense.text);
    assert!(distinct(&compressed.text) >= 4, "{}", compressed.text);

    // Per-step parity on the stream the production compressed generation emitted: the dense
    // model, teacher-forced on that stream, scores every emitted token within 8-bit rounding
    // of its best. With every logit within 0.02 of the dense range (the logit-parity bound),
    // a near-tie the compressed cache resolves the other way costs at most two such errors:
    // 0.04. Measured ≤ 0.004 here at 8 bits; 4-bit codes reach 0.15 and 2-bit 0.3–0.55.
    assert_eq!(stream.len(), NEW as usize);
    let Decoder::Causal(model) = &provider.model else {
        panic!("a qualified family is a causal decoder");
    };
    let mut dense_cache = model.make_cache();
    let rows = forced_logits(
        model,
        dense_cache.as_mut(),
        &prompt_ids,
        &stream[..stream.len() - 1],
    );
    let range = rows
        .iter()
        .flatten()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    for (position, (logits, &token)) in rows.iter().zip(&stream).enumerate() {
        let best = logits.iter().copied().fold(f32::MIN, f32::max);
        let gap = (best - logits[token as usize]) / range;
        assert!(
            gap <= 0.04,
            "position {position}: compressed emitted {token}, {gap} of the dense logit \
                 range below the dense choice"
        );
    }
    assert!(
        u64::from(compressed.usage.prompt_tokens) >= row.min_context_tokens,
        "{:?}",
        compressed.usage
    );
    assert_eq!(compressed.usage, dense.usage);
    assert_eq!(compressed.usage.generated_tokens, NEW);
    assert_eq!(
        dense.kv_cache,
        Some(core_llm::KvCacheReport::dense(Reason::PolicyDisabled, None))
    );
    let report = compressed.kv_cache.clone().unwrap();
    assert!(report.ran_compressed(), "{report:?}");
    assert_eq!(report.format, Some(row.format));
    assert_eq!(report.format_version, core_llm::KV_CACHE_FORMAT_VERSION);
    // Every decode step after the prefill reads both layers through the fused reader.
    assert!(
        report.counters.fused_attention_calls >= 2 * u64::from(NEW - 1),
        "{report:?}"
    );
    assert_eq!(report.counters.full_cache_dequantizations, 0);
    assert_eq!(report.counters.dense_fallback_events, 0);
    // Retained compressed bytes stay below the dense K/V of the same tokens
    // (2 layers × K,V × 1 KV head × head dim 64 × bf16).
    let tokens = u64::from(compressed.usage.total_tokens());
    assert!(report.counters.compressed_cache_bytes > 0);
    assert!(
        report.counters.compressed_cache_bytes < tokens * 2 * 2 * 64 * 2,
        "{report:?}"
    );
    // The same provider keeps a short request dense: short contexts decode slower compressed.
    let short = kv_generate(&provider, 64, Policy::Qualified, 8);
    assert_eq!(
        short.kv_cache,
        Some(core_llm::KvCacheReport::dense(
            Reason::BelowMinimumContext,
            None
        ))
    );
    assert_eq!(
        short.text,
        kv_generate(&provider, 64, Policy::Off, 8).text,
        "a refused request generates exactly the dense output"
    );
}

/// Per-position logits of `model` prefilling `prompt` on `cache` and then teacher-forced on
/// `stream`, one token per step: the prefill's last position, then one row per forced token.
fn forced_logits(
    model: &CausalLm,
    cache: &mut dyn KvCache,
    prompt: &[i32],
    stream: &[i32],
) -> Vec<Vec<f32>> {
    let host = |logits: Array| {
        let logits = logits.as_dtype(Dtype::Float32).unwrap();
        logits.eval().unwrap();
        logits.as_slice::<f32>().to_vec()
    };
    let mut rows = vec![host(model.step(&input_ids(prompt), cache, 0).unwrap())];
    for (i, &token) in stream.iter().enumerate() {
        let offset = (prompt.len() + i) as i32;
        rows.push(host(
            model.step(&input_ids(&[token]), cache, offset).unwrap(),
        ));
    }
    rows
}

/// AC3 logit parity: on the cache the production plan selects for a request at the family's
/// qualified minimum context, every teacher-forced decode position's logits stay within 8-bit
/// rounding of the dense cache's. Returns the largest error relative to the dense logit range.
fn compressed_logit_error(snapshot: &tempfile::TempDir, family: core_llm::KvModelFamily) -> f32 {
    let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == family)
        .unwrap();
    let provider = load_tiny(snapshot);
    let Decoder::Causal(model) = &provider.model else {
        panic!("a qualified family is a causal decoder");
    };
    let prompt = (0..row.min_context_tokens)
        .map(|i| ((i * 7) % 26 + 6) as i32)
        .collect::<Vec<_>>();
    let stream = (0..40).map(|i| (i * 5) % 26 + 6).collect::<Vec<i32>>();
    let KvPlan::Compressed { format, reader } = provider.plan_kv_cache(
        core_llm::KvCompressionPolicy::Qualified,
        prompt.len(),
        stream.len() as u32,
        1,
        false,
    ) else {
        panic!("the qualified context must plan compressed");
    };
    let (mut cache, refused) = crate::kv_policy::select_compressed_cache(
        model,
        reader,
        prompt.len(),
        admit_any_transition(),
    );
    assert_eq!(refused, None);
    let compressed = forced_logits(model, cache.as_mut(), &prompt, &stream);
    let report = crate::kv_policy::compressed_report(format, None, cache.as_ref()).unwrap();
    assert!(report.ran_compressed(), "{report:?}");
    assert_eq!(
        report.counters.fused_attention_calls,
        2 * stream.len() as u64
    );
    let mut dense_cache = model.make_cache();
    let dense = forced_logits(model, dense_cache.as_mut(), &prompt, &stream);
    let range = dense
        .iter()
        .flatten()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    assert!(range > 0.0);
    dense
        .iter()
        .zip(&compressed)
        .flat_map(|(dense, compressed)| dense.iter().zip(compressed))
        .map(|(dense, compressed)| (dense - compressed).abs() / range)
        .fold(0.0f32, f32::max)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn compressed_kv_decode_logits_match_dense_within_8_bit_rounding() {
    for (identity, qk_norm, family) in [
        (
            json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
            false,
            core_llm::KvModelFamily::Llama,
        ),
        (
            json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            true,
            core_llm::KvModelFamily::Qwen3,
        ),
    ] {
        let snapshot = tiny_snapshot(identity, 40_960, qk_norm);
        // Measured ~0.009 at 8 bits (bf16 kernel rounding included) against ~0.3 for the same
        // cache at 2 bits; 0.02 is the bound the SC-20671 8-bit prompt-cache test uses.
        let error = compressed_logit_error(&snapshot, family);
        eprintln!("sc-20679 {family:?}: K8V8 decode logit error {error}");
        assert!(
            error < 0.02,
            "{family:?}: K8V8 decode logits drifted {error}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::compressed_kv_decode_logits_match_dense_within_8_bit_rounding");
}

/// sc-20688 review (finding 1) at the provider: `generate_batch` decodes the plain text
/// requests together, and two concurrent qualified requests — each one a single request would
/// run compressed — both decode dense with `BatchedDecode`, the reason the single-request plan
/// gives a batch, as does a short one beside them; their tokens are exactly the un-opted
/// batch's. A request the batch cannot serve (a stop string) runs alone, a single sequence,
/// and qualifies. Every request streams under its own index with exactly one `Done`, and no
/// batched prompt reaches the compressed prefix store.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn generate_batch_decodes_concurrent_qualified_requests_dense_as_batched() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    let provider = load_tiny(&tiny_snapshot(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        40_960,
        true,
    ));
    let words = |count: usize, salt: usize| {
        (0..count)
            .map(|i| format!("w{}", (i * 7 + salt) % 26 + 6))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let request = |text: String, stop: Vec<String>, policy| TextLlmRequest {
        messages: vec![Message::text(Role::User, text)],
        sampling: core_llm::Sampling::greedy(),
        max_new_tokens: 4,
        seed: Some(0),
        kv_compression: policy,
        stop,
        ..Default::default()
    };
    let requests = |policy| {
        vec![
            request(words(10_300, 0), Vec::new(), policy),
            request(words(10_450, 3), Vec::new(), policy),
            request(words(64, 5), Vec::new(), policy),
            request(words(10_300, 1), vec!["never-emitted".into()], policy),
        ]
    };
    // A batch of only plain requests builds no fused reader (nor page pool or prefix store):
    // the product batch never arms paged compression, so it sets none of it up.
    let fresh = load_tiny(&tiny_snapshot(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        40_960,
        true,
    ));
    for out in fresh.generate_batch(&requests(Policy::Qualified)[..3], &mut |_, _| {}) {
        assert_eq!(
            out.unwrap().kv_cache.unwrap().fallback,
            Some(Reason::BatchedDecode)
        );
    }
    assert!(
        fresh.kv_reader.get().is_none(),
        "a batch that cannot run compressed builds no reader"
    );
    // Each batched request on its own qualifies: the batch, not the request, refuses it.
    for (i, req) in requests(Policy::Qualified).iter().enumerate().take(2) {
        let prompt = provider.render_prompt(req, &req.messages).unwrap().1.len();
        assert!(
            matches!(
                provider.plan_kv_cache(Policy::Qualified, prompt, req.max_new_tokens, 1, false),
                KvPlan::Compressed { .. }
            ),
            "request {i} alone runs compressed"
        );
    }
    let generate = |policy| {
        let (mut tokens, mut dones) = (vec![0usize; 4], vec![0usize; 4]);
        let outputs = provider
            .generate_batch(&requests(policy), &mut |i, event| match event {
                CoreEvent::Token { .. } => tokens[i] += 1,
                CoreEvent::Done { .. } => dones[i] += 1,
            })
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(dones, vec![1; 4]);
        for (i, out) in outputs.iter().enumerate() {
            assert_eq!(out.usage.generated_tokens, 4, "request {i}");
            assert!(tokens[i] > 0, "request {i} streamed");
        }
        outputs
    };
    let outputs = generate(Policy::Qualified);
    let off = generate(Policy::Off);
    for (i, (out, off)) in outputs.iter().zip(&off).enumerate().take(3) {
        assert_eq!(
            out.kv_cache,
            Some(core_llm::KvCacheReport::dense(Reason::BatchedDecode, None)),
            "request {i}"
        );
        assert_eq!(out.text, off.text, "request {i}");
    }
    // Served alone, on the single-request (contiguous) compressed path.
    let alone = outputs[3].kv_cache.as_ref().unwrap();
    assert!(alone.ran_compressed(), "{alone:?}");
    assert_eq!(alone.counters.pool_held_bytes, 0);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::generate_batch_decodes_concurrent_qualified_requests_dense_as_batched");
}

/// sc-20688 review round 2: a batch whose other request is cancelled before it starts leaves
/// one survivor, which is not decoded on the continuous path (no page pool, no prefix store):
/// it runs alone through the single-request path, compressed on the contiguous cache.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_lone_batch_survivor_runs_on_the_contiguous_path_not_paged() {
    use core_llm::KvCompressionPolicy as Policy;
    let provider = load_tiny(&tiny_snapshot(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        40_960,
        true,
    ));
    let words = |count: usize, salt: usize| {
        (0..count)
            .map(|i| format!("w{}", (i * 7 + salt) % 26 + 6))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let request = |text: String| TextLlmRequest {
        messages: vec![Message::text(Role::User, text)],
        sampling: core_llm::Sampling::greedy(),
        max_new_tokens: 4,
        seed: Some(0),
        kv_compression: Policy::Qualified,
        ..Default::default()
    };
    let requests = vec![request(words(10_300, 0)), request(words(10_450, 3))];
    requests[1].cancel.cancel();
    let mut outputs = provider
        .generate_batch(&requests, &mut |_, _| {})
        .into_iter();
    let survivor = outputs.next().unwrap().unwrap();
    assert!(matches!(outputs.next().unwrap(), Err(CoreError::Canceled)));
    let report = survivor.kv_cache.as_ref().unwrap();
    assert!(report.ran_compressed(), "{report:?}");
    assert_eq!(
        report.counters.pool_held_bytes, 0,
        "the contiguous cache, not a shared page pool"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_lone_batch_survivor_runs_on_the_contiguous_path_not_paged");
}

/// The two epics together (sc-24432 × sc-20669): a qualified compressed request runs through
/// the engine like any other — an explicit prompt-lookup route verifies its drafts on the
/// compressed cache and rolls the rejected ones back exactly — and still reports the
/// compressed cache it ran on. It reads the cross-turn prefix store but never feeds it: the
/// store holds dense K/V and a compressed cache never re-enters it, by name.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_compressed_request_speculates_on_the_compressed_cache() {
    use core_llm::{KvCompressionPolicy as Policy, Speculative, SpeculativeProposer};
    let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == core_llm::KvModelFamily::Qwen3)
        .unwrap();
    let snapshot = tiny_snapshot_with(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        row.min_context_tokens + 1024,
        true,
        CONTEXT_SENSITIVE_GAINS,
    );
    let provider = load_tiny(&snapshot);
    let words = usize::try_from(row.min_context_tokens).unwrap();
    // A repeating pattern, so prompt lookup finds n-gram drafts to propose.
    let text = (0..words)
        .map(|i| format!("w{}", (i * 7) % 26 + 6))
        .collect::<Vec<_>>()
        .join(" ");
    let request = TextLlmRequest {
        messages: vec![Message::text(Role::User, text)],
        sampling: core_llm::Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        kv_compression: Policy::Qualified,
        speculative: Some(Speculative::proposer(SpeculativeProposer::PromptLookup, 4)),
        ..Default::default()
    };
    // The same request run plainly on the compressed cache: the tokens speculation must keep.
    let off_request = TextLlmRequest {
        speculative: Some(Speculative::Off),
        ..request.clone()
    };
    let (off, off_ids) = run(&provider, &off_request);
    assert!(off.kv_cache.as_ref().is_some_and(|kv| kv.ran_compressed()));
    let (output, ids) = run(&provider, &request);
    assert_eq!(
        ids, off_ids,
        "speculation on the compressed cache keeps the plain tokens"
    );
    let kv = output.kv_cache.expect("a generation reports its KV cache");
    assert!(kv.ran_compressed(), "{kv:?}");
    assert!(kv.counters.fused_attention_calls > 0, "{kv:?}");
    let decode = output.decode.expect("an engine request reports its decode");
    assert_eq!(decode.proposer, ProposerKind::PromptLookup, "{decode:?}");
    assert!(
        decode.verify_steps > 0 && decode.proposed_tokens > 0,
        "{decode:?}"
    );
    assert_eq!(output.usage.generated_tokens, 16);
    assert_eq!(
        decode.prefix_cache.reason.as_deref(),
        Some(core_llm::PREFIX_COMPRESSED_NOT_STORED)
    );
    assert!(provider.prefix.borrow().is_empty(), "nothing was stored");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_compressed_request_speculates_on_the_compressed_cache");
}

/// The Qwen3 fixture at its compressed qualification row, and the row's minimum context.
fn qualified_qwen3_fixture() -> (tempfile::TempDir, usize) {
    let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == core_llm::KvModelFamily::Qwen3)
        .unwrap();
    let snapshot = tiny_snapshot_with(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        row.min_context_tokens + 1024,
        true,
        CONTEXT_SENSITIVE_GAINS,
    );
    (snapshot, usize::try_from(row.min_context_tokens).unwrap())
}

/// Dense → compressed prefix reuse (sc-24437 × sc-20681): a dense turn stores its KV; the
/// compressed turn of the same prompt hits it — imported into the compressed cache, so only
/// the suffix is prefilled — and decodes exactly a cold compressed run's tokens, while the
/// store is left as the dense turn left it (a compressed cache never re-enters it).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_compressed_turn_imports_the_dense_turns_prefix() {
    use core_llm::KvCompressionPolicy as Policy;
    let (snapshot, words) = qualified_qwen3_fixture();
    const NEW: u32 = 12;
    let cold = load_tiny(&snapshot);
    let (cold_out, _, cold_ids) = kv_generate_words(&cold, words, (7, 26), Policy::Qualified, NEW);
    assert!(cold_out.kv_cache.unwrap().ran_compressed());
    assert_eq!(cold_out.decode.unwrap().prefix_hit_tokens, 0);

    let provider = load_tiny(&snapshot);
    kv_generate(&provider, words, Policy::Off, NEW);
    let stored = (
        provider.prefix.borrow().keys(),
        provider.prefix.borrow().resident_bytes(),
    );
    assert_eq!(stored.0.len(), 1, "the dense turn stores its KV");
    let (out, prompt_ids, ids) =
        kv_generate_words(&provider, words, (7, 26), Policy::Qualified, NEW);
    let kv = out.kv_cache.unwrap();
    assert!(kv.ran_compressed(), "{kv:?}");
    let decode = out.decode.unwrap();
    assert_eq!(decode.prefix_cache.path, "hit", "{decode:?}");
    assert_eq!(decode.prefix_hit_tokens as usize, prompt_ids.len() - 1);
    assert_eq!(
        decode.prefix_cache.reason.as_deref(),
        Some(core_llm::PREFIX_COMPRESSED_NOT_STORED)
    );
    assert_eq!(
        ids, cold_ids,
        "the imported prefix decodes the cold compressed run"
    );
    assert_eq!(
        (
            provider.prefix.borrow().keys(),
            provider.prefix.borrow().resident_bytes()
        ),
        stored,
        "the compressed turn leaves the store unchanged"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_compressed_turn_imports_the_dense_turns_prefix");
}

/// A compressed turn whose cache declines the stored dense prefix (here: a reader its cache
/// cannot bind, so the selection runs dense) prefills the whole prompt and says so: no reuse
/// reported, the declined import named.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_declined_prefix_import_is_named_not_counted_as_reuse() {
    use core_llm::KvCompressionPolicy as Policy;
    let (snapshot, words) = qualified_qwen3_fixture();
    let provider = load_tiny(&snapshot);
    #[allow(clippy::arc_with_non_send_sync)]
    let reader =
        crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(UnbindableReader));
    assert!(provider.kv_reader.set(Ok(reader)).is_ok());
    kv_generate(&provider, words, Policy::Off, 4);
    let out = kv_generate(&provider, words, Policy::Qualified, 4);
    assert_eq!(
        out.kv_cache.unwrap().fallback,
        Some(core_llm::KvCacheFallbackReason::ReaderUnavailable)
    );
    let decode = out.decode.unwrap();
    assert_eq!(decode.prefix_hit_tokens, 0, "{decode:?}");
    assert_eq!(decode.prefix_cache.path, "miss", "{decode:?}");
    assert_eq!(
        decode.prefix_cache.reason.as_deref(),
        Some(core_llm::PREFIX_COMPRESSED_IMPORT_DECLINED)
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_declined_prefix_import_is_named_not_counted_as_reuse");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn qwen3_compressed_kv_matches_dense_at_its_qualified_context() {
    assert_compressed_matches_dense(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        true,
        core_llm::KvModelFamily::Qwen3,
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::qwen3_compressed_kv_matches_dense_at_its_qualified_context");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn llama_compressed_kv_matches_dense_at_its_qualified_context() {
    assert_compressed_matches_dense(
        json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        false,
        core_llm::KvModelFamily::Llama,
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::llama_compressed_kv_matches_dense_at_its_qualified_context");
}

/// AC1: a model whose family has no qualification row stays dense when opted in, and says so;
/// so does every non-text path the plan refuses before any cache exists.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn unqualified_models_stay_dense_with_their_reason() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    // Mistral shares the Llama decoder but is not the family the evidence measured.
    let snapshot = tiny_snapshot(
        json!({"architectures": ["MistralForCausalLM"], "model_type": "mistral"}),
        2048,
        false,
    );
    let provider = load_tiny(&snapshot);
    let output = kv_generate(&provider, 64, Policy::Qualified, 4);
    assert_eq!(
        output.kv_cache,
        Some(core_llm::KvCacheReport::dense(
            Reason::UnqualifiedModel,
            None
        ))
    );
    // The plan refuses a multimodal request on a qualified family before any cache exists.
    let qwen = load_tiny(&tiny_snapshot(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        40_960,
        true,
    ));
    let KvPlan::Dense(report) = qwen.plan_kv_cache(Policy::Qualified, 20_000, 0, 1, true) else {
        panic!("a multimodal request must plan dense");
    };
    assert_eq!(report.fallback, Some(Reason::UnsupportedRequest));
    assert!(matches!(
        qwen.plan_kv_cache(Policy::Qualified, 20_000, 64, 1, false),
        KvPlan::Compressed { .. }
    ));
    assert!(matches!(
        qwen.plan_kv_cache(Policy::Off, 20_000, 64, 1, false),
        KvPlan::Dense(report) if report.fallback == Some(Reason::PolicyDisabled)
    ));
    // The plan bounds the final context, not just the prompt: a fit-boundary prompt whose
    // budget runs past the evidenced 40 960-token window stays dense.
    assert!(matches!(
        qwen.plan_kv_cache(Policy::Qualified, 40_448, 512, 1, false),
        KvPlan::Compressed { .. }
    ));
    assert!(matches!(
        qwen.plan_kv_cache(Policy::Qualified, 40_448, 513, 1, false),
        KvPlan::Dense(report) if report.fallback == Some(Reason::AboveQualifiedContext)
    ));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::unqualified_models_stay_dense_with_their_reason");
}

/// AC1 at the production entry point: the request's token budget counts against the
/// qualified range. A Qwen3 prompt that fits the evidenced 40 960-token window runs compressed
/// only while prompt + `max_new_tokens` stays inside it.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_final_context_bounds_a_production_compressed_request() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    let provider = load_tiny(&tiny_snapshot(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        41_472,
        true,
    ));
    // The chat template's own tokens, so the prompt lands 8 tokens inside the window.
    let one_word = TextLlmRequest {
        messages: vec![Message::text(Role::User, "w6")],
        ..Default::default()
    };
    let overhead = provider
        .render_prompt(&one_word, &one_word.messages)
        .unwrap()
        .1
        .len()
        - 1;
    let words = 40_960 - overhead - 8;
    let (inside, prompt_ids, _) =
        kv_generate_words(&provider, words, (7, 26), Policy::Qualified, 1);
    let headroom = u32::try_from(40_960 - prompt_ids.len()).unwrap();
    assert_eq!(headroom, 8);
    assert!(inside.kv_cache.unwrap().ran_compressed());
    let fits = kv_generate(&provider, words, Policy::Qualified, headroom);
    assert!(fits.kv_cache.unwrap().ran_compressed());
    let past = kv_generate(&provider, words, Policy::Qualified, headroom + 1);
    assert_eq!(
        past.kv_cache,
        Some(core_llm::KvCacheReport::dense(
            Reason::AboveQualifiedContext,
            None
        ))
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::the_final_context_bounds_a_production_compressed_request");
}

/// A provider assembled from parts cannot tell Llama from a Mistral or dense Qwen2 checkpoint
/// sharing its decoder, so it never qualifies as a table family.
#[test]
fn a_provider_from_parts_is_never_a_qualified_family() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    let snapshot = tiny_packed_capable_snapshot();
    let tokenizer = Tokenizer::from_file(snapshot.path().join("tokenizer.json")).unwrap();
    // A Mistral-shaped decoder: `Architecture::Llama`, exactly what `from_parts` receives.
    let model = tiny_packed_capable_model();
    assert_eq!(model.config().architecture, Architecture::Llama);
    let provider = LlamaProvider::from_parts(model, tokenizer, vec![99]);
    assert!(matches!(
        provider.plan_kv_cache(Policy::Qualified, 40_000, 64, 1, false),
        KvPlan::Dense(report) if report.fallback == Some(Reason::UnqualifiedModel)
    ));
}

/// sc-20682: on the generic fused-attention contract the compressed estimate differs from the
/// dense one by exactly the K/V term (dense K/V out, the format-derived compressed bytes in);
/// a contract that cannot run compressed keeps its dense estimate.
#[test]
fn compressed_admission_replaces_only_the_kv_term() {
    let geometry = LlmMemoryGeometry {
        query_heads: 24,
        kv_heads: 8,
        head_dim: 128,
        layers: 28,
        element_bytes: 2,
        score_element_bytes: 4,
        hidden_size: 3072,
        intermediate_size: 8192,
        vocab_size: 128_256,
        recurrent_bytes: 0,
    };
    let format = core_llm::KvCompressionFormat::GroupAffineK8V8;
    let (prompt, new) = (32_768_usize, 512_u32);
    let total = prompt as u64 + u64::from(new);
    let compressed_kv =
        crate::kv_policy::compressed_request_kv_bytes(format, &geometry, prompt as u64, total)
            .unwrap();
    // The MLX dense K/V: block-rounded buffers plus the window of growth copies.
    let dense_kv = crate::kv_policy::dense_final_kv_bytes(geometry.kv_shape(), total).unwrap();
    assert!(compressed_kv < dense_kv);
    let dense =
        estimate_mlx_request_bytes(prompt, new, geometry, 0, 0, MlxWorkspaceContract::Chunked)
            .unwrap();
    let compressed = estimate_mlx_compressed_request_bytes(
        prompt,
        new,
        geometry,
        0,
        0,
        MlxWorkspaceContract::Chunked,
        compressed_kv,
    )
    .unwrap();
    assert_eq!(dense - compressed, dense_kv - compressed_kv);
    assert_eq!(
        estimate_mlx_compressed_request_bytes(
            prompt,
            new,
            geometry,
            0,
            0,
            MlxWorkspaceContract::Eager,
            compressed_kv,
        ),
        estimate_mlx_request_bytes(prompt, new, geometry, 0, 0, MlxWorkspaceContract::Eager)
    );
}

/// A mid-generation admission refusal (a compressed cache's refused dense transition) reaches
/// the contract as the typed resource error, not a backend failure.
#[test]
fn a_typed_engine_refusal_is_the_contract_resource_error() {
    let evidence = core_llm::RequestResourceExhausted {
        prompt_tokens: 3,
        max_new_tokens: 2,
        max_context_tokens: 9,
        required_bytes: 8,
        available_bytes: 7,
    };
    assert!(matches!(
        to_core(crate::Error::ResourceExhausted(evidence)),
        CoreError::RequestResourceExhausted(mapped) if mapped == evidence
    ));
}

/// Restores the operational memory budget when a test that pins it ends.
struct MemoryBudget;

impl MemoryBudget {
    fn pin(bytes: u64) -> Self {
        std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, bytes.to_string());
        Self
    }
}

impl Drop for MemoryBudget {
    fn drop(&mut self) {
        std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
    }
}

/// E7 at load: the prefix cache's budget is what the load's admission leaves — never more —
/// so it cannot push a load past its admission budget, and a load that fits without it still
/// loads (with no cache) at the exact boundary.
#[test]
fn the_load_admits_the_prefix_cache_budget_up_to_its_headroom() {
    let dir = tiny_prefix_snapshot();
    let mut spec = LoadSpec::dense(dir.display().to_string());
    let required = crate::load_memory::required_bytes(&spec).unwrap();
    let headroom = 4096;
    spec.prefix_cache_bytes = Some(u64::MAX);
    {
        let _budget = MemoryOverride::set(required + headroom);
        let provider = LlamaProvider::load(&spec).unwrap();
        assert_eq!(provider.prefix_cache_budget(), headroom);
        assert_eq!(
            provider.load_report().unwrap().prefix_cache_bytes,
            Some(headroom),
            "the report names the settled budget"
        );
        spec.prefix_cache_bytes = Some(100);
        assert_eq!(
            LlamaProvider::load(&spec).unwrap().prefix_cache_budget(),
            100
        );
        spec.prefix_cache_bytes = None;
        assert_eq!(
            LlamaProvider::load(&spec).unwrap().prefix_cache_budget(),
            headroom.min(core_llm::defaults::MLX.prefix_cache_bytes)
        );
    }
    spec.prefix_cache_bytes = Some(u64::MAX);
    {
        // At the boundary the budget settles down to nothing rather than refusing.
        let _budget = MemoryOverride::set(required);
        let provider = LlamaProvider::load(&spec).unwrap();
        assert_eq!(provider.prefix_cache_budget(), 0);
        assert_eq!(provider.load_report().unwrap().prefix_cache_bytes, Some(0));
    }
    {
        let _budget = MemoryOverride::set(required - 1);
        assert!(LlamaProvider::load(&spec).is_err(), "past admission");
    }
}

/// The cache's part before any lookup: off with a zero budget, bypassed (named) for either
/// multimodal route, else a miss — the only request that reads or feeds it.
#[test]
fn the_prefix_route_is_off_bypassed_or_a_miss() {
    assert_eq!(prefix_path_for(false, false, false), ("off", None));
    assert_eq!(prefix_path_for(false, true, true), ("off", None));
    for (multimodal, gemma4_mm) in [(true, false), (false, true), (true, true)] {
        assert_eq!(
            prefix_path_for(true, multimodal, gemma4_mm),
            ("bypassed", Some(PREFIX_MULTIMODAL_BYPASS)),
            "multimodal {multimodal}, gemma4 {gemma4_mm}"
        );
    }
    assert_eq!(prefix_path_for(true, false, false), ("miss", None));
}

/// A load that settled a zero budget never reads or feeds the cache: no lookup, no store,
/// and every turn reports `off`.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn an_off_cache_is_never_read_or_fed() {
    use core_llm::Speculative;
    for provider in [causal_provider(), qwen35_mtp_provider()] {
        let provider = provider.with_prefix_budget(0);
        let user = Message::user("t3 t9 t4 t11");
        let (out1, _) = run(&provider, &chat(vec![user.clone()], Speculative::Off));
        let turn2 = chat(
            vec![user, Message::assistant(out1.text), Message::user("t5 t7")],
            Speculative::Off,
        );
        let report = run(&provider, &turn2).0.decode.unwrap();
        assert_eq!(report.prefix_cache.path, "off");
        assert_eq!(report.prefix_hit_tokens, 0);
        assert_eq!(provider.prefix_cache_stats(), PrefixStats::default());
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::an_off_cache_is_never_read_or_fed");
}

/// A tied-embedding decoder snapshot at `vocab × h` (h = 256: every projection a whole number
/// of Q4/Q8 groups), Gemma 2's soft caps and sandwich norms when `gemma`, with a word-level
/// tokenizer over the whole vocabulary.
fn promotion_snapshot(dir: &Path, model_type: &str, arch: &str, vocab: i32, gemma: bool) {
    let (h, softcap) = (256, gemma);
    let (inter, layers, heads, kv, hd) = (512, 2, 2, 1, 128);
    let mut cfg = json!({
        "architectures": [arch], "model_type": model_type, "hidden_size": h,
        "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": hd, "vocab_size": vocab, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": true, "max_position_embeddings": 8192,
    });
    if softcap {
        cfg["attn_logit_softcapping"] = json!(50.0);
        cfg["final_logit_softcapping"] = json!(30.0);
        cfg["query_pre_attn_scalar"] = json!(hd);
        cfg["sliding_window"] = json!(4096);
    }
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
    std::fs::write(
        dir.join("tokenizer.json"),
        format!(
            r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
            entries.join(", ")
        ),
    )
    .unwrap();
    let mut seed = 1u64;
    let mut r = |shape: &[i32]| {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
            })
            .collect();
        Array::from_slice(&v, shape)
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
    };
    let mut t: Vec<(String, Array)> = vec![
        ("model.embed_tokens.weight".into(), r(&[vocab, h])),
        ("model.norm.weight".into(), r(&[h])),
    ];
    for i in 0..layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        for n in ["input_layernorm", "post_attention_layernorm"] {
            t.push((p(&format!("{n}.weight")), r(&[h])));
        }
        if softcap {
            for n in ["pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                t.push((p(&format!("{n}.weight")), r(&[h])));
            }
        }
        t.push((p("self_attn.q_proj.weight"), r(&[heads * hd, h])));
        t.push((p("self_attn.k_proj.weight"), r(&[kv * hd, h])));
        t.push((p("self_attn.v_proj.weight"), r(&[kv * hd, h])));
        t.push((p("self_attn.o_proj.weight"), r(&[h, heads * hd])));
        t.push((p("mlp.gate_proj.weight"), r(&[inter, h])));
        t.push((p("mlp.up_proj.weight"), r(&[inter, h])));
        t.push((p("mlp.down_proj.weight"), r(&[h, inter])));
    }
    Array::save_safetensors(
        t.iter().map(|(k, v)| (k.as_str(), v)),
        None,
        dir.join("model.safetensors"),
    )
    .unwrap();
}

/// sc-24446 (d): a one-token request's MLX working set is priced, under either activation
/// dtype ([`crate::primitives::activation`]).
///
/// * **LLM decode** (every provider load): a Gemma GeGLU returns BF16, so the forward stays
///   BF16 and promotes nothing — `promoted_request_bytes` is 0 — and the measured
///   second-request working set on a tied 65,536-token fixture, dense and at load-time Q4,
///   stays within the request estimate, as a SwiGLU decoder's does. Measured both as the exact
///   settled `phys_footprint` peak growth (driver wake, MLX cache and host heap included) and
///   as MLX's active peak growth; the estimate covers the larger.
/// * **A role pinned to `f32`** (the LTX-2.5 text encoder): the GeGLU leaves the stream `f32`,
///   every dense matmul materializes an `f32` copy of its BF16 weight (the LM head's alone
///   vocabulary × hidden × 4 bytes), and the promotion term is what covers the measured
///   one-token forward — it exceeds the estimate without it.
///
/// The final hidden state is `f32` exactly when the model reports promoting activations.
///
/// MUTATION: return `Some(0)` from `Decoder::promoted_request_bytes` and the pinned cases go
/// RED; make `CausalLm::activations_promote` ignore the role and the LLM-decode Gemma cases do.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_one_token_request_working_set_is_priced_under_either_activation_dtype() {
    use crate::primitives::activation::ActivationRole;
    for (model_type, arch, gemma) in [
        ("llama", "LlamaForCausalLM", false),
        ("gemma2", "Gemma2ForCausalLM", true),
    ] {
        for quantize in [None, Some(Quantize::Q4)] {
            let dir = tempfile::tempdir().unwrap();
            promotion_snapshot(dir.path(), model_type, arch, 65_536, gemma);
            let mut spec = LoadSpec::dense(dir.path().display().to_string());
            spec.quantize = quantize;
            spec.prefix_cache_bytes = Some(0);
            let label = format!("{model_type} {quantize:?}");

            // LLM decode, through the provider.
            let provider = LlamaProvider::load(&spec).unwrap();
            let req = TextLlmRequest {
                messages: vec![Message::user("t3 t9 t4 t11")],
                sampling: Sampling::greedy(),
                max_new_tokens: 1,
                seed: Some(0),
                speculative: Some(core_llm::Speculative::Off),
                ..Default::default()
            };
            let prompt = rendered_ids(&provider, &req, true);
            let estimate = provider
                .speculative_request_bytes(SpeculativeRoute::Plain, None, prompt.len(), 1, 0)
                .unwrap();
            let Decoder::Causal(m) = &provider.model else {
                unreachable!("a causal fixture")
            };
            assert_hidden_dtype_matches_promotion(m, &label);
            assert!(!m.activations_promote(), "{label}: LLM decode promotes");
            assert_eq!(provider.model.promoted_request_bytes(), Some(0), "{label}");
            provider.generate(&req, &mut |_| {}).unwrap();
            let working_set = second_run_peak(|| {
                provider.generate(&req, &mut |_| {}).unwrap();
            });
            assert!(
                working_set.max() <= estimate,
                "{label}: working set {working_set:?} exceeds the estimate {estimate}"
            );

            // A role pinned to f32, on the decoder directly.
            if !gemma {
                continue;
            }
            let cfg_value = read_config_value(dir.path()).unwrap();
            let mut cfg = ModelConfig::from_json(&cfg_value).unwrap();
            cfg.activation_role = ActivationRole::LtxTextEncoder;
            let w = Weights::from_dir(dir.path()).unwrap();
            let quant = quantize.map(|q| quant_spec(q).unwrap());
            let pinned = Decoder::Causal(Box::new(
                CausalLm::from_weights_with(&w, "", cfg, quant).unwrap(),
            ));
            let Decoder::Causal(m) = &pinned else {
                unreachable!()
            };
            assert_hidden_dtype_matches_promotion(m, &label);
            assert!(m.activations_promote(), "{label}: the pinned role promotes");
            let promoted = pinned.promoted_request_bytes().unwrap();
            assert!(promoted > 0, "{label}");
            let ids = input_ids(&[3, 9, 4, 11]);
            let decoder_estimate = estimate_mlx_request_bytes(
                4,
                1,
                pinned.memory_geometry(),
                0,
                0,
                pinned.workspace_contract(),
            )
            .unwrap();
            let forward = || {
                m.decode_logits(&ids, &mut m.new_cache(), 0)
                    .unwrap()
                    .eval()
                    .unwrap();
            };
            forward();
            let working_set = second_run_peak(forward);
            assert!(
                working_set.max() <= decoder_estimate + promoted,
                "{label} pinned: working set {working_set:?} exceeds {}",
                decoder_estimate + promoted
            );
            // The promoted copies are GPU transients: MLX's active peak sees them even where
            // the settled footprint reuses the driver's pooled pages. Without the term, the
            // decoder's own tensors (the estimate less the driver's wake) do not cover them.
            assert!(
                working_set.active > decoder_estimate - MLX_REQUEST_WAKE_BYTES,
                "{label} pinned: the promotion term is not load-bearing ({working_set:?} vs \
                     {decoder_estimate})"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_one_token_request_working_set_is_priced_under_either_activation_dtype");
}

/// The final hidden state is `f32` exactly when `m` reports promoting activations.
fn assert_hidden_dtype_matches_promotion(m: &CausalLm, label: &str) {
    let hidden = m
        .hidden_states(&input_ids(&[3, 9, 4]), &mut m.new_cache(), 0)
        .unwrap();
    let dtype = hidden.last().unwrap().dtype();
    assert_eq!(
        dtype == mlx_rs::Dtype::Float32,
        m.activations_promote(),
        "{label}: activation dtype {dtype:?}"
    );
}

/// A request's working set two ways (the model already materialized by a first run): the
/// exact `phys_footprint` peak growth after the driver has settled — what macOS charges, MLX's
/// buffer cache, the host heap and the driver's wake included — and MLX's own peak active
/// growth, which still sees a GPU transient whose pages the driver recycled from its pool.
fn second_run_peak(run: impl Fn()) -> WorkingSet {
    let footprint = crate::test_fixture::footprint::peak_growth(&run);
    mlx_rs::memory::clear_cache();
    let before = mlx_rs::memory::get_active_memory();
    mlx_rs::memory::reset_peak_memory();
    run();
    let active = (mlx_rs::memory::get_peak_memory() - before) as u64;
    WorkingSet { footprint, active }
}

#[derive(Debug)]
struct WorkingSet {
    footprint: u64,
    active: u64,
}

impl WorkingSet {
    fn max(&self) -> u64 {
        self.footprint.max(self.active)
    }
}

/// sc-24446: the request estimate covers every one-token request working set the 2026-10-01
/// exact-peak re-probes measured on real weights — the kernel's `phys_footprint` maximum over
/// a second identical one-token "Hi" request after the driver settled (MLX's cache, the host
/// heap and the driver's wake included), on the current build: BF16 GeGLU for every LLM
/// decode path, and the `f32` GeGLU (a role pinned to it, with the promotion term) for the
/// Gemma rows marked so. Recorded evidence, not a machine golden: the relation is pinned.
///
/// MUTATION: drop the driver's wake from the request estimate and the BF16 rows go RED.
#[test]
fn the_request_estimate_covers_the_exact_footprint_working_sets() {
    let geometry = |q, kv, hd, layers, hidden, inter, vocab, recurrent| LlmMemoryGeometry {
        query_heads: q,
        kv_heads: kv,
        head_dim: hd,
        layers,
        element_bytes: 4,
        score_element_bytes: 4,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: vocab,
        recurrent_bytes: recurrent,
    };
    let llama = geometry(32, 8, 64, 16, 2048, 8192, 128_256, 0);
    let qwen3_small = geometry(16, 8, 128, 28, 2048, 6144, 151_936, 0);
    let qwen3_8b = geometry(32, 8, 128, 36, 4096, 12_288, 151_936, 0);
    let gemma2 = geometry(8, 4, 256, 26, 2304, 9216, 256_000, 0);
    let gemma4 = geometry(16, 8, 256, 48, 3840, 15_360, 262_144, 0);
    let qwen38 = frozen_dense_qwen35_config();
    let qwen38_geometry = geometry(
        24,
        4,
        256,
        64,
        5120,
        17_408,
        248_320,
        64 * 48 * 128 * (128 + 4) * 4,
    );
    let qwen35 = MlxWorkspaceContract::Qwen35 {
        config: &qwen38,
        prism: false,
    };
    let qwen36_inter = Qwen35Config::from_json(
        &serde_json::from_str::<serde_json::Value>(include_str!(
            "../../../testdata/load_admission/qwen3.6-35b-a3b.json"
        ))
        .unwrap()["config"],
    )
    .unwrap()
    .intermediate_size as u64;
    let qwen36 = geometry(
        16,
        2,
        256,
        40,
        2048,
        qwen36_inter,
        248_320,
        40 * 32 * 128 * (128 + 4) * 4,
    );
    let (g2_head, g2_mlp) = (256_000 * 2304, 9216 * 2304);
    let (g4_head, g4_mlp) = (262_144 * 3840, 15_360 * 3840);
    let promoted = |head, largest| promoted_weight_bytes(head, largest).unwrap();
    use MlxWorkspaceContract::{Chunked, Eager};
    // (row, geometry, contract, rendered prompt tokens, promoted bytes, measured footprint)
    let cases = [
        ("Llama 3.2 1B BF16", llama, Chunked, 36, 0, 171_000_000u64),
        ("Llama 3.2 1B Q4/Q8", llama, Chunked, 36, 0, 226_000_000),
        ("Qwen3-1.7B", qwen3_small, Chunked, 13, 0, 237_000_000),
        ("Qwen3-8B", qwen3_8b, Chunked, 13, 0, 269_000_000),
        (
            "Gemma 2 2B-it (BF16 GeGLU)",
            gemma2,
            Eager,
            10,
            0,
            195_000_000,
        ),
        (
            "Gemma 4 enhancer (BF16 GeGLU)",
            gemma4,
            Eager,
            14,
            0,
            295_000_000,
        ),
        ("Qwen3.8-27B", qwen38_geometry, qwen35, 13, 0, 382_000_000),
        ("Qwen3.6-35B-A3B", qwen36, Eager, 13, 0, 251_000_000),
        (
            "Gemma 2 2B-it BF16 (f32 GeGLU)",
            gemma2,
            Eager,
            10,
            promoted(g2_head, g2_mlp),
            3_811_000_000,
        ),
        (
            "Gemma 2 2B-it Q4 (f32 GeGLU)",
            gemma2,
            Eager,
            10,
            promoted(g2_head, g2_mlp / 64 * 2),
            2_668_000_000,
        ),
        (
            "Gemma 4 enhancer BF16 (f32 GeGLU)",
            gemma4,
            Eager,
            14,
            promoted(g4_head, g4_mlp),
            6_411_000_000,
        ),
        (
            "Gemma 4 enhancer Q4 (f32 GeGLU)",
            gemma4,
            Eager,
            14,
            promoted(g4_head, g4_mlp / 64 * 2),
            4_659_000_000,
        ),
    ];
    for (row, geometry, contract, prompt, promoted, measured) in cases {
        let estimate =
            estimate_mlx_request_bytes(prompt, 1, geometry, 0, 0, contract).unwrap() + promoted;
        assert!(
            estimate >= measured,
            "{row}: estimate {estimate} < measured footprint {measured}"
        );
    }
}

/// sc-24446: the K/V caches hold whole 256-position blocks — over the committed positions and
/// over any verify width written past them — and a prompt-lookup / draft overshoot that
/// crosses into a new block charges the rest of that block.
///
/// MUTATION: multiply the runtime K/V padding by zero, or drop `kv_overshoot_padding`, and
/// this goes RED.
#[test]
fn kv_block_padding_covers_committed_and_verify_positions() {
    let g = LlmMemoryGeometry {
        query_heads: 8,
        kv_heads: 4,
        head_dim: 64,
        layers: 6,
        element_bytes: 4,
        score_element_bytes: 4,
        hidden_size: 512,
        intermediate_size: 1024,
        vocab_size: 4096,
        recurrent_bytes: 0,
    };
    let kv_position = 6 * 4 * 64 * 4 * 2;
    let runtime =
        |committed, width| mlx_runtime_request_bytes(8, committed, width, 8, g, true).unwrap();
    // 10 committed positions allocate a whole block: 246 more than 256 committed do.
    assert_eq!(runtime(10, 0) - runtime(256, 0), 246 * kv_position);
    // Three verify positions past 255 committed: the three and the new block's rest.
    assert_eq!(
        runtime(255, 3) - runtime(255, 0),
        (3 + 254 - 1) * kv_position
    );
    assert_eq!(kv_overshoot_padding(255, 3), Some(512 - (256 + 3)));
    assert_eq!(kv_overshoot_padding(100, 3), Some(0));
}

/// sc-24446: a prompt-lookup verify step whose overshoot crosses into a new K/V block is
/// charged that block's rest, beyond the overshoot positions themselves.
///
/// MUTATION: drop `kv_overshoot_padding` from `speculative_request_bytes` and this goes RED.
#[test]
fn a_lookup_overshoot_into_a_new_block_charges_the_block() {
    let provider = causal_provider();
    let g = provider.model.memory_geometry();
    let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
    let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
    let price = |route| {
        provider
            .speculative_request_bytes(route, None, 200, 50, 0)
            .unwrap()
    };
    let off = price(SpeculativeRoute::Plain);
    let lookup = price(SpeculativeRoute::PromptLookup { width: 8 });
    // 250 committed positions + 8 overshoot = 258: a second block, 248 positions beyond.
    assert_eq!(lookup - off, 8 * (kv_position + rows) + 248 * kv_position);
}

/// sc-24446: a decoder whose `param_groups` leaves a source it read unconsumed is refused at
/// load — the source would stay resident outside the admitted bound.
///
/// MUTATION: drop the `leftover > 0` refusal from `materialize_decoder` and this goes RED.
#[test]
fn an_unconsumed_source_is_a_load_error() {
    let dir = crate::test_fixture::Fixture::new("mlx-llm-leftover-", None);
    let path = dir.join("model.safetensors");
    let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
    Array::save_safetensors([("used", &a), ("orphan", &a)], None, &path).unwrap();
    let mut w = Weights::from_file(&path).unwrap();
    let used = w
        .require("used")
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
    w.require("orphan").unwrap();
    let err = materialize_decoder(&mut w, &[vec![used]]).unwrap_err();
    assert!(err.to_string().contains("load materialization"), "{err}");
}

/// sc-24446: the in-flight window prices each command buffer's crossing op — a long eager
/// prefill's `prompt × prompt` score block per head, or a `prompt × inter` MLP activation —
/// not just the 50 MB per-buffer cap.
///
/// MUTATION: drop the crossing op from the window and this goes RED.
#[test]
fn the_in_flight_window_prices_each_buffers_crossing_op() {
    let g = LlmMemoryGeometry {
        query_heads: 8,
        kv_heads: 4,
        head_dim: 256,
        layers: 26,
        element_bytes: 4,
        score_element_bytes: 4,
        hidden_size: 2304,
        intermediate_size: 9216,
        vocab_size: 256_000,
        recurrent_bytes: 0,
    };
    let prompt = 8192u64;
    let committed = prompt + 1;
    let fixed = (round_up(committed, 256).unwrap() - committed) * (26 * 4 * 256 * 4 * 2)
        + MLX_EVAL_BUFFER_WINDOW * MLX_MAX_OPS_PER_BUFFER * MLX_ALLOCATION_PAGE_BYTES
        + MLX_REQUEST_WAKE_BYTES;
    let in_flight =
        mlx_runtime_request_bytes(prompt as usize, committed, 0, prompt, g, true).unwrap() - fixed;
    let scores = prompt * prompt * 8 * 4;
    assert!(
        scores > prompt * 9216 * 4,
        "the eager score block is the crossing op"
    );
    assert_eq!(
        in_flight,
        MLX_EVAL_BUFFER_WINDOW * (MLX_MAX_BUFFER_BYTES + scores)
    );
    assert!(in_flight > 10 * MLX_EVAL_BUFFER_WINDOW * MLX_MAX_BUFFER_BYTES);
}

/// The request estimate covers every one-token request working set the sc-24446 guarded
/// probes recorded on real weights (`native-memory-admission.md`: a second identical
/// one-token "Hi" request on the materialized model), computed from each model's geometry
/// and rendered prompt length. Recorded evidence, not a machine golden: the relation is
/// pinned. Two limits, both stated rather than hidden:
///
/// * the probes recorded MLX's **active** peak, not `phys_footprint` (cache retention and
///   host heap uncounted); the exact footprint working sets are pinned by
///   `the_request_estimate_covers_the_exact_footprint_working_sets`;
/// * every Gemma row ran the `f32` GeGLU, so it is pinned with the promotion term — today
///   that is the path a role pinned to `f32` (the LTX-2.5 text encoder) runs.
///
/// Before sc-24446 the estimate was 6–9 MB for the Llama / Qwen3 cases (measured 35–87 MB)
/// and 14–31 MB for the Gemma cases (measured 2.4–5.4 GB).
#[test]
fn the_request_estimate_covers_the_recorded_one_token_working_sets() {
    let geometry = |q, kv, hd, layers, hidden, inter, vocab, recurrent| LlmMemoryGeometry {
        query_heads: q,
        kv_heads: kv,
        head_dim: hd,
        layers,
        element_bytes: 4,
        score_element_bytes: 4,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: vocab,
        recurrent_bytes: recurrent,
    };
    let llama = geometry(32, 8, 64, 16, 2048, 8192, 128_256, 0);
    let qwen3_small = geometry(16, 8, 128, 28, 2048, 6144, 151_936, 0);
    let qwen3_8b = geometry(32, 8, 128, 36, 4096, 12_288, 151_936, 0);
    let gemma2 = geometry(8, 4, 256, 26, 2304, 9216, 256_000, 0);
    let gemma4 = geometry(16, 8, 256, 48, 3840, 15_360, 262_144, 0);
    let qwen38 = frozen_dense_qwen35_config();
    let qwen38_geometry = geometry(
        24,
        4,
        256,
        64,
        5120,
        17_408,
        248_320,
        64 * 48 * 128 * (128 + 4) * 4,
    );
    let qwen35 = MlxWorkspaceContract::Qwen35 {
        config: &qwen38,
        prism: false,
    };
    let (g2_head, g2_mlp) = (256_000 * 2304, 9216 * 2304);
    let (g4_head, g4_mlp) = (262_144 * 3840, 15_360 * 3840);
    let promoted = |head, largest| promoted_weight_bytes(head, largest).unwrap();
    // (config, geometry, contract, rendered prompt tokens, promoted bytes, measured bytes)
    let cases = [
        (
            "Llama 3.2 1B",
            llama,
            MlxWorkspaceContract::Chunked,
            36,
            0,
            41_000_000u64,
        ),
        (
            "Qwen3-1.7B",
            qwen3_small,
            MlxWorkspaceContract::Chunked,
            13,
            0,
            46_000_000,
        ),
        (
            "Qwen3-8B",
            qwen3_8b,
            MlxWorkspaceContract::Chunked,
            13,
            0,
            87_000_000,
        ),
        ("Qwen3.8-27B", qwen38_geometry, qwen35, 13, 0, 236_000_000),
        (
            "Gemma 2 2B-it BF16",
            gemma2,
            MlxWorkspaceContract::Eager,
            10,
            promoted(g2_head, g2_mlp),
            3_528_000_000,
        ),
        (
            "Gemma 2 2B-it Q4",
            gemma2,
            MlxWorkspaceContract::Eager,
            10,
            promoted(g2_head, g2_mlp / 64 * 2),
            2_426_000_000,
        ),
        (
            "Gemma 4 enhancer BF16",
            gemma4,
            MlxWorkspaceContract::Eager,
            14,
            promoted(g4_head, g4_mlp),
            5_416_000_000,
        ),
        (
            "Gemma 4 enhancer Q4",
            gemma4,
            MlxWorkspaceContract::Eager,
            14,
            promoted(g4_head, g4_mlp / 64 * 2),
            4_466_000_000,
        ),
    ];
    for (config, geometry, contract, prompt, promoted, measured) in cases {
        let estimate =
            estimate_mlx_request_bytes(prompt, 1, geometry, 0, 0, contract).unwrap() + promoted;
        assert!(
            estimate >= measured,
            "{config}: estimate {estimate} < measured {measured}"
        );
    }
    assert!(promoted_weight_bytes(u64::MAX, 1).is_none());
}

/// E7 per request: held entries are reclaimable. A request short by exactly what the cache
/// holds is admitted after evicting it; one byte shorter still is refused and leaves the
/// cache as it was.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_request_reclaims_the_prefix_cache_up_to_the_admission_boundary() {
    use core_llm::Speculative;
    let provider = causal_provider();
    run(
        &provider,
        &chat(vec![Message::user("t3 t9 t4")], Speculative::Off),
    );
    let held = provider.prefix_cache_resident_bytes();
    assert!(held > 0);
    let req = chat(vec![Message::user("t11 t12 t13 t14")], Speculative::Off);
    let prompt = rendered_ids(&provider, &req, true);
    let required = provider
        .speculative_request_bytes(
            SpeculativeRoute::Plain,
            None,
            prompt.len(),
            req.max_new_tokens,
            0,
        )
        .unwrap();
    assert!(
        required > held,
        "the test needs a shortfall the cache can cover: {required} vs {held}"
    );

    {
        let _budget = MemoryOverride::set(required - held - 1);
        let err = provider.generate(&req, &mut |_| {}).unwrap_err();
        assert!(
            matches!(err, CoreError::RequestResourceExhausted(_)),
            "{err}"
        );
        assert_eq!(
            provider.prefix_cache_resident_bytes(),
            held,
            "kept on refusal"
        );
    }
    {
        let _budget = MemoryOverride::set(required - held);
        let before = provider.prefix_cache_stats().evicted;
        provider.generate(&req, &mut |_| {}).unwrap();
        assert_eq!(
            provider.prefix_cache_stats().evicted - before,
            1,
            "reclaimed"
        );
        assert_eq!(
            provider.prefix.borrow().len(),
            1,
            "only this request's entry"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_request_reclaims_the_prefix_cache_up_to_the_admission_boundary");
}

/// E7 for the snapshot a hybrid request leaves behind: admitted with the request when the
/// memory holds both, and at one byte short the request still runs — without keeping it.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_qwen35_request_keeps_its_snapshot_only_when_admission_holds_it() {
    use core_llm::Speculative;
    let req = chat(vec![Message::user("t3 t9 t4 t11")], Speculative::Off);
    let probe = qwen35_mtp_provider();
    let prompt = rendered_ids(&probe, &req, true);
    let boundary = rendered_ids(&probe, &req, false).len();
    let required = probe
        .speculative_request_bytes(
            SpeculativeRoute::Plain,
            None,
            prompt.len(),
            req.max_new_tokens,
            0,
        )
        .unwrap();
    let snapshot = probe.prefix_snapshot_bytes(Some(boundary), false).unwrap();
    {
        let provider = qwen35_mtp_provider();
        let _budget = MemoryOverride::set(required + snapshot - 1);
        let out = provider.generate(&req, &mut |_| {}).unwrap();
        let report = out.decode.unwrap();
        assert_eq!(report.prefix_hit_tokens, 0);
        assert!(provider.prefix.borrow().is_empty(), "no room: nothing kept");
        let reason = report.prefix_cache.reason.unwrap();
        assert!(reason.starts_with("not kept: admission"), "{reason}");
    }
    {
        let provider = qwen35_mtp_provider();
        let _budget = MemoryOverride::set(required + snapshot);
        provider.generate(&req, &mut |_| {}).unwrap();
        assert_eq!(
            provider.prefix.borrow().keys(),
            vec![prompt[..boundary].to_vec()]
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_qwen35_request_keeps_its_snapshot_only_when_admission_holds_it");
}

/// sc-24446 (E5): a provider loads with the MLX defaults-table speculative default, and a
/// request that leaves the option unset runs whatever the provider's default is — here set to
/// `auto` so the hook is observable — while an explicit off (new or legacy spelling) wins.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn an_unset_speculative_option_runs_the_providers_default() {
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    let mut provider =
        LlamaProvider::load(&LoadSpec::dense(fixture.target.to_string_lossy())).unwrap();
    assert_eq!(
        provider.speculative_default(),
        core_llm::defaults::MLX.speculative,
        "loaded with the defaults table's default"
    );
    let prompt = &core_llm_testkit::draft_model_prompts()[0];
    core_llm_testkit::check_speculative_default(
        &provider,
        core_llm::defaults::MLX.speculative,
        prompt,
        8,
    );
    provider.set_speculative_default(core_llm::Speculative::Auto);
    core_llm_testkit::check_speculative_default(&provider, core_llm::Speculative::Auto, prompt, 8);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::an_unset_speculative_option_runs_the_providers_default");
}

/// An 8-bit reader for the K8V8 identity whose backend the cache refuses to bind.
#[derive(Debug)]
struct UnbindableReader;

impl crate::primitives::RetainedPackedKernel for UnbindableReader {
    fn cache_identity(&self) -> &str {
        crate::primitives::PACKED_METAL_B8_IDENTITY
    }
    fn backend(&self) -> &str {
        "cpu"
    }
    fn retained_host_bytes_estimate(&self) -> usize {
        0
    }
    fn dispatch(
        &self,
        _args: &crate::primitives::PackedAttentionArgs<'_>,
    ) -> crate::error::Result<Array> {
        Err(crate::error::Error::Msg("never dispatched".into()))
    }
    fn code_bits(&self) -> crate::primitives::PackedCodeBits {
        crate::primitives::PackedCodeBits::Eight
    }
}

/// sc-20682 AC2 through the production entry point: under a budget that holds the compressed
/// estimate but not the dense one, the qualified request is admitted and runs compressed,
/// while the same request un-opted is refused at the dense price; a request whose selection
/// then refuses the compressed cache is admitted again at the dense price before it runs.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn admission_prices_compressed_kv_only_when_the_request_runs_it() {
    use core_llm::{KvCompressionFormat as Format, KvCompressionPolicy as Policy};
    let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == core_llm::KvModelFamily::Qwen3)
        .unwrap();
    let words = usize::try_from(row.min_context_tokens).unwrap();
    // As deep as the evidence model (28 layers): the compressed cache's per-layer transients
    // amortize over the depth, so a two-layer toy prices above dense.
    let snapshot = tiny_snapshot(
        json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "num_hidden_layers": 28,
        }),
        row.min_context_tokens + 1024,
        true,
    );
    // No cross-turn prefix cache: its held entries are reclaimable memory admission would
    // evict to fit (sc-24437, E7), and this pins the budget against the request alone.
    let without_prefix_cache = |provider: LlamaProvider| {
        *provider.prefix.borrow_mut() = PrefixCache::with_budget(0);
        provider
    };
    let provider = without_prefix_cache(load_tiny(&snapshot));
    const NEW: u32 = 8;
    let prompt = kv_generate(&provider, words, Policy::Off, NEW)
        .usage
        .prompt_tokens as usize;
    let estimate = |format| {
        provider
            .admission_estimate(format, prompt, NEW, 0, 0)
            .unwrap()
    };
    let (compressed, dense) = (estimate(Some(Format::GroupAffineK8V8)), estimate(None));
    assert!(compressed < dense, "{compressed} vs {dense}");

    let budget = MemoryBudget::pin(compressed);
    let output = kv_generate(&provider, words, Policy::Qualified, NEW);
    assert!(output.kv_cache.as_ref().unwrap().ran_compressed());
    let refuse = |provider: &LlamaProvider, policy| {
        let request = TextLlmRequest {
            messages: vec![Message::text(
                Role::User,
                (0..words)
                    .map(|i| format!("w{}", (i * 7) % 26 + 6))
                    .collect::<Vec<_>>()
                    .join(" "),
            )],
            sampling: core_llm::Sampling::greedy(),
            max_new_tokens: NEW,
            seed: Some(0),
            kv_compression: policy,
            ..Default::default()
        };
        match provider.generate(&request, &mut |_| {}) {
            Err(CoreError::RequestResourceExhausted(evidence)) => evidence,
            other => panic!("expected a dense-priced refusal, got {other:?}"),
        }
    };
    let evidence = refuse(&provider, Policy::Off);
    assert_eq!(
        (evidence.required_bytes, evidence.available_bytes),
        (dense, compressed)
    );

    let unbindable = without_prefix_cache(load_tiny(&snapshot));
    #[allow(clippy::arc_with_non_send_sync)]
    let reader =
        crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(UnbindableReader));
    assert!(unbindable.kv_reader.set(Ok(reader)).is_ok());
    let evidence = refuse(&unbindable, Policy::Qualified);
    assert_eq!(evidence.required_bytes, dense);
    drop(budget);
    let output = kv_generate(&unbindable, words, Policy::Qualified, NEW);
    let report = output.kv_cache.unwrap();
    assert_eq!(
        report.fallback,
        Some(core_llm::KvCacheFallbackReason::ReaderUnavailable)
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::admission_prices_compressed_kv_only_when_the_request_runs_it");
}

/// A tiny hybrid (Qwen3.5-style GatedDeltaNet + full attention) snapshot the provider loads
/// as its hybrid decoder.
fn tiny_hybrid_snapshot() -> tempfile::TempDir {
    let dir = tiny_snapshot(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        2048,
        true,
    );
    let (tensors, config) = crate::snapshot::tests::tiny_qwen35(false);
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();
    dir
}

/// sc-20683 AC2: the cross-backend compressed-KV conformance table
/// (`core_llm_testkit::kv_policy_cases`) through MLX's production plan, with the fused reader:
/// the same table Candle's plan passes without one. Each case plans on a decoder the provider
/// actually loaded — a plain one the fused reader reads, one outside its geometry (head
/// dimension 96) and a hybrid recurrent one — test-armed as the case's table family (no tiny
/// fixture is a measured architecture: unarmed, every one has no family).
#[test]
fn mlx_kv_plan_conforms_to_the_cross_backend_policy_table() {
    use core_llm_testkit::{kv_policy_conformance, KvBackendDecision, KvCaseDecoder, KvReader};
    let load = |identity, qk_norm| load_tiny_unarmed(&tiny_snapshot(identity, 2048, qk_norm));
    let mut plain = load(
        json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        true,
    );
    let mut outside_geometry = load(
        json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "head_dim": 96}),
        false,
    );
    let mut hybrid = load_tiny_unarmed(&tiny_hybrid_snapshot());
    assert!(matches!(hybrid.model, Decoder::Qwen35(_)));
    for provider in [&plain, &outside_geometry, &hybrid] {
        assert_eq!(
            provider.kv_family, None,
            "no tiny fixture is a measured model"
        );
    }
    kv_policy_conformance(KvReader::Fused, |case| {
        let provider = match case.decoder {
            KvCaseDecoder::Plain => &mut plain,
            KvCaseDecoder::UnsupportedGeometry => &mut outside_geometry,
            KvCaseDecoder::Hybrid => &mut hybrid,
        };
        provider.kv_family = case.family;
        match provider.plan_kv_cache(
            case.policy,
            usize::try_from(case.context_tokens).unwrap(),
            case.max_new_tokens,
            case.batch,
            case.multimodal,
        ) {
            KvPlan::Compressed { format, .. } => KvBackendDecision::Compressed(format),
            KvPlan::Dense(report) => KvBackendDecision::Dense(report),
            KvPlan::Unreported => panic!("a product plan always reports"),
        }
    });
}

/// sc-20688 review (finding 9): the compressed-cache fingerprint a prefix store keys on tells
/// apart loads that the config alone does not — a Q4 and a Q8 load of the same weights, and
/// weights replaced at the same path — while a reload of unchanged weights keeps it.
#[test]
fn the_cache_fingerprint_names_the_load_quantization_and_the_weights() {
    let snapshot = tiny_snapshot(
        json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        2048,
        false,
    );
    let fingerprint = |quantize| {
        let provider = LlamaProvider::load(&core_llm::LoadSpec {
            quantize,
            ..core_llm::LoadSpec::dense(snapshot.path().to_string_lossy().to_string())
        })
        .unwrap();
        let Decoder::Causal(model) = &provider.model else {
            panic!("a causal decoder");
        };
        model.cache_fingerprint()
    };
    let dense = fingerprint(None);
    let q4 = fingerprint(Some(core_llm::Quantize::Q4));
    let q8 = fingerprint(Some(core_llm::Quantize::Q8));
    assert_ne!(q4, q8, "Q4 and Q8 loads of the same weights");
    assert_ne!(dense, q4);
    assert_eq!(
        fingerprint(None),
        dense,
        "an unchanged reload keeps its fingerprint"
    );
    // Replace the weights at the same path (another draw, so another size and mtime).
    let replacement = tiny_snapshot_with(
        json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        2048,
        false,
        CONTEXT_SENSITIVE_GAINS,
    );
    let weights = snapshot.path().join("model.safetensors");
    let before = std::fs::metadata(&weights).unwrap().modified().unwrap();
    std::fs::copy(replacement.path().join("model.safetensors"), &weights).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&weights)
        .unwrap();
    file.set_modified(before + std::time::Duration::from_secs(1))
        .unwrap();
    drop(file);
    assert_ne!(
        fingerprint(None),
        dense,
        "replaced weights at the same path"
    );
    // The identity alone moves the fingerprint of an otherwise identical decoder.
    let model = tiny_causal_model(4, 2, 64);
    let base = model.cache_fingerprint();
    let named = tiny_causal_model(4, 2, 64).with_weights_identity("model.safetensors:1:2");
    assert_ne!(named.cache_fingerprint(), base);
}

/// sc-20688 review (finding 2): a production load plans as a table family only for a measured
/// architecture. The tiny Llama and Qwen3 fixtures dispatch as those families (their campaign
/// identity) but are not Llama-3.2-3B or Qwen3-1.7B, so an opted-in request at a qualified
/// context reports `UnqualifiedModel` — as a Llama-3.1-8B or Qwen3-8B load does
/// (`core_llm::qualified_kv_model_family`, tested on their configs in core-llm).
#[test]
fn a_load_of_an_unmeasured_architecture_has_no_table_family() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    for (identity, qk_norm, family) in [
        (
            json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
            false,
            "llama",
        ),
        (
            json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            true,
            "qwen",
        ),
    ] {
        let provider = load_tiny_unarmed(&tiny_snapshot(identity, 40_960, qk_norm));
        assert_eq!(provider.campaign_family, Some(family));
        assert_eq!(provider.kv_family, None);
        let KvPlan::Dense(report) = provider.plan_kv_cache(Policy::Qualified, 20_000, 64, 1, false)
        else {
            panic!("an unmeasured architecture never plans compressed");
        };
        assert_eq!(
            report,
            core_llm::KvCacheReport::dense(Reason::UnqualifiedModel, None)
        );
    }
}

/// sc-20682 review: the admission estimate covers the MLX allocator's measured peak of the
/// same request, through the prompt step AND a decode that crosses several 256-token growth
/// blocks, on a KV-dominant decoder (28 layers of 8 KV heads x 128 over a 128-wide residual
/// stream, so K/V rather than activations set the peak). Both the compressed run and the same
/// request run dense (the price a dense re-admission charges) must fit their estimates.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn admission_estimates_cover_the_measured_peak_compressed_and_dense() {
    use core_llm::{KvCompressionFormat as Format, KvCompressionPolicy as Policy};
    use mlx_rs::memory;
    let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == core_llm::KvModelFamily::Qwen3)
        .unwrap();
    let words = usize::try_from(row.min_context_tokens).unwrap();
    let snapshot = tiny_snapshot(
        json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "num_hidden_layers": 28, "num_attention_heads": 8,
            "num_key_value_heads": 8, "head_dim": 128,
        }),
        row.min_context_tokens + 2048,
        true,
    );
    let provider = load_tiny(&snapshot);
    const NEW: u32 = 600;
    // Warm both paths (kernel compilation, allocator pools) before measuring.
    let prompt = kv_generate(&provider, words, Policy::Qualified, 4)
        .usage
        .prompt_tokens as usize;
    kv_generate(&provider, words, Policy::Off, 4);
    for (policy, format) in [
        (Policy::Qualified, Some(Format::GroupAffineK8V8)),
        (Policy::Off, None),
    ] {
        memory::clear_cache();
        let baseline = memory::get_active_memory() as u64;
        memory::reset_peak_memory();
        let output = kv_generate(&provider, words, policy, NEW);
        let peak = (memory::get_peak_memory() as u64).saturating_sub(baseline);
        assert_eq!(output.usage.generated_tokens, NEW);
        assert_eq!(
            output.kv_cache.as_ref().unwrap().ran_compressed(),
            format.is_some()
        );
        let estimate = provider
            .admission_estimate(format, prompt, NEW, 0, 0)
            .unwrap();
        eprintln!("sc-20682 {policy:?}: measured peak {peak} B, estimate {estimate} B");
        assert!(
            peak <= estimate,
            "{policy:?}: measured peak {peak} B exceeds the admission estimate {estimate} B"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::admission_estimates_cover_the_measured_peak_compressed_and_dense");
}

/// A reader that binds to the K8V8 cache and faults on every dispatch.
#[derive(Debug)]
struct FaultingReader;

impl crate::primitives::RetainedPackedKernel for FaultingReader {
    fn cache_identity(&self) -> &str {
        crate::primitives::PACKED_METAL_B8_IDENTITY
    }
    fn backend(&self) -> &str {
        "mlx-metal"
    }
    fn retained_host_bytes_estimate(&self) -> usize {
        0
    }
    fn dispatch(
        &self,
        _args: &crate::primitives::PackedAttentionArgs<'_>,
    ) -> crate::error::Result<Array> {
        Err(crate::error::Error::Msg("injected dispatch fault".into()))
    }
    fn code_bits(&self) -> crate::primitives::PackedCodeBits {
        crate::primitives::PackedCodeBits::Eight
    }
}

/// AC1: a compressed generation that the cache moves to dense part-way, or whose reader the
/// selection refuses, reports the dense outcome with its reason — never a compressed label.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_reader_fault_or_refusal_is_reported_as_a_dense_fallback() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionFormat as Format};
    let model = tiny_packed_capable_model();
    let prompt = (0..40).map(|i| i % 31 + 1).collect::<Vec<i32>>();
    let config = GenerationConfig {
        max_new_tokens: 4,
        seed: Some(0),
        ..Default::default()
    };
    let run = |reader: crate::primitives::CompiledKernelHandle| {
        let (mut cache, refused) = crate::kv_policy::select_compressed_cache(
            &model,
            reader,
            prompt.len(),
            admit_any_transition(),
        );
        // The production path: the engine's plain loop on the selected cache.
        crate::decode::generate_prompt_lookup_on(
            &model,
            cache.as_mut(),
            &prompt,
            &config,
            &crate::decode::SpeculativeConfig {
                max_ngram: 3,
                num_draft: 0,
            },
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        crate::kv_policy::compressed_report(Format::GroupAffineK8V8, refused, cache.as_ref())
            .unwrap()
    };

    #[allow(clippy::arc_with_non_send_sync)]
    let faulting =
        crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(FaultingReader));
    let report = run(faulting);
    assert_eq!(report.format, Some(Format::GroupAffineK8V8));
    assert_eq!(report.fallback, Some(Reason::RuntimeFallback), "{report:?}");
    assert!(!report.ran_compressed());
    assert!(report.counters.dense_fallback_events > 0);
    assert!(
        report
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("dispatch")),
        "{report:?}"
    );

    let refused = crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
        crate::primitives::OpaqueCompiledKernel::new(
            "sc20679-refused",
            "cpu",
            0,
            std::sync::Arc::new(()),
        ),
    ));
    let report = run(refused);
    assert_eq!(report.format, None);
    assert_eq!(
        report.fallback,
        Some(Reason::ReaderUnavailable),
        "{report:?}"
    );
    assert!(report.detail.is_some());

    let healthy =
        crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight).unwrap();
    assert!(run(healthy.clone()).ran_compressed());

    // sc-20688 review (finding 7): a decoder outside the fused reader's geometry (head
    // dimension 32) is refused with its true reason, `UnsupportedGeometry` and the shared
    // stage's detail — not `ReaderUnavailable` — even with a healthy reader.
    let narrow = tiny_causal_model(4, 2, 32);
    let (cache, refused) = crate::kv_policy::select_compressed_cache(
        &narrow,
        healthy,
        prompt.len(),
        admit_any_transition(),
    );
    let report =
        crate::kv_policy::compressed_report(Format::GroupAffineK8V8, refused, cache.as_ref())
            .unwrap();
    assert_eq!(
        report,
        core_llm::KvCacheReport::dense(
            Reason::UnsupportedGeometry,
            crate::kv_policy::geometry_refusal(narrow.config())
        )
    );
    assert!(report
        .detail
        .as_deref()
        .is_some_and(|detail| detail.contains("head dimension")));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::a_reader_fault_or_refusal_is_reported_as_a_dense_fallback");
}

/// AC2: no dense K/V mirror survives an attention call on the compressed path. After the
/// prefill and after every decode step, the compressed cache's dense fallback owns no K/V
/// (handle accounting), and MLX active memory above the pre-request baseline is the compressed
/// representation's own device bytes (byte accounting) — far below what any one layer's dense
/// K or V of the same tokens would add.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn no_dense_kv_mirror_survives_a_compressed_attention_call() {
    use mlx_rs::memory;
    // Four K/V heads of 128 make one layer's dense K (1.5 MB at 1536 bf16 tokens) dwarf the
    // accounting slack.
    let model = tiny_causal_model(4, 4, 128);
    let reader =
        crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight).unwrap();
    let prompt = (0..1536).map(|i| i % 31 + 1).collect::<Vec<i32>>();
    let step = |cache: &mut dyn KvCache, ids: &[i32], offset: usize| {
        let logits = model.step(&input_ids(ids), cache, offset as i32).unwrap();
        logits.eval().unwrap();
    };
    // Warm every lazily retained model and reader state on a throwaway cache first.
    {
        let (mut warm, refused) = crate::kv_policy::select_compressed_cache(
            &model,
            reader.clone(),
            prompt.len(),
            admit_any_transition(),
        );
        assert_eq!(refused, None);
        step(warm.as_mut(), &prompt, 0);
        step(warm.as_mut(), &[3], prompt.len());
        warm.reset().unwrap();
    }
    memory::clear_cache();
    let baseline = memory::get_active_memory() as u64;

    let (mut cache, refused) = crate::kv_policy::select_compressed_cache(
        &model,
        reader,
        prompt.len(),
        admit_any_transition(),
    );
    assert_eq!(refused, None);
    const SLACK: u64 = 256 * 1024;
    let check = |cache: &dyn KvCache, tokens: u64| {
        let dense = cache
            .compressed_dense_fallback()
            .expect("the compressed cache owns an explicit dense fallback");
        assert_eq!(
            dense.retained_snapshot().unwrap(),
            None,
            "the dense fallback holds K/V after an attention call"
        );
        assert!(dense.events().is_empty(), "{:?}", dense.events());
        let evidence = cache.packed_evidence().unwrap();
        assert!(!evidence.dense_active && evidence.fallback_reasons.is_empty());
        assert_eq!(evidence.full_cache_dequantizations, 0);
        let storage = cache.compressed_storage().unwrap().unwrap();
        assert_eq!(storage.tokens, tokens);
        // One layer's dense K of the same tokens: what the smallest surviving mirror adds.
        let one_dense_tensor = tokens * 4 * 128 * storage.element_bytes;
        assert!(
            one_dense_tensor > 4 * SLACK,
            "the fixture must expose a mirror"
        );
        let growth = (memory::get_active_memory() as u64).saturating_sub(baseline);
        assert!(
            growth <= storage.device_bytes() + SLACK,
            "{tokens} tokens: active memory grew {growth} B; the compressed cache retains \
                 {} B (a dense K or V of one layer is {one_dense_tensor} B)",
            storage.device_bytes()
        );
    };
    step(cache.as_mut(), &prompt, 0);
    check(cache.as_ref(), prompt.len() as u64);
    // Decode steps cross the next 32-token group boundary and fill its residual.
    for i in 0..40 {
        let offset = prompt.len() + i;
        step(cache.as_mut(), &[(i as i32) % 31 + 1], offset);
        check(cache.as_ref(), offset as u64 + 1);
    }
    let evidence = cache.packed_evidence().unwrap();
    assert!(evidence.accepted_direct_calls >= 2 * 40, "{evidence:?}");
    cache.reset().unwrap();
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::no_dense_kv_mirror_survives_a_compressed_attention_call");
}

/// sc-20680: the K8V8 page pool and dense block pool one synthetic model's paged requests
/// draw from.
fn paged_pools(
    model: &CausalLm,
    page_tokens: usize,
) -> (
    std::rc::Rc<std::cell::RefCell<crate::primitives::BlockPool>>,
    std::rc::Rc<std::cell::RefCell<crate::primitives::PackedPagePool>>,
) {
    let cfg = model.config();
    (
        crate::primitives::BlockPool::new(16),
        crate::primitives::PackedPagePool::new(
            cfg.num_layers,
            cfg.num_kv_heads as usize,
            cfg.head_dim as usize,
            page_tokens,
            crate::primitives::PackedCodeBits::Eight,
        )
        .unwrap(),
    )
}

/// A paged request whose prompt and final context the Qwen3 row admits (the synthetic decode
/// itself is short).
fn paged_request<'a>(
    policy: core_llm::KvCompressionPolicy,
    dense_pool: &'a std::rc::Rc<std::cell::RefCell<crate::primitives::BlockPool>>,
    packed_pool: &'a std::rc::Rc<std::cell::RefCell<crate::primitives::PackedPagePool>>,
    reader: Option<&'a crate::primitives::CompiledKernelHandle>,
) -> crate::primitives::PagedCacheRequest<'a> {
    crate::primitives::PagedCacheRequest {
        policy,
        family: Some(core_llm::KvModelFamily::Qwen3),
        prompt_tokens: 20_000,
        max_new_tokens: 1_024,
        batch: 1,
        dense_pool,
        packed_pool,
        reader,
    }
}

fn largest_relative_error(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> f32 {
    let range = reference
        .iter()
        .flatten()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    assert!(range > 0.0);
    reference
        .iter()
        .zip(candidate)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (a - b).abs() / range)
        .fold(0.0f32, f32::max)
}

/// sc-20680 AC1/AC2 on a synthetic GQA model: with the qualified opt-in, the paged selection
/// runs K8V8 pages read by the fused paged reader; its teacher-forced decode logits equal the
/// contiguous compressed cache's exactly (same codes, same kernels, read through the page
/// table instead of a contiguous array) and stay within 8-bit rounding of the dense paged
/// cache's. Every decode attention call is fused; no call gathers.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn paged_compressed_kv_logits_match_contiguous_compressed_and_dense_paged() {
    let model = tiny_causal_model(4, 2, 64);
    let reader =
        crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight).unwrap();
    let prompt = (0..150).map(|i| (i * 7) % 31 + 1).collect::<Vec<i32>>();
    let stream = (0..60).map(|i| (i * 5) % 31 + 1).collect::<Vec<i32>>();
    let (dense_pool, packed_pool) = paged_pools(&model, 64);
    let mut selection = model.select_paged_cache(paged_request(
        core_llm::KvCompressionPolicy::Qualified,
        &dense_pool,
        &packed_pool,
        Some(&reader),
    ));
    assert!(selection.is_compressed());
    let paged = forced_logits(&model, selection.cache_mut(), &prompt, &stream);
    let report = selection.report().unwrap();
    assert!(report.ran_compressed(), "{report:?}");
    assert_eq!(
        report.counters.fused_attention_calls,
        2 * stream.len() as u64
    );
    assert_eq!(report.counters.dense_gather_fallbacks, 0);
    assert!(report.counters.compressed_cache_bytes > 0);
    assert_eq!(
        packed_pool.borrow().live_pages(),
        ((prompt.len() + stream.len()) / 32 * 32).div_ceil(64)
    );

    let (mut contiguous, refused) = crate::kv_policy::select_compressed_cache(
        &model,
        reader,
        prompt.len(),
        admit_any_transition(),
    );
    assert_eq!(refused, None);
    let contiguous = forced_logits(&model, contiguous.as_mut(), &prompt, &stream);
    assert_eq!(paged, contiguous, "paged and contiguous compressed logits");

    let mut dense = model.new_paged_cache(16);
    let dense = forced_logits(&model, &mut dense, &prompt, &stream);
    let error = largest_relative_error(&dense, &paged);
    assert!(error < 0.02, "paged K8V8 vs dense paged: {error}");
    drop(selection);
    assert_eq!(packed_pool.borrow().live_pages(), 0);
    crate::model_inference_device::record_model_inference_device();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::paged_compressed_kv_logits_match_contiguous_compressed_and_dense_paged");
}

/// sc-20680 AC3 and the policy's refusals: with the policy off the selection is the existing
/// dense paged cache, bit-identical to `new_paged_cache`, and every refused request runs it
/// with the reason the report names.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn paged_selection_is_the_dense_paged_cache_when_off_or_refused() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    let model = tiny_causal_model(4, 2, 64);
    let reader =
        crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight).unwrap();
    let prompt = (0..70).map(|i| (i * 3) % 31 + 1).collect::<Vec<i32>>();
    let stream = (0..20).map(|i| (i * 11) % 31 + 1).collect::<Vec<i32>>();
    let (dense_pool, packed_pool) = paged_pools(&model, 64);
    let mut off = model.select_paged_cache(paged_request(
        Policy::Off,
        &dense_pool,
        &packed_pool,
        Some(&reader),
    ));
    assert!(!off.is_compressed());
    assert_eq!(off.report().unwrap().fallback, Some(Reason::PolicyDisabled));
    assert!(off
        .cache_mut()
        .as_any_mut()
        .downcast_mut::<crate::primitives::PagedKvCache>()
        .is_some());
    let off_logits = forced_logits(&model, off.cache_mut(), &prompt, &stream);
    let mut established = model.new_paged_cache(16);
    assert_eq!(
        off_logits,
        forced_logits(&model, &mut established, &prompt, &stream),
        "feature off is the established dense paged path"
    );
    assert_eq!(packed_pool.borrow().live_pages(), 0, "no page was touched");

    let reason = |request: crate::primitives::PagedCacheRequest<'_>| {
        let selection = model.select_paged_cache(request);
        assert!(!selection.is_compressed());
        selection.report().unwrap().fallback
    };
    let qualified = || paged_request(Policy::Qualified, &dense_pool, &packed_pool, Some(&reader));
    assert_eq!(
        reason(crate::primitives::PagedCacheRequest {
            prompt_tokens: 100,
            ..qualified()
        }),
        Some(Reason::BelowMinimumContext)
    );
    assert_eq!(
        reason(crate::primitives::PagedCacheRequest {
            max_new_tokens: 40_000,
            ..qualified()
        }),
        Some(Reason::AboveQualifiedContext),
        "the final context bounds the qualified maximum"
    );
    assert_eq!(
        reason(crate::primitives::PagedCacheRequest {
            family: None,
            ..qualified()
        }),
        Some(Reason::UnqualifiedModel)
    );
    assert_eq!(
        reason(crate::primitives::PagedCacheRequest {
            batch: 2,
            ..qualified()
        }),
        Some(Reason::BatchedDecode),
        "a sequence decoding beside another is refused as the static plan refuses a batch"
    );
    assert_eq!(
        reason(crate::primitives::PagedCacheRequest {
            reader: None,
            ..qualified()
        }),
        Some(Reason::ReaderUnavailable)
    );
    let other_geometry = crate::primitives::PackedPagePool::new(
        2,
        1,
        64,
        64,
        crate::primitives::PackedCodeBits::Eight,
    )
    .unwrap();
    assert_eq!(
        reason(crate::primitives::PagedCacheRequest {
            packed_pool: &other_geometry,
            ..qualified()
        }),
        Some(Reason::UnsupportedGeometry)
    );
    // A head dimension the fused reader does not implement is refused before the pool.
    assert_eq!(
        tiny_causal_model(2, 1, 32)
            .select_paged_cache(qualified())
            .report()
            .unwrap()
            .fallback,
        Some(Reason::UnsupportedGeometry)
    );
    crate::model_inference_device::record_model_inference_device();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::paged_selection_is_the_dense_paged_cache_when_off_or_refused");
}

/// sc-20680 AC2 page reuse, fragmentation and cancellation at the model boundary: requests
/// decoding interleaved on one page pool, one cancelled mid-stream (its cache dropped) and its
/// pages recycled by a later request, each produce exactly the logits they produce alone on a
/// fresh pool; the pool's live pages return to zero.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn paged_compressed_requests_share_a_fragmented_pool_and_cancellation_frees_pages() {
    let model = tiny_causal_model(4, 2, 64);
    let reader =
        crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight).unwrap();
    let prompts =
        [40, 75, 120].map(|len| (0..len).map(|i| (i * 7 + len) % 31 + 1).collect::<Vec<_>>());
    let streams = [0, 1, 2].map(|r| (0..50).map(|i| (i * 5 + r) % 31 + 1).collect::<Vec<_>>());
    let alone = |r: usize, tokens: usize| {
        let (dense_pool, packed_pool) = paged_pools(&model, 32);
        let mut selection = model.select_paged_cache(paged_request(
            core_llm::KvCompressionPolicy::Qualified,
            &dense_pool,
            &packed_pool,
            Some(&reader),
        ));
        forced_logits(
            &model,
            selection.cache_mut(),
            &prompts[r],
            &streams[r][..tokens],
        )
    };
    let (dense_pool, shared) = paged_pools(&model, 32);
    let select = || {
        model.select_paged_cache(paged_request(
            core_llm::KvCompressionPolicy::Qualified,
            &dense_pool,
            &shared,
            Some(&reader),
        ))
    };
    let host = |logits: Array| {
        let logits = logits.as_dtype(Dtype::Float32).unwrap();
        logits.eval().unwrap();
        logits.as_slice::<f32>().to_vec()
    };
    let mut a = select();
    let mut b = select();
    let mut rows = [Vec::new(), Vec::new(), Vec::new()];
    for (r, selection) in [(0, &mut a), (1, &mut b)] {
        rows[r].push(host(
            model
                .step(&input_ids(&prompts[r]), selection.cache_mut(), 0)
                .unwrap(),
        ));
    }
    // Interleaved decode; A is cancelled (dropped) after 30 tokens, C then takes its pages.
    #[allow(clippy::needless_range_loop)] // `i` indexes both requests' streams
    for i in 0..50 {
        for (r, selection) in [(0, &mut a), (1, &mut b)] {
            if r == 0 && i >= 30 {
                continue;
            }
            let offset = (prompts[r].len() + i) as i32;
            rows[r].push(host(
                model
                    .step(&input_ids(&[streams[r][i]]), selection.cache_mut(), offset)
                    .unwrap(),
            ));
        }
        if i == 30 {
            let before = shared.borrow().live_pages();
            let a_pages = a
                .cache_mut()
                .as_any_mut()
                .downcast_mut::<crate::primitives::PagedPackedKvCache>()
                .unwrap()
                .page_ids()
                .len();
            drop(std::mem::replace(&mut a, select()));
            assert_eq!(shared.borrow().live_pages(), before - a_pages);
        }
    }
    assert_eq!(rows[0], alone(0, 30), "A before its cancellation");
    assert_eq!(rows[1], alone(1, 50), "B interleaved with A and C");
    let mut c = a;
    rows[2].push(host(
        model
            .step(&input_ids(&prompts[2]), c.cache_mut(), 0)
            .unwrap(),
    ));
    for (i, &token) in streams[2].iter().enumerate() {
        let offset = (prompts[2].len() + i) as i32;
        rows[2].push(host(
            model
                .step(&input_ids(&[token]), c.cache_mut(), offset)
                .unwrap(),
        ));
    }
    assert_eq!(rows[2], alone(2, 50), "C on A's recycled pages");
    let pages = c
        .cache_mut()
        .as_any_mut()
        .downcast_mut::<crate::primitives::PagedPackedKvCache>()
        .unwrap()
        .page_ids()
        .to_vec();
    assert!(
        pages.windows(2).any(|w| w[1] != w[0] + 1),
        "C's pages are fragmented: {pages:?}"
    );
    drop(b);
    drop(c);
    assert_eq!(shared.borrow().live_pages(), 0);
    crate::model_inference_device::record_model_inference_device();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::paged_compressed_requests_share_a_fragmented_pool_and_cancellation_frees_pages");
}

/// sc-20680 AC1: a fused paged attention call neither builds nor keeps a dense K/V copy.
/// Retained: MLX's active memory grows by no more than the page pool's measured arrays plus
/// the sequence's page table and residuals. Transient: each decode step's MLX peak above its
/// own starting point stays within the pool growth of that step plus the sequence's table and
/// residuals and a slack far below one layer's dense K (a gather built and dropped inside the
/// call shows up in the peak even though nothing survives it).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn no_dense_kv_copy_survives_a_paged_compressed_attention_call() {
    use mlx_rs::memory;
    let model = tiny_causal_model(4, 4, 128);
    let reader =
        crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight).unwrap();
    let prompt = (0..1536).map(|i| i % 31 + 1).collect::<Vec<i32>>();
    let step = |cache: &mut dyn KvCache, ids: &[i32], offset: usize| {
        let logits = model.step(&input_ids(ids), cache, offset as i32).unwrap();
        logits.eval().unwrap();
    };
    {
        let (dense_pool, packed_pool) = paged_pools(&model, 64);
        let mut warm = model.select_paged_cache(paged_request(
            core_llm::KvCompressionPolicy::Qualified,
            &dense_pool,
            &packed_pool,
            Some(&reader),
        ));
        step(warm.cache_mut(), &prompt, 0);
        step(warm.cache_mut(), &[3], prompt.len());
    }
    memory::clear_cache();
    let baseline = memory::get_active_memory() as u64;
    let (dense_pool, packed_pool) = paged_pools(&model, 64);
    let mut selection = model.select_paged_cache(paged_request(
        core_llm::KvCompressionPolicy::Qualified,
        &dense_pool,
        &packed_pool,
        Some(&reader),
    ));
    const SLACK: u64 = 256 * 1024;
    let check = |selection: &crate::primitives::PagedCacheSelection, tokens: u64| {
        let cache = selection.cache();
        let evidence = cache.packed_evidence().unwrap();
        assert!(evidence.dense_gather_calls.is_empty(), "{evidence:?}");
        let storage = cache.compressed_storage().unwrap().unwrap();
        assert_eq!(storage.tokens, tokens);
        let pool = packed_pool.borrow().storage();
        let (_, page_metadata) = packed_pool.borrow().page_bytes();
        let per_sequence = storage.device_metadata_bytes - pool.live_pages * page_metadata;
        let one_dense_tensor = tokens * 4 * 128 * storage.element_bytes;
        assert!(
            one_dense_tensor > 4 * SLACK,
            "the fixture must expose a copy"
        );
        let growth = (memory::get_active_memory() as u64).saturating_sub(baseline);
        assert!(
            growth <= pool.code_bytes + pool.metadata_bytes + per_sequence + SLACK,
            "{tokens} tokens: active memory grew {growth} B; the page pool holds {} B and \
                 the sequence's table and residuals {per_sequence} B (one dense K or V of one \
                 layer is {one_dense_tensor} B)",
            pool.code_bytes + pool.metadata_bytes
        );
    };
    step(selection.cache_mut(), &prompt, 0);
    check(&selection, prompt.len() as u64);
    let pool_bytes = || {
        let storage = packed_pool.borrow().storage();
        storage.code_bytes + storage.metadata_bytes
    };
    let mut worst_transient = 0u64;
    for i in 0..40 {
        let offset = prompt.len() + i;
        let pool_before = pool_bytes();
        memory::reset_peak_memory();
        let start = memory::get_active_memory() as u64;
        step(selection.cache_mut(), &[(i as i32) % 31 + 1], offset);
        let transient = (memory::get_peak_memory() as u64).saturating_sub(start);
        check(&selection, offset as u64 + 1);
        let storage = selection.cache().compressed_storage().unwrap().unwrap();
        let (_, page_metadata) = packed_pool.borrow().page_bytes();
        let per_sequence = storage.device_metadata_bytes
            - packed_pool.borrow().live_pages() as u64 * page_metadata;
        // One layer's dense bf16 K of the history, the smallest gather a step could build.
        let one_dense_tensor = (offset as u64 + 1) * 4 * 128 * 2;
        assert!(one_dense_tensor > 4 * SLACK);
        // A step that grows the pool copies it: the old arrays stay alive until the grown
        // ones replace them, so that step's transient is the grown pool (sc-20681's bounded
        // growth copies more often than doubling did). Any other step may add only its
        // table, residuals and slack.
        let pool_after = pool_bytes();
        let growth_copy = if pool_after > pool_before {
            2 * pool_after - pool_before
        } else {
            0
        };
        let allowance = growth_copy + per_sequence + SLACK;
        assert!(
            transient <= allowance,
            "decode step {i}: MLX peak rose {transient} B above the step's start; allowed \
                 {allowance} B (one dense K of one layer is {one_dense_tensor} B)"
        );
        worst_transient = worst_transient.max(transient);
    }
    eprintln!("worst per-step decode transient: {worst_transient} B");
    assert_eq!(
        selection.report().unwrap().counters.fused_attention_calls,
        2 * 40
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED provider::tests::no_dense_kv_copy_survives_a_paged_compressed_attention_call");
}

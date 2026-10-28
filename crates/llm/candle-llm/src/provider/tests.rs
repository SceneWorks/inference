//! `provider.rs`'s unit tests, moved out of the production file as pure code motion (sc-27409).

use super::{
    bonsai_sampling_defaults, can_load, can_load_vision, cuda_usable_memory_bytes, emit_content,
    ensure_supported_device, eos_token_ids, expand_vision_placeholders, host_load_budget,
    is_frozen_qwen38_config, load_memory_requirements, merged_frame_timestamps,
    prompt_opens_thinking, qwen35_dense_prefix, substitute_vision_placeholders,
    validate_context_window, video_placeholder_text, HostRule, JsonMask, LoadWorkingSet,
    EAGER_ATTN_QUERY_CHUNK_SIZE,
};
use super::{Decode as _, Decoder};
use crate::decode::DecodePath;
use crate::models::CausalLm;
use candle_core::Tensor;
use core_llm::ProposerKind;
use std::collections::HashMap;

#[test]
fn actual_cpu_provider_exposes_its_owned_release_fence() {
    let (cfg, weights) = tiny_llama_parts();
    let vocab = cfg["vocab_size"].as_u64().unwrap() as usize;
    let tokenizer = core_llm::Tokenizer::from_json(&synthetic_tokenizer_json(vocab)).unwrap();
    let provider =
        super::LlamaProvider::from_parts(tiny_causal_from(cfg, weights), tokenizer, vec![]);
    let engine: &dyn core_llm::TextLlm = &provider;
    engine.synchronize_before_release().unwrap();
}

#[test]
fn partial_load_failure_fences_after_drop_and_preserves_primary_error() {
    use std::{cell::RefCell, rc::Rc};
    struct Partial(Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Partial {
        fn drop(&mut self) {
            self.0.borrow_mut().push("partial-drop");
        }
    }
    for cancelled in [false, true] {
        let events = Rc::new(RefCell::new(Vec::new()));
        let result: core_llm::Result<()> = {
            let trace = events.clone();
            let _fence = super::LoadFailureFence::new(move || {
                trace.borrow_mut().push("owned-fence");
                Err(core_llm::Error::Unsupported(
                    "injected cleanup failure".into(),
                ))
            });
            let _partial = Partial(events.clone());
            if cancelled {
                Err(core_llm::Error::Canceled)
            } else {
                Err(core_llm::Error::Load("primary load error".into()))
            }
        };
        if cancelled {
            assert!(matches!(result, Err(core_llm::Error::Canceled)));
        } else {
            assert!(
                matches!(result,Err(core_llm::Error::Load(message)) if message=="primary load error")
            );
        }
        assert_eq!(*events.borrow(), vec!["partial-drop", "owned-fence"]);
    }
    let mut fence = super::LoadFailureFence::new(|| -> core_llm::Result<()> {
        panic!("successful load must not add a synchronization")
    });
    fence.disarm();
    drop(fence);
}

/// A device's row of the decode-defaults table, read directly (not through the provider's own
/// reader) so a provider that stops following the table is caught.
fn table_depths(device: &candle_core::Device) -> core_llm::RecommendedDepths {
    crate::device::decode_defaults(device).recommended_depths
}

#[test]
fn accelerator_only_selector_is_exact_to_qwen38_and_bonsai() {
    let frozen: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/reference/qwen38/config.json"
    )))
    .unwrap();
    assert!(is_frozen_qwen38_config(&frozen));

    let mut flat = frozen.clone();
    flat["model_type"] = serde_json::json!("qwen3_5_text");
    flat.as_object_mut().unwrap().remove("text_config");
    assert!(
        !is_frozen_qwen38_config(&flat),
        "generic flat qwen3_5_text checkpoints retain CPU support"
    );

    let mut other_geometry = frozen;
    other_geometry["text_config"]["vocab_size"] = serde_json::json!(32_000);
    assert!(!is_frozen_qwen38_config(&other_geometry));
}

#[test]
fn accelerator_only_snapshots_reject_cpu_before_weight_inventory() {
    let qwen = tempfile::tempdir().unwrap();
    std::fs::write(
        qwen.path().join("config.json"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/config.json"
        )),
    )
    .unwrap();
    let error = ensure_supported_device(qwen.path(), &candle_core::Device::Cpu)
        .expect_err("Qwen3.8 CPU load must be rejected before missing weights are inspected");
    assert!(error
        .to_string()
        .contains("Candle CPU inference is unsupported"));

    let bonsai = tempfile::tempdir().unwrap();
    std::fs::write(
        bonsai.path().join("config.json"),
        br#"{"model_type":"prism_hadamard_qwen35"}"#,
    )
    .unwrap();
    let error = ensure_supported_device(bonsai.path(), &candle_core::Device::Cpu)
        .expect_err("Bonsai CPU load must be rejected before missing weights are inspected");
    assert!(error
        .to_string()
        .contains("Candle CPU inference is unsupported"));
}
use crate::decode::{ConstraintMask, RewindableConstraintMask};
use core_llm::{
    ConstraintDecodeTable, Content, ImageRef, LlmMemoryGeometry, LoadSpec, Message, Role, VideoRef,
};

#[test]
fn qwen35_dense_checkpoint_selects_one_decoder_root() {
    let select = |keys: &[&str]| qwen35_dense_prefix(|key| keys.contains(&key));
    assert_eq!(
        select(&["model.language_model.embed_tokens.weight"]).unwrap(),
        "model.language_model"
    );
    // The pinned Altworld/Hemmingway-1 index uses this flat text-only layout; MTP remains
    // rooted at `mtp.*` and is loaded independently of the decoder prefix.
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

#[test]
fn request_estimate_uses_the_eager_attention_runtime_tile() {
    let geometry = LlmMemoryGeometry {
        query_heads: 40,
        kv_heads: 4,
        head_dim: 128,
        layers: 64,
        element_bytes: 4,
        score_element_bytes: 4,
        hidden_size: 5120,
        intermediate_size: 17_408,
        vocab_size: 248_320,
        recurrent_bytes: 0,
    };
    let chunked = core_llm::estimate_chunked_request_bytes(
        29_600,
        128,
        geometry,
        0,
        0,
        EAGER_ATTN_QUERY_CHUNK_SIZE,
    )
    .unwrap();
    let eager = core_llm::estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap();
    assert!(chunked < eager);
    assert!(chunked < 36_000_000_000, "bounded Candle peak: {chunked}");
}

#[test]
fn dense_cuda_load_admission_separates_host_staging_from_current_vram() {
    // Frozen Qwen3.8 parent inventory from release/qwen38-bonsai-artifacts.json. Candle reads
    // the 18 shards serially; this is the total payload and the largest individual shard.
    let payload = 55_563_006_776;
    let largest_shard = 3_988_973_152;
    let dense = LoadWorkingSet::resident_payload(payload, 0, 0, false);
    let (host_required, device_required) =
        load_memory_requirements(payload, largest_shard, 0, true, dense).unwrap();

    assert_eq!(host_required, largest_shard);
    assert_eq!(device_required, Some(69_453_758_470));
    assert_eq!(host_load_budget(true, Some(102_171_148_288)), None);

    // A launch-time snapshot can cap but never inflate the current post-load CUDA capacity.
    let current_free = 60_000_000_000;
    let available =
        core_llm::effective_memory_budget(Some(current_free), Some(102_171_148_288)).unwrap();
    let error = core_llm::admit_load_memory(device_required.unwrap(), available)
        .expect_err("current CUDA shortfall must fail closed");
    assert!(error
        .to_string()
        .contains("only 60000000000 bytes are available"));
}

#[test]
fn cuda_request_admission_counts_idle_async_pool_memory() {
    // Reproduce the observed zero-driver-free shape with a synthetic pool snapshot. The pool
    // counters were not captured by RC3, so this test deliberately makes no claim about their
    // exact campaign values.
    let total = 100;
    let reserved = total;
    let used = 60;
    let available = cuda_usable_memory_bytes(0, total, Some((reserved, used))).unwrap();
    assert_eq!(available, 40);
    core_llm::admit_request_memory(36, available)
        .expect("reusable pool memory must remain available to the owning allocator");
}

#[test]
fn cuda_request_admission_rejects_real_pool_shortfall() {
    let gib = 1_u64 << 30;
    let available =
        cuda_usable_memory_bytes(gib / 2, 8 * gib, Some((4 * gib, 7 * gib / 2))).unwrap();
    assert_eq!(available, gib);
    let error = core_llm::admit_request_memory(gib + 1, available)
        .expect_err("live allocations and external use must remain unavailable");
    assert!(error
        .to_string()
        .contains("only 1073741824 bytes are available"));
}

#[test]
fn cuda_request_admission_fails_closed_on_invalid_pool_snapshot() {
    assert_eq!(cuda_usable_memory_bytes(9, 8, None), None);
    assert_eq!(cuda_usable_memory_bytes(0, 8, Some((7, 8))), None);
    assert_eq!(cuda_usable_memory_bytes(0, 8, Some((9, 0))), None);
    assert_eq!(cuda_usable_memory_bytes(1, 8, Some((8, 0))), None);
}

#[test]
fn load_memory_requirements_remain_checked() {
    let dense = |payload, copy| LoadWorkingSet::resident_payload(payload, 0, copy, false);
    assert!(load_memory_requirements(u64::MAX, 1, 0, true, dense(u64::MAX, 0)).is_none());
    assert!(load_memory_requirements(1, 1, u64::MAX, true, dense(1, 0)).is_none());
    assert!(
        load_memory_requirements(1, 1, 0, true, dense(u64::MAX / 4 * 3, u64::MAX / 4)).is_none()
    );
    let gguf = LoadWorkingSet {
        source: u64::MAX / 2,
        cast: u64::MAX / 2,
        copy: 0,
        held: 0,
        host: HostRule::Gguf,
    };
    for cuda in [true, false] {
        assert!(
            load_memory_requirements(1, 1, 0, cuda, gguf).is_none(),
            "{cuda}"
        );
    }
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
fn stop_matches_utf8_after_incremental_detokenization_and_never_leaks_tail() {
    let mut matcher = core_llm::StopMatcher::new(["βγ".to_string(), "βγmore".to_string()]);
    let mut detok = core_llm::IncrementalDetok::default();
    let halt = std::cell::Cell::new(false);
    let (mut text, mut index, mut last) = (String::new(), 0, 0);
    for decoded in ["α", "α�", "αβ", "αβ�", "αβγleak"] {
        if let Some(delta) = detok.push(decoded) {
            emit_content(
                delta,
                1,
                (&mut matcher, &halt),
                &mut text,
                &mut index,
                &mut last,
                &mut |_| {},
            );
        }
    }
    emit_content(
        "more leak",
        2,
        (&mut matcher, &halt),
        &mut text,
        &mut index,
        &mut last,
        &mut |_| {},
    );
    assert_eq!(text, "α");
    assert!(halt.get());
}

#[test]
fn content_stop_is_trimmed_across_token_boundaries() {
    let mut matcher = core_llm::StopMatcher::new(["<STOP>".to_string()]);
    let halt = std::cell::Cell::new(false);
    let mut streamed = String::new();
    let mut index = 0;
    let mut last_id = 0;
    let mut events = Vec::new();
    emit_content(
        "answer<ST",
        1,
        (&mut matcher, &halt),
        &mut streamed,
        &mut index,
        &mut last_id,
        &mut |event| events.push(event),
    );
    emit_content(
        "OP>leak",
        2,
        (&mut matcher, &halt),
        &mut streamed,
        &mut index,
        &mut last_id,
        &mut |event| events.push(event),
    );
    assert_eq!(streamed, "answer");
    assert!(halt.get());
    assert_eq!(events.len(), 1);
}

#[test]
fn bonsai_sampling_defaults_match_official_thinking_modes() {
    let defaults = bonsai_sampling_defaults();
    assert_eq!(defaults.thinking.temperature, 1.0);
    assert_eq!(defaults.thinking.top_p, 0.95);
    assert_eq!(defaults.thinking.top_k, 20);
    assert_eq!(defaults.thinking.presence_penalty, 0.0);
    assert_eq!(defaults.thinking.repetition_penalty, 1.0);
    assert_eq!(defaults.thinking.repetition_context, 0);

    assert_eq!(defaults.non_thinking.temperature, 0.7);
    assert_eq!(defaults.non_thinking.top_p, 0.8);
    assert_eq!(defaults.non_thinking.top_k, 20);
    assert_eq!(defaults.non_thinking.presence_penalty, 1.5);
    assert_eq!(defaults.non_thinking.repetition_penalty, 1.0);
    assert_eq!(defaults.non_thinking.repetition_context, 0);
}

#[test]
fn prism_mlx_weightless_probe_requires_declared_embedded_vision() {
    let dir = tempfile::tempdir().unwrap();
    let config = |vision: bool| {
        serde_json::json!({
            "model_type": "prism_hadamard_qwen35",
            "components": {"text": true, "vision": vision, "mtp": false},
            "vision_config": {"depth": 1}
        })
    };
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config(true)).unwrap(),
    )
    .unwrap();
    let spec = LoadSpec::dense(dir.path().to_string_lossy());
    assert!(can_load(&spec));
    assert!(can_load_vision(&spec));

    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config(false)).unwrap(),
    )
    .unwrap();
    assert!(can_load(&spec), "text Bonsai remains loadable");
    assert!(!can_load_vision(&spec));

    let associated = spec.clone().with_projector("external.gguf");
    assert!(!can_load(&associated));
    assert!(!can_load_vision(&associated));
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

#[test]
fn prompt_opens_thinking_matches_template_modes() {
    // A Qwen3-style thinking/auto generation prompt opens the block and leaves it unclosed.
    assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
    // A no-think (Disabled) prompt renders a closed `<think></think>`.
    assert!(!prompt_opens_thinking(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    // A prior closed reasoning turn followed by a fresh open block still opens.
    assert!(prompt_opens_thinking(
        "<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"
    ));
    // No reasoning markers at all (a non-thinking template).
    assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
}

fn qwen3vl_video_oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("../models/testdata/qwen3vl_video_oracle.json")).unwrap()
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

/// `merged_frame_timestamps` averages within each `temporal_patch_size` group (padding the last)
/// to one timestamp per emitted vision frame — matching `Qwen3VLProcessor._calculate_timestamps`.
#[test]
fn merged_timestamps_match_hf_reference() {
    let j = qwen3vl_video_oracle();
    let fps = j["fps"].as_f64().unwrap() as f32;
    let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
    let indices: Vec<f32> = j["frames_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
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
fn invalid_video_timestamps_fail_before_prompt_rendering() {
    let frame = || ImageRef::new(1, 1, vec![0, 0, 0]).unwrap();
    for timestamps in [vec![0.0, f32::NAN], vec![0.5, 0.25], vec![-0.1, 0.0]] {
        let video = VideoRef {
            frames: vec![frame(), frame()],
            timestamps,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![Content::Video(video)],
            thinking: None,
            tool_calls: Vec::new(),
        }];
        let error = substitute_vision_placeholders(&messages, 2)
            .expect_err("invalid timestamp sequence accepted");
        assert!(error.to_string().contains("timestamp"));
    }
}

#[test]
fn expanded_media_tokens_count_against_context_window() {
    validate_context_window(64, 48, 16).unwrap();
    let error =
        validate_context_window(64, 49, 16).expect_err("expanded prompt over context was accepted");
    assert!(error.to_string().contains("expanded prompt (49 tokens)"));
    assert!(error.to_string().contains("context window 64"));
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
    let expanded = expand_vision_placeholders(&raw, vid, &counts).unwrap();
    assert_eq!(expanded, expanded_hf, "expanded video ids vs HF processor");
}

/// **The video M-RoPE positions over the oracle grid are well-formed and per-frame-reset.** Feed
/// the expanded video id stream + the `video_grid_thw` through `mrope_positions_mm`: the temporal
/// row must reset to the frame's cursor at each frame (Qwen3-VL's synthetic time axis splits each
/// `[t,h,w]` into `t` per-frame `[1,h,w]` blocks). The HF-pinned exact-row check lives in
/// `deepstack`'s mrope oracle test; here we confirm the provider's video grid drives the same path
/// consistently.
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
    let img = vid - 1; // a distinct unused image id
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
/// E6 (sc-24132): the static KV cache preallocates K/V for the whole request bound at its
/// first step, and the geometry-based admission estimate already covers that preallocation —
/// `estimate_request_bytes` charges KV for `prompt + max_new_tokens` positions over every layer
/// at `element_bytes` (4), while the static cache holds only the full-attention layers in the
/// compute dtype — so a request admitted under the estimate cannot fail its preallocation
/// (and a request past the model bound fails closed with the typed error before allocating).
#[test]
fn qwen35_admission_covers_the_static_kv_preallocation() {
    use crate::decode::StepModel;

    let (_cfg, model) = crate::models::qwen35::tests::text_model();
    let (prompt_tokens, max_new_tokens) = (11usize, 21u32);
    let capacity = prompt_tokens + max_new_tokens as usize;
    let preallocation = model.static_kv_bytes(capacity) as u64;
    assert!(preallocation > 0);
    let decoder = Decoder::Qwen35(model);
    let geometry = decoder.memory_geometry();
    let kv_term = (capacity as u64)
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * geometry.element_bytes
        * 2;
    assert!(
        preallocation <= kv_term,
        "static preallocation {preallocation} exceeds the KV term {kv_term} admission charges"
    );
    let estimate =
        core_llm::estimate_request_bytes(prompt_tokens, max_new_tokens, geometry, 0, 0).unwrap();
    assert!(estimate >= preallocation + geometry.recurrent_bytes);

    // What the step seam really allocates for that request is exactly the priced number.
    let Decoder::Qwen35(model) = &decoder else {
        unreachable!()
    };
    // (The live bytes also carry one ring slot per linear layer — the recurrent term, priced
    // separately; sc-24131.)
    let live_recurrent = model.recurrent_state_bytes(1) as u64;
    let cache = model.new_cache_for(capacity, 0).unwrap();
    assert_eq!(
        cache.memory().live_bytes as u64,
        preallocation + live_recurrent
    );
    assert_eq!(cache.kv_kind(), crate::primitives::KvCacheKind::Static);
    // A declared overshoot is part of the bound (and of the priced preallocation).
    let cache = model.new_cache_for(capacity, 3).unwrap();
    assert_eq!(cache.kv_capacity(), Some(capacity + 3));
    assert_eq!(
        cache.memory().live_bytes as u64,
        model.static_kv_bytes(capacity + 3) as u64 + live_recurrent
    );
}

/// A tiny dense causal decoder from a JSON config and seeded weights (llama or Gemma 4 keys).
fn tiny_causal_from(cfg: serde_json::Value, weights: HashMap<String, Tensor>) -> CausalLm {
    let cfg = crate::config::ModelConfig::from_json(&cfg).unwrap();
    CausalLm::from_weights(
        &crate::primitives::Weights::from_map(weights, candle_core::Device::Cpu),
        "",
        cfg,
    )
    .unwrap()
}

fn tiny_llama_for_admission() -> CausalLm {
    let (cfg, w) = tiny_llama_parts();
    tiny_causal_from(cfg, w)
}

/// The tiny llama decoder's config and seeded weights (3 layers, vocab 40).
fn tiny_llama_parts() -> (serde_json::Value, HashMap<String, Tensor>) {
    llama_parts(40, 32, 64, 4, 2, 3)
}

/// A seeded llama decoder of the given geometry: its config and f32 weights.
fn llama_parts(
    vocab: usize,
    hidden: usize,
    inter: usize,
    heads: usize,
    kv_heads: usize,
    layers: usize,
) -> (serde_json::Value, HashMap<String, Tensor>) {
    use crate::primitives::{SplitMix64, TokenRng};
    let head_dim = hidden / heads;
    let mut rng = SplitMix64::new(0x000A_D417);
    let mut rand = |dims: &[usize]| {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
        Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap()
    };
    let mut w = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".to_string(),
        rand(&[vocab, hidden]),
    );
    w.insert("model.norm.weight".to_string(), rand(&[hidden]));
    w.insert("lm_head.weight".to_string(), rand(&[vocab, hidden]));
    for i in 0..layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        w.insert(p("input_layernorm.weight"), rand(&[hidden]));
        w.insert(p("post_attention_layernorm.weight"), rand(&[hidden]));
        w.insert(
            p("self_attn.q_proj.weight"),
            rand(&[heads * head_dim, hidden]),
        );
        w.insert(
            p("self_attn.k_proj.weight"),
            rand(&[kv_heads * head_dim, hidden]),
        );
        w.insert(
            p("self_attn.v_proj.weight"),
            rand(&[kv_heads * head_dim, hidden]),
        );
        w.insert(
            p("self_attn.o_proj.weight"),
            rand(&[hidden, heads * head_dim]),
        );
        w.insert(p("mlp.gate_proj.weight"), rand(&[inter, hidden]));
        w.insert(p("mlp.up_proj.weight"), rand(&[inter, hidden]));
        w.insert(p("mlp.down_proj.weight"), rand(&[hidden, inter]));
    }
    (
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv_heads,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 256, "tie_word_embeddings": false
        }),
        w,
    )
}

/// sc-24140: a llama-family load reports its weight census (the `LoadRecord` said `None` for
/// every non-qwen3_5 architecture), and the requested format reaches `CausalLm`: dense keeps
/// all 3 × 7 projections and the head dense; `Quantize::Q8` makes the layer projections GGML
/// and keeps the head dense.
#[test]
fn a_llama_family_load_reports_its_census_for_the_requested_format() {
    use core_llm::{LoadSpec, Quantize};
    let (cfg, weights) = tiny_llama_parts();
    let dir = tempfile::Builder::new()
        .prefix("candle-llama-census-")
        .tempdir()
        .unwrap();
    std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        synthetic_tokenizer_json(40),
    )
    .unwrap();
    candle_core::safetensors::save(&weights, dir.path().join("model.safetensors")).unwrap();
    let load = |quantize| {
        super::LlamaProvider::load(&LoadSpec {
            quantize,
            ..LoadSpec::dense(dir.path().display().to_string())
        })
        .expect("load the synthetic llama snapshot")
        .load_record()
    };
    let dense = load(None);
    let census = dense
        .census
        .expect("a llama-family load reports its census");
    assert_eq!(census.projections.dense.count, 3 * 7 + 1);
    assert_eq!(census.projections.ggml.count, 0);
    let q8 = load(Some(Quantize::Q8));
    assert_eq!(q8.requested, Some(Quantize::Q8));
    let census = q8.census.expect("census");
    assert_eq!(
        census.projections.ggml.count,
        3 * 7,
        "every layer projection is Q8"
    );
    assert_eq!(
        census.projections.dense.count, 1,
        "the head stays dense under GGML"
    );
    assert_eq!(census.projections.dense.params, 40 * 32);
}

/// The shared Gemma 4 decoder fixture: sliding layers 2 KV heads × 8, full layers
/// `attention_k_eq_v` with 1 global KV head × 16 — equal `kv_heads × head width` per layer.
///
/// With `k_eq_v` off (`k_eq_v = false`) the full layers keep the ordinary two KV heads at the
/// global width 16 (upstream gates `num_global_key_value_heads` on the flag) and get their own
/// `v_proj`: 2 × 16 per full layer against the sliding layers' 2 × 8, so the full layers are
/// the widest and the scalar `num_key_value_heads × head_dim` (2 × 8) under-prices them.
fn tiny_gemma4_for_admission_with(k_eq_v: bool) -> CausalLm {
    use crate::primitives::{SplitMix64, TokenRng};
    let g: serde_json::Value = serde_json::from_str(include_str!(
        "../../../testdata/gemma4/gemma4_decoder_goldens.json"
    ))
    .unwrap();
    let mut w = HashMap::new();
    for (key, spec) in g["weights"].as_object().unwrap() {
        let shape: Vec<usize> = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let data: Vec<f32> = spec["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        w.insert(
            key.clone(),
            Tensor::from_vec(data, shape, &candle_core::Device::Cpu).unwrap(),
        );
    }
    let mut cfg = g["config"].clone();
    if !k_eq_v {
        cfg["text_config"]["attention_k_eq_v"] = serde_json::json!(false);
        let text = &cfg["text_config"];
        let hidden = text["hidden_size"].as_u64().unwrap() as usize;
        let kv_heads = text["num_key_value_heads"].as_u64().unwrap() as usize;
        let global_head_dim = text["global_head_dim"].as_u64().unwrap() as usize;
        let mut rng = SplitMix64::new(0x6E4F_A11D);
        let mut rand = |dims: [usize; 2]| {
            let data: Vec<f32> = (0..dims[0] * dims[1])
                .map(|_| (rng.next_f32() - 0.5) * 0.2)
                .collect();
            Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap()
        };
        let layer_types = text["layer_types"].as_array().unwrap().clone();
        for (i, kind) in layer_types.iter().enumerate() {
            if kind.as_str() == Some("full_attention") {
                let rows = kv_heads * global_head_dim;
                w.insert(
                    format!("model.layers.{i}.self_attn.k_proj.weight"),
                    rand([rows, hidden]),
                );
                w.insert(
                    format!("model.layers.{i}.self_attn.v_proj.weight"),
                    rand([rows, hidden]),
                );
            }
        }
    }
    tiny_causal_from(cfg, w)
}

/// E6 (sc-24138): the provider decodes the llama family through the engine on the step seam's
/// static KV cache, which preallocates every caching layer's K/V for the request bound at
/// once, and admission prices it: the causal geometry carries the one widest layer's
/// `(kv_heads, head width)`, so `layers × kv_heads × head_dim × 2` covers every layer.
///
/// * llama (uniform) and Gemma 4 with `k_eq_v` (sliding 2 × 8, full 1 × 16 — equal widths):
///   the KV term is **exactly** the preallocation — no inflation (the old independent maxima,
///   2 heads × 16, charged twice what any layer holds);
/// * Gemma 4 without `k_eq_v` (full 2 × 16 wider than sliding 2 × 8): the full layers set the
///   geometry, and the scalar `num_key_value_heads × head_dim` would under-price the cache.
///
/// The request is long enough (9 + 200 positions) that the KV term dominates the estimate, so
/// `priced >= preallocation` fails if the KV geometry is lost. The provider's causal engine
/// path runs no proposer, so it declares no verify overshoot and its request is priced on the
/// `Off` plan for exactly `prompt + max_new_tokens` positions; the seam's own overshoot
/// contract (`static_kv_bytes(capacity + overshoot)` exactly) is checked here too, but no
/// provider path prices one for this family.
#[test]
fn causal_admission_covers_the_static_kv_preallocation() {
    use crate::decode::StepModel;
    use crate::primitives::DecodeCache;

    for (label, model, exact) in [
        ("llama", tiny_llama_for_admission(), true),
        ("gemma4 k_eq_v", tiny_gemma4_for_admission_with(true), true),
        (
            "gemma4 full wider",
            tiny_gemma4_for_admission_with(false),
            false,
        ),
    ] {
        let (prompt_tokens, max_new_tokens) = (9usize, 200u32);
        let capacity = prompt_tokens + max_new_tokens as usize;
        let preallocation = model.static_kv_bytes(capacity) as u64;
        assert!(preallocation > 0, "{label}");
        let layout = model.kv_layout();
        let widest = layout
            .layers
            .iter()
            .flatten()
            .map(|l| l.kv_heads * l.key_dim.max(l.value_dim))
            .max()
            .unwrap();
        let decoder = Decoder::Causal(model);
        let geometry = decoder.memory_geometry();
        assert_eq!(
            (geometry.kv_heads, geometry.head_dim),
            {
                let (h, d) = layout.widest_layer();
                (h as u64, d as u64)
            },
            "{label}: the geometry is the widest layer's"
        );
        assert_eq!(
            geometry.kv_heads * geometry.head_dim,
            widest as u64,
            "{label}: one layer's heads × width, never the maxima of two layer types"
        );
        let kv_term = (capacity as u64)
            * geometry.layers
            * geometry.kv_heads
            * geometry.head_dim
            * geometry.element_bytes
            * 2;
        if exact {
            // CPU f32 cache elements are the geometry's 4 bytes: nothing is over-charged.
            assert_eq!(
                kv_term, preallocation,
                "{label}: the KV term is exactly the preallocation (no inflation)"
            );
        } else {
            assert!(
                preallocation <= kv_term,
                "{label}: static preallocation {preallocation} exceeds the KV term {kv_term}"
            );
        }
        // What the provider charges this request (the causal engine path: no proposer).
        let engine = super::DecodeRoute::PLAIN_ENGINE;
        let priced =
            super::priced_request_bytes(&decoder, engine, prompt_tokens, max_new_tokens, 0, false)
                .unwrap();
        // With the CUDA-graph runner on, the runner wraps this engine too, so its graph
        // workspace is priced (sc-24140): one step token count (`K + 1 = 1`) over the
        // request's reach.
        let graph_term = crate::decode::graph_workspace_admission_bytes(
            &decoder.step_memory_geometry(0),
            capacity as u64,
            1,
        )
        .unwrap();
        assert!(graph_term > 0, "{label}");
        assert_eq!(
            super::priced_request_bytes(&decoder, engine, prompt_tokens, max_new_tokens, 0, true,)
                .unwrap()
                - priced,
            graph_term,
            "{label}"
        );
        let Decoder::Causal(model) = &decoder else {
            unreachable!()
        };
        // The cache the engine builds for that request (`new_cache_for(prompt + budget, 0)`)
        // is exactly the priced preallocation, and the estimate covers it.
        let cache = model.new_cache_for(capacity, 0).unwrap();
        assert_eq!(cache.kv_kind(), crate::primitives::KvCacheKind::Static);
        assert_eq!(cache.memory().live_bytes as u64, preallocation, "{label}");
        assert!(
            priced >= preallocation,
            "{label}: priced {priced} < preallocation {preallocation}"
        );
        // The seam's overshoot contract: a declared overshoot is part of the bound, and the
        // cache is exactly `static_kv_bytes(capacity + overshoot)` (the provider's causal path
        // declares none — see above).
        let cache = model.new_cache_for(capacity, 3).unwrap();
        assert_eq!(cache.kv_capacity(), Some(capacity + 3), "{label}");
        assert_eq!(
            cache.memory().live_bytes as u64,
            model.static_kv_bytes(capacity + 3) as u64,
            "{label}"
        );
    }

    // The non-`k_eq_v` fixture is where the full layers are widest: the scalar geometry the
    // family read before sc-24138 would under-price the static cache.
    let gemma4 = tiny_gemma4_for_admission_with(false);
    let cfg = gemma4.config();
    let scalar = (cfg.num_kv_heads * cfg.head_dim) as u64;
    let (h, d) = gemma4.kv_layout().widest_layer();
    assert_eq!((h, d), (2, 16), "the full layers (2 × 16) are the widest");
    let capacity = 26u64;
    let scalar_term = capacity * cfg.num_layers as u64 * scalar * 4 * 2;
    assert!(
        scalar_term < gemma4.static_kv_bytes(capacity as usize) as u64,
        "the scalar geometry under-prices the full layers"
    );
}

/// sc-24441 (E3): with the CUDA-graph switch on, a load the runner can never capture on
/// names why in the load's fallbacks — the load device's (or the build's) refusal first, in
/// the words `backend_capabilities` uses (`cuda_graphs: not_cuda: …` on this CPU test host;
/// `cuda_feature_off` in a build without CUDA), else the decoder's own `graph_support`
/// refusal — here a dense llama with device positions off (`positions_host_scalar`) — and
/// names nothing with the switch off. The device-positions default is process state read
/// once, so each setting runs in a child process of its own.
#[test]
fn a_graph_load_names_a_decoder_the_runner_cannot_capture() {
    use core_llm::TextLlm;
    const CHILD: &str = "CANDLE_LLM_GRAPH_REFUSAL_TEST_CHILD";
    let Ok(setting) = std::env::var(CHILD) else {
        let name = std::thread::current().name().unwrap().to_owned();
        for setting in ["0", "1"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name, "--nocapture"])
                .env(CHILD, setting)
                .env(crate::primitives::DEVICE_POSITIONS_ENV, setting)
                .status()
                .unwrap();
            assert!(status.success(), "device positions {setting}");
        }
        return;
    };
    let (cfg, weights) = tiny_llama_parts();
    let dir = write_snapshot(&cfg, &weights, 40);
    let load = |graphs: bool| {
        super::LlamaProvider::load(&core_llm::LoadSpec {
            cuda_graphs: Some(graphs),
            ..spec_at(dir.path(), None)
        })
        .unwrap()
    };
    let provider = load(true);
    let fallbacks = provider.load_report().unwrap().fallbacks;
    // The device a graph load opens: under the load's switch, which on CUDA puts it on its
    // own stream (a device opened outside it would be on the legacy stream and refuse).
    let device = {
        let _graphs = crate::decode::cuda_graphs_scope(Some(true));
        crate::device::select_device().unwrap()
    };
    let device_refusal = crate::backend::cuda_graphs_device_refusal(&device);
    if !device.is_cuda() {
        // The CPU case: the device refuses every capture, named first, whatever the decoder.
        let reason = if cfg!(feature = "cuda") {
            "not_cuda"
        } else {
            "cuda_feature_off"
        };
        assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
        assert!(
            fallbacks[0].starts_with(&format!("cuda_graphs: {reason}: ")),
            "{fallbacks:?}"
        );
    }
    if setting == "1" {
        assert_eq!(provider.model.graph_support(), Ok(()), "capturable");
        assert_eq!(fallbacks, device_refusal.into_iter().collect::<Vec<_>>());
    } else {
        assert_eq!(
            fallbacks,
            vec![device_refusal.unwrap_or_else(|| "cuda_graphs: positions_host_scalar".into())]
        );
        assert!(load(false).load_report().unwrap().fallbacks.is_empty());
    }
}

/// sc-24441 (E3): the load names the device-and-build graph refusal ahead of the decoder's,
/// and the decoder's when the device can capture.
#[test]
fn the_device_graph_refusal_is_named_before_the_decoders() {
    let device = Some("cuda_graphs: not_cuda: CUDA graphs need a CUDA load device".to_string());
    assert_eq!(
        super::graph_load_refusal(device.clone(), Err("positions_host_scalar")),
        device
    );
    assert_eq!(super::graph_load_refusal(device.clone(), Ok(())), device);
    assert_eq!(
        super::graph_load_refusal(None, Err("positions_host_scalar")).as_deref(),
        Some("cuda_graphs: positions_host_scalar")
    );
    assert_eq!(super::graph_load_refusal(None, Ok(())), None);
    // A build without CUDA (or a CPU device in a CUDA build) refuses by the backend's words.
    let cpu = crate::backend::cuda_graphs_device_refusal(&candle_core::Device::Cpu).unwrap();
    let reason = if cfg!(feature = "cuda") {
        "not_cuda"
    } else {
        "cuda_feature_off"
    };
    assert_eq!(
        Some(cpu),
        crate::backend::capabilities_for_device("candle-cpu", &candle_core::Device::Cpu)
            .cuda_graphs
            .reason
    );
    assert!(
        crate::backend::cuda_graphs_device_refusal(&candle_core::Device::Cpu)
            .unwrap()
            .starts_with(&format!("cuda_graphs: {reason}: ")),
    );
}

/// sc-24441: the reference loop's record names the decode attention when the decoder's
/// device positions are active (its cached steps run it over the growing cache), for both
/// families, and the selector otherwise.
#[test]
fn the_reference_loop_reports_the_decode_attention_with_device_positions() {
    use crate::primitives::AttnFormulation;
    let mut causal = tiny_llama_for_admission();
    let (_, mut hybrid) = crate::models::qwen35::tests::text_model();
    assert_eq!(
        Decoder::Causal(tiny_llama_for_admission()).attn_formulation(),
        AttnFormulation::Gqa
    );
    assert_eq!(
        Decoder::Qwen35(crate::models::qwen35::tests::text_model().1).attn_formulation(),
        AttnFormulation::Gqa
    );
    causal.set_device_positions(true);
    hybrid.set_device_positions(true);
    let (causal, hybrid) = (Decoder::Causal(causal), Decoder::Qwen35(hybrid));
    assert_eq!(causal.attn_formulation(), AttnFormulation::DecodeAttention);
    assert_eq!(hybrid.attn_formulation(), AttnFormulation::DecodeAttention);
}

/// sc-24441 (E7): a decoder whose caches stage device positions is priced, on every engine
/// admission (the request, the Gemma-style spliced re-admission, a `draft_model` draft's
/// request), for the position buffers plus one step's decode-attention workspace — a
/// `K + 1`-query step over the whole cache — and for nothing when it does not. Both decoder
/// families; both device-positions settings; literal bytes (4 heads, width 8).
#[test]
fn device_positions_price_their_buffers_and_the_decode_attention_workspace() {
    use crate::primitives::DevicePositions;
    use candle_quant_kernels::decode_attention_workspace_bytes;
    let engine = |drafts| super::DecodeRoute::Engine {
        proposer: ProposerKind::PromptLookup,
        drafts,
    };
    let decoders = |on: bool| {
        let mut causal = tiny_llama_for_admission();
        causal.set_device_positions(on);
        let (_, mut hybrid) = crate::models::qwen35::tests::text_model();
        hybrid.set_device_positions(on);
        [Decoder::Causal(causal), Decoder::Qwen35(hybrid)]
    };
    // (prompt, budget, K, the device-positions bytes). Positions = prompt + budget; the
    // cache spans positions + K; 64-key chunks (sc-24446).
    let cases = [
        (40usize, 24u32, 3u32, 1_416u64),
        (40, 24, 0, 296),
        (300, 24, 3, 3_976),
    ];
    for (off, on) in decoders(false).iter().zip(decoders(true).iter()) {
        for &(prompt, budget, k, bytes) in &cases {
            let positions = prompt + budget as usize;
            assert_eq!(
                bytes as usize,
                DevicePositions::BYTES
                    + decode_attention_workspace_bytes(
                        1,
                        4,
                        k as usize + 1,
                        positions + k as usize,
                        8,
                    ),
            );
            assert_eq!(off.device_step_bytes(k, positions as u64), Some(0));
            assert_eq!(on.device_step_bytes(k, positions as u64), Some(bytes));
            for graphs in [false, true] {
                let priced = |d: &Decoder| {
                    super::priced_request_bytes(d, engine(k), prompt, budget, 0, graphs).unwrap()
                };
                assert_eq!(priced(on), priced(off) + bytes, "request, graphs={graphs}");
                let spliced =
                    |d: &Decoder| super::spliced_prompt_bytes(d, positions, k, graphs).unwrap();
                assert_eq!(
                    spliced(on),
                    spliced(off) + bytes,
                    "spliced, graphs={graphs}"
                );
            }
            // The reference loop runs no static cache: nothing to price.
            let reference = |d: &Decoder| {
                super::priced_request_bytes(
                    d,
                    super::DecodeRoute::Reference,
                    prompt,
                    budget,
                    0,
                    false,
                )
            };
            assert_eq!(reference(on), reference(off));
        }
        // A draft (sc-24436) caches the reach plus `K + 1` and replays up to `K + 1` tokens:
        // 4 queries over 40 + 24 + 4 positions.
        assert_eq!(
            super::draft_request_bytes(on, 40, 24, 3).unwrap(),
            super::draft_request_bytes(off, 40, 24, 3).unwrap() + 1_416
        );
    }
}

/// sc-24140 (E6): the Gemma 4 soft-token splice re-admits its expanded prompt with the
/// static preallocation and, when the CUDA-graph runner wraps the engine, the runner's graph
/// workspace over the same reach (one step token count: no proposer).
#[test]
fn the_spliced_prompt_readmission_prices_the_graph_workspace_when_the_runner_is_on() {
    let decoder = Decoder::Causal(tiny_gemma4_for_admission_with(false));
    let capacity = 40usize;
    let kv = decoder.static_kv_bytes(capacity) as u64;
    assert!(kv > 0);
    assert_eq!(
        super::spliced_prompt_bytes(&decoder, capacity, 0, false),
        Some(kv)
    );
    let graphs = crate::decode::graph_workspace_admission_bytes(
        &decoder.step_memory_geometry(0),
        capacity as u64,
        1,
    )
    .unwrap();
    assert!(graphs > 0);
    assert_eq!(
        super::spliced_prompt_bytes(&decoder, capacity, 0, true),
        Some(kv + graphs)
    );
    // A prompt-lookup plan (sc-24433) preallocates the verify overshoot too, and its graphs
    // span `1 ..= K + 1` step token counts over the overshot reach.
    let k = 4u32;
    let overshot = decoder.static_kv_bytes(capacity + k as usize) as u64;
    assert!(overshot > kv);
    assert_eq!(
        super::spliced_prompt_bytes(&decoder, capacity, k, false),
        Some(overshot)
    );
    let graphs_k = crate::decode::graph_workspace_admission_bytes(
        &decoder.step_memory_geometry(k as usize),
        capacity as u64 + u64::from(k),
        k + 1,
    )
    .unwrap();
    assert_eq!(
        super::spliced_prompt_bytes(&decoder, capacity, k, true),
        Some(overshot + graphs_k)
    );
}

fn nvfp4_spec(source: &str) -> core_llm::LoadSpec {
    core_llm::LoadSpec {
        source: source.into(),
        projector_source: None,
        quantize: Some(core_llm::Quantize::Nvfp4),
        cuda_graphs: None,
        mtp_head_source: None,
        prefix_cache_bytes: None,
        draft_source: None,
    }
}

/// sc-24135 AC2: NVFP4 on a CPU device is a typed refusal naming the capability, settled by the
/// helper `LlamaProvider::load` calls before the accelerator gate, admission or any weight read.
#[test]
fn nvfp4_on_cpu_is_a_typed_refusal_naming_the_capability() {
    let device = candle_core::Device::Cpu;
    match super::nvfp4_format(&nvfp4_spec("snapshot-dir"), &device) {
        Err(core_llm::Error::Unsupported(msg)) => {
            assert!(msg.starts_with("nvfp4: "), "names the capability: {msg}");
            assert!(msg.contains("sm_120"), "names the floor: {msg}");
            assert!(msg.contains("Cpu"), "names the device: {msg}");
        }
        other => panic!("expected a typed Unsupported refusal, got {other:?}"),
    }
    // Non-NVFP4 requests never touch the gate.
    for quantize in [
        None,
        Some(core_llm::Quantize::Q4),
        Some(core_llm::Quantize::Q8),
    ] {
        let spec = core_llm::LoadSpec {
            quantize,
            ..nvfp4_spec("snapshot-dir")
        };
        assert!(super::nvfp4_format(&spec, &device).unwrap().is_none());
    }
    // A GGUF source is refused by name on any device (it is already block-quantized).
    match super::nvfp4_format(&nvfp4_spec("model.gguf"), &device) {
        Err(core_llm::Error::Unsupported(msg)) => assert!(msg.starts_with("nvfp4: "), "{msg}"),
        other => panic!("expected a typed Unsupported refusal, got {other:?}"),
    }
}

/// sc-24135 AC2 with a mocked sub-sm_120 capability: the device gate's refusal reaches the
/// backend-neutral contract as `Unsupported`, naming the capability, the floor and the device.
#[test]
fn a_sub_sm120_refusal_reaches_the_contract_as_unsupported() {
    for cap in [(8, 9), (9, 0), (10, 0)] {
        let refusal = candle_quant_kernels::nvfp4_refusal_for_compute_cap(cap).unwrap();
        match super::to_core(crate::Error::Nvfp4Refused(refusal)) {
            core_llm::Error::Unsupported(msg) => {
                assert!(msg.starts_with("nvfp4: "), "{msg}");
                assert!(msg.contains("sm_120"), "{msg}");
                assert!(msg.contains(&format!("sm_{}{}", cap.0, cap.1)), "{msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}

/// sc-24135 AC2 through the real gate: `nvfp4_format` → `Nvfp4Context::require_with` (a real
/// cuBLASLt handle, the capability probe mocked to a sub-sm_120 device) → the typed refusal →
/// the contract's `Unsupported`, naming the capability, the floor and the mocked device.
#[cfg(feature = "cuda")]
#[test]
fn nvfp4_format_refuses_a_mocked_sub_sm120_device_as_unsupported() {
    let Ok(device) = crate::device::new_cuda_for_test() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    for cap in [(8, 9), (10, 0)] {
        let gate = |d: &candle_core::Device| {
            crate::primitives::projection::ProjectionFormat::nvfp4_with_cap_probe(d, |_| Ok(cap))
        };
        match super::nvfp4_format_with(&nvfp4_spec("snapshot-dir"), &device, gate) {
            Err(core_llm::Error::Unsupported(msg)) => {
                assert!(msg.starts_with("nvfp4: "), "{msg}");
                assert!(msg.contains("sm_120"), "names the floor: {msg}");
                assert!(
                    msg.contains(&format!("sm_{}{}", cap.0, cap.1)),
                    "names the device: {msg}"
                );
            }
            Err(other) => panic!("expected Unsupported for {cap:?}, got {other:?}"),
            Ok(_) => panic!("a mocked sm_{}{} device served NVFP4", cap.0, cap.1),
        }
    }
}

/// sc-24140: the family rule `nvfp4_format` applies before admission and the device gate. The
/// qwen3_5 hybrid **and** the llama family (a `CausalLm` — Qwen3-8B's `qwen3`, Llama/Mistral,
/// Gemma, …) reach the gate; a Prism/Bonsai snapshot (already packed affine-2) is refused by
/// name before it, so no memory or device error can mask the refusal. The refusing gate below
/// panics if reached.
#[test]
fn nvfp4_family_rule_serves_qwen35_and_the_llama_family_and_refuses_prism() {
    let root = tempfile::tempdir().unwrap();
    let spec = nvfp4_spec(&root.path().to_string_lossy());
    for config in [
        r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#,
        r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
        r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        r#"{"architectures":["MistralForCausalLM"],"model_type":"mistral"}"#,
        r#"{"architectures":["Gemma2ForCausalLM"],"model_type":"gemma2"}"#,
        r#"{"architectures":["DeepseekV2ForCausalLM"],"model_type":"deepseek_v2"}"#,
        r#"{"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl"}"#,
    ] {
        std::fs::write(root.path().join("config.json"), config).unwrap();
        let reached = std::cell::Cell::new(false);
        let gate = |_: &candle_core::Device| -> crate::Result<_> {
            reached.set(true);
            Err(crate::Error::Unsupported("nvfp4: gate reached".into()))
        };
        match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
            Err(core_llm::Error::Unsupported(msg)) => assert_eq!(msg, "nvfp4: gate reached"),
            other => panic!("{config}: expected the gate's refusal, got {other:?}"),
        }
        assert!(reached.get(), "{config} must reach the device gate");
    }

    std::fs::write(
        root.path().join("config.json"),
        r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"prism_hadamard_qwen35"}"#,
    )
    .unwrap();
    let gate = |_: &candle_core::Device| -> crate::Result<_> {
        panic!("the device gate ran before the family refusal")
    };
    match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
        Err(core_llm::Error::Unsupported(msg)) => {
            assert!(msg.starts_with("nvfp4: "), "{msg}");
            assert!(msg.contains("Prism/Bonsai"), "names the family: {msg}");
            assert!(msg.contains("dense snapshot"), "names the reason: {msg}");
        }
        other => panic!("expected the family refusal, got {other:?}"),
    }
}

/// sc-24140 feature-end review: a packed MLX-affine snapshot — a `quantization` block over
/// `<stem>.weight` + `<stem>.scales` — is refused by the model gate, from the index's
/// `weight_map` or the shard headers (never tensor data), on the llama family and the hybrid
/// alike; the loader would otherwise refuse its triples only after reading every weight. A
/// prepared Q4/Q8 snapshot (the block over dense weights), a triple without the block and an
/// orphan `.scales` are not packed. The device gate below panics if the refusal misses it.
#[test]
fn nvfp4_gate_refuses_a_packed_mlx_affine_snapshot_before_the_device_gate() {
    let llama = serde_json::json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"});
    let block = serde_json::json!({"group_size": 64, "bits": 4});
    let with_block = |mut config: serde_json::Value| {
        config["quantization"] = block.clone();
        config
    };
    let triple = [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.scales",
        "model.layers.0.mlp.up_proj.biases",
    ];
    let refusal = |config: &serde_json::Value, names: &[&str], index: bool| {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
        super::write_test_snapshot_tensors(root.path(), names, index);
        let gate = |_: &candle_core::Device| -> crate::Result<_> {
            Err(crate::Error::Unsupported("nvfp4: gate reached".into()))
        };
        let spec = nvfp4_spec(&root.path().to_string_lossy());
        match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
            Err(core_llm::Error::Unsupported(msg)) => msg,
            other => panic!("{config}: expected an Unsupported refusal, got {other:?}"),
        }
    };
    let hybrid = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"quantization": block.clone()}
    });
    for (config, index) in [
        (with_block(llama.clone()), true),
        (with_block(llama.clone()), false),
        (hybrid, false),
    ] {
        let msg = refusal(&config, &triple, index);
        assert!(msg.starts_with("nvfp4: "), "{msg}");
        assert!(
            msg.contains("packed MLX-affine"),
            "{config} (index {index}): {msg}"
        );
        assert!(msg.contains("model.layers.0.mlp.up_proj.scales"), "{msg}");
        assert!(msg.contains("dense snapshot"), "{msg}");
        assert!(!msg.contains("  "), "{msg:?}");
    }
    // Not packed: each reaches the device gate.
    let dense = [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];
    for (config, names) in [
        (with_block(llama.clone()), &dense[..]),
        (llama.clone(), &triple[..]),
        (
            with_block(llama.clone()),
            &[
                "model.layers.0.mlp.up_proj.scales",
                "model.layers.0.mlp.down_proj.weight",
            ][..],
        ),
    ] {
        for index in [true, false] {
            assert_eq!(
                refusal(&config, names, index),
                "nvfp4: gate reached",
                "{config} {names:?} (index {index})"
            );
        }
    }
}

/// On a build with no CUDA backend the whole load refuses NVFP4 before reading a byte: the
/// source here does not exist, so any later step would have failed with a Load/IO error.
#[cfg(not(feature = "cuda"))]
#[test]
fn nvfp4_load_without_cuda_refuses_before_reading_the_snapshot() {
    // A snapshot path that does not exist, inside a self-removing root.
    let root = tempfile::tempdir().unwrap();
    let missing = root.path().join("no-snapshot-here");
    match super::LlamaProvider::load(&nvfp4_spec(&missing.to_string_lossy())) {
        Err(core_llm::Error::Unsupported(msg)) => assert!(msg.starts_with("nvfp4: "), "{msg}"),
        Err(other) => panic!("expected the NVFP4 refusal first, got {other:?}"),
        Ok(_) => panic!("NVFP4 cannot load without CUDA"),
    }
}

/// Write `config` and `weights` as a one-shard snapshot — float tensors in bf16, as shipped
/// snapshots are; packed `U32` codes as they are — beside a synthetic `vocab`-token tokenizer.
fn write_snapshot(
    config: &serde_json::Value,
    weights: &HashMap<String, Tensor>,
    vocab: usize,
) -> tempfile::TempDir {
    write_snapshot_as(config, weights, vocab, |_| candle_core::DType::BF16)
}

/// [`write_snapshot`] with each float tensor stored in `dtype(key)`.
fn write_snapshot_as(
    config: &serde_json::Value,
    weights: &HashMap<String, Tensor>,
    vocab: usize,
    dtype: impl Fn(&str) -> candle_core::DType,
) -> tempfile::TempDir {
    use candle_core::DType;
    let dir = tempfile::Builder::new()
        .prefix("candle-admission-")
        .tempdir()
        .unwrap();
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        synthetic_tokenizer_json(vocab),
    )
    .unwrap();
    let stored: HashMap<String, Tensor> = weights
        .iter()
        .map(|(key, t)| {
            let t = match t.dtype() {
                DType::U32 => t.clone(),
                _ => t.to_dtype(dtype(key)).unwrap(),
            };
            (key.clone(), t)
        })
        .collect();
    candle_core::safetensors::save(&stored, dir.path().join("model.safetensors")).unwrap();
    dir
}

/// The graph runner is pinned off: a CUDA estimate otherwise follows the process switch
/// (the parameter-cache term, sc-24441), which other tests flip concurrently.
fn spec_at(source: &std::path::Path, quantize: Option<core_llm::Quantize>) -> core_llm::LoadSpec {
    core_llm::LoadSpec {
        quantize,
        cuda_graphs: Some(false),
        ..core_llm::LoadSpec::dense(source.display().to_string())
    }
}

/// The GGML copy a load really built, on the device: the census's GGML blocks plus candle's
/// CUDA row padding (`MATRIX_ROW_PADDING`, 512 elements) past each tensor.
fn ggml_copy_built(
    census: &crate::primitives::projection::WeightCensus,
    dtype: candle_core::quantized::GgmlDType,
) -> u64 {
    let padding = (512 * dtype.type_size() / dtype.block_size()) as u64;
    census.projections.ggml.resident_bytes + census.projections.ggml.count * padding
}

fn loaded_census(spec: &core_llm::LoadSpec) -> crate::primitives::projection::WeightCensus {
    super::LlamaProvider::load(spec)
        .expect("load the synthetic snapshot")
        .load_record()
        .census
        .expect("a census")
}

/// E6 (sc-24140): the device bound is the load's resident source, 25 percent headroom over
/// it, and the copies the loader builds beside it; the host domain is unchanged except for a
/// llama-family GGUF on a host device, whose dequantized f32 map *is* host memory.
#[test]
fn admission_prices_the_working_set_beside_the_resident_source() {
    let payload = 16_400_000_000u64; // ~Qwen3-8B bf16
    let staging = payload / 10;
    let bound =
        |cuda, working| load_memory_requirements(payload, staging, 0, cuda, working).unwrap();
    let copy = 7_400_000_000;
    let dense = LoadWorkingSet::resident_payload(payload, 0, 0, false);
    let quantized = LoadWorkingSet::resident_payload(payload, 0, copy, false);
    assert_eq!(bound(true, dense), (staging, Some(payload + payload / 4)));
    assert_eq!(
        bound(true, quantized),
        (staging, Some(payload + payload / 4 + copy)),
        "the copy is built on the device beside the source, not on the host"
    );
    assert_eq!(bound(false, quantized), (3 * payload, None));
    let prism = LoadWorkingSet::resident_payload(payload, 0, 0, true);
    assert_eq!(
        bound(true, prism),
        (2 * payload, Some(payload + payload / 4))
    );
    // An f16 snapshot: the bf16 casts sit beside the source, on the device only.
    let cast = payload;
    let f16 = LoadWorkingSet::resident_payload(payload, cast, copy, false);
    assert_eq!(
        bound(true, f16),
        (staging, Some(payload + payload / 4 + cast + copy))
    );
    assert_eq!(bound(false, f16), (3 * payload, None));

    // A llama-family GGUF: its f32 map is the source, its bf16 casts and copy sit beside it.
    let (map, cast) = (32_800_000_000u64, 2_500_000_000u64);
    let gguf = LoadWorkingSet {
        source: map,
        cast,
        copy,
        held: 0,
        host: HostRule::Gguf,
    };
    let working = map + map / 4 + cast + copy;
    assert_eq!(bound(true, gguf), (2 * payload, Some(working)));
    assert_eq!(bound(false, gguf), (working, None));
}

/// E6 (sc-24140 review): a quantizing load is priced at exactly the copy its loader builds —
/// the layer projections only, never the dense embeddings and head the whole-payload rule
/// charged for — checked against the census of the real load for Q8_0 and Q4_K, with NVFP4's
/// eligible shapes priced and its 40-row head (which `load_eligible` keeps dense) not.
#[test]
fn quantized_load_estimates_price_exactly_the_copy_the_loader_builds() {
    use crate::primitives::projection::nvfp4_device_bytes;
    use candle_core::quantized::GgmlDType;
    use core_llm::Quantize;
    let (cfg, weights) = tiny_llama_parts();
    let dir = write_snapshot(&cfg, &weights, 40);
    let estimate = |quantize, cuda| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), cuda).unwrap()
    };
    let dense = estimate(None, true);
    let payload = dense.payload_bytes;
    let source_bytes: u64 = weights.values().map(|t| (t.elem_count() * 2) as u64).sum();
    assert!(payload >= source_bytes, "the file holds every bf16 source");
    assert_eq!(dense.device_required_bytes, Some(payload + payload / 4));
    assert_eq!(
        (
            dense.source_bytes,
            dense.cast_copy_bytes,
            dense.quantized_copy_bytes
        ),
        (payload, 0, 0)
    );

    let q8 = estimate(Some(Quantize::Q8), true);
    let census = loaded_census(&spec_at(dir.path(), Some(Quantize::Q8)));
    assert_eq!(
        census.projections.ggml.count, 21,
        "3 layers x 7 projections"
    );
    assert_eq!(
        q8.quantized_copy_bytes,
        ggml_copy_built(&census, GgmlDType::Q8_0),
        "exactly the Q8_0 copy the loader built"
    );
    let (wider, wider_weights) = llama_parts(80, 32, 64, 4, 2, 3);
    let wider = write_snapshot(&wider, &wider_weights, 80);
    assert_eq!(
        super::LlamaProvider::load_memory_estimate(
            &spec_at(wider.path(), Some(Quantize::Q8)),
            true
        )
        .unwrap()
        .quantized_copy_bytes,
        q8.quantized_copy_bytes,
        "the dense embedding and head are never priced: a wider vocabulary leaves the copy as is"
    );
    assert_eq!(
        q8.device_required_bytes,
        Some(payload + payload / 4 + q8.quantized_copy_bytes)
    );
    assert_eq!(q8.host_required_bytes, dense.host_required_bytes);
    assert!(
        source_bytes + q8.quantized_copy_bytes > dense.device_required_bytes.unwrap(),
        "sources plus copy (the load's peak) exceed the dense-only bound, the pricing at R"
    );
    let off_cuda = estimate(Some(Quantize::Q8), false);
    assert_eq!(
        (
            off_cuda.device_required_bytes,
            off_cuda.quantized_copy_bytes
        ),
        (None, 0)
    );

    // NVFP4: every eligible layer projection; the 40-row head fails `N % 16` and stays dense.
    let (hidden, inter, kv) = (32, 64, 16);
    let layer: u64 = [
        (hidden, hidden),
        (kv, hidden),
        (kv, hidden),
        (hidden, hidden),
        (inter, hidden),
        (inter, hidden),
        (hidden, inter),
    ]
    .iter()
    .map(|&(rows, cols)| nvfp4_device_bytes(rows, cols).unwrap())
    .sum();
    assert_eq!(nvfp4_device_bytes(40, hidden), None);
    assert_eq!(
        estimate(Some(Quantize::Nvfp4), true).quantized_copy_bytes,
        3 * layer
    );

    // Q4_K needs 256-wide rows: the same check on a geometry it can quantize.
    let (cfg, weights) = llama_parts(64, 256, 512, 4, 2, 2);
    let dir = write_snapshot(&cfg, &weights, 64);
    let spec = spec_at(dir.path(), Some(Quantize::Q4));
    let q4 = super::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
    let census = loaded_census(&spec);
    assert_eq!(census.projections.ggml.count, 14);
    assert_eq!(
        q4.quantized_copy_bytes,
        ggml_copy_built(&census, GgmlDType::Q4K)
    );
}

/// sc-24140 review: the decoder casts every tensor it keeps dense to bf16 (`to_dtype`, which
/// shares a bf16 tensor's storage but copies any other float), so a CUDA load of an f16 or
/// f32 snapshot holds a 2-byte cast of each such tensor beside its source. Admission charges
/// it explicitly — not out of the 25 percent headroom — for every tensor the loader keeps
/// dense, and not for a projection it consumes whole into its quantized copy (or an NVFP4
/// shape it keeps dense, which it charges). A bf16 snapshot is charged none.
#[test]
fn a_non_bf16_snapshot_is_charged_the_bf16_cast_beside_its_source() {
    use crate::primitives::projection::nvfp4_device_bytes;
    use candle_core::quantized::GgmlDType;
    use candle_core::DType;
    use core_llm::Quantize;
    let (cfg, weights) = tiny_llama_parts();
    let elems = |keep: &dyn Fn(&str) -> bool| -> u64 {
        weights
            .iter()
            .filter(|(key, _)| keep(key))
            .map(|(_, t)| t.elem_count() as u64)
            .sum()
    };
    let all = elems(&|_| true);
    let projections = elems(&|key| key.contains("_proj."));
    let estimate = |dir: &tempfile::TempDir, quantize, cuda| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), cuda).unwrap()
    };

    let bf16 = write_snapshot(&cfg, &weights, 40);
    assert_eq!(estimate(&bf16, None, true).cast_copy_bytes, 0);
    for dtype in [DType::F16, DType::F32] {
        let dir = write_snapshot_as(&cfg, &weights, 40, |_| dtype);
        let dense = estimate(&dir, None, true);
        let payload = dense.payload_bytes;
        assert_eq!(
            dense.cast_copy_bytes,
            2 * all,
            "{dtype:?}: every tensor, cast to bf16"
        );
        assert_eq!(
            dense.device_required_bytes,
            Some(payload + payload / 4 + 2 * all),
            "{dtype:?}"
        );
        let q8 = estimate(&dir, Some(Quantize::Q8), true);
        assert_eq!(
            q8.cast_copy_bytes,
            2 * (all - projections),
            "{dtype:?}: the quantized projections are consumed, not cast"
        );
        assert_eq!(
            q8.quantized_copy_bytes,
            estimate(&bf16, Some(Quantize::Q8), true).quantized_copy_bytes
        );
        // NVFP4 also consumes the layer projections; the 40-row head stays dense, cast.
        assert_eq!(nvfp4_device_bytes(40, 32), None);
        assert_eq!(
            estimate(&dir, Some(Quantize::Nvfp4), true).cast_copy_bytes,
            2 * (all - projections)
        );
        let host = estimate(&dir, None, false);
        assert_eq!(
            (host.cast_copy_bytes, host.host_required_bytes),
            (0, 3 * payload),
            "{dtype:?}: off CUDA the host rule covers the f32 conversion"
        );
        // The loader really keeps its projections quantized from this source.
        let census = loaded_census(&spec_at(dir.path(), Some(Quantize::Q8)));
        assert_eq!(
            q8.quantized_copy_bytes,
            ggml_copy_built(&census, GgmlDType::Q8_0)
        );
    }

    // A bf16 snapshot with f32 norms (a common export): only the norms are cast.
    let mixed = write_snapshot_as(&cfg, &weights, 40, |key| {
        if key.contains("norm") {
            DType::F32
        } else {
            DType::BF16
        }
    });
    assert_eq!(
        estimate(&mixed, None, true).cast_copy_bytes,
        2 * elems(&|key| key.contains("norm"))
    );
}

/// sc-24140 review: a persisted `quantization` block is priced only where the loader honours
/// it — a llama-family snapshot re-quantizes on load, the qwen3_5 loader reads only the
/// request and keeps the projections dense.
#[test]
fn a_persisted_block_is_priced_only_where_the_loader_honours_it() {
    use candle_core::quantized::GgmlDType;
    use core_llm::Quantize;
    let (mut cfg, weights) = tiny_llama_parts();
    cfg["quantization"] = serde_json::json!({ "bits": 8 });
    let dir = write_snapshot(&cfg, &weights, 40);
    let spec = spec_at(dir.path(), None);
    let estimate = super::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
    let census = loaded_census(&spec);
    assert_eq!(
        census.projections.ggml.count, 21,
        "the llama loader re-quantizes"
    );
    assert_eq!(
        estimate.quantized_copy_bytes,
        ggml_copy_built(&census, GgmlDType::Q8_0)
    );

    let dir = synthetic_qwen35_snapshot_without_mtp();
    let path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    config["quantization"] = serde_json::json!({ "bits": 8 });
    config["text_config"]["quantization"] = serde_json::json!({ "bits": 8 });
    std::fs::write(&path, config.to_string()).unwrap();
    let estimate = |quantize| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), true).unwrap()
    };
    assert_eq!(
        estimate(None).quantized_copy_bytes,
        0,
        "qwen3_5 ignores the persisted block"
    );
    let census = loaded_census(&spec_at(dir.path(), None));
    assert_eq!(
        census.projections.ggml.count, 0,
        "and loads dense, as priced"
    );
    assert!(
        estimate(Some(Quantize::Q8)).quantized_copy_bytes > 0,
        "an explicit request is priced"
    );
}

/// sc-24140 review: a GGML format over an MLX-affine triple is priced as the Q8_0 repack only
/// where the loader performs one — an 8-bit format over an 8-bit triple, at the repacked
/// weight's Q8_0 bytes (checked against the real load). A 4-bit triple costs nothing: the
/// loader refuses it with its own typed error, under the persisted 4-bit format and under an
/// explicit Q8 request alike.
#[test]
fn an_affine_triple_is_priced_only_where_the_loader_repacks_it() {
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device};
    use core_llm::Quantize;
    let (cfg, dense_weights) = tiny_llama_parts();
    let stem = "model.layers.0.self_attn.q_proj";
    // The q projection [32, 32] as an affine triple with one 32-wide group per row.
    let with_triple = |bits: usize| {
        let mut weights = dense_weights.clone();
        let codes = Tensor::zeros((32, 32 * bits / 32), DType::U32, &Device::Cpu).unwrap();
        let group = Tensor::ones((32, 1), DType::F32, &Device::Cpu).unwrap();
        weights.insert(format!("{stem}.weight"), codes);
        weights.insert(format!("{stem}.scales"), group.clone());
        weights.insert(format!("{stem}.biases"), group);
        let mut cfg = cfg.clone();
        cfg["quantization"] = serde_json::json!({ "bits": bits, "group_size": 32 });
        write_snapshot(&cfg, &weights, 40)
    };
    let without_q = {
        let mut weights = dense_weights.clone();
        weights.remove(&format!("{stem}.weight"));
        weights
    };
    let copy = |dir: &tempfile::TempDir, quantize| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), true)
            .unwrap()
            .quantized_copy_bytes
    };

    let eight = with_triple(8);
    let census = loaded_census(&spec_at(eight.path(), None));
    assert_eq!(
        census.projections.ggml.count, 21,
        "the triple repacked to Q8_0"
    );
    assert_eq!(
        copy(&eight, None),
        ggml_copy_built(&census, GgmlDType::Q8_0)
    );

    let four = with_triple(4);
    for (quantize, refusal, bits) in [
        (None, "MLX affine projection requires Q8", 4),
        (Some(Quantize::Q8), "invalid MLX affine Q8 triple", 8),
    ] {
        let mut cfg = cfg.clone();
        cfg["quantization"] = serde_json::json!({ "bits": bits, "group_size": 32 });
        let others = write_snapshot(&cfg, &without_q, 40);
        assert_eq!(
            copy(&four, quantize),
            copy(&others, quantize),
            "{quantize:?}: the 4-bit triple is not priced"
        );
        match super::LlamaProvider::load(&spec_at(four.path(), quantize)) {
            Err(error) => assert!(error.to_string().contains(refusal), "{error}"),
            Ok(_) => panic!("a 4-bit affine triple loaded under {quantize:?}"),
        }
    }
}

/// The synthetic qwen3_5 decoder (every projection 32 or 64 wide, so Q8_0 can quantize it)
/// written as a snapshot: with the MTP head, or with the MoE bank instead.
fn q8_capable_qwen35_snapshot(moe: bool) -> tempfile::TempDir {
    qwen35_snapshot(moe, !moe, true)
}

/// [`q8_capable_qwen35_snapshot`] with the MoE bank and the MTP head chosen independently.
/// A MoE snapshot's MTP predictor layer carries a sparse-MoE FFN (router, experts, shared
/// expert) instead of the dense MLP — as the 35B-A3B checkpoint ships it (sc-24438) — with every
/// MoE tensor seeded non-zero random, the experts fused (Qwen3.6) or per expert (Qwen3.5).
fn qwen35_snapshot(moe: bool, mtp: bool, fused: bool) -> tempfile::TempDir {
    let mut json = crate::models::qwen35::tests::synthetic_cfg_json();
    let text = json["text_config"].as_object_mut().unwrap();
    text.insert("linear_value_head_dim".into(), serde_json::json!(8));
    if moe {
        text.insert("model_type".into(), serde_json::json!("qwen3_5_moe_text"));
        text.insert("num_experts".into(), serde_json::json!(3));
        text.insert("num_experts_per_tok".into(), serde_json::json!(2));
        text.insert("moe_intermediate_size".into(), serde_json::json!(32));
        text.insert(
            "shared_expert_intermediate_size".into(),
            serde_json::json!(32),
        );
    }
    text.insert(
        "mtp_num_hidden_layers".into(),
        serde_json::json!(u32::from(mtp)),
    );
    let cfg = crate::models::Qwen35Config::from_json(&json).unwrap();
    let tensors: HashMap<String, Tensor> = if moe && mtp {
        crate::models::qwen35::tests::seeded_moe_mtp_tensors(&cfg, fused)
    } else {
        let weights = crate::models::qwen35::tests::synthetic_snapshot_weights(&cfg);
        weights
            .keys()
            .map(|k| (k.to_string(), weights.get(k).unwrap().clone()))
            .collect()
    };
    let mut config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
    });
    config["text_config"] = json["text_config"].clone();
    write_snapshot(&config, &tensors, cfg.vocab_size as usize)
}

/// E6 (sc-24140 review) on the qwen3_5 hybrid: the priced copy is exactly the one the loader
/// builds — the Gated DeltaNet and attention projections, the MLP (or every MoE expert slice
/// and the shared expert), the head (quantized under every format here) and the MTP head —
/// and never the per-head `in_proj_a` / `in_proj_b`, the router or the embeddings.
#[test]
fn qwen35_q8_estimate_prices_exactly_the_copy_the_loader_builds() {
    use crate::primitives::projection::ggml_device_bytes;
    use candle_core::quantized::GgmlDType;
    use core_llm::Quantize;
    let estimate = |dir: &tempfile::TempDir| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), Some(Quantize::Q8)), true)
            .unwrap()
            .quantized_copy_bytes
    };
    // Dense: 3 DeltaNet layers x 3 + 1 attention layer x 4 + 4 MLPs x 3 + the head + the MTP's
    // q/k/v/o, MLP and fc — against the census of the real load.
    let dense = q8_capable_qwen35_snapshot(false);
    let census = loaded_census(&spec_at(dense.path(), Some(Quantize::Q8)));
    assert_eq!(census.projections.ggml.count, 34);
    assert_eq!(estimate(&dense), ggml_copy_built(&census, GgmlDType::Q8_0));

    // MoE: the same mixers and head, and per layer every expert's gate / up (sliced from the
    // stacked `gate_up_proj`) and down, plus the shared expert — each a [32, 32] Q8_0 tensor.
    let q8 = |rows: u64, cols: u64| ggml_device_bytes(rows * cols, GgmlDType::Q8_0).unwrap();
    let linear = q8(48, 32) + q8(32, 32) + q8(32, 32);
    let attention = q8(64, 32) + 2 * q8(16, 32) + q8(32, 32);
    let ffn = (3 * 3 + 3) * q8(32, 32);
    let moe = q8_capable_qwen35_snapshot(true);
    assert_eq!(
        estimate(&moe),
        3 * linear + attention + 4 * ffn + q8(50, 32)
    );
    let census = loaded_census(&spec_at(moe.path(), Some(Quantize::Q8)));
    assert_eq!(census.projections.ggml.count, 62);
    assert_eq!(estimate(&moe), ggml_copy_built(&census, GgmlDType::Q8_0));
}

/// sc-24140 review: a qwen3_5 MoE snapshot loads under `Quantize::Q8` on every device and
/// decodes. On a host device the loader hands the GGML quantizer each expert as a view
/// narrowed out of the stacked tensors, which candle's quantizer read from the storage's
/// start — a size-check panic at load (debug) or every expert quantized from the first
/// expert's rows (release) — until `QuantizedLinear::quantize` compacted its source.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_qwen35_moe_snapshot_loads_q8_and_decodes() {
    use core_llm::{Message, Quantize, Sampling, TextLlm, TextLlmRequest};
    let dir = q8_capable_qwen35_snapshot(true);
    let provider = super::LlamaProvider::load(&spec_at(dir.path(), Some(Quantize::Q8)))
        .expect("load the MoE snapshot Q8");
    assert!(provider.is_quantized());
    let request = TextLlmRequest {
        messages: vec![Message::user("t3 t7 t11 t2")],
        sampling: Sampling::greedy(),
        max_new_tokens: 4,
        seed: Some(0),
        ..Default::default()
    };
    let out = provider.generate(&request, &mut |_| {}).expect("decode");
    assert_eq!(out.usage.generated_tokens, 4);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_qwen35_moe_snapshot_loads_q8_and_decodes");
}

/// sc-24440: a CUDA load prices the indexed MoE dispatch's expert-address tables — every
/// MoE layer's three projections at the widest entry per expert — and a dense model none.
#[test]
fn a_load_prices_the_moe_expert_tables() {
    let cfg = |extra: serde_json::Value| {
        let mut c = serde_json::json!({
            "model_type": "qwen2_moe", "architectures": ["Qwen2MoeForCausalLM"],
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 5,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 64,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 64,
        });
        c.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        c
    };
    let moe = cfg(serde_json::json!({
        "num_experts": 12, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 64,
    }));
    let priced = super::PricedDecoder::from_config(Some(&moe));
    assert!(matches!(priced, super::PricedDecoder::Causal(_)));
    let per = candle_quant_kernels::MAX_TABLE_BYTES_PER_EXPERT as u64;
    assert_eq!(priced.moe_indexed_table_bytes(), 5 * 3 * 12 * per);
    let dense = super::PricedDecoder::from_config(Some(&cfg(serde_json::json!({}))));
    assert_eq!(dense.moe_indexed_table_bytes(), 0);
}

/// sc-24440 review: a CUDA load estimate's device requirement carries the indexed MoE
/// dispatch's expert tables — the same snapshot with its experts declared away (the config
/// alone changes; every tensor is the same) requires exactly the tables' bytes less on the
/// device, and nothing less off it.
#[test]
fn a_cuda_load_estimate_requires_the_moe_expert_tables_on_the_device() {
    use core_llm::LoadSpec;
    let dir = qwen35_snapshot(true, false, true);
    let spec = LoadSpec::dense(dir.path().display().to_string());
    let device = |cuda: bool| {
        super::LlamaProvider::load_memory_estimate(&spec, cuda)
            .unwrap()
            .device_required_bytes
    };
    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    let tables = super::PricedDecoder::from_config(Some(&config)).moe_indexed_table_bytes();
    assert!(tables > 0, "the MoE snapshot prices expert tables");
    let (moe_cuda, moe_host) = (device(true).unwrap(), device(false));
    fn strip(v: &mut serde_json::Value) {
        if let Some(object) = v.as_object_mut() {
            object.remove("num_experts");
            object.values_mut().for_each(strip);
        }
    }
    strip(&mut config);
    std::fs::write(&config_path, config.to_string()).unwrap();
    assert_eq!(
        super::PricedDecoder::from_config(Some(&config)).moe_indexed_table_bytes(),
        0,
        "the stripped config is dense"
    );
    assert_eq!(moe_cuda - device(true).unwrap(), tables);
    assert_eq!(moe_host, device(false), "off CUDA there are no tables");
}

/// sc-24440: admission's MLP term covers the MoE dispatch's per-token working set — the
/// geometry's intermediate width is at least the routed experts' bytes per token row over the
/// `3 · element_bytes` the estimators charge per intermediate unit.
#[test]
fn the_admission_geometry_covers_the_moe_dispatch() {
    use core_llm::Quantize;
    let dir = q8_capable_qwen35_snapshot(true);
    let provider = super::LlamaProvider::load(&spec_at(dir.path(), Some(Quantize::Q8)))
        .expect("load the MoE snapshot Q8");
    let Decoder::Qwen35(m) = &provider.model else {
        panic!("the qwen3_5 MoE snapshot loads the hybrid decoder")
    };
    let moe = m.moe_step_bytes_per_token();
    assert!(moe > 0, "an MoE model prices its routed dispatch");
    let geometry = provider.model.memory_geometry();
    assert!(geometry.intermediate_size * 3 * geometry.element_bytes >= moe);
    assert!(geometry.intermediate_size >= m.config().intermediate_size.max(0) as u64);
}

/// Last-position logits of a loaded provider (either decoder) over a fixed prompt, on the
/// host.
fn provider_logits(provider: &super::LlamaProvider) -> Vec<f32> {
    use candle_core::{DType, Device};
    let prompt = vec![1u32, 5, 9, 2, 7];
    let host = |t: Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    match &provider.model {
        Decoder::Causal(m) => {
            let ids = Tensor::from_vec(prompt, (1, 5), m.device()).unwrap();
            host(m.decode_logits(&ids, &mut m.new_cache(), 0).unwrap())
        }
        Decoder::Qwen35(m) => {
            let ids = Tensor::from_vec(prompt, (1, 5), m.device()).unwrap();
            host(m.decode_logits(&ids, &mut m.new_cache(), 0).unwrap())
        }
    }
}

/// A tiny `Phi3ForCausalLM` snapshot: the llama parts with q‖k‖v fused into `qkv_proj` and
/// gate‖up into `gate_up_proj`, 64 wide (Q8_0; Q4_0 under Q4), bf16 on disk.
fn phi3_snapshot() -> tempfile::TempDir {
    let (mut cfg, mut w) = llama_parts(40, 64, 64, 2, 1, 2);
    cfg["architectures"] = serde_json::json!(["Phi3ForCausalLM"]);
    cfg["model_type"] = serde_json::json!("phi3");
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let mut take = |s: &str| w.remove(&p(s)).unwrap();
        let qkv = Tensor::cat(
            &[
                take("self_attn.q_proj.weight"),
                take("self_attn.k_proj.weight"),
                take("self_attn.v_proj.weight"),
            ],
            0,
        )
        .unwrap();
        let gate_up = Tensor::cat(
            &[take("mlp.gate_proj.weight"), take("mlp.up_proj.weight")],
            0,
        )
        .unwrap();
        w.insert(p("self_attn.qkv_proj.weight"), qkv);
        w.insert(p("mlp.gate_up_proj.weight"), gate_up);
    }
    write_snapshot(&cfg, &w, 40)
}

/// Prepare `src` as a `quantize` tier through the registered candle preparer.
fn prepare_tier(src: &std::path::Path, quantize: core_llm::Quantize) -> tempfile::TempDir {
    let out = tempfile::Builder::new()
        .prefix("candle-tier-")
        .tempdir()
        .unwrap();
    crate::prepare::prepare(&core_llm::PrepareSpec::quantized(
        src.to_path_buf(),
        out.path().to_path_buf(),
        quantize,
    ))
    .expect("prepare the tier");
    out
}

/// sc-19375, one decoder family: a `quantize` tier of `src` stores **every** tensor
/// `quantized(key)` names as GGML blocks (`U8`) — a fraction of the dense bytes — and writes
/// every other tensor exactly as read; it loads with no request (its persisted block drives
/// it) to the logits of a load-time quantization of the source, and admission prices its
/// device copy exactly as that load-time copy.
fn assert_family_tier(
    tag: &str,
    src: &std::path::Path,
    quantize: core_llm::Quantize,
    quantized: impl Fn(&str) -> bool,
) {
    let tier = prepare_tier(src, quantize);
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tier.path().join("config.json")).unwrap())
            .unwrap();
    assert_eq!(config["quantization"]["storage"], "ggml", "{tag}");
    let source = super::snapshot_tensor_headers(src).unwrap();
    let stored = super::snapshot_tensor_headers(tier.path()).unwrap();
    assert_eq!(
        source.keys().collect::<std::collections::BTreeSet<_>>(),
        stored.keys().collect(),
        "{tag}: same tensor set"
    );
    let elems = |h: &super::TensorHeader| h.shape.iter().product::<usize>() as u64;
    let (mut dense_bytes, mut block_bytes, mut blocks) = (0u64, 0u64, 0);
    for (key, header) in &source {
        if quantized(key) {
            assert_eq!(
                stored[key].dtype, "U8",
                "{tag}: {key} must be stored as blocks"
            );
            assert_eq!(
                stored[key].shape[..header.shape.len() - 1],
                header.shape[..header.shape.len() - 1],
                "{tag}: {key} keeps its rows"
            );
            dense_bytes += 2 * elems(header);
            block_bytes += elems(&stored[key]);
            blocks += 1;
        } else {
            assert_eq!(&stored[key], header, "{tag}: {key} is written as read");
        }
    }
    assert!(blocks > 0, "{tag}");
    let bound = match quantize {
        core_llm::Quantize::Q4 => 0.3,
        _ => 0.55,
    };
    assert!(
        (block_bytes as f64) < bound * dense_bytes as f64,
        "{tag}: {block_bytes} B of blocks for {dense_bytes} dense bf16 B"
    );

    let prepared = super::LlamaProvider::load(&spec_at(tier.path(), None)).unwrap();
    assert!(prepared.is_quantized(), "{tag}: loads quantized");
    let at_load = super::LlamaProvider::load(&spec_at(src, Some(quantize))).unwrap();
    assert_eq!(
        provider_logits(&prepared),
        provider_logits(&at_load),
        "{tag}: stored blocks load to the load-time-quantized model exactly"
    );
    let copy = |dir: &std::path::Path, quantize| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir, quantize), true)
            .unwrap()
            .quantized_copy_bytes
    };
    assert_eq!(
        copy(tier.path(), None),
        copy(src, Some(quantize)),
        "{tag}: stored blocks priced as the load-time copy"
    );
}

/// sc-19375: Phi-3's fused `qkv_proj` / `gate_up_proj` are stored as GGML blocks and split
/// by rows at load; a Phi-3 tier prepared before then (dense-rounded weights under a bare
/// block) still loads quantized.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_phi3_tier_stores_its_fused_projections_as_blocks() {
    use core_llm::Quantize;
    let src = phi3_snapshot();
    for quantize in [Quantize::Q8, Quantize::Q4] {
        assert_family_tier(&format!("phi3 {quantize:?}"), src.path(), quantize, |key| {
            key.ends_with("_proj.weight")
        });
    }

    let legacy = tempfile::Builder::new()
        .prefix("candle-phi3-legacy-")
        .tempdir()
        .unwrap();
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.path().join("config.json")).unwrap())
            .unwrap();
    config["quantization"] = serde_json::json!({ "bits": 8 });
    std::fs::write(legacy.path().join("config.json"), config.to_string()).unwrap();
    for file in ["tokenizer.json", "model.safetensors"] {
        std::fs::copy(src.path().join(file), legacy.path().join(file)).unwrap();
    }
    let loaded = super::LlamaProvider::load(&spec_at(legacy.path(), None)).unwrap();
    assert!(
        loaded.is_quantized(),
        "a dense-rounded Phi-3 tier loads quantized"
    );
    let at_load = super::LlamaProvider::load(&spec_at(src.path(), Some(Quantize::Q8))).unwrap();
    assert_eq!(provider_logits(&loaded), provider_logits(&at_load));

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_phi3_tier_stores_its_fused_projections_as_blocks");
}

/// sc-19375: a qwen3_5 tier stores every projection its loader quantizes — the Gated
/// DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj`, attention, MLP, the MTP head and the
/// MoE experts (stacked, or per expert — sc-24438) — as GGML blocks, and its loader reads them
/// from the stored-block `quantization` block; the per-head `in_proj_a` / `in_proj_b`, the
/// router, the embeddings and the head stay as read. A qwen3_5 tier prepared before then
/// (dense-rounded weights under a bare block) still loads dense, exactly as it did. The MoE
/// snapshots with an MTP head (its predictor layer a sparse-MoE block, sc-24438) prepare and
/// price that head too, in both expert layouts.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_qwen35_tier_stores_its_projections_and_experts_as_blocks() {
    use core_llm::Quantize;
    for (moe, mtp, fused) in [
        (false, true, true),
        (true, false, true),
        (true, true, true),
        (true, true, false),
    ] {
        let src = qwen35_snapshot(moe, mtp, fused);
        let headers = super::snapshot_tensor_headers(src.path()).unwrap();
        let expected = |key: &str| crate::models::qwen35::quantizes_tensor(key);
        // The tricky keys really are in the fixture, on the right side of the rule.
        let has = |suffix: &str| headers.keys().any(|k| k.ends_with(suffix) && expected(k));
        assert!(has("linear_attn.in_proj_qkv.weight") && has("linear_attn.in_proj_z.weight"));
        assert_eq!(has("mlp.experts.gate_up_proj"), moe && fused);
        assert_eq!(has("mlp.experts.0.gate_proj.weight"), moe && !fused);
        assert_eq!(
            has("mtp.layers.0.mlp.shared_expert.up_proj.weight"),
            moe && mtp
        );
        assert_eq!(has("mtp.fc.weight"), mtp);
        let moe = format!("{moe} mtp={mtp} fused={fused}");
        assert!(headers
            .keys()
            .any(|k| k.ends_with("in_proj_a.weight") && !expected(k)));
        for quantize in [Quantize::Q8, Quantize::Q4] {
            assert_family_tier(
                &format!("qwen3_5 moe={moe} {quantize:?}"),
                src.path(),
                quantize,
                expected,
            );
        }

        let legacy = tempfile::Builder::new()
            .prefix("candle-qwen35-legacy-")
            .tempdir()
            .unwrap();
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(src.path().join("config.json")).unwrap())
                .unwrap();
        config["quantization"] = serde_json::json!({ "bits": 8 });
        std::fs::write(legacy.path().join("config.json"), config.to_string()).unwrap();
        for file in ["tokenizer.json", "model.safetensors"] {
            std::fs::copy(src.path().join(file), legacy.path().join(file)).unwrap();
        }
        let loaded = super::LlamaProvider::load(&spec_at(legacy.path(), None)).unwrap();
        assert!(
            !loaded.is_quantized(),
            "moe={moe}: a bare block loads dense"
        );
        let dense = super::LlamaProvider::load(&spec_at(src.path(), None)).unwrap();
        assert_eq!(
            provider_logits(&loaded),
            provider_logits(&dense),
            "moe={moe}"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_qwen35_tier_stores_its_projections_and_experts_as_blocks");
}

/// A tiny `qwen3` GGUF (2 layers, hidden 32, vocab 40): Q8_0 matrices, f32 norms, and a
/// sibling tokenizer. Returns every tensor's element count and the layer projections'.
fn write_tiny_gguf(path: &std::path::Path) -> (u64, Vec<u64>) {
    write_tiny_gguf_arch(path, "qwen3")
}

/// [`write_tiny_gguf`] with `general.architecture = arch` (`"qwen3"` or `"llama"`; a llama
/// GGUF carries no per-head q/k norms).
fn write_tiny_gguf_arch(path: &std::path::Path, arch: &str) -> (u64, Vec<u64>) {
    use crate::primitives::{SplitMix64, TokenRng};
    use candle_core::quantized::gguf_file::{self, Value as Meta};
    use candle_core::quantized::{GgmlDType, QTensor};
    let (vocab, hidden, inter, kv, layers) = (40usize, 32usize, 64usize, 16usize, 2usize);
    let mut rng = SplitMix64::new(0x0006_6F0F);
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let (mut total, mut projections) = (0u64, Vec::new());
    let mut add = |name: String, dims: &[usize], projection: bool| {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
        let t = Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap();
        let dtype = if dims.len() == 2 {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        total += n as u64;
        if projection {
            projections.push(n as u64);
        }
        tensors.push((name, QTensor::quantize(&t, dtype).unwrap()));
    };
    add("token_embd.weight".into(), &[vocab, hidden], false);
    add("output_norm.weight".into(), &[hidden], false);
    add("output.weight".into(), &[vocab, hidden], false);
    for i in 0..layers {
        let b = |s: &str| format!("blk.{i}.{s}");
        add(b("attn_norm.weight"), &[hidden], false);
        add(b("ffn_norm.weight"), &[hidden], false);
        if arch == "qwen3" {
            add(b("attn_q_norm.weight"), &[8], false);
            add(b("attn_k_norm.weight"), &[8], false);
        }
        add(b("attn_q.weight"), &[hidden, hidden], true);
        add(b("attn_k.weight"), &[kv, hidden], true);
        add(b("attn_v.weight"), &[kv, hidden], true);
        add(b("attn_output.weight"), &[hidden, hidden], true);
        add(b("ffn_gate.weight"), &[inter, hidden], true);
        add(b("ffn_up.weight"), &[inter, hidden], true);
        add(b("ffn_down.weight"), &[hidden, inter], true);
    }
    let key = |s: &str| format!("{arch}.{s}");
    let metadata = [
        (
            "general.architecture".to_string(),
            Meta::String(arch.into()),
        ),
        (key("attention.head_count"), Meta::U32(4)),
        (key("attention.head_count_kv"), Meta::U32(2)),
        (key("attention.key_length"), Meta::U32(8)),
        (key("embedding_length"), Meta::U32(hidden as u32)),
        (key("block_count"), Meta::U32(layers as u32)),
        (key("feed_forward_length"), Meta::U32(inter as u32)),
        (key("context_length"), Meta::U32(256)),
        (key("attention.layer_norm_rms_epsilon"), Meta::F32(1e-6)),
        (key("rope.freq_base"), Meta::F32(1e6)),
    ];
    let metadata: Vec<(&str, &Meta)> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    gguf_file::write(
        &mut std::fs::File::create(path).unwrap(),
        &metadata,
        &tensor_refs,
    )
    .unwrap();
    std::fs::write(
        path.parent().unwrap().join("tokenizer.json"),
        synthetic_tokenizer_json(vocab),
    )
    .unwrap();
    (total, projections)
}

/// sc-20683 review: a GGUF's HF identity is reconstructed from `general.architecture`, which
/// labels Mistral and other llama-shaped fine-tunes `"llama"` too, so no GGUF load names a
/// compressed-KV table family — an opted-in request is `UnqualifiedModel`.
#[test]
fn a_gguf_load_names_no_compressed_kv_family() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
    for arch in ["llama", "qwen3"] {
        let dir = tempfile::Builder::new()
            .prefix("candle-gguf-kv-family-")
            .tempdir()
            .unwrap();
        let path = dir.path().join(format!("tiny-{arch}.gguf"));
        write_tiny_gguf_arch(&path, arch);
        let provider = super::LlamaProvider::load(&spec_at(&path, None)).unwrap();
        assert_eq!(provider.kv_model_family(), None, "{arch}");
        assert_eq!(
            provider
                .kv_cache_plan(Policy::Qualified, 40_000, 64, 1, false)
                .fallback,
            Some(Reason::UnqualifiedModel),
            "{arch}"
        );
    }
}

/// sc-24140 review: a llama-family GGUF load dequantizes every tensor into a dense f32 map on
/// its device and keeps it while it builds the decoder beside it — a bf16 cast of every
/// tensor it keeps dense (on CUDA) and, under Q8, the Q8_0 copy of the layer projections. The
/// estimate prices that working set (the f32 map as the source, with the 25 percent headroom
/// over it) instead of the file plus 25 percent, which is several times below it; on a host
/// device the f32 map is host memory. The map and the copy are checked against the loader's.
#[test]
fn gguf_admission_prices_the_dequantized_map_and_the_copies_beside_it() {
    use candle_core::quantized::GgmlDType;
    use core_llm::Quantize;
    let dir = tempfile::Builder::new()
        .prefix("candle-gguf-admission-")
        .tempdir()
        .unwrap();
    let path = dir.path().join("tiny-Q8_0.gguf");
    let (elems, projections) = write_tiny_gguf(&path);
    let projection_elems: u64 = projections.iter().sum();
    let estimate = |quantize, cuda| {
        super::LlamaProvider::load_memory_estimate(&spec_at(&path, quantize), cuda).unwrap()
    };

    let dense = estimate(None, true);
    let payload = std::fs::metadata(&path).unwrap().len();
    assert_eq!(dense.payload_bytes, payload);
    assert_eq!(
        dense.source_bytes,
        4 * elems,
        "every tensor dequantized to f32"
    );
    assert_eq!(
        dense.cast_copy_bytes,
        2 * elems,
        "every tensor cast to bf16 beside it"
    );
    assert_eq!(dense.quantized_copy_bytes, 0);
    assert_eq!(dense.device_required_bytes, Some(5 * elems + 2 * elems));
    assert!(
        dense.device_required_bytes.unwrap() > 4 * (payload + payload / 4),
        "the file plus 25 percent, the pricing at R, is several times below the working set"
    );

    let q8 = estimate(Some(Quantize::Q8), true);
    assert_eq!(q8.source_bytes, 4 * elems);
    assert_eq!(
        q8.cast_copy_bytes,
        2 * (elems - projection_elems),
        "the embedding, head and norms stay dense"
    );
    let copy: u64 = projections.iter().map(|&n| n / 32 * 34 + 544).sum();
    assert_eq!(q8.quantized_copy_bytes, copy);
    assert_eq!(
        q8.device_required_bytes,
        Some(5 * elems + q8.cast_copy_bytes + copy)
    );
    let host = estimate(Some(Quantize::Q8), false);
    assert_eq!(host.device_required_bytes, None);
    assert_eq!(
        host.host_required_bytes,
        5 * elems + copy,
        "on a host device the f32 map and the copy are host memory"
    );

    // The loader holds exactly that f32 map, and quantizes exactly the priced projections.
    let ck = crate::gguf::GgufCheckpoint::open(&path, &candle_core::Device::Cpu).unwrap();
    let map: u64 = ck
        .weights
        .keys()
        .map(|k| {
            let t = ck.weights.get(k).unwrap();
            (t.elem_count() * t.dtype().size_in_bytes()) as u64
        })
        .sum();
    assert_eq!(map, dense.source_bytes);
    let census = loaded_census(&spec_at(&path, Some(Quantize::Q8)));
    assert_eq!(census.projections.ggml.count, projections.len() as u64);
    assert_eq!(copy, ggml_copy_built(&census, GgmlDType::Q8_0));
}

/// E6 (sc-24140 review): an NVFP4 load is priced at exactly the packed copy it builds — its
/// nibbles and swizzled block scales, per eligible projection — checked against the census.
#[cfg(feature = "cuda")]
#[test]
fn nvfp4_load_estimate_prices_exactly_the_copy_the_loader_builds() {
    use core_llm::Quantize;
    let Ok(device) = crate::device::new_cuda_for_test() else {
        candle_quant_kernels::skip_without_sm120("no CUDA device");
        return;
    };
    if crate::primitives::projection::ProjectionFormat::nvfp4(&device).is_err() {
        candle_quant_kernels::skip_without_sm120("CUDA device below the NVFP4 floor");
        return;
    }
    let (cfg, weights) = tiny_llama_parts();
    let dir = write_snapshot(&cfg, &weights, 40);
    let spec = spec_at(dir.path(), Some(Quantize::Nvfp4));
    let estimate = super::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
    let census = loaded_census(&spec);
    assert_eq!(
        census.projections.nvfp4.count, 21,
        "the 40-row head stays dense"
    );
    assert_eq!(
        estimate.quantized_copy_bytes,
        census.projections.nvfp4.resident_bytes
    );
}

/// E6: admission prices exactly the recurrent state a Qwen3.5-family request's cache holds.
/// The provider's reference/MTP cache (`Decode::make_cache`) keeps no checkpoint ring — one
/// live state per linear layer, `1 + REFERENCE_MAX_CHECKPOINTS` states; the engine's step
/// cache for `K` drafts keeps a per-token checkpoint ring of `K + 2` states per linear layer,
/// and the geometry for that request charges exactly the ring's bytes — the old start-of-step
/// checkpoint term is gone (sc-24131).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen35_admission_prices_every_recurrent_state_the_cache_holds() {
    use crate::decode::{StepModel, StepRequest};
    use crate::models::qwen35::{REFERENCE_MAX_CHECKPOINTS, STEP_MAX_CHECKPOINTS};
    use crate::models::Qwen35Cache;
    use crate::primitives::nn::input_ids;

    let (_cfg, model) = crate::models::qwen35::tests::text_model();
    let one_state = model.recurrent_state_bytes(1) as u64;
    assert!(one_state > 0);
    assert_eq!(model.recurrent_state_bytes(4), 4 * one_state as usize);
    let decoder = Decoder::Qwen35(model);
    let geometry = decoder.memory_geometry();
    assert_eq!(
        geometry.recurrent_bytes,
        one_state * (1 + REFERENCE_MAX_CHECKPOINTS as u64),
        "the geometry charges the live state plus the provider cache's checkpoints"
    );
    // The admission estimate carries the whole recurrent term (charged once with MTP off).
    let without = core_llm::LlmMemoryGeometry {
        recurrent_bytes: 0,
        ..geometry
    };
    let est = |g| core_llm::estimate_chunked_request_bytes(8, 4, g, 0, 0, 64).unwrap();
    assert_eq!(est(geometry) - est(without), geometry.recurrent_bytes);

    // What the provider's cache really holds at steady state (prefill + several steps).
    let device = candle_core::Device::Cpu;
    let mut cache = decoder.make_cache();
    decoder
        .step(&input_ids(&[1, 7, 3], &device).unwrap(), cache.as_mut(), 0)
        .unwrap();
    for (i, t) in [42, 9, 2, 11].into_iter().enumerate() {
        decoder
            .step(
                &input_ids(&[t], &device).unwrap(),
                cache.as_mut(),
                3 + i as i32,
            )
            .unwrap();
    }
    let held = cache.as_any_mut().downcast_mut::<Qwen35Cache>().unwrap();
    assert_eq!(held.max_checkpoints(), REFERENCE_MAX_CHECKPOINTS);
    assert_eq!(
        held.recurrent_bytes() as u64,
        geometry.recurrent_bytes,
        "provider cache holds {} recurrent bytes, admission charges {}",
        held.recurrent_bytes(),
        geometry.recurrent_bytes
    );

    // The engine's step cache for `K` drafts: the ring is the whole recurrent term, priced
    // exactly, from creation (preallocated) through decoding (written in place).
    let Decoder::Qwen35(model) = &decoder else {
        unreachable!()
    };
    for k in 0..=5usize {
        let step_geometry = decoder.step_memory_geometry(k);
        let mut step = model.new_cache_for(16, k).unwrap();
        assert_eq!(
            step.max_checkpoints(),
            k + 1,
            "K={k}: the step start + K + 1 positions"
        );
        assert_eq!(
            step.recurrent_bytes() as u64,
            one_state * (k as u64 + 2),
            "K={k}: a ring of K + 2 states per linear layer"
        );
        assert_eq!(
            step_geometry.recurrent_bytes,
            step.recurrent_bytes() as u64,
            "K={k}: admission charges exactly the ring"
        );
        model
            .forward_step(&mut step, StepRequest::last(&[1, 7, 3]))
            .unwrap();
        for t in [42, 9, 2, 11] {
            model
                .forward_step(&mut step, StepRequest::last(&[t]))
                .unwrap();
        }
        assert_eq!(
            step_geometry.recurrent_bytes,
            step.recurrent_bytes() as u64,
            "K={k}: nothing grew while decoding"
        );
    }

    // The unbounded step cache keeps STEP_MAX_CHECKPOINTS positions behind the current one.
    let mut step = StepModel::new_cache(model);
    model
        .forward_step(&mut step, StepRequest::last(&[1, 7, 3]))
        .unwrap();
    for t in [42, 9, 2, 11] {
        model
            .forward_step(&mut step, StepRequest::last(&[t]))
            .unwrap();
    }
    assert_eq!(step.checkpoint_offsets().len(), STEP_MAX_CHECKPOINTS);
    assert_eq!(
        step.recurrent_bytes() as u64,
        one_state * (1 + STEP_MAX_CHECKPOINTS as u64)
    );
    assert_eq!(
        decoder
            .memory_geometry_with_checkpoints(STEP_MAX_CHECKPOINTS)
            .recurrent_bytes,
        step.recurrent_bytes() as u64
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::qwen35_admission_prices_every_recurrent_state_the_cache_holds");
}

/// E6 (sc-24134, sc-24140): with the CUDA-graph runner switched on, every request the runner
/// wraps is priced for every graph it may capture — exactly [`graph_workspace_admission_bytes`]
/// over the step geometry, the request's reach (`prompt + budget + K`) and the `K + 1` step
/// token counts: an MTP request (`K`) and a request whose speculation is off (the engine with
/// no proposer, `K = 0`) alike. The reference loop, which the runner never wraps, is priced as
/// before.
#[test]
fn admission_includes_the_cuda_graph_workspace_when_the_runner_is_on() {
    use super::DecodeRoute;

    let (_cfg, model) = crate::models::qwen35::tests::text_model();
    let decoder = Decoder::Qwen35(model);
    let (prompt, budget, k) = (11usize, 21u32, 3u32);
    let price = |route, graphs| {
        super::priced_request_bytes(&decoder, route, prompt, budget, 0, graphs).unwrap()
    };
    let graph_term = |drafts: u32| {
        crate::decode::graph_workspace_admission_bytes(
            &decoder.step_memory_geometry(drafts as usize),
            (prompt as u64) + u64::from(budget) + u64::from(drafts),
            drafts + 1,
        )
        .unwrap()
    };
    let mtp = DecodeRoute::Engine {
        proposer: ProposerKind::Mtp,
        drafts: k,
    };
    let off = DecodeRoute::PLAIN_ENGINE;
    assert!(graph_term(k) > graph_term(0) && graph_term(0) > 0);
    // sc-24446: an MTP route also prices the plain step a demoted request captures (its
    // verify steps ask for hidden rows, so that shape is outside the priced set) — one
    // token-row over the same reach; a route without hidden-row steps does not.
    let demoted = crate::decode::graph_demoted_step_admission_bytes(
        &decoder.step_memory_geometry(k as usize),
        (prompt as u64) + u64::from(budget) + u64::from(k),
    )
    .unwrap();
    assert!(demoted > 0);
    assert_eq!(
        price(mtp, true) - price(mtp, false),
        graph_term(k) + demoted
    );
    let lookup = DecodeRoute::Engine {
        proposer: ProposerKind::PromptLookup,
        drafts: k,
    };
    assert_eq!(price(lookup, true) - price(lookup, false), graph_term(k));
    assert_eq!(price(off, true) - price(off, false), graph_term(0));
    assert_eq!(
        price(DecodeRoute::Reference, true),
        price(DecodeRoute::Reference, false)
    );
}

#[test]
fn mtp_admission_prices_the_step_cache_checkpoints_and_the_verify_overshoot() {
    use super::DecodeRoute;

    let (_cfg, model) = crate::models::qwen35::tests::text_model();
    let one_state = model.recurrent_state_bytes(1) as u64;
    let decoder = Decoder::Qwen35(model);
    let (prompt, budget) = (11usize, 21u32);
    let k = 3u32;
    let price =
        |route| super::priced_request_bytes(&decoder, route, prompt, budget, 0, false).unwrap();
    let on_bytes = price(DecodeRoute::Engine {
        proposer: ProposerKind::Mtp,
        drafts: k,
    });
    let off_bytes = price(DecodeRoute::PLAIN_ENGINE);
    let reference_bytes = price(DecodeRoute::Reference);

    // The same request priced on the reference geometry with the width set: what admission
    // would charge if it forgot the engine's cache. The step geometry's ring term — `K + 2`
    // states per linear layer against the reference cache's one (sc-24131) — is the only
    // difference, charged once: the engine rolls back by slot selection, never by a clone.
    let on_reference_geometry = core_llm::estimate_chunked_request_bytes_with_recurrent_copies(
        prompt,
        budget,
        decoder.memory_geometry(),
        0,
        k,
        EAGER_ATTN_QUERY_CHUNK_SIZE,
        1,
    )
    .unwrap();
    let ring_term = u64::from(k + 1) * one_state;
    assert!(ring_term > 0);
    assert_eq!(
        on_bytes - on_reference_geometry,
        ring_term,
        "an Enabled request is priced on the step cache's per-token checkpoint ring, once"
    );

    // A request whose speculation is off runs the engine with no proposer (sc-24140): its
    // cache's `K = 0` ring holds two states per linear layer against the reference cache's
    // one — the only difference from the reference loop's price.
    assert_eq!(
        decoder.step_memory_geometry(0).recurrent_bytes,
        2 * one_state
    );
    assert_eq!(off_bytes - reference_bytes, one_state);

    // Against the Off request: `K` more ring states plus at least the K positions the verify
    // step writes past the budget (each `layers * kv_heads * head_dim * 2 * element` bytes).
    let geometry = decoder.step_memory_geometry(k as usize);
    assert_eq!(geometry.recurrent_bytes, u64::from(k + 2) * one_state);
    let kv_per_position =
        geometry.layers * geometry.kv_heads * geometry.head_dim * geometry.element_bytes * 2;
    assert!(
        on_bytes >= off_bytes + u64::from(k) * one_state + u64::from(k) * kv_per_position,
        "Enabled {on_bytes} vs Off {off_bytes}: ring {k} x {one_state}, overshoot {k} x {kv_per_position}"
    );
}

/// sc-24140 / sc-24433: one decision picks the loop — and so the price. A plan with a
/// proposer (MTP, prompt lookup) always runs the engine with it; a request whose speculation
/// is off runs the engine with none — text and multimodal alike (sc-24446: the decision no
/// longer reads the request's modality) — unless the reference loop is selected. A proposer
/// that cannot run is a named fallback, never silent.
#[test]
fn a_request_decodes_on_the_engine_unless_the_reference_loop_is_selected() {
    use super::DecodeRoute;
    use core_llm::{SpeculativePlan, SpeculativeProposer};

    let (_dir, mut provider) = synthetic_qwen35_provider_without_mtp();
    assert_eq!(provider.decode_path(), DecodePath::StepModel, "the default");
    let off = SpeculativePlan::Off;
    let mtp = SpeculativePlan::Run {
        proposer: SpeculativeProposer::Mtp,
        depth: 3,
    };
    let lookup = SpeculativePlan::Run {
        proposer: SpeculativeProposer::PromptLookup,
        depth: 4,
    };
    let engine = |proposer, drafts| (DecodeRoute::Engine { proposer, drafts }, None);
    assert_eq!(
        provider.decode_route(off),
        (DecodeRoute::PLAIN_ENGINE, None)
    );
    assert_eq!(provider.decode_route(mtp), engine(ProposerKind::Mtp, 3));
    assert_eq!(
        provider.decode_route(lookup),
        engine(ProposerKind::PromptLookup, 4)
    );
    let (route, why) = provider.decode_route(SpeculativePlan::Run {
        proposer: SpeculativeProposer::DraftModel,
        depth: 2,
    });
    assert_eq!(route, DecodeRoute::PLAIN_ENGINE);
    assert!(why.unwrap().contains("`draft_model` has no draft model"));
    provider
        .set_decode_path(DecodePath::Reference)
        .expect("the reference loop is selectable");
    assert_eq!(provider.decode_route(off), (DecodeRoute::Reference, None));
    assert_eq!(
        provider.decode_route(mtp),
        engine(ProposerKind::Mtp, 3),
        "an MTP plan runs the engine whatever the selector says"
    );
    assert_eq!(
        provider.decode_route(lookup),
        engine(ProposerKind::PromptLookup, 4),
        "so does a prompt-lookup plan"
    );
    assert!(provider.set_decode_path(DecodePath::PromptLookup).is_err());
    assert_eq!(provider.decode_path(), DecodePath::Reference);
}

/// sc-24436: with a resident draft, a `draft_model` plan runs the engine with the draft as
/// its proposer — the reference selector does not change that — and admission prices the
/// draft's own prefill and cache on top of the target's (E7).
#[test]
fn a_resident_draft_routes_draft_model_onto_the_engine_and_is_priced() {
    use super::DecodeRoute;
    use core_llm::{SpeculativePlan, SpeculativeProposer, TextLlm};

    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    let mut provider = super::LlamaProvider::load(&fixture.spec_with_draft()).unwrap();
    assert!(provider.load_report().unwrap().draft.unwrap().is_resident());
    let plan = SpeculativePlan::Run {
        proposer: SpeculativeProposer::DraftModel,
        depth: 3,
    };
    let route = DecodeRoute::Engine {
        proposer: ProposerKind::DraftModel,
        drafts: 3,
    };
    assert_eq!(provider.decode_route(plan), (route, None));
    provider
        .set_decode_path(DecodePath::Reference)
        .expect("the reference loop is selectable");
    assert_eq!(provider.decode_route(plan), (route, None));

    // E7: the draft's request is priced beside the target's, only on its own route.
    let (prompt, budget) = (40usize, 24u32);
    let draft = &provider.draft.as_ref().unwrap().model;
    let expected = super::draft_request_bytes(draft, prompt, budget, 3).unwrap();
    assert!(expected > 0);
    assert_eq!(provider.draft_bytes(route, prompt, budget), Some(expected));
    let geometry = draft.memory_geometry_with_checkpoints(5);
    let reach = (prompt as u64 + u64::from(budget) + 4)
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * geometry.element_bytes
        * 2;
    assert!(
        expected >= reach,
        "the draft cache spans the request's reach plus K + 1"
    );
    assert_eq!(
        provider.draft_bytes(DecodeRoute::PLAIN_ENGINE, prompt, budget),
        Some(0)
    );
    let lookup = DecodeRoute::Engine {
        proposer: ProposerKind::PromptLookup,
        drafts: 3,
    };
    assert_eq!(provider.draft_bytes(lookup, prompt, budget), Some(0));

    // Both admission prices carry the draft on its route: the pre-preprocessing one every
    // request takes, and the Gemma 4 spliced-prompt one over the expanded prompt.
    assert_eq!(
        provider.request_bytes(route, prompt, budget, 0),
        Some(provider.request_bytes(lookup, prompt, budget, 0).unwrap() + expected)
    );
    assert_eq!(
        provider.spliced_request_bytes(route, prompt, budget),
        Some(
            provider
                .spliced_request_bytes(lookup, prompt, budget)
                .unwrap()
                + expected
        )
    );
}

/// sc-24436 E7, end to end: under a memory budget that holds the target's own request but
/// not the draft's beside it, a `draft_model` request is refused as resource-exhausted while
/// `off` and prompt lookup at the same depth are admitted. The budget override is process
/// state, so the check runs in a child process of its own.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_draft_model_request_is_refused_without_room_for_the_draft() {
    use super::DecodeRoute;
    use core_llm::{Sampling, Speculative, SpeculativePlan, SpeculativeProposer, TextLlm};

    const CHILD: &str = "SCENEWORKS_DRAFT_ADMISSION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let name = std::thread::current().name().unwrap().to_owned();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--include-ignored"])
            .env(CHILD, "1")
            .env_remove(core_llm::AVAILABLE_MEMORY_OVERRIDE)
            .envs(
                std::iter::once((
                    "MODEL_INFERENCE_CASE_FQN",
                    "provider::tests::a_draft_model_request_is_refused_without_room_for_the_draft#child=0",
                ))
                .filter(|_| cfg!(feature = "model-inference-proof")),
            )
            .status()
            .unwrap();
        assert!(status.success());
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_PARENT_COMPLETED provider::tests::a_draft_model_request_is_refused_without_room_for_the_draft");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    // The cross-turn prefix cache is off here (sc-24437): its held entries are reclaimable
    // (E7), so with it on the draft request would evict the earlier requests' entries and fit
    // — this test pins the draft's own admission boundary, not the cache's.
    let mut spec = fixture.spec_with_draft();
    spec.prefix_cache_bytes = Some(0);
    let provider = super::LlamaProvider::load(&spec).unwrap();
    let prompts = core_llm_testkit::draft_model_prompts();
    let budget = 8u32;
    let request = |speculative| {
        core_llm_testkit::bench_request(&prompts[0], speculative, &Sampling::greedy(), budget)
    };
    let off = request(Speculative::Off);
    let lookup = request(Speculative::proposer(SpeculativeProposer::PromptLookup, 3));
    let draft = request(Speculative::proposer(SpeculativeProposer::DraftModel, 3));
    let prompt = provider
        .generate(&off, &mut |_| {})
        .unwrap()
        .usage
        .prompt_tokens as usize;
    let price = |plan| {
        let (route, _) = provider.decode_route(plan);
        provider.request_bytes(route, prompt, budget, 0).unwrap()
    };
    let run = |proposer| SpeculativePlan::Run { proposer, depth: 3 };
    // The target's request at depth 3 plus the draft's own beside it, priced independently
    // of the route's admission so the refusal below is the admission's, not this sum's.
    let lookup_price = price(run(SpeculativeProposer::PromptLookup));
    let draft_price =
        super::draft_request_bytes(&provider.draft.as_ref().unwrap().model, prompt, budget, 3)
            .unwrap();
    let with_draft = lookup_price + draft_price;
    let target_only = price(SpeculativePlan::Off).max(lookup_price);
    assert!(draft_price > 0 && target_only < with_draft);
    assert!(matches!(
        provider
            .decode_route(run(SpeculativeProposer::DraftModel))
            .0,
        DecodeRoute::Engine {
            proposer: ProposerKind::DraftModel,
            ..
        }
    ));
    std::env::set_var(
        core_llm::AVAILABLE_MEMORY_OVERRIDE,
        (with_draft - 1).to_string(),
    );
    for admitted in [&off, &lookup] {
        provider
            .generate(admitted, &mut |_| {})
            .expect("the target's own request fits the budget");
    }
    let refused = provider.generate(&draft, &mut |_| panic!("a refused request emitted a token"));
    assert!(
        matches!(refused, Err(core_llm::Error::RequestResourceExhausted(_))),
        "{refused:?}"
    );
    std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
    let admitted = provider.generate(&draft, &mut |_| {}).unwrap();
    assert_eq!(
        admitted.decode.unwrap().proposer,
        ProposerKind::DraftModel,
        "with room it runs the draft"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_draft_model_request_is_refused_without_room_for_the_draft#child=0");
}

/// sc-24446: a load whose steps can ask for hidden rows — a Qwen3.5 snapshot with a native
/// MTP head, or any load naming a companion head — prices the hidden-row captures on top of
/// the plain ones (the runner keys a shape on `want_hidden`); one without a head does not.
#[test]
fn a_load_with_an_mtp_head_prices_its_hidden_row_captures() {
    let estimate = |dir: &tempfile::TempDir, companion: bool| {
        let spec = core_llm::LoadSpec {
            cuda_graphs: Some(true),
            mtp_head_source: companion.then(|| "companion-head".to_string()),
            ..core_llm::LoadSpec::dense(dir.path().display().to_string())
        };
        super::LlamaProvider::load_memory_estimate(&spec, true)
            .unwrap()
            .graph_param_cache_bytes
    };
    let (plain, headed) = (
        qwen35_snapshot(false, false, false),
        qwen35_snapshot(false, true, false),
    );
    assert_eq!(estimate(&plain, false), 32 << 20);
    assert_eq!(estimate(&headed, false), 64 << 20);
    assert_eq!(estimate(&plain, true), 64 << 20);
}

/// sc-24441 (E7): a CUDA load the graph runner wraps prices the parameter cache its
/// captures leave resident for the process — 32 MiB, once per load — and nothing with the
/// runner off or off CUDA; a resident draft is never wrapped, so its admission beside the
/// target carries none.
#[test]
fn a_cuda_load_under_the_graph_runner_prices_the_parameter_cache_once() {
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    let with = |graphs: bool| core_llm::LoadSpec {
        cuda_graphs: Some(graphs),
        ..fixture.spec_with_draft()
    };
    let estimate = |graphs, cuda| super::LlamaProvider::load_memory_estimate(&with(graphs), cuda);
    let (on, off) = (
        estimate(true, true).unwrap(),
        estimate(false, true).unwrap(),
    );
    assert_eq!(on.graph_param_cache_bytes, 32 << 20);
    assert_eq!(off.graph_param_cache_bytes, 0);
    assert_eq!(
        on.device_required_bytes,
        Some(off.device_required_bytes.unwrap() + (32 << 20))
    );
    let host = estimate(true, false).unwrap();
    assert_eq!(
        (host.graph_param_cache_bytes, host.device_required_bytes),
        (0, None)
    );
    // Unset (sc-24446), a CUDA estimate takes the CUDA row's graph default (on: priced) from
    // the `cuda` it is asked about, not from the device this process would open — on a
    // CPU-only host as on a CUDA one. (A host estimate carries none whatever the switch: the
    // `host` case above.)
    {
        let _switch = crate::decode::graph::cuda_graphs_policy_guard(None);
        // Neither switch set, and a build that can capture (no candle-flash-attn).
        if std::env::var_os(crate::decode::graph::CUDA_GRAPHS_ENV).is_none()
            && std::env::var_os(crate::device::CUDA_STREAM_ENV).is_none()
            && !cfg!(feature = "flash-attn")
        {
            let unset = |cuda| {
                super::LlamaProvider::load_memory_estimate(
                    &core_llm::LoadSpec {
                        cuda_graphs: None,
                        ..fixture.spec_with_draft()
                    },
                    cuda,
                )
                .unwrap()
                .graph_param_cache_bytes
            };
            let cuda_default = if core_llm::defaults::CANDLE_CUDA.cuda_graphs {
                on.graph_param_cache_bytes
            } else {
                0
            };
            assert_eq!(unset(true), cuda_default);
        }
    }
    // The draft beside a graph-wrapped target: exactly room for its own unwrapped load.
    let draft = super::LlamaProvider::load_memory_estimate(
        &core_llm::LoadSpec {
            cuda_graphs: Some(false),
            ..core_llm::LoadSpec::dense(fixture.draft.to_string_lossy())
        },
        true,
    )
    .unwrap()
    .device_required_bytes
    .unwrap();
    let target = on.device_required_bytes.unwrap();
    let (_, refusal) = super::LlamaProvider::admit_draft(
        &with(true),
        true,
        (on.host_required_bytes, u64::MAX),
        Some((target, target + draft)),
    )
    .unwrap()
    .unwrap();
    assert_eq!(refusal, None, "the draft carries no parameter cache");
}

/// sc-24436 E7: a named draft is admitted at load only beside the target, in the host domain
/// and (on CUDA) the device domain; without room it is refused by name and the target still
/// loads; a target that does not fit is refused exactly as before.
#[test]
fn a_draft_is_admitted_at_load_only_beside_the_target() {
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    // The graph runner pinned off: a CUDA estimate otherwise follows the process switch.
    let unwrapped = |spec: core_llm::LoadSpec| core_llm::LoadSpec {
        cuda_graphs: Some(false),
        ..spec
    };
    let spec = unwrapped(fixture.spec_with_draft());
    let target = super::LlamaProvider::load_memory_estimate(&spec, false)
        .unwrap()
        .host_required_bytes;
    let draft = super::LlamaProvider::load_memory_estimate(
        &unwrapped(core_llm::LoadSpec::dense(fixture.draft.to_string_lossy())),
        false,
    )
    .unwrap()
    .host_required_bytes;
    assert!(
        draft > 0 && draft < target,
        "the draft is the smaller model"
    );
    let admit = |host, device| super::LlamaProvider::admit_draft(&spec, false, host, device);

    let (draft_spec, refusal) = admit((target, target + draft), None).unwrap().unwrap();
    assert_eq!(refusal, None);
    assert_eq!(draft_spec.source, fixture.draft.to_string_lossy());
    assert_eq!(draft_spec.draft_source, None);
    let (_, refusal) = admit((target, target + draft - 1), None).unwrap().unwrap();
    assert!(refusal.unwrap().starts_with("draft model:"));
    assert!(
        admit((target, target - 1), None).is_err(),
        "the target alone"
    );
    // Off CUDA the draft carries no device bytes, so a full device domain refuses nothing.
    let (_, refusal) = admit((target, u64::MAX), Some((target, target)))
        .unwrap()
        .unwrap();
    assert_eq!(
        refusal, None,
        "an off-CUDA draft estimate carries no device bytes"
    );
    // On CUDA the device domain refuses the draft on its own, with the host unbounded.
    let device_bytes = |spec: &core_llm::LoadSpec| {
        super::LlamaProvider::load_memory_estimate(spec, true)
            .unwrap()
            .device_required_bytes
            .unwrap()
    };
    let target_device = device_bytes(&spec);
    let draft_device = device_bytes(&unwrapped(core_llm::LoadSpec::dense(
        fixture.draft.to_string_lossy(),
    )));
    assert!(draft_device > 0);
    let cuda = |device| super::LlamaProvider::admit_draft(&spec, true, (target, u64::MAX), device);
    let (_, refusal) = cuda(Some((target_device, target_device + draft_device)))
        .unwrap()
        .unwrap();
    assert_eq!(refusal, None, "room for both on the device");
    let (_, refusal) = cuda(Some((target_device, target_device + draft_device - 1)))
        .unwrap()
        .unwrap();
    let refusal = refusal.expect("no device room for the draft");
    assert!(refusal.starts_with("draft model:"), "{refusal}");
    // No draft named: nothing to admit.
    assert!(super::LlamaProvider::admit_draft(
        &core_llm::LoadSpec::dense(fixture.target.to_string_lossy()),
        false,
        (target, target),
        None
    )
    .unwrap()
    .is_none());
}

/// sc-24436 E2/E7 through the real load (sc-24432 feature-end review): a budget that fits the
/// target but not target + draft refuses the draft by name in BOTH the load report's `draft`
/// and its `fallbacks`, and the target loads alone.
#[test]
fn a_draft_refused_by_load_admission_is_named_in_the_load_fallbacks() {
    use core_llm::TextLlm;
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    let spec = fixture.spec_with_draft();
    let draft_source = fixture.draft.to_string_lossy().into_owned();
    let target = admitted_bytes(&core_llm::LoadSpec::dense(fixture.target.to_string_lossy()));
    // A draft is never graph-wrapped: priced as a draft, not as a target.
    let draft = admitted_bytes_as(&super::draft_load_spec(&spec, &draft_source), false);
    let provider =
        super::with_load_budget(target + draft - 1, || super::LlamaProvider::load(&spec))
            .expect("the target fits and loads alone");
    let report = provider.load_report().unwrap();
    let refused = report.draft.clone().expect("the named draft is reported");
    let refusal = refused.refusal.clone().expect("refused by load admission");
    assert!(refusal.starts_with("draft model:"), "{refusal}");
    assert_eq!(refused.source, draft_source);
    assert!(
        report.fallbacks.contains(&refusal),
        "the refused draft is named in the load fallbacks: {:?}",
        report.fallbacks
    );
    assert!(provider
        .descriptor()
        .capabilities
        .proposer(core_llm::SpeculativeProposer::DraftModel)
        .is_none());
}

/// E2 (sc-24432 feature-end review): a qwen3_5 config declaring a native MTP head the
/// snapshot does not carry — or a variant this runtime does not run (two predictor layers,
/// dedicated MTP embeddings) — loads the target plain with a named `mtp:` load fallback and
/// MTP unadvertised, and decodes; a PARTIAL `mtp.*` set is still a load error (integrity).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_configured_mtp_head_the_snapshot_cannot_run_is_a_named_load_fallback() {
    use core_llm::{Message, Sampling, TextLlm, TextLlmRequest};
    let variant = |edit: &dyn Fn(&mut serde_json::Map<String, serde_json::Value>),
                   keep: &dyn Fn(&str) -> bool| {
        let mut json = crate::models::qwen35::tests::synthetic_cfg_json();
        let text = json["text_config"].as_object_mut().unwrap();
        text.insert("linear_value_head_dim".into(), serde_json::json!(8));
        text.insert("mtp_num_hidden_layers".into(), serde_json::json!(1));
        let cfg = crate::models::Qwen35Config::from_json(&json).unwrap();
        let weights = crate::models::qwen35::tests::synthetic_snapshot_weights(&cfg);
        assert!(weights.keys().any(|k| k.starts_with("mtp.")), "a full head");
        let tensors: HashMap<String, Tensor> = weights
            .keys()
            .filter(|k| keep(k))
            .map(|k| (k.to_string(), weights.get(k).unwrap().clone()))
            .collect();
        edit(json["text_config"].as_object_mut().unwrap());
        let mut config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
        });
        config["text_config"] = json["text_config"].clone();
        write_snapshot(&config, &tensors, cfg.vocab_size as usize)
    };
    let request = TextLlmRequest {
        messages: vec![Message::user("t3 t7 t11 t2")],
        sampling: Sampling::greedy(),
        max_new_tokens: 3,
        seed: Some(0),
        ..Default::default()
    };
    let no_head: &dyn Fn(&str) -> bool = &|k| !k.starts_with("mtp.");
    let all: &dyn Fn(&str) -> bool = &|_| true;
    type Edit = fn(&mut serde_json::Map<String, serde_json::Value>);
    type Keep<'a> = &'a dyn Fn(&str) -> bool;
    let cases: [(Edit, Keep, &str); 3] = [
        (|_| {}, no_head, "stores no `mtp.*` tensor"),
        (
            |t| {
                t.insert("mtp_num_hidden_layers".into(), serde_json::json!(2));
            },
            all,
            "declares 2 predictor layers",
        ),
        (
            |t| {
                t.insert(
                    "mtp_use_dedicated_embeddings".into(),
                    serde_json::json!(true),
                );
            },
            all,
            "dedicated MTP embeddings",
        ),
    ];
    for (edit, keep, named) in cases {
        let dir = variant(&edit, keep);
        let provider = super::LlamaProvider::load(&spec_at(dir.path(), None))
            .unwrap_or_else(|e| panic!("{named}: the target still loads (E2): {e}"));
        let fallbacks = provider.load_report().unwrap().fallbacks;
        assert_eq!(fallbacks.len(), 1, "{named}: {fallbacks:?}");
        assert!(
            fallbacks[0].starts_with("mtp: ") && fallbacks[0].contains(named),
            "{named}: {fallbacks:?}"
        );
        assert!(mtp_advertised(&provider).is_none(), "{named}");
        assert!(provider.mtp.is_none(), "{named}");
        let out = provider.generate(&request, &mut |_| {}).unwrap();
        assert_eq!(out.usage.generated_tokens, 3, "{named}");
    }
    // A partial head is a corrupt snapshot, not an absent accelerator: the load fails, naming
    // the missing tensor.
    let partial = variant(&|_| {}, &|k| k != "mtp.fc.weight");
    let error = super::LlamaProvider::load(&spec_at(partial.path(), None))
        .err()
        .expect("a partial `mtp.*` set fails the load");
    assert!(error.to_string().contains("mtp.fc.weight"), "{error}");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_configured_mtp_head_the_snapshot_cannot_run_is_a_named_load_fallback");
}

/// A tokenizer.json whose vocab is `t0..t{vocab-1}` (whitespace WordLevel), so every id of
/// the synthetic decoder decodes to a distinct piece.
fn synthetic_tokenizer_json(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
    format!(
        r#"{{
                "version": "1.0",
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }},
                "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }}
            }}"#,
        entries.join(", ")
    )
}

/// The synthetic Qwen3.5 decoder written as a weights-free snapshot with no `mtp.*` tensors
/// and `mtp_num_hidden_layers = 0`.
fn synthetic_qwen35_snapshot_without_mtp() -> tempfile::TempDir {
    let (cfg, weights, cfg_json) = crate::models::qwen35::tests::text_model_snapshot_parts();
    let dir = tempfile::Builder::new()
        .prefix("candle-qwen35-no-mtp-")
        .tempdir()
        .unwrap();
    let mut config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
    });
    config["text_config"] = cfg_json["text_config"].clone();
    assert_eq!(config["text_config"]["mtp_num_hidden_layers"], 0);
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        synthetic_tokenizer_json(cfg.vocab_size as usize),
    )
    .unwrap();
    let tensors: std::collections::HashMap<String, candle_core::Tensor> = weights
        .keys()
        .map(|k| (k.to_string(), weights.get(k).unwrap().clone()))
        .collect();
    assert!(tensors.keys().all(|k| !k.starts_with("mtp.")));
    candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
    dir
}

/// [`synthetic_qwen35_snapshot_without_mtp`] loaded as a provider with the backend's default
/// graph switch (keep the directory alive with it).
fn synthetic_qwen35_provider_without_mtp() -> (tempfile::TempDir, super::LlamaProvider) {
    use core_llm::LoadSpec;

    let dir = synthetic_qwen35_snapshot_without_mtp();
    let provider = super::LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string()))
        .expect("load the synthetic Qwen3.5 snapshot");
    (dir, provider)
}

/// sc-24139: the load's CUDA-graph policy is settled at load and recorded, every generation
/// runs under it (not under whatever the process switch says later), and the contract carries
/// the decode report and the load report a product renders — the same record the evidence
/// harness reads, never a guess.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_load_settles_the_graph_policy_and_the_contract_carries_the_reports() {
    use core_llm::{LoadSpec, Message, ProposerKind, Sampling, TextLlm, TextLlmRequest};

    let _process = crate::decode::graph::cuda_graphs_policy_guard(Some(false));
    let dir = synthetic_qwen35_snapshot_without_mtp();
    let source = dir.path().display().to_string();
    let request = TextLlmRequest {
        messages: vec![Message::user("t3 t7 t11 t2 t7 t11")],
        sampling: Sampling::greedy(),
        max_new_tokens: 4,
        seed: Some(0),
        ..Default::default()
    };

    // `Some(true)` wins over a process switch that is off, and the load records it.
    let on = super::LlamaProvider::load(&LoadSpec {
        cuda_graphs: Some(true),
        ..LoadSpec::dense(source.clone())
    })
    .expect("load with graphs requested");
    assert_eq!(on.load_record().cuda_graphs, Some(true));
    // The load report carries the settled switch, not a guess from the request.
    assert_eq!(on.load_report().unwrap().cuda_graphs, Some(true));
    // Flipping the process switch after the load does not change the loaded model's policy.
    crate::decode::set_cuda_graphs(Some(false));
    let out = on.generate(&request, &mut |_| {}).expect("generate");
    let report = out
        .decode
        .clone()
        .expect("the provider reports its decode path");
    assert!(report.cuda_graphs.enabled, "the load's Some(true) governs");
    assert_eq!(report, on.last_decode_record().unwrap().report(true));
    assert_eq!(report.proposer, ProposerKind::None);
    // `Auto` on a snapshot without a head: the engine with no proposer, since sc-24140.
    assert_eq!(report.path, "step_model");
    // The prefill forward plus one per generated token after the first: the prefix cache's
    // snapshot of the hybrid's recurrent state at the end of the rendered conversation
    // (sc-24437) is captured inside that one prefill forward, not by splitting it
    // (sc-24446).
    assert_eq!(report.prefix_cache.path, "miss");
    assert_eq!(report.prefill_forwards, 1);
    assert_eq!(
        report.target_forwards,
        u64::from(out.usage.generated_tokens)
    );

    // `None` keeps the process switch at load (off here); the report says so.
    let default = super::LlamaProvider::load(&LoadSpec::dense(source)).expect("load");
    assert_eq!(default.load_record().cuda_graphs, Some(false));
    crate::decode::set_cuda_graphs(Some(true));
    let out = default.generate(&request, &mut |_| {}).expect("generate");
    assert!(!out.decode.expect("reported").cuda_graphs.enabled);

    // The load report names the requested format and the resident projection kinds.
    let load = default
        .load_report()
        .expect("the provider reports its load");
    assert_eq!(load.requested, None);
    // `None` requested, the process switch (off) at load settled it: the report says `false`.
    assert_eq!(load.cuda_graphs, Some(false));
    assert_eq!(load.projections.len(), 1, "{:?}", load.projections);
    assert_eq!(load.projections[0].kind, "dense");
    assert!(load.projections[0].count > 0 && load.projections[0].resident_bytes > 0);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::the_load_settles_the_graph_policy_and_the_contract_carries_the_reports");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn auto_on_a_qwen35_snapshot_without_a_head_runs_prompt_lookup_and_explicit_mtp_runs_plain() {
    // AC3 (sc-24130), revised by sc-24433 AC2, weights-free: the synthetic Qwen3.5 decoder
    // written as a snapshot with no `mtp.*` tensors and `mtp_num_hidden_layers = 0`. The
    // provider advertises no MTP but does advertise prompt lookup, so `auto` (the new option
    // or the legacy `mtp: Auto` alias) runs the engine with prompt lookup at its recommended
    // depth, and the record names it; an explicit MTP (or `draft_model`) request the model
    // does not advertise decodes plainly with the reason named (E2: explicit fallback, never
    // failure — the sc-24432 feature-end review); `off` is the engine
    // with no proposer on the static KV cache (sc-24140), one verify step per token after the
    // first — and the same tokens.
    use core_llm::{Message, MtpMode, Sampling, TextLlm, TextLlmRequest};

    let (_dir, mut provider) = synthetic_qwen35_provider_without_mtp();
    assert!(provider.descriptor().capabilities.mtp.is_none());
    assert_eq!(
        provider
            .descriptor()
            .capabilities
            .proposer(core_llm::SpeculativeProposer::PromptLookup),
        Some(super::prompt_lookup_capabilities(
            crate::device::decode_backend(provider.model.device())
        ))
    );

    let request = |mtp| TextLlmRequest {
        messages: vec![Message::user("t3 t7 t11 t2 t7 t11")],
        sampling: Sampling::greedy(),
        max_new_tokens: 6,
        seed: Some(0),
        mtp: Some(mtp),
        ..Default::default()
    };
    let out = provider
        .generate(&request(MtpMode::Auto), &mut |_| {})
        .expect("Auto decodes without a head");
    assert_eq!(out.usage.generated_tokens, 6);
    assert!(out.mtp.is_none(), "no MTP ran");
    let report = out.decode.clone().unwrap();
    assert_eq!(report.proposer, ProposerKind::PromptLookup);
    assert_eq!(report.path, "prompt_lookup");
    assert_eq!(
        report.draft_tokens,
        Some(table_depths(provider.model.device()).prompt_lookup)
    );
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    let record = provider.last_decode_record().unwrap();
    assert_eq!(record.proposer, ProposerKind::PromptLookup);
    assert_eq!(record.path, DecodePath::PromptLookup, "the engine ran");
    assert_eq!(record.kv_cache, crate::primitives::KvCacheKind::Static);

    let off = provider
        .generate(&request(MtpMode::Off), &mut |_| {})
        .unwrap();
    assert_eq!(off.text, out.text, "prompt lookup changes the speed only");
    let record = provider.last_decode_record().unwrap();
    assert_eq!(record.path, DecodePath::StepModel);
    assert_eq!(record.proposer, ProposerKind::None);
    assert_eq!(record.proposed_tokens, 0);
    assert_eq!(
        (
            record.target_forwards,
            record.prefill_forwards,
            record.verify_steps
        ),
        (6, 1, 5),
        "the prefill plus one single-token verify step per token after the first"
    );
    assert_eq!(record.replay_forwards, 0);
    assert!(off.decode.unwrap().fallbacks.is_empty());
    for (speculative, proposer) in [
        (
            core_llm::Speculative::from(MtpMode::Enabled { draft_tokens: 2 }),
            "mtp",
        ),
        (
            core_llm::Speculative::proposer(core_llm::SpeculativeProposer::DraftModel, 2),
            "draft_model",
        ),
    ] {
        let plain = provider
            .generate(
                &TextLlmRequest {
                    speculative: Some(speculative),
                    ..request(MtpMode::Off)
                },
                &mut |_| {},
            )
            .expect("an unadvertised proposer decodes plainly");
        assert_eq!(plain.text, off.text, "{proposer}: the plain path's tokens");
        let report = plain.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::None, "{proposer}");
        assert_eq!(
            report.fallbacks,
            vec![format!(
                "speculative: `{proposer}` is not available for this model (this model \
                 does not advertise it; decoded without a proposer)"
            )]
        );
    }

    // The reference loop stays selectable as the oracle for a request whose speculation is
    // off; it names itself, and on CPU f32 (where the two caches are bit-identical) it is the
    // same decode.
    provider
        .set_decode_path(DecodePath::Reference)
        .expect("the reference loop is selectable");
    let reference = provider
        .generate(&request(MtpMode::Off), &mut |_| {})
        .unwrap();
    let record = provider.last_decode_record().unwrap();
    assert_eq!(record.path, DecodePath::Reference);
    assert_eq!(record.kv_cache, crate::primitives::KvCacheKind::Growing);
    assert_eq!(record.proposer, ProposerKind::None);
    if provider.model.device().is_cpu() {
        assert_eq!(reference.text, out.text, "the same tokens on both loops");
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::auto_on_a_qwen35_snapshot_without_a_head_runs_prompt_lookup_and_explicit_mtp_runs_plain");
}

/// A tiny Qwen3 (`qwen3`: the llama-family `CausalLm` with per-head q/k RMSNorm) written as a
/// weights-free snapshot (sc-24433): the Causal fixture of the speculative parity suite.
fn synthetic_qwen3_snapshot() -> tempfile::TempDir {
    let (vocab, heads, layers) = (40usize, 4usize, 3usize);
    let (mut cfg, mut weights) = llama_parts(vocab, 32, 64, heads, 2, layers);
    let head_dim = 32 / heads;
    cfg["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
    cfg["model_type"] = serde_json::json!("qwen3");
    cfg["head_dim"] = serde_json::json!(head_dim);
    for i in 0..layers {
        for norm in ["q_norm", "k_norm"] {
            weights.insert(
                format!("model.layers.{i}.self_attn.{norm}.weight"),
                Tensor::ones(head_dim, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            );
        }
    }
    write_snapshot(&cfg, &weights, vocab)
}

/// [`synthetic_qwen3_snapshot`] loaded as a provider (keep the directory alive with it).
fn synthetic_causal_provider() -> (tempfile::TempDir, super::LlamaProvider) {
    let dir = synthetic_qwen3_snapshot();
    let provider =
        super::LlamaProvider::load(&core_llm::LoadSpec::dense(dir.path().display().to_string()))
            .expect("load the synthetic Qwen3 snapshot");
    assert!(matches!(provider.model, Decoder::Causal(_)), "a CausalLm");
    (dir, provider)
}

/// The Qwen-VL vision specials of the synthetic VLM fixtures: the last three ids of the
/// vocabulary, in `(vision_start, image_pad, vision_end)` order.
fn vl_special_ids(vocab: usize) -> (usize, usize, usize) {
    (vocab - 3, vocab - 2, vocab - 1)
}

/// [`synthetic_tokenizer_json`] with the Qwen-VL specials (`<|vision_start|>`,
/// `<|image_pad|>`, `<|vision_end|>`) in place of the vocabulary's last three pieces, as added
/// tokens so the placeholder string the provider renders tokenizes to exactly those ids.
fn synthetic_vl_tokenizer_json(vocab: usize) -> String {
    let (start, pad, end) = vl_special_ids(vocab);
    let specials = [
        (start, "<|vision_start|>"),
        (pad, "<|image_pad|>"),
        (end, "<|vision_end|>"),
    ];
    let piece = |i: usize| {
        specials
            .iter()
            .find(|(id, _)| *id == i)
            .map_or_else(|| format!("t{i}"), |(_, s)| s.to_string())
    };
    let entries: Vec<String> = (0..vocab)
        .map(|i| format!("\"{}\": {i}", piece(i)))
        .collect();
    let added: Vec<String> = specials
        .iter()
        .map(|(id, content)| {
            format!(
                r#"{{ "id": {id}, "content": "{content}", "single_word": false,
                         "lstrip": false, "rstrip": false, "normalized": false,
                         "special": true }}"#
            )
        })
        .collect();
    format!(
        r#"{{
                "version": "1.0",
                "added_tokens": [{}],
                "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }},
                "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }}
            }}"#,
        added.join(", "),
        entries.join(", ")
    )
}

/// A seeded one-block Qwen3-VL ViT tower (`model.visual.*`, no DeepStack taps) of the image
/// processor's own geometry (patch 16, temporal patch 2, merge 2), whose merger emits
/// `out_hidden` rows, and its `vision_config`: a 256 × 256 image is 64 language tokens.
fn synthetic_vision_tower(out_hidden: usize) -> (serde_json::Value, HashMap<String, Tensor>) {
    use crate::primitives::{SplitMix64, TokenRng};
    let (hid, inter, heads, positions) = (16usize, 32usize, 2usize, 16usize);
    let merge_dim = hid * 4;
    let mut rng = SplitMix64::new(0x0056_1510);
    let mut rand = |dims: &[usize]| {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
        Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap()
    };
    let mut w = HashMap::new();
    let mut put = |key: &str, dims: &[usize]| {
        w.insert(format!("model.visual.{key}"), rand(dims));
    };
    put("patch_embed.proj.weight", &[hid, 3, 2, 16, 16]);
    put("patch_embed.proj.bias", &[hid]);
    put("pos_embed.weight", &[positions, hid]);
    for (key, dims) in [
        ("norm1.weight", vec![hid]),
        ("norm1.bias", vec![hid]),
        ("norm2.weight", vec![hid]),
        ("norm2.bias", vec![hid]),
        ("attn.qkv.weight", vec![3 * hid, hid]),
        ("attn.qkv.bias", vec![3 * hid]),
        ("attn.proj.weight", vec![hid, hid]),
        ("attn.proj.bias", vec![hid]),
        ("mlp.linear_fc1.weight", vec![inter, hid]),
        ("mlp.linear_fc1.bias", vec![inter]),
        ("mlp.linear_fc2.weight", vec![hid, inter]),
        ("mlp.linear_fc2.bias", vec![hid]),
    ] {
        put(&format!("blocks.0.{key}"), &dims);
    }
    put("merger.norm.weight", &[hid]);
    put("merger.norm.bias", &[hid]);
    put("merger.linear_fc1.weight", &[merge_dim, merge_dim]);
    put("merger.linear_fc1.bias", &[merge_dim]);
    put("merger.linear_fc2.weight", &[out_hidden, merge_dim]);
    put("merger.linear_fc2.bias", &[out_hidden]);
    let config = serde_json::json!({
        "depth": 1, "hidden_size": hid, "num_heads": heads, "intermediate_size": inter,
        "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
        "out_hidden_size": out_hidden, "num_position_embeddings": positions
    });
    (config, w)
}

/// Write a synthetic VLM snapshot: `config`, the Qwen-VL tokenizer, and `weights` stored f32.
fn write_vl_snapshot(
    config: &serde_json::Value,
    weights: &HashMap<String, Tensor>,
    vocab: usize,
) -> tempfile::TempDir {
    let dir = tempfile::Builder::new()
        .prefix("candle-vl-fixture-")
        .tempdir()
        .unwrap();
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        synthetic_vl_tokenizer_json(vocab),
    )
    .unwrap();
    candle_core::safetensors::save(weights, dir.path().join("model.safetensors")).unwrap();
    dir
}

/// A tiny Qwen3-VL (`qwen3_vl`: the generic-causal `CausalLm` decoder under
/// `model.language_model.*`, interleaved M-RoPE) with a [`synthetic_vision_tower`] carrying
/// one DeepStack tap (after ViT block 0, fused after decoder layer 0), loaded as a provider
/// (keep the directory alive with it) — the Causal multimodal fixture (sc-24433, sc-24446).
fn synthetic_qwen3vl_provider() -> (tempfile::TempDir, super::LlamaProvider) {
    let (vocab, hidden, heads, layers) = (40usize, 32usize, 4usize, 3usize);
    let head_dim = hidden / heads;
    let (_, llama) = llama_parts(vocab, hidden, 64, heads, 2, layers);
    let mut weights: HashMap<String, Tensor> = llama
        .into_iter()
        .map(|(k, t)| match k.strip_prefix("model.") {
            Some(rest) => (format!("model.language_model.{rest}"), t),
            None => (k, t),
        })
        .collect();
    for i in 0..layers {
        for norm in ["q_norm", "k_norm"] {
            weights.insert(
                format!("model.language_model.layers.{i}.self_attn.{norm}.weight"),
                Tensor::ones(head_dim, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            );
        }
    }
    let (mut vision_config, vision) = synthetic_vision_tower(hidden);
    weights.extend(vision);
    // The DeepStack tap's merger (post-shuffle norm over the merged patch rows): seeded
    // rows of the tower's own merger geometry.
    let merge_dim = vision_config["hidden_size"].as_u64().unwrap() as usize * 4;
    vision_config["deepstack_visual_indexes"] = serde_json::json!([0]);
    let mut rng = crate::primitives::SplitMix64::new(0x0d33_5ac4);
    for (leaf, dims) in [
        ("norm.weight", vec![merge_dim]),
        ("norm.bias", vec![merge_dim]),
        ("linear_fc1.weight", vec![merge_dim, merge_dim]),
        ("linear_fc1.bias", vec![merge_dim]),
        ("linear_fc2.weight", vec![hidden, merge_dim]),
        ("linear_fc2.bias", vec![hidden]),
    ] {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| crate::primitives::TokenRng::next_f32(&mut rng) - 0.5)
            .collect();
        weights.insert(
            format!("model.visual.deepstack_merger_list.0.{leaf}"),
            Tensor::from_vec(data, dims, &candle_core::Device::Cpu).unwrap(),
        );
    }
    let config = serde_json::json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": vl_special_ids(vocab).1,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": hidden, "intermediate_size": 64, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": 2, "head_dim": head_dim,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 512,
            "rope_scaling": {
                "mrope_interleaved": true, "mrope_section": [2, 1, 1], "rope_type": "default"
            }
        },
        "vision_config": vision_config,
    });
    let dir = write_vl_snapshot(&config, &weights, vocab);
    let provider =
        super::LlamaProvider::load(&core_llm::LoadSpec::dense(dir.path().display().to_string()))
            .expect("load the synthetic Qwen3-VL snapshot");
    assert!(matches!(provider.model, Decoder::Causal(_)), "a CausalLm");
    assert!(provider.descriptor.capabilities.supports_vision);
    (dir, provider)
}

/// The synthetic Qwen3.5 hybrid ([`synthetic_qwen35_snapshot_without_mtp`]'s decoder) with a
/// [`synthetic_vision_tower`], loaded as a provider (keep the directory alive with it) — the
/// Qwen35 multimodal fixture (sc-24433).
fn synthetic_qwen35_vl_provider() -> (tempfile::TempDir, super::LlamaProvider) {
    let (cfg, weights, cfg_json) = crate::models::qwen35::tests::text_model_snapshot_parts();
    let vocab = cfg.vocab_size as usize;
    let mut tensors: HashMap<String, Tensor> = weights
        .keys()
        .map(|k| (k.to_string(), weights.get(k).unwrap().clone()))
        .collect();
    let (vision_config, vision) = synthetic_vision_tower(cfg.hidden_size as usize);
    tensors.extend(vision);
    let mut text_config = cfg_json["text_config"].clone();
    // Room for the 64 image tokens beside the prompt and the continuation.
    text_config["max_position_embeddings"] = serde_json::json!(512);
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "image_token_id": vl_special_ids(vocab).1,
        "text_config": text_config,
        "vision_config": vision_config,
    });
    let dir = write_vl_snapshot(&config, &tensors, vocab);
    let provider =
        super::LlamaProvider::load(&core_llm::LoadSpec::dense(dir.path().display().to_string()))
            .expect("load the synthetic Qwen3.5 VLM snapshot");
    assert!(matches!(provider.model, Decoder::Qwen35(_)), "the hybrid");
    assert!(provider.descriptor.capabilities.supports_vision);
    (dir, provider)
}

/// A 256 × 256 RGB gradient (the image processor's minimum area: 64 language tokens).
fn gradient_image() -> core_llm::ImageRef {
    let (w, h) = (256u32, 256u32);
    let pixels = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [x as u8, y as u8, ((x + y) / 2) as u8]
        })
        .collect();
    core_llm::ImageRef::new(w, h, pixels).unwrap()
}

/// The multimodal fixture prompts: the gradient image then a repetitive text turn, and the
/// text turn then the image (the placeholder run sits mid-prompt, not only at its head).
fn image_fixture_prompts() -> Vec<core_llm_testkit::BenchPrompt> {
    use core_llm::{Content, Message, Role};
    use core_llm_testkit::{BenchPrompt, PromptClass};
    let text = "t3 t7 t11 t2 t7 t11 t3 t7 t11 t2";
    let prompt = |id: &str, content: Vec<Content>| BenchPrompt {
        id: id.into(),
        class: PromptClass::Predictable,
        messages: vec![Message {
            role: Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        }],
    };
    vec![
        prompt(
            "image_then_text",
            vec![Content::Image(gradient_image()), Content::text(text)],
        ),
        prompt(
            "text_then_image",
            vec![Content::text(text), Content::Image(gradient_image())],
        ),
    ]
}

/// The streamed token events of one generation, and its output.
fn token_events(
    provider: &super::LlamaProvider,
    req: &core_llm::TextLlmRequest,
) -> (Vec<(u32, String)>, core_llm::TextLlmOutput) {
    use core_llm::{StreamEvent, TextLlm};
    let mut events = Vec::new();
    let out = provider
        .generate(req, &mut |event| {
            if let StreamEvent::Token { id, text, .. } = event {
                events.push((id, text));
            }
        })
        .expect("generate");
    (events, out)
}

/// The fixture prompts of the speculative parity suite: repetitive enough that prompt lookup
/// finds matches in the prompt and in the looping greedy continuation of a random decoder.
fn fixture_prompts() -> Vec<core_llm_testkit::BenchPrompt> {
    use core_llm_testkit::{BenchPrompt, PromptClass};
    vec![
        BenchPrompt::user(
            "repeat",
            PromptClass::Predictable,
            "t3 t7 t11 t2 t7 t11 t3 t7 t11 t2",
        ),
        BenchPrompt::user("sparse", PromptClass::OpenEnded, "t5 t9 t1 t30 t22"),
    ]
}

/// The parity case table for one provider: prompt lookup at depths 1, 3, the recommended 4
/// and the advertised max, `auto` (resolving to `auto_proposer`), plus `extra` rows.
fn parity_cases(
    auto_proposer: ProposerKind,
    extra: &[core_llm_testkit::ParityCase],
) -> Vec<core_llm_testkit::ParityCase> {
    use core_llm::{Speculative, SpeculativeProposer};
    use core_llm_testkit::ParityCase;
    let mut cases: Vec<ParityCase> = [1, 3, table_depths(&candle_core::Device::Cpu).prompt_lookup]
        .into_iter()
        .chain([super::PROMPT_LOOKUP_MAX_DEPTH])
        .map(|depth| ParityCase {
            speculative: Speculative::proposer(SpeculativeProposer::PromptLookup, depth),
            expect_proposer: ProposerKind::PromptLookup,
        })
        .collect();
    cases.push(ParityCase {
        speculative: Speculative::Auto,
        expect_proposer: auto_proposer,
    });
    cases.extend_from_slice(extra);
    cases
}

/// Every parity row ran the engine with the depth it asked for, and prompt lookup actually
/// drafted — and had drafts accepted — somewhere in the table (a suite where nothing was
/// proposed would pass vacuously).
fn assert_rows_drafted(rows: &[core_llm_testkit::ParityRow]) {
    for row in rows {
        assert!(
            row.report.fallbacks.is_empty(),
            "{:?}",
            row.report.fallbacks
        );
        assert_eq!(row.report.kv_cache, "static", "{}", row.prompt_id);
        if let core_llm::Speculative::Proposer { depth, .. } = row.speculative {
            assert_eq!(row.report.draft_tokens, Some(depth));
        }
        assert!(row.report.verify_steps > 0);
    }
    let lookup: Vec<_> = rows
        .iter()
        .filter(|r| r.report.proposer == ProposerKind::PromptLookup)
        .collect();
    assert!(lookup.iter().any(|r| r.report.proposed_tokens > 0));
    assert!(lookup.iter().any(|r| r.report.accepted_tokens > 0));
}

/// sc-24433 AC1 (Causal): on the Qwen3 fixture, `{proposer: prompt_lookup, depth: 4}` — and
/// every other depth up to the advertised max, and `auto` — runs the engine with prompt
/// lookup (the report says so) and emits the `off` greedy stream token for token.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn prompt_lookup_is_greedy_exact_on_the_qwen3_causal_fixture() {
    let (_dir, mut provider) = synthetic_causal_provider();
    let rows = core_llm_testkit::check_speculative_greedy_parity(
        &provider,
        &fixture_prompts(),
        &parity_cases(ProposerKind::PromptLookup, &[]),
        24,
    )
    .unwrap_or_else(|failures| panic!("{failures}"));
    assert_eq!(rows.len(), 2 * 5);
    assert!(rows.iter().all(|r| r.report.path == "prompt_lookup"));
    assert_rows_drafted(&rows);

    // The `off` the table was compared against is the engine with no proposer; it is in turn
    // the reference `Decode` loop's stream, token event for token event.
    let off = |prompt| {
        core_llm_testkit::bench_request(
            prompt,
            core_llm::Speculative::Off,
            &core_llm::Sampling::greedy(),
            24,
        )
    };
    let prompts = fixture_prompts();
    let engine: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            let (events, out) = token_events(&provider, &off(prompt));
            assert_eq!(out.decode.unwrap().path, "step_model");
            events
        })
        .collect();
    provider
        .set_decode_path(DecodePath::Reference)
        .expect("the reference loop is selectable");
    for (prompt, engine) in prompts.iter().zip(engine) {
        let (reference, out) = token_events(&provider, &off(prompt));
        assert_eq!(out.decode.unwrap().path, "reference");
        assert_eq!(reference.len(), 24, "{}", prompt.id);
        assert_eq!(reference, engine, "{}: engine off vs reference", prompt.id);
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prompt_lookup_is_greedy_exact_on_the_qwen3_causal_fixture");
}

/// sc-24433 AC1 (Qwen35) and AC2: on the qwen3_5 hybrid without a head the same table holds
/// with `auto` resolving to prompt lookup; with an MTP head `auto` resolves to MTP, and MTP
/// and prompt lookup are both greedy-exact against `off`.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn prompt_lookup_and_auto_are_greedy_exact_on_the_qwen35_fixtures() {
    use core_llm::{Speculative, SpeculativeProposer};
    use core_llm_testkit::ParityCase;

    let (_dir, provider) = synthetic_qwen35_provider_without_mtp();
    let rows = core_llm_testkit::check_speculative_greedy_parity(
        &provider,
        &fixture_prompts(),
        &parity_cases(ProposerKind::PromptLookup, &[]),
        24,
    )
    .unwrap_or_else(|failures| panic!("{failures}"));
    assert_eq!(rows.len(), 2 * 5);
    assert_rows_drafted(&rows);

    let dir = q8_capable_qwen35_snapshot(false);
    let with_head =
        super::LlamaProvider::load(&core_llm::LoadSpec::dense(dir.path().display().to_string()))
            .expect("load the synthetic Qwen3.5 snapshot with an MTP head");
    assert!(core_llm::TextLlm::descriptor(&with_head)
        .capabilities
        .mtp
        .is_some());
    let mtp = |depth| ParityCase {
        speculative: Speculative::proposer(SpeculativeProposer::Mtp, depth),
        expect_proposer: ProposerKind::Mtp,
    };
    let rows = core_llm_testkit::check_speculative_greedy_parity(
        &with_head,
        &fixture_prompts(),
        &parity_cases(ProposerKind::Mtp, &[mtp(1), mtp(3)]),
        16,
    )
    .unwrap_or_else(|failures| panic!("{failures}"));
    assert_eq!(rows.len(), 2 * 7);
    let auto: Vec<_> = rows
        .iter()
        .filter(|r| r.speculative == Speculative::Auto)
        .collect();
    // `auto` runs the head at the depth the snapshot advertises as recommended, whatever the
    // backend default currently is (sc-24446).
    let recommended = core_llm::TextLlm::descriptor(&with_head)
        .capabilities
        .proposer(SpeculativeProposer::Mtp)
        .expect("the head is advertised")
        .recommended_depth;
    assert!(auto
        .iter()
        .all(|r| r.report.proposer == ProposerKind::Mtp
            && r.report.draft_tokens == Some(recommended)));

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prompt_lookup_and_auto_are_greedy_exact_on_the_qwen35_fixtures");
}

/// sc-24438 AC1: the advertised max depth is finite and backend-true — the widest verify the
/// NVFP4 decode GEMV serves (8 rows = 1 + 7 drafts), on the legacy MTP field and the
/// per-proposer list alike — and a request above it runs at it, with the clamp named in
/// `DecodeReport::fallbacks`, emitting exactly `off`'s stream: prompt lookup on both
/// families, and MTP (the new option and the legacy field) on the Qwen3.8 head.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_too_deep_request_runs_at_the_advertised_max_and_names_the_clamp() {
    use core_llm::{MtpMode, Sampling, Speculative, SpeculativeProposer, TextLlm};
    assert_eq!(super::SPECULATIVE_MAX_DEPTH, 7);
    let prompt = &fixture_prompts()[0];
    let request = |spec| core_llm_testkit::bench_request(prompt, spec, &Sampling::greedy(), 16);
    let clamp = |proposer: &str, asked: u32, max: u32| {
        vec![format!(
            "speculative: `{proposer}` depth {asked} clamped to {max} (advertised 1..={max})"
        )]
    };
    let check = |provider: &super::LlamaProvider,
                 req: core_llm::TextLlmRequest,
                 kind: ProposerKind,
                 max: u32,
                 label: &str| {
        let (off, _) = token_events(provider, &request(Speculative::Off));
        provider.validate(&req).expect(label);
        let (events, out) = token_events(provider, &req);
        let report = out.decode.unwrap();
        assert_eq!(report.proposer, kind, "{label}");
        assert_eq!(report.draft_tokens, Some(max), "{label}");
        assert_eq!(events, off, "{label}: the clamped run is greedy-exact");
        report.fallbacks
    };

    let (_causal_dir, causal) = synthetic_causal_provider();
    let dir = q8_capable_qwen35_snapshot(false);
    let with_head =
        super::LlamaProvider::load(&core_llm::LoadSpec::dense(dir.path().display().to_string()))
            .expect("load the synthetic Qwen3.5 snapshot with an MTP head");
    let asked = super::PROMPT_LOOKUP_MAX_DEPTH + 5;
    for (label, provider) in [("causal", &causal), ("qwen35", &with_head)] {
        let fallbacks = check(
            provider,
            request(Speculative::proposer(
                SpeculativeProposer::PromptLookup,
                asked,
            )),
            ProposerKind::PromptLookup,
            super::PROMPT_LOOKUP_MAX_DEPTH,
            label,
        );
        assert_eq!(
            fallbacks,
            clamp("prompt_lookup", asked, super::PROMPT_LOOKUP_MAX_DEPTH),
            "{label}"
        );
    }

    let caps = &with_head.descriptor().capabilities;
    let mtp = caps.proposer(SpeculativeProposer::Mtp).unwrap();
    assert_eq!(
        (mtp.max_depth, mtp.recommended_depth),
        (
            super::MTP_MAX_DEPTH,
            table_depths(with_head.model.device()).mtp
        )
    );
    assert_eq!(
        caps.mtp,
        Some(core_llm::MtpCapabilities {
            max_draft_tokens: mtp.max_depth,
            recommended_draft_tokens: mtp.recommended_depth,
        }),
        "the legacy field agrees with the proposer list"
    );
    let fallbacks = check(
        &with_head,
        request(Speculative::proposer(SpeculativeProposer::Mtp, 40)),
        ProposerKind::Mtp,
        super::MTP_MAX_DEPTH,
        "mtp",
    );
    assert_eq!(fallbacks, clamp("mtp", 40, super::MTP_MAX_DEPTH));
    let mut legacy = request(Speculative::Off);
    legacy.speculative = None;
    legacy.mtp = Some(MtpMode::Enabled { draft_tokens: 40 });
    let fallbacks = check(
        &with_head,
        legacy,
        ProposerKind::Mtp,
        super::MTP_MAX_DEPTH,
        "legacy",
    );
    assert_eq!(fallbacks, clamp("mtp", 40, super::MTP_MAX_DEPTH));

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_too_deep_request_runs_at_the_advertised_max_and_names_the_clamp");
}

/// sc-24446 (E5): `auto` reaches the engine with its acceptance monitor and an explicit
/// proposer does not. On the synthetic Qwen3.5 snapshot `auto` resolves to its MTP head, whose
/// random weights never draft a token the target accepts (asserted as the premise): `auto` is
/// demoted at the end of its first window and the report says where, while the same proposer
/// asked for explicitly runs to the end. Both stream exactly `off`'s tokens.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn auto_is_monitored_and_an_explicit_proposer_is_not() {
    use core_llm::{Sampling, Speculative, SpeculativeProposer};
    let dir = q8_capable_qwen35_snapshot(false);
    let provider =
        super::LlamaProvider::load(&core_llm::LoadSpec::dense(dir.path().display().to_string()))
            .expect("load the synthetic Qwen3.5 snapshot with an MTP head");
    let prompt = &fixture_prompts()[0];
    let request = |spec| core_llm_testkit::bench_request(prompt, spec, &Sampling::greedy(), 48);
    let (off, _) = token_events(&provider, &request(Speculative::Off));
    // Untimed (the static threshold decides, no plain probe): the timed monitor's decisions
    // are pinned on a deterministic clock in the engine's tests.
    let (auto_events, auto) = core_llm::with_decode_clock(None, || {
        token_events(&provider, &request(Speculative::Auto))
    });
    let auto = auto.decode.unwrap();
    let depth = auto.draft_tokens.expect("auto ran a proposer");
    let explicit = Speculative::proposer(SpeculativeProposer::Mtp, depth);
    let (asked_events, asked) = token_events(&provider, &request(explicit));
    let asked = asked.decode.unwrap();
    assert_eq!(auto_events, off, "auto is greedy-exact");
    assert_eq!(asked_events, off, "the explicit request is greedy-exact");
    assert_eq!(
        (auto.proposer, asked.proposer),
        (ProposerKind::Mtp, ProposerKind::Mtp)
    );
    assert_eq!(asked.speculative_demoted_at, None, "{asked:?}");
    // The fixture head never pays at this depth (accepted 0 over the whole explicit run), so
    // `auto` is demoted at the end of its first window and stops proposing.
    assert_eq!(asked.accepted_tokens, 0, "fixture premise: {asked:?}");
    assert_eq!(
        auto.speculative_demoted_at,
        Some(1 + u64::from(core_llm::ACCEPTANCE_PROBE_VERIFIES)),
        "{auto:?}"
    );
    assert!(auto.proposed_tokens < asked.proposed_tokens);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::auto_is_monitored_and_an_explicit_proposer_is_not");
}

/// sc-24438 AC2: a qwen3_5 MoE snapshot carrying an MTP head — its predictor layer a
/// sparse-MoE block, as the 35B-A3B ships it, in the fused (Qwen3.6) and the per-expert
/// (Qwen3.5) expert layout — loads (before, `Qwen35Mtp` refused it and the whole load failed),
/// advertises MTP on both fields, and **runs** it: `{mtp}` at depths 1, 3 and the advertised
/// max, the legacy `enabled 3` and `auto` each report the MTP proposer with drafts proposed,
/// and emit exactly `off`'s stream.
/// Under Q8 the head is priced: the estimate equals the copy the load builds, head included,
/// and exceeds the headless MoE snapshot's.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_moe_snapshot_with_an_mtp_head_runs_mtp() {
    use core_llm::{MtpMode, Quantize, Sampling, Speculative, SpeculativeProposer, TextLlm};
    let estimate = |dir: &tempfile::TempDir| {
        super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), Some(Quantize::Q8)), true)
            .unwrap()
            .quantized_copy_bytes
    };
    let headless = q8_capable_qwen35_snapshot(true);
    for fused in [true, false] {
        let label = if fused { "fused" } else { "per-expert" };
        let dir = qwen35_snapshot(true, true, fused);
        let provider = super::LlamaProvider::load(&core_llm::LoadSpec::dense(
            dir.path().display().to_string(),
        ))
        .expect("a MoE checkpoint with an MTP head loads");
        assert!(provider.mtp.is_some(), "{label}");
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

        let prompt = &fixture_prompts()[0];
        let request = |spec| core_llm_testkit::bench_request(prompt, spec, &Sampling::greedy(), 12);
        let (off, _) = token_events(&provider, &request(Speculative::Off));
        let mut legacy = request(Speculative::Off);
        legacy.speculative = None;
        legacy.mtp = Some(MtpMode::Enabled { draft_tokens: 3 });
        // Epic AT1: the explicit option at depths 1, 3 and the advertised max.
        let explicit = |depth| {
            (
                depth,
                request(Speculative::proposer(SpeculativeProposer::Mtp, depth)),
            )
        };
        for (case, (depth, req)) in [
            ("explicit 1", explicit(1)),
            ("explicit 3", explicit(3)),
            ("explicit max", explicit(mtp.max_depth)),
            ("legacy", (3, legacy)),
            // `auto` runs at the advertised recommended depth (sc-24446), not a literal.
            ("auto", (mtp.recommended_depth, request(Speculative::Auto))),
        ] {
            provider.validate(&req).expect(case);
            let (events, out) = token_events(&provider, &req);
            let report = out.decode.unwrap();
            assert_eq!(report.proposer, ProposerKind::Mtp, "{label} {case}");
            assert_eq!(report.draft_tokens, Some(depth), "{label} {case}");
            assert!(
                report.proposed_tokens > 0,
                "{label} {case}: the head drafted"
            );
            assert!(report.fallbacks.is_empty(), "{label} {case}");
            assert_eq!(events, off, "{label} {case}: greedy-exact");
        }

        // The head is priced and built: under Q8 the estimate is the load's built copy
        // (the MTP head's attention, MoE experts, shared expert and `fc` included).
        let census = loaded_census(&spec_at(dir.path(), Some(Quantize::Q8)));
        assert_eq!(
            estimate(&dir),
            ggml_copy_built(&census, candle_core::quantized::GgmlDType::Q8_0),
            "{label}"
        );
        assert!(
            estimate(&dir) > estimate(&headless),
            "{label}: the head is priced"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_moe_snapshot_with_an_mtp_head_runs_mtp");
}

/// sc-24433 (scope e): on the qwen3_5 hybrid a multimodal request runs prompt lookup over the
/// image-expanded history — the placeholder run included — and is greedy-exact against `off`.
/// A placeholder id a proposal copies is verified like any other draft: the verify forward
/// embeds it as the text token it is at a text position, exactly what the token-at-a-time loop
/// does when the model emits it, so lookup over the spliced history is sound (at worst a
/// rejected draft).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn prompt_lookup_is_greedy_exact_on_a_qwen35_multimodal_request() {
    use core_llm::{Speculative, SpeculativeProposer};
    use core_llm_testkit::ParityCase;

    let (_dir, provider) = synthetic_qwen35_vl_provider();
    let lookup = |depth| ParityCase {
        speculative: Speculative::proposer(SpeculativeProposer::PromptLookup, depth),
        expect_proposer: ProposerKind::PromptLookup,
    };
    let rows = core_llm_testkit::check_speculative_greedy_parity(
        &provider,
        &image_fixture_prompts(),
        &[
            lookup(1),
            lookup(table_depths(&candle_core::Device::Cpu).prompt_lookup),
            lookup(super::PROMPT_LOOKUP_MAX_DEPTH),
            ParityCase {
                speculative: Speculative::Auto,
                expect_proposer: ProposerKind::PromptLookup,
            },
        ],
        24,
    )
    .unwrap_or_else(|failures| panic!("{failures}"));
    assert_eq!(rows.len(), 2 * 4);
    assert!(rows.iter().all(|r| r.report.path == "prompt_lookup"));
    assert!(rows.iter().all(|r| r.generated_tokens == 24));
    assert_rows_drafted(&rows);
    // The rows ran the multimodal prefill: the 64 image tokens are in the prompt.
    let (_, out) = token_events(
        &provider,
        &core_llm_testkit::bench_request(
            &image_fixture_prompts()[1],
            Speculative::Auto,
            &core_llm::Sampling::greedy(),
            4,
        ),
    );
    assert!(out.usage.prompt_tokens > 64, "{:?}", out.usage);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prompt_lookup_is_greedy_exact_on_a_qwen35_multimodal_request");
}

/// sc-24446 (epic sc-24432 E1, E8): a Qwen3-VL (generic-causal) multimodal request decodes
/// on the engine — its DeepStack / M-RoPE prefill fills the engine's step cache — and the
/// engine's `off` stream is the reference loop's, token event for token event, on both image
/// prompts (the image run at the head and mid-prompt). With the reference loop selected as
/// the `off` oracle, prompt lookup at depths 1, 3 and the advertised max — and `auto`, which
/// resolves to it — runs on the engine over the image-expanded history and emits exactly the
/// reference stream.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen3vl_multimodal_decodes_on_the_engine_greedy_exact_against_the_reference_loop() {
    use core_llm::{Speculative, SpeculativeProposer};
    use core_llm_testkit::ParityCase;

    let (_dir, mut provider) = synthetic_qwen3vl_provider();
    let prompts = image_fixture_prompts();
    let off = |prompt| {
        core_llm_testkit::bench_request(prompt, Speculative::Off, &core_llm::Sampling::greedy(), 24)
    };
    let engine: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            let (events, out) = token_events(&provider, &off(prompt));
            let report = out.decode.unwrap();
            assert_eq!(report.path, "step_model", "{}", prompt.id);
            assert_eq!(report.kv_cache, "static", "{}", prompt.id);
            assert!(out.usage.prompt_tokens > 64, "{:?}", out.usage);
            events
        })
        .collect();
    provider
        .set_decode_path(DecodePath::Reference)
        .expect("the reference loop is selectable");
    for (prompt, engine) in prompts.iter().zip(engine) {
        let (reference, out) = token_events(&provider, &off(prompt));
        assert_eq!(out.decode.unwrap().path, "reference");
        assert_eq!(out.usage.generated_tokens, 24, "{}", prompt.id);
        assert!(!reference.is_empty(), "{}", prompt.id);
        assert_eq!(reference, engine, "{}: engine off vs reference", prompt.id);
    }

    let lookup = |depth| ParityCase {
        speculative: Speculative::proposer(SpeculativeProposer::PromptLookup, depth),
        expect_proposer: ProposerKind::PromptLookup,
    };
    let rows = core_llm_testkit::check_speculative_greedy_parity(
        &provider,
        &prompts,
        &[
            lookup(1),
            lookup(3),
            lookup(super::PROMPT_LOOKUP_MAX_DEPTH),
            ParityCase {
                speculative: Speculative::Auto,
                expect_proposer: ProposerKind::PromptLookup,
            },
        ],
        24,
    )
    .unwrap_or_else(|failures| panic!("{failures}"));
    assert_eq!(rows.len(), 2 * 4);
    assert!(rows.iter().all(|r| r.report.path == "prompt_lookup"));
    assert_rows_drafted(&rows);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::qwen3vl_multimodal_decodes_on_the_engine_greedy_exact_against_the_reference_loop");
}

/// sc-24446 (epic sc-24432 E2, E3): the report of a Qwen3-VL multimodal request names the
/// engine path and the proposer that ran, and no fallback — `auto` runs prompt lookup on the
/// engine, `off` the engine with no proposer — exactly as the same request text-only does.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_qwen3vl_multimodal_report_names_the_engine_and_no_fallback() {
    use core_llm::{Sampling, Speculative};

    let (_dir, provider) = synthetic_qwen3vl_provider();
    let image = &image_fixture_prompts()[0];
    let text = &fixture_prompts()[0];
    for (label, prompt) in [("image", image), ("text", text)] {
        let request = |speculative| {
            core_llm_testkit::bench_request(prompt, speculative, &Sampling::greedy(), 12)
        };
        let auto = token_events(&provider, &request(Speculative::Auto))
            .1
            .decode
            .expect("reported");
        assert_eq!(auto.path, "prompt_lookup", "{label}");
        assert_eq!(auto.proposer, ProposerKind::PromptLookup, "{label}");
        assert!(auto.verify_steps > 0, "{label}");
        assert!(auto.fallbacks.is_empty(), "{label}: {:?}", auto.fallbacks);
        let off = token_events(&provider, &request(Speculative::Off))
            .1
            .decode
            .expect("reported");
        assert_eq!(off.path, "step_model", "{label}");
        assert_eq!(off.proposer, ProposerKind::None, "{label}");
        assert!(off.fallbacks.is_empty(), "{label}: {:?}", off.fallbacks);
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_qwen3vl_multimodal_report_names_the_engine_and_no_fallback");
}

/// The report the output carries is the record's, with the provider's fallback reasons passed
/// through verbatim and in order.
#[test]
fn the_request_report_carries_the_provider_fallbacks_verbatim() {
    let record =
        crate::decode::DecodeRecord::plain(DecodePath::Reference, 3, 3, Default::default());
    let reasons = vec![
        "speculative: `mtp` is not available for this model".to_string(),
        "speculative: second reason".to_string(),
    ];
    let report = super::request_report(&record, true, reasons.clone());
    assert_eq!(report.fallbacks, reasons);
    assert_eq!(
        report,
        core_llm::DecodeReport {
            fallbacks: reasons,
            ..record.report(true)
        }
    );
    assert!(super::request_report(&record, false, Vec::new())
        .fallbacks
        .is_empty());
}

/// sc-24433 AC3: the benchmark harness runs the whole prompt set (predictable and open-ended)
/// on the Qwen3 fixture under `off`, prompt lookup and `auto`, and writes one baseline-format
/// JSON row per (prompt, option) with the report's proposer and accepted length.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_benchmark_harness_writes_a_baseline_document_for_the_fixture_model() {
    use core_llm::{Speculative, SpeculativeProposer};
    let (_dir, provider) = synthetic_causal_provider();
    let config = core_llm_testkit::BenchConfig {
        model: "synthetic-qwen3".into(),
        backend: "candle-cpu".into(),
        max_new_tokens: 12,
        options: vec![
            Speculative::Off,
            Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
            Speculative::Auto,
        ],
        warmup: false,
        sampling: core_llm::Sampling::greedy(),
        repeats: 1,
        thinking: core_llm_testkit::BenchThinking::Default,
    };
    let prompts = core_llm_testkit::speculative_prompt_set();
    let doc = core_llm_testkit::run_speculative_bench(&provider, &prompts, &config)
        .unwrap_or_else(|e| panic!("{e}"));
    let out = tempfile::tempdir().unwrap();
    let path = out.path().join("speculative-baseline.json");
    doc.write_new(&path).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["schema"], core_llm_testkit::BENCH_SCHEMA);
    assert_eq!(json["backend"], "candle-cpu");
    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 5 * 3);
    let proposers: Vec<_> = rows[..3].iter().map(|r| r["proposer"].clone()).collect();
    assert_eq!(
        proposers,
        [
            serde_json::json!("none"),
            serde_json::json!("prompt_lookup"),
            serde_json::json!("prompt_lookup")
        ]
    );
    for row in rows {
        assert_eq!(row["generated_tokens"], 12);
        assert_eq!(row["timing_source"], "backend");
        for key in [
            "decode_tok_s",
            "ttft_ms",
            "mean_accepted_length",
            "accepted_tokens",
        ] {
            assert!(row.get(key).is_some(), "row lacks `{key}`: {row}");
        }
    }
    assert!(rows[1]["mean_accepted_length"].is_number());
    assert!(
        rows[0]["mean_accepted_length"].is_null(),
        "off never verifies drafts"
    );
    assert_eq!(json["sampling"]["temperature"], 0.0);

    // The product's default on candle-cuda is `auto` + temperature: a seeded stochastic run
    // of the same table records its knobs and every row's sampler path.
    let stochastic = core_llm_testkit::BenchConfig {
        sampling: core_llm_testkit::parse_bench_sampling(r#"{"temperature": 0.7, "top_p": 0.9}"#)
            .unwrap(),
        ..config
    };
    let doc = core_llm_testkit::run_speculative_bench(&provider, &prompts, &stochastic)
        .unwrap_or_else(|e| panic!("{e}"));
    let json = doc.to_json();
    assert_eq!(json["sampling"]["temperature"], 0.7);
    assert_eq!(json["sampling"]["top_p"], 0.9);
    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 5 * 3);
    for row in rows {
        let sampler = row["sampler"].as_str().unwrap_or_else(|| panic!("{row}"));
        assert!(!sampler.is_empty() && sampler != "none", "{row}");
        if provider.model.device().is_cpu() {
            // No device sampler on the CPU: every stochastic draw is the host's.
            assert_eq!(sampler, "host:device_unavailable", "{row}");
        }
        assert_eq!(row["generated_tokens"], 12);
    }
    assert_eq!(rows[1]["proposer"], "prompt_lookup");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::the_benchmark_harness_writes_a_baseline_document_for_the_fixture_model");
}

/// sc-24134, sc-24139, sc-24140: with the CUDA-graph switch on — the one the provider was
/// loaded under (`LoadSpec::cuda_graphs`; the stream is settled at load) — a request whose
/// speculation is off runs through the runner (the engine with no proposer) and its record
/// says what the runner did with each step; a request decoded on the reference loop never runs
/// through the runner, and its record says why rather than a bare `graph: none`. With the
/// switch off neither record shows the runner.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_runner_wraps_the_off_path_and_a_reference_record_names_why_it_did_not_run() {
    use core_llm::{LoadSpec, Message, Sampling, TextLlm, TextLlmRequest};

    let dir = synthetic_qwen35_snapshot_without_mtp();
    let request = TextLlmRequest {
        messages: vec![Message::user("t3 t7 t11 t2")],
        sampling: Sampling::greedy(),
        max_new_tokens: 3,
        seed: Some(0),
        mtp: None,
        ..Default::default()
    };
    let load = |on: bool| {
        super::LlamaProvider::load(&LoadSpec {
            cuda_graphs: Some(on),
            ..LoadSpec::dense(dir.path().display().to_string())
        })
        .expect("load the synthetic Qwen3.5 snapshot")
    };
    let run = |provider: &super::LlamaProvider| {
        provider.generate(&request, &mut |_| {}).unwrap();
        provider.last_decode_record().unwrap()
    };

    let (mut on, mut off) = (load(true), load(false));
    let engine = run(&on);
    assert_eq!(engine.path, DecodePath::StepModel);
    // Every target forward went through the runner — once each, or, for the step a graph is
    // captured at (sc-24441: the dense decoders capture on CUDA), as the eager reference plus
    // the eager re-run of the capture check.
    assert!(
        engine.cuda_graphs.eager + engine.cuda_graphs.replayed >= engine.target_forwards,
        "every step of the Off request went through the runner: {}",
        engine.cuda_graphs.describe()
    );
    // An eager step names why — unless the runner could capture this decoder (sc-24441: the
    // dense decoders capture on CUDA), where an eager step without a reason is a prefill
    // segment, a warm-up or a capture's eager reference, and a 3-token request can end before
    // any shape is captured or replayed.
    match engine.cuda_graphs.fallback_reason {
        Some(reason) => assert_ne!(reason, crate::decode::graph::REASON_REFERENCE_PATH),
        None => assert!(
            on.model.device().is_cuda() && on.model.graph_support().is_ok(),
            "an eager step with no reason on a decoder the runner cannot capture: {}",
            engine.cuda_graphs.describe()
        ),
    }
    assert_eq!(run(&off).cuda_graphs.label(), "none");

    let describe = |provider: &mut super::LlamaProvider| {
        provider.set_decode_path(DecodePath::Reference).unwrap();
        let record = run(provider);
        assert_eq!(record.path, DecodePath::Reference);
        record.cuda_graphs.describe()
    };
    assert_eq!(
        describe(&mut on),
        "graph: none replayed=0 eager=0 captured=0 fallback=reference_path"
    );
    assert_eq!(
        describe(&mut off),
        "graph: none replayed=0 eager=0 captured=0"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::the_runner_wraps_the_off_path_and_a_reference_record_names_why_it_did_not_run");
}

/// sc-24140 (E7): a static KV cache's capacity refusal reaches the contract as the typed
/// `RequestResourceExhausted` — the request's geometry plus the preallocation the requested
/// positions would take against the one the cache holds — never an opaque backend error; an
/// unrelated engine error still maps as before, and without a request's geometry the refusal
/// is still the request's fault (`InvalidRequest`), not the backend's.
#[test]
fn a_kv_capacity_refusal_is_a_typed_resource_exhaustion() {
    let (_dir, provider) = synthetic_qwen35_provider_without_mtp();
    let (requested, capacity) = (70usize, 64usize);
    let error = provider.request_error(
        crate::Error::KvCapacityExceeded {
            requested,
            capacity,
        },
        9,
        61,
    );
    let core_llm::Error::RequestResourceExhausted(evidence) = error else {
        panic!("expected RequestResourceExhausted, got {error:?}");
    };
    assert_eq!(
        evidence,
        core_llm::RequestResourceExhausted {
            prompt_tokens: 9,
            max_new_tokens: 61,
            max_context_tokens: provider.descriptor.capabilities.max_context_tokens,
            required_bytes: provider.model.static_kv_bytes(requested) as u64,
            available_bytes: provider.model.static_kv_bytes(capacity) as u64,
        }
    );
    assert!(evidence.required_bytes > evidence.available_bytes);
    assert!(matches!(
        provider.request_error(crate::Error::Canceled, 9, 61),
        core_llm::Error::Canceled
    ));
    assert!(matches!(
        super::to_core(crate::Error::KvCapacityExceeded {
            requested,
            capacity
        }),
        core_llm::Error::InvalidRequest(_)
    ));
}

/// sc-24140 (E0 / E2): a temperature + top-p request whose speculation is off — ChatWorks'
/// default — decodes on the engine with no proposer and draws every token through `sample`:
/// on CUDA the device sampler (zero logits rows copied to the host), on CPU the host reference
/// and the record says why; every draw is recorded either way.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_stochastic_off_request_draws_through_the_device_sampler_where_there_is_one() {
    use core_llm::{Message, Sampling, TextLlm, TextLlmRequest};

    let (_dir, provider) = synthetic_qwen35_provider_without_mtp();
    let request = TextLlmRequest {
        messages: vec![Message::user("t3 t7 t11 t2 t7 t11")],
        sampling: Sampling {
            temperature: 0.8,
            top_p: 0.9,
            ..Sampling::greedy()
        },
        max_new_tokens: 8,
        seed: Some(5),
        mtp: None,
        ..Default::default()
    };
    let out = provider.generate(&request, &mut |_| {}).unwrap();
    let record = provider.last_decode_record().unwrap();
    assert_eq!(record.path, DecodePath::StepModel);
    // A stop token is drawn but not emitted.
    let stopped = out.finish_reason == Some(core_llm::FinishReason::Stop);
    let draws = u64::from(out.usage.generated_tokens) + u64::from(stopped);
    assert!(draws > 0);
    assert_eq!(
        record.sampler.device_draws + record.sampler.host_draws,
        draws,
        "every draw is recorded: {:?}",
        record.sampler
    );
    if provider.model.device().is_cuda() {
        assert_eq!(record.sampler.label(), "device");
        assert_eq!(record.sampler.device_draws, draws);
        assert_eq!(
            record.sampler.logits_to_host, 0,
            "no logits row reached the host"
        );
    } else {
        assert_eq!(record.sampler.label(), "host:device_unavailable");
        assert_eq!(record.sampler.host_draws, draws);
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_stochastic_off_request_draws_through_the_device_sampler_where_there_is_one");
}

// ---- Prism/Bonsai companion MTP head (epic sc-24432, story sc-24444), Candle twin. ----
//
// `LlamaProvider::load` refuses Prism/Bonsai on a CPU device (it needs an accelerator), so
// the real load path — admission, attach, the load report — runs on the synthetic dense
// qwen3_5 target without a native head (the same attach seam), and the Prism target runs
// through `load_dir` on the CPU plus the same `attach_mtp_head` the load calls.

/// A companion head for `text` (else `default_text`), in a self-removing directory.
fn companion_head(text: &serde_json::Value) -> tempfile::TempDir {
    let head = tempfile::Builder::new()
        .prefix("candle-mtp-head-")
        .tempdir()
        .unwrap();
    crate::models::qwen35::tests::write_companion_head(head.path(), text, 5, None);
    head
}

/// The synthetic dense qwen3_5 target's `text_config`.
fn dense_text() -> serde_json::Value {
    crate::models::qwen35::tests::text_model_snapshot_parts().2["text_config"].clone()
}

/// The bytes a load of `spec` admits in the domain the load's budget caps — the device's on
/// a CUDA test host (the operational budget is device headroom there, and the prefix cache
/// settles in it), the host's otherwise.
fn admitted_bytes(spec: &core_llm::LoadSpec) -> u64 {
    admitted_bytes_as(spec, true)
}

/// What load admission prices `spec` at on this host's load device — as the provider's target
/// (`target`: wrapped by the graph runner under its policy) or as a resident draft (never
/// wrapped: no graph parameter cache).
fn admitted_bytes_as(spec: &core_llm::LoadSpec, target: bool) -> u64 {
    let cuda = crate::device::select_device().unwrap().is_cuda();
    let estimate = super::LlamaProvider::load_memory_estimate_as(spec, cuda, target).unwrap();
    match cuda {
        true => estimate.device_required_bytes.unwrap(),
        false => estimate.host_required_bytes,
    }
}

/// The graph parameter cache `spec` adds over `base` on this host's load device: non-zero only
/// where the runner wraps the load (CUDA, graphs on) and `spec` asks for hidden-row captures
/// (an MTP head) that `base` does not.
fn graph_cache_over(spec: &core_llm::LoadSpec, base: &core_llm::LoadSpec) -> u64 {
    let cuda = crate::device::select_device().unwrap().is_cuda();
    let cache = |s| {
        super::LlamaProvider::load_memory_estimate(s, cuda)
            .unwrap()
            .graph_param_cache_bytes
    };
    cache(spec) - cache(base)
}

fn with_head(target: &tempfile::TempDir, head: &std::path::Path) -> core_llm::LoadSpec {
    core_llm::LoadSpec::dense(target.path().display().to_string())
        .with_mtp_head(head.display().to_string())
}

/// The Prism fixture as a CPU provider through `load_dir`, with `head` attached as `load` does.
fn prism_provider(head: Option<&std::path::Path>) -> (tempfile::TempDir, super::LlamaProvider) {
    let target = tempfile::Builder::new()
        .prefix("candle-prism-target-")
        .tempdir()
        .unwrap();
    crate::models::qwen35::tests::write_prism_snapshot(target.path(), 6);
    std::fs::write(
        target.path().join("tokenizer.json"),
        synthetic_tokenizer_json(64),
    )
    .unwrap();
    std::fs::write(
        target.path().join("generation_config.json"),
        r#"{"eos_token_id":999}"#,
    )
    .unwrap();
    let mut provider =
        super::LlamaProvider::load_dir(target.path(), &candle_core::Device::Cpu, None).unwrap();
    provider.attach_mtp_head(head);
    (target, provider)
}

fn head_request(speculative: core_llm::Speculative) -> core_llm::TextLlmRequest {
    core_llm::TextLlmRequest {
        messages: vec![core_llm::Message::user("t3 t9 t40 t11 t3 t9 t40 t11 t7")],
        sampling: core_llm::Sampling::greedy(),
        max_new_tokens: 20,
        seed: Some(3),
        speculative: Some(speculative),
        ..Default::default()
    }
}

fn mtp_advertised(provider: &super::LlamaProvider) -> Option<core_llm::ProposerCapabilities> {
    core_llm::TextLlm::descriptor(provider)
        .capabilities
        .proposer(core_llm::SpeculativeProposer::Mtp)
}

/// An attached head: advertised at the backend's max depth, its eight projections resident
/// (Q8_0 repacks of its 4-bit triples) in the census, and `{proposer: mtp}` at depths 1, 3 and
/// the advertised max emits exactly `off`'s greedy tokens with the head drafting.
fn assert_head_decodes_the_off_tokens(provider: &super::LlamaProvider, label: &str) {
    use core_llm::{Speculative, SpeculativeProposer, TextLlm};
    let fallbacks = provider.load_report().unwrap().fallbacks;
    assert!(fallbacks.is_empty(), "{label}: {fallbacks:?}");
    let advertised = mtp_advertised(provider).expect("the attached head is advertised");
    assert_eq!(advertised.max_depth, super::MTP_MAX_DEPTH, "{label}");
    assert_eq!(
        advertised.recommended_depth,
        table_depths(provider.model.device()).mtp,
        "{label}"
    );
    let census = provider.load_record().census.unwrap();
    assert_eq!(
        census.projections.ggml.count, 8,
        "{label}: the head's projections"
    );

    let (off, off_out) = token_events(provider, &head_request(Speculative::Off));
    assert_eq!(off.len(), 20, "{label}");
    let distinct: std::collections::BTreeSet<_> = off.iter().map(|(id, _)| id).collect();
    assert!(
        distinct.len() > 2,
        "{label}: fixture is degenerate: {off:?}"
    );
    assert_eq!(off_out.decode.unwrap().proposer, ProposerKind::None);
    for depth in [1, 3, advertised.max_depth] {
        let req = head_request(Speculative::proposer(SpeculativeProposer::Mtp, depth));
        let (events, out) = token_events(provider, &req);
        assert_eq!(events, off, "{label}: depth {depth}");
        let report = out.decode.expect("the engine reports its path");
        assert_eq!(report.proposer, ProposerKind::Mtp, "{label}: depth {depth}");
        assert_eq!(report.draft_tokens, Some(depth), "{label}: depth {depth}");
        assert!(
            report.proposed_tokens > 0,
            "{label}: depth {depth}: no drafts"
        );
        assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    }
}

/// Story sc-24444 AC2 on Candle: a companion head attached to a Prism target (packed
/// embedding and `lm_head`) and, through the real load, to a dense qwen3_5 target decodes
/// `{proposer: mtp}` identical to `off`; without the head neither advertises MTP and the
/// Prism target decodes the same greedy tokens.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_companion_head_decodes_mtp_with_the_off_tokens() {
    use core_llm::Speculative;
    let prism_head = companion_head(&crate::models::qwen35::tests::prism_text_config());
    let (_prism, prism) = prism_provider(Some(prism_head.path()));
    assert_eq!(
        core_llm::TextLlm::descriptor(&prism).family,
        "prism_hadamard_qwen35"
    );
    assert_head_decodes_the_off_tokens(&prism, "prism");
    let (_bare_dir, bare) = prism_provider(None);
    assert!(mtp_advertised(&bare).is_none());
    assert_eq!(bare.load_record().census.unwrap().projections.ggml.count, 0);
    assert_eq!(
        token_events(&bare, &head_request(Speculative::Off)).0,
        token_events(&prism, &head_request(Speculative::Off)).0
    );

    let target = synthetic_qwen35_snapshot_without_mtp();
    let head = companion_head(&dense_text());
    let dense = super::LlamaProvider::load(&with_head(&target, head.path())).unwrap();
    assert_head_decodes_the_off_tokens(&dense, "dense");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::a_companion_head_decodes_mtp_with_the_off_tokens");
}

/// Story sc-24444 AC2 / E2 on Candle: through the real load, a head whose geometry does not
/// match and a missing head are each a named load fallback — the model still loads, advertises
/// no MTP and decodes; a head named for a non-Qwen3.5 target or for a target with its own
/// predictor is a named fallback too.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_unattachable_companion_head_is_a_named_load_fallback_never_a_load_failure() {
    use core_llm::{Speculative, SpeculativeProposer, TextLlm};
    let target = synthetic_qwen35_snapshot_without_mtp();
    let mut text = dense_text();
    text["intermediate_size"] = serde_json::json!(128);
    let mismatched = companion_head(&text);
    let provider = super::LlamaProvider::load(&with_head(&target, mismatched.path()))
        .expect("the target still loads (E2)");
    let fallbacks = provider.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(fallbacks[0].starts_with("mtp_head: "), "{fallbacks:?}");
    assert!(
        fallbacks[0].contains("intermediate_size 128 != target 64"),
        "{fallbacks:?}"
    );
    assert!(
        fallbacks[0].contains("the model loaded without a companion head"),
        "{fallbacks:?}"
    );
    assert!(mtp_advertised(&provider).is_none());
    // An explicit MTP request then decodes plainly, the reason named (E2), never refused.
    let (mtp_ids, mtp_out) = token_events(
        &provider,
        &head_request(Speculative::proposer(SpeculativeProposer::Mtp, 3)),
    );
    let (ids, out) = token_events(&provider, &head_request(Speculative::Off));
    assert_eq!(ids.len(), 20);
    assert!(out.decode.unwrap().fallbacks.is_empty());
    assert_eq!(mtp_ids, ids, "the plain path's tokens");
    let mtp_report = mtp_out.decode.unwrap();
    assert_eq!(mtp_report.proposer, ProposerKind::None);
    assert_eq!(mtp_report.fallbacks.len(), 1, "{:?}", mtp_report.fallbacks);
    assert!(
        mtp_report.fallbacks[0].starts_with("speculative: `mtp` is not available"),
        "{:?}",
        mtp_report.fallbacks
    );

    let missing = with_head(&target, &target.path().join("no-such-head"));
    let provider = super::LlamaProvider::load(&missing).expect("a missing head is not fatal");
    let fallbacks = provider.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(fallbacks[0].contains("is not a directory"), "{fallbacks:?}");

    let head = companion_head(&dense_text());
    let (_causal_dir, mut causal) = synthetic_causal_provider();
    causal.attach_mtp_head(Some(head.path()));
    let fallbacks = causal.load_record().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(
        fallbacks[0].starts_with(super::CANDLE_MTP_HEAD_FALLBACK),
        "{fallbacks:?}"
    );
    assert!(
        fallbacks[0].contains(&head.path().display().to_string()),
        "{fallbacks:?}"
    );
    assert!(causal.mtp.is_none());
    assert!(mtp_advertised(&causal).is_none());

    let native = q8_capable_qwen35_snapshot(false);
    let native = super::LlamaProvider::load(&with_head(&native, head.path()))
        .expect("the native-head target loads");
    let fallbacks = native.load_report().unwrap().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(
        fallbacks[0].contains("already carries its own MTP predictor"),
        "{fallbacks:?}"
    );
    assert!(mtp_advertised(&native).is_some(), "its own head still runs");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::an_unattachable_companion_head_is_a_named_load_fallback_never_a_load_failure");
}

/// Story sc-24444 E7 on Candle: the head's priced bytes are its payload plus what the build
/// allocates beside it (each 4-bit triple's Q8_0 repack, each norm's f32 `1 + w`), and load
/// admission adds them to the target's — through the real load, a budget that fits the target
/// but not target + head refuses the head by name and loads the target; one more byte admits
/// both.
/// Story sc-24444 E7 with sc-24436/sc-24437 on Candle: the companion head, a draft model and
/// the prefix cache are admitted together through the real load — the head on top of the
/// target and the admitted draft, the prefix cache in what they leave. A budget that fits
/// target + draft but not the head keeps the draft, refuses the head by name and gives the
/// cache the rest; one that fits all three plus `x` bytes settles the cache at exactly `x`.
#[test]
fn companion_head_draft_and_prefix_cache_are_admitted_together() {
    use core_llm::TextLlm;
    let target = synthetic_qwen35_snapshot_without_mtp();
    let head = companion_head(&dense_text());
    let head_bytes = super::companion_head_bytes(head.path()).unwrap();
    let source = target.path().display().to_string();
    let mut spec = with_head(&target, head.path()).with_draft(source.clone());
    spec.prefix_cache_bytes = Some(u64::MAX / 4);
    // The target as this spec loads it (with a head, the graph runner's hidden-row captures
    // are priced too), the draft as a draft (never graph-wrapped).
    let target_bytes = admitted_bytes(&with_head(&target, head.path()));
    let draft_bytes = admitted_bytes_as(&super::draft_load_spec(&spec, &source), false);

    let short = super::with_load_budget(target_bytes + draft_bytes + head_bytes - 1, || {
        super::LlamaProvider::load(&spec)
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
    let all = super::with_load_budget(target_bytes + draft_bytes + head_bytes + x, || {
        super::LlamaProvider::load(&spec)
    })
    .expect("all three fit");
    let report = all.load_report().unwrap();
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    assert!(report.draft.as_ref().unwrap().is_resident(), "{report:?}");
    assert_eq!(report.prefix_cache_bytes, Some(x));
    assert!(mtp_advertised(&all).is_some());
}

#[test]
fn companion_head_bytes_are_counted_by_load_admission() {
    use crate::primitives::projection::ggml_device_bytes;
    use candle_core::quantized::GgmlDType;
    let target = synthetic_qwen35_snapshot_without_mtp();
    let head = companion_head(&dense_text());
    let head_bytes = super::companion_head_bytes(head.path()).unwrap();
    let payload = core_llm::checkpoint_payload_bytes(head.path()).unwrap();
    let geometry = crate::models::Qwen35Config::from_json(&dense_text())
        .unwrap()
        .companion_mtp_geometry();
    let repack: u64 = geometry
        .matrices()
        .iter()
        .map(|(_, [rows, cols])| ggml_device_bytes((rows * cols) as u64, GgmlDType::Q8_0).unwrap())
        .sum();
    let norms: u64 = geometry.norms().iter().map(|(_, w)| *w as u64 * 4).sum();
    assert_eq!(head_bytes, payload + repack + norms);

    let spec = with_head(&target, head.path());
    let dense = core_llm::LoadSpec::dense(target.path().display().to_string());
    // The head's bytes are admitted on top of the target's own price, which a head changes
    // only by the graph runner's hidden-row captures (sc-24446: CUDA with graphs on).
    let target_bytes = admitted_bytes(&spec);
    assert_eq!(
        target_bytes,
        admitted_bytes(&dense) + graph_cache_over(&spec, &dense),
        "the target's own price; the head is admitted on top of it"
    );
    let short = super::with_load_budget(target_bytes + head_bytes - 1, || {
        super::LlamaProvider::load(&spec)
    })
    .expect("the target fits and still loads (E2)");
    let fallbacks = short.load_record().fallbacks;
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    assert!(
        fallbacks[0].starts_with("mtp_head: refused by load admission"),
        "{fallbacks:?}"
    );
    assert!(
        fallbacks[0].contains(&head_bytes.to_string()),
        "{fallbacks:?}"
    );
    assert!(mtp_advertised(&short).is_none());
    let fits = super::with_load_budget(target_bytes + head_bytes, || {
        super::LlamaProvider::load(&spec)
    })
    .expect("target + head fit");
    // Nothing beside target + head is left for the prefix cache: it settles to 0 bytes, named
    // (E2), and the head attaches.
    assert_eq!(fits.load_record().prefix_cache_bytes, Some(0));
    assert_eq!(
        fits.load_record().fallbacks,
        core_llm::prefix_budget_fallback(
            crate::device::decode_backend(&crate::device::select_device().unwrap()),
            None,
            0,
        )
        .into_iter()
        .collect::<Vec<_>>()
    );
    assert!(mtp_advertised(&fits).is_some());
    // The prefix-cache snapshot of an MTP request carries the head's KV too (sc-24437).
    let snapshot = |mtp| fits.prefix_snapshot_bytes(64, Some(64), mtp).unwrap();
    assert!(
        snapshot(true) > snapshot(false) + 4 * 64,
        "the head's KV is priced"
    );
}

/// sc-24446 (E5): a provider loads with its device's defaults-table speculative default, and a
/// request that leaves the option unset runs whatever the provider's default is — here set to
/// `auto` so the hook is observable — while an explicit off (new or legacy spelling) wins.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_unset_speculative_option_runs_the_providers_default() {
    let root = tempfile::tempdir().unwrap();
    let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
    let mut provider =
        super::LlamaProvider::load(&LoadSpec::dense(fixture.target.to_string_lossy())).unwrap();
    assert_eq!(
        provider.speculative_default(),
        crate::device::decode_defaults(provider.model.device()).speculative,
        "loaded with the defaults table's default"
    );
    let prompt = &core_llm_testkit::draft_model_prompts()[0];
    core_llm_testkit::check_speculative_default(
        &provider,
        crate::device::decode_defaults(provider.model.device()).speculative,
        prompt,
        8,
    );
    provider.set_speculative_default(core_llm::Speculative::Auto);
    core_llm_testkit::check_speculative_default(&provider, core_llm::Speculative::Auto, prompt, 8);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::an_unset_speculative_option_runs_the_providers_default");
}

/// The cross-turn prefix cache through the provider (sc-24437).
mod prefix_cache {
    use super::super::*;
    use super::{synthetic_tokenizer_json, tiny_causal_from, tiny_llama_parts, write_snapshot};
    use core_llm::Speculative;

    fn tokenizer(vocab: usize) -> Tokenizer {
        Tokenizer::from_json(&synthetic_tokenizer_json(vocab)).unwrap()
    }

    fn causal_provider() -> LlamaProvider {
        let (cfg, weights) = tiny_llama_parts();
        LlamaProvider::from_parts(tiny_causal_from(cfg, weights), tokenizer(40), vec![])
    }

    fn qwen35_provider(with_mtp: bool) -> LlamaProvider {
        let (model, mtp) = if with_mtp {
            let (_, model, head) = crate::models::qwen35::tests::text_model_with_mtp();
            (model, Some(head))
        } else {
            (crate::models::qwen35::tests::text_model().1, None)
        };
        let mut descriptor = descriptor_for_qwen35(model.config(), model.device());
        if mtp.is_some() {
            descriptor.capabilities.mtp = Some(core_llm::MtpCapabilities {
                max_draft_tokens: u32::MAX,
                recommended_draft_tokens: 3,
            });
        }
        LlamaProvider {
            descriptor,
            model: Decoder::Qwen35(model),
            mtp,
            ..LlamaProvider::from_parts(
                tiny_causal_from(tiny_llama_parts().0, tiny_llama_parts().1),
                tokenizer(50),
                vec![],
            )
        }
    }

    fn chat(messages: Vec<Message>, speculative: Speculative) -> TextLlmRequest {
        TextLlmRequest {
            messages,
            sampling: Sampling::greedy(),
            max_new_tokens: 8,
            seed: Some(3),
            speculative: Some(speculative),
            ..Default::default()
        }
    }

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
                    date_string: None,
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

    /// A cold run: the reference `Decode` loop on a fresh provider (no engine, no cache).
    fn reference(fresh: LlamaProvider, req: &TextLlmRequest) -> Vec<u32> {
        let mut fresh = fresh;
        fresh.set_decode_path(DecodePath::Reference).unwrap();
        run(&fresh, &chat(req.messages.clone(), Speculative::Off)).1
    }

    /// AC1 on the softmax family through the provider: turn 2 restores the shared run of turn
    /// 1's cached `prompt + reply` (N), and decodes exactly what a cold run decodes.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn a_causal_second_turn_reports_its_prefix_hit_and_matches_a_cold_run() {
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
        let mut held = p1.clone();
        held.extend(ids1.iter().map(|&t| t as i32));
        held.truncate(p1.len() + ids1.len() - 1);
        let n = held
            .iter()
            .zip(&p2)
            .take_while(|(a, b)| a == b)
            .count()
            .min(p2.len() - 1);
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
        assert_eq!(report, provider.last_decode_record().unwrap().report(false));
        assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
        assert_eq!(ids2, run(&causal_provider(), &turn2).1);
        assert_eq!(ids2, reference(causal_provider(), &turn2));
        assert!(provider.prefix_cache_resident_bytes() <= provider.prefix_cache_budget());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prefix_cache::a_causal_second_turn_reports_its_prefix_hit_and_matches_a_cold_run");
    }

    /// AC2 through the provider: the hybrid snapshots at the end of turn 1's conversation;
    /// turn 2 and turn 3 restore the recurrent state there (the ring's oldest position) and
    /// match a cold run — plain and with the MTP head resuming from the stored boundary.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn a_qwen35_second_turn_restores_the_boundary_state_and_matches_a_cold_run() {
        for speculative in [
            Speculative::Off,
            Speculative::proposer(SpeculativeProposer::Mtp, 3),
        ] {
            let mtp = speculative != Speculative::Off;
            let provider = qwen35_provider(mtp);
            let user = Message::user("t3 t9 t4 t11 t3 t9 t4 t11");
            let turn1 = chat(vec![user.clone()], speculative);
            let (out1, _) = run(&provider, &turn1);
            assert_eq!(out1.decode.as_ref().unwrap().prefix_hit_tokens, 0);
            let boundary = rendered_ids(&provider, &turn1, false).len();
            assert_eq!(provider.prefix_cache().keys().len(), 1);
            assert_eq!(provider.prefix_cache().keys()[0].len(), boundary);
            let held = provider.prefix_cache_resident_bytes();
            let estimate = provider
                .prefix_snapshot_bytes(0, Some(boundary), mtp)
                .unwrap();
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
                assert_eq!(report.prefix_hit_tokens, boundary as u64, "{speculative:?}");
                assert_eq!(report.prefix_cache.path, "hit");
                assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
                if mtp {
                    assert_eq!(report.proposer, ProposerKind::Mtp);
                    assert!(report.proposed_tokens > 0);
                }
                assert_eq!(ids, run(&qwen35_provider(mtp), &turn).1, "{speculative:?}");
                assert_eq!(
                    ids,
                    reference(qwen35_provider(mtp), &turn),
                    "{speculative:?}"
                );
            }
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prefix_cache::a_qwen35_second_turn_restores_the_boundary_state_and_matches_a_cold_run");
    }

    /// The reference loop (the parity oracle) never reads or feeds the cache, and says so.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn the_reference_loop_prefills_cold_and_names_why() {
        let mut provider = causal_provider();
        provider.set_decode_path(DecodePath::Reference).unwrap();
        let req = chat(vec![Message::user("t3 t9 t4")], Speculative::Off);
        for _ in 0..2 {
            let report = run(&provider, &req).0.decode.unwrap();
            assert_eq!(report.prefix_hit_tokens, 0);
            assert_eq!(report.prefix_cache.path, "bypassed");
            let reason = report.prefix_cache.reason.unwrap();
            assert!(reason.starts_with("the reference decode loop"), "{reason}");
            assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
        }
        assert!(provider.prefix_cache().is_empty());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prefix_cache::the_reference_loop_prefills_cold_and_names_why");
    }

    /// Sets the operational memory override for one scope (tests run on one thread,
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

    /// E7 at load: the budget is what the load's admission leaves, never more; at the exact
    /// boundary the load still loads, with no cache.
    #[test]
    fn the_load_admits_the_prefix_cache_budget_up_to_its_headroom() {
        let (cfg, weights) = tiny_llama_parts();
        let dir = write_snapshot(&cfg, &weights, 40);
        let mut spec = LoadSpec::dense(dir.path().display().to_string());
        // In the domain the operational budget caps — the device's on a CUDA test host, where
        // the prefix cache lives too; the host's otherwise.
        let required = super::admitted_bytes(&spec);
        let headroom = 4096;
        // The estimate names what the load asks for, beside what it requires.
        let asked = |spec: &LoadSpec| {
            LlamaProvider::load_memory_estimate(spec, false)
                .unwrap()
                .prefix_cache_bytes
        };
        assert_eq!(
            asked(&spec),
            crate::device::decode_backend_for(false)
                .defaults()
                .prefix_cache_bytes,
            "unset: the backend's defaults-table budget"
        );
        spec.prefix_cache_bytes = Some(u64::MAX);
        assert_eq!(asked(&spec), u64::MAX);
        assert_eq!(
            super::admitted_bytes(&spec),
            required,
            "the budget is never part of what the load requires"
        );
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
            let provider = LlamaProvider::load(&spec).unwrap();
            assert_eq!(
                provider.prefix_cache_budget(),
                headroom.min(
                    crate::device::decode_defaults(provider.model.device()).prefix_cache_bytes
                )
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

    /// A paged KV cache keeps no snapshot and the report says why — not as a failed copy.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn a_paged_cache_names_why_it_keeps_no_snapshot() {
        let model = tiny_causal_from(tiny_llama_parts().0, tiny_llama_parts().1);
        let mut cache = model.new_paged_step_cache(4);
        let prompt = [3, 9, 4, 11, 5];
        StepModel::forward_step(
            &model,
            &mut cache,
            crate::decode::StepRequest::last(&prompt),
        )
        .unwrap();
        let err = PrefixCache::with_budget(1 << 30)
            .store(&prompt, &[], &cache, None, None)
            .unwrap_err();
        assert_eq!(prefix_store_reason(&err), PREFIX_PAGED_NOT_SNAPSHOTTED);
        assert_eq!(
            prefix_store_reason(&crate::Error::Msg("copy".into())),
            PREFIX_COPY_FAILED
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prefix_cache::a_paged_cache_names_why_it_keeps_no_snapshot");
    }

    fn required_bytes(provider: &LlamaProvider, req: &TextLlmRequest) -> u64 {
        let prompt = rendered_ids(provider, req, true);
        priced_request_bytes(
            &provider.model,
            DecodeRoute::PLAIN_ENGINE,
            prompt.len(),
            req.max_new_tokens,
            0,
            false,
        )
        .unwrap()
    }

    /// E7 per request, on the softmax family: a request keeps its KV copy only when memory
    /// holds the request and the copy; held entries are evicted to make room, and one byte
    /// short of that the request still runs — without keeping anything.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn a_request_keeps_its_snapshot_only_when_admission_holds_it() {
        let req = chat(vec![Message::user("t11 t12 t13 t14")], Speculative::Off);
        let probe = causal_provider();
        let prompt = rendered_ids(&probe, &req, true);
        let required = required_bytes(&probe, &req);
        let snapshot = probe
            .prefix_snapshot_bytes(prompt.len() + req.max_new_tokens as usize, None, false)
            .unwrap();
        {
            let provider = causal_provider();
            let _budget = MemoryOverride::set(required + snapshot - 1);
            let report = provider
                .generate(&req, &mut |_| {})
                .unwrap()
                .decode
                .unwrap();
            assert!(provider.prefix_cache().is_empty(), "no room: nothing kept");
            let reason = report.prefix_cache.reason.unwrap();
            assert!(reason.starts_with("not kept: admission"), "{reason}");
        }
        {
            let provider = causal_provider();
            let _budget = MemoryOverride::set(required + snapshot);
            provider.generate(&req, &mut |_| {}).unwrap();
            assert_eq!(provider.prefix_cache().len(), 1);
            assert!(provider.prefix_cache_resident_bytes() <= snapshot);
        }
        // With an entry held, a request short by exactly the entry reclaims it.
        let provider = causal_provider();
        run(
            &provider,
            &chat(vec![Message::user("t3 t9 t4")], Speculative::Off),
        );
        let held = provider.prefix_cache_resident_bytes();
        assert!(held > 0);
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
            assert_eq!(provider.prefix_cache_stats().evicted - before, 1);
            assert!(
                provider.prefix_cache().is_empty(),
                "no room left for its own copy"
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::prefix_cache::a_request_keeps_its_snapshot_only_when_admission_holds_it");
    }
}

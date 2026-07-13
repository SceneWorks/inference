//! Drives the registered candle `SnapshotPreparer` through the `core-llm` snapshot-preparer
//! conformance helper (story 7662) — the convert+quantize peer of the text-provider conformance in
//! `conformance.rs`. A passing run here de-provisionalizes the `prepare_snapshot` seam across a second
//! backend.
//!
//! The synthetic tests need no model weights (they run in CI): a tiny snapshot is prepared dense
//! (passthrough) and re-quantized (Q8 with 32-aligned dims, Q4 with 256-aligned dims), and each
//! prepared snapshot is loaded back through `load_for_model` / `LlamaProvider` to prove it is a
//! genuinely quantized, loadable model. Gated tests run the same `check_snapshot_preparer` against
//! real HF snapshots and a GGUF.
//!
//! ```text
//! CANDLE_LLM_TEST_MODEL=/path/SmolLM2     # HF dense + Q8
//! CANDLE_LLM_QWEN3_MODEL=/path/Qwen3-0.6B # HF Q4 (256-aligned)
//! CANDLE_LLM_GGUF=/path/Model.gguf        # GGUF dense + Q8
//!   cargo test --features cuda --test prepare -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use candle_core::{DType, Device, Tensor};

use candle_llm::primitives::sampler::{SplitMix64, TokenRng};
use candle_llm::LlamaProvider;
use core_llm::{
    detect_format, load_for_model, prepare_snapshot, LoadSpec, Message, ModelFormat, PrepareReport,
    PrepareSpec, Quantize, TextLlmRequest,
};
use core_llm_testkit::{check_snapshot_preparer, SnapshotPreparerProfile};

const VOCAB: usize = 32;
static SEQ: AtomicU32 = AtomicU32::new(0);

fn unique_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "candle-llm-prepare-{}-{tag}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn randn(shape: (usize, usize), rng: &mut SplitMix64) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::ones((d,), DType::F32, &Device::Cpu).unwrap()
}

/// A WordLevel `tokenizer.json` whose vocab is `t0..t{VOCAB-1}`, so every model token decodes to a
/// distinct piece.
fn tokenizer_json() -> String {
    let entries: Vec<String> = (0..VOCAB).map(|i| format!("\"t{i}\": {i}")).collect();
    format!(
        r#"{{ "version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null, "decoder": null,
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
        entries.join(", ")
    )
}

/// Write a tiny synthetic HF snapshot whose projection in-dims all equal `hidden` (and `inter`), so a
/// `hidden`/`inter` that is block-aligned (32 for Q8, 256 for Q4) can be re-quantized.
fn write_synthetic(tag: &str, hidden: usize, inter: usize) -> PathBuf {
    let dir = unique_dir(tag);
    let config = format!(
        r#"{{ "hidden_size": {hidden}, "intermediate_size": {inter}, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": {VOCAB},
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 999 }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();

    let head_dim = hidden / 2;
    let (qd, kvd) = (head_dim * 2, head_dim); // 2 query heads, 1 kv head
    let mut rng = SplitMix64::new(0xBEEF);
    let mut arrays: HashMap<String, Tensor> = HashMap::new();
    arrays.insert(
        "model.embed_tokens.weight".into(),
        randn((VOCAB, hidden), &mut rng),
    );
    arrays.insert("model.norm.weight".into(), ones(hidden));
    arrays.insert("lm_head.weight".into(), randn((VOCAB, hidden), &mut rng));
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        arrays.insert(p("input_layernorm.weight"), ones(hidden));
        arrays.insert(p("post_attention_layernorm.weight"), ones(hidden));
        arrays.insert(p("self_attn.q_proj.weight"), randn((qd, hidden), &mut rng));
        arrays.insert(p("self_attn.k_proj.weight"), randn((kvd, hidden), &mut rng));
        arrays.insert(p("self_attn.v_proj.weight"), randn((kvd, hidden), &mut rng));
        arrays.insert(p("self_attn.o_proj.weight"), randn((hidden, qd), &mut rng));
        arrays.insert(p("mlp.gate_proj.weight"), randn((inter, hidden), &mut rng));
        arrays.insert(p("mlp.up_proj.weight"), randn((inter, hidden), &mut rng));
        arrays.insert(p("mlp.down_proj.weight"), randn((hidden, inter), &mut rng));
    }
    candle_core::safetensors::save(&arrays, dir.join("model.safetensors")).unwrap();
    dir
}

/// A dense source is detected as safetensors and prepared as a passthrough (returned as-is, nothing
/// written), and the returned snapshot loads and generates.
#[test]
fn synthetic_dense_passthrough_and_loads() {
    let src = write_synthetic("dense", 8, 16);
    assert_eq!(detect_format(&src).unwrap(), ModelFormat::Safetensors);

    let out = unique_dir("dense-out");
    let report = prepare_snapshot(&PrepareSpec::dense(&src, &out)).unwrap();
    assert!(
        report.passthrough,
        "dense already-loadable source is a passthrough"
    );
    assert_eq!(report.quantized, None);
    assert_eq!(report.out_dir, src, "passthrough returns the source dir");
    assert!(report.num_tensors > 0);

    // The full contract helper: prepare -> report self-consistency -> load_for_model, plus the
    // unknown-source Unsupported path.
    check_snapshot_preparer(&SnapshotPreparerProfile {
        source: src.clone(),
        out_dir: unique_dir("dense-check"),
        quantize: None,
    })
    .unwrap();

    let _ = std::fs::remove_dir_all(&src);
}

/// A Q8 prepare re-quantizes the projections, stamps a `quantization` block, and the prepared
/// snapshot loads back as a genuinely quantized model.
#[test]
fn synthetic_q8_writes_quantized_snapshot() {
    quant_round_trip("q8", 32, 32, Quantize::Q8);
}

/// Q4_K (block size 256) needs 256-aligned projection in-dims.
#[test]
fn synthetic_q4_writes_quantized_snapshot() {
    quant_round_trip("q4", 256, 256, Quantize::Q4);
}

fn quant_round_trip(tag: &str, hidden: usize, inter: usize, quant: Quantize) {
    let src = write_synthetic(tag, hidden, inter);
    let out = unique_dir(&format!("{tag}-out"));

    let report = prepare_snapshot(&PrepareSpec::quantized(&src, &out, quant)).unwrap();
    assert!(
        !report.passthrough,
        "a quantized prepare writes a fresh snapshot"
    );
    assert_eq!(report.quantized, Some(quant));
    assert_eq!(report.out_dir, out);
    assert!(out.join("model.safetensors").is_file());
    assert!(out.join("tokenizer.json").is_file());

    // The written config carries the quantization block.
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    let bits = if matches!(quant, Quantize::Q4) { 4 } else { 8 };
    assert_eq!(cfg["quantization"]["bits"], serde_json::json!(bits));

    // Loading the prepared snapshot dense (no load-time quantize) yields a quantized model, because
    // the loader honors the persisted block — and it generates.
    let provider =
        LlamaProvider::load(&LoadSpec::dense(out.to_str().unwrap().to_string())).unwrap();
    assert!(
        provider.is_quantized(),
        "{tag}: persisted block must load quantized"
    );

    check_snapshot_preparer(&SnapshotPreparerProfile {
        source: src.clone(),
        out_dir: unique_dir(&format!("{tag}-check")),
        quantize: Some(quant),
    })
    .unwrap();

    let _ = std::fs::remove_dir_all(&src);
    let _ = std::fs::remove_dir_all(&out);
}

/// A multimodal snapshot (a `vision_config` block) is declined by the text preparer, so
/// `prepare_snapshot` reports no backend (Unsupported) rather than mis-preparing it.
#[test]
fn vlm_source_is_declined() {
    let dir = unique_dir("vlm");
    std::fs::write(
        dir.join("config.json"),
        r#"{"architectures":["LlavaForConditionalGeneration"],"model_type":"llava",
            "text_config":{"architectures":["LlamaForCausalLM"]},"vision_config":{"hidden_size":16}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    std::fs::write(dir.join("model.safetensors"), b"\x00").unwrap();

    match prepare_snapshot(&PrepareSpec::dense(&dir, unique_dir("vlm-out"))) {
        Err(core_llm::Error::Unsupported(m)) => {
            assert!(m.contains("no linked backend can prepare"), "{m}")
        }
        other => panic!("expected Unsupported for a VLM source, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// --- gated real-model conformance: prepare a real snapshot, run the helper, and generate ---

fn real_check(source: PathBuf, quant: Option<Quantize>, tag: &str) {
    let out = unique_dir(&format!("real-{tag}"));
    let report: PrepareReport = prepare_snapshot(&PrepareSpec {
        source: source.clone(),
        out_dir: out.clone(),
        quantize: quant,
    })
    .unwrap_or_else(|e| panic!("{tag}: prepare failed: {e}"));
    assert_eq!(report.quantized, quant);

    check_snapshot_preparer(&SnapshotPreparerProfile {
        source,
        out_dir: unique_dir(&format!("real-{tag}-check")),
        quantize: quant,
    })
    .unwrap_or_else(|e| panic!("{tag}: {e}"));

    // The prepared snapshot generates coherently (acceptance: candle loader generates).
    let llm = load_for_model(&LoadSpec::dense(
        report.out_dir.to_string_lossy().to_string(),
    ))
    .unwrap_or_else(|e| panic!("{tag}: load prepared snapshot: {e}"));
    let req = TextLlmRequest::new(vec![Message::user("The capital of France is")], 8);
    let text = llm
        .complete(&req)
        .unwrap_or_else(|e| panic!("{tag}: generate: {e}"))
        .text;
    assert!(
        !text.is_empty(),
        "{tag}: prepared snapshot produced no text"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
#[ignore = "needs a real HF snapshot via CANDLE_LLM_TEST_MODEL"]
fn real_hf_dense() {
    real_check(
        std::env::var("CANDLE_LLM_TEST_MODEL")
            .expect("set CANDLE_LLM_TEST_MODEL")
            .into(),
        None,
        "hf-dense",
    );
}

/// Q8_0 (block size 32) applies broadly — SmolLM2's dims suffice.
#[test]
#[ignore = "needs a real HF snapshot via CANDLE_LLM_TEST_MODEL (Q8)"]
fn real_hf_q8() {
    real_check(
        std::env::var("CANDLE_LLM_TEST_MODEL")
            .expect("set CANDLE_LLM_TEST_MODEL")
            .into(),
        Some(Quantize::Q8),
        "hf-q8",
    );
}

/// Q4_K (block size 256) needs 256-aligned dims — Qwen3 (hidden 1024) qualifies, SmolLM2 does not.
#[test]
#[ignore = "needs a Qwen3 snapshot via CANDLE_LLM_QWEN3_MODEL (Q4; dims must be 256-aligned)"]
fn real_hf_q4_qwen3() {
    real_check(
        std::env::var("CANDLE_LLM_QWEN3_MODEL")
            .expect("set CANDLE_LLM_QWEN3_MODEL")
            .into(),
        Some(Quantize::Q4),
        "hf-q4",
    );
}

#[test]
#[ignore = "needs a GGUF via CANDLE_LLM_GGUF"]
fn real_gguf_dense() {
    real_check(
        std::env::var("CANDLE_LLM_GGUF")
            .expect("set CANDLE_LLM_GGUF")
            .into(),
        None,
        "gguf-dense",
    );
}

/// Convert a GGUF and bake Q8 in one step (block size 32 — broadly applicable).
#[test]
#[ignore = "needs a GGUF via CANDLE_LLM_GGUF (Q8)"]
fn real_gguf_q8() {
    real_check(
        std::env::var("CANDLE_LLM_GGUF")
            .expect("set CANDLE_LLM_GGUF")
            .into(),
        Some(Quantize::Q8),
        "gguf-q8",
    );
}

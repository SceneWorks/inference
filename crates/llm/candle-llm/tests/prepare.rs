//! Drives the registered candle `SnapshotPreparer` through the `core-llm` snapshot-preparer
//! conformance helper (story 7662) — the convert+quantize peer of the text-provider conformance in
//! `conformance.rs`. A passing run here de-provisionalizes the `prepare_snapshot` seam across a second
//! backend.
//!
//! The synthetic tests need no model weights (they run in CI): a tiny snapshot is prepared dense
//! (passthrough) and re-quantized (Q8 and Q4 — Q4_K on 256-aligned input dims, Q4_0 on 32- but not
//! 256-aligned ones), and each prepared snapshot is loaded back through `load_for_model` /
//! `LlamaProvider` to prove it is a genuinely quantized, loadable model. Gated tests run the same
//! `check_snapshot_preparer` against real HF snapshots and a GGUF.
//!
//! The stored-block tests load through `LlamaProvider::load`, which opens the selected device:
//! built with `--features metal` or `--features cuda` they rebuild the stored blocks on that
//! accelerator (their fixtures are bf16, so a load-time quantization there sees the same weights).
//!
//! ```text
//! CANDLE_LLM_TEST_MODEL=/path/SmolLM2     # HF dense + Q8
//! CANDLE_LLM_QWEN3_MODEL=/path/Qwen3-0.6B # HF Q4 (Q4_K throughout: 256-aligned)
//! CANDLE_LLM_GGUF=/path/Model.gguf        # GGUF dense + Q8
//!   cargo test --features cuda --test prepare -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use candle_llm::primitives::sampler::{SplitMix64, TokenRng};
use candle_llm::{load_for_model, prepare_snapshot, LlamaProvider};
use core_llm::{
    detect_format, LoadSpec, Message, ModelFormat, PrepareReport, PrepareSpec, Quantize,
    TextLlmRequest,
};
use core_llm_testkit::{check_snapshot_preparer, SnapshotPreparerProfile};

const VOCAB: usize = 32;
mod common;
use common::Fixture;

/// A fresh fixture root that removes itself on `Drop` (sc-17755).
///
/// This replaced a hand-rolled `temp_dir().join(format!("candle-llm-prepare-{pid}-{tag}-{n}"))` whose
/// `remove_dir_all` prelude only made it idempotent *within* one process — every new PID left another
/// tree behind. Callers must bind the returned [`Fixture`] for as long as they read the path.
fn unique_dir(tag: &str) -> Fixture {
    Fixture::new(&format!("candle-llm-prepare-{tag}-"), None)
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

/// Write a tiny synthetic f32 HF snapshot whose projection in-dims all equal `hidden` (and
/// `inter`), so a `hidden`/`inter` that is 32-aligned can be re-quantized (Q8_0; Q4_K where it is
/// also 256-aligned, else Q4_0).
fn write_synthetic(tag: &str, hidden: usize, inter: usize) -> Fixture {
    write_synthetic_as(tag, hidden, inter, DType::F32)
}

/// [`write_synthetic`] with every tensor stored in `dtype`.
fn write_synthetic_as(tag: &str, hidden: usize, inter: usize, dtype: DType) -> Fixture {
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
    for t in arrays.values_mut() {
        *t = t.to_dtype(dtype).unwrap();
    }
    candle_core::safetensors::save(&arrays, dir.join("model.safetensors")).unwrap();
    dir
}

/// `into_causal_lm` hands over the very decoder `causal_lm` borrows (identical logits), and the
/// owned decoder is `Send` — what an engine that owns it across threads (YuE, sc-19380) needs.
#[test]
fn into_causal_lm_moves_the_loaded_decoder_out() {
    fn assert_send<T: Send>(_: &T) {}
    let src = write_synthetic("into-causal", 8, 16);
    let provider = LlamaProvider::load(&LoadSpec::dense(src.to_string_lossy())).unwrap();
    let borrowed = logits(&provider);
    let model = provider.into_causal_lm().expect("a llama-family decoder");
    assert_send(&model);
    let ids = Tensor::from_vec(vec![1u32, 5, 9, 2, 7], (1, 5), model.device()).unwrap();
    let mut cache = model.new_cache();
    let owned = model
        .decode_logits(&ids, &mut cache, 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(owned, borrowed);
}

/// A dense source is detected as safetensors and prepared as a passthrough (returned as-is, nothing
/// written), and the returned snapshot loads and generates.
#[test]
fn synthetic_dense_passthrough_and_loads() {
    let src = write_synthetic("dense", 8, 16);
    assert_eq!(detect_format(&src).unwrap(), ModelFormat::Safetensors);

    let out = unique_dir("dense-out");
    let report =
        prepare_snapshot(&PrepareSpec::dense(src.to_path_buf(), out.to_path_buf())).unwrap();
    assert!(
        report.passthrough,
        "dense already-loadable source is a passthrough"
    );
    assert_eq!(report.quantized, None);
    assert_eq!(report.out_dir, *src, "passthrough returns the source dir");
    assert!(report.num_tensors > 0);

    // The full contract helper: prepare -> report self-consistency -> load_for_model, plus the
    // unknown-source Unsupported path.
    let check_out = unique_dir("dense-check");
    check_snapshot_preparer(
        &SnapshotPreparerProfile {
            source: src.to_path_buf(),
            out_dir: check_out.to_path_buf(),
            quantize: None,
        },
        &candle_llm::snapshot_preparer_registry().unwrap(),
        &candle_llm::text_registry().unwrap(),
    )
    .unwrap();
}

/// A Q8 prepare re-quantizes the projections, stamps a `quantization` block, and the prepared
/// snapshot loads back as a genuinely quantized model.
#[test]
fn synthetic_q8_writes_quantized_snapshot() {
    quant_round_trip("q8", 32, 32, Quantize::Q8);
}

/// Q4 on 256-aligned projection in-dims: Q4_K (block size 256) throughout. A 32- but not
/// 256-aligned in-dim falls back to Q4_0 (see `prepared_tiers_store_ggml_blocks_and_load_identically`).
#[test]
fn synthetic_q4_writes_quantized_snapshot() {
    quant_round_trip("q4", 256, 256, Quantize::Q4);
}

fn quant_round_trip(tag: &str, hidden: usize, inter: usize, quant: Quantize) {
    let src = write_synthetic(tag, hidden, inter);
    let out = unique_dir(&format!("{tag}-out"));

    let report = prepare_snapshot(&PrepareSpec::quantized(
        src.to_path_buf(),
        out.to_path_buf(),
        quant,
    ))
    .unwrap();
    assert!(
        !report.passthrough,
        "a quantized prepare writes a fresh snapshot"
    );
    assert_eq!(report.quantized, Some(quant));
    assert_eq!(report.out_dir, *out);
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

    let check_out = unique_dir(&format!("{tag}-check"));
    check_snapshot_preparer(
        &SnapshotPreparerProfile {
            source: src.to_path_buf(),
            out_dir: check_out.to_path_buf(),
            quantize: Some(quant),
        },
        &candle_llm::snapshot_preparer_registry().unwrap(),
        &candle_llm::text_registry().unwrap(),
    )
    .unwrap();
}

/// The safetensors header of one file: `name -> (dtype, shape)`.
fn header(path: &std::path::Path) -> HashMap<String, (String, Vec<usize>)> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let v: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    v.as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.as_str() != "__metadata__")
        .map(|(k, t)| {
            let shape = t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            (k.clone(), (t["dtype"].as_str().unwrap().to_string(), shape))
        })
        .collect()
}

/// Last-position logits of a loaded llama-family provider over a fixed prompt, on the host.
fn logits(provider: &LlamaProvider) -> Vec<f32> {
    let model = provider.causal_lm().expect("a llama-family decoder");
    let ids = Tensor::from_vec(vec![1u32, 5, 9, 2, 7], (1, 5), model.device()).unwrap();
    let mut cache = model.new_cache();
    model
        .decode_logits(&ids, &mut cache, 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// sc-19375: a Q4 / Q8 tier stores its layer projections **already quantized** — each a `U8`
/// GGML block tensor `[rows, blocks, block_bytes]` of the tier's block type — so the tier is a
/// fraction of the dense source, and loading it (no load-time request: the persisted block drives
/// it) builds exactly the model a load-time quantization of the dense source builds: identical
/// logits. The Q4 case uses a 288-wide MLP, so `down_proj` exercises the Q4_0 fallback.
///
/// The loads run on the selected device, so a `--features metal` / `--features cuda` build
/// rebuilds the stored blocks on that accelerator; the bf16 fixture keeps a load-time quantization
/// there (bf16 compute) quantizing exactly the weights the preparer did.
#[test]
fn prepared_tiers_store_ggml_blocks_and_load_identically() {
    for (tag, hidden, inter, quant, blocks) in [
        (
            "blocks-q4",
            256usize,
            288usize,
            Quantize::Q4,
            [144usize, 18],
        ),
        ("blocks-q8", 64, 96, Quantize::Q8, [34, 34]),
    ] {
        let src = write_synthetic_as(tag, hidden, inter, DType::BF16);
        let out = unique_dir(&format!("{tag}-out"));
        prepare_snapshot(&PrepareSpec::quantized(
            src.to_path_buf(),
            out.to_path_buf(),
            quant,
        ))
        .unwrap();

        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["quantization"]["storage"], serde_json::json!("ggml"));

        // Every layer projection is stored as GGML blocks of the tier's type; the rest stay dense.
        let tensors = header(&out.join("model.safetensors"));
        let source = header(&src.join("model.safetensors"));
        let mut projections = 0;
        for (name, (dtype, shape)) in &tensors {
            if name.ends_with("_proj.weight") {
                projections += 1;
                let (_, src_shape) = &source[name];
                let (rows, cols) = (src_shape[0], src_shape[1]);
                let block_bytes = if name.ends_with("down_proj.weight") {
                    blocks[1]
                } else {
                    blocks[0]
                };
                let block_len = if block_bytes == 144 { 256 } else { 32 };
                assert_eq!(dtype, "U8", "{tag}: {name} must be stored quantized");
                assert_eq!(
                    shape,
                    &vec![rows, cols / block_len, block_bytes],
                    "{tag}: {name} block layout"
                );
            } else {
                assert_eq!(dtype, &source[name].0, "{tag}: {name} stays dense");
            }
        }
        assert_eq!(projections, 14, "{tag}: 2 layers x 7 projections");

        // The tier holds only its own bytes: well under the dense bf16 source (Q4 is 0.28 of a
        // bf16 projection, Q8 0.53; the dense embeddings and head add a little).
        let tier = std::fs::metadata(out.join("model.safetensors"))
            .unwrap()
            .len();
        let dense = std::fs::metadata(src.join("model.safetensors"))
            .unwrap()
            .len();
        let bound = if matches!(quant, Quantize::Q4) {
            0.35
        } else {
            0.6
        };
        assert!(
            (tier as f64) < bound * dense as f64,
            "{tag}: tier {tier} B is not quantized storage (dense source {dense} B)"
        );

        let prepared =
            LlamaProvider::load(&LoadSpec::dense(out.to_str().unwrap().to_string())).unwrap();
        assert!(prepared.is_quantized(), "{tag}: loads quantized");
        let mut at_load = LoadSpec::dense(src.to_str().unwrap().to_string());
        at_load.quantize = Some(quant);
        let at_load = LlamaProvider::load(&at_load).unwrap();
        assert_eq!(
            logits(&prepared),
            logits(&at_load),
            "{tag}: stored blocks must load to the load-time-quantized model exactly"
        );
        // The weight reader keeps the stored blocks on the host (their `QTensor`s are built on
        // the device from there) and puts every dense tensor on the loading device.
        let device = candle_llm::device::select_device().unwrap();
        let weights = candle_llm::primitives::Weights::from_dir(&*out, &device).unwrap();
        for (name, (dtype, _)) in &tensors {
            let t = weights.get(name).unwrap();
            match dtype.as_str() {
                "U8" => assert!(t.device().is_cpu(), "{tag}: {name} read onto the host"),
                _ => assert!(
                    t.device().same_device(&device),
                    "{tag}: {name} on the device"
                ),
            }
        }

        // Stored blocks are never re-quantized to another tier.
        let mut other = LoadSpec::dense(out.to_str().unwrap().to_string());
        other.quantize = Some(if matches!(quant, Quantize::Q4) {
            Quantize::Q8
        } else {
            Quantize::Q4
        });
        assert!(
            LlamaProvider::load(&other).is_err(),
            "{tag}: re-quantizing stored blocks must be refused"
        );
        // Nor re-quantized to NVFP4: the model gate names the stored blocks before any device
        // or weight is touched.
        let mut nvfp4 = LoadSpec::dense(out.to_str().unwrap().to_string());
        nvfp4.quantize = Some(Quantize::Nvfp4);
        match LlamaProvider::load(&nvfp4) {
            Err(e) => assert!(
                e.to_string().contains("stored as GGML blocks"),
                "{tag}: {e}"
            ),
            Ok(_) => panic!("{tag}: NVFP4 over stored blocks must be refused"),
        }

        // Admission prices the stored blocks' device copy exactly as a load-time quantization of
        // the dense source (Q4_0 and Q4_K cost the same per weight, padding included) — and holds
        // the stored source bytes on the host, never the device.
        let estimate = |dir: &std::path::Path, quantize| {
            let mut spec = LoadSpec::dense(dir.to_str().unwrap().to_string());
            spec.quantize = quantize;
            LlamaProvider::load_memory_estimate(&spec, true).unwrap()
        };
        let stored = estimate(&out, None);
        assert_eq!(
            stored.quantized_copy_bytes,
            estimate(&src, Some(quant)).quantized_copy_bytes,
            "{tag}: stored blocks priced as the load-time copy"
        );
        let block_bytes: u64 = tensors
            .values()
            .filter(|(dtype, _)| dtype == "U8")
            .map(|(_, shape)| shape.iter().product::<usize>() as u64)
            .sum();
        assert_eq!(
            stored.source_bytes,
            stored.payload_bytes - block_bytes,
            "{tag}: the stored blocks are read onto the host, not the device"
        );
        assert_eq!(
            stored.host_required_bytes,
            core_llm::checkpoint_staging_bytes(&out).unwrap() + block_bytes,
            "{tag}: the host holds the stored blocks beside one shard of staging"
        );

        // A stored-block tier whose config lost its `quantization` block is refused, never
        // read as dense weights.
        let stripped = unique_dir(&format!("{tag}-stripped"));
        let mut bare = cfg.clone();
        bare.as_object_mut().unwrap().remove("quantization");
        std::fs::write(stripped.join("config.json"), bare.to_string()).unwrap();
        for file in ["tokenizer.json", "model.safetensors"] {
            std::fs::copy(out.join(file), stripped.join(file)).unwrap();
        }
        match LlamaProvider::load(&LoadSpec::dense(stripped.to_str().unwrap().to_string())) {
            Err(e) => assert!(
                e.to_string().contains("no quantization block"),
                "{tag}: {e}"
            ),
            Ok(_) => panic!("{tag}: stored blocks without a quantization block must be refused"),
        }
    }
}

/// Backward compatibility: a snapshot prepared before sc-19375 — dense weights carrying the
/// rounding plus a bare `quantization` block — still loads quantized, to the same logits as a
/// load-time quantization of those weights.
#[test]
fn dense_rounded_prepared_snapshot_still_loads_quantized() {
    let src = write_synthetic("legacy", 64, 96);
    let cfg_path = src.join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    cfg["quantization"] = serde_json::json!({ "bits": 8 });
    let legacy = unique_dir("legacy-snap");
    std::fs::write(legacy.join("config.json"), cfg.to_string()).unwrap();
    std::fs::copy(src.join("tokenizer.json"), legacy.join("tokenizer.json")).unwrap();
    std::fs::copy(
        src.join("model.safetensors"),
        legacy.join("model.safetensors"),
    )
    .unwrap();

    let loaded =
        LlamaProvider::load(&LoadSpec::dense(legacy.to_str().unwrap().to_string())).unwrap();
    assert!(loaded.is_quantized());
    let mut at_load = LoadSpec::dense(src.to_str().unwrap().to_string());
    at_load.quantize = Some(Quantize::Q8);
    let at_load = LlamaProvider::load(&at_load).unwrap();
    assert_eq!(logits(&loaded), logits(&at_load));
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

    let vlm_out = unique_dir("vlm-out");
    match prepare_snapshot(&PrepareSpec::dense(
        dir.to_path_buf(),
        vlm_out.to_path_buf(),
    )) {
        Err(core_llm::Error::Unsupported(m)) => {
            assert!(m.contains("no linked backend can prepare"), "{m}")
        }
        other => panic!("expected Unsupported for a VLM source, got {other:?}"),
    }
}

// --- gated real-model conformance: prepare a real snapshot, run the helper, and generate ---

fn real_check(source: PathBuf, quant: Option<Quantize>, tag: &str) {
    let out = unique_dir(&format!("real-{tag}"));
    let report: PrepareReport = prepare_snapshot(&PrepareSpec {
        source: source.clone(),
        out_dir: out.to_path_buf(),
        quantize: quant,
    })
    .unwrap_or_else(|e| panic!("{tag}: prepare failed: {e}"));
    assert_eq!(report.quantized, quant);

    let real_check_out = unique_dir(&format!("real-{tag}-check"));
    check_snapshot_preparer(
        &SnapshotPreparerProfile {
            source,
            out_dir: real_check_out.to_path_buf(),
            quantize: quant,
        },
        &candle_llm::snapshot_preparer_registry().unwrap(),
        &candle_llm::text_registry().unwrap(),
    )
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

/// Q4 on a real snapshot. Any 32-aligned model prepares Q4 (Q4_K where an input dim is 256-aligned,
/// Q4_0 where it is only 32-aligned — SmolLM2's hidden 576 included); Qwen3 (hidden 1024) keeps
/// this case on Q4_K throughout.
#[test]
#[ignore = "needs a real HF snapshot via CANDLE_LLM_QWEN3_MODEL (Q4)"]
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

//! Runs the registered `LlamaProvider` through the `core-llm` conformance suite (story 7201) — the
//! check that, passed by this *second, independent* backend, de-provisionalizes the contract (7237).
//!
//! Builds a tiny synthetic snapshot (no model weights needed, runs in CI) whose vocabulary is fully
//! covered by the tokenizer, so the suite's seed-determinism check sees genuinely distinct text for
//! distinct seeds. Separate gated tests run the same suite against real models — a Llama snapshot
//! (`CANDLE_LLM_TEST_MODEL`), a **Qwen3** snapshot (`CANDLE_LLM_QWEN3_MODEL`, exercising per-head q/k
//! RMSNorm + head_dim 128), **quantize-on-load** (Q8_0 on the Llama snapshot; Q4_K on Qwen3, whose
//! dims are 256-aligned for the Q4_K block size), and a **GGUF** checkpoint (`CANDLE_LLM_GGUF`). Story
//! 7264 broadens this coverage beyond the SmolLM2/Llama baseline validated in 7237.
//!
//! ```text
//! # On CUDA (the Windows target); drop `--features cuda` for the CPU path.
//! CANDLE_LLM_TEST_MODEL=/path/Llama-snapshot \
//! CANDLE_LLM_QWEN3_MODEL=/path/Qwen3-0.6B \
//! CANDLE_LLM_GGUF=/path/Model-Q4_K_M.gguf \
//!   cargo test --features cuda --test conformance -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use candle_llm::primitives::sampler::{SplitMix64, TokenRng};
use candle_llm::provider::PROVIDER_ID;
use candle_llm::LlamaProvider;
use core_llm::{load_for_model, LoadSpec, Message, Quantize, TextLlmRequest};
use core_llm_testkit::{textllm_conformance, TextLlmProfile};

const VOCAB: usize = 32;

fn randn(shape: (usize, usize), rng: &mut SplitMix64) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::ones((d,), DType::F32, &Device::Cpu).unwrap()
}

/// A tokenizer.json whose vocab is `t0..t{VOCAB-1}` (whitespace WordLevel), so every model token id
/// decodes to a distinct, non-empty piece — distinct seeds therefore yield distinct text.
fn tokenizer_json() -> String {
    let entries: Vec<String> = (0..VOCAB).map(|i| format!("\"t{i}\": {i}")).collect();
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

fn write_snapshot() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("candle-llm-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // eos_token_id outside the vocab so generation always runs to the token budget.
    let config = format!(
        r#"{{
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": {VOCAB},
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 999
        }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();

    let (h, v, inter, qd, kvd) = (8usize, VOCAB, 16usize, 8usize, 4usize);
    let mut rng = SplitMix64::new(0xBEEF);
    let mut arrays: HashMap<String, Tensor> = HashMap::new();
    arrays.insert("model.embed_tokens.weight".into(), randn((v, h), &mut rng));
    arrays.insert("model.norm.weight".into(), ones(h));
    arrays.insert("lm_head.weight".into(), randn((v, h), &mut rng));
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        arrays.insert(p("input_layernorm.weight"), ones(h));
        arrays.insert(p("post_attention_layernorm.weight"), ones(h));
        arrays.insert(p("self_attn.q_proj.weight"), randn((qd, h), &mut rng));
        arrays.insert(p("self_attn.k_proj.weight"), randn((kvd, h), &mut rng));
        arrays.insert(p("self_attn.v_proj.weight"), randn((kvd, h), &mut rng));
        arrays.insert(p("self_attn.o_proj.weight"), randn((h, qd), &mut rng));
        arrays.insert(p("mlp.gate_proj.weight"), randn((inter, h), &mut rng));
        arrays.insert(p("mlp.up_proj.weight"), randn((inter, h), &mut rng));
        arrays.insert(p("mlp.down_proj.weight"), randn((h, inter), &mut rng));
    }
    candle_core::safetensors::save(&arrays, dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn llama_provider_passes_core_llm_conformance() {
    let dir = write_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());

    // The closure loads a fresh provider; the suite drives it through every contract guarantee and
    // panics with an aggregated message on any failure.
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load synthetic provider")),
        &TextLlmProfile::cheap(),
    );

    // Sanity: the provider id the suite checked is the registered one.
    assert_eq!(PROVIDER_ID, "candle-llama");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs a real Llama snapshot via CANDLE_LLM_TEST_MODEL"]
fn real_model_passes_core_llm_conformance() {
    let dir = std::env::var("CANDLE_LLM_TEST_MODEL").expect("set CANDLE_LLM_TEST_MODEL");
    let spec = LoadSpec::dense(dir);
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load real provider")),
        &TextLlmProfile::cheap(),
    );
}

/// The full conformance suite on a **Qwen3** snapshot (per-head q/k RMSNorm, head_dim 128, tied
/// embeddings) — proves the BYO architecture dispatch holds up under the contract, not just Llama.
#[test]
#[ignore = "needs a Qwen3 snapshot via CANDLE_LLM_QWEN3_MODEL"]
fn qwen3_passes_core_llm_conformance() {
    let dir = std::env::var("CANDLE_LLM_QWEN3_MODEL").expect("set CANDLE_LLM_QWEN3_MODEL");
    let spec = LoadSpec::dense(dir);
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load qwen3 provider")),
        &TextLlmProfile::cheap(),
    );
}

/// Run the full conformance suite against a quantize-on-load model. Quantized providers must satisfy
/// every contract guarantee (streaming, cancel, seed-determinism, …), not merely load.
fn run_quantized_conformance(env_var: &str, quant: Quantize) {
    let Ok(dir) = std::env::var(env_var) else {
        eprintln!("skip: set {env_var}");
        return;
    };
    let spec = LoadSpec {
        source: dir,
        quantize: Some(quant),
    };
    textllm_conformance(
        || {
            let p = LlamaProvider::load(&spec).expect("load quantized provider");
            assert!(
                p.is_quantized(),
                "{quant:?}: provider must report quantized"
            );
            Box::new(p)
        },
        &TextLlmProfile::cheap(),
    );
}

/// Conformance on a **Q8_0 quantize-on-load** (block size 32 — broadly applicable; the Llama snapshot
/// suffices). Run against `CANDLE_LLM_TEST_MODEL`.
#[test]
#[ignore = "needs a Llama snapshot via CANDLE_LLM_TEST_MODEL (Q8 quantize-on-load)"]
fn quantized_q8_passes_core_llm_conformance() {
    run_quantized_conformance("CANDLE_LLM_TEST_MODEL", Quantize::Q8);
}

/// Conformance on a **Q4_K quantize-on-load**. Q4_K's block size is 256, so the projection `in`-dims
/// must be multiples of 256 — true of Qwen3 (hidden 1024) but not of SmolLM2 (hidden 576). Run
/// against `CANDLE_LLM_QWEN3_MODEL`, whose dims are 256-aligned.
#[test]
#[ignore = "needs a Qwen3 snapshot via CANDLE_LLM_QWEN3_MODEL (Q4 quantize-on-load; dims must be 256-aligned)"]
fn quantized_q4_passes_core_llm_conformance() {
    run_quantized_conformance("CANDLE_LLM_QWEN3_MODEL", Quantize::Q4);
}

/// The full conformance suite on a **GGUF** checkpoint loaded directly (story 7254) — proves the GGUF
/// load path produces a contract-conformant provider end-to-end (tokenizer from sibling/metadata,
/// stop tokens, chat template, streaming, …).
#[test]
#[ignore = "needs a GGUF via CANDLE_LLM_GGUF"]
fn gguf_passes_core_llm_conformance() {
    let gguf = std::env::var("CANDLE_LLM_GGUF").expect("set CANDLE_LLM_GGUF");
    let spec = LoadSpec::dense(gguf);
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load gguf provider")),
        &TextLlmProfile::cheap(),
    );
}

// --- story 7406: model-first resolution (core_llm::load_for_model) over the weightless probe ---

/// A `config.json`-only snapshot (no safetensors, no tokenizer) used to prove the `can_load` probe
/// is weightless and architecture-aware.
fn write_config_only(name: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("candle-llm-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    dir
}

/// Write a minimal, **zero-tensor** GGUF (V3) carrying a single `general.architecture` metadata
/// string — enough for the header-only `can_load` probe, with no tensor data at all, so a probe that
/// resolves it provably read no weights. Returns the `.gguf` file path. (Format per the GGUF spec:
/// little-endian magic `GGUF`, u32 version, u64 tensor_count, u64 metadata_kv_count, then KV pairs;
/// a string is a u64 length prefix + UTF-8 bytes, and value-type `8` is String.)
fn write_minimal_gguf(name: &str, arch: &str) -> PathBuf {
    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
    push_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes()); // value type 8 = String
    push_str(&mut buf, arch);

    let dir = std::env::temp_dir().join(format!("candle-llm-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.gguf"));
    std::fs::write(&path, &buf).unwrap();
    path
}

#[test]
fn can_load_is_weightless_and_architecture_aware() {
    // A directory with ONLY config.json (no shards): if the probe read weights this would fail. Each
    // of candle's dispatched families is recognized weightlessly — the acceptance "additionally
    // Gemma2/Phi3/GLM4/DeepSeek on candle-llm" at the resolution layer, without real weights.
    for (name, arch, model_type) in [
        ("llama", "LlamaForCausalLM", "llama"),
        ("mistral", "MistralForCausalLM", "mistral"),
        ("qwen3", "Qwen3ForCausalLM", "qwen3"),
        ("qwen2moe", "Qwen2MoeForCausalLM", "qwen2_moe"),
        ("gemma2", "Gemma2ForCausalLM", "gemma2"),
        ("glm4", "Glm4ForCausalLM", "glm4"),
        ("deepseek", "DeepseekV2ForCausalLM", "deepseek_v2"),
        ("phi3", "Phi3ForCausalLM", "phi3"),
    ] {
        let dir = write_config_only(
            &format!("canload-{name}"),
            &format!(r#"{{"architectures":["{arch}"],"model_type":"{model_type}"}}"#),
        );
        let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());
        assert!(
            candle_llm::provider::can_load(&spec),
            "{name}: text provider must claim the snapshot weightlessly"
        );
        assert!(
            !candle_llm::llava::can_load(&spec),
            "{name}: vision provider must decline a text snapshot"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // An unsupported architecture is declined (no panic, no silent default).
    let unknown = write_config_only(
        "canload-unknown",
        r#"{"architectures":["BertModel"],"model_type":"bert"}"#,
    );
    let uspec = LoadSpec::dense(unknown.to_str().unwrap().to_string());
    assert!(!candle_llm::provider::can_load(&uspec));
    let _ = std::fs::remove_dir_all(&unknown);

    // A multimodal snapshot: the text provider declines (a `vision_config` is present even though
    // the nested text arch is llama), the vision provider claims it.
    let vlm = write_config_only(
        "canload-vlm",
        r#"{"architectures":["LlavaForConditionalGeneration"],"model_type":"llava",
            "text_config":{"architectures":["LlamaForCausalLM"],"model_type":"llama"},
            "vision_config":{"hidden_size":16}}"#,
    );
    let vspec = LoadSpec::dense(vlm.to_str().unwrap().to_string());
    assert!(!candle_llm::provider::can_load(&vspec), "text provider must decline a VLM");
    assert!(candle_llm::llava::can_load(&vspec), "vision provider must claim a VLM");
    let _ = std::fs::remove_dir_all(&vlm);

    // A `*.gguf` file: the text provider reads ONLY the header (the fixtures below carry zero tensor
    // data) to confirm `general.architecture`. A supported arch (llama/qwen3) is claimed; the vision
    // provider declines either (story 7420 — replaces the earlier extension-only accept).
    for arch in ["llama", "qwen3"] {
        let path = write_minimal_gguf(&format!("canload-gguf-{arch}"), arch);
        let spec = LoadSpec::dense(path.to_str().unwrap().to_string());
        assert!(
            candle_llm::provider::can_load(&spec),
            "{arch}: text provider must claim a supported-arch GGUF weightlessly"
        );
        assert!(
            !candle_llm::llava::can_load(&spec),
            "{arch}: vision provider must decline a GGUF"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    // An unsupported / non-LLM GGUF arch (here `bert`) is declined — likewise weightlessly, from the
    // header alone — so `load_for_model` returns a clean `Unsupported` instead of routing it here.
    let bert = write_minimal_gguf("canload-gguf-bert", "bert");
    let bspec = LoadSpec::dense(bert.to_str().unwrap().to_string());
    assert!(
        !candle_llm::provider::can_load(&bspec),
        "text provider must decline an unsupported/non-LLM GGUF arch"
    );
    let _ = std::fs::remove_dir_all(bert.parent().unwrap());

    // A `*.gguf` path that doesn't exist (or isn't a parseable GGUF) is declined gracefully — the
    // header probe fails closed rather than claiming a file it can't read.
    assert!(!candle_llm::provider::can_load(&LoadSpec::dense("/no/such/model-Q4_K_M.gguf")));

    // A nonexistent snapshot path is declined gracefully.
    assert!(!candle_llm::provider::can_load(&LoadSpec::dense("/no/such/dir")));
}

#[test]
fn load_for_model_resolves_synthetic_snapshot_without_naming_a_provider() {
    let dir = write_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());

    // No provider id named: the resolver reads config.json, picks the candle text provider via its
    // can_load probe, loads it on CPU, and it generates — the full round-trip in CI.
    let llm = load_for_model(&spec).expect("load_for_model resolves the synthetic snapshot");
    assert_eq!(llm.descriptor().id, PROVIDER_ID);
    assert_eq!(llm.descriptor().backend, "candle");

    let req = TextLlmRequest::new(vec![Message::user("t1 t2 t3")], 4);
    let out = llm.complete(&req).expect("generate");
    assert!(!out.text.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_for_model_unknown_architecture_is_a_typed_error() {
    let dir = write_config_only(
        "lfm-unknown",
        r#"{"architectures":["BertModel"],"model_type":"bert"}"#,
    );
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());
    match load_for_model(&spec) {
        Err(core_llm::Error::Unsupported(m)) => {
            assert!(m.contains("no registered provider can serve"), "{m}");
            assert!(m.contains("bert"), "error should surface the model arch: {m}");
        }
        Err(e) => panic!("expected Unsupported, got error: {e}"),
        Ok(_) => panic!("expected Unsupported, got a loaded provider"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_for_model_unsupported_gguf_is_a_typed_error() {
    // A non-LLM GGUF (here `general.architecture = "bert"`): no provider claims it via its weightless
    // header probe, so model-first resolution returns a typed `Unsupported` — the same outcome as an
    // unsupported safetensors snapshot, not a generic load error from routing it to candle-llama and
    // failing deep in the GGUF reader (story 7420).
    let path = write_minimal_gguf("lfm-gguf-bert", "bert");
    let spec = LoadSpec::dense(path.to_str().unwrap().to_string());
    match load_for_model(&spec) {
        Err(core_llm::Error::Unsupported(m)) => {
            assert!(m.contains("no registered provider can serve"), "{m}");
        }
        Err(e) => panic!("expected Unsupported, got error: {e}"),
        Ok(_) => panic!("expected Unsupported, got a loaded provider"),
    }
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// Gated: a real GGUF (`CANDLE_LLM_GGUF`) is claimed by the weightless header probe, resolves through
/// model-first `load_for_model` (no provider id named), AND the probe is provably weightless —
/// truncating the file at its `tensor_data_offset` (dropping every tensor block) still resolves
/// `can_load`.
#[test]
#[ignore = "needs a GGUF via CANDLE_LLM_GGUF"]
fn gguf_resolves_through_load_for_model_and_probe_is_weightless() {
    use candle_core::quantized::gguf_file::Content;

    let gguf = std::env::var("CANDLE_LLM_GGUF").expect("set CANDLE_LLM_GGUF");
    let spec = LoadSpec::dense(gguf.clone());
    assert!(
        candle_llm::provider::can_load(&spec),
        "a real supported-arch GGUF must be claimed by the header probe"
    );

    // Model-first resolution: no provider id named, the resolver picks candle-llama via can_load.
    let llm = load_for_model(&spec).expect("load_for_model resolves a real GGUF");
    assert_eq!(llm.descriptor().backend, "candle");

    // Weightless: copy only the bytes up to `tensor_data_offset` (the magic + metadata + tensor-info
    // table, with NO tensor blocks) and confirm can_load still resolves — proving the probe read no
    // weights even on a real checkpoint.
    let mut f = std::fs::File::open(&gguf).expect("open gguf");
    let header_len = Content::read(&mut f).expect("read gguf header").tensor_data_offset as usize;
    let mut bytes = std::fs::read(&gguf).expect("read gguf");
    bytes.truncate(header_len);
    let dir = std::env::temp_dir().join(format!("candle-llm-gguf-trunc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let trunc = dir.join("header-only.gguf");
    std::fs::write(&trunc, &bytes).unwrap();
    let tspec = LoadSpec::dense(trunc.to_str().unwrap().to_string());
    assert!(
        candle_llm::provider::can_load(&tspec),
        "header-only (tensor-data-truncated) GGUF must still resolve — the probe is weightless"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

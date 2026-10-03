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

use candle_core::{DType, Device, Tensor};

use candle_llm::load_for_model;
use candle_llm::primitives::sampler::{SplitMix64, TokenRng};
use candle_llm::provider::PROVIDER_ID;
use candle_llm::LlamaProvider;
use core_llm::{LoadSpec, Message, Quantize, TextLlmRequest};
use core_llm_testkit::{textllm_conformance, TextLlmProfile};

const VOCAB: usize = 32;

mod common;
use common::Fixture;

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

fn write_snapshot() -> Fixture {
    write_snapshot_with("", false)
}

/// The tiny synthetic snapshot with `identity` (extra leading `config.json` fields, each followed
/// by a comma) and, for a Qwen3 decoder, unit per-head q/k RMSNorm weights.
fn write_snapshot_with(identity: &str, qk_norm: bool) -> Fixture {
    write_snapshot_with_heads(identity, qk_norm, 4)
}

/// [`write_snapshot_with`] with two query heads and one KV head of `head_dim` channels.
fn write_snapshot_with_heads(identity: &str, qk_norm: bool, head_dim: usize) -> Fixture {
    let fixture = Fixture::new("candle-llm-conformance-", None);
    let dir = &*fixture;
    // eos_token_id outside the vocab so generation always runs to the token budget.
    let config = format!(
        r#"{{ {identity}
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": {head_dim},
            "vocab_size": {VOCAB}, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 999
        }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();

    let (h, v, inter, qd, kvd) = (8usize, VOCAB, 16usize, 2 * head_dim, head_dim);
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
        if qk_norm {
            arrays.insert(p("self_attn.q_norm.weight"), ones(qd / 2));
            arrays.insert(p("self_attn.k_norm.weight"), ones(qd / 2));
        }
    }
    candle_core::safetensors::save(&arrays, dir.join("model.safetensors")).unwrap();
    fixture
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
        projector_source: None,
        quantize: Some(quant),
        cuda_graphs: None,
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
fn write_config_only(name: &str, config: &str) -> Fixture {
    let fixture = Fixture::new(&format!("candle-llm-{name}-"), None);
    std::fs::write(fixture.join("config.json"), config).unwrap();
    fixture
}

/// Write a minimal, **zero-tensor** GGUF (V3) carrying a single `general.architecture` metadata
/// string — enough for the header-only `can_load` probe, with no tensor data at all, so a probe that
/// resolves it provably read no weights. Returns the `.gguf` file path. (Format per the GGUF spec:
/// little-endian magic `GGUF`, u32 version, u64 tensor_count, u64 metadata_kv_count, then KV pairs;
/// a string is a u64 length prefix + UTF-8 bytes, and value-type `8` is String.)
fn write_minimal_gguf(name: &str, arch: &str) -> Fixture {
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

    let fixture = Fixture::new(
        &format!("candle-llm-{name}-"),
        Some(&format!("{name}.gguf")),
    );
    std::fs::write(&*fixture, &buf).unwrap();
    fixture
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
    }

    // An unsupported architecture is declined (no panic, no silent default).
    let unknown = write_config_only(
        "canload-unknown",
        r#"{"architectures":["BertModel"],"model_type":"bert"}"#,
    );
    let uspec = LoadSpec::dense(unknown.to_str().unwrap().to_string());
    assert!(!candle_llm::provider::can_load(&uspec));

    // A multimodal snapshot: the text provider declines (a `vision_config` is present even though
    // the nested text arch is llama), the vision provider claims it.
    let vlm = write_config_only(
        "canload-vlm",
        r#"{"architectures":["LlavaForConditionalGeneration"],"model_type":"llava",
            "text_config":{"architectures":["LlamaForCausalLM"],"model_type":"llama"},
            "vision_config":{"hidden_size":16}}"#,
    );
    let vspec = LoadSpec::dense(vlm.to_str().unwrap().to_string());
    assert!(
        !candle_llm::provider::can_load(&vspec),
        "text provider must decline a VLM"
    );
    assert!(
        candle_llm::llava::can_load(&vspec),
        "vision provider must claim a VLM"
    );

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
    }

    // An unsupported / non-LLM GGUF arch (here `bert`) is declined — likewise weightlessly, from the
    // header alone — so `load_for_model` returns a clean `Unsupported` instead of routing it here.
    let bert = write_minimal_gguf("canload-gguf-bert", "bert");
    let bspec = LoadSpec::dense(bert.to_str().unwrap().to_string());
    assert!(
        !candle_llm::provider::can_load(&bspec),
        "text provider must decline an unsupported/non-LLM GGUF arch"
    );

    // A `*.gguf` path that doesn't exist (or isn't a parseable GGUF) is declined gracefully — the
    // header probe fails closed rather than claiming a file it can't read.
    assert!(!candle_llm::provider::can_load(&LoadSpec::dense(
        "/no/such/model-Q4_K_M.gguf"
    )));

    // A nonexistent snapshot path is declined gracefully.
    assert!(!candle_llm::provider::can_load(&LoadSpec::dense(
        "/no/such/dir"
    )));
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
            assert!(
                m.contains("bert"),
                "error should surface the model arch: {m}"
            );
        }
        Err(e) => panic!("expected Unsupported, got error: {e}"),
        Ok(_) => panic!("expected Unsupported, got a loaded provider"),
    }
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
    let header_len = Content::read(&mut f)
        .expect("read gguf header")
        .tensor_data_offset as usize;
    let mut bytes = std::fs::read(&gguf).expect("read gguf");
    bytes.truncate(header_len);
    let fixture = Fixture::new("candle-llm-gguf-trunc-", Some("header-only.gguf"));
    let trunc = &*fixture;
    std::fs::write(trunc, &bytes).unwrap();
    let tspec = LoadSpec::dense(trunc.to_str().unwrap().to_string());
    assert!(
        candle_llm::provider::can_load(&tspec),
        "header-only (tensor-data-truncated) GGUF must still resolve — the probe is weightless"
    );
}

// ---- sc-20683: compressed-KV policy parity (Candle runs dense, with the shared reason) ----

const LLAMA_IDENTITY: &str = r#""architectures": ["LlamaForCausalLM"], "model_type": "llama","#;
const QWEN3_IDENTITY: &str = r#""architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3","#;
const MISTRAL_IDENTITY: &str =
    r#""architectures": ["MistralForCausalLM"], "model_type": "mistral","#;

/// The provider of a tiny snapshot with `identity` and `head_dim`-channel heads exactly as
/// production loads it: no tiny fixture is a measured architecture, so it has no table family.
fn load_kv_provider_unarmed(
    identity: &str,
    qk_norm: bool,
    head_dim: usize,
) -> (Fixture, LlamaProvider) {
    let dir = write_snapshot_with_heads(identity, qk_norm, head_dim);
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap().to_string()))
        .expect("load synthetic provider");
    assert_eq!(provider.kv_model_family(), None, "not a measured model");
    (dir, provider)
}

/// [`load_kv_provider_unarmed`] with head dimension 64 (one the fused reader reads), test-armed as
/// its decoder's dispatch family (`core_llm::kv_model_family` of its identity), as if it were that
/// family's measured architecture.
fn load_kv_provider(identity: &str, qk_norm: bool) -> (Fixture, LlamaProvider) {
    let (dir, mut provider) = load_kv_provider_unarmed(identity, qk_norm, 64);
    let config: serde_json::Value =
        serde_json::from_str(&format!("{{ {identity} \"_\": 0 }}")).unwrap();
    let model_type = config["model_type"].as_str().unwrap();
    let decoder = if model_type == "qwen3" {
        "qwen3"
    } else {
        "llama"
    };
    provider.arm_kv_model_family_for_tests(core_llm::kv_model_family(
        decoder,
        config["architectures"][0].as_str().unwrap(),
        model_type,
    ));
    (dir, provider)
}

/// A tiny hybrid (Qwen3.5-style: three GatedDeltaNet layers and one full-attention layer)
/// snapshot the provider loads as its hybrid decoder.
fn write_hybrid_snapshot() -> Fixture {
    let fixture = Fixture::new("candle-llm-conformance-hybrid-", None);
    let dir = &*fixture;
    let config = format!(
        r#"{{ "model_type": "qwen3_5", "text_config": {{
            "model_type": "qwen3_5_text", "hidden_size": 32, "num_hidden_layers": 4,
            "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0, "partial_rotary_factor": 0.5,
            "max_position_embeddings": 256, "tie_word_embeddings": false,
            "full_attention_interval": 4, "linear_num_value_heads": 4,
            "linear_num_key_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4, "eos_token_id": 999 }} }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    let (h, inter, conv, value, hv, hd) = (32usize, 64usize, 32usize, 16usize, 4usize, 8usize);
    let mut rng = SplitMix64::new(0x3527B);
    let mut arrays: HashMap<String, Tensor> = HashMap::new();
    let p = "model.language_model";
    let mut put = |key: String, dims: &[usize]| {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        arrays.insert(key, Tensor::from_vec(data, dims, &Device::Cpu).unwrap());
    };
    put(format!("{p}.embed_tokens.weight"), &[VOCAB, h]);
    put(format!("{p}.norm.weight"), &[h]);
    put("lm_head.weight".into(), &[VOCAB, h]);
    for i in 0..4 {
        let lp = |s: &str| format!("{p}.layers.{i}.{s}");
        put(lp("input_layernorm.weight"), &[h]);
        put(lp("post_attention_layernorm.weight"), &[h]);
        put(lp("mlp.gate_proj.weight"), &[inter, h]);
        put(lp("mlp.up_proj.weight"), &[inter, h]);
        put(lp("mlp.down_proj.weight"), &[h, inter]);
        if i < 3 {
            put(lp("linear_attn.in_proj_qkv.weight"), &[conv, h]);
            put(lp("linear_attn.in_proj_z.weight"), &[value, h]);
            put(lp("linear_attn.in_proj_a.weight"), &[hv, h]);
            put(lp("linear_attn.in_proj_b.weight"), &[hv, h]);
            put(lp("linear_attn.conv1d.weight"), &[conv, 1, 4]);
            put(lp("linear_attn.A_log"), &[hv]);
            put(lp("linear_attn.dt_bias"), &[hv]);
            put(lp("linear_attn.norm.weight"), &[4]);
            put(lp("linear_attn.out_proj.weight"), &[h, value]);
        } else {
            put(lp("self_attn.q_proj.weight"), &[4 * hd * 2, h]);
            put(lp("self_attn.k_proj.weight"), &[2 * hd, h]);
            put(lp("self_attn.v_proj.weight"), &[2 * hd, h]);
            put(lp("self_attn.o_proj.weight"), &[h, 4 * hd]);
            put(lp("self_attn.q_norm.weight"), &[hd]);
            put(lp("self_attn.k_norm.weight"), &[hd]);
        }
    }
    candle_core::safetensors::save(&arrays, dir.join("model.safetensors")).unwrap();
    fixture
}

/// AC2: the cross-backend compressed-KV conformance table (`core_llm_testkit::kv_policy_cases`) run
/// through Candle's production plan — the same table MLX's plan passes with its fused reader. Every
/// request reaches MLX's policy decision and reason, except that a request MLX runs compressed
/// reports `ReaderUnavailable` here; no report claims a compressed format or counter. Each case
/// plans on a decoder this backend actually loaded — a plain one, one outside the fused reader's
/// geometry (head dimension 96: `UnsupportedGeometry` on both backends) and a hybrid recurrent
/// one — test-armed as the case's table family.
#[test]
fn candle_kv_plan_conforms_to_the_cross_backend_policy_table() {
    use core_llm::KvModelFamily;
    use core_llm_testkit::{kv_policy_conformance, KvBackendDecision, KvCaseDecoder, KvReader};
    let (_llama_dir, llama) = load_kv_provider(LLAMA_IDENTITY, false);
    let (_qwen_dir, mut plain) = load_kv_provider(QWEN3_IDENTITY, true);
    // Mistral loads through the Llama decoder but is not a table family.
    let (_mistral_dir, mistral) = load_kv_provider(MISTRAL_IDENTITY, false);
    assert_eq!(llama.kv_model_family(), Some(KvModelFamily::Llama));
    assert_eq!(plain.kv_model_family(), Some(KvModelFamily::Qwen3));
    assert_eq!(mistral.kv_model_family(), None);
    let (_geometry_dir, mut outside_geometry) = load_kv_provider_unarmed(LLAMA_IDENTITY, false, 96);
    let hybrid_dir = write_hybrid_snapshot();
    let mut hybrid =
        LlamaProvider::load(&LoadSpec::dense(hybrid_dir.to_str().unwrap().to_string()))
            .expect("load the synthetic hybrid provider");
    assert_eq!(hybrid.kv_model_family(), None);
    kv_policy_conformance(KvReader::Unavailable, |case| {
        let provider = match case.decoder {
            KvCaseDecoder::Plain => &mut plain,
            KvCaseDecoder::UnsupportedGeometry => &mut outside_geometry,
            KvCaseDecoder::Hybrid => &mut hybrid,
        };
        provider.arm_kv_model_family_for_tests(case.family);
        KvBackendDecision::Dense(provider.kv_cache_plan(
            case.policy,
            case.context_tokens,
            case.max_new_tokens,
            case.batch,
            case.multimodal,
        ))
    });
}

/// AC1 through the production entry point: every generation reports a dense cache with the shared
/// reason — `PolicyDisabled` un-opted, the table's refusal for a short context or an unnamed
/// family — and opting in changes nothing about what Candle generates.
#[test]
fn candle_generations_report_the_dense_cache_and_generate_unchanged() {
    use core_llm::{KvCacheFallbackReason as Reason, KvCacheReport, KvCompressionPolicy as Policy};
    let generate = |provider: &LlamaProvider, policy| {
        let request = TextLlmRequest {
            messages: vec![Message::user("t5 t6 t7 t8")],
            sampling: core_llm::Sampling::greedy(),
            max_new_tokens: 8,
            seed: Some(0),
            kv_compression: policy,
            ..Default::default()
        };
        core_llm::TextLlm::generate(provider, &request, &mut |_| {}).unwrap()
    };
    let (_qwen_dir, qwen) = load_kv_provider(QWEN3_IDENTITY, true);
    let off = generate(&qwen, Policy::Off);
    let opted_in = generate(&qwen, Policy::Qualified);
    assert_eq!(
        off.kv_cache,
        Some(KvCacheReport::dense(Reason::PolicyDisabled, None))
    );
    assert_eq!(
        opted_in.kv_cache,
        Some(KvCacheReport::dense(Reason::BelowMinimumContext, None))
    );
    assert_eq!(
        (opted_in.text.as_str(), opted_in.usage),
        (off.text.as_str(), off.usage)
    );
    let (_mistral_dir, mistral) = load_kv_provider(MISTRAL_IDENTITY, false);
    assert_eq!(
        generate(&mistral, Policy::Qualified).kv_cache,
        Some(KvCacheReport::dense(Reason::UnqualifiedModel, None))
    );
}

/// AC1: a generation at Qwen3's qualified minimum context — one MLX runs on the fused compressed
/// reader — runs dense on Candle as `ReaderUnavailable`, claims no compressed format or counter,
/// and generates exactly what the un-opted request does.
#[test]
fn a_qualified_candle_generation_runs_dense_as_reader_unavailable() {
    use core_llm::{
        KvCacheCounters, KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy,
        KvModelFamily, KV_COMPRESSION_QUALIFICATIONS,
    };
    let min = KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == KvModelFamily::Qwen3)
        .unwrap()
        .min_context_tokens;
    let identity = format!(
        "{QWEN3_IDENTITY} \"max_position_embeddings\": {},",
        min + 1024
    );
    let (_dir, qwen) = load_kv_provider(&identity, true);
    let words = (0..min)
        .map(|i| format!("t{}", i % 26 + 6))
        .collect::<Vec<_>>()
        .join(" ");
    let generate = |policy| {
        let request = TextLlmRequest {
            messages: vec![Message::user(words.clone())],
            sampling: core_llm::Sampling::greedy(),
            max_new_tokens: 4,
            seed: Some(0),
            kv_compression: policy,
            ..Default::default()
        };
        core_llm::TextLlm::generate(&qwen, &request, &mut |_| {}).unwrap()
    };
    let off = generate(Policy::Off);
    let opted_in = generate(Policy::Qualified);
    assert!(u64::from(opted_in.usage.prompt_tokens) >= min);
    let report = opted_in
        .kv_cache
        .clone()
        .expect("Candle reports its KV cache");
    assert_eq!(report.fallback, Some(Reason::ReaderUnavailable));
    assert!(report
        .detail
        .as_deref()
        .is_some_and(|d| d.contains("Candle")));
    assert_eq!(report.format, None);
    assert_eq!(report.counters, KvCacheCounters::default());
    assert!(!report.ran_compressed());
    assert_eq!(
        (opted_in.text.as_str(), opted_in.usage),
        (off.text.as_str(), off.usage)
    );
}

/// AC1: Candle bounds the final context like MLX — a qualified-length prompt whose token budget
/// reaches past the evidenced window reports the table's `AboveQualifiedContext`, not
/// `ReaderUnavailable`.
#[test]
fn a_candle_token_budget_past_the_qualified_window_reports_the_table_refusal() {
    use core_llm::{
        KvCacheFallbackReason as Reason, KvCacheReport, KvCompressionPolicy as Policy,
        KvModelFamily, KV_COMPRESSION_QUALIFICATIONS,
    };
    let row = KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .find(|row| row.family == KvModelFamily::Qwen3)
        .unwrap();
    let max = row
        .max_context_tokens
        .expect("the Qwen3 row bounds its window");
    let identity = format!(
        "{QWEN3_IDENTITY} \"max_position_embeddings\": {},",
        max + 1024
    );
    let (_dir, qwen) = load_kv_provider(&identity, true);
    let words = (0..row.min_context_tokens)
        .map(|i| format!("t{}", i % 26 + 6))
        .collect::<Vec<_>>()
        .join(" ");
    let request = TextLlmRequest {
        messages: vec![Message::user(words)],
        sampling: core_llm::Sampling::greedy(),
        // The budget alone carries the final context past the window; every generated piece is
        // `t<N>`, so the stop string ends the decode at its first token.
        max_new_tokens: u32::try_from(max - row.min_context_tokens).unwrap(),
        stop: vec!["t".into()],
        seed: Some(0),
        kv_compression: Policy::Qualified,
        ..Default::default()
    };
    let output = core_llm::TextLlm::generate(&qwen, &request, &mut |_| {}).unwrap();
    assert!(u64::from(output.usage.prompt_tokens) >= row.min_context_tokens);
    assert_eq!(
        output.kv_cache,
        Some(KvCacheReport::dense(Reason::AboveQualifiedContext, None))
    );
}

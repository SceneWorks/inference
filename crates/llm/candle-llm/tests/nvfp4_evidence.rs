//! `nvfp4_evidence` — the real-weight evidence run for NVFP4 projections (sc-24135, epic sc-24128).
//!
//! One `#[ignore]`d test that loads a snapshot twice — bf16, then NVFP4 — and writes
//! one JSON document with, per row: the resident weight census (bytes and bits/param by projection
//! kind), device memory in use after load, the greedy 256-token fixture (tokens + decoded text), and
//! perplexity over a fixed token slice of a committed text file. The NVFP4 row also records the
//! first token index at which its greedy output diverges from the bf16 row (NVFP4 is lossy;
//! divergence is expected, its position is the evidence). A third, provider-level row loads the
//! same snapshot through `LlamaProvider::load` with `Quantize::Nvfp4` and records the provider's
//! `LoadRecord`, proving the selector reaches the loader end to end.
//!
//! The snapshot's `config.json` picks the family, as the provider does: a Qwen3.5/3.6/3.8 hybrid
//! (`Qwen35Model`, sc-24135) or a llama-family `CausalLm` — Qwen3-8B, whose NVFP4 load arrived in
//! sc-24140 (every attention/MLP projection and the LM head through the shared loader). Both run
//! the same fixture and the same perplexity slice, so the documents compare across families.
//!
//! Every input is passed in; nothing is derived from a cache:
//!
//! | variable                      | meaning                                                   |
//! |-------------------------------|-----------------------------------------------------------|
//! | `NVFP4_EVIDENCE_SNAPSHOT`     | snapshot directory (config.json, tokenizer*.json, shards) |
//! | `NVFP4_EVIDENCE_OUTPUT`       | JSON path to write (must not exist)                       |
//! | `NVFP4_EVIDENCE_PPL_TEXT`     | UTF-8 text file the perplexity slice is tokenized from    |
//! | `NVFP4_EVIDENCE_PPL_TOKENS`   | slice length in tokens (default 2048)                     |
//! | `NVFP4_EVIDENCE_NEW_TOKENS`   | greedy fixture length (default 256)                       |
//! | `NVFP4_EVIDENCE_LABEL`        | hardware label (default `RTX Pro 6000 / sm_120`)          |
//!
//! The greedy fixture is the sc-24129 decode-bench fixture: the same user prompt through the
//! snapshot's chat template, `temperature = 0`, no stop tokens, reference decode loop.
//!
//! Perplexity is `exp(mean NLL)` of tokens `1..N` given their prefix, over the first `N` tokens of
//! the text's plain encoding (no chat template, no special tokens), teacher-forced through the
//! decoder in 512-token chunks on one cache (the chunking changes memory, not the math).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor, D};
use candle_llm::decode::{generate_from_prefill, CancelFlag, GenerationConfig};
use candle_llm::device::select_device;
use candle_llm::models::{CausalLm, Qwen35Config, Qwen35Model};
use candle_llm::primitives::{input_ids, ProjectionFormat, ProjectionTally, WeightCensus, Weights};
use core_llm::{ChatTemplate, JinjaChatTemplate, LoadSpec, Message, RenderOptions, Tokenizer};
use serde_json::{json, Value};

const DEFAULT_LABEL: &str = "RTX Pro 6000 / sm_120";
/// The sc-24129 decode-bench fixture prompt, verbatim.
const FIXTURE_PROMPT: &str = "Write a detailed, multi-paragraph explanation of how transformer \
    language models generate text. Cover tokenization, self-attention, the key/value cache, and \
    greedy versus sampled decoding, and finish with the trade-offs of speculative decoding.";
const PPL_CHUNK: usize = 512;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn required(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| panic!("{name} must be set"))
}

fn device_used_bytes(device: &Device) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        device.synchronize().ok()?;
        let cuda = device.as_cuda_device().ok()?;
        let (free, total) = cuda.cuda_stream().context().mem_get_info().ok()?;
        return Some((total - free) as u64);
    }
    #[allow(unreachable_code)]
    {
        let _ = device;
        None
    }
}

fn tokenizer(snapshot: &Path) -> Tokenizer {
    Tokenizer::from_file(snapshot.join("tokenizer.json")).expect("tokenizer.json")
}

fn render_prompt(snapshot: &Path, user: &str) -> Vec<i32> {
    let template = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
        Ok(source) if !source.trim().is_empty() => JinjaChatTemplate::new(source),
        _ => JinjaChatTemplate::from_tokenizer_config_file(snapshot.join("tokenizer_config.json"))
            .expect("tokenizer_config.json chat template"),
    };
    let rendered = template
        .render_with(&[Message::user(user)], &RenderOptions::generation())
        .expect("render prompt");
    tokenizer(snapshot)
        .encode(&rendered, false)
        .expect("encode prompt")
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

fn tally_json(t: &ProjectionTally) -> Value {
    json!({
        "count": t.count,
        "params": t.params,
        "resident_bytes": t.resident_bytes,
        "unmeasured": t.unmeasured,
        "bits_per_param": t.bits_per_param(),
    })
}

fn census_json(c: &WeightCensus) -> Value {
    json!({
        "projections": {
            "dense": tally_json(&c.projections.dense),
            "ggml": tally_json(&c.projections.ggml),
            "prism": tally_json(&c.projections.prism),
            "nvfp4": tally_json(&c.projections.nvfp4),
            "total": tally_json(&c.projections.total()),
        },
        "other_tensors": tally_json(&c.other),
        "total": tally_json(&c.total()),
    })
}

/// The decoder under evidence: the qwen3_5 hybrid or a llama-family `CausalLm` (sc-24140).
enum Model {
    Hybrid(Qwen35Model),
    Causal(CausalLm),
}

/// Whether `config.json` names the qwen3_5 hybrid (else the llama family), by the provider's
/// own dispatch.
fn is_hybrid(config: &Value) -> bool {
    matches!(
        candle_llm::config::Architecture::from_config(config),
        Ok(candle_llm::config::Architecture::Qwen35)
    )
}

fn load(snapshot: &Path, device: &Device, format: Option<&ProjectionFormat>) -> Model {
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    let weights = Weights::from_dir(snapshot, device).expect("load weights");
    if !is_hybrid(&config) {
        let cfg = candle_llm::config::ModelConfig::from_json(&config).expect("llama-family config");
        return Model::Causal(
            CausalLm::from_weights_format(&weights, "", cfg, format).expect("build model"),
        );
    }
    let cfg = Qwen35Config::from_json(&config).expect("qwen3_5 config");
    let prefix = if weights.contains("model.language_model.embed_tokens.weight") {
        "model.language_model"
    } else {
        "model"
    };
    Model::Hybrid(
        Qwen35Model::from_weights_format(&weights, prefix, cfg, format).expect("build model"),
    )
    // `weights` drops here: every tensor the model did not keep (the bf16 NVFP4 sources) is freed.
}

impl Model {
    fn family(&self) -> &'static str {
        match self {
            Model::Hybrid(_) => "qwen35",
            Model::Causal(_) => "llama",
        }
    }

    fn weight_census(&self) -> WeightCensus {
        match self {
            Model::Hybrid(m) => m.weight_census(),
            Model::Causal(m) => m.weight_census(),
        }
    }

    /// The greedy fixture on the reference loop: a `decode_logits` prefill on the model's own
    /// growing cache, then the shared token-at-a-time loop.
    fn greedy(&self, prompt: &[i32], new_tokens: usize, device: &Device) -> Vec<i32> {
        let mut config = GenerationConfig {
            max_new_tokens: new_tokens,
            seed: Some(0),
            stop_tokens: Vec::new(),
            ..Default::default()
        };
        config.sampling.temperature = 0.0;
        let ids = input_ids(prompt, device).unwrap();
        let run = |decoder: &dyn candle_llm::decode::Decode,
                   cache: &mut dyn candle_llm::primitives::KvCache,
                   first: Tensor| {
            generate_from_prefill(
                decoder,
                cache,
                first,
                prompt.to_vec(),
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .expect("greedy generation")
            .tokens
        };
        match self {
            Model::Hybrid(m) => {
                let mut cache = m.new_cache();
                let first = m.decode_logits(&ids, &mut cache, 0).expect("prefill");
                run(m, &mut cache, first)
            }
            Model::Causal(m) => {
                let mut cache = m.new_cache();
                let first = m.decode_logits(&ids, &mut cache, 0).expect("prefill");
                run(m, &mut cache, first)
            }
        }
    }

    /// `(mean NLL, perplexity)` of `ids[1..]` given their prefixes.
    fn perplexity(&self, ids: &[i32], device: &Device) -> (f64, f64) {
        match self {
            Model::Hybrid(m) => {
                let mut cache = m.new_cache();
                perplexity_with(ids, device, |chunk, start| {
                    m.forward(chunk, &mut cache, start)
                        .expect("teacher-forced forward")
                })
            }
            Model::Causal(m) => {
                let mut cache = m.new_cache();
                perplexity_with(ids, device, |chunk, start| {
                    m.decode_logits_all(chunk, &mut cache, start)
                        .expect("teacher-forced forward")
                })
            }
        }
    }
}

/// `(mean NLL, perplexity)` of `ids[1..]` given their prefixes, teacher-forced through `forward`
/// (`[1, len, vocab]` logits for a chunk at a start position, on one cache) in `PPL_CHUNK`-token
/// chunks.
fn perplexity_with(
    ids: &[i32],
    device: &Device,
    mut forward: impl FnMut(&Tensor, i32) -> Tensor,
) -> (f64, f64) {
    let mut nll_sum = 0f64;
    let mut scored = 0usize;
    let mut start = 0usize;
    while start < ids.len() {
        let end = (start + PPL_CHUNK).min(ids.len());
        let chunk = &ids[start..end];
        let logits = forward(&input_ids(chunk, device).unwrap(), start as i32)
            .squeeze(0)
            .unwrap();
        // `logits` is [len, vocab]. Position i predicts ids[start + i + 1]; the last chunk's last
        // position has no target.
        let usable = if end == ids.len() {
            chunk.len() - 1
        } else {
            chunk.len()
        };
        if usable > 0 {
            let logp = candle_nn::ops::log_softmax(
                &logits
                    .narrow(0, 0, usable)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap(),
                D::Minus1,
            )
            .unwrap();
            let targets: Vec<u32> = ids[start + 1..start + 1 + usable]
                .iter()
                .map(|&t| t as u32)
                .collect();
            let targets = Tensor::from_vec(targets, (usable, 1), device).unwrap();
            let picked = logp
                .gather(&targets, 1)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            nll_sum -= picked as f64;
            scored += usable;
        }
        start = end;
    }
    let mean = nll_sum / scored as f64;
    (mean, mean.exp())
}

fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn first_divergence_is_the_first_differing_index() {
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
    assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
}

#[test]
#[ignore = "real weights: set NVFP4_EVIDENCE_SNAPSHOT / _OUTPUT / _PPL_TEXT on a CUDA sm_120 box"]
fn nvfp4_real_weight_evidence() {
    let snapshot = PathBuf::from(required("NVFP4_EVIDENCE_SNAPSHOT"));
    let output = PathBuf::from(required("NVFP4_EVIDENCE_OUTPUT"));
    assert!(!output.exists(), "{} already exists", output.display());
    let ppl_text_path = PathBuf::from(required("NVFP4_EVIDENCE_PPL_TEXT"));
    let ppl_tokens: usize = env_or("NVFP4_EVIDENCE_PPL_TOKENS", "2048").parse().unwrap();
    let new_tokens: usize = env_or("NVFP4_EVIDENCE_NEW_TOKENS", "256").parse().unwrap();
    let label = env_or("NVFP4_EVIDENCE_LABEL", DEFAULT_LABEL);

    let device = select_device().expect("device");
    let tok = tokenizer(&snapshot);
    let prompt = render_prompt(&snapshot, FIXTURE_PROMPT);
    let text_bytes = std::fs::read(&ppl_text_path).expect("read perplexity text");
    let text = String::from_utf8(text_bytes.clone()).expect("UTF-8 perplexity text");
    let all_ids: Vec<i32> = tok
        .encode(&text, false)
        .expect("encode perplexity text")
        .into_iter()
        .map(|id| id as i32)
        .collect();
    assert!(
        all_ids.len() >= ppl_tokens,
        "perplexity text encodes to {} tokens, fewer than the {ppl_tokens}-token slice",
        all_ids.len()
    );
    let slice = &all_ids[..ppl_tokens];
    let slice_ids_sha = sha256_hex(
        &slice
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<u8>>(),
    );

    let baseline_used = device_used_bytes(&device);
    let mut family = "";
    let mut rows = Vec::new();
    let mut bf16_tokens: Option<Vec<i32>> = None;
    for (name, nvfp4) in [("bf16", false), ("nvfp4", true)] {
        let format = nvfp4.then(|| ProjectionFormat::nvfp4(&device).expect("NVFP4 capability"));
        let started = Instant::now();
        let model = load(&snapshot, &device, format.as_ref());
        family = model.family();
        let load_secs = started.elapsed().as_secs_f64();
        let used_after_load = device_used_bytes(&device);
        let census = model.weight_census();

        let started = Instant::now();
        let tokens = model.greedy(&prompt, new_tokens, &device);
        let greedy_secs = started.elapsed().as_secs_f64();
        let text_out = tok
            .decode(&tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)
            .unwrap_or_default();

        let started = Instant::now();
        let (mean_nll, ppl) = model.perplexity(slice, &device);
        let ppl_secs = started.elapsed().as_secs_f64();
        let divergence = bf16_tokens
            .as_ref()
            .map(|reference| first_divergence(reference, &tokens));
        eprintln!(
            "[nvfp4-evidence] {family} {name}: load {load_secs:.1}s, resident weights {} B ({:?} bits/param; \
             projections {:?}), ppl {ppl:.4} over {ppl_tokens} tokens, first divergence vs bf16 \
             {divergence:?}",
            census.total().resident_bytes,
            census.total().bits_per_param(),
            census.projections.total().bits_per_param(),
        );
        rows.push(json!({
            "row": name,
            "load_seconds": load_secs,
            "device_used_bytes_after_load": used_after_load,
            "weight_census": census_json(&census),
            "greedy": {
                "new_tokens": new_tokens,
                "seconds": greedy_secs,
                "tokens": tokens,
                "text": text_out,
                "first_divergence_vs_bf16": divergence,
            },
            "perplexity": {
                "tokens": ppl_tokens,
                "scored_tokens": ppl_tokens - 1,
                "mean_nll": mean_nll,
                "ppl": ppl,
                "seconds": ppl_secs,
            },
        }));
        if !nvfp4 {
            bf16_tokens = Some(tokens);
        }
        drop(model);
    }

    // The provider path: the `Quantize::Nvfp4` selector through `LlamaProvider::load`.
    let started = Instant::now();
    let provider = candle_llm::LlamaProvider::load(&LoadSpec {
        source: snapshot.to_string_lossy().into_owned(),
        projector_source: None,
        quantize: Some(core_llm::Quantize::Nvfp4),
        cuda_graphs: None,
    })
    .expect("provider NVFP4 load");
    let provider_load_secs = started.elapsed().as_secs_f64();
    let record = provider.load_record();
    let provider_row = json!({
        "row": "provider_nvfp4",
        "load_seconds": provider_load_secs,
        "device_used_bytes_after_load": device_used_bytes(&device),
        "requested": format!("{:?}", record.requested),
        // The hybrid's census includes its MTP head; the llama family has none (sc-24140).
        "weight_census_incl_mtp": record.census.as_ref().map(census_json),
    });
    drop(provider);

    let doc = json!({
        "story": if family == "llama" { "sc-24140" } else { "sc-24135" },
        "model_family": family,
        "hardware_label": label,
        "snapshot": snapshot.to_string_lossy(),
        "device_used_bytes_before_load": baseline_used,
        "fixture": {
            "prompt": FIXTURE_PROMPT,
            "prompt_tokens": prompt.len(),
            "decode": "greedy, temperature 0, no stop tokens, reference loop",
        },
        "perplexity_slice": {
            "text_file": ppl_text_path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "text_sha256": sha256_hex(&text_bytes),
            "text_tokens_total": all_ids.len(),
            "slice": format!("first {ppl_tokens} tokens of the plain encoding (no template, no special tokens)"),
            "slice_ids_sha256_le_i32": slice_ids_sha,
            "chunk_tokens": PPL_CHUNK,
        },
        "rows": rows,
        "provider": provider_row,
    });
    std::fs::write(&output, serde_json::to_string_pretty(&doc).unwrap()).expect("write output");
}

/// AC2 on the real snapshot: run with `CANDLE_LLM_DEVICE=cpu` (the provider's explicit device
/// override). The NVFP4 load must fail with the typed, capability-naming refusal — before the
/// accelerator gate, admission or any weight read. Prints the refusal for the evidence record.
#[test]
#[ignore = "real snapshot: set NVFP4_EVIDENCE_SNAPSHOT and CANDLE_LLM_DEVICE=cpu"]
fn nvfp4_refused_on_cpu_for_the_real_snapshot() {
    let snapshot = required("NVFP4_EVIDENCE_SNAPSHOT");
    let started = Instant::now();
    match candle_llm::LlamaProvider::load(&LoadSpec {
        source: snapshot,
        projector_source: None,
        quantize: Some(core_llm::Quantize::Nvfp4),
        cuda_graphs: None,
    }) {
        Err(core_llm::Error::Unsupported(msg)) => {
            assert!(msg.starts_with("nvfp4: "), "{msg}");
            eprintln!(
                "[sc-24135] CPU NVFP4 refusal after {:.3}s: Unsupported({msg:?})",
                started.elapsed().as_secs_f64()
            );
        }
        Err(other) => panic!("expected the typed NVFP4 refusal, got {other:?}"),
        Ok(_) => panic!("NVFP4 must not load on CPU"),
    }
}

//! Real-weights GGUF parity tests (`#[ignore]` — need models on disk), story 7254.
//!
//! Point `CANDLE_LLM_GGUF` at a single `*.gguf` checkpoint and `CANDLE_LLM_TEST_MODEL` at the same
//! model's HF snapshot (for the reference safetensors weights + tokenizer), then:
//!
//! ```text
//! CANDLE_LLM_GGUF=/path/Model-F16.gguf CANDLE_LLM_TEST_MODEL=/path/HF-snapshot \
//!   cargo test --features cuda --test gguf -- --ignored --nocapture
//! ```
//!
//! ## What parity means here
//! A direct GGUF load is **lossless apart from the GGUF's own quantization**, so it should reproduce
//! the HF safetensors load. We measure the *behavioral* distribution (softmax over the next-token
//! logits) rather than raw-logit or greedy-token equality: raw-logit cosine over a large vocab is
//! dominated by tens of thousands of bf16-noisy background logits, and greedy-token equality is
//! brittle (a single near-tie diverges and cascades). A lossless `F16`/`BF16` GGUF additionally
//! reproduces HF's top-1 token and greedy continuation; a lossy k-quant only needs a high softmax
//! cosine (a broken dequant or a mis-permuted q/k projection collapses it toward 0 / gibberish).

use candle_core::DType;

use candle_llm::config::ModelConfig;
use candle_llm::decode::{generate, CancelFlag, GenerationConfig};
use candle_llm::device::select_device;
use candle_llm::gguf::GgufCheckpoint;
use candle_llm::models::CausalLm;
use candle_llm::primitives::sampler::SamplingParams;
use candle_llm::primitives::{input_ids, Weights};
use candle_llm::provider::{eos_token_ids, LlamaProvider};
use core_llm::{LoadSpec, Message, Sampling, StreamEvent, TextLlm, TextLlmRequest, Tokenizer};

const PROMPT: &str = "The capital of France is";

fn gguf_path() -> Option<String> {
    std::env::var("CANDLE_LLM_GGUF")
        .ok()
        .filter(|p| !p.is_empty())
}

fn hf_dir() -> Option<String> {
    std::env::var("CANDLE_LLM_TEST_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

/// Last-position prefill logits as host `f32`.
fn prefill_logits(model: &CausalLm, ids: &[i32]) -> Vec<f32> {
    let mut cache = model.new_cache();
    let arr = input_ids(ids, model.device()).unwrap();
    let logits = model.decode_logits(&arr, &mut cache, 0).unwrap();
    logits
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn greedy_tokens(model: &CausalLm, ids: &[i32], stop: &[i32], n: usize) -> Vec<i32> {
    let cfg = GenerationConfig {
        max_new_tokens: n,
        sampling: SamplingParams::default(), // greedy
        seed: Some(0),
        stop_tokens: stop.to_vec(),
    };
    generate(model, ids, &cfg, &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens
}

/// A directly-loaded GGUF tracks the HF safetensors load: high next-token softmax cosine, and — for a
/// lossless F16/BF16 GGUF — the same top-1 token and greedy continuation. This is the end-to-end
/// proof that the key remap, the config-from-metadata, and the q/k RoPE un-permute are all correct.
#[test]
#[ignore = "needs CANDLE_LLM_GGUF + CANDLE_LLM_TEST_MODEL"]
fn gguf_load_tracks_hf() {
    let (Some(gguf), Some(dir)) = (gguf_path(), hf_dir()) else {
        eprintln!("skip: set CANDLE_LLM_GGUF + CANDLE_LLM_TEST_MODEL");
        return;
    };
    let device = select_device().unwrap();

    // HF reference.
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let hf_model =
        CausalLm::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg.clone())
            .unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let stop = eos_token_ids(std::path::Path::new(&dir));
    let ids = encode(&tok, PROMPT);

    let hf_logits = prefill_logits(&hf_model, &ids);
    let hf_probs = softmax(&hf_logits);
    let hf_top1 = argmax(&hf_logits);
    let hf_tokens = greedy_tokens(&hf_model, &ids, &stop, 24);

    // Direct GGUF load.
    let ck = GgufCheckpoint::open(&gguf, &device).unwrap();
    // Config reconstructed from GGUF metadata must equal the HF config's load-relevant fields.
    assert_eq!(ck.config.hidden_size, cfg.hidden_size, "hidden_size");
    assert_eq!(ck.config.num_layers, cfg.num_layers, "num_layers");
    assert_eq!(ck.config.num_heads, cfg.num_heads, "num_heads");
    assert_eq!(ck.config.num_kv_heads, cfg.num_kv_heads, "num_kv_heads");
    assert_eq!(ck.config.head_dim, cfg.head_dim, "head_dim");
    assert_eq!(ck.config.vocab_size, cfg.vocab_size, "vocab_size");
    assert_eq!(
        ck.config.tie_word_embeddings, cfg.tie_word_embeddings,
        "tie"
    );

    let g_model = CausalLm::from_weights(&ck.weights, "", ck.config.clone()).unwrap();
    let g_logits = prefill_logits(&g_model, &ids);
    let probcos = cosine(&softmax(&g_logits), &hf_probs);
    let g_top1 = argmax(&g_logits);
    let g_tokens = greedy_tokens(&g_model, &ids, &stop, 24);
    let prefix = common_prefix(&g_tokens, &hf_tokens);
    let text = tok
        .decode(
            &g_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            true,
        )
        .unwrap();

    let label = std::path::Path::new(&gguf)
        .file_stem()
        .map(|s| s.to_string_lossy().to_uppercase())
        .unwrap_or_default();
    println!(
        "{label}: softmax-cosine {probcos:.4}  top1 {}  greedy-prefix {prefix}/24  :: {}",
        g_top1 == hf_top1,
        text.replace('\n', " ")
    );

    assert!(!text.trim().is_empty(), "produced no text");
    assert!(
        probcos >= 0.80,
        "softmax cosine {probcos:.4} < 0.80 (sanity floor)"
    );
    let lossless = label.contains("F16") || label.contains("BF16");
    if lossless {
        assert_eq!(g_top1, hf_top1, "lossless next-token mismatch");
        assert!(
            probcos >= 0.999,
            "lossless softmax cosine {probcos:.4} < 0.999"
        );
        assert!(
            prefix >= 20,
            "lossless greedy prefix {prefix}/24 — expected ~exact"
        );
    }
}

/// The tokenizer reconstructed from the GGUF's embedded metadata (the `else` branch when there is no
/// sibling `tokenizer.json`) encodes + decodes identically to the model's real `tokenizer.json`. This
/// guards the BPE reconstruction (vocab, merges, special/added tokens).
#[test]
#[ignore = "needs CANDLE_LLM_GGUF + CANDLE_LLM_TEST_MODEL"]
fn gguf_tokenizer_metadata_matches_hf() {
    let (Some(gguf), Some(dir)) = (gguf_path(), hf_dir()) else {
        eprintln!("skip: set CANDLE_LLM_GGUF + CANDLE_LLM_TEST_MODEL");
        return;
    };
    let device = select_device().unwrap();
    let ck = GgufCheckpoint::open(&gguf, &device).unwrap();
    let from_meta = match ck.tokenizer_from_metadata() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skip: GGUF tokenizer not reconstructable ({e})");
            return;
        }
    };
    let hf = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();

    for text in [
        PROMPT,
        "Hello, world!",
        "def add(a, b):\n    return a + b",
        "The quick brown fox jumps over the lazy dog.",
    ] {
        let a = from_meta.encode(text, false).unwrap();
        let b = hf.encode(text, false).unwrap();
        assert_eq!(
            a, b,
            "encode mismatch for {text:?}\n  meta: {a:?}\n  hf:   {b:?}"
        );
        let da = from_meta.decode(&a, true).unwrap();
        let db = hf.decode(&b, true).unwrap();
        assert_eq!(da, db, "decode mismatch for {text:?}");
    }
}

/// The literal story acceptance: a `*.gguf` streams coherent text through the backend-neutral
/// `core_llm::TextLlm` (the provider loads it directly — tokenizer from the GGUF metadata when there
/// is no sibling `tokenizer.json`).
#[test]
#[ignore = "needs CANDLE_LLM_GGUF"]
fn gguf_streams_through_textllm() {
    let Some(gguf) = gguf_path() else {
        eprintln!("skip: set CANDLE_LLM_GGUF");
        return;
    };
    let provider = LlamaProvider::load(&LoadSpec::dense(gguf)).expect("load gguf provider");

    let req = TextLlmRequest {
        messages: vec![Message::user(PROMPT)],
        sampling: Sampling::greedy(),
        max_new_tokens: 24,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let out = provider
        .generate(&req, &mut |ev| {
            if let StreamEvent::Token { text, .. } = ev {
                streamed.push_str(&text);
            }
        })
        .expect("generate");

    println!("gguf TextLlm stream :: {}", out.text.replace('\n', " "));
    assert!(!out.text.trim().is_empty(), "produced no text");
    assert_eq!(
        streamed, out.text,
        "streamed deltas must reconstruct the final text"
    );
}

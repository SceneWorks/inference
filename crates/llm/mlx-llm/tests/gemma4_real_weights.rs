//! Real-weights Gemma 4 unified forward (`#[ignore]` — needs the shipped text encoder on disk).
//!
//! `gemma4_decoder.rs` pins the block against a reference oracle on a 4-layer synthetic model.
//! That proves the *math*. It cannot prove the loader survives the real thing: 48 layers, 664
//! `model.layers.N.*` tensors whose shapes change with the layer type, a checkpoint that carries
//! **no** `v_proj` for its eight `full_attention` layers, per-layer `layer_scalar` buffers, and a
//! config that arrives as safetensors metadata rather than a `config.json`. This closes that half.
//!
//! Point the env var at the LTX-2.5 packed text encoder and run:
//!
//! ```text
//! MLX_LLM_TEST_GEMMA4_TE=/path/to/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
//!   cargo test -p mlx-llm --test gemma4_real_weights -- --ignored --nocapture
//! ```
//!
//! The path is supplied by the caller and never written down here — this crate does not name model
//! caches or their environment variables in source.
//!
//! **Cost.** The checkpoint is 26.3 GB of bf16 and the forward is memory-bound; it holds one model
//! resident and drops it as soon as the assertions are done. Run it alone.

use std::collections::HashMap;
use std::time::Instant;

use mlx_rs::memory;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};
use serde_json::Value;

use mlx_llm::config::{LayerAttentionType, ModelConfig, RopeType};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::{input_ids, Weights};

/// The decoder's own tensor prefix inside the packed encoder. The checkpoint also carries a vision
/// tower, an audio projector, an LTX-specific text-embedding projection, and the tokenizer — none
/// of which the decoder loads.
const DECODER_PREFIX: &str = "model.";

fn checkpoint() -> Option<String> {
    std::env::var("MLX_LLM_TEST_GEMMA4_TE").ok()
}

/// Read a safetensors file's `__metadata__` map without touching a byte of tensor payload.
///
/// The packed encoder has no `config.json`: the shipped `Gemma4UnifiedConfig` travels in the
/// header's `gemma_config` metadata string. Eight bytes of little-endian length, then that many
/// bytes of JSON header.
fn safetensors_metadata(path: &str) -> HashMap<String, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).expect("open checkpoint");
    let mut len = [0u8; 8];
    f.read_exact(&mut len).expect("read header length");
    let len = u64::from_le_bytes(len) as usize;
    f.seek(SeekFrom::Start(8)).expect("seek to header");
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).expect("read header");
    let header: Value = serde_json::from_slice(&buf).expect("parse safetensors header");
    header
        .get("__metadata__")
        .and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

fn config_from_checkpoint(path: &str) -> ModelConfig {
    let meta = safetensors_metadata(path);
    let raw = meta
        .get("gemma_config")
        .expect("the packed text encoder must carry its `gemma_config` metadata");
    let json: Value = serde_json::from_str(raw).expect("parse gemma_config");
    ModelConfig::from_json(&json).expect("the shipped config must parse")
}

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn host(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// The full 48-layer forward over the shipped bf16 text encoder.
///
/// Asserts, in order: the shipped config resolves the per-layer-type table this decoder was built
/// for; the loader consumes a checkpoint whose eight `full_attention` layers have no `v_proj` at
/// all; and the all-hidden-states forward LTX-2.5 consumes returns 49 finite `[1, S, 3840]` states
/// whose last entry is the final-normed one.
///
/// The finiteness check is not a formality. A wrong `head_dim` on the full layers, a value path
/// reading rotated keys, or a mis-sized shared K/V would not merely shift the numbers — it would
/// reshape a projection and either fail the load or produce NaNs through the softmax.
#[test]
#[ignore = "needs the shipped LTX-2.5 packed text encoder (26.3 GB) via MLX_LLM_TEST_GEMMA4_TE"]
fn gemma4_full_48_layer_forward_on_the_shipped_text_encoder() {
    let path = checkpoint().expect("set MLX_LLM_TEST_GEMMA4_TE");

    // ---- the shipped config resolves the table the decoder needs ----
    let cfg = config_from_checkpoint(&path);
    assert!(
        cfg.is_gemma4(),
        "the packed encoder must be a Gemma 4 config"
    );
    assert_eq!(cfg.num_layers, 48);
    assert_eq!(cfg.hidden_size, 3840);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.vocab_size, 262_144);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.final_logit_softcap, Some(30.0));
    assert_eq!(
        cfg.attn_scale(),
        1.0,
        "Gemma 4 attention scale is a literal 1.0"
    );

    let table = cfg.gemma4.as_ref().expect("a per-layer-type table");
    let full: Vec<usize> = (0..cfg.num_layers)
        .filter(|i| table.layer_type(*i) == LayerAttentionType::Full)
        .collect();
    assert_eq!(
        full,
        vec![5, 11, 17, 23, 29, 35, 41, 47],
        "the shipped schedule is every 6th layer, last included"
    );
    assert_eq!(table.sliding.head_dim, 256);
    assert_eq!(table.full.head_dim, 512);
    assert_eq!(table.sliding.num_kv_heads, 8);
    assert_eq!(table.full.num_kv_heads, 1);
    assert_eq!(table.sliding.sliding_window, Some(1024));
    assert_eq!(table.full.sliding_window, None);
    assert!(table.full.k_eq_v && !table.sliding.k_eq_v);
    assert_eq!(table.sliding.rope_type, RopeType::Default);
    assert_eq!(table.full.rope_type, RopeType::Proportional);

    // ---- the checkpoint really is shaped the way the table says ----
    let t0 = Instant::now();
    let weights = Weights::from_file(&path).expect("load the packed text encoder");
    let load_s = t0.elapsed().as_secs_f64();
    let decoder_tensors = weights
        .keys()
        .filter(|k| k.starts_with(DECODER_PREFIX))
        .count();
    println!(
        "loaded {} tensors ({decoder_tensors} under `{DECODER_PREFIX}`) in {load_s:.1}s",
        weights.len()
    );
    for i in 0..cfg.num_layers {
        let v_proj = format!("model.layers.{i}.self_attn.v_proj.weight");
        let has_v = weights.contains(&v_proj);
        match table.layer_type(i) {
            LayerAttentionType::Sliding => {
                assert!(has_v, "sliding layer {i} must ship its own {v_proj}")
            }
            // `attention_k_eq_v` — the weight does not exist. A decoder that expected one here
            // would fail the load, which is the whole reason this checkpoint needs the shared
            // projection rather than a copy of it.
            LayerAttentionType::Full => assert!(
                !has_v,
                "full layer {i} shares K/V; {v_proj} must not be in the checkpoint"
            ),
        }
        assert!(
            weights.contains(&format!("model.layers.{i}.layer_scalar")),
            "layer {i} must ship its `layer_scalar` buffer"
        );
    }

    // ---- build and run ----
    let t1 = Instant::now();
    let model = CausalLm::from_weights(&weights, "", cfg).expect("build the Gemma 4 decoder");
    let build_s = t1.elapsed().as_secs_f64();
    // Release the towers the decoder does not hold (vision, audio, the LTX text-embedding
    // projection, the packed tokenizer) before the forward, so peak reflects the decoder.
    drop(weights);
    memory::clear_cache();

    let prompt: Vec<i32> = (0..64).map(|i| (i * 37 + 11) % 100_000).collect();
    let t2 = Instant::now();
    let mut cache = model.new_cache();
    let states = model
        .hidden_states(&input_ids(&prompt), &mut cache, 0)
        .expect("48-layer hidden-state forward");

    // MLX is lazy: `hidden_states` returns 49 handles onto one unevaluated 48-layer graph. Forcing
    // only the last would submit the whole thing — every weight page-in, every layer's matmuls — as
    // a single Metal command buffer, which at this model's size exceeds what the GPU will accept and
    // comes back as `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`. Walking the states in
    // order evaluates one layer's subgraph at a time, which is also how a real caller consumes them.
    let mut per_layer = Vec::with_capacity(states.len());
    for (i, s) in states.iter().enumerate() {
        eval([s]).unwrap_or_else(|e| panic!("evaluating hidden_states[{i}]: {e}"));
        per_layer.push(host(s));
    }
    let last = per_layer.last().expect("a final state").clone();
    let forward_s = t2.elapsed().as_secs_f64();

    let peak_gb = memory::get_peak_memory() as f64 / 1e9;
    let active_gb = memory::get_active_memory() as f64 / 1e9;
    println!(
        "gemma4 real-weights: load {load_s:.1}s, build {build_s:.1}s, \
         forward({} tokens) {forward_s:.1}s | peak(active) {peak_gb:.1} GB, active {active_gb:.1} GB",
        prompt.len()
    );

    assert_eq!(
        states.len(),
        49,
        "48 layers plus the input embeddings, HF's `output_hidden_states` layout"
    );
    for (i, s) in states.iter().enumerate() {
        assert_eq!(
            s.shape(),
            &[1, prompt.len() as i32, 3840],
            "hidden_states[{i}] shape"
        );
    }
    assert!(finite(&last), "the final hidden state must be finite");
    // A stack that silently produced zeros (a fully-masked softmax, an all-zero projection) would
    // be finite too; require actual signal.
    let magnitude = last.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        magnitude > 1e-3,
        "final hidden state is degenerate: max|h| = {magnitude}"
    );

    // Every intermediate layer must be finite as well — a NaN introduced at layer 20 can be
    // normalized back into range by the last entry's final norm and would otherwise go unseen.
    for (i, h) in per_layer.iter().enumerate() {
        assert!(finite(h), "hidden_states[{i}] is not finite");
    }

    // One model resident at a time: release it before the harness moves on.
    drop(states);
    drop(cache);
    drop(model);
    memory::clear_cache();
}

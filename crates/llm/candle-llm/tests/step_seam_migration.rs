//! Tiny-config parity for the llama-family migration onto the step seams (epic sc-24128, story
//! sc-24138, AC1).
//!
//! Every model the story migrates — the generic llama family (a plain Llama and the Qwen3 dense
//! architecture Qwen3-8B uses), Gemma 4 (text and an embeddings prefill standing in for the
//! `gemma4_mm` splice), LLaVA, StarCoder2 (the StarVector-8B decoder) and the StarVector-1B
//! GPTBigCode decoder — is decoded here on a deterministic CPU-sized checkpoint, greedy and
//! sampled, and compared against a **golden captured from the pre-migration tree**.
//!
//! The goldens under `tests/goldens/sc24138/` were written by this file at the commit that
//! introduced it (`SC24138_WRITE_GOLDENS=1`), which ran only the pre-migration decode paths: the
//! `CausalLm` reference loop, LLaVA's own caption loop, StarCoder2 through the shared reference
//! loop and the StarVector-1B decoder with its layer-owned cache. Each golden holds the prompt, the
//! greedy tokens, the logits every greedy token was chosen from, and a sampled run's tokens.
//!
//! After the migration the decoders attend un-expanded (`AttnFormulation::Gqa`) by default on
//! every path — the reference loop, the growing and paged backings and the static cache are one
//! arithmetic — and the pre-migration `repeat_kv` expansion (`AttnFormulation::Expanded`) is the
//! explicitly selected comparison. Every model is held to its golden on each path it now has:
//!
//! * the golden producers select **`Expanded`**, the pre-migration arithmetic: the reference path
//!   with it reproduces the golden **bit for bit**, tokens and logits (E2);
//! * the **default** reference loop and growing backing are the static cache's arithmetic: their
//!   logit rows equal the static cache's **bit for bit**, and their tokens are the golden's;
//! * the **step seam on the static cache** produces the golden's greedy and sampled **tokens
//!   exactly**, and every logit row within [`STATIC_LOGIT_TOL`] of the golden's (the un-expanded
//!   attention GEMMs round differently in the last bits);
//! * the **engine** (no proposer, n-gram, and a self-draft) produces the golden's greedy tokens;
//! * the multimodal models (the Gemma 4 soft-token splice, LLaVA, StarCoder2 / StarVector with
//!   their conditioning prefix) prefill through the seam's cache and decode through the engine.
//!
//! **Where "bit for bit" against a golden holds.** The goldens were measured on Windows x86_64
//! MSVC (CPU f32; the Windows CPU and `--features cuda` lanes both reproduce them). Another
//! platform's libm / GEMM kernels round the same graph differently in the last bits — on Linux
//! (WSL Ubuntu, glibc) every golden token is reproduced while logits move by a few ULP — so,
//! as `architecture_forward.rs` scopes its goldens, a golden's logits are compared bit for bit
//! only in the measured configuration ([`goldens_bit_exact`]) and within [`STATIC_LOGIT_TOL`]
//! elsewhere; its tokens are compared exactly everywhere. Comparisons between two paths of this
//! tree (default growing vs static, and so on) are bit for bit on every platform.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use serde_json::{json, Value};

use candle_llm::config::ModelConfig;
use candle_llm::decode::{
    generate, generate_from_prefill_with_stop, generate_speculative, generate_step,
    generate_step_from_prefill, CancelFlag, DecodePath, DraftModelProposer, GenerationConfig,
    NgramProposer, NoProposer, SpeculativePrompt, StepModel, StepRequest, StreamEvent,
};
use candle_llm::llava::LlavaModel;
use candle_llm::models::{
    CausalLm, StarCoder2, StarCoder2Config, StarVectorDecoder, StarVectorDecoderGeometry,
};
use candle_llm::primitives::sampler::{sample, SamplingParams};
use candle_llm::primitives::{
    input_ids, AttnFormulation, DecodeCache, KvCache, KvCacheKind, SplitMix64, StepKvCache,
    TokenRng, Weights,
};

// ---- goldens -----------------------------------------------------------------------------------

const NEW_TOKENS: usize = 12;
const SAMPLED_SEED: u64 = 7;

/// One model's pre-migration decode.
#[derive(Clone, Debug, PartialEq)]
struct Golden {
    prompt: Vec<i32>,
    greedy_tokens: Vec<i32>,
    /// `greedy_logits[i]` is the row `greedy_tokens[i]` was the argmax of (the prefill row first).
    greedy_logits: Vec<Vec<f32>>,
    sampled_tokens: Vec<i32>,
}

impl Golden {
    fn to_json(&self) -> Value {
        json!({
            "prompt": self.prompt,
            "greedy_tokens": self.greedy_tokens,
            "greedy_logits": self.greedy_logits,
            "sampled_tokens": self.sampled_tokens,
        })
    }

    fn from_json(v: &Value) -> Self {
        let ids = |v: &Value| -> Vec<i32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect()
        };
        Self {
            prompt: ids(&v["prompt"]),
            greedy_tokens: ids(&v["greedy_tokens"]),
            greedy_logits: v["greedy_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect()
                })
                .collect(),
            sampled_tokens: ids(&v["sampled_tokens"]),
        }
    }
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("sc24138")
        .join(format!("{name}.json"))
}

/// Whether a golden's logits are compared bit for bit in this build: only in the configuration
/// the goldens were measured on, Windows x86_64 MSVC (see the module docs). Elsewhere tokens stay
/// exact and logits are held within [`STATIC_LOGIT_TOL`].
fn goldens_bit_exact(os: &str, arch: &str, target_env: &str) -> bool {
    matches!((os, arch, target_env), ("windows", "x86_64", "msvc"))
}

fn goldens_bit_exact_here() -> bool {
    goldens_bit_exact(
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(target_env = "msvc") {
            "msvc"
        } else {
            "other"
        },
    )
}

#[test]
fn golden_bit_exactness_is_limited_to_the_measured_configuration() {
    assert!(goldens_bit_exact("windows", "x86_64", "msvc"));
    for (os, arch, env) in [
        ("windows", "x86_64", "gnu"),
        ("windows", "aarch64", "msvc"),
        ("linux", "x86_64", "other"),
        ("linux", "aarch64", "other"),
        ("macos", "aarch64", "other"),
    ] {
        assert!(!goldens_bit_exact(os, arch, env), "{os}/{arch}/{env}");
    }
}

/// Logit rows against a golden's: bit for bit in the measured configuration, within
/// [`STATIC_LOGIT_TOL`] elsewhere (see the module docs).
fn assert_golden_rows(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    if goldens_bit_exact_here() {
        assert_eq!(got, want, "{what}: the golden's logits, bit for bit");
    } else {
        let diff = max_abs_diff(got, want);
        assert!(
            diff <= STATIC_LOGIT_TOL,
            "{what}: logits drift {diff} from the golden (off the measured configuration)"
        );
    }
}

/// A decode against a golden: prompt and tokens exactly, logits per [`assert_golden_rows`].
fn assert_golden(what: &str, got: &Golden, want: &Golden) {
    assert_eq!(got.prompt, want.prompt, "{what}: prompt");
    assert_eq!(
        got.greedy_tokens, want.greedy_tokens,
        "{what}: greedy tokens"
    );
    assert_eq!(
        got.sampled_tokens, want.sampled_tokens,
        "{what}: sampled tokens"
    );
    assert_golden_rows(what, &got.greedy_logits, &want.greedy_logits);
}

/// The committed golden for `name`, after checking that `before` (this tree's pre-migration path,
/// the `Expanded` arithmetic) still reproduces it — bit for bit in the measured configuration
/// ([`assert_golden`]). With `SC24138_WRITE_GOLDENS=1` the golden is (re)written from `before`
/// instead — done once, at the commit that introduced this file.
fn golden(name: &str, before: Golden) -> Golden {
    let path = golden_path(name);
    if std::env::var("SC24138_WRITE_GOLDENS").is_ok_and(|v| v == "1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = serde_json::to_string_pretty(&before.to_json()).unwrap();
        text.push('\n');
        std::fs::write(&path, text).unwrap();
        return before;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let want = Golden::from_json(&serde_json::from_str(&text).unwrap());
    assert_golden(
        &format!("{name}: the pre-migration (expanded) path"),
        &before,
        &want,
    );
    want
}

// ---- shared helpers ----------------------------------------------------------------------------

fn greedy_config(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

fn sampled_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 0,
        presence_penalty: 0.0,
        repetition_penalty: 1.1,
        repetition_context: 16,
    }
}

fn sampled_config(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: sampled_params(),
        seed: Some(SAMPLED_SEED),
        stop_tokens: Vec::new(),
    }
}

fn host(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn argmax(row: &[f32]) -> i32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as i32
}

fn tokens_of(out: &[StreamEvent]) -> Vec<i32> {
    out.iter()
        .filter_map(|e| match e {
            StreamEvent::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

/// Deterministic weight source: every tensor drawn from one seeded stream in insertion order.
struct Draw(SplitMix64, f32);

impl Draw {
    fn new(seed: u64) -> Self {
        Self(SplitMix64::new(seed), 0.5)
    }

    /// Wider weights: a tiny decoder whose greedy run would otherwise settle on one token.
    fn wide(seed: u64, scale: f32) -> Self {
        Self(SplitMix64::new(seed), scale)
    }

    fn rand(&mut self, dims: &[usize]) -> Tensor {
        let n: usize = dims.iter().product();
        let scale = self.1;
        let data: Vec<f32> = (0..n).map(|_| (self.0.next_f32() - 0.5) * scale).collect();
        Tensor::from_vec(data, dims.to_vec(), &Device::Cpu).unwrap()
    }

    /// A norm weight near one (so a wrong norm is visible, unlike an all-ones fixture).
    fn norm(&mut self, d: usize) -> Tensor {
        let data: Vec<f32> = (0..d)
            .map(|_| 1.0 + (self.0.next_f32() - 0.5) * 0.2)
            .collect();
        Tensor::from_vec(data, (d,), &Device::Cpu).unwrap()
    }
}

// ---- the llama family (CausalLm) ---------------------------------------------------------------

const VOCAB: usize = 48;
const HIDDEN: usize = 32;
const INTER: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;

/// A tiny `LlamaForCausalLM` (head_dim = hidden / heads) or `Qwen3ForCausalLM` (explicit head_dim
/// 16, per-head q/k RMSNorm — the Qwen3-8B block shape).
fn tiny_causal(qwen3: bool, seed: u64) -> CausalLm {
    let head_dim = if qwen3 { 16 } else { HIDDEN / HEADS };
    let cfg = if qwen3 {
        json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 3,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": head_dim,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
            "max_position_embeddings": 128, "tie_word_embeddings": false, "eos_token_id": 0
        })
    } else {
        json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 2,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 128, "tie_word_embeddings": false, "eos_token_id": 0
        })
    };
    let layers = if qwen3 { 3 } else { 2 };
    let mut d = Draw::new(seed);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert("model.embed_tokens.weight".into(), d.rand(&[VOCAB, HIDDEN]));
    w.insert("model.norm.weight".into(), d.norm(HIDDEN));
    w.insert("lm_head.weight".into(), d.rand(&[VOCAB, HIDDEN]));
    let (qd, kvd) = (HEADS * head_dim, KV_HEADS * head_dim);
    for i in 0..layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        w.insert(p("input_layernorm.weight"), d.norm(HIDDEN));
        w.insert(p("post_attention_layernorm.weight"), d.norm(HIDDEN));
        w.insert(p("self_attn.q_proj.weight"), d.rand(&[qd, HIDDEN]));
        w.insert(p("self_attn.k_proj.weight"), d.rand(&[kvd, HIDDEN]));
        w.insert(p("self_attn.v_proj.weight"), d.rand(&[kvd, HIDDEN]));
        w.insert(p("self_attn.o_proj.weight"), d.rand(&[HIDDEN, qd]));
        if qwen3 {
            w.insert(p("self_attn.q_norm.weight"), d.norm(head_dim));
            w.insert(p("self_attn.k_norm.weight"), d.norm(head_dim));
        }
        w.insert(p("mlp.gate_proj.weight"), d.rand(&[INTER, HIDDEN]));
        w.insert(p("mlp.up_proj.weight"), d.rand(&[INTER, HIDDEN]));
        w.insert(p("mlp.down_proj.weight"), d.rand(&[HIDDEN, INTER]));
    }
    let cfg = ModelConfig::from_json(&cfg).unwrap();
    CausalLm::from_weights(&Weights::from_map(w, Device::Cpu), "", cfg).unwrap()
}

/// The reference (pre-migration) decode of a `CausalLm`: `generate` for the tokens, and the same
/// greedy walk through `decode_logits` for the rows each token came from — with the pre-migration
/// `Expanded` arithmetic selected explicitly (the default is `Gqa`), restored afterwards.
fn causal_before(model: &mut CausalLm, prompt: &[i32]) -> Golden {
    let selected = model.attn_formulation();
    model.set_attn_formulation(AttnFormulation::Expanded);
    let golden = causal_reference(model, prompt);
    model.set_attn_formulation(selected);
    golden
}

/// The `CausalLm` reference loop's decode in the model's selected formulation: `generate` for the
/// tokens (greedy and sampled), the same greedy walk through `decode_logits` for the rows.
fn causal_reference(model: &CausalLm, prompt: &[i32]) -> Golden {
    let greedy = generate(
        model,
        prompt,
        &greedy_config(NEW_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens;
    let mut cache = model.new_cache();
    let mut logits = model
        .decode_logits(&input_ids(prompt, &Device::Cpu).unwrap(), &mut cache, 0)
        .unwrap();
    let mut rows = Vec::new();
    let mut walked = Vec::new();
    for _ in 0..NEW_TOKENS {
        let row = host(&logits);
        let t = argmax(&row);
        rows.push(row);
        walked.push(t);
        let offset = cache.offset();
        logits = model
            .decode_logits(&input_ids(&[t], &Device::Cpu).unwrap(), &mut cache, offset)
            .unwrap();
    }
    assert_eq!(
        walked, greedy,
        "the greedy walk is the reference loop's greedy decode"
    );
    let sampled = generate(
        model,
        prompt,
        &sampled_config(NEW_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens;
    Golden {
        prompt: prompt.to_vec(),
        greedy_tokens: greedy,
        greedy_logits: rows,
        sampled_tokens: sampled,
    }
}

const LLAMA_PROMPT: [i32; 8] = [3, 17, 5, 29, 3, 17, 5, 11];

#[test]
fn llama_reference_matches_its_pre_migration_golden() {
    let mut model = tiny_causal(false, 0x0001_1A4A);
    golden("llama", causal_before(&mut model, &LLAMA_PROMPT));
}

#[test]
fn qwen3_dense_reference_matches_its_pre_migration_golden() {
    let mut model = tiny_causal(true, 0x03E3);
    golden("qwen3_dense", causal_before(&mut model, &LLAMA_PROMPT));
}

// ---- Gemma 4 (the shared decoder fixture) -------------------------------------------------------

const SCALE_LLAVA: f32 = 1.5;
const SCALE_SC2: f32 = 1.5;
const GEMMA4_WIDEN: f64 = 10.0;
const GEMMA4_PROMPT: [i32; 9] = [1, 9, 4, 13, 2, 7, 11, 5, 3];
const GEMMA4_GOLDENS: &str = include_str!("../../testdata/gemma4/gemma4_decoder_goldens.json");

fn gemma4() -> CausalLm {
    let g: Value = serde_json::from_str(GEMMA4_GOLDENS).unwrap();
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
        // The fixture's weights are small enough that its greedy run settles on one token; the
        // projections are widened (norms and scalars left alone) so the decode is informative.
        let widen = !(key.contains("norm") || key.contains("layer_scalar"));
        let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
        let t = if widen {
            t.affine(GEMMA4_WIDEN, 0.0).unwrap()
        } else {
            t
        };
        w.insert(key.clone(), t);
    }
    let cfg = ModelConfig::from_json(&g["config"]).unwrap();
    CausalLm::from_weights(&Weights::from_map(w, Device::Cpu), "", cfg).unwrap()
}

/// A varied prompt longer than the fixture's sliding window (3), so decode steps attend a window
/// that slides over the cache.
fn gemma4_prompt() -> Vec<i32> {
    GEMMA4_PROMPT.to_vec()
}

#[test]
fn gemma4_reference_matches_its_pre_migration_golden() {
    let mut model = gemma4();
    golden("gemma4", causal_before(&mut model, &gemma4_prompt()));
}

/// The `gemma4_mm` splice, standing in for the vision/audio embedders: the prompt's embeddings
/// with rows `2..5` overwritten by deterministic "soft token" features, prefilled on ordinary 1-D
/// positions exactly as the provider's Gemma 4 multimodal branch does.
fn gemma4_mm_embeds(model: &CausalLm, prompt: &[i32]) -> Tensor {
    let embeds = model
        .embed(&input_ids(prompt, &Device::Cpu).unwrap())
        .unwrap();
    let hidden = embeds.dim(2).unwrap();
    let features = Draw::new(0x0006_E33A).rand(&[1, 3, hidden]);
    let head = embeds.narrow(1, 0, 2).unwrap();
    let tail = embeds.narrow(1, 5, prompt.len() - 5).unwrap();
    Tensor::cat(&[&head, &features, &tail], 1).unwrap()
}

/// The provider's Gemma 4 multimodal decode before the migration: an embeddings prefill into the
/// reference cache, then the shared reference loop.
fn embeds_reference(
    model: &CausalLm,
    embeds: &Tensor,
    history: &[i32],
    config: &GenerationConfig,
) -> Vec<i32> {
    let mut cache = model.new_cache();
    let first = model
        .decode_logits_from_embeds(embeds, &mut cache, 0)
        .unwrap();
    let mut events = Vec::new();
    generate_from_prefill_with_stop(
        model,
        &mut cache,
        first,
        history.to_vec(),
        config,
        &CancelFlag::new(),
        &mut |e| events.push(e),
        None,
        None,
    )
    .unwrap();
    tokens_of(&events)
}

/// The Gemma 4 splice's pre-migration decode, with the `Expanded` arithmetic selected explicitly
/// (restored afterwards).
fn gemma4_mm_before(model: &mut CausalLm, prompt: &[i32]) -> Golden {
    let selected = model.attn_formulation();
    model.set_attn_formulation(AttnFormulation::Expanded);
    let golden = gemma4_mm_reference(model, prompt);
    model.set_attn_formulation(selected);
    golden
}

fn gemma4_mm_reference(model: &CausalLm, prompt: &[i32]) -> Golden {
    let embeds = gemma4_mm_embeds(model, prompt);
    let greedy = embeds_reference(model, &embeds, prompt, &greedy_config(NEW_TOKENS));
    let mut cache = model.new_cache();
    let mut logits = model
        .decode_logits_from_embeds(&embeds, &mut cache, 0)
        .unwrap();
    let mut rows = Vec::new();
    for &t in &greedy {
        rows.push(host(&logits));
        let offset = cache.offset();
        logits = model
            .decode_logits(&input_ids(&[t], &Device::Cpu).unwrap(), &mut cache, offset)
            .unwrap();
    }
    let sampled = embeds_reference(model, &embeds, prompt, &sampled_config(NEW_TOKENS));
    Golden {
        prompt: prompt.to_vec(),
        greedy_tokens: greedy,
        greedy_logits: rows,
        sampled_tokens: sampled,
    }
}

#[test]
fn gemma4_mm_reference_matches_its_pre_migration_golden() {
    let mut model = gemma4();
    golden("gemma4_mm", gemma4_mm_before(&mut model, &gemma4_prompt()));
}

// ---- LLaVA -------------------------------------------------------------------------------------

const IMG_TOKEN: i32 = 7;
const V_IMG: usize = 8;
const V_PATCH: usize = 4;
const V_HIDDEN: usize = 16;

/// A tiny `LlavaForConditionalGeneration` (SigLIP 8×8 / 4×4 patches, 2-layer Llama decoder).
fn tiny_llava() -> LlavaModel {
    let guard = tempfile::Builder::new()
        .prefix("candle-llm-sc24138-llava-")
        .tempdir()
        .unwrap();
    let dir = guard.path();
    let cfg = json!({
        "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
        "image_token_index": IMG_TOKEN, "vision_feature_layer": -1,
        "vision_feature_select_strategy": "full", "projector_hidden_act": "gelu",
        "vision_config": {
            "image_size": V_IMG, "patch_size": V_PATCH, "num_channels": 3,
            "hidden_size": V_HIDDEN, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "layer_norm_eps": 1e-6
        },
        "text_config": {
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 2,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 128, "tie_word_embeddings": false, "eos_token_id": 0
        }
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let mut d = Draw::wide(0x0001_1A7A, SCALE_LLAVA);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    let vp = |s: &str| format!("vision_tower.vision_model.{s}");
    let patches = (V_IMG / V_PATCH) * (V_IMG / V_PATCH);
    w.insert(
        vp("embeddings.patch_embedding.weight"),
        d.rand(&[V_HIDDEN, 3, V_PATCH, V_PATCH]),
    );
    w.insert(vp("embeddings.patch_embedding.bias"), d.rand(&[V_HIDDEN]));
    w.insert(
        vp("embeddings.position_embedding.weight"),
        d.rand(&[patches, V_HIDDEN]),
    );
    let lp = |s: &str| vp(&format!("encoder.layers.0.{s}"));
    for norm in ["layer_norm1", "layer_norm2"] {
        w.insert(lp(&format!("{norm}.weight")), d.norm(V_HIDDEN));
        w.insert(lp(&format!("{norm}.bias")), d.rand(&[V_HIDDEN]));
    }
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        w.insert(
            lp(&format!("self_attn.{proj}.weight")),
            d.rand(&[V_HIDDEN, V_HIDDEN]),
        );
        w.insert(lp(&format!("self_attn.{proj}.bias")), d.rand(&[V_HIDDEN]));
    }
    w.insert(lp("mlp.fc1.weight"), d.rand(&[32, V_HIDDEN]));
    w.insert(lp("mlp.fc1.bias"), d.rand(&[32]));
    w.insert(lp("mlp.fc2.weight"), d.rand(&[V_HIDDEN, 32]));
    w.insert(lp("mlp.fc2.bias"), d.rand(&[V_HIDDEN]));
    w.insert(vp("post_layernorm.weight"), d.norm(V_HIDDEN));
    w.insert(vp("post_layernorm.bias"), d.rand(&[V_HIDDEN]));
    w.insert(
        "multi_modal_projector.linear_1.weight".into(),
        d.rand(&[HIDDEN, V_HIDDEN]),
    );
    w.insert(
        "multi_modal_projector.linear_1.bias".into(),
        d.rand(&[HIDDEN]),
    );
    w.insert(
        "multi_modal_projector.linear_2.weight".into(),
        d.rand(&[HIDDEN, HIDDEN]),
    );
    w.insert(
        "multi_modal_projector.linear_2.bias".into(),
        d.rand(&[HIDDEN]),
    );
    let lm = |s: &str| format!("language_model.{s}");
    let head_dim = HIDDEN / HEADS;
    let (qd, kvd) = (HEADS * head_dim, KV_HEADS * head_dim);
    w.insert(lm("model.embed_tokens.weight"), d.rand(&[VOCAB, HIDDEN]));
    w.insert(lm("model.norm.weight"), d.norm(HIDDEN));
    w.insert(lm("lm_head.weight"), d.rand(&[VOCAB, HIDDEN]));
    for i in 0..2 {
        let p = |s: &str| lm(&format!("model.layers.{i}.{s}"));
        w.insert(p("input_layernorm.weight"), d.norm(HIDDEN));
        w.insert(p("post_attention_layernorm.weight"), d.norm(HIDDEN));
        w.insert(p("self_attn.q_proj.weight"), d.rand(&[qd, HIDDEN]));
        w.insert(p("self_attn.k_proj.weight"), d.rand(&[kvd, HIDDEN]));
        w.insert(p("self_attn.v_proj.weight"), d.rand(&[kvd, HIDDEN]));
        w.insert(p("self_attn.o_proj.weight"), d.rand(&[HIDDEN, qd]));
        w.insert(p("mlp.gate_proj.weight"), d.rand(&[INTER, HIDDEN]));
        w.insert(p("mlp.up_proj.weight"), d.rand(&[INTER, HIDDEN]));
        w.insert(p("mlp.down_proj.weight"), d.rand(&[HIDDEN, INTER]));
    }
    candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
    LlavaModel::from_dir(dir, &Device::Cpu).unwrap()
}

/// A deterministic two-colour 8×8 RGB image.
fn llava_image() -> Vec<u8> {
    (0..V_IMG * V_IMG)
        .flat_map(|i| {
            if i % V_IMG < V_IMG / 2 {
                [200u8, 40, 30]
            } else {
                [20u8, 90, 210]
            }
        })
        .collect()
}

const LLAVA_PROMPT: [i32; 6] = [1, IMG_TOKEN, 12, 19, 4, 22];

fn llava_caption(
    model: &LlavaModel,
    features: &Tensor,
    params: &SamplingParams,
    seed: u64,
) -> Vec<i32> {
    let mut streamed = Vec::new();
    let out = model
        .generate(
            &LLAVA_PROMPT,
            features,
            params,
            NEW_TOKENS,
            Some(seed),
            &[],
            &CancelFlag::new(),
            &mut |id, _| streamed.push(id),
        )
        .unwrap();
    assert_eq!(out.tokens, streamed, "every generated token is streamed");
    out.tokens
}

/// LLaVA's caption before the migration, with the language decoder's `Expanded` arithmetic
/// selected explicitly through [`LlavaModel::set_attn_formulation`] (restored afterwards).
fn llava_before(model: &mut LlavaModel) -> Golden {
    let selected = model.language().attn_formulation();
    model.set_attn_formulation(AttnFormulation::Expanded);
    let golden = llava_reference(model);
    model.set_attn_formulation(selected);
    golden
}

/// LLaVA's caption (`LlavaModel::generate`, now through the seam on the growing backing) and the
/// rows its greedy tokens came from, in the language decoder's selected formulation.
fn llava_reference(model: &LlavaModel) -> Golden {
    let features = model.image_features(&llava_image(), V_IMG, V_IMG).unwrap();
    let greedy = llava_caption(model, &features, &SamplingParams::default(), 0);
    // The rows the caption loop chose from: the spliced prefill, then single-token steps.
    let expanded = candle_llm::llava::expand_image_tokens(
        &LLAVA_PROMPT,
        IMG_TOKEN,
        model.config().image_seq_length,
    );
    let lang = model.language();
    let embeds = lang
        .embed(&input_ids(&expanded, &Device::Cpu).unwrap())
        .unwrap();
    let spliced = candle_llm::llava::splice_image_features(
        &embeds,
        &expanded,
        &features.to_dtype(lang.compute_dtype()).unwrap(),
        IMG_TOKEN,
    )
    .unwrap();
    let mut cache = lang.new_cache();
    let mut logits = lang
        .decode_logits_from_embeds(&spliced, &mut cache, 0)
        .unwrap();
    let mut rows = Vec::new();
    for &t in &greedy {
        rows.push(host(&logits));
        let offset = cache.offset();
        logits = lang
            .decode_logits(&input_ids(&[t], &Device::Cpu).unwrap(), &mut cache, offset)
            .unwrap();
    }
    let sampled = llava_caption(model, &features, &sampled_params(), SAMPLED_SEED);
    Golden {
        prompt: LLAVA_PROMPT.to_vec(),
        greedy_tokens: greedy,
        greedy_logits: rows,
        sampled_tokens: sampled,
    }
}

#[test]
fn llava_caption_matches_its_pre_migration_golden() {
    let mut model = tiny_llava();
    golden("llava", llava_before(&mut model));
}

// ---- StarCoder2 (the StarVector-8B decoder) -------------------------------------------------------

const SC2: StarCoder2Config = StarCoder2Config {
    vocab_size: VOCAB,
    hidden_size: HIDDEN,
    intermediate_size: INTER,
    layers: 2,
    heads: HEADS,
    kv_heads: KV_HEADS,
    rope_theta: 1_000_000.0,
    layer_norm_eps: 1e-5,
};

fn tiny_starcoder2() -> StarCoder2 {
    let mut d = Draw::wide(0x05C2, SCALE_SC2);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    let p = "model.svg_transformer.transformer";
    let k = |s: &str| format!("{p}.{s}");
    w.insert(k("model.embed_tokens.weight"), d.rand(&[VOCAB, HIDDEN]));
    w.insert(k("model.norm.weight"), d.norm(HIDDEN));
    w.insert(k("model.norm.bias"), d.rand(&[HIDDEN]));
    let head_dim = HIDDEN / HEADS;
    for i in 0..SC2.layers {
        let l = |s: &str| k(&format!("model.layers.{i}.{s}"));
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            w.insert(l(&format!("{norm}.weight")), d.norm(HIDDEN));
            w.insert(l(&format!("{norm}.bias")), d.rand(&[HIDDEN]));
        }
        for (proj, rows) in [
            ("q_proj", HEADS * head_dim),
            ("k_proj", KV_HEADS * head_dim),
            ("v_proj", KV_HEADS * head_dim),
            ("o_proj", HIDDEN),
        ] {
            let cols = if proj == "o_proj" {
                HEADS * head_dim
            } else {
                HIDDEN
            };
            w.insert(
                l(&format!("self_attn.{proj}.weight")),
                d.rand(&[rows, cols]),
            );
            w.insert(l(&format!("self_attn.{proj}.bias")), d.rand(&[rows]));
        }
        w.insert(l("mlp.c_fc.weight"), d.rand(&[INTER, HIDDEN]));
        w.insert(l("mlp.c_fc.bias"), d.rand(&[INTER]));
        w.insert(l("mlp.c_proj.weight"), d.rand(&[HIDDEN, INTER]));
        w.insert(l("mlp.c_proj.bias"), d.rand(&[HIDDEN]));
    }
    StarCoder2::from_weights(&Weights::from_map(w, Device::Cpu), p, SC2).unwrap()
}

const SVG_PROMPT: [i32; 2] = [9, 31];
const VISION_ROWS: usize = 5;

/// The StarVector-8B prefill: projected "image" rows followed by the `<svg` prompt embeddings.
fn starcoder2_embeds(model: &StarCoder2) -> Tensor {
    let vision = Draw::new(0x5C21).rand(&[1, VISION_ROWS, HIDDEN]);
    let text = model
        .embed(&input_ids(&SVG_PROMPT, &Device::Cpu).unwrap())
        .unwrap();
    Tensor::cat(&[&vision, &text], 1).unwrap()
}

fn starcoder2_reference(model: &StarCoder2, config: &GenerationConfig) -> Vec<i32> {
    let mut cache: Box<dyn KvCache> = Box::new(model.cache());
    let first = model
        .logits_from_embeds(&starcoder2_embeds(model), cache.as_mut(), 0)
        .unwrap();
    let mut events = Vec::new();
    generate_from_prefill_with_stop(
        model,
        cache.as_mut(),
        first,
        SVG_PROMPT.to_vec(),
        config,
        &CancelFlag::new(),
        &mut |e| events.push(e),
        None,
        None,
    )
    .unwrap();
    tokens_of(&events)
}

/// StarCoder2's decode before the migration, with the `Expanded` arithmetic selected explicitly
/// (restored afterwards).
fn starcoder2_before(model: &mut StarCoder2) -> Golden {
    let selected = model.attn_formulation();
    model.set_attn_formulation(AttnFormulation::Expanded);
    let golden = starcoder2_golden_walk(model);
    model.set_attn_formulation(selected);
    golden
}

/// The reference loop's greedy and sampled decode plus the greedy rows, in the model's selected
/// formulation.
fn starcoder2_golden_walk(model: &StarCoder2) -> Golden {
    let greedy = starcoder2_reference(model, &greedy_config(NEW_TOKENS));
    let mut cache = model.cache();
    let mut logits = model
        .logits_from_embeds(&starcoder2_embeds(model), &mut cache, 0)
        .unwrap();
    let mut rows = Vec::new();
    for &t in &greedy {
        rows.push(host(&logits));
        let offset = cache.offset();
        let embed = model
            .embed(&input_ids(&[t], &Device::Cpu).unwrap())
            .unwrap();
        logits = model
            .logits_from_embeds(&embed, &mut cache, offset)
            .unwrap();
    }
    Golden {
        prompt: SVG_PROMPT.to_vec(),
        greedy_tokens: greedy,
        greedy_logits: rows,
        sampled_tokens: starcoder2_reference(model, &sampled_config(NEW_TOKENS)),
    }
}

#[test]
fn starcoder2_reference_matches_its_pre_migration_golden() {
    let mut model = tiny_starcoder2();
    golden("starcoder2", starcoder2_before(&mut model));
}

// ---- StarVector-1B (GPTBigCode, multi-query) ------------------------------------------------------

const SV1_GEOMETRY: StarVectorDecoderGeometry = StarVectorDecoderGeometry {
    hidden: HIDDEN,
    heads: HEADS,
    head_dim: HIDDEN / HEADS,
    layers: 2,
    max_positions: 64,
};

fn tiny_starvector_1b_weights() -> Weights {
    let mut d = Draw::new(0x051B);
    let g = SV1_GEOMETRY;
    let mut w: HashMap<String, Tensor> = HashMap::new();
    let p = "model.svg_transformer.transformer.transformer";
    let k = |s: &str| format!("{p}.{s}");
    w.insert(k("wte.weight"), d.rand(&[VOCAB, g.hidden]));
    w.insert(k("wpe.weight"), d.rand(&[g.max_positions, g.hidden]));
    w.insert(k("ln_f.weight"), d.norm(g.hidden));
    w.insert(k("ln_f.bias"), d.rand(&[g.hidden]));
    let qkv = g.hidden + 2 * g.head_dim;
    for i in 0..g.layers {
        let l = |s: &str| k(&format!("h.{i}.{s}"));
        for norm in ["ln_1", "ln_2"] {
            w.insert(l(&format!("{norm}.weight")), d.norm(g.hidden));
            w.insert(l(&format!("{norm}.bias")), d.rand(&[g.hidden]));
        }
        w.insert(l("attn.c_attn.weight"), d.rand(&[qkv, g.hidden]));
        w.insert(l("attn.c_attn.bias"), d.rand(&[qkv]));
        w.insert(l("attn.c_proj.weight"), d.rand(&[g.hidden, g.hidden]));
        w.insert(l("attn.c_proj.bias"), d.rand(&[g.hidden]));
        w.insert(l("mlp.c_fc.weight"), d.rand(&[INTER, g.hidden]));
        w.insert(l("mlp.c_fc.bias"), d.rand(&[INTER]));
        w.insert(l("mlp.c_proj.weight"), d.rand(&[g.hidden, INTER]));
        w.insert(l("mlp.c_proj.bias"), d.rand(&[g.hidden]));
    }
    Weights::from_map(w, Device::Cpu)
}

fn starvector_1b_embeds(decoder: &StarVectorDecoder) -> Tensor {
    let vision = Draw::new(0x51B1).rand(&[1, VISION_ROWS, HIDDEN]);
    let text = decoder
        .embeddings(&input_ids(&SVG_PROMPT, &Device::Cpu).unwrap())
        .unwrap();
    Tensor::cat(&[&vision, &text], 1).unwrap()
}

/// The StarVector-1B provider's caption loop before the migration: prefill the joined embeddings,
/// then sample / feed one token at a time through the decoder's layer-owned cache. Returns the
/// tokens and the rows they were drawn from.
///
/// Since the migration the decoder is stateless and the K/V live in the caller's cache: the same
/// loop over the step cache's growing backing (the reference concat) must still reproduce the
/// golden bit for bit.
fn starvector_1b_loop(
    decoder: &StarVectorDecoder,
    params: &SamplingParams,
    seed: u64,
) -> (Vec<i32>, Vec<Vec<f32>>) {
    let mut cache = decoder.new_step_cache();
    let embeds = starvector_1b_embeds(decoder);
    let mut logits = decoder.forward_embeds(&embeds, &mut cache).unwrap();
    let mut history = SVG_PROMPT.to_vec();
    let mut rng = SplitMix64::new(seed);
    let (mut tokens, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..NEW_TOKENS {
        rows.push(host(&logits));
        let id = sample(&logits, &history, params, &mut rng, None).unwrap();
        history.push(id);
        tokens.push(id);
        let embed = decoder
            .embeddings(&input_ids(&[id], &Device::Cpu).unwrap())
            .unwrap();
        logits = decoder.forward_embeds(&embed, &mut cache).unwrap();
    }
    (tokens, rows)
}

fn tiny_starvector_1b() -> StarVectorDecoder {
    StarVectorDecoder::from_weights_with_geometry(&tiny_starvector_1b_weights(), SV1_GEOMETRY)
        .unwrap()
}

#[test]
fn starvector_1b_decoder_matches_its_pre_migration_golden() {
    let decoder = tiny_starvector_1b();
    let (greedy, rows) = starvector_1b_loop(&decoder, &SamplingParams::default(), 0);
    let (sampled, _) = starvector_1b_loop(&decoder, &sampled_params(), SAMPLED_SEED);
    golden(
        "starvector_1b",
        Golden {
            prompt: SVG_PROMPT.to_vec(),
            greedy_tokens: greedy,
            greedy_logits: rows,
            sampled_tokens: sampled,
        },
    );
}

// =================================================================================================
// After the migration: the step seam, the engine and the multimodal prefill paths vs the goldens
// =================================================================================================

/// Largest per-logit absolute difference allowed between the static cache's un-expanded attention
/// and the golden's `repeat_kv`-expanded rows on CPU f32. The observed maximum is two orders of
/// magnitude below this; a wrong cache (a stale or misplaced position) moves logits by O(0.1).
const STATIC_LOGIT_TOL: f32 = 1e-4;

/// One parity-table line on stderr (`--nocapture` shows it; the evidence table is built from it).
fn report(model: &str, path: &str, diff: f32) {
    eprintln!("[sc24138-parity] {model:<14} {path:<8} tokens=identical max|dlogit|={diff:.3e}");
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len(), "row count");
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            assert_eq!(x.len(), y.len(), "row width");
            x.iter().zip(y).map(|(p, q)| (p - q).abs())
        })
        .fold(0.0f32, f32::max)
}

/// Walk `tokens` through the step seam from a prefilled `cache` whose prefill produced `first`,
/// returning the row each token was chosen from (the prefill row first).
fn step_walk<M: StepModel>(
    model: &M,
    cache: &mut M::Cache,
    first: Tensor,
    tokens: &[i32],
) -> Vec<Vec<f32>> {
    let mut rows = vec![host(&first)];
    for &t in &tokens[..tokens.len() - 1] {
        let out = model.forward_step(cache, StepRequest::last(&[t])).unwrap();
        rows.push(host(&out.logits));
    }
    rows
}

/// A token prompt through the seam: prefill + walk.
fn step_rows<M: StepModel>(model: &M, cache: &mut M::Cache, g: &Golden) -> Vec<Vec<f32>> {
    let first = model
        .forward_step(cache, StepRequest::last(&g.prompt))
        .unwrap()
        .logits;
    step_walk(model, cache, first, &g.greedy_tokens)
}

fn greedy_step(model: &CausalLm, g: &Golden, config: &GenerationConfig) -> Vec<i32> {
    generate_step(
        model,
        &g.prompt,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap()
    .0
    .tokens
}

/// Every seam path of a `CausalLm` held to its golden. `gqa` says whether every layer can attend
/// un-expanded (Gemma 4's sliding layers cannot).
fn causal_after(name: &str, model: &mut CausalLm, g: &Golden, gqa: bool) {
    // -- the static cache (the default step cache) --
    assert_eq!(model.step_kv_cache(), KvCacheKind::Static);
    let (out, record) = generate_step(
        model,
        &g.prompt,
        &greedy_config(NEW_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(out.tokens, g.greedy_tokens, "static step greedy tokens");
    assert_eq!(record.path, DecodePath::StepModel);
    assert_eq!(
        record.kv_cache,
        KvCacheKind::Static,
        "telemetry names the cache"
    );
    let formulation = if gqa {
        AttnFormulation::Gqa
    } else {
        AttnFormulation::Expanded
    };
    assert_eq!(
        record.attn_formulation, formulation,
        "telemetry names the arithmetic"
    );
    assert_eq!(
        greedy_step(model, g, &sampled_config(NEW_TOKENS)),
        g.sampled_tokens,
        "static step sampled tokens"
    );
    let capacity = g.prompt.len() + NEW_TOKENS;
    let mut fixed = model.new_cache_for(capacity, 0).unwrap();
    assert_eq!(fixed.kv_capacity(), Some(capacity));
    assert_eq!(
        fixed.memory().live_bytes,
        model.static_kv_bytes(capacity),
        "the preallocation is exactly the priced bytes"
    );
    let rows = step_rows(model, &mut fixed, g);
    let diff = max_abs_diff(&rows, &g.greedy_logits);
    report(name, "static", diff);
    assert!(diff <= STATIC_LOGIT_TOL, "static logits drift {diff}");

    // -- the growing backing through the seam, default formulation: the static cache's
    //    arithmetic, bit for bit, and the same label --
    assert_eq!(
        model.attn_formulation(),
        AttnFormulation::Gqa,
        "un-expanded is the default"
    );
    model.set_step_kv_cache(KvCacheKind::Growing);
    let (out, record) = generate_step(
        model,
        &g.prompt,
        &greedy_config(NEW_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(out.tokens, g.greedy_tokens, "growing step greedy tokens");
    assert_eq!(record.kv_cache, KvCacheKind::Growing);
    assert_eq!(
        record.attn_formulation, formulation,
        "the default growing cache attends like the static one"
    );
    let mut growing = model.new_step_cache();
    // Built explicitly: `new_cache_for` follows the growing selection made above.
    let mut fixed = model.new_static_cache(capacity).unwrap();
    assert_eq!(fixed.kv_kind(), KvCacheKind::Static);
    assert_eq!(
        step_rows(model, &mut growing, g),
        step_rows(model, &mut fixed, g),
        "{name}: the default growing backing is the static cache's arithmetic, bit for bit"
    );

    // -- the growing backing with `Expanded` selected: the pre-migration arithmetic, the golden
    //    (bit for bit in the measured configuration) --
    model.set_attn_formulation(AttnFormulation::Expanded);
    let (out, record) = generate_step(
        model,
        &g.prompt,
        &greedy_config(NEW_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(
        out.tokens, g.greedy_tokens,
        "growing expanded greedy tokens"
    );
    assert_eq!(record.attn_formulation, AttnFormulation::Expanded);
    let mut growing = model.new_step_cache();
    assert_golden_rows(
        &format!("{name}: growing, expanded"),
        &step_rows(model, &mut growing, g),
        &g.greedy_logits,
    );
    model.set_attn_formulation(AttnFormulation::Gqa);
    model.set_step_kv_cache(KvCacheKind::Static);

    // -- the engine: no proposer, prompt lookup, and the model as its own draft --
    let config = greedy_config(NEW_TOKENS);
    let run = generate_speculative(
        model,
        &mut NoProposer,
        SpeculativePrompt::Tokens(&g.prompt),
        &config,
        3,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(run.output.tokens, g.greedy_tokens, "engine, no proposer");
    let run = generate_speculative(
        model,
        &mut NgramProposer { max_ngram: 3 },
        SpeculativePrompt::Tokens(&g.prompt),
        &config,
        3,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(
        run.output.tokens, g.greedy_tokens,
        "engine, n-gram proposer"
    );
    assert_eq!(run.record.kv_cache, KvCacheKind::Static);
    let mut draft = DraftModelProposer::new(&*model, capacity, 3);
    let run = generate_speculative(
        model,
        &mut draft,
        SpeculativePrompt::Tokens(&g.prompt),
        &config,
        3,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(run.output.tokens, g.greedy_tokens, "engine, self-draft");
    assert_eq!(
        run.stats.accepted, run.stats.proposed,
        "a draft identical to the target is always accepted"
    );

    // -- the default reference loop: the static cache's arithmetic --
    // The `CausalLm` reference loop in its default formulation produces the golden's greedy and
    // sampled tokens, and the rows it chose them from are the static cache's, bit for bit — the
    // reference and the fast path are one arithmetic on two caches.
    let reference = causal_reference(model, &g.prompt);
    assert_eq!(
        reference.greedy_tokens, g.greedy_tokens,
        "{name}: default reference loop, greedy"
    );
    assert_eq!(
        reference.sampled_tokens, g.sampled_tokens,
        "{name}: default reference loop, sampled"
    );
    let mut fixed = model.new_static_cache(capacity).unwrap();
    assert_eq!(
        reference.greedy_logits,
        step_rows(model, &mut fixed, g),
        "{name}: the default reference loop is the static cache's arithmetic, bit for bit"
    );
}

#[test]
fn llama_decodes_to_its_golden_on_every_seam_path() {
    let mut model = tiny_causal(false, 0x0001_1A4A);
    let g = golden("llama", causal_before(&mut model, &LLAMA_PROMPT));
    causal_after("llama", &mut model, &g, true);
    // The paged backing, kept behind the same seam: the static cache's arithmetic by default, the
    // golden with `Expanded` selected.
    let mut paged = model.new_paged_step_cache(4);
    let mut fixed = model.new_static_cache(g.prompt.len() + NEW_TOKENS).unwrap();
    assert_eq!(
        step_rows(&model, &mut paged, &g),
        step_rows(&model, &mut fixed, &g),
        "paged (default) = static, bit for bit"
    );
    assert!(paged.as_paged().is_some());
    model.set_attn_formulation(AttnFormulation::Expanded);
    let mut paged = model.new_paged_step_cache(4);
    assert_golden_rows(
        "llama: paged, expanded",
        &step_rows(&model, &mut paged, &g),
        &g.greedy_logits,
    );
}

#[test]
fn qwen3_dense_decodes_to_its_golden_on_every_seam_path() {
    let mut model = tiny_causal(true, 0x03E3);
    let g = golden("qwen3_dense", causal_before(&mut model, &LLAMA_PROMPT));
    causal_after("qwen3_dense", &mut model, &g, true);
}

#[test]
fn gemma4_decodes_to_its_golden_on_every_seam_path() {
    let mut model = gemma4();
    let g = golden("gemma4", causal_before(&mut model, &gemma4_prompt()));
    causal_after("gemma4", &mut model, &g, false);
}

/// The Gemma 4 soft-token splice through the seam: the spliced embeddings prefill the step cache
/// (static and growing), the continuation decodes through the engine. The default growing
/// backing is the static cache's arithmetic bit for bit; with `Expanded` selected it is the golden.
#[test]
fn gemma4_mm_splice_decodes_to_its_golden_through_the_seam() {
    let mut model = gemma4();
    let prompt = gemma4_prompt();
    let g = golden("gemma4_mm", gemma4_mm_before(&mut model, &prompt));
    let embeds = gemma4_mm_embeds(&model, &prompt);
    let capacity = prompt.len() + NEW_TOKENS;
    let mut static_rows = Vec::new();
    for (backing, static_cache) in [("static", true), ("growing", false)] {
        let fresh = || -> StepKvCache {
            if static_cache {
                model.new_static_cache(capacity).unwrap()
            } else {
                model.new_step_cache()
            }
        };
        for (config, want) in [
            (greedy_config(NEW_TOKENS), &g.greedy_tokens),
            (sampled_config(NEW_TOKENS), &g.sampled_tokens),
        ] {
            let mut cache = fresh();
            let first = model.step_prefill_from_embeds(&embeds, &mut cache).unwrap();
            let (out, record) = generate_step_from_prefill(
                &model,
                &mut cache,
                first,
                &prompt,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap();
            assert_eq!(&out.tokens, want, "{backing}: gemma4_mm tokens");
            assert_eq!(record.path, DecodePath::StepModel);
        }
        let mut cache = fresh();
        let first = model.step_prefill_from_embeds(&embeds, &mut cache).unwrap();
        let rows = step_walk(&model, &mut cache, first, &g.greedy_tokens);
        if static_cache {
            let diff = max_abs_diff(&rows, &g.greedy_logits);
            report("gemma4_mm", "static", diff);
            assert!(diff <= STATIC_LOGIT_TOL, "gemma4_mm static drift {diff}");
            static_rows = rows;
        } else {
            assert_eq!(
                rows, static_rows,
                "gemma4_mm: the default growing backing is the static cache's arithmetic"
            );
        }
    }
    model.set_attn_formulation(AttnFormulation::Expanded);
    let mut cache = model.new_step_cache();
    let first = model.step_prefill_from_embeds(&embeds, &mut cache).unwrap();
    assert_golden_rows(
        "gemma4_mm: growing, expanded",
        &step_walk(&model, &mut cache, first, &g.greedy_tokens),
        &g.greedy_logits,
    );
}

/// LLaVA's caption now decodes through the seam (its own loop is gone): the provider path on the
/// growing backing with `Expanded` selected is the golden (`llava_before`); in the default
/// formulation it gives the golden's tokens, and the static backing gives the same tokens with
/// the default rows bit for bit.
#[test]
fn llava_caption_decodes_to_its_golden_through_the_seam() {
    let mut model = tiny_llava();
    let g = golden("llava", llava_before(&mut model));
    assert_eq!(model.language().attn_formulation(), AttnFormulation::Gqa);
    let default = llava_reference(&model);
    assert_eq!(
        default.greedy_tokens, g.greedy_tokens,
        "llava default greedy"
    );
    assert_eq!(
        default.sampled_tokens, g.sampled_tokens,
        "llava default sampled"
    );
    let features = model.image_features(&llava_image(), V_IMG, V_IMG).unwrap();
    // `llava_before` ran `LlavaModel::generate` (now the seam) and held it to the golden already;
    // here the same splice on the static backing.
    let lang = model.language();
    let expanded = candle_llm::llava::expand_image_tokens(
        &LLAVA_PROMPT,
        IMG_TOKEN,
        model.config().image_seq_length,
    );
    let embeds = lang
        .embed(&input_ids(&expanded, &Device::Cpu).unwrap())
        .unwrap();
    let spliced = candle_llm::llava::splice_image_features(
        &embeds,
        &expanded,
        &features.to_dtype(lang.compute_dtype()).unwrap(),
        IMG_TOKEN,
    )
    .unwrap();
    for (config, want) in [
        (greedy_config(NEW_TOKENS), &g.greedy_tokens),
        (sampled_config(NEW_TOKENS), &g.sampled_tokens),
    ] {
        let mut cache = lang.new_static_cache(expanded.len() + NEW_TOKENS).unwrap();
        let first = lang.step_prefill_from_embeds(&spliced, &mut cache).unwrap();
        let (out, record) = generate_step_from_prefill(
            lang,
            &mut cache,
            first,
            &expanded,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(&out.tokens, want, "llava static tokens");
        assert_eq!(record.kv_cache, KvCacheKind::Static);
        // The caller's prefill counts as the engine's `Prefilled` path counts it (sc-24131).
        assert_eq!(record.prefill_forwards, 1);
    }
    let mut cache = lang.new_static_cache(expanded.len() + NEW_TOKENS).unwrap();
    let first = lang.step_prefill_from_embeds(&spliced, &mut cache).unwrap();
    assert_eq!(
        step_walk(lang, &mut cache, first, &g.greedy_tokens),
        default.greedy_logits,
        "llava: the default caption rows are the static cache's, bit for bit"
    );
    // A cancel that lands after the prefill is the ordinary mid-stream cancellation, as before.
    let cancel = CancelFlag::new();
    let mut cache = lang.new_step_cache();
    let first = lang.step_prefill_from_embeds(&spliced, &mut cache).unwrap();
    cancel.cancel();
    let (out, record) = generate_step_from_prefill(
        lang,
        &mut cache,
        first,
        &expanded,
        &greedy_config(NEW_TOKENS),
        &cancel,
        &mut |_| {},
        None,
    )
    .unwrap();
    assert!(out.tokens.is_empty());
    assert_eq!(
        (record.target_forwards, record.prefill_forwards),
        (1, 1),
        "the caller's prefill is still one of the request's target forwards"
    );
    assert_eq!(
        out.finish_reason,
        candle_llm::decode::FinishReason::Cancelled
    );
}

/// StarCoder2 (StarVector-8B's decoder) through the seam: the conditioning prefix prefills the
/// step cache, the continuation decodes through the engine — the provider's path. The default
/// reference loop and growing backing are the static cache's arithmetic bit for bit; with
/// `Expanded` selected the growing backing is the golden.
#[test]
fn starcoder2_decodes_to_its_golden_through_the_seam() {
    let mut model = tiny_starcoder2();
    let g = golden("starcoder2", starcoder2_before(&mut model));
    assert_eq!(model.attn_formulation(), AttnFormulation::Gqa);
    let embeds = starcoder2_embeds(&model);
    let capacity = VISION_ROWS + SVG_PROMPT.len() + NEW_TOKENS;
    let mut static_rows = Vec::new();
    for static_cache in [true, false] {
        let fresh = || {
            if static_cache {
                model.new_static_cache(capacity).unwrap()
            } else {
                model.new_step_cache()
            }
        };
        for (config, want) in [
            (greedy_config(NEW_TOKENS), &g.greedy_tokens),
            (sampled_config(NEW_TOKENS), &g.sampled_tokens),
        ] {
            let mut cache = fresh();
            let first = model.step_prefill_from_embeds(&embeds, &mut cache).unwrap();
            let (out, _) = generate_step_from_prefill(
                &model,
                &mut cache,
                first,
                &SVG_PROMPT,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap();
            assert_eq!(&out.tokens, want, "starcoder2 static={static_cache}");
        }
        let mut cache = fresh();
        let first = model.step_prefill_from_embeds(&embeds, &mut cache).unwrap();
        let rows = step_walk(&model, &mut cache, first, &g.greedy_tokens);
        assert_eq!(
            StepModel::attn_formulation(&model, &cache),
            AttnFormulation::Gqa,
            "starcoder2 static={static_cache}: un-expanded by default on either backing"
        );
        if static_cache {
            let diff = max_abs_diff(&rows, &g.greedy_logits);
            report("starcoder2", "static", diff);
            assert!(diff <= STATIC_LOGIT_TOL, "starcoder2 static drift {diff}");
            static_rows = rows;
        } else {
            assert_eq!(
                rows, static_rows,
                "starcoder2: the default growing backing is the static cache's arithmetic"
            );
        }
    }
    // The default reference loop (the `Decode` trait) is the same arithmetic.
    let reference = starcoder2_golden_walk(&model);
    assert_eq!(reference.greedy_tokens, g.greedy_tokens);
    assert_eq!(reference.sampled_tokens, g.sampled_tokens);
    assert_eq!(
        reference.greedy_logits, static_rows,
        "starcoder2: the default reference loop is the static cache's arithmetic, bit for bit"
    );
    // `Expanded` selected: the growing backing is the pre-migration arithmetic — the golden.
    model.set_attn_formulation(AttnFormulation::Expanded);
    let mut cache = model.new_step_cache();
    assert_eq!(
        StepModel::attn_formulation(&model, &cache),
        AttnFormulation::Expanded
    );
    let first = model.step_prefill_from_embeds(&embeds, &mut cache).unwrap();
    assert_golden_rows(
        "starcoder2: growing, expanded",
        &step_walk(&model, &mut cache, first, &g.greedy_tokens),
        &g.greedy_logits,
    );
    model.set_attn_formulation(AttnFormulation::Gqa);
    assert_eq!(
        model
            .new_static_cache(capacity)
            .unwrap()
            .memory()
            .live_bytes,
        model.static_kv_bytes(capacity)
    );
}

/// The StarVector-1B decoder through the seam, on both backings: its multi-query fold attends the
/// one shared K/V head un-expanded either way, so both are the golden bit for bit.
#[test]
fn starvector_1b_decodes_to_its_golden_through_the_seam() {
    let decoder = tiny_starvector_1b();
    let (greedy, rows) = starvector_1b_loop(&decoder, &SamplingParams::default(), 0);
    let (sampled, _) = starvector_1b_loop(&decoder, &sampled_params(), SAMPLED_SEED);
    let g = golden(
        "starvector_1b",
        Golden {
            prompt: SVG_PROMPT.to_vec(),
            greedy_tokens: greedy,
            greedy_logits: rows,
            sampled_tokens: sampled,
        },
    );
    let embeds = starvector_1b_embeds(&decoder);
    let capacity = VISION_ROWS + SVG_PROMPT.len() + NEW_TOKENS;
    for static_cache in [true, false] {
        let fresh = || {
            if static_cache {
                decoder.new_static_cache(capacity).unwrap()
            } else {
                decoder.new_step_cache()
            }
        };
        for (config, want) in [
            (greedy_config(NEW_TOKENS), &g.greedy_tokens),
            (sampled_config(NEW_TOKENS), &g.sampled_tokens),
        ] {
            let mut cache = fresh();
            let first = decoder.forward_embeds(&embeds, &mut cache).unwrap();
            let (out, _) = generate_step_from_prefill(
                &decoder,
                &mut cache,
                first,
                &SVG_PROMPT,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap();
            assert_eq!(&out.tokens, want, "starvector-1b static={static_cache}");
        }
        let mut cache = fresh();
        let first = decoder.forward_embeds(&embeds, &mut cache).unwrap();
        assert_golden_rows(
            &format!("starvector-1b static={static_cache} rows"),
            &step_walk(&decoder, &mut cache, first, &g.greedy_tokens),
            &g.greedy_logits,
        );
    }
    // Past the learned positions the static cache fails closed before allocating.
    assert!(matches!(
        decoder.new_static_cache(SV1_GEOMETRY.max_positions + 1),
        Err(candle_llm::Error::KvCapacityExceeded { .. })
    ));
}

/// A static request past `max_position_embeddings` is refused with the typed error before any
/// allocation; a zero-capacity one is refused too.
#[test]
fn causal_static_cache_fails_closed_past_the_model_bound() {
    let model = tiny_causal(false, 0x0001_1A4A);
    assert!(matches!(
        model.new_cache_for(120, 9),
        Err(candle_llm::Error::KvCapacityExceeded {
            requested: 129,
            capacity: 128
        })
    ));
    assert!(model.new_static_cache(0).is_err());
    let cache = model.new_cache_for(120, 8).unwrap();
    assert_eq!(cache.kv_capacity(), Some(128));
}

/// **AC3.** `decode/speculative.rs` holds no decode loop and nothing `CausalLm`-typed; the n-gram
/// and draft-model proposers live in `decode/proposers.rs`.
#[test]
fn speculative_rs_holds_no_decode_loop() {
    let source = include_str!("../src/decode/speculative.rs");
    let code: String = source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for forbidden in [
        "fn generate",
        "fn ",
        "loop {",
        "while ",
        "for ",
        "CausalLm",
        "decode_logits",
        "KvCache",
    ] {
        assert!(
            !code.contains(forbidden),
            "decode/speculative.rs code contains `{forbidden}`"
        );
    }
    assert!(code.contains("pub struct SpeculativeStats"));
    let proposers = include_str!("../src/decode/proposers.rs");
    assert!(proposers.contains("pub struct NgramProposer"));
    assert!(proposers.contains("pub struct DraftModelProposer"));
}

/// A draft whose vocabulary is not the target's is refused before any inference (the check the
/// retired draft-model loop made, now the engine's).
#[test]
fn engine_refuses_a_draft_with_another_vocabulary() {
    let target = tiny_causal(false, 0x0001_1A4A);
    let draft = tiny_causal(true, 0x03E3);
    assert_eq!(target.vocab_size(), draft.vocab_size());
    // Same vocab: accepted. A proposer claiming another vocabulary: refused.
    struct Wide<'a>(DraftModelProposer<'a, CausalLm>);
    impl candle_llm::decode::Proposer for Wide<'_> {
        fn kind(&self) -> core_llm::ProposerKind {
            self.0.kind()
        }
        fn vocab_size(&self) -> Option<usize> {
            Some(VOCAB + 1)
        }
        fn warm(&mut self, p: &[i32], h: Option<&Tensor>) -> candle_llm::Result<()> {
            self.0.warm(p, h)
        }
        fn propose(
            &mut self,
            ctx: &candle_llm::decode::ProposeContext<'_>,
            s: &mut candle_llm::decode::DraftSampler<'_, '_>,
        ) -> candle_llm::Result<candle_llm::decode::Proposal> {
            self.0.propose(ctx, s)
        }
        fn commit(
            &mut self,
            cur: i32,
            accepted: &[i32],
            h: Option<&Tensor>,
            position: i32,
        ) -> candle_llm::Result<()> {
            self.0.commit(cur, accepted, h, position)
        }
    }
    let mut ok = DraftModelProposer::new(&draft, 32, 2);
    assert!(generate_speculative(
        &target,
        &mut ok,
        SpeculativePrompt::Tokens(&LLAMA_PROMPT),
        &greedy_config(4),
        2,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .is_ok());
    let mut wide = Wide(DraftModelProposer::new(&draft, 32, 2));
    let err = generate_speculative(
        &target,
        &mut wide,
        SpeculativePrompt::Tokens(&LLAMA_PROMPT),
        &greedy_config(4),
        2,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap_err();
    assert!(err.to_string().contains("vocab mismatch"), "{err}");
}

/// sc-24134 × sc-24138: the families migrated onto the step seam declare that their steps cannot
/// be replayed as CUDA graphs — every step's positions are Rust-side scalars (the RoPE offset or
/// the learned-position rows taken at the cache's host-side length, the KV written there) — and
/// the step cache names why it is not a graph backing on each backing, so the graph runner
/// refuses by name before any capture instead of trusting the trait's permissive default.
#[test]
fn migrated_families_declare_their_steps_uncapturable() {
    let causal = tiny_causal(false, 7);
    assert_eq!(
        StepModel::graph_support(&causal),
        Err("positions_host_scalar")
    );
    assert_eq!(
        StepModel::graph_support(&gemma4()),
        Err("positions_host_scalar")
    );
    assert_eq!(
        StepModel::graph_support(tiny_llava().language()),
        Err("positions_host_scalar")
    );
    assert_eq!(
        StepModel::graph_support(&tiny_starcoder2()),
        Err("positions_host_scalar")
    );
    assert_eq!(
        StepModel::graph_support(&tiny_starvector_1b()),
        Err("positions_host_scalar")
    );
    assert_eq!(
        DecodeCache::graph_support(&causal.new_step_cache()),
        Err("growing_kv")
    );
    assert_eq!(
        DecodeCache::graph_support(&causal.new_static_cache(8).unwrap()),
        Err("positions_host_scalar")
    );
}

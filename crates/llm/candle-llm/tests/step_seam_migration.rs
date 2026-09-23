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

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use serde_json::{json, Value};

use candle_llm::config::ModelConfig;
use candle_llm::decode::{
    generate, generate_from_prefill_with_stop, CancelFlag, GenerationConfig, StreamEvent,
};
use candle_llm::llava::LlavaModel;
use candle_llm::models::{
    CausalLm, StarCoder2, StarCoder2Config, StarVectorDecoder, StarVectorDecoderGeometry,
};
use candle_llm::primitives::sampler::{sample, SamplingParams};
use candle_llm::primitives::{input_ids, KvCache, SplitMix64, TokenRng, Weights};

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

/// The committed golden for `name`, after checking that `before` (this tree's pre-migration path)
/// still reproduces it bit-for-bit. With `SC24138_WRITE_GOLDENS=1` the golden is (re)written from
/// `before` instead — done once, at the commit that introduced this file.
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
    assert_eq!(
        before, want,
        "{name}: the pre-migration path drifted from its golden"
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
/// greedy walk through `decode_logits` for the rows each token came from.
fn causal_before(model: &CausalLm, prompt: &[i32]) -> Golden {
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
    let model = tiny_causal(false, 0x11A_4A);
    golden("llama", causal_before(&model, &LLAMA_PROMPT));
}

#[test]
fn qwen3_dense_reference_matches_its_pre_migration_golden() {
    let model = tiny_causal(true, 0x0_3E_3);
    golden("qwen3_dense", causal_before(&model, &LLAMA_PROMPT));
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
    let model = gemma4();
    golden("gemma4", causal_before(&model, &gemma4_prompt()));
}

/// The `gemma4_mm` splice, standing in for the vision/audio embedders: the prompt's embeddings
/// with rows `2..5` overwritten by deterministic "soft token" features, prefilled on ordinary 1-D
/// positions exactly as the provider's Gemma 4 multimodal branch does.
fn gemma4_mm_embeds(model: &CausalLm, prompt: &[i32]) -> Tensor {
    let embeds = model
        .embed(&input_ids(prompt, &Device::Cpu).unwrap())
        .unwrap();
    let hidden = embeds.dim(2).unwrap();
    let features = Draw::new(0x6E_33A).rand(&[1, 3, hidden]);
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

fn gemma4_mm_before(model: &CausalLm, prompt: &[i32]) -> Golden {
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
    let model = gemma4();
    golden("gemma4_mm", gemma4_mm_before(&model, &gemma4_prompt()));
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
    let mut d = Draw::wide(0x11A_7A, SCALE_LLAVA);
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

fn llava_before(model: &LlavaModel) -> Golden {
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
    let model = tiny_llava();
    golden("llava", llava_before(&model));
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
    let mut d = Draw::wide(0x5C_2, SCALE_SC2);
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
    let vision = Draw::new(0x5C_2_1).rand(&[1, VISION_ROWS, HIDDEN]);
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

fn starcoder2_before(model: &StarCoder2) -> Golden {
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
    let model = tiny_starcoder2();
    golden("starcoder2", starcoder2_before(&model));
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
    let mut d = Draw::new(0x5_1B);
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
    let vision = Draw::new(0x5_1B_1).rand(&[1, VISION_ROWS, HIDDEN]);
    let text = decoder
        .embeddings(&input_ids(&SVG_PROMPT, &Device::Cpu).unwrap())
        .unwrap();
    Tensor::cat(&[&vision, &text], 1).unwrap()
}

/// The StarVector-1B provider's caption loop before the migration: prefill the joined embeddings,
/// then sample / feed one token at a time through the decoder's layer-owned cache. Returns the
/// tokens and the rows they were drawn from.
fn starvector_1b_loop(
    decoder: &mut StarVectorDecoder,
    params: &SamplingParams,
    seed: u64,
) -> (Vec<i32>, Vec<Vec<f32>>) {
    decoder.reset();
    let prefix = VISION_ROWS + SVG_PROMPT.len();
    let embeds = starvector_1b_embeds(decoder);
    let mut logits = decoder.forward_embeds(&embeds, 0).unwrap();
    let mut history = SVG_PROMPT.to_vec();
    let mut rng = SplitMix64::new(seed);
    let (mut tokens, mut rows) = (Vec::new(), Vec::new());
    for index in 0..NEW_TOKENS {
        rows.push(host(&logits));
        let id = sample(&logits, &history, params, &mut rng, None).unwrap();
        history.push(id);
        tokens.push(id);
        let embed = decoder
            .embeddings(&input_ids(&[id], &Device::Cpu).unwrap())
            .unwrap();
        logits = decoder.forward_embeds(&embed, prefix + index).unwrap();
    }
    decoder.reset();
    (tokens, rows)
}

#[test]
fn starvector_1b_decoder_matches_its_pre_migration_golden() {
    let mut decoder =
        StarVectorDecoder::from_weights_with_geometry(&tiny_starvector_1b_weights(), SV1_GEOMETRY)
            .unwrap();
    let (greedy, rows) = starvector_1b_loop(&mut decoder, &SamplingParams::default(), 0);
    let (sampled, _) = starvector_1b_loop(&mut decoder, &sampled_params(), SAMPLED_SEED);
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

//! Gemma 4 through the ordinary `core_llm::TextLlm` surface, on a synthetic snapshot (sc-18772).
//!
//! `gemma4_decoder.rs` pins the decoder's numerics against a reference oracle. This suite asks the
//! other question: does a Gemma 4 checkpoint reach a caller through the **catalog and provider**
//! path — `can_load` → `load_for_model` → `descriptor` → `generate` — with image and audio
//! conditioning actually wired, at bf16 and at a quantized tier?
//!
//! Everything here runs on CPU against a tiny synthetic checkpoint, so it is a wiring gate, not a
//! quality one: the real-weight coherence legs live in `breadth.rs` behind env vars.
//!
//! The snapshot deliberately uses the **HF nested** layout (`model.language_model.*` beside
//! `model.vision_embedder` / `model.embed_vision` / `model.embed_audio`) — the layout
//! `google/gemma-4-12B-it` actually ships, and the one the loader could not read before this story.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_llm::models::gemma4_mm;
use candle_llm::provider::{can_load, can_load_audio, can_load_vision, LlamaProvider, PROVIDER_ID};
use candle_llm::{load_for_model_with, models::CausalLm};
use core_llm::{
    AudioRef, Content, ImageRef, LoadSpec, Message, ModelRequirements, Quantize, Role, Sampling,
    StreamEvent, TextLlm, TextLlmRequest,
};

mod common;

// --- the synthetic checkpoint's geometry -------------------------------------------------------
//
// Small everywhere, and the multimodal token ids are small too: `Gemma4MmConfig` reads every id from
// `config.json` rather than hard-coding upstream's 258880, so a 32-entry vocab exercises exactly the
// same splice code as the shipped 262144-entry one.
const VOCAB: usize = 32;
/// The WordLevel tokenizer's plain vocabulary (`t0..t{WORDS-1}`). Strictly smaller than [`VOCAB`] so
/// the marker ids below land on rows the WordLevel vocab does not also claim — an added token whose
/// id collides with an existing vocab entry gets re-numbered on load, and the provider's placeholder
/// expansion then finds zero markers to expand.
const WORDS: usize = 16;
/// Hidden/intermediate widths are multiples of 256 on purpose: candle quantizes through GGML block
/// quant (Q4K's block is 256, Q8_0's is 32) and `Projection::load` has no small-tensor fallback, so
/// a narrower synthetic model could not exercise the quantized tiers at all.
const HIDDEN: usize = 256;
const INTER: usize = 512;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const KV_HEADS: usize = 1;

const BOI: usize = 16;
const EOI: usize = 17;
const IMAGE_TOK: usize = 18;
const BOA: usize = 19;
const EOA: usize = 20;
const AUDIO_TOK: usize = 21;

/// Patch side in pixels; a patch flattens to `PATCH * PATCH * 3`.
const PATCH: usize = 2;
const PATCH_ELEMS: usize = PATCH * PATCH * 3;
const POSEMB: usize = 8;
const MAX_IMAGE_TOKENS: usize = 4;
const SAMPLES_PER_TOKEN: usize = 4;
const MAX_AUDIO_TOKENS: usize = 5;
const AUDIO_RATE: u32 = 16_000;

/// Deterministic pseudo-random weights — a fixed stream so a failure is reproducible.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

fn randn(shape: (usize, usize), rng: &mut Rng) -> Tensor {
    scaled(shape, rng, 0.4)
}

/// `randn` with an explicit scale.
///
/// The attention value/output projections use a much larger one on purpose. Gemma scales token
/// embeddings by `sqrt(hidden)` (16 here), so on an UNTRAINED model the residual stream dwarfs
/// whatever attention contributes, every position's final hidden state points essentially at its own
/// token embedding, and greedy decoding emits the same argmax no matter what is in the prompt. That
/// makes a conditioning test vacuously pass-or-fail on noise rather than on whether the features
/// reached the decoder. Amplifying the attention read-out puts the two on comparable footing, which
/// is what lets the assertions below actually observe the spliced rows.
fn scaled(shape: (usize, usize), rng: &mut Rng, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..shape.0 * shape.1).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// How hard the attention read-out is amplified — see [`scaled`].
const ATTN_SCALE: f32 = 6.0;

fn ones(d: usize) -> Tensor {
    Tensor::ones((d,), DType::F32, &Device::Cpu).unwrap()
}

/// A WordLevel `tokenizer.json` whose vocab is `t0..t{VOCAB-1}`, plus **added tokens** for Gemma 4's
/// `<|image|>` / `<|audio|>` markers.
///
/// The markers must be added tokens, not vocab entries: added tokens are matched before
/// pre-tokenization, so `<|image|>` tokenizes to exactly one id. Left in the plain vocab the
/// Whitespace pre-tokenizer would split it into `<|`, `image`, `|>` and the provider's placeholder
/// expansion would find nothing to expand.
fn tokenizer_json() -> String {
    let entries: Vec<String> = (0..WORDS).map(|i| format!("\"t{i}\": {i}")).collect();
    // `special: false` so the markers survive detokenization; a `special` added token is dropped by
    // `decode(.., skip_special_tokens)` in some paths, which would make outputs collide.
    let added = |id: usize, content: &str| {
        format!(
            r#"{{"id": {id}, "content": "{content}", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": false}}"#
        )
    };
    let markers = [
        added(BOI, "<|image>"),
        added(EOI, "<image|>"),
        added(IMAGE_TOK, "<|image|>"),
        added(BOA, "<|audio>"),
        added(EOA, "<audio|>"),
        added(AUDIO_TOK, "<|audio|>"),
    ];
    format!(
        r#"{{ "version": "1.0",
            "added_tokens": [{}],
            "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null, "decoder": null,
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
        markers.join(", "),
        entries.join(", ")
    )
}

fn config_json(with_vision: bool, with_audio: bool) -> String {
    let vision = if with_vision {
        format!(
            r#""vision_config": {{ "model_type": "gemma4_unified_vision",
                "model_patch_size": {PATCH}, "patch_size": {PATCH}, "pooling_kernel_size": 1,
                "mm_posemb_size": {POSEMB}, "num_soft_tokens": {MAX_IMAGE_TOKENS},
                "mm_embed_dim": {HIDDEN}, "output_proj_dims": {HIDDEN}, "rms_norm_eps": 1e-6 }},"#
        )
    } else {
        String::new()
    };
    let audio = if with_audio {
        format!(
            r#""audio_config": {{ "model_type": "gemma4_unified_audio",
                "audio_embed_dim": {SAMPLES_PER_TOKEN},
                "audio_samples_per_token": {SAMPLES_PER_TOKEN}, "rms_norm_eps": 1e-6 }},"#
        )
    } else {
        String::new()
    };
    format!(
        r#"{{ "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified", "tie_word_embeddings": false,
            "boi_token_id": {BOI}, "eoi_token_id": {EOI}, "image_token_id": {IMAGE_TOK},
            "boa_token_id": {BOA}, "eoa_token_index": {EOA}, "audio_token_id": {AUDIO_TOK},
            {vision}{audio}
            "text_config": {{
                "model_type": "gemma4_unified_text",
                "hidden_size": {HIDDEN}, "intermediate_size": {INTER},
                "num_hidden_layers": {LAYERS}, "num_attention_heads": {HEADS},
                "num_key_value_heads": {KV_HEADS}, "head_dim": {HEAD_DIM},
                "global_head_dim": {HEAD_DIM}, "vocab_size": {VOCAB},
                "rms_norm_eps": 1e-6, "sliding_window": 4,
                "attention_k_eq_v": false, "tie_word_embeddings": false,
                "eos_token_id": 31
            }} }}"#
    )
}

fn processor_json() -> String {
    format!(
        r#"{{ "audio_seq_length": {MAX_AUDIO_TOKENS}, "image_seq_length": {MAX_IMAGE_TOKENS},
            "feature_extractor": {{ "audio_samples_per_token": {SAMPLES_PER_TOKEN},
                                    "sampling_rate": {AUDIO_RATE} }} }}"#
    )
}

/// Write a synthetic Gemma 4 unified snapshot in the **HF nested** layout.
///
/// Returns the fixture guard; the caller loads from it, so the guard must stay bound.
fn write_snapshot(with_vision: bool, with_audio: bool) -> common::Fixture {
    let fx = common::Fixture::new("candle-llm-gemma4-mm-", None);
    let dir: &std::path::Path = fx.as_ref();
    std::fs::write(dir.join("config.json"), config_json(with_vision, with_audio)).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    if with_vision || with_audio {
        std::fs::write(dir.join("processor_config.json"), processor_json()).unwrap();
    }

    let mut rng = Rng(0x6734_4D4D);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    // --- decoder, nested under `model.language_model.*` ---
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        randn((VOCAB, HIDDEN), &mut rng),
    );
    t.insert("model.language_model.norm.weight".into(), ones(HIDDEN));
    // `lm_head` is a scaled identity slice, NOT a random matrix: row i reads hidden dim i, so the
    // argmax is the largest of the first VOCAB hidden coordinates. With a random head, an untrained
    // Gemma degenerates into "re-emit the last prompt token" — the residual stream (embeddings
    // scaled by sqrt(hidden)) dominates the final state — and then NO prompt content, spliced or
    // not, can move the output. Reading the hidden state directly keeps the toy model responsive to
    // the whole sequence, which is what lets a provider-level conditioning assertion mean anything.
    let mut head = vec![0f32; VOCAB * HIDDEN];
    for (i, row) in head.chunks_mut(HIDDEN).enumerate() {
        row[i] = 4.0;
    }
    t.insert(
        "lm_head.weight".into(),
        Tensor::from_vec(head, (VOCAB, HIDDEN), &Device::Cpu).unwrap(),
    );
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.language_model.layers.{i}.{s}");
        // Gemma 4's four-norm sandwich, both attention norms, and the per-layer scalar.
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(p(&format!("{norm}.weight")), ones(HIDDEN));
        }
        t.insert(p("self_attn.q_norm.weight"), ones(HEAD_DIM));
        t.insert(p("self_attn.k_norm.weight"), ones(HEAD_DIM));
        t.insert(
            p("layer_scalar"),
            Tensor::from_vec(vec![0.7f32], (1,), &Device::Cpu).unwrap(),
        );
        t.insert(
            p("self_attn.q_proj.weight"),
            randn((HEADS * HEAD_DIM, HIDDEN), &mut rng),
        );
        t.insert(
            p("self_attn.k_proj.weight"),
            randn((KV_HEADS * HEAD_DIM, HIDDEN), &mut rng),
        );
        t.insert(
            p("self_attn.v_proj.weight"),
            scaled((KV_HEADS * HEAD_DIM, HIDDEN), &mut rng, ATTN_SCALE),
        );
        t.insert(
            p("self_attn.o_proj.weight"),
            scaled((HIDDEN, HEADS * HEAD_DIM), &mut rng, ATTN_SCALE),
        );
        t.insert(p("mlp.gate_proj.weight"), randn((INTER, HIDDEN), &mut rng));
        t.insert(p("mlp.up_proj.weight"), randn((INTER, HIDDEN), &mut rng));
        t.insert(p("mlp.down_proj.weight"), randn((HIDDEN, INTER), &mut rng));
    }

    // --- vision embedder + projection, in the HF `model.` layout ---
    if with_vision {
        t.insert("model.vision_embedder.patch_ln1.weight".into(), ones(PATCH_ELEMS));
        t.insert(
            "model.vision_embedder.patch_ln1.bias".into(),
            Tensor::zeros((PATCH_ELEMS,), DType::F32, &Device::Cpu).unwrap(),
        );
        t.insert(
            "model.vision_embedder.patch_dense.weight".into(),
            randn((HIDDEN, PATCH_ELEMS), &mut rng),
        );
        t.insert(
            "model.vision_embedder.patch_dense.bias".into(),
            Tensor::zeros((HIDDEN,), DType::F32, &Device::Cpu).unwrap(),
        );
        t.insert("model.vision_embedder.patch_ln2.weight".into(), ones(HIDDEN));
        t.insert(
            "model.vision_embedder.patch_ln2.bias".into(),
            Tensor::zeros((HIDDEN,), DType::F32, &Device::Cpu).unwrap(),
        );
        // Give each position its own SHAPE, not just its own level: `pos_norm` is a LayerNorm and
        // removes any constant offset, so a plain ramp would make positions indistinguishable.
        let pos: Vec<f32> = (0..POSEMB * 2 * HIDDEN)
            .map(|i| (((i * 37) % 23) as f32) * 0.05)
            .collect();
        t.insert(
            "model.vision_embedder.pos_embedding".into(),
            Tensor::from_vec(pos, (POSEMB, 2, HIDDEN), &Device::Cpu).unwrap(),
        );
        t.insert("model.vision_embedder.pos_norm.weight".into(), ones(HIDDEN));
        t.insert(
            "model.vision_embedder.pos_norm.bias".into(),
            Tensor::zeros((HIDDEN,), DType::F32, &Device::Cpu).unwrap(),
        );
        t.insert(
            "model.embed_vision.embedding_projection.weight".into(),
            randn((HIDDEN, HIDDEN), &mut rng),
        );
    }
    if with_audio {
        t.insert(
            "model.embed_audio.embedding_projection.weight".into(),
            randn((HIDDEN, SAMPLES_PER_TOKEN), &mut rng),
        );
    }

    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
    fx
}

fn spec_of(fx: &common::Fixture) -> LoadSpec {
    let dir: &std::path::Path = fx.as_ref();
    LoadSpec::dense(dir.to_str().unwrap().to_string())
}

fn image(w: u32, h: u32, left: [u8; 3], right: [u8; 3]) -> ImageRef {
    let mut px = Vec::with_capacity(w as usize * h as usize * 3);
    for _y in 0..h {
        for x in 0..w {
            let c = if x < w / 2 { left } else { right };
            px.extend_from_slice(&c);
        }
    }
    ImageRef::new(w, h, px).unwrap()
}

fn request(content: Vec<Content>) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 4,
        seed: Some(0),
        ..Default::default()
    }
}

/// Generate and return the emitted **token ids**, asserting the streamed deltas reconstruct the
/// final text.
///
/// Ids rather than decoded text: this fixture's vocabulary is `t0..t15` plus markers, so several
/// distinct ids can decode to visually similar (or empty) pieces. Comparing ids asks the question
/// these tests actually mean — did the model take a different path — without depending on how the
/// toy tokenizer renders it.
fn generate(p: &dyn TextLlm, req: &TextLlmRequest) -> Vec<u32> {
    let mut streamed = String::new();
    let mut ids = Vec::new();
    let out = p
        .generate(req, &mut |ev| {
            if let StreamEvent::Token { id, text, .. } = ev {
                streamed.push_str(&text);
                ids.push(id);
            }
        })
        .expect("generate");
    assert_eq!(
        streamed, out.text,
        "streamed deltas must reconstruct the final text"
    );
    assert!(!ids.is_empty(), "generation produced no tokens");
    ids
}


/// The provider's own count of the expanded prompt — the public number that reflects placeholder
/// expansion, taken straight from `usage`.
fn prompt_tokens(p: &dyn TextLlm, req: &TextLlmRequest) -> u32 {
    p.generate(req, &mut |_| {})
        .expect("generate")
        .usage
        .prompt_tokens
}

/// Splice `count` feature rows onto the `marker` rows of an embedded sequence and assert that
/// exactly those rows changed.
///
/// This is placed on the embeddings, not on the logits or the emitted tokens, because that is the
/// value the fault appears in: a splice that silently did nothing leaves the sequence length, every
/// shape, and the whole downstream computation valid — only these rows would still hold the marker's
/// own token embedding.
fn assert_splice_replaces_marked_rows(fx: &common::Fixture, marker: i32, count: usize) {
    let dir: &std::path::Path = fx.as_ref();
    let cfg = candle_llm::config::ModelConfig::from_dir(dir).unwrap();
    let hidden = cfg.hidden_size as usize;
    let weights = candle_llm::primitives::Weights::from_dir(dir, &Device::Cpu).unwrap();
    let m = CausalLm::from_weights(&weights, "", cfg).expect("decoder loads");

    // A short sequence with one marker run, framed the way the provider frames it.
    let mut ids: Vec<i32> = vec![1, 2];
    ids.push(BOI as i32);
    ids.extend(std::iter::repeat_n(marker, count));
    ids.push(EOI as i32);
    ids.push(3);

    let t = candle_llm::primitives::nn::input_ids(&ids, &Device::Cpu).unwrap();
    let embeds = m.embed_input_ids(&t).expect("embed");
    // Distinctive feature rows, unlike any token embedding.
    let feats = Tensor::from_vec(
        (0..count * hidden).map(|i| 7.0 + i as f32).collect::<Vec<f32>>(),
        (count, hidden),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(embeds.dtype())
    .unwrap();
    let spliced = m
        .splice_vision_features(&embeds, &ids, &feats, &[marker])
        .expect("splice");

    let before: Vec<Vec<f32>> = embeds
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2()
        .unwrap();
    let after: Vec<Vec<f32>> = spliced
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2()
        .unwrap();
    assert_eq!(before.len(), ids.len());
    let mut replaced = 0usize;
    for (i, id) in ids.iter().enumerate() {
        let changed = before[i]
            .iter()
            .zip(&after[i])
            .any(|(a, b)| (a - b).abs() > 1e-3);
        if *id == marker {
            assert!(changed, "marker row {i} must be replaced by a feature row");
            replaced += 1;
        } else {
            assert!(!changed, "non-marker row {i} (id {id}) must be untouched");
        }
    }
    assert_eq!(replaced, count, "every reserved row must take a feature row");
}

// ---------------------------------------------------------------------------------------------
// catalog + descriptor
// ---------------------------------------------------------------------------------------------

/// A Gemma 4 snapshot must be claimed by the ordinary text provider and resolve **by model**.
///
/// This is the regression this story exists for on the candle side: `can_load` used to decline any
/// snapshot carrying a `vision_config` unless it was a Qwen wrapper, and `google/gemma-4-12B-it`
/// carries one — so the general-purpose Gemma 4 LLM was refused by the catalog outright.
#[test]
fn gemma4_snapshot_is_claimed_and_resolves_by_model() {
    let fx = write_snapshot(true, true);
    let spec = spec_of(&fx);
    assert!(
        can_load(&spec),
        "a Gemma 4 snapshot must be claimed despite carrying a vision_config"
    );
    let p = load_for_model_with(&spec, &ModelRequirements::default()).expect("resolve by model");
    assert_eq!(p.descriptor().id, PROVIDER_ID);
    assert_eq!(
        p.descriptor().family,
        "gemma4_unified",
        "the dispatched family tag"
    );
}

/// Whatever the descriptor advertises must be backed by tensors that are actually present, and the
/// weightless probes must agree with the loaded descriptor.
///
/// The three snapshots below are the whole truth table this story owes: both front-ends, vision
/// only, and neither. A descriptor that advertised a modality the checkpoint does not ship is the
/// advertised-but-absent failure, and it is what the `(false, ...)` rows catch.
#[test]
fn descriptor_advertises_exactly_the_front_ends_the_checkpoint_ships() {
    for (with_vision, with_audio) in [(true, true), (true, false), (false, false)] {
        let fx = write_snapshot(with_vision, with_audio);
        let spec = spec_of(&fx);
        let p = LlamaProvider::load(&spec).expect("load");
        let caps = &p.descriptor().capabilities;
        assert_eq!(
            caps.supports_vision, with_vision,
            "supports_vision must track the vision tensors (vision={with_vision})"
        );
        assert_eq!(
            caps.supports_audio, with_audio,
            "supports_audio must track the audio tensors (audio={with_audio})"
        );
        // Gemma 4 declares a `video_token_id`, but this provider ships no frame-sampling path, so
        // video stays declared-unsupported rather than advertised-and-broken.
        assert!(
            !caps.supports_video,
            "video must stay unsupported: there is no frame-sampling path"
        );
        // The weightless probes drive model-first routing and must agree with the loaded truth.
        assert_eq!(can_load_vision(&spec), with_vision, "weightless vision probe");
        assert_eq!(can_load_audio(&spec), with_audio, "weightless audio probe");
    }
}

/// A modality the checkpoint does not ship is REJECTED, not silently answered from the text.
///
/// This is the other half of "never advertised-but-absent": the capability gate has to bite.
#[test]
fn unsupported_modalities_are_rejected_not_dropped() {
    let fx = write_snapshot(false, false);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load text-only gemma4");

    let img = request(vec![
        Content::Image(image(4, 4, [255, 0, 0], [0, 0, 255])),
        Content::text("t1"),
    ]);
    let err = p.validate(&img).expect_err("image must be rejected");
    assert!(format!("{err}").contains("image"), "{err}");

    let aud = request(vec![
        Content::Audio(AudioRef::new(AUDIO_RATE, vec![0.5; 8]).unwrap()),
        Content::text("t1"),
    ]);
    let err = p.validate(&aud).expect_err("audio must be rejected");
    assert!(format!("{err}").contains("audio"), "{err}");

    // Generating (not just validating) must refuse too — `generate` calls `validate` first, so a
    // gate that only ran in `validate` would still be bypassable by a direct `generate`.
    assert!(p.generate(&img, &mut |_| {}).is_err());
    assert!(p.generate(&aud, &mut |_| {}).is_err());
}

// ---------------------------------------------------------------------------------------------
// conditioning actually reaches the decoder
// ---------------------------------------------------------------------------------------------

/// Text generation through the ordinary surface, at bf16-equivalent (dense) and at **both**
/// quantized tiers — AC bullet 1's wiring half on the deterministic CPU path.
#[test]
fn gemma4_generates_dense_and_at_each_quantized_tier() {
    let fx = write_snapshot(true, true);
    let dir: &std::path::Path = fx.as_ref();
    let src = dir.to_str().unwrap().to_string();

    for quantize in [None, Some(Quantize::Q4), Some(Quantize::Q8)] {
        let spec = LoadSpec {
            source: src.clone(),
            quantize,
        };
        let p = LlamaProvider::load(&spec)
            .unwrap_or_else(|e| panic!("load gemma4 (quantize={quantize:?}): {e}"));
        let ids = generate(&p, &request(vec![Content::text("t1 t2 t3")]));
        assert!(!ids.is_empty(), "quantize={quantize:?}: produced no tokens");
        assert!(
            p.is_quantized() == quantize.is_some(),
            "quantize={quantize:?}: is_quantized must report the tier that was applied"
        );
    }
}

/// The image span is reserved at exactly the right size AND the reserved rows are actually
/// overwritten by the tower's features.
///
/// Both halves are needed and neither implies the other. `usage.prompt_tokens` is the provider's own
/// count of the expanded sequence, so it catches a span that never expanded or expanded to the wrong
/// length; the splice check catches features that were computed and then dropped, which leaves the
/// length correct and every shape intact.
///
/// The assertion is deliberately NOT "the generated tokens differ". This fixture's weights are
/// random, and on an untrained Gemma the residual stream (token embeddings scaled by `sqrt(hidden)`)
/// dominates the final hidden state so completely that greedy decoding re-emits the last prompt
/// token whatever precedes it. A token-level check would therefore be green or red for reasons that
/// have nothing to do with the splice. The behavioural claim — that the model *answers* differently
/// about different images — is made on real weights in `breadth.rs`.
#[test]
fn image_span_is_reserved_and_spliced() {
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");
    assert!(p.descriptor().capabilities.supports_vision);

    let text_only = prompt_tokens(&p, &request(vec![Content::text("t1")]));
    let with_image = prompt_tokens(
        &p,
        &request(vec![
            Content::Image(image(4, 4, [255, 0, 0], [0, 0, 255])),
            Content::text("t1"),
        ]),
    );
    // A 4x4 image at a 2px patch and a 4-token budget is a 2x2 grid: boi + 4 soft tokens + eoi.
    let (gh, gw) = gemma4_mm::soft_token_grid(4, 4, MAX_IMAGE_TOKENS);
    assert_eq!((gh, gw), (2, 2), "the fixture's image must fill the budget");
    // The image request carries one extra content block, which renders to one marker token; the
    // expansion then replaces that token with `boi + soft tokens + eoi`.
    assert_eq!(
        with_image,
        text_only + 2 + (gh * gw) as u32,
        "the marker must expand to boi + {} soft tokens + eoi",
        gh * gw
    );

    assert_splice_replaces_marked_rows(&fx, IMAGE_TOK as i32, gh * gw);
}

/// The audio span is reserved at exactly the right size and its rows are spliced. See
/// [`image_span_is_reserved_and_spliced`] for why this is not a token-level assertion.
#[test]
fn audio_span_is_reserved_and_spliced() {
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");
    assert!(p.descriptor().capabilities.supports_audio);

    let text_only = prompt_tokens(&p, &request(vec![Content::text("t1")]));
    let samples = 8usize;
    let with_audio = prompt_tokens(
        &p,
        &request(vec![
            Content::Audio(AudioRef::new(AUDIO_RATE, vec![0.9; samples]).unwrap()),
            Content::text("t1"),
        ]),
    );
    let frames = gemma4_mm::audio_frame_count(samples, SAMPLES_PER_TOKEN, MAX_AUDIO_TOKENS);
    assert_eq!(frames, 2, "8 samples at 4 per token is 2 frames");
    assert_eq!(
        with_audio,
        text_only + 2 + frames as u32,
        "the marker must expand to boa + {frames} soft tokens + eoa"
    );

    assert_splice_replaces_marked_rows(&fx, AUDIO_TOK as i32, frames);
}

/// Image and audio in ONE request: both spans expand, and neither pass consumes the other's marker.
///
/// The two expansions run as separate passes over the same id vector, so a pass that matched the
/// wrong id would swallow the other modality's span — visible here as a prompt length that accounts
/// for only one of them.
#[test]
fn image_and_audio_together_both_reserve_their_spans() {
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");

    let img = || Content::Image(image(4, 4, [255, 0, 0], [0, 0, 255]));
    let aud = || Content::Audio(AudioRef::new(AUDIO_RATE, vec![0.9; 8]).unwrap());

    let base = prompt_tokens(&p, &request(vec![Content::text("t1")]));
    let only_image = prompt_tokens(&p, &request(vec![img(), Content::text("t1")]));
    let only_audio = prompt_tokens(&p, &request(vec![Content::text("t1"), aud()]));
    let both = prompt_tokens(&p, &request(vec![img(), Content::text("t1"), aud()]));

    let image_cost = only_image - base;
    let audio_cost = only_audio - base;
    assert!(image_cost > 0 && audio_cost > 0);
    assert_eq!(
        both,
        base + image_cost + audio_cost,
        "both spans must be reserved; a short count means one pass ate the other's marker"
    );
    // And the generate actually runs to completion with both modalities spliced.
    assert!(!generate(&p, &request(vec![img(), Content::text("t1"), aud()])).is_empty());
}

/// **Through the provider**, an image changes the generated tokens — and two images that differ only
/// in their spatial arrangement change them differently.
///
/// This is the assertion the story's "image conditioning demonstrated" claim rests on, and it is the
/// only one that observes the provider's own splice. Everything else here checks a piece: the tower
/// emits distinct rows, the span is the right length, `splice_vision_features` replaces the marked
/// rows. All of those stay green if `prepare_gemma4` computes the features and then forgets to
/// splice them, which is precisely the silent failure this catches.
///
/// The mirrored pair matters as much as the image/text pair: same pixels, same histogram, only the
/// left-right order differs. A pipeline that dropped the positional embedding (or collapsed its two
/// axes) would give both the same answer.
#[test]
fn image_conditioning_changes_the_provider_output() {
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");
    assert!(p.descriptor().capabilities.supports_vision);

    let with = |a: [u8; 3], b: [u8; 3]| {
        generate(
            &p,
            &request(vec![Content::Image(image(4, 4, a, b)), Content::text("t1")]),
        )
    };
    let text_only = generate(&p, &request(vec![Content::text("t1")]));
    let red_blue = with([255, 0, 0], [0, 0, 255]);
    let blue_red = with([0, 0, 255], [255, 0, 0]);

    assert_ne!(
        red_blue, text_only,
        "an image must change the output; identical to text-only means the splice never landed"
    );
    assert_ne!(
        red_blue, blue_red,
        "a mirrored image must change the output — the positional table distinguishes them"
    );
    // The same image twice is deterministic, so the differences above are conditioning, not noise.
    assert_eq!(red_blue, with([255, 0, 0], [0, 0, 255]), "generation is deterministic");
}

/// **Through the provider**, audio changes the generated tokens, and different waveforms change them
/// differently. The audio half of [`image_conditioning_changes_the_provider_output`].
#[test]
fn audio_conditioning_changes_the_provider_output() {
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");
    assert!(p.descriptor().capabilities.supports_audio);

    let clip = |v: Vec<f32>| {
        generate(
            &p,
            &request(vec![
                Content::Audio(AudioRef::new(AUDIO_RATE, v).unwrap()),
                Content::text("t1"),
            ]),
        )
    };
    let text_only = generate(&p, &request(vec![Content::text("t1")]));
    let rising: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
    let falling: Vec<f32> = (0..8).map(|i| 1.0 - i as f32 / 8.0).collect();

    assert_ne!(
        clip(rising.clone()),
        text_only,
        "audio must change the output; identical to text-only means the splice never landed"
    );
    assert_ne!(
        clip(rising.clone()),
        clip(falling),
        "different waveforms must produce different outputs"
    );
    assert_eq!(clip(rising.clone()), clip(rising), "generation is deterministic");
}

/// The chat template is taken from the sidecar `chat_template.jinja` when the snapshot ships one,
/// in preference to `tokenizer_config.json`'s embedded key and to the Llama-3 default.
///
/// `google/gemma-4-12B-it` ships exactly that way — a `chat_template.jinja` file and a
/// `tokenizer_config.json` with no `chat_template` key — so reading only the embedded key addresses
/// Gemma 4 in Llama-3's chat format. Nothing about that fails loudly: the prompt renders, the model
/// generates, every length assertion holds. It just answers badly. This pins the precedence so the
/// fallback cannot silently come back.
#[test]
fn the_sidecar_chat_template_wins_over_the_embedded_one() {
    let fx = write_snapshot(true, true);
    let dir: &std::path::Path = fx.as_ref();

    // Sidecar only: its marker must appear in the rendered prompt.
    std::fs::write(
        dir.join("chat_template.jinja"),
        "SIDECAR{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}",
    )
    .unwrap();
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load with sidecar");
    let ids = generate(&p, &request(vec![Content::text("t1")]));
    assert!(!ids.is_empty());

    // With BOTH present the sidecar still wins. The embedded template renders a different marker, so
    // the two are distinguishable by the prompt length they produce.
    let tok_cfg = serde_json::json!({
        "chat_template": "EMBEDDED EMBEDDED EMBEDDED EMBEDDED{% for m in messages %}{{ m['content'] }}{% endfor %}",
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    });
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string(&tok_cfg).unwrap(),
    )
    .unwrap();
    let with_both = LlamaProvider::load(&spec_of(&fx)).expect("load with both");
    let both_len = prompt_tokens(&with_both, &request(vec![Content::text("t1")]));

    // Now remove the sidecar: the embedded template takes over and renders a measurably longer
    // prompt (four marker words vs one), which is what proves the sidecar was being preferred.
    std::fs::remove_file(dir.join("chat_template.jinja")).unwrap();
    let embedded_only = LlamaProvider::load(&spec_of(&fx)).expect("load embedded only");
    let embedded_len = prompt_tokens(&embedded_only, &request(vec![Content::text("t1")]));

    assert_ne!(
        both_len, embedded_len,
        "with a sidecar present the embedded template must NOT be the one that rendered"
    );
}

/// A clip at the wrong sample rate is refused rather than silently reinterpreted.
///
/// The projector frames audio in *samples*, so feeding 8 kHz into a 16 kHz framing would halve the
/// effective duration of every frame — a wrong answer, not an error, unless this bites.
#[test]
fn a_wrong_sample_rate_is_refused() {
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");
    let req = request(vec![
        Content::Audio(AudioRef::new(8_000, vec![0.5; 8]).unwrap()),
        Content::text("t1"),
    ]);
    let err = p
        .generate(&req, &mut |_| {})
        .expect_err("a rate mismatch must be refused");
    let msg = format!("{err}");
    assert!(msg.contains("16000") && msg.contains("8000"), "{msg}");
}

/// Video is declared unsupported and refused with a message that says so — not rendered as an image
/// and not silently dropped to text.
#[test]
fn video_is_refused_with_a_reason() {
    use core_llm::VideoRef;
    let fx = write_snapshot(true, true);
    let p = LlamaProvider::load(&spec_of(&fx)).expect("load");
    let frames = vec![image(4, 4, [255, 0, 0], [0, 0, 255])];
    let req = request(vec![
        Content::Video(VideoRef::new(frames, vec![0.0]).unwrap()),
        Content::text("t1"),
    ]);
    // The capability gate rejects it before the provider ever reaches substitution.
    let err = p.validate(&req).expect_err("video must be rejected");
    assert!(format!("{err}").contains("video"), "{err}");
}

/// The soft-token span the prompt reserves must be exactly the number of feature rows the tower
/// produces, for every grid the budget allows.
///
/// A span one token too long leaves an unspliced placeholder carrying the raw `<|image|>` embedding;
/// one too short drops a patch. Both are silent — the model just answers slightly wrong — so the
/// count identity is pinned directly rather than inferred from a successful generate.
#[test]
fn the_reserved_span_matches_the_feature_row_count() {
    let fx = write_snapshot(true, true);
    let dir: &std::path::Path = fx.as_ref();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let proc_cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("processor_config.json")).unwrap())
            .unwrap();
    let mm = gemma4_mm::Gemma4MmConfig::from_json(&cfg, Some(&proc_cfg)).unwrap();
    let vcfg = mm.vision.expect("vision config");

    let weights = candle_llm::primitives::Weights::from_dir(dir, &Device::Cpu).unwrap();
    let tower = gemma4_mm::Gemma4VisionEmbedder::from_weights(
        &weights,
        gemma4_mm::Gemma4Layout::HfUnified,
        vcfg.clone(),
        DType::F32,
    )
    .unwrap();

    for (w, h) in [(4usize, 4usize), (8, 4), (4, 8), (2, 2), (6, 3)] {
        let (gh, gw) = gemma4_mm::soft_token_grid(w, h, vcfg.max_soft_tokens);
        let (tw, th) = (gw * vcfg.patch_pixels, gh * vcfg.patch_pixels);
        let resized = candle_llm::image::resize_bicubic_u8(
            &image(w as u32, h as u32, [255, 0, 0], [0, 0, 255]).pixels,
            h,
            w,
            th,
            tw,
        )
        .unwrap();
        let flat = gemma4_mm::patchify(&resized, tw, th, vcfg.patch_pixels).unwrap();
        let patches =
            gemma4_mm::patch_tensor(&flat, vcfg.patch_elems(), &Device::Cpu, DType::F32).unwrap();
        let feats = tower.forward(&patches, (gh, gw)).unwrap();
        assert_eq!(
            feats.dims()[0],
            gh * gw,
            "{w}x{h}: the tower must emit one row per reserved soft token"
        );
        assert!(
            gh * gw <= vcfg.max_soft_tokens,
            "{w}x{h}: grid {gh}x{gw} overruns the per-image budget"
        );
    }
}

/// The decoder itself still loads from the nested layout — the fix that lets `google/gemma-4-12B-it`
/// be read at all. Loading the same snapshot's decoder directly (no provider) must succeed.
#[test]
fn the_nested_decoder_layout_loads() {
    let fx = write_snapshot(true, true);
    let dir: &std::path::Path = fx.as_ref();
    let cfg = candle_llm::config::ModelConfig::from_dir(dir).expect("config parses");
    assert!(cfg.is_gemma4());
    let weights = candle_llm::primitives::Weights::from_dir(dir, &Device::Cpu).unwrap();
    assert!(
        weights.contains("model.language_model.embed_tokens.weight"),
        "the fixture must use the nested layout this test is about"
    );
    CausalLm::from_weights(&weights, "", cfg).expect("the nested decoder must load");
}




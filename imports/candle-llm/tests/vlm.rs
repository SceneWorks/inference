//! Vision-language (LLaVA) path tests (story 7262).
//!
//! A **synthetic CPU** model (tiny SigLIP tower + projector + tiny Llama, random weights) proves the
//! image-splice mechanics end-to-end with no weights and no GPU: the vision tower is image-sensitive,
//! its projected features are spliced into the right embedding rows, and the decode is deterministic.
//! The `#[ignore]`d **real-weights** test loads an actual LLaVA snapshot (`CANDLE_LLM_VLM_MODEL`),
//! captions an image through the `core-llm` multimodal contract on the selected device (CUDA with
//! `--features cuda`), and confirms the conformance suite's multimodal check passes via the **generate**
//! branch.

use std::collections::HashMap;

use candle_core::{Device, Tensor};

use candle_llm::decode::CancelFlag;
use candle_llm::llava::{expand_image_tokens, splice_image_features, LlavaModel};
use candle_llm::primitives::sampler::{SamplingParams, SplitMix64};
use candle_llm::primitives::{input_ids, TokenRng};

// ---- tiny synthetic LLaVA geometry -----------------------------------------------------------

const IMG_TOKEN: i32 = 7;
// Vision tower: 8×8 image, 4×4 patches → 2×2 grid = 4 patch tokens; hidden 16, 1 layer, 2 heads.
const V_IMG: usize = 8;
const V_PATCH: usize = 4;
const V_HIDDEN: usize = 16;
const V_INTER: usize = 32;
const V_HEADS: usize = 2;
const N_PATCHES: usize = 4;
// Language decoder: tiny 2-layer Llama, hidden 32, vocab 32.
const L_HIDDEN: usize = 32;
const L_INTER: usize = 64;
const L_HEADS: usize = 4;
const L_KV_HEADS: usize = 2;
const L_HEAD_DIM: usize = L_HIDDEN / L_HEADS; // 8
const VOCAB: usize = 32;

fn randf(dims: &[usize], rng: &mut SplitMix64) -> Tensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, dims.to_vec(), &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::from_vec(vec![1.0f32; d], (d,), &Device::Cpu).unwrap()
}

fn zeros(d: usize) -> Tensor {
    Tensor::from_vec(vec![0.0f32; d], (d,), &Device::Cpu).unwrap()
}

/// Build a tiny `LlavaForConditionalGeneration` snapshot and load it. A per-call atomic sequence
/// keeps the temp dir unique so concurrent tests never share (and delete) one another's snapshot.
fn build_tiny_llava() -> LlavaModel {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("candle-llm-vlm-{}-{uniq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let cfg = format!(
        r#"{{
            "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
            "image_token_index": {IMG_TOKEN}, "vision_feature_layer": -1,
            "vision_feature_select_strategy": "full", "projector_hidden_act": "gelu",
            "vision_config": {{
                "image_size": {V_IMG}, "patch_size": {V_PATCH}, "num_channels": 3,
                "hidden_size": {V_HIDDEN}, "intermediate_size": {V_INTER},
                "num_hidden_layers": 1, "num_attention_heads": {V_HEADS}, "layer_norm_eps": 1e-6
            }},
            "text_config": {{
                "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_size": {L_HIDDEN}, "intermediate_size": {L_INTER}, "num_hidden_layers": 2,
                "num_attention_heads": {L_HEADS}, "num_key_value_heads": {L_KV_HEADS},
                "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "tie_word_embeddings": false, "eos_token_id": 0
            }}
        }}"#
    );
    std::fs::write(dir.join("config.json"), cfg).unwrap();

    let mut rng = SplitMix64::new(0x5EED_F00D);
    let mut w: HashMap<String, Tensor> = HashMap::new();

    // --- vision tower (prefix vision_tower.vision_model) ---
    let vp = |s: &str| format!("vision_tower.vision_model.{s}");
    w.insert(
        vp("embeddings.patch_embedding.weight"),
        randf(&[V_HIDDEN, 3, V_PATCH, V_PATCH], &mut rng),
    );
    w.insert(
        vp("embeddings.patch_embedding.bias"),
        randf(&[V_HIDDEN], &mut rng),
    );
    w.insert(
        vp("embeddings.position_embedding.weight"),
        randf(&[N_PATCHES, V_HIDDEN], &mut rng),
    );
    {
        let lp = |s: &str| vp(&format!("encoder.layers.0.{s}"));
        w.insert(lp("layer_norm1.weight"), ones(V_HIDDEN));
        w.insert(lp("layer_norm1.bias"), zeros(V_HIDDEN));
        w.insert(lp("layer_norm2.weight"), ones(V_HIDDEN));
        w.insert(lp("layer_norm2.bias"), zeros(V_HIDDEN));
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            w.insert(
                lp(&format!("self_attn.{proj}.weight")),
                randf(&[V_HIDDEN, V_HIDDEN], &mut rng),
            );
            w.insert(
                lp(&format!("self_attn.{proj}.bias")),
                randf(&[V_HIDDEN], &mut rng),
            );
        }
        w.insert(lp("mlp.fc1.weight"), randf(&[V_INTER, V_HIDDEN], &mut rng));
        w.insert(lp("mlp.fc1.bias"), randf(&[V_INTER], &mut rng));
        w.insert(lp("mlp.fc2.weight"), randf(&[V_HIDDEN, V_INTER], &mut rng));
        w.insert(lp("mlp.fc2.bias"), randf(&[V_HIDDEN], &mut rng));
    }
    w.insert(vp("post_layernorm.weight"), ones(V_HIDDEN));
    w.insert(vp("post_layernorm.bias"), zeros(V_HIDDEN));

    // --- multi-modal projector (16 -> 32) ---
    w.insert(
        "multi_modal_projector.linear_1.weight".into(),
        randf(&[L_HIDDEN, V_HIDDEN], &mut rng),
    );
    w.insert(
        "multi_modal_projector.linear_1.bias".into(),
        randf(&[L_HIDDEN], &mut rng),
    );
    w.insert(
        "multi_modal_projector.linear_2.weight".into(),
        randf(&[L_HIDDEN, L_HIDDEN], &mut rng),
    );
    w.insert(
        "multi_modal_projector.linear_2.bias".into(),
        randf(&[L_HIDDEN], &mut rng),
    );

    // --- language decoder (prefix language_model) ---
    let lm = |s: &str| format!("language_model.{s}");
    w.insert(
        lm("model.embed_tokens.weight"),
        randf(&[VOCAB, L_HIDDEN], &mut rng),
    );
    w.insert(lm("model.norm.weight"), ones(L_HIDDEN));
    w.insert(lm("lm_head.weight"), randf(&[VOCAB, L_HIDDEN], &mut rng));
    let q_dim = L_HEADS * L_HEAD_DIM;
    let kv_dim = L_KV_HEADS * L_HEAD_DIM;
    for i in 0..2 {
        let p = |s: &str| lm(&format!("model.layers.{i}.{s}"));
        w.insert(p("input_layernorm.weight"), ones(L_HIDDEN));
        w.insert(p("post_attention_layernorm.weight"), ones(L_HIDDEN));
        w.insert(
            p("self_attn.q_proj.weight"),
            randf(&[q_dim, L_HIDDEN], &mut rng),
        );
        w.insert(
            p("self_attn.k_proj.weight"),
            randf(&[kv_dim, L_HIDDEN], &mut rng),
        );
        w.insert(
            p("self_attn.v_proj.weight"),
            randf(&[kv_dim, L_HIDDEN], &mut rng),
        );
        w.insert(
            p("self_attn.o_proj.weight"),
            randf(&[L_HIDDEN, q_dim], &mut rng),
        );
        w.insert(
            p("mlp.gate_proj.weight"),
            randf(&[L_INTER, L_HIDDEN], &mut rng),
        );
        w.insert(
            p("mlp.up_proj.weight"),
            randf(&[L_INTER, L_HIDDEN], &mut rng),
        );
        w.insert(
            p("mlp.down_proj.weight"),
            randf(&[L_HIDDEN, L_INTER], &mut rng),
        );
    }

    candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
    let model = LlavaModel::from_dir(&dir, &Device::Cpu).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    model
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context: 0,
    }
}

/// A solid-gray 8×8 image and a deterministic 8×8 gradient — two visibly different inputs.
fn gray() -> Vec<u8> {
    vec![127u8; V_IMG * V_IMG * 3]
}
fn gradient() -> Vec<u8> {
    let mut px = Vec::with_capacity(V_IMG * V_IMG * 3);
    for y in 0..V_IMG as u32 {
        for x in 0..V_IMG as u32 {
            px.push((x * 30) as u8);
            px.push((y * 30) as u8);
            px.push(((x + y) * 15) as u8);
        }
    }
    px
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The tiny SigLIP tower is image-sensitive, the projector lifts its features to the language hidden
/// size, and splicing them into the right embedding rows changes the first-token logits — i.e. the
/// image actually drives the decode (not silently dropped).
#[test]
fn image_features_flow_into_decode_cpu() {
    let model = build_tiny_llava();
    assert_eq!(model.config().image_seq_length, N_PATCHES);

    let dt = model.language().compute_dtype();
    let feat_gray = model.image_features(&gray(), V_IMG, V_IMG).unwrap();
    let feat_grad = model.image_features(&gradient(), V_IMG, V_IMG).unwrap();
    assert_eq!(feat_gray.dims(), &[1, N_PATCHES, L_HIDDEN]);
    // Two different images produce different projected features.
    assert!(
        max_abs_diff(&feat_gray, &feat_grad) > 1e-3,
        "distinct images must yield distinct vision features"
    );

    // Prompt with a single image token; expansion replaces it with N_PATCHES placeholders.
    let prompt = [1i32, IMG_TOKEN, 2, 3];
    let expanded = expand_image_tokens(&prompt, IMG_TOKEN, model.config().image_seq_length);
    assert_eq!(expanded.len(), prompt.len() - 1 + N_PATCHES);
    let ids = input_ids(&expanded, &Device::Cpu).unwrap();
    let embeds = model.language().embed(&ids).unwrap();

    let first_logits = |feat: &Tensor| {
        let f = feat.to_dtype(dt).unwrap();
        let spliced = splice_image_features(&embeds, &expanded, &f, IMG_TOKEN).unwrap();
        let mut cache = model.language().new_cache();
        model
            .language()
            .decode_logits_from_embeds(&spliced, &mut cache, 0)
            .unwrap()
    };
    let lg = first_logits(&feat_gray);
    let lr = first_logits(&feat_grad);
    assert!(
        max_abs_diff(&lg, &lr) > 1e-4,
        "different images must change the decoder's first-token logits"
    );
}

/// Greedy captioning is deterministic and produces the full token budget.
#[test]
fn generate_is_deterministic_cpu() {
    let model = build_tiny_llava();
    let feat = model.image_features(&gray(), V_IMG, V_IMG).unwrap();
    let prompt = [1i32, IMG_TOKEN, 2];
    let run = || {
        model
            .generate(
                &prompt,
                &feat,
                &greedy(),
                8,
                Some(0),
                &[], // no stop tokens: always produce the full budget
                &CancelFlag::new(),
                &mut |_, _| {},
            )
            .unwrap()
    };
    let a = run();
    let b = run();
    assert_eq!(a.tokens, b.tokens, "greedy caption must be deterministic");
    assert_eq!(a.tokens.len(), 8);
    assert!(a.tokens.iter().all(|&t| (t as usize) < VOCAB));
}

// ---- real-weights, env-gated (CUDA with --features cuda) --------------------------------------

mod real {
    use core_llm::{Content, ImageRef, LoadSpec, Message, Role, Sampling, TextLlm, TextLlmRequest};
    use core_llm_testkit::{check_multimodal, TextLlmProfile};

    use candle_llm::llava::LlavaProvider;

    fn snapshot() -> Option<String> {
        std::env::var("CANDLE_LLM_VLM_MODEL").ok()
    }

    /// A solid-gray 384×384 RGB image (the SigLIP input edge).
    fn gray_384() -> ImageRef {
        ImageRef::new(384, 384, vec![127u8; 384 * 384 * 3]).unwrap()
    }

    fn caption_request(max_new: u32) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    Content::Image(gray_384()),
                    Content::text("Describe this image in one short sentence."),
                ],
                thinking: None,
                tool_calls: Vec::new(),
            }],
            sampling: Sampling::greedy(),
            max_new_tokens: max_new,
            seed: Some(0),
            ..Default::default()
        }
    }

    /// A LLaVA snapshot captions an image through the contract, the multimodal conformance check
    /// passes via the **generate** branch, captioning is greedily deterministic, and a no-image
    /// request is rejected.
    #[test]
    #[ignore = "needs CANDLE_LLM_VLM_MODEL (a LlavaForConditionalGeneration snapshot dir)"]
    fn llava_captions_image_and_passes_multimodal_check() {
        let Some(snap) = snapshot() else {
            eprintln!("skip: set CANDLE_LLM_VLM_MODEL to a LLaVA snapshot dir");
            return;
        };
        let provider = LlavaProvider::load(&LoadSpec::dense(snap)).expect("load LLaVA provider");
        assert!(provider.descriptor().capabilities.supports_vision);

        // Caption a gray image.
        let mut streamed = String::new();
        let mut saw_done = false;
        let out = provider
            .generate(&caption_request(32), &mut |ev| match ev {
                core_llm::StreamEvent::Token { text, .. } => streamed.push_str(&text),
                core_llm::StreamEvent::Done { .. } => saw_done = true,
            })
            .expect("vision generate");
        println!("caption = {:?}", out.text);
        assert!(saw_done);
        assert!(out.usage.generated_tokens >= 3, "expected a real caption");
        assert!(
            out.text.chars().any(|c| c.is_ascii_alphabetic()),
            "caption should contain words: {:?}",
            out.text
        );
        assert_eq!(
            streamed.trim(),
            out.text.trim(),
            "streamed deltas reconstruct the caption"
        );

        // Greedy determinism: a second run is identical.
        let out2 = provider
            .generate(&caption_request(32), &mut |_| {})
            .unwrap();
        assert_eq!(out.text, out2.text, "greedy caption must be deterministic");

        // The conformance suite's multimodal check passes via the generate branch.
        check_multimodal(&provider, &TextLlmProfile::cheap())
            .expect("multimodal conformance check (generate branch)");

        // A request with no image is rejected (never silently captions nothing).
        let no_image = TextLlmRequest {
            messages: vec![Message::user("Describe this image.")],
            max_new_tokens: 8,
            ..Default::default()
        };
        assert!(provider.generate(&no_image, &mut |_| {}).is_err());
    }
}

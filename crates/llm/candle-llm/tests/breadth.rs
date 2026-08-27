//! Real-weights model-breadth tests (`#[ignore]` — need models on disk), story 7261.
//!
//! Each non-Llama architecture added to the `config.json` dispatch must stream coherent text through
//! the backend-neutral `core_llm::TextLlm` from a real HF snapshot. Point the per-family env var at a
//! snapshot dir and run (add `--features cuda` for the GPU path):
//!
//! ```text
//! CANDLE_LLM_PHI3_MODEL=/path/Phi-3-mini-4k-instruct \
//!   cargo test --features cuda --test breadth -- --ignored --nocapture
//! ```

use candle_llm::{load_for_model, provider::LlamaProvider};
use core_llm::{
    AudioRef, Content, ImageRef, LoadSpec, Message, Quantize, Role, Sampling, StreamEvent, TextLlm,
    TextLlmRequest,
};

/// Load the snapshot at `$env` **by model** (story 7406: `load_for_model`, naming no provider id /
/// family / backend), check its reported family tag, and assert it streams coherent, word-bearing
/// text (the streamed deltas reconstructing the final output).
fn assert_streams_coherent(env: &str, family: &str) {
    let Some(dir) = std::env::var(env).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: set {env}");
        return;
    };
    let spec = LoadSpec::dense(dir);
    // The weightless probe must accept the snapshot, and model-first resolution must route it to the
    // single generic candle text provider purely by architecture — the family is only known
    // post-load, so this exercises the `can_load`-not-`descriptor.family` resolution the story exists
    // for (e.g. Gemma2/GLM4/DeepSeek behind `candle-llama`).
    assert!(
        candle_llm::provider::can_load(&spec),
        "{family}: can_load must accept the snapshot"
    );
    let provider = load_for_model(&spec).expect("resolve + load provider by model");
    assert_eq!(
        provider.descriptor().id,
        "candle-llama",
        "resolved provider id"
    );
    assert_eq!(provider.descriptor().family, family, "reported family tag");

    let req = TextLlmRequest {
        messages: vec![Message::user("The capital of France is")],
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

    println!("[{family}] {}", out.text.replace('\n', " "));
    assert!(!out.text.trim().is_empty(), "{family}: produced no text");
    assert_eq!(
        streamed, out.text,
        "{family}: streamed deltas must reconstruct the final text"
    );
    assert!(
        out.text.chars().any(|c| c.is_alphabetic()),
        "{family}: output should contain words, not just punctuation"
    );
}

/// Phi-3: the Llama decoder shape with a packed `qkv_proj` + `gate_up_proj` (split at load).
#[test]
#[ignore = "needs a Phi-3 snapshot via CANDLE_LLM_PHI3_MODEL"]
fn phi3_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_PHI3_MODEL", "phi3");
}

/// Qwen2-MoE: Qwen2 attention (q/k/v bias) + a sparse MoE FFN (router + top-k experts + shared).
#[test]
#[ignore = "needs a Qwen2-MoE snapshot via CANDLE_LLM_QWEN2MOE_MODEL"]
fn qwen2_moe_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_QWEN2MOE_MODEL", "qwen2_moe");
}

/// Gemma-2: `(1+weight)` norms, embedding ×√hidden, GeGLU, soft-capped attention + final logits,
/// 4-norm sandwich block.
#[test]
#[ignore = "needs a Gemma-2 snapshot via CANDLE_LLM_GEMMA2_MODEL"]
fn gemma2_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_GEMMA2_MODEL", "gemma2");
}

/// GLM-4: 4-norm sandwich (standard RMSNorm), q/k/v bias, packed gate_up, and partial + interleaved
/// RoPE.
#[test]
#[ignore = "needs a GLM-4 snapshot via CANDLE_LLM_GLM4_MODEL"]
fn glm4_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_GLM4_MODEL", "glm4");
}

/// DeepSeek-V2: Multi-head Latent Attention (low-rank KV path + decoupled YaRN RoPE) and a fine-
/// grained MoE FFN (many routed experts + shared experts, a leading dense layer). Verified on
/// `deepseek-ai/DeepSeek-V2-Lite-Chat` (15.7B, fits 96GB).
#[test]
#[ignore = "needs a DeepSeek-V2 snapshot via CANDLE_LLM_DEEPSEEK_MODEL"]
fn deepseek_v2_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_DEEPSEEK_MODEL", "deepseek_v2");
}

/// Gemma 4 unified (sc-18761): the **per-layer-type** decoder — 40 `sliding_attention` layers
/// (head_dim 256, θ=10 000, window 1024, 8 KV heads) interleaved with 8 `full_attention` ones
/// (head_dim 512, proportional θ=1 000 000, 1 KV head, shared K/V), plain-`weight` norms, unit
/// attention scale, a scale-free value norm, and final-logit soft-capping at 30.
///
/// Point at a `google/gemma-4-12B-it` snapshot. `tests/gemma4_decoder.rs` carries the numeric
/// pinning on the deterministic CPU path; this is the real-weight coherence check the CUDA lane
/// runs — the shapes are large enough (48 layers, hidden 3840, vocab 262 144) that a per-layer table
/// wired wrong produces word salad rather than a shape error.
#[test]
#[ignore = "needs a Gemma 4 unified snapshot via CANDLE_LLM_GEMMA4_MODEL"]
fn gemma4_unified_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_GEMMA4_MODEL", "gemma4_unified");
}

// ---------------------------------------------------------------------------------------------
// Gemma 4 as a general-purpose LLM (sc-18772)
// ---------------------------------------------------------------------------------------------
//
// These are the real-weight legs of the story's acceptance: the synthetic suite
// (`gemma4_multimodal.rs`) proves the wiring on CPU in milliseconds, but only a trained checkpoint
// can show that Gemma 4 *answers* — and, for vision, that the tiling and positional resampling this
// crate had to choose without a reference implementation are actually right.
//
// Point `CANDLE_LLM_GEMMA4_MODEL` at a `google/gemma-4-12B-it` snapshot and run:
//
// ```text
// CANDLE_LLM_GEMMA4_MODEL=/path/to/gemma-4-12B-it \
//   cargo test -p candle-llm --test breadth -- --ignored --nocapture gemma4
// ```

/// Load the Gemma 4 snapshot at the story's env var with an explicit quantization tier, or print a
/// loud skip and return `None`. Never silently passes.
fn gemma4_provider(quantize: Option<Quantize>) -> Option<LlamaProvider> {
    let Some(dir) = std::env::var("CANDLE_LLM_GEMMA4_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        eprintln!("skip: set CANDLE_LLM_GEMMA4_MODEL to a google/gemma-4-12B-it snapshot");
        return None;
    };
    let spec = LoadSpec {
        source: dir,
        quantize,
    };
    Some(LlamaProvider::load(&spec).expect("load gemma 4"))
}

fn ask(p: &LlamaProvider, content: Vec<Content>, max_new_tokens: u32) -> String {
    let req = TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        ..Default::default()
    };
    p.generate(&req, &mut |_| {}).expect("generate").text
}

/// A `w x h` image split left/right into two flat colours.
fn split_image(w: u32, h: u32, left: [u8; 3], right: [u8; 3]) -> ImageRef {
    let mut px = Vec::with_capacity(w as usize * h as usize * 3);
    for _y in 0..h {
        for x in 0..w {
            px.extend_from_slice(if x < w / 2 { &left } else { &right });
        }
    }
    ImageRef::new(w, h, px).unwrap()
}

/// Gemma 4's chat template opens a reasoning channel (`<|channel>thought ... <channel|>`) before the
/// answer, and this provider does not advertise a thinking mode for it (the template gates reasoning
/// with its own `<|think|>` token rather than the `enable_thinking` kwarg the segmenter keys on), so
/// the markers arrive inline in the generated text.
///
/// Take everything after the final channel close as the answer. Without this the reasoning span is
/// scored as if it were the answer — and it routinely mentions BOTH colours on its way to the right
/// one, so a naive substring check would be both flaky and wrong.
fn answer_channel(text: &str) -> String {
    match text.rfind("<channel|>") {
        Some(i) => text[i + "<channel|>".len()..].to_string(),
        None => text.to_string(),
    }
}

/// Gemma 4 answers a prompt at a **quantized tier** through the ordinary provider surface.
///
/// The dense leg is `gemma4_unified_streams_coherent_text` above; this is the other half of the
/// story's "bf16 and a quantized tier" requirement. Q4 rather than Q8 because it is the tier that
/// stresses the per-layer-type table hardest: the full-attention layers' 512-wide heads and shared
/// K/V quantize on a different shape from the sliding layers' 256-wide ones.
#[test]
#[ignore = "needs a Gemma 4 snapshot via CANDLE_LLM_GEMMA4_MODEL"]
fn gemma4_answers_at_a_quantized_tier() {
    let Some(p) = gemma4_provider(Some(Quantize::Q4)) else {
        return;
    };
    assert!(p.is_quantized(), "the Q4 tier must actually be applied");
    let out = ask(&p, vec![Content::text("The capital of France is")], 12);
    println!("[gemma4 q4] {}", out.replace('\n', " "));
    assert!(!out.trim().is_empty(), "produced no text");
    assert!(
        out.chars().any(|c| c.is_alphabetic()),
        "output should contain words"
    );
    assert!(
        out.to_lowercase().contains("paris"),
        "a coherent Q4 Gemma 4 should name Paris; got {out:?}"
    );
}

/// **Image conditioning, demonstrated.** The model must answer a question whose only possible source
/// is the image, and must answer the mirrored image differently.
///
/// This is the leg that validates the two choices this crate had to make without a reference
/// implementation: the patch-grid tiling and the resampling of the 1120-entry positional table onto
/// that grid. A wrong tiling scrambles the patch order; a wrong (or dropped) positional resampling
/// makes left and right indistinguishable. Either one survives every shape assertion and shows up
/// only here, as the model getting the side wrong.
///
/// If this leg cannot be made to pass, the honest outcome is `supports_vision = false` in the
/// descriptor — advertising a vision path that answers at chance is exactly the failure the story
/// forbids.
#[test]
#[ignore = "needs a Gemma 4 snapshot via CANDLE_LLM_GEMMA4_MODEL"]
fn gemma4_answers_about_an_image() {
    let Some(p) = gemma4_provider(Some(Quantize::Q4)) else {
        return;
    };
    assert!(
        p.descriptor().capabilities.supports_vision,
        "the checkpoint must advertise vision for this leg to mean anything"
    );

    let question = "What colour is the left half of this image? Answer with one word.";
    let red_left = ask(
        &p,
        vec![
            Content::Image(split_image(896, 448, [220, 20, 20], [20, 20, 220])),
            Content::text(question),
        ],
        64,
    );
    let blue_left = ask(
        &p,
        vec![
            Content::Image(split_image(896, 448, [20, 20, 220], [220, 20, 20])),
            Content::text(question),
        ],
        64,
    );
    println!("[gemma4 image] red-left={red_left:?} blue-left={blue_left:?}");

    let (rl, bl) = (
        answer_channel(&red_left).to_lowercase(),
        answer_channel(&blue_left).to_lowercase(),
    );
    assert!(
        rl.contains("red") && !rl.contains("blue"),
        "a red left half must be answered 'red'; got {red_left:?}"
    );
    assert!(
        bl.contains("blue") && !bl.contains("red"),
        "a blue left half must be answered 'blue'; got {blue_left:?}"
    );
}

/// **Audio conditioning, demonstrated.** A clip measurably reaches the decoder: generation succeeds
/// with audio attached, and two clearly different waveforms produce different continuations.
///
/// Deliberately a *sensitivity* claim, not a transcription one. Gemma 4's audio front-end is a
/// single linear map over raw 640-sample frames with no encoder, so what this crate can honestly
/// demonstrate is that the samples reach the decoder and change what it says — not that the model
/// transcribes. Asserting a transcription would be asserting a capability of the checkpoint rather
/// than of this integration.
#[test]
#[ignore = "needs a Gemma 4 snapshot via CANDLE_LLM_GEMMA4_MODEL"]
fn gemma4_accepts_audio_conditioning() {
    let Some(p) = gemma4_provider(Some(Quantize::Q4)) else {
        return;
    };
    assert!(
        p.descriptor().capabilities.supports_audio,
        "the checkpoint must advertise audio for this leg to mean anything"
    );

    // One second at 16 kHz: a 440 Hz tone, and silence.
    let rate = 16_000usize;
    let tone: Vec<f32> = (0..rate)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
        .collect();
    let silence = vec![0.0f32; rate];

    let question = "Describe what you hear in a few words.";
    let a = ask(
        &p,
        vec![
            Content::Audio(AudioRef::new(16_000, tone).unwrap()),
            Content::text(question),
        ],
        16,
    );
    let b = ask(
        &p,
        vec![
            Content::Audio(AudioRef::new(16_000, silence).unwrap()),
            Content::text(question),
        ],
        16,
    );
    println!("[gemma4 audio] tone={a:?} silence={b:?}");

    assert!(!a.trim().is_empty(), "audio-conditioned generation produced no text");
    assert!(
        a.chars().any(|c| c.is_alphabetic()),
        "output should contain words"
    );
    assert_ne!(
        a, b,
        "a tone and silence must not produce identical continuations — identical output means the \
         audio never reached the decoder"
    );
}

//! Real-weights model-breadth tests (`#[ignore]` — need a snapshot on disk), story 7173.
//!
//! Each non-Llama architecture must load from a real HF snapshot and stream coherent text through the
//! backend-neutral `core_llm::TextLlm`, reporting the correct family tag (dispatched from
//! `config.json`). Point the per-family env var at a snapshot dir and run:
//!
//! ```text
//! MLX_LLM_GEMMA2_MODEL=/path/to/gemma-2-2b-it \
//!   cargo test --test integration -- real_breadth:: --ignored --nocapture
//! ```
//!
//! The synthetic-weights wiring gate (no download) is `tests/breadth.rs`; this is the parity-vs-real
//! check. Loading goes through the registered `mlx-llama` provider, whose descriptor family reflects
//! the architecture `config.json` dispatched to (the breadth lives behind one generic provider).

use core_llm::{
    AudioRef, Content, ImageRef, LoadSpec, Message, Quantize, Role, Sampling, StreamEvent, TextLlm,
    TextLlmRequest,
};
use mlx_llm::load_textllm;
use mlx_llm::provider::{LlamaProvider, PROVIDER_ID};

/// Load the snapshot at `$env` through the `mlx-llama` provider, check its reported family tag, and
/// assert it streams coherent, word-bearing text (the streamed deltas reconstructing the output).
fn assert_streams_coherent(env: &str, family: &str) {
    let Some(dir) = std::env::var(env).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: set {env}");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(&dir)).expect("load provider");
    assert_eq!(
        provider.descriptor().id,
        PROVIDER_ID,
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
#[ignore = "needs a Phi-3 snapshot via MLX_LLM_PHI3_MODEL"]
fn phi3_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_PHI3_MODEL", "phi3");
}

/// Qwen2-MoE: Qwen2 attention (q/k/v bias) + a sparse MoE FFN (router + top-k experts + shared).
#[test]
#[ignore = "needs a Qwen2-MoE snapshot via MLX_LLM_QWEN2MOE_MODEL"]
fn qwen2_moe_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_QWEN2MOE_MODEL", "qwen2_moe");
}

/// Gemma-2: `(1+weight)` norms, embedding ×√hidden, GeGLU, soft-capped attention + final logits,
/// the 4-norm sandwich block.
#[test]
#[ignore = "needs a Gemma-2 snapshot via MLX_LLM_GEMMA2_MODEL"]
fn gemma2_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_GEMMA2_MODEL", "gemma2");
}

/// GLM-4: 4-norm sandwich (standard RMSNorm), q/k/v bias, packed gate_up, partial + interleaved RoPE.
#[test]
#[ignore = "needs a GLM-4 snapshot via MLX_LLM_GLM4_MODEL"]
fn glm4_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_GLM4_MODEL", "glm4");
}

/// DeepSeek-V2: Multi-head Latent Attention (low-rank KV path + decoupled YaRN RoPE) and a
/// fine-grained MoE FFN (many routed experts + shared experts, a leading dense layer). Verified on
/// `deepseek-ai/DeepSeek-V2-Lite-Chat`.
#[test]
#[ignore = "needs a DeepSeek-V2 snapshot via MLX_LLM_DEEPSEEK_MODEL"]
fn deepseek_v2_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_DEEPSEEK_MODEL", "deepseek_v2");
}

// ---------------------------------------------------------------------------------------------
// Gemma 4 as a general-purpose LLM (sc-18772)
// ---------------------------------------------------------------------------------------------
//
// The MLX half of the story's real-weight acceptance. The synthetic wiring gate lives in
// `candle-llm`'s `gemma4_multimodal.rs` (identical host-side geometry, CPU-runnable in
// milliseconds); these legs need Metal and a `google/gemma-4-12B-it` snapshot:
//
// ```text
// MLX_LLM_GEMMA4_MODEL=/path/to/gemma-4-12B-it \
//   cargo test -p mlx-llm --test integration -- --ignored --nocapture gemma4
// ```
//
// MLX's default device is single-threaded and these hold one 12B model resident, so run them alone
// (`--test-threads=1`).

/// The env var carrying the Gemma 4 snapshot for these legs.
const GEMMA4_ENV: &str = "MLX_LLM_GEMMA4_MODEL";

/// Load the Gemma 4 snapshot at [`GEMMA4_ENV`] with an explicit quantization tier, or print a loud
/// skip and return `None`. Never silently passes.
fn gemma4_provider(quantize: Option<Quantize>) -> Option<LlamaProvider> {
    let Some(dir) = std::env::var(GEMMA4_ENV).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: set {GEMMA4_ENV} to a google/gemma-4-12B-it snapshot");
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

/// Gemma 4 unified at **bf16**, through the ordinary provider surface: the per-layer-type decoder
/// (40 sliding / 8 full layers, dual head dims, dual RoPE schedules, shared K/V on the full layers)
/// must stream coherent text and report its family tag.
///
/// This is also the leg that proves the HF `model.language_model.*` nesting loads at all — before
/// sc-18772 this crate only knew the flat `model.*` stem, so a `google/gemma-4-12B-it` snapshot
/// failed on its first weight lookup.
#[test]
#[ignore = "needs a Gemma 4 snapshot via MLX_LLM_GEMMA4_MODEL"]
fn gemma4_unified_streams_coherent_text() {
    assert_streams_coherent(GEMMA4_ENV, "gemma4_unified");
}

/// Gemma 4 at a **quantized tier** — the other half of the story's "bf16 and a quantized tier".
///
/// Q4 rather than Q8 because it stresses the per-layer-type table hardest: the full-attention
/// layers' 512-wide heads and shared K/V quantize on a different shape from the sliding layers'
/// 256-wide ones.
#[test]
#[ignore = "needs a Gemma 4 snapshot via MLX_LLM_GEMMA4_MODEL"]
fn gemma4_answers_at_a_quantized_tier() {
    let Some(p) = gemma4_provider(Some(Quantize::Q4)) else {
        return;
    };
    assert!(p.is_quantized(), "the Q4 tier must actually be applied");
    let out = ask(&p, vec![Content::text("The capital of France is")], 12);
    println!("[gemma4 q4] {}", out.replace('\n', " "));
    assert!(!out.trim().is_empty(), "produced no text");
    assert!(
        out.to_lowercase().contains("paris"),
        "a coherent Q4 Gemma 4 should name Paris; got {out:?}"
    );
}

/// **Image conditioning, demonstrated on MLX.** The model must answer a question whose only possible
/// source is the image, and answer the mirrored image differently.
///
/// This validates the two choices made without a reference implementation — the patch-grid tiling
/// and the resampling of the 1120-entry positional table onto that grid. A wrong tiling scrambles
/// patch order; a wrong or dropped positional resampling makes left and right indistinguishable.
/// Both survive every shape assertion and show up only here.
/// **Currently expected RED — vision is not advertised** (sc-18772). The candle leg of the same
/// name was measured against real weights and fails; see its doc comment for the evidence. Kept as
/// the gate that must go green before `supports_vision` is flipped on.
#[test]
#[ignore = "needs a Gemma 4 snapshot via MLX_LLM_GEMMA4_MODEL; currently RED - vision unvalidated"]
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

/// **Audio conditioning, demonstrated on MLX.** A clip measurably reaches the decoder: generation
/// succeeds with audio attached, and two clearly different waveforms produce different
/// continuations.
///
/// A sensitivity claim, not a transcription one — Gemma 4's audio front-end is a single linear map
/// over raw 640-sample frames with no encoder, so what this integration can honestly demonstrate is
/// that the samples reach the decoder and change what it says.
#[test]
#[ignore = "needs a Gemma 4 snapshot via MLX_LLM_GEMMA4_MODEL"]
fn gemma4_accepts_audio_conditioning() {
    let Some(p) = gemma4_provider(Some(Quantize::Q4)) else {
        return;
    };
    assert!(
        p.descriptor().capabilities.supports_audio,
        "the checkpoint must advertise audio for this leg to mean anything"
    );
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
    assert!(
        !a.trim().is_empty(),
        "audio-conditioned generation produced no text"
    );
    assert_ne!(
        a, b,
        "a tone and silence must not produce identical continuations — identical output means the \
         audio never reached the decoder"
    );
}

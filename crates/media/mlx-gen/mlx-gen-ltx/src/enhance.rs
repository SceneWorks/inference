//! S7 — LTX-2.3 prompt enhancement (sc-2845): rewrite the user prompt with Gemma-3 as an
//! autoregressive LLM before encoding. Optional, **default off**, and **not** numeric-parity (text
//! generation is stochastic and mlx-rs RNG isn't portable to mlx-python — a behavioral/smoke gate).
//!
//! Port of `mlx_video/models/ltx/text_encoder.py::LTX2TextEncoder.enhance_t2v / enhance_i2v` and
//! `models/ltx/enhance_prompt.py::enhance_with_model`, with the wiring from `generate_av.py`:
//! - Build the Gemma chat template (system turn + `"user prompt: {prompt}"` user turn + model turn).
//! - Tokenize with `add_special_tokens=false` (the template supplies the `<start_of_turn>` markers).
//! - Autoregressively sample (temperature 0.7; the censored path adds repetition-penalty 1.3 over a
//!   20-token window; top-k / top-p are disabled at the reference defaults but supported here) up to
//!   `max_tokens`, stopping on an end-of-turn / eos token.
//! - Detokenize the generated tokens and run [`clean_response`].
//!
//! The censored variant reuses the **already-loaded** text-encoder Gemma backbone
//! ([`GemmaModel::decode_logits`]); the uncensored variant loads a separate 4-bit Gemma — both go
//! through the same loop here ([`enhance`]), differing only in model + [`SampleParams`].
//!
//! **Stop tokens.** The reference hardcodes `token == 1 or token == 107`, but in the Gemma-3
//! tokenizer **107 is `\n`** (a newline) and `<end_of_turn>` is **106**; `generation_config.json`
//! gives the authoritative `eos_token_id = [1, 106]`. We stop on **{1, 106}** ([`STOP_TOKENS`]) —
//! the reference's `107` would truncate at the first newline (a latent bug in the reference).

use mlx_rs::{Array, Dtype};

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_llm::decode::{
    generate_cached_with, generate_from_prefill, ConstraintMask, GenerationConfig, StreamEvent,
};
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::{CausalLm, PrefixCache};
// The token sampler (temperature / top-k / top-p / repetition penalty) + seeded PRNG live in the core
// crate's shared `text_sample` module (sc-9561 / F-105) so the lens PromptReasoner reuses them rather
// than cloning. `SampleParams` stays part of this crate's public API via the re-export.
pub use mlx_gen::text_sample::SampleParams;
use mlx_gen::text_sample::{sample_token, SplitMix64};

use crate::gemma::GemmaModel;
use crate::tokenizer::LtxTokenizer;

/// Vendored default system prompts (the mlx_video wheel ships `enhance_prompt.py` / `text_encoder.py`
/// but **omits** the `prompts/` dir — so its enhancer silently FileNotFound→falls back; we vendor the
/// canonical `ltx_core` copies, identical across the SceneWorks venv and the upstream git checkout).
pub const T2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma_t2v_system_prompt.txt");
pub const I2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma_i2v_system_prompt.txt");
/// LTX-2 v1.2.0 Gemma-4 prompt contracts. These are separate assets: the Gemma-3 text and sampling
/// policy are not interchangeable with the Gemma-4 instruct enhancer.
pub const GEMMA4_T2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma4_t2v_system_prompt.txt");
pub const GEMMA4_I2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma4_i2v_system_prompt.txt");

/// Reference enhancement defaults (`generate_av.py` CLI).
pub const DEFAULT_MAX_TOKENS: usize = 512;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Reference enhancement default seed (`enhance_t2v(..., seed=42)`).
pub const DEFAULT_SEED: u64 = 42;
/// Upstream `GEMMA4_ENHANCE_GENERATION_KWARGS.max_new_tokens`.
pub const GEMMA4_DEFAULT_MAX_TOKENS: usize = 600;
/// Upstream Gemma-4 enhancer method default (`seed=10`; greedy makes it deterministic today).
pub const GEMMA4_DEFAULT_SEED: u64 = 10;
/// Upstream `GEMMA4_ENHANCE_GENERATION_KWARGS.no_repeat_ngram_size`.
pub const GEMMA4_NO_REPEAT_NGRAM: usize = 5;

/// Hard ceiling on enhance decode length (F-012 twin of the flux2 cap). Each decode step is a full
/// Gemma forward over a growing KV cache, so a request-supplied `enhance_max_tokens` must be capped
/// or a single `enhance_prompt=true` request becomes an effectively unbounded job. 4× the 512
/// reference default leaves room for legitimately long rewrites while bounding the worst case to
/// ~2048 forwards instead of billions. Cooperative cancellation ([`enhance`]'s `cancel`) also
/// interrupts the loop per decoded token (F-018).
pub const MAX_TOKENS_CAP: usize = 2048;

/// Resolve the decode budget from the request's `enhance_max_tokens`: the reference default
/// ([`DEFAULT_MAX_TOKENS`]) when unset, otherwise the requested value clamped to [`MAX_TOKENS_CAP`]
/// (F-012). A request is never *rejected* for asking too much — the advisory knob is silently capped
/// — so callers stay infallible. Inert on the happy path (the reference default is well under the cap).
pub fn clamp_max_tokens(requested: Option<u32>) -> usize {
    requested
        .map(|m| (m as usize).min(MAX_TOKENS_CAP))
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Resolve Gemma-4's distinct 600-token default while preserving the common request hard cap.
pub fn clamp_gemma4_max_tokens(requested: Option<u32>) -> usize {
    requested
        .map(|m| (m as usize).min(MAX_TOKENS_CAP))
        .unwrap_or(GEMMA4_DEFAULT_MAX_TOKENS)
}

/// Stop tokens: `<eos>` (1) and `<end_of_turn>` (106) — see the module note on the reference's `107`.
pub const STOP_TOKENS: [i32; 2] = [1, 106];

/// Per-call generation budget.
#[derive(Clone, Copy, Debug)]
pub struct EnhanceConfig {
    pub max_tokens: usize,
    pub seed: u64,
}

impl Default for EnhanceConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            seed: DEFAULT_SEED,
        }
    }
}

/// Build the Gemma-3 chat-templated string: a system turn, a `"user prompt: {prompt}"` user turn, and
/// the model generation prompt. Mirrors `_apply_chat_template([system, user])` and
/// `enhance_prompt._apply_chat_template(system, "user prompt: " + prompt)` (both produce this exact
/// string — system and user are both emitted as `user` turns in the reference).
fn chat_template(system_prompt: &str, user_prompt: &str) -> String {
    format!(
        "<start_of_turn>user\n{system_prompt}<end_of_turn>\n\
         <start_of_turn>user\nuser prompt: {user_prompt}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}

/// Reference `_clean_response`: strip surrounding whitespace, then drop a leading run of characters
/// that are neither word (`\w`: alphanumeric or `_`) nor whitespace (`\s`) — i.e. leading punctuation
/// / symbols (`re.sub(r"^[^\w\s]+", "", response)`).
pub fn clean_response(response: &str) -> String {
    let trimmed = response.trim();
    let cleaned = trimmed
        .trim_start_matches(|c: char| !(c.is_alphanumeric() || c == '_' || c.is_whitespace()));
    cleaned.to_string()
}

/// Run the autoregressive enhancement loop over `gemma` + `tokenizer`, returning the cleaned rewrite.
/// May return an empty string (e.g. the model immediately emits a stop token) — the caller decides
/// whether to fall back to the original prompt (the reference treats empty output as a failure).
/// `cancel` is the request's cooperative cancellation handle (F-018): checked before each of the up
/// to [`MAX_TOKENS_CAP`] Gemma decode steps and after the prefill, returning [`Error::Canceled`] so a
/// cancel during a multi-minute enhancement is honored (matching the denoise loops' per-step
/// contract). Each `decode_logits` step already forces a host sync, so the check observes the trip.
#[allow(clippy::too_many_arguments)]
pub fn enhance(
    gemma: &GemmaModel,
    tokenizer: &LtxTokenizer,
    system_prompt: &str,
    user_prompt: &str,
    cfg: &EnhanceConfig,
    sampler: &SampleParams,
    cancel: Option<&CancelFlag>,
) -> Result<String> {
    // Honor a cancel tripped before enhancement even begins (before the ~12B prefill forward, F-018).
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(Error::Canceled);
    }
    let formatted = chat_template(system_prompt, user_prompt);
    let prompt_ids = tokenizer.encode_chat(&formatted)?;
    if prompt_ids.is_empty() {
        return Ok(String::new());
    }

    // `history` carries the prompt + generated tokens; the repetition penalty looks at its tail (the
    // reference applies the penalty over `tokens[-context_size:]` of the running sequence).
    let mut history = prompt_ids.clone();
    let mut cache = gemma.new_cache();
    let mut rng = SplitMix64::new(cfg.seed);

    // Prefill on the full prompt → logits for the first generated token.
    let prompt_len = prompt_ids.len() as i32;
    let ids = Array::from_slice(&prompt_ids, &[1, prompt_len]);
    let mut logits = gemma.decode_logits(&ids, &mut cache, 0)?;

    let mut generated: Vec<i32> = Vec::new();
    for step in 0..cfg.max_tokens {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        // Pull the `[vocab]` logits to the host once, then draw from the shared host-side sampler.
        let logits_host = logits.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let next = sample_token(&logits_host, &history, sampler, &mut rng);
        generated.push(next);
        history.push(next);
        if STOP_TOKENS.contains(&next) {
            break;
        }
        // Feed the token back at its absolute position (the generated token at index `step` sits at
        // `prompt_len + step`, just past the prefilled prompt).
        let nxt = Array::from_slice(&[next], &[1, 1]);
        logits = gemma.decode_logits(&nxt, &mut cache, prompt_len + step as i32)?;
    }

    let text = tokenizer.decode(&generated)?;
    Ok(clean_response(&text))
}

/// The already-tokenized Gemma-4 prefill. T2V can reuse a cached textual prefix; I2V must prefill
/// the embeddings after the reference image's projected patch rows replace its soft-token span.
pub enum Gemma4EnhancePrefill {
    Text(Vec<i32>),
    Multimodal { input_ids: Vec<i32>, embeds: Array },
}

/// Preserve contract-bearing `mlx-llm` errors across the media-crate boundary. In particular, a
/// cancellation observed by the shared decode loop must remain `Canceled` all the way to the
/// worker instead of becoming an ordinary string error.
fn from_gemma4_decode(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Unsupported(message) => Error::Unsupported(message),
        mlx_llm::Error::Canceled => Error::Canceled,
        mlx_llm::Error::MissingTensor(key) => Error::MissingTensor(key),
        mlx_llm::Error::Io(error) => Error::Io(error),
        other => Error::Msg(format!("ltx_2_5 enhancer decode: {other}")),
    }
}

/// Hugging Face `NoRepeatNGramLogitsProcessor`, expressed through the shared decode constraint seam.
/// Before each greedy draw it bans the token that would complete any already-seen N-token gram.
struct NoRepeatNgram {
    n: usize,
    history: Vec<i32>,
    allowed: Vec<bool>,
}

impl NoRepeatNgram {
    fn new(n: usize, history: Vec<i32>, vocab_size: usize) -> Self {
        Self {
            n,
            history,
            allowed: vec![true; vocab_size],
        }
    }

    fn rebuild(&mut self) {
        self.allowed.fill(true);
        if self.n < 2 || self.history.len() < self.n - 1 {
            return;
        }
        let prefix = &self.history[self.history.len() - (self.n - 1)..];
        if self.history.len() < self.n {
            return;
        }
        for start in 0..=self.history.len() - self.n {
            if self.history[start..start + self.n - 1] == *prefix {
                let token = self.history[start + self.n - 1];
                if let Some(slot) = usize::try_from(token)
                    .ok()
                    .and_then(|index| self.allowed.get_mut(index))
                {
                    *slot = false;
                }
            }
        }
    }
}

impl ConstraintMask for NoRepeatNgram {
    fn allowed(&mut self) -> &[bool] {
        self.rebuild();
        &self.allowed
    }

    fn accept(&mut self, token: i32) {
        self.history.push(token);
    }
}

/// Run the v1.2.0 Gemma-4 enhancement generation policy over the shared decoder stack: greedy
/// decoding, five-gram suppression, final-logit soft-capping from `ModelConfig`, cancellation, and
/// the shared streaming loop. Text prefills use the reusable prefix cache; image prefills enter the
/// same loop after the caller's vision splice.
#[allow(clippy::too_many_arguments)]
pub fn enhance_gemma4(
    gemma: &CausalLm,
    tokenizer: &TextTokenizer,
    prefill: Gemma4EnhancePrefill,
    cfg: &EnhanceConfig,
    vocab_size: usize,
    cancel: &CancelFlag,
    prefix_cache: &mut PrefixCache,
) -> Result<String> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let history = match &prefill {
        Gemma4EnhancePrefill::Text(ids) => ids,
        Gemma4EnhancePrefill::Multimodal { input_ids, .. } => input_ids,
    };
    if history.is_empty() {
        return Ok(String::new());
    }
    let generation = GenerationConfig {
        max_new_tokens: cfg.max_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        seed: Some(cfg.seed),
        stop_tokens: STOP_TOKENS.to_vec(),
    };
    let mut no_repeat = NoRepeatNgram::new(GEMMA4_NO_REPEAT_NGRAM, history.clone(), vocab_size);
    let decode_cancel = mlx_llm::CancelFlag::new();
    let bridged_cancel = decode_cancel.clone();
    let mut on_event = |_event: StreamEvent| {
        if cancel.is_cancelled() {
            bridged_cancel.cancel();
        }
    };
    let output = match prefill {
        Gemma4EnhancePrefill::Text(prompt_ids) => generate_cached_with(
            gemma,
            &prompt_ids,
            &generation,
            &decode_cancel,
            &mut on_event,
            prefix_cache,
            Some(&mut no_repeat),
            None,
        ),
        Gemma4EnhancePrefill::Multimodal { input_ids, embeds } => {
            let mut cache = gemma.new_cache();
            let logits = gemma
                .decode_logits_from_embeds(&embeds, &mut cache, 0)
                .map_err(|e| Error::Msg(format!("ltx_2_5 enhancer multimodal prefill: {e}")))?;
            generate_from_prefill(
                gemma,
                &mut cache,
                logits,
                input_ids,
                &generation,
                &decode_cancel,
                &mut on_event,
                Some(&mut no_repeat),
                None,
            )
        }
    }
    .map_err(from_gemma4_decode)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }

    let ids: Vec<u32> = output.tokens.iter().map(|&id| id as u32).collect();
    Ok(clean_response(&tokenizer.decode(&ids, true)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest as _;

    #[test]
    fn clean_response_strips_leading_punctuation_and_whitespace() {
        assert_eq!(clean_response("  \n**Style: a fox"), "Style: a fox");
        assert_eq!(clean_response("\"quoted start"), "quoted start");
        // Faithful to the reference: `strip()` then `re.sub(r"^[^\w\s]+", "", …)` with NO final strip,
        // so the regex stops at the first whitespace and a space after the punctuation run survives.
        assert_eq!(clean_response("...:: hello"), " hello");
        // Already clean → unchanged (modulo surrounding whitespace).
        assert_eq!(clean_response("  a red fox  "), "a red fox");
        // Leading digits / underscores are word chars → preserved.
        assert_eq!(clean_response("3 cats"), "3 cats");
        // Empty / all-punctuation collapses to empty.
        assert_eq!(clean_response("   "), "");
        assert_eq!(clean_response("!!!"), "");
    }

    #[test]
    fn clamp_max_tokens_caps_pathological_request_only() {
        // Unset → reference default, untouched.
        assert_eq!(clamp_max_tokens(None), DEFAULT_MAX_TOKENS);
        // Below the cap → honored verbatim (happy path stays inert).
        assert_eq!(clamp_max_tokens(Some(1)), 1);
        assert_eq!(clamp_max_tokens(Some(256)), 256);
        // Exactly at the cap → honored.
        assert_eq!(
            clamp_max_tokens(Some(MAX_TOKENS_CAP as u32)),
            MAX_TOKENS_CAP
        );
        // Above the cap (incl. u32::MAX, the unbounded-job case) → clamped to the cap, not rejected.
        assert_eq!(
            clamp_max_tokens(Some(MAX_TOKENS_CAP as u32 + 1)),
            MAX_TOKENS_CAP
        );
        assert_eq!(clamp_max_tokens(Some(u32::MAX)), MAX_TOKENS_CAP);
        assert_eq!(clamp_gemma4_max_tokens(None), GEMMA4_DEFAULT_MAX_TOKENS);
        assert_eq!(clamp_gemma4_max_tokens(Some(64)), 64);
    }

    #[test]
    fn chat_template_matches_reference_format() {
        let t = chat_template("SYS", "a fox");
        assert_eq!(
            t,
            "<start_of_turn>user\nSYS<end_of_turn>\n\
             <start_of_turn>user\nuser prompt: a fox<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn vendored_prompts_are_present_and_nonempty() {
        assert!(T2V_SYSTEM_PROMPT.contains("Creative Assistant"));
        assert!(I2V_SYSTEM_PROMPT.contains("image-to-video"));
        assert!(GEMMA4_T2V_SYSTEM_PROMPT.contains("audio-visual caption"));
        assert!(GEMMA4_I2V_SYSTEM_PROMPT.contains("REFERENCE IMAGE"));
        assert_ne!(GEMMA4_T2V_SYSTEM_PROMPT, T2V_SYSTEM_PROMPT);
        assert_ne!(GEMMA4_I2V_SYSTEM_PROMPT, I2V_SYSTEM_PROMPT);
    }

    #[test]
    fn gemma4_v120_prompts_are_exact_pinned_upstream_bytes() {
        let sha256 = |bytes: &[u8]| format!("{:x}", sha2::Sha256::digest(bytes));
        assert_eq!(GEMMA4_T2V_SYSTEM_PROMPT.len(), 3_769);
        assert_eq!(
            sha256(GEMMA4_T2V_SYSTEM_PROMPT.as_bytes()),
            "0cddf69456bcd51e65430f848386295d9ac4d17d5df3ea65d5f3d8a9ad842f3c"
        );
        assert_eq!(GEMMA4_I2V_SYSTEM_PROMPT.len(), 4_708);
        assert_eq!(
            sha256(GEMMA4_I2V_SYSTEM_PROMPT.as_bytes()),
            "15992bfb757d3bbd83f2d27ad86e450fc4caffa0f7cb7523772a60e346ef3fee"
        );
    }

    #[test]
    fn gemma4_no_repeat_ngram_bans_only_the_repeated_completion() {
        let mut constraint = NoRepeatNgram::new(5, vec![1, 2, 3, 4, 9, 7, 1, 2, 3, 4], 16);
        let mask = constraint.allowed();
        assert!(!mask[9], "token 9 would repeat [1,2,3,4,9]");
        assert!(mask[8]);
        constraint.accept(8);
        assert!(
            constraint.allowed()[9],
            "the suffix changed after accepting 8"
        );
    }

    #[test]
    fn gemma4_shared_decode_cancellation_stays_typed() {
        assert!(matches!(
            from_gemma4_decode(mlx_llm::Error::Canceled),
            Error::Canceled
        ));
    }

    // `SampleParams` presets + `SplitMix64` determinism are covered in the shared
    // `mlx_gen::text_sample` tests (the sampler now lives there — sc-9561 / F-105).
}

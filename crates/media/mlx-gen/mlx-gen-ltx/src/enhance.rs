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
//! through the same decode here ([`enhance`]), differing only in model + [`SampleParams`]. The
//! decode is the shared MLX engine's token-at-a-time loop ([`generate_speculative`] with
//! [`NoProposer`], epic sc-24432 E8) over the Gemma-3 decoder, drawing every token through this
//! pipeline's own host sampler ([`sample_token`] with its seeded [`SplitMix64`]) so a seeded
//! enhancement is token-identical to the pre-engine loop.
//!
//! **Stop tokens.** The reference hardcodes `token == 1 or token == 107`, but in the Gemma-3
//! tokenizer **107 is `\n`** (a newline) and `<end_of_turn>` is **106**; `generation_config.json`
//! gives the authoritative `eos_token_id = [1, 106]`. We stop on **{1, 106}** ([`STOP_TOKENS`]) —
//! the reference's `107` would truncate at the first newline (a latent bug in the reference).

use mlx_rs::{Array, Dtype};

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_llm::core_llm::{DecodeReport, HostSampleReason, PathReport, SamplerPath};
use mlx_llm::decode::{
    generate_speculative, prefill_with_prefix, CacheRollback, ConstraintMask, EngineOptions,
    FinishReason, GenerationConfig, LogitsScope, NoProposer, Pipelining, PrefixPrefill,
    RewindableConstraintMask, Rollback, SampledToken, SpeculativePrompt, SpeculativeRun,
    SpeculativeTarget, StreamEvent, TargetOutput, TokenSampler,
};
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::{CausalLm, PrefixCache};
// The token sampler (temperature / top-k / top-p / repetition penalty) + seeded PRNG live in the core
// crate's shared `text_sample` module (sc-9561 / F-105) so the lens PromptReasoner reuses them rather
// than cloning. `SampleParams` stays part of this crate's public API via the re-export.
pub use mlx_gen::text_sample::SampleParams;
use mlx_gen::text_sample::{sample_token, SplitMix64};

use crate::gemma::{GemmaKvCache, GemmaModel};
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
    let generated = decode_gemma3(gemma, &prompt_ids, cfg, sampler, cancel)?;
    let text = tokenizer.decode(&generated)?;
    Ok(clean_response(&text))
}

/// The Gemma-3 decoder ([`GemmaModel::decode_logits`]) as a target of the shared MLX engine. It
/// serves the token-at-a-time loop only: [`decode_logits`](GemmaModel::decode_logits) returns the
/// last position's logits, so a verify forward over drafts is refused, never approximated.
struct Gemma3Target<'a>(&'a GemmaModel);

/// The rollback of a target that never verifies drafts: the token-at-a-time loop never arms it,
/// and a recovery is refused.
struct NoDraftRollback;

impl CacheRollback<GemmaKvCache> for NoDraftRollback {
    fn label(&self) -> &'static str {
        "none"
    }

    fn begin(&mut self, _: &mut GemmaKvCache) {}

    fn recover(&mut self, _: &mut GemmaKvCache, _: i32) -> mlx_llm::Result<Rollback> {
        Err(mlx_llm::Error::Unsupported(
            "the LTX-2.3 Gemma-3 enhancer decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }
}

impl SpeculativeTarget for Gemma3Target<'_> {
    type Cache = GemmaKvCache;
    type Rollback = NoDraftRollback;

    fn new_cache(&self) -> GemmaKvCache {
        self.0.new_cache()
    }

    fn cache_len(&self, cache: &GemmaKvCache) -> i32 {
        cache.offset()
    }

    fn rollback(&self, _: usize) -> NoDraftRollback {
        NoDraftRollback
    }

    fn forward(
        &self,
        cache: &mut GemmaKvCache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> mlx_llm::Result<TargetOutput> {
        if want_hidden || scope == LogitsScope::All {
            return Err(mlx_llm::Error::Unsupported(
                "the LTX-2.3 Gemma-3 enhancer target returns last-position logits only".into(),
            ));
        }
        let logits = self
            .0
            .decode_logits(ids, cache, rope_offset)
            .map_err(|e| mlx_llm::Error::Msg(format!("ltx-2.3 enhancer forward: {e}")))?;
        Ok(TargetOutput {
            logits,
            hidden: None,
        })
    }

    fn attention_label(&self) -> &'static str {
        // Native-GQA fused SDPA over the K/V cache (`GemmaModel::attn_step`).
        "gqa"
    }
}

/// The LTX-2.3 enhancer's draw on the engine's sampler seam: mlx-gen's shared host sampler
/// [`sample_token`] over the f32 host logits and the running prompt + generated history, from the
/// pipeline's own seeded [`SplitMix64`] — exactly the draw the pre-engine loop made (its
/// repetition penalty, top-k and nucleus follow the reference `make_sampler` /
/// `make_logits_processors`, not mlx-llm's device sampler), so a seeded enhancement is
/// token-identical. It reads the history (the penalty window), so the engine never pipelines it.
struct EnhanceSampler<'a> {
    knobs: &'a SampleParams,
    /// The same knobs in the engine's vocabulary (reported, never drawn from).
    params: SamplingParams,
    rng: SplitMix64,
    draws: u64,
    /// The last token drawn — on a stop-token end, the stop token the engine does not emit.
    last: Option<i32>,
}

impl<'a> EnhanceSampler<'a> {
    fn new(knobs: &'a SampleParams, seed: u64) -> Self {
        Self {
            knobs,
            params: SamplingParams {
                temperature: knobs.temperature,
                top_p: knobs.top_p,
                top_k: usize::try_from(knobs.top_k).unwrap_or(0),
                presence_penalty: 0.0,
                repetition_penalty: knobs.repetition_penalty.unwrap_or(1.0),
                repetition_context: knobs.repetition_context,
            },
            rng: SplitMix64::new(seed),
            draws: 0,
            last: None,
        }
    }
}

impl TokenSampler for EnhanceSampler<'_> {
    fn params(&self) -> &SamplingParams {
        &self.params
    }

    fn sample(
        &mut self,
        logits: &Array,
        history: &[i32],
        allowed: Option<&[bool]>,
    ) -> mlx_llm::Result<SampledToken> {
        if allowed.is_some() {
            return Err(mlx_llm::Error::Unsupported(
                "the LTX-2.3 enhancer sampler takes no constraint mask".into(),
            ));
        }
        // Pull the `[vocab]` logits to the host once, then draw from the shared host-side sampler.
        let logits_host = logits.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let token = sample_token(&logits_host, history, self.knobs, &mut self.rng);
        self.draws += 1;
        self.last = Some(token);
        Ok(SampledToken::Host(token))
    }

    fn argmax_rows(&mut self, _: &Array) -> mlx_llm::Result<Vec<i32>> {
        Err(mlx_llm::Error::Unsupported(
            "the LTX-2.3 enhancer decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }

    fn distribution(
        &mut self,
        _: &Array,
        _: &[i32],
        _: Option<&[bool]>,
    ) -> mlx_llm::Result<Vec<(i32, f32)>> {
        Err(mlx_llm::Error::Unsupported(
            "the LTX-2.3 enhancer decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }

    fn uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }

    fn path(&self) -> Option<SamplerPath> {
        (self.draws > 0).then_some(SamplerPath::Host(HostSampleReason::Reference))
    }
}

/// The LTX-2.3 enhancement decode on the shared MLX engine ([`generate_speculative`] with
/// [`NoProposer`]): prefill the prompt, then draw up to `cfg.max_tokens` tokens through
/// [`EnhanceSampler`], stopping on [`STOP_TOKENS`]. Returns the generated ids **including** a final
/// stop token (the pre-engine loop's list; the detokenizer drops special tokens). `cancel` is
/// checked after the prefill and, bridged onto the engine's flag, after every emitted token; a
/// cancel returns [`Error::Canceled`].
fn decode_gemma3(
    gemma: &GemmaModel,
    prompt_ids: &[i32],
    cfg: &EnhanceConfig,
    sampler: &SampleParams,
    cancel: Option<&CancelFlag>,
) -> Result<Vec<i32>> {
    let target = Gemma3Target(gemma);
    let mut cache = gemma.new_cache();
    // Prefill on the full prompt → logits for the first generated token.
    let ids = Array::from_slice(prompt_ids, &[1, prompt_ids.len() as i32]);
    let logits = gemma.decode_logits(&ids, &mut cache, 0)?;
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(Error::Canceled);
    }
    let mut draw = EnhanceSampler::new(sampler, cfg.seed);
    let generation = GenerationConfig {
        max_new_tokens: cfg.max_tokens,
        sampling: draw.params,
        seed: Some(cfg.seed),
        stop_tokens: STOP_TOKENS.to_vec(),
    };
    let decode_cancel = mlx_llm::CancelFlag::new();
    let bridged_cancel = decode_cancel.clone();
    let mut on_event = |_event: StreamEvent| {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            bridged_cancel.cancel();
        }
    };
    let run = generate_speculative(
        &target,
        &mut NoProposer,
        SpeculativePrompt::Prefilled {
            cache: &mut cache,
            logits,
            hidden: None,
            history: prompt_ids,
            position_delta: 0,
        },
        &generation,
        0,
        &decode_cancel,
        &mut on_event,
        EngineOptions {
            sampler: Some(&mut draw),
            // Every draw reads the host history, so nothing could pipeline; say so explicitly.
            pipelining: Pipelining::Off,
            ..EngineOptions::default()
        },
    )
    .map_err(|e| from_llm_decode(e, "ltx-2.3 enhancer decode"))?;
    let mut generated = run.output.tokens;
    match run.output.finish_reason {
        FinishReason::Cancelled => return Err(Error::Canceled),
        FinishReason::StopToken => generated.extend(draw.last),
        _ => {}
    }
    Ok(generated)
}

/// The already-tokenized Gemma-4 prefill. T2V can reuse a cached textual prefix; I2V must prefill
/// the embeddings after the reference image's projected patch rows replace its soft-token span.
pub enum Gemma4EnhancePrefill {
    Text(Vec<i32>),
    Multimodal { input_ids: Vec<i32>, embeds: Array },
}

/// Preserve contract-bearing `mlx-llm` errors across the media-crate boundary. In particular, a
/// cancellation observed by the shared decode loop must remain `Canceled` all the way to the
/// worker instead of becoming an ordinary string error. `context` names the decode in any other
/// error.
fn from_llm_decode(e: mlx_llm::Error, context: &str) -> Error {
    match e {
        mlx_llm::Error::Unsupported(message) => Error::Unsupported(message),
        mlx_llm::Error::Canceled => Error::Canceled,
        mlx_llm::Error::MissingTensor(key) => Error::MissingTensor(key),
        mlx_llm::Error::Io(error) => Error::Io(error),
        mlx_llm::Error::IncoherentLoad {
            name,
            bytes,
            cpu,
            gpu,
            attempts,
        } => Error::IncoherentLoad {
            name,
            bytes,
            cpu,
            gpu,
            attempts,
        },
        other => Error::Msg(format!("{context}: {other}")),
    }
}

/// [`from_llm_decode`] for the LTX-2.5 Gemma-4 enhancer.
fn from_gemma4_decode(e: mlx_llm::Error) -> Error {
    from_llm_decode(e, "ltx_2_5 enhancer decode")
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

/// The constraint's whole state is the token history (the mask is rebuilt from it on every
/// [`ConstraintMask::allowed`]), so a checkpoint is the history length and a rewind truncates the
/// history back to it. The token-at-a-time engine run the enhancer takes never rewinds; this keeps
/// the constraint exact under any proposer the shared engine can run.
impl RewindableConstraintMask for NoRepeatNgram {
    fn checkpoint(&self) -> usize {
        self.history.len()
    }

    fn rewind(&mut self, checkpoint: usize) {
        self.history.truncate(checkpoint);
    }
}

/// Why the multimodal (I2V) enhancer prefill never reads or feeds the prefix cache.
const MULTIMODAL_PREFIX_BYPASS: &str =
    "a multimodal prompt is never cached — its image rows are not in the token key";

/// The v1.2.0 Gemma-4 enhancement sampling policy: greedy, bounded by `cfg`, stopping on
/// [`STOP_TOKENS`].
fn gemma4_generation(cfg: &EnhanceConfig) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: cfg.max_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        seed: Some(cfg.seed),
        stop_tokens: STOP_TOKENS.to_vec(),
    }
}

/// The v1.2.0 Gemma-4 enhancement decode on the shared MLX engine ([`generate_speculative`] with
/// [`NoProposer`]): greedy, five-gram suppression through the engine's constraint seam, the
/// request's cancel bridged per token. Text prefills restore and feed the reusable prefix cache;
/// image prefills enter after the caller's vision splice and bypass it. The returned run carries
/// the engine's [`DecodeReport`] with the prefix cache's part filled in.
///
/// The no-repeat mask is host state that the next draw reads, so the engine draws on the host
/// (`sampler = host:constraint`) and does not pipeline: a look-ahead draw would need the mask
/// advanced by a token the host has not read yet.
fn decode_gemma4(
    gemma: &CausalLm,
    prefill: Gemma4EnhancePrefill,
    cfg: &EnhanceConfig,
    vocab_size: usize,
    cancel: &CancelFlag,
    prefix_cache: &mut PrefixCache,
) -> Result<SpeculativeRun> {
    let history = match &prefill {
        Gemma4EnhancePrefill::Text(ids) => ids,
        Gemma4EnhancePrefill::Multimodal { input_ids, .. } => input_ids,
    };
    let generation = gemma4_generation(cfg);
    let mut no_repeat = NoRepeatNgram::new(GEMMA4_NO_REPEAT_NGRAM, history.clone(), vocab_size);
    let decode_cancel = mlx_llm::CancelFlag::new();
    let bridged_cancel = decode_cancel.clone();
    let mut on_event = |_event: StreamEvent| {
        if cancel.is_cancelled() {
            bridged_cancel.cancel();
        }
    };
    let options = EngineOptions {
        constraint: Some(&mut no_repeat),
        ..EngineOptions::default()
    };
    match prefill {
        Gemma4EnhancePrefill::Text(prompt_ids) => {
            let PrefixPrefill {
                mut cache,
                logits,
                fed_tokens,
                ..
            } = prefill_with_prefix(
                gemma,
                prefix_cache,
                &prompt_ids,
                None,
                false,
                &decode_cancel,
            )
            .map_err(from_gemma4_decode)?;
            let mut run = generate_speculative(
                gemma,
                &mut NoProposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits,
                    hidden: None,
                    history: &prompt_ids,
                    position_delta: 0,
                },
                &generation,
                0,
                &decode_cancel,
                &mut on_event,
                options,
            )
            .map_err(from_gemma4_decode)?;
            prefix_cache
                .store_run(&prompt_ids, &run, cache, None, None)
                .map_err(from_gemma4_decode)?;
            // Measured, not looked up: the prompt positions the prefill did not feed.
            let hit = prompt_ids.len() - fed_tokens;
            run.report.prefix_hit_tokens = hit as u64;
            run.report.prefix_cache = PathReport {
                path: if hit > 0 { "hit" } else { "miss" }.into(),
                reason: None,
            };
            Ok(run)
        }
        Gemma4EnhancePrefill::Multimodal { input_ids, embeds } => {
            let mut cache = gemma.new_cache();
            let logits = gemma
                .decode_logits_from_embeds(&embeds, &mut cache, 0)
                .map_err(|e| Error::Msg(format!("ltx_2_5 enhancer multimodal prefill: {e}")))?;
            let mut run = generate_speculative(
                gemma,
                &mut NoProposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits,
                    hidden: None,
                    history: &input_ids,
                    position_delta: 0,
                },
                &generation,
                0,
                &decode_cancel,
                &mut on_event,
                options,
            )
            .map_err(from_gemma4_decode)?;
            run.report.prefix_cache = PathReport {
                path: "bypassed".into(),
                reason: Some(MULTIMODAL_PREFIX_BYPASS.into()),
            };
            Ok(run)
        }
    }
}

/// Run the v1.2.0 Gemma-4 enhancement generation policy over the shared decoder stack and the
/// shared MLX decode engine: greedy decoding, five-gram suppression, final-logit soft-capping from
/// `ModelConfig`, cancellation. Text prefills use the reusable prefix cache; image prefills enter
/// the same engine after the caller's vision splice. Returns the cleaned rewrite and the engine's
/// measured [`DecodeReport`] for the enhancement decode.
#[allow(clippy::too_many_arguments)]
pub fn enhance_gemma4(
    gemma: &CausalLm,
    tokenizer: &TextTokenizer,
    prefill: Gemma4EnhancePrefill,
    cfg: &EnhanceConfig,
    vocab_size: usize,
    cancel: &CancelFlag,
    prefix_cache: &mut PrefixCache,
) -> Result<(String, DecodeReport)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let empty = match &prefill {
        Gemma4EnhancePrefill::Text(ids) => ids.is_empty(),
        Gemma4EnhancePrefill::Multimodal { input_ids, .. } => input_ids.is_empty(),
    };
    if empty {
        return Ok((String::new(), DecodeReport::default()));
    }
    let run = decode_gemma4(gemma, prefill, cfg, vocab_size, cancel, prefix_cache)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }

    let ids: Vec<u32> = run.output.tokens.iter().map(|&id| id as u32).collect();
    Ok((clean_response(&tokenizer.decode(&ids, true)?), run.report))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-engine LTX-2.3 enhancement loop (sc-2845), verbatim but for its tokenizer: the
    /// oracle the engine port must reproduce token for token (E1/E8). It returns the generated ids
    /// including a final stop token.
    fn reference_gemma3_tokens(
        gemma: &GemmaModel,
        prompt_ids: &[i32],
        cfg: &EnhanceConfig,
        sampler: &SampleParams,
    ) -> Vec<i32> {
        let mut history = prompt_ids.to_vec();
        let mut cache = gemma.new_cache();
        let mut rng = SplitMix64::new(cfg.seed);
        let prompt_len = prompt_ids.len() as i32;
        let ids = Array::from_slice(prompt_ids, &[1, prompt_len]);
        let mut logits = gemma.decode_logits(&ids, &mut cache, 0).unwrap();
        let mut generated: Vec<i32> = Vec::new();
        for step in 0..cfg.max_tokens {
            let logits_host = logits
                .as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            let next = sample_token(&logits_host, &history, sampler, &mut rng);
            generated.push(next);
            history.push(next);
            if STOP_TOKENS.contains(&next) {
                break;
            }
            let nxt = Array::from_slice(&[next], &[1, 1]);
            logits = gemma
                .decode_logits(&nxt, &mut cache, prompt_len + step as i32)
                .unwrap();
        }
        generated
    }

    /// A tiny synthetic Gemma-3 with a vocabulary that holds both stop tokens and a sliding window
    /// short enough that the decode crosses it.
    fn tiny_gemma3() -> GemmaModel {
        use crate::gemma::hidden_state_pinning_tests::{tiny_cfg, tiny_weights};
        let mut cfg = tiny_cfg(3);
        cfg.sliding_window_pattern = 2;
        cfg.sliding_window = 6;
        let w = tiny_weights(&cfg, 128, 0.5);
        GemmaModel::from_weights_with_prefix(&w, cfg, None, "").unwrap()
    }

    /// E8 (sc-24446): the LTX-2.3 enhancer runs on the shared engine and is token-identical to the
    /// pre-engine loop — greedy, the censored sampler (temperature + repetition penalty), the
    /// uncensored one and a top-k / nucleus mix, over several seeds — with both a stop-token end
    /// and a budget end among the cases.
    #[test]
    fn gemma3_enhancer_engine_decode_matches_the_pre_engine_loop() {
        let gemma = tiny_gemma3();
        let prompt: Vec<i32> = vec![2, 5, 9, 17, 33, 64, 100, 7];
        let samplers = [
            SampleParams::temperature(0.0),
            SampleParams::censored(0.7),
            SampleParams::uncensored(1.0),
            SampleParams {
                temperature: 0.9,
                top_k: 12,
                top_p: 0.8,
                repetition_penalty: Some(1.1),
                repetition_context: 5,
            },
        ];
        let (mut stopped, mut budget) = (0, 0);
        for sampler in &samplers {
            for seed in 0..6u64 {
                let cfg = EnhanceConfig {
                    max_tokens: 20,
                    seed,
                };
                let want = reference_gemma3_tokens(&gemma, &prompt, &cfg, sampler);
                let got = decode_gemma3(&gemma, &prompt, &cfg, sampler, None).unwrap();
                assert_eq!(got, want, "{sampler:?} seed {seed}");
                if want.last().is_some_and(|t| STOP_TOKENS.contains(t)) {
                    stopped += 1;
                } else {
                    assert_eq!(want.len(), cfg.max_tokens, "{sampler:?} seed {seed}");
                    budget += 1;
                }
            }
        }
        assert!(
            stopped > 0 && budget > 0,
            "stopped {stopped}, budget {budget}"
        );
        // A cancel observed after the prefill stays typed, as the pre-engine loop's did.
        let cancel = CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            decode_gemma3(
                &gemma,
                &prompt,
                &EnhanceConfig::default(),
                &SampleParams::censored(0.7),
                Some(&cancel)
            ),
            Err(Error::Canceled)
        ));
    }
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

    #[test]
    fn gemma4_no_repeat_ngram_rewind_restores_the_checkpointed_mask() {
        let mut constraint = NoRepeatNgram::new(3, vec![4, 5, 6, 4, 5], 8);
        let before = constraint.allowed().to_vec();
        assert!(!before[6], "token 6 would repeat [4,5,6]");
        let checkpoint = constraint.checkpoint();
        for token in [7, 4, 5] {
            constraint.accept(token);
        }
        let explored = constraint.allowed().to_vec();
        assert!(!explored[7], "the explored history also bans 7 after [4,5]");
        constraint.rewind(checkpoint);
        assert_eq!(
            constraint.allowed(),
            before.as_slice(),
            "a rewind must restore exactly the checkpointed mask"
        );
    }

    /// The 4-layer `gemma4_unified` fixture `mlx-llm`'s decoder goldens are built from (the same
    /// one `gemma4_te`'s residency tests load): a real Gemma-4 config — sliding/full alternation,
    /// final-logit soft-capping — and its complete weight set. Returns the model and its vocab.
    fn tiny_gemma4() -> (CausalLm, usize) {
        const DECODER_GOLDENS: &str =
            include_str!("../../../../llm/testdata/gemma4/gemma4_decoder_goldens.json");
        let goldens: serde_json::Value =
            serde_json::from_str(DECODER_GOLDENS).expect("parse gemma4 decoder goldens");
        let cfg = mlx_llm::ModelConfig::from_json(&goldens["config"]).expect("fixture config");
        assert!(
            cfg.is_gemma4(),
            "the fixture must exercise the Gemma-4 path"
        );
        let floats = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        };
        let mut map = std::collections::HashMap::new();
        for (key, entry) in goldens["weights"].as_object().expect("weights object") {
            let shape: Vec<i32> = floats(&entry["shape"]).iter().map(|&x| x as i32).collect();
            let data: Vec<f32> = floats(&entry["data"]).iter().map(|&x| x as f32).collect();
            map.insert(key.clone(), Array::from_slice(&data, &shape));
        }
        let vocab = cfg.vocab_size as usize;
        let weights = mlx_llm::primitives::Weights::from_map(map);
        (
            CausalLm::from_weights(&weights, "", cfg).expect("tiny Gemma-4 CausalLm"),
            vocab,
        )
    }

    /// A prompt avoiding the fixture's stop id `1` whose greedy run is position-sensitive on this
    /// fixture: shifting the decode RoPE positions by one changes its tokens from the second on,
    /// so token identity also pins the engine's decode positions (many fixture prompts' argmax
    /// margins are wide enough to hide an off-by-one).
    const PARITY_PROMPT: [i32; 12] = [22, 26, 11, 20, 8, 16, 38, 35, 25, 7, 14, 21];

    fn parity_cfg() -> EnhanceConfig {
        EnhanceConfig {
            max_tokens: 48,
            seed: GEMMA4_DEFAULT_SEED,
        }
    }

    fn no_repeat(prompt: &[i32], vocab: usize) -> NoRepeatNgram {
        NoRepeatNgram::new(GEMMA4_NO_REPEAT_NGRAM, prompt.to_vec(), vocab)
    }

    fn has_repeated_ngram(tokens: &[i32], n: usize) -> bool {
        let grams: Vec<&[i32]> = tokens.windows(n).collect();
        grams
            .iter()
            .enumerate()
            .any(|(i, g)| grams[i + 1..].contains(g))
    }

    /// E8: the enhancer's text decode runs the shared engine, token-for-token the pre-engine plain
    /// loop (`generate_cached_with`, i.e. `stream::decode_loop`) under the same five-gram
    /// constraint, and a second request served from the prefix cache matches too. The report names
    /// what ran: no proposer, the host sampler for the constraint, the prefix cache miss then hit.
    #[test]
    fn gemma4_text_enhancer_engine_decode_matches_the_plain_loop() {
        let (model, vocab) = tiny_gemma4();
        let prompt = PARITY_PROMPT.to_vec();
        let cfg = parity_cfg();
        let generation = gemma4_generation(&cfg);

        let reference = mlx_llm::decode::generate_cached_with(
            &model,
            &prompt,
            &generation,
            &mlx_llm::CancelFlag::new(),
            &mut |_| {},
            &mut PrefixCache::with_budget(1 << 30),
            Some(&mut no_repeat(&prompt, vocab)),
            None,
        )
        .expect("plain loop")
        .tokens;
        assert!(
            reference.len() >= 16,
            "the fixture must decode a non-trivial run, got {reference:?}"
        );
        let unconstrained = mlx_llm::decode::generate(
            &model,
            &prompt,
            &generation,
            &mlx_llm::CancelFlag::new(),
            &mut |_| {},
        )
        .expect("unconstrained plain loop")
        .tokens;
        assert!(
            has_repeated_ngram(&unconstrained, GEMMA4_NO_REPEAT_NGRAM)
                && unconstrained != reference,
            "the five-gram ban must change this fixture's greedy run, or parity proves nothing \
             about the constraint: {unconstrained:?}"
        );

        let mut cache = PrefixCache::with_budget(1 << 30);
        let cancel = CancelFlag::new();
        for (turn, expected_path) in ["miss", "hit"].into_iter().enumerate() {
            let run = decode_gemma4(
                &model,
                Gemma4EnhancePrefill::Text(prompt.clone()),
                &cfg,
                vocab,
                &cancel,
                &mut cache,
            )
            .expect("engine decode");
            assert_eq!(
                run.output.tokens, reference,
                "turn {turn}: the engine must emit the plain loop's tokens"
            );
            let report = &run.report;
            assert_eq!(report.proposer, mlx_llm::core_llm::ProposerKind::None);
            assert_eq!(report.sampler, "host:constraint");
            assert_eq!(report.prefix_cache.path, expected_path, "turn {turn}");
            assert_eq!(
                report.prefix_hit_tokens > 0,
                expected_path == "hit",
                "turn {turn}: {report:?}"
            );
        }
    }

    /// The I2V enhancer prefill (spliced embeddings, caller-prefilled cache) on the engine matches
    /// the pre-engine `generate_from_prefill` loop token for token, and bypasses the prefix cache.
    #[test]
    fn gemma4_multimodal_enhancer_engine_decode_matches_the_plain_loop() {
        let (model, vocab) = tiny_gemma4();
        let prompt = PARITY_PROMPT.to_vec();
        let cfg = parity_cfg();
        let ids = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
        let embeds = model.embed(&ids).expect("embeds");

        let mut ref_cache = model.new_cache();
        let logits = model
            .decode_logits_from_embeds(&embeds, &mut ref_cache, 0)
            .expect("reference prefill");
        let reference = mlx_llm::decode::generate_from_prefill(
            &model,
            &mut ref_cache,
            logits,
            prompt.clone(),
            &gemma4_generation(&cfg),
            &mlx_llm::CancelFlag::new(),
            &mut |_| {},
            Some(&mut no_repeat(&prompt, vocab)),
            None,
        )
        .expect("plain loop")
        .tokens;
        assert!(reference.len() >= 16, "non-trivial run: {reference:?}");

        let mut cache = PrefixCache::with_budget(1 << 30);
        let run = decode_gemma4(
            &model,
            Gemma4EnhancePrefill::Multimodal {
                input_ids: prompt,
                embeds,
            },
            &cfg,
            vocab,
            &CancelFlag::new(),
            &mut cache,
        )
        .expect("engine decode");
        assert_eq!(run.output.tokens, reference);
        assert_eq!(run.report.prefix_cache.path, "bypassed");
        assert!(
            cache.is_empty(),
            "a multimodal prompt must not feed the cache"
        );
    }

    /// A seeded stochastic run under the same constraint also matches: the engine's `MlxSampler`
    /// and the plain loop draw through the same `sample_with_path` host draw from the same seeded
    /// stream, one uniform per row, in the same order (the enhancer itself is greedy; this pins
    /// that the constraint seam does not perturb the draw order either).
    #[test]
    fn gemma4_constrained_stochastic_engine_run_matches_the_plain_loop() {
        let (model, vocab) = tiny_gemma4();
        let prompt = PARITY_PROMPT.to_vec();
        let generation = GenerationConfig {
            sampling: SamplingParams {
                temperature: 0.9,
                top_p: 0.95,
                ..Default::default()
            },
            seed: Some(7),
            ..gemma4_generation(&parity_cfg())
        };
        let reference = mlx_llm::decode::generate_with(
            &model,
            &prompt,
            &generation,
            &mlx_llm::CancelFlag::new(),
            &mut |_| {},
            Some(&mut no_repeat(&prompt, vocab)),
            None,
        )
        .expect("plain loop")
        .tokens;
        let mut constraint = no_repeat(&prompt, vocab);
        let run = generate_speculative(
            &model,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&prompt),
            &generation,
            0,
            &mlx_llm::CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                constraint: Some(&mut constraint),
                ..EngineOptions::default()
            },
        )
        .expect("engine");
        assert!(reference.len() >= 8, "non-trivial run: {reference:?}");
        assert_eq!(run.output.tokens, reference);
        assert_eq!(run.report.sampler, "host:constraint");
    }

    // `SampleParams` presets + `SplitMix64` determinism are covered in the shared
    // `mlx_gen::text_sample` tests (the sampler now lives there — sc-9561 / F-105).
}
